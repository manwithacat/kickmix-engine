"""invsqrt task adapter: reproduces the official `verify_submission_invsqrt.verify()` with the engine as simulator.

Written by an AI model (Claude) under the owner's direction. The official verifier module is imported and its own
functions are used for the spec, seeds (`_initial_seed` and the SeedSequence split into 64-batch chunks), shot
inputs (`sample_shots`), the reference table (`reference_values`) and the cost model (`rotation_t_count`). The
engine replaces `km.Simulator`: one engine chunk = one kickmix `Simulator(batch_size=256, seed)` running the chunk's
batches in order (proven per shot in M1 gate 1). Phases arrive as the verifier's 2^-64-turn integers
(`half_turns_to_units(read_shot_phase(i))`, also proven in M1). Differences from the official path:
- the circuit is compiled from the .kmb file by the engine (no kickmix Circuit);
- the seed hash is the SHA-256 of the file bytes (the verifier hashes kickmix's re-serialisation; equal for
  canonical .kmb files, which `--cross-check` confirms per run);
- `reaction_depth` is not computed (None);
- only `batch_size=256` is supported;
- on FAIL, the error reported is the first failure in task order, with the official message text; for a dirty
  scratch qubit the failing chunk is re-run with a state dump to name the qubit as the official message does.
"""
from __future__ import annotations

import concurrent.futures
import importlib.util
import math
import os
import sys
import time

import numpy as np

from . import KINDS, run, seed_words
from .cache import ProgramCache, load_program


def default_verifier():
    p = os.environ.get('KICKMIX_ENGINE_INVSQRT_VERIFIER')
    if not p and os.environ.get('KICKMIX_ENGINE_VERIFIERS'):
        p = os.path.join(os.environ['KICKMIX_ENGINE_VERIFIERS'], 'invsqrt', 'verify_submission_invsqrt.py')
    return p


def load_verifier(path=None):
    path = path or default_verifier()
    if not path or not os.path.exists(path):
        raise FileNotFoundError('official invsqrt verifier not found: pass its path or set '
                                'KICKMIX_ENGINE_INVSQRT_VERIFIER (or KICKMIX_ENGINE_VERIFIERS to the verifiers directory)')
    name = 'verify_submission_invsqrt_official'
    if name in sys.modules and getattr(sys.modules[name], '__file__', None) == path:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def validate_registers_kmb(registers, num_qubits, spec):
    """The verifier's `validate_registers` rules on the .kmb register table; returns (x qubit ids, scratch ids)."""
    names = [n for n, _ in registers]
    if len(set(names)) != len(names) or names != ['x']:
        raise ValueError(f'Invalid register schema: {names}')
    x = registers[0][1]
    for isq, i in x:
        if not isq:
            raise ValueError(f"Register 'x' must contain only qubits, but contains km.b({i}).")
    if len(x) != spec.width:
        raise ValueError(f'Expected {spec.width} x qubits, got {len(x)}.')
    ids = [i for _, i in x]
    if len(set(ids)) != len(ids):
        raise ValueError('The x register must consist of distinct qubits.')
    s = set(ids)
    return ids, [k for k in range(num_qubits) if k not in s]


class InvSqrtTask:
    def __init__(self, kmb_path, *, spec=None, verifier_path=None, cache: ProgramCache | None = None):
        self.v = v = load_verifier(verifier_path)
        self.spec = spec or v.ProblemSpecInvSqrt.for_n_p(v.N_P)
        self.path = str(kmb_path)
        self.cache = cache if cache is not None else ProgramCache(enabled=False)
        self.prog = None
        self.load_info = None

    def verify(self, *, salt='', seed=None, threads=1, batch_size=256, group_chunks=None):
        v, spec = self.v, self.spec
        if spec.max_x <= 0 or spec.shots_per_x <= 0:
            raise ValueError(f'max_x and shots_per_x must be positive, got {spec.max_x} and {spec.shots_per_x}.')
        if batch_size != 256:
            raise ValueError('the engine path supports batch_size=256 only')
        num_shots = spec.num_shots
        if num_shots % batch_size:
            raise ValueError(f'num_shots must be a multiple of batch_size={batch_size}, got {num_shots}.')
        t0 = time.perf_counter()
        ex = concurrent.futures.ThreadPoolExecutor(2)
        try:
            fut = ex.submit(load_program, self.path, cache=self.cache) if self.prog is None else None
            # Reference values for every x (official function; Python, ~1-2 s) while the program loads/compiles.
            target, tol = v.reference_values(spec, np.arange(1, spec.max_x + 1, dtype=np.uint64))
            if fut is not None:
                self.prog, self.circuit_hash, self.load_info = fut.result()
                if self.load_info.get('save_needed'):
                    ex.submit(self.cache.save, self.circuit_hash, self.prog)
            prog = self.prog
            x_ids, scratch = validate_registers_kmb(prog.registers, prog.num_qubits, spec)
            if spec.beta and tol[-1] < 1 << 16:
                raise ValueError('The tolerance is too small for 64-bit phase arithmetic.')
            base_seed = v._initial_seed(self.circuit_hash, salt) if seed is None else int(seed)
            chunk_shots = v.BATCHES_PER_CHUNK * batch_size
            starts = range(0, num_shots, chunk_shots)
            seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(
                4 * len(starts), dtype=np.uint64).reshape(-1, 4)
            tasks = [(s, min(s + chunk_shots, num_shots), int.from_bytes(seeds[i], 'little'))
                     for i, s in enumerate(starts)]
            self.t_prepare = time.perf_counter() - t0
            # Groups of whole chunks with the same batch count (the last chunk may be short).
            gsize = group_chunks or max(16, 16 * threads)
            groups, cur = [], []
            for t in tasks:
                nb = (t[1] - t[0]) // batch_size
                if cur and (len(cur) >= gsize or (cur[0][1] - cur[0][0]) // batch_size != nb):
                    groups.append(cur)
                    cur = []
                cur.append(t)
            if cur:
                groups.append(cur)

            def sim(group):
                nb = (group[0][1] - group[0][0]) // batch_size
                xs = np.stack([v.sample_shots(spec, t[0], t[1]).reshape(nb, batch_size) for t in group])
                out = run(prog, [seed_words(t[2]) for t in group], xs, x_ids, s=1, threads=threads, counting=True)
                return xs, out

            min_margin = 1 << 64
            worst = (-1.0, 0)
            tof = tcount = nrot = 0
            lv = np.arange(129)
            pending = [ex.submit(sim, groups[0])]
            for gi, group in enumerate(groups):
                xs, out = pending.pop(0).result()
                if gi + 1 < len(groups):
                    pending.append(ex.submit(sim, groups[gi + 1]))
                fail, allowed, used = self._check_group(group, xs, out, target, tol, scratch, x_ids)
                if fail is not None:
                    raise AssertionError(fail)
                xf, allowed, used = xs.reshape(-1), allowed.reshape(-1), used.reshape(-1)
                ratio = np.divide(used, allowed, out=np.zeros(len(xf)), where=allowed > 0)
                min_margin = min(min_margin, int((allowed - used).min()))
                rmax = float(ratio.max())
                worst = max(worst, (rmax, -int(xf[ratio == rmax].min())))
                tof += int(out['toffoli_count'].sum())
                zl = out['z_levels']
                tcount += int(zl[:, 3].sum())
                nrot += int(zl[:, 4:].sum())
            self.t_run = time.perf_counter() - t0 - self.t_prepare
        finally:
            ex.shutdown(wait=True)
        toffoli_count = tof / num_shots
        t_base = tcount / num_shots
        n_rot = nrot / num_shots
        eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
        if n_rot and not eps > 0:
            raise AssertionError('No tolerance left for rotation synthesis error.')
        t_count = t_base + (n_rot * v.rotation_t_count(eps) if n_rot else 0)
        toffolis = toffoli_count + t_count / v.T_PER_TOFFOLI
        sc = prog.static_counts
        self.wall = time.perf_counter() - t0
        return {
            'status': 'PASS',
            'score': prog.num_qubits * toffolis,
            'toffoli_equivalent': toffolis,
            'toffoli_count': toffoli_count,
            'max_toffoli_count': int(sum(sc[KINDS.index(k)] for k in ('CCX', 'CCZ', 'CSWAP'))),
            't_count': t_base,
            'rotation_count': n_rot,
            'rotation_eps': eps,
            'num_qubits': prog.num_qubits,
            'reaction_depth': None,
            'num_ops': prog.num_ops,
            'verified_shots': num_shots,
            'worst_error_over_tolerance': worst[0],
            'worst_x': -worst[1],
            'circuit_sha256': self.circuit_hash,
        }

    def _check_group(self, group, xs, out, target, tol, scratch, x_ids):
        """The official per-batch checks in task order; returns (first failure message or None, allowed, used)."""
        bad_x = (out['x_out'] != xs).any(axis=2)          # [chunk, batch]
        dirty = out['scratch_dirty'].astype(bool).any(axis=2)
        theta = out['phase_units']
        allowed = tol[xs - np.uint64(1)]
        error = (theta - target[xs - np.uint64(1)]).view(np.int64)
        used = np.abs(error).view(np.uint64)
        bad_ph = (used > allowed).any(axis=2)
        anybad = bad_x | dirty | bad_ph
        if not anybad.any():
            return None, allowed, used
        c, b = np.argwhere(anybad)[0]
        if bad_x[c, b]:
            return 'x was not restored.', allowed, used
        if dirty[c, b] and len(scratch):
            # Name the first dirty scratch qubit like the official message: re-run this chunk with a state dump.
            t = group[c]
            o = run(self.prog, [seed_words(t[2])], xs[c:c + 1, :b + 1], x_ids, s=1, threads=1, counting=False,
                    dump=True)
            q = o['dump_q'][b]  # [nq, 4] words of this batch
            dq = np.array([q[i].any() for i in scratch])
            return f'Scratch qubit q{scratch[int(np.argmax(dq))]} did not return to |0>.', allowed, used
        i = int(np.flatnonzero(used[c, b] > allowed[c, b])[0])
        return (f'x={xs[c, b, i]}: phase error {error[c, b, i] / 2**64:.3e} turns '
                f'exceeds tolerance {allowed[c, b, i] / 2**64:.3e}.'), allowed, used
