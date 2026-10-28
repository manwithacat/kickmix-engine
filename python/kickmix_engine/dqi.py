"""DQI task adapter: reproduces the official `verify_submission_dqi.verify()` on the engine.

Written by an AI model (Claude) under the owner's direction. The official verifier module is imported and its
own functions are used for everything except simulation and (by default) input sampling: seeds (`_initial_seed`,
SeedSequence split), `optimal_dqi_weights`, `FieldTables`, `clopper_pearson_upper_bound` and `repetition_overhead`.
Input sampling: `sampler='engine'` (default) draws the inputs on the engine with an exact re-implementation of the
verifier's `sample_shots` over NumPy's PCG64 stream (start states taken from NumPy itself; proven identical on all
batches of the tested circuits by `tests/sampler_parity_dqi.py`), fused with the simulation; `sampler='official'`
calls the verifier's own `sample_shots` in Python (the reference path); `self_check=k` re-draws k batches (or all,
k=-1) with the official `sample_shots` and fails if any input differs. Differences from the official path, all
checked by the parity tests:
- the circuit is compiled from the .kmb file by the engine (no kickmix Circuit is loaded);
- the seed hash is the SHA-256 of the file bytes (the verifier hashes kickmix's re-serialisation; the tests
  check they are equal for the circuits used);
- registers are validated from the .kmb register table with the verifier's rules;
- inputs are packed to bit-sliced words with `pack_rows`, which equals `np.packbits(to_bits(...))` (tested);
- `reaction_depth` is not computed (reported as None).
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import importlib.util
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np

from . import KINDS, _engine, compile_kmb, run_batches, seed_words, words_to_bools
from .cache import ProgramCache, file_sha256, load_program

M64 = (1 << 64) - 1


def pcg_state_words(rng_seed) -> list:
    """Start state of `numpy.random.default_rng(rng_seed)` (PCG64) as [state_lo, state_hi, inc_lo, inc_hi, has_uint32,
    uinteger], taken from NumPy itself."""
    st = np.random.PCG64(rng_seed).state
    s, inc = st['state']['state'], st['state']['inc']
    return [s & M64, s >> 64, inc & M64, inc >> 64, int(st['has_uint32']), int(st['uinteger'])]


def pcg_state_dict(words) -> dict:
    w = [int(x) for x in words]
    return {'bit_generator': 'PCG64', 'state': {'state': w[0] | (w[1] << 64), 'inc': w[2] | (w[3] << 64)},
            'has_uint32': w[4], 'uinteger': w[5]}

def default_verifier():
    """Path of the official `verify_submission_dqi.py`: $KICKMIX_ENGINE_DQI_VERIFIER, else
    $KICKMIX_ENGINE_VERIFIERS/dqi/verify_submission_dqi.py. The verifier imports kickmix, so kickmix must be importable."""
    p = os.environ.get('KICKMIX_ENGINE_DQI_VERIFIER')
    if not p and os.environ.get('KICKMIX_ENGINE_VERIFIERS'):
        p = os.path.join(os.environ['KICKMIX_ENGINE_VERIFIERS'], 'dqi', 'verify_submission_dqi.py')
    return p


VERIFIER = None  # resolved at call time by default_verifier()
OP_GROUP_ORDER = ['NEG', 'BIT_INVERT', 'BIT_STORE0', 'BIT_STORE1', 'X', 'Z', 'R', 'HMR', 'CX', 'CZ', 'SWAP', 'CSWAP',
                  'CCX', 'CCZ', 'Z_POW', 'INFO_TAG', 'DEBUG_PRINT', 'POP_CONDITION', 'PUSH_CONDITION']


def load_verifier(path: str | None = None):
    path = path or default_verifier()
    if not path or not os.path.exists(path):
        raise FileNotFoundError('official DQI verifier not found: pass its path or set KICKMIX_ENGINE_DQI_VERIFIER '
                                '(or KICKMIX_ENGINE_VERIFIERS to the verifiers directory)')
    name = 'verify_submission_dqi_official'
    if name in sys.modules and getattr(sys.modules[name], '__file__', None) == path:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def pack_rows(values: np.ndarray, degree: int) -> np.ndarray:
    """(batch=256, regs) field elements -> uint64 [regs * degree, 4] bit-sliced rows, row = reg * degree + k.

    Equal to ``bools_to_words(to_bits(values, degree))`` (tested), without the large bool intermediate."""
    vt = np.ascontiguousarray(values.T)  # (regs, 256)
    out = np.empty((vt.shape[0], degree, 32), np.uint8)
    for k in range(degree):
        out[:, k, :] = np.packbits(((vt >> k) & 1).astype(np.uint8), axis=1, bitorder='little')
    return out.reshape(-1, 32).view(np.uint64)


def validate_registers_kmb(registers, spec):
    """The verifier's `validate_registers` rules on the .kmb register table. Returns (syndrome qubit ids,
    mbuc bit ids) in the verifier's order (register contents of S0..S{n-1}, then E0..E{m-1})."""
    regs = dict(registers)
    s_names = [f'S{i}' for i in range(spec.n)]
    e_names = [f'E{j}' for j in range(spec.m)]
    if len(regs) != len(registers) or set(regs) != set(s_names) | set(e_names):
        raise ValueError('Registers must be exactly S0..S{n-1} and E0..E{m-1}.')

    def collect(names, want_qubit, label):
        out = []
        for name in names:
            reg = regs[name]
            if len(reg) != spec.degree or not all(isq == want_qubit for isq, _ in reg):
                raise ValueError(f'Register {name!r} must be {spec.degree} {label}, got {len(reg)}.')
            out.extend(i for _, i in reg)
        if len(set(out)) != len(out):
            raise ValueError(f'{label} registers must not overlap.')
        return out

    return collect(s_names, True, 'qubits'), collect(e_names, False, 'classical bits')


# Sampler worker state (set in the parent before the fork).
_S = {}


def _sample(task):
    """One verifier batch: the verifier's own sampling, packed to engine words."""
    start, rng_seed = task
    v, spec, tables, wp = _S['v'], _S['spec'], _S['tables'], _S['wp']
    rng = np.random.default_rng(rng_seed)
    syndromes, mbuc, phases = v.sample_shots(spec, tables, rng, _S['batch_size'], wp)
    return pack_rows(syndromes, spec.degree), pack_rows(mbuc, spec.degree), phases


class DQITask:
    def __init__(self, kmb_path, *, m=None, n=None, degree=None, r=None, num_shots=None, verifier_path=None,
                 cache: ProgramCache | None = None):
        self.v = v = load_verifier(verifier_path)
        kw = {k: x for k, x in dict(m=m, n=n, degree=degree, r=r, num_shots=num_shots).items() if x is not None}
        self.spec = v.ProblemSpecDQI.benchmark_instance(**kw)
        self.path = str(kmb_path)
        self.prog = None
        self.cache = cache if cache is not None else ProgramCache(enabled=False)
        self.load_info = None

    def prepare(self, compile_threaded=True):
        """Load (program cache) or compile, and hash. Compile and cache loads release the GIL and overlap the hash."""
        t0 = time.perf_counter()
        self.prog, self.circuit_hash, self.load_info = load_program(self.path, cache=self.cache)
        if self.load_info.get('save_needed'):
            self.cache.save(self.circuit_hash, self.prog)
        self.t_prepare = time.perf_counter() - t0
        return self.prepare_checks()

    def prepare_checks(self):
        p, spec = self.prog, self.spec
        if p.static_counts[KINDS.index('Z_POW')]:
            raise ValueError('Z_POW gates are not allowed in Benchmark II.')
        self.syn_ids, self.e_ids = validate_registers_kmb(p.registers, spec)
        comp = np.asarray(p.bit_compact, dtype=np.int64)
        e_c = comp[np.asarray(self.e_ids, dtype=np.int64)]
        self.e_keep = np.flatnonzero(e_c >= 0)  # E bits no op touches cannot affect anything
        self.e_compact = e_c[self.e_keep]
        return self

    def sampler_args(self):
        """Fixed sampler parameters, computed with the verifier's own functions (as `sample_shots` would)."""
        v, spec = self.v, self.spec
        if getattr(self, '_sargs', None) is None:
            tables = v.FieldTables(v.km.GF2Field(spec.degree))
            wp = np.asarray(v.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2, dtype=np.float64)
            cdf = wp.cumsum()  # exactly what Generator.choice(p=wp) computes
            cdf /= cdf[-1]
            self._sargs = dict(m=spec.m, n=spec.n, degree=spec.degree, cdf=[float(x) for x in cdf],
                               log=[int(x) for x in tables.log], antilog=[int(x) for x in tables.antilog])
        return self._sargs

    def engine_sample(self, tasks, threads=1, values=False, e_rows=None):
        """The engine sampler for verifier tasks: dict of qin [n, n*deg, 4], bin [n, len(e_rows), 4], phase [n, 4],
        end_state [n, 6] and with values=True syndromes [n, 256, n_syn], mbuc [n, 256, m]."""
        a, spec = self.sampler_args(), self.spec
        e_rows = self.e_keep if e_rows is None else e_rows
        pcg = np.array([pcg_state_words(t[1]) for t in tasks], dtype=np.uint64).reshape(-1, 6)
        r = _engine.sample_dqi(a['m'], a['n'], a['degree'], a['cdf'], a['log'], a['antilog'],
                               [int(x) for x in e_rows], pcg, threads, values)
        k = len(tasks)
        out = dict(qin=r['qin'].reshape(k, spec.n * spec.degree, 4), bin=r['bin'].reshape(k, len(e_rows), 4),
                   phase=r['phase'].reshape(k, 4), end_state=r['end_state'].reshape(k, 6))
        if values:
            out['syndromes'] = r['syndromes'].reshape(k, 256, spec.n).astype(np.int64)
            out['mbuc'] = r['mbuc'].reshape(k, 256, spec.m).astype(np.int64)
        return out

    def self_check(self, tasks, which, threads=1):
        """Compare the engine sampler with the official `sample_shots` on tasks[i] for i in `which`: syndromes,
        mbuc, phases, packed words and the generator state after the batch. Returns a list of mismatching indices."""
        v, spec = self.v, self.spec
        sub = [tasks[i] for i in which]
        eng = self.engine_sample(sub, threads=threads, values=True)
        if not _S:
            self._set_sampler_state(256)
        bad = []
        for j, (i, t) in enumerate(zip(which, sub)):
            rng = np.random.default_rng(t[1])
            syn, mbuc, ph = v.sample_shots(spec, _S['tables'], rng, 256, _S['wp'])
            ok = (np.array_equal(syn, eng['syndromes'][j]) and np.array_equal(mbuc, eng['mbuc'][j])
                  and np.array_equal(ph, words_to_bools(eng['phase'][j]))
                  and np.array_equal(pack_rows(syn, spec.degree), eng['qin'][j])
                  and np.array_equal(pack_rows(mbuc, spec.degree)[self.e_keep], eng['bin'][j])
                  and rng.bit_generator.state == pcg_state_dict(eng['end_state'][j]))
            if not ok:
                bad.append(i)
        return bad

    def tasks(self, salt='', seed=None, batch_size=256):
        v = self.v
        base_seed = v._initial_seed(self.circuit_hash, salt) if seed is None else int(seed)
        starts = range(0, self.spec.num_shots, batch_size)
        seeds = np.random.SeedSequence(base_seed, pool_size=8).generate_state(
            8 * len(starts), dtype=np.uint64).reshape(-1, 2, 4)
        return [(s, int.from_bytes(seeds[i, 0], 'little'), int.from_bytes(seeds[i, 1], 'little'))
                for i, s in enumerate(starts)]

    def _set_sampler_state(self, batch_size):
        v, spec = self.v, self.spec
        _S.update(v=v, spec=spec, batch_size=batch_size,
                  tables=v.FieldTables(v.km.GF2Field(spec.degree)),
                  wp=v.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2)

    def sample(self, task, batch_size=256):
        """(qin words, E-bit words for the kept E bits, phases) for one verifier task."""
        if not _S:
            self._set_sampler_state(batch_size)
        q, b, ph = _sample((task[0], task[1]))
        return q, b[self.e_keep], ph

    def run_group(self, tasks, samples, threads, dump=False):
        """samples: list of `sample()` outputs."""
        qin = np.stack([s[0] for s in samples])
        bin_ = np.stack([s[1] for s in samples])
        return self.run_stacked(tasks, qin, bin_, threads, dump)

    def run_stacked(self, tasks, qin, bin_, threads, dump=False):
        sims = np.stack([seed_words(t[2]) for t in tasks])
        return run_batches(self.prog, sims, self.syn_ids, qin, self.e_compact, bin_, threads=threads, dump=dump)

    def verify(self, *, salt='', seed=None, threads=1, sampler='engine', self_check=0, samplers=None, group=None,
               batch_size=256, log2_delta=None):
        """Full verification; returns the official `verify()` summary (reaction_depth None).

        sampler='engine' (default): inputs drawn on the engine, fused with simulation, `threads` Rayon threads.
        self_check: number of batches (evenly spaced; -1 = all) re-drawn with the official `sample_shots` and compared
        before simulating; any difference raises RuntimeError. sampler='official': the verifier's `sample_shots` in
        Python (reference path; `samplers` processes, `group` batches per engine call, see `_verify_official`)."""
        if sampler == 'official':
            return self._verify_official(salt=salt, seed=seed, threads=threads, samplers=samplers, group=group,
                                         batch_size=batch_size, log2_delta=log2_delta)
        if sampler != 'engine':
            raise ValueError("sampler must be 'engine' or 'official'")
        v, spec = self.v, self.spec
        if batch_size != 256:
            raise ValueError('engine batches are 256 shots')
        log2_delta = v.DEFAULT_LOG2_DELTA if log2_delta is None else log2_delta
        t0 = time.perf_counter()
        saver = None
        ex = concurrent.futures.ThreadPoolExecutor(1)
        try:
            if self.prog is None:
                self.prog, self.circuit_hash, self.load_info = load_program(self.path, cache=self.cache, executor=ex)
                self.prepare_checks()
                if self.load_info.get('save_needed'):
                    saver = ex.submit(self.cache.save, self.circuit_hash, self.prog)  # overlaps the simulation
            self.t_prepare = time.perf_counter() - t0
            tasks = self.tasks(salt, seed, batch_size)
            self.self_check_result = None
            if self_check:
                n = len(tasks)
                which = list(range(n)) if self_check < 0 else sorted({round(i * (n - 1) / max(1, self_check - 1))
                                                                         for i in range(min(self_check, n))})
                ts = time.perf_counter()
                bad = self.self_check(tasks, which, threads=threads)
                self.self_check_result = dict(batches=len(which), mismatches=bad, seconds=time.perf_counter() - ts)
                if bad:
                    raise RuntimeError(f'sampler self-check failed on batches {bad[:10]} (engine sampler differs from '
                                       f'the official sample_shots); use sampler="official"')
            a = self.sampler_args()
            pcg = np.array([pcg_state_words(t[1]) for t in tasks], dtype=np.uint64).reshape(-1, 6)
            sims = np.stack([seed_words(t[2]) for t in tasks])
            qin_ids = [int(q) for q in self.syn_ids]
            tr = time.perf_counter()
            r = _engine.run_dqi(self.prog, a['m'], a['n'], a['degree'], a['cdf'], a['log'], a['antilog'],
                                [int(x) for x in self.e_keep], pcg, sims, qin_ids, [int(x) for x in self.e_compact],
                                threads, True)
            self.t_run = time.perf_counter() - tr
            k = len(tasks)
            val = words_to_bools(r['val_err'].reshape(k, 4))
            ph = words_to_bools(r['phase'].reshape(k, 4))
            expected = words_to_bools(r['expected'].reshape(k, 4))
            kc = r['kind_counts'].reshape(k, len(KINDS)).astype(np.int64)
            tof = kc[:, [KINDS.index(x) for x in ('CCX', 'CCZ', 'CSWAP')]].sum(axis=1)
            perr = (ph ^ expected) & ~val
            per_batch = [dict(index=i, val=int(val[i].sum()), phase=int(perr[i].sum()), toffoli=int(tof[i]),
                              counts=kc[i], val_shots=np.flatnonzero(val[i]).tolist(),
                              phase_shots=np.flatnonzero(perr[i]).tolist()) for i in range(k)]
            if saver is not None:
                saver.result()
        finally:
            ex.shutdown(wait=True)
        return self._summary(per_batch, log2_delta, t0)

    def _verify_official(self, *, salt='', seed=None, threads=1, samplers=None, group=None, batch_size=256,
                         log2_delta=None):
        """Reference path (M3): the verifier's own `sample_shots` in Python. `samplers` sampling processes (default 0
        for threads <= 2, else min(4, threads // 2); 0 = sample in this process while the engine runs), engine
        `threads`, batches submitted in groups of `group` (default max(4, threads)) with two groups in flight; at most
        `group` samples are prefetched.

        Order: fork the samplers (small parent), start loading/compiling in a thread, hash the file, start sampling,
        then join the program and run groups as samples arrive."""
        v, spec = self.v, self.spec
        if batch_size != 256:
            raise ValueError('engine batches are 256 shots')
        log2_delta = v.DEFAULT_LOG2_DELTA if log2_delta is None else log2_delta
        t0 = time.perf_counter()
        samplers = (0 if threads <= 2 else min(4, threads // 2)) if samplers is None else samplers
        group = group or max(4, threads)
        self._set_sampler_state(batch_size)
        pool = mp.get_context('fork').Pool(samplers) if samplers > 0 else None
        ex = concurrent.futures.ThreadPoolExecutor(2)
        try:
            fut_prog = None
            if self.prog is None:
                fut_prog = ex.submit(load_program, self.path, cache=self.cache)
                self.circuit_hash = file_sha256(self.path)  # needed now for the seeds (load_program hashes too)
            tasks = self.tasks(salt, seed, batch_size)
            todo = [(t[0], t[1]) for t in tasks]
            pending = []  # sampler results in task order (AsyncResult or a direct value)
            nxt = 0

            def refill():
                nonlocal nxt
                while pool is not None and nxt < len(todo) and len(pending) < 2 * group:
                    pending.append(pool.apply_async(_sample, (todo[nxt],)))
                    nxt += 1

            def take():
                nonlocal nxt
                refill()
                if pool is None:
                    r = _sample(todo[nxt])
                    nxt += 1
                    return r
                r = pending.pop(0).get()
                refill()
                return r

            refill()
            if fut_prog is not None:
                self.prog, h, self.load_info = fut_prog.result()
                assert h == self.circuit_hash
                self.prepare_checks()
                if self.load_info.get('save_needed'):
                    self.cache.save(self.circuit_hash, self.prog)
            self.t_prepare = time.perf_counter() - t0
            per_batch, inflight = [], []
            nq_in, nb_in = len(self.syn_ids), len(self.e_keep)
            for g0 in range(0, len(tasks), group):
                gt = tasks[g0:g0 + group]
                qin = np.empty((len(gt), nq_in, 4), np.uint64)
                bin_ = np.empty((len(gt), nb_in, 4), np.uint64)
                phases = np.empty((len(gt), 256), bool)
                for i in range(len(gt)):
                    q, b, ph = take()
                    qin[i], phases[i] = q, ph
                    np.take(b, self.e_keep, axis=0, out=bin_[i])
                    del q, b
                inflight.append((g0, phases, ex.submit(self.run_stacked, gt, qin, bin_, threads)))
                del qin, bin_
                while len(inflight) >= 2:
                    per_batch.extend(self._collect(*inflight.pop(0)))
            while inflight:
                per_batch.extend(self._collect(*inflight.pop(0)))
        finally:
            ex.shutdown(wait=True)
            if pool is not None:
                pool.terminate()
                pool.join()
        return self._summary(per_batch, log2_delta, t0)

    def _summary(self, per_batch, log2_delta, t0):
        v, spec = self.v, self.spec
        num_shots = spec.num_shots
        value_errors = sum(r['val'] for r in per_batch)
        phase_errors = sum(r['phase'] for r in per_batch)
        shot_counts = np.sum([r['counts'] for r in per_batch], axis=0)
        total_errors = value_errors + phase_errors
        eps_hat = total_errors / num_shots
        eps_plus = v.clopper_pearson_upper_bound(total_errors, num_shots, log2_delta=log2_delta)
        if eps_plus >= 1 / 16:
            raise AssertionError(
                f'Effective error rate {eps_plus:.4f} >= 1/16 at log2_delta={log2_delta} '
                f'(empirical={eps_hat:.4f}, value_errors={value_errors}/{num_shots}, '
                f'phase_errors={phase_errors}/{num_shots})')
        rep = v.repetition_overhead(eps_plus)
        toffolis = sum(r['toffoli'] for r in per_batch) / num_shots
        nq = self.prog.num_qubits
        sc = self.prog.static_counts
        weight_probs = v.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
        counts = {k: int(shot_counts[KINDS.index(k)]) for k in OP_GROUP_ORDER if k in KINDS and shot_counts[KINDS.index(k)]}
        out = {
            'status': 'PASS',
            'score': rep * (nq ** 2) * toffolis,
            'm': spec.m, 'n': spec.n, 'degree': spec.degree, 'r': spec.r,
            'verified_shots': num_shots,
            'value_errors': value_errors,
            'phase_errors': phase_errors,
            'empirical_error_rate': eps_hat,
            'effective_error_rate': eps_plus,
            'log2_delta': log2_delta,
            'repetition_factor': rep,
            'mean_error_weight': float(np.arange(len(weight_probs)) @ weight_probs),
            'toffoli_count': toffolis,
            'max_toffoli_count': int(sum(sc[KINDS.index(k)] for k in ('CCX', 'CCZ', 'CSWAP'))),
            'reaction_depth': None,
            'num_ops': self.prog.num_ops,
            'shot_operation_counts': counts,
            'num_qubits': nq,
            'num_bits': self.prog.orig_num_bits,
            'circuit_sha256': self.circuit_hash,
        }
        self.per_batch = per_batch
        self.wall = time.perf_counter() - t0
        return out

    @staticmethod
    def _collect(g0, phases, fut):
        r = fut.result()
        from . import words_to_bools
        val = words_to_bools(r['val_err'])
        ph = words_to_bools(r['phase'])
        perr = (ph ^ phases) & ~val
        return [dict(index=g0 + i, val=int(val[i].sum()), phase=int(perr[i].sum()),
                     toffoli=int(r['toffoli_count'][i]), counts=r['kind_counts'][i].astype(np.int64),
                     val_shots=np.flatnonzero(val[i]).tolist(), phase_shots=np.flatnonzero(perr[i]).tolist())
                for i in range(len(phases))]
