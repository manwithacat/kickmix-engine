"""dfthcsos (chemistry, FeMoCo76o113e) task adapter: reproduces the official `dfthcsos.verify.verify()` on the engine.

Written by an AI model (Claude) under the owner's direction. The official `dfthcsos` package is imported from the
competition's `verifiers/` directory and its own functions do everything except two things:

- **Simulation.** `recover_words` is `dfthcsos.scoreinfra.circuits.recover_words_from_circuits` with the kickmix
  `Simulator` replaced by `kickmix_engine.run_sequence`: one 256-shot engine sim per kickmix sim, alive across all
  2K circuit applications (forward k = 0..K-1, then reverse), so the measurement RNG streams and the operation counters
  continue exactly as kickmix's do. Between applications the verifier clears the simulator and writes back the
  selection register and the word batch it read; the engine writes back each shot's own selection and word qubits.
  The two are the same whenever the previous application passed its checks (selection unchanged; every repeat of an
  address loaded the same words; for a reverse step, the words equal the batch written back), and the checks run in
  the official order and raise at the first failure, so no step after a failure is ever consulted. The checks, their
  order and their messages are the official ones.
- **The FCIDUMP.** `load_reference` parses a 128 MB text file in pure Python (about 5 s). The adapter still runs the
  official `check_fcidump` (size and SHA-256) on every call, then reads the parser's packed float32 integrals from a
  cache keyed by that SHA-256 and by the SHA-256 of the official `reference.py`, and expands them with the official
  `_h2_full_from_lower` and the official `h1_thc` expression. The cache entry is written only after a round trip
  reproduced the parsed `tei` and `h1_thc` bit for bit, and it carries a SHA-256 of its arrays that is checked on
  every load. Cold runs use the official parser.

Everything else (`load_tree_edges`, `load_factorization`, `load_fixed_shifts`, `native_inputs`, `load_circuits`,
`submission_sha256`, `register_layout`, `evaluate_words`, `resource_costs`, `rejection_reason`,
`score_from_metrics`, the summary fields) is the official code, called in the official order. kickmix must be
importable: the official modules import it and the circuits are loaded with it.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import io
import os
import sys
import time
from operator import index
from pathlib import Path

import numpy as np

from . import compile_circuit, run_sequence, seed_words
from .cache import ProgramCache, _atomic_write_bytes, cache_dir, load_program

FCIDUMP_CACHE_FORMAT = 1


def default_verifier():
    """Directory of the official `dfthcsos` package: $KICKMIX_ENGINE_DFTHCSOS_VERIFIER, else
    $KICKMIX_ENGINE_VERIFIERS/dfthcsos."""
    p = os.environ.get('KICKMIX_ENGINE_DFTHCSOS_VERIFIER')
    if not p and os.environ.get('KICKMIX_ENGINE_VERIFIERS'):
        p = os.path.join(os.environ['KICKMIX_ENGINE_VERIFIERS'], 'dfthcsos')
    return p


def _package_dir(path):
    p = Path(path).resolve()
    if p.is_file() and p.name == 'verify.py':
        p = p.parent
    return p


class Official:
    """The official modules: `V` = dfthcsos.verify, `C` = dfthcsos.scoreinfra.circuits, `REF` = ...reference,
    `M` = ...dfthc_math."""

    def __init__(self, V, C, REF, M, package_dir):
        self.V, self.C, self.REF, self.M, self.dir = V, C, REF, M, package_dir


def load_verifier(path: str | None = None) -> Official:
    """Import the official `dfthcsos` package from `path` (the package directory or its `verify.py`)."""
    path = path or default_verifier()
    if not path or not os.path.exists(path):
        raise FileNotFoundError('official dfthcsos verifier not found: pass the dfthcsos package directory or set '
                                'KICKMIX_ENGINE_DFTHCSOS_VERIFIER (or KICKMIX_ENGINE_VERIFIERS to the verifiers '
                                'directory)')
    pkg = _package_dir(path)
    if not (pkg / 'verify.py').is_file() or not (pkg / 'scoreinfra' / 'circuits.py').is_file():
        raise FileNotFoundError(f'{pkg} is not the dfthcsos package (no verify.py / scoreinfra/circuits.py)')
    loaded = sys.modules.get('dfthcsos')
    if loaded is not None:
        here = Path(getattr(loaded, '__file__', '') or '').resolve().parent
        if here != pkg:
            raise ImportError(f'a different dfthcsos package is already imported ({here}); use a new process')
    parent = str(pkg.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    V = importlib.import_module('dfthcsos.verify')
    C = importlib.import_module('dfthcsos.scoreinfra.circuits')
    REF = importlib.import_module('dfthcsos.scoreinfra.reference')
    M = importlib.import_module('dfthcsos.scoreinfra.dfthc_math')
    return Official(V, C, REF, M, pkg)


# ---------------------------------------------------------------------------------------------------------------
# FCIDUMP cache


def _arrays_digest(h2_lower, oei, ecore, norb, nelec) -> str:
    h = hashlib.sha256()
    for a in (h2_lower, oei):
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(np.ascontiguousarray(a).tobytes())
    h.update(repr((float(ecore).hex(), int(norb), int(nelec))).encode())
    return h.hexdigest()


def _file_sha256(path) -> str:
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def fcidump_cache_path(off: Official, directory=None) -> str:
    ref_src = _file_sha256(off.REF.__file__)[:16]
    name = f'{off.REF.FCIDUMP_SHA256}.ref-{ref_src}.f{FCIDUMP_CACHE_FORMAT}.npz'
    return os.path.join(cache_dir(directory), 'fcidump', name)


def _reference_from_lower(off: Official, h2_lower, oei, ecore, norb, nelec) -> dict:
    """The official `load_reference` tail, on the parser's arrays."""
    tei = off.REF._h2_full_from_lower(h2_lower, norb)
    h1_thc = np.array(oei - 0.5 * np.einsum("pqqs->ps", tei))
    return {"orbitals": norb, "electrons": nelec, "ecore": ecore, "oei": oei, "tei": tei, "h1_thc": h1_thc}


def load_reference_cached(off: Official, fcidump_path, *, directory=None, enabled=True, info=None) -> dict:
    """`load_reference(fcidump_path)` through the SHA-keyed cache. The caller has run `check_fcidump`."""
    info = info if info is not None else {}
    cp = fcidump_cache_path(off, directory)
    if enabled and os.path.exists(cp):
        t0 = time.perf_counter()
        try:
            with np.load(cp, allow_pickle=False) as d:
                h2_lower, oei = d['h2_lower'], d['oei']
                ecore, norb, nelec = float(d['ecore']), int(d['norb']), int(d['nelec'])
                digest = str(d['digest'])
            npairs = norb * (norb + 1) // 2
            ok = (h2_lower.dtype == np.float32 and h2_lower.shape == (npairs, npairs) and oei.dtype == np.float64
                  and oei.shape == (norb, norb) and digest == _arrays_digest(h2_lower, oei, ecore, norb, nelec))
        except Exception as ex:  # noqa: BLE001 - any unreadable entry (bad zip, CRC, missing key) is a miss
            ok, info['fcidump_cache_error'] = False, f'{type(ex).__name__}: {ex}'
        if ok:
            ref = _reference_from_lower(off, h2_lower, oei, ecore, norb, nelec)
            info['fcidump'] = 'cache'
            info['fcidump_s'] = time.perf_counter() - t0
            return ref
        info['fcidump_cache_rejected'] = cp
    t0 = time.perf_counter()
    ref = off.REF.load_reference(fcidump_path)
    info['fcidump'] = 'parsed'
    info['fcidump_s'] = time.perf_counter() - t0
    if enabled:
        t1 = time.perf_counter()
        n = int(ref['orbitals'])
        rows, cols = np.tril_indices(n)
        tei = ref['tei']
        h2_lower = tei[rows[:, None], cols[:, None], rows[None, :], cols[None, :]].astype(np.float32)
        back = _reference_from_lower(off, h2_lower, ref['oei'], ref['ecore'], n, ref['electrons'])
        same = all(np.array_equal(back[k], ref[k]) for k in ('tei', 'h1_thc', 'oei')) and all(
            back[k] == ref[k] and type(back[k]) is type(ref[k]) for k in ('orbitals', 'electrons', 'ecore'))
        del back
        if same:
            os.makedirs(os.path.dirname(cp), exist_ok=True)
            buf = io.BytesIO()
            np.savez(buf, h2_lower=h2_lower, oei=ref['oei'], ecore=np.float64(ref['ecore']), norb=n,
                     nelec=int(ref['electrons']),
                     digest=_arrays_digest(h2_lower, ref['oei'], ref['ecore'], n, ref['electrons']))
            _atomic_write_bytes(cp, buf.getvalue())
            info['fcidump_cache_written'] = cp
        else:
            info['fcidump_cache_written'] = None
            info['fcidump_cache_error'] = 'round trip differs; not cached'
        info['fcidump_cache_write_s'] = time.perf_counter() - t1
    return ref


# ---------------------------------------------------------------------------------------------------------------
# Simulation on the engine


def _circuit_programs(off: Official, circuits_arg, family, fmt, cache: ProgramCache, info):
    """Engine programs for the family: from the .kmb files through the program cache when `circuits_arg` is a
    directory of .kmb files (the files kickmix loaded), else compiled from kickmix's text form of each circuit."""
    t0 = time.perf_counter()
    C = off.C
    progs = None
    if isinstance(circuits_arg, (str, Path)) and Path(circuits_arg).is_dir():
        d = Path(circuits_arg)
        use_kmb = fmt == 'kmb' or (fmt == 'auto' and C.kmb_path(d, 0).is_file())
        if use_kmb:
            progs, sources = [], []
            for k in range(len(family)):
                prog, sha, li = load_program(str(C.kmb_path(d, k)), cache=cache)
                if li.get('save_needed'):
                    cache.save(sha, prog)
                progs.append(prog)
                sources.append(li['source'])
            info['programs'] = sources
    if progs is None:
        progs = [compile_circuit(c) for c in family]
        info['programs'] = 'compiled from text'
    for k, (p, c) in enumerate(zip(progs, family)):
        if p.num_ops != len(c) or p.num_qubits != int(c.num_qubits):
            raise RuntimeError(f'engine program {k} does not match the kickmix circuit (ops {p.num_ops} vs {len(c)}, '
                               f'qubits {p.num_qubits} vs {c.num_qubits})')
    info['programs_s'] = time.perf_counter() - t0
    return progs


def recover_words(off: Official, circuits, programs, *, num_vectors, num_angles, shots_per_address, seed, threads=1,
                  info=None):
    """`recover_words_from_circuits` with the engine as simulator (same checks, order and messages).

    `programs` is a callable returning the engine programs of `circuits` (called after the official pre-checks,
    where kickmix would build its simulator)."""
    info = info if info is not None else {}
    C = off.C
    InvalidCircuits = C.InvalidCircuits
    circuits = tuple(circuits)
    C._reject_rotations(circuits)
    layout = C.register_layout(circuits)
    brot, batch_size = layout.brot, layout.batch_size
    expected = -(-num_angles // batch_size)
    if len(circuits) != expected:
        raise InvalidCircuits(
            f"{num_angles} angles in batches of J={batch_size} need {expected} "
            f"circuits, got {len(circuits)}"
        )
    selection_bits = (index(num_vectors) - 1).bit_length()
    if len(layout.selection) != selection_bits:
        raise InvalidCircuits(
            f"the selection register must have exactly {selection_bits} qubits "
            f"for {num_vectors} vectors, got {len(layout.selection)}"
        )
    try:
        repeats = index(shots_per_address)
    except TypeError as exc:
        raise ValueError("shots_per_address must be a positive integer") from exc
    if repeats < 1:
        raise ValueError("shots_per_address must be a positive integer")
    try:
        base_seed = index(seed)
    except TypeError as exc:
        raise ValueError("seed must be a non-negative integer") from exc
    if base_seed < 0:
        raise ValueError(f"seed must be non-negative, got {base_seed}")
    sim_seed = int.from_bytes(
        np.random.SeedSequence(base_seed, pool_size=8).generate_state(4, dtype=np.uint64),
        "little",
    )
    if brot > 62:
        raise NotImplementedError(f'brot = {brot} > 62 is not supported by the engine adapter')

    progs = programs()
    t0 = time.perf_counter()
    K = len(circuits)
    sel_ids = [q.id for q in layout.selection]
    word_ids = [[q.id for q in w] for w in layout.words]
    xreg = sel_ids + [i for w in word_ids for i in w]
    nsel = len(sel_ids)
    xw = max(1, -(-len(xreg) // 64))
    shots = num_vectors * repeats
    addresses = np.tile(np.arange(num_vectors, dtype=np.int64), repeats)
    x0 = np.zeros((shots, xw), np.uint64)
    x0[:, 0] = addresses.astype(np.uint64)  # selection bits 0..nsel-1 (nsel <= 63 for any feasible shot count)
    order = list(range(K)) + list(reversed(range(K)))
    info['prepare_s'] = time.perf_counter() - t0
    t1 = time.perf_counter()
    out = run_sequence([progs[k] for k in order], seed_words(sim_seed), x0, xreg, threads=threads, counting=True)
    info['simulate_s'] = time.perf_counter() - t1
    del x0
    t2 = time.perf_counter()
    x_out, dirty, phase = out['x_out'], out['dirty'], out['phase']
    sel_mask = np.uint64((1 << nsel) - 1)
    wmask = np.uint64((1 << brot) - 1)
    addr_u = addresses.astype(np.uint64)

    def field(x, start, width, mask):
        lo, off_ = divmod(start, 64)
        v = x[:, lo] >> np.uint64(off_)
        if off_ + width > 64:
            v = v | (x[:, lo + 1] << np.uint64(64 - off_))
        return v & mask

    def check_state(i, label):
        if np.any(phase[i]):
            raise InvalidCircuits(f"{label}: uncorrected phase (measurement kickback)")
        if not np.array_equal(field(x_out[i], 0, nsel, sel_mask), addr_u):
            raise InvalidCircuits(f"{label}: the selection register changed")
        if np.any(dirty[i]):
            raise InvalidCircuits(f"{label}: workspace qubits are not returned to |0>")

    def read_words(i, label):
        table = np.stack([field(x_out[i], nsel + brot * j, brot, wmask) for j in range(batch_size)], axis=1)
        table = table.astype(np.int64).reshape(repeats, num_vectors, batch_size)
        if np.any(table != table[0]):
            raise InvalidCircuits(
                f"{label}: repeated shots of the same address loaded different words"
            )
        return table[0]

    batches = []
    for k in range(K):
        check_state(k, f"ANGLE_{k}")
        batches.append(read_words(k, f"ANGLE_{k}"))
    words = np.concatenate(batches, axis=1)
    if np.any(words[:, num_angles:]):
        raise InvalidCircuits("padding words beyond N - 1 are not zero")
    for i, k in enumerate(reversed(range(K)), start=K):
        check_state(i, f"reverse ANGLE_{k}")
        actual = read_words(i, f"reverse ANGLE_{k}")
        if k > 0:
            if not np.array_equal(actual, batches[k - 1]):
                raise InvalidCircuits(
                    f"reverse ANGLE_{k} did not restore the ANGLE_{k - 1} word registers"
                )
        elif np.any(actual):
            raise InvalidCircuits("the reverse traversal did not clear the word registers")
    toffoli = int(out['toffoli_count'].sum()) / (num_vectors * repeats)
    info['checks_s'] = time.perf_counter() - t2
    return words[:, :num_angles], layout, toffoli


# ---------------------------------------------------------------------------------------------------------------
# Memory: the error_2 Gram tensor in C order (values moved, not recomputed)


def transpose_square_inplace(D, block=256):
    """Transpose a C-contiguous square matrix in place with block swaps (temporaries of at most 2 blocks)."""
    n = D.shape[0]
    for i in range(0, n, block):
        ie = min(i + block, n)
        D[i:ie, i:ie] = D[i:ie, i:ie].T.copy()
        for j in range(ie, n, block):
            je = min(j + block, n)
            tmp = D[i:ie, j:je].copy()
            D[i:ie, j:je] = D[j:je, i:ie].T
            D[j:je, i:ie] = tmp.T


def c_order_inplace(G):
    """`G` with the same values in a C-contiguous array, without a second N^4 buffer when `G` is the transposed view
    (p, q | s, t) -> (s, t | p, q) of a C-contiguous square N^2 x N^2 buffer, which is what the official
    `reconstruct_h2` einsum returns. Otherwise `G` itself (unchanged, as the official code would use it).
    Returns (array, how)."""
    if G.flags['C_CONTIGUOUS'] or G.ndim != 4:
        return G, 'already C order'
    n0, n1, n2, n3 = G.shape
    H = G.transpose(2, 3, 0, 1)
    if (n0, n1) != (n2, n3) or not H.flags['C_CONTIGUOUS'] or not H.flags['WRITEABLE']:
        return G, 'unchanged (layout not recognised)'
    D = H.reshape(n2 * n3, n0 * n1)  # a view: H is C-contiguous
    if not np.shares_memory(D, G):
        return G, 'unchanged (reshape copied)'
    transpose_square_inplace(D)
    return D.reshape(n0, n1, n2, n3), 'transposed in place'


class _COrderGram:
    """While active, the official `dfthc_math.reconstruct_h2` (looked up by name in `evaluate_dfthc_vectors`) returns
    its own result moved into C order, so `np.linalg.norm(h2_delta.ravel())` no longer copies N^4 doubles."""

    def __init__(self, M, info):
        self.M, self.info = M, info

    def __enter__(self):
        self.orig = orig = self.M.reconstruct_h2
        info = self.info

        def reconstruct_h2(X_rho_c):
            t = time.perf_counter()
            G = orig(X_rho_c)
            G, how = c_order_inplace(G)
            info['gram_c_order'] = how
            info['gram_c_order_s'] = time.perf_counter() - t
            return G
        self.M.reconstruct_h2 = reconstruct_h2
        return self

    def __exit__(self, *exc):
        self.M.reconstruct_h2 = self.orig
        return False


# ---------------------------------------------------------------------------------------------------------------
# The official flow


def load_inputs(off: Official, factorization, *, fcidump_cache=True, cache_directory=None, info=None):
    """`dfthcsos.verify.load_inputs` with the reference integrals read through the FCIDUMP cache."""
    V = off.V
    params = V.load_factorization(factorization)
    if np.shape(params["U"])[-1] != V.ORBITALS:
        raise ValueError(
            f"factorization has {np.shape(params['U'])[-1]} orbitals, {V.SYSTEM} has {V.ORBITALS}"
        )
    params.update(V.load_fixed_shifts())
    t0 = time.perf_counter()
    path = V.check_fcidump()
    if info is not None:
        info['check_fcidump_s'] = time.perf_counter() - t0
    ref = load_reference_cached(off, path, directory=cache_directory, enabled=fcidump_cache, info=info)
    w, W, S, beta_1 = V.native_inputs(params)
    return V.SubmissionInputs(
        factorization_file=None if isinstance(factorization, Mapping) else Path(factorization),
        params=params,
        ref=ref,
        w=w,
        W=W,
        S=S,
        beta_1=beta_1,
        energy_upper=V.UPPER_ENERGY,
    )


def verify(circuits, factorization=None, tree_edges=None, *, format='auto', shots_per_address=None, seed=None,
           salt='', threads=None, verifier=None, cache: ProgramCache | None = None, fcidump_cache=True,
           low_memory=True, info=None):
    """`dfthcsos.verify.verify` with the engine as simulator. Returns the official summary dict or raises the
    official exception. `info` (a dict) receives stage timings and cache events. `low_memory` moves the error_2
    Gram tensor into C order in place (see `_COrderGram`), saving one N^4 copy at the peak; values are unchanged."""
    info = info if info is not None else {}
    t_all = time.perf_counter()
    off = load_verifier(verifier)
    V, C = off.V, off.C
    cache = cache if cache is not None else ProgramCache(enabled=False)
    threads = threads or os.cpu_count() or 1
    if shots_per_address is None:
        shots_per_address = C.SHOTS_PER_ADDRESS
    if factorization is None:
        if not (isinstance(circuits, (str, Path)) and Path(circuits).is_dir()):
            raise ValueError(f"factorization is required unless circuits is a directory with {V.SUBMISSION_FILE}")
        factorization = Path(circuits) / V.SUBMISSION_FILE
    edges = V.load_tree_edges(tree_edges, circuits)
    t0 = time.perf_counter()
    inputs = load_inputs(off, factorization, fcidump_cache=fcidump_cache,
                         cache_directory=cache.dir if cache.enabled else None, info=info)
    info['inputs_s'] = time.perf_counter() - t0
    t0 = time.perf_counter()
    family = V.load_circuits(circuits, format=format)
    info['load_circuits_s'] = time.perf_counter() - t0
    # evaluate_circuits
    family = V.load_circuits(family, format=format)
    t0 = time.perf_counter()
    digest = V.submission_sha256(family, inputs.factorization_file or inputs.params, edges)
    info['submission_sha256_s'] = time.perf_counter() - t0
    words, layout, toffoli = recover_words(
        off, family, lambda: _circuit_programs(off, circuits, family, format, cache, info),
        num_vectors=inputs.num_vectors,
        num_angles=inputs.num_orbitals - 1,
        shots_per_address=shots_per_address,
        seed=C.initial_seed(digest, salt) if seed is None else seed,
        threads=threads, info=info,
    )
    brot = layout.brot
    t0 = time.perf_counter()
    if low_memory:
        with _COrderGram(off.M, info):
            metrics = V.evaluate_words(inputs, words, edges, brot)
    else:
        metrics = V.evaluate_words(inputs, words, edges, brot)
    info['evaluate_words_s'] = time.perf_counter() - t0
    resources = V.resource_costs(inputs, family, brot, toffoli)
    result = {
        "angle_words": words,
        "brot": brot,
        "batch_size": layout.batch_size,
        "metrics": metrics,
        "resources": resources,
        "score": V.score_from_metrics(metrics, resources),
        "submission_sha256": digest,
        "verified_shots": inputs.num_vectors * shots_per_address,
    }
    reason = V.rejection_reason(result["metrics"], result["resources"])
    if reason is not None:
        info['wall_s'] = time.perf_counter() - t_all
        raise AssertionError(reason)
    t0 = time.perf_counter()
    summary = {
        "status": "PASS",
        "score": float(result["score"]),
        "lambda_eff": result["metrics"]["lambda_eff"],
        "toffoli_count": result["resources"]["toffoli"],
        "num_qubits": result["resources"]["qubits"],
        "brot": result["brot"],
        "batch_size": result["batch_size"],
        "num_circuits": len(family),
        "reaction_depth": sum(c.reaction_depth() for c in family),
        "num_ops": sum(len(c) for c in family),
        "verified_shots": result["verified_shots"],
        "submission_sha256": result["submission_sha256"],
        "resources": result["resources"],
        "metrics": result["metrics"],
    }
    info['summary_s'] = time.perf_counter() - t0
    info['wall_s'] = time.perf_counter() - t_all
    return summary
