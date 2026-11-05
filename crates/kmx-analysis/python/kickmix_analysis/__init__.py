"""kickmix_analysis: streaming static analysis of kickmix .kmb circuits (Rust, crate kmx-analysis).

Separate from the verify wheel `kickmix_engine`. The Python reference is scripts/qhdl_lifetime_allocator.py and
scripts/qhdl_peak.py in the competition repository; this module reproduces their outputs byte for byte.

    liveness(path, *, model="allocator", cond_rule=False, register_gaps=False, profile=True) -> dict
        intervals = dict(logical, s, e, is_first, is_last)   # as build_intervals (int64 / bool), sorted by (wire, s)
        profile   = int32[num_ops]                           # as live_profile(..., register_gaps)
        peak, num_ops, num_qubits, num_refs, unconditional_resets, conditional_resets, reset_refs,
        registers (sorted int64), wire_first / wire_last (int64 per wire id, -1 untouched)
    live_profile(intervals, num_ops, registers, register_gaps=False) -> int32[num_ops]
    idle_gaps(path, run_s, run_e, *, register_gaps=False, cond_rule=False) -> (ga, gb, gw)
    recolour(path, logical, col, *, cond_rule=False) -> dict of the six uint32 qubit streams
"""
from __future__ import annotations

import numpy as np

from . import _kmx_analysis as _k

__version__ = _k.__version__
MODELS = ('allocator',)


def liveness(path, *, model='allocator', cond_rule=False, register_gaps=False, profile=True):
    if model not in MODELS:
        raise ValueError(f'model must be one of {MODELS}')
    return _k.liveness(str(path), cond_rule, register_gaps, profile)


def live_profile(intervals, num_ops, registers, register_gaps=False):
    regs = np.unique(np.asarray(sorted(int(x) for x in registers), dtype=np.int64))
    return _k.live_profile(np.ascontiguousarray(intervals['logical'], np.int64),
                           np.ascontiguousarray(intervals['s'], np.int64),
                           np.ascontiguousarray(intervals['e'], np.int64),
                           np.ascontiguousarray(intervals['is_first'], np.bool_),
                           np.ascontiguousarray(intervals['is_last'], np.bool_),
                           int(num_ops), regs, bool(register_gaps))


def idle_gaps(path, run_s, run_e, *, register_gaps=False, cond_rule=False):
    return _k.idle_gaps(str(path), np.ascontiguousarray(run_s, np.int64), np.ascontiguousarray(run_e, np.int64),
                        bool(register_gaps), bool(cond_rule))


def recolour(path, logical, col, *, cond_rule=False):
    return _k.recolour(str(path), np.ascontiguousarray(logical, np.int64), np.ascontiguousarray(col, np.int64),
                       bool(cond_rule))


# ------------------------------------------------------------------ GF(2) echelon objects (card E2, P2 primitive)
# Reference: tools/dfthc-refit/recover.py (Basis, rank, nullspace) and planner_shared.intersect / decompose in the
# competition repository. Same pivots (highest bit), same reduced vectors, same combination masks.
#
#   Gf2Basis(width)            add(v) -> bool, add_many(vs), reduce(v) -> (v, c), solve(v) -> c | None,
#                              solve_many(vs), contains(v), decompose(v) -> (residual, c), truncate(n), vectors, len
#   GroupSet(nb, width, groups) nb bases + a table of groups [(basis, v), ...]; build(sel), independent(g),
#                              add_group(g), greedy(order), first_pair(ids), positions(g), compute_status(ids),
#                              enabled(bx, px, by, py), compute_decomp(ids), quotient_cands(...), quotient_pair(c, cap),
#                              add_vec(b, v), size(b), sizes(), restore(sizes)
#   gf2_rank(rows), gf2_nullspace(rows, width), gf2_intersect(V1, V2, m)

Gf2Basis = _k.Gf2Basis
GroupSet = _k.GroupSet


def _bits(rows):
    return max((int(v).bit_length() for v in rows), default=0)


def gf2_rank(rows, width=None):
    """Rank of integer bit-rows; `width` is a hint (rows wider than it are measured)."""
    rows = list(rows)
    if width:
        try:
            return _k.gf2_rank(rows, int(width))
        except (OverflowError, TypeError):
            pass
    rows = [int(v) for v in rows]
    return _k.gf2_rank(rows, max(_bits(rows), 1))


def gf2_nullspace(rows, width):
    """recover.nullspace(rows, width)."""
    rows = list(rows)
    try:
        return _k.gf2_nullspace(rows, int(width), max(int(width), 1))
    except (OverflowError, TypeError):
        rows = [int(v) for v in rows]
        return _k.gf2_nullspace(rows, int(width), max(_bits(rows), 1))


def gf2_intersect(V1, V2, m):
    """planner_shared.intersect(V1, V2, m)."""
    try:
        return _k.gf2_intersect(V1, V2, int(m), max(int(m), 1))
    except (OverflowError, TypeError):
        V1 = [int(v) for v in V1]
        V2 = [int(v) for v in V2]
        return _k.gf2_intersect(V1, V2, int(m), max(_bits(V1), _bits(V2), 1))


# ------------------------------------------------------------------ tapped value simulator (card E3, P3 primitive)
# Reference: the bit-parallel value loop of scripts/qhdl_simulate.py (simulate) and its copies in
# scripts/dqi_l27c_andprints.py, dqi_l27c_blocks.py, dqi_l29c_rare.py and xpath_l33_dqi.py in the competition
# repository. Values only (phases ignored); HMR outcomes from CPython's Mersenne Twister, random.Random(seed)
# .getrandbits(lanes) per HMR, so outputs are identical to the Python loop's.
#
#   value_sim(path, wires, cbits, lanes, *, seed=1, outcomes='mt'|'zero'|'explicit', explicit=None,
#             points=None, windows=None, dep_cuts=None, prints=False, rare_cap=None, tflag=False, stop=None,
#             threads=None) -> dict
#     wires uint64[nq, W], cbits uint64[num_bits + 1, W] (lane i = bit i of the little-endian words, W = ceil(L/64))
#     points   = (ops, [wire list per op])            -> points uint64[sum, W], point_ptr (state before each op)
#     windows  = (start ops, end ops, [wire list])    -> d_win, d_wire, d_vec uint64[n, W], d_before0, d_after0
#                                                        (nonzero start ^ end deltas, ordered by (end op, window))
#     dep_cuts = ops                                  -> dep bool[ncuts, nq + 1] (rows w_0..w_{nq-1}, ALL lie in the
#                                                        span of the others), dep_rank
#     prints   -> and_op, and_kind, and_pw, and_flag, and_p, and_pc, and_o1, and_o2 (dqi_l27c_andprints fields)
#     rare_cap -> and_op, and_kind, and_pw, rare_lanes, rare_ptr (dqi_l29c_rare fields; ptr without the leading 0)
#     tflag    -> tflag int8 per Toffoli-like op (dqi_l27c_blocks)
#     always   -> end_wires uint64[nq, W], end_cbits, n_hmr, ops_run
#   pack_lanes(bool_or_uint8_rows) -> uint64[rows, W];  words_to_int(row) -> Python int (as qhdl_simulate._pack)


def _seed_key(seed):
    n = abs(int(seed))
    key = []
    while n:
        key.append(n & 0xFFFFFFFF)
        n >>= 32
    return key or [0]


def pack_lanes(rows):
    """Rows of L lane bits (bool/uint8, shape [n, L]) -> uint64[n, ceil(L/64)], lane i = bit i."""
    a = np.ascontiguousarray(np.asarray(rows).astype(np.uint8))
    if a.ndim == 1:
        a = a[None, :]
    n, L = a.shape
    W = (L + 63) // 64
    b = np.packbits(a, axis=1, bitorder='little')
    out = np.zeros((n, W * 8), np.uint8)
    out[:, :b.shape[1]] = b
    return out.view('<u8').reshape(n, W)


def words_to_int(row):
    return int.from_bytes(np.ascontiguousarray(row, '<u8').tobytes(), 'little')


def _csr(lists):
    lists = [np.asarray(x, np.int64).ravel() for x in lists]
    ptr = np.zeros(len(lists) + 1, np.int64)
    if lists:
        ptr[1:] = np.cumsum([len(x) for x in lists])
    flat = np.concatenate(lists) if lists else np.zeros(0, np.int64)
    return ptr, np.ascontiguousarray(flat, np.int64)


def value_sim(path, wires, cbits, lanes, *, seed=1, outcomes='mt', explicit=None, points=None, windows=None,
              dep_cuts=None, prints=False, rare_cap=None, tflag=False, stop=None, threads=None):
    import os
    wires = np.ascontiguousarray(wires, np.uint64)
    cbits = np.ascontiguousarray(cbits, np.uint64)
    W = (int(lanes) + 63) // 64
    kw = {}
    if points is not None:
        ops, lists = points
        kw['point_ops'] = np.ascontiguousarray(ops, np.int64)
        kw['point_ptr'], kw['point_wires'] = _csr(lists)
    if windows is not None:
        s, e, lists = windows
        kw['win_s'] = np.ascontiguousarray(s, np.int64)
        kw['win_e'] = np.ascontiguousarray(e, np.int64)
        kw['win_ptr'], kw['win_wires'] = _csr(lists)
    if dep_cuts is not None:
        kw['dep_ops'] = np.ascontiguousarray(dep_cuts, np.int64)
    if explicit is not None:
        kw['explicit'] = np.ascontiguousarray(explicit, np.uint64).ravel()
    if threads is None:
        threads = max(1, min(4, (os.cpu_count() or 2) - 1))   # 4 lane slices measured best on the M2 Pro (card E3)
    r = _k._value_sim(str(path), int(lanes), wires.ravel(), cbits.ravel(), outcomes, _seed_key(seed),
                      prints=bool(prints), rare_cap=None if rare_cap is None else int(rare_cap), tflag=bool(tflag),
                      stop=None if stop is None else int(stop), threads=int(threads), **kw)
    r['points'] = r['points'].reshape(-1, W)
    r['point_ptr'] = kw.get('point_ptr', np.zeros(1, np.int64))
    r['d_vec'] = r['d_vec'].reshape(-1, W)
    nq = wires.shape[0] if wires.ndim == 2 else wires.size // W
    r['dep'] = r['dep'].reshape(-1, nq + 1)
    r['end_wires'] = r['end_wires'].reshape(-1, W)
    r['end_cbits'] = r['end_cbits'].reshape(-1, W)
    return r


def mt_getrandbits(seed, k, n):
    """n successive random.Random(seed).getrandbits(k) draws as uint64[n, ceil(k/64)] (parity tests)."""
    return _k._mt_getrandbits(_seed_key(seed), int(k), int(n)).reshape(n, -1)


def gf2_dependent(rows):
    """rows uint64[n, W] -> (bool[n] row in the span of the others, rank)."""
    rows = np.ascontiguousarray(rows, np.uint64)
    n, W = rows.shape
    return _k._gf2_dependent(rows.ravel(), n, W)
