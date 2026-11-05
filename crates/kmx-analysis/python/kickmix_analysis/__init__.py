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
