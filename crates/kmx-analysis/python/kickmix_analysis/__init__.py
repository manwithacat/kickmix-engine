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
