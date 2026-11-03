"""Dump one DQI verifier draw (seeds, sampler parameters, register ids) as JSON for the `kickmix-gpu` binary.

The binary re-samples every batch on the CPU with the engine's exact sampler (`DqiSampler`) and simulates it on the
CPU engine and on the GPU, so this file carries only what the Python task adapter computes with the official
verifier's functions: the per-batch `default_rng` start states and simulator seeds (`DQITask.tasks`), the weight CDF
and field tables (`DQITask.sampler_args`), the syndrome qubit ids in sampler row order and the kept MBUC bit rows with
their compact bit ids (`DQITask.prepare_checks`). It also compiles (or loads from the program cache) the program and
prints the cache path so the binary can `Program::load` it.

Usage (from any directory; set KICKMIX_ENGINE_VERIFIERS and put the engine's python/ on PYTHONPATH):
    python dump_dqi_tasks.py CIRCUIT.kmb OUT.json [--salt S] [--seed N] [--cache-dir DIR]
Written by an AI model (Claude) under the owner's direction; see the repository NOTICE.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

from kickmix_engine import KINDS, seed_words
from kickmix_engine.cache import ProgramCache
from kickmix_engine.dqi import DQITask, pcg_state_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kmb')
    ap.add_argument('out')
    ap.add_argument('--salt', default='')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--cache-dir', default=None)
    a = ap.parse_args()
    cache = ProgramCache(a.cache_dir) if a.cache_dir else ProgramCache()
    t = DQITask(a.kmb, cache=cache)
    t0 = time.perf_counter()
    t.prepare()
    t_prep = time.perf_counter() - t0
    tasks = t.tasks(a.salt, a.seed, 256)
    sa = t.sampler_args()
    p = t.prog
    sc = list(p.static_counts)
    out = dict(
        circuit=os.path.abspath(a.kmb), circuit_sha256=t.circuit_hash, salt=a.salt, seed=a.seed,
        program_path=cache.program_path(t.circuit_hash),
        program=dict(num_qubits=p.num_qubits, num_bits=p.num_bits, orig_num_bits=p.orig_num_bits,
                     num_ops=p.num_ops, num_stream_words=p.num_stream_ops, num_rand=p.num_rand,
                     num_info_tags=p.num_info_tags, max_stack_depth=p.max_stack_depth, num_cx_runs=p.num_cx_runs,
                     num_angles=p.num_angles, reaction_depth=p.reaction_depth,
                     static_counts={k: int(v) for k, v in zip(KINDS, sc) if v}),
        spec=dict(m=t.spec.m, n=t.spec.n, degree=t.spec.degree, r=t.spec.r, max_errors=t.spec.max_errors,
                  num_shots=t.spec.num_shots),
        sampler=dict(m=sa['m'], n=sa['n'], degree=sa['degree'], cdf=[float(x).hex() for x in sa['cdf']],
                     log=sa['log'], antilog=sa['antilog'], e_rows=[int(x) for x in t.e_keep]),
        qin_ids=[int(q) for q in t.syn_ids],
        bin_ids=[int(x) for x in t.e_compact],
        batches=[dict(start=int(s), pcg=[int(x) for x in pcg_state_words(rs)],
                      sim_seed=[int(x) for x in seed_words(ss)]) for (s, rs, ss) in tasks],
        prepare_seconds=t_prep, load_info={k: str(v) for k, v in (t.load_info or {}).items()},
    )
    with open(a.out, 'w') as f:
        json.dump(out, f)
    print(json.dumps(dict(program=out['program'], spec=out['spec'], batches=len(tasks),
                          program_path=out['program_path'], prepare_seconds=t_prep, load_info=out['load_info']),
                     indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
