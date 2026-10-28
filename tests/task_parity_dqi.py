"""M3 gate 2: full DQI verification, engine adapter vs the official verify() on kickmix (run with the kickmix interpreter).

Usage: task_parity_dqi.py OUT.json --kmb A.kmb [--kmb B.kmb] [--salts ,salt-check-1,salt-check-2] [--workers 6] [--threads 10]
       [--sampler engine|official] [--self-check K] [--cache-dir DIR | --no-cache] [--verifier PATH]

The official verifier module is used unchanged; its `_run_parallel` is wrapped (call-through) only to keep the per-batch
results it already computes, so batch-level differences can be located. Compared: value_errors, phase_errors,
effective_error_rate, repetition_factor, toffoli_count, score, shot_operation_counts and the other summary fields,
plus per-batch (value errors, phase errors, Toffoli count, op counts).
"""
import argparse, json, os, sys, time
import numpy as np
import kickmix as km

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
import kickmix_engine as ke
from kickmix_engine import dqi

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--kmb', action='append', required=True)
ap.add_argument('--salts', default=',salt-check-1,salt-check-2')
ap.add_argument('--workers', type=int, default=6)
ap.add_argument('--threads', type=int, default=10)
ap.add_argument('--sampler', default='engine', choices=('engine', 'official'))
ap.add_argument('--self-check', type=int, default=0)
ap.add_argument('--cache-dir', default=None)
ap.add_argument('--no-cache', action='store_true')
ap.add_argument('--verifier', default=None)
a = ap.parse_args()
from kickmix_engine.cache import ProgramCache
cache = ProgramCache(a.cache_dir, enabled=not a.no_cache)
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')

v = dqi.load_verifier(a.verifier)
captured = []
_orig = v._run_parallel


def _wrapped(fn, tasks, max_workers):
    res = _orig(fn, tasks, max_workers)
    captured[:] = res
    return res


v._run_parallel = _wrapped
FIELDS = ['status', 'score', 'm', 'n', 'degree', 'r', 'verified_shots', 'value_errors', 'phase_errors',
          'empirical_error_rate', 'effective_error_rate', 'log2_delta', 'repetition_factor', 'mean_error_weight',
          'toffoli_count', 'max_toffoli_count', 'num_ops', 'shot_operation_counts', 'num_qubits', 'num_bits',
          'circuit_sha256']
results = dict(runs=[], kickmix=km.__file__, engine=ke._engine.__file__, engine_version=ke.engine_version(),
               uptime_start=os.popen('uptime').read().strip(), workers=a.workers, threads=a.threads, sampler=a.sampler,
               self_check=a.self_check, cache_dir=cache.dir if cache.enabled else None)
all_ok = True
for path in a.kmb:
    name = os.path.basename(path)[:12]
    circuit = km.Circuit.from_file(path, format='kmb')
    spec = v.ProblemSpecDQI.benchmark_instance()
    task = dqi.DQITask(path, verifier_path=a.verifier, cache=cache).prepare()
    for salt in a.salts.split(','):
        t0 = time.perf_counter()
        ref = v.verify(circuit, spec, salt=salt, max_workers=a.workers)
        t_ref = time.perf_counter() - t0
        ref_batches = list(captured)
        t0 = time.perf_counter()
        eng = task.verify(salt=salt, threads=a.threads, sampler=a.sampler, self_check=a.self_check)
        t_eng = time.perf_counter() - t0
        diffs = {f: (ref[f], eng[f]) for f in FIELDS if ref[f] != eng[f]}
        batch_diffs = []
        for i, (rb, eb) in enumerate(zip(ref_batches, task.per_batch)):
            ec = {k: int(c) for k, c in zip(ke.KINDS, eb['counts']) if c}
            if (rb[0], rb[1], rb[2]) != (eb['val'], eb['phase'], eb['toffoli']) or rb[3] != ec:
                batch_diffs.append(dict(batch=i, ref=rb[:3], eng=(eb['val'], eb['phase'], eb['toffoli'])))
        failing = [dict(batch=eb['index'], val_shots=eb['val_shots'], phase_shots=eb['phase_shots'])
                   for eb in task.per_batch if eb['val'] or eb['phase']]
        ok = not diffs and not batch_diffs and len(ref_batches) == len(task.per_batch) == 400
        all_ok &= ok
        run = dict(circuit=name, salt=salt, pass_=ok, official={f: ref[f] for f in FIELDS + ['reaction_depth']},
                   engine_reaction_depth=eng['reaction_depth'], field_diffs=diffs, batch_diffs=batch_diffs[:20],
                   n_batch_diffs=len(batch_diffs), engine_failing_shots=failing,
                   official_seconds=round(t_ref, 1), engine_seconds=round(t_eng, 1), load=os.getloadavg(),
                   self_check=getattr(task, 'self_check_result', None), load_info=task.load_info)
        results['runs'].append(run)
        print(name, repr(salt), 'PASS' if ok else 'FAIL', {k: ref[k] for k in ('value_errors', 'phase_errors',
              'repetition_factor', 'toffoli_count', 'score')}, 'diffs', diffs, len(batch_diffs),
              f'official {t_ref:.1f}s engine {t_eng:.1f}s', flush=True)
    del circuit, task
results['summary'] = dict(task_parity_pass=bool(all_ok), runs=len(results['runs']))
results['uptime_end'] = os.popen('uptime').read().strip()
print(results['summary'])
json.dump(results, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
sys.exit(0 if all_ok else 1)
