"""Addendum A proof: the engine's DQI input sampler equals the official `sample_shots` (run with an interpreter where
kickmix imports, since the official verifier module imports it).

Usage: sampler_parity_dqi.py OUT.json --kmb A.kmb [--kmb B.kmb] [--salts ,salt-check-1,salt-check-2] [--verifier PATH]
       [--batches all|0,1,2] [--threads 10]

For every verifier batch (seeds from the verifier's own `_initial_seed` and SeedSequence split; the circuit hash is
the file SHA-256, equal to the verifier's re-serialisation hash for these circuits, see M3 gate 1) the official
`sample_shots(spec, tables, default_rng(rng_seed), 256, weight_probs)` is compared with the engine sampler on:
syndromes [256, n] and mbuc [256, m] values, phases [256], the bit-sliced words the engine simulates (syndrome rows
and every E row, i.e. all m*degree MBUC rows, not only the rows the circuit touches) and the full PCG64 state after
the batch (state, inc, has_uint32, uinteger), which shows the same number of draws was consumed. Negative controls:
a perturbed generator state and a shifted weight CDF must mismatch.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from kickmix_engine import _engine, words_to_bools
from kickmix_engine import dqi
from kickmix_engine.cache import file_sha256

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--kmb', action='append', required=True)
ap.add_argument('--salts', default=',salt-check-1,salt-check-2')
ap.add_argument('--verifier', default=None)
ap.add_argument('--batches', default='all')
ap.add_argument('--threads', type=int, default=10)
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')

v = dqi.load_verifier(a.verifier)
spec = v.ProblemSpecDQI.benchmark_instance()
tables = v.FieldTables(v.km.GF2Field(spec.degree))
wp = v.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2
all_rows = np.arange(spec.m * spec.degree)
res = dict(verifier=v.__file__, numpy=np.__version__, engine=_engine.__file__, engine_version=_engine.engine_version(),
           runs=[], uptime=os.popen('uptime').read().strip())
all_ok = True


class _Task(dqi.DQITask):
    """DQITask without compiling: the sampler needs only the spec and the verifier (all E rows are compared)."""

    def __init__(self, path):
        self.v, self.spec, self.path = v, spec, path
        self.e_keep = all_rows
        self._sargs = None


for path in a.kmb:
    name = os.path.basename(path)[:12]
    t = _Task(path)
    t.circuit_hash = file_sha256(path)
    for salt in a.salts.split(','):
        tasks = t.tasks(salt)
        idx = list(range(len(tasks))) if a.batches == 'all' else [int(x) for x in a.batches.split(',')]
        t0 = time.perf_counter()
        eng = t.engine_sample([tasks[i] for i in idx], threads=a.threads, values=True)
        t_eng = time.perf_counter() - t0
        mism = dict(syndromes=0, mbuc=0, phases=0, qin_words=0, bin_words=0, end_state=0)
        bad_batches = []
        t_off = 0.0
        n_phase_true = 0
        for j, i in enumerate(idx):
            rng = np.random.default_rng(tasks[i][1])
            t1 = time.perf_counter()
            syn, mbuc, ph = v.sample_shots(spec, tables, rng, 256, wp)
            t_off += time.perf_counter() - t1
            checks = dict(
                syndromes=np.array_equal(syn, eng['syndromes'][j]),
                mbuc=np.array_equal(mbuc, eng['mbuc'][j]),
                phases=np.array_equal(ph, words_to_bools(eng['phase'][j])),
                qin_words=np.array_equal(dqi.pack_rows(syn, spec.degree), eng['qin'][j]),
                bin_words=np.array_equal(dqi.pack_rows(mbuc, spec.degree), eng['bin'][j]),
                end_state=rng.bit_generator.state == dqi.pcg_state_dict(eng['end_state'][j]),
            )
            for k, ok in checks.items():
                mism[k] += not ok
            if not all(checks.values()):
                bad_batches.append(i)
            n_phase_true += int(ph.sum())
        ok = not bad_batches and len(idx) > 0
        all_ok &= ok
        run = dict(circuit=name, salt=salt, batches=len(idx), pass_=ok, mismatches=mism, bad_batches=bad_batches[:20],
                   phases_true=n_phase_true, engine_seconds=round(t_eng, 3), official_seconds=round(t_off, 2))
        res['runs'].append(run)
        print(name, repr(salt), 'PASS' if ok else 'FAIL', mism, f'engine {t_eng:.2f}s official {t_off:.2f}s',
              flush=True)

# Negative controls (first circuit, empty salt, batch 0): a changed generator state and a changed CDF must mismatch.
t = _Task(a.kmb[0]); t.circuit_hash = file_sha256(a.kmb[0])
task0 = t.tasks('')[0]
syn, mbuc, ph = v.sample_shots(spec, tables, np.random.default_rng(task0[1]), 256, wp)
sa = t.sampler_args()
pcg = np.array([dqi.pcg_state_words(task0[1])], dtype=np.uint64)
pcg_bad = pcg.copy(); pcg_bad[0, 0] ^= np.uint64(1)
r1 = _engine.sample_dqi(sa['m'], sa['n'], sa['degree'], sa['cdf'], sa['log'], sa['antilog'], [], pcg_bad, 1, True)
cdf_bad = [0.0] + list(sa['cdf'][:-1]) + [1.0]  # every error weight shifted up by one
r2 = _engine.sample_dqi(sa['m'], sa['n'], sa['degree'], cdf_bad, sa['log'], sa['antilog'], [], pcg, 1, True)
ctrl = dict(
    state_flip_mbuc_differs=not np.array_equal(r1['mbuc'].reshape(256, spec.m), mbuc),
    cdf_change_syndromes_differ=not np.array_equal(r2['syndromes'].reshape(256, spec.n), syn),
)
res['negative_controls'] = ctrl
print('negative controls', ctrl)
res['summary'] = dict(sampler_parity_pass=bool(all_ok), controls_mismatch=all(ctrl.values()),
                      runs=len(res['runs']), batches=sum(r['batches'] for r in res['runs']))
res['uptime_end'] = os.popen('uptime').read().strip()
print(res['summary'])
json.dump(res, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
sys.exit(0 if all_ok and all(ctrl.values()) else 1)
