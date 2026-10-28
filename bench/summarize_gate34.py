"""Summarise gate 3/4 memwatch records (bench/dqi_gate34.sh).

Usage: summarize_gate34.py EVIDENCE_DIR TAG OUT.json [--pred-t1 MIB --pred-t10 MIB]
Every engine variant found (engine, engine-cold, engine-warm) is compared with the fork at each parallelism level:
median wall and speed-up, max peak footprint, prediction error (if given), and whether all results are identical.
"""
import argparse, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('dir'); ap.add_argument('tag'); ap.add_argument('out')
ap.add_argument('--pred-t1', type=float); ap.add_argument('--pred-t10', type=float)
a = ap.parse_args()
if os.path.exists(a.out):
    raise SystemExit(f'refusing to overwrite {a.out}')
pred = {1: a.pred_t1, 10: a.pred_t10}
rows = []
for f in sorted(glob.glob(f'{a.dir}/gate34_{a.tag}_*_r*.json')):
    m = re.search(r'_(fork|engine(?:-[a-z]+)?)_[wt](\d+)_r(\d+)\.json$', f)
    if not m:
        continue
    rec = json.load(open(f))
    res = rec['result'] or {}
    rows.append(dict(file=os.path.basename(f), who=m[1], n=int(m[2]), round=int(m[3]), rc=rec['returncode'],
                     wall=rec['wall_s'], fp=rec['peak_footprint_mib'], rss=rec['peak_rss_mib'], nproc=rec['max_processes'],
                     load=rec['uptime_before'].split('load averages:')[-1].strip(),
                     key={k: res.get(k) for k in ('value_errors', 'phase_errors', 'repetition_factor', 'toffoli_count', 'score')}))
S = {}
variants = sorted({r['who'] for r in rows if r['who'] != 'fork'})
for v in variants:
    for n in sorted({r['n'] for r in rows}):
        f = [r for r in rows if r['who'] == 'fork' and r['n'] == n]
        e = [r for r in rows if r['who'] == v and r['n'] == n]
        if not f or not e:
            continue
        fw, ew = float(np.median([r['wall'] for r in f])), float(np.median([r['wall'] for r in e]))
        ffp, efp = max(r['fp'] for r in f), max(r['fp'] for r in e)
        s = dict(fork_wall_median=fw, engine_wall_median=ew, speedup_median=fw / ew,
                 fork_walls=[r['wall'] for r in f], engine_walls=[r['wall'] for r in e],
                 fork_peak_footprint_mib_max=ffp, engine_peak_footprint_mib_max=efp,
                 engine_peak_footprint_mib_all=[r['fp'] for r in e], fork_peak_footprint_mib_all=[r['fp'] for r in f],
                 below_fork=efp < ffp, loads=[r['load'] for r in f + e],
                 results_identical=len({json.dumps(r['key'], sort_keys=True) for r in f + e}) == 1,
                 all_rc0=all(r['rc'] == 0 for r in f + e))
        if pred.get(n):
            s.update(prediction_mib=pred[n], engine_vs_prediction=efp / pred[n] - 1, within_20pct=abs(efp / pred[n] - 1) <= 0.2)
        S[f'{v}@{n}'] = s
summary = dict(comparisons=S, target_1p5x_at_10={k: S[k]['speedup_median'] >= 1.5 for k in S if k.endswith('@10')}, rows=rows)
json.dump(summary, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
for k, s in S.items():
    extra = f" pred {s['prediction_mib']:.0f} ({100 * s['engine_vs_prediction']:+.1f}%)" if 'prediction_mib' in s else ''
    print(f"{k}: fork {s['fork_wall_median']:.2f}s engine {s['engine_wall_median']:.2f}s x{s['speedup_median']:.2f} | "
          f"footprint fork {s['fork_peak_footprint_mib_max']:.0f} engine {s['engine_peak_footprint_mib_max']:.0f} MiB{extra} "
          f"identical={s['results_identical']} rc0={s['all_rc0']}")
print('1.5x at 10:', summary['target_1p5x_at_10'])
