"""Gate 3: throughput vs threads (engine only, counting on, verifier inputs). Run with the kickmix interpreter
(needed once to load the .kmb). Usage: scaling.py OUT.json [--chunks N] [--rounds R]"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import kickmix as km
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import kickmix_engine as ke
ROOT = '/Volumes/SSD/Kaggle/competitions/google-verifiable-quantum-circuit-compilation'
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--chunks', type=int, default=320)
ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--threads', default='1,2,4,8,10'); a = ap.parse_args()
c = km.Circuit.from_file(ROOT + '/outputs/kickmix-popfast-2026-10-27/v3sub/invsqrt.kmb', format='kmb')
prog = ke.compile_circuit(c)
MAX_X = 3 * (2**10 - 2)**2
n = a.chunks
xs = ((np.arange(n * 64 * 256, dtype=np.uint64) % np.uint64(MAX_X)) + np.uint64(1)).reshape(n, 64, 256)
seeds = [1000 + i for i in range(n)]
out = dict(shots=int(xs.size), runs=[])
for rnd in range(a.rounds):
    for t in [int(x) for x in a.threads.split(',')]:
        ld = os.getloadavg()[0]
        t0 = time.perf_counter(); ke.run(prog, seeds, xs, list(range(22)), s=1, threads=t, counting=True)
        dt = time.perf_counter() - t0
        out['runs'].append(dict(round=rnd, threads=t, seconds=dt, shots_per_s=xs.size / dt, load=ld))
        print(out['runs'][-1], flush=True)
sp = {}
for t in sorted({r['threads'] for r in out['runs']}):
    best = max(r['shots_per_s'] for r in out['runs'] if r['threads'] == t)
    med = float(np.median([r['shots_per_s'] for r in out['runs'] if r['threads'] == t]))
    sp[t] = dict(best_shots_per_s=best, median_shots_per_s=med)
base_b, base_m = sp[1]['best_shots_per_s'], sp[1]['median_shots_per_s']
for t in sp: sp[t]['speedup_best'] = sp[t]['best_shots_per_s'] / base_b; sp[t]['speedup_median'] = sp[t]['median_shots_per_s'] / base_m
out['summary'] = sp; out['uptime'] = subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()
out['pass_ge_7x_at_10'] = sp.get(10, {}).get('speedup_median', 0) >= 7
print(json.dumps(sp, indent=1)); json.dump(out, open(a.out, 'w'), indent=1)
