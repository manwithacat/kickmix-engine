"""Context for gate 3: kickmix `do` aggregate throughput with N forked processes (counting on), alternated with the
engine at N threads on the same inputs, so both see the same background load. Usage: kickmix_scaling.py OUT.json"""
import json, multiprocessing as mp, os, subprocess, sys, time
import numpy as np
import kickmix as km
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import kickmix_engine as ke
ROOT = '/Volumes/SSD/Kaggle/competitions/google-verifiable-quantum-circuit-compilation'
c = km.Circuit.from_file(ROOT + '/outputs/kickmix-popfast-2026-10-27/v3sub/invsqrt.kmb', format='kmb')
prog = ke.compile_circuit(c)
B = 1000
xs = ((np.arange(B * 256, dtype=np.uint64) % np.uint64(3 * 1022**2)) + 1).reshape(B, 256)
bits = (xs[:, None, :] >> np.arange(22, dtype=np.uint64)[None, :, None]) & 1 == 1
x_reg = km.array([km.q(i) for i in range(22)])
def work(i):
    sim = km.Simulator(batch_size=256, debug_print=False, counting_operations=True, seed=i + 1)
    t0 = time.perf_counter()
    for b in range(B):
        sim.clear_for_shot(); sim.write_across_shots(x_reg, bits[b]); sim.do(c)
    return time.perf_counter() - t0
out = []
for rnd in range(2):
    for n in (1, 2, 4, 8, 10):
        ld = os.getloadavg()[0]
        t0 = time.perf_counter()
        with mp.get_context('fork').Pool(n) as p: p.map(work, range(n))
        tk = time.perf_counter() - t0
        t0 = time.perf_counter()
        ke.run(prog, [i + 1 for i in range(n)], np.broadcast_to(xs[None], (n, B, 256)).copy(), list(range(22)), s=1, threads=n, counting=True)
        te = time.perf_counter() - t0
        out.append(dict(round=rnd, n=n, load=ld, kickmix_shots_per_s=n * B * 256 / tk, engine_shots_per_s=n * B * 256 / te))
        print(out[-1], flush=True)
json.dump(dict(runs=out, uptime=subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()), open(sys.argv[1], 'w'), indent=1)
