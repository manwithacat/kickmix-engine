"""Gate 2: single-core time per 256-shot batch, engine vs kickmix `do` (counting on), alternating runs.
Run with the kickmix interpreter. Usage: single_core.py OUT.json --kmb invsqrt.kmb [--verifier PATH] [--rounds N]
[--batches B]. Verifier default: $KICKMIX_ENGINE_INVSQRT_VERIFIER or $KICKMIX_ENGINE_VERIFIERS/invsqrt/..."""
import argparse, json, os, subprocess, sys, time
import numpy as np
import kickmix as km
import kickmix_engine as ke
from kickmix_engine.invsqrt import load_verifier
ap = argparse.ArgumentParser(); ap.add_argument('out'); ap.add_argument('--rounds', type=int, default=8)
ap.add_argument('--batches', type=int, default=256); ap.add_argument('--kmb', required=True)
ap.add_argument('--verifier', default=None); a = ap.parse_args()
v = load_verifier(a.verifier)
c = km.Circuit.from_file(a.kmb, format='kmb')
sp = v.ProblemSpecInvSqrt.for_n_p(10)
x_reg, scratch = v.validate_registers(c, sp)
prog = ke.compile_circuit(c)
B = a.batches
xs = np.stack([v.sample_shots(sp, i * 256, i * 256 + 256) for i in range(B)])
bits = [v.to_bits(x, 22) for x in xs]
def load(): return os.getloadavg()[0]
res = dict(kickmix_ms=[], engine_ms=[], engine_core_counting_off_ms=[], load=[])
for r in range(a.rounds):
    res['load'].append(load())
    sim = km.Simulator(batch_size=256, debug_print=False, counting_operations=True, seed=r + 1)
    t = 0.0
    for b in range(B):
        sim.clear_for_shot(); sim.write_across_shots(x_reg, bits[b])
        t0 = time.perf_counter(); sim.do(c); t += time.perf_counter() - t0
    res['kickmix_ms'].append(1e3 * t / B)
    t0 = time.perf_counter()
    ke.run(prog, [r + 1], xs[None], list(range(22)), s=1, threads=1, counting=True)
    res['engine_ms'].append(1e3 * (time.perf_counter() - t0) / B)
    t0 = time.perf_counter()
    ke.run(prog, [r + 1], xs[None], list(range(22)), s=1, threads=1, counting=False)
    res['engine_core_counting_off_ms'].append(1e3 * (time.perf_counter() - t0) / B)
med = lambda k: float(np.median(res[k]))
res['median'] = {k: med(k) for k in ('kickmix_ms', 'engine_ms', 'engine_core_counting_off_ms')}
res['ratio_engine_over_kickmix'] = res['median']['engine_ms'] / res['median']['kickmix_ms']
res['pass_le_1.2'] = res['ratio_engine_over_kickmix'] <= 1.2
res['uptime'] = subprocess.run(['uptime'], capture_output=True, text=True).stdout.strip()
res['note'] = 'engine time = whole ke.run call per batch (x write/read, units, counting on); kickmix time = sim.do only'
print(json.dumps(res, indent=1)); json.dump(res, open(a.out, 'w'), indent=1)
