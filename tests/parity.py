"""Per-shot parity of kickmix-engine vs kickmix on the invsqrt circuit (run with the kickmix interpreter).

Usage: parity.py OUT.json [--chunks 0,1,2,...] [--random-batches N] [--kmb PATH]
Compares, for every 256-shot batch and every shot: all qubit values, all classical bits (compacted ids mapped back; all
untouched bits must be zero in kickmix), the exact 128-bit phase V, the verifier's 2^-64-turn units, x register
read-back, scratch-nonzero flags; per chunk: per-op-kind counts, Z_POW level counts, Toffoli counts.
Includes a wrong-seed negative control that must mismatch.
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import kickmix as km

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python"))
import kickmix_engine as ke

ROOT = '/Volumes/SSD/Kaggle/competitions/google-verifiable-quantum-circuit-compilation'
VER = ROOT + '/data/official_full/verifiers/invsqrt/verify_submission_invsqrt.py'
spec_ = importlib.util.spec_from_file_location('vinv', VER); v = importlib.util.module_from_spec(spec_)
sys.modules['vinv'] = v; spec_.loader.exec_module(v)

ap = argparse.ArgumentParser()
ap.add_argument('out'); ap.add_argument('--chunks', default='0,1,2,5000,12236')
ap.add_argument('--random-batches', type=int, default=8)
ap.add_argument('--kmb', default=ROOT + '/outputs/kickmix-popfast-2026-10-27/v3sub/invsqrt.kmb')
a = ap.parse_args()

circuit = km.Circuit.from_file(a.kmb, format='kmb')
sp = v.ProblemSpecInvSqrt.for_n_p(10)
x_reg, scratch = v.validate_registers(circuit, sp)
xids = [e.id if hasattr(e, 'id') else int(str(e)[1:]) for e in x_reg]
nq, nbits_all = circuit.num_qubits, circuit.num_bits
prog = ke.compile_circuit(circuit)
bit_map = np.array(prog.bit_map, dtype=np.int64)
untouched = np.ones(nbits_all, bool); untouched[bit_map] = False
allq = km.array([km.q(i) for i in range(nq)])
allb = km.array([km.b(i) for i in range(nbits_all)])
kinds = ke.KINDS


def unpack(words_u64):  # [n, 4] u64 -> [n, 256] uint8
    return np.unpackbits(np.ascontiguousarray(words_u64).view(np.uint8).reshape(len(words_u64), 32), axis=1, bitorder='little')


def kickmix_chunk(seed, xs_batches, S):
    """Run one Simulator(batch_size=256*S, seed) like the verifier; returns per-batch records and stats."""
    BS = 256 * S
    sim = km.Simulator(batch_size=BS, debug_print=False, counting_operations=True, seed=seed)
    recs = []
    for xs in xs_batches:
        sim.clear_for_shot()
        sim.write_across_shots(x_reg, v.to_bits(xs, 22))
        sim.do(circuit)
        q = sim.read_across_shots(allq).astype(np.uint8)
        b = sim.read_across_shots(allb).astype(np.uint8)
        xr = v.from_bits(sim.read_across_shots(x_reg))
        dirty = sim.read_across_shots(scratch).any(axis=0).astype(np.uint8)
        V = np.zeros((BS, 2), np.uint64); U = np.zeros(BS, np.uint64)
        for i in range(BS):
            h = sim.read_shot_phase(i)
            val = int(h * (1 << 127)) % (1 << 128)
            V[i] = (val & ((1 << 64) - 1), val >> 64)
            U[i] = v.half_turns_to_units(h)
        recs.append((q, b, xr, dirty, V, U))
    st = sim.operation_statistics()
    d = dict(st)
    kc = np.array([d.get(k, 0) for k in kinds], np.uint64)
    zl = st.z_pow_clifford_hierarchy_counts
    z = np.zeros(129, np.uint64)
    for k_, c_ in dict(zl).items():
        z[int(k_)] = c_
    return recs, kc, z, int(st.num_toffoli_like)


def compare(name, seeds, xs_all, S, B, results, engine_seeds=None):
    """seeds: python ints per chunk; xs_all [n_chunks, B, 256*S]. Returns mismatch counters."""
    t0 = time.time()
    ref = [kickmix_chunk(sd, xs_all[c], S) for c, sd in enumerate(seeds)]
    t_ref = time.time() - t0
    es = seeds if engine_seeds is None else engine_seeds
    res = ke.run(prog, es, xs_all, xids, s=S, threads=2, counting=True, dump=True)
    mm = dict(qubit_bits=0, classical_bits=0, untouched_bits_nonzero=0, phase_exact_shots=0, units_shots=0,
              x_shots=0, dirty_shots=0, kind_count_chunks=0, zlevel_chunks=0, toffoli_chunks=0)
    nshots = 0
    for c, (recs, kc, z, tof) in enumerate(ref):
        for b, (q, bits, xr, dirty, V, U) in enumerate(recs):
            for k in range(S):
                u = (c * B + b) * S + k
                sl = slice(256 * k, 256 * k + 256)
                mm['qubit_bits'] += int((unpack(res['dump_q'][u]) != q[:, sl]).sum())
                eb = unpack(res['dump_bits'][u][:prog.num_bits])
                mm['classical_bits'] += int((eb != bits[bit_map][:, sl]).sum())
                mm['untouched_bits_nonzero'] += int(bits[untouched][:, sl].sum())
                ev = res['dump_vphase'][u]
                mm['phase_exact_shots'] += int((ev != V[sl]).any(axis=1).sum())
                mm['units_shots'] += int((res['phase_units'][c, b, sl] != U[sl]).sum())
                mm['x_shots'] += int((res['x_out'][c, b, sl] != xr[sl]).sum())
                mm['dirty_shots'] += int((res['scratch_dirty'][c, b, sl] != dirty[sl]).sum())
                nshots += 256
        mm['kind_count_chunks'] += int((res['kind_counts'][c] != kc).any())
        mm['zlevel_chunks'] += int((res['z_levels'][c] != z).any())
        mm['toffoli_chunks'] += int(res['toffoli_count'][c] != tof)
    ok = all(val == 0 for val in mm.values())
    results[name] = dict(shots=nshots, chunks=len(seeds), S=S, B=B, mismatches=mm, pass_=ok, kickmix_s=round(t_ref, 1),
                         toffoli_per_shot=float(res['toffoli_count'].sum() / nshots))
    print(name, results[name], flush=True)
    return ok, mm


results = {}
# 1. Verifier's own inputs and seeding.
circuit_hash = v._circuit_sha256(circuit)
base_seed = v._initial_seed(circuit_hash, '')
chunk_shots = v.BATCHES_PER_CHUNK * 256
starts = list(range(0, sp.num_shots, chunk_shots))
sd_all = np.random.SeedSequence(base_seed, pool_size=8).generate_state(4 * len(starts), dtype=np.uint64).reshape(-1, 4)
chunks = [int(x) for x in a.chunks.split(',')]
vseeds, vxs = [], []
for c in chunks:
    vseeds.append(int.from_bytes(sd_all[c], 'little'))
    st_, sp_ = starts[c], min(starts[c] + chunk_shots, sp.num_shots)
    xs = np.stack([v.sample_shots(sp, s0, s0 + 256) for s0 in range(st_, sp_, 256)])
    vxs.append(np.pad(xs, ((0, v.BATCHES_PER_CHUNK - len(xs)), (0, 0)), mode='edge') if len(xs) < 64 else xs)
vxs = np.stack(vxs)
ok1, _ = compare('verifier_chunks_' + a.chunks.replace(',', '_'), vseeds, vxs, 1, 64, results)
# negative control: engine seeded with seed+1 must mismatch.
okc, mmc = compare('control_wrong_seed', vseeds[:1], vxs[:1], 1, 64, results, engine_seeds=[vseeds[0] + 1])
results['control_wrong_seed']['expected_to_fail'] = True
results['control_wrong_seed']['control_ok'] = not okc
# 2. Random inputs (incl. x = 0), S = 4 sims per chunk, several seeds.
rng = np.random.default_rng(12345)
rs = [int.from_bytes(rng.bytes(32), 'little') for _ in range(3)]
rx = rng.integers(0, 1 << 22, size=(3, a.random_batches, 1024), dtype=np.uint64); rx[:, 0, :8] = 0
ok2, _ = compare('random_S4', rs, rx, 4, a.random_batches, results)
total = sum(r['shots'] for k, r in results.items() if k != 'control_wrong_seed')
results['summary'] = dict(total_compared_shots=total, parity_pass=bool(ok1 and ok2), control_mismatched=bool(not okc))
print(results['summary'])
json.dump(results, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
sys.exit(0 if (ok1 and ok2 and not okc) else 1)
