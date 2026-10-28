"""0.5.0 gate 1: per-shot parity of `run_sequence` against kickmix on dfthcsos families (run with a kickmix interpreter).

Usage: parity_dfthcsos.py OUT.json --family DIR [--family DIR ...] [--verifier DFTHCSOS_DIR] [--salt S]
       [--shots-per-address N] [--threads N] [--no-controls]

Oracle: the official `recover_words_from_circuits` itself, unchanged, with kickmix's Simulator wrapped (call-through)
so that the whole state is recorded after every `do()`: every qubit of the circuit, every classical bit, the phase
flip and the cumulative Toffoli-like count. Engine: one `run_sequence` call over the 2K circuit applications with
the verifier's seed, inputs and order, with dump=True. Compared per step and shot: every qubit, every classical bit,
the phase flip, the selection and word registers (x_out), the workspace flag; per step the Toffoli count (and every
other per-kind count against kickmix's operation_statistics deltas). Recovered words and toffoli/shot are compared
with the official return values.

Negative controls (must mismatch): wrong simulator seed (seed ^ 1); no RNG continuity (each step a fresh simulator,
i.e. a separate `run_sequence` call per step with the same seed). Written by an AI model (Claude) under the owner's
direction.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import kickmix as km

import kickmix_engine as ke
from kickmix_engine import dfthcsos

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--family', action='append', required=True)
ap.add_argument('--verifier', default=None)
ap.add_argument('--salt', default='')
ap.add_argument('--shots-per-address', type=int, default=None)
ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
ap.add_argument('--no-controls', action='store_true')
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')

off = dfthcsos.load_verifier(a.verifier)
V, C = off.V, off.C
_Real = km.Simulator


def packed(b):
    return np.packbits(np.asarray(b, bool), axis=-1, bitorder='little')


def words_to_shot_bits(words, shots):
    """[sims, n, 4] u64 bit-sliced -> bool [n, shots]."""
    w = np.ascontiguousarray(np.transpose(words, (1, 0, 2)))  # [n, sims, 4]
    b = np.unpackbits(w.view(np.uint8).reshape(w.shape[0], -1), axis=1, bitorder='little')
    return b[:, :shots].astype(bool)


def family_report(fam):
    t_fam = time.perf_counter()
    circuits = C.read_circuit_family(fam)
    K = len(circuits)
    npz = os.path.join(fam, V.SUBMISSION_FILE)
    params = V.load_factorization(npz)
    edges = V.load_tree_edges(None, fam)
    N = params['U'].shape[-1]
    R, B, _ = params['W'].shape
    num_vectors = N + R * B
    reps = a.shots_per_address or C.SHOTS_PER_ADDRESS
    shots = num_vectors * reps
    digest = C.submission_sha256(circuits, npz, edges)
    base_seed = C.initial_seed(digest, a.salt)
    sim_seed = int.from_bytes(np.random.SeedSequence(base_seed, pool_size=8).generate_state(4, dtype=np.uint64),
                              'little')
    progs = [ke.compile_kmb(C.kmb_path(fam, k)) for k in range(K)]
    order = list(range(K)) + list(reversed(range(K)))
    rep = {'family': os.path.abspath(fam), 'K': K, 'N': N, 'R': R, 'B': B, 'num_vectors': num_vectors,
           'shots_per_address': reps, 'shots': shots, 'tail_lanes': shots % 256, 'salt': a.salt,
           'submission_sha256': digest,
           'files_sha256': {os.path.basename(str(C.kmb_path(fam, k))): hashlib.sha256(
               open(C.kmb_path(fam, k), 'rb').read()).hexdigest() for k in range(K)}}

    snaps = []

    class Snap:
        def __init__(self, *args, **kw):
            self._s = _Real(*args, **kw)
            self._last = None

        def __getattr__(self, name):
            return getattr(self._s, name)

        def do(self, circuit):
            t = time.perf_counter()
            self._s.do(circuit)
            dt = time.perf_counter() - t
            nq = circuit.num_qubits
            k = [i for i, c in enumerate(circuits) if c is circuit][0]
            nb = progs[k].orig_num_bits
            q = self._s.read_across_shots(km.array([km.q(i) for i in range(nq)]))
            bits = self._s.read_across_shots(km.array([km.b(i) for i in range(nb)])) if nb else np.zeros((0, shots), bool)
            st = self._s.operation_statistics()
            counts = {str(kk): int(v) for kk, v in dict(st).items()}
            snaps.append({'k': k, 'q': packed(q), 'bits': packed(bits),
                          'phase': np.asarray(self._s.read_phase_flipped_across_shots(), bool),
                          'toffoli': int(st.num_toffoli_like), 'counts': counts, 'do_s': dt})

    saved = C.km
    C.km = type('kmproxy', (), {'Simulator': Snap, 'q': km.q, 'b': km.b, 'array': km.array, 'Circuit': km.Circuit})
    try:
        t0 = time.perf_counter()
        words_off, layout, tof_off = C.recover_words_from_circuits(
            circuits, num_vectors=num_vectors, num_angles=N - 1, shots_per_address=reps, seed=base_seed)
        rep['oracle_s'] = time.perf_counter() - t0
    finally:
        C.km = saved
    assert len(snaps) == len(order), (len(snaps), len(order))
    sel = [q.id for q in layout.selection]
    xreg = sel + [q.id for w in layout.words for q in w]
    public = set(xreg)
    xw = max(1, -(-len(xreg) // 64))
    addresses = np.tile(np.arange(num_vectors, dtype=np.uint64), reps)
    x0 = np.zeros((shots, xw), np.uint64)
    x0[:, 0] = addresses
    rep['xreg_width'] = len(xreg)

    t0 = time.perf_counter()
    out = ke.run_sequence([progs[k] for k in order], ke.seed_words(sim_seed), x0, xreg, threads=a.threads, dump=True)
    rep['engine_s'] = time.perf_counter() - t0

    def compare(out, label):
        steps = []
        tot = {'qubit_shots': 0, 'bit_shots': 0, 'phase_shots': 0, 'x_out_shots': 0, 'dirty_shots': 0,
               'toffoli_steps': 0, 'count_steps': 0}
        prev_counts = {}
        prev_tof = 0
        for i, k in enumerate(order):
            sn = snaps[i]
            nq = circuits[k].num_qubits
            q_off = np.unpackbits(sn['q'], axis=-1, bitorder='little')[:, :shots].astype(bool)
            q_eng = words_to_shot_bits(out['dump_q'][i][:, :nq, :], shots)
            nb = progs[k].orig_num_bits
            b_off = np.unpackbits(sn['bits'], axis=-1, bitorder='little')[:, :shots].astype(bool) if nb else None
            bc = progs[k].bit_compact
            if nb:
                db = words_to_shot_bits(out['dump_bits'][i], shots)
                b_eng = np.zeros((nb, shots), bool)
                for b in range(nb):
                    if bc[b] >= 0:
                        b_eng[b] = db[bc[b]]
            # x_out against the oracle's register qubits
            xo = out['x_out'][i]
            xbits = np.stack([(xo[:, j // 64] >> np.uint64(j % 64)) & np.uint64(1) for j in range(len(xreg))]).astype(bool)
            work = [qq for qq in range(nq) if qq not in public]
            dirty_off = q_off[work].any(axis=0) if work else np.zeros(shots, bool)
            tof_step = sn['toffoli'] - prev_tof
            prev_tof = sn['toffoli']
            delta = {kk: v - prev_counts.get(kk, 0) for kk, v in sn['counts'].items()}
            prev_counts = sn['counts']
            eng_counts = {kn: int(out['kind_counts'][i, j]) for j, kn in enumerate(ke.KINDS)}
            cmp_counts = {kk: (delta.get(kk, 0), eng_counts.get(kk, 0)) for kk in ('CX', 'CCX', 'CCZ', 'CSWAP', 'CZ',
                                                                                   'HMR', 'R', 'X', 'Z', 'SWAP')}
            count_mismatch = {kk: v for kk, v in cmp_counts.items() if v[0] != v[1]}
            m = {
                'qubit_shots': int((q_off != q_eng).any(axis=0).sum()),
                'bit_shots': int((b_off != b_eng).any(axis=0).sum()) if nb else 0,
                'phase_shots': int((sn['phase'] != out['phase'][i]).sum()),
                'x_out_shots': int((xbits != q_off[xreg]).any(axis=0).sum()),
                'dirty_shots': int((dirty_off != out['dirty'][i]).sum()),
                'toffoli_steps': int(int(out['toffoli_count'][i]) != tof_step),
                'count_steps': int(bool(count_mismatch)),
            }
            for kk in tot:
                tot[kk] += m[kk]
            steps.append({'step': i, 'k': k, 'mismatch': m, 'toffoli': tof_step, 'count_mismatch': count_mismatch})
        return steps, tot

    steps, tot = compare(out, 'engine')
    # words and toffoli/shot as the adapter computes them
    info = {}
    w_eng, _, tof_eng = dfthcsos.recover_words(off, circuits, lambda: progs, num_vectors=num_vectors, num_angles=N - 1,
                                               shots_per_address=reps, seed=base_seed, threads=a.threads, info=info)
    rep['parity'] = {'mismatch_totals': tot, 'steps': steps,
                     'words_equal_official': bool(np.array_equal(w_eng, words_off)),
                     'toffoli_per_shot_engine': tof_eng, 'toffoli_per_shot_official': tof_off,
                     'all_zero': all(v == 0 for v in tot.values()) and bool(np.array_equal(w_eng, words_off))
                     and tof_eng == tof_off}
    if not a.no_controls:
        ctl = {}
        o2 = ke.run_sequence([progs[k] for k in order], ke.seed_words(sim_seed ^ 1), x0, xreg, threads=a.threads,
                             dump=True)
        ctl['wrong_seed'] = compare(o2, 'wrong_seed')[1]
        del o2
        # fresh simulator per step: chain the steps through separate calls
        parts = []
        x = x0
        for i, k in enumerate(order):
            o = ke.run_sequence([progs[k]], ke.seed_words(sim_seed), x, xreg, threads=a.threads, dump=True)
            parts.append(o)
            x = o['x_out'][0]
        o3 = {kk: np.concatenate([p[kk] for p in parts]) for kk in ('x_out', 'dirty', 'phase', 'kind_counts',
                                                                    'toffoli_count')}
        nqm = max(p['dump_q'].shape[2] for p in parts)
        nbm = max(p['dump_bits'].shape[2] for p in parts)
        o3['dump_q'] = np.concatenate([np.pad(p['dump_q'], ((0, 0), (0, 0), (0, nqm - p['dump_q'].shape[2]), (0, 0)))
                                       for p in parts])
        o3['dump_bits'] = np.concatenate([np.pad(p['dump_bits'], ((0, 0), (0, 0), (0, nbm - p['dump_bits'].shape[2]),
                                                                  (0, 0))) for p in parts])
        del parts
        ctl['no_rng_continuity'] = compare(o3, 'no_rng_continuity')[1]
        del o3
        rep['negative_controls'] = ctl
        rep['controls_fail_as_required'] = all(any(v > 0 for v in c.values()) for c in ctl.values())
    rep['family_s'] = time.perf_counter() - t_fam
    return rep


report = {'engine_version': ke.__version__, 'engine_file': ke.__file__, 'kernel': ke.kernel_info()['kernel'],
          'kickmix_file': km.__file__, 'verifier_dir': str(off.dir), 'python': sys.version.split()[0],
          'threads': a.threads, 'load_start': os.getloadavg(), 'families': []}
for fam in a.family:
    r = family_report(fam)
    report['families'].append(r)
    print(json.dumps({k: r[k] for k in ('family', 'shots', 'tail_lanes', 'oracle_s', 'engine_s')}),
          json.dumps(r['parity']['mismatch_totals']), 'all_zero', r['parity']['all_zero'],
          'controls', r.get('negative_controls'), flush=True)
report['load_end'] = os.getloadavg()
report['all_zero'] = all(r['parity']['all_zero'] for r in report['families'])
report['controls_fail_as_required'] = all(r.get('controls_fail_as_required', True) for r in report['families'])
with open(a.out, 'w') as f:
    json.dump(report, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))
print('all_zero', report['all_zero'], 'controls_fail_as_required', report['controls_fail_as_required'])
