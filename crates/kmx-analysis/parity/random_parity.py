"""Randomized differential test: kickmix_analysis vs the Python reference on synthetic .kmb circuits with nested
condition blocks, R/HMR inside blocks, R_IF/HMR_IF in and out of blocks, registers (some reset mid-circuit, some never
touched) and untouched wire ids.

  python -I random_parity.py SCRIPTS_DIR STAGED_DIR WORKDIR N SEED OUT.json

Compares, per circuit: interval arrays (cond_rule False/True), reset_ref counts, profiles (register_gaps False/True),
qhdl_peak.analyse JSON for both liveness models (reference module from SCRIPTS_DIR vs patched module from STAGED_DIR
with backend='rust'), and the allocator's recoloured streams.
"""
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

scripts, staged, work, N, seed, out = sys.argv[1], sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
sys.path.insert(0, scripts)
import qhdl_kmb as K  # noqa: E402
import qhdl_lifetime_allocator as A  # noqa: E402
import qhdl_peak as P_ref  # noqa: E402
import kickmix_analysis as KA  # noqa: E402

spec = importlib.util.spec_from_file_location('qhdl_peak_staged', str(Path(staged) / 'qhdl_peak.py'))
P_new = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P_new)

ONEQ = [0x11, 0x10, 0x12, 0x50, 0x13]  # X Z R HMR Z_POW
IFQ = [0x92, 0xD0, 0x91]  # R_IF HMR_IF X_IF


def gen(rng, path):
    nq = rng.randint(3, 14)
    nops = rng.randint(1, 400)
    ot, st = [], {n: [] for n in ['qqq0', 'qqq1', 'qqq2', 'qq0', 'qq1', 'q0', 'b0', 'bc']}
    angles = 0
    depth = 0
    tags = []
    pool = list(range(nq))
    for _ in range(nops):
        r = rng.random()
        if r < 0.08:
            ot.append(0x81); st['bc'].append(rng.randrange(4)); depth += 1; continue
        if r < 0.15 and depth > 0:
            ot.append(0x01); depth -= 1; continue
        if r < 0.17:
            ot.append(0x06); tags.append(rng.choice([b'@enter:a', b'@exit:a', b'@enter:b', b'@exit:b', b'note'])); continue
        k = rng.random()
        if k < 0.2:
            q = rng.sample(pool, 3); t = rng.choice([0x31, 0x30, 0xB1])
            ot.append(t); st['qqq0'].append(q[0]); st['qqq1'].append(q[1]); st['qqq2'].append(q[2])
        elif k < 0.45:
            q = rng.sample(pool, 2); t = rng.choice([0x21, 0x20, 0x22, 0xA1])
            ot.append(t); st['qq0'].append(q[0]); st['qq1'].append(q[1])
        else:
            q = rng.choice(pool)
            t = rng.choice(ONEQ + IFQ + [0x12, 0x50, 0x92, 0xD0])
            ot.append(t); st['q0'].append(q)
            if t == 0x13:
                angles += 1
        t = ot[-1]
        if t & 0x40:
            st['b0'].append(rng.randrange(4))
        if t & 0x80:
            st['bc'].append(rng.randrange(4))
    while depth > 0:
        ot.append(0x01); depth -= 1
    regs = []
    cand = list(range(nq + rng.randint(0, 3)))  # ids beyond nq: registers never touched
    rng.shuffle(cand)
    nr = rng.randint(0, min(4, len(cand)))
    if nr:
        regs.append(('x', [(True, i) for i in cand[:nr]]))
    if rng.random() < 0.3 and len(cand) > nr:
        regs.append(('y', [(True, cand[nr]), (False, 0)]))
    K.write_kmb(str(path), np.array(ot, np.uint8), {n: np.array(v, np.uint32) for n, v in st.items()},
                angles=b'\0' * (16 * angles), tags=tags, registers=regs,
                num_qubits=nq + 3, num_bits=4)


def H(a, dt=None):
    a = np.ascontiguousarray(a if dt is None else np.asarray(a).astype(dt))
    return hashlib.sha256(a.tobytes()).hexdigest() + str(a.dtype)


rng = random.Random(seed)
work.mkdir(parents=True, exist_ok=True)
fails, stats = [], dict(circuits=0, cond_resets=0, cond_rule_changed=0, peak_json_checks=0, recolour_checks=0)
for i in range(N):
    p = work / f'r{i}.kmb'
    gen(rng, p)
    k = K.read(str(p))
    nops = len(k.op_types)
    regs = A.register_qubits(k)
    is_reset, depth = A.reset_ops(k)
    ops, qs, order, _ = A.references(k)
    if len(ops) == 0:
        continue
    stats['circuits'] += 1
    rr_counts = []
    for cond in (False, True):
        iv = A.build_intervals(k, ops, qs, order, is_reset, depth, regs, cond_rule=cond)
        lv = KA.liveness(str(p), cond_rule=cond)
        rr_counts.append(int(iv['reset_ref'].sum()))
        for key in ('logical', 's', 'e', 'is_first', 'is_last'):
            if H(iv[key]) != H(lv['intervals'][key]):
                fails.append((i, cond, key))
        if int(iv['reset_ref'].sum()) != lv['reset_refs']:
            fails.append((i, cond, 'reset_refs'))
        for rg in (False, True):
            a = A.live_profile(iv, nops, regs, rg)
            b = KA.live_profile(lv['intervals'], nops, regs, rg)
            if H(a, np.int64) != H(b, np.int64):
                fails.append((i, cond, 'profile', rg))
        if not cond:
            col = A.colour(iv, nops, regs, False)
            s_ref, _ = A.recolour(k, ops, qs, order, iv, col)
            s_new = KA.recolour(str(p), lv['intervals']['logical'], col)
            stats['recolour_checks'] += 1
            if any(H(s_ref[n], np.uint32) != H(s_new[n], np.uint32) for n in s_ref):
                fails.append((i, 'recolour'))
    stats['cond_resets'] += int(lv['conditional_resets'])
    stats['cond_rule_changed'] += int(rr_counts[1] != rr_counts[0])
    for liv in ('allocator', 'reset'):
        for lvl in (None, 'peak-1'):
            try:
                ra = json.dumps(P_ref.analyse(str(p), lvl, liveness=liv))
            except Exception as e:  # noqa: BLE001
                ra = 'EXC ' + repr(e)
            try:
                rb = json.dumps(P_new.analyse(str(p), lvl, liveness=liv, backend='rust'))
            except Exception as e:  # noqa: BLE001
                rb = 'EXC ' + repr(e)
            stats['peak_json_checks'] += 1
            if ra != rb:
                fails.append((i, 'peak', liv, lvl, ra[:200], rb[:200]))
res = dict(seed=seed, n=N, stats=stats, failures=fails[:50], n_failures=len(fails))
Path(out).write_text(json.dumps(res, indent=1))
print(json.dumps(dict(stats=stats, n_failures=len(fails), first=fails[:3])))
