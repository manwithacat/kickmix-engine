"""reaction_depth parity: the engine's compile-time `Program.reaction_depth` against kickmix's
`Circuit.reaction_depth()` (run with an interpreter where kickmix imports).

Usage: reaction_depth_parity.py OUT.json [--random N] [--kmb PATH ...] [--write-fixture PATH]

- `--random N` random circuits over every op kind the engine supports (bit conditions, nested PUSH/POP_CONDITION,
  INFO_TAG, NEG, top-level and conditioned R/HMR/SWAP/BIT_STORE, Clifford and non-Clifford Z_POW), with varying
  sizes and condition densities. Each is round-tripped through kickmix's .kmb writer and compiled by the engine from
  that file, and also from kickmix's text form.
- `--kmb PATH` real circuits: the engine compiles the file; kickmix loads it and computes `reaction_depth()`.
- `--write-fixture PATH` stores the first 25 random circuits (text, num_qubits, kickmix depth) as JSON for the
  kickmix-free test in tests/test_engine.py.
Exit code 0 only if every comparison is equal. Written by an AI model (Claude) under the owner's direction.
"""
import argparse, hashlib, json, os, sys, tempfile, time

import numpy as np
import kickmix as km

import kickmix_engine as ke

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--random', type=int, default=300)
ap.add_argument('--kmb', action='append', default=[])
ap.add_argument('--write-fixture', default=None)
ap.add_argument('--seed', type=int, default=20261028)
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')

ANGLES = ['1.0', '3.0', '0.5', '1.5', '0.25', '0.75', '0.125', '0.0625', '0.3125']


def random_circuit(rng):
    nq = int(rng.integers(3, 20))
    nb = int(rng.integers(1, 12))
    nops = int(rng.integers(5, 400))
    p_cond = float(rng.choice([0.0, 0.05, 0.2, 0.5]))
    p_push = float(rng.choice([0.0, 0.02, 0.08]))
    lines, depth = [], 0

    def q(k=1):
        return [f'q{i}' for i in rng.choice(nq, size=k, replace=False)]

    def b():
        return f'b{rng.integers(nb)}'

    for _ in range(nops):
        r = rng.random()
        cond = f' if {b()}' if rng.random() < p_cond else ''
        if r < p_push and depth < 4:
            lines.append(f'PUSH_CONDITION if {b()}'); depth += 1; continue
        if r < 2 * p_push and depth > 0:
            lines.append('POP_CONDITION'); depth -= 1; continue
        if r < 2 * p_push + 0.02:
            lines.append(f'INFO_TAG "t{rng.integers(100)}"'); continue
        kind = rng.choice(['X', 'Z', 'CX', 'CX', 'CZ', 'SWAP', 'SWAP', 'CCX', 'CCX', 'CCZ', 'CSWAP', 'HMR', 'HMR', 'R',
                           'BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT', 'NEG', 'Z_POW', 'Z_POW'])
        if kind in ('X', 'Z', 'R'):
            lines.append(f'{kind} {q()[0]}{cond}')
        elif kind in ('CX', 'CZ', 'SWAP'):
            lines.append(f'{kind} {" ".join(q(2))}{cond}')
        elif kind in ('CCX', 'CCZ', 'CSWAP'):
            lines.append(f'{kind} {" ".join(q(3))}{cond}')
        elif kind == 'HMR':
            lines.append(f'HMR {q()[0]} {b()}{cond}')
        elif kind in ('BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'):
            lines.append(f'{kind} {b()}{cond}')
        elif kind == 'NEG':
            lines.append(f'NEG{cond}')
        else:
            lines.append(f'Z_POW {q()[0]} {rng.choice(ANGLES)}{cond}')
    while depth:
        lines.append('POP_CONDITION'); depth -= 1
    lines.append(f'X q{nq - 1}')
    lines.append(f'BIT_INVERT b{nb - 1}')
    return '\n'.join(lines) + '\n'


res = dict(kickmix=km.__file__, engine=ke._engine.__file__, engine_version=ke.engine_version(), random=[], kmb=[])
ok = True
rng = np.random.default_rng(a.seed)
fixture = []
with tempfile.TemporaryDirectory() as td:
    for i in range(a.random):
        text = random_circuit(rng)
        kmx = os.path.join(td, 'c.kmx'); kmb = os.path.join(td, 'c.kmb')
        open(kmx, 'w').write(text)
        c = km.Circuit.from_file(kmx, format='kmx')
        c.to_file(kmb, format='kmb')
        want = int(c.reaction_depth())
        got_kmb = int(ke.compile_kmb(kmb).reaction_depth)
        got_txt = int(ke.compile_circuit(c).reaction_depth)
        eq = want == got_kmb == got_txt
        ok &= eq
        res['random'].append(dict(i=i, ops=len(c), kickmix=want, engine_kmb=got_kmb, engine_text=got_txt, equal=eq))
        if not eq:
            print('MISMATCH', i, want, got_kmb, got_txt, flush=True)
        if len(fixture) < 25:
            fixture.append(dict(text=str(c), num_qubits=c.num_qubits, reaction_depth=want))
depths = [r['kickmix'] for r in res['random']]
res['random_summary'] = dict(n=len(depths), all_equal=all(r['equal'] for r in res['random']),
                             depth_min=min(depths, default=None), depth_max=max(depths, default=None),
                             distinct_depths=len(set(depths)))
print('random', res['random_summary'], flush=True)

for path in a.kmb:
    t0 = time.perf_counter()
    prog = ke.compile_kmb(path)
    t1 = time.perf_counter()
    c = km.Circuit.from_file(path, format='kmb')
    want = int(c.reaction_depth())
    t2 = time.perf_counter()
    eq = want == int(prog.reaction_depth)
    ok &= eq
    rec = dict(path=os.path.basename(path), num_ops=len(c), kickmix=want, engine=int(prog.reaction_depth), equal=eq,
               engine_compile_s=round(t1 - t0, 3), kickmix_load_and_depth_s=round(t2 - t1, 3))
    res['kmb'].append(rec)
    print(rec, flush=True)
    del c, prog

res['pass'] = bool(ok)
json.dump(res, open(a.out, 'w'), indent=1)
if a.write_fixture:
    json.dump(dict(generated_by='tests/reaction_depth_parity.py', kickmix=os.path.basename(km.__file__),
                   circuits=fixture), open(a.write_fixture, 'w'))
    print('fixture', a.write_fixture, hashlib.sha256(open(a.write_fixture, 'rb').read()).hexdigest())
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
