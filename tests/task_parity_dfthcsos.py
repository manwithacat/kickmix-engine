"""0.5.0 gate 2: dfthcsos task parity, `kickmix-engine verify dfthcsos --cross-check` on real and mutated families.

Usage: task_parity_dfthcsos.py OUT.json --verifier DFTHCSOS_DIR --scratch DIR --cache-dir DIR
       [--family DIR ...] [--mutate-from DIR] [--mutate-sample DIR] [--wrong-npz-from DIR]
       [--salts ,s1] [--threads N] [--official-python PY]

Each case runs the CLI with --cross-check: the engine adapter in this interpreter, then the official
`dfthcsos/verify.py` CLI unchanged in --official-python (default: this interpreter), and compares every summary field
on PASS and the status and error text on FAIL. A case passes when `match` is true.

Mutants (built in --scratch with kickmix from the circuit text; each changes one circuit of a copy of the family):
  ws_dirty        X on a workspace qubit at the end of ANGLE_3             (workspace check)
  sel_changed     X on selection bit 0 at the end of ANGLE_5              (selection check)
  phase_kick      Z on word_0 bit 0 at the end of ANGLE_4                  (phase check)
  dirty_word      HMR of a clean workspace qubit into a new bit, then X on word_1 bit 2 if that bit, in ANGLE_6
                  (a word that depends on a measurement: repeated shots disagree)
  word_msb_flip   X on the top bit of word_0 at the end of ANGLE_7         (consistent wrong words: error gates)
  cx_in_words     CX word_0 bit 0 -> word_0 bit 1 at the start of ANGLE_2  (reverse restore check)
  z_pow           Z_POW q0 0.25 at the end of ANGLE_1                      (rotation rejection)
  missing_last    the last circuit removed                                  (circuit count)
  wrong_npz       the family's circuits with another family's submission.npz
  w_scaled        submission.npz with the weights W scaled by 1.05          (error_2 over the 20 mHa gate)
  sample_padding  (from --mutate-sample) X on the last word register's bit 0 at the end of the last circuit
                  (padding words must be zero)
Written by an AI model (Claude) under the owner's direction.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

import kickmix as km
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--verifier', required=True)
ap.add_argument('--scratch', required=True)
ap.add_argument('--cache-dir', required=True)
ap.add_argument('--family', action='append', default=[])
ap.add_argument('--mutate-from', default=None)
ap.add_argument('--mutate-sample', default=None)
ap.add_argument('--wrong-npz-from', default=None)
ap.add_argument('--salts', default='')
ap.add_argument('--shots-per-address', type=int, default=None)
ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
ap.add_argument('--official-python', default=None)
ap.add_argument('--only', default=None, help='comma-separated case names to run (default: all)')
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')
os.makedirs(a.scratch, exist_ok=True)


def copy_family(src, name):
    dst = os.path.join(a.scratch, name)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    for f in os.listdir(src):
        if f.startswith('angle_k_') or f == 'submission.npz':
            shutil.copy(os.path.join(src, f), dst)
    return dst


def k_count(d):
    k = 0
    while os.path.exists(os.path.join(d, f'angle_k_{k}.kmb')):
        k += 1
    return k


def rewrite(d, k, edit):
    p = os.path.join(d, f'angle_k_{k}.kmb')
    c = km.Circuit.from_file(p)
    lines = str(c).splitlines()
    head = [l for l in lines if l.startswith(('APPEND_TO_REGISTER', 'REGISTER'))]
    body = [l for l in lines if not l.startswith(('APPEND_TO_REGISTER', 'REGISTER'))]
    regs = c.register_data
    body = edit(body, c, regs)
    nc = km.Circuit('\n'.join(head + body) + '\n')
    nc.to_file(p, format='kmb')


def qid(regs, name, i):
    return regs[name][i].id


def workspace(c, regs):
    public = {q.id for n, r in regs.items() if n == 'selection' or n.startswith('word_') for q in r}
    return [i for i in range(c.num_qubits) if i not in public]


def mutants(src):
    out = []
    d = copy_family(src, 'ws_dirty')
    rewrite(d, 3, lambda b, c, r: b + [f'X q{workspace(c, r)[0]}'])
    out.append(('ws_dirty', d))
    d = copy_family(src, 'sel_changed')
    rewrite(d, 5, lambda b, c, r: b + [f'X q{qid(r, "selection", 0)}'])
    out.append(('sel_changed', d))
    d = copy_family(src, 'phase_kick')
    rewrite(d, 4, lambda b, c, r: b + [f'Z q{qid(r, "word_0", 0)}'])
    out.append(('phase_kick', d))
    d = copy_family(src, 'dirty_word')
    rewrite(d, 6, lambda b, c, r: b + [f'HMR q{workspace(c, r)[0]} b{c.num_bits}',
                                       f'X q{qid(r, "word_1", 2)} if b{c.num_bits}'])
    out.append(('dirty_word', d))
    d = copy_family(src, 'word_msb_flip')
    rewrite(d, 7, lambda b, c, r: b + [f'X q{r["word_0"][-1].id}'])
    out.append(('word_msb_flip', d))
    d = copy_family(src, 'cx_in_words')
    rewrite(d, 2, lambda b, c, r: [f'CX q{qid(r, "word_0", 0)} q{qid(r, "word_0", 1)}'] + b)
    out.append(('cx_in_words', d))
    d = copy_family(src, 'z_pow')
    rewrite(d, 1, lambda b, c, r: b + ['Z_POW q0 0.25'])
    out.append(('z_pow', d))
    d = copy_family(src, 'missing_last')
    os.remove(os.path.join(d, f'angle_k_{k_count(d) - 1}.kmb'))
    out.append(('missing_last', d))
    d = copy_family(src, 'w_scaled')
    p = os.path.join(d, 'submission.npz')
    with np.load(p, allow_pickle=False) as z:
        arrs = {k: np.array(z[k]) for k in z.files}
    arrs['W'] = arrs['W'] * 1.05
    np.savez(p, **arrs)
    out.append(('w_scaled', d))
    if a.wrong_npz_from:
        d = copy_family(src, 'wrong_npz')
        shutil.copy(os.path.join(a.wrong_npz_from, 'submission.npz'), os.path.join(d, 'submission.npz'))
        out.append(('wrong_npz', d))
    return out


cases = [(os.path.basename(os.path.normpath(f)), f) for f in a.family]
if a.mutate_from:
    cases += mutants(a.mutate_from)
if a.mutate_sample:
    d = copy_family(a.mutate_sample, 'sample_padding')
    K = k_count(d)

    def pad(b, c, r):
        J = len([n for n in r if n.startswith('word_')])
        return b + [f'X q{qid(r, f"word_{J - 1}", 0)}']
    rewrite(d, K - 1, pad)
    cases.append(('sample_padding', d))

if a.only:
    keep = set(a.only.split(','))
    cases = [c for c in cases if c[0] in keep]
results = []
salts = a.salts.split(',')
for name, d in cases:
    for salt in salts:
        cmd = [sys.executable, '-m', 'kickmix_engine', 'verify', 'dfthcsos', d, '--verifier', a.verifier,
               '--cross-check', '--threads', str(a.threads), '--cache-dir', a.cache_dir, '--salt', salt, '--timing']
        if a.shots_per_address:
            cmd += ['--shots-per-address', str(a.shots_per_address)]
        if a.official_python:
            cmd += ['--official-python', a.official_python]
        t = time.perf_counter()
        p = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.perf_counter() - t
        try:
            res = json.loads(p.stdout)
        except ValueError:
            res = {'match': False, 'parse_error': True, 'stdout_tail': p.stdout[-2000:]}
        rec = {'case': name, 'dir': d, 'salt': salt, 'returncode': p.returncode, 'wall_s': wall,
               'match': res.get('match'), 'engine_status': res.get('engine', {}).get('status'),
               'official_status': res.get('official', {}).get('status'),
               'engine_error': res.get('engine', {}).get('error'), 'official_error': res.get('official', {}).get('error'),
               'differences': res.get('differences'), 'seconds': res.get('seconds'),
               'engine_info': p.stderr[-4000:], 'result': res}
        results.append(rec)
        print(json.dumps({k: rec[k] for k in ('case', 'salt', 'match', 'engine_status', 'official_status',
                                              'engine_error', 'seconds')}), flush=True)

report = {'python': sys.executable, 'verifier': a.verifier, 'load_end': os.getloadavg(), 'cases': results,
          'all_match': all(r['match'] for r in results), 'n': len(results)}
with open(a.out, 'w') as f:
    json.dump(report, f, indent=1, default=str)
print('all_match', report['all_match'], 'n', len(results))
