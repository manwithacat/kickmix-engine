"""FAIL-path cross-checks: small mutated circuits that make the official verifiers fail (or count errors), each run
through `kickmix-engine verify --cross-check`, which runs the official verifier CLI as well and compares every
output field (status and error message on FAIL, the whole summary on PASS).

Usage: fail_paths_crosscheck.py OUT.json --scratch DIR [--official-python PY] [--official-workers N] [--threads N]
                                [--only CASE,CASE] [--keep]
Needs: KICKMIX_ENGINE_VERIFIERS (the competition's verifiers/ directory, with dqi/ and invsqrt/); an interpreter where
both kickmix and kickmix_engine import (this one); --official-python (default: this interpreter) runs the official
verifier. The base circuits are built with the official reference builders in that directory:
`dqi/dqi_reference_circuit.py` at a small instance (m=30, n=4, degree=5) and `invsqrt/nr_reference_circuit.py`
(n_p=10, the competition instance). Mutations append a few ops at the end of the circuit (kickmix text, rewritten as
.kmb by kickmix). The circuits are written under --scratch and deleted afterwards unless --keep.

Cases (expected official outcome in brackets):
  dqi_reference              unmodified small reference decoder [PASS, 0 errors]
  dqi_ancilla_all            X on one ancilla qubit [FAIL: every shot a value error]
  dqi_ancilla_cond           CX from a syndrome qubit onto an ancilla [FAIL: value errors on part of the shots]
  dqi_ancilla_rare           X on an ancilla under 5 nested PUSH_CONDITIONs on MBUC bits [PASS with ~1/32 value errors]
  dqi_syndrome_not_restored  X on a syndrome qubit [FAIL]
  inv_x_not_restored         X on x[0] [FAIL: 'x was not restored.']
  inv_x_not_restored_late    CX x[20] -> x[0]: x changes only for x >= 2^20, first in a late chunk [FAIL]
  inv_scratch_dirty          X on the last scratch qubit [FAIL: 'Scratch qubit q.. did not return to |0>.']
  inv_scratch_order          X on a high scratch qubit plus CX x[21] -> a lower scratch qubit; the first failing batch
                             has only the high one dirty, so it must be the one named [FAIL]
  inv_first_failure_wins     scratch dirty for x with bit 20 set, x not restored for x >= 2^21: the scratch failure
                             comes first in task order, so its message wins over the x check [FAIL]
Exit code 0 only if every case matches. Written by an AI model (Claude) under the owner's direction.
"""
import argparse, hashlib, importlib.util, json, os, shutil, subprocess, sys, time

import kickmix as km

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--scratch', required=True)
ap.add_argument('--official-python', default=sys.executable)
ap.add_argument('--official-workers', type=int, default=None)
ap.add_argument('--threads', type=int, default=os.cpu_count() or 1)
ap.add_argument('--only', default=None)
ap.add_argument('--keep', action='store_true')
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')
VD = os.environ.get('KICKMIX_ENGINE_VERIFIERS')
if not VD or not os.path.isdir(VD):
    sys.exit('set KICKMIX_ENGINE_VERIFIERS to the competition verifiers/ directory')
os.makedirs(a.scratch, exist_ok=True)
DQI_PARAMS = dict(m=30, n=4, degree=5)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def qid(e):
    return int(str(e)[1:])


def write_kmb(circuit_text, path):
    kmx = path[:-4] + '.kmx'
    with open(kmx, 'w') as f:
        f.write(circuit_text)
    km.Circuit.from_file(kmx, format='kmx').to_file(path, format='kmb')
    os.unlink(kmx)
    return path


base = {}
cases = {}
t0 = time.perf_counter()
# DQI: small reference decoder.
vd = load(os.path.join(VD, 'dqi', 'verify_submission_dqi.py'), 'vdqi_fp')
rd = load(os.path.join(VD, 'dqi', 'dqi_reference_circuit.py'), 'dqi_ref_fp')
c = rd.build_reference_circuit(**DQI_PARAMS)
spec = vd.ProblemSpecDQI.benchmark_instance(**DQI_PARAMS)
syn, mbuc, anc = vd.validate_registers(c, spec)
text = str(c).rstrip('\n') + '\n'
anc_ids, syn_ids, e_ids = [qid(x) for x in anc], [qid(x) for x in syn], [int(str(x)[1:]) for x in mbuc]
base['dqi'] = dict(num_qubits=c.num_qubits, num_ops=len(c), ancillas=len(anc_ids), syndrome_qubits=len(syn_ids))
dqi_flags = ['--m', str(DQI_PARAMS['m']), '--n', str(DQI_PARAMS['n']), '--degree', str(DQI_PARAMS['degree'])]
a_mid = anc_ids[len(anc_ids) // 2]
push = ''.join(f'PUSH_CONDITION if b{e}\n' for e in e_ids[:5])
pop = 'POP_CONDITION\n' * 5
cases['dqi_reference'] = ('dqi', text, dqi_flags)
cases['dqi_ancilla_all'] = ('dqi', text + f'X q{a_mid}\n', dqi_flags)
cases['dqi_ancilla_cond'] = ('dqi', text + f'CX q{syn_ids[0]} q{a_mid}\n', dqi_flags)
cases['dqi_ancilla_rare'] = ('dqi', text + push + f'X q{a_mid}\n' + pop, dqi_flags)
cases['dqi_syndrome_not_restored'] = ('dqi', text + f'X q{syn_ids[0]}\n', dqi_flags)
del c, text
# invsqrt: the official Newton-Raphson reference circuit at the competition size.
vi = load(os.path.join(VD, 'invsqrt', 'verify_submission_invsqrt.py'), 'vinv_fp')
nr = load(os.path.join(VD, 'invsqrt', 'nr_reference_circuit.py'), 'nr_ref_fp')
c = nr.build_nr_circuit(vi.N_P)
ispec = vi.ProblemSpecInvSqrt.for_n_p(vi.N_P)
x_reg, scratch = vi.validate_registers(c, ispec)
x_ids, s_ids = [qid(x) for x in x_reg], [qid(x) for x in scratch]
text = str(c).rstrip('\n') + '\n'
base['invsqrt'] = dict(num_qubits=c.num_qubits, num_ops=len(c), scratch=len(s_ids))
cases['inv_x_not_restored'] = ('invsqrt', text + f'X q{x_ids[0]}\n', [])
cases['inv_x_not_restored_late'] = ('invsqrt', text + f'CX q{x_ids[20]} q{x_ids[0]}\n', [])
cases['inv_scratch_dirty'] = ('invsqrt', text + f'X q{s_ids[-1]}\n', [])
cases['inv_scratch_order'] = ('invsqrt', text + f'X q{s_ids[-1]}\nCX q{x_ids[21]} q{s_ids[len(s_ids) // 3]}\n', [])
cases['inv_first_failure_wins'] = ('invsqrt', text + f'CX q{x_ids[20]} q{s_ids[len(s_ids) // 2]}\n'
                                   f'CX q{x_ids[21]} q{x_ids[0]}\n', [])
del c, text

only = set(a.only.split(',')) if a.only else None
res = dict(verifiers=VD, official_python=a.official_python, engine_python=sys.executable, kickmix=km.__file__,
           base=base, cases={}, build_s=round(time.perf_counter() - t0, 2))
ok = True
cache = os.path.join(a.scratch, 'cache')
for name, (task, text, flags) in cases.items():
    if only and name not in only:
        continue
    path = write_kmb(text, os.path.join(a.scratch, name + '.kmb'))
    sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()
    cmd = [sys.executable, '-m', 'kickmix_engine', 'verify', task, path, '--cross-check', '--threads', str(a.threads),
           '--official-python', a.official_python, '--cache-dir', cache] + flags
    if a.official_workers:
        cmd += ['--official-workers', str(a.official_workers)]
    t1 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    try:
        r = json.loads(p.stdout)
    except ValueError:
        r = dict(match=False, parse_error=p.stdout[-2000:], stderr=p.stderr[-2000:])
    eng, off = r.get('engine', {}), r.get('official', {})
    rec = dict(task=task, circuit_sha256=sha, size_bytes=os.path.getsize(path), match=bool(r.get('match')),
               differences=r.get('differences'), returncode=p.returncode, official_returncode=r.get('official_returncode'),
               engine_status=eng.get('status'), official_status=off.get('status'),
               engine_error=eng.get('error'), official_error=off.get('error'),
               value_errors=(eng.get('value_errors'), off.get('value_errors')),
               phase_errors=(eng.get('phase_errors'), off.get('phase_errors')),
               reaction_depth=(eng.get('reaction_depth'), off.get('reaction_depth')),
               seconds=r.get('seconds'), wall_s=round(time.perf_counter() - t1, 2), engine=eng, official=off)
    ok &= rec['match']
    res['cases'][name] = rec
    print(name, {k: rec[k] for k in ('match', 'engine_status', 'official_status', 'engine_error', 'official_error',
                                     'value_errors', 'phase_errors', 'wall_s')}, flush=True)
    if not a.keep:
        os.unlink(path)
if not a.keep:
    shutil.rmtree(cache, ignore_errors=True)
res['pass'] = bool(ok)
json.dump(res, open(a.out, 'w'), indent=1, default=str)
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
