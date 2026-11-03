#!/usr/bin/env python3
"""Release check for kickmix-engine: one entry point that builds the release artifact and runs every gate on it.

Written by an AI model (Claude) under the owner's direction.

    python tools/release_check.py --out-dir NEW_DIR \
        --verifiers /path/to/verifiers --kickmix-python /path/to/venv-with-kickmix/bin/python \
        --invsqrt INV.kmb --dqi DQI.kmb --dfthcsos SUBMISSION_DIR \
        [--rank-json STORED_RANK.json ...] [--qrom-words W.npy --qrom-plans DIR] \
        [--threads 4] [--only NAME,NAME] [--skip NAME,NAME] [--maturin PATH] [--python-for-venv 3.12]

What it runs (in this order; each check is PASS, FAIL or SKIP with the reason; logs go to NEW_DIR/logs):

  versions            Cargo.toml, pyproject.toml and the first CHANGELOG release heading agree
  cargo-test          `cargo test --workspace --locked` (qrom-plan's real-data test too with --qrom-words/--qrom-plans)
  wheel-build         `maturin build --release --locked`; the wheel is unpacked to NEW_DIR/site and every later
                      check imports kickmix_engine from there (the release artifact, not the source tree)
  wheel-smoke         fresh venv (uv, else python -m venv): install the wheel, import, __version__, CLI --help,
                      `doctor --json`, and pytest on tests/ without kickmix (what CI runs)
  pytest-kickmix      pytest on tests/ in --kickmix-python with KICKMIX_ENGINE_VERIFIERS set (kickmix-gated tests run)
  parity-invsqrt      tests/parity.py on --invsqrt (per-shot, plus the wrong-seed control)
  parity-dqi          tests/parity_dqi.py on --dqi (per-shot, plus wrong-seed and CX->CZ controls)
  parity-dfthcsos     tests/parity_dfthcsos.py on --dfthcsos (per step and shot, plus controls)
  reaction-depth      tests/reaction_depth_parity.py: 300 random circuits plus --invsqrt and --dqi
  fail-paths          tests/fail_paths_crosscheck.py (FAIL-path messages against the official CLI)
  crosscheck-invsqrt  `kickmix-engine verify invsqrt --cross-check`: every summary field equal to the official CLI's
  crosscheck-dqi      the same for DQI
  crosscheck-dfthcsos the same for the chemistry family
  profile-invsqrt     `profile invsqrt` at the official seed reproduces the official summary of crosscheck-invsqrt
  diff-dqi-rank       `diff dqi --profile --n 16` against each stored `dqi_fast_rank.py` result (--rank-json):
                      delta_t, its SE, predicted cost, precondition verdict and every paired Toffoli total equal
  diff-broken         `diff invsqrt` of --invsqrt against itself (agreement 1, delta 0) and against a copy with one
                      CX turned into a CZ (structural hunk found, semantic disagreement and candidate errors > 0)

Exit code 0 only if no check FAILed. A check whose inputs are missing is SKIPped and named in the table; a release
should not be cut with skips unless the reason is understood. NEW_DIR must not exist (evidence is never overwritten).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TESTS = os.path.join(ROOT, 'tests')


class Skip(Exception):
    pass


class Fail(Exception):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


class Ctx:
    def __init__(self, a):
        self.a = a
        self.out = os.path.abspath(a.out_dir)
        self.logs = os.path.join(self.out, 'logs')
        self.work = os.path.join(self.out, 'work')
        self.site = os.path.join(self.out, 'site')
        self.cache = os.path.join(self.out, 'engine-cache')
        self.wheel = None
        self.version = None
        self.results = {}       # name -> dict(status, seconds, detail)
        self.data = {}          # values passed between checks

    def env(self, kickmix_site=True):
        e = dict(os.environ)
        for k in ('VIRTUAL_ENV', 'PYTHONHOME'):
            e.pop(k, None)
        if kickmix_site:
            e['PYTHONPATH'] = self.site
        else:
            e.pop('PYTHONPATH', None)
        e['RAYON_NUM_THREADS'] = str(self.a.threads)
        e['CARGO_BUILD_JOBS'] = str(self.a.threads)
        e['KICKMIX_ENGINE_CACHE'] = self.cache
        if self.a.verifiers:
            e['KICKMIX_ENGINE_VERIFIERS'] = self.a.verifiers
        return e

    def run(self, name, cmd, *, env=None, cwd=None, ok_codes=(0,)):
        """Run a command, tee its output to logs/NAME.log; return (returncode, stdout)."""
        log = os.path.join(self.logs, f'{name}.log')
        with open(log, 'a') as f:
            f.write(f'$ {" ".join(map(str, cmd))}\n')
            f.flush()
            p = subprocess.run([str(c) for c in cmd], cwd=cwd or self.work, env=env or self.env(),
                               stdout=subprocess.PIPE, stderr=f, text=True)
            f.write(p.stdout)
            f.write(f'\n[exit {p.returncode}]\n')
        if p.returncode not in ok_codes:
            raise Fail(f'{os.path.basename(str(cmd[1] if len(cmd) > 1 else cmd[0]))} exited {p.returncode} '
                       f'(see logs/{name}.log)')
        return p.returncode, p.stdout

    def ke(self, name, args, **kw):
        return self.run(name, [self.a.kickmix_python, '-m', 'kickmix_engine'] + list(args), **kw)

    def need(self, *names):
        for n in names:
            v = getattr(self.a, n)
            if not v or not os.path.exists(v):
                raise Skip(f'--{n.replace("_", "-")} not given or missing')

    def need_wheel(self):
        if not self.wheel:
            raise Skip('no wheel (wheel-build did not pass)')


def last_json(stdout):
    """The last top-level JSON object printed (the CLI prints one indented object)."""
    i = stdout.find('{')
    if i < 0:
        raise Fail('no JSON in output')
    return json.loads(stdout[i:])


# ---------------------------------------------------------------- checks

def check_versions(c: Ctx):
    cargo = open(os.path.join(ROOT, 'Cargo.toml')).read()
    py = open(os.path.join(ROOT, 'pyproject.toml')).read()
    v_cargo = re.search(r'^version = "([^"]+)"', cargo, re.M).group(1)
    v_py = re.search(r'^version = "([^"]+)"', py, re.M).group(1)
    ch = open(os.path.join(ROOT, 'CHANGELOG.md')).read()
    m = re.search(r'^## (\S+)', ch, re.M)
    v_ch = m.group(1) if m else None
    lock = open(os.path.join(ROOT, 'Cargo.lock')).read()
    m = re.search(r'name = "kickmix-engine"\nversion = "([^"]+)"', lock)
    v_lock = m.group(1) if m else None
    vs = dict(cargo=v_cargo, pyproject=v_py, changelog=v_ch, cargo_lock=v_lock)
    c.version = v_cargo
    if len(set(vs.values())) != 1:
        raise Fail(f'versions differ: {vs}')
    return f'{v_cargo} (Cargo.toml, Cargo.lock, pyproject.toml, CHANGELOG)'


def check_cargo_test(c: Ctx):
    env = c.env(kickmix_site=False)
    real = ''
    if c.a.qrom_words and c.a.qrom_plans:
        env['QROM_PLAN_WORDS'] = c.a.qrom_words
        env['QROM_PLAN_PLANS'] = c.a.qrom_plans
        real = ' incl. qrom-plan real-data parity'
    _, out = c.run('cargo-test', ['cargo', 'test', '--workspace', '--locked', '-j', str(c.a.threads)], env=env,
                   cwd=ROOT)
    log = open(os.path.join(c.logs, 'cargo-test.log')).read()
    passed = sum(int(x) for x in re.findall(r'test result: ok\. (\d+) passed', log))
    if 'test result: FAILED' in log:
        raise Fail('a cargo test failed')
    if real and 'skipping' in log and 'QROM_PLAN_WORDS' in log:
        raise Fail('qrom-plan real-data test skipped although inputs were given')
    return f'{passed} Rust tests passed{real}'


def check_wheel_build(c: Ctx):
    maturin = c.a.maturin or shutil.which('maturin')
    if not maturin:
        raise Skip('maturin not found (pass --maturin)')
    dist = os.path.join(c.out, 'dist')
    c.run('wheel-build', [maturin, 'build', '--release', '--locked', '--out', dist, '-j', str(c.a.threads)],
          env=c.env(kickmix_site=False), cwd=ROOT)
    wheels = glob.glob(os.path.join(dist, '*.whl'))
    if len(wheels) != 1:
        raise Fail(f'expected one wheel, found {wheels}')
    c.wheel = wheels[0]
    name = os.path.basename(c.wheel)
    if c.version and f'-{c.version}-' not in name:
        raise Fail(f'wheel {name} does not carry version {c.version}')
    with zipfile.ZipFile(c.wheel) as z:
        z.extractall(c.site)
    return f'{name} sha256 {sha256(c.wheel)[:16]}...'


def check_wheel_smoke(c: Ctx):
    c.need_wheel()
    venv = os.path.join(c.out, 'smoke-venv')
    env = c.env(kickmix_site=False)
    uv = shutil.which('uv')
    if uv:
        c.run('wheel-smoke', [uv, 'venv', '--python', c.a.python_for_venv, venv], env=env)
        py = os.path.join(venv, 'bin', 'python')
        c.run('wheel-smoke', [uv, 'pip', 'install', '--python', py, c.wheel, 'pytest'], env=env)
    else:
        c.run('wheel-smoke', [sys.executable, '-m', 'venv', venv], env=env)
        py = os.path.join(venv, 'bin', 'python')
        c.run('wheel-smoke', [py, '-m', 'pip', 'install', c.wheel, 'pytest'], env=env)
    env.pop('KICKMIX_ENGINE_VERIFIERS', None)
    _, out = c.run('wheel-smoke', [py, '-I', '-c', 'import kickmix_engine as k, json; '
                                   'print(json.dumps({"version": k.__version__, "file": k.__file__}))'], env=env)
    info = json.loads(out.strip().splitlines()[-1])
    if c.version and info['version'] != c.version:
        raise Fail(f'installed __version__ {info["version"]} != {c.version}')
    if not info['file'].startswith(venv):
        raise Fail(f'imported from {info["file"]}, not the fresh venv')
    cli = os.path.join(venv, 'bin', 'kickmix-engine')
    for args in (['--help'], ['diff', '--help'], ['profile', '--help'], ['verify', '--help']):
        c.run('wheel-smoke', [cli] + args, env=env)
    _, out = c.run('wheel-smoke', [cli, 'doctor', '--json'], env=env, ok_codes=(0, 1))
    doc = last_json(out)
    _, out = c.run('wheel-smoke', [py, '-m', 'pytest', '-q', '-p', 'no:cacheprovider', TESTS], env=env)
    tail = [l for l in out.strip().splitlines() if l.strip()][-1]
    if 'failed' in tail or 'error' in tail:
        raise Fail(f'pytest: {tail}')
    kern = (doc.get('kernels') or {}).get('selected')
    if not kern:
        raise Fail(f'doctor selected no kernel: {doc.get("kernels")}')
    return f'{info["version"]} imports, CLI and doctor run (kernel {kern}); pytest without kickmix: {tail.strip("= ")}'


def check_pytest_kickmix(c: Ctx):
    c.need_wheel()
    c.need('kickmix_python')
    _, out = c.run('pytest-kickmix', [c.a.kickmix_python, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider',
                                      TESTS])
    lines = [l for l in out.strip().splitlines() if l.strip()]
    tail = lines[-1]
    if 'failed' in tail or 'error' in tail:
        raise Fail(f'pytest: {tail}')
    return f'with kickmix and verifiers: {tail.strip("= ")}'


def _json_report(c, name):
    return os.path.join(c.out, 'reports', f'{name}.json')


def check_parity_invsqrt(c: Ctx):
    c.need_wheel(); c.need('invsqrt', 'verifiers')
    out = _json_report(c, 'parity-invsqrt')
    c.run('parity-invsqrt', [c.a.kickmix_python, os.path.join(TESTS, 'parity.py'), out, '--kmb', c.a.invsqrt])
    s = json.load(open(out))['summary']
    if not (s['parity_pass'] and s['control_mismatched']):
        raise Fail(f'summary {s}')
    return f'{s["total_compared_shots"]:,} shots, 0 mismatches; wrong-seed control mismatches'


def check_parity_dqi(c: Ctx):
    c.need_wheel(); c.need('dqi', 'verifiers')
    out = _json_report(c, 'parity-dqi')
    scratch = os.path.join(c.work, 'parity-dqi-scratch')
    try:
        c.run('parity-dqi', [c.a.kickmix_python, os.path.join(TESTS, 'parity_dqi.py'), out, '--kmb', c.a.dqi,
                             '--scratch', scratch])
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    s = json.load(open(out))['summary']
    if not (s['parity_pass'] and s['controls_mismatched']):
        raise Fail(f'summary {s}')
    return f'{s["compared_shots"]:,} shots, 0 mismatches; wrong-seed and CX->CZ controls mismatch'


def check_parity_dfthcsos(c: Ctx):
    c.need_wheel(); c.need('dfthcsos', 'verifiers')
    out = _json_report(c, 'parity-dfthcsos')
    c.run('parity-dfthcsos', [c.a.kickmix_python, os.path.join(TESTS, 'parity_dfthcsos.py'), out,
                              '--family', c.a.dfthcsos, '--threads', c.a.threads,
                              '--verifier', os.path.join(c.a.verifiers, 'dfthcsos')])
    r = json.load(open(out))
    if not (r['all_zero'] and r['controls_fail_as_required']):
        raise Fail(f'all_zero {r["all_zero"]} controls {r["controls_fail_as_required"]}')
    f = r['families'][0]
    return f'{f.get("shots")} shots per step, every step equal; both controls mismatch'


def check_reaction_depth(c: Ctx):
    c.need_wheel(); c.need('invsqrt', 'dqi')
    out = _json_report(c, 'reaction-depth')
    _, so = c.run('reaction-depth', [c.a.kickmix_python, os.path.join(TESTS, 'reaction_depth_parity.py'), out,
                                     '--random', '300', '--kmb', c.a.invsqrt, '--kmb', c.a.dqi])
    return '300 random circuits + 2 frontier circuits equal to kickmix'


def check_fail_paths(c: Ctx):
    c.need_wheel(); c.need('verifiers')
    out = _json_report(c, 'fail-paths')
    scratch = os.path.join(c.work, 'fail-paths-scratch')
    c.run('fail-paths', [c.a.kickmix_python, os.path.join(TESTS, 'fail_paths_crosscheck.py'), out,
                         '--scratch', scratch, '--official-workers', c.a.threads, '--threads', c.a.threads])
    r = json.load(open(out))
    cases = r.get('cases') or {}
    if not r.get('pass') or not cases or not all(x.get('match') for x in cases.values()):
        raise Fail(f'cases not matching: {[k for k, x in cases.items() if not x.get("match")]}')
    return f'{len(cases)} FAIL/PASS-path cases match the official CLI (status, message, error counts)'


def _crosscheck(c: Ctx, task, target):
    c.need_wheel(); c.need('verifiers')
    if not target or not os.path.exists(target):
        raise Skip(f'--{task} not given or missing')
    name = f'crosscheck-{task}'
    args = ['verify', task, target, '--cross-check', '--threads', c.a.threads, '--official-python',
            c.a.kickmix_python]
    if task != 'dfthcsos':
        args += ['--official-workers', c.a.threads]
    rc, out = c.ke(name, args, ok_codes=(0, 1, 3))
    res = last_json(out)
    json.dump(res, open(_json_report(c, name), 'w'), indent=1)
    c.data[name] = res
    if rc != 0 or not res.get('match'):
        raise Fail(f'exit {rc}; differences {json.dumps(res.get("differences"))[:300]}')
    if res['engine'].get('status', 'PASS') != 'PASS' and 'score' not in res['engine']:
        raise Fail(f'engine and official agree but did not PASS: {res["engine"]}')
    sec = res['seconds']
    return (f'every field equal; score {res["engine"].get("score")}; engine {sec["engine"]:.1f} s, '
            f'official {sec["official"]:.1f} s ({c.a.threads} threads/workers)')


def check_crosscheck_invsqrt(c):
    return _crosscheck(c, 'invsqrt', c.a.invsqrt)


def check_crosscheck_dqi(c):
    return _crosscheck(c, 'dqi', c.a.dqi)


def check_crosscheck_dfthcsos(c):
    return _crosscheck(c, 'dfthcsos', c.a.dfthcsos)


def check_profile_invsqrt(c: Ctx):
    c.need_wheel(); c.need('invsqrt', 'verifiers')
    off = (c.data.get('crosscheck-invsqrt') or {}).get('official')
    if not off or 'score' not in off:
        raise Skip('needs a passing crosscheck-invsqrt (the official summary)')
    pdir = os.path.join(c.work, 'profile-invsqrt')
    os.makedirs(pdir)
    _, out = c.ke('profile-invsqrt', ['profile', 'invsqrt', c.a.invsqrt, '--out', pdir, '--draws', '1',
                                      '--salt-format', '', '--threads', c.a.threads])
    prof = json.load(open(last_json(out)['out']))
    d = prof['draws'][0]
    pairs = {'score': ('score', off['score']), 'toffoli_mean': ('toffoli_count', off['toffoli_count']),
             't_count': ('t_count', off['t_count']), 'rotation_count': ('rotation_count', off['rotation_count']),
             'rotation_eps': ('rotation_eps', off['rotation_eps']),
             'worst': ('worst_error_over_tolerance', off['worst_error_over_tolerance']),
             'worst_x': ('worst_x', off['worst_x'])}
    bad = {k: (d.get(k), v) for k, (_, v) in pairs.items() if d.get(k) != v}
    if prof['Q'] != off['num_qubits']:
        bad['Q'] = (prof['Q'], off['num_qubits'])
    if bad:
        raise Fail(f'plug-in vs official: {bad}')
    return f'plug-in profile equals the official summary on {len(pairs) + 1} fields ({prof["seconds"]} s)'


def check_diff_dqi_rank(c: Ctx):
    c.need_wheel(); c.need('verifiers')
    if not c.a.rank_json:
        raise Skip('no --rank-json given')
    done = []
    for i, rj in enumerate(c.a.rank_json):
        ref = json.load(open(rj))
        base, cand, prof = ref['base']['path'], ref['candidate']['path'], ref['base']['profile']
        for p, h in ((base, ref['base']['sha256']), (cand, ref['candidate']['sha256']),
                     (prof, ref['base']['profile_sha256'])):
            if not os.path.exists(p):
                raise Skip(f'{p} missing')
            if sha256(p) != h:
                raise Fail(f'{p} does not hash to the stored {h[:12]}')
        out = os.path.join(c.out, 'reports', f'diff-dqi-rank-{i}.json')
        c.ke('diff-dqi-rank', ['diff', 'dqi', base, cand, '--profile', prof, '--n', str(ref['N']), '--threads',
                               c.a.threads, '--full', '--out', out])
        got = json.load(open(out))['prediction']
        bad = {}
        for k in ('delta_t', 'delta_t_se', 'predicted_expected_cost'):
            if got[k] != ref[k]:
                bad[k] = (got[k], ref[k])
        if got['precondition']['verdict'] != ref['precondition']['verdict']:
            bad['verdict'] = (got['precondition']['verdict'], ref['precondition']['verdict'])
        if got['paired'] != ref['paired']:
            bad['paired'] = 'paired Toffoli totals differ'
        if bad:
            raise Fail(f'{os.path.basename(rj)}: {bad}')
        done.append(f'{ref["base"]["sha256"][:6]}->{ref["candidate"]["sha256"][:6]}')
    return f'{len(done)} pairs equal to the stored predictions ({", ".join(done)})'


def _kmb_types_offset(path):
    with open(path, 'rb') as f:
        h = f.read(16 + 16 + 72)
        nregs = int.from_bytes(h[28:32], 'little')
        num_ops = int.from_bytes(h[32:40], 'little')
        pos = len(h)
        for _ in range(nregs):
            f.seek(pos + 8)
            pos += 16 + int.from_bytes(f.read(8), 'little')
        f.seek(pos)
        if int.from_bytes(f.read(8), 'little') != 1:
            raise Fail(f'{path}: unexpected .kmb layout')
        return pos + 16, num_ops


def check_diff_broken(c: Ctx):
    c.need_wheel(); c.need('invsqrt', 'verifiers')
    off, n = _kmb_types_offset(c.a.invsqrt)
    data = bytearray(open(c.a.invsqrt, 'rb').read())
    idx = next(i for i in range(n // 2, n) if data[off + i] == 0x21)
    data[off + idx] = 0x20
    broken = os.path.join(c.work, 'broken-cx-to-cz.kmb')
    open(broken, 'wb').write(bytes(data))
    rself = os.path.join(c.out, 'reports', 'diff-self.json')
    rbro = os.path.join(c.out, 'reports', 'diff-broken.json')
    c.ke('diff-broken', ['diff', 'invsqrt', c.a.invsqrt, c.a.invsqrt, '--threads', c.a.threads, '--out', rself])
    c.ke('diff-broken', ['diff', 'invsqrt', c.a.invsqrt, broken, '--threads', c.a.threads, '--out', rbro])
    s, b = json.load(open(rself)), json.load(open(rbro))
    ss, bs = s['semantic'], b['semantic']
    problems = []
    if not s['structural']['identical']:
        problems.append('self pair not structurally identical')
    if ss['agree_fraction'] != 1.0 or ss['toffolis']['delta']['mean_per_shot'] != 0:
        problems.append(f'self pair agreement {ss["agree_fraction"]}, delta {ss["toffolis"]["delta"]}')
    if b['structural']['identical'] or not b['structural']['hunks']:
        problems.append('mutation not found structurally')
    vb = bs['verdicts']['b']
    if bs['agree_fraction'] >= 1.0 or (vb['value'] + vb['phase']) == 0 or any(bs['verdicts']['a'].values()):
        problems.append(f'broken pair agreement {bs["agree_fraction"]}, verdicts {bs["verdicts"]}')
    if problems:
        raise Fail('; '.join(problems))
    h = b['structural']['hunks'][0]
    return (f'self pair agrees on every shot; CX->CZ at op {idx}: {len(b["structural"]["hunks"])} hunk(s) '
            f'(first {h.get("a")}->{h.get("b")}), agreement {bs["agree_fraction"]:.3f}, '
            f'candidate errors {vb["value"]} value / {vb["phase"]} phase')


CHECKS = [
    ('versions', check_versions),
    ('cargo-test', check_cargo_test),
    ('wheel-build', check_wheel_build),
    ('wheel-smoke', check_wheel_smoke),
    ('pytest-kickmix', check_pytest_kickmix),
    ('parity-invsqrt', check_parity_invsqrt),
    ('parity-dqi', check_parity_dqi),
    ('parity-dfthcsos', check_parity_dfthcsos),
    ('reaction-depth', check_reaction_depth),
    ('fail-paths', check_fail_paths),
    ('crosscheck-invsqrt', check_crosscheck_invsqrt),
    ('crosscheck-dqi', check_crosscheck_dqi),
    ('crosscheck-dfthcsos', check_crosscheck_dfthcsos),
    ('profile-invsqrt', check_profile_invsqrt),
    ('diff-dqi-rank', check_diff_dqi_rank),
    ('diff-broken', check_diff_broken),
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out-dir', required=True, help='new directory for logs, reports, the wheel and work files')
    ap.add_argument('--verifiers', default=os.environ.get('KICKMIX_ENGINE_VERIFIERS'),
                    help="the competition's verifiers/ directory (default: $KICKMIX_ENGINE_VERIFIERS)")
    ap.add_argument('--kickmix-python', default=os.environ.get('KICKMIX_PYTHON', sys.executable),
                    help='interpreter where kickmix imports (official verifier, parity scripts); default this one')
    ap.add_argument('--maturin', default=os.environ.get('MATURIN'))
    ap.add_argument('--python-for-venv', default='3.12', help='Python for the fresh smoke venv (uv) (default 3.12)')
    ap.add_argument('--invsqrt'); ap.add_argument('--dqi'); ap.add_argument('--dfthcsos')
    ap.add_argument('--rank-json', action='append', default=[],
                    help='stored dqi_fast_rank.py / engine diff result to reproduce (repeatable)')
    ap.add_argument('--qrom-words'); ap.add_argument('--qrom-plans')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--only', default=None); ap.add_argument('--skip', default=None)
    a = ap.parse_args(argv)
    for k in ('verifiers', 'invsqrt', 'dqi', 'dfthcsos', 'qrom_words', 'qrom_plans', 'kickmix_python', 'maturin'):
        if getattr(a, k):
            setattr(a, k, os.path.abspath(getattr(a, k)) if os.path.exists(getattr(a, k)) else getattr(a, k))
    a.rank_json = [os.path.abspath(x) for x in a.rank_json]
    if os.path.exists(a.out_dir):
        sys.exit(f'refusing to reuse {a.out_dir}: pass a new directory')
    c = Ctx(a)
    for d in (c.logs, c.work, c.cache, os.path.join(c.out, 'reports')):
        os.makedirs(d)
    only = set(a.only.split(',')) if a.only else None
    skip = set(a.skip.split(',')) if a.skip else set()
    known = {n for n, _ in CHECKS}
    for n in (only or set()) | skip:
        if n not in known:
            sys.exit(f'unknown check {n}; known: {", ".join(sorted(known))}')
    git = subprocess.run(['git', '-C', ROOT, 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'], capture_output=True,
                           text=True).stdout.strip()
    meta = {'repo': ROOT, 'git_head': git, 'tracked_changes': bool(dirty), 'started': time.strftime('%FT%TZ', time.gmtime()),
            'threads': a.threads, 'kickmix_python': a.kickmix_python, 'verifiers': a.verifiers,
            'inputs': {k: getattr(a, k) for k in ('invsqrt', 'dqi', 'dfthcsos')}, 'load_start': os.getloadavg()}
    print(f'release check of {ROOT} at {git[:12]}{" (with uncommitted changes)" if dirty else ""} -> {c.out}', flush=True)
    for name, fn in CHECKS:
        if (only and name not in only) or name in skip:
            if name == 'versions' or name == 'wheel-build':
                pass  # always needed to know the version / have the artifact
            else:
                continue
        t0 = time.perf_counter()
        try:
            detail, status = fn(c), 'PASS'
        except Skip as ex:
            detail, status = str(ex), 'SKIP'
        except Fail as ex:
            detail, status = str(ex), 'FAIL'
        except Exception as ex:  # a crash in a check is a failure of that check, not of the harness
            detail, status = f'{type(ex).__name__}: {ex}', 'FAIL'
        sec = time.perf_counter() - t0
        c.results[name] = {'status': status, 'seconds': round(sec, 1), 'detail': detail}
        print(f'{status:4}  {name:20} {sec:7.1f} s  {detail}', flush=True)
    meta['finished'] = time.strftime('%FT%TZ', time.gmtime())
    meta['load_end'] = os.getloadavg()
    meta['wheel'] = c.wheel and {'file': os.path.basename(c.wheel), 'sha256': sha256(c.wheel)}
    json.dump({'meta': meta, 'results': c.results}, open(os.path.join(c.out, 'release-check.json'), 'w'), indent=1)
    w = max(len(n) for n in c.results)
    print('\n' + '=' * 78)
    print(f'{"check":{w}}  status  seconds')
    for n, r in c.results.items():
        print(f'{n:{w}}  {r["status"]:6}  {r["seconds"]:7.1f}')
    fails = [n for n, r in c.results.items() if r['status'] == 'FAIL']
    skips = [n for n, r in c.results.items() if r['status'] == 'SKIP']
    print('=' * 78)
    print(f'RESULT: {"FAIL" if fails else "PASS"}  ({len(c.results) - len(fails) - len(skips)} pass, {len(fails)} fail, '
          f'{len(skips)} skip){"  failed: " + ", ".join(fails) if fails else ""}')
    print(f'report: {os.path.join(c.out, "release-check.json")}')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
