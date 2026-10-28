#!/usr/bin/env python3
"""Build `release-manifest.json` from built wheels (and an sdist), for agents and automation.

Written by an AI model (Claude) under the owner's direction.

    python tools/make_release_manifest.py DIST_DIR [--status-dir DIR] [--git-commit SHA] [--git-ref REF]
        [--ci-run-id ID] [--ci-run-url URL] [--release-url-base URL] [--out DIST_DIR/release-manifest.json]

For every `*.whl` in DIST_DIR: filename, Python/ABI/platform tags, OS and architecture, SHA-256, size, the simulator
kernels the wheel contains with the CPU features each needs (read from `src/kernel.rs`, so the two cannot drift),
and the CI test status for that wheel per kernel (from the `test-status-*.json` files the workflow's test jobs write
into --status-dir). Also the sdist, the engine version (from Cargo.toml), git commit and ref, CI run id and URL.
Standard library only. Exits 1 if DIST_DIR has no wheels or a wheel's version differs from Cargo.toml.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHEMA = 'kickmix-engine-release-manifest/1'


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def cargo_version():
    with open(os.path.join(ROOT, 'Cargo.toml')) as f:
        m = re.search(r'^version\s*=\s*"([^"]+)"', f.read(), re.M)
    return m.group(1)


def kernel_features():
    """AVX2_FEATURES / AVX512_FEATURES as declared in src/kernel.rs."""
    src = open(os.path.join(ROOT, 'src', 'kernel.rs')).read()
    out = {}
    for name in ('AVX2_FEATURES', 'AVX512_FEATURES'):
        m = re.search(name + r'\s*:\s*&\[&str\]\s*=\s*&\[([^\]]*)\]', src, re.S)
        if not m:
            raise SystemExit(f'cannot find {name} in src/kernel.rs')
        out[name] = re.findall(r'"([^"]+)"', m.group(1))
    # The target_feature attributes must name the same features (they are string literals in sim.rs/dqi_sample.rs).
    for f in ('sim.rs', 'dqi_sample.rs'):
        attrs = re.findall(r'#\[target_feature\(enable = "([^"]+)"\)\]', open(os.path.join(ROOT, 'src', f)).read())
        want = sorted([','.join(out['AVX2_FEATURES']), ','.join(out['AVX512_FEATURES'])])
        if sorted(attrs) != want:
            raise SystemExit(f'{f}: target_feature attributes {attrs} do not match kernel.rs {want}')
    return out


def kernels_for(arch, feats):
    if arch == 'x86_64':
        return [dict(name='baseline', requires=[], note='x86-64 baseline (SSE2); runs on any x86_64 CPU'),
                dict(name='avx2', requires=feats['AVX2_FEATURES'], note='x86-64-v3 feature set'),
                dict(name='avx512', requires=feats['AVX512_FEATURES'], note='x86-64-v4 subset (v3 + AVX-512 F/BW/VL/DQ)')]
    if arch in ('arm64', 'aarch64'):
        return [dict(name='neon', requires=['neon'], note='aarch64 baseline')]
    return [dict(name='baseline', requires=[], note='portable build')]


def parse_wheel(fn):
    m = re.match(r'^(?P<name>[^-]+)-(?P<ver>[^-]+)(-\d[^-]*)?-(?P<py>[^-]+)-(?P<abi>[^-]+)-(?P<plat>[^-]+)\.whl$', fn)
    if not m:
        raise SystemExit(f'not a wheel filename: {fn}')
    plat = m['plat']
    first = plat.split('.')[0]
    if first.startswith('macosx'):
        os_name = 'macos'
    elif 'linux' in first:
        os_name = 'linux'
    elif first.startswith('win'):
        os_name = 'windows'
    else:
        os_name = 'unknown'
    arch = next((a for a in ('x86_64', 'aarch64', 'arm64', 'universal2', 'amd64') if first.endswith(a)), 'unknown')
    arch = {'amd64': 'x86_64'}.get(arch, arch)
    return dict(name=m['name'], version=m['ver'], python_tag=m['py'], abi_tag=m['abi'], platform_tag=plat,
                os=os_name, arch=arch)


def load_status(status_dir):
    """test-status-*.json -> {wheel filename: [status records]}."""
    out = {}
    if not status_dir or not os.path.isdir(status_dir):
        return out
    for p in sorted(glob.glob(os.path.join(status_dir, '**', 'test-status-*.json'), recursive=True)):
        try:
            rec = json.load(open(p))
        except (OSError, ValueError) as ex:
            print(f'warning: unreadable status file {p}: {ex}', file=sys.stderr)
            continue
        out.setdefault(rec.get('wheel'), []).append(rec)
    return out


def summarise(records, kernels):
    """Per kernel: 'passed' if every Python version passed it, 'failed' if any failed, else 'not run on CI'."""
    per = {}
    for k in kernels:
        res = [r.get('kernels', {}).get(k['name']) for r in records]
        res = [x for x in res if x is not None]
        outcomes = [x.get('outcome') for x in res]
        if not res:
            s = 'not run on CI'
        elif any(o == 'failed' for o in outcomes):
            s = 'failed'
        elif all(o == 'passed' for o in outcomes):
            s = 'passed'
        else:
            s = 'skipped: ' + '; '.join(sorted({x.get('reason', '') for x in res if x.get('outcome') == 'skipped'}))
        per[k['name']] = s
    return dict(per_kernel=per, runs=[dict(python=r.get('python'), runner=r.get('runner'), cpu=r.get('cpu_model'),
                                           kernels=r.get('kernels'), available=r.get('available'))
                                      for r in records])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dist')
    ap.add_argument('--status-dir', default=None)
    ap.add_argument('--git-commit', default=os.environ.get('GITHUB_SHA'))
    ap.add_argument('--git-ref', default=os.environ.get('GITHUB_REF'))
    ap.add_argument('--ci-run-id', default=os.environ.get('GITHUB_RUN_ID'))
    ap.add_argument('--ci-run-url', default=None)
    ap.add_argument('--release-url-base', default=None,
                    help='e.g. https://github.com/OWNER/REPO/releases/download/vX.Y.Z (adds a url per file)')
    ap.add_argument('--parity-evidence', default='README.md#recorded-outcomes',
                    help='where the per-shot parity evidence for this release is described')
    ap.add_argument('--out', default=None)
    a = ap.parse_args(argv)
    if a.ci_run_url is None and a.ci_run_id and os.environ.get('GITHUB_REPOSITORY'):
        a.ci_run_url = f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/{a.ci_run_id}"
    version = cargo_version()
    feats = kernel_features()
    status = load_status(a.status_dir)
    wheels = []
    for p in sorted(glob.glob(os.path.join(a.dist, '*.whl'))):
        fn = os.path.basename(p)
        info = parse_wheel(fn)
        if info['version'] != version:
            raise SystemExit(f'{fn}: version {info["version"]} != Cargo.toml {version}')
        ks = kernels_for(info['arch'], feats)
        w = dict(filename=fn, **{k: info[k] for k in ('python_tag', 'abi_tag', 'platform_tag', 'os', 'arch')},
                 requires_python='>=3.12' if info['abi_tag'] == 'abi3' else None,
                 sha256=sha256(p), size=os.path.getsize(p), kernels=ks,
                 kernel_selection='best available at start-up; override with KICKMIX_ENGINE_KERNEL='
                                  + '|'.join(k['name'] for k in ks if k['name'] != 'neon') if info['arch'] == 'x86_64'
                 else 'single kernel', test_status=summarise(status.get(fn, []), ks))
        if a.release_url_base:
            w['url'] = a.release_url_base.rstrip('/') + '/' + fn
        wheels.append(w)
    if not wheels:
        raise SystemExit(f'no wheels in {a.dist}')
    sdists = []
    for p in sorted(glob.glob(os.path.join(a.dist, '*.tar.gz'))):
        s = dict(filename=os.path.basename(p), sha256=sha256(p), size=os.path.getsize(p))
        if a.release_url_base:
            s['url'] = a.release_url_base.rstrip('/') + '/' + s['filename']
        sdists.append(s)
    all_status = [v for w in wheels for v in w['test_status']['per_kernel'].values()]
    man = dict(
        schema=SCHEMA,
        package='kickmix-engine',
        engine_version=version,
        git_commit=a.git_commit,
        git_ref=a.git_ref,
        ci_run_id=a.ci_run_id,
        ci_run_url=a.ci_run_url,
        created_utc=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        test_status=('passed' if all(s == 'passed' or s.startswith('skipped') for s in all_status)
                     and any(s == 'passed' for s in all_status) else
                     'failed' if any(s == 'failed' for s in all_status) else 'incomplete'),
        parity_evidence=a.parity_evidence,
        how_to_choose=('Pick the wheel whose platform_tag fits your OS/arch (pip does this). x86_64 wheels contain '
                       'all three kernels and select at start-up; run `kickmix-engine doctor --json --manifest '
                       '<this file>` to see the kernel your CPU gets and the matching wheel with its sha256.'),
        wheels=wheels,
        sdist=sdists,
    )
    out = a.out or os.path.join(a.dist, 'release-manifest.json')
    with open(out, 'w') as f:
        json.dump(man, f, indent=2)
        f.write('\n')
    print(json.dumps({'out': out, 'wheels': [w['filename'] for w in wheels], 'test_status': man['test_status']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
