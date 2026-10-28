"""`kickmix-engine doctor`: what this machine and installation look like to the engine.

Written by an AI model (Claude) under the owner's direction.

Reports the platform, the Python interpreter, the detected CPU features, the simulator kernels this build contains,
which of them this CPU can run and which one is selected, the engine version and extension hash, the program cache
directory, and whether kickmix and the official verifiers are importable. Given a release manifest
(`release-manifest.json` from a GitHub release, as a path or URL) it also names the wheel in that release that fits
this interpreter and platform, with its SHA-256, and whether the installed extension is that wheel's version.
`--json` prints one JSON object (stable keys, for agents); without it a short human-readable summary.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import platform
import re
import sys

from . import _engine, cpu_info, engine_version
from .cache import cache_dir

SCHEMA = 'kickmix-engine-doctor/1'


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _kickmix():
    try:
        spec = importlib.util.find_spec('kickmix')
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return dict(importable=False, error='kickmix is not installed in this interpreter')
    try:
        import kickmix  # noqa: F401
        from importlib import metadata
        try:
            ver = metadata.version('kickmix')
        except metadata.PackageNotFoundError:
            ver = getattr(kickmix, '__version__', None)
        return dict(importable=True, version=ver, file=getattr(kickmix, '__file__', None))
    except Exception as ex:  # noqa: BLE001 - report any import failure
        return dict(importable=False, error=f'{type(ex).__name__}: {ex}')


def _verifier(task):
    if task == 'dqi':
        from .dqi import default_verifier, load_verifier
    elif task == 'dfthcsos':
        from .dfthcsos import default_verifier, load_verifier
    else:
        from .invsqrt import default_verifier, load_verifier
    path = default_verifier()
    out = dict(path=path, found=bool(path and os.path.exists(path)))
    if not out['found']:
        out['importable'] = False
        out['error'] = ('not found: set KICKMIX_ENGINE_VERIFIERS to the competition verifiers/ directory'
                        if not path else 'file does not exist')
        return out
    out['sha256'] = _sha256(os.path.join(path, 'verify.py') if os.path.isdir(path) else path)
    try:
        load_verifier(path)
        out['importable'] = True
    except Exception as ex:  # noqa: BLE001
        out['importable'] = False
        out['error'] = f'{type(ex).__name__}: {ex}'
    return out


def _read_manifest(src):
    if re.match(r'^https?://', src):
        import urllib.request
        with urllib.request.urlopen(src, timeout=30) as r:  # noqa: S310 - user-supplied manifest URL
            return json.loads(r.read().decode())
    with open(src) as f:
        return json.load(f)


def _supported_tags():
    """Wheel tags this interpreter accepts, best first (packaging if available, else a conservative fallback)."""
    try:
        from packaging import tags
        return [str(t) for t in tags.sys_tags()], 'packaging'
    except ImportError:
        pass
    out = []
    major, minor = sys.version_info[:2]
    mach = platform.machine().lower()
    plats = []
    if sys.platform == 'darwin':
        arch = 'arm64' if mach in ('arm64', 'aarch64') else 'x86_64'
        mac = platform.mac_ver()[0] or '11.0'
        mmaj = int(mac.split('.')[0])
        for v in range(mmaj, 10, -1):
            plats.append(f'macosx_{v}_0_{arch}')
        plats += [f'macosx_10_{m}_{arch}' for m in range(15, 8, -1)] + ['macosx_10_9_universal2']
    elif sys.platform.startswith('linux'):
        arch = {'amd64': 'x86_64', 'arm64': 'aarch64'}.get(mach, mach)
        lib, ver = platform.libc_ver()
        if lib == 'glibc' and ver:
            gmaj, gmin = (int(x) for x in ver.split('.')[:2])
            for m in range(gmin, 16, -1):
                plats.append(f'manylinux_{gmaj}_{m}_{arch}')
            plats += [f'manylinux2014_{arch}', f'linux_{arch}']
    for p in plats:
        out.append(f'cp{major}{minor}-cp{major}{minor}-{p}')
        for m in range(minor, 1, -1):
            out.append(f'cp3{m}-abi3-{p}')
    return out, 'fallback'


def _wheel_tags(filename):
    name = filename[:-4] if filename.endswith('.whl') else filename
    parts = name.split('-')
    if len(parts) < 5:
        return set()
    py, abi, plat = parts[-3], parts[-2], parts[-1]
    return {f'{a}-{b}-{c}' for a in py.split('.') for b in abi.split('.') for c in plat.split('.')}


def match_manifest(manifest, kernel_name=None):
    supported, how = _supported_tags()
    rank = {t: i for i, t in enumerate(supported)}
    best = None
    for w in manifest.get('wheels', []):
        r = min((rank[t] for t in _wheel_tags(w.get('filename', '')) if t in rank), default=None)
        if r is not None and (best is None or r < best[0]):
            best = (r, w)
    out = dict(engine_version=manifest.get('engine_version'), git_commit=manifest.get('git_commit'),
               ci_run_id=manifest.get('ci_run_id'), tag_source=how, installed_version=engine_version()[0])
    if best is None:
        out.update(match=None, error='no wheel in this manifest fits this interpreter and platform')
        return out
    w = best[1]
    out['match'] = {k: w.get(k) for k in ('filename', 'sha256', 'size', 'platform_tag', 'python_tag', 'abi_tag',
                                          'kernels', 'url', 'test_status')}
    names = [k.get('name') for k in (w.get('kernels') or [])]
    out['selected_kernel_in_wheel'] = (kernel_name in names) if kernel_name else None
    out['installed_matches_manifest_version'] = manifest.get('engine_version') == out['installed_version']
    return out


def collect(manifest=None):
    ci = cpu_info()
    try:
        ki = _engine.kernel_info()
        kernel = dict(selected=ki['kernel'], selected_by=ki['selected_by'], requires=list(ki['kernel_requires']),
                      error=None)
    except ValueError as ex:
        kernel = dict(selected=None, selected_by=None, requires=None, error=str(ex))
    so = _engine.__file__
    ver, fmt = engine_version()
    cdir = cache_dir()
    progs = os.path.join(cdir, 'programs')
    rep = dict(
        schema=SCHEMA,
        engine=dict(version=ver, cache_format=fmt, extension=so, extension_sha256=_sha256(so),
                    package_dir=os.path.dirname(os.path.abspath(__file__))),
        platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine(),
                      platform=platform.platform(), libc=' '.join(platform.libc_ver()).strip() or None,
                      mac_ver=platform.mac_ver()[0] or None, cpu_count=os.cpu_count()),
        python=dict(version=platform.python_version(), implementation=platform.python_implementation(),
                    executable=sys.executable),
        cpu=dict(arch=ci['arch'], features={k: bool(v) for k, v in ci['features'].items()}),
        kernels=dict(compiled=list(ci['compiled']), available=list(ci['available']),
                     requirements={k: list(v) for k, v in ci['kernel_features'].items()},
                     env_var=ci['env_var'], env_request=ci['env_request'], **kernel),
        cache=dict(dir=cdir, exists=os.path.isdir(cdir),
                   programs=len([f for f in os.listdir(progs) if f.endswith('.kmxprog')]) if os.path.isdir(progs) else 0),
        kickmix=_kickmix(),
        verifiers=dict(env=os.environ.get('KICKMIX_ENGINE_VERIFIERS'), dqi=_verifier('dqi'),
                       invsqrt=_verifier('invsqrt'), dfthcsos=_verifier('dfthcsos')),
    )
    if manifest:
        try:
            rep['manifest'] = dict(source=manifest, **match_manifest(_read_manifest(manifest), kernel['selected']))
        except Exception as ex:  # noqa: BLE001
            rep['manifest'] = dict(source=manifest, error=f'{type(ex).__name__}: {ex}')
    rep['ok'] = bool(kernel['error'] is None)
    return rep


def _human(r):
    k = r['kernels']
    feats = ' '.join(f for f, d in r['cpu']['features'].items() if d) or '-'
    lines = [
        f"kickmix-engine {r['engine']['version']} ({r['engine']['extension']})",
        f"platform  {r['platform']['platform']} ({r['platform']['cpu_count']} CPUs)",
        f"python    {r['python']['version']} {r['python']['executable']}",
        f"cpu       {r['cpu']['arch']}: {feats}",
        f"kernels   compiled {', '.join(k['compiled'])}; available {', '.join(k['available'])}",
        (f"selected  {k['selected']} ({k['selected_by']})" if k['error'] is None else f"selected  ERROR: {k['error']}"),
        f"cache     {r['cache']['dir']} ({r['cache']['programs']} programs)",
        ('kickmix   ' + (f"{r['kickmix'].get('version')} {r['kickmix'].get('file')}" if r['kickmix']['importable']
                         else 'not importable: ' + r['kickmix']['error'])),
    ]
    for t in ('dqi', 'invsqrt', 'dfthcsos'):
        v = r['verifiers'][t]
        lines.append(f"verifier  {t}: " + ('importable ' + v['path'] if v.get('importable') else
                                           'NOT importable: ' + str(v.get('error'))))
    m = r.get('manifest')
    if m:
        if m.get('match'):
            w = m['match']
            lines.append(f"manifest  {w['filename']} sha256 {w['sha256']}")
            lines.append(f"          kernels {', '.join(x['name'] for x in w.get('kernels') or [])}; "
                         f"installed version matches: {m['installed_matches_manifest_version']}")
        else:
            lines.append('manifest  ' + str(m.get('error')))
    return '\n'.join(lines)


def main(a):
    rep = collect(a.manifest)
    print(json.dumps(rep, indent=2) if a.json else _human(rep))
    return 0 if rep['ok'] else 2
