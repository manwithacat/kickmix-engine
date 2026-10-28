#!/usr/bin/env python3
"""Run the parity-relevant tests once per simulator kernel this machine supports; write a status JSON.

Written by an AI model (Claude) under the owner's direction. Used by .github/workflows/wheels.yml (Linux x86_64).

    python tools/ci_kernel_tests.py --tests TESTS_DIR --wheel dist/X.whl --status OUT.json [--kernels a,b,c]

For each kernel (default: every kernel the installed build contains), checks in a child process whether
`KICKMIX_ENGINE_KERNEL=<kernel>` is accepted on this CPU. If not, the kernel is recorded as skipped with the reason
(and a GitHub notice is printed); if so, pytest runs on TESTS_DIR with that kernel forced, and the kernel the engine
actually used is checked. Exit 1 if any supported kernel fails, or if no kernel ran.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys

PROBE = 'import json, kickmix_engine as k; print(json.dumps(k.kernel_info()))'


def cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--tests', required=True)
    ap.add_argument('--wheel', required=True, help='the installed wheel file (its name keys the status record)')
    ap.add_argument('--status', required=True)
    ap.add_argument('--kernels', default=None)
    ap.add_argument('--pytest-args', default='-v -rs -p no:cacheprovider')
    a = ap.parse_args(argv)
    import kickmix_engine as k
    ci = k.cpu_info()
    kernels = a.kernels.split(',') if a.kernels else list(ci['compiled'])
    rec = dict(wheel=os.path.basename(a.wheel), python=platform.python_version(), runner=os.environ.get('RUNNER_NAME'),
               runner_os=os.environ.get('RUNNER_OS'), cpu_model=cpu_model(), arch=ci['arch'],
               features={f: bool(d) for f, d in ci['features'].items()}, available=list(ci['available']), kernels={})
    any_fail, ran = False, 0
    for kern in kernels:
        env = dict(os.environ, KICKMIX_ENGINE_KERNEL=kern)
        p = subprocess.run([sys.executable, '-c', PROBE], env=env, capture_output=True, text=True)
        if p.returncode != 0:
            reason = (p.stderr.strip().splitlines() or ['probe failed'])[-1]
            print(f'::notice title=kernel {kern} skipped::{reason}', flush=True)
            print(f'=== kernel {kern}: SKIPPED (this runner cannot run it): {reason}', flush=True)
            rec['kernels'][kern] = dict(outcome='skipped', reason=reason)
            continue
        used = json.loads(p.stdout)['kernel']
        print(f'=== kernel {kern}: running tests (engine reports kernel={used})', flush=True)
        t = subprocess.run([sys.executable, '-m', 'pytest', *a.pytest_args.split(), a.tests], env=env)
        ok = t.returncode == 0 and used == kern
        rec['kernels'][kern] = dict(outcome='passed' if ok else 'failed', pytest_returncode=t.returncode,
                                    engine_reported=used)
        print(f'=== kernel {kern}: {"PASSED" if ok else "FAILED"}', flush=True)
        any_fail |= not ok
        ran += 1
    os.makedirs(os.path.dirname(os.path.abspath(a.status)), exist_ok=True)
    with open(a.status, 'w') as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec['kernels'], indent=2))
    return 1 if (any_fail or ran == 0) else 0


if __name__ == '__main__':
    sys.exit(main())
