"""`kickmix-engine` command line. Written by an AI model (Claude) under the owner's direction.

    kickmix-engine verify dqi CIRCUIT.kmb [--salt S] [--threads N] [--cross-check] ...
    kickmix-engine verify invsqrt CIRCUIT.kmb [--salt S] [--threads N] [--cross-check] ...
    kickmix-engine cache info|prune|clear [--cache-dir DIR]
    kickmix-engine doctor [--json] [--manifest PATH_OR_URL]

`verify` prints the official verifier's summary JSON (same fields, including `reaction_depth`) and exits 0 on PASS,
1 on FAIL ({"status": "FAIL", "error": ...}), like the official CLI. The official
verifier file is required (it supplies seeds, sampling, the reference table and the statistics): pass --verifier or
set KICKMIX_ENGINE_VERIFIERS to the competition's `verifiers/` directory. The verifier imports kickmix, so kickmix must
be importable in this interpreter (the simulation itself does not use it).

--cross-check also runs the official verifier CLI unchanged (in --official-python, default this interpreter) and
prints {"engine": ..., "official": ..., "differences": ..., "match": bool, "engine_kernel": ...}; exit code 3 if
any field differs.

The simulator kernel (x86_64: baseline, avx2 or avx512, the best the CPU supports; aarch64: neon) is chosen once per
process; KICKMIX_ENGINE_KERNEL=baseline|avx2|avx512 forces one (exit 2 if this CPU cannot run it). `--timing` and
`--cross-check` report it. `doctor` reports the CPU features, kernels, versions, cache and verifier status, and with
--manifest the matching wheel of a release (`release-manifest.json`) with its SHA-256.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from fractions import Fraction

from .cache import ProgramCache

def _verifier_path(task, given):
    if given:
        return os.path.abspath(given)
    if task == 'dqi':
        from .dqi import default_verifier
    else:
        from .invsqrt import default_verifier
    return default_verifier()


def _engine_verify(a, cache):
    if a.task == 'dqi':
        from . import dqi
        kw = dict(m=a.m, n=a.n, degree=a.degree, r=a.r, num_shots=a.num_shots)
        t = dqi.DQITask(a.circuit_file, verifier_path=a.verifier, cache=cache, **kw)
        v = t.v
        log2_delta = v.DEFAULT_LOG2_DELTA if a.log2_delta is None else a.log2_delta
        v._validate_inputs(t.spec, a.batch_size, log2_delta, None)
        out = t.verify(salt=a.salt, seed=a.seed, threads=a.threads, sampler=a.sampler, self_check=a.self_check,
                       batch_size=a.batch_size, log2_delta=log2_delta)
        info = dict(load=t.load_info, prepare_s=t.t_prepare, wall_s=t.wall, self_check=getattr(t, 'self_check_result', None),
                    sampler=a.sampler)
        return out, info
    from . import invsqrt
    v = invsqrt.load_verifier(a.verifier)
    n_p = a.n_p if a.n_p is not None else v.N_P
    width = a.width if a.width is not None else 2 * n_p + 2
    max_x = a.max_x if a.max_x is not None else 3 * (2**n_p - 2)**2
    eta = a.eta if a.eta is not None else v.ETA
    system_qubits = a.system_qubits if a.system_qubits is not None else 3 * eta * n_p
    spec = v.ProblemSpecInvSqrt(width=width, max_x=max_x, beta=a.beta, precision_bits=a.precision_bits,
                                shots_per_x=a.shots_per_x or v.DEFAULT_SHOTS_PER_X, system_qubits=system_qubits)
    t = invsqrt.InvSqrtTask(a.circuit_file, spec=spec, verifier_path=a.verifier, cache=cache)
    out = t.verify(salt=a.salt, seed=a.seed, threads=a.threads, batch_size=a.batch_size)
    return out, dict(load=t.load_info, prepare_s=t.t_prepare, wall_s=t.wall)


def _official_cmd(a):
    cmd = [a.official_python or sys.executable, a.verifier, a.circuit_file, '--salt', a.salt]
    if a.seed is not None:
        cmd += ['--seed', str(a.seed)]
    if a.official_workers is not None:
        cmd += ['--max-workers', str(a.official_workers)]
    cmd += ['--batch-size', str(a.batch_size)]
    if a.task == 'dqi':
        for flag, val in (('--m', a.m), ('--n', a.n), ('--degree', a.degree), ('--r', a.r),
                          ('--num-shots', a.num_shots), ('--log2-delta', a.log2_delta)):
            if val is not None:
                cmd += [flag, str(val)]
    else:
        for flag, val in (('--n-p', a.n_p), ('--precision-bits', a.precision_bits), ('--shots-per-x', a.shots_per_x),
                          ('--eta', a.eta), ('--width', a.width), ('--max-x', a.max_x),
                          ('--system-qubits', a.system_qubits)):
            if val is not None:
                cmd += [flag, str(val)]
        if a.beta != Fraction(1):
            cmd += ['--beta', str(a.beta)]
    return cmd


def cmd_verify(a):
    a.verifier = _verifier_path(a.task, a.verifier)
    if not a.verifier or not os.path.exists(a.verifier):
        print(json.dumps({'status': 'FAIL', 'error': 'official verifier not found: pass --verifier or set '
                          'KICKMIX_ENGINE_VERIFIERS'}, indent=2))
        return 1
    from . import kernel_info
    try:
        kinfo = kernel_info()
    except ValueError as ex:
        print(json.dumps({'status': 'ERROR', 'error': str(ex)}, indent=2))
        return 2
    kernel = {'kernel': kinfo['kernel'], 'selected_by': kinfo['selected_by'], 'available': kinfo['available'],
              'requires': kinfo['kernel_requires'],
              'cpu_features': sorted(f for f, d in kinfo['features'].items() if d)}
    cache = ProgramCache(a.cache_dir, enabled=not a.no_cache)
    t0 = time.perf_counter()
    try:
        eng, info = _engine_verify(a, cache)
        rc = 0
    except (AssertionError, ValueError, RuntimeError, OSError) as ex:
        eng, info, rc = {'status': 'FAIL', 'error': str(ex)}, {}, 1
    info['total_s'] = time.perf_counter() - t0
    info['kernel'] = kernel
    if a.timing:
        print(json.dumps({'engine_info': info}, default=str), file=sys.stderr)
    if not a.cross_check:
        print(json.dumps(eng, indent=2))
        return rc
    cmd = _official_cmd(a)
    t1 = time.perf_counter()
    p = subprocess.run(cmd, capture_output=True, text=True)
    t_off = time.perf_counter() - t1
    try:
        off = json.loads(p.stdout)
    except ValueError:
        off = {'status': 'ERROR', 'error': 'could not parse the official verifier output',
               'stdout_tail': p.stdout[-2000:], 'stderr_tail': p.stderr[-2000:]}
    diffs = {}
    for k in sorted(set(eng) | set(off)):
        if eng.get(k) != off.get(k):
            diffs[k] = {'engine': eng.get(k), 'official': off.get(k)}
    res = {'match': not diffs, 'differences': diffs, 'engine': eng, 'official': off, 'engine_kernel': kernel,
           'official_command': cmd, 'official_returncode': p.returncode,
           'seconds': {'engine': round(info['total_s'], 3), 'official': round(t_off, 3)}}
    print(json.dumps(res, indent=2, default=str))
    return 0 if (not diffs and rc == 0) else (3 if diffs else rc)


def cmd_cache(a):
    c = ProgramCache(a.cache_dir)
    if a.action == 'info':
        from .cache import engine_key
        ents = c.entries()
        print(json.dumps({'dir': c.dir, 'engine_key': engine_key(), 'entries': len(ents),
                          'bytes': sum(s for _, s in ents), 'files': [f for f, _ in ents]}, indent=2))
    elif a.action == 'prune':
        print(json.dumps({'dir': c.dir, 'removed_other_builds': c.prune()}))
    else:
        print(json.dumps({'dir': c.dir, 'removed': c.clear()}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='kickmix-engine', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify', help='verify a circuit like the official verifier, with the engine as simulator')
    v.add_argument('task', choices=('dqi', 'invsqrt'))
    v.add_argument('circuit_file', help='.kmb circuit file')
    v.add_argument('--verifier', default=None, help='official verify_submission_*.py (default: $KICKMIX_ENGINE_VERIFIERS)')
    v.add_argument('--salt', default='')
    v.add_argument('--seed', type=int, default=None)
    v.add_argument('--threads', type=int, default=os.cpu_count() or 1)
    v.add_argument('--batch-size', type=int, default=256)
    v.add_argument('--cache-dir', default=None, help='program cache (default: $KICKMIX_ENGINE_CACHE or ~/.cache/kickmix-engine)')
    v.add_argument('--no-cache', action='store_true')
    v.add_argument('--timing', action='store_true', help='print engine timing and cache info to stderr')
    v.add_argument('--cross-check', action='store_true', help='also run the official verifier and diff the summaries')
    v.add_argument('--official-python', default=None, help='interpreter for the official verifier (needs kickmix)')
    v.add_argument('--official-workers', type=int, default=None, help="official --max-workers (default: the verifier's)")
    g = v.add_argument_group('dqi')
    g.add_argument('--sampler', choices=('engine', 'official'), default='engine')
    g.add_argument('--self-check', type=int, default=0, help='re-draw K batches (-1: all) with the official sample_shots')
    g.add_argument('--m', type=int); g.add_argument('--n', type=int); g.add_argument('--degree', type=int)
    g.add_argument('--r', type=int); g.add_argument('--num-shots', type=int); g.add_argument('--log2-delta', type=float)
    g = v.add_argument_group('invsqrt')
    g.add_argument('--n-p', type=int); g.add_argument('--beta', type=Fraction, default=Fraction(1))
    g.add_argument('--precision-bits', type=int, default=24); g.add_argument('--shots-per-x', type=int)
    g.add_argument('--eta', type=int); g.add_argument('--width', type=int); g.add_argument('--max-x', type=int)
    g.add_argument('--system-qubits', type=int)
    c = sub.add_parser('cache', help='inspect or clear the program cache')
    c.add_argument('action', choices=('info', 'prune', 'clear'))
    c.add_argument('--cache-dir', default=None)
    d = sub.add_parser('doctor', help='report CPU features, kernels, versions, cache and verifier status')
    d.add_argument('--json', action='store_true', help='one JSON object (for agents and automation)')
    d.add_argument('--manifest', default=None, help='release-manifest.json (path or URL): name the matching wheel')
    a = ap.parse_args(argv)
    if a.cmd == 'doctor':
        from .doctor import main as doctor_main
        return doctor_main(a)
    return cmd_verify(a) if a.cmd == 'verify' else cmd_cache(a)


if __name__ == '__main__':
    sys.exit(main())
