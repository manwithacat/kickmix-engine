"""`kickmix-engine` command line. Written by an AI model (Claude) under the owner's direction.

    kickmix-engine verify dqi CIRCUIT.kmb [--salt S] [--threads N] [--cross-check] ...
    kickmix-engine verify invsqrt CIRCUIT.kmb [--salt S] [--threads N] [--cross-check] ...
    kickmix-engine verify dfthcsos SUBMISSION_DIR [--salt S] [--threads N] [--cross-check] ...
    kickmix-engine diff TASK BASE.kmb CANDIDATE.kmb [--profile P.json] [--n 16] [--draws N] [--anchor HASH] ...
    kickmix-engine profile TASK BASE.kmb --out DIR [--draws N] [--anchor HASH] [--salt-format sweep-{}] ...
    kickmix-engine cache info|prune|clear [--cache-dir DIR]
    kickmix-engine doctor [--json] [--manifest PATH_OR_URL]
    kickmix-engine serve --socket PATH [--max-programs N]
    kickmix-engine --via PATH <any command above> | ping | stats | stop

`serve` keeps one engine process resident (compiled programs, plug-in reference tables and profiles stay in memory)
and `--via PATH` sends a command to it: the output and exit code are those of the one-shot command (see
python/kickmix_engine/resident.py).

`diff` compares two circuits of any task (`generic` for a circuit without a verifier) in three layers: structural
(op-level alignment, hunks, static deltas, register interface), semantic (both circuits on identical inputs and
simulator seeds; per-shot interface-state, phase and verdict agreement; the per-shot Toffoli change +- SE) and, with
`--profile`, a predicted expected cost through the task plug-in with the failing-shot precondition. `profile` records
a base's per-batch Toffolis and failing shots over a draw set once, for every later `diff --profile`. Both print JSON;
neither is a verified score (see python/kickmix_engine/diff.py).

`verify` prints the official verifier's summary JSON (same fields, including `reaction_depth`) and exits 0 on PASS,
1 on FAIL ({"status": "FAIL", "error": ...}), like the official CLI. The official
verifier file (for dfthcsos: the official `dfthcsos` package directory) is required (it supplies seeds, sampling, the reference table and the statistics): pass --verifier or
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
    elif task == 'dfthcsos':
        from .dfthcsos import default_verifier
    else:
        from .invsqrt import default_verifier
    return default_verifier()


def _engine_verify(a, cache):
    if a.task == 'dfthcsos':
        from . import dfthcsos
        info = {}
        out = dfthcsos.verify(a.circuit_file, a.factorization, a.tree_edges, format=a.format,
                              shots_per_address=a.shots_per_address, seed=a.seed, salt=a.salt, threads=a.threads,
                              verifier=a.verifier, cache=cache, fcidump_cache=not (a.no_cache or a.no_fcidump_cache),
                              low_memory=not a.no_low_memory, info=info)
        return out, info
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
    if a.task == 'dfthcsos':
        from .dfthcsos import _package_dir
        cmd = [a.official_python or sys.executable, str(_package_dir(a.verifier) / 'verify.py')]
        cmd += ([a.factorization] if a.factorization else []) + [a.circuit_file, '--salt', a.salt]
        if a.seed is not None:
            cmd += ['--seed', str(a.seed)]
        if a.tree_edges:
            cmd += ['--tree-edges', a.tree_edges]
        if a.format != 'auto':
            cmd += ['--format', a.format]
        if a.shots_per_address is not None:
            cmd += ['--shots-per-address', str(a.shots_per_address)]
        return cmd
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
    # The exceptions each official CLI reports as FAIL (the dfthcsos CLI also catches KeyError).
    caught = (AssertionError, ValueError, RuntimeError, OSError) + ((KeyError,) if a.task == 'dfthcsos' else ())
    try:
        eng, info = _engine_verify(a, cache)
        rc = 0
    except caught as ex:
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


def _draws(a):
    from .diff import DrawSet
    return DrawSet(anchor=a.anchor, salt_format=a.salt_format, count=a.draws)


def _plugin_spec(a):
    spec = {}
    if getattr(a, 'inputs', None):
        spec['inputs'] = [x for x in a.inputs.split(',') if x]
    if getattr(a, 'batches_per_draw', None):
        spec['batches_per_draw'] = a.batches_per_draw
    return spec


def cmd_diff(a):
    from .diff import diff, diff_family, write_new
    a.verifier = _verifier_path(a.task, a.verifier) if a.task != 'generic' else None
    cache = ProgramCache(a.cache_dir, enabled=not a.no_cache)
    if a.task == 'dfthcsos' and os.path.isdir(a.base):
        rep = diff_family(a.base, a.candidate, verifier_path=a.verifier, threads=a.threads, cache=cache,
                          structural_layer=not a.no_structural)
        if a.out:
            write_new(a.out, rep)
        from .diff import _json_default
        print(json.dumps(rep, indent=1, default=_json_default))
        return 0
    if a.task == 'dfthcsos':
        a.task = 'generic'  # single ANGLE_k circuits pair generically (uniform inputs on their registers)
    rep = diff(a.task, a.base, a.candidate, verifier_path=a.verifier, draws=None if a.profile else _draws(a), n=a.n,
               profile_json=a.profile, threads=a.threads, structural_layer=not a.no_structural,
               semantic_layer=not a.no_semantic, compare_state=not a.no_state, sweep=a.sweep,
               report_n=[int(x) for x in a.report_n.split(',')] if a.report_n else None, cache=cache,
               spec=_plugin_spec(a))
    if not a.full:
        rep = dict(rep)
        if 'prediction' in rep:
            rep['prediction'] = {k: v for k, v in rep['prediction'].items() if k not in ('paired', 'predicted_per_draw')}
        if 'structural' in rep:
            rep['structural'] = dict(rep['structural'], hunks=rep['structural']['hunks'][:a.show_hunks])
        if 'semantic' in rep:
            rep['semantic'] = dict(rep['semantic'], disagreements=rep['semantic']['disagreements'][:a.show_hunks])
    if a.out:
        write_new(a.out, rep)
    from .diff import _json_default
    print(json.dumps(rep, indent=1, default=_json_default))
    return 0


def cmd_profile(a):
    from .diff import Circuit, profile, write_new
    from .plugins import make_plugin
    a.verifier = _verifier_path(a.task, a.verifier) if a.task != 'generic' else None
    cache = ProgramCache(a.cache_dir, enabled=not a.no_cache)
    plugin = make_plugin(a.task, a.verifier, **_plugin_spec(a))
    circ = Circuit(a.base, plugin, cache)
    out = os.path.join(a.out, f'profile-{circ.sha[:12]}.json')
    if os.path.exists(out):
        print(json.dumps({'status': 'ERROR', 'error': f'refusing to overwrite {out}'}))
        return 1
    prof = profile(plugin, circ, _draws(a), threads=a.threads, chunk=a.chunk,
                   log=lambda m: print(m, file=sys.stderr, flush=True))
    prof['rerun'] = ' '.join(['kickmix-engine'] + sys.argv[1:])
    write_new(out, prof)
    print(json.dumps({k: prof[k] for k in prof if k != 'draws'} | {'out': out}, indent=1, default=str))
    return 0


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
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == '--via':
        if len(argv) < 2:
            print('--via needs the socket path of a running `kickmix-engine serve`', file=sys.stderr)
            return 2
        from .resident import via
        return via(argv[1], argv[2:])
    ap = argparse.ArgumentParser(prog='kickmix-engine', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    v = sub.add_parser('verify', help='verify a circuit like the official verifier, with the engine as simulator')
    v.add_argument('task', choices=('dqi', 'invsqrt', 'dfthcsos'))
    v.add_argument('circuit_file', help='.kmb circuit file (dfthcsos: the submission directory, or one circuit file)')
    v.add_argument('--verifier', default=None, help='official verify_submission_*.py, or for dfthcsos the dfthcsos '
                   'package directory (default: from $KICKMIX_ENGINE_VERIFIERS)')
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
    g = v.add_argument_group('dfthcsos')
    g.add_argument('--factorization', default=None, help='factorization .npz (default: submission.npz in the directory)')
    g.add_argument('--tree-edges', default=None, help='.npy/.npz with the Givens tree (default: submission.npz)')
    g.add_argument('--format', choices=('auto', 'kmb', 'kmx'), default='auto')
    g.add_argument('--shots-per-address', type=int, default=None)
    g.add_argument('--no-fcidump-cache', action='store_true', help='always parse the FCIDUMP (the official way)')
    g.add_argument('--no-low-memory', action='store_true',
                   help='leave the error_2 tensor in the official layout (one more N^4 copy at the peak)')
    g = v.add_argument_group('invsqrt')
    g.add_argument('--n-p', type=int); g.add_argument('--beta', type=Fraction, default=Fraction(1))
    g.add_argument('--precision-bits', type=int, default=24); g.add_argument('--shots-per-x', type=int)
    g.add_argument('--eta', type=int); g.add_argument('--width', type=int); g.add_argument('--max-x', type=int)
    g.add_argument('--system-qubits', type=int)
    tasks = ('dqi', 'invsqrt', 'dfthcsos', 'generic')
    for name, help_ in (('diff', 'compare two circuits: structural, semantic (paired run), predicted cost'),
                        ('profile', "record a base circuit's per-batch Toffolis and failing shots over a draw set")):
        p = sub.add_parser(name, help=help_)
        p.add_argument('task', choices=tasks)
        p.add_argument('base', help='.kmb circuit (the base)')
        if name == 'diff':
            p.add_argument('candidate', help='.kmb circuit to compare against the base')
            p.add_argument('--profile', default=None, help="the base's profile JSON (enables the cost prediction)")
            p.add_argument('--n', type=int, default=16, help='paired batches on the fixed schedule (default 16)')
            p.add_argument('--report-n', default=None, help='comma list of schedule prefixes to report too')
            p.add_argument('--sweep', default=None, help="candidate's stored full sweep (gate JSON or draws CSV) to compare")
            p.add_argument('--no-structural', action='store_true')
            p.add_argument('--no-semantic', action='store_true')
            p.add_argument('--no-state', action='store_true', help='skip the per-shot interface-state comparison (no dump)')
            p.add_argument('--show-hunks', type=int, default=50, help='hunks and disagreements printed (default 50)')
            p.add_argument('--full', action='store_true', help='print every hunk, paired unit and per-draw prediction')
        else:
            p.add_argument('--chunk', type=int, default=100, help='batches per engine call')
        p.add_argument('--out', default=None, help='diff: JSON file to write (new file only); profile: directory')
        p.add_argument('--draws', type=int, default=1, help='draws in the draw set (default 1; the project uses 48)')
        p.add_argument('--anchor', default=None, help="hash the draw seeds are derived from (default: the base's hash)")
        p.add_argument('--salt-format', default='sweep-{}', help="salt per draw, e.g. 'sweep-{}' ('' = the official default seed)")
        p.add_argument('--inputs', default=None, help='generic: comma list of input registers (default: all)')
        p.add_argument('--batches-per-draw', type=int, default=None, help='generic: batches per draw (default 16)')
        p.add_argument('--verifier', default=None)
        p.add_argument('--threads', type=int, default=4)
        p.add_argument('--cache-dir', default=None)
        p.add_argument('--no-cache', action='store_true')
    c = sub.add_parser('cache', help='inspect or clear the program cache')
    c.add_argument('action', choices=('info', 'prune', 'clear'))
    c.add_argument('--cache-dir', default=None)
    d = sub.add_parser('doctor', help='report CPU features, kernels, versions, cache and verifier status')
    d.add_argument('--json', action='store_true', help='one JSON object (for agents and automation)')
    d.add_argument('--manifest', default=None, help='release-manifest.json (path or URL): name the matching wheel')
    s = sub.add_parser('serve', help='keep this engine process resident and serve commands sent with --via')
    s.add_argument('--socket', required=True, help='Unix socket path to listen on (created mode 0600)')
    s.add_argument('--max-programs', type=int, default=8, help='compiled programs kept in memory (LRU; default 8)')
    a = ap.parse_args(argv)
    if a.cmd == 'serve':
        from .resident import serve
        return serve(a.socket, max_programs=a.max_programs)
    if a.cmd == 'doctor':
        from .doctor import main as doctor_main
        return doctor_main(a)
    if a.cmd == 'diff':
        return cmd_diff(a)
    if a.cmd == 'profile':
        return cmd_profile(a)
    return cmd_verify(a) if a.cmd == 'verify' else cmd_cache(a)


if __name__ == '__main__':
    sys.exit(main())
