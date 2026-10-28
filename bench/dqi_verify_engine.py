"""Engine DQI verification CLI (mirrors the official CLI's JSON output). Usage:
dqi_verify_engine.py circuit.kmb [--salt S] [--threads T] [--sampler engine|official] [--samplers P] [--group G]
(M3 bench entry point; the release CLI is `kickmix-engine verify dqi`.)"""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'python'))
from kickmix_engine import dqi

ap = argparse.ArgumentParser()
ap.add_argument('circuit_file'); ap.add_argument('--salt', default=''); ap.add_argument('--threads', type=int, default=1)
ap.add_argument('--sampler', default='engine'); ap.add_argument('--samplers', type=int, default=None); ap.add_argument('--group', type=int, default=None)
a = ap.parse_args()
task = dqi.DQITask(a.circuit_file)
try:
    out = task.verify(salt=a.salt, threads=a.threads, sampler=a.sampler, samplers=a.samplers, group=a.group)
except (AssertionError, ValueError, RuntimeError, OSError) as ex:
    print(json.dumps({'status': 'FAIL', 'error': str(ex)}, indent=2)); sys.exit(1)
out['engine'] = dict(threads=a.threads, sampler=a.sampler, samplers=a.samplers, group=a.group, wall_in_verify=task.wall,
                     prepare_s=task.t_prepare, program_bytes=task.prog.nbytes)
print(json.dumps(out, indent=2))
