"""Resident engine (`kickmix-engine serve` / `--via`): a served command prints what the one-shot CLI prints (timing
fields aside) and exits with the same code; the second request for the same files is answered from memory. Uses
the fixture circuit and its one-op mutant with the generic plug-in, so neither kickmix nor the competition files are
needed. Written by an AI model (Claude) under the owner's direction.
"""
import contextlib
import io
import json
import os
import threading

import pytest

from kickmix_engine import cli
from kickmix_engine.resident import Client, serve
from test_diff import KMB, mutate_one_op

TIMING = {'seconds', 'load_s', 'wall_s', 'prepare_s', 'total_s', 'memory'}


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in TIMING}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def one_shot(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


@pytest.fixture
def server(tmp_path):
    # AF_UNIX paths are limited to about 100 bytes; pytest's tmp_path is longer than that on macOS.
    sock = f'/tmp/ke-test-{os.getpid()}-{os.urandom(3).hex()}.sock'
    ready = threading.Event()
    logs = []
    t = threading.Thread(target=serve, args=(sock,), kwargs=dict(log=logs.append, ready=ready, max_programs=4), daemon=True)
    t.start()
    assert ready.wait(10)
    yield sock, logs
    try:
        Client(sock).stop()
    except OSError:
        pass
    t.join(10)
    assert not os.path.exists(sock)


def test_served_diff_equals_one_shot_and_hits_memory(tmp_path, server):
    sock, logs = server
    mut = str(tmp_path / 'mut.kmb')
    mutate_one_op(KMB, mut, which=3)
    argv = ['diff', 'generic', KMB, mut, '--threads', '1', '--draws', '1', '--n', '4', '--no-structural']
    rc, out, err = one_shot(argv)
    assert rc == 0
    c = Client(sock)
    assert c.ping()
    r1 = c.call(argv)
    r2 = c.call(argv)
    assert r1['rc'] == r2['rc'] == 0
    assert strip(json.loads(out)) == strip(json.loads(r1['stdout'])) == strip(json.loads(r2['stdout']))
    assert r2['memory']['program_hits'] >= 2 and r2['memory']['plugin_hits'] >= 1
    assert r2['memory']['requests'] == 2
    # a bad command reports argparse's exit code and message without killing the server
    bad = c.call(['diff'])
    assert bad['rc'] == 2 and 'usage' in bad['stderr']
    assert c.ping()
    stats = c.request({'cmd': 'stats'})
    assert json.loads(stats['stdout'])['requests'] == 3


def test_via_relays_output_and_exit_code(tmp_path, server, capsys):
    sock, _ = server
    mut = str(tmp_path / 'mut2.kmb')
    mutate_one_op(KMB, mut, which=1)
    argv = ['diff', 'generic', KMB, mut, '--threads', '1', '--draws', '1', '--n', '2', '--no-structural']
    rc_direct, out_direct, _ = one_shot(argv)
    rc = cli.main(['--via', sock] + argv)
    captured = capsys.readouterr()
    assert rc == rc_direct == 0
    assert strip(json.loads(captured.out)) == strip(json.loads(out_direct))
    assert cli.main(['--via', sock, 'ping']) == 0
    assert 'pong' in capsys.readouterr().out
