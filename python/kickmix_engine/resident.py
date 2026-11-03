"""Resident engine: a long-lived process that serves `kickmix-engine` commands over a Unix socket.

    kickmix-engine serve --socket /tmp/ke.sock [--max-programs 8]        # the server (foreground)
    kickmix-engine --via /tmp/ke.sock diff dqi BASE.kmb CAND.kmb ...     # any command, served by that process
    kickmix-engine --via /tmp/ke.sock ping | stats | stop

Each request is one JSON line ``{"argv": [...]}`` (or ``{"cmd": "ping" | "stats" | "stop"}``); the reply is one
JSON line ``{"rc", "stdout", "stderr", "seconds", "memory"}``. The server runs the ordinary CLI ``main(argv)`` with
stdout and stderr captured, so a served command prints exactly what the one-shot CLI prints and exits with the same
code; only the timing fields inside the JSON differ. What the process keeps between requests (``cache.Memory``):
compiled programs by content hash, the hashes of files it hashed itself (an unchanged file is not re-read), task
plug-ins with their reference tables, and parsed profile JSONs. Requests are served one at a time (the engine's
thread pool is the shared resource); clients queue on the socket. Written by an AI model (Claude) under the
owner's direction.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import sys
import time


def _recv_line(conn) -> bytes:
    buf = bytearray()
    while True:
        b = conn.recv(1 << 16)
        if not b:
            break
        buf += b
        if buf.endswith(b'\n'):
            break
    return bytes(buf)


def handle(req: dict, started: float) -> tuple[dict, bool]:
    """Serve one request in this process; returns (reply, keep_running)."""
    from .cache import memory
    mem = memory()
    t0 = time.perf_counter()
    cmd = req.get('cmd')
    if cmd == 'ping':
        return dict(rc=0, stdout='pong\n', stderr='', seconds=0.0, pid=os.getpid(), uptime_s=time.time() - started), True
    if cmd == 'stats':
        return dict(rc=0, stdout=json.dumps(mem.describe() if mem else {}, indent=1) + '\n', stderr='', seconds=0.0), True
    if cmd == 'stop':
        return dict(rc=0, stdout='stopping\n', stderr='', seconds=0.0), False
    argv = req.get('argv')
    if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
        return dict(rc=2, stdout='', stderr='request needs "argv": [str, ...] or "cmd"\n', seconds=0.0), True
    if argv and argv[0] == 'serve':
        return dict(rc=2, stdout='', stderr='a served process cannot serve\n', seconds=0.0), True
    from .cli import main
    out, err = io.StringIO(), io.StringIO()
    if mem is not None:
        mem.stats['requests'] += 1
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = main(argv)
            except SystemExit as ex:  # argparse errors and --help
                rc = ex.code if isinstance(ex.code, int) else (0 if ex.code is None else 1)
            except Exception as ex:  # noqa: BLE001 - report the failure to the client, keep serving
                import traceback
                traceback.print_exc()
                print(json.dumps({'status': 'ERROR', 'error': f'{type(ex).__name__}: {ex}'}))
                rc = 1
    finally:
        pass
    return dict(rc=int(rc or 0), stdout=out.getvalue(), stderr=err.getvalue(), seconds=time.perf_counter() - t0,
                memory=mem.describe() if mem else None), True


def serve(socket_path: str, *, max_programs: int = 8, log=None, ready=None, max_requests: int | None = None) -> int:
    """Serve requests on `socket_path` until a stop request (or `max_requests`); returns 0."""
    from .cache import disable_memory, enable_memory
    from . import kernel_info
    kernel_info()  # select the kernel once; fail early if KICKMIX_ENGINE_KERNEL is unusable
    enable_memory(max_programs=max_programs)
    log = log or (lambda m: print(m, file=sys.stderr, flush=True))
    if os.path.exists(socket_path):
        os.unlink(socket_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        srv.bind(socket_path)
    except OSError as ex:
        disable_memory()
        raise OSError(f'cannot bind {socket_path!r}: {ex} (AF_UNIX paths are limited to about 100 bytes)') from ex
    os.chmod(socket_path, 0o600)
    srv.listen(16)
    started = time.time()
    log(f'kickmix-engine serve: pid {os.getpid()} listening on {socket_path}')
    if ready is not None:
        ready.set()
    served = 0
    try:
        running = True
        while running and (max_requests is None or served < max_requests):
            conn, _ = srv.accept()
            with conn:
                try:
                    raw = _recv_line(conn)
                    req = json.loads(raw.decode('utf-8')) if raw.strip() else {}
                except ValueError as ex:
                    reply, running = dict(rc=2, stdout='', stderr=f'bad request: {ex}\n', seconds=0.0), True
                else:
                    reply, running = handle(req, started)
                served += 1
                try:
                    conn.sendall((json.dumps(reply) + '\n').encode('utf-8'))
                except OSError:
                    pass
                what = req.get('cmd') or ' '.join(req.get('argv', []))[:120]
                log(f'served #{served} rc={reply["rc"]} {reply["seconds"]:.2f}s: {what}')
    finally:
        srv.close()
        try:
            os.unlink(socket_path)
        except OSError:
            pass
        disable_memory()
    return 0


class Client:
    """Sends requests to a resident engine."""

    def __init__(self, socket_path: str, timeout: float | None = None):
        self.socket_path = socket_path
        self.timeout = timeout

    def request(self, req: dict) -> dict:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self.timeout)
        try:
            s.connect(self.socket_path)
            s.sendall((json.dumps(req) + '\n').encode('utf-8'))
            try:
                s.shutdown(socket.SHUT_WR)   # the server reads to the newline; a fast reply may already have closed
            except OSError:
                pass
            buf = bytearray()
            while True:
                b = s.recv(1 << 20)
                if not b:
                    break
                buf += b
        finally:
            s.close()
        if not buf:
            raise ConnectionError('resident engine closed the connection without a reply')
        return json.loads(buf.decode('utf-8'))

    def call(self, argv: list[str]) -> dict:
        return self.request({'argv': list(argv)})

    def json(self, argv: list[str]):
        """The served command's stdout parsed as JSON (raises on a non-zero exit)."""
        r = self.call(argv)
        if r['rc'] != 0:
            raise RuntimeError(f'resident command failed (rc {r["rc"]}): {r["stderr"][-2000:]}{r["stdout"][-2000:]}')
        return json.loads(r['stdout'])

    def ping(self) -> bool:
        try:
            return self.request({'cmd': 'ping'})['rc'] == 0
        except (OSError, ValueError, ConnectionError):
            return False

    def stop(self) -> dict:
        return self.request({'cmd': 'stop'})


def via(socket_path: str, argv: list[str]) -> int:
    """Run `argv` on the resident engine at `socket_path`, relaying its stdout, stderr and exit code."""
    c = Client(socket_path)
    if argv and argv[0] in ('ping', 'stats', 'stop') and len(argv) == 1:
        r = c.request({'cmd': argv[0]})
    else:
        r = c.call(argv)
    sys.stdout.write(r['stdout'])
    sys.stdout.flush()
    if r['stderr']:
        sys.stderr.write(r['stderr'])
        sys.stderr.flush()
    return r['rc']
