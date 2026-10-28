"""On-disk program cache: compiled programs keyed by circuit content (SHA-256 of the file bytes) and engine build.

Written by an AI model (Claude) under the owner's direction.

Layout under the cache directory (``KICKMIX_ENGINE_CACHE``, else ``$XDG_CACHE_HOME/kickmix-engine``, else
``~/.cache/kickmix-engine``):
    programs/<circuit sha256>.<engine key>.kmxprog   compiled program (checksummed, bounds-validated on load)
    hints/<sha256 of the real path>.json              {size, mtime_ns, sha256}: lets a warm run start loading the
                                                      program while the file is still being hashed. A hint is only
                                                      a guess; the content hash computed for the run decides.
The engine key is the crate version, the cache format version and a hash of the extension binary, so any rebuild
invalidates old entries. Writers write a unique temporary file in the same directory and ``os.replace`` it into
place, so concurrent readers see either a complete file or none; concurrent writers of the same key are harmless
(identical content, last rename wins). A file that fails to load is treated as a miss and replaced.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import tempfile
import time

from . import _engine

_KEY = None


def cache_dir(path=None) -> str:
    if path:
        return os.path.abspath(path)
    env = os.environ.get('KICKMIX_ENGINE_CACHE')
    if env:
        return os.path.abspath(env)
    base = os.environ.get('XDG_CACHE_HOME') or os.path.join(os.path.expanduser('~'), '.cache')
    return os.path.join(base, 'kickmix-engine')


def engine_key() -> str:
    global _KEY
    if _KEY is None:
        version, fmt = _engine.engine_version()
        with open(_engine.__file__, 'rb') as f:
            so = hashlib.sha256(f.read()).hexdigest()[:16]
        _KEY = f'v{version}-f{fmt}-{so}'
    return _KEY


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(1 << 24)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _atomic_write_bytes(path, data: bytes):
    d = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix='.tmp-', dir=d)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class ProgramCache:
    def __init__(self, path=None, enabled=True):
        self.dir = cache_dir(path)
        self.enabled = enabled
        self.events = []  # (event, seconds) for reporting

    def program_path(self, sha: str) -> str:
        return os.path.join(self.dir, 'programs', f'{sha}.{engine_key()}.kmxprog')

    def _hint_path(self, kmb_path) -> str:
        rp = os.path.realpath(kmb_path)
        return os.path.join(self.dir, 'hints', hashlib.sha256(rp.encode()).hexdigest() + '.json')

    def hint(self, kmb_path):
        try:
            st = os.stat(kmb_path)
            with open(self._hint_path(kmb_path)) as f:
                h = json.load(f)
            if h.get('size') == st.st_size and h.get('mtime_ns') == st.st_mtime_ns:
                return h.get('sha256')
        except (OSError, ValueError):
            pass
        return None

    def write_hint(self, kmb_path, sha):
        try:
            st = os.stat(kmb_path)
            os.makedirs(os.path.join(self.dir, 'hints'), exist_ok=True)
            _atomic_write_bytes(self._hint_path(kmb_path),
                                json.dumps(dict(size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha)).encode())
        except OSError:
            pass

    def load(self, sha):
        """The cached program for this content hash, or None."""
        p = self.program_path(sha)
        if not self.enabled or not os.path.exists(p):
            return None
        t0 = time.perf_counter()
        try:
            prog = _engine.Program.load(p)
        except (ValueError, OSError) as ex:
            self.events.append(('load_failed', str(ex)))
            return None
        self.events.append(('hit', time.perf_counter() - t0))
        return prog

    def save(self, sha, prog):
        if not self.enabled:
            return
        t0 = time.perf_counter()
        d = os.path.join(self.dir, 'programs')
        os.makedirs(d, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix='.tmp-', suffix='.kmxprog', dir=d)
        os.close(fd)
        try:
            prog.save(tmp)
            os.replace(tmp, self.program_path(sha))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.events.append(('saved', time.perf_counter() - t0))

    def entries(self):
        d = os.path.join(self.dir, 'programs')
        if not os.path.isdir(d):
            return []
        return sorted((f, os.path.getsize(os.path.join(d, f))) for f in os.listdir(d) if f.endswith('.kmxprog'))

    def prune(self):
        """Remove programs written by other engine builds (their key no longer matches). Returns the count."""
        d, n, key = os.path.join(self.dir, 'programs'), 0, engine_key()
        for f, _ in self.entries():
            if not f.endswith(f'.{key}.kmxprog'):
                try:
                    os.unlink(os.path.join(d, f))
                    n += 1
                except OSError:
                    pass
        return n

    def clear(self):
        n = 0
        for sub in ('programs', 'hints'):
            d = os.path.join(self.dir, sub)
            if os.path.isdir(d):
                for f in os.listdir(d):
                    try:
                        os.unlink(os.path.join(d, f))
                        n += 1
                    except OSError:
                        pass
        return n


def load_program(kmb_path, *, cache: ProgramCache | None = None, executor=None):
    """(program, content sha256, info) for a .kmb file, using the cache when enabled.

    Warm path: a valid path hint starts loading the cached program while the file is hashed; the computed hash must
    equal the hint, otherwise the loaded program is discarded. Without a usable hint the file is compiled (GIL
    released) while it is hashed. The caller saves a compiled program with `cache.save` (see info['save_needed']).
    info: dict(source='cache'|'compiled', hash_s, wall_s, save_needed).
    """
    from . import compile_kmb
    cache = cache or ProgramCache(enabled=False)
    own = executor is None
    ex = executor or concurrent.futures.ThreadPoolExecutor(1)
    t0 = time.perf_counter()
    info = {}
    try:
        guess = cache.hint(kmb_path) if cache.enabled else None
        if guess and os.path.exists(cache.program_path(guess)):
            fut, kind = ex.submit(cache.load, guess), 'cache'
        else:
            fut, kind = ex.submit(compile_kmb, kmb_path), 'compiled'
        th = time.perf_counter()
        sha = file_sha256(kmb_path)
        info['hash_s'] = time.perf_counter() - th
        prog = fut.result()
        if kind == 'cache' and (prog is None or sha != guess):
            prog, kind = compile_kmb(kmb_path), 'compiled'
        info['source'] = kind
        info['save_needed'] = cache.enabled and kind == 'compiled' and not os.path.exists(cache.program_path(sha))
        if cache.enabled:
            cache.write_hint(kmb_path, sha)
        info['wall_s'] = time.perf_counter() - t0
        return prog, sha, info
    finally:
        if own:
            ex.shutdown(wait=True)
