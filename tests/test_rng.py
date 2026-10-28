"""RNG chain: the engine's per-simulator ChaCha8 keys against a C++ standard-library reference (tests/chachakeys.cc).

- `test_keys_match_stored_reference`: against tests/data/rng_keys_ref.npz (written by that program; no compiler needed);
- `test_keys_match_cpp_reference`: builds tests/chachakeys.cc with the system C++ compiler ($CXX, else c++) and
  compares on fresh random seeds; skipped when no compiler is found.
Written by an AI model (Claude) under the owner's direction.
"""
import os
import shutil
import subprocess

import numpy as np
import pytest

import kickmix_engine as ke

HERE = os.path.dirname(os.path.abspath(__file__))
REF = os.path.join(HERE, 'data', 'rng_keys_ref.npz')
SRC = os.path.join(HERE, 'chachakeys.cc')


def engine_keys(seed, nsims):
    return np.asarray(ke.derive_keys([int(w) for w in ke.seed_words(seed)], nsims)).reshape(nsims, 11, 8)


def test_keys_match_stored_reference():
    ref = np.load(REF)
    for seed_words, keys in zip(ref['seed_words'], ref['keys']):
        seed = sum(int(w) << (32 * i) for i, w in enumerate(seed_words))
        assert np.array_equal(engine_keys(seed, keys.shape[0]), keys)


def _compiler():
    cxx = os.environ.get('CXX') or shutil.which('c++') or shutil.which('clang++') or shutil.which('g++')
    return cxx if cxx and shutil.which(cxx) else None


def test_keys_match_cpp_reference(tmp_path):
    cxx = _compiler()
    if cxx is None:
        pytest.skip('no C++ compiler found (set CXX); the stored-reference test still runs')
    exe = str(tmp_path / 'chachakeys')
    subprocess.run([cxx, '-O1', '-std=c++17', '-o', exe, SRC], check=True)
    rng = np.random.default_rng(20261028)
    for trial in range(6):
        seed = int.from_bytes(rng.bytes(32), 'little') if trial else 0
        out = str(tmp_path / f'k{trial}.bin')
        subprocess.run([exe, '5', out] + [str(int(w)) for w in ke.seed_words(seed)], check=True)
        ref = np.fromfile(out, dtype='<u4').reshape(5, 11, 8)
        assert np.array_equal(engine_keys(seed, 5), ref), trial
