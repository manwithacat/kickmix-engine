"""Lockstep pairs give the single-sim results on every driver: `run_batches` (even and odd batch counts, dumps),
`run_sequence` (several programs, a partial last sim, lane-restricted counts) and `run` (chunks of several batches),
on the fixture circuit. No kickmix needed. Written by an AI model (Claude) under the owner's direction.
"""
import os

import numpy as np
import pytest

import kickmix_engine as ke

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
KMB = os.path.join(DATA, 'random_ops.kmb')


def same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert np.array_equal(a[k], b[k]), k


@pytest.mark.parametrize('nb', [1, 2, 5])
def test_run_batches_lockstep_equals_single(nb):
    prog = ke.compile_kmb(KMB)
    rng = np.random.default_rng(nb)
    nq = prog.num_qubits
    qin = rng.integers(0, 1 << 64, size=(nb, nq, 4), dtype=np.uint64)
    bin_ = rng.integers(0, 1 << 64, size=(nb, prog.num_bits, 4), dtype=np.uint64)
    seeds = [ke.seed_words(100 + i) for i in range(nb)]
    args = (prog, seeds, list(range(nq)), qin, list(range(prog.num_bits)), bin_)
    a = ke.run_batches(*args, threads=2, dump=True, lockstep=True)
    b = ke.run_batches(*args, threads=1, dump=True, lockstep=False)
    same(a, b)
    assert prog.stream_stats()['uni_runs'] >= 1


@pytest.mark.parametrize('shots', [256, 700, 1024])
def test_run_sequence_lockstep_equals_single(shots):
    prog = ke.compile_kmb(KMB)
    rng = np.random.default_rng(shots)
    xreg = list(range(0, min(70, prog.num_qubits)))
    xw = max(1, -(-len(xreg) // 64))
    x0 = rng.integers(0, 1 << 64, size=(shots, xw), dtype=np.uint64)
    a = ke.run_sequence([prog, prog, prog], 777, x0, xreg, threads=2, dump=True, lockstep=True)
    b = ke.run_sequence([prog, prog, prog], 777, x0, xreg, threads=1, dump=True, lockstep=False)
    same(a, b)


@pytest.mark.parametrize('n_chunks', [1, 2, 3])
def test_run_lockstep_equals_single(n_chunks):
    prog = ke.compile_kmb(KMB)
    rng = np.random.default_rng(n_chunks)
    xreg = list(range(8))
    xs = rng.integers(0, 256, size=(n_chunks, 3, 256), dtype=np.uint64)
    seeds = [ke.seed_words(5 + i) for i in range(n_chunks)]
    a = ke.run(prog, seeds, xs, xreg, s=1, threads=2, dump=True, lockstep=True)
    b = ke.run(prog, seeds, xs, xreg, s=1, threads=1, dump=True, lockstep=False)
    same(a, b)
