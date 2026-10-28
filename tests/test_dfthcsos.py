"""Tests for 0.5.0: `run_sequence` (the dfthcsos simulator loop), the in-place C-order Gram tensor, the FCIDUMP cache
and the dfthcsos adapter. pytest tests/test_dfthcsos.py

- Without kickmix or the competition files (run in CI): `run_sequence` against a chain of independent `run` calls in
  which the RNG position is carried by dummy `R` ops on an extra always-|0> qubit (the method of the 2026-10-28
  feasibility study); extra lanes of a partly used 256-shot sim; the in-place transpose.
- With kickmix (skipped otherwise): `run_sequence` against kickmix's Simulator driven the verifier's way (do,
  clear_for_shot, write back), per shot and per operation count, with batch sizes that are and are not multiples of 256.
- With the official verifiers (KICKMIX_ENGINE_VERIFIERS, or KICKMIX_ENGINE_DFTHCSOS_VERIFIER; skipped otherwise): the
  FCIDUMP cache round trip and rejection of a corrupted entry; the adapter's summary against the official verify() on
  KICKMIX_ENGINE_DFTHCSOS_FAMILY (default: the competition's sample_submission/dfthcsos next to verifiers/, if present).
Written by an AI model (Claude) under the owner's direction.
"""
import json
import os

import numpy as np
import pytest

import kickmix_engine as ke

NQ, NX = 14, 6  # qubits; the first NX are the carried register


def random_circuit(rng, nops=400, nbits=6):
    lines = []
    for _ in range(nops):
        k = rng.integers(0, 10)
        q = rng.choice(NQ, 3, replace=False)
        b = int(rng.integers(0, nbits))
        if k == 0:
            lines.append(f'X q{q[0]}')
        elif k in (1, 2):
            lines.append(f'CX q{q[0]} q{q[1]}')
        elif k == 3:
            lines.append(f'CCX q{q[0]} q{q[1]} q{q[2]}')
        elif k == 4:
            lines.append(f'HMR q{q[0]} b{b}')
        elif k == 5:
            lines.append(f'R q{q[0]}')
        elif k == 6:
            lines.append(f'CCX q{q[0]} q{q[1]} q{q[2]} if b{b}')
        elif k == 7:
            lines.append(f'CZ q{q[0]} q{q[1]} if b{b}')
        elif k == 8:
            lines.append(f'CCZ q{q[0]} q{q[1]} q{q[2]}')
        else:
            lines.append(f'X q{q[0]} if b{b}')
    return '\n'.join(lines) + '\n'


def programs(seed=7, steps=5):
    rng = np.random.default_rng(seed)
    texts = [random_circuit(rng) for _ in range(steps)]
    return texts, [ke.compile_text(t, NQ) for t in texts]


def inputs(shots, seed=3):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 1 << NX, size=(shots, 1), dtype=np.uint64)


def test_run_sequence_matches_prefixed_chain():
    texts, progs = programs()
    shots, seed = 512, 0x1234_5678_9ABC_DEF0_1122_3344_5566_7788
    x0 = inputs(shots)
    xreg = list(range(NX))
    seq = ke.run_sequence(progs, seed, x0, xreg, threads=2)
    x, d = x0.reshape(1, 1, shots), 0
    for i, (t, p) in enumerate(zip(texts, progs)):
        pre = ke.compile_text(f'R q{NQ}\n' * d + t, NQ + 1)
        o = ke.run(pre, [ke.seed_words(seed)], x, xreg, s=shots // 256, threads=2)
        assert np.array_equal(o['x_out'].reshape(-1), seq['x_out'][i, :, 0]), i
        assert np.array_equal(o['scratch_dirty'].reshape(-1).astype(bool), seq['dirty'][i]), i
        assert np.array_equal(o['phase_units'].reshape(-1) != 0, seq['phase'][i]), i
        assert int(o['toffoli_count'].sum()) == int(seq['toffoli_count'][i]), i
        kc = o['kind_counts'].sum(axis=0)
        kc[ke.KINDS.index('R')] -= d * shots
        assert np.array_equal(kc, seq['kind_counts'][i]), i
        x, d = o['x_out'], d + p.num_rand
    # Negative control: without the RNG carried over, later steps differ.
    o = ke.run(progs[-1], [ke.seed_words(seed)], seq['x_out'][-2, :, 0].reshape(1, 1, shots), xreg, s=shots // 256)
    assert not (np.array_equal(o['x_out'].reshape(-1), seq['x_out'][-1, :, 0])
                and np.array_equal(o['phase_units'].reshape(-1) != 0, seq['phase'][-1]))


def test_run_sequence_partial_sim_lanes_follow_the_full_run():
    _, progs = programs(seed=11)
    seed = 99
    full = ke.run_sequence(progs, seed, np.vstack([inputs(300), np.zeros((212, 1), np.uint64)]), list(range(NX)))
    part = ke.run_sequence(progs, seed, inputs(300), list(range(NX)))
    for k in ('x_out', 'dirty', 'phase'):
        assert np.array_equal(full[k][:, :300], part[k]), k
    # Unconditioned CCZ executes on every shot: counted for the 300 real shots only.
    ccz = ke.KINDS.index('CCZ')
    assert np.array_equal(part['kind_counts'][:, ccz] * 512, full['kind_counts'][:, ccz] * 300)


def test_run_sequence_wide_register():
    # 70 carried qubits -> two words per shot; qubits 64.. live in the second word.
    text = ''.join(f'CX q{i} q{i + 70}\nCX q{i + 70} q{i}\nCX q{i} q{i + 70}\n' for i in range(70))  # swap out and back
    p = ke.compile_text(text, 140)
    rng = np.random.default_rng(1)
    x0 = rng.integers(0, 2**63, size=(256, 2), dtype=np.uint64)
    x0[:, 1] &= np.uint64((1 << 6) - 1)
    o = ke.run_sequence([p, p], 5, x0, list(range(70)))
    assert o['x_out'].shape == (2, 256, 2)
    # the circuit moves the register into the workspace: the register reads zero and the workspace is dirty exactly
    # where the input was nonzero; step 1 starts from the (zero) register and a cleared workspace
    assert not o['x_out'].any()
    assert np.array_equal(o['dirty'][0], (x0 != 0).any(axis=1)) and not o['dirty'][1].any()


def test_run_sequence_rejects_bad_inputs():
    _, progs = programs(steps=1)
    with pytest.raises(ValueError):
        ke.run_sequence(progs, 1, np.zeros((10, 2), np.uint64), list(range(NX)))
    with pytest.raises(ValueError):
        ke.run_sequence(progs, 1, np.zeros((10, 1), np.uint64), [0, 0])


def test_transpose_square_inplace_and_c_order():
    from kickmix_engine.dfthcsos import c_order_inplace, transpose_square_inplace
    rng = np.random.default_rng(0)
    for n in (1, 5, 255, 256, 257, 600):
        a = rng.standard_normal((n, n))
        b = a.copy()
        transpose_square_inplace(b, block=64)
        assert np.array_equal(b, a.T)
    X = rng.standard_normal((3, 4, 9, 9))
    G = np.einsum('rcpq,rcst->pqst', X, X, optimize=True)
    ref = np.ascontiguousarray(G)
    norm_ref = float(np.linalg.norm(G.ravel()))
    H, how = c_order_inplace(G)
    assert H.flags['C_CONTIGUOUS'] and np.array_equal(H, ref)
    assert float(np.linalg.norm(H.ravel())) == norm_ref
    if not G.flags['C_CONTIGUOUS']:
        assert how == 'transposed in place' and np.shares_memory(H, G)


def test_run_sequence_matches_kickmix():
    km = pytest.importorskip('kickmix')
    texts, progs = programs(seed=5, steps=4)
    circuits = [km.Circuit(t) for t in texts]
    xreg = list(range(NX))
    for shots in (300, 512):
        seed = 0xC0FFEE + shots
        x0 = inputs(shots, seed=shots)
        seq = ke.run_sequence(progs, seed, x0, xreg, threads=2)
        sim = km.Simulator(batch_size=shots, seed=seed, counting_operations=True, debug_print=False)
        reg = km.array([km.q(i) for i in xreg])
        bits = ((x0[:, 0][None, :] >> np.arange(NX, dtype=np.uint64)[:, None]) & np.uint64(1)).astype(bool)
        sim.write_across_shots(reg, bits)
        prev = {}
        for i, c in enumerate(circuits):
            if i:
                keep = sim.read_across_shots(reg)
                sim.clear_for_shot()
                sim.write_across_shots(reg, keep)
            sim.do(c)
            got = sim.read_across_shots(reg)
            eng = ((seq['x_out'][i, :, 0][None, :] >> np.arange(NX, dtype=np.uint64)[:, None]) & np.uint64(1)).astype(bool)
            assert np.array_equal(got, eng), (shots, i)
            work = sim.read_across_shots(km.array([km.q(j) for j in range(NX, NQ)])).any(axis=0)
            assert np.array_equal(work, seq['dirty'][i]), (shots, i)
            assert np.array_equal(sim.read_phase_flipped_across_shots(), seq['phase'][i]), (shots, i)
            st = {str(k): int(v) for k, v in dict(sim.operation_statistics()).items()}
            for k, v in st.items():
                if k in ke.KINDS:
                    assert v - prev.get(k, 0) == int(seq['kind_counts'][i, ke.KINDS.index(k)]), (shots, i, k)
            prev = st


# ---- with the official verifiers -----------------------------------------------------------------------------

def _verifier():
    from kickmix_engine.dfthcsos import default_verifier
    p = default_verifier()
    if not p or not os.path.isdir(p):
        pytest.skip('official dfthcsos verifier not found (set KICKMIX_ENGINE_VERIFIERS)')
    pytest.importorskip('kickmix')
    pytest.importorskip('scipy')
    return p


def _family(vdir):
    fam = os.environ.get('KICKMIX_ENGINE_DFTHCSOS_FAMILY')
    if not fam:
        cand = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(vdir))), 'sample_submission', 'dfthcsos')
        fam = cand if os.path.isdir(cand) else None
    if not fam or not os.path.isdir(fam):
        pytest.skip('no dfthcsos family (set KICKMIX_ENGINE_DFTHCSOS_FAMILY)')
    return fam


def test_fcidump_cache_roundtrip(tmp_path):
    from kickmix_engine import dfthcsos
    off = dfthcsos.load_verifier(_verifier())
    path = off.V.check_fcidump()
    info1, info2, info3 = {}, {}, {}
    r1 = dfthcsos.load_reference_cached(off, path, directory=str(tmp_path), info=info1)
    assert info1['fcidump'] == 'parsed' and info1['fcidump_cache_written']
    r2 = dfthcsos.load_reference_cached(off, path, directory=str(tmp_path), info=info2)
    assert info2['fcidump'] == 'cache'
    for k in r1:
        if isinstance(r1[k], np.ndarray):
            assert r1[k].dtype == r2[k].dtype and np.array_equal(r1[k], r2[k]), k
        else:
            assert r1[k] == r2[k] and type(r1[k]) is type(r2[k]), k
    assert list(r1) == list(r2)
    # A corrupted entry is rejected and the file is parsed again.
    cp = info1['fcidump_cache_written']
    raw = bytearray(open(cp, 'rb').read())
    raw[len(raw) // 2] ^= 0x01
    open(cp, 'wb').write(raw)
    dfthcsos.load_reference_cached(off, path, directory=str(tmp_path), info=info3)
    assert info3['fcidump'] == 'parsed'


def test_task_parity_against_official(tmp_path):
    from kickmix_engine import dfthcsos
    from kickmix_engine.cache import ProgramCache
    vdir = _verifier()
    fam = _family(vdir)
    off = dfthcsos.load_verifier(vdir)
    official = json.loads(json.dumps(off.V.verify(fam)))
    for _ in range(2):  # cold, then warm caches
        info = {}
        eng = dfthcsos.verify(fam, verifier=vdir, threads=2, cache=ProgramCache(str(tmp_path)), info=info)
        assert json.loads(json.dumps(eng)) == official
    assert info['fcidump'] == 'cache'
