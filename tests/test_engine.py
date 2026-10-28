"""Engine tests that need neither kickmix nor the competition files (run in CI): pytest tests/test_engine.py

- random-ops parity against a stored kickmix reference (tests/data, built by tests/make_fixtures.py);
- program cache: save/load round trip, corruption and truncation detection, cold then warm `load_program`;
- reaction depth against kickmix's `Circuit.reaction_depth()` on stored random circuits (tests/data, built by
  tests/reaction_depth_parity.py);
- the DQI sampler against a NumPy restatement of the official `sample_shots` (for several parameter sets), and the
  PCG64 stream against NumPy's own `random_raw`.
Written by an AI model (Claude) under the owner's direction.
"""
import json
import os
import shutil

import numpy as np
import pytest

import kickmix_engine as ke
from kickmix_engine import _engine, bools_to_words, words_to_bools
from kickmix_engine.cache import ProgramCache, load_program

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
KMB = os.path.join(DATA, 'random_ops.kmb')


def test_random_ops_matches_kickmix_reference():
    ref = np.load(os.path.join(DATA, 'random_ops_ref.npz'))
    prog = ke.compile_kmb(KMB)
    assert prog.num_ops == int(ref['num_ops'])
    nq = prog.num_qubits
    bit_map = np.asarray(prog.bit_map, dtype=np.int64)
    qin = bools_to_words(ref['qin'])           # [4, nq, 4]
    bin_ = bools_to_words(ref['bin'][:, bit_map])  # touched bits only, compact order
    r = ke.run_batches(prog, ref['seeds'], list(range(nq)), qin, list(range(prog.num_bits)), bin_, threads=2, dump=True)
    assert np.array_equal(words_to_bools(r['dump_q']), ref['qout'])
    assert np.array_equal(words_to_bools(r['dump_bits'])[:, :prog.num_bits], ref['bout'][:, bit_map])
    untouched = np.setdiff1d(np.arange(ref['bin'].shape[1]), bit_map)
    assert np.array_equal(ref['bout'][:, untouched], ref['bin'][:, untouched])
    assert np.array_equal(words_to_bools(r['phase']), ref['phase'])
    kinds = [str(k) for k in ref['count_kinds']]
    got = np.array([[int(r['kind_counts'][i, ke.KINDS.index(k)]) for k in kinds] for i in range(4)])
    assert np.array_equal(got, ref['counts'])
    # Negative control: a different seed changes measured classical bits.
    seeds = ref['seeds'].copy(); seeds[:, 0] ^= 1
    r2 = ke.run_batches(prog, seeds, list(range(nq)), qin, list(range(prog.num_bits)), bin_, threads=2, dump=True)
    assert not np.array_equal(words_to_bools(r2['dump_bits'])[:, :prog.num_bits], ref['bout'][:, bit_map])


def test_program_save_load_roundtrip(tmp_path):
    prog = ke.compile_kmb(KMB)
    p = str(tmp_path / 'p.kmxprog')
    prog.save(p)
    q = _engine.Program.load(p)
    for attr in ('num_qubits', 'num_bits', 'num_ops', 'num_stream_ops', 'num_rand', 'bit_map', 'bit_compact',
                 'orig_num_bits', 'num_info_tags', 'max_stack_depth', 'num_cx_runs', 'num_angles', 'static_counts',
                 'registers', 'reaction_depth'):
        assert getattr(q, attr) == getattr(prog, attr), attr
    raw = bytearray(open(p, 'rb').read())
    bad = bytearray(raw); bad[len(bad) // 2] ^= 0x10
    open(p, 'wb').write(bad)
    with pytest.raises(ValueError):
        _engine.Program.load(p)
    open(p, 'wb').write(raw[:-9])
    with pytest.raises(ValueError):
        _engine.Program.load(p)


def test_cache_cold_then_warm(tmp_path):
    kmb = str(tmp_path / 'c.kmb')
    shutil.copy(KMB, kmb)
    cache = ProgramCache(str(tmp_path / 'cache'))
    p1, sha1, info1 = load_program(kmb, cache=cache)
    assert info1['source'] == 'compiled' and info1['save_needed']
    cache.save(sha1, p1)
    p2, sha2, info2 = load_program(kmb, cache=cache)
    assert info2['source'] == 'cache' and sha2 == sha1 and not info2['save_needed']
    assert p2.static_counts == p1.static_counts and p2.num_stream_ops == p1.num_stream_ops
    # Changing the content invalidates the hint: same size and mtime would be needed to even try the cache.
    with open(kmb, 'r+b') as f:
        f.seek(0, 2); f.write(b'\0')
    with pytest.raises(ValueError):  # the appended byte makes the .kmb invalid, so it must be compiled (and fail)
        load_program(kmb, cache=cache)
    assert len(cache.entries()) == 1


def test_reaction_depth_matches_kickmix_reference():
    with open(os.path.join(DATA, 'reaction_depth_ref.json')) as f:
        ref = json.load(f)
    assert len(ref['circuits']) >= 20
    depths = set()
    for c in ref['circuits']:
        prog = ke.compile_text(c['text'], c['num_qubits'])
        assert prog.reaction_depth == c['reaction_depth'], c['text'][:200]
        depths.add(c['reaction_depth'])
    assert len(depths) > 5  # the fixture is not degenerate


def test_reaction_depth_small_cases():
    # kickmix docstring example, and the operand order (the target of CCX is the last operand).
    assert ke.compile_text('CCX q0 q1 q2\nCCX q2 q3 q4\n', 5).reaction_depth == 2
    assert ke.compile_text('Z_POW q1 0.75\nCCX q2 q3 q1\n', 4).reaction_depth == 2
    assert ke.compile_text('Z_POW q1 1.0\nCX q0 q1\n', 2).reaction_depth == 0
    # CZ moves q2's bit-flip depth into q3's phase depth; HMR reads it (+1) into b0; the conditioned X carries it
    # to q4 and the CCZ adds one more layer. Values checked with kickmix 1.0.0 and the 0.3.0 fork.
    assert ke.compile_text('CCX q0 q1 q2\nHMR q2 b0\nX q3 if b0\nCCZ q3 q4 q5\n', 6).reaction_depth == 2
    assert ke.compile_text('CCX q0 q1 q2\nCZ q2 q3\nHMR q3 b0\nX q4 if b0\nCCZ q4 q5 q6\n', 7).reaction_depth == 3


# ---- DQI sampler -------------------------------------------------------------------------------------------------
# `sample_shots_restated` restates `sample_shots` from the competition's official DQI verifier
# (verify_submission_dqi.py, Apache-2.0, Copyright Google LLC) so the generator logic can be tested without kickmix;
# tests/sampler_parity_dqi.py compares against the official function itself.

def sample_shots_restated(m, n, degree, log, antilog, rng, num_shots, weight_probs):
    order = (1 << degree) - 1
    max_value = order
    weights = rng.choice(len(weight_probs), size=num_shots, p=weight_probs)
    mbuc = rng.integers(0, max_value, size=(num_shots, m), endpoint=True, dtype=np.int64)
    syndromes = np.zeros((num_shots, n), dtype=np.int64)
    phases = np.zeros(num_shots, dtype=bool)
    rows = np.arange(1, n + 1, dtype=np.int64)[:, None]
    for shot in np.flatnonzero(weights):
        k = int(weights[shot])
        pos = rng.choice(m, size=k, replace=False)
        vals = rng.integers(1, max_value, size=k, endpoint=True, dtype=np.int64)
        exps = (log[vals][None, :] + rows * pos[None, :]) % order
        syndromes[shot] = np.bitwise_xor.reduce(antilog[exps], axis=1)
        phases[shot] = sum(int(x).bit_count() for x in vals & mbuc[shot, pos]) & 1 == 1
    return syndromes, mbuc, phases


def field_tables(degree):
    """log/antilog tables of GF(2^degree) for the first polynomial with x primitive."""
    order = (1 << degree) - 1
    for poly in range((1 << degree) + 1, 1 << (degree + 1), 2):
        a, seen = 1, []
        for _ in range(order):
            seen.append(a)
            a <<= 1
            if a >> degree:
                a ^= poly
        if len(set(seen)) == order:
            antilog = np.array(seen, dtype=np.int64)
            log = np.zeros(1 << degree, dtype=np.int64)
            log[antilog] = np.arange(order)
            return log, antilog
    raise AssertionError


def pcg_words(seed):
    st = np.random.PCG64(seed).state
    s, inc = st['state']['state'], st['state']['inc']
    M = (1 << 64) - 1
    return [s & M, s >> 64, inc & M, inc >> 64, st['has_uint32'], st['uinteger']]


@pytest.mark.parametrize('m,n,degree,nw', [(4095, 70, 12, 36), (63, 8, 6, 5), (1000, 20, 10, 11), (30, 4, 5, 3)])
def test_sampler_matches_numpy_restatement(m, n, degree, nw):
    log, antilog = field_tables(degree)
    g = np.random.default_rng(degree * 1000 + m)
    wp = g.random(nw) ** 3
    wp[g.integers(nw)] = 0.0
    wp /= wp.sum()
    cdf = wp.cumsum(); cdf /= cdf[-1]
    e_rows = list(range(m * degree))
    seeds = [int(g.integers(1 << 62)) * (1 << 190) + i for i in range(6)]
    pcg = np.array([pcg_words(s) for s in seeds], dtype=np.uint64)
    r = _engine.sample_dqi(m, n, degree, [float(x) for x in cdf], [int(x) for x in log], [int(x) for x in antilog],
                           e_rows, pcg, 2, True)
    syn = r['syndromes'].reshape(len(seeds), 256, n)
    mb = r['mbuc'].reshape(len(seeds), 256, m)
    ph = words_to_bools(r['phase'].reshape(len(seeds), 4))
    end = r['end_state'].reshape(len(seeds), 6)
    for i, s in enumerate(seeds):
        rng = np.random.default_rng(s)
        es, em, ep = sample_shots_restated(m, n, degree, log, antilog, rng, 256, wp)
        assert np.array_equal(syn[i], es) and np.array_equal(mb[i], em) and np.array_equal(ph[i], ep)
        st = rng.bit_generator.state
        assert [int(x) for x in end[i]] == pcg_words_from_state(st)
    # bit-sliced rows equal the values
    qin = words_to_bools(r['qin'].reshape(len(seeds), n * degree, 4))
    for i in range(len(seeds)):
        bits = ((syn[i][:, :, None] >> np.arange(degree)) & 1).astype(bool).transpose(1, 2, 0).reshape(-1, 256)
        assert np.array_equal(qin[i], bits)


def pcg_words_from_state(st):
    s, inc = st['state']['state'], st['state']['inc']
    M = (1 << 64) - 1
    return [s & M, s >> 64, inc & M, inc >> 64, st['has_uint32'], st['uinteger']]
