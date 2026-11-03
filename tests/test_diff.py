"""Tests for the structural diff, the paired runner and `diff`'s pure helpers. They need neither kickmix nor the
competition files: the fixture circuit `tests/data/random_ops.kmb` is diffed against itself and against a copy with
one op type byte changed (CX 0x21 <-> CZ 0x20 keep the operand arrays valid), and the generic plug-in pairs the two.
Written by an AI model (Claude) under the owner's direction.
"""
import os
import struct

import numpy as np
import pytest

import kickmix_engine as ke
from kickmix_engine.diff import compare_failures, schedule, structural
from kickmix_engine.pair import delta_stats, pair_run
from kickmix_engine.plugins import DrawSet, make_plugin

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
KMB = os.path.join(DATA, 'random_ops.kmb')


def types_packet_offset(data: bytes) -> tuple[int, int]:
    """(offset of the first op type byte, number of ops) of a version-2 .kmb."""
    pos = 16 + 16 + 72
    num_regs = struct.unpack_from('<I', data, 16 + 12)[0]
    for _ in range(num_regs):
        _, size = struct.unpack_from('<QQ', data, pos)
        pos += 16 + size
    ty, size = struct.unpack_from('<QQ', data, pos)
    assert ty == 1
    return pos + 16, size


def mutate_one_op(src_path, dst_path, which=0):
    """Copy the file, turning the `which`-th plain CX (0x21) into a CZ (0x20); returns the op index changed."""
    data = bytearray(open(src_path, 'rb').read())
    off, n = types_packet_offset(bytes(data))
    seen = 0
    for i in range(n):
        if data[off + i] == 0x21:
            if seen == which:
                data[off + i] = 0x20
                open(dst_path, 'wb').write(bytes(data))
                return i
            seen += 1
    pytest.skip('fixture has no plain CX op')


def test_structural_identical():
    d = structural(KMB, KMB)
    assert d['identical'] and d['num_hunks'] == 0 and d['hunks'] == []
    assert d['common_prefix'] == d['a']['num_ops'] == d['b']['num_ops']
    assert d['interface']['compatible'] and d['interface']['same_ids']
    prog = ke.compile_kmb(KMB)
    assert d['a']['num_ops'] == prog.num_ops
    assert d['a']['static_counts'] == {k: int(c) for k, c in zip(ke.KINDS, prog.static_counts) if c}


def test_structural_one_op_changed(tmp_path):
    dst = str(tmp_path / 'mut.kmb')
    i = mutate_one_op(KMB, dst, which=3)
    d = structural(KMB, dst)
    assert not d['identical'] and d['num_hunks'] == 1
    h = d['hunks'][0]
    assert h['a'] == [i, i + 1] and h['b'] == [i, i + 1]
    assert h['kinds'] == {'CZ': [0, 1], 'CX': [1, 0]} and h['toffoli_delta'] == 0
    assert d['common_prefix'] == i and d['common_suffix'] == d['a']['num_ops'] - i - 1
    assert d['delta']['static_counts'] == {'CZ': 1, 'CX': -1} and d['delta']['static_toffolis'] == 0
    # symmetric
    e = structural(dst, KMB)
    assert e['hunks'][0]['kinds'] == {'CZ': [1, 0], 'CX': [0, 1]}


def test_structural_truncated_tail(tmp_path):
    """A circuit that is a strict prefix of another: one hunk holding the extra ops, nothing else."""
    data = bytearray(open(KMB, 'rb').read())
    off, n = types_packet_offset(bytes(data))
    # Turn the last plain CX into a CZ *and* another earlier one: two separate hunks, anchors between them.
    idx = [i for i in range(n) if data[off + i] == 0x21]
    if len(idx) < 40:
        pytest.skip('fixture has too few CX ops')
    a, b = idx[5], idx[-5]
    data[off + a] = 0x20
    data[off + b] = 0x20
    dst = str(tmp_path / 'two.kmb')
    open(dst, 'wb').write(bytes(data))
    d = structural(KMB, dst, k=4, sample_bits=1)
    assert d['num_hunks'] == 2
    assert d['hunks'][0]['a'] == [a, a + 1] and d['hunks'][1]['a'] == [b, b + 1]
    assert d['matched_ops'] == n - 2


def test_pair_generic_self_and_mutant(tmp_path):
    prog = ke.compile_kmb(KMB)
    # the fixture has no register table: declare every qubit an input, so inputs are random bits per shot
    plugin = make_plugin('generic', batches_per_draw=4, qubits=range(prog.num_qubits))
    sha = 'ab' * 32
    ia = plugin.interface(prog, sha)
    draws = DrawSet(anchor=sha, count=1)
    batches = plugin.batches(sha, draws, 0)
    r = pair_run(plugin, ia, ia, batches, threads=2)
    assert r['agree_fraction'] == 1.0 and r['phase_equal_fraction'] == 1.0
    assert r['interface_equal_fraction'] == 1.0 and r['full_state_equal_fraction'] == 1.0
    assert r['toffolis']['delta']['mean_per_shot'] == 0.0 and r['disagreements'] == []
    dst = str(tmp_path / 'mut.kmb')
    # the fixture's first CX ops sit under PUSH_CONDITION on classical bits that are zero here (no bit inputs),
    # so they never execute; the sixth plain CX (op 46) runs unconditionally on random inputs
    mutate_one_op(KMB, dst, which=5)
    pb = ke.compile_kmb(dst)
    ib = plugin.interface(pb, 'cd' * 32)
    r2 = pair_run(plugin, ia, ib, batches, threads=2)
    assert r2['agree_fraction'] < 1.0 and r2['disagreements_total'] > 0
    assert r2['full_state_equal_fraction'] < 1.0
    assert r2['toffolis']['delta']['mean_per_shot'] == 0.0  # CX -> CZ changes no Toffoli


def test_schedule_and_helpers():
    s = schedule(16, 48, 400)
    assert s[0] == (0, 3) and s[1] == (1, 160) and len(set(s)) == 16
    with pytest.raises(ValueError):
        schedule(5, 1, 1)
    ds = delta_stats([256, 512], [255, 510], 256)
    assert ds['units'] == 2 and ds['shots'] == 512 and ds['mean_per_shot'] == pytest.approx(-1.5 / 256)
    assert not ds['constant'] and ds['se_per_shot'] > 0
    base = {(0, 1): {'value': [3], 'phase': []}, (0, 2): {'value': [], 'phase': [7]}}
    cand = {(0, 1): {'value': [3], 'phase': []}, (0, 2): {'value': [7], 'phase': []}}
    c = compare_failures(base, cand)
    assert c['mismatches'] == [] and len(c['type_changes']) == 1 and c['matched'] == 2
    cand[(0, 2)] = {'value': [], 'phase': []}
    assert len(compare_failures(base, cand)['mismatches']) == 1
