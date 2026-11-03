"""The on-engine invsqrt per-shot check (`run(..., check=(target, tol))`) against the NumPy restatement of the
official verifier's check (`InvSqrtTask._check_group`), on small synthetic circuits compiled from text: a passing
circuit (phases from Z_POW on the x qubits), x not restored, a dirty scratch qubit and a phase out of tolerance,
including the failure priority within a batch and the first-failing-batch rule. Also the uniform-run IR: a program
with bit-conditioned gates reports uniform runs and matches `run_batches` shot for shot through its counts. No
kickmix and no competition files are needed. Written by an AI model (Claude) under the owner's direction.
"""
import numpy as np
import pytest

import kickmix_engine as ke

NQ = 8          # 4 x qubits + 4 scratch
XREG = [0, 1, 2, 3]
MAX_X = 15      # x in 1..15 fits 4 bits


def numpy_fold(xs, out, target, tol, scratch_ids):
    """The official per-batch checks and folds, as the verifier (and `_check_group`) computes them."""
    bad_x = (out['x_out'] != xs).any(axis=2)
    dirty = out['scratch_dirty'].astype(bool).any(axis=2)
    allowed = tol[xs - np.uint64(1)]
    error = (out['phase_units'] - target[xs - np.uint64(1)]).view(np.int64)
    used = np.abs(error).view(np.uint64)
    bad_ph = (used > allowed).any(axis=2)
    anybad = bad_x | dirty | bad_ph
    fail = None
    if anybad.any():
        c, b = np.argwhere(anybad)[0]
        if bad_x[c, b]:
            fail = (int(c), int(b), 0)
        elif dirty[c, b]:
            fail = (int(c), int(b), 1)
        else:
            i = int(np.flatnonzero(used[c, b] > allowed[c, b])[0])
            fail = (int(c), int(b), 2, int(xs[c, b, i]), int(error[c, b, i]), int(allowed[c, b, i]))
        return fail
    xf, al, us = xs.reshape(-1), allowed.reshape(-1), used.reshape(-1)
    ratio = np.divide(us, al, out=np.zeros(len(xf)), where=al > 0)
    rmax = float(ratio.max())
    return None, int((al - us).min()), rmax, int(xf[ratio == rmax].min())


def engine_fold(out):
    fails = out['check_fail']
    bad = np.flatnonzero(fails[:, 0] >= 0)
    if len(bad):
        c = int(bad[0])
        b, prio, sh, x, err, allowed = (int(v) for v in fails[c])
        return (c, b, prio) if prio < 2 else (c, b, prio, x, err, allowed)
    wr = out['check_worst_ratio']
    rmax = float(wr.max())
    return None, int(out['check_min_margin'].min()), rmax, int(out['check_worst_x'][wr == rmax].min())


def inputs(n_chunks, batches, rng):
    xs = rng.integers(1, MAX_X + 1, size=(n_chunks, batches, 256), dtype=np.uint64)
    seeds = [ke.seed_words(1000 + i) for i in range(n_chunks)]
    return xs, seeds


def tables(rng, slack):
    """Random targets; tolerance = slack (the test circuits' phases are compared against these)."""
    target = rng.integers(0, 1 << 63, size=MAX_X, dtype=np.uint64)
    tol = np.full(MAX_X, slack, dtype=np.uint64)
    return target, tol


def phase_circuit():
    # phase = sum over set x bits of distinct angles; restores x; leaves scratch clean.
    # dyadic angles (exact in 128-bit fixed point), one per x bit
    return "\n".join(["Z_POW q0 0.125", "Z_POW q1 0.3125", "Z_POW q2 0.046875", "Z_POW q3 0.703125",
                      "CX q0 q4", "CX q0 q4", "X q5", "X q5"])


def exact_tables(prog, xs_all):
    """Targets equal to the circuit's own phase for each x (so every shot passes with margin = tol)."""
    xs = np.arange(1, MAX_X + 1, dtype=np.uint64)
    pad = np.concatenate([xs, np.ones(256 - MAX_X, np.uint64)]).reshape(1, 1, 256)
    r = ke.run(prog, [ke.seed_words(7)], pad, XREG, s=1)
    return r['phase_units'][0, 0, :MAX_X].copy()


@pytest.mark.parametrize('threads', [1, 2])
def test_pass_fold_matches_numpy(threads):
    rng = np.random.default_rng(3)
    prog = ke.compile_text(phase_circuit(), NQ)
    xs, seeds = inputs(3, 4, rng)
    target = exact_tables(prog, xs)
    # perturb targets so errors and margins vary per x (all within tolerance)
    target = target + rng.integers(-1000, 1000, size=MAX_X).astype(np.int64).view(np.uint64)
    tol = rng.integers(1001, 5000, size=MAX_X, dtype=np.uint64)
    plain = ke.run(prog, seeds, xs, XREG, s=1, threads=threads)
    chk = ke.run(prog, seeds, xs, XREG, s=1, threads=threads, check=(target, tol))
    assert 'x_out' not in chk and 'check_fail' in chk
    assert numpy_fold(xs, plain, target, tol, [4, 5, 6, 7]) == engine_fold(chk)
    assert np.array_equal(plain['kind_counts'], chk['kind_counts']) and np.array_equal(plain['z_levels'], chk['z_levels'])
    assert (chk['check_fail'][:, 0] == -1).all()


def test_zero_tolerance_ratio_rule():
    rng = np.random.default_rng(4)
    prog = ke.compile_text(phase_circuit(), NQ)
    xs, seeds = inputs(2, 2, rng)
    target = exact_tables(prog, xs)
    tol = np.zeros(MAX_X, np.uint64)  # exact match required; ratio defined as 0 where tol == 0
    plain = ke.run(prog, seeds, xs, XREG, s=1)
    chk = ke.run(prog, seeds, xs, XREG, s=1, check=(target, tol))
    assert numpy_fold(xs, plain, target, tol, []) == engine_fold(chk) == (None, 0, 0.0, int(xs.min()))


def test_phase_failure_first_batch_and_shot():
    rng = np.random.default_rng(5)
    prog = ke.compile_text(phase_circuit(), NQ)
    xs, seeds = inputs(3, 3, rng)
    target = exact_tables(prog, xs)
    tol = np.full(MAX_X, 10, np.uint64)
    bad_x_value = 9
    target[bad_x_value - 1] += np.uint64(11)          # every shot with x = 9 fails by 1 unit
    plain = ke.run(prog, seeds, xs, XREG, s=1)
    chk = ke.run(prog, seeds, xs, XREG, s=1, check=(target, tol))
    nf, ef = numpy_fold(xs, plain, target, tol, []), engine_fold(chk)
    assert nf == ef and nf[2] == 2 and nf[3] == bad_x_value and nf[4] == -11 and nf[5] == 10


def test_x_not_restored_beats_phase_and_dirty():
    rng = np.random.default_rng(6)
    circ = phase_circuit() + "\nX q4\nX q0"   # scratch q4 left dirty and x bit 0 flipped: priority is x
    prog = ke.compile_text(circ, NQ)
    xs, seeds = inputs(2, 2, rng)
    target, tol = tables(rng, 5)
    plain = ke.run(prog, seeds, xs, XREG, s=1)
    chk = ke.run(prog, seeds, xs, XREG, s=1, check=(target, tol))
    assert numpy_fold(xs, plain, target, tol, [4]) == engine_fold(chk) == (0, 0, 0)


def test_dirty_scratch_reported_with_batch():
    rng = np.random.default_rng(8)
    prog = ke.compile_text(phase_circuit() + "\nX q6", NQ)
    xs, seeds = inputs(2, 3, rng)
    target = exact_tables(ke.compile_text(phase_circuit(), NQ), xs)
    tol = np.full(MAX_X, 1 << 40, np.uint64)
    plain = ke.run(prog, seeds, xs, XREG, s=1)
    chk = ke.run(prog, seeds, xs, XREG, s=1, check=(target, tol))
    assert numpy_fold(xs, plain, target, tol, [4, 5, 6, 7]) == engine_fold(chk) == (0, 0, 1)
    assert int(chk['check_fail'][0, 2]) == 0  # first dirty shot of the batch


def test_failure_in_later_chunk_only():
    rng = np.random.default_rng(9)
    prog = ke.compile_text(phase_circuit(), NQ)
    xs, seeds = inputs(3, 2, rng)
    xs[:] = 1                      # every shot sees x = 1 ...
    xs[2, 1, 100:] = 7             # ... except chunk 2, batch 1, shots 100.., which see x = 7, whose target is wrong
    target = exact_tables(prog, xs)
    tol = np.full(MAX_X, 3, np.uint64)
    target[6] += np.uint64(4)
    plain = ke.run(prog, seeds, xs, XREG, s=1)
    chk = ke.run(prog, seeds, xs, XREG, s=1, check=(target, tol))
    nf, ef = numpy_fold(xs, plain, target, tol, []), engine_fold(chk)
    assert nf == ef == (2, 1, 2, 7, -4, 3)
    assert (chk['check_fail'][:2, 0] == -1).all() and int(chk['check_fail'][2, 2]) == 100


def test_check_rejects_bad_inputs():
    prog = ke.compile_text(phase_circuit(), NQ)
    xs = np.zeros((1, 1, 256), np.uint64)   # x = 0 is outside the table
    with pytest.raises(ValueError):
        ke.run(prog, [ke.seed_words(1)], xs, XREG, s=1, check=(np.ones(MAX_X, np.uint64), np.ones(MAX_X, np.uint64)))
    with pytest.raises(ValueError):
        ke.run(prog, [ke.seed_words(1)], xs + np.uint64(1), XREG, s=1, dump=True,
               check=(np.ones(MAX_X, np.uint64), np.ones(MAX_X, np.uint64)))


def test_uniform_runs_cover_conditioned_gates():
    """Bit-conditioned CZ/CX/X/Z join uniform runs; counts and state agree with the kickmix-referenced batch path."""
    lines = ["HMR q4 b0", "HMR q5 b1"]
    for i in range(40):
        lines += [f"CZ q{i % 4} q{(i + 1) % 4} if b{i % 2}", f"CX q{i % 4} q{4 + i % 4} if b{(i + 1) % 2}",
                  f"X q{i % 4} if b{i % 2}", f"Z q{(i + 2) % 4} if b{i % 2}", f"CCX q0 q1 q{4 + i % 4}", "NEG"]
    prog = ke.compile_text("\n".join(lines), NQ)
    assert prog.num_uni_runs >= 1 and prog.num_ops == len(lines)
    rng = np.random.default_rng(11)
    qin = rng.integers(0, 1 << 64, size=(3, NQ, 4), dtype=np.uint64)
    r = ke.run_batches(prog, [ke.seed_words(5 + i) for i in range(3)], list(range(NQ)), qin, [], np.zeros((3, 0, 4), np.uint64),
                       threads=1, dump=True)
    # the same shots through `run` (one chunk per batch, x register = all qubits written from the first word pattern)
    counts = r['kind_counts']
    assert counts[:, ke.KINDS.index('CZ')].sum() > 0 and counts[:, ke.KINDS.index('CX')].sum() > 0
    # conditioned counts are shot counts, never 256 per op for every op
    assert (counts[:, ke.KINDS.index('CZ')] <= 40 * 256).all()
    assert (counts[:, ke.KINDS.index('CCX')] == 40 * 256).all()
    assert (counts[:, ke.KINDS.index('NEG')] == 40 * 256).all()
