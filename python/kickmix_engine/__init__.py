"""kickmix-engine: bit-sliced CPU simulator core (M1 trial). AI-written under the owner's direction; see NOTICE."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import _engine
from ._engine import Program, derive_keys, kind_names, run as _run

KINDS = list(kind_names())
TOFFOLI_KINDS = ("CCX", "CCZ", "CSWAP")


def seed_words(seed: int) -> np.ndarray:
    """A 256-bit simulator seed (python int) as 8 little-endian uint32 words."""
    return np.array([(seed >> (32 * i)) & 0xFFFFFFFF for i in range(8)], dtype=np.uint32)


def compile_circuit(circuit) -> Program:
    """Compile a kickmix Circuit (loaded through the kickmix Python API) into an engine Program."""
    text = str(circuit)
    angles = []
    for line in text.splitlines():
        t = line.split()
        if t and t[0] == "Z_POW":
            h = Fraction(t[2]) * 2**127
            if h.denominator != 1:
                raise ValueError("angle not representable in 128-bit fixed point: " + line)
            angles.append(int(h) % (1 << 128))
    return Program.from_text(text, angles, circuit.num_qubits)


def run(prog: Program, seeds, xs, xreg, *, s: int = 1, threads: int = 1, counting: bool = True, dump: bool = False):
    """Run chunks. seeds: uint32 [n_chunks, 8] (or python ints); xs: uint64 [n_chunks, B, 256*s].

    Returns dict of arrays shaped like xs: x_out (register read back), scratch_dirty (any scratch qubit nonzero),
    phase_units (verifier's 2^-64-turn integers); kind_counts [n_chunks, 17] (see KINDS) and z_levels [n_chunks, 129]
    (Z_POW counts by Clifford-hierarchy level); toffoli_count [n_chunks] (CCX+CCZ+CSWAP, summed over shots).
    GIL is released during simulation. With dump=True also dump_* arrays (bit-sliced final state per 256-shot batch).
    """
    seeds = np.asarray([seed_words(x) if isinstance(x, int) else x for x in seeds], dtype=np.uint32).reshape(-1, 8)
    xs = np.ascontiguousarray(xs, dtype=np.uint64)
    n_chunks, b, per = xs.shape
    assert per == 256 * s and n_chunks == len(seeds)
    r = _run(prog, seeds, xs.reshape(-1), [int(q) for q in xreg], s, b, threads, counting, dump)
    out = {k: v.reshape(n_chunks, b, per) for k, v in r.items() if k in ("x_out", "scratch_dirty", "phase_units")}
    kc = r["kind_counts"].reshape(n_chunks, 17)
    out["kind_counts"] = kc
    out["z_levels"] = r["z_levels"].reshape(n_chunks, 129)
    out["toffoli_count"] = sum(kc[:, KINDS.index(k)] for k in TOFFOLI_KINDS)
    if dump:
        u = n_chunks * b * s
        nq, nb = prog.num_qubits, max(prog.num_bits, 1)
        out["dump_q"] = r["dump_q"].reshape(u, nq, 4)
        out["dump_bits"] = r["dump_bits"].reshape(u, nb, 4)
        out["dump_phase"] = r["dump_phase"].reshape(u, 4)
        out["dump_vphase"] = r["dump_vphase"].reshape(u, 256, 2)
    return out
