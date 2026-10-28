"""kickmix-engine: bit-sliced CPU simulator, per-shot compatible with kickmix, with task adapters that reproduce the
official invsqrt and DQI verifiers. Written by an AI model (Claude) under the owner's direction; see NOTICE."""
from __future__ import annotations

from fractions import Fraction

import numpy as np

from . import _engine
from ._engine import Program, derive_keys, engine_version, kind_names, run as _run, run_batches as _run_batches

__version__ = engine_version()[0]

KINDS = list(kind_names())
TOFFOLI_KINDS = ("CCX", "CCZ", "CSWAP")


def seed_words(seed: int) -> np.ndarray:
    """A 256-bit simulator seed (python int) as 8 little-endian uint32 words."""
    return np.array([(seed >> (32 * i)) & 0xFFFFFFFF for i in range(8)], dtype=np.uint32)


def compile_text(text: str, num_qubits: int) -> Program:
    """Compile kickmix circuit text (the `.kmx` form, as `str(circuit)` prints it). Z_POW angles are parsed exactly."""
    angles = []
    for line in text.splitlines():
        t = line.split()
        if t and t[0] == "Z_POW":
            h = Fraction(t[2]) * 2**127
            if h.denominator != 1:
                raise ValueError("angle not representable in 128-bit fixed point: " + line)
            angles.append(int(h) % (1 << 128))
    return Program.from_text(text, angles, num_qubits)


def compile_circuit(circuit) -> Program:
    """Compile a kickmix Circuit (loaded through the kickmix Python API) into an engine Program."""
    return compile_text(str(circuit), circuit.num_qubits)


def run(prog: Program, seeds, xs, xreg, *, s: int = 1, threads: int = 1, counting: bool = True, dump: bool = False):
    """Run chunks. seeds: uint32 [n_chunks, 8] (or python ints); xs: uint64 [n_chunks, B, 256*s].

    Returns dict of arrays shaped like xs: x_out (register read back), scratch_dirty (any scratch qubit nonzero),
    phase_units (verifier's 2^-64-turn integers); kind_counts [n_chunks, len(KINDS)] (see KINDS) and z_levels [n_chunks, 129]
    (Z_POW counts by Clifford-hierarchy level); toffoli_count [n_chunks] (CCX+CCZ+CSWAP, summed over shots).
    GIL is released during simulation. With dump=True also dump_* arrays (bit-sliced final state per 256-shot batch).
    """
    seeds = np.asarray([seed_words(x) if isinstance(x, int) else x for x in seeds], dtype=np.uint32).reshape(-1, 8)
    xs = np.ascontiguousarray(xs, dtype=np.uint64)
    n_chunks, b, per = xs.shape
    assert per == 256 * s and n_chunks == len(seeds)
    r = _run(prog, seeds, xs.reshape(-1), [int(q) for q in xreg], s, b, threads, counting, dump)
    out = {k: v.reshape(n_chunks, b, per) for k, v in r.items() if k in ("x_out", "scratch_dirty", "phase_units")}
    kc = r["kind_counts"].reshape(n_chunks, len(KINDS))
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


def compile_kmb(path) -> Program:
    """Compile a .kmb file directly (streamed in Rust; no kickmix Circuit and no text form)."""
    return Program.from_kmb(str(path))


def run_batches(prog: Program, seeds, qin_ids, qin, bin_ids, bin_, *, threads: int = 1, counting: bool = True,
                dump: bool = False):
    """Independent 256-shot batches, each one kickmix ``Simulator(batch_size=256, seed=seeds[i])``.

    qin: uint64 [n, len(qin_ids), 4] bit-sliced input qubit values (bit s of word s//64 = shot s);
    bin_: uint64 [n, len(bin_ids), 4] input classical bits (compact ids, see ``Program.bit_compact``).
    Returns val_err [n, 4] (an input qubit not restored or another qubit nonzero), phase [n, 4] (phase flipped),
    kind_counts [n, len(KINDS)], z_levels [n, 129], toffoli_count [n]; with dump=True also dump_q [n, nq, 4] and
    dump_bits [n, num_bits, 4]. The GIL is released during simulation; concurrent calls share one thread pool.
    """
    seeds = np.asarray([seed_words(x) if isinstance(x, int) else x for x in seeds], dtype=np.uint32).reshape(-1, 8)
    n = len(seeds)
    qin = np.ascontiguousarray(qin, dtype=np.uint64).reshape(-1)
    bin_ = np.ascontiguousarray(bin_, dtype=np.uint64).reshape(-1)
    r = _run_batches(prog, seeds, [int(q) for q in qin_ids], qin, [int(b) for b in bin_ids], bin_, threads, counting, dump)
    out = dict(val_err=r["val_err"].reshape(n, 4), phase=r["phase"].reshape(n, 4))
    out["kind_counts"] = kc = r["kind_counts"].reshape(n, len(KINDS))
    out["z_levels"] = r["z_levels"].reshape(n, 129)
    out["toffoli_count"] = sum(kc[:, KINDS.index(k)] for k in TOFFOLI_KINDS)
    if dump:
        out["dump_q"] = r["dump_q"].reshape(n, prog.num_qubits, 4)
        out["dump_bits"] = r["dump_bits"].reshape(n, max(prog.num_bits, 1), 4)
    return out


def words_to_bools(words):
    """uint64 [..., 4] bit-sliced words -> bool [..., 256] (shot axis last)."""
    w = np.ascontiguousarray(words, dtype=np.uint64)
    return np.unpackbits(w.view(np.uint8).reshape(*w.shape[:-1], 32), axis=-1, bitorder="little").astype(bool)


def bools_to_words(b):
    """bool [..., 256] -> uint64 [..., 4]."""
    p = np.packbits(np.ascontiguousarray(b, dtype=bool), axis=-1, bitorder="little")
    return np.ascontiguousarray(p).view(np.uint64)
