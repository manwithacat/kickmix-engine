"""Paired differential runner: two circuits, identical inputs and simulator seeds, compared shot for shot.

For any two kickmix circuits with the same register interface (registers matched by name), the runner draws inputs
once (a task plug-in's sampler or uniform random bits), runs both circuits on the same batches with the same
simulator seeds, and reports per shot: whether the interface qubits end in the same state, whether both sides are
clean (inputs restored, scratch zero), the exact phase difference in units of 2^-64 turns, and the plug-in's
acceptance verdict on each side; per paired unit (a 256-shot batch, or a chunk of batches when one simulator spans
several, as in invsqrt) the Toffolis executed on each side; and in aggregate the per-shot mean Toffoli change with
its standard error over paired units, agreement fractions and a bounded disagreement list.

"Same seeds" means the same ChaCha stream per simulator: for a rewrite that keeps the measurement structure the two
sides draw identical outcomes; otherwise the comparison is still paired by seed (the same convention the project's
48-draw gate uses), and the phase and verdict comparisons remain exact per shot. Nothing here is a verified score.
Written by an AI model (Claude) under the owner's direction.
"""
from __future__ import annotations

import math
import time

import numpy as np

from .plugins import BATCH_SHOTS, Batch, DrawSet, Interface, TaskPlugin


def delta_stats(base_tof, cand_tof, shots_per_unit) -> dict:
    """Per-shot mean Toffoli change over paired units with its standard error (unit means as the sample).
    `shots_per_unit` is a scalar or a per-unit array."""
    b = np.asarray(base_tof, dtype=np.float64)
    c = np.asarray(cand_tof, dtype=np.float64)
    if b.shape != c.shape or b.size == 0:
        raise ValueError("paired lists must be non-empty and equal length")
    d = (c - b) / np.asarray(shots_per_unit, dtype=np.float64)
    n = int(d.size)
    mean = float(d.mean())
    se = float(math.sqrt(d.var(ddof=1) / n)) if n > 1 else None
    return {"units": n, "shots": int(np.sum(np.broadcast_to(shots_per_unit, d.shape))), "mean_per_shot": mean,
            "se_per_shot": se, "min_unit_mean": float(d.min()), "max_unit_mean": float(d.max()),
            "constant": bool(np.all(np.abs(d - d[0]) < 1e-12))}


def check_interface(ia: Interface, ib: Interface) -> dict:
    """Registers must match by name, length and kind; ids may differ (the map is by position within a register)."""
    names_a, names_b = list(ia.regs), list(ib.regs)
    diffs = []
    if sorted(names_a) != sorted(names_b):
        diffs.append(f"register names differ: {names_a} vs {names_b}")
    for n in names_a:
        if n in ib.regs:
            ra, rb = ia.regs[n], ib.regs[n]
            if len(ra) != len(rb) or any(x[0] != y[0] for x, y in zip(ra, rb)):
                diffs.append(f"register {n!r} shape differs: {len(ra)} vs {len(rb)} entries")
    same_ids = not diffs and all(ia.regs[n] == ib.regs[n] for n in names_a)
    return {"compatible": not diffs, "same_ids": same_ids, "differences": diffs,
            "qubits": (ia.prog.num_qubits, ib.prog.num_qubits), "bits": (ia.prog.orig_num_bits, ib.prog.orig_num_bits)}


def verdict_codes(rows: list[dict]) -> np.ndarray:
    """[n, 256] uint8: 0 ok, 1 value error, 2 phase error."""
    v = np.zeros((len(rows), BATCH_SHOTS), np.uint8)
    for i, r in enumerate(rows):
        v[i, r["value_shots"]] = 1
        v[i, r["phase_shots"]] = 2
    return v


def units_of(plugin: TaskPlugin, batches: list[Batch]) -> list[list[int]]:
    """Paired units as lists of batch positions: single batches, or whole chunk prefixes for the chunk model."""
    if plugin.model != "chunks":
        return [[i] for i in range(len(batches))]
    groups = {}
    for i, b in enumerate(batches):
        groups.setdefault((b.draw, b.chunk), []).append(i)
    return list(groups.values())


def pair_run(plugin: TaskPlugin, ia: Interface, ib: Interface, batches: list[Batch], *, threads=1,
             compare_state=True, max_disagreements=1000, inputs=None) -> dict:
    """Run `batches` through both circuits and compare. Returns the per-shot and aggregate comparison."""
    t0 = time.perf_counter()
    iface = check_interface(ia, ib)
    if not iface["compatible"]:
        raise ValueError("register interfaces differ: " + "; ".join(iface["differences"]))
    if inputs is None:
        inputs = plugin.inputs(ia, batches)
    inputs_b = dict(inputs)
    if "bin_all" in inputs:
        inputs_b["bin"] = np.ascontiguousarray(inputs["bin_all"][:, ib.bin_keep])
    t1 = time.perf_counter()
    out_a = plugin.execute(ia, batches, inputs, threads, compare_state)
    out_b = plugin.execute(ib, batches, inputs_b, threads, compare_state)
    t_sim = time.perf_counter() - t1
    rows_a = plugin.judge(ia, batches, inputs, out_a)
    rows_b = plugin.judge(ib, batches, inputs_b, out_b)
    va, vb = verdict_codes(rows_a), verdict_codes(rows_b)
    n = len(batches)
    phase_delta = (out_b["phase_units"] - out_a["phase_units"]).view(np.int64)
    phase_equal = phase_delta == 0
    clean_a, clean_b = ~out_a["val_err"], ~out_b["val_err"]
    if compare_state and "state_q" in out_a and "state_q" in out_b:
        from . import words_to_bools
        sa = words_to_bools(out_a["state_q"][:, ia.qin_ids])  # [n, nin, 256]
        sb = words_to_bools(out_b["state_q"][:, ib.qin_ids])
        interface_equal = np.all(sa == sb, axis=1)
        full_state_equal = None
        if ia.prog.num_qubits == ib.prog.num_qubits and iface["same_ids"]:
            full_state_equal = np.all(out_a["state_q"] == out_b["state_q"], axis=(1, 2))
    else:
        interface_equal = None
        full_state_equal = None
    verdict_equal = va == vb
    agree = phase_equal & verdict_equal & (clean_a == clean_b)
    if interface_equal is not None:
        agree &= interface_equal
    # Toffolis per paired unit.
    units = units_of(plugin, batches)
    tof_a = np.array([sum(int(out_a["toffolis"][i]) for i in u) for u in units], np.int64)
    tof_b = np.array([sum(int(out_b["toffolis"][i]) for i in u) for u in units], np.int64)
    shots_u = np.array([len(u) * BATCH_SHOTS for u in units], np.int64)
    ds = delta_stats(tof_a, tof_b, shots_u)
    # Disagreements.
    dis = []
    bad = np.argwhere(~agree)
    for bi, sh in bad[:max_disagreements]:
        b = batches[bi]
        what = []
        if not phase_equal[bi, sh]:
            what.append(f"phase {int(phase_delta[bi, sh])}")
        if interface_equal is not None and not interface_equal[bi, sh]:
            what.append("interface state")
        if clean_a[bi, sh] != clean_b[bi, sh]:
            what.append(f"clean a={bool(clean_a[bi, sh])} b={bool(clean_b[bi, sh])}")
        if va[bi, sh] != vb[bi, sh]:
            what.append(f"verdict a={int(va[bi, sh])} b={int(vb[bi, sh])}")
        dis.append({"draw": b.draw, "batch": b.index, "shot": int(sh), "what": ", ".join(what)})
    shots = n * BATCH_SHOTS
    res = {
        "batches": n, "shots": shots, "units": len(units), "unit_model": plugin.model,
        "interface": iface,
        "agree_shots": int(agree.sum()), "agree_fraction": float(agree.mean()),
        "phase_equal_fraction": float(phase_equal.mean()),
        "interface_equal_fraction": (float(interface_equal.mean()) if interface_equal is not None else None),
        "full_state_equal_fraction": (float(full_state_equal.mean()) if full_state_equal is not None else None),
        "verdict_equal_fraction": float(verdict_equal.mean()),
        "phase_delta_units": {"min": int(phase_delta.min()), "max": int(phase_delta.max()),
                              "nonzero": int((phase_delta != 0).sum())},
        "verdicts": {"a": {"value": int((va == 1).sum()), "phase": int((va == 2).sum())},
                     "b": {"value": int((vb == 1).sum()), "phase": int((vb == 2).sum())}},
        "toffolis": {"a_total": int(tof_a.sum()), "b_total": int(tof_b.sum()),
                     "a_mean_per_shot": float(tof_a.sum() / shots), "b_mean_per_shot": float(tof_b.sum() / shots),
                     "delta": ds},
        "disagreements": dis, "disagreements_total": int(bad.shape[0]),
        "seconds": {"simulate": t_sim, "total": time.perf_counter() - t0, "threads": threads},
        # raw per-unit material for callers (profiles, predictions)
        "_rows_a": rows_a, "_rows_b": rows_b, "_units": units, "_tof_a": tof_a, "_tof_b": tof_b,
        "_inputs": inputs, "_out_a": out_a, "_out_b": out_b,
    }
    return res


def public(res: dict) -> dict:
    """The JSON-serialisable part of a `pair_run` result."""
    return {k: v for k, v in res.items() if not k.startswith("_")}
