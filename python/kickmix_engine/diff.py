"""`kickmix-engine diff`: compare two circuits in three layers, and `profile`: a reusable base profile.

1. Structural (no simulation, `_engine.structural_diff`): op-level alignment, hunks with per-kind and Toffoli
   deltas, static counts, qubits, bits and the register interface.
2. Semantic (the paired runner, `pair.py`): both circuits on identical inputs and seeds from the task plug-in's
   sampler over a fixed schedule of batches; per-shot interface-state, phase and verdict agreement; the per-shot mean
   Toffoli change with its standard error.
3. Cost prediction (with a base profile): the candidate's expected cost over the profile's draw set, through the
   plug-in's price model, assuming the candidate's failing shots equal the base's; the precondition is checked on
   the base's failing batches and on every paired batch (a candidate failure elsewhere is not excluded, and the
   coverage is reported). A profile records, once per base, every batch's Toffolis and failing shots over the draw
   set, so each candidate costs a short paired run instead of a full sweep.

This generalises the project's `scripts/dqi_fast_rank.py` (whose profile JSON this module reads and writes) from
DQI to any task plug-in. Every number here is an engine measurement or a static model / projection; the official
verifier is the only authority on a score. Written by an AI model (Claude) under the owner's direction.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from . import __version__
from .cache import ProgramCache, load_program
from .pair import pair_run, public
from .plugins import BATCH_SHOTS, Batch, DrawSet, TaskPlugin, make_plugin

REL_TIE = 1e-5


# ---------------------------------------------------------------- pure helpers

def schedule(n: int, draws: int, batches: int, stride: int = 157, offset: int = 3) -> list[tuple[int, int]]:
    """Fixed paired-batch schedule: item i is (draw i mod draws, batch (stride i + offset) mod batches). Prefixes
    nest, so one run at N gives every smaller N; it spreads over all draws and repeats no batch while n is below the
    period of the two residues."""
    out = [(i % draws, (stride * i + offset) % batches) for i in range(n)]
    if len(set(out)) != len(out):
        raise ValueError("schedule repeats a batch; choose a smaller n or another stride")
    return out


def compare_failures(base_fail: dict, cand_fail: dict) -> dict:
    """{(draw, batch): {"value": [...], "phase": [...]}} on both sides over the same checked batches. A batch matches
    when the set of failing shots (value or phase) is the same: the DQI cost charges R(value + phase) and a shot is
    never both, so an equal set means an equal k_d. A shot failing on both sides with a different type is reported
    in `type_changes`."""
    keys = sorted(set(base_fail) | set(cand_fail))
    mism, types = [], []
    for key in keys:
        b = base_fail.get(key, {"value": [], "phase": []})
        c = cand_fail.get(key, {"value": [], "phase": []})
        if set(b["value"]) | set(b["phase"]) != set(c["value"]) | set(c["phase"]):
            mism.append({"draw": key[0], "batch": key[1], "base": b, "candidate": c})
        elif sorted(b["value"]) != sorted(c["value"]):
            types.append({"draw": key[0], "batch": key[1], "base": b, "candidate": c})
    return {"batches": len(keys), "matched": len(keys) - len(mism), "mismatches": mism, "type_changes": types}


def rel_err(pred: float, full: float) -> float:
    return pred / full - 1


# ---------------------------------------------------------------- structural

def structural(path_a, path_b, *, k=16, sample_bits=6, max_hunks=1000) -> dict:
    """Layer 1 as a JSON-ready dict (kind names, deltas, interface check)."""
    from . import KINDS, _engine
    t0 = time.perf_counter()
    d = _engine.structural_diff(str(path_a), str(path_b), k, sample_bits, max_hunks)
    tof = lambda c: int(c[KINDS.index("CCX")] + c[KINDS.index("CCZ")] + c[KINDS.index("CSWAP")])
    named = lambda c: {KINDS[i]: int(c[i]) for i in range(len(KINDS)) if c[i]}

    def side(s):
        return {"num_ops": int(s["num_ops"]), "num_qubits": int(s["num_qubits"]), "num_bits": int(s["num_bits"]),
                "qubits_touched": int(s["qubits_touched"]), "static_toffolis": tof(s["static_counts"]),
                "static_counts": named(s["static_counts"]),
                "registers": [(n, len(r), "qubits" if all(x[0] for x in r) else "bits") for n, r in s["registers"]]}

    ra, rb = dict((n, r) for n, r in d["a"]["registers"]), dict((n, r) for n, r in d["b"]["registers"])
    reg_diffs = []
    if sorted(ra) != sorted(rb):
        reg_diffs.append(f"names {sorted(ra)} vs {sorted(rb)}")
    for n in ra:
        if n in rb and (len(ra[n]) != len(rb[n]) or any(x[0] != y[0] for x, y in zip(ra[n], rb[n]))):
            reg_diffs.append(f"{n}: shape differs")
    same_ids = not reg_diffs and all(ra[n] == rb[n] for n in ra)
    hunks = []
    for h in d["hunks"]:
        hunks.append({"a": [int(h["a"][0]), int(h["a"][1])], "b": [int(h["b"][0]), int(h["b"][1])],
                      "a_ops": int(h["a"][1] - h["a"][0]), "b_ops": int(h["b"][1] - h["b"][0]),
                      "toffoli_delta": int(h["toffoli_delta"]),
                      "kinds": {KINDS[i]: [int(h["a_kinds"][i]), int(h["b_kinds"][i])] for i in range(len(KINDS))
                                if h["a_kinds"][i] or h["b_kinds"][i]}})
    sa, sb = side(d["a"]), side(d["b"])
    return {"identical": bool(d["identical"]), "a": sa, "b": sb,
            "delta": {"num_ops": sb["num_ops"] - sa["num_ops"], "num_qubits": sb["num_qubits"] - sa["num_qubits"],
                      "num_bits": sb["num_bits"] - sa["num_bits"],
                      "static_toffolis": sb["static_toffolis"] - sa["static_toffolis"],
                      "static_counts": {k: int(d["b"]["static_counts"][i]) - int(d["a"]["static_counts"][i])
                                        for i, k in enumerate(KINDS)
                                        if d["b"]["static_counts"][i] != d["a"]["static_counts"][i]}},
            "interface": {"compatible": not reg_diffs, "same_ids": same_ids, "differences": reg_diffs},
            "common_prefix": int(d["common_prefix"]), "common_suffix": int(d["common_suffix"]),
            "matched_ops": int(d["matched_ops"]), "anchors": int(d["anchors"]),
            "num_hunks": int(d["num_hunks"]), "hunks": hunks, "hunks_shown": len(hunks),
            "alignment": "exact (prefix/suffix)" if int(d["num_hunks"]) <= 1 and int(d["anchors"]) == 0 else
                         f"prefix/suffix + {int(d['anchors'])} content-defined anchors (k={k}, 2^-{sample_bits})",
            "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------- circuits

class Circuit:
    """A compiled circuit with its hash and the plug-in's interface."""

    def __init__(self, path, plugin: TaskPlugin, cache: ProgramCache | None = None):
        t0 = time.perf_counter()
        self.path = str(path)
        self.prog, self.sha, self.load_info = load_program(self.path, cache=cache or ProgramCache(enabled=False))
        if self.load_info.get("save_needed") and cache is not None:
            cache.save(self.sha, self.prog)
        self.iface = plugin.interface(self.prog, self.sha)
        self.q = int(self.prog.num_qubits)
        self.load_s = time.perf_counter() - t0


# ---------------------------------------------------------------- profile

def chunk_groups(plugin: TaskPlugin, batches: list[Batch], chunk: int) -> list[list[Batch]]:
    """Groups of about `chunk` batches per engine call; for the chunk model whole simulator chunks stay together."""
    if plugin.model != "chunks":
        return [batches[lo:lo + chunk] for lo in range(0, len(batches), max(chunk, 1))]
    groups, cur = [], []
    for b in batches:
        if b.in_chunk == 0 and len(cur) >= chunk:
            groups.append(cur)
            cur = []
        cur.append(b)
    if cur:
        groups.append(cur)
    return groups


def _draw_record(plugin, circ, draws, d, rows, q) -> dict:
    fails = [{"batch": r["index"], "value_shots": r["value_shots"], "phase_shots": r["phase_shots"]}
             for r in rows if r["value_shots"] or r["phase_shots"]]
    summ = plugin.draw_summary(circ.iface, rows, q)
    rec = {"salt": draws.salt(d), "seed": str(plugin.draw_seed(circ.sha, draws, d)),
           "batch_toffolis": [int(r["toffolis"]) for r in rows], "failing": fails}
    rec.update({k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in summ.items()})
    return rec


def profile(plugin: TaskPlugin, circ: Circuit, draws: DrawSet, *, threads=4, chunk=100, log=None) -> dict:
    """Run every batch of every draw once; record per-batch Toffolis and failing shots (the plug-in's verdicts)."""
    t0 = time.perf_counter()
    recs = []
    for d in range(draws.count):
        td = time.perf_counter()
        batches = plugin.batches(circ.sha, draws, d)
        rows = []
        for sub in chunk_groups(plugin, batches, chunk):
            inputs = plugin.inputs(circ.iface, sub)
            out = plugin.execute(circ.iface, sub, inputs, threads, False)
            rows += [dict(r, index=sub[j].index) for j, r in enumerate(plugin.judge(circ.iface, sub, inputs, out))]
        rec = _draw_record(plugin, circ, draws, d, rows, circ.q)
        rec["seconds"] = round(time.perf_counter() - td, 2)
        recs.append(rec)
        if log:
            log(json.dumps({k: rec[k] for k in rec if k not in ("batch_toffolis", "failing", "margins")}
                           | {"failing_batches": len(rec["failing"])}, default=str))
    costs = [r["score"] for r in recs]
    return {"tier": "engine verify (sampled draws, fixed seeds; parity-tested with kickmix), local",
            "task": plugin.name, "engine_version": __version__,
            "circuit": circ.path, "sha256": circ.sha, "sha12": circ.sha[:12], "Q": circ.q,
            "draw_set": draws.describe(), "anchor": draws.anchor, "salt_format": draws.salt_format,
            "batches_per_draw": plugin.batches_per_draw, "batch_shots": BATCH_SHOTS, "threads": threads,
            "unit_model": plugin.model,
            "expected_cost": sum(costs) / len(costs),
            "failing_shots": sum(r["value_errors"] + r["phase_errors"] for r in recs),
            "failing_batches": sum(len(r["failing"]) for r in recs),
            "load_s": round(circ.load_s, 2), "seconds": round(time.perf_counter() - t0, 1), "draws": recs}


def draws_of_profile(prof: dict) -> DrawSet:
    return DrawSet(anchor=prof.get("anchor"), salt_format=prof.get("salt_format", "sweep-{}"), count=len(prof["draws"]))


def predict(plugin: TaskPlugin, prof: dict, q: int, delta_t: float) -> dict:
    p = plugin.price(prof["draws"], q, delta_t)
    p.update(draws=len(prof["draws"]), Q=q, delta_t=delta_t)
    return p


# ---------------------------------------------------------------- the three layers

def select_batches(plugin: TaskPlugin, sha: str, draws: DrawSet, items: list[tuple[int, int]], cache: dict | None = None) -> list[Batch]:
    """Batches (draw, index) in order; for the chunk model each selected batch's chunk prefix is included."""
    cache = {} if cache is None else cache
    out, seen = [], set()
    for d, b in items:
        if d not in cache:
            cache[d] = plugin.batches(sha, draws, d)
        bl = cache[d]
        if plugin.model == "chunks":
            c = bl[b].chunk
            for x in bl:
                if x.chunk == c and x.in_chunk <= bl[b].in_chunk and (d, x.index) not in seen:
                    seen.add((d, x.index))
                    out.append(x)
        elif (d, b) not in seen:
            seen.add((d, b))
            out.append(bl[b])
    return out


def diff(task: str, path_a, path_b, *, verifier_path=None, draws: DrawSet | None = None, n=16, profile_json=None,
         threads=4, structural_layer=True, semantic_layer=True, inputs=None, compare_state=True, report_n=None,
         sweep=None, cache: ProgramCache | None = None, spec=None, log=None) -> dict:
    """Compare circuit `path_b` (candidate) against `path_a` (base) for task plug-in `task`."""
    t_all = time.perf_counter()
    spec = dict(spec or {})
    if inputs:
        spec["inputs"] = inputs
    plugin = make_plugin(task, verifier_path, **spec)
    rep = {"task": task, "engine_version": __version__, "a": {"path": str(path_a)}, "b": {"path": str(path_b)},
           "tier": "structural: static; semantic: engine verify (fixed seeds), local; prediction: static model / projection"}
    if structural_layer:
        rep["structural"] = structural(path_a, path_b)
    prof = None
    if profile_json:
        prof = json.loads(open(profile_json).read())
        draws = draws_of_profile(prof)
    ca = Circuit(path_a, plugin, cache)
    cb = Circuit(path_b, plugin, cache)
    rep["a"].update(sha256=ca.sha, Q=ca.q, load_s=round(ca.load_s, 2))
    rep["b"].update(sha256=cb.sha, Q=cb.q, load_s=round(cb.load_s, 2))
    if prof is not None and prof["sha256"] != ca.sha:
        raise ValueError(f"profile is for {prof['sha12']}, base file hashes {ca.sha[:12]}")
    if draws is None:
        draws = DrawSet(anchor=ca.sha, count=1)
    if not semantic_layer:
        rep["seconds"] = time.perf_counter() - t_all
        return rep
    bpd = plugin.batches_per_draw
    sched = schedule(n, draws.count, bpd)
    fail_items = []
    if prof is not None:
        fail_items = sorted({(d, f["batch"]) for d, r in enumerate(prof["draws"]) for f in r["failing"]})
    items = sorted(set(sched) | set(fail_items))
    bcache = {}
    batches = select_batches(plugin, ca.sha, draws, items, bcache)
    res = pair_run(plugin, ca.iface, cb.iface, batches, threads=threads, compare_state=compare_state)
    rep["semantic"] = public(res)
    rep["semantic"]["draw_set"] = draws.describe()
    rep["semantic"]["schedule"] = f"draw i mod {draws.count}, batch (157 i + 3) mod {bpd}; N = {n}" + (
        f"; plus {len(fail_items)} base failing batches" if fail_items else "")
    pos = {(b.draw, b.index): i for i, b in enumerate(batches)}
    rows_a = {(b.draw, b.index): res["_rows_a"][i] for i, b in enumerate(batches)}
    rows_b = {(b.draw, b.index): res["_rows_b"][i] for i, b in enumerate(batches)}
    if prof is None:
        rep["seconds"] = time.perf_counter() - t_all
        return rep

    # Layer 3: prediction against the profile.
    # determinism: the base re-run must reproduce its profile on every paired unit
    det_bad = []
    for (d, b), r in rows_a.items():
        pr = prof["draws"][d]
        if plugin.model == "chunks":
            continue  # chunk totals are attributed differently; checked through failing shots below
        if pr["batch_toffolis"][b] != r["toffolis"]:
            det_bad.append({"draw": d, "batch": b, "profile": pr["batch_toffolis"][b], "rerun": r["toffolis"]})
    base_fail = {(d, f["batch"]): {"value": f["value_shots"], "phase": f["phase_shots"]}
                 for d, r in enumerate(prof["draws"]) for f in r["failing"]}
    for (d, b), r in rows_a.items():
        pf = base_fail.get((d, b), {"value": [], "phase": []})
        if (sorted(pf["value"]), sorted(pf["phase"])) != (sorted(r["value_shots"]), sorted(r["phase_shots"])):
            det_bad.append({"draw": d, "batch": b, "profile_fail": pf,
                            "rerun_fail": {"value": r["value_shots"], "phase": r["phase_shots"]}})
    det = {"paired_units_checked": len(rows_a), "equal": not det_bad, "mismatches": det_bad[:20]}
    if det_bad:
        raise RuntimeError(f"base profile is not reproducible: {det_bad[:3]}")
    cand_fail = {k: {"value": r["value_shots"], "phase": r["phase_shots"]} for k, r in rows_b.items()
                 if r["value_shots"] or r["phase_shots"]}
    checked = {k: base_fail.get(k, {"value": [], "phase": []}) for k in rows_b}
    cmp = compare_failures(checked, {k: cand_fail.get(k, {"value": [], "phase": []}) for k in rows_b})
    total_batches = prof["batches_per_draw"] * len(prof["draws"])
    precondition = {"verdict": "PASS" if not cmp["mismatches"] else "FAIL",
                    "base_failing_batches": len(fail_items), "base_failing_shots": prof["failing_shots"],
                    "paired_batches": n, "checked_batches": len(rows_b), "checked_shots": len(rows_b) * BATCH_SHOTS,
                    "total_shots": total_batches * BATCH_SHOTS, "coverage_pct": 100 * len(rows_b) / total_batches,
                    "matched_batches": cmp["matched"], "mismatches": cmp["mismatches"], "type_changes": cmp["type_changes"],
                    "meaning": "same failing-shot set on the checked batches (type changes do not alter k_d); "
                               "unchecked shots could still differ"}
    # dT for each reported N (schedule prefixes), from the paired units of the scheduled batches only
    report_n = sorted({int(x) for x in (report_n or [])} | {n})
    by_n = []
    from .pair import delta_stats
    for m in report_n:
        pairs = sched[:m]
        if plugin.model == "chunks":
            units = [u for u in res["_units"] if any((batches[i].draw, batches[i].index) in set(pairs) for i in u)]
            b_t = [sum(int(res["_rows_a"][i]["toffolis"]) for i in u) for u in units]
            c_t = [sum(int(res["_rows_b"][i]["toffolis"]) for i in u) for u in units]
            shots_u = [len(u) * BATCH_SHOTS for u in units]
        else:
            b_t = [prof["draws"][d]["batch_toffolis"][b] for d, b in pairs]
            c_t = [rows_b[(d, b)]["toffolis"] for d, b in pairs]
            shots_u = BATCH_SHOTS
        ds = delta_stats(b_t, c_t, shots_u)
        pred = predict(plugin, prof, cb.q, ds["mean_per_shot"])
        mean_t = sum(float(d["toffoli_mean"]) for d in prof["draws"]) / len(prof["draws"]) + ds["mean_per_shot"]
        row = {"N": m, "shots": ds["shots"], "delta_t": ds["mean_per_shot"], "delta_t_se": ds["se_per_shot"],
               "delta_constant": ds["constant"], "delta_range": [ds["min_unit_mean"], ds["max_unit_mean"]],
               "predicted_expected_cost": pred["expected_cost"],
               "rel_se": (ds["se_per_shot"] / mean_t) if ds["se_per_shot"] is not None else None}
        by_n.append(row)
    final = by_n[-1]
    pred = predict(plugin, prof, cb.q, final["delta_t"])
    full = None
    if sweep:
        full = full_sweep_cost(sweep)
        for row in by_n:
            row["full_expected_cost"] = full["expected_cost"]
            row["rel_error"] = rel_err(row["predicted_expected_cost"], full["expected_cost"])
            row["within_tie"] = abs(row["rel_error"]) <= REL_TIE
    rep["prediction"] = {
        "profile": {"path": str(profile_json), "sha256": prof["sha256"], "Q": prof["Q"], "expected_cost": prof["expected_cost"],
                    "draw_set": prof["draw_set"]},
        "model": pred["model"], "N": n, "by_n": by_n, "delta_t": final["delta_t"], "delta_t_se": final["delta_t_se"],
        "predicted_expected_cost": pred["expected_cost"], "predicted_per_draw": pred["per_draw"],
        "predicted_dcost_pct": 100 * (pred["expected_cost"] / prof["expected_cost"] - 1),
        "precondition": precondition, "determinism_check": det,
        "paired": [{"draw": d, "batch": b, "base_toffolis": rows_a[(d, b)]["toffolis"],
                    "candidate_toffolis": rows_b[(d, b)]["toffolis"]} for d, b in sched],
        "full_sweep": (({k: full[k] for k in ("path", "sha256", "draws", "expected_cost", "seconds")}
                        | {"rel_error": rel_err(pred["expected_cost"], full["expected_cost"]),
                           "within_tie": abs(rel_err(pred["expected_cost"], full["expected_cost"])) <= REL_TIE})
                       if full else None),
    }
    rep["seconds"] = time.perf_counter() - t_all
    return rep


def diff_family(dir_a, dir_b, *, verifier_path=None, threads=4, cache: ProgramCache | None = None, salt="",
                structural_layer=True) -> dict:
    """Chemistry (dfthcsos): two submission directories. The family cost is deterministic given the seed, so the
    plug-in prices each family by the full engine verification (official pipeline on the engine) and reports the
    cost, lambda_eff, Toffoli and qubit deltas; the ANGLE_k circuits are diffed structurally pair by pair."""
    import glob
    t0 = time.perf_counter()
    plugin = make_plugin("dfthcsos", verifier_path)
    rep = {"task": "dfthcsos", "engine_version": __version__, "a": {"path": str(dir_a)}, "b": {"path": str(dir_b)},
           "tier": "structural: static; cost: engine verify of the official pipeline (deterministic), local"}
    if structural_layer:
        pairs = []
        fa = {os.path.basename(p): p for p in glob.glob(os.path.join(str(dir_a), "*.kmb"))}
        fb = {os.path.basename(p): p for p in glob.glob(os.path.join(str(dir_b), "*.kmb"))}
        for name in sorted(set(fa) | set(fb), key=lambda s: (len(s), s)):
            if name in fa and name in fb:
                s = structural(fa[name], fb[name])
                pairs.append({"circuit": name, "identical": s["identical"], "num_hunks": s["num_hunks"],
                              "delta": s["delta"], "interface": s["interface"],
                              "hunks": s["hunks"][:20]})
            else:
                pairs.append({"circuit": name, "only_in": "a" if name in fa else "b"})
        rep["structural"] = {"circuits": pairs,
                             "static_toffoli_delta": sum(p["delta"]["static_toffolis"] for p in pairs if "delta" in p),
                             "identical_circuits": sum(1 for p in pairs if p.get("identical")),
                             "changed_circuits": sum(1 for p in pairs if "delta" in p and not p["identical"])}
    out = {}
    for side, d in (("a", dir_a), ("b", dir_b)):
        info = {}
        try:
            summ = plugin.verify_family(d, salt=salt, threads=threads, cache=cache, info=info)
            out[side] = {"status": "PASS", "score": summ["score"], "lambda_eff": summ["lambda_eff"],
                         "toffoli_count": summ["toffoli_count"], "num_qubits": summ["num_qubits"],
                         "submission_sha256": summ["submission_sha256"], "metrics": summ["metrics"],
                         "resources": summ["resources"], "seconds": info.get("wall_s")}
        except (AssertionError, ValueError, RuntimeError, OSError, KeyError) as ex:
            out[side] = {"status": "FAIL", "error": str(ex), "seconds": info.get("wall_s")}
        rep[side].update(out[side])
    if out["a"]["status"] == out["b"]["status"] == "PASS":
        rep["cost"] = {"a": out["a"]["score"], "b": out["b"]["score"],
                       "delta_pct": 100 * (out["b"]["score"] / out["a"]["score"] - 1),
                       "lambda_eff": [out["a"]["lambda_eff"], out["b"]["lambda_eff"]],
                       "toffoli_count": [out["a"]["toffoli_count"], out["b"]["toffoli_count"]],
                       "num_qubits": [out["a"]["num_qubits"], out["b"]["num_qubits"]]}
    rep["seconds"] = time.perf_counter() - t0
    return rep


def full_sweep_cost(path) -> dict:
    """Mean per-draw cost of a stored full sweep: a gate JSON (rows[].candidate.score) or a draws CSV."""
    import csv
    import hashlib
    p = str(path)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    if p.endswith(".csv"):
        with open(p) as f:
            rows = list(csv.DictReader(f))
        costs = [float(r["candidate_cost"]) for r in rows]
        seconds = None
    else:
        env = json.loads(open(p).read())
        rows = [r["candidate"] for r in env["rows"]]
        costs = [float(r["score"]) for r in rows]
        seconds = sum(float(r.get("seconds", 0)) for r in rows)
    return {"path": p, "sha256": h, "draws": len(costs), "expected_cost": sum(costs) / len(costs),
            "per_draw": costs, "seconds": seconds}


def write_new(path, obj):
    """Write JSON to a new file only (evidence directories are append-only)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "x") as f:
        f.write(json.dumps(obj, indent=1, default=_json_default) + "\n")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
