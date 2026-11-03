"""Task plug-ins for the paired runner and `diff` (see `pair.py`, `diff.py`).

The engine runs any kickmix circuit; what differs between the competition tasks is (1) which registers are inputs,
(2) how inputs and seeds are drawn, (3) what a shot must satisfy (the acceptance check) and (4) how a circuit is
priced. A plug-in supplies exactly those four things, each through the official verifier module where one exists,
and nothing else: the simulation, the pairing, the agreement statistics and the profile/prediction machinery are
shared. The official verifier stays the authority; a plug-in's verdicts are checked against it (parity tests), and a
predicted cost is a projection, never a verified score.

    TaskPlugin.interface(prog, sha)      -> Interface: input qubit ids, input bit rows, scratch, Q, metadata
    TaskPlugin.batches(sha, draw)        -> the verifier's batch structure for one draw (seeds; sampler states)
    TaskPlugin.inputs(iface, batches)    -> bit-sliced inputs for those batches (+ expected outcomes)
    TaskPlugin.judge(iface, batches, inputs, out) -> per-batch verdicts: failing shots by type, Toffolis, margins
    TaskPlugin.price(profile_draws, q, delta_t)   -> expected cost of a circuit with Q = q whose per-shot Toffolis are
                                                     the profile's plus delta_t, same failing shots (static model)

Draw sets. A draw is one run of the verifier at one seed. `DrawSet(anchor, salt_format, count)` names `count` draws
whose seeds are `verifier._initial_seed(anchor, salt_format.format(i))`: when `anchor` is a fixed hash the seeds do
not depend on the circuit, so two circuits can be paired draw for draw (the project's 48-draw convention uses a
fixed anchor and salts sweep-0..47). With `anchor=None` the circuit's own hash is used (the official seeding).

Written by an AI model (Claude) under the owner's direction.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np

from . import KINDS, _engine, bools_to_words, run, run_batches, seed_words, words_to_bools

TOFFOLI_IDX = [KINDS.index(k) for k in ("CCX", "CCZ", "CSWAP")]
BATCH_SHOTS = 256


@dataclasses.dataclass
class DrawSet:
    """`count` verifier draws with seeds `_initial_seed(anchor or circuit hash, salt_format.format(i))`."""
    anchor: str | None = None
    salt_format: str = "sweep-{}"
    count: int = 1

    def salt(self, i: int) -> str:
        return self.salt_format.format(i)

    def describe(self) -> str:
        a = (self.anchor or "circuit")[:12]
        return f"salts {self.salt(0)}..{self.salt(self.count - 1)} anchored on {a}"


@dataclasses.dataclass
class Interface:
    """A circuit's input interface as the plug-in reads it from the register table."""
    prog: object
    sha: str
    task: str
    q: int
    qin_ids: list            # input qubit ids, row order of the input words
    bin_rows: list           # original classical-bit ids that are inputs (row order of the input words)
    bin_compact: list        # their compact ids in this program (rows whose bit no op touches are dropped)
    bin_keep: np.ndarray     # indices into bin_rows that are kept
    regs: dict               # name -> [(is_qubit, id)]
    meta: dict = dataclasses.field(default_factory=dict)

    @property
    def scratch(self) -> list:
        s = set(self.qin_ids)
        return [i for i in range(self.prog.num_qubits) if i not in s]


@dataclasses.dataclass
class Batch:
    """One verifier batch: its draw, index in the draw, sampler state and simulator seed."""
    draw: int
    index: int
    sim_seed: int
    sampler: object = None     # plug-in specific (DQI: rng seed; invsqrt: shot range; generic: rng seed)
    chunk: int = 0             # invsqrt: chunk index (one simulator per chunk; batches of a chunk share its stream)
    in_chunk: int = 0          # invsqrt: position in the chunk


def toffolis_of(kind_counts) -> np.ndarray:
    kc = np.asarray(kind_counts)
    return kc[..., TOFFOLI_IDX].sum(axis=-1)


class TaskPlugin:
    """Base: a generic circuit. Inputs are the qubits of the registers named in `inputs` (default: every register's
    qubits) and the classical bits of those registers; inputs are drawn uniformly at random per batch (seeded from the
    draw and batch index); there is no acceptance check beyond pairing; the price is Q x mean Toffolis."""
    name = "generic"
    model = "batches"   # 'batches': independent 256-shot simulators; 'chunks': one simulator across a run of batches

    def __init__(self, verifier_path=None, inputs=None, batches_per_draw=16, qubits=None, bits=None, **spec):
        self.verifier_path = verifier_path
        self.input_regs = inputs
        self.input_qubits = None if qubits is None else [int(q) for q in qubits]
        self.input_bits = None if bits is None else [int(b) for b in bits]
        self.batches_per_draw = int(batches_per_draw)
        self.spec = spec
        self.v = None

    # -- interface -------------------------------------------------------------------------------------------
    def interface(self, prog, sha) -> Interface:
        regs = dict(prog.registers)
        if self.input_qubits is not None or self.input_bits is not None:
            # explicit ids (for circuits without a register table)
            qin, brows = list(self.input_qubits or []), list(self.input_bits or [])
            if len(set(qin)) != len(qin) or any(q >= prog.num_qubits for q in qin):
                raise ValueError("input qubits must be distinct and below num_qubits")
            return self._iface(prog, sha, qin, brows, {"_q": [(True, q) for q in qin], "_b": [(False, b) for b in brows]})
        names = list(regs) if self.input_regs is None else list(self.input_regs)
        for n in names:
            if n not in regs:
                raise ValueError(f"register {n!r} not in circuit (has {list(regs)})")
        qin = [i for n in names for isq, i in regs[n] if isq]
        brows = [i for n in names for isq, i in regs[n] if not isq]
        if len(set(qin)) != len(qin):
            raise ValueError("input registers share a qubit")
        return self._iface(prog, sha, qin, brows, regs)

    def _iface(self, prog, sha, qin, brows, regs, **meta) -> Interface:
        comp = np.asarray(prog.bit_compact, dtype=np.int64)
        bc = comp[np.asarray(brows, dtype=np.int64)] if brows else np.zeros(0, np.int64)
        keep = np.flatnonzero(bc >= 0)
        return Interface(prog=prog, sha=sha, task=self.name, q=int(prog.num_qubits), qin_ids=[int(q) for q in qin],
                         bin_rows=[int(b) for b in brows], bin_compact=[int(x) for x in bc[keep]], bin_keep=keep,
                         regs=regs, meta=dict(meta))

    # -- draws -----------------------------------------------------------------------------------------------
    def draw_seed(self, sha, draws: DrawSet, i: int) -> int:
        import hashlib
        anchor = draws.anchor or sha
        salt = draws.salt(i)
        if self.v is not None and hasattr(self.v, "_initial_seed"):
            return int(self.v._initial_seed(anchor, salt))
        return int(hashlib.sha256(f"{anchor}{salt}".encode()).hexdigest(), 16) if salt else int(anchor, 16)

    def batches(self, sha, draws: DrawSet, i: int) -> list[Batch]:
        base = self.draw_seed(sha, draws, i)
        n = self.batches_per_draw
        seeds = np.random.SeedSequence(base, pool_size=8).generate_state(8 * n, dtype=np.uint64).reshape(-1, 2, 4)
        return [Batch(draw=i, index=b, sim_seed=int.from_bytes(seeds[b, 1], "little"),
                      sampler=int.from_bytes(seeds[b, 0], "little")) for b in range(n)]

    # -- inputs ----------------------------------------------------------------------------------------------
    def inputs(self, iface: Interface, batches: list[Batch]) -> dict:
        """Uniform random input bits per batch, seeded by the batch's sampler seed (independent of the circuit)."""
        nq, nb = len(iface.qin_ids), len(iface.bin_rows)
        qin = np.empty((len(batches), nq, 4), np.uint64)
        bin_all = np.empty((len(batches), nb, 4), np.uint64)
        for i, b in enumerate(batches):
            rng = np.random.default_rng(b.sampler)
            qin[i] = rng.integers(0, 2**64, size=(nq, 4), dtype=np.uint64) if nq else 0
            bin_all[i] = rng.integers(0, 2**64, size=(nb, 4), dtype=np.uint64) if nb else 0
        return dict(qin=qin, bin_all=bin_all, bin=bin_all[:, iface.bin_keep] if nb else bin_all, expected=None)

    # -- execution -------------------------------------------------------------------------------------------
    def execute(self, iface: Interface, batches: list[Batch], inputs: dict, threads: int, dump: bool) -> dict:
        """Run the batches on this circuit; returns the per-shot record the pairing layer compares:
        state_q [n, nq, 4] (dump only), phase_units [n, 256], toffolis [n], kind_counts [n, K], z_levels [n, 129],
        val_err [n, 256] (an input qubit not restored or scratch nonzero), phase_flip [n, 256]."""
        seeds = np.stack([seed_words(b.sim_seed) for b in batches])
        r = run_batches(iface.prog, seeds, iface.qin_ids, inputs["qin"], iface.bin_compact, inputs["bin"],
                        threads=threads, dump=dump)
        out = dict(phase_units=r["phase_units"], toffolis=toffolis_of(r["kind_counts"]).astype(np.int64),
                   kind_counts=r["kind_counts"].astype(np.int64), z_levels=r["z_levels"].astype(np.int64),
                   val_err=words_to_bools(r["val_err"]), phase_flip=words_to_bools(r["phase"]))
        if dump:
            out["state_q"] = r["dump_q"]
        return out

    # -- acceptance ------------------------------------------------------------------------------------------
    def judge(self, iface: Interface, batches: list[Batch], inputs: dict, out: dict) -> list[dict]:
        """Per batch: value_shots (input not restored / scratch dirty), phase_shots (none for a generic circuit),
        toffolis. The generic plug-in has no phase reference, so it reports the restore check only."""
        return [dict(index=i, value_shots=np.flatnonzero(out["val_err"][i]).tolist(), phase_shots=[],
                     toffolis=int(out["toffolis"][i])) for i in range(len(batches))]

    # -- price -----------------------------------------------------------------------------------------------
    def price(self, profile_draws: list[dict], q: int, delta_t: float, iface: Interface | None = None) -> dict:
        per = [q * (float(d["toffoli_mean"]) + delta_t) for d in profile_draws]
        return dict(per_draw=per, expected_cost=sum(per) / len(per), model="Q x mean Toffolis (generic)")

    def draw_summary(self, iface: Interface, rows: list[dict], q: int) -> dict:
        ve = sum(len(r["value_shots"]) for r in rows)
        pe = sum(len(r["phase_shots"]) for r in rows)
        shots = len(rows) * BATCH_SHOTS
        tmean = sum(r["toffolis"] for r in rows) / shots
        return dict(value_errors=ve, phase_errors=pe, toffoli_mean=tmean, score=q * tmean)


# ---------------------------------------------------------------- DQI

class DQIPlugin(TaskPlugin):
    """DQI (Benchmark II): registers S0..S{n-1} (syndrome qubits) and E0..E{m-1} (classical mask bits); inputs and
    expected parity phases from the official `sample_shots` (engine re-implementation); a shot fails by value (an
    input qubit not restored or another qubit nonzero) or by phase (phase flip differs from the sampled parity);
    price R(k) x Q^2 x mean Toffolis with R from the verifier's Clopper-Pearson bound and repetition overhead."""
    name = "dqi"

    def __init__(self, verifier_path=None, **spec):
        from .dqi import load_verifier
        super().__init__(verifier_path, **spec)
        self.v = load_verifier(verifier_path)
        kw = {k: x for k, x in spec.items() if k in ("m", "n", "degree", "r", "num_shots") and x is not None}
        self.pspec = self.v.ProblemSpecDQI.benchmark_instance(**kw)
        self.batches_per_draw = self.pspec.num_shots // BATCH_SHOTS
        self._sargs = None

    def interface(self, prog, sha) -> Interface:
        from .dqi import validate_registers_kmb
        if prog.static_counts[KINDS.index("Z_POW")]:
            raise ValueError("Z_POW gates are not allowed in Benchmark II.")
        syn, e_ids = validate_registers_kmb(prog.registers, self.pspec)
        return self._iface(prog, sha, syn, e_ids, dict(prog.registers))

    def batches(self, sha, draws: DrawSet, i: int) -> list[Batch]:
        base = self.draw_seed(sha, draws, i)
        n = self.batches_per_draw
        seeds = np.random.SeedSequence(base, pool_size=8).generate_state(8 * n, dtype=np.uint64).reshape(-1, 2, 4)
        return [Batch(draw=i, index=b, sampler=int.from_bytes(seeds[b, 0], "little"),
                      sim_seed=int.from_bytes(seeds[b, 1], "little")) for b in range(n)]

    def sampler_args(self):
        if self._sargs is None:
            v, spec = self.v, self.pspec
            tables = v.FieldTables(v.km.GF2Field(spec.degree))
            wp = np.asarray(v.optimal_dqi_weights(spec.m, spec.max_errors, spec.degree, spec.r) ** 2, dtype=np.float64)
            cdf = wp.cumsum()
            cdf /= cdf[-1]
            self._sargs = dict(m=spec.m, n=spec.n, degree=spec.degree, cdf=[float(x) for x in cdf],
                               log=[int(x) for x in tables.log], antilog=[int(x) for x in tables.antilog])
        return self._sargs

    def inputs(self, iface: Interface, batches: list[Batch], threads: int = 1) -> dict:
        from .dqi import pcg_state_words
        a, spec = self.sampler_args(), self.pspec
        pcg = np.array([pcg_state_words(b.sampler) for b in batches], dtype=np.uint64).reshape(-1, 6)
        all_rows = list(range(spec.m * spec.degree))
        r = _engine.sample_dqi(a["m"], a["n"], a["degree"], a["cdf"], a["log"], a["antilog"], all_rows, pcg, threads, False)
        k = len(batches)
        bin_all = r["bin"].reshape(k, len(all_rows), 4)
        return dict(qin=r["qin"].reshape(k, spec.n * spec.degree, 4), bin_all=bin_all,
                    bin=np.ascontiguousarray(bin_all[:, iface.bin_keep]),
                    expected=words_to_bools(r["phase"].reshape(k, 4)))

    def judge(self, iface, batches, inputs, out) -> list[dict]:
        val, ph, exp = out["val_err"], out["phase_flip"], inputs["expected"]
        perr = (ph ^ exp) & ~val
        return [dict(index=i, value_shots=np.flatnonzero(val[i]).tolist(), phase_shots=np.flatnonzero(perr[i]).tolist(),
                     toffolis=int(out["toffolis"][i])) for i in range(len(batches))]

    def r_of_k(self, k: int, shots: int | None = None) -> float:
        v = self.v
        shots = shots or self.pspec.num_shots
        return float(v.repetition_overhead(v.clopper_pearson_upper_bound(k, shots, log2_delta=v.DEFAULT_LOG2_DELTA)))

    def price(self, profile_draws, q, delta_t, iface=None) -> dict:
        per = []
        for d in profile_draws:
            k = int(d["value_errors"]) + int(d["phase_errors"])
            per.append(self.r_of_k(k) * q * q * (float(d["toffoli_mean"]) + delta_t))
        return dict(per_draw=per, expected_cost=sum(per) / len(per), model="R(k) x Q^2 x mean Toffolis (DQI)")

    def draw_summary(self, iface, rows, q) -> dict:
        s = super().draw_summary(iface, rows, q)
        s["score"] = self.r_of_k(s["value_errors"] + s["phase_errors"]) * q * q * s["toffoli_mean"]
        return s


# ---------------------------------------------------------------- invsqrt

class InvSqrtPlugin(TaskPlugin):
    """invsqrt (Benchmark I): register x (22 qubits); shot i has x = i mod max_x + 1; one simulator per chunk of 64
    batches (its measurement stream continues across the chunk, so paired items are chunk prefixes); a shot fails if
    x is not restored, a scratch qubit is nonzero or |phase - target(x)| > tol(x) (the official reference_values);
    price Q x (mean Toffolis + (T + n_rot x rotation_t_count(eps)) / 4) with eps from the smallest margin, as the
    official cost model computes it."""
    name = "invsqrt"
    model = "chunks"

    def __init__(self, verifier_path=None, **spec):
        from .invsqrt import load_verifier
        super().__init__(verifier_path, **spec)
        self.v = v = load_verifier(verifier_path)
        self.pspec = spec.get("spec") or v.ProblemSpecInvSqrt.for_n_p(v.N_P)
        self.chunk_batches = v.BATCHES_PER_CHUNK
        self.batches_per_draw = self.pspec.num_shots // BATCH_SHOTS
        self._ref = None

    def reference(self):
        if self._ref is None:
            self._ref = self.v.reference_values(self.pspec, np.arange(1, self.pspec.max_x + 1, dtype=np.uint64))
        return self._ref

    def interface(self, prog, sha) -> Interface:
        from .invsqrt import validate_registers_kmb
        x_ids, scratch = validate_registers_kmb(prog.registers, prog.num_qubits, self.pspec)
        return self._iface(prog, sha, x_ids, [], dict(prog.registers))

    def batches(self, sha, draws: DrawSet, i: int) -> list[Batch]:
        v, spec = self.v, self.pspec
        base = self.draw_seed(sha, draws, i)
        chunk_shots = self.chunk_batches * BATCH_SHOTS
        starts = range(0, spec.num_shots, chunk_shots)
        seeds = np.random.SeedSequence(base, pool_size=8).generate_state(4 * len(starts), dtype=np.uint64).reshape(-1, 4)
        out = []
        for c, s in enumerate(starts):
            seed = int.from_bytes(seeds[c], "little")
            stop = min(s + chunk_shots, spec.num_shots)
            for j, st in enumerate(range(s, stop, BATCH_SHOTS)):
                out.append(Batch(draw=i, index=len(out), sim_seed=seed, sampler=(st, st + BATCH_SHOTS), chunk=c, in_chunk=j))
        return out

    def inputs(self, iface, batches, threads: int = 1) -> dict:
        xs = np.stack([self.v.sample_shots(self.pspec, *b.sampler) for b in batches])  # [n, 256] uint64
        return dict(xs=xs, expected=None)

    def execute(self, iface, batches, inputs, threads, dump) -> dict:
        """Runs whole chunk prefixes: the batches must be grouped by (draw, chunk) and each group must be a prefix
        of its chunk (in_chunk 0..b-1). Records are returned per batch in the given order."""
        groups = {}
        for i, b in enumerate(batches):
            groups.setdefault((b.draw, b.chunk), []).append(i)
        n = len(batches)
        xs = inputs["xs"]
        rec = dict(phase_units=np.zeros((n, 256), np.uint64), toffolis=np.zeros(n, np.int64),
                   kind_counts=np.zeros((n, len(KINDS)), np.int64), z_levels=np.zeros((n, 129), np.int64),
                   val_err=np.zeros((n, 256), bool), phase_flip=np.zeros((n, 256), bool),
                   x_out=np.zeros((n, 256), np.uint64), dirty=np.zeros((n, 256), bool))
        if dump:
            rec["state_q"] = np.zeros((n, iface.prog.num_qubits, 4), np.uint64)
        # Groups with the same prefix length run in one engine call.
        by_len = {}
        for key, idx in groups.items():
            seq = [batches[i].in_chunk for i in idx]
            if seq != list(range(len(idx))):
                raise ValueError(f"chunk {key} must be a prefix of its batches, got in_chunk {seq}")
            by_len.setdefault(len(idx), []).append(idx)
        for blen, glist in by_len.items():
            seeds = [seed_words(batches[idx[0]].sim_seed) for idx in glist]
            x = np.stack([xs[idx] for idx in glist])  # [g, blen, 256]
            r = run(iface.prog, seeds, x, iface.qin_ids, s=1, threads=threads, counting=True, dump=dump)
            for g, idx in enumerate(glist):
                for j, i in enumerate(idx):
                    rec["phase_units"][i] = r["phase_units"][g, j]
                    rec["x_out"][i] = r["x_out"][g, j]
                    rec["dirty"][i] = r["scratch_dirty"][g, j].astype(bool)
                    rec["val_err"][i] = (r["x_out"][g, j] != xs[i]) | rec["dirty"][i]
                    rec["phase_flip"][i] = (r["phase_units"][g, j] >> np.uint64(63)).astype(bool)
                    if dump:
                        rec["state_q"][i] = r["dump_q"][g * blen + j]
                # Counts are per chunk (the simulator's totals over its batches); attributed to the chunk's last
                # batch so that sums over a chunk are exact and per-batch means use chunk totals.
                last = idx[-1]
                rec["kind_counts"][last] = r["kind_counts"][g]
                rec["z_levels"][last] = r["z_levels"][g]
                rec["toffolis"][last] = toffolis_of(r["kind_counts"][g])
        return rec

    def judge(self, iface, batches, inputs, out) -> list[dict]:
        target, tol = self.reference()
        xs = inputs["xs"]
        rows = []
        for i, b in enumerate(batches):
            allowed = tol[xs[i] - np.uint64(1)]
            error = (out["phase_units"][i] - target[xs[i] - np.uint64(1)]).view(np.int64)
            used = np.abs(error).view(np.uint64)
            bad_ph = used > allowed
            ratio = np.divide(used, allowed, out=np.zeros(256), where=allowed > 0)
            rmax = float(ratio.max())
            rows.append(dict(index=i, value_shots=np.flatnonzero(out["val_err"][i]).tolist(),
                             phase_shots=np.flatnonzero(bad_ph & ~out["val_err"][i]).tolist(),
                             toffolis=int(out["toffolis"][i]), t_like=int(out["z_levels"][i, 3]),
                             rotations=int(out["z_levels"][i, 4:].sum()),
                             min_margin=int((allowed - used).min()), worst=rmax,
                             worst_x=int(xs[i][ratio == rmax].min()), margins=(allowed - used).view(np.int64)))
        return rows

    def cost(self, q, toffoli_mean, t_base, n_rot, min_margin) -> dict:
        v = self.v
        eps = math.pi * min_margin / 2**64 / n_rot if n_rot else None
        if n_rot and not eps > 0:
            return dict(score=math.inf, eps=eps, toffoli_equivalent=math.inf, reason="No tolerance left for rotation synthesis error.")
        t_count = t_base + (n_rot * v.rotation_t_count(eps) if n_rot else 0)
        tofs = toffoli_mean + t_count / v.T_PER_TOFFOLI
        return dict(score=q * tofs, eps=eps, toffoli_equivalent=tofs)

    def draw_summary(self, iface, rows, q) -> dict:
        s = super().draw_summary(iface, rows, q)
        shots = len(rows) * BATCH_SHOTS
        s["t_count"] = sum(r["t_like"] for r in rows) / shots
        s["rotation_count"] = sum(r["rotations"] for r in rows) / shots
        s["min_margin"] = min(r["min_margin"] for r in rows)
        s["worst"] = max((r["worst"], -r["worst_x"]) for r in rows)
        s["worst_x"] = -s["worst"][1]
        s["worst"] = s["worst"][0]
        c = self.cost(q, s["toffoli_mean"], s["t_count"], s["rotation_count"], s["min_margin"])
        s["rotation_eps"] = c["eps"]
        s["score"] = c["score"] if not (s["value_errors"] or s["phase_errors"]) else math.inf
        return s

    def price(self, profile_draws, q, delta_t, iface=None) -> dict:
        per = []
        for d in profile_draws:
            if d["value_errors"] or d["phase_errors"]:
                per.append(math.inf)
                continue
            c = self.cost(q, float(d["toffoli_mean"]) + delta_t, float(d["t_count"]), float(d["rotation_count"]),
                          int(d["min_margin"]))
            per.append(c["score"])
        return dict(per_draw=per, expected_cost=sum(per) / len(per),
                    model="Q x (mean Toffolis + (T + n_rot x rot_T(eps(min margin))) / 4) (invsqrt); a failing draw is inf")


# ---------------------------------------------------------------- chemistry

class ChemistryPlugin(TaskPlugin):
    """dfthcsos (chemistry): a family of ANGLE_k circuits plus a factorization, verified deterministically by the
    official pipeline on the engine (`dfthcsos.verify`). Its circuits have no per-shot tolerance: a family either
    passes (words recovered, residuals within 20 mHa) or fails, and the cost is lambda_eff x Q_total x T_total. The
    plug-in therefore prices by running the full engine verification (seconds), and pairs individual ANGLE_k circuits
    generically (uniform inputs on the selection register) when asked to diff two circuits of a family."""
    name = "dfthcsos"

    def __init__(self, verifier_path=None, **spec):
        super().__init__(verifier_path, **spec)

    def verify_family(self, directory, *, salt="", seed=None, threads=1, cache=None, info=None) -> dict:
        from . import dfthcsos
        return dfthcsos.verify(str(directory), salt=salt, seed=seed, threads=threads, verifier=self.verifier_path,
                               cache=cache, info=info)

    def price(self, profile_draws, q, delta_t, iface=None) -> dict:
        raise NotImplementedError("chemistry is priced by verify_family (deterministic official pipeline on the engine)")


PLUGINS = {"generic": TaskPlugin, "dqi": DQIPlugin, "invsqrt": InvSqrtPlugin, "dfthcsos": ChemistryPlugin}


def make_plugin(name: str, verifier_path=None, **spec) -> TaskPlugin:
    """A task plug-in; in a resident engine (``kickmix-engine serve``) the same (task, verifier, spec) returns the
    same instance, so reference tables such as invsqrt's `reference_values` are computed once per process."""
    if name not in PLUGINS:
        raise ValueError(f"unknown task plug-in {name!r}; have {sorted(PLUGINS)}")
    from .cache import memory
    mem = memory()
    if mem is None:
        return PLUGINS[name](verifier_path=verifier_path, **spec)
    key = (name, verifier_path, repr(sorted(spec.items())))
    if key in mem.plugins:
        mem.stats["plugin_hits"] += 1
    else:
        mem.plugins[key] = PLUGINS[name](verifier_path=verifier_path, **spec)
    return mem.plugins[key]
