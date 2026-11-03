# qrom-plan

Bit-packed GF(2) cost model for the VQCC chemistry `ANGLE_k` five-word QROM loaders (Lera's 128-qubit
`tensorqrom` emitter layout: 7 high / 4 low address bits, three low-cache phases). It ports the Python
kernel-relation planner that produced the Hedge Wizardry HW-12 and HW-13 loaders and adds an incremental word-flip
evaluator, so that word-level searches can price thousands of candidate flips.

Written by an AI model (Claude) under the direction of the repository owner, who reviews the work; see the crate's
tests and the VQCC experiment card `experiments/2026-11-03-chem-fast-qrom-cost-model.md`.

## Model

Loader `k` loads `delta = B_k XOR B_(k-1)` over 1,363 addresses. Its mixed ANF tensor has 85 high monomials, 15 low
monomials and 35 output bits; each high monomial is a row `F_h` in GF(2)^525. A **kernel relation** `r` has
`F_h . r = 0` for every `h`; a relation set `R` fixes `W = R^perp`. A **template** is `f (x) v` for a low form `f`
(at most two slots inside one cache phase) and a 35-bit direction `v`; `V_f` is the space of admissible `v`. A **plan**
is a template basis of `W`; two templates with the same `v` whose forms share a phase are one Toffoli (a paired
product). Toffolis per pass = 89 (caches) + products, so with `|R|` relations and perfect pairing
`T = 89 + (525 - |R|) / 2`.

A word flip at address `a`, column `j`, bit `b` changes the truth table of loaders `j / 5` and `j / 5 + 1` at one
entry; through the Mobius transform that is the rank-one update `F' = F + u (x) d`. The evaluator uses that:

* a plan relation survives iff `d . r = 0`; the survivors form a hyperplane of `span(R)`, so exactly one relation is
  lost whenever any is (re-base `r_i <- r_i + r_0`);
* the rank of any column sub-matrix `A` moves by +1 iff (`u` not in `Col A` and `d_A` not in `Row A`), by -1 iff
  (`u` in `Col A`, `d_A` in `Row A`, `d_A . x = 1` for `A x = u`), else 0; this prices the relation supply (single
  slots, within-phase slot pairs and triples, the full annihilator) without rebuilding anything;
* after a lost relation `W' = W + <d>`; one admissible template patches the fixed plan iff some `V_f` grows.

Bit-0 flips that change a column's coded status (bit 0 affine on every 16-address block) are not a single
truth-table change and fall back to the full recomputation.

## Commands

```sh
cargo build --release -p qrom-plan
qrom-plan info <words.npy> <k> [groups]                       # structure: rank, annihilator, kernel dims
qrom-plan check-plan <words.npy> <plan.txt>                   # exact verification of a stored plan
qrom-plan replan <words.npy> <plan.txt> <pair|structured|block> <seed_from> <seed_to> [trials] [samples] [repair_s] [out_prefix]
qrom-plan flips <words.npy> <plans_dir> <n> <seed> [threads] [examples] [exclude_coded 0|1]   # incremental vs full
qrom-plan bench <words.npy> <plans_dir> <n> <seed> [threads] [examples] [exclude_coded 0|1]   # incremental only
qrom-plan replan-flips <words.npy> <plans_dir> <n> <seed> [threads] [trials] [samples] [repair_s] [seeds_per_flip]
```

Plans are plain text (`k`, `groups`, `toffolis`, `rel 0x...`, `term a1 h1 a2 h2 s`, optional `count`/`seed`/`per`/
`design`); the VQCC script `scripts/chem_fast_qrom_plans.py` converts the planner's JSON. Words are a 2-D integer
`.npy` (addresses x word columns, 8-bit signed-gauge words). No third-party crates.

## Tests

`cargo test -p qrom-plan` runs unit tests and synthetic-table integration tests (structure, plan verification,
pairing with repair, rank-one update against direct ranks, incremental == full on random flips). With
`QROM_PLAN_WORDS` and `QROM_PLAN_PLANS` set, it also verifies every real plan in that directory.
