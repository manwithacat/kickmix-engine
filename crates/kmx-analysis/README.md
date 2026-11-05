# kmx-analysis (P1: streaming liveness / peak pass; P2: GF(2) echelon objects)

Static analysis over kickmix `.kmb` circuits, separate from the verify wheel. Engineering card E1
(`experiments/2026-11-05-engine-streaming-liveness.md` in the competition repo). The Python reference is
`scripts/qhdl_lifetime_allocator.py` / `scripts/qhdl_peak.py`; outputs are byte-identical on the parity set.

- `src/kmb.rs`: its own streaming `.kmb` reader (op types and the six qubit streams, chunked). The engine's
  `program::KmbOps` is not shared, so `src/` of the verify wheel is untouched.
- `src/liveness.rs`: one pass, one record per wire. Zero-after per reference (unconditional R/HMR at depth 0, plus
  the conditional-reset rule of `zero_after_refs` when `cond_rule`), intervals sorted by (wire, start), live profile,
  idle gaps for `qhdl peak`, allocator recolour.
- `py/`: PyO3 binding, extension `kickmix_analysis._kmx_analysis`; `python/kickmix_analysis/` is the package.

This directory is its own Cargo workspace on purpose, so the engine's root `Cargo.toml` and `Cargo.lock` stay
byte-identical.

```sh
cd crates/kmx-analysis
cargo rustc --release --offline -p kmx-analysis-py --lib -- -C link-arg=-undefined -C link-arg=dynamic_lookup
cp target/release/lib_kmx_analysis.dylib python/kickmix_analysis/_kmx_analysis.abi3.so
# separate venv (never the kickmix venv or the project): python3 -m venv --system-site-packages ../../.venv and a
# .pth file pointing at crates/kmx-analysis/python
```

Parity drivers: `parity/parity.py` (pinned circuits: intervals, profiles, `qhdl peak` JSON, allocator output) and
`parity/random_parity.py` (synthetic circuits with nested condition blocks and `_IF` resets).

## P2: GF(2) echelon objects (card E2, 2026-11-05)

`src/gf2.rs` (pure Rust, runtime widths; generalised from the engine's `crates/qrom-plan/src/gf2.rs`) and
`py/src/gf2_py.rs` (binding). Reference: `tools/dfthc-refit/recover.py` (`Basis`, `rank`, `nullspace`) and
`planner_shared.intersect` / `decompose` in the competition repository. Pivot = highest set bit of the partially
reduced vector, rows not mutually reduced, combination masks over the independent added vectors: every reduced
vector and mask equals the Python one, so the planners' random paths and plans are byte-identical.

- `Gf2Basis(width)`: `add`, `add_many`, `reduce`, `solve`, `solve_many`, `contains`, `decompose`, `truncate`,
  `vectors`, `len` (vectors are Python ints below `2**width`; wider ints raise OverflowError).
- `GroupSet(nb, width, groups)`: `nb` bases plus a table of groups `[(basis, vector), ...]` (a pair line's two
  ends) kept in Rust, so a pairing pass is a few calls: `build`, `build_values` (no masks), `independent`,
  `add_group`, `greedy`, `first_pair`, `positions`, `compute_status` + `enabled` (the rebuild exchange test),
  `compute_decomp` + `quotient_cands` + `quotient_pair` (`augment_quotient`), `add_vec`, `size`.
- `gf2_rank(rows)`, `gf2_nullspace(rows, width)`, `gf2_intersect(V1, V2, m)`.

Callers switch with `backend="rust"` (default `"python"`) in `tools/dfthc-refit` (`recover.make_basis/rank/
nullspace/structure`, `planner_shared.SharedPlanner/plan/intersect`, `planner_block.plan_generic`,
`price_words.gf2_rank`), or `DFTHC_GF2_BACKEND=rust` for the module default. Parity driver:
`parity/gf2_property.py REF_DFTHC_REFIT_DIR OUT.json` (randomized, against an unpatched copy).

## P3: tapped value simulator (card E3, 2026-11-05)

`src/valuesim.rs` (with `src/mt.rs`, CPython's Mersenne Twister, and `src/blake2b.rs`, unkeyed BLAKE2b) and
`py/src/valuesim_py.rs`. Reference: the bit-parallel value loop of `scripts/qhdl_simulate.py` (`simulate`) and its
copies in `scripts/dqi_l27c_andprints.py`, `dqi_l27c_blocks.py`, `dqi_l29c_rare.py` and `xpath_l33_dqi.py`. Values
only (phases ignored); one lane per shot; HMR outcomes are `random.Random(seed).getrandbits(L)` per HMR, drawn on
every HMR whatever the mask, exactly as the Python; `outcomes='zero'` and `'explicit'` give RNG-free runs.

- `value_sim(path, wires, cbits, lanes, *, seed, outcomes, points, windows, dep_cuts, prints, rare_cap, tflag,
  stop, threads)`: taps before the op at their index (index `num_ops` = end state): point snapshots of wire lists,
  window deltas (start ^ end per wire, nonzero only), GF(2) dependence of all wires plus the all-ones row at cuts
  (worker threads), and per Toffoli-like op the product lane count, fingerprints (`blake2b-64 >> 1`, canonical
  over complement), the andprints flags, rare-lane lists and the blocks target flags.
- `pack_lanes`, `words_to_int`, `mt_getrandbits`, `gf2_dependent` helpers.

The scripts gain `--backend python|rust` (default python); `qhdl_simulate.rust_module()` / `rust_state()` are shared.
Parity drivers live with the card's evidence (`outputs/engine-tapped-value-simulator-2026-11-05/drivers/`).
