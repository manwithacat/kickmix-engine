# kmx-analysis (P1: streaming liveness / peak pass)

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
