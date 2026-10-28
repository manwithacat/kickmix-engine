# Changelog

All notable changes to kickmix-engine. Dates are UTC. The project was written by an AI model (Claude) under the
owner's direction; see NOTICE.

## 0.3.0 (2026-10-28, unreleased)

### Added
- `reaction_depth` is now computed. It follows kickmix's `Circuit.reaction_depth()` rule for rule, in the same pass
  that compiles the `.kmb` file, and is stored with the cached program. `Program.reaction_depth` exposes it, and
  `verify` reports it, so every summary field now matches the official verifier. `--cross-check` no longer skips it.
  Checked against kickmix 1.0.0 and the kickmix-macos 0.3.0 fork on random circuits and on the DQI and invsqrt
  competition circuits (`tests/reaction_depth_parity.py`, plus a stored fixture in `tests/test_engine.py`).
- `tests/fail_paths_crosscheck.py`: small mutated circuits built from the official reference circuits. They cover a
  dirty ancilla (all, part or about 1/32 of the shots), a syndrome not restored, x not restored (early and late in task
  order) and dirty scratch (including which qubit is named, and which failure is reported first), each run through
  `verify --cross-check`.
- `kickmix_engine.compile_text(text, num_qubits)`: compile kickmix circuit text without kickmix.
- `tests/test_rng.py` is now a pytest module with a stored reference (`tests/data/rng_keys_ref.npz`). It also has a
  test that builds the C++ reference `tests/chachakeys.cc` with the system compiler, and skips when no compiler is found.
- CHANGELOG.md; `[tool.pytest.ini_options]` so that `pytest` from the repository root runs `tests/`.

### Changed
- Program cache format 2: one more scalar field (`reaction_depth`). Entries from 0.2.0 are not reused; remove them
  with `kickmix-engine cache prune`.
- Every test and bench script takes its inputs from arguments or environment variables (`--kmb`, `--verifier`,
  `KICKMIX_ENGINE_VERIFIERS`, `FORK_PY`, `KMB`, ...). None of them refers to the author's machine any more. The scripts
  no longer put `python/` on `sys.path` themselves: install the wheel, or use `PYTHONPATH=python` after `./build.sh`.
- The CI workflow runs the whole `tests/` directory.
- README rewritten for new users: install, where to get the verifiers and the test circuits, the `verify` command,
  measured results with their caveats, how to reproduce the parity claims, and limitations.

## 0.2.0 (2026-10-28)

- DQI support: u32 instruction stream, streaming `.kmb` compiler, DQI task adapter, exact Rust re-implementation of
  the verifier's `sample_shots`, on-disk program cache, `kickmix-engine verify` CLI with `--cross-check`, and a GitHub
  Actions workflow for abi3 wheels (macOS arm64, Linux x86_64).

## 0.1.0 (2026-10-27)

- First Rust trial: bit-sliced simulator core matching kickmix per shot on the invsqrt circuit.
