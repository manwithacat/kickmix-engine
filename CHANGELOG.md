# Changelog

All notable changes to kickmix-engine. Dates are UTC. The project was written by an AI model (Claude) under the
owner's direction; see NOTICE.

## 0.5.0 (2026-10-28)

### Added
- **Chemistry (`dfthcsos`) task.** `kickmix-engine verify dfthcsos SUBMISSION_DIR [--cross-check]` and
  `kickmix_engine.dfthcsos.verify(...)` reproduce the official `dfthcsos/verify.py` (FeMoCo76o113e): the official
  package is imported from the verifiers directory and its own functions do the loading, hashing, register checks,
  DF-THC metrics, cost model and score; the engine replaces kickmix's simulator, and the FCIDUMP is read through a
  cache. Options mirror the official CLI (`--factorization`, `--tree-edges`, `--format`, `--shots-per-address`,
  `--salt`, `--seed`), plus `--no-fcidump-cache` and `--no-low-memory`. `doctor` reports the dfthcsos verifier.
- `run_sequence(progs, seed, x0, xreg, *, threads, counting, dump)` (Rust `engine::run_sequence`): several programs
  applied in turn to one kickmix `Simulator(batch_size=shots, seed)`, as the dfthcsos verifier drives it. Each
  256-shot sim lives across all steps, so its ChaCha8 stream and operation counters continue; between steps it is
  cleared (`clear_for_shot`) and the carried register is written back. Registers wider than 64 qubits, and batch
  sizes that are not a multiple of 256 (the extra lanes of the last sim are zeroed between steps and not counted, as
  in kickmix), are supported. Per-step outputs: the carried register, a workspace flag, the phase flip and per-kind
  operation counts.
- `Sim::run_lanes` / `Sim::with_dims`: counting restricted to the real shots of a partly used sim, and sims sized for
  several programs. The counting-off and full-sim paths are the 0.4.0 loop (a const generic selects the masked
  variant), so invsqrt and DQI are unchanged.
- FCIDUMP cache: the official parser's packed float32 integrals, keyed by the FCIDUMP's SHA-256 and the SHA-256 of
  the official `reference.py`, under `<cache dir>/fcidump/`. The official `check_fcidump` (size and SHA-256 of the
  128 MB file) still runs on every verification. An entry is written only after a round trip reproduced the parsed
  `tei`, `h1_thc` and `oei` bit for bit, carries a SHA-256 of its arrays and is re-parsed if it does not verify.
- Lower peak memory with identical results: during `evaluate_words` the official Gram tensor (`reconstruct_h2`, an
  einsum that returns a transposed view of an N^2 x N^2 buffer) is transposed in place into C order, so the official
  `np.linalg.norm(h2_delta.ravel())` no longer copies 255 MiB. Values are moved, not recomputed.
- Tests: `tests/test_dfthcsos.py` (run_sequence against a chain of independent runs with the RNG carried by dummy
  `R` ops, partly used sims, wide registers, input errors, the in-place transpose; with kickmix: per-shot and
  per-count parity against kickmix's Simulator at 300 and 512 shots; with the official verifiers: FCIDUMP cache round
  trip and corruption, task parity on the sample submission). Scripts `tests/parity_dfthcsos.py` (per-shot parity of
  every step against the official loop, with negative controls) and `tests/task_parity_dfthcsos.py` (`--cross-check`
  on real and mutated families, PASS and FAIL).

### Changed
- Version 0.5.0. The program cache format is unchanged (2); entries are keyed by the extension hash as before.

## 0.4.0 (2026-10-28)

### Added
- Run-time kernel dispatch on x86_64 (`src/kernel.rs`). The simulator loop (`Sim::run`) and the exact DQI sampler are
  compiled for baseline x86-64, AVX2 (x86-64-v3: avx2, bmi1, bmi2, fma, lzcnt, movbe) and AVX-512 (v3 + avx512f,
  avx512bw, avx512vl, avx512dq) from one source. The best kernel the CPU supports is chosen once per process.
  `KICKMIX_ENGINE_KERNEL=baseline|avx2|avx512` forces one, and it is an error if the CPU lacks it. aarch64 keeps its
  single build, reported as `neon`.
- `kickmix_engine.kernel_info()` and `kickmix_engine.cpu_info()`. `verify --timing` and `--cross-check` report the
  kernel. The parity scripts record it in their JSON output.
- `kickmix-engine doctor [--json] [--manifest PATH_OR_URL]`: platform, Python, CPU features, compiled, available and
  selected kernels, engine version and extension hash, cache directory, whether kickmix and the official verifiers
  import, and (with a manifest) the matching wheel and its SHA-256.
- `release-manifest.json`, built by `tools/make_release_manifest.py`: every wheel with its tags, SHA-256, size,
  kernels and their required CPU features, and per-kernel CI test status, plus the version, commit, CI run and sdist.
  The workflow uploads it as an artifact on every run and attaches it to `v*` releases.
- `tools/ci_kernel_tests.py`: runs the tests once per kernel the machine supports and writes a status file.
- Tests for kernel selection and for the error when a kernel is unavailable.

### Changed
- CI (`wheels.yml`): the Linux x86_64 test job prints `kickmix-engine doctor --json` and runs the tests once per
  supported kernel (unsupported kernels are skipped with a notice). A new `manifest` job builds the manifest, and the
  release job attaches it and refuses to publish unless its `test_status` is `passed`. The macOS test commands are
  unchanged; a step now records their outcome for the manifest.
- `verify` exits 2 with `{"status": "ERROR"}` if `KICKMIX_ENGINE_KERNEL` cannot be honoured.
- The program cache format is unchanged (2). Entries are still keyed by the extension's hash, so 0.3.0 entries are
  not reused.

## 0.3.0 (2026-10-28, not released separately; included in 0.4.0)

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
