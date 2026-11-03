# Changelog

All notable changes to kickmix-engine. Dates are UTC. The project was written by an AI model (Claude) under the
owner's direction; see NOTICE.

## Unreleased (branch `perf-invsqrt-resident`, 2026-11-03; on top of `paired-runner`)

### Changed (simulator speed; cache format 3)
- **Uniform runs.** The compiler gathers NEG, Z, X, CZ, CX, CCZ and CCX ops, with or without a bit condition, into
  runs of three-word entries that the simulator executes without a data-dependent branch:
  `t = base & bits[cb] & q[a] & q[b] & q[c]; q[d] ^= t`. The qubit array now starts with a constant all-ones slot
  (absent operands) and eight rotating write-only sinks (phase ops XOR into a sink and, through a mask derived from
  the sink index, into the register-resident phase); the bit array starts with a constant all-ones bit. Counts come
  from one popcount per condition group times the run's static histogram, which equals the ordinary path's per-op
  popcounts. Plain CX stretches of eight or more remain CX runs. Qubit ids must now be below 65,527.
- **Lockstep pairs.** `run` (the invsqrt chunk model) simulates two independent 256-shot sims per op (`SimG<8>`,
  one decode, two states, own RNG streams and counts), which lets the core overlap the read-modify-write chains of
  arithmetic-heavy circuits. `run(..., lockstep=False)` keeps the single-sim path; results are identical.
- Measured on the frontier invsqrt circuit `bce2948ff24e` (Apple M2 Pro, loaded machine, 1 thread, per 256-shot
  batch with counting): 242 us before, 166 us with uniform runs, 111 us with lockstep pairs (2.17x).
  Synthetic: a random CX/CZ/CCX/X/Z mix 6.5 -> 1.6 ns per op per 256 shots. The NumPy per-x check that the card
  named as the bottleneck measured about 1% of invsqrt wall time.

### Added
- `run(..., check=(target, tol))` (Rust `engine::Check`): the invsqrt verifier's per-shot check on the engine
  (x restored, scratch clean, |theta - target(x)| <= tol(x) in 2^-64-turn units with the verifier's wrapping
  arithmetic, its failure order within a batch, min margin and worst-ratio folds), returned per chunk; the adapter
  no longer moves 200M per-shot words into NumPy. `Program.num_uni_runs`.
- **Resident engine.** `kickmix-engine serve --socket PATH [--max-programs N]` keeps one process alive;
  `kickmix-engine --via PATH <command>` runs any command there (`ping`, `stats`, `stop` are control requests). The
  server runs the ordinary CLI with captured output, so a served command prints what the one-shot CLI prints and
  exits with its code. It keeps compiled programs by content hash (unchanged files, by realpath, size, mtime_ns and
  inode, are not re-read), task plug-ins with their reference tables and parsed profile JSONs (`cache.Memory`).
  Python: `kickmix_engine.resident.Client(socket).call(argv) / .json(argv)`.
- Tests: `tests/test_invsqrt_check.py` (engine check against the NumPy restatement: pass folds, zero tolerance,
  phase/x/dirty failures and their priority, first failing chunk and batch, bad inputs; uniform runs with
  bit-conditioned gates), `tests/test_resident.py` (served output equals one-shot, memory hits, error relay,
  `--via`). The existing suite is unchanged and passes.

## Unreleased (branch `paired-runner`, 2026-11-03)

### Added
- **`kickmix-engine diff TASK BASE CANDIDATE`** (`kickmix_engine.diff.diff`): compare any two kickmix circuits in
  three layers. (1) *Structural* (`_engine.structural_diff`, Rust, no simulation): op-level alignment in file order
  (common prefix streamed in lockstep, common suffix from 64-bit op hashes, the middle aligned with content-defined
  k-gram anchors extended op by op), the changed regions as hunks with per-kind and Toffoli deltas, static counts,
  qubits, bits and a register-interface check; 56M-op DQI circuits diff in about 4-8 s at under 300 MiB.
  (2) *Semantic* (`kickmix_engine.pair.pair_run`): both circuits on identical inputs and simulator seeds over a fixed
  schedule of batches; per shot the interface-state, phase (exact, in 2^-64 turns) and acceptance-verdict agreement,
  per paired unit the Toffolis executed on each side, in aggregate the per-shot mean Toffoli change with its standard
  error and a bounded disagreement list. (3) *Cost prediction* (`--profile`): the candidate's expected cost over a
  base profile's draw set through the task plug-in's price model, with the failing-shot precondition checked on the
  base's failing batches and every paired batch. Output is JSON; `--out` writes a new file only.
- **`kickmix-engine profile TASK BASE --out DIR`** (`kickmix_engine.diff.profile`): one pass over a draw set
  recording every batch's Toffolis and failing shots, reused by every later `diff --profile`. Draw sets are
  `--draws N --anchor HASH --salt-format 'sweep-{}'`: seeds `_initial_seed(anchor, salt)`, so with a fixed anchor two
  circuits pair draw for draw. The JSON schema is the one the VQCC project's `scripts/dqi_fast_rank.py` wrote, which
  is now a thin wrapper over this module.
- **Task plug-ins** (`kickmix_engine.plugins`): one small class per task supplies the input interface (from the
  register table, with the official rules), the draw/batch structure, the input sampler, the per-shot acceptance
  check and the price model; everything else is shared. `dqi` (engine `sample_shots`, value/phase verdicts,
  R(k) x Q^2 x mean Toffolis), `invsqrt` (chunked simulators, the official `reference_values` tolerance per x,
  margins, the rotation-cost rule from the smallest margin), `dfthcsos` (family-level: both submissions verified
  by the official pipeline on the engine, ANGLE_k circuits diffed structurally) and `generic` (any circuit:
  uniform inputs on named registers or explicit `qubits=`, restore check only, Q x mean Toffolis).
- `run_batches` returns `phase_units` [n, 256]: each shot's final phase in units of 2^-64 turns (flip included,
  the invsqrt verifier's rounding). Existing keys are unchanged.
- `program::KmbOps`: a pull iterator over the ops of a .kmb file (`SrcOp` with original ids and a content hash);
  `Program::from_kmb` now consumes it, so parsing has one implementation.
- Tests: `tests/test_diff.py` (structural diff on the fixture and byte-mutated copies, the generic paired runner
  on a self pair and a mutant, the pure helpers). All run without kickmix.

### Parity (VQCC project card `experiments/2026-11-03-engine-general-paired-runner.md`)
- `diff dqi --profile` reproduced the project's five `dqi_fast_rank.py` validation predictions exactly (predicted
  cost, Toffoli change and its SE, precondition verdict, every paired Toffoli total).
- The invsqrt plug-in reproduced the official verifier's summary on two frontier circuits (score, Toffoli mean,
  T count, rotations, eps, worst error/tolerance and worst x) through both the adapter and the profile path.
- A deliberately broken rewrite (one CX turned into a CZ, Toffoli-neutral) was isolated by the structural layer and
  failed by the semantic layer; the genuine rewrite agreed on every shot.

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
