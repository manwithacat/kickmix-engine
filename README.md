# kickmix-engine

kickmix-engine is a CPU simulator for [kickmix](https://github.com/quantumlib/Qualtran/tree/main/kickmix) circuits,
written in Rust, plus a `kickmix-engine verify` command for the **invsqrt** and **DQI** tasks of the Kaggle competition
[Google Quantum AI: Verifiable Quantum Circuit Compilation](https://www.kaggle.com/competitions/google-verifiable-quantum-circuit-compilation)
(VQCC). It is an independent re-implementation of kickmix's simulation semantics. It has been checked shot for shot
against kickmix: every qubit, every classical bit, the exact phase and the operation counts. `verify` runs the
competition's official verifier logic with this engine as the simulator and prints the same summary JSON. With
`--cross-check` it also runs the official verifier and compares every field.

**Not affiliated with Google, Qualtran or Kaggle.** The official verifier running on kickmix is the reference.
This engine gets the same answer with less memory and in less time, and it can check itself against the reference
whenever you ask it to.

**Who wrote it.** The code, tests and documentation were written by an AI model (Claude, by Anthropic) under the
direction of the repository owner, James Barlow, who reviews the work and is responsible for it. See [NOTICE](NOTICE).

## Install

Requirements: Python 3.12 or later on macOS arm64 or Linux x86_64. NumPy is installed as a dependency.

```sh
# Wheels are attached to each GitHub release (abi3: one wheel per platform serves every Python >= 3.12).
pip install https://github.com/manwithacat/kickmix-engine/releases/download/v0.4.0/kickmix_engine-0.4.0-cp312-abi3-macosx_11_0_arm64.whl
pip install https://github.com/manwithacat/kickmix-engine/releases/download/v0.4.0/kickmix_engine-0.4.0-cp312-abi3-manylinux_2_28_x86_64.whl

# kickmix must also be importable: the official verifier modules import it, even though the engine never uses it to simulate.
pip install kickmix   # kickmix 1.0.0 from PyPI, or the faster fork below
```

Each release has a `SHA256SUMS.txt` and a machine-readable `release-manifest.json` (see
[For agents and automation](#for-agents-and-automation)). The Linux x86_64 wheel contains three simulator kernels
(baseline, AVX2, AVX-512) and picks the best one for the CPU at start-up; there is no separate AVX wheel to choose.
For kickmix you can use the PyPI package (the official build) or the
[kickmix-macos](https://github.com/manwithacat/kickmix-macos) fork, an openly AI-assisted fork by the same owner. The
benchmarks below compare against both.

Building from source needs Rust 1.80 or later:

```sh
pip install maturin && maturin build --release --locked -o dist && pip install dist/*.whl
# For an in-place development build on macOS: ./build.sh, then put python/ on PYTHONPATH.
```

## What you need besides the package

- **The official verifiers.** `verify_submission_dqi.py` and `verify_submission_invsqrt.py` ship in the `verifiers/`
  directory of the competition data. Download them from the competition's Data tab, or with
  `kaggle competitions download -c google-verifiable-quantum-circuit-compilation` once you have accepted the rules.
  They are not redistributed here. Point the engine at that directory:
  `export KICKMIX_ENGINE_VERIFIERS=/path/to/verifiers` (it must contain `dqi/` and `invsqrt/`).
- **Circuits to verify.** Use your own `.kmb` files, the competition's `sample_submission/`, or the public Kaggle
  dataset [manwithacat/vqcc-apache-public-circuits](https://www.kaggle.com/datasets/manwithacat/vqcc-apache-public-circuits).
  That dataset collects circuits published in Apache-2.0 notebooks, including the two DQI circuits used below
  (`dqi/4ee7a17b4cbc.kmb`, `dqi/b3354f60e459.kmb`) and `invsqrt/07314803a092.kmb`.

## Quickstart

```sh
export KICKMIX_ENGINE_VERIFIERS=/path/to/verifiers

kickmix-engine verify dqi circuits/dqi/4ee7a17b4cbc.kmb                    # prints the official summary JSON
kickmix-engine verify dqi circuits/dqi/4ee7a17b4cbc.kmb --salt my-salt --cross-check --official-workers 4
kickmix-engine verify invsqrt sample_submission/invsqrt.kmb --cross-check
```

From Python:

```python
from kickmix_engine.dqi import DQITask
summary = DQITask("4ee7a17b4cbc.kmb").verify(salt="", threads=8)   # the official verify() fields
```

## The `verify` command

```
kickmix-engine verify {dqi,invsqrt} CIRCUIT.kmb [--salt S] [--seed N] [--threads N] [--cross-check] [options]
kickmix-engine cache {info,prune,clear} [--cache-dir DIR]
kickmix-engine doctor [--json] [--manifest PATH_OR_URL]
python -m kickmix_engine ...    # same thing
```

- Output: the official verifier's summary fields, with the same names and values, including `reaction_depth` and
  `circuit_sha256`. On failure it prints `{"status": "FAIL", "error": ...}` with the official message text, as the
  official CLI does.
- Exit codes: 0 PASS; 1 FAIL; 3 when `--cross-check` finds any difference.
- `--cross-check` also runs the official verifier CLI, unchanged, in `--official-python` (default: the same
  interpreter), with `--official-workers` processes (default: the verifier's own default). It prints
  `{"match": bool, "differences": {...}, "engine": {...}, "official": {...}, "seconds": {...}}`. Every field is
  compared, on PASS and on FAIL.
- The task options mirror the official CLI's: `--m --n --degree --r --num-shots --log2-delta` for DQI, and
  `--n-p --beta --precision-bits --shots-per-x --eta --width --max-x --system-qubits` for invsqrt.
- `--verifier PATH` selects one verifier file. If it is not given, the engine uses `$KICKMIX_ENGINE_DQI_VERIFIER` or
  `$KICKMIX_ENGINE_INVSQRT_VERIFIER`, and then `$KICKMIX_ENGINE_VERIFIERS/{dqi,invsqrt}/verify_submission_*.py`.
- DQI input sampling: `--sampler engine` (the default) draws the inputs in Rust, reproducing the verifier's
  `sample_shots` draw for draw from NumPy's PCG64 stream. `--sampler official` calls the verifier's own `sample_shots`.
  `--self-check K` re-draws K batches (or all of them, with -1) with the official function and fails on any difference.
- Compiled programs are cached on disk, keyed by the SHA-256 of the circuit file and the engine build. The cache lives
  in `--cache-dir`, `$KICKMIX_ENGINE_CACHE`, `$XDG_CACHE_HOME/kickmix-engine` or `~/.cache/kickmix-engine`, and
  `--no-cache` turns it off. A DQI program takes about 257 MiB. `cache prune` removes entries written by other builds.
- `--timing` prints engine timings, cache events and the simulator kernel (`kernel`, how it was selected, the CPU
  features detected) to stderr. `--cross-check` output also carries `engine_kernel`. The summary itself has only the
  official verifier's fields, so it stays comparable.
- Exit code 2 (with `{"status": "ERROR", ...}`) when `KICKMIX_ENGINE_KERNEL` names a kernel this CPU cannot run.

## Kernels (x86_64)

The simulator loop and the DQI sampler are compiled three times into the x86_64 build: **baseline** (any x86-64
CPU), **avx2** (the x86-64-v3 set: `avx2 bmi1 bmi2 fma lzcnt movbe`) and **avx512** (v3 plus
`avx512f avx512bw avx512vl avx512dq`). Each copy is the same Rust source compiled with different target features, so
all three must give the same results bit for bit. At the first simulation call the engine picks the best kernel the
CPU supports (`is_x86_feature_detected!`). To force one, for testing or benchmarking:

```sh
KICKMIX_ENGINE_KERNEL=baseline kickmix-engine verify dqi c.kmb --timing   # or avx2, avx512; auto = default
```

The choice is made once per process. A kernel the CPU lacks is an error, not a silent fallback. On aarch64 (Apple
Silicon) there is a single kernel, reported as `neon`. From Python, `kickmix_engine.kernel_info()` returns the
selected kernel and the detected features, and `kickmix_engine.cpu_info()` returns the same information without
selecting a kernel. The parity scripts (`tests/parity.py`, `tests/parity_dqi.py`) record the kernel in their JSON
output. CI runs the tests once per kernel that the Linux runner supports.

## For agents and automation

Two machine-readable entry points let a program pick and check the right artefact without guessing:

- **`release-manifest.json`**, attached to every GitHub release and uploaded as the `release-manifest` artifact of
  every CI run (schema `kickmix-engine-release-manifest/1`). It lists `engine_version`, `git_commit`, `git_ref`,
  `ci_run_id`, `ci_run_url`, an overall `test_status` (`passed`, `failed` or `incomplete`) and `parity_evidence`.
  For every wheel it gives `filename`, `python_tag`, `abi_tag`, `platform_tag`, `os`, `arch`, `sha256`, `size`,
  `url` (releases only), `kernels` (each with the CPU features it `requires`) and `test_status.per_kernel`
  (`passed`, `failed`, `skipped: <reason>` or `not run on CI`), plus the sdist. It is built by
  `tools/make_release_manifest.py DIST_DIR --status-dir STATUS_DIR`, which takes the kernel feature lists from
  `src/kernel.rs` and the per-kernel test outcomes from the CI test jobs (`tools/ci_kernel_tests.py`).
- **`kickmix-engine doctor --json [--manifest PATH_OR_URL]`** (schema `kickmix-engine-doctor/1`) reports `platform`,
  `python`, `cpu.features`, `kernels` (`compiled`, `available`, `selected`, `selected_by`, `requires`, `error`),
  `engine` (version, cache format, extension path and SHA-256), `cache`, `kickmix` (importable, version) and
  `verifiers` (path, SHA-256, importable) for DQI and invsqrt. With `--manifest` it adds `manifest.match`, the wheel
  in that release that fits this interpreter and platform with its `sha256`, whether the selected kernel is in it, and
  whether the installed version is the manifest's. It exits 0, or 2 if the kernel override is invalid. Without
  `--json` it prints a short human-readable summary.

A typical agent flow: fetch the manifest, take `manifest.match.url` and `sha256` from `doctor --json --manifest URL`
(or match `platform_tag` yourself), install, check that `test_status.per_kernel[<selected kernel>]` is `passed`, then
run `verify ... --cross-check` once on a known circuit.

## Measured results

Full DQI verification of `4ee7a17b4cbc` (889 qubits, 55.8M ops, 102,400 shots, empty salt) on an Apple M2 Pro (6
performance + 4 efficiency cores). Engine 0.2.0 wheel built by GitHub Actions, against the official verifier CLI on
kickmix 1.0.0 (PyPI) and on the kickmix-macos 0.3.0 fork, each in a clean Python 3.13 venv. Medians of alternating
rounds (3 against the fork, 2 against 1.0.0). "Cold" includes compiling the circuit; "warm" loads it from the program
cache. All 34 runs returned the identical result (PASS, 0 value and 0 phase errors, R 1.19126, score
414781342770.7355).

| baseline | workers / threads | baseline time | engine cold | engine warm | speed-up warm (cold) | peak memory, baseline / engine |
| --- | --- | --- | --- | --- | --- | --- |
| kickmix-macos 0.3.0 | 1 | 49.5 s | 31.7 s | 30.7 s | 1.61x (1.56x) | 2,091 / 339-352 MiB |
| kickmix-macos 0.3.0 | 10 | 8.9 s | 5.7 s | 4.7 s | 1.89x (1.56x) | 8,251 / 417-418 MiB |
| kickmix 1.0.0 | 1 | 189.7 s | 32.3 s | 31.5 s | 6.0x (5.9x) | 8,073 / 340-344 MiB |
| kickmix 1.0.0 | 6 | 36.7 s | 6.8 s | 5.9 s | 6.2x (5.4x) | 9,211 / 388-389 MiB |

Caveats: one machine, one circuit, one salt. The machine was not idle (load average 3.0-7.4 from other work, recorded
per run), but the ratios held steady across the alternating rounds. Peak memory is the whole process tree's peak
physical footprint. The engine's single process is set against the verifier's worker processes, which is how each one
is meant to run. These numbers are for engine 0.2.0. Version 0.3.0 changes only the one-off compile step, which now
also computes `reaction_depth` (1.13 s for this circuit, measured locally, against 1.17 s for 0.2.0), so the table
was not re-measured. Version 0.4.0 adds the x86_64 kernels; on Apple Silicon the simulation code is the same
(DQI cross-check 5.7 s cold at 10 threads, as for 0.3.0). invsqrt is less accelerated: about 1.8x the fork at 10 threads on the sample submission
(31.5 s against 56.2 s, a single smoke run), because its per-shot checks still run in NumPy.

**On Kaggle's CPU notebooks** (2026-10-28, private benchmark notebook, internet off, all three builds installed from
wheels): Intel Xeon @ 2.20 GHz (CPU model 79, Broadwell class), 4 vCPUs = 2 cores x 2 threads, AVX2 but no AVX-512,
31 GiB RAM. Same circuit and salt, engine 0.2.0 (baseline x86-64 build), medians of 2 alternating rounds, memory as the
process tree's summed PSS (shared pages counted once, so lower than the macOS footprint figures). All 16 runs
returned the identical result.

| workers / threads | kickmix 1.0.0 | kickmix-macos 0.3.0 | engine cold | engine warm | warm vs fork | warm vs 1.0.0 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 488.4 s / 4,579 MiB | 175.3 s / 1,907 MiB | 79.6 s / 316 MiB | 79.4 s / 315 MiB | 2.2x | 6.2x |
| 4 | 282.7 s / 4,552 MiB | 87.0 s / 1,888 MiB | 41.7 s / 341 MiB | 41.2 s / 330 MiB | 2.1x | 6.9x |

**Kernels on Kaggle (engine 0.4.0, same hardware, second session; every kernel forced in turn; all 20 runs returned
the identical result).** The AVX2 kernel, which the engine selects automatically on this CPU, is 1.42x faster than
the baseline kernel at 4 threads:

| workers / threads | kickmix-macos 0.3.0 | engine baseline (cold / warm) | engine avx2 (cold / warm) | avx2 warm vs fork |
| --- | --- | --- | --- | --- |
| 1 | 167.7 s | 78.2 / 76.6 s | 61.7 / 60.7 s | 2.8x |
| 4 | 72.1 s | 40.0 / 38.7 s | 28.5 / 27.2 s | 2.7x |

Against kickmix 1.0.0 (first session's figures) that is roughly 8-10x; the fork's own time moved between the two
sessions (87.0 s vs 72.1 s at 4 workers), so treat cross-session ratios as approximate. Kaggle's CPU notebooks have no
AVX-512, so that kernel's speed is not measured here.

Parity record (all on the circuits named above; the oracle is the unmodified official verifier on kickmix):

| check | result |
| --- | --- |
| DQI per-shot parity | 2 circuits, 17 verifier batches + 6 random ones, 5,888 shots: 0 mismatches in all qubits, classical bits, exact phase, error flags and op counts. Both negative controls (wrong seed, one CX changed to CZ) mismatch |
| invsqrt per-shot parity | 7 verifier chunks + random multi-simulator chunks: 0 mismatches; the wrong-seed control mismatches |
| DQI sampler | 2 circuits x 3 salts x 400 batches: inputs, phases and the generator state after every batch identical to the official `sample_shots` |
| DQI task parity | 2 circuits x 3 salts: every summary field and all 400 per-batch results identical |
| `reaction_depth` | equal to kickmix's `Circuit.reaction_depth()` on both DQI circuits, two invsqrt circuits and thousands of random circuits covering every op kind |
| x86_64 kernels (0.4.0) | baseline and avx2, each forced in turn, on an x86_64 macOS build under Rosetta 2: invsqrt per-shot parity (106,496 shots) and DQI per-shot parity on 4ee7a17b4cbc (2,048 shots, controls mismatch), 0 mismatches; DQI `--cross-check` at salt-check-1 equal on every field against the official verifier on the same x86_64 interpreter. In CI on GitHub's Linux runner (AMD EPYC 9V74, Zen 4, AVX-512), the test suite (13 tests, including random-circuit parity against stored kickmix references and the sampler restatement) passes with each of baseline, avx2 and avx512 forced, on Python 3.12 and 3.13 |
| FAIL paths | dirty ancilla, syndrome not restored, x not restored, dirty scratch, error-rate and phase-tolerance failures: same status and message as the official verifier |

## Reproduce the parity claims

The scripts in `tests/` need kickmix and the official verifiers (set `KICKMIX_ENGINE_VERIFIERS`). They write a JSON
report, refuse to overwrite an existing one, and exit non-zero on any mismatch. Set `KICKMIX_ENGINE_KERNEL` to
check a particular x86_64 kernel (the report records which kernel ran). Compare summaries from the same platform:
`mean_error_weight` is computed by the verifier itself in NumPy and can differ in the last bit between arm64 and x86_64
NumPy builds. Run them from the repository root with
`kickmix_engine` installed (or `PYTHONPATH=python` after `./build.sh`).

```sh
export KICKMIX_ENGINE_VERIFIERS=/path/to/verifiers
C=/path/to/vqcc-apache-public-circuits/circuits      # the public dataset

# DQI per-shot parity (verifier batches incl. logged failing shots from tests/data, random batches, negative controls).
# --scratch receives a 0.5 GB mutated copy for the CX->CZ control; delete it afterwards.
python tests/parity_dqi.py g1.json --kmb $C/dqi/4ee7a17b4cbc.kmb --kmb $C/dqi/b3354f60e459.kmb --scratch ./scratch
# invsqrt per-shot parity
python tests/parity.py g1_invsqrt.json --kmb /path/to/sample_submission/invsqrt.kmb
# DQI sampler against the official sample_shots, all batches, three salts
python tests/sampler_parity_dqi.py sampler.json --kmb $C/dqi/4ee7a17b4cbc.kmb --kmb $C/dqi/b3354f60e459.kmb
# DQI task parity: full verification against the official verify(), field by field and per batch
python tests/task_parity_dqi.py g2.json --kmb $C/dqi/4ee7a17b4cbc.kmb --kmb $C/dqi/b3354f60e459.kmb
# reaction_depth against kickmix: random circuits plus real ones
python tests/reaction_depth_parity.py rd.json --random 2000 --kmb $C/dqi/4ee7a17b4cbc.kmb --kmb $C/invsqrt/07314803a092.kmb
# FAIL paths: small mutated circuits built from the official reference circuits, through --cross-check
python tests/fail_paths_crosscheck.py fail.json --scratch ./scratch
# Any circuit, any salt
kickmix-engine verify dqi your.kmb --salt anything --cross-check
```

## Tests

```sh
pip install pytest
pytest            # from the repository root; needs only the installed wheel (no kickmix, no competition files)
```

`tests/test_engine.py` checks the simulator against a stored kickmix reference for a random circuit over every op
kind. It also covers the program cache (round trip and corruption detection), `reaction_depth` against stored kickmix
values, and the DQI sampler against a NumPy restatement of `sample_shots`. `tests/test_rng.py` compares the random
number chain with a stored reference and, when a C++ compiler is available (`$CXX` or `c++`), with
`tests/chachakeys.cc` built on the spot. Otherwise that test is skipped. The fixtures in `tests/data` were produced
with kickmix by `tests/make_fixtures.py` and `tests/reaction_depth_parity.py`. The bench scripts in `bench/` take
their inputs as arguments (`--kmb`, `--verifier`) or environment variables (`FORK_PY`, `ENGINE_PY`, `VERIFIER`, `KMB`,
`EVIDENCE`), as each script's header documents.

To test every kernel your CPU supports: `python tools/ci_kernel_tests.py --tests tests --wheel dist/X.whl --status
status.json`. It runs pytest once per kernel with `KICKMIX_ENGINE_KERNEL` set, and records kernels the CPU lacks as
skipped. The parity scripts below honour the same variable.

## How it works

- **Kernels** (`src/kernel.rs`): run-time choice of the baseline, AVX2 or AVX-512 copy of the simulator and sampler
  on x86_64 (see [Kernels](#kernels-x86_64)).
- **Simulator** (`src/sim.rs`, `src/engine.rs`): 256 shots per group of four 64-bit words per qubit or bit. It follows
  kickmix's semantics: condition stack, 128-bit fixed-point angles, global phase, HMR/R outcomes from kickmix's random
  number chain (std::seed_seq, mt19937_64, then ChaCha8 keys per 256-shot simulator) and exact per-kind operation
  counts. Batches run in parallel with Rayon, and the GIL is released while they run.
- **Compiler** (`src/program.rs`): streams a `.kmb` file into a compact u32 instruction stream, with compacted
  classical-bit ids and runs of plain CX. In the same pass it computes kickmix's `reaction_depth`, which is stored
  with the cached program.
- **DQI sampler** (`src/dqi_sample.rs`): an exact re-implementation of the verifier's `sample_shots` on NumPy's PCG64
  stream, fused with the simulation.
- **Task adapters** (`python/kickmix_engine/dqi.py`, `invsqrt.py`): load the official verifier module and use its
  seeds, problem spec, reference tables, Clopper-Pearson bound, repetition factor and cost model unchanged.

## Limitations

- Only the invsqrt and DQI tasks. Chemistry (`dfthcsos`) is not supported.
- Only `.kmb` input and `--batch-size 256`. DEBUG_PRINT ops are rejected. Qubit ids must be below 65,536.
- kickmix must be importable, because the official verifier modules import it.
- `circuit_sha256` is the SHA-256 of the file bytes. The official verifier hashes kickmix's re-serialisation of the
  circuit. The two are equal for canonical `.kmb` files (as written by kickmix), and `--cross-check` compares them on
  every run. A non-canonical file would get different seeds, and the cross-check would report it.
- On FAIL the engine reports the first failure in task order, as the official verifier does. That has been checked on
  the FAIL cases listed above, not on every possible failure.
- Wheels are built for macOS arm64 and Linux x86_64 only. Big-endian hosts are not supported (cache format).
- Full per-shot DQI and invsqrt parity runs have been done for the baseline and AVX2 kernels (locally, x86_64 under
  Rosetta 2). The AVX-512 kernel has passed the CI test suite on an AMD Zen 4 runner but not yet the full per-shot DQI
  parity run, and Kaggle's CPU notebooks (Broadwell-class Xeon) have no AVX-512. Check `test_status.per_kernel` in the
  release manifest before relying on a kernel.
- Speed and memory have been measured on one Apple Silicon machine and on Kaggle's CPU notebooks, for one DQI circuit.
  Other machines, circuits and salts may differ; `verify --timing` reports what you get.

## Licence

Apache-2.0 (see [LICENSE](LICENSE)). The simulation semantics, the random-number chain and the `reaction_depth`
definition follow kickmix (Copyright Google LLC, Apache-2.0). The task adapters call the official verifiers (Copyright
Google LLC, Apache-2.0), which are not included. The DQI sampler reproduces NumPy's random-number algorithms
(BSD-3-Clause), without NumPy source code. Details are in [NOTICE](NOTICE), and changes are listed in
[CHANGELOG.md](CHANGELOG.md); plans are in [ROADMAP.md](ROADMAP.md).
