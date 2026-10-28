# kickmix-engine

A bit-sliced CPU simulator for [kickmix](https://github.com/quantumlib/Qualtran/tree/main/kickmix) circuits that
reproduces kickmix shot for shot, and a `kickmix-engine verify` command that runs the official verifiers of the
Google Quantum AI *Verifiable Quantum Circuit Compilation* (VQCC) competition for the **invsqrt** and **DQI** tasks
with this engine as the simulator. It reports the same summary fields as the official verifier, and
`--cross-check` runs the official verifier as well and compares the two.

It is not affiliated with or endorsed by Google. The official verifier on kickmix is the reference. This engine is a
faster way to get the same answer, and it can check itself against that reference whenever you want.

**AI assistance.** The code, tests and this README were written by an AI model (Claude, by Anthropic) under the
direction of the repository owner, James Barlow, who reviews it and is responsible for it. See [NOTICE](NOTICE).

## Quickstart

Requirements: Python 3.12 or later, NumPy, and **kickmix importable in the same interpreter**. The engine does not
use kickmix to simulate, but the official verifier module imports it, and the engine takes seeds, input sampling
rules, statistics and scoring from that module. You also need the competition's official verifier files (the
`verifiers/` directory of the competition data).

```sh
pip install kickmix_engine-0.2.0-cp312-abi3-<platform>.whl     # a wheel from the GitHub Actions build or release
export KICKMIX_ENGINE_VERIFIERS=/path/to/competition/verifiers  # contains dqi/ and invsqrt/

kickmix-engine verify dqi dqi.kmb --threads 10                  # prints the official summary JSON
kickmix-engine verify dqi dqi.kmb --salt salt-check-1 --cross-check --official-workers 4
kickmix-engine verify invsqrt invsqrt.kmb --threads 10 --cross-check
kickmix-engine cache info                                       # compiled-program cache (see below)
```

Exit codes: 0 PASS; 1 FAIL (`{"status": "FAIL", "error": ...}`, as the official CLI); 3 when `--cross-check` finds a
difference. `--timing` prints engine timing and cache events to stderr. `python -m kickmix_engine` works as well.

Build from source (Rust 1.80 or later):

```sh
pip install maturin && maturin build --release -o dist && pip install dist/*.whl
# or, for an in-place development build on macOS:  ./build.sh  (then put python/ on PYTHONPATH)
pip install pytest && pytest tests/test_engine.py   # needs neither kickmix nor the competition files
```

## What it does

- **Simulator** (`src/sim.rs`, `src/engine.rs`): 256 shots per word group (four 64-bit words per qubit or bit),
  using kickmix's semantics: condition stack, 128-bit fixed-point angles, global phase, HMR/R outcomes from kickmix's
  random-number chain (std::seed_seq, then mt19937_64, then ChaCha8 keys per 256-shot simulator), and exact
  per-kind operation counts. Batches run in parallel with Rayon. The GIL is released while they run.
- **Compiler** (`src/program.rs`): reads `.kmb` files directly into a u32 word stream with compacted classical-bit
  ids and runs of plain CX ops.
- **Program cache** (`python/kickmix_engine/cache.py`): compiled programs are stored on disk, keyed by the SHA-256 of
  the circuit file plus the engine build (crate version, cache format, extension hash). Writes are atomic
  (temporary file then rename), so concurrent runs are safe. Files are checksummed and bounds-checked when loaded,
  and any file that fails is treated as a miss. Location: `--cache-dir`, `$KICKMIX_ENGINE_CACHE`, or
  `~/.cache/kickmix-engine`. A DQI program takes about 257 MiB. `kickmix-engine cache prune` removes entries from other
  builds.
- **DQI input sampler** (`src/dqi_sample.rs`): an exact re-implementation of the verifier's `sample_shots`. It follows
  NumPy's PCG64 stream draw for draw, including the 32-bit half-word buffer, the Lemire bounded integers, Floyd
  sampling with a shuffle for `choice(replace=False)`, and the CDF search for `choice(p=...)`. Generator start states
  come from NumPy itself. It runs fused with the simulation, so no inputs pass through Python.
  `--sampler official` uses the verifier's own `sample_shots` instead (the reference path). `--self-check K`
  re-draws K batches (or all of them, with -1) with the official function and fails on any difference.
- **Task adapters** (`dqi.py`, `invsqrt.py`): seeds, task layout, reference tables, Clopper-Pearson bound, repetition
  factor and cost model all come from the official verifier module, unchanged.

Differences from the official output: `reaction_depth` is `null`, because it is not computed. The circuit hash is the
SHA-256 of the file bytes, while the official verifier hashes kickmix's re-serialisation of the circuit. These are
equal for canonical `.kmb` files, and `--cross-check` compares them on every run. Only `.kmb` input and
`batch_size=256` are supported. On FAIL the engine reports the first failure in task order, using the official
message text.

## Verify our claims

These results come from the development Mac (Apple M2 Pro, 6 performance and 4 efficiency cores). The oracle is the
unmodified official verifier on a kickmix build. Each command writes a JSON report and refuses to overwrite one.
Circuits: DQI `4ee7a17b4cbc…` (889 qubits, 55.8M ops; from the public Kaggle notebook "The dataset is now public now.
Enjoy!" by Heliosli) and `b3354f60e459…` (897 qubits; from "Quantum Circuits | 22.44 Local GM: Our Chemistry" by Agniv
Banerjee). Both notebooks are released under Apache-2.0, and the circuits are collected in the *VQCC public circuits*
Kaggle dataset. The invsqrt check uses the competition's `sample_submission/invsqrt.kmb`.

```sh
export KICKMIX_ENGINE_VERIFIERS=/path/to/verifiers
# 0. Kickmix-free unit and parity tests (also run in CI)
pytest tests/test_engine.py
# 1. Gate 1, per-shot parity on DQI: every qubit, classical bit, exact phase, verifier flags and op counts against
#    kickmix on verifier chunks and random seeds, plus negative controls (wrong seed, one flipped op).
#    The chunks holding the logged failing shots of these circuits (from salt checks run on Kaggle) are in
#    tests/data/dqi_logged_failing_shots.jsonl and are included by default.
python tests/parity_dqi.py g1.json --kmb 4ee7a17b4cbc.kmb --kmb b3354f60e459.kmb --scratch /tmp/g1
#    invsqrt: per-shot parity (qubits, bits, exact 128-bit phase, 2^-64-turn units, op and Z_POW-level counts)
python tests/parity.py g1_invsqrt.json --kmb invsqrt.kmb
# 2. Sampler proof: the engine sampler against the official sample_shots on all 400 batches x 3 salts,
#    values plus the PCG64 state after each batch
python tests/sampler_parity_dqi.py sampler.json --kmb 4ee7a17b4cbc.kmb --kmb b3354f60e459.kmb
# 2. Gate 2, task parity: full 102,400-shot verification at '', salt-check-1, salt-check-2, compared with the
#    official verify() field by field and for each of the 400 batches
python tests/task_parity_dqi.py g2.json --kmb 4ee7a17b4cbc.kmb --kmb b3354f60e459.kmb
# Any circuit, any salt: the CLI's cross-check against the official verifier CLI
kickmix-engine verify dqi your.kmb --salt anything --cross-check
kickmix-engine verify invsqrt your.kmb --cross-check
```

Recorded outcomes (2026-10-28):

| check | result |
| --- | --- |
| DQI per-shot parity | 17 verifier chunks + 6 random chunks, 5,888 shots: 0 mismatches; both negative controls mismatch |
| invsqrt per-shot parity | 7 verifier chunks + random multi-sim chunks: 0 mismatches; wrong-seed control mismatches |
| sampler | 2,400 batches (2 circuits x 3 salts x 400): syndromes, MBUC values, phases, bit-sliced rows and end generator state all identical; controls mismatch |
| DQI task parity | 2 circuits x 3 salts: every summary field (except `reaction_depth`) and all 400 per-batch results identical; scores 414781342770.7355 / 414781480388.3637 / 414780974766.7814 and 422346240055.8793 / 422346307712.4221 / 424285910872.26764 |
| invsqrt (sample submission) | `--cross-check`: all fields identical, score 2076774.233715282 |

Speed and memory against the kickmix fork are measured separately (see the project report). The release numbers are
to be taken with the published wheels.

## Licence

Apache-2.0 (see [LICENSE](LICENSE)). The simulation semantics and random-number chain follow kickmix (Copyright
Google LLC, Apache-2.0). The DQI sampler follows NumPy's random-number algorithms (BSD-3-Clause). The test file
restates the official verifier's `sample_shots` (Apache-2.0, Google LLC). Details are in [NOTICE](NOTICE).
