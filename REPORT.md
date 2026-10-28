# kickmix-engine M1 (Rust trial) report, 2026-10-27

Status: gates 1, 2 and 3 PASS (gate 3 decided by two idle reruns, 2026-10-27 23:37 and 2026-10-28 15:45; see 'Gate 3 idle reruns' below).
Code written by an AI model (Claude) under the owner's direction; semantics/RNG follow kickmix (Apache-2.0, Google LLC), see NOTICE.
Evidence: competition repo `outputs/engine-m1-2026-10-27/` (parity.json/log, single_core_*.json, scaling_*.json, kickmix_scaling_*.json).

## Design
- Rust crate (PyO3 abi3, Python >= 3.12, Rayon, numpy crate). `program.rs`: parses kickmix text (`str(circuit)`, circuit loaded with the
  kickmix Python API), compacts bit ids (invsqrt 8288 -> 366 touched), 10-byte fixed-width ops (u16 ids; 8 bytes was not reachable with
  u16 qubit+bit ids; revisit for M3), angle side table (128-bit V = half_turns*2^127, Clifford level per angle), CXRUN headers
  for maximal runs of bit-unconditioned CX (46,606 source ops -> 48,421 stream entries; headers add entries but remove dispatch).
- `sim.rs`: one 256-shot Sim = `W([u64;4])` words (LLVM vectorises to NEON; no explicit intrinsics), match-dispatched interpreter, condition
  stack, 128-bit angles per shot, HMR/R from vertical ChaCha8 (`rng.rs`), exact op counts (full-mask ops counted by a plain increment,
  others by popcount; Z_POW by Clifford level). x register in/out via 64x64 bit-matrix transposes.
- `rng.rs`: std::seed_seq::generate, mt19937_64, per-256-shot key derivation, ChaCha8 (4 double rounds, counter per 16 draws).
- `engine.rs`: unit = (chunk, 256-shot sim); chunk = one kickmix `Simulator(batch_size=256*S, seed)` running B batches on one RNG stream per sim
  (the verifier's chunk of 64 batches). Rayon over units, one `&Program`, GIL released (`py.detach`), outputs written to disjoint slices.
- NumPy API `kickmix_engine.run(prog, seeds, xs[chunk,batch,256*S], xreg, s, threads, counting, dump)` returns x_out, scratch_dirty,
  phase_units (verifier's 2^-64-turn ints), kind_counts, z_levels, toffoli_count; dump=True returns bit-sliced final state for parity.

## Gate 1: parity (PASS)
`tests/parity.py` against the kickmix fork (cxpop313 build), 163,840 shots compared: 7 verifier chunks (indices 0,1,2,3,5000,12238,12239 of the
200M-shot run: the verifier's own SHA-256 seed -> SeedSequence -> per-chunk seeds, `sample_shots` inputs, 64 batches per chunk, counting on),
plus 3 random-seed chunks with S=4 sims. Zero mismatches in: all 63 qubits x all shots, all classical bits (compacted ids mapped back; the
untouched bits are zero in kickmix), exact 128-bit phase V, verifier units, x read-back, scratch flags, per-kind op counts, Z_POW level counts,
Toffoli counts. Negative control (engine seeded with seed+1): 2,816,613 classical-bit mismatches and count mismatch, as required.
Note: with a wrong seed qubits and phase still agree (a correct circuit's outputs are outcome independent), so the RNG evidence is
the classical-bit and count agreement. `tests/test_rng.py`: keys identical to the spike's `chachakeys` (C++ std::seed_seq/mt19937_64).

## Gate 2: single core (PASS)
Per 256-shot batch, 256 batches, alternating runs, median of 10 (load average 9, includes my own runs): kickmix `do` (counting on) 0.2205 ms,
engine 0.1771 ms (whole `run` call per batch including x write/read and unit conversion; counting off 0.1731 ms) -> ratio 0.80 (limit 1.2).
Earlier 6-round run at load ~6: 0.226 vs 0.181 ms, ratio 0.80. Ratio is stable across load because both are single threaded.

## Gate 3: scaling (PENDING)
Shots/s median, 320 chunks x 64 batches, counting on: 1 thread 1.41e6; 10 threads 6.57e6; speedup median 1 / 1.84 / 3.40 / 4.95 / 4.66 for
1/2/4/8/10 threads (best-of-5: 1 / 1.97 / 3.67 / 5.40 / 6.35), load average 8.6-9.0. Three niced
processes (formal proofs, about 3 cores) and my own runs occupied the machine, and the Mac has 6 P + 4 E cores, so 7x at 10 threads is
borderline even idle (kickmix 10 processes reached 7.5x idle in the spike). Under the same load the engine out-scaled 10 kickmix processes
(kickmix_scaling_loaded1.json: 10 workers 6.1e6 vs 4.5e6 shots/s in the cleanest round). Rerun `bench/rerun_when_idle.sh TAG` when idle.

## Gate 3 idle reruns (PASS)
`bench/rerun_when_idle.sh`, 320 chunks x 64 batches, counting on, 5 rounds. Evidence: `scaling_idle-2026-10-27.json`, `scaling_idle-2026-10-28.json`
and the matching `single_core_*` and `kickmix_scaling_*` files in the evidence directory.
| run (load average) | 1 thread | 10 threads | speed-up at 10, median (best) | single core engine / kickmix |
| --- | --- | --- | --- | --- |
| 2026-10-27 23:37 (2.2) | 1.55e6 shots/s | 11.29e6 | 7.27x (7.27x) | 0.164 / 0.203 ms = 0.81 |
| 2026-10-28 15:45 (1.3-2.4) | 1.56e6 | 11.20e6 | 7.17x (7.24x) | 0.162 / 0.202 ms = 0.80 |
Against 10 kickmix processes in the same session (2026-10-28): 10.9e6 vs 8.6e6 shots/s (1.27x).
The 2026-10-27 rerun was not recorded here at the time; this section was added 2026-10-28.

## What was hardest for parity
Nothing was hard in the end; the spike's SEMANTICS.md was exact. The risk points were (1) reproducing std::seed_seq::generate and
mt19937_64 seeding bit-exactly, where an off-by-one in the index (k-1) mod n would silently give valid but different streams; the unit test
against the C++ reference isolated this, (2) the lane/bit mapping (u32 lane l, bit j = shot 32l+j = u64 word l/2), (3) counting semantics
(PUSH/POP always counted with full mask, conditional ops counted after the `if` refinement, CX runs counted per op).

## Verdict (design note rule)
Parity passed, single core is faster than kickmix, so Rust host is the choice unless the idle scaling rerun fails; a scaling shortfall
would point at E cores/memory bandwidth, not at the language, and would not trigger the C++ fallback (that is for parity failure).
Not done: explicit NEON/256-bit intrinsics (autovectorised 4x64 already beats kickmix), mmap'd program files, segments, DQI-size circuits.

# M3-DQI report, 2026-10-28 (branch `m3-dqi`, uncommitted)

Status: gates 1-3 PASS; gate 4 (speed) reported, 1.5x target at 10 threads missed (1.33x). Card:
`experiments/2026-10-28-engine-m3-dqi.md`; evidence: `outputs/engine-m3-dqi-2026-10-28/`. AI-written (Claude) under
the owner's direction; semantics from the kickmix fork source (`sim.h` apply loop, PopCounter, op-statistics groups,
`.kmb` reader).

## Changes
- IR (`program.rs`): M1's 10-byte ops (u16 ids) were replaced by a u32 word stream. Word 0 holds the opcode (base kind |
  `IF`=32) and qubit a; then only the words the op needs (b|c<<16, compact bit target, angle index, condition bit). CX
  runs are a header (24-bit count) plus one word per CX (a|b<<16). The `Builder` is shared by `from_text` (M1 path) and
  the new streaming `from_kmb` (one buffered reader per .kmb array; no kickmix Circuit, no `str(circuit)`; 1.2-1.7 s for 55.8M ops).
  Compact bit ids are u32, with `bit_compact` (original -> compact) for writing input bits. INFO_TAG is not in the stream:
  kickmix counts it with an all-ones mask regardless of the condition stack, so it is counted per run
  (`n_info_tags`). DQI: 67.1M words = 257 MiB (the .kmb is 505 MB), 1.55M CX runs, max stack depth 3.
- `sim.rs`: the generic path decodes the length from `OP_LEN`, takes `cond` from the last word for `*_IF`, and has one
  match on the base kind. CXRUN is a separate fast path. A 19th count slot holds INFO_TAG. `phase_flipped()` mirrors
  `read_phase_flipped_across_shots`.
- `engine.rs`: `run_batches` gives one kickmix `Simulator(batch_size=256, seed)` per batch: bit-sliced qubit and
  classical-bit inputs, value check on the engine (inputs restored, other qubits 0), phase words, per-batch counts.
  Rayon pools are shared by thread count, so concurrent calls share workers. M1's `run` is unchanged apart from 18 count slots.
- Python: `compile_kmb`, `run_batches`, word<->bool helpers; `kickmix_engine/dqi.py` (`DQITask`) reproduces `verify()`
  using the verifier module's own functions. Sampler processes are forked before compile, compile overlaps hashing and
  sampling, the prefetch window is bounded, and two groups are in flight.
- Tests and benches: `tests/parity_dqi.py` (gate 1), `tests/task_parity_dqi.py` (gate 2),
  `bench/dqi_verify_engine.py` (CLI), `bench/memwatch.py` (process-tree peak phys_footprint),
  `bench/dqi_gate34.sh`, `bench/summarize_gate34.py`, and `build.sh` (cargo build plus the macOS dynamic-lookup link
  flags; maturin is not needed).

## Results
| gate | outcome |
| --- | --- |
| 1 per-shot | 17 verifier chunks (incl. all 5 logged failing-shot chunks of these circuits) + 6 random chunks, 5,888 shots: 0 mismatches in qubits, all classical bits, exact phase, verifier flags, per-kind counts, Toffoli; both negative controls mismatch |
| 2 task | 2 circuits x 3 salts: every `verify()` field (except reaction_depth) and all 400 per-batch results identical |
| 3 memory | 382 MiB @1 thread (pred 329, +16%), 664 MiB @10 (pred 596, +11.5%); fork 2,090 / 8,249 MiB |
| 4 speed | fork/engine 51.2/29.2 s @1 (1.76x), 8.6/6.5 s @10 (1.33x); simulation alone 68.9 ms/batch single thread (kickmix `do` ~94 ms), 3.9 s for 400 batches @10 |

M1 invsqrt parity still passes on the new IR (139,264 shots, `m1_invsqrt_parity_regression.json`).

## What was hard and what remains
Nothing in the semantics was hard: the kickmix source answered every question (INFO_TAG counting, PUSH/POP counting with
a full mask, HMR/R drawing even when the condition is false, the condition read before the target). The `.kmb` integers
are little-endian on disk, despite the `_be` names in kickmix. Random uniform inputs gave no value errors on these
circuits (the decoders restore any syndrome), so the value-error path is covered by the logged salt-check-3 chunk.
The speed gap at 10 threads comes from outside the simulator: 1.2 s of serial compile/hash, plus the verifier's Python
`sample_shots` (~22 ms/batch) competing for cores. Next steps: a program cache keyed by content hash, an exact
vectorised sampler checked against `sample_shots` on all batches, and then the M5 question with simulation as the
remaining 3.9 s.

# M3 Addendum A report, 2026-10-28 (branch `m3-dqi`, uncommitted; package version 0.2.0)

Card: `experiments/2026-10-28-engine-m3-dqi.md` (Addendum A). Evidence: `outputs/engine-m3a-2026-10-28/` in the
competition repo (ENVIRONMENT.txt lists builds and hashes). Written by an AI model (Claude) under the owner's direction.
Scope change from the lead during the work: the formal gate 3/4 comparison will be run on GitHub-built wheels against the
kickmix-macos v0.3.0 release wheel. Local timings below are a labelled smoke test, not the gate.

## Changes
- **Program cache** (`python/kickmix_engine/cache.py`, `Program.save/load` in `program.rs`): files are keyed by the
  file SHA-256 plus the engine key (crate version, `CACHE_FORMAT`, first 16 hex characters of the extension's SHA-256).
  Writes go to a temporary file in the same directory and are then `os.replace`d. Each file carries a streaming 64-bit
  checksum, and the stream is bounds-validated on load (`Program::validate`). A failed load counts as a miss. A path
  hint (size + mtime_ns gives the sha) lets a warm run load the program while it hashes the file, and the computed hash
  must equal the hint. Timings for the DQI circuit: compile 1.1-1.2 s, save 0.15 s, load + validate 0.10 s, hash 0.26 s.
  Warm prepare is 0.27 s, against 1.2 s cold.
- **Exact DQI sampler** (`src/dqi_sample.rs`): NumPy's PCG64, including its 32-bit buffer, 53-bit doubles, Lemire
  bounded ints, `choice(p=)` by CDF search, and `choice(replace=False)` by Floyd then a shuffle, re-implemented draw for
  draw. Start states come from `np.random.PCG64(seed).state`. Output is bit-sliced rows directly: MBUC values are stored
  register-major and turned with a 64x64 transpose. Cost is 4.9 ms per batch on one thread, against about 7.5 ms for
  the official `sample_shots` alone.
  `engine::run_dqi` samples and simulates each batch in the same Rayon task, so no inputs cross into Python and no
  sampler processes are needed. `sample_dqi` exposes the sampler for proofs and self-checks.
  `DQITask.verify(sampler='engine'|'official', self_check=K)`. The M3 path is kept as `sampler='official'`.
- **invsqrt adapter** (`python/kickmix_engine/invsqrt.py`): the official flow (seeds, chunks, `sample_shots`,
  `reference_values`, cost model) with the engine's `run` as the simulator. Checks are vectorised per group of chunks,
  and the next group simulates while the current one is checked. FAIL messages use the official text.
- **CLI** `kickmix-engine verify {dqi,invsqrt}` plus `cache info|prune|clear`, with `--cross-check` (runs the official
  CLI and diffs every field) and `--timing`.
- **Release preparation**: README (quickstart, design, verify-our-claims), NOTICE (kickmix, the official verifiers and
  NumPy algorithms attributed; AI authorship stated), pyproject (0.2.0, console script, PEP 639 licence fields), and
  `.github/workflows/wheels.yml` (abi3 wheels for macOS arm64 and manylinux_2_28 x86_64, sdist, tests on py3.12/3.13
  from the installed wheel, GitHub release on `v*` tags). The workflow has not been run. Also kickmix-free tests
  (`tests/test_engine.py`: parity against a stored kickmix reference for a random 3,004-op circuit over every op kind,
  cache round trip and corruption, the sampler against a NumPy restatement of `sample_shots` for four parameter sets)
  and fixtures (`tests/data`, built by `tests/make_fixtures.py`).
- The bench scripts take interpreters as parameters: `bench/dqi_gate34.sh` uses `FORK_PY`, `ENGINE_PY`, `VERIFIER`, and
  runs the engine cold and warm. `bench/summarize_gate34.py` summarises each engine variant against the fork.

## Results
| check | outcome |
| --- | --- |
| sampler proof | 2 circuits x 3 salts x 400 batches = 2,400: syndromes, MBUC, phases, bit-sliced rows (all 49,140 E rows) and the PCG64 state after each batch identical to the official `sample_shots`; controls (state bit flip, shifted CDF) mismatch |
| gate 2 (engine sampler) | 6/6 PASS: every `verify()` field (except reaction_depth) and all 400 per-batch results identical; same scores as M3 |
| gate 1 / M1 regressions | DQI per-shot parity 5,888 shots, 0 mismatches, controls mismatch; invsqrt per-shot 106,496 shots, 0 mismatches |
| CLI cross-checks (installed wheel) | DQI 4ee7 salt-check-1 match (engine 5.6 s cold incl. self-check, official fork CLI 8.7 s at 10 workers); DQI FAIL path (`--num-shots 256`) identical error; invsqrt sample match (score 2076774.233715282; engine 31.5 s, official fork 56.2 s at 10 workers); invsqrt FAIL paths (`--max-x 2560 --precision-bits 30/26`) identical messages |
| smoke timing (not the gate; 2 rounds source build, 1 round local wheel, load 1.7-3.8) | 10 threads: fork 8.74-8.87 s, engine warm 4.50-4.57 s (1.91-1.97x), cold 5.49-5.55 s (1.59-1.60x). 1 thread: fork 50.9-51.1 s, engine warm 29.8-30.0 s (1.70x), cold 30.9-31.8 s (1.61-1.65x). Peak footprint at 10: engine 409-416 MiB against fork 8,244-8,247 MiB (M3: 664 MiB with sampler processes) |

## Not done / open
- Formal gates 3 and 4 on GitHub-built wheels (deferred by the lead). There is no new memory prediction yet; it should
  be written before that run. The no-sampler-process design should land near program 257 MiB + interpreter + 10 x
  per-thread state.
- `reaction_depth` is still not computed. Hash parity (file bytes against re-serialisation) is checked per run by
  `--cross-check` and was proven only for the circuits used.
- The invsqrt engine path keeps its checks in NumPy on the main thread (31.5 s at 10 threads, of which simulation is, by
  M1's measured 11M shots/s, about 18 s, an estimate). Moving the checks into Rust is the next speed step for invsqrt.
- The DQI and invsqrt adapters need kickmix importable, because the official verifier modules import it.

# Release clean-up report, 2026-10-28 (package version 0.3.0; branch `main`, uncommitted)

Card: `experiments/2026-10-28-engine-m3-dqi.md` ("Release clean-up"). Evidence: `outputs/engine-cleanup-2026-10-28/`
in the competition repo (ENVIRONMENT.txt lists the wheel, venvs, verifier and circuit hashes). Written by an AI model
(Claude) under the owner's direction. All checks below ran on a locally built 0.3.0 wheel installed into clean venvs.

## Changes
- **reaction_depth** (`src/program.rs`, `ReactionDepth`): a streaming restatement of kickmix's
  `compute_reaction_depth` (`circuit/circuit_util.cc`). Per qubit there is an x and a z depth and per bit a c depth,
  with a condition-depth stack. Every op kind follows kickmix's case, including the order of the bumps and the
  top-level resets of R, HMR, SWAP and BIT_STORE. It runs inside `Builder::op`, so it is computed once during
  compilation and stored in the program cache (cache format 2, a ninth scalar field). Plain CX, 88% of the DQI ops,
  takes an inlined fast path. DQI compile time is 1.13 s, against 1.17 s for 0.2.0 on the same machine. The adapters
  report `int(prog.reaction_depth)`, and `--cross-check` no longer skips the field. The first version had the operand
  order wrong. kickmix's `iter_ops` makes the *last* operand the target (2-qubit: control1, target; 3-qubit:
  control2, control1, target). Random-circuit parity caught it at once (224 of 300 mismatched). The minimal
  counterexample and the failed run are kept in the evidence `dev/` directory.
- **FAIL paths** (`tests/fail_paths_crosscheck.py`): the base circuits come from the official reference builders, the
  DQI decoder at m=30, n=4, degree=5 and the invsqrt Newton-Raphson circuit at n_p=10. A few ops are appended to each,
  and every case runs through `verify --cross-check`.
- **Paths and tests**: no script or document refers to the author's machine. Inputs come from arguments or environment
  variables. The scripts no longer edit `sys.path`. `tests/test_rng.py` became a pytest module with a stored
  reference, plus an optional C++ build of `tests/chachakeys.cc`. `tests/test_engine.py` gained `reaction_depth`
  tests on a stored fixture of 25 random circuits (from kickmix 1.0.0) and on hand-checked small cases. `pytest`
  from the repository root runs `tests/`, and so does CI.
- **Docs**: README rewritten for new users, CHANGELOG added, NOTICE updated (reaction-depth derivation, the test
  reference, fixtures, cross-check), version 0.3.0, repository URLs in the package metadata.

## Results (installed 0.3.0 wheel; oracle = unmodified official verifier on kickmix-macos 0.3.0 or kickmix 1.0.0)
| check | outcome |
| --- | --- |
| reaction_depth, random circuits | 2,000 circuits against the fork and 2,000 against kickmix 1.0.0 (3-400 ops, every kind, conditions, nested stacks, non-Clifford Z_POW; depths 1-89): all equal, compiled both from .kmb and from text |
| reaction_depth, real circuits | DQI 4ee7a17b4cbc and b3354f60e459: 9339 = 9339; invsqrt sample: 4005 = 4005; invsqrt 07314803a092: 795 = 795 (kickmix `Circuit.reaction_depth()`, both kickmix builds) |
| CLI `--cross-check`, all fields incl. reaction_depth | DQI 4ee7 salt-check-1 vs fork (score 414781480388.3637, 8-batch sampler self-check clean); DQI b3354 empty salt vs fork (422346240055.8793); DQI 4ee7 empty salt vs kickmix 1.0.0 (414781342770.7355); invsqrt sample vs fork and vs 1.0.0 (2076774.233715282); invsqrt 07314803a092 vs fork (102664.23957721348): 6/6 match, no differences |
| FAIL paths | 10/10 match (status and message): DQI reference PASS 0 errors; dirty ancilla on every shot (value_errors 102400/102400), on part (44358/102400), rarely (PASS with 3185 value errors, full summary equal); syndrome not restored; invsqrt x not restored (early and late); dirty scratch (`q259`); dirty-scratch naming order (`q259`); first failure in task order wins (`q141` before a later x failure) |
| M1 invsqrt per-shot parity | 106,496 shots, 0 mismatches; wrong-seed control mismatches |
| DQI gate-1 parity (short: 4ee7, empty-salt batches 0 and 399, salt-check-1 #17, logged failures #381/#78, 3 random) | 2,048 shots, 0 mismatches; both controls mismatch (wrong seed: 607,902 bits; CX to CZ: 54,731 qubit bits). The first attempt stopped on a missing `--scratch` directory (script bug, now created by the script; log kept) |
| pytest, fresh export of the tree with only the wheel installed | Python 3.12 (no kickmix): 11 passed; 3.13 (with kickmix): 11 passed; with no C++ compiler: 10 passed, 1 skipped with a message |
| wheel / sdist build | maturin `--release --locked`: builds cleanly (wheel .so sha256 7575e4d6...) |
Timings in these runs are not benchmarks (load average 6-17 from parallel work): cross-check engine/official 5.6/8.6 s
(DQI vs fork, 10 threads/workers), 4.6/35.7 s (DQI vs 1.0.0 at 6 workers), 35.0/55.5 s and 31.1/132.3 s (invsqrt
sample vs fork and 1.0.0).

## Still open
- The release table is for 0.2.0 wheels on one Mac; it was not re-measured for 0.3.0 (only compile changed).
- invsqrt checks still run in NumPy (about 1.6-1.8x the fork); no Linux or Kaggle numbers in the README.
- `cargo clippy` reports 22 style warnings in older code (indexing loops, manual slice copies); none are correctness
  issues and none were changed in this pass, to avoid touching the proven hot paths.
