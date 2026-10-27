# kickmix-engine M1 (Rust trial) report, 2026-10-27

Status: gates 1 and 2 PASS; gate 3 (scaling) NOT YET DECIDABLE: measured only under heavy background load, rerun when idle.
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

## What was hardest for parity
Nothing was hard in the end; the spike's SEMANTICS.md was exact. The risk points were (1) reproducing std::seed_seq::generate and
mt19937_64 seeding bit-exactly, where an off-by-one in the index (k-1) mod n would silently give valid but different streams; the unit test
against the C++ reference isolated this, (2) the lane/bit mapping (u32 lane l, bit j = shot 32l+j = u64 word l/2), (3) counting semantics
(PUSH/POP always counted with full mask, conditional ops counted after the `if` refinement, CX runs counted per op).

## Verdict (design note rule)
Parity passed, single core is faster than kickmix, so Rust host is the choice unless the idle scaling rerun fails; a scaling shortfall
would point at E cores/memory bandwidth, not at the language, and would not trigger the C++ fallback (that is for parity failure).
Not done: explicit NEON/256-bit intrinsics (autovectorised 4x64 already beats kickmix), mmap'd program files, segments, DQI-size circuits.
