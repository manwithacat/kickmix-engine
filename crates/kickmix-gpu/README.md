# kickmix-gpu: shot-parallel GPU simulation prototype (wgpu / Metal)

A feasibility prototype, not a release component. It runs a kickmix-engine compiled program on the GPU with one
thread per 128 (or 32) shots and compares every shot with the CPU engine, whose modules (`program.rs`, `rng.rs`,
`sim.rs`, `kernel.rs`, `dqi_sample.rs`, `engine.rs`) are compiled into the same binary by path, unchanged. The
experiment card with the design note, the parity evidence and the timings is
`experiments/2026-11-03-engine-gpu-shot-simulation.md` in the VQCC project; the raw reports are in that project's
`outputs/engine-gpu-2026-11-03/`.

Written by an AI model (Claude) under the owner's direction; semantics follow kickmix (Apache-2.0, Google LLC); see
the repository NOTICE.

## What it does

- `src/shader.wgsl`: a transcription of `Sim::run_impl` in WGSL. The engine's compiled u32 stream is executed
  verbatim (same opcodes, `OP_LEN`, CX-run headers, compact bit ids). Qubits live in a `[qubit][thread]` storage
  buffer and classical bits in `[bit][thread]`, so the uniform op index gives coalesced rows. Each thread holds its
  phase word, condition base and stack, and runs its own scalar ChaCha8 per 32-shot component (the keys come from the
  engine's `rng.rs` on the CPU), so HMR and R outcomes are the kickmix stream bit for bit. Op counts are per-thread
  `countOneBits(cond)` sums per kind, which add up to the CPU engine's per-batch `full x 256 + part`.
- The stream is cut on op boundaries into segments (default 8M words); each segment is one dispatch and the state
  persists in the buffers, which keeps dispatches under the macOS GPU watchdog and makes the state resumable.
- `--bundle K` re-encodes CX runs as bundles of up to K independent CXs (loads first, then stores) to test whether
  the per-op cost is a memory-latency chain. It is exact (checked) but slower on the M2 Pro; see the card.
- Not supported: Z_POW (128-bit angle accumulators; DQI programs cannot contain it), `.kmx` text, chemistry
  sequences, invsqrt's multi-batch RNG streams.

## Build and run

```sh
# standalone workspace root: the default kickmix-engine build and lockfile do not see wgpu
cargo build --release --manifest-path crates/kickmix-gpu/Cargo.toml -j2

# synthetic all-op program, both lane widths, segment boundaries, negative control
crates/kickmix-gpu/target/release/kickmix-gpu small --batches 5 --ops 30000 --seed 4242 [--bundle 8]

# one DQI verifier draw: dump the seeds and sampler parameters with the engine's Python adapter, then compare
KICKMIX_ENGINE_VERIFIERS=<verifiers dir> PYTHONPATH=python python crates/kickmix-gpu/tools/dump_dqi_tasks.py CIRCUIT.kmb tasks.json
crates/kickmix-gpu/target/release/kickmix-gpu dqi --tasks tasks.json [--batches N] [--repeat R] [--group G] \
    [--lanes 128|32] [--seg WORDS] [--wg 32] [--bundle K] [--compare-bits N] [--threads 4] [--cpu-timing] \
    [--control] [--sync-each] [--out report.json]
```

`dqi` samples every batch on the CPU with the engine's exact sampler, runs the CPU reference with the full state
(4 threads), runs the GPU, and reports per-shot mismatches in the final qubits, phase, value check and expected-phase
check, per-batch mismatches in all 18 op-kind counts, and (for the first `--compare-bits N` batches) all classical
bits. `--control` reruns batch 0 with one simulator-seed word changed and must report mismatches. Any wgpu
validation or device error fails the run (a dropped dispatch is never reported as a result).
