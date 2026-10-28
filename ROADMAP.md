# Roadmap

Each step is gated the same way as the work so far: exact parity with the unmodified official verifier first, then
speed and memory measured on named hardware, with failed runs kept. Order can change as measurements come in.

## Next

1. **Kernel results on real x86.** Per-kernel timing (baseline vs AVX2) on Kaggle's CPU notebooks with 0.4.0, and a
   full per-shot DQI parity run of the AVX-512 kernel on AVX-512 hardware. Outcome decides whether the x86 kernels need
   hand-written SIMD beyond what the compiler generates.
2. **invsqrt contract in Rust.** invsqrt is only about 1.8x the fork because its per-shot checks run in NumPy on the
   main thread. Move sampling, the reference table and the checks into the engine (as already done for DQI), with
   exhaustive equivalence to the verifier's Python.
3. **Chemistry (`dfthcsos`).** Done in 0.5.0 (`verify dfthcsos`). Next: the same comparison on Kaggle's CPU notebooks.
4. **Install paths.** A Kaggle dataset mirroring each release (offline notebooks), and a PyPI package once the name
   and a trusted-publisher setup are in place.

## Later

- A GPU backend, only if it beats a well-vectorised CPU kernel by a wide margin (simulation is about 3.9 s of a 4.7 s
  DQI run on 10 Apple cores; a Metal spike earlier matched 10 CPU cores, no more).
- Region-level tools on tagged circuits (stage tree, per-stage cost, live-qubit profile), as a basis for
  design-level search.
- Offer kickmix-compatible pieces upstream to Qualtran if the maintainers want them, within their policy on
  AI-assisted contributions.

## Not planned

- Replacing the official verifiers: they remain the reference, and `--cross-check` exists so anyone can confirm the
  engine against them on their own machine.
