//! kmx-analysis: static analysis primitives over kickmix `.kmb` circuits (engineering card E1, 2026-11-05).
//! Written by an AI model (Claude) under the owner's direction. Pure Rust, no PyO3; the Python binding is the `py`
//! member crate (module `kickmix_analysis`). Separate from the verify wheel (`kickmix_engine`): nothing here is used
//! by verification, and the engine's `src/` is not touched.
//!
//! The Python implementations in the competition repository (`scripts/qhdl_lifetime_allocator.py`,
//! `scripts/qhdl_peak.py`) are the reference; parity is byte-identical on the pinned circuits.
pub mod kmb;
pub mod liveness;
