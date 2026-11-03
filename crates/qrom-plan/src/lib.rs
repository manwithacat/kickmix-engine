//! qrom-plan: bit-packed GF(2) cost model for the VQCC chemistry ANGLE_k five-word QROM loaders.
//!
//! Ports the Python kernel-relation planner used for the Hedge Wizardry HW-12/HW-13 loaders (kernel relations,
//! template spaces, pairing, exchange repair, Lera-style terms) and adds an incremental word-flip evaluator
//! (rank-one updates of every relation-supply sub-matrix). Written by an AI model (Claude) under the owner's
//! direction; see the crate README and the experiment card
//! `experiments/2026-11-03-chem-fast-qrom-cost-model.md` in the VQCC project.

pub mod bits;
pub mod flips;
pub mod gf2;
pub mod loader;
pub mod npy;
pub mod plan;
pub mod problem;
pub mod rng;
pub mod search;
pub mod textio;

use std::path::Path;

/// Load a word table from a 2-D integer `.npy` file.
pub fn load_words(path: &Path) -> Result<loader::Words, String> {
    let arr = npy::read_npy_i64(path)?;
    loader::Words::from_i64(arr.shape.0, arr.shape.1, &arr.data)
}
