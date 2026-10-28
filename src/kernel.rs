//! Runtime kernel selection. Written by an AI model (Claude) under the owner's direction.
//!
//! The hot loops (the bit-sliced simulator `Sim::run` and the exact DQI sampler) are written once as
//! `#[inline(always)]` bodies and instantiated several times: a baseline copy, and on x86_64 copies compiled with
//! `#[target_feature(enable = ...)]` for AVX2 (the x86-64-v3 feature set) and AVX-512 (x86-64-v4). Every copy runs
//! the same source, so the same per-shot results are required of each (and checked: see README, "Kernels").
//! The kernel is chosen once per process, at first use: the best one the CPU supports, or the one named by
//! `KICKMIX_ENGINE_KERNEL=baseline|avx2|avx512` (an error if this CPU or build lacks it). On aarch64 there is one
//! kernel, reported as "neon" (NEON is part of the aarch64 baseline; the compiler vectorises with it).

use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    Baseline,
    Avx2,
    Avx512,
    Neon,
}

impl Kernel {
    pub fn name(self) -> &'static str {
        match self {
            Kernel::Baseline => "baseline",
            Kernel::Avx2 => "avx2",
            Kernel::Avx512 => "avx512",
            Kernel::Neon => "neon",
        }
    }
    fn parse(s: &str) -> Option<Kernel> {
        match s.trim().to_ascii_lowercase().as_str() {
            "baseline" => Some(Kernel::Baseline),
            "avx2" => Some(Kernel::Avx2),
            "avx512" => Some(Kernel::Avx512),
            "neon" => Some(Kernel::Neon),
            _ => None,
        }
    }
    /// CPU features this kernel is compiled for (and requires at run time).
    pub fn features(self) -> &'static [&'static str] {
        match self {
            Kernel::Baseline => &[],
            Kernel::Avx2 => AVX2_FEATURES,
            Kernel::Avx512 => AVX512_FEATURES,
            Kernel::Neon => &["neon"],
        }
    }
}

/// x86-64-v3 (as used for the AVX2 kernel). Keep in sync with the `target_feature` attributes in sim.rs and
/// dqi_sample.rs (attribute strings must be literals, so they are spelled out there).
pub const AVX2_FEATURES: &[&str] = &["avx2", "bmi1", "bmi2", "fma", "lzcnt", "movbe"];
/// x86-64-v4 subset used for the AVX-512 kernel: v3 plus these.
pub const AVX512_FEATURES: &[&str] =
    &["avx2", "bmi1", "bmi2", "fma", "lzcnt", "movbe", "avx512f", "avx512bw", "avx512vl", "avx512dq"];

/// Kernels compiled into this build, best last.
pub fn compiled() -> &'static [Kernel] {
    #[cfg(target_arch = "x86_64")]
    {
        &[Kernel::Baseline, Kernel::Avx2, Kernel::Avx512]
    }
    #[cfg(target_arch = "aarch64")]
    {
        &[Kernel::Neon]
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        &[Kernel::Baseline]
    }
}

/// (feature, detected) for the features the kernels care about.
pub fn detected_features() -> Vec<(&'static str, bool)> {
    #[cfg(target_arch = "x86_64")]
    {
        vec![
            ("sse4.2", std::is_x86_feature_detected!("sse4.2")),
            ("popcnt", std::is_x86_feature_detected!("popcnt")),
            ("avx", std::is_x86_feature_detected!("avx")),
            ("avx2", std::is_x86_feature_detected!("avx2")),
            ("bmi1", std::is_x86_feature_detected!("bmi1")),
            ("bmi2", std::is_x86_feature_detected!("bmi2")),
            ("fma", std::is_x86_feature_detected!("fma")),
            ("lzcnt", std::is_x86_feature_detected!("lzcnt")),
            ("movbe", std::is_x86_feature_detected!("movbe")),
            ("avx512f", std::is_x86_feature_detected!("avx512f")),
            ("avx512bw", std::is_x86_feature_detected!("avx512bw")),
            ("avx512vl", std::is_x86_feature_detected!("avx512vl")),
            ("avx512dq", std::is_x86_feature_detected!("avx512dq")),
        ]
    }
    #[cfg(target_arch = "aarch64")]
    {
        vec![("neon", std::arch::is_aarch64_feature_detected!("neon"))]
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        vec![]
    }
}

fn has_all(k: Kernel) -> bool {
    match k {
        Kernel::Baseline => cfg!(not(target_arch = "aarch64")),
        Kernel::Neon => cfg!(target_arch = "aarch64"),
        _ => {
            if !compiled().contains(&k) {
                return false;
            }
            let det = detected_features();
            k.features().iter().all(|f| det.iter().any(|(n, d)| n == f && *d))
        }
    }
}

/// Kernels this build contains and this CPU can run, best last.
pub fn available() -> Vec<Kernel> {
    compiled().iter().copied().filter(|&k| has_all(k)).collect()
}

pub const ENV: &str = "KICKMIX_ENGINE_KERNEL";

fn choose() -> Result<(Kernel, &'static str), String> {
    let avail = available();
    match std::env::var(ENV) {
        Ok(v) if !v.trim().is_empty() && !v.trim().eq_ignore_ascii_case("auto") => {
            let names: Vec<&str> = avail.iter().map(|k| k.name()).collect();
            let k = Kernel::parse(&v).ok_or_else(|| {
                format!("{}={:?} is not a kernel name (use auto, baseline, avx2 or avx512; this machine can run: {})", ENV, v, names.join(", "))
            })?;
            if !avail.contains(&k) {
                let why = if compiled().contains(&k) { "this CPU lacks the features it needs" } else { "this build does not contain it" };
                return Err(format!(
                    "{}={} requested, but {} (needs: {}; this machine can run: {})",
                    ENV,
                    k.name(),
                    why,
                    if k.features().is_empty() { "-".to_string() } else { k.features().join(",") },
                    names.join(", ")
                ));
            }
            Ok((k, "env"))
        }
        _ => avail.last().copied().map(|k| (k, "auto")).ok_or_else(|| "no kernel available".to_string()),
    }
}

static SELECTED: OnceLock<Result<(Kernel, &'static str), String>> = OnceLock::new();

/// The kernel for this process (chosen once), with how it was chosen ("auto" or "env"), or the override error.
pub fn selected() -> Result<(Kernel, &'static str), String> {
    SELECTED.get_or_init(choose).clone()
}

/// The kernel to run. Entry points call `selected()?` first, so the fallback is never used by them.
#[inline]
pub fn active() -> Kernel {
    match selected() {
        Ok((k, _)) => k,
        Err(_) => compiled()[0],
    }
}
