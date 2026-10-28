//! Exact DQI input sampler: reproduces the official verifier's `sample_shots(spec, tables, rng, 256, weight_probs)`
//! draw for draw, where `rng = numpy.random.default_rng(seed)` (a NumPy `Generator` over `PCG64`).
//!
//! Written by an AI model (Claude) under the owner's direction. The generator algorithms follow NumPy's
//! (BSD-3-Clause) `PCG64` (XSL-RR 128/64 with a 32-bit half-word buffer), `random_standard_uniform`
//! (53-bit doubles), the buffered Lemire bounded-integer method used by `Generator.integers` and by
//! `Generator.choice(..., replace=False)` (Floyd's algorithm followed by a Fisher-Yates shuffle), and
//! `Generator.choice(..., p=...)` (CDF + `searchsorted(side='right')`). The PCG64 start state is taken from
//! NumPy itself (`PCG64(seed).state`), so `SeedSequence` is not re-implemented. Equality with the official
//! `sample_shots` is proven by `tests/sampler_parity_dqi.py` (values and the final generator state).

pub const PCG_MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Pcg64 {
    pub state: u128,
    pub inc: u128,
    pub has_u32: bool,
    pub u32_buf: u32,
}

impl Pcg64 {
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(PCG_MULT).wrapping_add(self.inc);
        let hi = (self.state >> 64) as u64;
        let lo = self.state as u64;
        (hi ^ lo).rotate_right((hi >> 58) as u32)
    }
    #[inline(always)]
    pub fn next_u32(&mut self) -> u32 {
        if self.has_u32 {
            self.has_u32 = false;
            return self.u32_buf;
        }
        let n = self.next_u64();
        self.has_u32 = true;
        self.u32_buf = (n >> 32) as u32;
        n as u32
    }
    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
    /// Uniform in [0, rng] (inclusive), rng < 2^32: NumPy's `random_bounded_uint64` with `use_masked=false`.
    #[inline(always)]
    pub fn bounded(&mut self, rng: u64) -> u64 {
        debug_assert!(rng <= 0xFFFF_FFFF);
        if rng == 0 {
            return 0;
        }
        if rng == 0xFFFF_FFFF {
            return self.next_u32() as u64;
        }
        let rng = rng as u32;
        let excl = rng + 1;
        let mut m = (self.next_u32() as u64) * (excl as u64);
        let mut left = m as u32;
        if left < excl {
            let threshold = (u32::MAX - rng) % excl;
            while left < threshold {
                m = (self.next_u32() as u64) * (excl as u64);
                left = m as u32;
            }
        }
        m >> 32
    }
}

/// Fixed parameters of the sampler (from the verifier: spec, `FieldTables`, `optimal_dqi_weights(...)**2`).
pub struct DqiSampler {
    pub m: usize,
    pub n: usize,
    pub degree: usize,
    /// `cdf = weight_probs.cumsum(); cdf /= cdf[-1]` exactly as NumPy computes it.
    pub cdf: Vec<f64>,
    pub log: Vec<u32>,
    pub antilog: Vec<u32>,
    /// Which MBUC bit rows (row = j * degree + k) to emit, in this order.
    pub e_rows: Vec<u32>,
}

/// One batch of samples.
pub struct DqiBatch {
    /// [n * degree][4] bit-sliced syndrome qubit rows (row = i * degree + k).
    pub qin: Vec<u64>,
    /// [len(e_rows)][4] bit-sliced MBUC bit rows.
    pub bin: Vec<u64>,
    /// [4] expected phase flips.
    pub phase: [u64; 4],
    /// Values, only when requested: syndromes [256][n], mbuc [256][m].
    pub syn_vals: Vec<u16>,
    pub mbuc_vals: Vec<u16>,
}

impl DqiSampler {
    pub fn check(&self) -> Result<(), String> {
        let order = (1usize << self.degree) - 1;
        if self.degree == 0 || self.degree > 16 {
            return Err("degree must be in 1..=16".into());
        }
        if self.log.len() != order + 1 || self.antilog.len() != order {
            return Err("field tables have the wrong size".into());
        }
        if self.cdf.is_empty() || self.n == 0 || self.m == 0 || self.m > order {
            return Err("bad m, n or weights".into());
        }
        if self.cdf.len() - 1 > self.m {
            return Err("more error weights than positions".into());
        }
        if self.e_rows.iter().any(|&r| r as usize >= self.m * self.degree) {
            return Err("e_rows out of range".into());
        }
        Ok(())
    }

    /// `sample_shots(spec, tables, default_rng(seed), 256, weight_probs)`; `rng` holds the generator state and is
    /// advanced exactly as NumPy advances it.
    pub fn sample(&self, rng: &mut Pcg64, keep_values: bool) -> DqiBatch {
        const B: usize = 256;
        let (m, n, deg) = (self.m, self.n, self.degree);
        let order = ((1u64 << deg) - 1) as u64;
        let max_value = order; // (1 << degree) - 1
        // weights = rng.choice(len(p), size=256, p=p)
        let mut weights = [0usize; B];
        for w in weights.iter_mut() {
            let u = rng.next_f64();
            *w = self.cdf.partition_point(|&c| c <= u);
        }
        // mbuc = rng.integers(0, max_value, size=(256, m), endpoint=True), drawn in (shot, register) order and
        // stored register-major: mbuc_t[j * 256 + s].
        let mut mbuc_t = vec![0u16; B * m];
        for s in 0..B {
            for j in 0..m {
                mbuc_t[j * B + s] = rng.bounded(max_value) as u16;
            }
        }
        // Per shot with weight k > 0: positions (Floyd + shuffle), values, syndromes, phase.
        let mut syn = vec![0u16; B * n];
        let mut phase = [0u64; 4];
        let mut pos: Vec<u64> = Vec::with_capacity(64);
        for s in 0..B {
            let k = weights[s];
            if k == 0 {
                continue;
            }
            let pop = m as u64;
            pos.clear();
            for j in (pop - k as u64)..pop {
                let val = rng.bounded(j);
                if pos.contains(&val) {
                    pos.push(j);
                } else {
                    pos.push(val);
                }
            }
            for i in (1..k).rev() {
                let j = rng.bounded(i as u64) as usize;
                pos.swap(i, j);
            }
            let mut par = 0u32;
            let row = &mut syn[s * n..(s + 1) * n];
            for &p in pos.iter() {
                let val = 1 + rng.bounded(max_value - 1);
                par ^= (val as u32 & mbuc_t[p as usize * B + s] as u32).count_ones() & 1;
                // exps[i] = (log[val] + (i + 1) * pos) % order, stepped incrementally (same values).
                let step = p % order;
                let mut e = (self.log[val as usize] as u64 + step) % order;
                for r in row.iter_mut() {
                    *r ^= self.antilog[e as usize] as u16;
                    e += step;
                    if e >= order {
                        e -= order;
                    }
                }
            }
            if par == 1 {
                phase[s / 64] |= 1u64 << (s % 64);
            }
        }
        // Bit-slice: row (reg, bit k), bit s of word s/64 = shot s.
        let mut qin = vec![0u64; n * deg * 4];
        for s in 0..B {
            let (w, b) = (s / 64, s % 64);
            for i in 0..n {
                let mut v = syn[s * n + i];
                while v != 0 {
                    let k = v.trailing_zeros() as usize;
                    qin[(i * deg + k) * 4 + w] |= 1u64 << b;
                    v &= v - 1;
                }
            }
        }
        // MBUC rows: per register and 64-shot word, a 64x64 bit transpose (rows = shots -> rows = value bits).
        let mut full = vec![0u64; m * deg * 4];
        let mut t = [0u64; 64];
        for j in 0..m {
            for w in 0..4 {
                let col = &mbuc_t[j * B + w * 64..j * B + w * 64 + 64];
                for (x, &v) in t.iter_mut().zip(col) {
                    *x = v as u64;
                }
                crate::sim::transpose64(&mut t);
                for k in 0..deg {
                    full[(j * deg + k) * 4 + w] = t[k];
                }
            }
        }
        let mut bin = vec![0u64; self.e_rows.len() * 4];
        for (r, &src) in self.e_rows.iter().enumerate() {
            bin[r * 4..r * 4 + 4].copy_from_slice(&full[src as usize * 4..src as usize * 4 + 4]);
        }
        DqiBatch {
            qin,
            bin,
            phase,
            syn_vals: if keep_values { syn } else { Vec::new() },
            mbuc_vals: if keep_values {
                let mut v = vec![0u16; B * m];
                for j in 0..m {
                    for s in 0..B {
                        v[s * m + j] = mbuc_t[j * B + s];
                    }
                }
                v
            } else {
                Vec::new()
            },
        }
    }
}
