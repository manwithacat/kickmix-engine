//! CPython's Mersenne Twister (`random.Random(int)` seeding and `getrandbits(k)`), reproduced bit for bit so that
//! the value simulator draws exactly the HMR outcomes the Python reference draws (card E3).
//! Seeding: `init_by_array` over the 32-bit little-endian words of |seed| (one zero word for seed 0), as
//! `random_seed` in CPython's `_randommodule.c`. `getrandbits(k)`, k > 32: ceil(k/32) outputs, least significant word
//! first, the last one shifted right by 32 - (k mod 32) when k is not a multiple of 32.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER: u32 = 0x8000_0000;
const LOWER: u32 = 0x7fff_ffff;

#[derive(Clone)]
pub struct Mt19937 {
    mt: [u32; N],
    mti: usize,
}

impl Mt19937 {
    fn init_genrand(s: u32) -> Self {
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1_812_433_253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        Mt19937 { mt, mti: N }
    }

    pub fn init_by_array(key: &[u32]) -> Self {
        let mut g = Self::init_genrand(19_650_218);
        let mt = &mut g.mt;
        let (mut i, mut j) = (1usize, 0usize);
        let kl = key.len().max(1);
        let mut k = N.max(kl);
        while k > 0 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(*key.get(j).unwrap_or(&0))
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            if j >= kl {
                j = 0;
            }
            k -= 1;
        }
        k = N - 1;
        while k > 0 {
            mt[i] = (mt[i] ^ (mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                mt[0] = mt[N - 1];
                i = 1;
            }
            k -= 1;
        }
        mt[0] = 0x8000_0000;
        g
    }

    /// `random.Random(seed)` for a Python int seed (sign ignored, as CPython takes |seed|).
    pub fn from_seed(seed: i64) -> Self {
        let mut n = seed.unsigned_abs();
        let mut key = Vec::new();
        if n == 0 {
            key.push(0);
        }
        while n > 0 {
            key.push(n as u32);
            n >>= 32;
        }
        Self::init_by_array(&key)
    }

    /// Seed from explicit 32-bit key words (for seeds wider than 64 bits).
    pub fn from_key(key: &[u32]) -> Self {
        if key.is_empty() {
            Self::init_by_array(&[0])
        } else {
            Self::init_by_array(key)
        }
    }

    #[inline]
    pub fn genrand_u32(&mut self) -> u32 {
        if self.mti >= N {
            let mt = &mut self.mt;
            let mag = |y: u32| if y & 1 != 0 { MATRIX_A } else { 0 };
            for kk in 0..N - M {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag(y);
            }
            for kk in N - M..N - 1 {
                let y = (mt[kk] & UPPER) | (mt[kk + 1] & LOWER);
                mt[kk] = mt[kk + M - N] ^ (y >> 1) ^ mag(y);
            }
            let y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
            mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag(y);
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `getrandbits(k)` written into little-endian u64 words (`out` must hold ceil(k/64) words; it is overwritten).
    pub fn getrandbits_into(&mut self, k: usize, out: &mut [u64]) {
        for x in out.iter_mut() {
            *x = 0;
        }
        if k == 0 {
            return;
        }
        if k <= 32 {
            out[0] = (self.genrand_u32() >> (32 - k)) as u64;
            return;
        }
        let words = (k - 1) / 32 + 1;
        let mut left = k as i64;
        for i in 0..words {
            let mut r = self.genrand_u32();
            if left < 32 {
                r >>= 32 - left;
            }
            out[i / 2] |= (r as u64) << (32 * (i % 2));
            left -= 32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_first_outputs() {
        // CPython: random.Random(1).getrandbits(32) == 577090037, random.Random(0).getrandbits(32) == 3626764237
        assert_eq!(Mt19937::from_seed(1).genrand_u32(), 577_090_037);
        assert_eq!(Mt19937::from_seed(0).genrand_u32(), 3_626_764_237);
    }
}
