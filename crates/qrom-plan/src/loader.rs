//! From the word table to a loader's mixed ANF tensor (bit-packed). Reproduces the Python planner's
//! `coded_columns`, `pad_affine`, `loader_delta` and `coefficients` (and through it rank_parity.coefficients).

use crate::bits::B525;

pub const BROT: usize = 8; // word bits; bit 7 = bit 6 (signed gauge); 7 bits enter the ANF
pub const LOWBITS: usize = 4;
pub const LOW: usize = 1 << LOWBITS; // 16 addresses per block
pub const J: usize = 5; // words per loader
pub const OUT: usize = J * (BROT - 1); // 35 output bits
pub const SLOTS: usize = LOW - 1; // 15 nonlinear-or-linear low monomials 1..15
pub const N525: usize = SLOTS * OUT;

/// The word table: `rows` addresses x `cols` word columns of 8-bit signed-gauge words.
#[derive(Clone, Debug)]
pub struct Words {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<u8>,
}

impl Words {
    pub fn from_i64(rows: usize, cols: usize, data: &[i64]) -> Result<Self, String> {
        if data.len() != rows * cols {
            return Err("shape mismatch".into());
        }
        let mut out = Vec::with_capacity(data.len());
        for &v in data {
            if !(0..256).contains(&v) {
                return Err(format!("word value {v} outside 0..255"));
            }
            out.push(v as u8);
        }
        Ok(Words { rows, cols, data: out })
    }

    #[inline]
    pub fn get(&self, a: usize, j: usize) -> u8 {
        self.data[a * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, a: usize, j: usize, v: u8) {
        self.data[a * self.cols + j] = v;
    }

    pub fn loaders(&self) -> usize {
        (self.cols + J - 1) / J
    }

    /// Flip bit `bit` (0..=6) of word (a, j); bit 6 flips bit 7 too (signed gauge).
    pub fn flip(&mut self, a: usize, j: usize, bit: u8) {
        let mut v = self.get(a, j);
        v ^= 1 << bit;
        if bit == 6 {
            v ^= 1 << 7;
        }
        self.set(a, j, v);
    }

    /// Is bit 0 of column j affine on the 16-address block starting at b0 (a full block)?
    pub fn block_affine(&self, j: usize, b0: usize) -> bool {
        let mut pattern = 0u16;
        for a in 0..LOW {
            if self.get(b0 + a, j) & 1 == 1 {
                pattern |= 1 << a;
            }
        }
        affine_patterns().contains(&pattern)
    }

    /// Columns whose bit 0 is affine on every full 16-address block (the partial last block is always fine).
    pub fn coded_columns(&self) -> Vec<bool> {
        let full = self.rows - self.rows % LOW;
        (0..self.cols)
            .map(|j| (0..full).step_by(LOW).all(|b0| self.block_affine(j, b0)))
            .collect()
    }
}

/// The 32 affine functions on 4 address bits as 16-bit truth patterns, in the Python order (c in 0..2, m in 0..16).
pub fn affine_patterns() -> &'static [u16; 32] {
    static PATTERNS: std::sync::OnceLock<[u16; 32]> = std::sync::OnceLock::new();
    PATTERNS.get_or_init(|| {
        let mut out = [0u16; 32];
        for c in 0..2u32 {
            for m in 0..16u32 {
                let mut p = 0u16;
                for a in 0..16u32 {
                    if (c ^ ((m & a).count_ones() & 1)) == 1 {
                        p |= 1 << a;
                    }
                }
                out[(c * 16 + m) as usize] = p;
            }
        }
        out
    })
}

/// The table delta_k = B_k XOR B_(k-1), padded to whole 16-address blocks, with coded bit-0 affine padding.
#[derive(Clone, Debug)]
pub struct Delta {
    pub k: usize,
    pub rows: usize,        // original address count
    pub padded_rows: usize, // multiple of 16
    pub coded: Vec<usize>,  // word indices 0..5 with the coded (affine bit 0) structure in this loader
    pub data: Vec<[u8; J]>, // padded_rows entries
}

impl Delta {
    pub fn build(words: &Words, k: usize) -> Result<Self, String> {
        Self::build_with_coded(words, k, &words.coded_columns())
    }

    pub fn build_with_coded(words: &Words, k: usize, coded_cols: &[bool]) -> Result<Self, String> {
        let a_n = words.rows;
        let padded_rows = (a_n + LOW - 1) / LOW * LOW;
        let mut data = vec![[0u8; J]; padded_rows];
        for a in 0..a_n {
            for i in 0..J {
                let c = J * k + i;
                let bk = if c < words.cols { words.get(a, c) } else { 0 };
                let prev = if k > 0 { J * (k - 1) + i } else { usize::MAX };
                let bp = if k > 0 && prev < words.cols { words.get(a, prev) } else { 0 };
                data[a][i] = if k == 0 { bk } else { bk ^ bp };
            }
        }
        let is_coded = |c: usize| c < coded_cols.len() && coded_cols[c];
        let coded: Vec<usize> = (0..J)
            .filter(|&i| is_coded(J * k + i) && (k == 0 || is_coded(J * (k - 1) + i)))
            .collect();
        // pad_affine: extend bit 0 of the coded words on the partial last block by the first fitting affine pattern
        let b0 = a_n - a_n % LOW;
        let n = a_n - b0;
        if !coded.is_empty() && n > 0 && n < LOW {
            for &j in &coded {
                let mut fit = None;
                for &p in affine_patterns().iter() {
                    if (0..n).all(|i| ((p >> i) & 1) as u8 == data[b0 + i][j] & 1) {
                        fit = Some(p);
                        break;
                    }
                }
                let p = fit.ok_or_else(|| format!("bit 0 of word {j} is not affine on the last block"))?;
                for i in 0..LOW {
                    data[b0 + i][j] = (data[b0 + i][j] & !1) | ((p >> i) & 1) as u8;
                }
            }
        }
        Ok(Delta { k, rows: a_n, padded_rows, coded, data })
    }

    pub fn high_count(&self) -> usize {
        self.padded_rows / LOW
    }

    /// Truth table t[h][l] = 35-bit output word at address h * 16 + l (bits 0..7 of each of the 5 words).
    pub fn truth(&self) -> Vec<[u64; LOW]> {
        let high = self.high_count();
        let mut t = vec![[0u64; LOW]; high];
        for h in 0..high {
            for l in 0..LOW {
                let w = &self.data[h * LOW + l];
                let mut o = 0u64;
                for j in 0..J {
                    assert_eq!((w[j] >> 7) & 1, (w[j] >> 6) & 1, "signed gauge required");
                    o |= ((w[j] & 0x7f) as u64) << (j * (BROT - 1));
                }
                t[h][l] = o;
            }
        }
        t
    }
}

/// Mixed ANF tensor ac[h][l] (35-bit output words) from the truth table: Mobius (subset-sum) transform over the high
/// and the low address bits.
pub fn mobius(mut t: Vec<[u64; LOW]>) -> Vec<[u64; LOW]> {
    let high = t.len();
    let hbits = usize::BITS as usize - (high.max(2) - 1).leading_zeros() as usize;
    for bit in 0..hbits {
        for h in 0..high {
            if h & (1 << bit) != 0 {
                let src = t[h ^ (1 << bit)];
                for l in 0..LOW {
                    t[h][l] ^= src[l];
                }
            }
        }
    }
    for bit in 0..LOWBITS {
        for row in t.iter_mut() {
            for l in 0..LOW {
                if l & (1 << bit) != 0 {
                    row[l] ^= row[l ^ (1 << bit)];
                }
            }
        }
    }
    t
}

/// Rows F_h (h = 1..H-1) in GF(2)^525: bit (l - 1) * 35 + o is ac[h][l] bit o for l = 1..15.
pub fn rows_from_anf(ac: &[[u64; LOW]]) -> Vec<B525> {
    ac[1..]
        .iter()
        .map(|slice| {
            let mut r = B525::ZERO;
            for l in 1..LOW {
                let word = slice[l] & ((1u64 << OUT) - 1);
                let base = (l - 1) * OUT;
                // place 35 bits at offset base
                let wi = base >> 6;
                let sh = base & 63;
                r.0[wi] |= word << sh;
                if sh + OUT > 64 {
                    r.0[wi + 1] |= word >> (64 - sh);
                }
            }
            r
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_pattern_count() {
        let p = affine_patterns();
        let mut s: Vec<u16> = p.to_vec();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 32);
        assert_eq!(p[0], 0); // c = 0, m = 0: the zero function
        assert_eq!(p[16], 0xffff); // c = 1, m = 0: the constant 1
    }

    #[test]
    fn mobius_is_involutive_and_linear() {
        let mut t = vec![[0u64; LOW]; 6];
        t[3][5] = 0b1011;
        t[0][0] = 1;
        let a = mobius(t.clone());
        let b = mobius(a.clone());
        assert_eq!(b, t);
        // flipping one truth entry flips ac on the superset cone
        let mut t2 = t.clone();
        t2[1][4] ^= 1 << 7;
        let a2 = mobius(t2);
        for h in 0..6 {
            for l in 0..LOW {
                let expect = if (h & 1) == 1 && (l & 4) == 4 { 1 << 7 } else { 0 };
                assert_eq!(a[h][l] ^ a2[h][l], expect, "h {h} l {l}");
            }
        }
    }

    #[test]
    fn rows_pack_bits() {
        let mut ac = vec![[0u64; LOW]; 3];
        ac[1][1] = 1; // h = 1, l = 1, o = 0 -> bit 0
        ac[2][15] = 1 << 34; // h = 2, l = 15, o = 34 -> bit 14 * 35 + 34 = 524
        let rows = rows_from_anf(&ac);
        assert_eq!(rows.len(), 2);
        assert!(rows[0].get(0) && rows[0].count_ones() == 1);
        assert!(rows[1].get(524) && rows[1].count_ones() == 1);
    }
}
