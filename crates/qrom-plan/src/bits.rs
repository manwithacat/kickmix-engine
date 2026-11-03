//! Fixed-width GF(2) vectors packed in u64 words.

use std::ops::{BitAnd, BitXor, BitXorAssign};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Bits<const W: usize>(pub [u64; W]);

impl<const W: usize> Default for Bits<W> {
    fn default() -> Self {
        Bits([0; W])
    }
}

impl<const W: usize> Bits<W> {
    pub const ZERO: Self = Bits([0; W]);

    #[inline]
    pub fn bit(i: usize) -> Self {
        let mut b = Self::ZERO;
        b.set(i);
        b
    }

    #[inline]
    pub fn get(&self, i: usize) -> bool {
        (self.0[i >> 6] >> (i & 63)) & 1 == 1
    }

    #[inline]
    pub fn set(&mut self, i: usize) {
        self.0[i >> 6] |= 1u64 << (i & 63);
    }

    #[inline]
    pub fn clear(&mut self, i: usize) {
        self.0[i >> 6] &= !(1u64 << (i & 63));
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }

    /// Index of the highest set bit.
    #[inline]
    pub fn leading(&self) -> Option<usize> {
        for i in (0..W).rev() {
            if self.0[i] != 0 {
                return Some(i * 64 + 63 - self.0[i].leading_zeros() as usize);
            }
        }
        None
    }

    #[inline]
    pub fn count_ones(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }

    /// Parity of the inner product self . other over GF(2).
    #[inline]
    pub fn dot(&self, other: &Self) -> bool {
        let mut acc = 0u64;
        for i in 0..W {
            acc ^= self.0[i] & other.0[i];
        }
        acc.count_ones() & 1 == 1
    }

    /// Indices of set bits, ascending.
    pub fn ones(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.count_ones() as usize);
        for (wi, &w) in self.0.iter().enumerate() {
            let mut x = w;
            while x != 0 {
                let t = x.trailing_zeros() as usize;
                out.push(wi * 64 + t);
                x &= x - 1;
            }
        }
        out
    }

    pub fn from_u128(v: u128) -> Self {
        let mut b = Self::ZERO;
        b.0[0] = v as u64;
        if W > 1 {
            b.0[1] = (v >> 64) as u64;
        }
        b
    }

    pub fn to_hex(&self) -> String {
        // big-endian hex without leading zeros, matching Python's hex(int)
        let mut s = String::new();
        let mut started = false;
        for i in (0..W).rev() {
            if !started {
                if self.0[i] == 0 {
                    continue;
                }
                s.push_str(&format!("{:x}", self.0[i]));
                started = true;
            } else {
                s.push_str(&format!("{:016x}", self.0[i]));
            }
        }
        if !started {
            s.push('0');
        }
        format!("0x{}", s)
    }

    pub fn from_hex(text: &str) -> Result<Self, String> {
        let t = text.trim();
        let t = t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")).unwrap_or(t);
        let mut b = Self::ZERO;
        let bytes = t.as_bytes();
        let n = bytes.len();
        for (pos, &c) in bytes.iter().rev().enumerate() {
            let v = (c as char).to_digit(16).ok_or_else(|| format!("bad hex digit in {text}"))? as u64;
            let bit = pos * 4;
            if bit >= W * 64 {
                if v != 0 {
                    return Err(format!("hex value too wide for {} bits: {text}", W * 64));
                }
                continue;
            }
            b.0[bit >> 6] |= v << (bit & 63);
        }
        let _ = n;
        Ok(b)
    }
}

impl<const W: usize> BitXorAssign for Bits<W> {
    #[inline]
    fn bitxor_assign(&mut self, rhs: Self) {
        for i in 0..W {
            self.0[i] ^= rhs.0[i];
        }
    }
}

impl<const W: usize> BitXor for Bits<W> {
    type Output = Self;
    #[inline]
    fn bitxor(mut self, rhs: Self) -> Self {
        self ^= rhs;
        self
    }
}

impl<const W: usize> BitAnd for Bits<W> {
    type Output = Self;
    #[inline]
    fn bitand(mut self, rhs: Self) -> Self {
        for i in 0..W {
            self.0[i] &= rhs.0[i];
        }
        self
    }
}

/// 525-bit (low monomial, output) vectors: bit (l - 1) * 35 + o.
pub type B525 = Bits<9>;
/// Up to 128-bit vectors (high-monomial masks, sub-matrix columns up to 105 bits).
pub type B128 = Bits<2>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_roundtrip() {
        let mut b = B525::ZERO;
        b.set(0);
        b.set(64);
        b.set(524);
        let h = b.to_hex();
        assert_eq!(B525::from_hex(&h).unwrap(), b);
        assert_eq!(B525::from_hex("0x0").unwrap(), B525::ZERO);
        assert_eq!(Bits::<1>::from_hex("ff").unwrap().0[0], 255);
    }

    #[test]
    fn leading_and_dot() {
        let mut a = B128::ZERO;
        a.set(100);
        a.set(3);
        assert_eq!(a.leading(), Some(100));
        let mut b = B128::ZERO;
        b.set(3);
        assert!(a.dot(&b));
        b.set(100);
        assert!(!a.dot(&b));
        assert_eq!(a.ones(), vec![3, 100]);
    }
}
