//! Bit-sliced simulator of one 256-shot kickmix `Sim`. Semantics follow kickmix (Apache-2.0, Google LLC); see NOTICE.

use crate::program::*;
use crate::rng::{ChaCha8, Keys};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitXor, BitXorAssign, Not};

/// 256 shots; bit s of the word = shot s (word s/64, bit s%64). Matches kickmix's TWord lane order.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C, align(16))]
pub struct W(pub [u64; 4]);

impl W {
    pub const ZERO: W = W([0; 4]);
    pub const ONES: W = W([!0; 4]);
    #[inline(always)]
    pub fn popcnt(self) -> u64 {
        (self.0[0].count_ones() + self.0[1].count_ones() + self.0[2].count_ones() + self.0[3].count_ones()) as u64
    }
}
macro_rules! binop {
    ($tr:ident, $f:ident, $op:tt) => {
        impl $tr for W {
            type Output = W;
            #[inline(always)]
            fn $f(self, o: W) -> W {
                W([self.0[0] $op o.0[0], self.0[1] $op o.0[1], self.0[2] $op o.0[2], self.0[3] $op o.0[3]])
            }
        }
    };
}
binop!(BitAnd, bitand, &);
binop!(BitOr, bitor, |);
binop!(BitXor, bitxor, ^);
impl Not for W {
    type Output = W;
    #[inline(always)]
    fn not(self) -> W {
        W([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
    }
}
impl BitAndAssign for W {
    #[inline(always)]
    fn bitand_assign(&mut self, o: W) {
        *self = *self & o;
    }
}
impl BitXorAssign for W {
    #[inline(always)]
    fn bitxor_assign(&mut self, o: W) {
        *self = *self ^ o;
    }
}

/// Per-sim operation counts, summed over shots (per op kind, merged over `_IF` variants) plus Z_POW level counts.
#[derive(Clone)]
pub struct Counts {
    full: [u64; NKINDS],
    part: [u64; NKINDS],
    zfull: [u64; 129],
    zpart: [u64; 129],
}
impl Counts {
    pub fn new() -> Self {
        Counts { full: [0; NKINDS], part: [0; NKINDS], zfull: [0; 129], zpart: [0; 129] }
    }
    /// Per kind totals (CXRUN folded into CX) over shots.
    pub fn kinds(&self) -> [u64; 17] {
        let mut out = [0u64; 17];
        for k in 0..17 {
            out[k] = self.full[k] * 256 + self.part[k];
        }
        out
    }
    pub fn zlevels(&self) -> [u64; 129] {
        let mut out = [0u64; 129];
        for k in 0..129 {
            out[k] = self.zfull[k] * 256 + self.zpart[k];
        }
        out
    }
}

pub struct Sim {
    pub q: Vec<W>,
    pub bits: Vec<W>,
    pub phase: W,
    pub angles: Vec<u128>,
    stack: Vec<W>,
    rng: ChaCha8,
    pub counts: Counts,
}

impl Sim {
    pub fn new(prog: &Program, keys: Keys) -> Sim {
        Sim {
            q: vec![W::ZERO; prog.num_qubits],
            bits: vec![W::ZERO; prog.num_bits.max(1)],
            phase: W::ZERO,
            angles: vec![0u128; 256],
            stack: Vec::with_capacity(8),
            rng: ChaCha8::new(keys),
            counts: Counts::new(),
        }
    }

    /// kickmix `clear_for_shot`: zeroes qubits, bits, phase, angles, condition stack; the RNG stream continues.
    pub fn clear_for_shot(&mut self) {
        self.q.fill(W::ZERO);
        self.bits.fill(W::ZERO);
        self.phase = W::ZERO;
        self.angles.fill(0);
        self.stack.clear();
    }

    pub fn run<const COUNT: bool>(&mut self, prog: &Program) {
        let ops = prog.ops.as_ptr();
        let n = prog.ops.len();
        let qp = self.q.as_mut_ptr();
        let bp = self.bits.as_mut_ptr();
        let angles = &mut self.angles;
        let stack = &mut self.stack;
        let rng = &mut self.rng;
        let counts = &mut self.counts;
        let mut phase = self.phase;
        let mut base = W::ONES;
        let mut full = true;
        let mut i = 0usize;
        // SAFETY: Program construction guarantees qubit ids < num_qubits (Sim::new sizes q from it) and compact bit
        // ids < num_bits; `ops` has n entries and run headers are followed by that many ops.
        unsafe {
            while i < n {
                let op = *ops.add(i);
                i += 1;
                let mut cond = base;
                let mut cfull = full;
                if op.flags & F_IF != 0 {
                    cond = cond & *bp.add(op.cond as usize);
                    cfull = false;
                }
                let (a, b, c) = (op.a as usize, op.b as usize, op.c as usize);
                let mut mult = 1u64;
                let mut kind = op.kind;
                match op.kind {
                    K_CX => {
                        let t = cond & *qp.add(a);
                        *qp.add(b) ^= t;
                    }
                    K_CXRUN => {
                        let cnt = a;
                        let run = ops.add(i);
                        i += cnt;
                        if full {
                            for j in 0..cnt {
                                let r = *run.add(j);
                                let t = *qp.add(r.a as usize);
                                *qp.add(r.b as usize) ^= t;
                            }
                        } else {
                            for j in 0..cnt {
                                let r = *run.add(j);
                                let t = base & *qp.add(r.a as usize);
                                *qp.add(r.b as usize) ^= t;
                            }
                        }
                        mult = cnt as u64;
                        kind = K_CX;
                    }
                    K_CZ => phase ^= cond & *qp.add(a) & *qp.add(b),
                    K_CCX => {
                        let t = cond & *qp.add(a) & *qp.add(b);
                        *qp.add(c) ^= t;
                    }
                    K_X => *qp.add(a) ^= cond,
                    K_Z => phase ^= cond & *qp.add(a),
                    K_NEG => phase ^= cond,
                    K_CCZ => phase ^= cond & *qp.add(a) & *qp.add(b) & *qp.add(c),
                    K_SWAP => {
                        let t = (*qp.add(a) ^ *qp.add(b)) & cond;
                        *qp.add(a) ^= t;
                        *qp.add(b) ^= t;
                    }
                    K_CSWAP => {
                        let m = cond & *qp.add(a);
                        let t = (*qp.add(b) ^ *qp.add(c)) & m;
                        *qp.add(b) ^= t;
                        *qp.add(c) ^= t;
                    }
                    K_ST0 => *bp.add(b) &= !cond,
                    K_ST1 => {
                        let v = *bp.add(b) | cond;
                        *bp.add(b) = v;
                    }
                    K_INV => *bp.add(b) ^= cond,
                    K_HMR => {
                        let r = W(rng.next());
                        let nb = (*bp.add(b) & !cond) ^ (r & cond);
                        *bp.add(b) = nb;
                        phase ^= *qp.add(a) & nb & cond;
                        *qp.add(a) &= !cond;
                    }
                    K_R => {
                        let r = W(rng.next());
                        phase ^= *qp.add(a) & r & cond;
                        *qp.add(a) &= !cond;
                    }
                    K_ZPOW => {
                        let ang = prog.angles[b];
                        if COUNT {
                            let lv = prog.levels[b] as usize;
                            if cfull {
                                counts.zfull[lv] += 1;
                            } else {
                                counts.zpart[lv] += cond.popcnt();
                            }
                        }
                        let mask = *qp.add(a) & cond;
                        for w in 0..4 {
                            let mut m = mask.0[w];
                            while m != 0 {
                                let j = m.trailing_zeros() as usize;
                                m &= m - 1;
                                let s = w * 64 + j;
                                angles[s] = angles[s].wrapping_add(ang);
                            }
                        }
                    }
                    K_PUSH => {
                        base &= cond_bit_only(&op, bp);
                        stack.push(base);
                        full = false;
                        cond = W::ONES;
                        cfull = true;
                    }
                    K_POP => {
                        stack.pop();
                        match stack.last() {
                            Some(&t) => {
                                base = t;
                                full = false;
                            }
                            None => {
                                base = W::ONES;
                                full = true;
                            }
                        }
                        cond = W::ONES;
                        cfull = true;
                    }
                    _ => std::hint::unreachable_unchecked(),
                }
                if COUNT {
                    let k = kind as usize;
                    if cfull {
                        counts.full[k] += mult;
                    } else {
                        counts.part[k] += mult * cond.popcnt();
                    }
                }
            }
        }
        self.phase = phase;
    }
}

/// PUSH_CONDITION uses its bit operand directly (`current_base &= bits[b]`); `cond` already includes the base,
/// so re-read the bit itself. (The F_IF flag is always set on PUSH_CONDITION.)
#[inline(always)]
unsafe fn cond_bit_only(op: &Op, bp: *mut W) -> W {
    *bp.add(op.cond as usize)
}

/// In-place 64x64 bit-matrix transpose: out[c] bit r = in[r] bit c.
pub fn transpose64(a: &mut [u64; 64]) {
    let mut j = 32usize;
    let mut m: u64 = 0x0000_0000_FFFF_FFFF;
    while j != 0 {
        let mut k = 0usize;
        while k < 64 {
            let t = ((a[k] >> j) ^ a[k + j]) & m;
            a[k + j] ^= t;
            a[k] ^= t << j;
            k = (k + j + 1) & !j;
        }
        j >>= 1;
        m ^= m << j;
    }
}
