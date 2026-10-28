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
    pub full: [u64; NCOUNT],
    pub part: [u64; NCOUNT],
    zfull: [u64; 129],
    zpart: [u64; 129],
}
impl Default for Counts {
    fn default() -> Self {
        Self::new()
    }
}
impl Counts {
    pub fn new() -> Self {
        Counts { full: [0; NCOUNT], part: [0; NCOUNT], zfull: [0; 129], zpart: [0; 129] }
    }
    /// Per kind totals over the 256 shots (count slots, see `KIND_NAMES`).
    pub fn kinds(&self) -> [u64; NCOUNT] {
        let mut out = [0u64; NCOUNT];
        for k in 0..NCOUNT {
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
        Sim::with_dims(prog.num_qubits, prog.num_bits, prog.max_stack_depth, keys)
    }

    /// A sim with room for `num_qubits` qubits and `num_bits` compact bits, so several programs can run on it in
    /// turn (each program's ids are below its own counts; see `engine::run_sequence`).
    pub fn with_dims(num_qubits: usize, num_bits: usize, max_stack_depth: usize, keys: Keys) -> Sim {
        Sim {
            q: vec![W::ZERO; num_qubits.max(1)],
            bits: vec![W::ZERO; num_bits.max(1)],
            phase: W::ZERO,
            angles: vec![0u128; 256],
            stack: Vec::with_capacity(max_stack_depth.max(8)),
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

    /// Final phase per shot as kickmix `read_phase_flipped_across_shots` reads it: `Err` if any shot's phase is
    /// not a multiple of a half turn.
    pub fn phase_flipped(&self, has_angles: bool) -> Result<W, String> {
        self.phase_flipped_first(has_angles, 256)
    }

    /// `phase_flipped` over the first `n` shots only (the rest of the result is unspecified).
    pub fn phase_flipped_first(&self, has_angles: bool, n: usize) -> Result<W, String> {
        if !has_angles {
            return Ok(self.phase);
        }
        let mut out = W::ZERO;
        for s in 0..n.min(256) {
            let ph = (self.phase.0[s / 64] >> (s % 64)) & 1;
            let v = self.angles[s] ^ ((ph as u128) << 127);
            if v << 1 != 0 {
                return Err("phase angles other than 0 and a half turn are present".into());
            }
            out.0[s / 64] |= ((v >> 127) as u64) << (s % 64);
        }
        Ok(out)
    }

    /// Run the program on this sim with the process's kernel (see `kernel`): the same body compiled for baseline
    /// x86-64, AVX2 (x86-64-v3) or AVX-512 (x86-64-v4); on aarch64 the single (NEON) build.
    pub fn run<const COUNT: bool>(&mut self, prog: &Program) {
        self.dispatch::<COUNT, false>(prog, W::ONES)
    }

    /// Like `run`, but operation counts include only the shots set in `lanes` (kickmix counts only the first
    /// `batch_size` shots of its last, partly used 256-shot sim). The state of every lane is simulated as usual.
    pub fn run_lanes<const COUNT: bool>(&mut self, prog: &Program, lanes: W) {
        self.dispatch::<COUNT, true>(prog, lanes)
    }

    #[inline(always)]
    fn dispatch<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: W) {
        #[cfg(target_arch = "x86_64")]
        {
            use crate::kernel::Kernel;
            // SAFETY: `kernel::active()` returns Avx2/Avx512 only if `is_x86_feature_detected!` found every
            // feature these copies are compiled with.
            match crate::kernel::active() {
                Kernel::Avx512 => unsafe { self.run_avx512::<COUNT, MASK>(prog, lanes) },
                Kernel::Avx2 => unsafe { self.run_avx2::<COUNT, MASK>(prog, lanes) },
                _ => self.run_impl::<COUNT, MASK>(prog, lanes),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        self.run_impl::<COUNT, MASK>(prog, lanes)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,fma,lzcnt,movbe")]
    unsafe fn run_avx2<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: W) {
        self.run_impl::<COUNT, MASK>(prog, lanes)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,fma,lzcnt,movbe,avx512f,avx512bw,avx512vl,avx512dq")]
    unsafe fn run_avx512<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: W) {
        self.run_impl::<COUNT, MASK>(prog, lanes)
    }

    /// The simulator loop. `inline(always)` so each kernel wrapper above gets its own copy compiled with its
    /// target features (W's operations are `inline(always)` too and vectorise to the widest registers enabled).
    /// With `MASK = false` (every caller except `run_lanes`) `lanes` is unused and the loop is the 0.4.0 loop; with
    /// `MASK = true` every count goes to the per-shot sums (`part`) restricted to `lanes`.
    #[inline(always)]
    fn run_impl<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: W) {
        let code = prog.code.as_ptr();
        let n = prog.code.len();
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
        let nl = if MASK { lanes.popcnt() } else { 256 };
        if COUNT {
            // INFO_TAG: counted with an all-ones mask on every execution, no other effect.
            if MASK {
                counts.part[K_INFO as usize] += prog.n_info_tags * nl;
            } else {
                counts.full[K_INFO as usize] += prog.n_info_tags;
            }
        }
        // SAFETY: the Builder guarantees qubit ids < num_qubits (Sim::new sizes q from it), compact bit ids <
        // num_bits, angle indices < angles.len(), every opcode has its OP_LEN words and run headers are followed
        // by that many entries.
        unsafe {
            while i < n {
                let w0 = *code.add(i);
                let opc = (w0 & 0xFF) as u8;
                if opc == K_CXRUN {
                    let cnt = (w0 >> 8) as usize;
                    let run = code.add(i + 1);
                    i += 1 + cnt;
                    if full {
                        for j in 0..cnt {
                            let r = *run.add(j);
                            let t = *qp.add((r & 0xFFFF) as usize);
                            *qp.add((r >> 16) as usize) ^= t;
                        }
                        if COUNT {
                            if MASK {
                                counts.part[K_CX as usize] += cnt as u64 * nl;
                            } else {
                                counts.full[K_CX as usize] += cnt as u64;
                            }
                        }
                    } else {
                        for j in 0..cnt {
                            let r = *run.add(j);
                            let t = base & *qp.add((r & 0xFFFF) as usize);
                            *qp.add((r >> 16) as usize) ^= t;
                        }
                        if COUNT {
                            let m = if MASK { base & lanes } else { base };
                            counts.part[K_CX as usize] += cnt as u64 * m.popcnt();
                        }
                    }
                    continue;
                }
                let len = OP_LEN[opc as usize] as usize;
                debug_assert!(len != 0);
                let a = (w0 >> 16) as usize;
                let mut cond = base;
                let mut cfull = full;
                if opc & IF != 0 {
                    cond = base & *bp.add(*code.add(i + len - 1) as usize);
                    cfull = false;
                }
                let kind = opc & !IF;
                match kind {
                    K_CX => {
                        let b = (*code.add(i + 1) & 0xFFFF) as usize;
                        let t = cond & *qp.add(a);
                        *qp.add(b) ^= t;
                    }
                    K_CZ => {
                        let b = (*code.add(i + 1) & 0xFFFF) as usize;
                        phase ^= cond & *qp.add(a) & *qp.add(b);
                    }
                    K_SWAP => {
                        let b = (*code.add(i + 1) & 0xFFFF) as usize;
                        let t = (*qp.add(a) ^ *qp.add(b)) & cond;
                        *qp.add(a) ^= t;
                        *qp.add(b) ^= t;
                    }
                    K_CCX => {
                        let w1 = *code.add(i + 1);
                        let (b, c) = ((w1 & 0xFFFF) as usize, (w1 >> 16) as usize);
                        let t = cond & *qp.add(a) & *qp.add(b);
                        *qp.add(c) ^= t;
                    }
                    K_CCZ => {
                        let w1 = *code.add(i + 1);
                        let (b, c) = ((w1 & 0xFFFF) as usize, (w1 >> 16) as usize);
                        phase ^= cond & *qp.add(a) & *qp.add(b) & *qp.add(c);
                    }
                    K_CSWAP => {
                        let w1 = *code.add(i + 1);
                        let (b, c) = ((w1 & 0xFFFF) as usize, (w1 >> 16) as usize);
                        let m = cond & *qp.add(a);
                        let t = (*qp.add(b) ^ *qp.add(c)) & m;
                        *qp.add(b) ^= t;
                        *qp.add(c) ^= t;
                    }
                    K_X => *qp.add(a) ^= cond,
                    K_Z => phase ^= cond & *qp.add(a),
                    K_NEG => phase ^= cond,
                    K_ST0 => *bp.add(*code.add(i + 1) as usize) &= !cond,
                    K_ST1 => {
                        let t = *code.add(i + 1) as usize;
                        let v = *bp.add(t) | cond;
                        *bp.add(t) = v;
                    }
                    K_INV => *bp.add(*code.add(i + 1) as usize) ^= cond,
                    K_HMR => {
                        let t = *code.add(i + 1) as usize;
                        let r = W(rng.next());
                        let nb = (*bp.add(t) & !cond) ^ (r & cond);
                        *bp.add(t) = nb;
                        phase ^= *qp.add(a) & nb & cond;
                        *qp.add(a) &= !cond;
                    }
                    K_R => {
                        let r = W(rng.next());
                        phase ^= *qp.add(a) & r & cond;
                        *qp.add(a) &= !cond;
                    }
                    K_ZPOW => {
                        let idx = *code.add(i + 1) as usize;
                        let ang = *prog.angles.get_unchecked(idx);
                        if COUNT {
                            let lv = *prog.levels.get_unchecked(idx) as usize;
                            if MASK {
                                counts.zpart[lv] += if cfull { nl } else { (cond & lanes).popcnt() };
                            } else if cfull {
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
                        base &= *bp.add(*code.add(i + 1) as usize);
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
                i += len;
                if COUNT {
                    let k = kind as usize;
                    if MASK {
                        counts.part[k] += if cfull { nl } else { (cond & lanes).popcnt() };
                    } else if cfull {
                        counts.full[k] += 1;
                    } else {
                        counts.part[k] += cond.popcnt();
                    }
                }
            }
        }
        self.phase = phase;
    }
}

/// In-place 64x64 bit-matrix transpose: out[c] bit r = in[r] bit c.
#[inline]
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
