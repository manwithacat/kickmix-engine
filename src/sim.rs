//! Bit-sliced simulator of one or more 256-shot kickmix `Sim`s run in lockstep. Semantics follow kickmix
//! (Apache-2.0, Google LLC); see NOTICE.
//!
//! `Wn<L>` holds `L` 64-bit words = `L / 4` independent 256-shot sims (`W = Wn<4>` is one sim, `W2 = Wn<8>` two).
//! Lockstep means one decode of every op applied to both states; the sims are independent (own RNG stream, own
//! counts, own inputs), so the shot-level results equal two single runs. Running two sims per op lets the core
//! overlap the two independent read-modify-write chains that dominate arithmetic-heavy circuits.

use crate::program::*;
use crate::rng::{ChaCha8, Keys};
use std::ops::{BitAnd, BitAndAssign, BitOr, BitXor, BitXorAssign, Not};

/// `L` words of 64 shots; bit s of word s/64 = shot s. Words 4k..4k+4 are sim k (kickmix's TWord lane order).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(C, align(16))]
pub struct Wn<const L: usize>(pub [u64; L]);

/// One 256-shot sim.
pub type W = Wn<4>;
/// Two 256-shot sims in lockstep.
pub type W2 = Wn<8>;

impl<const L: usize> Wn<L> {
    pub const ZERO: Wn<L> = Wn([0; L]);
    pub const ONES: Wn<L> = Wn([!0; L]);
    /// Sims held (256 shots each).
    pub const SIMS: usize = L / 4;
    #[inline(always)]
    pub const fn splat(x: u64) -> Wn<L> {
        Wn([x; L])
    }
    /// Set shots over all sims.
    #[inline(always)]
    pub fn popcnt(self) -> u64 {
        let mut n = 0u64;
        for i in 0..L {
            n += self.0[i].count_ones() as u64;
        }
        n
    }
    /// Set shots of sim `s`.
    #[inline(always)]
    pub fn popcnt_sim(self, s: usize) -> u64 {
        let b = 4 * s;
        (self.0[b].count_ones() + self.0[b + 1].count_ones() + self.0[b + 2].count_ones() + self.0[b + 3].count_ones()) as u64
    }
    /// The 256 bits of sim `s`.
    #[inline(always)]
    pub fn sim(self, s: usize) -> W {
        Wn([self.0[4 * s], self.0[4 * s + 1], self.0[4 * s + 2], self.0[4 * s + 3]])
    }
    /// One random word per sim from that sim's stream.
    #[inline(always)]
    fn from_rngs(rngs: &mut [ChaCha8]) -> Wn<L> {
        let mut out = [0u64; L];
        for s in 0..L / 4 {
            let r = rngs[s].next();
            out[4 * s..4 * s + 4].copy_from_slice(&r);
        }
        Wn(out)
    }
}
macro_rules! binop {
    ($tr:ident, $f:ident, $op:tt) => {
        impl<const L: usize> $tr for Wn<L> {
            type Output = Wn<L>;
            #[inline(always)]
            fn $f(self, o: Wn<L>) -> Wn<L> {
                let mut r = [0u64; L];
                for i in 0..L {
                    r[i] = self.0[i] $op o.0[i];
                }
                Wn(r)
            }
        }
    };
}
binop!(BitAnd, bitand, &);
binop!(BitOr, bitor, |);
binop!(BitXor, bitxor, ^);
impl<const L: usize> Not for Wn<L> {
    type Output = Wn<L>;
    #[inline(always)]
    fn not(self) -> Wn<L> {
        let mut r = [0u64; L];
        for i in 0..L {
            r[i] = !self.0[i];
        }
        Wn(r)
    }
}
impl<const L: usize> BitAndAssign for Wn<L> {
    #[inline(always)]
    fn bitand_assign(&mut self, o: Wn<L>) {
        *self = *self & o;
    }
}
impl<const L: usize> BitXorAssign for Wn<L> {
    #[inline(always)]
    fn bitxor_assign(&mut self, o: Wn<L>) {
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

/// `L / 4` sims in lockstep (see the module doc). `Sim = SimG<4>` is the single-sim form every caller used before.
pub struct SimG<const L: usize> {
    pub q: Vec<Wn<L>>,
    pub bits: Vec<Wn<L>>,
    pub phase: Wn<L>,
    /// 256 per sim: sim k's shot j is `angles[256 k + j]`.
    pub angles: Vec<u128>,
    stack: Vec<Wn<L>>,
    rng: Vec<ChaCha8>,
    /// One per sim.
    pub counts: Vec<Counts>,
}

pub type Sim = SimG<4>;
pub type Sim2 = SimG<8>;

impl Sim {
    pub fn new(prog: &Program, keys: Keys) -> Sim {
        Sim::with_dims(prog.num_qubits, prog.num_bits, prog.max_stack_depth, keys)
    }

    /// A sim with room for `num_qubits` qubits and `num_bits` compact bits, so several programs can run on it in
    /// turn (each program's ids are below its own counts; see `engine::run_sequence`).
    pub fn with_dims(num_qubits: usize, num_bits: usize, max_stack_depth: usize, keys: Keys) -> Sim {
        SimG::with_keys(num_qubits, num_bits, max_stack_depth, vec![keys])
    }
}

impl<const L: usize> SimG<L> {
    /// `keys.len()` must equal `L / 4` (one ChaCha8 key set per sim).
    pub fn with_keys(num_qubits: usize, num_bits: usize, max_stack_depth: usize, keys: Vec<Keys>) -> SimG<L> {
        assert_eq!(keys.len(), L / 4, "one key set per sim");
        let mut q = vec![Wn::<L>::ZERO; num_qubits + Q_OFF as usize];
        q[Q_ONES as usize] = Wn::<L>::ONES;
        let mut bits = vec![Wn::<L>::ZERO; num_bits + B_OFF as usize];
        bits[B_ONES as usize] = Wn::<L>::ONES;
        SimG {
            q,
            bits,
            phase: Wn::<L>::ZERO,
            angles: vec![0u128; 256 * (L / 4)],
            stack: Vec::with_capacity(max_stack_depth.max(8)),
            rng: keys.into_iter().map(ChaCha8::new).collect(),
            counts: (0..L / 4).map(|_| Counts::new()).collect(),
        }
    }

    /// Lockstep sims from a program.
    pub fn new_lockstep(prog: &Program, keys: Vec<Keys>) -> SimG<L> {
        SimG::with_keys(prog.num_qubits, prog.num_bits, prog.max_stack_depth, keys)
    }

    /// Qubit `i` (the array holds the constant slots first; see `program::Q_OFF`).
    #[inline(always)]
    pub fn qb(&self, i: usize) -> Wn<L> {
        self.q[i + Q_OFF as usize]
    }
    #[inline(always)]
    pub fn qb_mut(&mut self, i: usize) -> &mut Wn<L> {
        &mut self.q[i + Q_OFF as usize]
    }
    /// Compact classical bit `i` (slot 0 of the array is the constant all-ones bit; see `program::B_OFF`).
    #[inline(always)]
    pub fn bt(&self, i: usize) -> Wn<L> {
        self.bits[i + B_OFF as usize]
    }
    #[inline(always)]
    pub fn bt_mut(&mut self, i: usize) -> &mut Wn<L> {
        &mut self.bits[i + B_OFF as usize]
    }
    /// Number of compact classical bits this sim holds.
    pub fn num_bits(&self) -> usize {
        self.bits.len() - B_OFF as usize
    }

    /// kickmix `clear_for_shot`: zeroes qubits, bits, phase, angles, condition stack; the RNG streams continue.
    pub fn clear_for_shot(&mut self) {
        self.q.fill(Wn::<L>::ZERO);
        self.q[Q_ONES as usize] = Wn::<L>::ONES;
        self.bits.fill(Wn::<L>::ZERO);
        self.bits[B_ONES as usize] = Wn::<L>::ONES;
        self.phase = Wn::<L>::ZERO;
        self.angles.fill(0);
        self.stack.clear();
    }

    /// Final phase per shot as kickmix `read_phase_flipped_across_shots` reads it: `Err` if any shot's phase is
    /// not a multiple of a half turn.
    pub fn phase_flipped(&self, has_angles: bool) -> Result<Wn<L>, String> {
        self.phase_flipped_first(has_angles, 64 * L)
    }

    /// `phase_flipped` over the first `n` shots of every sim only (the rest of the result is unspecified).
    pub fn phase_flipped_first(&self, has_angles: bool, n: usize) -> Result<Wn<L>, String> {
        if !has_angles {
            return Ok(self.phase);
        }
        let mut out = Wn::<L>::ZERO;
        for s in 0..n.min(64 * L) {
            let ph = (self.phase.0[s / 64] >> (s % 64)) & 1;
            let v = self.angles[s] ^ ((ph as u128) << 127);
            if v << 1 != 0 {
                return Err("phase angles other than 0 and a half turn are present".into());
            }
            out.0[s / 64] |= ((v >> 127) as u64) << (s % 64);
        }
        Ok(out)
    }

    /// `phase_flipped` over the shots set in `lanes` only (the others are unspecified; for partial sims in lockstep).
    pub fn phase_flipped_lanes(&self, has_angles: bool, lanes: Wn<L>) -> Result<Wn<L>, String> {
        if !has_angles {
            return Ok(self.phase);
        }
        let mut out = Wn::<L>::ZERO;
        for s in 0..64 * L {
            if (lanes.0[s / 64] >> (s % 64)) & 1 == 0 {
                continue;
            }
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
        self.dispatch::<COUNT, false>(prog, Wn::<L>::ONES)
    }

    /// Like `run`, but operation counts include only the shots set in `lanes` (kickmix counts only the first
    /// `batch_size` shots of its last, partly used 256-shot sim). The state of every lane is simulated as usual.
    pub fn run_lanes<const COUNT: bool>(&mut self, prog: &Program, lanes: Wn<L>) {
        self.dispatch::<COUNT, true>(prog, lanes)
    }

    #[inline(always)]
    fn dispatch<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: Wn<L>) {
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
    unsafe fn run_avx2<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: Wn<L>) {
        self.run_impl::<COUNT, MASK>(prog, lanes)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,fma,lzcnt,movbe,avx512f,avx512bw,avx512vl,avx512dq")]
    unsafe fn run_avx512<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: Wn<L>) {
        self.run_impl::<COUNT, MASK>(prog, lanes)
    }

    /// The simulator loop. `inline(always)` so each kernel wrapper above gets its own copy compiled with its
    /// target features (`Wn`'s operations are `inline(always)` too and vectorise to the widest registers enabled).
    /// With `MASK = false` (every caller except `run_lanes`) `lanes` is unused and the loop is the 0.4.0 loop; with
    /// `MASK = true` every count goes to the per-shot sums (`part`) restricted to `lanes`.
    #[inline(always)]
    fn run_impl<const COUNT: bool, const MASK: bool>(&mut self, prog: &Program, lanes: Wn<L>) {
        const { assert!(L % 4 == 0 && L > 0) };
        let ns = L / 4;
        let code = prog.code.as_ptr();
        let n = prog.code.len();
        let qp = self.q.as_mut_ptr();
        let bp = self.bits.as_mut_ptr();
        let angles = &mut self.angles;
        let stack = &mut self.stack;
        let rng = &mut self.rng;
        let counts = &mut self.counts;
        let mut phase = self.phase;
        let mut base = Wn::<L>::ONES;
        let mut full = true;
        let mut i = 0usize;
        // Shots counted for a fully executed op, per sim (a partial last sim counts only its lanes).
        let mut nls = [256u64; 8];
        if MASK {
            for s in 0..ns {
                nls[s] = lanes.popcnt_sim(s);
            }
        }
        if COUNT {
            // INFO_TAG: counted with an all-ones mask on every execution, no other effect.
            for s in 0..ns {
                if MASK {
                    counts[s].part[K_INFO as usize] += prog.n_info_tags * nls[s];
                } else {
                    counts[s].full[K_INFO as usize] += prog.n_info_tags;
                }
            }
        }
        // Count helpers: `full` adds one executed op per sim, `part` the set shots of `cond` per sim.
        macro_rules! count_full {
            ($k:expr) => {
                for s in 0..ns {
                    counts[s].full[$k] += 1;
                }
            };
        }
        macro_rules! count_part {
            ($k:expr, $cond:expr) => {{
                let c = $cond;
                for s in 0..ns {
                    counts[s].part[$k] += c.popcnt_sim(s);
                }
            }};
        }
        // SAFETY: the Builder guarantees qubit slots < num_qubits + Q_OFF (the sim sizes q from it; uniform
        // entries read slot Q_ONES and write the sink slots), bit slots < num_bits + B_OFF, angle indices <
        // angles.len() / sims, every opcode has its OP_LEN words and run headers are followed by that many entries
        // (`Program::validate` re-checks a loaded cache file).
        unsafe {
            while i < n {
                let w0 = *code.add(i);
                let opc = (w0 & 0xFF) as u8;
                if opc == K_UNIRUN || opc == K_UNIRUN_IF {
                    // Branch-free run of NEG/Z/X/CZ/CX/CCZ/CCX, bit-conditioned (K_UNIRUN_IF) or not:
                    // `t = base & bits[cb] & q[a] & q[b] & q[c]; q[d] ^= t`, absent operands reading the all-ones
                    // qubit slot; a phase entry's `d` is a sink slot, which also selects (without a branch) the XOR
                    // of `t` into the register-resident phase. Counts: per (condition, kind, count) item, the
                    // condition's shot count (one popcount per distinct condition) times the count, which equals
                    // the per-op `cond.popcnt()` of the ordinary path.
                    let cnt = (w0 >> 8) as usize;
                    let nitems = *code.add(i + 1) as usize;
                    let items = code.add(i + 2);
                    let ent = code.add(i + 2 + UNI_ITEM * nitems);
                    let ew = if opc == K_UNIRUN_IF { UNI_ENTRY_IF } else { UNI_ENTRY_PLAIN };
                    i += 2 + UNI_ITEM * nitems + ew * cnt;
                    if COUNT {
                        let mut last_cb = usize::MAX;
                        let mut m = [0u64; 8];
                        let mut plain_full = false;
                        for it in 0..nitems {
                            let cb = *items.add(UNI_ITEM * it) as usize;
                            let w = *items.add(UNI_ITEM * it + 1);
                            let (k, n) = (UNI_KINDS[(w & 0xFF) as usize] as usize, (w >> 8) as u64);
                            if cb != last_cb {
                                last_cb = cb;
                                plain_full = full && cb == B_ONES as usize && !MASK;
                                if !plain_full {
                                    let c = if MASK { base & *bp.add(cb) & lanes } else { base & *bp.add(cb) };
                                    for s in 0..ns {
                                        m[s] = c.popcnt_sim(s);
                                    }
                                }
                            }
                            if plain_full {
                                for s in 0..ns {
                                    counts[s].full[k] += n;
                                }
                            } else {
                                for s in 0..ns {
                                    counts[s].part[k] += n * m[s];
                                }
                            }
                        }
                    }
                    if opc == K_UNIRUN_IF {
                        for j in 0..cnt {
                            let e0 = *ent.add(UNI_ENTRY_IF * j);
                            let e1 = *ent.add(UNI_ENTRY_IF * j + 1);
                            let cb = *ent.add(UNI_ENTRY_IF * j + 2) as usize;
                            let d = (e1 >> 16) as usize;
                            let t = base & *bp.add(cb) & *qp.add((e0 & 0xFFFF) as usize) & *qp.add((e0 >> 16) as usize) & *qp.add((e1 & 0xFFFF) as usize);
                            let pm = Wn::<L>::splat(((d.wrapping_sub(Q_DUMMY as usize) < N_DUMMY as usize) as u64).wrapping_neg());
                            phase ^= t & pm;
                            *qp.add(d) ^= t;
                        }
                    } else {
                        for j in 0..cnt {
                            let e0 = *ent.add(UNI_ENTRY_PLAIN * j);
                            let e1 = *ent.add(UNI_ENTRY_PLAIN * j + 1);
                            let d = (e1 >> 16) as usize;
                            let t = base & *qp.add((e0 & 0xFFFF) as usize) & *qp.add((e0 >> 16) as usize) & *qp.add((e1 & 0xFFFF) as usize);
                            let pm = Wn::<L>::splat(((d.wrapping_sub(Q_DUMMY as usize) < N_DUMMY as usize) as u64).wrapping_neg());
                            phase ^= t & pm;
                            *qp.add(d) ^= t;
                        }
                    }
                    continue;
                }
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
                            for s in 0..ns {
                                if MASK {
                                    counts[s].part[K_CX as usize] += cnt as u64 * nls[s];
                                } else {
                                    counts[s].full[K_CX as usize] += cnt as u64;
                                }
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
                            for s in 0..ns {
                                counts[s].part[K_CX as usize] += cnt as u64 * m.popcnt_sim(s);
                            }
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
                        let r = Wn::<L>::from_rngs(rng);
                        let nb = (*bp.add(t) & !cond) ^ (r & cond);
                        *bp.add(t) = nb;
                        phase ^= *qp.add(a) & nb & cond;
                        *qp.add(a) &= !cond;
                    }
                    K_R => {
                        let r = Wn::<L>::from_rngs(rng);
                        phase ^= *qp.add(a) & r & cond;
                        *qp.add(a) &= !cond;
                    }
                    K_ZPOW => {
                        let idx = *code.add(i + 1) as usize;
                        let ang = *prog.angles.get_unchecked(idx);
                        if COUNT {
                            let lv = *prog.levels.get_unchecked(idx) as usize;
                            if MASK {
                                let c = cond & lanes;
                                for s in 0..ns {
                                    counts[s].zpart[lv] += if cfull { nls[s] } else { c.popcnt_sim(s) };
                                }
                            } else if cfull {
                                for s in 0..ns {
                                    counts[s].zfull[lv] += 1;
                                }
                            } else {
                                for s in 0..ns {
                                    counts[s].zpart[lv] += cond.popcnt_sim(s);
                                }
                            }
                        }
                        let mask = *qp.add(a) & cond;
                        for w in 0..L {
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
                        cond = Wn::<L>::ONES;
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
                                base = Wn::<L>::ONES;
                                full = true;
                            }
                        }
                        cond = Wn::<L>::ONES;
                        cfull = true;
                    }
                    _ => std::hint::unreachable_unchecked(),
                }
                i += len;
                if COUNT {
                    let k = kind as usize;
                    if MASK {
                        if cfull {
                            for s in 0..ns {
                                counts[s].part[k] += nls[s];
                            }
                        } else {
                            count_part!(k, cond & lanes);
                        }
                    } else if cfull {
                        count_full!(k);
                    } else {
                        count_part!(k, cond);
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
