//! Batch driver: Rayon over (chunk, 256-shot sim) units sharing one read-only program.
//! A "chunk" is one kickmix `Simulator(batch_size=256*S, seed=...)` run for `B` consecutive batches (one RNG stream
//! per 256-shot sim, continuing across batches), exactly as the invsqrt verifier drives it.

use crate::program::*;
use crate::rng::*;
use crate::sim::*;
use rayon::prelude::*;

pub struct RunSpec<'a> {
    pub prog: &'a Program,
    /// One seed per chunk: 8 little-endian u32 words of the 256-bit simulator seed.
    pub seeds: &'a [[u32; 8]],
    /// 256-shot sims per batch (kickmix batch_size = 256*S).
    pub s: usize,
    /// Batches per chunk.
    pub b: usize,
    /// Inputs, flat [chunk][batch][256*S].
    pub xs: &'a [u64],
    pub xreg: &'a [u16],
    pub counting: bool,
    pub dump: bool,
    pub threads: usize,
    /// invsqrt per-shot check on the engine (see `Check`); then `x_out`, `dirty` and `units` are not produced.
    pub check: Option<&'a Check<'a>>,
    /// Run chunk sims in lockstep pairs (`SimG<8>`); false runs every sim alone (same results, for tests).
    pub lockstep: bool,
}

/// The official invsqrt per-shot check, restated: shot with input `x` passes if the x register reads back `x`, no
/// scratch qubit is set and `|theta - target[x - 1]| <= tol[x - 1]` in units of 2^-64 turns (the difference taken
/// as a wrapping i64, its absolute value as u64, so -2^63 maps to 2^63 as the verifier's NumPy does).
pub struct Check<'a> {
    pub target: &'a [u64],
    pub tol: &'a [u64],
}

/// Per-chunk results of the check. `fail` has six i64 per chunk: batch (-1 = the chunk passed), priority
/// (0 x not restored, 1 scratch dirty, 2 phase out of tolerance; the verifier's order within a batch), shot, x,
/// error (i64) and allowed; the batch is the first failing one in the chunk, the shot the first of that priority.
/// `min_margin` is min(tol - |error|) and (`worst_ratio`, `worst_x`) the largest |error| / tol with, on ties, the
/// smallest x, both over every shot of the chunk (meaningful when the chunk passed).
pub struct CheckOut {
    pub min_margin: Vec<u64>,
    pub worst_ratio: Vec<f64>,
    pub worst_x: Vec<u64>,
    pub fail: Vec<i64>,
}

#[derive(Clone, Copy)]
struct UnitCheck {
    min_margin: u64,
    worst_ratio: f64,
    worst_x: u64,
    /// (batch, priority, shot, x, error, allowed)
    fail: Option<(usize, u8, usize, u64, i64, u64)>,
}

impl UnitCheck {
    fn new() -> Self {
        UnitCheck { min_margin: u64::MAX, worst_ratio: -1.0, worst_x: 0, fail: None }
    }
    fn merge(&mut self, o: &UnitCheck) {
        self.min_margin = self.min_margin.min(o.min_margin);
        if o.worst_ratio > self.worst_ratio || (o.worst_ratio == self.worst_ratio && o.worst_x < self.worst_x) {
            self.worst_ratio = o.worst_ratio;
            self.worst_x = o.worst_x;
        }
        if let Some(f) = o.fail {
            let key = |f: (usize, u8, usize, u64, i64, u64)| (f.0, f.1, f.2);
            if self.fail.map_or(true, |s| key(f) < key(s)) {
                self.fail = Some(f);
            }
        }
    }
}

#[derive(Default)]
pub struct Dump {
    /// [unit][qubit][4 words]
    pub q: Vec<u64>,
    /// [unit][compact bit][4 words]
    pub bits: Vec<u64>,
    /// [unit][4 words] global phase bits
    pub phase: Vec<u64>,
    /// [unit][256][lo, hi] final 128-bit phase V (phase flip applied at bit 127), as read by the verifier.
    pub vphase: Vec<u64>,
}

pub struct Out {
    pub x_out: Vec<u64>,
    pub dirty: Vec<u8>,
    pub units: Vec<u64>,
    /// [chunk][NCOUNT]
    pub kind_counts: Vec<u64>,
    /// [chunk][129]
    pub z_levels: Vec<u64>,
    pub dump: Dump,
    pub check: Option<CheckOut>,
}

#[derive(Clone, Copy)]
struct SP<T>(*mut T);
unsafe impl<T> Send for SP<T> {}
unsafe impl<T> Sync for SP<T> {}
impl<T> SP<T> {
    #[inline(always)]
    fn p(self) -> *mut T {
        self.0
    }
}

/// Output pointers shared by the unit tasks (each writes disjoint ranges).
#[derive(Clone, Copy)]
struct OutPtrs {
    px: SP<u64>,
    pd: SP<u8>,
    pu: SP<u64>,
    dq: SP<u64>,
    dbits: SP<u64>,
    dph: SP<u64>,
    dv: SP<u64>,
}

pub fn run(spec: &RunSpec) -> Result<Out, String> {
    crate::kernel::selected()?;
    let nch = spec.seeds.len();
    let (s, bb) = (spec.s, spec.b);
    let shots_per_batch = 256 * s;
    let total = nch * bb * shots_per_batch;
    if spec.xs.len() != total {
        return Err(format!("xs has {} entries, expected {}", spec.xs.len(), total));
    }
    if spec.xreg.len() > 64 || spec.xreg.iter().any(|&q| q as usize >= spec.prog.num_qubits) {
        return Err("bad x register".into());
    }
    let nq = spec.prog.num_qubits;
    let nb = spec.prog.num_bits.max(1);
    let is_x: Vec<bool> = {
        let mut v = vec![false; nq];
        for &q in spec.xreg {
            v[q as usize] = true;
        }
        v
    };
    let scratch: Vec<usize> = (0..nq).filter(|&i| !is_x[i]).collect();
    if let Some(ck) = spec.check {
        if ck.target.len() != ck.tol.len() {
            return Err("check: target and tol tables differ in length".into());
        }
        if spec.xs.iter().any(|&x| x == 0 || x > ck.target.len() as u64) {
            return Err("check: an input x is outside the reference table (x must be in 1..=len)".into());
        }
    }
    let checking = spec.check.is_some();
    let (mut x_out, mut dirty, mut units) = if checking {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        (vec![0u64; total], vec![0u8; total], vec![0u64; total])
    };
    let nunits_batches = nch * bb * s;
    let mut dump = Dump::default();
    if spec.dump {
        dump.q = vec![0; nunits_batches * nq * 4];
        dump.bits = vec![0; nunits_batches * nb * 4];
        dump.phase = vec![0; nunits_batches * 4];
        dump.vphase = vec![0; nunits_batches * 256 * 2];
    }
    let ptrs = OutPtrs {
        px: SP(x_out.as_mut_ptr()),
        pd: SP(dirty.as_mut_ptr()),
        pu: SP(units.as_mut_ptr()),
        dq: SP(dump.q.as_mut_ptr()),
        dbits: SP(dump.bits.as_mut_ptr()),
        dph: SP(dump.phase.as_mut_ptr()),
        dv: SP(dump.vphase.as_mut_ptr()),
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(spec.threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let nunits = nch * s;
    let lockstep = spec.lockstep;
    let results: Vec<Vec<(Counts, UnitCheck)>> = pool.install(|| {
        // Per-chunk: the master stream hands out 4 draws per sim, in order.
        let sim_seeds: Vec<Vec<[u32; 8]>> = spec
            .seeds
            .par_iter()
            .map(|sd| {
                let mut master = Mt64::from_seed_seq(sd);
                (0..s).map(|_| sim_seed_words(&mut master)).collect()
            })
            .collect();
        // Units (one 256-shot sim each) run in lockstep pairs; a trailing single unit runs alone.
        let npairs = if lockstep { nunits / 2 } else { 0 };
        let ntasks = npairs + (nunits - 2 * npairs);
        (0..ntasks)
            .into_par_iter()
            .with_min_len(1)
            .map(|t| {
                if t < npairs {
                    run_units::<8>(spec, &[2 * t, 2 * t + 1], &sim_seeds, &scratch, ptrs)
                } else {
                    run_units::<4>(spec, &[2 * npairs + (t - npairs)], &sim_seeds, &scratch, ptrs)
                }
            })
            .collect()
    });
    let mut kind_counts = vec![0u64; nch * NCOUNT];
    let mut z_levels = vec![0u64; nch * 129];
    let mut chunk_checks = vec![UnitCheck::new(); if checking { nch } else { 0 }];
    for (u, (c, uc)) in results.iter().flatten().enumerate() {
        let chunk = u / s;
        for (i, v) in c.kinds().iter().enumerate() {
            kind_counts[chunk * NCOUNT + i] += v;
        }
        for (i, v) in c.zlevels().iter().enumerate() {
            z_levels[chunk * 129 + i] += v;
        }
        if checking {
            chunk_checks[chunk].merge(uc);
        }
    }
    let check = if checking {
        let mut co = CheckOut { min_margin: Vec::with_capacity(nch), worst_ratio: Vec::with_capacity(nch), worst_x: Vec::with_capacity(nch), fail: Vec::with_capacity(nch * 6) };
        for uc in &chunk_checks {
            co.min_margin.push(uc.min_margin);
            co.worst_ratio.push(uc.worst_ratio);
            co.worst_x.push(uc.worst_x);
            match uc.fail {
                Some((b, p, sh, x, e, al)) => co.fail.extend_from_slice(&[b as i64, p as i64, sh as i64, x as i64, e, al as i64]),
                None => co.fail.extend_from_slice(&[-1, -1, -1, 0, 0, 0]),
            }
        }
        Some(co)
    } else {
        None
    };
    Ok(Out { x_out, dirty, units, kind_counts, z_levels, dump, check })
}

/// `L / 4` units (chunk sims, `unit = chunk * s + k`) through all `bb` batches on one lockstep `SimG<L>`: inputs
/// written per sim, the program run once per batch, outputs (or the invsqrt check) read per sim. Returns one
/// (counts, check) per unit, in `units` order.
fn run_units<const L: usize>(spec: &RunSpec, units: &[usize], sim_seeds: &[Vec<[u32; 8]>], scratch: &[usize], p: OutPtrs) -> Vec<(Counts, UnitCheck)> {
    let ns = L / 4;
    debug_assert_eq!(units.len(), ns);
    let (s, bb) = (spec.s, spec.b);
    let shots_per_batch = 256 * s;
    let nq = spec.prog.num_qubits;
    let nb = spec.prog.num_bits.max(1);
    let keys: Vec<_> = units.iter().map(|&u| keys_from_sim_seed(&sim_seeds[u / s][u % s])).collect();
    let mut sim = SimG::<L>::new_lockstep(spec.prog, keys);
    let width = spec.xreg.len();
    let mut rows = [0u64; 64];
    let mut ucs = vec![UnitCheck::new(); ns];
    let mut done = vec![false; ns];
    let off_of = |u: usize, b: usize| (((u / s) * bb + b) * shots_per_batch) + (u % s) * 256;
    for b in 0..bb {
        sim.clear_for_shot();
        for (si, &u) in units.iter().enumerate() {
            let off = off_of(u, b);
            for w in 0..4 {
                rows.copy_from_slice(&spec.xs[off + 64 * w..off + 64 * w + 64]);
                transpose64(&mut rows);
                for i in 0..width {
                    sim.qb_mut(spec.xreg[i] as usize).0[4 * si + w] = rows[i];
                }
            }
        }
        if spec.counting {
            sim.run::<true>(spec.prog);
        } else {
            sim.run::<false>(spec.prog);
        }
        let mut dw = Wn::<L>::ZERO;
        for &i in scratch {
            dw = dw | sim.qb(i);
        }
        for (si, &u) in units.iter().enumerate() {
            if done[si] {
                continue;
            }
            let off = off_of(u, b);
            let ub = ((u / s) * bb + b) * s + u % s;
            if let Some(ck) = spec.check {
                // The verifier's per-batch check in its order: x restored, scratch clean, phase in tolerance;
                // margins and the worst ratio over every shot of a passing batch.
                let mut bad_x = false;
                for w in 0..4 {
                    rows = [0u64; 64];
                    for i in 0..width {
                        rows[i] = sim.qb(spec.xreg[i] as usize).0[4 * si + w];
                    }
                    transpose64(&mut rows);
                    for r in 0..64 {
                        bad_x |= rows[r] != spec.xs[off + 64 * w + r];
                    }
                }
                let uc = &mut ucs[si];
                let mut fail: Option<(usize, u8, usize, u64, i64, u64)> = None;
                if bad_x {
                    fail = Some((b, 0, 0, 0, 0, 0));
                } else if dw.sim(si) != W::ZERO {
                    let sh = (0..256).find(|&sh| (dw.0[4 * si + sh / 64] >> (sh % 64)) & 1 != 0).unwrap();
                    fail = Some((b, 1, sh, spec.xs[off + sh], 0, 0));
                }
                if fail.is_none() {
                    for sh in 0..256 {
                        let ph = (sim.phase.0[4 * si + sh / 64] >> (sh % 64)) & 1;
                        let v = sim.angles[256 * si + sh] ^ ((ph as u128) << 127);
                        let theta = (((v >> 63) + 1) >> 1) as u64;
                        let x = spec.xs[off + sh];
                        let idx = (x - 1) as usize;
                        let (target, allowed) = (ck.target[idx], ck.tol[idx]);
                        let err = theta.wrapping_sub(target) as i64;
                        let used = err.unsigned_abs();
                        if used > allowed {
                            fail = Some((b, 2, sh, x, err, allowed));
                            break;
                        }
                        uc.min_margin = uc.min_margin.min(allowed - used);
                        let ratio = if allowed > 0 { used as f64 / allowed as f64 } else { 0.0 };
                        if ratio > uc.worst_ratio || (ratio == uc.worst_ratio && x < uc.worst_x) {
                            uc.worst_ratio = ratio;
                            uc.worst_x = x;
                        }
                    }
                }
                if fail.is_some() {
                    uc.fail = fail;
                    done[si] = true;
                }
                continue;
            }
            // SAFETY: each (chunk, batch, k) writes a disjoint range of the outputs.
            unsafe {
                for w in 0..4 {
                    rows = [0u64; 64];
                    for i in 0..width {
                        rows[i] = sim.qb(spec.xreg[i] as usize).0[4 * si + w];
                    }
                    transpose64(&mut rows);
                    for r in 0..64 {
                        *p.px.p().add(off + 64 * w + r) = rows[r];
                    }
                }
                for sh in 0..256 {
                    let ph = (sim.phase.0[4 * si + sh / 64] >> (sh % 64)) & 1;
                    let v = sim.angles[256 * si + sh] ^ ((ph as u128) << 127);
                    *p.pu.p().add(off + sh) = (((v >> 63) + 1) >> 1) as u64;
                    *p.pd.p().add(off + sh) = ((dw.0[4 * si + sh / 64] >> (sh % 64)) & 1) as u8;
                    if spec.dump {
                        *p.dv.p().add(ub * 512 + sh * 2) = v as u64;
                        *p.dv.p().add(ub * 512 + sh * 2 + 1) = (v >> 64) as u64;
                    }
                }
                if spec.dump {
                    for i in 0..nq {
                        for w in 0..4 {
                            *p.dq.p().add((ub * nq + i) * 4 + w) = sim.qb(i).0[4 * si + w];
                        }
                    }
                    for i in 0..spec.prog.num_bits {
                        for w in 0..4 {
                            *p.dbits.p().add((ub * nb + i) * 4 + w) = sim.bt(i).0[4 * si + w];
                        }
                    }
                    for w in 0..4 {
                        *p.dph.p().add(ub * 4 + w) = sim.phase.0[4 * si + w];
                    }
                }
            }
        }
        if spec.check.is_some() && done.iter().all(|&d| d) {
            break;
        }
    }
    (0..ns).map(|si| (sim.counts[si].clone(), ucs[si])).collect()
}

// ---------------------------------------------------------------------------------------------------------------
// M3: independent 256-shot batches, one kickmix `Simulator(batch_size=256, seed)` each (the DQI verifier's model),
// with bit-sliced qubit and classical-bit inputs and on-engine value checks.

/// Shared Rayon pools by thread count, so concurrent calls (from several Python threads) share workers.
pub fn pool(threads: usize) -> std::sync::Arc<rayon::ThreadPool> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let m = POOLS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut g = m.lock().unwrap();
    g.entry(threads.max(1))
        .or_insert_with(|| Arc::new(rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build().expect("rayon pool")))
        .clone()
}

pub struct BatchSpec<'a> {
    pub prog: &'a Program,
    /// One simulator seed per batch (8 LE u32 words of the 256-bit seed).
    pub seeds: &'a [[u32; 8]],
    /// Input qubit ids and their bit-sliced values, flat [batch][qin][4].
    pub qin_ids: &'a [u32],
    pub qin: &'a [u64],
    /// Input classical bits as compact ids and values, flat [batch][bin][4].
    pub bin_ids: &'a [u32],
    pub bin: &'a [u64],
    /// Value check: input qubits must be restored, all other qubits must be 0.
    pub counting: bool,
    pub dump: bool,
    pub threads: usize,
}

pub struct BatchOut {
    /// [batch][4] shots failing the value check.
    pub val_err: Vec<u64>,
    /// [batch][4] phase flipped (kickmix `read_phase_flipped_across_shots`).
    pub phase: Vec<u64>,
    /// [batch][256] final phase per shot in units of 2^-64 turns (the invsqrt verifier's rounding of the exact
    /// 128-bit phase, flip included), so two circuits can be compared phase for phase.
    pub units: Vec<u64>,
    /// [batch][NCOUNT]
    pub kind_counts: Vec<u64>,
    /// [batch][129]
    pub z_levels: Vec<u64>,
    /// dump: [batch][qubit][4], [batch][compact bit][4]
    pub dump_q: Vec<u64>,
    pub dump_bits: Vec<u64>,
}

pub fn run_batches(spec: &BatchSpec) -> Result<BatchOut, String> {
    crate::kernel::selected()?;
    let p = spec.prog;
    let nb = spec.seeds.len();
    let (nqi, nbi) = (spec.qin_ids.len(), spec.bin_ids.len());
    if spec.qin.len() != nb * nqi * 4 || spec.bin.len() != nb * nbi * 4 {
        return Err("input array sizes do not match ids x batches x 4".into());
    }
    let nq = p.num_qubits;
    let mut is_in = vec![false; nq];
    for &q in spec.qin_ids {
        let q = q as usize;
        if q >= nq || is_in[q] {
            return Err("input qubit id out of range or repeated".into());
        }
        is_in[q] = true;
    }
    if spec.bin_ids.iter().any(|&b| b as usize >= p.num_bits) {
        return Err("input bit id out of range (must be compact ids)".into());
    }
    let others: Vec<usize> = (0..nq).filter(|&i| !is_in[i]).collect();
    let has_angles = !p.angles.is_empty();
    let ncb = p.num_bits.max(1);

    let mut val_err = vec![0u64; nb * 4];
    let mut phase = vec![0u64; nb * 4];
    let mut units = vec![0u64; nb * 256];
    let mut kind_counts = vec![0u64; nb * NCOUNT];
    let mut z_levels = vec![0u64; nb * 129];
    let (mut dump_q, mut dump_bits) = (Vec::new(), Vec::new());
    if spec.dump {
        dump_q = vec![0u64; nb * nq * 4];
        dump_bits = vec![0u64; nb * ncb * 4];
    }
    let (pv, pp, pk, pz) = (SP(val_err.as_mut_ptr()), SP(phase.as_mut_ptr()), SP(kind_counts.as_mut_ptr()), SP(z_levels.as_mut_ptr()));
    let pu = SP(units.as_mut_ptr());
    let (pdq, pdb) = (SP(dump_q.as_mut_ptr()), SP(dump_bits.as_mut_ptr()));
    let errs: Vec<String> = pool(spec.threads).install(|| {
        (0..nb)
            .into_par_iter()
            .with_max_len(1)
            .filter_map(|b| {
                let qin = &spec.qin[b * nqi * 4..(b + 1) * nqi * 4];
                let bin = &spec.bin[b * nbi * 4..(b + 1) * nbi * 4];
                let (sim, bad, ph) = match simulate_batch(p, &spec.seeds[b], spec.qin_ids, qin, spec.bin_ids, bin, &others, has_angles, spec.counting) {
                    Ok(x) => x,
                    Err(e) => return Some(format!("batch {}: {}", b, e)),
                };
                let kc = sim.counts[0].kinds();
                let zl = sim.counts[0].zlevels();
                // SAFETY: batch b writes only its own disjoint slices.
                unsafe {
                    for w in 0..4 {
                        *pv.p().add(b * 4 + w) = bad.0[w];
                        *pp.p().add(b * 4 + w) = ph.0[w];
                    }
                    for sh in 0..256 {
                        let f = (sim.phase.0[sh / 64] >> (sh % 64)) & 1;
                        let v = sim.angles[sh] ^ ((f as u128) << 127);
                        *pu.p().add(b * 256 + sh) = (((v >> 63) + 1) >> 1) as u64;
                    }
                    for k in 0..NCOUNT {
                        *pk.p().add(b * NCOUNT + k) = kc[k];
                    }
                    for k in 0..129 {
                        *pz.p().add(b * 129 + k) = zl[k];
                    }
                    if spec.dump {
                        for i in 0..nq {
                            for w in 0..4 {
                                *pdq.p().add((b * nq + i) * 4 + w) = sim.qb(i).0[w];
                            }
                        }
                        for i in 0..p.num_bits {
                            for w in 0..4 {
                                *pdb.p().add((b * ncb + i) * 4 + w) = sim.bt(i).0[w];
                            }
                        }
                    }
                }
                None
            })
            .collect()
    });
    if let Some(e) = errs.into_iter().next() {
        return Err(e);
    }
    Ok(BatchOut { val_err, phase, units, kind_counts, z_levels, dump_q, dump_bits })
}

/// One 256-shot batch: kickmix `Simulator(batch_size=256, seed)`, bit-sliced inputs written, circuit run.
/// Returns the sim, the value-check failures (an input qubit changed or another qubit nonzero) and the phase flips.
#[allow(clippy::too_many_arguments)]
pub fn simulate_batch<'p>(
    p: &'p Program,
    seed: &[u32; 8],
    qin_ids: &[u32],
    qin: &[u64],
    bin_ids: &[u32],
    bin: &[u64],
    others: &[usize],
    has_angles: bool,
    counting: bool,
) -> Result<(Sim, W, W), String> {
    let mut master = Mt64::from_seed_seq(seed);
    let keys = keys_from_sim_seed(&sim_seed_words(&mut master));
    let mut sim = Sim::new(p, keys);
    for (j, &q) in qin_ids.iter().enumerate() {
        *sim.qb_mut(q as usize) = Wn([qin[4 * j], qin[4 * j + 1], qin[4 * j + 2], qin[4 * j + 3]]);
    }
    for (j, &c) in bin_ids.iter().enumerate() {
        *sim.bt_mut(c as usize) = Wn([bin[4 * j], bin[4 * j + 1], bin[4 * j + 2], bin[4 * j + 3]]);
    }
    if counting {
        sim.run::<true>(p);
    } else {
        sim.run::<false>(p);
    }
    let mut bad = W::ZERO;
    for (j, &q) in qin_ids.iter().enumerate() {
        bad = bad | (sim.qb(q as usize) ^ Wn([qin[4 * j], qin[4 * j + 1], qin[4 * j + 2], qin[4 * j + 3]]));
    }
    for &i in others {
        bad = bad | sim.qb(i);
    }
    let ph = sim.phase_flipped(has_angles)?;
    Ok((sim, bad, ph))
}

// ---------------------------------------------------------------------------------------------------------------
// Addendum A: DQI batches sampled on the engine (exact `sample_shots`, see dqi_sample.rs) and simulated in the
// same Rayon task, so no inputs cross into Python.

use crate::dqi_sample::{DqiSampler, Pcg64};

pub struct DqiSpec<'a> {
    pub prog: &'a Program,
    pub sampler: &'a DqiSampler,
    /// Start state of `default_rng(rng_seed)` per batch.
    pub pcg: &'a [Pcg64],
    /// Simulator seed per batch.
    pub seeds: &'a [[u32; 8]],
    /// Syndrome qubit ids, row order of the sampler's `qin` (i * degree + k).
    pub qin_ids: &'a [u32],
    /// Compact bit ids for the sampler's `e_rows`, same order.
    pub bin_ids: &'a [u32],
    pub counting: bool,
    pub threads: usize,
}

pub struct DqiOut {
    /// [batch][4]
    pub val_err: Vec<u64>,
    /// [batch][4] simulated phase flips
    pub phase: Vec<u64>,
    /// [batch][4] expected phase flips from the sampler
    pub expected: Vec<u64>,
    /// [batch][NCOUNT]
    pub kind_counts: Vec<u64>,
    /// [batch][129]
    pub z_levels: Vec<u64>,
}

pub fn run_dqi(spec: &DqiSpec) -> Result<DqiOut, String> {
    crate::kernel::selected()?;
    let p = spec.prog;
    let nb = spec.seeds.len();
    spec.sampler.check()?;
    if spec.pcg.len() != nb {
        return Err("one generator state per batch is required".into());
    }
    if spec.qin_ids.len() != spec.sampler.n * spec.sampler.degree || spec.bin_ids.len() != spec.sampler.e_rows.len() {
        return Err("qin_ids / bin_ids do not match the sampler rows".into());
    }
    let nq = p.num_qubits;
    let mut is_in = vec![false; nq];
    for &q in spec.qin_ids {
        let q = q as usize;
        if q >= nq || is_in[q] {
            return Err("input qubit id out of range or repeated".into());
        }
        is_in[q] = true;
    }
    if spec.bin_ids.iter().any(|&b| b as usize >= p.num_bits) {
        return Err("input bit id out of range (must be compact ids)".into());
    }
    let others: Vec<usize> = (0..nq).filter(|&i| !is_in[i]).collect();
    let has_angles = !p.angles.is_empty();
    let mut val_err = vec![0u64; nb * 4];
    let mut phase = vec![0u64; nb * 4];
    let mut expected = vec![0u64; nb * 4];
    let mut kind_counts = vec![0u64; nb * NCOUNT];
    let mut z_levels = vec![0u64; nb * 129];
    let (pv, pp, pe, pk, pz) =
        (SP(val_err.as_mut_ptr()), SP(phase.as_mut_ptr()), SP(expected.as_mut_ptr()), SP(kind_counts.as_mut_ptr()), SP(z_levels.as_mut_ptr()));
    let errs: Vec<String> = pool(spec.threads).install(|| {
        (0..nb)
            .into_par_iter()
            .with_max_len(1)
            .filter_map(|b| {
                let mut rng = spec.pcg[b].clone();
                let smp = spec.sampler.sample(&mut rng, false);
                let (sim, bad, ph) = match simulate_batch(p, &spec.seeds[b], spec.qin_ids, &smp.qin, spec.bin_ids, &smp.bin, &others, has_angles, spec.counting) {
                    Ok(x) => x,
                    Err(e) => return Some(format!("batch {}: {}", b, e)),
                };
                let kc = sim.counts[0].kinds();
                let zl = sim.counts[0].zlevels();
                // SAFETY: batch b writes only its own disjoint slices.
                unsafe {
                    for w in 0..4 {
                        *pv.p().add(b * 4 + w) = bad.0[w];
                        *pp.p().add(b * 4 + w) = ph.0[w];
                        *pe.p().add(b * 4 + w) = smp.phase[w];
                    }
                    for k in 0..NCOUNT {
                        *pk.p().add(b * NCOUNT + k) = kc[k];
                    }
                    for k in 0..129 {
                        *pz.p().add(b * 129 + k) = zl[k];
                    }
                }
                None
            })
            .collect()
    });
    if let Some(e) = errs.into_iter().next() {
        return Err(e);
    }
    Ok(DqiOut { val_err, phase, expected, kind_counts, z_levels })
}

pub struct SampleOut {
    pub qin: Vec<u64>,
    pub bin: Vec<u64>,
    pub phase: Vec<u64>,
    pub syn: Vec<u16>,
    pub mbuc: Vec<u16>,
    /// End generator state per batch: [state_lo, state_hi, inc_lo, inc_hi, has_u32, u32_buf].
    pub end_state: Vec<u64>,
}

/// The sampler alone (for proofs and self-checks).
pub fn sample_dqi(sampler: &DqiSampler, pcg: &[Pcg64], threads: usize, keep_values: bool) -> Result<SampleOut, String> {
    crate::kernel::selected()?;
    sampler.check()?;
    let res: Vec<(crate::dqi_sample::DqiBatch, Pcg64)> = pool(threads).install(|| {
        pcg.par_iter()
            .with_max_len(1)
            .map(|st| {
                let mut r = st.clone();
                let b = sampler.sample(&mut r, keep_values);
                (b, r)
            })
            .collect()
    });
    let mut out = SampleOut { qin: vec![], bin: vec![], phase: vec![], syn: vec![], mbuc: vec![], end_state: vec![] };
    for (b, r) in res {
        out.qin.extend_from_slice(&b.qin);
        out.bin.extend_from_slice(&b.bin);
        out.phase.extend_from_slice(&b.phase);
        out.syn.extend_from_slice(&b.syn_vals);
        out.mbuc.extend_from_slice(&b.mbuc_vals);
        out.end_state.extend_from_slice(&[r.state as u64, (r.state >> 64) as u64, r.inc as u64, (r.inc >> 64) as u64, r.has_u32 as u64, r.u32_buf as u64]);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------------------------------------
// 0.5.0: a sequence of programs on one kickmix `Simulator(batch_size=shots, seed)`, as the dfthcsos (chemistry)
// verifier drives it: `do(circuit_0)`, checks, `clear_for_shot()`, the public registers written back, `do(circuit_1)`,
// ... Each 256-shot sim (and so its ChaCha8 stream and its operation counters) lives across all the steps.

pub struct SeqSpec<'a> {
    /// One program per step, in the order they are applied.
    pub progs: &'a [&'a Program],
    /// The simulator seed (8 LE u32 words of the 256-bit seed): kickmix `Simulator(batch_size=shots, seed=...)`.
    pub seed: &'a [u32; 8],
    /// kickmix batch_size; need not be a multiple of 256 (the last sim's extra lanes start at zero, are reset to
    /// zero between steps, and are not counted, as in kickmix).
    pub shots: usize,
    /// Qubits written before step 0 and carried across `clear_for_shot` between steps (the public registers).
    pub xreg: &'a [u32],
    /// Inputs [shot][xw] u64, xw = ceil(len(xreg) / 64): bit i of word i / 64 is qubit xreg[i].
    pub x0: &'a [u64],
    pub counting: bool,
    /// Also return the whole state after every step (for parity tests).
    pub dump: bool,
    pub threads: usize,
}

pub struct SeqOut {
    /// [step][shot][xw] the xreg qubits after each step.
    pub x_out: Vec<u64>,
    /// [step][shot] 1 if any qubit below the step's program `num_qubits` that is not in xreg is nonzero.
    pub dirty: Vec<u8>,
    /// [step][shot] phase flipped (kickmix `read_phase_flipped_across_shots`).
    pub phase: Vec<u8>,
    /// [step][NCOUNT] operation counts of each step, summed over the `shots` shots.
    pub kind_counts: Vec<u64>,
    /// [step][129]
    pub z_levels: Vec<u64>,
    /// dump: [step][sim][qubit][4] (qubits up to the largest program), [step][sim][compact bit][4] (the step's
    /// program's compact ids), and the qubit and bit counts used.
    pub dump_q: Vec<u64>,
    pub dump_bits: Vec<u64>,
    pub dump_nq: usize,
    pub dump_nb: usize,
}

/// Per-step kind counts and Z_POW level counts of one 256-shot sim.
type StepCounts = (Vec<[u64; NCOUNT]>, Vec<[u64; 129]>);

pub fn run_sequence(spec: &SeqSpec) -> Result<SeqOut, String> {
    crate::kernel::selected()?;
    let ns = spec.progs.len();
    if ns == 0 {
        return Err("at least one program is required".into());
    }
    let shots = spec.shots;
    if shots == 0 {
        return Err("shots must be positive".into());
    }
    let width = spec.xreg.len();
    let xw = width.div_ceil(64).max(1);
    if spec.x0.len() != shots * xw {
        return Err(format!("x0 has {} entries, expected shots * ceil(len(xreg) / 64) = {}", spec.x0.len(), shots * xw));
    }
    let mut nq = spec.progs.iter().map(|p| p.num_qubits).max().unwrap();
    let nb = spec.progs.iter().map(|p| p.num_bits).max().unwrap();
    let depth = spec.progs.iter().map(|p| p.max_stack_depth).max().unwrap();
    for &q in spec.xreg {
        nq = nq.max(q as usize + 1);
    }
    let mut seen = vec![false; nq];
    for &q in spec.xreg {
        if seen[q as usize] {
            return Err("xreg repeats a qubit".into());
        }
        seen[q as usize] = true;
    }
    let scratch: Vec<Vec<usize>> = spec.progs.iter().map(|p| (0..p.num_qubits).filter(|&i| !seen[i]).collect()).collect();
    let has_angles: Vec<bool> = spec.progs.iter().map(|p| !p.angles.is_empty()).collect();
    let nsims = shots.div_ceil(256);

    let mut x_out = vec![0u64; ns * shots * xw];
    let mut dirty = vec![0u8; ns * shots];
    let mut phase = vec![0u8; ns * shots];
    let (px, pd, pp) = (SP(x_out.as_mut_ptr()), SP(dirty.as_mut_ptr()), SP(phase.as_mut_ptr()));
    let nbd = nb.max(1);
    let (mut dump_q, mut dump_bits) = (Vec::new(), Vec::new());
    if spec.dump {
        dump_q = vec![0u64; ns * nsims * nq * 4];
        dump_bits = vec![0u64; ns * nsims * nbd * 4];
    }
    let (pdq, pdb) = (SP(dump_q.as_mut_ptr()), SP(dump_bits.as_mut_ptr()));
    let mut master = Mt64::from_seed_seq(spec.seed);
    let sim_seeds: Vec<[u32; 8]> = (0..nsims).map(|_| sim_seed_words(&mut master)).collect();

    let res: Vec<Result<StepCounts, String>> = pool(spec.threads).install(|| {
        (0..nsims)
            .into_par_iter()
            .with_max_len(1)
            .map(|u| {
                let base = u * 256;
                let nl = (shots - base).min(256);
                let mut lanes = W::ZERO;
                for s in 0..nl {
                    lanes.0[s / 64] |= 1u64 << (s % 64);
                }
                let partial = nl < 256;
                let mut sim = Sim::with_dims(nq, nb, depth, keys_from_sim_seed(&sim_seeds[u]));
                // Inputs: transpose 64 shots x 64 register bits at a time.
                let mut rows = [0u64; 64];
                for w in 0..4 {
                    for xb in 0..xw {
                        for r in 0..64 {
                            let s = base + 64 * w + r;
                            rows[r] = if s < shots { spec.x0[s * xw + xb] } else { 0 };
                        }
                        transpose64(&mut rows);
                        for i in 0..(width - 64 * xb).min(64) {
                            sim.qb_mut(spec.xreg[64 * xb + i] as usize).0[w] = rows[i];
                        }
                    }
                }
                let mut kc = Vec::with_capacity(ns);
                let mut zl = Vec::with_capacity(ns);
                let mut keep = vec![W::ZERO; width];
                for (st, prog) in spec.progs.iter().enumerate() {
                    if st > 0 {
                        // The verifier's reset: clear_for_shot (the RNG stream continues), then the public registers
                        // are written back for the real shots only.
                        for (i, &q) in spec.xreg.iter().enumerate() {
                            keep[i] = sim.qb(q as usize) & lanes;
                        }
                        sim.clear_for_shot();
                        for (i, &q) in spec.xreg.iter().enumerate() {
                            *sim.qb_mut(q as usize) = keep[i];
                        }
                    }
                    match (spec.counting, partial) {
                        (true, false) => sim.run::<true>(prog),
                        (true, true) => sim.run_lanes::<true>(prog, lanes),
                        (false, _) => sim.run::<false>(prog),
                    }
                    let ph = sim.phase_flipped_first(has_angles[st], nl).map_err(|e| format!("step {}: {}", st, e))?;
                    let mut dw = W::ZERO;
                    for &i in &scratch[st] {
                        dw = dw | sim.qb(i);
                    }
                    // SAFETY: sim u writes only shots base..base+nl of each step.
                    unsafe {
                        for w in 0..4 {
                            for xb in 0..xw {
                                rows = [0u64; 64];
                                for i in 0..(width - 64 * xb).min(64) {
                                    rows[i] = sim.qb(spec.xreg[64 * xb + i] as usize).0[w];
                                }
                                transpose64(&mut rows);
                                for r in 0..64 {
                                    let s = base + 64 * w + r;
                                    if s < shots {
                                        *px.p().add((st * shots + s) * xw + xb) = rows[r];
                                    }
                                }
                            }
                        }
                        for l in 0..nl {
                            let s = base + l;
                            *pd.p().add(st * shots + s) = ((dw.0[l / 64] >> (l % 64)) & 1) as u8;
                            *pp.p().add(st * shots + s) = ((ph.0[l / 64] >> (l % 64)) & 1) as u8;
                        }
                        if spec.dump {
                            let ub = st * nsims + u;
                            for i in 0..nq {
                                for w in 0..4 {
                                    *pdq.p().add((ub * nq + i) * 4 + w) = sim.qb(i).0[w];
                                }
                            }
                            for i in 0..nbd.min(sim.num_bits()) {
                                for w in 0..4 {
                                    *pdb.p().add((ub * nbd + i) * 4 + w) = sim.bt(i).0[w];
                                }
                            }
                        }
                    }
                    kc.push(sim.counts[0].kinds());
                    zl.push(sim.counts[0].zlevels());
                    sim.counts[0] = Counts::new();
                }
                Ok((kc, zl))
            })
            .collect()
    });
    let mut kind_counts = vec![0u64; ns * NCOUNT];
    let mut z_levels = vec![0u64; ns * 129];
    for r in res {
        let (kc, zl) = r?;
        for st in 0..ns {
            for k in 0..NCOUNT {
                kind_counts[st * NCOUNT + k] += kc[st][k];
            }
            for k in 0..129 {
                z_levels[st * 129 + k] += zl[st][k];
            }
        }
    }
    Ok(SeqOut { x_out, dirty, phase, kind_counts, z_levels, dump_q, dump_bits, dump_nq: nq, dump_nb: nbd })
}
