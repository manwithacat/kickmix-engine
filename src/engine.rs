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

    let mut x_out = vec![0u64; total];
    let mut dirty = vec![0u8; total];
    let mut units = vec![0u64; total];
    let nunits_batches = nch * bb * s;
    let mut dump = Dump::default();
    if spec.dump {
        dump.q = vec![0; nunits_batches * nq * 4];
        dump.bits = vec![0; nunits_batches * nb * 4];
        dump.phase = vec![0; nunits_batches * 4];
        dump.vphase = vec![0; nunits_batches * 256 * 2];
    }
    let (px, pd, pu) = (SP(x_out.as_mut_ptr()), SP(dirty.as_mut_ptr()), SP(units.as_mut_ptr()));
    let (dq, dbits, dph, dv) = (SP(dump.q.as_mut_ptr()), SP(dump.bits.as_mut_ptr()), SP(dump.phase.as_mut_ptr()), SP(dump.vphase.as_mut_ptr()));

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(spec.threads.max(1))
        .build()
        .map_err(|e| e.to_string())?;
    let counts: Vec<Counts> = pool.install(|| {
        // Per-chunk: the master stream hands out 4 draws per sim, in order.
        let sim_seeds: Vec<Vec<[u32; 8]>> = spec
            .seeds
            .par_iter()
            .map(|sd| {
                let mut master = Mt64::from_seed_seq(sd);
                (0..s).map(|_| sim_seed_words(&mut master)).collect()
            })
            .collect();
        (0..nch * s)
            .into_par_iter()
            .with_min_len(1)
            .map(|u| {
                let (chunk, k) = (u / s, u % s);
                let keys = keys_from_sim_seed(&sim_seeds[chunk][k]);
                let mut sim = Sim::new(spec.prog, keys);
                let width = spec.xreg.len();
                let mut rows = [0u64; 64];
                for b in 0..bb {
                    let off = (chunk * bb + b) * shots_per_batch + k * 256;
                    let ub = (chunk * bb + b) * s + k;
                    sim.clear_for_shot();
                    for w in 0..4 {
                        rows.copy_from_slice(&spec.xs[off + 64 * w..off + 64 * w + 64]);
                        transpose64(&mut rows);
                        for i in 0..width {
                            sim.q[spec.xreg[i] as usize].0[w] = rows[i];
                        }
                    }
                    if spec.counting {
                        sim.run::<true>(spec.prog);
                    } else {
                        sim.run::<false>(spec.prog);
                    }
                    let mut dw = W::ZERO;
                    for &i in &scratch {
                        dw = dw | sim.q[i];
                    }
                    // SAFETY: each (chunk, batch, k) writes a disjoint range of the outputs.
                    unsafe {
                        for w in 0..4 {
                            rows = [0u64; 64];
                            for i in 0..width {
                                rows[i] = sim.q[spec.xreg[i] as usize].0[w];
                            }
                            transpose64(&mut rows);
                            for r in 0..64 {
                                *px.p().add(off + 64 * w + r) = rows[r];
                            }
                        }
                        for sh in 0..256 {
                            let ph = (sim.phase.0[sh / 64] >> (sh % 64)) & 1;
                            let v = sim.angles[sh] ^ ((ph as u128) << 127);
                            *pu.p().add(off + sh) = (((v >> 63) + 1) >> 1) as u64;
                            *pd.p().add(off + sh) = ((dw.0[sh / 64] >> (sh % 64)) & 1) as u8;
                            if spec.dump {
                                *dv.p().add(ub * 512 + sh * 2) = v as u64;
                                *dv.p().add(ub * 512 + sh * 2 + 1) = (v >> 64) as u64;
                            }
                        }
                        if spec.dump {
                            for i in 0..nq {
                                for w in 0..4 {
                                    *dq.p().add((ub * nq + i) * 4 + w) = sim.q[i].0[w];
                                }
                            }
                            for i in 0..spec.prog.num_bits {
                                for w in 0..4 {
                                    *dbits.p().add((ub * nb + i) * 4 + w) = sim.bits[i].0[w];
                                }
                            }
                            for w in 0..4 {
                                *dph.p().add(ub * 4 + w) = sim.phase.0[w];
                            }
                        }
                    }
                }
                sim.counts
            })
            .collect()
    });
    let mut kind_counts = vec![0u64; nch * NCOUNT];
    let mut z_levels = vec![0u64; nch * 129];
    for (u, c) in counts.iter().enumerate() {
        let chunk = u / s;
        for (i, v) in c.kinds().iter().enumerate() {
            kind_counts[chunk * NCOUNT + i] += v;
        }
        for (i, v) in c.zlevels().iter().enumerate() {
            z_levels[chunk * 129 + i] += v;
        }
    }
    Ok(Out { x_out, dirty, units, kind_counts, z_levels, dump })
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
    let mut kind_counts = vec![0u64; nb * NCOUNT];
    let mut z_levels = vec![0u64; nb * 129];
    let (mut dump_q, mut dump_bits) = (Vec::new(), Vec::new());
    if spec.dump {
        dump_q = vec![0u64; nb * nq * 4];
        dump_bits = vec![0u64; nb * ncb * 4];
    }
    let (pv, pp, pk, pz) = (SP(val_err.as_mut_ptr()), SP(phase.as_mut_ptr()), SP(kind_counts.as_mut_ptr()), SP(z_levels.as_mut_ptr()));
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
                let kc = sim.counts.kinds();
                let zl = sim.counts.zlevels();
                // SAFETY: batch b writes only its own disjoint slices.
                unsafe {
                    for w in 0..4 {
                        *pv.p().add(b * 4 + w) = bad.0[w];
                        *pp.p().add(b * 4 + w) = ph.0[w];
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
                                *pdq.p().add((b * nq + i) * 4 + w) = sim.q[i].0[w];
                            }
                        }
                        for i in 0..p.num_bits {
                            for w in 0..4 {
                                *pdb.p().add((b * ncb + i) * 4 + w) = sim.bits[i].0[w];
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
    Ok(BatchOut { val_err, phase, kind_counts, z_levels, dump_q, dump_bits })
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
        sim.q[q as usize] = W([qin[4 * j], qin[4 * j + 1], qin[4 * j + 2], qin[4 * j + 3]]);
    }
    for (j, &c) in bin_ids.iter().enumerate() {
        sim.bits[c as usize] = W([bin[4 * j], bin[4 * j + 1], bin[4 * j + 2], bin[4 * j + 3]]);
    }
    if counting {
        sim.run::<true>(p);
    } else {
        sim.run::<false>(p);
    }
    let mut bad = W::ZERO;
    for (j, &q) in qin_ids.iter().enumerate() {
        bad = bad | (sim.q[q as usize] ^ W([qin[4 * j], qin[4 * j + 1], qin[4 * j + 2], qin[4 * j + 3]]));
    }
    for &i in others {
        bad = bad | sim.q[i];
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
                let kc = sim.counts.kinds();
                let zl = sim.counts.zlevels();
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
