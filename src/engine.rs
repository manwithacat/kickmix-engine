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
    /// [chunk][17]
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
    let mut kind_counts = vec![0u64; nch * 17];
    let mut z_levels = vec![0u64; nch * 129];
    for (u, c) in counts.iter().enumerate() {
        let chunk = u / s;
        for (i, v) in c.kinds().iter().enumerate() {
            kind_counts[chunk * 17 + i] += v;
        }
        for (i, v) in c.zlevels().iter().enumerate() {
            z_levels[chunk * 129 + i] += v;
        }
    }
    Ok(Out { x_out, dirty, units, kind_counts, z_levels, dump })
}
