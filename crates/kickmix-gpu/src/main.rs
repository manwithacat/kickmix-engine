//! kickmix-gpu: feasibility prototype of shot-parallel GPU simulation (wgpu / Metal) of kickmix-engine programs,
//! with exact per-shot comparison against the CPU engine compiled into the same binary (the engine's own modules,
//! included by path, unchanged). Written by an AI model (Claude) under the owner's direction; see NOTICE.
//!
//!   kickmix-gpu small [--lanes 128|32] [--seg WORDS] [--batches N]        synthetic all-op program, CPU vs GPU
//!   kickmix-gpu dqi --tasks TASKS.json [options]                           one verifier draw of a DQI circuit
#![allow(dead_code)]
#[path = "../../../src/program.rs"]
pub mod program;
#[path = "../../../src/rng.rs"]
pub mod rng;
#[path = "../../../src/kernel.rs"]
pub mod kernel;
#[path = "../../../src/sim.rs"]
pub mod sim;
#[path = "../../../src/dqi_sample.rs"]
pub mod dqi_sample;
#[path = "../../../src/engine.rs"]
pub mod engine;
mod gpu;

use dqi_sample::{DqiSampler, Pcg64};
use gpu::{BatchInput, Gpu, Lanes};
use program::*;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use sim::W;
use std::collections::HashMap;
use std::time::Instant;

struct Args {
    cmd: String,
    kv: HashMap<String, String>,
    flags: Vec<String>,
}
fn parse_args() -> Args {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().unwrap_or_default();
    let (mut kv, mut flags) = (HashMap::new(), Vec::new());
    let rest: Vec<String> = it.collect();
    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].trim_start_matches("--").to_string();
        if i + 1 < rest.len() && !rest[i + 1].starts_with("--") {
            kv.insert(a, rest[i + 1].clone());
            i += 2;
        } else {
            flags.push(a);
            i += 1;
        }
    }
    Args { cmd, kv, flags }
}
impl Args {
    fn get<T: std::str::FromStr>(&self, k: &str, d: T) -> T {
        self.kv.get(k).and_then(|v| v.parse().ok()).unwrap_or(d)
    }
    fn has(&self, k: &str) -> bool {
        self.flags.iter().any(|f| f == k)
    }
    fn lanes(&self) -> Lanes {
        if self.get("lanes", 128usize) == 32 {
            Lanes::L32
        } else {
            Lanes::L128
        }
    }
}

/// Python `float.hex()` -> f64, exactly.
fn parse_hex_float(s: &str) -> Result<f64, String> {
    let (neg, s) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let s = s.strip_prefix("0x").ok_or("hex float without 0x")?;
    let (mant, exp) = s.split_once('p').ok_or("hex float without p")?;
    let exp: i32 = exp.parse().map_err(|_| "bad exponent")?;
    let (ip, fp) = mant.split_once('.').unwrap_or((mant, ""));
    let mut m: u64 = u64::from_str_radix(ip, 16).map_err(|_| "bad mantissa")?;
    let mut fbits = 0i32;
    for ch in fp.chars() {
        let d = ch.to_digit(16).ok_or("bad hex digit")?;
        m = m.checked_mul(16).and_then(|x| x.checked_add(d as u64)).ok_or("mantissa overflow")?;
        fbits += 4;
    }
    let v = (m as f64) * 2f64.powi(exp - fbits);
    Ok(if neg { -v } else { v })
}

#[derive(Deserialize)]
struct SamplerJ {
    m: usize,
    n: usize,
    degree: usize,
    cdf: Vec<String>,
    log: Vec<u32>,
    antilog: Vec<u32>,
    e_rows: Vec<u32>,
}
#[derive(Deserialize)]
struct BatchJ {
    start: u64,
    pcg: Vec<u64>,
    sim_seed: Vec<u32>,
}
#[derive(Deserialize)]
struct Tasks {
    circuit: String,
    circuit_sha256: String,
    salt: String,
    program_path: String,
    program: serde_json::Value,
    spec: serde_json::Value,
    sampler: SamplerJ,
    qin_ids: Vec<u32>,
    bin_ids: Vec<u32>,
    batches: Vec<BatchJ>,
}

/// CPU reference of one batch: final qubits, phase, value-check failures, counts, optionally bits.
struct CpuRef {
    q: Vec<W>,
    bits: Vec<W>,
    phase: W,
    bad: W,
    counts: [u64; NCOUNT],
}

fn cpu_reference(p: &Program, seeds: &[[u32; 8]], qin_ids: &[u32], qin: &[Vec<u64>], bin_ids: &[u32], bin: &[Vec<u64>], keep_bits: usize, threads: usize) -> Vec<CpuRef> {
    let nq = p.num_qubits;
    let mut is_in = vec![false; nq];
    for &q in qin_ids {
        is_in[q as usize] = true;
    }
    let others: Vec<usize> = (0..nq).filter(|&i| !is_in[i]).collect();
    let has_angles = !p.angles.is_empty();
    engine::pool(threads).install(|| {
        (0..seeds.len())
            .into_par_iter()
            .with_max_len(1)
            .map(|b| {
                let (sim, bad, _ph) = engine::simulate_batch(p, &seeds[b], qin_ids, &qin[b], bin_ids, &bin[b], &others, has_angles, true).expect("cpu batch");
                CpuRef {
                    q: sim.q.clone(),
                    bits: if b < keep_bits { sim.bits.clone() } else { Vec::new() },
                    phase: sim.phase,
                    bad,
                    counts: sim.counts.kinds(),
                }
            })
            .collect()
    })
}

#[derive(Default, serde::Serialize)]
struct Mismatch {
    q_shots: u64,
    q_words: u64,
    phase_shots: u64,
    val_shots: u64,
    perr_shots: u64,
    bits_words: u64,
    bits_batches_compared: usize,
    count_batches: u64,
    first: Option<String>,
}

fn popcnt4(w: &[u64]) -> u64 {
    w.iter().map(|x| x.count_ones() as u64).sum()
}

/// Compare the GPU output of a group of batches (offset `b0` in the references) with the CPU references.
fn compare(p: &Program, g: &gpu::GpuOut, refs: &[CpuRef], expected: &[[u64; 4]], qin_ids: &[u32], qin: &[Vec<u64>], mm: &mut Mismatch, stats: &mut Stats) {
    let nq = p.num_qubits.max(1);
    let nbits = p.num_bits.max(1);
    let mut is_in = vec![false; nq];
    for &q in qin_ids {
        is_in[q as usize] = true;
    }
    for (b, r) in refs.iter().enumerate() {
        let mut qdiff = [0u64; 4];
        for i in 0..nq {
            let gw = &g.q[(b * nq + i) * 4..(b * nq + i) * 4 + 4];
            for w in 0..4 {
                let d = gw[w] ^ r.q[i].0[w];
                if d != 0 {
                    mm.q_words += 1;
                    if mm.first.is_none() {
                        mm.first = Some(format!("batch {b} qubit {i} word {w}: gpu {:016x} cpu {:016x}", gw[w], r.q[i].0[w]));
                    }
                }
                qdiff[w] |= d;
            }
        }
        mm.q_shots += popcnt4(&qdiff);
        let gph = &g.phase[b * 4..b * 4 + 4];
        let mut pdiff = [0u64; 4];
        for w in 0..4 {
            pdiff[w] = gph[w] ^ r.phase.0[w];
        }
        mm.phase_shots += popcnt4(&pdiff);
        // Value check from the GPU state, as the engine computes it from the CPU state.
        let mut gbad = [0u64; 4];
        for (j, &qi) in qin_ids.iter().enumerate() {
            let gw = &g.q[(b * nq + qi as usize) * 4..(b * nq + qi as usize) * 4 + 4];
            for w in 0..4 {
                gbad[w] |= gw[w] ^ qin[b][4 * j + w];
            }
        }
        for i in 0..nq {
            if !is_in[i] {
                let gw = &g.q[(b * nq + i) * 4..(b * nq + i) * 4 + 4];
                for w in 0..4 {
                    gbad[w] |= gw[w];
                }
            }
        }
        let mut vd = [0u64; 4];
        let mut gperr = [0u64; 4];
        let mut cperr = [0u64; 4];
        for w in 0..4 {
            vd[w] = gbad[w] ^ r.bad.0[w];
            gperr[w] = (gph[w] ^ expected[b][w]) & !gbad[w];
            cperr[w] = (r.phase.0[w] ^ expected[b][w]) & !r.bad.0[w];
        }
        mm.val_shots += popcnt4(&vd);
        let mut pe = [0u64; 4];
        for w in 0..4 {
            pe[w] = gperr[w] ^ cperr[w];
        }
        mm.perr_shots += popcnt4(&pe);
        stats.val_errors += popcnt4(&r.bad.0);
        stats.phase_errors += popcnt4(&cperr);
        let gc = &g.kind_counts[b * NCOUNT..(b + 1) * NCOUNT];
        if gc != &r.counts[..] {
            mm.count_batches += 1;
            if mm.first.is_none() {
                mm.first = Some(format!("batch {b} counts: gpu {:?} cpu {:?}", gc, r.counts));
            }
        }
        for k in [K_CCX, K_CCZ, K_CSWAP] {
            stats.toffolis += r.counts[k as usize];
        }
        if !r.bits.is_empty() && !g.bits.is_empty() {
            mm.bits_batches_compared += 1;
            for i in 0..nbits {
                let gw = &g.bits[(b * nbits + i) * 4..(b * nbits + i) * 4 + 4];
                for w in 0..4 {
                    if gw[w] != r.bits[i].0[w] {
                        mm.bits_words += 1;
                        if mm.first.is_none() {
                            mm.first = Some(format!("batch {b} bit {i} word {w}: gpu {:016x} cpu {:016x}", gw[w], r.bits[i].0[w]));
                        }
                    }
                }
            }
        }
        stats.shots += 256;
    }
}

#[derive(Default, serde::Serialize)]
struct Stats {
    shots: u64,
    val_errors: u64,
    phase_errors: u64,
    toffolis: u64,
}

fn mismatch_total(m: &Mismatch) -> u64 {
    m.q_words + m.phase_shots + m.val_shots + m.perr_shots + m.bits_words + m.count_batches
}

// ---------------------------------------------------------------------------------------------------------------

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// A synthetic program with every supported op kind, IF variants, nested conditions, CX runs and INFO_TAGs.
fn synthetic_program(seed: u64, nops: usize, nq: usize, nb: usize) -> String {
    let mut r = XorShift(seed | 1);
    let mut s = String::new();
    let mut depth = 0usize;
    let q = |r: &mut XorShift| format!("q{}", r.below(nq as u64));
    let b = |r: &mut XorShift| format!("b{}", r.below(nb as u64));
    for _ in 0..nops {
        let k = r.below(100);
        let cond = if r.below(5) == 0 { format!(" if {}", b(&mut r)) } else { String::new() };
        let line = match k {
            0..=39 => {
                // a CX run
                let n = 1 + r.below(12);
                let mut t = String::new();
                for _ in 0..n {
                    let (a, c) = (r.below(nq as u64), r.below(nq as u64));
                    if a != c {
                        t.push_str(&format!("CX q{a} q{c}\n"));
                    }
                }
                s.push_str(&t);
                continue;
            }
            40..=47 => {
                let (a, c) = (r.below(nq as u64), r.below(nq as u64));
                if a == c {
                    continue;
                }
                format!("CX q{a} q{c}{cond}")
            }
            48..=53 => {
                let (a, c, d) = (r.below(nq as u64), r.below(nq as u64), r.below(nq as u64));
                if a == c || a == d || c == d {
                    continue;
                }
                let op = ["CCX", "CCZ", "CSWAP"][r.below(3) as usize];
                format!("{op} q{a} q{c} q{d}{cond}")
            }
            54..=59 => {
                let (a, c) = (r.below(nq as u64), r.below(nq as u64));
                if a == c {
                    continue;
                }
                let op = ["CZ", "SWAP"][r.below(2) as usize];
                format!("{op} q{a} q{c}{cond}")
            }
            60..=67 => format!("{} {}{cond}", ["X", "Z"][r.below(2) as usize], q(&mut r)),
            68..=69 => format!("NEG{cond}"),
            70..=77 => format!("{} {}{cond}", ["BIT_STORE0", "BIT_STORE1", "BIT_INVERT"][r.below(3) as usize], b(&mut r)),
            78..=85 => format!("HMR {} {}{cond}", q(&mut r), b(&mut r)),
            86..=89 => format!("R {}{cond}", q(&mut r)),
            90..=93 => {
                if depth < 2 {
                    depth += 1;
                    format!("PUSH_CONDITION if {}", b(&mut r))
                } else {
                    depth -= 1;
                    "POP_CONDITION".to_string()
                }
            }
            94..=96 => {
                if depth > 0 {
                    depth -= 1;
                    "POP_CONDITION".to_string()
                } else {
                    continue;
                }
            }
            _ => "INFO_TAG".to_string(),
        };
        s.push_str(&line);
        s.push('\n');
    }
    while depth > 0 {
        s.push_str("POP_CONDITION\n");
        depth -= 1;
    }
    s
}

fn cmd_small(a: &Args) -> Result<(), String> {
    let nq = a.get("nq", 40usize);
    let nb = a.get("nb", 24usize);
    let nops = a.get("ops", 4000usize);
    let nbatch = a.get("batches", 3usize);
    let seg = a.get("seg", 997usize);
    let threads = a.get("threads", 2usize);
    let text = synthetic_program(a.get("seed", 12345u64), nops, nq, nb);
    let p = Program::from_text(&text, &[], nq)?;
    eprintln!(
        "synthetic program: {} ops, {} words, {} qubits, {} bits, {} rand, depth {}, {} cx runs",
        p.n_source_ops, p.code.len(), p.num_qubits, p.num_bits, p.nrand, p.max_stack_depth, p.n_cx_runs
    );
    let mut r = XorShift(a.get("seed", 12345u64) ^ 0x9e37);
    let qin_ids: Vec<u32> = (0..(nq as u32 / 2)).collect();
    let bin_ids: Vec<u32> = (0..(p.num_bits as u32).min(8)).collect();
    let mut seeds = Vec::new();
    let (mut qin, mut bin) = (Vec::new(), Vec::new());
    for _ in 0..nbatch {
        let mut sd = [0u32; 8];
        for x in sd.iter_mut() {
            *x = r.next() as u32;
        }
        seeds.push(sd);
        qin.push((0..qin_ids.len() * 4).map(|_| r.next()).collect::<Vec<u64>>());
        bin.push((0..bin_ids.len() * 4).map(|_| r.next()).collect::<Vec<u64>>());
    }
    let refs = cpu_reference(&p, &seeds, &qin_ids, &qin, &bin_ids, &bin, nbatch, threads);
    let expected = vec![[0u64; 4]; nbatch];
    let mut all_ok = true;
    for lanes in [Lanes::L128, Lanes::L32] {
        let kmax = a.get("bundle", 0usize);
        let g = Gpu::new(lanes, a.get("wg", 32u32), kmax.max(1))?;
        eprintln!("gpu: {} ({}), init {:.3}s, lanes {}, bundle {}", g.adapter_name, g.backend, g.t_init, lanes.shots(), kmax);
        let (code, bst) = if kmax > 0 { gpu::bundle_stream(&p.code, kmax) } else { (p.code.clone(), Default::default()) };
        if kmax > 0 {
            eprintln!("  bundles: {}", serde_json::to_string(&bst).unwrap());
        }
        let (code_buf, _) = g.upload_code(&code);
        for &sw in &[seg, code.len() + 1] {
            let inputs: Vec<BatchInput> = (0..nbatch).map(|b| BatchInput { seed: &seeds[b], qin: &qin[b], bin: &bin[b] }).collect();
            let out = g.run_batches(&p, &code, &code_buf, &inputs, &qin_ids, &bin_ids, sw, true, true)?;
            let (mut mm, mut st) = (Mismatch::default(), Stats::default());
            compare(&p, &out, &refs, &expected, &qin_ids, &qin, &mut mm, &mut st);
            let ok = mismatch_total(&mm) == 0 && out.errors == 0;
            all_ok &= ok;
            eprintln!(
                "  seg {sw}: {} segments, rpos {}, errors {:#x}, kernel {:.4}s -> {} ({})",
                out.segments,
                out.rpos,
                out.errors,
                out.t_kernel,
                if ok { "MATCH" } else { "MISMATCH" },
                serde_json::to_string(&mm).unwrap()
            );
        }
        // Negative control: one seed word changed on batch 0 must change HMR/R outcomes.
        let mut bad_seed = seeds[0];
        bad_seed[3] ^= 1;
        let mut seeds2 = seeds.clone();
        seeds2[0] = bad_seed;
        let inputs: Vec<BatchInput> = (0..nbatch).map(|b| BatchInput { seed: &seeds2[b], qin: &qin[b], bin: &bin[b] }).collect();
        let out = g.run_batches(&p, &code, &code_buf, &inputs, &qin_ids, &bin_ids, seg, true, false)?;
        let (mut mm, mut st) = (Mismatch::default(), Stats::default());
        compare(&p, &out, &refs, &expected, &qin_ids, &qin, &mut mm, &mut st);
        let ctrl_ok = mismatch_total(&mm) > 0;
        all_ok &= ctrl_ok;
        eprintln!("  control (wrong seed): {} ({})", if ctrl_ok { "mismatches as expected" } else { "NO MISMATCH: test is blind" }, serde_json::to_string(&mm).unwrap());
    }
    println!("{}", if all_ok { "SMALL: PASS" } else { "SMALL: FAIL" });
    if all_ok {
        Ok(())
    } else {
        Err("small test failed".into())
    }
}

fn cmd_dqi(a: &Args) -> Result<(), String> {
    let tasks_path = a.kv.get("tasks").ok_or("--tasks TASKS.json required")?;
    let tj: Tasks = serde_json::from_str(&std::fs::read_to_string(tasks_path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
    let mut tj = tj;
    let program_path = a.kv.get("program").cloned().unwrap_or_else(|| tj.program_path.clone());
    let nbatch = a.get("batches", tj.batches.len()).min(tj.batches.len());
    // --repeat R tiles the draw R times (same seeds), to measure how the kernel scales with the thread count.
    let repeat = a.get("repeat", 1usize).max(1);
    if repeat > 1 {
        let base: Vec<BatchJ> = tj.batches[..nbatch].iter().map(|b| BatchJ { start: b.start, pcg: b.pcg.clone(), sim_seed: b.sim_seed.clone() }).collect();
        tj.batches = (0..repeat).flat_map(|_| base.iter().map(|b| BatchJ { start: b.start, pcg: b.pcg.clone(), sim_seed: b.sim_seed.clone() })).collect();
    }
    let nbatch = nbatch * repeat;
    let group = a.get("group", nbatch).max(1);
    let seg = a.get("seg", 8usize << 20);
    let threads = a.get("threads", 4usize);
    let keep_bits = a.get("compare-bits", 0usize).min(nbatch);
    let lanes = a.lanes();
    let wg = a.get("wg", 32u32);
    let sync_each = a.has("sync-each");
    let t0 = Instant::now();
    let p = Program::load(&program_path)?;
    let t_load = t0.elapsed().as_secs_f64();
    eprintln!("program loaded in {t_load:.2}s: {} ops, {} words, {} qubits, {} bits, {} rand", p.n_source_ops, p.code.len(), p.num_qubits, p.num_bits, p.nrand);
    let cdf: Vec<f64> = tj.sampler.cdf.iter().map(|s| parse_hex_float(s)).collect::<Result<_, _>>()?;
    let sampler = DqiSampler { m: tj.sampler.m, n: tj.sampler.n, degree: tj.sampler.degree, cdf, log: tj.sampler.log.clone(), antilog: tj.sampler.antilog.clone(), e_rows: tj.sampler.e_rows.clone() };
    sampler.check()?;
    let pcg: Vec<Pcg64> = tj.batches[..nbatch]
        .iter()
        .map(|b| Pcg64 { state: (b.pcg[0] as u128) | ((b.pcg[1] as u128) << 64), inc: (b.pcg[2] as u128) | ((b.pcg[3] as u128) << 64), has_u32: b.pcg[4] != 0, u32_buf: b.pcg[5] as u32 })
        .collect();
    let seeds: Vec<[u32; 8]> = tj.batches[..nbatch]
        .iter()
        .map(|b| {
            let mut s = [0u32; 8];
            s.copy_from_slice(&b.sim_seed[..8]);
            s
        })
        .collect();
    // Sampling on the CPU (engine sampler, exact `sample_shots`).
    let ts = Instant::now();
    let samples: Vec<dqi_sample::DqiBatch> = engine::pool(threads).install(|| pcg.par_iter().with_max_len(1).map(|st| sampler.sample(&mut st.clone(), false)).collect());
    let t_sample = ts.elapsed().as_secs_f64();
    let qin: Vec<Vec<u64>> = samples.iter().map(|s| s.qin.clone()).collect();
    let bin: Vec<Vec<u64>> = samples.iter().map(|s| s.bin.clone()).collect();
    let expected: Vec<[u64; 4]> = samples.iter().map(|s| s.phase).collect();
    drop(samples);
    eprintln!("sampled {nbatch} batches in {t_sample:.2}s ({threads} threads)");
    // CPU reference with full state.
    let tc = Instant::now();
    let refs = cpu_reference(&p, &seeds, &tj.qin_ids, &qin, &tj.bin_ids, &bin, keep_bits, threads);
    let t_cpu_ref = tc.elapsed().as_secs_f64();
    eprintln!("cpu reference ({threads} threads, state kept): {t_cpu_ref:.2}s");
    // CPU timing of the engine's fused DQI path (sampling + simulation), as `verify dqi` runs it.
    let mut t_cpu_run_dqi = None;
    let mut cpu_run_dqi_summary = json!(null);
    if a.has("cpu-timing") {
        let spec = engine::DqiSpec { prog: &p, sampler: &sampler, pcg: &pcg, seeds: &seeds, qin_ids: &tj.qin_ids, bin_ids: &tj.bin_ids, counting: true, threads };
        let tr = Instant::now();
        let out = engine::run_dqi(&spec)?;
        let t = tr.elapsed().as_secs_f64();
        let val: u64 = popcnt4(&out.val_err);
        let mut tof = 0u64;
        for b in 0..nbatch {
            for k in [K_CCX, K_CCZ, K_CSWAP] {
                tof += out.kind_counts[b * NCOUNT + k as usize];
            }
        }
        cpu_run_dqi_summary = json!({"seconds": t, "value_error_shots": val, "mean_toffolis": tof as f64 / (256.0 * nbatch as f64)});
        t_cpu_run_dqi = Some(t);
        eprintln!("cpu engine run_dqi ({threads} threads): {t:.2}s");
    }
    // GPU.
    let kmax = a.get("bundle", 0usize);
    let g = Gpu::new(lanes, wg, kmax.max(1))?;
    eprintln!("gpu: {} ({}), init {:.3}s, lanes {}, max buffer {} MB, max binding {} MB, bundle {}", g.adapter_name, g.backend, g.t_init, lanes.shots(), g.max_buffer >> 20, g.max_binding >> 20, kmax);
    let tb = Instant::now();
    let (code, bst) = if kmax > 0 { gpu::bundle_stream(&p.code, kmax) } else { (p.code.clone(), Default::default()) };
    let t_bundle = tb.elapsed().as_secs_f64();
    if kmax > 0 {
        eprintln!("bundled stream in {t_bundle:.2}s: {} words, {}", code.len(), serde_json::to_string(&bst).unwrap());
    }
    let (code_buf, t_code) = g.upload_code(&code);
    eprintln!("code uploaded ({} MB) in {t_code:.3}s", code.len() * 4 >> 20);
    let (mut mm, mut st) = (Mismatch::default(), Stats::default());
    let (mut t_upload, mut t_kernel, mut t_readback) = (0.0, 0.0, 0.0);
    let mut seg_times = Vec::new();
    let mut gpu_errors = 0u32;
    let mut segments = 0usize;
    let mut passes = 0usize;
    let mut gpu_toffolis = 0u64;
    let mut gpu_val = 0u64;
    let mut b0 = 0usize;
    while b0 < nbatch {
        let b1 = (b0 + group).min(nbatch);
        let inputs: Vec<BatchInput> = (b0..b1).map(|b| BatchInput { seed: &seeds[b], qin: &qin[b], bin: &bin[b] }).collect();
        let out = g.run_batches(&p, &code, &code_buf, &inputs, &tj.qin_ids, &tj.bin_ids, seg, keep_bits > b0, sync_each)?;
        t_upload += out.t_upload;
        t_kernel += out.t_kernel;
        t_readback += out.t_readback;
        seg_times.extend_from_slice(&out.seg_times);
        gpu_errors |= out.errors;
        segments = out.segments;
        passes += 1;
        for b in 0..(b1 - b0) {
            for k in [K_CCX, K_CCZ, K_CSWAP] {
                gpu_toffolis += out.kind_counts[b * NCOUNT + k as usize];
            }
        }
        let n_before = mm.val_shots;
        compare(&p, &out, &refs[b0..b1], &expected[b0..b1], &tj.qin_ids, &qin[b0..b1], &mut mm, &mut st);
        let _ = n_before;
        gpu_val += 0;
        eprintln!(
            "  batches {b0}..{b1}: {} segments, upload {:.2}s, kernel {:.2}s, readback {:.2}s, rpos {}, errors {:#x}, mismatches so far {}",
            out.segments, out.t_upload, out.t_kernel, out.t_readback, out.rpos, out.errors, mismatch_total(&mm)
        );
        b0 = b1;
    }
    let _ = gpu_val;
    // Negative control: batch 0 with one simulator-seed word changed.
    let mut control = json!(null);
    if a.has("control") && nbatch > 0 {
        let mut sd = seeds[0];
        sd[3] ^= 1;
        let inputs = vec![BatchInput { seed: &sd, qin: &qin[0], bin: &bin[0] }];
        let out = g.run_batches(&p, &code, &code_buf, &inputs, &tj.qin_ids, &tj.bin_ids, seg, false, false)?;
        let (mut cm, mut cs) = (Mismatch::default(), Stats::default());
        compare(&p, &out, &refs[0..1], &expected[0..1], &tj.qin_ids, &qin[0..1], &mut cm, &mut cs);
        control = json!({"wrong_seed_batch": 0, "mismatch": cm, "discriminates": mismatch_total(&cm) > 0});
        eprintln!("control (wrong seed, batch 0): {}", serde_json::to_string(&control).unwrap());
    }
    let parity = mismatch_total(&mm) == 0 && gpu_errors == 0;
    let report = json!({
        "circuit": tj.circuit, "circuit_sha256": tj.circuit_sha256, "salt": tj.salt, "program_path": program_path,
        "program": tj.program, "spec": tj.spec,
        "batches": nbatch, "shots": st.shots, "group": group, "seg_words": seg, "segments_per_pass": segments, "passes": passes,
        "bundle_kmax": kmax, "bundle_stats": bst, "bundle_s": t_bundle, "stream_words": code.len(),
        "lanes_per_thread": lanes.shots(), "threads_per_batch": lanes.threads_per_batch(), "gpu_threads": nbatch * lanes.threads_per_batch(), "workgroup": wg,
        "gpu": {"adapter": g.adapter_name, "backend": g.backend, "init_s": g.t_init, "code_upload_s": t_code, "upload_s": t_upload, "kernel_s": t_kernel, "readback_s": t_readback,
                "segment_s": seg_times, "errors": gpu_errors, "total_s": g.t_init + t_code + t_upload + t_kernel + t_readback, "total_with_sampling_s": t_sample + g.t_init + t_code + t_upload + t_kernel + t_readback},
        "cpu": {"threads": threads, "program_load_s": t_load, "sample_s": t_sample, "reference_with_state_s": t_cpu_ref, "run_dqi_s": t_cpu_run_dqi, "run_dqi": cpu_run_dqi_summary},
        "parity": {"gate": if parity { "PASS" } else { "FAIL" }, "mismatch": mm, "compared": {"qubits_per_shot": p.num_qubits, "bits_per_shot_batches": keep_bits, "kinds_per_batch": NCOUNT}},
        "results_cpu": {"value_error_shots": st.val_errors, "phase_error_shots": st.phase_errors, "mean_toffolis": st.toffolis as f64 / st.shots.max(1) as f64},
        "results_gpu": {"mean_toffolis": gpu_toffolis as f64 / st.shots.max(1) as f64},
        "control": control,
    });
    let s = serde_json::to_string_pretty(&report).unwrap();
    if let Some(out) = a.kv.get("out") {
        std::fs::write(out, &s).map_err(|e| e.to_string())?;
    }
    println!("{s}");
    Ok(())
}

fn main() {
    let a = parse_args();
    let r = match a.cmd.as_str() {
        "small" => cmd_small(&a),
        "dqi" => cmd_dqi(&a),
        _ => Err("usage: kickmix-gpu small|dqi [options]".into()),
    };
    if let Err(e) = r {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
