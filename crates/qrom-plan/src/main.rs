//! qrom-plan CLI.
//!
//!   qrom-plan info <words.npy> <k> [groups]
//!   qrom-plan check-plan <words.npy> <plan.txt>
//!   qrom-plan replan <words.npy> <plan.txt> <mode: pair|structured|block> <seed_from> <seed_to> [trials] [samples] [repair_s] [out_plan_prefix]
//!   qrom-plan flips <words.npy> <plans_dir> <n> <seed> [threads] [max_mismatch_examples] [exclude_coded_status_flips 0|1]
//!   qrom-plan bench <words.npy> <plans_dir> <n> <seed> [threads] [max_mismatch_examples] [exclude_coded_status_flips 0|1]
//!   qrom-plan replan-flips <words.npy> <plans_dir> <n> <seed> [threads] [trials] [samples] [repair_s] [seeds_per_flip]
//!
//! Plans are text files k0.txt .. k14.txt (see textio.rs). All output is JSON on stdout.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use qrom_plan::flips::{apply, changes_coded_status, loaders_of, Flip, FlipReport, LoaderModel, Recipe};
use qrom_plan::loader::Words;
use qrom_plan::plan::plan_once;
use qrom_plan::problem::{default_groups, Problem};
use qrom_plan::rng::Rng;
use qrom_plan::textio::{read_plan, write_plan, Json, PlanFile};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: qrom-plan <info|check-plan|replan|flips|bench> ...");
        std::process::exit(2);
    }
    let r = match args[1].as_str() {
        "info" => cmd_info(&args[2..]),
        "check-plan" => cmd_check_plan(&args[2..]),
        "replan" => cmd_replan(&args[2..]),
        "flips" => cmd_flips(&args[2..], true),
        "bench" => cmd_flips(&args[2..], false),
        "replan-flips" => cmd_replan_flips(&args[2..]),
        other => Err(format!("unknown command {other}")),
    };
    match r {
        Ok(json) => print!("{}", json.to_string_pretty()),
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}

fn arg<T: std::str::FromStr>(args: &[String], i: usize, default: T) -> T {
    args.get(i).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn cmd_info(args: &[String]) -> Result<Json, String> {
    let t0 = Instant::now();
    let words = qrom_plan::load_words(Path::new(&args[0]))?;
    let k: usize = args[1].parse().map_err(|e| format!("k: {e}"))?;
    let groups = if args.len() > 2 {
        read_groups(&args[2])?
    } else {
        default_groups()
    };
    let p = Problem::from_words(&words, k, &groups)?;
    let t_build = t0.elapsed().as_secs_f64();
    let rank = p.rank();
    let ann = p.annihilator().len();
    let singles = p.single_slot_relations().len();
    let pairs: Vec<usize> = p.within_phase_supports(2).iter().map(|s| p.support_kernel(s).len()).collect();
    let triples: Vec<usize> = p.within_phase_supports(3).iter().map(|s| p.support_kernel(s).len()).collect();
    let (zero_ok, spaces) = p.feasible(&[]);
    let t_all = t0.elapsed().as_secs_f64();
    Ok(Json::obj()
        .push("k", Json::Int(k as i64))
        .push("rows", Json::Int(words.rows as i64))
        .push("coded_words", Json::ints(p.coded.clone()))
        .push("H", Json::Int(p.h as i64))
        .push("rank", Json::Int(rank as i64))
        .push("annihilator_dim", Json::Int(ann as i64))
        .push("single_slot_relations", Json::Int(singles as i64))
        .push("pair_kernel_dims", Json::ints(pairs))
        .push("triple_kernel_dims", Json::ints(triples))
        .push("forms", Json::Int(p.forms.len() as i64))
        .push("planes", Json::Int(p.planes.len() as i64))
        .push("zero_relations_feasible", Json::Bool(zero_ok))
        .push("zero_relation_span", Json::Int(p.span_dim(&spaces) as i64))
        .push("seconds_build", Json::Float(t_build))
        .push("seconds_total", Json::Float(t_all)))
}

fn read_groups(text: &str) -> Result<Vec<Vec<usize>>, String> {
    text.split(';')
        .map(|g| g.split(',').map(|x| x.trim().parse::<usize>().map_err(|e| format!("groups: {e}"))).collect())
        .collect()
}

fn cmd_check_plan(args: &[String]) -> Result<Json, String> {
    let t0 = Instant::now();
    let words = qrom_plan::load_words(Path::new(&args[0]))?;
    let pf = read_plan(Path::new(&args[1]))?;
    let p = Problem::from_words(&words, pf.k, &pf.groups)?;
    let c = p.check_plan(&pf.relations, &pf.terms);
    Ok(Json::obj()
        .push("k", Json::Int(pf.k as i64))
        .push("relations", Json::Int(c.relations as i64))
        .push("relations_independent", Json::Bool(c.relations_independent))
        .push("all_relations", Json::Bool(c.all_relations))
        .push("span_feasible", Json::Bool(c.span_feasible))
        .push("templates_in_w", Json::Bool(c.templates_in_w))
        .push("templates_independent", Json::Bool(c.templates_independent))
        .push("pairs", Json::Int(c.pairs as i64))
        .push("singles", Json::Int(c.singles as i64))
        .push("pairs_valid", Json::Bool(c.pairs_valid))
        .push("forms_valid", Json::Bool(c.forms_valid))
        .push("reconstructs", Json::Bool(c.reconstructs))
        .push("toffolis", Json::Int(c.toffolis as i64))
        .push("stored_toffolis", pf.toffolis.map_or(Json::Null, |t| Json::Int(t as i64)))
        .push("ok", Json::Bool(c.ok() && pf.toffolis.map_or(true, |t| t == c.toffolis)))
        .push("seconds", Json::Float(t0.elapsed().as_secs_f64())))
}

fn cmd_replan(args: &[String]) -> Result<Json, String> {
    let words = qrom_plan::load_words(Path::new(&args[0]))?;
    let pf = read_plan(Path::new(&args[1]))?;
    let mode = args[2].as_str();
    let seed_from: u64 = args[3].parse().map_err(|e| format!("seed: {e}"))?;
    let seed_to: u64 = args[4].parse().map_err(|e| format!("seed: {e}"))?;
    let trials: usize = arg(args, 5, 12);
    let samples: usize = arg(args, 6, 24);
    let repair_s: f64 = arg(args, 7, 0.0);
    let out_prefix: Option<PathBuf> = args.get(8).map(PathBuf::from);
    let t0 = Instant::now();
    let p = Problem::from_words(&words, pf.k, &pf.groups)?;
    let t_build = t0.elapsed().as_secs_f64();
    let mut attempts = Vec::new();
    let mut best: Option<(usize, usize, u64)> = None;
    for seed in seed_from..seed_to {
        let t1 = Instant::now();
        let (rels, ok, spaces) = match mode {
            "pair" => {
                let (ok, spaces) = p.feasible(&pf.relations);
                (pf.relations.clone(), ok, spaces)
            }
            "structured" => p.structured_draw(pf.count.ok_or("plan has no count")?, seed),
            "block" => p.block_draw(&pf.design, pf.per.ok_or("plan has no per")?, seed),
            other => return Err(format!("unknown mode {other}")),
        };
        let t_draw = t1.elapsed().as_secs_f64();
        let mut row = Json::obj()
            .push("seed", Json::Int(seed as i64))
            .push("relations", Json::Int(rels.len() as i64))
            .push("span_feasible", Json::Bool(ok))
            .push("seconds_draw", Json::Float(t_draw));
        if ok {
            let t2 = Instant::now();
            if let Some((pairing, before, defects)) = plan_once(&p, &rels, &spaces, seed, trials, samples, repair_s) {
                let t_pair = t2.elapsed().as_secs_f64();
                let terms = p.terms(&pairing)?;
                let check = p.check_plan(&rels, &terms);
                row = row
                    .push("pairs", Json::Int(pairing.pairs.len() as i64))
                    .push("singles", Json::Int(pairing.singles.len() as i64))
                    .push("singles_before_repair", Json::Int(before as i64))
                    .push("defects_after_repair", Json::Int(defects as i64))
                    .push("products", Json::Int(pairing.products() as i64))
                    .push("toffolis", Json::Int(pairing.toffolis() as i64))
                    .push("terms_check_ok", Json::Bool(check.ok()))
                    .push("seconds_pair", Json::Float(t_pair));
                let key = (pairing.products(), usize::MAX - rels.len(), seed);
                if best.map_or(true, |b| (key.0, key.1) < (b.0, b.1)) {
                    best = Some(key);
                    if let Some(prefix) = &out_prefix {
                        let out = PlanFile {
                            k: pf.k,
                            groups: pf.groups.clone(),
                            toffolis: Some(pairing.toffolis()),
                            relations: rels.clone(),
                            terms,
                            ..Default::default()
                        };
                        write_plan(&PathBuf::from(format!("{}_best.txt", prefix.display())), &out)?;
                    }
                }
            } else {
                row = row.push("pairs", Json::Null).push("seconds_pair", Json::Float(t2.elapsed().as_secs_f64()));
            }
        }
        row = row.push("seconds", Json::Float(t1.elapsed().as_secs_f64()));
        attempts.push(row);
    }
    Ok(Json::obj()
        .push("k", Json::Int(pf.k as i64))
        .push("mode", Json::Str(mode.to_string()))
        .push("stored_toffolis", pf.toffolis.map_or(Json::Null, |t| Json::Int(t as i64)))
        .push("seconds_build", Json::Float(t_build))
        .push("best_products", best.map_or(Json::Null, |b| Json::Int(b.0 as i64)))
        .push("best_toffolis", best.map_or(Json::Null, |b| Json::Int((89 + b.0) as i64)))
        .push("best_relations", best.map_or(Json::Null, |b| Json::Int((usize::MAX - b.1) as i64)))
        .push("best_seed", best.map_or(Json::Null, |b| Json::Int(b.2 as i64)))
        .push("attempts", Json::Arr(attempts))
        .push("seconds_total", Json::Float(t0.elapsed().as_secs_f64())))
}

struct Models {
    words: Words,
    coded: Vec<bool>,
    models: Vec<LoaderModel>,
}

fn load_models(words_path: &Path, plans_dir: &Path) -> Result<Models, String> {
    let words = qrom_plan::load_words(words_path)?;
    let coded = words.coded_columns();
    let mut models = Vec::new();
    for k in 0..words.loaders() {
        let pf = read_plan(&plans_dir.join(format!("k{k}.txt")))?;
        if pf.k != k {
            return Err(format!("plan k{k}.txt says k = {}", pf.k));
        }
        let p = Problem::from_words(&words, k, &pf.groups)?;
        let check = p.check_plan(&pf.relations, &pf.terms);
        if !check.ok() {
            return Err(format!("plan k{k} does not verify: {check:?}"));
        }
        let recipe = Recipe { count: pf.count, design: pf.design.clone(), per: pf.per };
        models.push(LoaderModel::new(p, pf.relations, check.toffolis, recipe));
    }
    Ok(Models { words, coded, models })
}

fn report_json(r: &FlipReport) -> Json {
    Json::obj()
        .push("k", Json::Int(r.k as i64))
        .push("fallback_full", Json::Bool(r.fallback_full))
        .push("coded_changed", Json::Bool(r.coded_changed))
        .push("broken", Json::Bool(r.broken))
        .push("rebase_ok", Json::Bool(r.rebase_ok))
        .push("relations_after", Json::Int(r.relations_after as i64))
        .push("toffolis_fixed_after", Json::Int(r.toffolis_fixed_after as i64))
        .push("patch_exists", r.patch_exists.map_or(Json::Null, Json::Bool))
        .push("annihilator_after", Json::Int(r.annihilator_after as i64))
        .push("nullities_after", Json::ints(r.nullities_after.clone()))
}

fn cmd_flips(args: &[String], compare: bool) -> Result<Json, String> {
    let t0 = Instant::now();
    let n: usize = args[2].parse().map_err(|e| format!("n: {e}"))?;
    let seed: u64 = args[3].parse().map_err(|e| format!("seed: {e}"))?;
    let threads: usize = arg(args, 4, 1).max(1);
    let max_examples: usize = arg(args, 5, 5);
    let exclude_coded: usize = arg(args, 6, 0);
    let models = Arc::new(load_models(Path::new(&args[0]), Path::new(&args[1]))?);
    let t_load = t0.elapsed().as_secs_f64();
    let flips = Arc::new(draw_flips(&models, n, seed, exclude_coded == 1));
    let nullity_names: Vec<String> = models.models[0].subs.iter().map(|s| format!("{s:?}")).collect();

    // incremental pass (with the coded-status fallback), timed
    let t1 = Instant::now();
    let inc = run_threads(threads, flips.len(), {
        let models = models.clone();
        let flips = flips.clone();
        move |i| {
            let f = flips[i];
            let fallback = changes_coded_status(&models.words, &models.coded, &f);
            loaders_of(f.col, models.models.len())
                .into_iter()
                .map(|k| {
                    if fallback {
                        let w = apply(&models.words, &f);
                        let coded = w.coded_columns();
                        models.models[k].full(&w, &coded).unwrap()
                    } else {
                        models.models[k].incremental(&f)
                    }
                })
                .collect::<Vec<_>>()
        }
    });
    let t_inc = t1.elapsed().as_secs_f64();
    let fallbacks = inc.iter().filter(|rs| rs.iter().any(|r| r.fallback_full)).count();
    let loader_evals: usize = inc.iter().map(|rs| rs.len()).sum();
    let broken: usize = inc.iter().flatten().filter(|r| r.broken).count();
    let patchable: usize = inc.iter().flatten().filter(|r| r.patch_exists == Some(true)).count();
    let ann_up: usize = inc.iter().flatten().filter(|r| r.annihilator_after > 440).count();
    let mut supply_up = 0usize;
    for rs in inc.iter() {
        for r in rs {
            let base = models.models[r.k].nullities();
            if r.nullities_after.iter().zip(base.iter()).any(|(a, b)| a > b) {
                supply_up += 1;
            }
        }
    }
    let mut out = Json::obj()
        .push("flips", Json::Int(n as i64))
        .push("loader_evaluations", Json::Int(loader_evals as i64))
        .push("threads", Json::Int(threads as i64))
        .push("seconds_load_models", Json::Float(t_load))
        .push("seconds_incremental", Json::Float(t_inc))
        .push("flips_per_second_incremental", Json::Float(n as f64 / t_inc))
        .push("fallback_full_flips", Json::Int(fallbacks as i64))
        .push("loader_evals_relation_broken", Json::Int(broken as i64))
        .push("loader_evals_patchable_by_one_template", Json::Int(patchable as i64))
        .push("loader_evals_annihilator_grew", Json::Int(ann_up as i64))
        .push("loader_evals_some_supply_nullity_grew", Json::Int(supply_up as i64))
        .push("sub_matrices", Json::Int(nullity_names.len() as i64))
        .push("exclude_coded_status_flips", Json::Bool(exclude_coded == 1));
    if compare {
        let t2 = Instant::now();
        let full = run_threads(threads, flips.len(), {
            let models = models.clone();
            let flips = flips.clone();
            move |i| {
                let f = flips[i];
                let w = apply(&models.words, &f);
                let coded = w.coded_columns();
                loaders_of(f.col, models.models.len()).into_iter().map(|k| models.models[k].full(&w, &coded).unwrap()).collect::<Vec<_>>()
            }
        });
        let t_full = t2.elapsed().as_secs_f64();
        let mut mismatches = 0usize;
        let mut examples = Vec::new();
        let mut genuine_incremental = 0usize;
        let mut fallback_evals = 0usize;
        let mut fallback_rebase_invalid = 0usize;
        for (i, (a, b)) in inc.iter().zip(full.iter()).enumerate() {
            for (ra, rb) in a.iter().zip(b.iter()) {
                if ra.fallback_full {
                    // both paths ran the same full recomputation; the fixed plan is not a single truth change there
                    fallback_evals += 1;
                    if !rb.rebase_ok {
                        fallback_rebase_invalid += 1;
                    }
                    continue;
                }
                genuine_incremental += 1;
                let same = ra.broken == rb.broken
                    && ra.relations_after == rb.relations_after
                    && ra.toffolis_fixed_after == rb.toffolis_fixed_after
                    && ra.patch_exists == rb.patch_exists
                    && ra.annihilator_after == rb.annihilator_after
                    && ra.nullities_after == rb.nullities_after
                    && rb.rebase_ok;
                if !same {
                    mismatches += 1;
                    if examples.len() < max_examples {
                        let f = flips[i];
                        examples.push(
                            Json::obj()
                                .push("flip", Json::ints([f.addr, f.col, f.bit as usize]))
                                .push("incremental", report_json(ra))
                                .push("full", report_json(rb)),
                        );
                    }
                }
            }
        }
        out = out
            .push("seconds_full", Json::Float(t_full))
            .push("flips_per_second_full", Json::Float(n as f64 / t_full))
            .push("loader_evals_genuinely_incremental", Json::Int(genuine_incremental as i64))
            .push("loader_evals_fallback_full", Json::Int(fallback_evals as i64))
            .push("loader_evals_fallback_rebase_invalid", Json::Int(fallback_rebase_invalid as i64))
            .push("mismatches", Json::Int(mismatches as i64))
            .push("parity_ok", Json::Bool(mismatches == 0))
            .push("mismatch_examples", Json::Arr(examples));
    }
    out = out.push("seconds_total", Json::Float(t0.elapsed().as_secs_f64()));
    Ok(out)
}

/// Random flips (address, column, bit 0..=6); optionally only flips that keep every column's coded status.
fn draw_flips(models: &Models, n: usize, seed: u64, exclude_coded: bool) -> Vec<Flip> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let f = Flip { addr: rng.below(models.words.rows), col: rng.below(models.words.cols), bit: rng.below(7) as u8 };
        if exclude_coded && changes_coded_status(&models.words, &models.coded, &f) {
            continue;
        }
        out.push(f);
    }
    out
}

/// Re-plan both affected loaders after each flip with the plan's own recipe (structured draw at the plan's relation
/// count, or the block design), pairing and repair; reports the Toffoli change distribution and the rate.
fn cmd_replan_flips(args: &[String]) -> Result<Json, String> {
    let t0 = Instant::now();
    let n: usize = args[2].parse().map_err(|e| format!("n: {e}"))?;
    let seed: u64 = args[3].parse().map_err(|e| format!("seed: {e}"))?;
    let threads: usize = arg(args, 4, 1).max(1);
    let trials: usize = arg(args, 5, 12);
    let samples: usize = arg(args, 6, 24);
    let repair_s: f64 = arg(args, 7, 60.0);
    let seeds_per_flip: u64 = arg(args, 8, 1);
    let models = Arc::new(load_models(Path::new(&args[0]), Path::new(&args[1]))?);
    let flips = Arc::new(draw_flips(&models, n, seed, true));
    let t1 = Instant::now();
    let results = run_threads(threads, flips.len(), {
        let models = models.clone();
        let flips = flips.clone();
        move |i| {
            let f = flips[i];
            let w = apply(&models.words, &f);
            loaders_of(f.col, models.models.len())
                .into_iter()
                .map(|k| {
                    let m = &models.models[k];
                    let p = Problem::from_words(&w, k, &m.problem.groups).unwrap();
                    let mut best: Option<usize> = None;
                    let mut attempts = 0usize;
                    for s in 0..seeds_per_flip {
                        let rs = seed.wrapping_mul(1_000_003).wrapping_add(i as u64 * 97 + s);
                        let (rels, ok, spaces) = if m.recipe.per.is_some() && !m.recipe.design.is_empty() {
                            p.block_draw(&m.recipe.design, m.recipe.per.unwrap(), rs)
                        } else {
                            p.structured_draw(m.recipe.count.unwrap_or(m.relations.len()), rs)
                        };
                        attempts += 1;
                        if !ok {
                            continue;
                        }
                        if let Some((pairing, _, _)) = plan_once(&p, &rels, &spaces, rs, trials, samples, repair_s) {
                            let t = pairing.toffolis();
                            if best.map_or(true, |b| t < b) {
                                best = Some(t);
                            }
                        }
                    }
                    (k, m.toffolis, best, attempts)
                })
                .collect::<Vec<_>>()
        }
    });
    let t_replan = t1.elapsed().as_secs_f64();
    let mut deltas: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    let mut evals = 0usize;
    let mut attempts = 0usize;
    let mut failed = 0usize;
    let mut per_loader: Vec<(usize, usize, usize)> = vec![(0, 0, 0); models.models.len()]; // (evals, better, equal)
    for rs in &results {
        for &(k, t_plan, best, a) in rs {
            evals += 1;
            attempts += a;
            per_loader[k].0 += 1;
            match best {
                Some(t) => {
                    *deltas.entry(t as i64 - t_plan as i64).or_default() += 1;
                    if t < t_plan {
                        per_loader[k].1 += 1;
                    } else if t == t_plan {
                        per_loader[k].2 += 1;
                    }
                }
                None => failed += 1,
            }
        }
    }
    let delta_json = Json::Obj(deltas.iter().map(|(d, c)| (d.to_string(), Json::Int(*c as i64))).collect());
    let per_loader_json = Json::Arr(
        per_loader
            .iter()
            .enumerate()
            .map(|(k, (e, b, q))| {
                Json::obj()
                    .push("k", Json::Int(k as i64))
                    .push("plan_toffolis", Json::Int(models.models[k].toffolis as i64))
                    .push("evals", Json::Int(*e as i64))
                    .push("cheaper_than_plan", Json::Int(*b as i64))
                    .push("equal_to_plan", Json::Int(*q as i64))
            })
            .collect(),
    );
    Ok(Json::obj()
        .push("flips", Json::Int(n as i64))
        .push("loader_replans", Json::Int(evals as i64))
        .push("plan_attempts", Json::Int(attempts as i64))
        .push("seeds_per_flip", Json::Int(seeds_per_flip as i64))
        .push("threads", Json::Int(threads as i64))
        .push("trials", Json::Int(trials as i64))
        .push("repair_seconds_budget", Json::Float(repair_s))
        .push("replans_without_pairing", Json::Int(failed as i64))
        .push("toffoli_delta_histogram", delta_json)
        .push("per_loader", per_loader_json)
        .push("seconds_replan", Json::Float(t_replan))
        .push("flips_per_second", Json::Float(n as f64 / t_replan))
        .push("seconds_per_plan_attempt_cpu", Json::Float(t_replan * threads as f64 / attempts.max(1) as f64))
        .push("seconds_total", Json::Float(t0.elapsed().as_secs_f64())))
}

fn run_threads<T: Send + 'static>(threads: usize, n: usize, f: impl Fn(usize) -> T + Send + Sync + 'static) -> Vec<T> {
    let f = Arc::new(f);
    if threads <= 1 {
        return (0..n).map(|i| f(i)).collect();
    }
    let chunk = (n + threads - 1) / threads;
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let f = f.clone();
            std::thread::spawn(move || (t * chunk..((t + 1) * chunk).min(n)).map(|i| f(i)).collect::<Vec<T>>())
        })
        .collect();
    handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
}
