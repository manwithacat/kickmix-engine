//! Word-flip search helpers on top of the incremental evaluator (card
//! `experiments/2026-11-03-chem-word-flip-search.md`).
//!
//! * `screen`: price every single flip of a column range and bit set on its affected loaders and keep the flips
//!   that raise some relation-supply nullity (single slot, within-phase pair or triple) of a loader's fixed plan.
//! * `batch`: apply a set of flips, report the exact supply change on the affected loaders (full recomputation,
//!   no incremental shortcuts) and re-plan them with the plan's own recipe over a seed x relation-count lottery.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use crate::bits::{B128, B525};
use crate::flips::{changes_coded_status, loaders_of, sub_cols, Flip, LoaderModel};
use crate::gf2::{rank, Echelon};
use crate::loader::{Words, N525, OUT};
use crate::plan::plan_once;
use crate::problem::Problem;
use crate::textio::{write_plan, Json, PlanFile};

/// One (flip, loader) screen hit: which sub-matrices' nullities grew.
#[derive(Clone, Debug)]
pub struct Hit {
    pub flip: Flip,
    pub k: usize,
    pub broken: bool,
    pub patch: Option<bool>,
    pub grew: Vec<usize>,
    pub shrank: usize,
}

/// Screen all flips (addr in 0..rows, col in cols, bit in bits) on the loaders they touch. Flips that would change
/// a column's coded status are skipped (they remove the coded single-slot relations and are never a gain).
pub fn screen(words: &Arc<Words>, coded: &Arc<Vec<bool>>, models: &Arc<Vec<LoaderModel>>, cols: Vec<usize>, bits: Vec<u8>,
              threads: usize, only_loaders: Option<Vec<usize>>) -> (Vec<Hit>, ScreenStats) {
    let rows = words.rows;
    let mut flips: Vec<Flip> = Vec::with_capacity(rows * cols.len() * bits.len());
    for a in 0..rows {
        for &c in &cols {
            for &b in &bits {
                flips.push(Flip { addr: a, col: c, bit: b });
            }
        }
    }
    let flips = Arc::new(flips);
    let base: Arc<Vec<Vec<usize>>> = Arc::new(models.iter().map(|m| m.nullities()).collect());
    let only = Arc::new(only_loaders);
    let n = flips.len();
    let out = run_threads(threads, n, {
        let words = words.clone();
        let coded = coded.clone();
        let models = models.clone();
        let flips = flips.clone();
        let base = base.clone();
        let only = only.clone();
        move |i| {
            let f = flips[i];
            let mut hits = Vec::new();
            let mut skipped = false;
            let mut evals = 0usize;
            if changes_coded_status(&words, &coded, &f) {
                skipped = true;
                return (hits, skipped, evals);
            }
            for k in loaders_of(f.col, models.len()) {
                if let Some(o) = only.as_ref() {
                    if !o.contains(&k) {
                        continue;
                    }
                }
                evals += 1;
                let r = models[k].incremental(&f);
                let mut grew = Vec::new();
                let mut shrank = 0;
                for (si, (&a, &b)) in r.nullities_after.iter().zip(base[k].iter()).enumerate() {
                    if a > b {
                        grew.push(si);
                    } else if a < b {
                        shrank += 1;
                    }
                }
                if !grew.is_empty() {
                    hits.push(Hit { flip: f, k, broken: r.broken, patch: r.patch_exists, grew, shrank });
                }
            }
            (hits, skipped, evals)
        }
    });
    let mut hits = Vec::new();
    let mut stats = ScreenStats { flips: n, skipped_coded: 0, loader_evals: 0, hits: 0 };
    for (h, s, e) in out {
        stats.skipped_coded += s as usize;
        stats.loader_evals += e;
        stats.hits += h.len();
        hits.extend(h);
    }
    (hits, stats)
}

#[derive(Clone, Debug)]
pub struct ScreenStats {
    pub flips: usize,
    pub skipped_coded: usize,
    pub loader_evals: usize,
    pub hits: usize,
}

/// Exact supply report of one loader on a (possibly multiply) flipped table, against its fixed plan.
pub struct SupplyReport {
    pub k: usize,
    pub coded_same: bool,
    pub relations_valid: usize,
    pub surviving_span: usize,
    pub annihilator: usize,
    pub nullities: Vec<usize>,
    pub changed: Vec<(usize, usize, usize)>, // (sub index, before, after)
}

pub fn supply_report(m: &LoaderModel, p2: &Problem) -> SupplyReport {
    let coded_same = p2.coded == m.problem.coded;
    let relations_valid = m.relations.iter().filter(|r| p2.is_relation(r)).count();
    // dim of span(R) that stays in the annihilator = |R| - rank of the |R| x H matrix (r . F_h)
    let dots: Vec<B128> = m
        .relations
        .iter()
        .map(|r| {
            let mut v = B128::ZERO;
            for (hi, row) in p2.rows.iter().enumerate() {
                if row.dot(r) {
                    v.set(hi);
                }
            }
            v
        })
        .collect();
    let surviving_span = m.relations.len() - rank(&dots);
    let before = m.nullities();
    let nullities: Vec<usize> = m
        .subs
        .iter()
        .map(|sub| {
            let cols = sub_cols(sub);
            let restricted: Vec<B128> = p2
                .rows
                .iter()
                .map(|r| {
                    let mut v = B128::ZERO;
                    for (ci, &c) in cols.iter().enumerate() {
                        if r.get(c) {
                            v.set(ci);
                        }
                    }
                    v
                })
                .collect();
            cols.len() - rank(&restricted)
        })
        .collect();
    let changed = nullities
        .iter()
        .zip(before.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (a, b))| (i, *b, *a))
        .collect();
    SupplyReport { k: m.problem.k, coded_same, relations_valid, surviving_span, annihilator: N525 - p2.rank(), nullities, changed }
}

pub fn supply_json(m: &LoaderModel, r: &SupplyReport) -> Json {
    Json::obj()
        .push("k", Json::Int(r.k as i64))
        .push("coded_same", Json::Bool(r.coded_same))
        .push("plan_relations", Json::Int(m.relations.len() as i64))
        .push("plan_toffolis", Json::Int(m.toffolis as i64))
        .push("relations_still_valid", Json::Int(r.relations_valid as i64))
        .push("surviving_relation_span", Json::Int(r.surviving_span as i64))
        .push("annihilator", Json::Int(r.annihilator as i64))
        .push(
            "nullity_changes",
            Json::Arr(
                r.changed
                    .iter()
                    .map(|(i, b, a)| {
                        Json::obj()
                            .push("sub", Json::Int(*i as i64))
                            .push("slots", Json::ints(m.subs[*i].clone()))
                            .push("before", Json::Int(*b as i64))
                            .push("after", Json::Int(*a as i64))
                    })
                    .collect(),
            ),
        )
        .push("triple_nullity_sum", Json::Int(r.nullities.iter().zip(m.subs.iter()).filter(|(_, s)| s.len() == 3).map(|(n, _)| *n as i64).sum()))
        .push("pair_nullity_sum", Json::Int(r.nullities.iter().zip(m.subs.iter()).filter(|(_, s)| s.len() == 2).map(|(n, _)| *n as i64).sum()))
        .push("single_nullity_sum", Json::Int(r.nullities.iter().zip(m.subs.iter()).filter(|(_, s)| s.len() == 1).map(|(n, _)| *n as i64).sum()))
}

/// Re-plan one loader on a table over counts x seeds with the model's recipe (structured draw at each count, or the
/// block design when the recipe has one); returns the best (products, relations, count, seed, plan) and the
/// per-count Toffoli histograms.
pub struct ReplanResult {
    pub k: usize,
    pub best: Option<(usize, usize, usize, u64)>, // products, relations, count, seed
    pub best_plan: Option<PlanFile>,
    pub hist: Vec<(usize, Vec<(usize, usize)>)>, // per count: (toffolis, n)
    pub attempts: usize,
    pub infeasible: usize,
    pub seconds: f64,
}

pub fn replan_lottery(p: &Arc<Problem>, m_recipe_design: &[Vec<usize>], m_per: Option<usize>, counts: &[usize], seed_from: u64,
                      seed_to: u64, threads: usize, trials: usize, samples: usize, repair_s: f64) -> ReplanResult {
    let t0 = Instant::now();
    let jobs: Vec<(usize, u64)> = counts.iter().flat_map(|&c| (seed_from..seed_to).map(move |s| (c, s))).collect();
    let jobs = Arc::new(jobs);
    let design = Arc::new(m_recipe_design.to_vec());
    let out = run_threads(threads, jobs.len(), {
        let p = p.clone();
        let jobs = jobs.clone();
        let design = design.clone();
        move |i| {
            let (count, seed) = jobs[i];
            let (rels, ok, spaces) = if m_per.is_some() && !design.is_empty() {
                p.block_draw(&design, m_per.unwrap(), seed)
            } else {
                p.structured_draw(count, seed)
            };
            if !ok {
                return (count, seed, rels.len(), None);
            }
            match plan_once(&p, &rels, &spaces, seed, trials, samples, repair_s) {
                Some((pairing, _, _)) => {
                    let terms = p.terms(&pairing).ok();
                    let good = terms.as_ref().map_or(false, |t| p.check_plan(&rels, t).ok());
                    if !good {
                        return (count, seed, rels.len(), None);
                    }
                    (count, seed, rels.len(), Some((pairing.products(), rels, terms.unwrap())))
                }
                None => (count, seed, rels.len(), None),
            }
        }
    });
    let mut best: Option<(usize, usize, usize, u64)> = None;
    let mut best_plan = None;
    let mut hist: Vec<(usize, Vec<(usize, usize)>)> = counts.iter().map(|&c| (c, Vec::new())).collect();
    let mut infeasible = 0;
    for (count, seed, nrel, res) in out {
        let hi = counts.iter().position(|&c| c == count).unwrap();
        match res {
            None => infeasible += 1,
            Some((products, rels, terms)) => {
                let t = 89 + products;
                if let Some(e) = hist[hi].1.iter_mut().find(|e| e.0 == t) {
                    e.1 += 1;
                } else {
                    hist[hi].1.push((t, 1));
                }
                let key = (products, usize::MAX - nrel);
                if best.map_or(true, |b| key < (b.0, usize::MAX - b.1)) {
                    best = Some((products, nrel, count, seed));
                    best_plan = Some(PlanFile {
                        k: p.k,
                        groups: p.groups.clone(),
                        toffolis: Some(t),
                        count: Some(count),
                        seed: Some(seed),
                        relations: rels,
                        terms,
                        ..Default::default()
                    });
                }
            }
        }
    }
    for h in hist.iter_mut() {
        h.1.sort();
    }
    ReplanResult { k: p.k, best, best_plan, hist, attempts: jobs.len(), infeasible, seconds: t0.elapsed().as_secs_f64() }
}

pub fn replan_json(r: &ReplanResult, base_toffolis: usize) -> Json {
    Json::obj()
        .push("k", Json::Int(r.k as i64))
        .push("plan_toffolis", Json::Int(base_toffolis as i64))
        .push("best_toffolis", r.best.map_or(Json::Null, |b| Json::Int((89 + b.0) as i64)))
        .push("best_relations", r.best.map_or(Json::Null, |b| Json::Int(b.1 as i64)))
        .push("best_count", r.best.map_or(Json::Null, |b| Json::Int(b.2 as i64)))
        .push("best_seed", r.best.map_or(Json::Null, |b| Json::Int(b.3 as i64)))
        .push("delta_vs_plan", r.best.map_or(Json::Null, |b| Json::Int((89 + b.0) as i64 - base_toffolis as i64)))
        .push("attempts", Json::Int(r.attempts as i64))
        .push("infeasible_or_unpaired", Json::Int(r.infeasible as i64))
        .push(
            "toffoli_histogram_by_count",
            Json::Arr(
                r.hist
                    .iter()
                    .map(|(c, h)| {
                        Json::obj().push("count", Json::Int(*c as i64)).push(
                            "toffolis",
                            Json::Obj(h.iter().map(|(t, n)| (t.to_string(), Json::Int(*n as i64))).collect()),
                        )
                    })
                    .collect(),
            ),
        )
        .push("seconds", Json::Float(r.seconds))
}

/// Write the best plan of a lottery as text next to the batch outputs.
pub fn write_best(dir: &Path, r: &ReplanResult) -> Result<Option<String>, String> {
    if let Some(pf) = &r.best_plan {
        let path = dir.join(format!("k{}_best.txt", r.k));
        write_plan(&path, pf)?;
        return Ok(Some(path.display().to_string()));
    }
    Ok(None)
}

/// Parse a flips file: lines `addr col bit` (whitespace separated, `#` comments).
pub fn read_flips(path: &Path) -> Result<Vec<Flip>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (ln, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 3 {
            return Err(format!("{}:{}: need addr col bit", path.display(), ln + 1));
        }
        let addr: usize = parts[0].parse().map_err(|e| format!("{}:{}: {e}", path.display(), ln + 1))?;
        let col: usize = parts[1].parse().map_err(|e| format!("{}:{}: {e}", path.display(), ln + 1))?;
        let bit: u8 = parts[2].parse().map_err(|e| format!("{}:{}: {e}", path.display(), ln + 1))?;
        out.push(Flip { addr, col, bit });
    }
    Ok(out)
}

/// Apply flips in order (a repeated flip cancels).
pub fn apply_all(words: &Words, flips: &[Flip]) -> Words {
    let mut w = words.clone();
    for f in flips {
        w.flip(f.addr, f.col, f.bit);
    }
    w
}

/// Sanity: every relation of an echelon is independent (used by tests).
#[allow(dead_code)]
pub fn independent(rels: &[B525]) -> bool {
    let mut e = Echelon::<9>::new();
    rels.iter().all(|r| e.insert(*r))
}

pub fn run_threads<T: Send + 'static>(threads: usize, n: usize, f: impl Fn(usize) -> T + Send + Sync + 'static) -> Vec<T> {
    let f = Arc::new(f);
    if threads <= 1 || n <= 1 {
        return (0..n).map(|i| f(i)).collect();
    }
    // interleaved assignment so that expensive jobs spread over the threads
    let handles: Vec<_> = (0..threads)
        .map(|t| {
            let f = f.clone();
            std::thread::spawn(move || (t..n).step_by(threads).map(|i| (i, f(i))).collect::<Vec<(usize, T)>>())
        })
        .collect();
    let mut all: Vec<(usize, T)> = handles.into_iter().flat_map(|h| h.join().unwrap()).collect();
    all.sort_by_key(|x| x.0);
    all.into_iter().map(|x| x.1).collect()
}

#[allow(dead_code)]
fn _slots(_: usize) -> usize {
    OUT
}
