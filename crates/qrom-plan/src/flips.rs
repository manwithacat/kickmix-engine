//! Incremental word-flip pricing under a fixed plan.
//!
//! A flip of bit b of word column j at address a changes the truth table of loaders k = j / 5 and k + 1 at one entry
//! (high block a >> 4, low a & 15, output o = (j % 5) * 7 + b). Through the Mobius transform that is the rank-one
//! update F' = F + u (x) d with u = {h : h contains a >> 4} over the 85 high monomials and d = sum over low
//! monomials l containing a & 15 of e_(l, o). Everything the plan cares about follows from (u, d) and precomputed
//! solvers:
//! * a relation r of the fixed plan survives iff d . r = 0; the surviving set is a hyperplane of span(R), so at most
//!   one relation is lost (re-basing r_i <- r_i + r_0 for a broken r_0);
//! * the rank of any column sub-matrix A changes by +1 iff (u not in Col A and d_A not in Row A), by -1 iff
//!   (u in Col A, d_A in Row A and d_A . x = 1 for A x = u), else 0; nullities (relation supply per slot
//!   set, the annihilator) follow;
//! * if a relation broke, W' = W + <d>; the fixed plan needs one more product unless re-planned, and a single
//!   admissible template patches it iff some form space V_f grows.
//! Bit-0 flips that change a column's coded status (affine bit 0 on every block) are not a single truth-table
//! change; they fall back to the full recomputation.

use crate::bits::{Bits, B128, B525};
use crate::gf2::{Coords, Echelon};
use crate::loader::{Delta, Words, J, LOW, N525, OUT, SLOTS};
use crate::problem::{get35, put35, Problem, Spaces};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Flip {
    pub addr: usize,
    pub col: usize,
    pub bit: u8,
}

/// Rank-one update solver for one column sub-matrix of F (columns `cols`, width <= W * 64 bits).
#[derive(Clone, Debug)]
pub struct RankOne<const W: usize> {
    pub cols: Vec<usize>,
    pub width: usize,
    pub rank: usize,
    rows_ech: Echelon<W>,
    x_by_ahi: Vec<Option<Bits<W>>>,
}

impl<const W: usize> RankOne<W> {
    pub fn new(rows: &[B525], cols: Vec<usize>, high: usize) -> Self {
        let width = cols.len();
        let restricted: Vec<Bits<W>> = rows
            .iter()
            .map(|r| {
                let mut v = Bits::<W>::ZERO;
                for (ci, &c) in cols.iter().enumerate() {
                    if r.get(c) {
                        v.set(ci);
                    }
                }
                v
            })
            .collect();
        let mut rows_ech = Echelon::<W>::new();
        for &r in &restricted {
            rows_ech.insert(r);
        }
        let rank = rows_ech.len();
        // column space with coordinates over column positions
        let mut col_basis = Coords::<2, W>::new();
        for ci in 0..width {
            let mut v = B128::ZERO;
            for (hi, r) in restricted.iter().enumerate() {
                if r.get(ci) {
                    v.set(hi);
                }
            }
            col_basis.add_at(v, ci);
        }
        let x_by_ahi = (0..high).map(|a_hi| col_basis.solve(u_vector(a_hi, rows.len()))).collect();
        RankOne { cols, width, rank, rows_ech, x_by_ahi }
    }

    pub fn nullity(&self) -> usize {
        self.width - self.rank
    }

    /// Rank change under F + u(a_hi) (x) d, given d restricted to this sub-matrix.
    #[inline]
    pub fn delta_rank(&self, d_sub: Bits<W>, a_hi: usize) -> isize {
        if d_sub.is_zero() {
            return 0;
        }
        let d_in_row = self.rows_ech.contains(d_sub);
        match (&self.x_by_ahi[a_hi], d_in_row) {
            (None, false) => 1,
            (Some(x), true) => {
                if d_sub.dot(x) {
                    -1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    pub fn nullity_after(&self, d_sub: Bits<W>, a_hi: usize) -> usize {
        (self.width as isize - (self.rank as isize + self.delta_rank(d_sub, a_hi))) as usize
    }
}

/// u(a_hi): the 85-bit indicator of high monomials h = 1..=H with h containing a_hi (bit h - 1).
pub fn u_vector(a_hi: usize, high_rows: usize) -> B128 {
    let mut u = B128::ZERO;
    for h in 1..=high_rows {
        if h & a_hi == a_hi {
            u.set(h - 1);
        }
    }
    u
}

/// d: the 525-bit vector of (l, o) entries flipped by a truth change at low address a_lo, output o.
pub fn d_vector(a_lo: usize, o: usize) -> B525 {
    let mut d = B525::ZERO;
    for l in 1..LOW {
        if l & a_lo == a_lo {
            d.set((l - 1) * OUT + o);
        }
    }
    d
}

/// How a plan's relation set was drawn (for re-planning after a flip).
#[derive(Clone, Debug, Default)]
pub struct Recipe {
    pub count: Option<usize>,
    pub design: Vec<Vec<usize>>,
    pub per: Option<usize>,
}

/// Fixed plan for one loader with the precomputed solvers.
pub struct LoaderModel {
    pub problem: Problem,
    pub relations: Vec<B525>,
    pub toffolis: usize,
    pub recipe: Recipe,
    pub subs: Vec<Vec<usize>>, // slot sets: singles, within-phase pairs, within-phase triples
    pub sub_solvers: Vec<RankOne<2>>,
    pub full: RankOne<9>,
    pub spaces: Spaces,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlipReport {
    pub k: usize,
    pub fallback_full: bool,
    pub coded_changed: bool,
    pub broken: bool,
    pub rebase_ok: bool,
    pub relations_after: usize,
    pub toffolis_fixed_after: usize,
    pub patch_exists: Option<bool>,
    pub annihilator_after: usize,
    pub nullities_after: Vec<usize>,
}

impl LoaderModel {
    pub fn new(problem: Problem, relations: Vec<B525>, toffolis: usize, recipe: Recipe) -> Self {
        let mut subs: Vec<Vec<usize>> = (0..SLOTS).map(|s| vec![s]).collect();
        subs.extend(problem.within_phase_supports(2));
        subs.extend(problem.within_phase_supports(3));
        let high = problem.ac.len();
        let sub_solvers = subs.iter().map(|sub| RankOne::<2>::new(&problem.rows, sub_cols(sub), high)).collect();
        let full = RankOne::<9>::new(&problem.rows, (0..N525).collect(), high);
        let spaces = problem.form_spaces(&relations);
        LoaderModel { problem, relations, toffolis, recipe, subs, sub_solvers, full, spaces }
    }

    pub fn nullities(&self) -> Vec<usize> {
        self.sub_solvers.iter().map(|s| s.nullity()).collect()
    }

    /// Price a flip by rank-one algebra (no recomputation of the tensor).
    pub fn incremental(&self, flip: &Flip) -> FlipReport {
        let a_hi = flip.addr >> 4;
        let a_lo = flip.addr & (LOW - 1);
        let o = (flip.col % J) * 7 + flip.bit as usize;
        let d = d_vector(a_lo, o);
        let dch = Problem::chunks(&d);
        let broken_idx: Vec<usize> = (0..self.relations.len()).filter(|&i| self.relations[i].dot(&d)).collect();
        let broken = !broken_idx.is_empty();
        let mut patch_exists = None;
        if broken {
            let r0 = self.relations[broken_idx[0]];
            let c0 = Problem::chunks(&r0);
            let rebased: Vec<B525> = (0..self.relations.len())
                .filter(|&i| i != broken_idx[0])
                .map(|i| if self.relations[i].dot(&d) { self.relations[i] ^ r0 } else { self.relations[i] })
                .collect();
            let rch: Vec<[u64; SLOTS]> = rebased.iter().map(Problem::chunks).collect();
            let mut exists = false;
            for form in &self.problem.forms {
                let e0 = Problem::form_constraint(&c0, form.mask);
                if e0 == 0 {
                    continue;
                }
                let mut cons = Echelon::<1>::new();
                for c in &rch {
                    let e = Problem::form_constraint(c, form.mask);
                    if e != 0 {
                        cons.insert(Bits([e]));
                    }
                }
                if !cons.contains(Bits([e0])) {
                    exists = true;
                    break;
                }
            }
            patch_exists = Some(exists);
        }
        let nullities_after = self
            .subs
            .iter()
            .zip(self.sub_solvers.iter())
            .map(|(sub, solver)| {
                let mut ds = B128::ZERO;
                for (j, &s) in sub.iter().enumerate() {
                    put35(&mut ds.0, j * OUT, dch[s]);
                }
                solver.nullity_after(ds, a_hi)
            })
            .collect();
        FlipReport {
            k: self.problem.k,
            fallback_full: false,
            coded_changed: false,
            broken,
            rebase_ok: true,
            relations_after: self.relations.len() - broken as usize,
            toffolis_fixed_after: self.toffolis + broken as usize,
            patch_exists,
            annihilator_after: self.full.nullity_after(d, a_hi),
            nullities_after,
        }
    }

    /// Price a flip by recomputing the loader from the flipped word table.
    pub fn full(&self, flipped: &Words, coded_cols: &[bool]) -> Result<FlipReport, String> {
        let delta = Delta::build_with_coded(flipped, self.problem.k, coded_cols)?;
        let p2 = Problem::from_delta(&delta, &self.problem.groups);
        let coded_changed = p2.coded != self.problem.coded;
        let broken_idx: Vec<usize> = (0..self.relations.len()).filter(|&i| !p2.is_relation(&self.relations[i])).collect();
        let broken = !broken_idx.is_empty();
        let mut rebase_ok = true;
        let mut patch_exists = None;
        let rebased: Vec<B525> = if broken {
            let r0 = self.relations[broken_idx[0]];
            (0..self.relations.len())
                .filter(|&i| i != broken_idx[0])
                .map(|i| if broken_idx.contains(&i) { self.relations[i] ^ r0 } else { self.relations[i] })
                .collect()
        } else {
            self.relations.clone()
        };
        if !rebased.iter().all(|r| p2.is_relation(r)) {
            rebase_ok = false;
        }
        if broken && rebase_ok {
            // definitional check: a template of the new W' outside the old W = span of the old templates
            let mut w_old = Echelon::<9>::new();
            for (fi, vs) in self.spaces.iter().enumerate() {
                let f = self.problem.forms[fi].mask;
                for &v in vs {
                    w_old.insert(Problem::column(f, v));
                }
            }
            let new_spaces = p2.form_spaces(&rebased);
            let mut exists = false;
            'outer: for (fi, vs) in new_spaces.iter().enumerate() {
                let f = p2.forms[fi].mask;
                for &v in vs {
                    if !w_old.contains(Problem::column(f, v)) {
                        exists = true;
                        break 'outer;
                    }
                }
            }
            patch_exists = Some(exists);
        }
        let nullities_after = self
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
                cols.len() - crate::gf2::rank(&restricted)
            })
            .collect();
        Ok(FlipReport {
            k: self.problem.k,
            fallback_full: true,
            coded_changed,
            broken,
            rebase_ok,
            relations_after: self.relations.len() - broken as usize,
            toffolis_fixed_after: self.toffolis + broken as usize,
            patch_exists,
            annihilator_after: N525 - p2.rank(),
            nullities_after,
        })
    }
}

pub fn sub_cols(sub: &[usize]) -> Vec<usize> {
    let mut cols = Vec::with_capacity(sub.len() * OUT);
    for &s in sub {
        cols.extend(s * OUT..(s + 1) * OUT);
    }
    cols
}

/// Would this flip change the coded status (affine bit 0 on every full block) of its column?
pub fn changes_coded_status(words: &Words, coded_cols: &[bool], flip: &Flip) -> bool {
    if flip.bit != 0 {
        return false;
    }
    if coded_cols[flip.col] {
        return true; // one flip of an affine 16-pattern is never affine; the padding fit changes too
    }
    let full = words.rows - words.rows % LOW;
    if flip.addr >= full {
        return false;
    }
    let b0 = flip.addr - flip.addr % LOW;
    let mut bad = 0;
    for start in (0..full).step_by(LOW) {
        if start != b0 && !words.block_affine(flip.col, start) {
            bad += 1;
            if bad > 0 {
                return false;
            }
        }
    }
    let mut w = words.clone();
    w.flip(flip.addr, flip.col, 0);
    w.block_affine(flip.col, b0)
}

/// The word table with the flip applied.
pub fn apply(words: &Words, flip: &Flip) -> Words {
    let mut w = words.clone();
    w.flip(flip.addr, flip.col, flip.bit);
    w
}

/// Loaders touched by a flip of column `col`.
pub fn loaders_of(col: usize, loaders: usize) -> Vec<usize> {
    let k = col / J;
    let mut out = vec![k];
    if k + 1 < loaders {
        out.push(k + 1);
    }
    out
}

#[allow(dead_code)]
fn _unused(_: u64) -> u64 {
    get35(&[0, 0], 0)
}
