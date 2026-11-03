//! Template-basis plans with pairing bookkeeping and basis-exchange defect repair (a port of `qloop.Plan`).
//!
//! A plan is a basis of W made of templates (f, v). A template is a defect when it cannot be paired (same v, both
//! forms in one phase). One- and two-template exchanges replace defects by templates that keep the basis property
//! (the coefficient minor is nonzero) and lower the defect count.

use std::collections::HashMap;
use std::time::Instant;

use crate::bits::{Bits, B525};
use crate::gf2::{intersect, Coords};
use crate::problem::{Pairing, Problem, Spaces};
use crate::rng::Rng;

pub struct Plan<'a> {
    p: &'a Problem,
    spaces: &'a Spaces,
    rng: Rng,
    nph: usize,
    pub b: Vec<(u16, u64)>,
    pub target: usize,
    vpiv: Vec<Coords<1, 1>>,
    t: Vec<Vec<Bits<9>>>,
    index: HashMap<(u16, u64), usize>,
}

impl<'a> Plan<'a> {
    pub fn new(p: &'a Problem, relations: usize, spaces: &'a Spaces, pairing: &Pairing, seed: u64) -> Result<Self, String> {
        let mut b = Vec::new();
        for &(f1, f2, v) in &pairing.pairs {
            b.push((f1, v));
            b.push((f2, v));
        }
        for &(f, v) in &pairing.singles {
            b.push((f, v));
        }
        let target = p.n() - relations;
        if b.len() != target {
            return Err(format!("plan has {} templates, W has dimension {target}", b.len()));
        }
        let mut vpiv = Vec::with_capacity(spaces.len());
        for vs in spaces {
            let mut c = Coords::<1, 1>::new();
            for &v in vs {
                if !c.add(Bits([v])) {
                    return Err("V_f basis dependent".into());
                }
            }
            vpiv.push(c);
        }
        let mut plan = Plan { p, spaces, rng: Rng::new(seed), nph: p.phases.len(), b, target, vpiv, t: Vec::new(), index: HashMap::new() };
        plan.rebuild()?;
        Ok(plan)
    }

    fn rebuild(&mut self) -> Result<(), String> {
        let mut basis = Coords::<9, 9>::new();
        for &(f, v) in &self.b {
            if !basis.add(Problem::column(f, v)) {
                return Err("plan templates dependent".into());
            }
        }
        self.t = Vec::with_capacity(self.spaces.len());
        for (fi, vs) in self.spaces.iter().enumerate() {
            let f = self.p.forms[fi].mask;
            let mut cs = Vec::with_capacity(vs.len());
            for &w in vs {
                cs.push(basis.solve(Problem::column(f, w)).ok_or("V_f template outside the plan span")?);
            }
            self.t.push(cs);
        }
        self.reindex();
        Ok(())
    }

    fn reindex(&mut self) {
        self.index = self.b.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    }

    fn coords(&self, fi: usize, v: u64) -> Option<Bits<9>> {
        let beta = self.vpiv[fi].solve(Bits([v]))?;
        let mut c = Bits::<9>::ZERO;
        let mut m = beta.0[0];
        while m != 0 {
            c ^= self.t[fi][m.trailing_zeros() as usize];
            m &= m - 1;
        }
        Some(c)
    }

    fn replace(&mut self, idx: usize, f: u16, v: u64, c: Bits<9>) {
        debug_assert!(c.get(idx));
        let mut mask = c;
        mask.clear(idx);
        for cs in self.t.iter_mut() {
            for x in cs.iter_mut() {
                if x.get(idx) {
                    *x ^= mask;
                }
            }
        }
        self.b[idx] = (f, v);
        self.reindex();
    }

    fn is_single_form(&self, f: u16) -> bool {
        self.p.forms[self.p.form_of(f).unwrap()].phases.len() == self.nph
    }

    fn phase_of(&self, f: u16) -> usize {
        self.p.forms[self.p.form_of(f).unwrap()].phases[0]
    }

    fn groups_by_v(b: &[(u16, u64)]) -> HashMap<u64, Vec<usize>> {
        let mut by: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, &(_, v)) in b.iter().enumerate() {
            by.entry(v).or_default().push(i);
        }
        by
    }

    fn defects_of(&self, idxs: &[usize], b: &[(u16, u64)]) -> isize {
        let mut n = vec![0isize; self.nph];
        let mut s = 0isize;
        for &i in idxs {
            let f = b[i].0;
            if self.is_single_form(f) {
                s += 1;
            } else {
                n[self.phase_of(f)] += 1;
            }
        }
        let odd: isize = n.iter().filter(|x| *x & 1 == 1).count() as isize;
        let total: isize = n.iter().sum::<isize>() + s;
        (odd - s).max(total & 1)
    }

    pub fn defects(&self) -> (isize, HashMap<u64, Vec<usize>>) {
        let by = Self::groups_by_v(&self.b);
        let d = by.values().map(|ix| self.defects_of(ix, &self.b)).sum();
        (d, by)
    }

    fn removable(&self) -> (isize, Vec<usize>, HashMap<u64, Vec<usize>>) {
        let (d_total, by) = self.defects();
        let mut out = Vec::new();
        let mut keys: Vec<&u64> = by.keys().collect();
        keys.sort();
        for v in keys {
            let ix = &by[v];
            let d = self.defects_of(ix, &self.b);
            if d == 0 {
                continue;
            }
            for &i in ix {
                let rest: Vec<usize> = ix.iter().copied().filter(|&j| j != i).collect();
                if self.defects_of(&rest, &self.b) == d - 1 {
                    out.push(i);
                }
            }
        }
        (d_total, out, by)
    }

    fn delta_after(&self, remove: &[usize], add: &[(u16, u64)]) -> isize {
        let mut dirs: Vec<u64> = remove.iter().map(|&i| self.b[i].1).chain(add.iter().map(|t| t.1)).collect();
        dirs.sort();
        dirs.dedup();
        let by = Self::groups_by_v(&self.b);
        let empty = Vec::new();
        let before: isize = dirs.iter().map(|v| self.defects_of(by.get(v).unwrap_or(&empty), &self.b)).sum();
        let mut new_b = self.b.clone();
        for (&i, &t) in remove.iter().zip(add.iter()) {
            new_b[i] = t;
        }
        let by2 = Self::groups_by_v(&new_b);
        let after: isize = dirs.iter().map(|v| self.defects_of(by2.get(v).unwrap_or(&empty), &new_b)).sum();
        after - before
    }

    fn plane_common(&self) -> Vec<(usize, usize, Vec<u64>)> {
        let mut out = Vec::new();
        for &(i1, i2, _g) in &self.p.planes {
            let v1: Vec<Bits<1>> = self.spaces[i1].iter().map(|&v| Bits([v])).collect();
            let v2: Vec<Bits<1>> = self.spaces[i2].iter().map(|&v| Bits([v])).collect();
            let common: Vec<u64> = intersect(&v1, &v2, crate::loader::OUT).into_iter().map(|b| b.0[0]).collect();
            if !common.is_empty() {
                out.push((i1, i2, common));
            }
        }
        out
    }

    /// Exchange repair; returns the remaining defect count.
    pub fn repair(&mut self, time_budget_s: f64) -> isize {
        let t0 = Instant::now();
        let mut planes = self.plane_common();
        let (mut d, mut rem, mut by) = self.removable();
        let parity = (self.target % 2) as isize;
        while d > parity && t0.elapsed().as_secs_f64() < time_budget_s {
            let mut rem_bit: HashMap<usize, usize> = HashMap::new();
            for (n, &i) in rem.iter().enumerate() {
                rem_bit.insert(i, n);
            }
            let mut improved = false;
            // 1-exchange
            let mut defect_dirs: Vec<u64> = by.iter().filter(|(_, ix)| self.defects_of(ix, &self.b) > 0).map(|(v, _)| *v).collect();
            defect_dirs.sort();
            self.rng.shuffle(&mut defect_dirs);
            'one: for &v in &defect_dirs {
                let mut forms: Vec<usize> = (0..self.p.forms.len()).collect();
                self.rng.shuffle(&mut forms);
                for fi in forms {
                    let f = self.p.forms[fi].mask;
                    if self.index.contains_key(&(f, v)) {
                        continue;
                    }
                    let c = match self.coords(fi, v) {
                        Some(c) => c,
                        None => continue,
                    };
                    let mut hits: Vec<usize> = rem.iter().copied().filter(|&i| c.get(i)).collect();
                    self.rng.shuffle(&mut hits);
                    for &i in hits.iter().take(6) {
                        if self.delta_after(&[i], &[(f, v)]) <= -2 {
                            self.replace(i, f, v, c);
                            improved = true;
                            break 'one;
                        }
                    }
                }
            }
            if !improved {
                self.rng.shuffle(&mut planes);
                'two: for (i1, i2, common) in planes.iter() {
                    let (f1, f2) = (self.p.forms[*i1].mask, self.p.forms[*i2].mask);
                    let c1: Vec<Bits<9>> = common.iter().map(|&w| self.coords(*i1, w).expect("common in V_f1")).collect();
                    let c2: Vec<Bits<9>> = common.iter().map(|&w| self.coords(*i2, w).expect("common in V_f2")).collect();
                    let proj = |c: &Bits<9>| -> Bits<9> {
                        let mut m = Bits::<9>::ZERO;
                        for (&i, &n) in rem_bit.iter() {
                            if c.get(i) {
                                m.set(n);
                            }
                        }
                        m
                    };
                    let m1: Vec<Bits<9>> = c1.iter().map(proj).collect();
                    let m2: Vec<Bits<9>> = c2.iter().map(proj).collect();
                    if m1.iter().all(|x| x.is_zero()) && m2.iter().all(|x| x.is_zero()) {
                        continue;
                    }
                    let n = common.len();
                    let codes: Vec<u64> = if n <= 8 {
                        (1..(1u64 << n)).collect()
                    } else {
                        let mut c: Vec<u64> = (0..n).map(|i| 1u64 << i).collect();
                        for _ in 0..48 {
                            let r = self.rng.bits(n);
                            c.push(if r == 0 { 1 } else { r });
                        }
                        c
                    };
                    for code in codes {
                        let mut a = Bits::<9>::ZERO;
                        let mut bb = Bits::<9>::ZERO;
                        let mut v = 0u64;
                        let mut cc1 = Bits::<9>::ZERO;
                        let mut cc2 = Bits::<9>::ZERO;
                        let mut cm = code;
                        while cm != 0 {
                            let i = cm.trailing_zeros() as usize;
                            a ^= m1[i];
                            bb ^= m2[i];
                            v ^= common[i];
                            cc1 ^= c1[i];
                            cc2 ^= c2[i];
                            cm &= cm - 1;
                        }
                        if a.is_zero() || bb.is_zero() || a == bb {
                            continue;
                        }
                        let a_not_b = {
                            let mut x = a;
                            for i in 0..9 {
                                x.0[i] &= !bb.0[i];
                            }
                            x
                        };
                        let (t1, t2) = if !a_not_b.is_zero() {
                            (rem[a_not_b.leading().unwrap()], rem[bb.leading().unwrap()])
                        } else {
                            let mut b_not_a = bb;
                            for i in 0..9 {
                                b_not_a.0[i] &= !a.0[i];
                            }
                            (rem[a.leading().unwrap()], rem[b_not_a.leading().unwrap()])
                        };
                        if self.index.contains_key(&(f1, v)) || self.index.contains_key(&(f2, v)) {
                            continue;
                        }
                        if self.delta_after(&[t1, t2], &[(f1, v), (f2, v)]) <= -2 {
                            self.replace(t1, f1, v, cc1);
                            let mut cc2 = cc2;
                            if cc2.get(t1) {
                                let mut adj = cc1;
                                adj.clear(t1);
                                cc2 ^= adj;
                            }
                            assert!(cc2.get(t2));
                            self.replace(t2, f2, v, cc2);
                            improved = true;
                            break 'two;
                        }
                    }
                }
            }
            if !improved {
                break;
            }
            let r = self.removable();
            d = r.0;
            rem = r.1;
            by = r.2;
        }
        d
    }

    /// (pairs, singles) from the basis.
    pub fn pairing(&self) -> Pairing {
        let mut pairs = Vec::new();
        let mut singles = Vec::new();
        let by = Self::groups_by_v(&self.b);
        let mut keys: Vec<&u64> = by.keys().collect();
        keys.sort();
        for v in keys {
            let ix = &by[v];
            let mut per: Vec<Vec<u16>> = vec![Vec::new(); self.nph];
            let mut sing: Vec<u16> = Vec::new();
            for &i in ix {
                let f = self.b[i].0;
                if self.is_single_form(f) {
                    sing.push(f);
                } else {
                    per[self.phase_of(f)].push(f);
                }
            }
            for g in 0..self.nph {
                while per[g].len() >= 2 {
                    let a = per[g].pop().unwrap();
                    let b = per[g].pop().unwrap();
                    pairs.push((a, b, *v));
                }
                if let Some(a) = per[g].pop() {
                    if let Some(s) = sing.pop() {
                        pairs.push((a, s, *v));
                    } else {
                        singles.push((a, *v));
                    }
                }
            }
            while sing.len() >= 2 {
                let a = sing.pop().unwrap();
                let b = sing.pop().unwrap();
                pairs.push((a, b, *v));
            }
            for f in sing {
                singles.push((f, *v));
            }
        }
        Pairing { pairs, singles }
    }
}

/// Pair a relation set, repair, and extract terms: the HW-13 `plan_once`.
pub fn plan_once(p: &Problem, relations: &[B525], spaces: &Spaces, seed: u64, trials: usize, samples: usize, repair_s: f64)
    -> Option<(Pairing, usize, isize)> {
    let pairing = p.pair_basis(relations.len(), spaces, seed, trials, samples)?;
    let before = pairing.singles.len();
    if repair_s <= 0.0 || before <= (p.n() - relations.len()) % 2 {
        return Some((pairing, before, before as isize));
    }
    let mut plan = Plan::new(p, relations.len(), spaces, &pairing, seed).ok()?;
    let d = plan.repair(repair_s);
    let pairing = plan.pairing();
    Some((pairing, before, d))
}
