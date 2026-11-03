//! The loader cost model: kernel relations, template spaces, pairing and term extraction. A port of the Python
//! planner (`kr_planner.Problem`, `structured_search`, block draws) to bit-packed GF(2).
//!
//! Loader rows F_h (h = 1..85) live in GF(2)^525 (bit (l - 1) * 35 + o). A relation r has F_h . r = 0 for every h;
//! a relation set R fixes W = R^perp. A template is f (x) v for a low form f (slot mask, <= 2 slots inside one
//! cache phase) and an output direction v (35 bits); V_f is the space of v with f (x) v in W. The plan is a
//! template basis of W; two templates with the same v whose forms share a phase cost one Toffoli together.

use crate::bits::{Bits, B128, B525};
use crate::gf2::{intersect, nullspace, Coords, Echelon};
use crate::loader::{mobius, rows_from_anf, Delta, Words, LOW, N525, OUT, SLOTS};
use crate::rng::Rng;

pub const CACHE_TOFFOLIS: usize = 89; // (15 - 4) low cache + (85 - 7) high cache under Lera's emitter

#[derive(Clone, Debug)]
pub struct Form {
    pub mask: u16,
    pub phases: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Term {
    pub a1: u16,
    pub h1: B128,
    pub a2: u16,
    pub h2: B128,
    pub s: u64,
}

#[derive(Clone, Debug)]
pub struct Pairing {
    pub pairs: Vec<(u16, u16, u64)>,
    pub singles: Vec<(u16, u64)>,
}

impl Pairing {
    pub fn products(&self) -> usize {
        self.pairs.len() + self.singles.len()
    }
    pub fn toffolis(&self) -> usize {
        CACHE_TOFFOLIS + self.products()
    }
}

pub type Spaces = Vec<Vec<u64>>; // per form index: basis of V_f (35-bit vectors)

#[derive(Clone, Debug)]
pub struct Problem {
    pub k: usize,
    pub coded: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    pub rows: Vec<B525>, // F_h, h = 1..=H
    pub ac: Vec<[u64; LOW]>,
    pub h: usize,
    pub phases: Vec<Vec<usize>>, // slots per phase (sorted)
    pub forms: Vec<Form>,
    pub form_index: Vec<i32>, // mask -> form index or -1
    pub planes: Vec<(usize, usize, usize)>, // (form idx, form idx, phase)
    pub max_form: usize,
    pub enum_dim: usize,
}

/// Lera's default low groups for 4 low bits: {3,7,11,15}, {5,6,13,14}, {9,10,12}.
pub fn default_groups() -> Vec<Vec<usize>> {
    vec![vec![3, 7, 11, 15], vec![5, 6, 13, 14], vec![9, 10, 12]]
}

impl Problem {
    pub fn from_words(words: &Words, k: usize, groups: &[Vec<usize>]) -> Result<Self, String> {
        let delta = Delta::build(words, k)?;
        Ok(Self::from_delta(&delta, groups))
    }

    pub fn from_delta(delta: &Delta, groups: &[Vec<usize>]) -> Self {
        let ac = mobius(delta.truth());
        let rows = rows_from_anf(&ac);
        Self::new(delta.k, delta.coded.clone(), groups, rows, ac)
    }

    pub fn new(k: usize, coded: Vec<usize>, groups: &[Vec<usize>], rows: Vec<B525>, ac: Vec<[u64; LOW]>) -> Self {
        let max_form = 2;
        let singles: Vec<usize> = (1..LOW).filter(|m| m & (m - 1) == 0).map(|m| m - 1).collect();
        let phases: Vec<Vec<usize>> = groups
            .iter()
            .map(|g| {
                let mut p: Vec<usize> = singles.iter().copied().chain(g.iter().map(|m| m - 1)).collect();
                p.sort();
                p
            })
            .collect();
        let mut forms: Vec<Form> = Vec::new();
        let mut form_index = vec![-1i32; 1 << SLOTS];
        for (g, slots) in phases.iter().enumerate() {
            let mut fs: Vec<u16> = Vec::new();
            for size in 1..=max_form {
                for c in combinations(slots, size) {
                    fs.push(c.iter().map(|s| 1u16 << s).sum());
                }
            }
            for f in fs {
                if form_index[f as usize] < 0 {
                    form_index[f as usize] = forms.len() as i32;
                    forms.push(Form { mask: f, phases: vec![g] });
                } else {
                    forms[form_index[f as usize] as usize].phases.push(g);
                }
            }
        }
        let mut planes = Vec::new();
        for g in 0..phases.len() {
            let mut fs: Vec<usize> = (0..forms.len()).filter(|&i| forms[i].phases.contains(&g)).collect();
            fs.sort_by_key(|&i| forms[i].mask);
            for i in 0..fs.len() {
                for j in i + 1..fs.len() {
                    planes.push((fs[i], fs[j], g));
                }
            }
        }
        let h = rows.len();
        Problem { k, coded, groups: groups.to_vec(), rows, ac, h, phases, forms, form_index, planes, max_form, enum_dim: 7 }
    }

    pub fn n(&self) -> usize {
        N525
    }

    pub fn form_of(&self, mask: u16) -> Option<usize> {
        let i = self.form_index[mask as usize];
        if i < 0 {
            None
        } else {
            Some(i as usize)
        }
    }

    // ---------------------------------------------------------------- relation helpers

    #[inline]
    pub fn chunks(r: &B525) -> [u64; SLOTS] {
        let mut out = [0u64; SLOTS];
        for (j, slot) in out.iter_mut().enumerate() {
            let base = j * OUT;
            let wi = base >> 6;
            let sh = base & 63;
            let mut w = r.0[wi] >> sh;
            if sh + OUT > 64 {
                w |= r.0[wi + 1] << (64 - sh);
            }
            *slot = w & ((1u64 << OUT) - 1);
        }
        out
    }

    #[inline]
    pub fn form_constraint(chunks: &[u64; SLOTS], form: u16) -> u64 {
        let mut v = 0u64;
        let mut f = form;
        while f != 0 {
            let j = f.trailing_zeros() as usize;
            v ^= chunks[j];
            f &= f - 1;
        }
        v
    }

    /// The template f (x) v as a 525-bit vector.
    #[inline]
    pub fn column(form: u16, v: u64) -> B525 {
        let mut r = B525::ZERO;
        let mut f = form;
        while f != 0 {
            let j = f.trailing_zeros() as usize;
            let base = j * OUT;
            let wi = base >> 6;
            let sh = base & 63;
            r.0[wi] |= v << sh;
            if sh + OUT > 64 {
                r.0[wi + 1] |= v >> (64 - sh);
            }
            f &= f - 1;
        }
        r
    }

    pub fn is_relation(&self, r: &B525) -> bool {
        self.rows.iter().all(|row| !row.dot(r))
    }

    /// Basis of the annihilator of the row space (all relations).
    pub fn annihilator(&self) -> Vec<B525> {
        nullspace(&self.rows, N525)
    }

    pub fn rank(&self) -> usize {
        crate::gf2::rank(&self.rows)
    }

    /// V_f for every form: directions v with f (x) v inside W = R^perp.
    pub fn form_spaces(&self, relations: &[B525]) -> Spaces {
        let ch: Vec<[u64; SLOTS]> = relations.iter().map(Self::chunks).collect();
        self.forms
            .iter()
            .map(|form| {
                let mut cons = Echelon::<1>::new();
                for c in &ch {
                    let e = Self::form_constraint(c, form.mask);
                    if e != 0 {
                        cons.insert(Bits([e]));
                    }
                }
                if cons.is_empty() {
                    (0..OUT as u32).map(|i| 1u64 << i).collect()
                } else if cons.len() >= OUT {
                    Vec::new()
                } else {
                    cons.nullspace(OUT).into_iter().map(|b| b.0[0]).collect()
                }
            })
            .collect()
    }

    /// Dimension of the span of every template in W.
    pub fn span_dim(&self, spaces: &Spaces) -> usize {
        let mut e = Echelon::<9>::new();
        for (fi, vs) in spaces.iter().enumerate() {
            let f = self.forms[fi].mask;
            for &v in vs {
                e.insert(Self::column(f, v));
            }
        }
        e.len()
    }

    pub fn feasible(&self, relations: &[B525]) -> (bool, Spaces) {
        let spaces = self.form_spaces(relations);
        let ok = self.span_dim(&spaces) == N525 - relations.len();
        (ok, spaces)
    }

    /// Basis of the relations supported on the slots `sub` (size <= 3), embedded in GF(2)^525.
    pub fn support_kernel(&self, sub: &[usize]) -> Vec<B525> {
        assert!(sub.len() <= 3);
        let restricted: Vec<B128> = self
            .rows
            .iter()
            .map(|row| {
                let ch = Self::chunks(row);
                let mut v = B128::ZERO;
                for (j, &s) in sub.iter().enumerate() {
                    put35(&mut v.0, j * OUT, ch[s]);
                }
                v
            })
            .collect();
        nullspace(&restricted, sub.len() * OUT)
            .into_iter()
            .map(|r| {
                let mut full = B525::ZERO;
                for (j, &s) in sub.iter().enumerate() {
                    let c = get35(&r.0, j * OUT);
                    full ^= Self::column(1 << s, c);
                }
                full
            })
            .collect()
    }

    /// Slot subsets of `size` that lie inside one phase, each once, in combination order.
    pub fn within_phase_supports(&self, size: usize) -> Vec<Vec<usize>> {
        let all: Vec<usize> = (0..SLOTS).collect();
        combinations(&all, size)
            .into_iter()
            .filter(|sub| self.phases.iter().any(|p| sub.iter().all(|s| p.contains(s))))
            .collect()
    }

    // ---------------------------------------------------------------- pairing

    /// A template basis of W with as many pairs as possible (the Python planner's `pair_basis`).
    pub fn pair_basis(&self, relations: usize, spaces: &Spaces, seed: u64, trials: usize, samples: usize) -> Option<Pairing> {
        let mut rng = Rng::new(seed);
        let target = N525 - relations;
        let l = SLOTS;
        let single_idx: Vec<usize> = (0..l).map(|j| self.form_of(1 << j).unwrap()).collect();
        let mut w0 = Echelon::<9>::new();
        for j in 0..l {
            for &v in &spaces[single_idx[j]] {
                w0.insert(Self::column(1 << j, v));
            }
        }
        let qrank = target - w0.len();
        // common directions of every plane
        struct PlaneDirs {
            f1: u16,
            f2: u16,
            common: Vec<u64>,
        }
        let mut plane_dirs: Vec<PlaneDirs> = Vec::new();
        for &(i1, i2, _g) in &self.planes {
            let v1: Vec<Bits<1>> = spaces[i1].iter().map(|&v| Bits([v])).collect();
            let v2: Vec<Bits<1>> = spaces[i2].iter().map(|&v| Bits([v])).collect();
            let common: Vec<u64> = intersect(&v1, &v2, OUT).into_iter().map(|b| b.0[0]).collect();
            if !common.is_empty() {
                plane_dirs.push(PlaneDirs { f1: self.forms[i1].mask, f2: self.forms[i2].mask, common });
            }
        }
        let (multi, coord): (Vec<&PlaneDirs>, Vec<&PlaneDirs>) =
            plane_dirs.iter().partition(|p| p.f1.count_ones() > 1 || p.f2.count_ones() > 1);
        let enum_dim = self.enum_dim;
        let directions = |common: &[u64], rng: &mut Rng| -> Vec<u64> {
            if common.len() <= enum_dim {
                let mut out = Vec::with_capacity((1 << common.len()) - 1);
                for code in 1u64..(1u64 << common.len()) {
                    let mut v = 0u64;
                    let mut c = code;
                    while c != 0 {
                        v ^= common[c.trailing_zeros() as usize];
                        c &= c - 1;
                    }
                    out.push(v);
                }
                out
            } else {
                let mut out = common.to_vec();
                for _ in 0..samples {
                    let mut c = rng.bits(common.len());
                    let mut v = 0u64;
                    while c != 0 {
                        v ^= common[c.trailing_zeros() as usize];
                        c &= c - 1;
                    }
                    if v != 0 {
                        out.push(v);
                    }
                }
                out
            }
        };
        let mut coord_by_pair: Vec<((usize, usize), Vec<&[u64]>)> = Vec::new();
        for p in &coord {
            let i = p.f1.trailing_zeros() as usize;
            let j = p.f2.trailing_zeros() as usize;
            if let Some(e) = coord_by_pair.iter_mut().find(|e| e.0 == (i, j)) {
                e.1.push(&p.common);
            } else {
                coord_by_pair.push(((i, j), vec![&p.common]));
            }
        }
        let u_len: Vec<usize> = (0..l).map(|j| spaces[single_idx[j]].len()).collect();

        // the visiting order of the multi-slot planes is shuffled in place and persists across trials (as in Python)
        let mut order: Vec<usize> = (0..multi.len()).collect();
        let mut best: Option<Pairing> = None;
        for _trial in 0..trials {
            let mut piv = Echelon::<9>::new();
            let mut qspan = w0.clone();
            let mut local: Vec<Echelon<1>> = (0..l).map(|_| Echelon::new()).collect();
            let mut pairs: Vec<(u16, u16, u64)> = Vec::new();
            // 1. quotient stage
            let mut qdim = 0usize;
            let mut stuck = false;
            // qspan and piv only grow, so a (plane, direction) rejected once stays rejected: a plane whose
            // directions are enumerated exhaustively (dim <= enum_dim) is finished after one pass; sampled planes
            // keep drawing new directions on later passes, as in the Python planner.
            let mut exhausted = vec![false; multi.len()];
            while qdim < qrank && !stuck {
                stuck = true;
                rng.shuffle(&mut order);
                for &pi in order.iter() {
                    if qdim >= qrank {
                        break;
                    }
                    if exhausted[pi] {
                        continue;
                    }
                    let p = multi[pi];
                    if p.common.len() <= enum_dim {
                        exhausted[pi] = true;
                    }
                    for v in directions(&p.common, &mut rng) {
                        let a = Self::column(p.f1, v);
                        let b = Self::column(p.f2, v);
                        let ar_q = qspan.reduce(a);
                        let mut gain = 0;
                        if !ar_q.is_zero() {
                            gain += 1;
                            qspan.push_reduced(ar_q);
                        }
                        let br_q = qspan.reduce(b);
                        if !br_q.is_zero() {
                            gain += 1;
                        }
                        if !ar_q.is_zero() {
                            qspan.pop();
                        }
                        if gain == 0 {
                            continue;
                        }
                        let ar = piv.reduce(a);
                        if ar.is_zero() {
                            continue;
                        }
                        piv.push_reduced(ar);
                        let br = piv.reduce(b);
                        if br.is_zero() {
                            piv.pop();
                            continue;
                        }
                        piv.push_reduced(br);
                        qspan.insert(a);
                        qspan.insert(b);
                        pairs.push((p.f1, p.f2, v));
                        for f in [p.f1, p.f2] {
                            if f.count_ones() == 1 {
                                local[f.trailing_zeros() as usize].insert(Bits([v]));
                            }
                        }
                        qdim += gain;
                        stuck = false;
                        if qdim >= qrank {
                            break;
                        }
                    }
                }
            }
            // 2. coordinate stage with balanced deficits
            let mut progress = true;
            while progress {
                progress = false;
                let deficit: Vec<isize> = (0..l).map(|j| u_len[j] as isize - local[j].len() as isize).collect();
                let mut options: Vec<(isize, isize, f64, usize)> = Vec::new();
                for (oi, ((i, j), _)) in coord_by_pair.iter().enumerate() {
                    if deficit[*i] > 0 && deficit[*j] > 0 {
                        options.push((deficit[*i] + deficit[*j], deficit[*i].min(deficit[*j]), rng.f64(), oi));
                    }
                }
                options.sort_by(|x, y| y.0.cmp(&x.0).then(y.1.cmp(&x.1)).then(y.2.partial_cmp(&x.2).unwrap()));
                'opts: for &(_, _, _, oi) in &options {
                    let ((i, j), commons) = &coord_by_pair[oi];
                    let (i, j) = (*i, *j);
                    for common in commons.iter() {
                        let b1 = common.iter().copied().find(|&v| !local[i].reduce(Bits([v])).is_zero());
                        let b2 = common.iter().copied().find(|&v| !local[j].reduce(Bits([v])).is_zero());
                        let mut cands: Vec<u64> = Vec::new();
                        if let (Some(b1), Some(b2)) = (b1, b2) {
                            cands.extend([b1, b2, b1 ^ b2]);
                        }
                        cands.extend(directions(common, &mut rng));
                        for v in cands {
                            if v == 0 || local[i].reduce(Bits([v])).is_zero() || local[j].reduce(Bits([v])).is_zero() {
                                continue;
                            }
                            let a = Self::column(1 << i, v);
                            let b = Self::column(1 << j, v);
                            let ar = piv.reduce(a);
                            if ar.is_zero() {
                                continue;
                            }
                            piv.push_reduced(ar);
                            let br = piv.reduce(b);
                            if br.is_zero() {
                                piv.pop();
                                continue;
                            }
                            piv.push_reduced(br);
                            local[i].insert(Bits([v]));
                            local[j].insert(Bits([v]));
                            pairs.push((1 << i, 1 << j, v));
                            progress = true;
                            break 'opts;
                        }
                    }
                }
            }
            // 3. singles for whatever is left
            let mut singles: Vec<(u16, u64)> = Vec::new();
            if piv.len() < target {
                let mut order: Vec<usize> = (0..self.forms.len()).collect();
                rng.shuffle(&mut order);
                'forms: for fi in order {
                    let f = self.forms[fi].mask;
                    for &v in &spaces[fi] {
                        if piv.insert(Self::column(f, v)) {
                            singles.push((f, v));
                        }
                        if piv.len() == target {
                            break 'forms;
                        }
                    }
                }
            }
            if piv.len() != target {
                continue;
            }
            let cost = pairs.len() + singles.len();
            let perfect = 2 * pairs.len() + singles.len() == target && singles.len() <= target % 2;
            if best.as_ref().map_or(true, |b| cost < b.products()) {
                best = Some(Pairing { pairs, singles });
            }
            if perfect {
                break;
            }
        }
        best
    }

    /// Lera's (a1, b1, a2, b2, s) terms from a pairing; exact reconstruction asserted.
    pub fn terms(&self, pairing: &Pairing) -> Result<Vec<Term>, String> {
        let mut basis = Coords::<9, 9>::new();
        enum Col {
            Pair(u16, u16, u64),
            Single(u16, u64),
        }
        let mut cols = Vec::new();
        for &(f1, f2, v) in &pairing.pairs {
            if !basis.add(Self::column(f1, v)) || !basis.add(Self::column(f2, v)) {
                return Err("pair templates dependent".into());
            }
            cols.push(Col::Pair(f1, f2, v));
        }
        for &(f, v) in &pairing.singles {
            if !basis.add(Self::column(f, v)) {
                return Err("single template dependent".into());
            }
            cols.push(Col::Single(f, v));
        }
        let sols: Vec<Bits<9>> = self
            .rows
            .iter()
            .map(|row| basis.solve(*row).ok_or_else(|| "rows outside the template span".to_string()))
            .collect::<Result<_, _>>()?;
        let hmask = |idx: usize| -> B128 {
            let mut h = B128::ZERO;
            for (hi, s) in sols.iter().enumerate() {
                if s.get(idx) {
                    h.set(hi);
                }
            }
            h
        };
        let mut out = Vec::with_capacity(cols.len());
        let mut idx = 0;
        for c in &cols {
            match *c {
                Col::Pair(f1, f2, v) => {
                    out.push(Term { a1: f1, h1: hmask(idx), a2: f2, h2: hmask(idx + 1), s: v });
                    idx += 2;
                }
                Col::Single(f, v) => {
                    out.push(Term { a1: f, h1: hmask(idx), a2: 0, h2: B128::ZERO, s: v });
                    idx += 1;
                }
            }
        }
        if !self.reconstructs(&out) {
            return Err("reconstruction failed".into());
        }
        Ok(out)
    }

    /// Do the terms reproduce every row exactly?
    pub fn reconstructs(&self, terms: &[Term]) -> bool {
        let mut recon = vec![B525::ZERO; self.h];
        for t in terms {
            let c1 = Self::column(t.a1, t.s);
            for hi in t.h1.ones() {
                if hi >= self.h {
                    return false;
                }
                recon[hi] ^= c1;
            }
            if t.a2 != 0 {
                let c2 = Self::column(t.a2, t.s);
                for hi in t.h2.ones() {
                    if hi >= self.h {
                        return false;
                    }
                    recon[hi] ^= c2;
                }
            }
        }
        recon == self.rows
    }

    // ---------------------------------------------------------------- relation draws

    /// Single-slot relations (the coded bit-0 structure), independent, in slot order.
    pub fn single_slot_relations(&self) -> Vec<B525> {
        let mut out = Vec::new();
        let mut piv = Echelon::<9>::new();
        for s in 0..SLOTS {
            for r in self.support_kernel(&[s]) {
                if piv.insert(r) {
                    out.push(r);
                }
            }
        }
        out
    }

    /// `count` relations: single-slot ones first, then random kernel vectors of within-phase slot triples drawn
    /// round-robin by least slot usage (the Python planner's `structured_search(within_phase=True)`).
    pub fn structured_draw(&self, count: usize, seed: u64) -> (Vec<B525>, bool, Spaces) {
        let mut rng = Rng::new(seed);
        let supports = self.within_phase_supports(3);
        let kernels: Vec<Vec<B525>> = supports.iter().map(|s| self.support_kernel(s)).collect();
        let mut piv = Echelon::<9>::new();
        let mut relations = Vec::new();
        for r in self.single_slot_relations() {
            if piv.insert(r) {
                relations.push(r);
            }
        }
        let mut usage = vec![0usize; SLOTS];
        let mut guard = 0;
        while relations.len() < count && guard < 50 * count {
            guard += 1;
            let mut scored: Vec<(usize, f64, usize)> = supports
                .iter()
                .enumerate()
                .map(|(i, s)| (s.iter().map(|&j| usage[j]).sum(), rng.f64(), i))
                .collect();
            scored.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.partial_cmp(&y.1).unwrap()));
            let si = if rng.f64() < 0.7 { scored[0].2 } else { rng.below(supports.len()) };
            let ker = &kernels[si];
            if ker.is_empty() {
                continue;
            }
            let v = random_combination(ker, &mut rng);
            if !v.is_zero() && piv.insert(v) {
                relations.push(v);
                for &j in &supports[si] {
                    usage[j] += 1;
                }
            }
        }
        let (ok, spaces) = self.feasible(&relations);
        (relations, ok, spaces)
    }

    /// Block design: single-slot relations, then `per` independent random kernel vectors from each triple of the
    /// design (the HW-13 recipe for the generic loaders, `s2_block.run_design` with no top-up).
    pub fn block_draw(&self, design: &[Vec<usize>], per: usize, seed: u64) -> (Vec<B525>, bool, Spaces) {
        let mut rng = Rng::new(seed);
        let mut piv = Echelon::<9>::new();
        let mut relations = Vec::new();
        for r in self.single_slot_relations() {
            if piv.insert(r) {
                relations.push(r);
            }
        }
        for t in design {
            let ker = self.support_kernel(t);
            let want = per.min(ker.len());
            let mut got = 0;
            let mut guard = 0;
            while got < want && guard < 1000 {
                guard += 1;
                let v = random_combination(&ker, &mut rng);
                if !v.is_zero() && piv.insert(v) {
                    relations.push(v);
                    got += 1;
                }
            }
        }
        let (ok, spaces) = self.feasible(&relations);
        (relations, ok, spaces)
    }
}

fn random_combination(ker: &[B525], rng: &mut Rng) -> B525 {
    let mut v = B525::ZERO;
    let mut i = 0;
    while i < ker.len() {
        let take = (ker.len() - i).min(64);
        let mut code = rng.bits(take);
        while code != 0 {
            v ^= ker[i + code.trailing_zeros() as usize];
            code &= code - 1;
        }
        i += take;
    }
    v
}

#[inline]
pub fn put35(words: &mut [u64], base: usize, v: u64) {
    let wi = base >> 6;
    let sh = base & 63;
    words[wi] |= v << sh;
    if sh + OUT > 64 {
        words[wi + 1] |= v >> (64 - sh);
    }
}

#[inline]
pub fn get35(words: &[u64], base: usize) -> u64 {
    let wi = base >> 6;
    let sh = base & 63;
    let mut w = words[wi] >> sh;
    if sh + OUT > 64 {
        w |= words[wi + 1] << (64 - sh);
    }
    w & ((1u64 << OUT) - 1)
}

pub fn combinations(items: &[usize], size: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut cur = Vec::with_capacity(size);
    fn rec(items: &[usize], size: usize, start: usize, cur: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if cur.len() == size {
            out.push(cur.clone());
            return;
        }
        for i in start..items.len() {
            cur.push(items[i]);
            rec(items, size, i + 1, cur, out);
            cur.pop();
        }
    }
    rec(items, size, 0, &mut cur, &mut out);
    out
}

/// Verification of a stored plan against the loader: relation admissibility, span feasibility, template
/// independence, pairing validity (same direction, forms sharing a phase, forms in the <= 2-slot class) and exact
/// reconstruction.
#[derive(Clone, Debug, Default)]
pub struct PlanCheck {
    pub relations: usize,
    pub relations_independent: bool,
    pub all_relations: bool,
    pub span_feasible: bool,
    pub templates_in_w: bool,
    pub templates_independent: bool,
    pub pairs: usize,
    pub singles: usize,
    pub pairs_valid: bool,
    pub forms_valid: bool,
    pub reconstructs: bool,
    pub toffolis: usize,
}

impl PlanCheck {
    pub fn ok(&self) -> bool {
        self.relations_independent
            && self.all_relations
            && self.span_feasible
            && self.templates_in_w
            && self.templates_independent
            && self.pairs_valid
            && self.forms_valid
            && self.reconstructs
    }
}

impl Problem {
    pub fn check_plan(&self, relations: &[B525], terms: &[Term]) -> PlanCheck {
        let mut c = PlanCheck { relations: relations.len(), ..Default::default() };
        c.relations_independent = crate::gf2::rank(relations) == relations.len();
        c.all_relations = relations.iter().all(|r| self.is_relation(r));
        let (ok, _spaces) = self.feasible(relations);
        c.span_feasible = ok;
        let mut basis = Echelon::<9>::new();
        c.templates_in_w = true;
        c.templates_independent = true;
        c.pairs_valid = true;
        c.forms_valid = true;
        for t in terms {
            let mut templates = vec![(t.a1, t.s)];
            if t.a2 != 0 {
                templates.push((t.a2, t.s));
                c.pairs += 1;
                let (f1, f2) = (self.form_of(t.a1), self.form_of(t.a2));
                match (f1, f2) {
                    (Some(i1), Some(i2)) => {
                        let shared = self.forms[i1].phases.iter().any(|g| self.forms[i2].phases.contains(g));
                        if !shared {
                            c.pairs_valid = false;
                        }
                    }
                    _ => c.forms_valid = false,
                }
            } else {
                c.singles += 1;
                if self.form_of(t.a1).is_none() {
                    c.forms_valid = false;
                }
            }
            for (f, v) in templates {
                let col = Self::column(f, v);
                if relations.iter().any(|r| r.dot(&col)) {
                    c.templates_in_w = false;
                }
                if !basis.insert(col) {
                    c.templates_independent = false;
                }
            }
        }
        if basis.len() != N525 - relations.len() {
            c.templates_independent = false;
        }
        c.reconstructs = self.reconstructs(terms);
        c.toffolis = CACHE_TOFFOLIS + terms.len();
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_and_column_roundtrip() {
        let mut r = B525::ZERO;
        r.set(0);
        r.set(34);
        r.set(35);
        r.set(14 * 35 + 34);
        let ch = Problem::chunks(&r);
        assert_eq!(ch[0], (1 << 34) | 1);
        assert_eq!(ch[1], 1);
        assert_eq!(ch[14], 1 << 34);
        let c = Problem::column(0b11, 1 << 34);
        let ch2 = Problem::chunks(&c);
        assert_eq!(ch2[0], 1 << 34);
        assert_eq!(ch2[1], 1 << 34);
        assert_eq!(c.count_ones(), 2);
    }

    #[test]
    fn forms_and_planes_match_python_counts() {
        // Lera's layout: phases of 8, 8, 7 slots; forms <= 2 slots; planes per phase C(F_g, 2)
        let p = Problem::new(0, vec![], &default_groups(), vec![B525::ZERO; 85], vec![[0; LOW]; 86]);
        assert_eq!(p.phases.iter().map(|x| x.len()).collect::<Vec<_>>(), vec![8, 8, 7]);
        // forms: 15 singles + pairs within phases (28 + 28 + 21 minus the 6 single-bit pairs counted thrice)
        assert_eq!(p.forms.len(), 15 + 28 + 28 + 21 - 12);
        let per_phase = [8 + 28, 8 + 28, 7 + 21];
        let planes: usize = per_phase.iter().map(|f| f * (f - 1) / 2).sum();
        assert_eq!(p.planes.len(), planes);
    }
}
