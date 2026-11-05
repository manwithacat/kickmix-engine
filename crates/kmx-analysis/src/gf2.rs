//! GF(2) echelon objects with runtime widths (engineering card E2, P2 primitive, 2026-11-05).
//!
//! Generalised from the engine's `crates/qrom-plan/src/gf2.rs` (const-generic widths) to widths known only at run
//! time. The Python reference is `tools/dfthc-refit/recover.py` in the competition repository (`Basis`, `rank`,
//! `nullspace`) and `planner_shared.intersect` / `decompose`; pivot choice is the same (the highest set bit of the
//! partially reduced vector; stored rows are not mutually reduced), so every reduced vector and every combination
//! mask equals the Python one bit for bit.
//!
//! Vectors are packed little-endian `u64` words (bit i of the Python int = bit i % 64 of word i / 64).

/// Number of u64 words for `bits` bits (at least one).
#[inline]
pub fn words_for(bits: usize) -> usize {
    bits.div_ceil(64).max(1)
}

/// Highest set bit of a packed vector, or None if zero.
#[inline]
pub fn leading(v: &[u64]) -> Option<usize> {
    for i in (0..v.len()).rev() {
        let w = v[i];
        if w != 0 {
            return Some(i * 64 + 63 - w.leading_zeros() as usize);
        }
    }
    None
}

#[inline]
fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src) {
        *d ^= *s;
    }
}

#[inline]
fn get_bit(v: &[u64], i: usize) -> bool {
    (v[i / 64] >> (i % 64)) & 1 == 1
}

#[inline]
fn set_bit(v: &mut [u64], i: usize) {
    v[i / 64] |= 1u64 << (i % 64);
}

#[inline]
fn flip_bit(v: &mut [u64], i: usize) {
    v[i / 64] ^= 1u64 << (i % 64);
}

#[inline]
pub fn is_zero(v: &[u64]) -> bool {
    v.iter().all(|&w| w == 0)
}

/// Incremental echelon basis with combination tracking (`recover.Basis`). `width` bounds the vectors; the
/// combination masks have `width` bits too (at most `width` independent vectors).
#[derive(Clone, Debug)]
pub struct CoordBasis {
    pub width: usize,
    pub nw: usize,
    pub cw: usize,
    vals: Vec<u64>,   // rows * nw: reduced row values, in insertion order
    coords: Vec<u64>, // rows * cw: combination masks over the added vectors
    orig: Vec<u64>,   // rows * nw: the added vectors as given (`Basis.vectors`)
    piv: Vec<u32>,    // pivot bit -> row index + 1 (0 = none)
    rows: usize,
    tv: Vec<u64>, // scratch
    tc: Vec<u64>, // scratch
}

impl CoordBasis {
    pub fn new(width: usize) -> Self {
        let nw = words_for(width);
        let cw = words_for(width);
        CoordBasis {
            width,
            nw,
            cw,
            vals: Vec::new(),
            coords: Vec::new(),
            orig: Vec::new(),
            piv: vec![0; nw * 64],
            rows: 0,
            tv: vec![0; nw],
            tc: vec![0; cw],
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn clear(&mut self) {
        for r in 0..self.rows {
            let p = leading(&self.vals[r * self.nw..(r + 1) * self.nw]).unwrap();
            self.piv[p] = 0;
        }
        self.vals.clear();
        self.coords.clear();
        self.orig.clear();
        self.rows = 0;
    }

    /// Drop every row added after the first `n` (the rows are in insertion order).
    pub fn truncate(&mut self, n: usize) {
        while self.rows > n {
            let r = self.rows - 1;
            let p = leading(&self.vals[r * self.nw..(r + 1) * self.nw]).unwrap();
            self.piv[p] = 0;
            self.rows -= 1;
            self.vals.truncate(self.rows * self.nw);
            self.coords.truncate(self.rows * self.cw);
            self.orig.truncate(self.rows * self.nw);
        }
    }

    pub fn vector(&self, i: usize) -> &[u64] {
        &self.orig[i * self.nw..(i + 1) * self.nw]
    }

    /// `Basis.reduce`: (v reduced while its highest bit is a pivot, combination mask). Mutates `v` and `c`.
    #[inline]
    pub fn reduce_into(&self, v: &mut [u64], c: &mut [u64]) {
        let nw = self.nw;
        let cw = self.cw;
        let mut top = nw;
        loop {
            while top > 0 && v[top - 1] == 0 {
                top -= 1;
            }
            if top == 0 {
                return;
            }
            let p = (top - 1) * 64 + 63 - v[top - 1].leading_zeros() as usize;
            let r = self.piv[p];
            if r == 0 {
                return;
            }
            let r = (r - 1) as usize;
            xor_into(&mut v[..top], &self.vals[r * nw..r * nw + top]);
            xor_into(c, &self.coords[r * cw..(r + 1) * cw]);
        }
    }

    /// Reduce without tracking coordinates.
    #[inline]
    pub fn reduce_value(&self, v: &mut [u64]) {
        let nw = self.nw;
        let mut top = nw;
        loop {
            while top > 0 && v[top - 1] == 0 {
                top -= 1;
            }
            if top == 0 {
                return;
            }
            let p = (top - 1) * 64 + 63 - v[top - 1].leading_zeros() as usize;
            let r = self.piv[p];
            if r == 0 {
                return;
            }
            let r = (r - 1) as usize;
            xor_into(&mut v[..top], &self.vals[r * nw..r * nw + top]);
        }
    }

    /// True iff `v` lies in the span (`Basis.solve(v) is not None`).
    #[inline]
    pub fn contains(&self, v: &[u64]) -> bool {
        let mut t = v.to_vec();
        self.reduce_value(&mut t);
        is_zero(&t)
    }

    /// `Basis.solve`: combination mask, or None outside the span.
    pub fn solve(&self, v: &[u64]) -> Option<Vec<u64>> {
        let mut t = v.to_vec();
        let mut c = vec![0u64; self.cw];
        self.reduce_into(&mut t, &mut c);
        if is_zero(&t) {
            Some(c)
        } else {
            None
        }
    }

    /// `Basis.add`: true iff independent (then stored with the next index).
    pub fn add(&mut self, v: &[u64]) -> bool {
        let mut t = std::mem::take(&mut self.tv);
        let mut c = std::mem::take(&mut self.tc);
        t.copy_from_slice(&v[..self.nw]);
        c.iter_mut().for_each(|x| *x = 0);
        self.reduce_into(&mut t, &mut c);
        let out = match leading(&t) {
            None => false,
            Some(p) => {
                assert!(self.rows < self.cw * 64, "more independent vectors than the coordinate width");
                flip_bit(&mut c, self.rows);
                self.vals.extend_from_slice(&t);
                self.coords.extend_from_slice(&c);
                self.orig.extend_from_slice(&v[..self.nw]);
                self.rows += 1;
                self.piv[p] = self.rows as u32;
                true
            }
        };
        self.tv = t;
        self.tc = c;
        out
    }

    /// Reduce `v` (value only) and, if nonzero, store it as a temporary row with zero coordinates (removed with
    /// `truncate`); true iff independent. Only for independence tests: such rows must not be solved against.
    #[inline]
    pub fn push_temp(&mut self, v: &[u64]) -> bool {
        let mut t = std::mem::take(&mut self.tv);
        t.copy_from_slice(&v[..self.nw]);
        self.reduce_value(&mut t);
        let out = match leading(&t) {
            None => false,
            Some(p) => {
                self.vals.extend_from_slice(&t);
                self.coords.resize(self.coords.len() + self.cw, 0);
                self.orig.extend_from_slice(&t);
                self.rows += 1;
                self.piv[p] = self.rows as u32;
                true
            }
        };
        self.tv = t;
        out
    }

    /// `planner_shared.decompose`: (residual on non-pivot bits, combination mask).
    pub fn decompose(&self, v: &[u64]) -> (Vec<u64>, Vec<u64>) {
        let nw = self.nw;
        let mut t = v.to_vec();
        let mut res = vec![0u64; nw];
        let mut c = vec![0u64; self.cw];
        while let Some(p) = leading(&t) {
            let r = self.piv[p];
            if r != 0 {
                let r = (r - 1) as usize;
                xor_into(&mut t, &self.vals[r * nw..(r + 1) * nw]);
                xor_into(&mut c, &self.coords[r * self.cw..(r + 1) * self.cw]);
            } else {
                set_bit(&mut res, p);
                flip_bit(&mut t, p);
            }
        }
        (res, c)
    }
}

/// Plain echelon (no coordinates): `reduce_row` / `append_row` over a pivot dict.
#[derive(Clone, Debug)]
pub struct Echelon {
    pub nw: usize,
    vals: Vec<u64>,
    piv: Vec<u32>,
    rows: usize,
}

impl Echelon {
    pub fn new(width: usize) -> Self {
        let nw = words_for(width);
        Echelon { nw, vals: Vec::new(), piv: vec![0; nw * 64], rows: 0 }
    }

    pub fn len(&self) -> usize {
        self.rows
    }

    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    pub fn reduce(&self, v: &mut [u64]) {
        while let Some(p) = leading(v) {
            let r = self.piv[p];
            if r == 0 {
                return;
            }
            let r = (r - 1) as usize;
            xor_into(v, &self.vals[r * self.nw..(r + 1) * self.nw]);
        }
    }

    pub fn insert(&mut self, v: &[u64]) -> bool {
        let mut t = v.to_vec();
        self.reduce(&mut t);
        match leading(&t) {
            None => false,
            Some(p) => {
                self.vals.extend_from_slice(&t);
                self.rows += 1;
                self.piv[p] = self.rows as u32;
                true
            }
        }
    }

    pub fn row(&self, i: usize) -> &[u64] {
        &self.vals[i * self.nw..(i + 1) * self.nw]
    }
}

/// `recover.rank`.
pub fn rank(rows: &[Vec<u64>], width: usize) -> usize {
    let mut e = Echelon::new(width);
    for r in rows {
        e.insert(r);
    }
    e.len()
}

/// `recover.nullspace(rows, width)`: basis of {x : parity(x & r) = 0 for all r}, in the Python order (free bits
/// ascending; x = e_f + sum of pivots p whose reduced row has bit f). The reduced row echelon form is unique, so
/// the output equals the Python's whatever the elimination order.
pub fn nullspace(rows: &[Vec<u64>], width: usize) -> Vec<Vec<u64>> {
    let nw = words_for(width.max(rows.iter().map(|r| leading(r).map_or(0, |p| p + 1)).max().unwrap_or(0)));
    let mut e = Echelon::new(nw * 64);
    for r in rows {
        let mut t = r.clone();
        t.resize(nw, 0);
        e.insert(&t);
    }
    // full reduction, sorted by pivot ascending
    let mut rr: Vec<(usize, Vec<u64>)> = (0..e.len()).map(|i| {
        let v = e.row(i).to_vec();
        (leading(&v).unwrap(), v)
    }).collect();
    rr.sort_by_key(|(p, _)| *p);
    for i in 0..rr.len() {
        let (p, r) = (rr[i].0, rr[i].1.clone());
        for j in 0..rr.len() {
            if j != i && get_bit(&rr[j].1, p) {
                xor_into(&mut rr[j].1, &r);
            }
        }
    }
    let mut is_piv = vec![false; nw * 64];
    for (p, _) in &rr {
        is_piv[*p] = true;
    }
    let ow = words_for(width);
    let mut out = Vec::new();
    for f in 0..width {
        if is_piv[f] {
            continue;
        }
        let mut x = vec![0u64; ow.max(nw)];
        set_bit(&mut x, f);
        for (p, r) in &rr {
            if get_bit(r, f) {
                set_bit(&mut x, *p);
            }
        }
        out.push(x);
    }
    out
}

/// Inner-product parity of two packed vectors (lengths may differ).
#[inline]
pub fn dot(a: &[u64], b: &[u64]) -> bool {
    let mut s = 0u32;
    for (x, y) in a.iter().zip(b) {
        s ^= (x & y).count_ones();
    }
    s & 1 == 1
}

/// `planner_shared.intersect(V1, V2, m)`: basis of span(V1) cap span(V2), in the Python order.
pub fn intersect(v1: &[Vec<u64>], v2: &[Vec<u64>], m: usize) -> Vec<Vec<u64>> {
    let cons = if v2.len() < m { nullspace(v2, m) } else { Vec::new() };
    if cons.is_empty() {
        return v1.to_vec();
    }
    let n1 = v1.len();
    let ew = words_for(n1);
    let eqs: Vec<Vec<u64>> = cons
        .iter()
        .map(|e| {
            let mut q = vec![0u64; ew];
            for (i, v) in v1.iter().enumerate() {
                if dot(v, e) {
                    set_bit(&mut q, i);
                }
            }
            q
        })
        .collect();
    let vw = v1.iter().map(|v| v.len()).max().unwrap_or(1);
    let mut out = Vec::new();
    for c in nullspace(&eqs, n1) {
        let mut v = vec![0u64; vw];
        for i in 0..n1 {
            if get_bit(&c, i) {
                xor_into(&mut v, &v1[i]);
            }
        }
        out.push(v);
    }
    out
}

/// A set of `nb` coordinate bases plus a table of groups: group g is a short list of (basis, vector) items (a pair
/// line has two ends). Object-level operations on whole groups, so a planner pass costs a handful of Python calls
/// instead of millions (review §6). Semantics are those of the Python loops in `planner_shared.pair_lines` and
/// `augment_quotient`, item by item, in the same order.
pub struct GroupSet {
    pub nb: usize,
    pub width: usize,
    nw: usize,
    pub bases: Vec<CoordBasis>,
    gstart: Vec<usize>,     // group g items: gstart[g]..gstart[g + 1]
    ibasis: Vec<usize>,     // item -> basis
    ivals: Vec<u64>,        // item * nw
    ipos: Vec<i64>,         // item -> index in its basis when its group was last added (-1 = none)
    // per-item cached solve (status) or decomposition (quotient images)
    status: Vec<Option<Vec<u64>>>,
    status_ids: Vec<usize>,
    dec_res: Vec<Vec<u64>>,
    dec_c: Vec<Vec<u64>>,
    dec_ids: Vec<usize>,
    img: Vec<Vec<u64>>, // per item of the current quotient candidates: residual << 2 | bits (width + 2 bits)
}

impl GroupSet {
    pub fn new(nb: usize, width: usize, groups: Vec<Vec<(usize, Vec<u64>)>>) -> Self {
        let nw = words_for(width);
        let mut gstart = vec![0];
        let mut ibasis = Vec::new();
        let mut ivals = Vec::new();
        for g in groups {
            for (b, mut v) in g {
                assert!(b < nb, "basis index out of range");
                v.resize(nw, 0);
                ibasis.push(b);
                ivals.extend_from_slice(&v);
            }
            gstart.push(ibasis.len());
        }
        let ni = ibasis.len();
        GroupSet {
            nb,
            width,
            nw,
            bases: (0..nb).map(|_| CoordBasis::new(width)).collect(),
            gstart,
            ibasis,
            ivals,
            ipos: vec![-1; ni],
            status: vec![None; ni],
            status_ids: Vec::new(),
            dec_res: vec![Vec::new(); ni],
            dec_c: vec![Vec::new(); ni],
            dec_ids: Vec::new(),
            img: vec![Vec::new(); ni],
        }
    }

    pub fn ngroups(&self) -> usize {
        self.gstart.len() - 1
    }

    #[inline]
    fn item(&self, i: usize) -> &[u64] {
        &self.ivals[i * self.nw..(i + 1) * self.nw]
    }

    #[inline]
    fn items(&self, g: usize) -> std::ops::Range<usize> {
        self.gstart[g]..self.gstart[g + 1]
    }

    pub fn clear(&mut self) {
        for b in &mut self.bases {
            b.clear();
        }
    }

    pub fn marks(&self) -> Vec<usize> {
        self.bases.iter().map(|b| b.len()).collect()
    }

    pub fn restore(&mut self, marks: &[usize]) {
        for (b, &n) in self.bases.iter_mut().zip(marks) {
            b.truncate(n);
        }
    }

    /// Add the group's items in order (each `Basis.add`); returns how many were independent.
    pub fn add_group(&mut self, g: usize) -> usize {
        let mut k = 0;
        let nw = self.nw;
        for i in self.gstart[g]..self.gstart[g + 1] {
            let b = self.ibasis[i];
            let n = self.bases[b].len();
            if self.bases[b].add(&self.ivals[i * nw..(i + 1) * nw]) {
                self.ipos[i] = n as i64;
                k += 1;
            } else {
                self.ipos[i] = -1;
            }
        }
        k
    }

    /// The group's items are jointly independent of the bases (adding them all raises the rank by their number).
    /// Equals the Python `addable` (two ends in different bases: each outside its span; in one basis: a, c and a ^ c
    /// outside it).
    pub fn independent(&mut self, g: usize) -> bool {
        let pushed = self.push_temp_group(g);
        let ok = pushed.is_some();
        if let Some(bs) = pushed {
            self.pop_temp(&bs);
        }
        ok
    }

    /// Push the group's items as temporary rows; Some(bases pushed, in order) iff all were independent (on failure
    /// the partial pushes are already undone).
    fn push_temp_group(&mut self, g: usize) -> Option<Vec<usize>> {
        let mut bs = Vec::with_capacity(4);
        let (s, e) = (self.gstart[g], self.gstart[g + 1]);
        for i in s..e {
            let b = self.ibasis[i];
            let nw = self.nw;
            if !self.bases[b].push_temp(&self.ivals[i * nw..(i + 1) * nw]) {
                self.pop_temp(&bs);
                return None;
            }
            bs.push(b);
        }
        Some(bs)
    }

    fn pop_temp(&mut self, bs: &[usize]) {
        for &b in bs.iter().rev() {
            let n = self.bases[b].len();
            self.bases[b].truncate(n - 1);
        }
    }

    /// `build(selected)`: clear, then add every group; Err(group) if some item is dependent (the Python asserts).
    pub fn build(&mut self, selected: &[usize]) -> Result<(), usize> {
        self.clear();
        for &g in selected {
            let need = self.items(g).len();
            if self.add_group(g) != need {
                return Err(g);
            }
        }
        Ok(())
    }

    /// As `build`, but the rows carry no combination masks (zero): enough for `independent` / `first_pair`, which
    /// look at values only, and cheaper. Do not call `solve`, `positions`, `compute_status` or `compute_decomp`
    /// after it without a full `build`.
    pub fn build_values(&mut self, selected: &[usize]) -> Result<(), usize> {
        self.clear();
        for &g in selected {
            if self.push_temp_group(g).is_none() {
                return Err(g);
            }
        }
        Ok(())
    }

    /// Greedy pass: for each group in `order`, add it if independent; returns the accepted groups.
    pub fn greedy(&mut self, order: &[usize]) -> Vec<usize> {
        let mut out = Vec::new();
        for &g in order {
            if self.independent(g) {
                let n = self.gstart[g + 1] - self.gstart[g];
                let got = self.add_group(g);
                debug_assert_eq!(got, n);
                out.push(g);
            }
        }
        out
    }

    /// First (i, j), i < j, in lexicographic order with `ids[i]` independent of the bases and `ids[j]` independent
    /// after adding `ids[i]`. Bases are left unchanged.
    pub fn first_pair(&mut self, ids: &[usize]) -> Option<(usize, usize)> {
        for i in 0..ids.len() {
            let Some(bs) = self.push_temp_group(ids[i]) else { continue };
            for j in i + 1..ids.len() {
                if self.independent(ids[j]) {
                    self.pop_temp(&bs);
                    return Some((i, j));
                }
            }
            self.pop_temp(&bs);
        }
        None
    }

    /// Position of each item of group g in its basis when the group was last added (-1 if none).
    pub fn positions(&self, g: usize) -> Vec<(usize, i64)> {
        self.items(g).map(|i| (self.ibasis[i], self.ipos[i])).collect()
    }

    /// Cache `solve` of every item of the given groups (the rebuild pass's `status`).
    pub fn compute_status(&mut self, ids: &[usize]) {
        self.status_ids = ids.to_vec();
        for &g in ids {
            for i in self.items(g) {
                let b = self.ibasis[i];
                self.status[i] = self.bases[b].solve(self.item(i));
            }
        }
    }

    /// Groups (status order) whose every item is outside the span or has coordinate px in basis bx or py in by.
    pub fn enabled(&self, bx: usize, px: usize, by: usize, py: usize) -> Vec<usize> {
        let mut out = Vec::new();
        for &g in &self.status_ids {
            let mut ok = true;
            for i in self.items(g) {
                if let Some(c) = &self.status[i] {
                    let b = self.ibasis[i];
                    if (b == bx && get_bit(c, px)) || (b == by && get_bit(c, py)) {
                        continue;
                    }
                    ok = false;
                    break;
                }
            }
            if ok {
                out.push(g);
            }
        }
        out
    }

    /// Cache `decompose` of every item of the given groups (`augment_quotient`'s `dec`).
    pub fn compute_decomp(&mut self, ids: &[usize]) {
        self.dec_ids = ids.to_vec();
        for &g in ids {
            for i in self.items(g) {
                let b = self.ibasis[i];
                let (r, c) = self.bases[b].decompose(self.item(i));
                self.dec_res[i] = r;
                self.dec_c[i] = c;
            }
        }
    }

    /// Quotient candidates for removed vectors (bx, px), (by, py): groups (dec order) of two items whose images
    /// `residual << 2 | [b == bx and c_px] | [b == by and c_py] << 1` are both nonzero and differ when in one basis.
    pub fn quotient_cands(&mut self, bx: usize, px: usize, by: usize, py: usize) -> Vec<usize> {
        let iw = words_for(self.width + 2);
        let mut out = Vec::new();
        for k in 0..self.dec_ids.len() {
            let g = self.dec_ids[k];
            let r = self.items(g);
            assert_eq!(r.len(), 2, "quotient candidates need two-item groups");
            for i in r.clone() {
                let b = self.ibasis[i];
                let mut x = vec![0u64; iw];
                // residual << 2
                let res = &self.dec_res[i];
                for (w, &val) in res.iter().enumerate() {
                    x[w] |= val << 2;
                    if w + 1 < iw {
                        x[w + 1] |= val >> 62;
                    }
                }
                let c = &self.dec_c[i];
                if b == bx && get_bit(c, px) {
                    x[0] |= 1;
                }
                if b == by && get_bit(c, py) {
                    x[0] |= 2;
                }
                self.img[i] = x;
            }
            let (i1, i2) = (r.start, r.start + 1);
            let nz1 = !is_zero(&self.img[i1]);
            let nz2 = !is_zero(&self.img[i2]);
            if nz1 && nz2 && (self.ibasis[i1] != self.ibasis[i2] || self.img[i1] != self.img[i2]) {
                out.push(g);
            }
        }
        out
    }

    /// `_independent` over the images of two candidate groups: per basis, the images are linearly independent.
    fn images_independent(&self, g1: usize, g2: usize) -> bool {
        let iw = words_for(self.width + 2);
        let mut per: Vec<(usize, Echelon)> = Vec::new();
        for g in [g1, g2] {
            for i in self.items(g) {
                let b = self.ibasis[i];
                let pos = match per.iter().position(|(bb, _)| *bb == b) {
                    Some(p) => p,
                    None => {
                        per.push((b, Echelon::new(iw * 64)));
                        per.len() - 1
                    }
                };
                if !per[pos].1.insert(&self.img[i]) {
                    return false;
                }
            }
        }
        true
    }

    /// `augment_quotient` inner search over shuffled candidates: first (l1, l2) with l1 among the first `l1_cap`.
    pub fn quotient_pair(&self, cands: &[usize], l1_cap: usize) -> Option<(usize, usize)> {
        let n1 = cands.len().min(l1_cap);
        for i in 0..n1 {
            for j in i + 1..cands.len() {
                if self.images_independent(cands[i], cands[j]) {
                    return Some((cands[i], cands[j]));
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: u128, nw: usize) -> Vec<u64> {
        let mut out = vec![0u64; nw];
        out[0] = x as u64;
        if nw > 1 {
            out[1] = (x >> 64) as u64;
        }
        out
    }

    #[test]
    fn basis_small() {
        let mut b = CoordBasis::new(70);
        assert!(b.add(&v(0b110, 2)));
        assert!(b.add(&v(0b011, 2)));
        assert!(!b.add(&v(0b101, 2)));
        assert_eq!(b.solve(&v(0b101, 2)).unwrap()[0], 0b11);
        assert!(b.solve(&v(0b1000, 2)).is_none());
        b.truncate(1);
        assert!(b.solve(&v(0b011, 2)).is_none());
    }

    #[test]
    fn nullspace_small() {
        let ns = nullspace(&[v(0b11, 1)], 3);
        assert_eq!(ns, vec![v(0b011, 1), v(0b100, 1)]);
    }
}
