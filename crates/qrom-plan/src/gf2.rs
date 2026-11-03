//! Incremental GF(2) echelon bases (highest-bit pivots, as in the Python planner's `reduce_row`/`append_row`),
//! echelon bases with coordinates, and nullspaces.

use crate::bits::Bits;

/// Rows keyed by their highest set bit. Rows are not mutually reduced; reduction always eliminates the highest
/// remaining bit, which terminates because every stored row's leading bit is its pivot.
#[derive(Clone, Debug)]
pub struct Echelon<const W: usize> {
    rows: Vec<Bits<W>>,
    piv: Vec<u32>, // pivot bit -> row index + 1 (0 = none)
}

impl<const W: usize> Echelon<W> {
    pub fn new() -> Self {
        Echelon { rows: Vec::new(), piv: vec![0; W * 64] }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn rows(&self) -> &[Bits<W>] {
        &self.rows
    }

    /// Reduce `v` against the basis; zero iff `v` is in the span.
    #[inline]
    pub fn reduce(&self, mut v: Bits<W>) -> Bits<W> {
        while let Some(p) = v.leading() {
            let r = self.piv[p];
            if r == 0 {
                return v;
            }
            v ^= self.rows[(r - 1) as usize];
        }
        v
    }

    #[inline]
    pub fn contains(&self, v: Bits<W>) -> bool {
        self.reduce(v).is_zero()
    }

    /// Insert an already reduced nonzero vector.
    #[inline]
    pub fn push_reduced(&mut self, v: Bits<W>) {
        let p = v.leading().expect("nonzero");
        debug_assert_eq!(self.piv[p], 0);
        self.rows.push(v);
        self.piv[p] = self.rows.len() as u32;
    }

    /// Reduce and insert; true iff the vector was independent.
    #[inline]
    pub fn insert(&mut self, v: Bits<W>) -> bool {
        let r = self.reduce(v);
        if r.is_zero() {
            false
        } else {
            self.push_reduced(r);
            true
        }
    }

    /// Remove the most recently inserted row.
    #[inline]
    pub fn pop(&mut self) {
        if let Some(v) = self.rows.pop() {
            self.piv[v.leading().unwrap()] = 0;
        }
    }

    /// Reduced row echelon form of the stored rows (fully reduced, sorted by pivot ascending), as (pivot, row).
    fn rref(&self) -> Vec<(usize, Bits<W>)> {
        let mut rows: Vec<(usize, Bits<W>)> = self.rows.iter().map(|r| (r.leading().unwrap(), *r)).collect();
        rows.sort_by_key(|(p, _)| *p);
        for i in 0..rows.len() {
            let (p, r) = rows[i];
            for j in 0..rows.len() {
                if j != i && rows[j].1.get(p) {
                    rows[j].1 ^= r;
                }
            }
        }
        rows
    }

    /// A basis of the orthogonal complement (the vectors x of `width` bits with r . x = 0 for every stored row).
    pub fn nullspace(&self, width: usize) -> Vec<Bits<W>> {
        let rows = self.rref();
        let mut is_piv = vec![false; width];
        for (p, _) in &rows {
            is_piv[*p] = true;
        }
        let mut out = Vec::with_capacity(width - rows.len());
        for f in 0..width {
            if is_piv[f] {
                continue;
            }
            let mut x = Bits::<W>::bit(f);
            for (p, r) in &rows {
                if r.get(f) {
                    x.set(*p);
                }
            }
            out.push(x);
        }
        out
    }
}

impl<const W: usize> Default for Echelon<W> {
    fn default() -> Self {
        Self::new()
    }
}

/// Nullspace of a list of row vectors of `width` bits (the Python planner's `nullspace`).
pub fn nullspace<const W: usize>(rows: &[Bits<W>], width: usize) -> Vec<Bits<W>> {
    let mut e = Echelon::<W>::new();
    for &r in rows {
        e.insert(r);
    }
    if e.len() >= width {
        return Vec::new();
    }
    e.nullspace(width)
}

/// Rank of a list of vectors.
pub fn rank<const W: usize>(rows: &[Bits<W>]) -> usize {
    let mut e = Echelon::<W>::new();
    for &r in rows {
        e.insert(r);
    }
    e.len()
}

/// Echelon basis that also tracks, for every stored row, which inserted vectors sum to it (coordinates as a
/// bitmask over insertion indices; the Python planner's `Basis`).
#[derive(Clone, Debug)]
pub struct Coords<const W: usize, const C: usize> {
    rows: Vec<(Bits<W>, Bits<C>)>,
    piv: Vec<u32>,
    inserted: usize,
}

impl<const W: usize, const C: usize> Coords<W, C> {
    pub fn new() -> Self {
        Coords { rows: Vec::new(), piv: vec![0; W * 64], inserted: 0 }
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of vectors offered so far (independent or not); the next vector gets this index.
    pub fn inserted(&self) -> usize {
        self.inserted
    }

    /// Insert; true iff independent. Every offered vector consumes an index (dependent ones too), as in Python
    /// only when `count_dependent` is set; the planner's `Basis` counts only independent vectors.
    pub fn add(&mut self, v: Bits<W>) -> bool {
        assert!(self.inserted < C * 64, "coordinate width exceeded");
        let mut value = v;
        let mut coords = Bits::<C>::bit(self.inserted);
        while let Some(p) = value.leading() {
            let r = self.piv[p];
            if r == 0 {
                self.rows.push((value, coords));
                self.piv[p] = self.rows.len() as u32;
                self.inserted += 1;
                return true;
            }
            let (rv, rc) = self.rows[(r - 1) as usize];
            value ^= rv;
            coords ^= rc;
        }
        false
    }

    /// Insert with an explicit coordinate index (every offered vector owns an index, dependent ones included);
    /// true iff independent.
    pub fn add_at(&mut self, v: Bits<W>, index: usize) -> bool {
        assert!(index < C * 64, "coordinate width exceeded");
        let mut value = v;
        let mut coords = Bits::<C>::bit(index);
        while let Some(p) = value.leading() {
            let r = self.piv[p];
            if r == 0 {
                self.rows.push((value, coords));
                self.piv[p] = self.rows.len() as u32;
                self.inserted = self.inserted.max(index + 1);
                return true;
            }
            let (rv, rc) = self.rows[(r - 1) as usize];
            value ^= rv;
            coords ^= rc;
        }
        false
    }

    /// Coordinates of `v` over the inserted independent vectors, or None if outside the span.
    #[inline]
    pub fn solve(&self, v: Bits<W>) -> Option<Bits<C>> {
        let mut value = v;
        let mut coords = Bits::<C>::ZERO;
        while let Some(p) = value.leading() {
            let r = self.piv[p];
            if r == 0 {
                return None;
            }
            let (rv, rc) = self.rows[(r - 1) as usize];
            value ^= rv;
            coords ^= rc;
        }
        Some(coords)
    }
}

impl<const W: usize, const C: usize> Default for Coords<W, C> {
    fn default() -> Self {
        Self::new()
    }
}

/// Basis of the intersection of two subspaces given by spanning lists (vectors of `width` bits).
pub fn intersect<const W: usize>(v1: &[Bits<W>], v2: &[Bits<W>], width: usize) -> Vec<Bits<W>> {
    // Python: cons2 = annihilator of V2; eqs_e = (V1[i] . e)_i; combos = nullspace(eqs); common = sum V1[i].
    let cons2 = if v2.len() < width { nullspace(v2, width) } else { Vec::new() };
    if cons2.is_empty() {
        return v1.to_vec();
    }
    if v1.is_empty() {
        return Vec::new();
    }
    assert!(v1.len() <= 64, "intersect: V1 larger than 64 vectors");
    let eqs: Vec<Bits<1>> = cons2
        .iter()
        .map(|e| {
            let mut m = 0u64;
            for (i, v) in v1.iter().enumerate() {
                if v.dot(e) {
                    m |= 1 << i;
                }
            }
            Bits([m])
        })
        .collect();
    let combos = nullspace(&eqs, v1.len());
    combos
        .iter()
        .map(|c| {
            let mut v = Bits::<W>::ZERO;
            for i in c.ones() {
                v ^= v1[i];
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullspace_of_small_matrix() {
        // rows 1100, 0110 over 4 bits -> nullspace dim 2
        let rows = [Bits::<1>([0b0011]), Bits::<1>([0b0110])];
        let ns = nullspace(&rows, 4);
        assert_eq!(ns.len(), 2);
        for x in &ns {
            for r in &rows {
                assert!(!r.dot(x));
            }
        }
        assert_eq!(rank(&ns), 2);
    }

    #[test]
    fn coords_solve() {
        let mut c = Coords::<1, 1>::new();
        assert!(c.add(Bits([0b101])));
        assert!(c.add(Bits([0b011])));
        assert!(!c.add(Bits([0b110])));
        assert_eq!(c.solve(Bits([0b110])).unwrap().0[0], 0b11);
        assert!(c.solve(Bits([0b1000])).is_none());
    }

    #[test]
    fn intersect_planes() {
        let v1 = [Bits::<1>([0b001]), Bits::<1>([0b010])];
        let v2 = [Bits::<1>([0b010]), Bits::<1>([0b100])];
        let c = intersect(&v1, &v2, 3);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].0[0], 0b010);
    }

    #[test]
    fn echelon_pop() {
        let mut e = Echelon::<1>::new();
        assert!(e.insert(Bits([0b100])));
        assert!(e.insert(Bits([0b110])));
        assert!(!e.insert(Bits([0b010])));
        e.pop();
        assert!(e.insert(Bits([0b010])));
        assert_eq!(e.len(), 2);
    }
}
