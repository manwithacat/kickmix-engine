//! Python binding for the GF(2) objects (card E2). Vectors cross as Python ints (little-endian bytes); the
//! group-level calls keep the vectors in Rust so a planner pass needs few crossings.
use kmx_analysis::gf2 as G;
use pyo3::exceptions::{PyAssertionError, PyValueError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt, PyList, PyTuple};

pub fn int_to_words(ob: &Bound<'_, PyAny>, nw: usize) -> PyResult<Vec<u64>> {
    let py = ob.py();
    let b = ob.call_method1(intern!(py, "to_bytes"), (nw * 8, intern!(py, "little")))?;
    let bytes = b.downcast::<PyBytes>()?.as_bytes();
    let mut out = vec![0u64; nw];
    for (w, chunk) in out.iter_mut().zip(bytes.chunks_exact(8)) {
        *w = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    Ok(out)
}

pub fn words_to_int<'py>(py: Python<'py>, v: &[u64]) -> PyResult<Bound<'py, PyAny>> {
    let mut bytes = Vec::with_capacity(v.len() * 8);
    for w in v {
        bytes.extend_from_slice(&w.to_le_bytes());
    }
    let pb = PyBytes::new(py, &bytes);
    py.get_type::<PyInt>().call_method1(intern!(py, "from_bytes"), (pb, intern!(py, "little")))
}

fn ints_to_rows(rows: &Bound<'_, PyAny>, width: usize) -> PyResult<Vec<Vec<u64>>> {
    let nw = G::words_for(width);
    let mut out = Vec::new();
    for r in rows.try_iter()? {
        out.push(int_to_words(&r?, nw)?);
    }
    Ok(out)
}

fn rows_to_list<'py>(py: Python<'py>, rows: &[Vec<u64>]) -> PyResult<Bound<'py, PyList>> {
    let l = PyList::empty(py);
    for r in rows {
        l.append(words_to_int(py, r)?)?;
    }
    Ok(l)
}

/// Gf2Basis(width): recover.Basis with vectors below 2**width.
#[pyclass(module = "kickmix_analysis._kmx_analysis", name = "Gf2Basis")]
pub struct PyBasis {
    b: G::CoordBasis,
}

#[pymethods]
impl PyBasis {
    #[new]
    fn new(width: usize) -> Self {
        PyBasis { b: G::CoordBasis::new(width) }
    }

    #[getter]
    fn width(&self) -> usize {
        self.b.width
    }

    fn __len__(&self) -> usize {
        self.b.len()
    }

    fn add(&mut self, v: &Bound<'_, PyAny>) -> PyResult<bool> {
        let w = int_to_words(v, self.b.nw)?;
        Ok(self.b.add(&w))
    }

    fn add_many(&mut self, vs: &Bound<'_, PyAny>) -> PyResult<Vec<bool>> {
        let mut out = Vec::new();
        for v in vs.try_iter()? {
            let w = int_to_words(&v?, self.b.nw)?;
            out.push(self.b.add(&w));
        }
        Ok(out)
    }

    fn reduce<'py>(&self, py: Python<'py>, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
        let mut w = int_to_words(v, self.b.nw)?;
        let mut c = vec![0u64; self.b.cw];
        self.b.reduce_into(&mut w, &mut c);
        PyTuple::new(py, [words_to_int(py, &w)?, words_to_int(py, &c)?])
    }

    fn solve<'py>(&self, py: Python<'py>, v: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let w = int_to_words(v, self.b.nw)?;
        match self.b.solve(&w) {
            None => Ok(None),
            Some(c) => Ok(Some(words_to_int(py, &c)?)),
        }
    }

    fn contains(&self, v: &Bound<'_, PyAny>) -> PyResult<bool> {
        let w = int_to_words(v, self.b.nw)?;
        Ok(self.b.contains(&w))
    }

    fn solve_many<'py>(&self, py: Python<'py>, vs: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
        let l = PyList::empty(py);
        for v in vs.try_iter()? {
            let w = int_to_words(&v?, self.b.nw)?;
            match self.b.solve(&w) {
                None => l.append(py.None())?,
                Some(c) => l.append(words_to_int(py, &c)?)?,
            }
        }
        Ok(l)
    }

    fn decompose<'py>(&self, py: Python<'py>, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
        let w = int_to_words(v, self.b.nw)?;
        let (r, c) = self.b.decompose(&w);
        PyTuple::new(py, [words_to_int(py, &r)?, words_to_int(py, &c)?])
    }

    fn truncate(&mut self, n: usize) {
        self.b.truncate(n);
    }

    #[getter]
    fn vectors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        let l = PyList::empty(py);
        for i in 0..self.b.len() {
            l.append(words_to_int(py, self.b.vector(i))?)?;
        }
        Ok(l)
    }
}

/// GroupSet(nb, width, groups): nb coordinate bases and a table of groups [(basis, vector int), ...].
#[pyclass(module = "kickmix_analysis._kmx_analysis", name = "GroupSet", unsendable)]
pub struct PyGroupSet {
    g: G::GroupSet,
}

#[pymethods]
impl PyGroupSet {
    #[new]
    fn new(nb: usize, width: usize, groups: &Bound<'_, PyAny>) -> PyResult<Self> {
        let nw = G::words_for(width);
        let mut gs = Vec::new();
        for g in groups.try_iter()? {
            let mut items = Vec::new();
            for it in g?.try_iter()? {
                let it = it?;
                let b: usize = it.get_item(0)?.extract()?;
                if b >= nb {
                    return Err(PyValueError::new_err("basis index out of range"));
                }
                items.push((b, int_to_words(&it.get_item(1)?, nw)?));
            }
            gs.push(items);
        }
        Ok(PyGroupSet { g: G::GroupSet::new(nb, width, gs) })
    }

    fn __len__(&self) -> usize {
        self.g.ngroups()
    }

    fn size(&self, b: usize) -> usize {
        self.g.bases[b].len()
    }

    fn sizes(&self) -> Vec<usize> {
        self.g.marks()
    }

    fn restore(&mut self, marks: Vec<usize>) {
        self.g.restore(&marks);
    }

    fn clear(&mut self) {
        self.g.clear();
    }

    /// Clear and add the groups in order; AssertionError if an item is dependent (as the Python `assert`).
    fn build(&mut self, selected: Vec<usize>) -> PyResult<()> {
        self.g.build(&selected).map_err(|g| PyAssertionError::new_err(format!("group {g} dependent in build")))
    }

    /// build without combination masks (for independent / first_pair only).
    fn build_values(&mut self, selected: Vec<usize>) -> PyResult<()> {
        self.g
            .build_values(&selected)
            .map_err(|g| PyAssertionError::new_err(format!("group {g} dependent in build")))
    }

    fn independent(&mut self, g: usize) -> bool {
        self.g.independent(g)
    }

    fn add_group(&mut self, g: usize) -> usize {
        self.g.add_group(g)
    }

    fn greedy(&mut self, order: Vec<usize>) -> Vec<usize> {
        self.g.greedy(&order)
    }

    fn first_pair(&mut self, ids: Vec<usize>) -> Option<(usize, usize)> {
        self.g.first_pair(&ids)
    }

    fn positions(&self, g: usize) -> Vec<(usize, i64)> {
        self.g.positions(g)
    }

    fn compute_status(&mut self, ids: Vec<usize>) {
        self.g.compute_status(&ids);
    }

    fn enabled(&self, bx: usize, px: usize, by: usize, py: usize) -> Vec<usize> {
        self.g.enabled(bx, px, by, py)
    }

    fn compute_decomp(&mut self, ids: Vec<usize>) {
        self.g.compute_decomp(&ids);
    }

    fn quotient_cands(&mut self, bx: usize, px: usize, by: usize, py: usize) -> Vec<usize> {
        self.g.quotient_cands(bx, px, by, py)
    }

    fn quotient_pair(&self, cands: Vec<usize>, l1_cap: usize) -> Option<(usize, usize)> {
        self.g.quotient_pair(&cands, l1_cap)
    }

    /// Basis.add of a Python-int vector to basis b.
    fn add_vec(&mut self, b: usize, v: &Bound<'_, PyAny>) -> PyResult<bool> {
        let w = int_to_words(v, G::words_for(self.g.width))?;
        Ok(self.g.bases[b].add(&w))
    }
}

#[pyfunction]
pub fn gf2_rank(rows: &Bound<'_, PyAny>, width: usize) -> PyResult<usize> {
    let r = ints_to_rows(rows, width)?;
    Ok(G::rank(&r, width))
}

#[pyfunction]
pub fn gf2_nullspace<'py>(py: Python<'py>, rows: &Bound<'py, PyAny>, width: usize, row_width: usize) -> PyResult<Bound<'py, PyList>> {
    let r = ints_to_rows(rows, row_width.max(width))?;
    rows_to_list(py, &G::nullspace(&r, width))
}

#[pyfunction]
pub fn gf2_intersect<'py>(py: Python<'py>, v1: &Bound<'py, PyAny>, v2: &Bound<'py, PyAny>, m: usize, width: usize) -> PyResult<Bound<'py, PyList>> {
    let a = ints_to_rows(v1, width.max(m))?;
    let b = ints_to_rows(v2, width.max(m))?;
    rows_to_list(py, &G::intersect(&a, &b, m))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyBasis>()?;
    m.add_class::<PyGroupSet>()?;
    m.add_function(wrap_pyfunction!(gf2_rank, m)?)?;
    m.add_function(wrap_pyfunction!(gf2_nullspace, m)?)?;
    m.add_function(wrap_pyfunction!(gf2_intersect, m)?)?;
    Ok(())
}
