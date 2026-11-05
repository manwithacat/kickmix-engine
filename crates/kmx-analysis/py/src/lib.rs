//! Python binding for kmx-analysis: extension `kickmix_analysis._kmx_analysis`.
mod gf2_py;
use kmx_analysis::liveness as L;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// liveness(path, cond_rule=False, register_gaps=False, profile=True) -> dict
#[pyfunction]
#[pyo3(signature = (path, cond_rule=false, register_gaps=false, profile=true))]
fn liveness<'py>(
    py: Python<'py>,
    path: String,
    cond_rule: bool,
    register_gaps: bool,
    profile: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let (lv, prof) = py
        .detach(|| -> Result<_, String> {
            let lv = L::liveness(&path, cond_rule)?;
            let prof = if profile {
                let regs: Vec<i64> = lv.registers.iter().map(|&x| x as i64).collect();
                Some(L::live_profile(
                    &lv.iv.logical,
                    &lv.iv.s,
                    &lv.iv.e,
                    &lv.iv.is_first,
                    &lv.iv.is_last,
                    lv.header.num_ops as usize,
                    &regs,
                    register_gaps,
                ))
            } else {
                None
            };
            Ok((lv, prof))
        })
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    let iv = PyDict::new(py);
    iv.set_item("logical", PyArray1::from_vec(py, lv.iv.logical))?;
    iv.set_item("s", PyArray1::from_vec(py, lv.iv.s))?;
    iv.set_item("e", PyArray1::from_vec(py, lv.iv.e))?;
    iv.set_item("is_first", PyArray1::from_vec(py, lv.iv.is_first))?;
    iv.set_item("is_last", PyArray1::from_vec(py, lv.iv.is_last))?;
    d.set_item("intervals", iv)?;
    if let Some(p) = prof {
        let peak = p.iter().copied().max().unwrap_or(0);
        d.set_item("peak", peak)?;
        d.set_item("profile", PyArray1::from_vec(py, p))?;
    }
    d.set_item("num_ops", lv.header.num_ops)?;
    d.set_item("num_qubits", lv.header.num_qubits)?;
    d.set_item("num_refs", lv.counts.num_refs)?;
    d.set_item("unconditional_resets", lv.counts.unconditional_resets)?;
    d.set_item("conditional_resets", lv.counts.conditional_resets)?;
    d.set_item("reset_refs", lv.reset_refs)?;
    d.set_item("registers", PyArray1::from_vec(py, lv.registers.iter().map(|&x| x as i64).collect::<Vec<_>>()))?;
    d.set_item("wire_first", PyArray1::from_vec(py, lv.wire_first))?;
    d.set_item("wire_last", PyArray1::from_vec(py, lv.wire_last))?;
    d.set_item("cond_rule", cond_rule)?;
    d.set_item("register_gaps", register_gaps)?;
    Ok(d)
}

/// live_profile(logical, s, e, is_first, is_last, num_ops, registers, register_gaps=False) -> int32[num_ops]
#[pyfunction]
#[pyo3(signature = (logical, s, e, is_first, is_last, num_ops, registers, register_gaps=false))]
#[allow(clippy::too_many_arguments)]
fn live_profile<'py>(
    py: Python<'py>,
    logical: PyReadonlyArray1<'py, i64>,
    s: PyReadonlyArray1<'py, i64>,
    e: PyReadonlyArray1<'py, i64>,
    is_first: PyReadonlyArray1<'py, bool>,
    is_last: PyReadonlyArray1<'py, bool>,
    num_ops: usize,
    registers: PyReadonlyArray1<'py, i64>,
    register_gaps: bool,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let (l, s, e, f, la, r) =
        (logical.as_slice()?, s.as_slice()?, e.as_slice()?, is_first.as_slice()?, is_last.as_slice()?, registers.as_slice()?);
    if !(l.len() == s.len() && s.len() == e.len() && e.len() == f.len() && f.len() == la.len()) {
        return Err(PyValueError::new_err("interval arrays differ in length"));
    }
    if r.windows(2).any(|w| w[0] >= w[1]) {
        return Err(PyValueError::new_err("registers must be sorted and unique"));
    }
    if s.iter().chain(e.iter()).any(|&x| x < 0 || x as usize >= num_ops.max(1)) {
        return Err(PyValueError::new_err("interval endpoint out of range"));
    }
    let p = L::live_profile(l, s, e, f, la, num_ops, r, register_gaps);
    Ok(PyArray1::from_vec(py, p))
}

/// idle_gaps(path, run_s, run_e, register_gaps=False, cond_rule=False) -> (ga, gb, gw) int64, sorted by (wire, a)
#[pyfunction]
#[pyo3(signature = (path, run_s, run_e, register_gaps=false, cond_rule=false))]
fn idle_gaps<'py>(
    py: Python<'py>,
    path: String,
    run_s: PyReadonlyArray1<'py, i64>,
    run_e: PyReadonlyArray1<'py, i64>,
    register_gaps: bool,
    cond_rule: bool,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let rs = run_s.as_slice()?.to_vec();
    let re = run_e.as_slice()?.to_vec();
    let (a, b, w) = py
        .detach(|| L::idle_gaps(&path, &rs, &re, register_gaps, cond_rule))
        .map_err(PyValueError::new_err)?;
    Ok((PyArray1::from_vec(py, a), PyArray1::from_vec(py, b), PyArray1::from_vec(py, w)))
}

/// recolour(path, logical, col, cond_rule=False) -> dict(qqq0, qqq1, qqq2, qq0, qq1, q0) uint32 streams
#[pyfunction]
#[pyo3(signature = (path, logical, col, cond_rule=false))]
fn recolour<'py>(
    py: Python<'py>,
    path: String,
    logical: PyReadonlyArray1<'py, i64>,
    col: PyReadonlyArray1<'py, i64>,
    cond_rule: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let l = logical.as_slice()?.to_vec();
    let c = col.as_slice()?.to_vec();
    let streams = py.detach(|| L::recolour(&path, &l, &c, cond_rule)).map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    for (name, v) in ["qqq0", "qqq1", "qqq2", "qq0", "qq1", "q0"].iter().zip(streams) {
        d.set_item(*name, PyArray1::from_vec(py, v))?;
    }
    Ok(d)
}

#[pymodule]
fn _kmx_analysis(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(liveness, m)?)?;
    m.add_function(wrap_pyfunction!(live_profile, m)?)?;
    m.add_function(wrap_pyfunction!(idle_gaps, m)?)?;
    m.add_function(wrap_pyfunction!(recolour, m)?)?;
    gf2_py::register(m)?;
    Ok(())
}
