//! Python binding for the tapped value simulator (card E3). Arrays cross as flat NumPy arrays; the Python package
//! (`kickmix_analysis.value_sim`) reshapes them.
use kmx_analysis::mt::Mt19937;
use kmx_analysis::valuesim as V;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn v64(a: Option<PyReadonlyArray1<'_, i64>>) -> PyResult<Vec<i64>> {
    Ok(match a {
        Some(x) => x.as_slice()?.to_vec(),
        None => Vec::new(),
    })
}

/// _value_sim(path, lanes, wires, cbits, outcomes, seed_key, explicit, point_ops, point_ptr, point_wires,
///            win_s, win_e, win_ptr, win_wires, dep_ops, prints, rare_cap, tflag, stop, threads) -> dict
#[pyfunction]
#[pyo3(signature = (path, lanes, wires, cbits, outcomes, seed_key, explicit=None, point_ops=None, point_ptr=None,
    point_wires=None, win_s=None, win_e=None, win_ptr=None, win_wires=None, dep_ops=None, prints=false,
    rare_cap=None, tflag=false, stop=None, threads=1))]
#[allow(clippy::too_many_arguments)]
fn _value_sim<'py>(
    py: Python<'py>,
    path: String,
    lanes: usize,
    wires: PyReadonlyArray1<'py, u64>,
    cbits: PyReadonlyArray1<'py, u64>,
    outcomes: &str,
    seed_key: Vec<u32>,
    explicit: Option<PyReadonlyArray1<'py, u64>>,
    point_ops: Option<PyReadonlyArray1<'py, i64>>,
    point_ptr: Option<PyReadonlyArray1<'py, i64>>,
    point_wires: Option<PyReadonlyArray1<'py, i64>>,
    win_s: Option<PyReadonlyArray1<'py, i64>>,
    win_e: Option<PyReadonlyArray1<'py, i64>>,
    win_ptr: Option<PyReadonlyArray1<'py, i64>>,
    win_wires: Option<PyReadonlyArray1<'py, i64>>,
    dep_ops: Option<PyReadonlyArray1<'py, i64>>,
    prints: bool,
    rare_cap: Option<usize>,
    tflag: bool,
    stop: Option<u64>,
    threads: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let oc = match outcomes {
        "mt" => V::Outcomes::Mt(Mt19937::from_key(&seed_key)),
        "zero" => V::Outcomes::Zero,
        "explicit" => match explicit {
            Some(e) => V::Outcomes::Explicit(e.as_slice()?.to_vec()),
            None => return Err(PyValueError::new_err("outcomes='explicit' needs the explicit array")),
        },
        _ => return Err(PyValueError::new_err("outcomes must be 'mt', 'zero' or 'explicit'")),
    };
    let mut point_ptr = v64(point_ptr)?;
    if point_ptr.is_empty() {
        point_ptr.push(0);
    }
    let mut win_ptr = v64(win_ptr)?;
    if win_ptr.is_empty() {
        win_ptr.push(0);
    }
    let taps = V::Taps {
        point_ops: v64(point_ops)?,
        point_ptr,
        point_wires: v64(point_wires)?,
        win_s: v64(win_s)?,
        win_e: v64(win_e)?,
        win_ptr,
        win_wires: v64(win_wires)?,
        dep_ops: v64(dep_ops)?,
        prints,
        rare_cap,
        tflag,
    };
    let w = wires.as_slice()?.to_vec();
    let c = cbits.as_slice()?.to_vec();
    let r = py
        .detach(|| V::simulate(&path, lanes, &w, &c, oc, &taps, stop, threads))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("lanes", r.lanes)?;
    d.set_item("words", r.words)?;
    d.set_item("ops_run", r.ops_run)?;
    d.set_item("n_hmr", r.n_hmr)?;
    d.set_item("points", PyArray1::from_vec(py, r.points))?;
    d.set_item("d_win", PyArray1::from_vec(py, r.d_win))?;
    d.set_item("d_wire", PyArray1::from_vec(py, r.d_wire))?;
    d.set_item("d_vec", PyArray1::from_vec(py, r.d_vec))?;
    d.set_item("d_before0", PyArray1::from_vec(py, r.d_before0))?;
    d.set_item("d_after0", PyArray1::from_vec(py, r.d_after0))?;
    d.set_item("dep", PyArray1::from_vec(py, r.dep))?;
    d.set_item("dep_rank", PyArray1::from_vec(py, r.dep_rank))?;
    d.set_item("and_op", PyArray1::from_vec(py, r.and_op))?;
    d.set_item("and_kind", PyArray1::from_vec(py, r.and_kind))?;
    d.set_item("and_pw", PyArray1::from_vec(py, r.and_pw))?;
    d.set_item("and_flag", PyArray1::from_vec(py, r.and_flag))?;
    d.set_item("and_p", PyArray1::from_vec(py, r.and_p))?;
    d.set_item("and_pc", PyArray1::from_vec(py, r.and_pc))?;
    d.set_item("and_o1", PyArray1::from_vec(py, r.and_o1))?;
    d.set_item("and_o2", PyArray1::from_vec(py, r.and_o2))?;
    d.set_item("rare_lanes", PyArray1::from_vec(py, r.rare_lanes))?;
    d.set_item("rare_ptr", PyArray1::from_vec(py, r.rare_ptr))?;
    d.set_item("tflag", PyArray1::from_vec(py, r.tflag))?;
    d.set_item("end_wires", PyArray1::from_vec(py, r.end_wires))?;
    d.set_item("end_cbits", PyArray1::from_vec(py, r.end_cbits))?;
    Ok(d)
}

/// _mt_getrandbits(seed_key, k, n) -> uint64[n * ceil(k/64)]: n successive getrandbits(k) draws (parity tests)
#[pyfunction]
fn _mt_getrandbits<'py>(py: Python<'py>, seed_key: Vec<u32>, k: usize, n: usize) -> Bound<'py, PyArray1<u64>> {
    let mut g = Mt19937::from_key(&seed_key);
    let w = (k + 63) / 64;
    let mut out = vec![0u64; n * w.max(1)];
    for i in 0..n {
        g.getrandbits_into(k, &mut out[i * w.max(1)..(i + 1) * w.max(1)]);
    }
    PyArray1::from_vec(py, out)
}

/// _gf2_dependent(rows, n, w) -> (bool[n], rank): rows in the span of the others
#[pyfunction]
fn _gf2_dependent<'py>(
    py: Python<'py>,
    rows: PyReadonlyArray1<'py, u64>,
    n: usize,
    w: usize,
) -> PyResult<(Bound<'py, PyArray1<bool>>, usize)> {
    let r = rows.as_slice()?;
    if r.len() != n * w {
        return Err(PyValueError::new_err("rows: expected n * w words"));
    }
    let (dep, rank) = V::dependent(r, n, w);
    Ok((PyArray1::from_vec(py, dep), rank))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_value_sim, m)?)?;
    m.add_function(wrap_pyfunction!(_mt_getrandbits, m)?)?;
    m.add_function(wrap_pyfunction!(_gf2_dependent, m)?)?;
    Ok(())
}
