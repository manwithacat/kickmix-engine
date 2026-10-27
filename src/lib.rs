//! kickmix-engine: bit-sliced CPU simulator core, per-shot compatible with kickmix.
//! Written by an AI model (Claude) under the owner's direction; semantics follow kickmix (Apache-2.0, Google LLC).
pub mod engine;
pub mod program;
pub mod rng;
pub mod sim;

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

#[pyclass(name = "Program", module = "kickmix_engine._engine")]
struct PyProgram {
    inner: Arc<program::Program>,
}

#[pymethods]
impl PyProgram {
    #[staticmethod]
    fn from_text(text: &str, angles: Vec<u128>, num_qubits: usize) -> PyResult<PyProgram> {
        let p = program::Program::from_text(text, &angles, num_qubits).map_err(PyValueError::new_err)?;
        Ok(PyProgram { inner: Arc::new(p) })
    }
    #[getter]
    fn num_qubits(&self) -> usize { self.inner.num_qubits }
    #[getter]
    fn num_bits(&self) -> usize { self.inner.num_bits }
    #[getter]
    fn num_ops(&self) -> usize { self.inner.n_source_ops }
    #[getter]
    fn num_stream_ops(&self) -> usize { self.inner.ops.len() }
    #[getter]
    fn num_rand(&self) -> usize { self.inner.nrand }
    #[getter]
    fn bit_map(&self) -> Vec<u32> { self.inner.bit_map.clone() }
}

#[pyfunction]
fn kind_names() -> Vec<&'static str> {
    program::KIND_NAMES[..17].to_vec()
}

/// Keys of the first `nsims` Sims for one simulator seed (8 LE u32 words): u32 array [nsims, 11, 8].
#[pyfunction]
fn derive_keys<'py>(py: Python<'py>, seed: [u32; 8], nsims: usize) -> Bound<'py, PyArray1<u32>> {
    let ks = rng::derive_keys(&seed, nsims);
    let flat: Vec<u32> = ks.iter().flat_map(|k| k.iter().flat_map(|r| r.iter().copied())).collect();
    PyArray1::from_vec(py, flat)
}

#[pyfunction]
#[pyo3(signature = (prog, seeds, xs, xreg, s, b, threads=1, counting=true, dump=false))]
#[allow(clippy::too_many_arguments)]
fn run<'py>(
    py: Python<'py>,
    prog: &PyProgram,
    seeds: PyReadonlyArray2<u32>,
    xs: PyReadonlyArray1<u64>,
    xreg: Vec<u16>,
    s: usize,
    b: usize,
    threads: usize,
    counting: bool,
    dump: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let sd = seeds.as_array();
    if sd.ncols() != 8 {
        return Err(PyValueError::new_err("seeds must be [n_chunks, 8] uint32"));
    }
    let seeds_v: Vec<[u32; 8]> = sd.rows().into_iter().map(|r| { let mut a = [0u32; 8]; for i in 0..8 { a[i] = r[i]; } a }).collect();
    let xs_s = xs.as_slice()?;
    let p = prog.inner.clone();
    let out = py
        .detach(|| {
            let spec = engine::RunSpec { prog: &p, seeds: &seeds_v, s, b, xs: xs_s, xreg: &xreg, counting, dump, threads };
            engine::run(&spec)
        })
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("x_out", PyArray1::from_vec(py, out.x_out))?;
    d.set_item("scratch_dirty", PyArray1::from_vec(py, out.dirty))?;
    d.set_item("phase_units", PyArray1::from_vec(py, out.units))?;
    d.set_item("kind_counts", PyArray1::from_vec(py, out.kind_counts))?;
    d.set_item("z_levels", PyArray1::from_vec(py, out.z_levels))?;
    if dump {
        d.set_item("dump_q", PyArray1::from_vec(py, out.dump.q))?;
        d.set_item("dump_bits", PyArray1::from_vec(py, out.dump.bits))?;
        d.set_item("dump_phase", PyArray1::from_vec(py, out.dump.phase))?;
        d.set_item("dump_vphase", PyArray1::from_vec(py, out.dump.vphase))?;
    }
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgram>()?;
    m.add_function(wrap_pyfunction!(kind_names, m)?)?;
    m.add_function(wrap_pyfunction!(derive_keys, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    Ok(())
}
