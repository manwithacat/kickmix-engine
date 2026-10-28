//! kickmix-engine: bit-sliced CPU simulator core, per-shot compatible with kickmix.
//! Written by an AI model (Claude) under the owner's direction; semantics follow kickmix (Apache-2.0, Google LLC).
pub mod dqi_sample;
pub mod engine;
pub mod kernel;
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
    fn num_stream_ops(&self) -> usize { self.inner.code.len() }
    #[getter]
    fn num_rand(&self) -> usize { self.inner.nrand }
    #[getter]
    fn bit_map(&self) -> Vec<u32> { self.inner.bit_map.clone() }
    /// Compile a .kmb file directly (streamed; the GIL is released while reading).
    #[staticmethod]
    fn from_kmb(py: Python<'_>, path: String) -> PyResult<PyProgram> {
        let p = py.detach(|| program::Program::from_kmb(&path)).map_err(PyValueError::new_err)?;
        Ok(PyProgram { inner: Arc::new(p) })
    }
    /// original bit id -> compact id (-1 if untouched by any op).
    #[getter]
    fn bit_compact(&self) -> Vec<i64> {
        self.inner.bit_compact.iter().map(|&c| if c == u32::MAX { -1 } else { c as i64 }).collect()
    }
    #[getter]
    fn orig_num_bits(&self) -> usize { self.inner.orig_num_bits }
    #[getter]
    fn num_info_tags(&self) -> u64 { self.inner.n_info_tags }
    #[getter]
    fn max_stack_depth(&self) -> usize { self.inner.max_stack_depth }
    #[getter]
    fn num_cx_runs(&self) -> usize { self.inner.n_cx_runs }
    /// kickmix `Circuit.reaction_depth()`, computed once at compile time and stored in the program cache.
    #[getter]
    fn reaction_depth(&self) -> u64 { self.inner.reaction_depth }
    #[getter]
    fn num_angles(&self) -> usize { self.inner.angles.len() }
    /// Bytes held by the compiled program (code words + angle table + bit maps).
    #[getter]
    fn nbytes(&self) -> usize {
        let p = &self.inner;
        p.code.capacity() * 4 + p.angles.capacity() * 16 + p.levels.capacity() + p.bit_map.capacity() * 4 + p.bit_compact.capacity() * 4
    }
    /// Static per-kind op counts (circuit.operation_statistics()), in kind_names() order.
    #[getter]
    fn static_counts(&self) -> Vec<u64> { self.inner.static_counts.to_vec() }
    /// Write the compiled program to `path` (program-cache format; see `CACHE_FORMAT`). The GIL is released.
    fn save(&self, py: Python<'_>, path: String) -> PyResult<()> {
        let p = self.inner.clone();
        py.detach(|| p.save(&path)).map_err(PyValueError::new_err)
    }
    /// Read a program written by `save` (checksummed and bounds-validated). The GIL is released.
    #[staticmethod]
    fn load(py: Python<'_>, path: String) -> PyResult<PyProgram> {
        let p = py.detach(|| program::Program::load(&path)).map_err(PyValueError::new_err)?;
        Ok(PyProgram { inner: Arc::new(p) })
    }
    /// Registers from a .kmb file: list of (name, [(is_qubit, id)]).
    #[getter]
    fn registers(&self) -> Vec<(String, Vec<(bool, u32)>)> { self.inner.registers.clone() }
}

#[pyfunction]
fn kind_names() -> Vec<&'static str> {
    program::KIND_NAMES.to_vec()
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

/// Independent 256-shot batches (one kickmix Simulator(batch_size=256, seed) each). Inputs bit-sliced:
/// qin [n_batches * len(qin_ids) * 4] u64, bin [n_batches * len(bin_ids) * 4] u64 (compact bit ids).
#[pyfunction]
#[pyo3(signature = (prog, seeds, qin_ids, qin, bin_ids, bin, threads=1, counting=true, dump=false))]
#[allow(clippy::too_many_arguments)]
fn run_batches<'py>(
    py: Python<'py>,
    prog: &PyProgram,
    seeds: PyReadonlyArray2<u32>,
    qin_ids: Vec<u32>,
    qin: PyReadonlyArray1<u64>,
    bin_ids: Vec<u32>,
    bin: PyReadonlyArray1<u64>,
    threads: usize,
    counting: bool,
    dump: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let sd = seeds.as_array();
    if sd.ncols() != 8 {
        return Err(PyValueError::new_err("seeds must be [n_batches, 8] uint32"));
    }
    let seeds_v: Vec<[u32; 8]> = sd.rows().into_iter().map(|r| { let mut a = [0u32; 8]; for i in 0..8 { a[i] = r[i]; } a }).collect();
    let (qs, bs) = (qin.as_slice()?, bin.as_slice()?);
    let p = prog.inner.clone();
    let out = py
        .detach(|| {
            let spec = engine::BatchSpec { prog: &p, seeds: &seeds_v, qin_ids: &qin_ids, qin: qs, bin_ids: &bin_ids, bin: bs, counting, dump, threads };
            engine::run_batches(&spec)
        })
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("val_err", PyArray1::from_vec(py, out.val_err))?;
    d.set_item("phase", PyArray1::from_vec(py, out.phase))?;
    d.set_item("kind_counts", PyArray1::from_vec(py, out.kind_counts))?;
    d.set_item("z_levels", PyArray1::from_vec(py, out.z_levels))?;
    if dump {
        d.set_item("dump_q", PyArray1::from_vec(py, out.dump_q))?;
        d.set_item("dump_bits", PyArray1::from_vec(py, out.dump_bits))?;
    }
    Ok(d)
}

fn parse_pcg(st: &numpy::ndarray::ArrayView2<u64>) -> PyResult<Vec<dqi_sample::Pcg64>> {
    if st.ncols() != 6 {
        return Err(PyValueError::new_err("generator states must be [n, 6] uint64: state_lo, state_hi, inc_lo, inc_hi, has_uint32, uinteger"));
    }
    Ok(st
        .rows()
        .into_iter()
        .map(|r| dqi_sample::Pcg64 {
            state: (r[0] as u128) | ((r[1] as u128) << 64),
            inc: (r[2] as u128) | ((r[3] as u128) << 64),
            has_u32: r[4] != 0,
            u32_buf: r[5] as u32,
        })
        .collect())
}

fn make_sampler(m: usize, n: usize, degree: usize, cdf: Vec<f64>, log: Vec<u32>, antilog: Vec<u32>, e_rows: Vec<u32>) -> PyResult<dqi_sample::DqiSampler> {
    let s = dqi_sample::DqiSampler { m, n, degree, cdf, log, antilog, e_rows };
    s.check().map_err(PyValueError::new_err)?;
    Ok(s)
}

/// Exact DQI `sample_shots` for 256-shot batches. `pcg`: [n, 6] uint64 start states of `default_rng(rng_seed)`.
/// Returns qin [n * n_syn*degree * 4], bin [n * len(e_rows) * 4], phase [n * 4] (u64 words), end_state [n * 6]; with
/// values=True also syndromes [n * 256 * n_syn] and mbuc [n * 256 * m] (uint16).
#[pyfunction]
#[pyo3(signature = (m, n, degree, cdf, log, antilog, e_rows, pcg, threads=1, values=false))]
#[allow(clippy::too_many_arguments)]
fn sample_dqi<'py>(
    py: Python<'py>, m: usize, n: usize, degree: usize, cdf: Vec<f64>, log: Vec<u32>, antilog: Vec<u32>, e_rows: Vec<u32>,
    pcg: PyReadonlyArray2<u64>, threads: usize, values: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let s = make_sampler(m, n, degree, cdf, log, antilog, e_rows)?;
    let st = parse_pcg(&pcg.as_array())?;
    let out = py.detach(|| engine::sample_dqi(&s, &st, threads, values)).map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("qin", PyArray1::from_vec(py, out.qin))?;
    d.set_item("bin", PyArray1::from_vec(py, out.bin))?;
    d.set_item("phase", PyArray1::from_vec(py, out.phase))?;
    d.set_item("end_state", PyArray1::from_vec(py, out.end_state))?;
    if values {
        d.set_item("syndromes", PyArray1::from_vec(py, out.syn))?;
        d.set_item("mbuc", PyArray1::from_vec(py, out.mbuc))?;
    }
    Ok(d)
}

/// DQI batches sampled and simulated on the engine (one Rayon task per batch). Returns val_err, phase (simulated),
/// expected (sampled) [n * 4] u64 words, kind_counts [n * NCOUNT], z_levels [n * 129].
#[pyfunction]
#[pyo3(signature = (prog, m, n, degree, cdf, log, antilog, e_rows, pcg, seeds, qin_ids, bin_ids, threads=1, counting=true))]
#[allow(clippy::too_many_arguments)]
fn run_dqi<'py>(
    py: Python<'py>, prog: &PyProgram, m: usize, n: usize, degree: usize, cdf: Vec<f64>, log: Vec<u32>, antilog: Vec<u32>,
    e_rows: Vec<u32>, pcg: PyReadonlyArray2<u64>, seeds: PyReadonlyArray2<u32>, qin_ids: Vec<u32>, bin_ids: Vec<u32>,
    threads: usize, counting: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let s = make_sampler(m, n, degree, cdf, log, antilog, e_rows)?;
    let st = parse_pcg(&pcg.as_array())?;
    let sd = seeds.as_array();
    if sd.ncols() != 8 {
        return Err(PyValueError::new_err("seeds must be [n_batches, 8] uint32"));
    }
    let seeds_v: Vec<[u32; 8]> = sd.rows().into_iter().map(|r| { let mut a = [0u32; 8]; for i in 0..8 { a[i] = r[i]; } a }).collect();
    let p = prog.inner.clone();
    let out = py
        .detach(|| {
            let spec = engine::DqiSpec { prog: &p, sampler: &s, pcg: &st, seeds: &seeds_v, qin_ids: &qin_ids, bin_ids: &bin_ids, counting, threads };
            engine::run_dqi(&spec)
        })
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("val_err", PyArray1::from_vec(py, out.val_err))?;
    d.set_item("phase", PyArray1::from_vec(py, out.phase))?;
    d.set_item("expected", PyArray1::from_vec(py, out.expected))?;
    d.set_item("kind_counts", PyArray1::from_vec(py, out.kind_counts))?;
    d.set_item("z_levels", PyArray1::from_vec(py, out.z_levels))?;
    Ok(d)
}

/// A sequence of programs on one kickmix `Simulator(batch_size=shots, seed)` (the dfthcsos verifier's loop):
/// step i runs progs[i]; between steps `clear_for_shot()` and the xreg qubits are written back from the previous
/// step's own output. x0: uint64 [shots * xw], xw = ceil(len(xreg)/64). Returns x_out [steps * shots * xw],
/// dirty and phase (uint8 [steps * shots]), kind_counts [steps * NCOUNT], z_levels [steps * 129]; with dump=True
/// also dump_q [steps * sims * dump_nq * 4] and dump_bits [steps * sims * dump_nb * 4] (bit-sliced, 256 shots/sim).
#[pyfunction]
#[pyo3(signature = (progs, seed, x0, xreg, shots, threads=1, counting=true, dump=false))]
#[allow(clippy::too_many_arguments)]
fn run_sequence<'py>(
    py: Python<'py>,
    progs: Vec<PyRef<'py, PyProgram>>,
    seed: [u32; 8],
    x0: PyReadonlyArray1<u64>,
    xreg: Vec<u32>,
    shots: usize,
    threads: usize,
    counting: bool,
    dump: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let arcs: Vec<Arc<program::Program>> = progs.iter().map(|p| p.inner.clone()).collect();
    let x0s = x0.as_slice()?;
    let out = py
        .detach(|| {
            let refs: Vec<&program::Program> = arcs.iter().map(|a| a.as_ref()).collect();
            let spec = engine::SeqSpec { progs: &refs, seed: &seed, shots, xreg: &xreg, x0: x0s, counting, dump, threads };
            engine::run_sequence(&spec)
        })
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("x_out", PyArray1::from_vec(py, out.x_out))?;
    d.set_item("dirty", PyArray1::from_vec(py, out.dirty))?;
    if dump {
        d.set_item("dump_nq", out.dump_nq)?;
        d.set_item("dump_nb", out.dump_nb)?;
        d.set_item("dump_q", PyArray1::from_vec(py, out.dump_q))?;
        d.set_item("dump_bits", PyArray1::from_vec(py, out.dump_bits))?;
    }
    d.set_item("phase", PyArray1::from_vec(py, out.phase))?;
    d.set_item("kind_counts", PyArray1::from_vec(py, out.kind_counts))?;
    d.set_item("z_levels", PyArray1::from_vec(py, out.z_levels))?;
    Ok(d)
}

/// Engine identity for cache keys: (crate version, program-cache format).
#[pyfunction]
fn engine_version() -> (&'static str, u32) {
    (env!("CARGO_PKG_VERSION"), program::CACHE_FORMAT)
}

/// CPU features and kernels, without selecting one (never fails): dict(arch, compiled, available, features,
/// env_request, kernel_features). Used by `kickmix-engine doctor`.
#[pyfunction]
fn cpu_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("arch", std::env::consts::ARCH)?;
    d.set_item("os", std::env::consts::OS)?;
    d.set_item("compiled", kernel::compiled().iter().map(|k| k.name()).collect::<Vec<_>>())?;
    d.set_item("available", kernel::available().iter().map(|k| k.name()).collect::<Vec<_>>())?;
    let f = PyDict::new(py);
    for (n, v) in kernel::detected_features() {
        f.set_item(n, v)?;
    }
    d.set_item("features", f)?;
    let kf = PyDict::new(py);
    for k in kernel::compiled() {
        kf.set_item(k.name(), k.features().to_vec())?;
    }
    d.set_item("kernel_features", kf)?;
    d.set_item("env_var", kernel::ENV)?;
    d.set_item("env_request", std::env::var(kernel::ENV).ok())?;
    Ok(d)
}

/// The kernel this process uses (selected once, at first call): dict(kernel, selected_by, available, compiled,
/// features). Raises ValueError if KICKMIX_ENGINE_KERNEL names a kernel this CPU or build cannot run.
#[pyfunction]
fn kernel_info<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    let (k, how) = kernel::selected().map_err(PyValueError::new_err)?;
    let d = cpu_info(py)?;
    d.set_item("kernel", k.name())?;
    d.set_item("selected_by", how)?;
    d.set_item("kernel_requires", k.features().to_vec())?;
    Ok(d)
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyProgram>()?;
    m.add_function(wrap_pyfunction!(kind_names, m)?)?;
    m.add_function(wrap_pyfunction!(derive_keys, m)?)?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(run_batches, m)?)?;
    m.add_function(wrap_pyfunction!(sample_dqi, m)?)?;
    m.add_function(wrap_pyfunction!(run_dqi, m)?)?;
    m.add_function(wrap_pyfunction!(run_sequence, m)?)?;
    m.add_function(wrap_pyfunction!(engine_version, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_info, m)?)?;
    m.add_function(wrap_pyfunction!(kernel_info, m)?)?;
    Ok(())
}
