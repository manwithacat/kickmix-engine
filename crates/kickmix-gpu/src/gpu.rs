//! wgpu host side: device setup, the templated kernel, state buffers, segment dispatches and readback.
//! Written by an AI model (Claude) under the owner's direction.

use crate::program::*;
use crate::rng::{keys_from_sim_seed, sim_seed_words, Keys, Mt64};
use std::time::Instant;
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("shader.wgsl");
const SCATTER: &str = r#"
alias T = {T};
@group(0) @binding(0) var<storage, read_write> dst: array<T>;
@group(0) @binding(1) var<storage, read> rows: array<u32>;
@group(0) @binding(2) var<storage, read> vals: array<T>;
@group(0) @binding(3) var<uniform> S: vec4<u32>;
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) g: vec3<u32>) {
    let idx = g.x;
    let nt = S.y;
    if (idx >= S.x * nt) { return; }
    let r = idx / nt;
    let t = idx - r * nt;
    dst[rows[r] * nt + t] = vals[idx];
}
"#;

/// Straight-line WGSL for bundles of 1..=kmax CXs: every load before any store, in registers, so the compiler can
/// keep the loads in flight together (the point of the experiment).
fn bundle_cases(kmax: usize) -> String {
    let mut s = String::new();
    for k in 1..=kmax {
        s.push_str(&format!("            case {k}u: {{\n"));
        for j in 0..k {
            s.push_str(&format!("                let r{j} = code[run + {j}u]; let a{j} = (r{j} & 0xFFFFu) * NT + t; let b{j} = (r{j} >> 16u) * NT + t;\n"));
        }
        for j in 0..k {
            s.push_str(&format!("                let x{j} = q[a{j}];\n"));
        }
        for j in 0..k {
            s.push_str(&format!("                let y{j} = q[b{j}];\n"));
        }
        s.push_str("                if (full) {\n");
        for j in 0..k {
            s.push_str(&format!("                    q[b{j}] = y{j} ^ x{j};\n"));
        }
        s.push_str("                } else {\n");
        for j in 0..k {
            s.push_str(&format!("                    q[b{j}] = y{j} ^ (base & x{j});\n"));
        }
        s.push_str("                }\n            }\n");
    }
    s
}

pub const CTXT: usize = 10;
pub const CTXU: usize = 40;

/// Per-thread lane width: 32 shots (u32) or 128 shots (vec4<u32>).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lanes {
    L32,
    L128,
}
impl Lanes {
    pub fn shots(self) -> usize {
        match self {
            Lanes::L32 => 32,
            Lanes::L128 => 128,
        }
    }
    /// u32 words per thread value.
    pub fn ncomp(self) -> usize {
        self.shots() / 32
    }
    pub fn threads_per_batch(self) -> usize {
        256 / self.shots()
    }
    fn fill(self, src: &str) -> String {
        let (t, popc, keyload, shiftv, ones, zero) = match self {
            Lanes::L32 => ("u32", "return countOneBits(x);", "return keys[kb + m];", "u32", "0xFFFFFFFFu", "0u"),
            Lanes::L128 => (
                "vec4<u32>",
                "let p = countOneBits(x); return p.x + p.y + p.z + p.w;",
                "return vec4<u32>(keys[kb + m], keys[kb + 11u + m], keys[kb + 22u + m], keys[kb + 33u + m]);",
                "vec4<u32>",
                "vec4<u32>(0xFFFFFFFFu)",
                "vec4<u32>(0u)",
            ),
        };
        src.replace("{T}", t)
            .replace("{LANES}", &self.shots().to_string())
            .replace("{NCOMP}", &self.ncomp().to_string())
            .replace("{KEYSTRIDE}", &(11 * self.ncomp()).to_string())
            .replace("{POPC}", popc)
            .replace("{KEYLOAD}", keyload)
            .replace("{SHIFTV}", shiftv)
            .replace("{ONES}", ones)
            .replace("{ZERO}", zero)
    }
}

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub lanes: Lanes,
    pub wg: u32,
    pipeline: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    pub adapter_name: String,
    pub backend: String,
    pub max_buffer: u64,
    pub max_binding: u64,
    pub t_init: f64,
    err_flag: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// One 256-shot batch's inputs (bit-sliced as the engine's `simulate_batch` takes them).
pub struct BatchInput<'a> {
    pub seed: &'a [u32; 8],
    /// [qin_ids.len()][4]
    pub qin: &'a [u64],
    /// [bin_ids.len()][4]
    pub bin: &'a [u64],
}

pub struct GpuOut {
    /// [batch][qubit][4] final qubits (CPU `W` layout).
    pub q: Vec<u64>,
    /// [batch][4] phase words.
    pub phase: Vec<u64>,
    /// [batch][NCOUNT] op counts summed over the batch's shots (INFO_TAG included).
    pub kind_counts: Vec<u64>,
    /// [batch][bit][4] final classical bits, only when requested.
    pub bits: Vec<u64>,
    pub rpos: u32,
    pub errors: u32,
    pub t_upload: f64,
    pub t_kernel: f64,
    pub t_readback: f64,
    pub segments: usize,
    pub seg_times: Vec<f64>,
}

/// The eight u32 shot words of a `W` (word k = shots 32k..32k+32).
pub fn lane_words(w: &[u64]) -> [u32; 8] {
    let mut o = [0u32; 8];
    for k in 0..8 {
        o[k] = (w[k / 2] >> (32 * (k % 2))) as u32;
    }
    o
}
fn from_lane_words(o: &[u32; 8]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for k in 0..8 {
        w[k / 2] |= (o[k] as u64) << (32 * (k % 2));
    }
    w
}

/// GPU-only opcode: a bundle of k plain CXs that may be executed as "load every source and target, then store",
/// i.e. for j1 < j2 in the bundle target(j1) is neither a source nor the target of j2 (header `19 | k << 8`,
/// then k entries `a | b << 16` as in a CX run).
pub const K_CXBUNDLE: u8 = 19;

#[derive(Default, Debug, serde::Serialize)]
pub struct BundleStats {
    pub runs: u64,
    pub entries: u64,
    pub bundles: u64,
    pub hist: Vec<u64>,
    pub mean_size: f64,
}

/// Re-encode CX runs of the engine stream as greedy bundles of at most `kmax` independent CXs (other ops verbatim).
pub fn bundle_stream(code: &[u32], kmax: usize) -> (Vec<u32>, BundleStats) {
    let mut out = Vec::with_capacity(code.len() + code.len() / 8);
    let mut st = BundleStats { hist: vec![0; kmax + 1], ..Default::default() };
    let mut i = 0usize;
    let mut targets: Vec<u32> = Vec::with_capacity(kmax);
    while i < code.len() {
        let w0 = code[i];
        let opc = (w0 & 0xFF) as u8;
        if opc == K_CXRUN {
            let n = (w0 >> 8) as usize;
            st.runs += 1;
            st.entries += n as u64;
            let run = &code[i + 1..i + 1 + n];
            let mut j = 0usize;
            while j < n {
                targets.clear();
                let hdr = out.len();
                out.push(0);
                let mut k = 0usize;
                while j < n && k < kmax {
                    let r = run[j];
                    let (a, b) = (r & 0xFFFF, r >> 16);
                    if targets.iter().any(|&t| t == a || t == b) {
                        break;
                    }
                    targets.push(b);
                    out.push(r);
                    k += 1;
                    j += 1;
                }
                out[hdr] = K_CXBUNDLE as u32 | ((k as u32) << 8);
                st.bundles += 1;
                st.hist[k] += 1;
            }
            i += 1 + n;
            continue;
        }
        let len = OP_LEN[opc as usize] as usize;
        out.extend_from_slice(&code[i..i + len]);
        i += len;
    }
    st.mean_size = if st.bundles > 0 { st.entries as f64 / st.bundles as f64 } else { 0.0 };
    (out, st)
}

/// Cut the stream on op boundaries into segments of at most `seg_words` words.
pub fn segments(code: &[u32], seg_words: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let (mut i, mut start) = (0usize, 0usize);
    while i < code.len() {
        let w0 = code[i];
        let opc = (w0 & 0xFF) as u8;
        let len = if opc == K_CXRUN || opc == K_CXBUNDLE { 1 + (w0 >> 8) as usize } else { OP_LEN[opc as usize] as usize };
        assert!(len > 0, "bad opcode in stream");
        if i + len - start > seg_words && i > start {
            out.push((start as u32, i as u32));
            start = i;
        }
        i += len;
    }
    if i > start {
        out.push((start as u32, i as u32));
    }
    out
}

impl Gpu {
    pub fn new(lanes: Lanes, wg: u32, kmax: usize) -> Result<Gpu, String> {
        let t0 = Instant::now();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .map_err(|e| format!("no adapter: {e}"))?;
        let info = adapter.get_info();
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("kickmix-gpu"),
            required_features: wgpu::Features::empty(),
            required_limits: limits.clone(),
            memory_hints: wgpu::MemoryHints::Performance,
            trace: wgpu::Trace::Off,
        }))
        .map_err(|e| format!("request_device: {e}"))?;
        // Any validation or device error makes the run fail (a dropped dispatch must never pass as a result).
        let err_flag = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let ef = err_flag.clone();
        device.on_uncaptured_error(Box::new(move |e| {
            ef.store(true, std::sync::atomic::Ordering::SeqCst);
            eprintln!("wgpu error: {e}");
        }));
        let src = lanes.fill(SHADER).replace("{WG}", &wg.to_string()).replace("{BUNDLE_CASES}", &bundle_cases(kmax));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("sim"),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("sim"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let smod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("scatter"),
            source: wgpu::ShaderSource::Wgsl(lanes.fill(SCATTER).into()),
        });
        let scatter = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("scatter"),
            layout: None,
            module: &smod,
            entry_point: Some("scatter"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Gpu {
            device,
            queue,
            lanes,
            wg,
            pipeline,
            scatter,
            adapter_name: info.name.clone(),
            backend: format!("{:?}", info.backend),
            max_buffer: limits.max_buffer_size,
            max_binding: limits.max_storage_buffer_binding_size as u64,
            t_init: t0.elapsed().as_secs_f64(),
            err_flag,
        })
    }

    pub fn upload_code(&self, code: &[u32]) -> (wgpu::Buffer, f64) {
        let t0 = Instant::now();
        let buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("code"),
            contents: bytemuck::cast_slice(code),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.queue.submit([]);
        let _ = self.device.poll(wgpu::PollType::Wait);
        (buf, t0.elapsed().as_secs_f64())
    }

    fn read_buffer(&self, src: &wgpu::Buffer, offset: u64, size: u64, staging: &wgpu::Buffer) -> Vec<u8> {
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, offset, staging, 0, size);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(0..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll(wgpu::PollType::Wait).expect("poll");
        rx.recv().expect("map callback").expect("map");
        let v = slice.get_mapped_range().to_vec();
        staging.unmap();
        v
    }

    /// Scatter `[row][thread]` values of `vals` into `dst[rows[r] * NT + t]` in chunks of rows.
    fn scatter_rows(&self, dst: &wgpu::Buffer, rows: &[u32], vals: &[u32], nthreads: usize, ncomp: usize) {
        if rows.is_empty() {
            return;
        }
        let per_row = nthreads * ncomp; // u32 words per row
        let chunk_rows = (64usize << 20) / (per_row * 4).max(1);
        // One invocation per (row, thread) in a 1-D dispatch: stay under 65,535 workgroups of 64.
        let chunk_rows = chunk_rows.min((65_535 * 64) / nthreads.max(1)).clamp(1, rows.len());
        let rows_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rows"),
            size: (chunk_rows * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let vals_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vals"),
            size: (chunk_rows * per_row * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sparams"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scatter"),
            layout: &self.scatter.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: rows_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: vals_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            ],
        });
        let mut r0 = 0usize;
        while r0 < rows.len() {
            let n = chunk_rows.min(rows.len() - r0);
            self.queue.write_buffer(&rows_buf, 0, bytemuck::cast_slice(&rows[r0..r0 + n]));
            self.queue.write_buffer(&vals_buf, 0, bytemuck::cast_slice(&vals[r0 * per_row..(r0 + n) * per_row]));
            self.queue.write_buffer(&params, 0, bytemuck::cast_slice(&[n as u32, nthreads as u32, 0u32, 0u32]));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.scatter);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(((n * nthreads) as u32).div_ceil(64), 1, 1);
            }
            self.queue.submit([enc.finish()]);
            let _ = self.device.poll(wgpu::PollType::Wait);
            r0 += n;
        }
    }

    /// Run `prog` on the GPU for `batches` independent 256-shot sims (kickmix `Simulator(batch_size=256, seed)`
    /// each). `code_buf` is the uploaded stream. Returns final state and counts in the CPU engine's layouts.
    #[allow(clippy::too_many_arguments)]
    pub fn run_batches(
        &self,
        prog: &Program,
        code: &[u32],
        code_buf: &wgpu::Buffer,
        batches: &[BatchInput],
        qin_ids: &[u32],
        bin_ids: &[u32],
        seg_words: usize,
        read_bits: bool,
        sync_each: bool,
    ) -> Result<GpuOut, String> {
        if !prog.angles.is_empty() {
            return Err("Z_POW is not supported on the GPU prototype".into());
        }
        if prog.max_stack_depth > 8 {
            return Err("condition stack deeper than 8".into());
        }
        if seg_words > (1 << 24) {
            return Err("seg_words must be <= 2^24 (per-segment u32 counters)".into());
        }
        let nb = batches.len();
        let tpb = self.lanes.threads_per_batch();
        let ncomp = self.lanes.ncomp();
        let nt = nb * tpb;
        let nq = prog.num_qubits.max(1);
        let nbits = prog.num_bits.max(1);
        let tsz = (4 * ncomp) as u64;
        let q_bytes = nq as u64 * nt as u64 * tsz;
        let bits_bytes = nbits as u64 * nt as u64 * tsz;
        if q_bytes > self.max_binding || bits_bytes > self.max_binding || bits_bytes > self.max_buffer {
            return Err(format!(
                "state buffers exceed device limits: q {} MB, bits {} MB, max binding {} MB (use --group)",
                q_bytes >> 20,
                bits_bytes >> 20,
                self.max_binding >> 20
            ));
        }
        let t_up = Instant::now();
        let mk = |label: &str, size: u64| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size.max(16),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let q_buf = mk("q", q_bytes);
        let bits_buf = mk("bits", bits_bytes);
        // Context: phase 0, base ONES, stack 0, depth/rpos/counts 0.
        let mut ctxt = vec![0u32; nt * CTXT * ncomp];
        for t in 0..nt {
            for c in 0..ncomp {
                ctxt[(t * CTXT + 1) * ncomp + c] = u32::MAX;
            }
        }
        let ctxt_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ctxt"),
            contents: bytemuck::cast_slice(&ctxt),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let ctxu_buf = mk("ctxu", (nt * CTXU * 4) as u64);
        // Keys: thread t = b * tpb + g, component c = lane g * ncomp + c.
        let mut keyw = vec![0u32; nt * 11 * ncomp];
        for (b, bi) in batches.iter().enumerate() {
            let mut master = Mt64::from_seed_seq(bi.seed);
            let keys: Keys = keys_from_sim_seed(&sim_seed_words(&mut master));
            for g in 0..tpb {
                let t = b * tpb + g;
                for c in 0..ncomp {
                    let l = g * ncomp + c;
                    for m in 0..11 {
                        keyw[(t * ncomp + c) * 11 + m] = keys[m][l];
                    }
                }
            }
        }
        let keys_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("keys"),
            contents: bytemuck::cast_slice(&keyw),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let params = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Inputs: [row][thread][ncomp] words.
        let pack = |rows: usize, get: &dyn Fn(usize, usize) -> [u64; 4]| -> Vec<u32> {
            let mut v = vec![0u32; rows * nt * ncomp];
            for r in 0..rows {
                for b in 0..nb {
                    let lw = lane_words(&get(b, r));
                    for g in 0..tpb {
                        let t = b * tpb + g;
                        for c in 0..ncomp {
                            v[(r * nt + t) * ncomp + c] = lw[g * ncomp + c];
                        }
                    }
                }
            }
            v
        };
        let qv = pack(qin_ids.len(), &|b, r| {
            let s = &batches[b].qin[4 * r..4 * r + 4];
            [s[0], s[1], s[2], s[3]]
        });
        self.scatter_rows(&q_buf, qin_ids, &qv, nt, ncomp);
        drop(qv);
        let bv = pack(bin_ids.len(), &|b, r| {
            let s = &batches[b].bin[4 * r..4 * r + 4];
            [s[0], s[1], s[2], s[3]]
        });
        self.scatter_rows(&bits_buf, bin_ids, &bv, nt, ncomp);
        drop(bv);
        let _ = self.device.poll(wgpu::PollType::Wait);
        let t_upload = t_up.elapsed().as_secs_f64();

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sim"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: code_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: q_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: bits_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: ctxt_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ctxu_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: keys_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: params.as_entire_binding() },
            ],
        });
        let segs = segments(code, seg_words);
        let groups = (nt as u32).div_ceil(self.wg);
        let mut seg_times = Vec::new();
        let t_k = Instant::now();
        for &(s, e) in &segs {
            let ts = Instant::now();
            self.queue.write_buffer(&params, 0, bytemuck::cast_slice(&[s, e, nt as u32, 0u32]));
            let mut enc = self.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups, 1, 1);
            }
            self.queue.submit([enc.finish()]);
            if sync_each {
                self.device.poll(wgpu::PollType::Wait).map_err(|e| format!("poll: {e}"))?;
                seg_times.push(ts.elapsed().as_secs_f64());
            }
        }
        self.device.poll(wgpu::PollType::Wait).map_err(|e| format!("poll: {e}"))?;
        let t_kernel = t_k.elapsed().as_secs_f64();

        // Readback.
        let t_r = Instant::now();
        let chunk = (256u64 << 20).max((nt * CTXU * 4) as u64).max(q_bytes.min(256 << 20));
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: chunk,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let read_all = |buf: &wgpu::Buffer, bytes: u64| -> Vec<u32> {
            let mut out: Vec<u8> = Vec::with_capacity(bytes as usize);
            let mut off = 0u64;
            while off < bytes {
                let n = chunk.min(bytes - off);
                out.extend_from_slice(&self.read_buffer(buf, off, n, &staging));
                off += n;
            }
            bytemuck::cast_slice(&out).to_vec()
        };
        let qw = read_all(&q_buf, q_bytes);
        let cw = read_all(&ctxt_buf, (nt * CTXT * ncomp * 4) as u64);
        let uw = read_all(&ctxu_buf, (nt * CTXU * 4) as u64);
        let bw = if read_bits { read_all(&bits_buf, bits_bytes) } else { Vec::new() };
        // Reassemble into CPU layouts.
        let gather = |words: &[u32], rows: usize| -> Vec<u64> {
            // words: [row][thread][ncomp] -> [batch][row][4]
            let mut out = vec![0u64; nb * rows * 4];
            for r in 0..rows {
                for b in 0..nb {
                    let mut lw = [0u32; 8];
                    for g in 0..tpb {
                        let t = b * tpb + g;
                        for c in 0..ncomp {
                            lw[g * ncomp + c] = words[(r * nt + t) * ncomp + c];
                        }
                    }
                    let w = from_lane_words(&lw);
                    out[(b * rows + r) * 4..(b * rows + r) * 4 + 4].copy_from_slice(&w);
                }
            }
            out
        };
        let q = gather(&qw, nq);
        let bits = if read_bits { gather(&bw, nbits) } else { Vec::new() };
        let mut phase = vec![0u64; nb * 4];
        let mut kind_counts = vec![0u64; nb * NCOUNT];
        let mut errors = 0u32;
        let mut rpos = 0u32;
        for b in 0..nb {
            let mut lw = [0u32; 8];
            for g in 0..tpb {
                let t = b * tpb + g;
                for c in 0..ncomp {
                    lw[g * ncomp + c] = cw[(t * CTXT) * ncomp + c];
                }
                for k in 0..NCOUNT {
                    let lo = uw[t * CTXU + 2 + k] as u64;
                    let hi = uw[t * CTXU + 20 + k] as u64;
                    kind_counts[b * NCOUNT + k] += lo | (hi << 32);
                }
                errors |= uw[t * CTXU + 38];
                rpos = uw[t * CTXU + 1];
            }
            kind_counts[b * NCOUNT + K_INFO as usize] += prog.n_info_tags * 256;
            phase[b * 4..b * 4 + 4].copy_from_slice(&from_lane_words(&lw));
        }
        let t_readback = t_r.elapsed().as_secs_f64();
        if self.err_flag.load(std::sync::atomic::Ordering::SeqCst) {
            return Err("wgpu reported a validation or device error during this run (see stderr); results discarded".into());
        }
        Ok(GpuOut { q, phase, kind_counts, bits, rpos, errors, t_upload, t_kernel, t_readback, segments: segs.len(), seg_times })
    }
}
