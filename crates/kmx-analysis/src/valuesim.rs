//! Tapped bit-parallel value simulator (card E3, review P3).
//!
//! Model, exactly as the Python reference loop (`scripts/qhdl_simulate.py:simulate` and its copies in
//! `dqi_l27c_andprints.py`, `dqi_l27c_blocks.py`, `dqi_l29c_rare.py`, `xpath_l33_dqi.py`): values only, phases ignored.
//! Each lane is one shot; a wire or classical bit is a bitset over L lanes (little-endian u64 words, lane i = bit i).
//! X/CX/CCX/SWAP/CSWAP act exactly, R zeroes, HMR zeroes the wire and writes `getrandbits(L)` (CPython's Mersenne
//! Twister, drawn on every HMR whatever the mask) into its classical bit under the mask; PUSH_CONDITION/POP_CONDITION
//! and the `_IF` forms act per lane; BIT_STORE0/1/BIT_INVERT act on classical bits; Z, CZ, CCZ, Z_POW, NEG and
//! INFO_TAG change no value. The condition mask after a POP with an empty stack is all lanes, as in the Python.
//!
//! Taps (all read the state *before* the op at their index; index `num_ops` means the end state):
//! - points: (op, wire list) -> the wires' values;
//! - windows: (start op, end op, wire list) -> nonzero deltas start ^ end per wire, ordered by (end op, window);
//! - dependence cuts: op -> for the rows [w_0 .. w_{nq-1}, ALL], which rows lie in the span of the others (union of
//!   the null-space supports) and the rank (GF(2) elimination with combination tracking, in worker threads);
//! - Toffoli-like ops (class 0x30: CCZ, CCX, CSWAP and their `_IF` forms), before they act: product lane counts,
//!   fingerprints (blake2b-64 of the lane bytes >> 1, canonical = min over the value and its complement), the flags
//!   of `dqi_l27c_andprints.py`, rare-lane lists (0 < count <= cap) and the target flags of `dqi_l27c_blocks.py`.
use crate::blake2b::fp_words;
use crate::kmb::{read_header, Cur, Header, Mapped};
use crate::mt::Mt19937;
use std::sync::mpsc::{channel, sync_channel, Receiver, Sender, SyncSender};
use std::sync::{Arc, Mutex};

pub enum Outcomes {
    /// CPython `random.Random(seed).getrandbits(L)` per HMR
    Mt(Mt19937),
    /// every HMR outcome 0 (RNG-free check)
    Zero,
    /// explicit outcomes, W words per HMR in circuit order
    Explicit(Vec<u64>),
}

#[derive(Default)]
pub struct Taps {
    pub point_ops: Vec<i64>,
    pub point_ptr: Vec<i64>,
    pub point_wires: Vec<i64>,
    pub win_s: Vec<i64>,
    pub win_e: Vec<i64>,
    pub win_ptr: Vec<i64>,
    pub win_wires: Vec<i64>,
    pub dep_ops: Vec<i64>,
    pub prints: bool,
    pub rare_cap: Option<usize>,
    pub tflag: bool,
}

#[derive(Default)]
pub struct SimOut {
    pub lanes: usize,
    pub words: usize,
    pub ops_run: u64,
    pub n_hmr: u64,
    pub points: Vec<u64>,
    pub d_win: Vec<i64>,
    pub d_wire: Vec<i64>,
    pub d_vec: Vec<u64>,
    pub d_before0: Vec<bool>,
    pub d_after0: Vec<bool>,
    pub dep: Vec<bool>,
    pub dep_rank: Vec<i64>,
    pub and_op: Vec<i64>,
    pub and_kind: Vec<i8>,
    pub and_pw: Vec<i32>,
    pub and_flag: Vec<i8>,
    pub and_p: Vec<i64>,
    pub and_pc: Vec<i64>,
    pub and_o1: Vec<i64>,
    pub and_o2: Vec<i64>,
    pub rare_lanes: Vec<i32>,
    pub rare_ptr: Vec<i64>,
    pub tflag: Vec<i8>,
    pub end_wires: Vec<u64>,
    pub end_cbits: Vec<u64>,
}

// ------------------------------------------------------------------------------------------------ worker pool jobs
enum Job {
    /// fingerprints of a batch of (p, u, v) triples; `base` = index of the first AND record
    Prints { base: usize, data: Vec<u64> },
    /// dependence at one cut: rows nq values followed by ALL
    Dep { k: usize, rows: Vec<u64>, n: usize },
}
enum Res {
    Prints { base: usize, out: Vec<[i64; 4]> },
    Dep { k: usize, dep: Vec<bool>, rank: usize },
}

fn worker(rx: Arc<Mutex<Receiver<Job>>>, tx: Sender<Res>, w: usize, lanes: usize) {
    let nb = (lanes + 7) / 8;
    let all = all_words(lanes, w);
    let mut buf = Vec::with_capacity(8 * w);
    let mut comp = vec![0u64; w];
    loop {
        let job = {
            let g = rx.lock().unwrap();
            g.recv()
        };
        let job = match job {
            Ok(j) => j,
            Err(_) => return,
        };
        match job {
            Job::Prints { base, data } => {
                let n = data.len() / (3 * w);
                let mut out = Vec::with_capacity(n);
                let fp = |v: &[u64], buf: &mut Vec<u8>| (fp_words(v, nb, buf) >> 1) as i64;
                for i in 0..n {
                    let e = &data[3 * w * i..3 * w * (i + 1)];
                    let mut cf = [0i64; 3];
                    let mut fpp = 0i64;
                    for j in 0..3 {
                        let v = &e[j * w..(j + 1) * w];
                        for k in 0..w {
                            comp[k] = v[k] ^ all[k];
                        }
                        let x = fp(v, &mut buf);
                        let y = fp(&comp, &mut buf);
                        if j == 0 {
                            fpp = x;
                        }
                        cf[j] = if x < y { x } else { y };
                    }
                    let (c1, c2) = if cf[2] < cf[1] { (cf[2], cf[1]) } else { (cf[1], cf[2]) };
                    out.push([fpp, cf[0], c1, c2]);
                }
                let _ = tx.send(Res::Prints { base, out });
            }
            Job::Dep { k, rows, n } => {
                let (dep, rank) = dependent(&rows, n, w);
                let _ = tx.send(Res::Dep { k, dep, rank });
            }
        }
    }
}

fn all_words(lanes: usize, w: usize) -> Vec<u64> {
    let mut a = vec![u64::MAX; w];
    if lanes % 64 != 0 {
        a[w - 1] = (1u64 << (lanes % 64)) - 1;
    }
    if lanes == 0 {
        a.iter_mut().for_each(|x| *x = 0);
    }
    a
}

/// Rows `rows[i*w..(i+1)*w]`, i < n. Returns (row i lies in the span of the other rows, rank): the union of the
/// supports of the combinations that reduce a row to zero (a basis of the left null space), as `xpath_l33_dqi.eliminate`.
pub fn dependent(rows: &[u64], n: usize, w: usize) -> (Vec<bool>, usize) {
    let cw = (n + 63) / 64;
    let nbits = 64 * w;
    let mut piv: Vec<i32> = vec![-1; nbits];
    let mut bv: Vec<u64> = Vec::new();
    let mut bc: Vec<u64> = Vec::new();
    let mut depm = vec![0u64; cw];
    let mut x = vec![0u64; w];
    let mut c = vec![0u64; cw];
    let mut rank = 0usize;
    for idx in 0..n {
        x.copy_from_slice(&rows[idx * w..(idx + 1) * w]);
        c.iter_mut().for_each(|v| *v = 0);
        c[idx / 64] |= 1u64 << (idx % 64);
        let mut top = w; // words above `top` are zero
        loop {
            while top > 0 && x[top - 1] == 0 {
                top -= 1;
            }
            if top == 0 {
                for k in 0..cw {
                    depm[k] |= c[k];
                }
                break;
            }
            let hb = 64 * (top - 1) + 63 - x[top - 1].leading_zeros() as usize;
            let p = piv[hb];
            if p < 0 {
                piv[hb] = rank as i32;
                bv.extend_from_slice(&x);
                bc.extend_from_slice(&c);
                rank += 1;
                break;
            }
            let p = p as usize;
            let bvr = &bv[p * w..p * w + top];
            for k in 0..top {
                x[k] ^= bvr[k];
            }
            let bcr = &bc[p * cw..(p + 1) * cw];
            for k in 0..cw {
                c[k] ^= bcr[k];
            }
        }
    }
    let dep = (0..n).map(|i| depm[i / 64] >> (i % 64) & 1 == 1).collect();
    (dep, rank)
}

// ------------------------------------------------------------------------------------------------ the simulator
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Ev {
    // order within one op index: window starts, points, dependence cuts, window ends (all are reads; window ends
    // are ordered by window index, which with nondecreasing end ops is the Python append order)
    WinStart(usize),
    Point(usize),
    Dep(usize),
    WinEnd(usize),
}

#[inline(always)]
unsafe fn row(p: *mut u64, i: usize, w: usize) -> *mut u64 {
    p.add(i * w)
}

/// A shared output buffer that lane slices write into at disjoint word ranges.
#[derive(Clone, Copy)]
struct SPtr(*mut u64);
unsafe impl Send for SPtr {}
unsafe impl Sync for SPtr {}
impl SPtr {
    /// copy `src` to words [at, at + src.len()) (callers write disjoint ranges or are serialised by a barrier)
    #[inline]
    unsafe fn put(self, at: usize, src: &[u64]) {
        std::ptr::copy_nonoverlapping(src.as_ptr(), self.0.add(at), src.len());
    }
}

// per-slice flag bits of a Toffoli record (AND-combined over slices)
const F_PZ: u16 = 1; // product zero
const F_PA: u16 = 2; // product all lanes
const F_UZ: u16 = 4;
const F_UA: u16 = 8;
const F_VZ: u16 = 16;
const F_VA: u16 = 32;
const F_PU: u16 = 64; // product == first operand
const F_PV: u16 = 128; // product == second operand
const F_DP: u16 = 256; // CCX target == product (blocks flag 16)
const F_DZ: u16 = 512; // CCX target zero (blocks flag 32)
// per-slice window-delta bits (nonzero: OR; zero before / after: AND)
const W_NZ: u8 = 1;
const W_S0: u8 = 2;
const W_S1: u8 = 4;
const PB: usize = 2048; // fingerprint batch (ANDs)

struct Shared<'a> {
    path: &'a str,
    map: &'a Mapped,
    h: &'a Header,
    nq: usize,
    ncb: usize,
    w: usize,
    lanes: usize,
    end: u64,
    all: &'a [u64],
    taps: &'a Taps,
    evs: &'a [(u64, Ev)],
    wires: &'a [u64],
    cbits: &'a [u64],
    points: SPtr,
    dbuf: SPtr,
    endw: SPtr,
    endc: SPtr,
    depsnap: SPtr,
    pbatch: SPtr,
    barrier: &'a std::sync::Barrier,
    rec_and: bool,
}

#[derive(Default)]
struct SliceOut {
    and_op: Vec<i64>,
    and_kind: Vec<i8>,
    pw: Vec<u32>,
    fl: Vec<u16>,
    rare_lanes: Vec<i32>,
    rare_ptr: Vec<usize>,
    win_fl: Vec<u8>,
    n_hmr: u64,
}

/// Run every op on the lane words [w0, w1). All slices decode the same streams, so they see the same ops, taps and
/// Toffoli records in the same order; barriers serialise the taps that need whole rows (dependence cuts and
/// fingerprint batches).
fn run_slice(sh: &Shared, mut outcomes: Outcomes, w0: usize, w1: usize, jtx: SyncSender<Job>) -> Result<SliceOut, String> {
    let (h, nq, ncb, w, taps) = (sh.h, sh.nq, sh.ncb, sh.w, sh.taps);
    let ws = w1 - w0;
    let all = &sh.all[w0..w1];
    let path = sh.path;
    let mut o_ = SliceOut::default();
    let nwin_slots = taps.win_wires.len();
    if !taps.win_s.is_empty() {
        o_.win_fl = vec![0u8; nwin_slots];
    }
    let mut win_start: Vec<Option<Vec<u64>>> = vec![None; taps.win_s.len()];
    // ---- state (this slice)
    let mut st: Vec<u64> = Vec::with_capacity(nq * ws);
    for q in 0..nq {
        st.extend_from_slice(&sh.wires[q * w + w0..q * w + w1]);
    }
    let mut cb: Vec<u64> = Vec::with_capacity(ncb * ws);
    for b in 0..ncb {
        cb.extend_from_slice(&sh.cbits[b * w + w0..b * w + w1]);
    }
    let mut mask: Vec<u64> = all.to_vec();
    let mut mask_all = true; // mask == all lanes of this slice (exact test after every PUSH/POP)
    let mut stack: Vec<Vec<u64>> = Vec::new();
    let mut spare: Vec<Vec<u64>> = Vec::new();
    let mut mkbuf = vec![0u64; ws];
    let mut rnd_full = vec![0u64; w];
    let mut tmp = vec![0u64; ws];
    let mut tmp2 = vec![0u64; ws];
    // ---- streams (shared memory map)
    let m = sh.map;
    let mut ty = Cur::new(m, h.offs[0], h.num_ops)?;
    let mut s3 =
        [Cur::new(m, h.offs[1], h.n_qqq * 4)?, Cur::new(m, h.offs[2], h.n_qqq * 4)?, Cur::new(m, h.offs[3], h.n_qqq * 4)?];
    let mut s2 = [Cur::new(m, h.offs[4], h.n_qq * 4)?, Cur::new(m, h.offs[5], h.n_qq * 4)?];
    let mut s1 = Cur::new(m, h.offs[6], h.n_q * 4)?;
    let mut sb = Cur::new(m, h.off_b, h.n_b * 4)?;
    let mut sc = Cur::new(m, h.off_c, h.n_c * 4)?;
    let bad_q = |q: u32| -> Result<usize, String> {
        if (q as usize) < nq {
            Ok(q as usize)
        } else {
            Err(format!("qubit id {} out of range", q))
        }
    };
    let bad_b = |b: u32| -> Result<usize, String> {
        if (b as usize) < ncb {
            Ok(b as usize)
        } else {
            Err(format!("classical bit id {} out of range", b))
        }
    };
    let sp = st.as_mut_ptr();
    let cp = cb.as_mut_ptr();
    let rec_and = sh.rec_and;
    let rec_any = rec_and || taps.tflag;
    let rare_cap = taps.rare_cap;
    let mut n_and = 0usize;
    let mut ei = 0usize;
    let evs = sh.evs;
    let send_batch = |n: usize, base: usize| -> Result<(), String> {
        // leader only: copy the first n entries of the shared batch and queue them for hashing
        let mut data = vec![0u64; n * 3 * w];
        unsafe { std::ptr::copy_nonoverlapping(sh.pbatch.0, data.as_mut_ptr(), n * 3 * w) };
        jtx.send(Job::Prints { base, data }).map_err(|_| "worker pool stopped".to_string())
    };
    macro_rules! events_at {
        ($o:expr) => {
            while ei < evs.len() && evs[ei].0 == $o {
                match evs[ei].1 {
                    Ev::Point(i) => {
                        let (a, b) = (taps.point_ptr[i] as usize, taps.point_ptr[i + 1] as usize);
                        for slot in a..b {
                            let q = taps.point_wires[slot] as usize;
                            unsafe { sh.points.put(slot * w + w0, &st[q * ws..(q + 1) * ws]) };
                        }
                    }
                    Ev::WinStart(j) => {
                        let (a, b) = (taps.win_ptr[j] as usize, taps.win_ptr[j + 1] as usize);
                        let mut v = Vec::with_capacity((b - a) * ws);
                        for &q in &taps.win_wires[a..b] {
                            let q = q as usize;
                            v.extend_from_slice(&st[q * ws..(q + 1) * ws]);
                        }
                        win_start[j] = Some(v);
                    }
                    Ev::WinEnd(j) => {
                        let (a, b) = (taps.win_ptr[j] as usize, taps.win_ptr[j + 1] as usize);
                        let v0 = win_start[j].take().ok_or_else(|| "window end before its start".to_string())?;
                        for (r, slot) in (a..b).enumerate() {
                            let q = taps.win_wires[slot] as usize;
                            let s0 = &v0[r * ws..(r + 1) * ws];
                            let s1 = &st[q * ws..(q + 1) * ws];
                            for k in 0..ws {
                                tmp[k] = s0[k] ^ s1[k];
                            }
                            let mut f = 0u8;
                            if tmp.iter().any(|&x| x != 0) {
                                f |= W_NZ;
                            }
                            if s0.iter().all(|&x| x == 0) {
                                f |= W_S0;
                            }
                            if s1.iter().all(|&x| x == 0) {
                                f |= W_S1;
                            }
                            o_.win_fl[slot] = f;
                            unsafe { sh.dbuf.put(slot * w + w0, &tmp) };
                        }
                    }
                    Ev::Dep(k) => {
                        for q in 0..nq {
                            unsafe { sh.depsnap.put(q * w + w0, &st[q * ws..(q + 1) * ws]) };
                        }
                        if sh.barrier.wait().is_leader() {
                            let mut rows = vec![0u64; (nq + 1) * w];
                            unsafe { std::ptr::copy_nonoverlapping(sh.depsnap.0, rows.as_mut_ptr(), (nq + 1) * w) };
                            jtx.send(Job::Dep { k, rows, n: nq + 1 }).map_err(|_| "worker pool stopped".to_string())?;
                        }
                        sh.barrier.wait();
                    }
                }
                ei += 1;
            }
        };
    }
    for o in 0..sh.end {
        events_at!(o);
        let t = ty.u8()?;
        let cls = t & 0x30;
        let (mut a0, mut a1, mut a2) = (0usize, 0usize, 0usize);
        match cls {
            0x30 => {
                a0 = bad_q(s3[0].u32()?)?;
                a1 = bad_q(s3[1].u32()?)?;
                a2 = bad_q(s3[2].u32()?)?;
            }
            0x20 => {
                a0 = bad_q(s2[0].u32()?)?;
                a1 = bad_q(s2[1].u32()?)?;
            }
            0x10 => {
                a0 = bad_q(s1.u32()?)?;
            }
            _ => {}
        }
        if t & 0x40 != 0 && (cls == 0x20 || cls == 0x30) {
            return Err(format!("op {}: unknown op type 0x{:02x}", o, t));
        }
        let bidx = if t & 0x40 != 0 { Some(bad_b(sb.u32()?)?) } else { None };
        let cidx = if t & 0x80 != 0 { Some(bad_b(sc.u32()?)?) } else { None };
        if t == 0x06 {
            continue;
        }
        if t == 0x81 {
            let c = cidx.unwrap();
            let mut old = spare.pop().unwrap_or_else(|| vec![0u64; ws]);
            old.copy_from_slice(&mask);
            stack.push(old);
            for k in 0..ws {
                mask[k] &= cb[c * ws + k];
            }
            mask_all = mask == all;
            continue;
        }
        if t == 0x01 {
            match stack.pop() {
                Some(m) => {
                    mask.copy_from_slice(&m);
                    spare.push(m);
                }
                None => mask.copy_from_slice(all),
            }
            mask_all = mask == all;
            continue;
        }
        let base = t & 0x7F;
        // ops that change no value (Z, CZ, Z_POW, NEG; CCZ unless Toffoli taps are on): the Python still reads
        // their condition bit, which has no effect, so nothing to do
        if matches!(base, 0x10 | 0x20 | 0x13 | 0x00) || (base == 0x30 && !rec_any) {
            continue;
        }
        if cidx.is_none() && mask_all {
            unsafe {
                match base {
                    0x21 => {
                        let (x0, x1) = (row(sp, a0, ws), row(sp, a1, ws));
                        for k in 0..ws {
                            let x = *x0.add(k);
                            *x1.add(k) ^= x;
                        }
                        continue;
                    }
                    0x31 if !rec_any => {
                        let (x0, x1, x2) = (row(sp, a0, ws), row(sp, a1, ws), row(sp, a2, ws));
                        for k in 0..ws {
                            let p = *x0.add(k) & *x1.add(k);
                            *x2.add(k) ^= p;
                        }
                        continue;
                    }
                    0x22 => {
                        let (x0, x1) = (row(sp, a0, ws), row(sp, a1, ws));
                        for k in 0..ws {
                            let (a, d) = (*x0.add(k), *x1.add(k));
                            *x0.add(k) = d;
                            *x1.add(k) = a;
                        }
                        continue;
                    }
                    _ => {}
                }
            }
        }
        let mk: *const u64 = if let Some(c) = cidx {
            for k in 0..ws {
                mkbuf[k] = mask[k] & cb[c * ws + k];
            }
            mkbuf.as_ptr()
        } else {
            mask.as_ptr()
        };
        unsafe {
            match cls {
                0x30 => {
                    let (x0, x1, x2) = (row(sp, a0, ws), row(sp, a1, ws), row(sp, a2, ws));
                    let kind: i8 = match base {
                        0x31 => 1,
                        0x32 => 2,
                        0x30 => 0,
                        _ => -1,
                    };
                    if kind >= 0 && rec_any {
                        // product p and second operand v of this slice, before the op (tmp = p, tmp2 = v)
                        for k in 0..ws {
                            let (a, b, d, m) = (*x0.add(k), *x1.add(k), *x2.add(k), *mk.add(k));
                            let u = match kind {
                                1 => b,
                                2 => b ^ d,
                                _ => b & d,
                            };
                            tmp2[k] = u;
                            tmp[k] = a & u & m;
                        }
                        let ua = std::slice::from_raw_parts(x0 as *const u64, ws);
                        let dd = std::slice::from_raw_parts(x2 as *const u64, ws);
                        let (p, v) = (&tmp[..], &tmp2[..]);
                        let z = |s: &[u64]| s.iter().all(|&x| x == 0);
                        let isall = |s: &[u64]| s == all;
                        let mut f = 0u16;
                        if z(p) {
                            f |= F_PZ;
                        }
                        if isall(p) {
                            f |= F_PA;
                        }
                        if z(ua) {
                            f |= F_UZ;
                        }
                        if isall(ua) {
                            f |= F_UA;
                        }
                        if z(v) {
                            f |= F_VZ;
                        }
                        if isall(v) {
                            f |= F_VA;
                        }
                        if p == ua {
                            f |= F_PU;
                        }
                        if p == v {
                            f |= F_PV;
                        }
                        if dd == p {
                            f |= F_DP;
                        }
                        if z(dd) {
                            f |= F_DZ;
                        }
                        o_.fl.push(f);
                        o_.and_op.push(o as i64);
                        o_.and_kind.push(kind);
                        let pw: u32 = p.iter().map(|x| x.count_ones()).sum();
                        o_.pw.push(pw);
                        if let Some(cap) = rare_cap {
                            if pw > 0 && pw as usize <= cap {
                                for (k, &x) in p.iter().enumerate() {
                                    let mut y = x;
                                    while y != 0 {
                                        o_.rare_lanes.push((64 * (w0 + k)) as i32 + y.trailing_zeros() as i32);
                                        y &= y - 1;
                                    }
                                }
                            }
                            o_.rare_ptr.push(o_.rare_lanes.len());
                        }
                        if taps.prints {
                            let slot = n_and % PB;
                            sh.pbatch.put(slot * 3 * w + w0, p);
                            sh.pbatch.put(slot * 3 * w + w + w0, ua);
                            sh.pbatch.put(slot * 3 * w + 2 * w + w0, v);
                            if slot == PB - 1 {
                                if sh.barrier.wait().is_leader() {
                                    send_batch(PB, n_and + 1 - PB)?;
                                }
                                sh.barrier.wait();
                            }
                        }
                        n_and += 1;
                    }
                    match base {
                        0x31 => {
                            for k in 0..ws {
                                let p = *x0.add(k) & *x1.add(k) & *mk.add(k);
                                *x2.add(k) ^= p;
                            }
                        }
                        0x32 => {
                            for k in 0..ws {
                                let x = *x0.add(k) & (*x1.add(k) ^ *x2.add(k)) & *mk.add(k);
                                *x1.add(k) ^= x;
                                *x2.add(k) ^= x;
                            }
                        }
                        _ => {}
                    }
                }
                0x20 => {
                    let (x0, x1) = (row(sp, a0, ws), row(sp, a1, ws));
                    match base {
                        0x21 => {
                            for k in 0..ws {
                                let x = *x0.add(k) & *mk.add(k);
                                *x1.add(k) ^= x;
                            }
                        }
                        0x22 => {
                            for k in 0..ws {
                                let x = (*x0.add(k) ^ *x1.add(k)) & *mk.add(k);
                                *x0.add(k) ^= x;
                                *x1.add(k) ^= x;
                            }
                        }
                        _ => {}
                    }
                }
                0x10 => {
                    let x0 = row(sp, a0, ws);
                    match base {
                        0x11 => {
                            for k in 0..ws {
                                *x0.add(k) ^= *mk.add(k);
                            }
                        }
                        0x12 => {
                            for k in 0..ws {
                                *x0.add(k) &= !*mk.add(k);
                            }
                        }
                        0x50 => {
                            for k in 0..ws {
                                *x0.add(k) &= !*mk.add(k);
                            }
                            match &mut outcomes {
                                Outcomes::Mt(g) => g.getrandbits_into(sh.lanes, &mut rnd_full),
                                Outcomes::Zero => rnd_full.iter_mut().for_each(|x| *x = 0),
                                Outcomes::Explicit(v) => {
                                    let i = o_.n_hmr as usize;
                                    if v.len() < (i + 1) * w {
                                        return Err("explicit outcomes exhausted".into());
                                    }
                                    rnd_full.copy_from_slice(&v[i * w..(i + 1) * w]);
                                }
                            }
                            o_.n_hmr += 1;
                            let bb = row(cp, bidx.unwrap(), ws);
                            for k in 0..ws {
                                let m = *mk.add(k);
                                *bb.add(k) = (*bb.add(k) & !m) | (rnd_full[w0 + k] & m);
                            }
                        }
                        _ => {}
                    }
                }
                _ => {
                    if let Some(b) = bidx {
                        let bb = row(cp, b, ws);
                        match base {
                            0x40 => {
                                for k in 0..ws {
                                    *bb.add(k) &= !*mk.add(k);
                                }
                            }
                            0x41 => {
                                for k in 0..ws {
                                    *bb.add(k) |= *mk.add(k);
                                }
                            }
                            0x42 => {
                                for k in 0..ws {
                                    *bb.add(k) ^= *mk.add(k);
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
        }
    }
    events_at!(sh.end);
    if taps.prints && n_and % PB != 0 {
        if sh.barrier.wait().is_leader() {
            send_batch(n_and % PB, n_and - n_and % PB)?;
        }
        sh.barrier.wait();
    }
    if sh.end == h.num_ops
        && (!ty.exhausted() || s3.iter().any(|s| !s.exhausted()) || s2.iter().any(|s| !s.exhausted()) || !s1.exhausted()
            || !sb.exhausted() || !sc.exhausted())
    {
        return Err(format!("{}: operand streams do not match the op-type stream", path));
    }
    for q in 0..nq {
        unsafe { sh.endw.put(q * w + w0, &st[q * ws..(q + 1) * ws]) };
    }
    for b in 0..ncb {
        unsafe { sh.endc.put(b * w + w0, &cb[b * ws..(b + 1) * ws]) };
    }
    Ok(o_)
}

/// Run the simulator. `wires` holds nq*W words, `cbits` holds (num_bits+1)*W words (or more), W = ceil(lanes/64).
/// `stop`: simulate ops [0, stop) only (taps at index stop read the state there). `threads`: lane slices run in
/// parallel (at most W/8 slices, at least 8 words each) plus as many hashing / elimination workers.
#[allow(clippy::too_many_arguments)]
pub fn simulate(
    path: &str,
    lanes: usize,
    wires: &[u64],
    cbits: &[u64],
    outcomes: Outcomes,
    taps: &Taps,
    stop: Option<u64>,
    threads: usize,
) -> Result<SimOut, String> {
    let h: Header = read_header(path)?;
    let nq = h.num_qubits as usize;
    let w = (lanes + 63) / 64;
    if lanes == 0 {
        return Err("lanes must be positive".into());
    }
    if wires.len() != nq * w {
        return Err(format!("wires: expected {} words ({} qubits x {}), got {}", nq * w, nq, w, wires.len()));
    }
    if cbits.len() % w != 0 || cbits.len() / w < h.num_bits as usize + 1 {
        return Err("cbits: expected (num_bits + 1) x W words or more".into());
    }
    let ncb = cbits.len() / w;
    let all = all_words(lanes, w);
    let end = stop.map(|s| s.min(h.num_ops)).unwrap_or(h.num_ops);
    // ---- events
    let check_wires = |ws: &[i64], what: &str| -> Result<(), String> {
        if ws.iter().any(|&q| q < 0 || q as usize >= nq) {
            return Err(format!("{}: wire id out of range", what));
        }
        Ok(())
    };
    let check_ptr = |ptr: &[i64], n: usize, len: usize, what: &str| -> Result<(), String> {
        if ptr.len() != n + 1 || ptr[0] != 0 || ptr[n] as usize != len || ptr.windows(2).any(|x| x[0] > x[1]) {
            return Err(format!("{}: bad CSR offsets", what));
        }
        Ok(())
    };
    let np_ = taps.point_ops.len();
    let nw_ = taps.win_s.len();
    check_ptr(&taps.point_ptr, np_, taps.point_wires.len(), "points")?;
    check_wires(&taps.point_wires, "points")?;
    if taps.win_e.len() != nw_ {
        return Err("windows: start/end lengths differ".into());
    }
    check_ptr(&taps.win_ptr, nw_, taps.win_wires.len(), "windows")?;
    check_wires(&taps.win_wires, "windows")?;
    let mut evs: Vec<(u64, Ev)> = Vec::with_capacity(np_ + 2 * nw_ + taps.dep_ops.len());
    let chk_op = |o: i64| -> Result<u64, String> {
        if o < 0 || o as u64 > end {
            Err(format!("tap at op {} outside 0..={}", o, end))
        } else {
            Ok(o as u64)
        }
    };
    for (i, &o) in taps.point_ops.iter().enumerate() {
        evs.push((chk_op(o)?, Ev::Point(i)));
    }
    for j in 0..nw_ {
        if taps.win_s[j] > taps.win_e[j] {
            return Err("window ends before it starts".into());
        }
        evs.push((chk_op(taps.win_s[j])?, Ev::WinStart(j)));
        evs.push((chk_op(taps.win_e[j])?, Ev::WinEnd(j)));
    }
    for (k, &o) in taps.dep_ops.iter().enumerate() {
        evs.push((chk_op(o)?, Ev::Dep(k)));
    }
    evs.sort();
    // ---- shared buffers
    let mut out = SimOut { lanes, words: w, ..Default::default() };
    out.points = vec![0u64; taps.point_wires.len() * w];
    out.dep = vec![false; taps.dep_ops.len() * (nq + 1)];
    out.dep_rank = vec![0; taps.dep_ops.len()];
    let mut dbuf = vec![0u64; if nw_ > 0 { taps.win_wires.len() * w } else { 0 }];
    let mut endw = vec![0u64; nq * w];
    let mut endc = vec![0u64; ncb * w];
    let mut depsnap = vec![0u64; if taps.dep_ops.is_empty() { 0 } else { (nq + 1) * w }];
    if !taps.dep_ops.is_empty() {
        depsnap[nq * w..].copy_from_slice(&all);
    }
    let mut pbatch = vec![0u64; if taps.prints { PB * 3 * w } else { 0 }];
    let nlt = threads.max(1).min((w / 8).max(1));
    let bounds: Vec<(usize, usize)> = (0..nlt).map(|t| (w * t / nlt, w * (t + 1) / nlt)).collect();
    let barrier = std::sync::Barrier::new(nlt);
    let map = Mapped::open(path)?;
    let sh = Shared {
        path,
        map: &map,
        h: &h,
        nq,
        ncb,
        w,
        lanes,
        end,
        all: &all,
        taps,
        evs: &evs,
        wires,
        cbits,
        points: SPtr(out.points.as_mut_ptr()),
        dbuf: SPtr(dbuf.as_mut_ptr()),
        endw: SPtr(endw.as_mut_ptr()),
        endc: SPtr(endc.as_mut_ptr()),
        depsnap: SPtr(depsnap.as_mut_ptr()),
        pbatch: SPtr(pbatch.as_mut_ptr()),
        barrier: &barrier,
        rec_and: taps.prints || taps.rare_cap.is_some(),
    };
    let npool = threads.max(1);
    let need_pool = taps.prints || !taps.dep_ops.is_empty();
    let mut outs: Vec<Result<SliceOut, String>> = Vec::new();
    let mut results: Vec<Res> = Vec::new();
    std::thread::scope(|scope| {
        let (jtx, jrx): (SyncSender<Job>, Receiver<Job>) = sync_channel(2 * npool + 2);
        let (rtx, rrx) = channel::<Res>();
        let jrx = Arc::new(Mutex::new(jrx));
        if need_pool {
            for _ in 0..npool {
                let (r, t) = (jrx.clone(), rtx.clone());
                scope.spawn(move || worker(r, t, w, lanes));
            }
        }
        drop(rtx);
        let mut handles = Vec::new();
        let mut oc = Some(outcomes);
        for (t, &(w0, w1)) in bounds.iter().enumerate() {
            let o = if t + 1 == nlt {
                oc.take().unwrap()
            } else {
                match oc.as_ref().unwrap() {
                    Outcomes::Mt(g) => Outcomes::Mt(g.clone()),
                    Outcomes::Zero => Outcomes::Zero,
                    Outcomes::Explicit(v) => Outcomes::Explicit(v.clone()),
                }
            };
            let tx = jtx.clone();
            let shr = &sh;
            handles.push(scope.spawn(move || run_slice(shr, o, w0, w1, tx)));
        }
        drop(jtx);
        for hd in handles {
            outs.push(hd.join().unwrap_or_else(|_| Err("lane worker panicked".into())));
        }
        results.extend(rrx.iter());
    });
    let outs: Vec<SliceOut> = outs.into_iter().collect::<Result<_, _>>()?;
    // ---- merge the slices
    let o0 = &outs[0];
    let n_and = o0.and_op.len();
    if outs.iter().any(|s| s.and_op.len() != n_and) {
        return Err("lane slices disagree on the Toffoli records".into());
    }
    out.n_hmr = o0.n_hmr;
    out.ops_run = end;
    let mut fl = vec![u16::MAX; n_and];
    let mut pw = vec![0u32; n_and];
    for s in &outs {
        for i in 0..n_and {
            fl[i] &= s.fl[i];
            pw[i] += s.pw[i];
        }
    }
    if sh.rec_and {
        out.and_op = o0.and_op.clone();
        out.and_kind = o0.and_kind.clone();
        out.and_pw = pw.iter().map(|&x| x as i32).collect();
    }
    if taps.prints {
        out.and_flag = fl
            .iter()
            .map(|&g| {
                let mut f = 0i8;
                if g & F_PZ != 0 {
                    f |= 1;
                }
                if g & F_PA != 0 {
                    f |= 2;
                }
                if g & (F_UZ | F_UA | F_VZ | F_VA) != 0 {
                    // AND-combined per bit: "u is 0 on every slice" etc.; any of the four makes flag 4
                    f |= 4;
                }
                if g & (F_PU | F_PV) != 0 {
                    f |= 8;
                }
                f
            })
            .collect();
    }
    if let Some(cap) = taps.rare_cap {
        let mut starts = vec![0usize; outs.len()];
        for i in 0..n_and {
            let tot = pw[i] as usize;
            for (si, s) in outs.iter().enumerate() {
                let e = s.rare_ptr[i];
                if tot > 0 && tot <= cap {
                    out.rare_lanes.extend_from_slice(&s.rare_lanes[starts[si]..e]);
                }
                starts[si] = e;
            }
            out.rare_ptr.push(out.rare_lanes.len() as i64);
        }
    }
    if taps.tflag {
        out.tflag = (0..n_and)
            .map(|i| {
                if o0.and_kind[i] != 1 {
                    return 0i8;
                }
                let g = fl[i];
                ((if g & F_DP != 0 { 16 } else { 0 }) | (if g & F_DZ != 0 { 32 } else { 0 })) as i8
            })
            .collect();
    }
    if nw_ > 0 {
        for &(_, ev) in evs.iter() {
            if let Ev::WinEnd(j) = ev {
                let (a, b) = (taps.win_ptr[j] as usize, taps.win_ptr[j + 1] as usize);
                for slot in a..b {
                    let (mut nz, mut z0, mut z1) = (false, true, true);
                    for s in &outs {
                        let f = s.win_fl[slot];
                        nz |= f & W_NZ != 0;
                        z0 &= f & W_S0 != 0;
                        z1 &= f & W_S1 != 0;
                    }
                    if nz {
                        out.d_win.push(j as i64);
                        out.d_wire.push(taps.win_wires[slot]);
                        out.d_vec.extend_from_slice(&dbuf[slot * w..(slot + 1) * w]);
                        out.d_before0.push(z0);
                        out.d_after0.push(z1);
                    }
                }
            }
        }
    }
    if taps.prints {
        out.and_p = vec![0; n_and];
        out.and_pc = vec![0; n_and];
        out.and_o1 = vec![0; n_and];
        out.and_o2 = vec![0; n_and];
    }
    for r in results {
        match r {
            Res::Prints { base, out: v } => {
                for (i, x) in v.into_iter().enumerate() {
                    out.and_p[base + i] = x[0];
                    out.and_pc[base + i] = x[1];
                    out.and_o1[base + i] = x[2];
                    out.and_o2[base + i] = x[3];
                }
            }
            Res::Dep { k, dep, rank } => {
                out.dep[k * (nq + 1)..(k + 1) * (nq + 1)].copy_from_slice(&dep);
                out.dep_rank[k] = rank as i64;
            }
        }
    }
    out.end_wires = endw;
    out.end_cbits = endc;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dependent_small() {
        // rows: a, b, a^b, 0, c  (w = 1)
        let rows = [0b001u64, 0b010, 0b011, 0, 0b100];
        let (dep, rank) = dependent(&rows, 5, 1);
        assert_eq!(rank, 3);
        assert_eq!(dep, vec![true, true, true, true, false]);
    }
}

#[cfg(test)]
mod bench {
    //! cargo test --release --offline -p kmx-analysis --lib bench -- --ignored --nocapture (needs KMX_BENCH_KMB)
    use super::*;
    use crate::kmb::Chunked;
    #[test]
    #[ignore]
    fn decode_floor() {
        let Ok(path) = std::env::var("KMX_BENCH_KMB") else { return };
        let h = read_header(&path).unwrap();
        let t0 = std::time::Instant::now();
        const CAP: usize = 1 << 20;
        let mut ty = Chunked::open(&path, h.offs[0], h.num_ops, CAP).unwrap();
        let mut s = [h.offs[1], h.offs[2], h.offs[3], h.offs[4], h.offs[5], h.offs[6], h.off_b, h.off_c]
            .iter()
            .zip([h.n_qqq, h.n_qqq, h.n_qqq, h.n_qq, h.n_qq, h.n_q, h.n_b, h.n_c])
            .map(|(&o, n)| Chunked::open(&path, o, n * 4, CAP).unwrap())
            .collect::<Vec<_>>();
        let mut acc = 0u64;
        let mut hist = [0u64; 256];
        for _ in 0..h.num_ops {
            let t = ty.u8().unwrap();
            hist[t as usize] += 1;
            match t & 0x30 {
                0x30 => acc += (s[0].u32().unwrap() ^ s[1].u32().unwrap() ^ s[2].u32().unwrap()) as u64,
                0x20 => acc += (s[3].u32().unwrap() ^ s[4].u32().unwrap()) as u64,
                0x10 => acc += s[5].u32().unwrap() as u64,
                _ => {}
            }
            if t & 0x40 != 0 {
                acc += s[6].u32().unwrap() as u64;
            }
            if t & 0x80 != 0 {
                acc += s[7].u32().unwrap() as u64;
            }
        }
        println!("decode floor {:.3}s acc {}", t0.elapsed().as_secs_f64(), acc);
        for (t, &n) in hist.iter().enumerate() {
            if n > 0 {
                println!("op 0x{:02x}: {}", t, n);
            }
        }
        let nq = h.num_qubits as usize;
        let w = 64usize;
        let wires = vec![0u64; nq * w];
        let cb = vec![0u64; (h.num_bits as usize + 1) * w];
        let t1 = std::time::Instant::now();
        let taps = Taps { point_ptr: vec![0], win_ptr: vec![0], ..Default::default() };
        let r = simulate(&path, 4096, &wires, &cb, Outcomes::Zero, &taps, None, 1).unwrap();
        println!("simulate zero-state {:.3}s hmr {}", t1.elapsed().as_secs_f64(), r.n_hmr);
    }
}
