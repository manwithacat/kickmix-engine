//! Streaming liveness under the allocator model of `scripts/qhdl_lifetime_allocator.py` (the Python reference).
//!
//! Every qubit operand is a reference (op, wire). A wire's references split into intervals: an interval ends at a
//! reference after which the wire is |0> in every shot ("zero after", Python `reset_ref`) or at the wire's last touch;
//! the next touch after an end starts a new interval. Zero after is:
//!   * `cond_rule = false`: op type exactly R or HMR at condition depth 0 (Python `reset_ops` / `is_reset[so]`);
//!   * `cond_rule = true`: additionally the conditional-reset rule of Python `zero_after_refs`.
//! One record per wire is kept; nothing is sorted except the final interval list (by wire, then start).
use crate::kmb::{self, Header, HMR, HMR_IF, POP_CONDITION, PUSH_CONDITION, R, R_IF};

#[derive(Clone, Copy)]
struct Wire {
    touched: bool,
    last_op: i64,
    last_za: bool,
    last_block: i64,
    pre_exists: bool,
    pre_op: i64,
    pre_za: bool,
}

const FRESH: Wire = Wire {
    touched: false,
    last_op: -1,
    last_za: false,
    last_block: -2,
    pre_exists: false,
    pre_op: -1,
    pre_za: false,
};

/// One reference as seen by a scan, after the zero-after decision.
pub struct Touch {
    pub op: i64,
    pub wire: u32,
    /// stream slot (0..6: qqq0, qqq1, qqq2, qq0, qq1, q0)
    pub slot: usize,
    pub za: bool,
    /// previous touch of the same wire: (op, zero-after)
    pub prev: Option<(i64, bool)>,
}

pub struct ScanCounts {
    pub num_refs: u64,
    pub unconditional_resets: u64,
    pub conditional_resets: u64,
}

/// Stream the circuit once, deciding zero-after for every reference exactly as the Python reference.
pub fn scan<F: FnMut(&Touch)>(
    path: &str,
    cond_rule: bool,
    mut f: F,
) -> Result<(Header, ScanCounts), String> {
    let h0 = kmb::read_header(path)?;
    let regs = h0.register_qubits();
    let mut is_reg: Vec<bool> = Vec::new();
    for &q in &regs {
        let q = q as usize;
        if q >= is_reg.len() {
            is_reg.resize(q + 1, false);
        }
        is_reg[q] = true;
    }
    let mut wires: Vec<Wire> = vec![FRESH; (h0.num_qubits as usize).max(1)];
    let mut depth: i64 = 0; // condition depth in force at the op (Python condition_depth: before the op's own delta)
    let mut stack: Vec<i64> = Vec::new(); // open PUSH_CONDITION op indices
    let mut c = ScanCounts { num_refs: 0, unconditional_resets: 0, conditional_resets: 0 };
    let h = kmb::for_each_op(path, |o, t, qs, slot0| {
        let o = o as i64;
        let d = depth;
        // innermost enclosing block (Python block_of: stack top after the last PUSH/POP event strictly before o)
        let block = *stack.last().unwrap_or(&-1);
        let rr = t == R || t == HMR;
        if rr && d == 0 {
            c.unconditional_resets += 1;
        }
        let is_cond = (rr && d > 0) || t == R_IF || t == HMR_IF;
        if is_cond {
            c.conditional_resets += 1;
        }
        for (i, &q) in qs.iter().enumerate() {
            let qi = q as usize;
            if qi >= wires.len() {
                wires.resize(qi + 1, FRESH);
            }
            let w = wires[qi];
            let mut za = rr && d == 0;
            if cond_rule && !za && is_cond && !(d > 0 && !rr) {
                // (pre_exists, pre_za, admissible)
                let (exists, pza, ok_block) = if d > 0 {
                    let p = block;
                    if w.touched && w.last_block == p {
                        // touches since P form a suffix of block-P touches; the touch before it must precede P
                        (w.pre_exists, w.pre_za, !w.pre_exists || w.pre_op < p)
                    } else {
                        (w.touched, w.last_za, !w.touched || w.last_op < p)
                    }
                } else {
                    (w.touched, w.last_za, true) // an _IF outside any block: its own condition only
                };
                if ok_block {
                    za = if exists { pza } else { !(qi < is_reg.len() && is_reg[qi]) };
                }
            }
            c.num_refs += 1;
            f(&Touch {
                op: o,
                wire: q,
                slot: slot0 + i,
                za,
                prev: if w.touched { Some((w.last_op, w.last_za)) } else { None },
            });
            let ww = &mut wires[qi];
            if !w.touched {
                ww.touched = true;
                ww.pre_exists = false;
            } else if w.last_block != block {
                ww.pre_exists = true;
                ww.pre_op = w.last_op;
                ww.pre_za = w.last_za;
            }
            ww.last_op = o;
            ww.last_za = za;
            ww.last_block = block;
        }
        if t == PUSH_CONDITION {
            depth += 1;
            stack.push(o);
        } else if t == POP_CONDITION {
            depth -= 1;
            stack.pop();
        }
    })?;
    Ok((h, c))
}

#[derive(Default)]
pub struct Intervals {
    pub logical: Vec<i64>,
    pub s: Vec<i64>,
    pub e: Vec<i64>,
    pub is_first: Vec<bool>,
    pub is_last: Vec<bool>,
}

pub struct Liveness {
    pub header: Header,
    pub counts: ScanCounts,
    pub registers: Vec<u32>,
    pub iv: Intervals,
    /// number of references that are zero-after (Python reset_ref.sum())
    pub reset_refs: u64,
    /// per wire id: first / last touching op, -1 when never touched (length = max(num_qubits, max id + 1))
    pub wire_first: Vec<i64>,
    pub wire_last: Vec<i64>,
}

/// Intervals sorted by (wire, start), identical to Python build_intervals' (logical, s, e, is_first, is_last).
pub fn liveness(path: &str, cond_rule: bool) -> Result<Liveness, String> {
    struct Open {
        start: i64,
        open: bool,
        n: u32,
        last_iv: usize,
    }
    let mut opn: Vec<Open> = Vec::new();
    let mut wire_first: Vec<i64> = Vec::new();
    let mut wire_last: Vec<i64> = Vec::new();
    let mut raw: Vec<(u32, i64, i64, bool, bool)> = Vec::new();
    let mut reset_refs = 0u64;
    let (header, counts) = scan(path, cond_rule, |t| {
        let q = t.wire as usize;
        if q >= opn.len() {
            opn.resize_with(q + 1, || Open { start: -1, open: false, n: 0, last_iv: usize::MAX });
            wire_first.resize(q + 1, -1);
            wire_last.resize(q + 1, -1);
        }
        let w = &mut opn[q];
        match t.prev {
            None => {
                wire_first[q] = t.op;
                w.start = t.op;
            }
            Some((_, pza)) => {
                if pza {
                    w.start = t.op;
                }
            }
        }
        wire_last[q] = t.op;
        if t.za {
            reset_refs += 1;
            raw.push((t.wire, w.start, t.op, w.n == 0, false));
            w.n += 1;
            w.last_iv = raw.len() - 1;
            w.open = false;
        } else {
            w.open = true;
        }
    })?;
    let nq = (header.num_qubits as usize).max(opn.len());
    wire_first.resize(nq, -1);
    wire_last.resize(nq, -1);
    for (q, w) in opn.iter_mut().enumerate() {
        if wire_last[q] < 0 {
            continue;
        }
        if w.open {
            raw.push((q as u32, w.start, wire_last[q], w.n == 0, true));
            w.n += 1;
        } else {
            raw[w.last_iv].4 = true;
        }
    }
    raw.sort_unstable_by_key(|r| (r.0, r.1));
    let mut iv = Intervals::default();
    for r in &raw {
        iv.logical.push(r.0 as i64);
        iv.s.push(r.1);
        iv.e.push(r.2);
        iv.is_first.push(r.3);
        iv.is_last.push(r.4);
    }
    let registers = header.register_qubits();
    Ok(Liveness { header, counts, registers, iv, reset_refs, wire_first, wire_last })
}

/// Python live_profile: live wires just after each op; intervals occupy [s, e]. `registers` sorted unique.
pub fn live_profile(
    logical: &[i64],
    s: &[i64],
    e: &[i64],
    is_first: &[bool],
    is_last: &[bool],
    num_ops: usize,
    registers: &[i64],
    register_gaps: bool,
) -> Vec<i32> {
    let mut delta = vec![0i32; num_ops + 1];
    let base = if register_gaps { 0 } else { registers.len() as i32 };
    for i in 0..logical.len() {
        let reg = registers.binary_search(&logical[i]).is_ok();
        if reg && !register_gaps {
            continue;
        }
        let si = if reg && is_first[i] { 0 } else { s[i] };
        let ei = if reg && is_last[i] { num_ops as i64 - 1 } else { e[i] };
        delta[si as usize] += 1;
        delta[(ei + 1) as usize] -= 1;
    }
    // np.cumsum(delta)[:num_ops] + base
    let mut acc = 0i32;
    for x in delta.iter_mut() {
        acc += *x;
        *x = acc + base;
    }
    delta.truncate(num_ops);
    delta
}

/// Idle gaps of `qhdl_peak.analyse`: consecutive touches (a, b) of one wire, the wire live throughout (scratch: no
/// zero-after at a; register: also when zero-after at a unless `register_gaps`), with b - a > 1, that strictly
/// contain at least one whole run [run_s, run_e] (run_s > a and run_e < b). Returned sorted by (wire, a).
pub fn idle_gaps(
    path: &str,
    run_s: &[i64],
    run_e: &[i64],
    register_gaps: bool,
    cond_rule: bool,
) -> Result<(Vec<i64>, Vec<i64>, Vec<i64>), String> {
    let h = kmb::read_header(path)?;
    let regs = h.register_qubits();
    let mut out: Vec<(u32, i64, i64)> = Vec::new();
    scan(path, cond_rule, |t| {
        if let Some((a, za)) = t.prev {
            let b = t.op;
            let reg = regs.binary_search(&t.wire).is_ok();
            let cont = !za || (reg && !register_gaps);
            if cont && b - a > 1 {
                let i1 = run_s.partition_point(|&x| x <= a); // searchsorted(run_s, a, 'right')
                let i2 = run_e.partition_point(|&x| x < b); // searchsorted(run_e, b, 'left')
                if i2 > i1 {
                    out.push((t.wire, a, b));
                }
            }
        }
    })?;
    out.sort_unstable();
    let ga = out.iter().map(|x| x.1).collect();
    let gb = out.iter().map(|x| x.2).collect();
    let gw = out.iter().map(|x| x.0 as i64).collect();
    Ok((ga, gb, gw))
}

/// Allocator recolour: each reference gets col[interval index], intervals ordered as `logical` (sorted by wire, then
/// start). Returns the six qubit streams (qqq0, qqq1, qqq2, qq0, qq1, q0) in file order.
pub fn recolour(path: &str, logical: &[i64], col: &[i64], cond_rule: bool) -> Result<[Vec<u32>; 6], String> {
    let h = kmb::read_header(path)?;
    let lens = [h.n_qqq, h.n_qqq, h.n_qqq, h.n_qq, h.n_qq, h.n_q];
    let mut out: [Vec<u32>; 6] = Default::default();
    for (v, &n) in out.iter_mut().zip(lens.iter()) {
        v.reserve_exact(n as usize);
    }
    let mut cur: Vec<i64> = Vec::new(); // current interval index per wire
    let mut bad: Option<String> = None;
    scan(path, cond_rule, |t| {
        let q = t.wire as usize;
        if q >= cur.len() {
            cur.resize(q + 1, -1);
        }
        match t.prev {
            None => {
                let k = logical.partition_point(|&x| x < q as i64);
                if k >= logical.len() || logical[k] != q as i64 {
                    bad.get_or_insert_with(|| format!("wire {} has no interval", q));
                }
                cur[q] = k as i64;
            }
            Some((_, pza)) => {
                if pza {
                    cur[q] += 1;
                }
            }
        }
        let k = cur[q] as usize;
        let c = if k < col.len() { col[k] } else { -1 };
        if c < 0 || c > u32::MAX as i64 {
            bad.get_or_insert_with(|| format!("no colour for interval {}", k));
        }
        out[t.slot].push(c as u32);
    })?;
    if let Some(b) = bad {
        return Err(b);
    }
    Ok(out)
}
