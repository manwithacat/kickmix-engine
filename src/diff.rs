//! Structural diff of two .kmb circuits: op-level alignment in file order, with the changed regions (hunks), the
//! static per-kind and Toffoli deltas of each hunk, and a register-interface comparison. No simulation.
//!
//! Alignment: the common prefix is found while both files stream in lockstep (nothing stored); the remaining tails
//! are kept as 64-bit op hashes plus the op kind (9 bytes per op); the common suffix is taken from the tails; the
//! middle is aligned with content-defined anchors (k-gram hashes sampled where the hash has `sample_bits` low zero
//! bits, kept when unique on both sides, chained by a longest increasing subsequence) that are then extended op by
//! op in both directions. Everything between extended anchors is a hunk. For a circuit whose rewrite is local the
//! hunks are exact; for two unrelated circuits the result is one large hunk with any identical runs left matched.
//! Written by an AI model (Claude) under the owner's direction.

use crate::program::*;
use std::collections::HashMap;

pub struct DiffOpts {
    /// k-gram length of the anchors.
    pub k: usize,
    /// Keep a k-gram as an anchor candidate when its hash has this many low zero bits (density 2^-bits).
    pub sample_bits: u32,
    /// Report at most this many hunks in detail (the count is always exact).
    pub max_hunks: usize,
}

impl Default for DiffOpts {
    fn default() -> Self {
        DiffOpts { k: 16, sample_bits: 6, max_hunks: 1000 }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Hunk {
    pub a_start: u64,
    pub a_end: u64,
    pub b_start: u64,
    pub b_end: u64,
    pub a_kinds: [u64; NCOUNT],
    pub b_kinds: [u64; NCOUNT],
}

impl Hunk {
    pub fn toffoli_delta(&self) -> i64 {
        let t = |k: &[u64; NCOUNT]| (k[K_CCX as usize] + k[K_CCZ as usize] + k[K_CSWAP as usize]) as i64;
        t(&self.b_kinds) - t(&self.a_kinds)
    }
}

pub struct Side {
    pub num_ops: u64,
    pub num_qubits: usize,
    pub num_bits: usize,
    pub kinds: [u64; NCOUNT],
    pub registers: Vec<(String, Vec<(bool, u32)>)>,
    /// Highest qubit id touched by an op (+1), 0 if none.
    pub qubits_touched: usize,
}

pub struct DiffOut {
    pub a: Side,
    pub b: Side,
    pub identical: bool,
    pub common_prefix: u64,
    pub common_suffix: u64,
    pub matched_ops: u64,
    pub hunks: Vec<Hunk>,
    pub num_hunks: u64,
    pub anchors: u64,
}

fn side_from(ops: &KmbOps) -> Side {
    Side { num_ops: ops.num_ops, num_qubits: ops.num_qubits, num_bits: ops.num_bits, kinds: [0; NCOUNT], registers: ops.registers.clone(), qubits_touched: 0 }
}

#[inline]
fn tally(side: &mut Side, op: &SrcOp) {
    side.kinds[op.kind as usize] += 1;
    for &q in op.qubits() {
        side.qubits_touched = side.qubits_touched.max(q as usize + 1);
    }
}

/// Longest strictly increasing subsequence (by value) of `v`; returns the indices into `v`.
fn lis(v: &[u64]) -> Vec<usize> {
    let n = v.len();
    if n == 0 {
        return Vec::new();
    }
    let mut tails: Vec<usize> = Vec::new(); // index of the smallest tail of each length
    let mut prev = vec![usize::MAX; n];
    for i in 0..n {
        let x = v[i];
        let pos = tails.partition_point(|&t| v[t] < x);
        if pos > 0 {
            prev[i] = tails[pos - 1];
        }
        if pos == tails.len() {
            tails.push(i);
        } else {
            tails[pos] = i;
        }
    }
    let mut out = Vec::with_capacity(tails.len());
    let mut cur = *tails.last().unwrap();
    while cur != usize::MAX {
        out.push(cur);
        cur = prev[cur];
    }
    out.reverse();
    out
}

/// Anchor candidates of a hash sequence: (k-gram hash, start) for sampled k-grams.
fn candidates(h: &[u64], k: usize, bits: u32) -> Vec<(u64, u64)> {
    let mut out = Vec::new();
    if h.len() < k {
        return out;
    }
    let mask = (1u64 << bits) - 1;
    // Polynomial hash over a window of k op hashes, updated in O(1).
    const P: u64 = 0x100_0000_01b3;
    let mut pk = 1u64;
    for _ in 0..k {
        pk = pk.wrapping_mul(P);
    }
    let mut acc = 0u64;
    for i in 0..h.len() {
        acc = acc.wrapping_mul(P).wrapping_add(h[i] | 1);
        if i >= k {
            acc = acc.wrapping_sub(pk.wrapping_mul(h[i - k] | 1));
        }
        if i + 1 >= k {
            let g = (acc ^ (acc >> 29)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            if g & mask == 0 {
                out.push((g, (i + 1 - k) as u64));
            }
        }
    }
    out
}

/// Hashes unique on their own side; returns the map hash -> position.
fn unique(mut c: Vec<(u64, u64)>) -> HashMap<u64, u64> {
    c.sort_unstable();
    let mut m = HashMap::with_capacity(c.len());
    let mut i = 0;
    while i < c.len() {
        let mut j = i + 1;
        while j < c.len() && c[j].0 == c[i].0 {
            j += 1;
        }
        if j == i + 1 {
            m.insert(c[i].0, c[i].1);
        }
        i = j;
    }
    m
}

pub fn structural_diff(path_a: &str, path_b: &str, opts: &DiffOpts) -> Result<DiffOut, String> {
    let mut oa = KmbOps::open(path_a)?;
    let mut ob = KmbOps::open(path_b)?;
    let mut sa = side_from(&oa);
    let mut sb = side_from(&ob);

    // 1. Common prefix in lockstep.
    let mut prefix = 0u64;
    let (mut ta_h, mut ta_k): (Vec<u64>, Vec<u8>) = (Vec::new(), Vec::new());
    let (mut tb_h, mut tb_k): (Vec<u64>, Vec<u8>) = (Vec::new(), Vec::new());
    loop {
        let xa = oa.next()?;
        let xb = ob.next()?;
        match (xa, xb) {
            (Some(a), Some(b)) => {
                tally(&mut sa, &a);
                tally(&mut sb, &b);
                if a == b {
                    prefix += 1;
                } else {
                    ta_h.push(a.hash());
                    ta_k.push(a.kind);
                    tb_h.push(b.hash());
                    tb_k.push(b.kind);
                    break;
                }
            }
            (Some(a), None) => {
                tally(&mut sa, &a);
                ta_h.push(a.hash());
                ta_k.push(a.kind);
                break;
            }
            (None, Some(b)) => {
                tally(&mut sb, &b);
                tb_h.push(b.hash());
                tb_k.push(b.kind);
                break;
            }
            (None, None) => break,
        }
    }
    // 2. The tails.
    while let Some(a) = oa.next()? {
        tally(&mut sa, &a);
        ta_h.push(a.hash());
        ta_k.push(a.kind);
    }
    while let Some(b) = ob.next()? {
        tally(&mut sb, &b);
        tb_h.push(b.hash());
        tb_k.push(b.kind);
    }
    let (la, lb) = (ta_h.len(), tb_h.len());
    // 3. Common suffix (hash equality; a 64-bit hash collision would misalign one op, which the extension below
    //    would then report inside a hunk rather than hide).
    let mut suffix = 0usize;
    while suffix < la && suffix < lb && ta_h[la - 1 - suffix] == tb_h[lb - 1 - suffix] {
        suffix += 1;
    }
    let (ma, mb) = (la - suffix, lb - suffix);
    let mut out = DiffOut {
        a: sa,
        b: sb,
        identical: la == 0 && lb == 0,
        common_prefix: prefix,
        common_suffix: suffix as u64,
        matched_ops: prefix + suffix as u64,
        hunks: Vec::new(),
        num_hunks: 0,
        anchors: 0,
    };
    if ma == 0 && mb == 0 {
        return Ok(out);
    }
    // 4. Anchors in the middle.
    let ha = &ta_h[..ma];
    let hb = &tb_h[..mb];
    let ua = unique(candidates(ha, opts.k, opts.sample_bits));
    let ub = unique(candidates(hb, opts.k, opts.sample_bits));
    let mut pairs: Vec<(u64, u64)> = ua.iter().filter_map(|(h, &pa)| ub.get(h).map(|&pb| (pa, pb))).collect();
    pairs.sort_unstable();
    let keep = lis(&pairs.iter().map(|p| p.1).collect::<Vec<_>>());
    out.anchors = keep.len() as u64;
    // 5. Extend each anchor into a maximal matched run (not crossing the previous run).
    let mut runs: Vec<(usize, usize, usize)> = Vec::new(); // (a_start, b_start, len)
    let (mut pa_min, mut pb_min) = (0usize, 0usize);
    for &i in &keep {
        let (mut a0, mut b0) = (pairs[i].0 as usize, pairs[i].1 as usize);
        if a0 < pa_min || b0 < pb_min {
            continue;
        }
        let mut len = opts.k.min(ma - a0).min(mb - b0);
        // forward
        while a0 + len < ma && b0 + len < mb && ha[a0 + len] == hb[b0 + len] {
            len += 1;
        }
        // backward
        while a0 > pa_min && b0 > pb_min && ha[a0 - 1] == hb[b0 - 1] {
            a0 -= 1;
            b0 -= 1;
            len += 1;
        }
        // merge with the previous run when contiguous
        if let Some(last) = runs.last_mut() {
            if last.0 + last.2 == a0 && last.1 + last.2 == b0 {
                last.2 += len;
                pa_min = a0 + len;
                pb_min = b0 + len;
                continue;
            }
        }
        runs.push((a0, b0, len));
        pa_min = a0 + len;
        pb_min = b0 + len;
    }
    // 6. Hunks = gaps between runs (and before the first / after the last).
    let mut hunks = Vec::new();
    let (mut ca, mut cb) = (0usize, 0usize);
    let mut matched_middle = 0u64;
    let push_gap = |hunks: &mut Vec<Hunk>, a0: usize, a1: usize, b0: usize, b1: usize| {
        if a1 > a0 || b1 > b0 {
            let mut h = Hunk { a_start: prefix + a0 as u64, a_end: prefix + a1 as u64, b_start: prefix + b0 as u64, b_end: prefix + b1 as u64, ..Default::default() };
            for &k in &ta_k[a0..a1] {
                h.a_kinds[k as usize] += 1;
            }
            for &k in &tb_k[b0..b1] {
                h.b_kinds[k as usize] += 1;
            }
            hunks.push(h);
        }
    };
    for &(a0, b0, len) in &runs {
        push_gap(&mut hunks, ca, a0, cb, b0);
        matched_middle += len as u64;
        ca = a0 + len;
        cb = b0 + len;
    }
    push_gap(&mut hunks, ca, ma, cb, mb);
    out.matched_ops += matched_middle;
    out.num_hunks = hunks.len() as u64;
    hunks.truncate(opts.max_hunks);
    out.hunks = hunks;
    Ok(out)
}
