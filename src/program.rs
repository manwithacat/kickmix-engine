//! Compiled program: fixed-width op stream with compacted bit ids and CX run headers.
//! Semantics follow kickmix (Apache-2.0, Google LLC); see NOTICE.

use std::collections::HashMap;

pub const K_NEG: u8 = 0;
pub const K_POP: u8 = 1;
pub const K_ST0: u8 = 2;
pub const K_ST1: u8 = 3;
pub const K_INV: u8 = 4;
pub const K_Z: u8 = 5;
pub const K_X: u8 = 6;
pub const K_R: u8 = 7;
pub const K_ZPOW: u8 = 8;
pub const K_HMR: u8 = 9;
pub const K_CZ: u8 = 10;
pub const K_CX: u8 = 11;
pub const K_SWAP: u8 = 12;
pub const K_CCZ: u8 = 13;
pub const K_CCX: u8 = 14;
pub const K_CSWAP: u8 = 15;
pub const K_PUSH: u8 = 16;
/// Run header: `a` = number of following plain CX ops (unconditioned by a bit; the condition stack still applies).
pub const K_CXRUN: u8 = 17;
pub const NKINDS: usize = 18;
pub const KIND_NAMES: [&str; NKINDS] = [
    "NEG", "POP_CONDITION", "BIT_STORE0", "BIT_STORE1", "BIT_INVERT", "Z", "X", "R", "Z_POW", "HMR", "CZ", "CX",
    "SWAP", "CCZ", "CCX", "CSWAP", "PUSH_CONDITION", "CXRUN",
];
pub const F_IF: u8 = 1;

/// 10-byte fixed-width op. Operands: `a`,`b`,`c` qubit ids (HMR: a = qubit, b = compact bit target; BIT_*: b = target;
/// Z_POW: a = qubit, b = angle-table index); `cond` = compact bit id when `F_IF` is set.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Op {
    pub kind: u8,
    pub flags: u8,
    pub a: u16,
    pub b: u16,
    pub c: u16,
    pub cond: u16,
}

pub struct Program {
    pub ops: Vec<Op>,
    pub angles: Vec<u128>,
    pub levels: Vec<u8>,
    pub num_qubits: usize,
    pub num_bits: usize,
    pub bit_map: Vec<u32>,
    pub nrand: usize,
    pub n_source_ops: usize,
}

fn kind_of(name: &str) -> Option<(u8, usize)> {
    Some(match name {
        "NEG" => (K_NEG, 0),
        "POP_CONDITION" => (K_POP, 0),
        "BIT_STORE0" => (K_ST0, 0),
        "BIT_STORE1" => (K_ST1, 0),
        "BIT_INVERT" => (K_INV, 0),
        "Z" => (K_Z, 1),
        "X" => (K_X, 1),
        "R" => (K_R, 1),
        "Z_POW" => (K_ZPOW, 1),
        "HMR" => (K_HMR, 1),
        "CZ" => (K_CZ, 2),
        "CX" => (K_CX, 2),
        "SWAP" => (K_SWAP, 2),
        "CCZ" => (K_CCZ, 3),
        "CCX" => (K_CCX, 3),
        "CSWAP" => (K_CSWAP, 3),
        "PUSH_CONDITION" => (K_PUSH, 0),
        _ => return None,
    })
}

fn id(tok: &str, prefix: char) -> Result<u32, String> {
    let mut ch = tok.chars();
    if ch.next() != Some(prefix) {
        return Err(format!("expected {}<n>, got {:?}", prefix, tok));
    }
    tok[1..].parse::<u32>().map_err(|e| format!("bad id {:?}: {}", tok, e))
}

impl Program {
    /// Compile kickmix text (`str(circuit)`, .kmx). `angles` are the Z_POW angles in order of appearance as
    /// 128-bit integers V = half_turns * 2^127 mod 2^128 (exact; computed by the caller from the decimal text).
    pub fn from_text(text: &str, angles_in: &[u128], num_qubits: usize) -> Result<Program, String> {
        let mut raw: Vec<Op> = Vec::new();
        let mut bitmap: HashMap<u32, u16> = HashMap::new();
        let mut bit_map: Vec<u32> = Vec::new();
        let mut angles: Vec<u128> = Vec::new();
        let mut levels: Vec<u8> = Vec::new();
        let mut nrand = 0usize;
        let mut maxq: i64 = -1;
        let mut next_angle = 0usize;
        let cb = |orig: u32, bitmap: &mut HashMap<u32, u16>, bit_map: &mut Vec<u32>| -> Result<u16, String> {
            if let Some(&v) = bitmap.get(&orig) {
                return Ok(v);
            }
            if bit_map.len() >= 65535 {
                return Err("too many distinct classical bits for u16 ids".into());
            }
            let v = bit_map.len() as u16;
            bitmap.insert(orig, v);
            bit_map.push(orig);
            Ok(v)
        };
        for line in text.lines() {
            let mut toks: Vec<&str> = line.split_whitespace().collect();
            if toks.is_empty() || toks[0] == "APPEND_TO_REGISTER" || toks[0] == "REGISTER" {
                continue;
            }
            let (kind, nq) = kind_of(toks[0]).ok_or_else(|| format!("unsupported op {:?}", toks[0]))?;
            let mut op = Op { kind, flags: 0, a: 0, b: 0, c: 0, cond: 0 };
            if let Some(pos) = toks.iter().position(|&t| t == "if") {
                let orig = id(toks.get(pos + 1).ok_or("dangling if")?, 'b')?;
                op.cond = cb(orig, &mut bitmap, &mut bit_map)?;
                op.flags |= F_IF;
                toks.truncate(pos);
            }
            if kind == K_PUSH && op.flags & F_IF == 0 {
                return Err("PUSH_CONDITION without bit".into());
            }
            let args = &toks[1..];
            let mut qs = [0u32; 3];
            for i in 0..nq {
                qs[i] = id(args.get(i).ok_or("missing qubit operand")?, 'q')?;
                if qs[i] > 65535 {
                    return Err("qubit id exceeds u16".into());
                }
                maxq = maxq.max(qs[i] as i64);
            }
            op.a = qs[0] as u16;
            op.b = qs[1] as u16;
            op.c = qs[2] as u16;
            let rest = &args[nq..];
            match kind {
                K_ST0 | K_ST1 | K_INV => {
                    op.b = cb(id(rest.first().ok_or("missing bit")?, 'b')?, &mut bitmap, &mut bit_map)?;
                }
                K_HMR => {
                    op.b = cb(id(rest.first().ok_or("missing bit")?, 'b')?, &mut bitmap, &mut bit_map)?;
                    nrand += 1;
                }
                K_R => nrand += 1,
                K_ZPOW => {
                    let v = *angles_in.get(next_angle).ok_or("fewer angles than Z_POW ops")?;
                    next_angle += 1;
                    if angles.len() >= 65535 {
                        return Err("angle table exceeds u16".into());
                    }
                    let lo = v as u64;
                    let hi = (v >> 64) as u64;
                    let level = if lo != 0 { 128 - lo.trailing_zeros() as u8 } else if hi != 0 { 64 - hi.trailing_zeros() as u8 } else { 0 };
                    // kickmix words[0] is the low word of V, words[1] the high word.
                    angles.push(v);
                    levels.push(level);
                    op.b = (angles.len() - 1) as u16;
                }
                _ => {}
            }
            raw.push(op);
        }
        if next_angle != angles_in.len() {
            return Err("more angles than Z_POW ops".into());
        }
        let n_source_ops = raw.len();
        // CX run headers: maximal runs (>= 2) of CX without a bit condition.
        let mut ops: Vec<Op> = Vec::with_capacity(raw.len() + raw.len() / 8);
        let mut i = 0;
        while i < raw.len() {
            if raw[i].kind == K_CX && raw[i].flags & F_IF == 0 {
                let mut j = i;
                while j < raw.len() && raw[j].kind == K_CX && raw[j].flags & F_IF == 0 && j - i < 65535 {
                    j += 1;
                }
                if j - i >= 2 {
                    ops.push(Op { kind: K_CXRUN, flags: 0, a: (j - i) as u16, b: 0, c: 0, cond: 0 });
                    ops.extend_from_slice(&raw[i..j]);
                    i = j;
                    continue;
                }
            }
            ops.push(raw[i]);
            i += 1;
        }
        let num_qubits = num_qubits.max((maxq + 1) as usize);
        Ok(Program { ops, angles, levels, num_qubits, num_bits: bit_map.len(), bit_map, nrand, n_source_ops })
    }
}
