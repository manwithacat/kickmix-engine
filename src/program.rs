//! Compiled program: a u32 word stream with compacted classical-bit ids and CX run headers.
//! Semantics follow kickmix (Apache-2.0, Google LLC); see NOTICE.
//!
//! M3 layout (replaces M1's 10-byte fixed ops, whose u16 bit ids could not hold DQI's ~103k bits):
//! - word 0 of every op: bits 0..8 opcode (base kind | `IF`), bits 16..32 qubit `a` (qubit ids are < 65536).
//!   `CXRUN` header: bits 8..32 = number of following run entries (one word each: `a | b << 16`).
//! - then, in this order, only the words the opcode needs: `b | c << 16` (2- and 3-qubit ops), the compact
//!   bit target (BIT_*, HMR), the angle-table index (Z_POW), the compact condition bit (`*_IF`, PUSH_CONDITION).
//! INFO_TAG is not in the stream: kickmix counts it with an all-ones mask regardless of conditions and it has
//! no other effect, so its count is `n_info_tags` per shot.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

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
/// Count slot only (never in the stream).
pub const K_INFO: u8 = 17;
/// Stream only: run header of plain (bit-unconditioned) CX ops; counted as CX.
pub const K_CXRUN: u8 = 18;
/// Number of count slots (kinds 0..=17).
pub const NCOUNT: usize = 18;
pub const KIND_NAMES: [&str; NCOUNT] = [
    "NEG", "POP_CONDITION", "BIT_STORE0", "BIT_STORE1", "BIT_INVERT", "Z", "X", "R", "Z_POW", "HMR", "CZ", "CX",
    "SWAP", "CCZ", "CCX", "CSWAP", "PUSH_CONDITION", "INFO_TAG",
];
/// Opcode flag: conditioned on a classical bit (the last word of the op).
pub const IF: u8 = 32;
pub const MAX_RUN: usize = (1 << 24) - 1;

/// Number of qubit operands of a base kind.
pub fn nq_of(k: u8) -> usize {
    match k {
        K_Z | K_X | K_R | K_ZPOW | K_HMR => 1,
        K_CZ | K_CX | K_SWAP => 2,
        K_CCZ | K_CCX | K_CSWAP => 3,
        _ => 0,
    }
}
fn has_bit_target(k: u8) -> bool {
    matches!(k, K_ST0 | K_ST1 | K_INV | K_HMR)
}

/// Word length of every opcode (0 = invalid).
pub const fn op_len_table() -> [u8; 256] {
    let mut t = [0u8; 256];
    let mut k = 0u8;
    while k <= K_PUSH {
        let nq = match k {
            K_Z | K_X | K_R | K_ZPOW | K_HMR => 1,
            K_CZ | K_CX | K_SWAP => 2,
            K_CCZ | K_CCX | K_CSWAP => 3,
            _ => 0,
        };
        let mut len = 1u8;
        if nq >= 2 {
            len += 1;
        }
        if matches!(k, K_ST0 | K_ST1 | K_INV | K_HMR) {
            len += 1;
        }
        if k == K_ZPOW {
            len += 1;
        }
        if k == K_PUSH {
            len += 1;
            t[k as usize] = len;
        } else if k == K_POP {
            t[k as usize] = len;
        } else {
            t[k as usize] = len;
            t[(k | IF) as usize] = len + 1;
        }
        k += 1;
    }
    t
}
pub static OP_LEN: [u8; 256] = op_len_table();

pub struct Program {
    pub code: Vec<u32>,
    pub angles: Vec<u128>,
    pub levels: Vec<u8>,
    pub num_qubits: usize,
    /// Number of compact classical bits (bits used by at least one op).
    pub num_bits: usize,
    /// compact id -> original bit id.
    pub bit_map: Vec<u32>,
    /// original bit id -> compact id (u32::MAX if no op touches it); length = original num_bits.
    pub bit_compact: Vec<u32>,
    pub orig_num_bits: usize,
    pub nrand: usize,
    /// Source ops including INFO_TAG (= len(circuit)).
    pub n_source_ops: usize,
    pub n_info_tags: u64,
    pub max_stack_depth: usize,
    pub n_cx_runs: usize,
    /// kickmix `Circuit.reaction_depth()`, computed once while compiling (see `ReactionDepth`).
    pub reaction_depth: u64,
    /// Static per-kind op counts (as `circuit.operation_statistics()`), count slots.
    pub static_counts: [u64; NCOUNT],
    /// Registers from the .kmb file: (name, [(is_qubit, id)]).
    pub registers: Vec<(String, Vec<(bool, u32)>)>,
}

pub struct Builder {
    code: Vec<u32>,
    bit_compact: Vec<u32>,
    bit_map: Vec<u32>,
    angles: Vec<u128>,
    levels: Vec<u8>,
    nrand: usize,
    n_source_ops: usize,
    n_info: u64,
    maxq: i64,
    maxb: i64,
    run_hdr: Option<usize>,
    run_len: usize,
    n_runs: usize,
    depth: usize,
    max_depth: usize,
    static_counts: [u64; NCOUNT],
    rd: ReactionDepth,
}

/// kickmix's reaction depth (`Circuit::reaction_depth`, `compute_reaction_depth` in kickmix
/// `circuit/circuit_util.cc`, Apache-2.0, Google LLC), restated as a streaming tracker over source ops in file order.
/// Per qubit: the reaction depth of knowing whether a bit flip (`x`) or a phase flip (`z`) is present; per classical
/// bit (`c`, original ids): of knowing its value. Non-Clifford ops (CCX, CCZ, CSWAP, non-Clifford Z_POW) add one
/// layer; HMR makes a bit depend on the qubit's `z` depth plus one; ops under the condition stack or a bit condition
/// inherit the condition's depth. The result is the maximum over all entries. Every rule below mirrors the kickmix
/// switch case for case, including the order of the bumps (later bumps read values updated by earlier ones).
/// INFO_TAG, NEG and DEBUG_PRINT have no effect, so INFO_TAG (not in the engine stream) is skipped.
pub struct ReactionDepth {
    x: Vec<u64>,
    z: Vec<u64>,
    c: Vec<u64>,
    stack: Vec<u64>,
}

impl ReactionDepth {
    pub fn new() -> Self {
        // Qubit ids are < 65536 (checked by `Builder::op`), so the qubit tables never grow.
        ReactionDepth { x: vec![0; 1 << 16], z: vec![0; 1 << 16], c: Vec::new(), stack: vec![0] }
    }

    /// Fast path for the commonest op, a CX with no bit condition (same rule as the CX case of `apply`).
    #[inline(always)]
    pub fn cx_plain(&mut self, c1: u32, t: u32) {
        let (c1, t) = (c1 as usize & 0xFFFF, t as usize & 0xFFFF);
        let cd = *self.stack.last().unwrap();
        let xt = self.x[t].max(cd).max(self.x[c1]);
        self.x[t] = xt;
        let zc = self.z[c1].max(cd).max(self.z[t]);
        self.z[c1] = zc;
    }
    #[inline]
    fn grow_c(&mut self, b: usize) {
        if b >= self.c.len() {
            let n = (b + 1).max(self.c.len() * 2).max(64);
            self.c.resize(n, 0);
        }
    }
    #[inline]
    fn xb(&mut self, q: usize, d: u64) {
        if d > self.x[q] {
            self.x[q] = d;
        }
    }
    #[inline]
    fn zb(&mut self, q: usize, d: u64) {
        if d > self.z[q] {
            self.z[q] = d;
        }
    }
    #[inline]
    fn cbump(&mut self, b: usize, d: u64) {
        if d > self.c[b] {
            self.c[b] = d;
        }
    }

    /// One source op (base kind, condition bit, qubits, bit target, Z_POW angle), original bit ids.
    pub fn apply(&mut self, kind: u8, cond: Option<u32>, qs: &[u32], bit: Option<u32>, angle: Option<u128>) {
        if let Some(b) = bit {
            self.grow_c(b as usize);
        }
        if let Some(b) = cond {
            self.grow_c(b as usize);
        }
        let mut cd = *self.stack.last().unwrap();
        if let Some(b) = cond {
            cd = cd.max(self.c[b as usize]);
        }
        // kickmix: `condition_depth_stack.size() == 1 && !op.c_condition.is_bit()`.
        let top = self.stack.len() == 1 && cond.is_none();
        // Operand order as kickmix `Circuit::iter_ops` assigns it: 2-qubit ops are (control1, target), 3-qubit ops
        // (control2, control1, target), i.e. the target is the last operand.
        let q = |i: usize| qs[i] as usize;
        match kind {
            K_NEG => {}
            K_X => self.xb(q(0), cd),
            K_Z => self.zb(q(0), cd),
            K_INV => self.cbump(bit.unwrap() as usize, cd),
            K_ST0 | K_ST1 => {
                let b = bit.unwrap() as usize;
                if top {
                    self.c[b] = 0;
                } else {
                    self.cbump(b, cd);
                }
            }
            K_CX => {
                let (c1, t) = (q(0), q(1));
                self.xb(t, cd);
                self.xb(t, self.x[c1]);
                self.zb(c1, cd);
                self.zb(c1, self.z[t]);
            }
            K_CZ => {
                let (c1, t) = (q(0), q(1));
                self.zb(t, cd);
                self.zb(t, self.x[c1]);
                self.zb(c1, cd);
                self.zb(c1, self.x[t]);
            }
            K_SWAP => {
                let (c1, t) = (q(0), q(1));
                if top {
                    self.x.swap(t, c1);
                    self.z.swap(t, c1);
                } else {
                    self.xb(t, self.x[c1]);
                    self.zb(t, self.z[c1]);
                    self.xb(c1, self.x[t]);
                    self.zb(c1, self.z[t]);
                    self.zb(t, cd);
                    self.xb(t, cd);
                    self.zb(c1, cd);
                    self.xb(c1, cd);
                }
            }
            K_R => {
                let t = q(0);
                if top {
                    self.x[t] = 0;
                    self.z[t] = 0;
                } else {
                    self.xb(t, cd);
                }
            }
            K_HMR => {
                let (t, b) = (q(0), bit.unwrap() as usize);
                if top {
                    self.c[b] = self.z[t] + 1;
                    self.x[t] = 0;
                    self.z[t] = 0;
                } else {
                    self.cbump(b, cd);
                    self.cbump(b, self.z[t] + 1);
                    self.xb(t, cd);
                    self.zb(t, cd);
                }
            }
            K_CCX => {
                let (c2, c1, t) = (q(0), q(1), q(2));
                self.xb(t, cd + 1);
                self.xb(t, self.x[c1] + 1);
                self.xb(t, self.x[c2] + 1);
                self.zb(c1, cd + 1);
                self.zb(c1, self.z[t] + 1);
                self.zb(c1, self.x[c2] + 1);
                self.zb(c2, cd + 1);
                self.zb(c2, self.z[t] + 1);
                self.zb(c2, self.x[c1] + 1);
            }
            K_CSWAP => {
                let (c2, c1, t) = (q(0), q(1), q(2));
                let dx = self.x[t].max(self.x[c1]);
                let dz = self.z[t].max(self.z[c1]);
                let dxc = self.x[c2];
                for &u in &[t, c1] {
                    self.xb(u, cd + 1);
                    self.xb(u, dx + 1);
                    self.xb(u, dxc + 1);
                }
                for &u in &[t, c1] {
                    self.zb(u, cd + 1);
                    self.zb(u, dz + 1);
                    self.zb(u, dxc + 1);
                }
                self.zb(c2, cd + 1);
                self.zb(c2, dx + 1);
                self.zb(c2, dz + 1);
            }
            K_ZPOW => {
                let t = q(0);
                // FixedPrecisionAngle128::is_multiple_of_180_degrees: only the top bit of V may be set.
                if angle.unwrap() & !(1u128 << 127) == 0 {
                    self.zb(t, cd);
                } else {
                    self.zb(t, cd + 1);
                    self.zb(t, self.x[t] + 1);
                }
            }
            K_CCZ => {
                let (c2, c1, t) = (q(0), q(1), q(2));
                self.zb(t, cd + 1);
                self.zb(t, self.x[c1] + 1);
                self.zb(t, self.x[c2] + 1);
                self.zb(c1, cd + 1);
                self.zb(c1, self.x[t] + 1);
                self.zb(c1, self.x[c2] + 1);
                self.zb(c2, cd + 1);
                self.zb(c2, self.x[t] + 1);
                self.zb(c2, self.x[c1] + 1);
            }
            K_PUSH => self.stack.push(cd),
            K_POP => {
                if self.stack.len() > 1 {
                    self.stack.pop();
                }
            }
            _ => unreachable!("reaction depth: unhandled kind {}", kind),
        }
    }

    pub fn max_depth(&self) -> u64 {
        self.x.iter().chain(self.z.iter()).chain(self.c.iter()).copied().max().unwrap_or(0)
    }
}

impl Default for ReactionDepth {
    fn default() -> Self {
        Self::new()
    }
}

pub fn angle_level(v: u128) -> u8 {
    let lo = v as u64;
    let hi = (v >> 64) as u64;
    if lo != 0 {
        128 - lo.trailing_zeros() as u8
    } else if hi != 0 {
        64 - hi.trailing_zeros() as u8
    } else {
        0
    }
}

impl Builder {
    pub fn new(cap: usize) -> Self {
        Builder {
            code: Vec::with_capacity(cap),
            bit_compact: Vec::new(),
            bit_map: Vec::new(),
            angles: Vec::new(),
            levels: Vec::new(),
            nrand: 0,
            n_source_ops: 0,
            n_info: 0,
            maxq: -1,
            maxb: -1,
            run_hdr: None,
            run_len: 0,
            n_runs: 0,
            depth: 0,
            max_depth: 0,
            static_counts: [0; NCOUNT],
            rd: ReactionDepth::new(),
        }
    }

    fn cb(&mut self, orig: u32) -> u32 {
        let o = orig as usize;
        if o >= self.bit_compact.len() {
            self.bit_compact.resize((o + 1).max(self.bit_compact.len() * 2), u32::MAX);
        }
        self.maxb = self.maxb.max(orig as i64);
        let v = self.bit_compact[o];
        if v != u32::MAX {
            return v;
        }
        let v = self.bit_map.len() as u32;
        self.bit_compact[o] = v;
        self.bit_map.push(orig);
        v
    }

    fn close_run(&mut self) {
        if let Some(h) = self.run_hdr.take() {
            self.code[h] = K_CXRUN as u32 | ((self.run_len as u32) << 8);
            self.run_len = 0;
        }
    }

    pub fn info_tag(&mut self) {
        self.n_source_ops += 1;
        self.n_info += 1;
        self.static_counts[K_INFO as usize] += 1;
    }

    /// One source op. `kind` is a base kind (no IF); `cond` is the condition bit (`*_IF`) or PUSH_CONDITION's bit.
    pub fn op(&mut self, kind: u8, cond: Option<u32>, qs: &[u32], bit: Option<u32>, angle: Option<u128>) -> Result<(), String> {
        if kind > K_PUSH {
            return Err(format!("bad kind {}", kind));
        }
        let nq = nq_of(kind);
        if qs.len() != nq {
            return Err(format!("kind {} needs {} qubits, got {}", KIND_NAMES[kind as usize], nq, qs.len()));
        }
        for &q in qs {
            if q > 0xFFFF {
                return Err("qubit id exceeds 65535".into());
            }
            self.maxq = self.maxq.max(q as i64);
        }
        if kind == K_PUSH && cond.is_none() {
            return Err("PUSH_CONDITION without bit".into());
        }
        if kind == K_POP && cond.is_some() {
            return Err("POP_CONDITION with bit".into());
        }
        if has_bit_target(kind) != bit.is_some() {
            return Err(format!("{}: bit target mismatch", KIND_NAMES[kind as usize]));
        }
        if (kind == K_ZPOW) != angle.is_some() {
            return Err("angle mismatch".into());
        }
        self.n_source_ops += 1;
        self.static_counts[kind as usize] += 1;
        if kind == K_CX && cond.is_none() {
            self.rd.cx_plain(qs[0], qs[1]);
        } else {
            self.rd.apply(kind, cond, qs, bit, angle);
        }
        match kind {
            K_PUSH => {
                self.depth += 1;
                self.max_depth = self.max_depth.max(self.depth);
            }
            K_POP => self.depth = self.depth.saturating_sub(1),
            K_HMR | K_R => self.nrand += 1,
            _ => {}
        }
        if kind == K_CX && cond.is_none() {
            if self.run_hdr.is_none() || self.run_len >= MAX_RUN {
                self.close_run();
                self.run_hdr = Some(self.code.len());
                self.code.push(0);
                self.n_runs += 1;
            }
            self.code.push(qs[0] | (qs[1] << 16));
            self.run_len += 1;
            return Ok(());
        }
        self.close_run();
        let opc = if cond.is_some() && kind != K_PUSH { kind | IF } else { kind };
        let a = if nq >= 1 { qs[0] } else { 0 };
        self.code.push(opc as u32 | (a << 16));
        if nq == 2 {
            self.code.push(qs[1]);
        } else if nq == 3 {
            self.code.push(qs[1] | (qs[2] << 16));
        }
        if let Some(b) = bit {
            let c = self.cb(b);
            self.code.push(c);
        }
        if let Some(v) = angle {
            if self.angles.len() >= u32::MAX as usize {
                return Err("angle table too large".into());
            }
            self.angles.push(v);
            self.levels.push(angle_level(v));
            self.code.push((self.angles.len() - 1) as u32);
        }
        if let Some(c) = cond {
            let c = self.cb(c);
            self.code.push(c);
        }
        Ok(())
    }

    pub fn finish(mut self, num_qubits: usize, num_bits: usize, registers: Vec<(String, Vec<(bool, u32)>)>) -> Program {
        self.close_run();
        self.code.shrink_to_fit();
        let num_qubits = num_qubits.max((self.maxq + 1) as usize);
        let orig_num_bits = num_bits.max((self.maxb + 1) as usize);
        self.bit_compact.resize(orig_num_bits, u32::MAX);
        self.bit_compact.shrink_to_fit();
        Program {
            code: self.code,
            angles: self.angles,
            levels: self.levels,
            num_qubits,
            num_bits: self.bit_map.len(),
            bit_map: self.bit_map,
            bit_compact: self.bit_compact,
            orig_num_bits,
            nrand: self.nrand,
            n_source_ops: self.n_source_ops,
            n_info_tags: self.n_info,
            max_stack_depth: self.max_depth,
            n_cx_runs: self.n_runs,
            reaction_depth: self.rd.max_depth(),
            static_counts: self.static_counts,
            registers,
        }
    }
}

fn kind_of(name: &str) -> Option<u8> {
    Some(match name {
        "NEG" => K_NEG,
        "POP_CONDITION" => K_POP,
        "BIT_STORE0" => K_ST0,
        "BIT_STORE1" => K_ST1,
        "BIT_INVERT" => K_INV,
        "Z" => K_Z,
        "X" => K_X,
        "R" => K_R,
        "Z_POW" => K_ZPOW,
        "HMR" => K_HMR,
        "CZ" => K_CZ,
        "CX" => K_CX,
        "SWAP" => K_SWAP,
        "CCZ" => K_CCZ,
        "CCX" => K_CCX,
        "CSWAP" => K_CSWAP,
        "PUSH_CONDITION" => K_PUSH,
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

// .kmb reader (format of kickmix `Circuit::write_kmb_to`, version 2: little-endian integers, struct-of-arrays).
const KMB_MAGIC: [u8; 16] = [0xd7, 0x50, 0xc7, 0xd5, 0xc3, 0x29, 0xd3, 0x26, 0xe3, 0xcc, 0x9f, 0x68, 0x34, 0xf2, 0xb8, 0xbf];

fn rd_u32<R: Read>(r: &mut R) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}
fn rd_u64<R: Read>(r: &mut R) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}

/// Sequential reader of one u32 array packet.
struct ArrReader {
    r: BufReader<File>,
    left: u64,
}
impl ArrReader {
    fn open(path: &str, payload_off: u64, count: u64) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(payload_off)).map_err(|e| e.to_string())?;
        Ok(ArrReader { r: BufReader::with_capacity(1 << 20, f), left: count })
    }
    #[inline]
    fn next(&mut self) -> Result<u32, String> {
        if self.left == 0 {
            return Err("operand array exhausted".into());
        }
        self.left -= 1;
        rd_u32(&mut self.r)
    }
}

impl Program {
    /// Compile kickmix text (`str(circuit)`, .kmx). `angles` are the Z_POW angles in order of appearance as
    /// 128-bit integers V = half_turns * 2^127 mod 2^128 (exact; computed by the caller from the decimal text).
    pub fn from_text(text: &str, angles_in: &[u128], num_qubits: usize) -> Result<Program, String> {
        let mut bld = Builder::new(text.len() / 8);
        let mut next_angle = 0usize;
        for line in text.lines() {
            let mut toks: Vec<&str> = line.split_whitespace().collect();
            if toks.is_empty() || toks[0] == "APPEND_TO_REGISTER" || toks[0] == "REGISTER" {
                continue;
            }
            if toks[0] == "INFO_TAG" {
                bld.info_tag();
                continue;
            }
            let kind = kind_of(toks[0]).ok_or_else(|| format!("unsupported op {:?}", toks[0]))?;
            let mut cond = None;
            if let Some(pos) = toks.iter().position(|&t| t == "if") {
                cond = Some(id(toks.get(pos + 1).ok_or("dangling if")?, 'b')?);
                toks.truncate(pos);
            }
            let nq = nq_of(kind);
            let args = &toks[1..];
            let mut qs = [0u32; 3];
            for i in 0..nq {
                qs[i] = id(args.get(i).ok_or("missing qubit operand")?, 'q')?;
            }
            let rest = &args[nq..];
            let bit = if has_bit_target(kind) { Some(id(rest.first().ok_or("missing bit")?, 'b')?) } else { None };
            let angle = if kind == K_ZPOW {
                let v = *angles_in.get(next_angle).ok_or("fewer angles than Z_POW ops")?;
                next_angle += 1;
                Some(v)
            } else {
                None
            };
            bld.op(kind, cond, &qs[..nq], bit, angle)?;
        }
        if next_angle != angles_in.len() {
            return Err("more angles than Z_POW ops".into());
        }
        Ok(bld.finish(num_qubits, 0, Vec::new()))
    }

    /// Compile a .kmb file by streaming its arrays (no full copy of the circuit is held).
    pub fn from_kmb(path: &str) -> Result<Program, String> {
        let f = File::open(path).map_err(|e| e.to_string())?;
        let file_len = f.metadata().map_err(|e| e.to_string())?.len();
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut magic = [0u8; 16];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if magic != KMB_MAGIC {
            return Err("not a kmb file (magic bytes)".into());
        }
        if rd_u32(&mut r)? != 2 {
            return Err("unsupported kmb version".into());
        }
        let num_qubits = rd_u32(&mut r)? as usize;
        let num_bits = rd_u32(&mut r)? as usize;
        let num_regs = rd_u32(&mut r)? as usize;
        let mut h = [0u64; 9];
        for x in h.iter_mut() {
            *x = rd_u64(&mut r)?;
        }
        let [num_ops, n_qqq, n_qq, n_q, n_b, n_c, n_ang, n_info, n_info_chars] = h;
        let mut pos: u64 = 16 + 4 * 4 + 9 * 8;
        let mut registers = Vec::with_capacity(num_regs);
        for _ in 0..num_regs {
            if rd_u64(&mut r)? != 0 {
                return Err("expected register packet".into());
            }
            let size = rd_u64(&mut r)?;
            let name_len = rd_u32(&mut r)? as u64;
            let reg_len = rd_u32(&mut r)? as u64;
            if reg_len * 4 + name_len + 8 != size {
                return Err("register packet size".into());
            }
            let mut name = vec![0u8; name_len as usize];
            r.read_exact(&mut name).map_err(|e| e.to_string())?;
            let mut contents = Vec::with_capacity(reg_len as usize);
            for _ in 0..reg_len {
                let v = rd_u32(&mut r)?;
                contents.push((v & 0x8000_0000 != 0, v & 0x7FFF_FFFF));
            }
            registers.push((String::from_utf8(name).map_err(|e| e.to_string())?, contents));
            pos += 16 + size;
        }
        // Packet layout after the registers: types, qqq0, qqq1, qqq2, qq0, qq1, q0, b0, bc, angles, info tags.
        let sizes: [(u64, u64); 10] =
            [(1, num_ops), (2, n_qqq * 4), (3, n_qqq * 4), (4, n_qqq * 4), (5, n_qq * 4), (6, n_qq * 4), (7, n_q * 4), (8, n_b * 4), (9, n_c * 4), (10, n_ang * 16)];
        let mut offs = [0u64; 10];
        let mut p = pos;
        drop(r);
        let mut hf = File::open(path).map_err(|e| e.to_string())?;
        for (i, &(ty, sz)) in sizes.iter().enumerate() {
            hf.seek(SeekFrom::Start(p)).map_err(|e| e.to_string())?;
            if rd_u64(&mut hf)? != ty || rd_u64(&mut hf)? != sz {
                return Err(format!("kmb packet {} header mismatch", ty));
            }
            offs[i] = p + 16;
            p += 16 + sz;
        }
        hf.seek(SeekFrom::Start(p)).map_err(|e| e.to_string())?;
        if rd_u64(&mut hf)? != 11 || rd_u64(&mut hf)? != (n_info + 1) * 8 + n_info_chars || rd_u64(&mut hf)? != n_info {
            return Err("kmb info tag packet mismatch".into());
        }
        if p + 16 + (n_info + 1) * 8 + n_info_chars != file_len {
            return Err("kmb data leftover or truncated".into());
        }
        let mut types = BufReader::with_capacity(1 << 20, File::open(path).map_err(|e| e.to_string())?);
        types.seek(SeekFrom::Start(offs[0])).map_err(|e| e.to_string())?;
        let mut qqq = [ArrReader::open(path, offs[1], n_qqq)?, ArrReader::open(path, offs[2], n_qqq)?, ArrReader::open(path, offs[3], n_qqq)?];
        let mut qq = [ArrReader::open(path, offs[4], n_qq)?, ArrReader::open(path, offs[5], n_qq)?];
        let mut q0 = ArrReader::open(path, offs[6], n_q)?;
        let mut b0 = ArrReader::open(path, offs[7], n_b)?;
        let mut bc = ArrReader::open(path, offs[8], n_c)?;
        let mut ang = BufReader::with_capacity(1 << 16, File::open(path).map_err(|e| e.to_string())?);
        ang.seek(SeekFrom::Start(offs[9])).map_err(|e| e.to_string())?;
        let mut n_ang_used = 0u64;

        let mut bld = Builder::new((num_ops as usize) + (num_ops as usize) / 4);
        let mut tbuf = vec![0u8; 1 << 20];
        let mut done = 0u64;
        while done < num_ops {
            let n = ((num_ops - done) as usize).min(tbuf.len());
            types.read_exact(&mut tbuf[..n]).map_err(|e| e.to_string())?;
            for &t in &tbuf[..n] {
                if t == 0x06 {
                    bld.info_tag();
                    continue;
                }
                let low = t & 0x0F;
                if low == 7 {
                    return Err("DEBUG_PRINT ops are not supported".into());
                }
                let has_cond = t & 0x80 != 0;
                let nq = ((t >> 4) & 3) as usize;
                let has_b = t & 0x40 != 0;
                let kind = match (t & 0x7F, has_cond) {
                    (0x00, false) => K_NEG,
                    (0x00, true) => K_NEG,
                    (0x01, false) => K_POP,
                    (0x01, true) => K_PUSH,
                    (0x40, _) => K_ST0,
                    (0x41, _) => K_ST1,
                    (0x42, _) => K_INV,
                    (0x10, _) => K_Z,
                    (0x11, _) => K_X,
                    (0x12, _) => K_R,
                    (0x13, _) => K_ZPOW,
                    (0x50, _) => K_HMR,
                    (0x20, _) => K_CZ,
                    (0x21, _) => K_CX,
                    (0x22, _) => K_SWAP,
                    (0x30, _) => K_CCZ,
                    (0x31, _) => K_CCX,
                    (0x32, _) => K_CSWAP,
                    _ => return Err(format!("unsupported kmb op type 0x{:02x}", t)),
                };
                let mut qs = [0u32; 3];
                match nq {
                    1 => qs[0] = q0.next()?,
                    2 => {
                        qs[0] = qq[0].next()?;
                        qs[1] = qq[1].next()?;
                    }
                    3 => {
                        qs[0] = qqq[0].next()?;
                        qs[1] = qqq[1].next()?;
                        qs[2] = qqq[2].next()?;
                    }
                    _ => {}
                }
                let bit = if has_b { Some(b0.next()?) } else { None };
                let cond = if has_cond { Some(bc.next()?) } else { None };
                let angle = if kind == K_ZPOW {
                    let lo = rd_u64(&mut ang)?;
                    let hi = rd_u64(&mut ang)?;
                    n_ang_used += 1;
                    // kickmix words[0] is the low word of V = turns * 2^128, words[1] the high word.
                    Some((lo as u128) | ((hi as u128) << 64))
                } else {
                    None
                };
                bld.op(kind, cond, &qs[..nq], bit, angle)?;
            }
            done += n as u64;
        }
        if qqq.iter().any(|a| a.left != 0) || qq.iter().any(|a| a.left != 0) || q0.left != 0 || b0.left != 0 || bc.left != 0 || n_ang_used != n_ang {
            return Err("operand arrays not fully consumed".into());
        }
        let prog = bld.finish(num_qubits, num_bits, registers);
        if prog.num_qubits != num_qubits || prog.orig_num_bits != num_bits {
            return Err(format!(
                "header num_qubits/num_bits {}/{} disagree with ops and registers {}/{}",
                num_qubits, num_bits, prog.num_qubits, prog.orig_num_bits
            ));
        }
        Ok(prog)
    }
}

// ---------------------------------------------------------------------------------------------------------------
// Addendum A: on-disk program cache format. Little-endian hosts only (the arrays are written as raw words).
// Layout: magic, format version, scalar fields, static counts, registers, array lengths, arrays, then a 64-bit
// checksum over everything before it. The Python cache layer keys files by circuit SHA-256 + engine build.

pub const CACHE_MAGIC: &[u8; 8] = b"KMXEPRG\0";
/// Bump whenever the IR or this layout changes. Format 2 (0.3.0) adds `reaction_depth` as a ninth scalar field.
pub const CACHE_FORMAT: u32 = 2;

/// Streaming 64-bit checksum (4 multiply-rotate lanes over 8-byte words; independent of how the bytes are split
/// into calls). Detects truncation and corruption; it is not a cryptographic hash.
struct Ck {
    lanes: [u64; 4],
    words: usize,
    pend: [u8; 8],
    npend: usize,
    total: u64,
}
impl Ck {
    fn new() -> Self {
        Ck { lanes: [0x9E37_79B9_7F4A_7C15, 0xC2B2_AE3D_27D4_EB4F, 0x1656_67B1_9E37_79F9, 0x27D4_EB2F_1656_67C5], words: 0, pend: [0; 8], npend: 0, total: 0 }
    }
    #[inline(always)]
    fn word(&mut self, x: u64) {
        let l = self.words & 3;
        self.lanes[l] = (self.lanes[l] ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29);
        self.words += 1;
    }
    fn bytes(&mut self, mut b: &[u8]) {
        self.total += b.len() as u64;
        if self.npend > 0 {
            let k = (8 - self.npend).min(b.len());
            self.pend[self.npend..self.npend + k].copy_from_slice(&b[..k]);
            self.npend += k;
            b = &b[k..];
            if self.npend < 8 {
                return;
            }
            let x = u64::from_le_bytes(self.pend);
            self.word(x);
            self.npend = 0;
        }
        let mut chunks = b.chunks_exact(8);
        for c in &mut chunks {
            self.word(u64::from_le_bytes(c.try_into().unwrap()));
        }
        let r = chunks.remainder();
        self.pend[..r.len()].copy_from_slice(r);
        self.npend = r.len();
    }
    fn finish(&self) -> u64 {
        let mut s = Ck { lanes: self.lanes, words: self.words, pend: [0; 8], npend: 0, total: 0 };
        if self.npend > 0 {
            let mut t = [0u8; 8];
            t[..self.npend].copy_from_slice(&self.pend[..self.npend]);
            s.word(u64::from_le_bytes(t));
        }
        let mut h = self.total;
        for &x in &s.lanes {
            h = (h ^ x).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(31);
        }
        h ^ (h >> 33)
    }
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    // SAFETY: plain-old-data integer slices, read-only view.
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
    // SAFETY: plain-old-data integer slices; every bit pattern is a valid value.
    unsafe { std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v)) }
}

impl Program {
    pub fn save(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        if cfg!(target_endian = "big") {
            return Err("program cache is little-endian only".into());
        }
        let f = File::create(path).map_err(|e| e.to_string())?;
        let mut w = std::io::BufWriter::with_capacity(1 << 22, f);
        let mut ck = Ck::new();
        let mut put = |w: &mut std::io::BufWriter<File>, b: &[u8]| -> Result<(), String> {
            ck.bytes(b);
            w.write_all(b).map_err(|e| e.to_string())
        };
        let mut head: Vec<u8> = Vec::new();
        head.extend_from_slice(CACHE_MAGIC);
        head.extend_from_slice(&CACHE_FORMAT.to_le_bytes());
        for v in [self.num_qubits, self.num_bits, self.orig_num_bits, self.nrand, self.n_source_ops, self.n_info_tags as usize,
                  self.max_stack_depth, self.n_cx_runs, self.reaction_depth as usize] {
            head.extend_from_slice(&(v as u64).to_le_bytes());
        }
        for v in self.static_counts {
            head.extend_from_slice(&v.to_le_bytes());
        }
        head.extend_from_slice(&(self.registers.len() as u64).to_le_bytes());
        for (name, c) in &self.registers {
            head.extend_from_slice(&(name.len() as u64).to_le_bytes());
            head.extend_from_slice(name.as_bytes());
            head.extend_from_slice(&(c.len() as u64).to_le_bytes());
            for &(isq, id) in c {
                head.extend_from_slice(&(((isq as u32) << 31) | id).to_le_bytes());
            }
        }
        for n in [self.code.len(), self.angles.len(), self.levels.len(), self.bit_map.len(), self.bit_compact.len()] {
            head.extend_from_slice(&(n as u64).to_le_bytes());
        }
        put(&mut w, &head)?;
        put(&mut w, as_bytes(&self.code))?;
        put(&mut w, as_bytes(&self.angles))?;
        put(&mut w, &self.levels)?;
        put(&mut w, as_bytes(&self.bit_map))?;
        put(&mut w, as_bytes(&self.bit_compact))?;
        let sum = ck.finish();
        w.write_all(&sum.to_le_bytes()).map_err(|e| e.to_string())?;
        let f = w.into_inner().map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Program, String> {
        if cfg!(target_endian = "big") {
            return Err("program cache is little-endian only".into());
        }
        let f = File::open(path).map_err(|e| e.to_string())?;
        let file_len = f.metadata().map_err(|e| e.to_string())?.len();
        let mut r = BufReader::with_capacity(1 << 22, f);
        let mut ck = Ck::new();
        let mut take = |r: &mut BufReader<File>, b: &mut [u8]| -> Result<(), String> {
            r.read_exact(b).map_err(|e| format!("cache file truncated: {}", e))?;
            ck.bytes(b);
            Ok(())
        };
        let mut hdr = Vec::new();
        let u64s = |r: &mut BufReader<File>, take: &mut dyn FnMut(&mut BufReader<File>, &mut [u8]) -> Result<(), String>, hdr: &mut Vec<u8>| -> Result<u64, String> {
            let mut b = [0u8; 8];
            take(r, &mut b)?;
            hdr.extend_from_slice(&b);
            Ok(u64::from_le_bytes(b))
        };
        let mut magic = [0u8; 12];
        take(&mut r, &mut magic)?;
        if &magic[..8] != CACHE_MAGIC || u32::from_le_bytes(magic[8..12].try_into().unwrap()) != CACHE_FORMAT {
            return Err("not a program cache file of this format version".into());
        }
        let mut sc = [0u64; 9];
        for v in sc.iter_mut() {
            *v = u64s(&mut r, &mut take, &mut hdr)?;
        }
        let mut static_counts = [0u64; NCOUNT];
        for v in static_counts.iter_mut() {
            *v = u64s(&mut r, &mut take, &mut hdr)?;
        }
        let limit = file_len; // no length field can exceed the file
        let nreg = u64s(&mut r, &mut take, &mut hdr)?;
        if nreg > limit {
            return Err("corrupt cache header".into());
        }
        let mut registers = Vec::with_capacity(nreg as usize);
        for _ in 0..nreg {
            let nl = u64s(&mut r, &mut take, &mut hdr)?;
            if nl > limit {
                return Err("corrupt cache header".into());
            }
            let mut name = vec![0u8; nl as usize];
            take(&mut r, &mut name)?;
            let cl = u64s(&mut r, &mut take, &mut hdr)?;
            if cl * 4 > limit {
                return Err("corrupt cache header".into());
            }
            let mut words = vec![0u32; cl as usize];
            take(&mut r, as_bytes_mut(&mut words))?;
            let c = words.iter().map(|&v| (v >> 31 != 0, v & 0x7FFF_FFFF)).collect();
            registers.push((String::from_utf8(name).map_err(|e| e.to_string())?, c));
        }
        let mut lens = [0u64; 5];
        for v in lens.iter_mut() {
            *v = u64s(&mut r, &mut take, &mut hdr)?;
            if *v > limit {
                return Err("corrupt cache header".into());
            }
        }
        let mut code = vec![0u32; lens[0] as usize];
        take(&mut r, as_bytes_mut(&mut code))?;
        let mut angles = vec![0u128; lens[1] as usize];
        take(&mut r, as_bytes_mut(&mut angles))?;
        let mut levels = vec![0u8; lens[2] as usize];
        take(&mut r, &mut levels)?;
        let mut bit_map = vec![0u32; lens[3] as usize];
        take(&mut r, as_bytes_mut(&mut bit_map))?;
        let mut bit_compact = vec![0u32; lens[4] as usize];
        take(&mut r, as_bytes_mut(&mut bit_compact))?;
        let want = ck.finish();
        let mut sb = [0u8; 8];
        r.read_exact(&mut sb).map_err(|_| "cache file truncated (checksum)".to_string())?;
        if u64::from_le_bytes(sb) != want {
            return Err("cache checksum mismatch".into());
        }
        let mut extra = [0u8; 1];
        if r.read(&mut extra).map_err(|e| e.to_string())? != 0 {
            return Err("cache file has trailing data".into());
        }
        let p = Program {
            code,
            angles,
            levels,
            num_qubits: sc[0] as usize,
            num_bits: sc[1] as usize,
            bit_map,
            bit_compact,
            orig_num_bits: sc[2] as usize,
            nrand: sc[3] as usize,
            n_source_ops: sc[4] as usize,
            n_info_tags: sc[5],
            max_stack_depth: sc[6] as usize,
            n_cx_runs: sc[7] as usize,
            reaction_depth: sc[8],
            static_counts,
            registers,
        };
        // Structural checks so a corrupt but checksum-colliding file cannot index out of bounds.
        if p.bit_map.len() != p.num_bits || p.bit_compact.len() != p.orig_num_bits || p.levels.len() != p.angles.len() {
            return Err("cache file inconsistent".into());
        }
        p.validate()?;
        Ok(p)
    }

    /// Walk the stream and check every operand is in range (the interpreter uses unchecked indexing).
    pub fn validate(&self) -> Result<(), String> {
        let lens = op_len_table();
        let (nq, nb, na) = (self.num_qubits as u32, self.num_bits as u32, self.angles.len() as u32);
        let c = &self.code;
        let mut i = 0usize;
        let mut depth = 0usize;
        while i < c.len() {
            let w0 = c[i];
            let op = (w0 & 0xFF) as u8;
            if op == K_CXRUN {
                let n = (w0 >> 8) as usize;
                if i + 1 + n > c.len() {
                    return Err("cache stream: CX run overruns".into());
                }
                for &e in &c[i + 1..i + 1 + n] {
                    if (e & 0xFFFF) >= nq || (e >> 16) >= nq {
                        return Err("cache stream: qubit out of range".into());
                    }
                }
                i += 1 + n;
                continue;
            }
            let len = lens[op as usize] as usize;
            if len == 0 || i + len > c.len() {
                return Err("cache stream: bad opcode".into());
            }
            let base = op & !IF;
            let nqo = nq_of(base);
            if nqo >= 1 && (w0 >> 16) >= nq {
                return Err("cache stream: qubit out of range".into());
            }
            let mut k = i + 1;
            if nqo >= 2 {
                let w = c[k];
                if (w & 0xFFFF) >= nq || (nqo == 3 && (w >> 16) >= nq) {
                    return Err("cache stream: qubit out of range".into());
                }
                k += 1;
            }
            if has_bit_target(base) {
                if c[k] >= nb {
                    return Err("cache stream: bit out of range".into());
                }
                k += 1;
            }
            if base == K_ZPOW {
                if c[k] >= na {
                    return Err("cache stream: angle out of range".into());
                }
                k += 1;
            }
            if op & IF != 0 || base == K_PUSH {
                if c[k] >= nb {
                    return Err("cache stream: condition bit out of range".into());
                }
            }
            if base == K_PUSH {
                depth += 1;
                if depth > self.max_stack_depth {
                    return Err("cache stream: stack depth exceeds header".into());
                }
            }
            if base == K_POP {
                depth = depth.saturating_sub(1);
            }
            i += len;
        }
        Ok(())
    }
}
