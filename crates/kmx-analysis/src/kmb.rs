//! Streaming reader of the parts of a version-2 .kmb that liveness needs: header, registers, the op-type stream and
//! the six qubit-operand streams. Layout as kickmix `Circuit::write_kmb_to` and the engine's `program::KmbOps`
//! (carried here, not shared, so the verify path is untouched; tested against `scripts/qhdl_kmb.read`).
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};

pub const MAGIC: [u8; 16] = [
    0xd7, 0x50, 0xc7, 0xd5, 0xc3, 0x29, 0xd3, 0x26, 0xe3, 0xcc, 0x9f, 0x68, 0x34, 0xf2, 0xb8, 0xbf,
];
pub const POP_CONDITION: u8 = 0x01;
pub const PUSH_CONDITION: u8 = 0x81;
pub const R: u8 = 0x12;
pub const HMR: u8 = 0x50;
pub const R_IF: u8 = 0x92;
pub const HMR_IF: u8 = 0xD0;

#[derive(Clone, Debug)]
pub struct Header {
    pub num_qubits: u32,
    pub num_bits: u32,
    pub num_ops: u64,
    pub n_qqq: u64,
    pub n_qq: u64,
    pub n_q: u64,
    /// (name, [(is_qubit, id)])
    pub registers: Vec<(String, Vec<(bool, u32)>)>,
    /// payload offsets: op types, qqq0, qqq1, qqq2, qq0, qq1, q0
    pub offs: [u64; 7],
}

impl Header {
    /// Register qubit ids, sorted and de-duplicated (the keys of Python `register_qubits`).
    pub fn register_qubits(&self) -> Vec<u32> {
        let mut v: Vec<u32> =
            self.registers.iter().flat_map(|(_, c)| c.iter().filter(|x| x.0).map(|x| x.1)).collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

fn err<E: ToString>(path: &str, e: E) -> String {
    format!("{}: {}", path, e.to_string())
}

fn rd<const N: usize, Rd: Read>(r: &mut Rd) -> Result<[u8; N], String> {
    let mut b = [0u8; N];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(b)
}
fn rd_u32<Rd: Read>(r: &mut Rd) -> Result<u32, String> {
    Ok(u32::from_le_bytes(rd::<4, _>(r)?))
}
fn rd_u64<Rd: Read>(r: &mut Rd) -> Result<u64, String> {
    Ok(u64::from_le_bytes(rd::<8, _>(r)?))
}

pub fn read_header(path: &str) -> Result<Header, String> {
    let f = File::open(path).map_err(|e| err(path, e))?;
    let file_len = f.metadata().map_err(|e| err(path, e))?.len();
    let mut r = BufReader::with_capacity(1 << 16, f);
    let bad = |what: &str| format!("{}: not a valid version-2 .kmb ({})", path, what);
    if file_len < 16 + 16 + 72 {
        return Err(bad("file shorter than the header"));
    }
    if rd::<16, _>(&mut r)? != MAGIC {
        return Err(bad("wrong magic bytes"));
    }
    let version = rd_u32(&mut r)?;
    if version != 2 {
        return Err(bad(&format!("format version {}", version)));
    }
    let num_qubits = rd_u32(&mut r)?;
    let num_bits = rd_u32(&mut r)?;
    let num_regs = rd_u32(&mut r)?;
    let mut h = [0u64; 9];
    for x in h.iter_mut() {
        *x = rd_u64(&mut r)?;
    }
    let [num_ops, n_qqq, n_qq, n_q, n_b, n_c, n_ang, n_info, n_info_chars] = h;
    let mut pos: u64 = 16 + 16 + 72;
    let mut registers = Vec::with_capacity(num_regs as usize);
    for _ in 0..num_regs {
        if rd_u64(&mut r).map_err(|_| bad("truncated register packets"))? != 0 {
            return Err(bad("register packet expected"));
        }
        let size = rd_u64(&mut r)?;
        let name_len = rd_u32(&mut r)? as usize;
        let reg_len = rd_u32(&mut r)? as usize;
        let mut name = vec![0u8; name_len];
        r.read_exact(&mut name).map_err(|_| bad("truncated register packets"))?;
        let mut contents = Vec::with_capacity(reg_len);
        for _ in 0..reg_len {
            let v = rd_u32(&mut r).map_err(|_| bad("truncated register packets"))?;
            contents.push((v & 0x8000_0000 != 0, v & 0x7FFF_FFFF));
        }
        registers.push((String::from_utf8(name).map_err(|_| bad("register name"))?, contents));
        // Python advances by the bytes it read (8 + name + 4*len), not by `size`; keep the same walk.
        let _ = size;
        pos += 16 + 8 + name_len as u64 + 4 * reg_len as u64;
    }
    drop(r);
    let sizes: [(u64, u64); 10] = [
        (1, num_ops),
        (2, n_qqq * 4),
        (3, n_qqq * 4),
        (4, n_qqq * 4),
        (5, n_qq * 4),
        (6, n_qq * 4),
        (7, n_q * 4),
        (8, n_b * 4),
        (9, n_c * 4),
        (10, n_ang * 16),
    ];
    let mut hf = File::open(path).map_err(|e| err(path, e))?;
    let mut offs = [0u64; 7];
    let mut p = pos;
    for (i, &(ty, sz)) in sizes.iter().enumerate() {
        if p + 16 > file_len {
            return Err(bad("truncated before a stream"));
        }
        hf.seek(SeekFrom::Start(p)).map_err(|e| err(path, e))?;
        let t = rd_u64(&mut hf)?;
        let s = rd_u64(&mut hf)?;
        if t != ty {
            return Err(bad(&format!("packet type {}, expected {}", t, ty)));
        }
        if s != sz || p + 16 + s > file_len {
            return Err(bad(&format!("stream {}: bad size", ty)));
        }
        if i < 7 {
            offs[i] = p + 16;
        }
        p += 16 + sz;
    }
    if p + 24 > file_len {
        return Err(bad("truncated before info tags"));
    }
    hf.seek(SeekFrom::Start(p)).map_err(|e| err(path, e))?;
    if rd_u64(&mut hf)? != 11 {
        return Err(bad("info-tag packet expected"));
    }
    let size = rd_u64(&mut hf)?;
    if rd_u64(&mut hf)? != n_info {
        return Err(bad("info-tag count mismatch"));
    }
    let _ = n_info_chars;
    if p + 16 + size != file_len {
        return Err(bad("trailing bytes"));
    }
    Ok(Header { num_qubits, num_bits, num_ops, n_qqq, n_qq, n_q, registers, offs })
}

/// Chunked sequential reader of one little-endian u32 (or u8) packet.
struct Chunked {
    f: File,
    left_bytes: u64,
    buf: Vec<u8>,
    pos: usize,
    len: usize,
}

impl Chunked {
    fn open(path: &str, off: u64, nbytes: u64, cap: usize) -> Result<Self, String> {
        let mut f = File::open(path).map_err(|e| err(path, e))?;
        f.seek(SeekFrom::Start(off)).map_err(|e| err(path, e))?;
        Ok(Chunked { f, left_bytes: nbytes, buf: vec![0u8; cap], pos: 0, len: 0 })
    }
    #[inline(never)]
    fn refill(&mut self) -> Result<(), String> {
        let n = (self.buf.len() as u64).min(self.left_bytes) as usize;
        if n == 0 {
            return Err("stream exhausted".into());
        }
        self.f.read_exact(&mut self.buf[..n]).map_err(|e| e.to_string())?;
        self.left_bytes -= n as u64;
        self.pos = 0;
        self.len = n;
        Ok(())
    }
    #[inline]
    fn u8(&mut self) -> Result<u8, String> {
        if self.pos == self.len {
            self.refill()?;
        }
        let v = self.buf[self.pos];
        self.pos += 1;
        Ok(v)
    }
    #[inline]
    fn u32(&mut self) -> Result<u32, String> {
        if self.pos == self.len {
            self.refill()?; // chunk sizes are multiples of 4
        }
        let b = &self.buf[self.pos..self.pos + 4];
        self.pos += 4;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn exhausted(&self) -> bool {
        self.left_bytes == 0 && self.pos == self.len
    }
}

/// Stream every op in file order: `f(op_index, op_type, qubits, slot0)` where `qubits` are the operands in stream
/// order and `slot0` is the stream index (0 = qqq0, 3 = qq0, 5 = q0) of the first operand; operand i sits in stream
/// slot0 + i. Ops without qubit operands are passed with an empty slice. The class of an op is `t & 0x30`, exactly
/// as `qhdl_kmb.op_qubits`.
pub fn for_each_op<F: FnMut(u64, u8, &[u32], usize)>(path: &str, mut f: F) -> Result<Header, String> {
    let h = read_header(path)?;
    const CAP: usize = 1 << 20; // multiple of 4
    let mut types = Chunked::open(path, h.offs[0], h.num_ops, CAP)?;
    let mut st: Vec<Chunked> = Vec::with_capacity(6);
    for (i, n) in [h.n_qqq, h.n_qqq, h.n_qqq, h.n_qq, h.n_qq, h.n_q].iter().enumerate() {
        st.push(Chunked::open(path, h.offs[i + 1], n * 4, CAP)?);
    }
    let (mut c3, mut c2, mut c1) = (0u64, 0u64, 0u64);
    let mut q = [0u32; 3];
    for o in 0..h.num_ops {
        let t = types.u8()?;
        match t & 0x30 {
            0x30 => {
                c3 += 1;
                if c3 > h.n_qqq {
                    return Err(format!("{}: more 3-qubit ops than the header count", path));
                }
                q[0] = st[0].u32()?;
                q[1] = st[1].u32()?;
                q[2] = st[2].u32()?;
                f(o, t, &q[..3], 0);
            }
            0x20 => {
                c2 += 1;
                if c2 > h.n_qq {
                    return Err(format!("{}: more 2-qubit ops than the header count", path));
                }
                q[0] = st[3].u32()?;
                q[1] = st[4].u32()?;
                f(o, t, &q[..2], 3);
            }
            0x10 => {
                c1 += 1;
                if c1 > h.n_q {
                    return Err(format!("{}: more 1-qubit ops than the header count", path));
                }
                q[0] = st[5].u32()?;
                f(o, t, &q[..1], 5);
            }
            _ => f(o, t, &q[..0], 6),
        }
    }
    if c3 != h.n_qqq || c2 != h.n_qq || c1 != h.n_q || !types.exhausted() || st.iter().any(|s| !s.exhausted()) {
        return Err(format!("{}: operand counts do not match the op-type stream", path));
    }
    Ok(h)
}
