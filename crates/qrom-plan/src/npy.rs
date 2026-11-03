//! Minimal NumPy `.npy` reader for 2-D integer arrays (C order, little-endian).

use std::path::Path;

pub struct IntArray2 {
    pub shape: (usize, usize),
    pub data: Vec<i64>,
}

impl IntArray2 {
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> i64 {
        self.data[i * self.shape.1 + j]
    }
}

pub fn read_npy_i64(path: &Path) -> Result<IntArray2, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err("not an .npy file".into());
    }
    let major = bytes[6];
    let (hlen, start) = match major {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        _ => return Err(format!("unsupported .npy version {major}")),
    };
    let header = std::str::from_utf8(&bytes[start..start + hlen]).map_err(|_| "bad header")?;
    let descr = field(header, "descr").ok_or("no descr")?;
    let descr = descr.trim_matches(|c| c == '\'' || c == '"').to_string();
    let fortran = field(header, "fortran_order").ok_or("no fortran_order")?;
    if fortran.trim() != "False" {
        return Err("fortran_order arrays unsupported".into());
    }
    let shape_txt = field(header, "shape").ok_or("no shape")?;
    let dims: Vec<usize> = shape_txt
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>().map_err(|e| format!("shape: {e}")))
        .collect::<Result<_, _>>()?;
    if dims.len() != 2 {
        return Err(format!("expected a 2-D array, got shape {dims:?}"));
    }
    let n = dims[0] * dims[1];
    let body = &bytes[start + hlen..];
    let (size, signed): (usize, bool) = match descr.as_str() {
        "|i1" => (1, true),
        "|u1" => (1, false),
        "<i2" => (2, true),
        "<u2" => (2, false),
        "<i4" => (4, true),
        "<u4" => (4, false),
        "<i8" => (8, true),
        "<u8" => (8, false),
        other => return Err(format!("unsupported dtype {other}")),
    };
    if body.len() < n * size {
        return Err("truncated .npy body".into());
    }
    let mut data = Vec::with_capacity(n);
    for i in 0..n {
        let b = &body[i * size..(i + 1) * size];
        let v: i64 = match (size, signed) {
            (1, true) => b[0] as i8 as i64,
            (1, false) => b[0] as i64,
            (2, true) => i16::from_le_bytes([b[0], b[1]]) as i64,
            (2, false) => u16::from_le_bytes([b[0], b[1]]) as i64,
            (4, true) => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
            (4, false) => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as i64,
            (8, true) => i64::from_le_bytes(b.try_into().unwrap()),
            (8, false) => u64::from_le_bytes(b.try_into().unwrap()) as i64,
            _ => unreachable!(),
        };
        data.push(v);
    }
    Ok(IntArray2 { shape: (dims[0], dims[1]), data })
}

fn field<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let k = format!("'{key}':");
    let i = header.find(&k)? + k.len();
    let rest = &header[i..];
    let rest = rest.trim_start();
    if rest.starts_with('(') {
        let j = rest.find(')')?;
        return Some(&rest[..=j]);
    }
    let j = rest.find(',').unwrap_or(rest.len());
    let j2 = rest.find('}').unwrap_or(rest.len());
    Some(&rest[..j.min(j2)])
}

/// Write a 2-D array as little-endian int16 `.npy` (version 1.0, C order).
pub fn write_npy_i16(path: &Path, rows: usize, cols: usize, data: &[i64]) -> Result<(), String> {
    if data.len() != rows * cols {
        return Err("shape mismatch".into());
    }
    let mut header = format!("{{'descr': '<i2', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    let pad = 64 - (10 + header.len() + 1) % 64;
    header.push_str(&" ".repeat(pad % 64));
    header.push('\n');
    let mut bytes = Vec::with_capacity(10 + header.len() + 2 * data.len());
    bytes.extend_from_slice(b"\x93NUMPY\x01\x00");
    bytes.extend_from_slice(&(header.len() as u16).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for &v in data {
        if !(i16::MIN as i64..=i16::MAX as i64).contains(&v) {
            return Err(format!("value {v} does not fit int16"));
        }
        bytes.extend_from_slice(&(v as i16).to_le_bytes());
    }
    std::fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}
