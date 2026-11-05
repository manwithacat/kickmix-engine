//! Unkeyed BLAKE2b (RFC 7693) with a short digest, enough to reproduce Python's
//! `hashlib.blake2b(data, digest_size=8)` for the value simulator's fingerprints (card E3). No dependency.

const IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

#[inline]
fn compress(h: &mut [u64; 8], m: &[u64; 16], t: u128, last: bool) {
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in SIGMA.iter() {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// BLAKE2b of `data` with an `outlen`-byte digest (1..=8 here), returned as the little-endian integer of the digest.
pub fn blake2b_u64(data: &[u8], outlen: usize) -> u64 {
    assert!((1..=8).contains(&outlen));
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ outlen as u64;
    let mut m = [0u64; 16];
    let n = data.len();
    let mut off = 0usize;
    // all full blocks except the last block (which may be full)
    while n - off > 128 {
        for (i, w) in m.iter_mut().enumerate() {
            let b = &data[off + 8 * i..off + 8 * i + 8];
            *w = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        }
        off += 128;
        compress(&mut h, &m, off as u128, false);
    }
    let mut last = [0u8; 128];
    last[..n - off].copy_from_slice(&data[off..]);
    for (i, w) in m.iter_mut().enumerate() {
        let b = &last[8 * i..8 * i + 8];
        *w = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
    }
    compress(&mut h, &m, n as u128, true);
    let full = h[0];
    if outlen == 8 {
        full
    } else {
        full & ((1u64 << (8 * outlen)) - 1)
    }
}

/// `hashlib.blake2b(v.to_bytes(nb, 'little'), digest_size=8)` for a bitset held as little-endian u64 words.
/// `nb` <= 8 * words.len(); words beyond the value are zero by construction.
pub fn fp_words(words: &[u64], nb: usize, buf: &mut Vec<u8>) -> u64 {
    buf.clear();
    for w in words {
        buf.extend_from_slice(&w.to_le_bytes());
    }
    buf.truncate(nb);
    blake2b_u64(buf, 8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_vectors() {
        // hashlib.blake2b(x, digest_size=8).digest() read little-endian
        let le = |hex: &str| {
            let b: Vec<u8> = (0..8).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
            u64::from_le_bytes(b.try_into().unwrap())
        };
        assert_eq!(blake2b_u64(b"", 8), le("e4a6a0577479b2b4"));
        assert_eq!(blake2b_u64(b"abc", 8), le("d8bb14d833d59559"));
        let v: Vec<u8> = (0..512).map(|i| (i % 256) as u8).collect();
        assert_eq!(blake2b_u64(&v, 8), le("e4ac345f5aeb2a0e"));
    }
}
