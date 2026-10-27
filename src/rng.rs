//! kickmix's RNG chain: std::seed_seq -> mt19937_64 -> per-256-shot ChaCha8 keys -> vertical ChaCha8 stream.
//! Semantics follow kickmix (Apache-2.0, Google LLC); see NOTICE.

/// std::seed_seq::generate (the standard's algorithm) for `n` output words.
pub fn seed_seq_generate(v: &[u32], n: usize) -> Vec<u32> {
    let mut b = vec![0x8b8b8b8bu32; n];
    let s = v.len();
    let m = std::cmp::max(s + 1, n);
    let t = if n >= 623 { 11 } else if n >= 68 { 7 } else if n >= 39 { 5 } else if n >= 7 { 3 } else { (n - 1) / 2 };
    let p = (n - t) / 2;
    let q = p + t;
    for k in 0..m {
        let x = b[k % n] ^ b[(k + p) % n] ^ b[(k + n - 1) % n];
        let r1 = 1664525u32.wrapping_mul(x ^ (x >> 27));
        let add = if k == 0 { s as u32 } else if k <= s { ((k % n) as u32).wrapping_add(v[k - 1]) } else { (k % n) as u32 };
        let r2 = r1.wrapping_add(add);
        b[(k + p) % n] = b[(k + p) % n].wrapping_add(r1);
        b[(k + q) % n] = b[(k + q) % n].wrapping_add(r2);
        b[k % n] = r2;
    }
    for k in m..m + n {
        let x = b[k % n].wrapping_add(b[(k + p) % n]).wrapping_add(b[(k + n - 1) % n]);
        let r3 = 1566083941u32.wrapping_mul(x ^ (x >> 27));
        let r4 = r3.wrapping_sub((k % n) as u32);
        b[(k + p) % n] ^= r3;
        b[(k + q) % n] ^= r4;
        b[k % n] = r4;
    }
    b
}

pub struct Mt64 {
    mt: [u64; 312],
    idx: usize,
}

impl Mt64 {
    /// mt19937_64 constructed from a std::seed_seq over `seed`.
    pub fn from_seed_seq(seed: &[u32]) -> Self {
        let a = seed_seq_generate(seed, 624);
        let mut mt = [0u64; 312];
        for i in 0..312 {
            mt[i] = (a[2 * i] as u64) | ((a[2 * i + 1] as u64) << 32);
        }
        if (mt[0] & 0xFFFF_FFFF_8000_0000) == 0 && mt[1..].iter().all(|&x| x == 0) {
            mt[0] = 1u64 << 63;
        }
        Mt64 { mt, idx: 312 }
    }
    fn twist(&mut self) {
        const UM: u64 = 0xFFFF_FFFF_8000_0000;
        const LM: u64 = 0x7FFF_FFFF;
        for i in 0..312 {
            let x = (self.mt[i] & UM) | (self.mt[(i + 1) % 312] & LM);
            let xa = (x >> 1) ^ if x & 1 != 0 { 0xB5026F5AA96619E9 } else { 0 };
            self.mt[i] = self.mt[(i + 156) % 312] ^ xa;
        }
        self.idx = 0;
    }
    pub fn next(&mut self) -> u64 {
        if self.idx >= 312 {
            self.twist();
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= (y >> 29) & 0x5555555555555555;
        y ^= (y << 17) & 0x71D67FFFEDA60000;
        y ^= (y << 37) & 0xFFF7EEE000000000;
        y ^= y >> 43;
        y
    }
}

/// ChaCha8 key material of one 256-shot Sim: slot m (0..11) = state words 4..11, 13, 14, 15; 8 u32 lanes each.
pub type Keys = [[u32; 8]; 11];

/// The four master() outputs consumed for one Sim, as 8 LE u32 seed words.
pub fn sim_seed_words(master: &mut Mt64) -> [u32; 8] {
    let mut cs = [0u32; 8];
    for w in 0..4 {
        let r = master.next();
        cs[2 * w] = r as u32;
        cs[2 * w + 1] = (r >> 32) as u32;
    }
    cs
}

pub fn keys_from_sim_seed(cs: &[u32; 8]) -> Keys {
    let mut rng = Mt64::from_seed_seq(cs);
    let mut keys = [[0u32; 8]; 11];
    for m in 0..11 {
        let mut words = [0u64; 4];
        for k in 0..4 {
            words[k] = rng.next();
        }
        for l in 0..8 {
            keys[m][l] = (words[l / 2] >> (32 * (l % 2))) as u32;
        }
    }
    keys
}

/// Keys of the first `nsims` Sims of a Simulator seeded with 8 LE u32 words.
pub fn derive_keys(seed: &[u32; 8], nsims: usize) -> Vec<Keys> {
    let mut master = Mt64::from_seed_seq(seed);
    let seeds: Vec<[u32; 8]> = (0..nsims).map(|_| sim_seed_words(&mut master)).collect();
    seeds.iter().map(keys_from_sim_seed).collect()
}

/// Vertical ChaCha8 over 8 u32 lanes; yields one 256-bit word per `next()` (16 per block).
pub struct ChaCha8 {
    keys: Keys,
    counter: u32,
    buf: [[u32; 8]; 16],
    idx: usize,
}

macro_rules! qr {
    ($x:ident, $a:expr, $b:expr, $c:expr, $d:expr) => {
        for l in 0..8 {
            $x[$a][l] = $x[$a][l].wrapping_add($x[$b][l]);
            $x[$d][l] = ($x[$d][l] ^ $x[$a][l]).rotate_left(16);
            $x[$c][l] = $x[$c][l].wrapping_add($x[$d][l]);
            $x[$b][l] = ($x[$b][l] ^ $x[$c][l]).rotate_left(12);
            $x[$a][l] = $x[$a][l].wrapping_add($x[$b][l]);
            $x[$d][l] = ($x[$d][l] ^ $x[$a][l]).rotate_left(8);
            $x[$c][l] = $x[$c][l].wrapping_add($x[$d][l]);
            $x[$b][l] = ($x[$b][l] ^ $x[$c][l]).rotate_left(7);
        }
    };
}

impl ChaCha8 {
    pub fn new(keys: Keys) -> Self {
        ChaCha8 { keys, counter: 0, buf: [[0; 8]; 16], idx: 16 }
    }
    #[inline(never)]
    fn refill(&mut self) {
        let mut s = [[0u32; 8]; 16];
        let consts = [0x61707865u32, 0x3320646e, 0x79622d32, 0x6b206574];
        for i in 0..4 {
            s[i] = [consts[i]; 8];
        }
        for i in 0..8 {
            s[4 + i] = self.keys[i];
        }
        s[12] = [self.counter; 8];
        for i in 0..3 {
            s[13 + i] = self.keys[8 + i];
        }
        let mut x = s;
        for _ in 0..4 {
            qr!(x, 0, 4, 8, 12);
            qr!(x, 1, 5, 9, 13);
            qr!(x, 2, 6, 10, 14);
            qr!(x, 3, 7, 11, 15);
            qr!(x, 0, 5, 10, 15);
            qr!(x, 1, 6, 11, 12);
            qr!(x, 2, 7, 8, 13);
            qr!(x, 3, 4, 9, 14);
        }
        for i in 0..16 {
            for l in 0..8 {
                self.buf[i][l] = x[i][l].wrapping_add(s[i][l]);
            }
        }
        self.counter = self.counter.wrapping_add(1);
        self.idx = 0;
    }
    #[inline(always)]
    pub fn next(&mut self) -> [u64; 4] {
        if self.idx == 16 {
            self.refill();
        }
        let l = &self.buf[self.idx];
        self.idx += 1;
        [
            (l[0] as u64) | ((l[1] as u64) << 32),
            (l[2] as u64) | ((l[3] as u64) << 32),
            (l[4] as u64) | ((l[5] as u64) << 32),
            (l[6] as u64) | ((l[7] as u64) << 32),
        ]
    }
}
