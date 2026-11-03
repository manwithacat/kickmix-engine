// kickmix-gpu kernel: a transcription of kickmix-engine `Sim::run_impl` (src/sim.rs) in WGSL.
// One thread owns LANES shots held in one value of type T (u32 = 32 shots, vec4<u32> = 128 shots); component c of
// a thread for lane group g of a 256-shot sim is ChaCha lane g * NCOMP + c of kickmix's vertical RNG stream.
// State: qubits q[qubit * NT + t], classical bits bits[bit * NT + t] (coalesced across the threads of a simdgroup
// because every thread executes the same op at the same time), per-thread context in ctxt/ctxu between dispatches.
// Semantics follow kickmix (Apache-2.0, Google LLC); see the repository NOTICE. Written by an AI model (Claude)
// under the owner's direction. Placeholders in braces are filled by gpu.rs.

alias T = {T};
const LANES: u32 = {LANES}u;
const NCOMP: u32 = {NCOMP}u;
const KEYSTRIDE: u32 = {KEYSTRIDE}u;   // 11 * NCOMP key words per thread
const ONES: T = {ONES};
const ZERO: T = {ZERO};
const CTXT: u32 = 10u;    // T slots per thread: phase, base, stack[8]
const CTXU: u32 = 40u;    // u32 slots per thread: depth, rpos, cnt_lo[18], cnt_hi[18], error, pad

struct Params {
    seg_start: u32,
    seg_end: u32,
    nthreads: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read> code: array<u32>;
@group(0) @binding(1) var<storage, read_write> q: array<T>;
@group(0) @binding(2) var<storage, read_write> bits: array<T>;
@group(0) @binding(3) var<storage, read_write> ctxt: array<T>;
@group(0) @binding(4) var<storage, read_write> ctxu: array<u32>;
@group(0) @binding(5) var<storage, read> keys: array<u32>;
@group(0) @binding(6) var<uniform> P: Params;

fn popc(x: T) -> u32 {
    {POPC}
}

fn keyw(kb: u32, m: u32) -> T {
    {KEYLOAD}
}

fn rotl(x: T, n: u32) -> T {
    return (x << {SHIFTV}(n)) | (x >> {SHIFTV}(32u - n));
}

fn qr(x: ptr<function, array<T, 16>>, a: u32, b: u32, c: u32, d: u32) {
    (*x)[a] = (*x)[a] + (*x)[b];
    (*x)[d] = rotl((*x)[d] ^ (*x)[a], 16u);
    (*x)[c] = (*x)[c] + (*x)[d];
    (*x)[b] = rotl((*x)[b] ^ (*x)[c], 12u);
    (*x)[a] = (*x)[a] + (*x)[b];
    (*x)[d] = rotl((*x)[d] ^ (*x)[a], 8u);
    (*x)[c] = (*x)[c] + (*x)[d];
    (*x)[b] = rotl((*x)[b] ^ (*x)[c], 7u);
}

// ChaCha8 block `counter` of every component's stream (kickmix: key words 4..11, counter 12, nonce 13..15).
fn chacha_block(kb: u32, counter: u32, out: ptr<function, array<T, 16>>) {
    var s: array<T, 16>;
    s[0] = T(0x61707865u);
    s[1] = T(0x3320646eu);
    s[2] = T(0x79622d32u);
    s[3] = T(0x6b206574u);
    for (var m = 0u; m < 8u; m++) {
        s[4u + m] = keyw(kb, m);
    }
    s[12] = T(counter);
    s[13] = keyw(kb, 8u);
    s[14] = keyw(kb, 9u);
    s[15] = keyw(kb, 10u);
    var x = s;
    for (var r = 0u; r < 4u; r++) {
        qr(&x, 0u, 4u, 8u, 12u);
        qr(&x, 1u, 5u, 9u, 13u);
        qr(&x, 2u, 6u, 10u, 14u);
        qr(&x, 3u, 7u, 11u, 15u);
        qr(&x, 0u, 5u, 10u, 15u);
        qr(&x, 1u, 6u, 11u, 12u);
        qr(&x, 2u, 7u, 8u, 13u);
        qr(&x, 3u, 4u, 9u, 14u);
    }
    for (var k = 0u; k < 16u; k++) {
        (*out)[k] = x[k] + s[k];
    }
}

// One `ChaCha8::next()`: word rpos % 16 of block rpos / 16; the block is recomputed when a new one starts.
fn rnd(kb: u32, rpos: ptr<function, u32>, buf: ptr<function, array<T, 16>>) -> T {
    let p = *rpos;
    if ((p & 15u) == 0u) {
        chacha_block(kb, p >> 4u, buf);
    }
    *rpos = p + 1u;
    return (*buf)[p & 15u];
}

@compute @workgroup_size({WG})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    let NT = P.nthreads;
    if (t >= NT) {
        return;
    }
    let ct = t * CTXT;
    let cu = t * CTXU;
    var phase = ctxt[ct];
    var base = ctxt[ct + 1u];
    var stack: array<T, 8>;
    for (var k = 0u; k < 8u; k++) {
        stack[k] = ctxt[ct + 2u + k];
    }
    var depth = ctxu[cu];
    var rpos = ctxu[cu + 1u];
    var err = ctxu[cu + 38u];
    var cnt: array<u32, 18>;
    for (var k = 0u; k < 18u; k++) {
        cnt[k] = 0u;
    }
    var buf: array<T, 16>;
    let kb = t * KEYSTRIDE;
    if ((rpos & 15u) != 0u) {
        chacha_block(kb, rpos >> 4u, &buf);
    }
    var full = (depth == 0u);
    var i = P.seg_start;
    let end = P.seg_end;
    loop {
        if (i >= end) {
            break;
        }
        let w0 = code[i];
        let opc = w0 & 0xFFu;
        if (opc == 18u) {
            // CXRUN: n plain CX entries `a | b << 16`.
            let n = w0 >> 8u;
            let run = i + 1u;
            if (full) {
                for (var j = 0u; j < n; j++) {
                    let r = code[run + j];
                    let tv = q[(r & 0xFFFFu) * NT + t];
                    let bi = (r >> 16u) * NT + t;
                    q[bi] = q[bi] ^ tv;
                }
                cnt[11] += n * LANES;
            } else {
                for (var j = 0u; j < n; j++) {
                    let r = code[run + j];
                    let tv = base & q[(r & 0xFFFFu) * NT + t];
                    let bi = (r >> 16u) * NT + t;
                    q[bi] = q[bi] ^ tv;
                }
                cnt[11] += n * popc(base);
            }
            i = run + n;
            continue;
        }
        if (opc == 19u) {
            // CXBUNDLE: k independent plain CXs, loads first, then stores (see gpu.rs `bundle_stream`).
            let k = w0 >> 8u;
            let run = i + 1u;
            switch k {
{BUNDLE_CASES}                default: {
                    err = err | 0x80000000u;
                }
            }
            if (full) {
                cnt[11] += k * LANES;
            } else {
                cnt[11] += k * popc(base);
            }
            i = run + k;
            continue;
        }
        let kind = opc & 31u;
        // OP_LEN: 1 + (second operand word for kinds 2,3,4,8..16) + (condition word if IF).
        let len = 1u + ((0x1FF1Cu >> kind) & 1u) + ((opc >> 5u) & 1u);
        let a = (w0 >> 16u) * NT + t;
        var cond = base;
        var cfull = full;
        if ((opc & 32u) != 0u) {
            cond = base & bits[code[i + len - 1u] * NT + t];
            cfull = false;
        }
        switch kind {
            case 11u: { // CX
                let b = (code[i + 1u] & 0xFFFFu) * NT + t;
                q[b] = q[b] ^ (cond & q[a]);
            }
            case 10u: { // CZ
                let b = (code[i + 1u] & 0xFFFFu) * NT + t;
                phase = phase ^ (cond & q[a] & q[b]);
            }
            case 12u: { // SWAP
                let b = (code[i + 1u] & 0xFFFFu) * NT + t;
                let tv = (q[a] ^ q[b]) & cond;
                q[a] = q[a] ^ tv;
                q[b] = q[b] ^ tv;
            }
            case 14u: { // CCX
                let w1 = code[i + 1u];
                let b = (w1 & 0xFFFFu) * NT + t;
                let c = (w1 >> 16u) * NT + t;
                q[c] = q[c] ^ (cond & q[a] & q[b]);
            }
            case 13u: { // CCZ
                let w1 = code[i + 1u];
                let b = (w1 & 0xFFFFu) * NT + t;
                let c = (w1 >> 16u) * NT + t;
                phase = phase ^ (cond & q[a] & q[b] & q[c]);
            }
            case 15u: { // CSWAP
                let w1 = code[i + 1u];
                let b = (w1 & 0xFFFFu) * NT + t;
                let c = (w1 >> 16u) * NT + t;
                let m = cond & q[a];
                let tv = (q[b] ^ q[c]) & m;
                q[b] = q[b] ^ tv;
                q[c] = q[c] ^ tv;
            }
            case 6u: { // X
                q[a] = q[a] ^ cond;
            }
            case 5u: { // Z
                phase = phase ^ (cond & q[a]);
            }
            case 0u: { // NEG
                phase = phase ^ cond;
            }
            case 2u: { // BIT_STORE0
                let b = code[i + 1u] * NT + t;
                bits[b] = bits[b] & ~cond;
            }
            case 3u: { // BIT_STORE1
                let b = code[i + 1u] * NT + t;
                bits[b] = bits[b] | cond;
            }
            case 4u: { // BIT_INVERT
                let b = code[i + 1u] * NT + t;
                bits[b] = bits[b] ^ cond;
            }
            case 9u: { // HMR: the draw happens whatever the condition, as in kickmix.
                let b = code[i + 1u] * NT + t;
                let r = rnd(kb, &rpos, &buf);
                let nb = (bits[b] & ~cond) ^ (r & cond);
                bits[b] = nb;
                phase = phase ^ (q[a] & nb & cond);
                q[a] = q[a] & ~cond;
            }
            case 7u: { // R
                let r = rnd(kb, &rpos, &buf);
                phase = phase ^ (q[a] & r & cond);
                q[a] = q[a] & ~cond;
            }
            case 16u: { // PUSH_CONDITION
                base = base & bits[code[i + 1u] * NT + t];
                if (depth < 8u) {
                    stack[depth] = base;
                }
                depth = depth + 1u;
                full = false;
                cond = ONES;
                cfull = true;
            }
            case 1u: { // POP_CONDITION
                if (depth > 0u) {
                    depth = depth - 1u;
                }
                if (depth > 0u) {
                    base = stack[depth - 1u];
                    full = false;
                } else {
                    base = ONES;
                    full = true;
                }
                cond = ONES;
                cfull = true;
            }
            default: { // Z_POW (8) is not supported by this prototype; anything else is a corrupt stream.
                err = err | (1u << kind);
            }
        }
        i += len;
        cnt[kind] += select(popc(cond), LANES, cfull);
    }
    ctxt[ct] = phase;
    ctxt[ct + 1u] = base;
    for (var k = 0u; k < 8u; k++) {
        ctxt[ct + 2u + k] = stack[k];
    }
    ctxu[cu] = depth;
    ctxu[cu + 1u] = rpos;
    ctxu[cu + 38u] = err;
    for (var k = 0u; k < 18u; k++) {
        let lo = ctxu[cu + 2u + k] + cnt[k];
        if (lo < cnt[k]) {
            ctxu[cu + 20u + k] = ctxu[cu + 20u + k] + 1u;
        }
        ctxu[cu + 2u + k] = lo;
    }
}
