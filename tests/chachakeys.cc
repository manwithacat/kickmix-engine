// Reference for tests/test_rng.py: kickmix's per-simulator ChaCha8 key/nonce derivation written with the C++
// standard library (std::seed_seq, std::mt19937_64), independent of the engine's Rust re-implementation.
// The derivation follows kickmix (py/interactive_simulator.pybind.cc, sim/sim.h, util/chacha.h; Apache-2.0,
// Google LLC). Written by an AI model (Claude) under the owner's direction.
// Reproduces kickmix's per-sim ChaCha8 key/nonce derivation (py/interactive_simulator.pybind.cc +
// sim.h Sim ctor + chacha.h randomize + b256_neon randomize) and writes S*11*8 uint32:
// for sim s, word slot m in {4..11,13,14,15} -> index 0..10, 8 u32 lanes.
#include <random>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
int main(int argc, char** argv) {
    // usage: chachakeys S out.bin w0 w1 ... w7   (seed as 8 little-endian u32 words)
    int S = atoi(argv[1]);
    std::array<uint32_t, 8> sw{};
    for (int i = 0; i < 8; i++) sw[i] = (uint32_t)strtoull(argv[3 + i], 0, 10);
    std::seed_seq seq(sw.begin(), sw.end());
    std::mt19937_64 master(seq);
    std::vector<uint32_t> out;
    for (int s = 0; s < S; s++) {
        std::array<uint32_t, 8> cs{};
        for (int w = 0; w < 4; w++) { uint64_t r = master(); cs[2*w] = (uint32_t)r; cs[2*w+1] = (uint32_t)(r >> 32); }
        std::seed_seq sq(cs.begin(), cs.end());
        std::mt19937_64 rng(sq);
        for (int m = 0; m < 11; m++) {
            uint64_t words[4];
            for (int k = 0; k < 4; k++) words[k] = rng();
            for (int l = 0; l < 8; l++) out.push_back((uint32_t)(words[l/2] >> (32*(l%2))));
        }
    }
    FILE* f = fopen(argv[2], "wb"); fwrite(out.data(), 4, out.size(), f); fclose(f);
}
