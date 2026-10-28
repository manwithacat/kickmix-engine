"""RNG chain unit test: engine keys vs the C++ reference (std::seed_seq/mt19937_64) in chachakeys.cc."""
import subprocess, sys, os, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
import kickmix_engine as ke

# C++ reference binary (std::seed_seq / mt19937_64 / ChaCha8 keys), built from the GPU spike's chachakeys.cc.
CK = os.environ.get("KICKMIX_ENGINE_CHACHAKEYS",
                    "/Volumes/SSD/Kaggle/competitions/google-verifiable-quantum-circuit-compilation/outputs/gpu-spike-2026-10-27/chachakeys")
rng = np.random.default_rng(1)
for trial in range(4):
    seed = int.from_bytes(rng.bytes(32), "little") if trial else 0
    S = 5
    out = os.path.join(tempfile.mkdtemp(), "k.bin")
    subprocess.run([CK, str(S), out] + [str(int(w)) for w in ke.seed_words(seed)], check=True)
    ref = np.fromfile(out, dtype="<u4").reshape(S, 11, 8)
    got = ke.derive_keys([int(w) for w in ke.seed_words(seed)], S).reshape(S, 11, 8)
    assert (ref == got).all(), (trial, seed)
print("rng keys identical to chachakeys (4 seeds x 5 sims)")
