"""Build the kickmix-free test fixtures (run once, with an interpreter where kickmix imports).

Usage: make_fixtures.py OUT_DIR

Writes:
  random_ops.kmb       a random circuit (24 qubits, 40 bits, 3000 ops) over every op kind the engine supports, with
                       bit conditions, nested PUSH/POP_CONDITION, INFO_TAG and Clifford Z_POW angles
  random_ops_ref.npz   kickmix reference for 4 batches: seeds, random input states (all qubits and bits), and the
                       final qubits, classical bits, phase flips and per-kind op counts of `km.Simulator(batch_size=256,
                       counting_operations=True, seed=seed)` after `sim.do(circuit)`
The tests (tests/test_engine.py) compare the engine against these without needing kickmix.
"""
import os, sys
import numpy as np
import kickmix as km

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(20261028)
NQ, NB, NOPS = 24, 40, 3000
lines = []
depth = 0


def q(k=1):
    return [f'q{i}' for i in rng.choice(NQ, size=k, replace=False)]


def b():
    return f'b{rng.integers(NB)}'


for _ in range(NOPS):
    r = rng.random()
    cond = f' if {b()}' if rng.random() < 0.25 else ''
    if r < 0.03 and depth < 3:
        lines.append(f'PUSH_CONDITION if {b()}'); depth += 1; continue
    if r < 0.06 and depth > 0:
        lines.append('POP_CONDITION'); depth -= 1; continue
    if r < 0.08:
        lines.append(f'INFO_TAG "t{rng.integers(100)}"'); continue
    kind = rng.choice(['X', 'Z', 'CX', 'CX', 'CX', 'CZ', 'SWAP', 'CCX', 'CCZ', 'CSWAP', 'HMR', 'R', 'BIT_STORE0',
                       'BIT_STORE1', 'BIT_INVERT', 'NEG', 'Z_POW'])
    if kind in ('X', 'Z', 'R'):
        lines.append(f'{kind} {q()[0]}{cond}')
    elif kind in ('CX', 'CZ', 'SWAP'):
        lines.append(f'{kind} {" ".join(q(2))}{cond}')
    elif kind in ('CCX', 'CCZ', 'CSWAP'):
        lines.append(f'{kind} {" ".join(q(3))}{cond}')
    elif kind == 'HMR':
        lines.append(f'HMR {q()[0]} {b()}{cond}')
    elif kind in ('BIT_STORE0', 'BIT_STORE1', 'BIT_INVERT'):
        lines.append(f'{kind} {b()}{cond}')
    elif kind == 'NEG':
        lines.append(f'NEG{cond}')
    else:
        lines.append(f'Z_POW {q()[0]} {rng.choice(["1.0", "3.0"])}{cond}')  # half turns: Clifford, phase stays +-1
while depth:
    lines.append('POP_CONDITION'); depth -= 1
lines.append(f'X q{NQ - 1}')  # make sure the last qubit and bit exist
lines.append(f'BIT_INVERT b{NB - 1}')
text = '\n'.join(lines) + '\n'
kmx = os.path.join(out, 'random_ops.kmx')
open(kmx, 'w').write(text)
c = km.Circuit.from_file(kmx, format='kmx')
os.unlink(kmx)
kmb = os.path.join(out, 'random_ops.kmb')
c.to_file(kmb, format='kmb')
c = km.Circuit.from_file(kmb, format='kmb')
assert c.num_qubits == NQ and c.num_bits == NB, (c.num_qubits, c.num_bits)

allq = km.array([km.q(i) for i in range(NQ)])
allb = km.array([km.b(i) for i in range(NB)])
seeds = [int.from_bytes(rng.bytes(32), 'little') for _ in range(4)]
qin = rng.random((4, NQ, 256)) < 0.5
bin_ = rng.random((4, NB, 256)) < 0.5
qout = np.zeros_like(qin); bout = np.zeros_like(bin_); ph = np.zeros((4, 256), bool)
names = sorted({k for k in dict(c.operation_statistics())})
counts = {}
for i, sd in enumerate(seeds):
    sim = km.Simulator(batch_size=256, debug_print=False, counting_operations=True, seed=sd)
    sim.use_same_registers_as(c)
    sim.write_across_shots(allq, qin[i])
    sim.write_across_shots(allb, bin_[i])
    sim.do(c)
    qout[i] = sim.read_across_shots(allq)
    bout[i] = sim.read_across_shots(allb)
    ph[i] = sim.read_phase_flipped_across_shots()
    st = dict(sim.operation_statistics())
    counts[i] = st
kinds = sorted({k for st in counts.values() for k in st})
cnt = np.array([[counts[i].get(k, 0) for k in kinds] for i in range(4)], dtype=np.int64)
np.savez_compressed(os.path.join(out, 'random_ops_ref.npz'),
                    seeds=np.array([[(s >> (32 * j)) & 0xFFFFFFFF for j in range(8)] for s in seeds], dtype=np.uint32),
                    qin=qin, bin=bin_, qout=qout, bout=bout, phase=ph, count_kinds=np.array(kinds), counts=cnt,
                    num_ops=len(c), kickmix=km.__file__)
print('wrote', kmb, len(c), 'ops;', dict(c.operation_statistics()))
