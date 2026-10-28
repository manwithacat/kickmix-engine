"""M3 gate 1: per-shot parity of kickmix-engine vs kickmix on DQI circuits (run with the kickmix interpreter).

Usage: parity_dqi.py OUT.json --kmb A.kmb [--kmb B.kmb] [--failing failing_shots.jsonl] [--scratch DIR]

Per circuit:
- static checks: verifier hash (kickmix re-serialisation) == file SHA-256; num_qubits, num_bits, len(circuit), static
  op counts, register ids (verifier's validate_registers vs the engine's .kmb reader);
- verifier chunks (one chunk = one 256-shot batch with its own Simulator seed, exactly as `run_batch`): the empty-salt
  chunks 0,1,2,200,399, one salt-check-1 chunk, and every chunk that holds a logged failing shot of that circuit
  (seeds in the log are cross-checked against the derivation). Inputs: the verifier's sample_shots / to_bits.
- 3 random-seed chunks with uniformly random syndrome and MBUC inputs.
Compared per shot: all qubits, all classical bits, exact phase (read_shot_phase as a 128-bit fixed-point value),
verifier value/phase error flags; per chunk: every per-kind op count (dict(stats)) and the Toffoli count.
Negative controls (must mismatch): engine with sim seed + 1; engine on a copy of the circuit with one CX turned
into a CZ.
"""
import argparse, json, os, sys, time
import numpy as np
import kickmix as km

import kickmix_engine as ke
from kickmix_engine import dqi

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--kmb', action='append', required=True)
ap.add_argument('--failing', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                                                  'dqi_logged_failing_shots.jsonl'),
                help='JSONL of logged failing shots (default: the records for the two M3 circuits, from Kaggle salt checks)')
ap.add_argument('--scratch', required=True, help='directory for the mutated-circuit copy (created if needed; '
                'the copy is as large as the circuit, delete it afterwards)')
ap.add_argument('--empty-chunks', default='0,1,2,200,399')
a = ap.parse_args()
if os.path.exists(a.out):
    sys.exit(f'refusing to overwrite {a.out}')
os.makedirs(a.scratch, exist_ok=True)

failing = [json.loads(l) for l in open(a.failing)]


def kmb_optypes_offset(path):
    with open(path, 'rb') as f:
        h = f.read(16 + 16 + 72)
        nregs = int.from_bytes(h[28:32], 'little')
        num_ops = int.from_bytes(h[32:40], 'little')
        pos = len(h)
        for _ in range(nregs):
            f.seek(pos)
            ty, size = int.from_bytes(f.read(8), 'little'), int.from_bytes(f.read(8), 'little')
            assert ty == 0
            pos += 16 + size
        f.seek(pos)
        assert int.from_bytes(f.read(8), 'little') == 1
        return pos + 16, num_ops


def make_flipped(path, scratch):
    """Copy of `path` with the first CX at or after the middle of the circuit turned into a CZ (same operands)."""
    out = os.path.join(scratch, os.path.basename(path).replace('.kmb', '.flipped_cx_to_cz.kmb'))
    off, num_ops = kmb_optypes_offset(path)
    with open(path, 'rb') as f:
        f.seek(off)
        types = np.frombuffer(f.read(num_ops), np.uint8)
    idx = int(num_ops // 2 + np.flatnonzero(types[num_ops // 2:] == 0x21)[0])
    if not os.path.exists(out):
        with open(path, 'rb') as src, open(out, 'wb') as dst:
            while True:
                b = src.read(1 << 24)
                if not b:
                    break
                dst.write(b)
        with open(out, 'r+b') as dst:
            dst.seek(off + idx)
            dst.write(bytes([0x20]))
    return out, idx


def kickmix_batch(v, circuit, spec, syn_arr, mbuc_arr, anc, allq, allb, syndromes, mbuc, phases, sim_seed):
    """The body of the verifier's run_batch, plus full state reads."""
    sim = km.Simulator(batch_size=256, debug_print=False, counting_operations=True, seed=sim_seed)
    sim.use_same_registers_as(circuit)
    input_syndromes = v.to_bits(syndromes, spec.degree)
    sim.write_across_shots(syn_arr, input_syndromes)
    sim.write_across_shots(mbuc_arr, v.to_bits(mbuc, spec.degree))
    sim.do(circuit)
    phase_flipped = sim.read_phase_flipped_across_shots()
    reg_diff = sim.read_across_shots(syn_arr) ^ input_syndromes
    junk = sim.read_across_shots(anc)
    val_err = reg_diff.any(axis=0) | junk.any(axis=0)
    phase_err = (phase_flipped ^ phases) & ~val_err
    stats = sim.operation_statistics()
    V = []
    for i in range(256):
        h = sim.read_shot_phase(i)
        V.append(int(h * (1 << 127)) % (1 << 128))
    return dict(q=sim.read_across_shots(allq), b=sim.read_across_shots(allb), phase_flipped=np.asarray(phase_flipped),
                V=V, val_err=val_err, phase_err=phase_err, counts=dict(stats), toffoli=int(stats.num_toffoli_like))


def compare(ref, eng, i, prog, phases):
    """Mismatch counts between a kickmix batch record and engine batch i."""
    eq = ke.words_to_bools(eng['dump_q'][i])
    eb = ke.words_to_bools(eng['dump_bits'][i][:prog.num_bits])
    bit_map = np.asarray(prog.bit_map, dtype=np.int64)
    untouched = np.ones(ref['b'].shape[0], bool)
    untouched[bit_map] = False
    ev = ke.words_to_bools(eng['val_err'][i])
    eph = ke.words_to_bools(eng['phase'][i])
    eperr = (eph ^ phases) & ~ev
    eV = [int(x) << 127 for x in eph]
    ec = {k: int(c) for k, c in zip(ke.KINDS, eng['kind_counts'][i]) if c}
    return dict(
        qubit_bits=int((eq != ref['q']).sum()),
        classical_bits=int((eb != ref['b'][bit_map]).sum()),
        untouched_bits_nonzero=int(ref['b'][untouched].sum()),
        phase_flipped_shots=int((eph != ref['phase_flipped']).sum()),
        phase_exact_shots=int(sum(x != y for x, y in zip(eV, ref['V']))),
        value_error_shots=int((ev != ref['val_err']).sum()),
        phase_error_shots=int((eperr != ref['phase_err']).sum()),
        kind_counts=int(ec != ref['counts']),
        toffoli=int(int(eng['toffoli_count'][i]) != ref['toffoli']),
    ), dict(val=int(ev.sum()), phase=int(eperr.sum()), val_shots=np.flatnonzero(ev).tolist(),
            phase_shots=np.flatnonzero(eperr).tolist())


results = dict(circuits={}, started=time.strftime('%Y-%m-%d %H:%M:%S'), kickmix=km.__file__, uptime=os.popen('uptime').read().strip())
all_ok, controls_ok = True, True
for path in a.kmb:
    name = os.path.basename(path)[:12]
    print('==', name, flush=True)
    t0 = time.time()
    task = dqi.DQITask(path).prepare()
    v, spec, prog = task.v, task.spec, task.prog
    circuit = km.Circuit.from_file(path, format='kmb')
    R = results['circuits'][name] = dict(compile_s=round(task.t_prepare, 2))
    # Static checks.
    st = dict(circuit.operation_statistics())
    eng_static = {k: int(c) for k, c in zip(ke.KINDS, prog.static_counts) if c}
    syn_q, mbuc_b, anc_q = v.validate_registers(circuit, spec)
    static = dict(
        hash_equal=v._circuit_sha256(circuit) == task.circuit_hash,
        num_qubits=(circuit.num_qubits, prog.num_qubits), num_bits=(circuit.num_bits, prog.orig_num_bits),
        num_ops=(len(circuit), prog.num_ops), static_counts_equal=st == eng_static,
        syndrome_ids_equal=[x.id if hasattr(x, 'id') else int(str(x)[1:]) for x in syn_q] == task.syn_ids,
        mbuc_ids_equal=[x.id if hasattr(x, 'id') else int(str(x)[1:]) for x in mbuc_b] == task.e_ids,
        compact_bits=prog.num_bits, e_bits_kept=int(len(task.e_keep)), max_stack_depth=prog.max_stack_depth,
        info_tags=prog.num_info_tags, program_mib=round(prog.nbytes / 2**20, 1), stream_words=prog.num_stream_ops,
        cx_runs=prog.num_cx_runs)
    static_ok = (static['hash_equal'] and static['num_qubits'][0] == static['num_qubits'][1]
                 and static['num_bits'][0] == static['num_bits'][1] and static['num_ops'][0] == static['num_ops'][1]
                 and static['static_counts_equal'] and static['syndrome_ids_equal'] and static['mbuc_ids_equal'])
    static['pass'] = bool(static_ok)
    R['static'] = static
    print(static, flush=True)
    all_ok &= static_ok
    syn_arr, mbuc_arr = km.array(syn_q), km.array(mbuc_b)
    allq = km.array([km.q(i) for i in range(circuit.num_qubits)])
    allb = km.array([km.b(i) for i in range(circuit.num_bits)])
    task._set_sampler_state(256)
    # Chunk list.
    chunks = [('', int(c)) for c in a.empty_chunks.split(',')] + [('salt-check-1', 17)]
    logged = [f for f in failing if f['circuit'] == name]
    for f in logged:
        if (f['salt'], f['batch']) not in chunks:
            chunks.append((f['salt'], f['batch']))
    task_cache = {}
    rows = []
    for salt, c in chunks:
        if salt not in task_cache:
            task_cache[salt] = task.tasks(salt)
        t = task_cache[salt][c]
        rng = np.random.default_rng(t[1])
        syndromes, mbuc, phases = v.sample_shots(spec, dqi._S['tables'], rng, 256, dqi._S['wp'])
        ref = kickmix_batch(v, circuit, spec, syn_arr, mbuc_arr, anc_q, allq, allb, syndromes, mbuc, phases, t[2])
        smp = task.sample(t)  # the engine's own sampling path (pack_rows)
        assert (smp[2] == phases).all()
        pack_ok = bool((smp[0] == ke.bools_to_words(v.to_bits(syndromes, spec.degree))).all()
                       and (smp[1] == ke.bools_to_words(v.to_bits(mbuc, spec.degree))[task.e_keep]).all())
        eng = task.run_group([t], [smp], threads=1, dump=True)
        mm, eres = compare(ref, eng, 0, prog, phases)
        ok = pack_ok and all(x == 0 for x in mm.values())
        row = dict(salt=salt, chunk=c, start=t[0], mismatches=mm, pack_equal=pack_ok, pass_=ok,
                   kickmix_val_err=int(ref['val_err'].sum()), kickmix_phase_err=int(ref['phase_err'].sum()), engine=eres,
                   toffoli=ref['toffoli'])
        for f in logged:
            if (f['salt'], f['batch']) == (salt, c):
                row.setdefault('logged', []).append(dict(
                    kind=f['kind'], slot=f['slot'], seeds_match=(str(t[1]) == f['rng_seed'] and str(t[2]) == f['sim_seed']),
                    engine_flags=(f['slot'] in eres['val_shots'], f['slot'] in eres['phase_shots']),
                    kickmix_flags=(bool(ref['val_err'][f['slot']]), bool(ref['phase_err'][f['slot']]))))
                lg = row['logged'][-1]
                want = (True, False) if f['kind'] == 'value' else (False, True)
                if not (lg['seeds_match'] and lg['engine_flags'] == want and lg['kickmix_flags'] == want):
                    row['pass_'] = ok = False
        rows.append(row)
        all_ok &= ok
        print(salt or "''", c, 'ok' if ok else 'MISMATCH', mm, row.get('logged', ''), flush=True)
    # Random-seed chunks.
    rr = np.random.default_rng(20261028)
    for j in range(3):
        sim_seed = int.from_bytes(rr.bytes(32), 'little')
        syndromes = rr.integers(0, 1 << spec.degree, size=(256, spec.n))
        mbuc = rr.integers(0, 1 << spec.degree, size=(256, spec.m))
        phases = rr.integers(0, 2, size=256).astype(bool)
        if j == 0:
            syndromes[:16] = 0  # a few all-zero-syndrome shots
        ref = kickmix_batch(v, circuit, spec, syn_arr, mbuc_arr, anc_q, allq, allb, syndromes, mbuc, phases, sim_seed)
        smp = (dqi.pack_rows(syndromes, spec.degree), dqi.pack_rows(mbuc, spec.degree)[task.e_keep], phases)
        eng = task.run_group([(0, 0, sim_seed)], [smp], threads=1, dump=True)
        mm, eres = compare(ref, eng, 0, prog, phases)
        ok = all(x == 0 for x in mm.values())
        rows.append(dict(salt=None, chunk=f'random{j}', mismatches=mm, pass_=ok, kickmix_val_err=int(ref['val_err'].sum()),
                         kickmix_phase_err=int(ref['phase_err'].sum()), toffoli=ref['toffoli']))
        all_ok &= ok
        print('random', j, 'ok' if ok else 'MISMATCH', mm, int(ref['val_err'].sum()), flush=True)
    R['chunks'] = rows
    # Negative controls on the first logged failing chunk (else chunk 0).
    salt, c = next(((f['salt'], f['batch']) for f in logged), ('', 0))
    t = task_cache[salt][c]
    rng = np.random.default_rng(t[1])
    syndromes, mbuc, phases = v.sample_shots(spec, dqi._S['tables'], rng, 256, dqi._S['wp'])
    ref = kickmix_batch(v, circuit, spec, syn_arr, mbuc_arr, anc_q, allq, allb, syndromes, mbuc, phases, t[2])
    smp = task.sample(t)
    eng = task.run_group([(t[0], t[1], t[2] + 1)], [smp], threads=1, dump=True)
    mm_seed, _ = compare(ref, eng, 0, prog, phases)
    fpath, fidx = make_flipped(path, a.scratch)
    ftask = dqi.DQITask(fpath).prepare()
    eng = ftask.run_group([t], [smp], threads=1, dump=True)
    mm_flip, _ = compare(ref, eng, 0, ftask.prog, phases)
    del ftask
    c_ok = any(x for x in mm_seed.values()) and any(x for x in mm_flip.values())
    R['controls'] = dict(chunk=(salt, c), wrong_seed=mm_seed, flipped_op_index=fidx, flipped_cx_to_cz=mm_flip,
                         both_mismatch=bool(c_ok))
    controls_ok &= c_ok
    print('controls', R['controls'], flush=True)
    R['seconds'] = round(time.time() - t0, 1)
    del circuit
results['kernel'] = ke.kernel_info()  # which simulator kernel these results are for
results['summary'] = dict(
    parity_pass=bool(all_ok), controls_mismatched=bool(controls_ok),
    verifier_chunks={k: sum(1 for r in c['chunks'] if r['salt'] is not None) for k, c in results['circuits'].items()},
    random_chunks={k: sum(1 for r in c['chunks'] if r['salt'] is None) for k, c in results['circuits'].items()},
    compared_shots=256 * sum(len(c['chunks']) for c in results['circuits'].values()))
print(results['summary'])
json.dump(results, open(a.out, 'w'), indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
sys.exit(0 if (all_ok and controls_ok) else 1)
