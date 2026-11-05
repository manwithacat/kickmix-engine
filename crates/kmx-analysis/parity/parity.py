"""E1 parity driver: one (mode, backend, circuit) per process, result JSON written to OUT.

  python -I parity.py iv    {python|rust} SCRIPTS_DIR CIRCUIT.kmb OUT.json
      interval arrays (cond_rule False and True) and live profiles (register_gaps False and True): SHA-256 of the
      exact bytes (int64 / bool, profile cast to int64) plus counts.
  python -I parity.py peak  {python|rust} SCRIPTS_DIR CIRCUIT.kmb OUT.json LIVENESS
      qhdl_peak.analyse(...) JSON (backend='rust' only for the patched module); OUT is the json.dumps text.
  python -I parity.py alloc {python|rust} SCRIPTS_DIR CIRCUIT.kmb OUT.json OUT.kmb [--register-gaps]
      the allocator's main() pipeline (intervals, profile, colour, recolour, patch_qubits) without the independent
      equivalence check (that check is Python on both sides and memory-heavy on DQI); OUT.kmb hashed.
  python -I parity.py time  rust - CIRCUIT.kmb OUT.json
      endpoint: kickmix_analysis.liveness only (intervals + allocator profile + peak), wall time.

SCRIPTS_DIR is the directory whose qhdl_kmb / qhdl_lifetime_allocator / qhdl_peak are imported (the unpatched
competition scripts/ for the python reference; the patched staging copy for rust).
"""
import hashlib
import json
import sys
import time

import numpy as np


def h(a, dt=None):
    a = np.ascontiguousarray(a if dt is None else np.asarray(a).astype(dt))
    return dict(sha256=hashlib.sha256(a.tobytes()).hexdigest(), dtype=str(a.dtype), n=int(a.size))


def main():
    mode, backend, sdir, kmb, out = sys.argv[1:6]
    t0 = time.time()
    if sdir != '-':
        sys.path.insert(0, sdir)
    res = dict(mode=mode, backend=backend, circuit=kmb)
    if mode == 'iv':
        import qhdl_kmb as K
        import qhdl_lifetime_allocator as A
        k = K.read(kmb)
        nops = len(k.op_types)
        regs = A.register_qubits(k)
        for cond in (False, True):
            if backend == 'python':
                is_reset, depth = A.reset_ops(k)
                ops, qs, order, _ = A.references(k)
                iv = A.build_intervals(k, ops, qs, order, is_reset, depth, regs, cond_rule=cond)
                extra = dict(reset_refs=int(iv['reset_ref'].sum()), num_refs=int(len(ops)))
                del ops, qs, order
                for key in ('sq', 'so', 'first', 'last', 'reset_ref', 'start', 'end', 'interval_id'):
                    iv.pop(key)
                profs = {rg: A.live_profile(iv, nops, regs, register_gaps=rg) for rg in (False, True)}
            else:
                import kickmix_analysis as KA
                lv = KA.liveness(kmb, cond_rule=cond, register_gaps=False)
                iv = lv['intervals']
                extra = dict(reset_refs=int(lv['reset_refs']), num_refs=int(lv['num_refs']))
                profs = {False: lv['profile'], True: KA.live_profile(iv, nops, regs, register_gaps=True)}
            r = dict(extra, num_intervals=int(len(iv['s'])))
            for key in ('logical', 's', 'e', 'is_first', 'is_last'):
                r[key] = h(iv[key])
            for rg, p in profs.items():
                r[f'profile_register_gaps_{rg}'] = h(p, np.int64)
                r[f'peak_register_gaps_{rg}'] = int(p.max())
                r[f'ops_at_peak_register_gaps_{rg}'] = int((p == p.max()).sum())
            res[f'cond_rule_{cond}'] = r
            del iv, profs
    elif mode == 'peak':
        import qhdl_peak as P
        liv = sys.argv[6]
        r = P.analyse(kmb, None, liveness=liv) if backend == 'python' else P.analyse(kmb, None, liveness=liv,
                                                                                         backend='rust')
        text = json.dumps(r, indent=1)
        res.update(liveness=liv, json_sha256=hashlib.sha256(text.encode()).hexdigest(), json_bytes=len(text))
        with open(out.replace('.json', '.peak.json'), 'w') as f:
            f.write(text)
    elif mode == 'alloc':
        import qhdl_kmb as K
        import qhdl_lifetime_allocator as A
        out_kmb = sys.argv[6]
        rg = '--register-gaps' in sys.argv[7:]
        k = K.read(kmb)
        regs = A.register_qubits(k)
        nops = len(k.op_types)
        if backend == 'python':
            is_reset, _ = A.reset_ops(k)
            ops, qs, order, _ = A.references(k)
            iv = A.build_intervals(k, ops, qs, order, is_reset)
            prof = A.live_profile(iv, nops, regs, rg)
            col = A.colour(iv, nops, regs, rg)
            streams, new_qs = A.recolour(k, ops, qs, order, iv, col)
            newQ = int(new_qs.max()) + 1 if len(new_qs) else 0
        else:
            import kickmix_analysis as KA
            lv = KA.liveness(kmb, register_gaps=rg)
            iv, prof = lv['intervals'], lv['profile']
            col = A.colour(iv, nops, regs, rg)
            streams = KA.recolour(kmb, iv['logical'], col)
            nz = [int(v.max()) for v in streams.values() if len(v)]
            newQ = max(nz) + 1 if nz else 0
        reg_ids = np.array(sorted(regs), dtype=np.int64)
        newQ = max(newQ, int(reg_ids.max()) + 1 if len(reg_ids) else 0)
        K.patch_qubits(k, out_kmb, streams, {}, newQ)
        res.update(register_gaps=rg, lower_bound=int(prof.max()), num_qubits_out=newQ, col=h(col),
                   output_sha256=A.sha256(out_kmb))
    elif mode == 'time':
        import kickmix_analysis as KA
        t1 = time.time()
        lv = KA.liveness(kmb)
        t2 = time.time()
        p = lv['profile']
        res.update(liveness_call_s=round(t2 - t1, 3), peak=int(lv['peak']), ops_at_peak=int((p == lv['peak']).sum()),
                   num_ops=int(lv['num_ops']), num_refs=int(lv['num_refs']),
                   num_intervals=int(len(lv['intervals']['s'])))
    else:
        raise SystemExit(f'unknown mode {mode}')
    res['seconds'] = round(time.time() - t0, 3)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict)}))


if __name__ == '__main__':
    main()
