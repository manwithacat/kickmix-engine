"""Randomized differential tests: kickmix_analysis GF(2) objects vs the Python reference (card E2).

Reference: recover.Basis / rank / nullspace and planner_shared.intersect / decompose from an UNPATCHED copy of
tools/dfthc-refit (pass its directory). One "case" = one operation on an evolving basis (add, reduce, solve,
decompose, contains, truncate-free reset), with vectors drawn half from the current span (dependent) and half random,
so every comparison checks the pivot choice, the reduced vector and the combination mask. Plus nullspace, rank,
intersect and GroupSet cases (fewer: each is a whole-matrix operation).

    python gf2_property.py REF_DFTHC_REFIT_DIR OUT.json [--cases 100000] [--widths 35,525,893] [--seed 20261105]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out")
    ap.add_argument("--cases", type=int, default=100000)
    ap.add_argument("--matrix-cases", type=int, default=2000)
    ap.add_argument("--group-cases", type=int, default=2000)
    ap.add_argument("--widths", default="35,525,893")
    ap.add_argument("--seed", type=int, default=20261105)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(a.ref).resolve()))
    import recover as R
    import planner_shared as PS
    import kickmix_analysis as ka
    assert "dfthc-refit" in R.__file__ and not hasattr(R, "make_basis"), "reference must be the unpatched copy"
    report = dict(reference=str(Path(R.__file__).resolve()), kickmix_analysis=ka.__file__, seed=a.seed, widths={})
    fails = 0
    for width in [int(x) for x in a.widths.split(",")]:
        rng = random.Random(a.seed + width)
        t0 = time.time()
        st = dict(cases=0, add=0, add_dep=0, reduce=0, solve=0, solve_none=0, decompose=0, contains=0, resets=0,
                  mismatches=0, max_rank=0)

        def rvec(B, low_rank_pool):
            k = rng.random()
            if k < 0.45 and len(B):
                v = 0
                for x in B.vectors:
                    if rng.getrandbits(1):
                        v ^= x
                return v
            if k < 0.6 and low_rank_pool:
                v = 0
                for x in low_rank_pool:
                    if rng.getrandbits(1):
                        v ^= x
                return v
            nb = rng.choice([width, width, rng.randrange(1, width + 1)])
            v = rng.getrandbits(nb)
            if rng.random() < 0.2:      # sparse
                v = 0
                for _ in range(rng.randrange(1, 6)):
                    v |= 1 << rng.randrange(width)
            return v

        P, Q = R.Basis(), ka.Gf2Basis(width)
        pool = [rng.getrandbits(width) for _ in range(rng.randrange(1, 12))]
        target = rng.randrange(1, width + 8)
        for _ in range(a.cases):
            st["cases"] += 1
            op = rng.random()
            v = rvec(P, pool)
            if op < 0.4:
                x, y = P.add(v), Q.add(v)
                st["add"] += 1
                st["add_dep"] += not x
                ok = x == y
            elif op < 0.6:
                ok = P.reduce(v) == Q.reduce(v)
                st["reduce"] += 1
            elif op < 0.8:
                x = P.solve(v)
                ok = x == Q.solve(v)
                st["solve"] += 1
                st["solve_none"] += x is None
            elif op < 0.95:
                ok = PS.decompose(P, v) == Q.decompose(v)
                st["decompose"] += 1
            else:
                ok = (P.solve(v) is not None) == Q.contains(v)
                st["contains"] += 1
            ok = ok and len(P) == len(Q)
            if not ok:
                st["mismatches"] += 1
            st["max_rank"] = max(st["max_rank"], len(P))
            if len(P) >= target or rng.random() < 0.0004:
                if P.vectors != Q.vectors:
                    st["mismatches"] += 1
                P, Q = R.Basis(), ka.Gf2Basis(width)
                pool = [rng.getrandbits(width) for _ in range(rng.randrange(1, 12))]
                target = rng.randrange(1, width + 8)
                st["resets"] += 1
        # whole-matrix operations
        mt = dict(nullspace=0, rank=0, intersect=0, mismatches=0)
        for _ in range(a.matrix_cases):
            n = rng.randrange(0, min(width, 120) + 1)
            base = [rng.getrandbits(width) for _ in range(rng.randrange(1, max(2, n + 1)))]
            rows = []
            for _ in range(n):
                if rng.random() < 0.3 and base:
                    v = 0
                    for x in base:
                        if rng.getrandbits(1):
                            v ^= x
                    rows.append(v)
                else:
                    rows.append(rng.getrandbits(rng.randrange(1, width + 1)))
            if R.nullspace(rows, width) != ka.gf2_nullspace(rows, width):
                mt["mismatches"] += 1
            if R.rank(rows) != ka.gf2_rank(rows):
                mt["mismatches"] += 1
            m = rng.randrange(1, min(width, 64) + 1)
            V1 = R.nullspace([rng.getrandbits(m) for _ in range(rng.randrange(0, m))], m)
            V2 = R.nullspace([rng.getrandbits(m) for _ in range(rng.randrange(0, m))], m)
            if PS.intersect(V1, V2, m) != ka.gf2_intersect(V1, V2, m):
                mt["mismatches"] += 1
            mt["nullspace"] += 1
            mt["rank"] += 1
            mt["intersect"] += 1
        # GroupSet against the Python loops of pair_lines / augment_quotient
        gt = dict(cases=0, greedy=0, first_pair=0, first_pair_hits=0, enabled=0, enabled_nonempty=0, quotient=0,
                  quotient_hits=0, mismatches=0)
        for _ in range(a.group_cases):
            gt["cases"] += 1
            nb = rng.choice([1, 2, 5])
            gens = [[rng.getrandbits(width) for _ in range(rng.randrange(2, 10))] for _ in range(nb)]

            def gv(b):
                v = 0
                for x in gens[b]:
                    if rng.getrandbits(1):
                        v ^= x
                return v if rng.random() < 0.85 else rng.getrandbits(width)
            ng = rng.randrange(2, 40)
            ends = []
            for _ in range(ng):
                b1, b2 = rng.randrange(nb), rng.randrange(nb)
                ends.append(((b1, gv(b1)), (b2, gv(b2))))
            gs = ka.GroupSet(nb, width, ends)

            def build(selected):
                bases = [R.Basis() for _ in range(nb)]
                owner = [[] for _ in range(nb)]
                for li in selected:
                    for e, (b, vec) in enumerate(ends[li]):
                        assert bases[b].add(vec)
                        owner[b].append((li, e))
                return bases, owner

            def addable(li, bases):
                (b1, x), (b2, c) = ends[li]
                if b1 != b2:
                    return bases[b1].solve(x) is None and bases[b2].solve(c) is None
                B = bases[b1]
                return B.solve(x) is None and B.solve(c) is None and B.solve(x ^ c) is None
            order = list(range(ng))
            rng.shuffle(order)
            bases, owner = build([])
            sel = []
            for li in order:
                if addable(li, bases):
                    for e, (b, vec) in enumerate(ends[li]):
                        bases[b].add(vec)
                        owner[b].append((li, e))
                    sel.append(li)
            gs.clear()
            ok = gs.greedy(order) == sel
            gt["greedy"] += 1
            if sel:
                bases, owner = build(sel)
                gs.build(sel)
                pos = [{(li, e): i for i, (li, e) in enumerate(owner[b])} for b in range(nb)]
                selset = set(sel)
                rest_ids = [li for li in range(ng) if li not in selset]
                status = {li: [(b, bases[b].solve(vec)) for b, vec in ends[li]] for li in rest_ids}
                gs.compute_status(rest_ids)
                p = rng.choice(sel)
                (bx, _), (by, _) = ends[p]
                px, py = pos[bx][(p, 0)], pos[by][(p, 1)]
                ok = ok and gs.positions(p) == [(bx, px), (by, py)]
                enabled = []
                for li, stt in status.items():
                    good = True
                    for b, coords in stt:
                        if coords is None:
                            continue
                        if (b == bx and coords >> px & 1) or (b == by and coords >> py & 1):
                            continue
                        good = False
                        break
                    if good:
                        enabled.append(li)
                ok = ok and gs.enabled(bx, px, by, py) == enabled
                gt["enabled_nonempty"] += bool(enabled)
                gt["enabled"] += 1
                rest = [x for x in sel if x != p]
                b0, _ = build(rest)
                gs.build(rest)
                cand = enabled + [li for li in rest_ids if li not in enabled][:5]
                rng.shuffle(cand)
                want = None
                for i, l1 in enumerate(cand):
                    if not addable(l1, b0):
                        continue
                    b1_, _ = build(rest + [l1])
                    for j in range(i + 1, len(cand)):
                        if addable(cand[j], b1_):
                            want = (i, j)
                            break
                    if want:
                        break
                ok = ok and gs.first_pair(cand) == want
                gs.build_values(rest)          # value-only rows (planner_shared rebuild loop)
                ok = ok and gs.first_pair(cand) == want
                gt["first_pair_hits"] += want is not None
                gt["first_pair"] += 1
                # quotient images (augment_quotient)
                bases, owner = build(sel)
                gs.build(sel)
                dec = []
                for li in rest_ids:
                    (b1, x), (b2, c) = ends[li]
                    r1, c1 = PS.decompose(bases[b1], x)
                    r2, c2 = PS.decompose(bases[b2], c)
                    dec.append((li, b1, r1, c1, b2, r2, c2))
                gs.compute_decomp(rest_ids)

                def img(b, r, c):
                    xx = r << 2
                    if b == bx and c >> px & 1:
                        xx |= 1
                    if b == by and c >> py & 1:
                        xx |= 2
                    return xx
                cands = []
                for li, b1, r1, c1, b2, r2, c2 in dec:
                    i1, i2 = img(b1, r1, c1), img(b2, r2, c2)
                    if i1 and i2 and (b1 != b2 or i1 != i2):
                        cands.append((li, b1, i1, b2, i2))
                got = gs.quotient_cands(bx, px, by, py)
                ok = ok and got == [c[0] for c in cands]
                perm = list(range(len(cands)))
                rng.shuffle(perm)
                cands = [cands[i] for i in perm]
                cap = rng.choice([2, 5, 40])
                want = None
                for i, (l1, a1, x1, a2, x2) in enumerate(cands[:cap]):
                    for l2, d1, y1, d2, y2 in cands[i + 1:]:
                        if PS._independent([(a1, x1), (a2, x2), (d1, y1), (d2, y2)]):
                            want = (l1, l2)
                            break
                    if want:
                        break
                ok = ok and gs.quotient_pair([c[0] for c in cands], cap) == want
                gt["quotient_hits"] += want is not None
                gt["quotient"] += 1
            if not ok:
                gt["mismatches"] += 1
        fails += st["mismatches"] + mt["mismatches"] + gt["mismatches"]
        report["widths"][str(width)] = dict(basis=st, matrix=mt, groupset=gt, seconds=round(time.time() - t0, 1))
        print(width, report["widths"][str(width)], flush=True)
    report["failures"] = fails
    Path(a.out).write_text(json.dumps(report, indent=1))
    print("failures", fails)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
