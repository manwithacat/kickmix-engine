//! Integration tests on a synthetic signed-gauge word table (no external data). Set QROM_PLAN_WORDS and
//! QROM_PLAN_PLANS to run the real-data parity checks (W21 words and the HW-13 text plans).

use std::path::PathBuf;

use qrom_plan::bits::B525;
use qrom_plan::flips::{apply, changes_coded_status, loaders_of, Flip, LoaderModel, Recipe};
use qrom_plan::loader::{Words, N525};
use qrom_plan::plan::plan_once;
use qrom_plan::problem::{default_groups, Problem};
use qrom_plan::rng::Rng;
use qrom_plan::textio::read_plan;

/// 1,363 addresses x 15 word columns (three loaders) of random signed-gauge words; columns 5..15 get an affine bit 0
/// on every full block, so loader 0 is generic, loader 1 (B_1 xor B_0) is not coded and loader 2 (B_2 xor B_1) has
/// the coded structure of the real loaders 4..14.
fn synthetic_words(seed: u64) -> Words {
    let rows = 1363;
    let cols = 15;
    let mut rng = Rng::new(seed);
    let mut data = vec![0u8; rows * cols];
    for a in 0..rows {
        for j in 0..cols {
            let mut v = (rng.next_u64() & 0x7f) as u8;
            if (v >> 6) & 1 == 1 {
                v |= 0x80;
            }
            data[a * cols + j] = v;
        }
    }
    // coded columns: bit 0 = c ^ parity(m & a_low) per block (random affine function per block)
    for j in 5..cols {
        let mut b0 = 0;
        while b0 < rows {
            let m = (rng.next_u64() & 15) as usize;
            let c = (rng.next_u64() & 1) as u8;
            for a in 0..16 {
                if b0 + a >= rows {
                    break;
                }
                let bit = c ^ (((m & a).count_ones() & 1) as u8);
                let idx = (b0 + a) * cols + j;
                data[idx] = (data[idx] & !1) | bit;
            }
            b0 += 16;
        }
    }
    Words { rows, cols, data }
}

#[test]
fn synthetic_loaders_have_the_expected_structure() {
    let w = synthetic_words(1);
    let coded = w.coded_columns();
    assert_eq!(&coded[..5], &[false; 5]);
    assert_eq!(&coded[5..], &[true; 10]);
    let p0 = Problem::from_words(&w, 0, &default_groups()).unwrap();
    let p1 = Problem::from_words(&w, 2, &default_groups()).unwrap();
    assert!(Problem::from_words(&w, 1, &default_groups()).unwrap().coded.is_empty());
    assert_eq!(p0.h, 85);
    assert_eq!(p0.rank(), 85);
    assert_eq!(p0.annihilator().len(), N525 - 85);
    assert_eq!(p0.single_slot_relations().len(), 0);
    assert_eq!(p1.coded, vec![0, 1, 2, 3, 4]);
    assert_eq!(p1.single_slot_relations().len(), 55);
    let (ok, spaces) = p0.feasible(&[]);
    assert!(ok);
    assert_eq!(p0.span_dim(&spaces), N525);
}

#[test]
fn structured_plan_verifies_and_terms_reconstruct() {
    let w = synthetic_words(2);
    let p = Problem::from_words(&w, 2, &default_groups()).unwrap();
    let (rels, ok, spaces) = p.structured_draw(85, 3);
    assert!(ok, "85 relations should stay span-feasible on a coded loader");
    assert_eq!(rels.len(), 85);
    let (pairing, _before, _d) = plan_once(&p, &rels, &spaces, 3, 12, 24, 0.0).expect("pairing");
    let terms = p.terms(&pairing).unwrap();
    let check = p.check_plan(&rels, &terms);
    assert!(check.ok(), "{check:?}");
    assert_eq!(check.toffolis, 89 + terms.len());
    assert_eq!(2 * check.pairs + check.singles, N525 - rels.len());
}

#[test]
fn block_plan_with_repair_on_generic_loader() {
    let w = synthetic_words(3);
    let p = Problem::from_words(&w, 0, &default_groups()).unwrap();
    // the HW-13 loader-0 design: five disjoint within-phase triples under Lera's layout (slots = low monomial - 1)
    let design = vec![vec![0, 5, 12], vec![1, 3, 14], vec![2, 6, 10], vec![4, 7, 13], vec![8, 9, 11]];
    let phases = &p.phases;
    for t in &design {
        assert!(phases.iter().any(|ph| t.iter().all(|s| ph.contains(s))), "triple {t:?} outside every phase");
    }
    let (rels, ok, spaces) = p.block_draw(&design, 8, 5);
    assert!(ok);
    assert_eq!(rels.len(), 40);
    let (pairing, before, after) = plan_once(&p, &rels, &spaces, 5, 12, 24, 20.0).expect("pairing");
    assert!(after <= before as isize);
    let terms = p.terms(&pairing).unwrap();
    assert!(p.check_plan(&rels, &terms).ok());
}

#[test]
fn incremental_equals_full_on_random_flips() {
    let w = synthetic_words(4);
    let coded = w.coded_columns();
    let groups = default_groups();
    let mut models = Vec::new();
    for k in 0..3 {
        let p = Problem::from_words(&w, k, &groups).unwrap();
        let count = if k == 2 { 85 } else { 40 };
        let (rels, ok, spaces) = p.structured_draw(count, 7 + k as u64);
        assert!(ok);
        let (pairing, _, _) = plan_once(&p, &rels, &spaces, 1, 8, 24, 0.0).unwrap();
        let t = pairing.toffolis();
        models.push(LoaderModel::new(p, rels, t, Recipe::default()));
    }
    let mut rng = Rng::new(11);
    let mut checked = 0;
    let mut supply_changes = 0;
    for _ in 0..300 {
        let f = Flip { addr: rng.below(w.rows), col: rng.below(w.cols), bit: rng.below(7) as u8 };
        if changes_coded_status(&w, &coded, &f) {
            continue;
        }
        let w2 = apply(&w, &f);
        let coded2 = w2.coded_columns();
        assert_eq!(coded, coded2);
        for k in loaders_of(f.col, 3) {
            let a = models[k].incremental(&f);
            let b = models[k].full(&w2, &coded2).unwrap();
            assert!(b.rebase_ok, "rebase must hold for a single truth change: {f:?}");
            assert_eq!(a.broken, b.broken, "{f:?}");
            assert_eq!(a.relations_after, b.relations_after);
            assert_eq!(a.toffolis_fixed_after, b.toffolis_fixed_after);
            assert_eq!(a.patch_exists, b.patch_exists, "{f:?}");
            assert_eq!(a.annihilator_after, b.annihilator_after, "{f:?}");
            assert_eq!(a.nullities_after, b.nullities_after, "{f:?}");
            if a.nullities_after != models[k].nullities() {
                supply_changes += 1;
            }
            checked += 1;
        }
    }
    assert!(checked >= 400, "checked {checked}");
    assert!(supply_changes > 0, "some flip should move a kernel dimension");
}

#[test]
fn coded_bit0_flip_changes_status_and_falls_back() {
    let w = synthetic_words(5);
    let coded = w.coded_columns();
    let f = Flip { addr: 100, col: 7, bit: 0 };
    assert!(changes_coded_status(&w, &coded, &f));
    let f2 = Flip { addr: 100, col: 7, bit: 3 };
    assert!(!changes_coded_status(&w, &coded, &f2));
    let f3 = Flip { addr: 100, col: 2, bit: 0 };
    assert!(!changes_coded_status(&w, &coded, &f3));
    let w2 = apply(&w, &f);
    assert!(!w2.coded_columns()[7]);
}

#[test]
fn rank_one_update_matches_direct_rank() {
    use qrom_plan::bits::B128;
    use qrom_plan::flips::{d_vector, sub_cols, RankOne};
    use qrom_plan::gf2::rank;
    use qrom_plan::problem::put35;
    let w = synthetic_words(6);
    let p = Problem::from_words(&w, 2, &default_groups()).unwrap();
    let subs: Vec<Vec<usize>> = vec![vec![0], vec![2, 6], vec![2, 6, 10], vec![4, 8, 14]];
    let mut rng = Rng::new(3);
    for sub in &subs {
        let solver = RankOne::<2>::new(&p.rows, sub_cols(sub), p.ac.len());
        for _ in 0..200 {
            let a_hi = rng.below(86);
            let a_lo = rng.below(16);
            let o = rng.below(35);
            let d = d_vector(a_lo, o);
            let dch = Problem::chunks(&d);
            let mut ds = B128::ZERO;
            for (j, &s) in sub.iter().enumerate() {
                put35(&mut ds.0, j * 35, dch[s]);
            }
            // direct: rows' = rows + u (x) d
            let u = qrom_plan::flips::u_vector(a_hi, p.rows.len());
            let rows2: Vec<B525> = p.rows.iter().enumerate().map(|(h, r)| if u.get(h) { *r ^ d } else { *r }).collect();
            let cols = sub_cols(sub);
            let restricted: Vec<B128> = rows2
                .iter()
                .map(|r| {
                    let mut v = B128::ZERO;
                    for (ci, &c) in cols.iter().enumerate() {
                        if r.get(c) {
                            v.set(ci);
                        }
                    }
                    v
                })
                .collect();
            let direct = cols.len() - rank(&restricted);
            assert_eq!(solver.nullity_after(ds, a_hi), direct, "sub {sub:?} a_hi {a_hi} a_lo {a_lo} o {o}");
        }
    }
}

#[test]
fn real_data_parity_when_available() {
    let (Ok(words), Ok(plans)) = (std::env::var("QROM_PLAN_WORDS"), std::env::var("QROM_PLAN_PLANS")) else {
        eprintln!("QROM_PLAN_WORDS / QROM_PLAN_PLANS not set; skipping");
        return;
    };
    let w = qrom_plan::load_words(&PathBuf::from(words)).unwrap();
    for k in 0..w.loaders() {
        let pf = read_plan(&PathBuf::from(&plans).join(format!("k{k}.txt"))).unwrap();
        let p = Problem::from_words(&w, k, &pf.groups).unwrap();
        let c = p.check_plan(&pf.relations, &pf.terms);
        assert!(c.ok(), "k{k}: {c:?}");
        assert_eq!(Some(c.toffolis), pf.toffolis, "k{k}");
    }
}
