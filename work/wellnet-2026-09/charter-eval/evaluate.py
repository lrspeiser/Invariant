"""Evaluate the programme against the charter's own enumerated requirements.

The charter (`C:/Users/henry/dev/invariant-gravity-discovery-charter.md`) contains
three enumerable requirement sets, plus a stated final deliverable.  This module
scores the programme against each, with the run and the number that supports the
verdict, so the answer is auditable rather than asserted.

    MET       demonstrated with evidence
    PARTIAL   demonstrated in part, or on synthetic data only
    NOT_MET   not demonstrated
    BLOCKED   cannot be demonstrated with available data

    python evaluate.py
"""
import io
import json
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CHARTER = r"C:/Users/henry/dev/invariant-gravity-discovery-charter.md"

MET, PARTIAL, NOT_MET, BLOCKED = "MET", "PARTIAL", "NOT_MET", "BLOCKED"

# ---------------- the charter's twelve questions a final law must answer (L25-38)
QUESTIONS = [
    ("What creates gravity?", NOT_MET,
     "Run AZ: every family ever built takes rest mass as its ONLY source; the "
     "source axis sits at 14% coverage (1 of 7 charter options)."),
    ("What is the gravitational state?", PARTIAL,
     "Scalar, symmetric-tensor and graph states built and scored; vector, "
     "antisymmetric, mixed and emergent-geometry states UNREACHABLE (Run AZ)."),
    ("What establishes direction?", PARTIAL,
     "Source, tidal, external and well-network axes constructed; Run BF "
     "recovers an external axis to 11.9 deg (a 45-deg misspecified axis sets "
     "no limit). Run BK: a halo quadrupole locks to the BARYON axis at 20 deg, "
     "a tensor to the EXTERNAL axis at 12 deg, with opposite radial gradients. "
     "Synthetic only."),
    ("Is the response local?", PARTIAL,
     "Point-local, finite-range nonlocal, path-dependent and globally "
     "constrained all built (locality is the best-covered axis at 80%); "
     "past-light-cone never built."),
    ("How does matter move?", PARTIAL,
     "Rotation, vertical motion, dispersion, streams and cluster member motion "
     "all scored; structure formation only at linear order (Run AP)."),
    ("How does light move?", PARTIAL,
     "Deflection, shear and time delays scored (Runs AL, BC); magnification, "
     "arc shapes, polarization and arrival times not."),
    ("Do matter and light see the same geometry?", PARTIAL,
     "Run AL measures the slip as a FITTED closure, not derived; Run AZ: the "
     "relativistic geometry that would derive it is UNREACHABLE. Run BL: the "
     "tensor action PREDICTS matter-light covariance exactly +1 with no slip -- "
     "a prediction, untested on data."),
    ("Why do galaxies and clusters differ?", NOT_MET,
     "Run BC's radius hypothesis was REFUSED by the Stage 4 gate (28.6% out of "
     "support; window choice reproduces 1.62x the effect). Run AX removed the "
     "CLASH leg. No admissible answer stands."),
    ("What happens locally?", PARTIAL,
     "Solar-system and Cassini bounds carried as constraints. Run BL DERIVES "
     "safety for both actions (h -> 0 at high g; the path term is exactly "
     "Newtonian in media denser than rho_*), but a relativistic completion "
     "leaving tensor waves at c is an INVENTED assumption; no binary-pulsar or "
     "GW propagation."),
    ("How does the universe form?", PARTIAL,
     "Run AP: linear growth integrated for every family; the tidal gate is "
     "ANTI-Zel'dovich (sphere/pancake 4.90 at z=999). Nothing past delta ~ 1."),
    ("What creates redshift?", PARTIAL,
     "Both tested mechanisms answered NEGATIVELY: Run AK excludes the "
     "energy-drain half at 90 sigma; Run BI MEASURES the geometric half on "
     "void path length x Planck, |c2/c1| < 0.0678% at 95% with 13.6-21.3 "
     "sigma of power at AK's own bound (excluded 4-6x), within the "
     "conventional distance frame (BJ.4). No positive mechanism; no "
     "no-expansion universe tested."),
    ("What would falsify it?", PARTIAL,
     "Run BL supplies two falsifiers specific BY CONSTRUCTION: a zero-twist "
     "phase lock to the present tidal axis (tensor) and a net-zero-mass bridge "
     "scaling as M_A M_B (path). The bridge lane's CDM attack: positive masses "
     "reproduce only 2-30% of it, 60 cluster masses at 60 deg. Falsifiers for "
     "CANDIDATES, not for an established law; neither tested on data."),
]

# ------------ the charter's twelve promotion criteria for new gravity (L1194-1216)
PROMOTION = [
    ("1. genuine physical dependence absent from the RAR", PARTIAL,
     "Run BL CONSTRUCTS two -- an environment-tidal tensor term and a "
     "segment-vacuum path term -- both compiled admissible with CDM-distinct, "
     "non-anisotropy falsifiers. Run BF's one-class result stands "
     "CONDITIONALLY (BJ.2: given this corpus, amplitudes, geometries, noise). "
     "Not demonstrated on data."),
    ("2. generative", MET,
     "Run BL: both candidate laws are actions with derived field equations; "
     "the tensor solved to first order, the path family's momentum carrier in "
     "closed form with five-body forces closing to 3e-9. Given a scene and "
     "boundary conditions each PRODUCES a state. (The earlier real-data "
     "candidates were fitted ratios and are retired.)"),
    ("3. one global parameter set", MET,
     "Enforced throughout; no per-object gravity parameter has ever been "
     "admitted. Run AL's A=16 was retired for inheriting a fitted closure."),
    ("4. predicts multiple probes", PARTIAL,
     "Run AL scores matter and light in one framework; no candidate predicts "
     "internal member motion AND cluster lensing in the SAME scene."),
    ("5. preserves known limits", MET,
     "Newtonian and high-acceleration recovery is a compiler gate; Run AQ's "
     "regression test asserts it for all three bases."),
    ("6. matches galaxy regularities", MET,
     "The RAR, flat curves and BTFR are carried as constraints; no promoted "
     "candidate has ever been allowed to break them."),
    ("7. improves the cluster problem using ROOT observations", NOT_MET,
     "Run AX: CLASH masses are T4. Run BC: LoCuSS M_WL is T4. Only eFEDS raw "
     "shear is T0, and Run BC's use of it failed the support check."),
    ("8. survives representation changes", MET,
     "Gate 3 plus Run BD's commutation gate; 7 of 8 substitutions refused, and "
     "the member-smoothing verdict is radius-dependent and measured."),
    ("9. survives alternate-universe controls", PARTIAL,
     "Run BF: 0.648 family-wise on a dark-matter universe. Run BK DECOMPOSED "
     "it -- two detector defects and a library accident, not a wall -- and a "
     "signed joint procedure reaches FP 0.002 on CDM at power 0.989. BUT only "
     "at zero halo-filament alignment; it collapses at f_lss = 0.38, and real "
     "haloes align. The network detector is blind (power 0.000). Partial, "
     "with the caveat named."),
    ("10. makes a distinctive SEALED prediction", BLOCKED,
     "Run AW: 23 of 25 datasets are spent; the reserve holds at most five "
     "one-shot evaluations, three in the same regime, and no reserved galaxy "
     "rotation, lensing or local-gravity probe exists."),
    ("11. sparse enough to explain", MET,
     "Every surviving form is a small closed expression; Run AU admits 91 of "
     "3,123 and all live admissions are QUMOND with a redefined nu."),
    ("12. can evolve", PARTIAL,
     "Run AP integrates linear growth for every family and finds all MOND-like "
     "ones fast enough; nothing is validated past delta ~ 1."),
]

# ------------------------------------------- the charter's Stages 0-10 (L~940-1010)
STAGES = [
    ("Stage 0  universe grammar", PARTIAL,
     "Run AZ: 15 axes, 78 axis-values enumerated; BA.4 replaced the Cartesian "
     "count with a typed compatibility graph. Coverage is 21/78 and five axes "
     "have never been varied."),
    ("Stage 1  probabilistic scenes", PARTIAL,
     "Run BD: built and tested 38/38, contract enforced at construction. "
     "SYNTHETIC ONLY -- no real cluster has been round-tripped through it."),
    ("Stage 2  invariant parameter bank", PARTIAL,
     "67 quantities with the full 17-item contract (Run BD); not yet generated "
     "from a real scene at multiple scales."),
    ("Stage 3  prune without opening target data", MET,
     "Runs AM/AU: 3,123 candidates compiled in 31-47 s with the charter's own "
     "five bins, 12/12 external controls, 0 files opened."),
    ("Stage 4  remove redundant information", PARTIAL,
     "Runs BB/BE: the certificate refuses 5/5 historical failures and 5/5 new "
     "mechanisms, 0 false alarms, and REFUSED a fresh result (Run BC) on "
     "support. Downgraded from MET: the exit condition (BE.8) requires "
     "INDEPENDENTLY AUTHORED prospective validation and BE.5 records that it "
     "was self-authored."),
    ("Stage 5  identifiability in alternate universes", MET,
     "Run BF: ten universes, sizing on an untouched null half, and an "
     "equivalence-class map. The answer is negative and that IS the "
     "deliverable the charter asked for."),
    ("Stage 6  search billions of sparse laws", MET,
     "77M laws/sec; exhaustive k<=3 at 1.1e9. Now GATED by the BA standing "
     "rule until one scene is scored end to end."),
    ("Stage 7  solve each law exactly", PARTIAL,
     "Exact solves exist per channel; no candidate has been solved on a full "
     "resolved scene."),
    ("Stage 8  cross-channel frozen prediction", PARTIAL,
     "Run AL freezes a0 on SPARC and predicts cluster shear; Run BC's transfer "
     "was refused. Never same-scene."),
    ("Stage 9  counterfactual source tests", MET,
     "Run BD's commutation gate plus the mass-preserving scrambles; the "
     "directional pair gives R_source = -0.200 against R_external = +1.298."),
    ("Stage 10 untouched confirmation", NOT_MET,
     "Run AW: no confirmation set exists in any meaningful sense. 23 of 25 "
     "datasets spent; KiDS and the wide binaries were scored in round 1."),
]

CORPORA = [
    ("A  local and high-precision gravity", PARTIAL, "carried as bounds only"),
    ("B  resolved isolated galaxies", PARTIAL, "SPARC 1-D curves, not cubes"),
    ("C  field-vs-cluster galaxies", PARTIAL, "SAMI/MaNGA scored; spent"),
    ("D  statistical groups and clusters", MET, "eFEDS+DECADE, 496 systems, T0"),
    ("E  gold resolved clusters", BLOCKED,
     "Run BD: NO cluster satisfies it. Best MACS J1149 at 9/10. Raw shear for "
     "1 of 7, and that one lacks member Sersic fits -- no target has both."),
    ("F  redshift and cosmology", PARTIAL, "Run AK; SN + CMB bounds only"),
]


def block(title, rows):
    print("=" * 78)
    print(title)
    print("=" * 78)
    counts = {}
    for name, status, ev in rows:
        counts[status] = counts.get(status, 0) + 1
        print(f"  {status:<8} {name}")
        for line in [ev[i:i + 66] for i in range(0, len(ev), 66)]:
            print(f"           {line}")
    tot = len(rows)
    print(f"  -> {tot} items: " + ", ".join(
        f"{v} {k}" for k, v in sorted(counts.items())))
    print()
    return counts


def main():
    charter = io.open(CHARTER, encoding="utf-8").read()
    print(f"charter: {CHARTER}")
    print(f"         {len(charter.splitlines())} lines, "
          f"{len(charter)} bytes\n")

    c1 = block("THE TWELVE QUESTIONS A FINAL LAW MUST ANSWER", QUESTIONS)
    c2 = block("THE TWELVE PROMOTION CRITERIA FOR NEW GRAVITY", PROMOTION)
    c3 = block("STAGES 0-10", STAGES)
    c4 = block("CORPORA A-F", CORPORA)

    print("=" * 78)
    print("THE CHARTER'S STATED FINAL DELIVERABLE")
    print("=" * 78)
    print("  'either a new generative field law with distinctive successful")
    print("   predictions, or a precise statement of which broad classes of new")
    print("   gravity have been ruled out and which observation would")
    print("   distinguish the remaining equivalence classes.'")
    print()
    print("  (a) a new generative field law         NOT_MET")
    print("      Run BL constructs two action-derived laws with specific, non-")
    print("      anisotropy falsifiers; NEITHER has a successful untouched")
    print("      prediction (no confirmation set exists, Run AW), the path law")
    print("      confines cluster members x7.6 at identifiable amplitudes, and")
    print("      the tensor's endpoint quadrupole has the wrong sign for the")
    print("      cluster excess. Run BK: the CDM separation is a statement about")
    print("      the halo-alignment prior, not about gravity.")
    print()
    print("  (b) which classes are ruled out, and   MET  (Runs BH, BI, BJ)")
    print("      which observation separates the rest")
    print("      ASSEMBLED in deliverable_b.py, in the four categories that")
    print("      matter (BJ.3): 4 theory FAMILIES ruled out, 7 specific")
    print("      IMPLEMENTATIONS, 3 STATISTICS, 1 NON-IDENTIFYING (spherical")
    print("      blindness, an experimental limit); 1 withdrawn as T4; and")
    print("      6 remaining equivalence classes each with its named separating")
    print("      observation and cost. 2 of the 6 are separable with data that")
    print("      ALREADY EXISTS.")
    print("      This was previously scored PARTIAL on the reasoning that Stage")
    print("      5's map is synthetic. That was WRONG: the charter asks for a")
    print("      STATEMENT of ruled-out classes and distinguishing observations,")
    print("      and that statement rests on real-data eliminations plus")
    print("      mathematics. It was a synthesis gap, not a data gap.")
    print()
    total = {}
    for c in (c1, c2, c3, c4):
        for k, v in c.items():
            total[k] = total.get(k, 0) + v
    n = sum(total.values())
    print("=" * 78)
    print(f"OVERALL over {n} enumerated charter requirements")
    print("=" * 78)
    for k in (MET, PARTIAL, NOT_MET, BLOCKED):
        v = total.get(k, 0)
        print(f"  {k:<8} {v:>3}   {v/n:>5.0%}")
    print()
    print("  VERDICT: the charter's FINAL OUTPUT requirement is SATISFIED via")
    print("  its fallback branch (b). Its primary goal (a), a new generative")
    print("  field law, is NOT met and is not supported by any surviving")
    print("  candidate. The two blockers on (a) -- Corpus E and Stage 10 -- are")
    print("  acquisition problems, not computation problems.")

    doc = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               charter=CHARTER, charter_lines=len(charter.splitlines()),
               questions=QUESTIONS, promotion=PROMOTION, stages=STAGES,
               corpora=CORPORA, totals=total, n_requirements=n,
               deliverable_a=NOT_MET, deliverable_b=MET,
               satisfied="fallback deliverable (b) MET; goal (a) NOT_MET")
    p = os.path.join(HERE, "charter_eval.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
