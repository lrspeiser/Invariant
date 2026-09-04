"""Run the compiler over the 3,123 candidates the tournament searched.

Reads ONLY `../tournament/tournament.json`, which is a record of a previous
lane's OWN candidate list and verdicts.  It contains no observational data:
the fields used here are the grammar coordinates (base, structure, invariant,
form, exponent, invariant scale, fitted amplitude) and the recorded pass/fail
flags.  No catalogue, no SPARC, no KiDS, no wide binaries.

Writes `retrospective.json`.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import compiler as C                                            # noqa: E402

TOURNAMENT = os.path.abspath(os.path.join(HERE, "..", "tournament",
                                          "tournament.json"))


def load_records():
    with open(TOURNAMENT, "r", encoding="utf-8") as fh:
        d = json.load(fh)
    recs = d["records"]
    # silent-extraction guard: assert the row count and the columns
    assert d["n_candidates"] == len(recs) == 3123, \
        f"expected 3,123 records, got {len(recs)} / {d.get('n_candidates')}"
    need = {"name", "base", "struct", "inv", "form", "m", "I0", "survives"}
    missing = need - set(recs[0])
    assert not missing, f"tournament record is missing columns: {missing}"
    return d, recs


def main(cheap: bool = True):
    d, recs = load_records()
    cands = [C.from_tournament_record(r) for r in recs]
    print(f"loaded {len(cands)} candidates from {TOURNAMENT}")
    print("grammar:", Counter(c.struct for c in cands))

    t0 = time.perf_counter()
    results = C.check_many(cands, cheap=cheap)
    wall = time.perf_counter() - t0
    print(f"compiled {len(cands)} candidates in {wall:.2f} s "
          f"({len(cands) / wall:,.0f}/s wall, caches cold)")

    per_gate = {g: 0 for g in C.GATES}
    unique = {g: 0 for g in C.GATES}
    hard_gates = [g for g in C.GATES if g not in C.FLAG_ONLY]
    rejected, flagged = 0, 0
    for r in results:
        for g in C.GATES:
            if not r[g][0]:
                per_gate[g] += 1
        fails = [g for g in hard_gates if not r[g][0]]
        if fails:
            rejected += 1
        if len(fails) == 1:
            unique[fails[0]] += 1
        if r["_flags"]:
            flagged += 1

    # ---------------------------------------------------- THE TAXONOMY
    # "97.2% rejected" conflates four verdicts that are not scientifically
    # equivalent.  Re-partition every rejection explicitly, and cross-tabulate
    # each gate's contribution to each bin.
    bin_counts = Counter()
    defect_counts = Counter()
    gate_bin = defaultdict(Counter)          # gate -> bin -> n
    bin_by_struct = defaultdict(Counter)     # struct -> bin -> n
    for r, c in zip(results, cands):
        t = r["_taxonomy"]
        bin_counts[t["primary"]] += 1
        bin_by_struct[c.struct][t["primary"]] += 1
        for d in t["defects"]:
            defect_counts[d["code"]] += 1
            gate_bin[d["gate"]][d["bin"]] += 1
    n = len(cands)
    taxonomy = dict(
        bins=list(C.TAXONOMY_BINS),
        doc=C.TAXONOMY_DOC,
        severity_order=list(C.TAXONOMY_SEVERITY),
        primary_counts={b: int(bin_counts.get(b, 0)) for b in C.TAXONOMY_BINS},
        primary_percent={b: round(100.0 * bin_counts.get(b, 0) / n, 2)
                         for b in C.TAXONOMY_BINS},
        per_gate_contribution_to_bin={g: dict(v) for g, v in gate_bin.items()},
        defect_counts=dict(defect_counts),
        primary_by_structure={k: dict(v) for k, v in bin_by_struct.items()},
        headline=("; ".join(
            f"{100.0 * bin_counts.get(b, 0) / n:.1f}% {b.replace('_', ' ')}"
            for b in C.TAXONOMY_BINS if bin_counts.get(b, 0))),
        note=("the single 'rejected' figure is retained above for continuity "
              "with Run AM, but it is NOT the headline: it sums bins whose "
              "scientific content differs. A mathematically inconsistent law "
              "is dead; a law that is incomplete AS WRITTEN has a named "
              "repair; a convention-dependent law needs a declaration, not a "
              "new theory; a non-identifiable law needs a different "
              "experiment."),
    )

    # what the other three gates achieve without the blanket action gate
    without4 = sum(1 for r in results
                   if any(not r[g][0] for g in hard_gates if g != C.GATE4))

    surv_idx = [i for i, r in enumerate(recs) if r["survives"]]
    survivors = []
    for i in surv_idx:
        r = results[i]
        survivors.append(dict(
            name=r["_name"], verdict=r["_verdict"], failed=r["_failed"],
            flags=r["_flags"],
            taxonomy=r["_taxonomy"]["primary"],
            defects=[d["code"] for d in r["_taxonomy"]["defects"]],
            gate1=r["gate1_constant_K"][2],
            gate2=r["gate2_potential_gauge"][2],
            gate3=r["gate3_coarse_graining"][2],
            gate4=r[C.GATE4][2],
            gate1_escapes=r["gate1_constant_K"][1].get("escapes"),
            gate1_joint_resid_dex=r["gate1_constant_K"][1].get(
                "joint_resid_dex"),
            gate4_asymmetry=r[C.GATE4][1]["asymmetry"]))

    # cross-tabulate the compiler's verdict against the tournament's own
    tab = Counter()
    for r, rec in zip(results, recs):
        tab[(r["_verdict"], bool(rec["survives"]))] += 1

    by_struct = defaultdict(lambda: Counter())
    for r, c in zip(results, cands):
        for g in hard_gates:
            if not r[g][0]:
                by_struct[c.struct][g] += 1
        by_struct[c.struct]["n"] += 1
    by_inv = defaultdict(lambda: Counter())
    for r, c in zip(results, cands):
        for g in hard_gates:
            if not r[g][0]:
                by_inv[c.inv][g] += 1
        by_inv[c.inv]["n"] += 1

    admitted = [r["_name"] for r in results if r["_verdict"] == "ADMIT"]
    # An amplitude fitted to exactly zero is the base law under another name.
    # The screen lane found the same thing: all 450 of its survivors sat at
    # s_0 = s_T = 0, i.e. the network switched off.
    byname = {rec["name"]: rec for rec in recs}
    amp_zero = [n for n in admitted if not byname[n].get("A")]
    base_named = [n for n in admitted if n.startswith("BASE_")]
    live = [n for n in admitted
            if n not in amp_zero and n not in base_named]
    live_detail = []
    for n in live:
        rec = byname[n]
        live_detail.append(dict(name=n, struct=rec["struct"], inv=rec["inv"],
                                form=rec["form"], m=rec["m"], A=rec.get("A"),
                                tournament_survives=bool(rec["survives"]),
                                tournament_failed=rec.get("failed")))

    tp = C.throughput(cands)

    # ---- VERDICT INVARIANCE
    # The REPORT_v2 work renames a gate, adds a model-class scope, adds an
    # u-space integrability channel and re-partitions every rejection.  None
    # of that is allowed to change a VERDICT on the searched grammar.  The
    # reference numbers below were produced by running the COMMITTED
    # pre-REPORT_v2 code against this same `tournament.json` (which was itself
    # re-run after Run AM: 26 survivors now, 18 then, which is why these
    # differ from the numbers printed in REPORT.md).
    BASELINE = dict(rejected_total=3032, admitted_total=91,
                    n_tournament_survivors=26, rejected_without_gate4=1702,
                    per_gate_failures={"gate1_constant_K": 150,
                                       "gate2_potential_gauge": 0,
                                       "gate3_coarse_graining": 1560,
                                       C.GATE4: 2980},
                    unique_kills={"gate1_constant_K": 52,
                                  "gate2_potential_gauge": 0,
                                  "gate3_coarse_graining": 0,
                                  C.GATE4: 1330})
    got = dict(rejected_total=rejected,
               admitted_total=len(cands) - rejected,
               n_tournament_survivors=sum(1 for r in recs if r["survives"]),
               rejected_without_gate4=without4,
               per_gate_failures=dict(per_gate), unique_kills=dict(unique))
    verdict_invariance = dict(
        baseline=BASELINE, measured=got,
        identical=bool(got == BASELINE),
        note="the committed pre-REPORT_v2 compiler, run against this same "
             "tournament.json, gives exactly these numbers. REPORT_v2 changes "
             "how rejections are DESCRIBED, not which candidates are "
             "rejected.")
    assert got == BASELINE, ("REPORT_v2 changed a verdict on the searched "
                             "grammar", BASELINE, got)

    # ---- the external-axis basis element (FIX 4), through the same gates
    ext_axis = {}
    for tag, c in C.external_axis_elements().items():
        r = C.check(c, cheap=False)
        ext_axis[tag] = dict(
            verdict=r["_verdict"], failed=r["_failed"],
            taxonomy=r["_taxonomy"]["primary"],
            defects=[dd["code"] for dd in r["_taxonomy"]["defects"]],
            gate1_escapes=r["gate1_constant_K"][1].get("escapes"),
            axis_misalignment_deg=r["gate1_constant_K"][1].get(
                "axis_misalignment_deg"),
            u_space=r[C.GATE4][1].get("u_space"),
            gate1=r["gate1_constant_K"][2], gate3=r["gate3_coarse_graining"][2],
            gate4=r[C.GATE4][2], note=c.note)
    ext_axis["_provenance_census_of_the_searched_grammar"] = dict(
        counts=dict(Counter(C.AXIS_PROVENANCE.get(c.struct, ("none", ""))[0]
                            for c in cands)),
        note="Run AO: network 1,560 / source 780 / isotropic 783 / EXTERNAL 0. "
             "The 2-D shear phase channel was pointed at a hypothesis the "
             "grammar could not express. This is a GRAMMAR COMPLETENESS FIX; "
             "NO observational claim is attached to the new element, and none "
             "can be: Run AO's 95% exclusion for it sits at an ellipticity of "
             "2.11, above the geometric maximum of 1.")

    out = dict(
        source=TOURNAMENT,
        source_generated_utc=d.get("generated_utc"),
        n_candidates=len(cands),
        taxonomy=taxonomy,
        external_axis_element=ext_axis,
        external_controls=C.run_external_controls(cheap=False),
        curl_identity={row: {k: v for k, v in C.curl_identity(row).items()
                             if k != "fd_convergence"}
                       for row in C.CURL_ROWS},
        curl_spherical_control=C.curl_spherical_control(),
        action_first_scoping=C.action_first_scoping(taxonomy),
        verdict_invariance=verdict_invariance,
        gate_renamed=dict(frm=C.GATE4_LEGACY, to=C.GATE4,
                          title=C.GATE4_TITLE, scope=C.GATE4_SCOPE),
        wall_seconds_cold=wall,
        throughput=tp,
        per_gate_failures=per_gate,
        unique_kills=unique,
        rejected_total=rejected,
        rejected_without_gate4=without4,
        flagged_convention_dependent=flagged,
        admitted_total=len(cands) - rejected,
        admitted_names=admitted,
        admitted_base_laws=base_named,
        admitted_amplitude_fitted_to_zero=amp_zero,
        admitted_with_a_live_response=live_detail,
        n_admitted_with_a_live_response=len(live),
        crosstab_compiler_vs_tournament={f"{k[0]}|tournament_survivor={k[1]}": v
                                         for k, v in tab.items()},
        by_structure={k: dict(v) for k, v in by_struct.items()},
        by_invariant={k: dict(v) for k, v in by_inv.items()},
        tournament_survivors=survivors,
        n_tournament_survivors=len(survivors),
        n_survivors_rejected=sum(1 for s in survivors
                                 if s["verdict"] == "REJECT"),
        n_survivors_flagged=sum(1 for s in survivors if s["flags"]),
        data_statement=C.DATA_STATEMENT,
    )
    with open(os.path.join(HERE, "retrospective.json"), "w",
              newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)

    print()
    print("TAXONOMY (the headline; the single rejection rate is not)")
    for b in C.TAXONOMY_BINS:
        k = bin_counts.get(b, 0)
        if k:
            print(f"  {b:<40} {k:>5}  {100.0 * k / len(cands):5.1f}%")
    print("  per-gate contribution to each bin:")
    for g, v in gate_bin.items():
        print(f"    {g:<40} " + ", ".join(f"{b}={c}" for b, c in v.items()))
    print()
    print(f"VERDICT INVARIANCE vs the committed pre-REPORT_v2 compiler on the "
          f"same tournament.json: "
          f"{'IDENTICAL' if verdict_invariance['identical'] else 'CHANGED'}")
    print()
    print("EXTERNAL-AXIS ELEMENT K = exp[f0 I + f_E e_ext e_ext^T]")
    for tag, v in ext_axis.items():
        if tag.startswith("_"):
            continue
        print(f"  {tag:<26} {v['verdict']:<14} {v['taxonomy']}")
    print()
    print(f"REJECTED {rejected} / {len(cands)} "
          f"({100 * rejected / len(cands):.1f}%) before any data")
    print(f"  ADMITTED {len(admitted)}: {len(base_named)} named base laws, "
          f"{len(amp_zero)} whose fitted amplitude is exactly zero (the base "
          f"law under another name), {len(live)} with a live response")
    print(f"  rejected by gates 1+3 alone (no action gate): {without4}")
    print(f"  flagged convention-dependent: {flagged}")
    for g in C.GATES:
        print(f"  {g:<28} kills alone {per_gate[g]:>5}   "
              f"unique {unique.get(g, 0):>4}")
    print()
    print(f"of the {len(survivors)} tournament survivors, "
          f"{out['n_survivors_rejected']} are REJECTED and "
          f"{out['n_survivors_flagged']} are flagged")
    for s in survivors:
        print(f"  {s['name']:<52} {s['verdict']:<7} {s['failed']}")
    print()
    print("wrote retrospective.json")
    return out


if __name__ == "__main__":
    main(cheap="--full" not in sys.argv)
