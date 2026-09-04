"""bridge.py -- what the metadata is FOR.

    "This metadata is what allows the admissibility compiler to prune candidate
     laws before data fitting."   -- the charter

`prescreen(reads, registry, inventory)` takes the list of ontology quantities a
candidate law consumes and returns a verdict per gate, using ONLY metadata and
the availability inventory.  No data file is opened; no fit is performed.

The eight checks, and which charter Stage-3 bullet each implements:

  S1  UNITS          dimensional consistency.  Also catches the specific
                     error of feeding a dimensionful quantity to log or exp.
  S2  GAUGE          potential-gauge invariance.  A law reading a member of
                     the gauge family is FLAGGED with the measured 0.87 dex
                     spread between defensible rules, never silently accepted.
  S3  FRAME          coordinate and frame invariance.  A frame-fixed or
                     sky-frame quantity entering a scalar law is a bug.
  S4  COARSE         mass-partition and coarse-graining consistency.  A law
                     reading a CATALOGUE_DEPENDENT quantity must additionally
                     pass `commutation.erasure` before it may be evaluated on
                     anything averaged.
  S5  CAUSAL         causal validity.
  S6  IDENTIFIABLE   non-identifiability: does the law read a quantity that is
                     not independently measurable in the test sample?
  S7  RANK           redundancy: does the read set collapse under the exact
                     identities?  A three-variable law standing on two
                     independent directions is not a three-variable discovery.
  S8  AVAILABLE      NEW, and only possible once a scene layer exists: is
                     every quantity the law reads actually OBSERVED for the
                     clusters in the corpus?  The charter's "non-identifiable
                     on the available data ... requires a different
                     experiment" becomes a computable statement rather than a
                     judgement call.

The charter's four-way taxonomy is preserved: a candidate is
`mathematically_inconsistent`, `theory_contaminated`, `non_identifiable`,
`convention_dependent`, `admissible_but_redundant`, or `admissible`.  Only the
first is a rejection; the rest are classifications.  `theory_contaminated` is
the charter's own prohibition made mechanical: a candidate scored against a
convergence map or an NFW-defined radius is being tested against a product of
the theory it is meant to replace.

NO OBSERVATIONAL DATA IS OPENED BY THIS MODULE.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from metadata import Dim, DIMLESS, Quantity, Registry
from registry import EXACT_IDENTITIES

#: the measured spread between the four defensible global potential-boundary
#: rules, in dex, from Run AH.  Recorded here so the flag carries a number.
PHI_RULE_SPREAD_DEX = 0.87
PHI_GATE_MARGIN_DEX = 0.9

GATES = ("S1_units", "S2_gauge", "S3_frame", "S4_coarse", "S5_causal",
         "S6_identifiable", "S7_rank", "S8_available")

#: gates that CLASSIFY rather than reject
FLAG_ONLY = {"S2_gauge", "S6_identifiable", "S7_rank", "S8_available"}


def _q(reg: Registry, names: Sequence[str]) -> List[Quantity]:
    return [reg[n] for n in names]


# -------------------------------------------------------------------- gates
def s1_units(reg, reads, nonlinear_of=()) -> Tuple[bool, Dict[str, Any], str]:
    """Every argument of a nonlinear function must be dimensionless.

    `nonlinear_of` lists the quantities the law passes through a nonlinear
    function (nu(), a log, an exponential).  Each must either be dimensionless
    already or be paired with a constant of the same dimension.
    """
    bad = []
    scales = {q.name: q.dim for q in reg.all() if q.status == "constant"}
    for n in nonlinear_of:
        d = reg[n].dim
        if d.is_dimensionless():
            continue
        if not any(str(sd) == str(d) for sd in scales.values()):
            bad.append({"quantity": n, "dim": str(d),
                        "problem": "no declared universal constant shares "
                                   "this dimension, so no dimensionless "
                                   "argument can be formed"})
    ok = not bad
    return ok, {"n_nonlinear_args": len(nonlinear_of), "violations": bad,
                "available_scales": {k: str(v) for k, v in scales.items()}}, (
        "all nonlinear arguments can be made dimensionless" if ok
        else f"{len(bad)} nonlinear argument(s) cannot be made dimensionless")


def s2_gauge(reg, reads) -> Tuple[bool, Dict[str, Any], str]:
    qs = _q(reg, reads)
    unsafe = [q.name for q in qs if not q.is_gauge_safe()]
    gauge_family = [q.name for q in qs if q.gauge]
    # A gauge-FIXED quantity is admissible, but the choice of rule is a
    # convention, so reading one must raise the flag rather than pass
    # silently.  The first version returned `ok` on `unsafe` alone, so the
    # convention_dependent taxonomy branch could never be reached and a
    # potential-depth candidate was reported as plainly admissible.
    ok = not (unsafe or gauge_family)
    return ok, {"gauge_unsafe": unsafe, "gauge_dependent": gauge_family,
                "rule_spread_dex": PHI_RULE_SPREAD_DEX if gauge_family else 0.0,
                "gate_margin_dex": PHI_GATE_MARGIN_DEX,
                "spread_exceeds_margin":
                    bool(gauge_family)
                    and PHI_RULE_SPREAD_DEX > 0.6 * PHI_GATE_MARGIN_DEX}, (
        "no gauge-dependent quantity" if not gauge_family else
        f"reads {len(gauge_family)} gauge-fixed quantity(ies); the spread "
        f"between defensible rules is {PHI_RULE_SPREAD_DEX} dex against a "
        f"{PHI_GATE_MARGIN_DEX} dex margin, so the verdict must be reported "
        f"under all four rules")


def s3_frame(reg, reads) -> Tuple[bool, Dict[str, Any], str]:
    qs = _q(reg, reads)
    boost_fixed = [q.name for q in qs if q.boost == "FRAME_FIXED"]
    sky_frame = [q.name for q in qs if q.rotation == "FRAME_DEPENDENT"]
    # a sky-frame quantity is admissible only in a PAIR that forms an invariant
    ok = not boost_fixed
    return ok, {"boost_frame_fixed": boost_fixed,
                "rotation_frame_dependent": sky_frame,
                "note": "a FRAME_DEPENDENT quantity may enter only through an "
                        "invariant combination (an angle between two such "
                        "axes, or a contraction); entering alone it makes the "
                        "law depend on the observer's choice of north"}, (
        "frame-safe" if ok else
        f"reads {boost_fixed}, defined only in one named frame")


def s4_coarse(reg, reads) -> Tuple[bool, Dict[str, Any], str]:
    qs = _q(reg, reads)
    cat = [q.name for q in qs if q.coarse_grain == "CATALOGUE_DEPENDENT"]
    topo = [q.name for q in qs if q.coarse_grain == "TOPOLOGICAL"]
    nonlin = [q.name for q in qs if q.coarse_grain == "NONLINEAR"]
    needs_gate = sorted(set(cat) | set(topo) | set(nonlin))
    ok = not (cat or topo)
    return ok, {"catalogue_dependent": cat, "topological": topo,
                "nonlinear": nonlin, "must_pass_commutation_gate": needs_gate,
                "n_gate_required": len(needs_gate)}, (
        "safe to read off an averaged scene" if not needs_gate else
        f"{len(needs_gate)} quantity(ies) do not commute with averaging; the "
        f"candidate may only be evaluated on a RESOLVED scene unless "
        f"commutation.erasure clears the specific operation" +
        (f".  {cat + topo} additionally depend on the catalogue partition, so "
         f"a merge/split convergence test is mandatory." if not ok else ""))


def s5_causal(reg, reads) -> Tuple[bool, Dict[str, Any], str]:
    qs = _q(reg, reads)
    bad = [q.name for q in qs if q.causal in ("FUTURE", "UNDEFINED")]
    retarded = [q.name for q in qs if q.causal in ("RETARDED",
                                                   "PAST_LIGHT_CONE")]
    ok = not bad
    return ok, {"acausal": bad, "retarded_or_lightcone": retarded}, (
        "causally available" if ok else f"reads {bad}, not on the past light "
                                        f"cone of the predicted event")


def s6_identifiable(reg, reads) -> Tuple[bool, Dict[str, Any], str]:
    """Only a genuinely FREE latent field fails this gate.

    BUG 7 of this lane: the first version failed a candidate whenever any input
    was not DIRECTLY observed, which flagged Newtonian gravity itself -- both
    `g_N` and `r_3d` are constructed.  That verdict was true and useless.  The
    three-way class on each quantity separates the cases: a CONSTRUCTIBLE input
    is determined by the resolved scene, a MARGINALISABLE one is integrated
    over by the scene ensemble, and only a NON_IDENTIFIABLE one leaves the
    candidate untestable.
    """
    qs = _q(reg, reads)
    cls: Dict[str, List[str]] = {}
    for q in qs:
        cls.setdefault(q.identifiability, []).append(q.name)
    free = cls.get("non_identifiable", [])
    marg = cls.get("marginalisable", [])
    contaminated = [q.name for q in qs if q.derived_under_theory]
    notes = {q.name: q.measurability_note for q in qs
             if q.identifiability == "non_identifiable"
             or q.derived_under_theory}
    ok = not (free or contaminated)
    return ok, {"by_class": cls, "non_identifiable": free,
                "theory_contaminated": contaminated,
                "marginalised_by_the_ensemble": marg, "reasons": notes,
                "n_marginalised": len(marg)}, (
        f"reads {contaminated}, which is produced by INVERTING an assumed "
        f"gravity law; scoring a candidate against it is circular"
        if contaminated else (
        (f"identifiable; {len(marg)} input(s) are unmeasured but marginalised "
         f"by the scene ensemble: {marg}" if marg else
         "every input is measured or constructible") if ok else
        f"{len(free)} input(s) are free latent fields with no observational "
        f"handle ({free}); the candidate is NON-IDENTIFIABLE on this corpus "
        f"unless the universe that generates them is specified -- a statement "
        f"about the experiment, not about the physics"))


def s7_rank(reg, reads) -> Tuple[bool, Dict[str, Any], str]:
    """Does the read set collapse under the recorded exact identities?"""
    s = set(reads)
    collapses = []
    for target, inputs, relation in EXACT_IDENTITIES:
        if target in s and all(i in s for i in inputs):
            collapses.append({"redundant": target, "given": list(inputs),
                              "relation": relation})
    eff = len(s) - len(collapses)
    ok = not collapses
    return ok, {"n_declared": len(s), "n_independent": eff,
                "collapses": collapses,
                "identities_checked": len(EXACT_IDENTITIES)}, (
        f"{len(s)} inputs, {eff} independent directions" if ok else
        f"{len(s)} inputs collapse to {eff} independent directions under "
        f"{len(collapses)} exact identity(ies); the candidate is simpler than "
        f"its term count suggests")


def s8_available(reg, reads, inventory: Optional[Dict[str, Any]]
                 ) -> Tuple[bool, Dict[str, Any], str]:
    """Is every input actually OBSERVED anywhere in the corpus?

    `inventory` maps ontology quantity -> {cluster: bool}.  A law reading a
    quantity no cluster has is not wrong; it is untestable on this corpus, and
    the charter asks for exactly that distinction plus the observation that
    would change it.
    """
    if not inventory:
        return True, {"checked": False}, "no inventory supplied; not checked"
    per: Dict[str, List[str]] = {}
    for n in reads:
        have = [c for c, v in inventory.get(n, {}).items() if v]
        per[n] = sorted(have)
    missing = [n for n, v in per.items() if not v]
    # clusters that have EVERY input
    all_c = set()
    for v in inventory.values():
        all_c |= set(v)
    complete = sorted(c for c in all_c
                      if all(inventory.get(n, {}).get(c) for n in reads))
    ok = not missing
    return ok, {"checked": True, "per_quantity": per,
                "quantities_with_no_cluster": missing,
                "clusters_with_every_input": complete,
                "n_clusters_complete": len(complete)}, (
        f"testable on {len(complete)} cluster(s): {complete}" if complete else
        (f"NO cluster in the corpus has all inputs" +
         (f"; {missing} are observed nowhere" if missing else
          "; each input exists somewhere but never together")))


# ------------------------------------------------------------------ driver
def prescreen(reads: Sequence[str], reg: Registry,
              nonlinear_of: Sequence[str] = (),
              inventory: Optional[Dict[str, Any]] = None,
              name: str = "candidate") -> Dict[str, Any]:
    out: Dict[str, Any] = {"name": name, "reads": list(reads), "gates": {}}
    unknown = [n for n in reads if n not in reg]
    if unknown:
        out["error"] = (f"{unknown} are not in the registry, so they carry no "
                        f"metadata contract and cannot enter a scene")
        out["taxonomy"] = "mathematically_inconsistent"
        return out

    res = {
        "S1_units": s1_units(reg, reads, nonlinear_of),
        "S2_gauge": s2_gauge(reg, reads),
        "S3_frame": s3_frame(reg, reads),
        "S4_coarse": s4_coarse(reg, reads),
        "S5_causal": s5_causal(reg, reads),
        "S6_identifiable": s6_identifiable(reg, reads),
        "S7_rank": s7_rank(reg, reads),
        "S8_available": s8_available(reg, reads, inventory),
    }
    hard_fail, flags = [], []
    for g, (ok, detail, reason) in res.items():
        out["gates"][g] = {"pass": bool(ok), "reason": reason, **detail}
        if not ok:
            (flags if g in FLAG_ONLY else hard_fail).append(g)

    contaminated = out["gates"]["S6_identifiable"].get("theory_contaminated")
    if hard_fail:
        tax = "mathematically_inconsistent"
    elif contaminated:
        tax = "theory_contaminated"
    elif out["gates"]["S6_identifiable"].get("non_identifiable") \
            or "S8_available" in flags:
        tax = "non_identifiable"
    elif "S2_gauge" in flags:
        tax = "convention_dependent"
    elif "S7_rank" in flags:
        tax = "admissible_but_redundant"
    else:
        tax = "admissible"
    out["hard_failures"] = hard_fail
    out["flags"] = flags
    out["taxonomy"] = tax
    return out


def prescreen_many(cands: Iterable[Dict[str, Any]], reg: Registry,
                   inventory: Optional[Dict[str, Any]] = None
                   ) -> List[Dict[str, Any]]:
    return [prescreen(c["reads"], reg, c.get("nonlinear_of", ()), inventory,
                      c.get("name", "candidate")) for c in cands]
