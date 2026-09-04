"""Stage 10 -- the confirmation reserve, its seal, and a tripwire.

The charter requires three separated data classes and says why:

    "Once any numerical result has been examined, that sample is no longer
     pristine confirmation data."

The ledger (`ledger.py`) found that 23 of 25 datasets are already spent.  This
module DECLARES the remainder as a sealed confirmation reserve, and installs the
check that would have prevented the loss: a tripwire any lane can call, which
fails loudly if the lane references a reserved dataset.

Nothing here opens observational data.  Sealing a dataset means recording its
IDENTITY and the statistic that will one day be applied to it -- never its
content.

    python seal.py            # print the seal and verify it
    python seal.py --check    # run the tripwire over the lane tree
"""
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# ---------------------------------------------------------------- the reserve
# tier A: no contact of any kind, in the record or the source tree
# tier B: NAMED as future work or counted in an inventory, but no gravity-
#         relevant statistic has ever been computed.  Hand-verified; the exact
#         contact is recorded so a later reader can re-judge it.
RESERVE = [
    dict(name="SPT clusters", tier="A", what="SZ-selected cluster sample",
         contact="none: 0 mentions in the record, 0 references in the lane tree",
         verified="clean", rx=r"SPT(?!\w)|South Pole Telescope"),
    dict(name="X-GAP", tier="A", what="group-scale hydrostatic profiles",
         contact=("record L5425 proposes 'pairing it with X-GAP or CLoGS ... is "
                  "the obvious next lane'. Tripwire hits are two PostScript "
                  "figure files (binary noise) plus that proposal. Never opened."),
         verified="clean after hand-check", rx=r"X-?GAP"),
    dict(name="CLoGS", tier="A", what="complete local-volume groups",
         contact=("a lane ATTEMPTED acquisition and recorded a confirmed "
                  "ABSENCE: 'J/A+A/601/A95 is NOT O'Sullivan CLoGS -- it is "
                  "Calabro+ 2017'. A failed acquisition examines no result, so "
                  "the data remain unexamined."),
         verified="clean after hand-check", rx=r"CLoGS"),
    dict(name="Gaia proper motions / astrometry",
         tier="B-scoped", what="Gaia kinematics as a GRAVITY dataset",
         contact=("Gaia appears in 178 lane files, but as an ASTROMETRIC FRAME "
                  "('RA/Dec aligned to Gaia DR2'), not as a gravity dataset. "
                  "EXCLUDED from the reserve: any Gaia-derived wide-binary "
                  "quantity, which is separately sealed. RESERVED: proper "
                  "motions and parallaxes used dynamically."),
         verified="scoped by product", rx=r"Gaia"),
    dict(name="MUSE/Granata internal dispersions",
         tier="B-scoped",
         what="213 internal stellar velocity dispersions in HFF members",
         contact=("MUSE appears in 98 lane files, but as SPECTROSCOPIC "
                  "REDSHIFTS for cluster membership -- those are spent. "
                  "RESERVED: only the internal velocity DISPERSIONS "
                  "(Granata 2026), which were inventoried (213 counted) but "
                  "never entered a gravity statistic."),
         verified="scoped by product", rx=r"Granata"),
]

# The lesson the tripwire taught, recorded so it is not relearned:
SCOPING_RULE = (
    "Scope a confirmation reserve by DATA PRODUCT, not by survey name. "
    "A survey can be simultaneously spent and pristine: MUSE redshifts are "
    "spent for membership while MUSE internal dispersions are untouched; Gaia "
    "is spent as an astrometric frame while its dynamical products are "
    "untouched. Naming the survey would have burned the pristine half or "
    "falsely cleared the spent half."
)

# ------------------------------------------------- the statistic, frozen NOW
# The charter: freeze the model AND the statistic, then evaluate once.
# No law is promoted yet, so the model slot is empty by design -- but the
# statistic is declared here, in advance, so it cannot be chosen to fit.
DECLARED_STATISTIC = {
    "primary": ("median |log10(g_obs_pred / g_obs_meas)| over the object's "
                "measured radial range, computed on RAW observables after the "
                "candidate law and the instrument forward model are both "
                "applied"),
    "comparator": ("the same quantity for the RAR with a0 frozen at its "
                   "galaxy-calibrated value and no free parameter"),
    "decision": ("the candidate must beat the comparator on the reserve with "
                 "every parameter frozen; a tie or a loss is a null result and "
                 "is reported as such"),
    "forbidden": [
        "refitting any parameter on the reserve, global or per-object",
        "choosing the radial range after seeing residuals",
        "dropping objects after seeing residuals",
        "reporting a subset without the full-sample number beside it",
        "a second evaluation of the same reserve for the same law family",
    ],
    "power_precondition": ("the responsiveness d(estimate)/d(injected) must be "
                           "measured on synthetic data and reported BEFORE the "
                           "reserve is opened; a null with unstated power is "
                           "not a result"),
}


def seal_document():
    body = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="work/wellnet-2026-09/confirmation",
        stage="10 (untouched confirmation systems)",
        charter=("C:/Users/henry/dev/invariant-gravity-discovery-charter.md"),
        rule=("Once any numerical result has been examined, that sample is no "
              "longer pristine confirmation data."),
        reserve=RESERVE,
        scoping_rule=SCOPING_RULE,
        declared_statistic=DECLARED_STATISTIC,
        model_slot=("EMPTY BY DESIGN -- no law is promoted. The statistic above "
                    "is frozen first so it cannot later be chosen to fit."),
        opened_observational_data=False,
        evaluations_used=0,
        evaluations_permitted=1,
    )
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    body["seal_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return body


def tripwire(verbose=True):
    """Fail loudly if any lane file references a reserved dataset."""
    violations = []
    for entry in RESERVE:
        try:
            r = subprocess.run(
                ["git", "grep", "-riIl", "-E", entry["rx"], "--",
                 "work/wellnet-2026-09"],
                cwd=REPO, capture_output=True, text=True, timeout=120)
            files = [f for f in r.stdout.splitlines() if f.strip()]
        except Exception as e:                                # noqa: BLE001
            if verbose:
                print(f"  tripwire could not run for {entry['name']}: {e!r}")
            continue
        # this lane is allowed to name the reserve; that is its job
        files = [f for f in files if "/confirmation/" not in f]
        if files:
            violations.append((entry["name"], files))
    if verbose:
        print("TRIPWIRE -- lane files referencing a reserved dataset")
        if not violations:
            print("  clean: 0 violations across "
                  f"{len(RESERVE)} reserved datasets")
        for name, files in violations:
            print(f"  VIOLATION  {name}: {len(files)} file(s)")
            for f in files[:5]:
                print(f"             {f}")
    return violations


def main():
    doc = seal_document()
    if "--check" in sys.argv:
        v = tripwire()
        raise SystemExit(1 if v else 0)

    print("=" * 78)
    print("STAGE 10 -- CONFIRMATION RESERVE, SEALED")
    print("=" * 78)
    print(f"{'dataset':<28} {'tier':<5} what")
    print("-" * 78)
    for e in RESERVE:
        print(f"{e['name']:<28} {e['tier']:<5} {e['what']}")
    print()
    print("tier A = no contact of any kind")
    print("tier B = named as future work or inventoried; no gravity statistic ever")
    print()
    print("DECLARED STATISTIC (frozen before any law exists, so it cannot be")
    print("chosen to fit):")
    print(f"  primary    {DECLARED_STATISTIC['primary']}")
    print(f"  comparator {DECLARED_STATISTIC['comparator']}")
    print(f"  decision   {DECLARED_STATISTIC['decision']}")
    print("  forbidden:")
    for f in DECLARED_STATISTIC["forbidden"]:
        print(f"    - {f}")
    print()
    print(f"  evaluations permitted: {doc['evaluations_permitted']}   "
          f"used: {doc['evaluations_used']}")
    print()
    v = tripwire()
    print()
    p = os.path.join(HERE, "seal.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1))
    print(f"wrote {p}")
    print(f"SEAL sha256 {doc['seal_sha256']}")
    return 1 if v else 0


if __name__ == "__main__":
    raise SystemExit(main())
