"""The run registry: invalidate in-flight and completed work when a rule changes.

Motivating failure (Run AZ): the VizieR catalogue-validation rule was found to be
wrong -- both recorded detectors had separately accepted wrong content -- while
three lanes were already running against it.  There was no way to reach them, so
a known-invalid assumption kept producing plausible-looking output, and nothing
in the pipeline would have stopped that output entering the scientific register.

This module is the missing piece.  Every run declares what it depends on; when a
RULE changes, every run that depends on it is marked

    INVALIDATED_PENDING_RERUN

and its outputs are quarantined from the register automatically, whether the run
is still going or finished months ago.

    python registry.py                 # show the register
    python registry.py --invalidate R  # mark every run depending on rule R
"""
import hashlib
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DB = os.path.join(HERE, "registry.json")

OK = "VALID"
DEAD = "INVALIDATED_PENDING_RERUN"
QUARANTINE = "QUARANTINED"

# ------------------------------------------------------- output permission states
# A numerically quarantined result can still steer the researcher.  Seeing that an
# invalid run favours a radius law can shape which parameterisations are generated
# next, which data are acquired, which boundary rules are emphasised, and which
# later result gets described as confirmatory.  So invalidation has TWO effects,
# not one, and the second is the one that was missing.
DIAGNOSTIC_ONLY = "DIAGNOSTIC_ONLY"
ADMISSIBLE = "SCIENTIFICALLY_ADMISSIBLE"

PERMITTED_USE = {
    DIAGNOSTIC_ONLY: ("locate implementation defects ONLY. May not alter "
                      "scientific hypotheses, priors, candidate selection, "
                      "acquisition priorities, or register conclusions. Any "
                      "hypothesis inspired by such a run is EXPLORATORY and "
                      "requires a fresh, independently specified test."),
    ADMISSIBLE: ("may alter the model register and future experiment design"),
}

# Headline estimates and model rankings from a diagnostic-only run stay SEALED
# until the corrected re-run completes.  Logs and failure diagnostics stay open.
SEALED_KEYS = ("hierarchy", "ranking", "bic", "dbic", "beta", "alpha",
               "best_model", "significance", "sigma", "transfer", "held")

# ------------------------------------------------------------------- the rules
# A RULE is a shared assumption that runs depend on.  When one changes, its
# `version` bumps and every run pinned to an older version is invalidated.
RULES = {
    "catalogue_validation": dict(
        version=3,
        summary="how an ingest decides a catalogue fetch returned what it asked for",
        history={
            1: "HTTP status only -- broken: VizieR returns 200 for absent IDs",
            2: "two detectors, either alone sufficient: #Name: echo, or absence "
               "of CatalogsExamined -- BROKEN, Run AZ: CatalogsExamined did not "
               "fire on a 45.9 kB wrong-catalogue serve, and the #Name: echo "
               "passed on a payload that was the wrong PAPER entirely",
            3: "three detectors, ALL required: #Name: echo, no CatalogsExamined, "
               "AND #Title: matching expected author/year. Plus: VOTable HTTP 200 "
               "is not success -- read query-status metadata; a zero-row result "
               "is not an absence until success is verified.",
        }),
    "temperature_support": dict(
        version=2,
        summary="what happens outside the measured temperature range",
        history={
            1: "np.interp with no left/right -- CLAMPS silently, forcing "
               "dlnT/dlnr -> 0 exactly where the radial trend is read",
            2: "fail closed: `forbid` is the default for any headline statistic; "
               "`clamp` survives only as a bit-identical reproduction mode and "
               "must print its extrapolated fraction",
        }),
    "holdout_seal": dict(
        version=2,
        summary="how a sealed dataset is kept unreachable",
        history={
            1: "intention and prose -- BROKEN: Bench.__init__ calls _widebin(), "
               "so any bare Bench() loads a sealed probe, and all 15 lanes do",
            2: "loader-level enforcement: no data-loading side effects in "
               "constructors; access needs a one-shot token; every open appends "
               "to a ledger; CI fails if an ordinary test touches a sealed product",
        }),
    "confirmation_status": dict(
        version=2,
        summary="when a dataset stops being confirmation-grade",
        history={
            1: "binary spent/pristine on evidence of scoring",
            2: "six-level ladder -- Mentioned/Acquired/Transformed are NOT spent; "
               "Scored/Inspected/Decision-used ARE. Plus four independence axes: "
               "untouched outcome / objects / survey / reduction pipeline",
        }),
    "identifiability_gate": dict(
        version=1,
        summary="the sensitivity certificate required before opening real data",
        history={1: "introduced; see ../stage4/"}),
}


def _commit():
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "unknown"
    except Exception:                                          # noqa: BLE001
        return "unknown"


def _load():
    if os.path.exists(DB):
        return json.loads(io.open(DB, encoding="utf-8").read())
    return dict(runs=[], rules={k: v["version"] for k, v in RULES.items()})


def _save(db):
    io.open(DB, "w", encoding="utf-8", newline="\n").write(json.dumps(db, indent=1))


def register(run_id, lane, depends_on, outputs=(), note=""):
    """Record a run and the rule versions it was launched under."""
    db = _load()
    rec = dict(run_id=run_id, lane=lane, commit=_commit(),
               registered_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               depends_on={r: RULES[r]["version"] for r in depends_on},
               outputs=list(outputs), status=OK, note=note, invalidated_by=None,
               output_state=ADMISSIBLE)
    db["runs"] = [r for r in db["runs"] if r["run_id"] != run_id] + [rec]
    _save(db)
    return rec


def invalidate_for_rule(rule, verbose=True):
    """Mark every run pinned to an older version of `rule`."""
    db = _load()
    cur = RULES[rule]["version"]
    hit = []
    for r in db["runs"]:
        v = r["depends_on"].get(rule)
        if v is not None and v < cur:
            # A run must carry EVERY reason it is invalid, not just the first
            # one found -- otherwise a re-run fixes one cause and inherits the
            # others silently.  This bug was present in the first version.
            reasons = r.get("invalidated_by") or []
            if isinstance(reasons, str):
                reasons = [reasons]
            tag = f"{rule} v{v} -> v{cur}"
            if tag not in reasons:
                reasons.append(tag)
                hit.append(r)
            r["status"] = DEAD
            r["invalidated_by"] = reasons
            # the second effect: the outputs stop being usable as EVIDENCE,
            # not merely as numbers in the register
            r["output_state"] = DIAGNOSTIC_ONLY
    _save(db)
    if verbose:
        print(f"rule '{rule}' is at v{cur}")
        if not hit:
            print("  no runs were pinned to an older version")
        for r in hit:
            print(f"  {DEAD}  {r['run_id']:<28} {r['lane']}")
            for why in r["invalidated_by"]:
                print(f"    {why}")
            for o in r["outputs"]:
                print(f"    QUARANTINED output: {o}")
    return hit


def report():
    db = _load()
    print("=" * 78)
    print("RUN REGISTRY")
    print("=" * 78)
    print(f"{'rule':<24} {'v':>3}  summary")
    print("-" * 78)
    for k, v in RULES.items():
        print(f"{k:<24} {v['version']:>3}  {v['summary']}")
    print()
    if not db["runs"]:
        print("no runs registered yet")
        return db
    print(f"{'run':<28} {'status':<26} lane")
    print("-" * 78)
    for r in sorted(db["runs"], key=lambda x: x["run_id"]):
        print(f"{r['run_id']:<28} {r['status']:<26} {r['lane']}")
        st = r.get("output_state", ADMISSIBLE)
        if st != ADMISSIBLE:
            print(f"    output state: {st}")
        for why in (r["invalidated_by"] or []):
            print(f"    invalidated by: {why}")
    n_dead = sum(1 for r in db["runs"] if r["status"] != OK)
    print()
    print(f"{len(db['runs'])} runs, {n_dead} invalidated pending rerun")
    if n_dead:
        print()
        print(f"{DIAGNOSTIC_ONLY}: {PERMITTED_USE[DIAGNOSTIC_ONLY]}")
        print()
        print("Sealed keys (headline estimates and model rankings) from those runs:")
        print("  " + ", ".join(SEALED_KEYS))
        print("Logs and failure diagnostics remain open; the numbers above do not.")
    return db


if __name__ == "__main__":
    if "--invalidate" in sys.argv:
        invalidate_for_rule(sys.argv[sys.argv.index("--invalidate") + 1])
    else:
        report()
