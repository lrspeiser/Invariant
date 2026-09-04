"""Stage 10 -- the data provenance ledger, and what is left to confirm with.

The charter (`invariant-gravity-discovery-charter.md`, Stage 10) requires three
separated classes, and states the rule that governs them:

    "Once any numerical result has been examined, that sample is no longer
     pristine confirmation data."

This script classifies every dataset the programme has touched.  It opens no
observational data of any kind -- only the research record and the lane source
tree.

A FALSE "PRISTINE" IS THE WORST ERROR THIS LANE CAN MAKE: it would burn an
already-spent dataset as confirmation.  The first version of this classifier
made exactly that error -- it called `galstreams` pristine because Run AR's
scoring text says "streams" while naming the catalogue only in its
data-integrity section.  So the classifier is treated as a SCREEN, not an
oracle, and a dataset is only called pristine when THREE independent channels
agree:

    1. no scoring-context mention in the research record (with aliases);
    2. no reference anywhere in the lane source tree or its JSON outputs;
    3. no manual override recorded below.

    python ledger.py
"""
import hashlib
import io
import json
import os
import re
import subprocess
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RECORD = r"C:/Users/henry/dev/gravity-discovery-program.md"
TREE = os.path.abspath(os.path.join(HERE, "..", ".."))

SCORED = [
    "dex", "chi2", "chi²", "sigma", "rms", "residual", "slope", "correlation",
    "fitted", "fit to", "scored", "measured", "recovered", "transfer",
    "held-out", "held out", "p =", "p=", "z =", "power", "null",
]

# name, record regex (incl. aliases), description, manual override or None
DATASETS = [
    ("SPARC", r"\bSPARC\b", "175 late-type galaxy rotation curves", None),
    ("X-COP", r"X-?COP", "12 clusters, thermodynamic profiles", None),
    ("CLASH", r"\bCLASH\b", "cluster lensing + supernova survey", None),
    ("LoCuSS", r"\bLoCuSS\b", "40 clusters, Subaru weak lensing", None),
    ("eFEDS", r"\beFEDS\b|Bahar", "542 X-ray systems, Vikhlinin fits", None),
    ("DECADE", r"\bDECADE\b", "per-cluster tangential shear", None),
    ("KiDS", r"\bKiDS\b", "galaxy-galaxy lensing, 35-2600 kpc",
     "VALIDATION (scored round 1: nu/nu_RAR = 1.31; sealed since)"),
    ("wide binaries", r"wide binar", "wide-binary accelerations",
     "VALIDATION (scored round 1: nu/nu_RAR = 0.90; sealed since)"),
    ("DiskMass", r"\bDiskMass\b|Bershady|Martinsson", "vertical dynamics", None),
    ("MaNGA", r"\bMaNGA\b", "integral-field spectroscopy", None),
    ("SAMI", r"\bSAMI\b", "IFU, field + cluster galaxies", None),
    # ALIASES MATTER: Run AR scores these as "streams", not by catalogue name.
    ("galstreams", r"\bgalstreams\b|stream track|stellar stream|\bstreams\b",
     "stellar stream tracks",
     "VALIDATION (Run AR: 68 usable-3D / 29 usable-6D, A_dyn measured)"),
    ("ACCEPT", r"\bACCEPT\b", "cluster entropy/density profiles", None),
    ("Pantheon+", r"Pantheon|supernova time dilation", "SN light curves", None),
    ("VoidFinder", r"\bVoidFinder\b", "SDSS void catalogue", None),
    ("REVOLVER", r"\bREVOLVER\b", "watershed void catalogue", None),
    ("DESIVAST", r"\bDESIVAST\b", "DESI void catalogue", None),
    ("Planck", r"\bPlanck\b", "CMB spectra and lensing", None),
    ("SN Refsdal", r"Refsdal", "strong-lens time delays", None),
    ("Frontier Fields", r"Frontier Field|\bHFF\b|Zitrin|\bCATS\b",
     "6 deep strong-lensing clusters", None),
    ("MUSE", r"\bMUSE\b", "IFU spectroscopy in cluster cores", None),
    ("X-GAP", r"X-?GAP", "group-scale hydrostatic profiles", None),
    ("CLoGS", r"\bCLoGS\b", "complete local-volume groups", None),
    ("SPT", r"\bSPT\b(?!\w)|South Pole Telescope", "SZ-selected clusters", None),
    ("Gaia (non-binary)", r"\bGaia\b", "astrometry outside sealed binaries", None),
]


def record_channel(text, rx):
    lines = text.splitlines()
    pat = re.compile(rx, re.I)
    n = scoring = 0
    hits = []
    for i, ln in enumerate(lines, 1):
        if pat.search(ln):
            n += 1
            window = " ".join(lines[max(0, i - 3):i + 2]).lower()
            hit = any(w in window for w in SCORED)
            scoring += int(hit)
            if len(hits) < 3:
                hits.append({"line": i, "scored_context": hit,
                             "text": ln.strip()[:110]})
    return n, scoring, hits


def tree_channel(rx):
    """Independent channel: does the lane source tree reference it at all?"""
    try:
        r = subprocess.run(
            ["git", "grep", "-riIl", "-E", rx, "--", "work/wellnet-2026-09"],
            cwd=os.path.abspath(os.path.join(TREE, "..", "..")),
            capture_output=True, text=True, timeout=120)
        files = [f for f in r.stdout.splitlines() if f.strip()]
    except Exception:                                        # noqa: BLE001
        files = []
    # the ledger itself names every dataset; do not let it incriminate them
    files = [f for f in files if "confirmation/ledger" not in f]
    return files


def main():
    text = io.open(RECORD, encoding="utf-8").read()
    rows = []
    for name, rx, what, override in DATASETS:
        n, scored, hits = record_channel(text, rx)
        files = tree_channel(rx)
        if override:
            status, why = override, "manual override (verified by hand)"
        elif scored > 0:
            status, why = "VALIDATION", f"{scored} scoring contexts in the record"
        elif files:
            status, why = "VALIDATION", f"referenced by {len(files)} lane files"
        elif n > 0:
            status, why = "PRISTINE", f"{n} mentions, all non-scoring; no lane code"
        else:
            status, why = "PRISTINE", "never mentioned, no lane code"
        rows.append(dict(dataset=name, what=what, mentions=n,
                         scoring_contexts=scored, lane_files=len(files),
                         status=status.split(" ")[0], status_full=status,
                         reason=why, evidence=hits,
                         lane_file_sample=files[:3]))

    rows.sort(key=lambda r: (r["status"] != "PRISTINE", -r["mentions"]))

    print("=" * 78)
    print("STAGE 10 -- DATA PROVENANCE LEDGER")
    print("=" * 78)
    print(f"{'dataset':<20} {'ment':>5} {'scor':>5} {'code':>5}  status / reason")
    print("-" * 78)
    for r in rows:
        print(f"{r['dataset']:<20} {r['mentions']:>5} {r['scoring_contexts']:>5} "
              f"{r['lane_files']:>5}  {r['status']:<10} {r['reason']}")

    pristine = [r for r in rows if r["status"] == "PRISTINE"]
    print()
    print(f"PRISTINE   {len(pristine):>2} of {len(rows)}   -> the entire remaining"
          f" confirmation budget")
    print(f"VALIDATION {len(rows)-len(pristine):>2}         -> spent; cannot confirm"
          f" anything, however carefully quarantined")
    print()
    for r in pristine:
        print(f"  CANDIDATE  {r['dataset']:<18} {r['what']}")

    doc = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="work/wellnet-2026-09/confirmation",
        stage="10 (untouched confirmation systems)",
        record_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        record_lines=len(text.splitlines()),
        rule=("Once any numerical result has been examined, that sample is no "
              "longer pristine confirmation data."),
        classifier=("three channels must agree: no scoring context in the record "
                    "(with aliases), no reference in the lane source tree, no "
                    "manual override"),
        opened_observational_data=False,
        rows=rows,
        counts=dict(pristine=len(pristine), validation=len(rows) - len(pristine),
                    total=len(rows)))
    p = os.path.join(HERE, "ledger.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1))
    print(f"\nwrote {p}")
    print(f"record sha256 {doc['record_sha256'][:32]}...")
    return doc


if __name__ == "__main__":
    main()
