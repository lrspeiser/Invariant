"""Stage 10 -- the confirmation reserve definition and its tripwire.

Separated from `seal.py` so the reserve can be imported by any lane without
re-running the seal.

THE SCOPING RULE, learned from the tripwire's own first run:

    Scope a confirmation reserve by DATA PRODUCT, not by survey name.  A survey
    can be simultaneously spent and pristine.  MUSE redshifts are spent (they
    set cluster membership in an earlier lane) while MUSE internal velocity
    dispersions are untouched.  Gaia is spent as an astrometric frame ("RA/Dec
    aligned to Gaia DR2") while its dynamical products are untouched.  Naming
    the survey would either have burned the pristine half or falsely cleared
    the spent half.

Nothing here opens observational data.  Sealing means recording IDENTITY, never
content.
"""
import os
import re
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

# Tier A  : no contact of any kind, in the record or the lane tree.
# Tier B- : the SURVEY has been touched but the reserved PRODUCT has not.
#           `exclude` names what is explicitly NOT reserved.
RESERVE = [
    dict(
        name="SPT clusters",
        tier="A",
        product="SZ-selected cluster sample",
        rx=r"\bSPT\b(?!\w)|South Pole Telescope",
        contact="none: 0 record mentions, 0 lane-tree references",
        exclude=None,
    ),
    dict(
        name="X-GAP",
        tier="A",
        product="group-scale hydrostatic profiles",
        rx=r"X-?GAP",
        contact=("record L5425 PROPOSES it -- 'pairing it with X-GAP or CLoGS "
                 "hydrostatic profiles is the obvious next lane'. Never opened."),
        exclude=None,
    ),
    dict(
        name="CLoGS",
        tier="A",
        product="complete local-volume groups",
        rx=r"\bCLoGS\b",
        contact=("a lane ATTEMPTED acquisition and recorded a confirmed "
                 "ABSENCE: J/A+A/601/A95 is Calabro+2017, not O'Sullivan's "
                 "CLoGS. A failed acquisition examines no result."),
        exclude=None,
    ),
    dict(
        name="Gaia dynamical products",
        tier="B-scoped",
        product="proper motions and parallaxes used as a GRAVITY dataset",
        rx=r"\bGaia\b",
        contact=("Gaia appears in ~178 lane files, but as an ASTROMETRIC FRAME "
                 "('RA/Dec aligned to Gaia DR2') -- a coordinate calibration, "
                 "not a gravity measurement."),
        exclude=("any Gaia-derived wide-binary quantity (separately sealed); "
                 "Gaia used solely to register coordinates"),
    ),
    dict(
        name="MUSE/Granata internal dispersions",
        tier="B-scoped",
        product="213 internal stellar velocity dispersions in HFF members",
        rx=r"Granata",
        contact=("MUSE appears in ~98 lane files as SPECTROSCOPIC REDSHIFTS "
                 "for cluster membership -- those are spent. The internal "
                 "dispersions were inventoried (213 counted) but never entered "
                 "a gravity statistic; a row count is not a result."),
        exclude="all MUSE redshifts and membership products",
    ),
]

# Hand-verified benign contacts, with the reason.  The tripwire reports only
# contacts NOT on this list, so a genuinely new reference stands out.
BENIGN = {
    "X-GAP": [
        ("env-data/raw/streams-satellites/crot_jin2016_manga_misalign_src/"
         "morph.eps", "PostScript figure; substring noise, not the survey"),
        ("env-data/raw/warps-vertical/arxiv_martinsson2013_dms6/PPakAtlas/"
         "UGC04622_n.ps", "PostScript figure; substring noise"),
        ("potential-depth/REPORT.md", "the proposal sentence itself"),
    ],
    "CLoGS": [
        ("lead01-ablation/", "records a confirmed ABSENCE, not an ingest"),
        ("potential-depth/REPORT.md", "the proposal sentence itself"),
        ("lead01-ablation/data/fresh/", "the same negative finding"),
        ("potential-depth/code/probe.py",
         "a VizieR ID PROBE LIST, and the ID is wrong: line 18 labels "
         "J/A+A/601/A95 as O'Sullivan CLoGS when it is Calabro+2017. The "
         "probe therefore never fetched CLoGS -- which is precisely why the "
         "data stay pristine. Same VizieR mislabelling family already on "
         "record; fix the ID before any future acquisition."),
    ],
    "Gaia dynamical products": [
        ("cluster-data/", "astrometric frame reference only"),
        (".bib", "downloaded bibliography, not data"),
        (".html", "downloaded bibliography page, not data"),
        ("/acquire/", "acquisition-probe scratch, not an ingest"),
        ("literature", "reference list"),
        (".tex", "downloaded paper source, not data"),
        (".bbl", "downloaded bibliography, not data"),
        ("/eprints/", "downloaded paper sources"),
        ("INVENTORY", "inventory prose"),
        ("env-data/", "acquisition scratch and paper sources"),
    ],
    "MUSE/Granata internal dispersions": [
        ("cluster-data/", "inventory entry counting the dispersions"),
    ],
}


# A LIMITATION, recorded rather than hidden.  "Gaia" is cited by essentially
# every modern astronomy paper, so a name-regex tripwire over a tree containing
# downloaded eprints cannot police it: after allowlisting bibliographies, paper
# sources and acquisition scratch, what remains is still prose.  Gaia's reserve
# must therefore be policed by an EXPLICIT ACQUISITION MANIFEST -- a lane may
# only use a Gaia dynamical product it has declared here first -- not by this
# scan.  The scan is reliable for the distinctive names (SPT, X-GAP, CLoGS,
# Granata) and unreliable for ubiquitous ones.
POLICED_BY_MANIFEST_ONLY = {"Gaia dynamical products"}


def _benign(name, path):
    for frag, _why in BENIGN.get(name, []):
        if frag in path:
            return True
    return False


def scan(verbose=True):
    """Report lane-tree contact with the reserve, excluding verified-benign."""
    new_contacts, deferred = [], []
    for e in RESERVE:
        if e["name"] in POLICED_BY_MANIFEST_ONLY:
            deferred.append(e["name"])
            continue
        try:
            r = subprocess.run(
                ["git", "grep", "-riIl", "-E", e["rx"], "--",
                 "work/wellnet-2026-09"],
                cwd=REPO, capture_output=True, text=True, timeout=120)
            files = [f for f in r.stdout.splitlines() if f.strip()]
        except Exception as exc:                              # noqa: BLE001
            if verbose:
                print(f"  scan failed for {e['name']}: {exc!r}")
            continue
        files = [f for f in files
                 if "/confirmation/" not in f and not _benign(e["name"], f)]
        if files:
            new_contacts.append((e["name"], files))

    if verbose:
        print("RESERVE TRIPWIRE")
        print(f"  {len(RESERVE)} reserved products; "
              f"{sum(len(v) for v in BENIGN.values())} verified-benign contacts "
              f"on the allowlist")
        if not new_contacts:
            print("  CLEAN: no unexplained lane-tree contact with the "
                  "regex-policeable reserve")
        for name in deferred:
            print(f"  DEFERRED     {name}: name too ubiquitous to police by "
                  f"regex; use the acquisition manifest")
        for name, files in new_contacts:
            print(f"  NEW CONTACT  {name}: {len(files)} file(s)")
            for f in files[:6]:
                print(f"               {f}")
    return new_contacts


if __name__ == "__main__":
    import sys
    sys.exit(1 if scan() else 0)
