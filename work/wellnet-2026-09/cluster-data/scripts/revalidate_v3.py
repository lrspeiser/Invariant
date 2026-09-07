# -*- coding: utf-8 -*-
"""Re-probe every velocity-lane rejection under `catalogue_validation` v3.

WHY THIS EXISTS
===============
`scripts/vizier.py` claimed in its docstring to check "that the echoed table
identifier matches the one we asked for".  It did not.  It computed `stem` and
`echoed` and never used either; the verdict was

    ok = nrows > 0 and bool(tables or titles)

which is v1 plus a row count.  D1 was dead code, D2 (`CatalogsExamined`) was
never looked for, and D3 (`#Title:` match) was captured but only printed.  That
puts every verdict the module produced at registry rule `catalogue_validation`
v2, which the registry records as BROKEN.

A rejection made by a client with a dead D1 is not a trustworthy absence: the
failure mode D1 exists to catch is precisely the one this lane hit, where
VizieR serves an unrelated REAL catalogue under HTTP 200.  So every identifier
in `velocities/_PRODUCT7_INDEX.json` -> `not_found` is re-probed here under the
corrected rule.

WHAT IS NOT RE-PROBED
=====================
`J/A+A/709/A254` (Granata et al. 2026) matches the CONFIRMATION-RESERVE token
`granata` (`goldcluster/guard.py` RESERVE_TOKENS).  It is recorded by identity
only and is NOT contacted.  Its round-1 verdict is carried forward and flagged
as not re-validated.

EVIDENCE
========
`velocities/raw/` is covered by `cluster-data/.gitignore` (`raw/`), so raw
payloads are NOT preserved by the repository.  The round-1 claim that "raw
evidence for each is preserved in velocities/raw/probe_*.tsv" is unbacked: the
directory is ignored, and six of the twelve identifiers have no archived
payload under any checkout.  This run therefore records detector verdicts,
byte counts and SHA-256s in `velocities/REVALIDATION_v3.json`, which IS
tracked; payloads go to `velocities/raw/` for local inspection only.

The JSON is rewritten after every probe and a re-run resumes from it, so a
killed run never loses a completed probe.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizier  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
VEL = os.path.join(LANE, "velocities")
RAW = os.path.join(VEL, "raw")
OUT_JSON = os.path.join(VEL, "REVALIDATION_v3.json")

#: seconds allowed per probe.  A nonexistent -source= makes VizieR run an
#: all-sky cross-match before it answers, so these are slow by construction.
PROBE_TIMEOUT = 100

#: bytes of response prefix kept per probe.  Every v3 detector reads header
#: lines, which arrive first; a wrong-serve streams megabytes of cross-match
#: results that we neither need nor want to buffer.
PROBE_MAX_BYTES = 262144

#: reserve tokens, copied from goldcluster/guard.py.  Matched case-insensitively
#: against the identifier AND the expected-title strings.
RESERVE_TOKENS = (
    "spt_", "spt-", "sptcl", "south_pole", "southpole",
    "x-gap", "xgap", "x_gap", "clogs", "granata", "muse_", "muse-",
    "gaia_dr", "gaiadr", "gaia_edr",
)
SEALED_TOKENS = ("kids", "wide_binar", "wide-binar")

# identifier, want_title (D3), round-1 verdict, what the round-1 rejection rested on
TARGETS = [
    ("J/A+A/656/A147", ("Mercurio",), "NOT FOUND",
     "identifier echo: Cooper+2013 fallback returned instead of the request"),
    ("J/A+A/574/A11", ("Karman",), "NOT FOUND",
     "identifier echo: Cooper+2013 fallback returned instead of the request"),
    ("J/A+A/599/A28", ("Karman",), "NOT FOUND",
     "identifier echo: Cooper+2013 fallback returned instead of the request"),
    ("J/A+A/587/A80", ("Caminha",), "NOT FOUND",
     "identifier echo: Cooper+2013 fallback returned instead of the request"),
    ("J/ApJ/781/L40", ("Ebeling",), "NOT FOUND",
     "explicit '#INFO Error=Table or Catalog not found'"),
    ("J/ApJ/693/L56", ("Ma+",), "NOT FOUND",
     "explicit 'Table or Catalog not found' error"),
    ("J/ApJ/684/160", ("Ma+",), "NOT FOUND",
     "explicit 'Table or Catalog not found' error"),
    ("J/A+A/588/A99", ("Limousin",), "NOT FOUND",
     "identifier echo: Cooper+2013 fallback returned instead of the request"),
    ("J/ApJ/767/15", ("Rines",), "EXISTS BUT NOT APPLICABLE",
     "content: HeCS redshift range 0.1<z<0.3 excludes A2029 at z=0.0773"),
    ("J/ApJ/819/63", ("Rines",), "EXISTS BUT NOT APPLICABLE",
     "content: HeCS-SZ range 0.02<z<0.3, no target cluster in the sample"),
    ("J/ApJS/240/39", ("Golovich",), "EXISTS BUT EMPTY FOR OUR TARGETS",
     "content: table1 gives Ng=0 for A2744 and MACS J1149"),
    ("J/A+A/633/A139", ("Ciocan",), "EXISTS BUT UNUSABLE",
     "content: tablea1 has no RA, no Dec and no redshift column"),
    ("J/A+A/709/A254", ("Granata",), "EXISTS BUT UNUSABLE",
     "content: Sersic structural parameters only, no redshift column"),
]

#: sub-tables whose COLUMN LIST carries a round-1 content claim, re-checked live
COLUMN_AUDIT = {
    "J/A+A/633/A139": ["J/A+A/633/A139/tablea1"],
    "J/ApJS/240/39": ["J/ApJS/240/39/table1"],
    "J/ApJ/767/15": ["J/ApJ/767/15/table1"],
    "J/ApJ/819/63": ["J/ApJ/819/63/table1"],
}


def reserved(*fields):
    blob = " ".join(fields).lower()
    return ([t for t in RESERVE_TOKENS if t in blob]
            + [t for t in SEALED_TOKENS if t in blob])


def sha(txt):
    return hashlib.sha256(txt.encode("utf-8", "replace")).hexdigest()


def probe(source, want_title=()):
    """vizier.probe with this script's shorter per-probe timeout."""
    orig = vizier.fetch
    vizier.fetch = (lambda u, timeout=PROBE_TIMEOUT, max_bytes=PROBE_MAX_BYTES:
                    orig(u, timeout=timeout, max_bytes=max_bytes))
    try:
        return vizier.probe(source, want_title=want_title, maxrows=5,
                            verbose=False)
    finally:
        vizier.fetch = orig


def main():
    os.makedirs(RAW, exist_ok=True)

    done, carried = {}, {}
    if os.path.exists(OUT_JSON):
        try:
            prev = json.load(io.open(OUT_JSON, encoding="utf-8"))
            done = {r["identifier"]: r for r in prev.get("results", [])
                    if r.get("verdict") not in (None, "REQUEST_FAILED")}
            # revalidate_v3_content.py writes into the same file; do not clobber
            # its block just because the probe half was re-run.
            carried = {k: v for k, v in prev.items()
                       if k.startswith("content_audit")}
        except Exception:                                       # noqa: BLE001
            done, carried = {}, {}

    out = {
        "rule": vizier.RULE,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "why": ("scripts/vizier.py asserted a #Name: echo check in its docstring "
                "but never performed one -- `stem`/`echoed` were computed and "
                "discarded, and the verdict was `nrows > 0 and bool(tables or "
                "titles)`. Every round-1 rejection was therefore made at "
                "registry rule catalogue_validation v2, recorded as BROKEN. "
                "This file re-probes those rejections under v3."),
        "detectors": {
            "D1": "#Name: echoes the exact requested identifier (or its parent)",
            "D2": "no CatalogsExamined= anywhere in the payload",
            "D3": "#Title: contains the expected author/year",
        },
        "verdict_vocabulary": {
            "SERVED": "D1-D3 all pass: the payload IS the requested catalogue",
            "ABSENT_ERROR": "VizieR returned an explicit #INFO Error=",
            "ABSENT_FALLBACK": "D1 or D2 failed: a DIFFERENT catalogue came back",
            "ABSENT_HTML": "a generic HTML page came back",
            "INDETERMINATE": "D1/D2 pass but the expected title does not match",
            "NOT_RE_PROBED_RESERVED": "confirmation-reserve token; not contacted",
        },
        "raw_payload_note": (
            "velocities/raw/ is gitignored (cluster-data/.gitignore: 'raw/'), so "
            "payloads are NOT preserved by the repo. The sha256 and byte count "
            "recorded per entry are the committed evidence."),
        "results": [],
    }
    out.update(carried)

    def flush():
        io.open(OUT_JSON, "w", newline="\r\n", encoding="utf-8").write(
            json.dumps(out, indent=1, ensure_ascii=False) + "\n")

    print("%-16s %-22s %-6s %-6s %-6s %-5s %s"
          % ("VERDICT", "IDENTIFIER", "D1", "D2", "D3", "rows", "note"), flush=True)
    print("-" * 110, flush=True)

    for source, want_title, round1, rested_on in TARGETS:
        if source in done:
            out["results"].append(done[source])
            print("%-16s %-22s (resumed)"
                  % (done[source].get("verdict"), source), flush=True)
            flush()
            continue

        rec = {"identifier": source, "round1_verdict": round1,
               "round1_rested_on": rested_on, "want_title": list(want_title)}

        hits = reserved(source, " ".join(want_title))
        if hits:
            rec.update(verdict="NOT_RE_PROBED_RESERVED", reserve_tokens=hits,
                       note=("matches a confirmation-reserve/sealed token; "
                             "recorded by identity only, not contacted. The "
                             "round-1 verdict is carried forward UNVALIDATED."))
            print("%-16s %-22s %-6s %-6s %-6s %-5s token=%s"
                  % ("RESERVED-SKIP", source, "-", "-", "-", "-",
                     ",".join(hits)), flush=True)
            out["results"].append(rec)
            flush()
            continue

        verdict, url, txt, d = probe(source, want_title)
        rec.update(verdict=verdict, url=url,
                   bytes=len(txt.encode("utf-8", "replace")), sha256=sha(txt),
                   D1_name_echo=d.get("D1_name_echo"),
                   D2_no_catalogs_examined=d.get("D2_no_catalogs_examined"),
                   D3_title_match=d.get("D3_title_match"),
                   names_seen=d.get("names_seen"),
                   titles_seen=d.get("titles_seen"),
                   served_instead=d.get("served_instead"),
                   known_fallback=d.get("known_fallback"),
                   catalogs_examined=d.get("catalogs_examined"),
                   vizier_error=d.get("vizier_error"),
                   n_rows=d.get("n_rows"), tables_seen=d.get("tables_seen"),
                   truncated=d.get("truncated"), probe_error=d.get("error"))
        if txt:
            fn = "probe_v3_" + source.replace("/", "_") + ".tsv"
            io.open(os.path.join(RAW, fn), "w", newline="\n",
                    encoding="utf-8").write(txt)
            rec["local_payload"] = "velocities/raw/" + fn

        print("%-16s %-22s %-6s %-6s %-6s %-5s %s"
              % (verdict, source, d.get("D1_name_echo"),
                 d.get("D2_no_catalogs_examined"), d.get("D3_title_match"),
                 d.get("n_rows"),
                 (",".join(d.get("served_instead") or [])[:44]
                  or d.get("vizier_error")
                  or (d.get("titles_seen") or [""])[0][:52])), flush=True)

        # re-check the round-1 content claims that a column list can settle
        if verdict == "SERVED" and source in COLUMN_AUDIT:
            rec["column_audit"] = []
            for tbl in COLUMN_AUDIT[source]:
                tv, _turl, ttxt, _td = probe(tbl)
                cols = []
                if ttxt:
                    try:
                        _m, cols, _u, _r = vizier.parse_tsv(ttxt)
                    except Exception:                           # noqa: BLE001
                        cols = []
                rec["column_audit"].append(
                    {"table": tbl, "verdict": tv, "columns": cols,
                     "sha256": sha(ttxt),
                     "bytes": len(ttxt.encode("utf-8", "replace"))})
                print("    column-audit %-26s %-16s cols=%s"
                      % (tbl, tv, ",".join(cols)[:56]), flush=True)

        out["results"].append(rec)
        flush()

    flush()
    print("\nWROTE", os.path.relpath(OUT_JSON, LANE).replace("\\", "/"),
          flush=True)

    served = [r for r in out["results"] if r["verdict"] == "SERVED"]
    absent = [r for r in out["results"] if str(r["verdict"]).startswith("ABSENT")]
    skipped = [r for r in out["results"]
               if r["verdict"] == "NOT_RE_PROBED_RESERVED"]
    other = [r for r in out["results"]
             if r["verdict"] not in ("SERVED", "NOT_RE_PROBED_RESERVED")
             and not str(r["verdict"]).startswith("ABSENT")]
    print("\nSERVED   (catalogue really is there) : %d" % len(served), flush=True)
    print("ABSENT   (rejection upheld)          : %d" % len(absent), flush=True)
    print("RESERVED (not re-probed)             : %d" % len(skipped), flush=True)
    print("OTHER                                : %d" % len(other), flush=True)
    for r in other:
        print("   !! %s %s" % (r["identifier"], r["verdict"]), flush=True)
    return out


if __name__ == "__main__":
    main()
