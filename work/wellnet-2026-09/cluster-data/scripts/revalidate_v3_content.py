# -*- coding: utf-8 -*-
"""Second half of the v3 re-validation: re-check the CONTENT claims.

`revalidate_v3.py` settles whether each rejected identifier is really absent.
Four of the twelve came back SERVED -- the catalogue exists and D1-D3 all pass
-- so their round-1 rejection does not rest on absence at all.  It rests on a
claim about what is inside:

  J/A+A/633/A139  "tablea1 has no RA, no Dec and no redshift column"
  J/ApJS/240/39   "table1 gives Ng=0 for A2744 and MACS J1149"
  J/ApJ/767/15    "HeCS covers 0.1<z<0.3, which excludes A2029 at z=0.0773"
  J/ApJ/819/63    "HeCS-SZ covers 0.02<z<0.3, no target cluster in the sample"

A content claim made through a client with a dead D1 is worth exactly as
little as an absence claim made the same way -- it could have been read off
the wrong catalogue.  Each is therefore re-derived here from the live table,
fetched under v3, and the outcome is appended to
`velocities/REVALIDATION_v3.json` as `content_audit`.

Nothing here opens a sealed or confirmation-reserve product.
"""
from __future__ import annotations

import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vizier  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
VEL = os.path.join(LANE, "velocities")
OUT_JSON = os.path.join(VEL, "REVALIDATION_v3.json")

#: the seven clusters this lane is about
TARGET_CLUSTERS = ["2744", "0416", "0717", "1149", "2248", "S1063", "370", "2029"]


def grab(table, want_title, maxrows="unlimited"):
    """Fetch a table under v3 and return (cols, rows, detectors)."""
    verdict, url, txt, d = vizier.probe(table, want_title=want_title,
                                        maxrows=maxrows, verbose=False)
    if verdict != "SERVED":
        return None, None, d, verdict
    try:
        _meta, cols, _units, rows = vizier.parse_tsv(txt)
    except Exception as exc:                                    # noqa: BLE001
        return None, None, dict(d, parse_error=repr(exc)), verdict
    return cols, rows, d, verdict


def main():
    out = json.load(io.open(OUT_JSON, encoding="utf-8"))
    audit = []

    # ---------------------------------------------------------------- 1
    # Ciocan+2020: does tablea1 really carry no RA, no Dec and no redshift?
    cols, rows, d, verdict = grab("J/A+A/633/A139/tablea1", ("Ciocan",))
    a = {"identifier": "J/A+A/633/A139/tablea1", "serve_verdict": verdict,
         "round1_claim": "no RA, no Dec and no redshift column",
         "columns": cols, "n_rows": (len(rows) if rows else None)}
    if cols is not None:
        low = [c.lower() for c in cols]
        has_ra = any(c.startswith(("ra", "_ra", "raj")) for c in low)
        has_de = any(c.startswith(("de", "_de", "dej", "dec")) for c in low)
        has_z = any(c == "z" or c.startswith(("z_", "cz", "redshift", "e_z"))
                    for c in low)
        a.update(has_ra=has_ra, has_dec=has_de, has_redshift=has_z,
                 claim_upheld=not (has_ra or has_de or has_z))
    audit.append(a)
    print("1) %-26s cols=%s" % ("J/A+A/633/A139/tablea1",
                                ",".join(cols or [])[:80]), flush=True)
    print("   RA=%s Dec=%s z=%s -> round-1 claim upheld: %s"
          % (a.get("has_ra"), a.get("has_dec"), a.get("has_redshift"),
             a.get("claim_upheld")), flush=True)

    # ---------------------------------------------------------------- 2
    # Golovich+2019: is Ng really 0 for A2744 and MACS J1149?
    cols, rows, d, verdict = grab("J/ApJS/240/39/table1", ("Golovich",))
    a = {"identifier": "J/ApJS/240/39/table1", "serve_verdict": verdict,
         "round1_claim": "Ng=0 for A2744 and MACS J1149; MACS J0717 not in sample",
         "columns": cols, "n_rows": (len(rows) if rows else None)}
    if cols and rows:
        iname = cols.index("Name") if "Name" in cols else 1
        ing = cols.index("Ng") if "Ng" in cols else None
        hits = []
        for r in rows:
            nm = r[iname] if iname < len(r) else ""
            if any(t.lower() in nm.lower() for t in TARGET_CLUSTERS):
                hits.append({"Name": nm,
                             "Ng": (r[ing] if ing is not None and ing < len(r)
                                    else None)})
        a["target_rows"] = hits
        nz = [h for h in hits
              if h["Ng"] not in (None, "", "0") and h["Ng"].strip() not in ("0",)]
        a["rows_with_nonzero_Ng"] = nz
        a["claim_upheld"] = not nz
    audit.append(a)
    print("2) %-26s target rows: %s" % ("J/ApJS/240/39/table1",
                                        a.get("target_rows")), flush=True)
    print("   nonzero Ng among targets: %s -> claim upheld: %s"
          % (a.get("rows_with_nonzero_Ng"), a.get("claim_upheld")), flush=True)

    # ---------------------------------------------------------------- 3/4
    # HeCS and HeCS-SZ: is any target cluster actually in the cluster table?
    for ident, tbl, who, claim in (
            ("J/ApJ/767/15", "J/ApJ/767/15/table1", ("Rines",),
             "HeCS covers 0.1<z<0.3; A2029 (z=0.0773) is below the range"),
            ("J/ApJ/819/63", "J/ApJ/819/63/table2", ("Rines",),
             "HeCS-SZ covers 0.02<z<0.3; no target cluster in the sample")):
        cols, rows, d, verdict = grab(tbl, who)
        a = {"identifier": tbl, "serve_verdict": verdict,
             "round1_claim": claim, "columns": cols,
             "n_rows": (len(rows) if rows else None)}
        if cols and rows:
            namecols = [i for i, c in enumerate(cols)
                        if c.lower() in ("name", "cluster", "rxc", "simbadname")]
            zcols = [i for i, c in enumerate(cols) if c.lower() in ("z", "cz")]
            hits = []
            for r in rows:
                blob = " ".join(r[i] for i in namecols if i < len(r))
                if any(t.lower() in blob.lower() for t in TARGET_CLUSTERS):
                    hits.append({"row": blob,
                                 "z": [r[i] for i in zcols if i < len(r)]})
            a["target_rows"] = hits
            a["claim_upheld"] = not hits
            if zcols and rows:
                zs = []
                for r in rows:
                    try:
                        zs.append(float(r[zcols[0]]))
                    except Exception:                           # noqa: BLE001
                        pass
                if zs:
                    # HeCS tabulates z; HeCS-SZ tabulates cz in km/s.  Label the
                    # column for what it is, and convert, so the round-1 range
                    # claim (0.02<z<0.3) can be compared against it directly.
                    lo, hi = min(zs), max(zs)
                    if cols[zcols[0]].lower() == "cz":
                        a["cz_range_observed_km_s"] = [lo, hi]
                        a["z_range_implied"] = [round(lo / 299792.458, 4),
                                                round(hi / 299792.458, 4)]
                        a["units_note"] = ("the column is cz in km/s, not z; "
                                           "converted so the round-1 range claim "
                                           "can be compared directly")
                    else:
                        a["z_range_observed"] = [lo, hi]
                        a["z_range_implied"] = [lo, hi]
        audit.append(a)
        print("%s %-26s target rows: %s  z-range: %s -> claim upheld: %s"
              % ("3)" if "767" in ident else "4)", tbl, a.get("target_rows"),
                 a.get("z_range_implied"), a.get("claim_upheld")), flush=True)

    out["content_audit"] = audit
    out["content_audit_note"] = (
        "The four identifiers that came back SERVED were rejected in round 1 on "
        "CONTENT, not absence. Those content claims were also made through the "
        "client with the dead D1, so each is re-derived here from the live "
        "table fetched under v3.")
    io.open(OUT_JSON, "w", newline="\r\n", encoding="utf-8").write(
        json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print("\nWROTE content_audit into velocities/REVALIDATION_v3.json", flush=True)

    upheld = [a for a in audit if a.get("claim_upheld") is True]
    broken = [a for a in audit if a.get("claim_upheld") is False]
    print("content claims upheld : %d" % len(upheld), flush=True)
    print("content claims BROKEN : %d" % len(broken), flush=True)
    for a in broken:
        print("   !! %s" % a["identifier"], flush=True)


if __name__ == "__main__":
    main()
