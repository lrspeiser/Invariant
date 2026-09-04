"""Fetch NED positions/redshifts/types for the polar-ring systems that are NOT
already covered by Yu et al. 2026 Table 1 (which itself carries NED coordinates
and redshifts for the 40 kinematically confirmed PRGs)."""
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = ["NGC 4632", "NGC 6156", "NGC 4111", "NGC 3718", "IC 2006", "NGC 4753",
         "NGC 5907", "MCG -05-07-001", "ESO 235-G 058", "ESO 503-G 017",
         "AM 1934-563", "NGC 3808B", "NGC 4650A", "NGC 2685", "NGC 0660",
         "NGC 4262", "UGC 07576", "UGC 09796", "UGC 04261", "IC 1689"]
URL = "https://ned.ipac.caltech.edu/tap/sync"
COLS = "prefname, ra, dec, z, zunc, prefphytype, z_bibcode"

rows, hdr, missing = [], None, []
for n in NAMES:
    q = "SELECT %s FROM NEDTAP.objdir WHERE prefname = '%s'" % (COLS, n)
    got = False
    for _ in range(3):
        try:
            r = requests.get(URL, params={"request": "doQuery", "LANG": "ADQL",
                                          "QUERY": q, "FORMAT": "csv"}, timeout=120)
        except Exception:
            time.sleep(3)
            continue
        if r.status_code == 200 and r.text.lstrip().startswith("prefname"):
            got = True
            break
        time.sleep(3)
    if not got:
        missing.append(n)
        continue
    ls = [l for l in r.text.strip().split("\n") if l.strip()]
    hdr = ls[0]
    if len(ls) > 1:
        rows.extend(ls[1:])
    else:
        missing.append(n)

out = os.path.join(HERE, "ned_objdir_polar_ring_systems.csv")
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write((hdr or COLS.replace(", ", ",")) + "\n" + "\n".join(rows) + "\n")
blob = open(out, "rb").read()
json.dump({
    "file": os.path.basename(out),
    "source_url": URL,
    "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sha256": hashlib.sha256(blob).hexdigest(),
    "bytes": len(blob),
    "row_count": len(rows),
    "column_count": 7,
    "columns": [
        {"name": "prefname", "unit": "NED preferred object name"},
        {"name": "ra", "unit": "deg (J2000)"},
        {"name": "dec", "unit": "deg (J2000)"},
        {"name": "z", "unit": "dimensionless (NED preferred redshift)"},
        {"name": "zunc", "unit": "dimensionless"},
        {"name": "prefphytype", "unit": "NED preferred physical object type (G = galaxy)"},
        {"name": "z_bibcode", "unit": "bibcode of the adopted redshift"},
    ],
    "query": "One ADQL query per object to NED TAP (sync): SELECT %s FROM NEDTAP.objdir WHERE prefname = '<name>'" % COLS,
    "extraction": "Concatenated raw NED TAP CSV rows, values unmodified. %d of %d queried names resolved." % (len(rows), len(NAMES)),
    "unresolved_names": missing,
    "note": "Supplementary positions/redshifts for polar-ring systems discussed in POLAR_RINGS.md. The 40 kinematically confirmed PRGs already carry NED coordinates and redshifts via yu2026_table1_confirmed_PRGs.tsv. NED prefname matching is exact: an unresolved name means the queried string is not NED's preferred designation, not that the object is unknown to NED.",
}, open(out + ".manifest.json", "w", encoding="utf-8"), indent=2)
print("resolved %d of %d; unresolved: %s" % (len(rows), len(NAMES), missing))
