"""Fetch NED main-information rows for the polar-ring systems inventoried here.

NED's TAP service was timing out during this acquisition window (every sync
query >50 s), so the classic `objsearch` CGI endpoint is used instead, one
object at a time, with `of=ascii_bar`.
"""
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
URL = "https://ned.ipac.caltech.edu/cgi-bin/objsearch"
UA = {"User-Agent": "wellnet-2026-09 polar-ring acquisition (research; leonard@horizon3.net)"}

NAMES = ["NGC 4650A", "NGC 2685", "NGC 660", "NGC 4262", "UGC 7576", "UGC 9796",
         "IC 1689", "ARP 230", "NGC 4632", "NGC 6156", "NGC 7625", "ESO 415-G026",
         "AM 2020-504", "ESO 603-G21", "NGC 5122", "UGC 4385", "NGC 2748",
         "UGC 9562", "IC 51", "UGC 5119", "NGC 5128", "NGC 5014", "ESO 474-G26",
         "ESO 576-G069", "MCG -05-07-001", "NGC 4111", "NGC 3718", "IC 2006",
         "NGC 4753", "NGC 5907", "UGC 4261", "AM 1934-563", "ESO 235-G58",
         "ESO 503-G17", "NGC 3808B", "PGC 006101", "PGC 089058"]

HEADER = ("No.|Object Name|RA|DEC|Type|Velocity|Redshift|Redshift Flag|"
          "Magnitude and Filter|Separation|References|Notes|Photometry Points|"
          "Positions|Redshift Points|Diameter Points|Associations")

rows, missing = [], []
for n in NAMES:
    got = None
    for _ in range(2):
        try:
            r = requests.get(URL, headers=UA, timeout=60, params={
                "objname": n, "extend": "no", "out_csys": "Equatorial",
                "out_equinox": "J2000.0", "of": "ascii_bar",
                "obj_sort": "RA or Longitude", "list_limit": "5", "img_stamp": "NO"})
        except Exception:
            time.sleep(2)
            continue
        if r.status_code == 200 and "Object Name|RA|DEC" in r.text:
            got = r.text
            break
        time.sleep(2)
    if got is None:
        missing.append(n + " [no response]")
        continue
    data = [l.strip() for l in got.split("\n")
            if l.strip().startswith("1|") and "|" in l]
    if data:
        rows.append(data[0])
    else:
        missing.append(n + " [not resolved by NED]")

out = os.path.join(HERE, "ned_objdir_polar_ring_systems.csv")
with open(out, "w", encoding="utf-8", newline="") as f:
    f.write(HEADER + "\n" + "\n".join(rows) + "\n")
blob = open(out, "rb").read()
json.dump({
    "file": os.path.basename(out),
    "source_url": URL,
    "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sha256": hashlib.sha256(blob).hexdigest(),
    "bytes": len(blob),
    "row_count": len(rows),
    "column_count": len(HEADER.split("|")),
    "delimiter": "| (pipe) -- NED 'ascii_bar' output format",
    "columns": [
        {"name": "No.", "unit": "NED result index within the single-object query (always 1)"},
        {"name": "Object Name", "unit": "NED preferred name"},
        {"name": "RA", "unit": "deg (J2000)"},
        {"name": "DEC", "unit": "deg (J2000)"},
        {"name": "Type", "unit": "NED object type (G = galaxy)"},
        {"name": "Velocity", "unit": "km/s (heliocentric cz)"},
        {"name": "Redshift", "unit": "dimensionless"},
        {"name": "Redshift Flag", "unit": "NED redshift quality/source flag"},
        {"name": "Magnitude and Filter", "unit": "mag, with the filter appended"},
        {"name": "Separation", "unit": "arcmin (0 for a name query)"},
        {"name": "References", "unit": "count of NED references"},
        {"name": "Notes", "unit": "count"},
        {"name": "Photometry Points", "unit": "count"},
        {"name": "Positions", "unit": "count"},
        {"name": "Redshift Points", "unit": "count"},
        {"name": "Diameter Points", "unit": "count"},
        {"name": "Associations", "unit": "count"},
    ],
    "query": ("GET https://ned.ipac.caltech.edu/cgi-bin/objsearch?objname=<name>&extend=no"
              "&out_csys=Equatorial&out_equinox=J2000.0&of=ascii_bar"
              "&obj_sort=RA+or+Longitude&list_limit=5&img_stamp=NO -- one request per object"),
    "extraction": ("Raw NED ascii_bar main-information row per object, values unmodified; only the "
                   "single data line (the one beginning '1|') is kept from each response and the "
                   "shared header is written once. %d of %d queried names resolved."
                   % (len(rows), len(NAMES))),
    "unresolved_names": missing,
    "note": ("SUPPLEMENTARY. Positions and redshifts for the 40 kinematically confirmed PRGs are "
             "already carried by yu2026_table1_confirmed_PRGs.tsv (Yu et al. 2026 took them from "
             "NED), and NGC 4632 / NGC 6156 by Deg et al. 2023. NED's TAP service was timing out "
             "throughout this acquisition window, so the classic objsearch CGI was used instead. "
             "An unresolved name means NED did not match that spelling, not that the object is "
             "unknown to NED."),
}, open(out + ".manifest.json", "w", encoding="utf-8"), indent=2)
print("resolved %d of %d" % (len(rows), len(NAMES)))
if missing:
    print("unresolved:", missing)
