"""Probe candidate VizieR catalogue identifiers and report which really exist.

VizieR returns HTTP 200 with a generic HTML page for a nonexistent -source=,
so existence is decided by parsing the payload, never by the status code.
"""
import sys
import requests

CANDIDATES = [
    ("J/MNRAS/485/4726", "Koposov+2019 Orphan-Chenab 6-D"),
    ("J/ApJ/863/L20",    "Price-Whelan & Bonaca 2018 GD-1"),
    ("J/ApJ/891/161",    "Ibata+2020 GD-1 / STREAMFINDER"),
    ("J/ApJ/914/123",    "Ibata+2021 STREAMFINDER Gaia EDR3 streams"),
    ("J/ApJ/823/157",    "Ishigaki+2016 Pal 5 LOS velocities"),
    ("J/MNRAS/501/2279", "Vasiliev+2021 Sagittarius stream"),
    ("J/ApJ/892/L37",    "Bonaca+2020 GD-1 spur/gap"),
    ("J/ApJ/819/1",      "Ibata, Lewis & Martin 2016 Pal 5"),
    ("J/MNRAS/470/2410", "Sohn+ / Pal5 alt"),
    ("J/ApJ/889/70",     "Li+2020 S5 stream spectroscopy"),
    ("J/MNRAS/516/731",  "Li+2022 S5 DR1 stream spectroscopy"),
    ("J/A+A/635/L3",     "Antoja+2020 Sagittarius PM map"),
    ("J/ApJ/833/31",     "Sesar+2016"),
    ("J/MNRAS/520/5225", "Mateu 2023 galstreams"),
]

HDR = {"User-Agent": "gravity-research-acquisition/1.0 (academic data acquisition)"}
BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source={cat}&-out.all&-out.max=unlimited"


def probe(cat):
    url = BASE.format(cat=cat)
    try:
        r = requests.get(url, timeout=120, headers=HDR)
    except Exception as e:
        return "ERR", str(e), 0, url
    txt = r.text
    low = txt[:3000].lower()
    if "<html" in low or "<!doctype" in low:
        return "HTML(not a catalogue)", "", len(txt), url
    # a real VizieR TSV echoes the catalogue name in the ####### header comments
    echoed = cat.lower() in txt.lower()
    lines = txt.splitlines()
    data = 0
    hdr = None
    for i, l in enumerate(lines):
        if l.startswith("#") or not l.strip():
            continue
        if hdr is None:
            hdr = l
            continue
        if set(l.strip()) <= set("- \t"):
            continue
        data += 1
    # subtract the units row
    if data:
        data -= 1
    return ("OK" if echoed else "TSV-but-id-not-echoed"), (hdr or "")[:110], data, url


print("%-20s %-24s %8s  %s" % ("CATALOGUE", "STATUS", "ROWS", "DESC"))
found = []
for cat, desc in CANDIDATES:
    status, hdr, n, url = probe(cat)
    print("%-20s %-24s %8d  %s" % (cat, status, n, desc))
    if status == "OK" and n > 0:
        found.append((cat, desc, n))
print("\nEXISTING catalogues:", len(found))
for c, d, n in found:
    print("   %-20s %6d rows  %s" % (c, n, d))
