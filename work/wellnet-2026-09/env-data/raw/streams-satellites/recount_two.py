"""Careful, section-aware recount of the two files where the generic heuristic
disagreed with the manifest. A VizieR ASU-TSV section is:
    <header row> <units row> <dashes row> <data rows...>
and a new section begins after a '#Table' marker.
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))

for fn in ("crot_califa_kinclass_kalinova2017.raw.tsv", "sat_mcconnachie2012_refs.tsv"):
    p = os.path.join(BASE, fn)
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
    sections = []
    cur = None
    for l in lines:
        if l.startswith("#Table"):
            cur = {"name": l.split()[1] if len(l.split()) > 1 else "?", "rows": []}
            sections.append(cur)
            continue
        if l.startswith("#"):
            continue
        if not l.strip():
            continue
        if cur is None:
            cur = {"name": "(implicit)", "rows": []}
            sections.append(cur)
        cur["rows"].append(l)
    print("=== %s" % fn)
    total = 0
    for s in sections:
        rows = s["rows"]
        # strip header, units, and a dashes rule row if present
        n = len(rows)
        drop = 0
        if n > 0:
            drop += 1                                   # header
        if n > 1:
            drop += 1                                   # units
        if n > 2 and set(rows[2].replace("\t", "").strip()) <= set("- "):
            drop += 1                                   # dashes rule
        data = n - drop
        total += data
        print("   section %-28s lines=%4d dropped=%d data=%4d" % (s["name"], n, drop, data))
    print("   TOTAL DATA ROWS = %d" % total)
    import json
    man = json.load(open(p + ".manifest.json", encoding="utf-8"))
    print("   manifest row_count = %s  -> %s" %
          (man["row_count"], "AGREES" if man["row_count"] == total else "DISAGREES"))
    print()
