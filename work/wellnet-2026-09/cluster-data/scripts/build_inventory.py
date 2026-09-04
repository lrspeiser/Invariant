"""Classify every acquired data file by (cluster, product) and emit inventory.json.

Product comes from the lane subdirectory.  Cluster comes from the filename,
falling back to the manifest's `cluster` key.  Anything that cannot be
classified is listed explicitly rather than dropped, so nothing goes missing
silently.
"""
import json
import os
import re
import glob

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PRODUCT_BY_DIR = {
    "members": "P1_member_galaxies",
    "bcg": "P2_bcg",
    "icl": "P3_icl",
    "gas": "P4_gas",
    "weaklensing": "P5_weak_lensing",
    "stronglensing": "P6_strong_lensing",
    "velocities": "P7_member_velocities",
}

CLUSTERS = [
    ("Abell 2744",           r"(a2744|abell[_ ]?2744|abell2744)"),
    ("MACS J0416.1-2403",    r"(macs[_ ]?j?0416|m0416|macs0416)"),
    ("MACS J0717.5+3745",    r"(macs[_ ]?j?0717|m0717|macs0717)"),
    ("MACS J1149.5+2223",    r"(macs[_ ]?j?1149|m1149|macs1149)"),
    ("Abell S1063",          r"(as1063|abell[_ ]?s?1063|abells1063|1063s|rxc?[_ ]?j?2248|rxj2248|j2248)"),
    ("Abell 370",            r"(a370|abell[_ ]?370|abell370)"),
    ("Abell 2029",           r"(a2029|abell[_ ]?2029)"),
]

MULTI_HINTS = ("hff6", "kluge", "demaio", "montes_trujillo", "jimenez_teja",
               "richard", "accept_all", "accept_table1", "mcxc", "xcop_raw",
               "shipley2018_hff", "lauer2014_bcg_metric", "donzelli2011_bcg_profiles",
               "cccp", "herbonnet", "meneghetti")


def classify_cluster(relpath, manifest):
    base = os.path.basename(relpath).lower()
    hits = [name for name, pat in CLUSTERS if re.search(pat, base)]
    if len(hits) == 1:
        return hits[0], "filename"
    if len(hits) > 1:
        return "MULTI: " + "; ".join(hits), "filename"
    if any(h in base for h in MULTI_HINTS):
        return "MULTI-CLUSTER TABLE", "filename-hint"
    c = (manifest or {}).get("cluster")
    if isinstance(c, list):
        return "MULTI: " + "; ".join(map(str, c)), "manifest"
    if c:
        return str(c), "manifest"
    return "UNCLASSIFIED", "none"


def main():
    rows = []
    for root, dirs, files in os.walk(LANE):
        dirs[:] = [d for d in dirs if d not in ("scripts", "__pycache__", ".git")]
        for fn in files:
            if fn.endswith((".manifest.json", ".py", ".pyc", ".md")):
                continue
            if fn in ("inventory.json", "validation_report.json"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, LANE).replace("\\", "/")
            top = rel.split("/")[0]
            product = PRODUCT_BY_DIR.get(top, "OTHER/" + top)
            is_raw_envelope = "/raw/" in rel or rel.startswith("raw/")
            mpath = path + ".manifest.json"
            man = None
            if os.path.exists(mpath):
                try:
                    man = json.load(open(mpath, encoding="utf-8"))
                except Exception:
                    man = None
            cluster, how = classify_cluster(rel, man)
            rows.append({
                "file": rel,
                "product": product,
                "cluster": cluster,
                "cluster_from": how,
                "is_raw_upstream_envelope": is_raw_envelope,
                "has_manifest": man is not None,
                "bytes": os.path.getsize(path),
                "row_count": (man or {}).get("row_count"),
                "source_url": (man or {}).get("source_url"),
                "note_head": ((man or {}).get("note") or "")[:220],
            })

    matrix = {}
    for r in rows:
        if r["is_raw_upstream_envelope"]:
            continue
        matrix.setdefault(r["cluster"], {}).setdefault(r["product"], []).append(r["file"])

    out = {"lane": LANE.replace("\\", "/"), "n_files": len(rows),
           "matrix": matrix, "files": sorted(rows, key=lambda r: r["file"])}
    with open(os.path.join(LANE, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    order = ["Abell 2744", "MACS J0416.1-2403", "MACS J0717.5+3745",
             "MACS J1149.5+2223", "Abell S1063", "Abell 370", "Abell 2029"]
    prods = ["P1_member_galaxies", "P2_bcg", "P3_icl", "P4_gas",
             "P5_weak_lensing", "P6_strong_lensing", "P7_member_velocities"]
    print("%-22s %s" % ("CLUSTER", " ".join("%-4s" % p.split("_")[0] for p in prods)))
    for c in order:
        cells = []
        for p in prods:
            n = len(matrix.get(c, {}).get(p, []))
            cells.append("%-4s" % (n if n else "-"))
        print("%-22s %s" % (c, " ".join(cells)))
    print()
    for c in sorted(matrix):
        if c not in order:
            print("OTHER KEY:", c, {p: len(v) for p, v in matrix[c].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
