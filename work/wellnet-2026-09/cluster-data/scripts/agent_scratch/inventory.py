# -*- coding: utf-8 -*-
"""Final inventory: group files by cluster and report row counts."""
import os, json, glob, collections

MEM = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data\members"

CL = [("A2744", ["A2744_", "HFFDS_abell2744"]),
      ("MACS0416", ["MACS0416_"]),
      ("MACS0717", ["MACS0717_", "HFFDS_macs0717"]),
      ("MACS1149", ["MACS1149_", "HFFDS_macs1149"]),
      ("AS1063", ["AS1063_", "HFFDS_abell1063"]),
      ("A370", ["A370_", "HFFDS_abell370"]),
      ("A2029", ["A2029_"])]

mans = {os.path.basename(p)[:-len(".manifest.json")]: json.load(open(p, encoding="utf-8"))
        for p in glob.glob(os.path.join(MEM, "*.manifest.json"))
        if not os.path.basename(p).startswith("_")}
# also handle HFFDS_macs0416 which starts with MACS0416 prefix ambiguity
extra = {"HFFDS_macs0416clu_v3.9.zout": "MACS0416", "HFFDS_macs0416clu_v3.9.fout": "MACS0416"}

assigned = collections.defaultdict(list)
for name, m in sorted(mans.items()):
    tag = None
    for cl, prefixes in CL:
        if any(name.startswith(p) for p in prefixes):
            tag = cl
            break
    if name in extra:
        tag = extra[name]
    if name.startswith("MACS0717_MACS1149_MACS0416_Ebeling"):
        tag = "SHARED-Ebeling"
    if name.startswith("HFF6_Shipley"):
        tag = "SHARED-Shipley6"
    if name.startswith("Granata2026_AA709"):
        tag = "SHARED-raw"
    assigned[tag or "UNASSIGNED"].append((name, m["row_count"], m["column_count"]))

for cl, _ in CL:
    fs = assigned.get(cl, [])
    print("=" * 78)
    print("%s   (%d files, %d rows)" % (cl, len(fs), sum(f[1] for f in fs)))
    for n, r, c in fs:
        print("   %-62s %7d rows %4d cols" % (n[:62], r, c))
for k in ["SHARED-Ebeling", "SHARED-Shipley6", "SHARED-raw", "UNASSIGNED"]:
    if assigned.get(k):
        print("=" * 78)
        print(k)
        for n, r, c in assigned[k]:
            print("   %-62s %7d rows %4d cols" % (n[:62], r, c))
print("=" * 78)
print("TOTAL manifested data files:", sum(len(v) for v in assigned.values()))
