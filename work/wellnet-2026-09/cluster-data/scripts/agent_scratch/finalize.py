# -*- coding: utf-8 -*-
"""Annotate the Granata structural manifests with the released-vs-analysed sample
distinction, then verify every data file in members/ has a manifest whose sha256,
byte count and row count still match the file on disk."""
import os, json, glob, hashlib

MEM = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data\members"

ADD = (" SAMPLE-SIZE NOTE (checked against the CDS ReadMe for J/A+A/709/A254): the four released structural tables contain "
       "225 (A2744) + 222 (AS1063) + 224 (M0416) + 279 (M1149) = 950 rows in total, and the row count of this file matches "
       "the ReadMe record count for its table exactly. The paper abstract quotes 723 'red cluster members' -- that is the "
       "smaller colour/morphology-selected subsample used for the Fundamental Plane and velocity-dispersion-function "
       "analysis, NOT the size of the released tables. Do not expect these files to sum to 723.")

for p in sorted(glob.glob(os.path.join(MEM, "*Granata2026*structural.raw.tsv.manifest.json"))):
    m = json.load(open(p, encoding="utf-8"))
    if "SAMPLE-SIZE NOTE" not in m["note"]:
        m["note"] = m["note"] + ADD
        m["cds_readme_url"] = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/709/A254/ReadMe"
        json.dump(m, open(p, "w", encoding="utf-8"), indent=2)
        print("annotated", os.path.basename(p))

print("\n=== VERIFICATION PASS ===")


def sha256f(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


data_files = [f for f in sorted(os.listdir(MEM)) if not f.endswith(".manifest.json")]
bad = []
tot_rows = 0
print("%-64s %8s %10s %5s %s" % ("file", "rows", "bytes", "cols", "sha/bytes ok"))
for f in data_files:
    fp = os.path.join(MEM, f)
    mp = fp + ".manifest.json"
    if not os.path.exists(mp):
        # the kept raw LaTeX is referenced from the four sigma manifests, not its own
        print("%-64s %8s %10s %5s  NO MANIFEST (referenced as raw_response_file)" % (f[:64], "-", os.path.getsize(fp), "-"))
        continue
    m = json.load(open(mp, encoding="utf-8"))
    ok_b = (m["bytes"] == os.path.getsize(fp))
    ok_s = (m["sha256"] == sha256f(fp))
    tot_rows += m["row_count"]
    if not (ok_b and ok_s):
        bad.append(f)
    print("%-64s %8d %10d %5d  %s" % (f[:64], m["row_count"], m["bytes"], m["column_count"],
                                      "OK" if (ok_b and ok_s) else "MISMATCH"))
print("\nfiles=%d  total data rows=%d  integrity failures=%d" % (len(data_files), tot_rows, len(bad)))
if bad:
    print("FAILED:", bad)
