"""JOB2 satellites lane: VizieR acquisition with per-file manifests."""
import os
import sys
import json

HERE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data\raw\streams-satellites"
sys.path.insert(0, HERE)
from _manifest import write_manifest, sha256_of, http_get, assert_vizier_tsv  # noqa: E402

BASE = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv?-source=%s&-out.all&-out.max=unlimited"

# (out_name, vizier source, human note, measurement_or_model)
JOBS = [
    # --- McConnachie 2012 Local Group dwarfs ---
    ("sat_mcconnachie2012_table1.tsv", "J/AJ/144/4/table1",
     "McConnachie 2012 AJ 144,4 Table 1: positional data for LG + nearby dwarfs", None),
    ("sat_mcconnachie2012_table2.tsv", "J/AJ/144/4/table2",
     "McConnachie 2012 Table 2", None),
    ("sat_mcconnachie2012_table3.tsv", "J/AJ/144/4/table3",
     "McConnachie 2012 Table 3", None),
    ("sat_mcconnachie2012_table4.tsv", "J/AJ/144/4/table4",
     "McConnachie 2012 Table 4", None),
    ("sat_mcconnachie2012_table5.tsv", "J/AJ/144/4/table5",
     "McConnachie 2012 Table 5", None),
    ("sat_mcconnachie2012_refs.tsv", "J/AJ/144/4/refs",
     "McConnachie 2012 reference list", "provenance"),
    # --- ELVES (Carlsten+2022 ApJ 933,47) ---
    ("sat_elves_table6.tsv", "J/ApJ/933/47/table6", "ELVES Table 6", None),
    ("sat_elves_table7.tsv", "J/ApJ/933/47/table7", "ELVES Table 7", None),
    ("sat_elves_table8.tsv", "J/ApJ/933/47/table8", "ELVES Table 8", None),
    ("sat_elves_table9.tsv", "J/ApJ/933/47/table9", "ELVES Table 9", None),
    ("sat_elves_table10.tsv", "J/ApJ/933/47/table10", "ELVES Table 10", None),
    # --- ELVES II (Carlsten+2022 ApJ 927,44) ---
    ("sat_elves2_table4.tsv", "J/ApJ/927/44/table4",
     "ELVES II early-type satellite photometry + GC data", None),
    # --- Carlsten+2021 ApJ 922,267 dwarf photometry & structures ---
    ("sat_carlsten2021_table1.tsv", "J/ApJ/922/267/table1", "Carlsten+2021 Table 1", None),
    ("sat_carlsten2021_table4.tsv", "J/ApJ/922/267/table4", "Carlsten+2021 Table 4", None),
    ("sat_carlsten2021_table5.tsv", "J/ApJ/922/267/table5", "Carlsten+2021 Table 5", None),
    # --- Martin+2016 PAndAS M31 satellite structural parameters ---
    ("sat_martin2016_table1.tsv", "J/ApJ/833/167/table1", "Martin+2016 PAndAS Table 1", None),
    ("sat_martin2016_table2.tsv", "J/ApJ/833/167/table2", "Martin+2016 PAndAS Table 2", None),
    ("sat_martin2016_table3.tsv", "J/ApJ/833/167/table3", "Martin+2016 PAndAS Table 3", None),
    # --- Collins+2013 M31 dSph kinematics ---
    ("sat_collins2013_table1.tsv", "J/ApJ/768/172/table1", "Collins+2013 Table 1", None),
    ("sat_collins2013_table2.tsv", "J/ApJ/768/172/table2", "Collins+2013 Table 2", None),
    ("sat_collins2013_table3.tsv", "J/ApJ/768/172/table3", "Collins+2013 Table 3", None),
    ("sat_collins2013_table4.tsv", "J/ApJ/768/172/table4", "Collins+2013 Table 4", None),
    # --- Battaglia+2022 Gaia eDR3 systemic motions ---
    ("sat_battaglia2022_pmem.tsv", "J/A+A/657/A54/pmem",
     "Battaglia+2022 A&A 657,A54 member-star probabilities", None),
    # --- SAGA DR2 (Mao+2021) as VizieR fallback for SAGA ---
    ("sat_saga_dr2_table2.tsv", "J/ApJ/907/85/table2", "SAGA II Table 2", None),
    ("sat_saga_dr2_table3.tsv", "J/ApJ/907/85/table3", "SAGA II Table 3", None),
]


def parse_tsv(path):
    """Return (colnames, units, ndata) from a VizieR asu-tsv payload."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().splitlines()
    hdr = None
    for i, l in enumerate(lines):
        if l.startswith("#") or not l.strip():
            continue
        hdr = i
        break
    cols = lines[hdr].split("\t")
    units = lines[hdr + 1].split("\t") if hdr + 1 < len(lines) else [""] * len(cols)
    data = [l for l in lines[hdr + 3:] if l.strip() and not l.startswith("#")]
    return cols, units, len(data)


def main():
    results = {}
    for name, src, note, mm in JOBS:
        dest = os.path.join(HERE, name)
        url = BASE % src
        try:
            http_get(url, dest)
            cat = src.rsplit("/", 1)[0] if "/" in src else src
            assert_vizier_tsv(dest, expect_catalog=cat, min_rows=1)
            cols, units, n = parse_tsv(dest)
            write_manifest(
                dest, url,
                query="GET " + url,
                columns=[{"name": c, "unit": u} for c, u in zip(cols, units)],
                row_count=n,
                note=note,
                measurement_or_model=mm or (
                    "MEASUREMENT (observed positions/photometry/velocities) unless a "
                    "column name indicates a fitted or dynamical mass -- see per-column "
                    "notes in _SECTION_SATELLITES.md"),
                extra={"vizier_source": src, "vizier_catalog": cat,
                       "acquisition_job": "JOB2 streams-satellites"},
            )
            results[name] = {"ok": True, "rows": n, "cols": len(cols),
                             "colnames": cols, "src": src}
        except Exception as e:
            print("FAIL %-40s %s: %s" % (name, type(e).__name__, e))
            results[name] = {"ok": False, "err": "%s: %s" % (type(e).__name__, e),
                             "src": src}
            if os.path.exists(dest):
                os.rename(dest, dest + ".FAILED")
    with open(os.path.join(HERE, "_vizier_fetch_log.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\n==== SUMMARY ====")
    for k, v in results.items():
        if v.get("ok"):
            print("OK   %-38s rows=%-6d cols=%-3d %s" % (k, v["rows"], v["cols"], v["src"]))
        else:
            print("FAIL %-38s %s" % (k, v["err"][:110]))


if __name__ == "__main__":
    main()
