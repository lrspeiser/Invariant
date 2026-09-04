"""Verbatim transcription of Croom et al. 2021 (SAMI DR3, MNRAS 505, 991)
Table 3 -- the eight cluster regions with their observed/good/all target counts
-- from the arXiv source 2101.12224.

Two jobs:
  1. Give the lane the per-cluster observed-target counts and the second
     (northern) pointing centre used for Abell 168, neither of which is in
     Owers+2017 Table 1 or in any Data Central table.
  2. Serve as an INDEPENDENT cross-check that the Owers+2017 sigma_200, R200
     and virial M200 transcription is right.  The script asserts agreement.
"""
import hashlib
import json
import os
import re
import tarfile
from datetime import datetime, timezone

import requests

OUT = os.path.dirname(os.path.abspath(__file__))
EPRINT = "https://arxiv.org/e-print/2101.12224"
UA = {"User-Agent": "gravity-research-acquisition/1.0 (mailto:leonard@horizon3.net)"}

CANON = {"APMCC 917": "APMCC 0917", "Abell 168": "Abell 168",
         "Abell 4038": "Abell 4038", "EDCC 442": "EDCC 442",
         "Abell 3880": "Abell 3880", "Abell 2399": "Abell 2399",
         "Abell 119": "Abell 119", "Abell  85": "Abell 85"}


def sha(b):
    h = hashlib.sha256(); h.update(b); return h.hexdigest()


def main():
    tgz = os.path.join(OUT, "croom2021_dr3_arxiv_2101.12224.tar.gz")
    if not os.path.exists(tgz):
        r = requests.get(EPRINT, timeout=600, headers=UA)
        r.raise_for_status()
        assert r.content[:2] == b"\x1f\x8b"
        open(tgz, "wb").write(r.content)
    raw = open(tgz, "rb").read()
    with tarfile.open(tgz, "r:gz") as tf:
        tex = tf.extractfile("sami_dr3.tex").read().decode("utf-8", "replace")

    envs = [m.group(0) for m in
            re.finditer(r"\\begin\{table\*\}.*?\\end\{table\*\}", tex, re.S)]
    hits = [e for e in envs if "table:clusters" in e]
    assert len(hits) == 1, "found %d cluster tables among %d" % (len(hits), len(envs))
    body = hits[0]

    rows, extra_centres = [], []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("%") or "&" not in s:
            continue
        if not re.match(r"^(APMCC|Abell|EDCC)", s):
            continue
        s = re.sub(r"\\\\\s*$", "", s.rstrip())
        cells = [c.strip() for c in s.split("&")]
        if "(N)" in cells[0]:                 # second pointing centre for A168
            extra_centres.append({"name": "Abell 168 (N)", "RAdeg": cells[1],
                                  "DEdeg": cells[2]})
            continue
        assert len(cells) == 10, "row has %d cells: %r" % (len(cells), s)
        po, pg, pa = cells[7].split("/")
        so, sg, sa = cells[8].split("/")
        rows.append({
            "Name": CANON[cells[0]], "RAdeg": cells[1], "DEdeg": cells[2],
            "z_clus": cells[3], "sigma_200": cells[4], "R_200": cells[5],
            "M_200_virial": cells[6],
            "Primary_N_obs": po.strip(), "Primary_N_good": pg.strip(),
            "Primary_N_all": pa.strip(),
            "Secondary_N_obs": so.strip(), "Secondary_N_good": sg.strip(),
            "Secondary_N_all": sa.strip(),
            "Primary_completeness_pct": cells[9].replace("\\%", "").replace("%", "").strip(),
        })

    assert len(rows) == 8, "parsed %d clusters" % len(rows)
    assert len(extra_centres) == 1, "expected the single Abell 168 (N) centre"

    n_all = sum(int(r["Primary_N_all"]) + int(r["Secondary_N_all"]) for r in rows)
    assert n_all == 1433, ("target totals sum to %d; InputCatClustersDR3 has 1433"
                           % n_all)
    n_obs = sum(int(r["Primary_N_obs"]) + int(r["Secondary_N_obs"]) for r in rows)
    print("input-catalogue targets  %d  (matches InputCatClustersDR3 row count)" % n_all)
    print("observed cluster targets %d  (Table 3); CubeObs has 896 cluster-region "
          "galaxies with a best cube -- a surplus of %d" % (n_obs, 896 - n_obs))

    # cross-check against the Owers 2017 transcription
    import csv
    ow = {r["Name"]: r for r in csv.DictReader(
        open(os.path.join(OUT, "owers2017_table1_clusters.tsv")), delimiter="\t")}
    for r in rows:
        o = ow[r["Name"]]
        for k in ["RAdeg", "DEdeg", "z_clus", "sigma_200", "R_200"]:
            assert float(r[k]) == float(o[k]), \
                "%s %s: DR3 %s vs Owers %s" % (r["Name"], k, r[k], o[k])
        assert float(r["M_200_virial"]) == float(o["M_200_virial"]), r["Name"]
    print("cross-check: DR3 Table 3 reproduces Owers+2017 Table 1 exactly for "
          "RA, Dec, z_clus, sigma_200, R_200 and virial M_200 in all 8 clusters")

    cols = list(rows[0].keys())
    p = os.path.join(OUT, "croom2021_table3_cluster_targets.tsv")
    with open(p, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")

    units = {
        "Name": ("-", "Cluster name"),
        "RAdeg": ("deg", "J2000 RA of the cluster centre [OBSERVABLE]"),
        "DEdeg": ("deg", "J2000 Dec of the cluster centre [OBSERVABLE]"),
        "z_clus": ("-", "Cluster redshift, biweight location of members [OBSERVABLE]"),
        "sigma_200": ("km/s", "Velocity dispersion of members within R200 [OBSERVABLE]"),
        "R_200": ("Mpc", "R200 = 0.17 sigma_200 / H(z) [MODEL-DERIVED, isothermal sphere]"),
        "M_200_virial": ("1e14 Msun", "Virial mass [MODEL-DERIVED, assumes virial equilibrium; RANK ONLY]"),
        "Primary_N_obs": ("-", "Primary targets observed by SAMI"),
        "Primary_N_good": ("-", "Primary targets in the input catalogue not flagged bad"),
        "Primary_N_all": ("-", "All primary targets in the input catalogue"),
        "Secondary_N_obs": ("-", "Secondary targets observed"),
        "Secondary_N_good": ("-", "Secondary targets not flagged bad"),
        "Secondary_N_all": ("-", "All secondary targets"),
        "Primary_completeness_pct": ("percent", "Primary N_obs / N_good"),
    }
    man = {
        "file": os.path.basename(p),
        "source_url": EPRINT,
        "source_file_within_archive": "sami_dr3.tex",
        "raw_archive": os.path.basename(tgz),
        "raw_archive_sha256": sha(raw),
        "raw_archive_bytes": len(raw),
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha(open(p, "rb").read()),
        "bytes": os.path.getsize(p),
        "row_count": len(rows),
        "column_count": len(cols),
        "columns": [{"name": c, "unit": units[c][0], "description": units[c][1]}
                    for c in cols],
        "query": "HTTP GET %s ; tar -xzf ; parse the unique table* labelled "
                 "table:clusters in sami_dr3.tex" % EPRINT,
        "extraction": "Verbatim transcription. The 'Abell 168 (N)' partial row, which "
                      "carries only a second pointing centre, is recorded separately "
                      "below and not as a cluster row.",
        "abell_168_second_centre": extra_centres[0],
        "assertions_passed": [
            "exactly one table* labelled table:clusters",
            "8 cluster rows plus 1 partial row for the Abell 168 northern centre",
            "Primary_N_all + Secondary_N_all sums to 1433, the InputCatClustersDR3 row count",
            "RA, Dec, z_clus, sigma_200, R_200 and virial M_200 agree exactly with "
            "the independent Owers+2017 Table 1 transcription for all 8 clusters",
        ],
        "reference": "Croom et al. 2021, MNRAS 505, 991, Table 3",
        "note": "Observed-target total from this table is %d, whereas CubeObs lists "
                "896 unique cluster-region galaxies with a best cube. The %d-galaxy "
                "surplus is cluster-region objects with cubes that are not counted "
                "as observed primary or secondary targets in Table 3."
                % (n_obs, 896 - n_obs),
    }
    json.dump(man, open(p + ".manifest.json", "w"), indent=1)
    print("wrote", p, len(rows), "rows")


if __name__ == "__main__":
    main()
