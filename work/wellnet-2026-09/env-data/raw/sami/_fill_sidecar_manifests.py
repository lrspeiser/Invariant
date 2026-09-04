"""Every file in this directory must have its own <name>.manifest.json, per
the well-network brief.  The cleaned TSVs and the raw arXiv tarballs are
already described inside their sibling manifests; this writes the sidecars so
the rule holds literally, each one pointing at the authoritative record.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

OUT = os.path.dirname(os.path.abspath(__file__))


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def nrows(p):
    with open(p, encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f) - 1


PARENT = {
    "croom2021_dr3_arxiv_2101.12224.tar.gz":
        ("croom2021_table3_cluster_targets.tsv.manifest.json",
         "https://arxiv.org/e-print/2101.12224",
         "Unmodified arXiv source tarball for Croom et al. 2021 (SAMI DR3, "
         "MNRAS 505, 991). Contains sami_dr3.tex, from which Table 3 was "
         "transcribed. Kept unmodified as the upstream response."),
    "owers2017_arxiv_1703.00997.tar.gz":
        ("owers2017_table1_clusters.tsv.manifest.json",
         "https://arxiv.org/e-print/1703.00997",
         "Unmodified arXiv source tarball for Owers et al. 2017 (MNRAS 468, "
         "1824). Contains clusters.tex, from which Table 1 was transcribed. "
         "Kept unmodified as the upstream response."),
}


def main():
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    made = []
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f)
        if (not os.path.isfile(p) or f.endswith(".manifest.json")
                or f.startswith("_") or f.endswith(".md")):
            continue
        mp = p + ".manifest.json"
        if os.path.exists(mp):
            continue

        if f in PARENT:
            par, url, note = PARENT[f]
            man = {"file": f, "kind": "raw upstream archive, unmodified",
                   "source_url": url, "retrieved_utc": ts,
                   "sha256": sha(p), "bytes": os.path.getsize(p),
                   "row_count": None, "column_count": None, "columns": [],
                   "query": "HTTP GET " + url,
                   "authoritative_manifest": par, "note": note}
        elif f.endswith(".tsv"):
            par = f[:-4] + ".vot.manifest.json"
            assert os.path.exists(os.path.join(OUT, par)), "no parent for " + f
            pm = json.load(open(os.path.join(OUT, par)))
            man = {"file": f,
                   "kind": "cleaned TSV derived from the sibling VOTable by a "
                           "straight astropy Table -> pandas dump: no unit "
                           "conversion, no cut, no join",
                   "derived_from": pm["file"],
                   "source_url": pm["source_url"], "query": pm["query"],
                   "retrieved_utc": pm["retrieved_utc"],
                   "sha256": sha(p), "bytes": os.path.getsize(p),
                   "row_count": nrows(p), "column_count": pm["column_count"],
                   "columns": pm["columns"],
                   "authoritative_manifest": par}
            assert man["row_count"] == pm["row_count"], \
                "%s: %d rows vs %d in parent" % (f, man["row_count"], pm["row_count"])
        elif f == "sami_inventory_counts.json":
            man = {"file": f,
                   "kind": "DERIVED summary counts, produced by _summary_stats.py "
                           "from sami_dr3_master_galaxy_inventory.tsv. No network "
                           "access, no new measurement.",
                   "derived_from": "sami_dr3_master_galaxy_inventory.tsv",
                   "built_utc": ts, "sha256": sha(p), "bytes": os.path.getsize(p),
                   "row_count": None, "column_count": None, "columns": [],
                   "query": "python _summary_stats.py"}
        else:
            raise AssertionError("no manifest rule for " + f)

        json.dump(man, open(mp, "w"), indent=1)
        made.append(f)

    print("wrote %d sidecar manifests" % len(made))
    for f in made:
        print("  ", f)

    # final audit
    missing = [f for f in sorted(os.listdir(OUT))
               if os.path.isfile(os.path.join(OUT, f))
               and not f.endswith(".manifest.json") and not f.startswith("_")
               and not f.endswith(".md")
               and not os.path.exists(os.path.join(OUT, f + ".manifest.json"))]
    assert not missing, "still unmanifested: %r" % missing
    print("AUDIT PASS: every data file in the directory has a manifest")


if __name__ == "__main__":
    main()
