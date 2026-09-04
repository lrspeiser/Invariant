"""Verbatim transcription of Owers et al. 2017 (MNRAS 468, 1824) Table 1,
the SAMI cluster property table, from the arXiv LaTeX source (1703.00997).

Owers 2017 is NOT in VizieR (checked: J/MNRAS/468/1824 -> "Table or Catalog
not found", and Vizier.find_catalogs found no Owers/SAMI cluster catalogue),
so the arXiv source is the machine-readable route.

Guards against the recorded failure mode of a table split across two table*
environments: the number of parsed rows is asserted against the paper's
stated sample size of eight clusters, the names are checked against the
eight named in the abstract, and the totals for N_mem are checked against
the stated "1935 and 2899 confirmed members within R200 and 2R200".
"""
import hashlib
import json
import os
import re
import tarfile
from datetime import datetime, timezone

import requests

OUT = os.path.dirname(os.path.abspath(__file__))
EPRINT = "https://arxiv.org/e-print/1703.00997"
UA = {"User-Agent": "gravity-research-acquisition/1.0 (mailto:leonard@horizon3.net)"}

# The eight cluster names as given in the Owers 2017 abstract (line 169 of
# clusters.tex).  Used to verify the parsed table, not to build it.
ABSTRACT_NAMES = ["APMCC0917", "A168", "A4038", "EDCC442", "A3880", "A2399", "A119", "A85"]

CANON = {
    "APMCC 917": "APMCC 0917",
    "Abell 168": "Abell 168",
    "Abell 4038": "Abell 4038",
    "EDCC 442": "EDCC 442",
    "Abell 3880": "Abell 3880",
    "Abell 2399": "Abell 2399",
    "Abell 119": "Abell 119",
    "Abell  85": "Abell 85",
}


def sha256_bytes(b):
    h = hashlib.sha256()
    h.update(b)
    return h.hexdigest()


def get_source():
    tgz = os.path.join(OUT, "owers2017_arxiv_1703.00997.tar.gz")
    if not os.path.exists(tgz):
        r = requests.get(EPRINT, timeout=300, headers=UA)
        r.raise_for_status()
        assert r.content[:2] == b"\x1f\x8b", "not a gzip stream"
        with open(tgz, "wb") as f:
            f.write(r.content)
    raw = open(tgz, "rb").read()
    with tarfile.open(tgz, "r:gz") as tf:
        names = tf.getnames()
        texs = [n for n in names if n.endswith(".tex")]
        assert texs == ["clusters.tex"], "unexpected tex files: %r" % texs
        tex = tf.extractfile("clusters.tex").read().decode("utf-8", "replace")
    return tgz, raw, tex, names


def parse_table1(tex):
    # Locate EVERY table* environment, then find the one(s) containing the
    # cluster table.  Do not assume there is exactly one.
    envs = [
        m.group(0)
        for m in re.finditer(r"\\begin\{table\*\}.*?\\end\{table\*\}", tex, re.S)
    ]
    hits = [e for e in envs if "clus_table" in e]
    assert len(hits) == 1, "expected 1 table* with label clus_table, found %d of %d" % (
        len(hits),
        len(envs),
    )
    body = hits[0]

    rows = []
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("%"):          # commented-out superseded rows
            continue
        if "&" not in s:
            continue
        if not re.match(r"^(APMCC|Abell|EDCC)", s):
            continue
        s = s.rstrip()
        s = re.sub(r"\\\\\s*$", "", s)
        cells = [c.strip() for c in s.split("&")]
        assert len(cells) == 11, "row has %d cells, expected 11: %r" % (len(cells), s)
        name = cells[0].strip()

        def pm(x):
            m = re.match(r"^([-\d.]+)\s*\\pm\s*([-\d.]+)$", x.replace("$", "").strip())
            if m:
                return m.group(1), m.group(2)
            return x.replace("$", "").strip(), ""

        sig, e_sig = pm(cells[4])
        mc, e_mc = pm(cells[6])
        mv, e_mv = pm(cells[7])
        nm_r200, nm_2r200 = cells[8].split("/")
        nz_r200, nz_2r200 = cells[9].split("/")
        c_r200, c_2r200 = cells[10].split("/")
        rows.append(
            {
                "Name": CANON[name],
                "Name_as_printed": name,
                "RAdeg": cells[1],
                "DEdeg": cells[2],
                "z_clus": cells[3],
                "sigma_200": sig,
                "e_sigma_200": e_sig,
                "R_200": cells[5],
                "M_200_caustic": mc,
                "e_M_200_caustic": e_mc,
                "M_200_virial": mv,
                "e_M_200_virial": e_mv,
                "N_mem_R200": nm_r200.strip(),
                "N_mem_2R200": nm_2r200.strip(),
                "N_z_R200": nz_r200.strip(),
                "N_z_2R200": nz_2r200.strip(),
                "Compl_R200_pct": c_r200.strip(),
                "Compl_2R200_pct": c_2r200.strip(),
            }
        )
    return rows


def main():
    tgz, raw, tex, names = get_source()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = parse_table1(tex)

    # --- assertions against the paper's own stated numbers -----------------
    assert len(rows) == 8, "parsed %d clusters, paper states 8" % len(rows)
    squashed = {r["Name"].replace(" ", "").replace("Abell", "A").replace("APMCC0917", "APMCC0917") for r in rows}
    want = {n.replace("Abell", "A") for n in ABSTRACT_NAMES}
    assert squashed == want, "name mismatch: %r vs %r" % (squashed, want)
    # The paper's own Section 5.1 text states 1935 / 2899 members within
    # R200 / 2R200.  Summing the published Table 1 gives 1941 / 2901 -- the
    # paper is internally inconsistent by 6 and 2 galaxies (the LaTeX source
    # carries commented-out superseded rows for A168 and A119, so the table
    # was revised at some point).  A silently truncated table would be short
    # by hundreds, so the guard is a tolerance, and the exact discrepancy is
    # recorded in the manifest rather than papered over.
    tot_r200 = sum(int(r["N_mem_R200"]) for r in rows)
    tot_2r200 = sum(int(r["N_mem_2R200"]) for r in rows)
    assert abs(tot_r200 - 1935) <= 10, "N_mem(<R200) sums to %d, paper says 1935" % tot_r200
    assert abs(tot_2r200 - 2899) <= 10, "N_mem(<2R200) sums to %d, paper says 2899" % tot_2r200
    print("N_mem(<R200)  table sum = %d, paper text states 1935 (delta %+d)"
          % (tot_r200, tot_r200 - 1935))
    print("N_mem(<2R200) table sum = %d, paper text states 2899 (delta %+d)"
          % (tot_2r200, tot_2r200 - 2899))
    tot_nz = sum(int(r["N_z_2R200"]) for r in rows)
    print("N_z(<2R200) total = %d (paper abstract: 21,257 reliable redshifts over "
          "the whole survey incl. non-members outside 2R200)" % tot_nz)

    cols = list(rows[0].keys())
    tsv = os.path.join(OUT, "owers2017_table1_clusters.tsv")
    with open(tsv, "w", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(r[c] for c in cols) + "\n")
    tsv_bytes = open(tsv, "rb").read()

    units = {
        "Name": ("-", "Cluster name, canonicalised to the form used in the SAMI DR3 paper"),
        "Name_as_printed": ("-", "Name exactly as printed in the LaTeX source"),
        "RAdeg": ("deg", "J2000 right ascension of the adopted cluster centre"),
        "DEdeg": ("deg", "J2000 declination of the adopted cluster centre"),
        "z_clus": ("-", "Cluster redshift, biweight location of members within 2 R200 [OBSERVABLE]"),
        "sigma_200": ("km/s", "Line-of-sight velocity dispersion of spectroscopic members within R200, biweight scale estimator (Beers+1990) [OBSERVABLE, up to the R200 aperture]"),
        "e_sigma_200": ("km/s", "1-sigma uncertainty on sigma_200"),
        "R_200": ("Mpc", "Virial radius estimate R200 = 0.17 sigma_200 / H(z) Mpc, iterated (Carlberg+1997 singular isothermal sphere) [MODEL-DERIVED: assumes an isothermal, virialised halo]"),
        "M_200_caustic": ("1e14 Msun", "Caustic mass within R200 (Diaferio 1999, F_beta = 0.7 from Serra+2011) [MODEL-DERIVED: assumes spherical symmetry and a calibrated F_beta; a total dynamical mass, i.e. dark-matter-dependent]"),
        "e_M_200_caustic": ("1e14 Msun", "1-sigma statistical uncertainty on the caustic mass (does not include the F_beta systematic)"),
        "M_200_virial": ("1e14 Msun", "Corrected virial mass within R200 (Girardi+1998 prescription, surface-pressure term C ~ 0.19 M_vir) [MODEL-DERIVED: assumes virial equilibrium and spherical symmetry; a total dynamical mass, i.e. dark-matter-dependent]"),
        "e_M_200_virial": ("1e14 Msun", "1-sigma uncertainty on the virial mass"),
        "N_mem_R200": ("-", "Number of spectroscopically confirmed members within R200 [OBSERVABLE count]"),
        "N_mem_2R200": ("-", "Number of spectroscopically confirmed members within 2 R200"),
        "N_z_R200": ("-", "Total number of reliable redshifts within R200"),
        "N_z_2R200": ("-", "Total number of reliable redshifts within 2 R200"),
        "Compl_R200_pct": ("percent", "Spectroscopic completeness within R200 for r_petro < 19.4"),
        "Compl_2R200_pct": ("percent", "Spectroscopic completeness within 2 R200 for r_petro < 19.4"),
    }

    man = {
        "file": "owers2017_table1_clusters.tsv",
        "source_url": EPRINT,
        "source_file_within_archive": "clusters.tex",
        "raw_archive": os.path.basename(tgz),
        "raw_archive_sha256": sha256_bytes(raw),
        "raw_archive_bytes": len(raw),
        "raw_archive_members": len(names),
        "retrieved_utc": ts,
        "sha256": sha256_bytes(tsv_bytes),
        "bytes": len(tsv_bytes),
        "row_count": len(rows),
        "column_count": len(cols),
        "columns": [{"name": c, "unit": units[c][0], "description": units[c][1]} for c in cols],
        "query": "HTTP GET %s ; tar -xzf ; parse the unique table* environment "
                 "labelled clus_table in clusters.tex" % EPRINT,
        "extraction": "Verbatim transcription of the published LaTeX table. No unit "
                      "conversion, no derivation, no join. Commented-out superseded "
                      "rows for Abell 168 and Abell 119 (LaTeX '%' lines carrying "
                      "slightly different sigma_200 values) were skipped; the live "
                      "rows are the published ones.",
        "assertions_passed": [
            "exactly one table* environment carries the label clus_table",
            "8 data rows parsed; paper states 8 clusters",
            "cluster names match the eight named in the abstract",
            "sum(N_mem < R200) within 10 of the 1935 stated in Section 5.1",
            "sum(N_mem < 2 R200) within 10 of the 2899 stated in the abstract",
            "every row has exactly 11 ampersand-separated cells",
        ],
        "INTERNAL_INCONSISTENCY_IN_SOURCE": (
            "Summing the published Table 1 gives 1941 members within R200 and 2901 "
            "within 2 R200, whereas the paper's abstract and Section 5.1 state 1935 "
            "and 2899.  The transcription here reproduces the TABLE, which is the "
            "per-cluster breakdown actually needed.  The LaTeX source carries "
            "commented-out superseded rows for Abell 168 (192/276 live vs 195/279 "
            "commented) and Abell 119 (372/578 live vs 370/576 commented), so the "
            "table appears to have been revised without the running text being "
            "re-summed.  Neither variant reproduces 1935/2899 exactly."
        ),
        "reference": "Owers et al. 2017, MNRAS 468, 1824, 'The SAMI Galaxy Survey: "
                     "cluster observations and the effect of environment', Table 1",
        "PROVENANCE_WARNING": (
            "sigma_200 and z_clus are measured from member galaxy redshifts and are "
            "OBSERVABLES (sigma_200 only up to the choice of the R200 aperture, which "
            "is itself model-derived). R_200, M_200_caustic and M_200_virial are all "
            "MODEL-DERIVED total dynamical masses/radii that assume dynamical "
            "equilibrium and spherical symmetry, and are therefore dark-matter "
            "dependent. Under the well-network brief they may be used ONLY to RANK "
            "environments, never as observations. Consequently R/R200 from "
            "InputCatClustersDR3 is also model-normalised: the projected separation "
            "is observable, the R200 division is not."
        ),
        "cross_check": (
            "Croom et al. 2021 (SAMI DR3) Table 3 reprints RA, Dec, z_clus, "
            "sigma_200, R_200 and the virial M_200 for the same eight clusters; "
            "every value agrees with this transcription."
        ),
    }
    with open(tsv + ".manifest.json", "w") as f:
        json.dump(man, f, indent=1)

    for r in rows:
        print("%-11s z=%-7s sigma=%4s+-%-3s R200=%-5s Mcaus=%-5s Mvir=%-5s Nmem=%s/%s"
              % (r["Name"], r["z_clus"], r["sigma_200"], r["e_sigma_200"], r["R_200"],
                 r["M_200_caustic"], r["M_200_virial"], r["N_mem_R200"], r["N_mem_2R200"]))
    print("wrote", tsv, len(rows), "rows,", len(cols), "cols")


if __name__ == "__main__":
    main()
