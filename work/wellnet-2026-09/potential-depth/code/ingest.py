"""Ingest every ladder source into this lane, with manifests and row-count assertions.

Sources and their status:
  A  SPARC rotation curves            local repo, work/gravitylab/data.py
  B  optical groups (J/A+A/690/A52)   local repo, item-04 roadmap source
  C  Sun+2009 43 Chandra groups       arXiv 0805.2320 LaTeX  (NOT in VizieR)
  D  Lovisari+2015 20 XMM groups      arXiv 1409.3845 LaTeX  (NOT in VizieR)
  E  Gonzalez+2013 15 systems         arXiv 1309.3565 LaTeX  (NOT in VizieR)
  F  X-COP 12 clusters                X-COP public release, via the programme's
                                      unified table (588 resolved points)

C and D were parsed on 2026-09-02 by work/.../lane06_groups/parse_tables.py from
arXiv LaTeX that is still on disk.  This module RE-PARSES the LaTeX independently
and asserts agreement, because a silent LaTeX extraction failure is a recorded
failure mode of this programme (a two-`table*` layout once returned 59 of 100 rows
with no error).

SEALED HOLDOUTS: KiDS and wide binaries are never loaded here. The unified table
contains them; they are dropped by probe name at read time and never touched.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
DATA = os.path.join(LANE, "data")
RAW = os.path.join(DATA, "raw")
os.makedirs(RAW, exist_ok=True)

REPO = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration"
SCRATCH = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
           "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad")
LANE06 = SCRATCH + "/lane06_groups"

FORBIDDEN_PROBES = {"weak_lensing", "wide_binary"}   # KiDS + El-Badry: SEALED


def now():
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: str) -> str:
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_manifest(path, *, source_url, exact_query, row_count, columns,
                   note, **extra):
    m = {
        "file": os.path.basename(path),
        "source_url": source_url,
        "exact_query": exact_query,
        "retrieved_utc": now(),
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "note": note,
    }
    m.update(extra)
    with open(path + ".manifest.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, indent=2)
    return m


def copy_in(src, name):
    dst = os.path.join(DATA, name)
    shutil.copyfile(src, dst)
    return dst


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


# --------------------------------------------------------------------------
# C. Sun+2009 -- independent re-parse of the arXiv LaTeX, for verification
# --------------------------------------------------------------------------
_NUM = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _clean(tok):
    """Strip LaTeX; return (value, flag). flag S = scaling relation (parens),
    E = tier-2 extrapolated (asterisk)."""
    t = tok.strip()
    flag = ""
    if "*" in t:
        flag += "E"
    if "(" in t and ")" in t:
        flag += "S"
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = t.replace("$", "").replace("{", "").replace("}", "").replace("*", "")
    t = t.replace("(", "").replace(")", "")
    m = _NUM.search(t)
    return (float(m.group()) if m else None), flag


def reparse_sun2009():
    """Independently count the data rows of Sun+2009 Table 6 in ms.tex.

    The table is a plain `tabular` (NOT a deluxetable, and NOT AASTeX
    startdata/enddata -- assuming otherwise silently returns the whole file).
    Structure: \\caption{...} ... \\begin{tabular}{...} \\hline \\hline
    <header row> \\\\ <units row> \\\\ \\hline <data rows> \\hline \\hline
    \\end{tabular}.  Data rows are counted as backslash-backslash-terminated
    records between the first single \\hline after the header and the closing
    \\hline \\hline.
    """
    p = os.path.join(LANE06, "arxiv", "0805.2320", "ms.tex")
    src = open(p, encoding="utf-8", errors="replace").read()
    bs2 = chr(92) + chr(92)
    out = []
    i = src.find("Derived properties")
    while i != -1:
        cap = src[i:i + 70].replace("\n", " ")
        tb = src.find("begin{tabular}", i)
        te = src.find("end{tabular}", tb)
        seg = src[tb:te]
        # data block starts after the LAST \hline that precedes the first data
        # row, i.e. after the header/units rows
        h = seg.find("hline", seg.find("hline", seg.find("hline") + 1) + 1)
        # closing double hline
        tail = seg.rfind("hline")
        body = seg[h + len("hline"):tail]   # +len: else the first data row is
        #  swallowed into a fragment that still contains "hline" and is filtered
        rows = [r for r in body.split(bs2) if r.strip() and "hline" not in r]
        out.append((cap, len(rows)))
        i = src.find("Derived properties", i + 10)
    return out


# --------------------------------------------------------------------------
# E. Gonzalez+2013 -- parse the two deluxetables (WMAP cosmology tables only)
# --------------------------------------------------------------------------
def parse_gonzalez():
    p = os.path.join(RAW, "gonzalez2013", "ms.tex")
    src = open(p, encoding="utf-8", errors="replace").read()

    def block(caption):
        i = src.find(caption)
        assert i != -1, caption
        j = src.find("startdata", i)
        k = src.find("enddata", j)
        return src[j + len("startdata"):k]

    def rows_of(body):
        """Split on \\\\ then STRIP \\hline separators from each fragment.

        Doing the naive `if "hline" in line: continue` silently drops the first
        row of every block that follows a rule -- here that lost 2 of 15 rows
        (Abell 0478 after the \\hline, and Abell 2390 which has no trailing
        \\\\ before \\enddata). Both are recovered by cleaning, not skipping.
        """
        out = []
        for line in body.split(chr(92) + chr(92)):
            line = line.replace(chr(92) + "hline", " ")
            # drop whole-line comments FIRST: the block opens with a
            # "%cluster z T ..." legend glued to the first data row, and
            # testing startswith("%") before stripping it loses that row.
            line = "\n".join(l for l in line.splitlines()
                             if not l.strip().startswith("%")).strip()
            if not line:
                continue
            cells = [c.strip() for c in line.split("&")]
            if len(cells) < 3 or not cells[0].startswith("Abell"):
                continue
            out.append(cells)
        return out

    props = rows_of(block("Observed Cluster Properties}"))
    fr = rows_of(block("Derived Mass Fractions ($r<"))
    assert len(props) == 15, f"Gonzalez properties rows {len(props)} != 15"
    assert len(fr) == 15, f"Gonzalez fraction rows {len(fr)} != 15"

    fracs = {}
    for c in fr:
        name = " ".join(c[0].split())
        fg, _ = _clean(c[1])
        fs, _ = _clean(c[2]) if "---" not in c[2] else (None, "")
        fracs[name] = (fg, fs)

    out = []
    for c in props:
        name = " ".join(c[0].split())
        z, _ = _clean(c[1])
        T, _ = _clean(c[2])
        r500, _ = _clean(c[5])            # Mpc
        M500, _ = _clean(c[6])            # 1e14 Msun
        Mgas, _ = _clean(c[7])            # 1e13 Msun
        Mst3d = None
        if "---" not in c[9]:
            Mst3d, _ = _clean(c[9])       # 1e13 Msun, deprojected
        fg, fs = fracs.get(name, (None, None))
        out.append(dict(name=name, z=z, kT_keV=T, r500_Mpc=r500,
                        M500_1e14=M500, Mgas500_1e13=Mgas,
                        Mstar3d500_1e13=Mst3d, fgas=fg, fstar=fs))
    return out


def main():
    made = []

    # ---- A. SPARC ---------------------------------------------------------
    sparc_src = (REPO + "/runs/gravity/roadmap/"
                 "item-02-shape-anisotropy-v1-source/sparc_table1.tsv")
    print(f"SPARC table1 present: {os.path.exists(sparc_src)}")

    # ---- B. optical groups -----------------------------------------------
    gsrc = (REPO + "/runs/gravity/roadmap/"
            "item-04-baryonic-compactness-v1-source/group-features.tsv")
    g = read_tsv(gsrc)
    assert len(g) == 52, f"optical groups {len(g)} != 52"
    dst = copy_in(gsrc, "optical_groups_features.tsv")
    write_manifest(
        dst,
        source_url=("https://vizier.cds.unistra.fr/viz-bin/asu-tsv?"
                    "-source=J/A%2BA/690/A52/tablec4&Group=<id>"
                    "&-out=Group,GalID,SpecObjID,RAJ2000,DEJ2000,zsp,Lr"
                    "&-out.max=unlimited"),
        exact_query="copied unmodified from " + gsrc,
        row_count=len(g), columns=[{"name": k, "unit": ""} for k in g[0]],
        note=("52 SDSS spectroscopic groups, members pulled one group at a time "
              "from VizieR J/A+A/690/A52/tablec4 (912 member rows). Provenance "
              "and per-group SHA-256 in the sibling sample-manifest.json of the "
              "roadmap item-04 source directory. total_mass_msun is a STELLAR "
              "mass from r-band luminosity: hot gas is NOT included, which at "
              "group scale is the dominant baryon reservoir. sigma_gap / "
              "sigma_mad are gapper / MAD velocity dispersions from 10-34 "
              "members and are noisy."),
        upstream_manifest=(REPO + "/runs/gravity/roadmap/"
                           "item-04-baryonic-compactness-v1-source/"
                           "sample-manifest.json"),
        member_rows=912, groups=52)
    made.append(("optical_groups_features.tsv", len(g)))

    # ---- C. Sun+2009 ------------------------------------------------------
    s = read_tsv(LANE06 + "/derived/sun2009_groups.tsv")
    assert len(s) == 43, f"Sun+2009 rows {len(s)} != 43"
    caps = reparse_sun2009()
    print("  Sun+2009 independent LaTeX re-parse of 'Derived properties' blocks:")
    for cap, n in caps:
        print(f"    {n:3d} rows  <- {cap}")
    assert any(n == 43 for _, n in caps), \
        f"independent re-parse did not find a 43-row block: {caps}"
    dst = copy_in(LANE06 + "/derived/sun2009_groups.tsv", "sun2009_groups.tsv")
    write_manifest(
        dst,
        source_url="https://arxiv.org/e-print/0805.2320",
        exact_query=("GET https://arxiv.org/e-print/0805.2320 ; tar xzf ; parse "
                     "ms.tex deluxetable 'Derived properties of groups I' "
                     "(Table 6 of Sun et al. 2009, ApJ 693, 1142)"),
        row_count=len(s), columns=[{"name": k, "unit": ""} for k in s[0]],
        note=("43 Chandra galaxy groups. Cosmology H0=73, OmegaM=0.24. "
              "M500_Msun and M2500_Msun are HYDROSTATIC X-ray total masses; "
              "Mgas*_Msun = fgas * M. FLAGS: _flag='S' means the value came "
              "from a scaling relation not a measurement, 'E' means tier-2 "
              "extrapolated beyond the detected radius. r2500 measured for all "
              "43; r500 measured for only 23. NOT in VizieR "
              "(J/ApJ/693/1142 -> 'Table or Catalog not found')."),
        parsed_by=(LANE06 + "/parse_tables.py (2026-09-02)"),
        independent_reparse="43-row block confirmed in ms.tex by ingest.py",
        latex_sha256=sha256_file(LANE06 + "/arxiv/0805.2320/ms.tex"),
        tarball_sha256=sha256_file(LANE06 + "/arxiv/0805.2320.tar.gz"))
    made.append(("sun2009_groups.tsv", len(s)))

    # ---- D. Lovisari+2015 -------------------------------------------------
    lo = read_tsv(LANE06 + "/derived/lovisari2015_groups.tsv")
    assert len(lo) == 20, f"Lovisari+2015 rows {len(lo)} != 20"
    tex = open(LANE06 + "/arxiv/1409.3845/Lovisari_groups.tex",
               encoding="utf-8", errors="replace").read()
    for nm in [r["name"] for r in lo]:
        assert nm in tex.replace(" ", ""), f"{nm} not found in Lovisari LaTeX"
    dst = copy_in(LANE06 + "/derived/lovisari2015_groups.tsv",
                  "lovisari2015_groups.tsv")
    write_manifest(
        dst,
        source_url="https://arxiv.org/e-print/1409.3845",
        exact_query=("GET https://arxiv.org/e-print/1409.3845 ; tar xzf ; parse "
                     "Lovisari_groups.tex Table 3 (Lovisari, Reiprich & "
                     "Schellenberger 2015, A&A 573, A118)"),
        row_count=len(lo), columns=[{"name": k, "unit": ""} for k in lo[0]],
        note=("20 XMM-Newton galaxy groups. Cosmology H0=70, OmegaM=0.27. "
              "M500/M2500 are HYDROSTATIC X-ray total masses (paper tabulates "
              "1e13 Msun, converted here); Mgas in 1e12 Msun, converted. Both "
              "radii tabulated for all 20, so the sample is homogeneous. "
              "NOT in VizieR (J/A+A/573/A118 -> not found)."),
        parsed_by=(LANE06 + "/parse_tables.py (2026-09-02)"),
        independent_check="all 20 object names re-found in the arXiv LaTeX",
        latex_sha256=sha256_file(LANE06 + "/arxiv/1409.3845/Lovisari_groups.tex"),
        tarball_sha256=sha256_file(LANE06 + "/arxiv/1409.3845.tar.gz"))
    made.append(("lovisari2015_groups.tsv", len(lo)))

    # ---- E. Gonzalez+2013 -------------------------------------------------
    gz = parse_gonzalez()
    cols = ["name", "z", "kT_keV", "r500_Mpc", "M500_1e14", "Mgas500_1e13",
            "Mstar3d500_1e13", "fgas", "fstar"]
    dst = os.path.join(DATA, "gonzalez2013_baryons.tsv")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(cols) + "\n")
        for r in gz:
            f.write("\t".join("" if r[c] is None else str(r[c])
                              for c in cols) + "\n")
    units = ["name", "redshift", "keV", "Mpc", "1e14 Msun", "1e13 Msun",
             "1e13 Msun", "dimensionless", "dimensionless"]
    write_manifest(
        dst,
        source_url="https://arxiv.org/e-print/1309.3565",
        exact_query=("GET https://arxiv.org/e-print/1309.3565 ; tar xzf ; parse "
                     "ms.tex deluxetables 'Observed Cluster Properties' and "
                     "'Derived Mass Fractions (r<r500)' (Gonzalez, Sivanandam, "
                     "Zabludoff & Zaritsky 2013, ApJ 778, 14)"),
        row_count=len(gz), columns=[{"name": c, "unit": u}
                                    for c, u in zip(cols, units)],
        note=("15 systems, kT 2.06-10.6 keV. M500 and r500 are HYDROSTATIC "
              "X-ray. Mgas500 measured from XMM. Mstar3d500 is the "
              "DEPROJECTED stellar mass (BCG + ICL + satellites) from "
              "Gonzalez+2007 photometry -- this is the only source in this "
              "lane with a DIRECTLY MEASURED stellar mass, and it is what "
              "calibrates the stellar term for Sun+2009 and Lovisari+2015. "
              "The last 3 rows (A0478, A2029, A2390) have X-ray only and no "
              "photometry, so Mstar/fstar are blank BY DESIGN, not by "
              "extraction failure. WMAP cosmology tables used, not the Planck "
              "duplicates that appear later in the same file. "
              "NOT in VizieR (J/ApJ/778/14 -> not found)."),
        rows_with_stellar_mass=sum(1 for r in gz if r["Mstar3d500_1e13"]),
        tarball_sha256=sha256_file(RAW + "/gonzalez2013_1309.3565.tar.gz"))
    made.append(("gonzalez2013_baryons.tsv", len(gz)))

    # ---- F. X-COP + SPARC resolved profiles, from the unified table -------
    up = SCRATCH + "/unified_gravity_table_v2.csv"
    with open(up, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    kept = [r for r in rows if r["probe"] not in FORBIDDEN_PROBES]
    dropped = len(rows) - len(kept)
    kept = [r for r in kept if r["probe"] in ("hydrostatic", "rotation_curve")]
    dst = os.path.join(DATA, "resolved_profiles_sparc_xcop.csv")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(kept)
    nx = sum(1 for r in kept if r["probe"] == "hydrostatic")
    ns = sum(1 for r in kept if r["probe"] == "rotation_curve")
    assert nx == 588 and ns == 3389, (nx, ns)
    write_manifest(
        dst,
        source_url=("X-COP public data release "
                    "https://dominiqueeckert.wixsite.com/xcop/data ; "
                    "SPARC Lelli, McGaugh & Schombert 2016 AJ 152, 157"),
        exact_query=("subset of " + up + " : probe in {hydrostatic, "
                     "rotation_curve}; probes weak_lensing and wide_binary "
                     "DROPPED UNREAD as permanent sealed holdouts"),
        row_count=len(kept),
        columns=[{"name": k, "unit": u} for k, u in zip(
            rows[0].keys(),
            ["id", "type", "kpc", "m/s^2", "m/s^2", "ratio", "ratio",
             "class", "text", "text"])],
        note=("Resolved g_bar(r) and g_obs(r). hydrostatic = 12 X-COP "
              "clusters, 588 points, 121-1640 kpc, XMM+Planck; g_bar from "
              "deprojected n_e plus measured stellar profiles for 7 of 12 "
              "(10% of gas assumed for the other 5); g_obs from the "
              "hydrostatic equation. rotation_curve = 175 SPARC galaxies, "
              "3389 points, g_bar at Upsilon*=0.5 disk / 0.7 bulge. "
              f"{dropped} rows of the source table were dropped unread "
              "because they are SEALED HOLDOUTS (KiDS weak lensing, wide "
              "binaries); ephemeris and strong-lensing rows were also dropped "
              "as out of scope for this lane."),
        source_table_sha256=sha256_file(up),
        source_table_rows=len(rows),
        sealed_holdout_rows_dropped_unread=dropped,
        build_scripts=[SCRATCH + "/build_unified.py",
                       SCRATCH + "/extend_unified.py",
                       SCRATCH + "/build_xcop.py"])
    made.append(("resolved_profiles_sparc_xcop.csv", len(kept)))

    print("\ningested:")
    for n, c in made:
        print(f"  {c:6d} rows  {n}")
    return made


if __name__ == "__main__":
    main()
