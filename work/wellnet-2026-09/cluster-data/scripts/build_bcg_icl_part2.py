# -*- coding: utf-8 -*-
"""Build BCG (Product 2) and ICL (Product 3) data products -- part 2.
Sources: Jimenez-Teja+2018, de Oliveira+2022, DeMaio+2018, Shipley+2018."""
import os, re, json, hashlib, datetime

ROOT = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
BS   = chr(92)

def sha256(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def manifest(path, source_url, exact_query, row_count, columns, extraction, note,
             raw_response_file, extra=None):
    m = {
        "file": os.path.basename(path),
        "source_url": source_url,
        "exact_query": exact_query,
        "retrieved_utc": NOW,
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "row_count": row_count,
        "column_count": len(columns),
        "columns": columns,
        "extraction": extraction,
        "note": note,
        "raw_response_file": raw_response_file,
    }
    if extra:
        m.update(extra)
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    print("  manifest ->", os.path.basename(path) + ".manifest.json")

def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join("" if v is None else str(v) for v in r) + "\n")
    print("WROTE", os.path.basename(path), "rows=", len(rows))

def clean_cell(s):
    s = s.strip()
    s = s.replace(BS + BS, "").strip()
    s = s.replace("$", "")
    s = s.replace(BS + "pm", "+/-")
    s = s.replace(BS + "phantom", "")
    s = re.sub(BS + r"[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def latex_rows(text):
    for ln in text.split("\n"):
        L = ln.strip()
        if not L or L.startswith("%"):
            continue
        if L.startswith(BS + "hline"):
            continue
        if "&" not in L:
            continue
        yield [clean_cell(c) for c in L.split("&")]

def sup_sub(s):
    """Convert LaTeX ^{a}_{b} asymmetric errors to +a/-b form (macros already stripped)."""
    return s

print("=" * 70)
print("PRODUCT 2 / 3 build -- part 2   ", NOW)
print("=" * 70)

# =====================================================================
# 4. JIMENEZ-TEJA et al. 2018 (arXiv:1803.04981 = ApJ 857, 79) -- ICL
# =====================================================================
print("\n[4] Jimenez-Teja+2018 ICL fractions (CICLE)")
p = os.path.join(ROOT, "icl", "raw", "jimenez_teja2018_arXiv1803.04981.tex")
tex = open(p, encoding="utf-8", errors="replace").read()
# The file has several deluxetables. Anchor on the ICL-fraction tablecaption, then take the
# FIRST startdata block that follows it.
anchor = tex.find("Results yielded by CICLE")
assert anchor > 0, "ICL-fraction tablecaption not found"
m = re.search(r"\\startdata(.*?)\\enddata", tex[anchor:], re.S)
assert m, "no startdata block after the ICL-fraction tablecaption"
body = m.group(1)
assert "A2744" in body and "MACS0717" in body, "anchored block does not look like the ICL-fraction table"
hdr = ["Cluster", "fICL_F435W_pct", "R_F435W_kpc", "fICL_F606W_pct", "R_F606W_kpc",
       "fICL_F625W_pct", "R_F625W_kpc", "fICL_F814W_pct", "R_F814W_kpc"]
rows = []
for cells in latex_rows(body):
    if len(cells) != 9:
        continue
    rows.append(cells)
assert len(rows) == 11, "JT2018 expected 11 clusters (tablecaption: 'the 11 clusters in our sample'), got %d" % len(rows)
out = os.path.join(ROOT, "icl", "jimenez_teja2018_icl_fractions_cicle.tsv")
write_tsv(out, hdr, rows)
manifest(out, "https://arxiv.org/e-print/1803.04981",
    "GET https://arxiv.org/e-print/1803.04981 -> tar.gz -> YJimenez-Teja.tex -> deluxetable startdata block",
    len(rows),
    [{"name": "Cluster", "unit": ""}] +
    [{"name": h, "unit": ("percent" if h.startswith("fICL") else "kpc")} for h in hdr[1:]],
    "arXiv source tarball; the single deluxetable startdata/enddata block was parsed. Row count asserted ==11 "
    "against the paper's own tablecaption ('the 11 clusters in our sample'). arXiv id VERIFIED by title match: "
    "'Unveiling the dynamical state of massive clusters through the ICL fraction'.",
    "*** PRODUCT 3 (ICL fraction). Jimenez-Teja et al. 2018, ApJ 857, 79. "
    "TARGET CLUSTERS PRESENT: A2744, MACS0416, MACS0717, MACS1149. "
    "NOT present: Abell 370, Abell S1063 (the paper states neither was in this sample), Abell 2029. "
    "BANDS: HST F435W, F606W, F625W, F814W (observed frame; empty cell = filter not observed for that cluster). "
    "ICL/BCG SEPARATION METHOD: CICLE (Chebyshev-Fourier Intracluster Light Estimator) -- a wavelet/CHEF "
    "(Chebyshev-Fourier orthonormal basis) decomposition that models and removes the BCG without assuming a "
    "parametric profile for it. This is a fully mathematical BCG/ICL disentanglement with NO surface-brightness "
    "threshold and NO fixed radius cut, so these fractions are NOT directly comparable to the SB-cut fractions "
    "of Montes & Trujillo 2018. "
    "RADIAL RANGE: per-filter, given in the R_*_kpc columns -- the ICL is integrated out to the radius where the "
    "azimuthally averaged ICL flux profile reaches its minimum (beyond which instrumental/background light dominates). "
    "Ranges 63.6-626.3 kpc across the sample. "
    "SB LIMIT: set by the parent imaging depth, not by a chosen cut -- HFF images (A2744, MACS0416, MACS0717, "
    "MACS1149) reach ~28.8 AB mag (F435W, F606W) and ~29.1 AB mag (F814W) for a 5-sigma point source in a "
    "0.4 arcsec diameter aperture; CLASH images reach ~27.2 (F435W), ~27.6 (F606W), ~27.7 (F814W) AB mag. "
    "MASS-MODEL DEPENDENCE: none -- purely photometric. Cluster membership uses SPECTROSCOPIC redshifts; the "
    "paper estimates this can underestimate the total cluster luminosity by up to ~19.7% for the worst cluster, "
    "adding at most ~4.39% to the quoted ICL fractions. "
    "Denominator convention: ICL fraction = ICL light / total cluster light (ICL + all cluster members INCLUDING the BCG).",
    "icl/raw/jimenez_teja2018_arXiv1803.04981.tex")

# =====================================================================
# 5. de OLIVEIRA et al. 2022 (arXiv:2202.08289) -- ICL for A370 & AS1063
# =====================================================================
print("\n[5] de Oliveira+2022 ICL fractions A370 / AS1063 (CICLE)")
p = os.path.join(ROOT, "icl", "raw", "deoliveira2022_arXiv2202.08289.tex")
tex = open(p, encoding="utf-8", errors="replace").read()
envs = re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", tex, re.S)
body = None
for e in envs:
    if "ICL fractions and ICL radius" in e:
        body = e
assert body, "de Oliveira ICL table not found"
hdr = ["Cluster", "fICL_F435W_pct", "R_F435W_kpc", "fICL_F606W_pct", "R_F606W_kpc",
       "fICL_F814W_pct", "R_F814W_kpc"]
rows = []
for cells in latex_rows(body):
    if len(cells) != 7:
        continue
    if not cells[0].startswith("Abell"):
        continue
    rows.append(cells)
assert len(rows) == 2, "de Oliveira expected 2 clusters, got %d" % len(rows)
out = os.path.join(ROOT, "icl", "deoliveira2022_icl_fractions_A370_AS1063.tsv")
write_tsv(out, hdr, rows)
manifest(out, "https://arxiv.org/e-print/2202.08289",
    "GET https://arxiv.org/e-print/2202.08289 -> tar.gz -> ICL_on_A370_and_AS1063.tex -> table* env "
    "captioned 'ICL fractions and ICL radius measured for A370 and AS1063'",
    len(rows),
    [{"name": "Cluster", "unit": ""}] +
    [{"name": h, "unit": ("percent" if h.startswith("fICL") else "kpc")} for h in hdr[1:]],
    "arXiv source tarball; LaTeX table* environment identified by caption text. Row count asserted ==2.",
    "*** PRODUCT 3 (ICL fraction) for ABELL 370 and ABELL S1063 -- the two HFF clusters MISSING from "
    "Jimenez-Teja et al. 2018. de Oliveira, Jimenez-Teja et al. 2022 (MNRAS), 'The intracluster light on "
    "Frontier Fields clusters Abell 370 and Abell S1063'. "
    "BANDS: HST F435W, F606W, F814W (observed frame). "
    "ICL/BCG SEPARATION METHOD: CICLE -- same wavelet/CHEF-based algorithm as Jimenez-Teja+2018, so these two "
    "rows ARE directly comparable to jimenez_teja2018_icl_fractions_cicle.tsv and together the two files give a "
    "homogeneous CICLE ICL fraction for all six HFF clusters. No SB threshold, no radius cut, no parametric "
    "assumption about the BCG or ICL shape. "
    "RADIAL RANGE: per-filter, in the R_*_kpc columns (114.2-331.1 kpc); ICL integrated to the radius where the "
    "ICL flux profile is minimum. "
    "Note the strong band-to-band spread (A370: 7.1% in F435W vs 25.1% in F606W) -- the F435W ICL radius is "
    "roughly a third of the F606W one, so the band difference is largely an aperture difference, not a colour effect. "
    "MASS-MODEL DEPENDENCE: none for the tabulated fractions. (The paper additionally plots f_ICL vs R_200 using "
    "R_200 = 2.57 Mpc for A370 from Lah+2009 and 2.7 Mpc for AS1063 from Sartoris+2020 -- those R_200 values are "
    "dynamical and are NOT part of this table.) "
    "Errors combine photometric error, cluster-membership (spectroscopic completeness) error, and the empirical "
    "CICLE BCG/ICL separation error from 10 mock-image realizations per filter.",
    "icl/raw/deoliveira2022_arXiv2202.08289.tex")

# =====================================================================
# 6. DeMAIO et al. 2018 (arXiv:1710.11313 = MNRAS 474, 3009)
# =====================================================================
print("\n[6] DeMaio+2018 BCG+ICL luminosity / stellar mass")
p = os.path.join(ROOT, "bcg", "raw", "demaio2018_arXiv1710.11313_sm_lum_fit_table.tex")
body = open(p, encoding="utf-8", errors="replace").read()
hdr = ["Cluster", "z", "colour_gradient_F110W_F160W", "best_fit_radius_kpc",
       "L_lt10kpc_1e11Lsun", "L_lt50kpc_1e11Lsun", "L_lt100kpc_1e11Lsun",
       "Mstar_lt10kpc_1e11Msun", "Mstar_lt50kpc_1e11Msun", "Mstar_lt100kpc_1e11Msun"]
rows = []
for cells in latex_rows(body):
    if len(cells) != 10:
        continue
    if cells[0] in ("Cluster", ""):
        continue
    if not re.match(r"^[A-Z]", cells[0]):
        continue
    rows.append(cells)
assert len(rows) == 23, "DeMaio2018 expected 23 systems (paper: 'This sample of 23 groups and clusters'), got %d" % len(rows)
out = os.path.join(ROOT, "bcg", "demaio2018_bcg_icl_luminosity_mass.tsv")
write_tsv(out, hdr, rows)
manifest(out, "https://arxiv.org/e-print/1710.11313",
    "GET https://arxiv.org/e-print/1710.11313 -> tar.gz -> tables/sm_lum_fit_table_chab_july42017.tex "
    "(\\input into the sidewaystable* labelled table:best_fit)",
    len(rows),
    [{"name": "Cluster", "unit": ""}, {"name": "z", "unit": ""},
     {"name": "colour_gradient_F110W_F160W", "unit": "mag/arcsec2 per log(kpc)"},
     {"name": "best_fit_radius_kpc", "unit": "kpc"},
     {"name": "L_lt10kpc_1e11Lsun", "unit": "1e11 Lsun"},
     {"name": "L_lt50kpc_1e11Lsun", "unit": "1e11 Lsun"},
     {"name": "L_lt100kpc_1e11Lsun", "unit": "1e11 Lsun"},
     {"name": "Mstar_lt10kpc_1e11Msun", "unit": "1e11 Msun"},
     {"name": "Mstar_lt50kpc_1e11Msun", "unit": "1e11 Msun"},
     {"name": "Mstar_lt100kpc_1e11Msun", "unit": "1e11 Msun"}],
    "arXiv source tarball; the table body lives in a separate \\input file (tables/sm_lum_fit_table_chab_july42017.tex), "
    "parsed directly. Row count asserted ==23 against the paper text 'This sample of 23 groups and clusters'. "
    "arXiv id VERIFIED by title match: 'Lost but not Forgotten: Intracluster Light in Galaxy Groups and Clusters'. "
    "Asymmetric errors appear as 'a b' after macro stripping, meaning +a/-b as printed in the source "
    "(x^{+a}_{-b}); symmetric errors appear as '+/- e'.",
    "DeMaio et al. 2018, MNRAS 474, 3009. TARGET CLUSTERS PRESENT: RXJ2248 (= Abell S1063 = RXC J2248.7-4431), "
    "MACS0416 (= MACSJ0416.1-2403), MACS1149 (= MACSJ1149.5+2223). "
    "*** MACS0717 IS DELIBERATELY EXCLUDED by the authors: 'we exclude MACS0717 from the final sample because it "
    "is a very dynamic systems of 4 merging clusters with no clear central BCG to which to anchor the radial "
    "profiles'. *** Abell 2744, Abell 370 and Abell 2029 are not in this sample either (not CLASH / not observed). "
    "BAND: HST WFC3-IR F160W for the luminosities (the F110W-F160W colour gradient is the fitted quantity); "
    "stellar masses use a Chabrier IMF and a BC03 SSP colour-M/L relation. "
    "*** ICL/BCG SEPARATION METHOD: NONE. These are BCG+ICL COMBINED quantities integrated in fixed CIRCULAR "
    "APERTURES of r<10, 50 and 100 kpc centred on the BCG. The r<10 kpc aperture is BCG-dominated and the "
    "r<100 kpc aperture contains BCG+ICL; the paper does not decompose them. Treat the aperture difference "
    "(100 kpc minus 10 kpc) as an ICL PROXY only, not as a published ICL measurement. *** "
    "RADIAL RANGE / SB LIMIT: the underlying SB profiles are terminated once 3 consecutive dlog(r)=0.05 bins have "
    ">0.2 mag/arcsec2 uncertainty, which occurs at ~26-27 mag/arcsec2 for the CLASH clusters and ~25-26 "
    "mag/arcsec2 for the groups. "
    "MASS-MODEL DEPENDENCE: the tabulated L and M* are photometric and aperture-based, so NOT mass-model "
    "dependent. (The companion sample table's M500/r500 ARE X-ray derived -- see demaio2018_cluster_sample.tsv.) "
    "Quoted M* uncertainties exclude the systematic M/L (model + IMF) error and assume no M/L gradient in the ICL.",
    "bcg/raw/demaio2018_arXiv1710.11313_sm_lum_fit_table.tex")

# ---- DeMaio 2018 sample table (context: z, kT, M500, r500)
p = os.path.join(ROOT, "bcg", "raw", "demaio2018_arXiv1710.11313_sample_table.tex")
body = open(p, encoding="utf-8", errors="replace").read()
hdr = ["Fullname", "Cluster", "z", "kT_keV", "M500_1e14Msun", "r500_kpc", "HST_Xray_source"]
rows = []
for cells in latex_rows(body):
    if len(cells) != 7:
        continue
    if cells[0] in ("Fullname", ""):
        continue
    rows.append(cells)
assert len(rows) == 23, "DeMaio2018 sample expected 23 rows, got %d" % len(rows)
out = os.path.join(ROOT, "bcg", "demaio2018_cluster_sample.tsv")
write_tsv(out, hdr, rows)
manifest(out, "https://arxiv.org/e-print/1710.11313",
    "GET https://arxiv.org/e-print/1710.11313 -> tar.gz -> tables/sample_table_June2017.tex",
    len(rows),
    [{"name": "Fullname", "unit": ""}, {"name": "Cluster", "unit": ""}, {"name": "z", "unit": ""},
     {"name": "kT_keV", "unit": "keV"}, {"name": "M500_1e14Msun", "unit": "1e14 Msun"},
     {"name": "r500_kpc", "unit": "kpc"}, {"name": "HST_Xray_source", "unit": ""}],
    "arXiv source tarball; \\input table file parsed directly. Row count asserted ==23.",
    "DeMaio et al. 2018 sample definition table -- supplied as the aperture/context companion to "
    "demaio2018_bcg_icl_luminosity_mass.tsv (it maps the short cluster keys to full names and gives r500 so the "
    "10/50/100 kpc apertures can be expressed in r/r500). "
    "*** MASS-MODEL DEPENDENCE FLAG: M500 and r500 here are X-RAY DERIVED -- converted from X-ray temperature kT "
    "using the Vikhlinin et al. 2009 kT-M relation. They are hydrostatic-equivalent masses and carry that "
    "systematic. They are NOT used to define the ICL, only to characterise the clusters. *** "
    "TARGET CLUSTERS PRESENT: RXJ2248 (= Abell S1063), MACS0416, MACS1149. MACS0717 excluded by the authors.",
    "bcg/raw/demaio2018_arXiv1710.11313_sample_table.tex")

# ---- DeMaio 2018 ICL colour profiles -- BOTH split table environments
print("\n[6b] DeMaio+2018 ICL colour profiles (LaTeX split-table trap)")

def parse_colour_blocks(text):
    """Return {cluster: [(log_r, value), ...]} from a DeMaio colour-profile table file."""
    out = {}
    order = []
    cur = None
    for ln in text.split("\n"):
        L = ln.strip()
        if not L:
            continue
        if L.startswith("%"):
            continue                       # commented-out duplicate block -> skip
        if "multicolumn" in L:
            names = re.findall(r"multicolumn\{2\}\{c\}\{([^}]*)\}", L)
            cur = [n.strip() for n in names]
            order.append(cur)
            for n in cur:
                out.setdefault(n, [])
            continue
        if "log(r" in L:
            continue                       # per-block column header
        if L.startswith(BS + "hline"):
            continue
        if "&" not in L or cur is None:
            continue
        cells = [clean_cell(c) for c in L.split("&")]
        for i, name in enumerate(cur):
            r = cells[2 * i] if 2 * i < len(cells) else ""
            v = cells[2 * i + 1] if 2 * i + 1 < len(cells) else ""
            if r and v:
                out[name].append((r, v))
    return out, order

t1 = open(os.path.join(ROOT, "icl", "raw", "demaio2018_arXiv1710.11313_tabular_color_table.tex"),
          encoding="utf-8", errors="replace").read()
t2 = open(os.path.join(ROOT, "icl", "raw", "demaio2018_arXiv1710.11313_tabular_color_table_x2.tex"),
          encoding="utf-8", errors="replace").read()
d1, o1 = parse_colour_blocks(t1)
d2, o2 = parse_colour_blocks(t2)
overlap = set(d1) & set(d2)
assert not overlap, "unexpected cluster overlap between the two split tables: %s" % overlap
prof = dict(d1)
prof.update(d2)
# The paper's sample is 23 systems; assert we recovered all of them from BOTH environments.
assert len(prof) == 23, ("colour profiles: expected 23 clusters across the TWO split table environments "
                         "(%d from table 1 + %d from table 2), got %d -- LaTeX split-table truncation!"
                         % (len(d1), len(d2), len(prof)))
hdr = ["Cluster", "log_r_kpc", "F110W_minus_F160W_mag", "source_table_env"]
rows = []
for cl in list(d1.keys()) + list(d2.keys()):
    env = "tabular_color_table.tex" if cl in d1 else "tabular_color_table_x2.tex"
    for r, v in prof[cl]:
        rows.append([cl, r, v, env])
out = os.path.join(ROOT, "icl", "demaio2018_icl_colour_profiles.tsv")
write_tsv(out, hdr, rows)
print("   clusters recovered: %d from table env 1, %d from table env 2, total %d"
      % (len(d1), len(d2), len(prof)))
manifest(out, "https://arxiv.org/e-print/1710.11313",
    "GET https://arxiv.org/e-print/1710.11313 -> tar.gz -> tables/tabular_color_table.tex AND "
    "tables/tabular_color_table_x2.tex (BOTH \\input files, from two separate table* environments)",
    len(rows),
    [{"name": "Cluster", "unit": ""}, {"name": "log_r_kpc", "unit": "log10(kpc)"},
     {"name": "F110W_minus_F160W_mag", "unit": "mag"}, {"name": "source_table_env", "unit": ""}],
    "*** LATEX SPLIT-TABLE TRAP HANDLED: the colour profiles are split across TWO table* environments "
    "(table:tabular_color and table:tabular_color_x2). Parsing only the first would have silently dropped 7 of the "
    "23 systems -- INCLUDING the target cluster MACS1149. In addition, the last two cluster blocks inside "
    "tabular_color_table.tex are COMMENTED OUT (leading %) because they were moved into the continuation table; "
    "the parser skips commented lines so those 7 clusters are counted exactly once. Recovered 16 clusters from "
    "environment 1 and 7 from environment 2 = 23, asserted against the paper's stated sample of 23 groups and "
    "clusters. Reshaped from the published 4-clusters-wide wide format into long format.",
    "DeMaio et al. 2018, MNRAS 474, 3009 -- radial colour profiles of the BCG+ICL. "
    "TARGET CLUSTERS PRESENT: RXJ2248 (= Abell S1063), MACS0416, MACS1149. MACS0717 excluded by the authors. "
    "*** THIS IS A COLOUR PROFILE, NOT A SURFACE-BRIGHTNESS PROFILE. It gives F110W-F160W colour as a function of "
    "radius; it does NOT give mu_ICL(r). DeMaio et al. 2018 present their surface-brightness profiles only as a "
    "figure -- no mu(r) table is published in the paper or its arXiv source. *** "
    "BAND: HST WFC3-IR F110W minus F160W (observed frame). "
    "ICL/BCG SEPARATION METHOD: none -- this is the BCG+ICL system profiled continuously in radius. The paper "
    "treats the inner ~10 kpc as BCG-dominated and radii beyond that as increasingly ICL-dominated, but applies "
    "no decomposition. "
    "RADIAL RANGE: log10(r/kpc) from about 0.54 (~3.5 kpc) outward; each profile is terminated where the colour "
    "uncertainty exceeds 0.2 mag in 3 consecutive dlog(r)=0.05 bins. "
    "SB LIMIT: profiles reach ~26-27 mag/arcsec2 for the CLASH clusters, ~25-26 mag/arcsec2 for the groups. "
    "MASS-MODEL DEPENDENCE: none -- purely photometric.",
    "icl/raw/demaio2018_arXiv1710.11313_tabular_color_table.tex ; "
    "icl/raw/demaio2018_arXiv1710.11313_tabular_color_table_x2.tex")

print("\nPART 2 COMPLETE")
