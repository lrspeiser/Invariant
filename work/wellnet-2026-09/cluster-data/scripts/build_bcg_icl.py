# -*- coding: utf-8 -*-
"""Build BCG (Product 2) and ICL (Product 3) data products + manifests."""
import os, re, json, hashlib, datetime, shutil

ROOT = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
SCR  = r"C:\Users\henry\AppData\Local\Temp\claude\C--Users-henry-dev\a2309145-5e60-4815-97f2-bb0c877edc0d\scratchpad"
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def sha256(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()

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
    if extra: m.update(extra)
    mp = path + ".manifest.json"
    with open(mp,"w",encoding="utf-8") as f:
        json.dump(m,f,indent=2,ensure_ascii=False)
    print("  manifest ->", os.path.basename(mp))
    return m

def write_tsv(path, header, rows):
    with open(path,"w",encoding="utf-8",newline="\n") as f:
        f.write("\t".join(header)+"\n")
        for r in rows:
            f.write("\t".join("" if v is None else str(v) for v in r)+"\n")
    print("WROTE", path, "rows=",len(rows))

# ---------- LaTeX helpers ----------
BS = chr(92)   # a single backslash

def clean_cell(s):
    s = s.strip()
    s = s.replace(BS + BS, "").strip()          # drop LaTeX row terminator
    s = s.replace("$", "")
    s = s.replace(BS + "pm", "+/-")
    s = s.replace(BS + "til", "~")
    s = s.replace(BS + "hline", "")
    s = s.replace(BS + "bigstar", "")
    s = re.sub(BS + r"[a-zA-Z]+", "", s)        # strip any remaining macros
    s = s.replace("{", "").replace("}", "")
    s = s.replace("^", "^").replace("_", "_")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def latex_rows(text):
    """Yield lists of cells for lines that look like data rows."""
    for ln in text.split("\n"):
        L = ln.strip()
        if not L or L.startswith("%") or L.startswith(r"\hline"): continue
        if "&" not in L: continue
        cells = [clean_cell(c) for c in L.split("&")]
        yield cells

# ---------- VizieR TSV parse ----------
def parse_vizier_tsv(path):
    """Return (header, units, rows). VizieR asu-tsv: comment lines '#', then
    header, units, dashes, data."""
    txt = open(path, encoding="utf-8", errors="replace").read()
    lines = [l.rstrip("\r") for l in txt.split("\n")]
    if any("Error=Table or Catalog not found" in l or "does not exist in catalog" in l for l in lines):
        raise RuntimeError("VizieR NOT FOUND response: "+path)
    body = [l for l in lines if not l.startswith("#")]
    # locate header: first non-empty line followed later by a dashes line
    hi = None
    for i,l in enumerate(body):
        if l.strip()=="" : continue
        if hi is None: hi = i; continue
    # simpler: find dashes line
    di = None
    for i,l in enumerate(body):
        if l.startswith("---") and "\t" in l: di = i; break
    if di is None: raise RuntimeError("no dashes separator in "+path)
    header = body[di-2].split("\t")
    units  = body[di-1].split("\t")
    rows = []
    for l in body[di+1:]:
        if l.strip()=="" : continue
        if l.startswith("#"): continue
        c = l.split("\t")
        if len(c) < len(header)-2: continue
        rows.append(c)
    return header, units, rows

print("="*70); print("PRODUCT 2 / 3 build   ", NOW); print("="*70)

# =====================================================================
# 1. KLUGE 2020 (VizieR J/ApJS/247/43/bcg) -- BCG+ICL Sersic, g' band
# =====================================================================
print("\n[1] Kluge+2020 BCG/ICL Sersic (J/ApJS/247/43/bcg)")
raw = os.path.join(ROOT,"bcg","raw","kluge2020_bcg_icl_sersic_raw.tsv")
_src=os.path.join(ROOT,"bcg","raw","kluge2020_bcg_raw.tsv")
if os.path.exists(_src) and not os.path.exists(raw): shutil.copyfile(_src, raw)
hdr, units, rows = parse_vizier_tsv(raw)
assert len(rows)==170, "Kluge2020 expected 170 rows (ReadMe: table4.dat 170 records), got %d"%len(rows)
out = os.path.join(ROOT,"bcg","kluge2020_bcg_icl_sersic.tsv")
write_tsv(out, hdr, rows)
K20_COLDESC = {
 "Cluster":"Cluster name","BCG":"Brightest cluster galaxy name","RAJ2000":"RA (J2000, sexagesimal)",
 "DEJ2000":"Dec (J2000, sexagesimal)","z":"Redshift","Scale":"Angular scale kpc/arcsec",
 "n1":"Sersic index, inner component","re1":"Effective radius, inner component (arcsec)",
 "SBe1":"Effective g' surface brightness, inner component (mag/arcsec2)",
 "n2":"Sersic index, OUTER component (=ICL in double-Sersic decomposition)",
 "re2":"Effective radius, outer/ICL component (arcsec)",
 "SBe2":"Effective g' surface brightness, outer/ICL component (mag/arcsec2)",
 "f2":"Luminosity fraction in outer (ICL) component = f_ICL^Sx",
 "re":"Effective radius from direct integration (arcsec)",
 "SBe":"Effective SB from direct integration (g' mag/arcsec2)",
 "Mtot":"Integrated absolute g' brightness of BCG+ICL (mag)",
 "Type":"Accretion signature type","Sel":"Lauer+2014 selection","HST":"HST used for inner profile",
}
cols = [{"name":h,"unit":(units[i].strip() if i<len(units) else "")} for i,h in enumerate(hdr)]
for c in cols:
    if c["name"] in K20_COLDESC: c["description"]=K20_COLDESC[c["name"]]
manifest(out,
  "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
  "-source=J/ApJS/247/43/bcg&-out=**&-out.max=unlimited",
  len(rows), cols,
  "VizieR asu-tsv download of catalogue J/ApJS/247/43 table 'bcg' (= paper Tables 1 and 4 merged by CDS). "
  "Row count asserted against ReadMe (table1.dat 170 records, table4.dat 170 records).",
  "Kluge et al. 2020, ApJS 247, 43 (2020ApJS..247...43K). 170 local (z<~0.08) northern galaxy clusters, "
  "Wendelstein WWFI, g' band ONLY. Semimajor-axis SB profiles of BCG+ICL measured to a limiting "
  "SB = 30 g' mag/arcsec2 (background inhomogeneity floor dSB>31 g' mag/arcsec2). "
  "BCG/ICL SEPARATION METHOD: none applied for the single-Sersic systems (71% of sample) -- the fit is to the "
  "BCG+ICL system as a whole. For the 29% (49/170) fitted with a DOUBLE SERSIC, the outer Sersic component "
  "(n2, re2, SBe2) is identified with the ICL and f2 is the ICL luminosity fraction (Kluge+2021 'method d'). "
  "Parameters are corrected for PSF broadening, Galactic dust extinction, cosmic dimming and are K-corrected. "
  "Radial range: profiles extend from the HST/deconvolved-WWFI-resolved core out to the SB=30 g' mag/arcsec2 isophote. "
  "NOT tied to any mass model -- purely photometric.",
  "bcg/raw/kluge2020_bcg_icl_sersic_raw.tsv",
  {"catalogue_id_echoed":"J/ApJS/247/43/bcg",
   "vizier_readme":"https://cdsarc.cds.unistra.fr/ftp/J/ApJS/247/43/ReadMe"})

# --- A2029 single-cluster extract
i_cl = hdr.index("Cluster")
a2029 = [r for r in rows if r[i_cl].strip()=="A2029"]
assert len(a2029)==1, "expected exactly 1 A2029 row, got %d"%len(a2029)
out2 = os.path.join(ROOT,"bcg","kluge2020_bcg_A2029_IC1101.tsv")
write_tsv(out2, hdr, a2029)
manifest(out2,
  "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
  "-source=J/ApJS/247/43/bcg&-out=**&-out.max=unlimited  [then filtered Cluster=='A2029']",
  1, cols,
  "Single-row extract (Cluster=='A2029') from kluge2020_bcg_icl_sersic.tsv.",
  "ABELL 2029 BCG = IC 1101. g' band. SINGLE-Sersic BCG+ICL system: n1=5.55+/-0.26, "
  "re1=2.61e2 arcsec, SBe1=26.08+/-0.14 g' mag/arcsec2, re(direct integration)=3.29e2 arcsec, "
  "SBe=26.53, Mtot=-25.85+/-0.12 g' mag. Angular scale 1.484 kpc/arcsec (z=0.0779). "
  "NO double-Sersic decomposition exists for A2029 (n2/re2/SBe2/f2 are blank), therefore NO "
  "per-cluster ICL luminosity fraction is available for A2029 from Kluge+2020/2021. "
  "SB limit 30 g' mag/arcsec2. Not mass-model dependent.",
  "bcg/raw/kluge2020_bcg_icl_sersic_raw.tsv")

# =====================================================================
# 2. KLUGE 2021 (VizieR J/ApJS/252/27/table2) -- host cluster params
# =====================================================================
print("\n[2] Kluge+2021 host cluster parameters (J/ApJS/252/27/table2)")
raw = os.path.join(ROOT,"icl","raw","kluge2021_table2_raw.tsv")
hdr2, units2, rows2 = parse_vizier_tsv(raw)
assert len(rows2)==170, "Kluge2021 expected 170 rows, got %d"%len(rows2)
out3 = os.path.join(ROOT,"icl","kluge2021_host_cluster_params.tsv")
write_tsv(out3, hdr2, rows2)
cols2 = [{"name":h,"unit":(units2[i].strip() if i<len(units2) else "")} for i,h in enumerate(hdr2)]
manifest(out3,
  "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
  "-source=J/ApJS/252/27/table2&-out=**&-out.max=unlimited",
  len(rows2), cols2,
  "VizieR asu-tsv download of J/ApJS/252/27 table2. Row count asserted against ReadMe (table2.dat, 170 records).",
  "Kluge et al. 2021, ApJS 252, 27 -- 'Photometric dissection of intracluster light'. INCLUDES A2029. "
  "*** CAVEAT: this deposited table contains HOST CLUSTER parameters only (velocity dispersion, richness, "
  "gravitational radius/mass, integrated satellite brightness, phase-space densities) -- it does NOT contain "
  "per-cluster ICL fractions. Kluge+2021 report ICL fractions only as SAMPLE AVERAGES over the 170 clusters "
  "(f_ICL^MT=71+/-22%, f_ICL^SB27=34+/-19%, f_ICL^DV=48+/-20%, f_ICL^Sx=52+/-21%; all g' band, SB limit 30 "
  "g' mag/arcsec2). No per-cluster ICL fraction for A2029 is published. *** "
  "The four separation methods are: (a) integrated-magnitude cut M<-21.85 g'; (b) SB cut SB>27 g' mag/arcsec2; "
  "(c) de Vaucouleurs fit to SB<23 g' mag/arcsec2 plus excess light above it; (d) double-Sersic decomposition. "
  "MASS-MODEL FLAG: the 'gravitational mass' Mg and radius Radg columns are dynamical (virial-type) estimates "
  "derived from the satellite galaxy distribution -- they are NOT used to define the ICL, but do not treat them "
  "as independent of dynamical assumptions.",
  "icl/raw/kluge2021_table2_raw.tsv",
  {"catalogue_id_echoed":"J/ApJS/252/27/table2"})

# =====================================================================
# 3. MONTES & TRUJILLO 2018 (arXiv:1710.03240 = MNRAS 474, 917)
# =====================================================================
print("\n[3] Montes & Trujillo 2018 (arXiv:1710.03240)")
MT_TEX = os.path.join(ROOT,"icl","raw","montes_trujillo2018_arXiv1710.03240_ff_mnras.tex")
tex = open(MT_TEX, encoding="utf-8", errors="replace").read()

def grab_env(text, label):
    """Return the body of the table* environment carrying \label{label}."""
    envs = re.findall(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", text, re.S)
    for e in envs:
        if label in e: return e
    raise RuntimeError("env %s not found"%label)

# ---- Table 1: positions + SB limits (BCG product)
b = grab_env(tex, r"\label{table:table1}")
BANDS = ["F435W","F606W","F814W","F105W","F125W","F160W"]
hdr_t1 = ["Cluster","z","RA_J2000","Dec_J2000",
          "SBlim_F435W","SBlim_F606W","SBlim_F814W","SBlim_F105W","SBlim_F125W","SBlim_F140W","SBlim_F160W"]
rows_t1=[]
for cells in latex_rows(b):
    if len(cells)!=11: continue
    if not re.match(r"^(Abell|MACS)", cells[0]): continue
    rows_t1.append(cells)
assert len(rows_t1)==6, "MT2018 Table1 expected 6 HFF clusters, got %d"%len(rows_t1)
out = os.path.join(ROOT,"bcg","montes_trujillo2018_hff_centres_sblimits.tsv")
write_tsv(out, hdr_t1, rows_t1)
manifest(out, "https://arxiv.org/e-print/1710.03240",
  "GET https://arxiv.org/e-print/1710.03240 -> tar.gz -> ff_mnras.tex -> table* env with \label{table:table1}",
  len(rows_t1),
  [{"name":"Cluster","unit":""},{"name":"z","unit":""},
   {"name":"RA_J2000","unit":"hh:mm:ss"},{"name":"Dec_J2000","unit":"dd:mm:ss"}]+
  [{"name":"SBlim_%s"%b_,"unit":"mag/arcsec2"} for b_ in ["F435W","F606W","F814W","F105W","F125W","F140W","F160W"]],
  "arXiv source tarball; LaTeX table* environment parsed by \label. Row count asserted ==6 against the paper "
  "text ('the six HFF clusters'). arXiv id VERIFIED by title match: 'Intracluster Light at the Frontier II: "
  "The Frontier Fields Clusters' (NOTE: the id 1710.07300 suggested upstream is a DIFFERENT paper -- a machine-"
  "learning paper 'FigureQA' -- and was NOT used).",
  "Montes & Trujillo 2018, MNRAS 474, 917. Table 1. Six HFF clusters: A2744, MACSJ0416.1-2403, MACSJ0717.5+3745, "
  "MACSJ1149.5+2223, Abell S1063, Abell 370. Coordinates are the adopted CLUSTER CENTRE (BCG-centred for the "
  "single-BCG clusters; A2744/MACS0416/MACS0717/MACS1149 have multiple BCGs and the paper works with 'BCG(s)'). "
  "SB limits are 3-sigma above sky in 3x3 arcsec2 boxes, all measured at the F160W spatial resolution. "
  "This file provides POSITION + DEPTH only -- it contains NO BCG magnitude and NO light profile.",
  "icl/raw/montes_trujillo2018_arXiv1710.03240_ff_mnras.tex")

# ---- Table 2: ICL fractions (ICL product)
b2 = grab_env(tex, r"\label{table:table2}")
hdr_t2 = ["Cluster","fICL_26_lt_muV_lt_27_pct","fICL_muV_gt_26_pct",
          "fICL_50kpc_lt_R_lt_Rlimit_pct","fICL_R_lt_Rlimit_pct","fICL_R_lt_R500_pct"]
rows_t2=[]
for cells in latex_rows(b2):
    if len(cells)!=6: continue
    if not re.match(r"^(Abell|MACS)", cells[0]): continue
    rows_t2.append(cells)
assert len(rows_t2)==6, "MT2018 Table2 expected 6 clusters, got %d"%len(rows_t2)
out = os.path.join(ROOT,"icl","montes_trujillo2018_icl_fractions.tsv")
write_tsv(out, hdr_t2, rows_t2)
manifest(out, "https://arxiv.org/e-print/1710.03240",
  "GET https://arxiv.org/e-print/1710.03240 -> tar.gz -> ff_mnras.tex -> table* env with \label{table:table2}",
  len(rows_t2),
  [{"name":"Cluster","unit":""},
   {"name":"fICL_26_lt_muV_lt_27_pct","unit":"percent"},
   {"name":"fICL_muV_gt_26_pct","unit":"percent"},
   {"name":"fICL_50kpc_lt_R_lt_Rlimit_pct","unit":"percent"},
   {"name":"fICL_R_lt_Rlimit_pct","unit":"percent"},
   {"name":"fICL_R_lt_R500_pct","unit":"percent"}],
  "arXiv source tarball; LaTeX table* environment parsed by \label. Row count asserted ==6 against paper text.",
  "*** PRODUCT 3 (ICL fraction) for all six HFF clusters: A2744, MACSJ0416.1-2403, MACSJ0717.5+3745, "
  "MACSJ1149.5+2223, Abell S1063, Abell 370. *** "
  "BAND: rest-frame V (synthesised from the 7 HST ACS+WFC3 bands listed in montes_trujillo2018_hff_centres_sblimits.tsv). "
  "SB LIMIT: 30.2-31.9 mag/arcsec2 depending on band (see the companion centres/SB-limits file); the ICL analysis "
  "itself is cut at mu_V = 26 mag/arcsec2. "
  "ICL/BCG SEPARATION METHOD -- TWO DIFFERENT ONES ARE TABULATED AND THEY MUST NOT BE MIXED: "
  "(i) columns 'fICL_26_lt_muV_lt_27_pct' and 'fICL_muV_gt_26_pct' use a pure SURFACE-BRIGHTNESS CUT in V "
  "(a SB slice 26<mu_V<27, and everything fainter than mu_V=26 respectively); "
  "(ii) columns 'fICL_50kpc_lt_R_lt_Rlimit_pct', 'fICL_R_lt_Rlimit_pct' and 'fICL_R_lt_R500_pct' instead assume the ICL "
  "follows a LINEAR (log-space) profile fitted to the V-band SB profile at R>50 kpc and extrapolate that fit inward "
  "under the BCG, i.e. a FIXED 50 kpc RADIUS CUT plus an extrapolation. "
  "RADIAL RANGE: the ICL is measured out to R_limit ~ 120 kpc (the paper's outermost reliable bin). "
  "MASS-MODEL DEPENDENCE FLAG: the final column (R<R500) requires R500, which is a mass-model/X-ray derived radius -- "
  "that column is NOT purely photometric. The other four columns are photometric only. "
  "Cluster members are removed by masking before the ICL is measured.",
  "icl/raw/montes_trujillo2018_arXiv1710.03240_ff_mnras.tex")
