# -*- coding: utf-8 -*-
"""Product 3 support -- Montes & Trujillo 2018 radial stellar-population profiles,
which define R_limit (the outer radius of the ICL measurement) for each HFF cluster."""
import os, re, json, hashlib, datetime

ROOT = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\cluster-data"
NOW  = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
BS   = chr(92)

def sha256(p): return hashlib.sha256(open(p, 'rb').read()).hexdigest()

def manifest(path, source_url, exact_query, row_count, columns, extraction, note,
             raw_response_file, extra=None):
    m = {"file": os.path.basename(path), "source_url": source_url, "exact_query": exact_query,
         "retrieved_utc": NOW, "sha256": sha256(path), "bytes": os.path.getsize(path),
         "row_count": row_count, "column_count": len(columns), "columns": columns,
         "extraction": extraction, "note": note, "raw_response_file": raw_response_file}
    if extra: m.update(extra)
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
    s = s.strip().replace(BS + BS, "").strip()
    s = s.replace("$", "").replace(BS + "pm", "+/-")
    s = s.replace(BS + "cdots", "")
    s = re.sub(BS + r"[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", s).strip()

print("=" * 70); print("Montes & Trujillo 2018 radial profiles / R_limit  ", NOW); print("=" * 70)

p = os.path.join(ROOT, "icl", "raw", "montes_trujillo2018_arXiv1710.03240_ff_mnras.tex")
tex = open(p, encoding="utf-8", errors="replace").read()
envs = re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", tex, re.S)
body = [e for e in envs if "table:agemet" in e]
assert len(body) == 1, "expected 1 agemet table env, got %d" % len(body)
body = body[0]

CLUSTERS = ["Abell 2744", "MACSJ0416.1-2403", "MACSJ0717.5+3745",
            "MACSJ1149.5+2223", "Abell S1063", "Abell 370"]
# verify the column ordering straight from the table's own multicolumn header
hdr_line = [l for l in body.split("\n") if "multicolumn" in l]
assert len(hdr_line) == 1, "expected 1 multicolumn header line"
names = [n.strip() for n in re.findall(r"multicolumn\{2\}\{c\}\{([^}]*)\}", hdr_line[0])]
assert names == CLUSTERS, "cluster column order changed: %s" % names

rows = []
rlimit = {c: None for c in CLUSTERS}
for ln in body.split("\n"):
    L = ln.strip()
    if not L or L.startswith("%") or "multicolumn" in L or "&" not in L:
        continue
    if "Bin (kpc)" in L or "tabular" in L or "caption" in L:
        continue
    cells = [clean_cell(c) for c in L.split("&")]
    if len(cells) != 13:
        continue
    m = re.match(r"^([\d.]+)\s*-\s*([\d.]+)$", cells[0])
    if not m:
        continue
    r_in, r_out = float(m.group(1)), float(m.group(2))
    for i, cl in enumerate(CLUSTERS):
        age, feh = cells[1 + 2 * i], cells[2 + 2 * i]
        if not age or not feh:
            continue
        rows.append([cl, cells[0], r_in, r_out, age, feh])
        rlimit[cl] = r_out

assert len(rows) > 0, "no data rows parsed"
nbins = {}
for r in rows:
    nbins[r[0]] = nbins.get(r[0], 0) + 1
print("  radial bins per cluster:", nbins)
print("  R_limit (kpc) per cluster:", rlimit)
assert set(nbins) == set(CLUSTERS), "missing clusters: %s" % (set(CLUSTERS) - set(nbins))
assert rlimit["Abell 2744"] == 200.0, "A2744 R_limit should be 200 kpc, got %s" % rlimit["Abell 2744"]

hdr = ["Cluster", "bin_kpc", "R_inner_kpc", "R_outer_kpc", "Age_Gyr", "FeH"]
out = os.path.join(ROOT, "icl", "montes_trujillo2018_radial_age_metallicity.tsv")
write_tsv(out, hdr, rows)
manifest(out, "https://arxiv.org/e-print/1710.03240",
    "GET https://arxiv.org/e-print/1710.03240 -> tar.gz -> ff_mnras.tex -> table env with "
    "\\label{table:agemet} (Appendix table)",
    len(rows),
    [{"name": "Cluster", "unit": ""}, {"name": "bin_kpc", "unit": "kpc"},
     {"name": "R_inner_kpc", "unit": "kpc"}, {"name": "R_outer_kpc", "unit": "kpc"},
     {"name": "Age_Gyr", "unit": "Gyr"}, {"name": "FeH", "unit": "dex"}],
    "arXiv source tarball; LaTeX table environment identified by \\label{table:agemet}. The published table is "
    "6-clusters-wide (13 columns: 1 radial bin + 6x[Age,Fe/H]); reshaped to long format. The cluster column "
    "ordering was verified against the table's own \\multicolumn header rather than assumed. Cells printed as "
    "\\cdots (no measurement) are dropped, which is what makes the last populated bin per cluster equal R_limit. "
    "Assertion: Abell 2744 R_limit == 200 kpc.",
    "*** RADIAL-RANGE DEFINITION for the Montes & Trujillo 2018 ICL measurement, plus the ICL stellar-population "
    "gradients. This is the companion file to icl/montes_trujillo2018_icl_fractions.tsv -- the 'R_limit' referred "
    "to in that file's column names is the outermost radius listed here for each cluster. "
    "R_limit per cluster (kpc): Abell 2744 = 200; MACSJ0416.1-2403 = 93.9; MACSJ0717.5+3745 = 137; "
    "MACSJ1149.5+2223 = 137; Abell S1063 = 137; Abell 370 = 137. "
    "Innermost bin is 0-0.5 kpc, so the radial coverage is 0 -> R_limit, centred on the BCG(s). "
    "The paper defines the ICL region as R > 50 kpc, so the ICL proper is sampled only over 50 kpc -> R_limit, "
    "i.e. roughly 50-94 kpc for MACS0416, 50-137 kpc for four clusters, and 50-200 kpc for A2744. "
    "R_limit is defined by the authors as the farthest spatial bin with accurate ages and metallicities (bins "
    "where more than 4 filters are reliable). "
    "Ages and metallicities are medians over 500 jackknife realizations of the photometry, from SED fitting the "
    "7-band HST photometry against Vazdekis et al. 2016 models. "
    "MASS-MODEL DEPENDENCE: none -- photometric/SED based.",
    "icl/raw/montes_trujillo2018_arXiv1710.03240_ff_mnras.tex")

print("\nPART 5 COMPLETE")
