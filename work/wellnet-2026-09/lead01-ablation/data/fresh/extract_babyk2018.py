#!/usr/bin/env python3
"""
Extract machine-readable tables from the arXiv LaTeX source of
Babyk & McNamara et al. 2018, ApJ 857, 32 (arXiv:1803.00020).

ACQUISITION ONLY. No science, no derived quantities, no model comparison.

Trap guards (all four known silent-extraction traps):
 (a) These are plain `tabular` inside `table*`, NOT `deluxetable`. We never use a
     startdata/enddata parser; we bound on \begin{tabular}...\end{tabular}.
 (b) We never split a data block on \hline and drop fragments. \hline is only
     ever recognised as a standalone rule token on its own line.
 (c) Comments are stripped BEFORE any startswith('%') test, and we assert that
     no data row was lost to a comment glued onto it.
 (d) Every table is split across TWO `table*` environments ("Continued."). We
     concatenate both parts and assert the combined count.
"""
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(BASE, "raw", "babyk2018_src", "ScalRelBabyk_revised.tex")
OUT = BASE

SOURCE_URL = "https://arxiv.org/e-print/1803.00020"
BIBCODE = "2018ApJ...857...32B"
N_EXPECTED = 94  # stated in abstract (line 60) and conclusions (line 864)


def strip_comment(line):
    """Remove a LaTeX comment. Trap (c): done BEFORE any startswith('%') test,
    and honouring escaped \\% so we never truncate a legitimate value."""
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def tabular_blocks(text):
    """Yield (caption, label, body) for every table*/table environment."""
    for m in re.finditer(r"\\begin\{(table\*?|table)\}(.*?)\\end\{\1\}", text, re.S):
        env = m.group(2)
        cap = re.search(r"\\caption\{(.*?)\}\s*(?:\\label\{(.*?)\})?", env, re.S)
        caption = cap.group(1).strip() if cap else ""
        label = cap.group(2) if (cap and cap.group(2)) else ""
        tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", env, re.S)
        if not tab:
            continue
        yield caption, label, tab.group(1)


def data_rows(body):
    """Return the data rows of one tabular body.

    Header rows are everything up to and including the column-number legend
    row `(1) & (2) & ...`. Trap (b): \hline is only dropped when it is the
    entire (stripped) line; it never causes a fragment to be skipped.
    """
    raw_lines = body.split("\n")
    cleaned = []
    lost_to_comment = 0
    for ln in raw_lines:
        c = strip_comment(ln)
        if c.strip() == "" and ln.strip() != "" and not ln.strip().startswith("%"):
            lost_to_comment += 1
        cleaned.append(c)

    # Locate the column-number legend row, e.g. "(1)    &    (2)  & ..."
    start = 0
    for i, ln in enumerate(cleaned):
        s = ln.strip()
        if re.match(r"^\(1\)\s*&", s):
            start = i + 1
            break
    else:
        raise AssertionError("column-number legend row (1)&(2)&... not found")

    rows = []
    for ln in cleaned[start:]:
        s = ln.strip()
        if s == "" or s == r"\hline":  # trap (b): standalone rule only
            continue
        s = s.rstrip()
        if s.endswith("\\\\"):
            s = s[:-2]
        if "&" not in s:
            continue
        rows.append([c.strip() for c in s.split("&")])
    return rows, lost_to_comment


PM = r"$\pm$"


def split_pm(cell):
    """Split 'v$\\pm$e' (or 'v$\\pm$e1$\\pm$e2') into (value, err1, err2)."""
    parts = cell.split(PM)
    parts = [clean_scalar(p) for p in parts]
    while len(parts) < 3:
        parts.append("")
    return parts[0], parts[1], parts[2]


def clean_scalar(s):
    s = s.replace("$", "").replace("\\surd", "Y").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").strip()
    return s


def num(s):
    """Parse to float, tolerating a trailing '*' distance flag. '' -> None."""
    s = s.strip().rstrip("*").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(path, cols, nrows, query, notes):
    mf = {
        "file": os.path.basename(path),
        "source_url": SOURCE_URL,
        "bibcode": BIBCODE,
        "arxiv_id": "1803.00020",
        "retrieved_utc": RETRIEVED,
        "sha256": sha256(path),
        "bytes": os.path.getsize(path),
        "row_count": nrows,
        "columns_with_units": cols,
        "query_issued": query,
        "notes": notes,
    }
    with open(path + ".manifest.json", "w", encoding="utf-8") as f:
        json.dump(mf, f, indent=2)
    return mf


RETRIEVED = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

with open(TEX, "r", encoding="utf-8", errors="replace") as f:
    text = f.read()

blocks = list(tabular_blocks(text))
print(f"found {len(blocks)} tabular environments")

# --- Trap (d): group the two-part tables by caption ------------------------
# Table 1 = "List of early-type galaxy properties." + following "Continued."
# Table 2 = "The best-fit parameters for spectra extracted from within 5$r_e$..."
# Table 3 = "The best-fit parameters for an isothermal $\beta$-model."
groups = {"tab1": [], "tab2": [], "tab3": []}
current = None
for caption, label, body in blocks:
    if label in groups:
        current = label
        groups[current].append(body)
    elif caption.startswith("Continued.") and current is not None:
        groups[current].append(body)
    else:
        current = None  # unrelated table (scaling-relation fits etc.)

for k, v in groups.items():
    assert len(v) == 2, f"{k}: expected 2 table* parts (trap d), got {len(v)}"
    print(f"{k}: {len(v)} parts concatenated")

parsed = {}
for k, parts in groups.items():
    allrows = []
    for p in parts:
        rows, lost = data_rows(p)
        assert lost == 0, f"{k}: {lost} rows lost to comment stripping (trap c)"
        allrows.extend(rows)
    assert len(allrows) == N_EXPECTED, (
        f"{k}: row count {len(allrows)} != stated sample size {N_EXPECTED}")
    print(f"{k}: {len(allrows)} rows  first={allrows[0][0]!r}  last={allrows[-1][0]!r}")
    parsed[k] = allrows

# consistency: same object list, same order, in all three tables
n1 = [clean_scalar(r[0]) for r in parsed["tab1"]]
n2 = [clean_scalar(r[0]) for r in parsed["tab2"]]
n3 = [clean_scalar(r[0]) for r in parsed["tab3"]]
assert n1 == n2 == n3, "object name lists differ between tables"
print(f"name lists identical across tab1/tab2/tab3 ({len(n1)} objects)")

# --------------------------------------------------------------------------
# Table 1 -> TSV
t1_cols = [
    "Name [-]", "RA_gal_l [deg]", "DEC_gal_b [deg]", "ObsID [-]",
    "Exposure_before [ks]", "Exposure_after [ks]", "MorphType [-]",
    "BCG_flag [Y/blank]", "cD_flag [Y/blank]", "z [-]", "D_A [Mpc]",
    "D_A_isVirgoAssumed [0/1]", "D_L [Mpc]", "D_L_isVirgoAssumed [0/1]",
    "NH_gal [1e20 cm-2]", "NH_intrinsic_or_total [1e20 cm-2]",
]
t1_path = os.path.join(OUT, "babyk2018_table1_sample.tsv")
with open(t1_path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(t1_cols) + "\n")
    for r in parsed["tab1"]:
        name = clean_scalar(r[0])
        ra, dec = clean_scalar(r[1]), clean_scalar(r[2])
        obsid = clean_scalar(r[3])
        exp = clean_scalar(r[4])
        eb, ea = (exp.split("/") + [""])[:2] if "/" in exp else (exp, "")
        mt = clean_scalar(r[5])
        bcg = "Y" if "surd" in r[6] or clean_scalar(r[6]) == "Y" else ""
        cd = "Y" if "surd" in r[7] or clean_scalar(r[7]) == "Y" else ""
        z = clean_scalar(r[8])
        da_raw, dl_raw = clean_scalar(r[9]), clean_scalar(r[10])
        da_v = "1" if da_raw.endswith("*") else "0"
        dl_v = "1" if dl_raw.endswith("*") else "0"
        nh = clean_scalar(r[11])
        nh1, nh2 = (nh.split("/") + [""])[:2] if "/" in nh else (nh, "")
        f.write("\t".join([
            name, ra, dec, obsid, eb.strip(), ea.strip(), mt, bcg, cd, z,
            da_raw.rstrip("*"), da_v, dl_raw.rstrip("*"), dl_v,
            nh1.strip(), nh2.strip()]) + "\n")

# Table 2 -> TSV
t2_cols = [
    "Name [-]", "r_5re [kpc]", "r_5re_err [kpc]", "kT [keV]", "kT_err [keV]",
    "log10_fX_0.5-2.0keV [log10(erg cm-2 s-1)]",
    "log10_fX_err [dex]", "LX_0.5-2.0keV [1e40 erg s-1]",
    "LX_err [1e40 erg s-1]", "Z [Zsun]", "Z_err [Zsun]",
    "Cstat_per_dof [-]", "sigma_c [km s-1]", "sigma_c_err [km s-1]",
]
t2_path = os.path.join(OUT, "babyk2018_table2_spectra_5re.tsv")
with open(t2_path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(t2_cols) + "\n")
    for r in parsed["tab2"]:
        name = clean_scalar(r[0])
        re5, re5e, _ = split_pm(r[1])
        kt, kte, _ = split_pm(r[2])
        fx, fxe, _ = split_pm(r[3])
        lx, lxe, _ = split_pm(r[4])
        z, ze, _ = split_pm(r[5])
        chi = clean_scalar(r[6])
        sg, sge, _ = split_pm(r[7])
        f.write("\t".join([name, re5, re5e, kt, kte, fx, fxe, lx, lxe,
                           z, ze, chi, sg, sge]) + "\n")

# Table 3 -> TSV
t3_cols = [
    "Name [-]", "beta [-]", "beta_err [-]", "r_c [kpc]", "r_c_err [kpc]",
    "rho_0 [1e-24 g cm-3]", "rho_0_err [1e-24 g cm-3]", "chi2 [-]",
    "M_gas_lt_5re [1e11 Msun]", "M_gas_err [1e11 Msun]",
    "M_tot_hydrostatic_lt_5re [1e13 Msun]",
    "M_tot_err_stat [1e13 Msun]", "M_tot_err_sys [1e13 Msun]",
]
t3_path = os.path.join(OUT, "babyk2018_table3_betamodel_masses.tsv")
with open(t3_path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(t3_cols) + "\n")
    for r in parsed["tab3"]:
        name = clean_scalar(r[0])
        b, be, _ = split_pm(r[1])
        rc, rce, _ = split_pm(r[2])
        rho, rhoe, _ = split_pm(r[3])
        chi = clean_scalar(r[4])
        mg, mge, _ = split_pm(r[5])
        mt, mte1, mte2 = split_pm(r[6])
        f.write("\t".join([name, b, be, rc, rce, rho, rhoe, chi,
                           mg, mge, mt, mte1, mte2]) + "\n")

# --------------------------------------------------------------------------
# Joined per-object table (straight key merge on Name; no derived quantities)
j_cols = [
    "Name [-]", "MorphType [-]", "BCG_flag [Y/blank]", "cD_flag [Y/blank]",
    "z [-]", "D_A [Mpc]", "D_L [Mpc]",
    "r_5re [kpc]", "r_5re_err [kpc]",
    "kT [keV]", "kT_err [keV]",
    "LX_0.5-2.0keV [1e40 erg s-1]", "LX_err [1e40 erg s-1]",
    "Z [Zsun]", "Z_err [Zsun]",
    "sigma_c [km s-1]", "sigma_c_err [km s-1]",
    "M_gas_lt_5re [1e11 Msun]", "M_gas_err [1e11 Msun]",
    "M_tot_hydrostatic_lt_5re [1e13 Msun]",
    "M_tot_err_stat [1e13 Msun]", "M_tot_err_sys [1e13 Msun]",
]


def load_tsv(p):
    with open(p, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        return hdr, [dict(zip(hdr, l.rstrip("\n").split("\t"))) for l in f if l.strip()]


_, T1 = load_tsv(t1_path)
_, T2 = load_tsv(t2_path)
_, T3 = load_tsv(t3_path)
i1 = {r["Name [-]"]: r for r in T1}
i2 = {r["Name [-]"]: r for r in T2}
i3 = {r["Name [-]"]: r for r in T3}

j_path = os.path.join(OUT, "babyk2018_joined_per_object.tsv")
nmiss = 0
with open(j_path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(j_cols) + "\n")
    for nm in n1:
        a, b, c = i1[nm], i2[nm], i3[nm]
        row = [nm, a["MorphType [-]"], a["BCG_flag [Y/blank]"],
               a["cD_flag [Y/blank]"], a["z [-]"], a["D_A [Mpc]"],
               a["D_L [Mpc]"],
               b["r_5re [kpc]"], b["r_5re_err [kpc]"],
               b["kT [keV]"], b["kT_err [keV]"],
               b["LX_0.5-2.0keV [1e40 erg s-1]"], b["LX_err [1e40 erg s-1]"],
               b["Z [Zsun]"], b["Z_err [Zsun]"],
               b["sigma_c [km s-1]"], b["sigma_c_err [km s-1]"],
               c["M_gas_lt_5re [1e11 Msun]"], c["M_gas_err [1e11 Msun]"],
               c["M_tot_hydrostatic_lt_5re [1e13 Msun]"],
               c["M_tot_err_stat [1e13 Msun]"], c["M_tot_err_sys [1e13 Msun]"]]
        for v in (row[7], row[9], row[17], row[19]):
            if num(v) is None:
                nmiss += 1
        f.write("\t".join(row) + "\n")

print(f"joined: {len(n1)} rows, {nmiss} missing core numeric cells")

notes_common = (
    "Extracted from arXiv LaTeX source (plain `tabular` inside `table*`, NOT "
    "deluxetable). Table is split across two `table*` environments "
    "(main + 'Continued.'); both parts concatenated. Row count asserted "
    "against the 94 objects stated in the abstract and conclusions. "
    "Total mass is HYDROSTATIC: derived from an isothermal beta-model fit to "
    "the Chandra X-ray surface brightness plus the single-temperature spectral "
    "fit, via the hydrostatic-equilibrium equation (paper Sect. 'mass', "
    "eq. at line 409). It is NOT an NFW fit and NOT a lensing/light-tied "
    "parametric model."
)

m1 = write_manifest(t1_path, t1_cols, len(parsed["tab1"]),
                    f"curl -sL {SOURCE_URL} -> tar xzf -> ScalRelBabyk_revised.tex, Table 1 (label tab1) + 'Continued.'",
                    notes_common + " RA/DEC as printed in the paper's Table 1 are galactic coordinates (l,b) despite the '(J2000)' header; verify before any positional use.")
m2 = write_manifest(t2_path, t2_cols, len(parsed["tab2"]),
                    f"curl -sL {SOURCE_URL} -> tar xzf -> ScalRelBabyk_revised.tex, Table 2 (label tab2) + 'Continued.'",
                    notes_common + " sigma_c is the LEDA central stellar velocity dispersion. f_X is tabulated as log10 of the unabsorbed 0.5-2.0 keV flux. Z entries without an error were frozen at 0.5 Zsun in the fit.")
m3 = write_manifest(t3_path, t3_cols, len(parsed["tab3"]),
                    f"curl -sL {SOURCE_URL} -> tar xzf -> ScalRelBabyk_revised.tex, Table 3 (label tab3) + 'Continued.'",
                    notes_common + " M_tot carries TWO errors: statistical then systematic. Where the surface-brightness profile does not reach 5r_e the mass profiles were extrapolated to 5r_e by the authors using the slope of the last 20 points in log-log space.")
mj = write_manifest(j_path, j_cols, len(n1),
                    "inner join of babyk2018_table1/2/3 on Name (key merge only, no derived quantities)",
                    notes_common + " Radius definition for every mass column is r = 5 r_e (five optical half-light radii), tabulated per object in kpc as r_5re.")

print("wrote:")
for p in (t1_path, t2_path, t3_path, j_path):
    print(" ", os.path.basename(p), os.path.getsize(p), "bytes")
