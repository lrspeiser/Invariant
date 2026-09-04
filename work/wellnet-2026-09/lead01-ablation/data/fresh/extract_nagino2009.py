#!/usr/bin/env python3
"""
Extract tables from the arXiv LaTeX source of
Nagino & Matsushita 2009, A&A 501, 157 (arXiv:0903.2540).

ACQUISITION ONLY. No science, no residuals, no acceleration ratios,
no model comparison.

Same four trap guards as the Babyk extractor. These are plain `tabular`
inside `table*` (NOT deluxetable), single-part (asserted), one data row per
physical line, terminated by \\\\.
"""
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
TEX = os.path.join(BASE, "raw", "nagino2009_src", "aa.tex")
OUT = BASE
SOURCE_URL = "https://arxiv.org/e-print/0903.2540"
BIBCODE = "2009A&A...501..157N"
N_EXPECTED = 22  # stated in abstract (line 40) and conclusions (line 1101)
RETRIEVED = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def strip_comment(line):
    out, i = [], 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2]); i += 2; continue
        if c == "%":
            break
        out.append(c); i += 1
    return "".join(out)


def clean(s):
    s = s.replace("$", "").replace("~", " ")
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    return s.replace("{", "").replace("}", "").strip()


def num(s):
    s = clean(s)
    if s in ("", "---", "--", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def grab_table(caption_re):
    """Return the data rows of the single `table*` whose caption matches."""
    with open(TEX, encoding="utf-8", errors="replace") as f:
        text = f.read()
    hits = []
    for m in re.finditer(r"\\begin\{table\*?\}(.*?)\\end\{table\*?\}", text, re.S):
        env = m.group(1)
        cap = re.search(r"\\caption\{(.*?)\}", env, re.S)
        if cap and re.search(caption_re, cap.group(1)):
            hits.append(env)
    # trap (d): assert the table is NOT split across environments
    assert len(hits) == 1, f"{caption_re!r}: expected 1 table env, got {len(hits)}"
    tab = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", hits[0], re.S)
    assert tab, "no tabular found"
    rows, lost = [], 0
    started = False
    for ln in tab.group(1).split("\n"):
        c = strip_comment(ln)
        if c.strip() == "" and ln.strip() != "" and not ln.strip().startswith("%"):
            lost += 1
        s = c.strip()
        # trap (b): \hline / \hline\hline only ever recognised standalone
        s_norm = s.replace(r"\hline", "").strip()
        if s_norm == "":
            continue
        s = s_norm
        if s.endswith("\\\\"):
            s = s[:-2]
        if "&" not in s:
            continue
        cells = [x.strip() for x in s.split("&")]
        # header rows carry no leading galaxy identifier of the form NAME+digits
        if not started:
            if re.match(r"^(IC|NGC|UGC|ESO)\s*\d", cells[0]):
                started = True
            else:
                continue
        rows.append(cells)
    assert lost == 0, f"{caption_re!r}: {lost} rows lost to comment stripping (trap c)"
    return rows


t1 = grab_table(r"Galaxy sample in the XMM-Newton archive data")
t2 = grab_table(r"Integrated \$M/L_B\$ and \$M/L_K\$")
for nm, rows in (("sample", t1), ("mass-to-light", t2)):
    assert len(rows) == N_EXPECTED, f"{nm}: {len(rows)} != {N_EXPECTED}"
    print(f"{nm}: {len(rows)} rows  first={clean(rows[0][0])!r}  last={clean(rows[-1][0])!r}")

n1 = [clean(r[0]) for r in t1]
n2 = [clean(r[0]) for r in t2]
assert n1 == n2, "galaxy lists differ between the two tables"
print(f"name lists identical across both tables ({len(n1)} objects)")

ARCMIN_TO_RAD = math.pi / (180.0 * 60.0)

cols = [
    "Name [-]",
    "T_type_RC3 [-]",
    "D [Mpc]",
    "r_e [arcmin]",
    "r_e [kpc] (RECONSTRUCTED = D*1000*r_e_arcmin*pi/10800)",
    "log10_L_B [log10(Lsun_B)]",
    "sigma [km s-1]",
    "NH [1e20 cm-2]",
    "Xray_morph_class [XC=compact/XE=extended]",
    "Environment [-]",
    "r_0.5re [kpc] (RECONSTRUCTED)",
    "r_3re [kpc] (RECONSTRUCTED)",
    "r_6re [kpc] (RECONSTRUCTED)",
    "M_over_LB_lt_0.5re [Msun/Lsun_B]",
    "M_over_LB_lt_3re [Msun/Lsun_B]",
    "M_over_LB_lt_6re [Msun/Lsun_B]",
    "M_over_LK_lt_0.5re [Msun/Lsun_K]",
    "M_over_LK_lt_3re [Msun/Lsun_K]",
    "M_over_LK_lt_6re [Msun/Lsun_K]",
    "M_tot_hydrostatic_lt_0.5re [Msun] (RECONSTRUCTED = M_over_LB * 10^log10_L_B)",
    "M_tot_hydrostatic_lt_3re [Msun] (RECONSTRUCTED)",
    "M_tot_hydrostatic_lt_6re [Msun] (RECONSTRUCTED)",
]

path = os.path.join(OUT, "nagino2009_etg_masstolight.tsv")
nmiss = 0
with open(path, "w", encoding="utf-8", newline="") as f:
    f.write("\t".join(cols) + "\n")
    for a, b in zip(t1, t2):
        name = clean(a[0])
        ttype, D = clean(a[1]), num(a[2])
        re_am = num(a[3])
        logLB, sig, nh = num(a[4]), num(a[5]), num(a[6])
        note = clean(a[7]) if len(a) > 7 else ""
        xcls = "XC" if "X_C" in note or "X_C" in a[7] else ("XE" if "X_E" in a[7] else "")
        env = ",".join(p for p in note.replace("X_C", "").replace("X_E", "")
                       .split(",") if p.strip()).strip(", ")
        re_kpc = D * 1000.0 * re_am * ARCMIN_TO_RAD if (D and re_am) else None
        mlb = [num(b[1]), num(b[2]), num(b[3])]
        mlk = [num(b[4]), num(b[5]), num(b[6])]
        LB = 10.0 ** logLB if logLB is not None else None
        mtot = [(m * LB if (m is not None and LB) else None) for m in mlb]
        rr = [(re_kpc * k if re_kpc else None) for k in (0.5, 3.0, 6.0)]

        def fmt(v, p=4):
            return "" if v is None else (f"{v:.{p}g}")
        for v in mlb[:2]:
            if v is None:
                nmiss += 1
        f.write("\t".join([
            name, ttype, fmt(D), fmt(re_am), fmt(re_kpc),
            fmt(logLB), fmt(sig), fmt(nh), xcls, env,
            fmt(rr[0]), fmt(rr[1]), fmt(rr[2]),
            fmt(mlb[0]), fmt(mlb[1]), fmt(mlb[2]),
            fmt(mlk[0]), fmt(mlk[1]), fmt(mlk[2]),
            fmt(mtot[0]), fmt(mtot[1]), fmt(mtot[2]),
        ]) + "\n")

print(f"wrote {path} ({os.path.getsize(path)} bytes), "
      f"{nmiss} missing values in the 0.5re/3re M/L_B columns")

h = hashlib.sha256()
with open(path, "rb") as f:
    for ch in iter(lambda: f.read(1 << 16), b""):
        h.update(ch)

manifest = {
    "file": os.path.basename(path),
    "source_url": SOURCE_URL,
    "bibcode": BIBCODE,
    "arxiv_id": "0903.2540",
    "retrieved_utc": RETRIEVED,
    "sha256": h.hexdigest(),
    "bytes": os.path.getsize(path),
    "row_count": N_EXPECTED,
    "columns_with_units": cols,
    "query_issued": (f"curl -sL {SOURCE_URL} -> tar xzf -> aa.tex; "
                     "Table 'Galaxy sample in the XMM-Newton archive data' "
                     "joined on galaxy name with Table 'Integrated M/L_B and "
                     "M/L_K at 0.5, 3 and 6 r_e'"),
    "notes": (
        "Total mass is HYDROSTATIC and NON-PARAMETRIC: computed by the authors "
        "directly from the deprojected ISM temperature and gas-density "
        "gradients via M(R) = -kT(R)R/(G*mu*m_p) * (dln n/dln R + dln T/dln R) "
        "(paper Sect. 'Mass profiles', eq. 2). It is NOT an NFW fit; NFW is "
        "cited in the introduction only as literature context. "
        "IMPORTANT: this paper tabulates NO GAS MASS. It provides M_tot/L at "
        "three radii, L_B, sigma, r_e and distance, but no M_gas(<r). "
        "L_K is used by the authors (computed from 2MASS) but is NOT tabulated, "
        "so M/L_K cannot be inverted to a mass from this table alone; only the "
        "M/L_B route is closed. "
        "Columns marked RECONSTRUCTED are unit conversions/products computed "
        "here from the tabulated columns, not values printed in the paper; the "
        "raw tabulated M/L and log L_B columns are retained alongside them. "
        "'---' in the 6 r_e columns means the profile did not reach 6 r_e "
        "(NGC1549, NGC4365, NGC4526, NGC4552); for NGC1549, NGC4477 and NGC5322 "
        "the mass profile inside 0.5 r_e rests on XMM-Newton data only. "
        "Distances D are the paper's adopted values (Mpc), not redshifts."
    ),
}
with open(path + ".manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)
print("manifest written")
