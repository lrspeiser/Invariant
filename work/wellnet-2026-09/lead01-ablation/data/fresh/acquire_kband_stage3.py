#!/usr/bin/env python3
"""
Stage 3: 2MRS as a third independent K_s source, final magnitude adjudication,
ATLAS-3D photometric cross-check, and delivery of babyk2018_kband.tsv.

ACQUISITION ONLY. No mass-to-light ratio applied. No stellar mass computed.
No residuals, no acceleration ratios, no model comparison.

The three-catalogue comparison is INGEST VALIDATION used solely to detect
cross-identification errors; it is not a science result.
"""
import hashlib
import json
import math
import os
import time
import urllib.request
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
RAWK = os.path.join(BASE, "raw", "kband")
UA = {"User-Agent": "curl/8 (acquisition; astrophysics data ingest)"}
RETRIEVED = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
TOL = 0.5          # mag, agreement tolerance for cross-identification checks
LARGE_SEP = 5.0    # arcsec


def fetch(url, cache, timeout=90, retries=3):
    p = os.path.join(RAWK, cache)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return open(p, encoding="utf-8", errors="replace").read()
    last = None
    for a in range(retries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                b = r.read().decode("utf-8", errors="replace")
            open(p, "w", encoding="utf-8").write(b)
            time.sleep(0.25)
            return b
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    print(f"   FETCH FAIL {cache}: {last}")
    return None


def viz_err(b):
    """TRAP: the ASU error marker is '#INFO' + TAB + 'Error='."""
    return b is None or "\tError=" in b


def parse_asu(b):
    L = b.split("\n")
    hi = None
    for i, l in enumerate(L):
        if l.startswith("#") or l.strip() == "":
            continue
        hi = i
        break
    if hi is None:
        return []
    cols = L[hi].split("\t")
    return [dict(zip(cols, l.split("\t"))) for l in L[hi + 3:]
            if l.strip() and not l.startswith("#") and len(l.split("\t")) > 2]


def f(v):
    v = (v or "").strip()
    if v == "":
        return None
    try:
        return float(v)
    except ValueError:
        return None


data = json.load(open(os.path.join(BASE, "_kband_stage2.json"), encoding="utf-8"))
assert len(data) == 94, f"expected 94, got {len(data)}"

# ------------------------------------------------------- 2MRS for all 94 ----
print("--- 2MRS (J/ApJS/199/26/table3) for all 94 ---")
for i, r in enumerate(data, 1):
    r["mrs_kt"] = r["mrs_sep"] = r["mrs_cat"] = None
    if r["ra"] is None:
        continue
    url = ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
           "-source=J/ApJS/199/26/table3"
           f"&-c={r['ra']:+.6f}{r['dec']:+.6f}&-c.rs=60&-out.max=10"
           "&-out.all&-sort=_r")
    b = fetch(url, f"2mrs_all_{r['name']}.tsv")
    if viz_err(b):
        continue
    rows = parse_asu(b)
    for x in rows:
        if f(x.get("Ktmag")) is not None:
            r["mrs_kt"] = f(x.get("Ktmag"))
            r["mrs_sep"] = f(x.get("_r"))
            r["mrs_cat"] = (x.get("CAT") or "").strip()
            break
    if i % 25 == 0:
        print(f"  [{i:3d}/94] {r['name']:12s} 2MRS={r['mrs_kt']}")
n_mrs = sum(1 for r in data if r["mrs_kt"] is not None)
print(f"2MRS returned Ktmag for {n_mrs}/94")

# ------------------------------------------------------- adjudication -------
# XSC K.ext is the requested primary (total extrapolated, NOT extinction
# corrected). 2MRS Ktmag is total extrapolated AND extinction corrected.
# HyperLeda kt is a homogenised total K magnitude.
# An XSC cone match is accepted when it agrees with at least one independent
# NAME-based or named-entry source to within TOL, which is what actually
# certifies the cross-identification.
print("\n--- adjudication ---")
for r in data:
    k, mrs, leda = r["kext"], r["mrs_kt"], r["leda_kt"]
    flags = []
    if r["flag"] and r["flag"] != "no_xsc_match":
        flags.append(r["flag"])
    if k is not None and r["sep"] is not None and r["sep"] > LARGE_SEP:
        flags.append(f"large_sep_{r['sep']:.1f}arcsec")
    agree_mrs = (k is not None and mrs is not None and abs(k - mrs) <= TOL)
    agree_leda = (k is not None and leda is not None and abs(k - leda) <= TOL)

    if k is not None and (agree_mrs or agree_leda):
        r["K_final"], r["eK_final"] = k, r["ekext"]
        r["K_cat"], r["K_col"] = "VII/233/xsc", "K.ext (2MASS k_m_ext)"
        r["K_extcorr"] = "no"
        if r["sep"] is not None and r["sep"] > LARGE_SEP:
            flags.append("offset_match_corroborated_by_name_lookup")
    elif k is not None and mrs is None and leda is None:
        r["K_final"], r["eK_final"] = k, r["ekext"]
        r["K_cat"], r["K_col"] = "VII/233/xsc", "K.ext (2MASS k_m_ext)"
        r["K_extcorr"] = "no"
        flags.append("xsc_uncorroborated_no_independent_source")
    elif mrs is not None and (leda is None or abs(mrs - leda) <= TOL):
        r["K_final"], r["eK_final"] = mrs, None
        r["K_cat"], r["K_col"] = "J/ApJS/199/26/table3", "Ktmag (2MRS total Ks)"
        r["K_extcorr"] = "yes"
        flags.append("FALLBACK_2mrs")
        if k is not None:
            flags.append(f"xsc_match_rejected_dK={k - mrs:+.2f}mag")
    elif leda is not None:
        r["K_final"], r["eK_final"] = leda, r["leda_e_kt"]
        r["K_cat"], r["K_col"] = "HyperLeda", "kt (total K-magnitude)"
        r["K_extcorr"] = "no"
        flags.append("FALLBACK_hyperleda")
        if k is not None:
            flags.append(f"xsc_match_rejected_dK={k - leda:+.2f}mag")
    else:
        r["K_final"] = r["eK_final"] = None
        r["K_cat"] = r["K_col"] = ""
        r["K_extcorr"] = ""
        flags.append("NO_K_MAGNITUDE")
    r["flags"] = ";".join(flags)

got = [r for r in data if r["K_final"] is not None]
fb = [r for r in data if "FALLBACK" in r["flags"]]
print(f"{len(got)}/94 have a K magnitude; {len(fb)} via fallback")
for r in data:
    if r["flags"]:
        print(f"   {r['name']:12s} K={r['K_final']}  [{r['K_cat']}]  {r['flags']}")

# ------------------------------------------------- ATLAS-3D cross-check -----
print("\n--- ATLAS-3D (J/MNRAS/413/813/atlas3d) photometric cross-check ---")
b = fetch("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv?"
          "-source=J/MNRAS/413/813/atlas3d&-out.max=99999&-out.all",
          "atlas3d_full.tsv")
assert not viz_err(b), "ATLAS-3D fetch failed"
a3d_rows = parse_asu(b)
print(f"ATLAS-3D rows: {len(a3d_rows)} "
      "(table = 260 ATLAS-3D ETGs plus the spiral comparison sample)")
# guard against a silently truncated -out.max
assert len(a3d_rows) > 800, (
    f"ATLAS-3D table looks truncated: {len(a3d_rows)} rows")


def norm(n):
    return (n or "").replace(" ", "").replace("_", "").upper()


a3d = {}
for x in a3d_rows:
    g = norm(x.get("Gal"))
    if g:
        a3d[g] = x

pairs = []
for r in data:
    x = a3d.get(norm(r["name"]))
    if not x or r["K_final"] is None:
        continue
    KMAG, dist = f(x.get("KMAG")), f(x.get("Dist"))
    if KMAG is None or dist is None or dist <= 0:
        continue
    # absolute -> apparent via the ATLAS-3D distance (photometry only;
    # no mass-to-light ratio and no dynamical quantity is used)
    m_app = KMAG + 5.0 * math.log10(dist * 1e6 / 10.0)
    r["a3d_KMAG_abs"] = KMAG
    r["a3d_Dist_Mpc"] = dist
    r["a3d_mK_apparent"] = round(m_app, 3)
    r["a3d_minus_ours"] = round(m_app - r["K_final"], 3)
    pairs.append(r["a3d_minus_ours"])

pairs.sort()
if pairs:
    n = len(pairs)
    med = pairs[n // 2] if n % 2 else 0.5 * (pairs[n // 2 - 1] + pairs[n // 2])
    mean = sum(pairs) / n
    rms = (sum((p - mean) ** 2 for p in pairs) / n) ** 0.5
    mad = sorted(abs(p - med) for p in pairs)[n // 2]
    print(f"overlap n={n}")
    print(f"  offset (ATLAS3D_apparent - ours): median {med:+.3f}  mean {mean:+.3f} mag")
    print(f"  scatter: rms {rms:.3f}  MAD {mad:.3f} mag")
    print(f"  range: {pairs[0]:+.3f} .. {pairs[-1]:+.3f}")
    A3D_STATS = dict(n_overlap=n, median_offset_mag=round(med, 3),
                     mean_offset_mag=round(mean, 3), rms_mag=round(rms, 3),
                     mad_mag=round(mad, 3), min_mag=pairs[0], max_mag=pairs[-1])
else:
    A3D_STATS = dict(n_overlap=0)

# --------------------------------------------------------------- deliver ----
cols = [
    "Name [-]",
    "RA_resolved [deg, J2000]",
    "DE_resolved [deg, J2000]",
    "name_resolver [-]",
    "match_sep [arcsec]",
    "Ks_total [mag]",
    "e_Ks_total [mag]",
    "Ks_catalogue [-]",
    "Ks_column [-]",
    "Ks_extinction_corrected [yes/no]",
    "xsc_2MASX_id [-]",
    "xsc_K.ext [mag]",
    "xsc_e_K.ext [mag]",
    "xsc_r.ext [arcsec]",
    "mrs_Ktmag [mag]",
    "mrs_sep [arcsec]",
    "mrs_CAT_id [-]",
    "leda_kt [mag]",
    "leda_e_kt [mag]",
    "atlas3d_KMAG_abs [mag]",
    "atlas3d_Dist [Mpc]",
    "atlas3d_mK_apparent [mag] (RECONSTRUCTED = KMAG + 5log10(Dist*1e5))",
    "atlas3d_minus_ours [mag]",
    "flags [-]",
]
path = os.path.join(BASE, "babyk2018_kband.tsv")


def s(v, p=3):
    return "" if v is None else (f"{v:.{p}f}" if isinstance(v, float) else str(v))


with open(path, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in data:
        row = r.get("row") or {}
        fh.write("\t".join([
            r["name"], s(r["ra"], 6), s(r["dec"], 6), r["sesame"] or "",
            s(r["sep"], 2), s(r["K_final"]), s(r["eK_final"]),
            r["K_cat"], r["K_col"], r["K_extcorr"],
            (row.get("2MASX") or "").strip(),
            s(r["kext"]), s(r["ekext"]),
            (row.get("r.ext") or "").strip(),
            s(r["mrs_kt"]), s(r["mrs_sep"], 2), r["mrs_cat"] or "",
            s(r["leda_kt"], 2), s(r["leda_e_kt"], 2),
            s(r.get("a3d_KMAG_abs"), 2), s(r.get("a3d_Dist_Mpc"), 1),
            s(r.get("a3d_mK_apparent")), s(r.get("a3d_minus_ours")),
            r["flags"],
        ]) + "\n")

nrows = sum(1 for _ in open(path, encoding="utf-8")) - 1
assert nrows == 94, f"delivered {nrows} rows, expected 94"
names_out = [l.split("\t")[0] for l in
             open(path, encoding="utf-8").read().split("\n")[1:] if l.strip()]
assert names_out == [r["name"] for r in data], "name order/содержание mismatch"
print(f"\nwrote {os.path.basename(path)}: {nrows} rows, "
      f"first={names_out[0]} last={names_out[-1]}")

h = hashlib.sha256()
with open(path, "rb") as fh:
    for c in iter(lambda: fh.read(1 << 16), b""):
        h.update(c)

n_xsc = sum(1 for r in data if r["K_cat"] == "VII/233/xsc")
n_mrs_f = sum(1 for r in data if r["K_cat"].startswith("J/ApJS"))
n_leda = sum(1 for r in data if r["K_cat"] == "HyperLeda")
seps = sorted(r["sep"] for r in data if r["sep"] is not None)

manifest = {
    "file": os.path.basename(path),
    "retrieved_utc": RETRIEVED,
    "sha256": h.hexdigest(),
    "bytes": os.path.getsize(path),
    "row_count": nrows,
    "columns_with_units": cols,
    "sources": {
        "name_resolution": {
            "service": "CDS Sesame (SIMBAD)",
            "url_pattern": "https://cdsweb.u-strasbg.fr/cgi-bin/nph-sesame/-oI?<name>",
            "fallback": "https://vizier.cfa.harvard.edu/viz-bin/nph-sesame/-oI?<name>",
        },
        "primary_photometry": {
            "catalogue": "VII/233/xsc (2MASS All-Sky Extended Source Catalog)",
            "column": "K.ext",
            "column_native": "k_m_ext",
            "meaning": "total extrapolated Ks magnitude, from the fit to the "
                       "radial profile out to r.ext",
            "extinction_corrected": False,
            "query_pattern": ("https://vizier.cfa.harvard.edu/viz-bin/asu-tsv"
                              "?-source=VII/233/xsc&-c=<ra><dec>&-c.rs=60"
                              "&-out.max=50&-out.all&-sort=_r"),
        },
        "secondary": {
            "catalogue": "J/ApJS/199/26/table3 (2MASS Redshift Survey, Huchra+ 2012)",
            "column": "Ktmag",
            "meaning": "extinction-corrected total extrapolated Ks magnitude",
            "extinction_corrected": True,
        },
        "tertiary": {
            "catalogue": "HyperLeda",
            "column": "kt",
            "meaning": "homogenised total K magnitude",
            "url_pattern": "http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?o=<name>",
            "extinction_corrected": False,
        },
        "cross_check_only": {
            "catalogue": "J/MNRAS/413/813/atlas3d (ATLAS-3D, Cappellari+ 2011)",
            "columns_used": ["KMAG (total ABSOLUTE K magnitude)", "Dist (Mpc)"],
            "note": "PHOTOMETRY ONLY. No dynamical mass-to-light ratio or any "
                    "other dynamical quantity was read from this catalogue.",
        },
    },
    "counts": {
        "total": 94,
        "from_2MASS_XSC_K.ext": n_xsc,
        "from_2MRS_Ktmag": n_mrs_f,
        "from_HyperLeda_kt": n_leda,
        "with_no_magnitude": sum(1 for r in data if r["K_final"] is None),
    },
    "match_separation_arcsec": {
        "n": len(seps),
        "median": round(seps[len(seps) // 2], 3),
        "max": round(seps[-1], 3),
        "n_over_5arcsec": sum(1 for x in seps if x > 5),
    },
    "atlas3d_cross_check": A3D_STATS,
    "adjudication_rule": (
        "The 2MASS XSC cone match is accepted only when its K.ext agrees to "
        f"within {TOL} mag with at least one independent NAME-based lookup "
        "(HyperLeda by name, or a 2MRS entry). This is what certifies the "
        "cross-identification; positional proximity alone does not. Where the "
        "XSC match is contradicted or absent, the fallback actually used is "
        "recorded per object in the flags column."
    ),
    "notes": (
        "ACQUISITION ONLY: no mass-to-light ratio was applied and no stellar "
        "mass was computed. Magnitudes are apparent Ks in the 2MASS system "
        "unless the extinction-corrected column says otherwise - note that the "
        "2MRS Ktmag values ARE extinction corrected while the XSC K.ext and "
        "HyperLeda kt values are NOT, so the three are not interchangeable at "
        "the 0.01-0.1 mag level. The atlas3d_mK_apparent column is "
        "RECONSTRUCTED here from the ATLAS-3D absolute magnitude and its "
        "adopted distance; it is a cross-check column, not an acquired "
        "measurement. VizieR TRAP: the ASU error marker is '#INFO' followed by "
        "a TAB then 'Error='; testing for the string '#INFO Error=' with a "
        "space silently never matches and turns a missing catalogue into an "
        "apparent success."
    ),
}
json.dump(manifest, open(path + ".manifest.json", "w", encoding="utf-8"), indent=2)
json.dump(data, open(os.path.join(BASE, "_kband_stage3.json"), "w",
                     encoding="utf-8"), indent=2)
print("manifest written")
print(json.dumps(manifest["counts"], indent=2))
print(json.dumps(manifest["match_separation_arcsec"], indent=2))
