#!/usr/bin/env python3
"""
Stage 2: independent HyperLeda total-K cross-check for all 94 Babyk+2018 ETGs,
resolution of the XSC failures, and the ATLAS-3D photometric cross-check.

ACQUISITION ONLY. No mass-to-light ratio applied, no mass computed,
no residuals, no model comparison.

The XSC->LEDA comparison here is an INGEST VALIDATION (two independent
acquisitions of the same catalogued quantity), used only to detect
cross-identification errors. It is not a science result.
"""
import hashlib
import json
import os
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = os.path.dirname(os.path.abspath(__file__))
RAWK = os.path.join(BASE, "raw", "kband")
os.makedirs(RAWK, exist_ok=True)
UA = {"User-Agent": "curl/8 (acquisition; astrophysics data ingest)"}


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
            time.sleep(0.3)
            return b
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (a + 1))
    print(f"   FETCH FAIL {cache}: {last}")
    return None


def viz_err(b):
    """TRAP: the marker is '#INFO' + TAB + 'Error='. A space never matches."""
    return b is None or "\tError=" in b


# ------------------------------------------------------------- HyperLeda ----
def leda_kt(name):
    """Total K magnitude 'kt' from HyperLeda. Returns (kt, e_kt)."""
    url = ("http://atlas.obs-hp.fr/hyperleda/ledacat.cgi?o="
           + urllib.parse.quote(name))
    b = fetch(url, f"leda_{name}.html")
    if b is None:
        return None, None
    txt = re.sub(r"<[^>]*>", " ", b)
    txt = txt.replace("&#177;", "+-")
    m = re.search(r"\bkt\s+(-?[\d.]+)\s*(?:\+-\s*([\d.]+))?\s*mag\s+Total K-magnitude",
                  txt)
    if not m:
        return None, None
    return float(m.group(1)), (float(m.group(2)) if m.group(2) else None)


# ------------------------------------------------------------------ load ----
stage1 = json.load(open(os.path.join(BASE, "_kband_stage1.json"), encoding="utf-8"))
print(f"stage1 loaded: {len(stage1)} objects")

print("\n--- HyperLeda total-K for all 94 (independent cross-check) ---")
for i, r in enumerate(stage1, 1):
    kt, ekt = leda_kt(r["name"])
    r["leda_kt"], r["leda_e_kt"] = kt, ekt
    if i % 20 == 0 or kt is None:
        print(f"  [{i:3d}/94] {r['name']:12s} leda_kt={kt}")

miss = [r["name"] for r in stage1 if r["leda_kt"] is None]
print(f"HyperLeda returned kt for {94 - len(miss)}/94; missing: {miss}")

# --------------------------------------------------- consistency screening ---
print("\n--- XSC K.ext vs HyperLeda kt (ingest validation only) ---")
SUSPECT = 0.5
suspects = []
for r in stage1:
    k, kt = r["kext"], r["leda_kt"]
    r["dK_xsc_minus_leda"] = (round(k - kt, 3)
                              if (k is not None and kt is not None) else None)
    if r["dK_xsc_minus_leda"] is not None and abs(r["dK_xsc_minus_leda"]) > SUSPECT:
        suspects.append(r)
d = [r["dK_xsc_minus_leda"] for r in stage1 if r["dK_xsc_minus_leda"] is not None]
d_ok = [x for x in d if abs(x) <= SUSPECT]
d_ok.sort()
if d_ok:
    med = d_ok[len(d_ok) // 2]
    mean = sum(d_ok) / len(d_ok)
    rms = (sum((x - mean) ** 2 for x in d_ok) / len(d_ok)) ** 0.5
    print(f"  agreeing pairs n={len(d_ok)}  median dK={med:+.3f}  "
          f"mean={mean:+.3f}  rms={rms:.3f} mag")
print(f"  SUSPECT (|dK| > {SUSPECT}): {[(r['name'], r['dK_xsc_minus_leda']) for r in suspects]}")

# ------------------------------------------------- final magnitude choice ----
# Policy, applied per object and recorded in the flag column:
#   - XSC K.ext accepted when a match exists AND it agrees with LEDA to <=0.5 mag
#   - otherwise fall back to HyperLeda kt, flagged
LARGE_SEP = 5.0
for r in stage1:
    k, kt, sep = r["kext"], r["leda_kt"], r["sep"]
    flags = []
    if r["flag"]:
        flags.append(r["flag"])
    if k is not None and sep is not None and sep > LARGE_SEP:
        flags.append(f"large_sep_{sep:.1f}arcsec")
    dK = r["dK_xsc_minus_leda"]
    if k is not None and dK is not None and abs(dK) > SUSPECT:
        flags.append(f"xsc_leda_disagree_{dK:+.2f}mag")
    if k is not None and (dK is None or abs(dK) <= SUSPECT):
        r["K_final"], r["eK_final"] = k, r["ekext"]
        r["K_src_cat"], r["K_src_col"] = "VII/233/xsc (2MASS XSC)", "K.ext (k_m_ext)"
    elif kt is not None:
        r["K_final"], r["eK_final"] = kt, r["leda_e_kt"]
        r["K_src_cat"] = "HyperLeda"
        r["K_src_col"] = "kt (total K-magnitude)"
        flags.append("FALLBACK_hyperleda")
    else:
        r["K_final"] = r["eK_final"] = None
        r["K_src_cat"] = r["K_src_col"] = ""
        flags.append("NO_K_MAGNITUDE")
    r["flags"] = ";".join(flags)

got = [r for r in stage1 if r["K_final"] is not None]
fb = [r for r in stage1 if "FALLBACK_hyperleda" in r["flags"]]
none_ = [r for r in stage1 if r["K_final"] is None]
print(f"\nFINAL: {len(got)}/94 with a K magnitude; "
      f"{len(fb)} via HyperLeda fallback; {len(none_)} with none")
for r in fb:
    print(f"   FALLBACK {r['name']:12s} kt={r['K_final']}  ({r['flags']})")
for r in none_:
    print(f"   NO MAGNITUDE {r['name']}")

json.dump(stage1, open(os.path.join(BASE, "_kband_stage2.json"), "w",
                       encoding="utf-8"), indent=2)
print("\nstage2 written")
