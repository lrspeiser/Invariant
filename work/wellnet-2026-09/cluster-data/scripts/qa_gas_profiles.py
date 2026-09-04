"""Sanity scan of the acquired electron-density and temperature profiles.

Acquisition QA only.  Checks the things that silently break a downstream
baryonic model: unordered radial bins, non-positive densities, non-monotonic
n_e(r), and how many radial decades each profile actually spans.  Nothing is
fitted; nothing is modified.
"""
import glob
import json
import os
import numpy as np

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_tsv(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        lines = [l.rstrip("\n") for l in f if l.strip()]
    hdr = lines[0].split("\t")
    rows = [l.split("\t") for l in lines[1:]]
    return hdr, rows


def col(hdr, rows, name):
    i = hdr.index(name)
    out = []
    for r in rows:
        try:
            out.append(float(r[i]))
        except (ValueError, IndexError):
            out.append(np.nan)
    return np.array(out)


report = []
for path in sorted(glob.glob(os.path.join(LANE, "gas", "accept_*.tsv"))):
    base = os.path.basename(path)
    if "table1" in base or "ReadMe" in base:
        continue
    hdr, rows = read_tsv(path)
    if "nelec" not in hdr:
        continue
    rin, rout = col(hdr, rows, "Rin"), col(hdr, rows, "Rout")
    ne = col(hdr, rows, "nelec")
    tx = col(hdr, rows, "Tx")
    rmid = 0.5 * (rin + rout)
    order = np.argsort(rmid)
    rs, nes, txs = rmid[order], ne[order], tx[order]
    good = np.isfinite(nes) & (nes > 0)
    d = np.diff(nes[good])
    report.append({
        "file": "gas/" + base,
        "n_bins": int(len(rows)),
        "radii_sorted_as_stored": bool(np.all(np.diff(rmid) > 0)),
        "r_min_Mpc": round(float(np.nanmin(rin)), 5),
        "r_max_Mpc": round(float(np.nanmax(rout)), 4),
        "radial_decades": round(float(np.log10(np.nanmax(rout) /
                                               max(np.nanmin(rmid), 1e-6))), 2),
        "n_e_all_positive": bool(np.all(nes[np.isfinite(nes)] > 0)),
        "n_e_strictly_decreasing_outward": bool(np.all(d < 0)),
        "n_e_n_upward_steps": int(np.sum(d > 0)),
        "n_e_range_cm-3": [float("%.3e" % np.nanmin(nes)), float("%.3e" % np.nanmax(nes))],
        "T_all_positive": bool(np.all(txs[np.isfinite(txs)] > 0)),
        "T_range_keV": [round(float(np.nanmin(txs)), 3), round(float(np.nanmax(txs)), 3)],
        "n_nan_rows": int(np.sum(~np.isfinite(nes))),
    })

dest = os.path.join(LANE, "gas", "gas_profile_qa.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump({"note": ("Acquisition QA on the ACCEPT deprojected profiles. "
                        "'n_e_n_upward_steps' counts bins where the density rises "
                        "outward; a handful is normal measurement noise in a "
                        "deprojection, a large number would indicate a substructured "
                        "or merging system where a single spherical profile is a poor "
                        "description."),
               "profiles": report}, f, indent=2)

print("%-34s %5s %6s %9s %8s %6s %s" % (
    "file", "bins", "sorted", "r_max/Mpc", "decades", "up", "T range (keV)"))
for r in report:
    print("%-34s %5d %6s %9.3f %8.2f %6d %s" % (
        r["file"], r["n_bins"], r["radii_sorted_as_stored"], r["r_max_Mpc"],
        r["radial_decades"], r["n_e_n_upward_steps"], r["T_range_keV"]))
