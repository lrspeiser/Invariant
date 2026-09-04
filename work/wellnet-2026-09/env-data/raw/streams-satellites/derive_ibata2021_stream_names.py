"""DERIVED product (clearly labelled as such, NOT a measurement).

The Ibata et al. 2021 (ApJ 914, 123) VizieR table J/ApJ/914/123/table1 gives
per-star Gaia EDR3 astrometry, photometry and heliocentric radial velocities,
plus a stream identification label in the range 1-32. The paper states that this
label is decoded only by "the different colors of the streams in the bottom
panel of Figure 4" -- there is NO published table, and no ReadMe note, mapping
label number to stream name.

This script recovers a probable mapping by nearest-track assignment: for each
label group, every member star is matched to the closest point of every
galstreams v1.2.1 `ibata2021` celestial track, and the track with the smallest
median angular separation wins. The result is a NAMING CONVENIENCE only. It adds
no physical information and must never be treated as a measurement.
"""
import glob
import io
import os
import warnings

import numpy as np
import pandas as pd
import astropy.units as u
import astropy.coordinates as ac
from astropy.table import QTable

warnings.filterwarnings("ignore")
BASE = os.path.dirname(os.path.abspath(__file__))
TRACKS = os.path.join(BASE, "galstreams_data", "galstreams", "tracks")
OUT = os.path.join(BASE, "stream_ibata2021_label_to_name_DERIVED.tsv")

txt = open(os.path.join(BASE, "stream_ibata2021_streamfinder_members.vizier.tsv"),
           encoding="utf-8").read()
body = [l for l in txt.splitlines() if l.strip() and not l.startswith("#")]
df = pd.read_csv(io.StringIO("\n".join([body[0]] + body[3:])), sep="\t", dtype=str)
df["ra"] = pd.to_numeric(df["RAJ2000"], errors="coerce")
df["de"] = pd.to_numeric(df["DEJ2000"], errors="coerce")
df["lab"] = pd.to_numeric(df["Stream"], errors="coerce")
df["hrv"] = pd.to_numeric(df["HRV"], errors="coerce")
print("member stars:", len(df), " labels:", df["lab"].nunique())

tf = sorted(glob.glob(os.path.join(TRACKS, "track.*.ibata2021.ecsv")))
tf = [f for f in tf if not f.endswith(".summary.ecsv")]
print("galstreams ibata2021 tracks:", len(tf))
tracks = []
for f in tf:
    t = QTable.read(f)
    name = os.path.basename(f).split(".")[2]
    tracks.append((name, ac.SkyCoord(ra=np.asarray(t["ra"]) * u.deg,
                                     dec=np.asarray(t["dec"]) * u.deg)))

rows = []
for lab, grp in df.groupby("lab"):
    c = ac.SkyCoord(ra=grp["ra"].values * u.deg, dec=grp["de"].values * u.deg)
    best, best_med, second = None, 1e9, 1e9
    for name, tc in tracks:
        idx, d2d, _ = c.match_to_catalog_sky(tc)
        med = float(np.median(d2d.deg))
        if med < best_med:
            second = best_med
            best, best_med = name, med
        elif med < second:
            second = med
    n_rv = int(grp["hrv"].notna().sum())
    n_rv_ok = int((grp["hrv"].abs() < 600).sum())
    rows.append(dict(Stream_label=int(lab), n_stars=len(grp), n_HRV=n_rv,
                     n_HRV_credible=n_rv_ok,
                     probable_name=best,
                     median_sep_deg=round(best_med, 4),
                     next_best_sep_deg=round(second, 4),
                     ratio=round(second / best_med, 2) if best_med > 0 else "",
                     ra_med=round(float(np.median(grp["ra"])), 3),
                     de_med=round(float(np.median(grp["de"])), 3)))

rows.sort(key=lambda r: r["Stream_label"])
cols = ["Stream_label", "n_stars", "n_HRV", "n_HRV_credible", "probable_name",
        "median_sep_deg", "next_best_sep_deg", "ratio", "ra_med", "de_med"]
with open(OUT, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(r[c]) for c in cols) + "\n")

print("\n%-6s %6s %5s %5s  %-22s %9s %9s %6s" %
      ("label", "nstars", "nRV", "nRVok", "probable_name", "med_sep", "next_sep", "ratio"))
for r in rows:
    print("%-6d %6d %5d %5d  %-22s %9.3f %9.3f %6s" %
          (r["Stream_label"], r["n_stars"], r["n_HRV"], r["n_HRV_credible"],
           r["probable_name"], r["median_sep_deg"], r["next_best_sep_deg"], r["ratio"]))

conf = [r for r in rows if r["median_sep_deg"] < 1.0]
print("\nassignments with median separation < 1 deg:", len(conf), "of", len(rows))
print("total stars:", sum(r["n_stars"] for r in rows),
      " total HRV:", sum(r["n_HRV"] for r in rows),
      " credible HRV:", sum(r["n_HRV_credible"] for r in rows))
print("wrote", OUT)
