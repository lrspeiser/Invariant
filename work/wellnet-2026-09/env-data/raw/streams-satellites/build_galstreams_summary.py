"""Build a cleaned, flat summary TSV of the galstreams v1.2.1 track library.

Reads ONLY data files (ECSV tables) extracted from the galstreams release
tarball. No code from the downloaded package is imported or executed.

InfoFlags semantics, verbatim from galstreams/core.py (v1.2.1):
    # bit 0: 0 = great circle by construction
    # bit 1: 0 = no distance track available (only mean or central value reported)
    # bit 2: 0 = no proper motion data available (only mean or central value reported)
    # bit 3: 0 = no radial velocity data available (only mean or central value reported)

=> bit0=1 : on-sky track is an EMPIRICAL (measured) track
   bit0=0 : on-sky track is a GREAT CIRCLE BY CONSTRUCTION (geometric model)
   bit1=1 : MEASURED distance track along the stream
   bit2=1 : MEASURED proper-motion track along the stream
   bit3=1 : MEASURED radial-velocity track along the stream

Out-of-plane geometry columns computed here (pure coordinate transforms of the
published measurements, no potential and no mass model assumed anywhere):
   pole_b  : Galactic latitude of the stream's mid-point orbital-plane pole.
             |pole_b| ~ 90 deg  -> stream orbit lies IN the Galactic disc plane
             |pole_b| ~ 0  deg  -> stream orbit is POLAR (perpendicular to disc)
             => inc_to_disc = 90 - |pole_b| is the orbit inclination w.r.t. the disc.
   z_min/z_max/absz_max : height above the Galactic plane along the track (kpc),
             only meaningful when has_D == 1.
   R_gc_min/R_gc_max : Galactocentric spherical radius along the track (kpc),
             only meaningful when has_D == 1.
Galactocentric frame: astropy default (Sun at R0 = 8.122 kpc, z_sun = 20.8 pc).
"""
import collections
import glob
import os
import sys
import warnings

import numpy as np
import astropy.units as u
import astropy.coordinates as ac
from astropy.table import QTable

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
TRACKS = os.path.join(BASE, "galstreams_data", "galstreams", "tracks")
OUT = os.path.join(BASE, "galstreams_track_summary.tsv")

summaries = sorted(glob.glob(os.path.join(TRACKS, "*.summary.ecsv")))
print("summary files found:", len(summaries))

rows = []
orphans = []
for sf in summaries:
    tf = sf.replace(".summary.ecsv", ".ecsv")
    base = os.path.basename(sf).replace(".summary.ecsv", "")
    if not os.path.exists(tf):
        # UPSTREAM PACKAGING DEFECT in galstreams v1.2.1: summary present, track absent.
        orphans.append(base)
        continue
    parts = base.split(".")
    # naming: track.<st|ep|po>.<Name>.<ref>
    kind = parts[1] if len(parts) > 2 else ""
    ref = parts[-1]
    st = QTable.read(sf)
    d = {k: st[k][0] for k in st.colnames}
    flags = str(d["InfoFlags"])
    rec = dict(
        TrackFileBase=base,
        StreamName=str(d.get("StreamName", "")),
        StreamShortName=str(d.get("StreamShortName", "")),
        TrackRef=ref,
        TrackType=kind,
        InfoFlags=flags,
        # flag digit meanings (Mateu 2023, sec. 'Individual stream track
        # implementations'): char0 = 0 if the stream is ASSUMED to be a great
        # circle, 1 if not; chars 1,2,3 = distance / proper motion / radial
        # velocity track available (1) or not (0); '2' = available BUT with a
        # documented caveat.
        has_empirical_track=int(flags[0] != "0"),
        has_D=int(flags[1] != "0"),
        has_pm=int(flags[2] != "0"),
        has_vrad=int(flags[3] != "0"),
        D_caveat=int(flags[1] == "2"),
        pm_caveat=int(flags[2] == "2"),
        vrad_caveat=int(flags[3] == "2"),
    )

    # --- pole (orbital plane normal at mid-point) ---
    pole = ac.SkyCoord(ra=d["pole.ra"], dec=d["pole.dec"], frame="icrs")
    pg = pole.galactic
    rec["pole_ra_deg"] = round(float(pole.ra.deg), 5)
    rec["pole_dec_deg"] = round(float(pole.dec.deg), 5)
    rec["pole_l_deg"] = round(float(pg.l.deg), 5)
    rec["pole_b_deg"] = round(float(pg.b.deg), 5)
    # inclination of the stream orbital plane w.r.t. the Galactic disc plane
    rec["orbit_inc_to_disc_deg"] = round(90.0 - abs(float(pg.b.deg)), 4)

    # --- mid point ---
    mid = ac.SkyCoord(ra=d["mid.ra"], dec=d["mid.dec"], frame="icrs")
    rec["mid_l_deg"] = round(float(mid.galactic.l.deg), 5)
    rec["mid_b_deg"] = round(float(mid.galactic.b.deg), 5)
    rec["mid_distance_kpc"] = round(float(u.Quantity(d["mid.distance"]).to_value(u.kpc)), 5)

    for k in ("width_phi2", "width_pm_phi1_cosphi2", "width_pm_phi2"):
        if k in d:
            try:
                rec[k] = round(float(u.Quantity(d[k]).value), 5)
            except Exception:
                rec[k] = ""

    # --- track file: N points, sky span, |b| range, z range, R_gc range ---
    tt = QTable.read(tf)
    n = len(tt)
    rec["n_track_points"] = n
    c = ac.SkyCoord(
        ra=np.asarray(tt["ra"]) * u.deg,
        dec=np.asarray(tt["dec"]) * u.deg,
        distance=np.asarray(tt["distance"]) * u.kpc,
        frame="icrs",
    )
    g = c.galactic
    b = g.b.deg
    rec["b_min_deg"] = round(float(np.min(b)), 4)
    rec["b_max_deg"] = round(float(np.max(b)), 4)
    rec["absb_min_deg"] = round(float(np.min(np.abs(b))), 4)
    rec["absb_max_deg"] = round(float(np.max(np.abs(b))), 4)
    # angular length along the track (sum of consecutive separations)
    sep = c[:-1].separation(c[1:]).deg
    rec["track_length_deg"] = round(float(np.sum(sep)), 4)

    dist = np.asarray(tt["distance"], dtype=float)
    rec["dist_min_kpc"] = round(float(np.min(dist)), 5)
    rec["dist_max_kpc"] = round(float(np.max(dist)), 5)

    gc = c.transform_to(ac.Galactocentric())
    z = gc.z.to_value(u.kpc)
    rgc = np.sqrt(gc.x.to_value(u.kpc) ** 2 + gc.y.to_value(u.kpc) ** 2 + z ** 2)
    rec["z_min_kpc"] = round(float(np.min(z)), 4)
    rec["z_max_kpc"] = round(float(np.max(z)), 4)
    rec["absz_max_kpc"] = round(float(np.max(np.abs(z))), 4)
    rec["R_gc_min_kpc"] = round(float(np.min(rgc)), 4)
    rec["R_gc_max_kpc"] = round(float(np.max(rgc)), 4)

    # radial velocity: flag whether the column is a real varying track or filler
    rv = np.asarray(tt["radial_velocity"], dtype=float)
    rec["rv_ptp_kms"] = round(float(np.ptp(rv)), 6)
    pmra = np.asarray(tt["pm_ra_cosdec"], dtype=float)
    rec["pmra_ptp_masyr"] = round(float(np.ptp(pmra)), 6)
    rec["dist_ptp_kpc"] = round(float(np.ptp(dist)), 6)

    # ------------------------------------------------------------------
    # JOINT flag+data classification. NEITHER the InfoFlags NOR the data
    # alone is sufficient (see REPORT): galstreams v1.2.1 ships 85 ibata2024
    # tracks whose distance column is identically 1.000 kpc -- a PLACEHOLDER --
    # while InfoFlags claims a distance track exists for 68 of them.
    # ------------------------------------------------------------------
    nuD = len(np.unique(np.round(dist, 6)))
    nuP = len(np.unique(np.round(pmra, 6)))
    nuR = len(np.unique(np.round(rv, 6)))
    is_geom = kind in ("ep", "po")   # great circle built from end points or pole

    # RULE: the library FLAG governs whether a quantity is a measured track.
    # The DATA can only DOWNGRADE that claim, never promote it. Letting the data
    # promote a flag=0 column is wrong: galstreams fills unmeasured columns with
    # filler that is sometimes non-constant junk (see defect notes below).
    defects = []

    # NB: ECSV round-trip leaves float noise (0.9999999999999946), so an exact
    # == 1.0 test silently finds NOTHING. Use a tolerance.
    is_ph1 = np.allclose(dist, 1.0, rtol=0.0, atol=1e-9)
    if rec["has_D"] == 0:
        rec["D_status"] = "GEOMETRIC_INTERPOLATION" if (is_geom and nuD > 1) else "ABSENT"
    elif is_ph1:
        rec["D_status"] = "PLACEHOLDER_1KPC"
        defects.append("distance_flag_set_but_column_is_1kpc_placeholder")
    elif nuD == 1:
        rec["D_status"] = "SINGLE_MEAN_VALUE"
        defects.append("distance_flag_set_but_column_is_constant")
    else:
        rec["D_status"] = "MEASURED_TRACK"

    if rec["has_pm"] == 0:
        rec["pm_status"] = "ABSENT"
    elif nuP == 1:
        rec["pm_status"] = "SINGLE_MEAN_VALUE"
        defects.append("pm_flag_set_but_column_is_constant")
    else:
        rec["pm_status"] = "MEASURED_TRACK"

    # 999 is a null sentinel; |v_r| > 1000 km/s is unphysical for a MW stream
    # (Galactic escape speed is ~550 km/s).
    is_999 = np.allclose(rv, 999.0, rtol=0.0, atol=1e-6)
    rv_absurd = bool(np.max(np.abs(rv)) > 1000.0)
    if rec["has_vrad"] == 0:
        rec["vrad_status"] = "ABSENT"
        if nuR > 1:
            defects.append("vrad_flag_clear_but_column_populated_with_filler")
        if rv_absurd:
            defects.append("vrad_filler_is_unphysical_gt_1000kms")
    elif is_999:
        rec["vrad_status"] = "SENTINEL_999"
        defects.append("vrad_flag_set_but_column_is_999_sentinel")
    elif nuR == 1:
        rec["vrad_status"] = "SINGLE_MEAN_VALUE"
        defects.append("vrad_flag_set_but_column_is_constant")
    elif rv_absurd:
        rec["vrad_status"] = "UNPHYSICAL"
        defects.append("vrad_flag_set_but_values_exceed_1000kms")
    else:
        rec["vrad_status"] = "MEASURED_TRACK"

    rec["data_defects"] = ";".join(defects)

    rec["sky_status"] = "EMPIRICAL_TRACK" if rec["has_empirical_track"] else "GREAT_CIRCLE_ASSUMED"

    # A track is usable for OUT-OF-PLANE 3-D geometry only if the sky track is
    # empirical AND the distance is a genuine measured track.
    rec["usable_3d"] = int(rec["sky_status"] == "EMPIRICAL_TRACK"
                           and rec["D_status"] == "MEASURED_TRACK")
    rec["usable_6d"] = int(rec["usable_3d"]
                           and rec["pm_status"] == "MEASURED_TRACK"
                           and rec["vrad_status"] == "MEASURED_TRACK")

    # z / R_gc are meaningless when the distance is a placeholder or a constant.
    if rec["D_status"] in ("PLACEHOLDER_1KPC",):
        for k in ("z_min_kpc", "z_max_kpc", "absz_max_kpc",
                  "R_gc_min_kpc", "R_gc_max_kpc"):
            rec[k] = ""

    rows.append(rec)

cols = [
    "TrackFileBase", "StreamName", "StreamShortName", "TrackRef", "TrackType",
    "InfoFlags", "has_empirical_track", "has_D", "has_pm", "has_vrad",
    "D_caveat", "pm_caveat", "vrad_caveat",
    "sky_status", "D_status", "pm_status", "vrad_status",
    "usable_3d", "usable_6d", "data_defects",
    "n_track_points", "track_length_deg",
    "pole_ra_deg", "pole_dec_deg", "pole_l_deg", "pole_b_deg",
    "orbit_inc_to_disc_deg",
    "mid_l_deg", "mid_b_deg", "mid_distance_kpc",
    "b_min_deg", "b_max_deg", "absb_min_deg", "absb_max_deg",
    "dist_min_kpc", "dist_max_kpc",
    "z_min_kpc", "z_max_kpc", "absz_max_kpc", "R_gc_min_kpc", "R_gc_max_kpc",
    "width_phi2", "width_pm_phi1_cosphi2", "width_pm_phi2",
    "dist_ptp_kpc", "pmra_ptp_masyr", "rv_ptp_kms",
]

with open(OUT, "w", encoding="utf-8", newline="") as fh:
    fh.write("\t".join(cols) + "\n")
    for r in rows:
        fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")

print("wrote", OUT, "rows", len(rows))

# ---- assertions / census ----
print("\nORPHAN summaries (summary present, track file MISSING upstream): %d" % len(orphans))
for o in orphans:
    print("   ", o)
assert len(rows) + len(orphans) == len(summaries), "row count + orphans != summary file count"

fl = collections.Counter(r["InfoFlags"] for r in rows)
print("\nInfoFlags census (bit0 empirical-sky, bit1 D, bit2 PM, bit3 Vrad):")
for k, v in sorted(fl.items()):
    print("  %s : %3d" % (k, v))
print("\nn tracks                      :", len(rows))
print("n distinct StreamName         :", len(set(r["StreamName"] for r in rows)))
print("n distinct StreamShortName    :", len(set(r["StreamShortName"] for r in rows)))
print("has_empirical_track (sky meas):", sum(r["has_empirical_track"] for r in rows))
print("has_D  (distance track)       :", sum(r["has_D"] for r in rows))
print("has_pm (proper-motion track)  :", sum(r["has_pm"] for r in rows))
print("has_vrad (radial-vel track)   :", sum(r["has_vrad"] for r in rows))
print("FULL 6-D (D & pm & vrad)      :",
      sum(1 for r in rows if r["has_D"] and r["has_pm"] and r["has_vrad"]))
print("FULL 6-D and empirical sky    :",
      sum(1 for r in rows if r["has_empirical_track"] and r["has_D"] and r["has_pm"] and r["has_vrad"]))
print("sky-track ONLY (0 of D/pm/rv) :",
      sum(1 for r in rows if not (r["has_D"] or r["has_pm"] or r["has_vrad"])))
print("with a '2' caveat digit       :",
      sum(1 for r in rows if "2" in r["InfoFlags"]))

# ---- INDEPENDENT NUMERICAL VERIFICATION OF THE FLAGS ----
# A quantity is a genuine *track* only if it actually varies along the stream.
# Where a flag says 'not available' galstreams fills the column with a constant
# (mean/central) value, so peak-to-peak == 0. Cross-check flag vs data.
print("\nFlag-vs-data cross-check (ptp>0 means the column really varies):")
for name, flag, key in (("distance", "has_D", "dist_ptp_kpc"),
                        ("propermot", "has_pm", "pmra_ptp_masyr"),
                        ("radialvel", "has_vrad", "rv_ptp_kms")):
    tol = 1e-6
    both = sum(1 for r in rows if r[flag] and abs(r[key]) > tol)
    flag_no_var = [r["TrackFileBase"] for r in rows if r[flag] and abs(r[key]) <= tol]
    var_no_flag = [r["TrackFileBase"] for r in rows if not r[flag] and abs(r[key]) > tol]
    print("  %-9s flag=1 & varies: %3d | flag=1 but CONSTANT: %2d | flag=0 but varies: %2d"
          % (name, both, len(flag_no_var), len(var_no_flag)))
    for x in flag_no_var[:8]:
        print("       flag=1 yet constant ->", x)
    for x in var_no_flag[:8]:
        print("       flag=0 yet varies   ->", x)

print("\n=== JOINT flag+data status census ===")
for key in ("sky_status", "D_status", "pm_status", "vrad_status"):
    print(" ", key, dict(collections.Counter(r[key] for r in rows)))
u3 = [r for r in rows if r["usable_3d"]]
u6 = [r for r in rows if r["usable_6d"]]
print("\nusable_3d (empirical sky + MEASURED distance track) :", len(u3))
print("usable_6d (that, plus measured PM and Vrad tracks)  :", len(u6))
print("distinct streams among usable_6d                    :",
      len(set(r["StreamName"] for r in u6)))

ph = [r for r in rows if r["D_status"] == "PLACEHOLDER_1KPC"]
print("\nPLACEHOLDER_1KPC tracks (distance identically 1.000 kpc):", len(ph))
print("  refs:", dict(collections.Counter(r["TrackRef"] for r in ph)))
print("  of these, InfoFlags falsely claims a distance track:",
      sum(1 for r in ph if r["has_D"]))

# out-of-plane leverage: orbit inclination to the disc, among usable_3d
inc = sorted(u3, key=lambda r: -r["orbit_inc_to_disc_deg"])
print("\nTop out-of-plane (most nearly POLAR) usable_3d tracks, by orbit "
      "inclination to the Galactic disc:")
print("   %-34s %7s %9s %9s %9s" % ("track", "inc_deg", "|z|max", "Rgc_max", "npts"))
for r in inc[:15]:
    print("   %-34s %7.2f %9s %9s %9d" % (r["TrackFileBase"], r["orbit_inc_to_disc_deg"],
                                          r["absz_max_kpc"], r["R_gc_max_kpc"],
                                          r["n_track_points"]))
hi = [r for r in u3 if r["absz_max_kpc"] != "" and float(r["absz_max_kpc"]) >= 10.0]
print("\nusable_3d tracks reaching |z| >= 10 kpc off the Galactic plane:", len(hi))
hi20 = [r for r in u3 if r["absz_max_kpc"] != "" and float(r["absz_max_kpc"]) >= 20.0]
print("usable_3d tracks reaching |z| >= 20 kpc off the Galactic plane:", len(hi20))
