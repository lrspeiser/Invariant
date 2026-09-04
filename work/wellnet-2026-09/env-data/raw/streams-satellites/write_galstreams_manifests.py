import csv
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from _manifest import write_manifest  # noqa: E402

RET = "2026-09-03T21:20:00Z"


def tsv_cols_rows(p):
    with open(p, encoding="utf-8") as fh:
        rd = csv.reader(fh, delimiter="\t")
        cols = next(rd)
        n = sum(1 for _ in rd)
    return cols, n


# ---------------------------------------------------------------- raw tarballs
write_manifest(
    os.path.join(BASE, "galstreams_v1.2.1.tar.gz"),
    source_url="https://codeload.github.com/cmateu/galstreams/tar.gz/refs/tags/v1.2.1",
    query="HTTP GET https://codeload.github.com/cmateu/galstreams/tar.gz/refs/tags/v1.2.1",
    columns=[], row_count=None,
    extraction="Unmodified upstream release tarball of the galstreams package, tag v1.2.1 "
               "(published 2026-06-03T17:20:14Z per the GitHub releases API). Contains 437 "
               "track ECSV data files plus the master_log. ONLY data files were read; no code "
               "from this archive was imported or executed.",
    measurement_or_model="MIXED - see galstreams_track_summary.tsv for a per-track label.",
    note="galstreams: Mateu 2023, MNRAS 520, 5225 (doi:10.1093/mnras/stad321), "
         "arXiv:2204.10326. Repo github.com/cmateu/galstreams. The published paper "
         "describes v1.0 (126 tracks / 95 streams); this tag v1.2.1 ships 217 tracks.",
    retrieved_utc=RET,
    extra={"n_track_ecsv_files": 437, "n_summary_ecsv_files": 220,
           "n_main_track_ecsv_files": 217},
)

write_manifest(
    os.path.join(BASE, "galstreams_paper_2204.10326.tar.gz"),
    source_url="https://arxiv.org/e-print/2204.10326",
    query="HTTP GET https://arxiv.org/e-print/2204.10326",
    columns=[], row_count=None,
    extraction="Unmodified arXiv LaTeX source tarball for Mateu 2023 (galstreams paper). "
               "Used to recover (a) the authoritative InfoFlags legend and (b) the published "
               "summary table, which is SPLIT across super_table_1of2.tex and "
               "super_table_2of2.tex.",
    measurement_or_model="PAPER SOURCE (documentation).",
    note="Table t:super_summary_table is split across two files. Both were parsed; "
         "63 + 63 = 126 rows recovered, matching the paper's own \\Ntracks=126 macro.",
    retrieved_utc=RET,
)

# ------------------------------------------------------- master log (raw data)
ml = os.path.join(BASE, "galstreams_data", "galstreams", "lib", "master_log.txt")
with open(ml, encoding="utf-8", errors="replace") as fh:
    lines = [l for l in fh.read().splitlines() if l.strip()]
write_manifest(
    ml,
    source_url="https://codeload.github.com/cmateu/galstreams/tar.gz/refs/tags/v1.2.1",
    source_file_within_archive="galstreams-1.2.1/galstreams/lib/master_log.txt",
    query="tarfile extraction of galstreams/lib/master_log.txt from the v1.2.1 tarball",
    columns=[
        {"name": "Imp", "unit": "implementation method: ep=end points, po=pole, st=stars/knots"},
        {"name": "On", "unit": "1 = library default track for this stream, 0 = alternate"},
        {"name": "TrackName", "unit": "track identifier"},
        {"name": "Name", "unit": "stream name"},
        {"name": "TrackRefs", "unit": "track reference key"},
        {"name": "TrackRefsLatex", "unit": "track reference (LaTeX)"},
        {"name": "Notes", "unit": "free text"},
        {"name": "from-Members", "unit": "1 = track derived from member stars"},
        {"name": "width_phi2", "unit": "deg"},
        {"name": "width_phi2_Ref", "unit": "reference"},
        {"name": "width_pm_phi1_cosphi2", "unit": "mas/yr"},
        {"name": "width_pm_phi1_cosphi2_Ref", "unit": "reference"},
        {"name": "width_pm_phi2", "unit": "mas/yr"},
        {"name": "width_pm_phi2_Ref", "unit": "reference"},
        {"name": "total_v_sigma", "unit": "km/s"},
        {"name": "total_v_sigma_Ref", "unit": "reference"},
        {"name": "Lv_stream", "unit": "dimensionless"},
        {"name": "L_Ref", "unit": "reference"},
        {"name": "width_comment", "unit": "free text"},
    ],
    row_count=len(lines) - 1,
    extraction="Raw upstream file, unmodified. Whitespace-aligned text table; first line is "
               "a '#'-prefixed header.",
    measurement_or_model="MEASUREMENT (compiled literature stream widths/dispersions) plus "
                         "bookkeeping columns. The 'Imp' column records whether the celestial "
                         "track is a GREAT-CIRCLE CONSTRUCTION (ep/po = MODEL of the track "
                         "shape) or built from member stars (st = MEASUREMENT).",
    note="219 data rows.",
    retrieved_utc=RET,
)

# -------------------------------------------------------------- cleaned tables
p = os.path.join(BASE, "galstreams_track_summary.tsv")
cols, n = tsv_cols_rows(p)
UNITS = {
    "TrackFileBase": "galstreams track file stem",
    "StreamName": "stream name", "StreamShortName": "short name",
    "TrackRef": "first-author+year key", "TrackType": "ep=end points, po=pole, st=stars/knots",
    "InfoFlags": "4-char galstreams flag (sky/D/PM/Vrad)",
    "has_empirical_track": "1 = celestial track empirical, 0 = great circle assumed",
    "has_D": "1 = distance flagged available", "has_pm": "1 = proper motion flagged available",
    "has_vrad": "1 = radial velocity flagged available",
    "D_caveat": "1 = flag digit is '2' (available with caveat)",
    "pm_caveat": "1 = flag digit is '2'", "vrad_caveat": "1 = flag digit is '2'",
    "sky_status": "EMPIRICAL_TRACK | GREAT_CIRCLE_ASSUMED",
    "D_status": "MEASURED_TRACK | SINGLE_MEAN_VALUE | GEOMETRIC_INTERPOLATION | PLACEHOLDER_1KPC",
    "pm_status": "MEASURED_TRACK | SINGLE_MEAN_VALUE | ABSENT",
    "vrad_status": "MEASURED_TRACK | SINGLE_MEAN_VALUE | ABSENT",
    "usable_3d": "1 = empirical sky track AND measured distance track",
    "usable_6d": "1 = usable_3d AND measured PM track AND measured Vrad track",
    "n_track_points": "count", "track_length_deg": "deg",
    "pole_ra_deg": "deg", "pole_dec_deg": "deg", "pole_l_deg": "deg", "pole_b_deg": "deg",
    "orbit_inc_to_disc_deg": "deg; 90-|pole_b|; 0=orbit in the disc plane, 90=polar orbit",
    "mid_l_deg": "deg", "mid_b_deg": "deg", "mid_distance_kpc": "kpc",
    "b_min_deg": "deg", "b_max_deg": "deg", "absb_min_deg": "deg", "absb_max_deg": "deg",
    "dist_min_kpc": "kpc", "dist_max_kpc": "kpc",
    "z_min_kpc": "kpc (Galactocentric)", "z_max_kpc": "kpc (Galactocentric)",
    "absz_max_kpc": "kpc (Galactocentric); blank when D_status=PLACEHOLDER_1KPC",
    "R_gc_min_kpc": "kpc", "R_gc_max_kpc": "kpc",
    "width_phi2": "deg", "width_pm_phi1_cosphi2": "mas/yr", "width_pm_phi2": "mas/yr",
    "dist_ptp_kpc": "kpc (peak-to-peak along track)",
    "pmra_ptp_masyr": "mas/yr (peak-to-peak)", "rv_ptp_kms": "km/s (peak-to-peak)",
}
write_manifest(
    p,
    source_url="https://codeload.github.com/cmateu/galstreams/tar.gz/refs/tags/v1.2.1",
    source_file_within_archive="galstreams-1.2.1/galstreams/tracks/*.ecsv (217 track + 217 summary files)",
    query="build_galstreams_summary.py -- reads each track.*.summary.ecsv and its "
          "track.*.ecsv with astropy.table.QTable.read; no galstreams code executed",
    columns=[{"name": c, "unit": UNITS.get(c, "")} for c in cols],
    row_count=n,
    extraction=(
        "One row per galstreams v1.2.1 track. Flag columns are transcribed verbatim from "
        "the ECSV summary files. Geometry columns (pole_*, orbit_inc_to_disc_deg, b_*, z_*, "
        "R_gc_*) are pure coordinate transforms of the published track using astropy's "
        "default Galactocentric frame (R0=8.122 kpc, z_sun=20.8 pc). NO gravitational "
        "potential, mass model or halo enters any column."
    ),
    measurement_or_model=(
        "PER-ROW. sky_status/D_status/pm_status/vrad_status give the label. "
        "EMPIRICAL_TRACK + MEASURED_TRACK = MEASUREMENT. GREAT_CIRCLE_ASSUMED = the "
        "celestial track is a geometric great-circle interpolation between measured end "
        "points or about a measured pole (MODEL of track shape, not of gravity). "
        "GEOMETRIC_INTERPOLATION distance = linear interpolation between two measured end-point "
        "distances (MODEL). SINGLE_MEAN_VALUE = one constant literature value, not a track. "
        "PLACEHOLDER_1KPC = the distance column is identically 1.000 kpc and is NOT DATA. "
        "IMPORTANT: galstreams excludes orbit-fit/PREDICTED PM and Vrad tracks by policy "
        "(Mateu 2023), and reports OBSERVED heliocentric values without solar-reflex "
        "correction, so no Galactic potential is presupposed anywhere in the library."
    ),
    note=(
        "217 tracks; 147 distinct StreamName; 127 distinct StreamShortName. "
        "CLASSIFICATION RULE: the library InfoFlag governs whether a quantity counts as a "
        "measured track; the data can only DOWNGRADE that claim, never promote it. "
        "Census: sky EMPIRICAL_TRACK 192 / GREAT_CIRCLE_ASSUMED 25; "
        "distance MEASURED_TRACK 69, PLACEHOLDER_1KPC 68, SINGLE_MEAN_VALUE 2, "
        "GEOMETRIC_INTERPOLATION 6, ABSENT 72; "
        "PM MEASURED_TRACK 165, SINGLE_MEAN_VALUE 6, ABSENT 46; "
        "Vrad MEASURED_TRACK 98, UNPHYSICAL 15, ABSENT 104. "
        "usable_3d=69 (60 streams), usable_6d=33 (30 streams). "
        "26 usable_3d tracks reach |z|>=10 kpc, 11 reach |z|>=20 kpc. "
        "--- UPSTREAM DEFECTS FOUND IN galstreams v1.2.1 (102 tracks carry at least one) --- "
        "(1) 68 ibata2024 tracks have InfoFlags asserting a distance track while the "
        "distance column is identically 1.000 kpc, a placeholder (float noise at 1e-15, so "
        "an exact ==1.0 test silently finds nothing). Their z/R_gc columns are left BLANK here. "
        "(2) 15 tracks advertise InfoFlags=1111 (full 6-D) but their radial-velocity column "
        "is unphysical, exceeding 1000 km/s against a Galactic escape speed of ~550 km/s; "
        "the worst, track.st.Hydrus.ibata2024, reaches 9,561,412 km/s (32x the speed of light) "
        "and track.st.NGC1261b.ibata2024 reaches -32,929,072 km/s. The percentile pattern "
        "(sane median, divergent tails) indicates spline blow-up at the track ends. "
        "(3) 16 tracks have the Vrad flag CLEAR yet a populated, non-constant Vrad column "
        "(filler that must not be read as data). "
        "(4) track.st.Pal5.pricewhelan2019 has Vrad identically 999.0, a null sentinel. "
        "(5) 6 tracks have the PM flag set but a constant PM column. "
        "(6) 3 summary files (Jhelum-broad/-narrow/-spur, viswanathan2023) have NO "
        "corresponding track file in the v1.2.1 tarball and are excluded from this table "
        "(220 summary files, 217 tracks). "
        "CONSEQUENCE: InfoFlags must NOT be trusted on its own; every quantity was "
        "cross-checked against the data, and the per-row 'data_defects' column records "
        "each disagreement."
    ),
    retrieved_utc=RET,
)

p = os.path.join(BASE, "galstreams_paper_supertable.tsv")
cols, n = tsv_cols_rows(p)
U2 = {"StreamName": "stream name", "TrackName": "track id", "InfoFlags": "4-char flag",
      "Imp": "ep|po|st", "On": "1 = default track", "Length_deg": "deg",
      "RA_i_deg": "deg", "DEC_i_deg": "deg", "D_i_kpc": "kpc",
      "RA_f_deg": "deg", "DEC_f_deg": "deg", "D_f_kpc": "kpc",
      "TRefs": "track reference code", "DRefs": "discovery reference code"}
write_manifest(
    p,
    source_url="https://arxiv.org/e-print/2204.10326",
    source_file_within_archive="super_table_1of2.tex + super_table_2of2.tex",
    query="extract_galstreams_supertable.py -- parses BOTH halves of the split LaTeX table",
    columns=[{"name": c, "unit": U2.get(c, "")} for c in cols],
    row_count=n,
    extraction="Verbatim transcription of the published table. No unit conversion, no "
               "derivation. GUARDED AGAINST THE SPLIT-TABLE FAILURE MODE: 63 rows from "
               "1of2 + 63 rows from 2of2 = 126 = the paper's own \\Ntracks macro (asserted).",
    measurement_or_model="MEASUREMENT (published end points, lengths, distances) with the "
                         "InfoFlags/Imp columns labelling which tracks are great-circle "
                         "constructions rather than empirical tracks.",
    note="126 rows = paper \\Ntracks. 101 distinct StreamName values; the paper's "
         "\\Nunique=95 counts streams AFTER merging compound names (AAU-ATLAS/AAU-AliqaUma, "
         "Cetus/Cetus-New/Cetus-Palca, Jhelum/Jhelum-a/Jhelum-b, M68/M68-Fjorm, "
         "NGC3201/NGC3201-Gjoll), so 101 != 95 is expected, not a transcription error. "
         "This table describes galstreams v1.0; the shipped v1.2.1 has 217 tracks.",
    retrieved_utc=RET,
)
print("\nDONE")
