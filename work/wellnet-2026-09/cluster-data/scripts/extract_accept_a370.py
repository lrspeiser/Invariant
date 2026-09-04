"""Extract the Abell 370 profile from the raw ACCEPT all_profiles.dat.

The earlier gas pass missed this one: ACCEPT names the cluster ABELL_0370 with
a zero-padded number, so a search for 'ABELL_370' finds nothing.  Same class of
identifier trap as ABELL_1063S standing for Abell S1063.  The result is a
32-bin Chandra deprojection, versus the 4-shell Umetsu+2022 deprojection that
was the only Abell 370 gas profile in the lane until now.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch import write_manifest, LANE  # noqa: E402

SRC = os.path.join(LANE, "gas", "accept_all_profiles.dat")
DEST = os.path.join(LANE, "gas", "accept_ABELL_0370.tsv")
KEY = "ABELL_0370"

with open(SRC, encoding="utf-8", errors="replace") as f:
    lines = f.read().splitlines()

header = lines[0].lstrip("#").split()
rows = [l.split() for l in lines[1:] if l.strip() and l.split()[0] == KEY]
assert rows, "no rows for " + KEY
ncol = {len(r) for r in rows}
assert ncol == {len(header)}, "ragged rows: %s vs header %d" % (ncol, len(header))

with open(DEST, "w", encoding="utf-8", newline="\n") as f:
    f.write("\t".join(header) + "\n")
    for r in rows:
        f.write("\t".join(r) + "\n")

i_rin, i_rout = header.index("Rin"), header.index("Rout")
r_min = min(float(r[i_rin]) for r in rows) * 1000.0
r_max = max(float(r[i_rout]) for r in rows) * 1000.0

UNITS = {
    "Name": "ACCEPT cluster identifier", "Rin": "Mpc", "Rout": "Mpc",
    "nelec": "cm-3, deprojected electron density", "neerr": "cm-3",
    "Kitpl": "keV cm2, entropy (interpolated T)", "Kflat": "keV cm2, entropy (flat T)",
    "Kerr": "keV cm2", "Pitpl": "dyne cm-2", "Pflat": "dyne cm-2", "Perr": "dyne cm-2",
    "Mgrav": "g, HYDROSTATIC mass - NOT an observation", "Merr": "g",
    "Tx": "keV, PROJECTED spectroscopic temperature interpolated onto the density bins",
    "Txerr": "keV", "Lambda": "erg cm3 s-1, cooling function",
    "tcool5/2": "Gyr", "t52err": "Gyr", "tcool3/2": "Gyr", "t32err": "Gyr",
}

write_manifest(
    DEST,
    "http://www.pa.msu.edu/astro/MC2/accept/data/all_profiles.dat (via Wayback snapshot 20141022053622)",
    exact_query="rows of accept_all_profiles.dat whose column 1 (Name) == 'ABELL_0370'",
    note=("Abell 370. ACCEPT (Cavagnolo et al. 2009, ApJS 182, 12) Chandra ACIS deprojected "
          "profile, " + str(len(rows)) + " radial bins spanning " + ("%.1f-%.1f" % (r_min, r_max)) +
          " kpc. n_e(r) [nelec] IS a deprojection of the Chandra surface brightness assuming "
          "spherical symmetry. T(r) [Tx] IS a radial profile but is the PROJECTED spectroscopic "
          "temperature interpolated onto the density bins, not a deprojected 3D temperature. NO "
          "core excision. Mgrav is hydrostatic and must not be treated as an observation. "
          "IDENTIFIER TRAP: ACCEPT zero-pads the Abell number, so this cluster is 'ABELL_0370', "
          "not 'ABELL_370'; an earlier pass searching for the unpadded name concluded Abell 370 "
          "was absent from ACCEPT and fell back on the 4-shell Umetsu+2022 deprojection. This is "
          "the same trap as ACCEPT's 'ABELL_1063S' standing for Abell S1063. This file supersedes "
          "a370_umetsu2022_chandra_deprojection.tsv as the primary Abell 370 gas profile; the "
          "Umetsu file is retained because it carries a genuinely DEPROJECTED temperature and a "
          "core-excised global kT, which ACCEPT does not."),
    extraction=("Exact subset of the raw ACCEPT all_profiles.dat: every row with Name == "
                "'ABELL_0370'. Values copied verbatim (whitespace re-delimited to TAB, no "
                "numerical change). Header row = ACCEPT column names. Row count and column "
                "count asserted against the raw file before writing. Raw upstream file kept "
                "alongside as accept_all_profiles.dat."),
    row_count=len(rows), column_count=len(header),
    columns=[{"name": h, "unit": UNITS.get(h, "")} for h in header],
    extra={"cluster": "Abell 370", "product": "gas_xray_profile",
           "accept_name_key": KEY, "redshift": 0.375,
           "core_excised": False, "deprojected": True,
           "derived_assumes_newtonian_hse": False,
           "r_min_kpc": round(r_min, 2), "r_max_kpc": round(r_max, 2),
           "n_radial_bins": len(rows),
           "has_ne_profile": True, "has_T_profile": True,
           "T_profile_is_projected_not_deprojected": True,
           "contains_hse_mass_column": True,
           "hse_mass_columns": ["Mgrav", "Merr"],
           "raw_response_file": "accept_all_profiles.dat"})

print("wrote %s: %d rows x %d cols, %.1f-%.1f kpc" %
      (os.path.basename(DEST), len(rows), len(header), r_min, r_max))
