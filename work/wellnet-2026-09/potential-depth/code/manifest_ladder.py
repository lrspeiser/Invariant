"""Write the manifest for potential_depth_ladder.csv."""
import collections
import csv
import datetime
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
P = os.path.join(LANE, "potential_depth_ladder.csv")

rows = list(csv.DictReader(open(P, encoding="utf-8")))
b = open(P, "rb").read()

UNITS = {
    "system": "object identifier, prefixed by source",
    "class": "ladder rung name",
    "class_rank": "1..6, isolated galaxies to massive clusters",
    "source": "catalogue",
    "probe": "measurement type for g_obs",
    "r_kpc": "kpc",
    "Mb_Msun": "Msun -- baryonic mass implied by g_bar at this radius",
    "g_bar": "m s^-2 -- G M_b(<r)/r^2 (disk field for SPARC)",
    "g_obs": "m s^-2",
    "nu_obs": "dimensionless -- g_obs/g_bar",
    "abs_Phi_b": "m^2 s^-2 -- |Phi_b(r)| with the stated outer boundary condition",
    "S_shape": "dimensionless -- |Phi_b|/(g_bar r); >= 1 for spherical M_b",
    "e_lg_gbar": "dex, 1 sigma, per-point random",
    "e_lg_gobs": "dex, 1 sigma, per-point random",
    "sys_lg_Mb": "dex, 1 sigma, COHERENT per system (mass calibration)",
    "tier": "1 = primary, 2 = flagged (stellar-mass-only baryons)",
    "phi_method": "how |Phi_b| was obtained for this row",
    "baryon_method": "what M_b contains",
    "gobs_method": "how g_obs was measured",
    "assumes": "physical assumptions the row's g_obs depends on",
}

cnt = collections.Counter(r["class"] for r in rows)
src = collections.Counter(r["source"] for r in rows)

man = {
    "file": os.path.basename(P),
    "produced_by": "code/ladder.py (this lane)",
    "retrieved_utc": datetime.datetime.now(datetime.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ"),
    "sha256": hashlib.sha256(b).hexdigest(),
    "bytes": len(b),
    "row_count": len(rows),
    "column_count": len(rows[0]),
    "columns": [{"name": k, "unit": UNITS[k]} for k in rows[0]],
    "n_systems": len(set(r["system"] for r in rows)),
    "rows_by_class": dict(cnt),
    "rows_by_source": dict(src),
    "conventions": {
        "Phi_b": ("Phi_b(r) = -[ Int_r^Rmax g_bar dr' + g_bar(Rmax)*Rmax ], "
                  "i.e. all baryonic mass inside Rmax and a Newtonian "
                  "point-mass tail outside it. Rmax is the outermost radius at "
                  "which that system's baryons are measured. Identical to "
                  "work/wellnet-2026-09/phi_rank.py tail='point'."),
        "lower_bound": ("For spherical M_b(<r), |Phi_b(r)| >= g_bar(r)*r "
                        "exactly, so every single-radius row is a strict LOWER "
                        "BOUND on |Phi_b|, not an estimate."),
        "stellar_masses": ("Measured only for the 12 Gonzalez+2013 systems and "
                           "7 of 12 X-COP clusters. Everywhere else stars come "
                           "from log10(M*/Mgas) = 7.598 - 0.620 log10(Mgas/Msun), "
                           "fitted here on those 12 Gonzalez systems "
                           "(rms 0.064 dex over log10 Mgas = 12.82-13.83) and "
                           "EXTRAPOLATED below that range for the groups."),
        "g_obs_optical_groups": ("eta sigma^2 / r_rms with eta = 2 (isotropic "
                                 "isothermal sphere). Anisotropy is not "
                                 "measured; eta in [1,3] is the plausible range "
                                 "and moves log nu by -0.30 to +0.18 dex."),
    },
    "sealed_holdouts": ("KiDS weak lensing and wide binaries are NOT in this "
                        "file and were dropped by probe name before any value "
                        "was read (66 rows of the 4093-row source table)."),
    "known_limitations": [
        "The optical-group rung (tier 2) has stellar-only baryons: its g_bar is "
        "a lower bound and its nu_obs an upper bound by a factor of a few.",
        "Sun+2009 r500 quantities exist for only 23 of 43 groups; rows flagged "
        "'S' (scaling relation) are excluded, so most Sun systems contribute "
        "one radius (r2500) and therefore S = 1.",
        "Every X-ray rung assumes hydrostatic equilibrium. A uniform 10-30% HSE "
        "bias is a CLASS-LEVEL systematic and is budgeted in results.json.",
        "SPARC rows use Upsilon* = 0.5 disk / 0.7 bulge, which this programme "
        "has separately measured to disagree with dynamical determinations by "
        "0.4-0.55 dex; that is also a class-level systematic.",
    ],
}
with open(P + ".manifest.json", "w", encoding="utf-8", newline="\n") as f:
    json.dump(man, f, indent=2)
print(f"wrote {P}.manifest.json  rows={len(rows)} cols={len(rows[0])} "
      f"systems={man['n_systems']}")
for k, v in sorted(cnt.items()):
    print(f"   {k:<18} {v}")
