"""Quantify the claim that member-galaxy line-of-sight depth z_a is NOT measured.

For each target cluster we have measured sky coordinates (RA, Dec) and, for
spectroscopic members, a redshift.  The two sky coordinates convert to physical
transverse coordinates (x_a, y_a) at the cluster's angular diameter distance
with negligible error.  The redshift does NOT convert to a depth: the observed
cz is the sum of a Hubble-flow term and a peculiar-velocity term, and inside a
cluster the peculiar term dominates by a large factor.

This script computes, per cluster:
  * kpc per arcsec, so the astrometric error can be put in physical units;
  * the Hubble-flow velocity produced by a 1 Mpc real depth displacement;
  * that velocity as a fraction of the measured velocity dispersion, i.e. the
    signal-to-noise with which 1 Mpc of depth could ever be detected;
  * the spurious depth range implied if the measured dispersion were (wrongly)
    read as Hubble flow, compared with the cluster's own physical size.

sigma_v values are literature values, cited per cluster, and are used ONLY to
set the scale of this argument -- nothing is fitted here.
"""
import json
import os
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

LANE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
C_KMS = 299792.458

# z, sigma_v (km/s), citation for sigma_v, approximate R200 (Mpc) for scale
CLUSTERS = [
    ("Abell 2744",        0.3080, 1497, "Owers et al. 2011, ApJ 728, 27 (global, 3 substructures)", 2.0),
    ("MACS J0416.1-2403", 0.3960, 1000, "Balestra et al. 2016, ApJS 224, 33 / Bergamini et al. 2021", 1.8),
    ("MACS J0717.5+3745", 0.5458, 1660, "Ebeling et al. 2014, ApJ 781, L40 (quadruple merger)", 2.1),
    ("MACS J1149.5+2223", 0.5420, 1840, "Grillo et al. 2016, ApJ 822, 78 / Golovich et al. 2019", 2.1),
    ("Abell S1063",       0.3480, 1380, "Gomez-Reyes / Caminha et al. 2016, A&A 587, A80 (CLASH-VLT)", 2.1),
    ("Abell 370",         0.3750, 1170, "Lagattuta et al. 2019, MNRAS 485, 3738 (MUSE + literature)", 2.0),
    ("Abell 2029",        0.0773, 1152, "Sohn et al. 2019, ApJ 871, 129 (Hectospec, ~1200 members)", 2.0),
]

# Typical astrometric precision of the acquired catalogues (Gaia-aligned HST/
# Subaru astrometry).  Conservative.
ASTROM_ARCSEC = 0.10

rows = []
for name, z, sig, cite, r200 in CLUSTERS:
    kpc_per_arcsec = COSMO.kpc_proper_per_arcmin(z).value / 60.0
    Hz = COSMO.H(z).to(u.km / u.s / u.Mpc).value
    # A 1 Mpc PROPER depth displacement maps to this much observed velocity.
    v_per_Mpc = Hz  # km/s per proper Mpc, at this redshift
    snr_1Mpc = v_per_Mpc / sig
    # If the full measured dispersion were read as Hubble flow, this depth:
    spurious_depth_Mpc = sig / Hz
    rows.append({
        "cluster": name,
        "z": z,
        "kpc_per_arcsec": round(kpc_per_arcsec, 4),
        "transverse_position_error_kpc_at_0.10_arcsec": round(kpc_per_arcsec * ASTROM_ARCSEC, 4),
        "H_of_z_km_s_Mpc": round(Hz, 2),
        "sigma_v_km_s": sig,
        "sigma_v_reference": cite,
        "velocity_from_1_Mpc_of_depth_km_s": round(v_per_Mpc, 2),
        "detectability_of_1_Mpc_depth_in_units_of_sigma_v": round(snr_1Mpc, 4),
        "spurious_depth_if_sigma_v_read_as_Hubble_flow_Mpc": round(spurious_depth_Mpc, 2),
        "approx_R200_Mpc": r200,
        "spurious_depth_over_2R200": round(spurious_depth_Mpc / (2 * r200), 2),
    })

out = {
    "question": "Which of (x_a, y_a, z_a) are measured for cluster member galaxies?",
    "answer_x_y": (
        "MEASURED. Every acquired member catalogue carries RA and Dec. At the adopted "
        "astrometric precision of 0.10 arcsec (Gaia-aligned HST/Subaru astrometry; the "
        "BUFFALO strong-lensing catalogue is explicitly Gaia-aligned), the transverse "
        "position error is 0.2 to 0.7 kpc depending on cluster redshift -- three to four "
        "orders of magnitude below the scales the downstream model resolves. For all "
        "practical purposes x_a and y_a are exact."),
    "answer_z": (
        "NOT MEASURED. There is no observable that yields the line-of-sight depth of a "
        "cluster member. The only line-of-sight datum is the redshift, and the observed "
        "cz is the sum of a Hubble-flow term H(z)*d and a peculiar-velocity term v_pec. "
        "These two enter as a single scalar and are exactly degenerate: no amount of "
        "spectroscopic precision separates them. The table below quantifies how badly the "
        "peculiar term dominates."),
    "consequence": (
        "The downstream code MUST sample z_a rather than read it, and every prediction "
        "must be marginalised over that sampling. Treating spectroscopic redshift as a "
        "depth coordinate would inject a spurious line-of-sight elongation of order ten "
        "times the cluster's true extent."),
    "caveats": [
        "PHOTOMETRIC redshifts are worse still: sigma_z ~ 0.03-0.05 (1+z) corresponds to "
        "hundreds of Mpc of apparent depth. They constrain membership, not position.",
        "The SOURCE redshift of a strong-lensing image IS a genuine third measured "
        "quantity, but it fixes the source's cosmological distance, not the depth of any "
        "cluster member. It does not help with z_a.",
        "A2029 at z=0.077 is the least unfavourable case only because H(z) is smallest "
        "there; it is still hopeless in absolute terms.",
        "Line-of-sight elongation of the cluster as a WHOLE is separately constrained by "
        "combining lensing with X-ray/SZ (e.g. Umetsu et al. 2022 for Abell 370). That "
        "constrains one global shape parameter, not the depth of individual galaxies.",
    ],
    "assumptions": "FlatLambdaCDM H0=70, Om0=0.3, used only for distance and H(z) conversions.",
    "astrometric_precision_assumed_arcsec": ASTROM_ARCSEC,
    "per_cluster": rows,
}

dest = os.path.join(LANE, "los_depth_argument.json")
with open(dest, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("%-20s %6s %8s %10s %12s %10s %8s" % (
    "cluster", "z", "kpc/\"", "dx_err/kpc", "1Mpc->km/s", "S/N(1Mpc)", "fake/2R200"))
for r in rows:
    print("%-20s %6.4f %8.3f %10.3f %12.1f %10.4f %8.2f" % (
        r["cluster"], r["z"], r["kpc_per_arcsec"],
        r["transverse_position_error_kpc_at_0.10_arcsec"],
        r["velocity_from_1_Mpc_of_depth_km_s"],
        r["detectability_of_1_Mpc_depth_in_units_of_sigma_v"],
        r["spurious_depth_over_2R200"]))
