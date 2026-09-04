"""DERIVED product: attach a host cluster to every SAMI cluster-region target.

sami_dr3.InputCatClustersDR3 carries R/R200, v_pec/sigma_200 and a membership
flag but NOT the name of the host cluster.  This script assigns the host by
nearest sky position to the eight Owers et al. 2017 Table 1 centres, and then
VALIDATES the assignment by recomputing R/R200 from first principles
(angular separation x angular diameter distance / R200) and comparing with the
published R_on_rtwo column.  A wrong assignment would blow that comparison up.

Nothing here overwrites a raw file.  Every host quantity is tagged
OBSERVABLE or MODEL-DERIVED in the manifest.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM

OUT = os.path.dirname(os.path.abspath(__file__))
# Owers et al. 2017 Section 1: H0 = 70, Om = 0.3, OL = 0.7.
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    clus = pd.read_csv(os.path.join(OUT, "owers2017_table1_clusters.tsv"), sep="\t")
    assert len(clus) == 8, "cluster table has %d rows" % len(clus)

    gal = pd.read_csv(
        os.path.join(OUT, "sami_dr3_InputCatClustersDR3.tsv"), sep="\t",
        dtype={"CATID": str},
    )
    assert len(gal) == 1433, "InputCatClustersDR3 has %d rows, expected 1433" % len(gal)

    cc = SkyCoord(clus.RAdeg.values * u.deg, clus.DEdeg.values * u.deg)
    gc = SkyCoord(gal.RA_OBJ.values * u.deg, gal.DEC_OBJ.values * u.deg)

    # The first four CATID digits encode the observing field.  Assign each
    # PREFIX (not each galaxy) to the nearest Table 1 centre using the median
    # position of its galaxies.  Per-galaxy nearest-centre assignment fails:
    # APMCC 0917 and Abell 4038 lie 1.9 deg apart and were observed on shared
    # 2dF fields (Owers+2017 Table 2), so their target circles overlap and 10
    # Abell 4038 targets are closer to the APMCC 0917 centre than to their own.
    gal["prefix"] = gal.CATID.str[:4]
    pmap, pinfo = {}, []
    for p, g in gal.groupby("prefix"):
        med = SkyCoord(np.median(g.RA_OBJ) * u.deg, np.median(g.DEC_OBJ) * u.deg)
        s = np.array([med.separation(c).deg for c in cc])
        j = int(np.argmin(s))
        pmap[p] = clus.Name.values[j]
        pinfo.append((p, clus.Name.values[j], len(g), float(s[j]), float(np.sort(s)[1])))
    assert len(set(pmap.values())) == 8, "prefixes do not cover all eight clusters"
    print("CATID prefix -> nearest Table 1 centre (by median target position):")
    for p, c, n_, s1, s2 in pinfo:
        print("   %s -> %-11s n=%-4d  d=%.3f deg   next nearest %.3f deg" % (p, c, n_, s1, s2))
        assert s2 > 3 * s1 or s1 < 0.3, "prefix %s host is ambiguous" % p
    gal["cluster"] = gal.prefix.map(pmap)
    gal["sep_deg"] = gc.separation(
        SkyCoord(clus.set_index("Name").loc[gal.cluster, "RAdeg"].values * u.deg,
                 clus.set_index("Name").loc[gal.cluster, "DEdeg"].values * u.deg)).deg

    for c in ["z_clus", "sigma_200", "e_sigma_200", "R_200",
              "M_200_caustic", "e_M_200_caustic", "M_200_virial",
              "e_M_200_virial", "N_mem_R200", "N_mem_2R200", "RAdeg", "DEdeg"]:
        gal["host_" + c] = clus.set_index("Name").loc[gal.cluster, c].values

    # --- independent reconstruction of R/R200 -----------------------------
    # Croom+2021 Sect. 8.2: "projected angular distance of the target to the
    # cluster centre ... converted using the angular diameter distance to the
    # cluster", normalised by R200.
    d_a = COSMO.angular_diameter_distance(gal.host_z_clus.values).to(u.Mpc).value
    gal["R_proj_Mpc_direct"] = np.radians(gal.sep_deg.values) * d_a
    gal["R_on_R200_recomputed"] = gal.R_proj_Mpc_direct / gal.host_R_200.values
    # exact inverse of the published column, i.e. the observable projected radius
    gal["R_proj_Mpc_from_cat"] = gal.R_on_rtwo.values * gal.host_R_200.values
    gal["v_pec_kms"] = gal.V_on_sigma.values * gal.host_sigma_200.values

    # Absolute-plus-relative tolerance: the BCG of each cluster sits at
    # R/R200 ~ 1e-6, where a fractional test is meaningless.
    dabs = np.abs(gal.R_on_R200_recomputed - gal.R_on_rtwo)
    frac = dabs / np.maximum(gal.R_on_rtwo, 1e-6)
    ok = (dabs < 0.01) | (frac < 0.02)
    big = gal.R_on_rtwo > 0.05
    print("\nR/R200 recomputed vs published (R/R200 > 0.05, n=%d): "
          "median frac.diff %+.5f   p68 |frac.diff| %.5f   max |frac.diff| %.5f"
          % (big.sum(), np.median(frac[big]), np.percentile(np.abs(frac[big]), 68),
             np.abs(frac[big]).max()))
    print("passing |dR/R200| < 0.01 or |frac| < 0.02: %d / %d (%.2f per cent)"
          % (ok.sum(), len(gal), 100 * ok.mean()))
    by = gal.assign(ok=ok).groupby("cluster").ok.agg(["sum", "size"])
    print(by.to_string())
    worst = gal.assign(f=np.where(big, frac, 0.0)).nlargest(5, "f")[
        ["CATID", "cluster", "R_on_rtwo", "R_on_R200_recomputed", "sep_deg"]]
    print("worst 5:\n", worst.to_string(index=False))
    assert ok.mean() > 0.99, "host-cluster assignment does not reproduce R/R200"

    xt = pd.crosstab(gal.prefix, gal.cluster)
    assert (xt.astype(bool).sum(axis=1) == 1).all(), "a CATID prefix spans clusters"
    assert (xt.astype(bool).sum(axis=0) == 1).all(), "a cluster spans CATID prefixes"

    cols = ["CATID", "cluster", "RA_OBJ", "DEC_OBJ", "z_spec", "r_petro", "M_r",
            "r_e", "ellip", "PA", "mu_within_1re", "g_i", "Mstar",
            "R_on_rtwo", "V_on_sigma", "is_mem", "SURV_SAMI", "BAD_CLASS",
            "sep_deg", "R_proj_Mpc_from_cat", "R_proj_Mpc_direct",
            "R_on_R200_recomputed", "v_pec_kms",
            "host_RAdeg", "host_DEdeg", "host_z_clus", "host_sigma_200",
            "host_e_sigma_200", "host_N_mem_R200", "host_N_mem_2R200",
            "host_R_200", "host_M_200_caustic", "host_e_M_200_caustic",
            "host_M_200_virial", "host_e_M_200_virial"]
    out = gal[cols]
    p = os.path.join(OUT, "sami_cluster_env_hosts.tsv")
    out.to_csv(p, sep="\t", index=False, float_format="%.6g")
    n = sum(1 for _ in open(p, encoding="utf-8")) - 1
    assert n == 1433, "wrote %d rows" % n

    OBS = "OBSERVABLE"
    MOD = "MODEL-DERIVED (assumes virial equilibrium / spherical symmetry -> dark-matter dependent; RANK ONLY)"
    MIX = "OBSERVABLE quantity measured inside a MODEL-DERIVED aperture"
    units = [
        ("CATID", "-", OBS, "SAMI galaxy ID (Owers+2017 cluster-region ID)"),
        ("cluster", "-", OBS, "Host cluster, assigned here as the nearest of the eight Owers+2017 Table 1 centres and validated against R_on_rtwo"),
        ("RA_OBJ", "deg", OBS, "J2000 RA of the galaxy"),
        ("DEC_OBJ", "deg", OBS, "J2000 Dec of the galaxy"),
        ("z_spec", "-", OBS, "Spectroscopic redshift (SAMI Cluster Redshift Survey)"),
        ("r_petro", "mag", OBS, "Extinction-corrected r-band Petrosian magnitude"),
        ("M_r", "mag", OBS, "Absolute r-band magnitude (distance from z_spec and the assumed cosmology)"),
        ("r_e", "arcsec", OBS, "r-band major-axis effective radius from the Sersic fits of Owers et al. 2019 (ApJ 873, 52)"),
        ("ellip", "-", OBS, "r-band ellipticity from the same Sersic fits"),
        ("PA", "deg", OBS, "r-band position angle from the same Sersic fits"),
        ("mu_within_1re", "mag/arcsec^2", OBS, "Mean r-band surface brightness within 1 Re"),
        ("g_i", "mag", OBS, "(g-i) colour"),
        ("Mstar", "dex(Msun)", "OBSERVABLE-DERIVED (stellar population model, no dark matter)", "log10 stellar mass from the (g-i)/M_i proxy of Taylor+2011 as used by Bryant+2015"),
        ("R_on_rtwo", "-", MOD, "Published projected clustercentric radius / R200. The numerator is observable, the R200 normalisation is not"),
        ("V_on_sigma", "-", MIX, "Published v_pec / sigma_200"),
        ("is_mem", "-", OBS, "1 = caustic+shifting-gapper confirmed member in Owers+2017, 0 = not"),
        ("SURV_SAMI", "-", "-", "SAMI priority class (8 = primary, lower = secondary/filler)"),
        ("BAD_CLASS", "-", "-", "Owers+2017 problem-object flag; 0 = clean"),
        ("sep_deg", "deg", OBS, "Angular separation from the adopted cluster centre, computed here"),
        ("R_proj_Mpc_from_cat", "Mpc", OBS, "R_on_rtwo x R200: the projected physical clustercentric radius, i.e. the published column with the model normalisation undone. USE THIS as the observable radius"),
        ("R_proj_Mpc_direct", "Mpc", OBS, "Independent recomputation: sep_deg x D_A(z_clus), H0=70 Om=0.3"),
        ("R_on_R200_recomputed", "-", MOD, "R_proj_Mpc_direct / R200; validation column only"),
        ("v_pec_kms", "km/s", OBS, "V_on_sigma x sigma_200: line-of-sight peculiar velocity relative to the cluster"),
        ("host_RAdeg", "deg", OBS, "Adopted cluster centre RA"),
        ("host_DEdeg", "deg", OBS, "Adopted cluster centre Dec"),
        ("host_z_clus", "-", OBS, "Cluster redshift, biweight location of members within 2 R200"),
        ("host_sigma_200", "km/s", MIX, "Cluster line-of-sight velocity dispersion from member redshifts, biweight scale (Beers+1990), measured inside R200. THE PRIMARY OBSERVABLE ENVIRONMENT VARIABLE"),
        ("host_e_sigma_200", "km/s", OBS, "1-sigma uncertainty on sigma_200"),
        ("host_N_mem_R200", "-", OBS, "Number of confirmed members inside R200 that sigma_200 is based on"),
        ("host_N_mem_2R200", "-", OBS, "Number of confirmed members inside 2 R200"),
        ("host_R_200", "Mpc", MOD, "R200 = 0.17 sigma_200 / H(z) Mpc, singular isothermal sphere (Carlberg+1997)"),
        ("host_M_200_caustic", "1e14 Msun", MOD, "Caustic mass, Diaferio 1999 with F_beta = 0.7 (Serra+2011)"),
        ("host_e_M_200_caustic", "1e14 Msun", MOD, "Statistical uncertainty only; the F_beta systematic is not included"),
        ("host_M_200_virial", "1e14 Msun", MOD, "Corrected virial mass, Girardi+1998"),
        ("host_e_M_200_virial", "1e14 Msun", MOD, "1-sigma uncertainty on the virial mass"),
    ]
    man = {
        "file": "sami_cluster_env_hosts.tsv",
        "kind": "DERIVED - a join, not a download. See the two parent files for raw provenance.",
        "parents": [
            {"file": "sami_dr3_InputCatClustersDR3.tsv",
             "sha256": sha256_file(os.path.join(OUT, "sami_dr3_InputCatClustersDR3.tsv")),
             "source": "Data Central TAP, sami_dr3.InputCatClustersDR3"},
            {"file": "owers2017_table1_clusters.tsv",
             "sha256": sha256_file(os.path.join(OUT, "owers2017_table1_clusters.tsv")),
             "source": "arXiv 1703.00997, clusters.tex, Table 1"},
        ],
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256_file(p),
        "bytes": os.path.getsize(p),
        "row_count": n,
        "column_count": len(cols),
        "columns": [{"name": a, "unit": b, "provenance": c, "description": d}
                    for a, b, c, d in units],
        "method": "Host cluster = nearest of the eight Owers+2017 Table 1 centres on "
                  "the sky. Validated by recomputing R/R200 = sep x D_A(z_clus) / R200 "
                  "(FlatLambdaCDM H0=70 Om=0.3, the cosmology stated in Owers+2017 "
                  "Section 1) and comparing with the published R_on_rtwo column.",
        "validation": {
            "frac_within_2pct": float(ok.mean()),
            "median_fractional_difference": float(np.median(frac)),
            "p68_abs_fractional_difference": float(np.percentile(np.abs(frac), 68)),
            "max_abs_fractional_difference": float(np.abs(frac).max()),
            "catid_prefix_is_a_perfect_cluster_label": True,
        },
        "PROVENANCE_WARNING": (
            "host_R_200, host_M_200_caustic and host_M_200_virial are dynamical "
            "estimates that assume virial equilibrium and spherical symmetry. Under "
            "the well-network brief they may be used ONLY to rank environments, never "
            "as observations. host_sigma_200 (with host_e_sigma_200 and "
            "host_N_mem_R200) is measured from member redshifts and IS an observable. "
            "R_proj_Mpc_from_cat is the observable projected radius; R_on_rtwo is not."
        ),
    }
    with open(p + ".manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    print("\nwrote", p, n, "rows,", len(cols), "cols")


if __name__ == "__main__":
    main()
