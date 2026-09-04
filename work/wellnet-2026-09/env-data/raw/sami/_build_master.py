"""DERIVED product: one row per SAMI DR3 galaxy with a data cube, saying which
arm it belongs to (cluster / GAMA field-group / filler), what internal
kinematics it has, what structural photometry it has, and -- for the cluster
arm -- what host-environment quantities are attached and where they came from.

This is the table the matched-sample lane actually needs.  It is a join of
files already downloaded; nothing new is fetched.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
S = lambda n: os.path.join(OUT, n)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def rd(n, **kw):
    return pd.read_csv(S(n), sep="\t", dtype={"CATID": str}, **kw)


def main():
    co = rd("sami_dr3_CubeObs.tsv")
    assert len(co) == 3712
    best = co[co.ISBEST == 1].copy()
    assert len(best) == 3245 and best.CATID.is_unique

    ARM = {1: "GAMA", 2: "cluster", 3: "filler", 4: "fstar_GAMA", 5: "fstar_cluster"}
    best["arm"] = best.CATSOURCE.map(ARM)
    gal = best[best.WARNSTAR == 0].copy()
    assert len(gal) == 3068, "expected 3068 galaxies with cubes, got %d" % len(gal)
    print("galaxies with a best cube: %d  (%s)"
          % (len(gal), dict(gal.arm.value_counts())))

    sk = rd("sami_dr3_samiDR3Stelkin.tsv")
    sk = sk[sk.CUBEIDPUB.isin(best.CUBEIDPUB)]
    gk = rd("sami_dr3_samiDR3gaskinPA.tsv")
    gk = gk[gk.CUBEIDPUB.isin(best.CUBEIDPUB)]
    el = rd("sami_dr3_EmissionLine1compDR3.tsv",
            usecols=["CATID", "CUBEIDPUB", "VDISP_GAS_RE", "VDISP_GAS_RE_ERR",
                     "V_GAS_RE", "V_GAS_RE_ERR", "HALPHA_RE", "HALPHA_RE_ERR"])
    el = el[el.CUBEIDPUB.isin(best.CUBEIDPUB)]
    mo = rd("sami_dr3_VisualMorphologyDR3.tsv")
    de = rd("sami_dr3_DensityCatDR3.tsv")
    mg = rd("sami_dr3_MGEPhotomUnregDR3.tsv")
    # a galaxy can have both SDSS and VST MGE fits; prefer VST where both exist
    mg["rank"] = (mg.photometry.astype(str).str.upper().str.contains("VST")).astype(int)
    mg = mg.sort_values(["CATID", "rank"], ascending=[True, False]).drop_duplicates("CATID")

    icl = pd.read_csv(S("sami_cluster_env_hosts.tsv"), sep="\t", dtype={"CATID": str})
    assert len(icl) == 1433
    iga = rd("sami_dr3_InputCatGAMADR3.tsv")
    assert len(iga) == 5536
    ifi = rd("sami_dr3_InputCatFiller.tsv")

    m = gal[["CATID", "CUBEIDPUB", "arm", "CUBEFWHM", "CUBETEXP",
             "WARNMULT", "WARNSK2M", "WARNSKER", "WARNRE", "WARNWCS",
             "WARNEMFT", "WARNSKEM", "WARNFCAL", "WARNZ"]].copy()

    # ---- structural photometry, taken from the arm's own input catalogue ----
    ph_c = icl[["CATID", "RA_OBJ", "DEC_OBJ", "z_spec", "r_petro", "M_r", "r_e",
                "ellip", "PA", "mu_within_1re", "g_i", "Mstar"]]
    ph_g = iga[["CATID", "RA_OBJ", "DEC_OBJ", "z_spec", "r_petro", "M_r", "r_e",
                "ellip", "PA", "mu_within_1re", "g_i", "Mstar"]]
    ph = pd.concat([ph_c, ph_g], ignore_index=True).drop_duplicates("CATID")
    m = m.merge(ph, on="CATID", how="left")

    m = m.merge(mg[["CATID", "photometry", "ReMGE", "mMGE", "PAMGE",
                    "epsMGE_Re", "epsMGE_LW", "dist2NNeigh"]], on="CATID", how="left")
    m = m.merge(mo.rename(columns={"TYPE": "morph_type"}), on="CATID", how="left")
    m = m.merge(de[["CATID", "SurfaceDensity", "SurfaceDensity_err",
                    "SurfaceDensityFlag"]].drop_duplicates("CATID"),
                on="CATID", how="left")

    m = m.merge(sk[["CUBEIDPUB", "SIGMA_RE", "SIGMA_RE_ERR", "SIGMA_RE_MGE",
                    "SIGMA_RE_MGE_ERR", "LAMBDAR_RE", "LAMBDAR_RE_ERR",
                    "VSIGMA_RE", "VSIGMA_RE_ERR", "PA_STELKIN", "PA_STELKIN_ERR",
                    "APER_CORR_FLAG", "MEAN_K51_RE"]], on="CUBEIDPUB", how="left")
    m = m.merge(gk[["CUBEIDPUB", "PA_GASKIN", "PA_GASKIN_ERR"]],
                on="CUBEIDPUB", how="left")
    m = m.merge(el.drop(columns=["CATID"]), on="CUBEIDPUB", how="left")

    envc = ["cluster", "R_on_rtwo", "V_on_sigma", "is_mem", "R_proj_Mpc_from_cat",
            "v_pec_kms", "host_z_clus", "host_sigma_200", "host_e_sigma_200",
            "host_N_mem_R200", "host_R_200", "host_M_200_caustic", "host_M_200_virial"]
    m = m.merge(icl[["CATID"] + envc], on="CATID", how="left")

    # ---------------- availability flags -----------------------------------
    m["has_stelkin_sigma_re"] = m.SIGMA_RE.notna() & (m.SIGMA_RE > 0)
    m["has_lambda_re"] = m.LAMBDAR_RE.notna()
    m["has_stelkin_pa"] = m.PA_STELKIN.notna()
    m["has_gaskin_pa"] = m.PA_GASKIN.notna()
    m["has_gas_vdisp_re"] = m.VDISP_GAS_RE.notna()
    # Every DR3 cube has 2-moment stellar kinematic maps and emission-line maps
    # produced (WARNSK2M and WARNEMFT are 0 for all 3068), so "has resolved
    # kinematics" is universal; what varies is quality.
    m["has_resolved_kin_maps"] = (m.WARNSK2M == 0) & (m.WARNEMFT == 0)
    m["kin_quality_clean"] = (m.WARNSK2M == 0) & (m.WARNSKER == 0) & \
                             (m.WARNMULT == 0) & (m.WARNZ == 0) & (m.WARNWCS == 0)
    m["has_struct_full"] = (m.r_e.notna() & m.ellip.notna() & m.Mstar.notna())
    m["has_mge"] = m.ReMGE.notna()
    m["has_morph"] = m.morph_type.notna()
    m["has_env_cluster"] = m.R_on_rtwo.notna() & m.host_sigma_200.notna()

    # inclination from the r-band Sersic ellipticity (thin-disc approximation)
    q = 1.0 - m.ellip
    m["incl_deg_thin"] = np.degrees(np.arccos(np.clip(q, 0, 1)))

    p = S("sami_dr3_master_galaxy_inventory.tsv")
    m.to_csv(p, sep="\t", index=False, float_format="%.6g")
    n = sum(1 for _ in open(p, encoding="utf-8")) - 1
    assert n == 3068, "wrote %d rows" % n

    # ---------------- report ----------------------------------------------
    def tab(mask, label):
        s = m[mask].arm.value_counts()
        print("  %-34s total %5d   cluster %4d   GAMA %4d   filler %4d"
              % (label, mask.sum(), s.get("cluster", 0), s.get("GAMA", 0),
                 s.get("filler", 0)))

    print("\n=== SAMI DR3 galaxies with a best data cube: %d ===" % len(m))
    tab(m.has_stelkin_sigma_re, "sigma within Sersic Re")
    tab(m.has_lambda_re, "lambda_R(Re) spin proxy")
    tab(m.has_stelkin_pa, "stellar kinematic PA (resolved)")
    tab(m.has_gaskin_pa, "gas kinematic PA (resolved)")
    tab(m.has_stelkin_pa | m.has_gaskin_pa, "ANY resolved kinematic PA")
    tab(m.has_stelkin_pa & (m.WARNSK2M == 0) & (m.WARNSKER == 0),
        "stellar kin, no WARNSK2M/WARNSKER")
    tab(m.has_gas_vdisp_re, "ionised-gas sigma within Re")
    tab(m.has_struct_full, "r_e + ellip + Mstar (Sersic)")
    tab(m.has_mge, "MGE Re + ellipticity")
    tab(m.has_morph, "visual morphology")
    tab(m.has_env_cluster, "R/R200 + host sigma_200")
    tab(m.has_env_cluster & (m.is_mem == 1), "  ... and a confirmed member")
    tab(m.SurfaceDensity.notna(), "5th-nearest-neighbour density")

    core = (m.has_stelkin_sigma_re & m.has_struct_full & m.has_mge & m.has_morph)
    tab(core, "kinematics + structure + morph")
    tab(core & m.has_env_cluster, "  ... and cluster environment")
    tab(core & m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 1.0),
        "  ... member inside R200")

    print()
    tab(m.has_resolved_kin_maps, "2-moment stellar + em-line maps")
    tab(m.kin_quality_clean, "  ... clean quality flags")

    print()
    lt = m.morph_type >= 2  # early-spiral (2), early/late (2.5), late spiral (3)
    tab(core & lt, "late-type (TYPE>=2) with the core set")
    tab(core & lt & m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 1.0),
        "  ... member inside R200")
    tab(core & lt & m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 2.0),
        "  ... member inside 2 R200")
    tab(core & lt & m.kin_quality_clean & m.has_env_cluster & (m.is_mem == 1)
        & (m.R_on_rtwo <= 1.0), "  ... inside R200, clean flags")
    lt3 = m.morph_type >= 2.5
    tab(core & lt3 & m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 1.0),
        "TYPE>=2.5 member inside R200, core set")

    print("\ncluster arm, members inside R200, by host:")
    sel = m[(m.arm == "cluster") & (m.is_mem == 1) & (m.R_on_rtwo <= 1.0)]
    g = sel.groupby("cluster").agg(
        n=("CATID", "size"),
        n_sigma_re=("has_stelkin_sigma_re", "sum"),
        n_lambda=("has_lambda_re", "sum"),
        n_latetype=("morph_type", lambda s: int((s >= 2).sum())),
        sigma_200=("host_sigma_200", "first"),
        e_sigma_200=("host_e_sigma_200", "first"),
        N_mem=("host_N_mem_R200", "first"))
    print(g.sort_values("sigma_200").to_string())

    man = {
        "file": os.path.basename(p),
        "kind": "DERIVED - a join of the SAMI DR3 tables already downloaded here "
                "plus sami_cluster_env_hosts.tsv. No new network access.",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256_file(p),
        "bytes": os.path.getsize(p),
        "row_count": n,
        "column_count": m.shape[1],
        "columns": [{"name": c, "dtype": str(m[c].dtype)} for c in m.columns],
        "row_definition": "One row per unique SAMI DR3 galaxy that has a data cube "
                          "(CubeObs.ISBEST = 1 and WARNSTAR = 0). Calibration stars "
                          "are excluded. 3068 rows, matching the DR3 headline sample.",
        "parents": [{"file": f, "sha256": sha256_file(S(f))} for f in [
            "sami_dr3_CubeObs.tsv", "sami_dr3_samiDR3Stelkin.tsv",
            "sami_dr3_samiDR3gaskinPA.tsv", "sami_dr3_EmissionLine1compDR3.tsv",
            "sami_dr3_VisualMorphologyDR3.tsv", "sami_dr3_DensityCatDR3.tsv",
            "sami_dr3_MGEPhotomUnregDR3.tsv", "sami_dr3_InputCatGAMADR3.tsv",
            "sami_dr3_InputCatFiller.tsv", "sami_cluster_env_hosts.tsv"]],
        "notes": [
            "arm: 'cluster' = CATSOURCE 2 (InputCatClustersDR3, the eight Owers+2017 "
            "clusters); 'GAMA' = CATSOURCE 1 (field and group, GAMA equatorial "
            "regions); 'filler' = CATSOURCE 3.",
            "Where a galaxy has both SDSS and VST MGE fits, the VST row is kept.",
            "incl_deg_thin is arccos(1 - ellip): a THIN-DISC approximation, valid "
            "only for late types, with no intrinsic-thickness correction. It is a "
            "convenience column, not a measurement.",
            "host_M_200_caustic and host_M_200_virial are MODEL-DERIVED dynamical "
            "masses assuming virial equilibrium and spherical symmetry: RANK ONLY. "
            "host_sigma_200 with host_e_sigma_200 and host_N_mem_R200 is the "
            "observable environment variable.",
            "R_proj_Mpc_from_cat is the observable projected clustercentric radius; "
            "R_on_rtwo divides it by the model-derived R200.",
        ],
    }
    with open(p + ".manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    print("\nwrote", p, n, "rows,", m.shape[1], "cols")


if __name__ == "__main__":
    main()
