"""Final counts for SAMI_INVENTORY.md and the lane report. Writes
sami_inventory_counts.json. Reads only files already in this directory.
"""
import json
import os

import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
m = pd.read_csv(os.path.join(OUT, "sami_dr3_master_galaxy_inventory.tsv"),
                sep="\t", dtype={"CATID": str})
assert len(m) == 3068

o = {}
o["galaxies_with_cubes_total"] = int(len(m))
o["by_arm"] = {k: int(v) for k, v in m.arm.value_counts().items()}

kin = {
    "resolved_2moment_stellar_and_emission_maps": m.has_resolved_kin_maps,
    "clean_kinematic_quality_flags": m.kin_quality_clean,
    "aperture_sigma_within_Sersic_Re": m.has_stelkin_sigma_re,
    "aperture_sigma_within_MGE_Re": m.SIGMA_RE_MGE.notna(),
    "lambda_R_Re_spin_proxy": m.has_lambda_re,
    "V_over_sigma_Re": m.VSIGMA_RE.notna(),
    "stellar_kinematic_PA": m.has_stelkin_pa,
    "gas_kinematic_PA": m.has_gaskin_pa,
    "ionised_gas_sigma_within_Re": m.has_gas_vdisp_re,
    "ionised_gas_velocity_within_Re": m.V_GAS_RE.notna(),
}
struct = {
    "Sersic_Re_ellip_PA_plus_Mstar": m.has_struct_full,
    "MGE_Re_and_ellipticity": m.has_mge,
    "visual_morphology": m.has_morph,
    "stellar_mass": m.Mstar.notna(),
    "Sersic_index_n": m.arm.eq("GAMA"),   # GAMA arm only; see gama_dr4 supplement
}
env = {
    "cluster_R_over_R200_and_host_sigma200": m.has_env_cluster,
    "confirmed_cluster_member": m.has_env_cluster & (m.is_mem == 1),
    "member_inside_R200": m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 1),
    "member_inside_0p5_R200": m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 0.5),
    "fifth_nearest_neighbour_density": m.SurfaceDensity.notna(),
}


def br(mask):
    s = m[mask].arm.value_counts()
    return {"total": int(mask.sum()), "cluster": int(s.get("cluster", 0)),
            "GAMA": int(s.get("GAMA", 0)), "filler": int(s.get("filler", 0))}


o["internal_kinematics"] = {k: br(v) for k, v in kin.items()}
o["structural_photometry"] = {k: br(v) for k, v in struct.items()}
o["environment"] = {k: br(v) for k, v in env.items()}

core = m.has_stelkin_sigma_re & m.has_struct_full & m.has_mge & m.has_morph
lt = m.morph_type >= 2
mem_in = m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 1)
o["matched_sample_cuts"] = {
    "kinematics_plus_structure_plus_morphology": br(core),
    "  ...late_type_TYPE_ge_2": br(core & lt),
    "  ...cluster_member_inside_R200": br(core & mem_in),
    "  ...late_type_member_inside_R200": br(core & lt & mem_in),
    "  ...late_type_member_inside_R200_clean_flags": br(core & lt & mem_in & m.kin_quality_clean),
    "  ...late_type_member_inside_2R200": br(core & lt & m.has_env_cluster
                                             & (m.is_mem == 1) & (m.R_on_rtwo <= 2)),
    "  ...TYPE_ge_2p5_member_inside_R200": br(core & (m.morph_type >= 2.5) & mem_in),
    "  ...late_type_GAMA_field_group_comparison_pool": br(core & lt & m.arm.eq("GAMA")),
}

sel = m[m.has_env_cluster & (m.is_mem == 1) & (m.R_on_rtwo <= 1)]
per = sel.groupby("cluster").agg(
    n_members_with_cubes=("CATID", "size"),
    n_sigma_Re=("has_stelkin_sigma_re", "sum"),
    n_lambda_Re=("has_lambda_re", "sum"),
    n_late_type=("morph_type", lambda s: int((s >= 2).sum())),
    host_sigma_200_kms=("host_sigma_200", "first"),
    host_e_sigma_200_kms=("host_e_sigma_200", "first"),
    host_N_members_R200=("host_N_mem_R200", "first"),
    host_R200_Mpc=("host_R_200", "first"),
    host_M200_virial_1e14=("host_M_200_virial", "first"),
).sort_values("host_sigma_200_kms")
o["per_cluster_members_inside_R200"] = json.loads(
    per.reset_index().to_json(orient="records"))

allc = m[m.has_env_cluster]
o["per_cluster_all_cluster_arm"] = json.loads(
    allc.groupby("cluster").agg(
        n_with_cubes=("CATID", "size"),
        n_members=("is_mem", "sum"),
        n_inside_R200=("R_on_rtwo", lambda s: int((s <= 1).sum())),
        median_R_proj_Mpc=("R_proj_Mpc_from_cat", "median"),
        max_R_proj_Mpc=("R_proj_Mpc_from_cat", "max"),
    ).reset_index().to_json(orient="records"))

ms = m[m.Mstar.notna()]
o["stellar_mass_range_log10Msun"] = {
    a: {"n": int((ms.arm == a).sum()),
        "p5": float(np.percentile(ms[ms.arm == a].Mstar, 5)),
        "median": float(np.median(ms[ms.arm == a].Mstar)),
        "p95": float(np.percentile(ms[ms.arm == a].Mstar, 95))}
    for a in ["cluster", "GAMA"]}
z = m[m.z_spec.notna()]
o["redshift_range"] = {
    a: {"min": float(z[z.arm == a].z_spec.min()),
        "median": float(z[z.arm == a].z_spec.median()),
        "max": float(z[z.arm == a].z_spec.max())}
    for a in ["cluster", "GAMA"]}

p = os.path.join(OUT, "sami_inventory_counts.json")
json.dump(o, open(p, "w"), indent=1)
print(json.dumps(o, indent=1))
print("\nwrote", p)
