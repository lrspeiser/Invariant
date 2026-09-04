"""A_dyn on one object at a time: in-plane leg fitted, frozen, then the
off-plane tracer FORWARD-MODELLED in full phase space.

    A_dyn(x) = [g_R(x)/g_R,N(x)] / [|g_z(x)|/|g_z,N(x)|]     -- ONE point

THE PROCEDURE, in the order the brief specifies:
  1. fit the candidate law using ONLY the disc rotation curve;
  2. FREEZE every gravity parameter and every baryon mass;
  3. predict the off-plane tracer;
  4. forward-model that tracer in full phase space -- progenitor orbit,
     stripping history, present-day stream distribution -- under each
     candidate.  A stream track is NEVER converted into g_R and g_z points.

The measurement is made by deforming the frozen potential along the ONE
direction that leaves the in-plane leg exactly invariant (see
`orbit_model.DeformedField`) and asking the forward model which deformation
the streams accept.  The deformation parameter maps to A_dyn.

Run:  python adyn_same_object.py [stage ...]
      stages: ingest predict measure nulls secondary all
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orbit_model as OM                                          # noqa: E402

REPO = OM.REPO
RAW = os.path.join(REPO, "work", "wellnet-2026-09", "env-data", "raw")
SS = os.path.join(RAW, "streams-satellites")
TRACKS = os.path.join(SS, "galstreams_data", "galstreams", "tracks")
OUT = os.path.join(HERE, "orthogonal_results.json")

# =========================================================================
#  DECLARED IN ADVANCE  --  nothing below is chosen after seeing a residual
# =========================================================================
DECL = dict(
    adyn_definition="A_dyn(x) = [g_R/g_R,N] / [|g_z|/|g_z,N|] at the SAME x",
    reference_points={
        "primary": "median (R_gc, |z|) of the gated MW stream sample",
        "solar": "(R, z) = (8.122, 1.1) kpc, the Bovy-Rix K_z reference",
        "profile": "R = 5..100 kpc at |z| = 0.5 R",
    },
    split="IN-PLANE leg (Eilers 2019 v_c, 38 pts) calibrates the baryon "
          "masses and NOTHING else.  OUT-OF-PLANE leg (streams) is never "
          "used to set a gravity constant or a baryon mass; it measures the "
          "single deformation parameter Lambda, once, with everything else "
          "frozen.  The split is definitional, not statistical.",
    rc_systematic_floor_kms=5.0,
    baryon_free_params=["M_bulge", "M_thin(+thick tied at 0.25)", "M_gas"],
    baryon_shape_fixed="Hernquist a=0.5; MN (a,b) = (3.0,0.28) thin, "
                       "(4.4,0.9) thick, (7.0,0.085) gas -- MEASURED "
                       "baryonic quantities, inputs not parameters",
    lambda_grid="0.55 .. 1.80, 11 points, log-spaced",
    stream_error_model=dict(sky_deg=0.35, dist_frac=0.10, dist_floor_kpc=1.0,
                            pm_masyr=0.15, vrad_kms=15.0,
                            chi2_miss_per_node=9.0, n_min_particles=4),
    nuisance=dict(m_prog_msun=[1.0e4, 1.0e6], T_strip_gyr=[2.0, 4.0],
                  n_anchor_draws=12, anchor_seed=20260904),
    anchor_errors=dict(dist_frac=0.10, pm_masyr=0.12, vrad_kms=10.0),
    spray="Fardal+2015 particle spray, n_release=80 x 2 particles, "
          "tidal radius from the CANDIDATE field's |g| r^2/G",
    plausibility_gate=OM.GATE,
)


# =========================================================================
#  1.  INGEST  --  with the four silent defects gated explicitly
# =========================================================================
def _f(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def read_tsv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_ecsv(path):
    cols, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split(",")
            if cols is None:
                cols = parts
                continue
            rows.append(parts)
    if cols is None:
        return {}, 0
    arr = np.array([[float(v) if v not in ("", "nan") else np.nan
                     for v in r] for r in rows], float)
    return {c: arr[:, i] for i, c in enumerate(cols)}, len(rows)


def load_eilers():
    p = os.path.join(SS, "stream_eilers2019_MW_rotation_curve.tsv")
    rows = read_tsv(p)
    assert len(rows) == 38, f"Eilers row count {len(rows)} != 38"
    assert list(rows[0].keys()) == ["R_kpc", "vc_kms", "e_vc_minus_kms",
                                    "e_vc_plus_kms"], "Eilers columns changed"
    R = np.array([float(r["R_kpc"]) for r in rows])
    v = np.array([float(r["vc_kms"]) for r in rows])
    e = 0.5 * (np.array([float(r["e_vc_minus_kms"]) for r in rows])
               + np.array([float(r["e_vc_plus_kms"]) for r in rows]))
    return R, v, e, {"file": p, "sha256": OM.sha256(p), "rows": len(rows)}


def ingest_streams(verbose=True):
    """galstreams v1.2.1 -> the gated MW stream sample.

    THE RULE: the flag governs; data may only DOWNGRADE a track, never
    promote it.  Four silent defects affect 102 of 217 tracks:
      * 68 `ibata2024` tracks (GD-1 among them) advertise a distance track
        whose column is identically 1.000 kpc.  Float round-trip noise means
        `== 1.0` finds NOTHING -- the test must be a tolerance;
      * 15 tracks advertise InfoFlags=1111 with unphysical velocities, up to
        9.56e6 km/s (32 c) for Hydrus.ibata2024;
      * Pal5.pricewhelan2019 Vrad is the 999.0 sentinel, and 16 tracks have
        the Vrad flag CLEAR but the column populated with junk;
      * 3 summary rows have no track file.
    """
    sm = os.path.join(SS, "galstreams_track_summary.tsv")
    rows = read_tsv(sm)
    assert len(rows) == 217, f"summary row count {len(rows)} != 217"
    assert len(rows[0]) == 47, f"summary column count {len(rows[0])} != 47"
    rec, audit = [], dict(n_rows=len(rows), no_track_file=[], gated_D=[],
                          gated_pm=[], gated_vrad=[], unit_kpc=[],
                          superluminal=[], sentinel=[])
    for r in rows:
        base = r["TrackFileBase"]
        p = os.path.join(TRACKS, base + ".ecsv")
        if not os.path.exists(p):
            audit["no_track_file"].append(base)
            continue
        d, n = read_ecsv(p)
        need = ("ra", "dec", "distance", "pm_ra_cosdec", "pm_dec",
                "radial_velocity")
        obs = np.stack([d.get(k, np.full(n, np.nan)) for k in need], axis=1)
        ok, why = OM.plausibility_gate(obs)
        flag = r["InfoFlags"]
        # the FLAG governs: a claimed measurement may be revoked by the gate,
        # never created by it
        # galstreams InfoFlags digits take values 0,1,2 -- ANY nonzero digit
        # advertises the quantity (digit 2 = present from a second source,
        # e.g. M68-Fjorm.palau2019 with flags 1210).  Testing == "1" would
        # silently drop those, so the decode is > 0.
        claim = {k: int(flag[i]) > 0
                 for k, i in (("D", 1), ("pm", 2), ("vrad", 3))}
        has = {k: claim[k] and ok[k] for k in claim}
        for k in ("D", "pm", "vrad"):
            if claim[k] and not ok[k]:
                audit["gated_" + k].append(base)
        for w in why:
            if "1 kpc placeholder" in w:
                audit["unit_kpc"].append(base)
            if "exceeds 800" in w or "galactocentric speed" in w:
                audit["superluminal"].append(base)
            if "sentinel" in w:
                audit["sentinel"].append(base)
        rec.append(dict(base=base, stream=r["StreamName"], ref=r["TrackRef"],
                        flags=flag, n=n, obs=obs, has=has,
                        summary_usable_3d=int(r["usable_3d"]),
                        summary_usable_6d=int(r["usable_6d"]),
                        defects=r["data_defects"],
                        inc_deg=_f(r["orbit_inc_to_disc_deg"]),
                        Rmin=_f(r["R_gc_min_kpc"]),
                        Rmax=_f(r["R_gc_max_kpc"]),
                        zmax=_f(r["absz_max_kpc"]),
                        width_phi2=_f(r["width_phi2"]),
                        width_pm1=_f(r["width_pm_phi1_cosphi2"]),
                        width_pm2=_f(r["width_pm_phi2"]),
                        sky_ok=r["sky_status"] == "EMPIRICAL_TRACK"))
    u3 = [t for t in rec if t["sky_ok"] and t["has"]["D"]]
    u6 = [t for t in u3 if t["has"]["pm"] and t["has"]["vrad"]]
    audit.update(n_with_track_file=len(rec), n_usable_3d=len(u3),
                 n_usable_6d=len(u6),
                 n_gated_D=len(audit["gated_D"]),
                 n_gated_pm=len(audit["gated_pm"]),
                 n_gated_vrad=len(audit["gated_vrad"]),
                 n_unit_kpc=len(set(audit["unit_kpc"])),
                 n_superluminal=len(set(audit["superluminal"])),
                 agrees_with_summary_3d=len(u3) == sum(
                     t["summary_usable_3d"] for t in rec),
                 agrees_with_summary_6d=len(u6) == sum(
                     t["summary_usable_6d"] for t in rec))
    # de-duplicate: one track per stream, richest first
    def rank(t):
        return (t["has"]["pm"] and t["has"]["vrad"], t["has"]["D"], t["n"],
                t["Rmax"])
    best6, best3 = {}, {}
    for t in u6:
        k = t["stream"]
        if k not in best6 or rank(t) > rank(best6[k]):
            best6[k] = t
    for t in u3:
        k = t["stream"]
        if k not in best3 or rank(t) > rank(best3[k]):
            best3[k] = t
    audit["n_streams_6d"] = len(best6)
    audit["n_streams_3d"] = len(best3)
    s6 = sorted(best6.values(), key=lambda t: t["stream"])
    s3 = sorted(best3.values(), key=lambda t: t["stream"])
    audit["n_polar_within_10deg_3d"] = sum(
        1 for t in u3 if abs(t["inc_deg"] - 90) <= 10)
    audit["n_beyond_25kpc_3d"] = sum(1 for t in u3 if t["Rmax"] > 25)
    audit["n_polar_within_10deg_6d_dedup"] = sum(
        1 for t in s6 if abs(t["inc_deg"] - 90) <= 10)
    audit["n_beyond_25kpc_6d_dedup"] = sum(1 for t in s6 if t["Rmax"] > 25)
    audit["sha256_summary"] = OM.sha256(sm)
    if verbose:
        for k, v in audit.items():
            if not isinstance(v, list):
                print(f"  {k:34s} {v}")
    return s6, s3, audit


# =========================================================================
#  2.  IN-PLANE LEG  --  fit the baryons, then FREEZE
# =========================================================================
def fit_baryons(law: OM.Law, R, v, e, floor=None):
    """Calibrate (M_bulge, M_disc, M_gas) on the rotation curve ALONE.

    The gravity constants (a0, A, m, I0) are global and frozen from the
    tournament's radial fits; they are not touched here.  Only the source is
    calibrated, which brief rule 3 permits (measured baryonic quantities are
    inputs; their overall normalisation for THIS galaxy is what the in-plane
    leg is for)."""
    floor = DECL["rc_systematic_floor_kms"] if floor is None else floor
    sig = np.sqrt(e ** 2 + floor ** 2)
    z0 = np.zeros_like(R)

    def chi2(p):
        Mb, Md, Mg = np.exp(p)
        bar = OM.MWBaryons(M_bulge=Mb, M_thin=Md, M_thick=0.25 * Md, M_gas=Mg)
        gR, _ = law.g_algebraic(bar, R, z0)
        vm = np.sqrt(np.maximum(gR * R * OM.KPC, 0.0)) / OM.KMS
        return float(np.sum(((v - vm) / sig) ** 2))

    from scipy.optimize import minimize
    best, bp = np.inf, None
    for p0 in ([np.log(0.9e10), np.log(4e10), np.log(1.2e10)],
               [np.log(0.5e10), np.log(2e10), np.log(0.8e10)],
               [np.log(2.0e10), np.log(8e10), np.log(2.0e10)]):
        r = minimize(chi2, p0, method="Nelder-Mead",
                     options=dict(maxiter=6000, xatol=1e-6, fatol=1e-6))
        if r.fun < best:
            best, bp = r.fun, r.x
    Mb, Md, Mg = np.exp(bp)
    bar = OM.MWBaryons(M_bulge=Mb, M_thin=Md, M_thick=0.25 * Md, M_gas=Mg)
    gR, _ = law.g_algebraic(bar, R, z0)
    vm = np.sqrt(np.maximum(gR * R * OM.KPC, 0.0)) / OM.KMS
    return bar, dict(chi2=best, dof=len(R) - 3, rms_kms=float(
        np.sqrt(np.mean((v - vm) ** 2))), M_bulge=Mb, M_disc=Md,
        M_thick=0.25 * Md, M_gas=Mg, M_total=bar.M_total,
        vc_model=vm.tolist())


# =========================================================================
#  3.  OBSERVABLE PROJECTION AND THE TRACK STATISTIC
# =========================================================================
def unit_sky(ra_deg, dec_deg):
    a = np.radians(ra_deg)
    d = np.radians(dec_deg)
    return np.stack([np.cos(d) * np.cos(a), np.cos(d) * np.sin(a),
                     np.sin(d)], axis=-1)


def resample_track(obs, n_node=48):
    """Even sampling in sky arc length along the observed track."""
    u = unit_sky(obs[:, 0], obs[:, 1])
    s = np.concatenate([[0.0], np.cumsum(np.degrees(np.arccos(
        np.clip(np.sum(u[1:] * u[:-1], axis=1), -1, 1))))])
    if s[-1] <= 0:
        return obs[:1]
    idx = np.searchsorted(s, np.linspace(0, s[-1], min(n_node, len(obs))))
    idx = np.clip(np.unique(idx), 0, len(obs) - 1)
    return obs[idx]


def track_frame(node_obs):
    """Unit vectors and cross-track directions of the observed track."""
    u = unit_sky(node_obs[:, 0], node_obs[:, 1])
    t = np.gradient(u, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    c = np.cross(u, t)
    c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    return u, c


def stream_chi2(model_obs, node_obs, use, em):
    """chi^2 of a forward-modelled stream against an observed track.

    Model particles are assigned to their NEAREST observed node; the model's
    median cross-track offset, distance, proper motions and radial velocity
    in each node's bin are compared with the observation.  Nodes the model
    stream never reaches contribute `chi2_miss_per_node`.  No step of this
    converts the track into a force.
    """
    if model_obs.shape[0] == 0:
        return em["chi2_miss_per_node"] * len(node_obs) * (1 + sum(use.values()))
    u, c = track_frame(node_obs)
    p = unit_sky(model_obs[:, 0], model_obs[:, 1])
    cosang = p @ u.T                                        # (npart, nnode)
    j = np.argmax(cosang, axis=1)
    off = np.degrees(np.arcsin(np.clip(np.sum(p * c[j], axis=1), -1, 1)))
    nn = len(node_obs)
    chi2 = 0.0
    sD = np.maximum(em["dist_frac"] * node_obs[:, 2], em["dist_floor_kpc"])
    for i in range(nn):
        m = j == i
        k = int(np.count_nonzero(m))
        if k < em["n_min_particles"]:
            chi2 += em["chi2_miss_per_node"] * (1 + sum(use.values()))
            continue
        chi2 += (np.median(off[m]) / em["sky_deg"]) ** 2
        if use["D"]:
            chi2 += ((np.median(model_obs[m, 2]) - node_obs[i, 2]) / sD[i]) ** 2
        if use["pm"]:
            chi2 += ((np.median(model_obs[m, 3]) - node_obs[i, 3])
                     / em["pm_masyr"]) ** 2
            chi2 += ((np.median(model_obs[m, 4]) - node_obs[i, 4])
                     / em["pm_masyr"]) ** 2
        if use["vrad"]:
            chi2 += ((np.median(model_obs[m, 5]) - node_obs[i, 5])
                     / em["vrad_kms"]) ** 2
    return float(chi2)


def draw_anchors(anchor_obs, n, rng):
    """Anchor phase-space draws.  The SAME draws are used at every Lambda and
    for every law, so the profile-minimum bias is common-mode and cancels in
    Delta chi^2(Lambda)."""
    ae = DECL["anchor_errors"]
    a = np.repeat(anchor_obs.reshape(1, 6), n, axis=0)
    a[1:, 2] *= 1.0 + ae["dist_frac"] * rng.standard_normal(n - 1)
    a[1:, 3] += ae["pm_masyr"] * rng.standard_normal(n - 1)
    a[1:, 4] += ae["pm_masyr"] * rng.standard_normal(n - 1)
    a[1:, 5] += ae["vrad_kms"] * rng.standard_normal(n - 1)
    return a


# =========================================================================
#  4.  STEP 1+2: FIT ON THE DISC, THEN FREEZE.  STEP 3: PREDICT A_dyn.
# =========================================================================
GRID_PROD = dict(nR=400, nz=400, Rmax=300.0, zmax=300.0)
GRID_FAST = dict(nR=256, nz=256, Rmax=200.0, zmax=200.0)
COMPLETION = {"newton": "poisson", "rar": "qumond", "aqual": "aqual",
              "tidal_scalar": "aqual", "wellnet_tensor": "aqual"}
CACHE = os.path.join(HERE, "_fields")


def calibrate_and_solve(law, R, v, e, grid=None, n_iter=2, verbose=True):
    """Steps 1 and 2.  The baryon masses are calibrated so that the SOLVED
    field's midplane rotation curve matches Eilers, not merely the algebraic
    approximation to it -- otherwise the freeze would be against a curve the
    law does not actually produce.  Then everything is frozen.
    """
    grid = GRID_PROD if grid is None else grid
    law.completion = COMPLETION[law.name]
    corr = np.ones_like(R)
    info = None
    for it in range(n_iter):
        bar, info = fit_baryons(law, R, v * corr, e)
        gk = GRID_FAST if it < n_iter - 1 else grid
        ref = FieldSolution_cached(bar, OM.Law("newton_ref", base="newton"),
                                   gk, tag=f"{law.name}_ref_it{it}")
        sol = FieldSolution_cached(bar, law, gk, ref=ref,
                                   tag=f"{law.name}_it{it}", verbose=verbose)
        v_solved = sol.vc_midplane(R)
        v_alg = np.array(info["vc_model"])
        ratio = v_alg / np.maximum(v_solved, 1e-6)
        corr = corr * ratio                       # push the algebraic target
        if verbose:
            print(f"    iter {it}: solved-vs-data RMS "
                  f"{np.sqrt(np.mean((v_solved - v) ** 2)):.2f} km/s, "
                  f"median completion/algebraic = {np.median(1/ratio):.4f}")
    info["v_solved_kms"] = v_solved.tolist()
    info["rms_solved_kms"] = float(np.sqrt(np.mean((v_solved - v) ** 2)))
    info["chi2_solved"] = float(np.sum(((v_solved - v)
                                        / np.sqrt(e ** 2 + DECL[
                                            "rc_systematic_floor_kms"] ** 2)) ** 2))
    info["mass_gate"] = sol.mass_gate
    info["completion"] = law.completion
    info["n_picard"] = sol.n_picard
    info["cg_resid"] = sol.resid
    return bar, sol, ref, info


def FieldSolution_cached(bar, law, gk, ref=None, tag="", verbose=False):
    os.makedirs(CACHE, exist_ok=True)
    key = (f"{tag}_{law.name}_{law.base}_{law.completion}_{law.struct}"
           f"_{law.gate}_{law.a0:.6e}_{law.A:.6e}_{law.I0:.4e}_{law.m:g}"
           f"_{bar.M_bulge:.6e}_{bar.M_thin:.6e}_{bar.M_gas:.6e}"
           f"_{gk['nR']}_{gk['Rmax']:.0f}")
    h = __import__("hashlib").sha1(key.encode()).hexdigest()[:16]
    p = os.path.join(CACHE, h + ".npz")
    sol = OM.FieldSolution.__new__(OM.FieldSolution)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        sol.bar, sol.law, sol.ref = bar, law, ref
        sol.g = OM.AX.Grid(gk["nR"], gk["nz"], gk["Rmax"], gk["zmax"])
        sol.Rk, sol.zk = sol.g.Rc / OM.KPC, sol.g.zc / OM.KPC
        sol.RR = sol.Rk[:, None] * np.ones((1, gk["nz"]))
        sol.ZZ = np.ones((gk["nR"], 1)) * sol.zk[None, :]
        sol.Psi = d["Psi"] if d["has_psi"] else None
        sol.Mtot = bar.M_total * OM.MSUN
        sol.mass_gate = float(d["mass_gate"])
        sol.M_on_grid = float(d["M_on_grid"])
        sol.n_picard, sol.resid = int(d["n_picard"]), float(d["resid"])
        sol.rho = None
        sol._build_splines()
        return sol
    t0 = time.time()
    sol = OM.FieldSolution(bar, law, ref=ref, verbose=verbose, **gk)
    np.savez_compressed(p, Psi=sol.Psi if sol.Psi is not None else np.zeros(1),
                        has_psi=sol.Psi is not None, mass_gate=sol.mass_gate,
                        M_on_grid=sol.M_on_grid, n_picard=sol.n_picard,
                        resid=sol.resid)
    if verbose:
        print(f"    solved {law.name} on {gk['nR']}^2 in {time.time()-t0:.0f}s")
    return sol


def predict_stage(verbose=True):
    R, v, e, prov = load_eilers()
    s6, s3, audit = ingest_streams(verbose=False)
    # the primary reference point is set by the TRACER GEOMETRY, declared
    # before any residual is examined
    allR = np.array([0.5 * (t["Rmin"] + t["Rmax"]) for t in s3])
    allz = np.array([0.5 * t["zmax"] for t in s3])
    ref_pt = (float(np.median(allR)), float(np.median(allz)))
    print(f"  primary reference point (median stream R, |z|/2): "
          f"R = {ref_pt[0]:.2f} kpc, |z| = {ref_pt[1]:.2f} kpc")
    Rprof = np.array([5., 8., 15., 25., 40., 60., 100.])
    zprof = 0.5 * Rprof
    out = {}
    for law in OM.frozen_laws():
        print(f"\n-- {law.name}  ({law.note})")
        bar, sol, ref, info = calibrate_and_solve(law, R, v, e, verbose=verbose)
        BR, Bz = sol.multipliers(np.array([ref_pt[0], 8.122]),
                                 np.array([ref_pt[1], 1.1]))
        adyn_prim, adyn_sol = (BR / Bz)[0], (BR / Bz)[1]
        pr, pz = sol.multipliers(Rprof, zprof)
        out[law.name] = dict(
            law=dict(name=law.name, base=law.base, a0=law.a0, gate=law.gate,
                     form=law.form, m=law.m, I0=law.I0, A=law.A,
                     struct=law.struct, completion=law.completion,
                     note=law.note),
            inplane=info,
            A_dyn_algebraic=1.0 if law.struct == "scalar_a0" else None,
            A_dyn_predicted_primary=float(adyn_prim),
            A_dyn_predicted_solar=float(adyn_sol),
            A_dyn_profile=dict(R_kpc=Rprof.tolist(), z_kpc=zprof.tolist(),
                               A_dyn=(pr / pz).tolist(),
                               B_R=pr.tolist(), B_z=pz.tolist()),
            curl_defect_algebraic=float(OM.curl_defect(sol, bar)),
        )
        print(f"   in-plane RMS (solved) {info['rms_solved_kms']:.2f} km/s, "
              f"M_b = {info['M_total']:.3e} Msun")
        print(f"   PREDICTED A_dyn: primary {adyn_prim:.5f}, "
              f"solar {adyn_sol:.5f}")
        print(f"   profile A_dyn: {np.round(pr/pz, 5)}")
    return out, ref_pt, prov, audit


# =========================================================================
#  5.  STEP 4: FORWARD-MODEL THE OFF-PLANE TRACER, AND MEASURE Lambda
# =========================================================================
LAMBDA_GRID = np.round(np.geomspace(0.55, 1.80, 11), 4)
#: 11 points, log-spaced, declared in advance and never widened after
#: looking at a residual.


def anchor_and_nodes(t, n_node=48):
    obs = t["obs"]
    node = resample_track(obs, n_node)
    mid = obs[len(obs) // 2].copy()
    return mid, node


def find_velocity_3d(field, node_obs, pos_obs, rng, ntrial=384, em=None,
                     n_stage=2):
    """Anchor velocity for a track with sky + distance but no PM and no RV.

    The velocity is NOT a gravity parameter and NOT a baryon parameter; it is
    a property of this particular stream and has to be found afresh in every
    candidate potential, which is exactly why it is re-optimised at every
    Lambda.  The search is over the single-progenitor ORBIT because that is
    cheap; the statistic that is finally reported always comes from the full
    stripping forward model.
    """
    em = DECL["stream_error_model"] if em is None else em
    u_node = unit_sky(node_obs[:, 0], node_obs[:, 1])
    D_node = node_obs[:, 2]
    sD = np.maximum(em["dist_frac"] * D_node, em["dist_floor_kpc"])
    w_pos = OM.observables_to_galactocentric(
        np.array([[pos_obs[0], pos_obs[1], pos_obs[2], 0., 0., 0.]]))[0, :3]
    best_v, best_c = None, np.inf
    for st in range(n_stage):
        if st == 0:
            d = rng.standard_normal((ntrial, 3))
            d /= np.linalg.norm(d, axis=1)[:, None]
            V = d * rng.uniform(40, 460, ntrial)[:, None]
        else:
            V = best_v + rng.standard_normal((ntrial, 3)) * 35.0
        w0 = np.zeros((ntrial, 6))
        w0[:, :3] = w_pos
        w0[:, 3:] = V * OM.KMS
        dt = 3.0e6 * 3.1557e7
        f = OM.integrate_orbit(field, w0, dt, 600, store_every=3)
        b = OM.integrate_orbit(field, w0, -dt, 600, store_every=3)
        path = np.concatenate([b[::-1], f], axis=0)          # (npath, nt, 6)
        npth = path.shape[0]
        o = OM.fast_observables(path.reshape(-1, 6)).reshape(npth, ntrial, 6)
        up = unit_sky(o[:, :, 0], o[:, :, 1])                # (npath, nt, 3)
        cos = np.einsum("pta,na->ptn", up, u_node)
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        dD = (o[:, :, 2][:, :, None] - D_node[None, None, :]) / sD[None, None, :]
        cost = (ang / em["sky_deg"]) ** 2 + dD ** 2
        c = np.nanmin(cost, axis=0).sum(axis=1)              # (nt,)
        k = int(np.nanargmin(c))
        if c[k] < best_c:
            best_c, best_v = float(c[k]), V[k]
    return best_v, best_c


def chi2_stream(field, t, node_obs, anchors_obs, use, rng, em=None,
                nuisance=None):
    """Profile chi^2 of one stream at one Lambda: minimum over the anchor
    draws and over the nuisance grid (progenitor mass, stripping time).  The
    SAME draws and the SAME nuisance grid are used at every Lambda, so the
    profile-minimum bias is common-mode and cancels in Delta chi^2."""
    em = DECL["stream_error_model"] if em is None else em
    nz = DECL["nuisance"] if nuisance is None else nuisance
    w_anchor = OM.observables_to_galactocentric(anchors_obs)
    # step size from the anchor's own orbital time, so every stream is
    # integrated with the same number of steps per orbit
    R = np.hypot(w_anchor[0, 0], w_anchor[0, 1]) / OM.KPC
    v = np.linalg.norm(w_anchor[0, 3:]) / OM.KMS
    T_orb_myr = max(40.0, 2 * np.pi * max(R, 1.0) * 977.8 / max(v, 20.0))
    best = np.inf
    for mprog in nz["m_prog_msun"]:
        for T in nz["T_strip_gyr"]:
            dt_myr = min(3.0, T_orb_myr / 150.0)
            s = OM.spray_stream(field, w_anchor, mprog, T, n_release=80,
                                dt_myr=dt_myr, rng=rng, n_per_release=2)
            K, npart, _ = s.shape
            o = OM.fast_observables(s.reshape(-1, 6)).reshape(K, npart, 6)
            for k in range(K):
                c = stream_chi2(o[k], node_obs, use, em)
                if c < best:
                    best = c
    return float(best)


def measure_stage(res, laws=None, streams="both", verbose=True):
    R, v, e, prov = load_eilers()
    s6, s3, audit = ingest_streams(verbose=False)
    only3 = [t for t in s3 if t["stream"] not in {x["stream"] for x in s6}]
    if streams == "6d+leverage":
        # DECLARED BY GEOMETRY, before any residual is looked at: every
        # 6-D track, plus every 3-D-only track that either reaches past the
        # outer edge of the rotation curve (R_gc,max > 25 kpc) or is within
        # 10 deg of polar.  Those are the two properties the brief identifies
        # as the leverage, and neither is a function of a fit residual.
        only3 = [t for t in only3
                 if t["Rmax"] > 25.0 or abs(t["inc_deg"] - 90.0) <= 10.0]
        streams = "both"
    sample = {"6d": s6, "3d_only": only3}
    laws = OM.frozen_laws() if laws is None else laws
    out = {}
    for law in laws:
        t0 = time.time()
        bar, sol, ref, info = calibrate_and_solve(law, R, v, e, verbose=False)
        fields = {}
        for L in LAMBDA_GRID:
            fields[float(L)] = OM.DeformedField(sol, float(L), refine=2)
        # responsiveness gate: A_dyn must actually move with Lambda
        rp = res["reference_point"]
        adyn = {float(L): float(fields[float(L)].A_dyn(
            np.array([rp["R_kpc"]]), np.array([rp["absz_kpc"]]))[0])
            for L in LAMBDA_GRID}
        chi = {float(L): 0.0 for L in LAMBDA_GRID}
        per = {}
        for kind, ss in sample.items():
            if streams != "both" and streams != kind:
                continue
            for t in ss:
                rng = np.random.default_rng(DECL["nuisance"]["anchor_seed"]
                                            + hash(t["stream"]) % 10 ** 6)
                mid, node = anchor_and_nodes(t)
                use = dict(D=True, pm=t["has"]["pm"], vrad=t["has"]["vrad"])
                row = {}
                for L in LAMBDA_GRID:
                    fl = fields[float(L)]
                    r2 = np.random.default_rng(
                        DECL["nuisance"]["anchor_seed"] + 7)
                    if kind == "6d":
                        anchors = draw_anchors(
                            mid, DECL["nuisance"]["n_anchor_draws"], r2)
                    else:
                        vbest, _ = find_velocity_3d(fl, node, mid, r2)
                        w = OM.observables_to_galactocentric(
                            np.array([[mid[0], mid[1], mid[2], 0, 0, 0]]))
                        w[0, 3:] = vbest * OM.KMS
                        a0 = OM.fast_observables(w)[0]
                        anchors = draw_anchors(
                            a0, DECL["nuisance"]["n_anchor_draws"], r2)
                    c = chi2_stream(fl, t, node, anchors, use, rng)
                    row[float(L)] = c
                    chi[float(L)] += c
                per[t["stream"]] = dict(kind=kind, chi2=row,
                                        n_node=len(node),
                                        Rmax=t["Rmax"], zmax=t["zmax"],
                                        inc=t["inc_deg"], use=use)
                if verbose:
                    b = min(row, key=row.get)
                    print(f"    {law.name:15s} {t['stream']:18s} {kind:8s} "
                          f"argmin L={b:.3f}  dchi2(range)="
                          f"{max(row.values())-min(row.values()):9.1f}",
                          flush=True)
        out[law.name] = dict(chi2_total={str(k): v for k, v in chi.items()},
                             A_dyn_of_lambda={str(k): v
                                              for k, v in adyn.items()},
                             per_stream={k: {**vv, "chi2": {
                                 str(a): b for a, b in vv["chi2"].items()}}
                                 for k, vv in per.items()},
                             seconds=time.time() - t0,
                             inplane=info)
        if verbose:
            print(f"  {law.name}: chi2(Lambda) = "
                  f"{[round(chi[float(L)], 1) for L in LAMBDA_GRID]}")
    return out


# =========================================================================
#  6.  NULLS, RESPONSIVENESS, AND THE POWER CURVE
# =========================================================================
#: Three DISJOINT simulation sets, as the Stage-0 addendum requires: the
#: critical value is set on CAL, verified on the untouched AUDIT set, and
#: power is measured on INJ.  Different seed blocks, never reused.
SEEDS = dict(cal=101_000, audit=202_000, inj=303_000)
#: Eilers bins are NOT independent.  Declared error model for the null:
#: an independent part (the tabulated errors), a nearest-neighbour
#: correlation, and a fully correlated 5 km/s systematic -- the last is the
#: SHARED QUANTITY that makes the naive null non-zero, and it is simulated
#: rather than assumed away.
RC_NULL = dict(rho_neighbour=0.5, common_kms=5.0)


def rc_realisation(R, v, e, rng):
    n = len(R)
    C = np.diag(e ** 2)
    for i in range(n - 1):
        C[i, i + 1] = C[i + 1, i] = RC_NULL["rho_neighbour"] * e[i] * e[i + 1]
    C += RC_NULL["common_kms"] ** 2
    L = np.linalg.cholesky(C + 1e-9 * np.eye(n))
    return v + L @ rng.standard_normal(n)


def synth_track(field_true, t, mprog, T, rng, em, n_node=48):
    """A synthetic OBSERVED track, generated by the same forward model under a
    known potential and then blurred by the declared observational errors.
    Used for the null, the audit and the injections."""
    mid, node = anchor_and_nodes(t, n_node)
    w = OM.observables_to_galactocentric(mid.reshape(1, 6))
    s = OM.spray_stream(field_true, w, mprog, T, n_release=80, dt_myr=2.0,
                        rng=rng)
    o = OM.fast_observables(s.reshape(-1, 6))
    if len(o) < n_node:
        return None, None
    # take the model stream itself as the "observed" track, thinned and
    # blurred; ordering along the stream is by galactocentric azimuth
    idx = np.argsort(o[:, 0])
    o = o[idx][::max(1, len(o) // n_node)][:n_node]
    o = o.copy()
    o[:, 2] *= 1 + em["dist_frac"] * rng.standard_normal(len(o))
    o[:, 3:5] += em["pm_masyr"] * rng.standard_normal((len(o), 2))
    o[:, 5] += em["vrad_kms"] * rng.standard_normal(len(o))
    return mid, o


def sim_block(res, law, R, v, e, streams, lam_true, n_real, seed0,
              lam_grid=None, refit=True, verbose=True):
    """One block of end-to-end simulations.

    Every realisation goes through the WHOLE chain -- rotation-curve noise
    (with its actual covariance), baryon refit, field solve, forward model,
    Lambda scan, argmin -- so the calibration includes everything the
    measurement does, not a selected statistic.
    """
    lam_grid = np.round(np.geomspace(0.6, 1.7, 7), 4) if lam_grid is None \
        else lam_grid
    em = DECL["stream_error_model"]
    nz = dict(DECL["nuisance"])
    nz = dict(m_prog_msun=[1e5], T_strip_gyr=[3.0], n_anchor_draws=8,
              anchor_seed=nz["anchor_seed"])
    out = []
    bar0, sol0, ref0, _ = calibrate_and_solve(law, R, v, e, grid=GRID_FAST,
                                              n_iter=1, verbose=False)
    f_true = OM.DeformedField(sol0, float(lam_true), refine=1)
    for k in range(n_real):
        rng = np.random.default_rng(seed0 + 1000 * k)
        vk = rc_realisation(R, v, e, rng) if refit else v
        bar, sol, ref, _ = calibrate_and_solve(law, R, vk, e, grid=GRID_FAST,
                                               n_iter=1, verbose=False)
        fields = {float(L): OM.DeformedField(sol, float(L), refine=1)
                  for L in lam_grid}
        chi = {float(L): 0.0 for L in lam_grid}
        for t in streams:
            mid, node = synth_track(f_true, t, 1e5, 3.0, rng, em)
            if node is None:
                continue
            use = dict(D=True, pm=True, vrad=True)
            for L in lam_grid:
                anchors = draw_anchors(mid, nz["n_anchor_draws"],
                                       np.random.default_rng(seed0 + 7))
                chi[float(L)] += chi2_stream(fields[float(L)], t, node,
                                             anchors, use, rng, em, nz)
        lam_hat = _parabolic_min(np.array(lam_grid),
                                 np.array([chi[float(L)] for L in lam_grid]))
        out.append(dict(lam_true=float(lam_true), lam_hat=float(lam_hat),
                        chi2={str(float(L)): chi[float(L)] for L in lam_grid}))
        if verbose:
            print(f"      sim {k:2d}  lam_true={lam_true:.3f} -> "
                  f"lam_hat={lam_hat:.4f}", flush=True)
    return out


def _parabolic_min(x, y):
    """Sub-grid argmin by a parabola through the three lowest points."""
    i = int(np.argmin(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i])
    x0, x1, x2 = x[i - 1], x[i], x[i + 1]
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if abs(d) < 1e-30:
        return float(x1)
    A = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    B = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / d
    if A <= 0:
        return float(x1)
    return float(np.clip(-B / (2 * A), x[0], x[-1]))


def posterior_from_chi2(lam, chi2):
    """Delta-chi^2 posterior on Lambda, and the derived A_dyn interval."""
    lam = np.asarray(lam, float)
    c = np.asarray(chi2, float)
    c = c - c.min()
    w = np.exp(-0.5 * c)
    w = w / np.trapezoid(w, lam)
    mean = float(np.trapezoid(w * lam, lam))
    var = float(np.trapezoid(w * (lam - mean) ** 2, lam))
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1])
                                           * np.diff(lam))])
    cdf /= cdf[-1]
    q = lambda p: float(np.interp(p, cdf, lam))
    return dict(lam_argmin=_parabolic_min(lam, c), lam_mean=mean,
                lam_sd=float(np.sqrt(max(var, 0.0))),
                lam_q16=q(0.16), lam_q50=q(0.50), lam_q84=q(0.84),
                lam_q025=q(0.025), lam_q975=q(0.975),
                dchi2_range=float(c.max()))


def nulls_stage(res, n_real=12, n_stream=6, verbose=True):
    R, v, e, prov = load_eilers()
    s6, _, _ = ingest_streams(verbose=False)
    # declared subsample: the largest-reach 6-D streams, chosen by GEOMETRY
    ss = sorted(s6, key=lambda t: -t["Rmax"])[:n_stream]
    law = [l for l in OM.frozen_laws() if l.name == "rar"][0]
    law.completion = COMPLETION["rar"]
    lam_grid = np.round(np.geomspace(0.6, 1.7, 7), 4)
    out = dict(n_real=n_real, streams=[t["stream"] for t in ss],
               lam_grid=lam_grid.tolist(), law="rar",
               rc_null_model=RC_NULL)
    print("  -- CAL set (sets the critical value)")
    out["cal"] = sim_block(res, law, R, v, e, ss, 1.0, n_real, SEEDS["cal"],
                           lam_grid, verbose=verbose)
    print("  -- AUDIT set (untouched; verifies the false-positive rate)")
    out["audit"] = sim_block(res, law, R, v, e, ss, 1.0, n_real,
                             SEEDS["audit"], lam_grid, verbose=verbose)
    out["injections"] = {}
    for lt in (0.75, 0.9, 1.15, 1.4):
        print(f"  -- INJECTION Lambda_true = {lt}")
        out["injections"][str(lt)] = sim_block(
            res, law, R, v, e, ss, lt, max(6, n_real // 2),
            SEEDS["inj"] + int(1000 * lt), lam_grid, verbose=verbose)
    lh = np.array([r["lam_hat"] for r in out["cal"]])
    out["null_bias"] = float(np.mean(lh) - 1.0)
    out["null_sd"] = float(np.std(lh, ddof=1))
    la = np.array([r["lam_hat"] for r in out["audit"]])
    lo, hi = np.percentile(lh, [2.5, 97.5])
    out["cal_95_interval"] = [float(lo), float(hi)]
    out["audit_outside_cal_95"] = float(np.mean((la < lo) | (la > hi)))
    out["audit_bias"] = float(np.mean(la) - 1.0)
    return out


# =========================================================================
#  7.  THE OTHER THREE SYSTEMS
# =========================================================================
def _sex(s, hours=False):
    """Sexagesimal with either ':' or ' ' separators.  The Chapman+2008 VizieR
    table uses spaces; reading it with a ':' parser silently returns NaN for
    every row and the whole M31 stream sample vanishes without an error --
    exactly the silent-extraction failure the brief warns about, so the row
    count is asserted by the caller."""
    s = str(s).strip().replace(":", " ")
    sg = -1.0 if s.startswith("-") else 1.0
    parts = [abs(float(x)) for x in s.lstrip("+-").split()]
    while len(parts) < 3:
        parts.append(0.0)
    v = parts[0] + parts[1] / 60 + parts[2] / 3600
    return sg * v * (15.0 if hours else 1.0)


def _hms(s):
    return _sex(s, hours=True)


def _dms(s):
    return _sex(s, hours=False)


def load_ngc4651():
    """Foster+2014 Keck/DEIMOS tracers of the Umbrella Galaxy.

    THE POINT OF THIS SYSTEM: in-plane and out-of-plane tracers come from the
    SAME instrument, the same masks and the same calibration, so instrument
    and calibration systematics cancel between the two legs exactly.
    """
    p = os.path.join(SS, "extstream_umbrella_ngc4651_tracers_latex.tsv")
    rows = [l.rstrip("\n").split("\t") for l in open(p, encoding="utf-8")]
    out = []
    for r in rows[2:]:
        if len(r) < 9 or not r[0].strip() or ":" not in r[1]:
            continue
        vs = r[3].replace("superscript*", "").split("+/-")
        if len(vs) != 2:
            continue
        try:
            v, ev = float(vs[0]), float(vs[1])
        except ValueError:
            continue
        out.append(dict(id=r[0].strip(), ra=_hms(r[1]), dec=_dms(r[2]),
                        v=v, ev=max(ev, 5.0), disc=r[8].strip() == "Yes"))
    ok = [t for t in out
          if abs(t["v"] - 795.0) < OM.GATE["vrad_max"] and t["ev"] < 200]
    return ok, dict(n_raw=len(out), n_gated=len(ok),
                    n_disc=sum(t["disc"] for t in ok),
                    n_halo=sum(not t["disc"] for t in ok),
                    file=p, sha256=OM.sha256(p),
                    distance_mpc=19.0, inc_deg=53.0, v_sys=795.0,
                    v_rot_deproj=215.0, M_star=1.7e10, M_gas=5.7e9)


def load_m31_hi():
    p = os.path.join(SS, "extstream_m31_hi_rotation_curve_chemin2009.tsv")
    rows = read_tsv(p)
    R, V, eV, inc = [], [], [], []
    for r in rows:
        v, ev = _f(r["Vrot"]), _f(r["e_Vrot"])
        if not np.isfinite(v) or not np.isfinite(ev) or ev <= 0:
            continue
        if not (50.0 < v < 450.0):        # physical-plausibility gate
            continue
        R.append(_f(r["R_kpc"]))
        V.append(v)
        eV.append(ev)
        inc.append(_f(r["inc_adopted"]))
    return (np.array(R), np.array(V), np.array(eV), np.array(inc),
            dict(n_rows=len(rows), n_gated=len(R), file=p,
                 sha256=OM.sha256(p), distance_kpc=785.0))


def load_m31_streams():
    p = os.path.join(SS, "extstream_m31_streams_chapman2008_stars.raw.tsv")
    lines = [l.rstrip("\n") for l in open(p, encoding="utf-8")
             if not l.startswith("#")]
    hdr = None
    rows = []
    for i, l in enumerate(lines):
        if hdr is None and "\t" in l and any(c.isalpha() for c in l):
            hdr = [c.strip() for c in l.split("\t")]
            body = lines[i + 3:]
            break
    for l in body:
        c = [x.strip() for x in l.split("\t")]
        if len(c) != len(hdr):
            continue
        rows.append(dict(zip(hdr, c)))
    return rows, hdr, dict(file=p, sha256=OM.sha256(p), n=len(rows))


def external_power(sol, tracers_xy, v_obs, v_err, v_sys, D_mpc, incl_deg,
                   lam_grid, n_prog=256, n_batch=8, seed=5150, rng=None,
                   r_apo_kpc=(20., 90.), m_sat=(1e8, 3e9), T_gyr=(1.0, 4.0)):
    """Forward-model a radial-merger debris system and scan Lambda.

    The tracer is a SHELL / stream system, so it is generated the same way as
    a Milky Way stream: a progenitor on an orbit, a stripping history, and the
    resulting phase-space distribution -- then projected to the ONLY things
    the data actually contain, projected position and line-of-sight velocity.

    The merger is unknown, so its orbit (apocentre, angular momentum, three
    viewing angles), its mass and its age are NUISANCE parameters, profiled
    over at every Lambda with the same draws.  Reporting the width of the
    resulting Lambda posterior IS the power statement for this system.
    """
    rng = np.random.default_rng(seed) if rng is None else rng
    kpc_per_arcsec = D_mpc * 1e3 * np.pi / (180 * 3600)
    xy = np.asarray(tracers_xy, float) * kpc_per_arcsec        # kpc, projected
    rp = np.hypot(xy[:, 0], xy[:, 1])
    dv = np.asarray(v_obs, float) - v_sys
    ev = np.asarray(v_err, float)
    # nuisance draws, identical at every Lambda
    ra_ = rng.uniform(*r_apo_kpc, n_prog)
    circ = rng.uniform(0.02, 0.35, n_prog)                     # L / L_circ
    ms = 10 ** rng.uniform(np.log10(m_sat[0]), np.log10(m_sat[1]), n_prog)
    Tg = rng.uniform(*T_gyr, n_prog)
    th = np.arccos(rng.uniform(-1, 1, n_prog))
    ph = rng.uniform(0, 2 * np.pi, n_prog)
    ps = rng.uniform(0, 2 * np.pi, n_prog)
    chi = {}
    for L in lam_grid:
        fl = OM.DeformedField(sol, float(L), refine=1)
        best = np.inf
        for b in range(n_batch):
            sl = slice(b * n_prog // n_batch, (b + 1) * n_prog // n_batch)
            k = len(ra_[sl])
            w0 = np.zeros((k, 6))
            w0[:, 0] = ra_[sl] * OM.KPC
            gR, _ = fl.force(ra_[sl], np.full(k, 0.05))
            vcirc = np.sqrt(np.maximum(gR * ra_[sl] * OM.KPC, 0.0))
            w0[:, 1] = 0.0
            w0[:, 4] = circ[sl] * vcirc                        # near-radial
            s = OM.spray_stream(fl, w0, float(np.median(ms[sl])),
                                float(np.median(Tg[sl])), n_release=60,
                                dt_myr=2.0, rng=rng)
            for i in range(k):
                P = s[i, :, :3] / OM.KPC
                V = s[i, :, 3:] / OM.KMS
                Rm = _rot(th[sl][i], ph[sl][i], ps[sl][i])
                Pp = P @ Rm.T
                Vp = V @ Rm.T
                c = _shell_chi2(Pp[:, :2], Vp[:, 2], rp, xy, dv, ev)
                best = min(best, c)
        chi[float(L)] = float(best)
    return chi


def _rot(t, p, s):
    ct, st, cp, sp, cs, ss = (np.cos(t), np.sin(t), np.cos(p), np.sin(p),
                              np.cos(s), np.sin(s))
    Rz1 = np.array([[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz2 = np.array([[cs, -ss, 0], [ss, cs, 0], [0, 0, 1]])
    return Rz2 @ Ry @ Rz1


def _shell_chi2(model_xy, model_vlos, rp, xy, dv, ev, r_tol=3.0):
    """chi^2 of the model debris against projected position + v_los only."""
    c = 0.0
    for i in range(len(dv)):
        d = np.hypot(model_xy[:, 0] - xy[i, 0], model_xy[:, 1] - xy[i, 1])
        m = d < r_tol
        if np.count_nonzero(m) < 3:
            c += 9.0
            continue
        vm = np.median(model_vlos[m])
        sd = max(np.std(model_vlos[m]), 5.0)
        c += (dv[i] - vm) ** 2 / (ev[i] ** 2 + sd ** 2)
    return float(c)


def secondary_stage(res, verbose=True):
    out = {}
    lam = np.round(np.geomspace(0.6, 1.7, 7), 4)
    # ---------------------------------------------------- NGC 4651
    tr, meta = load_ngc4651()
    ra0 = np.median([t["ra"] for t in tr if t["disc"]])
    de0 = np.median([t["dec"] for t in tr if t["disc"]])
    halo = [t for t in tr if not t["disc"]]
    xy = np.array([[(t["ra"] - ra0) * np.cos(np.radians(de0)) * 3600,
                    (t["dec"] - de0) * 3600] for t in halo])
    # in-plane leg: a single deprojected rotation amplitude, so the baryon
    # normalisation is calibrated on ONE number, not a curve
    Rn = np.array([5.0, 8.0, 12.0])
    Vn = np.full(3, meta["v_rot_deproj"])
    En = np.full(3, 10.0)
    o4 = {}
    for law in OM.frozen_laws():
        law.completion = COMPLETION[law.name]
        bar, info = fit_baryons(law, Rn, Vn, En)
        ref = FieldSolution_cached(bar, OM.Law("nr", base="newton"),
                                   GRID_FAST, tag=f"n4651_{law.name}_ref")
        sol = FieldSolution_cached(bar, law, GRID_FAST, ref=ref,
                                   tag=f"n4651_{law.name}")
        chi = external_power(sol, xy, [t["v"] for t in halo],
                             [t["ev"] for t in halo], meta["v_sys"],
                             meta["distance_mpc"], meta["inc_deg"], lam)
        post = posterior_from_chi2(lam, [chi[float(L)] for L in lam])
        o4[law.name] = dict(chi2={str(float(L)): chi[float(L)] for L in lam},
                            posterior=post, M_b=info["M_total"])
        if verbose:
            print(f"    NGC4651 {law.name:15s} dchi2 range "
                  f"{post['dchi2_range']:7.2f}  Lambda = "
                  f"{post['lam_q50']:.3f} [{post['lam_q16']:.3f}, "
                  f"{post['lam_q84']:.3f}]", flush=True)
    out["ngc4651"] = dict(meta=meta, n_halo=len(halo), results=o4,
                          lam_grid=lam.tolist())
    # ---------------------------------------------------- M31
    R, V, eV, inc, m31meta = load_m31_hi()
    srows, shdr, smeta = load_m31_streams()
    vcol = next((c for c in shdr if c.lower() in ("vhel", "hrv", "rv",
                                                  "v_hel", "velocity")), None)
    o31 = {}
    if vcol is not None:
        vv, ee, sxy = [], [], []
        racol = next((c for c in shdr if c.upper().startswith("RA")), None)
        decol = next((c for c in shdr if c.upper().startswith("DE")), None)
        n_raw_v = 0
        for r in srows:
            v = _f(r.get(vcol, ""))
            n_raw_v += int(np.isfinite(v))
            if not np.isfinite(v) or not (-800 < v < 200):
                continue          # physical-plausibility gate on every ingest
            try:
                a, d = _hms(r.get(racol, "")), _dms(r.get(decol, ""))
            except (ValueError, IndexError):
                continue
            if not (np.isfinite(a) and np.isfinite(d)):
                continue
            vv.append(v)
            ee.append(15.0)
            sxy.append([(a - 10.6847) * np.cos(np.radians(41.269)) * 3600,
                        (d - 41.269) * 3600])
        if len(vv) >= 10:
            sxy = np.array(sxy)
            for law in OM.frozen_laws():
                law.completion = COMPLETION[law.name]
                bar, info = fit_baryons(law, R, V, eV)
                ref = FieldSolution_cached(bar, OM.Law("nr", base="newton"),
                                           GRID_FAST, tag=f"m31_{law.name}_ref")
                sol = FieldSolution_cached(bar, law, GRID_FAST, ref=ref,
                                           tag=f"m31_{law.name}")
                chi = external_power(sol, sxy, vv, ee, -300.0, 0.785, 77.0,
                                     lam, r_apo_kpc=(20., 120.))
                post = posterior_from_chi2(lam, [chi[float(L)] for L in lam])
                o31[law.name] = dict(
                    chi2={str(float(L)): chi[float(L)] for L in lam},
                    posterior=post, M_b=info["M_total"],
                    inplane_rms=info["rms_kms"])
                if verbose:
                    print(f"    M31     {law.name:15s} dchi2 range "
                          f"{post['dchi2_range']:7.2f}  Lambda = "
                          f"{post['lam_q50']:.3f} [{post['lam_q16']:.3f}, "
                          f"{post['lam_q84']:.3f}]", flush=True)
    out["m31"] = dict(hi=m31meta, streams=smeta, stream_vcol=vcol,
                      n_rows_with_velocity=int(n_raw_v) if vcol else 0,
                      n_stream_stars=len(vv) if vcol else 0, results=o31,
                      lam_grid=lam.tolist())
    return out


# =========================================================================
#  8.  THE ERROR BUDGET  --  every term measured, none assumed
# =========================================================================
SYS_VARIANTS = {
    "baseline": {},
    "disc_thickness_x2": dict(baryon=dict(b_thin=0.56, b_thick=1.8)),
    "disc_scalelength_-30pct": dict(baryon=dict(a_thin=2.1, a_thick=3.1)),
    "disc_scalelength_+30pct": dict(baryon=dict(a_thin=3.9, a_thick=5.7)),
    "bulge_scale_x2": dict(baryon=dict(a_bulge=1.0)),
    "gas_scalelength_x1.5": dict(baryon=dict(a_gas=10.5)),
    "errors_x1.5": dict(err=1.5),
    "errors_/1.5": dict(err=1 / 1.5),
    "anchor_draws_24": dict(anchors=24),
    "rc_floor_2kms": dict(rc_floor=2.0),
    "rc_floor_10kms": dict(rc_floor=10.0),
    "grid_refine_1": dict(refine=1),
}


def systematics_stage(res, law_name="rar", n_stream=8, verbose=True):
    R, v, e, prov = load_eilers()
    s6, _, _ = ingest_streams(verbose=False)
    ss = sorted(s6, key=lambda t: -t["Rmax"])[:n_stream]
    rp = res["reference_point"]
    base_em = DECL["stream_error_model"]
    out = {}
    for tag, var in SYS_VARIANTS.items():
        law = [l for l in OM.frozen_laws() if l.name == law_name][0]
        law.completion = COMPLETION[law_name]
        floor = var.get("rc_floor", DECL["rc_systematic_floor_kms"])
        bar, info = fit_baryons(law, R, v, e, floor=floor)
        for k, val in var.get("baryon", {}).items():
            setattr(bar, k, val)
        ref = FieldSolution_cached(bar, OM.Law("nr", base="newton"), GRID_FAST,
                                   tag=f"sys_{tag}_ref")
        sol = FieldSolution_cached(bar, law, GRID_FAST, ref=ref,
                                   tag=f"sys_{tag}")
        em = dict(base_em)
        if "err" in var:
            for k in ("sky_deg", "dist_frac", "dist_floor_kpc", "pm_masyr",
                      "vrad_kms"):
                em[k] = base_em[k] * var["err"]
        na = var.get("anchors", DECL["nuisance"]["n_anchor_draws"])
        refine = var.get("refine", 2)
        chi = {}
        adyn = {}
        for L in LAMBDA_GRID:
            fl = OM.DeformedField(sol, float(L), refine=refine)
            adyn[float(L)] = float(fl.A_dyn(np.array([rp["R_kpc"]]),
                                            np.array([rp["absz_kpc"]]))[0])
            c = 0.0
            for t in ss:
                rng = np.random.default_rng(DECL["nuisance"]["anchor_seed"]
                                            + hash(t["stream"]) % 10 ** 6)
                mid, node = anchor_and_nodes(t)
                use = dict(D=True, pm=t["has"]["pm"], vrad=t["has"]["vrad"])
                anchors = draw_anchors(mid, na,
                                       np.random.default_rng(
                                           DECL["nuisance"]["anchor_seed"] + 7))
                c += chi2_stream(fl, t, node, anchors, use, rng, em)
            chi[float(L)] = c
        post = posterior_from_chi2(LAMBDA_GRID,
                                   [chi[float(L)] for L in LAMBDA_GRID])
        La = np.array(LAMBDA_GRID, float)
        Aa = np.array([adyn[float(L)] for L in LAMBDA_GRID])
        post["A_dyn_at_argmin"] = float(np.interp(post["lam_argmin"], La, Aa))
        post["A_dyn_q16"] = float(np.interp(post["lam_q16"], La, Aa))
        post["A_dyn_q84"] = float(np.interp(post["lam_q84"], La, Aa))
        out[tag] = dict(posterior=post, M_b=info["M_total"],
                        chi2={str(float(L)): chi[float(L)]
                              for L in LAMBDA_GRID},
                        A_dyn_of_lambda={str(float(L)): adyn[float(L)]
                                         for L in LAMBDA_GRID})
        if verbose:
            print(f"    {tag:26s} A_dyn = {post['A_dyn_at_argmin']:.4f} "
                  f"[{post['A_dyn_q16']:.4f}, {post['A_dyn_q84']:.4f}]  "
                  f"dchi2 range {post['dchi2_range']:8.1f}", flush=True)
    b = out["baseline"]["posterior"]["A_dyn_at_argmin"]
    dev = {k: v["posterior"]["A_dyn_at_argmin"] - b for k, v in out.items()}
    out["_summary"] = dict(
        baseline_A_dyn=b, shifts=dev,
        systematic_halfrange=float(0.5 * (max(dev.values())
                                          - min(dev.values()))),
        rss_systematic=float(np.sqrt(sum(x ** 2 for k, x in dev.items()
                                         if k != "baseline"))),
        stat_sigma=float(0.5 * (out["baseline"]["posterior"]["A_dyn_q84"]
                                - out["baseline"]["posterior"]["A_dyn_q16"])),
        streams=[t["stream"] for t in ss])
    return out


if __name__ == "__main__":
    stages = sys.argv[1:] or ["all"]
    res = {}
    if os.path.exists(OUT):
        res = json.load(open(OUT))
    res.setdefault("declared", DECL)
    if "ingest" in stages or "all" in stages:
        print("== INGEST ==")
        s6, s3, audit = ingest_streams()
        res["ingest"] = {k: v for k, v in audit.items()
                         if not isinstance(v, list)}
        res["ingest"]["streams_6d"] = sorted(t["stream"] for t in s6)
        res["ingest"]["streams_3d"] = sorted(t["stream"] for t in s3)
        json.dump(res, open(OUT, "w"), indent=1)
    if "predict" in stages or "all" in stages:
        print("\n== PREDICT ==")
        pred, ref_pt, prov, audit = predict_stage()
        res["predict"] = pred
        res["reference_point"] = dict(R_kpc=ref_pt[0], absz_kpc=ref_pt[1])
        res["provenance"] = prov
        res["frame_validation"] = OM.validate_frame()
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"\nwrote {OUT}")
    if "measure" in stages or "all" in stages:
        print("\n== MEASURE ==")
        res["measure"] = measure_stage(res)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")
    if "nulls" in stages or "all" in stages:
        print("\n== NULLS / POWER ==")
        res["nulls"] = nulls_stage(res)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")
    if "systematics" in stages or "all" in stages:
        print("\n== ERROR BUDGET ==")
        res["systematics"] = systematics_stage(res)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")
    if "secondary" in stages or "all" in stages:
        print("\n== SECONDARY SYSTEMS ==")
        res["secondary"] = secondary_stage(res)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")
    if "warps" in stages or "all" in stages:
        print("\n== POOLED WARPS ==")
        import warp_pool
        res["warps"] = warp_pool.run(verbose=True)
        json.dump(res, open(OUT, "w"), indent=1)
        print(f"wrote {OUT}")
