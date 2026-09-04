"""run_mechanism.py -- JOB 1: WHY do the current detectors confuse a triaxial
collisionless halo with a tensor response?

Run BF measured the confusion (family-wise 0.648 on the dark-matter universe).
This module measures its MECHANISM.  Every entry is a measured separation with
an error, not an argument.

  M1  the RADIAL PROFILE of the quadrupole amplitude
  M2  its dependence on BARYONIC MORPHOLOGY -- a tensor response is sourced by
      the field the baryons make, a halo's shape is not
  M3  the relation between the quadrupole PHASE and independently measured axes
  M4  the joint behaviour of MATTER and LIGHT
  M5  the response to the COARSE-GRAINING and COMMUTATION operations already
      built in scene/commutation.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import zlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import guard                                  # noqa: E402
import worker as W                            # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
N_DIAG = int(os.environ.get("N_DIAG", 300))

ARMS = {
    "U03_mond": ("U03_mond_scalar", None, 1.0, 1.0),
    "H0_scalar_null": ("H0_scalar_null", None, 1.0, 1.0),
    "U10_systematics": ("U10_systematics", None, 3.0, 1.0),
    "U02_cdm": ("U02_cdm", None, 1.0, 1.0),
    "U05_thresh": ("U05_tensor_axis", 0.0200293, 1.0, 1.0),
    "U05_fid": ("U05_tensor_axis", 0.5, 1.0, 1.0),
    "U05_A2": ("U05_tensor_axis", 2.0, 1.0, 1.0),
    "U06_fid": ("U06_wellnet", 0.06, 1.0, 1.0),
}


def _agg(pool, key, sub=None):
    v = []
    for r in pool:
        d = r["_diag"]
        x = d[key] if sub is None else d[key][sub]
        if np.isfinite(x):
            v.append(x)
    v = np.array(v, float)
    if len(v) == 0:
        return dict(mean=float("nan"), sd=float("nan"), sem=float("nan"), n=0)
    return dict(mean=float(v.mean()), sd=float(v.std()),
                sem=float(v.std() / np.sqrt(len(v))), n=len(v))


# --------------------------------------------------------------- M2 helper
def morph_slope(pool):
    """Slope of the dimensionless quadrupole power on the baryon ellipticity.

    Pooled over every cluster of every corpus in the arm, so the slope is the
    population relation, not a per-corpus fit.  Quote the SLOPE with its
    standard error, never a correlation.
    """
    x, y = [], []
    for r in pool:
        d = r["_diag"]
        x.extend(d["ell_bar"])
        y.extend(d["power"])
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 20:
        return dict(slope=float("nan"), se=float("nan"), t=float("nan"), n=len(x))
    A = np.stack([np.ones_like(x), x], 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    s2 = float(res @ res) / max(len(x) - 2, 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    return dict(slope=float(beta[1]), se=se,
                t=float(beta[1] / se) if se > 0 else float("nan"),
                n=int(len(x)), mean_power=float(np.mean(y)))


# ----------------------------------------------------- M4: member dynamics
def member_quadrupole(cd):
    """Azimuthal m=2 modulation of the member velocity dispersion.

    A metric quadrupole -- from a triaxial halo OR from a tensor response --
    must appear in the MATTER sector as well as in the light sector.  This
    measures whether it does, in this generator, and on which axis.

    Model:  v_i^2 ~ sigma^2(R_i) [1 + a cos 2(phi_i - phi_0)],
    fitted as a weighted regression of v^2 on {1, cos2phi, sin2phi} within
    radial bins, with sigma^2(R) absorbed by the per-bin constant.
    """
    R = np.hypot(cd["mem_x"], cd["mem_y"])
    ph = np.arctan2(cd["mem_y"], cd["mem_x"])
    v2 = cd["mem_v"] ** 2
    ok = cd["mem_p"] > 0.6
    R5 = cd["R500"]
    P = np.zeros((2, 2))
    acc = np.zeros(2)
    for a, b in ((0.1, 0.5), (0.5, 1.0), (1.0, 2.2)):
        s = ok & (R >= a * R5) & (R < b * R5)
        n = int(s.sum())
        if n < 15:
            continue
        t = ph[s]
        D = np.stack([np.ones(n), np.cos(2 * t), np.sin(2 * t)], 1)
        y = v2[s]
        try:
            XtXi = np.linalg.inv(D.T @ D)
        except np.linalg.LinAlgError:
            continue
        beta = XtXi @ (D.T @ y)
        c0 = float(beta[0])
        if not np.isfinite(c0) or c0 < 1.0e3:      # sigma < ~32 km/s: unusable
            continue
        # v is Gaussian with variance c0, so Var(v^2) = 2 c0^2 and the
        # FRACTIONAL amplitude a = beta[1:]/c0 has covariance 2 * (D'D)^-1
        Cf = 2.0 * XtXi[np.ix_([1, 2], [1, 2])]
        try:
            Pi = np.linalg.inv(Cf)
        except np.linalg.LinAlgError:
            continue
        a = np.clip(beta[1:] / c0, -2.0, 2.0)
        P += Pi
        acc += Pi @ a
    if np.linalg.cond(P) > 1e12 or P[0, 0] <= 0:
        return None
    Cc = np.linalg.inv(P)
    amp = Cc @ acc
    return complex(amp[0], amp[1]), float(np.sqrt(np.trace(Cc) / 2.0))


def dynamics_check(n=120):
    """M4: is the quadrupole present in the MATTER sector too?"""
    from universes import corpus as cp
    from universes import generate as gn
    from universes import physics as ph
    lib = gn.get_lib()
    out = {}
    for arm, spec in (("U02_cdm", ARMS["U02_cdm"]),
                      ("U05_fid", ARMS["U05_fid"]),
                      ("U03_mond", ARMS["U03_mond"])):
        pb, pe, sds = [], [], []
        for k in range(n):
            rng = np.random.default_rng(9_100_000 + 977 * k + zlib.crc32(arm.encode()) % 1000)
            u = W.make_universe(spec, rng)
            C = cp.draw_corpus(u, lib, rng, n_gal=1, n_clu=12, n_sn=5)
            for cd in C.clu:
                r = member_quadrupole(cd)
                if r is None:
                    continue
                Zm, sd = r
                for ang, acc in ((cd["pa_bar_obs"], pb), (cd["axis_ext_obs"], pe)):
                    a = np.deg2rad(ang)
                    acc.append(float(Zm.real * np.cos(2 * a) + Zm.imag * np.sin(2 * a)))
                sds.append(sd)
        pb = np.array(pb)
        pe = np.array(pe)

        def summ(v):
            v = v[np.isfinite(v)]
            lo, hi = np.quantile(v, [0.02, 0.98])
            t = v[(v >= lo) & (v <= hi)]
            return dict(mean_trimmed=float(t.mean()),
                        sem_trimmed=float(t.std() / np.sqrt(len(t))),
                        median=float(np.median(v)),
                        n=int(len(v)))
        out[arm] = dict(n_clusters=len(pb), proj_baryon_axis=summ(pb),
                        proj_external_axis=summ(pe),
                        median_per_cluster_error=float(np.median(sds)))
    return out


# --------------------------------------------- M5: commutation / coarse-graining
def commutation_check():
    """M5: does the quadrupole survive azimuthal averaging of the SOURCE?

    A triaxial collisionless halo is a SOURCE with a shape; an external-axis
    tensor is a LAW.  ``AzimuthalAverage`` keeps every source's radius and
    randomises its angles: it destroys a source's own axis and leaves an
    externally imposed one untouched.  So the two mechanisms, which are
    observationally confusable, are exactly opposite under this operation.

    Reported: the P2 quadrupole of the radial field on a shell, before and
    after, for each (law, axis) pair.
    """
    sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "scene")))
    import commutation as CM
    rng = np.random.default_rng(4242)
    flat = CM.flattened_cluster_scene(n_gal=400, seed=31, q_z=0.45,
                                  n_diffuse=30000)
    ext_axis = np.array([1.0, 0.0, 0.0])          # orthogonal to the flattening
    src_axis = CM.SourceAlignedTensor.principal_axis(flat)
    ops = {"azimuthal_average": CM.AzimuthalAverage(),
           "spherical_average": CM.SphericalAverage(n_dir=24)}
    laws = {"newton_on_triaxial_source": CM.Newtonian(),
            "external_axis_tensor_A0.5": CM.ExternalAxisTensor(A=0.5, axis=ext_axis),
            "source_aligned_tensor_A0.5": CM.SourceAlignedTensor(A=0.5)}
    radii = [500.0 * CM.KPC, 1000.0 * CM.KPC, 1500.0 * CM.KPC]
    out = {"source_principal_axis": src_axis.tolist(),
           "external_axis": ext_axis.tolist(), "results": {}}
    for lname, law in laws.items():
        row = {}
        for aname, axis in (("about_external_axis", ext_axis),
                            ("about_source_axis", src_axis)):
            q0 = [CM.shell_quadrupole(law, flat, r, axis=axis, n_dir=192, n_rot=4)
                  for r in radii]
            per_op = {}
            for oname, op in ops.items():
                qs = []
                for k in range(3 if not op.deterministic else 1):
                    s2 = op(flat, np.random.default_rng(100 + k))
                    qs.append([CM.shell_quadrupole(law, s2, r, axis=axis,
                                                   n_dir=192, n_rot=4)
                               for r in radii])
                qa = np.array(qs).mean(0)
                keep = np.abs(qa) / np.maximum(np.abs(np.array(q0)), 1e-12)
                per_op[oname] = dict(
                    after=[float(v) for v in qa],
                    surviving_fraction=[float(v) for v in keep],
                    surviving_fraction_median=float(np.median(keep)))
            row[aname] = dict(before=[float(v) for v in q0], ops=per_op)
        out["results"][lname] = row
    out["radii_kpc"] = [r / CM.KPC for r in radii]
    return out


def main():
    guard.start()
    t0 = time.time()
    out = {"n_diag": N_DIAG}

    print("M1-M3: per-arm quadrupole diagnostics", flush=True)
    arms = {}
    for arm, spec in ARMS.items():
        jobs = [(spec, 6_600_000 + 5003 * zlib.crc32(arm.encode()) % 100000 + i, 12)
                for i in range(N_DIAG)]
        pool = W.run_batch(jobs, diag=True)
        arms[arm] = dict(
            n_corpora=len(pool),
            amplitude=_agg(pool, "amp_median"),
            snr=_agg(pool, "snr_median"),
            # M1 radial profile of the dimensionless quadrupole power
            radial_profile_studentised=[_agg(pool, "prof", k) for k in range(3)],
            radial_profile_Q2=[_agg(pool, "prof_Q2", k) for k in range(3)],
            # M3 phase
            concentration_about_baryon_axis=_agg(pool, "R_bar"),
            concentration_about_external_axis=_agg(pool, "R_ext"),
            median_phase_error_vs_baryon_axis_deg=_agg(pool, "err_bar_median"),
            median_phase_error_vs_external_axis_deg=_agg(pool, "err_ext_median"),
            raw_projection_baryon=_agg(pool, "raw_proj_bar"),
            raw_projection_external=_agg(pool, "raw_proj_ext"),
            raw_difference=_agg(pool, "raw_diff"),
            # M2 morphology
            morphology_slope=morph_slope(pool),
        )
        print(f"  {arm:<18} amp={arms[arm]['amplitude']['mean']:.4f} "
              f"snr={arms[arm]['snr']['mean']:5.2f} "
              f"R_bar={arms[arm]['concentration_about_baryon_axis']['mean']:.3f} "
              f"R_ext={arms[arm]['concentration_about_external_axis']['mean']:.3f}",
              flush=True)
    out["M1_M3_arms"] = arms
    W.close_pool()

    print("M4: matter-sector quadrupole (member dynamics)", flush=True)
    out["M4_matter_sector"] = dynamics_check()

    print("M5: commutation / coarse-graining", flush=True)
    out["M5_commutation"] = commutation_check()

    out["provenance"] = guard.stop()
    out["elapsed_s"] = time.time() - t0
    p = os.path.join(RES, "M_mechanism.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
