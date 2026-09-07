"""test_lane.py -- tests for the paired emitter, the invariants and the guards.

Every lane in this programme has found a real bug in its own first version.
This one found the projection-grid error (T2: Sigma too large by 6-60x with
a 1/R dependence) and the beam-smearing bias of the ring-fitted rotation curve
(T5: a 0.1 dex noise-free scatter in the radial boost), both before any
result was produced.  They run every time.

    python test_lane.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(HERE, "..", "cdm-separation")))

import forward2 as F2            # noqa: E402
import halo as H                 # noqa: E402
import invariants as IV          # noqa: E402
import paired as P               # noqa: E402
import gen1 as W                 # noqa: E402
from universes import corpus as cp       # noqa: E402
from universes.baryons import G          # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(dict(test=name, passed=bool(ok), detail=detail))
    print(f"  {'PASS' if ok else 'FAIL':<5} {name:<52} {detail}")
    return ok


# ----------------------------------------------------------------- T1 pairing
def t1_pairing():
    """U4..U9 at knob = 0 must reproduce U3 BIT FOR BIT on a paired set (same
    scene, same noise streams), and two different universes must share their
    noise: the difference of two catalogues is the signal difference only."""
    lib = W.get_lib("train")
    gm = lib.geoms[2]
    gal = lib.galaxies[5]
    u3 = P.make_universe("U03_mond_scalar", None, 1.0, 1.0, 77)
    ref_c = P.emit_cluster(u3, gm, 77)
    ref_g = P.emit_galaxy(u3, gal, 77)
    worst = 0.0
    for uid in ("U04_env_scalar", "U05_tensor_axis", "U06_wellnet", "U07_memory",
                "U08_ep_slip", "U09_path_redshift"):
        u = P.make_universe(uid, 0.0, 1.0, 1.0, 77)
        c = P.emit_cluster(u, gm, 77)
        g = P.emit_galaxy(u, gal, 77)
        for k in ("e1", "e2", "mem_v", "kT_obs", "y_sz", "z_src_phot"):
            worst = max(worst, float(np.max(np.abs(c[k] - ref_c[k]))))
        for k in ("v_map", "sz_obs"):
            worst = max(worst, float(np.max(np.abs(g[k] - ref_g[k]))))
    check("T1 knob = 0 deformations reproduce U3 bit for bit", worst == 0.0,
          f"max |difference| = {worst:.2e}")
    # noise shared between two DIFFERENT universes: e1 - e1' equals the
    # difference of the noise-free signals
    u2 = P.make_universe("U02_cdm", None, 1.0, 1.0, 77)
    dm = H.draw_cluster_halo(gm.clu, u2.params, np.random.default_rng(1), H.knobs())
    c2 = P.emit_cluster(u2, gm, 77, dm=dm)
    same_src = np.allclose(c2["src_x"], ref_c["src_x"]) and np.allclose(c2["z_src_phot"], ref_c["z_src_phot"])
    d = c2["e1"] - ref_c["e1"]
    # the difference must be far smoother than shape noise: its rms across
    # sources is that of a smooth field (~0.01-0.1), not sqrt(2) * 0.26
    check("T1 two universes share sources, photo-z and shape noise",
          same_src and np.std(d) < 0.15,
          f"sources identical: {same_src}; rms(e1 difference) = {np.std(d):.3f} vs shape noise 0.26")


# --------------------------------------------------------- T2 closed forms
def t2_projection():
    v = 1000.0
    rg = np.geomspace(1, 20000, 300)
    g = v ** 2 / rg
    Rq = np.array([20.0, 100.0, 300.0, 1000.0, 3000.0])
    S, Sb = IV.project_sigma(rg, g, Rq)
    e1 = float(np.max(np.abs(S / (v ** 2 / (4 * G * Rq)) - 1)))
    e2 = float(np.max(np.abs(Sb / (v ** 2 / (2 * G * Rq)) - 1)))
    check("T2 analysis projection matches the SIS closed form", max(e1, e2) < 0.03,
          f"max relative error Sigma {e1:.1e}, Sigmabar {e2:.1e} (first version: 6-60x, 1/R)")
    import forward as BK
    M200, c = 1e15, 4.0
    r2 = cp.r200_of(M200)
    rs = r2 / c
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    gg = G * M200 * mu(rg / rs) / mu(c) / rg ** 2
    S, Sb = IV.project_sigma(rg, gg, Rq)
    Sa, Sba, _, _ = BK.nfw_sigma(Rq, M200, c)
    e = float(np.max(np.abs(np.concatenate([S / Sa - 1, Sb / Sba - 1]))))
    check("T2 analysis projection matches the analytic NFW", e < 0.01, f"max relative error {e:.1e}")
    # generator 2's shell projection against the same closed form
    M = M200 * mu(rg / rs) / mu(c)
    S2, Sb2 = F2.shell_project(rg, M, Rq)
    e = float(np.max(np.abs(np.concatenate([S2 / Sa - 1, Sb2 / Sba - 1]))))
    check("T2 generator-2 shell projection matches the analytic NFW", e < 0.02,
          f"max relative error {e:.1e}")


def t3_hse_jeans():
    v = 1000.0
    rg = np.geomspace(1, 20000, 300)
    g = v ** 2 / rg
    rho = 1e5 * (rg / 100.0) ** -2.0
    kT = IV.hydrostatic_kT(rg, g, rho)
    e = float(np.max(np.abs(kT[50:250] / (v ** 2 / 2 * IV.KEV_PER_KMS2) - 1)))
    check("T3 analysis hydrostatic integral on an isothermal SIS", e < 0.01, f"max relative error {e:.1e}")
    nu = (rg / 100.0) ** -2.0
    sl = IV.jeans_sigma_los(rg, g, nu, np.array([50.0, 200.0, 800.0]))
    e = float(np.max(np.abs(sl / (v / np.sqrt(2)) - 1)))
    check("T3 analysis Jeans on an isothermal SIS", e < 0.02, f"max relative error {e:.1e}")
    kT2 = F2.hse_kT2(rg, g, rho)
    e = float(np.max(np.abs(kT2[50:250] / (v ** 2 / 2 * F2.KEV_PER_KMS2) - 1)))
    check("T3 generator-2 hydrostatic sum on an isothermal SIS", e < 0.03, f"max relative error {e:.1e}")
    # Osipkov-Merritt with r_a -> infinity is isotropic: sigma_los = v/sqrt2
    sl2 = F2.om_jeans_los(rg, g, nu, np.array([50.0, 200.0, 800.0]), ra=1e7)
    e = float(np.max(np.abs(sl2 / (v / np.sqrt(2)) - 1)))
    check("T3 generator-2 Jeans (isotropic limit) on an isothermal SIS", e < 0.03, f"max relative error {e:.1e}")


# ------------------------------------------------ T4 invariance of the reduction
def t4_invariance():
    lib = W.get_lib("train")
    u = P.make_universe("U02_cdm", None, 1.0, 1.0, 9)
    dm = H.draw_cluster_halo(lib.geoms[4].clu, u.params, np.random.default_rng(2), H.knobs())
    cd = P.emit_cluster(u, lib.geoms[4], 9, dm=dm)
    r0 = IV.reduce_cluster(cd)
    # rotate the catalogue by 41 degrees, rotate the observed axes with it
    dth = np.deg2rad(41.0)
    cd2 = dict(cd)
    x, y = cd["src_x"], cd["src_y"]
    cd2["src_x"] = x * np.cos(dth) - y * np.sin(dth)
    cd2["src_y"] = x * np.sin(dth) + y * np.cos(dth)
    cd2["e1"] = cd["e1"] * np.cos(2 * dth) - cd["e2"] * np.sin(2 * dth)
    cd2["e2"] = cd["e1"] * np.sin(2 * dth) + cd["e2"] * np.cos(2 * dth)
    mx, my = cd["mem_x"], cd["mem_y"]
    cd2["mem_x"] = mx * np.cos(dth) - my * np.sin(dth)
    cd2["mem_y"] = mx * np.sin(dth) + my * np.cos(dth)
    cd2["pa_bar_obs"] = (cd["pa_bar_obs"] + 41.0) % 180.0
    cd2["axis_ext_obs"] = (cd["axis_ext_obs"] + 41.0) % 180.0
    r1 = IV.reduce_cluster(cd2)
    law = _flat_law()
    F0, F1 = IV.finish_cluster(r0, law), IV.finish_cluster(r1, law)
    P0 = IV.cluster_phase_features({k: np.array([F0[k]]) for k in F0})
    P1 = IV.cluster_phase_features({k: np.array([F1[k]]) for k in F1})
    worst_inv = max(abs(F0[k] - F1[k]) / max(abs(F0[k]), 1e-3) for k in F0
                    if k not in ("pa_bar", "ax_ext") and not k.startswith(("z_re", "z_im", "z_c"))
                    and np.isfinite(F0[k]) and np.isfinite(F1[k]))
    worst_ph = max(abs(P0[k][0] - P1[k][0]) for k in P0)
    check("T4 cluster reduction is invariant under a catalogue rotation",
          worst_inv < 0.02 and worst_ph < 0.05,
          f"worst relative change {worst_inv:.1e}; worst phase-feature change {worst_ph:.1e}")
    # permute the sources and the members
    rng = np.random.default_rng(0)
    ps, pm = rng.permutation(len(x)), rng.permutation(len(mx))
    cd3 = dict(cd)
    for k in ("src_x", "src_y", "e1", "e2", "w", "z_src_phot"):
        cd3[k] = cd[k][ps]
    for k in ("mem_x", "mem_y", "mem_v", "mem_p", "mem_m_obs"):
        cd3[k] = cd[k][pm]
    F3 = IV.finish_cluster(IV.reduce_cluster(cd3), law)
    worst = max(abs(F0[k] - F3[k]) for k in F0 if np.isfinite(F0[k]) and np.isfinite(F3[k]))
    check("T4 cluster reduction is invariant under source and member permutation",
          worst < 1e-6, f"worst change {worst:.1e}")
    # galaxy: the m=3 phase is stored RELATIVE to the disc axis, so a common
    # rotation of the sky frame (pa_obs + delta, axis_ext_obs + delta) with a
    # transposed pixel grid (an exact 90-degree rotation) leaves every feature
    # unchanged except the raw stored axis
    u3 = P.make_universe("U03_mond_scalar", None, 1.0, 1.0, 9)
    for gal in lib.galaxies:
        gd = P.emit_galaxy(u3, gal, 9)
        r = IV.reduce_galaxy(gd)
        if r is not None:
            break
    F = IV.finish_galaxy(r, law)
    gd2 = dict(gd)
    gd2["v_map"] = np.rot90(gd["v_map"]).copy()
    gd2["v_err"] = np.rot90(gd["v_err"]).copy()
    gd2["I_map"] = np.rot90(gd["I_map"]).copy()
    gd2["mask"] = np.rot90(gd["mask"]).copy()
    gd2["pa_obs"] = gd["pa_obs"] + 90.0
    gd2["axis_ext_obs"] = (gd["axis_ext_obs"] + 90.0) % 180.0
    F2_ = IV.finish_galaxy(IV.reduce_galaxy(gd2), law)
    P0 = IV.galaxy_phase_features({k: np.array([F[k]]) for k in F})
    P1 = IV.galaxy_phase_features({k: np.array([F2_[k]]) for k in F2_})
    worst = max(abs(F[k] - F2_[k]) for k in F if k != "psi_obs" and np.isfinite(F[k]) and np.isfinite(F2_[k]))
    worst_ph = max(abs(P0[k][0] - P1[k][0]) for k in P0)
    check("T4 galaxy reduction is invariant under a 90-degree frame rotation",
          worst < 1e-3 and worst_ph < 1e-3, f"worst change {worst:.1e}; phase features {worst_ph:.1e}")


class _FlatLaw:
    support = (-13.0, -7.0)

    def lognu(self, g):
        return np.zeros_like(np.asarray(g, float))

    def frac_outside(self, g):
        return 0.0


def _flat_law():
    return _FlatLaw()


# ------------------------------------------------ T5 zero points, noise free
def t5_zero_points():
    """A noise-free Newtonian universe must give a vertical/radial boost
    contrast of zero and a flat residual; a noise-free MOND universe the same
    contrast (the law boosts both components alike)."""
    lib = W.get_lib("train")
    out = {}
    for uid in ("U01_baryons_newton", "U03_mond_scalar"):
        u = P.make_universe(uid, None, 1.0, 1.0, 5)
        u.noise_scale = 1e-6
        u.sys_scale = 1e-6
        d1, d0 = [], []
        for gal in lib.galaxies:
            r = IV.reduce_galaxy(P.emit_galaxy(u, gal, 5))
            if r is None:
                continue
            d1.append(r["dz"][1])
            d0.append(r["dz"][0])
        out[uid] = (float(np.mean(d1)), float(np.std(d1)), float(np.mean(d0)), float(np.std(d0)))
    ok = all(abs(v[0]) < 0.03 and v[1] < 0.08 for v in out.values())
    check("T5 noise-free boost-isotropy contrast is zero in U1 and U3",
          ok, "; ".join(f"{k[:3]}: dz(2Rd) {v[0]:+.3f} +- {v[1]:.3f}, dz(1Rd) {v[2]:+.3f} +- {v[3]:.3f}"
                        for k, v in out.items()))
    # and a CDM universe with its halo must be NEGATIVE, by the amount the truth says
    u = P.make_universe("U02_cdm", None, 1.0, 1.0, 5)
    u.noise_scale = 1e-6
    u.sys_scale = 1e-6
    meas, true = [], []
    for gal in lib.galaxies:
        dm = H.draw_galaxy_halo(gal, u.params, np.random.default_rng(3), H.knobs(q_max=0.0))
        gd = P.emit_galaxy(u, gal, 5, dm=dm)
        r = IV.reduce_galaxy(gd)
        if r is None:
            continue
        T = gd["_truth"]
        meas.append(r["dz"][1])
        true.append(np.log10(T["gz_true"][1] / T["gzN"][1]) - np.log10(T["gR_true"][1] / T["gN"][1]))
    m, t = float(np.mean(meas)), float(np.mean(true))
    check("T5 noise-free CDM contrast is negative and tracks the truth",
          m < -0.1 and abs(m - t) < 0.05, f"measured {m:+.3f}, true {t:+.3f}")


# ------------------------------------------------ T6 monopole matching
def t6_monopole_matching():
    """A pure a0 shift must be absorbed by the cross-fitted law: the residual
    features of U3 with a0 x 10^0.15 must equal those of U3 to within the
    change a re-fitted law cannot follow."""
    lib = W.get_lib("train")
    from universes import physics as ph
    gi = np.arange(30)
    ci = np.arange(12)
    feats = {}
    for tag, shift in (("base", 0.0), ("shift", 0.15)):
        u = P.make_universe("U03_mond_scalar", None, 1.0, 1.0, 13)
        u.params["a0"] = u.params["a0"] * 10 ** shift
        gal_d = [P.emit_galaxy(u, lib.galaxies[i], 13) for i in gi]
        clu_d = [P.emit_cluster(u, lib.geoms[i], 13) for i in ci]
        A = IV.analyse_corpus(gal_d, clu_d, None, split_seed=1)
        feats[tag] = A
    def col(A, grp, k):
        return np.array([F[k] for F in A[grp]], float)
    dres = np.nanmean(col(feats["shift"], "gal", "res_mean") - col(feats["base"], "gal", "res_mean"))
    dly = np.nanmean(col(feats["shift"], "gal", "ly_5") - col(feats["base"], "gal", "ly_5"))
    dwl = np.nanmean(col(feats["shift"], "clu", "wl_lA") - col(feats["base"], "clu", "wl_lA"))
    dt = np.nanmean(col(feats["shift"], "clu", "t_lA") - col(feats["base"], "clu", "t_lA"))
    check("T6 an a0 shift moves the RAW boost but not the RESIDUALS",
          abs(dly) > 0.03 and abs(dres) < 0.3 * abs(dly) and abs(dt) < 0.3 * abs(dly),
          f"raw galaxy boost moves {dly:+.3f}; residual moves {dres:+.4f}; cluster kT residual {dt:+.4f}; WL {dwl:+.4f}")


# ------------------------------------------------ T7 guards
def t7_guard():
    import guard
    from universes import provenance as pv
    led = guard.start()
    ok_sealed = ok_reserve = ok_foreign = False
    try:
        io.open("C:/data/KiDS_dr4_shear.fits")
    except pv.SealedHoldoutTouched:
        ok_sealed = True
    except Exception:                                          # noqa: BLE001
        pass
    try:
        io.open("/some/path/x-gap_profiles.csv")
    except pv.SealedHoldoutTouched:
        ok_reserve = True
    except Exception:                                          # noqa: BLE001
        pass
    try:
        io.open("C:/Users/henry/dev/gravity-discovery-program.md")
    except pv.ForeignReadError:
        ok_foreign = True
    except Exception:                                          # noqa: BLE001
        pass
    guard.stop()
    check("T7 a SEALED path raises before it can be read", ok_sealed, "KiDS token guarded")
    check("T7 a CONFIRMATION-RESERVE path raises", ok_reserve, "X-GAP token guarded")
    check("T7 a foreign read outside the lane raises", ok_foreign, "lane-root guard active")


# ------------------------------------------------ T8 generator agreement
def t8_generator_agreement():
    """The two generators must agree in SIGN and roughly in size on the two
    non-directional invariants: the boost-isotropy contrast at 2 R_d and the
    galaxy residual scatter, CDM minus class."""
    lib = W.get_lib("train")
    vals = {}
    for tag, uid in (("U2", "U02_cdm"), ("U3", "U03_mond_scalar")):
        u = P.make_universe(uid, None, 1.0, 1.0, 21)
        gd = []
        for gal in lib.galaxies[:30]:
            dm = H.draw_galaxy_halo(gal, u.params, np.random.default_rng(4), H.knobs()) if uid == "U02_cdm" else None
            gd.append(P.emit_galaxy(u, gal, 21, dm=dm))
        cd = [P.emit_cluster(u, lib.geoms[i], 21) for i in range(4)] if uid != "U02_cdm" else \
             [P.emit_cluster(u, lib.geoms[i], 21, dm=H.draw_cluster_halo(lib.geoms[i].clu, u.params, np.random.default_rng(i), H.knobs())) for i in range(4)]
        A = IV.analyse_corpus(gd, cd, None, split_seed=2)
        vals[tag] = (float(np.nanmean([F["dz_1"] for F in A["gal"]])), A["corpus"]["rar_scatter_oof"])
    g1 = (vals["U2"][0] - vals["U3"][0], vals["U2"][1] - vals["U3"][1])
    seed, out = F2.run_set2((21, [dict(tag="C", kind="class"), dict(tag="D", kind="cdm", halo={})], {}))
    g2 = (float(np.nanmean(out["D"]["gal"]["dz_1"]) - np.nanmean(out["C"]["gal"]["dz_1"])),
          float(out["D"]["corpus"]["rar_scatter_oof"] - out["C"]["corpus"]["rar_scatter_oof"]))
    check("T8 both generators: CDM minus class boost-isotropy contrast is NEGATIVE",
          g1[0] < -0.05 and g2[0] < -0.05, f"generator 1 {g1[0]:+.3f}, generator 2 {g2[0]:+.3f}")
    check("T8 both generators: CDM minus class residual scatter is POSITIVE",
          g1[1] > 0 and g2[1] > 0, f"generator 1 {g1[1]:+.3f}, generator 2 {g2[1]:+.3f}")


# ------------------------------------------------ T9 halo counterfactual machinery
def t9_counterfactuals():
    """rescale_halo must move the enclosed mass by the requested amount and
    leave the shape; the dark-disc vertical field must reduce to the spherical
    one at f_dd = 0 and exceed it at f_dd = 1."""
    lib = W.get_lib("train")
    gal = lib.galaxies[3]
    p = {"shmr_scatter": 0.16, "c_norm": 5.0, "fbar": 0.155}
    dm = H.draw_galaxy_halo(gal, p, np.random.default_rng(1), H.knobs())
    dm2 = H.rescale_halo(dm, dlogM=0.3)
    R = np.array([2.0, 5.0, 10.0]) * gal.Rd
    ratio = dm2["M200"] / dm["M200"]
    r_in = dm2["Mdm"](R[0]) / dm["Mdm"](R[0])
    check("T9 halo mass counterfactual moves M200 by 10^0.3", abs(np.log10(ratio) - 0.3) < 1e-9,
          f"M200 ratio 10^{np.log10(ratio):.3f}; enclosed at 2Rd x{r_in:.2f}")
    dm0 = dict(dm, f_dd=0.0, q_h=1.0)
    dm1 = dict(dm, f_dd=1.0, q_h=1.0)
    g0 = H.halo_vertical_field(R, gal.hz, dm0)
    g1 = H.halo_vertical_field(R, gal.hz, dm1)
    sph = G * dm["Mdm"](R) * gal.hz / R ** 3
    check("T9 dark-disc vertical field: spherical at f_dd = 0, larger at f_dd = 1",
          np.allclose(g0, sph) and np.all(g1 > 3 * g0), f"g_z(f_dd=1)/g_z(f_dd=0) = {np.round(g1 / g0, 1)}")


def main():
    print("=" * 74)
    print("extraction lane tests")
    print("=" * 74)
    for fn in (t1_pairing, t2_projection, t3_hse_jeans, t4_invariance, t5_zero_points,
               t6_monopole_matching, t7_guard, t8_generator_agreement, t9_counterfactuals):
        try:
            fn()
        except Exception as e:                                 # noqa: BLE001
            check(f"{fn.__name__} raised", False, repr(e)[:200])
    n_pass = sum(r["passed"] for r in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} tests passed")
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    with open(os.path.join(HERE, "results", "T_tests.json"), "w") as f:
        json.dump(dict(n_pass=n_pass, n_total=len(RESULTS), tests=RESULTS), f, indent=1)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
