"""test_bridge.py -- the bridge lane's validation suite.

Every test names what it checks and, where the check found a bug during
development, what the bug was.  Bugs this suite caught, in order found:

  B1  the lensing estimator's numerical floor was 1e-3 in kappa -- the same
      size as the P signal at eps = 0.003 -- because the endpoints' shear was
      computed by FFT on a finite field and the model profile was
      extrapolated as 1/r (a point mass falls as 1/r^2).  Fixed: analytic
      endpoint shear, measured tail slope.  (T6 pins the floor at < 3e-5.)
  B2  the QUMOND phantom density is NaN on the grid point that lands on the
      saddle (g_N = 0), and one NaN pixel poisoned every map through the FFT.
      Fixed: cell-averaged sub-sampling of the integrable singularity.  (T10)
  B3  the tensor l=2 density read the ODE's Dirichlet boundary node when a
      grid point landed on a body centre: ONE cell carried 1e17 Msun.  Caught
      by the sphere-integral test.  (T9)
  B4  members.py took the carrier acceleration outward-positive against an
      inward-positive Newtonian magnitude and reported BL's confinement (x7.6)
      as expulsion (x-5.3).  Caught by the BL comparison.  (T11)

    python test_bridge.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import guard                                    # noqa: E402
import scene as S                               # noqa: E402
import lensing as L                             # noqa: E402
import estimator as E                           # noqa: E402
import compiler as C                            # noqa: E402

RES = os.path.join(HERE, "results")
RESULTS: dict = {}
KPC, MPC, MSUN, G = S.KPC, S.MPC, S.MSUN, S.G
BL_P_MIDPOINT = 0.1316029            # path_results.json, bridge P at the midpoint


def _rec(name, **kw):
    RESULTS[name] = kw
    return kw


def _small_fields(sc=None, dx=80.0 * KPC, n_dir=1000):
    sc = sc or S.TwoBody()
    grid = S.default_grid(sc.D, x_half=1.0, R_max_frac=0.6, dx=dx)
    return S.PathFields.build(sc, grid, n_dir)


# --------------------------------------------------------------- columns, P
def test_T1_half_column_matches_quadrature_and_BL():
    import path_family as PF
    comp = C.Plummer(3e14 * MSUN, 400 * KPC, (-2 * MPC, 0, 0))
    rng = np.random.default_rng(1)
    z = rng.normal(size=(8, 3)) * 1.5 * MPC
    n = rng.normal(size=(8, 3))
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    mine = S.half_column(comp, z, n)
    bl = PF.Scene([comp]).column(z, n)
    s = np.geomspace(1e-5 * KPC, 3e5 * KPC, 200000)
    quad = np.array([np.trapezoid(comp.rho(z[i][None, :] + s[:, None] * n[i][None, :]), s)
                     for i in range(8)])
    e_q = float(np.max(np.abs(mine - quad) / quad))
    e_bl = float(np.max(np.abs(mine - bl) / bl))
    # the far-tail series branch
    zf = np.array([[-2 * MPC + 30 * MPC, 0.1 * KPC, 0]])
    nf = np.array([[1.0, 0, 0]])
    tail = float(S.half_column(comp, zf, nf)[0])
    tail_q = float(np.trapezoid(comp.rho(zf + s[:, None] * nf), s))
    _rec("T1_column", max_rel_err_vs_quadrature=e_q, max_rel_err_vs_BL=e_bl,
         far_tail_rel_err=abs(tail - tail_q) / tail_q)
    assert e_q < 1e-6 and e_bl < 1e-12, (e_q, e_bl)
    assert abs(tail - tail_q) / tail_q < 1e-3


def test_T2_P_at_the_midpoint_reproduces_BL():
    sc = S.TwoBody()
    P = float(S.P_total(S.pairwise_P(sc, np.zeros((1, 3)), 4000))[0])
    _rec("T2_P_midpoint", P=P, BL=BL_P_MIDPOINT, rel=abs(P - BL_P_MIDPOINT) / BL_P_MIDPOINT)
    assert abs(P - BL_P_MIDPOINT) / BL_P_MIDPOINT < 1e-4


def test_T3_direction_quadrature_converged():
    sc = S.TwoBody()
    pts = np.array([[0, 0, 0], [0, 0.5 * MPC, 0], [1.0 * MPC, 0, 0], [-1.7 * MPC, 0, 0]])
    P2 = S.P_total(S.pairwise_P(sc, pts, 2000))
    P8 = S.P_total(S.pairwise_P(sc, pts, 8000))
    rel = float(np.max(np.abs(P2 - P8) / P8))
    _rec("T3_ndir", max_rel_2000_vs_8000=rel)
    assert rel < 1e-3, rel


def test_T4_pairwise_decomposition_is_exactly_bilinear():
    sc = S.TwoBody(rho_f=1e-25)
    pts = np.array([[0, 0.2 * MPC, 0], [0.8 * MPC, 0.4 * MPC, 0]])
    Pgh = S.pairwise_P(sc, pts, 800)
    scale = {"A": 1.7, "B": 0.6, "F": 2.5}
    sc2 = S.TwoBody(MA=1.7 * sc.MA, MB=0.6 * sc.MB, rho_f=2.5e-25)
    direct = S.P_total(S.pairwise_P(sc2, pts, 800))
    rel = float(np.max(np.abs(S.P_total(Pgh, scale) - direct) / direct))
    _rec("T4_bilinear", max_rel=rel)
    assert rel < 1e-12, rel


# ----------------------------------------------------------- compensation
def test_T5_carrier_bridge_is_compensated_and_newton_is_zero():
    pf = _small_fields()
    br = S.rho_eff_bridge("P", pf, 0.3, S.RHO_STAR_FID, include_endpoint_cross=False)
    w = pf.grid.volume_weights()
    m = (np.abs(pf.grid.xs)[:, None] < 0.95 * pf.grid.xs[-1]) & \
        (pf.grid.Rs[None, :] < 0.95 * pf.grid.Rs[-1])
    net = float((br["specific"] * w * m).sum())
    tabs = float((np.abs(br["specific"]) * w * m).sum())
    nb = S.rho_eff_bridge("newton", pf)
    _rec("T5_compensation", net_over_abs=net / tabs, abs_Msun=tabs / MSUN,
         newton_specific_max=float(np.abs(nb["specific"]).max()))
    assert abs(net / tabs) < 2e-2, net / tabs
    assert np.abs(nb["specific"]).max() == 0.0


def test_T6_estimator_floor_on_a_newtonian_pair():
    """B1: the floor was 1e-3 with FFT endpoints and a 1/r tail."""
    pf = _small_fields()
    sky = L.default_sky(pf.scene.D)
    sv = L.Survey()
    maps = E.law_kappa_maps("newton", pf, sky, sv.sigma_crit_eff)
    geom = E.Geometry(xA=maps["xA"], xB=maps["xB"])
    res = E.bridge_statistics(maps["g1"], maps["g2"], sky, geom, sv.sigma_crit_eff)
    _rec("T6_floor", **{k: float(res[k]) for k in E.STAT_KEYS if k != "A_mf"})
    assert abs(res["Q"]) < 3e-5 and abs(res["DQ"]) < 3e-5, (res["Q"], res["DQ"])


# ------------------------------------------------------------------ lensing
def test_T7_shear_round_trip_and_KS_inversion():
    sky = L.default_sky(4 * MPC)
    sv = L.Survey()
    M, a = 3e14 * MSUN, 400 * KPC
    Sig = L.plummer_sigma(M, a, sky, 0.0)
    kap = Sig / sv.sigma_crit_eff
    g1, g2 = L.shear_from_kappa(kap, sky.pix, pad=3)
    X1, X2 = sky.mesh()
    gt, gx = L.tangential_shear(g1, g2, X1, X2, 0.0)
    R = np.sqrt(X1 ** 2 + X2 ** 2)
    gt_an = (M / (math.pi * (R ** 2 + a ** 2)) - Sig) / sv.sigma_crit_eff
    sel = (R > 0.3 * MPC) & (R < 2.0 * MPC)
    e1 = float(np.max(np.abs(gt[sel] - gt_an[sel])) / gt_an[sel].max())
    kb = L.kappa_from_shear(g1, g2, sky.pix, pad=3)
    e2 = float(np.max(np.abs((kb - kb.mean()) - (kap - kap.mean()))) / kap.max())
    # the analytic endpoint shear must agree with the FFT one where both are valid
    rg = np.geomspace(2e-2 * a, 40 * a, 300)
    rho = S.Plummer(M, a).rho(np.stack([rg, 0 * rg, 0 * rg], -1))
    a1, a2, ak = L.radial_shear_map(rg, rho, sky, 0.0, sv.sigma_crit_eff)
    e3 = float(np.max(np.abs(a1[sel] - g1[sel])) / np.abs(g1[sel]).max())
    _rec("T7_shear", fft_vs_analytic_gt=e1, KS_round_trip=e2, analytic_map_vs_fft=e3,
         cross_shear_max=float(np.abs(gx[sel]).max()))
    assert e1 < 1e-2 and e2 < 1e-2 and e3 < 2e-2, (e1, e2, e3)


def test_T8_estimator_is_linear():
    pf = _small_fields()
    sky = L.default_sky(pf.scene.D)
    sv = L.Survey()
    scr = sv.sigma_crit_eff
    m = E.law_kappa_maps("P", pf, sky, scr, eps=0.03, rho_star=S.RHO_STAR_FID)
    geom = E.Geometry(xA=m["xA"], xB=m["xB"])
    rng = np.random.default_rng(3)
    nz = L.noise_maps(sky, sv, rng)
    st = lambda a, b: E.stat_vector(E.bridge_statistics(a, b, sky, geom, scr))[:-1]  # noqa: E731
    full, ends = st(m["g1"], m["g2"]), st(m["g1_ends"], m["g2_ends"])
    fulln = st(m["g1"] + nz["e1"], m["g2"] + nz["e2"])
    endsn = st(m["g1_ends"] + nz["e1"], m["g2_ends"] + nz["e2"])
    # the exact identity under linearity: stats(full+n) - stats(ends+n) =
    # stats(full) - stats(ends).  (The first version compared against
    # stats(full) alone and mistook the deterministic 1e-5 floor of
    # stats(ends) for nonlinearity on the smallest statistics.)
    rel = float(np.max(np.abs((fulln - endsn) - (full - ends)) / np.abs(full - ends)))
    hom = float(np.max(np.abs(st(2 * m["g1"] - m["g1_ends"], 2 * m["g2"] - m["g2_ends"])
                              - (2 * full - ends)) / np.abs(full)))
    _rec("T8_linearity", max_rel_identity=rel, max_rel_homogeneity=hom)
    assert rel < 1e-8 and hom < 1e-8, (rel, hom)


# --------------------------------------------------------------- competitors
def test_T9_tensor_l2_term_has_zero_net_mass():
    """B3: a grid point on a body centre read the ODE boundary node."""
    pf = _small_fields()
    comp = pf.scene.groups["A"][0]
    pts = pf.grid.points()
    rho2 = S.tensor_l2_density(comp, pts, np.array([1.0, 0, 0]), 1.0).reshape(pf.grid.shape)
    w = pf.grid.volume_weights()
    r = np.linalg.norm(pts - comp.c, axis=1).reshape(pf.grid.shape)
    m = r < 1.0 * MPC
    net = float((rho2 * w * m).sum())
    tabs = float((np.abs(rho2) * w * m).sum())
    ic = int(np.argmin(np.abs(pf.grid.xs - comp.c[0])))
    _rec("T9_tensor_l2", net_over_abs=net / tabs, abs_Msun=tabs / MSUN,
         centre_cell=float(rho2[ic, 0]))
    assert abs(net / tabs) < 3e-2, net / tabs
    assert abs(rho2[ic, 0]) < 1e-20, rho2[ic, 0]


def test_T10_qumond_phantom_closed_form_and_finite_saddle():
    """B2: the saddle point was NaN and poisoned every map."""
    M, a = 3e14 * MSUN, 400 * KPC
    one = S.TwoBody(M, M, a, 10 * a)
    one.groups = {"A": [S.Plummer(M, a, (0.0, 0.0, 0.0))], "B": [], "F": []}
    r = np.array([0.5, 1.0, 2.0, 5.0]) * a
    pts = np.stack([r, 0 * r, 0 * r], -1)
    ph = S.phantom_density(one, pts, "rar")
    # spherical reference: rho_ph = (1/4 pi G r^2) d/dr [r^2 (nu - 1) g_N]
    rg = np.geomspace(0.05 * a, 20 * a, 4000)
    gN = G * S.Plummer(M, a).M_enc(rg) / rg ** 2
    f = rg ** 2 * (S.nu_rar(gN / S.A0) - 1.0) * gN
    ref = np.gradient(f, rg) / (4 * math.pi * G * rg ** 2)
    rel = float(np.max(np.abs(ph - np.interp(r, rg, ref)) / np.abs(np.interp(r, rg, ref))))
    pf = _small_fields()
    br = S.rho_eff_bridge("qumond", pf)
    _rec("T10_qumond", max_rel_vs_spherical_reference=rel,
         all_finite=bool(np.all(np.isfinite(br["specific"]))),
         saddle_value=float(br["specific"][int(np.argmin(np.abs(pf.grid.xs))), 0]))
    assert rel < 2e-3, rel
    assert np.all(np.isfinite(br["specific"]))


def test_T11_member_confinement_sign_and_magnitude_match_BL():
    """B4: the inward/outward convention."""
    import members as Mb
    res = Mb.member_scan(n_dir=6000)
    k = res["rows"]["1e-24"]["k_per_eps"][20]
    fac = 1.0 + 0.3 * k
    _rec("T11_members", k_per_eps_20kpc=k, factor_eps0p3=fac, BL=Mb.BL_CARRIER_20KPC)
    assert k > 0, "the carrier term must CONFINE (BL: x7.6 at 20 kpc)"
    assert abs(fac / Mb.BL_CARRIER_20KPC - 1.0) < 0.15, fac


# ------------------------------------------------------------------ guard
def test_T12_guard_raises_on_sealed_reserve_and_foreign_paths():
    from universes import provenance as pv
    guard.arm()
    hits = {}
    for tag, path, exc in (("sealed", "C:/data/KiDS_dr4_shear.fits", pv.SealedHoldoutTouched),
                           ("reserve", "/some/path/clogs_profiles.csv", pv.SealedHoldoutTouched),
                           ("reserve2", "/x/spt_cluster_cat.fits", pv.SealedHoldoutTouched),
                           ("foreign", os.path.join(os.path.dirname(HERE), "synthesis",
                                                    "path_results.json"), pv.ForeignReadError)):
        try:
            io.open(path)
            hits[tag] = False
        except exc:
            hits[tag] = True
        except Exception:                                      # noqa: BLE001
            hits[tag] = False
    _rec("T12_guard", **hits)
    assert all(hits.values()), hits


# ------------------------------------------------------------- compiler
def test_T13_two_body_probe_is_consulted_only_when_declared():
    P = C.probes()
    assert C.GATE1_BRIDGE_PROBE in P and P[C.GATE1_BRIDGE_PROBE].pts.shape == (15, 3)
    base = C.Candidate("n", base="newton", struct="none", inv="one", form="off", A=0.0)
    r = C.check(base, cheap=True)
    assert r["_verdict"] == "ADMIT"
    # a Yukawa below threshold: non-identifiable, with and without a NULL bridge field
    yk = C.external_controls()["XCS_yukawa_subthreshold"][0]
    r0 = C.check(yk, cheap=True)
    assert r0["_taxonomy"]["primary"] == "non_identifiable_on_this_bench"
    pb = P[C.GATE1_BRIDGE_PROBE]
    yk_null = C.Candidate(**{**yk.__dict__, "name": "yk_nullbridge",
                             "bridge_field": lambda pts: 1.05 * pb.gN})
    r1 = C.check(yk_null, cheap=True)
    assert r1["_taxonomy"]["primary"] == "non_identifiable_on_this_bench", r1["_taxonomy"]
    assert r1["gate1_constant_K"][1]["two_body_probe_consulted"] is True
    assert "d_two_body_probe" not in r1["gate1_constant_K"][1]["escapes"]
    # a field that is NOT a stretch of g_N on the two-body probe: escape (d)
    yk_br = C.Candidate(**{**yk.__dict__, "name": "yk_bridge",
                           "bridge_field": lambda pts: pb.gN * (1.0 + 0.6 * np.exp(
                               -(pts[:, 0] ** 2) / (0.5 * MPC) ** 2))})
    r2 = C.check(yk_br, cheap=True)
    _rec("T13_probe", null_bridge_resid=r1["gate1_constant_K"][1]["two_body_probe_resid_dex"],
         bridge_resid=r2["gate1_constant_K"][1]["two_body_probe_resid_dex"],
         escapes=r2["gate1_constant_K"][1]["escapes"], verdict=r2["_verdict"])
    assert "d_two_body_probe" in r2["gate1_constant_K"][1]["escapes"], r2["gate1_constant_K"][1]
    assert r2["_verdict"] == "ADMIT"


def test_T14_prior_verdicts_unchanged():
    p = os.path.join(RES, "baseline_compare.json")
    if not os.path.exists(p):
        _rec("T14_baseline", skipped="run baseline.py --snapshot (pre-patch) and --compare")
        return
    d = json.load(open(p, encoding="utf-8"))
    _rec("T14_baseline", n_compared=d["n_compared"], n_changed=d["n_changed"],
         controls=f"{d['n_agree']}/{d['n_external_controls']}")
    assert d["n_changed"] == 0 and d["external_controls_all_agree"], d["changed"]


# ------------------------------------------------------------ projections
def test_T15_elliptical_projection_and_abel():
    import cdm_attack as A
    sky = L.default_sky(4 * MPC, pix=40 * KPC)
    M, a = 3e14 * MSUN, 400 * KPC
    for q, th in ((1.0, 0.0), (0.6, 30.0)):
        Sig = A.elliptical_plummer_sigma(M, a, q, th, sky, 0.0)
        Mtot = float(Sig.sum() * sky.pix ** 2)
        assert abs(Mtot / M - 1.0) < 0.05, (q, Mtot / M)     # the field truncates at ~4 Mpc
    circ = A.elliptical_plummer_sigma(M, a, 1.0, 0.0, sky, 0.0)
    assert np.allclose(circ, L.plummer_sigma(M, a, sky, 0.0))
    rg = np.geomspace(2e-2 * a, 40 * a, 300)
    rho = S.Plummer(M, a).rho(np.stack([rg, 0 * rg, 0 * rg], -1))
    R = np.array([0.5, 1.0, 2.0]) * a
    Sab = S.abel_project(rg, rho, R)
    San = M * a ** 2 / (math.pi * (R ** 2 + a ** 2) ** 2)
    rel = float(np.max(np.abs(Sab - San) / San))
    _rec("T15_projection", abel_rel_err=rel)
    assert rel < 2e-3, rel


def test_T16_certificate_ids_are_typed():
    p = os.path.join(RES, "certificate_bridge.json")
    if not os.path.exists(p):
        _rec("T16_certificate", skipped="run certify.py")
        return
    d = json.load(open(p, encoding="utf-8"))
    ids = list(d["certificates"].keys())
    import re
    ok = all(re.fullmatch(r"[A-Z]+(\.[A-Z0-9_]+)+\.\d{3}", i) for i in ids)
    _rec("T16_certificate", ids=ids, typed=ok)
    assert ok and ids, ids


# ------------------------------------------------------------------- runner
def main():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    t0 = time.perf_counter()
    passed, failed = [], []
    for name, fn in tests:
        s = time.perf_counter()
        try:
            fn()
            passed.append(name)
            print(f"PASS  {name:<62} {time.perf_counter() - s:6.1f}s")
        except AssertionError as e:
            failed.append((name, str(e)[:400]))
            print(f"FAIL  {name:<62} {time.perf_counter() - s:6.1f}s\n      {str(e)[:400]}")
        except Exception as e:                                # noqa: BLE001
            failed.append((name, f"{type(e).__name__}: {e}"[:400]))
            print(f"ERROR {name:<62} {time.perf_counter() - s:6.1f}s\n      {type(e).__name__}: {e}")
    wall = time.perf_counter() - t0
    print(f"\n{len(passed)} passed, {len(failed)} failed in {wall:.1f}s")
    payload = dict(lane="work/wellnet-2026-09/bridge",
                   generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   n_tests=len(tests), n_passed=len(passed), n_failed=len(failed),
                   failures=failed, wall_seconds=wall, results=RESULTS,
                   bugs_caught=["B1 estimator floor (FFT endpoints, 1/r tail)",
                                "B2 QUMOND saddle NaN poisoning every map",
                                "B3 tensor l=2 boundary-node spike (1e17 Msun in one cell)",
                                "B4 members.py inward/outward sign convention"],
                   provenance=guard.summary())
    with open(os.path.join(RES, "tests.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, indent=1, default=float)
    return 1 if failed else 0


if __name__ == "__main__":
    guard.arm()
    sys.exit(main())
