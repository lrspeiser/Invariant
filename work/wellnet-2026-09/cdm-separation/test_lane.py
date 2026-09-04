"""test_lane.py -- tests for this lane's estimator, forward model and guards.

Every lane in this programme has found a real bug in its own first
implementation.  This one found a SIGN ERROR (see forward.f_halo) by comparing
two independently written forward models, and a covariance mis-calibration in
the first version of the studentisation.  The tests that caught them are T6 and
T3 below and they run every time.

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

import estimators as E      # noqa: E402
import forward as F         # noqa: E402
import pipeline as PL       # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(dict(test=name, passed=bool(ok), detail=detail))
    print(f"  {'PASS' if ok else 'FAIL':<5} {name:<46} {detail}")
    return ok


# ---------------------------------------------------------------- T1, T2, T3
def synth_catalogue(rng, phi0_deg, amp, n=20000, noise=0.26, R500=1000.0):
    """A catalogue with a KNOWN pure m=2 shear pattern and no monopole."""
    rr = np.sqrt(rng.uniform((0.15 * R500) ** 2, (2.2 * R500) ** 2, n))
    pp = rng.uniform(0, 2 * np.pi, n)
    d = 2 * (pp - np.deg2rad(phi0_deg))
    gt = amp * np.cos(d)
    gx = amp * np.sin(d)
    g1 = -(gt * np.cos(2 * pp) - gx * np.sin(2 * pp))
    g2 = -(gt * np.sin(2 * pp) + gx * np.cos(2 * pp))
    return dict(name="T", z=0.3, R500=R500,
                src_x=rr * np.cos(pp), src_y=rr * np.sin(pp),
                e1=g1 + rng.normal(0, noise, n), e2=g2 + rng.normal(0, noise, n),
                w=np.full(n, 1.0), z_src_phot=np.full(n, 1.2),
                pa_bar_obs=0.0, axis_ext_obs=phi0_deg, ell_bar_obs=0.2)


def t1_recovery():
    rng = np.random.default_rng(1)
    errs, amps = [], []
    truth = 0.02
    for k in range(120):
        p0 = float(rng.uniform(0, 180))
        cd = synth_catalogue(rng, p0, truth)
        q = E.cluster_quadrupole(cd, F.sigma_crit)
        Z, C, n = E.combine_bins(q)
        pa = np.rad2deg(0.5 * np.angle(Z)) % 180.0
        d = abs((pa - p0) % 180.0)
        errs.append(min(d, 180 - d))
        amps.append(abs(Z))
    med = float(np.median(errs))
    bias = float(np.mean(amps) / truth - 1.0)
    check("T1 phase recovered on an injected quadrupole", med < 3.0,
          f"median phase error {med:.2f} deg")
    check("T1 amplitude recovered within 10%", abs(bias) < 0.10,
          f"amplitude bias {bias:+.3f}")


def t2_null_debias():
    """The debiased power Q2 must have mean ZERO on a pure-monopole field."""
    rng = np.random.default_rng(2)
    q2, raw = [], []
    for k in range(150):
        cd = synth_catalogue(rng, 0.0, 0.0)
        q = E.cluster_quadrupole(cd, F.sigma_crit)
        for b in q["bins"]:
            if b["ok"]:
                q2.append(b["Q2"])
                raw.append(abs(b["Z"]) ** 2)
    q2 = np.array(q2)
    raw = np.array(raw)
    z = float(np.mean(q2) / (np.std(q2) / np.sqrt(len(q2))))
    check("T2 debiased quadrupole power is null-centred", abs(z) < 3.0,
          f"mean/sem = {z:+.2f} (undebiased would be "
          f"{np.mean(raw) / (np.std(q2) / np.sqrt(len(q2))):+.1f})")


def t3_covariance():
    """chi^2 = (Z-Ztrue)' C^-1 (Z-Ztrue) must average 2 for a 2-vector."""
    rng = np.random.default_rng(3)
    chi = []
    truth, p0 = 0.02, 37.0
    a = np.deg2rad(2 * p0)
    for k in range(300):
        cd = synth_catalogue(rng, p0, truth, n=6000)
        q = E.cluster_quadrupole(cd, F.sigma_crit)
        Z, C, n = E.combine_bins(q)
        # the injected pattern has gamma_t = gamma_x = amp, so Z_true has
        # modulus `truth` (the two channels agree by construction here)
        Zt = truth * np.array([np.cos(a), np.sin(a)])
        d = np.array([Z.real, Z.imag]) - Zt
        chi.append(float(d @ np.linalg.inv(C) @ d))
    m = float(np.mean(chi))
    check("T3 covariance is calibrated (E[chi2] = 2)", 1.4 < m < 2.8,
          f"mean chi2 = {m:.2f}")


def t4_equivariance():
    """Rotating the catalogue by delta must rotate arg(Z) by 2 delta exactly."""
    rng = np.random.default_rng(4)
    cd = synth_catalogue(rng, 20.0, 0.03, n=30000)
    q = E.cluster_quadrupole(cd, F.sigma_crit)
    Z0, _, _ = E.combine_bins(q)
    dth = np.deg2rad(33.0)
    x, y = cd["src_x"].copy(), cd["src_y"].copy()
    cd2 = dict(cd)
    cd2["src_x"] = x * np.cos(dth) - y * np.sin(dth)
    cd2["src_y"] = x * np.sin(dth) + y * np.cos(dth)
    e1, e2 = cd["e1"], cd["e2"]
    cd2["e1"] = e1 * np.cos(2 * dth) - e2 * np.sin(2 * dth)
    cd2["e2"] = e1 * np.sin(2 * dth) + e2 * np.cos(2 * dth)
    q2 = E.cluster_quadrupole(cd2, F.sigma_crit)
    Z1, _, _ = E.combine_bins(q2)
    got = np.rad2deg(np.angle(Z1 / Z0)) % 360.0
    want = (2 * 33.0) % 360.0
    d = min(abs(got - want), 360 - abs(got - want))
    check("T4 estimator is rotationally equivariant", d < 2.0,
          f"phase rotated by {got:.2f} deg, expected {want:.2f}")


# --------------------------------------------------------------------- T5
def t5_forward_internal():
    Rg = np.geomspace(5, 6000, 4000)
    S, Sb, rs, r200 = F.nfw_sigma(Rg, 1e15, 4.5)
    num = np.concatenate(([0.0], np.cumsum(
        0.5 * (S[1:] * Rg[1:] + S[:-1] * Rg[:-1]) * np.diff(Rg)))) * 2 * np.pi
    Sb_num = num / (np.pi * Rg ** 2)
    i = np.searchsorted(Rg, [200, 800, 2500])
    r = float(np.max(np.abs(Sb[i] / Sb_num[i] - 1.0)))
    check("T5 analytic NFW Sigmabar matches direct integration", r < 5e-3,
          f"max relative error {r:.2e}")
    kap0 = S / 1e10
    f = F.f_tensor(Rg, kap0, 0.5, 300.0)
    psi = F.quad_potential(Rg, f)
    d1 = np.gradient(psi, Rg)
    d2 = np.gradient(d1, Rg)
    lhs = 0.5 * (d2 + d1 / Rg - 4 * psi / Rg ** 2)
    m = (Rg > 30) & (Rg < 3000)
    e = float(np.max(np.abs(lhs[m] - f[m])) / np.max(np.abs(f[m])))
    check("T5 m=2 Green's function solves its own ODE", e < 1e-4,
          f"max relative residual {e:.2e}")


def t6_sign_agreement():
    """THE TEST THAT FOUND THE BUG.

    A collisionless halo whose major axis lies near the BARYON major axis must
    give S_bar > 0 in BOTH forward models.  The first version of forward.f_halo
    put the minor axis where the major axis belongs and returned S_bar < 0,
    disagreeing in sign with Run BF's generator for the same physical universe.
    """
    import worker as W
    from universes import corpus as cp
    from universes import generate as gn
    from universes import physics as ph
    lib = gn.get_lib()
    vals_bf = []
    for k in range(25):
        rng = np.random.default_rng(600 + k)
        u = ph.draw_universe("U02_cdm", rng)
        C = cp.draw_corpus(u, lib, rng, n_gal=1, n_clu=12, n_sn=5)
        vals_bf.append(PL.statistics(PL.cluster_rows(C, ph.sigma_crit))["S_bar"])
    vals_fw = []
    for k in range(25):
        rng = np.random.default_rng(700 + k)
        cl = F.corpus("halo", rng, n_clu=12, e_halo=0.45, mis_deg=22.0)
        vals_fw.append(PL.statistics(PL.cluster_rows(cl, F.sigma_crit))["S_bar"])
    a, b = float(np.mean(vals_bf)), float(np.mean(vals_fw))
    check("T6 both forward models agree in SIGN on S_bar", a > 2 and b > 2,
          f"BF generator {a:+.2f}, independent model {b:+.2f}")
    # and on the sign of the tensor statistic
    vt_bf, vt_fw = [], []
    for k in range(25):
        rng = np.random.default_rng(800 + k)
        u = ph.draw_universe("U05_tensor_axis", rng, knob=1.0)
        C = cp.draw_corpus(u, lib, rng, n_gal=1, n_clu=12, n_sn=5)
        vt_bf.append(PL.statistics(PL.cluster_rows(C, ph.sigma_crit))["S_ext"])
        rng = np.random.default_rng(900 + k)
        cl = F.corpus("tensor", rng, n_clu=12, A_tensor=0.4)
        vt_fw.append(PL.statistics(PL.cluster_rows(cl, F.sigma_crit))["S_ext"])
    a, b = float(np.mean(vt_bf)), float(np.mean(vt_fw))
    check("T6 both forward models agree in SIGN on S_ext", a > 1 and b > 1,
          f"BF generator {a:+.2f}, independent model {b:+.2f}")


def t7_galaxy():
    """The galaxy m=3 estimator: phase and amplitude.

    The injected modulation is radius dependent,
    q(R) = q_amp u/(1+u) with u = (R/2R_d)^2, so the amplitude the estimator
    recovers is the ring-weighted mean of q(R) over 1-5 R_d, about 0.5 q_amp,
    not q_amp itself.  The test therefore checks the SLOPE, not the value.
    """
    rng = np.random.default_rng(7)
    errs = []
    for k in range(120):
        gd = F.emit_galaxy(rng, "tensor", q_amp=0.6)
        r = E.galaxy_m3(gd)
        if r is None:
            continue
        W_, Cw = r
        psi_hat = 0.5 * np.rad2deg(np.angle(complex(W_.real, W_.imag)))
        psi_true = (gd["axis_ext_obs"] - gd["pa_obs"])
        d = abs((psi_hat - psi_true) % 180.0)
        errs.append(min(d, 180 - d))
    med = float(np.median(errs))
    check("T7 galaxy m=3 phase recovered", med < 18.0,
          f"median |psi_hat - psi_obs| = {med:.1f} deg at q=0.6 "
          f"(null 45; floor ~10 from the 12 deg axis error)")
    qs = [0.0, 0.25, 0.5, 1.0]
    amp = []
    for q in qs:
        v = []
        for k in range(60):
            gd = F.emit_galaxy(rng, "tensor", q_amp=q)
            r = E.galaxy_m3(gd)
            if r is None:
                continue
            W_, Cw = r
            psi = np.deg2rad(gd["axis_ext_obs"] - gd["pa_obs"])
            u = np.array([np.cos(2 * psi), np.sin(2 * psi)])
            v.append(float(u @ np.array([W_.real, W_.imag])))
        amp.append(float(np.mean(v)))
    sl = float(np.polyfit(qs, amp, 1)[0])
    check("T7 galaxy m=3 amplitude is responsive", 0.25 < sl < 0.8,
          f"d(aligned projection)/d(q_amp) = {sl:.3f}, expected ~0.5")


def t8_provenance():
    import guard
    from universes import provenance as pv
    led = guard.start()
    ok_sealed = ok_reserve = False
    try:
        io.open("C:/data/KiDS_dr4_shear.fits")
    except pv.SealedHoldoutTouched:
        ok_sealed = True
    except Exception:                                          # noqa: BLE001
        ok_sealed = False
    try:
        io.open("/some/path/clogs_profiles.csv")
    except pv.SealedHoldoutTouched:
        ok_reserve = True
    except Exception:                                          # noqa: BLE001
        ok_reserve = False
    ok_foreign = False
    try:
        io.open("C:/Users/henry/dev/gravity-discovery-program.md")
    except pv.ForeignReadError:
        ok_foreign = True
    except Exception:                                          # noqa: BLE001
        ok_foreign = False
    guard.stop()
    check("T8 a SEALED path raises before it can be read", ok_sealed,
          "KiDS token guarded")
    check("T8 a CONFIRMATION-RESERVE path raises", ok_reserve,
          "CLoGS token guarded")
    check("T8 a foreign read outside the lane raises", ok_foreign,
          "lane-root guard active")


def t9_exchangeability():
    """Scrambling the external-axis labels between clusters must return S_ext
    to its null: the statistic must carry no information about which cluster a
    label came from."""
    rng = np.random.default_rng(9)
    real, scram = [], []
    for k in range(60):
        cl = F.corpus("tensor", rng, n_clu=12, A_tensor=0.4)
        rows = PL.cluster_rows(cl, F.sigma_crit)
        real.append(PL.statistics(rows)["S_ext"])
        ax = [c["axis_ext_obs"] for c in cl]
        perm = rng.permutation(len(cl))
        for c, j in zip(cl, perm):
            c["axis_ext_obs"] = ax[j]
        scram.append(PL.statistics(PL.cluster_rows(cl, F.sigma_crit))["S_ext"])
    a, b = float(np.mean(real)), float(np.mean(scram))
    sb = float(np.std(scram) / np.sqrt(len(scram)))
    check("T9 label-scrambled arm returns S_ext to the null",
          a > 3 and abs(b) < 3 * max(sb, 1e-9) + 0.6,
          f"real {a:+.2f}, scrambled {b:+.2f} +- {sb:.2f}")


def t10_misspecified_axis():
    """A misspecified axis is a NULL detector: S_45 must not respond to A."""
    rng = np.random.default_rng(10)
    amps = [0.0, 0.15, 0.3, 0.6]
    m45, mex = [], []
    for A in amps:
        v45, vex = [], []
        for k in range(40):
            cl = F.corpus("tensor", rng, n_clu=12, A_tensor=A)
            S = PL.statistics(PL.cluster_rows(cl, F.sigma_crit))
            v45.append(S["S_45"])
            vex.append(S["S_ext"])
        m45.append(float(np.mean(v45)))
        mex.append(float(np.mean(vex)))
    s45 = np.polyfit(amps, m45, 1)[0]
    sex = np.polyfit(amps, mex, 1)[0]
    check("T10 a 45-degree misspecified axis does not respond",
          abs(s45) < 0.15 * abs(sex),
          f"d S_45/dA = {s45:+.2f} vs d S_ext/dA = {sex:+.2f}")


def main():
    print("=" * 74)
    print("cdm-separation lane tests")
    print("=" * 74)
    for fn in (t1_recovery, t2_null_debias, t3_covariance, t4_equivariance,
               t5_forward_internal, t6_sign_agreement, t7_galaxy,
               t8_provenance, t9_exchangeability, t10_misspecified_axis):
        fn()
    n_pass = sum(r["passed"] for r in RESULTS)
    print(f"\n{n_pass}/{len(RESULTS)} tests passed")
    with open(os.path.join(HERE, "results", "T_tests.json"), "w") as f:
        json.dump(dict(n_pass=n_pass, n_total=len(RESULTS), tests=RESULTS),
                  f, indent=1)
    return 0 if n_pass == len(RESULTS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
