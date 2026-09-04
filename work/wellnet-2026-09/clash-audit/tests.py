"""
Self-tests.  Every lane in this programme has found real bugs in its own first
implementation; these are the checks that found this lane's.
"""
from __future__ import annotations
import math

import numpy as np

import ingest as I
import stats as S
import nullsim as N

KPC, MPC, MSUN, G = I.KPC, I.MPC, I.MSUN, I.G
PASS, FAIL = [], []


def check(name, ok, detail=""):
    (PASS if ok else FAIL).append(name)
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}  {detail}")


def main():
    D = I.load_all(verbose=False)
    T = I.points_table(D)
    C = D["clusters"]

    # 1 -- ingest counts and identifiers
    check("84 rows, 20 clusters",
          len(T["r"]) == 84 and len(set(T["name"].tolist())) == 20)
    check("table1 RAR counts match fig2 row counts per cluster",
          all(int((T["name"] == n).sum()) == I.load_tab1()[n]["n_rar"] for n in C))

    # 2 -- a deliberately wrong -source= must be rejected by the identifier echo
    try:
        I._vizier(I.FIG2, "J/ApJ/896/71/fig2", I.FIG2_COLS, 84)
        check("wrong catalogue identifier is rejected", False)
    except AssertionError:
        check("wrong catalogue identifier is rejected", True)
    try:
        I._vizier(I.FIG2, "J/ApJ/896/70/fig2", I.FIG2_COLS[:-1], 84)
        check("truncated column list is rejected", False)
    except AssertionError:
        check("truncated column list is rejected", True)

    # 3 -- tie-corrected ranks.  CLASH has exactly repeated radii, so the
    #      argsort(argsort(v)) bug found in Run AL would fire here.
    v = np.array([100.0, 100.0, 100.0, 200.0, 200.0])
    naive = np.argsort(np.argsort(v)).astype(float)
    check("ranks are tie-corrected (Run AL bug does not recur)",
          abs(S.rank(v)[0] - S.rank(v)[1]) < 1e-12 and abs(naive[0] - naive[1]) > 0.5,
          f"tied ranks {S.rank(v)[:3]} vs naive {naive[:3]}")

    # 4 -- NFW self-consistency: R500 from the profile == R500 from M500
    n0 = "A2261"
    c = C[n0]
    R = I.r_delta(c["M200"], c["c200"], c["z"], 500.0)
    M = float(I.nfw_mass(R, c["M200"], c["c200"], c["z"]))
    A = (4 / 3) * math.pi * 500 * I.rhoc(c["z"])
    check("r_delta solves the overdensity condition",
          abs(M * MSUN / (A * R ** 3) - 1) < 1e-9,
          f"ratio-1 = {M*MSUN/(A*R**3)-1:.2e}")

    # 5 -- NFW mass scaling identity: at fixed c200, R500 prop M200^(1/3) exactly
    R1 = I.r_delta(c["M200"], c["c200"], c["z"], 500.0)
    R2 = I.r_delta(8 * c["M200"], c["c200"], c["z"], 500.0)
    check("R500 prop M200^(1/3) at fixed c200", abs(R2 / R1 - 2.0) < 1e-6,
          f"ratio {R2/R1:.8f} vs 2")

    # 6 -- the two excess statistics agree on the paired 100->600 kpc drop
    from scipy.optimize import brentq
    gb, go, r, nm = T["gb"], T["go"], T["r"], T["name"]
    a_dm = S.excess_a0(gb, go)
    a_inv = np.array([math.log10(10 ** brentq(
        lambda la, x=x_, y=y_: x_ * I.nu_rar(x_ / 10 ** la) - y_, -13, -6,
        xtol=1e-13) / 1.2e-10) for x_, y_ in zip(gb, go)])
    d = []
    for cn in sorted(set(nm.tolist())):
        m6 = (nm == cn) & (np.abs(r / KPC - 600) < 1)
        m1 = (nm == cn) & (np.abs(r / KPC - 100) < 1)
        if m6.sum() == 1 and m1.sum() == 1:
            d.append((float(a_dm[m6] - a_dm[m1]), float(a_inv[m6] - a_inv[m1])))
    d = np.array(d)
    check("deep-MOND a0 and full-RAR-inversion a0 agree on the 100->600 drop",
          len(d) == 11 and abs(d[:, 0].mean() - d[:, 1].mean()) < 0.02,
          f"n={len(d)}  {d[:,0].mean():+.4f} vs {d[:,1].mean():+.4f}")

    # 7 -- reproduce the record's published CLASH number
    check("record's within-CLASH a0 drop reproduced (-0.347 +- 0.057)",
          abs(d[:, 1].mean() + 0.347) < 0.01
          and abs(d[:, 1].std(ddof=1) / math.sqrt(len(d)) - 0.057) < 0.01,
          f"{d[:,1].mean():+.4f} +- {d[:,1].std(ddof=1)/math.sqrt(len(d)):.4f}, n={len(d)}")

    # 8 -- the rank identity, on the real design
    names = sorted(C)
    Dm = np.column_stack([(nm == cn).astype(float) for cn in names])
    lr = np.log10(r / KPC)
    lt = np.log10(r / np.array([C[cn]["R500_lens"] for cn in nm]))
    ra = np.linalg.matrix_rank(np.column_stack([Dm, lr]), tol=1e-9)
    rc = np.linalg.matrix_rank(np.column_stack([Dm, lr, lt]), tol=1e-9)
    check("log(r/R500) adds no rank beside cluster indicators + log r",
          ra == rc == 21, f"rank {ra} -> {rc}")

    # 9 -- fixed-effects slope is INVARIANT to the per-cluster normaliser
    y = S.excess_y(gb, go)
    s1 = S.fe_slope(lr, y, nm)
    s2 = S.fe_slope(lt, y, nm)
    rng = np.random.default_rng(7)
    fake = {cn: 10 ** rng.uniform(-1, 1) for cn in names}
    s3 = S.fe_slope(np.log10(r / np.array([C[cn]["R500_lens"] * fake[cn]
                                           for cn in nm])), y, nm)
    check("fixed-effects slope is bit-invariant under ANY per-cluster normaliser",
          abs(s1 - s2) < 1e-10 and abs(s1 - s3) < 1e-10,
          f"{s1:.10f} / {s2:.10f} / {s3:.10f}")

    # 10 -- Abel projection: recover an NFW Sigma from its own rho
    class _NfwTruth:
        z = 0.35
        name = "test"

        def rho(self, rr):
            M200, c200 = 1.5e15, 4.0
            r200 = (3 * M200 * MSUN / (4 * math.pi * 200 * I.rhoc(self.z))) ** (1 / 3)
            rs = r200 / c200
            mu = math.log(1 + c200) - c200 / (1 + c200)
            rho_s = M200 * MSUN / (4 * math.pi * rs ** 3 * mu)
            xx = np.asarray(rr) / rs
            return np.where(np.asarray(rr) > N.TRUNC, 0.0,
                            rho_s / (xx * (1 + xx) ** 2))
    t = _NfwTruth()
    Rt = np.geomspace(0.1 * MPC, N.FIT_RMAX, 8)
    num = N.Truth.sigma(t, Rt)
    ana = N.nfw_sigma(Rt, 1.5e15, 4.0, 0.35)
    rel = np.abs(num / ana - 1)
    check("Abel projection reproduces the analytic NFW Sigma",
          rel.max() < 0.01, f"max rel err {rel.max():.5f} at TRUNC = "
                            f"{N.TRUNC/MPC:.0f} Mpc")
    # the bug this check found: at the original TRUNC = 5 Mpc the deficit was
    # 6.5% at R = 1.5 Mpc and grows outward, biasing the fitted NFW.
    old = N.TRUNC
    N.TRUNC = 5.0 * MPC
    bad = np.abs(N.Truth.sigma(t, Rt) / ana - 1).max()
    N.TRUNC = old
    check("that check would have failed at the original TRUNC = 5 Mpc",
          bad > 0.05, f"max rel err would have been {bad:.4f}")

    # 11 -- the NFW fitter recovers injected parameters from a noiseless NFW
    Sig = N.nfw_sigma(N.R_FIT, 1.2e15, 3.5, 0.35)
    M, cc = N.fit_nfw(Sig, 0.35, 0.05 * Sig)
    check("NFW fitter recovers (M200,c200) from a noiseless NFW",
          abs(M / 1.2e15 - 1) < 0.03 and abs(cc / 3.5 - 1) < 0.03,
          f"M {M:.3e} c {cc:.3f}")

    # 12 -- the truth generator delivers the excess it was asked for
    n1 = "A2261"
    m = nm == n1
    tr = N.Truth(n1, r[m], np.log10(gb[m]), C[n1]["z"], 0.50, 0.0)
    gt = tr.gbar(r[m]) * I.nu_rar(tr.gbar(r[m]) / I.A0) * 10 ** 0.50
    check("flat truth has exactly the excess it was given",
          np.abs(S.excess_y(tr.gbar(r[m]), gt) - 0.50).max() < 1e-9)

    # 13 -- a sloped truth has exactly the slope it was given
    R5t = C[n1]["R500_lens"]
    tr2 = N.Truth(n1, r[m], np.log10(gb[m]), C[n1]["z"], 0.50, -0.30, R500_ref=R5t)
    gt2 = (tr2.gbar(r[m]) * I.nu_rar(tr2.gbar(r[m]) / I.A0)
           * 10 ** tr2.excess_true(r[m]))
    sl = S.ols_slope(np.log10(r[m] / R5t), S.excess_y(tr2.gbar(r[m]), gt2))
    check("sloped truth has exactly the injected slope",
          abs(sl + 0.30) < 1e-9, f"{sl:+.10f}")

    # 14 -- baryonic mass recovered from g_bar is a real mass
    Mb = S.baryonic_mass(r, gb)
    check("M_bar(<r) from g_bar is monotone and cluster-scale within each cluster",
          all(np.all(np.diff(Mb[nm == cn][np.argsort(r[nm == cn])]) > 0)
              for cn in names) and 1e12 < np.median(Mb) < 1e15,
          f"median {np.median(Mb):.2e} Msun")

    # 15 -- the R_b,g normaliser puts g_bar on BOTH axes (a shared-quantity trap)
    x, norm, Rb, thr = S.radial_definitions(T, C)
    # The excess statistics carry g_bar in the denominator (y subtracts
    # log nu_RAR(g_bar/a0); a0_eff = g_obs^2/g_bar), so a normaliser whose
    # BETWEEN-cluster variation tracks the cluster's baryon amplitude puts the
    # same quantity on both axes -- the shared-denominator pattern.
    #
    # MY FIRST DIAGNOSTIC WAS WRONG AND THIS CHECK CAUGHT IT.  It asserted
    # |corr(log(r/R_b,g), log g_bar)| > 0.9 pooled over all 84 points and got
    # -0.64, LOWER than r_physical's -0.78: pooled over radius that correlation
    # is dominated by g_bar falling with r and says nothing about the
    # normaliser.  The correct diagnostic is BETWEEN clusters at fixed radius.
    m200 = np.abs(r / KPC - 200) < 1e-6
    n200 = [cn for cn in names if (m200 & (nm == cn)).sum() == 1]
    lgb200 = np.array([float(np.log10(gb[m200 & (nm == cn)][0])) for cn in n200])
    con = {}
    for k in ("R500_lens", "R500_xray", "R500_TX", "Rb_gas", "Rb_M", "Rb_g"):
        v = np.array([norm[cn][k] for cn in n200])
        ok = np.isfinite(v)
        con[k] = S.pear(np.log10(v[ok]), lgb200[ok])
    # SECOND WRONG GUESS, ALSO CAUGHT HERE.  I then predicted R_b,g would top
    # the table; it is +0.53, while R_b,M is -0.99.  R_b,M is the radius at which
    # M_bar(<R) reaches a global constant, and M_bar ~ g_bar r^2, so a
    # baryon-richer cluster reaches it at a smaller radius -- an almost exact
    # anticorrelation.  A "baryon-only" normaliser is NOT automatically a clean
    # control: it must be checked against the baryon amplitude, because g_bar is
    # already in the denominator of the excess.
    worst = max(con, key=lambda k: abs(con[k]))
    check("a baryon-only normaliser is severely contaminated by g_bar",
          worst == "Rb_M" and abs(con["Rb_M"]) > 0.9,
          "between-cluster corr(log R_norm, log g_bar at 200 kpc): "
          + ", ".join(f"{k}={v:+.2f}" for k, v in con.items())
          + f"  -> worst = {worst}")

    print(f"\n{len(PASS)}/{len(PASS)+len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
    return len(FAIL)


if __name__ == "__main__":
    raise SystemExit(main())
