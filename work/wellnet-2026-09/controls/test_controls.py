"""test_controls.py -- every control in `controls.py`, on a worked example.

Run it as a script to reproduce `controls_validation.json`:

    python test_controls.py            # full run, a few minutes
    python test_controls.py --fast     # smaller simulation counts

Each `test_*` function is also a plain pytest test. The worked examples are
deliberately ones where the control has something to CATCH: a physics-free
twin that manufactures gain, a cluster whose well network is real, a rank
statistic that is blind to its own parameter, a pipeline that treats the two
arms differently, a correlation with a shared denominator, and a blind
evaluation that re-solves its coefficients on the blind set.

Where real data exist they are used: the shared-denominator example is the
actual LoCuSS catalogue (Mulroy 2019 + the Subaru weak-lensing masses), and it
reproduces the retracted rho_p = -0.304 and its correction from the raw
tables.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import controls as C                                        # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ACQ = os.path.abspath(os.path.join(
    HERE, "..", "..", "gravity-cluster-audit-2026-09", "acquire"))
OUT = os.path.join(HERE, "controls_validation.json")
FAST = "--fast" in sys.argv
RESULTS: dict = {}


def _rec(name, payload):
    RESULTS[name] = payload
    return payload


def hdr(s):
    print("\n" + "=" * 78)
    print(s)
    print("=" * 78)


# ==========================================================================
#  shared fixtures
# ==========================================================================
def make_cluster(seed=20260904, N=200, segregate=True, filament=True):
    """One mock cluster with BOTH kinds of structure a well-network claim
    would live on: mass segregation (mass correlated with radius) and an
    anisotropic geometry (members concentrated along an axis)."""
    rng = np.random.default_rng(seed)
    r = 10.0 ** rng.uniform(-1.3, 0.35, N)                 # Mpc
    lm = rng.normal(10.4, 0.55, N)
    if segregate:
        lm = lm - 0.55 * (np.log10(r) - np.log10(r).mean())
    m = 10.0 ** lm
    d = rng.normal(size=(N, 3))
    if filament:
        d[:, 2] *= 3.0                                     # a preferred axis
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return C.ClusterSource(r[:, None] * d, m, cid=np.zeros(N, np.int64),
                           name="mock-cluster")


def make_field(n=40, dx=1.0, seed=5):
    """A resolved 3-D map with a real quadrupole and a noisy observation."""
    rng = np.random.default_rng(seed)
    ax = (np.arange(n) - (n - 1) / 2.0) * dx
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2) + 1e-6
    ct = Z / R
    rho = np.exp(-R / 5.0) * (1.0 + 0.8 * ct ** 2)
    model = np.log10(np.maximum(np.exp(-R / 5.0), 1e-12))
    sig = np.full(rho.shape, 0.08) * (1.0 + 0.5 * (R / R.max()))
    value = np.log10(np.maximum(rho, 1e-12)) + rng.normal(0, 1, rho.shape) * sig
    return C.FieldSource(rho, dx, value=value, sigma=sig, model=model,
                         mask=R < 0.45 * n * dx, name="mock-field")


def best_subset_on_train(X, y, split, k=3, rng=None, ntry=4000):
    """A deliberately small search, run on TRAIN ONLY, that overfits.

    Train is fully accessible on purpose: the frozen-coefficient guard exists
    to protect the BLIND set, not to make model selection impossible.
    """
    rng = rng or np.random.default_rng(0)
    tr = split == "train"
    Xt, yt = X[tr], y[tr]
    p = X.shape[1]
    best, bidx = np.inf, None
    for _ in range(ntry):
        idx = np.r_[0, rng.choice(np.arange(1, p), k, replace=False)]
        A = Xt[:, idx]
        c = np.linalg.lstsq(A, yt, rcond=None)[0]
        r = float(np.sqrt(((yt - A @ c) ** 2).mean()))
        if r < best:
            best, bidx = r, idx
    return bidx, best


# ==========================================================================
#  1  RESIDUAL NULLS
# ==========================================================================
def test_residual_null():
    hdr("CONTROL 1 -- residual nulls (point source and field)")
    u = C.synthetic_universe("tensor", 3, n_sys=60, n_r=10, n_t=5).data
    src = u.source
    out = {}

    # (a) the field generalisation: (object, shell) blocks
    cr = C.residual_null(src, 11, block=u.block_angular, radius_from=u.r)
    cr.check(1e-12)
    print("   blocks: " + str(cr.meta["blocks"]))
    for k, v in cr.invariants.items():
        print(f"      invariant {k:<28} residual {v[2]:.3e}")
    for k, v in cr.destroyed.items():
        print(f"      destroyed {k:<28} {v[0]!r} -> {v[1]!r}")
    out["angular_block"] = cr.summary()

    # (b) Run J's perm_g, byte for byte
    cg = C.residual_null(src, 11, block="object", standardise=False)
    cg.check(1e-12)
    out["perm_g_legacy"] = cg.summary()

    # (c) what it catches: the manufactured gain of a small search
    Xs, ns, _ = C.build_bank(u, ("scalar", "radial", "gal"))
    Xt, nt, _ = C.build_bank(u, ("scalar", "radial", "gal", "tensor"))
    split = C._split_labels(u.sysid, seed=0)
    Yn, nrec = C.residual_null_batch(u.source, 12, 1, block=u.block_angular,
                                     radius_from=u.r)
    nrec.check(1e-9)
    gains = {}
    for label, y in (("real", src.value), ("perm_g twin", Yn[:, 0])):
        rb, _, _, _ = C._frozen_blind_rms(Xs, y, split, ns)
        rt, _, _, _ = C._frozen_blind_rms(Xt, y, split, nt)
        gains[label] = {"blind_scalar": float(rb), "blind_scalar_tensor": float(rt),
                        "tensor_gain_pct": float(C._gain(float(rb), float(rt)))}
        print(f"      {label:<12} blind scalar {float(rb):.5f} -> "
              f"scalar+tensor {float(rt):.5f}   "
              f"({gains[label]['tensor_gain_pct']:+.2f}%)")
    out["injected_tensor_vs_its_own_null"] = gains

    assert cr.invariants["block_mean_residual"][2] < 1e-12
    assert cr.invariants["sigma_map"][2] == 0.0
    assert cr.invariants["model"][2] == 0.0
    assert cg.invariants["within_block_residual_multiset"][2] < 1e-12
    assert gains["real"]["tensor_gain_pct"] > gains["perm_g twin"]["tensor_gain_pct"]
    return _rec("control_1_residual_null", out)


def test_residual_null_field():
    hdr("CONTROL 1 -- residual null on a resolved 3-D field")
    f = make_field()
    cr = C.residual_null(f, 7, block="object+shell", nshell=8)
    cr.check(1e-9)
    for k, v in cr.invariants.items():
        print(f"      invariant {k:<34} residual {v[2]:.3e}")
    a0, a1 = cr.destroyed["angular_rms_of_shell_mean_resid"]
    print(f"      angular structure of the residual: {a0:.5f} -> {a1:.5f} "
          f"({100*(1-a1/a0):.1f}% removed)")
    assert cr.invariants["block_mean_residual"][2] < 1e-9
    assert a1 < 0.4 * a0
    return _rec("control_1_residual_null_field", cr.summary())


# ==========================================================================
#  2, 3, 4  CLUSTER CONTROLS
# ==========================================================================
def test_cluster_controls():
    hdr("CONTROLS 2, 3, 4 -- position scramble, mass scramble, smoothed source")
    cl = make_cluster()
    W0 = cl.network_energy()
    print(f"   mock cluster: {cl.N} members, W = {W0:.4e} Msun^2/Mpc")

    ps = C.position_scramble(cl, 21)
    ps.check(1e-12)
    ms = C.mass_scramble(cl, 21)
    ms.check(1e-9)
    sm = C.smoothed_source(cl)
    sm.check(1e-12)
    for nm, cr in (("position_scramble", ps), ("mass_scramble", ms),
                   ("smoothed_source", sm)):
        print(f"   {nm}")
        for k, v in cr.invariants.items():
            print(f"      invariant {k:<30} residual {v[2]:.3e}")
        for k, v in cr.destroyed.items():
            print(f"      destroyed {k:<30} {C._j(v[0])} -> {C._j(v[1])}")

    # the ensemble identity: E[W | position scramble] = sum m_i m_j / r_>
    mf = C.meanfield_network_energy(cl)
    n = 40 if FAST else 300
    Ws = np.array([C.position_scramble(cl, 1000 + i).data.network_energy()
                   for i in range(n)])
    z = (W0 - Ws.mean()) / Ws.std(ddof=1)
    rel = abs(Ws.mean() - mf) / mf
    print(f"\n   {n} position scrambles: mean W {Ws.mean():.4e} +- "
          f"{Ws.std(ddof=1):.2e}")
    print(f"      closed-form mean field  {mf:.4e}   relative "
          f"difference {rel:.2e}")
    print(f"      observed W sits {z:+.1f} sigma above the scrambled ensemble")
    print(f"      => controls 2 and 4 are the same null: the smoothed source "
          f"IS the ensemble mean of the position scramble")

    # what a naive well-network claim would look like, calibrated
    p = float((1 + (Ws >= W0).sum()) / (n + 1))
    print(f"      p(W >= observed | geometry destroyed, radial profile fixed) "
          f"= {p:.4f}")

    # the profile-only statistic MUST NOT move under control 2
    prof0, prof1 = cl.radial_profile(), ps.data.radial_profile()
    print(f"      radial mass profile L1 change under control 2: "
          f"{np.abs(prof0 - prof1).sum():.3e} Msun (must be exactly 0)")

    assert ps.invariants["radial_mass_profile"][2] == 0.0
    assert ps.invariants["member_radii_relative"][2] < 1e-12
    assert ms.invariants["positions"][2] == 0.0
    assert rel < 0.05
    assert z > 3.0
    return _rec("controls_2_3_4_cluster", {
        "position_scramble": ps.summary(), "mass_scramble": ms.summary(),
        "smoothed_source": sm.summary(),
        "W_observed": W0, "W_meanfield_closed_form": mf,
        "W_scramble_mean": float(Ws.mean()), "W_scramble_sd": float(Ws.std(ddof=1)),
        "n_scrambles": n, "meanfield_relative_error": rel,
        "z_of_observed": float(z), "p_geometry": p,
        "radial_profile_L1_change": float(np.abs(prof0 - prof1).sum())})


def test_smoothed_field():
    hdr("CONTROL 4 -- smoothed source on a resolved field")
    f = make_field()
    cr = C.smoothed_source(f)
    cr.check(1e-12)
    for k, v in cr.invariants.items():
        print(f"      invariant {k:<32} residual {v[2]:.3e}")
    sa = cr.destroyed["shell_anisotropy"]
    q2 = cr.destroyed["quadrupole_l2"]
    q4 = cr.destroyed["hexadecapole_l4"]
    print(f"      shell anisotropy (lattice-free)  {sa[0]:.5f} -> {sa[1]:.3e}")
    print(f"      quadrupole   l=2                 {q2[0]:.5f} -> {q2[1]:.3e}")
    print(f"      hexadecapole l=4                 {q4[0]:.5f} -> {q4[1]:.5f}")
    print(f"      l=4 does NOT go to zero: a CUBIC grid has an l=4 anisotropy "
          f"of its own.")
    print(f"      the smoothed field is purely radial by construction, so its "
          f"l=4 = {cr.meta['lattice_l4_floor']:.5f} IS the lattice floor.")
    print(f"      the source's l=4 excess over the floor is "
          f"{q4[0] - q4[1]:+.5f}; l=2 has no cubic term and vanishes exactly.")
    assert cr.invariants["total_mass"][2] < 1e-12
    assert cr.invariants["M_enclosed_at_shell_edges"][2] < 1e-11
    assert sa[1] < 1e-12 and sa[0] > 0.05
    assert q2[1] < 1e-10
    return _rec("control_4_smoothed_field", cr.summary())


# ==========================================================================
#  5  SYNTHETIC KNOWN-LAW UNIVERSES
# ==========================================================================
def test_known_laws():
    hdr("CONTROL 5 -- five known-law universes through the full pipeline")
    r = C.known_law_suite(seed=11, B=99 if FAST else 199,
                          B_nl=49 if FAST else 99, verbose=True)
    for k, v in r["requirements"].items():
        assert v, f"requirement {k} FAILED"
    return _rec("control_5_known_laws", {"table": r["table"],
                                         "requirements": r["requirements"],
                                         "per_law": {
        l: {"family": r["per_law"][l]["family"],
            "structure": r["per_law"][l]["structure"]} for l in r["per_law"]}})


def test_tensor_fpr():
    hdr("CONTROL 5 -- false-positive rate for 'tensor effect in scalar data'")
    r = C.tensor_false_positive_rate(
        n_universes=40 if FAST else 200, B=99 if FAST else 199,
        n_universes_nl=15 if FAST else 40, B_nl=49 if FAST else 99,
        seed=20260904, verbose=True)
    print("\n   per generating law:")
    for l, v in r["per_law"].items():
        print(f"      {l:<8} n={v['n']:<4} naive {100*v['fpr_naive']:5.1f}%  "
              f">1% {100*v['fpr_threshold1pct']:5.1f}%  calibrated "
              f"{100*v['fpr_calibrated']:5.1f}%  median gain "
              f"{v['median_gain_T_pct']:+.2f}%")
    slim = {k: v for k, v in r.items() if k not in ("rows", "rows_nonlocal")}
    slim["n_universes"] = len(r["rows"])
    slim["p_values_tensor"] = [round(x["p_T"], 4) for x in r["rows"]]
    slim["gains_tensor_pct"] = [round(x["gain_T"], 4) for x in r["rows"]]
    # the calibrated rule must be near its nominal size, and must be far below
    # the naive rule -- that gap is the whole point of the control
    assert r["fpr_calibrated"] < 0.15
    assert r["fpr_naive"] > 3 * r["fpr_calibrated"]
    assert r["p_uniformity_ks"]["p"] > 0.01
    return _rec("control_5_tensor_false_positive_rate", slim)


# ==========================================================================
#  6  PARAMETER SENSITIVITY
# ==========================================================================
def test_parameter_responsiveness():
    hdr("CONTROL 6 -- parameter sensitivity, reproducing the monotone trap")
    cl = load_locuss()
    kappas = np.logspace(3.0, 6.0, 7)                # three decades of kappa
    print("   the model quantity is Y_pred = kappa t f_gas (the deep-MOND")
    print("   limit of the amplified-pressure law). kappa multiplies it, so")
    print("   every RANK statistic of Y_pred is exactly invariant in kappa.")

    def ch(k):
        return locuss_chain(cl["M_WL"], cl["M_gas"], cl["L_K"], cl["z"],
                            cl["kT"], k)
    Yobs = ch(0.0)["E_obs"] ** 2 - 1.0

    stats = {
        "spearman(Y_pred, kT)":
            lambda k: float(np.corrcoef(C._rank(ch(k)["Ypred_deep"]),
                                        C._rank(cl["kT"]))[0, 1]),
        "partial_spearman(Y_pred, kT | M_WL)":
            lambda k: C.partial_spearman(ch(k)["Ypred_deep"], cl["kT"],
                                         cl["M_WL"]),
        "median Y_pred":
            lambda k: float(np.median(ch(k)["Ypred_deep"])),
        "mean ln(E_obs/E_pred)":
            lambda k: float(np.mean(np.log(ch(0.0)["E_obs"]
                                           / ch(k)["E_pred"]))),
        "chi2-like mean sq ln residual":
            lambda k: float(np.mean(np.log(ch(0.0)["E_obs"]
                                           / ch(k)["E_pred"]) ** 2)),
    }
    out, blind = {}, []
    for nm, fn in stats.items():
        r = C.assert_parameter_responsive(fn, kappas, name=nm,
                                          raise_on_fail=False)
        out[nm] = r
        if not r["passed"]:
            blind.append(nm)
    print(f"\n   {len(blind)} of {len(stats)} statistics are BLIND to kappa "
          f"over three decades: {blind}")

    # and the suite must raise when any member is blind
    raised = False
    try:
        C.responsiveness_suite(stats, kappas, verbose=False)
    except C.ParameterBlindError as e:
        raised = True
        print(f"   responsiveness_suite raised, as it must: {str(e)[:90]}")
    assert raised, "a blind statistic did not stop the run"
    assert "spearman(Y_pred, kT)" in blind
    assert "partial_spearman(Y_pred, kT | M_WL)" in blind
    assert out["median Y_pred"]["passed"]
    assert out["chi2-like mean sq ln residual"]["passed"]
    assert out["spearman(Y_pred, kT)"]["spread"] == 0.0
    return _rec("control_6_parameter_sensitivity",
                {"kappa_grid": [float(k) for k in kappas],
                 "blind_statistics": blind, "detail": out,
                 "suite_raised": raised})


# ==========================================================================
#  7  EXCHANGEABILITY
# ==========================================================================
def _pipeline_common(src, bug=None):
    """A pipeline with every operation class the brief names: aperture,
    masking, sampling/binning, smoothing and interpolation.

    `bug` injects one of the two ways an arm quietly stops being exchangeable.
    """
    is_ctl = bool(src.extra.get("is_control"))
    r = src.coord[:, 0]
    C.trace_note("aperture", r_max=25.0)
    keep = np.where(r < 25.0, 1.0, 0.0)
    v = np.where(np.isfinite(src.value), src.value, 0.0) * keep
    edges = np.quantile(r, np.linspace(0.0, 1.0, 9))
    b = np.digitize(r, edges[1:-1])
    prof = np.array([v[b == k].mean() for k in range(8)])
    kern = np.array([0.25, 0.5, 0.25])
    if bug == "kernel" and is_ctl:
        kern = np.array([0.10, 0.80, 0.10])      # same shape, different values
    prof = np.convolve(prof, kern, mode="same")
    if bug == "extra_pass" and not is_ctl:
        prof = np.convolve(prof, kern, mode="same")   # one arm smoothed twice
    fine = np.interp(np.linspace(0.0, 7.0, 33), np.arange(8.0), prof)
    return float(np.sqrt((fine ** 2).mean()))


def test_exchangeability():
    hdr("CONTROL 7 -- exchangeability of the true and control arms")
    u = C.synthetic_universe("mond", 5, n_sys=30, n_r=8, n_t=4).data
    ctrl = C.residual_null(u.source, 6, block=u.block_angular,
                           radius_from=u.r).data
    ctrl.extra["is_control"] = True

    print("   (a) both arms through identical code")
    ok = C.check_exchangeability(lambda s: _pipeline_common(s), u.source,
                                 ctrl, name="identical-treatment")

    print("\n   (b) the TRUE arm gets a second smoothing pass")
    raised, bad = False, None
    try:
        C.check_exchangeability(lambda s: _pipeline_common(s, "extra_pass"),
                                u.source, ctrl, name="extra-smoothing-pass")
    except C.ExchangeabilityError as e:
        raised, bad = True, str(e)
        print(f"      raised, as it must: {bad[:110]}")

    print("\n   (c) same op count, but the CONTROL arm uses a different kernel")
    kern_rec = C.check_exchangeability(
        lambda s: _pipeline_common(s, "kernel"), u.source, ctrl,
        name="different-kernel", raise_on_fail=False)
    conv_warn = [w for w in kern_rec["warnings"]
                 if "convolve" in w["msg"] and w["kind"] == "arg1"]
    print(f"      value-level warnings on the convolution kernel: "
          f"{len(conv_warn)}")
    strict_raised = False
    try:
        C.check_exchangeability(lambda s: _pipeline_common(s, "kernel"),
                                u.source, ctrl, name="different-kernel-strict",
                                strict=True, verbose=False)
    except C.ExchangeabilityError as e:
        strict_raised = True
        print(f"      strict=True raises: {str(e)[:110]}")

    print("\n   (d) randomness inside the pipeline")
    rnd_raised = False
    try:
        C.check_exchangeability(
            lambda s: float(np.random.normal(0, 1) + s.value.mean()),
            u.source, ctrl, name="random-pipeline", verbose=False)
    except C.ExchangeabilityError as e:
        rnd_raised = True
        print(f"      refused: {str(e)[:100]}")

    assert ok["passed"] and ok["n_errors"] == 0
    assert ok["n_ops_true"] == ok["n_ops_control"] >= 8
    assert raised, "an asymmetric pipeline was not caught"
    assert len(conv_warn) >= 1, "a different smoothing kernel was not surfaced"
    assert strict_raised, "strict mode did not raise on the kernel difference"
    assert rnd_raised, "a random pipeline was not caught"
    return _rec("control_7_exchangeability",
                {"identical": ok, "extra_pass_raised": raised,
                 "extra_pass_message": bad,
                 "different_kernel": kern_rec,
                 "different_kernel_warnings_on_kernel_arg": len(conv_warn),
                 "different_kernel_strict_raised": strict_raised,
                 "random_pipeline_raised": rnd_raised})


# ==========================================================================
#  8  SHARED DENOMINATOR -- the real LoCuSS retraction, from the raw tables
# ==========================================================================
G_SI = 6.67430e-11
MSUN = 1.98892e30
MPC = 3.0856775814913673e22
A0_SI = 1.2e-10
H0_SI = 70.0 * 1000.0 / MPC
OM, OL = 0.3, 0.7
RHO_C0 = 3.0 * H0_SI ** 2 / (8.0 * math.pi * G_SI)
UPSILON_K = 0.73
KEV_J = 1.602176634e-16
M_P = 1.67262192369e-27
C_LIGHT = 2.99792458e8
MU_MOL = 0.6


def _read_tsv(path):
    lines = [l.rstrip("\n").rstrip("\r")
             for l in open(path, encoding="utf-8") if l.strip()]
    hdr_ = lines[0].split("\t")
    rows = []
    for ln in lines[1:]:
        q = ln.split("\t")
        q += [""] * (len(hdr_) - len(q))
        assert len(q) == len(hdr_), (path, len(q), len(hdr_))
        rows.append(dict(zip(hdr_, q)))
    return hdr_, rows


def _f(s):
    s = s.strip()
    return None if s in ("", "--") else float(s)


def load_locuss():
    """LoCuSS: Mulroy 2019 observables + Subaru weak-lensing masses.

    Row and column counts are asserted after the ingest, per the standing
    brief: a silent extraction failure returns fewer rows with no error.
    """
    _, rs = _read_tsv(os.path.join(ACQ, "mulroy2019_sample.tsv"))
    _, ro = _read_tsv(os.path.join(ACQ, "mulroy2019_observables.tsv"))
    assert len(rs) == 41 and len(ro) == 41, (len(rs), len(ro))
    obs = {r["Name"]: r for r in ro}
    assert set(obs) == {r["Name"] for r in rs}, "name mismatch between tables"
    keep, dropped = [], []
    for r in rs:
        o = obs[r["Name"]]
        d = dict(name=r["Name"], z=_f(r["z"]),
                 M_WL=_f(r["M_WL"]),
                 M_WL_e=0.5 * (_f(r["M_WL_ep"]) + _f(r["M_WL_em"])),
                 kT=_f(o["kT_X_ce"]),
                 kT_e=0.5 * (_f(o["kT_X_ce_ep"]) + _f(o["kT_X_ce_em"])),
                 M_gas=_f(o["M_gas"]),
                 M_gas_e=0.5 * (_f(o["M_gas_ep"]) + _f(o["M_gas_em"])),
                 L_K=_f(o["L_K_tot"]),
                 L_K_e=None if _f(o["L_K_tot_ep"]) is None
                 else 0.5 * (_f(o["L_K_tot_ep"]) + _f(o["L_K_tot_em"])))
        if any(d[k] is None for k in ("M_WL", "kT", "M_gas", "L_K")):
            dropped.append(d["name"])
            continue
        keep.append(d)
    assert len(keep) == 40 and dropped == ["Abell2697"], (len(keep), dropped)
    out = {k: np.array([c[k] for c in keep], float)
           for k in ("z", "M_WL", "M_WL_e", "kT", "kT_e", "M_gas", "M_gas_e",
                     "L_K", "L_K_e")}
    out["name"] = [c["name"] for c in keep]
    return out


def locuss_t(kT):
    """t = 3 kT / (mu m_p c^2)."""
    return 3.0 * np.asarray(kT, float) * KEV_J / (MU_MOL * M_P * C_LIGHT ** 2)


def locuss_chain(M_WL, M_gas, L_K, z, kT=None, kappa=0.0):
    """The exact Run K forward chain. Returns E_obs, E_pred and the deep-MOND
    prediction Y_pred = kappa t f_gas.

    M_WL enters E_obs TWICE: once as the numerator gravity and once through
    r500, which is derived from it. That is the shared denominator.
    """
    Ez2 = OM * (1.0 + z) ** 3 + OL
    r500 = (3.0 * (M_WL * 1e14 * MSUN)
            / (4.0 * np.pi * 500.0 * RHO_C0 * Ez2)) ** (1.0 / 3.0)
    M_b = M_gas + UPSILON_K * L_K * 1e12 / 1e14
    f_gas = M_gas / M_b
    gNb = G_SI * (M_b * 1e14 * MSUN) / r500 ** 2
    gb = C.nu_rar(gNb / A0_SI) * gNb
    gWL = G_SI * (M_WL * 1e14 * MSUN) / r500 ** 2
    out = {"E_obs": gWL / gb, "M_b": M_b, "f_gas": f_gas, "g_N_b": gNb}
    if kT is not None:
        t = locuss_t(kT)
        delta = kappa * t * M_gas / M_b
        gNe = gNb * (1.0 + delta)
        out["E_pred"] = (C.nu_rar(gNe / A0_SI) * gNe) / gb
        out["Ypred_deep"] = kappa * t * f_gas
        out["t"] = t
    return out


def locuss_E(M_WL, M_gas, L_K, z):
    return locuss_chain(M_WL, M_gas, L_K, z)["E_obs"]


def test_shared_denominator():
    hdr("CONTROL 8 -- shared-denominator detector on the real LoCuSS sample")
    cl = load_locuss()
    print(f"   ingested {len(cl['name'])} clusters (41 in the tables, "
          f"Abell2697 dropped for a missing L_K)")
    E = locuss_E(cl["M_WL"], cl["M_gas"], cl["L_K"], cl["z"])
    print(f"   E_obs median {np.median(E):.3f}  range {E.min():.2f}-{E.max():.2f}"
          f"   (record: median 1.62, range 1.22-2.34)")
    rho = C.partial_spearman(np.log(E), np.log(cl["kT"]), np.log(cl["M_WL"]))
    print(f"   naive partial Spearman rho_p(E, kT | M_WL) = {rho:+.4f}"
          f"   (the RETRACTED value: -0.304)")

    inputs = {
        "M_WL": {"value": cl["M_WL"], "sigma": cl["M_WL_e"], "dist": "lognormal"},
        "M_gas": {"value": cl["M_gas"], "sigma": cl["M_gas_e"], "dist": "lognormal"},
        "L_K": {"value": cl["L_K"], "sigma": cl["L_K_e"], "dist": "lognormal"},
        "kT": {"value": cl["kT"], "sigma": cl["kT_e"], "dist": "lognormal"},
        "z": {"value": cl["z"], "sigma": np.zeros_like(cl["z"])},
    }
    exprs = {"lnE": "log(locuss_E(M_WL, M_gas, L_K, z))",
             "lnM": "log(M_WL)", "lnT": "log(kT)"}
    env = {"locuss_E": locuss_E}
    nn = 800 if FAST else 4000

    rank_est = (lambda s: C.partial_spearman(s["lnE"], s["lnT"], s["lnM"]))
    rep_rank = C.shared_denominator_report(
        inputs, exprs, rank_est, null_carrier="kT", seed=20260904,
        ndraw=nn // 2, nnull=nn, env=env, series_order=["lnE", "lnM", "lnT"],
        carrier_series="lnT", nboot_eiv=0)

    def slope_est(s):
        X = np.column_stack([np.ones(len(s["lnE"])), s["lnM"], s["lnT"]])
        return float(np.linalg.lstsq(X, s["lnE"], rcond=None)[0][2])
    print("\n   the SAME detector on the naive OLS partial slope, which is the "
          "estimator\n   the record's -0.12 null expectation belongs to:")
    rep_slope = C.shared_denominator_report(
        inputs, exprs, slope_est, null_carrier="kT", seed=20260905,
        ndraw=nn // 2, nnull=nn, env=env, series_order=["lnE", "lnM", "lnT"],
        carrier_series="lnT", nboot_eiv=200 if not FAST else 0)

    ec = rep_rank["induced_error_correlation"]["lnE|lnM"]["mean"]
    ns = rep_slope["null_structural"]
    eiv = rep_slope["eiv"]
    print("\n   against the record of Run K:")
    print(f"      error correlation ln E_obs vs ln M_WL   "
          f"{ec:+.3f}      record +0.96")
    print(f"      naive partial Spearman                 {rho:+.4f}    "
          f"record -0.304")
    print(f"      naive OLS partial slope                "
          f"{rep_slope['statistic_observed']:+.4f}    record -0.155")
    print(f"      its null expectation                   "
          f"{ns['mean']:+.4f}    record -0.12")
    print(f"      p of the observed against its own null  "
          f"{ns['p_two_sided']:.3f}     record 0.563")
    print(f"      EIV slope d lnE/d lnT                  "
          f"{eiv['carrier_beta']:+.4f}    record -0.166")
    print(f"      EIV mass-slope attenuation             "
          f"{eiv['beta'][0] / eiv['beta_naive'][0]:.3f}     record 0.66")

    assert rep_rank["has_shared_input"]
    assert rep_rank["shared_inputs"]["lnE|lnM"] == ["M_WL"]
    assert ec > 0.9
    assert abs(rho + 0.304) < 0.01
    assert abs(rep_slope["statistic_observed"] + 0.155) < 0.005
    assert ns["mean"] < -0.05, "the structural null should NOT be centred on 0"
    assert ns["p_two_sided"] > 0.2, "against its own null this is not a detection"
    assert abs(eiv["carrier_beta"] + 0.166) < 0.05
    return _rec("control_8_shared_denominator", {
        "n_clusters": len(cl["name"]),
        "E_obs_median": float(np.median(E)),
        "E_obs_range": [float(E.min()), float(E.max())],
        "naive_partial_spearman": rho,
        "report_rank_statistic": rep_rank,
        "report_naive_slope": rep_slope,
        "record_comparison": {
            "error_correlation": {"measured": ec, "record": 0.96},
            "naive_partial_spearman": {"measured": rho, "record": -0.304},
            "naive_partial_slope": {
                "measured": rep_slope["statistic_observed"], "record": -0.155},
            "null_expectation": {"measured": ns["mean"], "record": -0.12},
            "p_against_own_null": {"measured": ns["p_two_sided"], "record": 0.563},
            "eiv_slope": {"measured": eiv["carrier_beta"], "record": -0.166},
            "mass_slope_attenuation": {
                "measured": float(eiv["beta"][0] / eiv["beta_naive"][0]),
                "record": 0.66}}})


def test_eiv_unbiased():
    hdr("CONTROL 8 -- is the errors-in-variables estimator unbiased?")
    r = C.validate_eiv(n=40, rho_err=0.96, nsim=60 if FAST else 300,
                       betas=(-0.6, -0.3, 0.0, 0.3, 0.6), seed=7)
    assert r["unbiased"], "the EIV estimator is biased somewhere in the range"
    assert r["bias_reduction_factor"] > 4.0
    return _rec("control_8_eiv_validation", r)


# ==========================================================================
#  9  FROZEN COEFFICIENTS
# ==========================================================================
def test_frozen_coefficients():
    hdr("CONTROL 9 -- frozen-coefficient enforcement (the Run J bug)")
    # a small sample and a wide bank: the regime Run J was in, where the
    # search has enough freedom to fit the noise
    u = C.synthetic_universe("mond", 31, n_sys=24, n_r=8, n_t=4).data
    X, names, _ = C.build_bank(u, ("scalar", "radial", "gal", "tensor",
                                   "nonlocal"))
    y = u.source.value - u.source.model          # residual against the RAR
    split = C._split_labels(u.sysid, seed=0)

    # a small search, on TRAIN only -- exactly what the machinery does
    K = 6
    idx, train_best = best_subset_on_train(X, y, split, k=K,
                                           rng=np.random.default_rng(3),
                                           ntry=2000 if FAST else 20000)
    print(f"   {int((split=='train').sum())} train / "
          f"{int((split=='blind').sum())} blind points, bank of "
          f"{X.shape[1]-1} atoms")
    print(f"   searched subsets on train; best k={K} law uses "
          f"{[names[i] for i in idx[1:]]}")

    base = C.SplitData(X[:, :1], y, split, atoms=("1",), name="RAR+offset")
    mb = base.fit()
    rb_tr, rb_bl = base.train_rms(mb), base.evaluate(mb)["rms"]

    cand = C.SplitData(X[:, idx], y, split,
                       atoms=tuple(names[i] for i in idx), name="candidate")
    mc = cand.fit()
    rc_tr, rc_bl = cand.train_rms(mc), cand.evaluate(mc)["rms"]

    refit = C._unguarded_refit_on_holdout(X[:, idx], y, split)

    g_tr = 100 * (rb_tr - rc_tr) / rb_tr
    g_bl = 100 * (rb_bl - rc_bl) / rb_bl
    g_bad = 100 * (rb_bl - refit) / rb_bl
    print(f"   baseline (RAR + a global offset)  train {rb_tr:.5f}   "
          f"blind {rb_bl:.5f} dex")
    print(f"   candidate, coefficients FROZEN     train {rc_tr:.5f} "
          f"({g_tr:+.2f}%)   blind {rc_bl:.5f} ({g_bl:+.2f}%)")
    print(f"   candidate, coefficients RE-SOLVED ON BLIND        "
          f"blind {refit:.5f} ({g_bad:+.2f}%)")
    flips = (g_bl < 0 <= g_bad)
    print(f"   the bug is worth {g_bad - g_bl:+.2f} percentage points"
          + ("; it turns a REFUTATION into a DISCOVERY" if flips else
             "; here it inflates rather than flips, which is the same bug at a "
             "smaller amplitude"))
    print(f"   (Run J: +2.17% re-solved vs -3.73% frozen, a 5.90 pp swing)")

    # the guard: the wrong thing cannot be expressed
    blocked = {}
    for label, fn in (
            ("attribute .blind", lambda: cand.blind),
            ("attribute .y", lambda: cand.y),
            ("second touch of blind", lambda: cand.evaluate(mc)),
            ("evaluate raw arrays", lambda: cand.evaluate(np.zeros(len(idx)))),
            ("evaluate a model fitted elsewhere", lambda: cand.evaluate(mb)),
            ("max_touches>1 with no reason",
             lambda: C.SplitData(X[:, :1], y, split, max_touches=5)),
    ):
        try:
            fn()
            blocked[label] = "NOT BLOCKED"
        except (C.SealedHoldoutError, C.FrozenSealError, ValueError) as e:
            blocked[label] = type(e).__name__
        print(f"      {label:<36} -> {blocked[label]}")
    tamper = "NOT BLOCKED"
    try:
        mc.coef[0] = 1e9
    except ValueError:
        tamper = "read-only array"
    print(f"      {'mutate frozen coefficients':<36} -> {tamper}")
    forge = "NOT BLOCKED"
    try:
        object.__setattr__(mc, "_coef", np.zeros_like(mc.coef))
        cand._check(mc)
    except C.FrozenSealError:
        forge = "FrozenSealError (HMAC seal)"
    print(f"      {'forge coefficients past __setattr__':<36} -> {forge}")

    assert g_bad > g_bl, "the anti-pattern did not inflate the blind gain"
    assert all(v != "NOT BLOCKED" for v in blocked.values())
    assert tamper != "NOT BLOCKED" and forge != "NOT BLOCKED"
    return _rec("control_9_frozen_coefficients", {
        "atoms": [names[i] for i in idx],
        "baseline_train": rb_tr, "baseline_blind": rb_bl,
        "frozen_train": rc_tr, "frozen_blind": rc_bl,
        "refit_on_blind": refit,
        "gain_train_pct": g_tr, "gain_blind_frozen_pct": g_bl,
        "gain_blind_refit_pct": g_bad,
        "bug_size_pp": g_bad - g_bl, "sign_flipped": flips,
        "n_train_points": int((split == "train").sum()),
        "n_blind_points": int((split == "blind").sum()),
        "n_atoms_in_bank": int(X.shape[1] - 1), "K": K,
        "run_J_reference": {"refit_pct": 2.17, "frozen_pct": -3.73,
                            "swing_pp": 5.90},
        "blocked": blocked, "tamper": tamper, "forge": forge,
        "audit": cand.report()})


# ==========================================================================
#  driver
# ==========================================================================
TESTS = [test_residual_null, test_residual_null_field, test_cluster_controls,
         test_smoothed_field, test_parameter_responsiveness,
         test_exchangeability, test_shared_denominator, test_eiv_unbiased,
         test_frozen_coefficients, test_known_laws, test_tensor_fpr]


def main():
    t0 = time.time()
    print("=" * 78)
    print("CONTROL HARNESS VALIDATION -- work/wellnet-2026-09/controls")
    print("=" * 78)
    status = {}
    for fn in TESTS:
        t1 = time.time()
        try:
            fn()
            status[fn.__name__] = {"passed": True,
                                   "seconds": round(time.time() - t1, 2)}
        except Exception as exc:
            status[fn.__name__] = {"passed": False, "error": repr(exc),
                                   "seconds": round(time.time() - t1, 2)}
            print(f"\n   !! {fn.__name__} FAILED: {exc!r}")
    RESULTS["_status"] = status
    RESULTS["_meta"] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "fast_mode": FAST, "numpy": np.__version__,
        "python": sys.version.split()[0],
        "total_seconds": round(time.time() - t0, 1),
        "n_passed": sum(1 for v in status.values() if v["passed"]),
        "n_tests": len(status)}
    with open(OUT, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(C._j(RESULTS), fh, indent=1)
    hdr("SUMMARY")
    for k, v in status.items():
        print(f"   {'PASS' if v['passed'] else 'FAIL'}  {k:<34} "
              f"{v['seconds']:7.2f}s"
              + ("" if v["passed"] else "   " + v["error"][:70]))
    print(f"\n   {RESULTS['_meta']['n_passed']}/{len(status)} passed in "
          f"{RESULTS['_meta']['total_seconds']}s")
    print(f"   written: {OUT}")
    return 0 if all(v["passed"] for v in status.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
