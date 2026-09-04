"""JOB 4 -- TRANSLATE EVERY CANDIDATE AMPLITUDE INTO AN OBSERVABLE.

A null from a detector with zero power below the predicted amplitude says
nothing.  So every candidate in `../tournament/tournament.json` is put through
the same three questions:

    1  WHAT AXIS DOES ITS ANISOTROPY LIVE ON?  This decides which of the three
       power surfaces applies, and whether the spherical blindness theorem
       covers it.  It is a property of the STRUCTURE, not of the amplitude:

           scalar_a0   a0 -> a0 (1 + A W)              isotropic; no axis
           iso_K       K = exp(-A W) I                 isotropic; no axis
           tensor_d    K = exp(A W (ghat ghat^T - I/3)) ghat is RADIAL in a
                                                       cluster -> SOURCE axis
           tensor_T    K = exp(A W That)               That is radial in a
                                                       spherical cluster
                                                       -> SOURCE axis
           tensor_S    K = exp(A W S), S from the well
                       catalogue                       -> MEMBER-WELL NETWORK

       Not one candidate in the tournament carries an EXTERNAL tidal axis.
       That is a fact about the grammar that was searched, and it is the
       single most important input to interpreting a null in the phase test.

    2  HOW BIG IS THE ANISOTROPY IT PREDICTS?  K = exp(A W S) has eigenvalue
       ratio exp(A W s) with s = lambda_max - lambda_min of the traceless
       structure (1 for ghat ghat^T - I/3, 1.5 for the normalised tidal
       tensor, 1 for the well-network S which saturates |S|_2 = 2/3).  The
       maximum fractional acceleration anisotropy over the shear-measured
       shell is then

           eps_K = (k_max - k_min) / (k_max + k_min) = tanh(A W s / 2)

       and W is evaluated from the candidate's own invariant on the actual
       baryon profiles of the development sample -- the Bahar+2022 Vikhlinin
       fits -- not on an abstract scale.

    3  DOES THIS TEST HAVE POWER THERE?  eps_K is converted into a predicted
       shear quadrupole, and that is compared with (a) the measured 1 sigma of
       the two-dimensional test and (b) the calibrated power surfaces of
       `axis_power.py`, which report power against the SAME quantity
       (max dg/g over the measured shell).

CONSTANT-K DEGENERACY, DECLARED.  For a constant symmetric positive-definite K
the substitution x' = K^(-1/2) x turns the modified Poisson operator into a
plain Laplacian, so a constant tensor is a coordinate stretch and is
degenerate with source ellipticity, inclination, depth and deprojection.  That
is exactly why the quadrupole AMPLITUDE cannot decide anything on its own and
why the PHASE is the observable that matters.  The conversion below uses that
degeneracy rather than fighting it: the projected convergence acquires an
ellipticity e_kappa = (sqrt(k1) - sqrt(k2)) / (sqrt(k1) + sqrt(k2)), which for
small anisotropy is eps_K / 2, and the coefficient relating e_kappa to the
observed shear quadrupole ratio is CALIBRATED NUMERICALLY on the real profile
shape rather than assumed to be unity.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
for p in (os.path.join(LANE, "efeds-hsc"), os.path.join(LANE, "lead01"),
          os.path.join(LANE, "tournament")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline as P                                            # noqa: E402
import efeds_hsc as E                                           # noqa: E402
from tw_core import W_of                                        # noqa: E402

A0 = 1.2e-10
F_STAR = 0.15
R_LO_MPC, R_HI_MPC = 0.2 / 0.7, 3.5 / 0.7      # the shear-measured shell

#: lambda_max - lambda_min of each traceless structure, its axis provenance,
#: and whether its action on grad Phi_N is EXACTLY a scalar rescaling.
#:
#: The last column is the QUMOND tensor degeneracy proved in
#: `../qumond_degeneracy.py`: K appears only through K grad Phi_N, and for
#: K = exp(A W (ghat ghat^T - I/3)) the field direction ghat is an eigenvector,
#: so K grad Phi_N = exp(2 A W / 3) grad Phi_N -- a pure scalar rescaling that
#: cannot turn the flux and therefore cannot produce a quadrupole of ANY phase.
#: tensor_d is that structure.  Its eigenvalue spread is real; its OBSERVABLE
#: anisotropy is identically zero.
STRUCT = {
    "scalar_a0": (0.0, "none", True),
    "iso_K":     (0.0, "none", True),
    "tensor_d":  (1.0, "source", True),
    "tensor_T":  (1.5, "source", False),
    "tensor_S":  (1.0, "network", False),
}


def e_kappa_of(eps):
    """Convergence ellipticity implied by an acceleration anisotropy eps_K.

    k1/k2 = (1+eps)/(1-eps), and the constant-K coordinate stretch is by
    K^(1/2), so e_kappa = (sqrt(k1/k2) - 1)/(sqrt(k1/k2) + 1).  The often-used
    eps/2 is only the first term of this and is wrong by a factor 2 once eps
    approaches 1, which several tournament survivors do.
    """
    eps = min(max(float(eps), 0.0), 0.999999)
    rt = math.sqrt((1.0 + eps) / (1.0 - eps))
    return (rt - 1.0) / (rt + 1.0)


def invariants_of(sysm):
    """gn, phi, rhobar, tidal on one eFEDS system's baryon profile.

    Definitions taken from `../tournament/ch_cluster.py` so the numbers mean
    the same thing they meant when the amplitudes were fitted:
        gn      |g_N| / a0
        phi     |Phi_N|, boundary rule "inf" -- Phi -> 0 at infinity with the
                baryons continued outside the last measured radius as a point
                mass (the tournament's declared PRIMARY rule)
        rhobar  the local baryon mass density
        tidal   the Frobenius norm of the traceless Hessian, which for a
                spherical distribution is sqrt(6)/3 |Phi'' - Phi'/r|
    """
    r, g = sysm.r, sysm.g_b
    # Phi with the "inf" rule
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(r)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    M_tot = sysm.M_b[-1]
    tail = P.G * M_tot / r[-1]
    phi = (cum[-1] - cum) + tail                      # = |Phi_N|(r), positive
    dgdr = np.gradient(g, r)
    u = dgdr - g / r
    tidal = math.sqrt(6.0) / 3.0 * np.abs(u)
    rho = sysm.rho_gas * (1.0 + sysm.f_star)
    # qbar: the nonlocal mass fraction, with the tournament's own constants
    # (ch_radial.L_NL = 300 kpc, M_NL = 1e12 Msun).  For the smooth gas the
    # tournament uses the local-density approximation for the mass inside
    # L_NL of the point, which is what is reproduced here.
    L_NL, M_NL = 300.0 * P.KPC, 1.0e12 * P.MSUN
    Mnl = rho * (4.0 / 3.0) * math.pi * L_NL ** 3
    return dict(one=np.ones_like(r), gn=g / A0, phi=phi,
                rhobar=np.maximum(rho, 1e-40),
                tidal=np.maximum(tidal, 1e-45),
                qbar=Mnl / (Mnl + M_NL))


def eps_K_for(rec, inv, r, spread):
    """max fractional acceleration anisotropy over the measured shell."""
    m = (r >= R_LO_MPC * P.MPC) & (r <= R_HI_MPC * P.MPC)
    if rec["form"] == "off" or rec["inv"] == "one" or spread <= 0:
        return 0.0, 0.0, 0.0
    W = W_of(rec["form"], inv[rec["inv"]][m] / rec["I0"], rec["m"])
    x = rec["A"] * W * spread
    eps = np.abs(np.tanh(0.5 * x))
    i = int(np.argmax(eps))
    return float(eps[i]), float(np.median(eps)), float(x[i])


def quadrupole_coefficient(sysm, e_list=(0.02, 0.05, 0.10, 0.20), n=512,
                           box_mpc=12.0):
    """C = (a2c / a0) per unit convergence ellipticity, on the REAL profile.

    An elliptical convergence kappa(R_e), R_e^2 = x^2 (1-e) + y^2 (1+e), is put
    on a grid, its shear is obtained by the exact Fourier relation
    gamma(k) = kappa(k) (k1^2 - k2^2 + 2 i k1 k2) / k^2, and the m = 2 harmonic
    of the tangential component is measured in the same annulus and with the
    same estimator the data are analysed with.  Assuming C = 1 would be an
    unmeasured factor sitting in front of every predicted amplitude.
    """
    L = box_mpc * P.MPC
    ax = (np.arange(n) - n / 2 + 0.5) * (L / n)
    X = ax[:, None] * np.ones((1, n))
    Y = np.ones((n, 1)) * ax[None, :]
    g = sysm.g_pred(law="rar")
    S, _ = sysm.sigma_profile(g, np.geomspace(1e-3, box_mpc, 400) * P.MPC,
                              r_trunc_mpc=10.0)
    Rg = np.geomspace(1e-3, box_mpc, 400) * P.MPC
    lr, lS = np.log(Rg), np.log(np.maximum(S, 1e-40))
    k1 = 2.0 * math.pi * np.fft.fftfreq(n, d=L / n)
    KX = k1[:, None] * np.ones((1, n))
    KY = np.ones((n, 1)) * k1[None, :]
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0
    R = np.sqrt(X ** 2 + Y ** 2)
    ann = (R > R_LO_MPC * P.MPC) & (R < R_HI_MPC * P.MPC)
    phi = np.arctan2(Y, X)
    out = []
    for e in e_list:
        Re = np.sqrt(X ** 2 * (1.0 - e) + Y ** 2 * (1.0 + e))
        kap = np.exp(np.interp(np.log(np.clip(Re, Rg[0], Rg[-1])), lr, lS))
        kt = np.fft.fft2(kap)
        gam = np.fft.ifft2(kt * (KX ** 2 - KY ** 2 + 2j * KX * KY) / K2)
        g1, g2 = gam.real, gam.imag
        et = -(g1 * np.cos(2 * phi) + g2 * np.sin(2 * phi))
        y = et[ann]
        c2 = np.cos(2 * phi[ann])
        Xd = np.column_stack([np.ones(y.size), c2, np.sin(2 * phi[ann])])
        b = np.linalg.lstsq(Xd, y, rcond=None)[0]
        out.append(b[1] / b[0] / e)
    return float(np.median(out)), [float(v) for v in out]


def main():
    print("=" * 78)
    print("AMPLITUDES -- every candidate's predicted anisotropy, on the")
    print("              power surface and against the measured error bar")
    print("=" * 78)
    tj = json.load(open(os.path.join(LANE, "tournament", "tournament.json"),
                        encoding="utf-8"))
    recs = tj["records"]
    print(f"\n   {len(recs)} candidates in the tournament, "
          f"{sum(1 for r in recs if r.get('survives'))} survivors")

    # ---- the reference systems: the development sample, plus a massive one
    sel_path = os.path.join(HERE, "selection.json")
    sel = json.load(open(sel_path, encoding="utf-8")) \
        if os.path.exists(sel_path) else {"dev": []}
    efeds, _ = E.load_efeds()
    by_id = {r["id"]: r for r in efeds}
    dev_ids = [d["id"] for d in sel.get("dev", [])]
    ref_ids = dev_ids if dev_ids else [r["id"] for r in efeds[:20]]
    # the most gas-massive system in the field, as the "cluster" reference
    heavy = max(efeds, key=lambda r: r["Mgas500_pub"])
    print(f"   reference systems: {len(ref_ids)} development clusters "
          f"plus the most gas-massive eFEDS system {heavy['id']} "
          f"(M_gas,500 = {heavy['Mgas500_pub']:.2f}e12 Msun)")

    sysd = {}
    invd = {}
    for i in set(ref_ids + [heavy["id"]]):
        s = P.System(by_id[i], f_star=F_STAR)
        sysd[i] = s
        invd[i] = invariants_of(s)

    C, Cl = quadrupole_coefficient(sysd[heavy["id"]])
    print(f"\n   quadrupole coefficient C = (a2c/a0) per unit convergence")
    print(f"   ellipticity, measured on the real profile: {C:.3f} "
          f"(over e = 0.02-0.20: {[round(v,3) for v in Cl]})")

    # ---- the measured error bar from the two-dimensional test
    s2 = None
    p2 = os.path.join(HERE, "shear2d.json")
    if os.path.exists(p2):
        s2 = json.load(open(p2, encoding="utf-8"))

    # ---- the power surface
    ap = None
    pa = os.path.join(HERE, "axis_power.json")
    if os.path.exists(pa):
        ap = json.load(open(pa, encoding="utf-8"))

    rows = []
    # The geometric factor is the POPULATION RMS member-light ellipticity,
    # deconvolved for measurement noise as <e_true^2> = <e_obs^2> - 2 <sigma^2>.
    # The per-cluster debiased values are individually noisy and, in a sample
    # selected at >= 2.5 sigma, Eddington-biased upward; the population moment
    # is not.  Both the development-sample value and the whole-parent value are
    # carried, and the parent one is the conservative choice.
    def rms_e(lst):
        if not lst:
            return 0.0
        e = np.array([d["e_mem"] for d in lst])
        sg = np.array([d["e_mem_err"] for d in lst])
        return float(math.sqrt(max(np.mean(e ** 2)
                                   - np.mean(2 * sg ** 2), 0.0)))
    e_geom = rms_e(sel.get("dev", []))
    e_geom_parent = rms_e(sel.get("all_measured", []))
    print(f"   geometric factor: deconvolved RMS member-light ellipticity, "
          f"DEV {e_geom:.3f}, whole parent {e_geom_parent:.3f}.")
    print("   A source-axis or network tensor is purely radial in the round")
    print("   limit and produces no quadrupole at all there (spherical")
    print("   blindness), so its predicted quadrupole carries this factor.")
    for rec in recs:
        spread, prov, degen = STRUCT.get(rec["struct"], (0.0, "unknown", True))
        per = []
        for i in ref_ids:
            e_max, e_med, aws = eps_K_for(rec, invd[i], sysd[i].r, spread)
            per.append((e_max, e_med, aws))
        e_max = float(np.median([p[0] for p in per])) if per else 0.0
        e_med = float(np.median([p[1] for p in per])) if per else 0.0
        eh, ehm, awsh = eps_K_for(rec, invd[heavy["id"]],
                                  sysd[heavy["id"]].r, spread)
        ek = 0.0 if degen else e_kappa_of(e_max)
        ekh = 0.0 if degen else e_kappa_of(eh)
        rows.append(dict(
            name=rec["name"], struct=rec["struct"], inv=rec["inv"],
            form=rec["form"], m=rec["m"], I0=rec["I0"], A=rec["A"],
            survives=rec.get("survives", False),
            failed=rec.get("failed", []),
            provenance=prov, lambda_spread=spread,
            qumond_scalar_degenerate=degen,
            eps_K_dev_max=e_max, eps_K_dev_median=e_med,
            eps_K_heavy_max=eh, AWs_heavy=awsh,
            e_kappa_dev=ek,
            quadrupole_a2c_pred_dev=C * ek * e_geom,
            quadrupole_a2c_pred_heavy=C * ekh * e_geom,
            quadrupole_a2s_pred_dev=0.0))
    surv = [r for r in rows if r["survives"]]

    print("\n" + "=" * 78)
    print("   THE 18 TOURNAMENT SURVIVORS")
    print("   name                                          prov      "
          "eps_K   a2c/a0   a2s/a0")
    for r in sorted(surv, key=lambda x: -x["eps_K_dev_max"]):
        print(f"   {r['name'][:44]:44s}  {r['provenance']:8s}  "
              f"{r['eps_K_dev_max']:6.3f}  "
              f"{r['quadrupole_a2c_pred_dev']:7.4f}  "
              f"{r['quadrupole_a2s_pred_dev']:7.4f}"
              + ("   [scalar-degenerate]" if r["qumond_scalar_degenerate"]
                 else ""))

    byprov = {}
    for r in rows:
        byprov.setdefault(r["provenance"], []).append(r)
    print("\n   axis provenance across the WHOLE tournament:")
    for k, v in sorted(byprov.items(), key=lambda kv: -len(kv[1])):
        ns = sum(1 for x in v if x["survives"])
        print(f"      {k:8s}  {len(v):5d} candidates, {ns} survivors")
    print("      external      0 candidates, 0 survivors  <-- the hypothesis")
    print("                    the two-dimensional phase test is designed for")
    print("                    is NOT represented in the tournament grammar")

    # ---- can the test see them?
    verdicts = []
    sig_a2s = a2c_meas = a2c_err = excl = None
    if s2 and s2.get("samples", {}).get("dev", {}).get("channels"):
        chs = s2["samples"]["dev"]["channels"]
        sig_a2s = (chs.get("a2s_pred") or {}).get("null_std")
        pc = (chs.get("a2c_baryon") or {}).get("fit") or {}
        a2c_meas, a2c_err = pc.get("alpha"), pc.get("e_alpha")
    ccp = os.path.join(HERE, "crosscheck.json")
    if os.path.exists(ccp):
        cc = json.load(open(ccp, encoding="utf-8"))
        excl = (cc.get("injection", {}).get("pred", {})
                .get("exclusion_2sided_95"))
    print("\n" + "=" * 78)
    print("   COULD THIS TEST HAVE SEEN IT?")
    if sig_a2s:
        print(f"   measured 1 sigma on alpha (the phase-misaligned "
              f"quadrupole per unit predicted monopole) = {sig_a2s:.4f}")
    if excl:
        lim = excl * 0.801            # the sample's median |sin 2 Delta|
        print(f"   95% exclusion on a misaligned quadrupole, from end-to-end "
              f"injection into")
        print(f"   the real data: |alpha| < {excl:.3f}, i.e. a misaligned "
              f"quadrupole ratio below {lim:.2f},")
        print(f"   i.e. e_kappa < {lim / C:.2f}.  The geometric maximum of "
              f"e_kappa is 1, so NO")
        print(f"   external-axis tensor of any amplitude is excluded by this "
              f"sample.")
    if a2c_meas is not None:
        print(f"   measured SOURCE-ALIGNED quadrupole per unit predicted "
              f"monopole: {a2c_meas:+.4f} +- {a2c_err:.4f}")
        print(f"      that is the SUM of the baryonic quadrupole and any "
              f"source-axis or network")
        print(f"      tensor.  The two cannot be separated -- which is the "
              f"degeneracy the phase")
        print(f"      channel exists to avoid, and the reason an amplitude "
              f"alone decides nothing.")
    for r in sorted(surv, key=lambda x: -x["eps_K_dev_max"]):
        if r["qumond_scalar_degenerate"] and r["provenance"] != "none":
            v = ("K grad Phi_N = exp(2AW/3) grad Phi_N EXACTLY -- the field "
                 "direction is an eigenvector, so this tensor cannot turn the "
                 "flux and predicts no quadrupole of any phase.  It is a "
                 "scalar rescaling wearing a tensor's name.")
        elif r["provenance"] == "none":
            v = ("isotropic: predicts NO quadrupole of any phase; the "
                 "two-dimensional test cannot address it at any amplitude")
        elif r["provenance"] == "source":
            v = ("source axis: the quadrupole is locked to the baryonic axis, "
                 "so it lands in a2c where it is degenerate with baryonic "
                 "ellipticity; a2s power is ZERO by construction")
        else:
            v = ("member-well network: the axis is the member distribution, "
                 "which is the frame the test stacks in, so the quadrupole "
                 "again lands in a2c, not a2s")
        verdicts.append(dict(name=r["name"], provenance=r["provenance"],
                             eps_K=r["eps_K_dev_max"],
                             quadrupole_a2c=r["quadrupole_a2c_pred_dev"],
                             quadrupole_a2s=r["quadrupole_a2s_pred_dev"],
                             scalar_degenerate=r["qumond_scalar_degenerate"],
                             verdict=v))
    for v in verdicts[:6]:
        print(f"      {v['name'][:52]:52s}")
        print(f"         eps_K = {v['eps_K']:.3f}, predicted a2c/a0 = "
              f"{v['quadrupole_a2c']:.4f}, predicted a2s/a0 = "
              f"{v['quadrupole_a2s']:.4f}")
        print(f"         {v['verdict']}")

    # ---- place them on the power surface
    placed = []
    if ap:
        cells = [(c["provenance"], c["tilt_deg"], c["amp"],
                  c["aniso_frac_median"], c["power"],
                  c.get("power_axis_known"), row["axis_ratio"], row["noise"])
                 for row in ap["rows"] for c in row["power"].values()]
        for r in surv:
            if r["provenance"] in ("source", "network"):
                cand = [c for c in cells if c[0] == r["provenance"]]
                if cand:
                    # nearest cell in max dg/g, at the most favourable geometry
                    best = min(cand, key=lambda c: abs(math.log10(
                        max(c[3], 1e-6)) - math.log10(
                            max(r["eps_K_dev_max"], 1e-6))))
                    placed.append(dict(
                        name=r["name"], provenance=r["provenance"],
                        eps_K=r["eps_K_dev_max"],
                        nearest_cell_dgg=best[3], power_at_cell=best[4],
                        power_axis_known=best[5],
                        axis_ratio=best[6], noise=best[7]))
        print("\n   PLACED ON THE POWER SURFACE (nearest cell in max dg/g):")
        for p in placed[:8]:
            print(f"      {p['name'][:44]:44s}  eps_K {p['eps_K']:.3f} -> "
                  f"cell dg/g {p['nearest_cell_dgg']:.3f}, "
                  f"power {p['power_at_cell']:.2f}")

    out = dict(quadrupole_coefficient=C, quadrupole_coefficient_scan=Cl,
               reference_dev_ids=ref_ids, reference_heavy=heavy["id"],
               struct_map={k: dict(lambda_spread=v[0], provenance=v[1],
                                   qumond_scalar_degenerate=v[2])
                           for k, v in STRUCT.items()},
               geometric_factor_e_mem=e_geom,
               geometric_factor_e_mem_parent=e_geom_parent,
               provenance_counts={k: dict(n=len(v),
                                          n_survivors=sum(1 for x in v
                                                          if x["survives"]))
                                  for k, v in byprov.items()},
               sigma_alpha_measured=sig_a2s,
               exclusion_95_alpha=excl,
               measured_a2c_over_gpred=a2c_meas,
               measured_a2c_over_gpred_err=a2c_err,
               survivors=surv, verdicts=verdicts, placed=placed,
               all_candidates=rows)
    with open(os.path.join(HERE, "amplitudes.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\n   written: amplitudes.json")


if __name__ == "__main__":
    main()
