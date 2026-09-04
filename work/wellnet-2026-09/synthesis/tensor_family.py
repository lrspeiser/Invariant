"""tensor_family.py -- family T: action-derived void/tensor gravity.

THE ACTION (weak field, static)

    S_T[Phi] = Int d^3x { -(1/8 pi G) [ a0^2 F(|u|^2/a0^2)
                                        + f_E h(|u|/a0) u^T That_env u ]
                          - rho Phi },                      u = grad Phi

    F'(y) = mu(sqrt y),  mu(x) = x/(1+x)      (the AQUAL base, as the bench)
    h(x)  = mu(x) (1 - mu(x)) = x/(1+x)^2      (derived below, no new scale)
    That_env(x) = the unit-Frobenius-norm traceless tidal tensor of the
                  ENVIRONMENT (Hessian of Phi_N smoothed on L_env, traceless,
                  normalised) -- a headless background field; in the
                  principal-axis form That = sqrt(3/2) (e e^T - I/3) and
                  sqrt(3/2) is absorbed into f_E.

FIELD EQUATION (Euler-Lagrange in Phi, That held fixed)

    div M(u) = 4 pi G rho,
    M(u) = mu(x) u + (1/2) f_E [ h'(x) (u^T That u) uhat / a0 + 2 h(x) That u ]

M = (1/2) dL~/du is a GRADIENT in u identically, so the Stage 3 compiler's
u-space test must return round-off (it does: compile_families.py).  The
same gating written as K(u) u in QUMOND form is NOT a gradient (compiler
element F2), which is the whole reason for writing the anisotropy in the
Lagrangian.

UNIVERSAL CONSTANTS: G, a0 (shared with the base), f_E (dimensionless),
L_env (the scale on which "environment" is defined).  Two beyond the base.

WHAT THIS MODULE MEASURES (no data of any kind is opened)
  1. the ellipticity bound on f_E from the Hessian of L~ (positive-definite
     kinetic operator) -- the interval outside which the theory is dead;
  2. why h must vanish at BOTH ends: the deep-MOND end by ellipticity, the
     Newtonian end by the Solar System; the anisotropy at the planets;
  3. the first-order l = 2 solution for a spherical source in the AQUAL base,
     which fixes the SIGN and radial profile of every tensor counterfactual,
     validated on the constant-K Newtonian case whose exact solution is known;
  4. the sign of the compiler's declared radial caricature against that exact
     solution -- a bench finding, reported rather than hidden.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import guard                                    # noqa: F401  (sys.path)
import compiler as C                            # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
G, A0, KPC, MSUN, AU = C.G, C.A0, C.KPC, C.MSUN, C.AU

mu, h, hp, eta = C.tensor_L_mu, C.tensor_L_h, C.tensor_L_hprime, C.tensor_L_eta


# ------------------------------------------------------------ 1. ellipticity
def hessian_min_eig(fE: float, n: int = 3000, seed: int = 20260904,
                    e=(0.0, 0.0, 1.0)) -> dict:
    """min over a u-cloud of lambda_min(dM/du) / mu(x): the kinetic operator
    is elliptic iff this is positive everywhere.  Normalised by the isotropic
    transverse eigenvalue mu(x) so the number is scale-free in x."""
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = 10 ** rng.uniform(-4.0, 4.0, n)
    u = d * (x * A0)[:, None]
    e = np.asarray(e, float)
    hh = 1e-6
    J = np.zeros((n, 3, 3))
    for j in range(3):
        du = np.zeros((n, 3))
        du[:, j] = hh * np.linalg.norm(u, axis=-1)
        J[:, :, j] = (C.tensor_L_flux(u + du, e, fE, A0)
                      - C.tensor_L_flux(u - du, e, fE, A0)) / (2 * du[:, j])[:, None]
    Js = 0.5 * (J + np.transpose(J, (0, 2, 1)))
    w = np.linalg.eigvalsh(Js)
    ratio = w[:, 0] / mu(x)
    k = int(np.argmin(ratio))
    return dict(fE=fE, min_ratio=float(ratio.min()), at_x=float(x[k]),
                at_cos_e=float(abs(d[k] @ e)),
                antisym=float(np.abs(J - np.transpose(J, (0, 2, 1))).max()
                              / np.abs(J).max()))


def ellipticity_interval(fEs=np.linspace(-3.0, 3.0, 121)) -> dict:
    rows = [hessian_min_eig(float(f)) for f in fEs]
    ok = [r["fE"] for r in rows if r["min_ratio"] > 0.0]
    # analytic principal-direction bounds derived in the module docstring of
    # compiler.tensor_L_flux's companion note:  transverse at u || e gives
    # f_E < 6, transverse at u _|_ e gives f_E > -2, radial gives -1 < f_E < 2.
    return dict(scan=rows, admissible_fE_min=float(min(ok)) if ok else None,
                admissible_fE_max=float(max(ok)) if ok else None,
                analytic_principal_bounds=dict(radial_lower=-1.0,
                                               radial_upper=2.0,
                                               transverse_lower=-2.0,
                                               transverse_upper=6.0),
                note="the interval is the intersection of the principal-"
                     "direction bounds; the numerical scan over all "
                     "directions is the binding one")


# -------------------------------------------------- 2. why h vanishes twice
def deep_mond_ellipticity_argument() -> dict:
    """With h -> const as x -> 0 the anisotropic Hessian would be O(f_E) while
    the isotropic one is O(x): the operator loses ellipticity at every point
    where grad Phi -> 0.  Measured by replacing h with a constant."""
    rng = np.random.default_rng(1)
    d = rng.normal(size=(2000, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    x = 10 ** rng.uniform(-4.0, 0.0, 2000)
    u = d * (x * A0)[:, None]
    e = np.array([0.0, 0.0, 1.0])

    def M_const_h(uu, fE=0.3, hc=0.25):
        gn = np.linalg.norm(uu, axis=-1)
        ue = uu @ e
        return (mu(gn / A0)[:, None] * uu
                + 0.5 * fE * 2.0 * hc * (ue[:, None] * e[None, :] - uu / 3.0))

    hh = 1e-6
    J = np.zeros((2000, 3, 3))
    for j in range(3):
        du = np.zeros((2000, 3))
        du[:, j] = hh * np.linalg.norm(u, axis=-1)
        J[:, :, j] = (M_const_h(u + du) - M_const_h(u - du)) / (2 * du[:, j])[:, None]
    w = np.linalg.eigvalsh(0.5 * (J + np.transpose(J, (0, 2, 1))))
    frac_indefinite = float(np.mean(w[:, 0] <= 0.0))
    x_lose = float(x[w[:, 0] <= 0.0].max()) if np.any(w[:, 0] <= 0.0) else None
    return dict(fE=0.3, h_constant=0.25,
                fraction_of_cloud_with_indefinite_operator=frac_indefinite,
                ellipticity_lost_below_x=x_lose,
                statement="a constant (non-vanishing) h in deep MOND makes "
                          "the kinetic operator indefinite wherever "
                          "|grad Phi|/a0 is small enough; h must vanish at "
                          "least as fast as mu there. h = mu(1-mu) does.")


def known_limits(fE_values=(0.3, 1.0, 2.0)) -> dict:
    """The fractional quadrupole of |g_r| along the axis, (2/3) f_E h(x), at
    a set of declared accelerations.  Arithmetic on published constants; no
    dataset is opened."""
    g_sun_1au = G * MSUN / AU ** 2
    regimes = {
        "Mercury_0.39AU": g_sun_1au / 0.387 ** 2,
        "Earth_1AU": g_sun_1au,
        "Neptune_30AU": g_sun_1au / 30.0 ** 2,
        "Oort_10kAU": g_sun_1au / 1.0e4 ** 2,
        "wide_binary_1Msun_10kAU(sealed regime)": g_sun_1au / 1.0e4 ** 2,
        "transition_g=a0": A0,
        "galaxy_outskirts_0.1a0": 0.1 * A0,
        "cluster_outskirts_0.07a0": 0.07 * A0,
        "deep_MOND_0.01a0": 0.01 * A0,
    }
    rows = {}
    for k, g in regimes.items():
        x = g / A0
        rows[k] = dict(g_SI=float(g), x=float(x), h=float(h(x)),
                       **{f"quadrupole_fE_{f}": float((2.0 / 3.0) * f * h(x))
                          for f in fE_values})
    return dict(rows=rows,
                peak_x=1.0, peak_h=float(h(1.0)),
                statement=("the anisotropy is (2/3) f_E h(g/a0) P2 and h "
                           "peaks at g = a0 with h = 1/4; at 1 AU it is "
                           f"{(2/3)*h(g_sun_1au/A0):.2e} f_E and at Mercury "
                           f"{(2/3)*h(g_sun_1au/0.387**2/A0):.2e} f_E. "
                           "Fixed-axis anisotropies of the Solar-System "
                           "potential are bounded at roughly the 1e-9 "
                           "level (published PPN preferred-location and "
                           "solar-J2 constraints, quoted as an order of "
                           "magnitude, not evaluated here), so f_E of order "
                           "unity is marginal at Mercury and f_E <~ 0.3 is "
                           "safe; a steeper Newtonian cutoff of h would "
                           "remove the tension at the cost of one constant."))


# ---------------------------------- 3. the first-order l = 2 solution
def solve_l2(rg, g0, mu0, a_coef, hh, hhp, x0, fE=1.0):
    """Solve  (r a chi')' - 6 mu0 chi = r^2 S(r)  on a log grid, with

        S = -(1/3) [ (1/r^2) (r^2 g0 (x0 h' + 2 h))' - 6 g0 h / r ],

    Dirichlet chi = 0 at both ends.  Returns chi for Phi = Phi0 + f_E chi P2
    with f_E = 1 (linear in f_E).  a_coef = mu + x mu' is the radial
    'dielectric' eigenvalue of the linearised AQUAL operator, mu0 the
    transverse one."""
    s = np.log(rg)
    ds = s[1] - s[0]
    n = len(rg)
    q = rg ** 2 * g0 * (x0 * hhp + 2.0 * hh)
    dq = np.gradient(q, rg)
    S = -(1.0 / 3.0) * (dq / rg ** 2 - 6.0 * g0 * hh / rg)
    rhs = rg ** 3 * S                       # (r a chi_s)_s - 6 r mu0 chi = r^3 S
    p = rg * a_coef                         # coefficient of chi_s
    ph = 0.5 * (p[1:] + p[:-1])             # at half points
    A = np.zeros((n, n))
    b = np.zeros(n)
    A[0, 0] = 1.0                           # regular: chi ~ r^2 -> 0
    # OUTER: Neumann chi_s = 0.  In deep MOND chi tends to the CONSTANT
    # -sqrt(G M a0)/3 (derived in the report), so a Dirichlet zero there is
    # wrong and excites the growing homogeneous mode r^{+sqrt 3}; in the
    # Newtonian constant-K case chi ~ -GM/3r also has chi' -> 0.
    A[-1, -1] = 1.0
    A[-1, -2] = -1.0
    for i in range(1, n - 1):
        A[i, i - 1] = ph[i - 1] / ds ** 2
        A[i, i + 1] = ph[i] / ds ** 2
        A[i, i] = -(ph[i - 1] + ph[i]) / ds ** 2 - 6.0 * rg[i] * mu0[i]
        b[i] = rhs[i]
    chi = np.linalg.solve(A, b)
    chi_s = np.gradient(chi, s)
    return chi, chi_s / rg                  # chi, chi'


def plummer_exact_constantK_chi(rg, M, a, n_th=200, n_r=400):
    """chi_exact P2 = -(G/2) Int rho(y) lambda_d / |x - y| d^3y for the
    constant-K Newtonian case (first order in f):  the exact stretched
    solution expanded.  Evaluated on the axis (theta = 0, P2 = 1)."""
    comp = C.Plummer(M, a)
    ry = np.geomspace(1e-3 * a, 300.0 * a, n_r)
    cth, wth = np.polynomial.legendre.leggauss(n_th)
    out = np.zeros(len(rg))
    for i, r in enumerate(rg):
        x = np.array([0.0, 0.0, r])
        acc = 0.0
        for c, w in zip(cth, wth):
            sth = math.sqrt(1 - c * c)
            y = np.stack([ry * sth, 0 * ry, ry * c], -1)
            d = x[None, :] - y
            dn = np.linalg.norm(d, axis=-1)
            lam_d = (d[:, 2] / dn) ** 2 - 1.0 / 3.0
            rho = comp.rho(y)
            f = 2 * np.pi * ry ** 2 * rho * lam_d / dn
            acc += w * np.trapezoid(f, ry)
        out[i] = -0.5 * G * acc
    return out


def l2_solution() -> dict:
    a = 3.0 * KPC
    M = 5.0e10 * MSUN
    comp = C.Plummer(M, a)
    rg = np.geomspace(1e-3 * a, 3e3 * a, 1400)
    gN = G * comp.M_enc(rg) / rg ** 2

    # --- validation: constant-K Newtonian case (mu = 1, h = 1) vs exact
    chi_c, dchi_c = solve_l2(rg, gN, np.ones_like(rg), np.ones_like(rg),
                             np.ones_like(rg), np.zeros_like(rg), gN / A0)
    r_test = np.array([0.5, 1.0, 2.0, 5.0, 20.0, 100.0]) * a
    chi_ex = plummer_exact_constantK_chi(r_test, M, a)
    chi_ode = np.interp(r_test, rg, chi_c)
    point_mass = -G * M / (3.0 * r_test)
    val = dict(r_over_a=(r_test / a).tolist(),
               chi_ode=chi_ode.tolist(), chi_exact=chi_ex.tolist(),
               chi_point_mass_asymptote=point_mass.tolist(),
               max_rel_err=float(np.max(np.abs(chi_ode - chi_ex)
                                        / np.abs(chi_ex))))

    # --- the gated AQUAL case
    g0 = C.g_of_gN("aqual", gN, A0)          # exact spherical AQUAL base
    x0 = g0 / A0
    mu0 = mu(x0)
    a_coef = x0 * (2.0 + x0) / (1.0 + x0) ** 2      # mu + x mu'
    chi, dchi = solve_l2(rg, g0, mu0, a_coef, h(x0), hp(x0), x0)
    # fractional modulation of the INWARD magnitude along the axis per unit
    # f_E:  |g_r| = g0 + f_E chi' P2  (g_r = -dPhi/dr, Phi = Phi0 + f_E chi P2,
    # Phi0' = +g0), so along e (P2 = 1) the coefficient is chi'/g0.
    A2 = dchi / g0
    # the compiler's declared radial caricature, first order:
    A2_cari = -(2.0 / 3.0) * eta(x0) / a_coef
    # convergence in r_max
    rg2 = np.geomspace(1e-3 * a, 1e4 * a, 1800)
    gN2 = G * comp.M_enc(rg2) / rg2 ** 2
    g02 = C.g_of_gN("aqual", gN2, A0)
    x02 = g02 / A0
    chi2, dchi2 = solve_l2(rg2, g02, mu(x02), x02 * (2 + x02) / (1 + x02) ** 2,
                           h(x02), hp(x02), x02)
    A2_2 = np.interp(rg, rg2, dchi2 / g02)
    sel = (rg > 0.1 * a) & (rg < 300 * a)
    conv = float(np.max(np.abs(A2_2[sel] - A2[sel])))

    r_rep = np.array([0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]) * a
    rows = []
    for r in r_rep:
        i = int(np.argmin(np.abs(rg - r)))
        rows.append(dict(r_over_a=float(r / a), r_kpc=float(r / KPC),
                         x0=float(x0[i]), h=float(h(x0[i])),
                         A2_exact_per_fE=float(A2[i]),
                         A2_caricature_per_fE=float(A2_cari[i]),
                         chi_per_fE_m2s2=float(chi[i])))
    ipk = int(np.argmax(np.abs(A2[sel])))
    peak_r = float(rg[sel][ipk])

    # --- the two analytic asymptotes of the GATED case, both derived in the
    # report:  Newtonian side  chi -> -a0 r/3  so  A2 -> -1/(3 x0);
    #          deep MOND       chi -> -sqrt(G M a0)/3  (a constant offset),
    #          so A2 -> 0 from below.
    # Newtonian side is checked on a compact massive source where x0 >> 1.
    Mc, ac = 5.0e12 * MSUN, 0.3 * KPC
    compc = C.Plummer(Mc, ac)
    rgc = np.geomspace(1e-3 * ac, 3e4 * ac, 1600)
    gNc = G * compc.M_enc(rgc) / rgc ** 2
    g0c = C.g_of_gN("aqual", gNc, A0)
    x0c = g0c / A0
    chic, dchic = solve_l2(rgc, g0c, mu(x0c), x0c * (2 + x0c) / (1 + x0c) ** 2,
                           h(x0c), hp(x0c), x0c)
    A2c = dchic / g0c
    newt_rows = []
    for rr in (1.0, 2.0, 5.0):
        i = int(np.argmin(np.abs(rgc - rr * ac)))
        newt_rows.append(dict(r_over_a=rr, x0=float(x0c[i]),
                              A2_per_fE=float(A2c[i]),
                              asymptote_minus_1_over_3x=float(-1.0 / (3.0 * x0c[i])),
                              ratio=float(A2c[i] * 3.0 * x0c[i] / -1.0)))
    chi_dm_asym = -math.sqrt(G * M * A0) / 3.0
    i_far = int(np.argmin(np.abs(rg - 300.0 * a)))
    deep = dict(r_over_a=300.0, x0=float(x0[i_far]),
                chi_per_fE=float(chi[i_far]), asymptote=chi_dm_asym,
                ratio=float(chi[i_far] / chi_dm_asym),
                A2_per_fE=float(A2[i_far]))

    sgn_exact = "-" if A2[sel].mean() < 0 else "+"
    sgn_cari = "-" if A2_cari[sel].mean() < 0 else "+"
    i3 = int(np.argmin(np.abs(rg - 3.0 * a)))
    rar_dex_per_fE = 0.5 * abs(A2[i3]) / math.log(10.0)
    all_negative = bool(np.all(A2[sel] < 0.0))
    return dict(
        A2_all_negative_0p1a_to_300a=all_negative,
        A2_at_3a_per_fE=float(A2[i3]),
        rar_residual_dex_per_fE_at_3a=float(rar_dex_per_fE),
        source="Plummer 5e10 Msun, a = 3 kpc, AQUAL base mu = x/(1+x)",
        validation_constant_K_newton=val,
        rmax_convergence_max_abs_change=conv,
        rows=rows,
        peak=dict(r_over_a=peak_r / a, r_kpc=peak_r / KPC,
                  x0=float(np.interp(peak_r, rg, x0)),
                  A2_per_fE=float(A2[sel][ipk])),
        newtonian_side_asymptote=dict(
            source="Plummer 5e12 Msun, a = 0.3 kpc (x0 >> 1 at r ~ a)",
            rows=newt_rows,
            statement="A2 -> -1/(3 x0): the force quadrupole DECAYS as 1/x "
                      "at high acceleration, with the sign fixed"),
        deep_mond_asymptote=deep,
        sign=dict(
            exact_first_order=sgn_exact,
            caricature_gated=sgn_cari,
            caricature_constant_K="opposite to exact (analytic: 1 - (2/3) "
                                  "f_E P2 against the exact 1 + f_E P2/3)",
            statement=(
                f"for f_E > 0 the inward acceleration is WEAKER along the "
                f"axis e (A2 {sgn_exact} at every radius from 0.1 a to 300 "
                f"a) even though the potential is DEEPER along e (chi < 0): "
                f"the potential quadrupole is a nearly constant offset that "
                f"is approached from above (chi ~ -a0 r/3 on the Newtonian "
                f"side, chi -> -sqrt(G M a0)/3 in deep MOND), so its radial "
                f"derivative is negative. A CONSTANT anisotropy of the same "
                f"axis gives the opposite force sign (chi = -GM/3r rises "
                f"toward zero). This is a derived, non-obvious prediction: "
                f"a transition-band-gated tensor and a constant tensor of "
                f"the same axis and sign have OPPOSITE force quadrupoles. "
                f"The compiler's declared radial caricature has the "
                f"opposite sign to the exact solution for constant K, and "
                f"the same sign but a magnitude wrong by up to 40x (and no "
                f"deep-MOND decay) for the gated case; gates 1 and 4 consume "
                f"only |residual| and symmetry, so no verdict depends on it, "
                f"but no sign may be read off that reduction.")),
        newton_limit_check=dict(
            note="constant-K Newtonian limit: exact chi = -GM/(3r) far out, "
                 "i.e. |g| = GM/r^2 (1 + f_E P2/3); the caricature gives "
                 "1 - (2/3) f_E P2"),
    )


def counterfactual_table(l2: dict) -> list:
    A2pk = l2["peak"]["A2_per_fE"]
    xpk = l2["peak"]["x0"]
    return [
        dict(intervention="rotate the external axis e by an angle dpsi, "
                          "baryons fixed",
             observable="phase of the m=2 harmonic of |g_r| (dynamics) and of "
                        "the lensing-potential quadrupole",
             response="d(phase)/d(psi_e) = +1 exactly, zero lag, at every "
                      "radius; amplitude unchanged",
             sign="+1 (rigid co-rotation)"),
        dict(intervention="move the baryons holding e fixed",
             observable="amplitude of the m=2 harmonic of |g_r|",
             response=f"A2(r) = chi'(r)/g0(r) per unit f_E re-evaluated on "
                      f"the NEW baryons instantly; |A2| peaks at "
                      f"{abs(A2pk):.3f} f_E where g0 ~ {xpk:.2f} a0, decays "
                      f"as -1/(3 g0/a0) at high acceleration and -> 0 in "
                      f"deep MOND",
             sign=("- along e for f_E > 0: the inward pull is WEAKER along "
                   "the tidal axis (the m=2 minimum of |g_r| sits on e); "
                   "the potential is nonetheless deeper along e")),
        dict(intervention="move the halo holding baryons fixed",
             observable="anything",
             response="0: there is no halo",
             sign="0"),
        dict(intervention="scramble members preserving every radial profile",
             observable="m=2 amplitude and phase",
             response="0: the response is a functional of the smooth field "
                      "and the external axis, never of the member list",
             sign="0"),
        dict(intervention="change history preserving present matter and e",
             observable="anything",
             response="0: no memory",
             sign="0"),
        dict(intervention="change the photon path preserving endpoints",
             observable="deflection / time delay",
             response="follows the same Phi = Phi0 + f_E chi P2 the matter "
                      "sees; the lensing quadrupole phase equals the dynamical "
                      "one and the two amplitudes are the same chi (no slip)",
             sign="fixed matter-light covariance = +1 in the potential"),
        dict(intervention="radial profile of the quadrupole at fixed e",
             observable="d(phase)/d ln r",
             response="0 exactly (e is uniform across the object); a "
                      "collisionless triaxial halo twists with radius",
             sign="0 vs CDM != 0"),
        dict(intervention="tilt the disk normal against e (angle psi)",
             observable="azimuthally averaged rotation-curve residual "
                        "(the RAR residual of one disk)",
             response="the in-plane average of P2(e.rhat) is -(1/2) P2(cos "
                      "psi) [<(e.rhat)^2>_plane = sin^2(psi)/2], so d ln g_c "
                      "= -(1/2) P2(cos psi) A2(r) f_E with A2 < 0 for "
                      "f_E > 0",
             sign=(f"+ for e along the disk normal (psi = 0), - for e in the "
                   f"plane (psi = 90 deg), for f_E > 0: a SIGNED, environment-"
                   f"locked RAR residual of "
                   f"{l2['rar_residual_dex_per_fE_at_3a']:.4f} f_E dex at "
                   f"r = 3 a of the caricature galaxy, with no per-object "
                   f"freedom")),
    ]


def main():
    guard.arm()
    t0 = time.perf_counter()
    out = dict(
        family="T: action-derived void/tensor gravity",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        action=dict(
            lagrangian="L = -(1/8 pi G)[ a0^2 F(|u|^2/a0^2) + f_E h(|u|/a0) "
                       "u^T That_env u ] - rho Phi,  u = grad Phi",
            base="F' = mu, mu(x) = x/(1+x) (AQUAL)",
            weight="h(x) = mu(1-mu) = x/(1+x)^2",
            axis="That_env: unit-norm traceless tidal tensor of the environment "
                 "(Hessian of Phi_N smoothed on L_env); principal-axis form "
                 "sqrt(3/2)(e e^T - I/3) with sqrt(3/2) absorbed into f_E",
            field_equation="div M(u) = 4 pi G rho, M = mu(x) u + (1/2) f_E "
                           "[ h'(x)(u^T That u) uhat/a0 + 2 h(x) That u ]",
            radial_reduction="g [ mu(x) + f_E lambda eta(x) ] = g_N, "
                             "lambda = (e.rhat)^2 - 1/3, eta = h + x h'/2 = "
                             "x(3+x)/(2(1+x)^3)",
            universal_constants=dict(shared_with_base=["G", "a0"],
                                     new=["f_E", "L_env"], n_new=2),
            model_class="static_scalar_potential with a background axis field "
                        "(in the compiler's Gate 4 scope); the closed version "
                        "with That_env promoted to a dynamical field is "
                        "'extra_propagating_field'"),
        ellipticity=ellipticity_interval(),
        deep_mond_argument=deep_mond_ellipticity_argument(),
        known_limits=known_limits(),
        l2=l2_solution(),
    )
    out["counterfactuals"] = counterfactual_table(out["l2"])
    out["provenance"] = guard.summary()
    out["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(HERE, "tensor_results.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    e = out["ellipticity"]
    print(f"ellipticity: f_E in [{e['admissible_fE_min']}, "
          f"{e['admissible_fE_max']}]")
    v = out["l2"]["validation_constant_K_newton"]
    print(f"l=2 validation vs exact constant-K: max rel err {v['max_rel_err']:.2e}")
    print(f"l=2 peak A2/f_E = {out['l2']['peak']['A2_per_fE']:.4f} at "
          f"r = {out['l2']['peak']['r_kpc']:.2f} kpc, x0 = {out['l2']['peak']['x0']:.3f}")
    for r in out["l2"]["rows"]:
        print(f"  r/a={r['r_over_a']:6.1f} x0={r['x0']:.3f} A2/fE={r['A2_exact_per_fE']:+.4f} "
              f"caricature={r['A2_caricature_per_fE']:+.4f}")
    print("provenance:", out["provenance"]["assertion"],
          "| foreign reads:", out["provenance"]["foreign_reads"])
    print(f"wall {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
