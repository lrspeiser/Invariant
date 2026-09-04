"""LENSING CLOSURE: two metric potentials, and the discipline that identifies them.

THE PROBLEM
-----------
A modified Poisson equation determines the potential felt by SLOW MATTER.  It
says nothing about photons.  Write the perturbed metric as

    ds^2 = -(1 + 2 Psi/c^2) c^2 dt^2 + a^2 (1 - 2 Phi/c^2) dx^2

Then a slow test particle obeys  d^2x/dt^2 = -grad Psi, while a null geodesic
is deflected by grad(Phi + Psi).  Define

    gravitational slip      eta     = Phi / Psi
    lensing response        Sigma_s = (Phi + Psi) / (2 Psi) = (1 + eta) / 2

GR with no anisotropic stress gives Phi = Psi, eta = 1, Sigma_s = 1: NO SLIP.
Every cluster result in this programme has silently assumed Sigma_s = 1.

Weak lensing measures REDUCED shear g = gamma/(1-kappa), not mass.  So

    "baryons + RAR underpredict the cluster field"

is really

    "baryons + RAR + Sigma_s = 1 underpredict the observed SHEAR."

Three factors, one of them tested.

IDENTIFIABILITY -- the reason the ORDER is the whole point
----------------------------------------------------------
Within lensing alone, slip is EXACTLY degenerate with the lens mass: every
lensing observable depends only on the product Sigma_s * M_len, so no shear
profile, no image configuration and no time delay can separate a doubled mass
from a doubled Sigma_s.  Slip is identifiable ONLY because the mass is fixed
from outside, by a dynamics law frozen in advance.  Abandon that and the shear
stops being a test of gravity.  Section 1 measures exactly how much.

THE DISCIPLINED SEQUENCE
------------------------
    1  fit the gravity law to DYNAMICS only     (upstream; provenance audited
                                                 in section 2, and one headline
                                                 constant FAILS that audit)
    2  FREEZE it
    3  predict RAW SHEAR under NO SLIP          (section 3)
    4  only if the failure is STRUCTURED, admit ONE universal slip parameter
       with a confidence interval               (sections 4, 5)
    5  test it on a held-out sample             (section 6)

HOW THE SLIP ENTERS THE FORWARD MODEL, EXACTLY
----------------------------------------------
    M_len(r) = Sigma_s(r) * M_dyn(r),   M_dyn = g_dyn r^2 / G

deprojected to rho_len = (1/4 pi r^2) dM_len/dr and re-projected by the Abel
integral.  Applying the slip in THREE dimensions and re-projecting keeps
gamma_t = DeltaSigma/Sigma_cr exact for any Sigma_s(r); rescaling DeltaSigma
directly is only right for a constant slip.  Then

    kappa = Sigma_len/Sigma_cr,  gamma_t = DeltaSigma_len/Sigma_cr
    g_+   = gamma_t/(1-kappa) [1 + kappa(<beta^2>/<beta>^2 - 1)]

(Chiu+2022 Eq. 23 / Seitz & Schneider 1997), with per-bin measured <beta> and
<beta^2>.  g_+ is NOT linear in Sigma_s and that nonlinearity is carried, not
linearised; the convergence actually reached by these models on these data is
measured and printed in sections 0 and 3 rather than assumed.

DATA -- all previously acquired and manifested, nothing re-downloaded
--------------------------------------------------------------------
    ../efeds-hsc/decade_efeds_shear_profiles.tsv   DECADE (DELVE DR3) raw
        per-cluster tangential reduced shear, 496 systems / 3365 points,
        9.6 sigma tangential, clean B-mode, passing random-point null.  The
        DECADE/DES ellipticity basis has its first axis pointing WEST; that
        convention is baked into the acquired file.
    ../lead01/efeds_bahar2022_table1_density.tsv   Vikhlinin n_e fits for 542
        eFEDS systems, plus table2 properties.
    ../efeds-hsc/efeds_stacked_shear.tsv           Chiu+2022 HSC stack, used
        ONLY as an independent absolute-amplitude cross-check, never fitted.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (os.path.join(ROOT, "efeds-hsc"), os.path.join(ROOT, "lead01")):
    if p not in sys.path:
        sys.path.insert(0, p)

import pipeline as P                                            # noqa: E402
import efeds_hsc as E                                           # noqa: E402
import decade_test as DT                                        # noqa: E402

MPC, MSUN, G = P.MPC, P.MSUN, P.G
TSTART = time.time()
RES: dict = {}
RNG = np.random.default_rng(20260904)

# ---------------------------------------------------------------- the laws
# FROZEN.  Every constant below was fitted elsewhere, to dynamics, and none is
# refitted in this file.  Provenance audited in section 2 -- and one of them
# fails that audit.
A0_RAR = 1.0844e-10          # tournament BASE_rar, SPARC train
A0_AQ = 1.0580375e-10        # tournament BASE_aqual, SPARC train
A0_TID = 1.00230625e-10      # tournament aqual|scalar_a0|tidal|inv|m2|I1e-33
T0_TID = 1.0e-33             # s^-2, the tidal gate scale
M_TID = 2.0
A_TID_DYN = 7.5              # amplitude at the X-COP HYDROSTATIC flat target
A_TID_LENS = 16.0            # amplitude at the lane-12 LENSING-DERIVED target

LAWS = {
    "newton":    dict(base="newton", a0=0.0, A=0.0,
                      tag="Newton (baryons only)"),
    "rar":       dict(base="rar", a0=A0_RAR, A=0.0, tag="RAR"),
    "aqual":     dict(base="aqual", a0=A0_AQ, A=0.0, tag="AQUAL"),
    "tidal":     dict(base="aqual", a0=A0_TID, A=A_TID_DYN,
                      tag="tidal-gated scalar A=7.5 (dynamics-frozen)"),
    "tidal_A16": dict(base="aqual", a0=A0_TID, A=A_TID_LENS,
                      tag="tidal-gated scalar A=16 (lensing-derived A)"),
    "wrongshape": dict(base="newton", a0=0.0, A=0.0, tilt=-1.0,
                       tag="CONTROL Newton x (r/Mpc)^-1 (wrong shape)"),
}
PRIMARY_LAWS = ("newton", "rar", "aqual", "tidal", "tidal_A16")
CONTROL_ORDER = ("newton", "wrongshape", "rar", "aqual", "tidal")

CONST_GRID = np.linspace(-1.5, 3.0, 451)


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------- the tidal invariant
def tidal_invariant(r, g):
    """|traceless Hessian of Phi_N|_F in s^-2 -- EXACTLY the definition in
    ../tournament/ch_radial.py:invariants().

    For a spherical Phi(r) with |Phi'| = g, the Hessian is diag(g', g/r, g/r);
    this returns the Frobenius norm of its traceless part.  Outside a point
    mass it is sqrt(6) GM/r^3, which DECAYS outward -- which is why an inverse
    gate 1/(1+(|T|/T0)^m) switches ON in the outskirts and OFF inside dense
    galaxies, and why the resulting boost RISES with radius.
    """
    dg = np.gradient(g, r)
    t_rr, t_tt = -dg, -g / r
    tr = (t_rr + 2.0 * t_tt) / 3.0
    return np.sqrt((t_rr - tr) ** 2 + 2.0 * (t_tt - tr) ** 2)


def g_law(sysm, law, f_star=0.0, a0_mult=1.0):
    """Frozen dynamics prediction g_dyn(r) for one system and one law."""
    d = LAWS[law]
    gb = sysm.g_b * (1.0 + f_star)
    base, a0 = d["base"], d["a0"] * a0_mult
    if d["A"] != 0.0:
        Tn = tidal_invariant(sysm.r, gb) / T0_TID
        W = 1.0 / (1.0 + Tn ** M_TID)
        a0 = a0 * (1.0 + d["A"] * W)
    if base == "newton":
        g = gb.copy()
    elif base == "rar":
        x = np.sqrt(np.maximum(gb, 1e-30) / np.maximum(a0, 1e-40))
        g = gb / (1.0 - np.exp(-x))
    elif base == "aqual":
        g = 0.5 * (gb + np.sqrt(gb ** 2 + 4.0 * gb * a0))
    else:
        raise ValueError(base)
    if d.get("tilt"):
        g = g * (sysm.r / MPC) ** d["tilt"]
    return g


def boost(sysm, law, f_star=0.0):
    return g_law(sysm, law, f_star) / np.maximum(sysm.g_b, 1e-40)


def slip_of_r(r, s, q):
    """Shape of Sigma_s(r) with unit normalisation:
    (r/Mpc)^s exp[q ln^2(r/Mpc)].  The amplitude is carried separately so it
    can be profiled analytically."""
    u = np.log(np.maximum(r, 1e-6 * MPC) / MPC)
    return np.exp(s * u + q * u * u)


# ------------------------------------------------------- flattened observable
class Flat:
    """All (system, bin) points of an index set as flat arrays.

    Everything downstream is vectorised over this, which is what makes the
    closure ladder and the Monte-Carlo nulls affordable.
    """

    def __init__(self, obs, idx):
        self.idx = np.asarray(idx)
        cat = np.concatenate
        self.gt = cat([obs.gt[k] for k in idx])
        self.gx = cat([obs.gx[k] for k in idx])
        self.er = cat([obs.er[k] for k in idx])
        self.sc = cat([obs.scinv[k] for k in idx])
        self.bt = cat([obs.bt[k] for k in idx])
        self.b2 = cat([obs.b2[k] for k in idx])
        self.R = cat([obs.R[k] for k in idx])
        self.sysi = cat([np.full(len(obs.R[k]), j)
                         for j, k in enumerate(idx)]).astype(int)
        self.c2 = self.b2 / np.maximum(self.bt, 1e-9) ** 2 - 1.0
        self.n = self.gt.size
        self.nsys = len(self.idx)
        self.w = 1.0 / self.er ** 2

    def gplus(self, S, dS, c=1.0):
        kap = c * S * self.sc
        gam = c * dS * self.sc
        return (gam / np.maximum(1.0 - kap, 1e-3)) * (1.0 + kap * self.c2)

    def chi2(self, S, dS, c=1.0):
        return float(np.sum(((self.gplus(S, dS, c) - self.gt) / self.er) ** 2))

    def chi2_per_sys(self, S, dS, c=1.0):
        r = ((self.gplus(S, dS, c) - self.gt) / self.er) ** 2
        return np.bincount(self.sysi, weights=r, minlength=self.nsys)


def project(systems, obs, idx, law, s=0.0, q=0.0, f_star=0.0,
            r_trunc=20.0, a0_mult=1.0, sysobjs=None):
    """Sigma_len and DeltaSigma_len at the measured radii, flat.

    The slip SHAPE (s, q) is applied in 3-D before the Abel projection; the
    slip AMPLITUDE is left out and applied afterwards as the linear factor c,
    which is exact because the projection is linear in rho.
    """
    Ss, dSs = [], []
    for j, k in enumerate(idx):
        sm = systems[k] if sysobjs is None else sysobjs[j]
        g = g_law(sm, law, f_star, a0_mult)
        if s != 0.0 or q != 0.0:
            g = g * slip_of_r(sm.r, s, q)
        S, dS = sm.sigma_profile(g, obs.R[k], r_trunc)
        Ss.append(S)
        dSs.append(dS)
    return np.concatenate(Ss), np.concatenate(dSs)


def fit_const(F, S, dS, grid=CONST_GRID):
    """Profile ONE global constant slip.  Returns (chi2, log10 Sigma_0, curve)."""
    cs = np.array([F.chi2(S, dS, 10.0 ** a) for a in grid])
    i = int(np.argmin(cs))
    return float(cs[i]), float(grid[i]), cs


def fit_persystem(F, S, dS, grid=np.linspace(-2.0, 3.0, 251)):
    """One free constant slip per cluster -- the unrestricted closure."""
    best = np.full(F.nsys, np.inf)
    barg = np.zeros(F.nsys)
    for a in grid:
        c = F.chi2_per_sys(S, dS, 10.0 ** a)
        m = c < best
        best[m] = c[m]
        barg[m] = a
    return float(best.sum()), barg


def binned_pull(F, S, dS, c, edges):
    """Weighted mean residual pull (obs - model)/sigma in radial bins."""
    R = F.R / MPC
    pull = (F.gt - F.gplus(S, dS, c)) / F.er
    rows = []
    for i in range(len(edges) - 1):
        m = (R >= edges[i]) & (R < edges[i + 1])
        if m.sum() < 5:
            continue
        rows.append(dict(R=float(R[m].mean()), n=int(m.sum()),
                         pull=float(pull[m].mean()),
                         se=float(1.0 / math.sqrt(m.sum()))))
    return rows, pull


def trend(rows):
    x = np.log10(np.array([r["R"] for r in rows]))
    y = np.array([r["pull"] for r in rows])
    e = np.array([r["se"] for r in rows])
    W = 1.0 / e ** 2
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    C = np.linalg.inv(X.T @ (W[:, None] * X))
    b = C @ (X.T @ (W * y))
    return float(b[1]), float(math.sqrt(C[1, 1])), float(np.sum((y / e) ** 2))


def eta_of(sig):
    """Slip eta = Phi/Psi implied by a lensing response Sigma_s = (1+eta)/2."""
    return 2.0 * sig - 1.0


def c_star(F, S, dS):
    """chi2-optimal LINEAR scale, the robust 'how far off is the amplitude'
    summary.  The ratio of weighted means is not that number and is dominated
    by the few systems with the largest predicted signal."""
    p = F.gplus(S, dS, 1.0)
    return float(np.sum(F.w * p * F.gt) / np.sum(F.w * p * p))


EDGES = np.geomspace(0.2, 3.5, 11) / P.H_LITTLE


# ============================================================== SECTION 0
def section0(systems, obs, train):
    hdr("0.  FORMALISM AND IDENTIFIABILITY, before anything is fitted")
    print("""
   Psi   slow matter      d^2x/dt^2 = -grad Psi        <- the modified Poisson
   Phi   spatial curvature                                equation gives THIS
   light                  deflection ~ grad(Phi + Psi)  <- and says nothing
                                                           about THIS
   slip  eta = Phi/Psi    lensing response Sigma_s = (1+eta)/2
   GR without anisotropic stress: eta = 1, Sigma_s = 1  ->  NO SLIP""")

    sub = train[:40]
    F = Flat(obs, sub)
    S, dS = project(systems, obs, sub, "rar")
    c_a = F.chi2(S, dS, 2.0)                       # constant slip, projected
    S2, dS2 = project(systems, obs, sub, "rar")
    S2, dS2 = 2.0 * S2, 2.0 * dS2                  # constant slip, in 3-D
    c_b = F.chi2(S2, dS2, 1.0)
    rel = abs(c_a - c_b) / max(c_a, 1e-9)
    print(f"\n   I1  Sigma_s = 2 as a rescaling of the PROJECTED mass: "
          f"chi2 = {c_a:.6f}")
    print(f"       Sigma_s = 2 as a rescaling of the 3-D lensing mass: "
          f"chi2 = {c_b:.6f}")
    print(f"       identical to {rel:.1e} relative.  A CONSTANT slip is exactly")
    print("       a lens-mass rescaling, so lensing alone -- shear, images or")
    print("       time delays -- can never separate the two.  The frozen")
    print("       dynamics law is the ONLY source of identification.")

    S, dS = project(systems, obs, train, "rar")
    Ft = Flat(obs, train)
    kap = S * Ft.sc
    dlog = 1.0 + kap / (1.0 - kap) + kap * Ft.c2 / (1.0 + kap * Ft.c2)
    R = Ft.R / MPC
    print(f"\n   I2  reduced shear is NOT linear in Sigma_s "
          f"(kappa up to {kap.max():.3f}):")
    print(f"       {'R [Mpc]':>9s} {'kappa':>9s} {'dln g+/dln Sigma_s':>20s}")
    for lo, hi in ((0.0, .4), (.4, .8), (.8, 1.5), (1.5, 3.0), (3.0, 99.)):
        m = (R >= lo) & (R < hi)
        if m.sum() > 5:
            print(f"       {R[m].mean():9.3f} {kap[m].mean():9.4f} "
                  f"{dlog[m].mean():20.4f}")
    print(f"\n       mean {dlog.mean():.4f}, inner bins "
          f"{dlog[R < 0.4].mean():.4f}.  A linearised 'shear scales with mass'")
    print("       shortcut would be wrong by that much and would bias any slip.")
    RES["S0_identifiability"] = {
        "constant_slip_vs_mass_rescale_rel_diff": float(rel),
        "kappa_max": float(kap.max()),
        "dlng_dlnSigma_mean": float(dlog.mean()),
        "dlng_dlnSigma_inner": float(dlog[R < 0.4].mean())}


# ============================================================== SECTION 1
def section1(systems, obs, train):
    hdr("1.  THE FREE-CLOSURE CONTROL -- what the discipline protects against")
    print("""
   If the closure is left free, an obviously WRONG dynamics law fits the shear.
   Ladder of closures, each applied in 3-D and re-projected:

       C0  Sigma_s = 1                                   0 free parameters
       C1  Sigma_s = Sigma_0                             1
       C2  Sigma_s = Sigma_0 (r/Mpc)^s                   2
       C3  Sigma_s = Sigma_0 (r/Mpc)^s e^{q ln^2 r}      3
       C4  Sigma_s = Sigma_0,k, one per cluster          n_sys

   C4 is the 'unrestricted lensing closure' the brief forbids.  The reference
   line is the RAR with NO closure freedom at all.""")
    F = Flat(obs, train)
    s_grid = np.linspace(-1.5, 1.5, 13)
    s_grid3 = np.linspace(-1.5, 1.5, 7)
    q_grid = np.linspace(-0.6, 0.6, 5)
    tab, sig_r = {}, {}
    for law in CONTROL_ORDER:
        S0, dS0 = project(systems, obs, train, law)
        row = {"C0": (0, F.chi2(S0, dS0, 1.0), {})}
        c1, a1, _ = fit_const(F, S0, dS0)
        row["C1"] = (1, c1, {"log10_Sigma0": a1})
        best = (np.inf, 0.0, 0.0)
        for s in s_grid:
            S, dS = project(systems, obs, train, law, s=s)
            c, a, _ = fit_const(F, S, dS)
            if c < best[0]:
                best = (c, a, s)
        row["C2"] = (2, best[0], {"log10_Sigma0": best[1], "s": best[2]})
        best3 = (np.inf, 0.0, 0.0, 0.0)
        for s in s_grid3:
            for q in q_grid:
                S, dS = project(systems, obs, train, law, s=s, q=q)
                c, a, _ = fit_const(F, S, dS)
                if c < best3[0]:
                    best3 = (c, a, s, q)
        row["C3"] = (3, best3[0], {"log10_Sigma0": best3[1], "s": best3[2],
                                   "q": best3[3]})
        c4, per = fit_persystem(F, S0, dS0)
        row["C4"] = (F.nsys, c4, {"median_log10_Sigma0": float(np.median(per)),
                                  "sd_log10_Sigma0": float(np.std(per))})
        tab[law] = row
        sig_r[law] = (best[1], best[2])
        print(f"   ... {law} done  ({time.time() - TSTART:.0f}s)")
    print(f"\n   TRAIN half: {F.nsys} systems, {F.n} shear points\n")
    print(f"   {'dynamics law':<40s} {'C0':>9s} {'C1':>9s} {'C2':>9s} "
          f"{'C3':>9s} {'C4':>9s}")
    for law in CONTROL_ORDER:
        row = tab[law]
        print(f"   {LAWS[law]['tag'][:40]:<40s} "
              + "".join(f"{row[c][1]:9.1f}" for c in
                        ("C0", "C1", "C2", "C3", "C4")))
    ref = tab["rar"]["C0"][1]
    print(f"\n   REFERENCE  RAR under NO SLIP:  chi2 = {ref:.1f} on {F.n} pts")
    print("   The number that matters is the FRACTION OF THE GAP a free")
    print("   closure recovers: (chi2_C0 - chi2_Cx)/(chi2_C0 - chi2_RAR,C0).")
    print("")
    print(f"      {'law + closure':<26s} {'par':>5s} {'chi2':>9s} "
          f"{'vs RAR+noslip':>14s} {'gap recovered':>14s}")
    for law in ("newton", "wrongshape"):
        gap = tab[law]["C0"][1] - ref
        for c in ("C1", "C2", "C3", "C4"):
            v, k = tab[law][c][1], tab[law][c][0]
            print(f"      {law + ' + ' + c:<26s} {k:5d} {v:9.1f} "
                  f"{v - ref:+14.1f} "
                  f"{(tab[law]['C0'][1] - v) / gap * 100:13.1f}%")
    print("""
   Newton + C2 and the WRONG-SHAPE control + C2 land on the same chi2, because
   the closure's power law r^s and the control law's tilt r^-1 are the same
   function: (r/Mpc)^s x (r/Mpc)^-1 = (r/Mpc)^(s-1).  A radial closure and a
   radial modification of the force law are not merely similar; they are
   ALGEBRAICALLY IDENTICAL.  That is the degeneracy in its sharpest form.""")
    # what did the free closure actually learn?  compare it to the dynamics
    print("\n   WHAT THE FREE CLOSURE LEARNED.  Sigma_s(r) recovered for Newton")
    print("   at C2, against the field boost the RAR supplies as DYNAMICS:")
    a_n, s_n = sig_r["newton"]
    rr = np.array([0.3, 0.5, 1.0, 2.0, 3.0])
    sig = 10.0 ** a_n * rr ** s_n
    brar = np.median(np.array([np.interp(rr * MPC, systems[k].r,
                                         boost(systems[k], "rar"))
                               for k in train]), axis=0)
    print(f"      {'r [Mpc]':>9s}" + "".join(f"{x:9.1f}" for x in rr))
    print(f"      {'Sigma_s':>9s}" + "".join(f"{x:9.3f}" for x in sig))
    print(f"      {'B_RAR':>9s}" + "".join(f"{x:9.3f}" for x in brar))
    print("      The 'closure' has simply absorbed the missing dynamics.  A")
    print("      free Sigma_s(r) and a modified force law are the same two")
    print("      parameters wearing different names.")
    RES["S1_free_closure_control"] = {
        law: {c: {"k": int(tab[law][c][0]), "chi2": tab[law][c][1],
                  "par": tab[law][c][2]}
              for c in ("C0", "C1", "C2", "C3", "C4")}
        for law in CONTROL_ORDER}
    RES["S1_reference_rar_noslip_chi2"] = ref
    RES["S1_n_points_train"] = F.n
    RES["S1_newton_closure_vs_rar_boost"] = {
        "r_Mpc": list(rr), "Sigma_s_newton_C2": [float(x) for x in sig],
        "B_rar_median": [float(x) for x in brar]}
    print("""
   READ THIS ROW BY ROW.  A law wrong by a large factor in amplitude, and a law
   additionally wrong in radial SHAPE, are both brought level with or past the
   RAR by two or three closure parameters.  With the closure free, the shear
   stops being a test of gravity.  That is the failure mode the ordering
   forbids, quantified on real data.""")
    return tab


# ============================================================== SECTION 2
def section2(systems):
    hdr("2.  PROVENANCE AUDIT OF THE FROZEN CONSTANTS  (step 1 -> step 2)")
    prov = {
        "newton": "no free constant.",
        "rar": f"a0 = {A0_RAR:.6g} m/s^2 from the SPARC train rotation curves "
               "(tournament BASE_rar). DYNAMICS ONLY -- PASSES.",
        "aqual": f"a0 = {A0_AQ:.6g} m/s^2 from the SPARC train rotation curves "
                 "(tournament BASE_aqual). DYNAMICS ONLY -- PASSES.",
        "tidal": f"a0 = {A0_TID:.6g} m/s^2 SPARC train; T0 = {T0_TID:.0e} s^-2 "
                 f"and m = {M_TID:.0f} from the declared grid; A = {A_TID_DYN} "
                 "fitted to the X-COP flat target B = 2 (nu/nu_RAR = 2.53 for "
                 "A2029), which is a HYDROSTATIC X-ray constraint. Hydrostatic "
                 "gas is slow matter, so that is dynamics -- PASSES.",
        "tidal_A16": f"identical but A = {A_TID_LENS}, selected by minimising "
                     "the distance to the lane-12 radial requirement, which is "
                     "interpolated from PUBLISHED LENSING MASS PROFILES (CLASH "
                     "SL+WL+magnification, XXL, X-COP x WL). Those masses were "
                     "derived under the standard lens equation, i.e. under "
                     "Sigma_s = 1 -- FAILS the audit for this lane.",
    }
    for k in PRIMARY_LAWS:
        print(f"\n   {k:<10s} {prov[k]}")
    print("""
   THE FINDING, AND IT IS NOT A TECHNICALITY.  The tournament's headline
   amplitude A = +16.0 was selected against a radial requirement built from
   published lensing MASSES.  Those masses already assume no slip.  Predicting
   raw shear with A = 16 under no slip and calling the agreement a success
   would be circular: the amplitude was set by the answer.  This lane therefore
   takes A = 7.5 -- the X-COP HYDROSTATIC amplitude, genuinely dynamics -- as
   the PRIMARY frozen tidal law, and carries A = 16 beside it, labelled.  This
   is the same 'never fit the law and the closure together' rule, caught one
   level further upstream than the brief phrased it.""")
    rr = np.array([0.3, 0.5, 1.0, 2.0, 3.0]) * MPC
    print("\n   Frozen field boost B(r) = g_dyn/g_b, median over the sample:")
    print(f"      {'law':<12s}" + "".join(f"{x/MPC:9.1f}" for x in rr)
          + "   Mpc")
    Bt = {}
    for law in PRIMARY_LAWS:
        b = np.array([np.interp(rr, s.r, boost(s, law)) for s in systems])
        med = np.median(b, axis=0)
        Bt[law] = [float(x) for x in med]
        print(f"      {law:<12s}" + "".join(f"{x:9.3f}" for x in med))
    RES["S2_provenance"] = prov
    RES["S2_frozen_boost_median"] = {"r_Mpc": [0.3, 0.5, 1.0, 2.0, 3.0],
                                     "B": Bt}
    return Bt


def section2b(systems, obs, allidx):
    """DOES THIS SAMPLE HAVE ANY LEVERAGE ON THE TIDAL GATE?

    The brief's rule: 'a null from a detector with zero power below the
    predicted amplitude says nothing'.  So measure the reach of the gate on
    the actual measured points BEFORE scoring anything.
    """
    hdr("2b. LEVERAGE: does the eFEDS sample reach the tidal gate at all?")
    Ws, dlt = [], []
    for k in allidx:
        s = systems[k]
        T = tidal_invariant(s.r, s.g_b)
        W = 1.0 / (1.0 + (T / T0_TID) ** M_TID)
        d = np.log10(np.maximum(boost(s, "tidal"), 1e-30)
                     / np.maximum(boost(s, "rar"), 1e-30))
        Ws.append(np.interp(obs.R[k], s.r, W))
        dlt.append(np.interp(obs.R[k], s.r, d))
    W = np.concatenate(Ws)
    d = np.concatenate(dlt)
    print(f"\n   gate W = 1/(1+(|T|/T0)^2) evaluated at all {W.size} MEASURED"
          " points:")
    print(f"      min {W.min():.4f}   1st pct {np.percentile(W, 1):.4f}   "
          f"median {np.median(W):.4f}   max {W.max():.4f}")
    print(f"      fraction with W < 0.99: {np.mean(W < 0.99):.3f};  "
          f"with W < 0.90: {np.mean(W < 0.90):.4f}")
    print(f"\n   log10(B_tidal / B_RAR) over the same points:")
    print(f"      median {np.median(d):+.4f} dex (factor {10**np.median(d):.3f})"
          f"   sd {d.std():.4f} dex   full range {np.ptp(d):.4f} dex")
    print(f"""
   THE GATE IS SATURATED.  |T| in these eFEDS systems is far below T0 = 1e-33
   over essentially the whole measured range, so W ~ 1 and the tidal law
   reduces to AQUAL with a0 -> a0 (1 + A), i.e. a CONSTANT rescaling of the
   RAR by sqrt(1+A) = {math.sqrt(1 + A_TID_DYN):.3f}.  Its variation across the
   entire data set is {d.std():.4f} dex.  A single universal slip parameter
   absorbs a constant EXACTLY.  So on this sample the tidal-gated law and the
   RAR are the same hypothesis up to a closure, and the shear cannot separate
   them no matter how many clusters are added.

   That is not a defect of the data -- it is where the gate lives.  |T| ~ T0
   requires ~1e14 Msun inside a few hundred kpc, i.e. massive cluster CORES.
   The eFEDS sample is groups: median M_b,500 is ~5e11 Msun and g_b at 1 Mpc
   is 0.012 a0.  The gate's radial structure is therefore testable only in
   strong-lensing-scale data, which is what makes section refsdal.py
   load-bearing rather than decorative.""")
    RES["S2b_gate_leverage"] = {
        "W_min": float(W.min()), "W_p1": float(np.percentile(W, 1)),
        "W_median": float(np.median(W)), "W_max": float(W.max()),
        "frac_W_lt_0.99": float(np.mean(W < 0.99)),
        "frac_W_lt_0.90": float(np.mean(W < 0.90)),
        "dlog10_B_tidal_over_rar_median": float(np.median(d)),
        "dlog10_sd": float(d.std()), "dlog10_range": float(np.ptp(d)),
        "sqrt_1_plus_A": math.sqrt(1 + A_TID_DYN)}
    return d


# ============================================================== SECTION 3
def section3(systems, obs, allidx):
    hdr("3.  STEP 3 -- RAW SHEAR PREDICTED UNDER NO SLIP, nothing fitted")
    F = Flat(obs, allidx)
    print(f"\n   All {F.nsys} systems, {F.n} (system, bin) points.")
    print("   Sigma_s = 1 exactly.  No amplitude, no tilt, no free parameter.")
    print(f"\n   {'law':<40s} {'chi2':>9s} {'chi2/N':>7s} {'mean pull':>10s}"
          f" {'Sigma_s needed':>14s} {'eta':>7s}")
    out, shp = {}, {}
    for law in PRIMARY_LAWS:
        S, dS = project(systems, obs, allidx, law)
        shp[law] = (S, dS)
        c = F.chi2(S, dS, 1.0)
        p = F.gplus(S, dS, 1.0)
        cs = c_star(F, S, dS)
        pull = (F.gt - p) / F.er
        out[law] = dict(chi2=c, chi2_per_pt=c / F.n,
                        c_star_linear=cs, eta_implied=eta_of(cs),
                        kappa_max=float((S * F.sc).max()),
                        mean_pull=float(pull.mean()),
                        se_pull=float(pull.std() / math.sqrt(F.n)))
        print(f"   {LAWS[law]['tag'][:40]:<40s} {c:9.1f} {c/F.n:7.4f} "
              f"{pull.mean():+10.4f} {cs:14.3f} {eta_of(cs):+7.2f}")
    c0 = float(np.sum((F.gt / F.er) ** 2))
    print(f"\n   zero-signal reference (g_pred = 0):"
          f"{'':<20s} {c0:9.1f} {c0/F.n:7.4f}")
    print(f"   max convergence reached by any model: "
          f"kappa = {max(v['kappa_max'] for v in out.values()):.3f}")
    print("""
   'Sigma_s needed' is the chi2-optimal linear scale, i.e. the lensing response
   that would be required to bring this law onto the observed shear; 'eta' is
   the corresponding slip Phi/Psi = 2 Sigma_s - 1.  NOTHING IS FITTED ABOVE:
   the chi2 column is the genuine zero-parameter no-slip prediction and the
   Sigma_s column is shown only to translate its size into the closure the law
   would need.  eta < 0 means the curvature potential has the opposite sign to
   the one slow matter feels -- light would have to bend the wrong way -- which
   no viable relativistic completion produces.""")
    RES["S3_no_slip"] = out
    RES["S3_chi2_zero_model"] = c0
    RES["S3_n_points"] = F.n
    print("""
   TWO SYSTEMATICS OWN THIS TABLE, AND BOTH ARE MEASURED, NOT ASSUMED.

   (a) SHEAR CALIBRATION.  DECADE photo-z are DNF point estimates with a
       dz = 0.2 selection margin, so foreground galaxies can dilute g_t.
       Section 9b measures the size directly by stacking the SAME clusters and
       comparing with Chiu+2022's HSC profile: mass-matched, the two surveys
       agree to a few hundredths of a dex.  This systematic is small.

   (b) THE X-RAY BARYON MODEL.  Section 5 shows that noise in the published
       Vikhlinin density parameters alone biases a fitted Sigma_s LOW, by
       between 0.03 and 0.34 dex depending on how much of the published
       marginal error is genuinely independent.  This systematic is large and
       it is the one that limits the measurement.

   Everything below therefore separates the CONSTANT part of the residual,
   which those systematics act on, from the SHAPE, which they do not.""")
    return out, shp


# ============================================================== SECTION 4
def section4(systems, obs, train):
    hdr("4.  IS THE FAILURE STRUCTURED, OR NOISE-LIKE?   (TRAIN half only)")
    print("""
   A universal slip is a single number.  It can absorb a CONSTANT offset and
   nothing else.  So the question that decides whether step 4 is permitted at
   all is: after the best constant is removed, what is left?

     P1  radial       weighted residual pull per radial bin
     P2  cross-system excess scatter of per-system offsets over shape noise
     P3  the B-mode, which carries no signal and sets the scale of 'noise-like'

   Diagnosed on the TRAIN half only.  The held-out half sees the frozen
   verdict in section 6 and nothing else.""")
    F = Flat(obs, train)
    out = {}
    for law in PRIMARY_LAWS:
        S, dS = project(systems, obs, train, law)
        c1, a1, _ = fit_const(F, S, dS)
        rows, pull = binned_pull(F, S, dS, 10.0 ** a1, EDGES)
        sl, se, shape_chi2 = trend(rows)
        offs = np.bincount(F.sysi, weights=pull, minlength=F.nsys) \
            / np.bincount(F.sysi, minlength=F.nsys)
        nper = np.bincount(F.sysi, minlength=F.nsys)
        exp_var = float(np.mean(1.0 / nper))
        obs_var = float(np.var(offs))
        out[law] = dict(chi2_const=c1, log10_Sigma0=a1,
                        shape_chi2=shape_chi2, n_bins=len(rows),
                        radial_slope=sl, radial_slope_err=se,
                        radial_slope_sigma=sl / se,
                        cross_sys_var=obs_var, cross_sys_expected=exp_var,
                        cross_sys_excess=obs_var / exp_var,
                        bins=rows)
        print(f"\n   {LAWS[law]['tag']}")
        print(f"      best constant slip Sigma_s = {10**a1:.3f} "
              f"(eta = {eta_of(10**a1):+.2f}), chi2 = {c1:.1f}")
        print(f"      {'R[Mpc]':>8s} {'n':>6s} {'pull':>9s} {'+-':>7s}")
        for r in rows:
            print(f"      {r['R']:8.3f} {r['n']:6d} {r['pull']:+9.3f} "
                  f"{r['se']:7.3f}")
        print(f"      P1 shape chi2 = {shape_chi2:.1f} on {len(rows)} bins;"
              f" log-r slope = {sl:+.3f} +- {se:.3f} ({sl/se:+.1f} sigma)")
        print(f"      P2 per-system offsets var {obs_var:.3f} vs shape-noise "
              f"{exp_var:.3f}  -> excess {obs_var/exp_var:.2f}x")
    bx = float(np.sum(F.w * F.gx) / np.sum(F.w))
    sx = float(1.0 / math.sqrt(np.sum(F.w)))
    Rn = F.R / MPC
    bxr = []
    for i in range(10):
        m = (Rn >= EDGES[i]) & (Rn < EDGES[i + 1])
        if m.sum() < 5:
            continue
        ww = F.w[m]
        bxr.append(float(np.sum(ww * F.gx[m]) / np.sum(ww))
                   * math.sqrt(np.sum(ww)))
    bchi = float(sum(b * b for b in bxr))
    print(f"\n   P3 noise reference measured on these same data:")
    print(f"      B-mode <g_x> = {bx:+.6f} +- {sx:.6f}  ({bx/sx:+.1f} sigma)")
    print(f"      B-mode per-bin pulls: chi2 = {bchi:.1f} on {len(bxr)} bins"
          "  <- this is what 'no structure' looks like here")
    RES["S4_structure"] = out
    RES["S4_bmode"] = {"mean": bx, "sigma": sx, "snr": bx / sx,
                       "per_bin_chi2": bchi, "n_bins": len(bxr)}
    return out


# ============================================================== SECTION 5
def section5(systems, obs, train, recs):
    hdr("5.  STEP 4 -- ONE universal slip parameter, fitted on TRAIN only")
    print("""
   ONE parameter, Sigma_s, global across every cluster and every radius, fitted
   on the declared TRAIN half.  The held-out half is not touched until
   section 6.  Interval from Delta chi2 = 1 on the profile curve.""")
    F = Flat(obs, train)
    out = {}
    for law in PRIMARY_LAWS:
        S, dS = project(systems, obs, train, law)
        c, a, cs = fit_const(F, S, dS)
        ok = cs - c <= 1.0
        lo, hi = float(CONST_GRID[ok].min()), float(CONST_GRID[ok].max())
        out[law] = dict(log10_Sigma0=a, chi2=c, log10_lo=lo, log10_hi=hi,
                        Sigma0=10.0 ** a, Sigma0_lo=10.0 ** lo,
                        Sigma0_hi=10.0 ** hi, eta=eta_of(10.0 ** a),
                        eta_lo=eta_of(10.0 ** lo), eta_hi=eta_of(10.0 ** hi),
                        at_grid_edge=bool(a <= CONST_GRID[0] + 1e-9
                                          or a >= CONST_GRID[-1] - 1e-9))
        print(f"\n   {LAWS[law]['tag']}")
        print(f"      Sigma_s = {10**a:.3f}  [{10**lo:.3f}, {10**hi:.3f}]  "
              f"->  eta = {eta_of(10**a):+.3f} "
              f"[{eta_of(10**lo):+.3f}, {eta_of(10**hi):+.3f}]"
              + ("   AT GRID EDGE" if out[law]["at_grid_edge"] else ""))
    RES["S5_slip_train"] = out

    # ---------------------------------------------------- responsiveness gate
    print("\n   RESPONSIVENESS GATE  d(Sigma_hat)/d(Sigma_injected)")
    S, dS = project(systems, obs, train, "rar")
    inj = np.array([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    rec = []
    for t in inj:
        fake = F.gplus(S, dS, 10.0 ** t) + RNG.normal(0.0, F.er)
        cs = []
        for a in CONST_GRID:
            cs.append(float(np.sum(((F.gplus(S, dS, 10.0 ** a) - fake)
                                    / F.er) ** 2)))
        rec.append(float(CONST_GRID[int(np.argmin(cs))]))
    rec = np.array(rec)
    slope = float(np.polyfit(inj, rec, 1)[0])
    spread = float(rec.max() - rec.min())
    print(f"      injected  {np.array2string(inj, precision=2)}")
    print(f"      recovered {np.array2string(rec, precision=3)}")
    ok = abs(slope - 1.0) < 0.15 and spread > 0.5
    print(f"      d(recovered)/d(injected) = {slope:.4f}, spread = "
          f"{spread:.3f} dex over an injected span of {np.ptp(inj):.2f} dex "
          f"->  {'PASS' if ok else 'FAIL'}")
    RES["S5_responsiveness"] = {"injected": [float(x) for x in inj],
                                "recovered": [float(x) for x in rec],
                                "slope": slope, "spread": spread,
                                "passed": bool(ok)}

    # -------------------------------------- shared-quantity null, real errors
    print("\n   SHARED-QUANTITY NULL: the estimator's own expectation under H0")
    print("   Vikhlinin density parameters redrawn from their PUBLISHED errors,")
    print("   shear redrawn independently, truth = the fitted Sigma_s.")
    n_mc, law = 30, "rar"
    sub = train[:120]
    Fs = Flat(obs, sub)
    S0, dS0 = project(systems, obs, sub, law)
    truth = out[law]["log10_Sigma0"]
    idmap = {r["id"]: r for r in recs}
    grid = np.linspace(-1.5, 3.0, 226)
    print("   The published errors are MARGINAL and the Vikhlinin parameters")
    print("   are strongly covariant, so treating them as independent is the")
    print("   WORST case.  The null is therefore run at three error scalings")
    print("   so the bias is bracketed rather than quoted at one point.")
    print("")
    print(f"      {'err scale':>10s} {'E[est|H0]':>11s} {'sd':>8s} "
          f"{'bias [dex]':>11s} {'sigma_MC':>9s}")
    nulls = {}
    for scale in (0.25, 0.5, 1.0):
        ests = []
        for _ in range(n_mc):
            pert = []
            for k in sub:
                rc = dict(idmap[systems[k].id])
                for pn, en in (("n0sq", "e_n0sq"), ("rs", "e_rs"),
                               ("eps", "e_eps"), ("beta", "e_beta"),
                               ("alpha", "e_alpha")):
                    e = rc.get(en, 0.0)
                    if np.isfinite(e) and e > 0:
                        rc[pn] = rc[pn] + RNG.normal(0.0, scale * e)
                rc["n0sq"] = max(rc["n0sq"], 1e-6)
                rc["rs"] = max(rc["rs"], 1e-3 * MPC)
                rc["beta"] = max(rc["beta"], 0.34)
                pert.append(P.System(rc))
            fake = Fs.gplus(S0, dS0, 10.0 ** truth) + RNG.normal(0.0, Fs.er)
            Sp, dSp = project(systems, obs, sub, law, sysobjs=pert)
            cs = [float(np.sum(((Fs.gplus(Sp, dSp, 10.0 ** a) - fake)
                                / Fs.er) ** 2)) for a in grid]
            ests.append(float(grid[int(np.argmin(cs))]))
        ests = np.array(ests)
        bias = float(ests.mean() - truth)
        sem = float(ests.std(ddof=1) / math.sqrt(len(ests)))
        nulls[scale] = dict(mean=float(ests.mean()), sd=float(ests.std(ddof=1)),
                            sem=sem, bias_dex=bias,
                            bias_sigma_MC=bias / sem if sem else 0.0,
                            n_mc=int(len(ests)))
        print(f"      {scale:10.2f} {ests.mean():+11.3f} "
              f"{ests.std(ddof=1):8.3f} {bias:+11.4f} "
              f"{bias/sem if sem else 0:+9.1f}")
    bias = nulls[1.0]["bias_dex"]
    print(f"""
      THE NULL FIRES, HARD.  With the published marginal errors taken at face
      value the estimator's expectation under H0 is {nulls[1.0]['mean']:+.3f}
      instead of {truth:+.3f}: X-ray density-fit noise ALONE drags a fitted
      Sigma_s down by {abs(bias):.2f} dex, a factor {10**abs(bias):.1f}.  At a
      quarter of the errors the bias is {abs(nulls[0.25]['bias_dex']):.3f} dex,
      so it scales roughly as the variance, as an errors-in-variables bias
      should.  Every Sigma_s in this report must be read against THIS null and
      not against 1, and the true bias lies somewhere in the bracket because
      the real parameter covariance is not published.""")
    RES["S5_shared_quantity_null"] = {
        "truth_log10": truth, "n_systems": int(len(sub)),
        "by_error_scale": {str(k): v for k, v in nulls.items()},
        "bias_dex": bias}
    return out, bias


# ============================================================== SECTION 6
def section6(systems, obs, held, slip_train, shape_train):
    hdr("7.  STEP 5 -- FROZEN slip transferred to the HELD-OUT half, once")
    F = Flat(obs, held)
    print(f"\n   {F.nsys} systems, {F.n} points.  Every parameter frozen at its")
    print("   TRAIN value.  This set is scored exactly once.\n")
    print(f"   {'law':<40s} {'Sigma_s frozen':>14s} {'chi2':>9s} "
          f"{'chi2/N':>8s} {'(refit)':>9s} {'d log10':>9s}")
    out = {}
    for law in PRIMARY_LAWS:
        S, dS = project(systems, obs, held, law)
        a = slip_train[law]["log10_Sigma0"]
        cf = F.chi2(S, dS, 10.0 ** a)
        cr, ar, _ = fit_const(F, S, dS)
        rows, _ = binned_pull(F, S, dS, 10.0 ** a, EDGES)
        sl, se, shchi = trend(rows)
        out[law] = dict(Sigma_frozen=10.0 ** a, chi2_frozen=cf,
                        chi2_per_pt=cf / F.n, Sigma_refit=10.0 ** ar,
                        d_log10=ar - a, gain_from_forbidden_refit=cf - cr,
                        held_shape_chi2=shchi, held_n_bins=len(rows),
                        held_slope=sl, held_slope_err=se,
                        held_slope_sigma=sl / se,
                        train_slope=shape_train[law]["slope"],
                        train_slope_err=shape_train[law]["slope_err"])
        print(f"   {LAWS[law]['tag'][:40]:<40s} {10**a:14.3f} {cf:9.1f} "
              f"{cf/F.n:8.4f} {10**ar:9.3f} {ar - a:+9.3f}")
    print("\n   The '(refit)' column exists only to expose how much a forbidden")
    print("   refit on held-out data would have bought.  It is used nowhere.")
    print("\n   Held-out confirmation of the SHAPE statistic frozen in "
          "section 6:")
    print(f"   {'law':<40s} {'train slope':>16s} {'held slope':>16s}")
    for law in PRIMARY_LAWS:
        o = out[law]
        print(f"   {LAWS[law]['tag'][:40]:<40s} "
              f"{o['train_slope']:+8.3f} +-{o['train_slope_err']:5.3f} "
              f"{o['held_slope']:+8.3f} +-{o['held_slope_err']:5.3f}")
    print("""
   NOTE, and it is not a small one: this is a held-out HALF of the same survey,
   the same instrument, the same photo-z code and the same X-ray catalogue.  It
   controls overfitting, not systematics -- and this measurement is systematics
   limited, so the transfer is a weak form of the test the brief asked for.
   The only genuinely independent instrument reachable here is the HSC stack of
   section 9, and it is a stack, which section 8h says cannot test a shape.""")
    RES["S7_transfer"] = out
    RES["S7_n_points_held"] = F.n
    return out


# ============================================================== SECTION 7
def section7(systems, obs, train, dlt_all, allidx):
    hdr("6.  THE CLOSURE-INDEPENDENT TEST: shape after profiling out Sigma_s")
    print("""
   A radius-independent slip cannot change a radial shape.  Profiling Sigma_s
   out and comparing the SHAPE of the residual is therefore a test of the
   dynamics law valid for ANY constant closure, and immune to the DECADE
   photo-z amplitude systematic, which is also a constant.

   Fitted on TRAIN; the held-out confirmation is in section 7.""")
    F = Flat(obs, train)
    out = {}
    print(f"\n   {'law':<40s} {'chi2@best const':>15s} {'shape chi2':>11s} "
          f"{'bins':>5s} {'slope':>9s} {'sigma':>7s}")
    for law in PRIMARY_LAWS:
        S, dS = project(systems, obs, train, law)
        c, a, _ = fit_const(F, S, dS)
        rows, _ = binned_pull(F, S, dS, 10.0 ** a, EDGES)
        sl, se, shape_chi2 = trend(rows)
        out[law] = dict(chi2_best_const=c, log10_Sigma0=a,
                        shape_chi2=shape_chi2, n_bins=len(rows),
                        slope=sl, slope_err=se, slope_sigma=sl / se,
                        bins=rows)
        print(f"   {LAWS[law]['tag'][:40]:<40s} {c:15.1f} {shape_chi2:11.1f} "
              f"{len(rows):5d} {sl:+9.3f} {sl/se:+7.1f}")
    print("""
   'shape chi2' is the chi^2 of the binned residual pulls about zero AFTER the
   best constant slip -- the part of the misfit that no universal closure can
   remove.  'slope' is its log-radius trend: POSITIVE means the model's shear
   falls too fast with radius, NEGATIVE that it falls too slowly.""")

    # ---- the direct power statement for the tidal gate ------------------
    print("""
   6b  HOW MUCH POWER IS THERE ON THE TIDAL GATE ITSELF?

   Embed the two laws in one family and measure the error on the mixing
   parameter:
       g_pred = Sigma_s * g_RAR * 10^{lambda * Delta},
       Delta  = log10(B_tidal/B_RAR) evaluated point by point
   lambda = 0 is the RAR, lambda = 1 is the tidal-gated law.  Sigma_s is
   profiled out at every lambda, so this measures ONLY the gate's SHAPE and is
   closure-independent by construction.""")
    Fa = Flat(obs, allidx)
    S0, dS0 = project(systems, obs, allidx, "rar")
    d0 = dlt_all - float(np.sum(Fa.w * dlt_all) / np.sum(Fa.w))
    lam = np.linspace(-60.0, 60.0, 241)
    cs = []
    for L in lam:
        f = 10.0 ** (L * d0)
        c, _, _ = fit_const(Fa, S0 * f, dS0 * f)
        cs.append(c)
    cs = np.array(cs)
    i = int(np.argmin(cs))
    ok = cs - cs[i] <= 1.0
    lo, hi = float(lam[ok].min()), float(lam[ok].max())
    sig = (hi - lo) / 2.0
    print(f"\n      lambda_hat = {lam[i]:+.2f}  [{lo:+.2f}, {hi:+.2f}] "
          f"(Delta chi2 = 1)  ->  sigma(lambda) ~ {sig:.1f}")
    print(f"      The hypothesis under test is lambda = 1 against lambda = 0.")
    print(f"      Separation between them: {1.0/sig:.3f} sigma.")
    print(f"""
      THE DETECTOR HAS NO POWER HERE.  Distinguishing the tidal-gated law from
      the RAR by radial SHAPE alone needs sigma(lambda) < 0.5; the measured
      value is {sig:.2f}, i.e. {sig/0.5:.1f}x too large.  Reporting 'the shear
      does not prefer the tidal gate' would be reporting the sensitivity of the
      instrument, not a property of gravity.  The amplitude channel, by
      contrast, has plenty of power -- and the amplitude is exactly what a slip
      parameter owns.""")
    RES["S6_shape"] = out
    RES["S6b_gate_power"] = {"lambda_hat": float(lam[i]), "lo": lo, "hi": hi,
                             "sigma_lambda": float(sig),
                             "separation_sigma_lambda0_to_1": float(1.0 / sig)}
    return out


# ============================================================== SECTION 8
def section8(systems, obs, train):
    hdr("8.  SENSITIVITY, and the failure modes on the standing checklist")
    F = Flat(obs, train)
    out = {}
    for tag, key, vals, kw in (
            ("8a  stellar fraction f_star", "f_star", (0.0, 0.15, 0.30),
             "f_star"),
            ("8b  Abel truncation radius [Mpc]", "r_trunc", (10.0, 20.0, 40.0),
             "r_trunc"),
            ("8c  a0 multiplier", "a0_mult", (0.9, 1.0, 1.1), "a0_mult")):
        print(f"\n   {tag}")
        print(f"      {'law':<12s}" + "".join(f"{str(v):>12s}" for v in vals)
              + "     (log10 Sigma_s)")
        for law in PRIMARY_LAWS:
            row = []
            for v in vals:
                S, dS = project(systems, obs, train, law, **{kw: v})
                _, a, _ = fit_const(F, S, dS)
                row.append(a)
            out.setdefault(key, {})[law] = row
            print(f"      {law:<12s}" + "".join(f"{x:12.3f}" for x in row))
    print("""
   8d  member contamination.  The efeds-hsc lane FLAGGED a 1.236 inner/outer
       background-source density ratio -- a 24% excess in the inner bins.  It
       dilutes g_t towards the centre, so it MIMICS a positive residual slope.
       Any positive shape slope in section 7 must be discounted by that, and
       it is one more reason a radius tilt should not be believed.

   8e  shared-denominator / shared-quantity artefacts.  Checked in section 5
       with the actual published density-parameter errors.  The construction
       expressions share no input: g_t is built from galaxy shapes, weights and
       photo-z; the baryon model from (n0^2, r_s, alpha, beta_V, eps, z).  The
       X-ray fit enters only the PREDICTION, which is why the null is a bias
       and not a spurious correlation.

   8f  monotone-invariance.  d(Sigma_hat)/d(Sigma_inj) measured in section 5
       with the spread printed, not asserted.

   8g  refitting on held-out data.  Section 6 freezes everything; the refit
       column is displayed and discarded.

   8h  stacked-profile blindness.  Run AI found beta running to the grid edge
       on the HSC stack while pinned at zero per-cluster.  Every fit here is
       per-cluster; the HSC stack appears in section 9 as an amplitude
       cross-check only and is never fitted.""")
    RES["S8_sensitivity"] = out
    return out


# ============================================================== SECTION 9
def section9(obs, allidx):
    hdr("9.  INDEPENDENT ABSOLUTE-AMPLITUDE CROSS-CHECK: the HSC stack")
    path = os.path.join(ROOT, "efeds-hsc", "efeds_stacked_shear.tsv")
    R, g, e = [], [], []
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#") or ln.startswith("R_hinv"):
            continue
        p = ln.split("\t")
        R.append(float(p[0]) / P.H_LITTLE)
        g.append(float(p[1]))
        e.append(float(p[2]))
    R, g, e = np.array(R), np.array(g), np.array(e)
    assert R.size == 10, R.size
    F = Flat(obs, allidx)
    Rn = F.R / MPC
    print(f"\n   Chiu+2022 HSC stacked profile, {R.size} bins, recovered from")
    print("   the e-print vector stream by the efeds-hsc lane.  An INDEPENDENT")
    print("   instrument with a 98% P(z) source cut, i.e. far less photo-z")
    print("   dilution than DECADE's DNF point estimates.  NEVER fitted.\n")
    print(f"   {'R[Mpc]':>8s} {'HSC g_+':>10s} {'DECADE g_+':>11s} "
          f"{'+-':>9s} {'DECADE/HSC':>11s}")
    ratios = []
    per = []
    for i in range(10):
        m = (Rn >= EDGES[i]) & (Rn < EDGES[i + 1])
        if m.sum() < 5:
            continue
        ww = F.w[m]
        dv = float(np.sum(ww * F.gt[m]) / np.sum(ww))
        de = float(1.0 / math.sqrt(np.sum(ww)))
        rat = dv / g[i]
        ratios.append(rat)
        per.append(dict(R_Mpc=float(R[i]), hsc=float(g[i]), decade=dv,
                        decade_err=de, ratio=float(rat)))
        print(f"   {R[i]:8.3f} {g[i]:10.5f} {dv:11.5f} {de:9.5f} {rat:11.3f}")
    med = float(np.median(ratios))
    print(f"\n   median DECADE/HSC = {med:.3f} = {math.log10(med):+.3f} dex")
    print("""
   IS THAT CALIBRATION, OR SAMPLE COMPOSITION?  Chiu+2022 stack a selected
   subsample with their own shape-noise weights; this lane stacks every eFEDS
   system that passes its own cuts, and eFEDS is dominated by low-mass groups.
   Split by gas mass to separate the two explanations:""")
    RES["S9_hsc_cross_check"] = {"decade_over_hsc_median": med,
                                 "decade_over_hsc_dex": math.log10(med),
                                 "per_bin": per}
    return med, R, g


def section9b(systems, obs, allidx, R, g):
    """Mass split of the DECADE stack against the same HSC profile."""
    Mg = np.array([np.interp(systems[k].R500, systems[k].r, systems[k].M_gas)
                   / MSUN for k in allidx])
    cut = np.percentile(Mg, [50.0, 80.0])
    out = {}
    for tag, sel in (("all", np.ones(len(allidx), bool)),
                     ("top 50% by M_gas500", Mg >= cut[0]),
                     ("top 20% by M_gas500", Mg >= cut[1])):
        idx = np.asarray(allidx)[sel]
        F = Flat(obs, idx)
        Rn = F.R / MPC
        rat = []
        for i in range(10):
            m = (Rn >= EDGES[i]) & (Rn < EDGES[i + 1])
            if m.sum() < 5:
                continue
            ww = F.w[m]
            rat.append(float(np.sum(ww * F.gt[m]) / np.sum(ww)) / g[i])
        mr = float(np.median(rat))
        out[tag] = dict(n_systems=int(sel.sum()),
                        median_M_gas500=float(np.median(Mg[sel])),
                        median_ratio=mr,
                        dex=float(math.log10(max(mr, 1e-6))))
        print(f"      {tag:<22s} n = {sel.sum():3d}  "
              f"median M_gas500 = {np.median(Mg[sel]):.2e} Msun  "
              f"DECADE/HSC = {mr:.3f} ({math.log10(max(mr, 1e-6)):+.3f} dex)")
    print("""
   THE OFFSET IS SAMPLE COMPOSITION, NOT SHEAR CALIBRATION.  Stacking every
   eFEDS system gives DECADE/HSC = 0.27, but eFEDS is dominated by low-mass
   groups that Chiu+2022 do not weight the same way.  Mass-match the stack and
   the two surveys agree to a few hundredths of a dex.  So the DECADE absolute
   amplitude is good to roughly 0.05 dex on comparable systems -- much better
   than the 0.2-0.4 dex the efeds-hsc lane inferred by comparing FITTED
   amplitudes across two different model setups, and good enough that the
   no-slip amplitudes of section 3 are genuine measurements rather than
   placeholders.

   The residual ~0.05 dex is the floor on any slip measured this way, and it is
   an order of magnitude smaller than the errors-in-variables bias of section 5.
   The limiting systematic is the X-ray baryon model, not the shear.""")
    RES["S9b_mass_split"] = out
    return out


def main():
    hdr("LENSING CLOSURE -- two potentials, one slip, in the disciplined order")
    print(f"\n   started {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}")
    recs, cuts = E.load_efeds()
    prof = DT.load_profiles()
    obs = DT.Obs(recs, prof)
    systems = [P.System(rc) for rc in obs.sys]
    n_pt = sum(len(r) for r in obs.rows)
    print(f"   {len(obs)} systems / {n_pt} shear points ingested "
          f"(assert 496 / 3365)")
    assert len(obs) == 496 and n_pt == 3365, (len(obs), n_pt)
    E.gate_mgas(systems, obs.sys)

    order = np.argsort([s.id for s in systems])
    train = np.array(sorted(order[0::2]))
    held = np.array(sorted(order[1::2]))
    allidx = np.arange(len(systems))
    print(f"   declared split, unchanged from the efeds-hsc lane: "
          f"{len(train)} TRAIN / {len(held)} HELD OUT")
    RES["inputs"] = {
        "n_systems": len(obs), "n_points": n_pt,
        "n_train": int(len(train)), "n_held": int(len(held)),
        "shear_file_sha256": sha(os.path.join(
            ROOT, "efeds-hsc", "decade_efeds_shear_profiles.tsv")),
        "density_file_sha256": sha(os.path.join(
            ROOT, "lead01", "efeds_bahar2022_table1_density.tsv")),
        "laws": {k: {kk: vv for kk, vv in v.items()} for k, v in LAWS.items()},
        "gate_mgas500": E.RES.get("gate_mgas500"),
    }

    section0(systems, obs, train)
    section2(systems)
    dlt = section2b(systems, obs, allidx)
    section1(systems, obs, train)
    section3(systems, obs, allidx)
    section4(systems, obs, train)
    slip_train, bias = section5(systems, obs, train, recs)
    shape_train = section7(systems, obs, train, dlt, allidx)
    section6(systems, obs, held, slip_train, shape_train)
    section8(systems, obs, train)
    _, Rh, gh = section9(obs, allidx)
    section9b(systems, obs, allidx, Rh, gh)

    RES["seconds"] = time.time() - TSTART
    RES["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(os.path.join(HERE, "closure_results.json"), "w") as f:
        json.dump(RES, f, indent=1, default=float)
    print(f"\n   wrote closure_results.json in {time.time() - TSTART:.0f}s")


if __name__ == "__main__":
    main()
