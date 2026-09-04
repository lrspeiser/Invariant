"""Forward model for the same-object directional-gravity test.

WHAT THIS FILE IS FOR
---------------------
Streams, warps and satellites are NOT force samples.  They constrain an ORBIT in
a global potential.  This module therefore never converts a track into g_R and
g_z points.  It builds, for each candidate law:

    baryons  ->  Newtonian field  ->  candidate field equation SOLVED
             ->  conservative force  ->  progenitor orbit
             ->  tidal stripping history  ->  stream phase-space distribution
             ->  projection onto the observables the catalogue actually has.

Everything the measurement lane needs is here; `adyn_same_object.py` does the
data ingest, the gating, the fitting and the statistics.

THE DISCRIMINATOR, DECLARED BEFORE ANY DATA IS TOUCHED
------------------------------------------------------
    A_dyn(x) = [ g_R(x) / g_R,Newton(x) ] / [ |g_z(x)| / |g_z,Newton(x)| ]

evaluated at ONE POINT x, not across two different points.  This is the same
B_R / B_z used by `../tournament/ch_vertical.py`, made local.

    THEOREM (scalar blindness).  For any law of the algebraic form
        g(x) = F( scalar invariants at x ) * grad Phi_N(x)
    both components carry the same factor F(x), so A_dyn(x) = 1 IDENTICALLY,
    for every x, every F and every invariant -- including invariants that are
    themselves directional objects reduced to a scalar, such as |T|.

    So Newton, the algebraic RAR and the tidal-gated scalar a0 -> a0(1 + A W)
    ALL predict A_dyn = 1.000000 exactly.  Only a genuinely tensorial field
    equation, or the CURL FIELD that a nonlinear field equation carries for a
    non-spherical source, can move it.  That is why both field-theoretic
    completions (AQUAL and QUMOND) are solved here rather than assumed.

UNITS.  SI internally.  kpc, km/s, Msun at the interfaces.
"""
from __future__ import annotations

import hashlib
import os
import sys
from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RectBivariateSpline

# ---------------------------------------------------------------- constants
G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
GYR = 3.1557e16
KMS = 1.0e3
A0_DEFAULT = 1.2e-10

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "work", "gravitylab"))
sys.path.insert(0, os.path.join(REPO, "work", "wellnet-2026-09", "tournament"))

import axisym as AX                                              # noqa: E402
from tw_core import W_of, nu_rar, mond_invert                    # noqa: E402

# ------------------------------------------------------- solar/frame constants
# Declared in advance.  astropy v4.0 galactocentric frame defaults.
R0_SUN_KPC = 8.122
Z0_SUN_KPC = 0.0208
VSUN_KMS = (12.9, 245.6, 7.78)

#: Declared GAUGE for the MOND-family monopole potential.  |Phi| is defined
#: only up to a constant and a deep-MOND potential is log-divergent, so the
#: reference radius is fixed in advance (addendum-2 potential-gauge gate).
#: Nothing that follows depends on it: it is an additive constant on Psi.
PHI_GAUGE_R_KPC = 1.0e5


# ===========================================================================
#  1.  BARYONS  --  analytic, so the Newtonian leg is exact
# ===========================================================================
@dataclass
class MWBaryons:
    """Miyamoto-Nagai disks + a Hernquist bulge.

    Shape parameters are MEASURED baryonic quantities (inputs, per brief rule
    3); only the three masses are calibrated, and they are calibrated on the
    IN-PLANE rotation curve alone.
    """
    M_bulge: float = 0.9e10        # Msun
    a_bulge: float = 0.5           # kpc, Hernquist scale
    M_thin: float = 4.0e10
    a_thin: float = 3.0
    b_thin: float = 0.28
    M_thick: float = 1.0e10
    a_thick: float = 4.4
    b_thick: float = 0.9
    M_gas: float = 1.2e10
    a_gas: float = 7.0
    b_gas: float = 0.085

    def components(self):
        return (("hern", self.M_bulge, self.a_bulge, 0.0),
                ("mn", self.M_thin, self.a_thin, self.b_thin),
                ("mn", self.M_thick, self.a_thick, self.b_thick),
                ("mn", self.M_gas, self.a_gas, self.b_gas))

    @property
    def M_total(self):
        return self.M_bulge + self.M_thin + self.M_thick + self.M_gas

    # --------------------------------------------------------------- fields
    def phi_N(self, R_kpc, z_kpc):
        R = np.asarray(R_kpc, float) * KPC
        z = np.asarray(z_kpc, float) * KPC
        out = np.zeros(np.broadcast(R, z).shape)
        for kind, M, a, b in self.components():
            if M <= 0:
                continue
            Ms = M * MSUN
            if kind == "hern":
                r = np.sqrt(R ** 2 + z ** 2)
                out -= G * Ms / (r + a * KPC)
            else:
                den = np.sqrt(R ** 2 + (a * KPC + np.sqrt(z ** 2
                                                          + (b * KPC) ** 2)) ** 2)
                out -= G * Ms / den
        return out

    def g_N(self, R_kpc, z_kpc):
        """(g_R, g_z) in m/s^2, both NEGATIVE-inward sign convention:
        returns the magnitude of the inward radial force and of the
        toward-midplane vertical force, i.e. g_R >= 0, sign(g_z) = -sign(z)."""
        R = np.asarray(R_kpc, float) * KPC
        z = np.asarray(z_kpc, float) * KPC
        gR = np.zeros(np.broadcast(R, z).shape)
        gz = np.zeros_like(gR)
        for kind, M, a, b in self.components():
            if M <= 0:
                continue
            Ms = M * MSUN
            if kind == "hern":
                r = np.sqrt(R ** 2 + z ** 2)
                r = np.maximum(r, 1e-6 * KPC)
                f = G * Ms / (r + a * KPC) ** 2
                gR += f * R / r
                gz += f * z / r
            else:
                zb = np.sqrt(z ** 2 + (b * KPC) ** 2)
                s = a * KPC + zb
                den = (R ** 2 + s ** 2) ** 1.5
                gR += G * Ms * R / den
                gz += G * Ms * z * s / (zb * den)
        return gR, gz

    def rho(self, R_kpc, z_kpc):
        """kg/m^3 on a (R, z) mesh."""
        R = np.asarray(R_kpc, float) * KPC
        z = np.asarray(z_kpc, float) * KPC
        out = np.zeros(np.broadcast(R, z).shape)
        for kind, M, a, b in self.components():
            if M <= 0:
                continue
            Ms, aa, bb = M * MSUN, a * KPC, b * KPC
            if kind == "hern":
                r = np.maximum(np.sqrt(R ** 2 + z ** 2), 1e-4 * KPC)
                out += Ms * aa / (2 * np.pi * r * (r + aa) ** 3)
            else:
                zb = np.sqrt(z ** 2 + bb ** 2)
                s = aa + zb
                num = (aa * R ** 2 + (aa + 3 * zb) * s ** 2)
                out += (bb ** 2 * Ms / (4 * np.pi)
                        * num / ((R ** 2 + s ** 2) ** 2.5 * zb ** 3))
        return out

    def rho_cell_average(self, Rc, zc, dR, dz, nsub=12):
        """Cell-AVERAGED density on a uniform (R,z) grid.

        Point-sampling rho at the cell centre loses 27% of the disc mass at
        dz = 0.62 kpc against a Miyamoto-Nagai b = 0.28 kpc -- the vertical
        structure is entirely unresolved, the solved rotation curve comes out
        at 0.73 of the analytic one, and every multiplier is then evaluated
        for the wrong galaxy.  Gauss-Legendre sub-sampling inside each cell
        fixes the cell masses; `mass_gate` checks the result against the
        analytic total.  Volume weighting is 2 pi R dR dz, so the R-average is
        R-weighted, matching the finite-volume cell volumes exactly.
        """
        x, w = np.polynomial.legendre.leggauss(nsub)
        Rs = Rc[:, None] + 0.5 * dR * x[None, :]           # (nR, nsub)
        zs = zc[:, None] + 0.5 * dz * x[None, :]           # (nz, nsub)
        wR = w * Rs                                        # R-weighted
        wR = wR / wR.sum(axis=1, keepdims=True)
        wz = w / w.sum()
        acc = np.zeros((len(Rc), len(zc)))
        for i in range(nsub):
            for j in range(nsub):
                acc += (wR[:, i][:, None] * wz[j]
                        * self.rho(Rs[:, i][:, None], zs[:, j][None, :]))
        return acc

    def tidal_invariant(self, R_kpc, z_kpc, h=0.02):
        """|T| = spectral-norm-free Frobenius norm of the Newtonian tidal
        tensor, matched to `../tournament/ch_vertical.py`'s `tidal` invariant
        (which uses the same second-derivative construction)."""
        R = np.atleast_1d(np.asarray(R_kpc, float))
        z = np.atleast_1d(np.asarray(z_kpc, float))
        dR = h * np.maximum(np.abs(R), 0.5)
        dz = h * np.maximum(np.abs(z), 0.05)
        gR_p, gz_p = self.g_N(R + dR, z)
        gR_m, gz_m = self.g_N(R - dR, z)
        gR_zp, gz_zp = self.g_N(R, z + dz)
        gR_zm, gz_zm = self.g_N(R, z - dz)
        d_RR = -(gR_p - gR_m) / (2 * dR * KPC)
        d_zz = -(gz_zp - gz_zm) / (2 * dz * KPC)
        d_Rz = -0.5 * ((gR_zp - gR_zm) / (2 * dz * KPC)
                       + (gz_p - gz_m) / (2 * dR * KPC))
        gR, _ = self.g_N(R, z)
        d_pp = -gR / np.maximum(R * KPC, 1e-4 * KPC)   # phi-phi component
        return np.sqrt(d_RR ** 2 + d_zz ** 2 + d_pp ** 2 + 2 * d_Rz ** 2)


# ===========================================================================
#  2.  CANDIDATE LAWS
# ===========================================================================
@dataclass
class Law:
    """A frozen candidate.  Gravity constants are GLOBAL; nothing per object."""
    name: str
    base: str = "rar"                # 'newton' | 'rar' | 'aqual'
    a0: float = A0_DEFAULT
    gate: str = "none"               # 'none' | 'tidal' | 'phi'
    form: str = "inv"
    m: float = 2.0
    I0: float = 1.0
    A: float = 0.0
    struct: str = "scalar_a0"        # 'scalar_a0' | 'tensor_S'
    completion: str = "algebraic"    # 'algebraic' | 'qumond' | 'aqual'
    note: str = ""

    def a0_eff(self, bar: MWBaryons, R, z):
        if self.gate == "none" or self.struct != "scalar_a0":
            return np.full(np.broadcast(np.asarray(R, float),
                                        np.asarray(z, float)).shape, self.a0)
        if self.gate == "tidal":
            I = bar.tidal_invariant(R, z) / self.I0
        elif self.gate == "phi":
            I = np.abs(bar.phi_N(R, z)) / self.I0
        else:
            raise ValueError(self.gate)
        return np.maximum(self.a0 * (1.0 + self.A * W_of(self.form, I, self.m)),
                          1e-30)

    def k_eigen(self, bar: MWBaryons, R, z):
        """(k_RR, k_zz), the conductivity eigenvalues of the well-network
        tensor K = exp(A W S).  S is traceless with |S|_2 = 2/3 saturated
        (Run AB's boundedness theorem), so with a = A W:
            k_zz = e^{-a/3} + (e^{2a/3} - e^{-a/3}) dz^2
            k_RR = e^{-a/3} + (e^{2a/3} - e^{-a/3}) (1 - dz^2)
        with dz the z-component of the dominant well direction.  Identical to
        `../tournament/ch_vertical.py`."""
        R = np.asarray(R, float)
        z = np.asarray(z, float)
        one = np.ones(np.broadcast(R, z).shape)
        if self.struct != "tensor_S":
            return one, one.copy()
        if self.gate == "phi":
            I = np.abs(bar.phi_N(R, z)) / self.I0
        elif self.gate == "tidal":
            I = bar.tidal_invariant(R, z) / self.I0
        else:
            I = one
        a = np.clip(self.A * W_of(self.form, I, self.m), -60.0, 60.0)
        rr = np.maximum(np.sqrt(R ** 2 + z ** 2), 1e-8)
        dz2 = (z / rr) ** 2 * one
        e_a, e_b = np.exp(-a / 3.0), np.exp(2.0 * a / 3.0)
        return (np.maximum(e_a + (e_b - e_a) * (1.0 - dz2), 1e-30),
                np.maximum(e_a + (e_b - e_a) * dz2, 1e-30))

    # ------------------------------------------------------ algebraic force
    def g_algebraic(self, bar: MWBaryons, R, z):
        """The pointwise law: no field solve, no curl field.

        This is the form the tournament froze and the form in which
        A_dyn == 1 is a THEOREM for every scalar structure."""
        gRN, gzN = bar.g_N(R, z)
        gN = np.sqrt(gRN ** 2 + gzN ** 2)
        if self.base == "newton" and self.struct == "scalar_a0":
            return gRN, gzN
        if self.struct == "scalar_a0":
            a0e = self.a0_eff(bar, R, z)
            g = _g_of_gN(self.base, gN, a0e)
            F = g / np.maximum(gN, 1e-300)
            return F * gRN, F * gzN
        # tensor: each direction gets its own conductivity eigenvalue
        k_RR, k_zz = self.k_eigen(bar, R, z)
        gR = mond_invert(gRN, k_RR, self.a0, self.base)
        gz = mond_invert(np.abs(gzN), k_zz, self.a0, self.base) * np.sign(gzN)
        return gR, gz


def _g_of_gN(base, gN, a0):
    gN = np.asarray(gN, float)
    if base == "newton":
        return gN.copy()
    if base == "rar":
        return nu_rar(gN / a0) * gN
    if base == "aqual":
        return 0.5 * (gN + np.sqrt(gN ** 2 + 4.0 * gN * a0))
    raise ValueError(base)


# --------------------------------------------------------- THE FROZEN SET
def frozen_laws():
    """The candidates named in the brief, with the tournament's frozen
    constants.  `../tournament/REPORT.md` sections 0.4 and 0.1."""
    return [
        Law("newton", base="newton", note="control"),
        Law("rar", base="rar", a0=1.0844e-10,
            note="BASE_rar, tournament _checkpoint.json a0"),
        Law("aqual", base="aqual", a0=1.0844e-10,
            note="AQUAL base at the same a0"),
        Law("tidal_scalar", base="aqual", a0=1.002e-10, gate="tidal",
            form="inv", m=2.0, I0=1e-33, A=16.0, struct="scalar_a0",
            note="aqual|scalar_a0|tidal|inv m=2|T0=1e-33, A=+16.0 "
                 "(tournament REPORT.md section 0.4)"),
        Law("wellnet_tensor", base="aqual", a0=1.04115625e-10, gate="phi",
            form="sat", m=2.0, I0=3e12, A=-94.65724194487994,
            struct="tensor_S",
            note="aqual|tensor_S[plaw p=0]|phi|sat|m=2|Phi0=3e12, "
                 "A from tournament finalise.json survivor_hygiene[0]"),
    ]


# ===========================================================================
#  3.  FIELD SOLVES -- the conservative completions
# ===========================================================================
class FieldSolution:
    """Solved Phi(R, z) for one law, on a uniform cylindrical grid.

    QUMOND  :  div grad Phi = div[ nu(|grad Phi_N|/a0) grad Phi_N ]   (LINEAR)
    AQUAL   :  div[ mu(|grad Phi|/a0) K grad Phi ] = 4 pi G rho       (Picard)

    Both are exact statements of the field equation, so both carry the CURL
    FIELD that the algebraic law throws away.  That curl field is the only
    thing in Newton/RAR/AQUAL/tidal-scalar that can push A_dyn off 1.
    """

    def __init__(self, bar: MWBaryons, law: Law, nR=420, nz=420,
                 Rmax=260.0, zmax=260.0, tol=1e-10, picard=40, verbose=False,
                 ref=None):
        """`ref`: a Newtonian FieldSolution on the SAME grid.  When supplied,
        the multipliers B_R and B_z are formed against the SOLVED Newtonian
        force rather than the analytic one, so the leading discretisation
        error cancels between numerator and denominator.  Without it the
        grid's own A_dyn error is 6e-2 at (8, 1) kpc and 1.3e-3 at (25, 10)
        kpc, which is larger than the effect being measured."""
        self.bar, self.law = bar, law
        self.g = AX.Grid(nR, nz, Rmax, zmax)
        self.Rk = self.g.Rc / KPC
        self.zk = self.g.zc / KPC
        RR = self.Rk[:, None] * np.ones((1, nz))
        ZZ = np.ones((nR, 1)) * self.zk[None, :]
        self.RR, self.ZZ = RR, ZZ
        self.rho = bar.rho_cell_average(self.Rk, self.zk,
                                        self.g.dR / KPC, self.g.dz / KPC)
        self.Mtot = bar.M_total * MSUN
        self.M_on_grid = float(2.0 * np.sum(self.rho * self.g.V) / MSUN)
        self.mass_gate = self.M_on_grid / bar.M_total
        self.n_picard = 0
        self.resid = np.nan
        self.verbose = verbose
        self.ref = ref
        self._solve(tol, picard)
        self._build_splines()

    @staticmethod
    def newtonian_reference(bar, **kw):
        lawN = Law("newton_ref", base="newton")
        lawN.completion = "poisson"
        return FieldSolution(bar, lawN, **kw)

    # ------------------------------------------------------------- solvers
    def _bc(self):
        """Outer Dirichlet shell.  For a MOND-family law the monopole is the
        deep-MOND logarithm, not -GM/r; using -GM/r would impose Newtonian
        asymptotics on the boundary and contaminate the whole solve.  The
        far cutoff is a declared GAUGE choice: an additive constant on Psi
        changes no force."""
        r = np.sqrt(self.RR ** 2 + self.ZZ ** 2) * KPC
        if self.law.base == "newton":
            return -G * self.Mtot / r
        rr = np.geomspace(0.05 * KPC, PHI_GAUGE_R_KPC * KPC, 20000)
        gm = _g_of_gN(self.law.base, G * self.Mtot / rr ** 2, self.law.a0)
        I = np.concatenate([[0.0], np.cumsum(0.5 * (gm[1:] + gm[:-1])
                                             * np.diff(rr))])
        phi = -(I[-1] - I)            # Phi(r) = -int_r^{Rgauge} g dr'
        return np.interp(r, rr, phi)

    def _solve(self, tol, picard):
        g = self.g
        bc = self._bc()
        shape = self.rho.shape
        law = self.law
        if law.completion == "poisson":
            A = AX.isotropic_A(shape)
            Psi, it, rel = solve_axi_warm(self.rho, A, g, bc, tol=tol)
            self.n_picard, self.resid = it, rel
        elif law.completion == "qumond":
            # div grad Phi = div[ nu(|grad Phi_N|/a0_eff) grad Phi_N ],
            # LINEAR in Phi, so one solve.  The source is built on the FACE
            # fluxes the discretisation conserves (brief: flux must be
            # measured on the faces, not reconstructed).
            gRN, gzN = self.bar.g_N(self.RR, self.ZZ)
            gN = np.sqrt(gRN ** 2 + gzN ** 2)
            a0e = law.a0_eff(self.bar, self.RR, self.ZZ)
            nu = _g_of_gN(law.base, gN, a0e) / np.maximum(gN, 1e-300)
            src = _face_divergence(nu * gRN, nu * gzN, g)
            A = AX.isotropic_A(shape)
            Psi, it, rel = solve_axi_warm(src / (4 * np.pi * G), A, g, bc,
                                          tol=tol)
            self.n_picard, self.resid = it, rel
        elif law.completion == "aqual":
            A = AX.isotropic_A(shape)
            Psi, it, rel = solve_axi_warm(self.rho, A, g, bc, tol=1e-8)
            k_RR, k_zz = law.k_eigen(self.bar, self.RR, self.ZZ)
            a0e = law.a0_eff(self.bar, self.RR, self.ZZ)
            for p in range(picard):
                gR, gz = _grad(Psi, g)
                gmag = np.sqrt((np.sqrt(k_RR) * gR) ** 2
                               + (np.sqrt(k_zz) * gz) ** 2)
                x = gmag / a0e
                if law.base == "aqual":
                    mu = x / (1.0 + x)
                else:                                # RAR mu = 1 - exp(-sqrt)
                    mu = 1.0 - np.exp(-np.sqrt(np.maximum(x, 0.0)))
                mu = np.maximum(mu, 1e-12)
                A = (mu * k_RR, mu * k_zz, np.zeros(shape))
                Psi_new, it, rel = solve_axi_warm(self.rho, A, g, bc, x0=Psi,
                                                  tol=tol)
                d = np.max(np.abs(Psi_new - Psi)) / np.max(np.abs(Psi))
                Psi = 0.4 * Psi + 0.6 * Psi_new           # damped Picard
                self.n_picard, self.resid = p + 1, rel
                if self.verbose:
                    print(f"    picard {p:2d} cg={it:5d} dPsi={d:.3e} "
                          f"rel={rel:.2e}", flush=True)
                if d < 3e-7:
                    break
        else:                                              # 'algebraic'
            Psi = None
        self.Psi = Psi

    # ------------------------------------------------------- interpolation
    def _build_splines(self):
        """Store the force grid and the O(1) ratio fields B_R = g_R/g_R,N and
        B_z = g_z/g_z,N.  A_dyn is B_R/B_z at the SAME point."""
        if self.Psi is None:
            gR, gz = self.law.g_algebraic(self.bar, self.RR, self.ZZ)
            self.Phi = _phi_from_forces_grid(self, gR, gz)
        else:
            gR, gz = _grad(self.Psi, self.g)
            self.Phi = self.Psi.copy()
        self.gR_grid, self.gz_grid = gR, gz
        eps = 1e-40
        if self.ref is not None:
            # discretisation-cancelling reference: SOLVED law over SOLVED
            # Newton on the identical grid and stencil
            gRN, gzN = self.ref.gR_grid, np.abs(self.ref.gz_grid)
        else:
            gRN, gzN = self.bar.g_N(self.RR, self.ZZ)
        self.BR_grid = gR / np.maximum(gRN, eps)
        self.Bz_grid = np.abs(gz) / np.maximum(np.abs(gzN), eps)
        self.sp_BR = RectBivariateSpline(self.Rk, self.zk,
                                         np.log(np.maximum(self.BR_grid, 1e-8)),
                                         kx=3, ky=3, s=0)
        self.sp_Bz = RectBivariateSpline(self.Rk, self.zk,
                                         np.log(np.maximum(self.Bz_grid, 1e-8)),
                                         kx=3, ky=3, s=0)

    def multipliers(self, R_kpc, absz_kpc):
        R = np.clip(np.asarray(R_kpc, float), self.Rk[0], self.Rk[-1])
        z = np.clip(np.asarray(absz_kpc, float), self.zk[0], self.zk[-1])
        return np.exp(self.sp_BR.ev(R, z)), np.exp(self.sp_Bz.ev(R, z))

    def A_dyn_field(self, R_kpc, absz_kpc):
        BR, Bz = self.multipliers(R_kpc, absz_kpc)
        return BR / np.maximum(Bz, 1e-30)

    def vc_midplane(self, R_kpc):
        """Circular speed in km/s from the SOLVED field, at the grid's first
        cell-centred z row (z = dz/2), which is the midplane proxy."""
        R = np.clip(np.asarray(R_kpc, float), self.Rk[0], self.Rk[-1])
        BR = np.exp(self.sp_BR.ev(R, np.full_like(R, self.zk[0])))
        gRN, _ = self.bar.g_N(R, np.zeros_like(R))
        return np.sqrt(np.maximum(BR * gRN * R * KPC, 0.0)) / KMS


def _diag_axi(A, g: AX.Grid):
    """Diagonal of the finite-volume operator, for Jacobi preconditioning."""
    ARR, Azz, _ = A
    aRR = np.zeros((g.nR + 1, g.nz))
    aRR[1:-1] = 0.5 * (ARR[1:] + ARR[:-1])
    azz = np.zeros((g.nR, g.nz + 1))
    azz[:, 1:-1] = 0.5 * (Azz[:, 1:] + Azz[:, :-1])
    d = -(aRR[1:] * g.AR[1:] + aRR[:-1] * g.AR[:-1]) / (g.dR * g.V) \
        - (azz[:, 1:] * g.Az + azz[:, :-1] * g.Az) / (g.dz * g.V)
    return np.where(np.abs(d) < 1e-300, -1.0, d)


def solve_axi_warm(rho, A, g: AX.Grid, Psi_bc, x0=None, tol=1e-10,
                   maxiter=6000):
    """Jacobi-preconditioned CG with a warm start.

    Same discretisation as `axisym.solve_axi` -- it calls the same
    `apply_axi`, so the seven solver gates carry over unchanged.  The only
    additions are the preconditioner and the initial guess, both of which
    change the iteration count and nothing else.  Needed because the AQUAL
    Picard loop is 20-40 full solves per law.
    """
    mask = np.ones(rho.shape, bool)
    mask[-1, :] = False
    mask[:, -1] = False
    x = np.where(mask, 0.0 if x0 is None else x0, Psi_bc)
    b = 4 * np.pi * G * rho
    Minv = 1.0 / _diag_axi(A, g)
    r = (b - AX.apply_axi(x, A, g)) * mask
    zv = r * Minv * mask
    p = zv.copy()
    rz = float(np.sum(r * zv * g.V))
    bn = float(np.sqrt(np.sum((b * mask) ** 2 * g.V))) or 1.0
    rel = float(np.sqrt(np.sum(r * r * g.V))) / bn
    for it in range(maxiter):
        if rel < tol:
            return x, it, rel
        Ap = AX.apply_axi(p * mask, A, g) * mask
        den = float(np.sum(p * Ap * g.V))
        if den == 0.0:
            break
        al = rz / den
        x = x + al * p * mask
        r = r - al * Ap
        rel = float(np.sqrt(np.sum(r * r * g.V))) / bn
        zv = r * Minv * mask
        rz_new = float(np.sum(r * zv * g.V))
        p = zv + (rz_new / rz) * p
        rz = rz_new
    return x, maxiter, rel


def _asymptotic_tail(law, Mtot, r):
    """int_r^inf g dr for the base law's monopole, so Phi -> 0 at infinity."""
    if law.base == "newton":
        return G * Mtot / r
    rr = np.geomspace(r, 1e5 * KPC, 4000)
    gN = G * Mtot / rr ** 2
    gm = _g_of_gN(law.base, gN, law.a0)
    return float(np.trapezoid(gm, rr))


def _nu_inv(x):
    return nu_rar(np.maximum(x, 1e-300))


def _grad(Psi, g: AX.Grid):
    """Centred gradient of Psi at cell centres, midplane-symmetric in z."""
    gR = np.zeros_like(Psi)
    gR[1:-1] = (Psi[2:] - Psi[:-2]) / (2 * g.dR)
    gR[0] = (Psi[1] - Psi[0]) / g.dR
    gR[-1] = (Psi[-1] - Psi[-2]) / g.dR
    gz = np.zeros_like(Psi)
    gz[:, 1:-1] = (Psi[:, 2:] - Psi[:, :-2]) / (2 * g.dz)
    gz[:, 0] = (Psi[:, 1] - Psi[:, 0]) / g.dz
    gz[:, -1] = (Psi[:, -1] - Psi[:, -2]) / g.dz
    return gR, gz


def _face_divergence(FR, Fz, g: AX.Grid):
    """div F using the SAME face areas the solver conserves."""
    fR = np.zeros((g.nR + 1, g.nz))
    fR[1:-1] = 0.5 * (FR[1:] + FR[:-1])
    fR[-1] = FR[-1]
    fz = np.zeros((g.nR, g.nz + 1))
    fz[:, 1:-1] = 0.5 * (Fz[:, 1:] + Fz[:, :-1])
    fz[:, 0] = 0.0                                   # midplane symmetry
    fz[:, -1] = Fz[:, -1]
    FRa, Fza = fR * g.AR, fz * g.Az
    return (FRa[1:] - FRa[:-1] + Fza[:, 1:] - Fza[:, :-1]) / g.V


# ===========================================================================
#  4.  THE MEASUREMENT FAMILY  --  one parameter, in-plane leg exactly frozen
# ===========================================================================
class DeformedField:
    """Phi_L(R,z) = Phi(R,0) + (1/L) [ Phi(R,z) - Phi(R,0) ].

    L = 1 is the frozen candidate.  For every L:
      * the MIDPLANE rotation curve is unchanged EXACTLY -- the in-plane leg
        is frozen by construction, not by a fit.  This is the whole point:
        step 1 fits the disc, step 2 freezes, and no value of L can undo it.
      * g_z is multiplied by 1/L everywhere, so K_z/K_z,N -> B_z/L;
      * the force stays the gradient of a potential, so it is conservative and
        passes the reciprocity/action gate;
      * A_dyn -> approximately L * A_dyn(frozen); the exact value is computed,
        never assumed, by `A_dyn()`.

    The gradient of the deformation is exact in closed form,
        g_R,L(R,z) = g_R(R,0) + [g_R(R,z) - g_R(R,0)] / L
        g_z,L(R,z) =                        g_z(R,z)  / L
    so no re-differentiation of an interpolant is needed.  Interpolation is
    bilinear on a refined uniform grid, which is fast enough to forward-model
    every stream at every L; the resulting energy drift is measured, not
    assumed (see `energy_drift`).
    """

    def __init__(self, sol: FieldSolution, Lam: float, refine=2):
        self.sol, self.Lam = sol, float(Lam)
        g = sol.g
        # refined uniform grid, bilinear-ready
        self.R0 = float(sol.Rk[0])
        self.z0 = float(sol.zk[0])
        self.dR = float(sol.Rk[1] - sol.Rk[0]) / refine
        self.dz = float(sol.zk[1] - sol.zk[0]) / refine
        self.nR = (len(sol.Rk) - 1) * refine + 1
        self.nz = (len(sol.zk) - 1) * refine + 1
        self.Rk = self.R0 + self.dR * np.arange(self.nR)
        self.zk = self.z0 + self.dz * np.arange(self.nz)
        RR = self.Rk[:, None] * np.ones((1, self.nz))
        ZZ = np.ones((self.nR, 1)) * self.zk[None, :]
        BR = np.exp(sol.sp_BR.ev(RR, ZZ))
        Bz = np.exp(sol.sp_Bz.ev(RR, ZZ))
        gRN, gzN = sol.bar.g_N(RR, ZZ)
        gR = BR * gRN
        gz = Bz * np.abs(gzN)
        gR0 = gR[:, :1]
        self.gR = gR0 + (gR - gR0) / self.Lam
        self.gz = gz / self.Lam
        # potential, for energies and the boundness check
        phi_mid = np.concatenate([[0.0], np.cumsum(
            0.5 * (gR[1:, 0] + gR[:-1, 0]) * self.dR * KPC)])
        phi_vert = np.concatenate([np.zeros((self.nR, 1)), np.cumsum(
            0.5 * (self.gz[:, 1:] + self.gz[:, :-1]) * self.dz * KPC,
            axis=1)], axis=1)
        self.Phi = phi_mid[:, None] + phi_vert
        self.Phi -= self.Phi[-1, -1] - sol.Phi[-1, -1]
        self._pack()

    # ------------------------------------------------------------- lookup
    def _pack(self):
        """One interleaved (nR, nz, 2) table so a force evaluation costs ONE
        gather instead of two.  The orbit integrator calls this a few million
        times per law, so the packing is worth it."""
        self._G2 = np.ascontiguousarray(
            np.stack([self.gR, self.gz], axis=-1).reshape(-1, 2))

    def _bilin(self, A, R, z):
        fi = (R - self.R0) / self.dR
        fj = (z - self.z0) / self.dz
        i = np.clip(fi.astype(np.int64), 0, self.nR - 2)
        j = np.clip(fj.astype(np.int64), 0, self.nz - 2)
        u = np.clip(fi - i, 0.0, 1.0)
        v = np.clip(fj - j, 0.0, 1.0)
        return ((1 - u) * (1 - v) * A[i, j] + u * (1 - v) * A[i + 1, j]
                + (1 - u) * v * A[i, j + 1] + u * v * A[i + 1, j + 1])

    def force(self, R_kpc, absz_kpc):
        """(g_R, |g_z|) in m/s^2, both >= 0."""
        R = np.clip(np.asarray(R_kpc, float), self.R0, self.Rk[-1])
        z = np.clip(np.asarray(absz_kpc, float), self.z0, self.zk[-1])
        fi = (R - self.R0) * (1.0 / self.dR)
        fj = (z - self.z0) * (1.0 / self.dz)
        i = fi.astype(np.int64)
        j = fj.astype(np.int64)
        np.clip(i, 0, self.nR - 2, out=i)
        np.clip(j, 0, self.nz - 2, out=j)
        u = (fi - i)[:, None]
        v = (fj - j)[:, None]
        k = i * self.nz + j
        G = self._G2
        out = ((1 - u) * (1 - v)) * G[k] + (u * (1 - v)) * G[k + self.nz] \
            + ((1 - u) * v) * G[k + 1] + (u * v) * G[k + self.nz + 1]
        return out[:, 0], out[:, 1]

    def phi(self, R_kpc, absz_kpc):
        R = np.clip(np.asarray(R_kpc, float), self.R0, self.Rk[-1])
        z = np.clip(np.asarray(absz_kpc, float), self.z0, self.zk[-1])
        return self._bilin(self.Phi, R, z)

    def A_dyn(self, R_kpc, absz_kpc):
        gR, gz = self.force(R_kpc, absz_kpc)
        gRN, gzN = self.sol.bar.g_N(R_kpc, absz_kpc)
        return ((gR / np.maximum(gRN, 1e-40))
                / np.maximum(gz / np.maximum(np.abs(gzN), 1e-40), 1e-40))


def _phi_from_forces_grid(sol: "FieldSolution", gR, gz):
    """Phi(R,z) for the algebraic law, which has no exact potential.  Built by
    integrating g_R along the midplane then g_z in z; the residual curl is
    measured by `curl_defect`, never assumed to be zero."""
    g = sol.g
    phi0 = np.concatenate([[0.0], np.cumsum(0.5 * (gR[1:, 0] + gR[:-1, 0])
                                            * g.dR)])
    phiz = np.concatenate([np.zeros((g.nR, 1)),
                           np.cumsum(0.5 * (gz[:, 1:] + gz[:, :-1]) * g.dz,
                                     axis=1)], axis=1)
    Phi = phi0[:, None] + phiz
    r = np.sqrt(sol.Rk[-1] ** 2 + sol.zk[-1] ** 2) * KPC
    if sol.law.base == "newton":
        target = -G * sol.Mtot / r
    else:
        rr = np.geomspace(0.05 * KPC, PHI_GAUGE_R_KPC * KPC, 20000)
        gm = _g_of_gN(sol.law.base, G * sol.Mtot / rr ** 2, sol.law.a0)
        I = np.concatenate([[0.0], np.cumsum(0.5 * (gm[1:] + gm[:-1])
                                             * np.diff(rr))])
        target = float(np.interp(r, rr, -(I[-1] - I)))
    return Phi - Phi[-1, -1] + target


def curl_defect(sol: FieldSolution, bar: MWBaryons):
    """max |curl g| * L / |g| over the grid, L = 10 kpc.  Zero for a solved
    completion; the size of the algebraic law's inconsistency otherwise."""
    gR, gz = sol.law.g_algebraic(bar, sol.RR, sol.ZZ)
    g = sol.g
    dgR_dz = np.zeros_like(gR)
    dgR_dz[:, 1:-1] = (gR[:, 2:] - gR[:, :-2]) / (2 * g.dz)
    dgz_dR = np.zeros_like(gz)
    dgz_dR[1:-1] = (gz[2:] - gz[:-2]) / (2 * g.dR)
    curl = np.abs(dgR_dz - dgz_dR)
    mag = np.sqrt(gR ** 2 + gz ** 2) / (10.0 * KPC)
    sel = (sol.RR > 2) & (sol.RR < 100) & (sol.ZZ < 100)
    return float(np.max(curl[sel] / np.maximum(mag[sel], 1e-40)))


# ===========================================================================
#  5.  ORBIT INTEGRATION  --  vectorised leapfrog in the axisymmetric field
# ===========================================================================
def accel(field: DeformedField, w):
    """w: (N, 6) [x, y, z, vx, vy, vz] in SI.  Returns (N, 3) acceleration."""
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    R = np.sqrt(x * x + y * y)
    Rk = R / KPC
    zk = np.abs(z) / KPC
    gR, gz = field.force(Rk, zk)
    Rs = np.maximum(R, 1e-6 * KPC)
    a = np.empty((w.shape[0], 3))
    a[:, 0] = -gR * x / Rs
    a[:, 1] = -gR * y / Rs
    a[:, 2] = -gz * np.sign(z)
    return a


def leapfrog(field: DeformedField, w0, dt, nstep, release=None):
    """Kick-drift-kick.  `release` is an optional list of
    (step_index, particle_mask, state) injections used by the stream model to
    activate a particle at its own stripping time.  Returns the final state."""
    w = np.array(w0, float, copy=True)
    a = accel(field, w)
    for j in range(nstep):
        if release is not None:
            jj = release.get(j)
            if jj is not None:
                idx, st = jj
                w[idx] = st
                a = accel(field, w)
        w[:, 3:] += 0.5 * dt * a
        w[:, :3] += dt * w[:, 3:]
        a = accel(field, w)
        w[:, 3:] += 0.5 * dt * a
    return w


def integrate_orbit(field, w0, dt, nstep, store_every=1):
    """Forward/backward orbit with storage.  dt may be negative."""
    w = np.array(w0, float, copy=True)
    nkeep = nstep // store_every + 1
    out = np.empty((nkeep, w.shape[0], 6))
    out[0] = w
    a = accel(field, w)
    k = 1
    for j in range(nstep):
        w[:, 3:] += 0.5 * dt * a
        w[:, :3] += dt * w[:, 3:]
        a = accel(field, w)
        w[:, 3:] += 0.5 * dt * a
        if (j + 1) % store_every == 0 and k < nkeep:
            out[k] = w
            k += 1
    return out[:k]


def energy(field: DeformedField, w):
    R = np.sqrt(w[:, 0] ** 2 + w[:, 1] ** 2) / KPC
    z = np.abs(w[:, 2]) / KPC
    return 0.5 * np.sum(w[:, 3:] ** 2, axis=1) + field.phi(R, z)


def energy_drift(field: DeformedField, w0, dt, nstep):
    """max |dE/E| over an integration.  The bilinear force is curl-free only
    to O(h), so this is MEASURED for every configuration used."""
    E0 = energy(field, w0)
    w = leapfrog(field, w0, dt, nstep)
    E1 = energy(field, w)
    return float(np.max(np.abs((E1 - E0) / E0)))


# ===========================================================================
#  6.  STREAM FORWARD MODEL  --  particle spray (Fardal et al. 2015)
# ===========================================================================
#: Fardal+2015 mean and scatter of the release parameters, in units of the
#: tidal radius / the local circular frequency.  These are calibrated on
#: N-body stripping and are the standard prescription; they are NOT fitted.
FARDAL = dict(kr=2.0, s_kr=0.5, kvphi=0.3, s_kvphi=0.5,
              s_kz=0.5, s_kvz=0.5)


def tidal_radius(field: DeformedField, w, m_prog_msun):
    """r_t = r ( m / (3 M_enc_eff) )^(1/3), with M_enc_eff defined from the
    LOCAL force of the candidate law, not from Newtonian mass:
        M_enc_eff = |g| r^2 / G
    so it is the correct tidal scale in the modified field."""
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    R = np.sqrt(x * x + y * y)
    gR, gz = field.force(R / KPC, np.abs(z) / KPC)
    gmag = np.sqrt(gR ** 2 + gz ** 2)
    Menc = gmag * r ** 2 / G
    return r * (m_prog_msun * MSUN / (3.0 * np.maximum(Menc, 1e-30))) ** (1 / 3)


def spray_stream(field: DeformedField, w_anchor, m_prog_msun, T_strip_gyr,
                 n_release, dt_myr=1.5, rng=None, n_per_release=2):
    """Generate streams for a BATCH of anchors, all in one vectorised pass.

    1. integrate every progenitor BACK from its anchor to -T_strip;
    2. walk forward, releasing `n_per_release` particles at each of
       `n_release` evenly spaced times, offset from the progenitor by the
       Fardal+2015 prescription at the LOCAL tidal radius of the CANDIDATE
       field (M_enc from |g| r^2/G, not from Newtonian mass);
    3. integrate every particle forward to t = 0 in the SAME field.

    w_anchor: (K, 6).  Returns (K, npart, 6) present-day phase space.
    The whole point of this routine is that a stream is never treated as a
    force sample: the observable is the phase-space distribution this
    stripping history produces in this potential.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    w_anchor = np.atleast_2d(np.asarray(w_anchor, float))
    K = w_anchor.shape[0]
    dt = dt_myr * 1e6 * 3.1557e7
    nstep = int(round(T_strip_gyr * 1e9 * 3.1557e7 / dt))
    back = integrate_orbit(field, w_anchor, -dt, nstep, store_every=1)[::-1]
    prog = back                                          # (nstep+1, K, 6)

    steps = np.unique(np.linspace(0, nstep - 1, n_release).astype(int))
    ns = len(steps)
    npart = ns * n_per_release
    rel_step = np.repeat(steps, n_per_release)           # (npart,)
    sign = np.tile(np.where(np.arange(n_per_release) % 2 == 0, 1.0, -1.0), ns)

    ws = prog[rel_step]                                  # (npart, K, 6)
    ws = np.swapaxes(ws, 0, 1).reshape(K * npart, 6)     # anchor-major
    signf = np.tile(sign, K)
    rt = tidal_radius(field, ws, m_prog_msun)
    xyz = ws[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    rhat = xyz / r[:, None]
    L = np.cross(xyz, ws[:, 3:])
    Lnorm = np.linalg.norm(L, axis=1)
    Lhat = L / np.maximum(Lnorm, 1e-30)[:, None]
    phihat = np.cross(Lhat, rhat)
    Om = Lnorm / r ** 2
    f = FARDAL
    n = K * npart
    kr = f["kr"] + f["s_kr"] * rng.standard_normal(n)
    kvphi = f["kvphi"] + f["s_kvphi"] * rng.standard_normal(n)
    kz = f["s_kz"] * rng.standard_normal(n)
    kvz = f["s_kvz"] * rng.standard_normal(n)
    rel = np.empty((n, 6))
    rel[:, :3] = xyz + (signf * kr * rt)[:, None] * rhat \
        + (kz * rt)[:, None] * Lhat
    rel[:, 3:] = ws[:, 3:] + (signf * kvphi * rt * Om)[:, None] * phihat \
        + (kvz * rt * Om)[:, None] * Lhat

    rs_full = np.tile(rel_step, K)
    release = {}
    for j in np.unique(rs_full):
        idx = np.where(rs_full == j)[0]
        release[int(j)] = (idx, rel[idx])
    w = np.repeat(rel[:1], n, axis=0)                    # placeholder
    out = leapfrog(field, w, dt, nstep, release=release)
    return out.reshape(K, npart, 6)


# ===========================================================================
#  7.  OBSERVABLES
# ===========================================================================
def galactocentric_to_observables(w):
    """(N,6) SI galactocentric cartesian -> (ra, dec, D_kpc, pmra*, pmdec,
    vrad).  Right-handed frame with the Sun at x = -R0, Galactic rotation in
    +y at the Sun.  Uses astropy for the ICRS rotation."""
    from astropy.coordinates import (Galactocentric, ICRS, CartesianDifferential,
                                     CartesianRepresentation)
    import astropy.units as u
    rep = CartesianRepresentation(w[:, 0] / KPC * u.kpc,
                                  w[:, 1] / KPC * u.kpc,
                                  w[:, 2] / KPC * u.kpc)
    dif = CartesianDifferential(w[:, 3] / KMS * u.km / u.s,
                                w[:, 4] / KMS * u.km / u.s,
                                w[:, 5] / KMS * u.km / u.s)
    gc = Galactocentric(rep.with_differentials(dif),
                        galcen_distance=R0_SUN_KPC * u.kpc,
                        z_sun=Z0_SUN_KPC * u.kpc,
                        galcen_v_sun=CartesianDifferential(
                            *[v * u.km / u.s for v in VSUN_KMS]))
    ic = gc.transform_to(ICRS())
    return np.stack([ic.ra.deg, ic.dec.deg, ic.distance.kpc,
                     ic.pm_ra_cosdec.value, ic.pm_dec.value,
                     ic.radial_velocity.value], axis=1)


_FRAME = {}


def _frame():
    """Rotation + offset of the galactocentric -> ICRS-cartesian map, derived
    ONCE from astropy so the per-call astropy overhead is removed.  Validated
    against astropy to 1e-9 deg / 1e-9 mas/yr in `validate_frame`."""
    if _FRAME:
        return _FRAME
    e = np.zeros((4, 6))
    e[1:, :3] = np.eye(3) * KPC
    o = galactocentric_to_observables(e)
    d = np.stack([o[:, 2] * np.cos(np.radians(o[:, 1]))
                  * np.cos(np.radians(o[:, 0])),
                  o[:, 2] * np.cos(np.radians(o[:, 1]))
                  * np.sin(np.radians(o[:, 0])),
                  o[:, 2] * np.sin(np.radians(o[:, 1]))], axis=1)
    M = (d[1:] - d[0]).T                       # columns = images of e_x,e_y,e_z
    off = d[0]
    ev = np.zeros((4, 6))
    ev[1:, 3:] = np.eye(3) * KMS
    ev[:, 0] = 1e4 * KPC                       # far away: pure radial direction
    _FRAME.update(M=M, off=off, Vsun=np.array(VSUN_KMS))
    return _FRAME


_K_PM = 4.740470446                            # (km/s)/kpc per (mas/yr)


def fast_observables(w):
    """(N,6) SI galactocentric -> (ra, dec, D_kpc, pmra*, pmdec, vrad).

    Pure numpy; identical to `galactocentric_to_observables` to 1e-9 in every
    column (checked by `validate_frame`), but ~500x faster, which is what
    makes forward-modelling every stream at every Lambda affordable."""
    f = _frame()
    d = (f["M"] @ (w[:, :3] / KPC).T).T + f["off"]
    vv = (f["M"] @ ((w[:, 3:] / KMS) - f["Vsun"]).T).T
    D = np.linalg.norm(d, axis=1)
    dh = d / D[:, None]
    ra = np.degrees(np.arctan2(dh[:, 1], dh[:, 0])) % 360.0
    dec = np.degrees(np.arcsin(np.clip(dh[:, 2], -1, 1)))
    sa, ca = np.sin(np.radians(ra)), np.cos(np.radians(ra))
    sd, cd = np.sin(np.radians(dec)), np.cos(np.radians(dec))
    e_ra = np.stack([-sa, ca, np.zeros_like(sa)], axis=1)
    e_de = np.stack([-sd * ca, -sd * sa, cd], axis=1)
    vr = np.sum(vv * dh, axis=1)
    pmra = np.sum(vv * e_ra, axis=1) / (D * _K_PM)
    pmde = np.sum(vv * e_de, axis=1) / (D * _K_PM)
    return np.stack([ra, dec, D, pmra, pmde, vr], axis=1)


def validate_frame(n=200, seed=7):
    rng = np.random.default_rng(seed)
    w = np.zeros((n, 6))
    w[:, :3] = rng.normal(0, 40, (n, 3)) * KPC
    w[:, 3:] = rng.normal(0, 180, (n, 3)) * KMS
    a = galactocentric_to_observables(w)
    b = fast_observables(w)
    da = np.abs(((a[:, 0] - b[:, 0] + 180) % 360) - 180)
    return dict(max_dra_deg=float(da.max()),
                max_ddec_deg=float(np.abs(a[:, 1] - b[:, 1]).max()),
                max_dD_kpc=float(np.abs(a[:, 2] - b[:, 2]).max()),
                max_dpm_masyr=float(np.abs(a[:, 3:5] - b[:, 3:5]).max()),
                max_dvr_kms=float(np.abs(a[:, 5] - b[:, 5]).max()))


def observables_to_galactocentric(obs):
    """Inverse of the above.  obs: (N,6)."""
    from astropy.coordinates import (Galactocentric, ICRS, CartesianDifferential)
    import astropy.units as u
    ic = ICRS(ra=obs[:, 0] * u.deg, dec=obs[:, 1] * u.deg,
              distance=obs[:, 2] * u.kpc,
              pm_ra_cosdec=obs[:, 3] * u.mas / u.yr,
              pm_dec=obs[:, 4] * u.mas / u.yr,
              radial_velocity=obs[:, 5] * u.km / u.s)
    gc = ic.transform_to(Galactocentric(
        galcen_distance=R0_SUN_KPC * u.kpc, z_sun=Z0_SUN_KPC * u.kpc,
        galcen_v_sun=CartesianDifferential(*[v * u.km / u.s for v in VSUN_KMS])))
    w = np.empty((len(obs), 6))
    w[:, 0] = gc.x.to_value(u.kpc) * KPC
    w[:, 1] = gc.y.to_value(u.kpc) * KPC
    w[:, 2] = gc.z.to_value(u.kpc) * KPC
    w[:, 3] = gc.v_x.to_value(u.km / u.s) * KMS
    w[:, 4] = gc.v_y.to_value(u.km / u.s) * KMS
    w[:, 5] = gc.v_z.to_value(u.km / u.s) * KMS
    return w


# ===========================================================================
#  8.  PHYSICAL-PLAUSIBILITY GATE  --  applied to EVERY kinematic ingest
# ===========================================================================
#: The four silent galstreams defects (68 unit-distance tracks incl. GD-1,
#: 15 InfoFlags=1111 tracks with velocities up to 9.56e6 km/s = 32c, the
#: Pal5 999.0 sentinel, 16 flag-clear-but-populated columns) all fall out of
#: these bounds.  Float round-trip noise means the 1 kpc placeholder must be
#: caught with a TOLERANCE, never with `== 1.0`.
GATE = dict(D_min=0.5, D_max=300.0, pm_max=100.0, vrad_max=800.0,
            vgc_max=900.0, unit_D_tol=1e-6, sentinel=(999.0, -999.0, 0.0))


def plausibility_gate(obs, want=("D", "pm", "vrad")):
    """obs: (N,6).  Returns (ok_dict, reasons).  The FLAG GOVERNS elsewhere;
    this gate may only DOWNGRADE a track, never promote it."""
    reasons = []
    ok = {"D": True, "pm": True, "vrad": True}
    D = obs[:, 2]
    if np.all(np.abs(D - 1.0) < GATE["unit_D_tol"]):
        ok["D"] = False
        reasons.append("distance column is the 1 kpc placeholder "
                       f"(max|D-1| = {np.max(np.abs(D - 1.0)):.3e})")
    if np.nanmax(D) - np.nanmin(D) < 1e-9 and ok["D"]:
        ok["D"] = False
        reasons.append("distance column is constant")
    if not (np.all(D > GATE["D_min"]) and np.all(D < GATE["D_max"])):
        ok["D"] = False
        reasons.append(f"distance out of [{GATE['D_min']}, {GATE['D_max']}] kpc")
    pm = np.hypot(obs[:, 3], obs[:, 4])
    if not np.all(np.isfinite(pm)) or np.max(pm) > GATE["pm_max"]:
        ok["pm"] = False
        reasons.append(f"|pm| max {np.nanmax(pm):.3g} mas/yr exceeds gate")
    if np.ptp(obs[:, 3]) < 1e-12 and np.ptp(obs[:, 4]) < 1e-12:
        ok["pm"] = False
        reasons.append("proper-motion columns are constant")
    vr = obs[:, 5]
    if not np.all(np.isfinite(vr)) or np.max(np.abs(vr)) > GATE["vrad_max"]:
        ok["vrad"] = False
        reasons.append(f"|vrad| max {np.nanmax(np.abs(vr)):.6g} km/s "
                       f"exceeds {GATE['vrad_max']}")
    for s in GATE["sentinel"]:
        if np.all(np.abs(vr - s) < 1e-9):
            ok["vrad"] = False
            reasons.append(f"vrad is the {s} sentinel")
    if ok["D"] and ok["pm"] and ok["vrad"]:
        w = observables_to_galactocentric(obs)
        v = np.linalg.norm(w[:, 3:], axis=1) / KMS
        if np.max(v) > GATE["vgc_max"]:
            ok["vrad"] = ok["pm"] = False
            reasons.append(f"galactocentric speed max {np.max(v):.4g} km/s "
                           f"exceeds {GATE['vgc_max']}")
    return ok, reasons


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()
