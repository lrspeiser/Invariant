"""physics.py -- the ten benchmark alternate universes, as generative field laws.

Every universe supplies, for a scene:

    matter potential   Phi_m(x)        -> moves stars, gas, member galaxies
    light potential    Phi_l(x)        -> deflects photons, sets time delays
    redshift map       z_obs(z_cosmo)  -> spectra and light-curve durations

and nothing else.  The detector forward model (instrument.py) is IDENTICAL
across universes, so every difference in the emitted data comes from the law.

Base and deformations
---------------------
U3 (the MOND/AQUAL-like scalar universe) is the BASE.  U4-U9 are each a
one-knob deformation of U3 that returns exactly U3 when the knob is zero:

    U4  kappa = 0   ->  U3      environment coupling
    U5  A     = 0   ->  U3      external-axis tensor amplitude
    U6  B     = 0   ->  U3      well-network coupling
    U7  Mamp  = 0   ->  U3      memory amplitude
    U8  zeta  = 0   ->  U3      matter/light coupling difference
    U9  eps   = 0   ->  U3      path-redshift amplitude

That makes "at what amplitude does the effect become observable?" a
well-posed question with a single number per universe, and it makes the
U3-vs-Ux pair the natural power curve.  U1, U2 and U10 are structurally
different worlds, not deformations.

Approximations, declared
------------------------
* Spherical scenes (clusters): the tensor universe is solved EXACTLY to first
  order in A via the l=2 Green's function -- see ``tensor_quadrupole``.  It is
  a genuine potential, so dynamics and lensing are automatically consistent.
* Disk scenes (galaxies): the tensor response is applied as the leading-order
  local relation g_i = K_ij dPhi/dx_j.  This is exact for slowly varying K and
  captures the observable of interest (the radial/vertical enhancement ratio);
  the neglected term is O(A * |grad ln f| * L).  DECLARED, not hidden.
* AQUAL/QUMOND curl corrections for a thin disk are not solved; the algebraic
  relation g = nu(g_N/a0) g_N is used.  For a Freeman disk this differs from a
  full QUMOND solve by a few per cent, which is inside the scatter this suite
  injects and is common to every universe that shares the base, so it cancels
  in every pairwise comparison against the base.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .baryons import A0, C_KMS, G

PHI0_ENV = 1000.0 ** 2          # (km/s)^2, declared normalisation for depth
LH_MPC = C_KMS / 70.0           # Hubble length [Mpc]


# ======================================================================
# cosmology -- computed here, no data file
# ======================================================================
_OM, _OL, _H0 = 0.3, 0.7, 70.0


def _E(z):
    return np.sqrt(_OM * (1 + z) ** 3 + _OL)


_ZG = np.linspace(0.0, 4.0, 2001)
_DCG = np.concatenate(([0.0], np.cumsum(
    0.5 * (1.0 / _E(_ZG[1:]) + 1.0 / _E(_ZG[:-1])) * np.diff(_ZG)))) * (C_KMS / _H0)


def comoving_Mpc(z):
    return np.interp(z, _ZG, _DCG)


def D_A(z):
    return comoving_Mpc(z) / (1.0 + z)


def D_A12(z1, z2):
    """Flat-universe angular diameter distance between two redshifts [Mpc]."""
    return (comoving_Mpc(z2) - comoving_Mpc(z1)) / (1.0 + z2)


def sigma_crit(zl, zs):
    """Msun / kpc^2.  Sigma_cr = c^2 D_s / (4 pi G D_l D_ls)."""
    Dl, Ds = D_A(zl) * 1e3, D_A(zs) * 1e3                 # kpc
    Dls = D_A12(zl, zs) * 1e3
    Dls = np.where(Dls <= 0, np.nan, Dls)
    return C_KMS ** 2 * Ds / (4.0 * np.pi * G * Dl * Dls)


# ======================================================================
# the scalar response family (the null: "any sufficiently smooth scalar")
# ======================================================================
def nu_rar(x):
    x = np.maximum(np.asarray(x, float), 1e-14)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def nu_simple(x):
    x = np.maximum(np.asarray(x, float), 1e-14)
    return 0.5 + np.sqrt(0.25 + 1.0 / x)


def nu_standard(x):
    x = np.maximum(np.asarray(x, float), 1e-14)
    return np.sqrt(0.5 + 0.5 * np.sqrt(1.0 + 4.0 / x ** 2))


def nu_alpha(x, al):
    x = np.maximum(np.asarray(x, float), 1e-14)
    return (0.5 + 0.5 * np.sqrt(1.0 + 4.0 * x ** (-al))) ** (1.0 / al)


def nu_smooth_perturbed(x, coefs, base=nu_rar):
    """RAR times a smooth random multiplicative response in log10 x.

    This is the null the brief demands: a response OUTSIDE any fixed grammar,
    not an off-grid member of the search bank.  coefs are 4 random amplitudes
    on a fixed smooth basis, so the perturbation is C-infinity and has no
    special structure at any particular x.
    """
    lx = np.log10(np.maximum(np.asarray(x, float), 1e-14))
    u = np.clip(lx, -4.0, 4.0) / 4.0
    b = np.array([u, u * u - 1.0 / 3.0, u ** 3 - 0.6 * u,
                  np.cos(2.0 * np.pi * u)])
    return base(x) * np.exp(np.tensordot(np.asarray(coefs), b, axes=(0, 0)))


SCALAR_FAMILIES = ("rar", "simple", "standard", "alpha", "smooth",
                   "sigma_gated", "phi_gated")


def draw_scalar_null(rng):
    """Draw one member of the scalar-null family.

    Seven QUALITATIVELY different families, three of which are not functions
    of g_N/a0 at all (they gate on surface density, potential depth or an
    unbounded smooth perturbation).  Returns a spec consumed by ``response``.
    """
    fam = SCALAR_FAMILIES[rng.integers(len(SCALAR_FAMILIES))]
    spec = {"family": fam}
    if fam == "alpha":
        spec["alpha"] = float(rng.uniform(0.7, 3.0))
    elif fam == "smooth":
        spec["coefs"] = (rng.normal(scale=0.14, size=4)).tolist()
        spec["base"] = ("rar", "simple", "standard")[rng.integers(3)]
    elif fam == "sigma_gated":
        spec["sig_c"] = float(10 ** rng.uniform(0.6, 1.4))     # Msun/pc^2
        spec["p"] = float(rng.uniform(0.35, 0.65))
    elif fam == "phi_gated":
        spec["phi_c"] = float(10 ** rng.uniform(3.6, 4.4))     # (km/s)^2
        spec["p"] = float(rng.uniform(0.35, 0.65))
    return spec


def scalar_nu(spec, x, sigma=None, phi=None):
    """Evaluate a scalar response.  x = g_N/a0_eff; sigma, phi optional gates."""
    f = spec["family"]
    if f == "rar":
        return nu_rar(x)
    if f == "simple":
        return nu_simple(x)
    if f == "standard":
        return nu_standard(x)
    if f == "alpha":
        return nu_alpha(x, spec["alpha"])
    if f == "smooth":
        base = {"rar": nu_rar, "simple": nu_simple, "standard": nu_standard}[spec["base"]]
        return nu_smooth_perturbed(x, spec["coefs"], base)
    if f == "sigma_gated":
        s = np.maximum(sigma if sigma is not None else 1.0, 1e-6)
        return 1.0 + (s / spec["sig_c"]) ** (-spec["p"])
    if f == "phi_gated":
        p = np.maximum(phi if phi is not None else 1.0, 1e-6)
        return 1.0 + (p / spec["phi_c"]) ** (-spec["p"])
    raise ValueError(f)


# ======================================================================
# tensor: exact l=2 response for a spherical base
# ======================================================================
def tensor_quadrupole(rg, g0, f_of_r):
    """chi(r) for Phi = Phi0(r) + A * P2(cos theta_axis) * chi(r).

    From  div[(I + A f(r) Q) grad Phi] = 4 pi G rho,  Q = (3 n n^T - I)/2,
    n constant, to first order in A:

        lap Phi1 = -P2(u) S(r),      S = F' - F/r,   F(r) = f(r) * Phi0'(r)

    and the l=2 Green's function gives

        chi(r) = (1/5) [ r^-3 int_0^r S s^4 ds  +  r^2 int_r^inf S s^-1 ds ].

    Phi0' = +g0 (g0 is the inward acceleration magnitude, Phi0' = g0 > 0).
    Returns chi on rg.  Genuine potential -> dynamics and lensing consistent.
    """
    rg = np.asarray(rg, float)
    F = f_of_r * g0
    dF = np.gradient(F, rg)
    S = dF - F / rg
    w = np.diff(rg)
    i1 = np.concatenate(([0.0], np.cumsum(
        0.5 * (S[1:] * rg[1:] ** 4 + S[:-1] * rg[:-1] ** 4) * w)))
    t = S / rg
    i2f = np.concatenate(([0.0], np.cumsum(0.5 * (t[1:] + t[:-1]) * w)))
    i2 = i2f[-1] - i2f
    return 0.2 * (i1 / rg ** 3 + rg ** 2 * i2)


def P2(u):
    return 0.5 * (3.0 * u * u - 1.0)


# ======================================================================
# universe registry
# ======================================================================
UNIVERSES = {
    "U01_baryons_newton": "standard gravity, baryons only",
    "U02_cdm": "standard gravity with collisionless dark matter",
    "U03_mond_scalar": "MOND/AQUAL-like scalar universe",
    "U04_env_scalar": "scalar environment-dependent universe",
    "U05_tensor_axis": "tensor, direction-dependent vacuum universe",
    "U06_wellnet": "reciprocal nonlocal well-network universe",
    "U07_memory": "universe with gravitational memory",
    "U08_ep_slip": "photons and matter couple differently",
    "U09_path_redshift": "geometric path-redshift universe",
    "U10_systematics": "realistic astrophysical and observational systematics only",
}

FIDUCIAL = {
    "U04_env_scalar": {"kappa": 0.60, "s": 0.50},
    "U05_tensor_axis": {"A": 0.50},
    "U06_wellnet": {"B": 0.06, "q": 0.5, "lam": 150.0},
    "U07_memory": {"Mamp": 0.20, "tau": 3.0},
    "U08_ep_slip": {"zeta": 0.10},
    "U09_path_redshift": {"eps": 0.030},
}

KNOB = {"U04_env_scalar": "kappa", "U05_tensor_axis": "A",
        "U06_wellnet": "B", "U07_memory": "Mamp",
        "U08_ep_slip": "zeta", "U09_path_redshift": "eps"}


@dataclass
class UniverseSpec:
    uid: str
    params: dict = field(default_factory=dict)
    sys_scale: float = 1.0          # multiplier on every SYSTEMATIC amplitude
    noise_scale: float = 1.0        # multiplier on every STATISTICAL error
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.uid


def draw_universe(uid, rng, knob=None, sys_scale=None):
    """Draw one universe: its OWN free constants from a prior, not a point."""
    p = {}
    # every non-Newtonian world carries a global acceleration scale with a
    # modest prior width -- a universe is a family, not a single point
    p["a0"] = A0 * 10 ** rng.normal(0.0, 0.04)
    p["nu"] = {"family": "rar"}
    if uid == "U01_baryons_newton":
        p["newton"] = True
    elif uid == "U02_cdm":
        p["newton"] = True
        p["fbar"] = float(10 ** rng.normal(np.log10(0.155), 0.04))
        p["c_norm"] = float(rng.uniform(4.2, 5.6))
        p["shmr_scatter"] = 0.16
    elif uid == "U10_systematics":
        p["newton"] = True
    else:
        p["newton"] = False
    if uid == "U04_env_scalar":
        p.update(FIDUCIAL[uid]); p["kappa"] = float(rng.normal(p["kappa"], 0.06))
    if uid == "U05_tensor_axis":
        p.update(FIDUCIAL[uid]); p["A"] = float(rng.normal(p["A"], 0.05))
    if uid == "U06_wellnet":
        p.update(FIDUCIAL[uid]); p["B"] = float(rng.normal(p["B"], 0.007))
        p["lam"] = float(p["lam"] * 10 ** rng.normal(0, 0.05))
    if uid == "U07_memory":
        p.update(FIDUCIAL[uid]); p["Mamp"] = float(rng.normal(p["Mamp"], 0.025))
    if uid == "U08_ep_slip":
        p.update(FIDUCIAL[uid]); p["zeta"] = float(rng.normal(p["zeta"], 0.012))
    if uid == "U09_path_redshift":
        p.update(FIDUCIAL[uid]); p["eps"] = float(rng.normal(p["eps"], 0.004))
    if knob is not None and uid in KNOB:
        p[KNOB[uid]] = float(knob)
    ss = sys_scale if sys_scale is not None else (3.0 if uid == "U10_systematics" else 1.0)
    return UniverseSpec(uid=uid, params=p, sys_scale=float(ss))


def draw_scalar_null_universe(rng, sys_scale=1.0):
    """A universe that is a random member of the scalar-null family.

    This is the calibration null for every directional / network / path
    detector: NOT an off-grid member of a search bank, but an arbitrary smooth
    scalar response, including families that gate on surface density or
    potential depth rather than on g_N/a0.
    """
    p = {"a0": A0 * 10 ** rng.normal(0.0, 0.06), "newton": False,
         "nu": draw_scalar_null(rng)}
    return UniverseSpec(uid="H0_scalar_null", params=p, sys_scale=sys_scale,
                        label="scalar-null " + p["nu"]["family"])


# ======================================================================
# the law, applied to a spherical (cluster) scene
# ======================================================================
def cluster_field(u: UniverseSpec, clu, rg, dm=None):
    """Return the radial matter/light acceleration on the grid rg.

    dm: optional dict of collisionless-halo arrays for U02.
    Returns dict with g_m(r), g_l(r), and the quadrupole chi(r) (or None).
    """
    p = u.params
    Mb = clu.Mbar(rg)
    gN = G * Mb / rg ** 2
    out = {"gN": gN}

    if u.uid == "U02_cdm":
        gN_tot = gN + G * dm["Mdm"](rg) / rg ** 2
        out["g_m"] = gN_tot
        out["g_l"] = gN_tot
        out["chi"] = None
        return out

    if p.get("newton", False):
        out["g_m"] = gN
        out["g_l"] = gN
        out["chi"] = None
        return out

    a0 = p["a0"]
    if u.uid == "U04_env_scalar":
        dphi = clu.phi_depth(rg)
        a0 = a0 * (1.0 + p["kappa"] * (dphi / PHI0_ENV) ** p["s"])
    sig = Mb / (np.pi * rg ** 2) / 1e6          # Msun/pc^2, crude projected
    nu = scalar_nu(p["nu"], gN / a0, sigma=sig, phi=clu.phi_depth(rg))
    g = nu * gN

    if u.uid == "U07_memory":
        g = g * (1.0 + p["Mamp"] * np.exp(-clu.t_merge / p["tau"]))

    out["g_m"] = g
    if u.uid == "U08_ep_slip":
        out["g_l"] = (nu ** (1.0 + p["zeta"])) * gN
        if u.uid == "U07_memory":
            out["g_l"] = out["g_l"] * (1.0 + p["Mamp"] * np.exp(-clu.t_merge / p["tau"]))
    else:
        out["g_l"] = g

    if u.uid == "U05_tensor_axis":
        rt = 0.30 * clu.R500
        f = (rg / rt) ** 2 / (1.0 + (rg / rt) ** 2)
        out["chi"] = p["A"] * tensor_quadrupole(rg, g, f)
    else:
        out["chi"] = None
    return out


def cluster_potential_1d(rg, g):
    """Phi(r) - Phi(r_max) from the radial acceleration (gauge-safe)."""
    seg = 0.5 * (g[1:] + g[:-1]) * np.diff(rg)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    return cum - cum[-1]        # <= 0, zero at the outer boundary


# ======================================================================
# the law, applied to a disk (galaxy) scene
# ======================================================================
def galaxy_field(u: UniverseSpec, gal, R, dm=None):
    """Radial and vertical accelerations for a disk galaxy at radii R.

    Returns g_R (matter), g_R_light, g_z (matter), and the in-plane m=2
    modulation amplitude of g_R (zero unless the universe is directional).
    """
    p = u.params
    R = np.asarray(R, float)
    gN = gal.gN(R)
    Sig = gal.Sigma(R)
    gzN = 2.0 * np.pi * G * Sig                 # thin-disk vertical field

    if u.uid == "U02_cdm":
        gdm = G * dm["Mdm"](R) / R ** 2
        gR = gN + gdm
        gz = gzN + gdm * 0.0 + G * dm["Mdm"](R) / R ** 2 * (gal.hz / R)
        return {"gN": gN, "g_R": gR, "g_Rl": gR, "g_z": gz,
                "quad_amp": np.zeros_like(R), "quad_pa": 0.0, "nu": gR / gN}

    if p.get("newton", False):
        return {"gN": gN, "g_R": gN, "g_Rl": gN, "g_z": gzN,
                "quad_amp": np.zeros_like(R), "quad_pa": 0.0,
                "nu": np.ones_like(R)}

    a0 = p["a0"]
    if u.uid == "U04_env_scalar":
        a0 = a0 * (1.0 + p["kappa"] * (gal.phi_depth(R) / PHI0_ENV) ** p["s"])
    nu = scalar_nu(p["nu"], gN / a0, sigma=Sig / 1e6, phi=gal.phi_depth(R))
    gR = nu * gN
    gz = nu * gzN

    if u.uid == "U06_wellnet":
        # same law as the cluster path: an extra reciprocal pair potential
        # -B (S/S0)^(q/2) * T, with T the softened baryonic potential.
        from .scenes import LAM_NET, Q_NET, S0_NET
        lam = LAM_NET
        Sloc = G * gal.Mbar / (R ** 2 + lam ** 2) + gal.S_ext
        Tloc = -G * gal.Mbar * (Sloc / S0_NET) ** (0.5 * Q_NET) / np.sqrt(R ** 2 + lam ** 2)
        dT = np.gradient(Tloc, R)
        gR = gR + p["B"] * dT
        gz = gz * (1.0 + p["B"] * dT / np.maximum(gR, 1e-12))

    if u.uid == "U07_memory":
        m = 1.0 + p["Mamp"] * np.exp(-gal.t_merge / p["tau"])
        gR, gz = gR * m, gz * m

    gRl = gR
    if u.uid == "U08_ep_slip":
        gRl = (nu ** (1.0 + p["zeta"])) * gN

    quad = np.zeros_like(R)
    quad_pa = 0.0
    if u.uid == "U05_tensor_axis":
        # leading-order local tensor response, g_i = K_ij dPhi/dx_j
        rt = 3.0 * gal.Rd
        f = (R / rt) ** 2 / (1.0 + (R / rt) ** 2)
        quad = 1.5 * p["A"] * f          # amplitude of cos(2 dphi) in g_R
        quad_pa = gal.axis_ext_deg
        # vertical: the external axis lies in the sky plane, so its projection
        # onto the disk normal depends on inclination -> a real g_z/g_R change
        cz = np.cos(np.deg2rad(gal.incl_deg))
        gz = gz * (1.0 + p["A"] * f * P2(cz))

    return {"gN": gN, "g_R": gR, "g_Rl": gRl, "g_z": gz,
            "quad_amp": quad, "quad_pa": quad_pa, "nu": gR / gN}


# ======================================================================
# redshift branch
# ======================================================================
def observed_redshift(u: UniverseSpec, z_cos, void_frac, dist_Mpc=None):
    """1 + z_obs and the light-curve duration stretch factor.

    The geometric path-redshift universe accrues redshift along low-density
    path segments.  Because the mechanism is geometric (a path-length effect
    on the null geodesic congruence), it stretches TIME by exactly the same
    factor -- so it is NOT excluded by the supernova time-dilation constraint.
    A non-time-stretching (tired-light) variant is excluded a priori and is
    not simulated.
    """
    z_cos = np.asarray(z_cos, float)
    onepz = 1.0 + z_cos
    if u.uid != "U09_path_redshift":
        return onepz, onepz
    L = comoving_Mpc(z_cos) if dist_Mpc is None else np.asarray(dist_Mpc, float)
    extra = u.params["eps"] * np.asarray(void_frac, float) * L / LH_MPC
    return onepz * (1.0 + extra), onepz * (1.0 + extra)
