"""The synthetic A2029-like cluster, and the field-galaxy control.

The smooth component is the REAL X-COP baryonic mass profile of A2029 --
M_b(<r) reconstructed as g_b r^2 / G from the bench's baryonic acceleration
column, 38 points from 125 to 1644 kpc, M_b(<R500 = 1414 kpc) = 1.44e14 Msun.
That is an observable-derived baryon profile (X-ray gas plus stars); no
dark-matter-inferred mass enters.  Cached in a2029_baryons.npz with a manifest.

The 300 members are a STATISTICAL population, not A2029's catalogue: masses
resampled from the AXES luminosity function of the 140 brightest group members
measured earlier in this programme, positions drawn from the gas mass profile,
isotropic.  This is the same prescription the earlier QUMOND-lumpiness
calculation used, deliberately, so the two results are comparable.

SELF-INCLUSION.  A field point sitting 20 kpc from a member has that member as
its nearest well, and with any steeply falling w_a that one well dominates the
sum, driving |S| towards its maximum.  The formula in the brief has no
self-exclusion, so the default here is that every well counts everywhere; the
alternative (drop wells inside r_core) is implemented and its effect reported,
because it changes the answer to the galaxy-limit question and should not be
chosen silently.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from wellnet import G, A0, KPC, MSUN

HERE = Path(__file__).resolve().parent
R500_KPC = 1414.0


def load_profile():
    d = np.load(HERE / "a2029_baryons.npz")
    return d["r_m"], d["Mbar_kg"]


def gas_density_profile(r, M):
    """rho(r) from the enclosed-mass profile, plus the two extrapolations."""
    lnr, lnM = np.log(r), np.log(M)
    cpo = np.polyfit(lnr, lnM, 4)
    sl = np.clip(np.polyval(np.polyder(cpo), lnr), 0.02, 3.5)
    rho = np.exp(np.polyval(cpo, lnr)) * sl / (4 * math.pi * r ** 3)
    out = r > 0.35 * r[-1]
    p_out = float(np.clip(np.polyfit(np.log(r[out]), np.log(rho[out]), 1)[0],
                          -5.0, -2.0))
    return rho, p_out


def draw_members(r, M, ngal=300, fstar=0.15, seed=20260903, rng=None):
    """Member masses and positions."""
    rng = rng or np.random.default_rng(seed)
    lum = np.load(HERE / "lumfun.npz")["mL"]
    lu = rng.choice(lum, size=ngal, replace=True)
    Mg = (fstar * M[-1]) * (lu / lu.sum())
    u = rng.random(ngal)
    rg = np.interp(u * M[-1], M, r)
    ct = 2 * rng.random(ngal) - 1
    ph = 2 * math.pi * rng.random(ngal)
    st = np.sqrt(1 - ct ** 2)
    pos = np.stack([rg * st * np.cos(ph), rg * st * np.sin(ph), rg * ct], 1)
    return pos, Mg


def deposit_cic(pos, mass, n, Lbox):
    """Cloud-in-cell deposition onto an n^3 grid, returns density."""
    dx = Lbox / n
    rho = np.zeros((n, n, n))
    idx = (pos + Lbox / 2) / dx - 0.5
    i0 = np.floor(idx).astype(int)
    f = idx - i0
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                w = ((f[:, 0] if a else 1 - f[:, 0])
                     * (f[:, 1] if b else 1 - f[:, 1])
                     * (f[:, 2] if c else 1 - f[:, 2]))
                ii, jj, kk = i0[:, 0] + a, i0[:, 1] + b, i0[:, 2] + c
                ok = ((ii >= 0) & (ii < n) & (jj >= 0) & (jj < n)
                      & (kk >= 0) & (kk < n))
                np.add.at(rho, (ii[ok], jj[ok], kk[ok]),
                          w[ok] * mass[ok] / dx ** 3)
    return rho


def build(n=128, Lbox=6000.0 * KPC, ngal=300, fstar=0.15, seed=20260903,
          smooth_members=False):
    """Density grid, member catalogue and the spherical mass profile."""
    r, M = load_profile()
    rho_r, p_out = gas_density_profile(r, M)
    dx = Lbox / n
    ax = (np.arange(n) - n / 2 + 0.5) * dx
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)

    rho_gas = np.interp(R, r, rho_r)
    out = R > r[-1]
    # outward continuation: the fitted outer power law with a Gaussian taper,
    # because a bare rho ~ r^-2 continued to the box corner would ADD 1.2e14
    # Msun of invented gas -- more than the measured M_b(<R500).  The taper
    # brings M_tot to ~1.7e14 Msun, i.e. the measured profile plus a plausible
    # outskirt to ~R200.
    rho_gas[out] = (rho_r[-1] * (R[out] / r[-1]) ** p_out
                    * np.exp(-((R[out] - r[-1]) / (800.0 * KPC)) ** 2))
    rho_gas[R < r[0]] = rho_r[0]
    rho_gas *= (1 - fstar)
    inR = R <= r[-1]
    rho_gas *= ((1 - fstar) * M[-1]) / (rho_gas[inR].sum() * dx ** 3)

    pos, Mg = draw_members(r, M, ngal, fstar, seed)
    if smooth_members:
        # the identical member mass, spherically averaged -- the control the
        # earlier lumpiness calculation used
        rho_gal = np.zeros_like(rho_gas)
        nb = 220
        edges = np.linspace(0, R.max(), nb + 1)
        rg = np.sqrt((pos ** 2).sum(1))
        idx = np.clip(np.digitize(rg, edges) - 1, 0, nb - 1)
        shellM = np.bincount(idx, weights=Mg, minlength=nb)
        vol = 4 * math.pi / 3 * (edges[1:] ** 3 - edges[:-1] ** 3)
        prof = shellM / vol
        which = np.clip(np.digitize(R.ravel(), edges) - 1, 0, nb - 1)
        rho_gal = prof[which].reshape(R.shape)
    else:
        rho_gal = deposit_cic(pos, Mg, n, Lbox)

    rho = rho_gas + rho_gal
    # spherical enclosed-mass profile of the ACTUAL grid source, for the BC
    nb = 400
    edges = np.linspace(0, float(R.max()), nb + 1)
    which = np.clip(np.digitize(R.ravel(), edges) - 1, 0, nb - 1)
    shell = np.bincount(which, weights=rho.ravel(), minlength=nb) * dx ** 3
    Menc = np.cumsum(shell)
    rc = 0.5 * (edges[1:] + edges[:-1])
    # gas-only profile, resolution independent, for the analytic Newtonian
    # potential used by the environmental gates
    rg = np.linspace(0.5 * dx, float(R.max()), 1200)
    rho_g_r = np.interp(rg, r, rho_r)
    o2 = rg > r[-1]
    rho_g_r[o2] = (rho_r[-1] * (rg[o2] / r[-1]) ** p_out
                   * np.exp(-((rg[o2] - r[-1]) / (800.0 * KPC)) ** 2))
    rho_g_r[rg < r[0]] = rho_r[0]
    rho_g_r *= (1 - fstar)
    dM = 4 * math.pi * rg ** 2 * rho_g_r * np.gradient(rg)
    Mgas_r = np.cumsum(dM)
    Mgas_r *= ((1 - fstar) * M[-1]) / np.interp(r[-1], rg, Mgas_r)
    dM = np.gradient(Mgas_r)
    return dict(rho=rho, dx=dx, ax=ax, X=X, Y=Y, Z=Z, R=R, pos=pos, Mg=Mg,
                r_prof=rc, M_prof=Menc, Lbox=Lbox, n=n, r_dat=r, M_dat=M,
                r_gas=rg, M_gas=Mgas_r, dM_gas=dM,
                Mtot=float(rho.sum() * dx ** 3))


# ------------------------------------------------------ field-galaxy control
def field_galaxy(seed=17, n_neighbour=24, Rfield=5000.0 * KPC,
                 Mgal=5.0e10 * MSUN, Rd=3.0 * KPC):
    """A SPARC-like disc galaxy in a field environment.

    The host is one well of 5e10 Msun at the origin; the neighbours are a
    Schechter-ish draw within 5 Mpc at roughly the field number density of
    L > 0.1 L* galaxies (~0.02 Mpc^-3 -> ~10 within 5 Mpc; 24 is generous).
    """
    rng = np.random.default_rng(seed)
    m = 10.0 ** rng.uniform(9.0, 11.2, n_neighbour) * MSUN
    u = rng.random(n_neighbour) ** (1.0 / 3.0)
    rr = Rfield * np.maximum(u, 0.15)
    ct = 2 * rng.random(n_neighbour) - 1
    ph = 2 * math.pi * rng.random(n_neighbour)
    st = np.sqrt(1 - ct ** 2)
    pos = np.stack([rr * st * np.cos(ph), rr * st * np.sin(ph), rr * ct], 1)
    pos = np.concatenate([np.zeros((1, 3)), pos])
    mass = np.concatenate([[Mgal], m])
    return pos, mass, Mgal, Rd
