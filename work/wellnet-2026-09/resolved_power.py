"""Can the resolved-versus-scrambled test detect a well-network effect at all?

This is the go/no-go for the headline experiment, and it can be answered now,
before the cluster data arrives, because it depends only on geometry, effect size
and noise.

THE STATISTIC

For each candidate law solve the field on four source models built from the same
cluster:

    rho_true      actual member positions and masses
    rho_smooth    same radial mass profile, angularly averaged
    rho_angscram  same masses and clustercentric radii, angles randomised
    rho_masscram  same positions, member masses permuted

and form

    Delta_resolved = lnL(D | rho_true, M) - lnL(D | rho_smooth, M)
    p_geometry     = [1 + #{Delta_scramble >= Delta_true}] / [1 + N_scramble]

A genuine well-network theory must give Delta_resolved > 0 on unseen clusters AND
the actual configuration must beat almost all mass-preserving scrambles. If true
positions do no better than scrambled ones, the model is fitting the radial
baryonic profile and nothing about the connections between wells.

WHY THE ANGULAR SCRAMBLE IS THE RIGHT NULL

It holds the radial mass profile EXACTLY fixed while destroying the geometry of
the well network. That is the only null that isolates the proposed mechanism:
anything a spherically averaged fit could have produced is identical in both arms
by construction. It is also the null demanded by the spherical blindness theorem
(see spherical_blindness.py) -- since the transverse eigenvalue is invisible in
spherical symmetry, a test that does not break sphericity cannot see the
mechanism at all, and an angular scramble is exactly the operation that varies
the symmetry breaking while holding everything else.

WHAT THIS MODULE ANSWERS

Given N members, an effect size, and a lensing-map signal-to-noise, what is the
POWER of p_geometry < 0.05? If the answer is low at plausible effect sizes, the
resolved tournament should not be run in that configuration, and the honest thing
is to say so before spending the compute rather than after.

Noise is parameterised by the signal-to-noise of the deflection map rather than
by a specific survey, because that is the quantity that actually sets the power
and it can be mapped onto any survey afterwards.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_grammar import (GPU, KPC, MSUN, A0, Poisson, div, grad, invariants,
                           nu_rar, tidal_tensor, xp)                    # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
G = 6.674e-11


# --------------------------------------------------------------- the cluster
def cluster(n=64, L_kpc=3000.0, ngal=200, Mgas=1.0e14, Mstar=2.0e12,
            rs_kpc=400.0, seed=0, positions=None, masses=None):
    """Smooth gas plus discrete members. Returns rho and the member table."""
    L = L_kpc * KPC
    h = L / n
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    r = xp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    rs = rs_kpc * KPC
    gas = 1.0 / (1.0 + (r / rs) ** 2) ** 1.5
    gas *= Mgas * MSUN / float(gas.sum() * h ** 3)

    rng = np.random.default_rng(seed)
    if positions is None:
        # NFW-ish projected radial distribution, isotropic angles
        u = rng.random(ngal)
        rad = rs_kpc * KPC * (u / (1.0 - u * 0.92)) ** 0.5
        rad = np.clip(rad, 0.05 * rs, 3.0 * rs)
        ct = rng.uniform(-1, 1, ngal)
        ph = rng.uniform(0, 2 * math.pi, ngal)
        st = np.sqrt(1 - ct ** 2)
        positions = np.stack([rad * st * np.cos(ph), rad * st * np.sin(ph),
                              rad * ct], 1)
    if masses is None:
        masses = 10 ** rng.normal(0, 0.5, len(positions))
        masses = masses / masses.sum() * Mstar * MSUN

    stars = xp.zeros((n, n, n))
    sig = 1.5 * h
    for p, m in zip(positions, masses):
        d2 = ((X - p[0]) ** 2 + (Y - p[1]) ** 2 + (Z - p[2]) ** 2)
        b = xp.exp(-d2 / (2 * sig ** 2))
        s = float(b.sum()) * h ** 3
        if s > 0:
            stars += b * (m / s)
    return gas + stars, h, np.asarray(positions), np.asarray(masses), gas


def smooth_source(rho, h):
    """Angularly average rho onto shells: same radial profile, no structure."""
    n = rho.shape[0]
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    r = xp.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    nb = 96
    edges = xp.linspace(0, float(r.max()) * 1.0001, nb + 1)
    idx = xp.clip(((r / (edges[1] - edges[0]))).astype(xp.int32), 0, nb - 1)
    out = xp.zeros_like(rho)
    flat_i = idx.ravel()
    flat_r = rho.ravel()
    s = xp.bincount(flat_i, weights=flat_r, minlength=nb)
    c = xp.bincount(flat_i, minlength=nb)
    mean = s / xp.maximum(c, 1)
    out = mean[idx]
    # preserve total mass exactly
    out *= float(rho.sum()) / float(out.sum())
    return out


def scramble_angles(pos, rng):
    """Keep every clustercentric radius, randomise the direction."""
    rad = np.linalg.norm(pos, axis=1)
    ct = rng.uniform(-1, 1, len(rad))
    ph = rng.uniform(0, 2 * math.pi, len(rad))
    st = np.sqrt(1 - ct ** 2)
    return np.stack([rad * st * np.cos(ph), rad * st * np.sin(ph),
                     rad * ct], 1)


# --------------------------------------------------------- the well network
def wellnet_K(pos, mass, n, h, p=1.0, Lw_kpc=500.0, sT=1.0, eps=1e-12):
    """S^ij from the member network, then K = exp[sT S] (traceless part only).

    Returned as six symmetric components. The isotropic part is omitted
    deliberately: by Corollary 2 of the spherical blindness theorem a constant
    isotropic part is degenerate with a rescaling of G, so including it would
    make the test partly a test of G.
    """
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None, None] * xp.ones((1, n, n))
    Y = xp.ones((n, 1, n)) * ax[None, :, None]
    Z = xp.ones((n, n, 1)) * ax[None, None, :]
    Lw = Lw_kpc * KPC
    M0 = float(np.median(mass))
    acc = [xp.zeros((n, n, n)) for _ in range(6)]
    wsum = xp.zeros((n, n, n))
    for pnt, m in zip(pos, mass):
        dx, dy, dz = X - pnt[0], Y - pnt[1], Z - pnt[2]
        d = xp.sqrt(dx ** 2 + dy ** 2 + dz ** 2 + (0.5 * h) ** 2)
        w = ((m / M0) ** p) / (1.0 + (d / Lw) ** 2)
        nx, ny, nz = dx / d, dy / d, dz / d
        acc[0] += w * (nx * nx - 1.0 / 3.0)
        acc[1] += w * (ny * ny - 1.0 / 3.0)
        acc[2] += w * (nz * nz - 1.0 / 3.0)
        acc[3] += w * (nx * ny)
        acc[4] += w * (nx * nz)
        acc[5] += w * (ny * nz)
        wsum += xp.abs(w)
    S = tuple(a / (eps + wsum) for a in acc)
    # exp of a traceless symmetric field, small-argument safe
    from linearisation_error import sexpm
    return sexpm(tuple(sT * s for s in S))


def solve_qumond(rho, h, K=None):
    """One Poisson solve. K = None means the plain QUMOND baseline."""
    n = rho.shape[0]
    P = Poisson(n, h)
    Phi_N = P.solve(4.0 * math.pi * G * rho)
    gx, gy, gz = grad(Phi_N, h)
    gm = xp.sqrt(gx ** 2 + gy ** 2 + gz ** 2) + 1e-30
    nu = nu_rar(gm / A0)
    if K is None:
        vx, vy, vz = nu * gx, nu * gy, nu * gz
    else:
        k11, k22, k33, k12, k13, k23 = K
        vx = nu * (k11 * gx + k12 * gy + k13 * gz)
        vy = nu * (k12 * gx + k22 * gy + k23 * gz)
        vz = nu * (k13 * gx + k23 * gy + k33 * gz)
    return P.solve(div(vx, vy, vz, h))


def deflection_map(Psi, h, axis=2):
    S = Psi.sum(axis=axis) * h
    a1 = xp.gradient(S, h, axis=0)
    a2 = xp.gradient(S, h, axis=1)
    return xp.stack([a1, a2])


def annulus(n, h, rmin_kpc, rmax_kpc):
    ax = (xp.arange(n) - n / 2 + 0.5) * h
    X = ax[:, None] * xp.ones((1, n))
    Y = xp.ones((n, 1)) * ax[None, :]
    R = xp.sqrt(X ** 2 + Y ** 2)
    return (R > rmin_kpc * KPC) & (R < rmax_kpc * KPC)


def main():
    print("=" * 78)
    print("RESOLVED vs SCRAMBLED -- power of the headline experiment")
    print("=" * 78)
    n = 48
    NGAL = int(os.environ.get("NGAL", 200))
    NSCRAM = int(os.environ.get("NSCRAM", 40))
    t0 = time.time()

    ALPHA = 0.05
    # A permutation p-value cannot go below 1/(1+N). With too few scrambles the
    # threshold is UNREACHABLE and the power is zero by construction rather than
    # by physics -- which is exactly what a first run of this module reported.
    pmin = 1.0 / (1.0 + NSCRAM)
    if pmin > ALPHA:
        raise SystemExit(
            f"NSCRAM={NSCRAM} gives a minimum attainable p of {pmin:.4f}, "
            f"above alpha={ALPHA}. Need NSCRAM >= {int(np.ceil(1/ALPHA))-1}.")
    print(f"   alpha = {ALPHA}, {NSCRAM} scrambles, minimum attainable "
          f"p = {pmin:.4f}")

    rho, h, pos, mass, gas = cluster(n=n, ngal=NGAL)
    msk = annulus(n, h, 200.0, 1200.0)
    print(f"\n   {NGAL} members + smooth gas on a {n}^3 grid, "
          f"{float(msk.sum()):.0f} map pixels in the annulus")

    rho_sm = smooth_source(rho, h)
    print(f"   mass conservation smooth/true = "
          f"{float(rho_sm.sum())/float(rho.sum()):.9f}")

    rng = np.random.default_rng(1)
    results = {}

    a_sm = deflection_map(solve_qumond(rho_sm, h, None), h)
    ref = float(xp.sqrt((a_sm[:, msk] ** 2).mean()))

    for sT in (0.0, 0.1, 0.3, 0.6, 1.0):
        K_true = None if sT == 0 else wellnet_K(pos, mass, n, h, sT=sT)
        a_true = deflection_map(solve_qumond(rho, h, K_true), h)
        sig = float(xp.sqrt(((a_true[:, msk] - a_sm[:, msk]) ** 2).mean()))
        print(f"\n   sT = {sT:4.2f}   |a_true - a_smooth| / |a_smooth| = "
              f"{sig/ref:.4f}")

        # PREDICTED maps for the scramble ensemble. The statistic is a
        # LIKELIHOOD against the observed map, not an RMS of the difference: a
        # scrambled configuration puts its lumps in the WRONG PLACES, which a
        # likelihood penalises and an RMS-of-difference is blind to. The first
        # version of this module used the RMS and reported a discriminating
        # margin of only 3.4% of the signal, most of which was the statistic's
        # fault rather than the physics'.
        preds = []
        for k in range(NSCRAM):
            ps = scramble_angles(pos, rng)
            rs_, _, _, _, _ = cluster(n=n, ngal=NGAL, positions=ps, masses=mass)
            Ks = None if sT == 0 else wellnet_K(ps, mass, n, h, sT=sT)
            preds.append(deflection_map(solve_qumond(rs_, h, Ks), h)[:, msk])

        at = a_true[:, msk]
        asm = a_sm[:, msk]

        row, margins = {}, {}
        for snr in (3.0, 10.0, 30.0, 100.0):
            noise = ref / snr
            hits, trials, zs = 0, 200, []
            for t in range(trials):
                obs = at + xp.asarray(rng.normal(0, noise, at.shape))

                def chi2(pred):
                    d = obs - pred
                    return float((d * d).sum()) / (noise ** 2)

                c_sm = chi2(asm)
                d_true = c_sm - chi2(at)
                d_scr = np.array([c_sm - chi2(q) for q in preds])
                pg = (1 + int((d_scr >= d_true).sum())) / (1 + NSCRAM)
                hits += int(pg <= ALPHA)
                zs.append((d_true - d_scr.mean()) / max(d_scr.std(), 1e-30))
            row[snr] = hits / trials
            margins[snr] = float(np.median(zs))
            print(f"      map S/N {snr:6.1f}   power(p_geometry <= {ALPHA}) = "
                  f"{hits/trials:.2f}   median margin "
                  f"{np.median(zs):7.2f} scramble-sd")
        results[sT] = {"signal_over_ref": sig / ref, "power": row,
                       "median_margin_scramble_sd": margins}

    print("\n" + "=" * 78)
    print("   READING THIS TABLE")
    print("   Delta_resolved compares the TRUE configuration against the")
    print("   spherically averaged one, as a chi-squared on the deflection map.")
    print("   p_geometry asks how many mass- and radius-preserving angular")
    print("   scrambles do at least as well. Power is the fraction of trials in")
    print("   which the true configuration beats 95% of scrambles.")
    print("   The margin column is the honest effect size: how many scramble")
    print("   standard deviations separate the truth from a random arrangement")
    print("   of the same mass at the same radii.")
    print(f"\n   elapsed {time.time()-t0:.0f}s")

    with open(os.path.join(HERE, "resolved_power.json"), "w") as f:
        json.dump({"ngal": NGAL, "n_grid": n, "n_scramble": NSCRAM,
                   "alpha": ALPHA,
                   "statistic": "chi2 map likelihood vs spherical reference",
                   "results": {str(k): v for k, v in results.items()}}, f,
                  indent=1)
    print(f"   written: resolved_power.json")


if __name__ == "__main__":
    main()
