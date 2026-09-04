"""
Galaxy density field, peculiar potential, tidal tensor and |grad Phi| on a
comoving Cartesian grid, for one survey cap.

Physics
-------
Comoving peculiar potential at a = 1:

    lap(phi) = (3/2) Omega_m H0^2 delta            [ (km/s)^2 / (Mpc/h)^2 ]

with H0 = 100 km/s per Mpc/h, so phi is in (km/s)^2, grad(phi) in
(km/s)^2/(Mpc/h) and T_ij = d_i d_j phi in (km/s)^2/(Mpc/h)^2.

Selection
---------
delta is estimated against a random catalogue that carries the SAME angular
footprint (binary pixel mask from the galaxies themselves) and the SAME radial
number density n(r) measured from the galaxies.  Consequences, stated plainly:

  * the radial monopole of delta is zero BY CONSTRUCTION.  Any real radial
    density gradient is absorbed into the selection function.  All surviving
    structure in delta is transverse, which is exactly the structure that gives
    two sources at the same distance different lines of sight.
  * angular completeness is assumed uniform inside the footprint.
  * delta is set to 0 outside the mask.  The potential therefore only contains
    the contribution of *observed* fluctuations; mass outside the survey is
    treated as being at the mean density.  This matters much more for
    grad(phi) (large-scale sensitive) than for T_ij (small-scale sensitive).
"""
from __future__ import annotations

import numpy as np

from common import FootprintMask, OMEGA_M, R_MAX_VOID

H0_UNITS = 100.0  # km/s per Mpc/h
POISSON_A = 1.5 * OMEGA_M * H0_UNITS ** 2  # (km/s)^2 / (Mpc/h)^2 per unit delta

try:
    import cupy as cp
    _HAVE_GPU = True
except Exception:  # pragma: no cover
    cp = None
    _HAVE_GPU = False


def _smooth_dim(n):
    """Smallest 2,3,5-smooth integer >= n (good FFT sizes)."""
    best = None
    p = 1
    while p < 2 * n + 8:
        q = p
        while q < 2 * n + 8:
            r = q
            while r < 2 * n + 8:
                if r >= n and (best is None or r < best):
                    best = r
                r *= 5
            q *= 3
        p *= 2
    return int(best)


class DensityField:
    def __init__(self, gal_xyz, mask: FootprintMask, dx=4.0, smooth=5.0,
                 pad_factor=1.35, n_random_mult=20, seed=12345,
                 r_max=R_MAX_VOID, name="cap"):
        self.name = name
        self.dx = float(dx)
        self.smooth = float(smooth)
        self.mask = mask
        self.r_max = float(r_max)
        g = np.asarray(gal_xyz, float)
        self.n_gal = len(g)

        lo = g.min(0)
        hi = g.max(0)
        cen = 0.5 * (lo + hi)
        span = (hi - lo) * pad_factor + 6.0 * self.smooth
        self.shape = tuple(_smooth_dim(int(np.ceil(s / dx))) for s in span)
        self.origin = cen - 0.5 * np.array(self.shape) * dx

        rng = np.random.default_rng(seed)
        rnd = self._make_randoms(g, rng, n_random_mult)
        self.n_ran = len(rnd)

        ng = self._cic(g)
        nr = self._cic(rnd)
        alpha = self.n_gal / self.n_ran
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.where(nr > 0, ng / (alpha * np.maximum(nr, 1e-30)) - 1.0, 0.0)
        # cells with too few randoms are outside / edge -> mean density
        thresh = 0.15 * np.median(nr[nr > 0])
        self.cell_in_survey = nr > thresh
        delta[~self.cell_in_survey] = 0.0
        self.delta_raw = delta.astype(np.float32)
        self._solve()

    # ------------------------------------------------------------------
    def _make_randoms(self, g, rng, mult):
        r = np.linalg.norm(g, axis=1)
        edges = np.arange(0.0, self.r_max + 10.0, 10.0)
        cnt, _ = np.histogram(r, bins=edges)
        n_target = int(mult * len(g))
        # draw radii from the empirical n(r) r^2 dr distribution == the galaxy
        # radial histogram itself, then uniform angles inside the mask
        p = cnt.astype(float)
        p /= p.sum()
        which = rng.choice(len(cnt), size=n_target, p=p)
        rr = edges[which] + rng.random(n_target) * np.diff(edges)[which]
        out = []
        need = n_target
        while need > 0:
            m = int(need * 2.5) + 1000
            u = rng.random(m)
            v = rng.random(m)
            ra = 360.0 * u
            dec = np.degrees(np.arcsin(2.0 * v - 1.0))
            ok = self.mask.contains(ra, dec)
            ra = ra[ok]
            dec = dec[ok]
            out.append(np.stack([ra, dec], axis=1))
            need -= len(ra)
        ang = np.vstack(out)[:n_target]
        ra = np.radians(ang[:, 0])
        dec = np.radians(ang[:, 1])
        cd = np.cos(dec)
        return np.stack([rr * cd * np.cos(ra), rr * cd * np.sin(ra),
                         rr * np.sin(dec)], axis=1)

    def _cic(self, pts):
        nx, ny, nz = self.shape
        f = (pts - self.origin) / self.dx
        i0 = np.floor(f).astype(np.int64)
        d = f - i0
        flat = np.zeros(nx * ny * nz, np.float64)
        for a in (0, 1):
            wx = d[:, 0] if a else 1 - d[:, 0]
            ix = i0[:, 0] + a
            for b in (0, 1):
                wy = d[:, 1] if b else 1 - d[:, 1]
                iy = i0[:, 1] + b
                for c in (0, 1):
                    wz = d[:, 2] if c else 1 - d[:, 2]
                    iz = i0[:, 2] + c
                    ok = ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
                          & (iz >= 0) & (iz < nz))
                    lin = (ix[ok] * ny + iy[ok]) * nz + iz[ok]
                    flat += np.bincount(lin, weights=(wx * wy * wz)[ok],
                                        minlength=nx * ny * nz)
        return flat.reshape(self.shape)

    # ------------------------------------------------------------------
    def _solve(self):
        xp = cp if _HAVE_GPU else np
        nx, ny, nz = self.shape
        d = xp.asarray(self.delta_raw, dtype=xp.float32)
        kx = xp.asarray(2 * np.pi * np.fft.fftfreq(nx, self.dx), xp.float32)
        ky = xp.asarray(2 * np.pi * np.fft.fftfreq(ny, self.dx), xp.float32)
        kz = xp.asarray(2 * np.pi * np.fft.rfftfreq(nz, self.dx), xp.float32)
        dk = xp.fft.rfftn(d)
        del d
        k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2
              + kz[None, None, :] ** 2)
        # Gaussian smoothing
        w = xp.exp(-0.5 * k2 * (self.smooth ** 2))
        dk *= w
        self.delta = xp.asnumpy(xp.fft.irfftn(dk, s=self.shape)).astype(np.float32) \
            if _HAVE_GPU else np.fft.irfftn(dk, s=self.shape).astype(np.float32)
        # potential
        inv = xp.where(k2 > 0, 1.0 / xp.where(k2 > 0, k2, 1), 0.0).astype(xp.float32)
        phik = -POISSON_A * dk * inv
        del dk, inv
        # gradient
        self.grad = np.empty((3,) + self.shape, np.float32)
        ks = [kx[:, None, None], ky[None, :, None], kz[None, None, :]]
        for a in range(3):
            ga = xp.fft.irfftn(1j * ks[a] * phik, s=self.shape)
            self.grad[a] = xp.asnumpy(ga) if _HAVE_GPU else ga
            del ga
        # tidal tensor, 6 unique components in order xx,yy,zz,xy,xz,yz
        self.tidal = np.empty((6,) + self.shape, np.float32)
        pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        for m, (a, b) in enumerate(pairs):
            ta = xp.fft.irfftn(-ks[a] * ks[b] * phik, s=self.shape)
            self.tidal[m] = xp.asnumpy(ta) if _HAVE_GPU else ta
            del ta
        del phik, k2
        if _HAVE_GPU:
            cp.get_default_memory_pool().free_all_blocks()
        self.gradmag = np.sqrt((self.grad ** 2).sum(0)).astype(np.float32)

    # ------------------------------------------------------------------
    def sample(self, field, pts):
        """Trilinear interpolation of a scalar grid at Cartesian points."""
        nx, ny, nz = self.shape
        f = (np.asarray(pts, float) - self.origin) / self.dx
        i0 = np.floor(f).astype(np.int64)
        d = f - i0
        inside = ((i0[:, 0] >= 0) & (i0[:, 0] < nx - 1) & (i0[:, 1] >= 0)
                  & (i0[:, 1] < ny - 1) & (i0[:, 2] >= 0) & (i0[:, 2] < nz - 1))
        out = np.zeros(len(f))
        if not np.any(inside):
            return out, inside
        ii = i0[inside]
        dd = d[inside]
        acc = np.zeros(len(ii))
        for a in (0, 1):
            wx = dd[:, 0] if a else 1 - dd[:, 0]
            for b in (0, 1):
                wy = dd[:, 1] if b else 1 - dd[:, 1]
                for c in (0, 1):
                    wz = dd[:, 2] if c else 1 - dd[:, 2]
                    acc += wx * wy * wz * field[ii[:, 0] + a, ii[:, 1] + b,
                                                ii[:, 2] + c]
        out[inside] = acc
        return out, inside

    def tidal_ll(self, pts, u):
        """T_ij u^i u^j at points, for a single unit direction u."""
        pairs = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
        coef = []
        for (a, b) in pairs:
            c = u[a] * u[b]
            coef.append(c if a == b else 2.0 * c)
        tot = np.zeros(len(pts))
        ins = None
        for m in range(6):
            v, ins = self.sample(self.tidal[m], pts)
            tot += coef[m] * v
        return tot, ins

    def in_survey(self, pts):
        v, ins = self.sample(self.cell_in_survey.astype(np.float32), pts)
        return (v > 0.5) & ins
