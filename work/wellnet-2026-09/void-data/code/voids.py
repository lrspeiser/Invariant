"""
Void catalogue loading and exact line-of-sight ray tracing.

Two geometry families are handled exactly (no gridding, no proxy):

  VoidFinder  -- a void is a UNION OF SPHERES ("holes").  A ray-sphere
                 intersection is analytic; the per-void occupancy along a ray
                 is the union of the resulting intervals.

  V2 (ZOBOV / VIDE / REVOLVER) -- a void is a closed watershed cell whose
                 boundary is released as a triangle soup.  We intersect the ray
                 with every triangle of the candidate voids and use crossing
                 parity to build the inside intervals.  Watertightness is
                 asserted (even crossing count) rather than assumed.

All coordinates are the fiducial comoving Cartesian frame of common.py, which
is byte-for-byte the frame DESIVAST used.
"""
from __future__ import annotations

import os

import numpy as np
from astropy.io import fits

from common import R_MAX_VOID


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_voidfinder(path):
    """Return (maximals dict, holes dict) for a DESIVAST VoidFinder file."""
    with fits.open(path) as h:
        hdr = dict(h[0].header)
        m = h["MAXIMALS"].data
        ho = h["HOLES"].data
        maximals = {k: np.asarray(m[k]).astype(float if m[k].dtype.kind == "f" else np.int64)
                    for k in m.columns.names}
        holes = {k: np.asarray(ho[k]).astype(float if ho[k].dtype.kind == "f" else np.int64)
                 for k in ho.columns.names}
    n_void = int(hdr.get("VOID", len(maximals["VOID"])))
    assert len(maximals["VOID"]) == n_void, (
        f"{os.path.basename(path)}: MAXIMALS rows {len(maximals['VOID'])} "
        f"!= header VOID {n_void}")
    return hdr, maximals, holes


def load_v2(path):
    """Return (header, voids dict, triangles dict or None) for a V2 file."""
    with fits.open(path) as h:
        hdr = dict(h[0].header)
        v = h["VOIDS"].data
        voids = {k: np.asarray(v[k]).astype(float if v[k].dtype.kind == "f" else np.int64)
                 for k in v.columns.names}
        tri = None
        if "TRIANGLE" in [x.name for x in h]:
            t = h["TRIANGLE"].data
            tri = {k: np.asarray(t[k]).astype(float if t[k].dtype.kind == "f" else np.int64)
                   for k in t.columns.names}
    return hdr, voids, tri


def load_vast_sdss_holes(holes_txt, maximal_txt):
    """SDSS DR7 VAST VoidFinder (Zenodo v1.3.1, Planck2018 comoving)."""
    ho = np.loadtxt(holes_txt, skiprows=1)
    mx = np.loadtxt(maximal_txt, skiprows=1)
    holes = {"X": ho[:, 0], "Y": ho[:, 1], "Z": ho[:, 2],
             "RADIUS": ho[:, 3], "VOID": ho[:, 4].astype(np.int64)}
    maximals = {"X": mx[:, 0], "Y": mx[:, 1], "Z": mx[:, 2], "RADIUS": mx[:, 3],
                "VOID": mx[:, 4].astype(np.int64), "EDGE": mx[:, 5].astype(np.int64),
                "R": mx[:, 6], "RA": mx[:, 7], "DEC": mx[:, 8], "R_EFF": mx[:, 9]}
    return maximals, holes


# --------------------------------------------------------------------------
# interval algebra
# --------------------------------------------------------------------------
def union_length(intervals, t_lo, t_hi):
    """Total length of the union of [a,b] intervals clipped to [t_lo, t_hi]."""
    if len(intervals) == 0:
        return 0.0, []
    iv = np.asarray(intervals, float)
    iv[:, 0] = np.clip(iv[:, 0], t_lo, t_hi)
    iv[:, 1] = np.clip(iv[:, 1], t_lo, t_hi)
    iv = iv[iv[:, 1] > iv[:, 0]]
    if len(iv) == 0:
        return 0.0, []
    iv = iv[np.argsort(iv[:, 0])]
    merged = [iv[0].tolist()]
    for a, b in iv[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    total = sum(b - a for a, b in merged)
    return total, merged


# --------------------------------------------------------------------------
# VoidFinder: union of spheres
# --------------------------------------------------------------------------
class SphereUnionVoids:
    """Union-of-spheres void geometry with exact ray intersection."""

    def __init__(self, holes, maximals=None, name="VoidFinder"):
        self.c = np.stack([holes["X"], holes["Y"], holes["Z"]], axis=1)
        self.R = np.asarray(holes["RADIUS"], float)
        self.void_id = np.asarray(holes["VOID"], np.int64)
        self.maximals = maximals
        self.name = name
        self.c2 = np.einsum("ij,ij->i", self.c, self.c)

    def n_voids(self):
        return len(np.unique(self.void_id))

    def ray_intervals(self, u, t_max, void_subset=None):
        """
        Intervals of the ray  x(t) = t * u,  t in [0, t_max],  that lie inside
        the union of spheres.  `u` must be a unit vector.
        """
        sel = slice(None) if void_subset is None else void_subset
        c = self.c[sel]
        R = self.R[sel]
        c2 = self.c2[sel]
        b = c @ u                                # projection of centre on ray
        disc = R * R - (c2 - b * b)              # R^2 - d_perp^2
        hit = disc > 0
        if not np.any(hit):
            return 0.0, []
        s = np.sqrt(disc[hit])
        bh = b[hit]
        t1 = bh - s
        t2 = bh + s
        iv = np.stack([t1, t2], axis=1)
        iv = iv[iv[:, 1] > 0.0]
        return union_length(iv, 0.0, t_max)

    def ray_intervals_many(self, U, t_max):
        """Vectorised over rays: U is (N,3) unit vectors, t_max is (N,)."""
        out_len = np.zeros(len(U))
        out_iv = []
        for i in range(len(U)):
            L, iv = self.ray_intervals(U[i], t_max[i])
            out_len[i] = L
            out_iv.append(iv)
        return out_len, out_iv


# --------------------------------------------------------------------------
# V2: triangulated watershed boundaries
# --------------------------------------------------------------------------
class TriangleVoids:
    """
    Watershed voids bounded by triangle soup, with exact ray crossing parity.

    Candidate voids for a ray are prefiltered with per-void bounding spheres.
    """

    def __init__(self, voids, tri, name="V2"):
        self.name = name
        self.vid = np.asarray(voids["VOID"], np.int64)
        self.vcen = np.stack([voids["X"], voids["Y"], voids["Z"]], axis=1)
        self.vrad = np.asarray(voids["RADIUS"], float)
        tvid = np.asarray(tri["VOID"], np.int64)
        p1 = np.stack([tri["P1_X"], tri["P1_Y"], tri["P1_Z"]], axis=1)
        p2 = np.stack([tri["P2_X"], tri["P2_Y"], tri["P2_Z"]], axis=1)
        p3 = np.stack([tri["P3_X"], tri["P3_Y"], tri["P3_Z"]], axis=1)
        # group triangles by void id (contiguous sort)
        order = np.argsort(tvid, kind="stable")
        self.tvid = tvid[order]
        self.p1 = p1[order]
        self.p2 = p2[order]
        self.p3 = p3[order]
        uniq, start = np.unique(self.tvid, return_index=True)
        self.tri_void_ids = uniq
        self.tri_start = start
        self.tri_stop = np.append(start[1:], len(self.tvid))
        self._slice = {int(v): (int(a), int(b))
                       for v, a, b in zip(uniq, self.tri_start, self.tri_stop)}
        # bounding sphere per void from its own triangle vertices
        self.bs_c = np.zeros((len(uniq), 3))
        self.bs_r = np.zeros(len(uniq))
        for k, v in enumerate(uniq):
            a, b = self._slice[int(v)]
            pts = np.concatenate([self.p1[a:b], self.p2[a:b], self.p3[a:b]], axis=0)
            lo = pts.min(0)
            hi = pts.max(0)
            cen = 0.5 * (lo + hi)
            self.bs_c[k] = cen
            self.bs_r[k] = np.linalg.norm(pts - cen, axis=1).max()
        self.bs_c2 = np.einsum("ij,ij->i", self.bs_c, self.bs_c)

    def n_voids(self):
        return len(self.tri_void_ids)

    def _candidates(self, u):
        b = self.bs_c @ u
        d2 = self.bs_c2 - b * b
        return np.where((d2 < self.bs_r ** 2) & (b + self.bs_r > 0))[0]

    def ray_intervals(self, u, t_max, eps=1e-9):
        """Inside-intervals via Moller-Trumbore crossings and parity."""
        cand = self._candidates(u)
        iv_all = []
        n_odd = 0
        for k in cand:
            v = int(self.tri_void_ids[k])
            a, b = self._slice[v]
            v0 = self.p1[a:b]
            e1 = self.p2[a:b] - v0
            e2 = self.p3[a:b] - v0
            pv = np.cross(u, e2)
            det = np.einsum("ij,ij->i", e1, pv)
            ok = np.abs(det) > eps
            if not np.any(ok):
                continue
            inv = np.zeros_like(det)
            inv[ok] = 1.0 / det[ok]
            tv = -v0                       # ray origin is 0
            uu = np.einsum("ij,ij->i", tv, pv) * inv
            qv = np.cross(tv, e1)
            vv = (qv @ u) * inv
            tt = np.einsum("ij,ij->i", e2, qv) * inv
            good = ok & (uu >= -1e-12) & (vv >= -1e-12) & (uu + vv <= 1 + 1e-12) & (tt > 0)
            ts = np.sort(tt[good])
            if len(ts) == 0:
                continue
            if len(ts) % 2 == 1:
                n_odd += 1
                continue                    # not watertight along this ray: skip
            iv_all.extend([[ts[i], ts[i + 1]] for i in range(0, len(ts), 2)])
        L, merged = union_length(iv_all, 0.0, t_max)
        return L, merged, n_odd


# --------------------------------------------------------------------------
def combine_caps_spheres(paths, name):
    """Merge NGC + SGC VoidFinder hole tables into one SphereUnionVoids."""
    xs, ys, zs, rs, vs = [], [], [], [], []
    offset = 0
    n_max = 0
    for p in paths:
        hdr, mx, ho = load_voidfinder(p)
        xs.append(ho["X"]); ys.append(ho["Y"]); zs.append(ho["Z"])
        rs.append(ho["RADIUS"]); vs.append(ho["VOID"] + offset)
        offset += int(ho["VOID"].max()) + 1
        n_max += len(mx["VOID"])
    holes = {"X": np.concatenate(xs), "Y": np.concatenate(ys),
             "Z": np.concatenate(zs), "RADIUS": np.concatenate(rs),
             "VOID": np.concatenate(vs)}
    return SphereUnionVoids(holes, name=name), n_max
