"""FOUR ENVIRONMENTAL VARIABLES, built on ONE sample and ONE set of folds.

The programme has tested potential depth four ways (Runs AD, AE, AH, AI) and
found nothing.  Potential depth is only one reading of the cross-class result,
and it is the worst-behaved candidate available: it is defined only up to a
boundary convention whose two admissible global rules differ by 0.87 dex
against an off/on margin of 0.9 dex (Run AH.6), and 98.6% of it at the
shear-measured radii is a function of (log g_bar, log r) (Run AI.6).

This module constructs, at every point of every system's radial grid, four
PHYSICALLY DISTINCT environmental quantities from the same baryonic mass model:

  V1  potential depth        DeltaPhi_b(r) = Phi_b(r_ref) - Phi_b(r),
                             an operational DIFFERENCE with a prespecified
                             reference rule; five rules, 'fixed10Mpc' primary.
  V2  vector external field  g_ext = sum_a G M_a d_a / |d_a|^3
                             -- opposing wells CANCEL.
  V3  directionless well     W_eps = sum_a G M_a / (d_a^2 + eps^2)
      strength               -- opposing wells DO NOT cancel.  Bare sum over
                             catalogue rows, so it must pass coarse-graining.
  V4  tidal tensor           T_ij = d_i d_j Phi_b, magnitude |T~| (traceless
                             Frobenius), shape, and the EIGENVECTORS kept
                             separately rather than folded into a scalar.

THE TWO STRUCTURAL FACTS THAT DECIDE HOW V2 AND V3 MUST BE BUILT

  * The vector sum over ALL mass, self included, is exactly g_bar (Newton's
    shell theorem).  So a V2 that includes the system's own mass is IDENTICALLY
    the acceleration already carried by f(g_bar, r), and the test is vacuous.
    V2 is therefore EXTERNAL-ONLY by theorem, not by choice.
  * The directionless sum has no shell theorem.  W over the system's own mass
    is NOT G M(<r)/r^2 and is a genuinely new function of radius.  V3 therefore
    includes the self term -- and that is exactly the term whose value depends
    on how finely the cataloguer resolved the mass, which is why the
    coarse-graining gate is mandatory here and nowhere else.

DECLARED IN ADVANCE, BEFORE ANY RESIDUAL WAS EXAMINED
  boundary rules   primary 'fixed10Mpc'; sensitivity 'fixed5Mpc', 'fixed3Mpc',
                   '2xR500', '10xrs'                      (Run AI's five)
  Phi_0            1e12 m^2/s^2      a0  1.2e-10 m/s^2     T_0  1e-33 s^-2
  smoothing        eps = 50 kpc primary; 20 and 200 kpc sensitivity
  well masses      M_a = M_gas(<R500) x (1 + f_star), f_star = 0 primary,
                   computed by this lane's own chain (median 1.0022 of
                   Bahar+2022's published M_gas,500, 0.0440 dex scatter)
  well geometry    3-D comoving separation from (RA, Dec, z), converted to
                   proper at the redshift of the TARGET system
  redshift error   no error column is published; sigma_z = 0.005 (1+z) is
                   ASSUMED for the null and the sensitivity, not measured
  truncation       8 Mpc for the lensing projection, for EVERY model including
                   beta = 0, so the comparison stays like for like
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LANE, "efeds-hsc"))
sys.path.insert(0, os.path.join(LANE, "lead01"))

import pipeline as P            # noqa: E402
import efeds_hsc as E           # noqa: E402
import decade_test as D         # noqa: E402
import lead01 as L              # noqa: E402

MPC, KPC, MSUN, G = P.MPC, P.KPC, P.MSUN, P.G
A0 = 1.2e-10
PHI0 = 1.0e12
T0 = 1.0e-33
RULES = ["fixed10Mpc", "fixed5Mpc", "fixed3Mpc", "2xR500", "10xrs"]
PRIMARY_RULE = "fixed10Mpc"
EPS_PRIMARY = 50.0 * KPC
EPS_SENS = [20.0 * KPC, 200.0 * KPC]
R_TRUNC_MPC = 8.0                      # = 0.8 x 10 Mpc, fixed for all models
SIGMA_Z_ASSUMED = 0.005                # (1+z); assumed, not published
N_DIR = 12                             # icosahedral direction average
FIT_RANGE_MPC = (0.2, 5.0)             # where the shear points live

RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# --------------------------------------------------------------------- ingest
def load_all(f_star=0.0):
    """Ingest with row and column counts ASSERTED, and the M_gas gate run."""
    h1, d1 = L.read_tsv(os.path.join(LANE, "lead01",
                                     "efeds_bahar2022_table1_density.tsv"))
    h2, d2 = L.read_tsv(os.path.join(LANE, "lead01",
                                     "efeds_bahar2022_table2.tsv"))
    assert len(d1) == 542, f"table1 rows {len(d1)} != 542"
    assert len(d2) == 542, f"table2 rows {len(d2)} != 542"
    assert len(h1) == 19, f"table1 cols {len(h1)} != 19"
    assert len(h2) == 40, f"table2 cols {len(h2)} != 40"
    for c in ("n0", "e_n0", "E_n0", "rs", "epsilon", "beta", "alpha"):
        assert c in h1, f"table1 missing {c}"
    for c in ("RAJ2000", "DEJ2000", "z", "R500", "Mgas500"):
        assert c in h2, f"table2 missing {c}"
    print(f"   ingest table1: 542 rows x 19 cols   table2: 542 rows x 40 cols"
          f"   identifier J/A+A/661/A7 echoed")

    recs, cuts = E.load_efeds()
    assert len(recs) == 542, len(recs)

    pth = os.path.join(LANE, "efeds-hsc", "decade_efeds_shear_profiles.tsv")
    with open(pth, encoding="utf-8") as f:
        raw = [ln for ln in f if not ln.startswith("#")]
    assert len(raw) == 5412 and len(raw[0].rstrip("\n").split("\t")) == 13, \
        (len(raw), len(raw[0].split("\t")))
    prof = D.load_profiles()
    npts = sum(len(v) for v in prof.values())
    assert len(prof) == 536 and npts == 4228, (len(prof), npts)
    print(f"   ingest DECADE profiles: {len(raw)-1} rows x 13 cols on disk -> "
          f"{len(prof)} systems, {npts} rows with finite g_t")

    obs = D.Obs(recs, prof)
    ntot = sum(len(b) for b in obs.rows)
    assert len(obs) == 496 and ntot == 3365, (len(obs), ntot)
    print(f"   matched: {len(obs)} systems, {ntot} (system, bin) shear points")

    systems = [P.System(r, f_star) for r in obs.sys]
    E.gate_mgas(systems, obs.sys)
    RES["ingest"] = dict(table1_rows=542, table1_cols=19, table2_rows=542,
                         table2_cols=40, decade_systems=len(prof),
                         decade_rows=npts, matched_systems=len(obs),
                         matched_points=ntot, cuts=cuts,
                         decade_rows_on_disk=len(raw) - 1, decade_cols=13,
                         gate_mgas500=E.RES.get("gate_mgas500"))
    return recs, obs, systems


# ----------------------------------------------------------- the well network
def _fib_dirs(n):
    i = np.arange(n) + 0.5
    ph = np.arccos(1.0 - 2.0 * i / n)
    th = math.pi * (1.0 + 5.0 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(ph),
                            np.sin(th) * np.sin(ph), np.cos(ph)])


class Wells:
    """Every catalogued concentration in the field, as a 3-D point set.

    Masses are M_gas(<R500) x (1+f_star) from THIS lane's chain, so a redraw of
    the published density errors moves the well masses too -- which is exactly
    the shared-quantity channel the null has to contain.
    """

    def __init__(self, recs, f_star=0.0, dz=None):
        z = np.array([r["z"] for r in recs])
        if dz is not None:
            z = np.maximum(z + dz, 1e-3)
        ra = np.radians(np.array([r["RA"] for r in recs]))
        de = np.radians(np.array([r["DE"] for r in recs]))
        dc = np.array([P.d_com(zz) for zz in z])
        self.z = z
        self.pos = np.column_stack([dc * np.cos(de) * np.cos(ra),
                                    dc * np.cos(de) * np.sin(ra),
                                    dc * np.sin(de)])
        self.M = np.array([_mgas500(r, f_star) for r in recs])
        self.id = [r["id"] for r in recs]
        self.index = {i: k for k, i in enumerate(self.id)}

    def neighbours(self, sid, z_target):
        """Vectors from the target's centre to every OTHER well, proper metres."""
        k = self.index[sid]
        d = np.delete(self.pos, k, axis=0) - self.pos[k]
        return d / (1.0 + z_target), np.delete(self.M, k)


def _mgas500(rec, f_star):
    s = P.System(rec, f_star)
    return float(np.interp(s.R500, s.r, s.M_b))


# ---------------------------------------------------- the four variables, V1-V4
def v1_potential_depth(s, rule=PRIMARY_RULE):
    d, r_ref = s.dphi(rule)
    return np.log10(np.maximum(d, 1e-6) / PHI0), r_ref


def v3_self(s, eps=EPS_PRIMARY, n_s=1200, r_max_mpc=R_TRUNC_MPC):
    """W_self(r) = pi G / r * int rho(s) s ln[((r+s)^2+e^2)/((r-s)^2+e^2)] ds.

    Exact angular integral for a spherical rho; the log singularity at s = r is
    integrable and is regularised by eps.  There is NO shell theorem here: this
    is not G M(<r)/r^2 and it does not become it in any limit.  The source is
    truncated at the same 8 Mpc used by the lensing projection.
    """
    r = s.r
    sg = np.geomspace(r[0], r_max_mpc * MPC, n_s)
    lr = np.log(r)
    rho = np.exp(np.interp(np.log(sg), lr,
                           np.log(np.maximum(s.rho_gas, 1e-300)))) \
        * (1.0 + s.f_star)
    num = (r[:, None] + sg[None, :]) ** 2 + eps ** 2
    den = (r[:, None] - sg[None, :]) ** 2 + eps ** 2
    ker = sg[None, :] * np.log(num / den)
    return (math.pi * G / r) * np.trapezoid(ker * rho[None, :], sg, axis=1)


def v4_self(s):
    """Traceless tidal invariants of the spherical baryonic field, ANALYTIC.

    With y = 4 pi G rho r / g = 3 rho / <rho>, the eigenvalues of the traceless
    tidal tensor are lam_r = (g/r)(2y/3 - 2) and lam_t = (g/r)(1 - y/3) twice,
    so lam_r / lam_t = -2 IDENTICALLY and

        |T~| = sqrt(6) (g/r) |1 - rho/<rho>| .

    Two consequences that are properties of sphericity, not of this sample:
    the tidal SHAPE carries exactly one bit (the sign of rho/<rho> - 1), and
    the principal eigenvector is radial, so the eigenvector information is
    degenerate until an external tide breaks it.
    """
    r, g = s.r, s.g_b
    rho = s.rho_gas * (1.0 + s.f_star)
    Mb = np.maximum(s.M_b, 1e-30)
    rho_bar = 3.0 * Mb / (4.0 * math.pi * r ** 3)
    # For any monotonically declining rho, rho(r) <= <rho>(r) identically, so
    # q lives in [0, 1].  The clip only bites at the innermost grid point,
    # where the cumulative M_b starts from exactly zero and <rho> is 0/0.
    q = np.clip(rho / np.maximum(rho_bar, 1e-40), 0.0, 1.0)
    gr = g / r
    lam_r = gr * (2.0 * q - 2.0)
    lam_t = gr * (1.0 - q)
    Tmag = math.sqrt(6.0) * gr * np.abs(1.0 - q)
    return Tmag, q, lam_r, lam_t


def external_fields(s, wells, eps=EPS_PRIMARY, n_rad=28, n_dir=N_DIR,
                    full_tensor=True):
    """|g_ext|, W_ext and |T~_ext|, direction-averaged on spheres of radius r.

    The average is over n_dir Fibonacci directions on each of n_rad log-spaced
    radii, then interpolated in log r onto the system's own grid.  These fields
    vary by less than a per cent across a system, so the coarse radial grid is
    not a limitation -- and that near-constancy is itself the result for V2.
    """
    d, M = wells.neighbours(s.id, s.z)
    rg = np.geomspace(s.r[0], s.r[-1], n_rad)
    dirs = _fib_dirs(n_dir)
    x = rg[:, None, None] * dirs[None, :, :]                # (nr, nd, 3)
    dd = d[None, None, :, :] - x[:, :, None, :]             # (nr, nd, na, 3)
    r2 = np.sum(dd ** 2, axis=-1)
    rr = np.sqrt(r2)
    GM = G * M[None, None, :]
    gvec = np.sum(GM[..., None] * dd / rr[..., None] ** 3, axis=2)
    gmag = np.sqrt(np.sum(gvec ** 2, axis=-1)).mean(axis=1)
    W = np.sum(GM / (r2 + eps ** 2), axis=2).mean(axis=1)
    if full_tensor:
        near = np.argsort(np.sum(d ** 2, axis=1))[:60]
        u = dd[:, :, near, :] / rr[:, :, near, None]
        c = GM[:, :, near] / rr[:, :, near] ** 3
        T = -(3.0 * u[..., :, None] * u[..., None, :]
              - np.eye(3)[None, None, None, :, :]) * c[..., None, None]
        T = T.sum(axis=2)                                    # (nr, nd, 3, 3)
        Tm = np.sqrt(np.sum(T ** 2, axis=(-2, -1))).mean(axis=1)
    else:
        Tm = np.zeros_like(W)
    lr, lg = np.log(s.r), np.log(rg)
    out = [np.exp(np.interp(lr, lg, np.log(np.maximum(v, 1e-300))))
           for v in (gmag, W, Tm)]
    return out[0], out[1], out[2]


def build_variables(systems, wells, eps=EPS_PRIMARY, rule=PRIMARY_RULE,
                    verbose=True):
    """Every variable on every system's radial grid.  Returns a dict of lists."""
    V = {k: [] for k in ("x1", "x2", "x3", "x4a", "x4b", "x4c", "x4d",
                         "lgb", "lr", "W_self", "W_ext", "T_self", "T_ext",
                         "gext")}
    for s in systems:
        x1, _ = v1_potential_depth(s, rule)
        gext, Wext, Text = external_fields(s, wells, eps)
        Wself = v3_self(s, eps)
        Tself, q, _, _ = v4_self(s)
        Wtot = Wself + Wext
        Ttot = np.sqrt(Tself ** 2 + Text ** 2)
        V["x1"].append(x1)
        V["x2"].append(np.log10(np.maximum(gext, 1e-300) / A0))
        V["x3"].append(np.log10(np.maximum(Wtot, 1e-300) / A0))
        V["x4a"].append(np.log10(np.maximum(Ttot, 1e-300) / T0))
        V["x4b"].append(q)
        V["x4c"].append(np.minimum(Text / np.maximum(Tself, 1e-300), 1.0))
        V["x4d"].append(np.log10(np.maximum(Text, 1e-300) / T0))
        V["lgb"].append(np.log10(np.maximum(s.g_b, 1e-30)))
        V["lr"].append(np.log10(s.r / MPC))
        V["W_self"].append(Wself)
        V["W_ext"].append(Wext)
        V["T_self"].append(Tself)
        V["T_ext"].append(Text)
        V["gext"].append(gext)
    if verbose:
        print(f"   built {len(systems)} systems x {len(systems[0].r)} radii")
    return {k: np.array(v) for k, v in V.items()}


# ------------------------------------------------- collinearity with (g_b, r)
def collinearity(V, keys, mask):
    """R^2 of each variable on a QUADRATIC in (log g_b, log r), and the residual.

    Run AI measured 0.9863 for potential depth: 98.6% of the variable at the
    shear-measured radii is a function of acceleration and radius, leaving
    0.087 dex of leverage.  The same number for every candidate here.
    """
    lg = V["lgb"][mask].ravel()
    lr = V["lr"][mask].ravel()
    X = np.column_stack([np.ones_like(lg), lg, lg ** 2, lr, lr ** 2, lg * lr])
    out, coefs = {}, {}
    for k in keys:
        y = V[k][mask].ravel()
        good = np.isfinite(y) & np.isfinite(lg) & np.isfinite(lr)
        c, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
        res = y[good] - X[good] @ c
        ss = np.var(y[good])
        out[k] = dict(R2=float(1.0 - np.var(res) / ss) if ss > 0 else 0.0,
                      resid_rms=float(np.std(res)), total_rms=float(np.sqrt(ss)))
        coefs[k] = c
    return out, coefs


def residualise(V, keys, coefs):
    """Project the (log g_b, log r) quadratic out of each variable."""
    out = {}
    for k in keys:
        lg, lr = V["lgb"], V["lr"]
        X = np.stack([np.ones_like(lg), lg, lg ** 2, lr, lr ** 2, lg * lr], -1)
        out[k] = V[k] - X @ coefs[k]
    return out


# ----------------------------------------------------- the coarse-graining gate
def _mesh_representation(s, n_cell, f_star, r_max):
    """The SAME continuous mass as n_cell catalogue rows, mass at centroids.

    A 3-D spherical-polar mesh whose cells carry the exact enclosed mass of the
    continuum profile and sit at their own centre of mass.  n_cell = 1 is the
    catalogue's own choice: one row, all the mass, at the centre.
    """
    Mtot = float(np.interp(r_max, s.r, s.M_b))
    if n_cell == 1:
        return np.zeros((1, 3)), np.array([Mtot])
    n_r = max(1, int(round(n_cell ** (1.0 / 3.0))))
    n_ang = max(1, int(round(n_cell / n_r)))
    edges = np.geomspace(s.r[0], r_max, n_r + 1)
    Mc = np.interp(edges, s.r, s.M_b)
    dM = np.diff(Mc) / n_ang
    # mass-weighted mean radius of each shell, so the coarse mesh is the best
    # possible single-point stand-in and the drift measured is not a
    # placement artefact
    ok = s.r <= r_max
    rr, MM = s.r[ok], s.M_b[ok]
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (rr[1:] + rr[:-1])
                                           * np.diff(MM))])
    num = np.interp(edges, rr, cum)
    rc = np.diff(num) / np.maximum(np.diff(np.interp(edges, rr, MM)), 1e-30)
    rc = np.clip(rc, edges[:-1], edges[1:])
    dirs = _fib_dirs(n_ang)
    pos = (rc[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
    m = np.repeat(dM, n_ang)
    return pos, m


def _W_of_points(pos, m, probes, eps):
    d2 = np.sum((probes[:, None, :] - pos[None, :, :]) ** 2, axis=-1)
    return np.sum(G * m[None, :] / (d2 + eps ** 2), axis=1)


def _g_of_points(pos, m, probes, eps):
    d = pos[None, :, :] - probes[:, None, :]
    r2 = np.sum(d ** 2, axis=-1) + eps ** 2
    v = np.sum(G * m[None, :, None] * d / r2[..., None] ** 1.5, axis=1)
    return np.sqrt(np.sum(v ** 2, axis=-1))


def coarse_grain_gate(s, wells, eps=EPS_PRIMARY,
                      Ns=(1, 10, 64, 512, 4096, 32768)):
    """UNIFORM and SELECTIVE refinement of the identical continuous mass.

    Uniform refinement is known to be toothless: splitting every row into k
    equal pieces at the same place multiplies a mass-exponent-p sum by
    k^(1-p), a global constant that any fitted amplitude absorbs, and for the
    p = 1 sums here it changes nothing at all.  The test with teeth refines ONE
    object and leaves its neighbour alone.

    Reference for the drift is the continuum: the exact angular integral for W,
    and Newton's G M(<r)/r^2 for the vector sum.
    """
    r_max = 8.0 * MPC
    probes_r = np.geomspace(0.3 * MPC, 4.0 * MPC, 8)
    dirs = _fib_dirs(6)
    probes = (probes_r[:, None, None] * dirs[None, :, :]).reshape(-1, 3)
    pr = np.repeat(probes_r, 6)

    W_ref = np.interp(pr, s.r, v3_self(s, eps))
    g_ref = np.interp(pr, s.r, s.g_b)

    def drift(v, ref):
        e = np.log10(np.maximum(v, 1e-300) / ref)
        return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))

    dW, dg, series = {}, {}, {}
    for N in Ns:
        pos, m = _mesh_representation(s, N, s.f_star, r_max)
        W = _W_of_points(pos, m, probes, eps)
        gv = _g_of_points(pos, m, probes, eps)
        dW[N], mW = drift(W, W_ref)
        dg[N], mg = drift(gv, g_ref)
        series[str(N)] = dict(n_rows=int(len(m)), drift_W_dex=dW[N],
                              drift_g_dex=dg[N], drift_W_max_dex=mW,
                              drift_g_max_dex=mg)

    # SELECTIVE refinement, the test with teeth.  The same continuous mass is
    # split at 1 Mpc; the INNER part is refined to N cells while the OUTER part
    # is left as ONE row at its own centre of mass, and then the other way
    # round.  Under UNIFORM refinement a mass exponent p cancels exactly (the
    # brief's p = 0.5, 1, 2 all give the identical drift), so uniform
    # refinement cannot discriminate; splitting the object unevenly does.
    r_split = 1.0 * MPC
    M_in = float(np.interp(r_split, s.r, s.M_b))
    M_out = float(np.interp(r_max, s.r, s.M_b)) - M_in
    ok = (s.r > r_split) & (s.r <= r_max)
    r_out_com = float(np.trapezoid(s.r[ok] * np.gradient(s.M_b)[ok])
                      / max(np.trapezoid(np.gradient(s.M_b)[ok]), 1e-30))
    r_out_com = float(np.clip(r_out_com, r_split, r_max))
    pout = np.array([[r_out_com, 0.0, 0.0]])
    N_ref = 262144
    pin, min_ = _mesh_representation(s, N_ref, s.f_star, r_split)
    Wr = _W_of_points(np.vstack([pin, pout]),
                      np.concatenate([min_, [M_out]]), probes, eps)
    gr = _g_of_points(np.vstack([pin, pout]),
                      np.concatenate([min_, [M_out]]), probes, eps)
    selW, selg = {}, {}
    for N in Ns:
        pin, min_ = _mesh_representation(s, N, s.f_star, r_split)
        pos = np.vstack([pin, pout])
        m = np.concatenate([min_, [M_out]])
        selW[N], _ = drift(_W_of_points(pos, m, probes, eps), Wr)
        selg[N], _ = drift(_g_of_points(pos, m, probes, eps), gr)
        series[str(N)]["drift_W_selective_dex"] = selW[N]
        series[str(N)]["drift_g_selective_dex"] = selg[N]

    def slope(dr):
        n = [N for N in Ns if dr.get(N, 0) > 1e-12]
        if len(n) < 3:
            return None
        x = np.log(np.array(n, float))
        y = np.log(np.array([dr[N] for N in n]))
        return float(-np.polyfit(x, y, 1)[0])

    # a mass exponent p != 1 under UNIFORM refinement: the toothlessness check
    uni = {}
    for p in (0.5, 1.0, 2.0):
        pos, m = _mesh_representation(s, 512, s.f_star, r_max)
        v1 = np.sum(G * m[None, :] ** p / (np.sum(
            (probes[:, None, :] - pos[None, :, :]) ** 2, -1) + eps ** 2), 1)
        pos2 = np.repeat(pos, 4, axis=0)
        m2 = np.repeat(m / 4.0, 4)
        v2 = np.sum(G * m2[None, :] ** p / (np.sum(
            (probes[:, None, :] - pos2[None, :, :]) ** 2, -1) + eps ** 2), 1)
        uni[str(p)] = float(np.std(np.log10(v2 / v1)))

    # The EXTERNAL sums, which is what V2 and V4d actually use.  Every
    # neighbour is 10-100 Mpc away and the probe sits inside a 4 Mpc sphere,
    # so a neighbour is point-like to the probe by a wide margin.  Refine the
    # NEIGHBOURS and measure the drift: this is the gate that V2 has to pass
    # and the internal-dominated V3 does not.
    d, M = wells.neighbours(s.id, s.z)
    near = np.argsort(np.sum(d ** 2, axis=1))[:60]
    ext = {}
    R_typ = 1.5 * MPC          # a neighbour's own R500-ish extent, generous
    rng = np.random.default_rng(20260904)
    base_W = _W_of_points(d[near], M[near], probes, eps)
    base_g = _g_of_points(d[near], M[near], probes, eps)
    for K in (1, 8, 64, 512):
        if K == 1:
            pos, m = d[near], M[near]
        else:
            u = rng.normal(size=(len(near), K, 3))
            u /= np.linalg.norm(u, axis=-1, keepdims=True)
            rad = R_typ * rng.random((len(near), K)) ** (1.0 / 3.0)
            pos = (d[near][:, None, :] + u * rad[..., None]).reshape(-1, 3)
            m = np.repeat(M[near] / K, K)
        ext[str(K)] = dict(
            drift_W_ext_dex=drift(_W_of_points(pos, m, probes, eps), base_W)[0],
            drift_g_ext_dex=drift(_g_of_points(pos, m, probes, eps), base_g)[0])

    return dict(system=s.id, N_grid=list(Ns), series=series,
                external_refinement=ext,
                drift_W_1row_dex=dW[Ns[0]], drift_g_1row_dex=dg[Ns[0]],
                drift_W_finest_dex=dW[Ns[-1]], drift_g_finest_dex=dg[Ns[-1]],
                beta_N_W=slope(dW), beta_N_g=slope(dg),
                beta_N_W_selective=slope(selW), beta_N_g_selective=slope(selg),
                uniform_refinement_scatter_by_p=uni,
                uniform_note="a x4 uniform split at fixed positions leaves the "
                             "p=1 sum bit-identical and rescales p!=1 by a "
                             "global constant: zero scatter for every p, which "
                             "is why uniform refinement has no teeth")


# ---------------------------------------------------------- responsiveness dS/dq
_RECKEYS = ("id", "z", "DA", "rs", "R500", "n0sq", "eps", "beta", "alpha",
            "e_n0sq", "e_rs", "e_eps", "e_beta", "e_alpha", "Mgas500_pub",
            "l_Mgas", "T", "RA", "DE")


def _rec_of(s, **over):
    r = {k: s.__dict__[k] for k in _RECKEYS if k in s.__dict__}
    r.update(over)
    return r


def variable_responsiveness(systems, wells, n_sys=40):
    """dS/d(theta) != 0 for every headline construction knob, with the spread.

    The programme has caught five monotone-blind statistics.  Before any of
    these variables is fitted, check that each one actually MOVES when the
    quantity it claims to measure moves, and print the spread.
    """
    out = {}
    sub = systems[:n_sys]

    def med(s, v):
        m = (s.r > 0.2 * MPC) & (s.r < 5 * MPC)
        return float(np.median(v[m]))

    def med_rule(s, ru):
        x, r_ref = v1_potential_depth(s, ru)
        m = (s.r > 0.2 * MPC) & (s.r < min(5 * MPC, 0.8 * r_ref))
        return float(np.median(x[m])) if m.sum() > 2 else float("nan")

    v = np.array([[med_rule(s, ru) for ru in RULES] for s in sub])
    out["V1_vs_boundary_rule"] = dict(
        rules=RULES, mean_value=[float(x) for x in np.nanmean(v, 0)],
        spread_dex=float(np.ptp(np.nanmean(v, 0))),
        per_system_spread_sd=float(np.nanstd(np.nanmax(v, 1) - np.nanmin(v, 1))),
        note="each rule is evaluated only inside 0.8 r_ref, since DeltaPhi -> 0"
             " at r_ref by construction")

    eg = [EPS_SENS[0], EPS_PRIMARY, EPS_SENS[1]]
    v = np.array([[med(s, np.log10(v3_self(s, e) / A0)) for e in eg]
                  for s in sub])
    out["V3_vs_smoothing_eps"] = dict(
        eps_kpc=[e / KPC for e in eg], mean_value=[float(x) for x in v.mean(0)],
        spread_dex=float(np.ptp(v.mean(0))),
        per_system_spread_sd=float(np.std(np.ptp(v, axis=1))))

    v = []
    for s in sub[:8]:
        row = []
        for f in (0.5, 1.0, 2.0):
            w2 = Wells.__new__(Wells)
            w2.__dict__.update(wells.__dict__)
            w2.M = wells.M * f
            row.append(med(s, np.log10(external_fields(
                s, w2, EPS_PRIMARY, full_tensor=False)[0] / A0)))
        v.append(row)
    v = np.array(v)
    out["V2_vs_well_mass_scale"] = dict(
        factors=[0.5, 1.0, 2.0], mean_value=[float(x) for x in v.mean(0)],
        d_per_dex=float((v[:, 2] - v[:, 0]).mean() / (2 * math.log10(2.0))),
        spread=float(np.std((v[:, 2] - v[:, 0]) / (2 * math.log10(2.0)))))

    v = []
    for s in sub:
        row = []
        for da in (-0.1, 0.0, 0.1):
            s2 = P.System(_rec_of(s, alpha=s.alpha + da), s.f_star)
            row.append(med(s2, np.log10(np.maximum(v4_self(s2)[0], 1e-40) / T0)))
        v.append(row)
    v = np.array(v)
    out["V4a_vs_density_alpha"] = dict(
        d_per_unit_alpha=float((v[:, 2] - v[:, 0]).mean() / 0.2),
        spread=float(np.std((v[:, 2] - v[:, 0]) / 0.2)))

    v = []
    for s in sub:
        row = []
        for db in (-0.05, 0.0, 0.05):
            s2 = P.System(_rec_of(s, beta=s.beta + db), s.f_star)
            row.append(med(s2, v4_self(s2)[1]))
        v.append(row)
    v = np.array(v)
    out["V4b_shape_vs_density_beta"] = dict(
        d_per_unit_beta=float((v[:, 2] - v[:, 0]).mean() / 0.1),
        spread=float(np.std((v[:, 2] - v[:, 0]) / 0.1)))
    return out


# --------------------------------------------------------------------- driver
def main():
    hdr("A.  INGEST, with row and column counts asserted")
    recs, obs, systems = load_all(0.0)
    wells = Wells(recs, 0.0)
    print(f"   well network: {len(wells.M)} catalogued concentrations, "
          f"M_b(<R500) median {np.median(wells.M)/MSUN:.3e} Msun")
    d = np.sqrt(np.sum((wells.pos[:, None, :] - wells.pos[None, :, :]) ** 2,
                       axis=-1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(1) / MPC
    print(f"   nearest-neighbour comoving separation: median {np.median(nn):.2f}"
          f" Mpc, 10th pct {np.percentile(nn,10):.2f}, min {nn.min():.3f}")
    RES["well_network"] = dict(
        n_wells=int(len(wells.M)),
        Mb_R500_median_Msun=float(np.median(wells.M) / MSUN),
        nn_median_Mpc=float(np.median(nn)),
        nn_p10_Mpc=float(np.percentile(nn, 10)),
        nn_min_Mpc=float(nn.min()),
        sigma_z_assumed=SIGMA_Z_ASSUMED,
        los_distance_error_Mpc=float(
            (P.CLIGHT / (70e3 / MPC)) * SIGMA_Z_ASSUMED * 1.35 / MPC),
        note="no redshift-error column is published by Bahar+2022; the LOS "
             "distance error implied by the assumed sigma_z is comparable to "
             "the median neighbour separation, so the 3-D well geometry along "
             "the line of sight is only marginally resolved")

    hdr("B.  THE FOUR VARIABLES on every system's radial grid")
    V = build_variables(systems, wells)
    rr = V["lr"]
    mask = (rr > math.log10(FIT_RANGE_MPC[0])) & (rr < math.log10(FIT_RANGE_MPC[1]))
    print(f"   {int(mask.sum())} (system, radius) points in "
          f"{FIT_RANGE_MPC[0]}-{FIT_RANGE_MPC[1]} Mpc")

    for k, lbl in [("x1", "V1 log10 DeltaPhi/Phi0"),
                   ("x2", "V2 log10 |g_ext|/a0"),
                   ("x3", "V3 log10 W/a0"),
                   ("x4a", "V4a log10 |T~|/T0"),
                   ("x4b", "V4b rho/<rho> (tidal shape)"),
                   ("x4d", "V4d log10 |T~_ext|/T0")]:
        v = V[k][mask]
        print(f"   {lbl:32s} median {np.median(v):+9.4f}   "
              f"spread(10-90) {np.percentile(v,90)-np.percentile(v,10):7.4f}")

    hdr("C.  THE TWO STRUCTURAL FACTS, MEASURED")
    ws, we = V["W_self"][mask], V["W_ext"][mask]
    ts, te = V["T_self"][mask], V["T_ext"][mask]
    ge = V["gext"][mask]
    gb = 10.0 ** V["lgb"][mask]
    print(f"   |g_ext| / a0                      median {np.median(ge)/A0:.3e}")
    print(f"   |g_ext| / g_bar                   median {np.median(ge/gb):.3e}")
    print(f"   W_self / a0                       median {np.median(ws)/A0:.3e}")
    print(f"   W_ext  / W_self                   median {np.median(we/ws):.3e}")
    print(f"   |T~_ext| / |T~_self|              median {np.median(te/ts):.3e}")
    print(f"   W_self / g_bar  (no shell theorem) median {np.median(ws/gb):.3f}")
    RES["structural"] = dict(
        gext_over_a0_median=float(np.median(ge) / A0),
        gext_over_gbar_median=float(np.median(ge / gb)),
        Wself_over_a0_median=float(np.median(ws) / A0),
        Wext_over_Wself_median=float(np.median(we / ws)),
        Text_over_Tself_median=float(np.median(te / ts)),
        Wself_over_gbar_median=float(np.median(ws / gb)),
        cancellation_cost_dex=float(np.log10(np.median(ws) / np.median(ge))))
    print(f"\n   THE COST OF DIRECTIONAL CANCELLATION: "
          f"{RES['structural']['cancellation_cost_dex']:.2f} dex between the "
          f"directionless sum and the vector sum on the same catalogue.")

    # within-object radial leverage of each variable
    hdr("D.  WITHIN-OBJECT RADIAL LEVERAGE (the design's whole basis)")
    lev = {}
    for k in ("x1", "x2", "x3", "x4a", "x4b", "x4d"):
        per = []
        for i in range(len(systems)):
            v = V[k][i][mask[i]]
            if v.size > 2 and np.all(np.isfinite(v)):
                per.append(np.ptp(v))
        lev[k] = dict(median_within_object_range=float(np.median(per)),
                      p90=float(np.percentile(per, 90)))
        print(f"   {k:5s} median within-object range over 0.2-5 Mpc: "
              f"{np.median(per):9.5f}   p90 {np.percentile(per,90):9.5f}")
    # and the between-object spread at a matched radius, for contrast
    j = int(np.argmin(np.abs(V["lr"][0] - math.log10(1.0))))
    for k in ("x1", "x2", "x3", "x4a", "x4b", "x4d"):
        v = V[k][:, j]
        lev[k]["between_object_sd_at_1Mpc"] = float(np.std(v[np.isfinite(v)]))
        print(f"   {k:5s} between-object sd at 1 Mpc: "
              f"{lev[k]['between_object_sd_at_1Mpc']:9.5f}")
    RES["leverage"] = lev

    hdr("E.  COLLINEARITY with a quadratic in (log g_bar, log r)")
    keys = ["x1", "x2", "x3", "x4a", "x4b", "x4d"]
    col, coefs = collinearity(V, keys, mask)
    for k in keys:
        print(f"   {k:5s} R^2 = {col[k]['R2']:.4f}   residual "
              f"{col[k]['resid_rms']:.4f}  of total {col[k]['total_rms']:.4f}")
    RES["collinearity"] = col
    # and at the SHEAR-MEASURED radii only, directly comparable to Run AI's
    # 0.9863 / 0.087 dex for potential depth
    shear_mask = np.zeros_like(mask)
    for i in range(len(systems)):
        for Rj in obs.R[i]:
            shear_mask[i, int(np.argmin(np.abs(systems[i].r - Rj)))] = True
    col_s, _ = collinearity(V, keys, shear_mask)
    print("   -- at the shear-measured radii only --")
    for k in keys:
        print(f"   {k:5s} R^2 = {col_s[k]['R2']:.4f}   residual "
              f"{col_s[k]['resid_rms']:.4f}  of total {col_s[k]['total_rms']:.4f}")
    RES["collinearity_at_shear_radii"] = col_s

    hdr("F.  RESPONSIVENESS of each construction, dS/dtheta with the spread")
    resp = variable_responsiveness(systems, wells)
    print(json.dumps(resp, indent=2))
    RES["variable_responsiveness"] = resp

    hdr("G.  COARSE-GRAINING GATE on V2 and V3")
    order = np.argsort([-s.M_b[-1] for s in systems])
    gates = []
    for i in list(order[:2]) + [int(order[len(order) // 2])]:
        gate = coarse_grain_gate(systems[i], wells)
        gates.append(gate)
        print(f"\n   system {gate['system']}")
        print(f"      {'N rows':>8s}  {'drift W (dex)':>14s}  "
              f"{'drift |g| (dex)':>16s}  {'W sel':>8s}  {'g sel':>8s}")
        for N in gate["N_grid"]:
            e = gate["series"][str(N)]
            print(f"      {e['n_rows']:8d}  {e['drift_W_dex']:14.6f}  "
                  f"{e['drift_g_dex']:16.6f}  "
                  f"{e['drift_W_selective_dex']:8.4f}  "
                  f"{e['drift_g_selective_dex']:8.4f}")
        print(f"      beta_N (= -dln drift/dln N):  W {gate['beta_N_W']:.4f}, "
              f"g {gate['beta_N_g']:.4f}; selective  W "
              f"{gate['beta_N_W_selective']:.4f}, "
              f"g {gate['beta_N_g_selective']:.4f}")
        print("      EXTERNAL wells refined into K pieces of 1.5 Mpc extent:")
        for K, e in gate["external_refinement"].items():
            print(f"        K={K:>4s}  drift W_ext {e['drift_W_ext_dex']:.6f} "
                  f"dex   drift |g_ext| {e['drift_g_ext_dex']:.6f} dex")
        print(f"      uniform x4 split, scatter by mass exponent p: "
              f"{gate['uniform_refinement_scatter_by_p']}")
    RES["coarse_graining"] = gates

    np.savez_compressed(os.path.join(HERE, "envvars_table.npz"),
                        **{k: v for k, v in V.items()},
                        mask=mask,
                        ids=np.array([s.id for s in systems]))
    with open(os.path.join(HERE, "envvars_build.json"), "w",
              encoding="utf-8") as f:
        json.dump(RES, f, indent=1)
    print(f"\n   wrote envvars_table.npz and envvars_build.json")
    return recs, obs, systems, wells, V, mask


if __name__ == "__main__":
    main()
