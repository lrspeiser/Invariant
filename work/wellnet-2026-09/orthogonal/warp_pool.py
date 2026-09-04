"""Pooled warped-disc direction test (system 4 of the lane).

WHY POOLING IS NOT OPTIONAL.  Run W measured, on the single warped disc
NGC 2685, a Spearman correlation of +0.904 between ring orientation and
radius over 21 tilted rings with full 3-D spin normals.  A direction test on
one warped galaxy therefore measures RADIUS and calls it direction.  Different
galaxies present different orientations at the same radius; that, and only
that, breaks the degeneracy.  The estimator below is explicitly a
fixed-effects regression:

    ln V_obs - ln V_pred(Lambda=1)  =  a_i  +  b ln R  +  c sin^2 psi

  * a_i  absorbs distance, stellar M/L and global inclination error PER GALAXY
  * b    absorbs ANY common radial trend, so a radius effect cannot be read
         as a direction effect -- this is the Run W protection, made explicit
  * c    is the direction term, and it is what maps onto A_dyn

G_BAR FOR A WARPED SOURCE IS DIRECTION-DEPENDENT and the axisymmetric solver
does not apply.  The Newtonian field here is therefore NOT taken from an
axisymmetric solve: it is integrated exactly over the tabulated tilted-ring
geometry as a sum of circular wires, which has no discretisation error at all
in the source geometry (only the wire softening, declared and varied).

THE RING CONDITION, exactly.  For a ring of radius r whose plane is tilted by
psi from the inner-disc plane, a point at ring azimuth phi sits at
    R = r sqrt(cos^2 phi + sin^2 phi cos^2 psi),   z = r sin phi sin psi
and the inward force along the ring radius is
    g_ring(phi) = g_R (R/r) + g_z (z/r).
V^2 = r <g_ring>_phi.  At psi = 0 this reduces to V^2 = r g_R(r), exactly.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import orbit_model as OM                                          # noqa: E402

WV = os.path.join(OM.REPO, "work", "wellnet-2026-09", "env-data", "raw",
                  "warps-vertical")
RINGS = os.path.join(WV, "vizier_verheijen2001_uma_tiltedring_rotcurves.tsv")
SAMPLE = os.path.join(WV, "vizier_verheijen2001_uma_sample.tsv")
PHOT = os.path.join(WV, "vizier_verheijen2001_uma_photometry.tsv")

#: declared in advance
WARP_MIN_DEG = 2.0          # a ring is 'warped' if psi exceeds this
#: Wire softening.  Each "wire" stands for an ANNULUS of finite width, so the
#: softening has to scale with that width or the a -> R logarithmic
#: singularity is regularised inconsistently from annulus to annulus and the
#: wire sum does not converge to the exact Freeman disc (measured: a fixed
#: 0.05 kpc gives -10% at 5 kpc and gets WORSE with more annuli).
#: eps_j^2 = (0.5 da_j)^2 + h_z^2.
SOFT_HZ_KPC = 0.20          # disc half-thickness entering the softening
V_ERR_FRAC = 0.05           # fractional velocity error floor
V_ERR_FLOOR = 5.0           # km/s
LAM_GRID = np.round(np.geomspace(0.5, 2.0, 13), 4)
N_PHI_SRC = 96              # azimuthal nodes per source wire
N_PHI_RING = 32             # azimuthal nodes per target ring


def _vizier(path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8")
             if not l.startswith("#")]
    hdr, body = None, []
    for i, l in enumerate(lines):
        if l.startswith("recno"):
            hdr = [c.strip() for c in l.split("\t")]
            body = lines[i + 3:]
            break
    out = []
    for l in body:
        c = [x.strip() for x in l.split("\t")]
        if len(c) != len(hdr):
            continue
        out.append(dict(zip(hdr, c)))
    return out, hdr


def _f(x, d=np.nan):
    try:
        return float(x)
    except (TypeError, ValueError):
        return d


def normal_from(i_deg, pa_deg):
    """Unit normal of a ring from inclination and position angle."""
    i = np.radians(i_deg)
    p = np.radians(pa_deg)
    return np.array([np.sin(i) * np.sin(p), -np.sin(i) * np.cos(p), np.cos(i)])


def load_galaxies():
    rings, _ = _vizier(RINGS)
    samp, _ = _vizier(SAMPLE)
    phot, _ = _vizier(PHOT)
    dist = {}
    for p in phot:
        m, A, M = _f(p["Bmag"]), _f(p["AB"]), _f(p["BMAG"])
        if np.isfinite(m) and np.isfinite(M):
            dist[p["Name"].strip()] = 10 ** (((m - A - M) / 5.0) + 1.0) / 1e6
    einc = {s["Name"].strip(): _f(s["e_iadopt"], 3.0) for s in samp}
    by = {}
    for r in rings:
        by.setdefault(r["Name"].strip(), []).append(r)
    gals = []
    for name, rr in sorted(by.items()):
        rr = sorted(rr, key=lambda x: _f(x["Rad"]))
        R = np.array([_f(x["Rad"]) for x in rr])
        V = np.array([_f(x["Vrot"]) for x in rr])
        inc = np.array([_f(x["Incl"]) for x in rr])
        pa = np.array([_f(x["PA"]) for x in rr])
        m = np.isfinite(R) & np.isfinite(V) & np.isfinite(inc) \
            & np.isfinite(pa) & (V > 10) & (V < 500)      # plausibility gate
        R, V, inc, pa = R[m], V[m], inc[m], pa[m]
        if len(R) < 5 or name not in dist:
            continue
        n0 = normal_from(inc[0], pa[0])
        psi = np.degrees(np.arccos(np.clip(
            [abs(float(normal_from(i, p) @ n0)) for i, p in zip(inc, pa)],
            -1, 1)))
        if np.max(psi) <= WARP_MIN_DEG:
            continue
        D = dist[name]
        Rk = R * (D * 1e3) * np.pi / (180 * 3600)          # arcsec -> kpc
        gals.append(dict(name=name, R_kpc=Rk, V=V, inc=inc, pa=pa, psi=psi,
                         D_mpc=D, e_inc=einc.get(name, 3.0),
                         n_rings=len(Rk)))
    return gals


# ------------------------------------------------------- exact warped field
_WFCACHE = {}


def warped_field(gal, Sigma0, Rd, targets_R, targets_z, n_src=80):
    """Sigma0 enters LINEARLY, so the unit-Sigma0 field is cached per
    (galaxy, Rd, target set) and scaled.  The MOND response is applied
    afterwards and is of course not linear."""
    key = (gal["name"] if "name" in gal else "_", round(float(Rd), 6),
           int(n_src), targets_R.tobytes(), np.asarray(targets_z).tobytes(),
           N_PHI_SRC, SOFT_HZ_KPC)
    hit = _WFCACHE.get(key)
    if hit is None:
        hit = _warped_field_unit(gal, Rd, targets_R, targets_z, n_src)
        if len(_WFCACHE) > 20000:
            _WFCACHE.clear()
        _WFCACHE[key] = hit
    return Sigma0 * hit[0], Sigma0 * hit[1]


def _warped_field_unit(gal, Rd, targets_R, targets_z, n_src=80):
    """(g_R, g_z) at cylindrical (R, z) in the INNER-DISC frame, from an
    exponential surface density laid on the tabulated tilted-ring geometry.

    Each source annulus is a circular wire of radius a, mass dM, and normal
    interpolated from the tabulated i(R), PA(R).  The field is the exact wire
    integral -- no grid, so the source geometry carries no discretisation
    error.  Only the wire softening is approximate, and it is varied.
    """
    # LOG-spaced source annuli: a uniform grid leaves the inner disc
    # unresolved and the wire integration then misses the exact Freeman
    # circular speed by 35% at 2 kpc (see `_check_freeman`).
    a = np.geomspace(0.02, max(gal["R_kpc"][-1] * 2.0, 8.0), n_src)
    edges = np.concatenate([[a[0] ** 2 / a[1]], np.sqrt(a[1:] * a[:-1]),
                            [a[-1] ** 2 / a[-2]]])
    dM = 2 * np.pi * 1.0 * (Rd ** 2) * (
        (1 + edges[:-1] / Rd) * np.exp(-edges[:-1] / Rd)
        - (1 + edges[1:] / Rd) * np.exp(-edges[1:] / Rd))        # exact shells
    eps2 = (0.5 * np.diff(edges)) ** 2 + SOFT_HZ_KPC ** 2
    # ring normals, extrapolated flat inside and held at the last value outside
    nn = np.stack([normal_from(i, p) for i, p in zip(gal["inc"], gal["pa"])])
    n0 = nn[0]
    Rt = gal["R_kpc"]
    src_n = np.empty((n_src, 3))
    for k in range(3):
        src_n[:, k] = np.interp(a, Rt, nn[:, k], left=n0[k], right=nn[-1, k])
    src_n /= np.linalg.norm(src_n, axis=1)[:, None]
    # rotate every normal into the inner-disc frame where n0 -> +z
    Rot = _align(n0)
    src_n = src_n @ Rot.T
    phi = (np.arange(N_PHI_SRC) + 0.5) * 2 * np.pi / N_PHI_SRC
    gR = np.zeros(len(targets_R))
    gz = np.zeros(len(targets_R))
    X = np.stack([targets_R, np.zeros_like(targets_R), targets_z], axis=1)
    for j in range(n_src):
        e1, e2 = _basis(src_n[j])
        s = a[j] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
        d = X[:, None, :] - s[None, :, :]                     # (nt, nphi, 3)
        r2 = np.sum(d * d, axis=2) + eps2[j]
        w = (dM[j] / N_PHI_SRC) / r2 ** 1.5
        # g_vec = -sum w d ; the INWARD components are therefore +f
        f = np.sum(w[:, :, None] * d, axis=1)
        gR += f[:, 0]
        gz += f[:, 2]
    c = OM.G * OM.MSUN / OM.KPC ** 2
    return np.maximum(gR * c, 0.0), gz * c


def _check_freeman(Sigma0=3e8, Rd=2.5, R=np.array([2., 5., 10., 20.])):
    """Gate: the wire integration must reproduce the exact Freeman disc."""
    gal = dict(R_kpc=np.array([0.5, 30.0]), inc=np.array([0.0, 0.0]),
               pa=np.array([0.0, 0.0]))
    gR, gz = warped_field(gal, Sigma0, Rd, R, np.zeros_like(R))
    v = np.sqrt(np.maximum(gR * R * OM.KPC, 0.0)) / OM.KMS
    ve = OM.AX.freeman_vc(R, Sigma0 * OM.MSUN / OM.KPC ** 2, Rd) / 1e3
    return dict(R_kpc=R.tolist(), v_wire=v.tolist(), v_freeman=ve.tolist(),
                max_frac_err=float(np.max(np.abs(v / ve - 1.0))),
                gz_midplane_max=float(np.max(np.abs(gz))),
                soft_hz_kpc=SOFT_HZ_KPC)


def _align(n):
    """Rotation taking n to +z."""
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3) if n[2] > 0 else -np.eye(3)
    c = float(n @ z)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s ** 2)


def _basis(n):
    t = np.array([1.0, 0.0, 0.0])
    if abs(n[0]) > 0.9:
        t = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(n, t)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def ring_V(gal, law, Sigma0, Rd, r, psi_deg, Lam):
    """Predicted circular speed of a ring of radius r tilted by psi, under a
    candidate law deformed by Lambda.  Exact ring average, no small-psi
    expansion."""
    ps = np.radians(psi_deg)
    n_ph = 1 if abs(psi_deg) < 1e-9 else N_PHI_RING
    ph = (np.arange(n_ph) + 0.5) * 2 * np.pi / n_ph
    rr = np.sqrt(np.cos(ph) ** 2 + (np.sin(ph) * np.cos(ps)) ** 2)
    R = r * rr
    z = r * np.sin(ph) * np.sin(ps)
    gRN, gzN = warped_field(gal, Sigma0, Rd, R, np.abs(z))
    gRN0, _ = warped_field(gal, Sigma0, Rd, R, np.zeros_like(R))
    gN = np.sqrt(gRN ** 2 + gzN ** 2)
    gN0 = gRN0
    if law["base"] == "newton":
        F, F0 = np.ones_like(gN), np.ones_like(gN0)
    else:
        F = OM._g_of_gN(law["base"], gN, law["a0"]) / np.maximum(gN, 1e-300)
        F0 = OM._g_of_gN(law["base"], gN0, law["a0"]) / np.maximum(gN0, 1e-300)
    gR, gz = F * gRN, F * gzN
    gR0 = F0 * gRN0
    gR_L = gR0 + (gR - gR0) / Lam            # in-plane leg exactly preserved
    gz_L = gz / Lam
    g_ring = gR_L * (R / r) + gz_L * (np.abs(z) / r)
    return float(np.sqrt(max(np.mean(g_ring) * r * OM.KPC, 0.0)) / OM.KMS)


def fit_inplane(gal, law, flat):
    """In-plane leg for ONE galaxy: the unwarped rings alone, then freeze."""
    from scipy.optimize import minimize
    R, V = gal["R_kpc"][flat], gal["V"][flat]
    sig = np.maximum(V_ERR_FRAC * V, V_ERR_FLOOR)

    def chi2(p):
        S0, Rd = np.exp(p)
        vm = np.array([ring_V(gal, law, S0, Rd, r, 0.0, 1.0) for r in R])
        return float(np.sum(((V - vm) / sig) ** 2))

    best, bp = np.inf, None
    for p0 in ([np.log(3e8), np.log(2.5)], [np.log(1e8), np.log(4.0)]):
        r = minimize(chi2, p0, method="Nelder-Mead",
                     options=dict(maxiter=120, xatol=1e-3, fatol=1e-2))
        if r.fun < best:
            best, bp = r.fun, r.x
    return float(np.exp(bp[0])), float(np.exp(bp[1])), best


def run(verbose=True):
    gals = load_galaxies()
    laws = [dict(name=l.name, base=l.base, a0=l.a0)
            for l in OM.frozen_laws() if l.name in ("newton", "rar", "aqual")]
    out = dict(n_galaxies=len(gals), warp_min_deg=WARP_MIN_DEG,
               soft_hz_kpc=SOFT_HZ_KPC, freeman_gate=_check_freeman(), lam_grid=LAM_GRID.tolist(),
               galaxies=[dict(name=g["name"], n_rings=g["n_rings"],
                              psi_max=float(g["psi"].max()),
                              R_kpc=[float(x) for x in g["R_kpc"]],
                              psi_deg=[float(x) for x in g["psi"]],
                              D_mpc=g["D_mpc"]) for g in gals],
               results={})
    # Run W's degeneracy, measured on THIS sample before anything is fitted
    from scipy.stats import spearmanr
    rho_within = []
    for g in gals:
        if len(set(np.round(g["psi"], 3))) > 2:
            rho_within.append(float(spearmanr(g["R_kpc"], g["psi"]).statistic))
    allR = np.concatenate([g["R_kpc"] for g in gals])
    allP = np.concatenate([g["psi"] for g in gals])
    out["degeneracy"] = dict(
        spearman_R_psi_within_galaxy=rho_within,
        median_within=float(np.median(rho_within)) if rho_within else None,
        spearman_R_psi_pooled=float(spearmanr(allR, allP).statistic),
        note="within a galaxy orientation tracks radius (Run W's +0.904 on "
             "NGC 2685); pooled across galaxies the correlation is what is "
             "left after different galaxies present different psi at the "
             "same R, and the fixed-effects b ln R term removes the rest")
    for law in laws:
        rows = []
        for g in gals:
            flat = g["psi"] <= WARP_MIN_DEG
            if flat.sum() < 3 or (~flat).sum() < 1:
                continue
            S0, Rd, c2 = fit_inplane(g, law, flat)
            for L in LAM_GRID:
                for k in np.where(~flat)[0]:
                    vp = ring_V(g, law, S0, Rd, g["R_kpc"][k], g["psi"][k],
                                float(L))
                    if vp <= 0:
                        continue
                    rows.append((g["name"], float(L), float(g["R_kpc"][k]),
                                 float(g["psi"][k]),
                                 float(np.log(g["V"][k]) - np.log(vp)),
                                 float(max(V_ERR_FRAC, V_ERR_FLOOR
                                           / g["V"][k]))))
            if verbose:
                print(f"    warp {law['name']:8s} {g['name']:8s} "
                      f"flat={int(flat.sum()):2d} warped={int((~flat).sum()):2d}"
                      f"  Sigma0={S0:.3g} Rd={Rd:.2f} chi2={c2:.1f}",
                      flush=True)
        out["results"][law["name"]] = _fixed_effects(rows)
        if verbose:
            r = out["results"][law["name"]]
            print(f"  {law['name']}: c(Lambda=1) = {r['c_at_lam1']:+.4f} "
                  f"+- {r['c_err_at_lam1']:.4f}, best Lambda = "
                  f"{r['lam_best']:.3f}, dchi2 range {r['dchi2_range']:.2f}")
    return out


def _fixed_effects(rows):
    """Two DISTINCT regressions, because they are not the same question.

    Model A, used to scan Lambda:   y(Lambda) = a_i + b ln R
      Lambda enters through V_pred, and there is NO sin^2 psi column.  Putting
      one in would be a design bug: the free c would absorb exactly the
      psi-dependence that Lambda produces, and chi^2(Lambda) would come out
      flat by construction (measured: dchi2 range 0.07 over Lambda in
      [0.5, 2.0] with the c column in, against 40+ with it out).

    Model B, evaluated ONLY at Lambda = 1:  y = a_i + b ln R + c sin^2 psi
      c is then a model-independent statement about whether the warped rings
      depart from the frozen law in a way that tracks orientation, with the
      per-galaxy intercept and the common radial trend already removed.
    """
    import collections
    byL = collections.defaultdict(list)
    for nm, L, R, psi, res, err in rows:
        byL[L].append((nm, R, psi, res, err))
    names = sorted({r[0] for r in rows})

    def design(rr, with_psi):
        nm = [x[0] for x in rr]
        R = np.array([x[1] for x in rr])
        ps = np.array([x[2] for x in rr])
        y = np.array([x[3] for x in rr])
        w = 1.0 / np.array([x[4] for x in rr]) ** 2
        k = len(names) + (2 if with_psi else 1)
        X = np.zeros((len(rr), k))
        for i, n in enumerate(nm):
            X[i, names.index(n)] = 1.0
        X[:, len(names)] = np.log(np.maximum(R, 1e-3))
        if with_psi:
            X[:, -1] = np.sin(np.radians(ps)) ** 2
        A = X.T @ (w[:, None] * X) + 1e-10 * np.eye(k)
        b = np.linalg.solve(A, X.T @ (w * y))
        r = y - X @ b
        return b, float(np.sum(w * r ** 2)), np.linalg.inv(A), len(rr)

    chi = {}
    for L, rr in sorted(byL.items()):
        _, c2, _, n = design(rr, with_psi=False)
        chi[float(L)] = c2
    L = np.array(sorted(chi))
    c2 = np.array([chi[x] for x in L])
    i1 = int(np.argmin(np.abs(L - 1.0)))
    bB, chiB, covB, n = design(byL[L[i1]], with_psi=True)
    _, chiA1, _, _ = design(byL[L[i1]], with_psi=False)
    return dict(chi2={str(x): chi[x] for x in L},
                lam_best=float(L[int(np.argmin(c2))]),
                lam_at_grid_edge=bool(np.argmin(c2) in (0, len(L) - 1)),
                dchi2_range=float(c2.max() - c2.min()),
                c_at_lam1=float(bB[-1]),
                c_err_at_lam1=float(np.sqrt(max(covB[-1, -1], 0.0))),
                b_lnR_at_lam1=float(bB[len(names)]),
                dchi2_adding_psi=float(chiA1 - chiB),
                chi2_dof_A=float(chiA1 / max(n - len(names) - 1, 1)),
                n_points=int(n), n_galaxies=len(names))


if __name__ == "__main__":
    import json
    r = run()
    print(json.dumps({k: v for k, v in r.items() if k != "galaxies"},
                     indent=1)[:3000])
