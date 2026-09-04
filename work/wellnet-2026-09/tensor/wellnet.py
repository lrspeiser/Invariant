"""TENSOR 1 -- well alignment.

    n_a(x)   = (x_a - x) / |x_a - x|
    S^ij(x)  = [ sum_a w_a(x) ( n_a^i n_a^j - delta^ij/3 ) ]
               / [ eps + sum_a |w_a(x)| ]
    K(x)     = exp[ s_0(x) I + s_T(x) S(x) ]

The delta/3 subtraction makes S traceless, so S carries only the DIRECTIONAL
information: whether the surrounding wells are concentrated along one axis
(prolate, S has one large positive eigenvalue), in a plane (oblate, one large
negative eigenvalue), or not at all (S = 0).  The normalisation by the total
weight makes S scale-free: doubling every mass, or doubling the number of
wells at fixed geometry, does not change S.  That is the single most important
structural fact about this tensor and it is what separates it from the earlier
QUMOND-lumpiness calculation, whose effect vanished as the population was made
smoother.

exp of a symmetric matrix is symmetric positive definite for any real
symmetric argument, so K is SPD by construction -- but the gate verifies it
numerically anyway, because "by construction" is how the last three bugs got
in.

WHAT s_0 AND s_T ARE.  The brief specifies K = exp[s_0(x) I + s_T(x) S(x)]
with free globals p, q, s, m, L, M_0, Phi_0 but does not say what the two
scalar fields are.  Phi_0 appears in the global list and nowhere else in the
tensor, so the reading taken here is that s_0 and s_T are amplitudes times a
dimensionless environmental gate built from the local baryonic potential:

    s_0(x) = A_0 * g(x),     s_T(x) = A_T * g(x)

    gate "none" : g = 1
    gate "phi"  : g = u/(1+u),           u = |Phi_N(x)| / Phi_0
    gate "gn"   : g = 1/(1 + (|g_N|/a0)^m)

Phi_N and g_N are the NEWTONIAN potential and field of the baryons, so K is a
functional of the source alone and does not add a second nonlinearity on top
of mu(X).  This choice is stated here rather than buried: it is the only place
in the construction where a cluster can be told apart from a galaxy, because S
itself cannot -- it is scale-free.

UNITS: SI throughout (m, kg, s), matching work/gravitylab/solver.py.
"""
from __future__ import annotations

import numpy as np

G = 6.674e-11
A0 = 1.2e-10
KPC = 3.0856775814913673e19
MSUN = 1.98892e30

# symmetric 3x3 stored as (..., 6) = (xx, yy, zz, xy, xz, yz)
SYM = ("xx", "yy", "zz", "xy", "xz", "yz")


# --------------------------------------------------------------- backends
def get_xp(gpu=True):
    """CuPy if asked for and available, else NumPy."""
    if gpu:
        try:
            import cupy as cp
            cp.cuda.Device(0).compute_capability
            return cp
        except Exception:
            pass
    return np


def asnumpy(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


# ------------------------------------------------- symmetric 3x3 algebra
def sym3_from_full(M):
    """(...,3,3) -> (...,6). Symmetrises on the way."""
    M = np.asarray(M)
    return np.stack([M[..., 0, 0], M[..., 1, 1], M[..., 2, 2],
                     0.5 * (M[..., 0, 1] + M[..., 1, 0]),
                     0.5 * (M[..., 0, 2] + M[..., 2, 0]),
                     0.5 * (M[..., 1, 2] + M[..., 2, 1])], axis=-1)


def sym3_to_full(m, xp=np):
    """(...,6) -> (...,3,3)."""
    xx, yy, zz, xy, xz, yz = (m[..., i] for i in range(6))
    r0 = xp.stack([xx, xy, xz], axis=-1)
    r1 = xp.stack([xy, yy, yz], axis=-1)
    r2 = xp.stack([xz, yz, zz], axis=-1)
    return xp.stack([r0, r1, r2], axis=-2)


def sym3_trace(m):
    return m[..., 0] + m[..., 1] + m[..., 2]


def sym3_square(m, xp=np):
    """M @ M for symmetric M, returned in the same 6-vector layout."""
    xx, yy, zz, xy, xz, yz = (m[..., i] for i in range(6))
    return xp.stack([
        xx * xx + xy * xy + xz * xz,
        xy * xy + yy * yy + yz * yz,
        xz * xz + yz * yz + zz * zz,
        xx * xy + xy * yy + xz * yz,
        xx * xz + xy * yz + xz * zz,
        xy * xz + yy * yz + yz * zz,
    ], axis=-1)


def sym3_det(m):
    xx, yy, zz, xy, xz, yz = (m[..., i] for i in range(6))
    return (xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz)
            + xz * (xy * yz - yy * xz))


def sym3_quad(m, v, xp=np):
    """v^T M v with v of shape (...,3)."""
    xx, yy, zz, xy, xz, yz = (m[..., i] for i in range(6))
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    return (xx * vx * vx + yy * vy * vy + zz * vz * vz
            + 2 * (xy * vx * vy + xz * vx * vz + yz * vy * vz))


def sym3_eigvals(m, xp=np):
    """Closed-form eigenvalues of a symmetric 3x3, descending.

    Trigonometric (Cardano) solution of the characteristic polynomial.
    Returned as (...,3) with l0 >= l1 >= l2.
    """
    xx, yy, zz, xy, xz, yz = (m[..., i] for i in range(6))
    q = (xx + yy + zz) / 3.0
    bxx, byy, bzz = xx - q, yy - q, zz - q
    # p^2 = tr(B^2)/6
    p2 = (bxx * bxx + byy * byy + bzz * bzz
          + 2.0 * (xy * xy + xz * xz + yz * yz)) / 6.0
    p = xp.sqrt(xp.maximum(p2, 0.0))
    psafe = xp.where(p > 0, p, 1.0)
    b = xp.stack([bxx, byy, bzz, xy, xz, yz], axis=-1) / psafe[..., None]
    r = sym3_det(b) / 2.0
    r = xp.clip(r, -1.0, 1.0)
    phi = xp.arccos(r) / 3.0
    l0 = q + 2.0 * p * xp.cos(phi)
    l2 = q + 2.0 * p * xp.cos(phi + 2.0 * np.pi / 3.0)
    l1 = 3.0 * q - l0 - l2
    deg = p <= 0
    l0 = xp.where(deg, q, l0)
    l1 = xp.where(deg, q, l1)
    l2 = xp.where(deg, q, l2)
    return xp.stack([l0, l1, l2], axis=-1)


def _dd1(a, d, xp):
    """(exp(a+d) - exp(a)) / d, stable as d -> 0.  Returns exp(a)*expm1(d)/d."""
    small = xp.abs(d) < 1e-7
    dsafe = xp.where(small, 1.0, d)
    ratio = xp.where(small, 1.0 + d / 2.0 + d * d / 6.0,
                     xp.expm1(dsafe) / dsafe)
    return xp.exp(a) * ratio


def sym3_expm(m, xp=np):
    """exp(M) for symmetric 3x3 via Sylvester / Newton divided differences.

        exp(M) = f[l0] I + f[l0,l1] (M - l0 I)
                          + f[l0,l1,l2] (M - l0 I)(M - l1 I)

    No eigenvectors are needed, which is what makes this vectorise.  The two
    first divided differences use expm1 and are stable for any spacing; the
    second is switched to its confluent limit exp(mean)/2 only when ALL three
    eigenvalues coincide to 1e-4 of the spectral scale, where the matrix
    factor it multiplies is itself O(1e-8).
    """
    lam = sym3_eigvals(m, xp)
    l0, l1, l2 = lam[..., 0], lam[..., 1], lam[..., 2]
    f01 = _dd1(l0, l1 - l0, xp)          # f[l0,l1]
    f12 = _dd1(l1, l2 - l1, xp)          # f[l1,l2]
    scale = xp.maximum(xp.abs(l0), xp.abs(l2)) + 1.0
    d02 = l2 - l0
    tiny = xp.abs(d02) < 1e-4 * scale
    d02s = xp.where(tiny, 1.0, d02)
    f012 = xp.where(tiny, xp.exp((l0 + l1 + l2) / 3.0) / 2.0,
                    (f12 - f01) / d02s)

    eye = xp.zeros_like(m)
    eye[..., 0] = 1.0
    eye[..., 1] = 1.0
    eye[..., 2] = 1.0
    A = m - l0[..., None] * eye          # M - l0 I
    B = m - l1[..., None] * eye          # M - l1 I
    # A and B commute (both polynomials in M) so AB is symmetric:
    #   AB = M^2 - (l0+l1) M + l0 l1 I
    M2 = sym3_square(m, xp)
    AB = M2 - (l0 + l1)[..., None] * m + (l0 * l1)[..., None] * eye
    return f01[..., None] * A + f012[..., None] * AB + xp.exp(l0)[..., None] * eye


def sym3_inv(m, xp=np):
    xx, yy, zz, xy, xz, yz = (m[..., i] for i in range(6))
    d = sym3_det(m)
    return xp.stack([(yy * zz - yz * yz), (xx * zz - xz * xz),
                     (xx * yy - xy * xy), -(xy * zz - yz * xz),
                     (xy * yz - yy * xz), -(xx * yz - xy * xz)],
                    axis=-1) / d[..., None]


# --------------------------------------------------------- weight families
WEIGHT_FAMILIES = ("plaw", "expo", "gscreen")


def well_weights(r, M, family, p=1.0, q=2.0, s=1.0, m=1.0, L=300.0 * KPC,
                 M_0=1.0e11 * MSUN, gn_mode="pair", gN_local=None, a0=A0,
                 xp=np):
    """w_a(x) for the three families in the brief.

    r    (..., N) distance from the field point to each well, metres
    M    (N,)     well masses, kg

    family "plaw"    : (M/M_0)^p [1 + (r/L)^q]^-s
           "expo"    : (M/M_0)^p exp[-(r/L)^q]
           "gscreen" : (M/M_0)^p / { [1+(g_N/a0)^m] [1+(r/L)^q]^s }

    gn_mode for "gscreen" -- the brief writes g_N inside a per-well weight
    without saying whose field it is, so both readings are implemented:
        "pair"  g_N = G M_a / r_a^2, the well's own field at the point.
                A genuine per-well reweighting.
        "local" g_N = |grad Phi_N(x)|, the total baryonic field at the point,
                supplied through gN_local.  This is a COMMON factor on every
                weight, so it cancels out of S exactly except through the eps
                regulariser -- i.e. it acts purely as an on/off switch for the
                whole tensor.  Both behaviours are reported.
    """
    mfac = (M / M_0) ** p
    x = r / L
    if family == "plaw":
        w = mfac * (1.0 + x ** q) ** (-s)
    elif family == "expo":
        w = mfac * xp.exp(-(x ** q))
    elif family == "gscreen":
        if gn_mode == "pair":
            gn = G * M / xp.maximum(r, 1e-6 * L) ** 2
        elif gn_mode == "local":
            if gN_local is None:
                raise ValueError("gn_mode='local' needs gN_local")
            gn = gN_local[..., None] * xp.ones_like(r)
        else:
            raise ValueError(gn_mode)
        w = mfac / ((1.0 + (gn / a0) ** m) * (1.0 + x ** q) ** s)
    else:
        raise ValueError(family)
    return w


# ------------------------------------------------------------ the S tensor
def S_tensor(points, wx, wm, family="plaw", eps_w=1e-3, chunk=1 << 15,
             xp=np, gN_local=None, exclude_nearest=False, **wkw):
    """S^ij(x) at every point.

    points (P,3) metres, wx (N,3) well positions, wm (N,) well masses.
    Returns (P,6).  Chunked over points so a 128^3 grid times 300 wells fits.

    Wells closer than 1e-9*L are dropped (n_a would be undefined).

    exclude_nearest: drop the single nearest well from every point's sum.  The
    brief's formula has no self-exclusion, so the default is False and every
    well counts everywhere -- which is why S is near its maximum inside any
    galaxy, the host being that point's dominant well.  A model in which a
    body does not anisotropise its own spacetime needs exclude_nearest=True,
    and the two give materially different answers to the galaxy-limit
    question, so both are mapped rather than one being chosen silently.
    """
    P = points.shape[0]
    out = xp.empty((P, 6), dtype=xp.float64)
    L = wkw.get("L", 300.0 * KPC)
    for i0 in range(0, P, chunk):
        i1 = min(P, i0 + chunk)
        d = wx[None, :, :] - points[i0:i1, None, :]        # (c,N,3)
        r = xp.sqrt(xp.sum(d * d, axis=-1))                # (c,N)
        good = r > 1e-9 * L
        if exclude_nearest:
            near = xp.argmin(xp.where(good, r, xp.inf), axis=1)
            good = good & (xp.arange(r.shape[1])[None, :] != near[:, None])
        rs = xp.where(good, r, 1.0)
        n = d / rs[..., None]
        gl = None if gN_local is None else gN_local[i0:i1]
        w = well_weights(rs, wm[None, :], family, gN_local=gl, xp=xp, **wkw)
        w = xp.where(good, w, 0.0)
        den = eps_w + xp.sum(xp.abs(w), axis=1)            # (c,)
        nx, ny, nz = n[..., 0], n[..., 1], n[..., 2]
        acc = xp.stack([
            xp.sum(w * (nx * nx - 1.0 / 3.0), axis=1),
            xp.sum(w * (ny * ny - 1.0 / 3.0), axis=1),
            xp.sum(w * (nz * nz - 1.0 / 3.0), axis=1),
            xp.sum(w * nx * ny, axis=1),
            xp.sum(w * nx * nz, axis=1),
            xp.sum(w * ny * nz, axis=1),
        ], axis=-1)
        out[i0:i1] = acc / den[:, None]
    return out


# -------------------------------------------------------------- the gates
def gate_field(kind, PhiN=None, gN=None, Phi_0=1.0e12, m=1.0, a0=A0, xp=np):
    """Dimensionless environmental gate g(x) in [0,1].

    "phi" uses the exponent m as well, g = u^m/(1+u^m) with u = |Phi_N|/Phi_0,
    because m is already a global of the brief's third weight family and
    because with m = 1 the gate saturates at a contrast of |Phi_cl|/|Phi_gal|
    -- roughly 50 -- which turns out not to be enough.
    """
    if kind == "none":
        return 1.0
    if kind == "phi":
        u = (xp.abs(PhiN) / Phi_0) ** m
        return u / (1.0 + u)
    if kind == "gn":
        return 1.0 / (1.0 + (gN / a0) ** m)
    raise ValueError(kind)


# --------------------------------------- continuum reference for the gate
def S_rr_continuum(R, r_max, weight, eps_w=1e-3, n_density=None, nr=400,
                   nt=400):
    """S_rr(R) for a spherically symmetric continuum of wells, by quadrature.

    The field point sits at (0,0,R).  For a spherically symmetric well
    distribution the tensor must be S = lambda(r)(rhat rhat - I/3), and

        S_rr = Int n(r') w(d) (nz^2 - 1/3) dV / (eps + Int n(r') w(d) dV)

    with nz = (r' cos(theta) - R)/d, d = |x' - x|.  The azimuthal integral is
    trivial, so this is a 2-D Gauss-Legendre quadrature.  It is the exact
    answer the Monte-Carlo well set in gate 5 must reproduce.

    weight(d) is w_a as a function of distance only (equal-mass wells).
    n_density(r') is the number density; uniform ball by default.
    """
    xr, wr = np.polynomial.legendre.leggauss(nr)
    xt, wt = np.polynomial.legendre.leggauss(nt)
    rp = 0.5 * r_max * (xr + 1.0)
    jr = 0.5 * r_max * wr
    ct = xt
    jt = wt
    RP, CT = np.meshgrid(rp, ct, indexing="ij")
    JW = jr[:, None] * jt[None, :]
    n = 1.0 if n_density is None else n_density(RP)
    d2 = RP ** 2 + R ** 2 - 2 * RP * R * CT
    d = np.sqrt(np.maximum(d2, 1e-30))
    nz = (RP * CT - R) / d
    w = weight(d)
    vol = 2.0 * np.pi * RP ** 2 * n            # dV = 2 pi r'^2 dr' dcos(theta)
    num = float(np.sum(JW * vol * w * (nz ** 2 - 1.0 / 3.0)))
    den = float(np.sum(JW * vol * w))
    return num / (eps_w + den), den


def K_wellnet(points, wx, wm, A_0=0.0, A_T=1.0, gate="none", PhiN=None,
              gN=None, Phi_0=1.0e12, gate_m=1.0, xp=np, **skw):
    """K(x) = exp[ s_0 I + s_T S ], returned as (P,6)."""
    S = S_tensor(points, wx, wm, xp=xp, gN_local=gN, **skw)
    g = gate_field(gate, PhiN=PhiN, gN=gN, Phi_0=Phi_0, m=gate_m, xp=xp)
    s0 = A_0 * g
    sT = A_T * g
    if np.isscalar(g):
        M = xp.empty_like(S)
        M[:] = sT * S
        M[:, 0] += s0
        M[:, 1] += s0
        M[:, 2] += s0
    else:
        M = sT[:, None] * S
        M[:, 0] += s0
        M[:, 1] += s0
        M[:, 2] += s0
    return sym3_expm(M, xp), S


# ------------------------------------------------------- radial reduction
def radial_S_amplitude(S, rhat, xp=np):
    """lambda(r) such that S = lambda (rhat rhat^T - I/3) if that form holds.

    For S of that form, rhat^T S rhat = (2/3) lambda, so lambda = 1.5 * S_rr.
    Returned together with the residual of the assumed form, which is what the
    isotropy gate actually tests.
    """
    Srr = sym3_quad(S, rhat, xp)
    lam = 1.5 * Srr
    # model tensor lam (rhat rhat - I/3)
    rx, ry, rz = rhat[..., 0], rhat[..., 1], rhat[..., 2]
    mod = xp.stack([rx * rx - 1 / 3, ry * ry - 1 / 3, rz * rz - 1 / 3,
                    rx * ry, rx * rz, ry * rz], axis=-1) * lam[..., None]
    num = xp.sqrt(xp.sum((S - mod) ** 2 * xp.array([1., 1., 1., 2., 2., 2.]),
                         axis=-1))
    den = xp.sqrt(xp.sum(S ** 2 * xp.array([1., 1., 1., 2., 2., 2.]), axis=-1))
    return lam, Srr, num, den
