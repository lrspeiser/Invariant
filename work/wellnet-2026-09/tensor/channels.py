"""TENSOR 2 -- pair channels.

    e_ab      = (x_b - x_a)/|x_b - x_a|
    W_ab(x)   = exp[ -d_perp^2 / (2 sigma_perp^2) ] exp[ -d_par^2 / (2 sigma_par^2) ]
    C^ij(x)   = sum_{a<b} w_ab W_ab(x) e_ab^i e_ab^j
    w_ab      = (M_a M_b / M_0^2)^p (d_ab/L)^-q exp[ -(d_ab/L)^s ]
    K(x)      = exp[ -alpha C(x) ]        (sign convention below)


WHAT d_par MEANS -- stated, because the naive choice is wrong
-------------------------------------------------------------
With no d_par factor at all the Gaussian tube is an INFINITE cylinder and every
pair contributes at every distance along its axis, so a cluster's 45,000 pairs
paint the whole box.  Two finite alternatives are implemented:

  mode="clip" (DEFAULT, used for every number in the report)
        t      = (x - m_ab) . e_ab            signed distance from the midpoint
        d_par  = max(0, |t| - d_ab/2)
     The tube is a capsule: flat along the segment that actually joins the two
     wells, Gaussian beyond each end.  This is the object the model is trying
     to describe -- a channel BETWEEN two wells -- and it is scale-correct: a
     pair 2 Mpc apart has a 2 Mpc channel, not a sigma_par-sized blob.

  mode="mid"
        d_par  = |t|
     Gaussian from the midpoint.  A widely separated pair then contributes
     nothing at either of its own wells unless sigma_par > d_ab/2, which makes
     the tensor depend on sigma_par far more strongly than on the geometry.
     Implemented for comparison and reported, not used as the default.

  mode="line"
        d_par  = 0   (infinite cylinder; the failure mode named above)

SIGN CONVENTION -- which sign strengthens gravity along a channel
-----------------------------------------------------------------
C is positive semi-definite (a weighted sum of outer products e e^T with
positive weights), so along a channel direction e, e^T C e > 0.

    K = exp[-alpha C],  alpha > 0   ->  K's eigenvalue along the channel is
        exp(-alpha e^T C e) < 1.  The medium is a POORER conductor of
        gravitational flux along the channel.  Because the field equation
        conserves flux, div[mu K grad Phi] = 4 pi G rho, a poorer conductor
        needs a LARGER gradient to carry the same flux.  So

            alpha > 0  =>  |g| is STRONGER along the channels joining wells.

    K = exp[+alpha C]   ->  better conduction along the channel, smaller |g|
        along it, flux funnelled into the channel.

Both are implemented (`sign=-1` and `sign=+1` multiplying alpha) and the
direction of the effect is verified numerically in the gates rather than
asserted -- see gate `channel_sign` in test_gates_wellnet.py.

UNITS: SI (m, kg, s).
"""
from __future__ import annotations

import numpy as np

from wellnet import G, A0, KPC, MSUN, sym3_expm, get_xp

MODES = {"clip": 0, "mid": 1, "line": 2}

_KERNEL_SRC = r"""
extern "C" __global__
void channel_C(const double* __restrict__ px,
               const double* __restrict__ py,
               const double* __restrict__ pz,
               const long long npt,
               const double* __restrict__ mx,
               const double* __restrict__ my,
               const double* __restrict__ mz,
               const double* __restrict__ ex,
               const double* __restrict__ ey,
               const double* __restrict__ ez,
               const double* __restrict__ half,
               const double* __restrict__ w,
               const int npair,
               const double inv2sp,      /* 1/(2 sigma_perp^2) */
               const double inv2sl,      /* 1/(2 sigma_par^2)  */
               const double cut_perp2,
               const double cut_par2,
               const int mode,
               double* __restrict__ out) /* (6, npt), structure of arrays */
{
    long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= npt) return;
    const double X = px[i], Y = py[i], Z = pz[i];
    double cxx = 0.0, cyy = 0.0, czz = 0.0, cxy = 0.0, cxz = 0.0, cyz = 0.0;
    for (int j = 0; j < npair; ++j) {
        const double dx = X - mx[j], dy = Y - my[j], dz = Z - mz[j];
        const double exj = ex[j], eyj = ey[j], ezj = ez[j];
        const double t = dx * exj + dy * eyj + dz * ezj;
        double perp2 = dx * dx + dy * dy + dz * dz - t * t;
        if (perp2 < 0.0) perp2 = 0.0;
        if (perp2 > cut_perp2) continue;
        double dpar;
        if (mode == 0) {                     /* clipped to the segment */
            const double a = fabs(t) - half[j];
            dpar = a > 0.0 ? a : 0.0;
        } else if (mode == 1) {              /* from the midpoint */
            dpar = fabs(t);
        } else {                             /* infinite line */
            dpar = 0.0;
        }
        const double dpar2 = dpar * dpar;
        if (dpar2 > cut_par2) continue;
        const double g = w[j] * exp(-perp2 * inv2sp - dpar2 * inv2sl);
        cxx += g * exj * exj;
        cyy += g * eyj * eyj;
        czz += g * ezj * ezj;
        cxy += g * exj * eyj;
        cxz += g * exj * ezj;
        cyz += g * eyj * ezj;
    }
    out[0 * npt + i] = cxx;
    out[1 * npt + i] = cyy;
    out[2 * npt + i] = czz;
    out[3 * npt + i] = cxy;
    out[4 * npt + i] = cxz;
    out[5 * npt + i] = cyz;
}
"""

_kernel_cache = {}


def _kernel():
    if "k" not in _kernel_cache:
        import cupy as cp
        _kernel_cache["k"] = cp.RawKernel(_KERNEL_SRC, "channel_C")
    return _kernel_cache["k"]


# ------------------------------------------------------------ pair build
def build_pairs(wx, wm, p=1.0, q=1.0, s=2.0, L=500.0 * KPC,
                M_0=1.0e11 * MSUN, d_min=1.0 * KPC, d_max=None, xp=np):
    """All a<b pairs with their weights and geometry.

    Returns dict with mid (P,3), e (P,3), half (P,), w (P,), d (P,).
    d_max, if given, drops pairs wider than that -- a separate, reported
    approximation from the tube cutoff.
    """
    N = wx.shape[0]
    ia, ib = xp.triu_indices(N, k=1)
    d3 = wx[ib] - wx[ia]
    d = xp.sqrt(xp.sum(d3 * d3, axis=1))
    keep = d > d_min
    if d_max is not None:
        keep &= d <= d_max
    ia, ib, d3, d = ia[keep], ib[keep], d3[keep], d[keep]
    e = d3 / d[:, None]
    mid = 0.5 * (wx[ia] + wx[ib])
    x = d / L
    w = ((wm[ia] * wm[ib]) / M_0 ** 2) ** p * x ** (-q) * xp.exp(-(x ** s))
    return dict(mid=mid, e=e, half=0.5 * d, w=w, d=d, ia=ia, ib=ib)


# ---------------------------------------------------------- the C tensor
def C_tensor(points, pairs, sigma_perp=200.0 * KPC, sigma_par=200.0 * KPC,
             mode="clip", n_sigma=4.0, xp=np, block=256, chunk=1 << 18):
    """C^ij(x) at every point.  (P,6).

    n_sigma sets the tube cutoff: a pair is skipped for a point further than
    n_sigma*sigma from the tube.  The dropped weight per pair is at most
    exp(-n_sigma^2/2) (3.4e-4 at 4 sigma, 1.5e-8 at 6).  The measured cost is
    reported by the gates, not assumed.
    """
    P = points.shape[0]
    cut_perp2 = (n_sigma * sigma_perp) ** 2
    cut_par2 = (n_sigma * sigma_par) ** 2
    inv2sp = 1.0 / (2.0 * sigma_perp ** 2)
    inv2sl = 1.0 / (2.0 * sigma_par ** 2)
    md = MODES[mode]

    if xp is not np:                                  # GPU path
        import cupy as cp
        out = cp.empty((6, P), dtype=cp.float64)
        args = (cp.ascontiguousarray(points[:, 0]),
                cp.ascontiguousarray(points[:, 1]),
                cp.ascontiguousarray(points[:, 2]), np.int64(P),
                cp.ascontiguousarray(pairs["mid"][:, 0]),
                cp.ascontiguousarray(pairs["mid"][:, 1]),
                cp.ascontiguousarray(pairs["mid"][:, 2]),
                cp.ascontiguousarray(pairs["e"][:, 0]),
                cp.ascontiguousarray(pairs["e"][:, 1]),
                cp.ascontiguousarray(pairs["e"][:, 2]),
                cp.ascontiguousarray(pairs["half"]),
                cp.ascontiguousarray(pairs["w"]),
                np.int32(pairs["w"].shape[0]),
                np.float64(inv2sp), np.float64(inv2sl),
                np.float64(cut_perp2), np.float64(cut_par2), np.int32(md),
                out)
        grid = (int((P + block - 1) // block),)
        _kernel()(grid, (block,), args)
        cp.cuda.Stream.null.synchronize()
        return cp.ascontiguousarray(out.T)

    # CPU reference path -- chunked over points, used by the gates to check
    # the kernel and for small grids.
    out = np.zeros((P, 6))
    mid, e, half, w = pairs["mid"], pairs["e"], pairs["half"], pairs["w"]
    for i0 in range(0, P, chunk // max(1, len(w) // 64)):
        i1 = min(P, i0 + max(1, chunk // max(1, len(w) // 64)))
        d = points[i0:i1, None, :] - mid[None, :, :]
        t = np.sum(d * e[None, :, :], axis=-1)
        perp2 = np.maximum(np.sum(d * d, axis=-1) - t ** 2, 0.0)
        if md == 0:
            dpar = np.maximum(np.abs(t) - half[None, :], 0.0)
        elif md == 1:
            dpar = np.abs(t)
        else:
            dpar = np.zeros_like(t)
        g = w[None, :] * np.exp(-perp2 * inv2sp - dpar ** 2 * inv2sl)
        g = np.where((perp2 > cut_perp2) | (dpar ** 2 > cut_par2), 0.0, g)
        ex, ey, ez = e[:, 0], e[:, 1], e[:, 2]
        out[i0:i1, 0] = g @ (ex * ex)
        out[i0:i1, 1] = g @ (ey * ey)
        out[i0:i1, 2] = g @ (ez * ez)
        out[i0:i1, 3] = g @ (ex * ey)
        out[i0:i1, 4] = g @ (ex * ez)
        out[i0:i1, 5] = g @ (ey * ez)
    return out


def K_channels(points, pairs, alpha=1.0, sign=-1, xp=np, **ckw):
    """K = exp[ sign * alpha * C ].  sign=-1 is the brief's exp[-alpha C].

    sign=-1 (alpha>0): weaker conduction along channels -> STRONGER |g| along
    them.  sign=+1: stronger conduction, weaker |g| along them.
    Returns (K (P,6), C (P,6)).
    """
    C = C_tensor(points, pairs, xp=xp, **ckw)
    return sym3_expm(sign * alpha * C, xp), C
