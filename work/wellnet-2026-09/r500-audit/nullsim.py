"""
JOB 1 -- the synthetic null for corr(r/R500, RAR residual).

Builds clusters in which the excess has NO true dependence on scaled radius,
pushes them through the *actual* X-COP publication-and-analysis chain
(noisy n_e and T -> hydrostatic mass -> M500/R500 -> profiles republished in
R500/T500 units -> the bench reads them back), and reports the induced
correlation.

THE CHAIN, AND WHERE R500 CAN ENTER
-----------------------------------
publication :  R500_fit solves  M_HSE(R) = (4/3)pi 500 rho_c(z) R^3
               T500_fit = G M500_fit mu m_p / (2 R500_fit)
               stored:  RW_X = r_coarse / R500_fit ,  T_X = kT / T500_fit
analysis    :  kT_recovered = T_X * T500(header) at r = RW_X * R500(header)

Because the header R500 IS the R500 used to scale, the round trip is exact and
R500 cancels identically from the numerator (verified in tests.py).  The
surviving channel is *estimator-level*: R500 is a monotone function of the same
hydrostatic mass whose excess is on the y-axis, so an upward mass fluctuation
raises y and raises R500 (lowering r/R500) at the same time.  That channel is
sign-definite negative and is exactly what this simulation measures.

Declared before any residual was looked at (2026-09-04):
  S1  Spearman corr(r/R_norm, y) pooled over the surviving X-COP points
  S2  rms of y about a pooled quadratic in log10(r/R_norm)   ("collapse")
  S3  slope dy/dlog10(r/R_norm) beyond 0.25 R500
"""
from __future__ import annotations
import math

import numpy as np

import ingest as I

G, KPC, MSUN, MP, MU, MU_E, A0 = I.G, I.KPC, I.MSUN, I.MP, I.MU, I.MU_E, I.A0
R_MIN, R_MAX = I.R_MIN_KPC * KPC, I.R_MAX_KPC * KPC


# ---------------------------------------------------------------- statistics
def rank(a):
    a = np.asarray(a, float)
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    sm = np.zeros(len(cnt))
    np.add.at(sm, inv, r)
    return (sm / cnt)[inv]


def pear(u, w):
    u = np.asarray(u, float) - np.mean(u)
    w = np.asarray(w, float) - np.mean(w)
    d = math.sqrt(float(u @ u) * float(w @ w))
    return float(u @ w / d) if d > 0 else 0.0


def spear(u, w):
    return pear(rank(u), rank(w))


def collapse_rms(t, y, deg=2):
    """S2: rms of y about a pooled polynomial in log10 t. Lower = tighter
    self-similar collapse."""
    lt = np.log10(t)
    c = np.polyfit(lt, y, deg)
    return float(np.std(y - np.polyval(c, lt)))


def slope_beyond(t, y, tmin=0.25):
    """S3: dy/dlog10 t, points with t > tmin."""
    m = t > tmin
    if m.sum() < 10:
        return float("nan")
    c = np.polyfit(np.log10(t[m]), y[m], 1)
    return float(c[0])


# ---------------------------------------------------------------- templates
class Template:
    """Everything needed to regenerate one cluster."""

    def __init__(self, c):
        self.name = c["name"]
        self.z = c["z"]
        self.R500_pub = c["R500_hse"]
        self.M500_pub = c["M500_hse"]
        self.r = 0.5 * (c["r_in_kpc"] + c["r_out_kpc"]) * KPC
        self.ne = c["ne_cm3"] * 1e6                      # m^-3
        self.sig_lnne = np.clip(0.5 * (c["ne_hi"] - c["ne_lo"]) / c["ne_cm3"],
                                1e-3, 1.0)
        self.r_coarse = c["rw_x"] * c["R500_hse"]
        self.sig_lnT = np.clip(c["et_x"] / c["t_x"], 1e-3, 1.0)
        p = I.build_profile(c)
        self.kT_obs = p["kT"]                            # reconstructed physical kT
        self.Mstar = p["Mstar"]
        self.gb_obs = p["gb"]
        self.go_obs = p["go"]
        self.rhoc = I.rho_c(c["z"])
        # boundary pressure for the HSE integration, at the outer edge of the
        # MEASURED temperature grid (beyond it the published T is clamped)
        self.r_bnd = self.r_coarse.max()

    def mgas(self, ne):
        rho = MU_E * ne * MP
        r = self.r
        return (4 / 3 * np.pi * r[0] ** 3 * rho[0]
                + np.concatenate([[0.], np.cumsum(4 * np.pi * rho[:-1] * r[:-1] ** 2
                                                  * np.diff(r))]))

    def gbar(self, ne):
        return G * (self.mgas(ne) + self.Mstar) / self.r ** 2


# ---------------------------------------------------------------- truth
def _loglog(r, M, R):
    """log-log interpolation of a mass profile. Used for BOTH the anchor and the
    overdensity crossing -- mixing np.interp with a log-linear crossing made the
    fixed-point iteration in the first version of make_truth neutrally stable and
    it drifted by up to 5.8% over 60 iterations (see REPORT.md, bug 1)."""
    return float(np.exp(np.interp(np.log(R), np.log(r), np.log(M))))


def _cross(r, M, A):
    """largest r with M(r) = A r^3, log-log interpolated."""
    f = np.log(M) - np.log(A * r ** 3)
    idx = np.where(np.diff(np.sign(f)) < 0)[0]
    if not len(idx):
        return float("nan")
    i = idx[-1]
    lr = np.log(r[i:i + 2])
    return float(np.exp(lr[0] + (lr[1] - lr[0]) * (-f[i]) / (f[i + 1] - f[i])))


def make_truth(T: Template, s_scaled=0.0, s_abs=0.0, shape_amp=0.0, rng=None):
    """True mass profile whose log-excess over the RAR is

        y_true(r) = a + s_scaled*log10(r/R500_true) + s_abs*log10(r/500 kpc)

    R500_true is DEFINED to be the published R500 and `a` is solved for so the
    overdensity condition holds there exactly.  That anchors the simulated
    population to the real one instead of inventing a mass scale, and it is
    non-iterative, so it cannot drift.

    Under the primary null s_scaled = s_abs = 0: the excess is a per-cluster
    constant with NO radial structure of any kind.
    """
    r = T.r
    gb = T.gbar(T.ne)
    Mrar = gb * I.nu_rar(gb / A0) * r ** 2 / G
    R5 = T.R500_pub
    A = (4 / 3) * np.pi * 500 * T.rhoc
    sh = s_scaled * np.log10(r / R5) + s_abs * np.log10(r / (500 * KPC))
    if shape_amp > 0.0 and rng is not None:
        # cluster-to-cluster diversity in the SHAPE of the excess: a random
        # low-order polynomial in log r, standardised so its amplitude is
        # interpretable in dex.  Real clusters differ in non-thermal pressure,
        # substructure and asphericity; without this the simulated collapse is
        # unrealistically tight and the discriminator's error bar is a fiction.
        u = np.log10(r / R5)
        u = (u - u.mean()) / max(u.std(), 1e-9)
        c1, c2 = rng.standard_normal(2)
        q = u ** 2
        q = (q - q.mean()) / max(q.std(), 1e-9)
        sh = sh + shape_amp * (c1 * u + c2 * q) / math.sqrt(2.0)
    base = Mrar * 10 ** sh
    a = math.log10(A * R5 ** 3 / _loglog(r, base, R5))
    y_true = a + sh
    M = base * 10 ** a
    go = G * M / r ** 2
    # temperature from hydrostatic equilibrium:  dP/dr = -mu m_p n_e g
    # integrate inward from the outer edge of the measured T grid
    P = np.zeros_like(r)
    integ = MU * MP * T.ne * go
    cum = np.concatenate([[0.], np.cumsum(0.5 * (integ[1:] + integ[:-1]) * np.diff(r))])
    Pb = np.interp(T.r_bnd, r, T.ne * T.kT_obs)
    cb = np.interp(T.r_bnd, r, cum)
    P = Pb + (cb - cum)
    kT = P / T.ne
    # the outward integration can graze zero at the last bin or two; floor it so
    # log(kT) is defined (those bins sit beyond the measured T grid anyway)
    kT = np.maximum(kT, 1e-4 * np.nanmax(kT))
    return dict(y_true=y_true, go=go, gb=gb, kT=kT, R500_true=R5, a=a)


# ---------------------------------------------------------------- observation
def _corr_noise(rng, n, sig, rho_corr, ell):
    """noise with a white part and a bin-correlated part (X-COP's L1
    deprojection correlates neighbouring bins)."""
    w = rng.standard_normal(n)
    if rho_corr <= 0:
        return sig * w
    k = np.arange(-3 * ell, 3 * ell + 1)
    ker = np.exp(-0.5 * (k / max(ell, 1e-6)) ** 2)
    ker /= np.sqrt((ker ** 2).sum())
    s = np.convolve(rng.standard_normal(n + len(ker)), ker, mode="same")[:n]
    return sig * (math.sqrt(1 - rho_corr) * w + math.sqrt(rho_corr) * s)


def observe(T, truth, rng, cfg):
    ne = T.ne * np.exp(_corr_noise(rng, len(T.ne), T.sig_lnne * cfg["ne_scale"],
                                   cfg["rho_corr"], cfg["ell"]))
    kT_c = np.interp(T.r_coarse, T.r, truth["kT"])
    kT_c = kT_c * np.exp(_corr_noise(rng, len(kT_c), T.sig_lnT * cfg["T_scale"],
                                     cfg["rho_corr"], 1.0))
    kT_c = kT_c * math.exp(rng.normal(0, cfg["T_calib"]))       # per-cluster calib
    return ne, kT_c


def hse_g(T, ne, kT_fine):
    lr = np.log(T.r)
    kT_fine = np.maximum(kT_fine, 1e-12 * np.nanmax(kT_fine))
    ne = np.maximum(ne, 1e-12 * np.nanmax(ne))
    return -(kT_fine / (MU * MP)) * (np.gradient(np.log(ne), lr)
                                     + np.gradient(np.log(kT_fine), lr)) / T.r


def infer_R500(T, ne, kT_c, mode="fit"):
    """The publication step: M500/R500 from the observed hydrostatic profile.
    `fit` smooths log M vs log r (a stand-in for X-COP's backward NFW fit);
    `raw` takes the pointwise crossing.
    """
    kT = np.interp(T.r, T.r_coarse, kT_c)
    go = hse_g(T, ne, kT)
    M = go * T.r ** 2 / G
    ok = np.isfinite(M) & (M > 0) & (T.r > 0.3 * T.R500_pub) & (T.r < 1.6 * T.R500_pub)
    if ok.sum() < 6:
        return np.nan
    lr, lM = np.log(T.r[ok]), np.log(M[ok])
    if mode == "fit":
        c = np.polyfit(lr, lM, 2)
        f = lambda x: np.exp(np.polyval(c, np.log(x)))
    else:
        f = lambda x: np.exp(np.interp(np.log(x), lr, lM))
    A = (4 / 3) * np.pi * 500 * T.rhoc
    lo, hi = 0.3 * T.R500_pub, 2.0 * T.R500_pub
    g = lambda R: f(R) - A * R ** 3
    if g(lo) * g(hi) > 0:
        return np.nan
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return math.sqrt(lo * hi)


def analyse(T, ne, kT_c, R500_hdr):
    """The bench, exactly: republish in R500 units, read back, compute y.

    kT_recovered = T_X * T500(hdr) at r = RW_X * R500_hdr, which by construction
    equals kT_c at r_coarse -- the cancellation.  We still write it out in full
    so the round trip is exercised rather than assumed.
    """
    M500 = (4 / 3) * np.pi * 500 * T.rhoc * R500_hdr ** 3
    T500 = G * M500 * MU * MP / (2 * R500_hdr)
    RW_X = T.r_coarse / R500_hdr
    T_X = kT_c / T500
    kT = np.interp(T.r, RW_X * R500_hdr, T_X * T500)
    go = hse_g(T, ne, kT)
    gb = T.gbar(ne)
    return go, gb


def one_realisation(TS, rng, cfg, s_scaled=0.0, s_abs=0.0):
    shape_amp = cfg.get("shape_amp", 0.0)
    out = dict(r=[], y=[], t_hse=[], t_true=[], t_scr=[], name=[],
               R500_obs={}, R500_true={}, y_true=[])
    R5o, R5t = {}, {}
    per = []
    for T in TS:
        truth = make_truth(T, s_scaled=s_scaled, s_abs=s_abs,
                           shape_amp=shape_amp, rng=rng)
        ne, kT_c = observe(T, truth, rng, cfg)
        R5 = infer_R500(T, ne, kT_c, mode=cfg["R500_mode"])
        if not np.isfinite(R5):
            R5 = T.R500_pub
        if cfg["R500_extra_frac"] > 0:
            R5 = R5 * math.exp(rng.normal(0, cfg["R500_extra_frac"]))
        go, gb = analyse(T, ne, kT_c, R5)
        m = (T.r > R_MIN) & (T.r < R_MAX) & (go > 0) & (gb > 0) & np.isfinite(go) & np.isfinite(gb)
        if m.sum() < 5:
            continue
        y = I.rar_residual(gb[m], go[m])
        good = np.isfinite(y)
        per.append((T.name, T.r[m][good], y[good], R5, truth["R500_true"],
                    truth["y_true"][m][good]))
        R5o[T.name] = R5
        R5t[T.name] = truth["R500_true"]
    if not per:
        return None
    names = np.concatenate([[p[0]] * len(p[1]) for p in per])
    r = np.concatenate([p[1] for p in per])
    y = np.concatenate([p[2] for p in per])
    t_hse = np.concatenate([np.full(len(p[1]), p[3]) for p in per])
    t_tru = np.concatenate([np.full(len(p[1]), p[4]) for p in per])
    y_true = np.concatenate([p[5] for p in per])
    # scrambled R500: permute the inferred R500 across clusters
    order = list(R5o)
    perm = rng.permutation(len(order))
    smap = {order[i]: R5o[order[perm[i]]] for i in range(len(order))}
    t_scr = np.array([smap[n] for n in names])
    return dict(name=names, r=r, y=y, y_true=y_true,
                R500_obs=t_hse, R500_true=t_tru, R500_scr=t_scr,
                R500_obs_map=R5o, R500_true_map=R5t)


def stats_of(res):
    r, y = res["r"], res["y"]
    d = dict(
        n=int(len(r)),
        S1_hse=spear(r / res["R500_obs"], y),
        S1_phys=spear(r, y),
        S1_true=spear(r / res["R500_true"], y),
        S1_scr=spear(r / res["R500_scr"], y),
        S1_truth_injected=spear(r / res["R500_true"], res["y_true"]),
        S2_hse=collapse_rms(r / res["R500_obs"], y),
        S2_phys=collapse_rms(r / (1000 * KPC), y),
        S2_scr=collapse_rms(r / res["R500_scr"], y),
        S3_hse=slope_beyond(r / res["R500_obs"], y),
    )
    return d


DEFAULT_CFG = dict(ne_scale=1.0, T_scale=1.0, T_calib=0.03, rho_corr=0.7,
                   ell=2.0, R500_mode="fit", R500_extra_frac=0.0, shape_amp=0.0)


def run(TS, n_real=400, seed=0, cfg=None, s_scaled=0.0, s_abs=0.0):
    cfg = dict(DEFAULT_CFG, **(cfg or {}))
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_real):
        res = one_realisation(TS, rng, cfg, s_scaled=s_scaled, s_abs=s_abs)
        if res is None:
            continue
        rows.append(stats_of(res))
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows], float) for k in keys}
