"""analysis.py -- the BLIND analysis pipeline.

It receives a Corpus and is never told which universe made it.  Everything it
uses is detector-facing: velocity maps, ellipticities, counts, redshifts,
image positions, durations.  It builds its own baryon model from the observed
photometry, its own rotation curves from the velocity fields, its own masses
from the shear, and its own residuals.

Structure

  1  extract per-object quantities from the raw detector data
  2  CROSS-FIT the scalar nuisance: fit a flexible scalar response nu-hat on
     one half of the galaxies, freeze it, predict the other half, and freeze
     it again to predict the CLUSTERS.  Every cluster residual below is a
     frozen cross-channel prediction, per charter Stage 8.
  3  compute the feature vector (the universe-agnostic summary the
     equivalence-class discriminator sees)
  4  compute the NAMED DETECTORS -- anisotropy, network, memory, EP-slip,
     path -- each with its own axis scan and look-elsewhere handling

The features are grouped into named CHANNELS so the equivalence map can be
recomputed channel by channel and answer "which missing observation would
separate them?" quantitatively.
"""
from __future__ import annotations

import zlib

import numpy as np

from .baryons import A0, C_KMS, G, disk_vc2, hernquist_M
from .physics import PHI0_ENV, comoving_Mpc, sigma_crit
from .scenes import LAM_NET, S0_NET

RAD_AS = np.pi / 180.0 / 3600.0


def _wls(y, X, w=None):
    X = np.atleast_2d(X)
    if X.shape[0] != len(y):
        X = X.T
    w = np.ones_like(y) if w is None else w
    A = X * w[:, None]
    try:
        return np.linalg.lstsq(A.T @ X, A.T @ y, rcond=None)[0]
    except Exception:
        return np.zeros(X.shape[1])


def _slope(x, y, w=None):
    """Weighted OLS slope of y on x.  Quote SLOPES, not correlations."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return 0.0
    x, y = x[m], y[m]
    w = np.ones_like(x) if w is None else np.asarray(w)[m]
    xb = np.average(x, weights=w)
    yb = np.average(y, weights=w)
    v = np.average((x - xb) ** 2, weights=w)
    if v <= 0:
        return 0.0
    return float(np.average((x - xb) * (y - yb), weights=w) / v)


# ====================================================================== galaxies
def galaxy_curve(gd):
    """Rotation curve from the velocity FIELD, using the OBSERVED geometry."""
    d = gd["dist_obs"] * 1e3
    ax = gd["ax_arcsec"] * d * RAD_AS                 # kpc under the assumed d
    X, Y = np.meshgrid(ax, ax, indexing="ij")
    pa = np.deg2rad(gd["pa_obs"])
    xr = X * np.cos(pa) + Y * np.sin(pa)
    yr = -X * np.sin(pa) + Y * np.cos(pa)
    inc = np.deg2rad(gd["incl_obs"])
    yd = yr / max(np.cos(inc), 1e-3)
    R = np.sqrt(xr ** 2 + yd ** 2) + 1e-9
    th = np.arctan2(yd, xr)
    m = gd["mask"] & (np.abs(np.cos(th)) > 0.32)
    if m.sum() < 25:
        return None
    Rf, tf = R[m], th[m]
    vf, ef = gd["v_map"][m], gd["v_err"][m]
    Rd = gd["Rd_obs"]
    edges = np.linspace(0.5 * Rd, 5.0 * Rd, 9)
    ctr, vR, verr = [], [], []
    basis = np.cos(tf) * np.sin(inc)
    w = 1.0 / ef ** 2
    for a, b in zip(edges[:-1], edges[1:]):
        s = (Rf >= a) & (Rf < b)
        if s.sum() < 6:
            continue
        num = np.sum(w[s] * vf[s] * basis[s])
        den = np.sum(w[s] * basis[s] ** 2)
        if den <= 0:
            continue
        v = num / den
        if v <= 5.0:
            continue
        ctr.append(0.5 * (a + b)); vR.append(v)
        verr.append(1.0 / np.sqrt(den))
    if len(ctr) < 4:
        return None
    ctr = np.array(ctr); vR = np.array(vR); verr = np.array(verr)

    # baryon model from the OBSERVED photometry, with the M/L gradient
    grad = 10 ** (gd["ml_grad_obs"] * (ctr / Rd - 1.0))
    v2 = (disk_vc2(ctr, gd["Md_obs"], Rd) * grad
          + disk_vc2(ctr, gd["Mg_obs"], gd["Rg_obs"])
          + G * hernquist_M(ctr, gd["Mb_obs"], gd["ab_obs"]) / ctr)
    gbar = np.maximum(v2, 1e-8) / ctr
    gobs = vR ** 2 / ctr

    # residual m=2 in the velocity field, after removing the fitted curve
    vmod = np.interp(Rf, ctr, vR) * basis
    res = (vf - vmod) / np.maximum(ef, 1.0)
    sel = (Rf > 1.0 * Rd) & (Rf < 5.0 * Rd)
    if sel.sum() > 30:
        # an m=2 modulation of v_c(R,phi) appears in v_los = v_c cos(phi) sin(i)
        # as m=1 and m=3, NOT m=2:  cos(phi) cos(2(phi-psi)) =
        #   1/2 [cos(3 phi - 2 psi) + cos(phi - 2 psi)].
        # The m=1 part is degenerate with the fitted rotation curve, so the
        # clean directional signature is the m=3 harmonic, whose phase is 2 psi.
        c2 = np.mean(res[sel] * np.cos(3 * tf[sel]))
        s2 = np.mean(res[sel] * np.sin(3 * tf[sel]))
    else:
        c2 = s2 = 0.0

    Sig = (gd["Md_obs"] / (2 * np.pi * Rd ** 2) * np.exp(-gd["Rv"] / Rd)
           + gd["Mg_obs"] / (2 * np.pi * gd["Rg_obs"] ** 2) * np.exp(-gd["Rv"] / gd["Rg_obs"]))
    gz_bar = 2 * np.pi * G * Sig
    gz_obs = gd["sz_obs"] ** 2 / max(gd["hz_obs"], 1e-3)
    gR_at = np.interp(gd["Rv"], ctr, gobs)
    gRb_at = np.interp(gd["Rv"], ctr, gbar)

    # gauge-safe potential depth from the observed baryons (primary rule)
    rgp = np.geomspace(0.1 * Rd, 10.0 * Rd, 128)
    gp = (disk_vc2(rgp, gd["Md_obs"], Rd) + disk_vc2(rgp, gd["Mg_obs"], gd["Rg_obs"])
          + G * hernquist_M(rgp, gd["Mb_obs"], gd["ab_obs"]) / rgp) / rgp
    seg = 0.5 * (gp[1:] + gp[:-1]) * np.diff(rgp)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    phi_dep = float(np.interp(2.0 * Rd, rgp, cum[-1] - cum))

    return {"R": ctr, "vR": vR, "verr": verr, "gbar": gbar, "gobs": gobs,
            "c2": float(c2), "s2": float(s2),
            "gz_obs": gz_obs, "gz_bar": np.maximum(gz_bar, 1e-10),
            "gR_at": gR_at, "gRb_at": np.maximum(gRb_at, 1e-10),
            "Mbar": gd["Md_obs"] + gd["Mg_obs"] + gd["Mb_obs"],
            "vflat": float(np.median(vR[-3:])),
            "S_ext": gd["S_ext_obs"], "axis_ext": gd["axis_ext_obs"],
            "pa": gd["pa_obs"], "phi_dep": phi_dep,
            "t_merge": gd["t_merge_proxy"], "void": gd["void_frac_obs"],
            "incl": gd["incl_obs"],
            "outer_slope": float(np.polyfit(np.log(ctr[-4:]), np.log(vR[-4:]), 1)[0]),
            }


# --------------------------------------------------- the frozen scalar model
class FrozenScalar:
    """nu-hat(g_bar): a flexible smooth scalar response, fitted then FROZEN.

    Fitted on a declared half of the galaxies, evaluated everywhere else.
    Deliberately more flexible than any of the injected scalar families, so a
    directional or network detection cannot be manufactured by a poor scalar
    interpolating function (the cross-fit the brief demands).
    """
    KNOTS = np.linspace(-13.0, -7.5, 7)

    def __init__(self, lg, ly, w=None):
        B = self._basis(lg)
        w = np.ones_like(ly) if w is None else w
        A = B * w[:, None]
        lam = 1e-3 * np.eye(B.shape[1])
        self.c = np.linalg.solve(B.T @ A + lam, A.T @ ly)

    @classmethod
    def _basis(cls, lg):
        lg = np.clip(np.asarray(lg, float), cls.KNOTS[0], cls.KNOTS[-1])
        h = cls.KNOTS[1] - cls.KNOTS[0]
        u = (lg[:, None] - cls.KNOTS[None, :]) / h
        return np.maximum(0.0, 1.0 - np.abs(u))

    def __call__(self, gbar):
        gb = np.maximum(np.asarray(gbar, float), 1e-16)
        # gbar arrives in (km/s)^2/kpc; the knots are in log10 SI
        lg = np.log10(gb / 3.0856775814913673e13)
        return 10 ** (self._basis(lg) @ self.c)


# ====================================================================== clusters
def cluster_reduce(cd, nu_hat):
    """Everything the analysis can make from one cluster's detector data."""
    R5 = cd["R500"]
    rr = np.sqrt(cd["src_x"] ** 2 + cd["src_y"] ** 2)
    ph_ = np.arctan2(cd["src_y"], cd["src_x"])
    et = -(cd["e1"] * np.cos(2 * ph_) + cd["e2"] * np.sin(2 * ph_))
    Scr = sigma_crit(cd["z"], np.maximum(cd["z_src_phot"], cd["z"] + 0.05))
    good = np.isfinite(Scr) & (cd["z_src_phot"] > cd["z"] + 0.1)
    w = cd["w"] * good

    edges = np.array([0.15, 0.35, 0.65, 1.0, 1.5, 2.3]) * R5
    Rb, dS, dSe = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = (rr >= a) & (rr < b) & good
        if s.sum() < 25:
            continue
        v = et[s] * Scr[s]
        Rb.append(np.sqrt(a * b)); dS.append(np.average(v, weights=w[s]))
        dSe.append(np.std(v) / np.sqrt(s.sum()))
    Rb = np.array(Rb); dS = np.array(dS); dSe = np.array(dSe)

    # baryon model from the OBSERVED gas + stellar masses
    Mb_ann = cd["Mgas_obs"] + cd["Mstar_obs"]
    Mb_at = lambda r: np.interp(np.asarray(r, float), cd["r_ann"], Mb_ann)
    gbar_at = lambda r: G * Mb_at(r) / np.asarray(r, float) ** 2

    # ---- lensing "mass" the analysis derives, and the frozen prediction ----
    from .corpus import projected_mass_2d
    dS_bar = []
    for R in Rb:
        rl = np.geomspace(max(0.05 * R5, R * 0.02), 6 * R5, 90)
        gl = gbar_at(rl)
        M2 = projected_mass_2d(rl, gl, np.array([0.97 * R, R, 1.03 * R]))
        Sbar = M2[1] / (np.pi * R ** 2)
        Sig = (M2[2] - M2[0]) / (2 * np.pi * R * 0.06 * R)
        dS_bar.append(Sbar - Sig)
    dS_bar = np.maximum(np.array(dS_bar), 1e-6)
    # frozen scalar prediction for the same radii
    nu_pred = nu_hat(gbar_at(Rb))

    # ---- member dynamics --------------------------------------------------
    Rm = np.sqrt(cd["mem_x"] ** 2 + cd["mem_y"] ** 2)
    ok = cd["mem_p"] > 0.6
    sig_bins, sig_R = [], []
    for a, b in zip([0.1, 0.4, 0.9], [0.4, 0.9, 1.8]):
        s = ok & (Rm >= a * R5) & (Rm < b * R5)
        if s.sum() < 10:
            continue
        v = cd["mem_v"][s]
        v = v[np.abs(v - np.median(v)) < 3.2 * (np.std(v) + 1)]
        if len(v) < 8:
            continue
        sig_bins.append(np.sqrt(max(np.var(v) - 30.0 ** 2, 1e2)))
        sig_R.append(np.sqrt(a * b) * R5)
    sig_bins = np.array(sig_bins); sig_R = np.array(sig_R)
    # declared dynamical estimator: M_dyn = 3 sigma^2 R / G  (isotropic, fixed)
    Mdyn = 3.0 * sig_bins ** 2 * sig_R / G if len(sig_R) else np.array([])

    # ---- X-ray hydrostatic ------------------------------------------------
    ra = cd["r_ann"]
    lnT = np.log(np.maximum(cd["kT_obs"], 1e-3))
    dlnT = np.gradient(lnT, np.log(ra))
    # counts ~ n_e^2 sqrt(kT) r^3  (a volume-integrated emission measure), so
    #   ln n_e = (ln counts - 0.5 ln kT - 3 ln r)/2.
    # Taking ln(counts)/2 as ln n_e leaves a spurious +1.5 in the slope.
    lnne = 0.5 * (np.log(np.maximum(cd["xray_counts"], 1.0))
                  - 0.5 * lnT - 3.0 * np.log(ra))
    dlnn = np.gradient(lnne, np.log(ra))
    kT_kms2 = cd["kT_obs"] * 1.5967e5
    Mhe = -kT_kms2 * ra / G * (dlnT + dlnn)
    Mhe = np.maximum(Mhe, 1e10)

    # ---- quadrupole of the shear field, with the axis scan ----------------
    sel = good & (rr > 0.2 * R5) & (rr < 2.2 * R5)
    q_amp = q_ext = q_bar = q_max = 0.0
    if sel.sum() > 200:
        etn = et[sel] - np.interp(rr[sel], Rb, dS / np.maximum(Scr[sel], 1e-6).mean())
        c2 = 2 * np.mean(etn * np.cos(2 * ph_[sel]))
        s2 = 2 * np.mean(etn * np.sin(2 * ph_[sel]))
        q_amp = float(np.hypot(c2, s2))
        pa_q = 0.5 * np.arctan2(s2, c2)
        q_ext = float(np.cos(2 * (pa_q - np.deg2rad(cd["axis_ext_obs"]))) * q_amp)
        q_bar = float(np.cos(2 * (pa_q - np.deg2rad(cd["pa_bar_obs"]))) * q_amp)
        # explicit axis SCAN (the look-elsewhere the misspecified-axis lesson demands)
        axes = np.linspace(0, np.pi, 36, endpoint=False)
        q_max = float(np.max([abs(np.mean(2 * etn * np.cos(2 * (ph_[sel] - a))))
                              for a in axes]))

    # ---- network: shear residual vs the member-derived well-strength map ---
    net_r = net_ctrl = 0.0
    if sel.sum() > 200:
        mp = np.stack([cd["mem_x"], cd["mem_y"]], 1)
        mm = cd["mem_m_obs"]
        sx, sy = cd["src_x"][sel], cd["src_y"][sel]
        d2 = (sx[:, None] - mp[None, :, 0]) ** 2 + (sy[:, None] - mp[None, :, 1]) ** 2
        S = (G * mm[None, :] / (d2 + LAM_NET ** 2)).sum(1)
        resid = et[sel] - np.interp(rr[sel], Rb, dS / np.maximum(Scr[sel], 1e-6).mean())
        Sr = S - _radial_trend(rr[sel], S)
        net_r = _slope(Sr / max(np.std(Sr), 1e-30), resid)
        rngc = np.random.default_rng(
            zlib.crc32(str(cd["name"]).encode()) & 0x7FFFFFFF)
        thp = rngc.uniform(0, 2 * np.pi, len(mm))
        rmp = np.hypot(mp[:, 0], mp[:, 1])
        mp2 = np.stack([rmp * np.cos(thp), rmp * np.sin(thp)], 1)
        d2c = (sx[:, None] - mp2[None, :, 0]) ** 2 + (sy[:, None] - mp2[None, :, 1]) ** 2
        Sc = (G * mm[None, :] / (d2c + LAM_NET ** 2)).sum(1)
        Sc = Sc - _radial_trend(rr[sel], Sc)
        net_ctrl = _slope(Sc / max(np.std(Sc), 1e-30), resid)

    # ---- gauge-safe potential depth from the observed baryons -------------
    rgp = np.geomspace(0.05 * R5, 3.0 * R5, 128)
    gp = gbar_at(rgp)
    seg = 0.5 * (gp[1:] + gp[:-1]) * np.diff(rgp)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    phi_dep = float(np.interp(0.7 * R5, rgp, cum[-1] - cum))

    return {"Rb": Rb, "dS": dS, "dSe": dSe, "dS_bar": dS_bar, "nu_pred": nu_pred,
            "gbar_Rb": gbar_at(Rb) if len(Rb) else np.array([]),
            "sig_R": sig_R, "Mdyn": Mdyn, "Mb_dyn": Mb_at(sig_R) if len(sig_R) else np.array([]),
            "gbar_dyn": gbar_at(sig_R) if len(sig_R) else np.array([]),
            "r_ann": ra, "Mhe": Mhe, "Mb_ann": Mb_ann, "gbar_ann": gbar_at(ra),
            "q_amp": q_amp, "q_ext": q_ext, "q_bar": q_bar, "q_max": q_max,
            "net_r": float(net_r), "net_ctrl": float(net_ctrl),
            "phi_dep": phi_dep, "R500": R5, "z": cd["z"],
            "t_merge": cd["t_merge_proxy"], "offset": cd["gas_gal_offset_obs"],
            "void": cd["void_frac_obs"], "ell": cd["ell_bar_obs"],
            "thetaE": cd["thetaE_kpc"] / R5, "n_sl": len(cd["sl_delays"]),
            "delay": float(np.median(np.abs(cd["sl_delays"]))) if len(cd["sl_delays"]) else 0.0,
            }


def _radial_trend(r, y, nb=8):
    e = np.quantile(r, np.linspace(0, 1, nb + 1))
    out = np.zeros_like(y)
    for a, b in zip(e[:-1], e[1:]):
        s = (r >= a) & (r <= b)
        if s.sum() > 2:
            out[s] = np.mean(y[s])
    return out


# ====================================================================== driver
CHANNELS = ("gal_rc", "gal_vert", "gal_env", "gal_aniso", "clu_wl", "clu_quad",
            "clu_net", "clu_dyn", "clu_xray", "clu_ep", "clu_mem", "clu_sl", "sn")


def analyse(C, split_seed=0):
    g = [galaxy_curve(gd) for gd in C.gal]
    g = [x for x in g if x is not None]
    if len(g) < 10:
        return None

    lg = np.concatenate([np.log10(x["gbar"] / 3.0856775814913673e13) for x in g])
    ly = np.concatenate([np.log10(np.maximum(x["gobs"] / x["gbar"], 1e-3)) for x in g])
    gid = np.concatenate([[i] * len(x["gbar"]) for i, x in enumerate(g)])
    rs = np.random.default_rng(split_seed)
    fold = rs.permutation(len(g)) % 2
    trn = fold[gid] == 0
    nu_hat = FrozenScalar(lg[trn], ly[trn])
    # the frozen model, evaluated OUT OF FOLD
    resid_out = ly[~trn] - np.log10(nu_hat(10 ** lg[~trn] * 3.0856775814913673e13))

    F, ch = {}, {}

    def put(name, chan, val):
        v = float(val) if np.isfinite(val) else 0.0
        # a hard clip: no single unstable ratio may dominate the discriminant
        F[name] = float(np.clip(v, -30.0, 30.0))
        ch[name] = chan

    # ---------------- gal_rc ------------------------------------------
    be = np.array([-12.5, -11.5, -10.8, -10.2, -9.4, -8.2])
    for k in range(len(be) - 1):
        s = (lg >= be[k]) & (lg < be[k + 1])
        put(f"rar_b{k}", "gal_rc", np.mean(ly[s]) if s.sum() > 8 else 0.0)
    put("rar_scatter", "gal_rc", np.std(resid_out) if len(resid_out) > 8 else 0.0)
    put("btfr_slope", "gal_rc", _slope(np.log10([x["Mbar"] for x in g]),
                                       np.log10([max(x["vflat"], 5) for x in g])))
    put("btfr_scat", "gal_rc", np.std(np.log10([max(x["vflat"], 5) for x in g])
                                      - 0.25 * np.log10([x["Mbar"] for x in g])))
    put("outer_slope", "gal_rc", np.median([x["outer_slope"] for x in g]))

    # ---------------- gal_vert ----------------------------------------
    dz = np.concatenate([np.log10(np.maximum(x["gz_obs"], 1e-12) / x["gz_bar"]) for x in g])
    dr = np.concatenate([np.log10(np.maximum(x["gR_at"], 1e-12) / x["gRb_at"]) for x in g])
    put("vert_mean", "gal_vert", np.median(dz))
    put("vert_minus_rad", "gal_vert", np.median(dz - dr))
    put("vert_scatter", "gal_vert", np.std(dz - dr))
    inc2 = np.repeat(np.array([x["incl"] for x in g]), 2)
    put("vert_vs_incl", "gal_vert", _slope(np.cos(np.deg2rad(inc2)), dz - dr))
    put("vert_p2_incl", "gal_vert",
        _slope(0.5 * (3 * np.cos(np.deg2rad(inc2)) ** 2 - 1), dz - dr))

    # ---------------- gal_env -----------------------------------------
    rg_ = np.array([np.mean(np.log10(np.maximum(x["gobs"] / x["gbar"], 1e-3))
                            - np.log10(nu_hat(x["gbar"]))) for x in g])
    put("env_S", "gal_env", _slope(np.log10([x["S_ext"] for x in g]), rg_))
    put("env_phi", "gal_env", _slope(np.log10([max(x["phi_dep"], 1) for x in g]), rg_))
    put("env_void", "gal_env", _slope(np.array([x["void"] for x in g]), rg_))
    put("env_tmerge", "gal_env", _slope(np.log10([x["t_merge"] for x in g]), rg_))
    put("env_resid_rms", "gal_env", np.std(rg_))

    # ---------------- gal_aniso ---------------------------------------
    psi = np.deg2rad(np.array([x["axis_ext"] - x["pa"] for x in g]))
    c2 = np.array([x["c2"] for x in g]); s2 = np.array([x["s2"] for x in g])
    put("gq_amp", "gal_aniso", np.mean(np.hypot(c2, s2)))
    put("gq_ext", "gal_aniso", np.mean(c2 * np.cos(2 * psi) + s2 * np.sin(2 * psi)))
    put("gq_disk", "gal_aniso", np.mean(c2))
    # misaligned control: the same statistic on an axis rotated by 45 deg.
    # A misspecified axis is a NULL detector -- this pair sizes that directly.
    put("gq_ext45", "gal_aniso",
        np.mean(c2 * np.cos(2 * psi + np.pi / 2) + s2 * np.sin(2 * psi + np.pi / 2)))

    # ---------------- clusters ----------------------------------------
    cl = [cluster_reduce(cd, nu_hat) for cd in C.clu]
    cl = [c for c in cl if len(c["Rb"]) >= 3]
    if len(cl) < 3:
        return None

    def cstack(key):
        return np.concatenate([c[key] for c in cl if len(c[key])])

    xs = np.concatenate([c["Rb"] / c["R500"] for c in cl])
    ys = np.concatenate([np.log10(np.clip(c["dS"] / c["dS_bar"], 0.03, 300.0)) for c in cl])
    yp = np.concatenate([np.log10(np.maximum(c["nu_pred"], 1e-6)) for c in cl])
    ok = np.isfinite(ys)
    for k, (a, b) in enumerate([(0.1, 0.45), (0.45, 0.9), (0.9, 2.4)]):
        s = ok & (xs >= a) & (xs < b)
        put(f"wl_b{k}", "clu_wl", np.median(ys[s]) if s.sum() > 2 else 0.0)
        put(f"wlres_b{k}", "clu_wl", np.median(ys[s] - yp[s]) if s.sum() > 2 else 0.0)
    put("wl_slope", "clu_wl", _slope(np.log10(xs[ok]), ys[ok]))

    put("q_amp", "clu_quad", np.mean([c["q_amp"] for c in cl]))
    put("q_ext", "clu_quad", np.mean([c["q_ext"] for c in cl]))
    put("q_bar", "clu_quad", np.mean([c["q_bar"] for c in cl]))
    put("q_max", "clu_quad", np.mean([c["q_max"] for c in cl]))
    put("q_ext_minus_bar", "clu_quad", np.mean([c["q_ext"] - c["q_bar"] for c in cl]))
    put("q_vs_ell", "clu_quad", _slope(np.array([c["ell"] for c in cl]),
                                       np.array([c["q_amp"] for c in cl])))

    put("net_r", "clu_net", np.mean([c["net_r"] for c in cl]))
    put("net_ctrl", "clu_net", np.mean([c["net_ctrl"] for c in cl]))
    put("net_excess", "clu_net", np.mean([c["net_r"] - c["net_ctrl"] for c in cl]))

    md = np.concatenate([np.log10(np.maximum(c["Mdyn"], 1e9)
                                  / np.maximum(c["Mb_dyn"], 1e9))
                         for c in cl if len(c["Mdyn"])])
    xd = np.concatenate([c["sig_R"] / c["R500"] for c in cl if len(c["Mdyn"])])
    pd = np.concatenate([np.log10(np.maximum(cnu, 1e-6)) for cnu in
                         [np.interp(c["gbar_dyn"], c["gbar_Rb"][::-1], c["nu_pred"][::-1])
                          for c in cl if len(c["Mdyn"])]])
    for k, (a, b) in enumerate([(0.0, 0.5), (0.5, 2.0)]):
        s = (xd >= a) & (xd < b) & np.isfinite(md)
        put(f"dyn_b{k}", "clu_dyn", np.median(md[s]) if s.sum() > 2 else 0.0)
        put(f"dynres_b{k}", "clu_dyn", np.median(md[s] - pd[s]) if s.sum() > 2 else 0.0)

    mh = np.concatenate([np.log10(np.maximum(c["Mhe"], 1e9) / np.maximum(c["Mb_ann"], 1e9))
                         for c in cl])
    xh = np.concatenate([c["r_ann"] / c["R500"] for c in cl])
    for k, (a, b) in enumerate([(0.05, 0.4), (0.4, 1.5)]):
        s = (xh >= a) & (xh < b) & np.isfinite(mh)
        put(f"he_b{k}", "clu_xray", np.median(mh[s]) if s.sum() > 2 else 0.0)
    put("he_slope", "clu_xray", _slope(np.log10(xh), mh))

    # ---- EP / slip: lensing vs dynamics vs hydrostatic at matched radii ----
    ratios_ld, ratios_lh = [], []
    for c in cl:
        if not len(c["Mdyn"]):
            continue
        R = c["sig_R"]
        M2 = np.interp(R, c["Rb"], np.maximum(c["dS"], 1e-6)) * np.pi * R ** 2 * 2.0
        ratios_ld.append(np.log10(np.maximum(M2, 1e9) / np.maximum(c["Mdyn"], 1e9)))
        Mh = np.interp(R, c["r_ann"], c["Mhe"])
        ratios_lh.append(np.log10(np.maximum(M2, 1e9) / np.maximum(Mh, 1e9)))
    if ratios_ld:
        rld = np.concatenate(ratios_ld); rlh = np.concatenate(ratios_lh)
        put("ep_ld", "clu_ep", np.median(rld))
        put("ep_lh", "clu_ep", np.median(rlh))
        put("ep_ld_scat", "clu_ep", np.std(rld))
        put("ep_ld_slope", "clu_ep", _slope(xd[:len(rld)], rld))
    else:
        for n in ("ep_ld", "ep_lh", "ep_ld_scat", "ep_ld_slope"):
            put(n, "clu_ep", 0.0)

    # ---- memory: residual vs the OBSERVED disturbance proxies -------------
    cres = np.array([np.median(np.log10(np.clip(c["dS"] / c["dS_bar"], 0.03, 300.0))
                               - np.log10(np.maximum(c["nu_pred"], 1e-6))) for c in cl])
    put("mem_shift", "clu_mem", _slope(np.log10([max(c["t_merge"], 1e-4) for c in cl]), cres))
    put("mem_offset", "clu_mem", _slope(np.log10([max(c["offset"], 1.0) for c in cl]), cres))
    put("mem_phi", "clu_mem", _slope(np.log10([max(c["phi_dep"], 1.0) for c in cl]), cres))

    put("sl_thetaE", "clu_sl", np.mean([c["thetaE"] for c in cl]))
    put("sl_frac", "clu_sl", np.mean([c["thetaE"] > 0 for c in cl]))
    put("sl_delay", "clu_sl", np.log10(1.0 + np.mean([c["delay"] for c in cl])))

    # ---------------- supernovae ---------------------------------------
    z, mag, dur, vf = C.sn["z_obs"], C.sn["mag"], C.sn["duration"], C.sn["void_frac"]
    dlm = 5 * np.log10(np.maximum(comoving_Mpc(z) * (1 + z), 1e-3)) + 25.0
    hr = mag - dlm
    put("sn_void", "sn", _slope(vf, hr))
    put("sn_dur", "sn", _slope(np.log10(1 + z), np.log10(np.maximum(dur, 1e-3))))
    put("sn_dur_void", "sn", _slope(vf, np.log10(np.maximum(dur, 1e-3)) - np.log10(1 + z)))
    put("sn_scat", "sn", np.std(hr))

    # ---- auxiliary arrays for the named identifiability experiments -------
    aux = {
        "lg": lg.tolist(), "ly": ly.tolist(),
        "gal_c3": c2.tolist(), "gal_s3": s2.tolist(),
        "gal_pa": [x["pa"] for x in g], "gal_axis_obs": [x["axis_ext"] for x in g],
        "clu_qamp": [c["q_amp"] for c in cl],
        "clu_qext": [c["q_ext"] for c in cl],
        "clu_axis_obs": [float(cd["axis_ext_obs"]) for cd in C.clu[:len(cl)]],
        "n_gal": len(g), "n_clu": len(cl),
    }

    detectors = {
        "aniso_ext": F["q_ext"], "aniso_max": F["q_max"],
        "aniso_ext_minus_bar": F["q_ext_minus_bar"],
        "network": F["net_excess"], "memory": F["mem_shift"],
        "ep_slip": F["ep_ld"], "path": F["sn_void"],
        "env": F["env_phi"], "gal_aniso": F["gq_ext"],
    }
    return {"features": F, "channels": ch, "detectors": detectors, "aux": aux}


# ---------------------------------------------------------------- estimators
def estimate_a0(lg, ly):
    """Recover the injected acceleration scale from the RAR points alone.

    lg = log10 g_bar [SI], ly = log10(g_obs/g_bar).  Fits the one-parameter
    RAR interpolating function with a0 free; returns log10 a0 [SI].
    """
    lg = np.asarray(lg, float); ly = np.asarray(ly, float)
    m = np.isfinite(lg) & np.isfinite(ly)
    lg, ly = lg[m], ly[m]
    if len(lg) < 20:
        return float("nan")
    grid = np.linspace(-10.6, -9.2, 141)
    cost = []
    for la in grid:
        x = 10 ** (lg - la)
        pred = -np.log10(1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-12))))
        cost.append(np.mean((ly - pred) ** 2))
    cost = np.array(cost)
    i = int(np.argmin(cost))
    if 0 < i < len(grid) - 1:
        d = 0.5 * (cost[i - 1] - cost[i + 1]) / (cost[i - 1] - 2 * cost[i] + cost[i + 1] + 1e-30)
        return float(grid[i] + d * (grid[1] - grid[0]))
    return float(grid[i])


def estimate_axis(aux, offset_deg=0.0):
    """Recover the EXTERNAL AXIS, one galaxy at a time.

    Each galaxy has its OWN external axis, so there is no global direction to
    stack; the recoverable statement is per-object.  The m=3 harmonic of the
    velocity-field residual has phase 2 psi with psi = (axis - PA), so

        axis_hat_j = PA_j + (1/2) arg(c3_j + i s3_j)     (mod 180 deg)

    Reported: the median angular error against the OBSERVED axis, and the
    amplitude-weighted concentration  R = |<exp(2i(axis_hat - axis_obs))>|,
    whose null distribution is set empirically on scalar universes.

    ``offset_deg`` rotates the assumed axis -- the misaligned control that
    showed power 0.03 at 45 degrees while the aligned case reached 1.00.
    """
    c3 = np.asarray(aux["gal_c3"], float)
    s3 = np.asarray(aux["gal_s3"], float)
    pa = np.asarray(aux["gal_pa"], float)
    axo = (np.asarray(aux["gal_axis_obs"], float) + offset_deg) % 180.0
    amp = np.hypot(c3, s3)
    ax_hat = (pa + 0.5 * np.rad2deg(np.arctan2(s3, c3))) % 180.0
    err = circ_err_deg(ax_hat, axo)
    w = amp / max(amp.sum(), 1e-30)
    R = np.abs(np.sum(w * np.exp(2j * np.deg2rad(ax_hat - axo))))
    proj = float(np.mean(c3 * np.cos(2 * np.deg2rad(axo - pa))
                         + s3 * np.sin(2 * np.deg2rad(axo - pa))))
    return {"median_err_deg": float(np.median(err)),
            "concentration_R": float(R),
            "aligned_projection": proj,
            "n": int(len(c3))}


def circ_err_deg(a, b):
    d = np.abs((np.asarray(a, float) - np.asarray(b, float)) % 180.0)
    return np.minimum(d, 180.0 - d)
