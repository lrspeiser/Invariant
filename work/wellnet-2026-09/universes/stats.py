"""stats.py -- calibrated separation testing.

The rule this lane works under: a test is worthless until it has been SIZED.
An earlier lane in this programme ran an obvious permutation test at a false
positive rate of 0.53-0.70 against a nominal 0.05.  So:

  * the discriminant is fitted ONLY on calibration draws and evaluated ONLY on
    disjoint audit draws (separate simulations for calibration and audit, as
    the charter requires);
  * the p-value comes from PERMUTING THE AUDIT LABELS, not from an asymptotic
    formula;
  * the same procedure is run on A-vs-A pairs (identical universe, different
    seeds) and the realised false-positive rate is reported before any verdict;
  * multiplicity across the 45 pairs is handled with the max-|z| null, not
    Bonferroni on an assumed Gaussian.

Everything is quoted as a slope or an effect size with its responsiveness
d(estimate)/d(injected), never as a correlation coefficient.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

Z_CAP = 8.5          # permutation nulls cannot resolve beyond this; declared


def robust_standardise(Xc, Xa):
    med = np.median(Xc, 0)
    mad = np.median(np.abs(Xc - med), 0) * 1.4826
    sd = np.std(Xc, 0)
    scale = np.where(mad > 1e-12, mad, np.where(sd > 1e-12, sd, 1.0))
    return (Xc - med) / scale, (Xa - med) / scale


def shrinkage_lda(X0, X1, shrink=None):
    """Ledoit-Wolf-shrunk linear discriminant.  Returns the weight vector."""
    n0, n1 = len(X0), len(X1)
    m0, m1 = X0.mean(0), X1.mean(0)
    Xc = np.vstack([X0 - m0, X1 - m1])
    n, p = Xc.shape
    S = Xc.T @ Xc / max(n - 2, 1)
    if shrink is None:
        # Ledoit-Wolf towards a scaled identity
        mu = np.trace(S) / p
        d2 = np.sum((S - mu * np.eye(p)) ** 2) / p
        b2 = min(np.sum([np.sum((np.outer(x, x) - S) ** 2) for x in Xc]) / (n ** 2 * p), d2)
        shrink = float(np.clip(b2 / max(d2, 1e-30), 0.02, 0.98))
    St = (1 - shrink) * S + shrink * (np.trace(S) / p) * np.eye(p)
    w = np.linalg.solve(St + 1e-9 * np.eye(p), m1 - m0)
    nw = np.linalg.norm(w)
    return w / nw if nw > 0 else w


def auc(s0, s1):
    """Mann-Whitney AUC with MID-RANKS.

    Sequential ranks (argsort of argsort) are wrong here: when every score is
    tied -- which happens whenever a channel's features are degenerate, e.g.
    a strong-lensing channel in universes that produce no arcs -- they hand the
    second group the top ranks and return AUC = 1.0 deterministically.  That
    bug produced an apparent z = 4.8 between two universes that are identical
    in that channel by construction.  Mid-ranks return 0.5, as they must.
    """
    a = np.concatenate([s0, s1])
    r = rankdata(a)
    n0, n1 = len(s0), len(s1)
    return float((r[n0:].sum() - n1 * (n1 + 1) / 2.0) / (n0 * n1))


def separation(Ac, Bc, Aa, Ba, n_perm=4000, seed=0):
    """Calibrated two-sample separation of two universes.

    Ac, Bc  calibration feature matrices  (n_cal, p)
    Aa, Ba  AUDIT feature matrices        (n_aud, p) -- never seen by the fit

    Returns AUC on the audit set, an exact permutation p-value, and a
    calibrated z = (AUC - mean_null)/sd_null.
    """
    rng = np.random.default_rng(seed)
    Xc, Xa = robust_standardise(np.vstack([Ac, Bc]), np.vstack([Aa, Ba]))
    n0c = len(Ac)
    w = shrinkage_lda(Xc[:n0c], Xc[n0c:])
    s = Xa @ w
    n0a = len(Aa)
    s0, s1 = s[:n0a], s[n0a:]
    obs = auc(s0, s1)
    stat = abs(obs - 0.5)
    # vectorised permutation null: the RANKS of the audit scores are fixed, so
    # a permutation only re-selects which ranks belong to group 1.
    n = len(s)
    r = rankdata(s)
    n1 = n - n0a
    sel = np.argsort(rng.random((n_perm, n)), axis=1)[:, :n1]
    R = r[sel].sum(1)
    null = (R - n1 * (n1 + 1) / 2.0) / (n0a * n1)
    p = (1.0 + np.sum(np.abs(null - 0.5) >= stat - 1e-12)) / (n_perm + 1.0)
    sd = np.std(null)
    if sd < 1e-9 or stat < 1e-9:
        # a degenerate channel carries no information; say so, do not
        # manufacture a z from a zero-variance null
        return {"auc": float(obs), "p": 1.0, "z": 0.0, "null_sd": float(sd),
                "n_cal": int(n0c), "n_aud": int(n0a), "z_capped": False,
                "degenerate": True}
    z = float(np.clip(stat / sd, 0.0, Z_CAP))
    return {"auc": float(obs), "p": float(p), "z": z,
            "null_sd": float(sd), "n_cal": int(n0c), "n_aud": int(n0a),
            "z_capped": bool(stat / max(sd, 1e-12) > Z_CAP)}


def detector_calibration(null_vals, alpha=0.05):
    """Empirical two-sided critical value p* with P(|S| >= p* | H0) = alpha.

    The brief's rule: do not declare a threshold, MEASURE it -- the nonlocal
    detector's 11.7% could not be repaired by moving the nominal alpha.
    """
    v = np.abs(np.asarray(null_vals, float))
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return {"crit": float("nan"), "n": int(len(v))}
    return {"crit": float(np.quantile(v, 1 - alpha)), "n": int(len(v)),
            "median": float(np.median(v)),
            "tail_ratio_p95_over_median": float(np.quantile(v, 0.95) /
                                                max(np.median(v), 1e-30))}


def rate_with_ci(hits, n):
    """Wilson 95% interval -- a rate with no interval is not a measurement."""
    if n == 0:
        return {"rate": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    ph = hits / n
    z = 1.959963985
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return {"rate": float(ph), "lo": float(max(0.0, c - h)),
            "hi": float(min(1.0, c + h)), "n": int(n), "hits": int(hits)}


def responsiveness(thetas, estimates):
    """d(estimate)/d(injected), with its standard error.

    Every headline number in this lane carries one.  Where it is consistent
    with zero the report must say that NO UPPER LIMIT HAS BEEN SET.
    """
    t = np.asarray(thetas, float)
    e = np.asarray(estimates, float)
    m = np.isfinite(t) & np.isfinite(e)
    t, e = t[m], e[m]
    if len(t) < 3 or np.ptp(t) == 0:
        return {"slope": float("nan"), "se": float("nan"), "t": float("nan"),
                "responsive": False, "n": int(len(t))}
    A = np.stack([np.ones_like(t), t], 1)
    coef, res, *_ = np.linalg.lstsq(A, e, rcond=None)
    resid = e - A @ coef
    dof = max(len(t) - 2, 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.inv(A.T @ A)
    se = float(np.sqrt(max(cov[1, 1], 0.0)))
    tv = coef[1] / se if se > 0 else np.inf
    return {"slope": float(coef[1]), "se": se, "t": float(tv),
            "responsive": bool(abs(tv) > 2.0), "n": int(len(t)),
            "range": [float(t.min()), float(t.max())]}


def threshold_amplitude(amps, zs, target=5.0):
    """Smallest amplitude at which the calibrated z reaches `target`.

    Fits z = k * amp^q on the amplitudes with z > 0 and inverts.  Returns None
    when no amplitude in the scan reaches the target and the extrapolation is
    outside the scanned range by more than 3x (an honest "not established").
    """
    a = np.asarray(amps, float)
    z = np.asarray(zs, float)
    m = (a > 0) & np.isfinite(z) & (z > 0.15)
    if m.sum() < 2:
        return {"amp": None, "reason": "no responsive amplitude in the scan"}
    if np.any(z[m] >= target):
        i = np.argmax(z >= target)
        lo = max(i - 1, 0)
        if z[i] == z[lo]:
            return {"amp": float(a[i]), "method": "direct"}
        return {"amp": float(np.interp(target, [z[lo], z[i]], [a[lo], a[i]])),
                "method": "interpolated"}
    la, lz = np.log(a[m]), np.log(z[m])
    q, c = np.polyfit(la, lz, 1)
    amp = float(np.exp((np.log(target) - c) / q)) if q > 0 else None
    if amp is None or amp > 3.0 * a.max():
        return {"amp": None, "reason": "extrapolation beyond 3x the scanned range",
                "power_law_index": float(q), "max_z_in_scan": float(np.nanmax(z))}
    return {"amp": amp, "method": "power-law extrapolation",
            "power_law_index": float(q)}


def separation_max(Ac, Bc, Aa, Ba, chan_idx, n_perm=3000, seed=0):
    """The test statistic is the MAX over {whole corpus, every channel}.

    The brief's rule: calibrate the ENTIRE SEARCH, not a selected statistic.
    A single discriminant over all ~60 features is diluted by the many
    features that carry nothing for a given pair, so an analyst would look at
    the channels too.  Taking the max and calibrating THE MAX under A-vs-A
    nulls is the honest version of that: the look-elsewhere cost of choosing a
    channel is inside the critical value.
    """
    out = {"full": separation(Ac, Bc, Aa, Ba, n_perm=n_perm, seed=seed)}
    for c, ix in chan_idx.items():
        if len(ix) < 2:
            continue
        out[c] = separation(Ac[:, ix], Bc[:, ix], Aa[:, ix], Ba[:, ix],
                            n_perm=n_perm, seed=seed + 17)
    zmax = max(v["z"] for v in out.values())
    best = max(out, key=lambda k: out[k]["z"])
    return {"z_max": float(zmax), "best_test": best,
            "z_full": out["full"]["z"], "auc_full": out["full"]["auc"],
            "per_test": {k: {"z": v["z"], "auc": v["auc"], "p": v["p"],
                             "degenerate": v.get("degenerate", False)}
                         for k, v in out.items()}}
