"""pipeline.py -- corpus -> candidate statistics.

Everything here is computed from the DETECTOR-LEVEL catalogue that
``universes.corpus.emit_cluster`` emits (per-source e1, e2, weights, photo-z,
member positions, X-ray annuli).  None of Run BF's detectors is used: the
quadrupole comes from ``estimators.cluster_quadrupole``, which fits monopole,
m=2 and m=4 jointly in tangential AND cross ellipticity per radial bin, and
returns a covariance.

The candidate statistics, each ONE number per corpus, each SIGNED, each
studentised by its own propagated error so its null distribution does not
depend on how big the quadrupole happens to be:

  S_bar    Sum_j Re[Z_j e^{-2i pa_bar,j}] / sigma_j / sqrt(N)
           "is there a quadrupole locked to the BARYON major axis?"
  S_ext    the same on the independently observed EXTERNAL axis
           "is there a quadrupole locked to the EXTERNAL axis?"
  S_diff   S_ext - S_bar with the covariance propagated: the SIGNED contrast.
           Run BF's aniso_ext_minus_bar is the two-sided |.| of an unstudentised
           version of this; the sign is the information it threw away.
  S_morph  slope of the dimensionless quadrupole power on the observed baryon
           ellipticity, studentised.  Phase-free.
  S_shape  studentised contrast of the quadrupole power between the inner and
           outer radial bins.  Phase-free.
  S_45     S_ext computed on an axis rotated by 45 degrees: the misspecified-axis
           control.  A misspecified axis is a null detector, so this must sit at
           zero in every arm; it sizes the axis requirement directly.

Also returned, as auxiliary DIAGNOSTICS (not candidates): the per-cluster
quadrupole amplitude, its phase, and the unstudentised projections, which is
what Job 1 needs to explain Run BF's confusion.
"""
from __future__ import annotations

import numpy as np

import estimators as E


def _safe(x, cap=200.0):
    x = float(x)
    if not np.isfinite(x):
        return 0.0
    return float(np.clip(x, -cap, cap))


def cluster_rows(C, sigma_crit_fn):
    """Per-cluster reduction of a corpus: the raw material for every statistic.

    ``C`` is either a ``universes.corpus.Corpus`` or a plain list of cluster
    dicts, so the same reduction runs on Run BF's generator and on the
    independently implemented forward model in ``forward.py``.
    """
    rows = []
    for cd in (C.clu if hasattr(C, "clu") else C):
        q = E.cluster_quadrupole(cd, sigma_crit_fn)
        if q is None:
            continue
        Z, Cc, n = E.combine_bins(q)
        if n == 0:
            continue
        vb, varb = E.project(Z, Cc, cd["pa_bar_obs"])
        ve, vare = E.project(Z, Cc, cd["axis_ext_obs"])
        v45, var45 = E.project(Z, Cc, cd["axis_ext_obs"] + 45.0)
        # covariance of the difference: Var(ve - vb) = u_e C u_e + u_b C u_b
        #                                             - 2 u_e C u_b
        ae, ab = np.deg2rad(cd["axis_ext_obs"]), np.deg2rad(cd["pa_bar_obs"])
        ue = np.array([np.cos(2 * ae), np.sin(2 * ae)])
        ub = np.array([np.cos(2 * ab), np.sin(2 * ab)])
        vard = float((ue - ub) @ Cc @ (ue - ub))
        # per-bin power, debiased
        bins = [b for b in q["bins"]]
        Q2 = [b["Q2"] for b in bins]
        mono = [b["mono"] for b in bins]
        # error on Q2 = |Z|^2 - tr(C):  Var ~ 4 Z^T C Z (for |Z| >> noise)
        #                                    + 2 tr(C^2) (for |Z| ~ 0)
        Q2e = []
        for b in bins:
            zz = np.array([b["Z"].real, b["Z"].imag])
            Q2e.append(np.sqrt(max(4.0 * zz @ b["C"] @ zz
                                   + 2.0 * np.trace(b["C"] @ b["C"]), 1e-30)))
        rows.append(dict(
            name=cd["name"], Z=Z, C=Cc, n=n, R500=float(cd["R500"]),
            pa_q=float(np.rad2deg(0.5 * np.angle(Z)) % 180.0),
            pa_bar=float(cd["pa_bar_obs"]), ax_ext=float(cd["axis_ext_obs"]),
            ell_bar=float(cd["ell_bar_obs"]),
            amp=float(abs(Z)),
            snr=float(abs(Z) / np.sqrt(max(np.trace(Cc) / 2.0, 1e-30))),
            proj_bar=vb, proj_bar_sd=np.sqrt(varb),
            proj_ext=ve, proj_ext_sd=np.sqrt(vare),
            proj_45=v45, proj_45_sd=np.sqrt(var45),
            diff=ve - vb, diff_sd=np.sqrt(max(vard, 1e-30)),
            Q2=np.array(Q2), Q2_err=np.array(Q2e), mono=np.array(mono),
            bin_n=np.array([b["n"] for b in bins]),
            # per-bin dimensionless quadrupole power, (Q/monopole)^2.  The
            # monopole divides out the lens mass, so this is the quantity a
            # halo's ellipticity sets and a tensor's amplitude sets, on the
            # same footing.  Normalising by the SUM of the squared monopoles
            # instead makes the ratio unstable whenever an outer bin is weak.
            # studentised power per bin.  Normalising the quadrupole by the
            # MONOPOLE looks natural but is unusable: the monopole of a
            # baryon-only cluster passes through zero, so the ratio has no
            # finite moments.  Q2/Q2_err is dimensionless, has null mean 0,
            # and its denominator is set by the shape noise, which is
            # independent of the baryon ellipticity by construction.
            prof=np.array([q / max(e, 1e-30) for q, e in zip(Q2, Q2e)]),
            Q2_tot=float(np.sum(Q2)),
            Q2_tot_err=float(np.sqrt(np.sum(np.asarray(Q2e) ** 2))),
            mono_ref=float(mono[1] if len(mono) > 1 else mono[0]),
        ))
    return rows


def _stud_sum(vals, sds):
    v = np.asarray(vals, float) / np.maximum(np.asarray(sds, float), 1e-30)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0
    return float(np.sum(v) / np.sqrt(len(v)))


def _slope_t(x, y, ye):
    """Weighted slope of y on x with its standard error -> t statistic.

    Quote slopes, not correlations.  Returns (slope, se, t).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = 1.0 / np.maximum(np.asarray(ye, float), 1e-30) ** 2
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[m], y[m], w[m]
    if len(x) < 4 or np.ptp(x) <= 0:
        return 0.0, 0.0, 0.0
    A = np.stack([np.ones_like(x), x], 1)
    W = A * w[:, None]
    try:
        Xi = np.linalg.inv(A.T @ W)
    except np.linalg.LinAlgError:
        return 0.0, 0.0, 0.0
    beta = Xi @ (W.T @ y)
    se = float(np.sqrt(max(Xi[1, 1], 0.0)))
    return float(beta[1]), se, float(beta[1] / se) if se > 0 else 0.0


def statistics(rows):
    """The candidate statistics for one corpus."""
    if len(rows) < 3:
        return None
    S = {}
    S["S_bar"] = _stud_sum([r["proj_bar"] for r in rows],
                           [r["proj_bar_sd"] for r in rows])
    S["S_ext"] = _stud_sum([r["proj_ext"] for r in rows],
                           [r["proj_ext_sd"] for r in rows])
    S["S_45"] = _stud_sum([r["proj_45"] for r in rows],
                          [r["proj_45_sd"] for r in rows])
    S["S_diff"] = _stud_sum([r["diff"] for r in rows],
                            [r["diff_sd"] for r in rows])

    # UNSTUDENTISED variants: the same projections averaged with equal weight,
    # which is the form Run BF's aniso_ext / aniso_ext_minus_bar take.  Keeping
    # both makes the false-positive rate on CDM decomposable into the part due
    # to discarding the sign and the part due to not studentising.
    S["S_ext_raw"] = float(np.mean([r["proj_ext"] for r in rows]))
    S["S_bar_raw"] = float(np.mean([r["proj_bar"] for r in rows]))
    S["S_diff_raw"] = float(np.mean([r["diff"] for r in rows]))
    S["S_45_raw"] = float(np.mean([r["proj_45"] for r in rows]))

    # ---- phase-free candidates ------------------------------------------
    # dimensionless quadrupole power: total debiased Q2 over the squared
    # monopole, so the amplitude of the lens itself divides out
    P = [r["Q2_tot"] / max(r["Q2_tot_err"], 1e-30) for r in rows]
    Pe = [1.0] * len(rows)
    b, se, t = _slope_t([r["ell_bar"] for r in rows], P, Pe)
    S["S_morph"] = _safe(t)
    S["_morph_slope"] = _safe(b, 1e9)
    S["_morph_se"] = _safe(se, 1e9)

    # radial shape: inner minus outer dimensionless power, studentised
    num, den = 0.0, 0.0
    for r in rows:
        if len(r["Q2"]) < 3:
            continue
        d = float(r["prof"][0] - r["prof"][-1])
        de = float(np.sqrt(2.0))
        if not np.isfinite(de) or de <= 0 or not np.isfinite(d):
            continue
        num += d / de ** 2
        den += 1.0 / de ** 2
    S["S_shape"] = _safe(num / np.sqrt(den) if den > 0 else 0.0)

    for k in list(S):
        S[k] = _safe(S[k])
    S["_n_clu"] = len(rows)
    S["_med_amp"] = float(np.median([r["amp"] for r in rows]))
    S["_med_snr"] = float(np.median([r["snr"] for r in rows]))
    return S


def diagnostics(rows):
    """Job 1 material: what the per-cluster quadrupole actually looks like."""
    def circ(a, b):
        d = np.abs((np.asarray(a) - np.asarray(b)) % 180.0)
        return np.minimum(d, 180.0 - d)
    paq = np.array([r["pa_q"] for r in rows])
    pab = np.array([r["pa_bar"] for r in rows])
    axe = np.array([r["ax_ext"] for r in rows])
    amp = np.array([r["amp"] for r in rows])
    w = amp / max(amp.sum(), 1e-30)
    return dict(
        n=len(rows),
        amp_median=float(np.median(amp)),
        snr_median=float(np.median([r["snr"] for r in rows])),
        # circular concentration R = |<e^{2i(pa_q - axis)}>|, amplitude-weighted
        R_bar=float(abs(np.sum(w * np.exp(2j * np.deg2rad(paq - pab))))),
        R_ext=float(abs(np.sum(w * np.exp(2j * np.deg2rad(paq - axe))))),
        err_bar_median=float(np.median(circ(paq, pab))),
        err_ext_median=float(np.median(circ(paq, axe))),
        # unstudentised projections -- exactly what BF's detector averages
        raw_proj_bar=float(np.mean([r["proj_bar"] for r in rows])),
        raw_proj_ext=float(np.mean([r["proj_ext"] for r in rows])),
        raw_diff=float(np.mean([r["diff"] for r in rows])),
        # radial profile of the dimensionless quadrupole power
        prof=[float(np.mean([r["prof"][k] for r in rows if len(r["prof"]) > k]))
              for k in range(3)],
        prof_Q2=[float(np.mean([r["Q2"][k] for r in rows if len(r["Q2"]) > k]))
                 for k in range(3)],
        ell_bar=[float(r["ell_bar"]) for r in rows],
        power=[float(r["Q2_tot"] / max(r["Q2_tot_err"], 1e-30)) for r in rows],
        mono_ref=[float(r["mono_ref"]) for r in rows],
    )
