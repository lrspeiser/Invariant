"""analyse.py -- what the open half's shear profiles actually show, sized against its own nulls.

The programme's standing lesson is that a trend is worth nothing until the same
pipeline has been run on something that cannot contain it. Eight shared-
denominator artefacts were found the hard way. So every number here is
accompanied by the null that sizes it, and the nulls are computed by the SAME
code path, not by an analytic estimate.

WHAT IS MEASURED
  1. the stacked profile g_t(R), g_x(R) over the open half
  2. the amplitude of each cluster's profile, and how it varies with the X-ray
     observables (kT, counts) and with redshift
  3. the radial dependence -- the thing round 2 said the surviving cluster
     excess is organised by

THE THREE NULLS
  N1  CROSS COMPONENT.  g_x is a parity-odd combination that gravitational
      lensing cannot produce. If <g_x> is not consistent with zero on the stack,
      the systematic floor is above the signal and nothing else here means
      anything.
  N2  RANDOM POINTINGS.  The identical estimator run at sky positions with no
      cluster, drawn from the same footprint. This is the additive-systematics
      floor of the catalogue, measured rather than assumed.
  N3  LABEL SCRAMBLE.  Cluster labels permuted against profiles. Any correlation
      between an X-ray observable and a shear amplitude must beat the
      distribution this produces, or it is an artefact of the marginals.

WHAT IS NOT DONE HERE.  No mass is fitted and no gravity law is scored. eRASS1's
M500/R500/Mgas500 are weak-lensing-calibrated scaling-relation products, so using
them against weak lensing would be circular; the admissible X-ray observables are
counts, fluxes and kT. This module characterises the data. Testing a law on it is
a later, pre-registered step -- and the sealed half is where a law gets confirmed.

    python analyse.py
"""
from __future__ import annotations

import io
import json
import math
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request

import numpy as np

import cosmo as C
import extract
import guard

HERE = os.path.dirname(os.path.abspath(__file__))
PROF = os.path.join(HERE, "profiles.jsonl")
RANDOM = os.path.join(HERE, "random_profiles.jsonl")
OUT = os.path.join(HERE, "analysis.json")

RNG_SEED = 20260907
N_SCRAMBLE = 2000


def load_profiles(path):
    recs = []
    if not os.path.exists(path):
        return recs
    for line in io.open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if r.get("profile"):
                recs.append(r)
    return recs


def _bins(recs):
    """Inverse-variance stack of g_t and g_x, bin by bin."""
    nb = extract.NBIN
    out = []
    for i in range(nb):
        gt, gx, w, R, n = [], [], [], [], 0
        for r in recs:
            b = r["profile"][i]
            if b["gt"] is None or b["err"] is None or not b["err"] > 0:
                continue
            iv = 1.0 / b["err"] ** 2
            gt.append(b["gt"] * iv)
            gx.append(b["gx"] * iv)
            w.append(iv)
            R.append(b["R"] * iv)
            n += b["n"]
        if not w:
            out.append(dict(R=None, gt=None, gx=None, err=None, n_cl=0, n_src=0))
            continue
        sw = sum(w)
        out.append(dict(R=sum(R) / sw, gt=sum(gt) / sw, gx=sum(gx) / sw,
                        err=math.sqrt(1.0 / sw), n_cl=len(w), n_src=n))
    return out


def amplitude(r):
    """One number per cluster: inverse-variance mean of g_t over its bins.

    Deliberately crude. A fitted mass would import a halo profile and a
    scaling relation, which is exactly what must not enter a quantity that a
    gravity law will later be scored against.
    """
    num = den = 0.0
    for b in r["profile"]:
        if b["gt"] is None or not (b["err"] or 0) > 0:
            continue
        iv = 1.0 / b["err"] ** 2
        num += b["gt"] * iv
        den += iv
    if den == 0:
        return None, None
    return num / den, math.sqrt(1.0 / den)


def weighted_corr(x, y, w):
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
    mx = np.sum(w * x) / np.sum(w)
    my = np.sum(w * y) / np.sum(w)
    cx, cy = x - mx, y - my
    denom = math.sqrt(np.sum(w * cx ** 2) * np.sum(w * cy ** 2))
    return float(np.sum(w * cx * cy) / denom) if denom > 0 else float("nan")


def weighted_slope(x, y, w):
    """Slope of y on x. Report the slope, not the correlation (Run AT's rule)."""
    x, y, w = np.asarray(x), np.asarray(y), np.asarray(w)
    sw = np.sum(w)
    mx, my = np.sum(w * x) / sw, np.sum(w * y) / sw
    sxx = np.sum(w * (x - mx) ** 2)
    return float(np.sum(w * (x - mx) * (y - my)) / sxx) if sxx > 0 else float("nan")


def scramble_null(x, y, w, n=N_SCRAMBLE, seed=RNG_SEED):
    """N3 -- the slope distribution under permuted labels."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    y = np.asarray(y)
    w = np.asarray(w)
    out = np.empty(n)
    idx = np.arange(len(x))
    for k in range(n):
        p = rng.permutation(idx)
        out[k] = weighted_slope(x, y[p], w[p])
    return out


def trend(recs, key, label):
    """Slope of shear amplitude on an X-ray observable, against its scramble null."""
    xs, ys, ws = [], [], []
    for r in recs:
        v = r.get(key)
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(v) or v <= 0:
            continue
        a, ea = amplitude(r)
        if a is None or not ea > 0:
            continue
        xs.append(math.log10(v))
        ys.append(a)
        ws.append(1.0 / ea ** 2)
    if len(xs) < 30:
        return dict(observable=label, n=len(xs), verdict="TOO FEW")
    s = weighted_slope(xs, ys, ws)
    null = scramble_null(xs, ys, ws)
    sd = float(np.std(null))
    z = (s - float(np.mean(null))) / sd if sd > 0 else float("nan")
    p = float((np.abs(null - np.mean(null)) >= abs(s - np.mean(null))).mean())
    return dict(observable=label, key=key, n=len(xs),
                slope=s, null_sd=sd, z=z, p_permutation=p,
                corr=weighted_corr(xs, ys, ws),
                verdict=("consistent with the label-scramble null" if p > 0.05
                         else "beats the scramble null at p=%.4f" % p))


# ------------------------------------------------------------- N2: randoms
def fetch_random_profiles(n_want, recs, seed=RNG_SEED):
    """Run the identical estimator at cluster-free positions in the same footprint.

    Positions are offsets of 1.0-1.5 deg from real clusters, which keeps them
    inside the DECADE footprint (which is patchy, so a uniform draw would mostly
    miss) while putting them well outside any cluster's 3.5 h^-1 Mpc aperture.
    """
    done = set()
    if os.path.exists(RANDOM):
        for line in io.open(RANDOM, encoding="utf-8"):
            if line.strip():
                done.add(json.loads(line)["name"])
    rng = np.random.default_rng(seed)
    fh = io.open(RANDOM, "a", newline="\n", encoding="utf-8")
    made = len(done)
    for r in recs:
        if made >= n_want:
            break
        nm = "RAND-" + r["name"]
        if nm in done:
            continue
        ang = rng.uniform(0, 2 * math.pi)
        off = rng.uniform(1.0, 1.5)
        ra = (r["ra"] + off * math.cos(ang) / max(1e-6,
              math.cos(math.radians(r["dec"])))) % 360.0
        dec = max(-89.0, min(89.0, r["dec"] + off * math.sin(ang)))
        fake = dict(name=nm, ra=ra, dec=dec, z=r["z"], zType="random",
                    cts500=r["cts500"], kt=r["kt"])
        rad = extract.cone_radius_deg(r["z"])
        q = ("SELECT %s FROM delve_dr3.decade_shear WHERE "
             "'t' = q3c_radial_query(ra, dec, %.6f, %.6f, %.6f) AND %s"
             % (extract.COLS, ra, dec, rad, extract.SEL))
        try:
            rows, nsrc = extract.profile_for(fake, extract.tap(q))
            fake["profile"] = rows
            fake["n_sources"] = nsrc
            fake["error"] = None
            if rows:
                made += 1
        except Exception as exc:                            # noqa: BLE001
            fake["profile"] = None
            fake["n_sources"] = None
            fake["error"] = str(exc)[:200]
        fh.write(json.dumps(fake) + "\n")
        fh.flush()
    fh.close()
    return load_profiles(RANDOM)


def main(n_random=120):
    sys.path.insert(0, os.path.dirname(HERE))
    from holdout import loader                              # noqa: E402
    loader.verify()

    recs = load_profiles(PROF)
    guard.assert_all_open([r["name"] for r in recs], "analysis input")
    print("profiles: %d clusters (open half only)" % len(recs), flush=True)

    stack = _bins(recs)

    print("fetching %d random-pointing profiles for N2 ..." % n_random, flush=True)
    rnd = fetch_random_profiles(n_random, recs)
    rstack = _bins(rnd)
    print("randoms: %d" % len(rnd), flush=True)

    # N1 -- cross component on the stack
    gx = [(b["gx"], b["err"]) for b in stack if b["gx"] is not None]
    chi2_x = sum((g / e) ** 2 for g, e in gx)
    gt = [(b["gt"], b["err"]) for b in stack if b["gt"] is not None]
    snr = math.sqrt(sum((g / e) ** 2 for g, e in gt))

    # N2 -- randoms
    rgt = [(b["gt"], b["err"]) for b in rstack if b["gt"] is not None]
    chi2_r = sum((g / e) ** 2 for g, e in rgt) if rgt else None

    out = dict(
        lane="clustershear",
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        half="OPEN", n_clusters=len(recs), n_randoms=len(rnd),
        sealed_untouched=len(loader.sealed_names()),
        estimator=dict(
            source="efeds-hsc/acquire_decade.py (Run BM), reused unchanged",
            bins=extract.NBIN, r_range_hinv_mpc=[extract.RMIN_H, extract.RMAX_H],
            background_dz=extract.DZ_BG, dgamma=extract.DGAMMA,
            sign_convention="measured, x axis west (Run BM)"),
        stack=stack,
        random_stack=rstack,
        nulls=dict(
            N1_cross=dict(chi2=chi2_x, dof=len(gx),
                          verdict=("consistent with zero" if chi2_x < 2 * len(gx)
                                   else "CROSS COMPONENT IS NOT ZERO -- "
                                        "systematics floor above the signal")),
            N2_randoms=dict(chi2=chi2_r, dof=len(rgt),
                            verdict=(None if chi2_r is None else
                                     "consistent with zero" if chi2_r < 2 * len(rgt)
                                     else "RANDOM POINTINGS SHOW SIGNAL -- "
                                          "additive systematic present")),
        ),
        stack_snr=snr,
        trends=[trend(recs, "kt", "X-ray temperature kT"),
                trend(recs, "cts500", "X-ray counts within R500"),
                trend(recs, "z", "redshift")],
        note=("Characterisation only. No mass is fitted and no gravity law is "
              "scored. eRASS1 M500/R500/Mgas500 are weak-lensing-calibrated and "
              "are excluded as circular."),
    )
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")

    print("\nstacked profile, open half (%d clusters):" % len(recs))
    print("  %-9s %-11s %-11s %-9s %s" % ("R [Mpc]", "g_t", "g_x", "err", "n_cl"))
    for b in stack:
        if b["gt"] is None:
            continue
        print("  %-9.3f %+11.5f %+11.5f %-9.5f %d"
              % (b["R"], b["gt"], b["gx"], b["err"], b["n_cl"]))
    print("\n  total S/N on g_t   : %.1f" % snr)
    print("  N1 cross  chi2/dof : %.1f/%d  -> %s"
          % (chi2_x, len(gx), out["nulls"]["N1_cross"]["verdict"]))
    if chi2_r is not None:
        print("  N2 randoms chi2/dof: %.1f/%d  -> %s"
              % (chi2_r, len(rgt), out["nulls"]["N2_randoms"]["verdict"]))
    print("\ntrends (slope of shear amplitude on log10 observable):")
    for t in out["trends"]:
        if t.get("verdict") == "TOO FEW":
            print("  %-28s n=%d TOO FEW" % (t["observable"], t["n"]))
            continue
        print("  %-28s n=%3d slope=%+.5f  z=%+.2f  p=%.4f  %s"
              % (t["observable"], t["n"], t["slope"], t["z"], t["p_permutation"],
                 t["verdict"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
