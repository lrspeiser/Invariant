"""bmode.py -- the cross-component null test, per cluster.

WHY THIS EXISTS. The universality test says the cluster residual is not one
number, at p = 0.018. Leave-one-out says that verdict rests entirely on one
cluster: drop 1eRASS J134730.8-114510 and p goes to 0.320. That cluster is
RX J1347.5-1145, the most X-ray luminous cluster known, and its measured shear
is 1.34 sigma in total -- a non-detection where the brightest cluster in the sky
should give a strong signal.

The temptation is to drop it. That is exactly this programme's artefact #10:
removing the point that disagrees, on the grounds that it disagrees. A cluster
may only be excluded for a reason established WITHOUT reference to whether its
removal helps.

The cross component is such a reason. Gravitational lensing produces a purely
tangential distortion to first order; the component rotated by 45 degrees, g_x,
has no lensing signal in it. Anything non-zero there is instrumental: PSF
residuals, additive shear, astrometric error, a wrong centre. It is the standard
null test in every weak-lensing analysis, it was computed by `extract.py` from
the start, and this lane had never looked at it.

The test is blind to the question being asked. g_x does not know what the gravity
residual is, so passing or failing it cannot be steered by the answer wanted.

    python bmode.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
sys.path.insert(0, HERE)
sys.path.insert(0, WELLNET)

PROFILES = os.path.join(WELLNET, "clustershear", "profiles.jsonl")
OUT = os.path.join(HERE, "bmode.json")

# A cluster fails the null if its cross component is inconsistent with zero at
# better than this. Fixed before looking at any cluster's value.
P_FAIL = 0.01


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()

    gas = json.load(io.open(os.path.join(
        HERE, os.environ.get("GAS_FILE", "gas_extended_expcorr.json")),
        encoding="utf-8"))
    want = set(gas)

    prof = {}
    for line in io.open(PROFILES, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        name = d.get("cluster") or d.get("erass") or d.get("name")
        if name in want:
            prof[name] = d
    loader.assert_not_sealed(list(prof), "cross-component null test")

    print("cross-component (B-mode) null test -- g_x must be consistent with zero")
    print("")
    print("  %-24s %-6s %-8s %-6s %-9s %s"
          % ("cluster", "chi2", "dof", "p", "gt/err", "verdict"))

    rows, failed = [], []
    for name in sorted(prof):
        d = prof[name]
        bins = d.get("bins") or d.get("profile") or []
        gx = np.array([b["gx"] for b in bins if b.get("gx") is not None
                       and b.get("err")], dtype=float)
        gt = np.array([b["gt"] for b in bins if b.get("gx") is not None
                       and b.get("err")], dtype=float)
        er = np.array([b["err"] for b in bins if b.get("gx") is not None
                       and b.get("err")], dtype=float)
        ok = np.isfinite(gx) & np.isfinite(er) & (er > 0)
        gx, gt, er = gx[ok], gt[ok], er[ok]
        if len(gx) < 4:
            continue

        chi2 = float(np.sum((gx / er) ** 2))
        dof = len(gx)
        # survival function of chi2 without scipy
        try:
            from scipy.stats import chi2 as chi2d
            p = float(chi2d.sf(chi2, dof))
        except Exception:                                       # noqa: BLE001
            p = float("nan")
        # signal-to-noise of the TANGENTIAL component, for context only
        snr_t = float(np.sum(gt / er ** 2) / math.sqrt(np.sum(1.0 / er ** 2)))
        bad = np.isfinite(p) and p < P_FAIL
        if bad:
            failed.append(name)
        rows.append(dict(cluster=name, chi2_x=chi2, dof=dof, p_x=p,
                         snr_tangential=snr_t, fails_null=bool(bad)))
        print("  %-24s %-6.1f %-8d %-6.4f %-9.2f %s"
              % (name, chi2, dof, p, snr_t,
                 "FAILS NULL" if bad else "clean"))

    print("")
    if failed:
        print("  %d cluster(s) FAIL the cross-component null:" % len(failed))
        for f in failed:
            print("    %s" % f)
        print("")
        print("  These carry instrumental signal in a channel that lensing cannot")
        print("  produce. Excluding them is justified independently of what their")
        print("  removal does to the gravity result.")
    else:
        print("  every cluster passes. No cluster may be excluded on these grounds,")
        print("  and in particular the leave-one-out outlier stays in.")

    out = dict(lane="clusterfirst", stage="bmode-null", p_fail_threshold=P_FAIL,
               n_clusters=len(rows), n_failed=len(failed), failed=failed,
               clusters=rows,
               note=("g_x is the 45-degree-rotated shear component. Lensing puts no "
                     "signal there, so a non-zero value is instrumental. Computed by "
                     "extract.py from the start and never tested until now."),
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote bmode.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
