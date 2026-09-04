"""Random-point null: the same measurement, around positions with no cluster.

If the shear estimator, the sign convention, the metacalibration response or
the survey geometry were producing a spurious tangential signal, it would show
up here.  The positions are drawn uniformly inside the eFEDS box, rejected if
they fall within 10 arcmin of any eFEDS X-ray system, and each is assigned the
redshift of a real system so that the background cut and the binning are
identical to the real measurement.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import time

import numpy as np

import pipeline as P
import efeds_hsc as E
import acquire_decade as A

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "decade_random_shear_profiles.tsv")
N_RANDOM = 250
SEED = 20260904


def main():
    print("=" * 78)
    print("ACQUIRE -- DECADE random-point null")
    print("=" * 78)
    recs, _ = E.load_efeds()
    rng = np.random.default_rng(SEED)
    cra = np.array([r["RA"] for r in recs])
    cde = np.array([r["DE"] for r in recs])
    done = set()
    if os.path.exists(OUT):
        for ln in open(OUT, encoding="utf-8"):
            if not ln.startswith("#") and not ln.startswith("id\t"):
                done.add(ln.split("\t")[0])
    f = open(OUT, "a", encoding="utf-8")
    if not done:
        f.write("# Random-point null for the DECADE shear measurement.  Same "
                "estimator,\n# same cuts, same binning; positions carry the "
                "redshift of a real eFEDS\n# system so the background cut is "
                "identical.\n")
        f.write("id\tz\tbin\tR_Mpc\ttheta_arcmin\tn\tgt\tgx\terr\tR11\tR22\t"
                "beta\tbeta2\n")
    made, tries, t0 = 0, 0, time.time()
    while made < N_RANDOM and tries < 20 * N_RANDOM:
        tries += 1
        ra = float(rng.uniform(126.5, 145.5))
        de = float(rng.uniform(-2.5, 5.5))
        d = np.hypot((cra - ra) * math.cos(math.radians(de)), cde - de)
        if d.min() < 10.0 / 60.0:
            continue
        src = recs[int(rng.integers(len(recs)))]
        rid = f"RND{made:04d}"
        if rid in done:
            made += 1
            continue
        rec = dict(src)
        rec.update(id=rid, RA=ra, DE=de)
        rad = min(A.THETA_CAP_DEG,
                  (A.RMAX_H / P.H_LITTLE) * P.MPC / rec["DA"] * 180 / math.pi)
        try:
            _, raw = A.cone(ra, de, rad)
        except Exception as exc:                                # noqa: BLE001
            print(f"   {rid} failed: {exc}")
            continue
        rows = A.profile_for(rec, raw)
        made += 1
        if rows is None:
            continue
        for i, r in enumerate(rows):
            f.write(f"{rid}\t{rec['z']:.4f}\t{i}\t{r['R']:.5f}\t"
                    f"{r['theta_arcmin']:.4f}\t{r['n']}\t{r['gt']:.7f}\t"
                    f"{r['gx']:.7f}\t{r['err']:.7f}\t{r['R11']:.5f}\t"
                    f"{r['R22']:.5f}\t{r['beta']:.5f}\t{r['beta2']:.5f}\n")
        f.flush()
        if made % 25 == 0:
            print(f"   {made}/{N_RANDOM}  {time.time() - t0:.0f} s")
    f.close()
    blob = open(OUT, "rb").read()
    json.dump({
        "file": os.path.basename(OUT),
        "source": "NOIRLab Astro Data Lab TAP, delve_dr3.decade_shear",
        "endpoint": A.TAP,
        "exact_query_template": A.COLS + " | " + A.SEL,
        "retrieved_utc": dt.datetime.now(dt.timezone.utc)
                           .strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
        "row_count": sum(1 for ln in blob.decode().split("\n")
                         if ln and not ln.startswith("#")
                         and not ln.startswith("id\t")),
        "n_random_positions": made, "seed": SEED,
        "rejection": "positions within 10 arcmin of any eFEDS system rejected",
        "note": "Null test.  A non-zero stacked tangential signal here would "
                "invalidate the real measurement.",
    }, open(OUT + ".manifest.json", "w", encoding="utf-8"), indent=1)
    print(f"\n   wrote {os.path.basename(OUT)}  ({made} positions, "
          f"{time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
