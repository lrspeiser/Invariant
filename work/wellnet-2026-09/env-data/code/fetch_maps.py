"""Download MaNGA DAP MAPS (HYB10-MILESHC-MASTARSSP) for the matched-pair sample.

One MAPS file per galaxy carries the resolved stellar velocity field, stellar
velocity dispersion, ionised-gas (H-alpha) velocity field and gas dispersion,
plus the elliptical-polar radius map -- i.e. everything needed to fit a rotation
curve and a dispersion profile for each member of a matched pair.

Usage:  python fetch_maps.py [--list <file of plateifu>] [--limit N]
"""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
CLEAN = os.path.join(LANE, "clean")
DEST = os.path.join(LANE, "raw", "manga", "maps")
DAPTYPE = "HYB10-MILESHC-MASTARSSP"
BASE = ("https://data.sdss.org/sas/dr17/manga/spectro/analysis/v3_1_1/3.1.0/"
        "%s/{plate}/{ifu}/manga-{plate}-{ifu}-MAPS-%s.fits.gz" % (DAPTYPE, DAPTYPE))


def one(plateifu, session):
    plate, ifu = plateifu.split("-")
    url = BASE.format(plate=plate, ifu=ifu)
    out = os.path.join(DEST, "manga-%s-MAPS-%s.fits.gz" % (plateifu, DAPTYPE))
    if os.path.exists(out) and os.path.getsize(out) > 100000:
        return plateifu, os.path.getsize(out), None, url
    try:
        r = session.get(url, timeout=300, stream=True)
        if r.status_code != 200:
            return plateifu, 0, "HTTP %s" % r.status_code, url
        n = 0
        tmp = out + ".part"
        with open(tmp, "wb") as f:
            for c in r.iter_content(1 << 20):
                f.write(c)
                n += len(c)
        os.replace(tmp, out)
        return plateifu, n, None, url
    except Exception as e:
        return plateifu, 0, "%s: %s" % (type(e).__name__, e), url


def main():
    import requests
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    os.makedirs(DEST, exist_ok=True)
    if a.list:
        want = [l.strip() for l in open(a.list) if l.strip()]
    else:
        p = pd.read_csv(os.path.join(CLEAN, "matched_pairs.csv"), low_memory=False)
        want = sorted(set(p["cl_plateifu"]) | set(p["fi_plateifu"]))
    if a.limit:
        want = want[:a.limit]
    print("targets: %d galaxies" % len(want), flush=True)

    s = requests.Session()
    t0 = time.time()
    ok, bad, total = [], [], 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for i, (pif, n, err, url) in enumerate(
                ex.map(lambda p: one(p, s), want), 1):
            if err:
                bad.append((pif, err, url))
            else:
                ok.append(pif)
                total += n
            if i % 50 == 0:
                print("  %4d/%d  %.2f GB  %.0fs" % (i, len(want), total / 1e9,
                                                    time.time() - t0), flush=True)
    print("done: %d ok, %d failed, %.2f GB, %.0f s"
          % (len(ok), len(bad), total / 1e9, time.time() - t0), flush=True)
    for b in bad[:20]:
        print("   FAIL", b, flush=True)

    files = sorted(f for f in os.listdir(DEST) if f.endswith(".fits.gz"))
    tot = sum(os.path.getsize(os.path.join(DEST, f)) for f in files)
    man = {
        "directory": "raw/manga/maps",
        "source_url_pattern": BASE,
        "retrieved_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "daptype": DAPTYPE,
        "drp_version": "v3_1_1", "dap_version": "3.1.0", "release": "SDSS DR17",
        "n_files": len(files), "total_bytes": tot,
        "n_requested": len(want), "n_failed": len(bad),
        "failures": [{"plateifu": b[0], "error": b[1], "url": b[2]} for b in bad],
        "selection": "union of the cluster and field members of every matched-pair "
                     "tier in clean/matched_pairs.csv",
        "contents": "DAP MAPS: STELLAR_VEL, STELLAR_SIGMA, EMLINE_GVEL, EMLINE_GSIGMA, "
                    "EMLINE_GFLUX, SPX_ELLCOO (elliptical polar radius), BINID and the "
                    "matching inverse-variance and mask extensions.",
        "files": [{"file": f, "bytes": os.path.getsize(os.path.join(DEST, f)),
                   "sha256": hashlib.sha256(
                       open(os.path.join(DEST, f), "rb").read()).hexdigest()}
                  for f in files],
    }
    mp = os.path.join(DEST, "maps.manifest.json")
    with open(mp, "w") as f:
        json.dump(man, f, indent=2)
    print("WROTE %s (%d files, %.2f GB)" % (mp, len(files), tot / 1e9), flush=True)


if __name__ == "__main__":
    main()
