"""fetch.py -- Chandra ACIS level-2 event lists for the six clusters.

WHY THIS LANE EXISTS. Every thermal profile the programme holds for these
clusters (`cluster-data/gas/accept_*.tsv`) is a SPHERICAL DEPROJECTION: the flat
X-ray image inverted under an assumption of spherical symmetry. That assumption
is imposed before any number reaches the table, so the asymmetry -- shock fronts,
cold fronts, the two mass peaks of Abell 2744, the four subclusters of MACS
J0717 -- is gone by construction. The repository's own gas QA records the
consequence: Abell 2744's deprojected density reverses 27 times in 58 bins,
"close to noise-dominated", because a sphere is the wrong model for a merger.

Event lists are upstream of that. Each row is one detected photon with a sky
position and an energy, so a map can be built on a GRID instead of a radius, and
asymmetry survives.

VALIDATION, three detectors, all required -- the same shape as
`voidcmb/fetch_planck.py` because a plain HTTP FITS fetch has the same failure
modes:

  D1  TRANSPORT   bytes on disk == Content-Length, file non-empty.
  D2  STRUCTURE   astropy opens it, an EVENTS extension exists, and it carries
                  the sky and energy columns a map actually needs.
  D3  IDENTITY    the header's OBS_ID equals the one requested AND INSTRUME is
                  ACIS. Filename agreement is not identity: the archive will
                  serve a 200 for a path assembled wrongly.

A DETECTOR THIS LANE ADDED. The Ocat search endpoint accepts `ra`/`dec`/`radius`
parameters, ignores them silently, and returns HTTP 200 with a large, correct-
looking table of unrelated engineering observations. Searching for Abell 370 by
position returned pointings at -80 deg. So `search.py` resolves targets by NAME
and then REJECTS any observation whose own RA/Dec is more than 0.25 deg from the
cluster -- the position is the detector, never the search parameter.

    python fetch.py
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")
OBS = os.path.join(HERE, os.environ.get("OBS_FILE", "observations.json"))
UA = "gravity-clusterxray/1.0 (research)"
CDA = "https://cxc.cfa.harvard.edu/cdaftp/byobsid"

#: deepest N observations per cluster -- enough for a hardness map, and it keeps
#: the download to a few hundred MB rather than the 5.4 Ms that exist
PER_CLUSTER = int(os.environ.get("PER_CLUSTER", "5"))
MIN_KS = float(os.environ.get("MIN_KS", "15.0"))

FITS_MAGIC = b"SIMPLE  ="
GZ_MAGIC = b"\x1f\x8b"


class ValidationFailure(RuntimeError):
    pass


def curl(url, out=None, timeout=900):
    cmd = ["curl", "-s", "--max-time", str(timeout), "-A", UA]
    if out:
        cmd += ["-o", out]
    cmd.append(url)
    p = subprocess.run(cmd, capture_output=not out)
    if p.returncode != 0:
        raise RuntimeError("curl exit %d for %s" % (p.returncode, url))
    return None if out else p.stdout.decode("utf-8", "replace")


def content_length(url):
    p = subprocess.run(["curl", "-sI", "--max-time", "120", "-A", UA, url],
                       capture_output=True)
    m = re.search(r"(?im)^content-length:\s*(\d+)", p.stdout.decode("utf-8", "replace"))
    return int(m.group(1)) if m else None


def evt2_url(obsid):
    """Find the evt2 file inside the observation's primary directory."""
    base = "%s/%s/%s/primary/" % (CDA, obsid[-1], obsid)
    html = curl(base)
    m = re.search(r'href="([^"]*evt2[^"]*\.fits(?:\.gz)?)"', html)
    if not m:
        raise ValidationFailure("no evt2 in %s" % base)
    return base + m.group(1), m.group(1)


def validate(path, obsid):
    """D1 already checked by the caller. D2 structure and D3 identity here."""
    from astropy.io import fits

    with open(path, "rb") as fh:
        head = fh.read(2)
    if head != GZ_MAGIC:
        with open(path, "rb") as fh:
            if fh.read(9) != FITS_MAGIC:
                raise ValidationFailure("neither gzip nor FITS magic: %s" % path)

    with fits.open(path, memmap=False) as hdul:
        ev = None
        for h in hdul:
            if (h.name or "").upper() == "EVENTS":
                ev = h
                break
        if ev is None:
            raise ValidationFailure("D2: no EVENTS extension in %s" % path)
        cols = {c.upper() for c in ev.columns.names}
        need = {"X", "Y", "ENERGY"}
        missing = need - cols
        if missing:
            raise ValidationFailure("D2: EVENTS lacks %s in %s" % (missing, path))

        hdr = ev.header
        got = str(hdr.get("OBS_ID", "")).strip()
        if got != str(obsid):
            raise ValidationFailure(
                "D3: header OBS_ID %r != requested %r -- the archive served a "
                "different observation" % (got, obsid))
        instr = str(hdr.get("INSTRUME", "")).upper()
        if "ACIS" not in instr:
            raise ValidationFailure("D3: INSTRUME %r is not ACIS" % instr)
        return dict(n_events=int(ev.header.get("NAXIS2", 0)),
                    obs_id=got, instrume=instr,
                    object=str(hdr.get("OBJECT", "")).strip(),
                    exposure_s=float(hdr.get("EXPOSURE", 0.0)),
                    date_obs=str(hdr.get("DATE-OBS", "")),
                    ra_nom=float(hdr.get("RA_NOM", float("nan"))),
                    dec_nom=float(hdr.get("DEC_NOM", float("nan"))))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    obs = json.load(io.open(OBS, encoding="utf-8"))
    os.makedirs(RAW, exist_ok=True)
    manifest_path = os.path.join(HERE, os.environ.get("MANIFEST_FILE", "raw_manifest.json"))
    manifest = {}
    if os.path.exists(manifest_path):
        manifest = json.load(io.open(manifest_path, encoding="utf-8"))

    total = 0
    for cluster, lst in obs.items():
        picks = [o for o in lst if o["exposure_ks"] >= MIN_KS][:PER_CLUSTER]
        print("\n%s -- %d observations, %.0f ks"
              % (cluster, len(picks), sum(o["exposure_ks"] for o in picks)), flush=True)
        for o in picks:
            oid = o["obsid"]
            key = "%s/%s" % (cluster, oid)
            if key in manifest and os.path.exists(
                    os.path.join(RAW, manifest[key]["file"])):
                print("   obsid %-7s cached" % oid, flush=True)
                continue
            try:
                url, fname = evt2_url(oid)
                want = content_length(url)
                dest = os.path.join(RAW, "%s_%s" % (cluster, fname))
                t0 = time.time()
                curl(url, out=dest)
                got = os.path.getsize(dest)
                if want and got != want:            # D1
                    raise ValidationFailure(
                        "D1: got %d bytes, Content-Length said %d" % (got, want))
                info = validate(dest, oid)          # D2 + D3
                manifest[key] = dict(
                    cluster=cluster, obsid=oid, url=url,
                    file=os.path.basename(dest), bytes=got,
                    sha256=sha256(dest),
                    retrieved_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    validated=True, **info)
                total += got
                print("   obsid %-7s %-34s %5.1f MB  %8d events  %.0fs"
                      % (oid, info["object"][:34], got / 1048576,
                         info["n_events"], time.time() - t0), flush=True)
            except Exception as exc:                # noqa: BLE001
                print("   obsid %-7s FAILED: %s" % (oid, str(exc)[:120]), flush=True)
        io.open(manifest_path, "w", newline="\n", encoding="utf-8").write(
            json.dumps(manifest, indent=1) + "\n")

    print("\n%d files, %.0f MB total" % (len(manifest), total / 1048576))
    return 0


if __name__ == "__main__":
    sys.exit(main())
