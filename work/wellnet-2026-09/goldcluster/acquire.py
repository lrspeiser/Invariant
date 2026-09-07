"""acquire.py -- fetch the candidate pool and cut it down to what is testable.

BE.7's instruction is that candidates must rank by how much NEW TELESCOPE TIME
they need, not by how much code already exists for them.  The seven clusters the
programme has worked on so far were chosen the second way -- they are the HFF
six plus A2029, i.e. the clusters with the most published products.  This module
starts from the other end: every X-ray-selected cluster in the sky, cut only by
what the measurement itself requires.

Source: eRASS1 primary cluster catalogue (Bulbul+2024, A&A 685, A106), VizieR
J/A+A/685/A106/emain, fetched under catalogue_validation v3.

The cuts, declared here before any coverage was probed:

    dec <= +40      DECam cannot reach further north, and DECam is the only
                    public per-source shear this programme can currently use
    0.05 <= z <= 0.7  below 0.05 the lensing kernel is negligible and the
                    cluster fills the field; above 0.7 the source density
                    behind it collapses
    CTS500 >= 100   fewer than ~100 X-ray counts inside R500 is not a profile,
                    it is a detection
    pcont <= 0.3    eRASS1's own contamination probability

Nothing here is a mass: the columns kept are count rates, counts, fluxes and kT.
M500, R500, Mgas500 and Fgas500 exist in the catalogue and are deliberately NOT
carried forward as observables -- they are scaling-relation products calibrated
on weak lensing, so scoring a gravity law against them would be circular.
R500 is kept for ONE purpose, bookkeeping the aperture, and is labelled as such.

    python acquire.py
"""
from __future__ import annotations

import io
import json
import os
import sys

import archives
import guard

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

SOURCE = "J/A+A/685/A106/emain"
WANT_TITLE = ("bulbul",)

DEC_MAX = 40.0
Z_MIN, Z_MAX = 0.05, 0.70
CTS_MIN = 100.0
PCONT_MAX = 0.3


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    guard.arm()
    os.makedirs(RAW, exist_ok=True)

    txt, rec = archives.vizier_fetch(SOURCE, want_title=WANT_TITLE,
                                     save_as="erass1_emain.tsv")
    print("eRASS1 fetched and validated v3: D1=%s D2=%s D3=%s rows=%d"
          % (rec["detectors"]["D1_name_echo"],
             rec["detectors"]["D2_no_catalogs_examined"],
             rec["detectors"]["D3_title_match"],
             rec["detectors"]["n_rows"]))

    cols, _units, rows = archives.parse_vizier_tsv(txt)
    ix = {c: k for k, c in enumerate(cols)}

    def g(r, c):
        k = ix[c]
        return r[k].strip() if k < len(r) else ""

    cand, rejected = [], dict(dec=0, z=0, counts=0, contaminated=0, incomplete=0)
    for r in rows:
        dec = _f(g(r, "DEJ2000"))
        z = _f(g(r, "zBest"))
        if dec is None or z is None:
            rejected["incomplete"] += 1
            continue
        if dec > DEC_MAX:
            rejected["dec"] += 1
            continue
        if not (Z_MIN <= z <= Z_MAX):
            rejected["z"] += 1
            continue
        cts = _f(g(r, "CTS500"), 0.0)
        if cts < CTS_MIN:
            rejected["counts"] += 1
            continue
        pc = _f(g(r, "pcont"), 1.0)
        if pc is not None and pc > PCONT_MAX:
            rejected["contaminated"] += 1
            continue
        cand.append(dict(
            name=g(r, "Name"), ra=_f(g(r, "RAJ2000")), dec=dec, z=z,
            zType=g(r, "zType"), cts500=cts, kt=g(r, "KT"),
            r500=_f(g(r, "R500")),          # bookkeeping aperture ONLY
            detlike=_f(g(r, "DetLike0")),
        ))

    cand.sort(key=lambda d: -d["cts500"])
    guard.assert_no_statistic({"n": len(cand)}, "acquire")
    io.open(os.path.join(RAW, "erass1_candidates.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(cand, indent=1) + "\n")

    spec = sum(1 for c in cand if "spec" in c["zType"])
    print("candidates=%d of %d  (spec-z %d, kT %d)"
          % (len(cand), len(rows), spec, sum(1 for c in cand if c["kt"])))
    print("rejected: %s" % rejected)
    return 0


if __name__ == "__main__":
    sys.exit(main())
