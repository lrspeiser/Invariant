"""coverage.py -- how many usable background sources sit behind each candidate.

This is the measurement BE.7 asks for.  A cluster's weak-lensing channel is not
"available" because it lies inside a survey's nominal boundary; DECADE is a
reprocessing of ARCHIVAL DECam pointings, so its footprint is patchy at the
degree scale.  The top eRASS1 cluster by X-ray counts has ~1 million DECADE
sources within 5 degrees and exactly ZERO within 0.5 -- a hole several degrees
across.  So coverage is asked per cluster, one query each, and the answer is
recorded whatever it is.

What is asked: COUNT(*) of sources that pass the DECADE cosmology selection and
lie behind the cluster in redshift, inside a box of +-0.5 deg.  What is NOT
asked: any shape, any ellipticity, any response, any per-source redshift.
guard.check_query enforces that distinction, so "no shear was opened" is a
property of the code and not a promise in a comment.

Resumable: one JSON line per cluster, appended and flushed; a re-run skips the
clusters already recorded.

    python coverage.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import archives
import guard

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "raw", "erass1_candidates.json")
OUT = os.path.join(HERE, "raw", "decade_coverage.jsonl")

HALF = 0.5          # deg; >= 2 Mpc at z > 0.08, and it bounds the query
DZ = 0.20           # background margin, as Chiu+2022 and Umetsu+2020 use

SEL = ("mcal_flags = 0 AND flags_foreground = 0 AND flags_footprint = 1 "
       "AND mcal_sel_noshear > 0 AND dnf_z > 0 AND dnf_z < 3")


def build_query(ra, dec, z):
    lo, hi = ra - HALF, ra + HALF
    if lo < 0 or hi > 360:                      # RA wrap
        where_ra = ("(ra BETWEEN %.5f AND 360.0 OR ra BETWEEN 0.0 AND %.5f)"
                    % (lo % 360, hi % 360))
    else:
        where_ra = "ra BETWEEN %.5f AND %.5f" % (lo, hi)
    return ("SELECT COUNT(*) AS n FROM delve_dr3.decade_shear WHERE %s AND %s "
            "AND dec BETWEEN %.5f AND %.5f AND dnf_z > %.4f"
            % (SEL, where_ra, dec - HALF, dec + HALF, z + DZ))


def main():
    guard.arm()
    cand = json.load(io.open(CAND, encoding="utf-8"))

    done = set()
    if os.path.exists(OUT):
        for line in io.open(OUT, encoding="utf-8"):
            if line.strip():
                try:
                    done.add(json.loads(line)["name"])
                except Exception:               # noqa: BLE001
                    pass
    todo = [c for c in cand if c["name"] not in done]
    print("candidates=%d  already recorded=%d  to probe=%d"
          % (len(cand), len(done), len(todo)), flush=True)

    fh = io.open(OUT, "a", newline="\n", encoding="utf-8")
    t0 = time.time()
    for i, c in enumerate(todo, 1):
        rec = dict(c)
        try:
            txt = archives.tap_count(build_query(c["ra"], c["dec"], c["z"]))
            rec["n_sources_box"] = int(txt.strip().splitlines()[-1])
            rec["error"] = None
        except Exception as exc:                # noqa: BLE001
            rec["n_sources_box"] = None
            rec["error"] = str(exc)[:200]
        rec["box_halfwidth_deg"] = HALF
        rec["background_dz"] = DZ
        guard.assert_no_statistic(rec, "coverage")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if i % 50 == 0:
            print("  %d/%d  %.0f min" % (i, len(todo), (time.time() - t0) / 60),
                  flush=True)
    fh.close()
    print("done: %d probed in %.0f min" % (len(todo), (time.time() - t0) / 60))
    return 0


if __name__ == "__main__":
    sys.exit(main())
