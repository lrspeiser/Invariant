"""rank.py -- rank candidates by HOW MUCH NEW TELESCOPE TIME THEY NEED.

This is the ordering BE.7 asks for, and it is deliberately not the ordering the
programme has used until now.  The seven clusters worked on so far were picked
because they had the most published products; that selects for clusters other
people have already spent time on, which is not the same as selecting for
clusters where the charter's complete experiment is cheapest to finish.

Each candidate is scored on the six channels.  A channel is PUBLIC if this lane
measured a public product covering that cluster, ABSENT if it measured its
absence, and UNMEASURED if the lane did not probe it (said so rather than
guessed).  The rank key is the count of PUBLIC channels, then the weak-lensing
source count, because C4 is the channel that cannot be substituted.

    C1  resolved baryons          DECam imaging underlying the shear catalogue
    C2  member internal dynamics  RESOLVED stellar kinematics -- see below
    C3  cluster dynamics          member redshifts (DESI DR1, SDSS DR17)
    C4  raw weak lensing          DECADE per-source metacalibration
    C5  strong lensing            multiple images / time delay
    C6  environment               the field around the cluster

C2 is scored ABSENT for every candidate and that is the honest answer, not a
gap in this lane's effort.  The only published resolved member kinematics for
any cluster in this programme's reach is the MUSE/Granata set, which is held in
the CONFIRMATION RESERVE and must not be opened.  So for every cluster here, C2
costs new IFU time.  That is exactly BE.7's point: the missing layer is the same
one everywhere, and it is what a proposal should ask for.

    python rank.py
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
RAW = os.path.join(HERE, "raw")
COV = os.path.join(RAW, "decade_coverage.jsonl")
SPEC = os.path.join(RAW, "spec_coverage.jsonl")

#: how many of the best-covered candidates get their spectroscopy probed.
#: None means all of them -- at ~25 clusters a minute the whole covered set is
#: about 25 minutes, which is cheaper than reasoning about a subsample.
N_SPEC_PROBE = None
#: a weak-lensing channel below this is not a profile, it is a handful of shapes
N_SRC_USABLE = 2000
#: a spectroscopic channel needs enough members for a sigma(R) field
N_SPEC_USABLE = 50


def load_coverage():
    recs = []
    if os.path.exists(COV):
        for line in io.open(COV, encoding="utf-8"):
            if line.strip():
                recs.append(json.loads(line))
    return recs


def probe_spectroscopy(cands):
    """C3 -- member redshifts from DESI DR1 and SDSS DR17, counted not opened."""
    done = {}
    if os.path.exists(SPEC):
        for line in io.open(SPEC, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[r["name"]] = r
    fh = io.open(SPEC, "a", newline="\n", encoding="utf-8")
    t0 = time.time()
    for i, c in enumerate(cands, 1):
        if c["name"] in done:
            continue
        # 0.5 deg box, and a generous +-0.02 redshift slab about the cluster
        ra, dec, z = c["ra"], c["dec"], c["z"]
        box = ("%s BETWEEN %.5f AND %.5f AND %s BETWEEN %.5f AND %.5f"
               % ("%s", ra - 0.5, ra + 0.5, "%s", dec - 0.5, dec + 0.5))
        rec = dict(name=c["name"])
        try:
            q = ("SELECT COUNT(*) AS n FROM desi_dr1.zpix WHERE "
                 + (box % ("mean_fiber_ra", "mean_fiber_dec"))
                 + " AND z BETWEEN %.5f AND %.5f AND zwarn = 0"
                 % (z - 0.02, z + 0.02))
            rec["desi_dr1"] = int(archives.tap_count(q).strip().splitlines()[-1])
        except Exception as exc:                                # noqa: BLE001
            rec["desi_dr1"] = None
            rec["desi_error"] = str(exc)[:150]
        try:
            q = ("SELECT COUNT(*) AS n FROM sdss_dr17.specobj WHERE "
                 + (box % ("ra", "dec"))
                 + " AND z BETWEEN %.5f AND %.5f AND zwarning = 0"
                 % (z - 0.02, z + 0.02))
            rec["sdss_dr17"] = int(archives.tap_count(q).strip().splitlines()[-1])
        except Exception as exc:                                # noqa: BLE001
            rec["sdss_dr17"] = None
            rec["sdss_error"] = str(exc)[:150]
        guard.assert_no_statistic(rec, "spec")
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        done[c["name"]] = rec
        if i % 25 == 0:
            print("   spec %d/%d  %.0f min"
                  % (i, len(cands), (time.time() - t0) / 60), flush=True)
    fh.close()
    return done


def score(c, spec):
    """Return the six-channel verdict for one candidate."""
    n_src = c.get("n_sources_box")
    s = spec.get(c["name"], {})
    n_spec = max((s.get("desi_dr1") or 0), (s.get("sdss_dr17") or 0))

    ch = {}
    # C4 -- the channel that cannot be substituted
    if n_src is None:
        ch["C4_weak_lensing"] = "UNMEASURED"
    elif n_src >= N_SRC_USABLE:
        ch["C4_weak_lensing"] = "PUBLIC"
    elif n_src > 0:
        ch["C4_weak_lensing"] = "THIN"
    else:
        ch["C4_weak_lensing"] = "ABSENT"

    # C1 -- the same DECam imaging that produced the shapes
    ch["C1_resolved_baryons"] = ("PUBLIC" if ch["C4_weak_lensing"] == "PUBLIC"
                                 else "ABSENT")

    # C3 -- member redshifts
    if c["name"] not in spec:
        ch["C3_cluster_dynamics"] = "UNMEASURED"
    elif n_spec >= N_SPEC_USABLE:
        ch["C3_cluster_dynamics"] = "PUBLIC"
    elif n_spec > 0:
        ch["C3_cluster_dynamics"] = "THIN"
    else:
        ch["C3_cluster_dynamics"] = "ABSENT"

    # C2 -- resolved member internal kinematics
    ch["C2_member_internal_dynamics"] = "ABSENT"

    # C5 -- strong lensing: not probed per cluster by this lane
    ch["C5_strong_lensing"] = "UNMEASURED"

    # C6 -- environment: the spectroscopy that gives C3 also gives the field
    ch["C6_environment"] = ch["C3_cluster_dynamics"]

    n_public = sum(1 for v in ch.values() if v == "PUBLIC")
    return ch, n_public, n_spec


def main():
    guard.arm()
    cov = load_coverage()
    covered = [c for c in cov if (c.get("n_sources_box") or 0) >= N_SRC_USABLE]
    covered.sort(key=lambda c: -(c["n_sources_box"]))
    print("coverage rows=%d  with >=%d sources=%d"
          % (len(cov), N_SRC_USABLE, len(covered)), flush=True)

    target = covered if N_SPEC_PROBE is None else covered[:N_SPEC_PROBE]
    print("probing spectroscopy for the top %d ..." % len(target), flush=True)
    spec = probe_spectroscopy(target)

    ranked = []
    for c in cov:
        ch, n_public, n_spec = score(c, spec)
        ranked.append(dict(
            name=c["name"], ra=c["ra"], dec=c["dec"], z=c["z"],
            zType=c["zType"], cts500=c["cts500"], kt=c["kt"],
            n_shear_sources=c.get("n_sources_box"),
            n_spec_members=n_spec if c["name"] in spec else None,
            channels=ch, n_channels_public=n_public,
            channels_needing_new_time=[k for k, v in ch.items()
                                       if v in ("ABSENT", "THIN")],
        ))
    ranked.sort(key=lambda r: (-r["n_channels_public"],
                               -(r["n_shear_sources"] or 0)))

    out = dict(
        lane="goldcluster",
        generated_utc=archives._utc(),
        inventory_only=True,
        rank_key="channels already public, then weak-lensing source count",
        thresholds=dict(n_src_usable=N_SRC_USABLE, n_spec_usable=N_SPEC_USABLE,
                        n_spec_probed=len(spec)),
        n_candidates=len(ranked),
        n_with_public_weak_lensing=sum(
            1 for r in ranked if r["channels"]["C4_weak_lensing"] == "PUBLIC"),
        clusters=ranked,
    )
    guard.assert_no_statistic(out, "ranking")
    io.open(os.path.join(HERE, "ranking.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(out, indent=1) + "\n")
    print("\nwrote ranking.json: %d candidates, %d with public weak lensing"
          % (out["n_candidates"], out["n_with_public_weak_lensing"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
