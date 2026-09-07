"""write_report.py -- REPORT.md, spent.json and provenance_ledger.json.

SPEC.md is the specification BE.7 asked for.  REPORT.md is the record of the run
that produced it: what was probed, what came back, what was corrected, and what
this lane cannot tell you.

`spent.json` is the confirmation-status ledger.  `confirmation_status` v2 rules
that Mentioned, Acquired and Transformed are NOT spent and that Scored,
Inspected and Decision-used ARE.  This lane's whole point is to stay on the left
of that line, so the ledger states, per dataset, exactly which rung it reached.

    python write_report.py
"""
from __future__ import annotations

import io
import json
import os
import statistics
import sys

import archives
import guard

HERE = os.path.dirname(os.path.abspath(__file__))

LADDER = ["Mentioned", "Acquired", "Transformed",
          "Scored", "Inspected", "Decision-used"]
NOT_SPENT = LADDER[:3]

SPEND = [
    dict(dataset="eRASS1 primary cluster catalogue (Bulbul+2024)",
         rung="Acquired",
         what="the catalogue was downloaded, validated and cut to a candidate "
              "pool. No X-ray quantity was compared to any prediction.",
         spent=False),
    dict(dataset="DECADE metacal shear (DELVE DR3)",
         rung="Mentioned",
         what="COUNTED behind each candidate. No shape, response or per-source "
              "redshift was retrieved; guard G2 makes that mechanical.",
         spent=False),
    dict(dataset="DESI DR1 / SDSS DR17 spectroscopy",
         rung="Mentioned",
         what="COUNTED inside a redshift slab per candidate. No redshift value "
              "was retrieved.",
         spent=False),
    dict(dataset="DES Y3 metacalibration shape catalogue",
         rung="Mentioned",
         what="probed by HTTP header, magic bytes and a remote HDF5 open. Five "
              "group names were read. No row was read.",
         spent=False),
    dict(dataset="BUFFALO HLSP",
         rung="Mentioned",
         what="directory listings only.", spent=False),
    dict(dataset="KiDS, wide binaries",
         rung="Mentioned",
         what="PERMANENTLY SEALED. Named here so the report is complete; the "
              "guard raises before any path matching them can be opened.",
         spent=False),
    dict(dataset="SPT, X-GAP, CLoGS, MUSE/Granata, Gaia DR",
         rung="Mentioned",
         what="CONFIRMATION RESERVE. Recorded by identity only. The Granata "
              "member dispersions are the scarcest channel in the whole "
              "programme and were deliberately left unopened.",
         spent=False),
]


def main():
    led = guard.arm()
    probes = json.load(io.open(os.path.join(HERE, "probes.json"), encoding="utf-8"))
    rank = json.load(io.open(os.path.join(HERE, "ranking.json"), encoding="utf-8"))
    cl = rank["clusters"]

    probed = [c for c in cl if c["n_shear_sources"] is not None]
    covered = [c for c in probed if c["channels"]["C4_weak_lensing"] == "PUBLIC"]
    thin = [c for c in probed if c["channels"]["C4_weak_lensing"] == "THIN"]
    absent = [c for c in probed if c["channels"]["C4_weak_lensing"] == "ABSENT"]
    ns = sorted(c["n_shear_sources"] for c in covered) or [0]
    withspec = [c for c in covered if c["n_spec_members"]]

    # ---------------------------------------------------------------- spend
    spend = dict(lane="goldcluster", generated_utc=archives._utc(),
                 rule="confirmation_status v2",
                 ladder=LADDER, not_spent_rungs=NOT_SPENT,
                 datasets=SPEND,
                 anything_spent=any(d["spent"] for d in SPEND),
                 statement="Nothing in this lane advanced past Acquired. Every "
                           "candidate remains confirmation-grade.")
    guard.assert_no_statistic(spend, "spent")
    io.open(os.path.join(HERE, "spent.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(spend, indent=1) + "\n")

    # ----------------------------------------------------------- provenance
    prov = guard.summary()
    prov["generated_utc"] = archives._utc()
    prov["note"] = ("reads of '<fd>' are subprocess pipe file descriptors, not "
                    "paths; see guard._allow_file_descriptors.")
    io.open(os.path.join(HERE, "provenance_ledger.json"), "w", newline="\n",
            encoding="utf-8").write(json.dumps(prov, indent=1) + "\n")

    # --------------------------------------------------------------- report
    L = []
    A = L.append
    A("# Run BO -- the Gold Cluster Acquisition Specification")
    A("")
    A("Lane `work/wellnet-2026-09/goldcluster/`. Registry `BO-goldcluster`.")
    A("Rendered from `probes.json`, `ranking.json` and `spent.json`.")
    A("The specification itself is `SPEC.md`; this is the record of the run.")
    A("")
    A("## What was asked")
    A("")
    A("BE.7: write a Gold Cluster Acquisition Specification **before a target is")
    A("chosen, so candidates rank by how much new telescope time they need rather")
    A("than by how much code already exists for them.**")
    A("")
    A("## What was found")
    A("")
    A("**The programme's cluster sample can grow by about two orders of magnitude")
    A("without a single new observation.** Before this lane it had a public")
    A("per-source shear catalogue for exactly one cluster, Abell 370, and its")
    A("cluster-scale correlation rested on twelve X-COP systems. This lane probed")
    A("%d eRASS1 clusters one at a time and found **%d with a public raw"
      % (len(probed), len(covered)))
    A("weak-lensing channel** -- median %d usable background sources each,"
      % int(statistics.median(ns)))
    A("maximum %d." % ns[-1])
    A("")
    A("| | clusters |")
    A("|---|---|")
    A("| candidates after the declared cuts | %d |" % rank["n_candidates"])
    A("| probed for shear coverage | %d |" % len(probed))
    A("| public weak lensing (>= %d sources) | **%d** |"
      % (rank["thresholds"]["n_src_usable"], len(covered)))
    A("| thin (1 to %d sources) | %d |"
      % (rank["thresholds"]["n_src_usable"] - 1, len(thin)))
    A("| no DECADE coverage at all | %d |" % len(absent))
    A("| covered AND with spectroscopic members | %d |" % len(withspec))
    A("")
    A("## The three things this lane had to measure rather than assume")
    A("")
    A("**1. Footprint is not coverage.** DECADE is a reprocessing of archival")
    A("DECam pointings, so its sky coverage is patchy at the degree scale. The")
    A("single best candidate in the pool by X-ray counts has about a million")
    A("DECADE sources within 5 degrees and exactly zero within 0.5 -- a hole")
    A("several degrees across, mapped on a 7x7 grid to confirm it. Any method that")
    A("assigned coverage from a survey boundary would have called that cluster")
    A("covered. Coverage is therefore asked per cluster, one query each.")
    A("")
    A("**2. The DES Y3 shape catalogue is public but is not remotely subsettable.**")
    A("312 GB in one HDF5, no authentication, `Accept-Ranges: bytes` present, HDF5")
    A("magic verified, and h5py opens it remotely in about a second -- the five")
    A("metacalibration groups were read that way. But listing a group's datasets,")
    A("and even opening one named dataset, did not return within ten minutes,")
    A("because the metadata is scattered through 312 GB and every seek is a fresh")
    A("HTTP round trip. It is a bulk-download route, not a remote-subset route.")
    A("Recorded as measured, because the opposite would have been an easy and")
    A("expensive thing to assume.")
    A("")
    A("**3. BUFFALO's six-cluster shear release is still not public.** arXiv:")
    A("2602.06904 says the pyRRG catalogues will appear at the STScI HLSP on")
    A("acceptance. Re-checked live: the tree is unchanged since July 2023 and")
    A("only `abell370` carries a lensing directory.")
    A("")
    A("## Two defects this lane found in the programme's own machinery")
    A("")
    A("**The shared provenance guard breaks any lane that shells out.**")
    A("`OpenLedger.check` normalises whatever it is handed into a path string.")
    A("Handed the integer 3 -- which is what `subprocess.run(capture_output=True)`")
    A("passes to `open()` for its pipes -- it produces the path \"3\", finds it")
    A("outside the lane root and raises ForeignReadError. The synthetic lanes never")
    A("hit this because they never shell out. This lane calls curl, so it failed on")
    A("its first archive probe. Worked around locally in `guard._allow_file_")
    A("descriptors` rather than by editing `universes/provenance.py`, so the lanes")
    A("that hash that file keep their receipts. **The underlying defect is still")
    A("there for the next lane that shells out.**")
    A("")
    A("**The cluster-data lane's VizieR client is at the broken v2 rule.**")
    A("`cluster-data/scripts/vizier.py` computes the `#Name:` echo into a local")
    A("variable and then never compares it -- the identifier-echo detector, which")
    A("is D1 of `catalogue_validation` v3 and the one that catches a wrong-paper")
    A("serve, is dead code. It also never checks for `CatalogsExamined` and never")
    A("matches `#Title:`. `archives.py` in this lane implements all three; the")
    A("older client should be retired or fixed.")
    A("")
    A("## Nothing was spent")
    A("")
    A("`confirmation_status` v2: Mentioned, Acquired and Transformed are not spent;")
    A("Scored, Inspected and Decision-used are. This lane reached **Acquired** for")
    A("the eRASS1 catalogue and **Mentioned** for everything else. No shear, no")
    A("spectrum and no kinematic value was retrieved for any cluster.")
    A("")
    A("That is enforced rather than asserted. `guard.check_query` parses every")
    A("ADQL projection and refuses any that names a shape, ellipticity, response or")
    A("per-source redshift -- it refuses, for instance, the exact query the")
    A("eFEDS lane legitimately used. `guard.assert_no_statistic` refuses to write")
    A("any JSON whose keys collide with a gravity observable. Both were tested")
    A("against their own failure cases before the lane ran.")
    A("")
    A("## Why that matters more than the sample size")
    A("")
    A("The programme has no sealed confirmation set: KiDS and the wide binaries")
    A("were scored in round 1, so they are validation. These %d clusters are"
      % len(covered))
    A("pristine and independent of the spent data on all four of v2's axes --")
    A("outcome, objects, survey and reduction pipeline. **Splitting them and")
    A("sealing half, before anything is scored, is the recommendation of `SPEC.md`")
    A("and the most valuable thing available here.** Splitting after a first look")
    A("is how the last confirmation set was lost.")
    A("")
    A("## What this lane does not tell you")
    A("")
    A("- Strong lensing was not probed per cluster; C5 is UNMEASURED, not absent.")
    A("- C1 is inferred from the existence of shapes, not verified band by band.")
    A("- Spectroscopic completeness is counted, not characterised; DESI fibre")
    A("  assignment is not radially uniform.")
    A("- The 0.5 deg box is a coverage test, not the measurement aperture.")
    A("- **No cluster here has resolved member internal kinematics.** That channel")
    A("  is absent for all %d and is what a telescope proposal should ask for."
      % rank["n_candidates"])
    A("")
    md = "\n".join(L) + "\n"
    io.open(os.path.join(HERE, "REPORT.md"), "w", newline="\n",
            encoding="utf-8").write(md)
    print("wrote REPORT.md, spent.json, provenance_ledger.json")
    del led
    return 0


if __name__ == "__main__":
    sys.exit(main())
