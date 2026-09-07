# Run BO -- the Gold Cluster Acquisition Specification

Lane `work/wellnet-2026-09/goldcluster/`. Registry `BO-goldcluster`.
Rendered from `probes.json`, `ranking.json` and `spent.json`.
The specification itself is `SPEC.md`; this is the record of the run.

## What was asked

BE.7: write a Gold Cluster Acquisition Specification **before a target is
chosen, so candidates rank by how much new telescope time they need rather
than by how much code already exists for them.**

## What was found

**The programme's cluster sample can grow by about two orders of magnitude
without a single new observation.** Before this lane it had a public
per-source shear catalogue for exactly one cluster, Abell 370, and its
cluster-scale correlation rested on twelve X-COP systems. This lane probed
1863 eRASS1 clusters one at a time and found **605 with a public raw
weak-lensing channel** -- median 13134 usable background sources each,
maximum 29920.

| | clusters |
|---|---|
| candidates after the declared cuts | 1863 |
| probed for shear coverage | 1863 |
| public weak lensing (>= 2000 sources) | **605** |
| thin (1 to 1999 sources) | 78 |
| no DECADE coverage at all | 1180 |
| covered AND with spectroscopic members | 251 |

## The three things this lane had to measure rather than assume

**1. Footprint is not coverage.** DECADE is a reprocessing of archival
DECam pointings, so its sky coverage is patchy at the degree scale. The
single best candidate in the pool by X-ray counts has about a million
DECADE sources within 5 degrees and exactly zero within 0.5 -- a hole
several degrees across, mapped on a 7x7 grid to confirm it. Any method that
assigned coverage from a survey boundary would have called that cluster
covered. Coverage is therefore asked per cluster, one query each.

**2. The DES Y3 shape catalogue is public but is not remotely subsettable.**
312 GB in one HDF5, no authentication, `Accept-Ranges: bytes` present, HDF5
magic verified, and h5py opens it remotely in about a second -- the five
metacalibration groups were read that way. But listing a group's datasets,
and even opening one named dataset, did not return within ten minutes,
because the metadata is scattered through 312 GB and every seek is a fresh
HTTP round trip. It is a bulk-download route, not a remote-subset route.
Recorded as measured, because the opposite would have been an easy and
expensive thing to assume.

**3. BUFFALO's six-cluster shear release is still not public.** arXiv:
2602.06904 says the pyRRG catalogues will appear at the STScI HLSP on
acceptance. Re-checked live: the tree is unchanged since July 2023 and
only `abell370` carries a lensing directory.

## Two defects this lane found in the programme's own machinery

**The shared provenance guard breaks any lane that shells out.**
`OpenLedger.check` normalises whatever it is handed into a path string.
Handed the integer 3 -- which is what `subprocess.run(capture_output=True)`
passes to `open()` for its pipes -- it produces the path "3", finds it
outside the lane root and raises ForeignReadError. The synthetic lanes never
hit this because they never shell out. This lane calls curl, so it failed on
its first archive probe. Worked around locally in `guard._allow_file_
descriptors` rather than by editing `universes/provenance.py`, so the lanes
that hash that file keep their receipts. **The underlying defect is still
there for the next lane that shells out.**

**The cluster-data lane's VizieR client is at the broken v2 rule.**
`cluster-data/scripts/vizier.py` computes the `#Name:` echo into a local
variable and then never compares it -- the identifier-echo detector, which
is D1 of `catalogue_validation` v3 and the one that catches a wrong-paper
serve, is dead code. It also never checks for `CatalogsExamined` and never
matches `#Title:`. `archives.py` in this lane implements all three; the
older client should be retired or fixed.

## Nothing was spent

`confirmation_status` v2: Mentioned, Acquired and Transformed are not spent;
Scored, Inspected and Decision-used are. This lane reached **Acquired** for
the eRASS1 catalogue and **Mentioned** for everything else. No shear, no
spectrum and no kinematic value was retrieved for any cluster.

That is enforced rather than asserted. `guard.check_query` parses every
ADQL projection and refuses any that names a shape, ellipticity, response or
per-source redshift -- it refuses, for instance, the exact query the
eFEDS lane legitimately used. `guard.assert_no_statistic` refuses to write
any JSON whose keys collide with a gravity observable. Both were tested
against their own failure cases before the lane ran.

## Why that matters more than the sample size

The programme has no sealed confirmation set: KiDS and the wide binaries
were scored in round 1, so they are validation. These 605 clusters are
pristine and independent of the spent data on all four of v2's axes --
outcome, objects, survey and reduction pipeline. **Splitting them and
sealing half, before anything is scored, is the recommendation of `SPEC.md`
and the most valuable thing available here.** Splitting after a first look
is how the last confirmation set was lost.

## What this lane does not tell you

- Strong lensing was not probed per cluster; C5 is UNMEASURED, not absent.
- C1 is inferred from the existence of shapes, not verified band by band.
- Spectroscopic completeness is counted, not characterised; DESI fibre
  assignment is not radially uniform.
- The 0.5 deg box is a coverage test, not the measurement aperture.
- **No cluster here has resolved member internal kinematics.** That channel
  is absent for all 1863 and is what a telescope proposal should ask for.

