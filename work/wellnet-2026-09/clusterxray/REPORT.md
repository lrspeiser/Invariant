# Run BR — Chandra event lists: the six clusters without the sphere

Lane `work/wellnet-2026-09/clusterxray/`. Registry `BR-clusterxray`.

## Why

Every thermal profile the programme held for these clusters —
`cluster-data/gas/accept_*.tsv` — is a **spherical deprojection**. The flat X-ray
image is inverted under an assumption of spherical symmetry, so the asymmetry is
gone before any number reaches the table. That is not a small loss for a
programme looking for a gravity law that might depend on local geometry, and the
repository's own gas QA already recorded the cost: Abell 2744's deprojected
density reverses 27 times in 58 bins, "close to noise-dominated", because a
sphere is the wrong model for a merging cluster.

Event lists sit upstream of that assumption. Each row is one photon with a sky
position and an energy.

## What was acquired

**27 ACIS observations, 246 MB, 3.3 million photons after filtering.**

| cluster | obs | photons | note |
|---|---|---|---|
| Abell 2029 | 5 | 1,736,562 | relaxed, strong cool core, z=0.077 |
| Abell 370 | 5 | 505,927 | merger, z=0.375 |
| MACS J1149 | 5 | 296,552 | merger, z=0.544 |
| MACS J0717 | 4 | 281,433 | quadruple merger, z=0.546 |
| Abell 2744 | 5 | 280,286 | major merger, z=0.308 |
| Abell S1063 | 3 | 230,841 | near-relaxed, z=0.348 |

Validated by three detectors, all required: bytes on disk equal Content-Length;
an EVENTS extension exists carrying X, Y and ENERGY; and the header's own OBS_ID
and INSTRUME match what was requested. Filename agreement is not identity — the
archive returns 200 for a wrongly assembled path.

The event files are **not committed** (246 MB). `raw_manifest.json` carries the
obsid, URL, byte count and SHA-256 of each, so they are re-fetchable exactly.
Same pattern as the >50 MB raw catalogues elsewhere in the programme.

## Two traps, both found the hard way

**The archive search silently ignores position.** `ocatList.do` accepts `ra`,
`dec` and `radius` parameters, ignores them, and returns HTTP 200 with a large,
correct-looking table of unrelated observations. A position search for Abell 370
returned engineering pointings at declination −80°. So `search.py` resolves by
NAME and then **rejects any observation whose own RA/Dec is more than 0.25° from
the cluster** — the position is the detector, never the search parameter. Same
failure shape as the VizieR fuzzy-fallback trap: a plausible payload under a
success code.

**The obvious thermal signal was the instrument.** The first hardness maps showed
a cool centre and hot outskirts in all six clusters, which looks like a cool core
everywhere. Measured against radius:

| cluster | 1′ | 3′ | 5′ | 7′ | 9′ | rise |
|---|---|---|---|---|---|---|
| Abell 2029 | 0.377 | 0.385 | 0.455 | 0.578 | 0.549 | +0.182 |
| Abell 370 | 0.535 | 0.640 | 0.649 | 0.669 | 0.671 | +0.082 |
| Abell 2744 | 0.547 | 0.589 | 0.662 | 0.682 | 0.679 | +0.112 |
| MACS J0717 | 0.437 | 0.552 | 0.619 | 0.617 | 0.615 | +0.122 |
| Abell S1063 | 0.486 | 0.553 | 0.626 | 0.638 | 0.653 | +0.125 |
| MACS J1149 | 0.509 | 0.583 | 0.655 | 0.662 | 0.670 | +0.120 |

**Mean rise +0.124, scatter between clusters only 0.030.** A relaxed cool-core
cluster at z=0.077 and a quadruple merger at z=0.546 cannot share a thermal
profile. That gradient is **ACIS vignetting** — energy dependent, so soft photons
lose more effective area off-axis and the ratio hardens outward whatever the gas
is doing. Correcting it properly needs exposure maps from CIAO and CALDB, neither
installed. It is removed empirically instead: vignetting is azimuthally
symmetric by construction, so subtracting each cluster's own smoothed radial
median removes it, along with any genuinely spherical thermal structure.

What survives is the **asymmetry** — how much hotter or cooler each direction is
than the average at the same radius. Which is precisely the quantity a spherical
deprojection cannot represent, and the reason this lane exists.

A third artefact was self-inflicted: the first de-radialisation used hard
annular bins and printed concentric rings across every map, which look exactly
like shock fronts. The profile is now built in fine annuli, smoothed, and
interpolated. **A correction that manufactures the signal it is looking for is
worse than no correction.**

## Asymmetry amplitude, after vignetting removal

| cluster | rms | note |
|---|---|---|
| MACS J0717 | 0.068 | quadruple merger — the most disturbed, as expected |
| Abell 2029 | 0.064 | relaxed, but has a known sloshing spiral |
| Abell S1063 | 0.052 | |
| Abell 370 | 0.046 | |
| MACS J1149 | 0.039 | |
| Abell 2744 | 0.037 | |

## What these maps are not

- **Not temperatures.** Hardness ratio is a proxy. Converting to keV needs an
  instrument response and a spectral fit per region. No CIAO, no CALDB.
- **Not comparable between clusters.** Redshifts run 0.077 to 0.546, so the same
  gas is seen through different spectral shift, and ACIS's effective area has
  degraded over the 24 years these observations span. Each map is scaled by its
  own percentiles for that reason.
- **Not point-source cleaned.** The small round blue and orange spots are AGN and
  foreground stars, not gas structure. Masking them needs a detection pass.
- **Not exposure corrected** beyond the empirical radial removal. Chip gaps and
  field edges produce the strong artefacts at the frame corners.
- **No gravity law is scored here.** This is acquisition and imaging.

## What it unlocks

A search that can see asymmetry at all. Every previous cluster input to this
programme was either a radial profile or a stacked mean, both of which are
azimuthally symmetric by construction — so no candidate law that depends on
direction, geometry or local structure could ever have been distinguished from
one that does not. That is now no longer true for six clusters.
