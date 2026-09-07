# Chandra ACIS event lists

**Asked for:** a gas temperature map

## What came back

An apparent cool core in every one of six clusters, relaxed and merging alike. Hardness rose +0.124 from centre to 8 arcmin with only 0.030 scatter BETWEEN clusters -- the same profile in a relaxed cool-core cluster at z=0.077 as in a quadruple merger at z=0.546.

## The detector

Two clusters of different dynamical state cannot share a thermal profile. When they appear to, the profile belongs to the instrument. The cause is energy-dependent vignetting: soft photons lose more effective area off-axis, so the band ratio hardens outward whatever the gas is doing.

## Note

Corrected properly with exposure maps from CIAO and CALDB. Corrected here empirically, by removing each cluster's own smoothed radial median -- which is valid only because vignetting is azimuthally symmetric by construction, and which removes any genuinely spherical thermal structure along with it.

**Implemented in:** `work/wellnet-2026-09/clusterxray/maps.py`
