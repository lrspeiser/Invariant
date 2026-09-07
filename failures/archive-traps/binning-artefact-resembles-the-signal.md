# the correction itself

**Asked for:** remove the radial gradient above

## What came back

Concentric rings across every map, one at each annular bin edge. Rings in a cluster thermal map resemble shock fronts, which are exactly what such a map is searched for.

## The detector

Build the radial profile in fine bins, smooth it, then interpolate, rather than subtracting bin by bin.

## Note

The most dangerous class in this list, because the artefact resembles the target. A correction must not manufacture the feature it is correcting for.

**Implemented in:** `work/wellnet-2026-09/clusterxray/maps.py`
