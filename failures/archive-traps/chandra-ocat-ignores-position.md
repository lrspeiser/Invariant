# Chandra ocatList.do

**Asked for:** observations near a sky position (ra, dec, radius parameters)

## What came back

HTTP 200 with the position parameters silently ignored. A search centred on Abell 370 at declination -1.6 deg returned engineering pointings at declination -80 deg. The table was large and correctly formed.

## The detector

Resolve targets by NAME, then reject any returned observation whose own RA/Dec lies more than a set tolerance from the target. The position is the detector, never the query parameter.

## Note

A later cross-match by eRASS1 designation returned zero matches for every cluster. Tested against Abell 2744's own J-name -- which has 98 ACIS observations -- it also returned zero. The name resolver does not handle those designations, so the zero was a broken search, not an absence. Fixed by downloading all 15,734 ACIS observations and matching locally, with two positive controls that must fire before any null is believed.

**Implemented in:** `work/wellnet-2026-09/clusterxray/search.py`
