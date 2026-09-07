# VizieR asu-tsv

**Asked for:** a specific catalogue identifier

## What came back

HTTP 200 and a DIFFERENT, REAL catalogue -- a near-infrared young stellar object survey (J/MNRAS/430/1125) served in place of a cluster redshift catalogue. Twelve identifiers were affected in one lane. In another run the payload was the wrong PAPER entirely, and in a third the response carried CatalogsExamined=10213 after a fuzzy fallback.

## The detector

Three detectors, ALL required, none sufficient alone. D1: the #Name: line must echo the EXACT identifier requested. D2: refuse any payload containing CatalogsExamined. D3: #Title: must match the expected author and year.

## Note

A zero-row response is not an absence until D1-D3 have passed. This programme's own client asserted the D1 check in its docstring and never performed it -- the echo was computed into a variable and discarded.

**Implemented in:** `work/wellnet-2026-09/goldcluster/archives.py`
