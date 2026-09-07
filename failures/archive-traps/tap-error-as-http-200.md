# IVOA TAP endpoints (NOIRLab Astro Data Lab)

**Asked for:** an ADQL query

## What came back

HTTP 200 wrapping a VOTable whose QUERY_STATUS is ERROR. Parsed as CSV it reads as zero rows -- a confident false negative. Triggered here by CAST(FLOOR(x) AS INTEGER) and by GROUP BY on an expression, both of which the ADQL parser rejects; and separately by a whole-table COUNT, which returns HTTP 504 after several minutes.

## The detector

Test for a leading <?xml envelope before parsing anything as data.

## Note

To sample a very large table without a full scan, use MOD(id, N) = 0 rather than TOP N, which returns a contiguous block, not a sample.

**Implemented in:** `work/wellnet-2026-09/goldcluster/archives.py`
