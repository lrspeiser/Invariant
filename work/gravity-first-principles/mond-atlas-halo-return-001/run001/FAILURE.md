# Preserved ingestion failure

The first attempt passed all five mathematical benchmark tests and verified all
source hashes, then stopped at the parameter parser before any halo scoring.
The parser required finite reduced chi-square; the original archive contains
30 selected rows with `inf` reduced chi-square across ten galaxies. Their
profile parameters are finite. This was an overly broad parser requirement.

Run002 retains all 525 selected parameter rows and the original `inf` values,
with a separate nonfinite_published_chi flag. No fit-quality cut is introduced;
finite physical profile parameters remain mandatory. The shapes are calibration
targets, not claims that these poorly constrained fits describe real halos.
