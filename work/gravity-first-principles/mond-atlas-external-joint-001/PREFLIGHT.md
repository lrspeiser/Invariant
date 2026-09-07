# Bounded joint external-field numerical endpoint

SOURCE_BLOCKED for observed environment claims; conditional numerical test only.
Read admission policy. Preserve all execution027 results. Fixed ell=0.5 kpc,
NGC2976 f4-stars-h0p4 source and unit applied normal field; no physics retuning.

Requested24kpc halfwidth finest grid has385^3=57,066,625 nodes,
456,533,000 bytes per full float64 array. A conservative20-array-equivalent
working allowance for density, epsilon, boundaries, harmonic faces, RHS/eigenvalue,
Krylov vectors, matvec and FFT scratch is9,130,660,000 bytes before small overhead.
This does not certify the required8,000,000,000-byte working cap. It is a
conservative allocation allowance, not a measured claim that every array is
simultaneously live. Physical available RAM (~75GB) does not override the cap.

Authorized bounded alternative selected before new solving: halfwidth(21,21,10.5)
at base spacing(0.25,0.25,0.125) and fine spacing(0.125,0.125,0.0625).
Fine shape337^3=38,272,753,306,182,024 bytes/full array;20-array allowance
6,123,640,480 bytes leaves room below8GB. Reuse prior fine18, base18 andbase24
samples with hashes. Run two new solves, not24fine. Compare fine18->fine21,
base21->fine21, and base21->base24. Same384 points plus origin; relative raw
and center-subtracted RMS<5%, eachheight<8%,1e-8 unit-field floor. No threshold
changes after results. Passing21kpc endpoint does not validate24fine.

Rerun analytic uniform/slab/polarity controls and bind existing independent
anisotropic sparse reference. Verify source hashes. CPU1thread, total300second
limit; sample process working memory at0.1second intervals and stop at callback/
stage boundary if sampled RSS exceeds8GB. Record actual sampled peak; it is
not a continuous-time maximum guarantee. Do not save full3D arrays. Preserve
budget/numerical failures rather than replacing cases or increasing resources.
