# One post-hoc acceleration-scale repair

POST_HOC_DEVELOPMENT_ONLY. Parent run001 summary has been inspected; this is an explicit adjustment on the same historically exposed development galaxies, not independent confirmation. No new grid will be expanded after this run. Freeze this addendum before reading rotation responses in this branch.

Use the original source definitions, 139 registered identities, eligibility, 102 expected eligible galaxies, equal-galaxy loss, mass factors, strengths, lengths, clock factors, mixtures and three five-fold assignments. Reuse the original public sources and measurement paper (Lelli et al. 2016, https://arxiv.org/abs/1606.09251); no halo quantities. Original frozen config is configs/mond_atlas_clock_relay_v1.json. Candidate families: fixed/adjusted MOND and repaired clock, kernel, finite p2/p3/mixture only.

Define M=1e9(0.5 mf L3.6+1.33 MHI), GM=G M, rM=sqrt(GM/a0). Replace L=lambda Rd by L=lambda rM for kernel and all finite profiles. Clock inner softening stays Rd, but Psi0=lambda sqrt(GM a0). All other equations stay frozen. q remains the p3 mixture fraction. This changes source scaling, not per-galaxy fit freedom.

Tests before reading Vobs: zero strength (absolute logspeed 1e-12); analytical clock potential differentiation (relative 1e-6); point-kernel/finite formulas independently evaluated (relative 1e-10); rM scaling as sqrt(M), asymptotic flat clock v^4 proportional M for fixed lambda/beta; all candidates finite. The existing parent mechanics tests remain applicable. No 3D, clock-energy, lensing, time-history or energy-conservation claim.

Run CPU. Record exact code/config/source hashes before access, tests log, every candidate, every training fold loss, every family held radial prediction/residual, per-galaxy scores, choices and parameter boundary frequencies. Compare fixed and adjusted MOND plus original run001 summary. No bootstrap discovery significance. Any failure is saved and stops scoring; output directory cannot be overwritten.
