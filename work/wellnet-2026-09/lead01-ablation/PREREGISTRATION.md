# Pre-registration — lead01-ablation
Written 2026-09-04, BEFORE any of the four tests below were computed.
Lane: `work/wellnet-2026-09/lead01-ablation/`.
Reused input: `work/wellnet-2026-09/potential-depth/potential_depth_ladder.csv`
(4,150 rows, 317 systems, sha256 recorded in `ablation.json`). The ladder is NOT
rebuilt.

## Declarations made in advance

### D1 — the ablation split (item 1)
Three training sets, one held-out set, evaluated with frozen coefficients once
each:

    A   train on class_rank == 1 (SPARC field galaxies)      -> predict rank 2-4 AND rank 5-6
    B   train on class_rank in {2,3,4} (groups)              -> predict rank 5-6
    C   train on class_rank in {1,2,3,4} (galaxies + groups) -> predict rank 5-6   [published]

Held-out set for the cluster comparison is rank >= 5 in every arm, identical
rows, so the three arms are directly comparable. Response variable, window,
system-level aggregation and the free quadratic in log g_bar are all taken
UNCHANGED from `potential-depth/code/analyse.py` section 8. Success metric:
frozen-coefficient RMS of log10(nu_obs/nu_RAR) on the held-out systems.

### D2 — the paired bootstrap (item 2)
Primary statistic: `DeltaRMS = RMS(M1) - RMS(M3)` on the held-out clusters, both
models frozen on the arm-C training set. Uncertainty by resampling the held-out
SYSTEMS with replacement, 20,000 draws, coefficients held frozen. Secondary:
the same with the training set resampled too (nested bootstrap), so the frozen
coefficients also vary. Reported quantities: the full paired difference
distribution, its 95% interval, and P(M1 better). A two-sided paired test on
per-object squared error is reported alongside. No model is declared better on
a point estimate.

### D3 — boundary rules (item 3), PRIMARY DECLARED HERE IN ADVANCE
The variable is the potential DIFFERENCE

    DeltaPhi_b(r; r_ref) = Int_r^r_ref g_b(s) ds

evaluated under four reference rules. **The PRIMARY rule is `BARY`.** This is
declared now, before any of the four has been computed, and it is primary
because it is the only one of the four that is (a) free of dark matter, (b)
scales with the object rather than with the instrument or with an arbitrary
absolute length, and (c) finite, so the variable does not depend on an
unmeasured tail.

    BARY  (PRIMARY)  r_ref = 10 * r_half,b, where r_half,b is the radius
                     enclosing half of the system's baryonic mass inside its
                     outermost measured radius.
    PHYS             r_ref = 2000 kpc for every system (fixed physical radius,
                     chosen to exceed the largest measured radius in the ladder).
    OVER             r_ref = r_200b, the radius at which the mean enclosed
                     BARYONIC density equals 200 rho_c(z). No dark matter enters.
    TAIL             r_ref -> infinity with the point-mass tail beyond the last
                     measured radius. This is the existing convention and
                     reproduces the published |Phi_b| column exactly.

Reported for each rule: beta fitted on rungs 1-4, the frozen transfer RMS on
rungs 5-6, and the spread across the four. Rows with r >= r_ref are reported as
a count, not silently dropped.

### D4 — the fresh sample (item 4)
The existing held-out clusters have had four models ranked on them and are
therefore validation data, not a holdout. A fresh sample is acquired by a
process that never computes a residual, a boost, or an acceleration ratio; the
primary model is frozen on the existing ladder FIRST; the fresh sample is then
scored ONCE. The seal timestamp, the sample identity and the frozen coefficient
values are written to `ablation.json` before the fresh sample is scored.

## Failure modes to be checked explicitly (standing brief)
shared-denominator artefacts simulated with the actual error covariance;
monotone-invariance gate d(beta)/d(q) verified numerically over a range;
no refitting on any held-out set; row and column counts asserted on every
ingest; sealed holdouts (KiDS, wide binaries) never loaded; no NFW-fitted or
lens-model masses used as observations.
