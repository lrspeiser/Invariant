"""Emit REPORT.md for the member-dynamics lane."""
import os

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\member-dynamics"

REPORT = r"""# member-dynamics lane -- report

Lane: `work/wellnet-2026-09/member-dynamics/`
Brief: `work/wellnet-2026-09/BRIEF.md` including the 2026-09-04 addendum.
Data: derived entirely from `work/wellnet-2026-09/env-data/` (330 verified
manifests). Nothing was downloaded by this lane.

**Sealed holdouts: KiDS and wide binaries were never loaded, listed, queried or
referenced.**

---

## 1. Power, stated before the answer

`power_prestatement.json` was written and closed before any field-minus-cluster
offset was evaluated. The width of the blocked sign-flip null depends only on
the magnitudes `|dY_i|` of the paired differences and on the host-system block
structure; it carries no information about their signs, so it can be quoted
first without breaking blindness.

| sample | pairs used / declared | host systems (effective) | sd of the estimator, log V | 3-sigma minimum detectable, log V | power vs H1 |
|---|---|---|---|---|---|
| MaNGA `B1_primary` | 10 / 23 | 5 (3.1) | 0.0793 | 0.238 | 0.05 |
| MaNGA `B2_disk_strict` | 84 / 121 | 26 (3.7) | 0.0117 | 0.035 | 0.26 |
| MaNGA `B3_late_wide` | 45 / 80 | 28 (16.2) | 0.0171 | 0.051 | 0.15 |
| **MaNGA `B4_disk_wide`** | **190 / 281** | **83 (9.7)** | **0.0069** | **0.021** | **0.61** |
| MaNGA `C1_xray_late` | 29 / 61 | 15 (10.1) | 0.0233 | 0.070 | 0.10 |
| MaNGA `C2_xray_disk` | 143 / 218 | 36 (5.5) | 0.0100 | 0.030 | 0.34 |
| SAMI `S1_latetype` | 58 / 108 | 7 (5.0) | 0.0133 | 0.040 | 0.22 |
| **SAMI `S2_diskbearing`** | **210 / 364** | **8 (5.8)** | **0.0066** | **0.020** | **0.65** |

Combining the two large, near-disjoint tiers (402 pairs, 92 host systems, two
galaxies in common):

* **statistical only: sd = 0.0048 dex in log V, 3-sigma reach 0.014 dex, power
  against H1 = 0.90.**
* **statistics plus the measured environmental systematic budget: sd = 0.0097
  dex, 3-sigma reach 0.029 dex in log V (0.058 dex in log g), power against
  H1 = 0.36.**

**The verdict on power is therefore: this sample can measure the offset to a few
thousandths of a dex statistically, but it CANNOT decisively separate the two
hypotheses, because the environmental systematics are the same size as the
effect.** The 3-sigma reach in log g, 0.058 dex, is larger than the tensor
lane's own 0.040 dex tolerance. That is the answer to "whether the sample can
distinguish them at all", and it is stated before the measurement below.

## 2. The measurement

Combined MaNGA `B4_disk_wide` + SAMI `S2_diskbearing`, 402 matched pairs:

> **Delta log10 V_internal = -0.0095 +- 0.0048 (stat) +- 0.0084 (syst)**
> **Delta log10 g_internal = -0.019 +- 0.010 (stat) +- 0.017 (syst)**

against

| hypothesis | prediction, Delta log g | prediction, Delta log V | separation from the measurement |
|---|---|---|---|
| **H1** potential-depth gate | **+0.031 +- 0.023** | +0.0155 +- 0.0115 | -2.6 sigma (stat+syst); -1.7 sigma once H1's own realisation scatter is included; -5.3 sigma if the nuisance covariates are explicitly regressed out rather than budgeted |
| **H2** acceleration-only / algebraic RAR | 0.000 | 0.000 | **-1.0 sigma -- consistent** |
| **H3** MOND with the external-field effect | -0.024 to -0.009 | -0.0122 to -0.0045 | **+0.34 / -0.63 sigma -- consistent** |

One-sided 95% upper limit on the internal-gravity boost of a cluster member:

* **Delta log g_int <= +0.013 dex** (statistics plus systematics), which lies
  below both H1's central value (+0.031) and the tensor lane's 0.040 dex
  tolerance. **78% of the H1 prediction band N(0.031, 0.023) is excluded; the
  lower 22% is not.**
* Delta log g_int <= -0.003 dex on statistics alone, which would exclude 93% of
  the band. The difference between those two lines is entirely the systematic
  budget, and that is the whole story of this lane.

**Plain answer to the question as posed.** A galaxy's internal stellar dynamics
does *not* measurably change when it sits inside a cluster, at fixed baryonic
content. The central value is slightly negative -- cluster members are
marginally *colder* than their matched field twins -- which is the direction
MOND's external-field effect predicts and the opposite of the direction a
potential-depth gate predicts. But the result is 1.0 sigma from zero and 0.3 to
0.6 sigma from the MOND-EFE bracket, so it does not distinguish those two
either.

---

## 3. What was measured, and why that quantity

**Y = log10 sigma_e_tot**, the flux-weighted aperture second velocity moment of
the **stars** inside 1 Re:

    sigma_e_tot^2 = < V^2 + sigma^2 >_F ,   R <= 1 Re

* MaNGA: computed here from the DR17 DAP `MAPS` cubes on disk
  (`code/extract_manga_kin.py`), on unique stellar-continuum Voronoi bins,
  weighted by bin flux x bin area, with `sqrt(SIGMA^2 - SIGMACORR^2)` and the
  systemic velocity removed as the flux-weighted mean inside the aperture.
* SAMI: `SIGMA_RE_MGE * sqrt(1 + VSIGMA_RE_MGE^2)` from the published DR3
  stellar-kinematics catalogue -- algebraically the identical construction. The
  MGE variants are used in both arms because MGE photometry is the only
  homogeneous photometry across SAMI's cluster and field arms.

Three reasons, all declared before residuals:

1. **Stars, not gas.** Cluster passage strips and shocks gas. Gas-star
   kinematic disagreement is used here only as a contamination *flag*, never as
   a measurement, exactly as the brief requires.
2. **The second moment, not the rotation speed.** Environment converts ordered
   rotation into random motion at roughly fixed kinetic energy -- the kinematic
   morphology-density relation. `V_rot` is contaminated by that; the second
   moment is not. Section 5 shows this is not a theoretical worry: choosing
   `V_rot` instead would have produced a large, highly significant, and entirely
   astrophysical "signal".
3. **No model.** sigma_e_tot needs no rotation-curve fit, no tilted-ring model,
   no inclination deprojection and no disk assumption, so it works identically
   on the S0s that make the large tiers large.

**Unit conversion, stated explicitly.** The tensor lane's "member violation" is
a boost of the member's internal *gravity*: its worst case, 0.687 dex, is quoted
there as "a factor 4.9", and 10^0.687 = 4.87. So H1 = +0.031 +- 0.023 dex in
log g. At fixed radius g = V^2/R, so Delta log V = 0.5 x Delta log g =
+0.0155 +- 0.0115. Both conventions are carried through the JSON. If the brief
instead meant 0.031 +- 0.023 in log V, the measurement sits -4.2 sigma away on
stat+syst and -1.6 sigma once the theory scatter is folded in: the conclusion
does not change.

**H3 is not the same hypothesis as H2, and it belongs in this test.** The brief
names two predictions. There are three. An *algebraic* RAR is local in g_N/a0
and predicts exactly zero (H2). A real MOND theory (AQUAL/QUMOND) is not local:
an external field suppresses the internal boost. The env-data lane measured this
sample's median |g_ext|/a0 = 0.17 to 0.20, squarely where that bites. H3 is
therefore computed pair by pair from each cluster member's own measured g_bar
and g_ext with nu(x) = 1/(1 - exp(-sqrt(x))), under the two standard
prescriptions (argument shift, quadrature), which bracket it at -0.0122 to
-0.0045 in log V.

---

## 4. The sample actually used

Full per-pair tables: `clean/sample_used_<survey>_<tier>.csv`. Tolerances,
environment ranges and host-system structure for every tier:
`member_dynamics.json -> sample_actually_used`.

### 4.1 Quality cuts and attrition

Declared, symmetric, and pair-dropping (a pair dies if either member fails):

* MaNGA: extraction succeeded; >= 50% of the 1 Re ellipse has usable stellar
  kinematics; >= 10 independent bins inside 1 Re; kinematic asymmetry
  `A_kin <= 0.0454` (the 80th percentile of the parent distribution).
* SAMI: kinemetry `k5/k1 <= 0.0939` (the same 80th-percentile rule);
  aperture-correction flag <= 1.

`A_kin` is a model-free point-reflection statistic: a regular rotator satisfies
V(-r) = -V(r) about the kinematic centre, so `A_kin = median |V(r)+V(-r)| /
(2 sigma_e_tot)` measures disturbance without fitting anything.

| sample | used / declared | attrition | cluster-arm pass | field-arm pass | host systems (effective, largest share) |
|---|---|---|---|---|---|
| MaNGA `B4_disk_wide` | 192 / 281 | 32% | 0.833 | 0.776 | 84 (9.6, 28%) |
| MaNGA `C2_xray_disk` | 144 / 218 | 34% | 0.817 | 0.761 | 36 (5.4, 40%) |
| SAMI `S2_diskbearing` | 210 / 364 | 42% | 0.687 | 0.742 | 8 (5.8, 23%) |
| SAMI `S1_latetype` | 58 / 108 | 46% | 0.657 | 0.731 | 7 (5.0, 28%) |

The cuts pass at nearly the same rate in the two arms (differences of 3 to 6
percentage points, in opposite directions in the two surveys), so they are not
silently building a biased cluster sample.

**The effective number of independent environments is 6 to 10, not 200.** MaNGA
`B4` nominally has 84 host groups but Coma alone supplies 28% of the pairs;
SAMI has exactly 8 clusters. Every uncertainty quoted here comes from a
bootstrap that resamples *host systems*, not pairs. The design effect
(system-bootstrap variance over naive pairwise variance) is 0.87 to 1.24 for the
large tiers and 6.8 for `B1_primary`.

### 4.2 Tolerances achieved on the sample actually used

MaNGA `B4_disk_wide`, 192 pairs:

| variable | declared tol | max abs | rms | mean | mean in units of its own error |
|---|---|---|---|---|---|
| `d_logMstar_nsa` | 0.10 | 0.0990 | 0.0552 | -0.0016 | -0.4 sigma |
| `d_logRd` | 0.10 | 0.0956 | 0.0398 | -0.0051 | **-1.8 sigma** |
| `d_logSigma_b` | 0.15 | 0.1097 | 0.0513 | -0.0050 | -1.3 sigma |
| `d_log_gbar_2p2Rd` | 0.10 | 0.0998 | 0.0514 | -0.0051 | -1.4 sigma |
| `d_incl_deg` | 10.0 | 9.874 | 5.124 | +0.776 | **+2.1 sigma** |
| `d_pym_r_BT_SE` | 0.15 | 0.1492 | 0.0768 | +0.0035 | +0.6 sigma |
| `d_z` | 0.010 | 0.0100 | 0.0045 | -0.0005 | -1.7 sigma |

SAMI `S2_diskbearing`, 210 pairs:

| variable | declared tol | max abs | rms | mean | mean in units of its own error |
|---|---|---|---|---|---|
| `d_logMstar` | 0.10 | 0.0992 | 0.0482 | +0.0016 | +0.5 sigma |
| `d_logRd` | 0.10 | 0.0941 | 0.0314 | -0.0020 | -0.9 sigma |
| `d_logSigma_b` | 0.15 | 0.0975 | 0.0465 | +0.0056 | +1.8 sigma |
| `d_log_gbar_2p2Rd` | 0.10 | 0.0975 | 0.0465 | +0.0056 | +1.8 sigma |
| `d_incl_deg` | 10.0 | 9.900 | 4.953 | -0.780 | **-2.3 sigma** |
| `d_z_spec` | 0.010 | 0.0098 | 0.0047 | +0.0006 | +1.9 sigma |

**Every pair is inside the declared box, and yet several variables carry a
statistically significant residual mean imbalance.** Being inside a hard box is
not the same as being balanced. That is why the headline estimator is not the
raw paired mean but the intercept of a regression of the paired difference on
the matching-variable differences, which is exactly zero under the null for any
residual imbalance. The adjustment moves the answer by 0.0034 dex in MaNGA
(raw -0.0014 -> adjusted -0.0048) and 0.0018 dex in SAMI (raw -0.0103 ->
adjusted -0.0121).

Inclination is imbalanced in *opposite* directions in the two surveys (+0.78 deg
in MaNGA, -0.78 deg in SAMI). A common inclination systematic would therefore
push the two surveys in opposite directions; they agree in sign instead.

**As env-data established and this lane carries: `logSigma_b` and
`log_gbar_2p2Rd` are one variable, not two** (r = 0.996 in MaNGA, exactly 1 in
SAMI -- Freeman's formula makes g_bar a constant times Sigma_b for an
exponential disk). `d_log_gbar_2p2Rd` is therefore deliberately **excluded**
from the covariate adjustment: including both would make the design matrix
collinear and would double-count one direction. The effective matching space is
about three directions (M*, R_d, and a projection of shape), not five.

**And, as env-data established, `f_gas` is not a matching variable in this
regime and was dropped.** HI is stripped: 17 of 494 cluster-arm detections
against 572 of 1603 in the field. The gas fraction is instead carried as a
contamination-budget term (`HI_detected`, below), which is the only honest use
left for it.

### 4.3 Environment spanned

| sample | host sigma_v (km/s) min/med/max | R/R_vir (R/R200) | \|g_ext\|/a0 | log \|Phi\| proxy (m^2/s^2) |
|---|---|---|---|---|
| MaNGA `B4_disk_wide` | 300 / 564 / 840 | 0.04 / 0.87 / 1.50 | 0.028 / 0.118 / 2.16 | 10.96 / 11.50 / 11.85 |
| MaNGA `C2_xray_disk` | 28 / 605 / 840 | 0.10 / 0.76 / 1.47 | 0.002 / 0.140 / 1.93 | 8.90 / 11.56 / 11.85 |
| SAMI `S2_diskbearing` | 492 / 690 / 1002 | 0.01 / 0.44 / 1.00 | 0.064 / 0.202 / 4.73 | 11.38 / 11.68 / 12.00 |
| SAMI `S1_latetype` | 492 / 765 / 1002 | 0.04 / 0.42 / 0.85 | 0.078 / 0.209 / 1.85 | 11.38 / 11.76 / 12.00 |

The tensor lane's member galaxy sits at |Phi_N| = 1.09e12 m^2/s^2, i.e.
log |Phi| = 12.04. **SAMI's deepest hosts reach log |Phi| = 12.00 and its median
member sits at 11.68**, so the sample does reach the regime the constraint is
about, within a factor of two in potential depth of the configuration the tensor
lane modelled. MaNGA's median member is a factor 3.5 shallower. The `C2` tier's
28 km/s minimum host is a reminder that the X-ray flag is a galaxy-level flag
("this galaxy sits in an X-ray emitting ICM"), not a richness cut, so a few of
its Tempel hosts are tiny.

---

## 5. Two ways to get a large, confident, and wrong answer

Both were found in this lane. Both would have produced a "detection" of about
the size H1 predicts.

### 5.1 Using the rotation speed as the gravity tracer

Same pairs, same machinery, same cuts -- only the tracer changes:

| tracer | MaNGA `B4` | MaNGA `C2` |
|---|---|---|
| sigma_e_tot (second moment) | **-0.0048 +- 0.0080** | **-0.0029 +- 0.0130** |
| sigma_e only (dispersion) | +0.0028 +- 0.0088 | +0.0023 +- 0.0128 |
| V_e only (rotation) | **-0.0689 +- 0.0234 (-2.9 sigma)** | -0.0502 +- 0.0273 (-1.8 sigma) |

Cluster members rotate 15% more slowly than their matched field twins at
2.9 sigma, while their dispersion is unchanged at 0.3 sigma, so their total
kinetic energy -- the quantity that actually traces the potential -- is
unchanged. That is the kinematic morphology-density relation, a structural
response to environment, and it is four times the size of H1 and of the opposite
sign. **A version of this test that had used a rotation velocity as "internal
dynamics" would have reported a large negative detection and it would have been
astrophysics.** The brief's instruction to control environmental effects below
the 5% velocity sensitivity is exactly this: the raw contamination in the naive
tracer is 15%.

### 5.2 Cutting on the observable

The cut set declared in `code/analyse.py` included `med_sigma_astro >= 40 km/s`
on both members of a pair, as a guard against the MaNGA DAP dispersion floor.
**That is a cut on the observable**, and applying it to both members of a
matched pair truncates the two arms unequally and biases the paired difference.
Scanning the threshold, in a sample where the underlying answer cannot depend on
it:

| MaNGA `B4`, sigma floor (km/s) | 0 | 30 | 40 | 50 | 60 | 70 | 80 |
|---|---|---|---|---|---|---|---|
| pairs | 192 | 192 | 190 | 173 | 152 | 120 | 108 |
| Delta log V | -0.0048 | -0.0048 | -0.0043 | +0.0014 | +0.0028 | +0.0072 | +0.0066 |

| SAMI `S2`, sigma floor (km/s) | 0 | 60 | 75 | 90 | 105 | 120 |
|---|---|---|---|---|---|---|
| pairs | 210 | 204 | 188 | 146 | 112 | 82 |
| Delta log V | -0.0121 | -0.0115 | -0.0108 | -0.0091 | -0.0065 | +0.0009 |

Both surveys reproduce the same artefact independently: a monotone march of
about +0.013 dex in log V -- the size of H1 -- produced purely by selection. A
"high-purity" subsample built by combining a sigma >= 60 floor with a median
asymmetry cut returned **+0.027 +- 0.012 (p = 0.03)** in MaNGA `B4`, which would
have read as a 2-sigma confirmation of H1.

The corrected primary cut set (`B`) simply drops the sigma floor. Four cut sets
are reported in full in `member_dynamics.json -> results_by_cutset`:

| sample | A declared (with sigma floor) | **B outcome-free** | C B + log M* >= 9.8 | D B + sigma >= 60 |
|---|---|---|---|---|
| MaNGA `B2_disk_strict` | +0.0096 (84) | +0.0091 (85) | +0.0193 (66) | +0.0197 (65) |
| MaNGA `B3_late_wide` | -0.0258 (45) | -0.0276 (46) | -0.0195 (39) | -0.0365 (32) |
| MaNGA `B4_disk_wide` | -0.0043 (190) | **-0.0048 (192)** | +0.0012 (163) | +0.0028 (152) |
| MaNGA `C1_xray_late` | -0.0492 (29) | -0.0492 (29) | -0.0637 (22) | -0.0576 (18) |
| MaNGA `C2_xray_disk` | -0.0030 (143) | **-0.0029 (144)** | +0.0025 (116) | +0.0044 (104) |
| SAMI `S1_latetype` | -0.0178 (58) | -0.0178 (58) | -0.0125 (53) | -0.0166 (32) |
| SAMI `S2_diskbearing` | -0.0121 (210) | **-0.0121 (210)** | -0.0106 (201) | -0.0091 (146) |

Cut set C imposes the same physical intent (stay away from the dispersion floor)
through a cut on log M*, which is a *matched* variable and therefore balanced by
construction. It moves MaNGA `B4` by +0.006 and SAMI `S2` by +0.002 -- a fifth
of what the sigma cut does. The change of primary cut set from A to B was made
because the artefact was identified, not because of which answer it gave: A and
B differ by 0.0005 dex.

---

## 6. Environmental contamination, priced in the units of the signal

For each diagnostic X, the field arm's own sensitivity dY/dX (at fixed M*, R_d,
B/T, inclination) is multiplied by the measured cluster-minus-field mean
difference in X. Every slope carries a host-system bootstrap error.

MaNGA `B4_disk_wide`:

| diagnostic | dY/dX (field) | delta<X> cluster-field | induced Delta log V | significant? |
|---|---|---|---|---|
| g-i colour | +0.177 | +0.067 mag | **+0.0111 +- 0.0035** | yes |
| continuum S/N | +0.0020 | -3.18 | **-0.0065 +- 0.0021** | yes |
| HI detected | -0.041 | -0.095 | **+0.0038 +- 0.0019** | yes |
| kinematic asymmetry A_kin | -2.19 | -0.0022 | +0.0043 +- 0.0022 | no |
| gas-star PA misalignment | +0.00015 /deg | +21.7 deg | +0.0032 +- 0.0031 | no |
| Sersic n | +0.0098 | +0.43 | +0.0040 +- 0.0023 | no |
| log Re (aperture) | -0.129 | +0.0097 dex | -0.0014 +- 0.0020 | no |
| aperture coverage | +0.25 | +0.0034 | +0.0006 +- 0.0012 | no |
| **quadrature total** | | | **0.0151** (significant terms only: 0.0134) | |

SAMI `S2_diskbearing` (with colour added so the two budgets price the same
physics):

| diagnostic | delta<X> | induced Delta log V | significant? |
|---|---|---|---|
| mu within 1 Re | -- | **-0.0077 +- 0.0026** | yes |
| gas-star PA misalignment | +26.0 deg | +0.0046 +- 0.0034 | no |
| g-i colour | -- | -0.0042 +- 0.0025 | no |
| kinemetry k5/k1 | -0.0008 | +0.0015 +- 0.0025 | no |
| aperture-correction flag | +0.014 | -0.0001 +- 0.0002 | no |
| **quadrature total** | | **0.0101** (significant only: 0.0077) | |

Two diagnostics that an earlier pass priced -- the median stellar dispersion and
lambda_R -- are *components of sigma_e_tot*, not independent contaminants.
Pricing them charges the signal against itself. They are reported as diagnostics
and excluded from the totals; the exclusion is recorded in the JSON.

**How far did the environmental control get?** To about **0.008 dex in log V
(0.017 dex in log g)** on the combined sample, against a signal of 0.0155
(0.031). That is 1.9% in velocity, against the brief's 5% target -- so the
astrophysical control *is* below the survey's velocity sensitivity, as asked,
but it is **not** below the size of the effect being tested. That is the reason
the power drops from 0.90 to 0.36.

Two reasons to think 0.008 dex is a conservative upper bound rather than a
realistic error:

1. Quadrature-summing every diagnostic assumes each is a full, uncorrected bias.
   **Explicitly regressing the nuisance covariates out** -- g-i colour, Sersic
   n, log Re, continuum S/N and A_kin for MaNGA; colour, k5/k1 and lambda_R for
   SAMI -- moves the combined answer by only **0.0016 dex**, to
   -0.0079 +- 0.0044. If that correction is trusted, H1 sits -5.3 sigma away on
   statistics alone (-1.9 sigma with its own scatter).
2. The dominant MaNGA term, colour, has a *positive* induced offset. Correcting
   for it pushes the measurement further from H1, not towards it. **No term in
   the budget can rescue H1 by having its sign flipped.**

The aperture definition contributes separately: MaNGA `B4` gives -0.0048
(1 Re), -0.0002 (0.5 Re), -0.0076 (fixed 3 kpc), -0.0093 (fixed 5 kpc), a
half-range of 0.0046 dex. The fixed physical apertures, which are immune to any
Re mismatch between the arms, are slightly *more* negative than 1 Re.

---

## 7. The offset as a function of potential depth and clustercentric radius

A potential-depth gate predicts a gradient; a galaxy/not-galaxy class step does
not. After removing the matching-variable dependence:

| sample | axis | slope | contrast (deep half - shallow half) |
|---|---|---|---|
| MaNGA `B4` | log \|Phi\| proxy | +0.006 +- 0.027 | -0.011 |
| MaNGA `B4` | R/R_vir | -0.021 +- 0.033 | -0.009 |
| MaNGA `B4` | R_proj (Mpc) | -0.027 +- 0.043 | +0.008 |
| MaNGA `C2` | log \|Phi\| proxy | -0.001 +- 0.021 | -0.007 |
| MaNGA `C2` | R/R_vir | -0.042 +- 0.037 | -0.034 |
| SAMI `S2` | log \|Phi\| proxy | +0.045 +- 0.036 | +0.010 |
| SAMI `S2` | R/R200 | +0.011 +- 0.025 | +0.009 |
| SAMI `S2` | R_proj (Mpc) | +0.010 +- 0.013 | +0.019 |

**No gradient is detected on any axis in any tier.** This is *not* evidence
against a gradient, and the sensitivity has been characterised rather than
assumed. Injecting a gradient of known amplitude and recovering it gives a slope
uncertainty of 0.027 (MaNGA) and 0.039 (SAMI) per dex of log |Phi|, over sampled
|Phi| ranges of 0.85 and 0.53 dex. So the **3-sigma minimum detectable
end-to-end offset across the sampled potential range is 0.068 dex (MaNGA) and
0.062 dex (SAMI) in log V** -- four times H1's entire predicted offset. **The
gradient test has essentially no power in this sample and should not be quoted
as a null.**

**Shared-denominator guard, applied.** SAMI's `R_on_rtwo = R_proj/R200` with
`R200 ~ sigma_200`, so R/R200 carries sigma in its denominator; putting sigma_v
on the other axis reproduces exactly the structure that retracted
rho_p = -0.304. The sigma-free `R_proj_Mpc_from_cat` is therefore carried and
reported alongside, and the depth axis uses sigma_v alone (never sigma^2/R,
which shares R with the radius axis).

---

## 8. Validation

| check | result |
|---|---|
| **Null simulation with the real error covariance and the real shared inputs** | Forward-simulate Y from a field-frozen baryonic relation using each galaxy's own measured covariates, add the measured per-galaxy errors, run the identical estimator, 2000 times. Bias = **-0.0003 +- 0.0071** (MaNGA `B4`), -0.0002 +- 0.0072 (SAMI `S2`), +0.0002 +- 0.0092 (MaNGA `C2`). **The estimator is unbiased under the null**, and its simulated width (0.0071) agrees with the system bootstrap (0.0080). |
| **Injection / monotone-invariance** | Injecting delta in {-0.05 ... +0.05} into the cluster arm returns d(estimate)/d(delta) = **1.000000** exactly, with a 0.100 dex spread across the injected range. The headline statistic is not degenerate in the parameter it is supposed to measure. |
| **Blocked sign-flip permutation null** | p = 0.50 (MaNGA `B4`), 0.070 (SAMI `S2`), 0.79 (MaNGA `C2`). Flips are applied per host system so the null carries the same system-level correlation as the data. |
| **Leave-one-system-out jackknife** | Max shift 0.0055 (MaNGA `B4`, 84 systems), 0.0047 (SAMI `S2`, 8 clusters). Dropping Abell 85 moves SAMI `S2` from -0.0121 to -0.0168; dropping Abell 4038 moves it to -0.0087. **No single cluster drives the result.** |
| **Cross-survey zero point** | On the 6 galaxies with both a MaNGA MAPS extraction and a SAMI measurement, MaNGA's sigma_e_tot is **0.052 dex below** SAMI's (sd 0.042, r = 0.96). Expected: env-data measured the two surveys' size scales to differ by +0.179 dex and the aperture is 1 Re in each survey's own Re. **Irrelevant to the result** -- each survey supplies both of its own arms, so a constant zero point cancels exactly in the paired difference -- but it forbids pooling at galaxy level, and only 10 of the 103 cross-matched galaxies have a MAPS cube on disk, so this check is weak. |

---

## 9. Failure modes from the brief -- explicitly checked

| failure mode | verdict | what was done |
|---|---|---|
| **Shared-denominator artefacts** | **PASS, and simulated** | sigma_e_tot shares no algebraic input with any matching variable: the aperture comes from the NSA elliptical-Petrosian Re, the matching from PyMorph's disk half-light radius, and neither M*, B/T nor inclination enters the measurement at all. The residual sharing is indirect (Re correlates with R_d; z scales both). It was not assumed to be harmless: the null was forward-simulated with the actual measured covariates and per-galaxy errors and returns a bias of -0.0003 +- 0.0071. SAMI's sigma-in-the-denominator radius was replaced by the sigma-free one wherever sigma appears on the other axis. |
| **Monotone-invariant statistics** | **PASS** | d(estimate)/d(injected offset) = 1.000000 over a +-0.05 dex injection range, spread 0.100 dex. The gradient statistic's sensitivity was measured the same way rather than assumed (section 7). |
| **Refitting on the held-out set** | **PASS** | The field-arm scaling relation used in the null simulation is fitted on field members only and frozen; the cluster arm never enters any fit. The estimator itself is a paired difference and fits nothing to the cluster arm. The pre-registered power was written to disk before any offset was computed. |
| **Silent extraction failures** | **PASS** | 645 of 645 paired galaxies had a MAPS cube on disk; 641 extracted (4 failures, all `few_bins_in_1Re`, reported not silently dropped). Over all 902 cubes, 898 extracted. The H-alpha channel index was verified against the DAP header (`C24 = Ha-6564` -> 0-based 23) rather than assumed; `BINID` channel 2 was verified to be "Stellar continua" and to be genuinely Voronoi-binned (504 bins in 2528 spaxels for a test cube) before deduplicating on it; the `SPX_ELLCOO` channel meanings were verified numerically against `REFF` and against an independent distance calculation. A units error in the first H3 computation (g_bar in m/s^2 treated as g_bar/a0) produced -2.26 dex and was caught by the number being physically absurd -- recorded here because it is the same class of error as the NSA h = 1 bug env-data found. |
| **Test bugs that look like solver bugs** | **N/A** | No PDE solver is exercised. |
| **Non-monotonic M(r) / clipped outer slopes** | **N/A** | No lensing deprojection is performed. |
| **Dark matter used as an observation** | **PASS** | Every quantity used is an observable: stellar kinematics, member-redshift velocity dispersions, sky geometry, X-ray luminosity, photometry. No `_rank_only` column (NFW masses, M500, R500) enters any measurement, cut, matching variable or axis. |
| **The shell average of a conductivity** | **N/A**, but the analogous choice was made and stated | The aperture average is flux-weighted over unique Voronoi bins with weight = mean flux x bin area, not mean flux, and not per spaxel. Using mean flux alone or per-spaxel weighting would over-weight the finely binned bright centre. |
| **Selective vs uniform refinement** | **N/A** | No coarse-graining. |
| **NEW: cutting on the observable** | **FOUND, quantified, corrected** | Section 5.2. Worth adding to the programme's checklist: a cut applied symmetrically to both members of a matched pair is *not* symmetric in its effect if it is a cut on the outcome. |
| **NEW: the tracer choice is a physics choice** | **FOUND, quantified** | Section 5.1. Rotation velocity and the second moment give answers that differ by 0.064 dex, four times the effect under test. |

---

## 10. What this does and does not establish

**Establishes.**

* A matched-pair measurement of the internal stellar dynamics of cluster
  galaxies against field twins, on 402 pairs from two independent surveys, with
  a system-level error budget: Delta log g_int = -0.019 +- 0.010 (stat)
  +- 0.017 (syst).
* A one-sided 95% upper limit Delta log g_int <= +0.013 dex, which excludes 78%
  of the tensor lane's H1 band and lies below its 0.040 dex tolerance.
* That the environmental systematic floor of this experiment is about 0.017 dex
  in log g -- the same size as the effect -- and *what* sets it
  (stellar-population colour differences at matched mass, continuum S/N,
  residual gas content).
* Two concrete ways this measurement can be made to produce a spurious
  H1-sized detection, both demonstrated on the real data.

**Does not establish.**

1. **It does not eliminate the potential-depth gate.** H1's lower edge survives.
   Combining the measurement with H1's own +-0.023 dex realisation scatter gives
   1.7 sigma, and the programme's own standard is that elimination requires a
   stated, quantified failure of a stated requirement. This is a *tension*, not
   a failure. What it does do is make H1 uncomfortable: the measured central
   value is on the wrong side of zero.
2. **It does not separate H2 from H3.** The measurement is -1.0 sigma from zero
   and 0.3 to 0.6 sigma from the MOND-EFE bracket. The two differ by 0.005 to
   0.012 dex in log V and the total error is 0.010.
3. **It says nothing about a gradient with potential depth**, for which the
   sample has no power (section 7). This matters, because a gradient is the one
   signature that would distinguish a potential-depth gate from the
   galaxy/not-galaxy class step the addendum names as the primary null.
4. **The deepest potentials are barely reached.** SAMI's median member sits at
   log|Phi| = 11.68 against the tensor lane's member at 12.04. That is a factor
   2.3 in Phi, and for a gate with m >= 2 at Phi_0 = 1e12 -- where 126 of the
   127 tensor survivors sit -- the response over that factor is large. **This
   sample sits just below the knee of the very gate it is testing.** A test at
   log|Phi| >= 12 needs the brightest members of the most massive systems, not
   disk galaxies at 0.4 R200.
5. **Nothing about the internal dynamics at small radii.** sigma_e_tot inside
   1 Re is an integrated quantity. The tensor lane's member boost was evaluated
   20 kpc from the member centre, comparable to 1 Re for these galaxies, so the
   comparison is fair -- but a radially resolved test would be stronger.
6. **The MaNGA/SAMI cross-calibration rests on 6 galaxies**, because only 10 of
   the 103 cross-matched galaxies have a MAPS cube on disk. Downloading the
   other 93 would settle it.
7. **H3 is a bracket, not a prediction.** The two standard EFE prescriptions
   differ by a factor 2.7, and both evaluate the internal acceleration at
   2.2 R_d rather than at the aperture where sigma_e_tot is measured.

---

## 11. Recommendation

The decisive version of this test is not more pairs. At 402 pairs the
statistical error is already 0.0048 dex and the systematic floor is 0.0084. It
is:

* **deeper potentials** -- cluster members at log|Phi| >= 12, i.e. massive early
  types near cluster cores, where a gate with m >= 2 at Phi_0 = 1e12 is fully on
  and the predicted offset is much larger than 0.031 dex. The fundamental plane
  of cluster versus field early types, which the tensor lane explicitly flagged
  as the missing quantitative limit, is the natural instrument and it does not
  need matched pairs;
* **the colour systematic removed at source** -- matched pairs selected to have
  the same *spectroscopic* stellar population (age, metallicity from the same
  IFU data), not just the same catalogue M*. That single term is 70% of the
  MaNGA budget;
* **a gradient with four times the current lever arm**, which needs the |Phi|
  range extended rather than the sample enlarged.

---

## 12. Files

```
member-dynamics/
  REPORT.md                       this file
  member_dynamics.json            all results, machine-readable
  power_prestatement.json         power, written before any offset was evaluated
  clean/
    manga_internal_kin.csv        645 paired galaxies, stellar kinematics
    manga_internal_kin_all.csv    all 902 MAPS cubes on disk
    sample_used_<survey>_<tier>.csv   the matched sample actually used, per tier
    PRODUCTS.manifest.json        sha256 / rows / columns for every product
  code/
    extract_manga_kin.py          DAP MAPS -> sigma_e_tot, A_kin, PA_kin, apertures
    power.py                      the pre-registered power statement
    analyse.py                    declarations, estimators, bootstrap, nulls
    analyse2.py                   three predictions, apertures, budget, combined
    analyse3.py                   budget errors, circularity fix, cross-survey
    analyse4.py                   the four cut sets and the sigma-cut artefact
    analyse5.py                   injection, gradient sensitivity, jackknife
    finalise.py                   tolerances achieved, upper limits, deliverable
    write_report.py               emits this file
```

Run order: `extract_manga_kin.py` -> `power.py` -> `analyse.py` -> `analyse2.py`
-> `analyse3.py` -> `analyse4.py` -> `analyse5.py` -> `finalise.py`.
`NW=1` runs the extractor serially; the machine's commit limit was shared with
another lane during this run.
"""

with open(os.path.join(LANE, "REPORT.md"), "w", encoding="utf-8") as fh:
    fh.write(REPORT)
print("wrote", os.path.join(LANE, "REPORT.md"), len(REPORT), "chars")
