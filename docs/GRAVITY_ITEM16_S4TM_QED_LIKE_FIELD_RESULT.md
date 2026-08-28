# Gravity Item 16: QED-like weak-field carriers on S4TM lenses

## Decision

`REJECT_ITEM16_QED_LIKE_WEAK_FIELD_EXPLORATION`

The tested carrier grammar is **not promoted as a gravitational law**. It improved on a
baryon-only GR calculation with one shared stellar-mass normalization, but it did not
clear the stronger calibration, permutation, or selection-stability gates. The recurring
positive-slip and density-dressed pattern is retained as a nonpromoted clue, not pruned
and not called new physics.

## Real problem

The test used the 40 grade-A strong lenses in the public S4TM catalog (Shu et al. 2017,
VizieR `J/ApJ/851/48`). The predictor-only freeze used each lens galaxy's redshifts,
photometric stellar mass, effective radius, and axis ratio. It deterministically assigned:

- 30 completely fresh exploration lenses, six in each outer fold;
- 10 reserved confirmation lenses whose velocity dispersions and Einstein radii were
  never requested or downloaded.

After the sample and all formulas were committed, the response phase queried only the 30
exploration names and read two real observables:

- SDSS stellar velocity dispersion, which probes the matter potential `Phi`;
- the HST/SIE Einstein radius, which probes the light potential `(Phi + Psi)/2`.

The code never read the catalog's inferred total lensing mass, dark-matter fraction, fit
chi-square, or degrees of freedom. The Einstein radius is nevertheless a published
image-model result rather than a direct image likelihood, so this experiment does not
satisfy the roadmap's later direct-lensing gate.

## Frozen physical construction

Stellar light was approximated by a spherical Hernquist profile matched to a de
Vaucouleurs effective radius. The two dimensionless responses were

`y_dyn = log[5 R_e sigma^2 / (G M_star)]`

and

`y_lens = log[pi R_E^2 Sigma_crit / (M_star f_Hernquist(<R_E))]`.

Every candidate used one universal stellar-mass multiplier learned only on its training
folds. No galaxy received its own mass multiplier or gravitational constant. Extra
carriers modified the two potentials together:

`mu_m(r) = 1 + sum_j A_j f_j(r, lambda_j, Sigma_star)`

`mu_l(r) = 1 + sum_j A_j p_j f_j(r, lambda_j, Sigma_star)`

where `p_j = (1 + eta_j)/2` was a frozen polarization/slip label. The grammar included
subtracted Yukawa carriers, rationalized one-loop running, vacuum-polarization
crossovers, density-dressed vertices, two-pole nonlocal crossovers, and carrier-loop-
vertex products. Each amplitude was nonnegative, every response stayed positive, and
each modification vanished as `r/lambda -> 0`.

This is a weak-field response grammar, not a covariant quantum field theory. Universal
coupling, positive residues, symmetry, dimensions, and the local limit are enforced;
Ward identities, a conserved covariant action, causality, and full ghost freedom are not
proved.

## Frozen search size and controls

- Raw PCG64 formula cells: **262,144**.
- Exact parameter equivalence classes: **261,053**.
- Post-response formula cells: **0**.
- Candidate-observable matrix values: **15,728,640**.
- Candidate training-residual evaluations including 99 full-selection nulls:
  **6,291,456,000**.
- Device: **NVIDIA GeForce RTX 5090** through CuPy 13.5.1.
- Matrix construction: **0.677 s**; observed screen plus null trials: **1.855 s**.
- Maximum CPU/GPU log-response difference: **2.63e-15**.
- Maximum fractional modification at 1 AU over the full grammar: **1.94e-6**, below the
  frozen `1e-5` ceiling.
- The injected linked-potential control beat GR; a pure-GR control did not prefer a
  nonzero carrier.
- Paid calls and API spend: **0**.

## Results

All 30 exploration objects passed quality. The selected linked-potential formulas gave:

| Comparison | Reference MSE | Candidate MSE | Relative change |
|---|---:|---:|---:|
| GR plus one shared stellar-mass scale | 0.09685 | 0.08222 | **15.10% better** |
| GR with separate dynamics/lensing calibration | 0.08399 | 0.08222 | **2.10% better** |
| Fixed flexible predictor nuisance model | 0.05624 | 0.08222 | **46.21% worse** |

The candidate improved over shared-scale GR in both channels:

- stellar dynamics: **17.62%**;
- Einstein-radius lensing: **11.08%**.

It also improved in both halves of stellar mass and effective radius. Those are real
positive patterns. They are not sufficient for promotion:

- the selection-aware 99-null permutation result was `p = 0.08`, above the frozen 0.05
  gate;
- the candidate gained only 2.10% beyond simply calibrating dynamics and lensing
  separately;
- it was 46.21% worse than the frozen flexible nuisance predictor;
- three folds chose `carrier_loop_vertex_product` and two chose
  `density_dressed_vertex`, below the required four-of-five family stability;
- exact amplitudes, transition lengths, and secondary carriers varied substantially.

Eleven of fourteen gates passed. The failed gates were improvement beyond separate
calibration, selection-aware significance, and family stability.

## What the repeated pattern means

Every outer fold selected the `positive_slip_mixed_mode` for its primary carrier, in
which the light response is 1.5 times the matter response. All folds also selected a
density-dependent construction, either directly or through the carrier-loop-vertex
product. That may be pointing to one of three things:

1. a real relation between stellar surface density and gravitational slip;
2. ordinary lens/dynamics systematics, including the spherical virial approximation,
   anisotropy, or stellar-population mass calibration;
3. selection noise in a 30-object sample searched with a very large formula library.

The present experiment cannot distinguish those explanations. In particular, the fitted
shared stellar-mass factors ranged from about 0.56 to 0.65 times the catalog's Chabrier
mass while the carrier supplied extra attraction. That tradeoff demonstrates why a
baryonic-mass error can imitate modified gravity unless one mass calibration must explain
both motion and light.

## Failure-space update

The result rejects promotion of the exact tested region: positive-residue, static,
spherical, one- or two-carrier crossovers over 0.025--800 kpc, with the four frozen slip
labels and a shared stellar-mass normalization, when represented by this Hernquist/virial
forward model on S4TM. It does **not** reject:

- a resolved Jeans or orbit model with measured anisotropy;
- direct strong-lens image likelihoods and non-spherical baryonic maps;
- action-derived scalar/vector/tensor theories outside these response bases;
- negative/interfering residues that are independently proven stable;
- time-dependent, environmental, nonlocal, or history fields in later roadmap items;
- ordinary object-to-object stellar IMF variation.

The five exact selected formulas, every null statistic digest, source checksum, frozen
role, and failed gate are preserved in
`runs/gravity/roadmap/item-16-s4tm-qed-field-v1.json`.

## Next action

Advance to Item 17, running gravitational strength. Freeze one universal running law and
test whether it predicts object-to-object structure beyond both shared and separate
calibrations. Carry the Item 16 positive-slip/density pattern only as a fixed comparator.
Do not open the ten S4TM confirmations unless an unchanged survivor earns explicit
authorization for confirmation.
