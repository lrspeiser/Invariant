# Gravity roadmap Item 7: PHANGS baryonic-composition attempt 1

## Decision

`INCONCLUSIVE_ITEM7_BARYONIC_COMPOSITION_QUALITY_GATE`

This is a **nonpromoted positive lead**, not a discovery, rejection, or confirmation. The frozen
nonlinear stellar/H I/molecular-mixture families improved held-out prediction substantially, but
the attempt fails two preregistered gates: two of the 33 exploration galaxies exceed the frozen
interpolated-velocity uncertainty limit, and the mass-stratified permutation result is `p=0.11`
rather than at most `0.05`.

## What was tested

The response-blind audit matched the 67-galaxy PHANGS CO kinematic sample of Lang et al. (2020)
to the 90-galaxy PHANGS global-property catalog of Leroy et al. (2021). Stellar mass, H I mass,
CO(2-1) luminosity, stellar sizes, inclination, and CO coverage were complete for 54 galaxies;
45 also had published radius sampling to at least `1.5*lstar`. A salted, mass-and-phase-stratified
split froze 33 exploration galaxies and sealed 12 confirmations before any `VRot`, `E_VRot`, or
`e_VRot` value was queried.

The primary response is published CO rotation speed linearly interpolated at `1.5*lstar`. The
known/nonqualifying region includes a fixed baryonic Tully-Fisher scaling, fixed Newtonian
mass-over-size scaling, flexible stellar or total-baryonic mass/size/acquisition models, and raw
stellar/H I/molecular main effects. Only nonlinear phase entropy, phase-boundary, and
composition-by-geometry interactions qualify.

The molecular proxy applies the paper's global CO aperture correction and fixed
`alpha_CO=4.35`, `R21=0.65` conversion. Atomic gas receives a `1.36` helium correction. The
measurement limitation is explicit: CO luminosity and CO velocity come from the same PHANGS-ALMA
data products, although velocity never enters the feature builder.

Primary sources:

- Lang et al. (2020), PHANGS CO kinematics and rotation curves:
  <https://doi.org/10.3847/1538-4357/ab9953>
- Leroy et al. (2021), PHANGS-ALMA global properties and CO aperture corrections:
  <https://doi.org/10.3847/1538-4365/ac17f3>
- Frozen source catalogs: `J/ApJ/897/122` and `J/ApJS/257/43` at CDS/VizieR.

## Frozen results

- Source boundary: 33 exploration composition queries, 33 metadata queries, 33 rotation-curve
  queries, and 1,321 returned curve rows; zero confirmation, mass-profile, lensing, or paid-model
  accesses.
- Representation: 31 quality-pass galaxies; NGC 0628 and NGC 3507 fail only the frozen maximum
  50% interpolated fractional-error rule. They were not replaced and the threshold was not
  relaxed.
- Strongest nonqualifying nested selector: MSE `0.00535146`, held-out `R^2=0.6908`.
- Qualifying nested selector: MSE `0.00441234`, held-out `R^2=0.7450`.
- Relative MSE improvement: `17.55%`.
- Unrestricted selection: a qualifying family in four of five outer folds.
- Slice robustness: positive improvement in low/high mass and atomic/molecular-dominant strata.
- Measurement robustness: positive gain against both velocity-error envelopes and the nearby
  `1.4*lstar` response.
- Mass-quartile-stratified 499-permutation test: `p=0.11`.
- Frozen gates: 9 of 11 pass.
- All 12 confirmation rotation curves remain sealed.

## Interpretation and next move

The result says that the frozen nonlinear mixture/geometry representation carries reproducible
held-out information beyond ordinary mass-and-size formulas in this exploration sample. It does
not establish a causal variable or new gravity law. The two quality failures make this attempt
formally inconclusive, while `p=0.11` leaves a meaningful probability of a flexible-fit signal.

Do not retune the PHANGS families on these 31 opened responses and do not open the 12 PHANGS
confirmations. Preserve the exact family as `NONPROMOTED_POSITIVE_LEAD` and replay its prior
functional form on a materially independent galaxy sample with independently published atomic,
molecular, stellar, and kinematic measurements. Only that fresh frozen replay can complete Item 7.
