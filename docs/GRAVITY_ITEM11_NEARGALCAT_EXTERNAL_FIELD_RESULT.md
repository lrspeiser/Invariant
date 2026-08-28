# Gravity roadmap Item 11: external baryonic field

## Decision

`INCONCLUSIVE_ITEM11_NEARGALCAT_QUALITY`

The exact published-summary external-field grammar is not promoted. The frozen quality floor
is missed, and the 119-galaxy valid diagnostic is strongly negative relative to the internal
baryon baseline. This does not reject a reconstructed three-dimensional external field,
vector tides, causal large-scale structure, or every environmental gravity theory.

## Frozen test

Before reading any NEARGALCAT galaxy row, commit
`dd38e82cebce836b821b4e4d93e029c202af18ce` froze:

- the 869-row Updated Nearby Galaxy Catalog source and predictor/response separation;
- exact name and 60-arcsecond coordinate exclusions against prior galaxy samples;
- a 75/25 split stratified by tidal-index sign and baryonic-mass half;
- a flexible eleven-variable internal-baryon ridge baseline;
- 262,144 PCG64 candidates across twelve labeled environment mechanisms;
- five-fold nested selection, one universal scalar external coefficient, and all gates.

Predictor access yielded 451 complete baryon/environment rows. The frozen predecessor audit
removed 105 normalized-name overlaps and 112 coordinate overlaps, leaving 325 independent
objects. Their exact split—243 exploration and 82 sealed confirmation—was committed at
`ef51079ce2f2dbaa2f783d87d07f80f104e1f556` before H I widths or rotation amplitudes were
queried.

## Variables and creative grammar

The internal baseline uses fixed-K-band stellar mass, helium-corrected H I mass, baryonic
mass, gas fraction, Holmberg diameter, B surface brightness, morphology, axial ratio,
distance, inclination, color, and a baryonic internal-acceleration proxy. It does not use
indicative/dynamical mass, radial velocity, object identity, neighbor identity, H I width, or
rotation amplitude as a predictor.

The external grammar combines the published nearest-neighbor tidal index `Theta1`,
five-neighbor index `Theta5`, and K-band luminosity density through:

- known signed-power transforms and group-boundary transitions;
- nearest-minus-collective dominance and combined-neighbor terms;
- isolation cavities and external-field suppression;
- internal/external resonance;
- neighbor coherence, environment saddles, and log-periodic environment terms.

Every cell records a known-transform, known-family combination, combination, or unresolved
origin label. Threshold, scale, power, phase, and target-blind internal modulation vary by
cell. No historical novelty is claimed, polarity duplicates are omitted, and no formula is
generated after response access.

## Quality result

Of 243 exploration rows, 119 pass all frozen checks. The largest failure is the preregistered
consistency requirement between published inclination-corrected rotation amplitude and the
simple `W50/(2 sin i)` reconstruction. This demonstrates that those catalogue quantities are
not interchangeable at the frozen 35% tolerance. The attempt misses both its 150-galaxy
minimum and 60% retention requirement, so its formal decision remains inconclusive. All 82
confirmation responses remain sealed.

## Held-out diagnostic

| Model | Held-out MSE | Held-out R2 |
|---|---:|---:|
| Internal-baryon ridge baseline | 0.0220111 | 0.5829 |
| Nested selected external-field term | 0.0250126 | 0.5261 |

The selected external terms increase MSE by `13.64%`. The mean per-galaxy MSE gain is
negative with paired sign-flip `p=0.990`. The result is negative in:

- both `Theta1` signs: 62 group and 57 isolated galaxies;
- both baryonic-mass halves;
- both gas-fraction halves;
- both distance halves.

Fold selections use different mechanisms and their universal coefficients change sign. Six
of fourteen gates pass.

The RTX 5090 screened `623,902,720` candidate-galaxy scoring combinations in `4.43` seconds
with CuPy 13.5.1. The maximum CPU/GPU component discrepancy was `2.78e-16`.

## Failure-space record

Record `NONPROMOTED_PUBLISHED_SCALAR_ENVIRONMENT_SUMMARY_REGION`:

- the exact twelve families, 262,144 seeded cells, and parameter ranges in the manifest;
- scalar transforms of `Theta1`, `Theta5`, and one-megaparsec K-band luminosity density with
  the tested internal modulations;
- one universal additive log-rotation coefficient after the fixed internal baseline;
- algebraic sign flips, rescalings, or renamings that add no new physical information.

Do not retune this region on the 119 opened valid responses or open the 82 confirmations to
rescue it. An external-field retry must reconstruct materially new information: neighbor
vectors and masses, a tidal tensor, filament geometry, time history, a field equation, or a
joint dynamics/lensing prediction.

## Boundaries

- This is a global rotation-amplitude test, not a resolved rotation-curve or lensing test.
- Published environment summaries may be noisy proxies for a true field.
- No external-field cause, alternative to general relativity, or historical novelty is
  established.
- A clean confirmation was preserved: zero confirmation responses were opened.

## Next real test

Advance to Item 12, dynamical age. Freeze formation-time and settling-time proxies on a fresh
real response. Candidate variables should include stellar-population age, specific star
formation, gas depletion, orbital-settling, and relaxation ratios while keeping morphology,
mass, and survey labels as controls. Do not reuse the 119 opened NEARGALCAT responses or open
the 82 confirmations.

## Replay evidence

- result file SHA-256:
  `4042cffcbd5dccd231997cacdede83154e1ad65d004ef48b74cf98a9ffd588a4`
- result content SHA-256:
  `3f532a0d6c94de8a3c3d43f38047a00298bda644ba2f8fb47d94ed0a2dc1a0e0`
- response-source SHA-256:
  `ad9949483af4e9d16a7840fc92c8c676f8ac418c513cedf250254ca770fea313`
- replay command:
  `python -m sigma_theory_compiler.gravity_item11_neargalcat_external_field check`
