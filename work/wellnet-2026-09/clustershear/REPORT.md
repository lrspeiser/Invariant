# Run BQ -- raw weak lensing for the open half of the gold-cluster pool

Lane `work/wellnet-2026-09/clustershear/`. Rendered from `analysis.json`
and `controls.json`.

## Headline

**A 31.8-sigma stacked cluster lensing profile from 301 clusters, with both
null controls passing** -- and **no gravity result**, which is the correct
outcome for a lane whose job was to build the dataset, not to test a law.

The sealed half was never queried: 304 clusters, 0 tokens spent.

## The stacked profile

| R [Mpc] | g_t | g_x | err | clusters |
|---|---|---|---|---|
| 0.335 | +0.01361 | +0.00030 | 0.00184 | 220 |
| 0.446 | +0.01171 | -0.00036 | 0.00142 | 251 |
| 0.594 | +0.01046 | +0.00126 | 0.00109 | 274 |
| 0.790 | +0.00846 | -0.00173 | 0.00084 | 285 |
| 1.053 | +0.00761 | +0.00020 | 0.00065 | 290 |
| 1.402 | +0.00599 | -0.00081 | 0.00049 | 294 |
| 1.867 | +0.00453 | +0.00034 | 0.00037 | 296 |
| 2.486 | +0.00302 | -0.00054 | 0.00028 | 298 |
| 3.308 | +0.00198 | +0.00017 | 0.00021 | 299 |
| 4.407 | +0.00128 | +0.00003 | 0.00016 | 301 |

Monotonic decline over a decade in radius, as a cluster profile must be.

## The nulls, both passed

| null | what it tests | result |
|---|---|---|
| N1 cross component | lensing cannot make a parity-odd signal | chi2/dof = 13.6/10 -- consistent with zero |
| N2 random pointings | the catalogue's additive systematic floor, measured not assumed | chi2/dof = 13.9/10 -- consistent with zero |

N2 ran the identical estimator at 120 cluster-free positions offset 1.0-1.5
deg from real clusters -- inside the same patchy DECADE footprint, well
outside any cluster aperture.

## Trends, each against its own label-scramble null

| observable | n | slope | z | p | verdict |
|---|---|---|---|---|---|
| X-ray temperature kT | 172 | +0.00491 | +5.07 | 0.0000 | beats the scramble null at p=0.0000 |
| X-ray counts within R500 | 301 | +0.00069 | +1.01 | 0.3195 | consistent with the label-scramble null |
| redshift | 301 | +0.00848 | +11.72 | 0.0000 | beats the scramble null at p=0.0000 |

## The controls that stop these being findings

Two of those trends have the shape of an artefact, so each was attacked.

**C1_geometry_removed** -- does the redshift trend survive dividing out Sigma_crit?

> SURVIVES -- not geometry

**C2_kT_controlled_for_z** -- does the kT trend survive a joint fit with redshift?

> SURVIVES -- kT carries information beyond redshift

**C3_fixed_redshift_slice** -- does the kT trend survive inside a narrow redshift band?

> SURVIVES at fixed redshift

**C4_redshift_controlled_for_mass_proxy** -- is the DeltaSigma redshift trend just flux-limited selection?

> SURVIVES -- not explained by the kT mass proxy alone

## What it means

NONE OF THIS IS A GRAVITY RESULT, and none of it was meant to be. Every surviving trend has a conventional explanation that this lane cannot exclude, and naming them is more useful than a generic caveat. (a) kT surviving at fixed redshift (C2, C3) is EXPECTED: kT tracks mass and mass lenses, so it is the pipeline passing a sanity check. (b) The X-ray-counts null is EXPECTED: counts depend on exposure and distance as much as on mass. (c) DeltaSigma rising with redshift at fixed kT (C4) is ALSO expected, for two reasons that have nothing to do with gravity -- DeltaSigma is a surface density, and self-similar clusters are more compact at higher redshift (R500 shrinks as E(z)^(-2/3)), so DeltaSigma rises with z at fixed mass; and the background geometry here rests on dnf_z photometric redshifts, whose bias grows with lens redshift in the same direction. Separating any of that from new physics needs a baryon model and a mass calibration that does not come from weak lensing -- which eRASS1's M500 does, so it is inadmissible for the job. What this lane DOES establish is a working, null-validated 301-cluster lensing dataset: cross component consistent with zero, random pointings consistent with zero, a 31.8-sigma stacked profile, and a monotonic decline from 0.0136 at 0.34 Mpc to 0.0013 at 4.4 Mpc. Testing a gravity law on it is a separate, pre-registered step, and confirming one belongs on the sealed half.

## The law search -- 679,120 model forms, and why the top 25 is not the answer

679,120 model forms (every subset of size 1-4 over 64 atoms in radius,
redshift and X-ray temperature), each fitted EXACTLY by weighted least
squares rather than sampled, at 701,553 forms per second over 1622 points.

All 25 of the top 25 beat all three null twins. **That headline is
misleading and here is why.** Every one of them sits at k=4, the
maximum complexity, because nothing in the ranking penalises
parameters; and 21 of the 25 share the same redshift core, so this is
one model wearing 21 hats, not 25 discoveries.

**The informative table is the complexity trade-off.**

| k | gain | null A randoms | null B property scramble | null C radial scramble | best form |
|---|---|---|---|---|---|
| 1 | 0.0738 | -0.0000 | 0.0259 | -0.0196 | `R^-1.0 T^+0.5` |
| 2 | 0.0846 | 0.0007 | 0.0300 | -0.0073 | `T exp(-R/2) + (1+z)^2/R` |
| 3 | 0.0932 | 0.0029 | 0.0158 | 0.0178 | `log(1+z) + log R log(1+z) + T/(1+R)` |
| 4 | 0.0947 | 0.0039 | 0.0458 | 0.0179 | `exp(-R) + log(1+z) + log R log(1+z) + T/(1+R)` |

Read it from the top row. **A single atom, `R^-1.0 T^+0.5`, reaches 0.0738 of the
0.0947 that four free parameters reach -- 78% of everything 679,120 forms
buy.** The other three parameters purchase 0.0210 more gain while the
property-scramble null climbs from 0.0259 to 0.0458. That is the shape of
overfitting, not of a law.

The k=1 form survives its nulls properly: the random pointings give
-0.0000 and the RADIAL scramble gives -0.0196 -- negative, i.e. worse than a
constant once the radial ordering is destroyed. So **the radial
structure is real.** But the property-scramble null still reaches
0.0259, about a third of the gain, because scrambling (z, kT) across
clusters leaves each profile's own radii intact -- null B cannot test a
purely radial term, which is exactly why null C was added.

None of this is a gravity law. `DeltaSigma ~ T^0.5 / R` is close to what
an isothermal sphere gives you for free, and the search had no baryon
model to work against, so it cannot distinguish a modified force from
ordinary cluster structure. What it does establish is that the
admissible variable set here is small enough that one term exhausts it.

## What was not done

- No mass was fitted; no gravity law was scored.
- eRASS1 `M500`, `R500`, `Mgas500` are weak-lensing-calibrated and were
  excluded throughout as circular.
- Cluster-member contamination of the source sample was not modelled; it
  dilutes the inner bins and its size is unmeasured here.
- The sealed half is untouched and stays that way until a law is
  pre-registered against it.

