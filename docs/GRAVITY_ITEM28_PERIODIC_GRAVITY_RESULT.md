# Item 28 periodic gravity result

## Decision

**INCONCLUSIVE QUALITY; NEGATIVE DIAGNOSTIC; ITEM COMPLETE.**

The frozen GHASP representation did not provide enough galaxies with reliable approaching and
receding rotation curves at the same six disk-scaled radii. Fifteen of 49 exploration galaxies and
71 curve points passed, versus the preregistered minima of 35 galaxies and 175 points. The quality
gate was not lowered. Ten confirmation galaxies remain sealed.

On the permitted diagnostic, the selected periodic formulas were 4.11% worse than the frozen
baryonic-light baseline and 3.19% worse than the strongest smooth ordinary model. The guarded
full-search permutation result was `p=0.17`; nine of 15 galaxies were individual counterexamples
relative to the flexible model. Four folds selected the baryonic-phase-coupling niche, but only
three agreed on a common wavelength and phase. Neither the universal-gravity track, the
phenomenon/publication track, nor the preregistered scoped-partial track passes.

This rejects only the exact Rc-light, exponential-disk/Sersic-bulge, two-sided interpolated GHASP
representation and the frozen finite periodic library. It does not reject all spatial, temporal,
log-periodic, resonance, wave, or phase-coupled theories. The result is preserved under the
equal-viability policy and does not reduce the starting priority of any distinct future mechanism.

## Frozen question

Can one universal periodic response, with no galaxy-specific wavelength, phase, amplitude, or
mass-to-light ratio, predict fresh resolved H-alpha rotation curves beyond smooth baryonic and
ordinary structural controls?

Every candidate used

`log10(v) = log10(v_baryonic) + A s W(g_bar) E(r/h)
              [sin(theta) + q2 sin(2 theta)] / (2 ln 10)`

with a positive bounded multiplier, a low-acceleration window, an optional fading radial envelope,
and one of four equally sized raw niches:

1. `spatial_linear`: `theta = 2 pi r_kpc / lambda_kpc + phi`;
2. `baryonic_orbital_clock`: `theta = 2 pi t_bar / T + phi`, with `t_bar` computed only from the
   frozen baryonic-light model;
3. `logarithmic_scale`: `theta = omega ln(r/h) + phi`;
4. `baryonic_phase_coupling`: `theta = 2 pi (r/h)/q + beta ln(g_bar/a_ref) + phi`.

The numbered niche order was not a probability ranking. Every niche received exactly 65,536 raw
cells and identical selection, null, baseline, counterexample, and reporting rules.

## Data isolation

The predictor side used only response-blind quantities from GHASP X Rc photometry:

- identity and sky coordinate;
- distance, morphology, inclination, and position angle;
- total Rc luminosity and one frozen stellar mass-to-light proxy;
- disk scale and central surface brightness;
- response-blind bulge, bar, and disk-break geometry;
- seeing per disk scale.

No published maximum velocity, dark-halo fit, dark-matter mass, GR-inferred total mass, or
mass-model parameter entered the predictor table. Identity-only listings established which GHASP
VI/VII objects had published curves before roles were assigned.

The audit intersected 170 photometric identities with 173 curve identities. After predictor
quality checks and vetoing every predecessor name or coordinate within 60 arcsec, 59 galaxies
remained. Five mass strata sealed two HMAC-ranked confirmations apiece and assigned the other 49
to five folds. The confirmation identities are not reproduced here as response queries because no
confirmation response was requested.

The source lineage is documented by the [GHASP VI curves](https://arxiv.org/abs/0805.0976),
[homogeneous GHASP VII reanalysis](https://arxiv.org/abs/0808.0132), and
[GHASP X Rc photometry](https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/MNRAS/453/2965?format=html&tex=true).

## Candidate audit

| Quantity | Value |
|---|---:|
| Raw candidates | 262,144 |
| Raw cells per niche | 65,536 |
| Admissible candidates | 223,283 |
| Spatial-linear admissible | 56,695 |
| Baryonic-orbital-clock admissible | 52,828 |
| Logarithmic-scale admissible | 56,831 |
| Baryonic-phase-coupling admissible | 56,929 |
| Maximum admitted one-AU fractional response | `9.6824e-6` |
| Admitted multiplier domain | `0.082085` to `12.182494` |

Cells were removed only by response-independent boundedness, minimum cycle-span, and local-gravity
filters. The different admitted counts do not change the equal raw opportunity.

The spatial and logarithmic families are known broad mathematical ideas, not historical novelties.
Spatial gravitational ripples have precedents such as
[gravitational Friedel oscillations](https://arxiv.org/abs/1804.00225), and logarithmic oscillations
have known links to [discrete scale invariance](https://arxiv.org/abs/2308.05904). The orbital-clock
and baryonic-phase-coupled branches are labeled potentially distinct combinations or syntheses,
not proven historical novelties. Creativity labels never entered selection or pruning.

## Response correction and quality result

The first authorized acquisition queried exactly the 49 exploration identities and zero
confirmations. GHASP VI documents the velocity-bin column as `NBins`, while GHASP VII exposes
`Nbins`. VizieR silently omitted the case-mismatched GHASP VI field, so 35 GHASP VI curves failed
integer parsing before evaluation.

The disclosed implementation correction requested the catalog-specific spelling and requeried only
the same 49 exploration identities. It changed no identity, role, fold, response value, threshold,
radius grid, candidate, baseline, formula, confirmation boundary, or gate. A later prose-only
correction changed the receipt's Item 29 label to the stable roadmap term `nonlinear
self-interaction`.

After the schema correction:

| Quality quantity | Frozen minimum | Observed |
|---|---:|---:|
| Valid exploration galaxies | 35 | 15 |
| Valid side-averaged curve points | 175 | 71 |
| Confirmation queries | 0 | 0 |

Failures were five curves with too few good raw points, two without enough points on each side,
and 27 without four jointly interpolable radii under the frozen maximum-gap rule. Every fold was
still represented, but two folds contained only one valid galaxy. This permits only a diagnostic,
not promotion.

## Diagnostic result

| Model | Out-of-fold log-velocity MSE |
|---|---:|
| Frozen baryonic-light baseline | `0.03854175` |
| Smooth flexible ordinary model | `0.03888629` |
| Selected periodic candidate | `0.04012735` |

The periodic candidate therefore changed performance by:

- `-4.11399%` versus the baryonic baseline;
- `-3.19152%` versus the flexible ordinary model;
- selection-aware `p=0.17` over 99 complete nested-search nulls;
- nine of 15 galaxy-level counterexamples versus the flexible model.

The dominant niche was baryonic phase coupling in four of five folds. Three folds selected the
exact same boundary-strength cell:

- positive polarity and amplitude `2`;
- phase coupling `-0.5`;
- phase `0`;
- radial period `0.25` disk scales;
- second harmonic `0.25`;
- envelope `4` disk scales;
- acceleration transition `1e-9 m/s^2` with power `1`.

The fourth fold selected a 10 Myr suppressing orbital-clock cell, and the fifth selected a
different phase-coupled cell with period four disk scales and coupling `2`. The frozen common-rule
gate required four folds to agree in niche, wavelength, and phase. Niche agreement reached four,
but wavelength and phase agreement each reached only three.

## Counterchecks and scoped regimes

The unchanged selected cells regressed on both physical sides relative to baryonic light:

- approaching: `-3.44%` versus baryonic and `-7.53%` versus flexible;
- receding: `-4.33%` versus baryonic, although `+2.63%` versus flexible.

Predeclared diagnostic slices also failed to form a supported phenomenon:

- inner radii: `+4.78%` versus baryonic but `-6.39%` versus flexible;
- outer radii: `-27.23%` versus baryonic and `+2.51%` versus flexible;
- high stellar mass: `-2.43%` versus baryonic and `+7.98%` versus flexible;
- low stellar mass: negative against both controls;
- low surface brightness: less than `1%` better than either control;
- high surface brightness: negative against both controls.

The strongest preregistered slice was the high-mass half at `+7.98%` versus flexible, below the
frozen 10% scoped-lead threshold; the global `p=0.17` also misses the scoped `p<=0.1` requirement.
No partial replication lead is retained.

## Search-engine controls and cost

All four target-blind injected periodic signals were recovered in their correct niche in all five
folds on the actual 71 predictor/radius rows. The zero-periodic-signal control and CPU/GPU identity
check passed. Thus the negative result is not explained by an inability of the implementation to
recover a periodic signal of the frozen injected strengths.

The diagnostic used:

- NVIDIA GeForce RTX 5090 through CuPy;
- 6,658,299,060 training-residual evaluations;
- 99 complete selection-aware null searches;
- four complete niche-injection searches;
- one zero-signal search;
- 110.01 seconds wall time;
- zero paid model calls and `$0` API spend.

## Two independent judgments

### Universal-gravity track

**Does not pass.** Quality fails, the periodic model loses to both baselines, the approaching and
receding sides do not jointly improve, radial/mass/surface-brightness halves do not jointly improve,
the guarded null fails, and wavelength/phase stability fails.

### Phenomenon/publication track

**Does not pass.** No global or preregistered scoped periodic relation beats the strong ordinary
model with the required null and stability evidence. There is no Item 28 paper lead. Any future
periodicity paper claim requires a newly frozen representation and unchanged fresh replication; it
cannot be obtained by retuning these 15 opened galaxies.

## Next action

Keep the exact failed regions in the counterexample/equivalence database. Do not retune the 15
opened valid curves, relax their grid after seeing the result, or query the ten confirmations.
Advance the numbered roadmap to Item 29, nonlinear self-interaction, on a fresh response. Continue
the Item 12/13 age relation and the Item 20/25 periodic hints only on their separate unchanged
cross-source publication tracks; none is privileged in the next mechanism search.

