# Length-dependent gravity: conditional local results

**34 of 54 fixed configurations lie within the declared historical local
screens; 20 lie outside.** All 54 pass numerical controls. All 324 isolated
monopole precession predictions lie within their six published intervals.
The distinguishing result comes from the external-field quadrupole: 69 of 108
background/configuration rows lie within its interval, and 39 outside.
Each complete card must satisfy both backgrounds, not select the favorable one.

This is a material change from the first-gradient candidates: higher sampled
acceleration scales can now satisfy these local diagnostics. No galaxy,
cluster, lensing, full ephemeris, relativistic or independent-validation pass
has been established. The discovery goal remains active.

## Fixed action and constants

The action is P=x+x K_m(x+h), K_m(u)=[Q_m(u)−u]/u, with the removable origin
defined by Q_m'(0)−1. Q_m is the previously registered bounded scalar action.
Here x=|∇ψ|²/a₀² and h=ℓ² Hψ:Hψ/a₀². Its full physical flux is
J_i=P_x ψ_i−ℓ² ∂_j(P_h ψ_ij). Including that derivative is essential.
The general higher-derivative framework and length-screening construction
are prior art; combining them with this bounded kernel is an explicit
effective-action ansatz, not a uniquely derived microscopic principle.
[Milgrom (2023)](https://arxiv.org/html/2305.01589v2)

Before physical predictions, the scan froze shapes 0.5, 1 and 2; a₀ values
5e−11, 1.2e−10 and 2e−10 m/s²; and ℓ=0, 0.001, 0.01, 0.1, 1 and 10 pc.
The finite regularizer is 1e−6, with 1e−7 and 1e−8 sensitivity checks.
Every card uses one length and acceleration scale across all six orbits and
both background assumptions. This grid is not a confidence interval.

The following quadrupole ranges span the two fixed backgrounds, in units of
1e−27 s⁻². The declared quadrupole screen is [−3,9] in those units. Listed
lengths are sampled values only, not continuous allowed intervals.

| Shape | a₀ (m/s²) | Q₂ at ℓ=0 | Q₂ at ℓ=0.1 pc | Q₂ at ℓ=1 pc | Sampled lengths within both screens (pc) |
| --- | --- | ---: | ---: | ---: | --- |
| 0.5 | 5.0e-11 | 1.800–2.198 | 0.519–0.633 | 0.054–0.066 | 0, 0.001, 0.01, 0.1, 1, 10 |
| 0.5 | 1.2e-10 | 12.306–13.421 | 2.502–2.882 | 0.258–0.299 | 0.1, 1, 10 |
| 0.5 | 2.0e-10 | 29.324–30.200 | 5.552–5.950 | 0.573–0.619 | 0.1, 1, 10 |
| 1 | 5.0e-11 | 0.990–1.571 | 0.582–0.747 | 0.060–0.078 | 0, 0.001, 0.01, 0.1, 1, 10 |
| 1 | 1.2e-10 | 17.170–23.009 | 3.804–5.004 | 0.393–0.519 | 0.1, 1, 10 |
| 1 | 2.0e-10 | 43.750–58.185 | 9.557–10.846 | 1.005–1.125 | 1, 10 |
| 2 | 5.0e-11 | 0.232–0.498 | 0.559–0.706 | 0.058–0.074 | 0, 0.001, 0.01, 0.1, 1, 10 |
| 2 | 1.2e-10 | 13.189–25.775 | 3.666–5.645 | 0.378–0.585 | 0.1, 1, 10 |
| 2 | 2.0e-10 | 37.458–61.568 | 8.120–12.209 | 0.854–1.276 | 1, 10 |

One pc and ten pc lie within both local screens for all nine shape/a₀ groups.
At 0.1 pc the highest-a₀ cases with shapes 1 and 2 do not satisfy both
backgrounds. Smaller nonzero lengths can initially increase the quadrupole;
screening is not a simple monotone multiplier on the scalar result.

## What was calculated and verified

The external solution uses ψ=−1/r−η_N z in units GM=a₀=1, and ℓ is converted
from pc to units of sqrt(GM/a₀). The asymptotic mapping
η_N ν(η_N)=g_external/a₀ follows the action's exact h=0 scalar limit.
The physical backgrounds, 1.9e−10 and 2.4e−10 m/s², are inherited published
scenarios, not a reconstructed Galactic field or a measured uncertainty range.

Two infinite-domain quadrupole representations are evaluated independently:
one integrates the full flux against the harmonic Green kernel, while the
other integrates the higher-derivative action term twice by parts. The latter
requires the full three-dimensional Hessian. Their maximum dimensionless
disagreement is 1.67e-16; maximum
128→256-node change is 1.38e-09; maximum
regularizer change is 2.72e-10.
No automatic 512/1024-node follow-up was needed. These changes are numerical
diagnostics, not certified error bounds or statistical uncertainty.

Seventeen new implementation tests cover high-precision kernel derivatives,
origin regularity, exact scalar recovery, independent scalar quadrupoles,
Cartesian versus polar flux, the bounded point-source asymptote, periodic
action variation and internal momentum conservation. SI Cartesian and
dimensionless radial calculations agree at all 648 periapse/apoapse probes;
maximum fractional disagreement is 2.46e-15.
Input snapshots, all 54 card hashes and every stored classification were
independently verified. A post-run lint cleanup binds sequential loop values
explicitly; the exact executed runner remains in the frozen input snapshots.

The first-order isolated precession range is
[-3.82088e-07, 9.84925e-05]
mas/century. The maximum sampled fractional perturbation is
1.02e-14. The external quadrupole
and isolated monopole are separate leading diagnostics, not a joint planetary
orbit and light-propagation fit.

## Evidence limits and next transfer

The quadrupole comparison uses the previously exposed Cassini result
Q₂=(3±3)e−27 s⁻² and the predeclared two-standard-deviation summary interval.
It is not a fresh analysis of Cassini observations or necessarily the latest
constraint. [Hees et al. (2014)](https://arxiv.org/abs/1402.6950)
The monopole uses INPOP10a supplementary-precession sensitivity intervals;
these are postfit-residual criteria, not Gaussian errors or a candidate-specific
likelihood. [Fienga et al. (2011)](https://arxiv.org/abs/1108.5546)
No new raw observations or reserved outcomes were opened.

The next test must use these same constants on observed source distributions.
Existing scalar galaxy and cluster scores cannot simply be attached to the
new cards: the new force depends on density derivatives. In particular, the
cluster adapter's piecewise linear stellar enclosed mass has slope jumps,
which imply discontinuous stellar density. Gas log-linear slopes and the
galaxy potential's cubic-Hermite derivative representation also need a
regularity audit. Source interpolation and its uncertainty must be declared
and tested before scoring the higher-derivative response. Lensing requires a
derived matter/photon coupling, which this static action does not supply.

Reproduce with `scripts/run_gravity_length_screening_local.py --output
<unused-directory>`. Evidence is in `work/gravity-first-principles/length-screening-local-001/`.
Result SHA-256: `66ff601b1012da7cbc555a27d8836723a2c6e7b23f393ead530da64e6e938a77`.
