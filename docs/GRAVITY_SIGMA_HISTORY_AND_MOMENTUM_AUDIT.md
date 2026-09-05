# SigmaGravity history transfer and coherent-monopole audit

The active Invariant branch now contains a selective, byte-preserved copy of
the local Sigma research history. The transfer adds useful alternatives and
failures to the search. It does not establish a new law that works in galaxies,
clusters and the Solar System.

## What was recovered

The principal source is the local `sigmagravity-frontiers-main` worktree at
`0364fdf361c3a9ccf16da6780ce97360983941c7`, whose final handoff is dated
2026-08-06. Four older explanatory documents came from `dev/sigmagravity` at
`2fe0ab5f580c0da2a6318f930a35e4daa03cd8b0`. The original worktrees were not edited.
This is a selective transfer, not a merge of every Sigma branch or a claim to
have inspected every local artifact.

The successful import contains 23 source files totaling 857,962 bytes: handoff,
validation matrix, historical scorecards, coherence reports, original code and
its dependency closure. The registry lists **128 scored formulas, 36 published
families, 222 protocols and 151 formula/action fragments**. These are overlapping
historical inventories, not that many independent successful theories. The
scorecard's “proximity percentages” are descriptive normalizations, not
probabilities, confidence levels, or comparable measurements across all rows.

`formula_scorecard.json` is a generated local file absent from the source Git
commit. Its actual bytes are hash-pinned with `git_blob=null`; the other 22
selected files have recorded Git blob identities. Exact working-file SHA-256
values, rather than newline-normalized Git blob IDs, identify the copied bytes.
The initial import stopped on this distinction and is retained as a partial
failure in `sigmagravity-import-001`; `sigmagravity-import-002` is authoritative.

No raw observations or reserved lensing/galaxy payloads were opened or copied.
Historical “held-out” scores remain historical evidence already exposed to
selection; this transfer does not make them fresh confirmation data.

## What the alternatives actually tell us

| Direction | Status carried into Invariant | Useful next step |
|---|---|---|
| Old phase and ordered-motion coherence | Phenomenology with fitted size/phase prescriptions and incomplete dynamics | Define one independent coherence observable and derive reciprocal coupling |
| RAR plus squared-coherence density bridge | Useful historical empirical comparator; no complete action or universal operational coherence field | Explain the response from field equations; test raw lens geometry |
| Measured density/coherence CPR0 | Exact interpolation failed its declared gates | Require a physically different mechanism before another fit |
| Coherent monopole P0696 | New conservation failure below; earlier spherical/curl controls were insufficient | Derive any missing reaction terms before considering a successor |
| Local vector coherence P0699 | Historical joint failure, including galaxy errors and cluster multiplicity | Audit continuum definition and conservation of a new action |
| Barycentric radial alignment P0701 | Historical weighted-galaxy and cluster-topology failures | One global radial direction did not encode the needed multi-center structure |
| Tensor/nonlocal nonlinear response | Broad category with numerous failed and incomplete specific ancestors | Derive a partition-invariant action and one physical metric |
| I4 thermodynamic stress / I5 baroclinicity | Not yet a tested source mechanism in the final handoff | Repair and validate the spectral measurement model first |

The compact machine queue is `configs/gravity_sigma_directions_v1.json`. It
also preserves the current bounded TRIMOND continuation. No old controller is
silently renamed and promoted as a new theory.

The later handoff supersedes optimistic statements in the older overview. Its
main lesson is that a promising radial acceleration curve did not ensure correct
cluster image roots, multiplicity, parity and positions. This is evidence about
the tested candidates, not a proof that every scalar theory must fail.

The unfinished I4/I5 route has a concrete prerequisite: one region's emission
measure was assigned to partial observation/CCD footprints without the required
geometry fractions. The handoff calls for fixed footprint fractions over 5,082
cells and a rerun of 494 regions before constructing the source variables.
Unresolved spectral complexity must also remain visible. We have not run this
measurement repair, derived I4/I5 maps, or scored a new gravity law from them.

## New test: does the coherent base conserve total momentum?

P0696 adds a radial potential about the baryonic center of mass:

\[
\Phi=\Phi_N+\int^r\delta g(s)\,ds,\qquad
\delta g=\frac{\sqrt{g_0^2+4a_0g_0}-g_0}{2},\qquad
g_0=\langle-\mathbf g_N\cdot\hat{\mathbf r}\rangle_{\rm sphere}.
\]

Curl-free forces and covariance under rotations/translations do not by
themselves imply zero total internal force. A closed instantaneous isolated
matter model must also account for momentum exchange consistently. An action
derivation provides this obligation; for comparison, standard QUMOND explicitly
derives its conservation laws from its action.
[Milgrom, *Quasi-linear formulation of MOND*](https://arxiv.org/abs/0911.5464).

We froze `gravity_coherent_momentum_v1.json` and committed it as `5c6c4928`
before computing the audit metrics. The test uses two analytic Plummer clouds,
with masses 1 and 2 at z=+2 and z=-1, both of scale radius 0.5. Units have
G=1 and a0=1. The entire source has center of mass zero; no equilibrium is
assumed or needed for an instantaneous internal-force test.

The shell field is calculated analytically. For each cloud displaced by d,
the mean potential is

\[
\langle\Phi_N\rangle=-\frac{2GM}
{\sqrt{(r+d)^2+b^2}+\sqrt{(r-d)^2+b^2}}.
\]

Differentiating gives g0 independently of the angular force quadrature. The
audit integrates `F_z = integral rho*g_z dV` over the source using Gaussian
angular quadrature and Simpson integration in log radius. Reflected, symmetric
and concentric sources are controls. Radial/angular resolution increases from
1025/128 to 2049/256, then the integration bounds widen to 0.00005–200 with
2305/320 nodes.

| Quantity | Result |
|---|---:|
| Asymmetric correction force F_z, widest integral | +0.337515922559 |
| Corresponding center-of-mass acceleration | +0.112506362346 a0 |
| Force divided by integral rho*abs(g_N) dV | 0.066742272383 |
| Largest normalized Newtonian net force, all scenes/grids | 8.97e-14 |
| Largest normalized symmetric correction force | 1.96e-18 |
| Reflection mismatch | 1.65e-16 relative |
| Force change on resolution refinement | 2.38e-13 relative |
| Force change on widening the radial domain | 1.79e-11 relative |
| Independent analytic-shell / angular-quadrature discrepancy | 5.12e-13 relative |
| Omitted mass fraction, widest asymmetric integral | 9.38e-6 |

All eight declared numerical controls pass. The nonzero force is **6.67% of
the specified internal Newtonian force-magnitude integral**; this percentage is
a normalization of a synthetic conservation defect, not an observational error
or discovery probability. The reflected source reverses the defect's direction.

Unmodified Sigma code was also executed from the sealed snapshot on Cartesian
49, 65 and 97 cubed grids. It returns positive correction forces 0.33473,
0.32539 and 0.32801 in the same units. This supports the same qualitative
defect but is **not a converged precision replay**: the finest differs from the
continuum force by 2.82%, the coarsest misintegrates the mass, and the 65 cubed
grid has a Newtonian quadrature imbalance of 0.07253. These limitations are
retained in the result. The resolved witness comes from the independently
checked continuum integral, not from declaring those Cartesian grids accurate.

Conclusion: **the exact P0696 base cannot serve as a closed momentum-conserving
instantaneous gravitational law.** This is a synthetic theoretical witness,
not an empirical rejection of all coherence models. A dynamical field or a
properly varied nonlocal energy could introduce reaction terms; those would
define a different, as-yet-unvalidated model. Simply subtracting the mean force
after solving would not establish a first-principles theory. This audit does
not evaluate P0697's routing addition or P0699/P0701's complete blends.

## Evidence and continuation

Evidence is retained under `work/gravity-first-principles/coherent-momentum-001/`:
configuration, exact running-source snapshots, input hashes, radial force
profiles, all controls, legacy replay and receipt. Result SHA-256:
`8ab9d42c79bc17863e4cf711143121effa36f6f1bf4af9153a43c7f05d8c6d99`.
Seven new analytic/implementation tests pass.

The next useful actions are to require momentum and source-partition audits
for any new coherence action; retain the I4/I5 measurement route as unresolved;
and complete the already-derived multi-field external-boundary solver before
claiming any Solar-System clearance for the 36 bounded TRIMOND cards. The
scalar galaxy and cluster development failures remain in the ledger. The
discovery goal remains active; no common three-regime law has been found.
