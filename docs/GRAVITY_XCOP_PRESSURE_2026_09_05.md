# Scalar gravity: cluster pressure and Solar System tension

The nine fixed scalar candidates do not yet supply a shared empirical solution.
At a0 = 5e-11 m/s², the only sampled scale within the earlier historical Cassini
screen, their nominal median predicted/observed cluster-pressure ratios are
0.566–0.635. Larger scales bring pressure closer to the measurements but exceed
that local screen. This is a conditional development result, not a proof against
all scalar laws, an untouched confirmation, or a gravity discovery.

The original latest-push integration remains the research base. The active goal
still requires galaxy outskirts, cluster matter and light, and precision local
gravity from one law and one global parameter set.

## Fixed scope and access

This campaign evaluates three action shapes at each of three previously declared
acceleration scales, plus Newtonian baryons and a canonical empirical radial
acceleration relation (RAR) comparator. Every model uses the same parameters in
all eight clusters. There are 21 fixed global nuisance scenarios and 1,848
pressure-profile predictions, plus 88 nominal refinement checks.

The clusters are A1644, A1795, A2142, A2255, A2319, A3266, A85 and ZW1215. Their
responses were previously exposed in the project and are development data.
A2029, A3158, A644 and RXC1825 remain reserved. The reader rejects unregistered
clusters before opening files. No reserved cluster payloads were extracted or
parsed. This study does not reopen other sealed products.

Only gas density, SZ pressure, available stellar profiles and pressure covariance
are used. The covariance FITS files contain catalogue mass metadata in another
HDU; those numeric columns were not parsed or used. R500 and P500 undo the release
units, rather than entering as candidate total-mass estimates. This remains
conditional on the release's distance and spherical reconstruction assumptions.

Thirty pressure bins are scored. Twenty-four initial bins are excluded by the
published beam/deconvolution limitation, twenty outer bins lack measured density
support, and eight measured outer pressure points are boundary conditions, never
targets. All 82 bin dispositions are retained. Scored radii collectively span
about 0.55–1.28 R500; this is an outskirts pressure test, not cluster-core,
member-orbit or lensing validation. The source publication describes deprojected
SZ pressure and the first-three-bin limitation in section 2.5.
[Ghirardini et al., thermodynamic profiles](https://arxiv.org/abs/1805.00042).

## Equation and observable corrections

For a regular isolated spherical source, the scalar field equation gives

    g_N(r) = G [M_gas(<r) + M_star(<r)] / r²
    y = g_N/a0, u = y² + epsilon²
    g(r) = g_N(r) [1 + u^(-1/4) (1 + u^m)^(-1 - 3/(4m))]

with epsilon = 1e-4 and m = 0.5, 1 or 2. These are previously constructed bounded
QUMOND action ansätze, not newly derived microscopic principles. The spherical
relation uses the total baryonic enclosed mass, without separately modifying and
adding component fields. Its regular spherical limit was independently checked
by the earlier nonperiodic field solver.

Let f = P_nonthermal/P_total and P_e be electron thermal pressure. With fixed gas
composition, the total-pressure equation implies

    d[P_e/(1-f)]/dr = -mu m_p n_e g
    P_e(r) = (1-f(r)) [P_e(r_b)/(1-f(r_b)) + integral_r^r_b mu m_p n_e(s) g(s) ds]

Here mu = 0.61, mu_e = 1.148 and f(r) = f_out r/r_b are declared assumptions.
For a varying pressure fraction, multiplying the pressure gradient by 1-f alone
omits a derivative term. The earlier Item59 adapter does that multiplication;
its nuisance can be interpreted as a gradient-support fraction, but not directly
as this total-pressure fraction. Its old results are retained and are not
relabelled as outputs of the corrected equation.
[Eckert et al., section 3.2, equation 6](https://arxiv.org/abs/1805.00034).

The old Item59 adapter also compares local P_e/n_e with released X-ray
temperatures. Those temperatures are projected spectroscopic measurements. This
campaign gives them no score; a verified line-of-sight and spectroscopic response
is required before using that observable. This limitation prevents inheriting
the old joint pressure/temperature interpretation.
[Ghirardini et al., sections 2.4.2 and 3.4](https://arxiv.org/abs/1805.00042).

Gas mass is integrated exactly for a log-linear density interpolation, with a
constant inner density to the origin and no outer extrapolation. Query radii
cannot change enclosed source mass. Published cumulative stellar profiles have
recorded monotone corrections, cubic inner continuation, and constant outer mass.
The maximum monotone corrections are 0.409% for A1795, 0.971% for A85 and 1.835%
for ZW1215. A2319 and A85 require some outer constant-mass continuation. The three
clusters without stellar profiles use one shared stellar/gas mass ratio. These
source choices need uncertainty treatment before stronger interpretation.

## Nominal comparison

The primary descriptive loss is the equal-cluster mean squared log10 pressure
ratio. The table reports its square root in dex. The pressure-ratio column is
the median of the eight cluster-specific median predicted/observed ratios.
Neither number is a significance test. All candidates use nominal f_out = 0.15,
stellar scale = 1 and missing stellar/gas ratio = 0.1.

| Model / shape | a0 (m/s²) | RMS pressure residual (dex) | Median pressure ratio | Prior Cassini screen |
|---|---:|---:|---:|---|
| Newtonian baryons | — | 0.42047 | 0.3878 | Not evaluated here |
| Empirical RAR | 1.2e-10 | 0.14063 | 0.7245 | Not inherited from bounded actions |
| m = 0.5 | 5e-11 | 0.25058 | 0.5664 | Within, both assumed fields |
| m = 1 | 5e-11 | 0.20382 | 0.6250 | Within, both assumed fields |
| m = 2 | 5e-11 | 0.19678 | 0.6349 | Within, both assumed fields |
| m = 0.5 | 1.2e-10 | 0.14591 | 0.7182 | Outside, both assumed fields |
| m = 1 | 1.2e-10 | 0.11651 | 0.7669 | Outside, both assumed fields |
| m = 2 | 1.2e-10 | 0.11494 | 0.7692 | Outside, both assumed fields |
| m = 0.5 | 2e-10 | 0.08132 | 0.8372 | Outside, both assumed fields |
| m = 1 | 2e-10 | 0.06170 | 0.8802 | Outside, both assumed fields |
| m = 2 | 2e-10 | 0.06111 | 0.8814 | Outside, both assumed fields |

Newtonian baryons is not a fitted dark-matter baseline. RAR is an empirical
comparison curve, not a complete candidate theory or an established cluster fit.
The experiment does not claim to outperform a full general-relativistic
dark-matter model.

Each of the three low-a0 candidates has greater pressure loss than RAR in all
eight clusters, under the nominal assumptions:

| Cluster | Scored bins | Published stellar profile | Pressure ratio, m=0.5 | m=1 | m=2 |
|---|---:|---|---:|---:|---:|
| A1644 | 4 | No | 0.5937 | 0.6467 | 0.6517 |
| A1795 | 3 | Yes | 0.6015 | 0.6580 | 0.6643 |
| A2142 | 4 | Yes | 0.5468 | 0.6115 | 0.6222 |
| A2255 | 4 | No | 0.5627 | 0.6277 | 0.6371 |
| A2319 | 4 | Yes | 0.6125 | 0.6930 | 0.7097 |
| A3266 | 3 | No | 0.5700 | 0.6223 | 0.6326 |
| A85 | 4 | Yes | 0.5629 | 0.6199 | 0.6271 |
| ZW1215 | 4 | Yes | 0.5462 | 0.6006 | 0.6072 |

## Covariance, nuisance and quality audit

The initial attempt, `xcop-pressure-001`, stopped before gravity scoring because
A2319's covariance radii differed from the high-level release. The successor
registration permits only a single recorded index-wise radius dilation, with
identical relative bin geometry to 1e-8. A2319's factor is 1.0163447251; the other
seven are 1. The pressure normalization is 1.0430175208 for A2319 and
1.0095198683 for the other seven. This empirical mapping does not establish its
astrophysical cause or validate every covariance assumption.

High-level quoted errors also differ from the mean-scaled native covariance
diagonal. Across all released bins, including excluded inner bins, ratios range
from 1.0011 to 1.7141. Three covariance treatments are therefore retained:
native correlations transferred to high-level error diagonals; mean-scaled native
covariance; and diagonal high-level errors. The first is the primary covariance
diagnostic, explicitly an unverified mapping assumption. None is described as a
complete source-and-observation likelihood. The release and exact archive/member
hashes are in the acquisition manifest.
[X-COP data release](https://dominiqueeckert.wixsite.com/xcop/data).

Every covariance includes uncertainty in the measured outer boundary:

    residual_i = prediction_i - observation_i
    k_i = (1-f_i)/(1-f_boundary)
    C_residual = B C_pressure B^T, with B = [I, -k]

The nominal model ordering is identical under all three covariance treatments.
Their absolute losses differ substantially: for m=2 at a0=5e-11, the mean
whitened squared residual per target is 21.03, 29.62 and 14.48 respectively.
These are conditional diagnostics, not p-values or calibrated reduced chi-squared
tests with all nuisance uncertainty accounted for.

The 21 global scenarios comprise nominal, twelve single changes and eight joint
corners. They vary f_out between 0 and 0.3, stellar scale between 0.7 and 1.3,
missing stellar/gas ratio between 0.05 and 0.2, density by coherent quoted lower
or upper errors, distance by ±10%, and pressure calibration by ±10%. Distance
changes consistently scale radii, inferred density, stellar mass and SZ pressure.
The joint corners combine f_out, stellar scale and distance. These are finite
sensitivity choices, not uncertainty intervals, per-cluster optimizations or an
exhaustive nuisance envelope.

All three low-a0 candidates remain worse than RAR in the aggregate in all 21
matched scenarios. For m=2, candidate-minus-RAR loss stays positive between
0.01312 and 0.02343 dex². All three 2e-10 candidates remain better than RAR in
all 21 scenarios, but remain outside the prior Cassini screen. The best shape
among those high-a0 candidates changes in two joint corners: m=1 in corner 2
and m=0.5 in corner 4. Consequently the exact shape ranking is not universally
robust, even though the low-a0 versus RAR comparison is.

Dropping the largest absolute comparative cluster contribution, trimming the
smallest and largest contributions symmetrically, or splitting by stellar-profile
availability preserves the sign of each nominal candidate-minus-RAR comparison.
For low-a0 m=2, the mean difference is 0.01894 dex²; omitting ZW1215 gives
0.01840 and symmetric trimming gives 0.01905. Both stellar-profile strata have
positive differences. Full object contributions and residual arrays are retained.

Raw comparative loss counts are not physical falsifications. The report records
zero *quality-verified* and zero *uncertainty-resolved* counterexamples because
the required audit is incomplete, not because there are no true discrepancies.
Missing pieces include joint density/stellar covariance, distance/calibration
cross-covariance, clumping, departures from spherical equilibrium, nonthermal
profile uncertainty, source continuations, selection and independent replication.
The classification is `QUALITY_LIMITED_EVIDENCE_RETAINED`; no family is pruned.

## Cross-regime implication and next work

The historical local screen is Q2 in [-3, 9] × 1e-27 s^-2, based on a published
Q2 = (3 ± 3) × 1e-27 s^-2 summary and the earlier declared two-sigma screen.
It uses two assumed constant Galactic external fields and a point-source scalar
calculation, not a current joint ephemeris refit or full Solar System pass.
[Hees et al., Cassini analysis](https://arxiv.org/abs/1402.6950).
The quadrupole method and field scenarios are documented in the
[prior audit](GRAVITY_EXTERNAL_QUADRUPOLE_2026_09_05.md) and
[Hees et al., interpolating-function constraints](https://arxiv.org/abs/1510.01369).

This finite grid points to a conflict between increasing cluster response and
keeping the local external-field quadrupole small. It does not exclude intermediate
parameters, other action shapes, different source assumptions or additional
fields. Raising a0 separately for clusters would violate the goal's universal
parameter requirement. The following work remains useful:

1. Build an observed galaxy-source reconstruction for the isolated field solver,
   using eligible development galaxies and explicit source uncertainty. No galaxy
   response was scored in this pressure campaign.
2. Test whether source geometry or additional fields provide a distinct cluster
   response while preserving signed member confinement and the scalar local limit.
   The multi-field external-boundary solver remains unsupported; it cannot inherit
   the scalar Cassini calculation.
3. Resolve the pressure-product mapping and projected-temperature response before
   stronger cluster likelihood claims. Retain this pressure result while that
   work proceeds.
4. Continue action-level conservation, perturbative health and matter/photon
   coupling. No lensing prediction or three-regime discovery is established.

## Numerical validation and reproduction

All 130 focused tests pass, including exact source-mass integrals checked by
independent quadrature, the varying-pressure-fraction equation checked by an ODE
solver, boundary covariance checked with independent noise draws, unit-invariant
whitening, source-access restrictions, and rejection of nonuniform bin remapping.
All focused lint checks pass. The largest nominal pressure change when refining
2,049 to 4,097 integration nodes is 7.1471e-6, or 0.000715%, below the predeclared
5e-5 target. This checks numerical integration, not the physical source assumptions.
CI includes these synthetic controls on Windows and Linux without opening data;
the remote CI matrix has not been run by this local experiment.

Evidence: `work/gravity-first-principles/xcop-pressure-002/` contains started
configuration and code hashes, all source packets and bin metadata, complete
model/scenario/object residuals, robustness comparisons, refinement results and
receipt. Result SHA-256:
`da012532daaf0e6bd932d3c53962798eb95f24844218e47699f6f5a1363d4ffc`.
The preflight-only failure and its exact implementation are retained in commit
`373a9340`, including the unchanged v1 contract. Successful scoring uses the v2
contract, which was declared after the normalization check but before scoring.
The existing source contract had CRLF runtime bytes while Git stores LF; its
exact hashed bytes are preserved under `input-snapshots/` and bound by
`input-snapshot-map.json`. The difference is verified to be newlines only.
All other recorded code/configuration/predecessor bytes match the committed
versions. Use the snapshot when reproducing the exact input hash on another OS;
the JSON values and all source-file hashes are identical under either newline form.

The 21 required density/pressure/stellar FITS payloads are hash-bound by the
existing development source contract; 29 already-exposed files were hydrated
locally, including eight temperature files whose numeric responses are not scored.
Use the existing archive with SHA-256
`0edf5038b419b70d070b73b22f4801e27f318b0854db61eec52142c27c140d94` and
`scripts/restore_gravity_xcop_development_covariance.py --archive PATH` to restore
only the eight covariance members. It verifies existing files and never replaces
them or extracts reserved products. Hydrate only the explicitly required LFS
members from the source contract, rather than broadly opening the archive.

Run `python scripts/run_gravity_xcop_pressure.py --output NEW_UNUSED_DIRECTORY`
with NumPy 2.2.6, SciPy 1.16.1 and Astropy 7.1.1, or compatible versions checked
by the control suite. Output directories and evidence files are append-only.
The figure can be regenerated with `scripts/plot_gravity_xcop_pressure.py` and
an unused `--output-stem`; it verifies both linked result hashes before plotting.
