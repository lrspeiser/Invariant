# Well-network programme — standing brief for every lane

Repo root: `C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration`
Your lane directory: `work/wellnet-2026-09/<lane>/`  (create it; write everything there)
Master record: `C:\Users\henry\dev\gravity-discovery-program.md` (read-only for you;
the orchestrator appends).

## Why this programme exists

Run J exhaustively enumerated every one-, two- and three-term point-local law over
a 1,898-atom bank — 1,135,961,540 forms at k=3 — and none beat the radial
acceleration relation out of sample. Training gain rose monotonically with
complexity while blind gain fell monotonically, and a physics-free twin with the
radial structure permuted away recovered +2.1 to +3.2 percentage points of the
apparent gain on its own.

The conclusion is NOT that gravity has been searched. It is that the SPARC bench
carries only two independent directions (a_N and r), so more transformations of
those two variables cannot manufacture a new measurement. This programme moves
the search to **operators, spatial structure, directions, nonlocality,
environment and time** — things that require new data and new solvers, not new
algebra.

## Hard constraints — these are not negotiable

1. **KiDS and wide binaries are PERMANENT SEALED HOLDOUTS.** They never enter any
   fit, screen, calibration or model-selection step, at any stage, for any
   reason. Do not load them. Do not look at them.
2. **No data that presupposes dark matter.** In particular: do NOT treat a
   GR-derived convergence map, an NFW-fitted mass, or a parametric lens model
   whose mass is tied to light by construction as if it were a raw observation.
   The observables are: image positions, redshifts, shear/ellipticity catalogues,
   X-ray surface brightness and temperature, SZ signal, galaxy velocities.
   Published mass maps may be used ONLY for debugging and must be labelled as
   such.
3. **Global gravity parameters only.** A candidate law gets universal constants.
   It never gets a parameter fitted per galaxy, per cluster or per object.
   Measured baryonic quantities (M_b, r, kT, M_gas) are inputs, not parameters.
4. **Do not kill a candidate merely because it fails somewhere.** If it beats
   Newton or GR in some regime, record where and by how much. Elimination
   requires a stated, quantified failure of a stated requirement.
5. **Blind protection.** Any fitting splits by object, declares the split before
   looking at residuals, and touches the held-out set once.

## Failure modes this programme has already been bitten by

Check for each of these explicitly; say in your report that you did.

* **Shared-denominator artefacts.** A quantity appearing on both axes makes the
  naive null non-zero. `rho_p = -0.304` was retracted for exactly this: `ln E_obs`
  and `ln M_WL` had errors correlated at +0.96, so the naive estimator's null
  expectation was -0.12 and the observed value sat at p = 0.563 against its own
  null. Simulate the null with the actual error covariance, or use errors-in-
  variables.
* **Monotone-invariant statistics.** A rank statistic was bit-identical across
  three decades of the parameter it was supposed to measure. For every headline
  statistic S(theta), verify numerically that dS/dtheta != 0 over the tested
  range and print the spread.
* **Refitting on the held-out set.** A blind evaluation that re-solves for
  coefficients using blind data reported +2.17% where the correct frozen-
  coefficient procedure reported -3.73%. Fit on train, FREEZE, then evaluate.
* **Silent extraction failures.** A LaTeX table split across two `table*`
  environments returned 59 of 100 rows with no error. VizieR returns HTTP 200
  with a generic page for a nonexistent `-source=`. Assert row counts and column
  counts after every ingest, and echo the identifier back.
* **Test bugs that look like solver bugs.** Zero-flux boundaries are
  self-contradictory for an isolated source; flux must be measured on the FACE
  fluxes the discretisation conserves; a sphere in r is an ellipsoid in the
  metric u. A flat error curve versus resolution means a modelling mismatch, not
  a discretisation error.
* **Non-monotonic M(r) and clipped outer slopes** in lensing deprojection.

## Provenance requirements for anything pulled from the web

Every downloaded file gets a sibling `<name>.manifest.json` with: source URL,
retrieval timestamp (UTC, ISO-8601), SHA-256, byte size, row count, column names
with units, and the exact query issued. Keep the raw upstream response unmodified
alongside any cleaned file. If a source turns out not to contain what the brief
assumed, SAY SO PLAINLY and describe what it does contain — a corrected premise
is worth more than a substituted proxy.

## Existing infrastructure you may build on

* `work/gravitylab/solver.py` — 3-D Cartesian finite-volume solver for
  `div[mu(X) K grad Psi] = 4 pi G rho`, open Dirichlet boundaries from the exact
  constant-K monopole `Psi = -GM/(sqrt(det K) sqrt(r^T K^-1 r))`. Passes 7/7
  gates: analytic order 1.99 / error 3.63e-4, flux 6.3e-15, curl 4.7e-17,
  Newtonian recovery 2.33e-4, domain convergence 0.089%.
* `work/gravitylab/axisym.py` — cylindrical (R,z) solver, Freeman disk 1.4e-2,
  flux 5.6e-14. Note the half-space normalisation: the grid is z >= 0.
* `work/gravitylab/hypersearch.py` — atom bank + float64 Gram + batched Cholesky.
  77M optimally-fitted candidate laws/sec on the RTX 5090 via CuPy.
* `work/gravitylab/evolve.py`, `exhaustive.py` — the evolutionary loop and the
  complete k<=3 enumeration, with the four-target control harness.
* `work/gravitylab/data.py` — SPARC ingest, cuts declared before residuals,
  frozen stratified split `e5f74522`.
* `work/gravity-cluster-audit-2026-09/acquire/` — LoCuSS (Mulroy 2019), X-COP
  hydrostatic masses, Herbonnet 2020 WL, XXL, DiskMass VI/VII, with manifests.

Environment: Windows 11, PowerShell primary, Bash available. Python 3.13,
CuPy 13.5.1 on an RTX 5090 (compute capability 120, 32 GB). torch is CPU-only.
Bash heredocs collapse backslash escapes in this environment — use the Write tool
or `chr(92)` when a string must contain a literal backslash.

## What a good report looks like

`REPORT.md` in your lane directory: what you did, the numbers, the assumptions
you had to make, what failed, and what you could NOT establish. Plus machine-
readable results JSON and the code that produced them. Do not summarise or
soften a negative result — a clean negative with a stated power is worth more
than a hedged positive.

---

# ADDENDUM, 2026-09-04 — state after Runs J to AB

Read this before designing anything. Eight lanes have reported; the full record
is `gravity-discovery-program.md` Runs J-AB.

## The two structural results that should shape every new candidate

**1. THE BOUNDEDNESS THEOREM.** A response factor confined to a bounded range
cannot change the asymptotic force law; it can only renormalise the constant in
front of it. Found independently in two families:
  * nonlocal kernel: Phi = -GMF/r exact outside a source, qbar in [0,1) bounds F,
    so r v_c^2 -> GM sup F. Flat curves impossible.
  * well-network / pair-channel tensors: |S|_2 < 2/3 identically (measured
    0.666666666666, saturating it), so dln g/dln r = -2.0000 for every candidate
    against -1.00001 for AQUAL/QUMOND. Of 158,406,840 settings, 450 survive and
    ALL are the network switched off.
The repair is the same both times: **make the response unbounded, or source it
from something that is not a bounded ratio.** Adding r-dependence does not help.
NOTE the necessary-but-not-sufficient point: for Phi = -(GM/r)F, the force is
g = (GM/r^2)[F - r F'], so a flat curve needs F - rF' proportional to r AND
requires F - rF' > 0 so gravity does not reverse. At the measured
dlnF/dlnr = 0.899 the force is the small difference 0.101 F of two larger terms.

**2. POTENTIAL DEPTH, REACHED BY THREE INDEPENDENT ROUTES.** A cluster at 1 Mpc
and a galaxy outskirt sit at the SAME g_N/a0 — that IS the RAR — so no function
of g_N/a0 can separate them. Tensor solves: every viable point of 1,920 uses a
potential-depth gate, none uses an acceleration gate. The ladder: beta fitted on
galaxies+groups (+0.17188) transfers to held-out clusters (+0.16866), cutting
out-of-sample error 0.2917 -> 0.1066 dex. The Stage-1 screen explains why: a
bounded anisotropy can only rescale G, so the gate IS the mechanism.
**But the anisotropy does no independent work** — a scalar
a0 -> a0 f(|Phi_N|/Phi_0) reproduces the whole tensor map.

## Standing objections that any new work must answer

  * **The galaxy/not-galaxy class step is a serious null, not a curiosity.** It
    HAS a fitted parameter (4 vs 3) and DOES predict for a classified object. It
    beats potential depth on BIC (dBIC 17.6) and on frozen transfer (0.0954 vs
    0.1066 dex). Treat it as the primary null.
  * **|Phi_b| is defined only up to a constant** and needs an operational
    boundary rule stated in advance. Run Z showed the residual at fixed
    (g_bar, r) is EXACTLY the shape factor S (corr = +1.0000), so the boundary
    rule DEFINES the variable rather than conditioning it. Prefer a potential
    DIFFERENCE with a prespecified reference rule, and repeat under several
    defensible rules with one declared primary.
  * **The clusters are now validation data, not a pristine holdout** — four
    models have been ranked on them. A decisive claim needs a fresh sample.
  * **Detector calibration is conditional, not general.** The 5.2% tensor
    false-positive rate is correctly SIZED for the synthetic nulls tested; 2/2
    injection recovery is not a power characterisation. Power needs 100-1000
    injections spanning amplitude, orientation, thickness, morphology and noise.
    The nonlocal detector's 11.7% cannot be repaired by declaring p <= 0.02 —
    estimate the empirical critical value p* with P(p <= p* | H0) = 0.05.
  * **Weak lensing measures reduced shear g = gamma/(1-kappa), not mass.** The
    strongest model-independent statement available is "baryons plus the adopted
    RAR underpredict the observed SHEAR". Score against raw shear wherever
    possible, never against a mass catalogue derived under the standard lens
    equation.
  * **Streams, warps and satellites are not force samples.** They constrain an
    orbit in a global potential and must be FORWARD-MODELLED under each law, not
    converted into independent g_R and g_z points.

## Two more failure modes now on the checklist

  * **The shell average of a conductivity is physics, not bookkeeping.** The
    arithmetic mean of k turns over, fakes a saturation and reported A_T = -12.8
    where the harmonic mean, calibrated against full 3-D solves, gives -4.7.
  * **Selective refinement, not uniform.** A mass exponent cancels EXACTLY under
    uniform coarse-graining (p = 0.5, 1, 2 all give drift 0.28013 to five
    figures). Only selective refinement has teeth, and there only p = 1 is
    admissible. The physical-scale vs catalogue-row discriminator is
    dln(drift)/dln L: -3.11 genuine kernel, -0.55 family C, +0.12 row-counting.

---

# ADDENDUM 2, 2026-09-04 — STAGE 0, and what the first six lanes settled

## STAGE 0: nothing is interpreted until the search itself is calibrated

**The scalar null is any sufficiently smooth scalar response, NOT an off-grid
member of the existing atom bank.** Run AC repaired a degenerate null (truth
exactly in the basis, giving power 1.00 by construction) by injecting between the
bank's grid points. That is still a friendly null: it asks only whether the
tensor family can beat a slightly misspecified member of the SAME grammar.

    H0 : g = F_scalar(invariants) grad Phi_N,  F any smooth scalar response
    H1 : genuinely directional tensor terms

Null simulations must include at least five QUALITATIVELY different scalar
families, plus realistic false-anisotropy generators: baryonic ellipticity and
line-of-sight deprojection error, radial mass-to-light gradients, miscentering,
unresolved member galaxies, line-of-sight structure, PSF and shear-calibration
error, source-redshift error, wrong external-field strength, and triaxial mass
projected at varying orientations. A flexible tensor model can absorb any of
these and call it anisotropy.

**Calibrate the ENTIRE SEARCH, not a selected statistic.** The test statistic is
the improvement of the best tensor law found ANYWHERE in the full search, and
every null realisation must pass through atom generation, coefficient fitting,
hyperparameter and scale selection, pruning, winner selection, choice of axis and
choice of reported statistic. Otherwise the look-elsewhere effect of the search
is missing from the calibration.

Use THREE disjoint simulation sets: calibration simulations to set the critical
value, UNTOUCHED audit simulations to verify the false-positive rate, and
injection simulations to measure power. "5% by construction" is not enough when
the 95th percentile was set on the same simulations.

**Cross-fit the scalar nuisance.** Fit a highly flexible scalar model on one
fold, predict a different fold, and ask whether the residual has directional
structure. That makes it much harder for a tensor atom to win merely because the
scalar interpolating function was imperfect.

**Programme-level multiplicity.** Six adaptive lanes are six chances to find
something. A 3-sigma result in one lane is not a programme-level 3-sigma result.
Either calibrate under full null reruns or hold one untouched confirmation sample
per lead. **The SPARC blind split has now been viewed by many runs and should be
treated as a validation set**, even though each individual run touched it once.
KiDS and wide binaries remain sealed but do not provide an untouched confirmation
set for cluster tensor effects.

## INVARIANCE AND IDENTIFIABILITY GATES — run before any fitting

**Constant-K degeneracy.** For constant symmetric positive-definite K, the
substitution x' = K^(-1/2) x turns div[K grad Phi] into a plain Laplacian.
**A constant tensor is a coordinate stretch plus a transformed source**,
degenerate with source ellipticity, inclination, line-of-sight depth, distance
and baryonic deprojection. Detectable physics must come from SPATIAL VARIATION of
K, an INDEPENDENTLY KNOWN axis, or DISAGREEMENT AMONG PROBES — never from fitting
a constant ellipsoid.

**Potential-gauge invariance.** Absolute Newtonian potential is undefined until a
boundary convention is fixed, and Run AH measured the two admissible global rules
differing by 0.87 dex against an off/on margin of 0.9 dex. Every potential law
must use an operational DIFFERENCE with a prespecified reference — the nearest
gravitational saddle, a fixed overdensity boundary, a fixed multiple of a
baryonic scale radius, or the edge of a reconstructed environmental volume — and
must survive several defensible definitions with one declared primary.

**Coarse-graining invariance.** Mandatory for well-network tensors. Represent the
identical continuous galaxy as one catalogue object, ten subcomponents, and N
stellar-mass cells, and require convergence below a stated threshold. Also test
merging neighbouring entries, changing the detection threshold, varying
deblending, moving mass between intracluster light and members, and changing mesh
resolution. **A real coherence scale must be universal and appear in the field
equation; it cannot be set by the cataloguer.**

**Reciprocity and action.** Require F(x,x') = F(x',x) unless a separate field
explicitly carries the missing momentum, and test whether a candidate could arise
from an action by checking symmetry of its functional Jacobian. This will not
prove a relativistic completion exists, but it rejects nonconservative candidates
immediately. Runs AB and AH measured every family violating the third law at
0.2-16.5 of GM1M2/d^2 with no carrier declared.

## SPLIT THE TENSOR HYPOTHESES BY AXIS PROVENANCE

Do not search for "some tensor atom". Three physically distinct hypotheses with
DIFFERENT SPHERICAL LIMITS:

    source axis          anisotropy vanishes as the source becomes spherical
    external tidal axis  a spherical source still has an anisotropic field
    member-well network  set by the catalogue's well distribution

Run AC's near-spherical control conflated these — an injected dhat dhat^T imposes
an external axis, so K is not spherically symmetric and the blindness theorem
does not cover it. **Source-axis and external-axis calibrations must be
separate.**

Select development clusters by INDEPENDENT geometry — large projected
ellipticity, a well-measured member or X-ray axis, at least a stated misalignment
between candidate axes, adequate background density — using baryonic, X-ray and
environment maps, **never the lensing residual that will be scored**. Include a
near-round negative control. Then predict TWO-DIMENSIONAL shear: the PHASE of the
quadrupole matters as much as its amplitude, and a tensor aligned with the
external tidal axis but misaligned with the visible cluster predicts a rotated
quadrupole that source ellipticity cannot easily imitate. Azimuthally averaging
discards most of the directional information.

**Translate amplitude into observables.** Every candidate amplitude must be
quoted as a maximum predicted fractional acceleration or shear quadrupole, and
placed on the power surface. A null from a detector with zero power below the
predicted amplitude says nothing.

## FOUR ENVIRONMENTAL VARIABLES, NOT ONE

Potential depth is only one interpretation of the cross-class result. Test on the
same systems and the same folds:

    1  potential depth, as an operational difference
    2  VECTOR external acceleration g_ext   — permits directional cancellation
    3  DIRECTIONLESS inverse-square well strength sum_a G M_a / d_a^2, which does
       NOT cancel opposing wells and is much closer to "many surrounding
       concentrations collectively change the local state"; needs a smoothing
       scale and must pass coarse-graining
    4  tidal magnitude and shape, with eigenvectors kept separately

Run AH found 13 of 18 survivors gated on the tidal invariant, and that a cluster
member galaxy exceeds a cluster shell by 151x in |T| while being the DEEPEST
potential — so 1 and 4 order the key probe oppositely.

**Use system fixed effects where possible**: ask whether radial variation WITHIN
one object follows the proposed second variable, which avoids the class-label
problem entirely.

## LENSING CLOSURE IS NOT OPTIONAL

A modified Poisson equation determines the potential for slow matter; it does not
tell photons what to do. Fit dynamics FIRST, freeze, then predict raw shear under
NO SLIP. Only if there is structured failure, permit ONE universal slip
parameter, and test it on a new sample. **Never fit the gravity law and an
unrestricted lensing closure simultaneously** — that makes almost any dynamics law
fit shear. Strong-lensing time delays come later: image positions constrain
derivatives of the lens potential, while time delays carry Fermat-potential depth
and can separate otherwise degenerate closures.

## TWO BRANCHES THAT MUST NOT BORROW CREDIT

**Void-path redshift** is a separate hypothesis. Success in galaxy or cluster
gravity is not evidence for it. Its null must include reconstructed peculiar
velocities, lensing magnification, host-galaxy effects, calibration drift, survey
selection, and the covariance introduced by using redshift to build the void
catalogue. And any surviving law must stretch TIME as well as frequency — DES
supernovae measure time dilation consistent with (1+z), so a mechanism that only
drains photon energy while leaving event durations unchanged is excluded.

**Formation and stability.** A static equation fitting present-day rotation
curves does not show the law helped matter organise. Perturb a homogeneous
baryonic background, extract the linear response, and test stability, unbounded
modes, emergence of a preferred cosmic axis, growth rate without cold dark
matter, filament/pancake overproduction, finiteness at both wavelength limits,
and whether the response becomes statistically isotropic. A cheap linear gate
before any expensive simulation.
