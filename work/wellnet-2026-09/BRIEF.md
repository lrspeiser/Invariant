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
