# External normal-field domain extension

Six new conditional source-field solves completed in132seconds on one CPU
thread, within the frozen300second budget. Two12kpc-domain cases were reused
with hashes. Physics, source conversions, thickness, smoothing lengths and unit
normal external field were unchanged. No astronomical response or measured
external amplitude was fitted. Full3D fields remained in memory only.

## Larger domains reduce the earlier boundary failure

Numbers below are relative RMS changes in center-subtracted vectors. Every
height group must additionally remain below8%.

| Smoothing ell | 12->18 kpc halfwidth | 18->24 kpc halfwidth | Largest height error in18->24 |
|---|---:|---:|---:|
| 0.25 kpc | 1.58% | 0.441% | 0.759% |
| 0.5 kpc | 1.67% | 0.493% | 0.552% |

All larger-domain comparisons pass, including raw fields. The raw18->24 changes
are0.487% and0.604%. This is concrete evidence that the smaller-box failures
were sensitive to domain extent, rather than a reason to retune the density law.

## One mesh check still fails

At the18kpc domain, halving the grid spacing gives center-relative global RMS
changes2.05% for ell0.25 and1.39% for ell0.5. However, ell0.25 at height0.2kpc
changes11.21%, failing the8% group threshold. The ell0.5 mesh comparisons all
pass; their worst height-group change is4.80%.

We therefore retain one numerical failure. A global RMS pass must not hide a
local group failure, and raw-field convergence must not be substituted for
convergence of the center-relative perturbation.

For the smoother ell0.5 case, the fine18 center acceleration is approximately
0.42737 times the imposed unit normal field. The center-relative field RMS is
0.08577 per unit applied field; it varies from0.01913 in the plane to0.16131 at
height1kpc. This is a conditional differential response in the imposed field,
not a percentage of an observed gravity discrepancy.

The finest mesh was run at18kpc halfwidth, not24kpc. Consequently the separate
passing ell0.5 domain and mesh checks are not a joint finest-mesh/largest-domain
validation. We explicitly do not claim that fine18 alone validates coarse24.
Ell0.25 also retains the stated height-group mesh failure. No extra physical
variations, hidden grid changes or further runs were added to replace these.

## Verification

Original analytic controls were rerun and the independent anisotropic sparse
reference was hash-bound. Source and code hashes were verified. All six solvers
converged; the largest boundary-forcing-normalized residual is7.03e-11.
An independently written arithmetic replay verifies all3,080 saved vector rows
and12 comparison cases; maximum metric difference is1.39e-17, with identical
pass/fail decisions. Hashes, partial progress and all samples remain available.

This narrows the numerical problem without claiming an observational external
field effect. Independently constrained surroundings, source/motion likelihoods
and a physically complete coupling remain outside this calculation.
