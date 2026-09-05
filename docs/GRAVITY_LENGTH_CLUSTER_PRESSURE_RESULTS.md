# Same constants across local screening and cluster pressure

Thirteen of the 54 previously registered length-action cards both lie within the
two historical local-gravity summary screens and predict cluster pressure better
than the fixed RAR comparator. Their primary loss is lower in every one of the
eight nominal development clusters. Their equal-cluster losses are lower in all
25 matched global scenarios, including all three covariance treatments. Removing
the most influential cluster or symmetrically trimming two clusters preserves
the primary-loss comparison's sign.

This is a conditional opening for the next galaxy calculation. It is not a
discovered first-principles law, a full Solar System fit, an adequate absolute
cluster fit, or independent confirmation. Joint source/response covariance and
geometry remain unresolved. Quality-verified and uncertainty-resolved
counterexample counts remain zero; raw comparative counts are reported separately.

## Fixed action and global constants

The inherited nonrelativistic trial action is

\[
P(x,h)=x+xK_m(x+h),\qquad K_m(u)=\frac{Q_m(u)-u}{u},
\]

\[
x=|\nabla\psi|^2/a_0^2,\qquad
h=\ell^2\psi_{,ij}\psi_{,ij}/a_0^2.
\]

The bounded scalar kernel and its removable origin are defined in the existing
length-action derivation. The physical flux is the full variation
\(J_i=P_x\psi_{,i}-\ell^2\partial_j(P_h\psi_{,ij})\).
For a regular spherical source, the inward acceleration is its radial component:

\[
g=P_xg_N-\ell^2\left[P_h'g_N'+P_h g_N''+
\frac{2P_h}{r}\left(g_N'-\frac{g_N}{r}\right)\right].
\]

Here \(g_N=GM/r^2\), \(g_N'=4\pi G\rho-2g_N/r\), and
\(g_N''=4\pi G\rho'-2g_N'/r+2g_N/r^2\). Gas and stars enter the same
mass, density and density derivative. Neither tangential Hessian component nor
the density-gradient term is omitted. A scalar interpolation shortcut would be
an incorrect implementation at nonzero length.

All 54 inherited cards were evaluated: three shapes (0.5, 1, 2), three acceleration
scales (5e-11, 1.2e-10, 2e-10 m/s²), and six lengths (0, .001, .01, .1, 1, 10 pc),
with epsilon=1e-6. Their card hashes match the local experiment exactly. The 20
cards outside the local screen were retained alongside the 34 within it. No
constants were changed by cluster or regime and no new winning card was selected.

## Source reconstruction before pressure prediction

The inherited stellar enclosed-mass interpolation has density jumps, which makes
naively taking higher derivatives unsuitable for this action. The new source
convolves positive \(dM/d\log r\) with a Gaussian of fixed logarithmic width,
interpolates the logarithm of that positive density, then integrates that same
density to obtain mass. The reconstructed mass and its derivatives therefore
describe one source. Independent quadrature, derivative, total-mass and homology
controls pass.

The first response-free audit used width .01. It passed numerical checks but
exceeded the fixed 2% stellar-mass fidelity limit at two A85 source knots; its
maximum change was 4.88%. That failure remains in `xcop-smooth-source-001`.
Before any new pressure prediction, width was narrowed to .0025 for every source
and cluster, with grid refinement from 16,385 to 32,769 nodes. No limit was relaxed.
The second audit passed: maximum stellar-mass change 1.23%, maximum gas-density
smoothing shift 0.0344 quoted errors, and maximum total-mass error 5.97e-8.
The inherited monotonic correction to the original stellar table is recorded
separately; these figures do not validate that older correction.

The campaign retains the original 21 global nuisance scenarios and adds four
nominal source probes: widths .005 and .01, a flat outer gas-density continuation,
and extension of the original outer slope to three times the last measured
radius. Gas otherwise continues its last measured slope to twice the outer
radius before a finite-mass closure. These exterior assumptions are unmeasured.
Both wider-width probes exceed the primary A85 stellar fidelity limit and are
explicitly flagged sensitivity probes, not fitted acceptable alternatives.
The primary-width density-shift and outer-closure cases pass their source checks.

The eight development clusters, 30 scored pressure points and eight unscored
boundary points are unchanged. The pressure prediction integrates
\(d[P_e/(1-f)]/dr=-\mu m_p n_e g\), with the same smoothed gas density used in
gravity and the inherited global nonthermal-pressure scenarios. No reserved
clusters or new raw data were read. Existing exposure means development evidence,
not an untouched test.

## Results

The following table fixes the universal length at **1 pc**, which lay within both
local screens for every sampled shape and acceleration scale before this cluster
run. The pressure ratio is the median of cluster median predicted/observed
ratios. Loss is the equal-cluster mean squared log10 pressure ratio.

| Shape m | a₀ (m/s²) | Pressure ratio | Loss (dex²) |
|---|---:|---:|---:|
| 0.5 | 5e-11 | 0.5664 | 0.062792 |
| 0.5 | 1.2e-10 | 0.7182 | 0.021291 |
| 0.5 | 2e-10 | 0.8372 | 0.006613 |
| 1 | 5e-11 | 0.6250 | 0.041544 |
| 1 | 1.2e-10 | 0.7669 | 0.013576 |
| 1 | 2e-10 | 0.8802 | 0.003807 |
| 2 | 5e-11 | 0.6349 | 0.038722 |
| 2 | 1.2e-10 | 0.7692 | 0.013213 |
| 2 | 2e-10 | 0.8814 | 0.003735 |
| RAR comparator | 1.2e-10 | 0.7245 | 0.019778 |
| Newtonian baryons | — | 0.3878 | 0.176798 |

The length term changes nominal cluster pressure by at most **8.68e-11 fractionally**
across this grid relative to the same card at zero length. The largest target
acceleration change is 1.27e-9 fractionally. Thus the mechanism's observed role
here is to reduce the local quadrupole while retaining the higher-a₀ cluster
response. The small pressure changes lie below the numerical pressure error;
they are not individually resolved detections of a length effect in clusters.

The source-reconstruction change itself shifts the unchanged Newtonian and RAR
comparators' nominal pressures by at most 9.03e-6 and 1.87e-5 fractionally relative
to the earlier cluster run. It does not explain the improvement over RAR.

All 18 lower-a₀ cards remain worse than RAR. Among the 34 local-compatible cards,
13 improve the primary cluster loss; all 13 also improve the other loss treatments
throughout the declared nuisance grid. These finite comparisons do not constitute
a confidence region, exhaustive nuisance envelope, likelihood evidence for a new
theory, or permission to prune a physical family.

## Numerical verification and reproducibility

All 56 models, including the two comparators, passed the full-population admission
gate: **11,200 profile predictions** over eight clusters and 25 scenarios. Each
case separately refined pressure integration and source representation. Maximum
relative pressure changes were 1.41e-7 and 1.81e-8, respectively, below the frozen
5e-5 limit. No object or failed case was removed to obtain admission.

The verification rehashed 30 input snapshots, replayed 11,200 profile losses,
33,600 covariance losses with a separate dense linear solve, and 110 object-influence
comparisons. A Cartesian tensor-flux calculation with 8,193-node Simpson integration
independently reproduced all 448 nominal profiles within 4.17e-8 fractionally.
The focused synthetic and analytic suite has **190 passing tests**. The figure
was visually checked.

Reproduction commands, from the research checkout with `PYTHONPATH=src` and one
BLAS thread, use new output directories to preserve earlier receipts:

```text
python scripts/run_gravity_length_cluster_pressure.py --output <new-run>
python scripts/report_gravity_length_cluster_pressure.py --run <new-run> --verification <new-verification> --outputs <new-output-directory>
```

Evidence in `work/gravity-first-principles/`:

| Record | SHA-256 of result.json |
|---|---|
| xcop-smooth-source-001 | c496cba4461ec392832bac85b262dc045a76d1f7f1eeec866a7a3aaf7d588498 |
| xcop-smooth-source-002 | 490eba0a4d5860e698ce33c85e0681e102838de1f4242cc819cf0db585fda263 |
| length-cluster-pressure-001 | 3d97d94829de7fe2bce4e9b0eeeb30a4816a053b8cf8b9211ca6bbf5ab15cdbc |
| length-cluster-pressure-verification-001 | 833842b61eed5b9f44636b3dc7cde1ce85b6642eb06616f20674aaf5e5451727 |

`length-cluster-pressure-analysis-001` retains the cross-regime counts, complete
card-level nuisance comparisons, source flags and comparator bridge. Exact
executed code and configuration bytes are preserved alongside the campaign.

## Next discriminating work

Implement and verify the higher spatial derivatives of the existing NGC3198
source reconstruction, then test these same 54 cards with the inherited galaxy
uncertainties and comparators. The previous scalar or multifield galaxy scores
cannot be assigned to new nonzero-length cards. In particular, differentiating a
C1 potential spline three times is not a validated galaxy solver.

The action remains a bounded ansatz within the published GQUMOND framework.
Derivation of its kernel and constants from physical principles, a matter/photon
completion, stability and time evolution, lensing, full precision local tests,
and independent validation all remain required. The research goal stays active.
