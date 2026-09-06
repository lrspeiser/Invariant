# Independent conditional external-field review

The implementation and saved comparison arithmetic pass independent review.
This does not make all physical field results numerically admitted: the applied
normal-field cases fail the prescribed enlarged-domain gates.

An independently assembled sparse anisotropic7x8x9 Dirichlet operator agrees
with the parent potential within3.05e-11. Its residual norm divided by the
boundary-forcing norm is4.581916e-11, matching the parent's4.581915e-11. This is
the correct nonzero normalization for the tested homogeneous equations; using
the identically zero volume source norm would not be meaningful. A zero
applied-field case would require an absolute residual or an explicit zero-solve
branch, because that boundary-forcing norm would also vanish. No zero-amplitude
case is invoked in the actual program.

All five bound code/protocol/manifest hashes and eight unique source-component
hashes were independently verified. No source arrays or observed velocities
were opened for this audit. All6,160 saved vector rows were grouped and all16
global/height comparison sets were independently recomputed; every RMS value
and every pass/fail flag agrees exactly.

## Numerically admitted pattern

For the applied field along the disk major axis, both smoothing lengths pass
the mesh and enlarged-domain criteria for raw and center-relative fields.
All quantities below are per unit of the imposed external acceleration, not
fractions of an observed gravitational anomaly.

| Density smoothing ell | Center field vector | RMS field relative to center | Midplane RMS relative field | z=1 kpc RMS relative field |
|---|---|---:|---:|---:|
| 0.25 kpc | (0.68664,-0.00173,0) | 0.13138 | 0.11530 | 0.16084 |
| 0.5 kpc | (0.66846,-0.00517,0) | 0.11518 | 0.10459 | 0.13629 |

The nonuniform conditional material model therefore produces a differential
response after the common center acceleration is removed. It is not just a
uniform translation, unlike the interior of the analytic sphere control. The
response varies with height and includes transverse components. It is not
necessarily a radial inward pull, and these numbers cannot establish an
observed external gravitational influence without an independently measured
environment and an admitted dynamical prediction.

## Remaining direction/domain failure

The applied normal-field calculations pass fine-to-finer mesh checks but fail
enlarged-domain checks. Raw-field RMS changes are5.51% and7.01%, above the5%
limit. Center-relative height-group changes near the disk reach19–29%, above
the8% limit. Normal-field source patterns and their contrast with the major-axis
case are therefore not admitted quantitative physics results yet. A small
elliptic-solver residual does not remove this boundary sensitivity.

The finer-box normal center fields and conditional patterns remain in the JSON
for diagnostic completeness, marked all_gates_passed=false. They must not be
promoted selectively because they appear physically interesting.

No implementation error or gate arithmetic discrepancy was found. The model
remains a fixed-density linear boundary response, not MOND's nonlinear external
field effect, a self-consistent equilibrium or a causal/relativistic theory.
