# Manufactured local-calibration sensitivity of existing source fields

CONDITIONAL_SOURCE_FIELD_SENSITIVITY; SOURCE_BLOCKED for observed calibration.
No new real-source PDE solve and no observed velocities/environment read or fit.
Freeze u_lab=[0,0.01,0.1,1,10,100], epsilon_lab=(u_lab+0.2)/(u_lab+1).
These are manufactured scenarios, not estimates of the Solar neighborhood or
laboratory. In the restricted constant-laboratory-coefficient calibration,
G_bare=epsilon_lab*G_measured. At fixed source density and epsilon geometry,
multiply existing SELF-generated candidate vectors by epsilon_lab, including
their consistently scaled self-field boundary. Never scale imposed external
fields with this rule. No external-program fields are inputs.

Inputs: normalization-001 theory receipt; refraction-program-001/coherence-scale/
run001 finer ell0.25/0.5 sampled vectors, and finer002 Newton sampled vectors.
Use same384 source points; assert positions match. Require existing selected
smoothed and Newton field convergence gates. Preserve unsmoothed-model failure;
do not import its force vectors. Bind source manifest/packet hashes and field
outputs before sensitivity calculation. Each scalar normalization preserves
field direction, source shape and signs; it does not fit or repair them.

For each case/height report norm ratio to Newton, number >Newton, radial inward
component (-g dot rhat), inward/outward counts and differences relative to
Newton. Radial enhancement ratios and thresholds require Newton and candidate
radial components both positive. Retain exceptions explicitly. A larger norm
does not imply more central attraction if directions differ.

Threshold algebra: if R>0 is a norm or positive-inward ratio at epsilon_lab=1,
enhancement requires epsilon_lab>1/R. For1<R<5, u_lab>(1/R-0.2)/(1-1/R).
R<=1 cannot enhance at finite u_lab>=0; R>5 enhances throughout allowedu>=0;
R=5 is equality at u=0 and enhancement onlyu>0. Report strict inequalities and
boundary cases. Thresholds are required conditions from these predictions,
not inferred true laboratory parameters. Counts are descriptive grid points,
not independent observations or confidence intervals.

Before field access: analytic constant-medium ratio cancellation and a small
manufactured variable-epsilon linear PDE scaling check (sameRHS andselfboundary
multiplied by epsilon_lab), tolerance1e-9. Explicit unchanged-boundary counterexample
is algebraic: a nonzero imposed homogeneous solution does not share the scaling.
Independent saved-vector/threshold arithmetic replay will follow. No physical
parameter selected by resulting enhancements.
