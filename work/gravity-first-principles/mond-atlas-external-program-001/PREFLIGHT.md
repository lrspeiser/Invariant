# Applied external field on the conditional neighborhood-density model

SOURCE_BLOCKED for observational external-field claims. No actual external mass,
external acceleration, galaxy velocity or lensing data are opened or fitted.
Use the exact NGC2976 f4-stars-h0p4 stellar/HI/CO source bindings in
mond-atlas-spatial-program-001/source-bindings.json and their inherited primary
S4G/THINGS/HERACLES references. The source is conditional, not measured 3D truth.

For each existing density scale ell=0.25,0.5 kpc, retain
epsilon=.2+.8*Gaussian(ell)*rho/[Gaussian(ell)*rho+1e7 Msun/kpc^3].
Solve div(epsilon grad deltaPhi)=0 with deltaPhi=-e.x at the rectangular boundary,
for a unit applied acceleration e along the major disk axis and along its normal.
This is the linear external-boundary response of the SAME fixed-density static
PDE. It adds to the isolated potential by superposition. It is not a MOND fit,
causal transport, physical reflection, equilibrium solution or relativistic theory.
No observed external amplitude is assumed. Unit-response fields can subsequently
be scaled to any declared amplitude; that is not an additional experiment.

Use base/fine/finer halfwidths(8,8,4), spacings(.25,.25,.125),
(.125,.125,.0625),(.0625,.0625,.03125), plus box halfwidths(12,12,6) atbase spacing.
Sample the existing384 source-field points and the origin. Report both raw
external response and g(point)-g(origin); the latter removes a common translation
relative to the center, not a proven mass-weighted center-of-mass acceleration.
Numerical comparisons require relative RMS <5% and each sampled height <8% for
both raw and center-relative vectors, with an absolute 1e-8 unit-field floor to
handle exact uniform controls. Preserve all failures. Own enlarged-box checks.

Pre-source controls: uniform epsilon gives exact uniform applied acceleration;
linear superposition/polarity; smooth slab epsilon=1+.2z has exact flux solution
Phi=-5*log(1+.2z), g_z=1/(1+.2z). Require finest sampled force RMS<.002 and
improvement under refinement. Independent analytic sphere transmission/flux and
separate sparse-operator controls provide additional reference checks.
The existing solver's RHS-relative residual is undefined for zero RHS. Evaluate
the full homogeneous residual relative to the Dirichlet boundary forcing instead,
with threshold1e-8 and cg_info0; do not use an arbitrary tiny RHS denominator.

Bound 16 source solves to360seconds of solve time, one CPU thread, no new saved
full 3D fields; retain sampled vectors, source hashes, controls and failures.
The next observational requirement is an independently constrained external
environment plus admitted source/motion/selection/noise likelihood. A response
to an invented boundary field is a conditional prediction, not evidence that
the observed environment causes a galaxy's apparent gravity anomaly.
