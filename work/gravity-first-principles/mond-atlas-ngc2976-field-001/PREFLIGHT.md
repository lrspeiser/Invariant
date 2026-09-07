# NGC2976 cell-integrated conditional forces

SOURCE_BLOCKED before implementation. No observed motion/lensing response read.
This adds a source adapter and runner, not a new gravity law or solver stencil.
Reuse unchanged disk-backed Poisson/QUMOND, moment/boundary and force-sampling
operators. QUMOND equations are bound to Milgrom 2010, arXiv:0911.5464. The source
packet bindings inherit S4G, THINGS and HERACLES measurement papers and limits.

Exactly integrate the frozen positive bilinear planar basis into solver cells
with the existing rectangular thin projector. Multiply by continuous exponential
layer integrals. No mass correction by renormalizing point samples is allowed.
A cell average retains the surface-density unit; after conversion to Msun/kpc2
and multiplication by a normalized inverse-kpc vertical average, the result is
Msun/kpc3. Report finite-domain tails, source integrals and grid moments.

Before real source packets are opened, run independent source-cell controls
(including subcell spikes that point sampling loses), existing analytic Plummer
and anisotropic manufactured fields, dense-versus-streamed solver controls and
force-sampling controls. Reuse the already verified source projector. Reject
zero/negative layer heights and nonzero source edge nodes; do not reinterpret a
zero-height sheet as a measured finite volume. All original sources remain intact.

Cross factors 1 and 4 with stellar height .1/.4 kpc, gas height .2 kpc. For each
of four sources solve base/lateral/vertical/larger-box grids, then sample fixed
rings at z=0,.1,.3 kpc. Fields are computed from combined ordinary matter before
the nonlinear MOND step. Retain all 16 fields, convergence checks and failures.
No response thresholds, source heights, conversions or masks are tuned to force
results. Thresholds are frozen in the config before these fields exist.

Differences between source-grid cases can include inverse regularization and
missing-source sensitivity. Height changes include different fitted planar
coefficients. Small PDE residuals alone do not establish physical or continuum
accuracy; per-source resolution and box checks are separately required. Unknown
exterior matter and instrumental/source uncertainty prevent observational scoring.
