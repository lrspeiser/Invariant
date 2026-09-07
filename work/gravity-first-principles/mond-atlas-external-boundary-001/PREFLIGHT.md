# Fixed-physics external normal-field boundary extension

Conditional numerical investigation; SOURCE_BLOCKED for observed environment
claims. No changed permittivity, density conversion, source thickness, smoothing
length, external strength or sample geometry. Unit applied acceleration along z.
Use f4-stars-h0p4 NGC2976 source, ell=0.25/0.5 kpc and
epsilon=0.2+0.8*Gaussian(ell)*rho/(Gaussian(ell)*rho+1e7).

Reuse original external-program-001/run001 box results (halfwidth12,12,6;
spacing0.25,0.25,0.125) with input/vector hashes. New base-spacing domains:
halfwidth18,18,9 and24,24,12. New fine domain:18,18,9 at spacing0.125,0.125,0.0625.
Two smoothing lengths give6 new solves. Use original384 points plus origin.

Compare12base->18base,18base->24base and18base->18fine. Raw and center-relative
vector RMS must be<5%, every height group<8%, same1e-8 unit-field floor. Retain
all failures. Fine18 does not alone validate coarse24: mesh/domain coupling
remains unresolved without a joint endpoint check, which is outside this fixed
increment. No observed external amplitude or velocity is read or fitted.

Before source solves: existing analytic slab, uniform coefficient and polarity
controls, plus separately assembled anisotropic sparse reference residual review.
Verify original bound source component hashes. Available RAM must exceed30GiB
before each grid; expected largest289^3 nodes. Bound job to300seconds with
solver callback checks; one CPU thread. Full3D fields RAM-only; save sampled
vectors, source mass accounting, residuals, convergence and budget failures.
No new private raw arrays. Do not enlarge the budget or change thresholds after
the results. An incomplete final case remains incomplete.
