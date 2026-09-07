# Force to emission integration contract

Status: THEORY_BENCHMARK_ONLY. Freeze this before executing the new adapter.
No observed source-region responses are accessed. Reuse the existing scalar
pressure balance and particle emission implementations, not a new force law.

Inputs are a steady axisymmetric surface column (kpc, consistent mass units,
km/s), its density-weighted inward radial force through the entire column,
independent integrated emission per ring, and explicit tracer line dispersion.
The emitting layer is thin; the dynamical column may have depth. This is a
conditional effective model, not a solution of vertical equilibrium or a
general warped/streaming gas flow. A supplied midplane force is not automatically
the required density-weighted column force. Pressure support and tracer line
width are distinct inputs. No implicit equality or observed normalization.

Use v_phi^2 = R g_column + R dPi/dR / Sigma, with zero mean radial and
vertical flow. Negative v_phi^2 invalidates the entire steady model; no clipping
or removal of offending rings. Radii strictly increase; emission nonnegative.
Project by Rx(inclination) then Rz(position angle), receding positive LOS.
Input ring emission divides uniformly among periodic angular quadrature nodes.
Reuse the existing exact Gaussian channel integral, conservative deposition,
normalized finite sampled beam, and complete finite-band/field flux accounting.

Predeclared controls: cold and supported harmonic solutions; independently
rotated Cartesian positions and velocities; face-on systemic LOS; linear
emission scaling; unchanged dynamics when only tracer width changes; impossible
balance rejection; nonzero mean flow rejection; reversed radii rejection;
independent voxel calculation using Gaussian CDF, direct tent assignment and
direct convolution. Relative cube tolerance 1e-11, geometry/analytic tolerance
1e-12. Explicit cropped-band/field case must retain positive losses. A 128 versus
256 azimuth calculation must differ by less than 1% cube L2; report failure if
not. No observational fit or source admission follows from passing controls.

References and physical derivation are already bound in
mond-atlas-pressure-support-001 (Wang et al. 2010, Iorio et al. 2017), and the
instrument primitives in mond-atlas-motion-controls-001. This step joins those
previously tested components; it does not count their old tests as new evidence.
