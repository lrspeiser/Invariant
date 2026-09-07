# Logarithmic static susceptibility: one conservative kernel tested

**THEORY_BENCHMARK_ONLY.** One kernel, two inherited oscillator frequencies, eight initial conditions at two numerical resolutions: 16 integrations, not16 distinct formulas. No observed galaxy, cluster, Solar-System or lensing scores. No claim that time creates energy. This developmental hypothesis follows the Gaussian kernel's inadequate outer radial shape.

The new coupling supplies a static inward response with finite effective total mass and a limited intermediate region resembling an inverse-radius force. All manufactured mechanics and trajectory gates passed. This establishes an internally consistent benchmark, not a successful gravity theory.

## Formula and independent limits

Write s=sqrt(r²+b²), C=omega² eta²/2, and h=eta sqrt(mi mj) sqrt(log(1+L/s)). The conserved Hamiltonian contains oscillator energy qdot²/2+omega²q²/2-omega²qh, plus softened Newton and particle kinetic energy. Completing the square gives omega²(q-h)²/2-omega²h²/2. At equilibrium q=h:

U_extra=-C mi mj log(1+L/s),

g_extra=C M L r/[s²(s+L)].

Here g is inward magnitude per test mass in the single spherical source interpretation. In actual time evolution the force is omega²q grad(h), and q can depart from h; an instantaneous inward force is not guaranteed for arbitrary oscillator states. Each pair carries an internal coordinate, not a propagating spacetime field.

The effective enclosed mass is (C M L/G)r³/[s²(s+L)], with density

rho_extra=(C M L/4piG)[L s²+b²(3s+2L)]/[s⁴(s+L)²].

All factors are nonnegative, and total extra mass is C M L/G. The center has finite density 3 C M L/[4piG b²(b+L)] and g proportional to r. At b=0, rho=M_extra L/[4pi r²(r+L)²], the Jaffe form; the force is C M L/[r(r+L)]. Thus b<<r<<L yields approximately1/r, and r>>L gives1/r². The softened model differs from Jaffe at the center. The frozen L/b=20 permits only a modest intermediate interval; it does not demonstrate a many-decade flat rotation curve.

Walter Jaffe,1983, *A simple model for the distribution of light in spherical galaxies*, MNRAS202,995-999, [primary publisher](https://academic.oup.com/mnras/article/202/4/995/1008198), DOI10.1093/mnras/202.4.995. Publisher metadata/abstract inspected; full ADS PDF retrieval failed. The density-potential equivalence above is independently derived and quadrature-checked, not represented as a verified transcription of the unavailable paper. Oscillator elimination uses the standard mechanical construction discussed in Caldeira and Leggett1983, Annals Physics149,374, section3, [DOI](https://doi.org/10.1016/0003-4916(83)90202-6), used in the preceding benchmark. These papers do not establish the proposed gravitational interpretation.

Units: q has sqrt(mass)*length, eta length/sqrt(mass), omega inverse time, and L,b lengths. C has length²/(mass*time²), so U has energy units and g acceleration units. Dimensionless constants G=1,eta=.15,L=1,b=.05 and omega=.2/2 lack a physical galaxy calibration. Changing omega at fixed eta changes both timescale and static strength C; this is not a pure memory-timescale comparison.

## What passed

Run002 completed in8.14seconds. All16 integrations conserve energy, momentum and angular momentum below the frozen1e-5 gate. Largest energy drift7.82e-14; exact circular-position error1.89e-12; base/fine final-position relative differences below5.88e-12 versus1e-3 gate. The perturbed initial tangential speeds are3%below circular, not a fit.

Independent potential finite differences agree with force to6.16e-9 absolute; enclosed-mass quadrature relative errors below9.65e-16; total-mass quadrature below2.23e-16. Positive-density and small/large-radius limits pass. Manufactured energy-gradient force errors below5.63e-11, mapped source-splitting acceleration errors2.23e-16 and corrected energy errors4.00e-15 pass. Zero coupling reproduces softened Newton exactly. Parent independent reduced-radial Radau replay agrees across8cases and3208samples to1.53e-11 maximum absolute error, without importing production functions. See [radial review](independent-radial-review/receipt.json) and [analytic review](independent-analytic-review/README.md). The separate analytic implementation verifies force differentiation to1.61e-10, mass quadrature to4.16e-17 and all18 saved static rows to8.89e-16; no substantive equation bug was found.

Run001 stopped at NumPy boolean JSON serialization before any trajectories. Its script, bindings, controls and partial output remain untouched. Run002 only unwraps NumPy scalars for serialization and selects a new output directory; no formula, gate or parameter changed. See SERIALIZATION_REPAIR.md.

## What remains challenging

- With frozen constants, total effective extra/source mass is only0.045% for omega=.2 and4.5% for omega=2. The extra/softened-Newton force fraction is C L/G*s/(s+L), hence bounded by these values. No evidence that this strength covers galaxy discrepancies.
- Increasing source mass at fixed radius increases both forces equally. There is no mass-triggered high-acceleration suppression. At b=0 the extra/Newton fraction tends to zero as r approaches0; for fixed positive b it instead tends to C L/G*b/(b+L), about0.214% for omega2. Both softened forces vanish at the exact center. This is not a tested Solar-System bound.
- Undamped oscillators do not relax automatically: at fixed separation with q initially0, q=h(1-cos(omega*t)). Preparing q=h is an assumed initial condition. Real formation histories, damping with a conserved reservoir and propagation are absent.
- The fixed-total-mass energy bound is -(G/b+C log(1+L/b))*sum(mi mj). Positive b is essential; the unsoftened logarithmic extra potential diverges negatively at contact. Fixed-particle-mass total energy can scale as N², so boundedness is not thermodynamic extensivity or general many-body orbital stability.
- Coincident source splitting works with explicitly mapped oscillator coordinates and subtraction of internal constant self-energy. It does not establish equivalence for arbitrary separated substructure or arbitrary internal states.
- The radial law recovers a known potential class. Re-expressing it with an oscillator does not independently predict its kernel, scale or strength. Positive spherical effective density is not evidence for actual hidden material and is not a measured three-dimensional galaxy source.
- No covariant spacetime theory, lensing prediction, cosmology, real source history or observational fitting was supplied. All response access remains zero. No additional kernels or tuning were attempted.
