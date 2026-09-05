# Overlapping ranges and elastic gravity: first mathematical assessment

The user's two ideas are retained as research directions. The first explicit
translations explain how extra gravity can become more important with distance,
and expose constraints that a successful theory must satisfy. None is a new
validated gravity law. No new astronomical response or reserved data was scored.

The existing source-tail precision repair remains pending. This assessment
does not change its failed derivative checks or admit that provider for
production.

## The distinction that matters for outer stars

In a circular orbit, inward acceleration is v^2/r. If outer speeds are nearly
constant, the required total inward acceleration fades approximately as 1/r.
It still gets weaker with distance, but more slowly than the 1/r^2 field of a
compact Newtonian source. It therefore becomes stronger relative to Newtonian
gravity as one moves out.

| Dominant extra pull | Pull when distance doubles | Outer speed when distance doubles |
| --- | --- | --- |
| Inverse distance, 1/r | Half | Unchanged |
| Constant pull | Unchanged | Multiplied by sqrt(2) |
| Literal Hooke spring, proportional to r | Twice | Twice |

These consequences assume circular orbits and a regime where the specified
extra pull dominates. They are not fits to a real disk. A literal spring is
incompatible with flat speeds in that regime. Nonlinear elasticity or a
finite-range elastic response is a different hypothesis and remains open.

A dimensionally consistent version of adding powers is

    g(r) = GM/r^2 [1 + (r/L)^p].

The radius is measured against one physical length L. The cases p=1, 2, 3 give
the three rows above. Their added potentials in units GM/L are respectively
log(r/L), r/L, and (r/L)^2/2. These unsaturated extra potentials cannot be set
to zero at infinity; any isolated theory using them needs a stated boundary
condition or a physical change of behavior at larger scales.

## A concrete overlapping-range model

One smooth range contribution is

    delta psi_lambda = -GM alpha [1-exp(-r/lambda)]/r,
    delta g_lambda/g_Newton = alpha B(r/lambda),
    B(u) = 1-(1+u) exp(-u).

Near the source B(u)=u^2/2+O(u^3); far beyond the range it approaches one.
Adding several such contributions produces a gradual increase in relative
pull. A finite sum eventually saturates, so its farthest-distance force again
falls as 1/r^2, with a larger coefficient. These are known subtracted-Yukawa
responses; the repository's Item 19 records their relation to Item 16. Their
interpretation as fundamental particles requires a separate action and sign
analysis. This audit claims only an effective static pair potential.

For an illustrative continuous ladder, choose d alpha=d lambda/L between
lambda_min and lambda_max. This choice is imposed for exploration, not
derived from microscopic principles. Its exact extra-force factor is

    E(r) = [lambda_max(1-exp(-r/lambda_max))
           -lambda_min(1-exp(-r/lambda_min))]/L.

There are three regimes:

- r much smaller than lambda_min: E is approximately
  r^2(1/lambda_min-1/lambda_max)/(2L), so the relative extra pull is small.
- lambda_min much smaller than r much smaller than lambda_max: E is
  approximately r/L, giving the desired approximately 1/r extra force.
- r much larger than lambda_max: E approaches
  (lambda_max-lambda_min)/L, so the growth stops.

This construction shows how overlapping ranges can produce an intermediate
flat-speed region. A logarithmic extra potential and its 1/r force are known
in the [Tohline-Kuhn/nonlocal-gravity literature](https://arxiv.org/abs/1111.4702).
That precedent does not validate this spectrum or its application to clusters.

The audit compares the closed-form continuum against independent quadrature,
checks that the force is the derivative of the potential, and verifies a
four-body energy-gradient calculation. Central pair forces conserve total
force and torque to numerical precision. Splitting a source element into two
coincident half-mass contributions leaves its external field unchanged. A real
extended galaxy would require the full source convolution; multiplying a
Newtonian disk curve by E measured from an arbitrary center is not that model.

## The mass-scaling constraint

For a universal mass-linear kernel with a genuinely flat compact-source
regime, the 1/r contribution gives

    v_flat^2 = GM/L.

With a fixed universal L, sixteen times the source mass gives four times the
speed. Published galaxy measurements instead place baryonic mass approximately
proportional to flat speed to the fourth power: sixteen times the mass then
corresponds roughly to twice the speed. The measured slope and its systematic
uncertainties are discussed by [Lelli et al.](https://arxiv.org/abs/1901.05966).

To obtain that scaling from this expression, the effective length would need
to emerge as L=sqrt(GM/a0), rather than be fitted separately for every object.
This is a constraint on the specified linear, fixed-range, compact-source
limit. It is not a population likelihood or a rejection of all multiscale
theories; finite-radius geometry and nonlinear source response require their
own calculations.

Nor can we simply give every independently named mass piece a sqrt(m) field.
Splitting a source into two equal coincident pieces then increases its field
by sqrt(2). The bookkeeping has changed while the physical source has not.
The audit retains that exact counterexample. A nonlinear equation for the
field of the entire continuous source avoids this particular construction.

## What an elastic interpretation would need

A literal Hooke law is too steep to supply a dominant flat-curve force.
A nonlinear constitutive response could instead make weak fields respond
more strongly. For example, a relation between source flux and field strength
of the form D=mu(g/a0) g, with mu approximately g/a0 in weak fields, gives
g approximately sqrt(GMa0)/r for an isolated sphere.

That static mathematical route is already present in the action-derived
[Bekenstein-Milgrom field theory](https://adsabs.harvard.edu/pdf/1984ApJ...286....7B):

    div[mu(|grad Phi|/a0) grad Phi] = 4 pi G rho.

It is an analogy to nonlinear material response, not evidence that space is
literally rubber. Deriving it from a field action addresses conservation;
it does not by itself derive a0, establish microscopic elasticity, or supply
all relativistic observables. [Verlinde's elastic-response proposal](https://arxiv.org/abs/1611.02269)
is another relevant precedent, distinct from a simple spring force.

To add a first-principles theory, we would need the actual degrees of freedom,
energy and couplings that determine the range weights or elastic response,
including any cluster-dependent behavior, with one universal rule. We would
then test the complete source field, conservation, stability and matter/light
predictions before claiming cross-regime success.

The Solar System is inside the Galaxy, so large-range contributions cannot
simply be declared absent here. A common uniform acceleration cancels from
relative planetary motion in the linear pair model; spatial gradients remain.
In nonlinear theories even a uniform external field can change internal
dynamics. The local test must include the Sun, planets and their external
environment rather than infer a pass from small heliocentric separations.

## Recovered project history and next discriminating work

The copied historical records are summaries of previously exposed work, not
fresh independent confirmation:

- Item 19 tested static scalar/vector massive-carrier libraries and recorded
  the exact subtracted-Yukawa equivalence. Its attractive, locally normalized
  branch cannot provide the requested point-source enhancement. Its empirical
  failures apply to the declared library and source model.
- Item 38 retained a Verlinde-like transition formula with an exploration
  improvement and unsuccessful unchanged transfers. It was explicitly not
  promoted; the cluster table was model-dependent and previously exposed.
- The G4 search retained a cross-scale parent, while its tested multiscale
  lane scored worse than its RAR comparator. Those radial projections were
  not complete conserved relativistic theories.

The next useful step is a physically derived nonlinear range spectrum or
elastic field that passes the source-partition and mass-scaling constraints
and changes the existing cluster/local tension for a reason beyond retuning.
The known linear ladder remains a comparator, not a newly discovered law.
The separate source-tail precision repair is still required for the current
length-sensitive action's galaxy calculations.

Exact controls, settings, source-document hashes and the result receipt are
retained in `work/gravity-first-principles/overlap-elastic-001`. The audit adds
zero astronomical scores and establishes zero validated universal laws.
