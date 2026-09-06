# Memory and repeated relay: first formula tests

Status: THEORY_BENCHMARK_ONLY. Four tests pass; the deterministic sweep contains 150 parameter combinations and 48 spatial modes. No galaxy, star-age, or lensing measurements were scored. This experiment tests whether the proposed mathematics behaves as intended, not whether gravity operates this way.

## The two formulas

Let rho be ordinary source density and K a finite spatial spreading rule. S is a secondary effective source, not a particle population or an amount of proven gravitational energy.

Single generation: `tau dS/dt + S = K rho`.

Repeated relay: `tau dS/dt + S = K(rho + alpha S)`.

Tau controls how quickly the response changes. Alpha controls how much the response feeds back into itself. Our ring operator preserves the sum of its input and is symmetric and invariant under cyclic translations; the geometry is manufactured and periodic. It is not a reconstruction of a real galaxy. This normalization is mathematical bookkeeping, not physical energy conservation.

In a spatial mode with eigenvalue lambda, the frequency response is

`S/rho = lambda / (1 - alpha lambda + i omega tau)`.

For a persistent uniform input, lambda=1. Its equilibrium response gain is `1/(1-alpha)` and its relaxation time is `tau/(1-alpha)`. With no feedback, waiting longer changes the transient, but does not increase the final static response beyond K rho.

## What emerged

| Feedback alpha | Equilibrium secondary-source gain | Relaxation time | Fraction built after 10 tau | Lag at omega tau=0.1 |
|---|---:|---:|---:|---:|
| 0 | 1 | tau | 99.995% | 5.71 degrees |
| 0.5 | 2 | 2 tau | 99.33% | 11.31 degrees |
| 0.9 | 10 | 10 tau | 63.21% | 45 degrees |
| 0.99 | 100 | 100 tau | 9.52% | 84.29 degrees |

These are secondary-source gains, not total gravitational-force multipliers. Acceleration still requires solving for the spatial field. In particular, the uniform ring mode is a response-system control, not the force of a uniform galaxy.

**Promising direction:** finite, subcritical feedback can smoothly accumulate a substantial secondary response. It also makes an additional falsifiable prediction: the response should trail changes in the ordinary source. Different spatial modes respond differently, so redistribution can alter structure instead of acting as one universal multiplier. For sigma=2 cells, the first two ring Fourier eigenvalues are approximately 0.73475 and 0.29116, while the uniform eigenvalue is 1. Feedback therefore preferentially amplifies the broadest mode in this manufactured example.

**Main challenge:** large enhancement comes with long memory in this formula. At alpha=0.99, a theoretical gain of 100 is not actually reached after 10 tau; the response is only 9.52 times the input at that age. Changing tau can make equilibration fast, but then this mechanism no longer provides long-lived age dependence. Old static rotation curves alone cannot identify tau at all: every positive tau has the same equilibrium.

At alpha=1 there is no finite equilibrium under persistent input: the response grows linearly. Above 1, it grows exponentially. The CSV keeps formal sinusoidal transfer values in those cases, but they are not stable long-time responses. They are excluded from numerical periodic-state claims. Thus an unrestricted chain of mini-generators is not viable in this linear toy form.

## What to change or test next

1. Start with a finite spatial kernel and feedback below its stability threshold. If the kernel normalization is Q rather than 1, the uniform-mode condition becomes alpha Q < 1. An untruncated NFW-shaped source kernel has integrated weight `ln(1+x)-x/(1+x)`, which keeps growing: 1.489 at x=10, 5.910 at x=1000, and 12.816 at x=1,000,000. A finite outer scale or other change is needed before applying this finite-budget recursion argument to it.
2. To separate large static enhancement from long lag, change the mechanism explicitly: for example, introduce an independently constrained response amplitude or nonlinear saturation. Those additions must then be tested for stability and predictive value; neither was fitted or validated here.
3. A true test of memory needs a source history or changing system, not just color-derived stellar age. The relevant clock in these equations starts when the mass distribution changes, which need not coincide with when its stars formed. Compare a static kernel first, then ask whether time information predicts independent motion better.
4. The temporal equation here does not enforce finite-speed spatial propagation. A causal kernel with retarded time, and a physical energy/momentum budget, remain necessary if this is to become a physical gravity theory. No light-deflection prediction follows from the present equation alone.

## Verification and limits

Independent scalar analytic solutions and a matrix-exponential reference agree with numerical integration. Maximum scaled step error is 1.18e-10; maximum scaled periodic-state error is 2.44e-9, below the frozen 1e-8 gate. Frequency checks start in the exact periodic state and cover the shorter of 20 radians or 20 relaxation times; they test integration, not spontaneous approach to equilibrium. Step checks separately test buildup from zero.

Tightening integration tolerance from 1e-6 to 1e-8 to 1e-10 reduces absolute error from 8.99e-7 to 1.40e-8 to 3.47e-10. Increasing ring resolution from 16 to 32 to 64 cells changes the first Fourier eigenvalue from 0.7347478 to 0.7347205 to 0.7347134. This is numerical convergence evidence for the toy periodic ring; it does not establish insensitivity to a galaxy's outer boundary or validate a three-dimensional source map.

Related primary research on spatial nonlocal effective sources: [Chicone and Mashhoon, Nonlocal Gravity: Modified Poisson's Equation](https://arxiv.org/abs/1111.4702). Our temporal relay law is a proposed test equation, not a reproduction or validation of that complete theory.

Reproduce from the repository root: `python -m unittest discover -s tests -p test_mond_atlas_delay_experiment.py -v`, then `python scripts/mond_atlas_delay_experiment.py --output <new-output-directory>`. Existing results are not overwritten.
