# Open-gravity builder and solver admission policy v1

This append-only policy applies to every new source builder, three-dimensional reconstruction,
numerical solver, and gravity/light operator added after this policy. It does not mutate or
reinterpret any already-frozen roadmap, receipt, result, or authorization boundary.

## Required evidence before response scoring

A new implementation may not advance to response scoring merely because it runs or produces
plausible-looking output. It must bind all of the following:

1. at least one real, public source dataset suitable for the inputs it claims to construct;
2. the primary paper documenting that dataset and its measurement or conversion assumptions;
3. at least one independent published solution, analytic limit, manufactured solution, or
   separately implemented reference calculation capable of falsifying the implementation; and
4. target-free checks for dimensions, limiting behavior, conservation where applicable,
   resolution and convergence, boundary sensitivity, and symmetry or invariance.

The source dataset used to construct a prediction must be distinguished from the response data
used to score it. Source assumptions, benchmark thresholds, exclusions, and numerical gates are
frozen before any response values are opened.

## Dimensional and observational fit

The source must contain the information needed by the implementation's advertised geometry.
A general three-dimensional builder therefore needs resolved spatial source information, or a
paper-defined deprojection whose assumptions and degeneracies are frozen and tested. A one-
dimensional rotation curve cannot by itself validate a three-dimensional reconstruction. A
spherical cluster profile can validate only the spherical sector of a three-dimensional solver,
not its general nonspherical behavior.

A primary paper can admit a theory-only or mechanics benchmark when no suitable observation is
available, but it cannot turn that implementation into an observationally tested result. Real-
data claims require a suitable public dataset in addition to the paper. If the available source
supports only a restricted symmetry, the receipt and result must retain that restriction.

Before implementation begins, every proposed builder or solver must therefore declare one of:

- `DATA_AND_PAPER_ADMITTED`: suitable public source data, its primary measurement paper, and an
  independent implementation benchmark are all identified;
- `THEORY_BENCHMARK_ONLY`: primary equations and independent benchmarks exist, but no suitable
  public source exists, so observational scoring is forbidden; or
- `SOURCE_BLOCKED`: a required source, paper, geometry, boundary condition, or validation target
  is missing.

## Fail-closed dispositions

- If a suitable public source does not exist, the implementation is `SOURCE_BLOCKED`.
- If its independent benchmark fails, it is `BENCHMARK_FAILED` and the discrepancy is retained
  as a counterexample.
- If it passes only synthetic or analytic checks but has no real source, it remains theory-only.
- Neither a source nor benchmark failure may be repaired using response residuals, target labels,
  source retuning, or threshold retuning.

An unconventional theory is allowed to disagree with GR, MOND, or another familiar model. It is
not allowed to disagree with its own declared equations, dimensions, boundary conditions, or a
published control it claims to reproduce. Novel theory branches remain explorable while broken
implementations remain unable to create scientific scores.

## Minimum receipt fields

Every admitted package must record exact dataset and paper identifiers; file or release hashes
where available; benchmark definitions and tolerances; source-versus-response classification;
all failed cases; numerical convergence evidence; access accounting; and an explicit statement
that every required source and implementation gate completed before response access.
