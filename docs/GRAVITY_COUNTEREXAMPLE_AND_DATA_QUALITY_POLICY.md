# Gravity counterexample and data-quality policy

This companion policy applies to the ordered search in
`docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md` without changing the byte-frozen goal
file or any completed receipt.

## Core rule

One empirical counterexample does not kill a formula family. Astronomical measurements combine
heterogeneous instruments, catalog reductions, distance estimates, inclinations, stellar
population models, selection effects, and incomplete nuisance variables. Every counterexample is
preserved, but an isolated exception first creates a data-quality and influence audit obligation.

## Distinguish two kinds of failure

1. A **hard theoretical veto** can be decisive in one case when the derivation itself produces a
   forbidden result: a ghost or unbounded Hamiltonian, loss of hyperbolicity, a negative or
   nonfinite physical observable, a broken conservation identity, or a verified violation of a
   local/strong-field constraint inside the theory's declared domain.
2. An **empirical counterexample** is never an automatic one-object veto. It changes the evidence
   in proportion to its verified quality, uncertainty, leverage, recurrence, and independence.

## Required empirical reporting

For every future real-data result, retain and report:

- the full held-out loss and every object-level counterexample;
- the counterexample fraction, never merely a zero/nonzero flag;
- the result after removing only the single most influential comparative residual;
- a frozen small-fraction trimmed comparison that removes the largest absolute comparative
  influences symmetrically between the candidate and its strongest ordinary baseline;
- whether either robust diagnostic changes the sign of the measured improvement;
- broad response-blind strata, so repeated failure in a physical regime is not hidden by trimming;
- a separate list of missing, invalid, ambiguous, or quality-limited measurements.

Dropping an object after inspecting its answer does not change the primary result. If removing one
object changes the conclusion, label the formula `SINGLE_OBJECT_SENSITIVE` and retain it for an
unchanged fresh-data test; do not promote it and do not erase it. A family-level empirical reject
requires an aggregate held-out failure, a repeated independent failure pattern, failure against a
stronger ordinary baseline, or an unchanged replication failure—not the existence of one object.

Confirmation objects remain sealed until their predeclared gate authorizes access. Suspected data
problems are checked from provenance or independent measurements, never by silently deleting an
unfavorable response.
