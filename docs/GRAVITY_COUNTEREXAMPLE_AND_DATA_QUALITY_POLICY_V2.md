# Gravity counterexample and data-quality policy v2

This is the mandatory policy for Item 38 and all later gravity-roadmap work. It strengthens, and
does not mutate, the byte-frozen historical policy used by completed Items 34–37:
`GRAVITY_COUNTEREXAMPLE_AND_DATA_QUALITY_POLICY.md`.

The machine-enforced contract is `configs/gravity_empirical_counterexample_policy_v1.json`,
implemented by `sigma_theory_compiler.gravity_counterexample_policy`.

## Core rule

One empirical counterexample never causes terminal rejection or pruning of an exact formula or its
wider family. Astronomical data can contain errors or incomplete nuisance information from
instruments, catalog reductions, distances, inclinations, stellar-population estimates, sample
selection, and missing variables. Preserve every mismatch, but treat an isolated exception first
as an audit and influence obligation.

An isolated counterexample also does not block promotion by itself. If the aggregate held-out,
uncertainty, baseline, influence, quality, and all other frozen promotion gates pass, the formula
may continue with the exception disclosed. Count alone is never decisive.

## Theoretical and empirical failures are different

A single verified hard theoretical witness can be decisive within its proved scope: a broken
conservation identity, a ghost or unbounded Hamiltonian, loss of hyperbolicity, a negative or
nonfinite physical observable, or a verified local/strong-field violation inside the declared
domain. Reject an entire family only when the proof covers the entire family.

An empirical mismatch instead changes the weight of evidence in proportion to its measurement
quality, uncertainty, leverage, recurrence, and independence. Future reports must separately count
raw worse-than-baseline objects, quality-verified objects, and counterexamples whose comparative
residual remains meaningful after uncertainty and covariance.

## Mandatory empirical audit

Every future real-data result retains and reports:

- full held-out loss and all object-level residuals;
- raw, quality-verified, and uncertainty-resolved counterexample counts and fractions;
- measurement uncertainty/covariance and instrument/reduction provenance;
- relevant distance, inclination, calibration, and alternate-reduction sensitivity;
- missing-variable and missing-not-at-random risks;
- the result after removing only the single most influential comparative residual;
- a frozen small-fraction symmetric influence trim;
- sign changes under both influence diagnostics;
- broad response-blind strata and unchanged independent replications;
- every missing, invalid, ambiguous, or quality-limited measurement.

Post-response exclusions never change the primary result. Suspected data problems are investigated
through provenance or independent measurements, never by silently dropping an unfavorable object.

## Mandatory states

- `ISOLATED_EMPIRICAL_COUNTEREXAMPLE_RETAINED`: exactly one uncertainty-resolved exception;
  retain, audit, and test unchanged on fresh data.
- `SINGLE_OBJECT_SENSITIVE_RETAINED`: removing one object or applying the frozen influence trim
  changes the aggregate sign; retain but withhold promotion.
- `SURVIVES_WITH_COUNTEREXAMPLES`: the candidate wins its aggregate frozen comparison despite
  multiple exceptions; disclose them and continue the other promotion gates.
- `ROBUST_SCOPED_NEGATIVE_EVIDENCE`: a quality-passing held-out sample loses robustly; preserve the
  formula and seek an unchanged independent replication.
- `REPLICATED_NEGATIVE_EVIDENCE_TESTED_REPRESENTATION`: an independent dataset repeats the
  unchanged loss; retire only the exact representation in the tested domain, not every formula in
  the mechanism family.

Missing object records, a failed frozen quality floor, or an incomplete audit forces a retained
incomplete/quality-limited state. Finite empirical samples do not automatically prune whole formula
families. Confirmation data remain sealed until their predeclared gates authorize access.
