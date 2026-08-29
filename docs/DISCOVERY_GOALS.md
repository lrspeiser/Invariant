# Discovery goals

Everything here is aimed at one outcome: **a verified new result in mathematics or
physics**.  Infrastructure appears only where it directly gates a discovery path.

Each goal states its **completion test** — the receipt or check that makes it done —
and its **discovery condition**, the strictly stronger event that would constitute an
actual finding.  Completing a goal is cheap; triggering its discovery condition is not,
and no goal may be reported as a discovery merely by completing.

Standing rules, unchanged: corpus absence is never novelty; survival is never proof;
kernel verification happens only in the kernel; negative receipts are deliverables. Every
empirical result is also claim-scoped: bounded publication, physical mechanism, and universal
theory are separate decisions, so failure outside a declared domain cannot silently erase a
valid in-domain result or be hidden from broader-claim adjudication.

## Current candidate inventory (what could become a discovery)

| Candidate set | Count | Status |
|---|---:|---|
| Continued fractions not in the builtin table | 32 | Survived 13 → 120 digits; screened only against a 15-entry table |
| Ulam spectral sidebands | 2 | Survived holdout; never screened against literature |
| Screened-gravity candidates | 23 (12 families) | Passed four synthetic gates + ghost/stability; K-mouflage class; **exploratory** confrontation with real rotation curves returned INFEASIBLE for all 12 (`runs/gpu-baryonic-screen/real-data-exploratory-v1.json`); the sealed no-refit trial is still unspent |
| Sealed negatives | 3 major | Genuine measured facts; modest as results |

---

## DG1 — Adjudicate the 32 continued fractions (highest value)

Build the prior-art corpus at real scale and screen every candidate.

**Completion test:** each of the 32 receives a typed classification —
`KNOWN_WITH_CITATION` (source + identifier) or `NOT_FOUND_IN_CORPUS` — against a corpus
of at least 10,000 records drawn from independently sourced material under the existing
source policy; a test asserts no candidate is left unclassified and that every
`KNOWN` carries a resolvable citation.

**Discovery condition:** at least one candidate classified `NOT_FOUND_IN_CORPUS` *and*
surviving DG2's proof routing.  Absence alone is never the claim.

## DG2 — Proof routing for continued-fraction identities

Classical CF identities are provable by mechanizable transformations (equivalence
transformations, Gauss's contiguous relations, the Euler–Minding correspondence between
a CF and its associated series).

**Completion test:** the router proves at least three *known* CF identities end to end
as controls, and every one of the 32 candidates receives `PROVED`, `REFUTED`, or a typed
`missing_proof_technique:<name>`.

**Discovery condition:** a candidate both `PROVED` and `NOT_FOUND_IN_CORPUS` — a machine
found, machine-proved, literature-screened identity.  This is the project's primary
target for "one new result".

**Status (receipt `runs/math/prior-art/cf-proof-routing-v1.json`): completed, discovery
condition NOT met.**  All 32 candidates are `PROVED`, none `REFUTED`, none left with a
missing technique.  The twenty already-`KNOWN` are re-proved by the equivalence chain as
controls; five cited classical identities are proved end to end from the analytic
techniques alone; a deliberately perturbed coefficient is refuted at the second decimal;
and Lambert's `coth` continued fraction — Bessel's recurrence, whose solutions are not
hypergeometric terms — returns the typed blocker instead of a proof.  Of the twelve that
were `INCONCLUSIVE_VALUE_MATCH`, two fall to Euler's series correspondence and ten to
Pincherle's theorem via a rational solution of the recurrence's Riccati equation; every one
of them lands on a `2F1`, `3F2` or `1F1` at a rational argument.  **That is exactly why the
count that matters is zero:** each proof exhibits its subject as an instance of a cited
classical family the corpus already carries, so all twelve are reclassified
`KNOWN_BY_PROOF_FAMILY`, and the intersection *proved and still absent* is empty.  Proof by
a classical family is prior art, not novelty.  None of these proofs is kernel-verified; the
receipt names the obstruction.

## DG3 — Widen the inverse-symbolic search

The 32 came from one family (continued fractions) at 1.19×10^8 ordinals.  Add
independently structured families: infinite series with polynomial term ratios,
infinite products, and bounded closed-form definite integrals.

**Completion test:** each new family enumerated at ≥10^8 ordinals with the digit-holdout
discipline enforced (found at ~10^-13, verified at 10^-120), and its classical members
rediscovered as controls (e.g. the Basel series, Wallis product).

**Discovery condition:** `NOT_IN_BUILTIN_TABLE` survivors that also clear DG1 screening.

**Status (receipts `runs/math/inverse-symbolic/families-v1.json`,
`families-screen-v1.json`, `families-proof-v1.json`): completed, discovery condition NOT
met.**  Three families were declared and enumerated exhaustively on the GPU for a total of
3,248,663,112 ordinals in 67 s: hypergeometric-type series `c0·Σ t_k` with a rational term
ratio (1.55×10^9 ordinals over 1.29×10^8 distinct series, 184M ordinals/s), infinite
products `c0·Π A(k)/B(k)` (1.55×10^9 ordinals over 1.29×10^8 distinct products, the exact
convergence test rejecting 99.6% before evaluation), and definite integrals
`c0·∫₀¹ x^a (1-x)^b K(x)^c dx` (1.49×10^8 ordinals over 1.24×10^7 quadratures).  Every
family rediscovered its declared classics — Basel, Gregory–Leibniz and `e = Σ 1/k!`; the
Wallis product in two grammar forms; the arctangent integral, `∫₀¹ -ln x/(1-x) dx = ζ(2)`
and a Beta instance — and a fabricated near-miss planted 10^-14 from `π` cleared the fp64
window and died at 60 digits, as it must.

1,626 fp64 matches all survived the 60- and 120-digit holdout, collapsing to **106 distinct
objects** once the declared grammar degeneracies are quotiented out.  Screening against a
69-seed corpus (45 independently encoded classical identities plus 24 cited parametric
theorems, expanded to 123 records with a closed provenance forest) adjudicated the 98 that
the built-in table did not already carry: 67 `KNOWN`, 15 `INCONCLUSIVE_VALUE_MATCH` and 16
`NOT_FOUND_IN_CORPUS`, with all 8 already-known controls recovered (100%).  Proof routing
then settled all 31 remaining candidates: **31
`PROVED`, 0 `REFUTED`, 0 `MISSING_TECHNIQUE`**, by six declared techniques — `hyperexpand`
on the candidate's own `pFq` parameters, the Weierstrass–Gauss Gamma product, a Hurwitz-zeta
log-power reduction whose quasi-polynomial fit is re-verified on extra coefficients, the
differentiated Beta integral, the cyclotomic substitution `1+x+…+x^{m-1} = (1-x^m)/(1-x)`,
and Euler's `2F1` integral representation.

**The count that matters is therefore zero.**  Every survivor is exhibited as an instance of
a cited classical family, so each is reclassified `KNOWN_BY_PROOF_FAMILY` rather than
counted; the intersection *absent from the corpus and not reducible by any declared
technique* is empty.  Two defects found along the way are worth recording because both would
have produced false results silently: `mpmath.quad` truncates its abscissa range and loses
seventy digits on an `x^(-11/12)` endpoint, which was killing true identities at the
verification stages until the integral evaluator was replaced by an exact Beta shortcut plus
a tanh-sinh rule whose range is solved from the requested precision; and a "terminating
series" test written against fp64 underflow rather than against a structural root of `P`
was discarding the exponential series.  None of these proofs is kernel-verified; the receipt
names the obstruction.

## DG4 — Spectral survey of every eligible sequence

The spectral instrument recovered Steinerberger's Ulam frequency to seven decimals and
found two unexplained sidebands.  It has been pointed at three sequences.  Point it at
all of them.

**Completion test:** every queue sequence with ≥64 exact terms is scanned and sealed,
with survivors and refutations both recorded.

**Discovery condition:** a surviving spectral bias on a sequence with no published
signal, holding on holdout, with the frequency reported to declared precision.

## DG5 — Six missing generators (enabler for DG4 and DG6)

`ulam_u_1_2`, `twin_prime_count_pi2_10_pow_k`, `gilbreath_leading_terms`,
`pascal_interior_multiplicity`, `recaman`, `reverse_and_add_base10`.

**Completion test:** each generator emits exact rows verified against its OEIS
reference values, and a fan-out epoch reports zero `missing_generator:*` gaps and zero
`upstream_blocked:generate_rows`.

**Discovery condition:** none directly — this is pure enablement, and it is on the list
only because it currently blocks eight problems including Ulam.

## DG6 — Holonomic survey of every eligible sequence

**Completion test:** every queue sequence run through the guesser; controls (Catalan,
Motzkin, derangements, factorial) recovered; results sealed.

**Discovery condition:** an exact annihilating operator for a sequence with no known
P-finite recurrence.

## DG7 — Gravity: local formula atlas to one no-dark-matter theory

The authoritative ordered program is
[`NO_DARK_MATTER_GRAVITY_DISCOVERY_GOALS.md`](NO_DARK_MATTER_GRAVITY_DISCOVERY_GOALS.md).
It replaces “finish the existing 23 candidates, then look at data” with a generative,
data-led ladder:

1. Freeze and benchmark the actual SPARC evaluator.
2. Search a broad formula space independently on all 139 admitted exploration galaxies,
   preserving every predictive local solution.
3. Collapse known rewrites and behaviorally equivalent formulas, then learn which measured
   baryonic features generate the per-galaxy variation.
4. Remove every object-specific gravitational parameter and open the 35-galaxy confirmation
   set once.
5. Transfer without refit to THINGS/LITTLE THINGS, the 12 X-COP clusters, and the 25 CLASH
   lensing clusters.
6. Invert the surviving local laws into shared weak-field operators and covariant actions,
   then test the same theory against Cassini, the Double Pulsar, GW170817, Planck, and DESI.

**Immediate completion test:** G0 and G1 PASS: a frozen observational/compute contract and a
139/139 predictive, radial-holdout, baryon-only formula atlas with origin/equivalence lineage
and family-level exclusion certificates.

**Discovery condition:** one zero-object-parameter, universally coupled covariant theory
forward-generates the galaxy and cluster solution classes and survives galaxy confirmation,
external galaxy replication, direct lensing, local/strong-field, gravitational-wave, and
cosmological gates without non-baryonic dark matter.

**Current measured boundary.** The full repository sample contains 175 SPARC galaxies and
3,391 points. The existing split leaves 139 admitted exploration galaxies and 35 confirmation
galaxies. Fourteen narrow law spaces produced 1,946 independent single-galaxy fits, but every
space's local parameter varies across the 139-galaxy population: 0/14 is universal. There is
no broad creative formula atlas, real-data equivalence graph, reconciliation law, cluster
result, lensing result, or direct-observation validation. Existing formal candidate work is
supporting infrastructure, not a prerequisite for attempting G0/G1 formula discovery.

---

## Sequencing

For gravity, DG7 now runs **G0 -> G1 -> G2 -> ... -> G12** in order; later empirical gates do
not open early. Across the broader system, gravity G0/G1 and the next externally sourced
creative-math test are the two primary generative tracks. DG5 remains a cheap enabler for
DG4/DG6. Old formal gravity adapters proceed only when they supply or test a candidate that
the new gravity ladder needs.

## What "we discovered something" would require

A single receipt chain showing: candidate generated without target access → survived a
holdout it never saw → screened against a corpus of declared scale and found absent →
proved by a checkable certificate → prior-art reviewed by a human before any public
claim.  Anything short of that chain is a candidate, and the site will say so.
