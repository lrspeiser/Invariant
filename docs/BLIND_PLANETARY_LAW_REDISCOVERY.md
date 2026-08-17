# Blind solar-system law rediscovery

A blinded benchmark that hands the discovery engine anonymized orbital data with zero
physics labels and asks whether it recovers three classical laws it was never told
about: Kepler's harmonic law (1619), Newton's inverse-square law (1687), and the
general-relativistic perihelion advance (1915).

* Campaign: `src/sigma_theory_compiler/blind_planetary_law_rediscovery_campaign.py`
* Public config (what Phase A reads): `configs/backgrounds/blind_planetary_public_worlds_v1.json`
* Sealed targets + provenance: `configs/backgrounds/blind_planetary_targets_v1.json`
* Receipt: `runs/math/blind-planetary-laws/campaign.json`
* Tests: `tests/test_blind_planetary_law_rediscovery_campaign.py`

Run it:

```
python -m sigma_theory_compiler.blind_planetary_law_rediscovery_campaign --root .
python -m sigma_theory_compiler.blind_planetary_law_rediscovery_campaign --root . --validate-checked
```

## What Phase A is allowed to see

Four worlds, opaque ids `world_01 .. world_04`, columns named only `x1`, `x2`, `x3`.
Every value is an exact rational (`{"numerator": …, "denominator": …}`). There are no
body names, no unit names, and no subject-matter words anywhere in the config: the
campaign tokenizes the config on letter runs and refuses to run if any member of
`FORBIDDEN_VOCABULARY` appears. That list covers the planet names, `au`, `year`,
`gravity`, `kepler`, `newton`, `einstein`, `relativity`, `orbit`, `perihelion`, `mass`,
`law`, and about eighty more. The check is a build-time guard, not only a test.

Everything that could give the answer away — the closed forms, what each column
actually is, the attributions, the citations, and the generative rule that produced the
rows — lives in the sealed fixture. The config commits to it by SHA-256 and nothing
else. Even the campaign's own name and output path are module constants rather than
config fields, so the file Phase A opens carries no hint at all.

## The blind protocol

Reused unchanged from `dozen_blind_rediscovery_campaign.py`:

1. Phase A runs with `builtins.open`, `io.open`, and `pathlib.Path.open` wrapped by a
   guard that denies every read of the targets fixture and counts the attempts.
2. One deliberate, instrumented probe tries to read the fixture inside the guard. It
   must be denied, and the receipt carries the denial certificate. If the probe ever
   succeeds, the build fails closed.
3. The Phase A root — every candidate, every stage receipt hash, every commitment, and
   the denial certificate — is hashed and frozen.
4. Only then is the fixture opened. Once. Atomically. Every commitment must open
   against the hash the config published before Phase A ran.
5. Zero candidates are generated, retried, or adjusted after the unseal. The receipt's
   chronology records `target_reads` at each step: `0, 0, 0, 0, 0, 0, 1, 1, 1`.

## The derived-view search (how world_03 was reachable at all)

Kepler's and Newton's laws relate two columns directly, so a bounded exponent search
over `(x1, x_response)` can find them. The relativistic law cannot be found that way:
the advance is not a power of the semi-major axis alone. It is a power of the *product*
`a(1 - e^2)` — the semi-latus rectum — and the engine has to construct that variable
before any law is visible.

So the campaign declares a transformation lane and logs it exhaustively. A **view** is
one member of a finite ordered grammar:

| family | template | views |
|---|---|---:|
| `power_pair` | `z = x_response^v · x1^(−u)`, `v ∈ 1..3`, `u ∈ −6..6 \ {0}`, `gcd(\|u\|, v) = 1` | 26 |
| `power_triple` | `z = x_response^v · x1^i · (1 − x2²)^j · x2^k`, `v ∈ 1..2`, `i, j, k ∈ −2..2` | 250 |

B1 (basis synthesis) **and** B2 (nonlinear coefficient search) are run on every view.
A view is *admitted* only when B1 returns its `constant` family, because a constant
derived column is the only statement in this grammar that does not depend on the row
index — and the config declares the row index an arbitrary label. Views are ordered by
total exponent degree, so the first admitted view is the Occam-minimal one, and every
strictly simpler view carries an exact rejection.

An admitted view with constant `c` is read back as a law:

* `power_pair`: `x_response = c^(1/v) · x1^(u/v)`
* `power_triple`: `x_response = (c · x1^(−i) · (1 − x2²)^(−j) · x2^(−k))^(1/v)`

The search space is 26 + 26 + 250 + 26 = 328 views, and a test pins those numbers. A
MISSED verdict therefore means "outside this declared grammar", never "impossible".

## Where the data comes from, and what it is not

**This is a rediscovery benchmark, not an observation claim.** No observational dataset
is opened at run time. The rows are computed in exact rational arithmetic from a
generative rule that is sealed with the targets, anchored to published values:

* Semi-major axes and eccentricities: JPL Solar System Dynamics, *Keplerian Elements
  for Approximate Positions of the Major Planets* (J2000, 1800–2050 table; E. M.
  Standish) — <https://ssd.jpl.nasa.gov/planets/approx_pos.html>
* Sidereal orbit periods, used **only** as fidelity anchors: NASA GSFC / NSSDCA
  *Planetary Fact Sheet* — <https://nssdc.gsfc.nasa.gov/planetary/factsheet/>
* 1 Ceres (`a = 2.7658 au`, `e = 0.0785`): JPL Small-Body Database
* `GM_sun = 1.32712440018 × 10^20 m³ s⁻²` (IAU 2009), `c = 299792458 m s⁻¹` (SI, exact),
  `au = 149597870700 m` (IAU 2012 Resolution B2, exact)

Ten bodies: the eight planets, Ceres, and Pluto.

The construction is forced by the engine's exactness discipline. B1 and B2 accept exact
rationals only — no tolerance, no rounding — and `a^(3/2)` is irrational for a generic
rational `a`. So each anchor axis is replaced by the nearest value whose square root is
a 12-decimal rational: `s = round(√a_published, 12)`, `x1 = s²`, `x2 = s³`. Then
`x2²/x1³ = 1` holds *exactly*, and `x1` differs from the published axis by at most
`9.6 × 10^-13` relative — far below the published precision. World `world_02` is
derived from those same two columns by `g = 4π² · a / T²` with `4π²` quantized to 50
decimals, so its inverse-square structure is a consequence of the world_01 rows and the
definition, not something inserted by hand. World `world_03` uses the same axes, the
published eccentricities, and the per-revolution advance
`Δϖ = 6πGM/(c²·a(1−e²))` in arcseconds, with the coefficient
`3888000·GM/(c²·au)` quantized to 50 decimals (the `6π` radians-per-revolution and
`648000/π` arcseconds-per-radian factors leave no `π`).

The honest cost of this is reported, not hidden. The constructed periods differ from
the published sidereal periods by up to `6.66 × 10^-4` relative (Uranus) — the physical
residual of the two-body point-mass idealization against real periods (planet-mass and
epoch-convention effects). That bound is stored in the sealed provenance under
`fidelity` and republished in the receipt.

After the unseal the campaign **replays the sealed rule**: it re-derives every column
from the sealed anchors in exact rational arithmetic, checks that each quantized root
is the correctly rounded 12-decimal square root (by the exact inequality
`(s − 5·10^-13)² ≤ a ≤ (s + 5·10^-13)²`), recomputes both declared constants from their
stated definitions, and requires every public row to match bit for bit. That is what
makes "the data came from the declared model and nothing was hand-tuned" a checked
statement rather than an assertion.

`world_04` duplicates `world_02` deliberately. It is reserved for the sibling
alternative-exclusion task, which asks whether competing exponents can be *excluded*
rather than whether the law can be found. Nothing new is claimed for it here.

## Proof routing and its boundary

Where the recovered law has an integer exponent structure the provers accept, the
campaign routes B5 (lemma decomposition) and B6 (quantified inequality proofs) on the
declared **Nat-typed monomial companion** `C(n) = n^d`, where `d` is the sum of the
absolute integer exponents on the predictor side, counting the `(1 − x2²)` base as
degree two. B5 emits the base case / successor identity / induction decomposition; B6
proves monotonicity and nonnegativity where B3 proposed and holdout-confirmed the
corresponding statement.

The boundary is stated in every route and in the receipt: **the emitted Lean proves the
monomial companion, not the physical law.** It is exact-locally checked in Python here
and is not kernel-verified in this receipt; `kernel_verified_lean` is `false`.

## Claims

```
rediscovery_of_classical_results ........... true
novelty_claimed ............................ false
real_observational_data_opened ............. false
data_computed_from_declared_model .......... true
machine_found_laws_unaided ................. computed at run time
kernel_verified_lean ....................... false
lean_proves_the_recovered_relation_itself .. false
```

`machine_found_laws_unaided` is true only when every world scores
`REDISCOVERED_EXACT`, the denied probe fired exactly once before the unseal, and no
candidate was generated after it. It is computed from the run, never asserted.
