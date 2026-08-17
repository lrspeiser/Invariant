# Head-to-head case study: Balmer 1885 and Bohr 1913

This is a benchmark against a real historical discovery, run twice, blinded, with the
result published next to what the humans actually had in hand.

Nothing here is new physics. Both halves rediscover settled nineteenth- and
twentieth-century results. What is being measured is the engine, not the world.

- Module: `src/sigma_theory_compiler/balmer_bohr_case_study.py`
- Tests: `tests/test_balmer_bohr_case_study.py`
- Public config (blind phase input): `configs/backgrounds/blind_indexed_value_rows_v1.json`
- Sealed fixture (opened once, after the freeze): `configs/backgrounds/blind_indexed_value_targets_v1.json`
- Receipt: `runs/math/case-studies/balmer-bohr-v1.json`
- One-time host measurement: `runs/math/case-studies/balmer-bohr-v1-runtime.json`
- Site pages: `/case-studies` and `/case-studies/balmer-bohr`

## Race 1 — the blind empirical race (what Balmer did)

### What the engine was given

Four anonymized rows, `(m, v)` with `m` running `1..4`:

| m | v |
|---|---|
| 1 | 6562.10 |
| 2 | 4860.74 |
| 3 | 4340.10 |
| 4 | 4101.20 |

Those are the four visible hydrogen wavelengths in units of 1e-10 m as Balmer had them,
from Ångström's 1868 solar-spectrum measurements and quoted in Balmer's 1885 note in
exactly these figures. Modern standard-air values for the same four lines are 6562.79,
4861.35, 4340.47 and 4101.74; the blind phase is deliberately given the nineteenth-century
figures, because reproducing Balmer's information state is the point.

The public config carries **no subject-matter vocabulary at all**. The forbidden-token
runtime guard from the blinded planetary campaign is reused and its list extended (147
tokens: the planetary list plus the spectroscopic one — `hydrogen`, `spectral`,
`wavelength`, `line`, `balmer`, `angstrom`, `atom`, `light`, and more). The check runs at
build time inside `_validate_config`, not only in the test suite, and a test greps the
committed config independently.

**The index convention is the hard one.** Balmer's index runs `3, 4, 5, 6`. The rows above
are labelled `1..4`, and the config states only that the label is arbitrary and that any
offset between the label and a meaningful ordinal has to be recovered by the search. The
engine was not handed the quantum numbers.

### The declared search space

One transformation lane, declared in the public config before the run:

```
z = v * (m + s)^i * ((m + s)^2 - c)^j
```

with `s` in `[0, 3]`, `i` in `[-4, 4]`, `j` in `[-4, 4]`, `c` in `[0, 9]`, and `c` pinned to
`0` whenever `j = 0` (the trailing factor is then identically one, so ten aliases would be
noise). That is **2916 declared views**, ordered by total weight `abs(i) + 2*abs(j)`, then
`s`, then `c`, then `i`, then `j`.

Every one of the 2916 is evaluated, B1-checked in exact rational arithmetic, and logged
with the number that decided it. The full log is in the receipt.

### Admission, and the one concession to real measurement

The planetary campaign admits a view when B1 returns its `constant` family — exact equality
across the derived column. That is the right test for computed rows and the wrong test for a
nineteenth-century table. **B1 is still run on all 2916 views here, and it refuses every one
of them, including the winner**, because measured rows are never exactly constant. That
refusal is recorded in the receipt rather than hidden.

Admission is therefore a declared relative-spread test: the derived column's maximum
deviation from its exact rational mean, divided by that mean, must not exceed the fit
tolerance declared in the config (`1e-4`). The constant reported is the exact rational mean
of the four derived values.

The tolerance is justified prospectively, from the source table rather than from the
answer: the 1868 figures are quoted to 0.01 in units of 1e-10 m (about 1.5e-6 relative at
6562), and Ångström's wavelength standard was later found to carry a systematic error of
roughly one part in 7000 to 8000. Both bounds are written into the config.

Because "you tuned the tolerance" is the obvious objection, the receipt publishes a
**tolerance robustness ladder**: the admitted set recomputed at 1e-6, 1e-5, 1e-4, 3e-4,
1e-3, 1e-2 and 1e-1. The admitted set is exactly one view — the right one — everywhere from
1e-4 to 1e-2, empty below, and grows to 18 views only at 1e-1, a thousand times looser than
declared. The verdict does not depend on the choice.

### Holdout

Three further rows, labelled `5, 6, 7`. **The public config carries only their labels; their
values are sealed** in the target fixture together with the classical formula. The engine
freezes its candidate, emits three predictions, and only then is anything opened. That is
the shape of the historical event: Balmer computed further members of the series from his
formula before they were confirmed.

The holdout values are the modern standard-air wavelengths of the next three members
(3970.07, 3889.05, 3835.38), which makes the test harder than the historical one: it crosses
a measurement scale as well as a prediction. The expected systematic offset of order 1e-4 is
stated in the fixture in advance, and the declared holdout tolerance is `3e-4`.

### Seal discipline

Identical to the planetary campaign, and reusing its guard class:

1. The public config is loaded and vocabulary-checked.
2. The sealed-fixture read guard is entered; it patches `builtins.open`, `io.open` and
   `pathlib.Path.open` and denies every read of the fixture path.
3. The stage ladder and the whole 2916-view search run inside the guard.
4. The candidate is frozen; the three holdout predictions are frozen.
5. One instrumented denied probe is fired and recorded: exactly one attempted read, exactly
   one denial, zero bytes exposed.
6. The Phase A root is sealed as a SHA-256 over everything above.
7. **One** atomic unseal. Every commitment must open: the fixture, the provenance block, the
   holdout block, and the target record.
8. The sealed rule is replayed on every row and checked against the sealed source table.
9. Comparison and scoring. Nothing is generated, retried, or tuned after the unseal.

The guard's scope is stated in its own certificate: it covers this process's owned Python
file-read surfaces. It is not an operating-system sandbox and does not claim to be.

### Verdict rule

`REDISCOVERED_EXACT` requires all three of:

- the frozen view exponents `(s, i, j, c)` equal the sealed ones exactly;
- the engine's constant, rounded in exact rational arithmetic to the precision at which the
  classical constant was published (one decimal place), equals the sealed constant exactly;
- every sealed holdout row is predicted within the declared holdout tolerance.

`PARTIAL` is an exact structure match that fails one of the other two. `MISSED` is anything
else, including no admitted view at all. All three verdicts are exercised in the tests.

## Race 2 — the derivation race (what Bohr did)

No data at all. Two declared postulates:

- **P1** — the negative charge orbits the positive charge under the Coulomb attraction
  `k*e^2/r^2`, with angular momentum restricted to `L = m_e*v*r = n*hbar`;
- **P2** — a change between two such states emits a single quantum carrying the whole
  difference, `h*nu = E(n_2) - E(n_1)`, with `nu = c/lambda`.

sympy then derives, and every step is recomputed on each run and re-derived independently
in the tests:

1. `r_n = n^2*hbar^2/(m_e*k*e^2)` — the Bohr radius at `n = 1`;
2. `E_n = -m_e*k^2*e^4/(2*hbar^2*n^2)`;
3. `1/lambda = (E(n_2) - E(n_1))/(h*c)`, which after `hbar = h/(2*pi)` collects into a single
   prefactor times `(1/n_1^2 - 1/n_2^2)`;
4. `R = 2*pi^2*m_e*e^4*k^2/(h^3*c)`.

Only then are CODATA 2018 constants substituted. The measured Rydberg constant is a
citation, never a fit.

The loop is closed symbolically: the `n_1 = 2` branch of the derived relation is
`lambda = (4/R)*M^2/(M^2 - 4)`, which is Balmer's form with `B = 4/R`. The sympy residual of
that identity is exactly zero, and `4/R` is then evaluated against Balmer's published
3645.6.

Two negative controls must fire, or the run aborts:

- setting the quantization rule to `L = n^2*hbar` gives `r_n ~ n^4` instead of `n^2`;
- giving the level formula `hbar^3` instead of `hbar^2` misses the measured constant by
  tens of orders of magnitude, because `hbar` carries dimensions and no constant can absorb
  a wrong power of it.

## What this does not show

- **The engine did not invent the grammar or the postulates.** Both are human declarations
  written before the run. What is measured is exhaustive, receipted selection inside a
  declared space. A `MISSED` would have meant "outside the declared grammar", never
  "impossible".
- **The engine was told which rows to fit and which to predict**, and told that the label
  might be offset from the meaningful ordinal. Balmer had to decide both for himself, from
  four numbers, with no assurance that any relation existed at all.
- **Wall-clock against human years is not a comparison of difficulty.** It is a comparison
  of two different activities. It is reported only because a head-to-head with an empty
  column on our side would be less honest, not more. The measurement is a one-time host
  reading kept in a separate file, outside every sealed hash, and is not a benchmark.
- **No historical working time is estimated.** Every duration in the receipt is either a
  cited interval between publications or the words "not precisely documented".
- **No observational dataset is opened at run time.** The transcribed values are published
  table entries carried in the fixture.
- **Nothing is novel, and no priority is claimed.**

## Reproducing

```
python -m sigma_theory_compiler.balmer_bohr_case_study --root .
python -m sigma_theory_compiler.balmer_bohr_case_study --root . --validate-checked
python -m pytest tests/test_balmer_bohr_case_study.py
```

The receipt is write-once: rebuilding compares bytes and refuses to overwrite a different
one. The runtime measurement file is written once and then left alone, so replays stay
byte-deterministic.

## Citations

- A. J. Ångström, *Recherches sur le spectre solaire* (Uppsala, 1868) — the four visible
  wavelengths used for the fit.
- J. J. Balmer, "Notiz über die Spectrallinien des Wasserstoffs", *Annalen der Physik und
  Chemie* **261**(5):80–87 (1885) — the formula, the constant 3645.6, and the prediction of
  further members.
- N. Bohr, "On the Constitution of Atoms and Molecules, Part I", *Philosophical Magazine*
  Series 6, **26**(151):1–25 (July 1913) — the derivation.
- CODATA 2018 recommended values for the electron mass, the electric constant and the proton
  mass; `h`, `e` and `c` are exact by the 2019 SI redefinition.
- NIST Atomic Spectra Database (A. Kramida, Yu. Ralchenko, J. Reader and the NIST ASD Team)
  — the modern standard-air wavelengths used as holdout.
