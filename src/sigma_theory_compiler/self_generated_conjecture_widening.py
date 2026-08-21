"""SELFGEN -- widening the self-generated conjecture loop past its elementary ceiling.

``self_generated_conjecture_adjudication`` (C3) closed the loop: the engine generates its
own objects, proposes statements about them, attaches the obligation that would settle each
statement, and discharges it.  Its own receipt pins ``proved_means_novel: false``, and it
pins the reason too -- every object in its pool is a **polynomial** sequence of degree at
most five (seed powers up to three, at most two applications of the partial-sum operator).
Its whole decision procedure is the finite-difference test for integer-valued rational
polynomials.  That test is complete on polynomials and says nothing about anything else, so
C3 cannot state, let alone settle, a question whose answer is not already visible in a
finite difference table.  Its mathematics is elementary *by construction*.

This module widens the universe until the elementary machinery no longer reaches it, and
then settles the wider statements anyway, exactly.

**The wider universe.**  A declared finite box of integer linear recurrences

    u(n + d) = c_1*u(n + d - 1) + ... + c_d*u(n),      u(0..d-1) given,

with order ``d``, coefficients ``c`` and initial values ``u`` ranging over declared integer
boxes.  Nobody names a target: the box is the entire input.  These sequences grow
exponentially, not polynomially, and every admitted object carries an **exact certificate
of non-elementarity**: an index at which the sixth forward difference is nonzero, which is
a proof that the object is not a polynomial of degree at most five and therefore is not,
and cannot be, a member of the C3 pool.  A control drives the point home from the other
side -- C3's own objects are fed to the widening gate and every one of them is refused.

**The wider statements.**  Four schemas, each a claim about *every* ``n >= 0``:

* ``divisibility_index_set`` -- ``{n >= 0 : m | u(n)} = {n >= 0 : n = j (mod q)}``.  This is
  a rank-of-apparition statement; on the Fibonacci object it is the classical ``3 | F(n)
  iff 4 | n``.
* ``modular_pure_period`` -- ``u(n + P) = u(n) (mod m)`` for every ``n >= 0`` with ``P``
  *minimal*.  On Fibonacci this is the Pisano period.
* ``cross_object_congruence`` -- ``u(n) = alpha*v(n) + beta*w(n) (mod m)`` for three
  distinct self-generated objects.
* ``zero_free_over_the_integers`` -- ``u(n) != 0`` for every ``n >= 0``.  This is an
  instance of the Skolem Problem, which is decidable for order at most four and **open for
  order five and above**.

**The decision procedure, and why it is complete.**  The state ``s(n) = (u(n), ...,
u(n + d - 1)) mod m`` lives in a set of size ``m**d``, and ``s(n + 1)`` is a function of
``s(n)`` alone.  Iterating from ``s(0)`` therefore reaches a repeated state after at most
``m**d`` steps; write ``mu`` for the index of the first repetition's earlier occurrence and
``lam`` for the cycle length.  Then for **every** ``n >= 0``

    u(n) = table[n]                       if n < mu + lam
    u(n) = table[mu + (n - mu) mod lam]    otherwise                    (mod m)

which is a finite, exact description of an infinite sequence.  Every one of the four
schemas reduces to a finite check against that table over a declared window whose length is
justified by a period argument.  There is no sampling anywhere: a verdict of ``PROVED``
covers all of ``n >= 0``, and the only arithmetic on the path is ``int``.

**What ``PROVED`` does not mean.**  It does not mean new.  The receipt pins
``proved_means_novel: false`` and ``prior_art_absence_establishes_novelty: false``, and it
carries an explicit prior-art triage: every proved statement is matched against a table of
published family theorems with citations, and the ones that match are reported as *known*.
The classical eventual-periodicity theorem for linear recurrences modulo ``m`` (Engstrom
1930/1931, Ward 1933) covers the first three schemas outright -- every statement they can
emit is an instance of it.  That is the expected outcome and the receipt says so.  The
statements that no family theorem covers are reported in a separate bucket together with
the record of what was searched, and the bucket carries the standing disclaimer that
absence from a search is not novelty.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import gcd
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

RESULT_SCHEMA = "invariant-self-generated-conjecture-widening-1.0"

#: The maximum polynomial degree the C3 pool can contain: ``seed_powers`` tops out at 3 and
#: ``sum_depths`` at 2, so ``deg <= 5``.  An object whose sixth forward difference is
#: nonzero somewhere is provably outside that pool.
ELEMENTARY_DEGREE_CEILING = 5

#: Sealed honesty claims.  A receipt that drops or flips any of these is invalid.
CLAIMS = {
    "targets_are_self_generated_not_caller_supplied": True,
    "objects_carry_an_exact_non_polynomial_certificate": True,
    "elementary_machinery_cannot_decide_these_statements": True,
    "decision_procedure_is_complete_over_all_nonnegative_integers": True,
    "verdict_is_exact_no_floating_point_on_certificate_path": True,
    "restating_generator_must_be_refused": True,
    "elementary_objects_must_be_refused_by_the_widening_gate": True,
    "proved_means_novel": False,
    "prior_art_absence_establishes_novelty": False,
}


class ConjectureWideningError(ValueError):
    """Raised when the widened generation/adjudication loop violates a declared invariant."""


# ---------------------------------------------------------------------------
# Component 0 -- prior art.  Declared before anything is generated, with citations.
# ---------------------------------------------------------------------------

#: Published results against which every ``PROVED`` statement is checked.  ``covers`` names
#: the statement schemas whose *every instance* is a consequence of the cited theorem.
PRIOR_ART: tuple[dict[str, Any], ...] = (
    {
        "key": "lrs_eventually_periodic_mod_m",
        "statement": (
            "every integer linear recurrence sequence is eventually periodic modulo every "
            "m >= 2, and purely periodic when the trailing coefficient is invertible mod m; "
            "the minimal period divides every period"
        ),
        "attribution": (
            "H. T. Engstrom, 'Periodicity in sequences defined by linear recurrence "
            "relations', Proc. Natl. Acad. Sci. USA 16 (1930) 663-665; H. T. Engstrom, 'On "
            "sequences defined by linear recurrence relations', Trans. Amer. Math. Soc. 33 "
            "(1931) 210-218; Morgan Ward, 'The arithmetical theory of linear recurring "
            "series', Trans. Amer. Math. Soc. 35 (1933) 600-628"
        ),
        "covers": ("divisibility_index_set", "modular_pure_period", "cross_object_congruence"),
        "applies_to": "every_object_in_the_box",
        "confidence": "family_theorem",
        "why_it_covers": (
            "each of these schemas asserts a property of the residue sequence u(n) mod m; "
            "the cited theorem makes that residue sequence a finite eventually-periodic "
            "object, so every true instance is a finite verification inside a known "
            "structure rather than a new fact about the integers"
        ),
    },
    {
        "key": "pisano_period",
        "statement": (
            "the period of the Fibonacci sequence modulo m, its multiplicativity over "
            "coprime moduli, and its behaviour on prime powers"
        ),
        "attribution": (
            "D. D. Wall, 'Fibonacci series modulo m', Amer. Math. Monthly 67 (1960) 525-532"
        ),
        "covers": ("modular_pure_period",),
        "applies_to": "fibonacci_object_only",
        "confidence": "pinned_identity",
        "why_it_covers": (
            "the Fibonacci object and its close relatives inside the declared box have their "
            "modular periods tabulated in the cited paper"
        ),
    },
    {
        "key": "lucas_rank_of_apparition",
        "statement": (
            "for a Lucas sequence U(P, Q) the set of indices n with m | U(n) is exactly the "
            "set of multiples of the rank of apparition alpha(m)"
        ),
        "attribution": (
            "E. Lucas, 'Theorie des fonctions numeriques simplement periodiques', Amer. J. "
            "Math. 1 (1878) 184-240 and 289-321; D. H. Lehmer, 'An extended theory of Lucas' "
            "functions', Ann. of Math. 31 (1930) 419-448"
        ),
        "covers": ("divisibility_index_set",),
        "applies_to": "order_two_lucas_initial_conditions_only",
        "confidence": "family_theorem",
        "why_it_covers": (
            "every order-two object in the declared box with u(0) = 0 and u(1) = 1 is a Lucas "
            "sequence U(P, Q), and the schema states exactly the rank-of-apparition property"
        ),
    },
    {
        "key": "nonnegative_coefficient_positivity_induction",
        "statement": (
            "if every coefficient of an integer linear recurrence is nonnegative, the "
            "coefficients sum to at least one, and d consecutive terms from some index N are "
            "all at least one, then every term from N onwards is at least one; the sign twist "
            "w(n) = (-1)^n u(n) satisfies the recurrence with coefficients (-1)^i c_i, so the "
            "same induction covers sign-alternating objects"
        ),
        "attribution": (
            "textbook induction on the recurrence; the sign-twist identity is the elementary "
            "observation that u(n) and (-1)^n u(n) satisfy recurrences whose coefficients "
            "differ by the factor (-1)^i"
        ),
        "covers": ("zero_free_by_positivity_induction",),
        "applies_to": "every_object_in_the_box",
        "confidence": "elementary_derivation",
        "why_it_covers": (
            "a zero-free verdict reached by this route is a one-line induction and carries no "
            "arithmetic content beyond it"
        ),
    },
    {
        "key": "skolem_problem_decidable_to_order_four",
        "statement": (
            "the Skolem Problem -- does an integer linear recurrence sequence have a zero "
            "term -- is decidable for order at most four and open for order five and above"
        ),
        "attribution": (
            "M. Mignotte, T. N. Shorey, R. Tijdeman, J. Reine Angew. Math. 349 (1984) 63-76; "
            "N. K. Vereshchagin, Mat. Zametki 38 (1985) 609-615; see also P. Bacik, "
            "'Completing the picture for the Skolem Problem on order-4 linear recurrence "
            "sequences', arXiv:2409.01221"
        ),
        "covers": (),
        "confidence": "section_reference",
        "why_it_covers": (
            "decidability of a class is not a proof of any particular instance, so this entry "
            "covers no schema; it is recorded because it fixes which orders in the declared "
            "box sit inside an open problem"
        ),
    },
    {
        "key": "skolem_local_obstruction_method",
        "statement": (
            "zero-freeness of an integer linear recurrence sequence can be certified by "
            "exhibiting a modulus m for which no term vanishes; the exponential local-global "
            "principle (Skolem's conjecture) asserts such an m always exists for a simple "
            "zero-free sequence, and is itself open"
        ),
        "attribution": (
            "Y. Bilu, F. Luca, J. Nieuwveld, J. Ouaknine, D. Purser, J. Worrell, 'Skolem "
            "Meets Schanuel', MFCS 2022, arXiv:2204.13417; J. Ouaknine et al., 'On the "
            "Skolem Problem and the Skolem Conjecture', LICS 2022, doi:10.1145/3531130.3533328; "
            "the Skolem tool, https://skolem.mpi-sws.org/"
        ),
        "covers": (),
        "confidence": "section_reference",
        "why_it_covers": (
            "this entry covers the METHOD used to prove zero-freeness, not the statements "
            "themselves; a zero-free verdict here is obtained by published means, which is "
            "why no such verdict may be called new, but the cited work does not assert the "
            "particular instances the box generates"
        ),
    },
)

#: What was searched, recorded on the receipt so the middle triage bucket is auditable.
PRIOR_ART_SEARCH_RECORD = {
    "searched": [
        (
            "runs/math/prior-art/cf-corpus-v1-manifest.json (13,637 records, 217 cited seeds) "
            "-- the corpus is continued-fraction expansions of analytic functions and "
            "quadratic surds and contains no linear-recurrence congruence records, so it "
            "cannot corroborate or refute anything in this module"
        ),
        "literature search for the Skolem Problem and its decidability frontier",
        (
            "literature search for the exponential local-global principle / Skolem "
            "conjecture and the modulus-certificate method for zero-freeness"
        ),
        "literature search for periodicity of linear recurrence sequences modulo m",
        "the PRIOR_ART table above, which is matched against every proved statement",
    ],
    "not_searched": [
        "OEIS (no offline copy available in this environment)",
        "MathSciNet / zbMATH (no access)",
        "any systematic enumeration of individual small-coefficient recurrence instances",
    ],
    "standing_disclaimer": (
        "absence from everything listed under 'searched' is not novelty; it is absence from "
        "what was searched, and nothing more"
    ),
}


# ---------------------------------------------------------------------------
# Component 1 -- the universe generates itself.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WideningConfig:
    """Declared parameter box.  Everything the pool contains follows from this record."""

    orders: tuple[int, ...] = (2, 3)
    coefficient_box: int = 2
    initial_box: int = 1
    window: int = 24
    prefix: int = 8
    moduli: tuple[int, ...] = (2, 3, 4, 5, 7, 8, 9, 11, 13, 16)
    periods: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 8, 12)
    max_orbit_states: int = 200_000
    max_period_claim: int = 400

    def as_json(self) -> dict[str, Any]:
        return {
            "object_family": "u(n+d) = c_1*u(n+d-1) + ... + c_d*u(n)",
            "orders": list(self.orders),
            "coefficient_box": f"c_i in [-{self.coefficient_box}, {self.coefficient_box}]",
            "initial_box": f"u(0..d-1) in [-{self.initial_box}, {self.initial_box}]",
            "observation_window": f"n in [0, {self.window})",
            "formation_window": f"n in [0, {self.prefix})",
            "declared_moduli": list(self.moduli),
            "declared_index_periods": list(self.periods),
            "orbit_state_cap": self.max_orbit_states,
            "max_period_claim": self.max_period_claim,
        }


def recurrence_values(coefficients: Sequence[int], initial: Sequence[int], count: int) -> list[int]:
    """The first ``count`` terms, exactly, as Python integers."""

    order = len(coefficients)
    values = list(initial[:order])
    while len(values) < count:
        nxt = 0
        for index, coefficient in enumerate(coefficients):
            nxt += coefficient * values[-1 - index]
        values.append(nxt)
    return values[:count]


def forward_difference_order(values: Sequence[int], depth: int) -> list[int]:
    """``Delta^depth`` applied to ``values``; length shrinks by ``depth``."""

    row = list(values)
    for _ in range(depth):
        row = [row[k + 1] - row[k] for k in range(len(row) - 1)]
    return row


def non_polynomial_certificate(values: Sequence[int]) -> dict[str, Any] | None:
    """An exact proof that ``values`` is not the restriction of a low-degree polynomial.

    ``Delta^(D+1) p = 0`` identically for every polynomial ``p`` of degree at most ``D``.  So
    a single nonzero entry of ``Delta^(D+1)`` inside the window is a proof that no degree-``D``
    polynomial agrees with the object on the window -- and hence that the object is not in
    the C3 pool, every member of which is a polynomial of degree at most five.

    Returns the largest ``D`` refuted this way together with a witness index and value.
    """

    best: dict[str, Any] | None = None
    for degree in range(len(values) - 1):
        row = forward_difference_order(values, degree + 1)
        nonzero = next((index for index, item in enumerate(row) if item != 0), None)
        if nonzero is None:
            break
        best = {
            "refuted_polynomial_degree_at_most": degree,
            "difference_order": degree + 1,
            "witness_index": nonzero,
            "witness_value": row[nonzero],
        }
    return best


def _boxes(
    order: int, coefficient_box: int, initial_box: int
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
    coefficient_range = range(-coefficient_box, coefficient_box + 1)
    initial_range = range(-initial_box, initial_box + 1)
    for coefficients in product(coefficient_range, repeat=order):
        for initial in product(initial_range, repeat=order):
            yield coefficients, initial


def build_pool(config: WideningConfig) -> dict[str, Any]:
    """Enumerate the declared box.  No target is named; the box is the whole input."""

    members: list[dict[str, Any]] = []
    refused_elementary: list[dict[str, Any]] = []
    for order in config.orders:
        for coefficients, initial in _boxes(order, config.coefficient_box, config.initial_box):
            if coefficients[-1] == 0:
                # trailing coefficient zero means the object is really of lower order and is
                # already generated there; dropping it keeps the box a disjoint union.
                continue
            values = recurrence_values(coefficients, initial, config.window)
            certificate = non_polynomial_certificate(values)
            object_id = (
                f"L{order}:c{'_'.join(str(item) for item in coefficients)}"
                f":u{'_'.join(str(item) for item in initial)}"
            )
            if (
                certificate is None
                or int(certificate["refuted_polynomial_degree_at_most"]) < ELEMENTARY_DEGREE_CEILING
            ):
                refused_elementary.append(
                    {
                        "object_id": object_id,
                        "reason": "not_certified_non_polynomial",
                        "refuted_polynomial_degree_at_most": (
                            None
                            if certificate is None
                            else int(certificate["refuted_polynomial_degree_at_most"])
                        ),
                    }
                )
                continue
            members.append(
                {
                    "object_id": object_id,
                    "definition": (
                        f"u(n+{order}) = "
                        + " + ".join(
                            f"({coefficient})*u(n+{order - 1 - index})"
                            for index, coefficient in enumerate(coefficients)
                        )
                        + f", u(0..{order - 1}) = {list(initial)}"
                    ),
                    "order": order,
                    "coefficients": list(coefficients),
                    "initial": list(initial),
                    "non_polynomial_certificate": certificate,
                    "values": values,
                }
            )

    seen: dict[tuple[int, ...], str] = {}
    unique: list[dict[str, Any]] = []
    duplicates = 0
    for record in sorted(members, key=lambda item: (item["order"], item["object_id"])):
        key = tuple(record["values"])
        if key in seen:
            duplicates += 1
            continue
        seen[key] = record["object_id"]
        unique.append(record)
    return {
        "config": config.as_json(),
        "enumerated": len(members) + len(refused_elementary),
        "refused_by_widening_gate": len(refused_elementary),
        "refused_sample": refused_elementary[:8],
        "duplicate_value_vectors_removed": duplicates,
        "objects": unique,
    }


# ---------------------------------------------------------------------------
# Component 2 -- the decision procedure.  Orbit exhaustion, complete over n >= 0.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResidueTable:
    """A finite exact description of ``u(n) mod m`` for every ``n >= 0``.

    ``values[n]`` is ``u(n) mod m`` for ``n < mu + lam``; beyond that the sequence repeats
    the block ``values[mu : mu + lam]``.  ``closure`` records the state identity
    ``s(mu) == s(mu + lam)`` that makes the description complete -- that single equality is
    the whole content of the completeness argument and a validator can recheck it alone.
    """

    modulus: int
    order: int
    mu: int
    lam: int
    values: tuple[int, ...]
    closure_state: tuple[int, ...]

    def at(self, index: int) -> int:
        if index < self.mu + self.lam:
            return self.values[index]
        return self.values[self.mu + (index - self.mu) % self.lam]


def residue_table(
    coefficients: Sequence[int], initial: Sequence[int], modulus: int, *, cap: int
) -> ResidueTable | None:
    """Iterate the companion map mod ``modulus`` to the first repeated state.

    Returns ``None`` only if the cap is reached before a repeat, in which case *no claim is
    made*; the cap never turns an unfinished computation into a verdict.
    """

    order = len(coefficients)
    state = tuple(value % modulus for value in initial[:order])
    positions: dict[tuple[int, ...], int] = {}
    values: list[int] = []
    index = 0
    while index < cap:
        if state in positions:
            mu = positions[state]
            return ResidueTable(
                modulus=modulus,
                order=order,
                mu=mu,
                lam=index - mu,
                values=tuple(values),
                closure_state=state,
            )
        positions[state] = index
        values.append(state[0])
        nxt = 0
        for offset, coefficient in enumerate(coefficients):
            nxt += coefficient * state[order - 1 - offset]
        state = (*state[1:], nxt % modulus)
        index += 1
    return None


def validate_residue_table(
    coefficients: Sequence[int], initial: Sequence[int], table: ResidueTable
) -> None:
    """Re-derive the table by direct iteration and recheck the single closure identity.

    This is deliberately *not* the cycle-detection code path: it takes ``mu`` and ``lam`` as
    given, replays the recurrence, and checks that the state at ``mu`` equals the state at
    ``mu + lam``.  That equality plus determinism of the companion map is the complete
    justification for extending the finite table to every ``n >= 0``.
    """

    order = len(coefficients)
    modulus = table.modulus
    if modulus < 2 or table.lam < 1 or table.mu < 0:
        raise ConjectureWideningError("residue table parameters are out of range")
    span = table.mu + table.lam
    if len(table.values) != span:
        raise ConjectureWideningError("residue table length disagrees with mu + lam")
    replay = [value % modulus for value in initial[:order]]
    while len(replay) < span + order:
        nxt = 0
        for offset, coefficient in enumerate(coefficients):
            nxt += coefficient * replay[len(replay) - 1 - offset]
        replay.append(nxt % modulus)
    if tuple(replay[:span]) != table.values:
        raise ConjectureWideningError("residue table values do not replay")
    early = tuple(replay[table.mu : table.mu + order])
    late = tuple(replay[table.mu + table.lam : table.mu + table.lam + order])
    if early != late:
        raise ConjectureWideningError("closure identity s(mu) = s(mu + lam) does not hold")
    if tuple(table.closure_state) != early:
        raise ConjectureWideningError("sealed closure state disagrees with the replay")


def table_json(table: ResidueTable) -> dict[str, Any]:
    return {
        "modulus": table.modulus,
        "order": table.order,
        "mu": table.mu,
        "lam": table.lam,
        "values": list(table.values),
        "closure_state": list(table.closure_state),
        "states_visited": table.mu + table.lam,
        "state_space_bound": table.modulus**table.order,
    }


def _lcm(left: int, right: int) -> int:
    return left * right // gcd(left, right)


# ---------------------------------------------------------------------------
# Component 3 -- statement schemas.  Declared before the data, finite, and small.
# ---------------------------------------------------------------------------

STATEMENT_KINDS = (
    "divisibility_index_set",
    "modular_pure_period",
    "cross_object_congruence",
    "zero_free_over_the_integers",
)

CROSS_COEFFICIENTS = (-2, -1, 1, 2)
CROSS_PAIR_SAMPLE = 400

ADMISSION_GATES = (
    "object_certified_non_polynomial",
    "claim_extends_beyond_formation_window",
    "predicate_within_declared_lattice",
    "sides_share_no_object_slot",
    "elementary_procedure_cannot_decide_it",
    "pool_share_at_least_two",
    "pool_separates_at_least_one",
)

#: A predicate satisfied by exactly one pool member is a description of that member rather
#: than a statement about the box, and a predicate satisfied by every member separates
#: nothing.  Both are refused.  The counts below are measured on the *observation window*,
#: which is data the emitter never saw (it forms from the shorter prefix).
MIN_POOL_SHARE = 2
MIN_POOL_SEPARATE = 1


def _predicate_holds_on_window(
    record: Mapping[str, Any], kind: str, parameters: Mapping[str, Any]
) -> bool:
    """Does this candidate's *predicate* hold for this object across the whole window?"""

    values = record["values"]
    if kind == "divisibility_index_set":
        modulus, period, offset = (
            int(parameters["m"]),
            int(parameters["q"]),
            int(parameters["j"]),
        )
        if modulus < 1 or period < 1:
            return False
        return all(
            (value % modulus == 0) == (index % period == offset)
            for index, value in enumerate(values)
        )
    if kind == "modular_pure_period":
        modulus, period = int(parameters["m"]), int(parameters["P"])
        if modulus < 1 or period < 1 or period >= len(values):
            return False
        return all(
            values[index] % modulus == values[index + period] % modulus
            for index in range(len(values) - period)
        )
    if kind == "zero_free_over_the_integers":
        return all(value != 0 for value in values)
    return False


def pool_content(
    pool: Sequence[Mapping[str, Any]],
    kind: str,
    parameters: Mapping[str, Any],
    cache: dict[tuple[Any, ...], dict[str, int]] | None = None,
) -> dict[str, int]:
    """How many pool members satisfy the predicate, and how many it separates."""

    if kind == "cross_object_congruence":
        key: tuple[Any, ...] = (
            kind,
            int(parameters.get("alpha", 0)),
            int(parameters.get("beta", 0)),
            int(parameters.get("m", 0)),
        )
    elif kind == "divisibility_index_set":
        key = (
            kind,
            int(parameters.get("m", 0)),
            int(parameters.get("q", 0)),
            int(parameters.get("j", 0)),
        )
    elif kind == "modular_pure_period":
        key = (kind, int(parameters.get("m", 0)), int(parameters.get("P", 0)))
    else:
        key = (kind,)
    if cache is not None and key in cache:
        return cache[key]

    if kind == "cross_object_congruence":
        modulus = int(parameters.get("m", 0)) or 1
        width = len(pool[0]["values"]) if pool else 0
        index: dict[tuple[int, ...], list[str]] = {}
        for record in pool:
            index.setdefault(
                tuple(value % modulus for value in record["values"][:width]), []
            ).append(record["object_id"])
        alpha, beta = int(parameters.get("alpha", 0)), int(parameters.get("beta", 0))
        by_id = {record["object_id"]: record for record in pool}
        share = 0
        separate = 0
        for left_id, right_id in _cross_pairs(list(by_id)):
            left = by_id[left_id]["values"][:width]
            right = by_id[right_id]["values"][:width]
            target = tuple(
                (alpha * a + beta * b) % modulus for a, b in zip(left, right, strict=True)
            )
            hit = [
                object_id
                for object_id in index.get(target, ())
                if object_id not in (left_id, right_id)
            ]
            if hit:
                share += len(hit)
            else:
                separate += 1
    else:
        share = sum(1 for record in pool if _predicate_holds_on_window(record, kind, parameters))
        separate = len(pool) - share
    content = {"share": share, "separate": separate}
    if cache is not None:
        cache[key] = content
    return content


def elementary_procedure_fails(record: Mapping[str, Any]) -> dict[str, Any]:
    """Show C3's admission step refusing this object, exactly.

    C3 admits an object only if the interpolating polynomial through the first ``deg + 1``
    values reproduces the object at *every* index of the window.  Here that polynomial is
    built through the first ``ELEMENTARY_DEGREE_CEILING + 1`` values and the first index at
    which it disagrees is returned.  Its existence is the reason C3's finite-difference
    decision procedure has nothing to say about any statement below.
    """

    values = list(record["values"])
    degree = ELEMENTARY_DEGREE_CEILING
    differences = forward_difference_order
    # Newton forward form: p(n) = sum_i C(n, i) * Delta^i u(0), i <= degree.
    coefficients = [differences(values, i)[0] for i in range(degree + 1)]
    for index in range(len(values)):
        total = 0
        binomial = 1
        for i, coefficient in enumerate(coefficients):
            if i:
                binomial = binomial * (index - i + 1) // i
            total += binomial * coefficient
        if total != values[index]:
            return {
                "interpolating_degree": degree,
                "first_disagreement_index": index,
                "polynomial_value": total,
                "object_value": values[index],
                "conclusion": (
                    "no polynomial of degree <= 5 agrees with this object on the window, so "
                    "the C3 closed-form gate drops it and its finite-difference decision "
                    "procedure never applies"
                ),
            }
    return {}


def emit_divisibility_statements(
    pool: Sequence[Mapping[str, Any]], config: WideningConfig
) -> list[dict[str, Any]]:
    """Propose ``{n : m | u(n)} = {n : n = j (mod q)}`` from the prefix alone."""

    emitted: list[dict[str, Any]] = []
    for record in pool:
        prefix = record["values"][: config.prefix]
        for modulus in config.moduli:
            hits = [index for index, value in enumerate(prefix) if value % modulus == 0]
            if not hits or len(hits) == len(prefix):
                continue
            for period in config.periods:
                if period == 1:
                    continue
                residues = {index % period for index in hits}
                if len(residues) != 1:
                    continue
                offset = residues.pop()
                if any(
                    index % period == offset and index not in hits for index in range(config.prefix)
                ):
                    continue
                emitted.append(
                    {
                        "kind": "divisibility_index_set",
                        "object_id": record["object_id"],
                        "parameters": {"m": modulus, "q": period, "j": offset},
                        "statement": (
                            f"{{n >= 0 : {modulus} | u(n)}} = {{n >= 0 : n = {offset} "
                            f"(mod {period})}}  [u = {record['object_id']}]"
                        ),
                        "formation_window": [0, config.prefix],
                        "claim_range": "n >= 0",
                    }
                )
                break
    return emitted


def emit_period_statements(
    pool: Sequence[Mapping[str, Any]], config: WideningConfig
) -> list[dict[str, Any]]:
    """Propose ``u(n + P) = u(n) (mod m)`` for all ``n >= 0`` with ``P`` minimal.

    The period is read off the *prefix*: the smallest ``P`` for which the observed prefix
    repeats.  A prefix that repeats by accident yields a false conjecture, which is exactly
    the risk the adjudicator exists to resolve.
    """

    emitted: list[dict[str, Any]] = []
    for record in pool:
        for modulus in config.moduli:
            prefix = [value % modulus for value in record["values"][: config.prefix]]
            for period in range(1, config.prefix):
                if all(
                    prefix[index] == prefix[index + period] for index in range(len(prefix) - period)
                ):
                    emitted.append(
                        {
                            "kind": "modular_pure_period",
                            "object_id": record["object_id"],
                            "parameters": {"m": modulus, "P": period},
                            "statement": (
                                f"for all n >= 0: u(n + {period}) = u(n) (mod {modulus}), and "
                                f"{period} is the least such period  [u = {record['object_id']}]"
                            ),
                            "formation_window": [0, config.prefix],
                            "claim_range": "n >= 0",
                        }
                    )
                    break
    return emitted


def _cross_pairs(order: Sequence[str]) -> list[tuple[str, str]]:
    size = len(order)
    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for offset in (1, 5, 11, 23, 47):
        for start in range(size):
            left = order[start]
            right = order[(start * 3 + offset) % size]
            if left == right or (left, right) in seen:
                continue
            seen.add((left, right))
            pairs.append((left, right))
            if len(pairs) >= CROSS_PAIR_SAMPLE:
                return pairs
    return pairs


def emit_cross_statements(
    pool: Sequence[Mapping[str, Any]], config: WideningConfig
) -> list[dict[str, Any]]:
    """Propose ``u(n) = alpha*v(n) + beta*w(n) (mod m)`` matched on the prefix."""

    width = config.prefix
    by_id = {record["object_id"]: record for record in pool}
    pairs = _cross_pairs([record["object_id"] for record in pool])
    emitted: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, int, int, int]] = set()
    for modulus in config.moduli:
        index: dict[tuple[int, ...], list[str]] = {}
        for record in pool:
            key = tuple(value % modulus for value in record["values"][:width])
            index.setdefault(key, []).append(record["object_id"])
        for alpha in CROSS_COEFFICIENTS:
            for beta in CROSS_COEFFICIENTS:
                for left_id, right_id in pairs:
                    left = by_id[left_id]["values"][:width]
                    right = by_id[right_id]["values"][:width]
                    target = tuple(
                        (alpha * a + beta * b) % modulus for a, b in zip(left, right, strict=True)
                    )
                    for target_id in index.get(target, ()):
                        key = (target_id, left_id, right_id, alpha, beta, modulus)
                        if target_id in (left_id, right_id) or key in seen:
                            continue
                        seen.add(key)
                        emitted.append(
                            {
                                "kind": "cross_object_congruence",
                                "object_id": target_id,
                                "parameters": {
                                    "alpha": alpha,
                                    "beta": beta,
                                    "m": modulus,
                                    "left_id": left_id,
                                    "right_id": right_id,
                                },
                                "statement": (
                                    f"for all n >= 0: {target_id}(n) = {alpha}*{left_id}(n) + "
                                    f"{beta}*{right_id}(n) (mod {modulus})"
                                ),
                                "formation_window": [0, width],
                                "claim_range": "n >= 0",
                            }
                        )
                        break
    return emitted


def emit_zero_free_statements(
    pool: Sequence[Mapping[str, Any]], config: WideningConfig
) -> list[dict[str, Any]]:
    """Propose ``u(n) != 0`` for every ``n >= 0`` when the window shows no zero."""

    emitted: list[dict[str, Any]] = []
    for record in pool:
        if any(value == 0 for value in record["values"]):
            continue
        emitted.append(
            {
                "kind": "zero_free_over_the_integers",
                "object_id": record["object_id"],
                "parameters": {"order": int(record["order"])},
                "statement": (
                    f"for all n >= 0: u(n) != 0  [u = {record['object_id']}, "
                    f"order {record['order']}]"
                ),
                "formation_window": [0, config.window],
                "claim_range": "n >= 0",
            }
        )
    return emitted


# ---------------------------------------------------------------------------
# Component 4 -- admission gates.
# ---------------------------------------------------------------------------


def admit(
    candidate: Mapping[str, Any],
    pool: Sequence[Mapping[str, Any]],
    config: WideningConfig,
    cache: dict[tuple[Any, ...], dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Run the seven gates.  ``admitted`` is the conjunction; refusals name the gate."""

    by_id = {record["object_id"]: record for record in pool}
    kind = candidate.get("kind")
    parameters = candidate.get("parameters") or {}
    record = by_id.get(str(candidate.get("object_id")))
    checks: dict[str, bool] = {}

    certificate = (record or {}).get("non_polynomial_certificate") or {}
    checks["object_certified_non_polynomial"] = (
        int(certificate.get("refuted_polynomial_degree_at_most", -1)) >= ELEMENTARY_DEGREE_CEILING
    )

    window = candidate.get("formation_window") or [0, 0]
    checks["claim_extends_beyond_formation_window"] = candidate.get("claim_range") == "n >= 0"

    if kind == "divisibility_index_set":
        modulus, period, offset = parameters.get("m"), parameters.get("q"), parameters.get("j")
        checks["predicate_within_declared_lattice"] = (
            modulus in config.moduli
            and period in config.periods
            and isinstance(period, int)
            and period > 1
            and isinstance(offset, int)
            and 0 <= offset < period
        )
        checks["sides_share_no_object_slot"] = True
    elif kind == "modular_pure_period":
        modulus, period = parameters.get("m"), parameters.get("P")
        checks["predicate_within_declared_lattice"] = (
            modulus in config.moduli
            and isinstance(period, int)
            and 1 <= period <= config.max_period_claim
        )
        checks["sides_share_no_object_slot"] = True
    elif kind == "cross_object_congruence":
        slots = (
            candidate.get("object_id"),
            parameters.get("left_id"),
            parameters.get("right_id"),
        )
        checks["predicate_within_declared_lattice"] = (
            parameters.get("alpha") in CROSS_COEFFICIENTS
            and parameters.get("beta") in CROSS_COEFFICIENTS
            and parameters.get("m") in config.moduli
        )
        checks["sides_share_no_object_slot"] = all(slots) and len(set(slots)) == 3
    elif kind == "zero_free_over_the_integers":
        checks["predicate_within_declared_lattice"] = (
            record is not None and int(window[1]) <= config.window
        )
        checks["sides_share_no_object_slot"] = True
    else:
        checks["predicate_within_declared_lattice"] = False
        checks["sides_share_no_object_slot"] = False

    checks["elementary_procedure_cannot_decide_it"] = bool(
        record is not None and elementary_procedure_fails(record)
    )

    # Content is measured whenever the predicate is *evaluable*, so an out-of-lattice
    # candidate is still refused on real counts rather than on a zero nobody computed.
    try:
        content = pool_content(pool, str(kind), parameters, cache)
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        content = {"share": 0, "separate": 0}
    checks["pool_share_at_least_two"] = content["share"] >= MIN_POOL_SHARE
    checks["pool_separates_at_least_one"] = content["separate"] >= MIN_POOL_SEPARATE

    refused = [gate for gate in ADMISSION_GATES if not checks.get(gate, False)]
    return {
        "checks": checks,
        "content": content,
        "admitted": not refused,
        "refused_gates": refused,
    }


# ---------------------------------------------------------------------------
# Component 5 -- obligations and their discharge.
# ---------------------------------------------------------------------------

ORBIT_ROUTE = "sigma_theory_compiler.self_generated_conjecture_widening.residue_table"

DECISION_PROCEDURE = (
    "the state s(n) = (u(n), ..., u(n+d-1)) mod m ranges over a set of size m**d and s(n+1) "
    "is a function of s(n); iterating from s(0) therefore repeats a state within m**d steps, "
    "giving mu and lam with s(n) = s(mu + (n - mu) mod lam) for every n >= mu; that finite "
    "table decides the claim for all n >= 0 with no sampling"
)


def _obligation(body: dict[str, Any]) -> dict[str, Any]:
    body["obligation_sha256"] = canonical_sha256(body)
    return body


def adjudicate_divisibility(
    record: Mapping[str, Any], parameters: Mapping[str, Any], config: WideningConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    modulus = int(parameters["m"])
    period = int(parameters["q"])
    offset = int(parameters["j"])
    coefficients, initial = record["coefficients"], record["initial"]
    table = residue_table(coefficients, initial, modulus, cap=config.max_orbit_states)
    obligation = _obligation(
        {
            "obligation_kind": "eventually_periodic_residue_set_equality",
            "decision_procedure": DECISION_PROCEDURE,
            "completeness": "complete_over_all_nonnegative_integers",
            "routed_to": [ORBIT_ROUTE],
            "inputs": {
                "coefficients": list(coefficients),
                "initial": list(initial),
                "m": modulus,
                "q": period,
                "j": offset,
                "orbit_state_cap": config.max_orbit_states,
            },
        }
    )
    if table is None:
        return obligation, {"verdict": "OPEN", "reason": "orbit cap reached", "route": ORBIT_ROUTE}
    validate_residue_table(coefficients, initial, table)
    horizon = table.mu + _lcm(period, table.lam)
    witness = None
    for index in range(horizon):
        divides = table.at(index) == 0
        claimed = index % period == offset
        if divides != claimed:
            witness = {
                "n": index,
                "u_mod_m": table.at(index),
                "divides": divides,
                "claimed": claimed,
            }
            break
    adjudication: dict[str, Any] = {
        "route": ORBIT_ROUTE,
        "residue_table": table_json(table),
        "checked_indices": horizon,
        "horizon_justification": (
            "both index sets are periodic in n with period lcm(q, lam) once n >= mu, so "
            "agreement on [0, mu + lcm(q, lam)) is agreement everywhere"
        ),
    }
    if witness is None:
        adjudication["verdict"] = "PROVED"
        adjudication["scope"] = "every integer n >= 0"
    else:
        adjudication["verdict"] = "REFUTED"
        adjudication["witness"] = witness
    return obligation, adjudication


def adjudicate_period(
    record: Mapping[str, Any], parameters: Mapping[str, Any], config: WideningConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    modulus = int(parameters["m"])
    claimed = int(parameters["P"])
    coefficients, initial = record["coefficients"], record["initial"]
    table = residue_table(coefficients, initial, modulus, cap=config.max_orbit_states)
    obligation = _obligation(
        {
            "obligation_kind": "minimal_pure_period_of_a_residue_sequence",
            "decision_procedure": DECISION_PROCEDURE
            + "; purity is mu = 0 and minimality is lam, because any value period is a state "
            "period and lam is the least state period",
            "completeness": "complete_over_all_nonnegative_integers",
            "routed_to": [ORBIT_ROUTE],
            "inputs": {
                "coefficients": list(coefficients),
                "initial": list(initial),
                "m": modulus,
                "claimed_period": claimed,
                "orbit_state_cap": config.max_orbit_states,
            },
        }
    )
    if table is None:
        return obligation, {"verdict": "OPEN", "reason": "orbit cap reached", "route": ORBIT_ROUTE}
    validate_residue_table(coefficients, initial, table)
    adjudication: dict[str, Any] = {
        "route": ORBIT_ROUTE,
        "residue_table": table_json(table),
        "true_preperiod": table.mu,
        "true_minimal_period": table.lam,
    }
    if table.mu == 0 and table.lam == claimed:
        adjudication["verdict"] = "PROVED"
        adjudication["scope"] = "every integer n >= 0"
    else:
        adjudication["verdict"] = "REFUTED"
        adjudication["witness"] = {
            "n": table.mu,
            "u_mod_m": table.at(table.mu),
            "u_shifted_mod_m": table.at(table.mu + claimed),
            "reason": (
                "preperiod is nonzero"
                if table.mu != 0
                else f"least period is {table.lam}, not {claimed}"
            ),
        }
        if table.mu == 0:
            index = next(
                (n for n in range(table.lam) if table.at(n) != table.at(n + claimed)),
                None,
            )
            if index is not None:
                adjudication["witness"] = {
                    "n": index,
                    "u_mod_m": table.at(index),
                    "u_shifted_mod_m": table.at(index + claimed),
                    "reason": f"least period is {table.lam}, not {claimed}",
                }
    return obligation, adjudication


def adjudicate_cross(
    target: Mapping[str, Any],
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    parameters: Mapping[str, Any],
    config: WideningConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    modulus = int(parameters["m"])
    alpha = int(parameters["alpha"])
    beta = int(parameters["beta"])
    obligation = _obligation(
        {
            "obligation_kind": "joint_orbit_residue_identity",
            "decision_procedure": DECISION_PROCEDURE
            + "; three residue sequences are jointly eventually periodic with preperiod "
            "max(mu_i) and period lcm(lam_i), so agreement on that window is agreement "
            "everywhere",
            "completeness": "complete_over_all_nonnegative_integers",
            "routed_to": [ORBIT_ROUTE],
            "inputs": {
                "target": {
                    "coefficients": list(target["coefficients"]),
                    "initial": list(target["initial"]),
                },
                "left": {
                    "coefficients": list(left["coefficients"]),
                    "initial": list(left["initial"]),
                },
                "right": {
                    "coefficients": list(right["coefficients"]),
                    "initial": list(right["initial"]),
                },
                "alpha": alpha,
                "beta": beta,
                "m": modulus,
                "orbit_state_cap": config.max_orbit_states,
            },
        }
    )
    tables = []
    for item in (target, left, right):
        table = residue_table(
            item["coefficients"], item["initial"], modulus, cap=config.max_orbit_states
        )
        if table is None:
            return obligation, {
                "verdict": "OPEN",
                "reason": "orbit cap reached",
                "route": ORBIT_ROUTE,
            }
        validate_residue_table(item["coefficients"], item["initial"], table)
        tables.append(table)
    preperiod = max(table.mu for table in tables)
    period = 1
    for table in tables:
        period = _lcm(period, table.lam)
    horizon = preperiod + period
    witness = None
    for index in range(horizon):
        combination = (alpha * tables[1].at(index) + beta * tables[2].at(index)) % modulus
        if tables[0].at(index) != combination:
            witness = {
                "n": index,
                "target_mod_m": tables[0].at(index),
                "combination_mod_m": combination,
            }
            break
    adjudication: dict[str, Any] = {
        "route": ORBIT_ROUTE,
        "joint_preperiod": preperiod,
        "joint_period": period,
        "checked_indices": horizon,
        "residue_tables": [table_json(table) for table in tables],
    }
    if witness is None:
        adjudication["verdict"] = "PROVED"
        adjudication["scope"] = "every integer n >= 0"
    else:
        adjudication["verdict"] = "REFUTED"
        adjudication["witness"] = witness
    return obligation, adjudication


POSITIVITY_ROUTE = "sigma_theory_compiler.self_generated_conjecture_widening.positivity_certificate"


def companion_matrix(coefficients: Sequence[int]) -> list[list[int]]:
    """``M`` with ``s(n + 1) = M s(n)`` for the state ``s(n) = (u(n), ..., u(n+d-1))``."""

    order = len(coefficients)
    rows = [[1 if column == row + 1 else 0 for column in range(order)] for row in range(order - 1)]
    rows.append([coefficients[order - 1 - column] for column in range(order)])
    return rows


def _matrix_multiply(
    left: Sequence[Sequence[int]], right: Sequence[Sequence[int]]
) -> list[list[int]]:
    size = len(left)
    return [
        [sum(left[row][k] * right[k][column] for k in range(size)) for column in range(size)]
        for row in range(size)
    ]


def matrix_power(matrix: Sequence[Sequence[int]], exponent: int) -> list[list[int]]:
    size = len(matrix)
    result = [[1 if row == column else 0 for column in range(size)] for row in range(size)]
    base = [list(row) for row in matrix]
    while exponent:
        if exponent & 1:
            result = _matrix_multiply(result, base)
        base = _matrix_multiply(base, base)
        exponent >>= 1
    return result


def characteristic_recurrence(matrix: Sequence[Sequence[int]]) -> list[int]:
    """Integer coefficients ``a`` with ``x(t+d) = a_1 x(t+d-1) + ... + a_d x(t)``.

    Faddeev--LeVerrier: with ``N_1 = M`` and ``b_1 = -tr(N_1)``, then ``N_k = M (N_{k-1} +
    b_{k-1} I)`` and ``b_k = -tr(N_k)/k``, the characteristic polynomial of an integer matrix
    is ``lambda^d + b_1 lambda^(d-1) + ... + b_d`` with every ``b_i`` an integer and every
    division exact.  Cayley--Hamilton then gives ``a_i = -b_i``, and since the state obeys
    ``s(n+1) = M s(n)`` every coordinate of ``s`` -- the object itself among them -- satisfies
    that recurrence.
    """

    size = len(matrix)
    current = [list(row) for row in matrix]
    coefficients: list[int] = []
    for step in range(1, size + 1):
        trace = sum(current[index][index] for index in range(size))
        value = Fraction(-trace, step)
        if value.denominator != 1:
            raise ConjectureWideningError("characteristic polynomial left the integers")
        coefficient = int(value)
        coefficients.append(coefficient)
        if step < size:
            shifted = [
                [
                    current[row][column] + (coefficient if row == column else 0)
                    for column in range(size)
                ]
                for row in range(size)
            ]
            current = _matrix_multiply(matrix, shifted)
    return [-value for value in coefficients]


POSITIVITY_ROUTE = "sigma_theory_compiler.self_generated_conjecture_widening.positivity_certificate"


def _induction_admits(twisted: Sequence[int]) -> str | None:
    """Which of the two induction hypotheses, if either, closes under this recurrence.

    ``flat`` -- every coefficient is nonnegative and they sum to at least one.  Then
    ``w(n+d) = sum_i c_i w(n+d-i) >= sum_i c_i >= 1`` whenever every ``w`` on the window is
    at least one, so ``w >= 1`` propagates on its own.

    ``monotone`` -- the hypothesis is strengthened to ``1 <= w(n) <= ... <= w(n+d-1)``.  Then
    each ``w(n+d-i)`` for ``i >= 2`` is at least ``w(n) >= 1`` and at most ``w(n+d-1)``, so

        w(n+d) >= (c_1 + sum_{i >= 2, c_i < 0} c_i) * w(n+d-1) + sum_{i >= 2, c_i > 0} c_i

    and the bracket being at least one delivers ``w(n+d) >= w(n+d-1) >= 1``, which is both
    halves of the hypothesis again.  This reaches recurrences the flat form cannot, such as
    ``x(t+2) = 3x(t+1) - x(t)``.
    """

    if all(coefficient >= 0 for coefficient in twisted) and sum(twisted) >= 1:
        return "flat"
    tail_negative = sum(value for value in twisted[1:] if value < 0)
    tail_positive = sum(value for value in twisted[1:] if value > 0)
    if twisted[0] + tail_negative >= 1 and tail_positive >= 0:
        return "monotone"
    return None


def _base_case_holds(window: Sequence[int], hypothesis: str) -> bool:
    if any(value < 1 for value in window):
        return False
    if hypothesis == "monotone":
        return all(window[index] <= window[index + 1] for index in range(len(window) - 1))
    return True


def positivity_certificate(
    coefficients: Sequence[int],
    initial: Sequence[int],
    *,
    max_start: int,
    window: int,
    decimations: Sequence[int] = (1, 2, 3, 4),
) -> dict[str, Any] | None:
    """An exact induction proving ``u(n) != 0`` from growth rather than from a modulus.

    Three ingredients, applied in that order:

    1. **Sign twist.**  With ``omega`` in ``{+1, -1}`` and a sign ``s``, ``w(n) = s *
       omega**n * u(n)`` satisfies the recurrence with coefficients ``omega**i * c_i``, so a
       sign-alternating object becomes a positive one without leaving the integers.
    2. **Decimation.**  ``s(n) = M**n s(0)`` for the companion matrix ``M``, so the
       subsequence ``v_r(t) = u(p t + r)`` satisfies the recurrence read off the
       characteristic polynomial of ``M**p`` (Cayley--Hamilton).  Splitting into ``p``
       residue classes can turn an oscillating object into ``p`` monotone ones.
    3. **Induction.**  Either the flat hypothesis ``w >= 1`` or the monotone hypothesis
       ``1 <= w(n) <= ... <= w(n+d-1)``, whichever the coefficients close under.

    Whatever is used, the conclusion is unconditional: ``u(n) != 0`` for every ``n >= 0``.
    The route is *sufficient, not complete* -- returning ``None`` proves nothing at all.
    """

    order = len(coefficients)
    span = max(window, (max_start + order + 1) * max(decimations) + max(decimations))
    values = recurrence_values(coefficients, initial, span)
    matrix = companion_matrix(coefficients)
    for period in decimations:
        if period == 1:
            twisted_source: list[tuple[int, list[int], list[int], int]] = []
            for omega in (1, -1):
                twisted = [
                    omega ** (index + 1) * coefficient
                    for index, coefficient in enumerate(coefficients)
                ]
                for sign in (1, -1):
                    twisted_source.append(
                        (
                            omega,
                            twisted,
                            [sign * omega**index * values[index] for index in range(len(values))],
                            sign,
                        )
                    )
            classes = [(0, entry) for entry in twisted_source]
        else:
            powered = matrix_power(matrix, period)
            base_coefficients = characteristic_recurrence(powered)
            classes = []
            for residue in range(period):
                decimated = [values[index] for index in range(residue, len(values), period)]
                for sign in (1, -1):
                    classes.append(
                        (
                            residue,
                            (
                                1,
                                base_coefficients,
                                [sign * value for value in decimated],
                                sign,
                            ),
                        )
                    )
        proofs: dict[int, dict[str, Any]] = {}
        for residue, (omega, twisted, series, sign) in classes:
            if residue in proofs:
                continue
            hypothesis = _induction_admits(twisted)
            if hypothesis is None:
                continue
            for start in range(max_start + 1):
                if start + order > len(series):
                    break
                if not _base_case_holds(series[start : start + order], hypothesis):
                    continue
                covered_from = start * period + residue
                if any(values[index] == 0 for index in range(min(covered_from, len(values)))):
                    break
                proofs[residue] = {
                    "residue_class": residue,
                    "omega": omega,
                    "sign": sign,
                    "hypothesis": hypothesis,
                    "twisted_coefficients": list(twisted),
                    "induction_start": start,
                    "base_case": list(series[start : start + order]),
                    "covers_indices_from": covered_from,
                }
                break
        if len(proofs) == period:
            covered_from = max(int(proof["covers_indices_from"]) for proof in proofs.values())
            if any(values[index] == 0 for index in range(min(covered_from, len(values)))):
                continue
            return {
                "decimation": period,
                "class_proofs": [proofs[residue] for residue in sorted(proofs)],
                "checked_below_start": list(values[:covered_from]),
                "argument": (
                    f"splitting n into the {period} residue classes mod {period} gives {period} "
                    "integer linear recurrences, each closed under a positivity induction; every "
                    f"index from {covered_from} on is covered by one of them, and u(n) != 0 was "
                    f"checked directly for n < {covered_from}"
                ),
            }
    return None


def adjudicate_zero_free(
    record: Mapping[str, Any], config: WideningConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Settle ``u(n) != 0`` for all ``n >= 0`` by a local obstruction, or refute it."""

    coefficients, initial = record["coefficients"], record["initial"]
    obligation = _obligation(
        {
            "obligation_kind": "zero_freeness_by_two_independent_routes",
            "decision_procedure": (
                "route A (positivity induction): twist by w(n) = s*omega^n*u(n); if the "
                "twisted coefficients are all nonnegative with sum at least one and w is at "
                "least one on d consecutive indices, induction gives w(n) >= 1 forever.  "
                "route B (local obstruction): exhibit a modulus m whose complete orbit table "
                "contains no zero residue; since u(n) = 0 implies u(n) = 0 (mod m), a "
                "zero-free residue table proves u(n) != 0 for every n >= 0.  Refutation is an "
                "explicit index with u(n) = 0"
            ),
            "completeness": (
                "sound and complete for refutation inside the scanned window; a PROVED "
                "verdict is unconditional, but failure to find an obstruction inside the "
                "declared modulus set decides nothing and is reported OPEN"
            ),
            "method_is_published": "skolem_local_obstruction_method",
            "routed_to": [POSITIVITY_ROUTE, ORBIT_ROUTE],
            "inputs": {
                "coefficients": list(coefficients),
                "initial": list(initial),
                "declared_moduli": list(config.moduli),
                "zero_scan_window": config.window,
                "orbit_state_cap": config.max_orbit_states,
            },
        }
    )
    values = record["values"]
    zero_index = next((index for index, value in enumerate(values) if value == 0), None)
    if zero_index is not None:
        return obligation, {
            "verdict": "REFUTED",
            "route": ORBIT_ROUTE,
            "witness": {"n": zero_index, "u_of_n": 0},
        }
    growth = positivity_certificate(
        coefficients, initial, max_start=config.window // 2, window=config.window
    )
    if growth is not None:
        return obligation, {
            "verdict": "PROVED",
            "route": POSITIVITY_ROUTE,
            "proof_route": "positivity_induction",
            "scope": "every integer n >= 0",
            "positivity_certificate": growth,
        }
    tried: list[int] = []
    for modulus in config.moduli:
        table = residue_table(coefficients, initial, modulus, cap=config.max_orbit_states)
        if table is None:
            continue
        tried.append(modulus)
        if 0 not in table.values:
            validate_residue_table(coefficients, initial, table)
            return obligation, {
                "verdict": "PROVED",
                "route": ORBIT_ROUTE,
                "proof_route": "local_obstruction",
                "scope": "every integer n >= 0",
                "certifying_modulus": modulus,
                "residue_table": table_json(table),
                "moduli_tried": tried,
                "argument": (
                    f"u(n) = 0 would force u(n) = 0 (mod {modulus}); the complete orbit table "
                    f"mod {modulus} has preperiod {table.mu} and period {table.lam} and "
                    "contains no zero residue, so no such n exists"
                ),
            }
    return obligation, {
        "verdict": "OPEN",
        "route": ORBIT_ROUTE,
        "reason": (
            "no zero in the observation window, no positivity induction, and no local "
            "obstruction inside the declared modulus set"
        ),
        "moduli_tried": tried,
        "zero_scan_window": config.window,
    }


# ---------------------------------------------------------------------------
# Component 6 -- prior-art triage.
# ---------------------------------------------------------------------------


def _entry_applies(entry: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    """Does this prior-art entry actually reach *this* object, or only its schema?

    A table that claimed every entry reached every object would inflate the known bucket and
    hide statements in it, so the two object-specific entries carry an ``applies_to`` scope
    and are checked against the object before they are allowed to cover anything.
    """

    scope = entry.get("applies_to", "every_object_in_the_box")
    if scope == "every_object_in_the_box":
        return True
    coefficients = list(source.get("coefficients") or ())
    initial = list(source.get("initial") or ())
    if scope == "fibonacci_object_only":
        return coefficients == [1, 1] and initial in ([0, 1], [1, 1])
    if scope == "order_two_lucas_initial_conditions_only":
        return len(coefficients) == 2 and initial[:1] == [0]
    return False


def triage(record: Mapping[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    """Match a proved statement against the declared family theorems.

    The bucket is decided by whether a *published theorem asserts the statement*, never by
    whether the statement felt interesting.  Three of the four schemas are covered outright
    by the classical eventual-periodicity theorem, so they land in ``proved_and_known`` by
    construction.  A zero-freeness verdict reached by the positivity induction is covered by
    the elementary-induction entry.  A zero-freeness verdict reached by the local obstruction
    is not asserted by any entry in the table -- the *method* is published but the instance is
    not -- so it lands in the middle bucket carrying the full search record and an explicit
    refusal to call it new.
    """

    kind = str(record.get("kind"))
    adjudication = dict(record.get("adjudication") or {})
    lookup = {entry["key"]: entry for entry in PRIOR_ART}
    covering = [
        entry for entry in PRIOR_ART if kind in entry["covers"] and _entry_applies(entry, source)
    ]
    if kind == "zero_free_over_the_integers":
        if adjudication.get("proof_route") == "positivity_induction":
            covering = [lookup["nonnegative_coefficient_positivity_induction"]]
        else:
            covering = []
    if covering:
        return {
            "bucket": "proved_and_known",
            "covered_by": [entry["key"] for entry in covering],
            "attribution": [entry["attribution"] for entry in covering],
        }
    obligation = dict(record.get("obligation") or {})
    method_key = obligation.get("method_is_published")
    method_entries = [entry for entry in PRIOR_ART if entry["key"] == method_key]
    order = int(
        (record.get("parameters") or {}).get("order", len(source.get("coefficients") or ()))
    )
    return {
        "bucket": "proved_and_prior_art_not_found",
        "covered_by": [],
        "method_is_published_as": [entry["key"] for entry in method_entries],
        "method_attribution": [entry["attribution"] for entry in method_entries],
        "decidability_context": [
            entry["attribution"]
            for entry in PRIOR_ART
            if entry["key"] == "skolem_problem_decidable_to_order_four"
        ],
        "order_of_the_object": order,
        "significance": (
            "routine: the Skolem Problem is decidable at this order, so an algorithm to "
            "settle this instance has existed since the 1980s"
            if order <= 4
            else "the Skolem Problem is open at this order in general, though this particular "
            "instance was settled by a published local-obstruction argument"
        ),
        "search_record": PRIOR_ART_SEARCH_RECORD,
        "novelty_claim": (
            "none.  no family theorem in the declared table asserts this particular "
            "statement, and it was not located in what was searched.  that is absence from a "
            "search and nothing more.  the method used to prove it is published, the object "
            "is of an order for which the underlying decision problem is already solved, and "
            "nothing here should be read as new mathematics"
        ),
    }


# ---------------------------------------------------------------------------
# Component 7 -- controls.
# ---------------------------------------------------------------------------


def elementary_objects_control(config: WideningConfig) -> dict[str, Any]:
    """C3's own objects, fed to the widening gate.  Every one must be refused."""

    from .self_generated_conjecture_adjudication import PoolConfig
    from .self_generated_conjecture_adjudication import build_pool as build_c3_pool

    c3 = build_c3_pool(PoolConfig(window=config.window))
    admitted: list[str] = []
    certified: list[dict[str, Any]] = []
    for member in c3["objects"]:
        certificate = non_polynomial_certificate(member["values"])
        degree = (
            -1 if certificate is None else int(certificate["refuted_polynomial_degree_at_most"])
        )
        if degree >= ELEMENTARY_DEGREE_CEILING:
            admitted.append(member["object_id"])
        certified.append({"object_id": member["object_id"], "refuted_degree": degree})
    return {
        "control": "elementary_objects_must_be_refused_by_the_widening_gate",
        "c3_objects_tested": len(c3["objects"]),
        "admitted_by_widening_gate": len(admitted),
        "max_refuted_degree_seen": max((row["refuted_degree"] for row in certified), default=-1),
        "passed": not admitted,
    }


def restating_generator(
    pool: Sequence[Mapping[str, Any]], config: WideningConfig
) -> list[dict[str, Any]]:
    """The degeneracy this module must refuse: a generator that echoes its input."""

    candidates: list[dict[str, Any]] = []
    for record in pool[:8]:
        prefix = list(record["values"][: config.prefix])
        candidates.append(
            {
                "kind": "observed_value_table",
                "disguise": "value_table",
                "object_id": record["object_id"],
                "parameters": {"values": prefix},
                "statement": f"u(n) = {prefix} for n in [0, {config.prefix})",
                "formation_window": [0, config.prefix],
                "claim_range": f"0 <= n < {config.prefix}",
            }
        )
        candidates.append(
            {
                "kind": "cross_object_congruence",
                "disguise": "self_relation",
                "object_id": record["object_id"],
                "parameters": {
                    "alpha": 1,
                    "beta": 2,
                    "m": config.moduli[0],
                    "left_id": record["object_id"],
                    "right_id": pool[0]["object_id"],
                },
                "statement": "u(n) = 1*u(n) + ... with the object on both sides",
                "formation_window": [0, config.prefix],
                "claim_range": "n >= 0",
            }
        )
        candidates.append(
            {
                "kind": "modular_pure_period",
                "disguise": "period_outside_the_lattice",
                "object_id": record["object_id"],
                "parameters": {"m": config.moduli[0], "P": config.max_period_claim + 1},
                "statement": "a period claim larger than the declared ceiling",
                "formation_window": [0, config.prefix],
                "claim_range": "n >= 0",
            }
        )
        candidates.append(
            {
                "kind": "divisibility_index_set",
                "disguise": "tautology",
                "object_id": record["object_id"],
                "parameters": {"m": 1, "q": 1, "j": 0},
                "statement": "for all n >= 0: 1 | u(n)",
                "formation_window": [0, config.prefix],
                "claim_range": "n >= 0",
            }
        )
    return candidates


def restatement_control(
    pool: Sequence[Mapping[str, Any]],
    config: WideningConfig,
    cache: dict[tuple[Any, ...], dict[str, int]] | None = None,
) -> dict[str, Any]:
    candidates = restating_generator(pool, config)
    rows: list[dict[str, Any]] = []
    by_disguise: dict[str, set[str]] = {}
    for candidate in candidates:
        verdict = admit(candidate, pool, config, cache)
        rows.append(
            {
                "disguise": candidate["disguise"],
                "object_id": candidate["object_id"],
                "admitted": verdict["admitted"],
                "refused_gates": verdict["refused_gates"],
            }
        )
        by_disguise.setdefault(candidate["disguise"], set()).update(verdict["refused_gates"])
    admitted = [row for row in rows if row["admitted"]]
    return {
        "control": "restating_generator_must_be_refused",
        "candidates": len(rows),
        "admitted": len(admitted),
        "passed": not admitted,
        "refusal_gates_by_disguise": {
            key: sorted(value) for key, value in sorted(by_disguise.items())
        },
    }


def planted_fibonacci_control(config: WideningConfig) -> dict[str, Any]:
    """``3 | F(n) iff 4 | n`` and Pisano period 8 mod 3 must both come back PROVED."""

    record = {
        "object_id": "planted:fibonacci",
        "order": 2,
        "coefficients": [1, 1],
        "initial": [0, 1],
        "values": recurrence_values([1, 1], [0, 1], config.window),
    }
    _, divisibility = adjudicate_divisibility(record, {"m": 3, "q": 4, "j": 0}, config)
    _, period = adjudicate_period(record, {"m": 3, "P": 8}, config)
    _, wrong = adjudicate_divisibility(record, {"m": 3, "q": 3, "j": 0}, config)
    return {
        "control": "planted_classical_fibonacci_facts_must_be_proved",
        "three_divides_F_n_iff_four_divides_n": divisibility.get("verdict"),
        "pisano_period_mod_3_is_8": period.get("verdict"),
        "deliberately_wrong_divisibility_claim": wrong.get("verdict"),
        "wrong_claim_witness": wrong.get("witness"),
        "reference": "D. D. Wall, Amer. Math. Monthly 67 (1960) 525-532",
        "passed": (
            divisibility.get("verdict") == "PROVED"
            and period.get("verdict") == "PROVED"
            and wrong.get("verdict") == "REFUTED"
            and wrong.get("witness") is not None
        ),
    }


def planted_zero_control(config: WideningConfig) -> dict[str, Any]:
    """Four legs, one per outcome the zero-freeness adjudicator can reach.

    * a sequence with an explicit zero must be ``REFUTED`` with the index;
    * a sequence the positivity induction settles must be ``PROVED`` by route A;
    * a sequence route A cannot touch but a modulus can must be ``PROVED`` by route B;
    * a sequence neither route reaches must be ``OPEN``.

    The fourth leg is the one that stops the module from overclaiming.  ``u(n+4) = -u(n+3) -
    u(n+2) - u(n+1) + u(n)`` from ``(-1, 1, -1, 1)`` runs ``-1, 1, -1, 1, -2, 3, -3, 3, -5,
    8, ...``; it has no zero anywhere the search reached, no decimation of it closes under a
    positivity induction, and every declared modulus admits a zero residue.  ``OPEN`` is the
    only verdict its certificate supports, and a procedure that answered ``PROVED`` here
    would be reporting a conclusion it did not reach.
    """

    def planted(name: str, coefficients: list[int], initial: list[int]) -> dict[str, Any]:
        return {
            "object_id": f"planted:{name}",
            "order": len(coefficients),
            "coefficients": coefficients,
            "initial": initial,
            "values": recurrence_values(coefficients, initial, config.window),
        }

    _, refuted = adjudicate_zero_free(planted("has_zero", [0, 1], [1, 0]), config)
    _, grown = adjudicate_zero_free(planted("fibonacci_shift", [1, 1], [1, 1]), config)
    _, obstructed = adjudicate_zero_free(planted("alternating_gap", [-2, -2], [-1, -1]), config)
    _, open_case = adjudicate_zero_free(
        planted("no_route", [-1, -1, -1, 1], [-1, 1, -1, 1]), config
    )
    return {
        "control": "planted_zero_free_verdicts_must_split_four_ways",
        "sequence_with_a_zero": refuted.get("verdict"),
        "witness": refuted.get("witness"),
        "settled_by_positivity_induction": grown.get("verdict"),
        "positivity_route": grown.get("proof_route"),
        "settled_by_local_obstruction": obstructed.get("verdict"),
        "obstruction_route": obstructed.get("proof_route"),
        "certifying_modulus": obstructed.get("certifying_modulus"),
        "neither_route": open_case.get("verdict"),
        "open_case_note": (
            "u = -1, 1, -1, 1, -2, 3, -3, 3, -5, 8, ... has no zero the search reached, but no "
            "decimation of it closes under a positivity induction and every declared modulus "
            "admits a zero residue, so the honest verdict is OPEN"
        ),
        "passed": (
            refuted.get("verdict") == "REFUTED"
            and refuted.get("witness") is not None
            and grown.get("verdict") == "PROVED"
            and grown.get("proof_route") == "positivity_induction"
            and obstructed.get("verdict") == "PROVED"
            and obstructed.get("proof_route") == "local_obstruction"
            and open_case.get("verdict") == "OPEN"
        ),
    }


def _contains_float(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, float):
        return True
    if isinstance(value, Mapping):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_float(item) for item in value)
    return False


def _stride_sample(items: Sequence[Any], limit: int) -> list[Any]:
    if limit <= 0 or len(items) <= limit:
        return list(items)
    picked = sorted({index * len(items) // limit for index in range(limit)})
    return [items[index] for index in picked]


# ---------------------------------------------------------------------------
# Component 8 -- the Skolem census.  An exhaustion, with the count that proves it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CensusConfig:
    """A declared box swept completely.  The counts must sum to the box size."""

    order: int = 4
    coefficient_box: int = 1
    initial_box: int = 1
    zero_scan: int = 80
    moduli: tuple[int, ...] = (2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 25, 27, 32)
    max_orbit_states: int = 60_000

    def box_size(self) -> int:
        return (2 * self.coefficient_box + 1) ** self.order * (
            2 * self.initial_box + 1
        ) ** self.order

    def as_json(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "coefficient_box": f"c_i in [-{self.coefficient_box}, {self.coefficient_box}]",
            "initial_box": f"u(0..d-1) in [-{self.initial_box}, {self.initial_box}]",
            "declared_box_size": self.box_size(),
            "zero_scan_indices": f"n in [0, {self.zero_scan})",
            "declared_moduli": list(self.moduli),
            "orbit_state_cap": self.max_orbit_states,
        }


def census(config: CensusConfig) -> dict[str, Any]:
    """Sweep the declared box completely and classify every member exactly.

    Three outcomes, and every member of the box lands in exactly one:

    * ``has_zero`` -- an explicit index ``n`` with ``u(n) = 0`` was exhibited.
    * ``zero_free`` -- either a positivity induction or a modulus whose complete orbit table
      omits zero was exhibited; both prove ``u(n) != 0`` for every ``n >= 0``.
    * ``unsettled`` -- neither; nothing is claimed about these.

    The three counts are asserted to sum to the declared box size, which is what makes the
    sweep an exhaustion rather than a sample.
    """

    has_zero = 0
    zero_free = 0
    unsettled: list[dict[str, Any]] = []
    visited = 0
    by_modulus: dict[int, int] = {}
    by_route: dict[str, int] = {}
    hardest: list[dict[str, Any]] = []
    for coefficients, initial in _boxes(config.order, config.coefficient_box, config.initial_box):
        visited += 1
        values = recurrence_values(coefficients, initial, config.zero_scan)
        if any(value == 0 for value in values):
            has_zero += 1
            continue
        growth = positivity_certificate(
            coefficients, initial, max_start=config.order + 4, window=config.zero_scan
        )
        if growth is not None:
            zero_free += 1
            by_route["positivity_induction"] = by_route.get("positivity_induction", 0) + 1
            continue
        certifier = None
        for modulus in config.moduli:
            table = residue_table(coefficients, initial, modulus, cap=config.max_orbit_states)
            if table is not None and 0 not in table.values:
                certifier = modulus
                break
        if certifier is None:
            unsettled.append({"coefficients": list(coefficients), "initial": list(initial)})
            continue
        zero_free += 1
        by_route["local_obstruction"] = by_route.get("local_obstruction", 0) + 1
        by_modulus[certifier] = by_modulus.get(certifier, 0) + 1
        if certifier >= 7:
            hardest.append(
                {
                    "coefficients": list(coefficients),
                    "initial": list(initial),
                    "certifying_modulus": certifier,
                }
            )
    total = has_zero + zero_free + len(unsettled)
    triaged = {
        "refuted": {
            "count": has_zero,
            "note": "an explicit index n with u(n) = 0 was exhibited for each of these",
        },
        "proved_and_known": {
            "count": by_route.get("positivity_induction", 0),
            "covered_by": ["nonnegative_coefficient_positivity_induction"],
            "note": (
                "zero-freeness follows from a one-line induction on the recurrence, applied "
                "after a sign twist and a decimation; nothing here is more than bookkeeping"
            ),
        },
        "proved_and_prior_art_not_found": {
            "count": by_route.get("local_obstruction", 0),
            "method_is_published_as": ["skolem_local_obstruction_method"],
            "note": (
                "each of these is proved by exhibiting a modulus whose complete orbit omits "
                "zero.  the method is published and the instances were not located in what "
                "was searched, which is absence from a search and not novelty"
            ),
            "search_record": PRIOR_ART_SEARCH_RECORD,
        },
        "open": {
            "count": len(unsettled),
            "note": (
                "neither route reached these.  that is a statement about the two routes, not "
                "about the sequences: nothing is claimed for or against a zero"
            ),
        },
    }
    return {
        "config": config.as_json(),
        "triage": triaged,
        "visited": visited,
        "counts": {
            "has_zero_with_explicit_witness": has_zero,
            "zero_free_with_modulus_certificate": zero_free,
            "unsettled": len(unsettled),
            "total": total,
        },
        "zero_free_proof_route_histogram": {key: by_route[key] for key in sorted(by_route)},
        "certifying_modulus_histogram": {str(key): by_modulus[key] for key in sorted(by_modulus)},
        "hardest_certificates_sample": sorted(
            hardest, key=lambda row: -int(row["certifying_modulus"])
        )[:12],
        "unsettled_sample": unsettled[:24],
        "partition_is_exact": total == config.box_size() == visited,
        "counting_argument": (
            f"the declared box has ({2 * config.coefficient_box + 1})^{config.order} coefficient "
            f"tuples times ({2 * config.initial_box + 1})^{config.order} initial tuples = "
            f"{config.box_size()} members; the sweep visited {visited} of them and assigned "
            "each to exactly one of three disjoint outcomes, so nothing was skipped"
        ),
    }


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


def run_loop(
    config: WideningConfig | None = None,
    *,
    max_per_kind: int = 60,
    census_config: CensusConfig | Sequence[CensusConfig] | None = None,
) -> dict[str, Any]:
    """Generate the wider universe, propose statements, attach obligations, adjudicate."""

    config = config or WideningConfig()
    pool = build_pool(config)
    objects = pool["objects"]
    if not objects:
        raise ConjectureWideningError("the declared box produced no non-elementary object")
    by_id = {record["object_id"]: record for record in objects}
    content_cache: dict[tuple[Any, ...], dict[str, int]] = {}

    controls = [
        elementary_objects_control(config),
        restatement_control(objects, config, content_cache),
        planted_fibonacci_control(config),
        planted_zero_control(config),
    ]

    emitted_by_kind = {
        "divisibility_index_set": _stride_sample(
            emit_divisibility_statements(objects, config), max_per_kind
        ),
        "modular_pure_period": _stride_sample(
            emit_period_statements(objects, config), max_per_kind
        ),
        "cross_object_congruence": _stride_sample(
            emit_cross_statements(objects, config), max_per_kind
        ),
        "zero_free_over_the_integers": _stride_sample(
            emit_zero_free_statements(objects, config), max_per_kind
        ),
    }
    candidates = [item for kind in STATEMENT_KINDS for item in emitted_by_kind[kind]]

    conjectures: list[dict[str, Any]] = []
    refused: list[dict[str, Any]] = []
    for candidate in candidates:
        gate = admit(candidate, objects, config, content_cache)
        record = dict(candidate)
        record["admission"] = gate
        if not gate["admitted"]:
            refused.append(
                {
                    "kind": candidate["kind"],
                    "object_id": candidate["object_id"],
                    "statement": candidate["statement"],
                    "refused_gates": gate["refused_gates"],
                }
            )
            continue
        source = by_id[candidate["object_id"]]
        parameters = candidate["parameters"]
        if candidate["kind"] == "divisibility_index_set":
            obligation, adjudication = adjudicate_divisibility(source, parameters, config)
        elif candidate["kind"] == "modular_pure_period":
            obligation, adjudication = adjudicate_period(source, parameters, config)
        elif candidate["kind"] == "cross_object_congruence":
            obligation, adjudication = adjudicate_cross(
                source,
                by_id[parameters["left_id"]],
                by_id[parameters["right_id"]],
                parameters,
                config,
            )
        else:
            obligation, adjudication = adjudicate_zero_free(source, config)
        record["obligation"] = obligation
        record["adjudication"] = adjudication
        record["elementary_procedure_refusal"] = elementary_procedure_fails(source)
        if adjudication["verdict"] == "PROVED":
            record["prior_art"] = triage(record, source)
        conjectures.append(record)

    verdicts: dict[str, int] = {}
    by_kind: dict[str, dict[str, int]] = {}
    buckets: dict[str, list[dict[str, Any]]] = {
        "proved_and_known": [],
        "proved_and_prior_art_not_found": [],
        "refuted": [],
        "open": [],
    }
    for record in conjectures:
        verdict = record["adjudication"]["verdict"]
        verdicts[verdict] = verdicts.get(verdict, 0) + 1
        by_kind.setdefault(record["kind"], {})[verdict] = (
            by_kind.setdefault(record["kind"], {}).get(verdict, 0) + 1
        )
        summary = {
            "kind": record["kind"],
            "object_id": record["object_id"],
            "statement": record["statement"],
        }
        if verdict == "PROVED":
            summary["prior_art"] = record["prior_art"]
            buckets[str(record["prior_art"]["bucket"])].append(summary)
        elif verdict == "REFUTED":
            summary["witness"] = record["adjudication"].get("witness")
            buckets["refuted"].append(summary)
        else:
            summary["reason"] = record["adjudication"].get("reason")
            buckets["open"].append(summary)

    controls.append(
        {
            "control": "gates_bite_on_the_honest_generator_too",
            "admitted": len(conjectures),
            "refused": len(refused),
            "passed": bool(conjectures) and bool(refused),
        }
    )
    controls.append(
        {
            "control": "adjudicator_is_not_a_constant_function",
            "proved": verdicts.get("PROVED", 0),
            "refuted": verdicts.get("REFUTED", 0),
            "passed": verdicts.get("PROVED", 0) > 0 and verdicts.get("REFUTED", 0) > 0,
        }
    )
    controls.append(
        {
            "control": "every_admitted_object_is_certified_non_polynomial",
            "objects": len(objects),
            "min_refuted_degree": min(
                int(record["non_polynomial_certificate"]["refuted_polynomial_degree_at_most"])
                for record in objects
            ),
            "elementary_degree_ceiling": ELEMENTARY_DEGREE_CEILING,
            "passed": all(
                int(record["non_polynomial_certificate"]["refuted_polynomial_degree_at_most"])
                >= ELEMENTARY_DEGREE_CEILING
                for record in objects
            ),
        }
    )
    failed = [entry for entry in controls if not entry["passed"]]

    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "claims": dict(CLAIMS),
        "prior_art": list(PRIOR_ART),
        "prior_art_search_record": dict(PRIOR_ART_SEARCH_RECORD),
        "pool": {
            "config": pool["config"],
            "enumerated": pool["enumerated"],
            "refused_by_widening_gate": pool["refused_by_widening_gate"],
            "refused_sample": pool["refused_sample"],
            "duplicate_value_vectors_removed": pool["duplicate_value_vectors_removed"],
            "objects_admitted": len(objects),
            "sample": [
                {
                    "object_id": record["object_id"],
                    "definition": record["definition"],
                    "first_values": record["values"][:8],
                    "non_polynomial_certificate": record["non_polynomial_certificate"],
                }
                for record in objects[:6]
            ],
        },
        "statement_kinds": list(STATEMENT_KINDS),
        "admission_gates": list(ADMISSION_GATES),
        "decision_procedure": DECISION_PROCEDURE,
        "controls": controls,
        "controls_passed": not failed,
        "counts": {
            "candidates_emitted": len(candidates),
            "candidates_emitted_by_kind": {
                kind: len(items) for kind, items in sorted(emitted_by_kind.items())
            },
            "refused_by_admission_gates": len(refused),
            "adjudicated": len(conjectures),
            "verdicts": verdicts,
            "verdicts_by_kind": by_kind,
            "triage": {key: len(value) for key, value in sorted(buckets.items())},
        },
        "triage": buckets,
        "conjectures": conjectures,
        "refused_sample": refused[:12],
        "limitations": [
            "PROVED is a statement about a self-generated object, never a novelty claim",
            (
                "the classical eventual-periodicity theorem for linear recurrences mod m "
                "covers every instance of three of the four schemas, so those verdicts are "
                "known by construction and are reported as such"
            ),
            (
                "zero-freeness verdicts are obtained by the published local-obstruction "
                "method; only the particular instances were not located, and instance "
                "absence from a search is not novelty"
            ),
            "the statement schema lattice is declared and finite; absence means absent here",
            (
                "an OPEN zero-freeness verdict means the declared modulus set contained no "
                "obstruction, not that the sequence has a zero"
            ),
        ],
    }
    if isinstance(census_config, CensusConfig):
        result["skolem_census"] = census(census_config)
    elif census_config:
        result["skolem_census_ladder"] = [census(item) for item in census_config]
    result["content_sha256"] = canonical_sha256(result)
    return result


def validate_receipt(value: Mapping[str, Any]) -> None:
    """Independently re-check a receipt: seal, claims, controls, exactness, certificates."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise ConjectureWideningError("unexpected schema version")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if _contains_float(body):
        raise ConjectureWideningError("receipt carries a floating-point value")
    if value.get("content_sha256") != canonical_sha256(body):
        raise ConjectureWideningError("receipt content hash changed")
    if dict(value.get("claims") or {}) != CLAIMS:
        raise ConjectureWideningError("sealed claims were altered")
    if not value.get("controls_passed"):
        raise ConjectureWideningError("a control failed; the run is void")
    for entry in value.get("controls") or ():
        if not entry.get("passed"):
            raise ConjectureWideningError(f"control did not pass: {entry.get('control')}")
    counts = value.get("counts") or {}
    conjectures = list(value.get("conjectures") or ())
    if counts.get("adjudicated") != len(conjectures):
        raise ConjectureWideningError("adjudicated count disagrees with the conjecture list")
    if not conjectures:
        raise ConjectureWideningError("no conjecture reached adjudication")
    if not counts.get("refused_by_admission_gates"):
        raise ConjectureWideningError("the admission gates refused nothing; they are vacuous")
    declared = dict(counts.get("verdicts") or {})
    if not declared.get("PROVED") or not declared.get("REFUTED"):
        raise ConjectureWideningError("the adjudicator returned one verdict for everything")

    tally: dict[str, int] = {}
    for record in conjectures:
        if not (record.get("admission") or {}).get("admitted"):
            raise ConjectureWideningError("an unadmitted candidate reached adjudication")
        obligation = dict(record.get("obligation") or {})
        sealed = {key: item for key, item in obligation.items() if key != "obligation_sha256"}
        if obligation.get("obligation_sha256") != canonical_sha256(sealed):
            raise ConjectureWideningError("obligation hash changed")
        adjudication = dict(record.get("adjudication") or {})
        verdict = adjudication.get("verdict")
        if verdict not in {"PROVED", "REFUTED", "OPEN"}:
            raise ConjectureWideningError(f"unknown verdict: {verdict}")
        if verdict == "REFUTED" and not adjudication.get("witness"):
            raise ConjectureWideningError("a refutation carries no witness")
        if verdict == "PROVED":
            bucket = (record.get("prior_art") or {}).get("bucket")
            if bucket not in {"proved_and_known", "proved_and_prior_art_not_found"}:
                raise ConjectureWideningError("a proved statement carries no prior-art triage")
        if not record.get("elementary_procedure_refusal"):
            raise ConjectureWideningError(
                "an admitted statement is about an object the elementary procedure could decide"
            )
        inputs = dict(obligation.get("inputs") or {})
        if record["kind"] == "zero_free_over_the_integers" and verdict == "PROVED":
            recheck_zero_free_certificate(
                list(inputs["coefficients"]), list(inputs["initial"]), adjudication
            )
        tally[str(verdict)] = tally.get(str(verdict), 0) + 1
    if tally != declared:
        raise ConjectureWideningError("verdict tally disagrees with the conjecture list")

    triage_counts = dict(counts.get("triage") or {})
    buckets = dict(value.get("triage") or {})
    if {key: len(item) for key, item in buckets.items()} != triage_counts:
        raise ConjectureWideningError("triage counts disagree with the triage lists")

    sweeps: list[Mapping[str, Any]] = []
    if "skolem_census" in value:
        sweeps.append(dict(value["skolem_census"]))
    sweeps.extend(dict(item) for item in (value.get("skolem_census_ladder") or ()))
    for sweep in sweeps:
        if not sweep.get("partition_is_exact"):
            raise ConjectureWideningError("the census partition is not exact; it is not exhaustive")
        sweep_counts = dict(sweep.get("counts") or {})
        parts = (
            int(sweep_counts.get("has_zero_with_explicit_witness", 0))
            + int(sweep_counts.get("zero_free_with_modulus_certificate", 0))
            + int(sweep_counts.get("unsettled", 0))
        )
        if parts != int(sweep_counts.get("total", -1)):
            raise ConjectureWideningError("census outcome counts do not sum to the total")
        declared_box = int((sweep.get("config") or {}).get("declared_box_size", -1))
        if int(sweep.get("visited", -1)) != declared_box or parts != declared_box:
            raise ConjectureWideningError("the census did not visit its whole declared box")
        routes = dict(sweep.get("zero_free_proof_route_histogram") or {})
        if sum(routes.values()) != int(sweep_counts.get("zero_free_with_modulus_certificate", -1)):
            raise ConjectureWideningError("census proof-route histogram does not sum")
        census_triage = dict(sweep.get("triage") or {})
        if sum(int(entry["count"]) for entry in census_triage.values()) != parts:
            raise ConjectureWideningError("census triage counts do not sum to the box")


def recheck_zero_free_certificate(
    coefficients: Sequence[int], initial: Sequence[int], adjudication: Mapping[str, Any]
) -> None:
    """Re-verify a zero-free certificate from its sealed inputs, not from the search state.

    Neither branch re-runs the search that produced the certificate.  The obstruction branch
    replays the recurrence, rechecks the single closure identity ``s(mu) = s(mu + lam)`` and
    scans the finite table for a zero.  The positivity branch rechecks the sign of the
    twisted coefficients, their sum, the base case and the finitely many indices below the
    induction start.  Both are cheap because a certificate is supposed to be.
    """

    route = adjudication.get("proof_route")
    if route == "local_obstruction":
        table = dict(adjudication.get("residue_table") or {})
        rebuilt = ResidueTable(
            modulus=int(table["modulus"]),
            order=int(table["order"]),
            mu=int(table["mu"]),
            lam=int(table["lam"]),
            values=tuple(int(item) for item in table["values"]),
            closure_state=tuple(int(item) for item in table["closure_state"]),
        )
        validate_residue_table(coefficients, initial, rebuilt)
        if 0 in rebuilt.values:
            raise ConjectureWideningError("a zero-free certificate contains a zero residue")
        return
    if route == "positivity_induction":
        certificate = dict(adjudication.get("positivity_certificate") or {})
        period = int(certificate["decimation"])
        proofs = list(certificate["class_proofs"])
        if period < 1 or len(proofs) != period:
            raise ConjectureWideningError("positivity certificate does not cover every class")
        order = len(coefficients)
        if period == 1:
            expected_by_omega = {
                omega: [
                    omega ** (index + 1) * coefficient
                    for index, coefficient in enumerate(coefficients)
                ]
                for omega in (1, -1)
            }
        else:
            base = characteristic_recurrence(matrix_power(companion_matrix(coefficients), period))
            expected_by_omega = {1: base}
        covered_from = max(int(proof["covers_indices_from"]) for proof in proofs)
        values = recurrence_values(coefficients, initial, covered_from + period * order + period)
        if any(value == 0 for value in values[:covered_from]):
            raise ConjectureWideningError("the object has a zero below the induction start")
        seen = set()
        for proof in proofs:
            residue = int(proof["residue_class"])
            omega = int(proof["omega"])
            sign = int(proof["sign"])
            start = int(proof["induction_start"])
            twisted = [int(item) for item in proof["twisted_coefficients"]]
            hypothesis = str(proof["hypothesis"])
            if residue in seen or not 0 <= residue < period:
                raise ConjectureWideningError("positivity classes are not a partition")
            seen.add(residue)
            if omega not in (1, -1) or sign not in (1, -1):
                raise ConjectureWideningError("positivity twist is not a sign")
            if twisted != expected_by_omega.get(omega):
                raise ConjectureWideningError("twisted coefficients do not match the twist")
            if _induction_admits(twisted) != hypothesis:
                raise ConjectureWideningError("the induction hypothesis does not close")
            if period == 1:
                series = [sign * omega**index * values[index] for index in range(len(values))]
            else:
                series = [sign * values[index] for index in range(residue, len(values), period)]
            base_case = series[start : start + order]
            if not _base_case_holds(base_case, hypothesis):
                raise ConjectureWideningError("positivity base case does not hold")
            if base_case != [int(item) for item in proof["base_case"]]:
                raise ConjectureWideningError("sealed positivity base case does not replay")
            if start * period + residue != int(proof["covers_indices_from"]):
                raise ConjectureWideningError("the class proof does not cover what it claims")
        return
    raise ConjectureWideningError(f"unknown zero-free proof route: {route}")


def write_receipt(result: Mapping[str, Any], output: str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(result) + b"\n")
    return path


def triage_digest(result: Mapping[str, Any]) -> dict[str, Any]:
    """The deliverable, stripped of the certificates: the three buckets and their provenance.

    Everything here is a projection of the sealed receipt -- the digest carries the receipt's
    own ``content_sha256`` so a reader can tell which run it came from and go back to the
    certificates that support any line of it.
    """

    ladder = list(result.get("skolem_census_ladder") or ())
    if "skolem_census" in result:
        ladder.append(result["skolem_census"])
    return {
        "schema_version": RESULT_SCHEMA + "-triage-digest",
        "receipt_content_sha256": result.get("content_sha256"),
        "claims": dict(result.get("claims") or {}),
        "counts": dict(result.get("counts") or {}),
        "triage": dict(result.get("triage") or {}),
        "prior_art": list(result.get("prior_art") or ()),
        "prior_art_search_record": dict(result.get("prior_art_search_record") or {}),
        "census_ladder": [
            {
                "config": sweep["config"],
                "counts": sweep["counts"],
                "triage": sweep["triage"],
                "zero_free_proof_route_histogram": sweep["zero_free_proof_route_histogram"],
                "certifying_modulus_histogram": sweep["certifying_modulus_histogram"],
                "partition_is_exact": sweep["partition_is_exact"],
                "counting_argument": sweep["counting_argument"],
                "unsettled_sample": sweep["unsettled_sample"],
            }
            for sweep in ladder
        ],
        "limitations": list(result.get("limitations") or ()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--output", default=None, help="write the sealed receipt here")
    parser.add_argument("--max-per-kind", type=int, default=60)
    parser.add_argument(
        "--census-order",
        type=int,
        action="append",
        default=None,
        help="sweep this order completely; repeat the flag for a ladder",
    )
    parser.add_argument("--census-coefficient-box", type=int, default=1)
    parser.add_argument("--census-initial-box", type=int, default=1)
    parser.add_argument("--census-zero-scan", type=int, default=80)
    parser.add_argument("--triage-output", default=None, help="write the triage digest here")
    parser.add_argument("--summary", action="store_true", help="print counts only")
    args = parser.parse_args(argv)

    census_config: Any = [
        CensusConfig(
            order=order,
            coefficient_box=args.census_coefficient_box,
            initial_box=args.census_initial_box,
            zero_scan=args.census_zero_scan,
        )
        for order in sorted(args.census_order or ())
    ]
    result = run_loop(max_per_kind=args.max_per_kind, census_config=census_config)
    validate_receipt(result)
    if args.output:
        write_receipt(result, args.output)
    if args.triage_output:
        write_receipt(triage_digest(result), args.triage_output)
    if args.summary:
        payload: Any = {"counts": result["counts"]}
        ladder = list(result.get("skolem_census_ladder") or ())
        if "skolem_census" in result:
            ladder.append(result["skolem_census"])
        if ladder:
            payload["skolem_census_ladder"] = [
                {
                    key: sweep[key]
                    for key in (
                        "config",
                        "counts",
                        "triage",
                        "zero_free_proof_route_histogram",
                        "certifying_modulus_histogram",
                        "partition_is_exact",
                    )
                }
                for sweep in ladder
            ]
    else:
        payload = result
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
