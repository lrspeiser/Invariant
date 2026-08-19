"""A search that is not allowed to report a null without a reachability certificate.

:mod:`sigma_theory_compiler.reachability_certificate` can decide, exactly, whether a target is
inside a declared fragment of the compositional grammar.  It shipped with no consumers, which
left the C1 falsifier live: *any exhaustive search reported as a negative result without a
reachability certificate for what it was looking for*.  A sweep that returns nothing is two
completely different facts wearing one word, and until something on the search path reads the
certificate, the engine cannot tell which of them it just produced.

This module is that consumer.  It defines the adjudication algebra, applies it to a real
exhaustive search it runs itself, and supplies the block that
:func:`sigma_theory_compiler.compositional_expression_search.run_campaign` now attaches to
every receipt it seals.

**The algebra.**  A search declares a *space* (a fragment of the grammar), a *target*, and an
*acceptance predicate* -- the thing that makes a hit count as a find rather than as a hit.
Combining what the search measured with what the certificate proves gives exactly four
outcomes, and there is no fifth:

``POSITIVE``
    the predicate had a witness.  No certificate is needed to believe a construction.
``REAL_NEGATIVE``
    the predicate had no witness **and** the certificate proves the target is inside the
    space.  The grammar could spell the target; the search exhausted the space; the absence is
    about the mathematics.  This is publishable under I6.
``UNINFORMATIVE_NULL``
    the predicate had no witness and the certificate either proves the target is *outside* the
    space, or resolves nothing.  The null is about the grammar, not about nature, and it must
    not be reported as a result.
``INCONSISTENT``
    the search and the certificate disagree -- the certificate proved the target reachable and
    the exhaustive enumeration found no program reaching it, or the certificate excluded a
    target the enumeration then produced.  Two independent computations of the same fact came
    out different, so both are suspect and the run fails closed.

The ``INCONSISTENT`` lane is not decoration.  The certificate computes reachability by upward
closure over *values*; the search here computes it by enumerating *programs* and running them.
Those are different algorithms over different objects, and the adjudicator cross-checks them
on every call.

**The search is real, and it is exhaustive.**  :func:`exhaustive_fragment_search` enumerates
every token sequence of a declared fragment at the real mode C program length of nine, filters
by the search module's own :func:`~.compositional_expression_search.program_status`, and
evaluates in exact ``Fraction`` arithmetic.  It emits a counting coverage argument in the S3
sense: the number of structurally valid programs it visited must equal the closed-form tree
count for that fragment, so "nothing was skipped" is a counted fact and not a spot check.
Fragments whose ordinal space exceeds :data:`MAX_ENUMERATED_SEQUENCES` are refused by name
rather than silently truncated -- a truncated sweep cannot support a negative at all.

**The campaign lane.**  The 1.911e12-candidate compositional campaign cannot be enumerated on
a CPU, so its reachability is established constructively instead: for each declared target this
module carries an explicit witness program in the real ordinal space, and proves the program
*exactly equal* to the target -- symbolically in mode C, and through ``summation`` or ``limit``
in mode F, which is the lane this module adds.  A target with no witness comes back
``UNRESOLVED``, and every one of its nulls is uninformative.  That is the honest state and the
receipt now says so per target: on the declared target list, mode C resolves nine of twelve and
mode F three of twelve.  ``zeta3`` has no mode C closed form but is reached in mode F by Apery's
series; ``catalan`` and ``euler_gamma`` are reached by neither, so the sweep's silence about
them is a statement about the grammar and is reported as one.

**Nothing here is believed because it was computed.**  Every adjudication is sealed and
re-derivable: :func:`verify_adjudication` re-runs the enumeration, re-verifies the certificate
with the certificate module's own verifier, re-derives the outcome, and requires the rebuilt
body to match the sealed one byte for byte.  :func:`certified_null_search_controls` then forges
twelve adjudications -- headed by a ``REAL_NEGATIVE`` claimed for a target proved outside the
space, which is precisely the lie this module exists to make impossible -- and every one of
them must be rejected.

Exact arithmetic throughout.  Targets are ``Fraction`` or sympy constants, values are
``Fraction``, the coverage argument is integer, and the symbolic lanes are sympy.  There is no
float on any path here.
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .compositional_expression_search import (
    _SYMPY_K,
    MODE_CONFIG,
    REAL_TARGETS,
    SERIES_START_INDEX,
    SYMBOLIC_TARGETS,
    TOKEN_NAMES,
    CompositionalSearchError,
    decode_ordinal,
    encode_program,
    mode_digits,
    program_status,
    render_infix,
    render_rpn,
    to_sympy,
    uses_variable,
)
from .reachability_certificate import (
    CERTIFICATE_SCHEMA,
    MODE_C_FULL,
    VERDICT_OUTSIDE,
    VERDICT_PROGRAM_REFUTED,
    VERDICT_REACHABLE,
    VERDICT_UNRESOLVED,
    Fragment,
    evaluate_exact,
    fragment_declaration,
    irrational_exclusion_certificate,
    parse_exact_rational,
    prove_symbolic_distinct,
    prove_symbolic_equal,
    rational_reachability_certificate,
    seal_certificate,
    symbolic_reachability_certificate,
    tokens_from_rpn,
    tree_counts_depth_free,
    verify_certificate,
)
from .sigma_core import canonical_sha256

#: The series index is the search module's own symbol.  A differently-assumed ``k`` would make
#: ``summation`` and ``limit`` answer a different question from the one the grammar poses, so it
#: is imported rather than re-declared.
SERIES_INDEX = _SYMPY_K

ADJUDICATION_SCHEMA = "invariant-certified-null-search-1.0"
RECEIPT_SCHEMA = "invariant-certified-null-search-receipt-1.0"

#: Outcomes.  A null is one of exactly two things, and a disagreement is neither of them.
OUTCOME_POSITIVE = "POSITIVE"
OUTCOME_REAL_NEGATIVE = "REAL_NEGATIVE"
OUTCOME_UNINFORMATIVE = "UNINFORMATIVE_NULL"
OUTCOME_INCONSISTENT = "INCONSISTENT"

OUTCOMES = (
    OUTCOME_POSITIVE,
    OUTCOME_REAL_NEGATIVE,
    OUTCOME_UNINFORMATIVE,
    OUTCOME_INCONSISTENT,
)

#: Why a null was uninformative.  Naming the reason is the whole point: "we found nothing"
#: carries no information, "the grammar cannot spell it" is a measurement of the grammar.
REASON_OUTSIDE_SPACE = "target_outside_declared_space"
REASON_REACHABILITY_UNRESOLVED = "reachability_unresolved"
REASON_WITNESS_REFUTED = "declared_witness_program_refuted"

#: The largest fragment ordinal space this module will enumerate on a CPU.  A fragment above it
#: is refused by name.  A truncated sweep cannot support a negative, so there is no partial
#: mode here on purpose.
MAX_ENUMERATED_SEQUENCES = 4_000_000

#: Witnesses recorded on a search block.  Beyond this the count is still exact; only the
#: listing is capped, and the block says so.
WITNESS_LISTING_CAP = 8


# ---------------------------------------------------------------------------
# The mode F fragment and its series/limit reachability lane
# ---------------------------------------------------------------------------

MODE_F_FULL = Fragment(
    name="mode_f_full",
    mode="F",
    token_names=tuple(TOKEN_NAMES[index] for index in mode_digits("F")),
    value_field="R",
)

#: How mode F reads a program.  ``series`` sums the term from :data:`SERIES_START_INDEX`;
#: ``limit`` takes the term's limit.  These are the search module's own two submodes.
SERIES_REDUCTIONS: dict[str, str] = {"series": "summation", "limit": "limit"}


def reduce_series_or_limit(expression: sp.Expr, submode: str) -> sp.Expr | None:
    """Close a mode F term into the real number the search compares against a target.

    Returns ``None`` when sympy leaves the reduction unevaluated, which is the honest answer
    for most of the space: an unevaluated ``Sum`` is not a value and must not be treated as one.
    """

    if submode not in SERIES_REDUCTIONS:
        raise CompositionalSearchError(f"unknown mode F submode {submode!r}")
    try:
        if submode == "series":
            value = sp.summation(expression, (SERIES_INDEX, SERIES_START_INDEX, sp.oo))
        else:
            value = sp.limit(expression, SERIES_INDEX, sp.oo)
    except (TypeError, ValueError, NotImplementedError, AttributeError, ZeroDivisionError):
        return None
    if value is None or value.has(sp.Sum) or value.has(sp.Limit):
        return None
    if value.free_symbols:
        return None
    if not value.is_finite:
        return None
    return value


def _series_program_block(tokens: Sequence[int], fragment: Fragment) -> dict[str, Any]:
    """Everything a series/limit certificate says about its program, from the tokens alone."""

    tokens = tuple(tokens)
    status = program_status(tokens)
    length_ok = len(tokens) == fragment.program_length
    mentions = uses_variable(tokens)
    admissible = status == "ok" and length_ok and mentions
    ordinal = encode_program(tokens, fragment.mode) if admissible else None
    roundtrip = ordinal is not None and decode_ordinal(ordinal, fragment.mode) == tokens
    expression = to_sympy(tokens) if status == "ok" else None
    return {
        "rpn": render_rpn(tokens),
        "infix": render_infix(tokens),
        "tokens": list(tokens),
        "program_status": status,
        "length_matches_mode": length_ok,
        "mentions_index_variable": mentions,
        "ordinal": ordinal,
        "ordinal_space_size": int(MODE_CONFIG[fragment.mode]["space_size"]),
        "codec_roundtrip": roundtrip,
        "term_sympy": None if expression is None else sp.srepr(expression),
        "term_printed": None if expression is None else str(expression),
    }


def series_limit_reachability_certificate(
    rpn: str,
    target_name: str,
    submode: str,
    fragment: Fragment = MODE_F_FULL,
) -> dict[str, Any]:
    """Certify one explicit mode F derivation of a named constant, exactly.

    Mode C asks what a program *is*; mode F asks what the series or sequence a program defines
    *converges to*, and that is a different question needing a different proof.  This lane
    closes the term with ``summation`` or ``limit`` and then hands the closed form to the same
    exact tactic ladder the mode C lane uses, so ``Sum_{k>=1} 1/k^2 = pi^2/6`` is proved rather
    than checked to a digit count.
    """

    if target_name not in SYMBOLIC_TARGETS:
        raise CompositionalSearchError(f"unknown symbolic target {target_name!r}")
    if submode not in SERIES_REDUCTIONS:
        raise CompositionalSearchError(f"unknown mode F submode {submode!r}")
    if fragment.mode != "F":
        raise CompositionalSearchError("the series/limit lane needs a mode F fragment")
    tokens = tokens_from_rpn(rpn, fragment)
    program = _series_program_block(tokens, fragment)
    target = SYMBOLIC_TARGETS[target_name]

    body: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "lane": "series_limit",
        "fragment": fragment_declaration(fragment),
        "target": {"kind": "symbolic", "name": target_name, "sympy": sp.srepr(target)},
        "submode": submode,
        "reduction": {
            "operator": SERIES_REDUCTIONS[submode],
            "index": str(SERIES_INDEX),
            "start_index": SERIES_START_INDEX,
        },
        "program": program,
    }

    expression = to_sympy(tokens) if program["program_status"] == "ok" else None
    if expression is None or not program["codec_roundtrip"]:
        body["verdict"] = VERDICT_UNRESOLVED
        body["proof"] = {
            "kind": "none",
            "statement": (
                "the program is not a well-formed mode F member: it must be structurally "
                "valid, of the declared length, and must mention the index variable"
            ),
        }
        return seal_certificate(body)

    closed = reduce_series_or_limit(expression, submode)
    body["reduction"]["closed_form_sympy"] = None if closed is None else sp.srepr(closed)
    body["reduction"]["closed_form_printed"] = None if closed is None else str(closed)
    if closed is None:
        body["verdict"] = VERDICT_UNRESOLVED
        body["proof"] = {
            "kind": "none",
            "statement": (
                "the declared reduction did not close: an unevaluated Sum or Limit is not a "
                "value and nothing is claimed from one"
            ),
        }
        return seal_certificate(body)

    equality = prove_symbolic_equal(closed, target)
    if equality["proved"]:
        body["verdict"] = VERDICT_REACHABLE
        body["proof"] = {
            "kind": "series_limit_identity",
            "tactic": equality["tactic"],
            "attempts": equality["attempts"],
            "statement": (
                "the declared reduction of this program is exactly equal to the target, proved "
                "symbolically; the target is therefore inside the mode F space"
            ),
        }
        return seal_certificate(body)

    refutation = prove_symbolic_distinct(closed, target)
    if refutation["refuted"]:
        body["verdict"] = VERDICT_PROGRAM_REFUTED
        body["proof"] = {
            "kind": "strict_inequality",
            "relation": refutation["relation"],
            "attempts": equality["attempts"],
            "scope": "this derivation only",
            "statement": (
                "the reduction is proved strictly unequal to the target, so this derivation is "
                "refuted; the grammar may still reach the target by another program"
            ),
        }
        return seal_certificate(body)

    body["verdict"] = VERDICT_UNRESOLVED
    body["proof"] = {
        "kind": "none",
        "attempts": equality["attempts"],
        "statement": "neither equality nor strict inequality was proved; nothing is claimed",
    }
    return seal_certificate(body)


def verify_series_limit_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive a series/limit certificate from its own declarations, trusting nothing.

    Two jobs, exactly as the certificate module does them: re-execute the substantive claims,
    then rebuild the whole body and require an exact match so that prose, tactic names and
    scope statements cannot drift away from what was actually proved.
    """

    if certificate.get("schema_version") != CERTIFICATE_SCHEMA:
        raise CompositionalSearchError(
            f"certificate rejected: unknown schema_version {certificate.get('schema_version')!r}"
        )
    if certificate.get("lane") != "series_limit":
        raise CompositionalSearchError("certificate rejected: not a series/limit certificate")
    if "content_sha256" not in certificate:
        raise CompositionalSearchError("certificate rejected: unsealed certificate")
    payload = {key: value for key, value in certificate.items() if key != "content_sha256"}
    if canonical_sha256(payload) != certificate["content_sha256"]:
        raise CompositionalSearchError("certificate rejected: seal does not match the body")

    declaration = dict(certificate.get("fragment", {}))
    fragment = Fragment(
        name=str(declaration.get("name")),
        mode=str(declaration.get("mode")),
        token_names=tuple(str(item) for item in declaration.get("token_names", ())),
        value_field=str(declaration.get("value_field")),
    )
    if declaration != fragment_declaration(fragment):
        raise CompositionalSearchError(
            "certificate rejected: declared fragment block does not match a recomputation"
        )
    if fragment.mode != "F":
        raise CompositionalSearchError("certificate rejected: series/limit lane needs mode F")

    target_block = certificate.get("target", {})
    name = str(target_block.get("name"))
    if name not in SYMBOLIC_TARGETS:
        raise CompositionalSearchError(f"certificate rejected: unknown target {name!r}")
    if sp.srepr(SYMBOLIC_TARGETS[name]) != target_block.get("sympy"):
        raise CompositionalSearchError(
            "certificate rejected: declared target srepr is not the search module's target"
        )

    submode = str(certificate.get("submode"))
    program = dict(certificate.get("program", {}))
    tokens = tuple(int(item) for item in program.get("tokens", ()))
    if not tokens:
        raise CompositionalSearchError("certificate rejected: certificate carries no tokens")
    allowed = set(fragment.token_indices)
    outside = sorted({TOKEN_NAMES[token] for token in tokens if token not in allowed})
    if outside:
        raise CompositionalSearchError(
            f"certificate rejected: program uses tokens outside the fragment: {outside}"
        )
    if program != _series_program_block(tokens, fragment):
        raise CompositionalSearchError(
            "certificate rejected: declared program block does not match a recomputation"
        )

    verdict = certificate.get("verdict")
    checks: dict[str, Any] = {"seal": True, "tokens_in_fragment": True}
    if verdict in {VERDICT_REACHABLE, VERDICT_PROGRAM_REFUTED}:
        expression = to_sympy(tokens)
        if expression is None:
            raise CompositionalSearchError("certificate rejected: program has no symbolic form")
        closed = reduce_series_or_limit(expression, submode)
        if closed is None:
            raise CompositionalSearchError(
                "certificate rejected: the declared reduction does not close on replay"
            )
        if sp.srepr(closed) != certificate.get("reduction", {}).get("closed_form_sympy"):
            raise CompositionalSearchError(
                "certificate rejected: declared closed form does not match a recomputation"
            )
        target = SYMBOLIC_TARGETS[name]
        if verdict == VERDICT_REACHABLE:
            replay = prove_symbolic_equal(closed, target)
            if not replay["proved"]:
                raise CompositionalSearchError(
                    "certificate rejected: the claimed series/limit identity does not replay"
                )
            checks["series_limit_identity_replayed"] = replay["tactic"]
        else:
            replay = prove_symbolic_distinct(closed, target)
            if not replay["refuted"]:
                raise CompositionalSearchError(
                    "certificate rejected: the claimed strict inequality does not replay"
                )
            checks["strict_inequality_replayed"] = replay["relation"]
        checks["codec_roundtrip"] = True
    elif verdict == VERDICT_UNRESOLVED:
        if certificate.get("proof", {}).get("kind") not in {None, "none"}:
            raise CompositionalSearchError(
                "certificate rejected: UNRESOLVED must not carry a substantive proof"
            )
    else:
        raise CompositionalSearchError(f"certificate rejected: unknown verdict {verdict!r}")

    rebuilt = series_limit_reachability_certificate(
        str(program["rpn"]), name, submode, fragment
    )
    if rebuilt["content_sha256"] != certificate["content_sha256"]:
        differing = sorted(
            key
            for key in set(rebuilt) | set(certificate)
            if key != "content_sha256" and rebuilt.get(key) != certificate.get(key)
        )
        raise CompositionalSearchError(
            f"certificate rejected: body does not match a regeneration: {differing}"
        )
    checks["regenerated"] = True
    return {"accepted": True, "verdict": verdict, "checks": checks}


def verify_any_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Route a certificate to the verifier that owns its lane.  No lane means no acceptance."""

    lane = certificate.get("lane")
    if lane == "series_limit":
        return verify_series_limit_certificate(certificate)
    return verify_certificate(certificate)


# ---------------------------------------------------------------------------
# The search: exhaustive, exact, and counted
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SearchDeclaration:
    """What a search is looking for, where, and what counts as having found it.

    ``known_derivations`` is the acceptance predicate and it is the reason a ``REAL_NEGATIVE``
    can exist at all: a search for a *new* derivation of a constant can exhaust a space that
    demonstrably contains the constant and still come back empty, and that null is about the
    mathematics rather than about the alphabet.
    """

    name: str
    fragment: Fragment
    target: str
    target_kind: str  # "rational" | "symbolic"
    objective: str
    known_derivations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.target_kind not in {"rational", "symbolic"}:
            raise CompositionalSearchError(f"unknown target kind {self.target_kind!r}")
        if self.target_kind == "symbolic" and self.target not in SYMBOLIC_TARGETS:
            raise CompositionalSearchError(f"unknown symbolic target {self.target!r}")
        if self.target_kind == "rational":
            parse_exact_rational(self.target)
        for rpn in self.known_derivations:
            tokens_from_rpn(rpn, self.fragment)


def _target_comparison(declaration: SearchDeclaration) -> tuple[str, Fraction | None]:
    """How the enumerator decides ``value == target``, decided once before the sweep.

    Three cases and no guessing: an exact rational compares by equality of ``Fraction``; a
    target sympy proves irrational can never equal a rational, so the comparison is decided in
    advance; a target whose rationality is *open* -- Catalan's constant is the standing example
    -- leaves the comparison undecided, and the sweep reports that instead of returning a zero
    it cannot justify.
    """

    if declaration.target_kind == "rational":
        return "exact_rational_equality", parse_exact_rational(declaration.target)
    target = SYMBOLIC_TARGETS[declaration.target]
    rationality = target.is_rational
    if rationality is False:
        return "irrational_by_sympy", None
    if rationality is True:
        return "exact_rational_equality", Fraction(sp.Rational(target))
    return "undecided", None


def exhaustive_fragment_search(declaration: SearchDeclaration) -> dict[str, Any]:
    """Enumerate every program of the declared fragment and report what matched.

    This is the search, not a model of one.  It walks the fragment's own mixed-radix ordinal
    space at the real mode C program length, decides validity with the search module's own
    ``program_status``, evaluates in exact ``Fraction`` arithmetic, and closes with a counting
    argument: the number of structurally valid programs visited must equal the closed-form tree
    count for the fragment.  That equality is the coverage certificate -- nothing was skipped,
    counted rather than sampled.
    """

    fragment = declaration.fragment
    if fragment.value_field != "Q":
        raise CompositionalSearchError(
            "the exact enumerator needs a rationally closed fragment; "
            f"{fragment.name} declares field {fragment.value_field}"
        )
    tokens = fragment.token_indices
    length = fragment.program_length
    space = len(tokens) ** length
    if space > MAX_ENUMERATED_SEQUENCES:
        raise CompositionalSearchError(
            f"fragment {fragment.name} has {space} token sequences, above the declared CPU "
            f"enumeration cap of {MAX_ENUMERATED_SEQUENCES}; a truncated sweep cannot support "
            "a negative, so this is refused rather than sampled"
        )

    comparison, target_value = _target_comparison(declaration)
    known = set(declaration.known_derivations)
    witnesses: list[dict[str, Any]] = []
    valid = 0
    evaluated = 0
    matched = 0
    accepted = 0
    for sequence in itertools.product(tokens, repeat=length):
        if program_status(sequence) != "ok":
            continue
        valid += 1
        value, _ = evaluate_exact(sequence)
        if value is None:
            continue
        evaluated += 1
        if target_value is None or value != target_value:
            continue
        matched += 1
        rpn = render_rpn(sequence)
        is_new = rpn not in known
        accepted += int(is_new)
        if len(witnesses) < WITNESS_LISTING_CAP:
            witnesses.append(
                {
                    "rpn": rpn,
                    "infix": render_infix(sequence),
                    "ordinal_in_mode_space": encode_program(sequence, fragment.mode),
                    "exact_value": str(value),
                    "already_catalogued": not is_new,
                }
            )

    closed_form_valid = tree_counts_depth_free(fragment)[-1]
    decided = comparison != "undecided"
    return {
        "declaration": {
            "name": declaration.name,
            "objective": declaration.objective,
            "target": declaration.target,
            "target_kind": declaration.target_kind,
            "known_derivations": list(declaration.known_derivations),
            "fragment": fragment_declaration(fragment),
        },
        "target_comparison": comparison,
        "comparison_decided": decided,
        "coverage": {
            "token_sequences_enumerated": space,
            "declared_space_size": space,
            "exhaustive": True,
            "structurally_valid_measured": valid,
            "structurally_valid_closed_form": closed_form_valid,
            "coverage_argument_holds": valid == closed_form_valid,
            "programs_with_a_value": evaluated,
            "argument": (
                "every token sequence of the fragment alphabet at the declared program length "
                "was visited; the structurally valid count is compared against the closed-form "
                "tree count for the same fragment, so completeness is counted, not sampled"
            ),
        },
        "witness_count": matched if decided else None,
        "accepted_count": accepted if decided else None,
        "witnesses": witnesses,
        "witnesses_listed": len(witnesses),
        "witness_listing_cap": WITNESS_LISTING_CAP,
    }


# ---------------------------------------------------------------------------
# Certificates for a declared search
# ---------------------------------------------------------------------------


def certificate_for(declaration: SearchDeclaration) -> dict[str, Any]:
    """The reachability certificate this search's null will be adjudicated against.

    The lane is forced by the declaration, never chosen for convenience: a rational target in a
    rationally closed fragment goes to the exact-image lane, a symbolic target to the field
    closure lane.  Both certificates are about the fragment the search actually enumerated.
    """

    if declaration.target_kind == "rational":
        return rational_reachability_certificate(
            declaration.target, declaration.fragment, exhaustive=True
        )
    return irrational_exclusion_certificate(declaration.target, declaration.fragment)


# ---------------------------------------------------------------------------
# Adjudication
# ---------------------------------------------------------------------------


def _bind_certificate(declaration: SearchDeclaration, certificate: Mapping[str, Any]) -> None:
    """The certificate must be about this search's space and this search's target.

    A certificate is a true statement about *something*.  Attaching a true statement about a
    different fragment, or about a different target, to a null is the cheapest possible forgery
    and it is the one this function exists to refuse.
    """

    if dict(certificate.get("fragment", {})) != fragment_declaration(declaration.fragment):
        raise CompositionalSearchError(
            "adjudication rejected: the certificate is about a different fragment from the one "
            "the search enumerated"
        )
    target_block = certificate.get("target", {})
    if declaration.target_kind == "rational":
        if target_block.get("kind") != "rational":
            raise CompositionalSearchError(
                "adjudication rejected: a rational target needs a rational certificate"
            )
        if parse_exact_rational(str(target_block.get("exact"))) != parse_exact_rational(
            declaration.target
        ):
            raise CompositionalSearchError(
                "adjudication rejected: the certificate is about a different target"
            )
        return
    if target_block.get("kind") != "symbolic":
        raise CompositionalSearchError(
            "adjudication rejected: a symbolic target needs a symbolic certificate"
        )
    if str(target_block.get("name")) != declaration.target:
        raise CompositionalSearchError(
            "adjudication rejected: the certificate is about a different target"
        )


#: Every statement the algebra can attach to an outcome, keyed by the case that produced it.
#: Kept as data so a reader can see the whole decision table at once and so no branch can
#: quietly acquire prose that overstates what it proved.
OUTCOME_STATEMENTS: dict[str, str] = {
    "comparison_undecided": (
        "the search could not decide equality with the target, so it measured nothing about "
        "it; the null is about the comparison, not about the mathematics"
    ),
    "positive": (
        "the acceptance predicate had a witness; a construction needs no certificate"
    ),
    "reachable_but_no_program": (
        "the certificate proves the target reachable and the exhaustive enumeration produced "
        "no program reaching it; two independent computations disagree"
    ),
    "real_negative": (
        "the certificate proves the target is inside the declared space and the search "
        "exhausted that space without an accepted witness; the absence is about the "
        "mathematics and is publishable as a negative"
    ),
    "excluded_but_program_found": (
        "the certificate excludes the target from the space and the enumeration produced a "
        "program reaching it; two independent computations disagree"
    ),
    "outside_space": (
        "the certificate proves the target is outside the declared space, so the null is a "
        "fact about the grammar and not about nature; it must not be reported as a negative "
        "result"
    ),
    "witness_refuted": (
        "the declared witness program was refuted, which says nothing about whether the "
        "grammar reaches the target by some other program; reachability is unestablished"
    ),
    "unresolved": (
        "no reachability argument fired, so the null distinguishes nothing; under C1 it must "
        "not be reported as a result"
    ),
}


def _outcome(
    verdict: str,
    witness_count: int | None,
    accepted_count: int | None,
) -> tuple[str, str | None, str]:
    """The whole algebra, in one place, as a total function of two measured facts."""

    if accepted_count is None or witness_count is None:
        return (
            OUTCOME_UNINFORMATIVE,
            REASON_REACHABILITY_UNRESOLVED,
            OUTCOME_STATEMENTS["comparison_undecided"],
        )
    # Contradictions are decided first, before anything is called a find.  A search that
    # produces a program for a target its own certificate excludes has produced *something*,
    # and reporting that as a positive would bury the disagreement under a success.
    if verdict == VERDICT_OUTSIDE and witness_count > 0:
        return (OUTCOME_INCONSISTENT, None, OUTCOME_STATEMENTS["excluded_but_program_found"])
    if verdict == VERDICT_REACHABLE and witness_count == 0:
        return (OUTCOME_INCONSISTENT, None, OUTCOME_STATEMENTS["reachable_but_no_program"])
    if accepted_count > 0:
        return (OUTCOME_POSITIVE, None, OUTCOME_STATEMENTS["positive"])
    if verdict == VERDICT_REACHABLE:
        return (OUTCOME_REAL_NEGATIVE, None, OUTCOME_STATEMENTS["real_negative"])
    if verdict == VERDICT_OUTSIDE:
        return (
            OUTCOME_UNINFORMATIVE,
            REASON_OUTSIDE_SPACE,
            OUTCOME_STATEMENTS["outside_space"],
        )
    if verdict == VERDICT_PROGRAM_REFUTED:
        return (
            OUTCOME_UNINFORMATIVE,
            REASON_WITNESS_REFUTED,
            OUTCOME_STATEMENTS["witness_refuted"],
        )
    return (
        OUTCOME_UNINFORMATIVE,
        REASON_REACHABILITY_UNRESOLVED,
        OUTCOME_STATEMENTS["unresolved"],
    )


def adjudicate_null(
    declaration: SearchDeclaration,
    search: Mapping[str, Any],
    certificate: Mapping[str, Any],
) -> dict[str, Any]:
    """Combine a search block and a verified certificate into one sealed adjudication."""

    _bind_certificate(declaration, certificate)
    report = verify_any_certificate(certificate)
    if not report.get("accepted"):
        raise CompositionalSearchError("adjudication rejected: the certificate did not verify")
    if not search.get("coverage", {}).get("coverage_argument_holds"):
        raise CompositionalSearchError(
            "adjudication rejected: the search's own coverage argument does not hold, so it "
            "cannot support any verdict about absence"
        )
    verdict = str(certificate.get("verdict"))
    witness_count = search.get("witness_count")
    accepted_count = search.get("accepted_count")
    outcome, reason, statement = _outcome(verdict, witness_count, accepted_count)
    body = {
        "schema_version": ADJUDICATION_SCHEMA,
        "search": dict(search),
        "certificate": dict(certificate),
        "certificate_verified": True,
        "certificate_verdict": verdict,
        "outcome": outcome,
        "uninformative_reason": reason,
        "statement": statement,
        "publishable_as_a_negative": outcome == OUTCOME_REAL_NEGATIVE,
    }
    return seal_certificate(body)


def certified_search(declaration: SearchDeclaration) -> dict[str, Any]:
    """Run the search, build its certificate, adjudicate, and refuse to publish a disagreement."""

    search = exhaustive_fragment_search(declaration)
    certificate = certificate_for(declaration)
    adjudication = adjudicate_null(declaration, search, certificate)
    if adjudication["outcome"] == OUTCOME_INCONSISTENT:
        raise CompositionalSearchError(
            f"search {declaration.name}: {adjudication['statement']}"
        )
    return adjudication


def verify_adjudication(adjudication: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild an adjudication end to end and require it to match, or reject it.

    The search is re-run from the declaration the adjudication itself carries, the certificate
    is re-verified by the module that issued it, the outcome is re-derived from the algebra,
    and the whole body is regenerated and compared.  Nothing on the record is taken on trust,
    including the record's own account of what the search measured.
    """

    if adjudication.get("schema_version") != ADJUDICATION_SCHEMA:
        raise CompositionalSearchError(
            f"adjudication rejected: unknown schema "
            f"{adjudication.get('schema_version')!r}"
        )
    if "content_sha256" not in adjudication:
        raise CompositionalSearchError("adjudication rejected: unsealed adjudication")
    payload = {key: value for key, value in adjudication.items() if key != "content_sha256"}
    if canonical_sha256(payload) != adjudication["content_sha256"]:
        raise CompositionalSearchError("adjudication rejected: seal does not match the body")
    if adjudication.get("outcome") not in OUTCOMES:
        raise CompositionalSearchError(
            f"adjudication rejected: unknown outcome {adjudication.get('outcome')!r}"
        )

    search = adjudication.get("search", {})
    declared = search.get("declaration", {})
    fragment_block = dict(declared.get("fragment", {}))
    fragment = Fragment(
        name=str(fragment_block.get("name")),
        mode=str(fragment_block.get("mode")),
        token_names=tuple(str(item) for item in fragment_block.get("token_names", ())),
        value_field=str(fragment_block.get("value_field")),
    )
    if fragment_block != fragment_declaration(fragment):
        raise CompositionalSearchError(
            "adjudication rejected: declared fragment block does not match a recomputation"
        )
    declaration = SearchDeclaration(
        name=str(declared.get("name")),
        fragment=fragment,
        target=str(declared.get("target")),
        target_kind=str(declared.get("target_kind")),
        objective=str(declared.get("objective")),
        known_derivations=tuple(str(item) for item in declared.get("known_derivations", ())),
    )

    replayed = exhaustive_fragment_search(declaration)
    if replayed != dict(search):
        raise CompositionalSearchError(
            "adjudication rejected: the search block does not match a re-run of its own "
            "declaration"
        )
    rebuilt = adjudicate_null(declaration, replayed, adjudication.get("certificate", {}))
    if rebuilt["content_sha256"] != adjudication["content_sha256"]:
        differing = sorted(
            key
            for key in set(rebuilt) | set(adjudication)
            if key != "content_sha256" and rebuilt.get(key) != adjudication.get(key)
        )
        raise CompositionalSearchError(
            f"adjudication rejected: body does not match a regeneration: {differing}"
        )
    return {
        "accepted": True,
        "outcome": adjudication["outcome"],
        "checks": {
            "seal": True,
            "search_replayed": True,
            "certificate_verified": True,
            "outcome_rederived": True,
            "regenerated": True,
        },
    }


# ---------------------------------------------------------------------------
# The campaign lane: reachability for the real 1.9e12-candidate sweep
# ---------------------------------------------------------------------------

#: Explicit mode C witnesses in the real 23^9 ordinal space, each proved *exactly* equal to its
#: target by the symbolic lane.  Seven come from the search module's own built-in identity
#: table; ``sqrt3`` and ``ln3`` are added here because the grammar plainly reaches them and a
#: certificate that said UNRESOLVED for them would be under-claiming rather than cautious.
MODE_C_WITNESSES: dict[str, str] = {
    "pi": "1 atan 2 sqr mul 1 mul 1 mul",
    "e": "1/2 exp sqr 1 mul 1 mul 1 mul",
    "sqrt2": "4 sqrt sqrt 1 mul 1 mul 1 mul",
    "sqrt3": "3 1/2 pow sqr sqrt 1 mul 1 mul",
    "ln2": "4 sqrt ln 1 mul 1 mul 1 mul",
    "ln3": "3 sqr ln 2 div 1 mul 1 mul",
    "zeta2": "1 atan 4 mul sqr 2 3 mul div",
    "phi": "1 5 sqrt add 2 recip mul 1 mul",
    "e_pi": "1 atan 4 mul exp 1 mul 1 mul",
}

#: Explicit mode F witnesses in the real 24^8 ordinal space, each closed by ``summation`` or
#: ``limit`` and then proved equal to its target.
MODE_F_WITNESSES: dict[str, tuple[str, str]] = {
    "zeta2": ("k sqr recip 2 sqr 4 div mul", "series"),
    "zeta3": ("k 3 pow recip 1 mul 1 mul", "series"),
    "e": ("k 1 sqr add k div k pow", "limit"),
}

#: Targets with no witness in either mode.  Named, because "we have no argument" is itself a
#: measurement and hiding it is what C1 forbids.
NO_KNOWN_CLOSED_FORM = ("catalan", "euler_gamma")


def campaign_reachability_certificates() -> list[dict[str, Any]]:
    """One verified reachability record per (mode, declared target) of the real campaign."""

    records: list[dict[str, Any]] = []
    for target in (str(row["name"]) for row in REAL_TARGETS):
        for mode in ("C", "F"):
            if mode == "C":
                rpn = MODE_C_WITNESSES.get(target)
                certificate = (
                    None
                    if rpn is None
                    else symbolic_reachability_certificate(rpn, target, MODE_C_FULL)
                )
                submode = "constant"
            else:
                witness = MODE_F_WITNESSES.get(target)
                rpn = None if witness is None else witness[0]
                submode = None if witness is None else witness[1]
                certificate = (
                    None
                    if witness is None
                    else series_limit_reachability_certificate(
                        witness[0], target, witness[1], MODE_F_FULL
                    )
                )
            if certificate is None:
                records.append(
                    {
                        "mode": mode,
                        "target": target,
                        "submode": submode,
                        "witness_rpn": None,
                        "verdict": VERDICT_UNRESOLVED,
                        "certificate_verified": False,
                        "certificate": None,
                        "statement": (
                            "no witness program is declared for this target in this mode, so "
                            "reachability is unestablished and every null here is "
                            "uninformative"
                        ),
                    }
                )
                continue
            report = verify_any_certificate(certificate)
            records.append(
                {
                    "mode": mode,
                    "target": target,
                    "submode": submode,
                    "witness_rpn": rpn,
                    "verdict": str(certificate["verdict"]),
                    "certificate_verified": bool(report["accepted"]),
                    "certificate": certificate,
                    "statement": (
                        "an explicit program in the declared ordinal space is proved exactly "
                        "equal to the target, so the target is inside the searched space"
                    ),
                }
            )
    return records


def adjudicate_campaign_nulls(
    records: Sequence[Mapping[str, Any]],
    findings: Mapping[str, Mapping[str, int]],
) -> list[dict[str, Any]]:
    """Adjudicate one (mode, target) cell of a campaign from its certificate and its yield.

    ``findings[mode][target]`` is the number of results the campaign is willing to call a find
    for that cell.  Zero is a null, and a null is exactly as informative as its certificate.
    """

    rows: list[dict[str, Any]] = []
    for record in records:
        mode = str(record["mode"])
        target = str(record["target"])
        found = int(findings.get(mode, {}).get(target, 0))
        verdict = str(record["verdict"])
        if verdict == VERDICT_REACHABLE and not record.get("certificate_verified"):
            raise CompositionalSearchError(
                f"campaign adjudication rejected: the {mode}/{target} certificate claims "
                "REACHABLE but did not verify"
            )
        # A campaign cell is not an exhaustive value search over a CPU-enumerable fragment, so
        # witness existence is exactly the certificate's verdict: a REACHABLE certificate is
        # itself an exhibited program.
        witness_count = 1 if verdict == VERDICT_REACHABLE else 0
        outcome, reason, statement = _outcome(verdict, witness_count, found)
        rows.append(
            {
                "mode": mode,
                "target": target,
                "submode": record.get("submode"),
                "findings": found,
                "certificate_verdict": verdict,
                "certificate_verified": bool(record.get("certificate_verified")),
                "witness_rpn": record.get("witness_rpn"),
                "outcome": outcome,
                "uninformative_reason": reason,
                "statement": statement,
                "publishable_as_a_negative": outcome == OUTCOME_REAL_NEGATIVE,
            }
        )
    return rows


def campaign_reachability_block(findings: Mapping[str, Mapping[str, int]]) -> dict[str, Any]:
    """The block :func:`run_campaign` attaches so no null it seals is uncertified.

    ``findings`` maps mode to target to the number of results the campaign counts as a find.
    Every declared target of every mode gets a row, whether or not it produced anything, which
    is what makes the block a complete account of the campaign's silence.
    """

    records = campaign_reachability_certificates()
    adjudication = adjudicate_campaign_nulls(records, findings)
    inconsistent = [row for row in adjudication if row["outcome"] == OUTCOME_INCONSISTENT]
    if inconsistent:
        cells = sorted(f"{row['mode']}/{row['target']}" for row in inconsistent)
        raise CompositionalSearchError(
            f"campaign reachability is inconsistent with the campaign's own findings: {cells}"
        )
    body = {
        "schema_version": ADJUDICATION_SCHEMA,
        "falsifier": (
            "C1: any exhaustive search reported as a negative result without a reachability "
            "certificate for what it was looking for"
        ),
        "scope": (
            "one row per declared target per mode; a row whose certificate does not prove the "
            "target reachable is an uninformative null and is not a result"
        ),
        "certificates": records,
        "null_adjudication": adjudication,
        "summary": {
            "cells": len(adjudication),
            "positive": sum(1 for row in adjudication if row["outcome"] == OUTCOME_POSITIVE),
            "real_negative": sum(
                1 for row in adjudication if row["outcome"] == OUTCOME_REAL_NEGATIVE
            ),
            "uninformative_null": sum(
                1 for row in adjudication if row["outcome"] == OUTCOME_UNINFORMATIVE
            ),
            "reachable_cells": sum(
                1 for row in adjudication if row["certificate_verdict"] == VERDICT_REACHABLE
            ),
            "targets_with_no_known_closed_form": list(NO_KNOWN_CLOSED_FORM),
            "every_null_carries_a_verdict": all(
                row["certificate_verdict"] in {VERDICT_REACHABLE, VERDICT_OUTSIDE,
                                               VERDICT_UNRESOLVED, VERDICT_PROGRAM_REFUTED}
                for row in adjudication
            ),
        },
    }
    return seal_certificate(body)


def verify_campaign_reachability_block(block: Mapping[str, Any]) -> dict[str, Any]:
    """Re-verify every certificate on a campaign block and re-derive every outcome."""

    if block.get("schema_version") != ADJUDICATION_SCHEMA:
        raise CompositionalSearchError(
            f"reachability block rejected: unknown schema {block.get('schema_version')!r}"
        )
    if "content_sha256" not in block:
        raise CompositionalSearchError("reachability block rejected: unsealed block")
    payload = {key: value for key, value in block.items() if key != "content_sha256"}
    if canonical_sha256(payload) != block["content_sha256"]:
        raise CompositionalSearchError("reachability block rejected: seal does not match")

    verified = 0
    for record in block.get("certificates", []):
        certificate = record.get("certificate")
        if certificate is None:
            if str(record.get("verdict")) != VERDICT_UNRESOLVED:
                raise CompositionalSearchError(
                    "reachability block rejected: a record with no certificate claims a verdict"
                )
            continue
        report = verify_any_certificate(certificate)
        if not report["accepted"]:
            raise CompositionalSearchError(
                "reachability block rejected: a certificate did not verify"
            )
        if str(certificate.get("verdict")) != str(record.get("verdict")):
            raise CompositionalSearchError(
                "reachability block rejected: a record's verdict is not its certificate's"
            )
        verified += 1

    findings = {
        mode: {
            str(row["target"]): int(row["findings"])
            for row in block.get("null_adjudication", [])
            if str(row["mode"]) == mode
        }
        for mode in ("C", "F")
    }
    rebuilt = campaign_reachability_block(findings)
    if rebuilt["content_sha256"] != block["content_sha256"]:
        differing = sorted(
            key
            for key in set(rebuilt) | set(block)
            if key != "content_sha256" and rebuilt.get(key) != block.get(key)
        )
        raise CompositionalSearchError(
            f"reachability block rejected: body does not match a regeneration: {differing}"
        )
    return {
        "accepted": True,
        "certificates_verified": verified,
        "cells": len(block.get("null_adjudication", [])),
    }


# ---------------------------------------------------------------------------
# The declared demonstration searches
# ---------------------------------------------------------------------------

#: Four tokens, 4^9 = 262,144 sequences, 7,168 structurally valid programs, 81 distinct exact
#: rationals.  Small enough to exhaust on a CPU at the real mode C program length, which is
#: what makes an exhaustive negative over it meaningful rather than rhetorical.
DEMO_FRAGMENT = Fragment("tiny_sub_div", "C", ("1", "2", "sub", "div"), "Q")

#: ``1/16`` has exactly one derivation in the demonstration fragment.  Cataloguing that one
#: derivation makes the search for a *new* derivation exhaust a space that provably contains
#: the target -- the only way a REAL_NEGATIVE can honestly arise.
DEMO_REACHABLE_TARGET = "1/16"
DEMO_KNOWN_DERIVATION = "1 2 div 2 div 2 div 2 div"

DEMO_SEARCHES: tuple[SearchDeclaration, ...] = (
    SearchDeclaration(
        name="real_negative_new_derivation_of_1_16",
        fragment=DEMO_FRAGMENT,
        target=DEMO_REACHABLE_TARGET,
        target_kind="rational",
        objective="a derivation of the target that is not already catalogued",
        known_derivations=(DEMO_KNOWN_DERIVATION,),
    ),
    SearchDeclaration(
        name="uninformative_sqrt2_is_irrational",
        fragment=DEMO_FRAGMENT,
        target="sqrt2",
        target_kind="symbolic",
        objective="any program of the fragment whose exact value is the target",
    ),
    SearchDeclaration(
        name="uninformative_1_9973_exceeds_the_height_ladder",
        fragment=DEMO_FRAGMENT,
        target="1/9973",
        target_kind="rational",
        objective="any program of the fragment whose exact value is the target",
    ),
    SearchDeclaration(
        name="uninformative_355_113_outside_the_exact_image",
        fragment=DEMO_FRAGMENT,
        target="355/113",
        target_kind="rational",
        objective="any program of the fragment whose exact value is the target",
    ),
    SearchDeclaration(
        name="uninformative_catalan_rationality_is_open",
        fragment=DEMO_FRAGMENT,
        target="catalan",
        target_kind="symbolic",
        objective="any program of the fragment whose exact value is the target",
    ),
    SearchDeclaration(
        name="positive_new_derivation_of_1_16_uncatalogued",
        fragment=DEMO_FRAGMENT,
        target=DEMO_REACHABLE_TARGET,
        target_kind="rational",
        objective="a derivation of the target that is not already catalogued",
    ),
)

EXPECTED_DEMO_OUTCOMES: dict[str, str] = {
    "real_negative_new_derivation_of_1_16": OUTCOME_REAL_NEGATIVE,
    "uninformative_sqrt2_is_irrational": OUTCOME_UNINFORMATIVE,
    "uninformative_1_9973_exceeds_the_height_ladder": OUTCOME_UNINFORMATIVE,
    "uninformative_355_113_outside_the_exact_image": OUTCOME_UNINFORMATIVE,
    "uninformative_catalan_rationality_is_open": OUTCOME_UNINFORMATIVE,
    "positive_new_derivation_of_1_16_uncatalogued": OUTCOME_POSITIVE,
}


def run_demo_searches() -> list[dict[str, Any]]:
    """Run every declared demonstration search and return its sealed adjudication."""

    return [certified_search(declaration) for declaration in DEMO_SEARCHES]


# ---------------------------------------------------------------------------
# Controls -- a positive is worth nothing without a forgery that must fail
# ---------------------------------------------------------------------------


def _reseal(body: Mapping[str, Any]) -> dict[str, Any]:
    return seal_certificate(dict(body))


def certified_null_search_controls() -> dict[str, Any]:
    """Honest adjudications must verify; forged ones must be rejected, every time.

    The headline probe is the exact lie C1 exists to prevent: a null over a fragment that
    provably cannot express its target, relabelled ``REAL_NEGATIVE`` and resealed so the seal
    itself is valid.  Only re-deriving the outcome from the certificate kills it, which is why
    the verifier re-derives rather than reads.
    """

    honest: list[dict[str, Any]] = []
    adjudications = {
        declaration.name: certified_search(declaration) for declaration in DEMO_SEARCHES
    }
    for name, adjudication in adjudications.items():
        expected = EXPECTED_DEMO_OUTCOMES[name]
        try:
            report = verify_adjudication(adjudication)
            accepted = bool(report["accepted"]) and adjudication["outcome"] == expected
            reason = None
        except CompositionalSearchError as error:
            accepted = False
            reason = str(error)
        honest.append(
            {
                "case": name,
                "outcome": adjudication["outcome"],
                "expected_outcome": expected,
                "uninformative_reason": adjudication.get("uninformative_reason"),
                "accepted": accepted,
                "reason": reason,
            }
        )

    real_negative = adjudications["real_negative_new_derivation_of_1_16"]
    sqrt2_null = adjudications["uninformative_sqrt2_is_irrational"]
    image_null = adjudications["uninformative_355_113_outside_the_exact_image"]
    height_null = adjudications["uninformative_1_9973_exceeds_the_height_ladder"]
    positive = adjudications["positive_new_derivation_of_1_16_uncatalogued"]

    probes: list[dict[str, Any]] = []

    def probe(name: str, build: Any) -> None:
        try:
            candidate = build()
        except CompositionalSearchError as error:
            probes.append({"probe": name, "rejected": True, "reason": f"build: {error}"})
            return
        try:
            verify_adjudication(candidate)
        except CompositionalSearchError as error:
            probes.append({"probe": name, "rejected": True, "reason": str(error)})
            return
        probes.append({"probe": name, "rejected": False, "reason": None})

    def _copy(value: Mapping[str, Any]) -> dict[str, Any]:
        return json.loads(json.dumps(value))

    def real_negative_for_a_target_outside_the_space() -> dict[str, Any]:
        candidate = _copy(sqrt2_null)
        candidate["outcome"] = OUTCOME_REAL_NEGATIVE
        candidate["uninformative_reason"] = None
        candidate["publishable_as_a_negative"] = True
        return _reseal(candidate)

    def uninformative_for_a_reachable_target() -> dict[str, Any]:
        candidate = _copy(real_negative)
        candidate["outcome"] = OUTCOME_UNINFORMATIVE
        candidate["uninformative_reason"] = REASON_OUTSIDE_SPACE
        candidate["publishable_as_a_negative"] = False
        return _reseal(candidate)

    def positive_claimed_with_no_accepted_witness() -> dict[str, Any]:
        candidate = _copy(real_negative)
        candidate["outcome"] = OUTCOME_POSITIVE
        candidate["uninformative_reason"] = None
        return _reseal(candidate)

    def certificate_swapped_for_a_different_target() -> dict[str, Any]:
        candidate = _copy(image_null)
        candidate["certificate"] = _copy(height_null["certificate"])
        return _reseal(candidate)

    def certificate_swapped_for_a_different_fragment() -> dict[str, Any]:
        wider = Fragment("tiny_neg_recip_div", "C", ("1", "2", "neg", "recip", "div"), "Q")
        candidate = _copy(image_null)
        candidate["certificate"] = rational_reachability_certificate("355/113", wider)
        return _reseal(candidate)

    def search_witness_count_inflated() -> dict[str, Any]:
        candidate = _copy(sqrt2_null)
        candidate["search"]["witness_count"] = 3
        candidate["search"]["accepted_count"] = 3
        candidate["outcome"] = OUTCOME_POSITIVE
        return _reseal(candidate)

    def coverage_argument_broken_but_verdict_kept() -> dict[str, Any]:
        candidate = _copy(real_negative)
        candidate["search"]["coverage"]["structurally_valid_measured"] = 1
        return _reseal(candidate)

    def known_derivation_quietly_dropped() -> dict[str, Any]:
        candidate = _copy(real_negative)
        candidate["search"]["declaration"]["known_derivations"] = []
        return _reseal(candidate)

    def certificate_seal_tampered() -> dict[str, Any]:
        candidate = _copy(real_negative)
        candidate["certificate"]["content_sha256"] = "0" * 64
        return _reseal(candidate)

    def unsealed_outcome_flip() -> dict[str, Any]:
        candidate = _copy(sqrt2_null)
        candidate["outcome"] = OUTCOME_REAL_NEGATIVE
        return candidate

    def positive_downgraded_to_a_publishable_negative() -> dict[str, Any]:
        candidate = _copy(positive)
        candidate["outcome"] = OUTCOME_REAL_NEGATIVE
        candidate["publishable_as_a_negative"] = True
        return _reseal(candidate)

    def unresolved_null_relabelled_real() -> dict[str, Any]:
        candidate = _copy(adjudications["uninformative_catalan_rationality_is_open"])
        candidate["outcome"] = OUTCOME_REAL_NEGATIVE
        candidate["uninformative_reason"] = None
        candidate["publishable_as_a_negative"] = True
        return _reseal(candidate)

    probe("real_negative_for_a_target_outside_the_space", real_negative_for_a_target_outside_the_space)
    probe("uninformative_for_a_reachable_target", uninformative_for_a_reachable_target)
    probe("positive_claimed_with_no_accepted_witness", positive_claimed_with_no_accepted_witness)
    probe("certificate_swapped_for_a_different_target", certificate_swapped_for_a_different_target)
    probe(
        "certificate_swapped_for_a_different_fragment",
        certificate_swapped_for_a_different_fragment,
    )
    probe("search_witness_count_inflated", search_witness_count_inflated)
    probe("coverage_argument_broken_but_verdict_kept", coverage_argument_broken_but_verdict_kept)
    probe("known_derivation_quietly_dropped", known_derivation_quietly_dropped)
    probe("certificate_seal_tampered", certificate_seal_tampered)
    probe("unsealed_outcome_flip", unsealed_outcome_flip)
    probe(
        "positive_downgraded_to_a_publishable_negative",
        positive_downgraded_to_a_publishable_negative,
    )
    probe("unresolved_null_relabelled_real", unresolved_null_relabelled_real)

    return {
        "honest": honest,
        "probes": probes,
        "all_honest_accepted": all(row["accepted"] for row in honest),
        "all_probes_rejected": all(row["rejected"] for row in probes),
        "honest_count": len(honest),
        "probe_count": len(probes),
    }


# ---------------------------------------------------------------------------
# Receipt and CLI
# ---------------------------------------------------------------------------


def build_receipt(findings: Mapping[str, Mapping[str, int]] | None = None) -> dict[str, Any]:
    """A sealed receipt: the demonstration searches, the campaign block, and the controls."""

    empty: dict[str, Mapping[str, int]] = {"C": {}, "F": {}}
    adjudications = run_demo_searches()
    block = campaign_reachability_block(empty if findings is None else findings)
    controls = certified_null_search_controls()
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "falsifier_closed": (
            "C1: a null result on this search path is now reported as REAL_NEGATIVE or as "
            "UNINFORMATIVE_NULL, and the label is derived from a verified reachability "
            "certificate rather than asserted"
        ),
        "demonstration_searches": adjudications,
        "demonstration_outcomes": {
            str(item["search"]["declaration"]["name"]): str(item["outcome"])
            for item in adjudications
        },
        "campaign_reachability": block,
        "controls": controls,
        "decision": (
            "CERTIFIED"
            if controls["all_honest_accepted"] and controls["all_probes_rejected"]
            else "CONTROLS_FAILED"
        ),
    }
    return seal_certificate(body)


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Seal, schema, control and outcome checks for a receipt this module wrote."""

    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise CompositionalSearchError("receipt rejected: schema changed")
    payload = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if canonical_sha256(payload) != receipt.get("content_sha256"):
        raise CompositionalSearchError("receipt rejected: seal does not match the body")
    controls = receipt.get("controls", {})
    if not (controls.get("all_honest_accepted") and controls.get("all_probes_rejected")):
        raise CompositionalSearchError("receipt rejected: controls are not green")
    for adjudication in receipt.get("demonstration_searches", []):
        verify_adjudication(adjudication)
    verify_campaign_reachability_block(receipt.get("campaign_reachability", {}))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument("--output", default=None, help="write the sealed receipt here")
    parser.add_argument(
        "--summary",
        action="store_true",
        help="print only the outcome table rather than the whole receipt",
    )
    arguments = parser.parse_args(argv)
    receipt = build_receipt()
    if arguments.summary:
        text = json.dumps(
            {
                "decision": receipt["decision"],
                "demonstration_outcomes": receipt["demonstration_outcomes"],
                "campaign_summary": receipt["campaign_reachability"]["summary"],
                "controls": {
                    "all_honest_accepted": receipt["controls"]["all_honest_accepted"],
                    "all_probes_rejected": receipt["controls"]["all_probes_rejected"],
                    "probe_count": receipt["controls"]["probe_count"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    else:
        text = json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False)
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8", newline="\n")
    else:
        print(text)
    return 0 if receipt["decision"] == "CERTIFIED" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
