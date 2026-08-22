"""Reachability certificates for the compositional expression search grammar.

:mod:`sigma_theory_compiler.compositional_expression_search` enumerates 1.911e12 programs
and is very good at saying no.  What it cannot say is *why* it said no.  When an exhaustive
sweep returns nothing, two completely different facts produce the same silence:

* the object does not exist, or
* the object exists but the grammar cannot spell it.

This module supplies the missing artefact.  Given the declared grammar and a target, it emits
a **certificate** whose verdict is one of

``REACHABLE``
    with the explicit derivation: the token program, its ordinal in the very space the GPU
    sweep enumerates, the codec round trip, and a step-by-step exact stack trace ending on the
    target.  A reader replays the trace; nothing is taken on trust.

``OUTSIDE_FRAGMENT``
    with a structural or counting argument -- never with "we looked and did not find it".
    Four arguments are implemented, each independently checkable:

    ``value_cap``
        ``|t|`` exceeds the grammar's declared ``VALUE_CAP``.  Every evaluator in the search
        (CUDA kernel, numpy reference, mpmath) kills such a program, so no program in *any*
        fragment of *either* mode can produce ``t``.  This is the only argument here that
        covers the full alphabet.
    ``field_closure``
        the fragment's tokens all carry an exact ``Q -> Q`` transfer function, so every value
        it produces is rational; a target proved irrational is therefore outside it.  The
        tokens *not* in the fragment carry checked escape witnesses (``sqrt: 2 |-> sqrt 2``,
        ``sin: 1 |-> sin 1``, ...) so "this fragment is rational" is a verified fact rather
        than a naming convention.
    ``height_bound``
        an exact integer ladder ``B_1..B_L`` bounding the naive height of any value a
        ``n``-node expression can hold.  ``height(t) > B_L`` puts ``t`` outside.  O(1) to
        check, and it extends to node budgets where enumeration is impossible.
    ``exact_image_exhaustion``
        the complete image of the fragment is *computed*, exactly, in ``Fraction``
        arithmetic, and the target is not in it.  This is a closed set, not a search log.

``UNRESOLVED``
    the honest third answer.  No argument fired and no derivation was found; the certificate
    says so and claims nothing.

**Why the node-budget image is exact, not an over-approximation.**  A valid program of length
``L`` is exactly an expression tree of ``L`` nodes in postorder, but only if the declared
``STACK_DEPTH`` never bites.  It does not, and that is proved rather than assumed:  validity
forces ``#terminals = #binary + 1``, so ``L = 2*#binary + #unary + 1`` and ``#terminals <=
(L + 1) / 2`` -- five for mode C, four for mode F, both under ``STACK_DEPTH = 6``.  The stack
never rises above the terminal count, so the depth cap is vacuous.  The module checks this
mechanically by counting depth-free trees and requiring the count to equal the search's own
depth-capped :func:`count_valid_programs`: 32,971,249,179 for mode C and 2,337,344,190 for
mode F, exactly.  Because of that, ``t not in A_L`` is a theorem about the fragment and not a
statement about how hard anybody looked.

**Scope, stated plainly.**  ``OUTSIDE_FRAGMENT`` means outside the *declared fragment*, and
the certificate names it.  Deciding membership in the full alphabet -- which mixes ``exp``,
``ln``, ``sin`` and rationals -- is the constant problem, and Richardson's theorem says it is
undecidable in general.  So the exact-image lane runs over the rational fragment, where the
question is decidable and is decided, and the full alphabet gets the ``value_cap`` argument
plus a symbolic lane that proves individual derivations exactly.

**The symbolic lane** upgrades the existing ``reachability_controls`` rediscovery check from
"agrees to 1e-40" to an exact sympy identity proof, and it is where the module earns its
sharpest control.  The search's own docstring records a family of impostors:
``2 atan(exp(exp(exp(3/2))))`` reproduces pi to 38 digits because ``atan`` saturates.  That
program is in the space.  A certificate claiming it derives pi is *refuted* here, exactly --
sympy decides ``2 atan(y) < pi`` for finite ``y`` -- and the verdict is ``PROGRAM_REFUTED``,
which refutes the path and says nothing about the grammar.

**Verification is two jobs, not one.**  :func:`verify_certificate` first re-derives every
claim from the certificate's own declarations -- it re-executes the program in exact
arithmetic, recomputes the image, recomputes the height ladder, replays the symbolic tactic --
which is what kills a certificate that is *wrong*.  It then rebuilds the whole body with this
module's own builders and requires an exact match, which is what kills a certificate that is
merely *misleading*: prose that overstates a scope, a tactic name that never fired, an
``argument`` block quoting a cap that is not the grammar's.  The tests walk several hundred
fields per certificate, mutating each one and resealing, and require every one of them to be
rejected.

Nothing on a certificate path is a float.  Terminals are ``Fraction``, the image is a set of
``Fraction``, the height ladder is exact integers, the symbolic lane is sympy, and the
grammar's ``VALUE_CAP`` is converted once to its exact rational value before any comparison.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from .compositional_expression_search import (
    MODE_CONFIG,
    STACK_DEPTH,
    SYMBOLIC_TARGETS,
    TERMINAL_COUNT,
    TERMINAL_FRACTIONS,
    TOKEN_ARITY,
    TOKEN_NAMES,
    VALUE_CAP,
    VARIABLE_INDEX,
    CompositionalSearchError,
    count_valid_programs,
    decode_ordinal,
    encode_program,
    mode_digits,
    program_status,
    render_infix,
    render_rpn,
    to_sympy,
)
from .sigma_core import canonical_sha256

CERTIFICATE_SCHEMA = "invariant-reachability-certificate-1.0"

#: The grammar declares its kill threshold as a float literal.  It is converted once, here, to
#: the exact rational the double denotes, so every comparison downstream is integer arithmetic.
VALUE_CAP_EXACT: Fraction = Fraction(VALUE_CAP)

VERDICT_REACHABLE = "REACHABLE"
VERDICT_OUTSIDE = "OUTSIDE_FRAGMENT"
VERDICT_UNRESOLVED = "UNRESOLVED"
VERDICT_PROGRAM_REFUTED = "PROGRAM_REFUTED"

#: Arguments the exact-rational lane may carry for an ``OUTSIDE_FRAGMENT`` verdict.  The
#: ``field_closure`` argument lives in the symbolic lane instead, because its target is not a
#: rational in the first place.
ARGUMENT_KINDS = (
    "value_cap",
    "height_bound",
    "exact_image_exhaustion",
)


# ---------------------------------------------------------------------------
# Exact transfer functions -- these *are* the proof that a fragment is Q-closed
# ---------------------------------------------------------------------------


def _q_recip(value: Fraction) -> Fraction | None:
    return None if value == 0 else 1 / value


def _q_div(left: Fraction, right: Fraction) -> Fraction | None:
    return None if right == 0 else left / right


#: Unary tokens with a total-or-explicitly-partial exact ``Q -> Q`` transfer function.
Q_UNARY: dict[str, Any] = {
    "neg": lambda value: -value,
    "recip": _q_recip,
    "sqr": lambda value: value * value,
}

#: Binary tokens with an exact ``Q x Q -> Q`` transfer function.
Q_BINARY: dict[str, Any] = {
    "add": lambda left, right: left + right,
    "sub": lambda left, right: left - right,
    "mul": lambda left, right: left * right,
    "div": _q_div,
}

#: Every token that keeps a rational value rational.  Terminals other than ``k`` qualify by
#: construction: :data:`TERMINAL_FRACTIONS` gives each one an exact ``Fraction``.
RATIONAL_TERMINAL_NAMES: tuple[str, ...] = tuple(
    TOKEN_NAMES[index] for index in range(TERMINAL_COUNT) if index != VARIABLE_INDEX
)
RATIONAL_TRANSFER_NAMES: frozenset[str] = frozenset(
    (*RATIONAL_TERMINAL_NAMES, *Q_UNARY, *Q_BINARY)
)

#: A rational input whose image under the token is provably irrational.  These make
#: "the fragment is rational" a checked claim: a fragment that admits any of these tokens is
#: rejected, and the rejection cites a specific number that leaves the field.
FIELD_ESCAPE_WITNESSES: dict[str, dict[str, Any]] = {
    "sqrt": {"inputs": ("2",), "image": "sqrt(2)"},
    "exp": {"inputs": ("1",), "image": "exp(1)"},
    "ln": {"inputs": ("2",), "image": "log(2)"},
    "sin": {"inputs": ("1",), "image": "sin(1)"},
    "cos": {"inputs": ("1",), "image": "cos(1)"},
    "atan": {"inputs": ("1",), "image": "atan(1)"},
    "pow": {"inputs": ("2", "1/2"), "image": "2**Rational(1,2)"},
}

_ESCAPE_EXPRESSIONS: dict[str, sp.Expr] = {
    "sqrt": sp.sqrt(2),
    "exp": sp.exp(1),
    "ln": sp.log(2),
    "sin": sp.sin(1),
    "cos": sp.cos(1),
    "atan": sp.atan(1),
    "pow": sp.Integer(2) ** sp.Rational(1, 2),
}


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fragment:
    """A declared sub-alphabet of one search mode, plus the value field it is claimed to keep.

    A fragment is not a hint or a heuristic: it is a subset of the token alphabet, so the set
    of programs it admits is a subset of the mode's programs and every statement proved about
    the fragment is a statement about a well-defined part of the real search space.
    """

    name: str
    mode: str
    token_names: tuple[str, ...]
    value_field: str  # "Q" (claimed rational) or "R" (no field claim)

    def __post_init__(self) -> None:
        if self.mode not in MODE_CONFIG:
            raise CompositionalSearchError(f"unknown mode: {self.mode}")
        if self.value_field not in {"Q", "R"}:
            raise CompositionalSearchError(f"unknown value field: {self.value_field}")
        allowed = {TOKEN_NAMES[index] for index in mode_digits(self.mode)}
        for name in self.token_names:
            if name not in allowed:
                raise CompositionalSearchError(
                    f"token {name!r} is outside the mode {self.mode} alphabet"
                )
        if len(set(self.token_names)) != len(self.token_names):
            raise CompositionalSearchError("fragment token list has duplicates")

    @property
    def program_length(self) -> int:
        return int(MODE_CONFIG[self.mode]["program_length"])

    @property
    def token_indices(self) -> tuple[int, ...]:
        return tuple(sorted(TOKEN_NAMES.index(name) for name in self.token_names))

    def split(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """``(terminals, unary, binary)`` names, each sorted by token index."""

        terminals: list[str] = []
        unary: list[str] = []
        binary: list[str] = []
        for index in self.token_indices:
            arity = TOKEN_ARITY[index]
            bucket = terminals if arity == 0 else (unary if arity == 1 else binary)
            bucket.append(TOKEN_NAMES[index])
        return tuple(terminals), tuple(unary), tuple(binary)


MODE_C_RATIONAL = Fragment(
    name="mode_c_rational",
    mode="C",
    token_names=(*RATIONAL_TERMINAL_NAMES, "neg", "recip", "sqr", "add", "sub", "mul", "div"),
    value_field="Q",
)

MODE_C_FULL = Fragment(
    name="mode_c_full",
    mode="C",
    token_names=tuple(TOKEN_NAMES[index] for index in mode_digits("C")),
    value_field="R",
)

BUILTIN_FRAGMENTS: dict[str, Fragment] = {
    MODE_C_RATIONAL.name: MODE_C_RATIONAL,
    MODE_C_FULL.name: MODE_C_FULL,
}


def fragment_from_declaration(declaration: Mapping[str, Any]) -> Fragment:
    """Rebuild a :class:`Fragment` from a certificate's own declaration, trusting nothing."""

    try:
        return Fragment(
            name=str(declaration["name"]),
            mode=str(declaration["mode"]),
            token_names=tuple(str(item) for item in declaration["token_names"]),
            value_field=str(declaration["value_field"]),
        )
    except KeyError as error:  # pragma: no cover - defensive
        raise CompositionalSearchError(f"fragment declaration missing {error}") from error


# ---------------------------------------------------------------------------
# Field closure: is this fragment really rational?
# ---------------------------------------------------------------------------


def field_closure_report(fragment: Fragment) -> dict[str, Any]:
    """Check the fragment's field claim mechanically, with escape witnesses for what it drops.

    A token belongs to the rational fragment iff this module holds an exact ``Fraction``
    transfer function for it.  That is not a tautology: the exact image below is computed by
    those very functions, so a token without one cannot be given an exact image at all.  For
    every token the fragment *excludes*, the report evaluates a declared rational input and
    asks sympy whether the image is rational; a ``False`` there is a proof that the token
    genuinely leaves the field.
    """

    present = [TOKEN_NAMES[index] for index in fragment.token_indices]
    with_transfer = [name for name in present if name in RATIONAL_TRANSFER_NAMES]
    without_transfer = [name for name in present if name not in RATIONAL_TRANSFER_NAMES]

    witnesses: list[dict[str, Any]] = []
    for name, item in FIELD_ESCAPE_WITNESSES.items():
        expression = _ESCAPE_EXPRESSIONS[name]
        is_rational = expression.is_rational
        witnesses.append(
            {
                "token": name,
                "rational_inputs": list(item["inputs"]),
                "image": str(item["image"]),
                "sympy_is_rational": None if is_rational is None else bool(is_rational),
                "proves_escape": is_rational is False,
                "in_fragment": name in present,
            }
        )
    witnesses.sort(key=lambda row: str(row["token"]))

    closed = not without_transfer
    escaping_in_fragment = sorted(
        str(row["token"]) for row in witnesses if row["in_fragment"] and row["proves_escape"]
    )
    return {
        "declared_field": fragment.value_field,
        "tokens_with_exact_rational_transfer": with_transfer,
        "tokens_without_exact_rational_transfer": without_transfer,
        "escape_witnesses": witnesses,
        "escaping_tokens_in_fragment": escaping_in_fragment,
        "rationally_closed": closed,
        "declaration_supported": (fragment.value_field != "Q") or closed,
    }


# ---------------------------------------------------------------------------
# Structural profile: why a node budget is the whole story
# ---------------------------------------------------------------------------


def token_profiles(fragment: Fragment) -> tuple[int, int, int]:
    terminals, unary, binary = fragment.split()
    return len(terminals), len(unary), len(binary)


def tree_counts_depth_free(fragment: Fragment) -> tuple[int, ...]:
    """Number of ``n``-node expression trees over the fragment, ignoring the stack cap."""

    terminals, unary, binary = token_profiles(fragment)
    length = fragment.program_length
    counts = [0] * (length + 1)
    if length >= 1:
        counts[1] = terminals
    for nodes in range(2, length + 1):
        pairs = sum(counts[left] * counts[nodes - 1 - left] for left in range(1, nodes - 1))
        counts[nodes] = unary * counts[nodes - 1] + binary * pairs
    return tuple(counts[1:])


def tree_counts_depth_capped(fragment: Fragment) -> tuple[int, ...]:
    """The same count with the declared :data:`STACK_DEPTH` enforced during RPN evaluation."""

    terminals, unary, binary = token_profiles(fragment)
    length = fragment.program_length
    results: list[int] = []
    for nodes in range(1, length + 1):
        state: dict[int, int] = {0: 1}
        for _ in range(nodes):
            nxt: dict[int, int] = {}
            for depth, count in state.items():
                if depth + 1 <= STACK_DEPTH and terminals:
                    nxt[depth + 1] = nxt.get(depth + 1, 0) + count * terminals
                if depth >= 1 and unary:
                    nxt[depth] = nxt.get(depth, 0) + count * unary
                if depth >= 2 and binary:
                    nxt[depth - 1] = nxt.get(depth - 1, 0) + count * binary
            state = nxt
        results.append(state.get(1, 0))
    return tuple(results)


def structural_profile(fragment: Fragment) -> dict[str, Any]:
    """The exact arity bookkeeping that makes ``L`` tokens the same thing as ``L`` tree nodes."""

    length = fragment.program_length
    shapes: list[dict[str, int]] = []
    for binary in range(length // 2 + 1):
        unary = length - 2 * binary - 1
        if unary < 0:
            continue
        shapes.append({"terminals": binary + 1, "unary": unary, "binary": binary})
    max_terminals = max(shape["terminals"] for shape in shapes)

    depth_free = tree_counts_depth_free(fragment)
    depth_capped = tree_counts_depth_capped(fragment)
    profile: dict[str, Any] = {
        "mode": fragment.mode,
        "program_length": length,
        "declared_stack_depth": STACK_DEPTH,
        "arity_identity": "terminals == binary + 1, so length == 2*binary + unary + 1",
        "unary_parity": (length - 1) % 2,
        "admissible_shapes": shapes,
        "max_terminals": max_terminals,
        "stack_depth_upper_bound": max_terminals,
        "stack_depth_constraint_binds": max_terminals > STACK_DEPTH,
        "tree_counts_depth_free": list(depth_free),
        "tree_counts_depth_capped": list(depth_capped),
        "depth_lemma_holds": depth_free == depth_capped,
    }
    if fragment.token_indices == tuple(sorted(mode_digits(fragment.mode))):
        declared = count_valid_programs(fragment.mode)
        profile["search_module_structurally_valid"] = int(declared["structurally_valid"])
        profile["matches_search_module_count"] = (
            depth_free[-1] == int(declared["structurally_valid"])
        )
    return profile


# ---------------------------------------------------------------------------
# The exact image of a rational fragment
# ---------------------------------------------------------------------------

_IMAGE_CACHE: dict[tuple[str, tuple[int, ...], int], tuple[frozenset[Fraction], ...]] = {}


def _image_key(fragment: Fragment) -> tuple[str, tuple[int, ...], int]:
    # The image is a function of the alphabet and the budget; the fragment's *name* is a label
    # and must not split the cache.
    return (fragment.mode, fragment.token_indices, fragment.program_length)


def exact_rational_image(fragment: Fragment) -> tuple[frozenset[Fraction], ...]:
    """``(A_1, ..., A_L)`` where ``A_n`` is the exact set of values of ``n``-node programs.

    Computed by upward closure in ``Fraction`` arithmetic.  Every element is a value some
    program in the fragment really takes, and every value some program takes is an element,
    because the stack cap is vacuous (:func:`structural_profile`) and the operator set is
    exactly the fragment's.  Membership in ``A_L`` is therefore *decidable* reachability, and
    non-membership is a theorem rather than a search log.

    The grammar's ``VALUE_CAP`` is applied at every level, against its exact rational value.
    That is what makes the closure the image and not merely a superset of it: the search kills
    a program the moment an intermediate breaches the cap, so an intermediate dropped here is
    an intermediate the machine would have dropped too.
    """

    report = field_closure_report(fragment)
    if not report["rationally_closed"]:
        raise CompositionalSearchError(
            "exact rational image needs a rationally closed fragment; "
            f"{fragment.name} carries {report['tokens_without_exact_rational_transfer']}"
        )
    key = _image_key(fragment)
    cached = _IMAGE_CACHE.get(key)
    if cached is not None:
        return cached

    terminal_names, unary_names, binary_names = fragment.split()
    cap = VALUE_CAP_EXACT
    base = {value for value in map(Fraction, terminal_names) if abs(value) <= cap}
    levels: list[frozenset[Fraction]] = [frozenset(base)]
    unary_ops = [Q_UNARY[name] for name in unary_names]
    binary_ops = [Q_BINARY[name] for name in binary_names]

    for nodes in range(2, fragment.program_length + 1):
        current: set[Fraction] = set()
        for value in levels[nodes - 2]:
            for operation in unary_ops:
                image = operation(value)
                if image is not None and abs(image) <= cap:
                    current.add(image)
        for left_nodes in range(1, nodes - 1):
            right_nodes = nodes - 1 - left_nodes
            if right_nodes < 1 or left_nodes > right_nodes:
                continue
            left_set = levels[left_nodes - 1]
            right_set = levels[right_nodes - 1]
            mirror = left_nodes != right_nodes
            for left in left_set:
                for right in right_set:
                    for operation in binary_ops:
                        image = operation(left, right)
                        if image is not None and abs(image) <= cap:
                            current.add(image)
                        if mirror:
                            mirrored = operation(right, left)
                            if mirrored is not None and abs(mirrored) <= cap:
                                current.add(mirrored)
        levels.append(frozenset(current))

    result = tuple(levels)
    _IMAGE_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# The height ladder: an O(1) counting argument that survives past enumeration
# ---------------------------------------------------------------------------


def height(value: Fraction) -> int:
    """Naive height ``max(|numerator|, denominator)`` of a rational in lowest terms."""

    return max(abs(value.numerator), value.denominator)


def height_bound_ladder(fragment: Fragment) -> tuple[int, ...]:
    """Exact integer bounds ``B_1..B_L`` with ``height(v) <= B_n`` for every ``v`` in ``A_n``.

    Per-operator bounds, all elementary:  ``neg`` and ``recip`` preserve height; ``sqr``
    squares it; ``mul`` and ``div`` on ``p/q`` and ``r/s`` give ``pr/qs`` so height multiplies;
    ``add`` and ``sub`` give ``(ps +- qr)/(qs)`` whose numerator is at most ``2*H1*H2``.  The
    ladder is a maximum over the fragment's own operators, so it is a bound for that fragment
    and not a generic one.  The recurrence is tight before the grammar's value cap is applied
    -- ``sqr`` applied eight times to the terminal ``5`` attains ``B_9 = 5**256`` exactly -- so
    nothing is being given away by loose algebra; the cap then removes that particular witness
    and leaves the bound conservative, which is the safe direction for an exclusion argument.
    """

    terminal_names, unary_names, binary_names = fragment.split()
    if not terminal_names:
        raise CompositionalSearchError("fragment has no terminals")
    bounds = [max(height(Fraction(name)) for name in terminal_names)]
    for nodes in range(2, fragment.program_length + 1):
        candidates: list[int] = []
        previous = bounds[nodes - 2]
        for name in unary_names:
            if name in {"neg", "recip"}:
                candidates.append(previous)
            elif name == "sqr":
                candidates.append(previous * previous)
            else:  # pragma: no cover - guarded by field_closure_report
                raise CompositionalSearchError(f"no height rule for unary {name!r}")
        for left_nodes in range(1, nodes - 1):
            right_nodes = nodes - 1 - left_nodes
            if right_nodes < 1:
                continue
            product = bounds[left_nodes - 1] * bounds[right_nodes - 1]
            for name in binary_names:
                if name in {"add", "sub"}:
                    candidates.append(2 * product)
                elif name in {"mul", "div"}:
                    candidates.append(product)
                else:  # pragma: no cover - guarded by field_closure_report
                    raise CompositionalSearchError(f"no height rule for binary {name!r}")
        bounds.append(max(candidates) if candidates else previous)
    return tuple(bounds)


# ---------------------------------------------------------------------------
# Derivation reconstruction
# ---------------------------------------------------------------------------

_Tree = tuple[Any, ...]


def _reconstruct(
    target: Fraction,
    nodes: int,
    levels: Sequence[frozenset[Fraction]],
    fragment: Fragment,
) -> _Tree | None:
    """Backward search for an ``nodes``-node tree over the fragment whose value is ``target``.

    Every candidate sub-value is required to live in the exact image at its own node count, so
    the recursion cannot wander outside the fragment and cannot fail on a value that is really
    there.  Iteration order is sorted, so the derivation a certificate carries is a function of
    the target and the fragment alone.
    """

    if target not in levels[nodes - 1]:
        return None
    terminal_names, unary_names, binary_names = fragment.split()
    if nodes == 1:
        for name in terminal_names:
            if Fraction(name) == target:
                return ("terminal", name)
        return None

    for name in unary_names:
        source: Fraction | None
        if name == "neg":
            source = -target
        elif name == "recip":
            source = None if target == 0 else 1 / target
        elif name == "sqr":
            source = _rational_sqrt(target)
        else:  # pragma: no cover - guarded by field_closure_report
            source = None
        if source is None or source not in levels[nodes - 2]:
            continue
        # Apply the operator forward before trusting the inverse: the partner solves below are
        # exact algebra, but a certificate is not the place to take algebra on faith.
        if Q_UNARY[name](source) != target:
            continue
        child = _reconstruct(source, nodes - 1, levels, fragment)
        if child is not None:
            return ("unary", name, child)

    for left_nodes in range(1, nodes - 1):
        right_nodes = nodes - 1 - left_nodes
        if right_nodes < 1:
            continue
        left_set = levels[left_nodes - 1]
        right_set = levels[right_nodes - 1]
        # Walk whichever side is smaller and solve for its partner: the operators are all
        # invertible in one argument, so this is a lookup rather than a second search.
        walk_left = len(left_set) <= len(right_set)
        walked = sorted(left_set if walk_left else right_set)
        for name in binary_names:
            for known in walked:
                if walk_left:
                    left = known
                    right = _binary_right_partner(name, left, target)
                else:
                    right = known
                    left = _binary_left_partner(name, right, target)
                if left is None or right is None:
                    continue
                if left not in left_set or right not in right_set:
                    continue
                if Q_BINARY[name](left, right) != target:
                    continue
                left_tree = _reconstruct(left, left_nodes, levels, fragment)
                if left_tree is None:
                    continue
                right_tree = _reconstruct(right, right_nodes, levels, fragment)
                if right_tree is None:
                    continue
                return ("binary", name, left_tree, right_tree)
    return None


def _binary_right_partner(name: str, left: Fraction, target: Fraction) -> Fraction | None:
    """The unique ``right`` with ``op(left, right) == target``, or ``None`` when there is none."""

    if name == "add":
        return target - left
    if name == "sub":
        return left - target
    if name == "mul":
        return None if left == 0 else target / left
    if name == "div":
        # ``left == 0`` would solve to ``right == 0`` and divide by zero, so it is refused
        # here rather than caught downstream.
        return None if (target == 0 or left == 0) else left / target
    return None  # pragma: no cover - guarded by field_closure_report


def _binary_left_partner(name: str, right: Fraction, target: Fraction) -> Fraction | None:
    """The unique ``left`` with ``op(left, right) == target``, or ``None`` when there is none."""

    if name == "add":
        return target - right
    if name == "sub":
        return target + right
    if name == "mul":
        return None if right == 0 else target / right
    if name == "div":
        return None if right == 0 else target * right
    return None  # pragma: no cover - guarded by field_closure_report


def _rational_sqrt(value: Fraction) -> Fraction | None:
    """The non-negative rational square root of ``value``, or ``None``."""

    if value < 0:
        return None
    numerator = _exact_isqrt(value.numerator)
    denominator = _exact_isqrt(value.denominator)
    if numerator is None or denominator is None:
        return None
    return Fraction(numerator, denominator)


def _exact_isqrt(number: int) -> int | None:
    if number < 0:
        return None
    root = _integer_sqrt(number)
    return root if root * root == number else None


def _integer_sqrt(number: int) -> int:
    if number < 2:
        return number
    guess = 1 << ((number.bit_length() + 1) // 2)
    while True:
        better = (guess + number // guess) // 2
        if better >= guess:
            return guess
        guess = better


def _tree_tokens(tree: _Tree) -> tuple[int, ...]:
    """Postorder token indices: exactly the RPN program the stack machine runs."""

    kind = tree[0]
    if kind == "terminal":
        return (TOKEN_NAMES.index(str(tree[1])),)
    if kind == "unary":
        return (*_tree_tokens(tree[2]), TOKEN_NAMES.index(str(tree[1])))
    return (
        *_tree_tokens(tree[2]),
        *_tree_tokens(tree[3]),
        TOKEN_NAMES.index(str(tree[1])),
    )


def evaluate_exact(tokens: Sequence[int]) -> tuple[Fraction | None, list[dict[str, Any]]]:
    """Run the stack machine in exact ``Fraction`` arithmetic and record every step.

    Returns ``(value, trace)``.  ``value`` is ``None`` when the program underflows, leaves a
    residue, divides by zero, needs a token with no exact rational transfer, or breaches the
    grammar's declared value cap -- the same kill rules the search's own evaluator applies,
    with the cap compared against its exact rational value rather than a float.
    """

    stack: list[Fraction] = []
    trace: list[dict[str, Any]] = []
    for step, token in enumerate(tokens):
        name = TOKEN_NAMES[token]
        arity = TOKEN_ARITY[token]
        if arity == 0:
            if token == VARIABLE_INDEX or token >= TERMINAL_COUNT:
                return None, trace
            value = TERMINAL_FRACTIONS[token]
        elif arity == 1:
            if name not in Q_UNARY or not stack:
                return None, trace
            value = Q_UNARY[name](stack.pop())
        else:
            if name not in Q_BINARY or len(stack) < 2:
                return None, trace
            right = stack.pop()
            left = stack.pop()
            value = Q_BINARY[name](left, right)
        if value is None:
            return None, trace
        if abs(value) > VALUE_CAP_EXACT:
            return None, trace
        stack.append(value)
        trace.append(
            {
                "step": step,
                "token": name,
                "arity": arity,
                "stack": [str(item) for item in stack],
            }
        )
    if len(stack) != 1:
        return None, trace
    return stack[0], trace


def max_stack_depth(tokens: Sequence[int]) -> int:
    depth = 0
    peak = 0
    for token in tokens:
        arity = TOKEN_ARITY[token]
        depth += 1 if arity == 0 else 1 - arity
        peak = max(peak, depth)
    return peak


# ---------------------------------------------------------------------------
# Certificate construction
# ---------------------------------------------------------------------------


def parse_exact_rational(value: Any) -> Fraction:
    """Exact parse.  Floats are refused outright: a certificate path carries no doubles."""

    if isinstance(value, bool):
        raise CompositionalSearchError("target must be a rational, not a bool")
    if isinstance(value, float):
        raise CompositionalSearchError(
            "float targets are refused; pass an exact string such as '355/113'"
        )
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, str):
        try:
            return Fraction(value)
        except ValueError as error:
            raise CompositionalSearchError(f"cannot parse exact rational {value!r}") from error
    raise CompositionalSearchError(f"cannot parse exact rational from {type(value).__name__}")


def _seal(body: Mapping[str, Any]) -> dict[str, Any]:
    payload = {key: value for key, value in body.items() if key != "content_sha256"}
    return {**payload, "content_sha256": canonical_sha256(payload)}


def _fragment_declaration(fragment: Fragment) -> dict[str, Any]:
    terminals, unary, binary = fragment.split()
    return {
        "name": fragment.name,
        "mode": fragment.mode,
        "value_field": fragment.value_field,
        "token_names": [TOKEN_NAMES[index] for index in fragment.token_indices],
        "terminals": list(terminals),
        "unary": list(unary),
        "binary": list(binary),
        "program_length": fragment.program_length,
        "declared_stack_depth": STACK_DEPTH,
    }


#: Public names for the two blocks a consumer must be able to rebuild identically.  A module
#: that adjudicates a search against a certificate has to bind the certificate to the fragment
#: the search really enumerated, and it has to seal its own record the same way; re-deriving
#: either of those elsewhere is how two supposedly identical canonical forms drift apart.
fragment_declaration = _fragment_declaration
seal_certificate = _seal


def _target_block(value: Fraction) -> dict[str, Any]:
    return {
        "kind": "rational",
        "exact": str(value),
        "numerator": value.numerator,
        "denominator": value.denominator,
        "height": height(value),
    }


def _derivation(tokens: Sequence[int], fragment: Fragment, target: Fraction) -> dict[str, Any]:
    tokens = tuple(tokens)
    status = program_status(tokens)
    ordinal = encode_program(tokens, fragment.mode) if status == "ok" else None
    roundtrip = ordinal is not None and decode_ordinal(ordinal, fragment.mode) == tokens
    value, trace = evaluate_exact(tokens)
    terminals = sum(1 for token in tokens if TOKEN_ARITY[token] == 0)
    unary = sum(1 for token in tokens if TOKEN_ARITY[token] == 1)
    binary = sum(1 for token in tokens if TOKEN_ARITY[token] == 2)
    return {
        "tokens": list(tokens),
        "rpn": render_rpn(tokens),
        "infix": render_infix(tokens),
        "program_status": status,
        "node_count": len(tokens),
        "terminal_count": terminals,
        "unary_count": unary,
        "binary_count": binary,
        "max_stack_depth": max_stack_depth(tokens),
        "ordinal": ordinal,
        "ordinal_space_size": int(MODE_CONFIG[fragment.mode]["space_size"]),
        "codec_roundtrip": roundtrip,
        "exact_value": None if value is None else str(value),
        "equals_target": value is not None and value == target,
        "trace": trace,
    }


def rational_reachability_certificate(
    target: Any,
    fragment: Fragment = MODE_C_RATIONAL,
    *,
    exhaustive: bool = True,
) -> dict[str, Any]:
    """Decide reachability of an exact rational in a rationally closed fragment.

    With ``exhaustive=True`` this is a decision procedure: the answer is ``REACHABLE`` with a
    replayable derivation or ``OUTSIDE_FRAGMENT`` with a completed image.  With
    ``exhaustive=False`` only the O(1) arguments run and an undecided target comes back
    ``UNRESOLVED`` rather than being guessed at.
    """

    value = parse_exact_rational(target)
    closure = field_closure_report(fragment)
    if not closure["declaration_supported"]:
        raise CompositionalSearchError(
            f"fragment {fragment.name} declares field {fragment.value_field} but carries "
            f"{closure['tokens_without_exact_rational_transfer']}"
        )
    if fragment.value_field != "Q":
        raise CompositionalSearchError(
            "the exact-rational lane needs a Q fragment; use the symbolic lane for mode_c_full"
        )

    profile = structural_profile(fragment)
    ladder = height_bound_ladder(fragment)
    body: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "lane": "exact_rational",
        "fragment": _fragment_declaration(fragment),
        "field_closure": closure,
        "structural_profile": profile,
        "target": _target_block(value),
        "value_cap_exact": str(VALUE_CAP_EXACT),
        "height_bound_ladder": [str(item) for item in ladder],
    }

    if abs(value) > VALUE_CAP_EXACT:
        body["verdict"] = VERDICT_OUTSIDE
        body["argument"] = {
            "kind": "value_cap",
            "scope": "every fragment of every mode, including the full alphabet",
            "statement": (
                "the grammar kills any intermediate whose magnitude exceeds VALUE_CAP, so no "
                "program in the declared space can hold this value"
            ),
            "target_abs": str(abs(value)),
            "value_cap_exact": str(VALUE_CAP_EXACT),
        }
        return _seal(body)

    if height(value) > ladder[-1]:
        body["verdict"] = VERDICT_OUTSIDE
        body["argument"] = {
            "kind": "height_bound",
            "scope": f"fragment {fragment.name}",
            "statement": (
                "an exact integer ladder bounds the height of every value an n-node program "
                "in this fragment can hold; the target exceeds the bound at the full budget"
            ),
            "node_budget": fragment.program_length,
            "bound": str(ladder[-1]),
            "target_height": height(value),
        }
        return _seal(body)

    if not exhaustive:
        body["verdict"] = VERDICT_UNRESOLVED
        body["argument"] = {
            "kind": "none",
            "statement": (
                "no O(1) argument fired and the exact image was not computed; this certificate "
                "claims nothing about reachability"
            ),
        }
        return _seal(body)

    levels = exact_rational_image(fragment)
    sizes = [len(level) for level in levels]
    body["exact_image"] = {
        "level_sizes": sizes,
        "budget": fragment.program_length,
        "recurrence": (
            "A_1 = terminals; A_n = {u(x) : x in A_{n-1}} union "
            "{b(x, y) : x in A_i, y in A_j, i + j = n - 1}"
        ),
        "exactness_ground": (
            "the declared stack cap is vacuous at this program length, so n-token programs and "
            "n-node trees are the same objects and A_L is the whole image"
        ),
        "depth_lemma_holds": profile["depth_lemma_holds"],
    }

    if value in levels[-1]:
        tree = _reconstruct(value, fragment.program_length, levels, fragment)
        if tree is None:  # pragma: no cover - contradicts the image definition
            raise CompositionalSearchError(
                "target is in the exact image but no derivation was reconstructed"
            )
        tokens = _tree_tokens(tree)
        derivation = _derivation(tokens, fragment, value)
        if not derivation["equals_target"] or derivation["program_status"] != "ok":
            raise CompositionalSearchError(
                "reconstructed derivation failed its own exact re-execution"
            )
        body["verdict"] = VERDICT_REACHABLE
        body["derivation"] = derivation
        return _seal(body)

    body["verdict"] = VERDICT_OUTSIDE
    body["argument"] = {
        "kind": "exact_image_exhaustion",
        "scope": f"fragment {fragment.name}",
        "statement": (
            "the complete image of the fragment at the declared program length was computed in "
            "exact rational arithmetic and the target is not an element; this is a closed set, "
            "not a search that stopped"
        ),
        "node_budget": fragment.program_length,
        "image_size_at_budget": sizes[-1],
        "level_sizes": sizes,
        "target_height": height(value),
        "height_bound_at_budget": str(ladder[-1]),
        "height_argument_would_suffice": height(value) > ladder[-1],
    }
    return _seal(body)


# ---------------------------------------------------------------------------
# Symbolic lane: exact proofs about one derivation over the full alphabet
# ---------------------------------------------------------------------------


def _tactic_simplify(gap: sp.Expr) -> bool:
    return bool(sp.simplify(gap) == 0)


def _tactic_expand_simplify(gap: sp.Expr) -> bool:
    return bool(sp.simplify(sp.expand(gap)) == 0)


def _tactic_rewrite_log(gap: sp.Expr) -> bool:
    return bool(sp.simplify(gap.rewrite(sp.log)) == 0)


def _tactic_tangent_with_bound(left: sp.Expr, right: sp.Expr) -> bool:
    """``tan(L) == tan(R)`` and ``|L - R| < pi/2`` force ``L == R``: tan has period pi.

    Both tangents must be finite for the argument to mean anything, so that is checked before
    the conclusion is drawn.  This is what proves the Machin-like identity
    ``pi = 4 (arctan 1/2 + arctan 1/3)``, which no amount of ``simplify`` will close.
    """

    left_tan = sp.expand_trig(sp.tan(left))
    right_tan = sp.expand_trig(sp.tan(right))
    if left_tan.is_finite is not True or right_tan.is_finite is not True:
        return False
    if sp.simplify(left_tan - right_tan) != 0:
        return False
    return (sp.Abs(left - right) < sp.pi / 2) == True


def prove_symbolic_equal(expression: sp.Expr, target: sp.Expr) -> dict[str, Any]:
    """Try an ordered ladder of exact tactics; report which one closed the gap, or none."""

    gap = sp.together(expression - target)
    attempts: list[dict[str, Any]] = []
    for name, tactic in (
        ("simplify", _tactic_simplify),
        ("expand_simplify", _tactic_expand_simplify),
        ("rewrite_log_simplify", _tactic_rewrite_log),
    ):
        try:
            proved = tactic(gap)
        except (TypeError, ValueError, NotImplementedError, AttributeError) as error:
            attempts.append({"tactic": name, "proved": False, "error": type(error).__name__})
            continue
        attempts.append({"tactic": name, "proved": proved})
        if proved:
            return {"proved": True, "tactic": name, "attempts": attempts}
    try:
        proved = _tactic_tangent_with_bound(expression, target)
    except (TypeError, ValueError, NotImplementedError, AttributeError) as error:
        attempts.append(
            {"tactic": "tangent_with_bound", "proved": False, "error": type(error).__name__}
        )
    else:
        attempts.append({"tactic": "tangent_with_bound", "proved": proved})
        if proved:
            return {"proved": True, "tactic": "tangent_with_bound", "attempts": attempts}
    return {"proved": False, "tactic": None, "attempts": attempts}


def prove_symbolic_distinct(expression: sp.Expr, target: sp.Expr) -> dict[str, Any]:
    """Exact strict-inequality refutation of one derivation, when sympy can decide one."""

    for name, relation in (("strictly_below", expression < target), ("strictly_above", expression > target)):
        try:
            decided = relation == True
        except TypeError:  # pragma: no cover - defensive
            decided = False
        if decided:
            return {"refuted": True, "relation": name}
    return {"refuted": False, "relation": None}


def tokens_from_rpn(text: str, fragment: Fragment) -> tuple[int, ...]:
    index = {name: position for position, name in enumerate(TOKEN_NAMES)}
    allowed = set(fragment.token_names)
    tokens: list[int] = []
    for name in text.split():
        if name not in index:
            raise CompositionalSearchError(f"unknown token {name!r}")
        if name not in allowed:
            raise CompositionalSearchError(
                f"token {name!r} is outside fragment {fragment.name}"
            )
        tokens.append(index[name])
    return tuple(tokens)


def _symbolic_program_block(tokens: Sequence[int], fragment: Fragment) -> dict[str, Any]:
    """Everything a symbolic certificate says about its program, derived from the tokens alone."""

    tokens = tuple(tokens)
    status = program_status(tokens)
    length_ok = len(tokens) == fragment.program_length
    ordinal = encode_program(tokens, fragment.mode) if (status == "ok" and length_ok) else None
    roundtrip = ordinal is not None and decode_ordinal(ordinal, fragment.mode) == tokens
    expression = to_sympy(tokens) if status == "ok" else None
    return {
        "rpn": render_rpn(tokens),
        "infix": render_infix(tokens),
        "tokens": list(tokens),
        "program_status": status,
        "length_matches_mode": length_ok,
        "ordinal": ordinal,
        "codec_roundtrip": roundtrip,
        "sympy": None if expression is None else sp.srepr(expression),
        "printed": None if expression is None else str(expression),
    }


def symbolic_reachability_certificate(
    rpn: str,
    target_name: str,
    fragment: Fragment = MODE_C_FULL,
) -> dict[str, Any]:
    """Certify -- or refute -- one explicit derivation of a named classical constant.

    ``REACHABLE`` here means the program was proved *exactly equal* to the target, not that it
    agrees to some digit count.  ``PROGRAM_REFUTED`` means this derivation was proved unequal;
    it says nothing about the grammar, and the certificate says so in as many words.
    """

    if target_name not in SYMBOLIC_TARGETS:
        raise CompositionalSearchError(f"unknown symbolic target {target_name!r}")
    tokens = tokens_from_rpn(rpn, fragment)
    program = _symbolic_program_block(tokens, fragment)
    expression = to_sympy(tokens) if program["program_status"] == "ok" else None
    target = SYMBOLIC_TARGETS[target_name]

    body: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "lane": "symbolic",
        "fragment": _fragment_declaration(fragment),
        "target": {"kind": "symbolic", "name": target_name, "sympy": sp.srepr(target)},
        "program": program,
    }

    if expression is None or not program["codec_roundtrip"]:
        body["verdict"] = VERDICT_UNRESOLVED
        body["argument"] = {
            "kind": "none",
            "statement": "program is not a well-formed member of the declared space",
        }
        return _seal(body)

    equality = prove_symbolic_equal(expression, target)
    if equality["proved"]:
        body["verdict"] = VERDICT_REACHABLE
        body["proof"] = {
            "kind": "symbolic_identity",
            "tactic": equality["tactic"],
            "attempts": equality["attempts"],
            "statement": "the program is exactly equal to the target, proved symbolically",
        }
        return _seal(body)

    refutation = prove_symbolic_distinct(expression, target)
    if refutation["refuted"]:
        body["verdict"] = VERDICT_PROGRAM_REFUTED
        body["proof"] = {
            "kind": "strict_inequality",
            "relation": refutation["relation"],
            "attempts": equality["attempts"],
            "scope": "this derivation only",
            "statement": (
                "the program is proved strictly unequal to the target, so this derivation is "
                "refuted; the grammar may still reach the target by another program"
            ),
        }
        return _seal(body)

    body["verdict"] = VERDICT_UNRESOLVED
    body["proof"] = {
        "kind": "none",
        "attempts": equality["attempts"],
        "statement": "neither equality nor strict inequality was proved; nothing is claimed",
    }
    return _seal(body)


def irrational_exclusion_certificate(
    target_name: str,
    fragment: Fragment = MODE_C_RATIONAL,
) -> dict[str, Any]:
    """Exclude an irrational constant from a rationally closed fragment, structurally.

    No enumeration happens here and none is needed.  Every token of the fragment carries an
    exact ``Q -> Q`` transfer function, so every program it admits evaluates into ``Q``; a
    target sympy proves irrational cannot be in the image of a map whose codomain is ``Q``.
    That is a one-line argument, and it is exactly the kind of statement an exhaustive sweep
    can never make about itself: ``sqrt2`` is absent from this fragment because of what the
    fragment *is*, not because the sweep came back empty.
    """

    if target_name not in SYMBOLIC_TARGETS:
        raise CompositionalSearchError(f"unknown symbolic target {target_name!r}")
    closure = field_closure_report(fragment)
    if fragment.value_field != "Q" or not closure["rationally_closed"]:
        raise CompositionalSearchError(
            f"field_closure exclusion needs a rationally closed fragment; {fragment.name} "
            f"carries {closure['tokens_without_exact_rational_transfer']}"
        )
    target = SYMBOLIC_TARGETS[target_name]
    is_rational = target.is_rational
    body: dict[str, Any] = {
        "schema_version": CERTIFICATE_SCHEMA,
        "lane": "symbolic",
        "fragment": _fragment_declaration(fragment),
        "field_closure": closure,
        "target": {
            "kind": "symbolic",
            "name": target_name,
            "sympy": sp.srepr(target),
            "sympy_is_rational": None if is_rational is None else bool(is_rational),
        },
    }
    if is_rational is False:
        body["verdict"] = VERDICT_OUTSIDE
        body["proof"] = {
            "kind": "field_closure",
            "scope": f"fragment {fragment.name}",
            "statement": (
                "every token of the fragment has an exact Q -> Q transfer function, so the "
                "image of the fragment lies in Q; the target is irrational and therefore "
                "outside it, by the shape of the fragment and not by a failed search"
            ),
            "image_field": "Q",
            "escape_witnesses": closure["escape_witnesses"],
        }
        return _seal(body)
    body["verdict"] = VERDICT_UNRESOLVED
    body["proof"] = {
        "kind": "none",
        "statement": (
            "the target is not known to be irrational, so field closure decides nothing here"
        ),
    }
    return _seal(body)


# ---------------------------------------------------------------------------
# Verification -- trusts nothing on the certificate, recomputes everything
# ---------------------------------------------------------------------------


def _reject(reason: str) -> None:
    raise CompositionalSearchError(f"certificate rejected: {reason}")


def _regenerate(certificate: Mapping[str, Any], fragment: Fragment) -> dict[str, Any]:
    """Rebuild the certificate from its own declarations, using the module's own builders.

    This is the second half of verification and it does a different job from the first.  The
    substantive checks re-derive the *claims* -- they re-execute the program, recompute the
    image, replay the tactic -- and catch a certificate that is wrong.  This catches a
    certificate that is merely misleading: prose that overstates the scope, a tactic name that
    never fired, an ``argument`` field quoting a cap that is not the grammar's.  Those cannot
    make a false statement pass, but a reader reads them, so they are pinned too.
    """

    lane = certificate.get("lane")
    if lane == "exact_rational":
        return rational_reachability_certificate(
            str(certificate["target"]["exact"]),
            fragment,
            exhaustive="exact_image" in certificate,
        )
    name = str(certificate["target"]["name"])
    program = certificate.get("program")
    if program is None:
        return irrational_exclusion_certificate(name, fragment)
    return symbolic_reachability_certificate(str(program["rpn"]), name, fragment)


def verify_certificate(certificate: Mapping[str, Any]) -> dict[str, Any]:
    """Re-derive every claim from the certificate's own declarations.

    Raises :class:`CompositionalSearchError` on the first failure.  The verifier shares no
    state with the generator: it re-parses the fragment, re-checks the field closure, re-runs
    the stack machine in exact arithmetic, recomputes the height ladder, and recomputes the
    exact image when an exhaustion argument is claimed.
    """

    if certificate.get("schema_version") != CERTIFICATE_SCHEMA:
        _reject(f"unknown schema_version {certificate.get('schema_version')!r}")
    if "content_sha256" not in certificate:
        _reject("unsealed certificate")
    payload = {key: value for key, value in certificate.items() if key != "content_sha256"}
    if canonical_sha256(payload) != certificate["content_sha256"]:
        _reject("seal does not match the body")

    fragment = fragment_from_declaration(certificate.get("fragment", {}))
    # Only ``name``, ``mode``, ``token_names`` and ``value_field`` are read back into the
    # Fragment; everything else in the block is derived, so the whole block must equal a fresh
    # build or a reader could be shown an alphabet the proof never used.
    if dict(certificate.get("fragment", {})) != _fragment_declaration(fragment):
        _reject("declared fragment block does not match a recomputation from its token list")
    # A verdict is scoped to a fragment, so a fragment's *name* must not be able to drift away
    # from its token set: a shrunken alphabet still wearing a trusted label would make a true
    # statement read as a much stronger one.
    builtin = BUILTIN_FRAGMENTS.get(fragment.name)
    if builtin is not None and fragment != builtin:
        _reject(
            f"fragment named {fragment.name!r} does not match the built-in fragment of that "
            "name"
        )

    closure = field_closure_report(fragment)
    if not closure["declaration_supported"]:
        _reject(
            f"fragment claims field {fragment.value_field} but admits "
            f"{closure['tokens_without_exact_rational_transfer']}; "
            f"escape witnesses: {closure['escaping_tokens_in_fragment']}"
        )

    lane = certificate.get("lane")
    if lane == "exact_rational":
        report = _verify_exact_rational(certificate, fragment, closure)
    elif lane == "symbolic":
        report = _verify_symbolic(certificate, fragment)
    else:
        _reject(f"unknown lane {lane!r}")
        raise AssertionError("unreachable")  # pragma: no cover

    # Substantive checks have passed; now pin the presentation.  Run last so that a certificate
    # making a false *claim* dies on the claim, with a message that names it, rather than on a
    # generic mismatch.
    try:
        rebuilt = _regenerate(certificate, fragment)
    except CompositionalSearchError as error:
        _reject(f"certificate cannot be regenerated from its own declarations: {error}")
    if rebuilt["content_sha256"] != certificate["content_sha256"]:
        differing = sorted(
            key
            for key in set(rebuilt) | set(certificate)
            if key != "content_sha256" and rebuilt.get(key) != certificate.get(key)
        )
        _reject(f"body does not match a regeneration from its own declarations: {differing}")
    report["checks"]["regenerated"] = True
    return report


def _verify_exact_rational(
    certificate: Mapping[str, Any],
    fragment: Fragment,
    closure: Mapping[str, Any],
) -> dict[str, Any]:
    if fragment.value_field != "Q":
        _reject("the exact-rational lane requires a Q fragment")
    target_block = certificate.get("target", {})
    if target_block.get("kind") != "rational":
        _reject("exact-rational lane needs a rational target")
    value = parse_exact_rational(str(target_block.get("exact")))
    # Recompute the whole block rather than spot-checking one field: a numerator that
    # disagrees with ``exact`` misleads a reader even when every proof step is sound.
    if dict(target_block) != _target_block(value):
        _reject("declared target block does not match a recomputation from its exact value")
    if certificate.get("field_closure") != dict(closure):
        _reject("declared field closure report does not match a recomputation")
    if certificate.get("value_cap_exact") != str(VALUE_CAP_EXACT):
        _reject("declared value cap does not match the grammar's own")

    profile = structural_profile(fragment)
    if not profile["depth_lemma_holds"]:
        _reject("depth lemma fails for this fragment; the node budget is not the program length")
    if certificate.get("structural_profile") != profile:
        _reject("declared structural profile does not match a recomputation")
    ladder = height_bound_ladder(fragment)
    declared_ladder = [str(item) for item in ladder]
    if list(certificate.get("height_bound_ladder", [])) != declared_ladder:
        _reject("height ladder does not match a recomputation")

    verdict = certificate.get("verdict")
    checks: dict[str, Any] = {
        "seal": True,
        "field_closure": bool(closure["declaration_supported"]),
        "depth_lemma": True,
        "height_ladder": True,
    }

    if verdict == VERDICT_REACHABLE:
        derivation = certificate.get("derivation")
        if not isinstance(derivation, Mapping):
            _reject("REACHABLE without a derivation")
        tokens = tuple(int(item) for item in derivation.get("tokens", ()))
        if not tokens:
            _reject("derivation carries no tokens")
        allowed = set(fragment.token_indices)
        outside = sorted({TOKEN_NAMES[token] for token in tokens if token not in allowed})
        if outside:
            _reject(f"derivation uses tokens outside the fragment: {outside}")
        if len(tokens) != fragment.program_length:
            _reject(
                f"derivation has {len(tokens)} tokens, mode {fragment.mode} requires "
                f"{fragment.program_length}"
            )
        status = program_status(tokens)
        if status != "ok":
            _reject(f"derivation is not a valid program: {status}")
        ordinal = encode_program(tokens, fragment.mode)
        if derivation.get("ordinal") != ordinal:
            _reject("declared ordinal is not the encoding of the declared tokens")
        if decode_ordinal(ordinal, fragment.mode) != tokens:
            _reject("ordinal does not decode back to the declared tokens")
        if render_rpn(tokens) != derivation.get("rpn"):
            _reject("declared rpn does not render from the declared tokens")
        recomputed, trace = evaluate_exact(tokens)
        if recomputed is None:
            _reject("derivation does not evaluate under the exact stack machine")
        if recomputed != value:
            _reject(
                f"derivation evaluates to {recomputed}, not to the declared target {value}"
            )
        if list(derivation.get("trace", [])) != trace:
            _reject("declared trace does not match an exact re-execution")
        if max_stack_depth(tokens) > STACK_DEPTH:
            _reject("derivation exceeds the declared stack depth")
        # Everything above is re-derived from the tokens alone, so the whole block must equal a
        # fresh build: that covers the rendered infix and the arity counts a reader might read
        # instead of the RPN.
        if dict(derivation) != _derivation(tokens, fragment, value):
            _reject("declared derivation block does not match a recomputation from its tokens")
        checks.update(
            {
                "tokens_in_fragment": True,
                "program_valid": True,
                "codec_roundtrip": True,
                "trace_replayed": True,
                "value_equals_target": True,
                "ordinal": ordinal,
            }
        )
        return {"accepted": True, "verdict": verdict, "checks": checks}

    if verdict == VERDICT_OUTSIDE:
        argument = certificate.get("argument")
        if not isinstance(argument, Mapping):
            _reject("OUTSIDE_FRAGMENT without an argument")
        kind = argument.get("kind")
        if kind not in ARGUMENT_KINDS:
            _reject(f"unknown argument kind {kind!r}")
        if kind == "value_cap":
            if not abs(value) > VALUE_CAP_EXACT:
                _reject(
                    f"value_cap argument does not hold: |{value}| does not exceed the cap"
                )
            checks["value_cap"] = True
        elif kind == "height_bound":
            if height(value) <= ladder[-1]:
                _reject(
                    f"height_bound argument does not hold: height {height(value)} is within "
                    f"the ladder bound {ladder[-1]}"
                )
            if str(argument.get("bound")) != str(ladder[-1]):
                _reject("declared height bound does not match a recomputation")
            checks["height_bound"] = True
        else:
            levels = exact_rational_image(fragment)
            sizes = [len(level) for level in levels]
            if list(argument.get("level_sizes", [])) != sizes:
                _reject("declared exact-image level sizes do not match a recomputation")
            if list(certificate.get("exact_image", {}).get("level_sizes", [])) != sizes:
                _reject("the certificate's exact_image block disagrees with its own argument")
            if certificate.get("exact_image", {}).get("depth_lemma_holds") is not True:
                _reject("exhaustion argument requires a certificate that asserts the depth lemma")
            if argument.get("image_size_at_budget") != sizes[-1]:
                _reject("declared image size at the budget does not match a recomputation")
            if value in levels[-1]:
                _reject(
                    "exhaustion argument does not hold: the target IS in the exact image "
                    f"at {fragment.program_length} nodes"
                )
            checks["exact_image_recomputed"] = True
            checks["image_size_at_budget"] = sizes[-1]
        return {"accepted": True, "verdict": verdict, "checks": checks}

    if verdict == VERDICT_UNRESOLVED:
        if "derivation" in certificate:
            _reject("UNRESOLVED must not carry a derivation")
        argument = certificate.get("argument", {})
        if isinstance(argument, Mapping) and argument.get("kind") not in {None, "none"}:
            _reject("UNRESOLVED must not carry a substantive argument")
        return {"accepted": True, "verdict": verdict, "checks": checks}

    _reject(f"unknown verdict {verdict!r}")
    raise AssertionError("unreachable")  # pragma: no cover


def _verify_symbolic(certificate: Mapping[str, Any], fragment: Fragment) -> dict[str, Any]:
    if "field_closure" in certificate and certificate["field_closure"] != dict(
        field_closure_report(fragment)
    ):
        _reject("declared field closure report does not match a recomputation")
    target_block = certificate.get("target", {})
    if target_block.get("kind") != "symbolic":
        _reject("symbolic lane needs a symbolic target")
    name = str(target_block.get("name"))
    if name not in SYMBOLIC_TARGETS:
        _reject(f"unknown symbolic target {name!r}")
    target = SYMBOLIC_TARGETS[name]
    if sp.srepr(target) != target_block.get("sympy"):
        _reject("declared target srepr does not match the search module's target")

    verdict = certificate.get("verdict")

    if verdict == VERDICT_OUTSIDE:
        proof = certificate.get("proof", {})
        if proof.get("kind") != "field_closure":
            _reject("OUTSIDE_FRAGMENT in the symbolic lane needs a field_closure proof")
        if fragment.value_field != "Q":
            _reject("field_closure exclusion requires a Q fragment")
        closure = field_closure_report(fragment)
        if not closure["rationally_closed"]:
            _reject(
                "field_closure exclusion requires every token to have an exact Q -> Q "
                f"transfer; {closure['tokens_without_exact_rational_transfer']} do not"
            )
        if target.is_rational is not False:
            _reject(
                f"field_closure exclusion needs a target proved irrational; sympy says "
                f"is_rational={target.is_rational!r} for {name}"
            )
        return {
            "accepted": True,
            "verdict": verdict,
            "checks": {
                "seal": True,
                "fragment_rationally_closed": True,
                "target_proved_irrational": True,
            },
        }

    if verdict == VERDICT_UNRESOLVED:
        if certificate.get("proof", {}).get("kind") not in {None, "none"}:
            _reject("UNRESOLVED must not carry a substantive proof")
        return {"accepted": True, "verdict": verdict, "checks": {"seal": True}}

    program = certificate.get("program", {})
    tokens = tuple(int(item) for item in program.get("tokens", ()))
    if not tokens:
        _reject("symbolic certificate carries no tokens")
    allowed = set(fragment.token_indices)
    outside = sorted({TOKEN_NAMES[token] for token in tokens if token not in allowed})
    if outside:
        _reject(f"program uses tokens outside the fragment: {outside}")
    if dict(program) != _symbolic_program_block(tokens, fragment):
        _reject("declared program block does not match a recomputation from its tokens")

    checks: dict[str, Any] = {"seal": True, "tokens_in_fragment": True}

    if verdict in {VERDICT_REACHABLE, VERDICT_PROGRAM_REFUTED}:
        if len(tokens) != fragment.program_length:
            _reject("program length does not match the mode")
        if program_status(tokens) != "ok":
            _reject("program is not structurally valid")
        ordinal = encode_program(tokens, fragment.mode)
        if program.get("ordinal") != ordinal:
            _reject("declared ordinal is not the encoding of the declared tokens")
        if decode_ordinal(ordinal, fragment.mode) != tokens:
            _reject("ordinal does not decode back to the declared tokens")
        expression = to_sympy(tokens)
        if expression is None:
            _reject("program has no symbolic form")
        if sp.srepr(expression) != program.get("sympy"):
            _reject("declared symbolic form does not match a recomputation")
        checks["codec_roundtrip"] = True

        proof = certificate.get("proof", {})
        if verdict == VERDICT_REACHABLE:
            if proof.get("kind") != "symbolic_identity":
                _reject("REACHABLE in the symbolic lane needs a symbolic_identity proof")
            replay = prove_symbolic_equal(expression, target)
            if not replay["proved"]:
                _reject(
                    "claimed symbolic identity does not replay: no exact tactic closes the gap"
                )
            checks["symbolic_identity_replayed"] = replay["tactic"]
        else:
            if proof.get("kind") != "strict_inequality":
                _reject("PROGRAM_REFUTED needs a strict_inequality proof")
            replay = prove_symbolic_distinct(expression, target)
            if not replay["refuted"]:
                _reject("claimed strict inequality does not replay")
            checks["strict_inequality_replayed"] = replay["relation"]
        return {"accepted": True, "verdict": verdict, "checks": checks}

    _reject(f"unknown verdict {verdict!r}")
    raise AssertionError("unreachable")  # pragma: no cover


# ---------------------------------------------------------------------------
# Controls -- the positives are worth nothing without these
# ---------------------------------------------------------------------------

#: A rational the mode C rational fragment provably cannot hold.  9973 is prime and no
#: nine-node expression over the fragment produces it or its reciprocal.
UNREACHABLE_PROBE = "1/9973"
#: Reachable, and a Diophantine approximation to pi rather than a token: the derivation is real.
REACHABLE_PROBE = "355/113"
#: Height beyond the ladder bound 5**256 but magnitude far under the value cap, so the height
#: argument fires on its own rather than being masked by the cap.
HEIGHT_PROBE = f"1/{5 ** 257}"
#: Magnitude past the declared value cap.
VALUE_CAP_PROBE = str(10**200)
#: The impostor from the search module's own docstring: agrees with pi to 38 digits, is not pi.
PI_IMPOSTOR_RPN = "3/2 exp exp exp atan 2 mul 1 mul"
#: pi = 4 arctan 1, the Gregory-Leibniz / Machin base point, exactly provable.
PI_HONEST_RPN = "1 atan 2 sqr mul 1 mul 1 mul"


def _reseal(certificate: Mapping[str, Any]) -> dict[str, Any]:
    return _seal(dict(certificate))


def reachability_certificate_controls() -> dict[str, Any]:
    """Honest certificates must be accepted; forged ones must be rejected, every time.

    The headline probe is the one the capability exists for: a certificate claiming
    ``REACHABLE`` for ``1/9973``, a target this module has *proved* is outside the fragment.
    It is built from a real, valid, correctly-encoded program, so nothing about its shape is
    wrong -- only its claim.  The verifier re-executes the program in exact arithmetic and the
    claim dies on the arithmetic, which is the only place a claim like that can be killed.
    """

    honest: list[dict[str, Any]] = []
    reachable = rational_reachability_certificate(REACHABLE_PROBE)
    unreachable = rational_reachability_certificate(UNREACHABLE_PROBE)
    by_height = rational_reachability_certificate(HEIGHT_PROBE)
    by_cap = rational_reachability_certificate(VALUE_CAP_PROBE)
    honest_pi = symbolic_reachability_certificate(PI_HONEST_RPN, "pi")
    impostor = symbolic_reachability_certificate(PI_IMPOSTOR_RPN, "pi")
    sqrt2_excluded = irrational_exclusion_certificate("sqrt2")
    catalan_open = irrational_exclusion_certificate("catalan")

    for label, certificate, expected in (
        ("reachable_355_113", reachable, VERDICT_REACHABLE),
        ("outside_1_9973_by_exhaustion", unreachable, VERDICT_OUTSIDE),
        ("outside_by_height_bound", by_height, VERDICT_OUTSIDE),
        ("outside_by_value_cap", by_cap, VERDICT_OUTSIDE),
        ("symbolic_pi_reachable", honest_pi, VERDICT_REACHABLE),
        ("symbolic_pi_impostor_refuted", impostor, VERDICT_PROGRAM_REFUTED),
        ("sqrt2_outside_rational_fragment", sqrt2_excluded, VERDICT_OUTSIDE),
        ("catalan_irrationality_open_so_unresolved", catalan_open, VERDICT_UNRESOLVED),
    ):
        try:
            report = verify_certificate(certificate)
            accepted = bool(report["accepted"]) and certificate["verdict"] == expected
            reason = None
        except CompositionalSearchError as error:
            accepted = False
            reason = str(error)
        honest.append(
            {
                "case": label,
                "verdict": certificate.get("verdict"),
                "expected_verdict": expected,
                "accepted": accepted,
                "reason": reason,
            }
        )

    probes: list[dict[str, Any]] = []

    def probe(name: str, build: Any) -> None:
        try:
            candidate = build()
        except CompositionalSearchError as error:
            probes.append({"probe": name, "rejected": True, "reason": f"build: {error}"})
            return
        try:
            verify_certificate(candidate)
        except CompositionalSearchError as error:
            probes.append({"probe": name, "rejected": True, "reason": str(error)})
            return
        probes.append({"probe": name, "rejected": False, "reason": None})

    def forged_reachability_for_unreachable_target() -> dict[str, Any]:
        candidate = json.loads(json.dumps(reachable))
        outside = parse_exact_rational(UNREACHABLE_PROBE)
        candidate["target"] = {
            "kind": "rational",
            "exact": str(outside),
            "numerator": outside.numerator,
            "denominator": outside.denominator,
            "height": height(outside),
        }
        return _reseal(candidate)

    def forged_reachability_doctored_trace() -> dict[str, Any]:
        candidate = json.loads(json.dumps(reachable))
        candidate["derivation"]["trace"][-1]["stack"] = [UNREACHABLE_PROBE]
        candidate["derivation"]["exact_value"] = UNREACHABLE_PROBE
        return _reseal(candidate)

    def forged_reachability_wrong_ordinal() -> dict[str, Any]:
        candidate = json.loads(json.dumps(reachable))
        candidate["derivation"]["ordinal"] = int(candidate["derivation"]["ordinal"]) + 1
        return _reseal(candidate)

    def forged_reachability_out_of_alphabet() -> dict[str, Any]:
        candidate = json.loads(json.dumps(reachable))
        tokens = list(candidate["derivation"]["tokens"])
        tokens[0] = TOKEN_NAMES.index("sqrt")
        candidate["derivation"]["tokens"] = tokens
        candidate["derivation"]["rpn"] = render_rpn(tokens)
        return _reseal(candidate)

    def forged_unreachability_for_reachable_target() -> dict[str, Any]:
        candidate = json.loads(json.dumps(unreachable))
        target = parse_exact_rational(REACHABLE_PROBE)
        candidate["target"] = {
            "kind": "rational",
            "exact": str(target),
            "numerator": target.numerator,
            "denominator": target.denominator,
            "height": height(target),
        }
        return _reseal(candidate)

    def forged_height_argument() -> dict[str, Any]:
        candidate = json.loads(json.dumps(by_height))
        target = parse_exact_rational(REACHABLE_PROBE)
        candidate["target"] = {
            "kind": "rational",
            "exact": str(target),
            "numerator": target.numerator,
            "denominator": target.denominator,
            "height": height(target),
        }
        return _reseal(candidate)

    def forged_value_cap_argument() -> dict[str, Any]:
        candidate = json.loads(json.dumps(by_cap))
        target = parse_exact_rational(REACHABLE_PROBE)
        candidate["target"] = {
            "kind": "rational",
            "exact": str(target),
            "numerator": target.numerator,
            "denominator": target.denominator,
            "height": height(target),
        }
        return _reseal(candidate)

    def _add_sqrt(candidate: dict[str, Any]) -> dict[str, Any]:
        # Renamed on purpose: the point of this probe is the *field closure* gate, so it must
        # not be short-circuited by the fragment-name gate.
        candidate["fragment"]["name"] = "mode_c_rational_plus_sqrt"
        candidate["fragment"]["token_names"] = sorted(
            {*candidate["fragment"]["token_names"], "sqrt"}
        )
        candidate["fragment"]["unary"] = sorted({*candidate["fragment"]["unary"], "sqrt"})
        return candidate

    def forged_field_closure_declaration() -> dict[str, Any]:
        return _reseal(_add_sqrt(json.loads(json.dumps(unreachable))))

    def forged_image_size() -> dict[str, Any]:
        candidate = json.loads(json.dumps(unreachable))
        sizes = list(candidate["argument"]["level_sizes"])
        sizes[-1] = int(sizes[-1]) + 1
        candidate["argument"]["level_sizes"] = sizes
        return _reseal(candidate)

    def unsealed_verdict_flip() -> dict[str, Any]:
        candidate = json.loads(json.dumps(unreachable))
        candidate["verdict"] = VERDICT_REACHABLE
        return candidate  # deliberately not resealed

    def forged_symbolic_identity() -> dict[str, Any]:
        candidate = json.loads(json.dumps(impostor))
        candidate["verdict"] = VERDICT_REACHABLE
        candidate["proof"] = {
            "kind": "symbolic_identity",
            "tactic": "simplify",
            "attempts": [],
            "statement": "forged",
        }
        return _reseal(candidate)

    def forged_fragment_name_drift() -> dict[str, Any]:
        """A true statement about three tokens, wearing the label of the real fragment."""

        candidate = json.loads(json.dumps(unreachable))
        candidate["fragment"]["token_names"] = ["1", "2", "add"]
        candidate["fragment"]["terminals"] = ["1", "2"]
        candidate["fragment"]["unary"] = []
        candidate["fragment"]["binary"] = ["add"]
        return _reseal(candidate)

    def forged_structural_profile() -> dict[str, Any]:
        candidate = json.loads(json.dumps(unreachable))
        candidate["structural_profile"]["stack_depth_constraint_binds"] = True
        return _reseal(candidate)

    def forged_exclusion_of_an_open_constant() -> dict[str, Any]:
        candidate = json.loads(json.dumps(catalan_open))
        candidate["verdict"] = VERDICT_OUTSIDE
        candidate["proof"] = {
            "kind": "field_closure",
            "scope": "fragment mode_c_rational",
            "statement": "forged",
            "image_field": "Q",
            "escape_witnesses": [],
        }
        return _reseal(candidate)

    def forged_exclusion_with_a_leaky_fragment() -> dict[str, Any]:
        return _reseal(_add_sqrt(json.loads(json.dumps(sqrt2_excluded))))

    probe("forged_reachability_for_unreachable_target", forged_reachability_for_unreachable_target)
    probe("forged_reachability_doctored_trace", forged_reachability_doctored_trace)
    probe("forged_reachability_wrong_ordinal", forged_reachability_wrong_ordinal)
    probe("forged_reachability_out_of_alphabet", forged_reachability_out_of_alphabet)
    probe("forged_unreachability_for_reachable_target", forged_unreachability_for_reachable_target)
    probe("forged_height_argument", forged_height_argument)
    probe("forged_value_cap_argument", forged_value_cap_argument)
    probe("forged_field_closure_declaration", forged_field_closure_declaration)
    probe("forged_exact_image_size", forged_image_size)
    probe("unsealed_verdict_flip", unsealed_verdict_flip)
    probe("forged_symbolic_identity_for_the_pi_impostor", forged_symbolic_identity)
    probe("forged_exclusion_of_an_open_constant", forged_exclusion_of_an_open_constant)
    probe("forged_exclusion_with_a_leaky_fragment", forged_exclusion_with_a_leaky_fragment)
    probe("forged_fragment_name_drift", forged_fragment_name_drift)
    probe("forged_structural_profile", forged_structural_profile)

    return {
        "honest": honest,
        "all_honest_accepted": all(row["accepted"] for row in honest),
        "probes": probes,
        "all_probes_rejected": all(row["rejected"] for row in probes),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_receipt(targets: Sequence[str]) -> dict[str, Any]:
    """A sealed receipt: one certificate per target, plus the control suite."""

    certificates = [rational_reachability_certificate(target) for target in targets]
    symbolic = [
        symbolic_reachability_certificate(PI_HONEST_RPN, "pi"),
        symbolic_reachability_certificate(PI_IMPOSTOR_RPN, "pi"),
    ]
    controls = reachability_certificate_controls()
    body = {
        "schema_version": CERTIFICATE_SCHEMA,
        "structural_profile": structural_profile(MODE_C_FULL),
        "rational_fragment": _fragment_declaration(MODE_C_RATIONAL),
        "certificates": certificates,
        "symbolic_certificates": symbolic,
        "controls": controls,
        "decision": (
            "CERTIFIED"
            if controls["all_honest_accepted"] and controls["all_probes_rejected"]
            else "CONTROLS_FAILED"
        ),
    }
    return _seal(body)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        help="exact rational target, e.g. 355/113 (repeatable)",
    )
    parser.add_argument("--output", default=None, help="write the sealed receipt here")
    arguments = parser.parse_args(argv)
    targets = arguments.target or [REACHABLE_PROBE, UNREACHABLE_PROBE, HEIGHT_PROBE, VALUE_CAP_PROBE]
    receipt = build_receipt(targets)
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
