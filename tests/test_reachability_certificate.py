"""Gates for reachability certificates.

The capability's whole claim is that an exhaustive negative can be told apart from a grammar
that could not spell the target.  That claim rests on four things being true, and these tests
pin all four.

*The node-budget model is the program model.*  A length-``L`` program is an ``L``-node tree
only if the declared stack cap never bites; the test recomputes the depth-free tree count and
requires it to equal the search module's own depth-capped ``count_valid_programs`` -- exactly,
32,971,249,179 for mode C and 2,337,344,190 for mode F.

*The exact image is the image.*  The dynamic program is checked against brute force on three
fragments small enough to enumerate completely: every token sequence -- 4^9 or 5^9 of them, at
the real mode C program length -- is generated, filtered by the search module's own
``program_status``, and evaluated in exact rational arithmetic.  The brute-force value set must
equal the closure set element for element.  If the closure over-approximated, an exhaustion
verdict would be a lie; this is the test that says it does not.  It is also the test that
caught a real bug: the backward solve for ``div`` admitted ``0 / 0``.

*A positive is replayable and a forgery is not.*  Every honest certificate is re-verified from
its own declarations, and fifteen forged ones -- headed by the certificate that claims
``REACHABLE`` for ``1/9973``, a target proved outside the fragment -- must be rejected.

*Nothing overclaims.*  Catalan's constant is not known to be irrational, so the field-closure
lane returns ``UNRESOLVED`` for it and a certificate that upgrades that to an exclusion is
refused.  The pi impostor from the search module's own docstring agrees with pi to 38 digits
and is refuted exactly.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest
import sympy as sp

from sigma_theory_compiler.compositional_expression_search import (
    BUILTIN_KNOWN_IDENTITIES,
    MODE_CONFIG,
    STACK_DEPTH,
    SYMBOLIC_TARGETS,
    TOKEN_NAMES,
    CompositionalSearchError,
    count_valid_programs,
    decode_ordinal,
    encode_program,
    program_status,
)
from sigma_theory_compiler.reachability_certificate import (
    ARGUMENT_KINDS,
    CERTIFICATE_SCHEMA,
    HEIGHT_PROBE,
    MODE_C_FULL,
    MODE_C_RATIONAL,
    PI_HONEST_RPN,
    PI_IMPOSTOR_RPN,
    REACHABLE_PROBE,
    UNREACHABLE_PROBE,
    VALUE_CAP_EXACT,
    VALUE_CAP_PROBE,
    VERDICT_OUTSIDE,
    VERDICT_PROGRAM_REFUTED,
    VERDICT_REACHABLE,
    VERDICT_UNRESOLVED,
    Fragment,
    build_receipt,
    evaluate_exact,
    exact_rational_image,
    field_closure_report,
    height,
    height_bound_ladder,
    irrational_exclusion_certificate,
    main,
    parse_exact_rational,
    rational_reachability_certificate,
    reachability_certificate_controls,
    structural_profile,
    symbolic_reachability_certificate,
    tree_counts_depth_capped,
    tree_counts_depth_free,
    verify_certificate,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

#: Level sizes of the exact image of the mode C rational fragment.  Pinned: a change here is a
#: change to every exhaustion verdict the module can emit, and must be deliberate.
EXPECTED_IMAGE_SIZES = [9, 25, 97, 378, 1446, 6352, 29574, 137586, 701726]

#: Small enough to enumerate every token sequence at the real mode C program length, and
#: between them structured enough to exercise both partial operators (``recip`` at zero and
#: ``div`` by zero), a non-commutative binary, and zero itself.
TINY_FRAGMENTS = (
    Fragment("tiny_recip_sub", "C", ("1", "2", "recip", "sub"), "Q"),
    Fragment("tiny_sub_div", "C", ("1", "2", "sub", "div"), "Q"),
    Fragment("tiny_neg_recip_div", "C", ("1", "2", "neg", "recip", "div"), "Q"),
)


@pytest.fixture(scope="module")
def image() -> tuple[frozenset[Fraction], ...]:
    return exact_rational_image(MODE_C_RATIONAL)


@pytest.fixture(scope="module")
def controls() -> dict:
    return reachability_certificate_controls()


# ---------------------------------------------------------------------------
# The structural lemma that makes a node budget equal to a program length
# ---------------------------------------------------------------------------


def test_depth_lemma_matches_the_search_modules_own_count() -> None:
    """Depth-free and depth-capped tree counts agree, and agree with the search module.

    This is the load-bearing lemma.  If the declared stack cap bit, the exact image would be a
    superset of the reachable set and an exhaustion verdict would prove nothing.
    """

    for mode in ("C", "F"):
        fragment = Fragment(
            name=f"mode_{mode.lower()}_full",
            mode=mode,
            token_names=tuple(
                TOKEN_NAMES[index] for index in MODE_CONFIG[mode]["digit_to_token"]
            ),
            value_field="R",
        )
        free = tree_counts_depth_free(fragment)
        capped = tree_counts_depth_capped(fragment)
        assert free == capped, f"stack cap bites in mode {mode}"
        declared = count_valid_programs(mode)["structurally_valid"]
        assert free[-1] == declared, f"mode {mode} tree count disagrees with the search module"

    assert tree_counts_depth_free(MODE_C_FULL)[-1] == 32_971_249_179
    profile = structural_profile(MODE_C_FULL)
    assert profile["matches_search_module_count"] is True
    assert profile["stack_depth_constraint_binds"] is False
    assert profile["max_terminals"] == 5
    assert profile["max_terminals"] <= STACK_DEPTH


def test_arity_identity_and_unary_parity() -> None:
    """``terminals == binary + 1`` forces the unary count's parity, per mode."""

    for mode, expected_parity in (("C", 0), ("F", 1)):
        fragment = Fragment(
            name=f"probe_{mode}",
            mode=mode,
            token_names=tuple(
                TOKEN_NAMES[index] for index in MODE_CONFIG[mode]["digit_to_token"]
            ),
            value_field="R",
        )
        profile = structural_profile(fragment)
        length = fragment.program_length
        assert profile["unary_parity"] == expected_parity
        for shape in profile["admissible_shapes"]:
            assert shape["terminals"] == shape["binary"] + 1
            assert shape["terminals"] + shape["unary"] + shape["binary"] == length
            assert shape["unary"] % 2 == expected_parity


# ---------------------------------------------------------------------------
# Field closure: a checked property, not a naming convention
# ---------------------------------------------------------------------------


def test_rational_fragment_is_closed_and_the_full_fragment_is_not() -> None:
    rational = field_closure_report(MODE_C_RATIONAL)
    assert rational["rationally_closed"] is True
    assert rational["declaration_supported"] is True
    assert rational["tokens_without_exact_rational_transfer"] == []

    full = field_closure_report(MODE_C_FULL)
    assert full["rationally_closed"] is False
    assert set(full["tokens_without_exact_rational_transfer"]) == {
        "sqrt",
        "exp",
        "ln",
        "sin",
        "cos",
        "atan",
        "pow",
    }
    # The full fragment declares field "R", so nothing is being claimed and nothing is violated.
    assert full["declaration_supported"] is True


def test_escape_witnesses_are_proved_not_asserted() -> None:
    """Every token dropped from the rational fragment carries a rational input that escapes."""

    report = field_closure_report(MODE_C_FULL)
    witnesses = {row["token"]: row for row in report["escape_witnesses"]}
    for token in ("sqrt", "exp", "ln", "sin", "cos", "atan", "pow"):
        assert witnesses[token]["proves_escape"] is True, token
        assert witnesses[token]["sympy_is_rational"] is False, token


def test_declaring_a_leaky_fragment_rational_is_refused() -> None:
    leaky = Fragment(
        name="leaky",
        mode="C",
        token_names=(*MODE_C_RATIONAL.token_names, "sqrt"),
        value_field="Q",
    )
    assert field_closure_report(leaky)["declaration_supported"] is False
    with pytest.raises(CompositionalSearchError, match="sqrt"):
        exact_rational_image(leaky)
    with pytest.raises(CompositionalSearchError):
        rational_reachability_certificate("1/2", leaky)


# ---------------------------------------------------------------------------
# The exact image really is the image
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fragment", TINY_FRAGMENTS, ids=lambda item: item.name)
def test_dp_image_equals_brute_force_on_a_tiny_fragment(fragment: Fragment) -> None:
    """Enumerate every token sequence of a small fragment and compare, element for element.

    This is the test that turns "the target is not in the closure" into a theorem.  Brute force
    here uses the search module's own :func:`program_status`, so the two sides agree on what a
    program even is, and it runs at the real mode C program length of nine, so the comparison
    is against the same object the certificates talk about.
    """

    length = fragment.program_length
    tokens = fragment.token_indices

    brute: set[Fraction] = set()
    valid = 0
    for sequence in itertools.product(tokens, repeat=length):
        if program_status(sequence) != "ok":
            continue
        valid += 1
        value, _ = evaluate_exact(sequence)
        if value is not None:
            brute.add(value)

    assert valid == tree_counts_depth_free(fragment)[-1]
    assert valid == tree_counts_depth_capped(fragment)[-1]
    assert brute == exact_rational_image(fragment)[-1]
    assert brute, "the fragment produced no values at all"


@pytest.mark.parametrize("fragment", TINY_FRAGMENTS, ids=lambda item: item.name)
def test_every_tiny_image_value_reconstructs_exhaustively(fragment: Fragment) -> None:
    """Not a sample: every element of the small image is rebuilt and replayed.

    A backward solve that inverts an operator wrongly -- ``div`` solving to ``0 / 0``, say --
    produces a program that does not evaluate at all.  This catches that exhaustively.
    """

    from sigma_theory_compiler.reachability_certificate import _reconstruct, _tree_tokens

    levels = exact_rational_image(fragment)
    for value in sorted(levels[-1]):
        tree = _reconstruct(value, fragment.program_length, levels, fragment)
        assert tree is not None, value
        tokens = _tree_tokens(tree)
        assert program_status(tokens) == "ok", value
        replayed, _ = evaluate_exact(tokens)
        assert replayed == value


def test_exact_image_level_sizes_are_pinned(image) -> None:
    assert [len(level) for level in image] == EXPECTED_IMAGE_SIZES


def test_exact_image_respects_the_declared_value_cap(image) -> None:
    """No level holds a value the grammar's own evaluator would have killed."""

    for level in image:
        assert max(abs(value) for value in level) <= VALUE_CAP_EXACT


def test_height_ladder_bounds_the_exact_image(image) -> None:
    """The O(1) counting argument is sound against the fully computed image, level by level."""

    ladder = height_bound_ladder(MODE_C_RATIONAL)
    assert len(ladder) == MODE_C_RATIONAL.program_length
    assert ladder[0] == 5
    assert ladder[-1] == 5**256
    for nodes, level in enumerate(image, start=1):
        assert max(height(value) for value in level) <= ladder[nodes - 1], nodes


def test_every_sampled_image_value_reconstructs_and_replays(image) -> None:
    """A deterministic sample of the image is rebuilt into a real program and re-executed.

    The image must contain nothing that is not genuinely reachable; each sampled value gets an
    explicit nine-token program whose exact re-execution returns that value, and whose ordinal
    round-trips through the search module's real mode C codec.
    """

    from sigma_theory_compiler.reachability_certificate import _reconstruct, _tree_tokens

    ordered = sorted(image[-1])
    sample = set(ordered[:: max(1, len(ordered) // 120)][:120])
    # Plus the corners a stride sample would step over.
    sample.update({ordered[0], ordered[-1], Fraction(0)})
    sample.add(max(ordered, key=height))
    sample.add(max(ordered, key=lambda value: value.denominator))
    sample.add(min(ordered, key=lambda value: abs(value) if value else Fraction(10**9)))
    assert len(sample) >= 100
    for value in sorted(sample):
        tree = _reconstruct(value, MODE_C_RATIONAL.program_length, image, MODE_C_RATIONAL)
        assert tree is not None, value
        tokens = _tree_tokens(tree)
        assert len(tokens) == MODE_C_RATIONAL.program_length
        assert program_status(tokens) == "ok"
        replayed, _ = evaluate_exact(tokens)
        assert replayed == value
        ordinal = encode_program(tokens, "C")
        assert decode_ordinal(ordinal, "C") == tokens
        assert 0 <= ordinal < MODE_CONFIG["C"]["space_size"]


# ---------------------------------------------------------------------------
# Direction (a): reachable, with the derivation
# ---------------------------------------------------------------------------


def test_reachable_certificate_lands_in_the_real_ordinal_space() -> None:
    certificate = rational_reachability_certificate(REACHABLE_PROBE)
    assert certificate["verdict"] == VERDICT_REACHABLE
    derivation = certificate["derivation"]

    assert derivation["program_status"] == "ok"
    assert derivation["codec_roundtrip"] is True
    assert derivation["equals_target"] is True
    assert derivation["exact_value"] == REACHABLE_PROBE
    assert derivation["node_count"] == 9
    assert derivation["terminal_count"] == derivation["binary_count"] + 1
    assert derivation["max_stack_depth"] <= STACK_DEPTH

    ordinal = derivation["ordinal"]
    assert 0 <= ordinal < MODE_CONFIG["C"]["space_size"] == 1_801_152_661_463
    assert decode_ordinal(ordinal, "C") == tuple(derivation["tokens"])

    # The trace is the proof: replay it by hand and land on the target.
    trace = derivation["trace"]
    assert len(trace) == 9
    assert trace[-1]["stack"] == [REACHABLE_PROBE]

    assert verify_certificate(certificate)["accepted"] is True


def test_small_rationals_are_all_reachable_and_replay() -> None:
    """A spread of ordinary rationals, each certified constructively end to end."""

    for text in ("1", "0", "-5", "22/7", "1/7", "355/113", "-13/17"):
        certificate = rational_reachability_certificate(text)
        assert certificate["verdict"] == VERDICT_REACHABLE, text
        assert verify_certificate(certificate)["accepted"] is True
        tokens = tuple(certificate["derivation"]["tokens"])
        value, _ = evaluate_exact(tokens)
        assert value == Fraction(text)


# ---------------------------------------------------------------------------
# Direction (b): outside, by argument rather than by absence of a hit
# ---------------------------------------------------------------------------


def test_unreachable_certificate_is_an_exhaustion_not_a_search(image) -> None:
    certificate = rational_reachability_certificate(UNREACHABLE_PROBE)
    assert certificate["verdict"] == VERDICT_OUTSIDE
    argument = certificate["argument"]
    assert argument["kind"] == "exact_image_exhaustion"
    assert argument["kind"] in ARGUMENT_KINDS
    assert argument["level_sizes"] == EXPECTED_IMAGE_SIZES
    assert argument["image_size_at_budget"] == EXPECTED_IMAGE_SIZES[-1]
    assert certificate["exact_image"]["depth_lemma_holds"] is True

    # The argument is not the height bound in disguise: this target is well inside the ladder.
    assert argument["height_argument_would_suffice"] is False
    assert height(Fraction(UNREACHABLE_PROBE)) == 9973
    assert Fraction(UNREACHABLE_PROBE) not in image[-1]

    assert verify_certificate(certificate)["accepted"] is True


def test_height_bound_certificate_fires_without_enumerating() -> None:
    certificate = rational_reachability_certificate(HEIGHT_PROBE, exhaustive=False)
    assert certificate["verdict"] == VERDICT_OUTSIDE
    assert certificate["argument"]["kind"] == "height_bound"
    assert certificate["argument"]["bound"] == str(5**256)
    assert certificate["argument"]["target_height"] == 5**257
    assert "exact_image" not in certificate
    assert verify_certificate(certificate)["accepted"] is True


def test_value_cap_certificate_covers_the_full_alphabet() -> None:
    certificate = rational_reachability_certificate(VALUE_CAP_PROBE, exhaustive=False)
    assert certificate["verdict"] == VERDICT_OUTSIDE
    assert certificate["argument"]["kind"] == "value_cap"
    assert "full alphabet" in certificate["argument"]["scope"]
    assert Fraction(VALUE_CAP_PROBE) > VALUE_CAP_EXACT
    assert verify_certificate(certificate)["accepted"] is True


def test_non_exhaustive_mode_says_unresolved_rather_than_guessing() -> None:
    certificate = rational_reachability_certificate(UNREACHABLE_PROBE, exhaustive=False)
    assert certificate["verdict"] == VERDICT_UNRESOLVED
    assert certificate["argument"]["kind"] == "none"
    assert "derivation" not in certificate
    assert verify_certificate(certificate)["accepted"] is True


def test_sqrt2_is_excluded_from_the_rational_fragment_structurally() -> None:
    certificate = irrational_exclusion_certificate("sqrt2")
    assert certificate["verdict"] == VERDICT_OUTSIDE
    assert certificate["proof"]["kind"] == "field_closure"
    assert certificate["target"]["sympy_is_rational"] is False
    assert verify_certificate(certificate)["accepted"] is True

    for name in ("pi", "e", "sqrt3", "ln2", "ln3", "zeta2", "phi", "e_pi"):
        excluded = irrational_exclusion_certificate(name)
        assert excluded["verdict"] == VERDICT_OUTSIDE, name
        assert verify_certificate(excluded)["accepted"] is True


def test_open_irrationality_comes_back_unresolved() -> None:
    """Catalan and Euler-Mascheroni are open; the module refuses to pretend otherwise."""

    for name in ("catalan", "euler_gamma"):
        assert SYMBOLIC_TARGETS[name].is_rational is None
        certificate = irrational_exclusion_certificate(name)
        assert certificate["verdict"] == VERDICT_UNRESOLVED, name
        assert certificate["proof"]["kind"] == "none"
        assert verify_certificate(certificate)["accepted"] is True


# ---------------------------------------------------------------------------
# The symbolic lane over the full alphabet
# ---------------------------------------------------------------------------


def test_symbolic_lane_proves_builtin_identities_and_never_overclaims() -> None:
    """Every mode C rediscovery identity is proved exactly or left honestly unresolved."""

    proved = 0
    for item in BUILTIN_KNOWN_IDENTITIES:
        if item["mode"] != "C":
            continue
        certificate = symbolic_reachability_certificate(str(item["rpn"]), str(item["target"]))
        assert certificate["verdict"] in {VERDICT_REACHABLE, VERDICT_UNRESOLVED}, item["id"]
        assert verify_certificate(certificate)["accepted"] is True
        if certificate["verdict"] == VERDICT_REACHABLE:
            proved += 1
            assert certificate["proof"]["kind"] == "symbolic_identity"
            expression = sp.sympify(certificate["program"]["sympy"])
            assert sp.simplify(expression - SYMBOLIC_TARGETS[str(item["target"])]) == 0 or (
                certificate["proof"]["tactic"] == "tangent_with_bound"
            )
    assert proved >= 8, "the symbolic lane stopped proving the rediscovery set"


def test_machin_identity_needs_the_tangent_tactic() -> None:
    """``pi = 4(arctan 1/2 + arctan 1/3)`` -- no amount of ``simplify`` closes this one."""

    certificate = symbolic_reachability_certificate(
        "1/2 atan 1/3 atan add 4 mul 1 mul", "pi"
    )
    assert certificate["verdict"] == VERDICT_REACHABLE
    assert certificate["proof"]["tactic"] == "tangent_with_bound"
    assert verify_certificate(certificate)["accepted"] is True


def test_pi_impostor_is_refuted_exactly() -> None:
    """The 38-digit impostor from the search module's own docstring is a real program."""

    certificate = symbolic_reachability_certificate(PI_IMPOSTOR_RPN, "pi")
    assert certificate["program"]["program_status"] == "ok"
    assert certificate["program"]["codec_roundtrip"] is True
    assert certificate["verdict"] == VERDICT_PROGRAM_REFUTED
    assert certificate["proof"]["kind"] == "strict_inequality"
    assert certificate["proof"]["relation"] == "strictly_below"
    assert certificate["proof"]["scope"] == "this derivation only"
    assert verify_certificate(certificate)["accepted"] is True


# ---------------------------------------------------------------------------
# Controls: the positives are worth nothing without these
# ---------------------------------------------------------------------------


def test_forged_reachability_for_an_unreachable_target_is_rejected() -> None:
    """The named control.  A well-formed certificate, a real ordinal, and a false claim.

    Nothing about this certificate is malformed: the tokens are in the fragment, the program is
    structurally valid, the ordinal encodes and decodes, the seal is fresh.  Only the target is
    swapped for ``1/9973``, which the module has proved is outside the fragment.  It must die.
    """

    honest = rational_reachability_certificate(REACHABLE_PROBE)
    assert honest["verdict"] == VERDICT_REACHABLE

    outside = Fraction(UNREACHABLE_PROBE)
    forged = json.loads(json.dumps(honest))
    forged["target"] = {
        "kind": "rational",
        "exact": str(outside),
        "numerator": outside.numerator,
        "denominator": outside.denominator,
        "height": height(outside),
    }
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = canonical_sha256(body)

    # Every structural property still checks out -- that is the point of the control.
    tokens = tuple(forged["derivation"]["tokens"])
    assert program_status(tokens) == "ok"
    assert encode_program(tokens, "C") == forged["derivation"]["ordinal"]
    assert decode_ordinal(forged["derivation"]["ordinal"], "C") == tokens

    with pytest.raises(CompositionalSearchError, match="not to the declared target"):
        verify_certificate(forged)


def test_forged_unreachability_for_a_reachable_target_is_rejected() -> None:
    honest = rational_reachability_certificate(UNREACHABLE_PROBE)
    reachable = Fraction(REACHABLE_PROBE)
    forged = json.loads(json.dumps(honest))
    forged["target"] = {
        "kind": "rational",
        "exact": str(reachable),
        "numerator": reachable.numerator,
        "denominator": reachable.denominator,
        "height": height(reachable),
    }
    body = {key: value for key, value in forged.items() if key != "content_sha256"}
    forged["content_sha256"] = canonical_sha256(body)
    with pytest.raises(CompositionalSearchError, match="IS in the exact image"):
        verify_certificate(forged)


def test_controls_suite_all_honest_accepted_and_all_probes_rejected(controls) -> None:
    assert controls["all_honest_accepted"] is True
    assert controls["all_probes_rejected"] is True
    assert len(controls["honest"]) == 8
    for row in controls["honest"]:
        assert row["accepted"] is True, row
        assert row["verdict"] == row["expected_verdict"], row


def test_every_named_probe_is_present_and_rejected(controls) -> None:
    """Pinned by name: a silently dropped probe is a silently weakened control."""

    expected = {
        "forged_reachability_for_unreachable_target",
        "forged_reachability_doctored_trace",
        "forged_reachability_wrong_ordinal",
        "forged_reachability_out_of_alphabet",
        "forged_unreachability_for_reachable_target",
        "forged_height_argument",
        "forged_value_cap_argument",
        "forged_field_closure_declaration",
        "forged_exact_image_size",
        "unsealed_verdict_flip",
        "forged_symbolic_identity_for_the_pi_impostor",
        "forged_exclusion_of_an_open_constant",
        "forged_exclusion_with_a_leaky_fragment",
        "forged_fragment_name_drift",
        "forged_structural_profile",
    }
    seen = {row["probe"] for row in controls["probes"]}
    assert seen == expected
    for row in controls["probes"]:
        assert row["rejected"] is True, row
        assert row["reason"]


def test_each_gate_bites_on_its_own(controls) -> None:
    """Probes must not die on each other's checks.

    A verifier whose gates overlap looks strong and is not: one early check can mask the
    failure of every later one.  So every probe is required to produce its own rejection
    reason, with exactly one documented exception -- the field-closure gate runs before the
    lane dispatch by design, so the exact-rational and symbolic forgeries of a leaky fragment
    are rejected by the same check and say so identically.
    """

    reasons = {row["probe"]: row["reason"] for row in controls["probes"]}
    shared_by_design = {
        "forged_field_closure_declaration",
        "forged_exclusion_with_a_leaky_fragment",
    }
    collisions = [
        {probe for probe, reason in reasons.items() if reason == text}
        for text in set(reasons.values())
    ]
    for group in collisions:
        if len(group) > 1:
            assert group == shared_by_design, group
    assert len(set(reasons.values())) == len(reasons) - 1
    assert reasons["forged_field_closure_declaration"] == (
        reasons["forged_exclusion_with_a_leaky_fragment"]
    )


def _leaf_paths(node, prefix=()):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _leaf_paths(value, (*prefix, key))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _leaf_paths(value, (*prefix, index))
    else:
        yield prefix, node


def _mutate(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, str):
        return value + "x" if value else "MUTATED"
    return "MUTATED"


def _assign(root, path, value):
    node = root
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


@pytest.mark.parametrize(
    "builder",
    [
        lambda: rational_reachability_certificate(REACHABLE_PROBE),
        lambda: rational_reachability_certificate(UNREACHABLE_PROBE),
        lambda: irrational_exclusion_certificate("sqrt2"),
    ],
    ids=["reachable", "exhaustion", "field_closure_exclusion"],
)
def test_no_field_can_be_tampered_and_resealed(builder) -> None:
    """Mutate every leaf in turn, reseal, and require rejection.

    Hand-written probes test the holes their author thought of.  This walks the whole body --
    several hundred fields per certificate -- and demands that changing any one of them, with a
    fresh and perfectly valid seal, kills the certificate.

    At most one field is allowed through: ``fragment.name``.  The name is an *input* to the
    fragment, not a derived claim, so renaming produces a consistently renamed and still-true
    certificate; renaming to a built-in fragment's name is caught separately, by the
    name-binding gate.  Certificates whose argument text quotes the fragment name -- the
    exhaustion and field-closure ones -- are immune even to that, because the quoted copy no
    longer matches.
    """

    from sigma_theory_compiler.reachability_certificate import _seal

    certificate = builder()
    survivors = []
    checked = 0
    for path, value in _leaf_paths(certificate):
        if path == ("content_sha256",):
            continue
        replacement = _mutate(value)
        if replacement == value:
            continue
        checked += 1
        candidate = json.loads(json.dumps(certificate))
        _assign(candidate, path, replacement)
        candidate = _seal(candidate)
        try:
            verify_certificate(candidate)
        except CompositionalSearchError:
            continue
        survivors.append(".".join(str(part) for part in path))

    assert checked >= 100, f"only {checked} fields were exercised"
    assert set(survivors) <= {"fragment.name"}, survivors


def test_verifier_rejects_a_tampered_seal() -> None:
    certificate = rational_reachability_certificate(REACHABLE_PROBE)
    tampered = dict(certificate)
    tampered["verdict"] = VERDICT_OUTSIDE
    with pytest.raises(CompositionalSearchError, match="seal does not match"):
        verify_certificate(tampered)


def test_verifier_rejects_an_unknown_schema() -> None:
    certificate = dict(rational_reachability_certificate(REACHABLE_PROBE))
    certificate["schema_version"] = "something-else-1.0"
    with pytest.raises(CompositionalSearchError, match="schema_version"):
        verify_certificate(certificate)


# ---------------------------------------------------------------------------
# Hygiene: exact arithmetic, determinism, receipts
# ---------------------------------------------------------------------------


def _walk(value, path="$"):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, f"{path}[{index}]")
    else:
        yield path, value


def test_certificates_carry_no_floats() -> None:
    """Repo rule: exact arithmetic only on a certificate path."""

    certificates = [
        rational_reachability_certificate(REACHABLE_PROBE),
        rational_reachability_certificate(UNREACHABLE_PROBE),
        rational_reachability_certificate(HEIGHT_PROBE, exhaustive=False),
        rational_reachability_certificate(VALUE_CAP_PROBE, exhaustive=False),
        symbolic_reachability_certificate(PI_HONEST_RPN, "pi"),
        symbolic_reachability_certificate(PI_IMPOSTOR_RPN, "pi"),
        irrational_exclusion_certificate("sqrt2"),
    ]
    for certificate in certificates:
        for path, value in _walk(certificate):
            assert not isinstance(value, float), f"{path} carries a float: {value!r}"


def test_float_targets_are_refused() -> None:
    with pytest.raises(CompositionalSearchError, match="float targets are refused"):
        parse_exact_rational(0.5)
    with pytest.raises(CompositionalSearchError):
        rational_reachability_certificate(3.14159)
    assert parse_exact_rational("355/113") == Fraction(355, 113)
    assert parse_exact_rational(7) == Fraction(7)


def test_certificates_are_deterministic() -> None:
    for builder in (
        lambda: rational_reachability_certificate(REACHABLE_PROBE),
        lambda: rational_reachability_certificate(UNREACHABLE_PROBE),
        lambda: symbolic_reachability_certificate(PI_HONEST_RPN, "pi"),
        lambda: irrational_exclusion_certificate("sqrt2"),
    ):
        first = builder()
        second = builder()
        assert first["content_sha256"] == second["content_sha256"]
        assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_receipt_seals_and_decides() -> None:
    receipt = build_receipt([REACHABLE_PROBE, UNREACHABLE_PROBE])
    assert receipt["schema_version"] == CERTIFICATE_SCHEMA
    assert receipt["decision"] == "CERTIFIED"
    assert receipt["controls"]["all_honest_accepted"] is True
    assert receipt["controls"]["all_probes_rejected"] is True
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert canonical_sha256(body) == receipt["content_sha256"]
    for certificate in receipt["certificates"]:
        assert verify_certificate(certificate)["accepted"] is True


def test_cli_writes_a_sealed_receipt(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    code = main(["--target", REACHABLE_PROBE, "--target", UNREACHABLE_PROBE, "--output", str(output)])
    assert code == 0
    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["decision"] == "CERTIFIED"
    assert [entry["verdict"] for entry in receipt["certificates"]] == [
        VERDICT_REACHABLE,
        VERDICT_OUTSIDE,
    ]


def test_module_runs_as_a_script(tmp_path: Path) -> None:
    output = tmp_path / "cli.json"
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "sigma_theory_compiler.reachability_certificate",
            "--target",
            REACHABLE_PROBE,
            "--output",
            str(output),
        ],
        cwd=root,
        env={"PYTHONPATH": str(root / "src"), "PATH": ""},
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["decision"] == "CERTIFIED"
