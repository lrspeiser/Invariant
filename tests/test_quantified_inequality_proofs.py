"""B6 quantified-inequality gates.

The specific danger here is a finite check masquerading as a proof over an infinite
domain.  Emission must require a symbolic argument; the window is only ever a falsifier.
"""

from __future__ import annotations

import pytest

from sigma_theory_compiler.quantified_inequality_proofs import (
    CLAIMS,
    SUPPORTED_RELATIONS,
    SYSTEM_CAPS,
    QuantifiedInequalityError,
    prove_quantified_inequality,
    validate_result,
)

QUADRATIC = {
    "namespace": "Invariant",
    "name": "quadraticGrows",
    "relation": "monotone_increasing",
    "coefficients": [0, 3, 1],
}


def _with(**overrides):
    return {**QUADRATIC, **overrides}


# ---------------------------------------------------------------------------
# Reaching an infinite domain
# ---------------------------------------------------------------------------


def test_quantifier_ranges_over_an_infinite_domain():
    result = prove_quantified_inequality(QUADRATIC)
    assert result["decision"] == "PROVED_LOCALLY"
    assert result["quantifier"] == "forall n : Nat"
    assert result["claims"]["quantifier_ranges_over_an_infinite_domain"] is True


def test_nonlinear_monotonicity_uses_the_exact_forward_difference():
    """Degree two is outside omega's completeness, so the difference carries the proof."""

    result = prove_quantified_inequality(QUADRATIC)
    assert result["difference_coefficients"] == [4, 2]  # (n+1)^2+3(n+1) - (n^2+3n) = 2n+4


def test_linear_and_cubic_cases_both_prove():
    linear = prove_quantified_inequality(
        _with(name="linearGrows", coefficients=[2, 5])
    )
    cubic = prove_quantified_inequality(
        _with(name="cubicNonneg", relation="nonnegative", coefficients=[7, 1, 2, 2])
    )
    assert linear["decision"] == "PROVED_LOCALLY"
    assert cubic["decision"] == "PROVED_LOCALLY"


# ---------------------------------------------------------------------------
# False statements are refused
# ---------------------------------------------------------------------------


def test_constant_is_not_strictly_increasing():
    result = prove_quantified_inequality(_with(name="constGrows", coefficients=[5]))
    assert result["decision"] == "REJECT"
    assert result["lean_source"] is None
    assert result["first_blocker"] == "finite_window_counterexample"


def test_constant_is_accepted_as_nondecreasing():
    result = prove_quantified_inequality(
        _with(name="constFlat", relation="monotone_nondecreasing", coefficients=[5])
    )
    assert result["decision"] == "PROVED_LOCALLY"


def test_rejection_emits_no_lean():
    result = prove_quantified_inequality(_with(name="constGrows", coefficients=[5]))
    assert result["lean_source"] is None
    assert result["lean_source_sha256"] is None


# ---------------------------------------------------------------------------
# The finite window is a falsifier, never a justification
# ---------------------------------------------------------------------------


def test_emission_requires_a_symbolic_argument_not_just_the_window():
    result = prove_quantified_inequality(QUADRATIC)
    assert result["symbolic_argument_holds"] is True
    assert result["claims"]["finite_window_check_alone_justifies_emission"] is False


def test_window_result_is_recorded_for_audit():
    result = prove_quantified_inequality(QUADRATIC)
    assert result["window_check"]["holds"] is True
    assert result["system_caps"]["verification_window"] == SYSTEM_CAPS["verification_window"]


# ---------------------------------------------------------------------------
# Declared scope and kernel boundary
# ---------------------------------------------------------------------------


def test_kernel_verified_is_always_false_here():
    result = prove_quantified_inequality(QUADRATIC)
    assert result["kernel_verified"] is False
    assert result["claims"]["exact_local_check_is_kernel_verification"] is False


def test_reals_and_analysis_are_explicitly_out_of_scope():
    result = prove_quantified_inequality(QUADRATIC)
    assert result["claims"]["covers_reals_limits_or_analysis"] is False
    assert "Mathlib is not a declared dependency" in result["scope"]


def test_generated_lean_uses_no_mathlib_tactic():
    source = prove_quantified_inequality(QUADRATIC)["lean_source"]
    for tactic in ("ring", "linarith", "nlinarith", "positivity"):
        assert f"  {tactic}" not in source
    assert "import Std.Tactic" in source
    assert "omega" in source


def test_supported_relations_are_declared_and_finite():
    result = prove_quantified_inequality(QUADRATIC)
    assert result["supported_relations"] == list(SUPPORTED_RELATIONS)


@pytest.mark.parametrize(
    "problem",
    [
        {"relation": "converges"},
        {"coefficients": [0, -3, 1]},
        {"name": "9bad"},
        {"namespace": "not an identifier"},
    ],
)
def test_out_of_scope_input_is_refused(problem):
    with pytest.raises(QuantifiedInequalityError):
        prove_quantified_inequality(_with(**problem))


def test_unexpected_keys_are_refused():
    with pytest.raises(QuantifiedInequalityError):
        prove_quantified_inequality({**QUADRATIC, "extra": 1})


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_result_is_deterministic_and_replays():
    first = prove_quantified_inequality(QUADRATIC)
    assert first == prove_quantified_inequality(QUADRATIC)
    validate_result(first)


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = prove_quantified_inequality(QUADRATIC)
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["kernel_verified"] = True
    with pytest.raises(QuantifiedInequalityError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


def test_claim_boundary_is_bound_into_every_result():
    assert prove_quantified_inequality(QUADRATIC)["claims"] == CLAIMS
