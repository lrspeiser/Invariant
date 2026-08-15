"""B5 lemma-decomposition gates.

The dangerous failure here is emitting confident, well-formed Lean for a statement that
is false, or letting an exact local check be read as kernel verification.  Both are
pinned below.
"""

from __future__ import annotations

import pytest

from sigma_theory_compiler.lemma_decomposition import (
    CLAIMS,
    SYSTEM_CAPS,
    LemmaDecompositionError,
    decompose_closed_form_proof,
    validate_result,
)

# a(0)=7, a(n+1)=a(n)+(6n^2+10n+5), closed form 2n^3+2n^2+n+7.
# This is the theorem the repository previously proved with a hand-written helper lemma.
TRUE_PROBLEM = {
    "namespace": "Invariant",
    "sequence_name": "generatedSequence",
    "base_value": 7,
    "step": [5, 10, 6],
    "closed_form": [7, 1, 2, 2],
}


def _with(**overrides):
    return {**TRUE_PROBLEM, **overrides}


# ---------------------------------------------------------------------------
# Decomposition structure
# ---------------------------------------------------------------------------


def test_proof_is_split_into_three_independent_obligations():
    result = decompose_closed_form_proof(TRUE_PROBLEM)
    assert result["decision"] == "DECOMPOSED"
    ids = [item["obligation_id"] for item in result["obligations"]]
    assert ids == ["base_case", "successor_identity", "main_induction"]


def test_algebra_and_induction_are_separated():
    """The point of decomposing is that the algebra carries no induction and vice versa."""

    result = decompose_closed_form_proof(TRUE_PROBLEM)
    by_id = {item["obligation_id"]: item for item in result["obligations"]}
    assert by_id["successor_identity"]["contains_induction"] is False
    assert by_id["base_case"]["contains_induction"] is False
    assert by_id["main_induction"]["contains_induction"] is True
    assert result["counts"]["obligations_with_induction"] == 1


def test_dependency_edges_are_explicit():
    result = decompose_closed_form_proof(TRUE_PROBLEM)
    by_id = {item["obligation_id"]: item for item in result["obligations"]}
    assert by_id["base_case"]["depends_on"] == []
    assert by_id["successor_identity"]["depends_on"] == []
    assert set(by_id["main_induction"]["depends_on"]) == {"base_case", "successor_identity"}


# ---------------------------------------------------------------------------
# False statements must never reach the kernel
# ---------------------------------------------------------------------------


def test_false_closed_form_is_rejected_and_emits_no_lean():
    result = decompose_closed_form_proof(_with(closed_form=[7, 1, 2, 3]))
    assert result["decision"] == "REJECT"
    assert result["lean_source"] is None
    assert result["first_blocker"] == "exact_local_check_failed:successor_identity"


def test_false_base_value_is_rejected():
    result = decompose_closed_form_proof(_with(base_value=8))
    assert result["decision"] == "REJECT"
    assert "base_case" in result["failed_obligations"]
    assert result["lean_source"] is None


def test_a_failing_obligation_is_named_individually():
    """Independent failure is the whole reason to decompose."""

    result = decompose_closed_form_proof(_with(base_value=8))
    failed = result["failed_obligations"]
    assert "base_case" in failed
    assert "successor_identity" not in failed


# ---------------------------------------------------------------------------
# Kernel-verification boundary
# ---------------------------------------------------------------------------


def test_kernel_verified_is_always_false_in_this_receipt():
    result = decompose_closed_form_proof(TRUE_PROBLEM)
    assert result["kernel_verified"] is False
    assert result["claims"]["exact_local_check_is_kernel_verification"] is False
    assert result["claims"]["kernel_verified_without_a_ci_receipt"] is False
    assert "must be checked by the pinned Lean" in result["kernel_verification_requirement"]


def test_generated_lean_uses_no_mathlib_tactic():
    """Mathlib is not a declared dependency; emitting `ring` would not compile here."""

    source = decompose_closed_form_proof(TRUE_PROBLEM)["lean_source"]
    for tactic in ("ring", "linarith", "push_cast", "nlinarith", "positivity"):
        assert f"  {tactic}" not in source
        assert f"[{tactic}]" not in source
    assert "import Std.Tactic" in source
    assert "omega" in source


def test_generated_lean_matches_the_verified_proof_skeleton():
    source = decompose_closed_form_proof(TRUE_PROBLEM)["lean_source"]
    assert "def generatedSequence : Nat → Nat" in source
    assert "induction n with" in source
    assert "| zero => rfl" in source
    assert "exact generatedSequenceSuccessorIdentity n" in source
    assert "INVARIANT_LEAN_DEPENDENCY_AUDIT_V1_BEGIN" in source


def test_dependency_audit_lists_every_cited_lemma():
    source = decompose_closed_form_proof(TRUE_PROBLEM)["lean_source"]
    assert "dependency=Invariant.generatedSequenceBaseCase" in source
    assert "dependency=Invariant.generatedSequenceSuccessorIdentity" in source
    assert "dependency=Nat.rec" in source


# ---------------------------------------------------------------------------
# Declared scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "problem",
    [
        {"base_value": -1},
        {"step": [-5, 10, 6]},
        {"closed_form": [7, 1, 2, -2]},
    ],
)
def test_negative_data_is_refused_rather_than_emitted(problem):
    with pytest.raises(LemmaDecompositionError, match="nat_domain_required"):
        decompose_closed_form_proof(_with(**problem))


def test_degree_cap_is_enforced():
    with pytest.raises(LemmaDecompositionError):
        decompose_closed_form_proof(_with(closed_form=[1] * (SYSTEM_CAPS["max_degree"] + 3)))


@pytest.mark.parametrize(
    "problem",
    [
        {"namespace": "not an identifier"},
        {"sequence_name": "9bad"},
        {"closed_form": []},
        {"step": [1.5]},
    ],
)
def test_malformed_problem_is_refused(problem):
    with pytest.raises(LemmaDecompositionError):
        decompose_closed_form_proof(_with(**problem))


def test_unexpected_problem_keys_are_refused():
    with pytest.raises(LemmaDecompositionError):
        decompose_closed_form_proof({**TRUE_PROBLEM, "extra": 1})


# ---------------------------------------------------------------------------
# Receipt integrity
# ---------------------------------------------------------------------------


def test_result_is_deterministic_and_replays():
    first = decompose_closed_form_proof(TRUE_PROBLEM)
    assert first == decompose_closed_form_proof(TRUE_PROBLEM)
    validate_result(first)


def test_reseal_after_tamper_fails_closed():
    from sigma_theory_compiler.sigma_core import canonical_sha256

    result = decompose_closed_form_proof(TRUE_PROBLEM)
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    body["kernel_verified"] = True
    with pytest.raises(LemmaDecompositionError):
        validate_result({**body, "content_sha256": canonical_sha256(body)})


def test_claim_boundary_is_bound_into_every_result():
    result = decompose_closed_form_proof(TRUE_PROBLEM)
    assert result["claims"] == CLAIMS
    assert result["claims"]["hand_written_helper_lemmas_required"] is False
    assert "does not establish novelty" in result["scope"]
