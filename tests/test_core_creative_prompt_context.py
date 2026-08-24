from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler import core_creative_prompt_context as C
from sigma_theory_compiler.claude_creativity_api import ClaudeCreativityError
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _context() -> dict[str, object]:
    symmetry = json.loads(
        (ROOT / "runs/math/symmetry-dimension-derivation/receipt.json").read_text()
    )
    learned = json.loads((ROOT / "runs/math/learned-invariant-discovery/receipt.json").read_text())
    state_pair = json.loads(
        (ROOT / "runs/math/state-pair-invariant-discovery/receipt.json").read_text()
    )
    uncertain = json.loads(
        (ROOT / "runs/math/uncertain-invariant-discovery/receipt.json").read_text()
    )
    grammar = json.loads((ROOT / "runs/math/expanded-typed-grammar/receipt.json").read_text())
    proof = json.loads((ROOT / "runs/math/independent-proof-plan-search/receipt.json").read_text())
    return C.build_creative_prompt_context(symmetry, learned, state_pair, uncertain, grammar, proof)


def test_prompt_context_is_sealed_creativity_first_and_broad() -> None:
    context = _context()
    C.validate_creative_prompt_context(context)
    assert context["creativity_policy"] == {
        "creativity_is_primary": True,
        "generate_multiple_mechanisms_before_falsification": True,
        "learn_higher_degree_rational_and_logarithmic_features": True,
        "learn_matrix_and_nonlinear_actions_from_state_pairs": True,
        "learn_multivariate_rational_features": True,
        "origin_labels_are_fallible_lineage_assessments": True,
        "preserve_joint_and_unit_hypothesis_branches": True,
        "retain_every_schema_admitted_idea": True,
        "retain_failed_and_underdetermined_invariant_branches": True,
        "retain_set_valued_uncertainty_branches": True,
        "uncertainty_does_not_prune": True,
    }
    assert len(context["first_principles_briefs"]) == 5
    drag = next(
        brief
        for brief in context["first_principles_briefs"]
        if brief["problem_id"] == "control.drag-similarity"
    )
    assert drag["invariant_coordinate_arity"] == 2
    assert drag["forced_form"].startswith("F(drag_force/")
    assert len(context["learned_invariant_briefs"]) == 3
    assert {brief["identifiability_status"] for brief in context["learned_invariant_briefs"]} == {
        "PASS_LEARNED_INVARIANT_BASIS",
        "REJECT_TRAIN_ONLY_INVARIANT_SPACE",
        "UNDERDETERMINED_RETAIN_CANDIDATE_SUBSPACE",
    }
    assert len(context["state_pair_invariant_briefs"]) == 7
    assert {brief["action_kind"] for brief in context["state_pair_invariant_briefs"]} == {
        "matrix_conjugation",
        "matrix_orthogonal",
        "nonlinear_polynomial",
        "nonlinear_polynomial_degree3",
        "rational_laurent",
        "rational_multivariate_laurent",
        "transcendental_logarithmic",
    }
    matrix = next(
        brief
        for brief in context["state_pair_invariant_briefs"]
        if brief["action_kind"] == "matrix_conjugation"
    )
    assert matrix["candidate_invariant_coordinates"] == ["a + d", "a*d - b*c"]
    rational = next(
        brief
        for brief in context["state_pair_invariant_briefs"]
        if brief["action_kind"] == "rational_laurent"
    )
    assert rational["candidate_invariant_coordinates"] == ["x + 1/x"]
    assert rational["retained_linear_invariant_basis"] == [
        "x + 1/x",
        "x**2 + 1/x**2",
    ]
    multivariate_rational = next(
        brief
        for brief in context["state_pair_invariant_briefs"]
        if brief["action_kind"] == "rational_multivariate_laurent"
    )
    assert multivariate_rational["candidate_invariant_coordinates"] == ["1/y + x"]
    logarithmic = next(
        brief
        for brief in context["state_pair_invariant_briefs"]
        if brief["action_kind"] == "transcendental_logarithmic"
    )
    assert logarithmic["candidate_invariant_coordinates"] == ["2*log(x) - log(y)"]
    assert len(context["uncertain_invariant_briefs"]) == 5
    assert {brief["observation_mode"] for brief in context["uncertain_invariant_briefs"]} == {
        "missingness",
        "noisy_interval",
        "one_sided_censoring",
        "joint_support",
        "unit_hypotheses",
    }
    noisy = next(
        brief
        for brief in context["uncertain_invariant_briefs"]
        if brief["observation_mode"] == "noisy_interval"
    )
    assert len(noisy["candidate_invariant_coordinates"]) == 3
    assert noisy["deployment_surviving_coordinates"] == ["a*b/c"]
    dependent = next(
        brief
        for brief in context["uncertain_invariant_briefs"]
        if brief["observation_mode"] == "joint_support"
    )
    assert dependent["dependence_semantics"] == (
        "finite_joint_support_without_marginal_factorization"
    )
    unit = next(
        brief
        for brief in context["uncertain_invariant_briefs"]
        if brief["observation_mode"] == "unit_hypotheses"
    )
    assert unit["retained_evidence_branches"] == [
        {"branch_ids": ["b_three_c_two"], "expression": "a*b/c"},
        {"branch_ids": ["b_half"], "expression": "a**2/b"},
    ]
    assert len(context["typed_formula_kinds"]) == 8
    assert "piecewise_relation" in context["typed_formula_kinds"]
    assert len(context["independent_proof_mechanisms"]) == 6
    assert context["origin_assessment_labels"] == C.ORIGIN_ASSESSMENTS


def test_prompt_context_rejects_policy_tampering() -> None:
    context = _context()
    changed = deepcopy(context)
    changed["creativity_policy"]["uncertainty_does_not_prune"] = False
    with pytest.raises(C.CoreCreativePromptContextError, match="seal"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_loss_of_multi_coordinate_coverage() -> None:
    context = _context()
    changed = deepcopy(context)
    drag = next(
        brief
        for brief in changed["first_principles_briefs"]
        if brief["problem_id"] == "control.drag-similarity"
    )
    drag["candidate_invariant_coordinates"] = drag["candidate_invariant_coordinates"][:1]
    drag["invariant_coordinate_arity"] = 1
    drag["forced_form"] = f"F({drag['candidate_invariant_coordinates'][0]}) = 0"
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="multi-coordinate"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_pruning_of_failed_learned_branch() -> None:
    context = _context()
    changed = deepcopy(context)
    changed["learned_invariant_briefs"] = [
        brief
        for brief in changed["learned_invariant_briefs"]
        if brief["identifiability_status"] != "REJECT_TRAIN_ONLY_INVARIANT_SPACE"
    ]
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="learned invariant brief"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_loss_of_nonlinear_action_branch() -> None:
    context = _context()
    changed = deepcopy(context)
    changed["state_pair_invariant_briefs"] = [
        brief
        for brief in changed["state_pair_invariant_briefs"]
        if brief["action_kind"] != "nonlinear_polynomial"
    ]
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="state-pair invariant brief"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_loss_of_laurent_feature_grammar() -> None:
    context = _context()
    changed = deepcopy(context)
    rational = next(
        brief
        for brief in changed["state_pair_invariant_briefs"]
        if brief["action_kind"] == "rational_laurent"
    )
    rational["feature_grammar"] = {
        "kind": "polynomial_monomials",
        "maximum_total_degree": 2,
    }
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="state-pair action coverage"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_pruning_of_censored_candidates() -> None:
    context = _context()
    changed = deepcopy(context)
    censored = next(
        brief
        for brief in changed["uncertain_invariant_briefs"]
        if brief["observation_mode"] == "one_sided_censoring"
    )
    censored["candidate_invariant_coordinates"] = censored["deployment_surviving_coordinates"]
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="uncertain invariant"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_factorization_of_joint_support() -> None:
    context = _context()
    changed = deepcopy(context)
    dependent = next(
        brief
        for brief in changed["uncertain_invariant_briefs"]
        if brief["observation_mode"] == "joint_support"
    )
    dependent["dependence_semantics"] = "independent_marginal_bounds"
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="uncertain invariant"):
        C.validate_creative_prompt_context(changed)


def test_prompt_context_rejects_resealed_unit_hypothesis_switch() -> None:
    context = _context()
    changed = deepcopy(context)
    unit = next(
        brief
        for brief in changed["uncertain_invariant_briefs"]
        if brief["observation_mode"] == "unit_hypotheses"
    )
    unit["retained_evidence_branches"][1]["branch_ids"] = ["b_three_c_two"]
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativePromptContextError, match="uncertain invariant"):
        C.validate_creative_prompt_context(changed)


def test_transport_injects_context_and_binds_actual_provider_prompt() -> None:
    context = _context()
    captured: dict[str, object] = {}

    def provider(method, url, headers, body, timeout):
        captured.update(
            {
                "body": body,
                "headers": dict(headers),
                "method": method,
                "timeout": timeout,
                "url": url,
            }
        )
        return 200, {
            "content": [{"text": json.dumps({"hypotheses": []}), "type": "text"}],
            "id": "msg_context_test",
            "type": "message",
        }

    transport = C.FirstPrinciplesContextTransport(context, transport=provider)
    request = {
        "messages": [
            {
                "content": json.dumps(
                    {"benchmark": {"blind_id": "blind.test"}, "instruction": "create"}
                ),
                "role": "user",
            }
        ],
        "output_config": {
            "format": {
                "schema": {
                    "properties": {
                        "role": {"const": "proposer"},
                        "steering_actions": {"items": {}, "type": "array"},
                    }
                }
            }
        },
    }
    status, _ = transport(
        "POST",
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": "test-only"},
        json.dumps(request).encode(),
        10.0,
    )
    assert status == 200
    provider_request = json.loads(captured["body"])
    provider_prompt_text = provider_request["messages"][0]["content"]
    provider_prompt = json.loads(provider_prompt_text)
    assert provider_prompt["creative_context"] == context
    evidence = transport.evidence_for("msg_context_test")
    assert evidence["creative_context_injected"] is True
    assert evidence["creative_context_sha256"] == context["content_sha256"]
    assert (
        evidence["provider_prompt_sha256"]
        == hashlib.sha256(provider_prompt_text.encode()).hexdigest()
    )
    assert "test-only" not in json.dumps(evidence, sort_keys=True)
    assert "x-api-key" in captured["headers"]


def test_transport_rejects_a_preexisting_context_slot() -> None:
    context = _context()
    transport = C.FirstPrinciplesContextTransport(
        context, transport=lambda *_: (500, {"type": "error"})
    )
    request = {
        "messages": [
            {
                "content": json.dumps({"creative_context": {}, "instruction": "create"}),
                "role": "user",
            }
        ]
    }
    with pytest.raises(ClaudeCreativityError, match="context slot"):
        transport(
            "POST",
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": "test-only"},
            json.dumps(request).encode(),
            10.0,
        )
