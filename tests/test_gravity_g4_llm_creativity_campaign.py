"""Controls for the budget-capped G4 LLM creativity campaign."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g4_llm_creativity_campaign import (
    FAILURE_OUTPUT_PATH,
    OUTPUT_PATH,
    GravityG4LlmCreativityError,
    build_failure_receipt,
    build_prompt,
    build_provider_request,
    build_receipt,
    load_config,
    validate_failure_receipt,
    validate_proposals,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _proposals() -> dict[str, Any]:
    labels = _config()["proposal_contract"]["origin_labels"]
    return {
        "schema_version": "g4-creative-proposals-1.0",
        "proposals": [
            {
                "proposal_id": f"g4-mock-family-{index:02d}",
                "title": f"Mock family {index}",
                "origin_self_assessment": labels[index % len(labels)],
                "known_analogue": "A disclosed mock analogue",
                "mechanism": "A typed mock mechanism for testing the quarantine boundary.",
                "equation_template": "g=g_bar+a0*c0*F(r/r0)",
                "variables": ["g_bar: acceleration", "r/r0: dimensionless"],
                "universal_parameters": ["c0", "r0"],
                "why_not_merely_a_rewrite": "The mock asserts an additional operator.",
                "expected_observational_signature": "A mock radial signature.",
                "cheapest_falsifier": "Reject on a held-out mock curve.",
                "likely_failure_mode": "The signature does not transfer.",
                "creativity_score": 3,
                "physical_plausibility_score": 2,
                "testability_score": 5,
            }
            for index in range(12)
        ],
    }


def _provider(
    _payload: dict[str, Any], _secret: str, config: dict[str, Any]
) -> dict[str, Any]:
    return {
        "content": [{"type": "text", "text": json.dumps(_proposals())}],
        "id": "msg_mock_0001",
        "model": config["provider"]["model"],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 2000, "output_tokens": 5000},
    }


def test_live_contract_is_one_call_bounded_and_confirmation_blind() -> None:
    config = _config()
    assert config["failure_context_binding"]["required_decision"] == (
        "BLOCK_G4_PHOTOMETRIC_CONSTRUCTION"
    )
    assert config["provider"]["model"] == "claude-opus-5"
    assert config["provider"]["maximum_calls"] == 4
    assert config["provider"]["maximum_call_usd"] == "5.000000"
    assert config["provider"]["maximum_campaign_usd"] == "20.000000"
    assert config["attempt_audit_before_final_retry"]["completed_inferences"] == 4
    assert config["attempt_audit_before_final_retry"]["usable_proposal_outputs"] == 0
    assert config["campaign_closed"] is True
    assert config["data_boundary"]["confirmation_evaluator_accesses_allowed"] == 0
    assert config["proposal_contract"]["historical_novelty_may_be_claimed"] is False


def test_prompt_and_request_expose_only_aggregate_failure_geometry() -> None:
    prompt = build_prompt(_config())
    assert "139 galaxies" in prompt
    assert "91,349.139" in prompt
    assert "observed target velocities" in prompt
    request = build_provider_request(_config(), prompt)
    assert request["model"] == "claude-opus-5"
    assert request["max_tokens"] == 20_000
    assert request["output_config"] == {"effort": "high"}


def test_mocked_call_is_quarantined_and_never_validated() -> None:
    receipt = build_receipt(ROOT, provider=_provider, secret="mock-secret")
    assert receipt["decision"] == "QUARANTINE_G4_LLM_PROPOSALS_FOR_TYPED_TESTING"
    assert receipt["counts"] == {
        "completed_inference_calls": 4,
        "confirmation_evaluator_accesses": 0,
        "provider_http_requests": 10,
        "proposals": 12,
    }
    assert receipt["claims"]["empirical_validation_completed"] is False
    assert receipt["claims"]["historical_novelty_established"] is False
    assert float(receipt["provider"]["conservative_usage_cost_ceiling_usd"]) <= 5.0


def test_unknown_origin_label_fails_closed() -> None:
    value = _proposals()
    value["proposals"][0]["origin_self_assessment"] = "definitely_novel"
    with pytest.raises(GravityG4LlmCreativityError, match="origin label"):
        validate_proposals(value, _config())


def test_exhausted_live_campaign_seals_an_honest_failure() -> None:
    receipt = build_failure_receipt(ROOT)
    validate_failure_receipt(receipt, root=ROOT)
    assert receipt["decision"] == "BLOCK_G4_LLM_CREATIVITY_CAMPAIGN_ENGINEERING_FAILURE"
    assert receipt["counts"]["completed_inference_calls"] == 4
    assert receipt["counts"]["usable_proposals"] == 0
    assert receipt["claims"]["historical_novelty_established"] is False


def test_checked_llm_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("live G4 LLM creativity call has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_llm_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("live G4 LLM creativity call has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["historical_novelty_established"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4LlmCreativityError, match="overstates novelty"):
        validate_receipt(tampered, root=ROOT)


def test_checked_llm_failure_receipt_is_sealed_if_present() -> None:
    path = ROOT / FAILURE_OUTPUT_PATH
    if not path.is_file():
        pytest.skip("G4 LLM failure receipt has not been sealed")
    validate_failure_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)
