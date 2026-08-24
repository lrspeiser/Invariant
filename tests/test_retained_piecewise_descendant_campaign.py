from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import retained_piecewise_descendant_campaign as D
from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"
FIXTURE_SECRET = "fixture-secret-never-persisted"


def _piecewise(expression: str) -> str:
    return json.dumps(
        {
            "branches": [
                {
                    "condition": {"comparator": "ge", "left": "x0", "right": "0"},
                    "expression": expression,
                }
            ],
            "default_expression": "0",
        },
        sort_keys=True,
    )


def _hypothesis(
    benchmark_id: str, role: str, ordinal: int, representation: str, expression: str
) -> dict[str, Any]:
    origins = (
        "known_rewrite",
        "cross_domain_synthesis",
        "proposed_new_construction",
        "uncertain",
    )
    return {
        "expression": expression,
        "falsifiers": ["open the fresh sealed evaluation rows"],
        "family": f"fixture_{role}_{ordinal}",
        "hypothesis_id": f"fixture.{benchmark_id}.{role}.{ordinal}",
        "invariants": ["integer-indexed exact relation"],
        "known_analogues": ["fixture analogue"],
        "llm_origin_assessment": origins[ordinal - 1],
        "proof_plan": ["exact replay", "seek an induction invariant"],
        "rationale": "Exercise retention, exact evaluation, and matched controls.",
        "representation": representation,
        "source_idea_domains": ["recurrences", "finite differences"],
        "synthesis_note": "Recombine parent_1 with parent_2 while retaining both branches.",
    }


class DescendantProviderFixture:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.post_count = 0

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        parsed = None if body is None else json.loads(body)
        self.requests.append(
            {
                "body": parsed,
                "headers": dict(headers),
                "method": method,
                "timeout": timeout,
                "url": url,
            }
        )
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": MODEL,
                "type": "model",
            }
        assert parsed is not None
        prompt = json.loads(parsed["messages"][0]["content"])
        benchmark_id = prompt["benchmark"]["fresh_task"]["task_id"]
        role = prompt["role"]
        representations = (
            ("piecewise_relation", _piecewise("x0")),
            ("piecewise_relation", _piecewise("x0 + 1")),
            ("sympy_expression", "x0"),
            ("other_typed_relation", "a deliberately retained uncompiled relation"),
        )
        hypotheses = [
            _hypothesis(benchmark_id, role, ordinal, representation, expression)
            for ordinal, (representation, expression) in enumerate(representations, 1)
        ]
        output = {
            "benchmark_id": benchmark_id,
            "hypotheses": hypotheses,
            "role": role,
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": {},
        }
        self.post_count += 1
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg_descendant_fixture_{self.post_count}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }


@pytest.fixture
def fixture_receipt(tmp_path: Path) -> tuple[dict[str, Any], DescendantProviderFixture]:
    provider = DescendantProviderFixture()
    receipt = D.run_live(
        ROOT,
        transport=provider,
        environment={"ANTHROPIC_API_KEY": FIXTURE_SECRET},
        home=tmp_path,
    )
    return receipt, provider


def test_fake_live_campaign_retains_24_descendants_and_preserves_all_parents(
    fixture_receipt: tuple[dict[str, Any], DescendantProviderFixture],
) -> None:
    receipt, provider = fixture_receipt
    assert receipt["claude_runtime"]["authenticated_messages_api_working"] is True
    assert receipt["claude_runtime"]["completed_calls"] == 6
    assert receipt["summary"]["descendant_ideas_retained"] == 24
    assert receipt["summary"]["admitted_executable_descendants"] == 18
    assert receipt["summary"]["nonexecutable_descendants_retained"] == 6
    assert receipt["summary"]["parent_branches_exposed"] == 8
    assert receipt["summary"]["parent_branches_preserved"] == 8
    assert receipt["summary"]["descendants_with_explicit_parent_lineage"] == 24
    assert len(receipt["parent_prediction_archive"]) == 8
    assert provider.post_count == 6
    post_requests = [request for request in provider.requests if request["method"] == "POST"]
    assert all(
        request["body"]["output_config"]["format"]["schema"]["properties"]["hypotheses"]
        == {
            "type": "array",
            "items": request["body"]["output_config"]["format"]["schema"]["properties"][
                "hypotheses"
            ]["items"],
            "minItems": 1,
        }
        for request in post_requests
    )
    assert all(
        call["provider_transport"]["wire_contract_adapter_used"] is True
        for call in receipt["claude_calls"]
    )
    assert FIXTURE_SECRET not in json.dumps(receipt, sort_keys=True)


def test_generation_view_is_target_blind_but_includes_opened_counterexamples(
    fixture_receipt: tuple[dict[str, Any], DescendantProviderFixture],
) -> None:
    receipt, _ = fixture_receipt
    encoded_payloads = json.dumps(
        [call["public_payload"] for call in receipt["claude_calls"]], sort_keys=True
    )
    assert "OEIS-A000330" not in encoded_payloads
    assert "OEIS-A005132" not in encoded_payloads
    assert "OEIS-A002858" not in encoded_payloads
    assert "oeis.org" not in encoded_payloads
    assert "known_reference_formula" not in encoded_payloads
    assert "holdout" not in encoded_payloads
    assert all(
        sum(parent["prior_counterexample"] is not None for parent in pool) == 8
        for pool in (
            call["public_payload"]["retained_parent_pool"] for call in receipt["claude_calls"]
        )
    )


def test_exact_evaluators_and_controls_agree_for_every_admitted_descendant(
    fixture_receipt: tuple[dict[str, Any], DescendantProviderFixture],
) -> None:
    receipt, _ = fixture_receipt
    admitted = [item for item in receipt["descendants"] if "execution" in item]
    assert admitted
    assert all(
        item["execution"]["primary_independent_exact_agreement"]
        and item["execution"]["resource_profile_exact_match"]
        and item["retention_status"] == "RETAINED_ACTIVE"
        for item in admitted
    )
    assert receipt["novelty_axes"]["separate_axes"] is True
    assert receipt["novelty_axes"]["behavior_novelty_is_literature_novelty"] is False


def test_rebind_reuses_stored_calls_without_provider_access(
    fixture_receipt: tuple[dict[str, Any], DescendantProviderFixture],
) -> None:
    receipt, provider = fixture_receipt
    request_count = len(provider.requests)
    assert D.rebind_receipt(ROOT, receipt) == receipt
    assert len(provider.requests) == request_count


def test_resealed_origin_label_cannot_be_promoted_to_novelty(
    fixture_receipt: tuple[dict[str, Any], DescendantProviderFixture],
) -> None:
    receipt, _ = fixture_receipt
    changed = copy.deepcopy(receipt)
    changed["claim_boundary"]["llm_origin_assessment_establishes_novelty"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(D.RetainedPiecewiseDescendantError, match="policy"):
        D.validate_receipt(changed, ROOT)


def test_stored_live_receipt_reproduces_exactly() -> None:
    path = ROOT / D.OUTPUT_PATH
    if not path.is_file():
        pytest.skip("live descendant receipt has not been generated yet")
    stored = json.loads(path.read_text(encoding="utf-8"))
    D.validate_receipt(stored, ROOT)
    assert stored == D.rebind_receipt(ROOT, stored)
