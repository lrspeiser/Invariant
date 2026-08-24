from __future__ import annotations

import copy
import json
from pathlib import Path

from sigma_theory_compiler import external_creativity_live_evidence as V
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_PATH = ROOT / V.OUTPUT_PATH


def test_promoted_live_evidence_is_sealed_and_claim_neutral() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    V.validate_evidence(evidence)
    assert len(evidence["calls"]) == 8
    assert evidence["usage"]["calls"] == 8
    assert evidence["usage"]["total_tokens"] <= 128000
    assert not evidence["claims"]["credential_material_included"]
    assert not evidence["claims"]["model_output_is_verifier_authority"]
    assert not evidence["claims"]["novel_formula_established"]
    assert not evidence["source_receipt"]["raw_prompts_or_outputs_copied"]
    serialized = json.dumps(evidence, sort_keys=True)
    assert "hypotheses" not in serialized
    assert "rationale" not in serialized
    assert "x-api-key" not in serialized


def test_committed_evidence_is_the_validated_core_apps_embedded_evidence() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    core_path = ROOT / V.CORE_PATH
    core = json.loads(core_path.read_text(encoding="utf-8"))
    assert V.build_evidence_from_core_receipt(core, root=ROOT) == evidence
    assert evidence["content_sha256"] == (
        "b13a9da8fd9b8213f6c2e94d91872d3403342f1cddb4be80e7e55e3d3f03bf7e"
    )


def test_live_completion_or_duplicate_provider_response_tamper_rejects() -> None:
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
    changed = copy.deepcopy(evidence)
    changed["claims"]["live_claude_api_campaign_completed"] = False
    changed["content_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "content_sha256"}
    )
    try:
        V.validate_evidence(changed)
    except ValueError as error:
        assert "claim boundary" in str(error)
    else:
        raise AssertionError("incomplete live campaign was accepted")

    changed = copy.deepcopy(evidence)
    changed["calls"][1]["api_response_id"] = changed["calls"][0]["api_response_id"]
    changed["content_sha256"] = canonical_sha256(
        {key: item for key, item in changed.items() if key != "content_sha256"}
    )
    try:
        V.validate_evidence(changed)
    except ValueError as error:
        assert "credential boundary" in str(error)
    else:
        raise AssertionError("duplicate live provider response was accepted")


def test_promoted_evidence_prefers_actual_provider_wire_hashes() -> None:
    call_evidence = {
        "api_response_id": "msg_provider",
        "credential_persisted": False,
        "model": "claude-opus-4-6",
        "model_evidence": {"capabilities_sha256": "a" * 64},
        "output_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "provider_prompt_sha256": "d" * 64,
        "raw_output_sha256": "e" * 64,
        "provider_raw_output_sha256": "f" * 64,
        "request_schema_sha256": "1" * 64,
        "provider_request_schema_sha256": "2" * 64,
        "usage": {"input_tokens": 10, "output_tokens": 5},
        "wire_contract_adapter_used": True,
    }
    source = {
        "schema_version": "fixture",
        "claude": {
            "status": "PASS",
            "completed_calls": 1,
            "required_calls": 1,
            "budget": {"calls": 1, "input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "calls": [
                {
                    "benchmark_id": "benchmark.fixture",
                    "evidence": call_evidence,
                    "role": "critic",
                    "status": "completed",
                }
            ],
        },
        "benchmarks": [
            {
                "benchmark_id": "benchmark.fixture",
                "capability_level": 1,
                "claims": {"known_formula_rediscovered": False},
                "ranked_candidates": [
                    {
                        "candidate_id": "candidate.fixture",
                        "expression": "x",
                        "holdout_loss": "1",
                        "proposer": "deterministic_portfolio",
                    }
                ],
                "target_kind": "bounded_unknown",
                "unique_behaviors": 1,
                "unique_proof_mechanisms": 1,
            }
        ],
    }
    source["content_sha256"] = canonical_sha256(source)
    evidence = V.build_evidence_from_receipt(source, source_file_sha256="3" * 64)

    assert evidence["calls"][0]["prompt_sha256"] == "d" * 64
    assert evidence["calls"][0]["raw_output_sha256"] == "f" * 64
    assert evidence["calls"][0]["request_schema_sha256"] == "2" * 64
    assert evidence["calls"][0]["wire_contract_adapter_used"] is True
    assert "x-api-key" not in json.dumps(evidence, sort_keys=True)
