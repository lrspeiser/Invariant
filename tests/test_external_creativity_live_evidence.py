from __future__ import annotations

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
    assert evidence["usage"]["total_tokens"] <= 64000
    assert not evidence["claims"]["credential_material_included"]
    assert not evidence["claims"]["model_output_is_verifier_authority"]
    assert not evidence["claims"]["novel_formula_established"]
    assert not evidence["source_receipt"]["raw_prompts_or_outputs_copied"]
    serialized = json.dumps(evidence, sort_keys=True)
    assert "hypotheses" not in serialized
    assert "rationale" not in serialized
    assert "x-api-key" not in serialized


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
