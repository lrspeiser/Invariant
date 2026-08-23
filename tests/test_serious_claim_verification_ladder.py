from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler import serious_claim_verification_ladder as L
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / L.OUTPUT_PATH


def _reseal_stage(stage: dict) -> None:
    body = {key: item for key, item in stage.items() if key != "content_sha256"}
    stage["content_sha256"] = canonical_sha256(body)


def _reseal_chain(chain: dict) -> None:
    body = {key: item for key, item in chain.items() if key != "content_sha256"}
    chain["content_sha256"] = canonical_sha256(body)


def _reseal_receipt(receipt: dict) -> None:
    body = {key: item for key, item in receipt.items() if key != "content_sha256"}
    receipt["content_sha256"] = canonical_sha256(body)


def test_stored_ladder_is_candidate_bound_ordered_and_fail_closed() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    L.validate_receipt(value, ROOT)
    chain = value["known_control_chain"]
    assert [stage["backend"] for stage in chain["stages"]] == list(L.REQUIRED_STAGES)
    assert chain["stages"][0]["previous_stage_sha256"] is None
    for left, right in zip(chain["stages"], chain["stages"][1:]):
        assert right["previous_stage_sha256"] == left["content_sha256"]
    assert value["summary"] == {
        "known_control_candidates": 2,
        "negative_controls_blocked": 2,
        "required_stage_order": list(L.REQUIRED_STAGES),
        "structural_mutations_rejected": 5,
        "status": "PASS_CANDIDATE_BOUND_LADDER_CALIBRATION",
    }
    assert value["release_gate"]["serious_claims_released"] == 0
    assert not value["claims"]["serious_claim_released"]


def test_bounded_unknowns_do_not_acquire_a_partial_ladder_release() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert {item["benchmark_id"] for item in value["negative_controls"]} == {
        "external.authority-oeis-005132",
        "external.authority-oeis-002858",
    }
    for item in value["negative_controls"]:
        assert {"cas", "smt", "interval", "lean"}.issubset(
            item["missing_or_failed_backends"]
        )
        assert item["status"] == "BLOCKED_INCOMPLETE_BACKEND_LADDER"
        assert item["serious_claim_released"] is False


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_stage",
        "reordered_stages",
        "candidate_scope_substitution",
        "broken_previous_stage_link",
        "backend_unavailable",
    ],
)
def test_semantically_resealed_chain_mutations_fail(mutation: str) -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    chain = value["known_control_chain"]
    if mutation == "missing_stage":
        del chain["stages"][2]
    elif mutation == "reordered_stages":
        chain["stages"][1], chain["stages"][2] = chain["stages"][2], chain["stages"][1]
    elif mutation == "candidate_scope_substitution":
        chain["stages"][3]["candidate_scope_sha256"] = "0" * 64
        _reseal_stage(chain["stages"][3])
    elif mutation == "broken_previous_stage_link":
        chain["stages"][4]["previous_stage_sha256"] = "0" * 64
        _reseal_stage(chain["stages"][4])
    elif mutation == "backend_unavailable":
        chain["stages"][4]["backend_available"] = False
        _reseal_stage(chain["stages"][4])
    _reseal_chain(chain)
    _reseal_receipt(value)
    with pytest.raises(L.SeriousClaimVerificationError):
        L.validate_receipt(value)


def test_current_sources_rebuild_the_stored_receipt_exactly() -> None:
    stored = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert L.build_receipt(ROOT) == stored
