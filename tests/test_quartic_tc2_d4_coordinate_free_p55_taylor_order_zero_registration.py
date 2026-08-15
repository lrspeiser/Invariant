from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_p55_taylor_order_zero_registration import (
    P55TaylorOrderZeroRegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/quartic_tc2_d4_coordinate_free_p55_taylor_order_zero_registration.json"
)
ARTIFACT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-p55-taylor-order-zero-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_all_15_order_zero_packets_are_exact_and_evaluation_bound(artifact: dict) -> None:
    evaluations = artifact["polarization_evaluations"]
    packets = artifact["registered_P55_Taylor_order_zero_packets"]
    assert len(evaluations) == len(packets) == 15
    assert len({row["evaluation_id"] for row in evaluations}) == 15
    evaluation_hashes = {row["content_sha256"] for row in evaluations}
    matrix_hash = artifact["coordinate_free_P55_order_zero_matrix"]["content_sha256"]
    for packet in packets:
        assert packet["Taylor_order"] == 0
        assert packet["evaluation_content_sha256"] in evaluation_hashes
        assert packet["coordinate_free_matrix_content_sha256"] == matrix_hash
        assert packet["content_sha256"] == _content_hash(packet)


def test_coordinate_free_matrix_consumes_exactly_three_axis_packets(artifact: dict) -> None:
    matrix = artifact["coordinate_free_P55_order_zero_matrix"]
    assert matrix["shape"] == [55, 55]
    assert matrix["identity"] == "P(n)=n1*P_1+n2*P_2+n3*P_3"
    assert len(matrix["axis_packet_content_sha256"]) == 3
    assert matrix["axis_sparse_entries_consumed"] == 144
    assert matrix["entries"]
    assert all(
        set(entry["linear_coefficients"]).issubset({"n1", "n2", "n3"})
        for entry in matrix["entries"]
    )


def test_manifest_advances_to_34_and_leaves_270(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 19
    assert counts["new_P55_Taylor_order_zero_packets_registered"] == 15
    assert counts["registered_symbolic_input_packets"] == 34
    assert counts["missing_symbolic_input_packets"] == 270
    manifest = {row["input_id"]: row for row in artifact["required_symbolic_input_manifest"]}
    family = manifest["polarized_P55_Taylor_packets"]
    assert family["registered_packets"] == 15
    assert family["registered_Taylor_orders"] == [0]
    assert family["missing_Taylor_orders"] == [1, 2, 3, 4]


def test_missing_contract_is_exact_and_does_not_infer_derivatives(artifact: dict) -> None:
    contract = artifact["missing_P55_Taylor_serialization_contract"]
    assert contract["status"] == "MISSING_REQUIRED_SERIALIZATION"
    assert contract["required_output_packets"] == 60
    assert contract["minimal_direct_contract"]["matrix_packets"] == 60
    assert contract["required_Taylor_orders"] == [1, 2, 3, 4]
    assert len(contract["evaluation_content_sha256"]) == 15
    assert len(contract["forbidden_inferences"]) == 4


def test_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"] == {
        "decision": "BLOCK",
        "admitted": False,
        "attempted": False,
        "blocker": "270 required symbolic input packets remain unregistered",
    }
    for claim in (
        "P55_Taylor_orders_one_through_four_registered",
        "complete_coordinate_free_coefficient_map_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_replay_and_semantic_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["registered_symbolic_input_packets"] = 35
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(P55TaylorOrderZeroRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
