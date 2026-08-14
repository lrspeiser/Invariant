from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_tc2_taylor_order_zero_registration import (
    TC2TaylorOrderZeroRegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("configs/backgrounds/quartic_tc2_d4_tc2_taylor_order_zero_registration.json")
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-tc2-taylor-order-zero-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_exact_unit_TC2_packet_is_sparse_linear_and_reference_bound(
    artifact: dict,
) -> None:
    exact = artifact["exact_TC2_order_zero_construction"]
    packet = exact["unit_matrix"]
    assert packet["shape"] == [55, 55]
    assert packet["nonzero_linear_coefficient_count"] == 8
    assert packet["distinct_output_rows"] == 8
    assert packet["right_support_columns"] == [54]
    assert packet["scalar_prefactor"] == "a10"
    assert exact["reference_e1_match"] is True
    assert exact["axis_certificates"][0]["legacy_dense_matrix_sha256"] == (
        "f439b4f7952f43600bd7078ba6f767f2de57de09a5a4477362a2eabe0aed967b"
    )


def test_all_15_TC2_order_zero_packets_bind_same_exact_unit_matrix(
    artifact: dict,
) -> None:
    packets = artifact["registered_TC2_Taylor_order_zero_packets"]
    matrix_hash = artifact["exact_TC2_order_zero_construction"]["unit_matrix"]["content_sha256"]
    assert len(packets) == 15
    assert len({packet["evaluation_id"] for packet in packets}) == 15
    for packet in packets:
        assert packet["Taylor_order"] == 0
        assert packet["unit_TC2_order_zero_content_sha256"] == matrix_hash
        assert packet["scalar_prefactor"] == "a10"
        assert packet["content_sha256"] == _content_hash(packet)


def test_manifest_advances_exactly_from_49_to_64(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 49
    assert counts["new_TC2_Taylor_order_zero_packets_registered"] == 15
    assert counts["registered_symbolic_input_packets"] == 64
    assert counts["missing_symbolic_input_packets"] == 240
    manifest = {row["input_id"]: row for row in artifact["required_symbolic_input_manifest"]}
    family = manifest["polarized_TC2_Taylor_packets"]
    assert family["registered_packets"] == 15
    assert family["registered_Taylor_orders"] == [0]
    assert family["missing_Taylor_orders"] == [1, 2, 3, 4]
    assert family["packet_content_sha256"] == [
        row["content_sha256"] for row in artifact["registered_TC2_Taylor_order_zero_packets"]
    ]


def test_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["counts"]["full_symbol_build_calls"] == 0
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"]["decision"] == "BLOCK"
    for claim in (
        "TC2_Taylor_orders_one_through_four_registered",
        "complete_coordinate_free_coefficient_map_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_exact_replay_and_resealed_semantic_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["registered_symbolic_input_packets"] = 65
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(TC2TaylorOrderZeroRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
