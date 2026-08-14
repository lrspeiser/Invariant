from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_p55_taylor_order_one_registration import (
    P55TaylorOrderOneRegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("configs/backgrounds/quartic_tc2_d4_p55_taylor_order_one_registration.json")
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-p55-taylor-order-one-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_all_15_materialized_packets_are_exactly_registered(artifact: dict) -> None:
    packets = artifact["registered_P55_Taylor_order_one_packets"]
    assert len(packets) == 15
    assert len({packet["evaluation_id"] for packet in packets}) == 15
    for packet in packets:
        assert packet["content_sha256"] == _content_hash(packet)
        assert packet["exact_linearity_from_basis_packets"] is True
        matrix = packet["P55_Taylor_order_one_matrix"]
        assert matrix["content_sha256"] == _content_hash(matrix)
        assert matrix["Taylor_order"] == 1
        assert matrix["shape"] == [55, 55]


def test_manifest_advances_exactly_from_64_to_79(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 64
    assert counts["new_P55_Taylor_order_one_packets_registered"] == 15
    assert counts["registered_symbolic_input_packets"] == 79
    assert counts["missing_symbolic_input_packets"] == 225
    records = {row["input_id"]: row for row in artifact["required_symbolic_input_manifest"]}
    family = records["polarized_P55_Taylor_packets"]
    assert family["registered_packets"] == 30
    assert family["registered_Taylor_orders"] == [0, 1]
    assert family["missing_Taylor_orders"] == [2, 3, 4]
    assert len(family["packet_content_sha256"]) == 30


def test_sparse_packet_census_is_nontrivial_and_bound(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["P55_Taylor_order_one_distinct_matrix_cells"] > 0
    assert counts["P55_Taylor_order_one_linear_coefficients"] > 0
    assert counts["P55_Taylor_order_one_packets_with_all_three_axes"] > 0
    assert counts["materializer_basis_jet_packets"] == 4
    assert counts["materializer_basis_axis_matrices"] == 12


def test_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["counts"]["consumer_full_symbol_build_calls"] == 0
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"]["decision"] == "BLOCK"
    for claim in (
        "P55_Taylor_orders_two_through_four_registered",
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
    tampered["counts"]["registered_symbolic_input_packets"] = 80
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(P55TaylorOrderOneRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
