from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_k55_taylor_order_zero_registration import (
    K55TaylorOrderZeroRegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_k55_taylor_order_zero_registration.json"
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-k55-taylor-order-zero-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_exact_K0_has_all_entrywise_residuals_zero(artifact: dict) -> None:
    exact = artifact["exact_K0_construction"]
    assert exact["K0"]["shape"] == [55, 55]
    assert exact["K0"]["nonzero_count"] > 0
    assert len(exact["projector_packets"]) == 6
    for key, value in exact.items():
        if key.endswith("residual_nonzero_entries") and not key.startswith("omit_"):
            assert value == 0
    assert exact["omit_cross_block_symmetrizer_nonzero_entries"] > 0


def test_all_15_K55_order_zero_packets_bind_same_K0(artifact: dict) -> None:
    packets = artifact["registered_K55_Taylor_order_zero_packets"]
    K0_hash = artifact["exact_K0_construction"]["K0"]["content_sha256"]
    assert len(packets) == 15
    assert len({packet["evaluation_id"] for packet in packets}) == 15
    for packet in packets:
        assert packet["Taylor_order"] == 0
        assert packet["K0_content_sha256"] == K0_hash
        assert packet["content_sha256"] == _content_hash(packet)


def test_manifest_advances_exactly_from_34_to_49(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 34
    assert counts["new_K55_Taylor_order_zero_packets_registered"] == 15
    assert counts["registered_symbolic_input_packets"] == 49
    assert counts["missing_symbolic_input_packets"] == 255
    manifest = {row["input_id"]: row for row in artifact["required_symbolic_input_manifest"]}
    family = manifest["polarized_K55_Taylor_packets"]
    assert family["registered_packets"] == 15
    assert family["registered_Taylor_orders"] == [0]
    assert family["missing_Taylor_orders"] == [1, 2, 3, 4]


def test_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["counts"]["full_symbol_build_calls"] == 0
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"]["decision"] == "BLOCK"
    for claim in (
        "K55_Taylor_orders_one_through_four_registered",
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
    tampered["counts"]["registered_symbolic_input_packets"] = 50
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(K55TaylorOrderZeroRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
