from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_k55_taylor_order_one_consumer import (
    K55TaylorOrderOneConsumerError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_k55_taylor_order_one_consumer.json"
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-k55-taylor-order-one-consumer/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_all_15_reference_packets_pass_exact_replay(artifact: dict) -> None:
    packets = artifact["registered_reference_e1_K55_Taylor_order_one_packets"]
    assert len(packets) == 15
    for packet in packets:
        assert packet["content_sha256"] == _content_hash(packet)
        assert packet["K55_order_one_symmetrizer_residual_nonzero_entries"] == 0
        assert packet["coordinate_free_admissible"] is False


def test_K0_coordinate_free_obstruction_is_decisive(artifact: dict) -> None:
    replay = artifact["flat_K0_spatial_axis_replay"]
    assert replay["axis_residual_nonzero_entries"] == {"n1": 0, "n2": 128, "n3": 128}
    assert replay["coordinate_free_K0_admissible"] is False


def test_manifest_advances_if_and_only_if_full_n_replay_passes(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["coordinate_free_K55_Taylor_order_one_packets_registered"] == 0
    assert counts["registered_symbolic_input_packets"] == 79
    assert counts["missing_symbolic_input_packets"] == 225
    assert artifact["phase_two"]["decision"] == "BLOCK"


def test_broad_claims_remain_false(artifact: dict) -> None:
    for claim in (
        "all_15_coordinate_free_K55_Taylor_order_one_packets_registered",
        "complete_coordinate_free_coefficient_map_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_exact_replay_and_resealed_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["coordinate_free_K55_Taylor_order_one_packets_registered"] = 15
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(K55TaylorOrderOneConsumerError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
