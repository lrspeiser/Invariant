from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_topology_changing_origin_classification_campaign import (
    QuarticTC2D4TopologyChangingOriginClassificationError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_tc2_d4_topology_changing_origin_classification_campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_direct_action_principal_class_has_zero_map(artifact: dict) -> None:
    validate_campaign(artifact)
    direct = artifact["exact_classification"]["direct_action_origin_no_go"]
    assert direct["declared_class"]["joint_domain_dimension"] == 2145
    assert direct["canonical_lift_support"]["R0T_K0_Ev_zero"] is True
    assert direct["deltaP_cokernel_map"]["rank"] == 0
    assert direct["deltaK_cokernel_map"]["rank"] == 0
    assert direct["joint_result"]["target_W_in_image"] is False
    assert direct["joint_result"]["direct_action_principal_origin_ruled_out"] is True


def test_all_canonical_input_selectors_are_exactly_classified(artifact: dict) -> None:
    selector = artifact["exact_classification"][
        "explicit_TC2_selector_classification"
    ]
    counts = selector["canonical_counts"]
    assert counts == {
        "selectors_checked": 55,
        "zero_projection_selectors": 16,
        "nonzero_projection_incapable_selectors": 34,
        "cokernel_capable_selectors": 5,
    }
    assert selector["canonical_capable_indices"] == [21, 44, 48, 51, 53]
    assert selector["canonical_capable_labels"] == [
        "w2[10]",
        "w1[0]",
        "w1[4]",
        "w1[7]",
        "w1[9]",
    ]
    assert 54 in selector["canonical_kernel_indices"]


def test_constructive_selector_controls_reach_only_the_scoped_block(
    artifact: dict,
) -> None:
    selector = artifact["exact_classification"][
        "explicit_TC2_selector_classification"
    ]
    rows = selector["constructive_rank_one_blocks"]
    assert len(rows) == 5
    assert all(row["rank_one_block_rank"] == 1 for row in rows)
    assert all(row["projected_energy_skew_equals_W"] for row in rows)
    assert all(not row["full_equal_eigenspace_compatibility_checked"] for row in rows)
    assert all(not row["covariant_origin_proved"] for row in rows)
    assert selector["registered_selector_control"]["rejected"] is True
    assert selector["minimal_escape_control"]["accepted"] is True


def test_general_selector_condition_and_first_blocker(artifact: dict) -> None:
    exact = artifact["exact_classification"]
    condition = exact["explicit_TC2_selector_classification"][
        "necessary_and_sufficient_condition"
    ]
    assert condition["R0T_kernel_dimension"] == 22
    assert condition["capable_selector_preimage_dimension"] == 24
    assert condition["capable_nonzero_projection_condition"] == "(a,b)!=(0,0)"
    assert "constraint row" in exact["exact_first_blocker"]["statement"]


def test_downstream_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["direct_second_order_action_principal_origin_ruled_out"] is True
    assert claims["all_55_canonical_TC2_input_selectors_classified"] is True
    assert claims["canonical_cokernel_capable_selector_count"] == 5
    for key in (
        "explicit_constraint_row_covariant_origin_constructed",
        "lower_jet_coupled_action_origin_ruled_out",
        "arbitrary_covariant_operator_origin_ruled_out",
        "constraint_propagation_for_topology_change_proved",
        "corrected_candidate_family_registered",
        "remaining_D4_selector_closed",
        "full_tube_Sylvester_identity",
        "CK1_closed",
        "CK3_closed",
        "TC2_closed",
        "B7_closed",
        "global_H7_closed",
        "lifespan_proved",
    ):
        assert claims[key] is False


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("exact_classification", "direct_action_origin_no_go", "joint_result", "joint_map_rank"), 1),
        (("exact_classification", "direct_action_origin_no_go", "joint_result", "target_W_in_image"), True),
        (("exact_classification", "explicit_TC2_selector_classification", "canonical_capable_indices"), [21]),
        (("exact_classification", "explicit_TC2_selector_classification", "constructive_rank_one_blocks", 0, "covariant_origin_proved"), True),
        (("claims", "arbitrary_covariant_operator_origin_ruled_out"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "direct_C12_cross_Hessian_pair", "rejected"), False),
    ],
)
def test_validator_rejects_rehashed_tampering(
    artifact: dict, path: tuple[str | int, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(QuarticTC2D4TopologyChangingOriginClassificationError):
        validate_campaign(_rehash(mutated))


def test_predecessor_tamper_fails_before_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["registered_operator_no_go"]["content_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4TopologyChangingOriginClassificationError):
        build_campaign(ROOT, path)
