from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_registered_operator_origin_no_go_campaign import (
    QuarticTC2D4RegisteredOperatorOriginNoGoError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_tc2_d4_registered_operator_origin_no_go_campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_broad_registered_support_class_has_zero_cokernel_image(
    artifact: dict,
) -> None:
    validate_campaign(artifact)
    exact = artifact["exact_no_go"]
    support = exact["constraint_support_audit"]
    induced = exact["induced_cokernel_map"]
    assert support["fixed_high_covector_times_zero_projector_zero"] is True
    assert support["escape_input_covector_times_zero_projector_zero"] is False
    assert induced["domain_dimension"] == 55
    assert induced["rank"] == 0
    assert induced["target_in_image"] is False
    assert induced["augmented_rank"] == 1


def test_all_registered_blocks_are_covered_and_project_to_zero(artifact: dict) -> None:
    rows = artifact["exact_no_go"]["registered_block_checks"]
    assert {row["name"] for row in rows} == {
        "reference_column_7",
        "reference_column_9",
        "reference_column_10",
        "variable_column_10",
    }
    assert all(row["right_support_columns"] == [54] for row in rows)
    assert all(row["zero_eigenspace_compression_zero"] for row in rows)


def test_e21_algebraic_escape_is_sensitive_positive_control(artifact: dict) -> None:
    exact = artifact["exact_no_go"]
    support = exact["constraint_support_audit"]
    positive = exact["positive_control"]
    assert support["support_intersection_empty"] is True
    assert positive["V_right_support_columns"] == [21]
    assert positive["energy_skew_equals_W"] is True
    assert positive["projected_energy_skew_equals_W"] is True
    assert positive["accepted"] is True


def test_scope_and_global_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["registered_linear_X_quartic_Horndeski_TC2_origin_ruled_out"] is True
    assert claims["registered_support_preserving_gauge_deformation_ruled_out"] is True
    for key in (
        "arbitrary_covariant_quartic_operator_ruled_out",
        "arbitrary_gauge_fixed_operator_deformation_ruled_out",
        "covariant_realization_constructed",
        "constraint_topology_changing_realization_constructed",
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
        (("exact_no_go", "induced_cokernel_map", "rank"), 1),
        (("exact_no_go", "induced_cokernel_map", "target_in_image"), True),
        (("exact_no_go", "constraint_support_audit", "registered_right_support_columns"), [21]),
        (("exact_no_go", "positive_control", "energy_skew_equals_W"), False),
        (("claims", "arbitrary_covariant_quartic_operator_ruled_out"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "silently_replace_e54_by_e21", "rejected"), False),
    ],
)
def test_validator_rejects_rehashed_tampering(
    artifact: dict, path: tuple[str, ...], value: object
) -> None:
    mutated = copy.deepcopy(artifact)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(QuarticTC2D4RegisteredOperatorOriginNoGoError):
        validate_campaign(_rehash(mutated))


def test_source_binding_tamper_fails_before_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["registered_action_spec"]["file_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4RegisteredOperatorOriginNoGoError):
        build_campaign(ROOT, path)
