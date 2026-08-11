from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_curl_companion_range_campaign import (
    QuarticTC2D4CurlCompanionRangeError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_curl_companion_range_campaign.json"


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_axis2_reference_is_exact_rotation_and_direct_replay(artifact: dict) -> None:
    validate_campaign(artifact)
    reference = artifact["exact_companion_audit"]["axis_2_reference"]
    assert reference["axis_swap_orthogonal"] is True
    assert reference["direct_axis_2_P55_matches_rotation"] is True
    assert len(reference["spectrum"]) == 7


def test_companion_has_one_rank_two_equal_eigenspace_obstruction(
    artifact: dict,
) -> None:
    audit = artifact["exact_companion_audit"]["equal_eigenspace_audit"]
    assert audit["eigenspaces_checked"] == 7
    assert audit["nonzero_compression_count"] == 1
    assert audit["sole_nonzero_eigenvalue"] == "0"
    assert audit["companion_compression_rank"] == 2
    assert audit["companion_compression_nonzero_entries"] == 10
    assert audit["companion_block_alone_Sylvester_compatible"] is False
    assert sum(not row["zero"] for row in audit["records"]) == 1


def test_rotated_direction1_witness_does_not_cancel_companion(artifact: dict) -> None:
    control = artifact["exact_companion_audit"]["rotation_control"]
    assert control["rotated_direction_1_W_rank"] == 2
    assert control["companion_and_rotated_W_span_dimension"] == 2
    assert control["companion_is_rotated_W_multiple"] is False
    assert control["simple_rotation_of_direction_1_certificate_cancels_companion"] is False


def test_complete_pure_curl_preserving_range_cannot_cancel_target(
    artifact: dict,
) -> None:
    audit = artifact["exact_companion_audit"]["pure_curl_completion_range"]
    declared = audit["declared_completion_class"]
    exact_map = audit["exact_range_map"]
    assert declared["raw_parameter_dimension"] == 605
    assert declared["effective_parameter_count"] == 363
    assert exact_map["matrix_shape"] == [528, 363]
    assert exact_map["rank"] == 297
    assert exact_map["target_augmented_rank"] == 298
    assert exact_map["target_in_image"] is False
    assert audit["result"]["pure_curl_self_compatible_completion_exists"] is False


def test_all_candidate_companion_witnesses_are_nonzero_but_scoped(
    artifact: dict,
) -> None:
    rows = artifact["exact_companion_audit"]["candidate_companion_witnesses"]
    assert len(rows) == 12
    assert len({row["companion_correction_compression_sha256"] for row in rows}) == 4
    assert all(row["companion_correction_compression_rank"] == 2 for row in rows)
    assert all(row["companion_correction_compression_nonzero_entries"] == 10 for row in rows)
    assert all(not row["companion_correction_alone_Sylvester_compatible"] for row in rows)
    assert all(not row["full_base_D4_RHS_evaluated"] for row in rows)


def test_full_base_rhs_and_global_claims_remain_fail_closed(artifact: dict) -> None:
    condition = artifact["exact_companion_audit"]["necessary_full_D4_condition"]
    assert condition["candidate_conditions"] == 12
    assert condition["base_D4_RHS_computed"] is False
    assert condition["condition_verified"] is False
    assert condition["condition_refuted"] is False
    claims = artifact["claims"]
    assert claims["axis2_companion_all_eigenspaces_audited"] is True
    assert claims["pure_curl_self_compatible_completion_ruled_out"] is True
    for key in (
        "full_axis2_base_D4_RHS_evaluated",
        "full_axis2_D4_compatibility_proved",
        "full_axis2_D4_obstruction_proved",
        "spatially_covariant_tensor_completion_proved",
        "all_spatial_direction_compatibility_proved",
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
        (("exact_companion_audit", "equal_eigenspace_audit", "companion_compression_rank"), 0),
        (("exact_companion_audit", "pure_curl_completion_range", "exact_range_map", "target_augmented_rank"), 297),
        (("exact_companion_audit", "pure_curl_completion_range", "exact_range_map", "target_in_image"), True),
        (("exact_companion_audit", "necessary_full_D4_condition", "base_D4_RHS_computed"), True),
        (("claims", "full_axis2_D4_obstruction_proved"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "cancel_with_additional_C23_curl_constraints", "rejected"), False),
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
    with pytest.raises(QuarticTC2D4CurlCompanionRangeError):
        validate_campaign(_rehash(mutated))


def test_curl_admission_binding_tamper_fails_before_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["curl_admission"]["content_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4CurlCompanionRangeError):
        build_campaign(ROOT, path)
