from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_matrix_curl_rank_one_completion_campaign import (
    QuarticTC2D4MatrixCurlRankOneCompletionError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import _with_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/backgrounds/quartic_tc2_d4_matrix_curl_rank_one_completion_campaign.json"
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-matrix-curl-rank-one-completion-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    value = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(value)
    return value


def test_checked_artifact_replays_exactly(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact


def test_full_generic_recurrence_is_replayed(artifact: dict) -> None:
    replay = artifact["exact_completion"]["full_D4_replay"]
    assert replay["directional_evaluations"] == 15
    assert replay["orders_1_through_3_mandatory_prerequisites"] is True
    assert replay["all_seven_eigenspaces_checked"] is True
    assert replay["base_D4_RHS_nonzero_entries"] == 64
    assert replay["eta_normalized_candidate_targets"] == 12
    assert replay["distinct_eta_normalized_targets"] == 1
    assert replay["normalized_target_rank"] == 2


def test_complete_transverse_curl_range_contains_target(artifact: dict) -> None:
    declared = artifact["exact_completion"]["declared_completion_class"]
    exact = artifact["exact_completion"]["exact_range_classification"]
    assert declared["curl_covector_dimension"] == 22
    assert declared["raw_matrix_parameter_dimension"] == 1210
    assert declared["single_fixed_direction_only"] is True
    assert exact["selector_projection_rank"] == 22
    assert exact["selector_target_plane_intersection_dimension"] == 2
    assert exact["wedge_range_rank"] == 473
    assert exact["target_augmented_rank"] == 473
    assert exact["target_in_image"] is True
    assert exact["quotient_target_zero"] is True


def test_completion_is_minimal_rank_one_and_constraint_supported(artifact: dict) -> None:
    construction = artifact["exact_completion"]["minimal_rank_one_completion"]
    assert construction["minimal_nonzero_block_rank"] == 1
    assert construction["rank_zero_impossible_because_target_nonzero"] is True
    assert construction["constructed_block_rank"] == 1
    assert construction["combined_curl_channel_count"] == 1
    assert construction["global_block_rank"] == 1
    assert construction["gradient_lift_annihilation_exact"] is True
    assert construction["zero_speed_target_cancelled_exactly"] is True
    assert construction["all_nonzero_eigenspace_compressions_zero"] is True


def test_all_candidates_are_compatible_at_fixed_frame(artifact: dict) -> None:
    result = artifact["exact_completion"]["candidate_result"]
    assert result["candidate_conditions_checked"] == 12
    assert result["candidate_compatibilities"] == 12
    assert result["candidate_obstructions"] == 0
    assert len(result["candidate_records"]) == 12
    assert all(row["D4_Sylvester_solvable"] for row in result["candidate_records"])
    assert all(
        row["nonzero_equal_eigenspace_compressions"] == {} for row in result["candidate_records"]
    )


def test_next_blocker_and_global_claims_stay_fail_closed(artifact: dict) -> None:
    blocker = artifact["exact_completion"]["first_blocker"]
    assert blocker["name"] == "global_angular_extension_and_constraint_admission"
    claims = artifact["claims"]
    for key in (
        "global_smooth_angular_extension_constructed",
        "additional_generic_directions_audited",
        "pseudodifferential_constraint_calculus_proved",
        "local_differential_operator_origin_proved",
        "covariant_action_origin_proved",
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
    assert artifact["counts"]["inferred_global_passes"] == 0


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("counts", "target_augmented_rank"), 474),
        (("counts", "constructed_block_rank"), 2),
        (("counts", "candidate_compatibilities"), 11),
        (("exact_completion", "exact_range_classification", "target_in_image"), False),
        (
            (
                "exact_completion",
                "minimal_rank_one_completion",
                "gradient_lift_annihilation_exact",
            ),
            False,
        ),
        (("claims", "global_smooth_angular_extension_constructed"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "infer_global_angular_extension", "rejected"), False),
    ],
)
def test_rehashed_semantic_tampering_is_rejected(
    artifact: dict, path: tuple[str, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(artifact)
    cursor = tampered
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    tampered = _with_hash(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(QuarticTC2D4MatrixCurlRankOneCompletionError):
        validate_campaign(tampered)


def test_tampered_obstruction_predecessor_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["generic_direction_obstruction"]["content_sha256"] = "0" * 64
    config = _with_hash({key: value for key, value in config.items() if key != "content_sha256"})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4MatrixCurlRankOneCompletionError):
        build_campaign(ROOT, path)


def test_config_source_and_test_are_hash_bound() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("campaign_source", "campaign_test"):
        path = ROOT / config[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]["file_sha256"]
