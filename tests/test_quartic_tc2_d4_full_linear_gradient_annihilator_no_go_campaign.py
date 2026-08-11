from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    QuarticTC2D4FullLinearGradientAnnihilatorNoGoError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/backgrounds/quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-full-linear-gradient-annihilator-no-go-campaign/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_committed_artifact_replays_exactly(artifact: dict) -> None:
    validate_campaign(artifact)
    assert artifact == json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_full_affine_class_includes_all_55_input_columns(artifact: dict) -> None:
    exact = artifact["exact_classification"]
    declared = exact["declared_operator_class"]
    affine = exact["full_affine_dimension"]
    assert declared["full_state_input_columns"] == 55
    assert declared["fixed_direction_1_block"] is True
    assert affine == {
        "raw_B2_B3_coefficients": 6050,
        "independent_spatial_gradient_constraints": 3025,
        "affine_dimension": 3025,
        "spatial_C23_freedom": 605,
        "qv_freedom_in_B2": 1210,
        "qv_freedom_in_B3": 1210,
        "axis2_relevant_free_B2_parameters": 1815,
    }


def test_all_22_qv_canonical_selectors_are_partitioned(artifact: dict) -> None:
    partition = artifact["exact_classification"]["canonical_qv_selector_partition"]
    assert partition["counts"] == {
        "selectors_checked": 22,
        "zero_projection_selectors": 11,
        "nonzero_incapable_selectors": 11,
        "capable_selectors": 0,
    }
    assert partition["kernel_indices"] == list(range(33, 44))
    assert partition["nonzero_incapable_indices"] == list(range(11))
    assert partition["capable_indices"] == []


def test_entire_qv_subspace_misses_target(artifact: dict) -> None:
    qv = artifact["exact_classification"]["qv_subspace_range"]
    assert qv["canonical_input_count"] == 22
    assert qv["selector_projection_rank"] == 11
    assert qv["selector_projection_kernel_dimension"] == 11
    assert qv["selector_target_plane_intersection_dimension"] == 0
    assert qv["effective_projected_parameters"] == 363
    assert qv["wedge_range_rank"] == 297
    assert qv["target_augmented_rank"] == 298
    assert qv["target_in_image"] is False
    assert qv["quotient_target_rank"] == 2


def test_combined_qv_and_c23_range_still_misses_target(artifact: dict) -> None:
    combined = artifact["exact_classification"]["combined_axis2_free_B2_range"]
    assert combined["canonical_input_count"] == 33
    assert combined["selector_projection_rank"] == 22
    assert combined["selector_target_plane_intersection_dimension"] == 0
    assert combined["effective_projected_parameters"] == 726
    assert combined["wedge_range_rank"] == 473
    assert combined["target_augmented_rank"] == 474
    assert combined["target_in_image"] is False
    assert combined["quotient_target_rank"] == 2


def test_candidate_no_go_and_escape_boundary_are_exact(artifact: dict) -> None:
    exact = artifact["exact_classification"]
    consequence = exact["candidate_consequence"]
    boundary = exact["escape_boundary"]
    assert consequence["base_axis2_D4_RHS_identically_zero"] is True
    assert consequence["candidate_conditions_checked"] == 12
    assert consequence["candidate_linear_gradient_annihilator_completions"] == 0
    assert consequence["candidate_no_go_results"] == 12
    assert (
        boundary["all_55_input_columns_classified_within_linear_gradient_annihilator_class"] is True
    )
    assert len(boundary["remaining_escape_options"]) == 4
    assert boundary["such_an_escape_constructed"] is False


def test_global_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["all_22_qv_canonical_selectors_classified"] is True
    assert claims["qv_selector_subspace_completion_ruled_out"] is True
    assert claims["combined_qv_and_C23_axis2_completion_ruled_out"] is True
    assert claims["full_linear_gradient_annihilator_completion_class_ruled_out"] is True
    for key in (
        "all_operator_classes_ruled_out",
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
        (("counts", "canonical_qv_capable_selectors"), 1),
        (("exact_classification", "canonical_qv_selector_partition", "capable_indices"), [0]),
        (("exact_classification", "qv_subspace_range", "target_in_image"), True),
        (("exact_classification", "qv_subspace_range", "wedge_range_rank"), 296),
        (("exact_classification", "combined_axis2_free_B2_range", "target_augmented_rank"), 473),
        (("exact_classification", "candidate_consequence", "candidate_no_go_results"), 11),
        (("claims", "all_operator_classes_ruled_out"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "analyze_qv_and_C23_ranges_separately_only", "rejected"), False),
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
    with pytest.raises(QuarticTC2D4FullLinearGradientAnnihilatorNoGoError):
        validate_campaign(_rehash(mutated))


def test_spatial_predecessor_binding_tamper_fails_before_classification(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["spatial_gradient_no_go"]["content_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4FullLinearGradientAnnihilatorNoGoError):
        build_campaign(ROOT, path)
