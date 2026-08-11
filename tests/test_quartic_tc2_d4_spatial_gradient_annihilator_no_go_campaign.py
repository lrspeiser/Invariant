from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_spatial_gradient_annihilator_no_go_campaign import (
    QuarticTC2D4SpatialGradientAnnihilatorNoGoError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/backgrounds/quartic_tc2_d4_spatial_gradient_annihilator_no_go_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-spatial-gradient-annihilator-no-go-campaign/campaign.json"
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


def test_polynomial_annihilator_system_is_exhaustive(artifact: dict) -> None:
    exact = artifact["exact_classification"]
    declared = exact["declared_operator_class"]
    system = exact["polynomial_annihilator_system"]
    assert declared["support_exhaustive_within_declared_class"] is True
    assert declared["nonlocal_or_higher_direction_dependence_included"] is False
    assert len(system["coefficient_conditions"]) == 6
    assert system["prototype_rank"] == 5
    assert system["prototype_nullity"] == 1
    assert system["canonical_residual_zero"] is True


def test_fixed_direction_one_slice_forces_the_bad_companion(artifact: dict) -> None:
    affine = artifact["exact_classification"]["exact_affine_solution"]
    assert affine["forced_direction_2_block"] == "B2*E1=-B1*E2"
    assert (
        affine["forced_direction_2_block_sha256"]
        == "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
    )
    assert all(affine["forced_block_checks"].values())


def test_affine_solution_is_exactly_the_c23_family(artifact: dict) -> None:
    affine = artifact["exact_classification"]["exact_affine_solution"]
    assert affine["raw_unknown_coefficients_after_fixing_B1"] == 3630
    assert affine["independent_affine_constraints"] == 3025
    assert affine["raw_affine_dimension"] == 605
    assert affine["free_matrix"] == "A in Mat(55,11)"
    assert affine["equivalent_constraint_family"] == "arbitrary output multiples of C23^[0..10]"


def test_entire_declared_class_misses_axis2_target(artifact: dict) -> None:
    projected = artifact["exact_classification"]["axis2_projected_range"]
    assert projected["effective_parameter_count"] == 363
    assert projected["matrix_shape"] == [528, 363]
    assert projected["range_rank"] == 297
    assert projected["target_augmented_rank"] == 298
    assert projected["target_in_image"] is False


def test_all_candidate_consequences_are_exact_and_scoped(artifact: dict) -> None:
    consequence = artifact["exact_classification"]["candidate_consequence"]
    assert consequence["base_axis2_D4_RHS_identically_zero"] is True
    assert consequence["candidate_conditions_checked"] == 12
    assert consequence["candidate_completions_in_declared_class"] == 0
    assert consequence["candidate_no_go_results"] == 12
    assert consequence["registered_nonzero_eta_values"] == [
        "-1088/15",
        "-34816/15",
        "1088/15",
        "34816/15",
    ]


def test_escape_boundary_is_explicit(artifact: dict) -> None:
    boundary = artifact["exact_classification"]["escape_boundary"]
    assert boundary["outside_gradient_input_columns"] == 22
    assert boundary["outside_gradient_input_column_indices"] == [
        *range(11),
        *range(33, 44),
    ]
    assert boundary["such_an_escape_constructed"] is False


def test_global_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["spatial_gradient_supported_linear_completion_class_exhaustive"] is True
    assert claims["fixed_B1_forces_axis2_companion_block"] is True
    assert claims["all_12_candidates_ruled_out_in_declared_completion_class"] is True
    assert claims["escape_requires_broader_support_or_operator_class"] is True
    for key in (
        "all_topology_changing_completions_ruled_out",
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
        (("counts", "raw_affine_dimension"), 606),
        (("exact_classification", "polynomial_annihilator_system", "prototype_rank"), 4),
        (("exact_classification", "exact_affine_solution", "independent_affine_constraints"), 3024),
        (("exact_classification", "axis2_projected_range", "target_in_image"), True),
        (("exact_classification", "candidate_consequence", "candidate_no_go_results"), 11),
        (("claims", "all_topology_changing_completions_ruled_out"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "claim_target_in_projected_range", "rejected"), False),
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
    with pytest.raises(QuarticTC2D4SpatialGradientAnnihilatorNoGoError):
        validate_campaign(_rehash(mutated))


def test_axis2_binding_tamper_fails_before_classification(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["axis2_base_rhs"]["content_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4SpatialGradientAnnihilatorNoGoError):
        build_campaign(ROOT, path)
