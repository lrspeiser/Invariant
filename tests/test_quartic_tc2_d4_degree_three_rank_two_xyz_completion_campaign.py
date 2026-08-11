from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_degree_three_rank_two_xyz_completion_campaign import (
    QuarticTC2D4DegreeThreeRankTwoXYZCompletionError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/backgrounds/quartic_tc2_d4_degree_three_rank_two_xyz_completion_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-degree-three-rank-two-xyz-completion-campaign/campaign.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _with_hash(body: dict) -> dict:
    result = copy.deepcopy(body)
    result["content_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


@pytest.fixture(scope="session")
def artifact() -> dict:
    value = build_campaign(ROOT, CONFIG)
    validate_campaign(value)
    assert value == json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return value


def test_exact_counts(artifact: dict) -> None:
    assert artifact["counts"] == {
        "bound_predecessors": 3,
        "directional_recurrence_evaluations": 15,
        "prior_candidate_obstructions": 12,
        "normalized_target_rank": 4,
        "transverse_selector_rank": 22,
        "minimal_completion_rank": 2,
        "new_curl_channels": 2,
        "new_candidate_direction_systems_evaluated": 12,
        "new_candidate_direction_compatibilities": 12,
        "new_candidate_direction_obstructions": 0,
        "total_certified_directions": 5,
        "remaining_declared_directions": 0,
        "negative_controls": 8,
        "inferred_global_passes": 0,
    }


def test_final_selector_closed(artifact: dict) -> None:
    selector = artifact["exact_completion"]["selector"]
    assert selector == {
        "frame_name": "xyz_1_2_2",
        "direction": ["1/3", "2/3", "2/3"],
        "final_declared_rational_frame": True,
        "remaining_declared_frames": 0,
    }
    assert artifact["selector_binding"]["remaining_declared_directions"] == 0


def test_prior_symbol_is_exactly_obstructed(artifact: dict) -> None:
    prior = artifact["exact_completion"]["prior_combined_symbol_audit"]
    assert prior["directional_evaluations"] == 15
    assert prior["all_seven_eigenspaces_checked_per_candidate"] is True
    assert prior["base_D4_RHS_nonzero_entries"] == 116
    assert prior["base_D4_RHS_sha256"] == (
        "e137be6d8bb6aaafdc45d12c79adc8c2b9e5e37ef511a5e4414b1661c0a0a0b7"
    )
    assert prior["prior_global_symbol_rank"] == 2
    assert prior["prior_global_symbol_sha256"] == (
        "59c6f074bd1ade330630e6f607e587b92ec7a1c11d2ce061b54079ea3571936b"
    )
    assert prior["candidate_compatibilities"] == 0
    assert prior["candidate_obstructions"] == 12
    assert len(prior["candidate_records"]) == 12
    for row in prior["candidate_records"]:
        assert row["D4_Sylvester_solvable"] is False
        assert set(row["nonzero_equal_eigenspace_compressions"]) == {"0"}
        zero = row["nonzero_equal_eigenspace_compressions"]["0"]
        assert zero["rank"] == 4
        assert zero["nonzero_entries"] == 56


def test_exact_range_classification(artifact: dict) -> None:
    exact_range = artifact["exact_completion"]["exact_range_classification"]
    assert exact_range["eta_normalized_targets"] == 12
    assert exact_range["distinct_eta_normalized_targets"] == 1
    assert exact_range["normalized_target_rank"] == 4
    assert exact_range["normalized_target_nonzero_entries"] == 56
    assert exact_range["normalized_target_sha256"] == (
        "767724a8936ceefbbeea530d0a64be0fa94c47decabede12e061223b71f73ab7"
    )
    assert exact_range["transverse_selector_rank"] == 22
    assert exact_range["selector_sha256"] == (
        "7ef398226365b9e42bd543a3b9c5b00c82621cbf8f67d76b2768e38e81441d26"
    )
    assert exact_range["target_plane_dimension"] == 4
    assert exact_range["selector_target_plane_intersection_dimension"] == 4
    assert exact_range["quotient_target_zero"] is True
    assert exact_range["target_in_full_transverse_curl_range"] is True


def test_rank_two_completion_is_sharp(artifact: dict) -> None:
    completion = artifact["exact_completion"]["minimal_rank_two_completion"]
    assert completion["rank_one_impossible_from_skew_rank_bound"] is True
    assert completion["lower_bound_completion_rank"] == 2
    assert completion["constructed_completion_rank"] == 2
    assert completion["elementary_curl_channels"] == 2
    assert completion["coordinate_pairs"] == [[11, 21], [15, 32]]
    assert completion["aligned_block_sha256"] == (
        "a415ea5ac18d5f43ab57103295df2935acb58e92bb1c532fed97e6ff11bba7b3"
    )
    assert completion["global_block_sha256"] == (
        "b091a2194f13b1b58bbd441a18773659bad91251752c403d7c297aff4a83f3ad"
    )
    assert len(completion["term_records"]) == 2


def test_exact_sphere_extension(artifact: dict) -> None:
    extension = artifact["exact_completion"]["exact_sphere_extension"]
    assert extension["envelope"] == "a_xyz(n)=(9/4)*n2*n3"
    assert extension["envelope_value_at_xyz"] == "1"
    assert extension["minimal_total_degree"] == 3
    assert extension["antipodally_odd"] is True
    assert extension["polynomial_and_smooth_on_S2"] is True
    assert extension["bounded_on_S2"] is True
    assert extension["prior_four_direction_extensions_zero"] is True
    assert extension["physical_gradient_lift_annihilated_identically"] is True
    assert extension["symbol_nonzero_entries"] == 63
    assert extension["symbol_sha256"] == (
        "cff78832e0ffe582290b200702afa857719fc6a47d5d0f3c7871e9125cb0cb0b"
    )
    assert extension["gradient_residual_sha256"] == (
        "54efa54b8d23c5fdf1e357239619821b52ed647990182ca8b3fd3e1cb57916f9"
    )


def test_all_corrected_candidates_are_compatible(artifact: dict) -> None:
    result = artifact["exact_completion"]["corrected_xyz_result"]
    assert result["candidate_conditions_checked"] == 12
    assert result["candidate_compatibilities"] == 12
    assert result["candidate_obstructions"] == 0
    assert len(result["candidate_records"]) == 12
    assert all(row["D4_Sylvester_solvable"] is True for row in result["candidate_records"])
    assert all(
        row["nonzero_equal_eigenspace_compressions"] == {} for row in result["candidate_records"]
    )


def test_claims_are_closed_world_and_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    true_claims = {
        "full_xyz_orders_one_through_four_recurrence_evaluated",
        "prior_combined_symbol_xyz_obstructed_all_12_candidates",
        "rank_four_target_in_full_transverse_curl_range",
        "minimal_rank_two_completion_constructed",
        "all_12_corrected_xyz_D4_compatibilities_proved",
        "all_five_declared_direction_certificates_closed",
    }
    false_claims = {
        "finite_selector_determines_full_direction_sphere",
        "full_direction_sphere_D4_compatibility_proved",
        "broader_matrix_curl_symbol_class_classified",
        "local_differential_operator_origin_proved",
        "covariant_action_origin_proved",
        "variable_coefficient_constraint_calculus_proved",
        "boundary_energy_admission_proved",
        "corrected_candidate_family_registered",
        "remaining_D4_selector_closed",
        "full_tube_Sylvester_identity",
        "CK1_closed",
        "CK3_closed",
        "TC2_closed",
        "B7_closed",
        "global_H7_closed",
        "lifespan_proved",
    }
    assert set(claims) == true_claims | false_claims
    assert all(claims[key] is True for key in true_claims)
    assert all(claims[key] is False for key in false_claims)


def test_negative_controls(artifact: dict) -> None:
    controls = artifact["negative_controls"]
    assert len(controls) == 8
    assert all(control["rejected"] is True for control in controls.values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "minimal_completion_rank"), 1),
        (("counts", "remaining_declared_directions"), 1),
        (("claims", "full_direction_sphere_D4_compatibility_proved"), True),
        (("claims", "TC2_closed"), True),
        (
            ("exact_completion", "exact_range_classification", "normalized_target_sha256"),
            "0" * 64,
        ),
        (("exact_completion", "minimal_rank_two_completion", "global_block_sha256"), "0" * 64),
    ],
)
def test_rehashed_semantic_tamper_rejected(
    artifact: dict, path: tuple[str, ...], value: object
) -> None:
    tampered = copy.deepcopy(artifact)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    tampered = _with_hash({key: item for key, item in tampered.items() if key != "content_sha256"})
    with pytest.raises(
        QuarticTC2D4DegreeThreeRankTwoXYZCompletionError,
        match="exact/fail-closed mismatch",
    ):
        validate_campaign(tampered)


def test_rehashed_unknown_claim_rejected(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["claims"]["theory_pass"] = True
    tampered = _with_hash({key: item for key, item in tampered.items() if key != "content_sha256"})
    with pytest.raises(
        QuarticTC2D4DegreeThreeRankTwoXYZCompletionError,
        match="exact/fail-closed mismatch",
    ):
        validate_campaign(tampered)


def test_predecessor_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["c23_predecessor"]["content_sha256"] = "0" * 64
    config = _with_hash({key: value for key, value in config.items() if key != "content_sha256"})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        QuarticTC2D4DegreeThreeRankTwoXYZCompletionError,
        match="bound input mismatch",
    ):
        build_campaign(ROOT, path)


def test_raw_source_and_test_bindings() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("campaign_source", "campaign_test"):
        path = ROOT / config[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]["file_sha256"]
