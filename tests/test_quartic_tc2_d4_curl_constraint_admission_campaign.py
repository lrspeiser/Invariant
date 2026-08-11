from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_curl_constraint_admission_campaign import (
    QuarticTC2D4CurlConstraintAdmissionError,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.quartic_tc2_diagonal_third_jet_campaign import _content_hash

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_tc2_d4_curl_constraint_admission_campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    return build_campaign(ROOT, CONFIG)


def _rehash(document: dict) -> dict:
    document.pop("content_sha256", None)
    document["content_sha256"] = _content_hash(document)
    return document


def test_minimal_v_has_exact_two_direction_curl_completion(artifact: dict) -> None:
    validate_campaign(artifact)
    operator = artifact["exact_admission"]["gauge_fixed_operator"]
    assert operator["source_constraint"] == (
        "C_12^[10]=partial_1(w2[10])-partial_2(w1[10])"
    )
    assert operator["direction_1_block_equals_V"] is True
    assert operator["direction_1_block_rank"] == 1
    assert operator["direction_2_companion_rank"] == 1
    assert operator["direction_3_block_zero"] is True
    assert operator["minimal_direction_block_count"] == 2


def test_directional_operator_annihilates_every_gradient_lift(artifact: dict) -> None:
    equivalence = artifact["exact_admission"]["physical_reduction_equivalence"]
    assert equivalence["directional_operator_times_gradient_lift_zero"] is True
    assert equivalence["constraint_surface_operator_zero"] is True
    assert equivalence["physical_second_order_solutions_unchanged"] is True
    controls = artifact["exact_admission"]["negative_control_residuals"]
    assert controls["omit_direction_2_companion"]["residual_nonzero"] is True
    assert controls["wrong_companion_sign"]["residual_nonzero"] is True
    assert controls["wrong_companion_field"]["residual_nonzero"] is True
    assert controls["exact_polynomial_witness"]["full_curl"] == "0"


def test_definition_and_curl_constraint_propagation_closes(artifact: dict) -> None:
    propagation = artifact["exact_admission"]["constraint_propagation"]
    assert propagation["spatial_output_coefficients"]["rank"] == 3
    definition = propagation["definition_constraint_propagation"]
    curl = propagation["curl_constraint_propagation"]
    assert definition["constraint_count"] == 33
    assert definition["map_rank"] == 1
    assert definition["homogeneous_in_constraints"] is True
    assert curl["constraint_count"] == 33
    assert curl["map_rank"] == 3
    assert curl["homogeneous_in_constraints_and_constraint_derivatives"] is True
    assert "partial_i(eta)" in curl["variable_coefficient_product_rule_closed"]


def test_quartic_coefficient_has_only_the_intended_d4_jet(artifact: dict) -> None:
    coefficient = artifact["exact_admission"]["coefficient_jet"]
    assert coefficient["orders_0_through_3_zero"] is True
    assert coefficient["ordered_lower_derivatives_checked"] == 85
    assert coefficient["ordered_fourth_derivatives_checked"] == 256
    assert coefficient["nonzero_ordered_fourth_derivatives"] == 24
    assert coefficient["canonical_D_0_D_2_D_3_D_9_value"] == "1"


def test_all_candidate_reference_tunings_are_bound_but_not_promoted(
    artifact: dict,
) -> None:
    reference = artifact["exact_admission"]["reference_D4_binding"]
    rows = reference["candidate_specializations"]
    assert len(rows) == 12
    assert all(row["reference_direction_D4_Sylvester_solvable"] for row in rows)
    assert all(row["gauge_fixed_curl_constraint_realization"] for row in rows)
    assert all(not row["covariant_action_origin"] for row in rows)
    assert all(not row["all_spatial_directions_checked"] for row in rows)


def test_global_and_covariant_claims_remain_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["minimal_V_gauge_fixed_curl_constraint_realized"] is True
    assert claims["canonical_constraint_surface_invariance_proved"] is True
    assert claims["candidate_reference_D4_admission_count"] == 12
    for key in (
        "covariant_action_origin_constructed",
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
        (("exact_admission", "gauge_fixed_operator", "direction_1_block_equals_V"), False),
        (("exact_admission", "gauge_fixed_operator", "minimal_direction_block_count"), 1),
        (("exact_admission", "physical_reduction_equivalence", "directional_operator_times_gradient_lift_zero"), False),
        (("exact_admission", "constraint_propagation", "curl_constraint_propagation", "map_rank"), 2),
        (("claims", "covariant_action_origin_constructed"), True),
        (("claims", "TC2_closed"), True),
        (("negative_controls", "omit_direction_2_companion", "rejected"), False),
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
    with pytest.raises(QuarticTC2D4CurlConstraintAdmissionError):
        validate_campaign(_rehash(mutated))


def test_topology_predecessor_tamper_fails_before_replay(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["topology_classification"]["content_sha256"] = "0" * 64
    config.pop("content_sha256")
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "tampered-config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(QuarticTC2D4CurlConstraintAdmissionError):
        build_campaign(ROOT, path)
