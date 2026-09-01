from __future__ import annotations

import copy
from functools import lru_cache

import pytest

from sigma_theory_compiler import (
    open_gravity_lane6_same_grid_nonspherical_predictions_v1 as lane,
)


@lru_cache(maxsize=1)
def _built() -> dict:
    config = lane.load_config()
    return lane.build_receipt(config)


def test_frozen_config_and_package_pins() -> None:
    config = lane.load_config()
    assert config["source_contract"]["source_cells"] == 225
    assert config["source_contract"]["measured_3d_objects"] == 0
    assert lane.file_sha256(lane._repo_path(lane.CONFIG_PATH)) == lane._CONFIG_RAW_SHA256
    assert lane.content_sha256(config) == lane._CONFIG_CONTENT_SHA256
    assert (
        lane.module_semantic_sha256(lane._repo_path(lane.MODULE_PATH))
        == lane._MODULE_SEMANTIC_SHA256
    )
    assert lane.file_sha256(lane._repo_path(lane.TEST_PATH)) == lane._TEST_RAW_SHA256


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "OPEN"),
        (("source_contract", "geometry_label"), "MEASURED_3D"),
        (("source_contract", "measured_3d_objects"), 3),
        (("source_contract", "response_opening_forbidden"), False),
        (("projection_contract", "kinematic_response_used"), True),
        (("access_contract", "scientific_response_rows"), 1),
        (("output_path",), "elsewhere.json"),
    ],
)
def test_config_mutations_are_rejected(path: tuple[str, ...], value: object) -> None:
    config = lane.load_config(verify_package=False)
    mutated = copy.deepcopy(config)
    target = mutated
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(lane.Lane6NonsphericalError):
        lane.validate_config(mutated)


def test_target_free_fixture_suite_covers_required_geometries() -> None:
    suite = lane.run_fixture_suite()
    assert suite["failed"] == 0
    assert suite["passed"] == 9
    assert set(suite["gates"]) == {
        "PUBLISHED_Q0_POINT_FORCE_IDENTITY",
        "GQNS_HELMHOLTZ_MANUFACTURED",
        "GQNS_EXACT_SPHERICAL_SHUTOFF",
        "BAR_BRANCH_ACTIVE",
        "SPIRAL_BRANCH_ACTIVE",
        "VERTICAL_THICKNESS_ORDERING",
        "SADDLE_EXACT_NULL",
        "UNIFORM_EXTERNAL_FIELD_SUPERPOSITION",
        "ROTATION_COVARIANCE",
    }
    assert suite["gates"]["GQNS_EXACT_SPHERICAL_SHUTOFF"]["metrics"]["anisotropy_A_Q"] < 1e-14


def test_all_225_cells_and_eight_mechanisms_are_retained() -> None:
    receipt = _built()
    assert receipt["cell_count"] == 225
    assert len(receipt["cells"]) == 225
    assert len({(row["object_id"], row["cell_id"]) for row in receipt["cells"]}) == 225
    assert [row["id"] for row in receipt["mechanisms"]] == [
        "NEWTON",
        "NFW_SOURCE_MATCHED_CONTROL",
        "AQUAL_SIMPLE_MU",
        "QUMOND_SIMPLE_NU",
        "REFRACTED_GRAVITY_DISKMASS_MEDIAN",
        "GP01_ELLIPTIC_N2_L035",
        "MASHHOON_RAHVAR_NLG_Q0",
        "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE",
    ]
    expected = {row["id"] for row in receipt["mechanisms"]}
    assert all(set(row["profiles"]) == expected for row in receipt["cells"])


def test_every_real_source_is_labeled_model_lifted_not_measured_3d() -> None:
    receipt = _built()
    assert receipt["source_contract"]["measured_3d_objects"] == 0
    assert {row["geometry_label"] for row in receipt["cells"]} == {
        "MODEL_LIFTED_2P5D_SOURCE_IN_FULL_3D_FIELD_SOLVER"
    }
    assert receipt["source_readiness"]["MEASURED_3D"] == "SOURCE_BLOCKED_ZERO_OBJECTS"


def test_primary_predictions_cover_each_development_object_once() -> None:
    receipt = _built()
    assert receipt["primary_cell_count"] == 3
    assert {row["object_id"] for row in receipt["primary_predictions"]} == {
        "NGC2903",
        "NGC3351",
        "NGC3627",
    }
    assert all(row["primary_cell"] is True for row in receipt["primary_predictions"])


def test_density_and_newton_replay_are_exact_for_all_cells() -> None:
    receipt = _built()
    assert all(row["gates"]["density_hash_replay"] for row in receipt["cells"])
    assert max(row["newton_profile_replay_relative"] for row in receipt["cells"]) == 0.0


def test_predecessor_failure_is_retained_not_erased() -> None:
    receipt = _built()
    assert receipt["retained_counterexample_count"] >= 1
    assert any(
        "predecessor_source_gate" in row["failed_gates"]
        for row in receipt["retained_counterexamples"]
    )
    assert "WITH_RETAINED_COUNTEREXAMPLES" in receipt["status"]


def test_common_source_systematic_envelopes_cover_all_mechanisms() -> None:
    receipt = _built()
    envelopes = receipt["source_systematic_envelopes"]
    assert len(envelopes) == 3 * 8 * 3
    assert all(row["status"] == "COMMON_SOURCE_SYSTEMATIC_ENVELOPE" for row in envelopes)
    assert all(1 <= row["valid_common_cell_count"] <= 72 for row in envelopes)


def test_new_branch_is_zero_fit_and_geometry_conditioned() -> None:
    receipt = _built()
    mechanism = next(
        row
        for row in receipt["mechanisms"]
        if row["id"] == "GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE"
    )
    assert mechanism["parameters"]["free_fitted_parameters"] == 0
    assert mechanism["parameters"]["historical_novelty_status"] == "UNASSESSED_NEW_AUDIT_SYNTHESIS"
    amplitudes = [
        row["solver_metrics"]["GQNS_GEOMETRY_CONDITIONED_NONLOCAL_SOURCE"]["anisotropy_A_Q"]
        for row in receipt["primary_predictions"]
    ]
    assert all(0.0 < value < 1.0 for value in amplitudes)


def test_published_nonlocal_q0_is_not_mislabeled_as_new_branch() -> None:
    receipt = _built()
    q0 = next(row for row in receipt["mechanisms"] if row["id"] == "MASHHOON_RAHVAR_NLG_Q0")
    assert q0["parameters"]["alpha_0"] == 10.94
    assert q0["parameters"]["mu_0_kpc_inverse"] == 0.059
    neighbor = receipt["nearest_neighbor_boundary"][0]
    assert neighbor["neighbor"] == "MASHHOON_RAHVAR_NLG_Q0"
    assert "vanishes exactly for a sphere" in neighbor["difference"]


def test_projection_contains_bar_spiral_and_vertical_discriminators() -> None:
    receipt = _built()
    for cell in receipt["primary_predictions"]:
        for profiles in cell["profiles"].values():
            assert len(profiles) == 3
            for point in profiles:
                assert {f"m{mode}_over_mean" for mode in range(1, 5)} <= point.keys()
                if "vertical_one_cell_rms_over_a0" in point:
                    assert point["vertical_one_cell_rms_over_a0"] >= 0.0
    assert any("thin-versus-thick" in value for value in receipt["unique_geometry_discriminators"])


def test_access_and_claim_boundaries_remain_zero_response() -> None:
    receipt = _built()
    assert receipt["access_contract"] == {
        "scientific_response_files": 0,
        "scientific_response_rows": 0,
        "scores": 0,
        "parameters_fit": 0,
        "network_calls_by_builder": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "development_only": True,
    }
    assert "observational preference" in receipt["claim_boundary"]["does_not_establish"]
    assert "historical novelty of GQNS" in receipt["claim_boundary"]["does_not_establish"]


def test_next_falsifier_is_an_independent_matched_geometry_pair() -> None:
    receipt = _built()
    falsifier = receipt["next_real_data_falsifier"]
    assert "round unbarred disk" in falsifier["primary"]
    assert "strongly barred" in falsifier["primary"]
    assert "unopened matched" in falsifier["confirmation_requirement"]
