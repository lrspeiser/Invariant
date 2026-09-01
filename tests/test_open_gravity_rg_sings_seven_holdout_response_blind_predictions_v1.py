from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_rg_sings_seven_holdout_response_blind_predictions_v1 as package,
)


def test_config_freezes_four_laws_no_fit_and_zero_response() -> None:
    config = package.load_config()
    assert config["candidate_contract"]["candidate_ids"] == [
        "NEWTON_3D_DST",
        "RAR_2016_ON_NEWTON_3D",
        "MOND_STANDARD_MU_ON_NEWTON_3D",
        "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
    ]
    assert config["candidate_contract"]["per_object_fitted_parameters"] == 0
    assert config["candidate_contract"]["global_fitted_parameters"] == 0
    assert all(value == 0 for value in config["response_boundary"].values())


def test_package_bindings_are_nonzero_and_shared_by_outputs() -> None:
    bindings = package._package_bindings()
    assert set(bindings) == {
        "config_raw_sha256",
        "config_content_sha256",
        "module_semantic_sha256",
        "test_raw_sha256",
    }
    assert all(len(value) == 64 and value != "0" * 64 for value in bindings.values())
    receipt = package.build_receipt(package.load_config())
    assert receipt["package_bindings"] == bindings


@pytest.mark.parametrize(
    "mutation",
    [
        ("candidate_contract", "a0_m_s2", 1.3e-10),
        ("candidate_contract", "global_fitted_parameters", 1),
        ("grid_contract", "fine_nodes_per_axis", 129),
        ("execution_contract", "expected_built_source_cells", 23),
        ("response_boundary", "velocity_values_opened", 1),
        ("claim_boundary", "refracted_gravity_preferred", True),
    ],
)
def test_material_config_mutations_fail_closed(mutation: tuple[str, str, object]) -> None:
    config = package.load_config()
    changed = copy.deepcopy(config)
    changed[mutation[0]][mutation[1]] = mutation[2]
    with pytest.raises(package.PredictionBuildError):
        package.validate_config(changed)


def test_predecessors_freeze_27_cells_24_built_and_three_failures() -> None:
    config = package.load_config()
    predecessors = package._load_predecessors(config)
    source = predecessors["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    assert source["source_cell_count"] == 27
    assert source["built_source_map_count"] == 24
    assert source["failed_source_conversion_count"] == 3
    cells = package._built_source_cells(source)
    assert len(cells) == 24
    assert len({package.cell_run_id(row) for row in cells}) == 24


def test_source_cell_arrays_are_exact_source_only_roles() -> None:
    config = package.load_config()
    source = package._load_predecessors(config)["SEVEN_HOLDOUT_SOURCE_BUILDER"]
    first = package._built_source_cells(source)[0]
    arrays = package._load_source_arrays(source, first)
    assert set(arrays) == {
        "stellar_surface_msun_pc2",
        "hi_surface_msun_pc2",
        "co_surface_msun_pc2",
        "x_pc",
        "y_pc",
    }
    assert {value.shape for value in arrays.values()} == {(192, 192)}


def test_candidate_controls_are_monotone_and_fixed() -> None:
    config = package.load_config()
    newton = [
        {
            "radius_kpc": 1.0,
            "radial_acceleration_m_s2": 1.0e-11,
        }
    ]
    refracted = [
        {
            "radius_kpc": 1.0,
            "radial_acceleration_m_s2": 1.5e-11,
        }
    ]
    profiles = package._candidate_profiles(config, newton, refracted)
    assert profiles["NEWTON_3D_DST"][0]["radial_acceleration_m_s2"] == 1.0e-11
    assert (
        profiles["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"][0]["radial_acceleration_m_s2"]
        == 1.5e-11
    )
    assert profiles["RAR_2016_ON_NEWTON_3D"][0]["radial_acceleration_m_s2"] > 1.0e-11
    assert profiles["MOND_STANDARD_MU_ON_NEWTON_3D"][0]["radial_acceleration_m_s2"] > 1.0e-11
    round_tripped = json.loads(package.canonical_bytes({"profiles": profiles}))["profiles"]
    assert set(round_tripped) == set(config["candidate_contract"]["candidate_ids"])


def _synthetic_grid(config: dict[str, object], scale: float) -> dict[str, object]:
    radii = package._radii(config)
    profiles = {
        candidate_id: [
            {"radius_kpc": radius, "radial_acceleration_m_s2": scale * (1.0 + radius)}
            for radius in radii
        ]
        for candidate_id in config["candidate_contract"]["candidate_ids"]
    }
    return {
        "profiles": profiles,
        "dimensionless_mass_relative_error": 0.0,
        "solver_metrics": {
            "newton_relative_residual": 0.0,
            "refracted_gravity": {"relative_residual": 0.0},
        },
    }


def test_numerical_mask_retains_failed_radii() -> None:
    config = package.load_config()
    fine = _synthetic_grid(config, 1.0)
    close = _synthetic_grid(config, 1.01)
    far = _synthetic_grid(config, 2.0)
    close_mask = package._numerical_mask(config, fine, close)
    far_mask = package._numerical_mask(config, fine, far)
    assert close_mask["eligible_radius_count"] == 291
    assert close_mask["failed_radius_count"] == 0
    assert far_mask["eligible_radius_count"] == 0
    assert far_mask["failed_radius_count"] == 291


def test_status_is_in_progress_without_response_access() -> None:
    state = package.status()
    assert state["status"] in {
        "IN_PROGRESS_RESPONSE_BLIND_CELL_PREDICTIONS",
        "PASS_RESPONSE_BLIND_ALL_CELL_PREDICTIONS_BUILT",
    }
    assert all(value == 0 for value in state["response_boundary"].values())


def test_write_receipt_refuses_until_all_cells_exist() -> None:
    config = package.load_config()
    receipt = package.build_receipt(config)
    if receipt["missing_prediction_cells"]:
        with pytest.raises(package.PredictionBuildError, match="remain incomplete"):
            package.write_receipt()


def test_atomic_no_clobber_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert package._atomic_no_clobber(output, b"one") == "CREATED"
    assert package._atomic_no_clobber(output, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(package.PredictionBuildError, match="differs"):
        package._atomic_no_clobber(output, b"two")
    assert output.read_bytes() == b"one"


def test_receipt_content_hash_rejects_mutation() -> None:
    config = package.load_config()
    receipt = package.build_receipt(config)
    changed = json.loads(json.dumps(receipt))
    changed["response_boundary"]["scores_computed"] = 1
    changed["content_sha256"] = package.content_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(package.PredictionBuildError, match="response leak"):
        package.validate_receipt(config, changed)


def test_package_seals_match_current_files() -> None:
    if package._CONFIG_RAW_SHA256 == "0" * 64:
        pytest.skip("package pins not yet sealed")
    assert (
        package.file_sha256(package._repo_path(package.CONFIG_PATH)) == package._CONFIG_RAW_SHA256
    )
    assert package.content_sha256(package.load_config()) == package._CONFIG_CONTENT_SHA256
    assert (
        package.module_semantic_sha256(package._repo_path(package.MODULE_PATH))
        == package._MODULE_SEMANTIC_SHA256
    )
    assert package.file_sha256(package._repo_path(package.TEST_PATH)) == package._TEST_RAW_SHA256
