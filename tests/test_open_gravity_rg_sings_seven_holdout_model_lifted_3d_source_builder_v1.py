from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_rg_sings_seven_holdout_model_lifted_3d_source_builder_v1 as package,
)


def test_config_freezes_source_paper_benchmark_and_response_gates() -> None:
    config = package.load_config()
    assert config["admission_rule"] == {
        "real_public_source_data_required": True,
        "primary_measurement_papers_required": True,
        "independent_known_answer_benchmark_required": True,
        "standard_gravity_controls_required_downstream": True,
        "missing_source_disposition": "SOURCE_BLOCKED",
        "paper_only_disposition": "THEORY_BENCHMARK_ONLY",
        "projected_source_plus_assumed_vertical_disposition": "MODEL_LIFTED_2P5D",
        "spherical_or_one_dimensional_data_proves_general_3d": False,
    }
    assert all(value == 0 for value in config["response_boundary"].values())


def test_holmberg_geometry_uncertainty_is_retained() -> None:
    config = package.load_config()
    row = next(value for value in config["objects"] if value["object_id"] == "UGC04305")
    assert row["primary_inclination_deg"] == 38.0
    assert row["inclination_cells_deg"] == [27.0, 38.0, 49.0]


@pytest.mark.parametrize(
    "mutation",
    [
        ("admission_rule", "real_public_source_data_required", False),
        ("admission_rule", "primary_measurement_papers_required", False),
        ("admission_rule", "independent_known_answer_benchmark_required", False),
        ("claim_boundary", "general_3d_gravity_validated", True),
        ("response_boundary", "velocity_values_opened", 1),
    ],
)
def test_material_config_mutations_fail_closed(mutation: tuple[str, str, object]) -> None:
    config = package.load_config()
    changed = copy.deepcopy(config)
    changed[mutation[0]][mutation[1]] = mutation[2]
    with pytest.raises(package.SourceBuildError, match="config semantics changed"):
        package.validate_config(changed)


def test_fastica_exact_two_component_reconstruction() -> None:
    rng = np.random.default_rng(7)
    star = rng.lognormal(mean=0.0, sigma=0.5, size=5000)
    dust = rng.lognormal(mean=-0.5, sigma=0.8, size=5000)
    stellar_color = -0.1
    dust_color = 0.6
    f36 = star + dust
    f45 = star * package._ratio45_over36(stellar_color) + dust * package._ratio45_over36(dust_color)
    mixing, _iterations, residual = package._fastica_two_source(
        np.vstack([f36, f45]), stellar_seed_color=-0.08, dust_seed_color=0.6
    )
    colors = sorted(package._component_colors(mixing))
    recovered_star, recovered_dust = package._decompose(f36, f45, colors[0], colors[1])
    assert residual <= 1.0e-10
    assert np.max(np.abs(recovered_star + recovered_dust - f36)) < 1.0e-12


def test_geometry_centers_are_finite_and_unique() -> None:
    config = package.load_config()
    coordinates = {(float(row["ra_deg"]), float(row["dec_deg"])) for row in config["objects"]}
    assert len(coordinates) == 7
    assert all(math.isfinite(value) for pair in coordinates for value in pair)


def test_exact_source_inventory_is_available_without_velocity_data() -> None:
    config = package.load_config()
    acquisition, irac2 = package._load_contracts(config)
    paths = package._source_paths(acquisition, irac2)
    assert len(paths) == 56
    assert not any("MOM1" in role or "MOM2" in role for _object_id, role in paths)


def test_benchmark_and_full_packet_are_deterministic() -> None:
    first, first_payloads = package.build_packet()
    assert len(first_payloads) == 120
    assert first["object_count"] == 7
    assert first["conversion_cell_count"] == 3
    assert first["geometry_cell_count"] == 9
    assert first["source_cell_count"] == 27
    assert first["built_source_map_count"] == 24
    assert first["failed_source_conversion_count"] == 3
    assert first["private_array_file_count"] == 120
    assert {
        row["object_id"]
        for row in first["source_cells"]
        if row["disposition"] == "SOURCE_CONVERSION_FAILED_UNPHYSICAL_FASTICA_COLOR_RETAINED"
    } == {"IC2574", "DDO154", "NGC6946"}
    assert all(first["benchmarks"]["passed"].values())
    assert first["response_boundary"]["velocity_values_opened"] == 0
    assert first["claim_boundary"]["source_maps_built"] is True
    assert first["claim_boundary"]["general_3d_gravity_validated"] is False
    assert first["content_sha256"] == package.content_sha256(package._without_hash(first))


def test_output_paths_are_fixed() -> None:
    config = package.load_config()
    assert config["output_path"] == package.OUTPUT_PATH.as_posix()
    assert config["private_output_directory"].startswith("work/private/")


def test_package_seals_match_current_files() -> None:
    assert (
        package.file_sha256(package._repo_path(package.CONFIG_PATH)) == package._CONFIG_RAW_SHA256
    )
    assert package.content_sha256(package.load_config()) == package._CONFIG_CONTENT_SHA256
    assert (
        package.module_semantic_sha256(package._repo_path(package.MODULE_PATH))
        == package._MODULE_SEMANTIC_SHA256
    )
    assert package.file_sha256(package._repo_path(package.TEST_PATH)) == package._TEST_RAW_SHA256
