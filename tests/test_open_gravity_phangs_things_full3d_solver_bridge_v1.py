from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_phangs_things_full3d_solver_bridge_v1 as bridge


@pytest.fixture(scope="module")
def config() -> dict:
    value = json.loads(Path(bridge.CONFIG_PATH).read_text(encoding="utf-8"))
    bridge.validate_config(value)
    return value


@pytest.fixture(scope="module")
def packet(config: dict) -> tuple[dict, dict]:
    return bridge.build_packet(config)


def test_admission_rule_requires_real_sources_and_published_benchmarks(config: dict) -> None:
    assert "public source dataset" in config["admission_rule"]
    assert "primary-paper or analytic benchmark" in config["admission_rule"]
    assert [row["id"] for row in config["real_source_anchors"]] == [
        "S4G_P5_STELLAR",
        "S4G_MASS_TO_LIGHT",
        "THINGS_HI",
        "PHANGS_ALMA_CO21",
    ]
    assert [row["id"] for row in config["published_solver_anchors"]] == [
        "FREEMAN_DISK",
        "CASERTANO_THICK_DISK",
        "AQUAL",
        "QUMOND",
        "AQUAL_QUMOND_EFE",
    ]


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "PUBLICATION_READY"),
        (("objects",), ["NGC2903"]),
        (("volume_contract", "primary_nodes_per_axis"), 13),
        (("volume_contract", "direct_volumetric_observation_claim"), True),
        (("solver_contract", "response_tuning_forbidden"), False),
        (("gate_contract", "source_builder_benchmarks_must_all_pass"), False),
        (("gate_contract", "primary_vs_convergence_radial_relative_difference_max"), 1.0),
        (("scientific_boundary", "response_rows_opened"), 1),
        (("scientific_boundary", "scores_computed"), 1),
        (("claim_boundary", "observational_preference_established"), True),
    ],
)
def test_material_config_mutations_fail_closed(
    config: dict, path: tuple[str, ...], replacement: object
) -> None:
    mutated = copy.deepcopy(config)
    cursor = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement
    with pytest.raises(bridge.Full3DBridgeError):
        bridge.validate_config(mutated)


def test_vertical_deposition_is_symmetric_normalized_and_resolves_thickness() -> None:
    coordinates = np.linspace(-30000.0, 30000.0, 25)
    thin = bridge.vertical_slab_fractions(coordinates, 2500.0, 200.0)
    thick = bridge.vertical_slab_fractions(coordinates, 2500.0, 1000.0)
    assert thin.sum() == pytest.approx(1.0, abs=1.0e-15)
    assert thick.sum() == pytest.approx(1.0, abs=1.0e-15)
    assert np.array_equal(thin, thin[::-1])
    assert np.array_equal(thick, thick[::-1])
    assert thin[12] > thick[12]
    assert thin[[0, -1]].tolist() == [0.0, 0.0]


def test_surface_deposition_conserves_a_synthetic_public_source() -> None:
    source_coordinates = np.linspace(-1000.0, 1000.0, 9)
    x_pc, y_pc = np.meshgrid(source_coordinates, source_coordinates, indexing="ij")
    sigma = np.full((9, 9), 12.5)
    solver_coordinates = np.linspace(-3000.0, 3000.0, 9)
    rho, mass = bridge.deposit_surface_component(
        sigma,
        x_pc,
        y_pc,
        250.0,
        solver_coordinates,
        750.0,
        300.0,
    )
    assert mass == pytest.approx(float(sigma.sum() * 250.0**2), rel=0.0, abs=1.0e-8)
    assert float(rho.sum() * 750.0**3) == pytest.approx(mass, rel=5.0e-16)
    assert np.count_nonzero(rho[[0, -1], :, :]) == 0
    assert np.count_nonzero(rho[:, [0, -1], :]) == 0
    assert np.count_nonzero(rho[:, :, [0, -1]]) == 0


def test_newton_spherical_boundary_reproduces_the_analytic_monopole_difference() -> None:
    grid = bridge.baseline.make_grid(9)
    mass = 0.03
    boundary = bridge.spherical_boundary(grid, mass, mond=False, integration_samples=100000)
    r_corner = math.sqrt(3.0)
    expected_face = -mass * (1.0 - 1.0 / r_corner)
    assert boundary[0, 4, 4] == pytest.approx(expected_face, rel=2.0e-8)
    assert boundary[0, 0, 0] == pytest.approx(0.0, abs=1.0e-14)
    assert np.count_nonzero(boundary[1:-1, 1:-1, 1:-1]) == 0


def test_packet_passes_source_paper_mass_residual_and_resolution_gates(
    packet: tuple[dict, dict],
) -> None:
    _private, receipt = packet
    assert receipt["inherited_benchmarks"] == {
        "source_builder_all_pass": True,
        "solver_baseline_all_pass": True,
    }
    assert receipt["object_count"] == 3
    assert receipt["field_solution_count"] == 18
    assert receipt["all_object_gates_pass"] is True
    for row in receipt["objects"]:
        assert row["all_gates_pass"] is True
        assert all(row["gates"].values())
        assert row["source_mass_relative_error"] < 2.0e-9
        assert row["primary"]["dimensionless_mass_relative_error"] < 1.0e-12
        assert row["primary"]["solver_metrics"]["newton_relative_residual"] < 1.0e-12
        assert row["primary"]["solver_metrics"]["qumond_relative_residual"] < 1.0e-12
        assert row["primary"]["solver_metrics"]["aqual_relative_residual"] < 2.0e-7
        assert row["primary"]["solver_metrics"]["aqual_converged"] is True
        assert max(row["profile_convergence_relative"].values()) < 0.08


def test_real_source_fields_are_distinct_without_using_a_response(
    packet: tuple[dict, dict],
) -> None:
    _private, receipt = packet
    for row in receipt["objects"]:
        fields = row["primary"]["field_hashes"]
        assert len(set(fields.values())) == len(fields)
        profiles = row["primary"]["profiles"]
        newton = profiles["NEWTON"]
        aqual = profiles["AQUAL_SIMPLE_MU"]
        qumond = profiles["QUMOND_SIMPLE_NU"]
        assert all(point["radial_acceleration_over_a0"] > 0.0 for point in newton)
        assert all(point["radial_acceleration_over_a0"] > 0.0 for point in aqual)
        assert all(point["radial_acceleration_over_a0"] > 0.0 for point in qumond)
        assert aqual[0]["radial_acceleration_over_a0"] > qumond[0]["radial_acceleration_over_a0"]
        assert qumond[0]["radial_acceleration_over_a0"] > newton[0]["radial_acceleration_over_a0"]


def test_private_packet_retains_reusable_cubes_but_no_response(
    packet: tuple[dict, dict],
) -> None:
    private, receipt = packet
    assert len(private["objects"]) == 3
    for row in private["objects"]:
        assert np.asarray(row["density_dimensionless"]).shape == (25, 25, 25)
        assert np.asarray(row["newton_potential_dimensionless"]).shape == (25, 25, 25)
        assert np.asarray(row["aqual_potential_dimensionless"]).shape == (25, 25, 25)
        assert np.asarray(row["qumond_potential_dimensionless"]).shape == (25, 25, 25)
    assert receipt["private_field_raw_sha256"] == bridge.content_sha256(private)
    assert receipt["scientific_boundary"]["response_values_opened"] == 0
    assert receipt["scientific_boundary"]["scores_computed"] == 0
    assert receipt["claim_boundary"]["observational_preference_established"] is False


def test_atomic_no_clobber_preserves_identical_and_rejects_different(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert bridge._atomic_no_clobber(path, b"one") == "CREATED"
    assert bridge._atomic_no_clobber(path, b"one") == "EXISTING_IDENTICAL"
    with pytest.raises(bridge.Full3DBridgeError):
        bridge._atomic_no_clobber(path, b"two")
    assert path.read_bytes() == b"one"


def test_package_hash_pins_match_after_seal() -> None:
    if bridge._MODULE_SEMANTIC_SHA256 == "0" * 64 or bridge._TEST_RAW_SHA256 == "0" * 64:
        pytest.skip("self pins are installed at the final mutation seal")
    assert (
        bridge.module_semantic_sha256(bridge._repo_path(bridge.MODULE_PATH))
        == bridge._MODULE_SEMANTIC_SHA256
    )
    assert bridge.file_sha256(bridge._repo_path(bridge.TEST_PATH)) == bridge._TEST_RAW_SHA256
