from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_common_3d_synthetic_universe_v1 as universe


@pytest.fixture(scope="module")
def packet() -> tuple[dict, universe.FixtureSet, dict]:
    config = universe.load_config()
    return config, universe.build_fixtures(config), universe.run_suite(config)


def test_config_and_exact_predecessor_chain(packet: tuple) -> None:
    config, _fixtures, _suite = packet
    universe.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "SOURCE_GEOMETRY_CONTRACT",
        "NEWTON_AQUAL_QUMOND_BASELINES",
        "GRAVITY_LIGHT_QUANTUM_CARDS",
        "HALO_AND_MODIFIED_GRAVITY_COMPARATORS",
        "GP01_FULL3D_DYNAMICS",
    ]
    assert all(len(row["commit"]) == 40 for row in config["bindings"])


def test_exact_fixture_inventory_and_oracle_coverage(packet: tuple) -> None:
    config, fixtures, suite = packet
    assert tuple(row["id"] for row in config["fixtures"]) == universe._FIXTURE_IDS
    assert len(config["fixtures"]) == 26
    assert len(fixtures.sources) == 17
    assert suite["gates"]["FIXTURE_INVENTORY_AND_ORACLES"]["metrics"] == {
        "fixtures": 26,
        "fixtures_with_oracles": 26,
    }


def test_all_fifteen_target_free_gates_pass(packet: tuple) -> None:
    config, _fixtures, suite = packet
    assert list(suite["gates"]) == config["required_gates"]
    assert suite["passed"] == 15
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["gates"].values())
    assert suite["real_response_scoring_eligible"] is False


def test_positive_source_arrays_have_exact_unit_mass(packet: tuple) -> None:
    _config, fixtures, _suite = packet
    for fixture_id, density in fixtures.sources.items():
        if fixture_id == "F17_VOID_CONTRAST":
            continue
        assert np.min(density) >= 0.0
        assert np.sum(density) * fixtures.grid.spacing**3 == pytest.approx(1.0, abs=3.0e-15)


def test_void_is_a_zero_mean_density_contrast(packet: tuple) -> None:
    _config, fixtures, suite = packet
    void = fixtures.sources["F17_VOID_CONTRAST"]
    centre = tuple(size // 2 for size in void.shape)
    assert abs(np.sum(void)) < 1.0e-12
    assert void[centre] < 0.0
    assert suite["gates"]["VOID_ZERO_MEAN_CONTRAST"]["passed"] is True


def test_disks_and_triaxial_fixture_are_geometrically_distinct(packet: tuple) -> None:
    _config, _fixtures, suite = packet
    metrics = suite["gates"]["DISK_THICKNESS_AND_TRIAXIAL_GEOMETRY"]["metrics"]
    assert metrics["thick_vertical_moment"] > metrics["thin_vertical_moment"]
    assert len(set(metrics["triaxial_principal_moments"])) == 3


def test_pair_binary_saddle_and_environment_contracts(packet: tuple) -> None:
    _config, fixtures, suite = packet
    symmetry = suite["gates"]["PAIR_BINARY_AND_SADDLE_SYMMETRIES"]["metrics"]
    assert abs(symmetry["binary_center_of_mass"]) < 1.0e-14
    assert symmetry["saddle_central_field"] < 1.0e-12
    assert fixtures.metadata["F10_SATELLITE_EXTERNAL_FIELD"]["external_acceleration"] == [
        0.25,
        0.0,
        0.0,
    ]


def test_filament_wall_sphere_and_merger_have_distinct_topologies(packet: tuple) -> None:
    _config, _fixtures, suite = packet
    metrics = suite["gates"]["FILAMENT_WALL_CLUSTER_TOPOLOGY"]["metrics"]
    assert metrics["filament_axis_ratio"] > 4.0
    assert metrics["wall_second_axis_ratio"] > 4.0
    assert metrics["cluster_moment_spread"] < 1.0e-12
    assert metrics["merger_axis_maxima"] == 2


def test_history_ray_clock_and_wave_oracles_are_independent(packet: tuple) -> None:
    _config, _fixtures, suite = packet
    assert suite["gates"]["SOURCE_HISTORY_CONSERVATION_AND_ORDER"]["passed"] is True
    assert suite["gates"]["LENSING_RAY_ORACLE"]["metrics"]["maximum_inverse_impact_error"] == 0.0
    assert suite["gates"]["CLOCK_ENDPOINT_NOT_PATH_RULE"]["metrics"]["path_spread"] == 0.0
    assert (
        suite["gates"]["GW_TRANSVERSE_TRACELESS_POLARIZATIONS"]["metrics"][
            "transverse_traceless_error"
        ]
        == 0.0
    )


def test_adversarial_controls_separate_invariance_from_corruption(packet: tuple) -> None:
    _config, _fixtures, suite = packet
    invariant = suite["gates"]["ADVERSARIAL_ROTATION_AND_SHUFFLE_INVARIANCE"]
    corrupt = suite["gates"]["ADVERSARIAL_SIGN_UNIT_TIME_CORRUPTION_DETECTED"]
    assert invariant["passed"] is True
    assert corrupt["metrics"] == {"sign": True, "unit": True, "chronology": True}


@pytest.mark.parametrize(
    "values, spacing",
    (
        (np.zeros((3, 3)), 1.0),
        (-np.ones((3, 3, 3)), 1.0),
        (np.full((3, 3, 3), np.nan), 1.0),
        (np.zeros((3, 3, 3)), 1.0),
    ),
)
def test_invalid_or_massless_positive_sources_fail_closed(
    values: np.ndarray, spacing: float
) -> None:
    with pytest.raises(universe.SyntheticUniverseError):
        universe._normalize(values, spacing)


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "bindings",
        "numerical_contract",
        "fixtures",
        "required_gates",
        "anti_leakage",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple, section: str) -> None:
    config, _fixtures, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(universe.SyntheticUniverseError, match="config semantics changed"):
        universe.validate_config(changed)


def test_noncanonical_receipt_path_is_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(universe, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(universe, "_read_json", forbidden)
    with pytest.raises(universe.SyntheticUniverseError, match="output path changed"):
        universe.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple) -> None:
    _config, _fixtures, _suite = packet
    receipt = universe.build_receipt()
    universe.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["does_not_establish"] = []
    forged["content_sha256"] = universe.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(universe.SyntheticUniverseError, match="not reproducible"):
        universe.validate_receipt_payload(forged)


def test_no_response_leakage_or_observational_authority(packet: tuple) -> None:
    config, _fixtures, suite = packet
    receipt = universe.build_receipt()
    assert config["anti_leakage"]["scientific_response_inputs"] == []
    assert config["anti_leakage"]["fixture_parameters_chosen_from_response"] is False
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert suite["real_response_scoring_eligible"] is False
    assert "real-data preference" in config["claim_boundary"]["does_not_establish"]
