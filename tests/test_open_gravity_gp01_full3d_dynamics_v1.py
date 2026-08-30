from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import open_gravity_gp01_full3d_dynamics_v1 as gp3d


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = gp3d.load_config()
    return config, gp3d.run_suite(config)


def test_config_and_exact_committed_predecessors(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    gp3d.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "GP01_FOUNDATION",
        "FULL3D_BASELINES",
        "COMPARATOR_MECHANICS",
    ]
    assert all(len(row["commit"]) == 40 for row in config["bindings"])


def test_all_fifteen_target_free_gates_pass(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert list(suite["gates"]) == config["required_gates"]
    assert suite["passed"] == 15
    assert suite["failed"] == 0
    assert all(bool(row["passed"]) for row in suite["gates"].values())


def test_full3d_static_branch_and_partial_dynamic_branch_are_distinct(
    packet: tuple[dict, dict],
) -> None:
    config, _suite = packet
    dispositions = {row["id"]: row["status"] for row in config["branch_dispositions"]}
    assert dispositions["GP01-ELLIPTIC"] == "PASS_TARGET_FREE_FULL3D_STATIC_MECHANICS"
    assert dispositions["GP01-TELEGRAPH"].startswith("PARTIAL_")
    assert dispositions["GP01-ACTION-PLACEHOLDER"] == "INCOMPLETE_QUARANTINE"


def test_manufactured_gain_solution_with_nonzero_boundary() -> None:
    grid = base.make_grid(9)
    exact = 0.2 + 0.1 * (1.0 - grid.x**2) * (1.0 - grid.y**2) * (1.0 - grid.z**2)
    boundary = exact.copy()
    length = 0.17
    target = exact - length**2 * base._constant_laplacian(exact, grid.spacing)
    result = gp3d.solve_quasi_static_gain(target, boundary, grid.spacing, length=length)
    assert result.relative_residual < 1.0e-12
    assert np.max(np.abs(result.gamma - exact)) < 1.0e-12


def test_local_zero_length_limit_is_exact() -> None:
    target = np.full((5, 5, 5), 0.4)
    boundary = np.zeros_like(target)
    result = gp3d.solve_quasi_static_gain(target, boundary, 0.5, length=0.0)
    assert np.all(result.gamma[1:-1, 1:-1, 1:-1] == 0.4)
    assert np.all(result.gamma[[0, -1], :, :] == 0.0)
    assert result.relative_residual == 0.0


def test_elliptic_solution_is_bounded_and_uniformly_elliptic(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    maximum = suite["gates"]["ELLIPTIC_MAXIMUM_PRINCIPLE"]["metrics"]["maximum"]
    coefficient = suite["gates"]["COEFFICIENT_ELLIPTICITY"]["metrics"]
    assert 0.0 <= maximum <= config["numerical_contract"]["Gamma_max"]
    assert coefficient["coefficient_minimum"] >= coefficient["declared_lower_bound"]


def test_rotation_and_conservative_curl_gates_are_numerically_resolved(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    rotation = suite["gates"]["ROTATION_COVARIANCE"]["metrics"]
    assert rotation["gamma_relative_error"] < 1.0e-9
    assert rotation["potential_relative_error"] < 1.0e-9
    assert (
        suite["gates"]["NONSYMMETRIC_CONSERVATIVE_CURL"]["metrics"]["maximum_interior_curl"]
        < 1.0e-11
    )


def test_saddle_and_external_field_are_retained(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    saddle = suite["gates"]["SADDLE_EXACT_NULL_TARGET"]["metrics"]
    assert saddle["central_field_magnitude"] < 1.0e-12
    assert saddle["central_target"] == pytest.approx(
        config["numerical_contract"]["Gamma_max"], abs=1.0e-8
    )
    assert (
        suite["gates"]["EXTERNAL_FIELD_SENSITIVITY"]["metrics"]["maximum_target_difference"]
        > 1.0e-3
    )


def test_spatial_kernel_identity_does_not_claim_infinite_space_convolution(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    metrics = suite["gates"]["SPATIAL_KERNEL_PDE_IDENTITY"]["metrics"]
    assert metrics["maximum_operator_residual"] < 1.0e-11
    assert metrics["finite_box_convolution_claimed"] is False


def test_telegraph_has_damping_persistence_and_only_a_necessary_speed_gate(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    decay = suite["gates"]["TELEGRAPH_ENERGY_DECAY"]["metrics"]
    persistence = suite["gates"]["TELEGRAPH_TEMPORAL_PERSISTENCE"]["metrics"]
    speed = suite["gates"]["TELEGRAPH_NECESSARY_SPEED"]["metrics"]
    assert decay["final_energy"] < decay["initial_energy"]
    assert persistence["post_source_state_maximum"] > 0.0
    assert speed["c_gamma_over_c"] <= 1.0
    assert speed["common_cone_proved"] is False


@pytest.mark.parametrize(
    "call",
    (
        lambda: gp3d.solve_quasi_static_gain(
            np.ones((5, 5, 5)), np.zeros((5, 5, 4)), 1.0, length=0.1
        ),
        lambda: gp3d.solve_quasi_static_gain(
            -np.ones((5, 5, 5)), np.zeros((5, 5, 5)), 1.0, length=0.1
        ),
        lambda: gp3d.solve_quasi_static_gain(
            np.ones((5, 5, 5)), np.zeros((5, 5, 5)), 1.0, length=-0.1
        ),
        lambda: gp3d.evolve_telegraph(
            np.zeros((5, 5, 5)),
            np.zeros((5, 5, 5)),
            np.zeros((5, 5, 5)),
            1.0,
            length=0.0,
            tau=1.0,
            dt=0.1,
            steps=1,
        ),
        lambda: gp3d.evolve_telegraph(
            np.zeros((5, 5, 5)),
            np.zeros((5, 5, 5)),
            np.zeros((5, 5, 5)),
            1.0,
            length=1.0,
            tau=1.0,
            dt=0.1,
            steps=0,
        ),
    ),
)
def test_invalid_solver_inputs_fail_closed(call) -> None:
    with pytest.raises(gp3d.GP01Full3DError):
        call()


def test_action_singularity_classes_are_retained(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    assert suite["gates"]["ACTION_SINGULARITIES_RETAINED"]["metrics"] == {
        "1": "V_POWER_DIVERGENCE_AND_VPRIME_DIVERGENCE",
        "2": "V_LOG_DIVERGENCE_AND_VPRIME_DIVERGENCE",
        "4": "V_FINITE_BUT_VPRIME_DIVERGENT_NONANALYTIC",
    }


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "bindings",
        "equations",
        "numerical_contract",
        "branch_dispositions",
        "required_gates",
        "remaining_blockers",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple[dict, dict], section: str) -> None:
    config, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(gp3d.GP01Full3DError, match="config semantics changed"):
        gp3d.validate_config(changed)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(gp3d, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(gp3d, "_read_json", forbidden)
    with pytest.raises(gp3d.GP01Full3DError, match="output path changed"):
        gp3d.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple[dict, dict]) -> None:
    _config, _suite = packet
    receipt = gp3d.build_receipt()
    gp3d.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["claim_boundary"]["does_not_establish"] = []
    forged["content_sha256"] = gp3d.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(gp3d.GP01Full3DError, match="not reproducible"):
        gp3d.validate_receipt_payload(forged)


def test_zero_access_and_narrow_claim_boundary(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    receipt = gp3d.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert suite["instantaneous_baryonic_source_remains"] is True
    assert "causal source completion" in config["claim_boundary"]["does_not_establish"]
    assert "observational preference" in config["claim_boundary"]["does_not_establish"]
    assert math.isfinite(suite["gates"]["TELEGRAPH_ENERGY_DECAY"]["metrics"]["final_energy"])
