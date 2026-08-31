from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_3d_synthetic_recovery_falsification_v1 as recovery
from sigma_theory_compiler import open_gravity_common_3d_synthetic_universe_v1 as universe


@pytest.fixture(scope="module")
def packet() -> tuple[dict, dict]:
    config = recovery.load_config()
    return config, recovery.run_suite(config)


def test_config_and_exact_predecessor_chain(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    recovery.validate_config(config)
    assert [row["role"] for row in config["bindings"]] == [
        "COMMON_SYNTHETIC_UNIVERSE",
        "NEWTON_AQUAL_QUMOND",
        "MOG_REFRACTED",
        "GP01_FULL3D",
    ]
    assert all(len(row["commit"]) == 40 for row in config["bindings"])


def test_exact_six_mechanism_inventory(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert tuple(row["id"] for row in config["mechanisms"]) == recovery._MECHANISMS
    assert suite["mechanisms"] == 6


def test_all_eleven_target_free_gates_pass(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    assert list(suite["gates"]) == config["required_gates"]
    assert suite["passed"] == 11
    assert suite["failed"] == 0
    assert all(row["passed"] is True for row in suite["gates"].values())


def test_every_mechanism_returns_a_finite_converged_field(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["gates"]["ALL_EXECUTABLE_MECHANISMS_RETURN_FINITE_FIELDS"]["metrics"]
    assert set(metrics["converged"]) == set(recovery._MECHANISMS)
    assert all(metrics["converged"].values())
    assert metrics["residuals"]["AQUAL"] < 2.0e-7


def test_injected_signatures_recover_their_generating_mechanism(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    metrics = suite["gates"]["SELF_INJECTION_NEAREST_SIGNATURE_RECOVERY"]["metrics"]
    assert all(injected == selected for injected, selected in metrics["recovered"].items())
    assert metrics["minimum_nearest_margin"] > 0.0


def test_all_six_mechanisms_retain_the_zero_source_null(packet: tuple[dict, dict]) -> None:
    _config, suite = packet
    maxima = suite["gates"]["ZERO_SOURCE_NULL_RETAINED"]["metrics"]["maximum_by_mechanism"]
    assert set(maxima) == set(recovery._MECHANISMS)
    assert all(value == 0.0 for value in maxima.values())


def test_nonspherical_fixture_distinguishes_all_implemented_fields(
    packet: tuple[dict, dict],
) -> None:
    config, suite = packet
    metrics = suite["gates"]["NONSYMMETRIC_CLOSEST_COMPARATORS_DISTINGUISHED"]["metrics"]
    assert (
        metrics["minimum_pairwise_distance"]
        > config["test_contract"]["minimum_distinct_signature_distance"]
    )
    assert len(metrics["pairwise_distances"]) == 15


def test_spherical_degeneracy_is_reported_without_extra_discovery_chance(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    metrics = suite["gates"]["SPHERICAL_DEGENERACY_REPORTED_NOT_PROMOTED"]["metrics"]
    assert metrics["spherical_aqual_qumond_difference"] < metrics["disk_aqual_qumond_difference"]
    assert metrics["independent_discovery_chance"] is False


def test_source_shuffle_rotation_high_field_and_environment_replays(
    packet: tuple[dict, dict],
) -> None:
    _config, suite = packet
    assert (
        suite["gates"]["SOURCE_SHUFFLE_BREAKS_SIGNATURE"]["metrics"][
            "newton_signature_relative_change"
        ]
        > 0.01
    )
    assert max(suite["gates"]["ROTATION_COVARIANCE_REPLAY"]["metrics"].values()) < 1.0e-9
    assert max(suite["gates"]["HIGH_ACCELERATION_NEWTON_LIMIT_REPLAY"]["metrics"].values()) < 1.0e-3
    assert (
        suite["gates"]["EXTERNAL_FIELD_AND_SADDLE_REPLAY"]["metrics"]["aqual_external_field_change"]
        > 0.1
    )


def test_parent_failures_are_not_erased(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    metrics = suite["gates"]["PARENT_COUNTEREXAMPLES_AND_BLOCKS_RETAINED"]["metrics"]
    assert metrics["gp01_local_generic_3d_control_only"] is True
    assert metrics["transport_branches_blocked"] == 2
    assert metrics["action_quarantined"] is True
    assert len(config["blocked_or_nonindependent"]) == 5


def test_unknown_mechanism_fails_closed(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    fixtures = universe.build_fixtures(universe.load_config())
    density = fixtures.sources[config["test_contract"]["primary_fixture"]]
    with pytest.raises(recovery.RecoveryError, match="unknown mechanism"):
        recovery.solve_mechanism("INVENTED", density, fixtures, {})


def test_each_public_solver_branch_has_deterministic_shape(packet: tuple[dict, dict]) -> None:
    config, _suite = packet
    fixtures = universe.build_fixtures(universe.load_config())
    density = fixtures.sources[config["test_contract"]["primary_fixture"]]
    parameters = {row["id"]: row["parameters"] for row in config["mechanisms"]}
    for mechanism in recovery._MECHANISMS:
        first, _residual, converged = recovery.solve_mechanism(
            mechanism, density, fixtures, parameters[mechanism]
        )
        second, _residual2, converged2 = recovery.solve_mechanism(
            mechanism, density, fixtures, parameters[mechanism]
        )
        assert first.shape == density.shape
        assert np.array_equal(first, second)
        assert converged and converged2


@pytest.mark.parametrize(
    "section",
    (
        "purpose",
        "bindings",
        "mechanisms",
        "test_contract",
        "required_gates",
        "blocked_or_nonindependent",
        "access_contract",
        "claim_boundary",
    ),
)
def test_every_semantic_section_is_hard_pinned(packet: tuple[dict, dict], section: str) -> None:
    config, _suite = packet
    changed = copy.deepcopy(config)
    changed[section] = None
    with pytest.raises(recovery.RecoveryError, match="config semantics changed"):
        recovery.validate_config(changed)


def test_noncanonical_receipt_path_rejected_before_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    reads = 0

    def forbidden(*_args: object, **_kwargs: object) -> dict:
        nonlocal reads
        reads += 1
        return {}

    monkeypatch.setattr(recovery, "OUTPUT_PATH", tmp_path / "private-response.json")
    monkeypatch.setattr(recovery, "_read_json", forbidden)
    with pytest.raises(recovery.RecoveryError, match="output path changed"):
        recovery.validate_receipt()
    assert reads == 0


def test_receipt_rebuild_and_coherent_forgery_rejection(packet: tuple[dict, dict]) -> None:
    _config, _suite = packet
    receipt = recovery.build_receipt()
    recovery.validate_receipt_payload(receipt)
    forged = copy.deepcopy(receipt)
    forged["suite"]["real_response_scoring_eligible"] = True
    forged["content_sha256"] = recovery.content_sha256(
        {key: value for key, value in forged.items() if key != "content_sha256"}
    )
    with pytest.raises(recovery.RecoveryError, match="not reproducible"):
        recovery.validate_receipt_payload(forged)


def test_zero_access_and_narrow_claim_boundary(packet: tuple[dict, dict]) -> None:
    config, suite = packet
    receipt = recovery.build_receipt()
    assert all(value == 0 for value in receipt["access_accounting"].values())
    assert suite["real_response_scoring_eligible"] is False
    assert "observational preference" in config["claim_boundary"]["does_not_establish"]
    assert "novelty" in config["claim_boundary"]["does_not_establish"]
