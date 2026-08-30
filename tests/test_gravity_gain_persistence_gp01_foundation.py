"""Tests for the append-only, no-response GP01 foundation."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_gain_persistence_gp01_foundation as gp01

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def config() -> dict[str, object]:
    return gp01.load_config(ROOT)


@pytest.fixture(scope="module")
def report(config: dict[str, object]) -> dict[str, object]:
    return gp01.build_synthetic_report(config)


def _set_path(value: dict[str, object], path: tuple[str | int, ...], replacement: object) -> None:
    cursor: object = value
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = replacement  # type: ignore[index]


def _coherently_rehash(receipt: dict[str, object]) -> None:
    receipt.pop("content_sha256", None)
    receipt["content_sha256"] = gp01._content_sha256(receipt)


def test_config_is_exactly_frozen_synthetic_only_and_never_response_scoreable(
    config: dict[str, object],
) -> None:
    assert gp01._content_sha256(config) == gp01.EXPECTED_CONFIG_CONTENT_SHA256
    assert {
        section: gp01._content_sha256(config[section])
        for section in gp01.EXPECTED_CONFIG_SECTION_SHA256
    } == gp01.EXPECTED_CONFIG_SECTION_SHA256
    assert config["status"] == "FROZEN_NO_RESPONSE_SYNTHETIC_PREFLIGHT"
    assert config["source_boundary"]["real_data_paths"] == []
    assert config["scope"]["observational_execution"] is False
    assert config["scope"]["campaign_scoring"] is False
    assert all(item["response_scoring_eligible"] is False for item in config["variants"])
    assert all(value == 0 for value in config["zero_access"].values())
    assert config["claim_boundary"]["response_scoring_unlocked"] is False
    assert config["claim_boundary"]["confirmation_opened"] is False


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "CONFIRMED"),
        (("scope", "physical_theory_claim"), True),
        (("equations", "local_nu"), "nu=42"),
        (("dimensions", "Phi_and_Phi_b"), "bananas"),
        (("parameters", "rho_star_over_source_reference_grid"), [-1.0]),
        (("variants", 4, "general_three_dimensional_theory"), True),
        (("claim_boundary", "transport_integrators_are_general_theory"), True),
        (("closures", "lensing"), "DERIVED"),
        (("action_placeholder", "promotion_status"), "CONFIRMED"),
        (("action_placeholder", "expression"), "S=0"),
        (("action_placeholder", "target_potential_derivative"), "V'=0"),
    ],
)
def test_every_semantic_config_mutation_fails_closed(
    config: dict[str, object], path: tuple[str | int, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(config)
    _set_path(tampered, path, replacement)
    with pytest.raises(gp01.GravityGainPersistenceFoundationError, match="content hash"):
        gp01.validate_config(tampered)


def test_gp01_local_control_has_exact_limits_and_spherical_scaling() -> None:
    for n in (1, 2, 4):
        assert gp01.nu_n(1e12, n) == pytest.approx(1.0, abs=1e-12)
        deep_dimensionless_g = gp01.nu_n(1e-12, n) * 1e-12
        assert deep_dimensionless_g == pytest.approx(math.sqrt(1e-12), rel=1e-6)

    mass, a_star, radius = 3.0, 0.02, 1e5
    g_b = mass / radius**2
    radial = np.asarray([[g_b, 0.0]])
    g = np.linalg.norm(gp01.local_algebraic_field(radial, a_star=a_star, n=2)[0])
    assert (radius * g) ** 2 == pytest.approx(mass * a_star, rel=1e-8)


def test_aqual_is_a_known_family_comparator_with_only_symmetry_limited_equivalence(
    config: dict[str, object],
) -> None:
    y = np.geomspace(1e-12, 1e12, 501)
    for n in (1, 2, 4):
        x, mu = gp01.aqual_parametric_mapping(y, n)
        np.testing.assert_allclose(mu * x, y, rtol=3e-16, atol=0.0)
        assert np.all(mu > 0.0)
        assert np.all(np.diff(x) > 0.0)
        assert mu[0] / x[0] == pytest.approx(1.0, rel=1e-6)
        assert mu[-1] == pytest.approx(1.0, abs=1e-12)
    aqual = next(item for item in config["variants"] if item["variant_id"] == "GP01-AQUAL")
    assert aqual["identity"] == "KNOWN_FAMILY_COMPARATOR"
    assert aqual["general_3d_equivalence"] is False
    assert aqual["equivalent_to_GP01_L_only_in"] == "spherical_or_valid_1D_curl_free_symmetry"


def test_t1_and_t2_exact_segments_and_all_line_validity_quarantines() -> None:
    s = np.linspace(0.0, 4.0, 41)
    force = np.exp(-s)
    t1 = gp01.transport_t1(
        s,
        force,
        np.ones_like(s),
        beta=0.5,
        l_reset=1.0,
        gamma_anchor=0.0,
    )
    np.testing.assert_allclose(t1, 0.5 * s, atol=1e-14)
    t2 = gp01.transport_t2(
        s,
        np.ones_like(s),
        force,
        l_g=1.0,
        gamma_anchor=0.0,
    )
    np.testing.assert_allclose(t2, 1.0 - np.exp(-s), atol=1e-14)

    cases = [
        ({"anchored": False}, "anchor"),
        ({"anchor_count": 2}, "anchors"),
        ({"crosses_separatrix": True}, "separatrix"),
        ({"closed_field_line": True}, "closed field"),
        ({"domain_exit": "GRID_EDGE"}, "domain exit"),
    ]
    for options, match in cases:
        with pytest.raises(gp01.GravityGainPersistenceFoundationError, match=match):
            gp01.transport_t2(
                s,
                np.ones_like(s),
                force,
                l_g=1.0,
                gamma_anchor=0.0,
                **options,
            )
    force_with_null = force.copy()
    force_with_null[-1] = 0.0
    with pytest.raises(gp01.GravityGainPersistenceFoundationError, match="field null"):
        gp01.transport_t2(
            s,
            np.ones_like(s),
            force_with_null,
            l_g=1.0,
            gamma_anchor=0.0,
        )


@pytest.mark.parametrize(
    "call",
    [
        lambda: gp01.nu_n(float("nan"), 2),
        lambda: gp01.bounded_gamma_target(1.0, w=float("nan"), a_star=1.0, n=2, gamma_max=1.0),
        lambda: gp01.environment_gate(1.0, 1.0, rho_star=float("inf"), tidal_star=1.0, q=1, r=1),
        lambda: gp01.transport_t1(
            [0.0, 1.0], [1.0, 0.5], [1.0, 1.0], beta=float("nan"), l_reset=1.0, gamma_anchor=0.0
        ),
        lambda: gp01.transport_t1(
            [0.0, 1.0], [1.0, 0.5], [1.0, 1.0], beta=0.5, l_reset=float("inf"), gamma_anchor=0.0
        ),
        lambda: gp01.transport_t2(
            [0.0, 1.0], [1.0, 1.0], [1.0, 0.5], l_g=float("nan"), gamma_anchor=0.0
        ),
        lambda: gp01.transport_t2(
            [0.0, 1.0], [1.0, 1.0], [1.0, float("nan")], l_g=1.0, gamma_anchor=0.0
        ),
        lambda: gp01.telegraph_characteristic_speed(0.0, 1.0),
    ],
)
def test_nonfinite_or_singular_scalar_inputs_fail_closed(call: Callable[[], object]) -> None:
    with pytest.raises(gp01.GravityGainPersistenceFoundationError):
        call()


def test_path_gate_uses_a_source_derived_multisource_loop(
    report: dict[str, object],
) -> None:
    local_log_f = np.asarray([0.0, -1.0, -2.0, -1.0, 0.0])
    assert gp01.closed_path_integral(local_log_f, np.ones(5)) == pytest.approx(0.0, abs=1e-15)
    multisource = next(
        item for item in report["fixture_results"] if item["fixture_id"] == "SYN-GP01-MULTISOURCE"
    )
    loop = multisource["source_derived_loop"]
    assert loop["loop_points_including_closure"] == 513
    assert loop["force_closure_error"] == 0.0
    assert abs(loop["closed_integral_W_dlnf"]) > 1e-8
    assert loop["path_independence_passed"] is False


def test_bounded_target_and_complete_pde_parameter_grid_coverage(
    report: dict[str, object],
) -> None:
    gamma_max = math.log(8.0)
    target = gp01.bounded_gamma_target(
        np.asarray([0.0, 1e-300, 0.01, 1e100]),
        w=0.7,
        a_star=0.01,
        n=2,
        gamma_max=gamma_max,
    )
    assert target[0] == pytest.approx(0.7 * gamma_max)
    assert np.all(np.isfinite(target))
    assert np.all((target >= 0.0) & (target <= 0.7 * gamma_max))

    coverage = report["parameter_grid_coverage"]
    assert coverage["declared_cell_counts"] == {
        "n": 3,
        "A_max": 3,
        "L_reset": 3,
        "T2_transport_L_g": 3,
        "environment": 36,
        "quasi_static_A_max_by_L_g_including_zero": 12,
        "telegraph_L_g_by_speed": 6,
    }
    assert coverage["exercised_cell_counts"] == coverage["declared_cell_counts"]
    assert coverage["exact_declared_values_exercised"] is True
    assert coverage["all_declared_cells_exercised"] is True
    assert all(cell["bounded"] for cell in coverage["environment_cells"])
    assert all(
        cell["bounded_under_declared_M_matrix_conditions"]
        for cell in coverage["quasi_static_cells"]
    )
    assert all(
        abs(cell["left_Dirichlet_Gamma"]) <= 1e-14 and abs(cell["right_Dirichlet_Gamma"]) <= 1e-14
        for cell in coverage["quasi_static_cells"]
    )
    zero_cells = [cell for cell in coverage["quasi_static_cells"] if cell["L_g_over_R_b"] == 0.0]
    assert len(zero_cells) == 3
    assert all(cell["zero_length_recovers_target"] is True for cell in zero_cells)
    assert all(cell["finite_positive_scales"] for cell in coverage["causal_cells"])
    assert all(
        cell["computed_c_Gamma_over_c"] == pytest.approx(cell["c_Gamma_over_c"])
        for cell in coverage["causal_cells"]
    )


def test_all_fixture_required_checks_are_computed_and_bound(
    config: dict[str, object], report: dict[str, object]
) -> None:
    declarations = {
        fixture["fixture_id"]: fixture["required_checks"]
        for fixture in config["synthetic_contract"]["fixtures"]
    }
    fixtures = {item["fixture_id"]: item for item in report["fixture_results"]}
    assert tuple(fixtures) == gp01.FIXTURE_IDS
    for fixture_id, result in fixtures.items():
        assert result["required_checks"] == declarations[fixture_id]
        assert [check["check_id"] for check in result["checks"]] == declarations[fixture_id]
        assert all(check["passed"] is True for check in result["checks"])
        assert result["all_required_checks_passed"] is True
        assert result["real_rows"] == 0
    assert fixtures["SYN-GP01-SPHERE"]["status"] == "PASS_COMPUTED_LOCAL_CONTROL"
    assert fixtures["SYN-GP01-DISK"]["induced_curl_analytic_rms"] > 1e-6
    assert fixtures["SYN-GP01-MULTISOURCE"]["induced_curl_analytic_rms"] > 1e-6
    assert fixtures["SYN-GP01-SADDLE"]["transport_refusal"]["refusal_observed"] is True
    assert fixtures["SYN-GP01-VOID"]["transport_refusal"]["refusal_observed"] is True


def test_action_target_and_singularity_audit_remain_quarantined(
    config: dict[str, object], report: dict[str, object]
) -> None:
    y = np.geomspace(1e-8, 1e8, 100)
    for n in (1, 2, 4):
        gamma, v_prime = gp01.action_target(y, n)
        np.testing.assert_allclose(gamma, np.log(gp01.nu_n(y, n)), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(v_prime, y * y * gp01.nu_n(y, n), rtol=0.0, atol=0.0)
    assert gp01.action_regularity_class(1).startswith("V_POWER_DIVERGENCE")
    assert gp01.action_regularity_class(2).startswith("V_LOG_DIVERGENCE")
    assert gp01.action_regularity_class(4).startswith("V_FINITE_BUT_VPRIME_DIVERGENT")
    assert config["action_placeholder"]["executable"] is False
    assert report["action_audit"]["healthy_action_completed"] is False
    assert report["action_audit"]["damping_derived_from_action"] is False


def test_filter_matrix_is_complete_conditional_and_fail_closed(
    report: dict[str, object],
) -> None:
    filters = {item["filter_id"]: item for item in report["filter_results"]}
    assert tuple(filters) == gp01.FILTER_IDS
    assert all(item["observed_as_designed"] for item in filters.values())
    assert filters["F04_EXACT_AQUAL_MAPPING"]["status"].startswith("PASS_KNOWN_FAMILY_")
    assert filters["F05_GENERAL_3D_CURL"]["status"].startswith("DESIGNED_FAIL_")
    assert filters["F07_ENVIRONMENTAL_CLOSED_PATH"]["status"].startswith("DESIGNED_FAIL_")
    assert filters["F08_FIELD_NULL_AND_SEPARATRIX"]["status"].startswith("QUARANTINED_")
    assert filters["F09_CONDITIONAL_BOUNDED_ELLIPTICITY"]["status"].startswith("CONDITIONAL_")
    assert filters["F11_CAUSAL_SOURCE_COMPLETION"]["status"].startswith("BLOCKED_")
    assert filters["F12_ENERGY_AND_ACTION"]["status"].startswith("ACTION_PLACEHOLDER_")
    pde = report["pde_contracts"]
    assert pde["telegraph_overshoot_is_bounded"] is False
    assert pde["telegraph_speed_is_sufficient_for_causality"] is False
    assert pde["instantaneous_baryonic_poisson_source_remains"] is True
    assert pde["quasi_static_equation_is_temporal_memory"] is False


def test_atomic_no_replace_preserves_a_nonidentical_existing_artifact(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_bytes(b"existing")
    with pytest.raises(gp01.GravityGainPersistenceFoundationError, match="refusing"):
        gp01._atomic_no_replace(target, b"replacement")
    assert target.read_bytes() == b"existing"
    assert gp01._atomic_no_replace(target, b"existing") == "EXISTING_IDENTICAL"


def test_built_receipt_is_fully_bound_to_files_and_exact_recomputation() -> None:
    receipt = gp01.build_receipt(ROOT)
    assert receipt["decision"] == gp01.DECISION
    assert receipt["config_binding"]["content_sha256"] == gp01.EXPECTED_CONFIG_CONTENT_SHA256
    assert receipt["config_binding"]["section_sha256"] == gp01.EXPECTED_CONFIG_SECTION_SHA256
    assert receipt["implementation_binding"]["source_sha256"] == gp01._file_sha256(
        ROOT / gp01.SOURCE_PATH
    )
    assert receipt["implementation_binding"]["test_sha256"] == gp01._file_sha256(
        ROOT / gp01.TEST_PATH
    )
    assert receipt["synthetic_report"] == gp01.build_synthetic_report(gp01.load_config(ROOT))
    assert receipt["limitations"] == list(gp01.LIMITATIONS)


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("status",), "CONFIRMED"),
        (("implementation_binding", "source_sha256"), "0" * 64),
        (("closures", "lensing"), "DERIVED"),
        (
            (
                "synthetic_report",
                "parameter_grid_coverage",
                "quasi_static_cells",
                0,
                "solution_minimum",
            ),
            -1.0,
        ),
        (("action_quarantine", "promotion_status"), "CONFIRMED"),
        (("counts", "real_rows"), 1),
        (("limitations",), []),
    ],
)
def test_coherently_rehashed_receipt_semantic_forgery_fails_closed(
    config: dict[str, object], path: tuple[str | int, ...], replacement: object
) -> None:
    tampered = copy.deepcopy(gp01.build_receipt(ROOT))
    _set_path(tampered, path, replacement)
    _coherently_rehash(tampered)
    with pytest.raises(gp01.GravityGainPersistenceFoundationError):
        gp01.validate_receipt(tampered, config, root=ROOT)


def test_stored_receipt_is_an_exact_deterministic_rebuild() -> None:
    receipt = gp01.check_receipt(ROOT)
    assert receipt == gp01.build_receipt(ROOT)
    assert receipt["counts"] == {
        "variants": 7,
        "synthetic_fixtures": 5,
        "theory_filters": 13,
        "designed_failures_and_quarantines": 6,
        "real_rows": 0,
        "response_scores": 0,
    }
    assert receipt["zero_access"]["response_rows_opened"] == 0
    assert receipt["claim_boundary"]["response_scoring_unlocked"] is False


def test_unhashed_receipt_tampering_is_rejected_before_semantic_checks() -> None:
    config = gp01.load_config(ROOT)
    tampered = json.loads(json.dumps(gp01.build_receipt(ROOT)))
    tampered["zero_access"]["response_rows_opened"] = 1
    with pytest.raises(gp01.GravityGainPersistenceFoundationError, match="content hash"):
        gp01.validate_receipt(tampered, config, root=ROOT)
