from __future__ import annotations

import copy
import json
from functools import lru_cache

import pytest

from sigma_theory_compiler import open_gravity_lane6_gqns_solar_source_domain_v2 as lane


@lru_cache(maxsize=1)
def _built() -> tuple[dict, dict[str, bytes]]:
    return lane.build_receipt()


def test_append_only_v1_and_exact_v2_files_are_hash_bound() -> None:
    config = lane.load_config()
    assert set(lane.validate_predecessor(config)) == {"config", "module", "test", "receipt"}
    assert lane._validate_package_files() == {
        "config_raw_sha256": lane._CONFIG_RAW_SHA256,
        "config_content_sha256": lane._CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": lane._MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": lane._TEST_RAW_SHA256,
    }


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "DECISIVELY_EXCLUDED"),
        (("corrections", "replacement_observational_decision"), "EXCLUDED"),
        (("matched_ephemeris_refit_preflight", "reference_source", "payload_opened"), True),
        (("matched_ephemeris_refit_preflight", "solver", "implementation_complete"), True),
        (("matched_ephemeris_refit_preflight", "fit_contract", "decision_authority"), True),
        (("matched_ephemeris_refit_preflight", "response_gate", "residual_values_opened"), 1),
        (("access_contract", "observational_response_rows_opened"), 1),
    ],
)
def test_material_mutations_fail_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(lane.load_config(verify_package=False))
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(lane.GQNSSolarV2Error):
        lane.validate_config(config)


def test_invalid_inpop_ratio_and_exclusion_are_withdrawn_not_erased() -> None:
    receipt, payloads = _built()
    theorem = json.loads(payloads["corrected-source-only-theorem.json"])
    assert receipt["observational_decision"] == "NOT_EVALUATED__MATCHED_N_BODY_REFIT_REQUIRED"
    assert theorem["observational_decision"] == receipt["observational_decision"]
    assert theorem["superseded_failure"]["v1_ratio"] == pytest.approx(21632.16367487)
    assert theorem["superseded_failure"]["used_for_v2_decision"] is False
    assert theorem["superseded_failure"]["v1_decision"].startswith("DECISIVELY_EXCLUDED")
    stress = theorem["D05_Neptune_source_only_acceleration"]
    assert stress["common_projection_radial_residual_max_m_s2"] > 1.0e-6
    assert "bound" not in stress
    assert "ratio" not in stress


def test_pairwise_spreads_have_true_minimum_maximum_and_minimax_value() -> None:
    _, payloads = _built()
    theorem = json.loads(payloads["corrected-source-only-theorem.json"])
    rows = theorem["pairwise_absolute_spreads"]
    spreads = [row["absolute_spread"] for row in rows]
    assert len(rows) == 6
    assert theorem["minimum_pairwise_absolute_spread"] == min(spreads)
    assert theorem["maximum_pairwise_absolute_spread"] == max(spreads)
    assert theorem["minimum_pairwise_absolute_spread"] == pytest.approx(9.306666548525522e-12)
    assert theorem["maximum_pairwise_absolute_spread"] == pytest.approx(0.44250304447190525)
    assert theorem["minimax_common_constant_maximum_residual_fraction"] == pytest.approx(
        theorem["maximum_pairwise_absolute_spread"] / 2.0
    )


def test_domain_claim_is_limited_to_declared_moon_asteroid_refinements() -> None:
    _, payloads = _built()
    theorem = json.loads(payloads["corrected-source-only-theorem.json"])
    domains = theorem["domain_statement"]
    assert set(domains["D05_D06_D07_declared_refinement_values"]) == {
        "D05_SUN_EIGHT_PLANETS",
        "D06_MOON_SPLIT",
        "D07_ASTEROID_RING",
    }
    assert set(domains["host_only_failures"]) == {
        "D00_SUN_SPHERE_ONLY",
        "D01_SUN_OBLATE_ONLY",
    }
    assert len(domains["remote_source_boundary"]) == 3
    assert "Only the D05-D06-D07" in domains["claim"]


def test_independent_self_force_and_target_minus_sun_checks_close() -> None:
    _, payloads = _built()
    checks = json.loads(payloads["independent-mechanics-projection-checks.json"])
    self_force = checks["self_force"]
    assert self_force["earth_EMB_alias_explicitly_checked"] is True
    assert len(self_force["cases"]) == 8
    assert self_force["maximum_norm_m_s2"] == 0.0
    relative = checks["relative_sun"]
    assert relative["independent_manual_cases"] == 12
    assert relative["maximum_newton_component_error_m_s2"] < 1.0e-18
    assert relative["maximum_dark_component_error_m_s2"] < 1.0e-18
    assert relative["point_sun_closed_form_error_m_s2"] < 1.0e-24


def test_independent_projection_normal_equations_and_reported_scales_close() -> None:
    _, payloads = _built()
    checks = json.loads(payloads["independent-mechanics-projection-checks.json"])
    projection = checks["inverse_square_projection"]
    assert projection["D05_common_scale"] == pytest.approx(0.21507470905469453)
    assert projection["common_normal_equation_relative_error"] < 2.0e-15
    assert max(projection["per_target_normal_equation_relative_errors"].values()) < 3.0e-15
    assert projection["reported_scale_max_absolute_error"] < 6.0e-14
    assert "no observational fit authority" in projection["interpretation"]


def test_concrete_de440_inpop_refit_contract_is_frozen_but_blocked() -> None:
    config = lane.load_config()
    preflight = config["matched_ephemeris_refit_preflight"]
    source = preflight["reference_source"]
    assert source["filename"] == "de440.bsp"
    assert source["published_md5"] == "c9d581bfd84209dbeee8b1583939b148"
    assert source["sha256"] is None
    assert source["payload_downloaded"] is False
    assert source["payload_opened"] is False
    assert preflight["dynamical_model"]["gqns_fitted_parameters"] == 0
    assert preflight["dynamical_model"]["gqns_variants"] == [
        "D05_SUN_EIGHT_PLANETS",
        "D06_MOON_SPLIT",
        "D07_ASTEROID_RING",
    ]
    assert preflight["solver"]["integrator"] == "scipy.integrate.solve_ivp DOP853"
    assert preflight["solver"]["implementation_complete"] is False
    assert preflight["fit_contract"]["same_response_rows_and_weights_for_every_model"] is True
    assert preflight["fit_contract"]["decision_authority"] is False
    gate = preflight["response_gate"]
    assert len(gate["blockers"]) == 6
    assert gate["opening_authorized"] is False
    assert gate["observational_files_opened"] == 0
    assert gate["observational_rows_opened"] == 0
    assert gate["residual_values_opened"] == 0


def test_claim_ceiling_and_access_are_nonobservational() -> None:
    receipt, payloads = _built()
    assert receipt["decision"] == lane._STATUS
    assert receipt["summary"]["v1_source_only_stress_result_preserved"] is True
    assert receipt["summary"]["v1_observational_exclusion_withdrawn"] is True
    assert receipt["summary"]["observational_response_rows_opened"] == 0
    assert receipt["summary"]["refit_execution_authorized"] is False
    assert receipt["retained_failures"]["solver_implementation_incomplete"] is True
    report = payloads["report.md"].decode()
    assert "21632.16367487 INPOP ratio are withdrawn" in report
    assert "not an observed or postfit residual" in report
    assert "makes no observational exclusion" in report


def test_deterministic_receipt_artifacts_and_atomic_replay(tmp_path) -> None:
    first, first_payloads = _built()
    second, second_payloads = lane.build_receipt()
    assert first == second
    assert first_payloads == second_payloads
    assert len(first["artifact_index"]) == 4
    for row in first["artifact_index"]:
        name = lane.Path(row["path"]).name
        payload = first_payloads[name]
        assert len(payload) == row["bytes"]
        assert lane.hashlib.sha256(payload).hexdigest() == row["sha256"]
    receipt_path = tmp_path / "receipt.json"
    receipt_payload = lane.canonical_bytes(first)
    assert lane._atomic_no_clobber(receipt_path, receipt_payload) == "CREATED"
    assert lane._atomic_no_clobber(receipt_path, receipt_payload) == "EXISTING_IDENTICAL"
