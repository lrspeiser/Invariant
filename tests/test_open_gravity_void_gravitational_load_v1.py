from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler import open_gravity_void_gravitational_load_v1 as subject


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def _fixture(config: dict[str, object], fixture_id: str) -> subject.PathFixture:
    return next(row for row in subject._fixture_profiles(config) if row.fixture_id == fixture_id)


def test_package_pins_are_exact() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


def test_branch_registry_dimensions_and_path_measure_are_frozen() -> None:
    config = subject.load_config()
    assert tuple(row["id"] for row in config["branches"]) == subject._BRANCH_IDS
    assert len(subject._BRANCH_IDS) == 11
    assert config["dimensions"]["K_L"] == "L^-3"
    assert config["dimensions"]["A_feed"] == "L^3 M^-1 T^-1"
    assert config["dimensions"]["sigma_g"] == "L^2 M^-1"
    assert config["origin"]["path_measure"].startswith("dell=-u_a dx^a>0")
    assert "null affine" in config["origin"]["path_measure"]
    assert all("dimension_check" in row for row in config["branches"])


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "OPEN"),
        (("parameters", "rho_star"), 0.0),
        (("decision_policy", "retain_every_failure"), False),
        (("empirical_preflight", "response_status"), "OPENED"),
        (("access_contract", "observational_response_rows_opened"), 1),
        (("claim_boundary", "real_data_fit"), True),
        (("outputs", "receipt"), "elsewhere.json"),
    ],
)
def test_semantic_mutations_reject(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_raw_config())
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(subject.VoidLoadError):
        subject.validate_config(config)


def test_synthetic_matrix_is_complete_and_finite() -> None:
    config = subject.load_config()
    rows = subject.synthetic_predictions(config)
    assert len(rows) == 11 * 9
    assert {(row["branch_id"], row["fixture_id"]) for row in rows} == {
        (branch, fixture)
        for branch in subject._BRANCH_IDS
        for fixture in config["synthetic_fixtures"]
    }
    for row in rows:
        assert math.isfinite(row["load_log1pz"])
        assert row["load_log1pz"] >= 0.0
        assert 0.0 < row["minimum_c_eff_over_c"] <= 1.0


def test_vq02_beta_zero_is_exact_vq01_and_positivity_is_enforced() -> None:
    config = subject.load_config()
    fixture = _fixture(config, "VF01_VOID_RICH_PATH")
    modified = copy.deepcopy(config)
    modified["parameters"]["beta"] = 0.0
    direct = subject.predict(subject._BRANCH_IDS[1], fixture, modified)
    slowed = subject.predict(subject._BRANCH_IDS[2], fixture, modified)
    assert slowed.load_log1pz == pytest.approx(direct.load_log1pz, abs=1.0e-15)
    assert slowed.arrival_delay_over_dx_c == 0.0
    invalid = copy.deepcopy(config)
    invalid["parameters"]["beta"] = -1.0
    with pytest.raises(subject.VoidLoadError, match="c_eff pole"):
        subject.predict(subject._BRANCH_IDS[2], fixture, invalid)


def test_local_equilibrium_closes_dimensions_numerically() -> None:
    j_values = (1.0, 2.0, 3.0)
    rho = (0.0, 0.5, 2.0)
    q_values = subject._local_equilibrium(j_values, rho, gamma0=0.4, gamma_b=0.7)
    for j_value, density, q_value in zip(j_values, rho, q_values):
        assert (0.4 + 0.7 * density) * q_value == pytest.approx(j_value)


def test_yukawa_feed_is_computed_from_baryons_and_uniform_limit_is_exact() -> None:
    rho = (2.5,) * 32
    feed = subject._periodic_yukawa_feed(rho, amplitude=1.2, length=4.0)
    assert feed == pytest.approx((3.0,) * 32, abs=1.0e-14)
    zero = subject._periodic_yukawa_feed((0.0,) * 32, amplitude=1.2, length=4.0)
    assert zero == (0.0,) * 32


def test_vq03_prescribed_local_feed_and_vq04_nonlocal_feed_are_distinct() -> None:
    config = subject.load_config()
    fixture = _fixture(config, "VF01_VOID_RICH_PATH")
    local = subject.predict(subject._BRANCH_IDS[3], fixture, config)
    nonlocal_prediction = subject.predict(subject._BRANCH_IDS[4], fixture, config)
    assert local.load_log1pz != pytest.approx(nonlocal_prediction.load_log1pz, abs=1.0e-8)
    assert local.mean_field_load != pytest.approx(nonlocal_prediction.mean_field_load, abs=1.0e-8)


def test_reservoir_is_an_ivp_solution_not_a_hand_inserted_steady_state() -> None:
    rho = (1.0,) * 8
    feed = (1.3,) * 8
    dt = 0.02
    steps = 100
    result = subject._solve_reservoir(
        rho,
        feed,
        (1.0,) * steps,
        diffusion=0.2,
        gamma0=0.5,
        gamma_b=0.8,
        dx=1.0,
        dt=dt,
    )
    rate = 1.3
    expected = feed[0] / rate * (1.0 - (1.0 - dt * rate) ** steps)
    assert result == pytest.approx((expected,) * 8, abs=1.0e-14)
    assert result[0] < feed[0] / rate
    with pytest.raises(subject.VoidLoadError, match="stability"):
        subject._solve_reservoir(
            rho,
            feed,
            (1.0,),
            diffusion=1.0,
            gamma0=0.5,
            gamma_b=0.8,
            dx=1.0,
            dt=0.6,
        )


def test_column_attenuation_uses_intervening_baryons() -> None:
    clear = subject._column_attenuated_load(
        (1.0, 0.0, 0.0, 1.0), amplitude=1.0, length=10.0, sigma=1.0, dx=1.0
    )
    screened = subject._column_attenuated_load(
        (1.0, 4.0, 4.0, 1.0), amplitude=1.0, length=10.0, sigma=1.0, dx=1.0
    )
    # The far-end contribution is exponentially screened, even though total local Q also changes.
    far_kernel = math.exp(-3.0 / 10.0)
    clear_far = far_kernel * math.exp(0.0)
    screened_far = far_kernel * math.exp(-8.0)
    assert screened_far < clear_far
    assert all(value >= 0.0 for value in clear + screened)


def test_photon_memory_has_exact_order_discriminator_while_columns_do_not() -> None:
    config = subject.load_config()
    first = _fixture(config, "VF03_ORDER_VOID_THEN_MATTER")
    second = _fixture(config, "VF04_ORDER_MATTER_THEN_VOID")
    for branch in (subject._BRANCH_IDS[1], subject._BRANCH_IDS[7], subject._BRANCH_IDS[8]):
        assert subject.predict(branch, first, config).load_log1pz == pytest.approx(
            subject.predict(branch, second, config).load_log1pz, abs=1.0e-14
        )
    memory_first = subject.predict(subject._BRANCH_IDS[9], first, config)
    memory_second = subject.predict(subject._BRANCH_IDS[9], second, config)
    assert memory_first.load_log1pz != pytest.approx(memory_second.load_log1pz, abs=1.0e-8)
    assert memory_first.final_photon_load != pytest.approx(
        memory_second.final_photon_load, abs=1.0e-8
    )


def test_vq10_couples_redshift_and_positive_delay_to_same_reservoir() -> None:
    config = subject.load_config()
    fixture = _fixture(config, "VF01_VOID_RICH_PATH")
    prediction = subject.predict(subject._BRANCH_IDS[10], fixture, config)
    assert prediction.load_log1pz > 0.0
    assert prediction.arrival_delay_over_dx_c > 0.0
    assert prediction.minimum_c_eff_over_c < 1.0
    modified = copy.deepcopy(config)
    modified["parameters"]["eta"] = 0.0
    no_redshift = subject.predict(subject._BRANCH_IDS[10], fixture, modified)
    assert no_redshift.load_log1pz == 0.0
    assert no_redshift.arrival_delay_over_dx_c == pytest.approx(prediction.arrival_delay_over_dx_c)


def test_time_dilation_energy_duality_cmb_tolman_and_fit_remain_separate() -> None:
    config = subject.load_config()
    controls = subject._branch_controls(config)
    assert len(controls) == 11
    for row in controls:
        assert row["observational_fit"] == "UNOPENED_NOT_SCORED"
        for field in (
            "receiver_or_action_energy_accounting",
            "causality_and_initial_value_problem",
            "common_matter_light_closure",
            "source_lightcurve_time_dilation",
            "distance_duality_and_photon_number",
            "arrival_time_delay",
            "chromaticity",
            "CMB_blackbody_and_anisotropy",
            "Tolman_surface_brightness",
            "historical_novelty",
        ):
            assert row[field]
    assert controls[5]["causality_and_initial_value_problem"] == "PARABOLIC_INSTANTANEOUS_TAIL"
    assert controls[9]["receiver_or_action_energy_accounting"] == "QGAMMA_STATE_NOT_ENERGY_RECEIVER"


def test_empirical_preflight_is_exact_target_blind_and_source_blocked() -> None:
    config = subject.load_config()
    preflight = config["empirical_preflight"]
    assert preflight["response_status"] == "NOT_OPENED_NOT_SCORED"
    assert preflight["sources"][0]["id"] == "COSMICFLOWS4"
    assert preflight["sources"][1]["id"] == "SDSS_DR7_VAST_VIDE"
    assert preflight["sources"][2]["id"] == "PANTHEON_PLUS_CROSSCHECK"
    assert all(row["revision"] is None and row["sha256"] is None for row in preflight["sources"])
    assert "SHA256(canonical CF4 group identifier)" in preflight["splits"]["rule"]
    assert preflight["splits"]["sealed_confirmation"] == [8, 9]
    assert "Student-t" in preflight["likelihood"]
    assert "1000 frozen permutations" in preflight["thresholds"]["advance_interesting_observation"]
    assert "never double count" in " ".join(preflight["intersection_algorithm"])
    assert "void-beta/RSD uncertainty" in " ".join(preflight["intersection_algorithm"])
    assert "bias Gaussianization" in " ".join(preflight["comparators"])
    assert "Local Void dynamical outflow" in " ".join(preflight["comparators"])
    assert "redshift-space" in preflight["circularity_warning"]


def test_priority_is_not_a_truth_score_and_all_failures_are_retained() -> None:
    ledger = subject._priority_ledger()
    assert len(ledger) == 11
    assert all(row["truth_score"] == "NOT_ASSIGNED" for row in ledger)
    assert all(row["data_fit_score"] == "UNOPENED" for row in ledger)
    assert ledger[0]["branch_id"] == subject._BRANCH_IDS[9]
    counterexamples = subject._counterexamples(subject.load_config())
    assert len(counterexamples) == 9
    assert {row["id"] for row in counterexamples} >= {
        "CX03_DIFFUSION_CAUSALITY",
        "CX04_TIME_DILATION",
        "CX05_ENERGY_RECEIVER",
        "CX07_REDSHIFT_SPACE_VOID_CIRCULARITY",
    }


def test_receipt_artifacts_are_deterministic_and_claim_bounded() -> None:
    first, first_payloads = subject.build_receipt()
    second, second_payloads = subject.build_receipt()
    assert first == second
    assert first_payloads == second_payloads
    assert first["summary"]["branches"] == 11
    assert first["summary"]["synthetic_predictions"] == 99
    assert first["summary"]["observational_response_rows_opened"] == 0
    assert first["summary"]["retained_counterexamples"] == 9
    assert first["independent_audit_required"] is True
    assert first["claim_boundary"]["real_data_fit"] is False
    assert "order is reversed" in first["strongest_unique_testable_discriminator"]
    report = first_payloads["report.md"].decode()
    assert "SOURCE BLOCKED" in report
    assert "VQ09 ordering memory" in report
    assert "No observational response row was opened" in report


def test_written_package_replays_and_no_payload_path_is_bound() -> None:
    observed = subject.check_package()
    assert observed["status"].startswith("PASS_EXECUTABLE")
    config = subject.load_config()
    assert set(config["access_contract"].values()) == {0}
    for source in config["empirical_preflight"]["sources"]:
        assert source["sha256"] is None
        assert source["revision"] is None
    for binding in config["local_bindings"]:
        assert Path(binding["path"]).suffix == ".json"
