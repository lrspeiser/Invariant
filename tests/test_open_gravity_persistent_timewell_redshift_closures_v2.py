from __future__ import annotations

import copy
import json

import pytest

from sigma_theory_compiler import open_gravity_persistent_timewell_redshift_closures_v2 as subject


def _raw_config() -> dict[str, object]:
    return json.loads(subject.CONFIG_PATH.read_text(encoding="utf-8"))


def test_exact_package_pins() -> None:
    assert subject.file_sha256(subject.CONFIG_PATH) == subject._CONFIG_RAW_SHA256
    assert subject.content_sha256(_raw_config()) == subject._CONFIG_CONTENT_SHA256
    assert subject.module_semantic_sha256() == subject._MODULE_SEMANTIC_SHA256
    assert subject.file_sha256(subject.TEST_PATH) == subject._TEST_RAW_SHA256


def test_strict_claim_boundaries_and_old_receipt_preservation() -> None:
    config = subject.load_config()
    claims = config["claim_boundary"]
    assert claims["closure_taxonomy_complete"] is False
    assert claims["global_identifiability_established"] is False
    assert claims["C08_independent_of_C07"] is False
    assert claims["congruence_independence_claimed"] is False
    assert claims["real_data_scored"] is False
    assert (
        subject.file_sha256(subject.Path(config["supersedes"]["receipt_path"]))
        == config["supersedes"]["receipt_sha256"]
    )
    assert subject.Path(config["supersedes"]["blocked_audit_record"]).is_file()


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("status",), "OPEN"),
        (("claim_boundary", "closure_taxonomy_complete"), True),
        (("claim_boundary", "C08_independent_of_C07"), True),
        (("galileo_metadata_manifest", "payload_rows_opened"), 1),
        (("galileo_frozen_analysis", "tau_seconds"), [1.0]),
        (("outputs", "receipt"), "wrong.json"),
    ],
)
def test_semantic_mutations_reject(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(_raw_config())
    target = config
    for key in path[:-1]:
        target = target[key]  # type: ignore[index,assignment]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(subject.PersistentRedshiftV2Error):
        subject.validate_config(config)


def test_physical_path_measure_is_positive_reparameterization_invariant_and_congruence_dependent() -> (
    None
):
    stationary = subject.physical_path_measure((2.0, 2.0), 0.0)
    boosted = subject.physical_path_measure((2.0, 2.0), 0.4)
    split = sum(
        subject.physical_path_measure((2.0 * fraction, 2.0 * fraction), 0.4)
        for fraction in (0.2, 0.3, 0.5)
    )
    assert stationary > 0.0
    assert boosted > 0.0
    assert split == pytest.approx(boosted, abs=1.0e-14)
    assert boosted != pytest.approx(stationary)
    with pytest.raises(subject.PersistentRedshiftV2Error):
        subject.physical_path_measure((2.0, 1.0), 0.0)


def test_all_cumulative_closures_use_physical_measure_and_no_affine_ds() -> None:
    config = subject.load_config()
    cumulative = {
        "C02_POTENTIAL_COLUMN",
        "C03_TIDAL_CURVATURE_COLUMN",
        "C04_DISPERSIVE_PERSISTENT_MEDIUM",
        "C06_PATH_MEMORY_OPACITY",
        "C07_GENERAL_TIME_VARYING_METRIC_PATH",
        "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY",
    }
    by_id = {row["id"]: row for row in config["closures"]}
    for closure_id in cumulative:
        assert by_id[closure_id]["path_measure"] == "dell=-uhat_a dx^a"
        assert " ds" not in by_id[closure_id]["equation"]
        assert "lambda" not in by_id[closure_id]["equation"]
    assert config["geometric_contract"]["forbidden_measure"].startswith("an unspecified affine ds")


def test_executable_common_histories_and_initial_conditions_generate_states() -> None:
    config = subject.load_config()
    assert config["synthetic_contract"]["initial_condition"].startswith("psi(0,x)=0")
    times, pulse_fast, derivative_fast = subject._memory_series("H_PULSE", 0.5)
    _, pulse_slow, derivative_slow = subject._memory_series("H_PULSE", 1.5)
    assert pulse_fast[0] == pulse_slow[0] == 0.0
    assert len(times) == len(pulse_fast) == len(derivative_fast) == 121
    assert pulse_fast[subject._time_index(2.0)] > pulse_slow[subject._time_index(2.0)]
    assert pulse_fast[subject._time_index(4.0)] < pulse_slow[subject._time_index(4.0)]
    assert derivative_fast != derivative_slow


def test_exact_endpoint_equivalence_and_causal_metric_nesting_are_computed() -> None:
    _, vectors = subject._signature_vectors()
    relations = subject._relation_rows(vectors)
    by_pair = {(row["left"], row["right"]): row for row in relations}
    endpoint = by_pair[("C00_GR_ENDPOINT", "C01_EXACT_GRADIENT_PATH")]
    nesting = by_pair[
        (
            "C07_GENERAL_TIME_VARYING_METRIC_PATH",
            "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY",
        )
    ]
    assert endpoint["relation"] == "EQUIVALENT_ON_EXECUTED_FAMILY"
    assert endpoint["minimum_fixture_vector_distance"] == "0.000000000000e+00"
    assert nesting["relation"] == "RIGHT_CONSTRAINED_SUBFAMILY_OF_LEFT"
    assert nesting["shared_member_vectors"] == 2
    assert all(row["global_identifiability_claimed"] is False for row in relations)


def test_no_hand_authored_metric_or_causal_fixture_outputs_remain() -> None:
    for row in subject._fixture_rows():
        assert "metric_dt" not in row
        assert "causal_dt" not in row
        general = subject._predict_members("C07_GENERAL_TIME_VARYING_METRIC_PATH", row)
        causal = subject._predict_members("C08_CAUSAL_METRIC_MEMORY_SUBFAMILY", row)
        assert causal == {key: general[key] for key in ("G_MEM_FAST", "G_MEM_SLOW")}


def test_gauge_chromatic_lens_roundtrip_and_expansion_controls_remain() -> None:
    rows = {(row["fixture_id"], row["case_id"]): row for row in subject._fixture_rows()}
    gauge = rows[("F03_POTENTIAL_ZERO_SHIFT", "base")]
    shifted = rows[("F03_POTENTIAL_ZERO_SHIFT", "shift_plus_5")]
    assert subject._predict_members("C00_GR_ENDPOINT", gauge)["BASE"] == pytest.approx(
        subject._predict_members("C00_GR_ENDPOINT", shifted)["BASE"]
    )
    assert subject._predict_members("C02_POTENTIAL_COLUMN", gauge) != subject._predict_members(
        "C02_POTENTIAL_COLUMN", shifted
    )

    low = rows[("F05_TWO_FREQUENCIES", "nu_1")]
    high = rows[("F05_TWO_FREQUENCIES", "nu_2")]
    low_value = subject._predict_members("C04_DISPERSIVE_PERSISTENT_MEDIUM", low)["TAU_FAST"]
    high_value = subject._predict_members("C04_DISPERSIVE_PERSISTENT_MEDIUM", high)["TAU_FAST"]
    assert low_value / high_value == pytest.approx(4.0)

    image_a = rows[("F06_LENS_TWO_IMAGES", "image_A")]
    image_b = rows[("F06_LENS_TWO_IMAGES", "image_B")]
    assert subject._predict_members("C00_GR_ENDPOINT", image_a) == subject._predict_members(
        "C00_GR_ENDPOINT", image_b
    )
    assert subject._predict_members(
        "C03_TIDAL_CURVATURE_COLUMN", image_a
    ) != subject._predict_members("C03_TIDAL_CURVATURE_COLUMN", image_b)

    outbound = rows[("F07_ROUND_TRIP", "outbound")]
    returning = rows[("F07_ROUND_TRIP", "return")]
    endpoint_sum = (
        subject._predict_members("C00_GR_ENDPOINT", outbound)["BASE"]
        + subject._predict_members("C00_GR_ENDPOINT", returning)["BASE"]
    )
    opacity_sum = (
        subject._predict_members("C06_PATH_MEMORY_OPACITY", outbound)["TAU_FAST"]
        + subject._predict_members("C06_PATH_MEMORY_OPACITY", returning)["TAU_FAST"]
    )
    assert endpoint_sum == pytest.approx(0.0)
    assert opacity_sum < 0.0

    expansion = rows[("F08_EXPANSION_TIME_DILATION", "z_0p5")]
    assert expansion["time_stretch"] == 1.5
    for closure_id in subject._CLOSURE_IDS:
        assert set(subject._predict_members(closure_id, expansion).values()) == {0.0}


def test_structural_triage_is_computed_and_not_an_empirical_score() -> None:
    config = subject.load_config()
    _, vectors = subject._signature_vectors()
    first = subject._triage_rows(config, vectors)
    second = subject._triage_rows(config, vectors)
    assert first == second
    assert {row["closure_id"] for row in first} == set(subject._CANDIDATES)
    assert all(
        row["score_kind"] == "DETERMINISTIC_STRUCTURAL_PREFLIGHT_NOT_DATA_OR_MODEL_SCORE"
        for row in first
    )
    assert all(row["empirical_rows_used"] == 0 for row in first)
    assert (
        next(row for row in first if row["closure_id"].startswith("C08_"))[
            "nested_in_general_neighbor"
        ]
        == 1
    )


def test_exact_galileo_metadata_manifest_grid_nuisance_and_likelihood_are_frozen() -> None:
    config = subject.load_config()
    manifest = config["galileo_metadata_manifest"]
    entries = manifest["directory_entries"]
    assert len(entries) == 14
    assert {row["name"] for row in entries} == {
        f"esoc2044{day}.{suffix}" for day in range(7) for suffix in ("clk", "sp3")
    }
    assert all(row["url"].endswith(row["name"]) for row in entries)
    assert manifest["payload_sha256_status"] == "NOT_DOWNLOADED_NOT_HASHED"
    assert manifest["payload_rows_opened"] == 0
    assert {(row["spacecraft"], row["prn"]) for row in manifest["satellite_identity"]} == {
        ("GSAT0201", "E18"),
        ("GSAT0202", "E14"),
    }
    analysis = config["galileo_frozen_analysis"]
    assert analysis["tau_seconds"] == [300.0, 1800.0, 7200.0, 21600.0, 43200.0]
    assert analysis["epoch_grid"].startswith("exact intersection")
    assert "AR(1)" in analysis["likelihood"]
    assert "E14 development" in analysis["likelihood"]
    assert "seven leave-one-day-out folds" in analysis["likelihood"]
    assert "each fitting nuisance and beta on six E14 days" in analysis["likelihood"]
    assert "leave-one-day-in" not in analysis["likelihood"]
    assert "periodic fixed point" in analysis["memory_template"]
    assert "no daily reset" in analysis["memory_template"]
    assert analysis["primary_statistic"].startswith("E18 minus-null")
    assert len(analysis["nuisance_design"]) == 6
    assert len(analysis["required_sign_checks"]) == 4


def test_deterministic_receipt_artifacts_and_strict_report() -> None:
    first, first_payloads = subject.build_receipt()
    second, second_payloads = subject.build_receipt()
    assert first == second
    assert first_payloads == second_payloads
    assert first["summary"]["closure_taxonomy_complete"] is False
    assert first["summary"]["global_identifiability_established"] is False
    assert first["summary"]["structural_pair_relations"] == 36
    assert first["summary"]["relation_counts"]["EQUIVALENT_ON_EXECUTED_FAMILY"] >= 1
    assert first["summary"]["relation_counts"]["RIGHT_CONSTRAINED_SUBFAMILY_OF_LEFT"] >= 1
    assert first["summary"]["maximum_reparameterization_error"] <= 1.0e-12
    assert first["summary"]["galileo_manifest_files"] == 14
    assert first["summary"]["observational_payload_rows_opened"] == 0
    report = first_payloads["strict-audit-report.md"].decode()
    assert "No unspecified affine `ds` remains" in report
    assert "C08_CAUSAL_METRIC_MEMORY_SUBFAMILY` is exactly nested" in report
    assert "not data scores" in report
    assert "SOURCE_BLOCKED" in report


def test_written_package_checks_and_forgery_rejects() -> None:
    observed = subject.check_package()
    assert observed["status"].startswith("PASS_STRICT_RESPONSE_FREE")
    forged = copy.deepcopy(observed)
    forged["summary"]["observational_payload_rows_opened"] = 1
    assert forged != subject.build_receipt()[0]
