from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.nasa_exoplanet_task1 import (
    CONFIG_PATH,
    EXPECTED_COLUMNS,
    FORBIDDEN_GENERATOR_WORDS,
    GENERATOR_PATH,
    NASAExoplanetTask1Error,
    build_source_uri,
    generator_leakage_audit,
    load_config,
    parse_snapshot,
    split_and_sanitize,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT)
RECEIPT_PATH = ROOT / CONFIG["outputs"]["receipt"]
TRAINING_PATH = ROOT / CONFIG["outputs"]["sanitized_training_rows"]
SNAPSHOT_PATH = ROOT / CONFIG["outputs"]["source_snapshot"]
CONFIRMATION_CONFIG_PATH = "configs/nasa_exoplanet_task1_confirmation.json"
CONFIRMATION_CONFIG = load_config(ROOT, CONFIRMATION_CONFIG_PATH)
CONFIRMATION_RECEIPT_PATH = ROOT / CONFIRMATION_CONFIG["outputs"]["receipt"]
CONFIRMATION_TRAINING_PATH = ROOT / CONFIRMATION_CONFIG["outputs"]["sanitized_training_rows"]
CONFIRMATION_SNAPSHOT_PATH = ROOT / CONFIRMATION_CONFIG["outputs"]["source_snapshot"]


@pytest.fixture(scope="module")
def receipt() -> dict:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def training() -> dict:
    return json.loads(TRAINING_PATH.read_text(encoding="utf-8"))


def test_source_is_external_https_and_query_is_frozen() -> None:
    uri = build_source_uri(CONFIG)
    assert uri.startswith("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?")
    assert "format=csv" in uri
    assert CONFIG["source"]["external_principal_id"] == "external.nasa-exoplanet-archive"
    assert CONFIG["source"]["query"].startswith("select hostname,pl_name,disc_year")
    assert CONFIG["source"]["query"].endswith("st_mass is not null")


def test_archived_snapshot_has_the_committed_external_identity(receipt: dict) -> None:
    raw = SNAPSHOT_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    assert hashlib.sha256(raw).hexdigest() == receipt["source"]["normalized_snapshot_sha256"]
    assert receipt["source"]["response_bytes"] == 795_928
    assert receipt["source"]["normalized_snapshot_sha256"] == (
        "a35e75d1fd3bca2480896b29f1516815a86468ccacbba7a851e629863d53accf"
    )


def test_snapshot_parses_to_large_real_host_inventory() -> None:
    eligible, exclusions = parse_snapshot(SNAPSHOT_PATH.read_bytes(), CONFIG)
    assert len(eligible) == 2_531
    assert len({row["host"] for row in eligible}) == 1_911
    assert exclusions == {
        "missing_or_invalid_required_value": 323,
        "relative_uncertainty_above_limit": 45,
    }


def test_header_or_source_drift_fails_closed() -> None:
    raw = SNAPSHOT_PATH.read_bytes()
    original = ",".join(EXPECTED_COLUMNS).encode()
    drifted = raw.replace(original, original.replace(b"hostname", b"host"), 1)
    with pytest.raises(NASAExoplanetTask1Error, match="columns changed"):
        parse_snapshot(drifted, CONFIG)


def test_split_is_host_disjoint_and_newest_by_host_order() -> None:
    eligible, _ = parse_snapshot(SNAPSHOT_PATH.read_bytes(), CONFIG)
    public, sealed, summary = split_and_sanitize(eligible, CONFIG)
    assert summary == {
        "holdout_host_count": 383,
        "holdout_maximum_discovery_year": 2026,
        "holdout_minimum_discovery_year": 2003,
        "holdout_row_count": 511,
        "host_intersection_count": 0,
        "training_host_count": 1528,
        "training_maximum_discovery_year": 2024,
        "training_minimum_discovery_year": 1996,
        "training_row_count": 2020,
    }
    assert len(public) == 2_020
    assert len(sealed) == 511
    assert {row["label"] for row in public}.isdisjoint(row["label"] for row in sealed)


def test_discovery_input_contains_only_neutral_rows(training: dict) -> None:
    assert training["claims"] == {
        "column_meanings_present": False,
        "holdout_rows_present": False,
        "target_formula_present": False,
    }
    assert training["anonymous_columns"] == ["x0", "x1", "x2"]
    assert len(training["rows"]) == 2_020
    assert all(set(row) == {"label", "uncertainties", "values"} for row in training["rows"])
    assert all(row["label"].startswith("r") for row in training["rows"])


def test_generator_and_training_input_pass_semantic_and_numeric_leak_scan(training: dict) -> None:
    audit = generator_leakage_audit(ROOT, training["rows"])
    assert audit["passed"] is True
    assert audit["forbidden_numeric_fingerprint_hits"] == []
    assert audit["forbidden_vocabulary_hits_in_generator"] == []
    assert audit["forbidden_vocabulary_hits_in_training_input"] == []
    generator_text = (ROOT / GENERATOR_PATH).read_text(encoding="utf-8").lower()
    assert "kepler" not in generator_text
    assert "nasa" not in generator_text
    assert "exoplanet" not in generator_text
    assert {"kepler", "nasa", "exoplanet"} <= FORBIDDEN_GENERATOR_WORDS


def test_new_lane_recovers_the_exact_three_column_structure(receipt: dict) -> None:
    new = receipt["candidate_phase"]["new_search"]
    old = receipt["candidate_phase"]["old_search"]
    assert new["candidate_budget"] == old["candidate_budget"] == 256
    assert new["best_candidate"]["exponents"] == [2, -3, 1]
    assert new["best_candidate"]["expression"] == "x0^2*x1^-3*x2 = constant"
    assert old["best_candidate"]["exponents"] == [2, -3, 0]
    assert receipt["target_opening"]["canonical_primitive_exponents"] == [2, -3, 1]


def test_every_compared_candidate_is_frozen_before_unseal(receipt: dict) -> None:
    assert [row["event"] for row in receipt["chronology"]] == [
        "source_snapshot_hashed",
        "host_disjoint_split_committed",
        "anonymous_training_input_audited",
        "all_candidates_frozen",
        "target_fixture_and_holdout_opened_for_scoring",
    ]
    assert all(row["holdout_rows_exposed_to_generator"] == 0 for row in receipt["chronology"])
    assert receipt["candidate_phase_sha256"] == canonical_sha256(receipt["candidate_phase"])


def test_real_holdout_performance_passes_every_prespecified_numeric_check(receipt: dict) -> None:
    assert receipt["checks"] == {
        "candidate_frozen_before_holdout_scoring": True,
        "exact_target_structure": True,
        "host_disjoint_split": True,
        "new_better_than_every_baseline": True,
        "new_better_than_old": True,
        "unit_rescaling_stability": True,
        "within_1sigma": True,
        "within_2sigma": True,
    }
    result = receipt["evaluation"]["new_holdout"]
    assert float(result["median_absolute_response_log_error"]) < 0.003
    assert float(result["within_1sigma_fraction"]) > 0.92
    assert float(result["within_2sigma_fraction"]) > 0.97


def test_new_relation_beats_old_and_flexible_prediction_baselines(receipt: dict) -> None:
    new_error = float(receipt["evaluation"]["new_holdout"]["median_absolute_response_log_error"])
    old_error = float(receipt["evaluation"]["old_holdout"]["median_absolute_response_log_error"])
    baseline_errors = {
        row["baseline_id"]: float(row["holdout_median_absolute_response_log_error"])
        for row in receipt["baselines"]
    }
    assert new_error < old_error
    assert new_error < min(baseline_errors.values())
    assert baseline_errors["unconstrained_multivariate_log_linear"] < 0.01
    assert baseline_errors["unconstrained_quadratic_log_predictors"] < 0.01


def test_random_comparison_is_budget_matched_and_reports_failures(receipt: dict) -> None:
    random_searches = receipt["candidate_phase"]["random_searches"]
    assert len(random_searches) == receipt["evaluation"]["random_replicates"] == 32
    assert {row["candidate_budget"] for row in random_searches} == {256}
    assert receipt["evaluation"]["random_best_exact_target_count"] == 1
    assert receipt["evaluation"]["random_better_or_equal_to_new_count"] == 1


def test_unit_conversion_changes_the_constant_not_the_structure(receipt: dict) -> None:
    original = receipt["candidate_phase"]["new_search"]["best_candidate"]
    converted = receipt["candidate_phase"]["unit_rescaling_search"]["best_candidate"]
    assert original["exponents"] == converted["exponents"] == [2, -3, 1]
    assert original["fit_log_constant"] != converted["fit_log_constant"]


def test_pilot_is_blocked_from_unlocking_task_2(receipt: dict) -> None:
    assert receipt["decision"] == "BLOCKED"
    assert receipt["observed_status"] == "EXPLORATORY_PASS_NOT_GATE_ELIGIBLE"
    assert receipt["claims"] == {
        "creative_method_established": False,
        "data_columns_are_independent_direct_measurements": False,
        "gate_eligible": False,
        "historically_novel": False,
        "independent_physical_confirmation": False,
        "known_result_recovered": True,
        "level5_eligible": False,
        "llm_calls_made": 0,
        "real_external_catalog_snapshot_used": True,
        "task_1_completed": False,
    }
    assert receipt["target_opening"]["claims"]["known_result_calibration"] is True
    assert receipt["target_opening"]["claims"]["independent_physical_confirmation"] is False


def test_training_and_receipt_commitments_open(receipt: dict, training: dict) -> None:
    assert receipt["training_artifact_sha256"] == canonical_sha256(training)
    assert receipt["split"]["training_commitment_sha256"] == canonical_sha256(training["rows"])
    assert receipt["config_sha256"] == canonical_sha256(CONFIG)


@pytest.mark.empirical_validation
def test_complete_committed_pilot_replays(receipt: dict, training: dict) -> None:
    assert validate_campaign(ROOT, receipt, training, SNAPSHOT_PATH.read_bytes()) == receipt


def test_receipt_or_artifact_schema_tamper_fails_closed(receipt: dict, training: dict) -> None:
    drifted_receipt = copy.deepcopy(receipt)
    drifted_receipt["schema_version"] = "changed"
    with pytest.raises(NASAExoplanetTask1Error, match="receipt schema changed"):
        validate_campaign(ROOT, drifted_receipt, training, SNAPSHOT_PATH.read_bytes())
    drifted_training = copy.deepcopy(training)
    drifted_training["schema_version"] = "changed"
    with pytest.raises(NASAExoplanetTask1Error, match="training artifact schema changed"):
        validate_campaign(ROOT, receipt, drifted_training, SNAPSHOT_PATH.read_bytes())


def test_config_itself_records_why_this_is_not_a_gate_pass() -> None:
    raw = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert raw["classification"]["gate_eligible"] is False
    assert "inspected" in raw["classification"]["reason"].lower()
    assert "inferred" in raw["source"]["dependency_warning"].lower()


def test_confirmation_query_and_thresholds_are_frozen_before_value_retrieval() -> None:
    config = CONFIRMATION_CONFIG
    assert config["classification"]["gate_eligible"] is True
    assert "untouched" in config["classification"]["reason"].lower()
    assert "default_flag=0" in config["source"]["query"]
    assert "pl_pubdate>='2020-01-01'" in config["source"]["query"]
    assert config["discovery"]["candidate_budget_per_run"] == 256
    assert config["discovery"]["exponent_bound"] == 12
    assert config["evaluation"] == CONFIG["evaluation"]


def test_gate_eligible_build_refuses_data_without_pre_retrieval_authorization() -> None:
    from sigma_theory_compiler.nasa_exoplanet_task1 import build_campaign

    with pytest.raises(NASAExoplanetTask1Error, match="lacks a frozen authorization"):
        build_campaign(
            ROOT,
            SNAPSHOT_PATH.read_bytes(),
            retrieved_at="2026-08-26T18:00:00Z",
            config_path=CONFIRMATION_CONFIG_PATH,
        )


@pytest.mark.empirical_validation
def test_confirmation_receipt_passes_when_external_run_has_been_completed() -> None:
    if not CONFIRMATION_RECEIPT_PATH.exists():
        pytest.skip("confirmation source values have not been retrieved after freeze yet")
    receipt = json.loads(CONFIRMATION_RECEIPT_PATH.read_text(encoding="utf-8"))
    training = json.loads(CONFIRMATION_TRAINING_PATH.read_text(encoding="utf-8"))
    assert receipt["authorization"] is not None
    assert receipt["config_path"] == CONFIRMATION_CONFIG_PATH
    assert receipt["decision"] in {"PASS", "REJECT"}
    assert receipt["claims"]["gate_eligible"] is True
    assert receipt["claims"]["task_1_completed"] is (receipt["decision"] == "PASS")
    assert validate_campaign(
        ROOT, receipt, training, CONFIRMATION_SNAPSHOT_PATH.read_bytes()
    ) == receipt
