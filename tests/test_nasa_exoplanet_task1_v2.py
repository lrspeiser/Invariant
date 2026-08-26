from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import nasa_exoplanet_task1 as v1
from sigma_theory_compiler.nasa_exoplanet_task1_v2 import (
    CONFIG_PATH,
    NASAExoplanetTask1V2Error,
    build_source_uri,
    load_config,
    validate_authorization,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_config(ROOT)
RECEIPT_PATH = ROOT / CONFIG["outputs"]["receipt"]
TRAINING_PATH = ROOT / CONFIG["outputs"]["sanitized_training_rows"]
SNAPSHOT_PATH = ROOT / CONFIG["outputs"]["source_snapshot"]
V1_REJECT_PATH = ROOT / "runs/math/nasa-exoplanet-task1/confirmation-v1.json"


def test_v2_uses_a_value_untouched_source_lane() -> None:
    query = CONFIG["source"]["query"]
    assert CONFIG["classification"]["gate_eligible"] is True
    assert "untouched" in CONFIG["classification"]["reason"].lower()
    assert "default_flag=0" in query
    assert "pl_pubdate>='2015-01-01'" in query
    assert "pl_pubdate<'2020-01-01'" in query
    assert "pl_pubdate>='2020-01-01'" not in query
    assert all(f"{column} is not null" in query for column in v1.VALUE_COLUMNS)
    assert build_source_uri(CONFIG).startswith(
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
    )


def test_v2_keeps_search_budgets_identical_to_v1() -> None:
    old_config = v1.load_config(ROOT, "configs/nasa_exoplanet_task1_confirmation.json")
    assert CONFIG["discovery"] == old_config["discovery"]
    assert CONFIG["split"] == old_config["split"]
    assert CONFIG["eligibility"]["maximum_relative_uncertainty"] == "0.5"


def test_v2_replaces_the_miscalibrated_coverage_gate_before_opening_values() -> None:
    evaluation = CONFIG["evaluation"]
    assert evaluation["minimum_empirical_1sigma_coverage"] == "0.6"
    assert evaluation["minimum_empirical_2sigma_coverage"] == "0.9"
    assert evaluation["maximum_p90_standardized_residual"] == "2.0"
    assert float(evaluation["reference_gaussian_1sigma_coverage"]) == pytest.approx(0.682689492)
    assert float(evaluation["reference_gaussian_2sigma_coverage"]) == pytest.approx(0.954499736)
    assert "Version 1 remains REJECT" in evaluation["revision_reason"]


def test_v1_reject_is_preserved_and_still_replays() -> None:
    receipt = json.loads(V1_REJECT_PATH.read_text(encoding="utf-8"))
    training = json.loads(
        (
            ROOT
            / "runs/math/nasa-exoplanet-task1/confirmation-training-v1.json"
        ).read_text(encoding="utf-8")
    )
    snapshot = ROOT / "runs/math/nasa-exoplanet-task1/nasa-ps-alternate-source-snapshot-v1.csv"
    assert receipt["decision"] == "REJECT"
    assert receipt["checks"]["within_1sigma"] is False
    assert receipt["checks"]["within_2sigma"] is False
    assert v1.validate_campaign(ROOT, receipt, training, snapshot.read_bytes()) == receipt


@pytest.mark.empirical_validation
def test_v2_confirmation_replays_after_the_frozen_run() -> None:
    if not RECEIPT_PATH.exists():
        pytest.skip("v2 values have not been retrieved after implementation freeze")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    training = json.loads(TRAINING_PATH.read_text(encoding="utf-8"))
    assert receipt["decision"] in {"PASS", "REJECT"}
    assert receipt["claims"]["task_1_completed"] is (receipt["decision"] == "PASS")
    assert receipt["claims"]["known_result_recovered"] is True
    assert receipt["claims"]["historically_novel"] is False
    assert receipt["claims"]["independent_physical_confirmation"] is False
    assert receipt["candidate_phase"]["new_search"]["candidate_budget"] == 256
    assert len(receipt["candidate_phase"]["random_searches"]) == 32
    assert validate_campaign(ROOT, receipt, training, SNAPSHOT_PATH.read_bytes()) == receipt


def test_v2_authorization_tamper_fails_after_the_frozen_run() -> None:
    if not RECEIPT_PATH.exists():
        pytest.skip("v2 authorization has not been created yet")
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    drifted = copy.deepcopy(receipt["authorization"])
    drifted["source_query_sha256"] = "0" * 64
    with pytest.raises(NASAExoplanetTask1V2Error, match="commitment changed"):
        validate_authorization(ROOT, drifted, CONFIG_PATH)

