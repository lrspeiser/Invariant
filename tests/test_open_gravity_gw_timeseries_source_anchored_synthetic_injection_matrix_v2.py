from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v2 as gw,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[1]


def test_v2_binds_v1_byte_exact_and_changes_only_the_test_expectation() -> None:
    config = gw.load_config()
    gw.validate_config(config)
    assert config["correction"] == {
        "blocked_v1_assertion": "RESULT_HASH_COUNT_INCORRECTLY_EXCLUDED_RETAINED_NUMERICAL_INVALID_EXECUTIONS",
        "expected_result_hash_count": 12_240,
        "numerical_invalid_result_hash_count": 680,
        "scored_result_hash_count": 11_560,
        "scope": "TEST_EXPECTATION_ONLY_NO_SCIENTIFIC_OR_MATRIX_SEMANTICS_CHANGED",
    }
    assert all(value == 0 for value in config["access_contract"].values())


def test_v2_material_mutation_fails_closed() -> None:
    config = copy.deepcopy(gw.load_config())
    config["correction"]["expected_result_hash_count"] = 11_560
    with pytest.raises(SchemaViolation):
        gw.validate_config(config, verify_hashes=False)


def test_v2_exact_rebuild_retains_invalid_result_hashes_and_all_gates() -> None:
    receipt = gw.check()
    stored = json.loads((ROOT / gw.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt == stored
    assert receipt["status"] == (
        "FROZEN_SYNTHETIC_ONLY_TEST_CORRECTION_COMPLETE_AWAITING_DISTINCT_AUDIT"
    )
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True
    rebuild = receipt["exact_rebuild"]
    assert rebuild["scenario_count"] == 680
    assert rebuild["attempted_matrix_cell_count"] == 16_320
    assert rebuild["scored_matrix_cell_count"] == 11_560
    assert rebuild["numerical_invalid_cell_count"] == 680
    assert rebuild["source_blocked_cell_count"] == 2_720
    assert rebuild["unadapted_cell_count"] == 1_360
    assert rebuild["replay_entry_count"] == 28_560
    assert rebuild["result_hash_count"] == 11_560 + 680
    assert rebuild["invariance_gates"]["pass"] is True
    assert all(value == 0 for value in receipt["access_accounting"].values())
