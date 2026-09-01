from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_lens_ray_clock_source_shaped_synthetic_injection_matrix_v1 as lens_matrix,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    ReplayEntry,
    SyntheticReplayLedger,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_only_sealed_lane1_lane7_source_anchors() -> None:
    config = lens_matrix.load_config()
    lens_matrix.validate_config(config)
    assert len(config["lenses"]) == 8
    assert {row["id"].split("_")[0] for row in config["source_anchors"]} == {
        "LANE1",
        "LANE7",
    }
    assert sum(row["bytes"] for row in config["source_anchors"]) == 113_366
    access = config["access_contract"]
    assert access["sealed_source_anchor_files_opened"] == 6
    assert access["sealed_source_anchor_bytes_opened"] == 113_366
    assert all(
        access[key] == 0
        for key in (
            "eso_pixels_decoded",
            "eso_spectral_rows_decoded",
            "slacs_confirmation_files_opened",
            "lens_response_tables_opened",
            "lens_response_rows_opened",
            "theory_or_nuisance_tuning_events",
            "network_calls",
            "model_calls",
            "paid_calls",
        )
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("suite_seed",), 1),
        (("law_constants", "path_alpha"), 0.5),
        (("noise", "fractional_sigma"), 0.5),
        (("scoring", "minimum_whitened_gap_for_distinct_signature"), 0.0),
        (("source_anchors",), []),
        (("adapter_blocks",), []),
        (("access_contract", "lens_response_rows_opened"), 1),
    ],
)
def test_material_config_mutation_fails_closed(
    path: tuple[str, ...], value: object
) -> None:
    config = copy.deepcopy(lens_matrix.load_config())
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(SchemaViolation):
        lens_matrix.validate_config(config, verify_hashes=False)


def test_response_blind_source_lift_has_exact_geometry_and_delay_metadata() -> None:
    items = lens_matrix._source_items(lens_matrix.load_config())
    assert len(items) == 8
    assert sum(item["metadata"]["published_delay_available"] for item in items) == 7
    missing = [
        item["lens"]
        for item in items
        if not item["metadata"]["published_delay_available"]
    ]
    assert missing == ["SDSS J1320+1644"]
    for item in items:
        values = item["values"]
        assert set(values) == set(lens_matrix._FEATURES)
        assert values["source.vector.impact-parameter-mpc"].shape == (2,)
        assert values["source.vector.path-exposure"].shape == (2,)
        assert values["source.vector.endpoint-log-redshift"].shape == (2,)
        assert np.all(values["source.vector.impact-parameter-mpc"] > 0.0)
        assert np.all(values["source.scalar.einstein-mass-kg"] > 0.0)
        assert item["metadata"]["maximum_lens_equation_residual"] < 1e-12


def test_common_abi_is_honest_and_incompatible_formulas_are_explicitly_blocked() -> None:
    config = lens_matrix.load_config()
    bindings = lens_matrix._bindings(config)
    executable = [row for row in bindings if row.status is BindingStatus.EXECUTABLE]
    blocked = [row for row in bindings if row.status is not BindingStatus.EXECUTABLE]
    assert [row.formula_id for row in executable] == config["mechanisms"]
    assert len(executable) == 5
    assert len(blocked) == 7
    assert {row.status for row in blocked} == {
        BindingStatus.SOURCE_BLOCKED,
        BindingStatus.UNADAPTED,
    }
    assert all(row.domains == ("strong-lens",) for row in executable)
    assert all(
        "source lift" in row.approximation_ceiling for row in executable
    )


def test_mechanism_outputs_share_state_and_obey_exact_limits() -> None:
    config = lens_matrix.load_config()
    item = lens_matrix._source_items(config)[0]
    values = item["values"]
    gr = lens_matrix._prediction(values, "gr")
    path = lens_matrix._prediction(values, "path")
    endpoint = lens_matrix._prediction(values, "endpoint")
    geometric = lens_matrix._prediction(values, "geometric")
    slip = lens_matrix._prediction(values, "slip")
    np.testing.assert_array_equal(
        gr["prediction.vector.light-potential"],
        2.0 * gr["prediction.vector.matter-potential"],
    )
    np.testing.assert_array_equal(
        slip["prediction.vector.matter-potential"],
        gr["prediction.vector.matter-potential"],
    )
    assert not np.array_equal(
        slip["prediction.vector.light-potential"],
        gr["prediction.vector.light-potential"],
    )
    assert path["prediction.scalar.differential-log-redshift"][0] != 0.0
    assert endpoint["prediction.vector.endpoint-log-redshift"][0] != gr[
        "prediction.vector.endpoint-log-redshift"
    ][0]
    assert not np.array_equal(
        geometric["prediction.scalar.time-delay-days"],
        gr["prediction.scalar.time-delay-days"],
    )
    alpha_zero = dict(config["law_constants"])
    alpha_zero["path_alpha"] = 0.0
    gamma_one = dict(config["law_constants"])
    gamma_one["same_state_slip_gamma"] = 1.0
    for key in lens_matrix._OUTPUTS:
        np.testing.assert_array_equal(
            lens_matrix._prediction(values, "path", alpha_zero)[key], gr[key]
        )
        np.testing.assert_array_equal(
            lens_matrix._prediction(values, "slip", gamma_one)[key], gr[key]
        )


def test_frozen_release_counts_claim_and_invariant_boundary() -> None:
    receipt = json.loads((ROOT / lens_matrix.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT"
    assert receipt["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True
    assert receipt["lens_count"] == 8
    assert receipt["mechanism_count"] == 5
    assert receipt["noise_family_count"] == 3
    assert receipt["scenario_count"] == 8 * 5 * 3
    assert receipt["common_abi_execution_count"] == 8 * 5
    assert receipt["candidate_comparison_count"] == 8 * 5 * 3 * 5
    assert receipt["blocked_ledger_entry_count"] == 8 * 5 * 3 * 7
    assert receipt["replay_entry_count"] == 8 * 5 * 3 * (7 + 2 * 5)
    assert receipt["invariance_gates"]["pass"] is True
    assert all(
        value <= 1e-12
        for key, value in receipt["invariance_gates"].items()
        if key.startswith("maximum_")
    )
    assert receipt["access_accounting"]["lens_response_rows_opened"] == 0


def test_zero_noise_responses_are_directly_generated_from_each_hidden_truth() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / lens_matrix.SCENARIOS_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    with np.load(ROOT / lens_matrix.VALUES_PATH, allow_pickle=False) as arrays:
        zero_rows = [row for row in rows if row["noise"]["family"] == "zero-noise"]
        assert len(zero_rows) == 8 * 5
        for row in zero_rows:
            truth_id = row["truth_formula_id"]
            comparison = next(
                candidate
                for candidate in row["candidate_comparisons"]
                if candidate["candidate_formula_id"] == truth_id
            )
            for output_id in lens_matrix._OUTPUTS:
                response_locator = row["value_locators"]["responses"][output_id]
                candidate_key = comparison["value_keys"][output_id]
                np.testing.assert_array_equal(
                    arrays[response_locator["key"]], arrays[candidate_key]
                )


def test_typed_scenarios_confusion_and_replay_chain_are_complete() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / lens_matrix.SCENARIOS_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    confusion = json.loads((ROOT / lens_matrix.CONFUSION_PATH).read_text(encoding="utf-8"))
    diagnostics = json.loads((ROOT / lens_matrix.DIAGNOSTICS_PATH).read_text(encoding="utf-8"))
    payload = json.loads((ROOT / lens_matrix.LEDGER_PATH).read_text(encoding="utf-8"))
    entries = tuple(
        ReplayEntry(
            **{
                **entry,
                "status": DiscoveryStatus(entry["status"]),
                "reason_codes": tuple(entry["reason_codes"]),
                "observable_ids": tuple(entry["observable_ids"]),
            }
        )
        for entry in payload["entries"]
    )
    ledger = SyntheticReplayLedger(payload["ledger_id"], entries, payload["schema_version"])
    assert len(rows) == 120
    assert len(ledger.entries) == 2040
    assert confusion["scenario_count"] == 120
    assert confusion["candidate_comparison_count"] == 600
    assert confusion["numerical_failure_count"] == 0
    assert confusion["no_hand_ranking"] is True
    assert diagnostics["pair_count"] == 8 * 10
    assert all(row["scenario"]["domain"] == "strong-lens" for row in rows)
    assert all(len(row["scenario"]["hidden_truth"]) == 1 for row in rows)
    assert all(len(row["candidate_comparisons"]) == 5 for row in rows)
    assert all(entry.claim_class == "SYNTHETIC_DIRECTIONAL_SIGNAL" for entry in entries)
    assert sum(entry.result_sha256 is not None for entry in entries) == 600
    assert sum(entry.status.value == "SOURCE_BLOCKED" for entry in entries) == 720
    assert sum(entry.status.value == "UNADAPTED" for entry in entries) == 120


def test_frozen_replay_is_byte_identical() -> None:
    receipt = lens_matrix.check()
    stored = json.loads((ROOT / lens_matrix.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["content_sha256"] == stored["content_sha256"]

