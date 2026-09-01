from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.open_gravity_full3d_phangs_synthetic_adapter_preflight_v1 import (
    _MECHANISM_ENTRYPOINTS,
)
from sigma_theory_compiler.open_gravity_full3d_phangs_synthetic_injection_matrix_v2 import (
    LEDGER_PATH,
    RECEIPT_PATH,
    SCENARIOS_PATH,
    VALUES_PATH,
    derive_release,
    load_config,
    validate_config,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import array_sha256
from sigma_theory_compiler.sigma_core import SchemaViolation, canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release() -> tuple[dict, bytes, bytes, bytes]:
    return derive_release()


def _rows(payload: bytes) -> list[dict]:
    return [json.loads(line) for line in payload.decode("utf-8").splitlines()]


def _value_bank(values: bytes) -> dict[tuple[str, str], dict]:
    return {
        (row["object_id"], truth["mechanism_id"]): truth
        for row in _rows(values)
        for truth in row["truth_responses"]
    }


def test_matrix_has_27_real_scenarios_and_243_common_abi_cells(release) -> None:
    receipt, values, scenarios, ledger = release
    assert receipt["status"] == "SEALED_MECHANISM_SPECIFIC_MATRIX_AWAITING_INDEPENDENT_AUDIT"
    assert receipt["object_count"] == 3
    assert receipt["mechanism_count"] == 9
    assert receipt["scenario_count"] == 27
    assert receipt["truth_generation_evaluations"] == 27
    assert receipt["candidate_common_abi_executions"] == 243
    assert receipt["completed_replay_cells"] == 243
    assert receipt["replay_entry_count"] == 486
    assert receipt["confusion_matrix_cells"] == 243
    assert receipt["recovery_row_count"] == 27
    assert receipt["truth_recovered_count"] == 27
    assert receipt["distinct_scenario_hash_count"] == 27
    assert receipt["distinct_execution_cell_hash_count"] == 243
    assert hashlib.sha256(values).hexdigest() == receipt["values_jsonl_sha256"]
    assert hashlib.sha256(scenarios).hexdigest() == receipt["scenarios_jsonl_sha256"]
    assert hashlib.sha256(ledger).hexdigest() == receipt["ledger_json_sha256"]
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == canonical_sha256(body)


def test_every_truth_response_hidden_id_and_scenario_hash_is_mechanism_specific(
    release,
) -> None:
    _, values, scenarios, _ = release
    mechanisms = tuple(sorted(_MECHANISM_ENTRYPOINTS))
    bank = _value_bank(values)
    rows = _rows(scenarios)
    expected_pairs = {
        (object_id, mechanism_id)
        for object_id in ("NGC2903", "NGC3351", "NGC3627")
        for mechanism_id in mechanisms
    }
    assert {
        (row["object_id"], row["truth_world"]["mechanism_id"]) for row in rows
    } == expected_pairs
    assert len({row["scenario"]["scenario_id"] for row in rows}) == 27
    assert len({row["scenario_sha256"] for row in rows}) == 27

    value_rows = {row["object_id"]: row for row in _rows(values)}
    for row in rows:
        scenario = row["scenario"]
        truth = row["truth_world"]
        mechanism_id = truth["mechanism_id"]
        object_id = row["object_id"]
        injection_id = mechanisms.index(mechanism_id)
        assert row["scenario_sha256"] == canonical_sha256(scenario)
        assert scenario["scenario_id"].endswith(f"truth.{mechanism_id.lower()}.v2")
        assert scenario["seed_lineage"]["scenario_id"] == scenario["scenario_id"]
        assert scenario["seed_lineage"]["truth_world_id"] == truth["truth_world_id"]
        assert truth["truth_world_id"] == f"truth.{mechanism_id.lower()}"
        assert truth["injection_id"] == injection_id
        truth_value = np.asarray(truth["truth_value"], dtype=np.int64)
        assert np.array_equal(truth_value, np.asarray([injection_id], dtype=np.int64))
        assert array_sha256(truth_value) == truth["truth_value_sha256"]
        assert scenario["hidden_truth"][0]["value_sha256"] == truth["truth_value_sha256"]

        bank_truth = bank[(object_id, mechanism_id)]
        response = np.asarray(bank_truth["value"], dtype=np.float64)
        assert response.shape == (17, 17, 17, 3)
        assert array_sha256(response) == truth["response_value_sha256"]
        assert bank_truth["response_artifact"] == {
            **scenario["scoring_responses"][0],
            "artifact_path": bank_truth["response_artifact"]["artifact_path"],
        }
        assert scenario["scoring_responses"][0]["value_sha256"] == array_sha256(response)

        uncertainty = np.asarray(value_rows[object_id]["uncertainty_value"], dtype=np.float64)
        assert uncertainty.shape == response.shape
        assert array_sha256(uncertainty) == truth["uncertainty_value_sha256"]
        assert scenario["uncertainties"][0]["artifact_sha256"] == truth["uncertainty_value_sha256"]


def test_every_recovery_row_maps_to_distinct_truth_candidate_and_ledger_cells(
    release,
) -> None:
    _, values, scenarios, ledger_bytes = release
    mechanisms = tuple(sorted(_MECHANISM_ENTRYPOINTS))
    bank = _value_bank(values)
    scenario_rows = _rows(scenarios)
    ledger = json.loads(ledger_bytes)
    entries = ledger["entries"]
    assert len(entries) == 486
    completed = [entry for entry in entries if entry["scenario_id"] is not None]
    assert len(completed) == 243
    completed_by_sequence = {entry["sequence"]: entry for entry in completed}
    matrix_keys = set()
    execution_hashes = set()

    for row in scenario_rows:
        scenario = row["scenario"]
        truth = row["truth_world"]
        object_id = row["object_id"]
        response = np.asarray(bank[(object_id, truth["mechanism_id"])]["value"], dtype=np.float64)
        candidates = row["candidate_executions"]
        assert {candidate["candidate_formula_id"] for candidate in candidates} == set(mechanisms)
        distances = {}
        scenario_ledger_cells = 0
        for candidate in candidates:
            candidate_id = candidate["candidate_formula_id"]
            matrix_key = (scenario["scenario_id"], candidate_id)
            assert matrix_key not in matrix_keys
            matrix_keys.add(matrix_key)
            assert candidate["execution_cell_sha256"] not in execution_hashes
            execution_hashes.add(candidate["execution_cell_sha256"])

            bank_candidate = bank[(object_id, candidate_id)]
            prediction = np.asarray(bank_candidate["value"], dtype=np.float64)
            artifact = candidate["artifact"]
            assert artifact == bank_candidate["prediction_artifact"]
            assert artifact["dtype"] == "float64"
            assert artifact["shape"] == [17, 17, 17, 3]
            assert artifact["axes"] == ["x", "y", "z", "component"]
            assert artifact["unit"] == "m s^-2"
            assert artifact["frame"] == "solver-source"
            assert array_sha256(prediction) == candidate["value_sha256"]
            assert candidate["output_sha256"] == canonical_sha256(
                {"prediction.vector.acceleration": artifact}
            )
            scale = max(float(np.sqrt(np.mean(response * response))), 1.0e-30)
            expected_distance = float(np.sqrt(np.mean((prediction - response) ** 2)) / scale)
            assert math.isclose(
                candidate["metrics"]["relative_rmse_to_mechanism_generated_response"],
                expected_distance,
                abs_tol=0.0,
                rel_tol=0.0,
            )

            completed_entry = completed_by_sequence[candidate["completed_ledger_sequence"]]
            assert completed_entry["entry_sha256"] == candidate["completed_ledger_entry_sha256"]
            assert completed_entry["scenario_id"] == scenario["scenario_id"]
            assert completed_entry["object_id"] == scenario["object_id"]
            assert completed_entry["truth_world_id"] == truth["truth_world_id"]
            assert completed_entry["formula_id"] == candidate_id
            assert completed_entry["result_sha256"] == candidate["output_sha256"]
            assert (
                completed_entry["metrics_sha256"]
                == hashlib.sha256(
                    json.dumps(
                        candidate["metrics"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            )
            assert (
                completed_entry["diagnostics_sha256"]
                == hashlib.sha256(
                    json.dumps(
                        candidate["diagnostics"],
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
            )
            eligible = entries[completed_entry["sequence"] - 1]
            assert eligible["status"] == "ELIGIBLE_NOT_RUN"
            assert eligible["binding_sha256"] == completed_entry["binding_sha256"]
            assert completed_entry["prior_entry_sha256"] == eligible["entry_sha256"]
            scenario_ledger_cells += 1
            distances[candidate_id] = expected_distance
        assert scenario_ledger_cells == 9

        recovery = row["injection_recovery"]
        recorded = {
            item["candidate_formula_id"]: item["relative_rmse"]
            for item in recovery["candidate_distances"]
        }
        assert recorded == distances
        minimum = min(distances.values())
        expected_winners = sorted(
            candidate_id
            for candidate_id, distance in distances.items()
            if math.isclose(distance, minimum, abs_tol=1.0e-15, rel_tol=0.0)
        )
        assert recovery["winner_formula_ids"] == expected_winners
        assert recovery["truth_formula_id"] == truth["mechanism_id"]
        assert recovery["truth_recovered"] == (truth["mechanism_id"] in expected_winners)
        assert recovery["truth_recovered"]

    assert len(matrix_keys) == 243
    assert len(execution_hashes) == 243
    text = scenarios.decode("utf-8")
    assert "self_distance" not in text
    assert "self_recovered" not in text


def test_append_only_provenance_claim_ceiling_and_stored_artifacts(release) -> None:
    receipt, values, scenarios, ledger = release
    assert receipt["predecessor_source_gate_failures_retained"] == 1
    assert receipt["raw_fits_inventory"]["count"] == 21
    assert len(receipt["transitive_source_bindings"]) == 18
    access = receipt["access_accounting"]
    assert access["scientific_response_files_opened"] == 0
    assert access["scientific_response_rows_opened"] == 0
    assert access["lensing_response_files_opened"] == 0
    assert access["real_scores_computed"] == 0
    assert not access["response_calibrated"]
    assert access["source_fits_files_opened_by_successor"] == 0

    config = load_config()
    forged = {**config, "claim_class": "EMPIRICAL_SUPPORT"}
    with pytest.raises(SchemaViolation, match="claim ceiling"):
        validate_config(forged, verify_predecessor=False)
    changed_predecessor = {
        **config,
        "predecessor": {
            **config["predecessor"],
            "receipt": {
                **config["predecessor"]["receipt"],
                "sha256": "0" * 64,
            },
        },
    }
    with pytest.raises(SchemaViolation, match="predecessor receipt changed"):
        validate_config(changed_predecessor)

    assert (ROOT / VALUES_PATH).read_bytes() == values
    assert (ROOT / SCENARIOS_PATH).read_bytes() == scenarios
    assert (ROOT / LEDGER_PATH).read_bytes() == ledger
    assert json.loads((ROOT / RECEIPT_PATH).read_text()) == receipt
