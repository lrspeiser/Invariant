from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v1 as gw,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    ReplayEntry,
    SyntheticReplayLedger,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[1]


def test_config_and_source_boundary_are_exact_and_response_blind() -> None:
    config = gw.load_config()
    gw.validate_config(config)
    inventory = gw._source_inventory(config)
    assert inventory["waveform"].shape == (2769, 2)
    assert inventory["lane2_prediction"]["method_passed"] is False
    assert inventory["lane2_prediction"]["real_response_authorized"] is False
    assert inventory["lane2_gates"]["strain_values_read"] == 0
    assert inventory["runner_audit"]["status"] == "PASS"
    access = config["access_contract"]
    assert access["source_anchor_files_opened"] == 19
    assert access["source_anchor_bytes_opened"] == 1_022_661
    assert access["infrastructure_files_hashed"] == 10
    assert access["infrastructure_bytes_hashed"] == 109_039
    assert all(
        access[key] == 0
        for key in (
            "strain_files_opened",
            "strain_samples_opened",
            "real_likelihood_responses_opened",
            "real_likelihood_values_computed",
            "psd_payload_arrays_opened",
            "calibration_payload_archives_opened",
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
        (("sample_grid", "samples"), 128),
        (("noise", "base_fraction_of_reference_signal"), 0.5),
        (("scoring", "distinct_gap"), 0.0),
        (("truth_mechanisms",), []),
        (("adapter_blocks",), []),
        (("access_contract", "strain_samples_opened"), 1),
    ],
)
def test_material_config_mutation_fails_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(gw.load_config())
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(SchemaViolation):
        gw.validate_config(config, verify_hashes=False)


def test_source_population_has_exact_network_geometry_axes_and_nuisances() -> None:
    config = gw.load_config()
    slots = gw._source_slots(config, gw._source_inventory(config))
    assert len(slots) == 2 * 2 * 2 * 5
    assert len({row["slot_id"] for row in slots}) == 40
    assert {row["event_id"] for row in slots} == {"GW150914", "GW170817"}
    for slot in slots:
        values = slot["values"]
        assert set(values) == set(gw._FEATURES)
        assert values["source.matrix.base-time-strain"].shape == (3, 256)
        assert values["source.matrix.base-frequency-real"].shape == (3, 128)
        assert values["source.matrix.psd-sigma"].shape == (3, 128)
        assert values["source.vector.frequency-hz"].shape == (128,)
        assert values["source.vector.time-seconds"].shape == (256,)
        assert np.all(np.isfinite(values["source.matrix.base-time-strain"]))
        if slot["event_id"] == "GW150914":
            assert values["source.vector.active-detector-mask"].tolist() == [1, 1, 0]
            np.testing.assert_array_equal(
                values["source.matrix.base-time-strain"][2], np.zeros(256)
            )


def test_common_abi_executable_and_blocked_inventory_is_honest() -> None:
    config = gw.load_config()
    bindings = gw._bindings(config)
    executable = [row for row in bindings if row.status is BindingStatus.EXECUTABLE]
    blocked = [row for row in bindings if row.status is not BindingStatus.EXECUTABLE]
    assert [row.formula_id for row in executable] == config["truth_mechanisms"]
    assert len(executable) == 17
    assert len(blocked) == 6
    assert sum(row.status is BindingStatus.SOURCE_BLOCKED for row in blocked) == 4
    assert sum(row.status is BindingStatus.UNADAPTED for row in blocked) == 2
    assert all(row.domains == ("gw-network",) for row in executable)
    assert all(row.required_features == gw._FEATURES for row in executable)


def test_mechanism_specific_generators_and_exact_aliases_are_retained() -> None:
    config = gw.load_config()
    slot = gw._source_slots(config, gw._source_inventory(config))[0]
    values = slot["values"]
    predictions = {
        formula_id: gw._adapter_callable(formula_id)(
            values, gw._truth_parameters(formula_id, config)
        )
        for formula_id in config["truth_mechanisms"]
    }
    for prediction in predictions.values():
        assert set(prediction) == set(gw._OUTPUTS)
        assert all(np.all(np.isfinite(value)) for value in prediction.values())
    for left, right in (
        ("CONTROL_FREE_DELAY", "LANE5_DELAY_MEMORY"),
        ("CONTROL_OU_NOISE", "LANE5_STOCHASTIC_OU_MEMORY"),
        ("CONTROL_SINGLE_LTI", "LANE5_EXPONENTIAL_MEMORY"),
        ("CONTROL_TWO_POLE_LTI", "LANE5_BIEXPONENTIAL_MEMORY"),
    ):
        for output_id in gw._OUTPUTS:
            np.testing.assert_array_equal(
                predictions[left][output_id], predictions[right][output_id]
            )
    assert not np.array_equal(
        predictions["LANE2_DYNAMIC_PHASE"]["prediction.matrix.time-strain"],
        predictions["GR_NETWORK_CONTROL"]["prediction.matrix.time-strain"],
    )
    assert not np.array_equal(
        predictions["CONTROL_SOURCE_RINGDOWN"]["prediction.matrix.time-strain"],
        predictions["GR_NETWORK_CONTROL"]["prediction.matrix.time-strain"],
    )


def test_frozen_receipt_counts_claim_failures_and_access_ceiling() -> None:
    receipt = json.loads((ROOT / gw.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT"
    assert receipt["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL"
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True
    assert receipt["event_count"] == 2
    assert receipt["source_population_slot_count"] == 40
    assert receipt["truth_mechanism_count"] == 17
    assert receipt["noise_family_count"] == 5
    assert receipt["scenario_count"] == 680
    assert receipt["executable_binding_count"] == 17
    assert receipt["blocked_binding_count"] == 6
    assert receipt["attempted_matrix_cell_count"] == 16_320
    assert receipt["scored_matrix_cell_count"] == 11_560
    assert receipt["numerical_invalid_cell_count"] == 680
    assert receipt["source_blocked_cell_count"] == 2_720
    assert receipt["unadapted_cell_count"] == 1_360
    assert receipt["replay_entry_count"] == 28_560
    assert receipt["invariance_gates"]["pass"] is True
    assert receipt["retained_upstream_optimizer_failures"] == {
        "method_passed": False,
        "optimizer_recovery_status": "FAIL_OPTIMIZER_RECOVERY_GATE",
        "registered_branch_power_status": "FAIL_BRANCH_OPTIMIZER_GATE",
        "reservoir_power_status": "FAIL_RESERVOIR_POWER_OPTIMIZER_GATE",
    }
    assert receipt["access_accounting"]["strain_samples_opened"] == 0
    assert receipt["access_accounting"]["real_likelihood_responses_opened"] == 0


def test_zero_noise_responses_are_directly_generated_from_hidden_truth() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / gw.SCENARIOS_PATH).read_text(encoding="utf-8").splitlines()
    ]
    with np.load(ROOT / gw.VALUES_PATH, allow_pickle=False) as arrays:
        zero_rows = [row for row in rows if row["noise"]["family"] == "zero-noise"]
        assert len(zero_rows) == 2 * 2 * 2 * 17
        for row in zero_rows:
            for output_id in gw._OUTPUTS:
                response_key = row["value_locators"]["responses"][output_id]["key"]
                prediction_key = row["value_locators"]["truth_prediction"][output_id]["key"]
                np.testing.assert_array_equal(arrays[response_key], arrays[prediction_key])


def test_typed_scenarios_confusion_and_replay_chain_are_complete() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / gw.SCENARIOS_PATH).read_text(encoding="utf-8").splitlines()
    ]
    confusion = json.loads((ROOT / gw.CONFUSION_PATH).read_text(encoding="utf-8"))
    payload = json.loads((ROOT / gw.LEDGER_PATH).read_text(encoding="utf-8"))
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
    assert len(rows) == 680
    assert len({row["scenario"]["scenario_id"] for row in rows}) == 680
    assert all(row["scenario"]["domain"] == "gw-network" for row in rows)
    assert all(len(row["scenario"]["hidden_truth"]) == 1 for row in rows)
    assert len(ledger.entries) == 28_560
    assert confusion["scenario_count"] == 680
    assert confusion["attempted_matrix_cell_count"] == 16_320
    assert confusion["scored_matrix_cell_count"] == 11_560
    assert confusion["numerical_invalid_cell_count"] == 680
    assert confusion["source_blocked_cell_count"] == 2_720
    assert confusion["unadapted_cell_count"] == 1_360
    assert confusion["no_hand_ranking"] is True
    assert all(entry.claim_class == "SYNTHETIC_DIRECTIONAL_SIGNAL" for entry in entries)
    assert sum(entry.result_sha256 is not None for entry in entries) == 11_560
    assert sum(entry.status.value == "NUMERICAL_INVALID" for entry in entries) == 680
    assert sum(entry.status.value == "SOURCE_BLOCKED" for entry in entries) == 2_720
    assert sum(entry.status.value == "UNADAPTED" for entry in entries) == 1_360


def test_frozen_replay_is_byte_identical() -> None:
    receipt = gw.check()
    stored = json.loads((ROOT / gw.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["content_sha256"] == stored["content_sha256"]
