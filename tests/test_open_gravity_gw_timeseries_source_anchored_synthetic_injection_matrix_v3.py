from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v3 as gw,
)
from sigma_theory_compiler.open_gravity_formula_execution_protocol_v1 import BindingStatus
from sigma_theory_compiler.open_gravity_observation_operators_v1 import SeedLineage
from sigma_theory_compiler.open_gravity_synthetic_replay_ledger_v1 import (
    DiscoveryStatus,
    ReplayEntry,
    SyntheticReplayLedger,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

ROOT = Path(__file__).resolve().parents[1]


def _complex(values: dict[str, np.ndarray]) -> np.ndarray:
    return (
        values["prediction.matrix.frequency-real"] + 1j * values["prediction.matrix.frequency-imag"]
    )


def test_config_binds_blocked_audit_predecessors_and_zero_response_access() -> None:
    config = gw.load_config()
    gw.validate_config(config)
    inventory = gw._source_inventory(config)
    assert inventory["blocked_audit"]["decision"] == "BLOCK"
    assert inventory["blocked_audit"]["blocking_finding"]["code"] == (
        "B01_FALSE_FFT_ROUNDTRIP_PASS_AND_INCONSISTENT_DUAL_DOMAIN_OBSERVATIONS"
    )
    assert inventory["v2_receipt"]["content_sha256"] == (
        "d6d517bd1dcc0aea5633dcf12042ae7b8298d79e1f694bee0c9737fbedaf5891"
    )
    assert config["representation_contract"]["canonical_domain"] == (
        "positive-frequency-rfft-bins-excluding-dc"
    )
    assert config["scoring"]["scored_outputs"] == list(gw._SCORING_OUTPUTS)
    assert config["scoring"]["derived_unscored_outputs"] == ["prediction.matrix.time-strain"]
    access = config["access_contract"]
    assert access["source_anchor_files_opened"] == 19
    assert access["infrastructure_files_hashed"] == 15
    assert access["target_free_nr_waveform_rows_opened"] == 2769
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
        (("representation_contract", "canonical_domain"), "time"),
        (("representation_contract", "nyquist_rule"), "ALLOW_COMPLEX"),
        (("scoring", "scored_outputs"), ["prediction.matrix.time-strain"]),
        (("access_contract", "strain_samples_opened"), 1),
        (("infrastructure_bindings",), []),
    ],
)
def test_material_v3_mutation_fails_closed(path: tuple[str, ...], value: object) -> None:
    config = copy.deepcopy(gw.load_config())
    cursor = config
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    with pytest.raises(SchemaViolation):
        gw.validate_config(config, verify_hashes=False)


def test_canonical_pair_enforces_dc_nyquist_and_rejects_mutations() -> None:
    rng = np.random.default_rng(44)
    proposed = rng.normal(size=(3, 128)) + 1j * rng.normal(size=(3, 128))
    time, canonical = gw._canonical_pair_from_positive_frequency(proposed, 256)
    errors = gw._fft_pair_errors(time, canonical)
    assert max(errors.values()) <= 1e-12
    np.testing.assert_array_equal(canonical[:, -1].imag, np.zeros(3))
    bad_nyquist = np.array(canonical, copy=True)
    bad_nyquist[0, -1] += 0.25j
    with pytest.raises(SchemaViolation):
        gw._fft_pair_errors(time, bad_nyquist)
    bad_dc_time = np.array(time, copy=True)
    bad_dc_time[0] += 0.5
    with pytest.raises(SchemaViolation):
        gw._fft_pair_errors(bad_dc_time, canonical)


def test_every_truth_candidate_prediction_is_forward_backward_consistent() -> None:
    config = gw.load_config()
    slots = gw._source_slots(config, gw._source_inventory(config))
    assert len(slots) == 40
    count = 0
    calibration_count = 0
    for slot in slots:
        if slot["noise_family"] == "published-psd-calibration-envelope":
            calibration_count += len(config["truth_mechanisms"])
        for formula_id in config["truth_mechanisms"]:
            prediction = gw._adapter_callable(formula_id)(
                slot["values"], gw._truth_parameters(formula_id, config)
            )
            errors = gw._fft_pair_errors(
                prediction["prediction.matrix.time-strain"], _complex(prediction)
            )
            assert max(errors.values()) <= 1e-12
            assert np.max(np.abs(_complex(prediction)[:, -1].imag)) == 0.0
            count += 1
    assert count == 680
    assert calibration_count == 136


def test_noise_is_drawn_once_and_response_and_noise_views_are_consistent() -> None:
    config = gw.load_config()
    slots = gw._source_slots(config, gw._source_inventory(config))
    family_index = {value: index for index, value in enumerate(config["noise_families"])}
    counts = {"zero": 0, "nonzero": 0}
    for slot in slots:
        for truth_index, formula_id in enumerate(config["truth_mechanisms"]):
            scenario_id = f"gw.{slot['slot_id']}.{formula_id.lower()}.v3"
            lineage = SeedLineage(
                config["suite_seed"],
                scenario_id,
                slot["slot_id"],
                f"truth.{formula_id.lower()}",
                family_index[slot["noise_family"]],
                0,
            )
            truth = gw._adapter_callable(formula_id)(
                slot["values"], gw._truth_parameters(formula_id, config)
            )
            response, _variance, metadata = gw._noise_response(truth, slot, lineage)
            response_errors = gw._fft_pair_errors(
                response["prediction.matrix.time-strain"], _complex(response)
            )
            noise_errors = gw._fft_pair_errors(
                response["prediction.matrix.time-strain"] - truth["prediction.matrix.time-strain"],
                _complex(response) - _complex(truth),
            )
            assert max((*response_errors.values(), *noise_errors.values())) <= 1e-12
            if slot["noise_family"] == "zero-noise":
                assert metadata["canonical_noise_draw_count"] == 0
                for output_id in gw._OUTPUTS:
                    np.testing.assert_array_equal(response[output_id], truth[output_id])
                counts["zero"] += 1
            else:
                assert metadata["canonical_noise_draw_count"] == 1
                counts["nonzero"] += 1
    assert counts == {"zero": 136, "nonzero": 544}


def test_scoring_uses_only_nondoublecounted_canonical_observables() -> None:
    comparisons = gw._comparisons()
    assert tuple(row.prediction_element_id for row in comparisons) == gw._SCORING_OUTPUTS
    assert all("time-strain" not in row.prediction_element_id for row in comparisons)
    config = gw.load_config()
    bindings = gw._bindings(config)
    assert sum(row.status is BindingStatus.EXECUTABLE for row in bindings) == 17
    assert sum(row.status is BindingStatus.SOURCE_BLOCKED for row in bindings) == 4
    assert sum(row.status is BindingStatus.UNADAPTED for row in bindings) == 2


def test_frozen_v3_receipt_counts_semantics_and_invariants() -> None:
    receipt = json.loads((ROOT / gw.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["status"] == "FROZEN_SYNTHETIC_ONLY_COMPLETE_AWAITING_DISTINCT_AUDIT"
    assert receipt["scientific_claim"] == "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION"
    assert receipt["independent_audit_completed"] is False
    assert receipt["distinct_independent_audit_required"] is True
    assert receipt["blocked_predecessor_audit"]["raw_sha256"] == (
        "2f854e342f0480378931684b7f7e3aa94fa0870b03e457ccefed043202f87108"
    )
    assert receipt["scored_output_ids"] == list(gw._SCORING_OUTPUTS)
    assert receipt["derived_unscored_output_ids"] == ["prediction.matrix.time-strain"]
    assert receipt["scenario_count"] == 680
    assert receipt["attempted_matrix_cell_count"] == 16_320
    assert receipt["scored_matrix_cell_count"] == 11_560
    assert receipt["numerical_invalid_cell_count"] == 680
    assert receipt["source_blocked_cell_count"] == 2_720
    assert receipt["unadapted_cell_count"] == 1_360
    assert receipt["replay_entry_count"] == 28_560
    assert receipt["invariance_gates"]["pass"] is True
    fft = receipt["fft_consistency"]
    assert fft["prediction_truth_and_candidate_check_count"] == 680
    assert fft["response_check_count"] == 680
    assert fft["noise_realization_check_count"] == 680
    assert fft["zero_noise_check_count"] == 136
    assert fft["nonzero_noise_check_count"] == 544
    assert fft["all_forward_backward_dc_nyquist_pass"] is True
    assert (
        max(
            value
            for key in ("prediction_maxima", "response_maxima", "noise_realization_maxima")
            for value in fft[key].values()
        )
        <= 1e-12
    )
    assert receipt["access_accounting"]["strain_samples_opened"] == 0
    assert receipt["access_accounting"]["real_likelihood_responses_opened"] == 0


def test_stored_truth_response_and_noise_pairs_are_independently_consistent() -> None:
    rows = [
        json.loads(line)
        for line in (ROOT / gw.SCENARIOS_PATH).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 680
    with np.load(ROOT / gw.VALUES_PATH, allow_pickle=False) as arrays:
        for row in rows:
            truth_values = {}
            response_values = {}
            for output_id in gw._OUTPUTS:
                truth_values[output_id] = arrays[
                    row["value_locators"]["truth_prediction"][output_id]["key"]
                ]
                response_values[output_id] = arrays[
                    row["value_locators"]["responses"][output_id]["key"]
                ]
            gw._fft_pair_errors(
                truth_values["prediction.matrix.time-strain"], _complex(truth_values)
            )
            gw._fft_pair_errors(
                response_values["prediction.matrix.time-strain"],
                _complex(response_values),
            )
            gw._fft_pair_errors(
                response_values["prediction.matrix.time-strain"]
                - truth_values["prediction.matrix.time-strain"],
                _complex(response_values) - _complex(truth_values),
            )
            if row["noise"]["family"] == "zero-noise":
                for output_id in gw._OUTPUTS:
                    np.testing.assert_array_equal(
                        response_values[output_id], truth_values[output_id]
                    )


def test_replay_chain_retains_every_scored_invalid_and_blocked_cell() -> None:
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
    assert len(ledger.entries) == 28_560
    assert sum(entry.result_sha256 is not None for entry in entries) == 12_240
    assert sum(entry.status.value == "NUMERICAL_INVALID" for entry in entries) == 680
    assert sum(entry.status.value == "SOURCE_BLOCKED" for entry in entries) == 2_720
    assert sum(entry.status.value == "UNADAPTED" for entry in entries) == 1_360


def test_frozen_v3_replay_is_byte_identical() -> None:
    receipt = gw.check()
    stored = json.loads((ROOT / gw.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["content_sha256"] == stored["content_sha256"]
