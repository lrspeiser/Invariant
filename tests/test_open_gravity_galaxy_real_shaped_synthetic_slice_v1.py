from __future__ import annotations

import copy
import hashlib
import json

import numpy as np
import pytest

from sigma_theory_compiler.open_gravity_galaxy_real_shaped_synthetic_slice_v1 import (
    build,
    check,
    derive_release,
    load_config,
    validate_config,
)
from sigma_theory_compiler.open_gravity_synthetic_scenario_packet_v1 import array_sha256
from sigma_theory_compiler.sigma_core import SchemaViolation


def test_real_source_shaped_known_answer_matrix_is_complete() -> None:
    receipt, population = derive_release()
    assert receipt["population_rows"] == 3 * 3 * 3 == 27
    assert receipt["formula_executions"] == 27 * 9 == 243
    assert receipt["replay_entry_count"] == 27 * 2 == 54
    assert receipt["population_jsonl_sha256"] == hashlib.sha256(population).hexdigest()
    assert receipt["scientific_response_rows_opened"] == 0
    assert receipt["real_scores_computed"] == 0
    assert not receipt["response_calibrated"]
    assert all(
        row["recovered"] == row["total"] == 9
        for row in receipt["recovery_by_noise_sigma_m_s"].values()
    )


def test_population_contains_typed_values_and_calculated_candidate_scores() -> None:
    receipt, population = derive_release()
    rows = [json.loads(line) for line in population.decode("utf-8").splitlines()]
    assert len(rows) == receipt["population_rows"]
    first = rows[0]
    assert first["scenario"]["expected_predictions"][0]["shape"] == [60, 3]
    assert first["scenario"]["formula_features"][0]["axes"] == ["radial_bin"]
    assert len(first["values"]["source.vector.acceleration"]) == 60
    assert len(first["diagnostics"]["candidate_scores"]) == 9
    assert first["metrics"]["best_parameter_cell_id"] == "scale.1p00"
    assert first["diagnostics"]["self_injection_recovered"]


def test_all_serialized_values_and_candidate_predictions_match_their_hashes() -> None:
    _, population = derive_release()
    for line in population.decode("utf-8").splitlines():
        row = json.loads(line)
        scenario = row["scenario"]
        references = {
            reference["element_id"]: reference
            for partition in ("formula_features", "scoring_responses", "hidden_truth")
            for reference in scenario[partition]
        }
        for element_id, reference in references.items():
            value = np.asarray(row["values"][element_id], dtype=reference["dtype"])
            assert list(value.shape) == reference["shape"]
            assert array_sha256(value) == reference["value_sha256"]
        uncertainty = scenario["uncertainties"][0]
        variance = np.asarray(
            row["values"]["uncertainty.circular-speed-variance"], dtype=np.float64
        )
        assert array_sha256(variance) == uncertainty["artifact_sha256"]
        for candidate in row["diagnostics"]["candidate_scores"]:
            prediction = candidate["prediction"]
            value = np.asarray(prediction["value"], dtype=prediction["artifact"]["dtype"])
            assert list(value.shape) == prediction["artifact"]["shape"]
            assert array_sha256(value) == prediction["artifact"]["value_sha256"]


def test_source_hash_and_claim_ceiling_fail_closed() -> None:
    config = load_config()
    forged_source = copy.deepcopy(config)
    forged_source["source_profile"]["sha256"] = "0" * 64
    with pytest.raises(SchemaViolation, match="source_profile bytes changed"):
        validate_config(forged_source)

    forged_claim = copy.deepcopy(config)
    forged_claim["claim_class"] = "EMPIRICAL_SUPPORT"
    with pytest.raises(SchemaViolation, match="claim ceiling"):
        validate_config(forged_claim, verify_sources=False)


def test_canonical_build_and_replay_are_identical() -> None:
    assert build() in {"CREATED:CREATED", "EXISTING_IDENTICAL:EXISTING_IDENTICAL"}
    assert check() == "VALID"
    assert build() == "EXISTING_IDENTICAL:EXISTING_IDENTICAL"
