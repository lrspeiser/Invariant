from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import pytest

import sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v3 as matrix
from sigma_theory_compiler.sigma_core import SchemaViolation


@pytest.fixture(scope="module")
def release_payloads():
    return matrix.derive_release()


def test_v3_binds_exact_blocked_audit_and_v2_counterevidence() -> None:
    config = matrix.load_config()
    matrix.validate_config(config)
    assert config["blocked_audit"]["raw_sha256"] == (
        "849aa96389af239148ce957256ad2285053ae3437d7a4210530a64cfb72abe23"
    )
    assert config["blocked_audit"]["content_sha256"] == (
        "a4217f84e6953a0807fe7f41c065b08ab7ba68a71c134146aa8cf4fa725ba3a6"
    )
    assert config["blocked_audit"]["decision"] == (
        "BLOCK_DIMENSIONAL_METADATA_HUBBLE_RATE_MISTYPED_AS_LENGTH"
    )


def test_explicit_feature_map_and_catalogue_rate_dimensions() -> None:
    config = matrix.load_config()
    catalogue = matrix._catalogue(matrix.base.load_config())
    assert set(matrix._FEATURE_METADATA) == set(matrix.base._FEATURES)
    for feature in matrix._RATE_FEATURES:
        unit, axes = matrix._FEATURE_METADATA[feature]
        element = catalogue.by_id()[feature]
        assert unit == element.canonical_unit == "km s^-1 Mpc^-1"
        assert axes == element.axes == ("object",)
        assert element.si_dimension == (0, 0, -1, 0, 0, 0, 0)

    mutated = deepcopy(config)
    mutated["repair"]["si_dimension"] = [0, 1, 0, 0, 0, 0, 0]
    with pytest.raises(SchemaViolation, match="repair contract"):
        matrix.validate_config(mutated, verify_hashes=False)


def test_all_1440_scenario_rate_references_are_repaired(release_payloads) -> None:
    _, _, scenarios_bytes, _, _, _, typed_diff_bytes = release_payloads
    rows = [json.loads(line) for line in scenarios_bytes.splitlines()]
    affected = [
        feature
        for row in rows
        for feature in row["scenario"]["formula_features"]
        if feature["element_id"] in matrix._RATE_FEATURES
    ]
    assert len(rows) == 720
    assert len(affected) == 1440
    assert all(feature["unit"] == "km s^-1 Mpc^-1" for feature in affected)
    assert all(feature["axes"] == ["object"] for feature in affected)
    typed_diff = json.loads(typed_diff_bytes)
    assert typed_diff["scenario_reference_occurrences"] == 1440
    assert typed_diff["unexpected_normalized_scenario_differences"] == 0


def test_numerical_matrix_and_geometry_are_byte_or_value_identical(release_payloads) -> None:
    receipt, values, _, _, confusion_bytes, diagnostics, typed_diff_bytes = release_payloads
    config = matrix.load_config()
    expected = config["expected_unchanged"]
    assert hashlib.sha256(values).hexdigest() == expected["values_npz_raw_sha256"]
    assert hashlib.sha256(diagnostics).hexdigest() == expected["geometry_diagnostics_raw_sha256"]
    assert receipt["scenario_count"] == expected["scenario_count"] == 720
    assert receipt["attempted_cell_count"] == expected["attempted_cell_count"] == 10800
    assert receipt["scored_cell_count"] == expected["scored_cell_count"] == 4320
    assert receipt["replay_entry_count"] == expected["replay_entry_count"] == 15120
    assert receipt["truth_recovery_count"] == expected["truth_recovery_count"] == 459
    assert (
        receipt["distinct_truth_recovery_count"] == expected["distinct_truth_recovery_count"] == 50
    )
    confusion = json.loads(confusion_bytes)
    predecessor = json.loads((matrix._ROOT / matrix.v2.CONFUSION_PATH).read_text(encoding="utf-8"))
    for key in (
        "truth_formula_ids",
        "candidate_formula_ids",
        "winner_membership_counts",
        "recovery_by_truth",
        "scenario_count",
        "attempted_cell_count",
        "scored_cell_count",
        "truth_recovery_count",
        "distinct_truth_recovery_count",
        "no_hand_ranking",
    ):
        assert confusion[key] == predecessor[key]
    typed_diff = json.loads(typed_diff_bytes)
    assert typed_diff["values_npz_byte_identical_to_v2"] is True
    assert typed_diff["geometry_diagnostics_byte_identical_to_v2"] is True
    assert typed_diff["numerical_confusion_fields_identical_to_v2"] is True


def test_v3_response_barrier_and_artifact_hashes(release_payloads) -> None:
    receipt, values, scenarios, ledger, confusion, diagnostics, typed_diff = release_payloads
    for key in (
        "cf4_measured_velocity_fields_decoded",
        "cf4_published_peculiar_velocity_fields_decoded",
        "validation_source_fields_decoded",
        "confirmation_source_fields_decoded",
        "pantheon_files_opened",
        "real_response_values_decoded",
        "real_scores",
    ):
        assert receipt["access_accounting"][key] == 0
    payloads = {
        "values.npz": values,
        "scenarios.jsonl": scenarios,
        "ledger.json": ledger,
        "confusion-matrix.json": confusion,
        "geometry-and-identifiability.json": diagnostics,
        "typed-contract-diff.json": typed_diff,
    }
    assert receipt["artifact_sha256"] == {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == matrix.base._json_sha256(body)
