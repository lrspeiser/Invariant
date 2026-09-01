from __future__ import annotations

import hashlib
import json

import pytest

import sigma_theory_compiler.open_gravity_void_cosmology_source_shaped_synthetic_injection_matrix_v2 as matrix


@pytest.fixture(scope="module")
def release_payloads():
    return matrix.derive_release()


def test_v2_binds_and_retains_v1_all_zero_failure() -> None:
    config = matrix.load_config()
    matrix.validate_config(config)
    predecessor = json.loads(
        matrix.base._repo_path(config["predecessor"]["receipt_path"]).read_text(encoding="utf-8")
    )
    assert predecessor["content_sha256"] == config["predecessor"]["receipt_content_sha256"]
    assert predecessor["scenario_count"] == 720
    assert predecessor["distinct_truth_recovery_count"] == 0
    assert config["repair"]["v1_zero_exposure_object_count"] == 8


def test_v2_selection_is_source_only_nonzero_and_exact(release_payloads) -> None:
    receipt, _, _, _, _, diagnostics_bytes = release_payloads
    diagnostics = json.loads(diagnostics_bytes)
    assert [int(row["identifier"]) for row in diagnostics["selected_cf4"]] == matrix.load_config()[
        "repair"
    ]["expected_identifiers"]
    assert all(
        float.fromhex(row["planck_void_length_mpc_hex"]) > 0.0
        or float.fromhex(row["wmap_void_length_mpc_hex"]) > 0.0
        for row in diagnostics["selected_cf4"]
    )
    assert diagnostics["source_selection_response_blind"] is True
    assert diagnostics["absolute_length_permutation_used"] is False
    assert receipt["geometry_gates"]["nonzero_source_coverage_per_selected_object"] is True


def test_v2_matrix_counts_blocks_and_access_boundary(release_payloads) -> None:
    receipt, _, scenarios, ledger_bytes, _, _ = release_payloads
    assert receipt["scenario_count"] == 720
    assert receipt["attempted_cell_count"] == 10800
    assert receipt["scored_cell_count"] == 4320
    assert receipt["replay_entry_count"] == 15120
    assert len(scenarios.splitlines()) == 720
    ledger = json.loads(ledger_bytes)
    blocked = [row for row in ledger["entries"] if row["status"] == "SOURCE_BLOCKED"]
    assert len(blocked) == 720 * 9
    assert {row["formula_id"] for row in blocked} == matrix.base._BLOCKED
    access = receipt["access_accounting"]
    for key in (
        "cf4_measured_velocity_fields_decoded",
        "cf4_published_peculiar_velocity_fields_decoded",
        "validation_source_fields_decoded",
        "confirmation_source_fields_decoded",
        "pantheon_files_opened",
        "real_response_values_decoded",
        "real_scores",
    ):
        assert access[key] == 0


def test_v2_has_nontrivial_source_signature_identifiability(release_payloads) -> None:
    receipt, _, _, _, confusion_bytes, diagnostics_bytes = release_payloads
    confusion = json.loads(confusion_bytes)
    diagnostics = json.loads(diagnostics_bytes)
    signature = diagnostics["signature_identifiability"]
    assert signature["pair_count"] == 15
    assert signature["exact_degenerate_pair_count"] < 15
    assert receipt["truth_recovery_count"] > 0
    assert confusion["no_hand_ranking"] is True


def test_v2_artifacts_and_receipt_hash_exactly(release_payloads) -> None:
    receipt, values, scenarios, ledger, confusion, diagnostics = release_payloads
    payloads = {
        "values.npz": values,
        "scenarios.jsonl": scenarios,
        "ledger.json": ledger,
        "confusion-matrix.json": confusion,
        "geometry-and-identifiability.json": diagnostics,
    }
    assert receipt["artifact_sha256"] == {
        name: hashlib.sha256(payload).hexdigest() for name, payload in payloads.items()
    }
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    assert receipt["content_sha256"] == matrix.base._json_sha256(body)
