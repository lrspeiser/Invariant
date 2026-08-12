from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_fitted_output_connection_covariant_origin_audit import (
    CLAIM_SEALS,
    CONFIG_PATH,
    EXPECTED_DIRECT_EVIDENCE,
    EXPECTED_PREDECESSORS,
    FIRST_BLOCKER,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    _load_bound,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / CONFIG_PATH
ARTIFACT = ROOT / OUTPUT_PATH


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _reseal(value: dict[str, object]) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


@pytest.fixture(scope="module")
def gate() -> dict[str, object]:
    return build_gate(CONFIG)


def test_exact_gate_matches_checked_artifact_and_replays(gate: dict[str, object]) -> None:
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert gate == checked == build_gate(CONFIG)
    assert checked["content_sha256"] == hashlib.sha256(
        _canonical({key: item for key, item in checked.items() if key != "content_sha256"})
    ).hexdigest()


def test_closed_world_action_and_source_schemas_are_exact(gate: dict[str, object]) -> None:
    counts = gate["gate_counts"]
    assert counts["covariant_action_specializations_bound"] == 12
    assert counts["full_source_D1_jacobians_bound"] == 12
    assert counts["full_source_D1_entries_per_candidate"] == 1683
    assert counts["action_record_schema_keys"] == 13
    assert counts["source_record_schema_keys"] == 18
    assert counts["registered_output_connection_functors"] == 0
    assert counts["registered_corrected_second_source_jet_entries"] == 0
    assert counts["complete_component_Frechet_D2_to_D4_tensors"] == 0
    for row in gate["candidate_records"]:
        assert set(row["covariant_action_specialization"]) == {"G2", "G3", "G4", "G5"}
        assert row["full_component_Frechet_tensors_orders_2_to_4_complete"] is False


def test_all_fitted_coefficients_lack_registered_origin_binding(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["fitted_connection_coefficients_per_candidate"] == 22
    assert counts["fitted_connection_coefficients_audited"] == 264
    assert counts["fitted_coefficients_with_action_root_provenance"] == 0
    assert counts["covariant_action_derived_connections"] == 0
    for row in gate["candidate_records"]:
        assert row["fitted_connection_nonzero_coefficients"] == 22
        assert row["fitted_coefficients_with_action_root_provenance"] == 0
        assert row["registered_output_connection_functors"] == 0
        assert row["registered_corrected_second_source_jet_entries"] == 0
        assert row["origin_decision"] == (
            "not_identifiable_from_registered_action_and_D1_source_schemas"
        )


def test_origin_nonidentifiability_does_not_promote_or_reject(
    gate: dict[str, object],
) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106920
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "registered_covariant_action_specializations_bound",
        "registered_full_source_D1_jacobians_bound",
        "fitted_connection_origin_schema_audited",
        "fitted_connection_value_solution_retained",
    }
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("origin_binding", "result boundary"),
        ("origin_claim", "result boundary"),
        ("second_source", "result boundary"),
        ("connection_functor", "result boundary"),
        ("fit_hash", "result boundary"),
        ("admit_D2F", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("unknown_key", "result boundary"),
        ("forge_predecessor", "predecessor binding"),
        ("forge_action", "direct evidence binding"),
        ("forge_source", "direct evidence binding"),
        ("forge_local", "local binding"),
    ],
)
def test_resealed_semantic_and_provenance_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "origin_binding":
        row["fitted_coefficients_with_action_root_provenance"] = 1
    elif mutation == "origin_claim":
        value["claim_seals"]["connection_derived_from_covariant_action"] = True
    elif mutation == "second_source":
        row["registered_corrected_second_source_jet_entries"] = 1
    elif mutation == "connection_functor":
        row["registered_output_connection_functors"] = 1
    elif mutation == "fit_hash":
        row["fitted_connection_content_sha256"] = "0" * 64
    elif mutation == "admit_D2F":
        value["claim_seals"]["cross_slice_D2F_entries_admitted"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "unknown_key":
        value["promotion"] = True
    elif mutation == "forge_predecessor":
        value["source_bindings"]["candidate_pother_one_form_connection"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_action":
        value["source_bindings"]["direct_evidence"]["covariant_action"]["artifact"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_source":
        value["source_bindings"]["direct_evidence"]["full_source_jacobian"]["artifact"][
            "content_sha256"
        ] = "0" * 64
    else:
        value["source_bindings"]["test"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        _validate_result(_reseal(value), root=ROOT)


def test_config_paths_and_closed_bindings(gate: dict[str, object]) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="path escapes"):
        _load_bound(
            ROOT,
            {"path": "../outside.json", "file_sha256": "0" * 64, "content_sha256": "0" * 64},
        )
    assert gate["source_bindings"]["direct_evidence"] == EXPECTED_DIRECT_EVIDENCE
    for label, binding in EXPECTED_PREDECESSORS.items():
        assert gate["source_bindings"][label] == binding
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        assert gate["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
