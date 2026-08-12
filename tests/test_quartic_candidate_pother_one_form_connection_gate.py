from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_candidate_pother_one_form_connection_gate import (
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


def test_candidate_bound_source_projection_is_exact(gate: dict[str, object]) -> None:
    counts = gate["gate_counts"]
    assert counts["full_source_Jacobians_bound"] == 12
    assert counts["Pother_one_form_entries_per_candidate"] == 990
    assert counts["Pother_one_form_nonzero_entries_per_candidate"] == 93
    assert counts["Pother_one_form_rank_per_candidate"] == 10
    for row in gate["candidate_records"]:
        one_form = row["candidate_bound_Pother_one_form"]
        assert one_form["shape"] == [11, 90]
        assert one_form["rank"] == 10
        assert one_form["entry_count"] == 990
        assert one_form["nonzero_entry_count"] == len(one_form["nonzero_entries"]) == 93
        assert one_form["source"] == (
            "registered_full_11x153_solved_source_Jacobian_principal_slice"
        )


def test_two_sided_system_and_witness_are_exact(gate: dict[str, object]) -> None:
    for row in gate["candidate_records"]:
        system = row["two_sided_reference_system"]
        assert system == {
            "equations": 8910,
            "unknowns": 11979,
            "coefficient_rank": 1870,
            "augmented_rank": 1870,
            "affine_solution_dimension": 10109,
            "consistent": True,
        }
        witness = row["free_variable_zero_connection_witness"]
        assert witness["algebraic_not_covariant_action_derived"] is True
        assert witness["Pother_direction_nonzero_count"] == 15
        assert witness["P10_direction_nonzero_count"] == 7
        assert witness["total_nonzero_count"] == 22
        assert witness["equations_checked"] == 8910
        assert witness["nonzero_residuals"] == 0


def test_consistency_does_not_promote_D2F_or_reject(gate: dict[str, object]) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106920
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "candidate_bound_Pother_one_form_registered",
        "full_source_Jacobian_Pother_slice_projected",
        "two_sided_reference_connection_system_consistent",
        "algebraic_reference_connection_solution_constructed",
        "algebraic_two_sided_residual_zero",
    }
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("one_form_rank", "result boundary"),
        ("one_form_value", "result boundary"),
        ("system_rank", "result boundary"),
        ("residual", "result boundary"),
        ("covariant_origin", "result boundary"),
        ("admit_D2F", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("unknown_key", "result boundary"),
        ("forge_predecessor", "predecessor binding"),
        ("forge_direct", "direct evidence binding"),
        ("forge_local", "local binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "one_form_rank":
        row["candidate_bound_Pother_one_form"]["rank"] = 6
    elif mutation == "one_form_value":
        row["candidate_bound_Pother_one_form"]["nonzero_entries"][0]["value"] = "0"
    elif mutation == "system_rank":
        row["two_sided_reference_system"]["augmented_rank"] = 1871
    elif mutation == "residual":
        row["free_variable_zero_connection_witness"]["nonzero_residuals"] = 1
    elif mutation == "covariant_origin":
        value["claim_seals"]["connection_derived_from_covariant_action"] = True
    elif mutation == "admit_D2F":
        value["claim_seals"]["cross_slice_D2F_entries_admitted"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "unknown_key":
        value["promotion"] = True
    elif mutation == "forge_predecessor":
        value["source_bindings"]["two_sided_connection_identifiability"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_direct":
        value["source_bindings"]["direct_evidence"]["principal_source_replay"][
            "file_sha256"
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
