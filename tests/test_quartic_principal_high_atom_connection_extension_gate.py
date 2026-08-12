from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_principal_high_atom_connection_extension_gate import (
    CLAIM_SEALS,
    CONFIG_PATH,
    EXPECTED_PREDECESSORS,
    EXPECTED_SEALS,
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
        _canonical({key: value for key, value in checked.items() if key != "content_sha256"})
    ).hexdigest()


def test_exact_subset_is_nine_by_ninety_principal_atoms(gate: dict[str, object]) -> None:
    subset = gate["subset_registry"]
    assert len(subset["left_atoms"]) == 9
    assert len(subset["right_atoms"]) == 90
    assert len(set(subset["left_atoms"])) == 9
    assert len({row["coordinate_atom"] for row in subset["right_atoms"]}) == 90
    assert {int(row["coordinate_atom"].split("[")[1][:-1]) for row in subset["right_atoms"]} == set(
        range(10)
    )
    assert subset["ordered_pair_cell_count"] == 810
    assert subset["output_entry_count"] == 8910


def test_registered_B10_extension_has_exactly_zero_effect(gate: dict[str, object]) -> None:
    counts = gate["gate_counts"]
    assert counts["source_scalar_row_entries_checked_per_candidate"] == 90
    assert counts["source_scalar_row_nonzero_entries"] == 0
    assert counts["restricted_connection_correction_entries_checked_per_candidate"] == 8910
    assert counts["restricted_connection_nonzero_corrections"] == 0
    assert counts["one_sided_values_materialized_per_candidate"] == 8910
    assert counts["one_sided_nonzero_values_per_candidate"] == 93
    for row in gate["candidate_records"]:
        assert row["source_scalar_row_manifest"]["entry_count"] == 90
        assert row["source_scalar_row_manifest"]["nonzero_entry_count"] == 0
        assert row["connection_correction_manifest"]["entry_count"] == 8910
        assert row["connection_correction_manifest"]["nonzero_entry_count"] == 0
        assert row["one_sided_value_manifest"]["entry_count"] == 8910
        assert row["one_sided_value_manifest"]["nonzero_entry_count"] == 93
        assert len(row["one_sided_value_manifest"]["nonzero_entries"]) == 93
        assert row["restricted_connection_extension_decision"] == (
            "exact_no_effect_on_P10_by_Pother_because_J_10_right_equals_zero"
        )


def test_candidate_lineage_binds_current_full_coverage_gate(gate: dict[str, object]) -> None:
    coverage = json.loads(
        (ROOT / EXPECTED_PREDECESSORS["full_d2f_high_atom_coverage"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    expected = {row["candidate_id"]: row for row in coverage["candidate_records"]}
    assert [row["candidate_id"] for row in gate["candidate_records"]] == sorted(expected)
    for row in gate["candidate_records"]:
        prior = expected[row["candidate_id"]]
        assert row["coefficients"] == prior["coefficients"]
        assert row["predecessor_ordered_pair_classification_root_sha256"] == prior[
            "ordered_pair_classification_root_sha256"
        ]
        assert row["predecessor_corrected_D2_submanifest_content_sha256"] == prior[
            "corrected_D2_submanifest_content_sha256"
        ]


def test_cross_slice_and_all_broad_claims_remain_fail_closed(gate: dict[str, object]) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "other_principal_atom_subset_exactly_registered",
        "scalar_source_row_10_zero_on_other_principal_subset",
        "registered_B10_connection_correction_zero_on_P10_by_Pother",
        "one_sided_P10_by_Pother_values_materialized",
    }
    assert gate["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106_920
    assert gate["gate_counts"]["complete_ordered_D2F_tensors_registered"] == 0
    assert gate["gate_counts"]["full_high_atom_good_unknown_identities_proved"] == 0
    assert gate["gate_counts"]["global_H7_closures"] == 0
    assert gate["gate_counts"]["nonlinear_PDE_closures"] == 0
    assert gate["gate_counts"]["lifespans_proved"] == 0
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert gate["data_seals"] == EXPECTED_SEALS
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("source_root", "result boundary"),
        ("correction_root", "result boundary"),
        ("sparse_value", "result boundary"),
        ("admit_cross_slice", "result boundary"),
        ("promote_identity", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("candidate_lineage", "result boundary"),
        ("forge_full_d2_predecessor", "predecessor binding"),
        ("forge_local_test", "local binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "source_root":
        row["source_scalar_row_manifest"]["dense_content_sha256"] = "0" * 64
    elif mutation == "correction_root":
        row["connection_correction_manifest"]["dense_content_sha256"] = "0" * 64
    elif mutation == "sparse_value":
        row["one_sided_value_manifest"]["nonzero_entries"][0]["value"] = "0"
    elif mutation == "admit_cross_slice":
        value["claim_seals"]["one_sided_P10_by_Pother_values_admitted_as_covariant_D2F"] = True
    elif mutation == "promote_identity":
        value["claim_seals"]["full_high_atom_good_unknown_identity_proved"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "candidate_lineage":
        row["predecessor_ordered_pair_classification_root_sha256"] = "0" * 64
    elif mutation == "forge_full_d2_predecessor":
        value["source_bindings"]["full_d2f_high_atom_coverage"]["content_sha256"] = "0" * 64
    else:
        value["source_bindings"]["test"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        _validate_result(_reseal(value), root=ROOT)


def test_config_paths_and_all_bindings_fail_closed(gate: dict[str, object]) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="path escapes"):
        _load_bound(
            ROOT,
            {"path": "../outside.json", "file_sha256": "0" * 64, "content_sha256": "0" * 64},
        )
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        assert gate["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
    for label, binding in EXPECTED_PREDECESSORS.items():
        assert gate["source_bindings"][label] == binding
