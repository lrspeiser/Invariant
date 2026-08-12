from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_reverse_principal_typed_map_curl_gate import (
    CLAIM_SEALS,
    CONFIG_PATH,
    EXPECTED_DIRECT_DEPENDENCIES,
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


def test_exact_flat_typed_map_is_directly_bound_and_registered(gate: dict[str, object]) -> None:
    theorem = gate["typed_map_theorem"]
    assert theorem["map_schema"] == "sigma-flat-coordinate-153-to-covariant-24-Jacobian-1.0"
    assert theorem["map_content_sha256"] == (
        "bbb9790adec7f1551945263bc6b7910204dcab3c51b0f6bc62e76553bf50246f"
    )
    assert gate["gate_counts"]["typed_coordinate_maps_registered"] == 1
    assert gate["gate_counts"]["typed_map_coordinate_atoms"] == 153
    assert gate["gate_counts"]["typed_map_covariant_jet_symbols"] == 24
    dependency = EXPECTED_DIRECT_DEPENDENCIES["variable_sylvester_coordinate_map"]
    assert gate["source_bindings"]["direct_dependencies"] == EXPECTED_DIRECT_DEPENDENCIES
    for label in ("source", "test", "artifact"):
        assert hashlib.sha256((ROOT / dependency[label]["path"]).read_bytes()).hexdigest() == (
            dependency[label]["file_sha256"]
        )


def test_reverse_values_and_corrected_curl_are_complete_candidate_bound_manifests(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["other_principal_atoms_mapped"] == 90
    assert counts["reverse_ordered_pair_cells_per_candidate"] == 810
    assert counts["reverse_entries_materialized_per_candidate"] == 8910
    assert counts["reverse_nonzero_entries_per_candidate"] == 75
    assert counts["corrected_curl_entries_checked_per_candidate"] == 8910
    assert counts["corrected_curl_nonzero_entries_per_candidate"] == 63
    assert counts["candidates_with_nonzero_corrected_curl"] == 12
    for row in gate["candidate_records"]:
        reverse = row["reverse_value_manifest"]
        curl = row["corrected_ordered_curl_manifest"]
        assert reverse["shape"] == [11, 90, 9]
        assert reverse["entry_count"] == 8910
        assert reverse["nonzero_entry_count"] == len(reverse["nonzero_entries"]) == 75
        assert curl["shape"] == [11, 90, 9]
        assert curl["entry_count"] == 8910
        assert curl["nonzero_entry_count"] == len(curl["nonzero_entries"]) == 63
        assert row["restricted_reverse_connection_nonzero_corrections"] == 0


def test_nonzero_curl_blocks_admission_without_rejecting_candidates(
    gate: dict[str, object],
) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106_920
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "flat_coordinate_to_covariant_jet_map_registered",
        "other_principal_to_Einstein_submap_registered",
        "reverse_Pother_by_P10_values_materialized",
        "corrected_cross_slice_curl_completely_materialized",
    }
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("map_hash", "result boundary"),
        ("reverse_value", "result boundary"),
        ("curl_value", "result boundary"),
        ("claim_zero_curl", "result boundary"),
        ("admit_slice", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("unknown_key", "result boundary"),
        ("forge_reverse_predecessor", "predecessor binding"),
        ("forge_extension_predecessor", "predecessor binding"),
        ("forge_direct_source", "direct dependency binding"),
        ("forge_direct_test", "direct dependency binding"),
        ("forge_direct_artifact", "direct dependency binding"),
        ("forge_local", "local binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "map_hash":
        value["typed_map_theorem"]["map_content_sha256"] = "0" * 64
    elif mutation == "reverse_value":
        row["reverse_value_manifest"]["nonzero_entries"][0]["value"] = "0"
    elif mutation == "curl_value":
        row["corrected_ordered_curl_manifest"]["nonzero_entries"][0]["value"] = "0"
    elif mutation == "claim_zero_curl":
        value["claim_seals"]["corrected_cross_slice_curl_zero"] = True
    elif mutation == "admit_slice":
        value["claim_seals"]["cross_slice_D2F_entries_admitted"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "unknown_key":
        value["promotion"] = True
    elif mutation == "forge_reverse_predecessor":
        value["source_bindings"]["reverse_source_map_identifiability"]["content_sha256"] = (
            "0" * 64
        )
    elif mutation == "forge_extension_predecessor":
        value["source_bindings"]["principal_high_atom_connection_extension"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_direct_source":
        value["source_bindings"]["direct_dependencies"]["variable_sylvester_coordinate_map"][
            "source"
        ]["file_sha256"] = "0" * 64
    elif mutation == "forge_direct_test":
        value["source_bindings"]["direct_dependencies"]["variable_sylvester_coordinate_map"][
            "test"
        ]["file_sha256"] = "0" * 64
    elif mutation == "forge_direct_artifact":
        value["source_bindings"]["direct_dependencies"]["variable_sylvester_coordinate_map"][
            "artifact"
        ]["content_sha256"] = "0" * 64
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
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        assert gate["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
    for label, binding in EXPECTED_PREDECESSORS.items():
        assert gate["source_bindings"][label] == binding
