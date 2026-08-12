from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_reverse_principal_source_map_identifiability_gate import (
    CLAIM_SEALS,
    CONFIG_PATH,
    EXPECTED_DIRECT_DEPENDENCIES,
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


def test_current_exact_schemas_register_no_typed_source_map(gate: dict[str, object]) -> None:
    audit = gate["schema_audit"]
    assert audit["predecessor_labels"] == sorted(EXPECTED_PREDECESSORS)
    assert audit["exact_map_keys_found"] == []
    assert audit["audit_result"] == "typed_coordinate_to_Einstein_derivative_map_absent"
    assert gate["gate_counts"]["registered_typed_coordinate_to_Einstein_maps"] == 0


def test_two_exact_maps_produce_distinct_reverse_values_for_all_candidates(
    gate: dict[str, object],
) -> None:
    theorem = gate["nonidentifiability_theorem"]
    witness = theorem["generic_witness"]
    assert witness["left_atom"] == "s11[0]"
    assert witness["right_atom"] == "s11[10]"
    assert witness["output_row"] == 10
    assert witness["generic_zero_map_value"] == "0"
    assert witness["generic_alternative_map_value"] == "-2*alpha"
    assert witness["generic_nonzero_entries"] == [
        {"output_row": 10, "right_field": 10, "value": "-2*alpha"}
    ]
    assert gate["gate_counts"]["two_map_nonidentifiability_witnesses"] == 12
    assert gate["gate_counts"]["witnesses_with_distinct_values"] == 12
    for row in gate["candidate_records"]:
        candidate_witness = row["ambiguity_witness"]
        assert candidate_witness["zero_map_value"] == "0"
        assert candidate_witness["alternative_map_value"] != "0"
        assert candidate_witness["values_disagree"] is True
        assert candidate_witness["restricted_reverse_connection_correction"] == "0"


def test_reverse_target_remains_unmaterialized_and_candidate_bound(gate: dict[str, object]) -> None:
    counts = gate["gate_counts"]
    assert counts["reverse_ordered_pair_cells_targeted_per_candidate"] == 810
    assert counts["reverse_output_entries_targeted_per_candidate"] == 8910
    assert counts["reverse_output_entries_materialized"] == 0
    assert counts["corrected_cross_slice_curl_certificates"] == 0
    extension = json.loads(
        (ROOT / EXPECTED_PREDECESSORS["principal_high_atom_connection_extension"]["path"])
        .read_text(encoding="utf-8")
    )
    prior = {row["candidate_id"]: row for row in extension["candidate_records"]}
    assert [row["candidate_id"] for row in gate["candidate_records"]] == sorted(prior)
    for row in gate["candidate_records"]:
        assert row["coefficients"] == prior[row["candidate_id"]]["coefficients"]
        assert row["predecessor_one_sided_dense_content_sha256"] == prior[row["candidate_id"]][
            "one_sided_value_manifest"
        ]["dense_content_sha256"]


def test_only_schema_audit_ambiguity_and_restricted_connection_claims_open(
    gate: dict[str, object],
) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "current_predecessor_schemas_audited_for_typed_source_map",
        "explicit_two_map_ambiguity_witness_constructed",
        "restricted_zero_extended_connection_reverse_correction_zero",
    }
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
        ("claim_map", "result boundary"),
        ("claim_reverse", "result boundary"),
        ("map_keys_found", "result boundary"),
        ("generic_derivative", "result boundary"),
        ("generic_nonzero_list", "result boundary"),
        ("target_count", "result boundary"),
        ("materialized_count", "result boundary"),
        ("unknown_top_level", "result boundary"),
        ("unknown_binding_key", "source binding keys"),
        ("zero_ambiguity", "result boundary"),
        ("change_coordinate", "result boundary"),
        ("claim_curl", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("candidate_lineage", "result boundary"),
        ("forge_extension", "predecessor binding"),
        ("forge_coverage", "predecessor binding"),
        ("forge_scalar_d2", "predecessor binding"),
        ("forge_repair", "predecessor binding"),
        ("forge_local_config", "local binding"),
        ("forge_local_test", "local binding"),
        ("forge_local_source", "local binding"),
        ("forge_direct_source", "direct dependency boundary"),
        ("forge_direct_test", "direct dependency boundary"),
        ("forge_direct_artifact_file", "direct dependency boundary"),
        ("forge_direct_artifact_content", "direct dependency boundary"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "claim_map":
        value["claim_seals"]["typed_coordinate_to_Einstein_derivative_map_registered"] = True
    elif mutation == "claim_reverse":
        value["claim_seals"]["reverse_Pother_by_P10_values_materialized"] = True
    elif mutation == "map_keys_found":
        value["schema_audit"]["exact_map_keys_found"] = [
            "coordinate_to_einstein_derivative_map"
        ]
    elif mutation == "generic_derivative":
        value["nonidentifiability_theorem"]["generic_witness"][
            "generic_alternative_map_value"
        ] = "-alpha"
    elif mutation == "generic_nonzero_list":
        value["nonidentifiability_theorem"]["generic_witness"][
            "generic_nonzero_entries"
        ].append({"output_row": 0, "right_field": 0, "value": "1"})
    elif mutation == "target_count":
        value["gate_counts"]["reverse_output_entries_targeted_per_candidate"] = 8_909
    elif mutation == "materialized_count":
        value["gate_counts"]["reverse_output_entries_materialized"] = 1
    elif mutation == "unknown_top_level":
        value["unregistered_promotion"] = True
    elif mutation == "unknown_binding_key":
        value["source_bindings"]["unexpected"] = {}
    elif mutation == "zero_ambiguity":
        row["ambiguity_witness"]["alternative_map_value"] = "0"
    elif mutation == "change_coordinate":
        value["nonidentifiability_theorem"]["generic_witness"]["left_atom"] = "s22[0]"
    elif mutation == "claim_curl":
        value["claim_seals"]["corrected_cross_slice_curl_zero"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "candidate_lineage":
        row["predecessor_one_sided_dense_content_sha256"] = "0" * 64
    elif mutation == "forge_extension":
        value["source_bindings"]["principal_high_atom_connection_extension"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_coverage":
        value["source_bindings"]["full_d2f_high_atom_coverage"]["content_sha256"] = "0" * 64
    elif mutation == "forge_scalar_d2":
        value["source_bindings"]["scalar_hessian_d2"]["content_sha256"] = "0" * 64
    elif mutation == "forge_repair":
        value["source_bindings"]["scalar_hessian_output_bundle_repair"][
            "content_sha256"
        ] = "0" * 64
    elif mutation == "forge_local_config":
        value["source_bindings"]["config"]["file_sha256"] = "0" * 64
    elif mutation == "forge_local_test":
        value["source_bindings"]["test"]["file_sha256"] = "0" * 64
    elif mutation == "forge_local_source":
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    elif mutation == "forge_direct_source":
        value["source_bindings"]["direct_dependencies"][
            "unspecialized_source_jacobian_campaign"
        ]["source"]["file_sha256"] = "0" * 64
    elif mutation == "forge_direct_test":
        value["source_bindings"]["direct_dependencies"][
            "unspecialized_source_jacobian_campaign"
        ]["test"]["file_sha256"] = "0" * 64
    elif mutation == "forge_direct_artifact_file":
        value["source_bindings"]["direct_dependencies"][
            "unspecialized_source_jacobian_campaign"
        ]["artifact"]["file_sha256"] = "0" * 64
    else:
        value["source_bindings"]["direct_dependencies"][
            "unspecialized_source_jacobian_campaign"
        ]["artifact"]["content_sha256"] = "0" * 64
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
    assert gate["source_bindings"]["direct_dependencies"] == EXPECTED_DIRECT_DEPENDENCIES
    dependency = EXPECTED_DIRECT_DEPENDENCIES["unspecialized_source_jacobian_campaign"]
    for label in ("source", "test", "artifact"):
        assert hashlib.sha256((ROOT / dependency[label]["path"]).read_bytes()).hexdigest() == (
            dependency[label]["file_sha256"]
        )
    direct_artifact = json.loads(
        (ROOT / dependency["artifact"]["path"]).read_text(encoding="utf-8")
    )
    assert direct_artifact["content_sha256"] == dependency["artifact"]["content_sha256"]
