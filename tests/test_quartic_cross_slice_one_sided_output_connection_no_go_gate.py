from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_cross_slice_one_sided_output_connection_no_go_gate import (
    CLAIM_SEALS,
    CONFIG_PATH,
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


def test_full_declared_connection_system_has_exact_rank_obstruction(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["connection_system_equations_per_candidate"] == 8910
    assert counts["connection_system_unknowns_per_candidate"] == 10_890
    assert counts["connection_system_coefficient_rank"] == 990
    assert counts["connection_system_augmented_rank"] == 991
    assert counts["consistent_connection_systems"] == 0
    for row in gate["candidate_records"]:
        system = row["coefficient_system"]
        assert system["equations"] == 8910
        assert system["unknowns"] == 10_890
        assert system["coefficient_rank"] == 990
        assert system["augmented_rank"] == 991
        assert system["consistent"] is False


def test_obstruction_partition_and_maximal_declared_repair_are_exact(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["zero_one_form_direction_obstruction_groups_per_candidate"] == 18
    assert counts["inconsistent_diagonal_groups_per_candidate"] == 15
    assert counts["compatible_groups_per_candidate"] == 3
    assert counts["compatible_pair_entries_repaired_per_candidate"] == 9
    for row in gate["candidate_records"]:
        partition = row["obstruction_partition"]
        assert partition["nonzero_curl_entries"] == 63
        assert partition["active_left_output_groups"] == 36
        assert len(partition["zero_one_form_direction_obstruction_groups"]) == 18
        assert len(partition["inconsistent_diagonal_groups"]) == 15
        assert len(partition["compatible_groups"]) == 3
        assert {item["left_atom"] for item in partition["compatible_groups"]} == {
            "s12[5]",
            "s13[6]",
            "s23[8]",
        }
        assert partition["compatible_pair_entries_repaired"] == 9


def test_declared_no_go_does_not_promote_or_reject(gate: dict[str, object]) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106_920
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "one_sided_arbitrary_output_row_connection_system_classified",
        "one_sided_connection_system_inconsistent",
        "maximal_declared_compatible_subdomain_repair_constructed",
    }
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("rank", "result boundary"),
        ("partition", "result boundary"),
        ("repair_value", "result boundary"),
        ("claim_two_sided", "result boundary"),
        ("admit_slice", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("unknown_key", "result boundary"),
        ("forge_typed", "predecessor binding"),
        ("forge_repair", "predecessor binding"),
        ("forge_local", "local binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "rank":
        row["coefficient_system"]["augmented_rank"] = 990
    elif mutation == "partition":
        row["obstruction_partition"]["compatible_group_count"] = 4
    elif mutation == "repair_value":
        row["obstruction_partition"]["compatible_groups"][0]["connection_value"] = "0"
    elif mutation == "claim_two_sided":
        value["claim_seals"]["two_sided_general_output_connection_classified"] = True
    elif mutation == "admit_slice":
        value["claim_seals"]["cross_slice_D2F_entries_admitted"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "unknown_key":
        value["promotion"] = True
    elif mutation == "forge_typed":
        value["source_bindings"]["typed_map_curl"]["content_sha256"] = "0" * 64
    elif mutation == "forge_repair":
        value["source_bindings"]["output_bundle_repair"]["content_sha256"] = "0" * 64
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
