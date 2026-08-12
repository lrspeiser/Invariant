from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_full_d2f_high_atom_coverage_gate import (
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
        _canonical({key: value for key, value in checked.items() if key != "content_sha256"})
    ).hexdigest()


def test_atom_registry_and_all_ordered_pair_cells_are_closed_world(
    gate: dict[str, object],
) -> None:
    registry = gate["atom_registry"]
    assert len(registry) == 153
    assert [row["coordinate_column"] for row in registry] == list(range(153))
    assert len({row["coordinate_atom"] for row in registry}) == 153
    assert sum(row["atom_class"] == "lower" for row in registry) == 54
    assert sum(row["atom_class"] == "principal_other" for row in registry) == 90
    assert sum(row["atom_class"] == "principal_high_field10" for row in registry) == 9
    ledger = gate["ordered_coverage_ledger"]
    assert ledger["shape"] == [11, 153, 153]
    assert ledger["ordered_pair_cell_count"] == 23_409
    assert ledger["ordered_D2F_entry_count"] == 257_499
    assert len(ledger["row_packets"]) == 153
    assert all(row["ordered_pair_cells"] == 153 for row in ledger["row_packets"])


def test_exact_disjoint_partition_counts_and_missing_high_atom_domain(
    gate: dict[str, object],
) -> None:
    ledger = gate["ordered_coverage_ledger"]
    assert ledger["pair_status_counts"] == {
        "corrected_admitted": 81,
        "lower_lower_not_registered": 2916,
        "lower_principal_not_registered": 5346,
        "naive_evaluated_not_admitted": 810,
        "other_principal_pair_not_registered": 8100,
        "principal_lower_not_registered": 5346,
        "reverse_principal_not_registered": 810,
    }
    assert sum(ledger["pair_status_counts"].values()) == 153**2
    assert sum(ledger["entry_status_counts"].values()) == 11 * 153**2
    assert ledger["principal_high_atom_entries"] == 11 * 99**2 == 107_811
    assert ledger["principal_high_atom_entries_admitted"] == 11 * 9**2 == 891
    assert ledger["principal_high_atom_entries_missing"] == 106_920
    assert ledger["full_ordered_D2F_entries_missing"] == 256_608


def test_candidates_remain_exactly_bound_blocked_and_unrejected(
    gate: dict[str, object],
) -> None:
    repair = json.loads(
        (ROOT / EXPECTED_PREDECESSORS["scalar_hessian_output_bundle_repair"]["path"])
        .read_text(encoding="utf-8")
    )
    expected_ids = sorted(row["candidate_id"] for row in repair["candidate_records"])
    assert [row["candidate_id"] for row in gate["candidate_records"]] == expected_ids
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    for row in gate["candidate_records"]:
        assert row["candidate_decision"] == "blocked"
        assert row["candidate_rejection_authorized"] is False
        assert row["corrected_admitted_entries"] == 891
        assert row["ordered_pair_classification_root_sha256"] == gate[
            "ordered_coverage_ledger"
        ]["ordered_pair_classification_root_sha256"]
        assert row["first_blocker"] == FIRST_BLOCKER


def test_only_domain_classification_and_repaired_slice_claims_are_open(
    gate: dict[str, object],
) -> None:
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "complete_ordered_D2F_coverage_domain_classified",
        "corrected_scalar_hessian_high_field10_submanifest_admitted",
        "remaining_principal_high_atom_domain_exactly_classified",
    }
    assert gate["gate_counts"]["complete_ordered_D2F_tensors_registered"] == 0
    assert gate["gate_counts"]["full_high_atom_good_unknown_identities_proved"] == 0
    assert gate["gate_counts"]["global_H7_closures"] == 0
    assert gate["gate_counts"]["nonlinear_PDE_closures"] == 0
    assert gate["gate_counts"]["lifespans_proved"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("row_root", "result boundary"),
        ("partition", "result boundary"),
        ("promote_full_d2", "result boundary"),
        ("promote_identity", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("forge_predecessor", "predecessor binding"),
        ("forge_source", "local binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    if mutation == "row_root":
        value["ordered_coverage_ledger"]["row_packets"][0][
            "row_classification_root_sha256"
        ] = "0" * 64
    elif mutation == "partition":
        value["ordered_coverage_ledger"]["pair_status_counts"]["corrected_admitted"] += 1
    elif mutation == "promote_full_d2":
        value["claim_seals"]["complete_ordered_D2F_tensor_registered"] = True
    elif mutation == "promote_identity":
        value["claim_seals"]["full_high_atom_good_unknown_identity_proved"] = True
    elif mutation == "reject_candidate":
        value["candidate_records"][0]["candidate_rejection_authorized"] = True
    elif mutation == "forge_predecessor":
        value["source_bindings"]["scalar_hessian_output_bundle_repair"][
            "content_sha256"
        ] = "0" * 64
    else:
        value["source_bindings"]["source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=message):
        _validate_result(_reseal(value), root=ROOT)


def test_config_paths_and_four_file_bindings_fail_closed(gate: dict[str, object]) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["policies"]["global_H7"] = "pass"
    with pytest.raises(ValueError, match="config boundary"):
        _validate_config(config)
    with pytest.raises(ValueError, match="path escapes"):
        _load_bound(
            ROOT,
            {"path": "../outside.json", "file_sha256": "0" * 64, "content_sha256": "0" * 64},
        )
    bindings = gate["source_bindings"]
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        assert bindings[label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
