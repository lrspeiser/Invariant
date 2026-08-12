from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_cross_slice_two_sided_connection_identifiability_gate import (
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


def test_zero_and_rank_six_completions_have_opposite_consistency(
    gate: dict[str, object],
) -> None:
    counts = gate["gate_counts"]
    assert counts["two_sided_equations_per_candidate"] == 8910
    assert counts["two_sided_connection_unknowns_per_candidate"] == 11_979
    assert counts["missing_Pother_one_form_entries_per_candidate"] == 990
    assert counts["curl_flattening_rank_per_candidate"] == 6
    for row in gate["candidate_records"]:
        assert row["curl_flattening"]["rank"] == 6
        assert row["curl_flattening"]["nonzero_entry_count"] == 63
        assert row["zero_completion_witness"] == {
            "Pother_one_form_rank": 0,
            "coefficient_rank": 990,
            "augmented_rank": 991,
            "consistent": False,
        }
        completion = row["rank_six_completion_witness"]
        assert completion["coefficient_rank"] == 1518
        assert completion["augmented_rank"] == 1518
        assert completion["consistent"] is True
        assert completion["equations_checked"] == 8910
        assert completion["nonzero_residuals"] == 0


def test_constructive_completion_is_exact_sparse_and_explicitly_synthetic(
    gate: dict[str, object],
) -> None:
    for row in gate["candidate_records"]:
        completion = row["rank_six_completion_witness"]
        one_form = completion["Pother_one_form"]
        connection = completion["P10_connection_variation"]
        assert completion["synthetic_not_source_registered"] is True
        assert one_form["shape"] == [11, 90]
        assert one_form["rank"] == 6
        assert one_form["nonzero_entry_count"] == len(one_form["nonzero_entries"]) == 54
        assert connection["shape"] == [11, 11, 9]
        assert connection["nonzero_entry_count"] == len(connection["nonzero_entries"]) == 9
        assert completion["Pother_connection_variation_nonzero_entries"] == 0


def test_non_identifiability_does_not_admit_reject_or_promote(
    gate: dict[str, object],
) -> None:
    assert gate["decision_counts"] == {"pass": 0, "reject": 0, "blocked": 12}
    assert gate["first_blocker"] == FIRST_BLOCKER
    assert gate["gate_counts"]["physically_registered_completions"] == 0
    assert gate["gate_counts"]["cross_slice_entries_admitted"] == 0
    assert gate["gate_counts"]["principal_high_atom_entries_missing_per_candidate"] == 106_920
    assert gate["claim_seals"] == CLAIM_SEALS
    assert {key for key, value in CLAIM_SEALS.items() if value} == {
        "two_sided_connection_premise_identifiability_classified",
        "zero_Pother_one_form_completion_inconsistent",
        "rank_six_synthetic_completion_constructed",
        "rank_six_minimal_in_zero_Pother_connection_subclass",
    }
    assert all(row["candidate_decision"] == "blocked" for row in gate["candidate_records"])
    assert not any(row["candidate_rejection_authorized"] for row in gate["candidate_records"])
    assert not any(gate["data_seals"].values())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("curl_rank", "result boundary"),
        ("zero_consistent", "result boundary"),
        ("completion_rank", "result boundary"),
        ("completion_residual", "result boundary"),
        ("source_register", "result boundary"),
        ("admit_slice", "result boundary"),
        ("reject_candidate", "result boundary"),
        ("unknown_key", "result boundary"),
        ("forge_predecessor", "predecessor binding"),
        ("forge_local", "local binding"),
    ],
)
def test_resealed_semantic_tampering_fails_closed(
    gate: dict[str, object], mutation: str, message: str
) -> None:
    value = json.loads(json.dumps(gate))
    row = value["candidate_records"][0]
    if mutation == "curl_rank":
        row["curl_flattening"]["rank"] = 5
    elif mutation == "zero_consistent":
        row["zero_completion_witness"]["consistent"] = True
    elif mutation == "completion_rank":
        row["rank_six_completion_witness"]["Pother_one_form"]["rank"] = 5
    elif mutation == "completion_residual":
        row["rank_six_completion_witness"]["nonzero_residuals"] = 1
    elif mutation == "source_register":
        value["claim_seals"]["candidate_bound_Pother_one_form_registered"] = True
    elif mutation == "admit_slice":
        value["claim_seals"]["cross_slice_D2F_entries_admitted"] = True
    elif mutation == "reject_candidate":
        row["candidate_rejection_authorized"] = True
    elif mutation == "unknown_key":
        value["promotion"] = True
    elif mutation == "forge_predecessor":
        value["source_bindings"]["one_sided_output_connection_no_go"][
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
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        assert gate["source_bindings"][label] == {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
    for label, binding in EXPECTED_PREDECESSORS.items():
        assert gate["source_bindings"][label] == binding
