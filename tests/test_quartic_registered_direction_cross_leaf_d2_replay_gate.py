from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_registered_direction_cross_leaf_d2_replay_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    _content_sha,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result():
    return build_gate(ROOT / CONFIG_PATH)


def _reseal(value):
    value["content_sha256"] = _content_sha(value)
    return value


def test_artifact_is_exact_live_replay(result):
    artifact = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    assert artifact == result
    _validate_result(artifact, root=ROOT)


def test_full_domain_partition(result):
    partition = result["typed_full_domain_partition"]
    assert [row["per_candidate"] for row in partition] == [
        242,
        5082,
        31702,
        31702,
        188771,
    ]
    assert sum(row["per_candidate"] for row in partition) == 257499
    assert partition[1]["status"] == "registered_here"


def test_exact_cross_leaf_census(result):
    assert result["generic_cross_leaf_packet_count"] == 380
    assert result["generic_cross_leaf_nonzero_roots"] == 602
    assert result["generic_cross_leaf_zero_roots"] == 49558
    for candidate in result["candidate_manifests"]:
        assert candidate["cross_atom_leaf_packet_count"] == 380
        assert candidate["cross_atom_leaf_roots"] == 50160
        assert len(candidate["cross_atom_leaf_packets"]) == 380


def test_all_off_diagonal_entries_are_candidate_bound(result):
    record_ids = set()
    for candidate in result["candidate_manifests"]:
        records = candidate["off_diagonal_records"]
        assert len(records) == 5082
        assert (
            sum(row["leaf_jet_status"] == "reused_same_atom_registered_leaf_jet" for row in records)
            == 44
        )
        assert all(row["D1_target_slot"] != row["derivative_slot"] for row in records)
        assert all(row["candidate_id"] == candidate["candidate_id"] for row in records)
        record_ids.update(row["record_id"] for row in records)
    assert len(record_ids) == 60984


def test_progress_counts_and_first_blocker(result):
    counts = result["gate_counts"]
    assert counts["new_off_diagonal_records_per_candidate"] == 5082
    assert counts["new_off_diagonal_records_all_candidates"] == 60984
    assert counts["registered_per_candidate"] == 5324
    assert counts["remaining_per_candidate"] == 252175
    assert "131_unregistered_derivative_directions" in result["first_blocker"]


def test_broad_claims_remain_closed(result):
    claims = result["claim_seals"]
    assert claims["all_registered_direction_off_diagonal_entries_sealed"] is True
    for key in (
        "complete_D2F",
        "full_high_atom_identity",
        "global_H7",
        "nonlinear_PDE",
        "physical_no_go",
        "candidate_rejected",
    ):
        assert claims[key] is False


@pytest.mark.parametrize(
    ("mutator"),
    [
        lambda value: value["gate_counts"].__setitem__("registered_per_candidate", 257499),
        lambda value: value["claim_seals"].__setitem__("complete_D2F", True),
        lambda value: value["candidate_manifests"][0]["off_diagonal_records"][0].__setitem__(
            "D2_merkle_root_sha256", "0" * 64
        ),
        lambda value: value["source_bindings"]["direct_evidence"].__setitem__(
            "unknown", {"path": "forbidden"}
        ),
    ],
)
def test_resealed_tamper_fails_closed(result, mutator):
    tampered = copy.deepcopy(result)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(ValueError, match="differs from exact live replay"):
        _validate_result(tampered, root=ROOT)


def test_unknown_top_level_key_fails_closed(result):
    tampered = copy.deepcopy(result)
    tampered["overclaim"] = True
    _reseal(tampered)
    with pytest.raises(ValueError, match="result keys changed"):
        _validate_result(tampered, root=ROOT)


def test_config_unknown_key_fails_closed():
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    config["unknown"] = True
    with pytest.raises(ValueError, match="config boundary changed"):
        _validate_config(config)
