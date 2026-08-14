from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_s13_metric_complement_abc_leaf_authority_gate import (
    OUTPUT_PATH,
    S13MetricComplementLeafAuthorityError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def test_checked_receipt_replays() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)


def test_nine_column_complement_completes_s13_family() -> None:
    counts = _load()["gate_counts"]
    assert counts["previous_s13_metric_columns"] == 1
    assert counts["new_s13_metric_coordinate_columns"] == 9
    assert counts["complete_s13_metric_family_columns"] == 10
    assert counts["new_leaf_derivative_roots_all_candidates"] == 285120
    assert (
        counts["nonzero_leaf_derivative_roots"] + counts["exact_zero_leaf_derivative_roots"]
        == 285120
    )


def test_existing_s13_atom_is_not_recounted() -> None:
    checked = _load()
    assert checked["exact_controls"]["previous_s13_atom_not_recounted"] == "s13[6]"
    for candidate in checked["candidate_manifests"]:
        assert len(candidate["direction_packets"]) == 180
        assert all(row["derivative_atom"] != "s13[6]" for row in candidate["direction_packets"])
        assert candidate["candidate_decision"].startswith("pass_s13_metric_complement")


def test_projection_family_is_complete_and_source_bound() -> None:
    packets = _load()["projection_packets"]
    assert [row["coordinate_atom"] for row in packets] == [f"s13[{field}]" for field in range(10)]
    assert sum(row["already_registered_leaf_authority"] for row in packets) == 1
    assert all(row["delta_H"] == "0" and row["delta_v"] == "0" for row in packets)


def test_D2_and_global_claims_remain_closed() -> None:
    checked = _load()
    counts, seals = checked["gate_counts"], checked["claim_seals"]
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert counts["new_ordered_D2_roots_registered"] == 0
    assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 77
    assert seals["complete_D2F"] is False and seals["global_H7"] is False


def test_every_candidate_packet_has_exact_census() -> None:
    for candidate in _load()["candidate_manifests"]:
        assert candidate["leaf_derivative_roots"] == 23760
        assert (
            candidate["nonzero_leaf_derivative_roots"]
            + candidate["exact_zero_leaf_derivative_roots"]
            == 23760
        )
        assert all(
            row["total_leaf_derivative_roots"] == 132 for row in candidate["direction_packets"]
        )


def test_resealed_projection_D2_and_scope_tamper_fails() -> None:
    mutations = (
        lambda value: value["projection_packets"][1]["delta_G_upper"].update({"G_00": "0"}),
        lambda value: value["gate_counts"].update({"new_ordered_D2_roots_registered": 1}),
        lambda value: value["claim_seals"].update({"complete_D2F": True}),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        corrupted["content_sha256"] = _content_sha(corrupted)
        with pytest.raises(S13MetricComplementLeafAuthorityError, match="result changed"):
            validate_campaign(corrupted, root=ROOT)
