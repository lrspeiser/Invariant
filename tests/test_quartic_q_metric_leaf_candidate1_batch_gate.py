from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_q_metric_leaf_candidate1_batch_gate import (
    OUTPUT_PATH,
    QMetricLeafCandidate1BatchError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def test_checked_batch_is_exact_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)


def test_candidate1_has_ten_independently_sealed_units_and_one_shared_dag() -> None:
    checked = _load()
    units = checked["units"]
    assert [unit["unit_index"] for unit in units] == list(range(10, 20))
    assert [unit["q_atom"] for unit in units] == [f"q[{field}]" for field in range(10)]
    assert {unit["candidate_ordinal"] for unit in units} == {1}
    assert len({unit["candidate_id"] for unit in units}) == 1
    assert len({unit["leaf_arithmetic_DAG_sha256"] for unit in units}) == 1
    assert all(
        unit["leaf_arithmetic_DAG_sha256"]
        == checked["shared_leaf_arithmetic_DAG"]["content_sha256"]
        for unit in units
    )
    assert sum(unit["leaf_derivative_roots"] for unit in units) == 26400


def test_unit_chain_binds_candidate0_head_and_each_candidate1_unit() -> None:
    checked = _load()
    predecessor = checked["unit_chain_predecessor_sha256"]
    for unit in checked["units"]:
        assert unit["predecessor_unit_content_sha256"] == predecessor
        predecessor = unit["content_sha256"]
    assert checked["unit_chain_head_sha256"] == predecessor


def test_dense_manifests_and_exact_root_census_are_complete() -> None:
    checked = _load()
    dag_sha = checked["shared_leaf_arithmetic_DAG"]["content_sha256"]
    for unit in checked["units"]:
        assert len(unit["direction_packets"]) == 20
        assert (
            unit["nonzero_leaf_derivative_roots"] + unit["exact_zero_leaf_derivative_roots"] == 2640
        )
        for packet in unit["direction_packets"]:
            assert packet["leaf_arithmetic_DAG_sha256"] == dag_sha
            assert (
                packet["nonzero_leaf_derivative_roots"] + packet["exact_zero_leaf_derivative_roots"]
                == 132
            )
            assert len(packet["dense_root_manifest_sha256"]) == 64


def test_cumulative_counts_hold_unique_and_d2_boundaries() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    assert (
        counts["cumulative_completed_units"],
        counts["remaining_units"],
        counts["cumulative_leaf_roots"],
        counts["remaining_leaf_roots"],
    ) == (20, 100, 52800, 264000)
    assert counts["new_leaf_roots"] == 26400
    assert counts["nonzero_leaf_roots"] + counts["exact_zero_leaf_roots"] == 26400
    assert counts["unique_registered_coordinate_columns_after"] == 143
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    seals = checked["claim_seals"]
    assert seals["candidate1_all_q_units_complete"] is True
    assert seals["all_120_units_complete"] is False
    assert seals["complete_q_metric_leaf_family_registered"] is False
    assert seals["all_153_unique_coordinate_leaf_authorities_registered"] is False
    assert seals["D2_entry_count_advanced"] is False


def test_resealed_chain_root_or_boundary_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["units"][0].update({"predecessor_unit_content_sha256": "0" * 64}),
        lambda value: value["units"][0]["direction_packets"][0].update(
            {"zero_default_arithmetic_root": 1}
        ),
        lambda value: value["gate_counts"].update({"remaining_units": 0}),
        lambda value: value["gate_counts"].update(
            {"registered_D2_entries_per_candidate_after": 5325}
        ),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        corrupted["content_sha256"] = _content_sha(corrupted)
        with pytest.raises(QMetricLeafCandidate1BatchError, match="checked result changed"):
            validate_campaign(corrupted, root=ROOT)
