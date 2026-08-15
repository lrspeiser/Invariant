from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_q_metric_leaf_unit_000_gate import (
    OUTPUT_PATH,
    QMetricLeafUnitError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def test_checked_unit_is_exact_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)


def test_first_candidate_q0_unit_materializes_all_2640_roots() -> None:
    checked = _load()
    unit = checked["leaf_unit"]
    assert unit["unit_index"] == 0
    assert unit["candidate_ordinal"] == 0
    assert unit["q_atom"] == "q[0]"
    assert unit["q_column"] == 0
    assert len(unit["target_atoms"]) == 20
    assert len(unit["direction_packets"]) == 20
    assert unit["leaf_derivative_roots"] == 2640
    assert unit["nonzero_leaf_derivative_roots"] > 0
    assert unit["exact_zero_leaf_derivative_roots"] > 0
    assert unit["nonzero_leaf_derivative_roots"] + unit["exact_zero_leaf_derivative_roots"] == 2640


def test_unit_dag_uses_expand_mul_and_full_two_jet_symbols() -> None:
    dag = _load()["leaf_arithmetic_DAG"]
    assert dag["allowed_operation"] == "exact_expand_mul_expression"
    assert dag["factor_terms_used"] is False
    assert len(dag["allowed_symbols"]) == 144
    allowed = set(dag["allowed_symbols"])
    assert sum(name.startswith("P") for name in allowed) == 40
    assert sum(name.startswith("S") for name in allowed) == 100
    assert sum(name.startswith("v_") for name in allowed) == 4
    assert all(set(node["free_symbols"]) <= allowed for node in dag["nodes"])


def test_each_target_packet_has_exact_dense_132_root_manifest() -> None:
    checked = _load()
    dag_sha = checked["leaf_arithmetic_DAG"]["content_sha256"]
    for packet in checked["leaf_unit"]["direction_packets"]:
        assert packet["derivative_atom"] == "q[0]"
        assert packet["derivative_coordinate_column"] == 0
        assert packet["total_leaf_derivative_roots"] == 132
        assert packet["leaf_arithmetic_DAG_sha256"] == dag_sha
        assert (
            packet["nonzero_leaf_derivative_roots"] + packet["exact_zero_leaf_derivative_roots"]
            == 132
        )


def test_progress_is_one_of_120_without_q_family_or_d2_promotion() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    seals = checked["claim_seals"]
    assert counts["completed_checkpoint_units"] == 1
    assert counts["remaining_checkpoint_units"] == 119
    assert counts["materialized_leaf_roots"] == 2640
    assert counts["remaining_leaf_roots"] == 314160
    assert counts["unique_registered_coordinate_columns_after"] == 143
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert seals["complete_q_metric_leaf_family_registered"] is False
    assert seals["all_153_unique_coordinate_leaf_authorities_registered"] is False
    assert seals["D2_entry_count_advanced"] is False


def test_resealed_root_progress_or_d2_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["leaf_unit"]["direction_packets"][0].update(
            {"zero_default_arithmetic_root": 1}
        ),
        lambda value: value["gate_counts"].update({"completed_checkpoint_units": 120}),
        lambda value: value["claim_seals"].update(
            {"complete_q_metric_leaf_family_registered": True}
        ),
        lambda value: value["gate_counts"].update(
            {"registered_D2_entries_per_candidate_after": 5325}
        ),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(QMetricLeafUnitError, match="checked result changed"):
            validate_campaign(corrupted, root=ROOT)
