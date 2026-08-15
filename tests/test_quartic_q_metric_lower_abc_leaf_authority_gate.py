from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_q_metric_lower_abc_leaf_authority_gate import (
    OUTPUT_PATH,
    QMetricLowerLeafAuthorityError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def _zero_first_nonzero_second_jet_tangent(value: dict) -> None:
    for packet in value["coordinate_tangent_packets"]:
        for label, coefficient in packet["delta_G_upper"].items():
            if "S" in coefficient:
                packet["delta_G_upper"][label] = "0"
                return
    raise AssertionError("checked q packet has no second-jet coefficient")


def test_checked_campaign_is_exact_tangent_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)
    assert checked["content_sha256"] == _content_sha(checked)


def test_all_ten_q_metric_tangents_retain_the_full_two_jet_domain() -> None:
    checked = _load()
    packets = checked["coordinate_tangent_packets"]
    assert [row["coordinate_atom"] for row in packets] == [f"q[{field}]" for field in range(10)]
    assert [row["coordinate_column"] for row in packets] == list(range(10))
    assert all(row["seed"]["dP"] == "0" and row["seed"]["dS"] == "0" for row in packets)
    assert all(len(row["delta_H"]) == 10 for row in packets)
    assert all(len(row["delta_G_upper"]) == 10 for row in packets)
    program = checked["coordinate_tangent_program"]
    assert program["primitive_symbol_count"] == 144
    assert program["materialized_unexpanded_exact_scalar_values"] == 200
    assert program["arbitrary_consistent_first_and_second_metric_jets"] is True
    assert program["no_flat_reference_jet_specialization"] is True
    assert program["factor_terms_or_expand_applied"] is False


def test_primitive_manifest_is_exactly_40_P_100_S_and_4_v_symbols() -> None:
    symbols = set(_load()["coordinate_tangent_program"]["primitive_symbols"])
    assert len(symbols) == 144
    assert sum(name.startswith("P") for name in symbols) == 40
    assert sum(name.startswith("S") for name in symbols) == 100
    assert sum(name.startswith("v_") for name in symbols) == 4
    serialized = json.dumps(_load()["coordinate_tangent_packets"], sort_keys=True)
    assert "S00_11" in serialized
    assert "P0_00" in serialized


def test_alias_accounting_is_recorded_without_promoting_q_leaf_authority() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    controls = checked["exact_controls"]["formal_slot_alias_correction"]
    assert controls == {
        "formal_slots": 145,
        "duplicate_excess": 2,
        "unique_columns_before": 143,
        "duplicate_atoms": ["s11[10]", "s22[10]"],
    }
    assert counts["predecessor_unique_registered_coordinate_columns"] == 143
    assert counts["unique_registered_coordinate_columns_after"] == 143
    assert counts["remaining_unique_coordinate_columns_without_A_B_C_leaf_authority"] == 10
    assert checked["claim_seals"]["all_153_unique_coordinate_leaf_authorities_registered"] is False


def test_resumable_plan_closes_inventory_but_materializes_no_leaf_root() -> None:
    checked = _load()
    plan = checked["resumable_leaf_composition_contract"]
    counts = checked["gate_counts"]
    assert len(plan["target_atoms"]) == 20
    assert len(plan["candidate_ids"]) == 12
    assert plan["q_tangent_atoms"] == [f"q[{field}]" for field in range(10)]
    assert plan["checkpoint_units"] == 120
    assert plan["planned_leaf_roots_all_candidates"] == 316800
    assert counts["planned_leaf_roots_all_candidates"] == 316800
    assert counts["materialized_leaf_roots_all_candidates"] == 0
    assert checked["claim_seals"]["complete_q_metric_leaf_family_registered"] is False


def test_d2_and_global_claims_remain_fail_closed() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    seals = checked["claim_seals"]
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["new_ordered_D2_roots_registered_per_candidate"] == 0
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert seals["D2_entry_count_advanced"] is False
    assert seals["complete_D2F"] is False
    assert seals["global_H7"] is False


def test_resealed_alias_second_jet_leaf_or_d2_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["gate_counts"].update({"predecessor_duplicate_formal_slot_excess": 0}),
        _zero_first_nonzero_second_jet_tangent,
        lambda value: value["gate_counts"].update({"materialized_leaf_roots_all_candidates": 1}),
        lambda value: value["gate_counts"].update(
            {"registered_D2_entries_per_candidate_after": 5325}
        ),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(QMetricLowerLeafAuthorityError, match="checked result changed"):
            validate_campaign(corrupted, root=ROOT)
