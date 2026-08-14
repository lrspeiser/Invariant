from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_s03_metric_abc_leaf_authority_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    S03MetricLeafAuthorityError,
    _content_sha,
    _matches_text_authority,
    _production_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def test_checked_campaign_is_exact_live_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)


def test_complete_s03_metric_family_registers_316800_roots() -> None:
    counts = _load()["gate_counts"]
    assert counts["new_s03_metric_coordinate_columns"] == 10
    assert counts["target_direction_pairs_per_candidate"] == 200
    assert counts["candidate_bound_target_direction_pairs"] == 2400
    assert counts["new_leaf_derivative_roots_per_candidate"] == 26400
    assert counts["new_leaf_derivative_roots_all_candidates"] == 316800
    assert (
        counts["nonzero_leaf_derivative_roots"] + counts["exact_zero_leaf_derivative_roots"]
        == 316800
    )


def test_s03_projection_packets_cover_all_metric_components() -> None:
    packets = _load()["projection_packets"]
    assert [row["coordinate_atom"] for row in packets] == [f"s03[{field}]" for field in range(10)]
    assert [row["coordinate_column"] for row in packets] == list(range(76, 86))
    assert [row["coordinate_atom"] for row in packets if row["exact_zero_projection"]] == [
        "s03[0]",
        "s03[9]",
    ]
    assert all(row["delta_H"] == "0" and row["delta_v"] == "0" for row in packets)


def test_every_candidate_has_200_dense_leaf_packets() -> None:
    candidates = _load()["candidate_manifests"]
    assert len(candidates) == 12
    for candidate in candidates:
        assert candidate["target_direction_pairs"] == 200
        assert candidate["leaf_derivative_roots"] == 26400
        assert len(candidate["direction_packets"]) == 200
        assert all(
            row["total_leaf_derivative_roots"] == 132 for row in candidate["direction_packets"]
        )


def test_zero_projection_is_exact_not_inferred() -> None:
    checked = _load()
    assert checked["gate_counts"]["exact_zero_covariant_projection_columns"] == 2
    assert checked["exact_controls"]["exact_zero_projection_atoms"] == ["s03[0]", "s03[9]"]
    assert checked["claim_seals"]["no_tensor_component_inferred_zero"] is True
    assert checked["exact_controls"]["infer_uncomputed_tensor_zero"] == {"rejected": True}


def test_D2_count_is_held_without_ordered_replay() -> None:
    counts = _load()["gate_counts"]
    assert counts["potential_candidate_bound_D2_records_blocked"] == 2640
    assert counts["new_ordered_D2_roots_registered"] == 0
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["registered_D2_entries_per_candidate_after"] == 5324


def test_scope_closes_only_ten_of_123_columns() -> None:
    checked = _load()
    counts, seals = checked["gate_counts"], checked["claim_seals"]
    assert counts["registered_coordinate_columns_after"] == 40
    assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 113
    assert seals["complete_s03_metric_family_registered"] is True
    assert seals["remaining_113_coordinate_columns_registered"] is False
    assert seals["complete_D2F"] is False and seals["global_H7"] is False


def test_materializer_text_hash_semantics_replay() -> None:
    bindings = _load()["source_bindings"]
    for role, relative in (("source", SOURCE_PATH), ("config", CONFIG_PATH), ("test", TEST_PATH)):
        assert bindings[role]["production_file_sha256"] == _production_sha(ROOT / relative)
    for bundle in bindings["evidence"].values():
        stem = bundle["stem"]
        assert _matches_text_authority(
            ROOT / f"src/sigma_theory_compiler/{stem}.py", bundle["source_sha256"]
        )
        assert _matches_text_authority(
            ROOT / f"configs/backgrounds/{stem}.json", bundle["config_sha256"]
        )
        assert _matches_text_authority(ROOT / f"tests/test_{stem}.py", bundle["test_sha256"])


def test_resealed_sign_leaf_D2_and_scope_tamper_fail_closed() -> None:
    mutations = (
        lambda value: value["projection_packets"][1]["delta_G_upper"].update({"G_00": "0"}),
        lambda value: value["candidate_manifests"][0]["direction_packets"][0].update(
            {"exact_zero_leaf_derivative_roots": 0}
        ),
        lambda value: value["gate_counts"].update({"new_ordered_D2_roots_registered": 1}),
        lambda value: value["claim_seals"].update({"complete_D2F": True}),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(S03MetricLeafAuthorityError, match="result changed"):
            validate_campaign(corrupted, root=ROOT)
