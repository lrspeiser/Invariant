from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_pgradient_abc_leaf_authority_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    PGradientLeafAuthorityError,
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


def test_four_scalar_gradient_columns_register_126720_leaf_roots() -> None:
    counts = _load()["gate_counts"]
    assert counts["new_scalar_gradient_coordinate_columns"] == 4
    assert counts["registered_target_atoms"] == 20
    assert counts["target_direction_pairs_per_candidate"] == 80
    assert counts["candidate_bound_target_direction_pairs"] == 960
    assert counts["new_leaf_derivative_roots_per_candidate"] == 10560
    assert counts["new_leaf_derivative_roots_all_candidates"] == 126720
    assert (
        counts["nonzero_leaf_derivative_roots"] + counts["exact_zero_leaf_derivative_roots"]
        == 126720
    )


def test_arbitrary_background_connection_tangents_are_explicit() -> None:
    checked = _load()
    packets = checked["tangent_packets"]
    assert [row["coordinate_atom"] for row in packets] == [
        "p0[10]",
        "p1[10]",
        "p2[10]",
        "p3[10]",
    ]
    assert [row["coordinate_column"] for row in packets] == [20, 31, 42, 53]
    assert packets[1]["delta_v"] == {"v_1": "1"}
    assert packets[1]["delta_H"]["H_22"] == "-GammaU_1_22"
    assert all(row["delta_G_upper"] == "0" for row in packets)
    assert checked["exact_controls"]["cylindrical_p1_scalar_H22"]["exact_delta_H22"] == "1"


def test_every_candidate_has_80_dense_exact_leaf_packets() -> None:
    candidates = _load()["candidate_manifests"]
    assert len(candidates) == 12
    assert len({row["candidate_id"] for row in candidates}) == 12
    for candidate in candidates:
        assert candidate["target_direction_pairs"] == 80
        assert candidate["leaf_derivative_roots"] == 10560
        assert len(candidate["direction_packets"]) == 80
        for packet in candidate["direction_packets"]:
            assert packet["total_leaf_derivative_roots"] == 132
            assert (
                packet["nonzero_leaf_derivative_roots"] + packet["exact_zero_leaf_derivative_roots"]
                == 132
            )
            assert packet["registered_arbitrary_background_connection_scope"] is True


def test_exact_zero_root_policy_does_not_set_connection_to_zero() -> None:
    checked = _load()
    dag = checked["leaf_arithmetic_DAG"]
    assert dag["node_count"] == len(dag["nodes"])
    assert len(dag["allowed_symbols"]) == 40
    assert "GammaU_1_22" in dag["allowed_symbols"]
    assert checked["claim_seals"]["no_connection_coefficient_assumed_zero"] is True
    assert checked["exact_controls"]["assume_connection_zero_on_arbitrary_background"] == {
        "rejected": True
    }


def test_D2_count_is_not_advanced_without_separate_replay() -> None:
    counts = _load()["gate_counts"]
    assert counts["potential_alias_expanded_D2_records_per_candidate"] == 88
    assert counts["potential_candidate_bound_D2_records_blocked"] == 1056
    assert counts["new_ordered_D2_roots_registered"] == 0
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert counts["remaining_D2_entries_per_candidate"] == 252175


def test_family_and_global_scope_remain_bounded() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    seals = checked["claim_seals"]
    assert counts["registered_coordinate_columns_after"] == 26
    assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 127
    assert seals["four_pgradient_coordinate_columns_registered"] is True
    assert seals["remaining_127_coordinate_columns_registered"] is False
    assert seals["complete_D2F"] is False
    assert seals["global_H7"] is False
    assert seals["candidate_theory_rejected"] is False


def test_materializer_text_hash_semantics_replay_in_this_worktree() -> None:
    checked = _load()
    bindings = checked["source_bindings"]
    for role, relative in (
        ("source", SOURCE_PATH),
        ("config", CONFIG_PATH),
        ("test", TEST_PATH),
    ):
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


def test_resealed_tangent_leaf_D2_or_scope_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["tangent_packets"][1]["delta_H"].update({"H_22": "GammaU_1_22"}),
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
        with pytest.raises(PGradientLeafAuthorityError, match="result changed"):
            validate_campaign(corrupted, root=ROOT)
