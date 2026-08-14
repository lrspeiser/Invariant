from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_remaining_scalar_hessian_abc_leaf_authority_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    RemainingScalarHessianLeafAuthorityError,
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


def test_complete_remaining_scalar_hessian_family_registers_126720_roots() -> None:
    counts = _load()["gate_counts"]
    assert counts["new_scalar_Hessian_coordinate_columns"] == 4
    assert counts["complete_scalar_Hessian_principal_family_columns"] == 9
    assert counts["candidate_bound_target_direction_pairs"] == 960
    assert counts["new_leaf_derivative_roots_per_candidate"] == 10560
    assert counts["new_leaf_derivative_roots_all_candidates"] == 126720
    assert (
        counts["nonzero_leaf_derivative_roots"] + counts["exact_zero_leaf_derivative_roots"]
        == 126720
    )


def test_four_projection_packets_are_exact_covariant_hessian_seeds() -> None:
    packets = _load()["projection_packets"]
    assert [row["coordinate_atom"] for row in packets] == [
        "s01[10]",
        "s02[10]",
        "s03[10]",
        "s33[10]",
    ]
    assert [row["coordinate_column"] for row in packets] == [64, 75, 86, 152]
    assert packets[2]["delta_H"] == {"H_03": "1"}
    assert all(row["delta_v"] == "0" and row["delta_G_upper"] == "0" for row in packets)


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


def test_zero_policy_retains_full_covariant_symbol_domain() -> None:
    checked = _load()
    dag = checked["leaf_arithmetic_DAG"]
    assert dag["node_count"] == len(dag["nodes"])
    assert "H_03" in dag["allowed_symbols"]
    assert "G_00" in dag["allowed_symbols"]
    assert checked["claim_seals"]["no_tensor_component_inferred_zero"] is True
    assert checked["exact_controls"]["infer_uncomputed_metric_tangent_zero"] == {"rejected": True}


def test_D2_count_is_preserved_without_ordered_replay() -> None:
    counts = _load()["gate_counts"]
    assert counts["potential_candidate_bound_D2_records_blocked"] == 1056
    assert counts["new_ordered_D2_roots_registered"] == 0
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert counts["remaining_D2_entries_per_candidate"] == 252175


def test_scope_closes_only_four_of_127_columns() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    seals = checked["claim_seals"]
    assert counts["registered_coordinate_columns_after"] == 30
    assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 123
    assert seals["complete_nine_direction_scalar_Hessian_principal_family_registered"] is True
    assert seals["remaining_123_coordinate_columns_registered"] is False
    assert seals["complete_D2F"] is False
    assert seals["global_H7"] is False


def test_materializer_text_hash_semantics_replay() -> None:
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


def test_resealed_sign_leaf_D2_and_scope_tamper_fail_closed() -> None:
    mutations = (
        lambda value: value["projection_packets"][2].update({"delta_H": {"H_03": "-1"}}),
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
        with pytest.raises(RemainingScalarHessianLeafAuthorityError, match="result changed"):
            validate_campaign(corrupted, root=ROOT)
