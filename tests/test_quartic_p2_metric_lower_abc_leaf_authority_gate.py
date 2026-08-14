from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_p2_metric_lower_abc_leaf_authority_gate import (
    OUTPUT_PATH,
    P2MetricLowerLeafAuthorityError,
    _content_sha,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def _load() -> dict:
    return json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def _zero_first_nonzero_hessian_tangent(value: dict) -> None:
    tangent = value["coordinate_tangent_packets"][1]["delta_H"]
    label = next(label for label, coefficient in tangent.items() if coefficient != "0")
    tangent[label] = "0"


def test_checked_campaign_is_exact_live_replay() -> None:
    checked = _load()
    assert build_campaign(root=ROOT) == checked
    validate_campaign(checked, root=ROOT)
    assert checked["content_sha256"] == _content_sha(checked)


def test_all_ten_p2_metric_tangents_are_materialized_by_exact_spatial_permutation() -> None:
    checked = _load()
    packets = checked["coordinate_tangent_packets"]
    assert [row["coordinate_atom"] for row in packets] == [f"p2[{field}]" for field in range(10)]
    assert [row["coordinate_column"] for row in packets] == list(range(32, 42))
    assert all(row["seed"]["dP_derivative"] == 2 for row in packets)
    assert all(len(row["delta_H"]) == 10 for row in packets)
    assert all(len(row["delta_G_upper"]) == 10 for row in packets)
    assert all(row["all_20_covariant_tangent_components_materialized"] for row in packets)
    program = checked["coordinate_tangent_program"]
    assert program["exact_spatial_permutation"] == "swap_indices_1_and_2_from_p1"
    assert program["no_flat_reference_specialization"] is True


def test_every_candidate_has_200_live_target_pairs_and_26400_roots() -> None:
    checked = _load()
    manifests = checked["candidate_manifests"]
    assert len(manifests) == 12
    for manifest in manifests:
        assert manifest["derivative_coordinate_columns"] == list(range(32, 42))
        assert manifest["target_direction_pairs"] == 200
        assert manifest["leaf_derivative_roots"] == 26400
        assert len(manifest["direction_packets"]) == 200
        assert manifest["nonzero_leaf_derivative_roots"] > 0
        assert manifest["exact_zero_leaf_derivative_roots"] > 0
        assert (
            manifest["nonzero_leaf_derivative_roots"] + manifest["exact_zero_leaf_derivative_roots"]
            == 26400
        )


def test_leaf_dag_is_closed_over_only_the_44_coordinate_jet_primitives() -> None:
    checked = _load()
    dag = checked["leaf_arithmetic_DAG"]
    assert len(dag["allowed_symbols"]) == 44
    allowed = set(dag["allowed_symbols"])
    assert dag["nodes"][0]["expression"] == "0"
    assert all(set(node["free_symbols"]) <= allowed for node in dag["nodes"])
    tangent_hashes = {row["content_sha256"] for row in checked["coordinate_tangent_packets"]}
    for manifest in checked["candidate_manifests"]:
        assert all(
            row["tangent_packet_sha256"] in tangent_hashes for row in manifest["direction_packets"]
        )


def test_exact_controls_distinguish_p2_and_reject_nonspatial_or_flat_substitution() -> None:
    controls = _load()["exact_controls"]
    assert controls["off_diagonal_seed_normalization"]["exact_value"] == "sqrt(2)/2"
    assert controls["derivative_index_distinction"] == {
        "p2_seed_derivative": 2,
        "replace_by_p1_seed_rejected": True,
    }
    assert controls["spatial_permutation_authority"]["eta_invariant"] is True
    assert controls["spatial_permutation_authority"]["time_space_swap_not_used"] is True
    assert controls["flat_reference_substitution_for_general_claim"]["rejected"] is True


def test_d2_and_remaining_lower_boundaries_stay_fail_closed() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    seals = checked["claim_seals"]
    assert counts["previous_missing_coordinate_columns"] == 28
    assert counts["new_p2_metric_coordinate_columns"] == 10
    assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 18
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["new_ordered_D2_roots_registered_per_candidate"] == 0
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert seals["D2_entry_count_advanced"] is False
    assert seals["complete_D2F"] is False
    assert seals["global_H7"] is False


def test_resealed_nonzero_tangent_index_root_or_d2_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["coordinate_tangent_packets"][0]["seed"].update({"dP_derivative": 1}),
        _zero_first_nonzero_hessian_tangent,
        lambda value: value["candidate_manifests"][0]["direction_packets"][0].update(
            {"zero_default_arithmetic_root": 1}
        ),
        lambda value: value["gate_counts"].update(
            {"registered_D2_entries_per_candidate_after": 5325}
        ),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(P2MetricLowerLeafAuthorityError, match="checked result changed"):
            validate_campaign(corrupted, root=ROOT)
