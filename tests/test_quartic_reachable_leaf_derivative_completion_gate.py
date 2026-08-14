from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_reachable_leaf_derivative_completion_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    SOURCE_PATH,
    TEST_PATH,
    ReachableLeafCompletionError,
    _content_sha,
    _file_sha,
    _legacy_text_sha,
    _production_text_sha,
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


def test_all_31680_obligations_are_partitioned_without_overlap() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    assert counts["reachable_leaf_derivative_obligations"] == 31680
    assert counts["registered_exact_leaf_derivative_roots"] == 31680
    assert counts["P10_leaf_derivative_roots"] == 7920
    assert counts["Pother_leaf_derivative_roots"] == 23760
    assert counts["P10_leaf_derivative_roots"] + counts["Pother_leaf_derivative_roots"] == 31680


def test_candidate_certificates_cover_20_exact_direction_packets() -> None:
    checked = _load()
    certificates = checked["candidate_certificates"]
    assert len(certificates) == 12
    assert len({row["candidate_id"] for row in certificates}) == 12
    for certificate in certificates:
        assert certificate["obligation_count"] == 2640
        assert certificate["direction_packet_count"] == 20
        assert certificate["leaf_root_count"] == 2640
        assert len(certificate["direction_root_manifests"]) == 20
        assert certificate["all_obligations_exactly_matched"] is True


def test_zero_roots_are_dense_manifest_certificates_not_inferences() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    assert counts["nonzero_leaf_derivative_roots"] == 396
    assert counts["exact_zero_leaf_derivative_roots"] == 31284
    assert (
        counts["nonzero_leaf_derivative_roots"] + counts["exact_zero_leaf_derivative_roots"]
        == 31680
    )
    assert checked["claim_seals"]["no_zero_leaf_derivative_inferred"] is True
    for certificate in checked["candidate_certificates"]:
        assert certificate["nonzero_leaf_roots"] == 33
        assert certificate["exact_zero_leaf_roots"] == 2607


def test_all_264_bounded_D2_records_are_exactly_replayed() -> None:
    checked = _load()
    assert checked["gate_counts"]["bounded_ordered_D2_roots_registered"] == 264
    assert checked["gate_counts"]["bounded_ordered_D2_roots_blocked"] == 0
    for certificate in checked["candidate_certificates"]:
        completion = certificate["bounded_D2_completion"]
        assert completion["bounded_ordered_D2_roots"] == 22
        assert completion["P10_roots"] == 7
        assert completion["Pother_roots"] == 15
        assert completion["Pother_exact_zero_roots"] == 2


def test_global_D2_count_is_preserved_at_5324_of_257499() -> None:
    checked = _load()
    counts = checked["gate_counts"]
    assert counts["registered_D2_entries_per_candidate_before"] == 5324
    assert counts["new_D2_entries_per_candidate"] == 0
    assert counts["registered_D2_entries_per_candidate_after"] == 5324
    assert counts["full_D2_entries_per_candidate"] == 257499
    assert counts["remaining_D2_entries_per_candidate"] == 252175
    assert counts["remaining_coordinate_columns_without_A_B_C_leaf_authority"] == 131


def test_scope_seals_block_full_tensor_and_H7_claims() -> None:
    seals = _load()["claim_seals"]
    assert seals["all_31680_reachable_leaf_derivative_roots_registered"] is True
    assert seals["all_264_bounded_ordered_D2_roots_registered"] is True
    assert seals["D2_entry_count_advanced"] is False
    assert seals["remaining_131_coordinate_leaf_families_registered"] is False
    assert seals["complete_D2F"] is False
    assert seals["global_H7"] is False
    assert seals["candidate_theory_rejected"] is False


def test_production_materializer_and_legacy_authority_hash_modes_are_explicit() -> None:
    checked = _load()
    bindings = checked["source_bindings"]
    for role, path in (
        ("source", SOURCE_PATH),
        ("config", CONFIG_PATH),
        ("test", TEST_PATH),
    ):
        assert bindings[role]["production_file_sha256"] == _production_text_sha(ROOT / path)
    evidence = bindings["evidence"]
    for role in ("differentiability", "p10_leaf", "pother_leaf", "p10_replay", "pother_replay"):
        bundle = evidence[role]
        stem = bundle["stem"]
        assert bundle["source_sha256"] == _legacy_text_sha(
            ROOT / f"src/sigma_theory_compiler/{stem}.py"
        )
        assert bundle["config_sha256"] == _legacy_text_sha(
            ROOT / f"configs/backgrounds/{stem}.json"
        )
        assert bundle["test_sha256"] == _legacy_text_sha(ROOT / f"tests/test_{stem}.py")
    coordinate = evidence["coordinate_projection"]
    stem = coordinate["stem"]
    assert coordinate["source_sha256"] == _file_sha(ROOT / f"src/sigma_theory_compiler/{stem}.py")


def test_resealed_leaf_zero_D2_or_scope_tamper_fails_closed() -> None:
    mutations = (
        lambda value: value["gate_counts"].update(
            {"registered_exact_leaf_derivative_roots": 31679}
        ),
        lambda value: value["candidate_certificates"][0].update({"exact_zero_leaf_roots": 2606}),
        lambda value: value["candidate_certificates"][0]["bounded_D2_completion"].update(
            {"bounded_ordered_D2_roots": 21}
        ),
        lambda value: value["claim_seals"].update({"complete_D2F": True}),
    )
    for mutate in mutations:
        corrupted = copy.deepcopy(_load())
        mutate(corrupted)
        _reseal(corrupted)
        with pytest.raises(ReachableLeafCompletionError, match="result changed"):
            validate_campaign(corrupted, root=ROOT)
