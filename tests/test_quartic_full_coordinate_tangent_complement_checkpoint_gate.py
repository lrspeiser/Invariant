from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_full_coordinate_tangent_complement_checkpoint_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    CoordinateTangentComplementError,
    _content_sha,
    _sha,
    _validate_config,
    _validate_result,
    build_gate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def checked() -> dict:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    rebuilt = build_gate(ROOT / CONFIG_PATH)
    assert value == rebuilt
    _validate_result(value, root=ROOT)
    return value


def _reseal(value: dict) -> None:
    value["content_sha256"] = _content_sha(value)


def test_all_missing_unique_coordinate_tangents_are_constructed(checked: dict) -> None:
    counts = checked["gate_counts"]
    assert counts["selected_candidates"] == 12
    assert counts["existing_formal_direction_records_per_candidate"] == 22
    assert counts["existing_unique_coordinate_vectors_per_candidate"] == 20
    assert counts["actual_missing_unique_coordinate_vectors_per_candidate_before_extension"] == 133
    assert counts["new_coordinate_tangent_certificates_per_candidate"] == 133
    assert counts["new_coordinate_tangent_certificates_all_candidates"] == 1596
    assert counts["unique_coordinate_vectors_per_candidate_after_extension"] == 153


def test_every_candidate_has_a_complete_exact_checkpoint_chain(checked: dict) -> None:
    basis = checked["basis_registry"]["coordinate_atom_basis"]
    basis_sha = checked["basis_registry"]["coordinate_atom_basis_sha256"]
    assert len(basis) == 153
    assert _sha(basis) == basis_sha
    all_certificate_ids = set()
    for manifest in checked["candidate_manifests"]:
        assert manifest["checkpoint_count"] == 8
        assert len(manifest["checkpoints"]) == 8
        assert sum(row["record_count"] for row in manifest["checkpoints"]) == 133
        predecessor = None
        candidate_columns = set()
        for index, checkpoint in enumerate(manifest["checkpoints"]):
            assert checkpoint["checkpoint_index"] == index
            assert checkpoint["checkpoint_predecessor_sha256"] == predecessor
            assert checkpoint["content_sha256"] == _content_sha(checkpoint)
            assert checkpoint["record_root_sha256"] == _sha(
                [row["content_sha256"] for row in checkpoint["records"]]
            )
            predecessor = checkpoint["content_sha256"]
            for record in checkpoint["records"]:
                assert record["content_sha256"] == _content_sha(record)
                assert record["candidate_id"] == manifest["candidate_id"]
                assert record["coordinate_atom_basis_sha256"] == basis_sha
                assert record["sparse_entries"] == [
                    {
                        "coefficient": "1",
                        "coordinate_column": record["coordinate_column"],
                    }
                ]
                assert record["physical_covariant_component_projection_registered"] is False
                assert record["content_sha256"] not in all_certificate_ids
                all_certificate_ids.add(record["content_sha256"])
                candidate_columns.add(record["coordinate_column"])
        assert predecessor == manifest["checkpoint_head_sha256"]
        assert len(candidate_columns) == 133
        assert manifest["coordinate_basis_complete_after_extension"] is True
    assert len(all_certificate_ids) == 1596


def test_alias_obstruction_explains_133_not_131(checked: dict) -> None:
    for manifest in checked["candidate_manifests"]:
        obstruction = manifest["alias_obstruction_certificate"]
        assert obstruction["content_sha256"] == _content_sha(obstruction)
        assert obstruction["basis_dimension"] == 153
        assert obstruction["formal_registered_slots"] == 22
        assert obstruction["unique_registered_coordinate_vectors"] == 20
        assert obstruction["duplicate_formal_slot_excess"] == 2
        assert obstruction["formal_typed_partition_unregistered_count"] == 131
        assert obstruction["actual_missing_unique_coordinate_vectors_before_extension"] == 133
        assert obstruction["partition_compatible_with_unique_coordinate_complement"] is False
        assert sum(row["formal_record_count"] - 1 for row in obstruction["duplicate_groups"]) == 2


def test_no_coordinate_certificate_is_misbooked_as_covariant_D2(checked: dict) -> None:
    counts = checked["gate_counts"]
    assert counts["physical_covariant_component_projections_registered"] == 0
    assert counts["new_D2_entries_registered_per_candidate"] == 0
    assert counts["registered_D2_entries_per_candidate"] == 5324
    assert counts["remaining_formal_D2_entries_per_candidate"] == 252175
    assert counts["full_D2_entries_per_candidate"] == 257499
    assert counts["complete_D2F_tensors"] == 0
    claims = checked["claim_seals"]
    assert claims["all_153_unique_coordinate_unit_tangents_registered_per_candidate"] is True
    assert claims["coordinate_tangent_is_covariant_projection"] is False
    assert claims["physical_covariant_component_projection_registered"] is False
    assert claims["new_D2_entries_registered"] is False
    assert claims["complete_D2F"] is False


def test_checkpoint_and_candidate_roots_are_exact(checked: dict) -> None:
    for manifest in checked["candidate_manifests"]:
        records = [
            record for checkpoint in manifest["checkpoints"] for record in checkpoint["records"]
        ]
        assert manifest["new_coordinate_tangent_root_sha256"] == _sha(
            [row["content_sha256"] for row in records]
        )
        assert manifest["content_sha256"] == _content_sha(manifest)
    assert checked["candidate_manifest_root_sha256"] == _sha(
        [row["content_sha256"] for row in checked["candidate_manifests"]]
    )


def test_first_blocker_is_now_covariant_projection_and_alias_reconciliation(
    checked: dict,
) -> None:
    assert "physical_coordinate_to_covariant_component_projection" in checked["first_blocker"]
    assert "22_formal_slots_to_20_unique_coordinate_vectors" in checked["first_blocker"]
    assert checked["downstream_admission_counts"] == {"pass": 0, "blocked": 12, "reject": 0}


def test_replay_is_deterministic(checked: dict) -> None:
    assert build_gate(ROOT / CONFIG_PATH) == checked
    assert build_gate(ROOT / CONFIG_PATH) == build_gate(ROOT / CONFIG_PATH)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["gate_counts"].__setitem__(
            "new_D2_entries_registered_per_candidate", 1
        ),
        lambda value: value["claim_seals"].__setitem__(
            "physical_covariant_component_projection_registered", True
        ),
        lambda value: value["candidate_manifests"][0]["alias_obstruction_certificate"].__setitem__(
            "duplicate_formal_slot_excess", 0
        ),
        lambda value: value["candidate_manifests"][0]["checkpoints"][0]["records"][0].__setitem__(
            "coordinate_column", -1
        ),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_resealed_tamper_fails_exact_replay(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(CoordinateTangentComplementError):
        _validate_result(tampered, root=ROOT)


def test_config_tamper_fails_closed() -> None:
    value = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    value["checkpoint_contract"]["checkpoint_size"] = 18
    with pytest.raises(CoordinateTangentComplementError, match="config boundary changed"):
        _validate_config(value)
