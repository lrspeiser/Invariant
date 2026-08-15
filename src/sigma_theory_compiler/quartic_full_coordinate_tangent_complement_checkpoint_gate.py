"""Checkpoint the exact coordinate-tangent complement without covariant overclaim."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-full-coordinate-tangent-complement-checkpoint-config-1.0"
RESULT_SCHEMA = "sigma-quartic-full-coordinate-tangent-complement-checkpoint-gate-1.0"
CAMPAIGN_ID = "quartic-full-coordinate-tangent-complement-checkpoint-001"
CONFIG_PATH = "configs/backgrounds/quartic_full_coordinate_tangent_complement_checkpoint_gate.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_full_coordinate_tangent_complement_checkpoint_gate.py"
)
TEST_PATH = "tests/test_quartic_full_coordinate_tangent_complement_checkpoint_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-full-coordinate-tangent-complement-checkpoint-gate/campaign.json"
)
CURRENT_D2 = {
    "path": (
        "runs/physics-language/quartic-registered-direction-cross-leaf-d2-replay-gate/campaign.json"
    ),
    "file_sha256": "f38f3abc8b9e07d8e4578fd8e3df528a8a1482ecf9480a8ef53752ed8eb5f17f",
    "content_sha256": "65e2235c7e12b39fdd06b69ba7ff4f9de793f0f4f139fb06f629ffdc4aac75a6",
}
TANGENT_AUTHORITY = {
    "path": (
        "runs/physics-language/quartic-p10-pother-coordinate-tangent-embedding-gate/campaign.json"
    ),
    "file_sha256": "d393050c41d58308a28a29a5b72b9da6f5ea0797b30ba97b885c3da8ca20efaa",
    "content_sha256": "fb4a55d74a8bfbe1009f13373883b5553c778441a04af143ad031f45de50e271",
}
CHECKPOINT_CONTRACT = {
    "checkpoint_size": 17,
    "expected_checkpoints_per_candidate": 8,
    "expected_new_coordinate_tangents_per_candidate": 133,
}
POLICIES = {
    "book_coordinate_tangent_as_covariant_projection": "forbidden",
    "book_tangent_certificate_as_D2_entry": "forbidden",
    "candidate_rejection": "forbidden",
    "complete_D2F": "fail_closed",
    "physical_covariant_projection": "fail_closed",
}
SEALS = {
    "GPU_execution_used": False,
    "live_SQLite_opened": False,
    "observations_opened": False,
}
FIRST_BLOCKER = (
    "register_a_physical_coordinate_to_covariant_component_projection_and_reconcile_the_"
    "22_formal_slots_to_20_unique_coordinate_vectors_before_advancing_D2_entry_counts"
)


class CoordinateTangentComplementError(ValueError):
    """Raised when an exact binding, checkpoint, or claim changes."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise CoordinateTangentComplementError("coordinate complement path escapes root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise CoordinateTangentComplementError("coordinate complement file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise CoordinateTangentComplementError("coordinate complement content binding changed")
    return value


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "current_d2_authority": CURRENT_D2,
        "tangent_authority": TANGENT_AUTHORITY,
        "checkpoint_contract": CHECKPOINT_CONTRACT,
        "policies": POLICIES,
        "seals": SEALS,
    }
    if value != expected:
        raise CoordinateTangentComplementError("coordinate complement config boundary changed")


def _load_authorities(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    current = _load_bound(root, CURRENT_D2)
    tangent = _load_bound(root, TANGENT_AUTHORITY)
    if (
        current.get("gate_counts", {}).get("selected_candidates") != 12
        or current.get("gate_counts", {}).get("registered_per_candidate") != 5324
        or current.get("gate_counts", {}).get("remaining_per_candidate") != 252175
        or current.get("gate_counts", {}).get("full_entries_per_candidate") != 257499
        or "131_unregistered_derivative_directions" not in current.get("first_blocker", "")
    ):
        raise CoordinateTangentComplementError("current D2 authority boundary changed")
    registry = tangent.get("coordinate_basis_registry", {})
    if (
        tangent.get("gate_counts", {}).get("selected_candidates") != 12
        or tangent.get("gate_counts", {}).get("embedding_records") != 264
        or tangent.get("gate_counts", {}).get("unique_coordinate_unit_vectors_per_candidate") != 20
        or registry.get("dimension") != 153
        or len(registry.get("coordinate_atom_basis", [])) != 153
        or registry.get("coordinate_atom_basis_sha256")
        != _sha(registry.get("coordinate_atom_basis"))
    ):
        raise CoordinateTangentComplementError("tangent authority boundary changed")
    current_ids = {row["candidate_id"] for row in current["candidate_manifests"]}
    tangent_ids = {row["candidate_id"] for row in tangent["candidate_manifests"]}
    if len(current_ids) != 12 or current_ids != tangent_ids:
        raise CoordinateTangentComplementError("candidate authority identities changed")
    return current, tangent


def _certificate(
    candidate_id: str,
    column: int,
    atom: str,
    basis_sha256: str,
) -> dict[str, Any]:
    body = {
        "basis_dimension": 153,
        "candidate_id": candidate_id,
        "coefficient_field": "Q",
        "coordinate_atom": atom,
        "coordinate_atom_basis_sha256": basis_sha256,
        "coordinate_column": column,
        "exact_squared_norm": "1",
        "physical_covariant_component_projection_registered": False,
        "sparse_entries": [{"coefficient": "1", "coordinate_column": column}],
        "support_size": 1,
        "theorem": "canonical_coordinate_chart_unit_tangent_e_j",
    }
    return {**body, "content_sha256": _sha(body)}


def _checkpoint(
    candidate_id: str,
    checkpoint_index: int,
    records: Sequence[Mapping[str, Any]],
    predecessor_sha256: str | None,
) -> dict[str, Any]:
    body = {
        "candidate_id": candidate_id,
        "checkpoint_index": checkpoint_index,
        "checkpoint_predecessor_sha256": predecessor_sha256,
        "complete": True,
        "first_coordinate_column": records[0]["coordinate_column"],
        "last_coordinate_column": records[-1]["coordinate_column"],
        "record_count": len(records),
        "record_root_sha256": _sha([row["content_sha256"] for row in records]),
        "records": [_copy(row) for row in records],
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_manifest(
    candidate: Mapping[str, Any],
    basis: Sequence[str],
    basis_sha256: str,
) -> dict[str, Any]:
    groups: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in candidate["embedding_records"]:
        groups[int(record["coordinate_column"])].append(record)
    existing_columns = sorted(groups)
    if len(candidate["embedding_records"]) != 22 or len(existing_columns) != 20:
        raise CoordinateTangentComplementError("existing direction alias census changed")
    duplicate_groups = [
        {
            "coordinate_atom": basis[column],
            "coordinate_column": column,
            "formal_coordinate_ordinals": sorted(int(row["coordinate_ordinal"]) for row in records),
            "formal_record_count": len(records),
        }
        for column, records in sorted(groups.items())
        if len(records) > 1
    ]
    duplicate_excess = sum(row["formal_record_count"] - 1 for row in duplicate_groups)
    if duplicate_excess != 2:
        raise CoordinateTangentComplementError("22-slot to 20-vector alias obstruction changed")
    complement_columns = [column for column in range(153) if column not in groups]
    if len(complement_columns) != 133:
        raise CoordinateTangentComplementError("coordinate tangent complement is not 133")
    records = [
        _certificate(
            str(candidate["candidate_id"]),
            column,
            str(basis[column]),
            basis_sha256,
        )
        for column in complement_columns
    ]
    checkpoints = []
    predecessor = None
    size = CHECKPOINT_CONTRACT["checkpoint_size"]
    for index, offset in enumerate(range(0, len(records), size)):
        checkpoint = _checkpoint(
            str(candidate["candidate_id"]),
            index,
            records[offset : offset + size],
            predecessor,
        )
        checkpoints.append(checkpoint)
        predecessor = checkpoint["content_sha256"]
    if len(checkpoints) != CHECKPOINT_CONTRACT["expected_checkpoints_per_candidate"]:
        raise CoordinateTangentComplementError("checkpoint count changed")
    obstruction_body = {
        "actual_missing_unique_coordinate_vectors_before_extension": 133,
        "basis_dimension": 153,
        "duplicate_formal_slot_excess": duplicate_excess,
        "duplicate_groups": duplicate_groups,
        "formal_registered_slots": 22,
        "formal_typed_partition_unregistered_count": 131,
        "partition_compatible_with_unique_coordinate_complement": False,
        "unique_registered_coordinate_vectors": 20,
    }
    obstruction = {**obstruction_body, "content_sha256": _sha(obstruction_body)}
    body = {
        "alias_obstruction_certificate": obstruction,
        "candidate_id": candidate["candidate_id"],
        "checkpoint_count": len(checkpoints),
        "checkpoint_head_sha256": checkpoints[-1]["content_sha256"],
        "checkpoints": checkpoints,
        "coordinate_basis_complete_after_extension": True,
        "existing_formal_embedding_records": 22,
        "existing_unique_coordinate_vectors": 20,
        "new_coordinate_tangent_certificates": 133,
        "new_coordinate_tangent_root_sha256": _sha([row["content_sha256"] for row in records]),
        "physical_covariant_component_projections_registered": 0,
        "unique_coordinate_vectors_after_extension": 153,
    }
    return {**body, "content_sha256": _sha(body)}


def _expected_body(
    root: Path,
    config_path: Path,
    current: Mapping[str, Any],
    tangent: Mapping[str, Any],
) -> dict[str, Any]:
    registry = tangent["coordinate_basis_registry"]
    basis = registry["coordinate_atom_basis"]
    basis_sha256 = registry["coordinate_atom_basis_sha256"]
    manifests = [
        _candidate_manifest(candidate, basis, basis_sha256)
        for candidate in tangent["candidate_manifests"]
    ]
    checkpoint_count = sum(row["checkpoint_count"] for row in manifests)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "pass_all_1596_missing_coordinate_unit_tangents_checkpointed_"
            "covariant_D2_advance_blocked"
        ),
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "coordinate_complement_theorem": {
            "name": "full_coordinate_basis_unit_tangent_complement",
            "exact_result": (
                "The sealed basis has 153 unique coordinate atoms. The 22 prior formal records "
                "represent only 20 unique coordinate columns, so their exact complement contains "
                "133—not 131—canonical unit tangents. All 133 are now independently certified "
                "and checkpointed for each of 12 candidates."
            ),
            "boundary": (
                "A coordinate-chart unit vector is not by itself a physical coordinate-to-"
                "covariant component projection. No new D2 arithmetic root is booked. The "
                "current 5,324/257,499 formal-slot authority therefore remains unchanged."
            ),
        },
        "basis_registry": {
            "coordinate_atom_basis": _copy(basis),
            "coordinate_atom_basis_sha256": basis_sha256,
            "dimension": 153,
        },
        "candidate_manifests": manifests,
        "candidate_manifest_root_sha256": _sha([row["content_sha256"] for row in manifests]),
        "checkpoint_contract": _copy(CHECKPOINT_CONTRACT),
        "gate_counts": {
            "selected_candidates": 12,
            "existing_formal_direction_records_per_candidate": 22,
            "existing_unique_coordinate_vectors_per_candidate": 20,
            "formal_partition_claimed_unregistered_directions_per_candidate": 131,
            "actual_missing_unique_coordinate_vectors_per_candidate_before_extension": 133,
            "new_coordinate_tangent_certificates_per_candidate": 133,
            "new_coordinate_tangent_certificates_all_candidates": 1596,
            "unique_coordinate_vectors_per_candidate_after_extension": 153,
            "checkpoint_receipts": checkpoint_count,
            "physical_covariant_component_projections_registered": 0,
            "new_D2_entries_registered_per_candidate": 0,
            "registered_D2_entries_per_candidate": 5324,
            "remaining_formal_D2_entries_per_candidate": 252175,
            "full_D2_entries_per_candidate": 257499,
            "complete_D2F_tensors": 0,
        },
        "claim_seals": {
            "all_153_unique_coordinate_unit_tangents_registered_per_candidate": True,
            "all_133_missing_unique_coordinate_tangents_checkpointed": True,
            "22_formal_slots_are_22_unique_coordinate_vectors": False,
            "coordinate_tangent_is_covariant_projection": False,
            "physical_covariant_component_projection_registered": False,
            "new_D2_entries_registered": False,
            "complete_D2F": False,
            "candidate_rejected": False,
        },
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "current_d2_authority": _copy(CURRENT_D2),
            "tangent_authority": _copy(TANGENT_AUTHORITY),
        },
        "data_seals": _copy(SEALS),
        "scope": (
            "candidate-bound completion of the 153-dimensional coordinate-chart unit-tangent "
            "basis with an exact alias obstruction; no physical covariant projection, D2-entry "
            "advance, full D2F, high-atom, H7, PDE, observation, or rejection claim"
        ),
    }
    if current["gate_counts"]["registered_per_candidate"] != 5324:
        raise CoordinateTangentComplementError("current D2 count changed during build")
    return body


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    current, tangent = _load_authorities(root)
    body = _expected_body(root, config_path, current, tangent)
    return {**body, "content_sha256": _sha(body)}


def _validate_result(value: Mapping[str, Any], *, root: Path) -> None:
    if value.get("content_sha256") != _content_sha(value):
        raise CoordinateTangentComplementError("coordinate complement result seal changed")
    expected = build_gate(_inside(root, CONFIG_PATH))
    if dict(value) != expected:
        raise CoordinateTangentComplementError("coordinate complement exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = _canonical(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise CoordinateTangentComplementError("refusing to replace checkpoint authority")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    root = args.config.resolve().parents[2]
    output = _inside(root, OUTPUT_PATH)
    if args.validate_checked:
        _validate_result(json.loads(output.read_text(encoding="utf-8")), root=root)
        return 0
    value = build_gate(args.config)
    _write_immutable(output, value)
    _validate_result(value, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
