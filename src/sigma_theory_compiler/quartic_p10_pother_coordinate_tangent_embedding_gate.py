"""Register exact coordinate-basis tangents for the bound P10/Pother target atoms."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-p10-pother-coordinate-tangent-embedding-config-1.0"
RESULT_SCHEMA = "sigma-quartic-p10-pother-coordinate-tangent-embedding-gate-1.0"
CAMPAIGN_ID = "quartic-p10-pother-coordinate-tangent-embedding-001"
CONFIG_PATH = "configs/backgrounds/quartic_p10_pother_coordinate_tangent_embedding_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_p10_pother_coordinate_tangent_embedding_gate.py"
TEST_PATH = "tests/test_quartic_p10_pother_coordinate_tangent_embedding_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-p10-pother-coordinate-tangent-embedding-gate/campaign.json"
)
FIRST_BLOCKER = (
    "differentiate_the_candidate_bound_D1_arithmetic_DAGs_along_the_264_registered_unit_"
    "tangents_and_seal_exact_ordered_mixed_D2_roots"
)
BASIS_SHA256 = "cdb30c510a24bc6e64bc78245ac6f69d9dfc207e7812fd2d8abeba8e03cb2525"
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/"
            "quartic_ordered_mixed_d2_root_minimal_registration_contract.py"
        ),
        "file_sha256": "0216f6ca9d6ee5b181a4447d59f7ebd1b9feb90f967545bf4a3ae046cc66d8fd",
    },
    "config": {
        "path": (
            "configs/backgrounds/quartic_ordered_mixed_d2_root_minimal_registration_contract.json"
        ),
        "file_sha256": "b9baf6225e7700a062b05a292e882f84d763b206d75daf8ee614a65f113e953e",
    },
    "test": {
        "path": ("tests/test_quartic_ordered_mixed_d2_root_minimal_registration_contract.py"),
        "file_sha256": "13c702bd6567cf046b4c3c47cd576c7264345d1d995b8fb0eedcb575402473fa",
    },
    "artifact": {
        "path": (
            "runs/physics-language/quartic-ordered-mixed-d2-root-minimal-registration-contract/"
            "campaign.json"
        ),
        "file_sha256": "d74c889619e25abe7c6672a391bdc1209d03079ba8b245ccc94e2f4f79aa6365",
        "content_sha256": "90ec881410d7a9dcbf94114b8fa52d73aa4963f2baf8a7430f78b817f1fabc25",
    },
}
EXPECTED_CONTRACT = {
    "basis_dimension": 153,
    "coefficient_field": "Q",
    "candidate_count": 12,
    "coordinate_records_per_candidate": 22,
    "embedding_records": 264,
    "embedding_semantics": (
        "the_tangent_of_coordinate_atom_at_coordinate_column_j_is_the_canonical_unit_vector_"
        "e_j_in_the_bound_153_atom_basis"
    ),
    "P10_rule": "coordinate_atom_field_index_equals_10",
    "Pother_rule": "coordinate_atom_field_index_is_between_0_and_9_inclusive",
}
EXPECTED_POLICIES = {
    "coordinate_tangent_admission": "require_live_atom_column_replay_and_unit_vector_exactness",
    "ordered_mixed_D2_roots": "fail_closed",
    "covariant_component_projection": "fail_closed",
    "complete_ordered_D2F": "fail_closed",
    "full_high_atom_identity": "fail_closed",
    "global_H7": "fail_closed",
    "nonlinear_PDE": "fail_closed",
    "lifespan": "fail_closed",
    "candidate_rejection": "forbidden",
}
EXPECTED_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
ATOM_PATTERN = re.compile(r"^s(?:01|02|03|11|12|13|22|23|33)\[(\d|10)\]$")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise ValueError("coordinate tangent path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "embedding_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("coordinate tangent config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("coordinate tangent artifact file binding changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("coordinate tangent artifact content binding changed")
    return value


def _load_predecessor(root: Path) -> dict[str, Any]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]:
            raise ValueError("coordinate tangent predecessor file binding changed")
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR["artifact"])
    if (
        predecessor.get("decision")
        != "pass_exact_264_slot_ordered_mixed_D2_registration_contract_downstream_blocked"
        or predecessor.get("gate_counts", {}).get("required_ordered_mixed_D2_roots") != 264
        or predecessor.get("gate_counts", {}).get("registered_direction_tangent_embeddings") != 0
        or len(predecessor.get("candidate_manifests", [])) != 12
    ):
        raise ValueError("coordinate tangent predecessor boundary changed")
    return predecessor


def _load_basis_registry(
    root: Path, predecessor: Mapping[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    binding = predecessor.get("source_bindings", {}).get("full_source_D1_artifact")
    if not isinstance(binding, Mapping):
        raise TypeError("coordinate tangent full D1 binding missing")
    source = _load_bound(root, binding)
    manifest = source.get("common_full_entry_manifest")
    certificates = source.get("certificates")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("shape") != [11, 153]
        or manifest.get("total_entry_count") != 1683
        or not isinstance(manifest.get("entries"), list)
        or len(manifest["entries"]) != 1683
        or not isinstance(certificates, list)
        or len(certificates) != 12
    ):
        raise ValueError("coordinate tangent full D1 registry changed")
    by_column: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in manifest["entries"]:
        by_column[int(row["coordinate_column"])].append(row)
    atoms = []
    for column in range(153):
        rows = by_column[column]
        atom_names = {str(row["coordinate_atom"]) for row in rows}
        source_rows = {int(row["source_row"]) for row in rows}
        if len(rows) != 11 or len(atom_names) != 1 or source_rows != set(range(11)):
            raise ValueError("coordinate tangent atom-column registry changed")
        atoms.append(atom_names.pop())
    basis_hashes = {
        row.get("provenance", {}).get("coordinate_atom_basis_sha256") for row in certificates
    }
    if _sha(atoms) != BASIS_SHA256 or basis_hashes != {BASIS_SHA256}:
        raise ValueError("coordinate tangent basis hash changed")
    return atoms, dict(binding)


def _field_index(atom: str) -> int:
    match = ATOM_PATTERN.fullmatch(atom)
    if match is None:
        raise ValueError("coordinate tangent target atom is not typed principal metric atom")
    return int(match.group(1))


def _embedding_record(
    obligation: Mapping[str, Any], atoms: list[str], basis_sha256: str
) -> dict[str, Any]:
    column = int(obligation["coordinate_column"])
    atom = str(obligation["coordinate_atom"])
    direction = str(obligation["direction_label"])
    if not 0 <= column < 153 or atoms[column] != atom:
        raise ValueError("coordinate tangent obligation atom-column mismatch")
    field_index = _field_index(atom)
    expected_direction = "P10" if field_index == 10 else "Pother"
    if direction != expected_direction:
        raise ValueError("coordinate tangent direction class mismatch")
    identity = {
        "candidate_id": obligation["candidate_id"],
        "coordinate_ordinal": obligation["coordinate_ordinal"],
        "obligation_id": obligation["obligation_id"],
        "coordinate_atom": atom,
        "coordinate_column": column,
        "direction_label": direction,
        "coordinate_atom_basis_sha256": basis_sha256,
    }
    sparse = [{"coordinate_column": column, "coefficient": "1"}]
    return {
        **identity,
        "embedding_id": _sha({**identity, "sparse_entries": sparse}),
        "basis_dimension": 153,
        "coefficient_field": "Q",
        "sparse_entries": sparse,
        "support_size": 1,
        "exact_squared_norm": "1",
        "canonical_coordinate_unit_tangent": True,
        "candidate_bound": True,
        "ordered_D2_arithmetic_root_registered": False,
        "candidate_rejection_authorized": False,
    }


def _embedding_manifests(predecessor: Mapping[str, Any], atoms: list[str]) -> list[dict[str, Any]]:
    manifests = []
    all_ids: set[str] = set()
    for candidate in predecessor["candidate_manifests"]:
        records = [_embedding_record(row, atoms, BASIS_SHA256) for row in candidate["obligations"]]
        ids = {row["embedding_id"] for row in records}
        if len(records) != 22 or len(ids) != 22 or ids & all_ids:
            raise AssertionError("coordinate tangent embedding identity collision")
        all_ids |= ids
        manifests.append(
            {
                "candidate_id": candidate["candidate_id"],
                "embedding_records": records,
                "registered_embeddings": 22,
                "P10_embedding_records": sum(row["direction_label"] == "P10" for row in records),
                "Pother_embedding_records": sum(
                    row["direction_label"] == "Pother" for row in records
                ),
                "unique_coordinate_unit_vectors": len(
                    {row["coordinate_column"] for row in records}
                ),
                "embedding_manifest_sha256": _sha(records),
                "candidate_decision": "pass_coordinate_tangents_downstream_blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    if len(all_ids) != 264:
        raise AssertionError("coordinate tangent candidate manifest incomplete")
    return manifests


def _expected_body(root: Path, config_path: Path, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    atoms, full_source_binding = _load_basis_registry(root, predecessor)
    manifests = _embedding_manifests(predecessor, atoms)
    records = [row for manifest in manifests for row in manifest["embedding_records"]]
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_all_264_candidate_bound_coordinate_unit_tangents_registered_D2_roots_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "embedding_theorem": {
            "name": "canonical_coordinate_atom_unit_tangent_embedding",
            "premises": (
                "The live full-D1 registry bijectively orders 153 coordinate atoms by columns "
                "0 through 152 and seals that ordered list. Each target obligation supplies an "
                "atom and its matching column; its P10/Pother label equals the field-index class."
            ),
            "exact_result": (
                "Every one of the 264 candidate-coordinate obligations has the exact sparse "
                "coordinate tangent e_j with sole coefficient one at its registered column. "
                "There are 20 unique unit vectors per candidate, represented by 7 P10 records "
                "and 15 Pother records because two P10 atoms occur twice."
            ),
            "boundary": (
                "This registers coordinate-chart tangent embeddings only. It does not identify "
                "a generic-G4 tensor-component projection, differentiate any arithmetic DAG, "
                "supply a mixed-D2 value, or prove a covariant physical map."
            ),
        },
        "coordinate_basis_registry": {
            "dimension": 153,
            "coordinate_atom_basis": atoms,
            "coordinate_atom_basis_sha256": BASIS_SHA256,
            "column_bijection_proved": True,
            "source_rows_per_column": 11,
        },
        "candidate_manifests": manifests,
        "embedding_manifest_sha256": _sha(records),
        "gate_counts": {
            "selected_candidates": 12,
            "embedding_records": 264,
            "registered_coordinate_tangent_embeddings": 264,
            "P10_embedding_records": 84,
            "Pother_embedding_records": 180,
            "unique_coordinate_unit_vectors_per_candidate": 20,
            "unique_coordinate_unit_vectors_global": 20,
            "registered_generic_term_component_projections": 0,
            "registered_ordered_mixed_D2_roots": 0,
            "blocked_ordered_mixed_D2_roots": 264,
            "corrected_second_source_jet_entries": 0,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "all_264_coordinate_unit_tangents_registered": True,
            "atom_column_bijection_replayed": True,
            "P10_Pother_field_index_classification_replayed": True,
            "generic_term_component_projection_registered": False,
            "ordered_mixed_D2_values_registered": False,
            "physical_covariant_component_map_proved": False,
            "physical_covariant_component_map_no_go_proved": False,
            "corrected_second_source_jet_registered": False,
            "complete_ordered_D2F_tensor_registered": False,
            "full_high_atom_good_unknown_identity_proved": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "accept_atom_column_mismatch": {"rejected": True},
            "accept_direction_field_class_mismatch": {"rejected": True},
            "accept_nonunit_or_multisupport_embedding": {"rejected": True},
            "promote_coordinate_tangent_to_covariant_component_map": {"rejected": True},
            "promote_tangent_embedding_to_D2_arithmetic_root": {"rejected": True},
            "reject_candidate_for_missing_D2_root": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(EXPECTED_PREDECESSOR),
            "full_source_D1_artifact": full_source_binding,
        },
        "scope": (
            "candidate-bound canonical coordinate-unit tangent embeddings for the current 264 "
            "P10/Pother obligations in the sealed 153-atom chart; no generic-term component "
            "projection, D2 root, corrected jet, covariant physical map or no-go, full D2F, "
            "high-atom identity, H7, PDE, lifespan, rejection, or observation"
        ),
    }


def _validate_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "full_source_D1_artifact",
    }:
        raise ValueError("coordinate tangent source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("coordinate tangent local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("coordinate tangent predecessor binding changed")
    predecessor = _load_predecessor(root)
    if (
        bindings["full_source_D1_artifact"]
        != predecessor["source_bindings"]["full_source_D1_artifact"]
    ):
        raise ValueError("coordinate tangent full D1 binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("coordinate tangent content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("coordinate tangent result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(root)
    body = _expected_body(root, config_path, predecessor)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def write_gate(config_path: Path) -> Path:
    result = build_gate(config_path)
    root = config_path.resolve().parents[2]
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    print(write_gate(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
