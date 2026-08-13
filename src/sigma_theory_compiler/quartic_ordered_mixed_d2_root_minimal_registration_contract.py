"""Seal the exact candidate-bound registration slots for 22 ordered mixed D2 roots."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-quartic-ordered-mixed-d2-root-minimal-registration-contract-config-1.0"
RESULT_SCHEMA = "sigma-quartic-ordered-mixed-d2-root-minimal-registration-contract-gate-1.0"
CAMPAIGN_ID = "quartic-ordered-mixed-d2-root-minimal-registration-contract-001"
CONFIG_PATH = "configs/backgrounds/quartic_ordered_mixed_d2_root_minimal_registration_contract.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_ordered_mixed_d2_root_minimal_registration_contract.py"
)
TEST_PATH = "tests/test_quartic_ordered_mixed_d2_root_minimal_registration_contract.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-ordered-mixed-d2-root-minimal-registration-contract/"
    "campaign.json"
)
FIRST_BLOCKER = (
    "register_candidate_bound_P10_Pother_tangent_embeddings_then_materialize_and_replay_"
    "the_264_ordered_mixed_D2_arithmetic_roots"
)
EXPECTED_PREDECESSOR = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/"
            "quartic_fitted_output_connection_component_map_schema_ambiguity_gate.py"
        ),
        "file_sha256": "599ce6c6069bf79b6e5bab1c5f0f3504c8c6775eb2cce0c5d76918acd56a36b1",
    },
    "config": {
        "path": (
            "configs/backgrounds/"
            "quartic_fitted_output_connection_component_map_schema_ambiguity_gate.json"
        ),
        "file_sha256": "63a914832a26029279e806b470609e3a76857fe55382ee53c90f5ea46c6605e8",
    },
    "test": {
        "path": (
            "tests/test_quartic_fitted_output_connection_component_map_schema_ambiguity_gate.py"
        ),
        "file_sha256": "05c6d4ce0c36bcd8689744407f54c6c912f518fcf4e61c123c96a27bb1b8b794",
    },
    "artifact": {
        "path": (
            "runs/physics-language/"
            "quartic-fitted-output-connection-component-map-schema-ambiguity-gate/"
            "campaign.json"
        ),
        "file_sha256": "0256f64acb53f38c0cada5e43a58c974b7f9bebe2529bdf7c3f08e65b9d2563f",
        "content_sha256": "3a3da9ecef30e596ae18cb8e76687338a9fe1bf8e7284ee009287420ce5613ec",
    },
}
REQUIRED_FIELDS = [
    "candidate_id",
    "coordinate_ordinal",
    "source_row",
    "coordinate_atom",
    "coordinate_column",
    "direction_label",
    "direction_tangent_basis_sha256",
    "direction_tangent_sparse_coefficients",
    "ordered_D2_arithmetic_root",
    "ordered_D2_arithmetic_dag_sha256",
    "generic_term_component_projection_rule_id",
    "output_bundle_projection_rule_id",
]
EXPECTED_CONTRACT = {
    "candidate_count": 12,
    "coordinate_obligations_per_candidate": 22,
    "obligation_count": 264,
    "coefficient_field": "Q_sqrt2",
    "ordered_derivative_semantics": (
        "differentiate_the_registered_D1_source_row_coordinate_atom_entry_along_the_"
        "registered_P10_or_Pother_153_basis_tangent"
    ),
    "unique_obligation_key": ["candidate_id", "coordinate_ordinal"],
    "required_registration_fields": REQUIRED_FIELDS,
}
EXPECTED_POLICIES = {
    "cross_registry_component_map": "require_live_candidate_bound_tensor_equivariant_registration",
    "ordered_mixed_D2_admission": (
        "require_live_tangent_embedding_and_exact_arithmetic_root_replay"
    ),
    "corrected_second_source_jet": "fail_closed",
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
        raise ValueError("ordered mixed D2 contract path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "registration_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("ordered mixed D2 contract config boundary changed")


def _load_predecessor(root: Path) -> dict[str, Any]:
    for label in ("source", "config", "test"):
        binding = EXPECTED_PREDECESSOR[label]
        if (
            set(binding) != {"path", "file_sha256"}
            or _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]
        ):
            raise ValueError("ordered mixed D2 predecessor file binding changed")
    binding = EXPECTED_PREDECESSOR["artifact"]
    path = _inside(root, binding["path"])
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("ordered mixed D2 predecessor artifact file changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("ordered mixed D2 predecessor artifact content changed")
    if (
        value.get("decision") != "pass_constructive_component_map_and_mixed_D2_schema_ambiguity"
        or value.get("decision_counts") != {"pass": 12, "blocked": 0, "reject": 0}
        or value.get("downstream_admission_counts") != {"pass": 0, "blocked": 12, "reject": 0}
    ):
        raise ValueError("ordered mixed D2 predecessor decision changed")
    return value


def _load_source_candidates(root: Path, predecessor: Mapping[str, Any]) -> list[str]:
    binding = (
        predecessor.get("source_bindings", {})
        .get("direct_evidence", {})
        .get("full_source_D1", {})
        .get("artifact")
    )
    if not isinstance(binding, Mapping) or set(binding) != {
        "path",
        "file_sha256",
        "content_sha256",
    }:
        raise TypeError("ordered mixed D2 full D1 binding missing")
    path = _inside(root, str(binding["path"]))
    if _file_sha(path) != binding["file_sha256"]:
        raise ValueError("ordered mixed D2 full D1 artifact file changed")
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("content_sha256") != binding["content_sha256"] or source.get(
        "content_sha256"
    ) != _content_sha(source):
        raise ValueError("ordered mixed D2 full D1 artifact content changed")
    certificates = source.get("certificates")
    if not isinstance(certificates, list) or len(certificates) != 12:
        raise ValueError("ordered mixed D2 candidate inventory changed")
    candidate_ids = sorted(str(row["candidate_id"]) for row in certificates)
    if len(set(candidate_ids)) != 12 or any(
        row.get("full_11x153_source_Jacobian_entrywise_materialized") is not True
        or row.get("full_component_Frechet_tensors_orders_2_to_4_complete") is not False
        for row in certificates
    ):
        raise ValueError("ordered mixed D2 full D1 candidate boundary changed")
    return candidate_ids


def _coordinate_templates(predecessor: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = predecessor.get("coordinate_records")
    if not isinstance(records, list) or len(records) != 22:
        raise ValueError("ordered mixed D2 coordinate inventory changed")
    templates = []
    for expected_ordinal, row in enumerate(records):
        if (
            row.get("coordinate_ordinal") != expected_ordinal
            or row.get("direction") not in {"P10", "Pother"}
            or row.get("direction_state_tangent_registered") is not False
            or row.get("ordered_mixed_D2F_root_registered") is not False
            or not row.get("D1_arithmetic_root")
            or not row.get("D1_arithmetic_dag_sha256")
        ):
            raise ValueError("ordered mixed D2 coordinate record changed")
        templates.append(
            {
                "coordinate_ordinal": expected_ordinal,
                "source_row": int(row["output_row"]),
                "coordinate_atom": str(row["atom"]),
                "coordinate_column": int(row["D1_manifest_coordinate_column"]),
                "direction_label": str(row["direction"]),
                "input_row_label": int(row["input_row"]),
                "target_beta": str(row["beta"]),
                "registered_D1_arithmetic_root": str(row["D1_arithmetic_root"]),
                "registered_D1_arithmetic_dag_sha256": str(row["D1_arithmetic_dag_sha256"]),
            }
        )
    if len({(row["source_row"], row["coordinate_atom"]) for row in templates}) != 20:
        raise ValueError("ordered mixed D2 unique D1 entry count changed")
    return templates


def _obligation_record(candidate_id: str, template: Mapping[str, Any]) -> dict[str, Any]:
    identity = {
        "candidate_id": candidate_id,
        "coordinate_ordinal": template["coordinate_ordinal"],
        "source_row": template["source_row"],
        "coordinate_atom": template["coordinate_atom"],
        "coordinate_column": template["coordinate_column"],
        "direction_label": template["direction_label"],
        "registered_D1_arithmetic_root": template["registered_D1_arithmetic_root"],
        "registered_D1_arithmetic_dag_sha256": template["registered_D1_arithmetic_dag_sha256"],
    }
    return {
        **identity,
        "obligation_id": _sha(identity),
        "input_row_label": template["input_row_label"],
        "target_beta": template["target_beta"],
        "ordered_derivative_semantics": EXPECTED_CONTRACT["ordered_derivative_semantics"],
        "direction_tangent_basis_sha256_registered": False,
        "direction_tangent_sparse_coefficients_registered": False,
        "generic_term_component_projection_rule_id_registered": False,
        "output_bundle_projection_rule_id_registered": False,
        "ordered_D2_arithmetic_root_registered": False,
        "ordered_D2_arithmetic_dag_sha256_registered": False,
        "obligation_status": "required_unregistered",
        "candidate_rejection_authorized": False,
    }


def _candidate_manifests(
    candidate_ids: list[str], templates: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    manifests = []
    all_ids: set[str] = set()
    for candidate_id in candidate_ids:
        obligations = [_obligation_record(candidate_id, row) for row in templates]
        ids = {row["obligation_id"] for row in obligations}
        if len(ids) != 22 or ids & all_ids:
            raise AssertionError("ordered mixed D2 obligation identity collision")
        all_ids |= ids
        manifests.append(
            {
                "candidate_id": candidate_id,
                "required_obligations": 22,
                "registered_obligations": 0,
                "blocked_obligations": 22,
                "obligations": obligations,
                "obligation_manifest_sha256": _sha(obligations),
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    if len(all_ids) != 264:
        raise AssertionError("ordered mixed D2 obligation manifest is incomplete")
    return manifests


def _expected_body(root: Path, config_path: Path, predecessor: Mapping[str, Any]) -> dict[str, Any]:
    candidate_ids = _load_source_candidates(root, predecessor)
    templates = _coordinate_templates(predecessor)
    manifests = _candidate_manifests(candidate_ids, templates)
    all_obligations = [row for manifest in manifests for row in manifest["obligations"]]
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_exact_264_slot_ordered_mixed_D2_registration_contract_downstream_blocked",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "contract_theorem": {
            "name": "complete_candidate_coordinate_product_registration_manifest",
            "premises": (
                "The bound predecessor supplies 12 candidate IDs and 22 fitted coordinate "
                "records, every one backed by a registered D1 arithmetic root. It supplies zero "
                "typed direction tangents and zero ordered mixed-D2 roots."
            ),
            "exact_result": (
                "The Cartesian product contains exactly 264 unique candidate-coordinate "
                "obligations. Each obligation names its registered D1 anchor and the exact typed "
                "fields that must be registered before an ordered mixed-D2 value can be admitted."
            ),
            "boundary": (
                "This is a closed-world registration interface and completeness result for the "
                "current 12 by 22 target set. It assigns no tangent, D2 value, covariant map, or "
                "physical impossibility theorem."
            ),
        },
        "registration_schema": {
            **_copy(EXPECTED_CONTRACT),
            "registered_cross_registry_projection_records": 0,
            "registered_direction_tangent_records": 0,
            "registered_ordered_D2_root_records": 0,
            "registration_status": "schema_complete_values_absent",
        },
        "coordinate_templates": templates,
        "candidate_manifests": manifests,
        "obligation_manifest_sha256": _sha(all_obligations),
        "gate_counts": {
            "selected_candidates": 12,
            "target_coordinates_per_candidate": 22,
            "required_ordered_mixed_D2_roots": 264,
            "unique_obligation_ids": 264,
            "registered_D1_anchors": 264,
            "unique_D1_row_atom_anchors_per_candidate": 20,
            "registered_generic_term_component_projections": 0,
            "registered_direction_tangent_embeddings": 0,
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
            "exact_264_slot_registration_contract_complete": True,
            "all_slots_candidate_bound_and_D1_anchored": True,
            "generic_term_component_projection_registered": False,
            "direction_tangent_embeddings_registered": False,
            "ordered_mixed_D2_values_registered": False,
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
            "infer_tangent_from_direction_label": {"rejected": True},
            "substitute_D1_root_for_ordered_D2_root": {"rejected": True},
            "invent_generic_term_component_projection": {"rejected": True},
            "admit_symbolic_obligation_id_as_arithmetic_root": {"rejected": True},
            "promote_registration_contract_to_physical_no_go": {"rejected": True},
            "reject_candidate_for_missing_registration": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": _copy(EXPECTED_PREDECESSOR),
            "full_source_D1_artifact": _copy(
                predecessor["source_bindings"]["direct_evidence"]["full_source_D1"]["artifact"]
            ),
        },
        "scope": (
            "candidate-bound exact registration obligations for the current 12 candidates and "
            "22 fitted ordered mixed-D2 coordinates; no tangent value, D2 arithmetic value, "
            "generic-term component projection, covariant no-go, corrected jet, full D2F, "
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
        raise ValueError("ordered mixed D2 source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("ordered mixed D2 local binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("ordered mixed D2 predecessor binding changed")
    predecessor = _load_predecessor(root)
    expected_source = predecessor["source_bindings"]["direct_evidence"]["full_source_D1"][
        "artifact"
    ]
    if bindings["full_source_D1_artifact"] != expected_source:
        raise ValueError("ordered mixed D2 full D1 binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("ordered mixed D2 content hash changed")
    _validate_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("ordered mixed D2 result boundary changed")


def build_contract(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor = _load_predecessor(root)
    body = _expected_body(root, config_path, predecessor)
    result = {**body, "content_sha256": _sha(body)}
    _validate_result(result, root=root)
    return result


def write_contract(config_path: Path) -> Path:
    result = build_contract(config_path)
    root = config_path.resolve().parents[2]
    output = _inside(root, OUTPUT_PATH)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    args = parser.parse_args()
    print(write_contract(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
