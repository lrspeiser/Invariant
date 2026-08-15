"""Audit registered variation/source evidence against the 22 jet selectors."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = (
    "sigma-quartic-fitted-output-connection-registered-variation-selection-audit-config-1.0"
)
RESULT_SCHEMA = "sigma-quartic-fitted-output-connection-registered-variation-selection-audit-1.0"
CAMPAIGN_ID = "quartic-fitted-output-connection-registered-variation-selection-audit-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_fitted_output_connection_registered_variation_selection_audit.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_fitted_output_connection_registered_variation_selection_audit.py"
)
TEST_PATH = "tests/test_quartic_fitted_output_connection_registered_variation_selection_audit.py"
OUTPUT_PATH = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-registered-variation-selection-audit/campaign.json"
)
FIRST_BLOCKER = (
    "candidate_bound_component_map_from_the_registered_G4_variation_or_source_DAG_into_"
    "the_22_output_connection_coordinates_or_exact_corrected_second_source_jet_values_required"
)
EXPECTED_PREDECESSOR = {
    "path": (
        "runs/physics-language/"
        "quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json"
    ),
    "file_sha256": "e0b87eb270d73f1fa7acb1ff31e0f234a545cf80c383fac21ffa0abc390a902b",
    "content_sha256": "b73d3bb175cf008f080ac900c0aed7f463f341d8efc8ebd4cdc4a8fbc3b6de21",
}
EXPECTED_INVENTORY = {
    "generic_G4_metric_variation": {
        "source": {
            "path": "src/sigma_theory_compiler/generic_g4_b4_termwise_normalization_campaign.py",
            "file_sha256": "7ce204576f2e26f84274195595f9ea8fa7d3855cadc15fc7bd0f4f8520a19c4e",
        },
        "config": {
            "path": "configs/generic_g4_b4_termwise_normalization_campaign.json",
            "file_sha256": "3e7d5b83889daba049c716fb991833a5245e31e3301d0509be10798b289f1742",
        },
        "test": {
            "path": "tests/test_generic_g4_b4_termwise_normalization_campaign.py",
            "file_sha256": "becdae92bf9913bbc5193fd71b32505c3ccd1a66350f8ad4d97065f6ed75a051",
        },
        "artifact": {
            "path": "runs/engine/generic-g4-b4-termwise-normalization-campaign.json",
            "file_sha256": "52b67ac2fbaa7cfa495e5da8088fa1aec9fc03b331e4e15bec3bea4dee35d224",
            "content_sha256": "a9e06a7d4c44ab40744fe54b0e7c11074b7d802355a2a743cbbb326febb35966",
        },
    },
    "generated_candidate_metric_variation": {
        "source": {
            "path": "src/sigma_theory_compiler/generated_candidate_metric_variation_execution_campaign.py",
            "file_sha256": "8404e31a6998e9f4475f1bce3cd6b1f19bd6aacf350a8deef7d2b30046aa980d",
        },
        "config": {
            "path": "configs/generated_candidate_metric_variation_execution_campaign.json",
            "file_sha256": "607a9b9335fbe9a4e6a37e9be5502b3eb33b11e864ceb8aca91e68e88a77a83c",
        },
        "test": {
            "path": "tests/test_generated_candidate_metric_variation_execution_campaign.py",
            "file_sha256": "8e93aadcb7b057c6d93a462f842fb68cb8cfb4cdfedb0fc15f3e39854b4c41c0",
        },
        "artifact": {
            "path": "runs/engine/generated-candidate-metric-variation-execution.json",
            "file_sha256": "8adcfbd3846eb60c55c837f972ebe82507d91def2224777f65c3cda69d2afb4e",
            "content_sha256": "bbd5ec183d7710141959361b555a218f7702c095023c06b158d35246569184d8",
        },
    },
    "universal_source_DAG": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_universal_source_dag_campaign.py",
            "file_sha256": "d63859371fadacf0b39ee0b5f344d1a1087cbfdbb47e696e43110e41dddfcbf4",
        },
        "config": {
            "path": "configs/backgrounds/quartic_universal_source_dag_campaign.json",
            "file_sha256": "74573cc055e5043539d3d7529816ee8da24b4f908d16611be053816eb00e1590",
        },
        "test": {
            "path": "tests/test_quartic_universal_source_dag_campaign.py",
            "file_sha256": "bd40881592c3d46780d93c493c8e3d66c476c078ab03cb54c47d90f057ccb6fa",
        },
        "artifact": {
            "path": "runs/physics-language/quartic-universal-source-dag-campaign/campaign.json",
            "file_sha256": "d1a8671b4b650abfd9f0d0ccc08f33d353e50b1ab65847628ff83b9cec0eac82",
            "content_sha256": "26220fe4a4c4ad0a7a911c9f64c5b69d634ab961f2dd123d654b14f2bf5431bf",
        },
    },
    "full_source_D1": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_full_source_jacobian_arithmetic_campaign.py",
            "file_sha256": "d2a04c214f8553a7e03f356debc77754e2ff73bb9c466f7dbfdf289e40732453",
        },
        "config": {
            "path": "configs/backgrounds/quartic_full_source_jacobian_arithmetic_campaign.json",
            "file_sha256": "b01f9a0d9c705409654ca03d340d45e4d68e68ae3f3aee6cbbfb29b6592d2dd5",
        },
        "test": {
            "path": "tests/test_quartic_full_source_jacobian_arithmetic_campaign.py",
            "file_sha256": "56091609506593f36426818d85da59afbad3dbbe95947e0945c46bfb06edd558",
        },
        "artifact": {
            "path": "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/campaign.json",
            "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
            "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
        },
    },
}
EXPECTED_SELECTION_CONTRACT = {
    "ambiguity_parameters": 22,
    "eligible_selector": (
        "candidate_bound_first_or_second_G4_X_jet_value_or_explicit_covariant_map_into_"
        "matching_output_coordinate"
    ),
    "evidence_inventory_closed_for_this_audit": True,
    "required_selection_rank": 22,
}
EXPECTED_POLICIES = {
    "covariant_variation_selection": "require_explicit_map_to_22_output_connection_coordinates",
    "corrected_second_source_jet": "require_candidate_bound_component_values",
    "cross_slice_D2F": "fail_closed",
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
        raise ValueError("registered variation audit path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "evidence_inventory": EXPECTED_INVENTORY,
        "selection_contract": EXPECTED_SELECTION_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("registered variation audit config boundary changed")


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("registered variation evidence bundle keys changed")
    for label, binding in bundle.items():
        expected_keys = (
            {"path", "file_sha256", "content_sha256"}
            if label == "artifact"
            else {
                "path",
                "file_sha256",
            }
        )
        if not isinstance(binding, Mapping) or set(binding) != expected_keys:
            raise ValueError("registered variation evidence binding keys changed")
        path = _inside(root, binding["path"])
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"registered variation evidence {label} file hash changed")
    artifact = json.loads(_inside(root, bundle["artifact"]["path"]).read_text(encoding="utf-8"))
    if artifact.get("content_sha256") != bundle["artifact"]["content_sha256"] or artifact.get(
        "content_sha256"
    ) != _content_sha(artifact):
        raise ValueError("registered variation evidence content binding changed")
    return artifact


def _load_predecessor(root: Path) -> dict[str, Any]:
    path = _inside(root, EXPECTED_PREDECESSOR["path"])
    if not path.is_file() or _file_sha(path) != EXPECTED_PREDECESSOR["file_sha256"]:
        raise ValueError("registered variation predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        value.get("content_sha256") != EXPECTED_PREDECESSOR["content_sha256"]
        or value.get("content_sha256") != _content_sha(value)
        or value.get("decision") != "pass_exact_finite_grid_first_and_second_jet_nonidentifiability"
        or value.get("gate_counts", {}).get("independent_ambiguity_parameters") != 22
        or len(value.get("coordinate_ambiguity_records", [])) != 22
    ):
        raise ValueError("registered variation predecessor boundary changed")
    return value


def _audit_evidence(
    evidence: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    generic = evidence["generic_G4_metric_variation"]
    if (
        set(generic)
        != {
            "schema_version",
            "campaign_id",
            "status",
            "campaign_source",
            "config_content_sha256",
            "formal_controls_artifact",
            "cadabra_script",
            "primary_source",
            "primary_source_transcription",
            "sign_and_variable_conventions",
            "canonical_term_count",
            "matched_term_count",
            "nonzero_residual_count",
            "term_records",
            "term_registry_root_sha256",
            "negative_controls",
            "metric_variation_normalization_pass",
            "scalar_equation_or_noether_rederived_here",
            "full_candidate_formal_pass_inferred",
            "global_energy_inferred",
            "observational_data_opened",
            "dark_matter_or_halo_inputs",
            "redshift_distance_inputs",
            "paid_llm_spend_usd",
            "interpretation",
            "content_sha256",
        }
        or generic.get("canonical_term_count") != 24
        or generic.get("matched_term_count") != 24
        or generic.get("nonzero_residual_count") != 0
        or len(generic.get("term_records", [])) != 24
    ):
        raise ValueError("registered generic G4 variation schema changed")

    generated = evidence["generated_candidate_metric_variation"]
    expected_families = {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 2,
    }
    if (
        generated.get("candidate_count") != 163
        or generated.get("family_counts") != expected_families
        or generated.get("metric_variation_execution_counts", {}).get(
            "candidate_backend_variations_executed"
        )
        != 0
        or len(generated.get("candidate_records", [])) != 163
    ):
        raise ValueError("registered generated metric variation schema changed")

    dag = evidence["universal_source_DAG"]
    dag_rows = dag.get("certificates")
    if (
        dag.get("counts")
        != {
            "selected": 12,
            "rejected": 0,
            "checkpoint_atoms_per_candidate": 2,
            "pure_derivative_component_roots_per_candidate": 88,
            "complete_component_tensors": 0,
            "affine_splits_proved": 0,
            "H7_closures": 0,
        }
        or not isinstance(dag_rows, list)
        or len(dag_rows) != 12
        or any(
            row.get("exact_component_derivative_roots_emitted") != 88
            or row.get("full_component_Frechet_tensors_complete") is not False
            or row.get("universal_acceleration_affine_split_proved") is not False
            for row in dag_rows
        )
    ):
        raise ValueError("registered universal source DAG schema changed")

    source = evidence["full_source_D1"]
    source_rows = source.get("certificates")
    if (
        source.get("counts", {}).get("full_source_entries_per_candidate") != 1683
        or source.get("counts", {}).get("full_source_jacobians_materialized") != 12
        or not isinstance(source_rows, list)
        or len(source_rows) != 12
        or any(
            row.get("full_11x153_source_Jacobian_entrywise_materialized") is not True
            or row.get("full_component_Frechet_tensors_orders_2_to_4_complete") is not False
            for row in source_rows
        )
    ):
        raise ValueError("registered full source D1 schema changed")
    dag_ids = sorted(str(row.get("candidate_id")) for row in dag_rows)
    source_ids = sorted(str(row.get("candidate_id")) for row in source_rows)
    if dag_ids != source_ids or len(set(dag_ids)) != 12:
        raise ValueError("registered source evidence candidate alignment changed")

    capabilities = [
        {
            "evidence": "generic_G4_metric_variation",
            "registered_units": 24,
            "registered_unit_type": "generic_metric_Euler_tensor_contractions",
            "candidate_bound_to_quartic_grid": False,
            "map_to_22_output_connection_coordinates": False,
            "corrected_second_source_jet_values": 0,
            "selector_equations_contributed": 0,
        },
        {
            "evidence": "generated_candidate_metric_variation",
            "registered_units": 163,
            "registered_unit_type": "candidate_specialized_metric_Euler_expressions",
            "candidate_bound_to_quartic_grid": False,
            "quartic_G4_X_grid_candidate_overlap": 0,
            "candidate_backend_variations_executed": 0,
            "map_to_22_output_connection_coordinates": False,
            "corrected_second_source_jet_values": 0,
            "selector_equations_contributed": 0,
        },
        {
            "evidence": "universal_source_DAG",
            "registered_units": 1056,
            "registered_unit_type": "pure_derivative_component_roots",
            "candidate_bound_to_quartic_grid": True,
            "candidate_count": 12,
            "full_component_Frechet_tensors_complete": False,
            "map_to_22_output_connection_coordinates": False,
            "corrected_second_source_jet_values": 0,
            "selector_equations_contributed": 0,
        },
        {
            "evidence": "full_source_D1",
            "registered_units": 20196,
            "registered_unit_type": "first_source_Jacobian_entries",
            "candidate_bound_to_quartic_grid": True,
            "candidate_count": 12,
            "source_Jacobian_shape": [11, 153],
            "complete_orders_2_to_4": False,
            "map_to_22_output_connection_coordinates": False,
            "corrected_second_source_jet_values": 0,
            "selector_equations_contributed": 0,
        },
    ]
    return capabilities, dag_ids


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    evidence: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    capabilities, candidate_ids = _audit_evidence(evidence)
    ambiguity = predecessor["coordinate_ambiguity_records"]
    selection_rows = [
        {
            "coordinate_ordinal": row["coordinate_ordinal"],
            "direction": row["direction"],
            "output_row": row["output_row"],
            "input_row": row["input_row"],
            "atom": row["atom"],
            "ambiguity_parameter": f"lambda_{row['coordinate_ordinal']}",
            "eligible_selector_equations_registered": 0,
            "parameter_selected": False,
        }
        for row in ambiguity
    ]
    if len(selection_rows) != 22 or [row["coordinate_ordinal"] for row in selection_rows] != list(
        range(22)
    ):
        raise ValueError("registered variation ambiguity coordinate ordering changed")
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "registered_variation_and_source_inventory_has_rank_zero_for_22_jet_selectors",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "registered_selection_theorem": {
            "name": "closed_inventory_rank_zero_for_action_jet_ambiguity_selection",
            "premises": (
                "The declared registered inventory contains the 24-term generic G4 metric Euler "
                "normalization, 163 generated candidate metric-Euler specializations, twelve "
                "candidate-aligned universal source DAGs, and twelve complete first source "
                "Jacobians. Eligible selectors must be candidate-bound first/second G4_X jet "
                "values or explicit maps into the matching 22 output-connection coordinates."
            ),
            "exact_result": (
                "The generic metric theorem has no output-coordinate map; the 163 generated "
                "specializations have zero overlap with the quartic G4_X grid; the source DAGs "
                "leave complete component Frechet tensors open; and the materialized source "
                "tensors are D1 only. Thus the registered selector matrix has shape 0-by-22, "
                "rank zero, and nullity 22. No lambda_i is selected."
            ),
            "boundary": (
                "This is a closed-world result for the four explicitly bound registered evidence "
                "bundles. It is neither a physical no-go nor evidence that a covariant variation "
                "rule cannot exist; adding a candidate-bound component map or corrected second "
                "source-jet values invalidates the premise and requires a new gate."
            ),
        },
        "candidate_ids": candidate_ids,
        "evidence_capabilities": capabilities,
        "coordinate_selection_records": selection_rows,
        "selection_matrix": {
            "rows": 0,
            "columns": 22,
            "rank": 0,
            "nullity": 22,
            "selected_parameters": 0,
            "unselected_parameters": 22,
        },
        "gate_counts": {
            "selected_candidates": 12,
            "registered_evidence_bundles": 4,
            "generic_G4_metric_Euler_terms": 24,
            "generated_metric_variation_candidates": 163,
            "generated_quartic_G4_X_grid_candidate_overlap": 0,
            "universal_source_DAG_pure_derivative_roots": 1056,
            "full_source_D1_entries": 20196,
            "eligible_selector_equations_registered": 0,
            "ambiguity_parameters_selected": 0,
            "ambiguity_parameters_remaining": 22,
            "registered_corrected_second_source_jet_entries": 0,
            "cross_slice_D2F_entries_admitted": 0,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "registered_variation_source_inventory_bound": True,
            "eligible_selector_schema_applied": True,
            "registered_selector_rank_zero": True,
            "all_22_ambiguity_parameters_remain_unselected": True,
            "physical_covariant_variation_no_go_proved": False,
            "covariant_output_connection_derivation_registered": False,
            "corrected_second_source_jet_registered": False,
            "cross_slice_D2F_entries_admitted": False,
            "complete_ordered_D2F_tensor_registered": False,
            "full_high_atom_good_unknown_identity_proved": False,
            "global_H7_energy_closed": False,
            "nonlinear_PDE_closed": False,
            "nonlinear_lifespan_proved": False,
            "candidate_theory_rejected": False,
            "observational_claim_made": False,
        },
        "exact_controls": {
            "treat_generic_metric_Euler_terms_as_output_connection_coordinates": {"rejected": True},
            "treat_pure_DAG_roots_as_complete_component_Frechet_tensors": {"rejected": True},
            "treat_D1_source_Jacobian_as_corrected_second_source_jet": {"rejected": True},
            "select_lambda_zero_from_absence_of_a_selector": {"rejected": True},
            "promote_closed_inventory_obstruction_to_physical_no_go": {"rejected": True},
            "reject_candidates_from_missing_selector": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": dict(EXPECTED_PREDECESSOR),
            "evidence_inventory": _copy(EXPECTED_INVENTORY),
        },
        "predecessor_decision": predecessor["decision"],
        "scope": (
            "candidate-bound closed inventory audit for selectors of the 22 fitted output-"
            "connection action-jet ambiguity parameters; no physical no-go, covariant derivation, "
            "corrected second-source jet, D2F admission, high-atom identity, H7, PDE, lifespan, "
            "candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "evidence_inventory",
    }:
        raise ValueError("registered variation source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("registered variation local source binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("registered variation predecessor binding changed")
    if bindings["evidence_inventory"] != EXPECTED_INVENTORY:
        raise ValueError("registered variation evidence inventory binding changed")


def _validated_inputs(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    predecessor = _load_predecessor(root)
    evidence = {label: _load_bundle(root, bundle) for label, bundle in EXPECTED_INVENTORY.items()}
    return predecessor, evidence


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("registered variation content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, evidence = _validated_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, evidence)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("registered variation result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, evidence = _validated_inputs(root)
    body = _expected_body(root, config_path, predecessor, evidence)
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
