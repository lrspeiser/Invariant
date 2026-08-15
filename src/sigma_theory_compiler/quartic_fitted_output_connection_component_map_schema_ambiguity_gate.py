"""Construct schema ambiguities for the missing G4/source-to-output component map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_SCHEMA = "sigma-quartic-fitted-output-connection-component-map-schema-ambiguity-config-1.0"
RESULT_SCHEMA = "sigma-quartic-fitted-output-connection-component-map-schema-ambiguity-gate-1.0"
CAMPAIGN_ID = "quartic-fitted-output-connection-component-map-schema-ambiguity-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_fitted_output_connection_component_map_schema_ambiguity_gate.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/"
    "quartic_fitted_output_connection_component_map_schema_ambiguity_gate.py"
)
TEST_PATH = "tests/test_quartic_fitted_output_connection_component_map_schema_ambiguity_gate.py"
OUTPUT_PATH = (
    "runs/physics-language/"
    "quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json"
)
FIRST_BLOCKER = (
    "register_the_typed_generic_term_to_source_component_projection_P10_Pother_state_"
    "tangent_embedding_and_22_ordered_mixed_D2F_roots"
)
EXPECTED_PREDECESSOR = {
    "path": (
        "runs/physics-language/"
        "quartic-fitted-output-connection-registered-variation-selection-audit/campaign.json"
    ),
    "file_sha256": "dfc8940a6f092de73da5641afd95c6cbf997b73ad63f8fa6f4ea3eaa8f395a20",
    "content_sha256": "6de93ca6700b21ff9f858a2b7f01d1a9d103271de1dde3f75385faaaa4a377d6",
}
EXPECTED_EVIDENCE = {
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
EXPECTED_CONTRACT = {
    "coefficient_field": "Q_sqrt2",
    "generic_term_basis_dimension": 24,
    "output_connection_basis_dimension": 22,
    "required_crosswalks": [
        "generic_term_id_to_source_component",
        "P10_Pother_direction_to_153_state_tangent",
        "ordered_mixed_D2F_root_for_each_output_coordinate",
        "corrected_source_jet_to_output_bundle_connection",
    ],
    "target_D1_membership_required": True,
}
EXPECTED_POLICIES = {
    "component_map_admission": "require_registered_crosswalk_and_live_arithmetic_root_replay",
    "corrected_second_source_jet": "fail_closed",
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
        raise ValueError("component map ambiguity path escapes project root")
    return path


def _copy(value: Any) -> Any:
    return json.loads(_canonical(value))


def _validate_config(value: Mapping[str, Any]) -> None:
    if value != {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessor": EXPECTED_PREDECESSOR,
        "direct_evidence": EXPECTED_EVIDENCE,
        "component_map_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }:
        raise ValueError("component map ambiguity config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = _inside(root, binding["path"])
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("component map bound artifact file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"] or value.get(
        "content_sha256"
    ) != _content_sha(value):
        raise ValueError("component map bound artifact content changed")
    return value


def _load_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("component map evidence bundle keys changed")
    for label in ("source", "config", "test"):
        binding = bundle[label]
        if (
            set(binding) != {"path", "file_sha256"}
            or _file_sha(_inside(root, binding["path"])) != binding["file_sha256"]
        ):
            raise ValueError("component map evidence file binding changed")
    artifact = bundle["artifact"]
    if set(artifact) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("component map evidence artifact binding keys changed")
    return _load_bound(root, artifact)


def _load_inputs(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    predecessor = _load_bound(root, EXPECTED_PREDECESSOR)
    if (
        predecessor.get("decision")
        != "registered_variation_and_source_inventory_has_rank_zero_for_22_jet_selectors"
        or predecessor.get("selection_matrix")
        != {
            "rows": 0,
            "columns": 22,
            "rank": 0,
            "nullity": 22,
            "selected_parameters": 0,
            "unselected_parameters": 22,
        }
        or len(predecessor.get("coordinate_selection_records", [])) != 22
    ):
        raise ValueError("component map predecessor boundary changed")
    generic = _load_bundle(root, EXPECTED_EVIDENCE["generic_G4_metric_variation"])
    dag = _load_bundle(root, EXPECTED_EVIDENCE["universal_source_DAG"])
    source = _load_bundle(root, EXPECTED_EVIDENCE["full_source_D1"])
    return predecessor, generic, dag, source


def _generic_terms(value: Mapping[str, Any]) -> tuple[list[str], sp.Matrix]:
    records = value.get("term_records")
    if (
        value.get("canonical_term_count") != 24
        or value.get("matched_term_count") != 24
        or value.get("nonzero_residual_count") != 0
        or not isinstance(records, list)
        or len(records) != 24
    ):
        raise ValueError("component map generic variation schema changed")
    ordered = sorted(records, key=lambda row: row["term_id"])
    term_ids = [str(row["term_id"]) for row in ordered]
    if len(set(term_ids)) != 24 or any(
        set(row)
        != {
            "term_id",
            "cadabra_fragment_sha256",
            "cadabra_coefficient",
            "B4_coefficient",
            "residual",
            "content_sha256",
        }
        for row in ordered
    ):
        raise ValueError("component map generic term record schema changed")
    coefficients = sp.Matrix([sp.Rational(str(row["cadabra_coefficient"])) for row in ordered])
    if (
        any(row["residual"] != "0" for row in ordered)
        or coefficients[0] == 0
        or coefficients[1] == 0
    ):
        raise ValueError("component map generic coefficient vector changed")
    return term_ids, coefficients


def _source_manifest(
    value: Mapping[str, Any],
) -> tuple[dict[tuple[int, str], Mapping[str, Any]], str]:
    manifest = value.get("common_full_entry_manifest")
    rows = value.get("certificates")
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("shape") != [11, 153]
        or manifest.get("total_entry_count") != 1683
        or not isinstance(manifest.get("entries"), list)
        or len(manifest["entries"]) != 1683
        or not isinstance(rows, list)
        or len(rows) != 12
        or any(
            row.get("full_11x153_source_Jacobian_entrywise_materialized") is not True
            or row.get("full_component_Frechet_tensors_orders_2_to_4_complete") is not False
            for row in rows
        )
    ):
        raise ValueError("component map source D1 schema changed")
    entries: dict[tuple[int, str], Mapping[str, Any]] = {}
    basis_sha = None
    for row in rows:
        current = row.get("provenance", {}).get("coordinate_atom_basis_sha256")
        if basis_sha is None:
            basis_sha = current
        elif current != basis_sha:
            raise ValueError("component map source coordinate basis changed")
    for entry in manifest["entries"]:
        key = (int(entry["source_row"]), str(entry["coordinate_atom"]))
        if key in entries:
            raise ValueError("component map source manifest contains duplicate row-atom")
        entries[key] = entry
    return entries, str(basis_sha)


def _dag_boundary(value: Mapping[str, Any], basis_sha: str) -> dict[str, Any]:
    rows = value.get("certificates")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ValueError("component map source DAG candidates changed")
    selected_sets = set()
    mixed_counts = set()
    for row in rows:
        evidence = row.get("evidence", {})
        coverage = evidence.get("coverage", {})
        if (
            row.get("provenance", {}).get("coordinate_atom_basis_sha256") != basis_sha
            or row.get("full_component_Frechet_tensors_complete") is not False
            or row.get("exact_component_derivative_roots_emitted") != 88
        ):
            raise ValueError("component map source DAG provenance changed")
        selected_sets.add(tuple(coverage.get("selected_atom_labels", [])))
        mixed_counts.add(coverage.get("mixed_multi_index_components_completed"))
    if selected_sets != {("q[0]", "p0[10]")} or mixed_counts != {0}:
        raise ValueError("component map source DAG mixed coverage changed")
    return {
        "candidate_count": 12,
        "coordinate_atom_basis_sha256": basis_sha,
        "pure_checkpoint_atoms": ["q[0]", "p0[10]"],
        "pure_derivative_roots": 1056,
        "mixed_multi_index_components_completed": 0,
        "full_component_Frechet_tensors_complete": False,
    }


def _sparse_matrix_records(matrix: sp.Matrix) -> list[dict[str, Any]]:
    return [
        {"output_coordinate": row, "generic_term": column, "value": str(matrix[row, column])}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


def _construct_ambiguities(
    predecessor: Mapping[str, Any],
    generic: Mapping[str, Any],
    dag: Mapping[str, Any],
    source: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    term_ids, coefficient_vector = _generic_terms(generic)
    entries, basis_sha = _source_manifest(source)
    dag_summary = _dag_boundary(dag, basis_sha)
    coordinates = predecessor["coordinate_selection_records"]
    jet_coordinates = _load_jet_coordinates(predecessor, root)
    identity_keys = ("coordinate_ordinal", "direction", "output_row", "input_row", "atom")
    if any(
        tuple(selection[key] for key in identity_keys) != tuple(jet[key] for key in identity_keys)
        for selection, jet in zip(coordinates, jet_coordinates, strict=True)
    ):
        raise ValueError("component map action-jet coordinate identities changed")
    beta = sp.Matrix([sp.sympify(row["beta"]) for row in jet_coordinates])
    if len(coordinates) != 22 or beta.rows != 22:
        raise ValueError("component map target coordinate count changed")

    target_records = []
    target_atoms = set()
    for row, beta_value in zip(coordinates, beta, strict=True):
        key = (int(row["output_row"]), str(row["atom"]))
        if key not in entries:
            raise ValueError("component map target D1 row-atom is not registered")
        entry = entries[key]
        target_atoms.add(key[1])
        target_records.append(
            {
                "coordinate_ordinal": row["coordinate_ordinal"],
                "direction": row["direction"],
                "output_row": row["output_row"],
                "input_row": row["input_row"],
                "atom": row["atom"],
                "beta": str(beta_value),
                "D1_manifest_coordinate_column": entry["coordinate_column"],
                "D1_arithmetic_root": entry["arithmetic_root"],
                "D1_arithmetic_dag_sha256": entry["arithmetic_dag_sha256"],
                "direction_state_tangent_registered": False,
                "ordered_mixed_D2F_root_registered": False,
                "zero_extension_D2_value": "0",
                "unit_extension_D2_value": "1",
                "both_extensions_preserve_registered_D1_value": True,
            }
        )
    if target_atoms & set(dag_summary["pure_checkpoint_atoms"]):
        raise ValueError("component map target atom unexpectedly has a pure DAG checkpoint")

    # The registered 24-vector imposes one scalar equation on each of 22 map rows.  M0 and M1
    # both reproduce beta exactly, while M1 differs by a nonzero vector orthogonal to c.
    base = sp.zeros(22, 24)
    for index in range(22):
        base[index, 0] = sp.simplify(beta[index] / coefficient_vector[0])
    alternate = base.copy()
    alternate[0, 0] += coefficient_vector[1]
    alternate[0, 1] -= coefficient_vector[0]
    if (
        base * coefficient_vector != beta
        or alternate * coefficient_vector != beta
        or base == alternate
    ):
        raise AssertionError("component map projection ambiguity construction failed")
    projection = {
        "generic_term_ids": term_ids,
        "generic_coefficient_vector": [str(value) for value in coefficient_vector],
        "target_beta_vector": [str(value) for value in beta],
        "matrix_shape": [22, 24],
        "unknown_entries": 528,
        "value_constraints": 22,
        "constraint_rank": 22,
        "affine_solution_dimension": 506,
        "base_sparse_entries": _sparse_matrix_records(base),
        "alternate_sparse_entries": _sparse_matrix_records(alternate),
        "base_residual_nonzero_count": 0,
        "alternate_residual_nonzero_count": 0,
        "maps_distinct": True,
        "covariance_or_index_equivariance_certified": False,
    }
    mixed = {
        **dag_summary,
        "target_coordinate_records": target_records,
        "target_coordinates": 22,
        "target_D1_memberships_found": 22,
        "unique_target_D1_row_atom_entries": len(
            {(row["output_row"], row["atom"]) for row in target_records}
        ),
        "target_atom_overlap_with_pure_DAG_checkpoints": 0,
        "direction_tangent_embeddings_registered": 0,
        "target_ordered_mixed_D2F_roots_registered": 0,
        "independent_mixed_D2_extension_parameters": 22,
        "explicit_witness_completions": 23,
    }
    return projection, mixed, target_records


def _load_jet_coordinates(predecessor: Mapping[str, Any], root: Path) -> list[Mapping[str, Any]]:
    # The selection predecessor retains coordinate identities but not beta; its direct predecessor
    # is bound in source_bindings and replayed here without broadening the configured inventory.
    binding = predecessor.get("source_bindings", {}).get("predecessor")
    if not isinstance(binding, Mapping):
        raise TypeError("component map action-jet predecessor binding missing")
    jet = _load_bound(root, binding)
    rows = jet.get("coordinate_ambiguity_records")
    if not isinstance(rows, list) or len(rows) != 22:
        raise ValueError("component map action-jet coordinate records missing")
    return rows


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    generic: Mapping[str, Any],
    dag: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    projection, mixed, target_records = _construct_ambiguities(
        predecessor, generic, dag, source, root
    )
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_constructive_component_map_and_mixed_D2_schema_ambiguity",
        "decision_counts": {"pass": 12, "blocked": 0, "reject": 0},
        "downstream_admission_counts": {"pass": 0, "blocked": 12, "reject": 0},
        "first_blocker": FIRST_BLOCKER,
        "component_map_schema_theorem": {
            "name": "registered_tensor_and_index_conventions_do_not_determine_the_cross_registry_map",
            "premises": (
                "The generic G4 registry supplies 24 exact abstract contraction coefficients. The "
                "source registry supplies a complete 11x153 D1 manifest on a common coordinate "
                "basis, including every target row-atom pair. The source DAG supplies only pure "
                "derivative checkpoints on q[0] and p0[10], with zero mixed multi-indices. No "
                "registered schema maps generic term IDs or P10/Pother directions into the target "
                "component and tangent bases."
            ),
            "exact_result": (
                "Two distinct exact 22x24 coefficient projections reproduce the same fitted beta "
                "vector; the complete solution space has affine dimension 506 before covariance "
                "constraints. Independently, zero and unit mixed-D2 extensions preserve every "
                "registered target D1 entry for each of 22 typed coordinate records, producing a "
                "22-parameter source-jet ambiguity."
            ),
            "boundary": (
                "The witness maps are schema completions, not covariant physical maps: the absent "
                "cross-registry tensor-equivariance and tangent-embedding schema is exactly what "
                "would distinguish admissible physics. This proves nonuniqueness using the current "
                "registered conventions, not a no-go for a future covariant derivation."
            ),
        },
        "generic_term_projection_ambiguity": projection,
        "mixed_D2_extension_ambiguity": mixed,
        "coordinate_records": target_records,
        "missing_schema": {
            "generic_term_id_to_source_component": False,
            "P10_Pother_direction_to_153_state_tangent": False,
            "ordered_mixed_D2F_root_for_each_output_coordinate": False,
            "corrected_source_jet_to_output_bundle_connection": False,
            "required_fields": [
                "generic_term_id",
                "source_row",
                "coordinate_atom",
                "direction_tangent_coefficients_in_153_basis",
                "ordered_D2_arithmetic_root",
                "ordered_D2_arithmetic_dag_sha256",
                "output_bundle_projection_rule_id",
                "candidate_id",
            ],
        },
        "gate_counts": {
            "selected_candidates": 12,
            "generic_term_basis_dimension": 24,
            "output_connection_basis_dimension": 22,
            "generic_projection_unknowns": 528,
            "generic_projection_value_constraints": 22,
            "generic_projection_affine_dimension": 506,
            "distinct_projection_witnesses": 2,
            "target_D1_memberships_found": 22,
            "unique_target_D1_row_atom_entries": mixed["unique_target_D1_row_atom_entries"],
            "target_pure_DAG_checkpoint_overlaps": 0,
            "mixed_multi_index_components_completed": 0,
            "target_direction_tangent_embeddings_registered": 0,
            "target_ordered_mixed_D2F_roots_registered": 0,
            "mixed_D2_extension_ambiguity_parameters": 22,
            "registered_corrected_second_source_jet_entries": 0,
            "cross_slice_D2F_entries_admitted": 0,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "claim_seals": {
            "all_22_target_D1_row_atom_entries_registered": True,
            "two_exact_term_projection_witnesses_constructed": True,
            "term_projection_affine_dimension_506_proved": True,
            "mixed_D2_22_parameter_ambiguity_constructed": True,
            "registered_cross_registry_component_map_unique": False,
            "physical_covariant_component_map_no_go_proved": False,
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
            "promote_schema_projection_witness_to_covariant_map": {"rejected": True},
            "infer_direction_tangent_from_P10_Pother_label_only": {"rejected": True},
            "treat_D1_membership_as_ordered_mixed_D2_value": {"rejected": True},
            "treat_pure_DAG_checkpoint_as_mixed_target_root": {"rejected": True},
            "promote_schema_ambiguity_to_physical_no_go": {"rejected": True},
            "reject_candidates_from_missing_component_map": {"rejected": True},
        },
        "data_seals": dict(EXPECTED_SEALS),
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "predecessor": dict(EXPECTED_PREDECESSOR),
            "direct_evidence": _copy(EXPECTED_EVIDENCE),
        },
        "predecessor_decision": predecessor["decision"],
        "scope": (
            "candidate-bound constructive schema ambiguity for maps from registered generic G4 "
            "terms and source D1/DAG conventions into the 22 fitted output coordinates; no "
            "physical no-go, covariant map, corrected source jet, D2F admission, high-atom "
            "identity, H7, PDE, lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        "source",
        "config",
        "test",
        "predecessor",
        "direct_evidence",
    }:
        raise ValueError("component map source binding keys changed")
    for label, relative in {
        "source": SOURCE_PATH,
        "config": CONFIG_PATH,
        "test": TEST_PATH,
    }.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("component map local source binding changed")
    if bindings["predecessor"] != EXPECTED_PREDECESSOR:
        raise ValueError("component map predecessor binding changed")
    if bindings["direct_evidence"] != EXPECTED_EVIDENCE:
        raise ValueError("component map direct evidence binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("component map content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, generic, dag, source = _load_inputs(validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, generic, dag, source)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("component map result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessor, generic, dag, source = _load_inputs(root)
    body = _expected_body(root, config_path, predecessor, generic, dag, source)
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
