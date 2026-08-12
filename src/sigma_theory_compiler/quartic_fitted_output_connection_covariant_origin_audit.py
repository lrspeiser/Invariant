"""Audit whether the fitted reference connection has registered covariant origin."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .quartic_candidate_pother_one_form_connection_gate import (
    _validate_result as validate_pother_gate,
)
from .quartic_row0_arithmetic_expansion_campaign import _content_hash_matches

CONFIG_SCHEMA = "sigma-quartic-fitted-output-connection-covariant-origin-audit-config-1.0"
RESULT_SCHEMA = "sigma-quartic-fitted-output-connection-covariant-origin-audit-1.0"
CAMPAIGN_ID = "quartic-fitted-output-connection-covariant-origin-audit-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_fitted_output_connection_covariant_origin_audit.json"
)
OUTPUT_PATH = (
    "runs/physics-language/quartic-fitted-output-connection-covariant-origin-audit/"
    "campaign.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_fitted_output_connection_covariant_origin_audit.py"
)
TEST_PATH = "tests/test_quartic_fitted_output_connection_covariant_origin_audit.py"
FIRST_BLOCKER = (
    "registered_covariant_second_source_jet_or_explicit_output_bundle_connection_functor_"
    "with_22_coefficient_action_root_provenance_not_available"
)

EXPECTED_PREDECESSORS = {
    "candidate_pother_one_form_connection": {
        "path": (
            "runs/physics-language/quartic-candidate-pother-one-form-connection-gate/"
            "campaign.json"
        ),
        "file_sha256": "f43b5623a46dc19541cd5a6323bd8b0c1ea7b63c7fbcfdb1e26bfe35a2f82bf6",
        "content_sha256": "c79256b901eb6e7b543938cae6e6cf4b41e9fe2778aa37077604cf39a869f93d",
    }
}
EXPECTED_DIRECT_EVIDENCE = {
    "covariant_action": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_dirac_hamiltonian_campaign.py",
            "file_sha256": "581a8daa447c9fb4096ca3b68052575ea9ad6842e37f938151fdf0ed931af510",
        },
        "config": {
            "path": "configs/backgrounds/quartic_dirac_hamiltonian_campaign.json",
            "file_sha256": "a1a9f4c228a469c4b31388d356f437350bf86b29efc3022c5373e278c567acc7",
        },
        "test": {
            "path": "tests/test_quartic_dirac_hamiltonian_campaign.py",
            "file_sha256": "0b4c56a13d8636d65ccacffd83bd418b8c9dda1bf5f45e1b071846abfccc2594",
        },
        "artifact": {
            "path": "runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json",
            "file_sha256": "68541766993d0d46f23dd2707c4e5db8bbf00dbdd9c442fc3802c1c2f7d9bb3f",
            "content_sha256": "69f6f67237020adab07f741b8de154465fa5d24984d78dfb2541da4567db2a47",
        },
    },
    "full_source_jacobian": {
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
            "path": (
                "runs/physics-language/quartic-full-source-jacobian-arithmetic-campaign/"
                "campaign.json"
            ),
            "file_sha256": "e893ebcaef464b958516279c557382fb76ecdb0fd542b3e3fed6a347076fcdae",
            "content_sha256": "1707b7258fd434f68b06c7af6bc447b4136624b9916992df8b412e048ab6538a",
        },
    },
}
ACTION_TOP_KEYS = {
    "binding_campaign_sha256",
    "certificates",
    "claim",
    "config_sha256",
    "content_sha256",
    "counts",
    "errors",
    "negative_controls",
    "primary_sources",
    "schema_version",
    "scope",
    "source_ir_sha256",
    "status",
    "symmetrizer_campaign_sha256",
}
ACTION_RECORD_KEYS = {
    "adm_hessian_and_primary_constraint",
    "candidate_id",
    "certified_local_jet_embedding",
    "claim",
    "coefficients",
    "covariant_action_specialization",
    "dirac_chain",
    "forward_homogeneous_invariant_domain",
    "on_shell_local_flrw_witness",
    "on_shell_quadratic_physical_hamiltonian",
    "schema_version",
    "scope",
    "status",
}
SOURCE_TOP_KEYS = {
    "certificates",
    "claim",
    "common_full_entry_manifest",
    "common_principal_arithmetic_packet",
    "config_sha256",
    "content_sha256",
    "counts",
    "errors",
    "physical_pencil_J_identity",
    "schema_version",
    "scope",
    "status",
    "upstream_sha256",
}
SOURCE_RECORD_KEYS = {
    "H7_derivative_loss_resolved",
    "candidate_id",
    "coefficients",
    "full_11x153_source_Jacobian_entrywise_materialized",
    "full_H7_commutator_closed",
    "full_component_Frechet_tensors_orders_2_to_4_complete",
    "global_dyadic_summation_applied",
    "lower_entries_entrywise_arithmetic",
    "paralinearization_remainder_bound_proved",
    "physical_pencil_J_identity_entry_residuals_zero",
    "physical_pencil_J_identity_proved",
    "principal_entries_entrywise_arithmetic",
    "provenance",
    "remaining_gate",
    "schema_version",
    "source_Jacobian_shape",
    "status",
    "total_entries_entrywise_arithmetic",
}
EXPECTED_CONTRACT = {
    "fitted_connection_coefficients_per_candidate": 22,
    "required_origin": (
        "explicit_output_bundle_connection_functor_or_covariant_second_source_jet"
    ),
    "required_binding": "each_fitted_coefficient_to_action_or_source_arithmetic_root",
    "closed_world_action_record_keys": 13,
    "closed_world_source_record_keys": 18,
    "admission": "forbid_value_fit_as_origin_proof",
}
EXPECTED_POLICIES = {
    "covariant_origin": "require_explicit_registered_derivation_and_live_root_replay",
    "numeric_fit": "insufficient_for_provenance",
    "cross_slice_admission": "fail_closed",
    "full_D2_promotion": "fail_closed",
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
CLAIM_SEALS = {
    "registered_covariant_action_specializations_bound": True,
    "registered_full_source_D1_jacobians_bound": True,
    "fitted_connection_origin_schema_audited": True,
    "fitted_connection_value_solution_retained": True,
    "covariant_output_connection_derivation_registered": False,
    "corrected_second_source_jet_registered": False,
    "fitted_coefficients_with_action_root_provenance": False,
    "connection_derived_from_covariant_action": False,
    "physical_two_sided_connection_registered": False,
    "corrected_cross_slice_curl_zero": False,
    "cross_slice_D2F_entries_admitted": False,
    "complete_ordered_D2F_tensor_registered": False,
    "full_high_atom_good_unknown_identity_proved": False,
    "global_H7_energy_closed": False,
    "nonlinear_PDE_closed": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
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
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError("fitted connection origin path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "direct_evidence": EXPECTED_DIRECT_EVIDENCE,
        "origin_contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("fitted connection origin config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("fitted connection origin artifact binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("fitted connection origin artifact file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("fitted connection origin artifact content binding changed")
    if not _content_hash_matches(value):
        raise ValueError("fitted connection origin artifact content changed")
    return value


def _records(value: Mapping[str, Any], key: str) -> dict[str, Mapping[str, Any]]:
    raw = value.get(key, [])
    if not isinstance(raw, list):
        raise TypeError("fitted connection origin candidate records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("fitted connection origin candidate set changed")
    return result


def _validate_file_bundle(root: Path, bundle: Mapping[str, Any]) -> dict[str, Any]:
    if set(bundle) != {"source", "config", "test", "artifact"}:
        raise ValueError("fitted connection origin direct bundle changed")
    for label in ("source", "config", "test"):
        binding = bundle[label]
        if set(binding) != {"path", "file_sha256"} or _file_sha(
            _inside(root, binding["path"])
        ) != binding["file_sha256"]:
            raise ValueError(f"fitted connection origin direct {label} binding changed")
    return _load_bound(root, bundle["artifact"])


def _validate_evidence(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    action = _validate_file_bundle(root, EXPECTED_DIRECT_EVIDENCE["covariant_action"])
    source = _validate_file_bundle(root, EXPECTED_DIRECT_EVIDENCE["full_source_jacobian"])
    if (
        set(action) != ACTION_TOP_KEYS
        or action.get("status") != "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        or action.get("counts")
        != {"local_on_shell_adm_dirac_hamiltonian_passed": 12, "rejected": 0, "selected": 12}
    ):
        raise ValueError("fitted connection origin action schema changed")
    if (
        set(source) != SOURCE_TOP_KEYS
        or source.get("status")
        != "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
        or source.get("counts", {}).get("full_source_entries_per_candidate") != 1683
    ):
        raise ValueError("fitted connection origin source schema changed")
    action_records = _records(action, "certificates")
    source_records = _records(source, "certificates")
    if set(action_records) != set(source_records):
        raise ValueError("fitted connection origin direct candidates disagree")
    for candidate_id in action_records:
        action_row = action_records[candidate_id]
        source_row = source_records[candidate_id]
        if (
            set(action_row) != ACTION_RECORD_KEYS
            or set(source_row) != SOURCE_RECORD_KEYS
            or set(action_row["covariant_action_specialization"]) != {"G2", "G3", "G4", "G5"}
            or action_row["coefficients"] != source_row["coefficients"]
            or source_row["full_component_Frechet_tensors_orders_2_to_4_complete"] is not False
        ):
            raise ValueError("fitted connection origin candidate evidence schema changed")
    return action, source


def _validate_predecessor(
    predecessor: Mapping[str, Any], action: Mapping[str, Any], source: Mapping[str, Any], root: Path
) -> None:
    validate_pother_gate(predecessor, root=root)
    maps = {
        "predecessor": _records(predecessor, "candidate_records"),
        "action": _records(action, "certificates"),
        "source": _records(source, "certificates"),
    }
    ids = set(maps["predecessor"])
    if any(set(rows) != ids for rows in maps.values()):
        raise ValueError("fitted connection origin predecessor candidates disagree")
    for candidate_id in ids:
        coefficients = maps["predecessor"][candidate_id]["coefficients"]
        if any(rows[candidate_id]["coefficients"] != coefficients for rows in maps.values()):
            raise ValueError("fitted connection origin predecessor coefficients disagree")


def _candidate_records(
    predecessor: Mapping[str, Any], action: Mapping[str, Any], source: Mapping[str, Any]
) -> list[dict[str, Any]]:
    prior = _records(predecessor, "candidate_records")
    actions = _records(action, "certificates")
    sources = _records(source, "certificates")
    result = []
    for candidate_id, row in sorted(prior.items()):
        witness = row["free_variable_zero_connection_witness"]
        sparse = {
            "Pother_direction_nonzero_entries": witness["Pother_direction_nonzero_entries"],
            "P10_direction_nonzero_entries": witness["P10_direction_nonzero_entries"],
        }
        if witness["total_nonzero_count"] != 22:
            raise ValueError("fitted connection origin witness count changed")
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": row["coefficients"],
                "covariant_action_specialization": actions[candidate_id][
                    "covariant_action_specialization"
                ],
                "covariant_action_specialization_sha256": _sha(
                    actions[candidate_id]["covariant_action_specialization"]
                ),
                "full_source_provenance": sources[candidate_id]["provenance"],
                "fitted_connection_content_sha256": _sha(sparse),
                "fitted_connection_nonzero_coefficients": 22,
                "fitted_coefficients_with_action_root_provenance": 0,
                "registered_output_connection_functors": 0,
                "registered_corrected_second_source_jet_entries": 0,
                "full_component_Frechet_tensors_orders_2_to_4_complete": False,
                "origin_decision": "not_identifiable_from_registered_action_and_D1_source_schemas",
                "cross_slice_admitted_entries": 0,
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _expected_body(
    root: Path,
    config_path: Path,
    predecessor: Mapping[str, Any],
    action: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    records = _candidate_records(predecessor, action, source)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "registered_action_and_D1_source_schemas_do_not_identify_covariant_origin_of_"
            "22_coefficient_fit_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "covariant_action_specializations_bound": 12,
            "full_source_D1_jacobians_bound": 12,
            "full_source_D1_entries_per_candidate": 1683,
            "action_record_schema_keys": 13,
            "source_record_schema_keys": 18,
            "fitted_connection_coefficients_per_candidate": 22,
            "fitted_connection_coefficients_audited": 264,
            "fitted_coefficients_with_action_root_provenance": 0,
            "registered_output_connection_functors": 0,
            "registered_corrected_second_source_jet_entries": 0,
            "complete_component_Frechet_D2_to_D4_tensors": 0,
            "covariant_action_derived_connections": 0,
            "cross_slice_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "origin_theorem": {
            "name": "registered_action_D1_source_provenance_nonidentifiability",
            "premises": (
                "Twelve exact G2/G3/G4/G5 covariant-action specializations and twelve complete "
                "11x153 first source Jacobians are bound and candidate-aligned."
            ),
            "closed_world_result": (
                "The exact 13-key action certificate schema and 18-key source certificate schema "
                "contain no output-bundle connection functor, corrected second source jet, or "
                "binding from any of the 22 fitted coefficients to an action/source arithmetic root."
            ),
            "provenance_reason": (
                "Pointwise satisfaction of the two-sided linear equations is extensional value "
                "evidence; derivation from a covariant action is intensional provenance and cannot "
                "be inferred without a registered derivation map and live root replay."
            ),
            "boundary": (
                "This is a no-go for origin inference from the registered action plus D1 schemas, "
                "not a proof that no covariant connection or corrected source jet exists."
            ),
        },
        "candidate_records": records,
        "exact_controls": {
            "infer_origin_from_numeric_fit": {"rejected": True},
            "treat_covariant_action_specialization_as_D2_source_jet": {"rejected": True},
            "admit_cross_slice_without_origin_binding": {
                "rejected": True,
                "cross_slice_entries_admitted": 0,
            },
            "reject_candidates_from_missing_origin": {"rejected": True},
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "general_covariant_typed_map_not_registered",
            "remaining_other_principal_by_other_principal_values_not_registered",
            "complete_high_atom_identity_TC1_TC2_TC3_TC5_B7_H7_PDE_lifespan_not_closed",
        ],
        "claim_seals": CLAIM_SEALS,
        "data_seals": EXPECTED_SEALS,
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "direct_evidence": EXPECTED_DIRECT_EVIDENCE,
            **EXPECTED_PREDECESSORS,
        },
        "scope": (
            "candidate-bound closed-world provenance audit of the 22-coefficient algebraic "
            "connection fit against registered covariant-action specializations and complete D1 "
            "source Jacobians; no physical/covariant connection, corrected D2 source, D2F "
            "admission, complete D2F, high-atom identity, H7, PDE, lifespan, rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("fitted connection origin source bindings missing")
    if set(bindings) != {"source", "config", "test", "direct_evidence", *EXPECTED_PREDECESSORS}:
        raise ValueError("fitted connection origin source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("fitted connection origin local binding changed")
    if bindings["direct_evidence"] != EXPECTED_DIRECT_EVIDENCE:
        raise ValueError("fitted connection origin direct evidence binding changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings[label] != expected:
            raise ValueError("fitted connection origin predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("fitted connection origin content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    action, source = _validate_evidence(validation_root)
    predecessor = _load_bound(
        validation_root, EXPECTED_PREDECESSORS["candidate_pother_one_form_connection"]
    )
    _validate_predecessor(predecessor, action, source, validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, action, source)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("fitted connection origin result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    action, source = _validate_evidence(root)
    predecessor = _load_bound(root, config["predecessors"]["candidate_pother_one_form_connection"])
    _validate_predecessor(predecessor, action, source, root)
    body = _expected_body(root, config_path, predecessor, action, source)
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
