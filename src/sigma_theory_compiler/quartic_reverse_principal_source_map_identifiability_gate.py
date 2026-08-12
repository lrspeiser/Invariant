"""Prove that reverse principal D2F values need a missing typed source map."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_full_d2f_high_atom_coverage_gate import (
    _validate_result as validate_full_d2f_coverage,
)
from .quartic_principal_high_atom_connection_extension_gate import (
    _validate_result as validate_principal_connection_extension,
)
from .quartic_scalar_hessian_d2_integrability_gate import FAMILY_SPECS
from .quartic_scalar_hessian_d2_integrability_gate import (
    _validate_result as validate_scalar_hessian_d2,
)
from .quartic_scalar_hessian_output_bundle_repair_gate import (
    _validate_result as validate_output_bundle_repair,
)
from .quartic_unspecialized_source_jacobian_campaign import _unspecialized_principal_blocks

CONFIG_SCHEMA = "sigma-quartic-reverse-principal-source-map-identifiability-config-1.0"
RESULT_SCHEMA = "sigma-quartic-reverse-principal-source-map-identifiability-gate-1.0"
CAMPAIGN_ID = "quartic-reverse-principal-source-map-identifiability-001"
CONFIG_PATH = (
    "configs/backgrounds/quartic_reverse_principal_source_map_identifiability_gate.json"
)
OUTPUT_PATH = (
    "runs/physics-language/quartic-reverse-principal-source-map-identifiability-gate/"
    "campaign.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_reverse_principal_source_map_identifiability_gate.py"
)
TEST_PATH = "tests/test_quartic_reverse_principal_source_map_identifiability_gate.py"
FIRST_BLOCKER = (
    "typed_coordinate_to_Einstein_derivative_map_for_90_other_principal_atoms_not_registered"
)
FORBIDDEN_MAP_KEYS = {
    "coordinate_to_einstein_derivative_map",
    "principal_atom_to_einstein_derivative_map",
    "typed_coordinate_to_block_frechet_map",
}

EXPECTED_DIRECT_DEPENDENCIES = {
    "unspecialized_source_jacobian_campaign": {
        "source": {
            "path": "src/sigma_theory_compiler/quartic_unspecialized_source_jacobian_campaign.py",
            "file_sha256": "f5a8649b52bd7f2384ee9087d0f5f6d8850a5c5bc443fbe823c6b09655ce9616",
        },
        "test": {
            "path": "tests/test_quartic_unspecialized_source_jacobian_campaign.py",
            "file_sha256": "82040e82e63f23b2e03df9351a8049cc957bc27414cabc85e49ac01132c83181",
        },
        "artifact": {
            "path": (
                "runs/physics-language/quartic-unspecialized-source-jacobian-campaign/"
                "campaign.json"
            ),
            "file_sha256": "8ecae346f75ba5bbeb266e486b96f48a0c76387513ff92d20d1bc68d8ecef22b",
            "content_sha256": "b60dbbb191f43d84d3d9c9e44e4adf70e4e7d729143905561b695cfabcaa7c72",
        },
    }
}

EXPECTED_PREDECESSORS = {
    "principal_high_atom_connection_extension": {
        "path": (
            "runs/physics-language/quartic-principal-high-atom-connection-extension-gate/"
            "campaign.json"
        ),
        "file_sha256": "e4ffc8f0d82f3c4381703f338a03cc334e15268aaf5b0c7f0dd1305ee96f8b92",
        "content_sha256": "33942664dc481ae112650c1f9ad1c4834687161601b54348b55a82139816c028",
    },
    "full_d2f_high_atom_coverage": {
        "path": "runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json",
        "file_sha256": "b9ce34960b766a6fe74a36a13190b0f050a1447884d599fad1eebfe189b32590",
        "content_sha256": "e7e4e4171aed90d07d68791183c58a696e77b9bed745f1018da2c5ee9438c38a",
    },
    "scalar_hessian_d2": {
        "path": "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json",
        "file_sha256": "654a0442e0d6ec0166eed1d15da163260658d6cd44192da9cbb4e4b2b88f105a",
        "content_sha256": "66f680eec0ab93169163f7f1e2055aac7713f45fcc2abf2f12907183e701b45c",
    },
    "scalar_hessian_output_bundle_repair": {
        "path": "runs/physics-language/quartic-scalar-hessian-output-bundle-repair-gate/campaign.json",
        "file_sha256": "e1ae98ebcb3c2739f7c84938d61ce9e7d2d209d4025f54a7d1d499a8495acfdb",
        "content_sha256": "688dcb478b86d44330f8a3623183e91c237bd91f31bd4e91bf5869098175973f",
    },
}
EXPECTED_IDENTIFIABILITY_CONTRACT = {
    "reverse_ordered_pair_count": 810,
    "reverse_output_entry_count": 8910,
    "registered_map_search_domain": "current_predecessor_exact_schemas",
    "zero_map": "all_partial_Gmunu_over_partial_Pother_atoms_zero",
    "alternative_map": (
        "partial_G00_over_partial_s11_field0_equals_1_all_other_entries_zero"
    ),
    "ambiguous_witness_coordinate": {
        "left_atom": "s11[0]",
        "right_atom": "s11[10]",
        "output_row": 10,
    },
}
EXPECTED_POLICIES = {
    "reverse_value_admission": (
        "require_registered_typed_coordinate_to_Einstein_derivative_map"
    ),
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
    "current_predecessor_schemas_audited_for_typed_source_map": True,
    "typed_coordinate_to_Einstein_derivative_map_registered": False,
    "reverse_source_values_identifiable_from_current_evidence": False,
    "explicit_two_map_ambiguity_witness_constructed": True,
    "restricted_zero_extended_connection_reverse_correction_zero": True,
    "reverse_Pother_by_P10_values_materialized": False,
    "corrected_cross_slice_curl_zero": False,
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
        raise ValueError("reverse source-map path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "direct_dependencies": EXPECTED_DIRECT_DEPENDENCIES,
        "identifiability_contract": EXPECTED_IDENTIFIABILITY_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("reverse source-map config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("reverse source-map predecessor binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("reverse source-map predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("reverse source-map predecessor content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("reverse source-map predecessor content hash changed")
    return value


def _validate_direct_dependencies(
    root: Path, dependencies: Mapping[str, Any]
) -> dict[str, Any]:
    if dependencies != EXPECTED_DIRECT_DEPENDENCIES:
        raise ValueError("reverse source-map direct dependency boundary changed")
    if set(dependencies) != {"unspecialized_source_jacobian_campaign"}:
        raise ValueError("reverse source-map direct dependency labels changed")
    bundle = dependencies["unspecialized_source_jacobian_campaign"]
    if not isinstance(bundle, Mapping) or set(bundle) != {"source", "test", "artifact"}:
        raise ValueError("reverse source-map direct dependency bundle changed")
    for label in ("source", "test"):
        binding = bundle[label]
        if not isinstance(binding, Mapping) or set(binding) != {"path", "file_sha256"}:
            raise ValueError("reverse source-map direct file binding shape changed")
        path = _inside(root, str(binding["path"]))
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"reverse source-map direct {label} file hash changed")
    artifact_binding = bundle["artifact"]
    if not isinstance(artifact_binding, Mapping):
        raise TypeError("reverse source-map direct artifact binding missing")
    artifact = _load_bound(root, artifact_binding)
    generic = artifact.get("generic_unspecialized_source_jacobian_control")
    if (
        artifact.get("status")
        != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
        or artifact.get("counts", {}).get("selected") != 12
        or not isinstance(generic, Mapping)
        or generic.get("passed") is not True
        or generic.get("unspecialized_block_extraction", {}).get("block_content_sha256")
        != _generic_ambiguity_witness()["unspecialized_block_sha256"]
    ):
        raise ValueError("reverse source-map direct artifact semantic replay changed")
    return artifact


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    raw = value.get("candidate_records", [])
    if not isinstance(raw, list):
        raise TypeError("reverse source-map candidate records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("reverse source-map candidate set changed")
    return result


def _recursive_keys(value: Any) -> set[str]:
    if isinstance(value, Mapping):
        return {str(key) for key in value} | set().union(
            *(_recursive_keys(item) for item in value.values()), set()
        )
    if isinstance(value, list):
        return set().union(*(_recursive_keys(item) for item in value), set())
    return set()


@cache
def _generic_ambiguity_witness() -> dict[str, Any]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    zero = {
        symbol: 0
        for symbol in list(data["gradient_lower"])
        + list(data["hessian_lower"])
        + list(data["einstein_upper"])
    }
    zero[data["m2"]] = 1
    zero[data["c20"]] = data["c20"]
    inverse = blocks["A"].subs(zero).inv()
    spec = next(item for item in FAMILY_SPECS if item[0] == "s11")
    _, _, _, kind, first, second, multiplicity = spec
    chunk = multiplicity * blocks[kind][first][second]
    g00 = next(symbol for symbol in data["einstein_upper"].free_symbols if str(symbol) == "G_00")
    derivative = (
        inverse
        * sp.diff(blocks["A"], g00).subs(zero)
        * inverse
        * chunk.subs(zero)
        - inverse * sp.diff(chunk, g00).subs(zero)
    ).applyfunc(sp.factor)
    nonzero = [
        {"output_row": row, "right_field": column, "value": str(derivative[row, column])}
        for row in range(11)
        for column in range(11)
        if derivative[row, column] != 0
    ]
    if nonzero != [{"output_row": 10, "right_field": 10, "value": "-2*alpha"}]:
        raise ValueError("reverse source-map ambiguity witness changed")
    return {
        "unspecialized_block_sha256": blocks["content_sha256"],
        "einstein_component": "G_00",
        "left_atom": "s11[0]",
        "right_atom": "s11[10]",
        "output_row": 10,
        "generic_zero_map_value": "0",
        "generic_alternative_map_value": "-2*alpha",
        "generic_nonzero_entries": nonzero,
    }


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("reverse source-map predecessor set changed")
    validate_principal_connection_extension(
        values["principal_high_atom_connection_extension"], root=root
    )
    validate_full_d2f_coverage(values["full_d2f_high_atom_coverage"], root=root)
    validate_scalar_hessian_d2(values["scalar_hessian_d2"], root=root)
    validate_output_bundle_repair(values["scalar_hessian_output_bundle_repair"], root=root)
    maps = {label: _records(value) for label, value in values.items()}
    ids = set(maps["principal_high_atom_connection_extension"])
    if any(set(rows) != ids for rows in maps.values()):
        raise ValueError("reverse source-map predecessor candidates disagree")
    for candidate_id in ids:
        coefficients = maps["principal_high_atom_connection_extension"][candidate_id][
            "coefficients"
        ]
        if any(rows[candidate_id].get("coefficients") != coefficients for rows in maps.values()):
            raise ValueError("reverse source-map predecessor coefficients disagree")
    present = set().union(*(_recursive_keys(value) for value in values.values()))
    if present & FORBIDDEN_MAP_KEYS:
        raise ValueError("reverse source-map predecessor unexpectedly registered a typed map")
    _generic_ambiguity_witness()


def _candidate_records(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    extension = _records(values["principal_high_atom_connection_extension"])
    witness = _generic_ambiguity_witness()
    result = []
    for candidate_id, row in sorted(extension.items()):
        alpha = sp.sympify(row["coefficients"]["a10"])
        alternative = sp.factor(-2 * alpha)
        if alternative == 0:
            raise ValueError("reverse source-map ambiguity vanished for a candidate")
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": row["coefficients"],
                "predecessor_one_sided_dense_content_sha256": row[
                    "one_sided_value_manifest"
                ]["dense_content_sha256"],
                "ambiguity_witness": {
                    "left_atom": witness["left_atom"],
                    "right_atom": witness["right_atom"],
                    "output_row": witness["output_row"],
                    "zero_map_value": "0",
                    "alternative_map_value": str(alternative),
                    "values_disagree": True,
                    "restricted_reverse_connection_correction": "0",
                },
                "reverse_values_materialized": 0,
                "corrected_cross_slice_curl_registered": False,
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _expected_body(
    root: Path, config_path: Path, values: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    witness = _generic_ambiguity_witness()
    audited_keys = set().union(*(_recursive_keys(value) for value in values.values()))
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "reverse_Pother_by_P10_values_not_identifiable_from_registered_source_map_"
            "schemas_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "predecessor_schema_keys_audited": len(audited_keys),
            "registered_typed_coordinate_to_Einstein_maps": 0,
            "reverse_ordered_pair_cells_targeted_per_candidate": 810,
            "reverse_output_entries_targeted_per_candidate": 8910,
            "reverse_output_entries_materialized": 0,
            "two_map_nonidentifiability_witnesses": 12,
            "witnesses_with_distinct_values": 12,
            "restricted_reverse_connection_nonzero_corrections": 0,
            "corrected_cross_slice_curl_certificates": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "schema_audit": {
            "predecessor_labels": sorted(values),
            "exact_map_keys_required": sorted(FORBIDDEN_MAP_KEYS),
            "exact_map_keys_found": [],
            "audit_result": "typed_coordinate_to_Einstein_derivative_map_absent",
        },
        "nonidentifiability_theorem": {
            "name": "reverse_principal_source_derivative_not_identifiable_without_typed_map",
            "source_formula": (
                "partial_g(-A^-1 C_11)="
                "A^-1(partial_g A)A^-1 C_11-A^-1(partial_g C_11)"
            ),
            "generic_witness": witness,
            "map_zero": "partial_Gmunu_over_partial_s11_field0=0_for_all_components",
            "map_alternative": "partial_G00_over_partial_s11_field0=1_others_zero",
            "compatibility_boundary": (
                "Neither map mutates or contradicts a predecessor field because the typed map "
                "is absent from every exact predecessor schema."
            ),
            "conclusion": (
                "The two unregistered maps produce 0 and -2*alpha at the same reverse D2F "
                "coordinate. Since alpha is nonzero for every candidate, current evidence does "
                "not determine that coordinate and cannot materialize the 8910-entry reverse slice."
            ),
        },
        "candidate_records": _candidate_records(values),
        "exact_controls": {
            "assume_zero_typed_map": {
                "rejected": True,
                "reason": "zero_map_not_registered",
            },
            "assume_alternative_typed_map": {
                "rejected": True,
                "reason": "alternative_map_not_registered",
            },
            "infer_physical_no_go_for_all_covariant_maps": {
                "rejected": True,
                "reason": "only_current_evidence_identifiability_is_classified",
            },
            "infer_corrected_cross_slice_curl": {
                "rejected": True,
                "reverse_values_materialized": 0,
            },
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "candidate_bound_chain_rule_from_metric_second_atoms_to_Einstein_components_missing",
            "general_output_connection_rows_B0_through_B9_not_registered",
            "remaining_other_principal_by_other_principal_values_not_registered",
            "complete_high_atom_good_unknown_identity_not_registered",
            "induced_TC1_TC2_TC3_TC5_B7_global_H7_PDE_and_lifespan_not_closed",
        ],
        "claim_seals": CLAIM_SEALS,
        "data_seals": EXPECTED_SEALS,
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            "direct_dependencies": EXPECTED_DIRECT_DEPENDENCIES,
            **EXPECTED_PREDECESSORS,
        },
        "scope": (
            "candidate-bound exact nonidentifiability classification for the reverse 11x90x9 "
            "Pother-by-P10 source slice under current registered schemas; no physical no-go for "
            "all covariant maps, reverse materialization, corrected curl, full D2F, high-atom "
            "identity, global H7, PDE, lifespan, candidate rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("reverse source-map bindings missing")
    if set(bindings) != {
        "source",
        "config",
        "test",
        "direct_dependencies",
        *EXPECTED_PREDECESSORS,
    }:
        raise ValueError("reverse source-map source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        binding = bindings.get(label)
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError("reverse source-map local binding changed")
        if binding.get("file_sha256") != _file_sha(_inside(root, relative)):
            raise ValueError("reverse source-map local binding hash changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("reverse source-map predecessor binding changed")
    _validate_direct_dependencies(root, bindings.get("direct_dependencies", {}))


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("reverse source-map content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _validate_direct_dependencies(validation_root, config["direct_dependencies"])
    predecessors = {
        label: _load_bound(validation_root, binding)
        for label, binding in EXPECTED_PREDECESSORS.items()
    }
    _validate_predecessors(predecessors, validation_root)
    expected = _expected_body(validation_root, config_path, predecessors)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("reverse source-map result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    _validate_direct_dependencies(root, config["direct_dependencies"])
    predecessors = {
        label: _load_bound(root, binding) for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors, root)
    body = _expected_body(root, config_path, predecessors)
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
