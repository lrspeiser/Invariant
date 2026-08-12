"""Project the registered Pother one-form and solve its two-sided reference system."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_cross_slice_two_sided_connection_identifiability_gate import (
    _validate_result as validate_two_sided_identifiability,
)
from .quartic_principal_high_atom_connection_extension_gate import (
    FAMILY_NAMES,
    _generic_principal_source,
)
from .quartic_row0_arithmetic_expansion_campaign import _content_hash_matches

CONFIG_SCHEMA = "sigma-quartic-candidate-pother-one-form-connection-config-1.0"
RESULT_SCHEMA = "sigma-quartic-candidate-pother-one-form-connection-gate-1.0"
CAMPAIGN_ID = "quartic-candidate-pother-one-form-connection-001"
CONFIG_PATH = "configs/backgrounds/quartic_candidate_pother_one_form_connection_gate.json"
OUTPUT_PATH = (
    "runs/physics-language/quartic-candidate-pother-one-form-connection-gate/campaign.json"
)
SOURCE_PATH = "src/sigma_theory_compiler/quartic_candidate_pother_one_form_connection_gate.py"
TEST_PATH = "tests/test_quartic_candidate_pother_one_form_connection_gate.py"
FIRST_BLOCKER = (
    "covariant_action_derived_output_connection_or_corrected_source_jet_required_before_"
    "cross_slice_D2F_admission_despite_reference_system_rank_1870_consistency"
)
P10 = tuple(f"{family}[10]" for family in FAMILY_NAMES)
POTHER = tuple(f"{family}[{field}]" for family in FAMILY_NAMES for field in range(10))

EXPECTED_PREDECESSORS = {
    "two_sided_connection_identifiability": {
        "path": (
            "runs/physics-language/quartic-cross-slice-two-sided-connection-identifiability-gate/"
            "campaign.json"
        ),
        "file_sha256": "c3ecb5b65a716d5c55ede9d165880e04614f525dce43d49a0f3d9b996a6a9a5a",
        "content_sha256": "71876e080a30eb3ec3f1b066c4107911a344d9f5005a2f480a7a35275101e17e",
    }
}
EXPECTED_DIRECT_EVIDENCE = {
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
    "principal_source_replay": {
        "path": (
            "src/sigma_theory_compiler/"
            "quartic_principal_high_atom_connection_extension_gate.py"
        ),
        "file_sha256": "a0c54de525ee10d9aaa0b03bc66737b94cfada3f84155b6f43dac97997ee3df7",
    },
}
EXPECTED_CONTRACT = {
    "registered_source_Jacobian_shape": [11, 153],
    "projected_Pother_one_form_shape": [11, 90],
    "two_sided_equation": "C_Aij+Omega_A^B_i*J_Bj-Omega_A^B_j*K_Bi=0",
    "equations": 8910,
    "connection_unknowns": 11979,
    "connection_solution_gauge": "all_free_variables_zero",
    "admission": "forbid_algebraically_fitted_connection_without_covariant_origin",
}
EXPECTED_POLICIES = {
    "Pother_one_form": "require_candidate_bound_full_source_Jacobian_replay",
    "connection_system": "exact_rational_radical_linear_solve",
    "cross_slice_admission": "fail_closed_without_covariant_connection_origin",
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
    "candidate_bound_Pother_one_form_registered": True,
    "full_source_Jacobian_Pother_slice_projected": True,
    "two_sided_reference_connection_system_consistent": True,
    "algebraic_reference_connection_solution_constructed": True,
    "algebraic_two_sided_residual_zero": True,
    "connection_derived_from_covariant_action": False,
    "corrected_source_extension_registered": False,
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
        raise ValueError("candidate Pother one-form path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "direct_evidence": EXPECTED_DIRECT_EVIDENCE,
        "contract": EXPECTED_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("candidate Pother one-form config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("candidate Pother one-form artifact binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("candidate Pother one-form artifact file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("candidate Pother one-form artifact content binding changed")
    return value


def _records(value: Mapping[str, Any], key: str = "candidate_records") -> dict[str, Mapping[str, Any]]:
    raw = value.get(key, [])
    if not isinstance(raw, list):
        raise TypeError("candidate Pother one-form records missing")
    result = {str(row.get("candidate_id")): row for row in raw if isinstance(row, Mapping)}
    if len(raw) != 12 or len(result) != 12:
        raise ValueError("candidate Pother one-form candidate set changed")
    return result


def _validate_direct_evidence(root: Path) -> dict[str, Any]:
    bundle = EXPECTED_DIRECT_EVIDENCE["full_source_jacobian"]
    for label in ("source", "config", "test"):
        binding = bundle[label]
        path = _inside(root, binding["path"])
        if set(binding) != {"path", "file_sha256"} or _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"candidate Pother one-form direct {label} binding changed")
    replay = EXPECTED_DIRECT_EVIDENCE["principal_source_replay"]
    if set(replay) != {"path", "file_sha256"} or _file_sha(
        _inside(root, replay["path"])
    ) != replay["file_sha256"]:
        raise ValueError("candidate Pother one-form principal replay binding changed")
    artifact = _load_bound(root, bundle["artifact"])
    if not _content_hash_matches(artifact):
        raise ValueError("candidate Pother one-form full source content replay changed")
    if artifact.get("status") != (
        "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed"
    ) or artifact.get("counts") != {
        "H7_closures": 0,
        "component_remainders_proved": 0,
        "full_source_entries_per_candidate": 1683,
        "full_source_jacobians_materialized": 12,
        "lower_entries_per_candidate": 594,
        "principal_entries_per_candidate": 1089,
        "rejected": 0,
        "selected": 12,
    }:
        raise ValueError("candidate Pother one-form full source boundary changed")
    manifest = artifact.get("common_full_entry_manifest", {})
    entries = manifest.get("entries", [])
    projected = {
        (int(item["source_row"]), str(item["coordinate_atom"]))
        for item in entries
        if item.get("family") == "principal" and item.get("coordinate_atom") in POTHER
    }
    if manifest.get("shape") != [11, 153] or len(projected) != 990 or projected != {
        (row, atom) for row in range(11) for atom in POTHER
    }:
        raise ValueError("candidate Pother one-form source projection coverage changed")
    return artifact


def _validate_predecessor(value: Mapping[str, Any], source: Mapping[str, Any], root: Path) -> None:
    validate_two_sided_identifiability(value, root=root)
    prior = _records(value)
    source_records = _records(source, "certificates")
    if set(prior) != set(source_records):
        raise ValueError("candidate Pother one-form predecessor candidates disagree")
    for candidate_id, row in prior.items():
        if source_records[candidate_id].get("coefficients") != row.get("coefficients"):
            raise ValueError("candidate Pother one-form predecessor coefficients disagree")


@cache
def _solve_packet(alpha_text: str, c20_text: str, curl_packet: str) -> dict[str, Any]:
    source = _generic_principal_source()
    substitutions = {
        source["alpha"]: sp.sympify(alpha_text),
        source["c20"]: sp.sympify(c20_text),
    }
    registered = sp.zeros(11, 9)
    pother = sp.zeros(11, 90)
    for family_index, family in enumerate(FAMILY_NAMES):
        registered[:, family_index] = source["chunks"][family][:, 10].subs(substitutions)
        for field in range(10):
            pother[:, family_index * 10 + field] = source["chunks"][family][
                :, field
            ].subs(substitutions)
    if registered.rank() != 1 or pother.rank() != 10:
        raise ValueError("candidate Pother one-form ranks changed")

    curl = json.loads(curl_packet)
    rhs = sp.zeros(810, 1)
    for entry in curl:
        if int(entry["output_row"]) != 10:
            raise ValueError("candidate Pother one-form curl output support changed")
        row = POTHER.index(str(entry["left_atom"])) * 9 + P10.index(
            str(entry["right_atom"])
        )
        rhs[row] = -sp.sympify(entry["value"])
    if sum(item != 0 for item in rhs) != 63:
        raise ValueError("candidate Pother one-form curl support changed")

    matrix = sp.zeros(810, 1089)
    for pother_index in range(90):
        for p10_index in range(9):
            equation = pother_index * 9 + p10_index
            for input_row in range(11):
                matrix[equation, pother_index * 11 + input_row] = registered[
                    input_row, p10_index
                ]
                matrix[equation, 990 + p10_index * 11 + input_row] = -pother[
                    input_row, pother_index
                ]
    solution, _, free_columns = matrix.gauss_jordan_solve(rhs, freevar=True)
    free_symbols = {symbol for symbol in solution.free_symbols if str(symbol).startswith("tau")}
    solution = solution.xreplace({symbol: 0 for symbol in free_symbols})
    residual = matrix * solution - rhs
    if len(free_columns) != 919 or any(residual):
        raise ValueError("candidate Pother one-form exact solve changed")

    one_form_entries = [
        {"output_row": row, "Pother_atom": POTHER[column], "value": str(sp.factor(value))}
        for row in range(11)
        for column in range(90)
        if (value := pother[row, column]) != 0
    ]
    pother_connection = [
        {
            "output_row": 10,
            "input_row": input_row,
            "Pother_atom": POTHER[pother_index],
            "value": str(sp.factor(value)),
        }
        for pother_index in range(90)
        for input_row in range(11)
        if (value := solution[pother_index * 11 + input_row]) != 0
    ]
    p10_connection = [
        {
            "output_row": 10,
            "input_row": input_row,
            "P10_atom": P10[p10_index],
            "value": str(sp.factor(value)),
        }
        for p10_index in range(9)
        for input_row in range(11)
        if (value := solution[990 + p10_index * 11 + input_row]) != 0
    ]
    if len(one_form_entries) != 93 or len(pother_connection) != 15 or len(p10_connection) != 7:
        raise ValueError("candidate Pother one-form sparse solve changed")
    return {
        "candidate_bound_Pother_one_form": {
            "shape": [11, 90],
            "rank": 10,
            "entry_count": 990,
            "nonzero_entry_count": 93,
            "nonzero_entries": one_form_entries,
            "dense_content_sha256": _sha([str(item) for item in pother]),
            "source": "registered_full_11x153_solved_source_Jacobian_principal_slice",
        },
        "two_sided_reference_system": {
            "equations": 8910,
            "unknowns": 11979,
            "coefficient_rank": 1870,
            "augmented_rank": 1870,
            "affine_solution_dimension": 10109,
            "consistent": True,
        },
        "free_variable_zero_connection_witness": {
            "algebraic_not_covariant_action_derived": True,
            "Pother_direction_nonzero_count": 15,
            "Pother_direction_nonzero_entries": pother_connection,
            "P10_direction_nonzero_count": 7,
            "P10_direction_nonzero_entries": p10_connection,
            "total_nonzero_count": 22,
            "equations_checked": 8910,
            "nonzero_residuals": 0,
            "residual_dense_content_sha256": _sha(["0"] * 8910),
        },
    }


def _candidate_records(predecessor: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = []
    for candidate_id, row in sorted(_records(predecessor).items()):
        coefficients = row["coefficients"]
        synthetic = row["rank_six_completion_witness"]
        synthetic_one_form = sp.zeros(11, 90)
        for entry in synthetic["Pother_one_form"]["nonzero_entries"]:
            synthetic_one_form[int(entry["input_row"]), POTHER.index(entry["Pother_atom"])] = (
                sp.sympify(entry["value"])
            )
        synthetic_connection = sp.zeros(99, 11)
        for entry in synthetic["P10_connection_variation"]["nonzero_entries"]:
            matrix_row = int(entry["output_row"]) * 9 + P10.index(entry["P10_atom"])
            synthetic_connection[matrix_row, int(entry["input_row"])] = sp.sympify(
                entry["value"]
            )
        curl_matrix = synthetic_connection * synthetic_one_form
        if _sha([str(item) for item in curl_matrix]) != row["curl_flattening"][
            "dense_content_sha256"
        ]:
            raise ValueError("candidate Pother one-form predecessor curl replay changed")
        curl_entries = [
            {
                "output_row": matrix_row // 9,
                "left_atom": POTHER[pother_index],
                "right_atom": P10[matrix_row % 9],
                "value": str(sp.factor(value)),
            }
            for matrix_row in range(99)
            for pother_index in range(90)
            if (value := curl_matrix[matrix_row, pother_index]) != 0
        ]
        if len(curl_entries) != 63:
            raise ValueError("candidate Pother one-form predecessor curl count changed")
        packet = _solve_packet(
            str(coefficients["a10"]),
            str(coefficients["c20"]),
            json.dumps(curl_entries, sort_keys=True, separators=(",", ":")),
        )
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients,
                "predecessor_curl_dense_content_sha256": row["curl_flattening"][
                    "dense_content_sha256"
                ],
                **packet,
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
    source_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    records = _candidate_records(predecessor)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "candidate_bound_Pother_one_form_selects_consistent_two_sided_reference_system_"
            "algebraic_connection_origin_not_registered_D2F_fail_closed"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "full_source_Jacobians_bound": 12,
            "Pother_one_form_entries_per_candidate": 990,
            "Pother_one_form_nonzero_entries_per_candidate": 93,
            "Pother_one_form_rank_per_candidate": 10,
            "two_sided_equations_per_candidate": 8910,
            "two_sided_connection_unknowns_per_candidate": 11979,
            "two_sided_coefficient_rank_per_candidate": 1870,
            "two_sided_augmented_rank_per_candidate": 1870,
            "two_sided_affine_solution_dimension_per_candidate": 10109,
            "consistent_two_sided_reference_systems": 12,
            "algebraic_connection_nonzero_entries_per_candidate": 22,
            "algebraic_connection_residual_nonzero_entries": 0,
            "covariant_action_derived_connections": 0,
            "cross_slice_entries_admitted": 0,
            "principal_high_atom_entries_missing_per_candidate": 106920,
            "complete_ordered_D2F_tensors_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "projection_theorem": {
            "name": "candidate_bound_Pother_source_one_form_two_sided_consistency",
            "registered_premise": (
                "The provenance-bound complete 11x153 solved-source Jacobian contains all 990 "
                "entries of its 11x90 principal Pother slice."
            ),
            "exact_result": (
                "For each candidate the projected one-form has rank 10 and 93 nonzero entries. "
                "The 8910-equation/11979-unknown two-sided reference connection system has "
                "coefficient and augmented rank 1870 and affine dimension 10109."
            ),
            "witness": (
                "Setting all free variables to zero yields 15 nonzero Pother-direction and seven "
                "nonzero P10-direction coefficients and zero residual in all 8910 equations."
            ),
            "boundary": (
                "This selects consistency over the predecessor K=0 no-go, but the fitted output "
                "connection is not derived from a covariant action or corrected source jet and "
                "therefore does not register a physical connection or any D2F entry."
            ),
        },
        "candidate_records": records,
        "exact_controls": {
            "promote_algebraic_fit_to_covariant_connection": {"rejected": True},
            "promote_reference_consistency_to_D2F": {
                "rejected": True,
                "cross_slice_entries_admitted": 0,
            },
            "promote_cross_slice_to_complete_D2F": {"rejected": True},
            "reject_candidates_from_missing_connection_origin": {"rejected": True},
        },
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "corrected_source_jet_not_registered",
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
        "source_evidence_content_sha256": source_evidence["content_sha256"],
        "scope": (
            "candidate-bound projection of the registered full source Jacobian to the 11x90 "
            "Pother one-form and exact consistency of the reference two-sided connection system; "
            "no covariant connection origin, corrected source jet, D2F admission, complete D2F, "
            "high-atom identity, H7, PDE, lifespan, rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("candidate Pother one-form source bindings missing")
    if set(bindings) != {"source", "config", "test", "direct_evidence", *EXPECTED_PREDECESSORS}:
        raise ValueError("candidate Pother one-form source binding keys changed")
    for label, relative in {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}.items():
        if bindings[label] != {
            "path": relative,
            "file_sha256": _file_sha(_inside(root, relative)),
        }:
            raise ValueError("candidate Pother one-form local binding changed")
    if bindings["direct_evidence"] != EXPECTED_DIRECT_EVIDENCE:
        raise ValueError("candidate Pother one-form direct evidence binding changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings[label] != expected:
            raise ValueError("candidate Pother one-form predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("candidate Pother one-form content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source_evidence = _validate_direct_evidence(validation_root)
    predecessor = _load_bound(
        validation_root, EXPECTED_PREDECESSORS["two_sided_connection_identifiability"]
    )
    if predecessor.get("content_sha256") != _content_sha(predecessor):
        raise ValueError("candidate Pother one-form predecessor content changed")
    _validate_predecessor(predecessor, source_evidence, validation_root)
    expected = _expected_body(validation_root, config_path, predecessor, source_evidence)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("candidate Pother one-form result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    source_evidence = _validate_direct_evidence(root)
    predecessor = _load_bound(root, config["predecessors"]["two_sided_connection_identifiability"])
    if predecessor.get("content_sha256") != _content_sha(predecessor):
        raise ValueError("candidate Pother one-form predecessor content changed")
    _validate_predecessor(predecessor, source_evidence, root)
    body = _expected_body(root, config_path, predecessor, source_evidence)
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
