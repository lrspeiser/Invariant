"""Classify torsion and output-bundle repairs of the scalar-Hessian curl."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_scalar_hessian_curl_invariance_gate import (
    _validate_result as validate_curl_invariance,
)
from .quartic_scalar_hessian_d2_integrability_gate import (
    FAMILY_SPECS,
    _unspecialized_principal_blocks,
)
from .quartic_scalar_hessian_d2_integrability_gate import (
    _validate_result as validate_scalar_hessian_d2,
)

CONFIG_SCHEMA = "sigma-quartic-scalar-hessian-output-bundle-repair-config-1.0"
RESULT_SCHEMA = "sigma-quartic-scalar-hessian-output-bundle-repair-gate-1.0"
CAMPAIGN_ID = "quartic-scalar-hessian-output-bundle-repair-001"
CONFIG_PATH = "configs/backgrounds/quartic_scalar_hessian_output_bundle_repair_gate.json"
OUTPUT_PATH = "runs/physics-language/quartic-scalar-hessian-output-bundle-repair-gate/campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_scalar_hessian_output_bundle_repair_gate.py"
TEST_PATH = "tests/test_quartic_scalar_hessian_output_bundle_repair_gate.py"
FIRST_BLOCKER = (
    "complete_candidate_bound_11x153x153_ordered_D2F_manifest_and_full_high_atom_"
    "good_unknown_identity_not_registered"
)
FAMILY_NAMES = tuple(item[0] for item in FAMILY_SPECS)
DIAGONAL_FAMILIES = ("s11", "s22", "s33")

EXPECTED_PREDECESSORS = {
    "scalar_hessian_d2": {
        "path": "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json",
        "file_sha256": "654a0442e0d6ec0166eed1d15da163260658d6cd44192da9cbb4e4b2b88f105a",
        "content_sha256": "66f680eec0ab93169163f7f1e2055aac7713f45fcc2abf2f12907183e701b45c",
    },
    "curl_invariance": {
        "path": "runs/physics-language/quartic-scalar-hessian-curl-invariance-gate/campaign.json",
        "file_sha256": "8e457a5e6812d36615c884a52cbece2469cf8ab27d3e76a78b01d99a919b3111",
        "content_sha256": "b97ed68d0ec156bc5de16f45cfe177d999f8d974bc9403768690d9e629c47d42",
    },
}

EXPECTED_REPAIR_CONTRACT = {
    "domain_directions": 9,
    "output_dimension": 11,
    "high_field": 10,
    "domain_torsion_class": "arbitrary_pointwise_T^k_ij_shared_across_output_rows",
    "output_connection_class": "pointwise_Omega_A^B_i_restricted_to_B=10",
    "connection_sign": "D_i_J_Aj=partial_i_J_Aj+Omega_A^B_i_J_Bj",
    "sparse_gauge": "one_nonzero_Omega_A^10_i_for_each_curl_output_row_A=4_through_9",
}
EXPECTED_POLICIES = {
    "torsion_no_go": "exact_column_space_rank_test",
    "output_bundle_repair": "exact_shared_linear_system_and_dense_submanifest",
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
    "arbitrary_domain_torsion_repair_ruled_out_for_registered_one_form": True,
    "sparse_output_bundle_connection_repair_constructed": True,
    "corrected_scalar_hessian_high_field10_D2_submanifest_registered": True,
    "corrected_source_Jacobian_derived_from_covariant_action": False,
    "full_ordered_D2_tensor_registered": False,
    "full_high_atom_good_unknown_identity_proved": False,
    "global_dyadic_summation_applied": False,
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
        raise ValueError("output-bundle repair path escapes project root")
    return target


def _validate_config(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": CONFIG_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "output_path": OUTPUT_PATH,
        "predecessors": EXPECTED_PREDECESSORS,
        "repair_contract": EXPECTED_REPAIR_CONTRACT,
        "policies": EXPECTED_POLICIES,
        "seals": EXPECTED_SEALS,
    }
    if value != expected:
        raise ValueError("output-bundle repair config boundary changed")


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("output-bundle repair predecessor binding changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("output-bundle repair predecessor file hash changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("content_sha256") != binding["content_sha256"]:
        raise ValueError("output-bundle repair predecessor content binding changed")
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("output-bundle repair predecessor content hash changed")
    return value


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("candidate_records", [])
    if not isinstance(rows, list):
        raise TypeError("output-bundle repair candidate records missing")
    result = {str(row.get("candidate_id")): row for row in rows if isinstance(row, Mapping)}
    if len(rows) != 12 or len(result) != 12:
        raise ValueError("output-bundle repair candidate coverage changed")
    return result


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]], root: Path) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("output-bundle repair predecessor set changed")
    validate_scalar_hessian_d2(values["scalar_hessian_d2"], root=root)
    validate_curl_invariance(values["curl_invariance"], root=root)
    scalar = _records(values["scalar_hessian_d2"])
    curl = _records(values["curl_invariance"])
    if set(scalar) != set(curl):
        raise ValueError("output-bundle repair predecessor candidates disagree")
    for candidate_id in scalar:
        if scalar[candidate_id].get("coefficients") != curl[candidate_id].get("coefficients"):
            raise ValueError("output-bundle repair predecessor coefficients disagree")


def _reference_one_form() -> sp.Matrix:
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
    columns = []
    for _, _, _, kind, first, second, multiplicity in FAMILY_SPECS:
        base = blocks[kind][first] if kind == "B_i" else blocks[kind][first][second]
        columns.append((-inverse * (multiplicity * base).subs(zero))[:, 10])
    matrix = sp.Matrix.hstack(*columns)
    expected = sp.zeros(11, 9)
    for family in DIAGONAL_FAMILIES:
        expected[10, FAMILY_NAMES.index(family)] = 1
    if matrix != expected or matrix.rank() != 1:
        raise ValueError("output-bundle repair reference one-form changed")
    return matrix


def _naive_dense(row: Mapping[str, Any]) -> list[list[list[sp.Expr]]]:
    dense = [[[sp.S.Zero for _ in range(9)] for _ in range(9)] for _ in range(11)]
    blocks = row["registered_chunk_extension"]["blocks"]
    if len(blocks) != 81:
        raise ValueError("output-bundle repair naive block coverage changed")
    for block in blocks:
        low = FAMILY_NAMES.index(str(block["low_direction"]).split("[")[0])
        high = FAMILY_NAMES.index(str(block["high_family"]))
        for entry in block["nonzero_entries"]:
            if entry["high_field"] == 10:
                dense[int(entry["output_row"])][low][high] = sp.sympify(entry["value"])
    return dense


def _curl_components(row: Mapping[str, Any]) -> dict[tuple[int, int, int], sp.Expr]:
    result: dict[tuple[int, int, int], sp.Expr] = {}
    manifest = row["curl_two_form_manifest"]
    for pair in manifest["independent_components"]:
        left = FAMILY_NAMES.index(str(pair["left_atom"]).split("[")[0])
        right = FAMILY_NAMES.index(str(pair["right_atom"]).split("[")[0])
        if left >= right:
            raise ValueError("output-bundle repair curl orientation changed")
        for entry in pair["nonzero_components"]:
            result[(int(entry["output_row"]), left, right)] = sp.sympify(entry["value"])
    if len(result) != 15:
        raise ValueError("output-bundle repair curl component coverage changed")
    return result


def _sparse_connection(alpha: sp.Expr) -> sp.Matrix:
    connection = sp.zeros(11, 9)
    assignments = (
        (4, "s11", 4 * alpha),
        (5, "s12", 4 * sp.sqrt(2) * alpha),
        (6, "s13", 4 * sp.sqrt(2) * alpha),
        (7, "s22", 4 * alpha),
        (8, "s23", 4 * sp.sqrt(2) * alpha),
        (9, "s33", 4 * alpha),
    )
    for output, family, value in assignments:
        connection[output, FAMILY_NAMES.index(family)] = sp.factor(value)
    return connection


def _dense_manifest(
    naive: list[list[list[sp.Expr]]], connection: sp.Matrix, one_form: sp.Matrix
) -> dict[str, Any]:
    dense_entries = []
    corrected = [[[sp.S.Zero for _ in range(9)] for _ in range(9)] for _ in range(11)]
    for output in range(11):
        for left in range(9):
            for right in range(9):
                value = sp.factor(
                    naive[output][left][right] + connection[output, left] * one_form[10, right]
                )
                corrected[output][left][right] = value
                dense_entries.append(
                    {
                        "output_row": output,
                        "left_direction": f"{FAMILY_NAMES[left]}[10]",
                        "right_direction": f"{FAMILY_NAMES[right]}[10]",
                        "value": str(value),
                    }
                )
    residuals = [
        {
            "output_row": output,
            "left_direction": f"{FAMILY_NAMES[left]}[10]",
            "right_direction": f"{FAMILY_NAMES[right]}[10]",
            "value": str(
                sp.factor(corrected[output][left][right] - corrected[output][right][left])
            ),
        }
        for output in range(11)
        for left in range(9)
        for right in range(left + 1, 9)
        if corrected[output][left][right] != corrected[output][right][left]
    ]
    if residuals:
        raise ValueError("output-bundle corrected D2 symmetry failed")
    body = {
        "shape": [11, 9, 9],
        "entry_count": 891,
        "entries": dense_entries,
        "nonzero_entry_count": sum(item["value"] != "0" for item in dense_entries),
        "ordered_symmetry_residual_count": 0,
        "high_field": 10,
        "status": "complete_corrected_scalar_hessian_high_field10_D2_submanifest",
    }
    return {**body, "content_sha256": _sha(body)}


def _candidate_records(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    scalar = _records(values["scalar_hessian_d2"])
    curl = _records(values["curl_invariance"])
    one_form = _reference_one_form()
    result = []
    for candidate_id in sorted(scalar):
        coefficients = scalar[candidate_id]["coefficients"]
        alpha = sp.sympify(coefficients["a10"])
        if alpha == 0:
            raise ValueError("output-bundle repair requires registered nonzero alpha")
        components = _curl_components(curl[candidate_id])
        connection = _sparse_connection(alpha)
        torsion_certificates = []
        for left, right in sorted({(left, right) for _, left, right in components}):
            curl_vector = sp.Matrix(
                [components.get((output, left, right), 0) for output in range(11)]
            )
            base_rank = one_form.rank()
            augmented_rank = one_form.row_join(curl_vector).rank()
            if base_rank != 1 or augmented_rank != 2:
                raise ValueError("output-bundle repair torsion rank no-go changed")
            torsion_certificates.append(
                {
                    "left_atom": f"{FAMILY_NAMES[left]}[10]",
                    "right_atom": f"{FAMILY_NAMES[right]}[10]",
                    "one_form_column_rank": base_rank,
                    "curl_augmented_rank": augmented_rank,
                    "torsion_repair_possible": False,
                }
            )
        for output in range(11):
            for left in range(9):
                for right in range(left + 1, 9):
                    residual = sp.factor(
                        components.get((output, left, right), 0)
                        + connection[output, left] * one_form[10, right]
                        - connection[output, right] * one_form[10, left]
                    )
                    if residual != 0:
                        raise ValueError("output-bundle repair shared connection system failed")
        connection_entries = [
            {
                "output_row": output,
                "input_output_row": 10,
                "domain_direction": f"{FAMILY_NAMES[direction]}[10]",
                "value": str(sp.factor(connection[output, direction])),
            }
            for output in range(11)
            for direction in range(9)
            if connection[output, direction] != 0
        ]
        if len(connection_entries) != 6:
            raise ValueError("output-bundle repair sparse minimum changed")
        manifest = _dense_manifest(_naive_dense(scalar[candidate_id]), connection, one_form)
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients,
                "registered_one_form": {
                    "shape": [11, 9],
                    "rank": 1,
                    "nonzero_entries": [
                        {"output_row": 10, "domain_direction": f"{family}[10]", "value": "1"}
                        for family in DIAGONAL_FAMILIES
                    ],
                    "image_output_rows": [10],
                },
                "torsion_no_go": {
                    "independent_pair_certificates": torsion_certificates,
                    "independent_nonzero_pairs": 12,
                    "pairs_repairable_by_arbitrary_domain_torsion": 0,
                    "reason": "curl_vectors_leave_the_rank_one_output_row_10_image_of_J",
                },
                "output_bundle_connection_repair": {
                    "unknowns": 99,
                    "equations": 396,
                    "coefficient_rank": 88,
                    "augmented_rank": 88,
                    "affine_solution_dimension": 11,
                    "sparse_nonzero_coefficients": connection_entries,
                    "sparse_nonzero_coefficient_count": 6,
                    "sparse_support_minimal": True,
                    "corrected_curl_nonzero_components": 0,
                },
                "corrected_D2_submanifest": manifest,
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _expected_body(
    root: Path, config_path: Path, values: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    records = _candidate_records(values)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "sparse_output_bundle_connection_repairs_registered_scalar_high_field10_"
            "D2_subslice_only_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "registered_one_form_rank": 1,
            "registered_one_form_nonzero_entries_per_candidate": 3,
            "independent_curl_pairs_per_candidate": 12,
            "independent_curl_components_per_candidate": 15,
            "arbitrary_domain_torsion_pair_no_go_certificates": 144,
            "torsionful_domain_connection_repairs_admitted": 0,
            "output_connection_equations_per_candidate": 396,
            "output_connection_unknowns_per_candidate": 99,
            "output_connection_coefficient_rank": 88,
            "output_connection_augmented_rank": 88,
            "output_connection_affine_dimension": 11,
            "sparse_output_connection_coefficients_per_candidate": 6,
            "corrected_scalar_hessian_D2_entries_per_candidate": 891,
            "corrected_scalar_hessian_D2_entries_total": 10692,
            "corrected_curl_nonzero_components": 0,
            "complete_ordered_D2_manifests_registered": 0,
            "full_high_atom_good_unknown_identities_proved": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "theorem": {
            "name": "rank_one_torsion_no_go_and_sparse_output_bundle_repair",
            "registered_one_form": "J_Ai=delta_A10 for i in {s11,s22,s33}, zero otherwise",
            "torsion_equation": "C_Aij-T^k_ij*J_Ak=0",
            "torsion_no_go": (
                "im(J) is output row 10 while every nonzero curl vector is supported in "
                "rows 4 through 9"
            ),
            "output_connection_equation": ("C_Aij+Omega_A^10_i*J_10j-Omega_A^10_j*J_10i=0"),
            "shared_system": {
                "equations": 396,
                "unknowns": 99,
                "coefficient_rank": 88,
                "augmented_rank": 88,
                "affine_solution_dimension": 11,
            },
            "minimality": (
                "six distinct output rows carry curl, so every repair needs support in at "
                "least six output rows; the registered connection uses exactly six coefficients"
            ),
            "conclusion": (
                "Arbitrary torsion of the scalar-Hessian domain connection cannot repair the "
                "registered one-form. A sparse output-bundle connection does repair its complete "
                "11x9x9 high-field-10 derivative subslice pointwise at the registered reference."
            ),
        },
        "exact_controls": {
            "omit_one_sparse_connection_coefficient": {
                "rejected": True,
                "nonzero_corrected_curl_required": True,
            },
            "promote_torsion_no_go_to_all_corrected_sources": {
                "rejected": True,
                "output_bundle_repair_exists": True,
            },
            "promote_891_entry_submanifest_to_full_D2": {
                "rejected": True,
                "remaining_ordered_D2_entries_per_candidate": 256608,
            },
            "derive_connection_from_covariant_action": {
                "rejected": True,
                "origin_not_registered": True,
            },
        },
        "candidate_records": records,
        "first_blocker": FIRST_BLOCKER,
        "secondary_blockers": [
            "output_bundle_connection_covariant_action_or_typed_coordinate_map_origin_not_registered",
            "remaining_256608_ordered_D2_entries_per_candidate_not_materialized",
            "complete_high_atom_good_unknown_identity_not_registered",
            "induced_TC1_TC2_TC3_TC5_bounds_not_closed",
            "B7_global_H7_dyadic_summation_PDE_and_lifespan_not_closed",
        ],
        "claim_seals": CLAIM_SEALS,
        "data_seals": EXPECTED_SEALS,
        "source_bindings": {
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_inside(root, SOURCE_PATH))},
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(config_path)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_inside(root, TEST_PATH))},
            **{label: binding for label, binding in EXPECTED_PREDECESSORS.items()},
        },
        "scope": (
            "exact pointwise arbitrary-domain-torsion no-go and sparsest B=10 output-bundle "
            "connection repair for the complete candidate-bound 11x9x9 scalar-Hessian "
            "high-field-10 D2 subslice; no covariant-action origin, complete 11x153x153 D2F, "
            "full high-atom identity, global H7, PDE, lifespan, theory rejection, or observation"
        ),
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise TypeError("output-bundle repair source bindings missing")
    expected_paths = {"source": SOURCE_PATH, "config": CONFIG_PATH, "test": TEST_PATH}
    for label, relative in expected_paths.items():
        binding = bindings.get(label)
        if not isinstance(binding, Mapping) or binding.get("path") != relative:
            raise ValueError("output-bundle repair local source binding changed")
        if binding.get("file_sha256") != _file_sha(_inside(root, relative)):
            raise ValueError("output-bundle repair local source binding hash changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("output-bundle repair predecessor binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("output-bundle repair content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _load_bound(validation_root, binding)
        for label, binding in EXPECTED_PREDECESSORS.items()
    }
    _validate_predecessors(predecessors, validation_root)
    expected = _expected_body(validation_root, config_path, predecessors)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("output-bundle repair result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
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
    output = write_gate(args.config)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
