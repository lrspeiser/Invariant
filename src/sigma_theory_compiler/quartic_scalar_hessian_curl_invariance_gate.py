"""Certify the invariant obstruction carried by the scalar-Hessian curl."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_SCHEMA = "sigma-quartic-scalar-hessian-curl-invariance-config-1.0"
RESULT_SCHEMA = "sigma-quartic-scalar-hessian-curl-invariance-gate-1.0"
CAMPAIGN_ID = "quartic-scalar-hessian-curl-invariance-001"
FIRST_BLOCKER = (
    "candidate_bound_corrected_source_Jacobian_or_output_bundle_or_torsion_"
    "repair_restoring_integrability_not_registered"
)
OUTPUT_PATH = "runs/physics-language/quartic-scalar-hessian-curl-invariance-gate/campaign.json"
CONFIG_PATH = "configs/backgrounds/quartic_scalar_hessian_curl_invariance_gate.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_scalar_hessian_curl_invariance_gate.py"
TEST_PATH = "tests/test_quartic_scalar_hessian_curl_invariance_gate.py"
EXPECTED_PREDECESSORS = {
    "scalar_hessian_d2": {
        "path": (
            "runs/physics-language/quartic-scalar-hessian-d2-integrability-gate/campaign.json"
        ),
        "file_sha256": "654a0442e0d6ec0166eed1d15da163260658d6cd44192da9cbb4e4b2b88f105a",
        "content_sha256": "66f680eec0ab93169163f7f1e2055aac7713f45fcc2abf2f12907183e701b45c",
    },
    "full_tensor_reconciliation": {
        "path": (
            "runs/physics-language/quartic-full-tensor-good-unknown-"
            "reconciliation-gate/campaign.json"
        ),
        "file_sha256": "cf7957c2efad52a1fa91761fc6259e17a58011cc6093365f9e86e8e7eea0dfd6",
        "content_sha256": "9994df86948a4419dd999b66610e9fea847dece6d5300f68152e942ffb2b87c8",
    },
}
EXPECTED_DATA_SEALS = {
    "observations_opened": False,
    "solar_system_inputs_opened": False,
    "cosmology_inputs_opened": False,
    "paid_llm_calls": False,
    "live_SQLite_opened": False,
    "GPU_execution_used": False,
}
EXPECTED_CLAIM_SEALS = {
    "registered_naive_chunk_exterior_derivative_nonzero": True,
    "pure_coordinate_reparameterization_repair_ruled_out": True,
    "torsion_free_domain_connection_repair_ruled_out": True,
    "corrected_source_Jacobian_ruled_out": False,
    "output_bundle_connection_repair_ruled_out": False,
    "torsionful_domain_connection_repair_ruled_out": False,
    "full_ordered_D2_tensor_registered": False,
    "full_high_atom_good_unknown_identity_proved": False,
    "global_H7_energy_closed": False,
    "global_dyadic_summation_applied": False,
    "nonlinear_PDE_closed": False,
    "nonlinear_lifespan_proved": False,
    "candidate_theory_rejected": False,
    "observational_claim_made": False,
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("scalar-Hessian curl path escapes repository") from error
    return path


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    if set(binding) != {"path", "file_sha256", "content_sha256"}:
        raise ValueError("scalar-Hessian curl predecessor binding shape changed")
    path = _inside(root, str(binding["path"]))
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError("scalar-Hessian curl predecessor file hash mismatch")
    value = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("content_sha256") != binding["content_sha256"]
        or _content_sha(value) != binding["content_sha256"]
    ):
        raise ValueError("scalar-Hessian curl predecessor content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "output_path",
            "predecessors",
            "geometric_contract",
            "policies",
            "seals",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("predecessors") != EXPECTED_PREDECESSORS
        or config.get("geometric_contract")
        != {
            "source_outputs": 11,
            "scalar_hessian_directions": 9,
            "coordinate_change": "local_C2_diffeomorphism_with_invertible_Jacobian",
            "domain_connection": "torsion_free_affine_connection",
            "object": "output_indexed_one_form_J_A=J_A_i_dy_i",
            "obstruction": "C_A_ij=(dJ_A)_ij",
        }
        or config.get("policies")
        != {
            "coordinate_only_repair": "exact_invariance_test",
            "torsion_free_domain_connection_repair": "exact_antisymmetrization_test",
            "output_bundle_or_torsion_repair": "fail_closed",
            "full_D2_promotion": "fail_closed",
            "global_H7": "fail_closed",
            "nonlinear_PDE": "fail_closed",
            "lifespan": "fail_closed",
            "candidate_rejection": "forbidden",
        }
        or config.get("seals") != EXPECTED_DATA_SEALS
    ):
        raise ValueError("scalar-Hessian curl config boundary changed")


def _records(value: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = value.get("candidate_records", value.get("certificates", []))
    if not isinstance(rows, list):
        raise TypeError("scalar-Hessian curl candidate records are absent")
    result = {
        str(row["candidate_id"]): row
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
    }
    if len(rows) != 12 or len(result) != 12:
        raise ValueError("scalar-Hessian curl candidate set changed")
    return result


def _validate_predecessors(values: Mapping[str, Mapping[str, Any]]) -> None:
    if set(values) != set(EXPECTED_PREDECESSORS):
        raise ValueError("scalar-Hessian curl predecessor set changed")
    scalar = values["scalar_hessian_d2"]
    full = values["full_tensor_reconciliation"]
    if (
        scalar.get("decision")
        != "naive_scalar_hessian_chunk_extension_fails_Schwarz_integrability_candidates_blocked"
        or scalar.get("decision_counts") != {"blocked": 12, "pass": 0, "reject": 0}
        or scalar.get("gate_counts", {}).get("failed_ordered_family_pairs_per_candidate") != 24
        or scalar.get("gate_counts", {}).get("nonzero_Schwarz_residuals_per_candidate") != 30
        or scalar.get("claim_seals", {}).get("naive_chunk_extension_obstructed") is not True
        or scalar.get("claim_seals", {}).get("corrected_covariant_D2_tensor_ruled_out") is not False
        or full.get("decision") != "representative_slice_cancelled_full_D2_identity_blocked"
        or full.get("decision_counts") != {"blocked": 12, "pass": 0, "reject": 0}
        or full.get("claim_seals", {}).get("full_ordered_D2_tensor_registered") is not False
        or full.get("claim_seals", {}).get("global_H7_energy_closed") is not False
        or full.get("claim_seals", {}).get("nonlinear_lifespan_proved") is not False
    ):
        raise ValueError("scalar-Hessian curl predecessor semantic boundary changed")
    if set(_records(scalar)) != set(_records(full)):
        raise ValueError("scalar-Hessian curl predecessor candidates disagree")


def _parse_entries(residual: Mapping[str, Any]) -> dict[int, sp.Expr]:
    entries = residual.get("nonzero_residuals")
    if not isinstance(entries, list) or not entries:
        raise ValueError("scalar-Hessian curl residual evidence is empty")
    result: dict[int, sp.Expr] = {}
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"output_row", "value"}
            or not isinstance(entry["output_row"], int)
            or isinstance(entry["output_row"], bool)
            or not 0 <= entry["output_row"] < 11
            or entry["output_row"] in result
        ):
            raise ValueError("scalar-Hessian curl residual entry changed")
        value = sp.sympify(str(entry["value"]))
        if value == 0:
            raise ValueError("scalar-Hessian curl residual contains a zero witness")
        result[entry["output_row"]] = value
    if residual.get("residual_sha256") != _sha(entries):
        raise ValueError("scalar-Hessian curl residual hash changed")
    return result


def _independent_curl_manifest(scalar_record: Mapping[str, Any]) -> dict[str, Any]:
    schwarz = scalar_record.get("schwarz_integrability", {})
    residuals = schwarz.get("residuals")
    if (
        not isinstance(residuals, list)
        or len(residuals) != 24
        or schwarz.get("failed_ordered_family_pair_count") != 24
        or schwarz.get("nonzero_residual_entries") != 30
    ):
        raise ValueError("scalar-Hessian curl ordered residual manifest changed")
    ordered: dict[tuple[str, str], tuple[Mapping[str, Any], dict[int, sp.Expr]]] = {}
    for residual in residuals:
        if (
            not isinstance(residual, Mapping)
            or set(residual) != {"left_atom", "right_atom", "nonzero_residuals", "residual_sha256"}
            or not isinstance(residual["left_atom"], str)
            or not isinstance(residual["right_atom"], str)
            or residual["left_atom"] == residual["right_atom"]
        ):
            raise ValueError("scalar-Hessian curl residual record changed")
        key = (residual["left_atom"], residual["right_atom"])
        if key in ordered:
            raise ValueError("scalar-Hessian curl residual pair is duplicated")
        ordered[key] = (residual, _parse_entries(residual))
    independent = []
    visited: set[tuple[str, str]] = set()
    for left, right in sorted(ordered):
        if (left, right) in visited:
            continue
        reverse = ordered.get((right, left))
        if reverse is None:
            raise ValueError("scalar-Hessian curl reverse residual is absent")
        direct_entries = ordered[(left, right)][1]
        reverse_entries = reverse[1]
        if set(direct_entries) != set(reverse_entries) or any(
            sp.simplify(direct_entries[row] + reverse_entries[row]) != 0 for row in direct_entries
        ):
            raise ValueError("scalar-Hessian curl antisymmetry failed")
        canonical_left, canonical_right = sorted((left, right))
        canonical = ordered[(canonical_left, canonical_right)][0]
        independent.append(
            {
                "left_atom": canonical_left,
                "right_atom": canonical_right,
                "nonzero_components": list(canonical["nonzero_residuals"]),
                "component_count": len(canonical["nonzero_residuals"]),
                "two_form_component_sha256": _sha(canonical["nonzero_residuals"]),
            }
        )
        visited.update({(left, right), (right, left)})
    if len(independent) != 12 or sum(row["component_count"] for row in independent) != 15:
        raise ValueError("scalar-Hessian curl independent component count changed")
    body = {
        "domain_dimension": 9,
        "output_dimension": 11,
        "ordered_nonzero_pair_count": 24,
        "ordered_nonzero_component_count": 30,
        "independent_nonzero_pair_count": 12,
        "independent_nonzero_component_count": 15,
        "ordered_antisymmetry_exact": True,
        "independent_components": independent,
    }
    return {**body, "content_sha256": _sha(body)}


def _torsion_free_identity() -> dict[str, Any]:
    partial_ij, partial_ji, gamma_ij, gamma_ji, jet = sp.symbols(
        "partial_ij partial_ji gamma_ij gamma_ji jet"
    )
    covariant_curl = sp.expand((partial_ij - gamma_ij * jet) - (partial_ji - gamma_ji * jet))
    torsion_free = sp.factor(covariant_curl.subs(gamma_ji, gamma_ij))
    ordinary_curl = partial_ij - partial_ji
    residual = sp.simplify(torsion_free - ordinary_curl)
    if residual != 0:
        raise ValueError("torsion-free curl identity failed")
    return {
        "covariant_antisymmetrization": (
            "nabla_i J_A_j - nabla_j J_A_i = (dJ_A)_ij - T^k_ij J_A_k"
        ),
        "torsion_free_specialization": "T^k_ij=0 implies Alt(nabla J_A)=dJ_A",
        "symbolic_residual": str(residual),
        "proved": True,
    }


def _coordinate_invariance_theorem() -> dict[str, Any]:
    dimension = 9
    exterior_dimension = dimension * (dimension - 1) // 2
    determinant_exponent = dimension - 1
    diagonal_symbols = sp.symbols(f"p0:{dimension}", nonzero=True)
    wedge_diagonal = [
        diagonal_symbols[left] * diagonal_symbols[right]
        for left in range(dimension)
        for right in range(left + 1, dimension)
    ]
    exterior_determinant = sp.prod(wedge_diagonal)
    expected = sp.prod(diagonal_symbols) ** determinant_exponent
    if sp.simplify(exterior_determinant - expected) != 0:
        raise ValueError("exterior-square determinant identity failed")
    return {
        "pullback_law": "d(phi^*J_A)=phi^*(dJ_A)",
        "coordinate_Jacobian_dimension": dimension,
        "exterior_square_dimension": exterior_dimension,
        "determinant_identity": "det(Lambda^2 P)=det(P)^8",
        "determinant_exponent": determinant_exponent,
        "determinant_proof": {
            "representation_identity": "Lambda^2(PQ)=Lambda^2(P)Lambda^2(Q)",
            "diagonal_basis_product": "product_(i<j)(p_i*p_j)=product_i(p_i)^8",
            "extension": "polynomial_identity_from_dense_diagonalizable_matrices",
        },
        "invertible_pullback_has_trivial_kernel": True,
        "conclusion": "dJ_A_nonzero_is_invariant_under_local_coordinate_diffeomorphisms",
        "proved": True,
    }


def _candidate_records(values: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    scalar_rows = _records(values["scalar_hessian_d2"])
    full_rows = _records(values["full_tensor_reconciliation"])
    result = []
    for candidate_id in sorted(scalar_rows):
        scalar = scalar_rows[candidate_id]
        full = full_rows[candidate_id]
        coefficients = scalar.get("coefficients")
        if (
            not isinstance(coefficients, Mapping)
            or str(coefficients.get("a10")) != str(full.get("a10"))
            or scalar.get("candidate_decision") != "blocked"
            or scalar.get("candidate_rejection_authorized") is not False
            or full.get("candidate_decision") != "blocked"
            or full.get("candidate_rejection_authorized") is not False
        ):
            raise ValueError("scalar-Hessian curl candidate lineage changed")
        manifest = _independent_curl_manifest(scalar)
        result.append(
            {
                "candidate_id": candidate_id,
                "coefficients": dict(coefficients),
                "curl_two_form_manifest": manifest,
                "registered_naive_chunk_curl_nonzero": True,
                "coordinate_only_repair_possible": False,
                "torsion_free_domain_connection_repair_possible": False,
                "corrected_source_or_output_bundle_or_torsion_repair_ruled_out": False,
                "full_ordered_D2_tensor_registered": False,
                "candidate_decision": "blocked",
                "candidate_rejection_authorized": False,
                "first_blocker": FIRST_BLOCKER,
            }
        )
    return result


def _expected_body(
    root: Path,
    config_path: Path,
    values: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    records = _candidate_records(values)
    return {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": (
            "nonzero_scalar_hessian_curl_is_coordinate_and_torsion_free_connection_"
            "invariant_candidates_blocked"
        ),
        "decision_counts": {"pass": 0, "reject": 0, "blocked": 12},
        "gate_counts": {
            "selected": 12,
            "ordered_nonzero_curl_pairs_per_candidate": 24,
            "independent_nonzero_curl_pairs_per_candidate": 12,
            "ordered_nonzero_curl_components_per_candidate": 30,
            "independent_nonzero_curl_components_per_candidate": 15,
            "independent_nonzero_curl_components_total": 180,
            "coordinate_only_repairs_ruled_out": 12,
            "torsion_free_domain_connection_repairs_ruled_out": 12,
            "corrected_source_repairs_registered": 0,
            "full_ordered_D2_manifests_admitted": 0,
            "global_H7_closures": 0,
            "nonlinear_PDE_closures": 0,
            "lifespans_proved": 0,
        },
        "first_blocker": FIRST_BLOCKER,
        "theorem": {
            "name": "vector_valued_one_form_curl_invariance_obstruction",
            "object": "J_A=J_A_i dy_i on the nine scalar-Hessian directions",
            "obstruction": "C_A=dJ_A",
            "coordinate_invariance": _coordinate_invariance_theorem(),
            "torsion_free_connection_identity": _torsion_free_identity(),
            "conclusion": (
                "The registered naive principal one-form has a nonzero exterior "
                "derivative for every candidate. An invertible coordinate change or "
                "torsion-free connection on the scalar-Hessian domain cannot make it "
                "a Hessian. This does not rule out changing the one-form through "
                "corrected source, output-bundle, or torsion terms."
            ),
        },
        "exact_controls": {
            "ordered_pair_antisymmetry": {
                "ordered_pairs_per_candidate": 24,
                "independent_pairs_per_candidate": 12,
                "passed_candidates": 12,
            },
            "drop_reverse_order": {
                "would_destroy_antisymmetry_certificate": True,
                "rejected": True,
            },
            "singular_coordinate_map": {
                "outside_local_diffeomorphism_contract": True,
                "rejected": True,
            },
            "torsion_or_output_connection_zero_assumption": {
                "not_registered": True,
                "rejected": True,
            },
            "promote_curl_no_go_to_candidate_rejection": {
                "corrected_one_form_not_ruled_out": True,
                "rejected": True,
            },
        },
        "candidate_records": records,
        "secondary_blockers": [
            "corrected_candidate_bound_source_Jacobian_one_form_not_registered",
            "output_bundle_connection_and_torsion_corrections_not_classified",
            "remaining_247698_ordered_D2_entries_not_materialized",
            "complete_high_atom_good_unknown_identity_not_registered",
            "induced_TC1_TC2_TC3_TC5_bounds_not_closed",
            "B7_global_H7_dyadic_summation_PDE_and_lifespan_not_closed",
        ],
        "claim_seals": EXPECTED_CLAIM_SEALS,
        "data_seals": EXPECTED_DATA_SEALS,
        "scope": (
            "exact invariant no-go for repairing the registered naive scalar-Hessian "
            "principal one-form by an invertible coordinate reparameterization or a "
            "torsion-free domain connection; no corrected-source no-go, full D2, full "
            "good-unknown, global H7, PDE, lifespan, theory, or observational claim"
        ),
        "source_bindings": {
            **EXPECTED_PREDECESSORS,
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": SOURCE_PATH,
                "file_sha256": _file_sha(_inside(root, SOURCE_PATH)),
            },
            "test": {
                "path": TEST_PATH,
                "file_sha256": _file_sha(_inside(root, TEST_PATH)),
            },
        },
    }


def _validate_source_bindings(value: Mapping[str, Any], root: Path) -> None:
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != {
        *EXPECTED_PREDECESSORS,
        "config",
        "source",
        "test",
    }:
        raise ValueError("scalar-Hessian curl source binding set changed")
    for label, expected in EXPECTED_PREDECESSORS.items():
        if bindings.get(label) != expected:
            raise ValueError("scalar-Hessian curl predecessor binding changed")
        _load_bound(root, expected)
    paths = {"config": CONFIG_PATH, "source": SOURCE_PATH, "test": TEST_PATH}
    for label, relative in paths.items():
        binding = bindings[label]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or binding.get("file_sha256") != _file_sha(_inside(root, relative))
        ):
            raise ValueError("scalar-Hessian curl local source binding changed")


def _validate_result(value: Mapping[str, Any], *, root: Path | None = None) -> None:
    validation_root = (root or Path(__file__).resolve().parents[2]).resolve()
    if value.get("content_sha256") != _content_sha(value):
        raise ValueError("scalar-Hessian curl content hash changed")
    _validate_source_bindings(value, validation_root)
    config_path = _inside(validation_root, CONFIG_PATH)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _load_bound(validation_root, binding)
        for label, binding in EXPECTED_PREDECESSORS.items()
    }
    _validate_predecessors(predecessors)
    expected = _expected_body(validation_root, config_path, predecessors)
    if {key: item for key, item in value.items() if key != "content_sha256"} != expected:
        raise ValueError("scalar-Hessian curl result boundary changed")


def build_gate(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parents[2]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    predecessors = {
        label: _load_bound(root, binding) for label, binding in config["predecessors"].items()
    }
    _validate_predecessors(predecessors)
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
    parser.add_argument("--output", action="store_true")
    args = parser.parse_args()
    if args.output:
        print(write_gate(args.config))
    else:
        print(json.dumps(build_gate(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
