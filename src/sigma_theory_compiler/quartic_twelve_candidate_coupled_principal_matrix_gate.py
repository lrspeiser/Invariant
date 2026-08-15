from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


class QuarticCoupledPrincipalMatrixGateError(RuntimeError):
    """Raised when the bounded principal-matrix census cannot be replayed."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise QuarticCoupledPrincipalMatrixGateError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise QuarticCoupledPrincipalMatrixGateError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QuarticCoupledPrincipalMatrixGateError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise QuarticCoupledPrincipalMatrixGateError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise QuarticCoupledPrincipalMatrixGateError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise QuarticCoupledPrincipalMatrixGateError(f"bound content hash mismatch: {path}")
    return path, value


def _sector(receipt: dict[str, Any], sector_id: str) -> dict[str, Any]:
    result = next(
        (item for item in receipt.get("sector_results", []) if item.get("sector_id") == sector_id),
        None,
    )
    if not isinstance(result, dict):
        raise QuarticCoupledPrincipalMatrixGateError(f"missing matter sector: {sector_id}")
    return result


def _gate(sector: dict[str, Any], gate_id: str) -> dict[str, Any]:
    result = next(
        (item for item in sector.get("gates", []) if item.get("gate_id") == gate_id),
        None,
    )
    if not isinstance(result, dict):
        raise QuarticCoupledPrincipalMatrixGateError(f"missing matter gate: {gate_id}")
    return result


def _matrix_census() -> dict[str, Any]:
    sizes = {"gravity": 11, "scalar": 1, "maxwell": 4, "fluid": 1}
    total_dimension = sum(sizes.values())
    diagonal_entries = sum(value * value for value in sizes.values())
    gravity_to_matter_zero = sizes["gravity"] * (
        sizes["scalar"] + sizes["maxwell"] + sizes["fluid"]
    )
    scalar_fluid_to_gravity_zero = (sizes["scalar"] + sizes["fluid"]) * sizes["gravity"]
    maxwell_to_gravity_scalar_zero = sizes["maxwell"]
    matter_sector_cross_zero = 18
    known_off_diagonal_zero = (
        gravity_to_matter_zero
        + scalar_fluid_to_gravity_zero
        + maxwell_to_gravity_scalar_zero
        + matter_sector_cross_zero
    )
    unresolved_maxwell_metric = sizes["maxwell"] * 10
    determined_entries = diagonal_entries + known_off_diagonal_zero
    if (
        total_dimension != 17
        or diagonal_entries != 139
        or known_off_diagonal_zero != 110
        or unresolved_maxwell_metric != 40
        or determined_entries + unresolved_maxwell_metric != total_dimension**2
    ):
        raise QuarticCoupledPrincipalMatrixGateError("matrix block census replay failed")
    return {
        "second_order_field_basis": [
            "quartic metric and gravitational scalar (11)",
            "chi_m (1)",
            "B_mu (4)",
            "tau (1)",
        ],
        "block_sizes": sizes,
        "second_order_dimension": total_dimension,
        "target_first_order_state_dimension": 85,
        "diagonal_entries_determined": diagonal_entries,
        "off_diagonal_zero_entries_determined": known_off_diagonal_zero,
        "entries_determined": determined_entries,
        "entries_unresolved": unresolved_maxwell_metric,
        "unresolved_block": {
            "equation_rows": "Lorenz-reduced Maxwell potential B_mu (4)",
            "second_derivative_columns": "physical metric g_rho_sigma (10)",
            "shape": [4, 10],
            "entry_count": unresolved_maxwell_metric,
        },
        "certified_zero_blocks": [
            {
                "rows": "quartic gravity+gravity-scalar (11)",
                "columns": "all matter fields (6)",
                "entry_count": gravity_to_matter_zero,
                "basis": "first-derivative minimally coupled matter stress has no second matter derivatives",
            },
            {
                "rows": "canonical scalar and irrotational fluid (2)",
                "columns": "quartic gravity+gravity-scalar (11)",
                "entry_count": scalar_fluid_to_gravity_zero,
                "basis": "first-derivative scalar Euler operators contain at most first metric derivatives and no phi_g dependency",
            },
            {
                "rows": "Maxwell potential (4)",
                "columns": "gravity scalar phi_g (1)",
                "entry_count": maxwell_to_gravity_scalar_zero,
                "basis": "the Maxwell action has no phi_g dependency",
            },
            {
                "rows": "distinct matter sectors",
                "columns": "distinct matter sectors",
                "entry_count": matter_sector_cross_zero,
                "basis": "registered additive matter action and matter direct-sum principal certificate",
            },
        ],
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if config.get("schema_version") != "invariant-quartic-coupled-principal-matrix-gate-config-1.0":
        raise QuarticCoupledPrincipalMatrixGateError("unsupported config schema")
    expected_policy = {
        "exact_partial_matrix_skeleton": True,
        "full_coupled_principal_matrix": False,
        "maxwell_metric_mixed_block_zero": False,
        "full_85_state_first_order_reduction": False,
        "full_coupled_symmetrizer": False,
        "sourced_gravity_constraints": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise QuarticCoupledPrincipalMatrixGateError("claims policy is absent or broadened")
    if config.get("field_block_sizes") != {
        "quartic_gravity_and_gravity_scalar": 11,
        "canonical_matter_scalar": 1,
        "lorenz_maxwell_potential": 4,
        "irrotational_fluid_potential": 1,
    }:
        raise QuarticCoupledPrincipalMatrixGateError("field block manifest changed")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {
        "sourced_metric_euler",
        "vacuum_first_order",
        "combined_matter",
        "universal_matter",
        "total_action",
    }:
        raise QuarticCoupledPrincipalMatrixGateError("closed binding manifest changed")
    sourced = bound["sourced_metric_euler"][1]
    vacuum = bound["vacuum_first_order"][1]
    combined = bound["combined_matter"][1]
    universal = bound["universal_matter"][1]
    total_action = bound["total_action"][1]
    if sourced.get("decision") != "PASS_SOURCED_METRIC_EULER_BINDING_ALL_TWELVE_ONLY":
        raise QuarticCoupledPrincipalMatrixGateError("sourced Euler predecessor changed")
    if total_action.get("decision") != "PASS_TOTAL_ACTION_HASH_BINDING_ALL_TWELVE_ONLY":
        raise QuarticCoupledPrincipalMatrixGateError("total action predecessor changed")
    if combined.get("decision") != "BOUNDED_PASS_MATTER_INTERFACE_WITH_TYPED_GRAVITY_BLOCK":
        raise QuarticCoupledPrincipalMatrixGateError("combined matter predecessor changed")
    matter_principal = combined.get("combined_matter_certificate", {}).get(
        "combined_matter_principal_compatibility", {}
    )
    if (
        matter_principal.get("second_order_components") != 6
        or matter_principal.get("second_derivative_cross_sector_blocks") != 0
        or matter_principal.get("principal_block_coefficients") != [[-1, 1]] * 5 + [[-3, 1]]
    ):
        raise QuarticCoupledPrincipalMatrixGateError("matter direct-sum principal changed")
    scalar = _sector(universal, "minimally_coupled_scalar")
    maxwell = _sector(universal, "maxwell_lorenz_gauge")
    for sector in (scalar, maxwell):
        action = _gate(sector, "action_level_universal_metric_coupling")
        if (
            action.get("outcome") != "PASS"
            or action.get("evidence", {}).get(
                "metric_matter_cross_second_derivative_principal_terms"
            )
            != 0
        ):
            raise QuarticCoupledPrincipalMatrixGateError(
                "matter action cross-block evidence changed"
            )
    maxwell_principal = _gate(maxwell, "principal_symbol_hyperbolicity")
    if maxwell_principal.get(
        "outcome"
    ) != "PASS" or "frozen local Minkowski frame" not in maxwell_principal.get("evidence", {}).get(
        "formulation", ""
    ):
        raise QuarticCoupledPrincipalMatrixGateError("Maxwell principal scope changed")
    matter_components = total_action.get("shared_matter_action", {}).get("components", [])
    if len(matter_components) != 3 or any(
        item.get("maximum_derivatives_per_matter_field") != 1 for item in matter_components
    ):
        raise QuarticCoupledPrincipalMatrixGateError("first-derivative matter action changed")

    sourced_records = {
        item.get("candidate_id"): item for item in sourced.get("candidate_results", [])
    }
    vacuum_records = {item.get("candidate_id"): item for item in vacuum.get("certificates", [])}
    expected_count = config.get("expected_candidate_count")
    if (
        expected_count != 12
        or len(sourced_records) != expected_count
        or set(sourced_records) != set(vacuum_records)
        or None in sourced_records
    ):
        raise QuarticCoupledPrincipalMatrixGateError("candidate set mismatch")
    census = _matrix_census()
    matter_principal_sha = _canonical_sha(matter_principal)
    results: list[dict[str, Any]] = []
    for candidate_id in sorted(sourced_records):
        vacuum_record = vacuum_records[candidate_id]
        if (
            vacuum_record.get("status") != "pass_exact_55_variable_principal_first_order_reduction"
            or vacuum_record.get("state_dimensions", {}).get("physical_space_first_order") != 55
        ):
            raise QuarticCoupledPrincipalMatrixGateError(
                f"vacuum first-order block failed: {candidate_id}"
            )
        skeleton = {
            "schema_version": "invariant-partial-coupled-principal-matrix-skeleton-1.0",
            "candidate_id": candidate_id,
            "sourced_metric_euler_sha256": sourced_records[candidate_id][
                "sourced_metric_euler_sha256"
            ],
            "vacuum_55_state_spatial_block_sha256": vacuum_record["source_spatial_block_sha256"],
            "combined_matter_principal_sha256": matter_principal_sha,
            "matrix_census": census,
            "full_matrix_status": "BLOCK",
        }
        results.append(
            {
                "candidate_id": candidate_id,
                "partial_matrix_skeleton_sha256": _canonical_sha(skeleton),
                "partial_matrix_skeleton": skeleton,
                "outcome": "BLOCK",
                "reason_code": "missing_nonlinear_lorenz_maxwell_metric_mixed_principal_block",
            }
        )

    minimal_contract = {
        "missing_registration": (
            "candidate-compatible nonlinear arbitrary-background Lorenz-Maxwell mixed principal derivative"
        ),
        "required_derivative": (
            "d E_Maxwell^mu / d(partial_alpha partial_beta g_rho_sigma) after the exact "
            "Lorenz gauge reduction used to obtain the four wave blocks"
        ),
        "required_shape": [4, 10],
        "required_outcome": (
            "materialize all 40 entries, proving zero or retaining their exact values; then "
            "embed the resulting 17-field symbol into an exact 85-state physical-space "
            "first-order reduction"
        ),
        "why_current_evidence_is_insufficient": (
            "the registered four-wave Maxwell block is scoped to a frozen local Minkowski "
            "frame and does not evaluate second metric derivatives introduced or cancelled "
            "by the nonlinear Lorenz reduction"
        ),
    }
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_quartic_twelve_candidate_coupled_principal_matrix_gate.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-coupled-principal-matrix-gate-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "TYPED_BLOCK_MISSING_MAXWELL_METRIC_MIXED_PRINCIPAL_BLOCK",
        "matrix_census": census,
        "candidate_results": results,
        "minimal_registration_contract": minimal_contract,
        "counts": {
            "candidates": 12,
            "second_order_dimension": 17,
            "target_first_order_dimension": 85,
            "determined_entries_per_candidate": 249,
            "unresolved_entries_per_candidate": 40,
            "determined_entries_total": 2988,
            "unresolved_entries_total": 480,
            "full_matrices_passed": 0,
            "typed_blocks": 12,
            "rejects": 0,
        },
        "claims": {
            "exact_partial_matrix_skeleton_all_twelve": True,
            "full_coupled_principal_matrix_any_candidate": False,
            "maxwell_metric_mixed_block_proved_zero": False,
            "full_85_state_first_order_reduction_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "sourced_gravity_constraints_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact 17-field coupled principal-matrix skeleton for all twelve quartic "
            "candidates, preserving every registered diagonal block and supported zero "
            "cross block; the nonlinear Lorenz-Maxwell 4x10 metric mixed block and hence "
            "the full 85-state reduction remain typed BLOCK. No symmetrizer, constraint, "
            "H7, universal-matter, or promotion conclusion is drawn"
        ),
        "source_bindings": {
            "config": {
                "path": config_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            **{
                name: {
                    "path": path.relative_to(repository).as_posix(),
                    "file_sha256": _file_sha(path),
                    "content_sha256": value["content_sha256"],
                }
                for name, (path, value) in bound.items()
            },
            "source": {
                "path": source_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(repository).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
    }
    return {**body, "content_sha256": _canonical_sha(body)}


def write_receipt(
    config_path: Path, output_path: Path, *, root: Path | None = None
) -> dict[str, Any]:
    receipt = build_receipt(config_path, root=root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    write_receipt(args.config.resolve(), args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
