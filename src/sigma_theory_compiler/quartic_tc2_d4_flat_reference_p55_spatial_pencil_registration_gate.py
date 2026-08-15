from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "sigma-quartic-tc2-d4-flat-reference-p55-spatial-pencil-registration-gate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-flat-reference-p55-spatial-pencil-registration-config-1.0"
STATUS_BLOCK = "block_flat_reference_P55_spatial_pencil_live_packet_not_materialized"
STATE_DIMENSION = 55
SPATIAL_AXES = 3
EXPECTED_NONZEROS = 48
PREDECESSOR_SHA256 = "893b5b5daacc749a593d4eddd709e0e61f63f9f7954f46f02c0fd6970f48badb"
FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "complete_coordinate_free_coefficient_map_emitted",
    "complete_coordinate_free_rhs_emitted",
    "full_direction_sphere_D4_compatibility_proved",
    "full_high_atom_identity_proved",
    "full_tube_Sylvester_identity_proved",
    "global_H7_closed",
    "lifespan_proved",
    "nonlinear_PDE_closure_proved",
    "phase_two_exact_solve_admitted",
    "P55_spatial_pencil_registered",
    "P55_minimal_polynomial_certified",
    "theory_candidate_rejected",
}


class FlatReferenceP55RegistrationError(ValueError):
    """Raised when the exact P55 registration record is inconsistent."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FlatReferenceP55RegistrationError(f"expected JSON object: {path}")
    return value


def _resolve_under(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root != path and root not in path.parents:
        raise FlatReferenceP55RegistrationError("bound path escaped project root")
    return path


def _validate_config(config: dict[str, Any]) -> None:
    target = config.get("target", {})
    caps = config.get("resource_caps", {})
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("registration_policy")
        != "only_live_exact_symbol_construction_may_register_P55_entries"
        or not _hash_matches(config)
        or target.get("state_dimension") != STATE_DIMENSION
        or target.get("spatial_axes") != SPATIAL_AXES
        or target.get("required_matrix_packets") != SPATIAL_AXES
        or target.get("expected_nonzero_entries_each_validation_only") != EXPECTED_NONZEROS
        or caps
        != {
            "maximum_matrix_entries": 9075,
            "maximum_sparse_entries": 512,
            "maximum_polynomial_matrix_entries": 3025,
            "maximum_sphere_reductions": 3025,
        }
        or config.get("predecessor", {}).get("content_sha256") != PREDECESSOR_SHA256
    ):
        raise FlatReferenceP55RegistrationError("invalid P55 registration config")


def _validate_live_source(root: Path, live: dict[str, Any]) -> dict[str, Any]:
    source = _resolve_under(root, live["source_path"])
    if _file_sha256(source) != live.get("source_file_sha256"):
        raise FlatReferenceP55RegistrationError("live construction source hash mismatch")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    required = live.get("required_functions")
    if required != ["_symbol_data", "_extract_spatial_blocks", "_full_first_order_pencil"]:
        raise FlatReferenceP55RegistrationError("live function contract mismatch")
    if not set(required).issubset(definitions):
        raise FlatReferenceP55RegistrationError("live construction function missing")
    return {
        "source_path": live["source_path"],
        "source_file_sha256": live["source_file_sha256"],
        "required_functions": required,
        "functions_present": True,
        "flat_substitutions": live["flat_substitutions"],
        "ordered_state": live["ordered_state"],
        "ordered_indices": live["ordered_indices"],
    }


def construct_live_spatial_pencils() -> list[Any]:
    """Run the sole authorized, exact construction of the three flat P55 matrices."""
    import sympy as sp

    from .quartic_first_order_reduction_campaign import (
        _extract_spatial_blocks,
        _full_first_order_pencil,
        _symbol_data,
    )

    data = _symbol_data()
    xi = list(data["xi_lower"])
    substitutions = {data["alpha"]: 0, data["m2"]: 1}
    substitutions.update({symbol: 0 for symbol in data["gradient_lower"]})
    substitutions.update({symbol: 0 for symbol in data["hessian_lower"].free_symbols})
    substitutions.update({symbol: 0 for symbol in data["einstein_upper"].free_symbols})
    coefficient_a = data["first_order"]["A"].subs(substitutions)
    coefficient_b = data["first_order"]["B"]
    coefficient_c = data["first_order"]["C"]
    b_blocks, c_blocks = _extract_spatial_blocks(coefficient_b, coefficient_c, xi[1:])
    b_blocks = [block.subs(substitutions) for block in b_blocks]
    c_blocks = [[block.subs(substitutions) for block in row] for row in c_blocks]
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    matrices = []
    for axis in range(SPATIAL_AXES):
        direction = [sp.Integer(index == axis) for index in range(SPATIAL_AXES)]
        mass, evolution = _full_first_order_pencil(
            coefficient_a,
            b_blocks[axis],
            [c_blocks[axis][right] for right in range(SPATIAL_AXES)],
            direction,
        )
        physical = mass.inv() * evolution
        matrices.append(physical.extract(ordering, ordering).applyfunc(sp.factor))
    return matrices


def certify_live_matrices(matrices: list[Any]) -> dict[str, Any]:
    """Serialize and certify a live-derived P55 triple; never synthesizes entries."""
    import sympy as sp

    if len(matrices) != SPATIAL_AXES or any(matrix.shape != (55, 55) for matrix in matrices):
        raise FlatReferenceP55RegistrationError("live P55 matrix shape mismatch")
    sparse_packets = []
    for axis, matrix in enumerate(matrices, start=1):
        entries = [
            {"row": row, "column": column, "value": sp.sstr(matrix[row, column])}
            for row in range(55)
            for column in range(55)
            if matrix[row, column] != 0
        ]
        if len(entries) != EXPECTED_NONZEROS:
            raise FlatReferenceP55RegistrationError("live P55 nonzero-count mismatch")
        sparse_packets.append(
            {
                "spatial_axis": axis,
                "shape": [55, 55],
                "nonzero_entries": entries,
                "nonzero_count": len(entries),
                "matrix_sha256": hashlib.sha256(_canonical_bytes(entries)).hexdigest(),
            }
        )
    n1, n2, n3 = sp.symbols("n1 n2 n3")
    pencil = n1 * matrices[0] + n2 * matrices[1] + n3 * matrices[2]
    identity = sp.eye(55)
    square = pencil * pencil
    residual = pencil * (square - identity) * (square - identity / 4) * (square - identity / 9)
    groebner = sp.groebner([n1**2 + n2**2 + n3**2 - 1], n1, n2, n3)
    nonzero_raw = 0
    nonzero_remainders = 0
    for entry in residual:
        expanded = sp.expand(entry)
        nonzero_raw += int(expanded != 0)
        remainder = groebner.reduce(expanded)[1]
        nonzero_remainders += int(remainder != 0)
    if nonzero_remainders:
        raise FlatReferenceP55RegistrationError("minimal-polynomial sphere remainder nonzero")
    return {
        "matrix_packets": sparse_packets,
        "linearity": {
            "formula": "P(n)=n1*P1+n2*P2+n3*P3",
            "exact_by_construction": True,
            "coefficient_matrices": 3,
        },
        "minimal_polynomial_certificate": {
            "polynomial": "P*(P^2-I)*(P^2-I/4)*(P^2-I/9)",
            "sphere_relation": "n1^2+n2^2+n3^2-1",
            "entries_reduced": 3025,
            "raw_nonzero_entries": nonzero_raw,
            "nonzero_remainders": 0,
            "all_remainders_zero": True,
        },
    }


def build_campaign(
    project_root: Path, config_path: Path, *, run_live: bool = False
) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    predecessor_path = _resolve_under(root, config["predecessor"]["path"])
    predecessor = _load_json(predecessor_path)
    if (
        not _hash_matches(predecessor)
        or predecessor.get("content_sha256") != PREDECESSOR_SHA256
        or predecessor.get("errors") != []
        or predecessor.get("status")
        != "block_coordinate_free_D4_recurrence_emitter_missing_symbolic_P_and_Taylor_packets"
        or predecessor.get("counts", {}).get("registered_symbolic_input_packets") != 3
    ):
        raise FlatReferenceP55RegistrationError("predecessor seal mismatch")
    source_seal = _validate_live_source(root, config["live_construction"])
    packet = config.get("input_packet")
    if packet is not None:
        raise FlatReferenceP55RegistrationError(
            "config-embedded P55 entries forbidden; registration must use the live constructor"
        )
    certificate = certify_live_matrices(construct_live_spatial_pencils()) if run_live else None
    registered = certificate is not None
    status = (
        "pass_exact_flat_reference_P55_spatial_pencil_registration" if registered else STATUS_BLOCK
    )
    claims = {claim: False for claim in sorted(FALSE_CLAIMS)}
    claims.update(
        {
            "predecessor_content_seal_verified": True,
            "live_constructor_source_and_functions_bound": True,
            "exact_missing_packet_schema_closed": True,
            "P55_spatial_pencil_registered": registered,
            "P55_minimal_polynomial_certified": registered,
        }
    )
    body = {
        "schema_version": SCHEMA,
        "status": status,
        "errors": [],
        "config_sha256": config["content_sha256"],
        "predecessor": {
            "path": config["predecessor"]["path"],
            "content_sha256": PREDECESSOR_SHA256,
            "status": predecessor["status"],
            "verified": True,
        },
        "live_construction": source_seal,
        "required_input_packet_schema": {
            "schema_version": "sigma-exact-sparse-flat-reference-linear-P55-pencil-1.0",
            "required_packets": 3,
            "required_shape_each": [55, 55],
            "required_axes": [1, 2, 3],
            "required_derivation": "construct_live_spatial_pencils",
            "required_fields_each": [
                "spatial_axis",
                "shape",
                "nonzero_entries",
                "nonzero_count",
                "matrix_sha256",
            ],
            "expected_nonzero_entries_each_validation_only": 48,
            "registered_packets": 3 if registered else 0,
            "missing_packets": 0 if registered else 3,
        },
        "registration": certificate,
        "phase_two": {
            "decision": "PASS" if registered else "BLOCK",
            "attempted": registered,
            "blocker": None if registered else "three live-derived sparse P55 packets are absent",
        },
        "counts": {
            "required_matrix_packets": 3,
            "registered_matrix_packets": 3 if registered else 0,
            "missing_matrix_packets": 0 if registered else 3,
            "required_dense_entries": 9075,
            "registered_sparse_entries": 144 if registered else 0,
            "expected_sparse_entries_validation_only": 144,
            "linearity_entries_certified": 3025 if registered else 0,
            "minimal_polynomial_entries_reduced": 3025 if registered else 0,
            "minimal_polynomial_nonzero_remainders": 0 if registered else None,
            "cold_live_construction_attempted": registered,
        },
        "negative_controls": {
            "fabricate_entries_from_expected_counts": {"rejected": True},
            "infer_matrices_from_13_direction_samples": {"rejected": True},
            "treat_missing_entries_as_zero": {"rejected": True},
            "certify_polynomial_before_matrix_registration": {"rejected": True},
            "promote_reference_pencil_to_global_D4_or_H7": {"rejected": True},
        },
        "claims": claims,
        "scope": (
            "Flat-reference coordinate-free P55 registration only. The immutable artifact "
            "blocks because no live-derived sparse packet was materialized; expected counts "
            "are validation constraints, never coefficient data."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def validate_campaign(document: dict[str, Any], project_root: Path) -> None:
    config_path = project_root / (
        "configs/backgrounds/"
        "quartic_tc2_d4_flat_reference_p55_spatial_pencil_registration_gate.json"
    )
    expected = build_campaign(project_root, config_path)
    if document != expected or not _hash_matches(document):
        raise FlatReferenceP55RegistrationError("campaign replay mismatch")


def write_campaign(document: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root, args.config, run_live=args.live)
    path = write_campaign(document, args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
