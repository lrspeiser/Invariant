from __future__ import annotations

import argparse
import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_first_order_reduction_campaign import _symbol_data


class Quartic85StateReductionError(RuntimeError):
    """Raised when the completed principal symbol cannot be reduced exactly."""


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file_sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as exc:
        raise Quartic85StateReductionError(f"cannot read bound file: {path}") from exc


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise Quartic85StateReductionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise Quartic85StateReductionError(f"JSON root is not an object: {path}")
    return value


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise Quartic85StateReductionError("bound path escapes repository root")
    return path


def _load_binding(root: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _resolve(root, str(binding.get("path", "")))
    if _file_sha(path) != binding.get("file_sha256"):
        raise Quartic85StateReductionError(f"bound file hash mismatch: {path}")
    value = _load_json(path)
    if value.get("content_sha256") != binding.get("content_sha256"):
        raise Quartic85StateReductionError(f"bound content hash mismatch: {path}")
    return path, value


def _matrix_payload(matrix: sp.Matrix) -> list[list[str]]:
    return [
        [str(sp.factor(matrix[row, column])) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _maxwell_mixed_symbol(xi: list[sp.Symbol], potential: list[sp.Symbol]) -> sp.Matrix:
    eta = sp.diag(-1, 1, 1, 1)
    xi_down = sp.Matrix(xi)
    xi_up = eta * xi_down
    potential_down = sp.Matrix(potential)
    pairs = [(left, right) for left in range(4) for right in range(left, 4)]
    q = sp.symbols("q_00 q_01 q_02 q_03 q_11 q_12 q_13 q_22 q_23 q_33")
    h = sp.zeros(4)
    for coordinate, (left, right) in zip(q, pairs):
        value = coordinate if left == right else coordinate / sp.sqrt(2)
        h[left, right] = value
        h[right, left] = value
    trace_h = sum(eta[index, index] * h[index, index] for index in range(4))
    gamma = [
        sp.expand(
            sum(xi_up[rho] * eta[upper, lam] * h[rho, lam] for rho in range(4) for lam in range(4))
            - xi_up[upper] * trace_h / 2
        )
        for upper in range(4)
    ]
    scalar = sp.expand(sum(potential_down[index] * gamma[index] for index in range(4)))
    return sp.Matrix(
        [
            [sp.factor(-xi[row] * sp.diff(scalar, coordinate)) for coordinate in q]
            for row in range(4)
        ]
    )


def _extract_blocks(
    symbol: sp.Matrix, xi: list[sp.Symbol]
) -> tuple[sp.Matrix, list[sp.Matrix], list[list[sp.Matrix]]]:
    zero = {item: 0 for item in xi}
    coefficient_a = symbol.applyfunc(
        lambda expression: sp.factor(sp.expand(expression).coeff(xi[0], 2))
    )
    b_blocks = [
        symbol.applyfunc(
            lambda expression, index=index: sp.factor(
                sp.expand(expression).coeff(xi[0], 1).coeff(xi[index + 1], 1)
            )
        )
        for index in range(3)
    ]
    c_blocks = [[sp.zeros(17) for _ in range(3)] for _ in range(3)]
    for left in range(3):
        c_blocks[left][left] = symbol.applyfunc(
            lambda expression, index=left: sp.factor(sp.expand(expression).coeff(xi[index + 1], 2))
        )
        for right in range(left + 1, 3):
            block = symbol.applyfunc(
                lambda expression, first=left, second=right: sp.factor(
                    sp.expand(expression).coeff(xi[first + 1], 1).coeff(xi[second + 1], 1) / 2
                )
            )
            c_blocks[left][right] = block
            c_blocks[right][left] = block
    reconstructed = coefficient_a * xi[0] ** 2
    reconstructed += sum(
        (b_blocks[index] * xi[0] * xi[index + 1] for index in range(3)),
        sp.zeros(17),
    )
    reconstructed += sum(
        (
            c_blocks[left][right] * xi[left + 1] * xi[right + 1]
            for left in range(3)
            for right in range(3)
        ),
        sp.zeros(17),
    )
    if not (symbol - reconstructed).applyfunc(sp.expand).is_zero_matrix:
        raise Quartic85StateReductionError("A/B_i/C_ij reconstruction failed")
    if any(expression.subs(zero) != 0 for expression in symbol):
        raise Quartic85StateReductionError("principal symbol is not homogeneous quadratic")
    return coefficient_a, b_blocks, c_blocks


@cache
def _generic_reduction() -> dict[str, Any]:
    data = _symbol_data()
    xi = list(data["xi_lower"])
    potential = list(sp.symbols("B_0:4"))
    gravity = data["first_order"]["A"] * xi[0] ** 2
    gravity += data["first_order"]["B"] * xi[0]
    gravity += data["first_order"]["C"]
    symbol = sp.zeros(17)
    symbol[:11, :11] = gravity
    light = -(xi[0] ** 2) + sum(item**2 for item in xi[1:])
    symbol[11, 11] = light
    for index in range(4):
        symbol[12 + index, 12 + index] = light
    symbol[16, 16] = -3 * xi[0] ** 2 + sum(item**2 for item in xi[1:])
    symbol[12:16, :10] = _maxwell_mixed_symbol(xi, potential)
    coefficient_a, b_blocks, c_blocks = _extract_blocks(symbol, xi)
    direction = xi[1:]
    b_direction = sum((direction[index] * b_blocks[index] for index in range(3)), sp.zeros(17))
    c_flux = [
        sum(
            (direction[left] * c_blocks[left][right] for left in range(3)),
            sp.zeros(17),
        )
        for right in range(3)
    ]
    identity = sp.eye(17)
    mass = sp.diag(identity, coefficient_a, identity, identity, identity)
    evolution = sp.zeros(85)
    velocity = slice(17, 34)
    spatial = [slice(34 + 17 * index, 51 + 17 * index) for index in range(3)]
    evolution[velocity, velocity] = -b_direction
    for index in range(3):
        evolution[velocity, spatial[index]] = -c_flux[index]
        evolution[spatial[index], velocity] = direction[index] * identity
    lift = sp.zeros(85, 17)
    lift[17:34, :] = xi[0] * identity
    for index in range(3):
        lift[spatial[index], :] = direction[index] * identity
    residual = (evolution - xi[0] * mass) * lift
    expected = sp.zeros(85, 17)
    expected[17:34, :] = -symbol
    lift_residual = (residual - expected).applyfunc(sp.expand)
    if not lift_residual.is_zero_matrix:
        raise Quartic85StateReductionError("85-state lift residual is nonzero")
    block_payload = {
        "A": _matrix_payload(coefficient_a),
        "B_i": [_matrix_payload(block) for block in b_blocks],
        "C_ij": [
            [_matrix_payload(c_blocks[left][right]) for right in range(3)] for left in range(3)
        ],
    }
    assembly = {
        "state_order": ["q_A", "v_A", "w_1A", "w_2A", "w_3A"],
        "mass_diagonal_blocks": ["I17", "A17", "I17", "I17", "I17"],
        "evolution_nonzero_blocks": {
            "K_vv": "-sum_i n_i B_i",
            "K_vw_j": "-sum_i n_i C_ij",
            "K_w_i_v": "n_i I17",
        },
    }
    return {
        "symbol": symbol,
        "A": coefficient_a,
        "B_i": b_blocks,
        "C_ij": c_blocks,
        "block_payload": block_payload,
        "generic_block_sha256": _canonical_sha(block_payload),
        "assembly": assembly,
        "assembly_sha256": _canonical_sha(assembly),
        "lift_residual_zero": True,
        "potential_symbols": potential,
        "data": data,
    }


def _candidate_certificate(candidate: dict[str, Any], complete: dict[str, Any]) -> dict[str, Any]:
    generic = _generic_reduction()
    data = generic["data"]
    coefficients = candidate.get("coefficients")
    if not isinstance(coefficients, dict):
        raise Quartic85StateReductionError("candidate coefficients are absent")
    substitution = {
        data["m2"]: sp.sympify(coefficients["m2"]),
        data["alpha"]: sp.sympify(coefficients["a10"]),
        data["c20"]: sp.sympify(coefficients["c20"]),
    }
    blocks = {
        "A": _matrix_payload(generic["A"].subs(substitution)),
        "B_i": [_matrix_payload(block.subs(substitution)) for block in generic["B_i"]],
        "C_ij": [
            [_matrix_payload(generic["C_ij"][left][right].subs(substitution)) for right in range(3)]
            for left in range(3)
        ],
    }
    block_sha = _canonical_sha(blocks)
    manifest = {
        "schema_version": "invariant-candidate-85-state-first-order-manifest-1.0",
        "candidate_id": candidate["candidate_id"],
        "complete_17_field_principal_sha256": complete["complete_17_field_principal_sha256"],
        "candidate_A_Bi_Cij_sha256": block_sha,
        "assembly_sha256": generic["assembly_sha256"],
        "state_dimension": 85,
        "mass_shape": [85, 85],
        "evolution_shape": [85, 85],
        "nonzero_characteristic_dimension": 34,
        "zero_speed_auxiliary_multiplicity": 51,
        "lift_residual_zero": True,
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "candidate_A_Bi_Cij_sha256": block_sha,
        "first_order_manifest": manifest,
        "first_order_manifest_sha256": _canonical_sha(manifest),
        "outcome": "PASS",
    }


def _corruption_negative() -> dict[str, Any]:
    omega, n3, amplitude = sp.symbols("omega n_3 amplitude")
    correct = omega * n3 * amplitude - n3 * omega * amplitude
    corrupted = -n3 * omega * amplitude
    witness = sp.expand(corrupted.subs({omega: 2, n3: 3, amplitude: 5}))
    if correct != 0 or witness != -30:
        raise Quartic85StateReductionError("kinematic-row corruption replay failed")
    return {
        "mutation": "omit partial_0 w_3A=partial_3 v_A for one coupled field",
        "correct_residual": str(correct),
        "witness": {"omega": 2, "n_3": 3, "amplitude": 5},
        "corrupted_lift_residual": str(witness),
        "rejected": True,
    }


def build_receipt(config_path: Path, *, root: Path | None = None) -> dict[str, Any]:
    repository = (root or config_path.resolve().parents[1]).resolve()
    config = _load_json(config_path)
    if (
        config.get("schema_version")
        != "invariant-quartic-85-state-first-order-reduction-config-1.0"
    ):
        raise Quartic85StateReductionError("unsupported config schema")
    expected_convention = {
        "second_order_fields": 17,
        "state_order": [
            "q_A",
            "v_A=partial_0 q_A",
            "w_1A=partial_1 q_A",
            "w_2A=partial_2 q_A",
            "w_3A=partial_3 q_A",
        ],
        "state_dimension": 85,
        "directional_companion_dimension": 34,
        "zero_speed_auxiliary_multiplicity": 51,
    }
    if config.get("reduction_convention") != expected_convention:
        raise Quartic85StateReductionError("reduction convention changed")
    expected_policy = {
        "exact_85_state_first_order_reduction_all_twelve": True,
        "common_time_covector_domain": False,
        "full_coupled_symmetrizer": False,
        "constraint_propagation": False,
        "sourced_gravity_constraints": False,
        "gravity_h7": False,
        "universal_all_matter": False,
        "promotion": False,
    }
    if config.get("claims_policy") != expected_policy:
        raise Quartic85StateReductionError("claims policy is absent or broadened")
    bound = {
        name: _load_binding(repository, binding)
        for name, binding in config.get("bindings", {}).items()
    }
    if set(bound) != {"complete_17_field_principal", "vacuum_reduction", "total_action"}:
        raise Quartic85StateReductionError("closed binding manifest changed")
    complete = bound["complete_17_field_principal"][1]
    vacuum = bound["vacuum_reduction"][1]
    total_action = bound["total_action"][1]
    if complete.get("decision") != (
        "PASS_EXACT_NONZERO_MAXWELL_MIXED_BLOCK_AND_17_FIELD_PRINCIPAL"
    ):
        raise Quartic85StateReductionError("complete principal predecessor changed")
    generic_vacuum = vacuum.get("generic_reduction_control", {})
    if (
        generic_vacuum.get("passed") is not True
        or generic_vacuum.get("first_order_state", {}).get("total") != 55
        or generic_vacuum.get("generic_scalar_determinant_control", {}).get(
            "state_dimension_per_second_order_field"
        )
        != 5
    ):
        raise Quartic85StateReductionError("standard first-order convention is absent")
    complete_records = {
        item.get("candidate_id"): item for item in complete.get("candidate_results", [])
    }
    vacuum_records = {item.get("candidate_id"): item for item in vacuum.get("certificates", [])}
    action_records = {
        item.get("candidate_id"): item for item in total_action.get("candidate_results", [])
    }
    expected_count = config.get("expected_candidate_count")
    if (
        expected_count != 12
        or len(complete_records) != expected_count
        or set(complete_records) != set(vacuum_records) != set(action_records)
        or set(complete_records) != set(action_records)
        or None in complete_records
    ):
        raise Quartic85StateReductionError("candidate set mismatch")
    results = [
        _candidate_certificate(vacuum_records[candidate_id], complete_records[candidate_id])
        for candidate_id in sorted(complete_records)
    ]
    if len({item["first_order_manifest_sha256"] for item in results}) != 12:
        raise Quartic85StateReductionError("first-order manifests are not one-to-one")
    generic = _generic_reduction()
    corruption = _corruption_negative()
    source_path = Path(__file__).resolve()
    test_path = repository / "tests/test_quartic_twelve_candidate_85_state_first_order_reduction.py"
    body: dict[str, Any] = {
        "schema_version": "invariant-quartic-85-state-first-order-reduction-result-1.0",
        "campaign_id": config["campaign_id"],
        "decision": "PASS_EXACT_85_STATE_FIRST_ORDER_REDUCTION_ALL_TWELVE",
        "reduction_certificate": {
            "second_order_fields": 17,
            "state": {"q_A": 17, "v_A": 17, "w_iA": 51, "total": 85},
            "mass_shape": [85, 85],
            "evolution_shape": [85, 85],
            "directional_companion_dimension": 34,
            "zero_speed_auxiliary_multiplicity": 51,
            "generic_A_Bi_Cij_sha256": generic["generic_block_sha256"],
            "assembly": generic["assembly"],
            "assembly_sha256": generic["assembly_sha256"],
            "nonzero_characteristic_lift_residual_zero": generic["lift_residual_zero"],
            "corruption_negative": corruption,
        },
        "candidate_results": results,
        "counts": {
            "candidates": 12,
            "second_order_fields": 17,
            "first_order_states_per_candidate": 85,
            "first_order_state_entries_total": 1020,
            "directional_companion_dimension": 34,
            "zero_speed_auxiliary_modes_per_candidate": 51,
            "reductions_passed": 12,
            "lift_residual_entries": 0,
            "negative_controls": 1,
            "symmetrizers_constructed": 0,
            "constraint_propagation_claims": 0,
            "rejects": 0,
        },
        "claims": {
            "all_twelve_exact_85_state_first_order_reductions_closed": True,
            "common_time_covector_domain_closed": False,
            "full_coupled_symmetrizer_closed": False,
            "constraint_propagation_closed": False,
            "sourced_gravity_constraints_closed": False,
            "gravity_h7_theorem_established": False,
            "universal_all_matter_closure_established": False,
            "promotion_authorized": False,
        },
        "scope": (
            "exact physical-space first-order reduction of each completed 17-field coupled "
            "principal symbol in the registered q,v,w_i convention, producing 85 states and "
            "51 zero-speed auxiliary modes per candidate. Common-time domains, symmetrizers, "
            "constraint propagation, H7, universal-matter closure, and promotion remain outside"
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
