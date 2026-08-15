from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as k1
from . import quartic_tc2_d4_physical_metric_transport_no_go as no_go

SCHEMA = "sigma-quartic-tc2-d4-alternative-symmetrizer-recurrence-audit-1.0"
CONFIG_SCHEMA = f"{SCHEMA.removesuffix('-1.0')}-config-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_alternative_symmetrizer_recurrence_audit.json"
SOURCE_PATH = "src/sigma_theory_compiler/quartic_tc2_d4_alternative_symmetrizer_recurrence_audit.py"
TEST_PATH = "tests/test_quartic_tc2_d4_alternative_symmetrizer_recurrence_audit.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-alternative-symmetrizer-recurrence-audit/campaign.json"
)
EIGENVALUES = (
    Fraction(1),
    Fraction(-1),
    Fraction(1, 2),
    Fraction(-1, 2),
    Fraction(1, 3),
    Fraction(-1, 3),
)


class AlternativeSymmetrizerRecurrenceAuditError(ValueError):
    """Raised when the ordered alternative audit or its seals change."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "ascii"
    )


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _with_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _content_hash(body)}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AlternativeSymmetrizerRecurrenceAuditError(f"expected object: {path}")
    return value


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    path = (root / binding["path"]).resolve()
    if root != path and root not in path.parents:
        raise AlternativeSymmetrizerRecurrenceAuditError("bound path escaped project root")
    value = _load_json(path)
    if (
        _file_sha256(path) != binding["file_sha256"]
        or value.get("content_sha256") != binding["content_sha256"]
        or not _hash_matches(value)
    ):
        raise AlternativeSymmetrizerRecurrenceAuditError(f"upstream mismatch: {binding['path']}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("policy")
        != "audit_cross_cluster_then_nonsymmetric_then_action_change_then_alternative_recurrence"
        or list(config.get("audit_order", []))
        != [
            "cross_cluster_symmetric_transport",
            "nonsymmetric_form",
            "change_P55_or_action_authority",
            "alternative_symmetrizer_recurrence",
        ]
        or set(config.get("upstreams", {}))
        != {
            "physical_no_go",
            "K55_frontier",
            "higher_P55",
            "P55_order_one",
            "flat_P55",
            "flat_action_metric",
            "projector_recipes",
        }
        or config.get("target")
        != {
            "evaluation_id": "subset_2",
            "Taylor_order": 3,
            "witness_direction": ["3/5", "4/5", "0"],
            "manifest_registered_before": 154,
            "manifest_required": 304,
            "required_rows": 117180,
        }
        or config.get("caps")
        != {
            "maximum_matrix_dimension": 22,
            "maximum_symmetric_unknowns_per_order": 253,
            "maximum_Taylor_order": 3,
            "maximum_directions": 1,
            "maximum_evaluations": 1,
            "maximum_output_bytes": 1048576,
            "maximum_registered_packets": 0,
            "maximum_emitted_rows": 0,
        }
        or not _hash_matches(config)
    ):
        raise AlternativeSymmetrizerRecurrenceAuditError("invalid alternative audit config")


def _matrix_packet(name: str, matrix: sp.Matrix) -> dict[str, Any]:
    entries = [
        {"row": row, "column": column, "value": sp.sstr(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]
    return _with_hash(
        {
            "schema_version": "sigma-exact-sparse-Qsqrt2-matrix-1.0",
            "name": name,
            "shape": [matrix.rows, matrix.cols],
            "entries": entries,
            "nonzero_entries": len(entries),
        }
    )


def _spectral_basis(
    base: dict[str, Any], direction: tuple[Fraction, Fraction, Fraction]
) -> tuple[sp.Matrix, list[sp.Rational], list[list[int]], list[dict[str, Any]]]:
    columns: list[sp.Matrix] = []
    eigenvalue_by_column: list[sp.Rational] = []
    blocks: list[list[int]] = []
    authority = []
    offset = 0
    for eigenvalue in EIGENVALUES:
        projector = no_go._sympy_matrix(poly._evaluate(base["Pi0"][eigenvalue], direction, 22, 22))
        basis = projector.columnspace()
        pivots = []
        rank = 0
        for column in range(22):
            candidate = projector[:, : column + 1]
            next_rank = candidate.rank()
            if next_rank > rank:
                pivots.append(column)
                rank = next_rank
        if len(basis) != len(pivots):
            raise AlternativeSymmetrizerRecurrenceAuditError("spectral basis authority changed")
        columns.extend(basis)
        block = list(range(offset, offset + len(basis)))
        blocks.append(block)
        offset += len(basis)
        eigenvalue_by_column.extend(
            [sp.Rational(eigenvalue.numerator, eigenvalue.denominator)] * len(basis)
        )
        authority.append(
            {
                "eigenvalue": str(eigenvalue),
                "multiplicity": len(basis),
                "left_to_right_independent_projector_columns": pivots,
            }
        )
    basis_matrix = sp.Matrix.hstack(*columns)
    if basis_matrix.shape != (22, 22) or basis_matrix.det() == 0:
        raise AlternativeSymmetrizerRecurrenceAuditError("incomplete spectral basis")
    return basis_matrix, eigenvalue_by_column, blocks, authority


def _block_symmetric_basis(blocks: list[list[int]]) -> tuple[list[sp.Matrix], list[dict[str, Any]]]:
    matrices = []
    labels = []
    for eigenvalue, block in zip(EIGENVALUES, blocks, strict=True):
        for position, row in enumerate(block):
            for column in block[position:]:
                matrix = sp.zeros(22)
                matrix[row, column] = 1
                matrix[column, row] = 1
                matrices.append(matrix)
                labels.append(
                    {
                        "eigenvalue": str(eigenvalue),
                        "spectral_row": row,
                        "spectral_column": column,
                    }
                )
    return matrices, labels


def _block_full_basis(blocks: list[list[int]]) -> list[sp.Matrix]:
    matrices = []
    for block in blocks:
        for row in block:
            for column in block:
                matrix = sp.zeros(22)
                matrix[row, column] = 1
                matrices.append(matrix)
    return matrices


def _canonical_next_coefficient(
    forcing: sp.Matrix,
    basis: sp.Matrix,
    inverse_basis: sp.Matrix,
    eigenvalues: list[sp.Rational],
) -> sp.Matrix:
    transformed = (basis.T * forcing * basis).applyfunc(sp.factor)
    coefficient = sp.zeros(22)
    for row in range(22):
        for column in range(row + 1, 22):
            if eigenvalues[row] != eigenvalues[column]:
                value = -transformed[row, column] / (eigenvalues[column] - eigenvalues[row])
                coefficient[row, column] = sp.factor(value)
                coefficient[column, row] = sp.factor(value)
            elif transformed[row, column] != 0:
                raise AlternativeSymmetrizerRecurrenceAuditError(
                    "equal-eigenspace forcing obstructs canonical recurrence"
                )
    return (inverse_basis.T * coefficient * inverse_basis).applyfunc(sp.factor)


def _equal_block_vector(matrix: sp.Matrix, basis: sp.Matrix, blocks: list[list[int]]) -> sp.Matrix:
    transformed = (basis.T * matrix * basis).applyfunc(sp.factor)
    return sp.Matrix(
        [
            transformed[row, column]
            for block in blocks
            for position, row in enumerate(block)
            for column in block[position + 1 :]
        ]
    )


def _data(root: Path, upstreams: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p_axes = [
        k1.exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]
    ]
    h_plus = k1.exact._matrix_from_packet(
        upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"]
    )
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    base = k1._base_data(p_axes, h_plus, recipes)
    p1_record = next(
        packet
        for packet in upstreams["P55_order_one"]["registered_P55_Taylor_order_one_packets"]
        if packet["evaluation_id"] == "subset_2"
    )
    p2_record = next(
        packet
        for packet in upstreams["higher_P55"][
            "registered_P55_Taylor_orders_two_through_four_packets"
        ]
        if packet["evaluation_id"] == "subset_2" and packet["Taylor_order"] == 2
    )
    companion1 = poly._multiply(
        poly._multiply(
            base["JT"], k1._linear_packet(p1_record["P55_Taylor_order_one_matrix"], [55, 55])
        ),
        base["J"],
    )
    companion2 = poly._multiply(
        poly._multiply(base["JT"], k1._linear_packet(p2_record, [55, 55])), base["J"]
    )
    residual55 = k1._sphere_packet(
        upstreams["K55_frontier"]["failure_checkpoint"]["sphere_symmetrizer_residual"],
        [55, 55],
    )
    residual22 = poly._multiply(poly._multiply(base["JT"], residual55), base["J"])
    direction = no_go.WITNESS_DIRECTION
    return {
        "base": base,
        "C0": no_go._sympy_matrix(poly._evaluate(base["C0"], direction, 22, 22)),
        "C1": no_go._sympy_matrix(poly._evaluate(companion1, direction, 22, 22)),
        "C2": no_go._sympy_matrix(poly._evaluate(companion2, direction, 22, 22)),
        "G0": no_go._sympy_matrix(poly._evaluate(base["G0"], direction, 22, 22)),
        "R3": no_go._sympy_matrix(poly._evaluate(residual22, direction, 22, 22)),
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load_json(config_path)
    _validate_config(config)
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    if (
        upstreams["physical_no_go"].get("decision") != "BLOCK_SERIALIZATION"
        or upstreams["K55_frontier"].get("failure_checkpoint", {}).get("evaluation_id")
        != "subset_2"
        or upstreams["K55_frontier"].get("failure_checkpoint", {}).get("Taylor_order") != 3
    ):
        raise AlternativeSymmetrizerRecurrenceAuditError("predecessor frontier changed")
    data = _data(root, upstreams)
    c0, c1, c2, g0, residual = (
        data["C0"],
        data["C1"],
        data["C2"],
        data["G0"],
        data["R3"],
    )
    basis, eigenvalues, blocks, basis_authority = _spectral_basis(
        data["base"], no_go.WITNESS_DIRECTION
    )
    inverse_basis = basis.inv()
    spectral_g1_basis, basis_labels = _block_symmetric_basis(blocks)
    lower_order_symmetric_basis = [
        (inverse_basis.T * item * inverse_basis).applyfunc(sp.factor) for item in spectral_g1_basis
    ]
    lower_order_nonsymmetric_basis = [
        (inverse_basis.T * item * inverse_basis).applyfunc(sp.factor)
        for item in _block_full_basis(blocks)
    ]

    # 1. Cross-eigenvalue transport can hit the target only by breaking the
    # already-sealed lower-order equation. The lower-order-preserving domain is
    # exactly block diagonal in this spectral basis.
    symmetric_basis = []
    for row in range(22):
        for column in range(row, 22):
            item = sp.zeros(22)
            item[row, column] = 1
            item[column, row] = 1
            symmetric_basis.append(item)
    cross_results = []
    for eigenvalue in (Fraction(1), Fraction(-1)):
        projector = no_go._sympy_matrix(
            poly._evaluate(data["base"]["Pi0"][eigenvalue], no_go.WITNESS_DIRECTION, 22, 22)
        )
        columns = [
            sp.Matrix(projector.T * (item * c1 - c1.T * item) * projector).reshape(484, 1)
            for item in symmetric_basis
        ]
        unrestricted_map = sp.Matrix.hstack(*columns)
        target = sp.Matrix(-(projector.T * residual * projector)).reshape(484, 1)
        preserving_map = sp.Matrix.hstack(
            *[
                sp.Matrix(projector.T * (item * c1 - c1.T * item) * projector).reshape(484, 1)
                for item in lower_order_symmetric_basis
            ]
        )
        cross_results.append(
            {
                "eigenvalue": str(eigenvalue),
                "unrestricted_symmetric_map_rank": unrestricted_map.rank(),
                "unrestricted_augmented_rank": unrestricted_map.row_join(target).rank(),
                "lower_order_preserving_map_rank": preserving_map.rank(),
                "lower_order_preserving_augmented_rank": preserving_map.row_join(target).rank(),
                "conclusion": "unrestricted cross-eigenvalue solutions exist but violate DeltaG2*C0-C0^T*DeltaG2=0",
            }
        )

    nonsymmetric_results = []
    for eigenvalue in (Fraction(1), Fraction(-1)):
        projector = no_go._sympy_matrix(
            poly._evaluate(data["base"]["Pi0"][eigenvalue], no_go.WITNESS_DIRECTION, 22, 22)
        )
        target = sp.Matrix(-(projector.T * residual * projector)).reshape(484, 1)
        transport_map = sp.Matrix.hstack(
            *[
                sp.Matrix(projector.T * (item * c1 - c1.T * item) * projector).reshape(484, 1)
                for item in lower_order_nonsymmetric_basis
            ]
        )
        nonsymmetric_results.append(
            {
                "eigenvalue": str(eigenvalue),
                "lower_order_preserving_nonsymmetric_variables": len(
                    lower_order_nonsymmetric_basis
                ),
                "transport_map_rank": transport_map.rank(),
                "augmented_rank": transport_map.row_join(target).rank(),
            }
        )

    # 3. An exact pencil mutation exists algebraically, but is rejected because
    # P55 order three is directly action-derived and hash bound.
    action_mutation = (-sp.Rational(1, 2) * g0.inv() * residual).applyfunc(sp.factor)
    action_mutation_remainder = (
        g0 * action_mutation - action_mutation.T * g0 + residual
    ).applyfunc(sp.factor)
    if not action_mutation_remainder.is_zero_matrix:
        raise AlternativeSymmetrizerRecurrenceAuditError("action mutation witness changed")

    # 4. Permit the alternative symmetrizer to use the order-one freedom that
    # the action-metric ansatz fixed. Solve the exact equal-eigenspace order-three
    # equation, then recover the unequal-gap coefficients canonically.
    candidate_columns = []
    candidate_g2 = []
    for spectral_g1 in spectral_g1_basis:
        delta_g1 = inverse_basis.T * spectral_g1 * inverse_basis
        forcing2 = (delta_g1 * c1 - c1.T * delta_g1).applyfunc(sp.factor)
        delta_g2 = _canonical_next_coefficient(forcing2, basis, inverse_basis, eigenvalues)
        forcing3 = (delta_g2 * c1 - c1.T * delta_g2 + delta_g1 * c2 - c2.T * delta_g1).applyfunc(
            sp.factor
        )
        candidate_columns.append(_equal_block_vector(forcing3, basis, blocks))
        candidate_g2.append(delta_g2)
    recurrence_map = sp.Matrix.hstack(*candidate_columns)
    target = _equal_block_vector(-residual, basis, blocks)
    solution_set = sp.linsolve((recurrence_map, target))
    if solution_set is sp.EmptySet:
        raise AlternativeSymmetrizerRecurrenceAuditError("alternative recurrence lost solution")
    solution = next(iter(solution_set))
    free_symbols = set().union(*(value.free_symbols for value in solution))
    coefficients = [
        sp.factor(value.subs({symbol: 0 for symbol in free_symbols})) for value in solution
    ]
    nonzero_coefficients = [
        {**basis_labels[index], "coefficient": sp.sstr(value)}
        for index, value in enumerate(coefficients)
        if value != 0
    ]
    if nonzero_coefficients != [
        {
            "eigenvalue": "1",
            "spectral_row": 0,
            "spectral_column": 2,
            "coefficient": "64/1875",
        },
        {
            "eigenvalue": "-1",
            "spectral_row": 3,
            "spectral_column": 5,
            "coefficient": "-64/1875",
        },
    ]:
        raise AlternativeSymmetrizerRecurrenceAuditError("canonical solution changed")
    spectral_delta_g1 = sp.zeros(22)
    for coefficient, item in zip(coefficients, spectral_g1_basis, strict=True):
        spectral_delta_g1 += coefficient * item
    delta_g1 = (inverse_basis.T * spectral_delta_g1 * inverse_basis).applyfunc(sp.factor)
    forcing2 = (delta_g1 * c1 - c1.T * delta_g1).applyfunc(sp.factor)
    delta_g2 = _canonical_next_coefficient(forcing2, basis, inverse_basis, eigenvalues)
    residual3_before_g3 = (
        residual + delta_g2 * c1 - c1.T * delta_g2 + delta_g1 * c2 - c2.T * delta_g1
    ).applyfunc(sp.factor)
    if any(_equal_block_vector(residual3_before_g3, basis, blocks)):
        raise AlternativeSymmetrizerRecurrenceAuditError("equal block did not cancel")
    delta_g3 = _canonical_next_coefficient(residual3_before_g3, basis, inverse_basis, eigenvalues)
    residual1 = (delta_g1 * c0 - c0.T * delta_g1).applyfunc(sp.factor)
    residual2 = (delta_g2 * c0 - c0.T * delta_g2 + forcing2).applyfunc(sp.factor)
    residual3 = (delta_g3 * c0 - c0.T * delta_g3 + residual3_before_g3).applyfunc(sp.factor)
    if (
        delta_g1 != delta_g1.T
        or delta_g2 != delta_g2.T
        or delta_g3 != delta_g3.T
        or not residual1.is_zero_matrix
        or not residual2.is_zero_matrix
        or not residual3.is_zero_matrix
    ):
        raise AlternativeSymmetrizerRecurrenceAuditError("alternative recurrence replay failed")

    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_witness_local_companion_alternative_symmetrizer_recurrence_block_55_state_global_registration",
        "decision": "BLOCK_SERIALIZATION",
        "errors": [],
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "audit_order": [
            {
                "alternative": "cross_cluster_symmetric_transport",
                "result": "REJECT_LOWER_ORDER_IDENTITY",
                "exact_sign_results": cross_results,
            },
            {
                "alternative": "nonsymmetric_form",
                "result": "REJECT_NOT_A_SYMMETRIZER",
                "exact_sign_results": nonsymmetric_results,
                "exact_reason": "even the 82-variable lower-order-preserving nonsymmetric domain has zero projected transport rank; dropping G_j=G_j^T also leaves the positive symmetric-energy cone",
            },
            {
                "alternative": "change_P55_or_action_authority",
                "result": "REJECT_CURRENT_SOURCE_AUTHORITY",
                "algebraic_delta_C3": _matrix_packet("delta_C3_action_mutation", action_mutation),
                "post_mutation_companion_residual_entries": 0,
                "exact_reason": "the mutation cancels R3 but changes the sealed action-derived P55 Taylor-order-three packet",
            },
            {
                "alternative": "alternative_symmetrizer_recurrence",
                "result": "PASS_WITNESS_LOCAL_CONSTRUCTION",
                "exact_reason": "use admissible order-one equal-eigenspace metric freedom, its canonical order-two unequal-gap transport, and an order-three unequal-gap completion",
            },
        ],
        "exact_witness_local_recurrence": {
            "evaluation_id": "subset_2",
            "direction": ["3/5", "4/5", "0"],
            "unit_sphere_residual": "(3/5)^2+(4/5)^2-1=0",
            "spectral_basis_authority": basis_authority,
            "symmetric_equal_eigenspace_order_one_variables": len(spectral_g1_basis),
            "order_three_equal_block_equations": recurrence_map.rows,
            "order_three_recurrence_map_rank": recurrence_map.rank(),
            "order_three_augmented_rank": recurrence_map.row_join(target).rank(),
            "canonical_nonzero_spectral_coefficients": nonzero_coefficients,
            "delta_G1": _matrix_packet("alternative_delta_G1", delta_g1),
            "delta_G2": _matrix_packet("alternative_delta_G2", delta_g2),
            "delta_G3": _matrix_packet("alternative_delta_G3", delta_g3),
            "symmetry_remainder_entries": [0, 0, 0],
            "companion_Taylor_identity_remainder_entries": [0, 0, 0],
        },
        "counts": {
            "alternatives_audited_in_declared_order": 4,
            "constructive_witness_local_alternatives": 1,
            "global_coordinate_free_alternatives": 0,
            "manifest_registered_before": 154,
            "manifest_registered_after": 154,
            "remaining_packets": 150,
            "emitted_rows": 0,
            "remaining_rows": 117180,
        },
        "claims": {
            "within_cluster_no_go_bypassed": False,
            "action_authority_changed": False,
            "exact_witness_local_alternative_recurrence_found": True,
            "full_55_state_recurrence_proved": False,
            "positive_symmetrizer_tube_proved": False,
            "global_coordinate_free_recurrence_proved": False,
            "higher_K55_registered": False,
            "higher_TC2_registered": False,
            "lower_Sylvester_registered": False,
            "rows_emitted": False,
        },
        "next_exact_stop_condition": {
            "required": "lift the companion correction through the exact 55-state transverse/cross formula, construct and replay DeltaG1/2/3 as Q(sqrt(2))[n]/(n1^2+n2^2+n3^2-1) packets for all 15 polarization evaluations, prove positivity on the declared tube, and supersede affected order-one K55/TC2 authority atomically",
            "block_if": "any equal-eigenspace recurrence map has augmented rank above coefficient rank, positivity cannot be retained on the declared tube, or any prior manifest family cannot be superseded atomically",
        },
        "negative_controls": {
            "use_unrestricted_cross_cluster_solution_without_order_two_replay": {"rejected": True},
            "label_nonsymmetric_bilinear_form_a_positive_symmetrizer": {"rejected": True},
            "mutate_action_derived_P55_without_new_source_authority": {"rejected": True},
            "register_witness_local_matrix_as_coordinate_free_packet": {"rejected": True},
            "advance_manifest_or_infer_rows": {"rejected": True},
        },
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha256(config_path)},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha256(root / SOURCE_PATH)},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha256(root / TEST_PATH)},
        },
        "scope": "Audits the four explicitly open alternatives in order. It rejects the first three under the sealed lower-order, symmetry, and action-source contracts, then constructs an exact symmetric companion recurrence through Taylor order three at the first rational unit-direction witness. The 55-state transverse/cross lift, positivity tube, and coordinate-free all-direction/all-evaluation family remain unproved. Because the route changes earlier symmetrizer coefficients, it registers zero packets and emits zero rows.",
    }
    document = _with_hash(body)
    if len(json.dumps(document).encode()) > config["caps"]["maximum_output_bytes"]:
        raise AlternativeSymmetrizerRecurrenceAuditError("output exceeded byte cap")
    return document


def validate_campaign(document: dict[str, Any], root: Path) -> None:
    if not _hash_matches(document) or document != build_campaign(root, root / CONFIG_PATH):
        raise AlternativeSymmetrizerRecurrenceAuditError("alternative recurrence replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    document = build_campaign(args.project_root.resolve(), args.config.resolve())
    data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    if args.output.exists():
        if args.output.read_bytes() != data:
            raise AlternativeSymmetrizerRecurrenceAuditError("immutable output conflict")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
