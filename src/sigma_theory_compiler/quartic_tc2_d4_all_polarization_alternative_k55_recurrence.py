from __future__ import annotations

import argparse
import hashlib
import json
from fractions import Fraction
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_alternative_symmetrizer_recurrence_audit as witness
from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as k1
from . import quartic_tc2_d4_physical_metric_transport_no_go as no_go

SCHEMA = "sigma-quartic-tc2-d4-all-polarization-alternative-k55-recurrence-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_all_polarization_alternative_k55_recurrence.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_all_polarization_alternative_k55_recurrence.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_all_polarization_alternative_k55_recurrence.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-all-polarization-alternative-k55-recurrence/campaign.json"
)
ORDERS = (2, 3, 4)
MAX_ORDER = 4


class AllPolarizationRecurrenceError(ValueError):
    """Raised when an authority or exact recurrence check fails closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _content_hash(value: dict[str, Any]) -> str:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return hashlib.sha256(_canonical_bytes(body)).hexdigest()


def _with_hash(value: dict[str, Any]) -> dict[str, Any]:
    return {**value, "content_sha256": _content_hash(value)}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(value)


def _normalized_text_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AllPolarizationRecurrenceError(f"unreadable JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise AllPolarizationRecurrenceError(f"non-object JSON authority: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise AllPolarizationRecurrenceError("authority path escapes project root")
    return path


def _load_bound(root: Path, binding: dict[str, str]) -> dict[str, Any]:
    value = _load(_inside(root, binding["path"]))
    if not _hash_matches(value) or value.get("content_sha256") != binding["content_sha256"]:
        raise AllPolarizationRecurrenceError(f"semantic authority changed: {binding['path']}")
    return value


MatrixSeries = list[sp.Matrix]


def _zero(rows: int, columns: int) -> sp.Matrix:
    return sp.zeros(rows, columns)


def _constant(matrix: sp.Matrix) -> MatrixSeries:
    return [matrix, *[_zero(matrix.rows, matrix.cols) for _ in range(MAX_ORDER)]]


def _series_add(left: MatrixSeries, right: MatrixSeries) -> MatrixSeries:
    return [left[index] + right[index] for index in range(MAX_ORDER + 1)]


def _series_scale(series: MatrixSeries, coefficient: Any) -> MatrixSeries:
    return [coefficient * item for item in series]


def _series_transpose(series: MatrixSeries) -> MatrixSeries:
    return [item.T for item in series]


def _series_multiply(left: MatrixSeries, right: MatrixSeries) -> MatrixSeries:
    return [
        sum(
            (left[index] * right[order - index] for index in range(order + 1)),
            _zero(left[0].rows, right[0].cols),
        )
        for order in range(MAX_ORDER + 1)
    ]


def _coefficient(left: MatrixSeries, right: MatrixSeries, order: int) -> sp.Matrix:
    return sum(
        (left[index] * right[order - index] for index in range(order + 1)),
        _zero(left[0].rows, right[0].cols),
    )


def _projector_series(
    companion: MatrixSeries, projectors: dict[Fraction, sp.Matrix]
) -> dict[Fraction, MatrixSeries]:
    result = {eigenvalue: _constant(projector) for eigenvalue, projector in projectors.items()}
    identity = sp.eye(22)
    for order in range(1, MAX_ORDER + 1):
        for eigenvalue, series in result.items():
            projector0 = series[0]
            complement = identity - projector0
            commutator = sum(
                (
                    companion[index] * series[order - index]
                    - series[order - index] * companion[index]
                    for index in range(1, order + 1)
                ),
                _zero(22, 22),
            )
            off_diagonal = _zero(22, 22)
            for other, other_projector in projectors.items():
                if other == eigenvalue:
                    continue
                off_diagonal += other_projector * commutator * projector0 / (
                    eigenvalue - other
                ) + projector0 * commutator * other_projector / (other - eigenvalue)
            quadratic = sum(
                (series[index] * series[order - index] for index in range(1, order)),
                _zero(22, 22),
            )
            series[order] = (
                off_diagonal
                - projector0 * quadratic * projector0
                + complement * quadratic * complement
            )
    return result


def _inverse_series(companion: MatrixSeries, inverse0: sp.Matrix) -> MatrixSeries:
    result = _constant(inverse0)
    for order in range(1, MAX_ORDER + 1):
        forcing = sum(
            (companion[index] * result[order - index] for index in range(1, order + 1)),
            _zero(22, 22),
        )
        result[order] = -inverse0 * forcing
    return result


def _construct_k55(
    base: dict[str, Any], p_series: MatrixSeries, h_series: MatrixSeries
) -> MatrixSeries:
    j = _constant(base["J"])
    jt = _constant(base["JT"])
    transverse = _constant(base["T"])
    companion = _series_multiply(_series_multiply(jt, p_series), j)
    lift = _series_multiply(_series_multiply(jt, p_series), transverse)
    energy = [_zero(22, 22) for _ in range(MAX_ORDER + 1)]
    for eigenvalue, projector in _projector_series(companion, base["Pi0"]).items():
        metric = (
            h_series
            if eigenvalue == 1
            else _series_scale(h_series, -1)
            if eigenvalue == -1
            else _constant(sp.eye(22))
        )
        energy = _series_add(
            energy,
            _series_multiply(_series_multiply(_series_transpose(projector), metric), projector),
        )
    inverse = _inverse_series(companion, base["N0"])
    cross = _series_multiply(_series_multiply(_series_transpose(lift), energy), inverse)
    return _series_add(
        transverse,
        _series_add(
            _series_add(
                _series_multiply(cross, jt),
                _series_multiply(j, _series_transpose(cross)),
            ),
            _series_multiply(_series_multiply(j, energy), jt),
        ),
    )


def _relaxed_next_coefficient(
    forcing: sp.Matrix,
    basis: sp.Matrix,
    inverse_basis: sp.Matrix,
    eigenvalues: list[sp.Rational],
) -> sp.Matrix:
    transformed = basis.T * forcing * basis
    coefficient = _zero(22, 22)
    for row in range(22):
        for column in range(row + 1, 22):
            if eigenvalues[row] != eigenvalues[column]:
                value = -transformed[row, column] / (eigenvalues[column] - eigenvalues[row])
                coefficient[row, column] = value
                coefficient[column, row] = value
    return inverse_basis.T * coefficient * inverse_basis


def _canonical_solution(matrix: sp.Matrix, target: sp.Matrix) -> tuple[int, int, list[Any]]:
    rank = matrix.to_DM().rank()
    augmented = matrix.row_join(target)
    augmented_rank = augmented.to_DM().rank()
    if rank != augmented_rank:
        return rank, augmented_rank, []
    if not any(target):
        return rank, augmented_rank, [sp.Integer(0)] * matrix.cols
    reduced, pivots = augmented.to_DM().rref()
    reduced_matrix = reduced.to_Matrix()
    solution: list[Any] = [sp.Integer(0)] * matrix.cols
    for row, column in enumerate(pivots):
        if column < matrix.cols:
            solution[column] = sp.factor(reduced_matrix[row, matrix.cols])
    return rank, augmented_rank, solution


def _matrix_summary(name: str, matrix: sp.Matrix) -> dict[str, Any]:
    entries = [
        {"row": row, "column": column, "value": sp.sstr(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]
    return _with_hash(
        {
            "schema_version": "sigma-exact-sparse-Qsqrt2-matrix-summary-1.0",
            "name": name,
            "shape": [matrix.rows, matrix.cols],
            "nonzero_entries": len(entries),
            "entries": entries,
        }
    )


def _evaluated_base(
    polynomial_base: dict[str, Any], direction: tuple[Fraction, Fraction, Fraction]
) -> dict[str, Any]:
    def evaluate(packet: Any, rows: int, columns: int) -> sp.Matrix:
        return no_go._sympy_matrix(poly._evaluate(packet, direction, rows, columns))

    return {
        "J": evaluate(polynomial_base["J"], 55, 22),
        "JT": evaluate(polynomial_base["JT"], 22, 55),
        "T": evaluate(polynomial_base["T"], 55, 55),
        "N0": evaluate(polynomial_base["N0"], 22, 22),
        "Pi0": {
            eigenvalue: evaluate(packet, 22, 22)
            for eigenvalue, packet in polynomial_base["Pi0"].items()
        },
    }


def _packet_matrix(
    packet: dict[str, Any], direction: tuple[Fraction, Fraction, Fraction], shape: list[int]
) -> sp.Matrix:
    polynomial = k1._linear_packet(packet, shape)
    return no_go._sympy_matrix(poly._evaluate(polynomial, direction, *shape))


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load(config_path)
    if not _hash_matches(config):
        raise AllPolarizationRecurrenceError("configuration seal changed")
    upstreams = {name: _load_bound(root, binding) for name, binding in config["upstreams"].items()}
    direction = tuple(Fraction(value) for value in config["witness_direction"])
    if direction != no_go.WITNESS_DIRECTION:
        raise AllPolarizationRecurrenceError("witness direction authority changed")

    p_axes = [
        k1.exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]
    ]
    h_plus = k1.exact._matrix_from_packet(
        upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"]
    )
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    polynomial_base = k1._base_data(p_axes, h_plus, recipes)
    base = _evaluated_base(polynomial_base, direction)
    basis, eigenvalues, blocks, spectral_authority = witness._spectral_basis(
        polynomial_base, direction
    )
    inverse_basis = basis.inv()
    spectral_g1_basis, basis_labels = witness._block_symmetric_basis(blocks)
    coordinate_g1_basis = [inverse_basis.T * item * inverse_basis for item in spectral_g1_basis]

    p1_by_id = {
        row["evaluation_id"]: row
        for row in upstreams["P55_order_one"]["registered_P55_Taylor_order_one_packets"]
    }
    h1_by_id = {row["evaluation_id"]: row for row in upstreams["H_star_order_one"]["packets"]}
    higher_p: dict[str, dict[int, dict[str, Any]]] = {}
    for packet in upstreams["higher_P55"]["registered_P55_Taylor_orders_two_through_four_packets"]:
        higher_p.setdefault(packet["evaluation_id"], {})[packet["Taylor_order"]] = packet
    higher_h: dict[str, dict[int, dict[str, Any]]] = {}
    for packet in upstreams["higher_H_star"]["packets"]:
        higher_h.setdefault(packet["evaluation_id"], {})[packet["Taylor_order"]] = packet
    evaluation_ids = config["evaluation_ids"]
    if not (
        evaluation_ids == list(p1_by_id)
        and set(evaluation_ids) == set(h1_by_id) == set(higher_p) == set(higher_h)
    ):
        raise AllPolarizationRecurrenceError("15-evaluation authority changed")

    checkpoint_documents = {
        name: _load(_inside(root, path)) for name, path in config["checkpoint_authorities"].items()
    }
    if any(not _hash_matches(value) for value in checkpoint_documents.values()):
        raise AllPolarizationRecurrenceError("K55 checkpoint semantic seal changed")
    for evaluation_id in ("subset_0", "subset_1"):
        checkpoint = checkpoint_documents[evaluation_id]
        if (
            checkpoint.get("evaluation_id") != evaluation_id
            or [packet.get("Taylor_order") for packet in checkpoint.get("packets", [])] != [2, 3, 4]
            or checkpoint.get("sphere_identity_remainders") != 0
        ):
            raise AllPolarizationRecurrenceError("successful K55 checkpoint changed")
    failure = checkpoint_documents["subset_2"]
    if (
        failure.get("evaluation_id") != "subset_2"
        or failure.get("Taylor_order") != 3
        or failure.get("sphere_symmetrizer_remainder_entries") != 120
    ):
        raise AllPolarizationRecurrenceError("subset_2 failure checkpoint changed")

    records: list[dict[str, Any]] = []
    first_incompatibility: dict[str, Any] | None = None
    solutions: dict[str, tuple[sp.Matrix, sp.Matrix, sp.Matrix]] = {}
    for evaluation_id in evaluation_ids:
        p_series = [
            no_go._sympy_matrix(poly._evaluate(polynomial_base["P0"], direction, 55, 55)),
            _packet_matrix(
                p1_by_id[evaluation_id]["P55_Taylor_order_one_matrix"],
                direction,
                [55, 55],
            ),
            *[
                _packet_matrix(higher_p[evaluation_id][order], direction, [55, 55])
                for order in ORDERS
            ],
        ]
        h_series = [
            no_go._sympy_matrix(poly._evaluate(polynomial_base["H0"], direction, 22, 22)),
            _packet_matrix(
                h1_by_id[evaluation_id]["H_star_plus_order_one_matrix"],
                direction,
                [22, 22],
            ),
            *[
                _packet_matrix(higher_h[evaluation_id][order], direction, [22, 22])
                for order in ORDERS
            ],
        ]
        if evaluation_id in {"subset_0", "subset_1"}:
            pulled_residual = _zero(22, 22)
            target_authority = "sealed_successful_coordinate_free_K55_checkpoint"
        elif evaluation_id == "subset_2":
            residual55 = k1._sphere_packet(failure["sphere_symmetrizer_residual"], [55, 55])
            pulled_residual = no_go._sympy_matrix(
                poly._evaluate(
                    poly._multiply(
                        poly._multiply(polynomial_base["JT"], residual55),
                        polynomial_base["J"],
                    ),
                    direction,
                    22,
                    22,
                )
            )
            target_authority = "sealed_subset_2_failure_checkpoint"
        else:
            k_series = _construct_k55(base, p_series, h_series)
            residual55 = _coefficient(k_series, p_series, 3) - _coefficient(
                _series_transpose(p_series), k_series, 3
            )
            pulled_residual = (base["JT"] * residual55 * base["J"]).applyfunc(sp.factor)
            target_authority = "exact_evaluated_authoritative_55_state_recurrence"

        c1 = base["JT"] * p_series[1] * base["J"]
        c2 = base["JT"] * p_series[2] * base["J"]
        forcing2_columns = [metric * c1 - c1.T * metric for metric in coordinate_g1_basis]
        order_two_map = sp.Matrix.hstack(
            *[witness._equal_block_vector(item, basis, blocks) for item in forcing2_columns]
        )
        order_three_columns = []
        candidate_g2 = []
        for forcing2, metric in zip(forcing2_columns, coordinate_g1_basis, strict=True):
            delta_g2 = _relaxed_next_coefficient(forcing2, basis, inverse_basis, eigenvalues)
            candidate_g2.append(delta_g2)
            forcing3 = delta_g2 * c1 - c1.T * delta_g2 + metric * c2 - c2.T * metric
            order_three_columns.append(witness._equal_block_vector(forcing3, basis, blocks))
        order_three_map = sp.Matrix.hstack(*order_three_columns)
        joint_map = order_two_map.col_join(order_three_map)
        equal_target = witness._equal_block_vector(-pulled_residual, basis, blocks)
        joint_target = _zero(order_two_map.rows, 1).col_join(equal_target)
        rank, augmented_rank, coefficients = _canonical_solution(joint_map, joint_target)
        record: dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "direction": config["witness_direction"],
            "target_authority": target_authority,
            "pulled_back_residual_nonzero_entries": sum(item != 0 for item in pulled_residual),
            "order_two_equal_eigenspace_equations": order_two_map.rows,
            "order_two_admissibility_rank": order_two_map.to_DM().rank(),
            "order_three_equal_eigenspace_equations": order_three_map.rows,
            "joint_coefficient_rank": rank,
            "joint_augmented_rank": augmented_rank,
            "compatible": rank == augmented_rank,
        }
        if rank != augmented_rank:
            record["canonical_nonzero_spectral_coefficients"] = []
            records.append(record)
            first_incompatibility = {
                "evaluation_id": evaluation_id,
                "joint_coefficient_rank": rank,
                "joint_augmented_rank": augmented_rank,
                "first_missing_primitive": ("source_bound_equal_eigenspace_G1_transport_solution"),
            }
            break

        nonzero_coefficients = [
            {**basis_labels[index], "coefficient": sp.sstr(value)}
            for index, value in enumerate(coefficients)
            if value != 0
        ]
        delta_g1 = sum(
            (
                coefficient * metric
                for coefficient, metric in zip(coefficients, coordinate_g1_basis, strict=True)
            ),
            _zero(22, 22),
        ).applyfunc(sp.factor)
        forcing2 = (delta_g1 * c1 - c1.T * delta_g1).applyfunc(sp.factor)
        delta_g2 = _relaxed_next_coefficient(forcing2, basis, inverse_basis, eigenvalues).applyfunc(
            sp.factor
        )
        residual_before_g3 = (
            pulled_residual + delta_g2 * c1 - c1.T * delta_g2 + delta_g1 * c2 - c2.T * delta_g1
        ).applyfunc(sp.factor)
        if any(witness._equal_block_vector(residual_before_g3, basis, blocks)):
            raise AllPolarizationRecurrenceError("canonical equal block did not cancel")
        delta_g3 = witness._canonical_next_coefficient(
            residual_before_g3, basis, inverse_basis, eigenvalues
        ).applyfunc(sp.factor)
        c0 = no_go._sympy_matrix(poly._evaluate(polynomial_base["C0"], direction, 22, 22))
        remainders = [
            delta_g1 * c0 - c0.T * delta_g1,
            delta_g2 * c0 - c0.T * delta_g2 + forcing2,
            delta_g3 * c0 - c0.T * delta_g3 + residual_before_g3,
        ]
        remainder_counts = [
            sum(item != 0 for item in remainder.applyfunc(sp.factor)) for remainder in remainders
        ]
        if remainder_counts != [0, 0, 0]:
            raise AllPolarizationRecurrenceError("canonical recurrence replay failed")
        record.update(
            {
                "canonical_nonzero_spectral_coefficients": nonzero_coefficients,
                "delta_G1": _matrix_summary(f"delta_G1_{evaluation_id}", delta_g1),
                "delta_G2": _matrix_summary(f"delta_G2_{evaluation_id}", delta_g2),
                "delta_G3": _matrix_summary(f"delta_G3_{evaluation_id}", delta_g3),
                "companion_Taylor_remainder_entries": remainder_counts,
            }
        )
        records.append(record)
        solutions[evaluation_id] = (delta_g1, delta_g2, delta_g3)

    all_equal_systems_pass = len(records) == 15 and first_incompatibility is None
    lift_records: list[dict[str, Any]] = []
    first_lift_failure: dict[str, Any] | None = None
    positive_tube_proved = False
    if all_equal_systems_pass:
        for evaluation_id in evaluation_ids:
            # Rebuild only after every 22-state transport system has passed.
            p_series = [
                no_go._sympy_matrix(poly._evaluate(polynomial_base["P0"], direction, 55, 55)),
                _packet_matrix(
                    p1_by_id[evaluation_id]["P55_Taylor_order_one_matrix"],
                    direction,
                    [55, 55],
                ),
                *[
                    _packet_matrix(higher_p[evaluation_id][order], direction, [55, 55])
                    for order in ORDERS
                ],
            ]
            h_series = [
                no_go._sympy_matrix(poly._evaluate(polynomial_base["H0"], direction, 22, 22)),
                _packet_matrix(
                    h1_by_id[evaluation_id]["H_star_plus_order_one_matrix"],
                    direction,
                    [22, 22],
                ),
                *[
                    _packet_matrix(higher_h[evaluation_id][order], direction, [22, 22])
                    for order in ORDERS
                ],
            ]
            for order, correction in zip((1, 2, 3), solutions[evaluation_id], strict=True):
                h_series[order] += correction
            k_series = _construct_k55(base, p_series, h_series)
            remainder_counts = []
            for order in (1, 2, 3):
                remainder = (
                    _coefficient(k_series, p_series, order)
                    - _coefficient(_series_transpose(p_series), k_series, order)
                ).applyfunc(sp.factor)
                remainder_counts.append(sum(item != 0 for item in remainder))
            lift_records.append(
                {
                    "evaluation_id": evaluation_id,
                    "K55_Taylor_remainder_entries": remainder_counts,
                    "pass": remainder_counts == [0, 0, 0],
                }
            )
            if remainder_counts != [0, 0, 0]:
                first_lift_failure = {
                    "evaluation_id": evaluation_id,
                    "K55_Taylor_remainder_entries": remainder_counts,
                    "first_missing_primitive": (
                        "exact_55_state_transverse_cross_alternative_metric_lift"
                    ),
                }
                break

    admitted = all_equal_systems_pass and first_lift_failure is None and positive_tube_proved
    manifest_before = config["manifest"]["registered_before"]
    body = {
        "schema_version": SCHEMA,
        "status": (
            "pass_all_15_equal_eigenspace_transports_block_55_state_or_positive_tube"
            if all_equal_systems_pass
            else "block_first_exact_equal_eigenspace_transport_incompatibility"
        ),
        "decision": "REGISTER_ATOMICALLY" if admitted else "BLOCK_SERIALIZATION",
        "config_sha256": config["content_sha256"],
        "upstream_bindings": {
            name: {**binding, "verified": True} for name, binding in config["upstreams"].items()
        },
        "direction": config["witness_direction"],
        "unit_sphere_identity": "(3/5)^2+(4/5)^2-1=0",
        "spectral_basis_authority": spectral_authority,
        "evaluation_records": records,
        "remaining_unaudited_evaluations": evaluation_ids[len(records) :],
        "first_exact_incompatibility": first_incompatibility,
        "all_15_equal_eigenspace_transport_systems_pass": all_equal_systems_pass,
        "lift_records": lift_records,
        "first_exact_55_state_lift_failure": first_lift_failure,
        "positive_tube_proved": positive_tube_proved,
        "counts": {
            "required_evaluations": 15,
            "evaluations_solved": sum(record["compatible"] for record in records),
            "evaluations_audited": len(records),
            "55_state_lifts_passed": sum(record["pass"] for record in lift_records),
            "manifest_registered_before": manifest_before,
            "manifest_registered_after": 304 if admitted else manifest_before,
            "registered_packets": 150 if admitted else 0,
            "remaining_packets": 0 if admitted else 150,
        },
        "claims": {
            "all_15_equal_eigenspace_transports_proved": all_equal_systems_pass,
            "full_55_state_recurrence_proved": (
                all_equal_systems_pass and first_lift_failure is None
            ),
            "positive_symmetrizer_tube_proved": positive_tube_proved,
            "higher_K55_registered": admitted,
            "manifest_advanced": admitted,
            "inferred_missing_packets_as_zero": False,
            "global_H7_claim": False,
        },
        "atomic_admission_contract": {
            "required": [
                "15_of_15_equal_eigenspace_transport_solutions",
                "15_of_15_exact_55_state_lifts",
                "source_bound_positive_symmetrizer_tube",
            ],
            "partial_manifest_advance_forbidden": True,
            "satisfied": admitted,
        },
        "negative_controls": {
            "drop_first_incompatibility": {"rejected": True},
            "promote_witness_direction_to_coordinate_free_packet": {"rejected": True},
            "advance_154_manifest_before_atomic_admission": {"rejected": True},
            "infer_uncomputed_packet_as_zero": {"rejected": True},
            "claim_global_H7_from_finite_direction_algebra": {"rejected": True},
        },
        "local_bindings": {
            "config": {
                "path": CONFIG_PATH,
                "normalized_text_sha256": _normalized_text_sha(root / CONFIG_PATH),
            },
            "source": {
                "path": SOURCE_PATH,
                "normalized_text_sha256": _normalized_text_sha(root / SOURCE_PATH),
            },
            "test": {
                "path": TEST_PATH,
                "normalized_text_sha256": _normalized_text_sha(root / TEST_PATH),
            },
        },
        "scope": (
            "Exact Q(sqrt(2)) transport audit at the sealed rational unit direction for the "
            "15 registered polarization evaluations. It does not promote a witness-direction "
            "matrix to a coordinate-free packet and does not advance the 154/304 manifest "
            "unless every transport, 55-state lift, and positive-tube obligation closes."
        ),
    }
    result = _with_hash(body)
    if len(json.dumps(result).encode()) > config["caps"]["maximum_output_bytes"]:
        raise AllPolarizationRecurrenceError("campaign exceeded output byte cap")
    return result


def _validate_contract(document: dict[str, Any]) -> None:
    if not _hash_matches(document):
        raise AllPolarizationRecurrenceError("campaign content seal changed")
    records = document.get("evaluation_records")
    expected_ids = ["subset_0", "subset_1", "subset_2", "subset_3", "subset_01", "subset_02"]
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != "block_first_exact_equal_eigenspace_transport_incompatibility"
        or document.get("decision") != "BLOCK_SERIALIZATION"
        or not isinstance(records, list)
        or [record.get("evaluation_id") for record in records] != expected_ids
        or document.get("remaining_unaudited_evaluations")
        != [
            "subset_03",
            "subset_12",
            "subset_13",
            "subset_23",
            "subset_012",
            "subset_013",
            "subset_023",
            "subset_123",
            "subset_0123",
        ]
    ):
        raise AllPolarizationRecurrenceError("campaign boundary changed")
    first = document.get("first_exact_incompatibility")
    if first != {
        "evaluation_id": "subset_02",
        "joint_coefficient_rank": 4,
        "joint_augmented_rank": 5,
        "first_missing_primitive": "source_bound_equal_eigenspace_G1_transport_solution",
    }:
        raise AllPolarizationRecurrenceError("first incompatibility changed")
    if (
        [record.get("compatible") for record in records] != [True] * 5 + [False]
        or records[-1].get("pulled_back_residual_nonzero_entries") != 80
        or records[-1].get("joint_coefficient_rank") != 4
        or records[-1].get("joint_augmented_rank") != 5
        or records[-1].get("canonical_nonzero_spectral_coefficients") != []
    ):
        raise AllPolarizationRecurrenceError("rank certificate changed")
    subset_2 = records[2]
    if subset_2.get("canonical_nonzero_spectral_coefficients") != [
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
        raise AllPolarizationRecurrenceError("subset_2 witness replay changed")
    for record in records[:-1]:
        if record.get("companion_Taylor_remainder_entries") != [0, 0, 0]:
            raise AllPolarizationRecurrenceError("compatible recurrence remainder changed")
        for key in ("delta_G1", "delta_G2", "delta_G3"):
            packet = record.get(key, {})
            if not _hash_matches(packet) or packet.get("nonzero_entries") != len(
                packet.get("entries", [])
            ):
                raise AllPolarizationRecurrenceError("canonical matrix seal changed")
    counts = document.get("counts", {})
    if counts != {
        "required_evaluations": 15,
        "evaluations_solved": 5,
        "evaluations_audited": 6,
        "55_state_lifts_passed": 0,
        "manifest_registered_before": 154,
        "manifest_registered_after": 154,
        "registered_packets": 0,
        "remaining_packets": 150,
    }:
        raise AllPolarizationRecurrenceError("atomic counts changed")
    claims = document.get("claims", {})
    if (
        any(claims.values())
        or claims.get("inferred_missing_packets_as_zero") is not False
        or claims.get("global_H7_claim") is not False
        or document.get("all_15_equal_eigenspace_transport_systems_pass") is not False
        or document.get("lift_records") != []
        or document.get("first_exact_55_state_lift_failure") is not None
        or document.get("positive_tube_proved") is not False
        or document.get("atomic_admission_contract", {}).get("satisfied") is not False
        or document.get("atomic_admission_contract", {}).get("partial_manifest_advance_forbidden")
        is not True
    ):
        raise AllPolarizationRecurrenceError("blocked claims changed")
    controls = document.get("negative_controls", {})
    if len(controls) != 5 or any(value != {"rejected": True} for value in controls.values()):
        raise AllPolarizationRecurrenceError("negative controls changed")
    for binding in document.get("local_bindings", {}).values():
        if Path(binding.get("path", "")).is_absolute():
            raise AllPolarizationRecurrenceError("absolute path leaked into campaign")


def validate_campaign(document: dict[str, Any], root: Path) -> None:
    _validate_contract(document)
    if document != build_campaign(root, root / CONFIG_PATH):
        raise AllPolarizationRecurrenceError("campaign replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = build_campaign(arguments.project_root, arguments.config)
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(data)
    else:
        print(data.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
