from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.polys.domains import QQ
from sympy.polys.matrices import DomainMatrix

from . import quartic_tc2_d4_all_polarization_alternative_k55_recurrence as prior
from . import quartic_tc2_d4_alternative_symmetrizer_recurrence_audit as witness
from . import quartic_tc2_d4_coordinate_free_k0_polynomial_packet as poly
from . import quartic_tc2_d4_coordinate_free_k55_order_one_registration as k1
from . import quartic_tc2_d4_physical_metric_transport_no_go as no_go

SCHEMA = "sigma-quartic-tc2-d4-independent-g2-alternative-k55-recurrence-1.0"
CONFIG_PATH = "configs/backgrounds/quartic_tc2_d4_independent_g2_alternative_k55_recurrence.json"
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_independent_g2_alternative_k55_recurrence.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_independent_g2_alternative_k55_recurrence.py"
OUTPUT_PATH = (
    "runs/physics-language/quartic-tc2-d4-independent-g2-alternative-k55-recurrence/campaign.json"
)


class IndependentG2RecurrenceError(ValueError):
    """Raised when the broader exact recurrence fails closed."""


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
        raise IndependentG2RecurrenceError(f"unreadable JSON authority: {path}") from error
    if not isinstance(value, dict):
        raise IndependentG2RecurrenceError(f"non-object JSON authority: {path}")
    return value


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise IndependentG2RecurrenceError("authority path escapes project root")
    return path


def _solve_small(matrix: sp.Matrix, target: sp.Matrix) -> list[Any]:
    if not any(target):
        return [sp.Integer(0)] * matrix.cols
    field = QQ.algebraic_field(sp.sqrt(2))
    reduced, pivots = matrix.row_join(target).to_DM().convert_to(field).rref()
    reduced_matrix = reduced.to_Matrix()
    solution: list[Any] = [sp.Integer(0)] * matrix.cols
    for row, column in enumerate(pivots):
        if column < matrix.cols:
            solution[column] = sp.factor(reduced_matrix[row, matrix.cols])
    return solution


def _canonical_solution(
    order_two_map: sp.Matrix, order_three_g1_map: sp.Matrix, equal_target: sp.Matrix
) -> tuple[int, int, list[Any]]:
    field = QQ.algebraic_field(sp.sqrt(2))
    if not any(equal_target):
        rank = order_two_map.to_DM().convert_to(field).rank()
        return rank, rank, [sp.Integer(0)] * (2 * order_two_map.cols)
    # Prefer the smallest broader solution: if independent equal-block G2
    # freedom alone reaches the target, keep G1 identically zero.
    pure_rank = order_two_map.to_DM().convert_to(field).rank()
    pure_augmented_rank = order_two_map.row_join(equal_target).to_DM().convert_to(field).rank()
    if pure_rank == pure_augmented_rank:
        return (
            pure_rank,
            pure_augmented_rank,
            [sp.Integer(0)] * order_two_map.cols + _solve_small(order_two_map, equal_target),
        )
    # Eliminate the independent G2 variables before RREF.  If L spans the
    # left nullspace of E, solvability of E*z = target-F*x is exactly
    # L*(target-F*x)=0.  This replaces one 60x104 RREF by two small solves.
    left_annihilator = order_two_map.T.to_DM().convert_to(field).nullspace().to_Matrix()
    g1_map = order_two_map.col_join(left_annihilator * order_three_g1_map)
    g1_target = sp.zeros(order_two_map.rows, 1).col_join(left_annihilator * equal_target)
    rank = g1_map.to_DM().convert_to(field).rank()
    augmented_rank = g1_map.row_join(g1_target).to_DM().convert_to(field).rank()
    if rank != augmented_rank:
        return rank, augmented_rank, []
    g1_coefficients = _solve_small(g1_map, g1_target)
    g1_vector = sp.Matrix(g1_coefficients)
    g2_target = equal_target - order_three_g1_map * g1_vector
    g2_coefficients = _solve_small(order_two_map, g2_target)
    if order_two_map * g1_vector != sp.zeros(order_two_map.rows, 1):
        raise IndependentG2RecurrenceError("canonical G1 left order-two kernel")
    if order_three_g1_map * g1_vector + order_two_map * sp.Matrix(g2_coefficients) != equal_target:
        raise IndependentG2RecurrenceError("canonical G1/G2 solution replay failed")
    return rank, augmented_rank, [*g1_coefficients, *g2_coefficients]


def _source_data(root: Path, predecessor_config: dict[str, Any]) -> dict[str, Any]:
    upstreams = {
        name: prior._load_bound(root, binding)
        for name, binding in predecessor_config["upstreams"].items()
    }
    direction = no_go.WITNESS_DIRECTION
    p_axes = [
        k1.exact._matrix_from_packet(packet) for packet in upstreams["flat_P55"]["matrix_packets"]
    ]
    h_plus = k1.exact._matrix_from_packet(
        upstreams["flat_action_metric"]["exact_construction"]["h_plus_0"]
    )
    recipes = upstreams["projector_recipes"]["exact_Lagrange_projector_recipes"]["recipes"]
    polynomial_base = k1._base_data(p_axes, h_plus, recipes)
    base = prior._evaluated_base(polynomial_base, direction)
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
    checkpoints = {
        name: _load(_inside(root, path))
        for name, path in predecessor_config["checkpoint_authorities"].items()
    }
    if any(not prior._hash_matches(value) for value in checkpoints.values()):
        raise IndependentG2RecurrenceError("checkpoint authority changed")
    return {
        "upstreams": upstreams,
        "polynomial_base": polynomial_base,
        "base": base,
        "p1": p1_by_id,
        "h1": h1_by_id,
        "higher_p": higher_p,
        "higher_h": higher_h,
        "checkpoints": checkpoints,
    }


def _series_for(
    evaluation_id: str, data: dict[str, Any]
) -> tuple[list[sp.Matrix], list[sp.Matrix]]:
    direction = no_go.WITNESS_DIRECTION
    polynomial_base = data["polynomial_base"]
    p_series = [
        no_go._sympy_matrix(poly._evaluate(polynomial_base["P0"], direction, 55, 55)),
        prior._packet_matrix(
            data["p1"][evaluation_id]["P55_Taylor_order_one_matrix"],
            direction,
            [55, 55],
        ),
        *[
            prior._packet_matrix(data["higher_p"][evaluation_id][order], direction, [55, 55])
            for order in prior.ORDERS
        ],
    ]
    h_series = [
        no_go._sympy_matrix(poly._evaluate(polynomial_base["H0"], direction, 22, 22)),
        prior._packet_matrix(
            data["h1"][evaluation_id]["H_star_plus_order_one_matrix"],
            direction,
            [22, 22],
        ),
        *[
            prior._packet_matrix(data["higher_h"][evaluation_id][order], direction, [22, 22])
            for order in prior.ORDERS
        ],
    ]
    return p_series, h_series


def _dm_construct_k55(
    base: dict[str, Any], p_series: list[sp.Matrix], h_series: list[sp.Matrix]
) -> list[Any]:
    """Run the same finite Taylor recurrence over the explicit Q(sqrt(2)) field."""
    field = QQ.algebraic_field(sp.sqrt(2))

    def convert(matrix: sp.Matrix) -> Any:
        return matrix.to_DM().convert_to(field)

    def zero(rows: int, columns: int) -> Any:
        return DomainMatrix.zeros((rows, columns), field)

    def constant(matrix: Any) -> list[Any]:
        return [matrix, *[zero(matrix.shape[0], matrix.shape[1]) for _ in range(4)]]

    def add(left: list[Any], right: list[Any]) -> list[Any]:
        return [left[index] + right[index] for index in range(5)]

    def scale(series: list[Any], coefficient: Any) -> list[Any]:
        scalar = field.convert(coefficient)
        return [item.scalarmul(scalar) for item in series]

    def transpose(series: list[Any]) -> list[Any]:
        return [item.transpose() for item in series]

    def multiply(left: list[Any], right: list[Any]) -> list[Any]:
        result = []
        for order in range(5):
            coefficient = zero(left[0].shape[0], right[0].shape[1])
            for index in range(order + 1):
                coefficient += left[index] * right[order - index]
            result.append(coefficient)
        return result

    j = constant(convert(base["J"]))
    jt = constant(convert(base["JT"]))
    transverse = constant(convert(base["T"]))
    p_dm = [convert(item) for item in p_series]
    h_dm = [convert(item) for item in h_series]
    companion = multiply(multiply(jt, p_dm), j)
    lift = multiply(multiply(jt, p_dm), transverse)
    projector0 = {eigenvalue: convert(item) for eigenvalue, item in base["Pi0"].items()}
    projectors = {eigenvalue: constant(item) for eigenvalue, item in projector0.items()}
    identity = DomainMatrix.eye((22, 22), field)
    for order in range(1, 5):
        for eigenvalue, series in projectors.items():
            p0 = series[0]
            complement = identity - p0
            commutator = zero(22, 22)
            for index in range(1, order + 1):
                commutator += (
                    companion[index] * series[order - index]
                    - series[order - index] * companion[index]
                )
            off_diagonal = zero(22, 22)
            for other, other_projector in projector0.items():
                if other == eigenvalue:
                    continue
                first_gap = field.convert(
                    sp.Rational(
                        (eigenvalue - other).numerator,
                        (eigenvalue - other).denominator,
                    )
                )
                second_gap = field.convert(
                    sp.Rational(
                        (other - eigenvalue).numerator,
                        (other - eigenvalue).denominator,
                    )
                )
                off_diagonal += (other_projector * commutator * p0).scalarmul(field.one / first_gap)
                off_diagonal += (p0 * commutator * other_projector).scalarmul(
                    field.one / second_gap
                )
            quadratic = zero(22, 22)
            for index in range(1, order):
                quadratic += series[index] * series[order - index]
            series[order] = off_diagonal - p0 * quadratic * p0 + complement * quadratic * complement
    energy = [zero(22, 22) for _ in range(5)]
    identity_series = constant(identity)
    for eigenvalue, projector in projectors.items():
        metric = (
            h_dm if eigenvalue == 1 else scale(h_dm, -1) if eigenvalue == -1 else identity_series
        )
        energy = add(energy, multiply(multiply(transpose(projector), metric), projector))
    inverse = constant(convert(base["N0"]))
    for order in range(1, 5):
        forcing = zero(22, 22)
        for index in range(1, order + 1):
            forcing += companion[index] * inverse[order - index]
        inverse[order] = -(inverse[0] * forcing)
    cross = multiply(multiply(transpose(lift), energy), inverse)
    return add(
        transverse,
        add(
            add(multiply(cross, jt), multiply(j, transpose(cross))),
            multiply(multiply(j, energy), jt),
        ),
    )


def _dm_coefficient(left: list[Any], right: list[Any], order: int) -> Any:
    field = left[0].domain
    result = DomainMatrix.zeros((left[0].shape[0], right[0].shape[1]), field)
    for index in range(order + 1):
        result += left[index] * right[order - index]
    return result


def _pulled_residual(
    evaluation_id: str,
    data: dict[str, Any],
    p_series: list[sp.Matrix],
    h_series: list[sp.Matrix],
) -> tuple[sp.Matrix, str]:
    base = data["base"]
    if evaluation_id in {"subset_0", "subset_1"}:
        checkpoint = data["checkpoints"][evaluation_id]
        if (
            checkpoint.get("evaluation_id") != evaluation_id
            or checkpoint.get("sphere_identity_remainders") != 0
        ):
            raise IndependentG2RecurrenceError("successful checkpoint changed")
        return sp.zeros(22), "sealed_successful_coordinate_free_K55_checkpoint"
    if evaluation_id == "subset_2":
        failure = data["checkpoints"]["subset_2"]
        residual55 = k1._sphere_packet(failure["sphere_symmetrizer_residual"], [55, 55])
        polynomial_base = data["polynomial_base"]
        pulled = no_go._sympy_matrix(
            poly._evaluate(
                poly._multiply(
                    poly._multiply(polynomial_base["JT"], residual55),
                    polynomial_base["J"],
                ),
                no_go.WITNESS_DIRECTION,
                22,
                22,
            )
        )
        return pulled, "sealed_subset_2_failure_checkpoint"
    k_series = _dm_construct_k55(base, p_series, h_series)
    field = k_series[0].domain
    p_dm = [item.to_DM().convert_to(field) for item in p_series]
    residual55 = _dm_coefficient(k_series, p_dm, 3) - _dm_coefficient(
        [item.transpose() for item in p_dm], k_series, 3
    )
    pulled = base["JT"].to_DM().convert_to(field) * residual55 * base["J"].to_DM().convert_to(field)
    return pulled.to_Matrix(), "exact_evaluated_authoritative_55_state_recurrence"


def _coefficient_records(
    coefficients: list[Any], labels: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {**labels[index], "coefficient": sp.sstr(value)}
        for index, value in enumerate(coefficients)
        if value != 0
    ]


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


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config = _load(config_path)
    if not _hash_matches(config):
        raise IndependentG2RecurrenceError("configuration seal changed")
    predecessor = _load(_inside(root, config["predecessor"]["path"]))
    if (
        not prior._hash_matches(predecessor)
        or predecessor.get("content_sha256") != config["predecessor"]["content_sha256"]
        or predecessor.get("first_exact_incompatibility", {}).get("evaluation_id") != "subset_02"
        or predecessor.get("first_exact_incompatibility", {}).get("joint_coefficient_rank") != 4
        or predecessor.get("first_exact_incompatibility", {}).get("joint_augmented_rank") != 5
    ):
        raise IndependentG2RecurrenceError("predecessor boundary changed")
    predecessor_config = _load(root / prior.CONFIG_PATH)
    if not prior._hash_matches(predecessor_config) or predecessor_config.get(
        "content_sha256"
    ) != predecessor.get("config_sha256"):
        raise IndependentG2RecurrenceError("predecessor config authority changed")
    data = _source_data(root, predecessor_config)
    evaluation_ids = config["evaluation_ids"]
    if evaluation_ids != list(data["p1"]):
        raise IndependentG2RecurrenceError("evaluation order changed")

    polynomial_base = data["polynomial_base"]
    base = data["base"]
    basis, eigenvalues, blocks, spectral_authority = witness._spectral_basis(
        polynomial_base, no_go.WITNESS_DIRECTION
    )
    inverse_basis = basis.inv()
    spectral_basis, labels = witness._block_symmetric_basis(blocks)
    coordinate_basis = [inverse_basis.T * item * inverse_basis for item in spectral_basis]
    records: list[dict[str, Any]] = []
    solutions: dict[str, tuple[sp.Matrix, sp.Matrix, sp.Matrix]] = {}
    first_incompatibility: dict[str, Any] | None = None

    for evaluation_id in evaluation_ids:
        p_series, h_series = _series_for(evaluation_id, data)
        residual, target_authority = _pulled_residual(evaluation_id, data, p_series, h_series)
        c1 = base["JT"] * p_series[1] * base["J"]
        c2 = base["JT"] * p_series[2] * base["J"]
        forcing2_columns = [metric * c1 - c1.T * metric for metric in coordinate_basis]
        order_two_map = sp.Matrix.hstack(
            *[witness._equal_block_vector(forcing, basis, blocks) for forcing in forcing2_columns]
        ).applyfunc(sp.factor)
        order_three_g1_columns = []
        for forcing2, metric in zip(forcing2_columns, coordinate_basis, strict=True):
            cross_g2 = prior._relaxed_next_coefficient(forcing2, basis, inverse_basis, eigenvalues)
            forcing3 = cross_g2 * c1 - c1.T * cross_g2 + metric * c2 - c2.T * metric
            order_three_g1_columns.append(witness._equal_block_vector(forcing3, basis, blocks))
        order_three_g1_map = sp.Matrix.hstack(*order_three_g1_columns).applyfunc(sp.factor)
        # An independent symmetric equal-eigenspace G2 coefficient contributes
        # the same equal-block transport map at the next Taylor order.
        equal_target = witness._equal_block_vector(-residual, basis, blocks)
        rank, augmented_rank, coefficients = _canonical_solution(
            order_two_map, order_three_g1_map, equal_target
        )
        record: dict[str, Any] = {
            "evaluation_id": evaluation_id,
            "direction": config["witness_direction"],
            "target_authority": target_authority,
            "pulled_back_residual_nonzero_entries": sum(item != 0 for item in residual),
            "order_two_equal_eigenspace_rank": order_two_map.to_DM().rank(),
            "G1_only_joint_rank": order_two_map.col_join(order_three_g1_map).to_DM().rank(),
            "broader_joint_unknowns": 2 * order_two_map.cols,
            "canonical_route_coefficient_rank": rank,
            "canonical_route_augmented_rank": augmented_rank,
            "compatible": rank == augmented_rank,
        }
        if rank != augmented_rank:
            record["canonical_G1_nonzero_spectral_coefficients"] = []
            record["canonical_equal_G2_nonzero_spectral_coefficients"] = []
            records.append(record)
            first_incompatibility = {
                "evaluation_id": evaluation_id,
                "canonical_route_coefficient_rank": rank,
                "canonical_route_augmented_rank": augmented_rank,
                "first_missing_primitive": ("symmetric_equal_eigenspace_G1_G2_transport_solution"),
            }
            break

        split = len(coordinate_basis)
        g1_coefficients = coefficients[:split]
        equal_g2_coefficients = coefficients[split:]
        delta_g1 = sum(
            (
                coefficient * metric
                for coefficient, metric in zip(g1_coefficients, coordinate_basis, strict=True)
            ),
            sp.zeros(22),
        ).applyfunc(sp.factor)
        forcing2 = (delta_g1 * c1 - c1.T * delta_g1).applyfunc(sp.factor)
        delta_g2 = prior._relaxed_next_coefficient(forcing2, basis, inverse_basis, eigenvalues)
        delta_g2 += sum(
            (
                coefficient * metric
                for coefficient, metric in zip(equal_g2_coefficients, coordinate_basis, strict=True)
            ),
            sp.zeros(22),
        )
        delta_g2 = delta_g2.applyfunc(sp.factor)
        residual_before_g3 = (
            residual + delta_g2 * c1 - c1.T * delta_g2 + delta_g1 * c2 - c2.T * delta_g1
        ).applyfunc(sp.factor)
        if any(witness._equal_block_vector(residual_before_g3, basis, blocks)):
            raise IndependentG2RecurrenceError("broader equal block did not cancel")
        delta_g3 = witness._canonical_next_coefficient(
            residual_before_g3, basis, inverse_basis, eigenvalues
        ).applyfunc(sp.factor)
        c0 = no_go._sympy_matrix(
            poly._evaluate(polynomial_base["C0"], no_go.WITNESS_DIRECTION, 22, 22)
        )
        remainders = [
            delta_g1 * c0 - c0.T * delta_g1,
            delta_g2 * c0 - c0.T * delta_g2 + forcing2,
            delta_g3 * c0 - c0.T * delta_g3 + residual_before_g3,
        ]
        remainder_counts = [
            sum(item != 0 for item in remainder.applyfunc(sp.factor)) for remainder in remainders
        ]
        if remainder_counts != [0, 0, 0]:
            raise IndependentG2RecurrenceError("broader recurrence replay failed")
        record.update(
            {
                "canonical_G1_nonzero_spectral_coefficients": _coefficient_records(
                    g1_coefficients, labels
                ),
                "canonical_equal_G2_nonzero_spectral_coefficients": _coefficient_records(
                    equal_g2_coefficients, labels
                ),
                "delta_G1": _matrix_summary(f"delta_G1_{evaluation_id}", delta_g1),
                "delta_G2": _matrix_summary(f"delta_G2_{evaluation_id}", delta_g2),
                "delta_G3": _matrix_summary(f"delta_G3_{evaluation_id}", delta_g3),
                "companion_Taylor_remainder_entries": remainder_counts,
            }
        )
        records.append(record)
        solutions[evaluation_id] = (delta_g1, delta_g2, delta_g3)

    all_transport_pass = len(records) == 15 and first_incompatibility is None
    lift_records: list[dict[str, Any]] = []
    first_lift_failure: dict[str, Any] | None = None
    corrected_metrics: dict[str, list[sp.Matrix]] = {}
    if all_transport_pass:
        for evaluation_id in evaluation_ids:
            p_series, h_series = _series_for(evaluation_id, data)
            for order, correction in zip((1, 2, 3), solutions[evaluation_id], strict=True):
                h_series[order] += correction
            corrected_metrics[evaluation_id] = h_series
            k_series = _dm_construct_k55(base, p_series, h_series)
            field = k_series[0].domain
            p_dm = [item.to_DM().convert_to(field) for item in p_series]
            remainder_counts = []
            symmetry_counts = []
            for order in (1, 2, 3):
                symmetry = k_series[order] - k_series[order].transpose()
                remainder = _dm_coefficient(k_series, p_dm, order) - _dm_coefficient(
                    [item.transpose() for item in p_dm], k_series, order
                )
                symmetry_counts.append(symmetry.nnz())
                remainder_counts.append(remainder.nnz())
            passed = symmetry_counts == [0, 0, 0] and remainder_counts == [0, 0, 0]
            lift_records.append(
                {
                    "evaluation_id": evaluation_id,
                    "K55_symmetry_remainder_entries": symmetry_counts,
                    "K55_symmetrizer_remainder_entries": remainder_counts,
                    "pass": passed,
                }
            )
            if not passed:
                first_lift_failure = {
                    "evaluation_id": evaluation_id,
                    "K55_symmetry_remainder_entries": symmetry_counts,
                    "K55_symmetrizer_remainder_entries": remainder_counts,
                    "first_missing_primitive": (
                        "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
                    ),
                }
                break

    all_lifts_pass = all_transport_pass and first_lift_failure is None
    positive_tube_proved = False
    tube_records: list[dict[str, Any]] = []
    # Positivity is intentionally attempted only after all fifteen exact lifts.
    if all_lifts_pass:
        epsilon = sp.Rational(1, 10**6)
        for evaluation_id in evaluation_ids:
            series = corrected_metrics[evaluation_id]
            h0 = series[0]
            lower, diagonal = h0.LDLdecomposition(hermitian=False)
            inverse_lower = lower.inv()
            transformed = [
                (inverse_lower * series[order] * inverse_lower.T).applyfunc(sp.factor)
                for order in (1, 2, 3)
            ]
            margins = []
            for row in range(22):
                perturbation_bound = sum(
                    epsilon**order * sum(sp.Abs(matrix[row, column]) for column in range(22))
                    for order, matrix in enumerate(transformed, start=1)
                )
                margins.append(sp.factor(diagonal[row, row] - perturbation_bound))
            passed = all(bool(margin > 0) for margin in margins)
            tube_records.append(
                {
                    "evaluation_id": evaluation_id,
                    "epsilon": "1/1000000",
                    "LDL_row_margin_positive": passed,
                    "minimum_numeric_margin": str(min(sp.N(margin, 12) for margin in margins)),
                }
            )
            if not passed:
                break
        positive_tube_proved = len(tube_records) == 15 and all(
            record["LDL_row_margin_positive"] for record in tube_records
        )

    admitted = all_transport_pass and all_lifts_pass and positive_tube_proved
    manifest_before = config["manifest"]["registered_before"]
    body = {
        "schema_version": SCHEMA,
        "status": (
            "pass_all_15_independent_G2_transport_lift_and_positive_tube"
            if admitted
            else "block_first_exact_broader_symmetrizer_obstruction"
        ),
        "decision": "REGISTER_ATOMICALLY" if admitted else "BLOCK_SERIALIZATION",
        "config_sha256": config["content_sha256"],
        "predecessor_binding": {**config["predecessor"], "verified": True},
        "preserved_authorities": {
            "P55_or_action_mutated": False,
            "predecessor_upstreams": predecessor_config["upstreams"],
        },
        "direction": config["witness_direction"],
        "unit_sphere_identity": "(3/5)^2+(4/5)^2-1=0",
        "broader_ansatz": {
            "G1_equal_eigenspace_symmetric_unknowns": 52,
            "G2_equal_eigenspace_symmetric_unknowns": 52,
            "joint_unknowns": 104,
            "G3_equal_eigenspace_unknowns_used": 0,
            "reason_G3_equal_freedom_is_irrelevant_at_order_three": (
                "equal-eigenspace G3 commutes with C0 and cannot change the order-three target"
            ),
        },
        "spectral_basis_authority": spectral_authority,
        "evaluation_records": records,
        "remaining_unaudited_evaluations": evaluation_ids[len(records) :],
        "first_exact_broader_transport_incompatibility": first_incompatibility,
        "all_15_broader_transport_systems_pass": all_transport_pass,
        "lift_records": lift_records,
        "first_exact_55_state_lift_failure": first_lift_failure,
        "all_15_exact_55_state_lifts_pass": all_lifts_pass,
        "positive_tube_records": tube_records,
        "positive_tube_proved": positive_tube_proved,
        "counts": {
            "required_evaluations": 15,
            "transport_evaluations_audited": len(records),
            "transport_evaluations_solved": sum(record["compatible"] for record in records),
            "55_state_lifts_passed": sum(record["pass"] for record in lift_records),
            "positive_tubes_proved": sum(
                record["LDL_row_margin_positive"] for record in tube_records
            ),
            "manifest_registered_before": manifest_before,
            "manifest_registered_after": 304 if admitted else manifest_before,
            "registered_packets": 150 if admitted else 0,
            "remaining_packets": 0 if admitted else 150,
        },
        "claims": {
            "all_15_broader_transports_proved": all_transport_pass,
            "all_15_exact_55_state_lifts_proved": all_lifts_pass,
            "positive_symmetrizer_tube_proved": positive_tube_proved,
            "higher_K55_registered": admitted,
            "manifest_advanced": admitted,
            "missing_packets_inferred_as_zero": False,
            "global_H7_claim": False,
        },
        "atomic_admission_contract": {
            "required": [
                "15_of_15_broader_transport_solutions",
                "15_of_15_exact_55_state_lifts",
                "15_of_15_positive_symmetrizer_tubes",
            ],
            "partial_manifest_advance_forbidden": True,
            "satisfied": admitted,
        },
        "negative_controls": {
            "mutate_sealed_P55_or_action_authority": {"rejected": True},
            "use_nonsymmetric_equal_eigenspace_freedom": {"rejected": True},
            "promote_witness_direction_to_coordinate_free_packet": {"rejected": True},
            "advance_manifest_before_all_15_and_positivity": {"rejected": True},
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
            "Exact Q(sqrt(2)) audit at the sealed rational direction. The only new freedom is "
            "independently source-bound symmetric equal-eigenspace G2 transport. No action or "
            "P55 authority is changed, and no witness matrix is promoted to a coordinate-free "
            "packet. Manifest admission remains atomic and no global H7 claim is made."
        ),
    }
    result = _with_hash(body)
    if len(json.dumps(result).encode()) > config["caps"]["maximum_output_bytes"]:
        raise IndependentG2RecurrenceError("campaign exceeded output byte cap")
    return result


def _validate_contract(document: dict[str, Any]) -> None:
    if not _hash_matches(document):
        raise IndependentG2RecurrenceError("campaign content seal changed")
    records = document.get("evaluation_records")
    expected_ids = [
        "subset_0",
        "subset_1",
        "subset_2",
        "subset_3",
        "subset_01",
        "subset_02",
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
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != "block_first_exact_broader_symmetrizer_obstruction"
        or document.get("decision") != "BLOCK_SERIALIZATION"
        or not isinstance(records, list)
        or [record.get("evaluation_id") for record in records] != expected_ids
        or document.get("remaining_unaudited_evaluations") != []
        or document.get("first_exact_broader_transport_incompatibility") is not None
        or document.get("all_15_broader_transport_systems_pass") is not True
    ):
        raise IndependentG2RecurrenceError("broader transport boundary changed")
    if any(
        not record.get("compatible")
        or record.get("canonical_route_coefficient_rank")
        != record.get("canonical_route_augmented_rank")
        or record.get("companion_Taylor_remainder_entries") != [0, 0, 0]
        for record in records
    ):
        raise IndependentG2RecurrenceError("transport certificate changed")
    for record in records:
        for key in ("delta_G1", "delta_G2", "delta_G3"):
            packet = record.get(key, {})
            if not _hash_matches(packet) or packet.get("nonzero_entries") != len(
                packet.get("entries", [])
            ):
                raise IndependentG2RecurrenceError("canonical matrix seal changed")
    subset_2 = records[2]
    if (
        subset_2.get("canonical_G1_nonzero_spectral_coefficients")
        != [
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
        ]
        or subset_2.get("canonical_equal_G2_nonzero_spectral_coefficients") != []
    ):
        raise IndependentG2RecurrenceError("subset_2 predecessor witness changed")
    subset_02 = records[5]
    if (
        subset_02.get("pulled_back_residual_nonzero_entries") != 80
        or subset_02.get("canonical_route_coefficient_rank") != 4
        or subset_02.get("canonical_route_augmented_rank") != 4
        or subset_02.get("canonical_G1_nonzero_spectral_coefficients") != []
        or subset_02.get("canonical_equal_G2_nonzero_spectral_coefficients")
        != [
            {
                "eigenvalue": "1",
                "spectral_row": 0,
                "spectral_column": 2,
                "coefficient": "22528*sqrt(2)/140625",
            },
            {
                "eigenvalue": "-1",
                "spectral_row": 3,
                "spectral_column": 5,
                "coefficient": "-22528*sqrt(2)/140625",
            },
        ]
    ):
        raise IndependentG2RecurrenceError("subset_02 broader solution changed")
    lifts = document.get("lift_records")
    if (
        not isinstance(lifts, list)
        or [record.get("evaluation_id") for record in lifts] != ["subset_0", "subset_1", "subset_2"]
        or [record.get("pass") for record in lifts] != [True, True, False]
        or lifts[-1].get("K55_symmetry_remainder_entries") != [0, 0, 0]
        or lifts[-1].get("K55_symmetrizer_remainder_entries") != [0, 0, 72]
        or document.get("first_exact_55_state_lift_failure")
        != {
            "evaluation_id": "subset_2",
            "K55_symmetry_remainder_entries": [0, 0, 0],
            "K55_symmetrizer_remainder_entries": [0, 0, 72],
            "first_missing_primitive": (
                "exact_55_state_transverse_cross_lift_of_independent_G2_metric"
            ),
        }
        or document.get("all_15_exact_55_state_lifts_pass") is not False
        or document.get("positive_tube_records") != []
        or document.get("positive_tube_proved") is not False
    ):
        raise IndependentG2RecurrenceError("55-state lift boundary changed")
    if document.get("counts") != {
        "required_evaluations": 15,
        "transport_evaluations_audited": 15,
        "transport_evaluations_solved": 15,
        "55_state_lifts_passed": 2,
        "positive_tubes_proved": 0,
        "manifest_registered_before": 154,
        "manifest_registered_after": 154,
        "registered_packets": 0,
        "remaining_packets": 150,
    }:
        raise IndependentG2RecurrenceError("atomic counts changed")
    if document.get("claims") != {
        "all_15_broader_transports_proved": True,
        "all_15_exact_55_state_lifts_proved": False,
        "positive_symmetrizer_tube_proved": False,
        "higher_K55_registered": False,
        "manifest_advanced": False,
        "missing_packets_inferred_as_zero": False,
        "global_H7_claim": False,
    }:
        raise IndependentG2RecurrenceError("claims changed")
    if (
        document.get("preserved_authorities", {}).get("P55_or_action_mutated") is not False
        or document.get("atomic_admission_contract", {}).get("satisfied") is not False
        or document.get("atomic_admission_contract", {}).get("partial_manifest_advance_forbidden")
        is not True
    ):
        raise IndependentG2RecurrenceError("authority or admission contract changed")
    controls = document.get("negative_controls", {})
    if len(controls) != 6 or any(value != {"rejected": True} for value in controls.values()):
        raise IndependentG2RecurrenceError("negative controls changed")
    for binding in document.get("local_bindings", {}).values():
        if Path(binding.get("path", "")).is_absolute():
            raise IndependentG2RecurrenceError("absolute path leaked into campaign")


def validate_campaign(document: dict[str, Any], root: Path) -> None:
    _validate_contract(document)
    if document != build_campaign(root, root / CONFIG_PATH):
        raise IndependentG2RecurrenceError("campaign replay mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    result = build_campaign(arguments.project_root, arguments.config)
    data = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    if arguments.output is None:
        print(data.decode(), end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_bytes(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
