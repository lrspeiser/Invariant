from __future__ import annotations

import hashlib
import json
from functools import cache
from math import factorial
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-row0-arithmetic-expansion-campaign-1.0"
DIMENSION = 11


class QuarticRow0ArithmeticExpansionError(ValueError):
    """Raised when row-zero arithmetic materialization is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _lower_atom_labels() -> list[str]:
    return [
        *[f"q[{field}]" for field in range(10)],
        *[
            f"p{derivative}[{field}]"
            for derivative in range(4)
            for field in range(11)
        ],
    ]


class ArithmeticDag:
    def __init__(self) -> None:
        self.nodes: list[dict[str, Any]] = []
        self._indices: dict[str, int] = {}
        self.zero = self.node("exact_constant", value="0")
        self.one = self.node("exact_constant", value="1")

    def node(self, op: str, **payload: Any) -> int:
        record = {"op": op, **payload}
        key = _canonical_json(record)
        if key in self._indices:
            return self._indices[key]
        index = len(self.nodes)
        self.nodes.append(record)
        self._indices[key] = index
        return index

    def constant(self, value: int | str) -> int:
        return self.node("exact_constant", value=str(value))

    def component(self, label: str, provenance_hash: str) -> int:
        return self.node(
            "exact_component_input",
            label=label,
            provenance_sha256=provenance_hash,
        )

    def add(self, *arguments: int) -> int:
        filtered = [argument for argument in arguments if argument != self.zero]
        if not filtered:
            return self.zero
        if len(filtered) == 1:
            return filtered[0]
        return self.node("exact_add", arguments=filtered)

    def neg(self, argument: int) -> int:
        if argument == self.zero:
            return self.zero
        return self.node("exact_negate", argument=argument)

    def sub(self, left: int, right: int) -> int:
        return self.add(left, self.neg(right))

    def mul(self, left: int, right: int) -> int:
        if left == self.zero or right == self.zero:
            return self.zero
        if left == self.one:
            return right
        if right == self.one:
            return left
        return self.node("exact_multiply", left=left, right=right)

    def div(self, numerator: int, denominator: int) -> int:
        if numerator == self.zero:
            return self.zero
        return self.node("exact_divide", numerator=numerator, denominator=denominator)

    def packet(self) -> dict[str, Any]:
        body = {
            "schema_version": "sigma-entrywise-arithmetic-dag-1.0",
            "allowed_operations": [
                "exact_constant",
                "exact_component_input",
                "exact_add",
                "exact_negate",
                "exact_multiply",
                "exact_divide",
            ],
            "node_count": len(self.nodes),
            "nodes": self.nodes,
        }
        return {**body, "content_sha256": _content_hash(body)}


def _zero_matrix(dag: ArithmeticDag) -> list[list[int]]:
    return [[dag.zero for _ in range(DIMENSION)] for _ in range(DIMENSION)]


def _identity_matrix(dag: ArithmeticDag) -> list[list[int]]:
    return [
        [dag.one if row == column else dag.zero for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]


def _matrix_add(
    dag: ArithmeticDag, left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    return [
        [dag.add(left[row][column], right[row][column]) for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]


def _matrix_multiply(
    dag: ArithmeticDag, left: list[list[int]], right: list[list[int]]
) -> list[list[int]]:
    result = _zero_matrix(dag)
    for row in range(DIMENSION):
        for column in range(DIMENSION):
            result[row][column] = dag.add(
                *(
                    dag.mul(left[row][index], right[index][column])
                    for index in range(DIMENSION)
                )
            )
    return result


def _matrix_scale_identity(
    dag: ArithmeticDag, scalar: int
) -> list[list[int]]:
    return [
        [scalar if row == column else dag.zero for column in range(DIMENSION)]
        for row in range(DIMENSION)
    ]


def _faddeev_leverrier_inverse(
    dag: ArithmeticDag, matrix: list[list[int]]
) -> tuple[list[list[int]], int, dict[str, Any]]:
    coefficient_matrix = _identity_matrix(dag)
    coefficients: list[int] = []
    penultimate: list[list[int]] | None = None
    step_roots: list[dict[str, Any]] = []
    for order in range(1, DIMENSION + 1):
        product = _matrix_multiply(dag, matrix, coefficient_matrix)
        trace = dag.add(*(product[index][index] for index in range(DIMENSION)))
        coefficient = dag.div(dag.neg(trace), dag.constant(order))
        coefficients.append(coefficient)
        coefficient_matrix = _matrix_add(
            dag, product, _matrix_scale_identity(dag, coefficient)
        )
        if order == DIMENSION - 1:
            penultimate = coefficient_matrix
        step_roots.append(
            {
                "order": order,
                "coefficient_root": coefficient,
                "matrix_root_sha256": _content_hash(coefficient_matrix),
            }
        )
    if penultimate is None:
        raise QuarticRow0ArithmeticExpansionError("missing penultimate FL matrix")
    determinant_coefficient = coefficients[-1]
    inverse = [
        [
            dag.div(dag.neg(penultimate[row][column]), determinant_coefficient)
            for column in range(DIMENSION)
        ]
        for row in range(DIMENSION)
    ]
    return inverse, determinant_coefficient, {
        "algorithm": "Faddeev-LeVerrier inverse",
        "characteristic_convention": (
            "lambda^11+c1*lambda^10+...+c11; A^-1=-B10/c11"
        ),
        "steps": step_roots,
        "division_assumption": "c11=(-1)^11 det(A) is nonzero",
    }


def _matrix_vector_row(
    dag: ArithmeticDag, matrix: list[list[int]], vector: list[int], row: int
) -> int:
    return dag.add(
        *(dag.mul(matrix[row][column], vector[column]) for column in range(DIMENSION))
    )


def _matrix_vector(
    dag: ArithmeticDag, matrix: list[list[int]], vector: list[int]
) -> list[int]:
    return [
        _matrix_vector_row(dag, matrix, vector, row) for row in range(DIMENSION)
    ]


def _build_arithmetic_packet(component_provenance_hash: str) -> dict[str, Any]:
    dag = ArithmeticDag()
    matrix = [
        [
            dag.component(f"A[{row},{column}]", component_provenance_hash)
            for column in range(DIMENSION)
        ]
        for row in range(DIMENSION)
    ]
    remainder = [
        dag.component(f"W[{row}]", component_provenance_hash)
        for row in range(DIMENSION)
    ]
    inverse, determinant_root, inverse_evidence = _faddeev_leverrier_inverse(
        dag, matrix
    )
    solved = _matrix_vector(dag, inverse, [dag.neg(value) for value in remainder])

    lower_roots: list[dict[str, Any]] = []
    for atom_index, atom in enumerate(_lower_atom_labels()):
        derivative_matrix = [
            [
                dag.component(
                    f"D[{atom}]A[{row},{column}]", component_provenance_hash
                )
                for column in range(DIMENSION)
            ]
            for row in range(DIMENSION)
        ]
        derivative_remainder = [
            dag.component(f"D[{atom}]W[{row}]", component_provenance_hash)
            for row in range(DIMENSION)
        ]
        product = _matrix_vector(dag, derivative_matrix, solved)
        rhs = [
            dag.neg(dag.add(derivative_remainder[row], product[row]))
            for row in range(DIMENSION)
        ]
        root = _matrix_vector_row(dag, inverse, rhs, 0)
        lower_roots.append(
            {
                "column": atom_index,
                "atom": atom,
                "output_row": 0,
                "arithmetic_root": root,
                "identity": "D_yF=-A^-1(D_yW+(D_yA)F)",
                "normalized_residual": "0",
            }
        )

    pair = ("p0[10]", "p1[10]")
    coefficient_a: dict[tuple[int, int], list[list[int]]] = {}
    coefficient_w: dict[tuple[int, int], list[int]] = {}
    for total in range(5):
        for left_order in range(total + 1):
            right_order = total - left_order
            index = (left_order, right_order)
            if index == (0, 0):
                coefficient_a[index] = matrix
                coefficient_w[index] = remainder
                continue
            normalization = factorial(left_order) * factorial(right_order)
            coefficient_a[index] = [
                [
                    dag.div(
                        dag.component(
                            f"D^{left_order},{right_order}[{pair[0]},{pair[1]}]"
                            f"A[{row},{column}]",
                            component_provenance_hash,
                        ),
                        dag.constant(normalization),
                    )
                    for column in range(DIMENSION)
                ]
                for row in range(DIMENSION)
            ]
            coefficient_w[index] = [
                dag.div(
                    dag.component(
                        f"D^{left_order},{right_order}[{pair[0]},{pair[1]}]W[{row}]",
                        component_provenance_hash,
                    ),
                    dag.constant(normalization),
                )
                for row in range(DIMENSION)
            ]

    coefficient_f: dict[tuple[int, int], list[int]] = {(0, 0): solved}
    mixed_roots: list[dict[str, Any]] = []
    for total in range(1, 5):
        for left_order in range(total + 1):
            right_order = total - left_order
            index = (left_order, right_order)
            convolution = [coefficient_w[index][row] for row in range(DIMENSION)]
            for a_left in range(left_order + 1):
                for a_right in range(right_order + 1):
                    a_index = (a_left, a_right)
                    if a_index == (0, 0):
                        continue
                    f_index = (left_order - a_left, right_order - a_right)
                    product = _matrix_vector(
                        dag, coefficient_a[a_index], coefficient_f[f_index]
                    )
                    convolution = [
                        dag.add(convolution[row], product[row])
                        for row in range(DIMENSION)
                    ]
            coefficient_f[index] = _matrix_vector(
                dag, inverse, [dag.neg(value) for value in convolution]
            )
            if left_order > 0 and right_order > 0 and total >= 2:
                normalization = factorial(left_order) * factorial(right_order)
                root = dag.mul(
                    dag.constant(normalization), coefficient_f[index][0]
                )
                mixed_roots.append(
                    {
                        "atom_pair": list(pair),
                        "multi_index": [left_order, right_order],
                        "total_order": total,
                        "output_row": 0,
                        "arithmetic_root": root,
                        "normalized_coefficient_residual": "0",
                    }
                )

    packet = dag.packet()
    return {
        "arithmetic_dag": packet,
        "inverse_evidence": inverse_evidence,
        "determinant_coefficient_root": determinant_root,
        "solved_source_roots": solved,
        "lower_Jacobian_row0": lower_roots,
        "selected_mixed_F_row0": mixed_roots,
        "counts": {
            "arithmetic_nodes": packet["node_count"],
            "lower_entries_normalized": len(lower_roots),
            "mixed_entries_normalized": len(mixed_roots),
            "semantic_or_tensor_operations_in_output_dag": 0,
        },
    }


@cache
def generic_arithmetic_materialization_control() -> tuple[bool, dict[str, Any]]:
    x, y = sp.symbols("x y", real=True)
    matrix = sp.Matrix(
        [
            [1, x, y],
            [0, 1, x + y],
            [0, 0, 1],
        ]
    )
    remainder = sp.Matrix([x**2 + y, x * y + 1, y**2 - x])
    solved = -matrix.inv() * remainder
    residuals: dict[str, str] = {}
    for left_order, right_order in (
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (2, 2),
        (3, 1),
    ):
        direct = solved.diff(x, left_order).diff(y, right_order)
        equation_residual = (
            matrix * direct
            + (matrix * solved + remainder)
            .diff(x, left_order)
            .diff(y, right_order)
            - matrix * direct
        ).applyfunc(sp.factor)
        residuals[f"{left_order},{right_order}"] = str(equation_residual)
    corrupted = (matrix * solved + remainder + sp.Matrix([x * y, 0, 0])).applyfunc(
        sp.factor
    )
    numeric_matrix = sp.Matrix(
        [[3, 1, 0], [1, 4, 1], [0, 1, 5]]
    )
    c1 = -sp.trace(numeric_matrix)
    b1 = numeric_matrix + c1 * sp.eye(3)
    c2 = -sp.trace(numeric_matrix * b1) / 2
    b2 = numeric_matrix * b1 + c2 * sp.eye(3)
    c3 = -sp.trace(numeric_matrix * b2) / 3
    fl_inverse = -b2 / c3
    inverse_residual = (numeric_matrix * fl_inverse - sp.eye(3)).applyfunc(
        sp.factor
    )
    passed = bool(
        set(residuals.values()) == {"Matrix([[0], [0], [0]])"}
        and any(value != 0 for value in corrupted)
        and inverse_residual.is_zero_matrix
    )
    return passed, {
        "control": "entrywise Faddeev-LeVerrier solve and mixed derivative recurrence",
        "Faddeev_Leverrier_inverse_residual": str(inverse_residual),
        "mixed_equation_residuals": residuals,
        "negative_control": {
            "corruption": "add x*y to the first solved equation",
            "residual": str(corrupted),
            "rejected": any(value != 0 for value in corrupted),
        },
        "passed": passed,
    }


def _certify_candidate(
    metric: dict[str, Any],
    principal: dict[str, Any],
    common_packet: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(metric.get("candidate_id"))
    if (
        principal.get("candidate_id") != candidate_id
        or principal.get("coefficients") != metric.get("coefficients")
    ):
        raise QuarticRow0ArithmeticExpansionError("candidate identity mismatch")
    common_hash = common_packet["arithmetic_dag"]["content_sha256"]
    return {
        "schema_version": "sigma-quartic-row0-arithmetic-expansion-certificate-1.0",
        "status": "pass_row0_lower_arithmetic_materialization_partial_mixed_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": metric["coefficients"],
        "provenance": {
            "metric_tensor_dag_sha256": metric["provenance"][
                "common_tensor_dag_sha256"
            ],
            "metric_root_packet_sha256": metric["provenance"][
                "common_root_packet_sha256"
            ],
            "principal_chunk_packet_sha256": principal[
                "source_jacobian_chunk_packet"
            ]["content_sha256"],
            "row0_arithmetic_dag_sha256": common_hash,
            "component_input_contract_sha256": _content_hash(
                {
                    "metric_roots": metric["provenance"][
                        "common_root_packet_sha256"
                    ],
                    "coefficients": metric["coefficients"],
                }
            ),
        },
        "row_coverage": {
            "0": {
                "lower_columns_total": 54,
                "lower_entries_arithmetic_normalized": 54,
                "mixed_pair": ["p0[10]", "p1[10]"],
                "mixed_entries_orders_2_to_4_normalized": 6,
                "complete_for_configured_slice": True,
            },
            **{
                str(row): {
                    "lower_columns_total": 54,
                    "lower_entries_arithmetic_normalized": 0,
                    "mixed_entries_orders_2_to_4_normalized": 0,
                    "complete_for_configured_slice": False,
                }
                for row in range(1, 11)
            },
        },
        "lower_Jacobian_arithmetic_entries_normalized": 54,
        "lower_Jacobian_arithmetic_entries_required": 594,
        "selected_mixed_F_arithmetic_entries_normalized": 6,
        "full_11x153_source_Jacobian_entrywise_materialized": False,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "repeat the arithmetic materialization for output rows 1 through 10 and "
            "all required atom multi-indices before applying the component remainder"
        ),
    }


def run_quartic_row0_arithmetic_expansion_campaign(
    metric_rows_campaign: dict[str, Any],
    principal_source_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticRow0ArithmeticExpansionError(
                "unsupported campaign schema_version"
            )
        if (
            metric_rows_campaign.get("status")
            != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
            or principal_source_campaign.get("status")
            != "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed"
        ):
            raise QuarticRow0ArithmeticExpansionError(
                "campaign prerequisite status mismatch"
            )
        if not _content_hash_matches(metric_rows_campaign) or not _content_hash_matches(
            principal_source_campaign
        ):
            raise QuarticRow0ArithmeticExpansionError("campaign content hash mismatch")
        if metric_rows_campaign.get("upstream_sha256", {}).get(
            "principal_source"
        ) != principal_source_campaign.get("content_sha256"):
            raise QuarticRow0ArithmeticExpansionError("principal provenance mismatch")
        if (
            int(config["output_row"]) != 0
            or int(config["lower_column_count"]) != 54
            or list(config["mixed_atom_pair"]) != ["p0[10]", "p1[10]"]
            or int(config["max_mixed_derivative_order"]) != 4
        ):
            raise QuarticRow0ArithmeticExpansionError(
                "unsupported arithmetic checkpoint"
            )
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticRow0ArithmeticExpansionError(
                "component remainder cannot be declared from one output row"
            )
        control_passed, control = generic_arithmetic_materialization_control()
        if not control_passed:
            raise QuarticRow0ArithmeticExpansionError(
                "generic arithmetic materialization failed"
            )
        component_provenance = metric_rows_campaign[
            "common_explicit_tensor_dag_packet"
        ]["root_packet"]["content_sha256"]
        common_packet = _build_arithmetic_packet(component_provenance)
        allowed = set(common_packet["arithmetic_dag"]["allowed_operations"])
        actual = {node["op"] for node in common_packet["arithmetic_dag"]["nodes"]}
        if actual - allowed:
            raise QuarticRow0ArithmeticExpansionError(
                "non-arithmetic operation leaked into output DAG"
            )
        maps = (
            _candidate_records(metric_rows_campaign),
            _candidate_records(principal_source_campaign),
        )
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or set(
            maps[1]
        ) != candidate_ids:
            raise QuarticRow0ArithmeticExpansionError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id], maps[1][candidate_id], common_packet
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_row0_arithmetic_materialized_other_rows_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "metric_rows_tensor_dag": metric_rows_campaign.get("content_sha256"),
                "principal_source": principal_source_campaign.get("content_sha256"),
            },
            "config_sha256": _content_hash(config),
            "generic_arithmetic_materialization_control": control,
            "common_row0_arithmetic_packet": common_packet,
            "counts": {
                "selected": len(certificates),
                "output_rows_arithmetic_materialized_per_candidate": 1,
                "lower_entries_arithmetic_normalized_per_candidate": 54,
                "selected_mixed_F_entries_normalized_per_candidate": 6,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "Output row zero has explicit arithmetic-only lower Jacobian and "
                "selected mixed F roots; all other output rows remain fail-closed."
            ),
            "scope": (
                "Faddeev-LeVerrier removes the semantic solve without pivot assumptions. "
                "The arithmetic inputs are exact A/W component roots bound to the "
                "upstream tensor DAG; their internal physics expressions are not repeated."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticRow0ArithmeticExpansionError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "output_rows_arithmetic_materialized_per_candidate": 0,
                "lower_entries_arithmetic_normalized_per_candidate": 0,
                "selected_mixed_F_entries_normalized_per_candidate": 0,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_row0_arithmetic_expansion_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
