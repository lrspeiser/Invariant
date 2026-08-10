from __future__ import annotations

from math import factorial
from typing import Any

from .quartic_row0_arithmetic_expansion_campaign import (
    ArithmeticDag,
    _faddeev_leverrier_inverse,
    _lower_atom_labels,
    _matrix_vector,
    _matrix_vector_row,
)

DIMENSION = 11


def build_output_rows_arithmetic_packet(
    output_rows: tuple[int, ...], component_provenance_hash: str
) -> dict[str, Any]:
    if not output_rows or any(not 0 <= row < DIMENSION for row in output_rows):
        raise ValueError("output rows must be a nonempty subset of 0..10")
    if len(set(output_rows)) != len(output_rows):
        raise ValueError("output rows must be unique")
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
        for output_row in output_rows:
            lower_roots.append(
                {
                    "column": atom_index,
                    "atom": atom,
                    "output_row": output_row,
                    "arithmetic_root": _matrix_vector_row(
                        dag, inverse, rhs, output_row
                    ),
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
                for output_row in output_rows:
                    mixed_roots.append(
                        {
                            "atom_pair": list(pair),
                            "multi_index": [left_order, right_order],
                            "total_order": total,
                            "output_row": output_row,
                            "arithmetic_root": dag.mul(
                                dag.constant(normalization),
                                coefficient_f[index][output_row],
                            ),
                            "normalized_coefficient_residual": "0",
                        }
                    )

    packet = dag.packet()
    return {
        "arithmetic_dag": packet,
        "output_rows": list(output_rows),
        "inverse_evidence": inverse_evidence,
        "determinant_coefficient_root": determinant_root,
        "solved_source_roots": solved,
        "lower_Jacobian": lower_roots,
        "selected_mixed_F": mixed_roots,
        "counts": {
            "arithmetic_nodes": packet["node_count"],
            "output_rows": len(output_rows),
            "lower_entries_normalized": len(lower_roots),
            "mixed_entries_normalized": len(mixed_roots),
            "semantic_or_tensor_operations_in_output_dag": 0,
        },
    }
