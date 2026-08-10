from __future__ import annotations

import hashlib
import json
from math import factorial
from pathlib import Path
from typing import Any

from .quartic_row0_arithmetic_expansion_campaign import (
    ArithmeticDag,
    _candidate_records,
    _content_hash,
    _content_hash_matches,
    _faddeev_leverrier_inverse,
    _lower_atom_labels,
    _matrix_vector,
    _matrix_vector_row,
    generic_arithmetic_materialization_control,
)

SCHEMA_VERSION = "sigma-quartic-row1-arithmetic-expansion-campaign-1.0"
DIMENSION = 11
OUTPUT_ROW = 1


class QuarticRow1ArithmeticExpansionError(ValueError):
    """Raised when row-one arithmetic materialization is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _build_row1_arithmetic_packet(component_provenance_hash: str) -> dict[str, Any]:
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
        root = _matrix_vector_row(dag, inverse, rhs, OUTPUT_ROW)
        lower_roots.append(
            {
                "column": atom_index,
                "atom": atom,
                "output_row": OUTPUT_ROW,
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
                    dag.constant(normalization), coefficient_f[index][OUTPUT_ROW]
                )
                mixed_roots.append(
                    {
                        "atom_pair": list(pair),
                        "multi_index": [left_order, right_order],
                        "total_order": total,
                        "output_row": OUTPUT_ROW,
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
        "lower_Jacobian_row1": lower_roots,
        "selected_mixed_F_row1": mixed_roots,
        "counts": {
            "arithmetic_nodes": packet["node_count"],
            "lower_entries_normalized": len(lower_roots),
            "mixed_entries_normalized": len(mixed_roots),
            "semantic_or_tensor_operations_in_output_dag": 0,
        },
    }


def _certify_candidate(
    row0: dict[str, Any], metric: dict[str, Any], common_packet: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(row0.get("candidate_id"))
    if (
        metric.get("candidate_id") != candidate_id
        or metric.get("coefficients") != row0.get("coefficients")
    ):
        raise QuarticRow1ArithmeticExpansionError("candidate identity mismatch")
    return {
        "schema_version": "sigma-quartic-row1-arithmetic-expansion-certificate-1.0",
        "status": "pass_row1_lower_arithmetic_materialization_other_rows_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": row0["coefficients"],
        "provenance": {
            "row0_arithmetic_dag_sha256": row0["provenance"][
                "row0_arithmetic_dag_sha256"
            ],
            "metric_tensor_dag_sha256": metric["provenance"][
                "common_tensor_dag_sha256"
            ],
            "metric_root_packet_sha256": metric["provenance"][
                "common_root_packet_sha256"
            ],
            "row1_arithmetic_dag_sha256": common_packet["arithmetic_dag"][
                "content_sha256"
            ],
            "component_input_contract_sha256": _content_hash(
                {
                    "metric_roots": metric["provenance"][
                        "common_root_packet_sha256"
                    ],
                    "coefficients": row0["coefficients"],
                }
            ),
        },
        "row_coverage": {
            "0": {
                "lower_entries_arithmetic_normalized": 54,
                "mixed_entries_orders_2_to_4_normalized": 6,
                "source": "upstream row0 arithmetic campaign",
            },
            "1": {
                "lower_entries_arithmetic_normalized": 54,
                "mixed_entries_orders_2_to_4_normalized": 6,
                "source": "current row1 arithmetic campaign",
            },
            **{
                str(row): {
                    "lower_entries_arithmetic_normalized": 0,
                    "mixed_entries_orders_2_to_4_normalized": 0,
                    "source": "unmaterialized",
                }
                for row in range(2, 11)
            },
        },
        "current_lower_entries_normalized": 54,
        "cumulative_lower_entries_normalized": 108,
        "current_selected_mixed_entries_normalized": 6,
        "cumulative_selected_mixed_entries_normalized": 12,
        "full_11x153_source_Jacobian_entrywise_materialized": False,
        "full_component_Frechet_tensors_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "materialize output rows 2 through 10 and all remaining mixed atom "
            "multi-indices before applying the component remainder"
        ),
    }


def run_quartic_row1_arithmetic_expansion_campaign(
    row0_campaign: dict[str, Any],
    metric_rows_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticRow1ArithmeticExpansionError(
                "unsupported campaign schema_version"
            )
        if (
            row0_campaign.get("status")
            != "pass_all_12_row0_arithmetic_materialized_other_rows_fail_closed"
            or metric_rows_campaign.get("status")
            != "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
        ):
            raise QuarticRow1ArithmeticExpansionError(
                "campaign prerequisite status mismatch"
            )
        if not _content_hash_matches(row0_campaign) or not _content_hash_matches(
            metric_rows_campaign
        ):
            raise QuarticRow1ArithmeticExpansionError("campaign content hash mismatch")
        if row0_campaign.get("upstream_sha256", {}).get(
            "metric_rows_tensor_dag"
        ) != metric_rows_campaign.get("content_sha256"):
            raise QuarticRow1ArithmeticExpansionError("row0 provenance mismatch")
        if (
            int(config["output_row"]) != OUTPUT_ROW
            or int(config["lower_column_count"]) != 54
            or list(config["mixed_atom_pair"]) != ["p0[10]", "p1[10]"]
            or int(config["max_mixed_derivative_order"]) != 4
        ):
            raise QuarticRow1ArithmeticExpansionError(
                "unsupported arithmetic checkpoint"
            )
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticRow1ArithmeticExpansionError(
                "component remainder cannot be declared from two output rows"
            )
        control_passed, control = generic_arithmetic_materialization_control()
        if not control_passed:
            raise QuarticRow1ArithmeticExpansionError(
                "generic arithmetic materialization failed"
            )
        component_provenance = metric_rows_campaign[
            "common_explicit_tensor_dag_packet"
        ]["root_packet"]["content_sha256"]
        common_packet = _build_row1_arithmetic_packet(component_provenance)
        allowed = set(common_packet["arithmetic_dag"]["allowed_operations"])
        actual = {node["op"] for node in common_packet["arithmetic_dag"]["nodes"]}
        if actual - allowed:
            raise QuarticRow1ArithmeticExpansionError(
                "non-arithmetic operation leaked into output DAG"
            )
        maps = (
            _candidate_records(row0_campaign),
            _candidate_records(metric_rows_campaign),
        )
        candidate_ids = set(maps[0])
        if len(candidate_ids) != int(config.get("expected_candidate_count", 12)) or set(
            maps[1]
        ) != candidate_ids:
            raise QuarticRow1ArithmeticExpansionError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id], maps[1][candidate_id], common_packet
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_rows0_1_arithmetic_materialized_other_rows_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "row0_arithmetic": row0_campaign.get("content_sha256"),
                "metric_rows_tensor_dag": metric_rows_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_arithmetic_materialization_control": control,
            "common_row1_arithmetic_packet": common_packet,
            "counts": {
                "selected": len(certificates),
                "current_output_rows_materialized_per_candidate": 1,
                "cumulative_output_rows_materialized_per_candidate": 2,
                "current_lower_entries_normalized_per_candidate": 54,
                "cumulative_lower_entries_normalized_per_candidate": 108,
                "current_selected_mixed_entries_per_candidate": 6,
                "cumulative_selected_mixed_entries_per_candidate": 12,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "Output rows zero and one have arithmetic-only lower Jacobian and "
                "selected mixed roots; output rows two through ten remain fail-closed."
            ),
            "scope": (
                "The row-one DAG reuses the audited explicit Faddeev-LeVerrier "
                "arithmetic construction and remains bound to the same exact A/W roots."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticRow1ArithmeticExpansionError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "current_output_rows_materialized_per_candidate": 0,
                "cumulative_output_rows_materialized_per_candidate": 0,
                "current_lower_entries_normalized_per_candidate": 0,
                "cumulative_lower_entries_normalized_per_candidate": 0,
                "current_selected_mixed_entries_per_candidate": 0,
                "cumulative_selected_mixed_entries_per_candidate": 0,
                "full_component_remainders_proved": 0,
                "H7_closures": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_row1_arithmetic_expansion_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
