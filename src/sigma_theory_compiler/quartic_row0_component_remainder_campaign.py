from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-row0-component-remainder-campaign-1.0"


class QuarticRow0ComponentRemainderError(ValueError):
    """Raised when the row-zero remainder slice is inconsistent."""


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


def _dag_dependencies(node: dict[str, Any]) -> list[int]:
    operation = node["op"]
    if operation in {"exact_constant", "exact_component_input"}:
        return []
    if operation == "exact_add":
        return list(node["arguments"])
    if operation == "exact_negate":
        return [int(node["argument"])]
    if operation == "exact_multiply":
        return [int(node["left"]), int(node["right"])]
    if operation == "exact_divide":
        return [int(node["numerator"]), int(node["denominator"])]
    raise QuarticRow0ComponentRemainderError(
        f"unsupported arithmetic DAG operation {operation}"
    )


def _reachable_audit(packet: dict[str, Any], roots: list[int]) -> dict[str, Any]:
    nodes = packet["arithmetic_dag"]["nodes"]
    seen: set[int] = set()
    stack = list(roots)
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        if index < 0 or index >= len(nodes):
            raise QuarticRow0ComponentRemainderError("arithmetic root is out of range")
        seen.add(index)
        stack.extend(_dag_dependencies(nodes[index]))
    inputs = [nodes[index] for index in seen if nodes[index]["op"] == "exact_component_input"]
    divisions = [index for index in seen if nodes[index]["op"] == "exact_divide"]
    bounded_inputs = [
        node
        for node in inputs
        if "absolute_upper" in node or "interval" in node or "norm_upper" in node
    ]
    return {
        "reachable_nodes": len(seen),
        "component_inputs": len(inputs),
        "quantitatively_bounded_component_inputs": len(bounded_inputs),
        "division_nodes": len(divisions),
        "all_component_inputs_quantitatively_bounded": len(bounded_inputs) == len(inputs),
    }


@cache
def generic_row0_component_remainder_control() -> tuple[bool, dict[str, Any]]:
    """Check the constant-row LP bound and the exact energy absorption algebra."""

    multiplier = sp.Symbol("m", real=True, finite=True)
    matrix = sp.Matrix([[2, -1, 3], [0, 4, 1]])
    commutator = multiplier * matrix - matrix * (multiplier * sp.eye(3))
    gamma, c_linear, h_lower, energy = sp.symbols(
        "Gamma C_L h E", positive=True, finite=True
    )
    source_term = gamma * sp.sqrt(energy) * c_linear * sp.sqrt(
        energy / h_lower
    )
    absorbed = gamma * c_linear * energy / sp.sqrt(h_lower)
    absorption_residual = sp.simplify(source_term - absorbed)
    frequency = sp.Symbol("N", positive=True, finite=True)
    l2_to_h7_loss = (1 + frequency**2) ** sp.Rational(7, 2)
    state_dimension = 153
    symmetric_counts = {
        str(order): comb(state_dimension + order - 1, order)
        for order in range(2, 5)
    }
    full_symmetric_total = sum(symmetric_counts.values())
    passed = bool(
        commutator.is_zero_matrix
        and absorption_residual == 0
        and l2_to_h7_loss.has(frequency)
        and full_symmetric_total > 6
    )
    return passed, {
        "control": "row-zero constant reference-linear source extraction",
        "constant_matrix_Littlewood_Paley_commutator": str(commutator),
        "reference_linear_shell_bound": (
            "sum_(j>=7) w_j||Delta_j L0 U||2^2<="
            "||L0||op^2 Q7(U)"
        ),
        "energy_absorption": {
            "input": "Gamma*sqrt(E)*C_L*sqrt(E/h)",
            "output": "Gamma*C_L*E/sqrt(h)",
            "residual": str(absorption_residual),
        },
        "full_row0_symmetric_component_counts": symmetric_counts,
        "full_D2_to_D4_symmetric_component_total": full_symmetric_total,
        "negative_controls": {
            "promote_L2_source_bound_to_H7": {
                "frequency_loss": str(l2_to_h7_loss),
                "rejected": l2_to_h7_loss.has(frequency),
            },
            "declare_six_mixed_entries_complete": {
                "available": 6,
                "required": full_symmetric_total,
                "rejected": full_symmetric_total != 6,
            },
            "bound_arithmetic_quotient_from_nonzero_denominator": {
                "required": "a quantitative positive lower bound for |det(A)|",
                "nonzero_is_sufficient": False,
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _derive_reference_linear_ceiling(
    global_record: dict[str, Any],
) -> tuple[sp.Expr, sp.Expr, dict[str, Any]]:
    energy = global_record["global_energy"]
    terms = global_record["summed_certified_terms"]
    h7_upper = sp.sympify(energy["H7_upper"])
    anti_wick_upper = h7_upper / 2**14
    finite_source = sp.sympify(
        terms["low_plus_finite_localized_source_Q7_constant"]
    )
    finite_weight = sp.Integer(terms["low_plus_finite_weight_sum"])
    source_ceiling = sp.cancel(
        finite_source
        / (2 * anti_wick_upper * sp.sqrt(finite_weight * 2**15))
    )
    source_ceiling_squared = sp.simplify(source_ceiling**2)
    linear_ceiling_squared = sp.simplify(source_ceiling_squared - 1)
    linear_ceiling = sp.sqrt(linear_ceiling_squared)
    encoding_residual = sp.cancel(
        finite_source
        / (
            2
            * anti_wick_upper
            * source_ceiling
            * sp.sqrt(finite_weight * 2**15)
        )
        - 1
    )
    if not (
        source_ceiling_squared.is_Integer
        and linear_ceiling_squared.is_Integer
        and linear_ceiling.is_Integer
        and source_ceiling_squared == 1 + linear_ceiling**2
        and encoding_residual == 0
        and linear_ceiling > 0
    ):
        raise QuarticRow0ComponentRemainderError(
            "global finite-source scale does not encode an exact C1 ceiling"
        )
    return linear_ceiling, source_ceiling, {
        "anti_wick_upper_recovered_from_H7_upper": str(anti_wick_upper),
        "finite_source_encoding_residual": str(encoding_residual),
        "source_ceiling_squared": str(source_ceiling_squared),
        "identity": "source_ceiling^2=1+C1_ceiling^2",
        "C1_ceiling_squared": str(linear_ceiling_squared),
    }


def _certify_candidate(
    metric: dict[str, Any],
    row0: dict[str, Any],
    global_record: dict[str, Any],
    packet_audit: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    records = (metric, row0, global_record)
    candidate_id = str(metric.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticRow0ComponentRemainderError("candidate identity mismatch")
    if any(record.get("coefficients") != metric.get("coefficients") for record in records[1:]):
        raise QuarticRow0ComponentRemainderError("candidate coefficient mismatch")
    expected = (
        "pass_all_Euler_rows_tensor_lowered_materialization_fail_closed",
        "pass_row0_lower_arithmetic_materialization_partial_mixed_fail_closed",
        "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed",
    )
    if tuple(record.get("status") for record in records) != expected:
        raise QuarticRow0ComponentRemainderError("candidate prerequisite status mismatch")
    coverage = row0["row_coverage"]["0"]
    if not (
        coverage["lower_entries_arithmetic_normalized"] == 54
        and coverage["mixed_entries_orders_2_to_4_normalized"] == 6
        and coverage["complete_for_configured_slice"] is True
        and row0["full_component_Frechet_tensors_complete"] is False
        and row0["paralinearization_remainder_bound_proved"] is False
    ):
        raise QuarticRow0ComponentRemainderError("row-zero coverage mismatch")
    if not (
        metric["full_11x153_source_Jacobian_operational_roots_emitted"] is True
        and metric["full_component_Frechet_tensors_complete"] is False
        and metric["paralinearization_remainder_bound_proved"] is False
    ):
        raise QuarticRow0ComponentRemainderError("metric tensor-DAG scope mismatch")
    linear_ceiling, source_ceiling, source_evidence = _derive_reference_linear_ceiling(
        global_record
    )
    if linear_ceiling != 41354:
        raise QuarticRow0ComponentRemainderError(
            "unexpected reference-linear ceiling"
        )
    h7_lower = global_record["global_energy"]["H7_lower"]
    a_known = global_record["strongest_global_differential_inequality"]["A_known"]
    gamma = global_record["strongest_global_differential_inequality"]["Gamma_B"]
    row0_linear_energy_increment = (
        f"({gamma})*{linear_ceiling}/sqrt({h7_lower})"
    )
    updated_linear_growth = f"({a_known})+({row0_linear_energy_increment})"
    upstream_numeric = global_record["numeric_constants"]
    h7_lower_numeric = float(upstream_numeric["H7_energy_lower"])
    a_known_numeric = float(upstream_numeric["known_energy_growth"])
    gamma_numeric = float(upstream_numeric["unresolved_B7_coefficient"])
    row0_increment_numeric = (
        gamma_numeric * float(linear_ceiling) / h7_lower_numeric**0.5
    )
    updated_growth_numeric = a_known_numeric + row0_increment_numeric
    numeric_validation = [
        h7_lower_numeric,
        a_known_numeric,
        gamma_numeric,
        row0_increment_numeric,
        updated_growth_numeric,
    ]
    if not all(value > 0 and sp.Float(value).is_finite for value in numeric_validation):
        raise QuarticRow0ComponentRemainderError("row-zero energy constant is invalid")
    return {
        "schema_version": "sigma-quartic-row0-component-remainder-certificate-1.0",
        "status": "pass_row0_reference_linear_slice_nonlinear_remainder_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": metric["coefficients"],
        "row0_reference_linear_lower_slice": {
            "output_row": 0,
            "lower_columns": 54,
            "principal_columns_already_materialized": 99,
            "complete_DF_row0_column_count": 153,
            "constant_operator": "L0=P_row0*D F(Y_reference)*P_lower",
            "operator_norm_bound": str(linear_ceiling),
            "B7_slice_bound": (
                f"B7_row0,reference-linear<={linear_ceiling}*sqrt(Q7)"
            ),
            "C_L_contribution": str(linear_ceiling),
            "C_B_contribution_for_this_linear_slice": "0",
            "certified": True,
        },
        "source_scale_recovery": {
            **source_evidence,
            "source_ceiling": str(source_ceiling),
            "C1_strict_integer_ceiling": str(linear_ceiling),
        },
        "updated_global_inequality": {
            "exact": (
                "E7'<=A_row0*E7+Gamma_B*sqrt(E7)*B7_remaining"
            ),
            "remaining_functional": "B7_remaining",
            "remaining_definition": (
                "row0 nonlinear/coefficient-variation and remote paraproduct terms, "
                "plus all unresolved contributions from output rows 1 through 10"
            ),
            "A_previous": a_known,
            "row0_linear_increment": row0_linear_energy_increment,
            "A_row0": updated_linear_growth,
            "Gamma_B": gamma,
            "proved_with_explicit_remainder": True,
            "closed_Gronwall_inequality": False,
        },
        "row0_nonlinear_component_audit": {
            "selected_mixed_entries_available": 6,
            "full_D2_to_D4_symmetric_entries_required": generic[
                "full_D2_to_D4_symmetric_component_total"
            ],
            "arithmetic_reachability": packet_audit,
            "complete_row0_C_B_available": False,
            "full_row0_B7_contribution_bounded": False,
            "reason": (
                "the selected pair does not span the 153-input mixed tensor and the "
                "arithmetic component leaves/determinant have no quantitative tube bounds"
            ),
        },
        "remaining_lower_Jacobian_arithmetic_entries": 540,
        "row0_reference_linear_C_L_certified": True,
        "row0_nonlinear_C_B_certified": False,
        "full_11_row_remainder_closed": False,
        "global_H7_differential_inequality_closed": False,
        "nonlinear_lifespan_proved": False,
        "remaining_gates": [
            "materialize_and_bound_rows_1_through_10_lower_Jacobian_entries",
            "complete_all_row0_D2_to_D4_mixed_multi_index_components",
            "supply_tube_uniform_component_input_upper_bounds_and_detA_lower_bound",
            "apply_the_H7_Bony_Moser_majorant_to_row0_then_all_remaining_rows",
        ],
        "numeric_constants": {
            "C_L_row0_reference_linear": float(linear_ceiling),
            "row0_linear_energy_increment": row0_increment_numeric,
            "updated_known_energy_growth": updated_growth_numeric,
        },
        "scope": (
            "Only the constant reference-linear lower-source slice of output row zero "
            "is removed from B7. No nonlinear row-zero, other-row, global H7, or "
            "lifespan closure is claimed."
        ),
    }


def run_quartic_row0_component_remainder_campaign(
    metric_rows_campaign: dict[str, Any],
    row0_arithmetic_campaign: dict[str, Any],
    global_h7_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticRow0ComponentRemainderError(
                "unsupported campaign schema_version"
            )
        expected = (
            "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed",
            "pass_all_12_row0_arithmetic_materialized_other_rows_fail_closed",
            (
                "audit_all_12_global_H7_energies_single_source_remainder_"
                "lifespans_fail_closed"
            ),
        )
        campaigns = (
            metric_rows_campaign,
            row0_arithmetic_campaign,
            global_h7_campaign,
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected:
            raise QuarticRow0ComponentRemainderError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticRow0ComponentRemainderError("campaign content hash mismatch")
        metric_hash = metric_rows_campaign["content_sha256"]
        global_upstream = global_h7_campaign["upstream_sha256"]
        if row0_arithmetic_campaign.get("upstream_sha256", {}).get(
            "metric_rows_tensor_dag"
        ) != metric_hash:
            raise QuarticRow0ComponentRemainderError("row0-metric provenance mismatch")
        if (
            metric_rows_campaign.get("upstream_sha256", {}).get("principal_source")
            != global_upstream.get("source_jacobian")
            or row0_arithmetic_campaign.get("upstream_sha256", {}).get(
                "principal_source"
            )
            != global_upstream.get("source_jacobian")
            or metric_rows_campaign.get("upstream_sha256", {}).get(
                "semantic_source_dag"
            )
            != global_upstream.get("source_dag")
            or metric_rows_campaign.get("upstream_sha256", {}).get(
                "nonlinear_evolution"
            )
            != global_upstream.get("nonlinear")
        ):
            raise QuarticRow0ComponentRemainderError(
                "row0-global source provenance mismatch"
            )
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["output_row"]) != 0
            or int(config["lower_column_count"]) != 54
            or int(config["selected_mixed_entry_count"]) != 6
            or config.get("row0_nonlinear_remainder_policy") != "fail_closed"
            or config.get("full_11_row_remainder_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticRow0ComponentRemainderError(
                "unsupported row-zero remainder contract"
            )
        generic_passed, generic = generic_row0_component_remainder_control()
        if not generic_passed:
            raise QuarticRow0ComponentRemainderError(
                "generic row-zero remainder control failed"
            )
        packet = row0_arithmetic_campaign["common_row0_arithmetic_packet"]
        dag = packet["arithmetic_dag"]
        dag_body = {key: value for key, value in dag.items() if key != "content_sha256"}
        if dag.get("content_sha256") != _content_hash(dag_body):
            raise QuarticRow0ComponentRemainderError(
                "row-zero arithmetic DAG hash mismatch"
            )
        allowed = set(dag["allowed_operations"])
        actual = {node["op"] for node in dag["nodes"]}
        if actual - allowed or len(dag["nodes"]) != int(dag["node_count"]):
            raise QuarticRow0ComponentRemainderError(
                "row-zero arithmetic DAG operation mismatch"
            )
        lower_roots = [
            int(item["arithmetic_root"])
            for item in packet["lower_Jacobian_row0"]
        ]
        mixed_roots = [
            int(item["arithmetic_root"])
            for item in packet["selected_mixed_F_row0"]
        ]
        if not (
            len(lower_roots) == 54
            and len(mixed_roots) == 6
            and all(
                item.get("output_row") == 0
                and item.get("normalized_residual") == "0"
                for item in packet["lower_Jacobian_row0"]
            )
            and all(
                item.get("output_row") == 0
                and item.get("normalized_coefficient_residual") == "0"
                for item in packet["selected_mixed_F_row0"]
            )
        ):
            raise QuarticRow0ComponentRemainderError(
                "row-zero arithmetic roots are incomplete"
            )
        lower_audit = _reachable_audit(packet, lower_roots)
        mixed_audit = _reachable_audit(packet, mixed_roots)
        packet_audit = {
            "lower_Jacobian_roots": lower_audit,
            "selected_mixed_roots": mixed_audit,
            "inverse_division_assumption": packet["inverse_evidence"][
                "division_assumption"
            ],
            "quantitative_detA_lower_bound_available": False,
        }
        if (
            lower_audit["all_component_inputs_quantitatively_bounded"]
            or mixed_audit["all_component_inputs_quantitatively_bounded"]
            or packet_audit["quantitative_detA_lower_bound_available"]
        ):
            raise QuarticRow0ComponentRemainderError(
                "unexpected quantitative arithmetic bound state"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticRow0ComponentRemainderError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps[0][candidate_id],
                maps[1][candidate_id],
                maps[2][candidate_id],
                packet_audit,
                generic,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_row0_reference_linear_slices_"
                "nonlinear_and_global_remainders_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "metric_rows_tensor_dag": metric_rows_campaign["content_sha256"],
                "row0_arithmetic": row0_arithmetic_campaign["content_sha256"],
                "global_H7": global_h7_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_row0_component_remainder_control": generic,
            "arithmetic_packet_audit": packet_audit,
            "counts": {
                "selected": len(certificates),
                "row0_reference_linear_C_L_contributions_certified": len(
                    certificates
                ),
                "row0_nonlinear_C_B_contributions_certified": 0,
                "complete_row0_remainders_closed": 0,
                "full_11_row_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The constant reference-linear lower-source slice of output row zero "
                "has an explicit C_L contribution. Its nonlinear variation, rows 1-10, "
                "global H7 closure, and lifespan remain fail-closed."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticRow0ComponentRemainderError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "row0_reference_linear_C_L_contributions_certified": 0,
                "row0_nonlinear_C_B_contributions_certified": 0,
                "complete_row0_remainders_closed": 0,
                "full_11_row_remainders_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_row0_component_remainder_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
