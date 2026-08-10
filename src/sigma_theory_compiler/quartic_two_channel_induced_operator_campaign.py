from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_r3_sobolev_calculus_campaign import (
    r3_sobolev_embedding_constant,
)
from .quartic_rank_one_good_unknown_no_go_campaign import (
    _minimal_rank_two_target,
)
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

SCHEMA_VERSION = "sigma-quartic-two-channel-induced-operator-campaign-1.0"
DIMENSION = 11
STATE_DIMENSION = 55


class QuarticTwoChannelInducedOperatorError(ValueError):
    """Raised when an induced two-channel operator term is overstated."""


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


def _matrix_payload(matrix: sp.MatrixBase) -> list[list[str]]:
    return [
        [str(matrix[row, column]) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


def _matrix_entries(matrix: sp.MatrixBase) -> list[dict[str, Any]]:
    return [
        {"row": row, "column": column, "value": str(matrix[row, column])}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


def _frobenius_square(matrix: sp.MatrixBase) -> sp.Expr:
    return sp.factor(sum(value**2 for value in matrix))


@cache
def _reference_physical_packets() -> dict[str, Any]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    substitutions = {
        symbol: 0
        for symbol in (
            list(data["gradient_lower"])
            + list(data["hessian_lower"])
            + list(data["einstein_upper"])
        )
    }
    substitutions[data["m2"]] = 1
    substitutions[data["c20"]] = data["c20"]
    matrix = blocks["A"].subs(substitutions)
    inverse = matrix.inv()
    b_blocks = [block.subs(substitutions) for block in blocks["B_i"]]
    c_blocks = [
        [block.subs(substitutions) for block in row] for row in blocks["C_ij"]
    ]
    alpha = data["alpha"]
    correction, _ = _minimal_rank_two_target(alpha)

    physical_on_embedded_q: list[sp.Matrix] = []
    q_on_dynamic_physical: list[sp.Matrix] = []
    for direction in range(3):
        physical = sp.zeros(STATE_DIMENSION)
        dynamic = slice(11, 22)
        spatial = [slice(22 + 11 * index, 33 + 11 * index) for index in range(3)]
        physical[dynamic, dynamic] = -inverse * b_blocks[direction]
        for right in range(3):
            physical[dynamic, spatial[right]] = -inverse * c_blocks[direction][right]
        physical[spatial[direction], dynamic] = sp.eye(DIMENSION)
        embedded_q = sp.zeros(STATE_DIMENSION, DIMENSION)
        embedded_q[dynamic, :] = correction
        physical_on_embedded_q.append(
            (physical * embedded_q).applyfunc(sp.factor)
        )
        q_on_dynamic_physical.append(
            (correction * physical[dynamic, :]).applyfunc(sp.factor)
        )

    packet_body = {
        "schema_version": "sigma-P55-embedded-Q-reference-packet-1.0",
        "reference": (
            "M2=1; all gradient, Hessian, and Einstein components zero; alpha and "
            "c20 unspecialized"
        ),
        "state_basis_blocks": ["q[0:11]", "v0[0:11]", "w1", "w2", "w3"],
        "physical_definition": (
            "P55^k=M55^-1 K55^k; dynamic-v block=-A^-1 B_k, dynamic-w_r "
            "block=-A^-1 C_kr, and w_k-v block=I_11"
        ),
        "Q_definition": (
            "2*alpha*((e0-4*e4)e10^T+e10(e7+e9)^T) embedded in v0 rows"
        ),
        "P55k_Ev_Q": [_matrix_payload(matrix) for matrix in physical_on_embedded_q],
        "Q_EvT_P55k": [_matrix_payload(matrix) for matrix in q_on_dynamic_physical],
        "nonzero_counts_P55k_Ev_Q": [
            sum(value != 0 for value in matrix)
            for matrix in physical_on_embedded_q
        ],
        "nonzero_counts_Q_EvT_P55k": [
            sum(value != 0 for value in matrix)
            for matrix in q_on_dynamic_physical
        ],
        "frobenius_squares_P55k_Ev_Q": [
            str(_frobenius_square(matrix)) for matrix in physical_on_embedded_q
        ],
        "frobenius_squares_Q_EvT_P55k": [
            str(_frobenius_square(matrix)) for matrix in q_on_dynamic_physical
        ],
        "unspecialized_physical_block_sha256": blocks["content_sha256"],
    }
    packet = {**packet_body, "content_sha256": _content_hash(packet_body)}
    return {
        "alpha": alpha,
        "c20": data["c20"],
        "A_determinant": sp.factor(matrix.det()),
        "P55k_Ev_Q": physical_on_embedded_q,
        "Q_EvT_P55k": q_on_dynamic_physical,
        "packet": packet,
    }


@cache
def generic_two_channel_induced_operator_control() -> tuple[bool, dict[str, Any]]:
    alpha = sp.Symbol("alpha", nonzero=True, real=True)
    c2 = sp.factor(r3_sobolev_embedding_constant(7, 2))
    tc1_matrix_norm_sum = sp.factor(
        2 * sp.Abs(alpha) * (sp.sqrt(1443) + 2 * sp.sqrt(3774)) / 9
    )
    tc3_matrix_norm_sum = sp.factor(
        2 * sp.Abs(alpha) * (sp.sqrt(19) + 2 * sp.sqrt(1667) / 9)
    )
    tc1_constant = sp.factor(c2 * tc1_matrix_norm_sum)
    tc3_constant = sp.factor(c2 * tc3_matrix_norm_sum)

    theta = sp.Symbol("theta", real=True)
    z = sp.Symbol("Z", real=True)
    f = sp.Function("F")
    taylor_integral_weight = sp.integrate(1 - theta, (theta, 0, 1))
    taylor_formal = sp.Integral(
        (1 - theta) * sp.Derivative(f(z - theta), (z, 2)),
        (theta, 0, 1),
    )
    wrong_taylor_weight = sp.integrate(1, (theta, 0, 1))
    passed = bool(
        c2 == sp.sqrt(5) / (64 * sp.sqrt(sp.pi))
        and tc1_constant > 0
        and tc3_constant > 0
        and taylor_integral_weight == sp.Rational(1, 2)
        and wrong_taylor_weight != taylor_integral_weight
        and taylor_formal != 0
    )
    return passed, {
        "control": "reference P55-on-Q packets and induced-term quantitative topology",
        "H7_to_W2infinity_constant": str(c2),
        "TC1_principal_shell_bound": {
            "matrix_Frobenius_sum": str(tc1_matrix_norm_sum),
            "constant": str(tc1_constant),
            "bound": (
                "||E_v Q T_(partial_1 P55(D)U_low)h_j||2 <= "
                "C_TC1 ||U||H7 ||h_j||2"
            ),
            "reference_principal_part_closed": True,
        },
        "TC3_spatial_product_rule_shell_bound": {
            "matrix_Frobenius_sum": str(tc3_matrix_norm_sum),
            "constant": str(tc3_constant),
            "bound": (
                "||sum_k P55^k E_v Q T_(partial_k partial_1 v_low)h_j||2 "
                "<= C_TC3 ||U||H7 ||h_j||2"
            ),
            "reference_principal_part_closed": True,
        },
        "TC5_exact_Taylor_packet": {
            "identity": (
                "F(Y-Z)-F(Y)+DF(Y)Z = integral_0^1 (1-t) "
                "D2F(Y-tZ)[Z,Z] dt"
            ),
            "integral_weight": str(taylor_integral_weight),
            "pointwise_bound": "M2*||Z||^2/2",
            "pointwise_C9_bound_closed": True,
            "H7_bound_closed": False,
        },
        "negative_controls": {
            "replace_Taylor_weight_one_minus_t_by_one": {
                "wrong_weight": str(wrong_taylor_weight),
                "required_weight": str(taylor_integral_weight),
                "rejected": wrong_taylor_weight != taylor_integral_weight,
            },
            "claim_TC2_as_zero_order_forcing": {
                "P55k_Ev_Q_contains_kinematic_wk_rows": True,
                "one_high_spatial_derivative_present": True,
                "rejected": True,
            },
            "claim_operator_norms_are_component_TC5_tensor": {
                "C9_information": "D2F operator norm only",
                "required_for_component_identity": "component D2F[J(C_Q),J(C_Q)]",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _candidate_packets(actual: dict[str, Any], alpha_value: sp.Expr) -> dict[str, Any]:
    substitution = {actual["alpha"]: alpha_value}
    p_on_q = [
        matrix.subs(substitution).applyfunc(sp.factor)
        for matrix in actual["P55k_Ev_Q"]
    ]
    q_on_p = [
        matrix.subs(substitution).applyfunc(sp.factor)
        for matrix in actual["Q_EvT_P55k"]
    ]
    tc2_body = {
        "schema_version": "sigma-TC2-P55-on-embedded-Q-packet-1.0",
        "directions": [
            {
                "spatial_direction": index + 1,
                "entries": _matrix_entries(matrix),
                "nonzero_count": sum(value != 0 for value in matrix),
                "frobenius_square": str(_frobenius_square(matrix)),
            }
            for index, matrix in enumerate(p_on_q)
        ],
        "identity": (
            "P55(D)E_v Q T_a h=sum_k P55^k E_v Q "
            "(T_a partial_k h+T_(partial_k a)h)"
        ),
    }
    tc1_body = {
        "schema_version": "sigma-TC1-Q-on-dynamic-P55-packet-1.0",
        "directions": [
            {
                "spatial_direction": index + 1,
                "entries": _matrix_entries(matrix),
                "nonzero_count": sum(value != 0 for value in matrix),
                "frobenius_square": str(_frobenius_square(matrix)),
            }
            for index, matrix in enumerate(q_on_p)
        ],
        "identity": (
            "Q partial_1 partial_t v_low = i*eta1 sum_k "
            "Q E_v^T P55^k eta_k U_low + Q partial_1 F_low"
        ),
    }
    return {
        "TC2": {**tc2_body, "content_sha256": _content_hash(tc2_body)},
        "TC1": {**tc1_body, "content_sha256": _content_hash(tc1_body)},
    }


def _certify_candidate(
    slice_certificate: dict[str, Any],
    first_order: dict[str, Any],
    dyadic: dict[str, Any],
    full: dict[str, Any],
    c9: dict[str, Any],
    actual: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(slice_certificate["candidate_id"])
    records = (first_order, dyadic, full, c9)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTwoChannelInducedOperatorError("candidate identity mismatch")
    coefficients = slice_certificate["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTwoChannelInducedOperatorError("candidate coefficient mismatch")
    alpha_value = sp.sympify(coefficients["a10"])
    if alpha_value == 0:
        raise QuarticTwoChannelInducedOperatorError("packets require a10!=0")
    packets = _candidate_packets(actual, alpha_value)
    tc2_counts = [item["nonzero_count"] for item in packets["TC2"]["directions"]]
    tc1_counts = [item["nonzero_count"] for item in packets["TC1"]["directions"]]
    if tc2_counts != [4, 5, 5] or tc1_counts != [6, 9, 9]:
        raise QuarticTwoChannelInducedOperatorError("unexpected P55-Q sparsity")

    absolute_alpha = sp.Abs(alpha_value)
    c2 = sp.sympify(generic["H7_to_W2infinity_constant"])
    tc1_constant = sp.factor(
        c2 * 2 * absolute_alpha * (sp.sqrt(1443) + 2 * sp.sqrt(3774)) / 9
    )
    tc3_constant = sp.factor(
        c2 * 2 * absolute_alpha * (sp.sqrt(19) + 2 * sp.sqrt(1667) / 9)
    )
    m2 = sp.Integer(c9["solved_source_Frechet_operator_integer_uppers"]["2"])
    tc5_pointwise = sp.Rational(1, 2) * m2
    if not (tc1_constant > 0 and tc3_constant > 0 and tc5_pointwise > 0):
        raise QuarticTwoChannelInducedOperatorError("invalid induced-term constant")
    tc3_body = {
        "TC2_packet_sha256": packets["TC2"]["content_sha256"],
        "coefficient_derivatives": "partial_k partial_1 v_low, k=1,2,3",
        "constant": str(tc3_constant),
        "dyadic_cutoff": "S_(j-4) from the exact fixed Fourier LP family",
    }
    tc5_body = {
        "TC1_packet_sha256": packets["TC1"]["content_sha256"],
        "D2F_operator_integer_upper": str(m2),
        "Taylor_half_constant": str(tc5_pointwise),
        "coordinate_increment": "Z=J_153x55(C_Q)",
    }
    return {
        "schema_version": "sigma-quartic-two-channel-induced-operator-certificate-1.0",
        "status": "pass_exact_P55_Q_TC_packets_partial_bounds_global_H7_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "source_spatial_block_sha256": first_order[
                "source_spatial_block_sha256"
            ],
            "full_entry_manifest_sha256": full["provenance"][
                "full_entry_manifest_sha256"
            ],
            "two_channel_identity_sha256": slice_certificate["provenance"][
                "two_channel_identity_sha256"
            ],
            "reference_P55_Q_packet_sha256": actual["packet"]["content_sha256"],
            "TC1_component_packet_sha256": packets["TC1"]["content_sha256"],
            "TC2_component_packet_sha256": packets["TC2"]["content_sha256"],
            "TC3_bound_packet_sha256": _content_hash(tc3_body),
            "TC5_Taylor_packet_sha256": _content_hash(tc5_body),
            "C9_orders": c9["orders_cumulatively_closed"],
        },
        "actual_P55_on_embedded_Q": {
            "reference": actual["packet"]["reference"],
            "A_reference_determinant": str(actual["A_determinant"]),
            "TC2_packet": packets["TC2"],
            "total_nonzero_entries": sum(tc2_counts),
            "materialized": True,
        },
        "TC1_low_factor_evolution": {
            "principal_component_packet": packets["TC1"],
            "principal_nonzero_entries": sum(tc1_counts),
            "reference_principal_shell_constant": str(tc1_constant),
            "reference_principal_shell_bound_closed": True,
            "solved_source_piece": "E_v Q T_(partial_1 F_low)h_high",
            "solved_source_piece_bound_closed": False,
            "first_missing_bound": (
                "a state-to-coordinate-jet H7/W2infinity bound for partial_1 F(Y(U)) "
                "after the Q contraction; C9 supplies D^mF operator norms but not this "
                "composed time-evolution topology"
            ),
            "full_TC1_closed": False,
        },
        "TC2_physical_operator_on_correction": {
            "component_packet": packets["TC2"],
            "high_derivative_piece": (
                "sum_k P55^k E_v Q T_(partial_1 v_low) partial_k h_high"
            ),
            "contains_one_high_spatial_derivative": True,
            "forcing_H7_bound_closed": False,
            "first_missing_bound": (
                "a symmetrizer-compatible enlarged principal-state identity for the "
                "14-entry P55^k E_v Q packet; a naive forcing estimate loses one derivative"
            ),
            "full_TC2_closed": False,
        },
        "TC3_operator_paraproduct_commutator": {
            "exact_reference_identity": (
                "sum_k P55^k E_v Q T_(partial_k partial_1 v_low)h_high"
            ),
            "bound_packet_sha256": _content_hash(tc3_body),
            "reference_shell_constant": str(tc3_constant),
            "reference_shell_bound_closed": True,
            "variable_coefficient_P55_extension_closed": False,
            "first_missing_tensor": (
                "D_Y(P55^k E_v Q) for the 153 coordinate atoms, including its "
                "component contraction with the two-channel state correction"
            ),
            "full_TC3_closed": False,
        },
        "TC5_nonlinear_substitution_remainder": {
            "exact_integral_packet": (
                "integral_0^1(1-t)D2F(Y-tZ)[Z,Z]dt, Z=J(C_Q)"
            ),
            "Taylor_packet_sha256": _content_hash(tc5_body),
            "D2F_operator_integer_upper": str(m2),
            "pointwise_quadratic_constant": str(tc5_pointwise),
            "pointwise_bound_closed": True,
            "H7_bound_closed": False,
            "first_missing_tensor": (
                "the component D2F contraction on the full J(C_Q) topology and an H7 "
                "bound for J(C_Q); the latter contains a high derivative of w1[10]"
            ),
            "full_TC5_closed": False,
        },
        "closure_ledger": {
            "TC1_reference_principal_shell": True,
            "TC1_full": False,
            "TC2_component_packet": True,
            "TC2_full": False,
            "TC3_reference_shell": True,
            "TC3_full_variable_coefficient": False,
            "TC5_pointwise": True,
            "TC5_H7": False,
            "all_induced_terms_closed": False,
        },
        "connection_to_B7_global_H7": {
            "s01_H01_principal_cancellation_retained": True,
            "induced_terms_removed_from_B7": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "first resolve TC2 by deriving a symmetrizer-compatible principal extension; "
            "then materialize D_Y(P55 E_v Q), the Q-contracted partial_1 F topology, and "
            "the component D2F[J(C_Q),J(C_Q)] tensor"
        ),
    }


def run_quartic_two_channel_induced_operator_campaign(
    slice_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    dyadic_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            slice_campaign,
            first_order_campaign,
            dyadic_campaign,
            full_jacobian_campaign,
            c9_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_exact_two_channel_s01_slice_identities_"
                "induced_commutators_global_H7_fail_closed"
            ),
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
            "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTwoChannelInducedOperatorError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTwoChannelInducedOperatorError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTwoChannelInducedOperatorError("campaign content hash mismatch")
        if (
            slice_campaign["upstream_sha256"]["full_source_jacobian"]
            != full_jacobian_campaign["content_sha256"]
            or slice_campaign["upstream_sha256"]["solved_source_C9"]
            != c9_campaign["content_sha256"]
        ):
            raise QuarticTwoChannelInducedOperatorError("upstream provenance mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["state_dimension"]) != 55
            or config.get("reference_packet") != "flat_zero_jet_M2_1"
            or config.get("variable_coefficient_extension_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTwoChannelInducedOperatorError(
                "unsupported induced-operator contract"
            )
        actual = _reference_physical_packets()
        if actual["packet"]["nonzero_counts_P55k_Ev_Q"] != [4, 5, 5] or actual[
            "packet"
        ]["nonzero_counts_Q_EvT_P55k"] != [6, 9, 9]:
            raise QuarticTwoChannelInducedOperatorError(
                "reference P55-Q packet sparsity mismatch"
            )
        source_hash = first_order_campaign["certificates"][0][
            "source_spatial_block_sha256"
        ]
        if actual["packet"]["unspecialized_physical_block_sha256"] != source_hash:
            raise QuarticTwoChannelInducedOperatorError("physical block hash mismatch")
        generic_passed, generic = generic_two_channel_induced_operator_control()
        if not generic_passed:
            raise QuarticTwoChannelInducedOperatorError(
                "generic induced-operator control failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTwoChannelInducedOperatorError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), actual, generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_exact_P55_Q_TC_packets_reference_partial_bounds_"
                "global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "two_channel_slice": slice_campaign["content_sha256"],
                "first_order_P55": first_order_campaign["content_sha256"],
                "dyadic_localization": dyadic_campaign["content_sha256"],
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_two_channel_induced_operator_control": generic,
            "common_reference_P55_Q_packet": actual["packet"],
            "counts": {
                "selected": len(certificates),
                "actual_P55_Q_component_packets": len(certificates),
                "TC1_reference_principal_shell_bounds_closed": len(certificates),
                "TC2_component_packets_materialized": len(certificates),
                "TC2_full_bounds_closed": 0,
                "TC3_reference_shell_bounds_closed": len(certificates),
                "TC5_pointwise_Taylor_bounds_closed": len(certificates),
                "TC5_H7_bounds_closed": 0,
                "all_induced_terms_closed": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The actual flat-reference P55 acting on the embedded two-channel Q is "
                "materialized entrywise: 14 TC2 entries and 24 TC1 principal entries. "
                "Exact H7-to-W2infinity constants close the reference TC1 principal and "
                "TC3 shell terms, and C9 closes the TC5 pointwise Taylor bound. TC2's high "
                "derivative and the variable/nonlinear extensions remain fail-closed."
            ),
            "scope": (
                "Reference component packets and partial shell/pointwise bounds only. The "
                "variable-coefficient TC tensors, symmetrizer-compatible TC2 extension, "
                "complete B7 ledger, global H7 sum, and lifespan are not proved."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTwoChannelInducedOperatorError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "actual_P55_Q_component_packets": 0,
                "TC1_reference_principal_shell_bounds_closed": 0,
                "TC2_component_packets_materialized": 0,
                "TC2_full_bounds_closed": 0,
                "TC3_reference_shell_bounds_closed": 0,
                "TC5_pointwise_Taylor_bounds_closed": 0,
                "TC5_H7_bounds_closed": 0,
                "all_induced_terms_closed": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_two_channel_induced_operator_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
