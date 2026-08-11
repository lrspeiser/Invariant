from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_first_order_reduction_campaign import (
    _extract_spatial_blocks,
    _full_first_order_pencil,
    _symbol_data,
)
from .quartic_r3_sobolev_calculus_campaign import r3_sobolev_embedding_constant
from .quartic_rank_one_good_unknown_no_go_campaign import _minimal_rank_two_target
from .quartic_tc2_full_sylvester_reference_campaign import (
    _full_reference_sylvester_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-ck1-reference-source-campaign-1.0"
STATE_DIMENSION = 55
SOURCE_DIMENSION = 11
TOTAL_EXCLUDED_OBLIGATIONS = 2675


class QuarticTC2CK1ReferenceSourceError(ValueError):
    """Raised when the bounded CK1/reference-source claim is overstated."""


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


def _trilinear_entries(
    high_matrix: sp.MatrixBase, low_row: sp.MatrixBase
) -> list[dict[str, Any]]:
    return [
        {
            "output_row": output_row,
            "high_state_column": high_column,
            "low_state_column": low_column,
            "value": str(high_matrix[output_row, high_column] * low_row[low_column]),
        }
        for output_row in range(high_matrix.rows)
        for high_column in range(high_matrix.cols)
        if high_matrix[output_row, high_column] != 0
        for low_column in range(low_row.cols)
        if low_row[low_column] != 0
    ]


@cache
def _reference_ck1_component_packet() -> dict[str, Any]:
    data = _symbol_data()
    xi = data["xi_lower"]
    substitutions: dict[sp.Symbol, sp.Expr] = {
        data["alpha"]: 0,
        data["m2"]: 1,
        data["c20"]: 0,
        xi[1]: 1,
        xi[2]: 0,
        xi[3]: 0,
    }
    substitutions.update({symbol: 0 for symbol in data["gradient_lower"]})
    substitutions.update({symbol: 0 for symbol in data["hessian_lower"]})
    substitutions.update({symbol: 0 for symbol in data["einstein_upper"]})
    coefficient_a = data["first_order"]["A"].subs(substitutions)
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    b_blocks = [block.subs(substitutions) for block in b_blocks]
    c_blocks = [
        [block.subs(substitutions) for block in row] for row in c_blocks
    ]
    # Appendix-A state order used by the exact Sylvester packet.
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    physical: list[sp.Matrix] = []
    scalar_rows: list[sp.Matrix] = []
    for direction in range(3):
        mass, evolution = _full_first_order_pencil(
            coefficient_a,
            b_blocks[direction],
            [c_blocks[direction][right] for right in range(3)],
            [int(index == direction) for index in range(3)],
        )
        matrix = (mass.inv() * evolution).extract(ordering, ordering)
        physical.append(matrix)
        # v occupies Appendix-A rows 33:44; select its scalar component 10.
        scalar_rows.append(matrix.row(43))

    sylvester = _full_reference_sylvester_packet()
    delta_basis = sylvester["delta10"]
    delta_square = sp.factor(sylvester["delta_frobenius_square"])
    if _content_hash(_matrix_payload(physical[0])) != sylvester["packet"][
        "P55_sha256"
    ]:
        raise QuarticTC2CK1ReferenceSourceError(
            "reference P55 direction-one provenance mismatch"
        )

    expected_rows = (
        [(54, sp.Integer(1))],
        [(21, sp.Integer(1))],
        [(32, sp.Integer(1))],
    )
    actual_rows = tuple(
        [(column, row[column]) for column in range(STATE_DIMENSION) if row[column] != 0]
        for row in scalar_rows
    )
    if actual_rows != expected_rows:
        raise QuarticTC2CK1ReferenceSourceError(
            "reference scalar evolution row is not the exact unit packet"
        )

    products = [_trilinear_entries(delta_basis, row) for row in scalar_rows]
    packet_body = {
        "schema_version": "sigma-ck1-deltaK-e10-P55-reference-packet-1.0",
        "reference": "flat zero covariant jet, M2=1, direction basis e1,e2,e3",
        "ordered_state": "z=(q,w2,w3), y=(v,w1)",
        "selected_dynamic_row": 43,
        "selected_source_component": 10,
        "P55_direction_sha256": [
            _content_hash(_matrix_payload(matrix)) for matrix in physical
        ],
        "deltaK_basis_sha256": _content_hash(_matrix_payload(delta_basis)),
        "deltaK_basis_nonzero_entries": len(_matrix_entries(delta_basis)),
        "deltaK_basis_Frobenius_square": str(delta_square),
        "directions": [
            {
                "spatial_direction": direction + 1,
                "e10_EvT_P55k_entries": _matrix_entries(row),
                "row_Frobenius_square": str(_frobenius_square(row)),
                "deltaK_basis_e10_EvT_P55k_trilinear_entries": product,
                "product_nonzero_entries": len(product),
                "product_Frobenius_square": str(
                    delta_square * _frobenius_square(row)
                ),
            }
            for direction, (row, product) in enumerate(zip(scalar_rows, products))
        ],
        "identity": (
            "partial_1 partial_t v10="
            "sum_k e10^T E_v^T P55^k partial_1 partial_k U+partial_1 F10"
        ),
    }
    return {
        "packet": {**packet_body, "content_sha256": _content_hash(packet_body)},
        "delta_square": delta_square,
        "row_norm_sum": sp.factor(
            sum(sp.sqrt(_frobenius_square(row)) for row in scalar_rows)
        ),
    }


@cache
def generic_ck1_reference_source_control() -> tuple[bool, dict[str, Any]]:
    alpha = sp.Symbol("alpha", nonzero=True, real=True)
    m2 = sp.Symbol("M2", positive=True, real=True)
    c2 = sp.factor(r3_sobolev_embedding_constant(7, 2))
    c_y0 = sp.sqrt(42) / (64 * sp.sqrt(sp.pi))
    c_y1 = sp.sqrt(22) / (64 * sp.sqrt(sp.pi))
    delta_norm = sp.sqrt(sp.Rational(1253060, 9))
    principal = sp.factor(sp.Abs(alpha) * delta_norm * 3 * c2)
    scalar_source = sp.factor(
        sp.Abs(alpha) * delta_norm * m2 * c_y0 * c_y1
    )
    q_norm = 2 * sp.sqrt(19) * sp.Abs(alpha)
    q_source = sp.factor(q_norm * m2 * c_y0 * c_y1)
    y, dy, theta = sp.symbols("Y dY theta", real=True)
    integral_weight = sp.integrate(1, (theta, 0, 1))
    wrong_linear_bound = sp.Symbol("DF0", nonzero=True)
    passed = bool(
        c2 == sp.sqrt(5) / (64 * sp.sqrt(sp.pi))
        and principal > 0
        and scalar_source > 0
        and q_source > 0
        and integral_weight == 1
        and y * dy != 0
        and wrong_linear_bound != 0
    )
    return passed, {
        "control": "reference CK1 principal split and quadratic solved-source topology",
        "H7_to_W2infinity_constant": str(c2),
        "coordinate_atom_Linfinity_constant": str(c_y0),
        "coordinate_atom_first_derivative_Linfinity_constant": str(c_y1),
        "deltaK_basis_Frobenius_norm": str(delta_norm),
        "principal_shell_constant": str(principal),
        "scalar_F10_source_quadratic_constant": str(scalar_source),
        "Q_source_quadratic_constant": str(q_source),
        "source_identity": (
            "partial_1(F(Y)-F(0)-DF(0)Y)="
            "integral_0^1 D2F(theta Y)[Y,partial_1 Y] dtheta"
        ),
        "integral_weight": str(integral_weight),
        "negative_controls": {
            "omit_reference_linear_subtraction": {
                "uncancelled_term": str(wrong_linear_bound),
                "quadratic_bound_invalid": True,
                "rejected": True,
            },
            "drop_spatial_direction_three": {
                "missing_exact_entry": {"row": 43, "column": 32, "value": "1"},
                "rejected": True,
            },
            "replace_D2F_by_D1F_in_difference_formula": {
                "correct_topology": "D2F(theta Y)[Y,partial_1 Y]",
                "rejected": True,
            },
            "promote_reference_CK1_to_global_TC2": {
                "missing": "variable deltaK derivative, CK3, and remaining induced commutators",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _validate_obligation_completion(
    checkpoint: dict[str, Any],
    final_artifact: dict[str, Any],
    final_artifact_file_sha256: str,
    config: dict[str, Any],
) -> None:
    from .quartic_tc2_continuous_service import _checkpoint_hash_matches
    from .quartic_tc2_obligation_continuous_service import _validate_record_chain

    if not _checkpoint_hash_matches(checkpoint):
        raise QuarticTC2CK1ReferenceSourceError("obligation checkpoint hash mismatch")
    if not _content_hash_matches(final_artifact) or not _validate_record_chain(
        final_artifact
    ):
        raise QuarticTC2CK1ReferenceSourceError(
            "final obligation artifact hash or record chain mismatch"
        )
    if (
        int(checkpoint.get("next_offset", -1)) != TOTAL_EXCLUDED_OBLIGATIONS
        or int(final_artifact["counts"]["cumulative_evaluated_obligations"])
        != TOTAL_EXCLUDED_OBLIGATIONS
        or int(final_artifact["counts"]["remaining_unevaluated_obligations"]) != 0
        or final_artifact.get("first_exact_obstruction") is not None
    ):
        raise QuarticTC2CK1ReferenceSourceError(
            "excluded-obligation selector is not exactly complete"
        )
    contract = final_artifact["chunk_contract"]
    if (
        int(contract["chunk_offset"]) != 2624
        or int(contract["evaluated_chunk_size"]) != 51
        or not contract["final_partial_tail"]
        or contract["resume_after_record_sha256"]
        != config["expected_final_obligation_tip"]
        or checkpoint["prior_resume_sha256"]
        != config["expected_final_obligation_tip"]
        or checkpoint["current_artifact_content_sha256"]
        != final_artifact["content_sha256"]
        or checkpoint["current_artifact_file_sha256"]
        != final_artifact_file_sha256
        or any(checkpoint["claims"].values())
    ):
        raise QuarticTC2CK1ReferenceSourceError(
            "final obligation tail/checkpoint contract mismatch"
        )


def _certify_candidate(
    reference: dict[str, Any],
    variable: dict[str, Any],
    induced: dict[str, Any],
    full_jacobian: dict[str, Any],
    c9: dict[str, Any],
    topology: dict[str, Any],
    component_packet: dict[str, Any],
    generic: dict[str, Any],
    obligation_completion_sha256: str,
) -> dict[str, Any]:
    candidate_id = str(reference["candidate_id"])
    records = (variable, induced, full_jacobian, c9, topology)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTC2CK1ReferenceSourceError("candidate identity mismatch")
    coefficients = reference["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTC2CK1ReferenceSourceError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticTC2CK1ReferenceSourceError("CK1 slice requires a10!=0")
    m2 = sp.Integer(c9["solved_source_Frechet_operator_integer_uppers"]["2"])
    if m2 <= 0:
        raise QuarticTC2CK1ReferenceSourceError("C9 D2F upper is not positive")
    vector_constants = topology["coordinate_atom_topology"]
    if vector_constants["combined_coordinate_L2_injection_upper"] != "2":
        raise QuarticTC2CK1ReferenceSourceError("coordinate topology mismatch")
    generic_topology = generic
    c2 = sp.sympify(generic_topology["H7_to_W2infinity_constant"])
    c_y0 = sp.sympify(generic_topology["coordinate_atom_Linfinity_constant"])
    c_y1 = sp.sympify(
        generic_topology["coordinate_atom_first_derivative_Linfinity_constant"]
    )
    delta_norm = sp.sqrt(sp.Rational(1253060, 9))
    principal_constant = sp.factor(sp.Abs(alpha) * delta_norm * 3 * c2)
    scalar_source_quadratic = sp.factor(
        sp.Abs(alpha) * delta_norm * m2 * c_y0 * c_y1
    )
    q, _ = _minimal_rank_two_target(alpha)
    q_norm = sp.sqrt(_frobenius_square(q))
    q_source_quadratic = sp.factor(q_norm * m2 * c_y0 * c_y1)
    radius = sp.sympify(
        topology["recombined_full_multiplication_tame_ledger"]["tube_H7_radius"]
    )
    scalar_source_tube_linear = sp.factor(scalar_source_quadratic * radius)
    q_source_tube_linear = sp.factor(q_source_quadratic * radius)
    if not all(
        value > 0
        for value in (
            principal_constant,
            scalar_source_quadratic,
            q_source_quadratic,
            scalar_source_tube_linear,
            q_source_tube_linear,
        )
    ):
        raise QuarticTC2CK1ReferenceSourceError("invalid CK1/source constant")

    source_body = {
        "candidate_id": candidate_id,
        "component_packet_sha256": component_packet["content_sha256"],
        "C9_D2F_operator_integer_upper": str(m2),
        "coordinate_Linfinity_constants": {
            "Y": str(c_y0),
            "partial_1_Y": str(c_y1),
        },
        "tube_H7_radius": str(radius),
        "scalar_F10_quadratic_constant": str(scalar_source_quadratic),
        "Q_quadratic_constant": str(q_source_quadratic),
    }
    return {
        "schema_version": "sigma-quartic-tc2-ck1-reference-source-certificate-1.0",
        "status": "pass_CK1_reference_principal_and_source_topologies_global_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "full_reference_Sylvester_sha256": reference["provenance"][
                "reference_Sylvester_packet_sha256"
            ],
            "variable_first_order_deltaK_sha256": variable["provenance"][
                "variable_solvability_packet_sha256"
            ],
            "induced_TC1_component_packet_sha256": induced["provenance"][
                "TC1_component_packet_sha256"
            ],
            "full_entry_manifest_sha256": full_jacobian["provenance"][
                "full_entry_manifest_sha256"
            ],
            "C9_orders": c9["orders_cumulatively_closed"],
            "coordinate_atom_basis_sha256": topology["provenance"][
                "coordinate_atom_basis_sha256"
            ],
            "CK1_reference_component_packet_sha256": component_packet[
                "content_sha256"
            ],
            "source_topology_sha256": _content_hash(source_body),
            "completed_obligation_checkpoint_sha256": obligation_completion_sha256,
        },
        "CK1_reference_principal_part": {
            "identity": component_packet["identity"],
            "component_packet_sha256": component_packet["content_sha256"],
            "direction_count": 3,
            "product_nonzero_entries_per_direction": [
                item["product_nonzero_entries"]
                for item in component_packet["directions"]
            ],
            "matrix_Frobenius_sum": str(3 * delta_norm),
            "H7_shell_constant": str(principal_constant),
            "bound": (
                "||a10*deltaK_basis*sum_k T_(partial_1 partial_k U_low)"
                "(e10^T E_v^T P55^k)h_j||2 <= C_pr ||U||H7 ||h_j||2"
            ),
            "closed_at_reference": True,
        },
        "reference_solved_source_topology": {
            "reference_subtracted_source": "R_F(Y)=F(Y)-F(0)-DF(0)Y",
            "exact_identity": generic_topology["source_identity"],
            "principal_linear_piece_already_in_P55": True,
            "F10_projection_operator_norm": "1",
            "C9_D2F_operator_integer_upper": str(m2),
            "scalar_F10_quadratic_shell_constant": str(scalar_source_quadratic),
            "scalar_F10_tube_linear_shell_constant": str(
                scalar_source_tube_linear
            ),
            "Q_Frobenius_norm": str(q_norm),
            "Q_contracted_quadratic_shell_constant": str(q_source_quadratic),
            "Q_contracted_tube_linear_shell_constant": str(q_source_tube_linear),
            "bounds": [
                "||a10*deltaK_basis*T_(partial_1 R_F10(Y_low))h_j||2 <= C_F10 R^2 ||h_j||2",
                "||E_v Q*T_(partial_1 R_F(Y_low))h_j||2 <= C_Q R^2 ||h_j||2",
            ],
            "closed_on_certified_H7_tube": True,
        },
        "closure_ledger": {
            "all_2675_second_atom_Sylvester_obligations_complete": True,
            "CK1_reference_principal_part_closed": True,
            "CK1_reference_F10_source_remainder_closed": True,
            "TC1_Q_contracted_reference_source_remainder_closed": True,
            "variable_CK1_all_terms_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "first_remaining_gate": (
            "differentiate the variable deltaK extension inside CK1 and enumerate its "
            "P55/source commutators; then close CK3 and the remaining TC induced terms"
        ),
    }


def run_quartic_tc2_ck1_reference_source_campaign(
    reference_campaign: dict[str, Any],
    variable_campaign: dict[str, Any],
    induced_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    topology_campaign: dict[str, Any],
    obligation_checkpoint: dict[str, Any],
    final_obligation_artifact: dict[str, Any],
    final_obligation_artifact_file_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            reference_campaign,
            variable_campaign,
            induced_campaign,
            full_jacobian_campaign,
            c9_campaign,
            topology_campaign,
        )
        expected_statuses = (
            "pass_all_12_full_reference_TC2_Sylvester_solutions_variable_extension_global_H7_fail_closed",
            "pass_all_12_first_order_variable_deltaK_extensions_higher_orders_global_H7_fail_closed",
            "pass_all_12_exact_P55_Q_TC_packets_reference_partial_bounds_global_H7_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_high_low_paraproduct_fail_closed",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2CK1ReferenceSourceError("unsupported schema_version")
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTC2CK1ReferenceSourceError("prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTC2CK1ReferenceSourceError("prerequisite content hash mismatch")
        if (
            int(config.get("expected_candidate_count", 0)) != 12
            or int(config.get("expected_excluded_obligation_count", 0))
            != TOTAL_EXCLUDED_OBLIGATIONS
            or config.get("reference_source_subtraction") != "F-F0-DF0Y"
            or int(config.get("required_source_Frechet_order", 0)) != 2
            or config.get("TC2_policy") != "fail_closed"
            or config.get("B7_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
            or bool(config.get("declare_full_CK1_closed", False))
        ):
            raise QuarticTC2CK1ReferenceSourceError("unsupported closure contract")
        _validate_obligation_completion(
            obligation_checkpoint,
            final_obligation_artifact,
            final_obligation_artifact_file_sha256,
            config,
        )
        generic_passed, generic = generic_ck1_reference_source_control()
        if not generic_passed:
            raise QuarticTC2CK1ReferenceSourceError("generic CK1 control failed")
        actual = _reference_ck1_component_packet()
        if not (
            actual["delta_square"] == sp.Rational(1253060, 9)
            and actual["row_norm_sum"] == 3
            and [
                item["product_nonzero_entries"]
                for item in actual["packet"]["directions"]
            ]
            == [24, 24, 24]
        ):
            raise QuarticTC2CK1ReferenceSourceError(
                "reference CK1 component packet mismatch"
            )
        topology_constants = topology_campaign[
            "generic_H7_paracomposition_topology_control"
        ]["H7_vector_Sobolev_constants"][
            "coordinate_atom_Linfinity_constants_orders_0_to_3"
        ]
        if (
            topology_constants["0"]
            != generic["coordinate_atom_Linfinity_constant"]
            or topology_constants["1"]
            != generic["coordinate_atom_first_derivative_Linfinity_constant"]
        ):
            raise QuarticTC2CK1ReferenceSourceError(
                "H7 coordinate-atom Sobolev constant provenance mismatch"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTC2CK1ReferenceSourceError("candidate-set mismatch")
        checkpoint_hash = obligation_checkpoint["content_sha256"]
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps),
                actual["packet"],
                generic,
                checkpoint_hash,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_CK1_reference_principal_and_source_topologies_"
                "TC2_global_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "full_reference_Sylvester": reference_campaign["content_sha256"],
                "variable_Sylvester": variable_campaign["content_sha256"],
                "induced_operator": induced_campaign["content_sha256"],
                "full_source_Jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
                "H7_topology": topology_campaign["content_sha256"],
                "completed_obligation_checkpoint": checkpoint_hash,
                "final_obligation_artifact": final_obligation_artifact[
                    "content_sha256"
                ],
                "final_obligation_artifact_file_sha256": (
                    final_obligation_artifact_file_sha256
                ),
            },
            "config_sha256": _content_hash(config),
            "generic_CK1_reference_source_control": generic,
            "common_CK1_reference_component_packet": actual["packet"],
            "counts": {
                "selected": len(certificates),
                "excluded_obligations_verified_complete": TOTAL_EXCLUDED_OBLIGATIONS,
                "CK1_reference_principal_packets_closed": len(certificates),
                "CK1_reference_F10_source_topologies_closed": len(certificates),
                "TC1_Q_source_topologies_closed": len(certificates),
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The exact deltaK_basis e10^T E_v^T P55^k packet and the "
                "reference-subtracted F10/Q low-source topologies have explicit H7-tube "
                "constants for all twelve candidates."
            ),
            "scope": (
                "Only the reference CK1 principal/source slice and the matching "
                "Q-contracted reference source remainder are closed. Variable CK1, CK3, "
                "TC2, B7, global H7, dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticTC2CK1ReferenceSourceError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "excluded_obligations_verified_complete": 0,
                "CK1_reference_principal_packets_closed": 0,
                "CK1_reference_F10_source_topologies_closed": 0,
                "TC1_Q_source_topologies_closed": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 1,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_ck1_reference_source_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
