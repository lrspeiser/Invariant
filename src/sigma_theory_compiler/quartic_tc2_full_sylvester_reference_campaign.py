from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import _first_order_generalized_pencil
from .quartic_first_order_reduction_campaign import (
    _extract_spatial_blocks,
    _full_first_order_pencil,
    _symbol_data,
)
from .quartic_r3_sobolev_calculus_campaign import (
    r3_sobolev_embedding_constant,
)

SCHEMA_VERSION = "sigma-quartic-tc2-full-sylvester-reference-campaign-1.0"
STATE_DIMENSION = 55


class QuarticTC2FullSylvesterReferenceError(ValueError):
    """Raised when the full reference Sylvester audit is overstated."""


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


@cache
def _full_reference_sylvester_packet() -> dict[str, Any]:
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
    mass, evolution = _full_first_order_pencil(
        coefficient_a,
        b_blocks[0],
        [c_blocks[0][right] for right in range(3)],
        [1, 0, 0],
    )
    physical_original = mass.inv() * evolution
    # Appendix-A order: z=(q,w2,w3), y=(v,w1).
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    physical = physical_original.extract(ordering, ordering)
    coupling = physical[33:55, 0:33]
    companion = physical[33:55, 33:55]

    identity22 = sp.eye(22)
    nonzero_spectrum = (
        sp.Integer(1),
        sp.Integer(-1),
        sp.Rational(1, 2),
        sp.Rational(-1, 2),
        sp.Rational(1, 3),
        sp.Rational(-1, 3),
    )
    companion_projectors: dict[sp.Expr, sp.Matrix] = {}
    for eigenvalue in nonzero_spectrum:
        projector = identity22
        for other in nonzero_spectrum:
            if other != eigenvalue:
                projector *= (companion - other * identity22) / (
                    eigenvalue - other
                )
        companion_projectors[eigenvalue] = projector.applyfunc(sp.factor)

    action = _first_order_generalized_pencil(data["action_symbol"], xi[0])
    action_a = action["A"].subs(substitutions)
    action_b = action["B"].subs(substitutions)
    h_plus = action_b.row_join(action_a).col_join(
        action_a.row_join(sp.zeros(11))
    )
    companion_energy = sp.zeros(22)
    for eigenvalue, projector in companion_projectors.items():
        metric = (
            h_plus
            if eigenvalue == 1
            else -h_plus
            if eigenvalue == -1
            else identity22
        )
        companion_energy += projector.T * metric * projector
    companion_energy = companion_energy.applyfunc(sp.factor)
    cross = (
        coupling.T * companion_energy * companion.inv()
    ).applyfunc(sp.factor)
    # The positive scalar zero-block weight drops out of K B because TC2 is entirely
    # in the nonzero companion block. Unit weight is sufficient for this exact audit.
    energy = sp.zeros(STATE_DIMENSION)
    energy[0:33, 0:33] = sp.eye(33)
    energy[0:33, 33:55] = cross
    energy[33:55, 0:33] = cross.T
    energy[33:55, 33:55] = companion_energy
    symmetrizer_residual = (
        energy * physical - physical.T * energy
    ).applyfunc(sp.factor)

    full_spectrum = (sp.Integer(0), *nonzero_spectrum)
    identity55 = sp.eye(STATE_DIMENSION)
    projectors: dict[sp.Expr, sp.Matrix] = {}
    for eigenvalue in full_spectrum:
        projector = identity55
        for other in full_spectrum:
            if other != eigenvalue:
                projector *= (physical - other * identity55) / (
                    eigenvalue - other
                )
        projectors[eigenvalue] = projector.applyfunc(sp.factor)

    q = sp.zeros(11)
    q[0, 10] = 2
    q[4, 10] = -8
    q[10, 7] = 2
    q[10, 9] = 2
    p_eq = sp.zeros(STATE_DIMENSION, 11)
    p_eq[44:55, :] = q
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1

    column_records: list[dict[str, Any]] = []
    delta_basis: dict[int, sp.Matrix] = {}
    all_diagonal_blocks_zero = True
    for column in (7, 9, 10):
        block = p_eq[:, column] * high.T
        skew = (energy * block - block.T * energy).applyfunc(sp.factor)
        diagonal_conditions: list[dict[str, Any]] = []
        for eigenvalue, projector in projectors.items():
            compression = (projector.T * skew * projector).applyfunc(sp.factor)
            zero = compression.is_zero_matrix
            all_diagonal_blocks_zero = all_diagonal_blocks_zero and zero
            diagonal_conditions.append(
                {
                    "eigenvalue": str(eigenvalue),
                    "zero": zero,
                    "nonzero_entries": sum(value != 0 for value in compression),
                }
            )
        delta = sp.zeros(STATE_DIMENSION)
        for left, left_projector in projectors.items():
            for right, right_projector in projectors.items():
                if left != right:
                    delta += (
                        left_projector.T
                        * skew
                        * right_projector
                        / (left - right)
                    )
        delta = delta.applyfunc(sp.factor)
        residual = (
            delta * physical - physical.T * delta + skew
        ).applyfunc(sp.factor)
        delta_basis[column] = delta
        column_records.append(
            {
                "low_field_column": column,
                "skew_nonzero_entries": sum(value != 0 for value in skew),
                "diagonal_solvability_conditions": diagonal_conditions,
                "deltaK_nonzero_entries": sum(value != 0 for value in delta),
                "deltaK_rank": delta.rank(),
                "deltaK_Hermitian": delta.equals(delta.T),
                "Sylvester_residual_zero": residual.is_zero_matrix,
            }
        )
    delta10 = delta_basis[10]
    delta_frobenius_square = sp.factor(sum(value**2 for value in delta10))
    minimum_gap = min(
        abs(left - right)
        for left in full_spectrum
        for right in full_spectrum
        if left != right
    )
    sparse_body = {
        "schema_version": "sigma-full-TC2-Sylvester-reference-packet-1.0",
        "reference": "flat zero jet, M2=1, direction e1",
        "ordered_state": "z=(q,w2,w3), y=(v,w1)",
        "spectrum": [str(value) for value in full_spectrum],
        "minimum_distinct_spectral_gap": str(minimum_gap),
        "P55_sha256": _content_hash(_matrix_payload(physical)),
        "K55_pairing_sha256": _content_hash(_matrix_payload(energy)),
        "K55_P55_minus_P55dagger_K55_zero": symmetrizer_residual.is_zero_matrix,
        "TC2_columns": column_records,
        "all_eigenspace_diagonal_solvability_blocks_zero": (
            all_diagonal_blocks_zero
        ),
        "deltaK_column10_entries": _matrix_entries(delta10),
        "deltaK_column10_Frobenius_square": str(delta_frobenius_square),
    }
    return {
        "packet": {**sparse_body, "content_sha256": _content_hash(sparse_body)},
        "delta10": delta10,
        "delta_frobenius_square": delta_frobenius_square,
    }


@cache
def generic_tc2_full_sylvester_reference_control() -> tuple[bool, dict[str, Any]]:
    lambda_left, lambda_right = sp.symbols("lambda_left lambda_right", real=True)
    skew = sp.Symbol("R_lr")
    delta = skew / (lambda_left - lambda_right)
    residual = sp.factor(
        (lambda_right - lambda_left) * delta + skew
    )
    gap = sp.Rational(1, 6)
    delta_norm_factor = sp.factor(1 / gap)
    failed_diagonal = sp.Symbol("R_ll", nonzero=True)
    passed = bool(residual == 0 and delta_norm_factor == 6 and failed_diagonal != 0)
    return passed, {
        "control": "Hermitian Sylvester equation in the K55-orthogonal P55 eigenbasis",
        "equation": "deltaK P55-P55^dagger deltaK=-R_TC2",
        "eigenbasis_formula": {
            "off_diagonal": (
                "Pi_lambda^dagger deltaK Pi_mu = "
                "Pi_lambda^dagger R_TC2 Pi_mu/(lambda-mu)"
            ),
            "diagonal_solvability": (
                "Pi_lambda^dagger R_TC2 Pi_lambda=0 for every eigenvalue lambda"
            ),
            "generic_off_diagonal_residual": str(residual),
        },
        "spectral_gap": {
            "minimum": str(gap),
            "inverse_upper": str(delta_norm_factor),
        },
        "negative_controls": {
            "divide_diagonal_block_by_zero_gap": {
                "nonzero_diagonal_residual": str(failed_diagonal),
                "rejected": True,
            },
            "omit_projector_off_diagonal_pair": {
                "generic_residual": str(skew),
                "rejected": skew != 0,
            },
            "claim_reference_solution_is_variable_coefficient_solution": {
                "missing": (
                    "D_Y of the eigenspace-diagonal solvability blocks and of deltaK"
                ),
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    minimal: dict[str, Any],
    tc2: dict[str, Any],
    induced: dict[str, Any],
    symmetrizer: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(minimal["candidate_id"])
    records = (tc2, induced, symmetrizer)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTC2FullSylvesterReferenceError("candidate identity mismatch")
    coefficients = minimal["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTC2FullSylvesterReferenceError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticTC2FullSylvesterReferenceError("Sylvester slice requires a10!=0")
    packet = reference["packet"]
    if not (
        packet["all_eigenspace_diagonal_solvability_blocks_zero"]
        and all(item["Sylvester_residual_zero"] for item in packet["TC2_columns"])
        and packet["TC2_columns"][2]["deltaK_rank"] == 4
    ):
        raise QuarticTC2FullSylvesterReferenceError("reference solvability mismatch")
    energy = symmetrizer["energy_equivalence"]
    lower = sp.sympify(energy["K55_2_lower"])
    delta_norm = sp.Abs(alpha) * sp.sqrt(reference["delta_frobenius_square"])
    coefficient_radius = sp.factor(lower / (2 * delta_norm))
    if not coefficient_radius > 0:
        raise QuarticTC2FullSylvesterReferenceError("deltaK positivity radius invalid")
    c2 = r3_sobolev_embedding_constant(7, 2)
    ck3_constant = sp.factor(delta_norm * c2)
    identity_payload = {
        "candidate_id": candidate_id,
        "reference_packet_sha256": packet["content_sha256"],
        "alpha": str(alpha),
        "deltaK": "alpha*ell10*deltaK_column10",
        "Sylvester_residual": "0",
        "positivity_radius": str(coefficient_radius),
    }
    variable_gate_payload = {
        "identity_sha256": _content_hash(identity_payload),
        "first_gate": (
            "D_Y[Pi_lambda^dagger(K55 B-B^dagger K55)Pi_lambda]=0"
        ),
        "coordinate_atom_count": 153,
        "available_K55_information": "C4 operator norm envelopes, not component projectors",
    }
    return {
        "schema_version": "sigma-quartic-tc2-full-sylvester-reference-certificate-1.0",
        "status": "pass_full_reference_Sylvester_solution_variable_extension_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "TC2_component_packet_sha256": tc2["provenance"][
                "TC2_component_packet_sha256"
            ],
            "K55_energy_equivalence_sha256": _content_hash(energy),
            "reference_Sylvester_packet_sha256": packet["content_sha256"],
            "candidate_deltaK_identity_sha256": _content_hash(identity_payload),
            "variable_coefficient_gate_sha256": _content_hash(
                variable_gate_payload
            ),
        },
        "full_reference_Sylvester_solution": {
            "spectrum": packet["spectrum"],
            "minimum_distinct_gap": packet["minimum_distinct_spectral_gap"],
            "all_diagonal_solvability_blocks_zero": True,
            "Hermitian": True,
            "deltaK_formula": "deltaK=alpha*ell10*deltaK_column10",
            "deltaK_nonzero_entries": packet["TC2_columns"][2][
                "deltaK_nonzero_entries"
            ],
            "deltaK_rank": packet["TC2_columns"][2]["deltaK_rank"],
            "deltaK_Frobenius_norm": str(delta_norm),
            "exact_Sylvester_residual_zero": True,
            "columns_7_and_9_already_K55_symmetric": (
                packet["TC2_columns"][0]["skew_nonzero_entries"] == 0
                and packet["TC2_columns"][1]["skew_nonzero_entries"] == 0
            ),
        },
        "positivity_smallness": {
            "condition": "|ell10|<=rho_deltaK",
            "rho_deltaK": str(coefficient_radius),
            "proof": "||deltaK||2<=||deltaK||F<=lambda_K55/2",
            "preserved_lower_margin": str(lower / 2),
            "closed_at_reference": True,
        },
        "CK1_time_derivative_cost": {
            "term": "alpha*deltaK_column10*partial_1 partial_t v0[10]_low",
            "principal_and_source_split_required": True,
            "closed": False,
            "first_missing_bound": (
                "the component packet for deltaK_column10 E10^T E_v^T P55^k and "
                "the Q-independent partial_1 F_10 low-state source topology"
            ),
        },
        "CK3_spatial_derivative_cost": {
            "term": "alpha*deltaK_column10*partial_i partial_1 v0[10]_low",
            "reference_H7_to_W2infinity_constant": str(c2),
            "reference_shell_coefficient": str(ck3_constant),
            "reference_low_factor_bound_closed": True,
            "variable_projector_derivative_closed": False,
        },
        "first_variable_coefficient_obstruction": {
            "condition": variable_gate_payload["first_gate"],
            "why_first": (
                "a nonzero derivative of an equal-eigenvalue compression cannot be divided "
                "by a spectral gap and destroys Sylvester solvability before positivity or "
                "commutator estimates are relevant"
            ),
            "missing_tensor": (
                "componentwise D_Y of all seven spectral projectors, K55, and the TC2 block "
                "contracted into the 153 coordinate-atom diagonal compressions"
            ),
            "available_bounds_insufficient": (
                "mixed K55/P55 operator envelopes do not determine whether these component "
                "compressions vanish"
            ),
            "closed": False,
        },
        "connection_to_TC2_B7_global_H7": {
            "reference_TC2_Sylvester_absorption": True,
            "variable_coefficient_TC2_absorption": False,
            "CK1_closed": False,
            "CK3_fully_closed": False,
            "TC2_closed": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "materialize the 153 first derivatives of every eigenspace-diagonal solvability "
            "compression; only if they vanish, differentiate the off-diagonal projector "
            "formula and close CK1/CK3"
        ),
    }


def run_quartic_tc2_full_sylvester_reference_campaign(
    minimal_no_go_campaign: dict[str, Any],
    tc2_no_go_campaign: dict[str, Any],
    induced_campaign: dict[str, Any],
    full_symmetrizer_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            minimal_no_go_campaign,
            tc2_no_go_campaign,
            induced_campaign,
            full_symmetrizer_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_minimal_coupled_deltaK_same_high_state_no_gos_"
                "TC2_global_H7_fail_closed"
            ),
            (
                "pass_all_12_exact_TC2_unchanged_K55_no_gos_"
                "reciprocal_blocks_missing_global_H7_fail_closed"
            ),
            (
                "pass_all_12_exact_P55_Q_TC_packets_reference_partial_bounds_"
                "global_H7_fail_closed"
            ),
            "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2FullSylvesterReferenceError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTC2FullSylvesterReferenceError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTC2FullSylvesterReferenceError("campaign content hash mismatch")
        if (
            minimal_no_go_campaign["upstream_sha256"]["TC2_symmetrizer_no_go"]
            != tc2_no_go_campaign["content_sha256"]
            or tc2_no_go_campaign["upstream_sha256"]["induced_operator"]
            != induced_campaign["content_sha256"]
        ):
            raise QuarticTC2FullSylvesterReferenceError("upstream provenance mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or config.get("reference_direction") != "e1"
            or config.get("deltaK_class") != "full_nonspectral_Hermitian"
            or config.get("variable_extension_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTC2FullSylvesterReferenceError(
                "unsupported full Sylvester contract"
            )
        generic_passed, generic = generic_tc2_full_sylvester_reference_control()
        if not generic_passed:
            raise QuarticTC2FullSylvesterReferenceError(
                "generic Sylvester control failed"
            )
        reference = _full_reference_sylvester_packet()
        if not (
            reference["packet"]["K55_P55_minus_P55dagger_K55_zero"]
            and reference["packet"][
                "all_eigenspace_diagonal_solvability_blocks_zero"
            ]
            and reference["delta_frobenius_square"] == sp.Rational(1253060, 9)
        ):
            raise QuarticTC2FullSylvesterReferenceError(
                "reference packet control mismatch"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTC2FullSylvesterReferenceError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), reference
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_full_reference_TC2_Sylvester_solutions_"
                "variable_extension_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "minimal_coupled_no_go": minimal_no_go_campaign["content_sha256"],
                "TC2_symmetrizer_no_go": tc2_no_go_campaign["content_sha256"],
                "induced_operator": induced_campaign["content_sha256"],
                "full_K55_symmetrizer": full_symmetrizer_campaign[
                    "content_sha256"
                ],
            },
            "config_sha256": _content_hash(config),
            "generic_full_Sylvester_control": generic,
            "common_full_reference_Sylvester_packet": reference["packet"],
            "counts": {
                "selected": len(certificates),
                "full_reference_Sylvester_solutions": len(certificates),
                "Hermitian_deltaK_solutions": len(certificates),
                "reference_positivity_radii": len(certificates),
                "reference_CK3_low_factor_bounds": len(certificates),
                "variable_coefficient_solvability_proofs": 0,
                "CK1_closures": 0,
                "TC2_closures": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The full non-spectral Hermitian Sylvester equation is exactly solvable at "
                "the flat e1 reference: every equal-eigenvalue TC2 skew compression vanishes, "
                "columns 7/9 are already symmetric, and column 10 has a 24-entry rank-four "
                "deltaK with zero residual and an explicit positivity radius. The first "
                "unresolved variable gate is the 153-atom derivative of the diagonal "
                "solvability compressions, which operator norms cannot decide."
            ),
            "scope": (
                "Reference algebraic absorption and CK3 low-factor constant only. Variable "
                "solvability, CK1, differentiated projectors, TC2, B7, H7, and lifespan "
                "remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2FullSylvesterReferenceError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "full_reference_Sylvester_solutions": 0,
                "Hermitian_deltaK_solutions": 0,
                "reference_positivity_radii": 0,
                "reference_CK3_low_factor_bounds": 0,
                "variable_coefficient_solvability_proofs": 0,
                "CK1_closures": 0,
                "TC2_closures": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_full_sylvester_reference_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
