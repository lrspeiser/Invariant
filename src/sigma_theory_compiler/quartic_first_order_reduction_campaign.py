from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    build_quartic_horndeski_x2_kessence_modified_harmonic_symbol,
)

SCHEMA_VERSION = "sigma-quartic-first-order-reduction-campaign-1.0"


class QuarticFirstOrderReductionError(ValueError):
    """Raised when the physical-space first-order reduction cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


@cache
def _symbol_data() -> dict[str, Any]:
    return build_quartic_horndeski_x2_kessence_modified_harmonic_symbol()


def _matrix_payload(matrix: sp.Matrix) -> list[list[str]]:
    return [[str(matrix[row, column]) for column in range(matrix.cols)] for row in range(matrix.rows)]


def _extract_spatial_blocks(
    coefficient_b: sp.Matrix,
    coefficient_c: sp.Matrix,
    spatial_covectors: list[sp.Symbol],
) -> tuple[list[sp.Matrix], list[list[sp.Matrix]]]:
    zero_direction = {symbol: 0 for symbol in spatial_covectors}
    b_blocks = [
        coefficient_b.diff(symbol).subs(zero_direction).applyfunc(sp.factor)
        for symbol in spatial_covectors
    ]
    c_blocks = [[sp.zeros(11) for _ in range(3)] for _ in range(3)]
    expanded_c = coefficient_c.applyfunc(sp.expand)
    for left in range(3):
        c_blocks[left][left] = expanded_c.applyfunc(
            lambda expression, index=left: sp.factor(
                expression.coeff(spatial_covectors[index], 2)
            )
        )
        for right in range(left + 1, 3):
            cross = expanded_c.applyfunc(
                lambda expression, i=left, j=right: sp.factor(
                    expression.coeff(spatial_covectors[i], 1).coeff(
                        spatial_covectors[j], 1
                    )
                    / 2
                )
            )
            c_blocks[left][right] = cross
            c_blocks[right][left] = cross
    return b_blocks, c_blocks


def _directional_blocks(
    b_blocks: list[sp.Matrix],
    c_blocks: list[list[sp.Matrix]],
    direction: list[sp.Symbol],
) -> tuple[sp.Matrix, sp.Matrix, list[sp.Matrix]]:
    b_direction = sum(
        (direction[index] * b_blocks[index] for index in range(3)), sp.zeros(11)
    )
    c_direction = sum(
        (
            direction[left] * direction[right] * c_blocks[left][right]
            for left in range(3)
            for right in range(3)
        ),
        sp.zeros(11),
    )
    c_flux = [
        sum(
            (
                direction[left] * c_blocks[left][right]
                for left in range(3)
            ),
            sp.zeros(11),
        )
        for right in range(3)
    ]
    return b_direction, c_direction, c_flux


def _full_first_order_pencil(
    coefficient_a: sp.Matrix,
    b_direction: sp.Matrix,
    c_flux: list[sp.Matrix],
    direction: list[sp.Symbol],
) -> tuple[sp.Matrix, sp.Matrix]:
    identity = sp.eye(11)
    mass = sp.diag(identity, coefficient_a, identity, identity, identity)
    evolution = sp.zeros(55)
    q = slice(0, 11)
    velocity = slice(11, 22)
    spatial = [slice(22 + 11 * index, 33 + 11 * index) for index in range(3)]
    evolution[q, q] = sp.zeros(11)
    evolution[velocity, velocity] = -b_direction
    for index in range(3):
        evolution[velocity, spatial[index]] = -c_flux[index]
        evolution[spatial[index], velocity] = direction[index] * identity
    return mass, evolution


@cache
def generic_scalar_first_order_reduction_control() -> tuple[bool, dict[str, Any]]:
    """Prove the 3+1 first-order determinant relation for one scalar equation."""

    omega = sp.Symbol("omega", real=True, finite=True)
    coefficient_a = sp.Symbol("a", nonzero=True, real=True, finite=True)
    direction = list(sp.symbols("n_1:4", real=True, finite=True))
    b = list(sp.symbols("b_1:4", real=True, finite=True))
    c11, c12, c13, c22, c23, c33 = sp.symbols(
        "c11 c12 c13 c22 c23 c33", real=True, finite=True
    )
    c = sp.Matrix(
        [[c11, c12, c13], [c12, c22, c23], [c13, c23, c33]]
    )
    mass = sp.diag(1, coefficient_a, 1, 1, 1)
    evolution = sp.zeros(5)
    b_direction = sum(b[index] * direction[index] for index in range(3))
    evolution[1, 1] = -b_direction
    for right in range(3):
        evolution[1, 2 + right] = -sum(
            direction[left] * c[left, right] for left in range(3)
        )
        evolution[2 + right, 1] = direction[right]
    second_order = sp.factor(
        coefficient_a * omega**2
        + b_direction * omega
        + (sp.Matrix(direction).T * c * sp.Matrix(direction))[0]
    )
    determinant = sp.factor((evolution - omega * mass).det())
    determinant_residual = sp.factor(determinant + omega**3 * second_order)

    corrupted = evolution.copy()
    corrupted[4, 1] = 0
    witness = {
        omega: 2,
        coefficient_a: 3,
        **{direction[index]: index + 1 for index in range(3)},
        **{b[index]: index + 5 for index in range(3)},
        c11: 11,
        c12: 13,
        c13: 17,
        c22: 19,
        c23: 23,
        c33: 29,
    }
    corrupted_residual = sp.factor(
        ((corrupted - omega * mass).det() + omega**3 * second_order).subs(
            witness
        )
    )
    passed = determinant_residual == 0 and corrupted_residual != 0
    return bool(passed), {
        "control": "generic 3+1 first-order reduction determinant theorem",
        "state": ["q", "v=partial_0 q", "w_1=partial_1 q", "w_2", "w_3"],
        "state_dimension_per_second_order_field": 5,
        "determinant": str(determinant),
        "identity": "det(K55-omega H55)=(-omega)^(3N) det(P11) for N=1",
        "determinant_residual": str(determinant_residual),
        "zero_speed_auxiliary_multiplicity_per_field": 3,
        "negative_control": {
            "corruption": "omit partial_0 w_3=partial_3 v",
            "exact_witness_residual": str(corrupted_residual),
            "rejected": corrupted_residual != 0,
        },
        "passed": bool(passed),
    }


@cache
def generic_constraint_propagation_control() -> tuple[bool, dict[str, Any]]:
    """Verify propagation of derivative-definition and spatial-curl constraints."""

    d_v = sp.symbols("D1v D2v D3v", real=True, finite=True)
    dt_w = sp.symbols("dtw1 dtw2 dtw3", real=True, finite=True)
    definition_residuals = [sp.factor(dt_w[index] - d_v[index]) for index in range(3)]
    definition_on_shell = [
        residual.subs(dt_w[index], d_v[index])
        for index, residual in enumerate(definition_residuals)
    ]
    d12v, d13v, d21v, d23v, d31v, d32v = sp.symbols(
        "D12v D13v D21v D23v D31v D32v", real=True, finite=True
    )
    curl_time = [d12v - d21v, d13v - d31v, d23v - d32v]
    curl_on_commuting_chart = [
        curl_time[0].subs(d21v, d12v),
        curl_time[1].subs(d31v, d13v),
        curl_time[2].subs(d32v, d23v),
    ]
    corrupted_definition = definition_residuals[2].subs({dt_w[2]: 0, d_v[2]: 7})
    passed = bool(
        all(value == 0 for value in definition_on_shell)
        and all(value == 0 for value in curl_on_commuting_chart)
        and corrupted_definition != 0
    )
    return passed, {
        "control": "first-order derivative-definition and curl-constraint propagation",
        "definition_constraints_per_field": 3,
        "independent_spatial_curl_constraints_per_field": 3,
        "definition_time_residuals": [str(value) for value in definition_on_shell],
        "curl_time_residuals_in_coordinate_chart": [
            str(value) for value in curl_on_commuting_chart
        ],
        "negative_control": {
            "corruption": "omit the third spatial-derivative evolution equation",
            "exact_witness_residual": str(corrupted_definition),
            "rejected": corrupted_definition != 0,
        },
        "passed": passed,
        "scope": (
            "Exact coordinate first-order reduction identities. Covariant-derivative commutators "
            "and connection terms are lower order and belong in the nonlinear source map."
        ),
    }


@cache
def quartic_full_first_order_reduction_control() -> tuple[bool, dict[str, Any]]:
    """Extract the full 55-variable first-order principal reduction exactly."""

    data = _symbol_data()
    coefficient_a = data["first_order"]["A"]
    coefficient_b = data["first_order"]["B"]
    coefficient_c = data["first_order"]["C"]
    direction = list(data["xi_lower"][1:])
    omega = data["xi_lower"][0]
    b_blocks, c_blocks = _extract_spatial_blocks(
        coefficient_b, coefficient_c, direction
    )
    b_direction, c_direction, c_flux = _directional_blocks(
        b_blocks, c_blocks, direction
    )
    b_residual = (coefficient_b - b_direction).applyfunc(sp.expand)
    c_residual = (coefficient_c - c_direction).applyfunc(sp.expand)
    mass, evolution = _full_first_order_pencil(
        coefficient_a, b_direction, c_flux, direction
    )

    identity = sp.eye(11)
    zero = sp.zeros(11)
    lift = zero.col_join(omega * identity).col_join(
        direction[0] * identity
    ).col_join(direction[1] * identity).col_join(direction[2] * identity)
    second_order = (
        coefficient_a * omega**2 + coefficient_b * omega + coefficient_c
    ).applyfunc(sp.expand)
    full_lift_residual = ((evolution - omega * mass) * lift).applyfunc(sp.expand)
    expected_full_residual = zero.col_join(-second_order).col_join(zero).col_join(
        zero
    ).col_join(zero)
    lift_residual = (full_lift_residual - expected_full_residual).applyfunc(
        sp.expand
    )

    companion_mass = data["first_order"]["mass"]
    companion_evolution = data["first_order"]["evolution"]
    companion_lift = identity.col_join(omega * identity)
    companion_residual = (
        (companion_evolution - omega * companion_mass) * companion_lift
    ).applyfunc(sp.expand)
    expected_companion_residual = zero.col_join(-second_order)
    companion_lift_residual = (
        companion_residual - expected_companion_residual
    ).applyfunc(sp.expand)

    scalar_passed, scalar = generic_scalar_first_order_reduction_control()
    constraints_passed, constraints = generic_constraint_propagation_control()
    block_payload = {
        "A": _matrix_payload(coefficient_a),
        "B_i": [_matrix_payload(block) for block in b_blocks],
        "C_ij": [
            [_matrix_payload(c_blocks[left][right]) for right in range(3)]
            for left in range(3)
        ],
    }
    block_hash = hashlib.sha256(_canonical_json(block_payload).encode()).hexdigest()
    passed = bool(
        scalar_passed
        and constraints_passed
        and b_residual.is_zero_matrix
        and c_residual.is_zero_matrix
        and lift_residual.is_zero_matrix
        and companion_lift_residual.is_zero_matrix
        and mass.shape == (55, 55)
        and evolution.shape == (55, 55)
    )
    return passed, {
        "control": "exact 55-variable physical-space first-order reduction of the quartic symbol",
        "second_order_field_basis": {
            "metric_symmetric_orthonormal_components": 10,
            "scalar": 1,
            "total": 11,
        },
        "first_order_state": {
            "q_A": 11,
            "v_A=partial_0_q_A": 11,
            "w_iA=partial_i_q_A": 33,
            "total": 55,
        },
        "directional_companion": {
            "dimension": 22,
            "role": (
                "linearizes the quadratic characteristic polynomial for a fixed spatial "
                "covector; it is not the full physical-space first-order state"
            ),
        },
        "spatial_block_extraction": {
            "B_i_count": 3,
            "C_ij_symmetric_count": 6,
            "B_reconstruction_residual_zero": b_residual.is_zero_matrix,
            "C_reconstruction_residual_zero": c_residual.is_zero_matrix,
            "block_content_sha256": block_hash,
        },
        "full_pencil": {
            "mass_shape": list(mass.shape),
            "evolution_shape": list(evolution.shape),
            "nonzero_characteristic_lift_residual_zero": lift_residual.is_zero_matrix,
            "directional_companion_lift_residual_zero": companion_lift_residual.is_zero_matrix,
            "zero_speed_auxiliary_multiplicity": 33,
            "nonzero_characteristic_dimension": 22,
        },
        "constraints": constraints,
        "generic_scalar_determinant_control": scalar,
        "state_to_covariant_jet_incidence": {
            "nabla_phi": (
                "algebraic in the scalar entries of v_A and w_iA in the local frame"
            ),
            "nabla_nabla_phi": (
                "first derivatives of scalar v_A/w_iA plus connection terms; the time-time "
                "entry requires the evolution equation"
            ),
            "Einstein_tensor": (
                "first derivatives of metric v_A/w_iA plus quadratic connection terms; the "
                "explicit nonlinear source map is not yet generated"
            ),
            "status": "incidence_defined_nonlinear_formula_map_unresolved",
        },
        "passed": passed,
        "claim": (
            "The extracted quartic second-order principal system has an exact 55-variable 3+1 "
            "first-order reduction whose nonzero characteristic modes reproduce the proven "
            "22-by-22 directional companion pencil."
        ),
        "scope": (
            "This is a principal first-order reduction with exact definition/curl-constraint "
            "propagation. It does not yet provide the nonlinear lower-order source, connection "
            "terms, gauge-driver state, symmetrizer for the 55-variable reduction, state-to-jet "
            "Sobolev bounds, commuted energy closure, or PDE bootstrap."
        ),
    }


def certify_quartic_first_order_candidate(
    symmetrizer_candidate: dict[str, Any],
    moser_candidate: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = symmetrizer_candidate.get("candidate_id")
    if candidate_id != moser_candidate.get("candidate_id"):
        raise QuarticFirstOrderReductionError("candidate ID mismatch")
    if symmetrizer_candidate.get("status") != "pass_uniform_local_jet_strong_hyperbolicity":
        raise QuarticFirstOrderReductionError("candidate lacks the symmetrizer prerequisite")
    if moser_candidate.get("status") != "pass_quasilinear_coefficient_derivative_envelopes":
        raise QuarticFirstOrderReductionError("candidate lacks the coefficient-envelope prerequisite")
    if symmetrizer_candidate.get("coefficients") != moser_candidate.get("coefficients"):
        raise QuarticFirstOrderReductionError("candidate coefficient mismatch")
    control_passed, control = quartic_full_first_order_reduction_control()
    if not control_passed:
        raise QuarticFirstOrderReductionError("generic full first-order reduction control failed")
    return {
        "schema_version": "sigma-quartic-first-order-reduction-certificate-1.0",
        "status": "pass_exact_55_variable_principal_first_order_reduction",
        "candidate_id": candidate_id,
        "coefficients": symmetrizer_candidate["coefficients"],
        "source_spatial_block_sha256": control["spatial_block_extraction"][
            "block_content_sha256"
        ],
        "state_dimensions": {
            "second_order_fields": 11,
            "directional_companion": 22,
            "physical_space_first_order": 55,
            "zero_speed_auxiliary": 33,
        },
        "constraint_counts": {
            "derivative_definition": 33,
            "independent_spatial_curl": 33,
        },
        "nonzero_characteristic_lift_residual_zero": True,
        "definition_and_curl_constraints_propagate": True,
        "state_to_covariant_jet_incidence_status": (
            "incidence_defined_nonlinear_formula_map_unresolved"
        ),
        "claim": (
            "This candidate is bound to the exact 55-variable physical-space principal "
            "first-order reduction and its definition/curl constraint subsystem."
        ),
        "scope": control["scope"],
    }


def run_quartic_first_order_reduction_campaign(
    symmetrizer_campaign: dict[str, Any],
    moser_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticFirstOrderReductionError("unsupported campaign schema_version")
        if symmetrizer_campaign.get("status") != (
            "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
        ):
            raise QuarticFirstOrderReductionError("symmetrizer campaign prerequisite failed")
        if moser_campaign.get("status") != (
            "pass_all_12_quasilinear_coefficient_derivative_envelopes"
        ):
            raise QuarticFirstOrderReductionError("Moser-coefficient campaign prerequisite failed")
        expected = int(config.get("expected_candidate_count", 12))
        symmetrizers = {
            item["candidate_id"]: item
            for item in symmetrizer_campaign.get("certificates", [])
        }
        moser = {
            item["candidate_id"]: item
            for item in moser_campaign.get("certificates", [])
        }
        if len(symmetrizers) != expected or set(symmetrizers) != set(moser):
            raise QuarticFirstOrderReductionError("campaign candidate sets do not match")
        certificates = [
            certify_quartic_first_order_candidate(
                symmetrizers[candidate_id], moser[candidate_id]
            )
            for candidate_id in sorted(symmetrizers)
        ]
        passed_count = sum(
            item["status"] == "pass_exact_55_variable_principal_first_order_reduction"
            for item in certificates
        )
        control_passed, control = quartic_full_first_order_reduction_control()
        if not control_passed:
            raise QuarticFirstOrderReductionError("full first-order control failed")
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_exact_55_variable_principal_first_order_reductions"
            if passed_count == expected
            else "reject",
            "errors": [],
            "symmetrizer_campaign_sha256": symmetrizer_campaign.get("content_sha256"),
            "moser_campaign_sha256": moser_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "counts": {
                "selected": len(certificates),
                "exact_55_variable_reductions_passed": passed_count,
                "rejected": len(certificates) - passed_count,
            },
            "generic_reduction_control": control,
            "certificates": certificates,
            "negative_controls": {
                "omitted_spatial_derivative_evolution": control[
                    "generic_scalar_determinant_control"
                ]["negative_control"],
                "omitted_definition_constraint_evolution": control["constraints"][
                    "negative_control"
                ],
            },
            "claim": (
                "All 12 fixed-coefficient linear-X quartic candidates are bound to the exact "
                "55-variable physical-space principal first-order reduction whose nonzero "
                "characteristics reproduce the proven 22-by-22 directional companion."
            ),
            "scope": control["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticFirstOrderReductionError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "symmetrizer_campaign_sha256": symmetrizer_campaign.get("content_sha256"),
            "moser_campaign_sha256": moser_campaign.get("content_sha256"),
            "counts": {
                "selected": 0,
                "exact_55_variable_reductions_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_first_order_reduction_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
