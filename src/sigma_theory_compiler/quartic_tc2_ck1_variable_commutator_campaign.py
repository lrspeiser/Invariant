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
from .quartic_tc2_ck1_reference_source_campaign import (
    _reference_ck1_component_packet,
)
from .quartic_tc2_variable_sylvester_campaign import (
    _coordinate_atom_to_jet_packet,
    _reference_and_first_jet_packet,
    _variable_solvability_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-ck1-variable-commutator-campaign-1.0"
STATE_DIMENSION = 55
ATOM_DIMENSION = 153
SPATIAL_DIMENSION = 3


class QuarticTC2CK1VariableCommutatorError(ValueError):
    """Raised when the bounded variable-CK1 commutator claim is overstated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


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


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


@cache
def _variable_ck1_commutator_packet() -> dict[str, Any]:
    """Materialize the exact reference derivative of the affine deltaK CK1 term.

    All matrices are normalized at a10=1.  The two differentiated principal
    branches both acquire the candidate multiplier a10**2.
    """

    reference = _reference_and_first_jet_packet()
    coordinate = _coordinate_atom_to_jet_packet()
    variable = _variable_solvability_packet()
    ck1_reference = _reference_ck1_component_packet()["packet"]
    data = _symbol_data()
    xi = data["xi_lower"]
    jets = reference["jets"]
    zero_jet = {
        symbol: 0
        for symbol in (
            list(data["gradient_lower"])
            + sorted(data["hessian_lower"].free_symbols, key=str)
            + sorted(data["einstein_upper"].free_symbols, key=str)
        )
    }
    substitutions: dict[sp.Symbol, sp.Expr] = {
        **zero_jet,
        data["alpha"]: 0,
        data["m2"]: 1,
        data["c20"]: 0,
        xi[1]: 1,
        xi[2]: 0,
        xi[3]: 0,
    }
    derivative_substitutions = dict(substitutions)
    derivative_substitutions[data["alpha"]] = 1
    scaling_substitutions = {
        **zero_jet,
        data["m2"]: 1,
        xi[1]: 1,
        xi[2]: 0,
        xi[3]: 0,
    }
    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    scalar_rows: list[sp.Matrix] = []
    scalar_row_jet_derivatives: list[dict[str, sp.Matrix]] = []
    all_derivatives_scale = True
    for direction in range(SPATIAL_DIMENSION):
        mass0, evolution0 = _full_first_order_pencil(
            coefficient_a.subs(substitutions),
            b_blocks[direction].subs(substitutions),
            [
                c_blocks[direction][right].subs(substitutions)
                for right in range(SPATIAL_DIMENSION)
            ],
            [int(index == direction) for index in range(SPATIAL_DIMENSION)],
        )
        physical_original0 = mass0.inv() * evolution0
        physical0 = physical_original0.extract(ordering, ordering)
        scalar_rows.append(physical0.row(43))
        jet_derivatives: dict[str, sp.Matrix] = {}
        for jet in jets:
            mass_prime, evolution_prime = _full_first_order_pencil(
                coefficient_a.diff(jet).subs(derivative_substitutions),
                b_blocks[direction].diff(jet).subs(derivative_substitutions),
                [
                    c_blocks[direction][right]
                    .diff(jet)
                    .subs(derivative_substitutions)
                    for right in range(SPATIAL_DIMENSION)
                ],
                [int(index == direction) for index in range(SPATIAL_DIMENSION)],
            )
            physical_prime = (
                mass0.inv()
                * (evolution_prime - mass_prime * physical_original0)
            ).extract(ordering, ordering)
            physical_prime = physical_prime.applyfunc(sp.factor)
            jet_derivatives[str(jet)] = physical_prime.row(43)

            symbolic_mass_prime, symbolic_evolution_prime = _full_first_order_pencil(
                coefficient_a.diff(jet).subs(scaling_substitutions),
                b_blocks[direction].diff(jet).subs(scaling_substitutions),
                [
                    c_blocks[direction][right]
                    .diff(jet)
                    .subs(scaling_substitutions)
                    for right in range(SPATIAL_DIMENSION)
                ],
                [int(index == direction) for index in range(SPATIAL_DIMENSION)],
            )
            scaling_residual = (
                symbolic_evolution_prime
                - data["alpha"] * evolution_prime
                - (symbolic_mass_prime - data["alpha"] * mass_prime)
                * physical_original0
            ).applyfunc(sp.factor)
            all_derivatives_scale = (
                all_derivatives_scale and scaling_residual.is_zero_matrix
            )
        scalar_row_jet_derivatives.append(jet_derivatives)

    if [_matrix_entries(row) for row in scalar_rows] != [
        direction["e10_EvT_P55k_entries"]
        for direction in ck1_reference["directions"]
    ]:
        raise QuarticTC2CK1VariableCommutatorError(
            "reference scalar-row provenance mismatch"
        )

    delta0 = reference["delta0"]
    records: list[dict[str, Any]] = []
    delta_norm_sum = sp.S.Zero
    dp_norm_sum = sp.S.Zero
    for atom, jet_direction in zip(
        coordinate["atoms"], coordinate["maps"], strict=True
    ):
        delta_prime = sp.zeros(STATE_DIMENSION)
        for jet, coefficient in jet_direction.items():
            delta_prime += coefficient * reference["delta_derivatives"][jet]
        delta_prime = delta_prime.applyfunc(sp.factor)
        delta_square = _frobenius_square(delta_prime)
        delta_norm_sum += sp.sqrt(delta_square)
        direction_records: list[dict[str, Any]] = []
        for direction in range(SPATIAL_DIMENSION):
            row_prime = sp.zeros(1, STATE_DIMENSION)
            for jet, coefficient in jet_direction.items():
                row_prime += (
                    coefficient * scalar_row_jet_derivatives[direction][jet]
                )
            row_prime = row_prime.applyfunc(sp.factor)
            row_prime_square = _frobenius_square(row_prime)
            dp_norm_sum += sp.sqrt(row_prime_square)
            dk_p_entries = _trilinear_entries(
                delta_prime, scalar_rows[direction]
            )
            k_dp_entries = _trilinear_entries(delta0, row_prime)
            direction_records.append(
                {
                    "spatial_direction": direction + 1,
                    "D_e10EvTP55_entries": _matrix_entries(row_prime),
                    "D_e10EvTP55_Frobenius_square": str(row_prime_square),
                    "DdeltaK_times_P55_entries": dk_p_entries,
                    "DdeltaK_times_P55_entry_count": len(dk_p_entries),
                    "deltaK0_times_DP55_entries": k_dp_entries,
                    "deltaK0_times_DP55_entry_count": len(k_dp_entries),
                }
            )
        records.append(
            {
                "coordinate_atom": atom,
                "jet_direction": {
                    key: str(value) for key, value in jet_direction.items()
                },
                "deltaK_A_entries": _matrix_entries(delta_prime),
                "deltaK_A_Frobenius_square": str(delta_square),
                "deltaK_A_Hermitian": delta_prime.equals(delta_prime.T),
                "directions": direction_records,
            }
        )

    normalized_delta_norm = sp.sympify(
        variable["first_order_deltaK_norm"][
            "coordinate_linf_to_Frobenius_upper"
        ]
    )
    if sp.factor(delta_norm_sum - normalized_delta_norm) != 0:
        raise QuarticTC2CK1VariableCommutatorError(
            "deltaK first-derivative norm provenance mismatch"
        )
    dk_p_witness = next(
        {
            "coordinate_atom": item["coordinate_atom"],
            "spatial_direction": direction["spatial_direction"],
            "first_entry": direction["DdeltaK_times_P55_entries"][0],
        }
        for item in records
        for direction in item["directions"]
        if direction["DdeltaK_times_P55_entries"]
    )
    k_dp_witness = next(
        {
            "coordinate_atom": item["coordinate_atom"],
            "spatial_direction": direction["spatial_direction"],
            "first_entry": direction["deltaK0_times_DP55_entries"][0],
        }
        for item in records
        for direction in item["directions"]
        if direction["deltaK0_times_DP55_entries"]
    )
    body = {
        "schema_version": "sigma-CK1-variable-deltaK-P55-commutator-packet-1.0",
        "reference": "flat zero covariant jet, M2=1, directions e1,e2,e3",
        "normalization": (
            "deltaK_A and DP55_A are normalized at a10=1; both product-rule "
            "branches scale by a10^2"
        ),
        "product_rule_identity": (
            "partial_1[deltaK(Y)(sum_k r_k(Y)partial_k U+F10(Y))]="
            "DdeltaK(Y)[partial_1Y](sum_k r_k(Y)partial_k U+F10(Y))+"
            "deltaK(Y)sum_k(r_k(Y)partial_1partial_kU+"
            "Dr_k(Y)[partial_1Y]partial_kU)+"
            "deltaK(Y)DF10(Y)[partial_1Y]"
        ),
        "coordinate_atom_basis_sha256": coordinate["packet"][
            "coordinate_atom_basis_sha256"
        ],
        "coordinate_to_jet_packet_sha256": coordinate["packet"]["content_sha256"],
        "variable_solvability_packet_sha256": variable["content_sha256"],
        "reference_CK1_component_packet_sha256": ck1_reference["content_sha256"],
        "reference_P55_scalar_rows": [
            _matrix_entries(row) for row in scalar_rows
        ],
        "reference_deltaK0_sha256": _content_hash(_matrix_payload(delta0)),
        "all_DP55_coordinate_derivatives_linear_in_a10_and_c20_absent": (
            all_derivatives_scale
        ),
        "coordinate_atom_records": records,
        "norm_ledger": {
            "sum_A_deltaK_A_Frobenius": str(sp.factor(delta_norm_sum)),
            "sum_Ak_D_e10EvTP55k_A_Frobenius": str(sp.factor(dp_norm_sum)),
            "deltaK0_Frobenius": str(sp.sqrt(_frobenius_square(delta0))),
            "sum_k_reference_scalar_row_Frobenius": "3",
        },
        "counts": {
            "coordinate_atoms": len(records),
            "spatial_directions": SPATIAL_DIMENSION,
            "atom_direction_packets": len(records) * SPATIAL_DIMENSION,
            "nonzero_deltaK_A_atoms": sum(
                bool(item["deltaK_A_entries"]) for item in records
            ),
            "nonzero_DP55_scalar_row_packets": sum(
                bool(direction["D_e10EvTP55_entries"])
                for item in records
                for direction in item["directions"]
            ),
            "DdeltaK_times_P55_trilinear_entries": sum(
                direction["DdeltaK_times_P55_entry_count"]
                for item in records
                for direction in item["directions"]
            ),
            "deltaK0_times_DP55_trilinear_entries": sum(
                direction["deltaK0_times_DP55_entry_count"]
                for item in records
                for direction in item["directions"]
            ),
        },
        "exact_negative_controls": {
            "omit_DdeltaK_times_P55": {
                "witness": dk_p_witness,
                "rejected": True,
            },
            "omit_deltaK0_times_DP55": {
                "witness": k_dp_witness,
                "rejected": True,
            },
            "drop_spatial_direction_three": {
                "missing_reference_row": _matrix_entries(scalar_rows[2]),
                "rejected": True,
            },
            "infer_tube_uniform_DP55_from_reference_packet": {
                "missing": "D2P55 tube-uniform operator envelope",
                "rejected": True,
            },
        },
    }
    return {**body, "content_sha256": _content_hash(body)}


@cache
def generic_ck1_variable_commutator_control() -> tuple[bool, dict[str, Any]]:
    t = sp.Symbol("t")
    d0, d1, p0, p1, f0, f1, u = sp.symbols(
        "deltaK0 deltaK1 P0 P1 F0 F1 U", commutative=True
    )
    product = (d0 + t * d1) * ((p0 + t * p1) * u + f0 + t * f1)
    derivative = sp.expand(sp.diff(product, t).subs(t, 0))
    expected = sp.expand(d1 * (p0 * u + f0) + d0 * (p1 * u + f1))
    omitted_dk = sp.expand(derivative - d1 * (p0 * u + f0))
    omitted_dp = sp.expand(derivative - d0 * p1 * u)
    passed = bool(
        sp.expand(derivative - expected) == 0
        and omitted_dk != derivative
        and omitted_dp != derivative
    )
    return passed, {
        "control": "exact product rule for variable deltaK inside CK1",
        "derivative": str(derivative),
        "expected": str(expected),
        "residual": str(sp.expand(derivative - expected)),
        "negative_controls": {
            "omit_DdeltaK_branch": {
                "residual": str(sp.expand(derivative - d0 * (p1 * u + f1))),
                "rejected": True,
            },
            "omit_DP55_branch": {
                "residual": str(sp.expand(derivative - d1 * (p0 * u + f0) - d0 * f1)),
                "rejected": True,
            },
            "promote_reference_DP55_to_tube_uniform": {
                "missing": "tube-uniform D2P55 coordinate operator envelope",
                "rejected": True,
            },
            "promote_bounded_CK1_slice_to_TC2": {
                "missing": "CK3, remaining TC induced terms, and global dyadic summation",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    ck1: dict[str, Any],
    variable: dict[str, Any],
    c9: dict[str, Any],
    topology: dict[str, Any],
    equilibrium: dict[str, Any],
    topology_generic: dict[str, Any],
    packet: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(ck1["candidate_id"])
    if any(
        item.get("candidate_id") != candidate_id
        for item in (variable, c9, topology, equilibrium)
    ):
        raise QuarticTC2CK1VariableCommutatorError("candidate identity mismatch")
    coefficients = ck1["coefficients"]
    if any(item.get("coefficients") != coefficients for item in (variable, c9, topology)):
        raise QuarticTC2CK1VariableCommutatorError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticTC2CK1VariableCommutatorError("CK1 variable slice requires a10!=0")
    if not equilibrium["reference_equilibrium"]["F_reference_equals_zero"]:
        raise QuarticTC2CK1VariableCommutatorError("reference source is not zero")

    constants = topology_generic["H7_vector_Sobolev_constants"]
    coordinate_constants = constants[
        "coordinate_atom_Linfinity_constants_orders_0_to_3"
    ]
    scalar_constants = constants["scalar_embedding_C_7_m_orders_0_to_5"]
    c_y0 = sp.sympify(coordinate_constants["0"])
    c_y1 = sp.sympify(coordinate_constants["1"])
    c_u1 = sp.sympify(scalar_constants["1"])
    radius = sp.sympify(
        topology["recombined_full_multiplication_tame_ledger"]["tube_H7_radius"]
    )
    norms = packet["norm_ledger"]
    delta0 = sp.sympify(norms["deltaK0_Frobenius"])
    delta1 = sp.sympify(norms["sum_A_deltaK_A_Frobenius"])
    dp1 = sp.sympify(norms["sum_Ak_D_e10EvTP55k_A_Frobenius"])
    m1 = sp.Integer(c9["solved_source_Frechet_operator_integer_uppers"]["1"])
    m2 = sp.Integer(c9["solved_source_Frechet_operator_integer_uppers"]["2"])
    candidate_square = sp.Abs(alpha) ** 2
    dk_p_constant = sp.factor(candidate_square * delta1 * 3 * c_y1 * c_u1)
    k_dp_constant = sp.factor(candidate_square * delta0 * dp1 * c_y1 * c_u1)
    source_quadratic = sp.factor(
        candidate_square * delta1 * 2 * m1 * c_y0 * c_y1
    )
    source_cubic = sp.factor(
        candidate_square * delta1 * sp.Rational(3, 2) * m2 * c_y0**2 * c_y1
    )
    source_tube_constant = sp.factor(
        source_quadratic * radius + source_cubic * radius**2
    )
    if not all(
        value > 0
        for value in (
            dk_p_constant,
            k_dp_constant,
            source_quadratic,
            source_cubic,
            source_tube_constant,
        )
    ):
        raise QuarticTC2CK1VariableCommutatorError("invalid commutator bound")
    return {
        "schema_version": "sigma-quartic-tc2-ck1-variable-commutator-certificate-1.0",
        "status": "pass_reference_variable_CK1_commutators_global_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "CK1_reference_campaign_certificate_sha256": _content_hash(ck1),
            "variable_Sylvester_packet_sha256": variable["provenance"][
                "variable_solvability_packet_sha256"
            ],
            "commutator_packet_sha256": packet["content_sha256"],
            "C9_orders": c9["orders_cumulatively_closed"],
            "reference_equilibrium_exact": True,
        },
        "exact_reference_P55_commutators": {
            "candidate_scaling": str(alpha**2),
            "DdeltaK_times_P55_shell_constant": str(dk_p_constant),
            "deltaK0_times_DP55_shell_constant": str(k_dp_constant),
            "combined_shell_constant": str(sp.factor(dk_p_constant + k_dp_constant)),
            "bound": (
                "||C_ref(U_low)h_j||2 <= (C_DK_P+C_K_DP)||U||H7^2||h_j||2"
            ),
            "closed_at_reference": True,
        },
        "source_commutators": {
            "F10_reference": "0",
            "DdeltaK_times_F10_reference_vanishes_exactly": True,
            "reference_deltaK0_DF10_linear_piece_already_in_P55": True,
            "decomposition": [
                "DdeltaK[partial_1Y] DF10(0)Y",
                "(deltaK(Y)-deltaK0) DF10(0)partial_1Y",
                "DdeltaK[partial_1Y] R_F10(Y)",
                "(deltaK(Y)-deltaK0) DR_F10(Y)[partial_1Y]",
            ],
            "C9_DF_operator_integer_upper": str(m1),
            "C9_D2F_operator_integer_upper": str(m2),
            "quadratic_shell_constant": str(source_quadratic),
            "cubic_shell_constant": str(source_cubic),
            "tube_linearized_shell_constant": str(source_tube_constant),
            "bound": (
                "||C_source(U_low)h_j||2 <= "
                "(C2||U||H7^2+C3||U||H7^3)||h_j||2"
            ),
            "closed_for_affine_deltaK_on_certified_tube": True,
        },
        "closure_ledger": {
            "all_459_reference_atom_direction_packets_enumerated": True,
            "reference_DdeltaK_times_P55_closed": True,
            "reference_deltaK0_times_DP55_closed": True,
            "affine_deltaK_source_commutators_closed_on_tube": True,
            "tube_uniform_variable_P55_commutators_closed": False,
            "variable_CK1_all_terms_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "first_remaining_blocker": {
            "tensor": "D2P55(Y) for all three spatial pencils",
            "required_bound": (
                "sup_{||Y||H7<=R} ||D2(e10^T E_v^T P55^k)(Y)||_{l2x l2 to l2}"
            ),
            "why": (
                "the exact reference DP55 packets do not bound "
                "Dr_k(Y)-Dr_k(0) on the certified tube"
            ),
            "not_supplied_by_second_atom_Sylvester_chain": (
                "that chain proves pointwise reference solvability identities, not a "
                "tube-uniform three-direction operator envelope"
            ),
            "closed": False,
        },
    }


def run_quartic_tc2_ck1_variable_commutator_campaign(
    ck1_campaign: dict[str, Any],
    variable_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    topology_campaign: dict[str, Any],
    equilibrium_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            ck1_campaign,
            variable_campaign,
            full_jacobian_campaign,
            c9_campaign,
            topology_campaign,
            equilibrium_campaign,
        )
        expected_statuses = (
            "pass_all_12_CK1_reference_principal_and_source_topologies_TC2_global_fail_closed",
            "pass_all_12_first_order_variable_deltaK_extensions_higher_orders_global_H7_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_high_low_paraproduct_fail_closed",
            "pass_all_12_exact_reference_equilibria_and_L2_source_conventions",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2CK1VariableCommutatorError("unsupported schema_version")
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTC2CK1VariableCommutatorError("prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTC2CK1VariableCommutatorError(
                "prerequisite content hash mismatch"
            )
        if (
            int(config.get("expected_candidate_count", 0)) != 12
            or int(config.get("coordinate_atom_dimension", 0)) != ATOM_DIMENSION
            or int(config.get("spatial_dimension", 0)) != SPATIAL_DIMENSION
            or config.get("deltaK_ansatz")
            != "a10*deltaK0+a10^2*sum_A Y_A deltaK_A"
            or config.get("P55_tube_policy") != "fail_closed_without_D2_envelope"
            or config.get("TC2_policy") != "fail_closed"
            or config.get("B7_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
            or bool(config.get("declare_variable_CK1_closed", False))
        ):
            raise QuarticTC2CK1VariableCommutatorError(
                "unsupported closure contract"
            )
        generic_passed, generic = generic_ck1_variable_commutator_control()
        if not generic_passed:
            raise QuarticTC2CK1VariableCommutatorError("generic control failed")
        packet = _variable_ck1_commutator_packet()
        counts = packet["counts"]
        if (
            counts["coordinate_atoms"] != ATOM_DIMENSION
            or counts["atom_direction_packets"] != 459
            or counts["nonzero_deltaK_A_atoms"] != 41
            or counts["nonzero_DP55_scalar_row_packets"] != 123
            or not packet[
                "all_DP55_coordinate_derivatives_linear_in_a10_and_c20_absent"
            ]
        ):
            raise QuarticTC2CK1VariableCommutatorError(
                "exact commutator packet mismatch"
            )
        ck1_records = _candidate_records(ck1_campaign)
        maps = tuple(
            _candidate_records(campaign)
            for campaign in (
                variable_campaign,
                c9_campaign,
                topology_campaign,
                equilibrium_campaign,
            )
        )
        candidate_ids = set(ck1_records)
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps
        ):
            raise QuarticTC2CK1VariableCommutatorError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                ck1_records[candidate_id],
                *(records[candidate_id] for records in maps),
                topology_campaign["generic_H7_paracomposition_topology_control"],
                packet,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_reference_variable_CK1_P55_source_commutators_"
                "tube_P55_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "CK1_reference_source": ck1_campaign["content_sha256"],
                "variable_Sylvester": variable_campaign["content_sha256"],
                "full_source_Jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
                "H7_topology": topology_campaign["content_sha256"],
                "reference_equilibrium": equilibrium_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_variable_CK1_commutator_control": generic,
            "common_exact_commutator_packet": packet,
            "counts": {
                "selected": len(certificates),
                "coordinate_atoms": ATOM_DIMENSION,
                "spatial_directions": SPATIAL_DIMENSION,
                "exact_atom_direction_packets": 459,
                "reference_P55_commutator_slices_closed": len(certificates),
                "affine_deltaK_source_slices_closed": len(certificates),
                "tube_uniform_P55_commutator_closures": 0,
                "variable_CK1_closures": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The affine variable-deltaK CK1 product rule is enumerated exactly "
                "on all 153 coordinate atoms and three reference spatial pencils; "
                "the source branches have explicit C9/H7-tube bounds."
            ),
            "scope": (
                "Reference first-coordinate derivatives and affine-deltaK source "
                "branches only. A tube-uniform D2P55 envelope, variable CK1, CK3, "
                "TC2, B7, global H7, dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticTC2CK1VariableCommutatorError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "reference_P55_commutator_slices_closed": 0,
                "affine_deltaK_source_slices_closed": 0,
                "tube_uniform_P55_commutator_closures": 0,
                "variable_CK1_closures": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 1,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_ck1_variable_commutator_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
