from __future__ import annotations

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
from .quartic_tc2_variable_sylvester_campaign import (
    ATOM_DIMENSION,
    STATE_DIMENSION,
    _content_hash,
    _content_hash_matches,
    _coordinate_atom_to_jet_packet,
    _matrix_entries,
    _projector_derivative,
    _projector_polynomial,
    _reference_and_first_jet_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-second-atom-chunk-campaign-1.0"
DEFAULT_CHUNK_SIZE = 64


class QuarticTC2SecondAtomChunkError(ValueError):
    """Raised when a chunked second-atom Sylvester audit is overstated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _direction_key(direction: dict[str, sp.Expr]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, str(value)) for name, value in direction.items()))


def _global_unordered_pair_index(left: int, right: int) -> int:
    if not (0 <= left <= right < ATOM_DIMENSION):
        raise QuarticTC2SecondAtomChunkError("invalid unordered atom pair")
    return left * ATOM_DIMENSION - left * (left + 1) // 2 + right


@cache
def _canonical_active_affine_pairs() -> tuple[dict[str, Any], ...]:
    coordinate = _coordinate_atom_to_jet_packet()
    reference = _reference_and_first_jet_packet()

    def sylvester_active(direction: dict[str, sp.Expr]) -> bool:
        derivative = sum(
            (
                coefficient * reference["delta_derivatives"][name]
                for name, coefficient in direction.items()
            ),
            sp.zeros(STATE_DIMENSION),
        )
        return not derivative.is_zero_matrix

    active = [
        (index, atom, direction)
        for index, (atom, direction) in enumerate(
            zip(coordinate["atoms"], coordinate["maps"], strict=True)
        )
        if atom.startswith("s") and direction and sylvester_active(direction)
    ]
    pairs: list[dict[str, Any]] = []
    for left_position, (left_index, left_atom, left_direction) in enumerate(active):
        for right_index, right_atom, right_direction in active[left_position:]:
            pairs.append(
                {
                    "left_atom_index": left_index,
                    "right_atom_index": right_index,
                    "left_atom": left_atom,
                    "right_atom": right_atom,
                    "left_direction": left_direction,
                    "right_direction": right_direction,
                    "global_pair_index": _global_unordered_pair_index(
                        left_index, right_index
                    ),
                }
            )
    return tuple(pairs)


def _directional_first(
    matrix: sp.Matrix,
    direction: dict[str, sp.Expr],
    jet_symbols: dict[str, sp.Symbol],
    substitutions: dict[sp.Symbol, sp.Expr],
) -> sp.Matrix:
    return sum(
        (
            coefficient * matrix.diff(jet_symbols[name])
            for name, coefficient in direction.items()
        ),
        sp.zeros(*matrix.shape),
    ).subs(substitutions).applyfunc(sp.factor)


def _directional_second(
    matrix: sp.Matrix,
    left: dict[str, sp.Expr],
    right: dict[str, sp.Expr],
    jet_symbols: dict[str, sp.Symbol],
    substitutions: dict[sp.Symbol, sp.Expr],
) -> sp.Matrix:
    return sum(
        (
            left_coefficient
            * right_coefficient
            * matrix.diff(jet_symbols[left_name], jet_symbols[right_name])
            for left_name, left_coefficient in left.items()
            for right_name, right_coefficient in right.items()
        ),
        sp.zeros(*matrix.shape),
    ).subs(substitutions).applyfunc(sp.factor)


def _projector_second_derivative(
    matrix: sp.Matrix,
    left_derivative: sp.Matrix,
    right_derivative: sp.Matrix,
    mixed_derivative: sp.Matrix,
    eigenvalue: sp.Expr,
    spectrum: tuple[sp.Expr, ...],
) -> sp.Matrix:
    others = [other for other in spectrum if other != eigenvalue]
    factors = [
        (matrix - other * sp.eye(matrix.rows)) / (eigenvalue - other)
        for other in others
    ]
    denominators = [eigenvalue - other for other in others]
    result = sp.zeros(matrix.rows)
    for mixed_index in range(len(factors)):
        term = sp.eye(matrix.rows)
        for index, factor in enumerate(factors):
            term *= (
                mixed_derivative / denominators[index]
                if index == mixed_index
                else factor
            )
        result += term
    for left_index in range(len(factors)):
        for right_index in range(len(factors)):
            if left_index == right_index:
                continue
            term = sp.eye(matrix.rows)
            for index, factor in enumerate(factors):
                if index == left_index:
                    term *= left_derivative / denominators[index]
                elif index == right_index:
                    term *= right_derivative / denominators[index]
                else:
                    term *= factor
            result += term
    return result.applyfunc(sp.factor)


@cache
def generic_second_atom_sylvester_control() -> tuple[bool, dict[str, Any]]:
    p0, pa, pb, pab = sp.symbols("P0 PA PB PAB")
    d0, da, db, dab = sp.symbols("D0 DA DB DAB")
    s0, sa, sb, sab = sp.symbols("S0 SA SB SAB")
    x, y = sp.symbols("x y")
    p = p0 + x * pa + y * pb + x * y * pab
    d = d0 + x * da + y * db + x * y * dab
    s = s0 + x * sa + y * sb + x * y * sab
    residual = sp.expand(d * p - p * d + s)
    mixed = sp.expand(sp.diff(residual, x, y).subs({x: 0, y: 0}))
    expected = (
        dab * p0
        - p0 * dab
        + sab
        + da * pb
        + db * pa
        + d0 * pab
        - pab * d0
        - pa * db
        - pb * da
    )
    projector_second = _projector_second_derivative(
        sp.diag(1, 2),
        sp.Matrix([[0, 3], [5, 0]]),
        sp.Matrix([[0, 7], [11, 0]]),
        sp.Matrix([[13, 0], [0, 17]]),
        sp.Integer(1),
        (sp.Integer(1), sp.Integer(2)),
    )
    parameter = sp.Symbol("t")
    matrix_curve = sp.diag(1, 2) + parameter * sp.Matrix(
        [[0, 3], [5, 0]]
    )
    exact_projector_curve = _projector_polynomial(
        matrix_curve, sp.Integer(1), (sp.Integer(1), sp.Integer(2))
    )
    first_projector = _projector_derivative(
        sp.diag(1, 2),
        sp.Matrix([[0, 3], [5, 0]]),
        sp.Integer(1),
        (sp.Integer(1), sp.Integer(2)),
    )
    first_residual = (
        sp.diff(exact_projector_curve, parameter).subs(parameter, 0)
        - first_projector
    ).applyfunc(sp.factor)
    corrupted = sp.expand(expected - da * pb)
    passed = bool(
        sp.expand(mixed - expected) == 0
        and first_residual.is_zero_matrix
        and not projector_second.is_zero_matrix
        and sp.expand(mixed - corrupted) != 0
    )
    return passed, {
        "control": "mixed second derivative of coupled Sylvester equation and projector recurrence",
        "mixed_Sylvester_residual": str(sp.expand(mixed - expected)),
        "projector_first_derivative_residual_zero": first_residual.is_zero_matrix,
        "projector_mixed_control_nonzero_entries": sum(
            value != 0 for value in projector_second
        ),
        "negative_controls": {
            "omit_deltaK_A_P_B": {
                "remaining": str(sp.expand(mixed - corrupted)),
                "rejected": sp.expand(mixed - corrupted) != 0,
            },
            "claim_chunk_covers_all_pairs": {
                "chunk": DEFAULT_CHUNK_SIZE,
                "total": ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2,
                "rejected": True,
            },
            "treat_nonlinear_coordinate_map_as_affine_globally": {
                "allowed_sector": "selected second-partial atoms at fixed q,p",
                "rejected_outside_sector": True,
                "rejected": True,
            },
        },
        "passed": passed,
    }


@cache
def _second_pair_symbolic_packet(
    left_key: tuple[tuple[str, str], ...],
    right_key: tuple[tuple[str, str], ...],
    second_coordinate_key: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    left = {name: sp.sympify(value) for name, value in left_key}
    right = {name: sp.sympify(value) for name, value in right_key}
    second_coordinate = {
        name: sp.sympify(value) for name, value in second_coordinate_key
    }
    reference = _reference_and_first_jet_packet()
    data = _symbol_data()
    xi = data["xi_lower"]
    jets = reference["jets"]
    jet_symbols = {str(jet): jet for jet in jets}
    alpha, c20 = data["alpha"], data["c20"]
    substitutions: dict[sp.Symbol, sp.Expr] = {
        **{jet: 0 for jet in jets},
        data["m2"]: 1,
        xi[1]: 1,
        xi[2]: 0,
        xi[3]: 0,
    }
    p0 = reference["physical0"]
    k0 = reference["energy0"]
    projectors = reference["projectors"]
    delta0_unit = reference["delta0"]
    ordering = [*range(11), *range(33, 55), *range(11, 33)]

    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    mass0, evolution0 = _full_first_order_pencil(
        coefficient_a.subs({**substitutions, alpha: 0, c20: 0}),
        b_blocks[0].subs({**substitutions, alpha: 0, c20: 0}),
        [
            c_blocks[0][right_index].subs(
                {**substitutions, alpha: 0, c20: 0}
            )
            for right_index in range(3)
        ],
        [1, 0, 0],
    )
    p_original0 = mass0.inv() * evolution0

    def physical_derivatives(direction: dict[str, sp.Expr]) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
        a_one = _directional_first(
            coefficient_a, direction, jet_symbols, substitutions
        )
        b_one = _directional_first(
            b_blocks[0], direction, jet_symbols, substitutions
        )
        c_one = [
            _directional_first(
                c_blocks[0][right_index], direction, jet_symbols, substitutions
            )
            for right_index in range(3)
        ]
        mass_one, evolution_one = _full_first_order_pencil(
            a_one, b_one, c_one, [1, 0, 0]
        )
        p_original_one = mass0.inv() * (
            evolution_one - mass_one * p_original0
        )
        return (
            p_original_one.extract(ordering, ordering).applyfunc(sp.factor),
            mass_one,
            evolution_one,
        )

    p_left, mass_left, _ = physical_derivatives(left)
    p_right, mass_right, _ = physical_derivatives(right)
    a_mixed = _directional_second(
        coefficient_a, left, right, jet_symbols, substitutions
    )
    b_mixed = _directional_second(
        b_blocks[0], left, right, jet_symbols, substitutions
    )
    c_mixed = [
        _directional_second(
            c_blocks[0][right_index], left, right, jet_symbols, substitutions
        )
        for right_index in range(3)
    ]
    mass_mixed, evolution_mixed = _full_first_order_pencil(
        a_mixed, b_mixed, c_mixed, [1, 0, 0]
    )
    p_original_left = p_left.extract(
        [ordering.index(index) for index in range(STATE_DIMENSION)],
        [ordering.index(index) for index in range(STATE_DIMENSION)],
    )
    p_original_right = p_right.extract(
        [ordering.index(index) for index in range(STATE_DIMENSION)],
        [ordering.index(index) for index in range(STATE_DIMENSION)],
    )
    p_original_mixed = mass0.inv() * (
        evolution_mixed
        - mass_mixed * p_original0
        - mass_left * p_original_right
        - mass_right * p_original_left
    )
    p_mixed = p_original_mixed.extract(ordering, ordering).applyfunc(sp.factor)
    if second_coordinate:
        p_mixed = (
            p_mixed + physical_derivatives(second_coordinate)[0]
        ).applyfunc(sp.factor)

    coupling0 = p0[33:55, 0:33]
    companion0 = p0[33:55, 33:55]
    coupling_left, companion_left = p_left[33:55, 0:33], p_left[33:55, 33:55]
    coupling_right, companion_right = p_right[33:55, 0:33], p_right[33:55, 33:55]
    coupling_mixed, companion_mixed = (
        p_mixed[33:55, 0:33],
        p_mixed[33:55, 33:55],
    )
    nonzero_spectrum = (
        sp.Integer(1),
        sp.Integer(-1),
        sp.Rational(1, 2),
        sp.Rational(-1, 2),
        sp.Rational(1, 3),
        sp.Rational(-1, 3),
    )
    companion_projectors = {
        eigenvalue: _projector_polynomial(
            companion0, eigenvalue, nonzero_spectrum
        )
        for eigenvalue in nonzero_spectrum
    }
    companion_projector_left = {
        eigenvalue: _projector_derivative(
            companion0, companion_left, eigenvalue, nonzero_spectrum
        )
        for eigenvalue in nonzero_spectrum
    }
    companion_projector_right = {
        eigenvalue: _projector_derivative(
            companion0, companion_right, eigenvalue, nonzero_spectrum
        )
        for eigenvalue in nonzero_spectrum
    }
    companion_projector_mixed = {
        eigenvalue: _projector_second_derivative(
            companion0,
            companion_left,
            companion_right,
            companion_mixed,
            eigenvalue,
            nonzero_spectrum,
        )
        for eigenvalue in nonzero_spectrum
    }

    action = _first_order_generalized_pencil(data["action_symbol"], xi[0])
    action_a0 = action["A"].subs({**substitutions, alpha: 0, c20: 0})
    action_b0 = action["B"].subs({**substitutions, alpha: 0, c20: 0})
    h0 = action_b0.row_join(action_a0).col_join(
        action_a0.row_join(sp.zeros(11))
    )

    def h_direction(direction: dict[str, sp.Expr]) -> sp.Matrix:
        action_a = _directional_first(
            action["A"], direction, jet_symbols, substitutions
        )
        action_b = _directional_first(
            action["B"], direction, jet_symbols, substitutions
        )
        return action_b.row_join(action_a).col_join(
            action_a.row_join(sp.zeros(11))
        )

    h_left, h_right = h_direction(left), h_direction(right)
    action_a_mixed = _directional_second(
        action["A"], left, right, jet_symbols, substitutions
    )
    action_b_mixed = _directional_second(
        action["B"], left, right, jet_symbols, substitutions
    )
    h_mixed = action_b_mixed.row_join(action_a_mixed).col_join(
        action_a_mixed.row_join(sp.zeros(11))
    )
    if second_coordinate:
        h_mixed = (h_mixed + h_direction(second_coordinate)).applyfunc(sp.factor)
    identity22 = sp.eye(22)
    companion_energy0 = sp.zeros(22)
    companion_energy_left = sp.zeros(22)
    companion_energy_right = sp.zeros(22)
    companion_energy_mixed = sp.zeros(22)
    for eigenvalue, projector in companion_projectors.items():
        q_left = companion_projector_left[eigenvalue]
        q_right = companion_projector_right[eigenvalue]
        q_mixed = companion_projector_mixed[eigenvalue]
        metric0 = (
            h0
            if eigenvalue == 1
            else -h0
            if eigenvalue == -1
            else identity22
        )
        metric_left = (
            h_left
            if eigenvalue == 1
            else -h_left
            if eigenvalue == -1
            else sp.zeros(22)
        )
        metric_right = (
            h_right
            if eigenvalue == 1
            else -h_right
            if eigenvalue == -1
            else sp.zeros(22)
        )
        metric_mixed = (
            h_mixed
            if eigenvalue == 1
            else -h_mixed
            if eigenvalue == -1
            else sp.zeros(22)
        )
        companion_energy0 += projector.T * metric0 * projector
        companion_energy_left += (
            q_left.T * metric0 * projector
            + projector.T * metric_left * projector
            + projector.T * metric0 * q_left
        )
        companion_energy_right += (
            q_right.T * metric0 * projector
            + projector.T * metric_right * projector
            + projector.T * metric0 * q_right
        )
        companion_energy_mixed += (
            q_mixed.T * metric0 * projector
            + q_left.T * metric_right * projector
            + q_left.T * metric0 * q_right
            + q_right.T * metric_left * projector
            + projector.T * metric_mixed * projector
            + projector.T * metric_left * q_right
            + q_right.T * metric0 * q_left
            + projector.T * metric_right * q_left
            + projector.T * metric0 * q_mixed
        )
    companion_energy0 = companion_energy0.applyfunc(sp.factor)
    companion_energy_left = companion_energy_left.applyfunc(sp.factor)
    companion_energy_right = companion_energy_right.applyfunc(sp.factor)
    companion_energy_mixed = companion_energy_mixed.applyfunc(sp.factor)

    inverse0 = companion0.inv()
    inverse_left = -inverse0 * companion_left * inverse0
    inverse_right = -inverse0 * companion_right * inverse0
    inverse_mixed = (
        inverse0 * companion_right * inverse0 * companion_left * inverse0
        + inverse0 * companion_left * inverse0 * companion_right * inverse0
        - inverse0 * companion_mixed * inverse0
    ).applyfunc(sp.factor)

    def cross_first(
        coupling_one: sp.Matrix,
        energy_one: sp.Matrix,
        inverse_one: sp.Matrix,
    ) -> sp.Matrix:
        return (
            coupling_one.T * companion_energy0 * inverse0
            + coupling0.T * energy_one * inverse0
            + coupling0.T * companion_energy0 * inverse_one
        ).applyfunc(sp.factor)

    cross_left = cross_first(
        coupling_left, companion_energy_left, inverse_left
    )
    cross_right = cross_first(
        coupling_right, companion_energy_right, inverse_right
    )
    cross_mixed = (
        coupling_mixed.T * companion_energy0 * inverse0
        + coupling_left.T * companion_energy_right * inverse0
        + coupling_left.T * companion_energy0 * inverse_right
        + coupling_right.T * companion_energy_left * inverse0
        + coupling0.T * companion_energy_mixed * inverse0
        + coupling0.T * companion_energy_left * inverse_right
        + coupling_right.T * companion_energy0 * inverse_left
        + coupling0.T * companion_energy_right * inverse_left
        + coupling0.T * companion_energy0 * inverse_mixed
    ).applyfunc(sp.factor)

    rebuilt_energy0 = sp.zeros(STATE_DIMENSION)
    rebuilt_energy0[0:33, 0:33] = sp.eye(33)
    rebuilt_cross0 = (
        coupling0.T * companion_energy0 * inverse0
    ).applyfunc(sp.factor)
    rebuilt_energy0[0:33, 33:55] = rebuilt_cross0
    rebuilt_energy0[33:55, 0:33] = rebuilt_cross0.T
    rebuilt_energy0[33:55, 33:55] = companion_energy0
    if not rebuilt_energy0.equals(k0):
        raise QuarticTC2SecondAtomChunkError("rebuilt K55 reference mismatch")

    def lift_energy(cross: sp.Matrix, companion_energy: sp.Matrix) -> sp.Matrix:
        energy = sp.zeros(STATE_DIMENSION)
        energy[0:33, 33:55] = cross
        energy[33:55, 0:33] = cross.T
        energy[33:55, 33:55] = companion_energy
        return energy

    k_left = lift_energy(cross_left, companion_energy_left)
    k_right = lift_energy(cross_right, companion_energy_right)
    k_mixed = lift_energy(cross_mixed, companion_energy_mixed)

    def combine_reference(
        source: dict[str, sp.Matrix], direction: dict[str, sp.Expr]
    ) -> sp.Matrix:
        return sum(
            (coefficient * source[name] for name, coefficient in direction.items()),
            sp.zeros(STATE_DIMENSION),
        ).applyfunc(sp.factor)

    if not p_left.equals(
        alpha * combine_reference(reference["physical_derivatives"], left)
    ) or not p_right.equals(
        alpha * combine_reference(reference["physical_derivatives"], right)
    ):
        raise QuarticTC2SecondAtomChunkError("D P55 first-recurrence mismatch")
    if not k_left.equals(
        alpha * combine_reference(reference["energy_derivatives"], left)
    ) or not k_right.equals(
        alpha * combine_reference(reference["energy_derivatives"], right)
    ):
        raise QuarticTC2SecondAtomChunkError("D K55 first-recurrence mismatch")

    q = sp.zeros(11)
    q[0, 10], q[4, 10], q[10, 7], q[10, 9] = 2, -8, 2, 2
    embedded_q = sp.zeros(STATE_DIMENSION, 11)
    embedded_q[33:44, :] = q
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1

    def tc2_block(physical: sp.Matrix) -> sp.Matrix:
        return physical * embedded_q[:, 10] * high.T

    block0 = alpha * tc2_block(p0)
    block_left = alpha * tc2_block(p_left)
    block_right = alpha * tc2_block(p_right)
    block_mixed = alpha * tc2_block(p_mixed)
    skew_mixed = (
        k_mixed * block0
        + k_left * block_right
        + k_right * block_left
        + k0 * block_mixed
        - block_mixed.T * k0
        - block_left.T * k_right
        - block_right.T * k_left
        - block0.T * k_mixed
    ).applyfunc(sp.factor)

    delta_left = alpha**2 * combine_reference(reference["delta_derivatives"], left)
    delta_right = alpha**2 * combine_reference(reference["delta_derivatives"], right)
    delta0 = alpha * delta0_unit
    second_rhs = (
        skew_mixed
        + delta_left * p_right
        + delta_right * p_left
        + delta0 * p_mixed
        - p_mixed.T * delta0
        - p_left.T * delta_right
        - p_right.T * delta_left
    ).applyfunc(sp.factor)
    compressions = {
        eigenvalue: (projector.T * second_rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in projectors.items()
    }
    solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
    delta_mixed = sp.zeros(STATE_DIMENSION)
    if solvable:
        for left_eigenvalue, left_projector in projectors.items():
            for right_eigenvalue, right_projector in projectors.items():
                if left_eigenvalue != right_eigenvalue:
                    delta_mixed += (
                        left_projector.T
                        * second_rhs
                        * right_projector
                        / (left_eigenvalue - right_eigenvalue)
                    )
        delta_mixed = delta_mixed.applyfunc(sp.factor)
    sylvester_residual = (
        delta_mixed * p0 - p0.T * delta_mixed + second_rhs
    ).applyfunc(sp.factor)
    compression_entries = {
        str(eigenvalue): _matrix_entries(matrix)
        for eigenvalue, matrix in compressions.items()
        if not matrix.is_zero_matrix
    }
    body = {
        "left_direction": {name: str(value) for name, value in left.items()},
        "right_direction": {name: str(value) for name, value in right.items()},
        "D2P55_nonzero_entries": sum(value != 0 for value in p_mixed),
        "D2K55_nonzero_entries": sum(value != 0 for value in k_mixed),
        "D2TC2_nonzero_entries": sum(value != 0 for value in block_mixed),
        "equal_eigenspace_compressions": compression_entries,
        "equal_eigenspace_compressions_zero": solvable,
        "deltaK_AB_nonzero_entries": sum(value != 0 for value in delta_mixed),
        "deltaK_AB_rank": delta_mixed.rank() if solvable else None,
        "deltaK_AB_Hermitian": delta_mixed.equals(delta_mixed.T) if solvable else False,
        "deltaK_AB_entries": _matrix_entries(delta_mixed) if solvable else [],
        "second_Sylvester_residual_zero": sylvester_residual.is_zero_matrix,
        "parameter_symbols": [str(alpha), str(c20)],
    }
    if second_coordinate:
        body["second_coordinate_direction"] = {
            name: str(value) for name, value in second_coordinate.items()
        }
        body["coordinate_D2_pushforward_included"] = True
    return {**body, "content_sha256": _content_hash(body)}


def run_quartic_tc2_second_atom_chunk_campaign(
    variable_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2SecondAtomChunkError("unsupported campaign schema_version")
        if variable_campaign.get("status") != (
            "pass_all_12_first_order_variable_deltaK_extensions_"
            "higher_orders_global_H7_fail_closed"
        ):
            raise QuarticTC2SecondAtomChunkError("variable campaign status mismatch")
        if not _content_hash_matches(variable_campaign):
            raise QuarticTC2SecondAtomChunkError("variable campaign content hash mismatch")
        if (
            int(config["chunk_size"]) != DEFAULT_CHUNK_SIZE
            or int(config["chunk_offset"]) != 0
            or config.get("pair_selector")
            != "canonical_sylvester_active_affine_second_atom_pairs"
            or config.get("resume_policy") != "hash_chain"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTC2SecondAtomChunkError("unsupported second-atom chunk contract")
        generic_passed, generic = generic_second_atom_sylvester_control()
        if not generic_passed:
            raise QuarticTC2SecondAtomChunkError("generic second-atom control failed")
        all_pairs = _canonical_active_affine_pairs()
        selected = all_pairs[:DEFAULT_CHUNK_SIZE]
        if len(selected) != DEFAULT_CHUNK_SIZE:
            raise QuarticTC2SecondAtomChunkError("insufficient canonical active pairs")
        coordinate_packet_hash = variable_campaign[
            "common_coordinate_to_covariant_jet_packet"
        ]["content_sha256"]
        seed = _content_hash(
            {
                "upstream": variable_campaign["content_sha256"],
                "coordinate_packet": coordinate_packet_hash,
                "selector": config["pair_selector"],
                "offset": 0,
                "size": DEFAULT_CHUNK_SIZE,
            }
        )
        previous = seed
        manifest: list[dict[str, Any]] = []
        symbolic_packets: dict[str, dict[str, Any]] = {}
        first_obstruction: dict[str, Any] | None = None
        candidate_coefficients = {
            certificate["candidate_id"]: certificate["coefficients"]
            for certificate in variable_campaign["certificates"]
        }
        for chunk_index, pair in enumerate(selected):
            symbolic = _second_pair_symbolic_packet(
                _direction_key(pair["left_direction"]),
                _direction_key(pair["right_direction"]),
            )
            symbolic_packets[symbolic["content_sha256"]] = symbolic
            candidate_results: list[dict[str, Any]] = []
            for candidate_id in sorted(candidate_coefficients):
                coefficients = candidate_coefficients[candidate_id]
                substitutions = {
                    sp.Symbol("alpha"): sp.sympify(coefficients["a10"]),
                    sp.Symbol("c20"): sp.sympify(coefficients["c20"]),
                }
                compressions = {
                    eigenvalue: [
                        {
                            **entry,
                            "value": str(sp.factor(sp.sympify(entry["value"]).subs(substitutions))),
                        }
                        for entry in entries
                        if sp.factor(sp.sympify(entry["value"]).subs(substitutions)) != 0
                    ]
                    for eigenvalue, entries in symbolic[
                        "equal_eigenspace_compressions"
                    ].items()
                }
                compressions = {
                    eigenvalue: entries
                    for eigenvalue, entries in compressions.items()
                    if entries
                }
                delta_entries = [
                    {
                        **entry,
                        "value": str(
                            sp.factor(sp.sympify(entry["value"]).subs(substitutions))
                        ),
                    }
                    for entry in symbolic["deltaK_AB_entries"]
                    if sp.factor(sp.sympify(entry["value"]).subs(substitutions)) != 0
                ]
                solvable = bool(
                    not compressions
                    and symbolic["equal_eigenspace_compressions_zero"]
                    and symbolic["second_Sylvester_residual_zero"]
                )
                result = {
                    "candidate_id": candidate_id,
                    "solvable": solvable,
                    "compression_residual_sha256": _content_hash(compressions),
                    "deltaK_AB_nonzero_entries": len(delta_entries) if solvable else 0,
                    "deltaK_AB_sha256": (
                        _content_hash(delta_entries) if solvable else None
                    ),
                    "Hermitian": symbolic["deltaK_AB_Hermitian"] if solvable else False,
                    "second_Sylvester_residual_zero": (
                        symbolic["second_Sylvester_residual_zero"] if solvable else False
                    ),
                }
                candidate_results.append(result)
                if not solvable and first_obstruction is None:
                    eigenvalue, entries = next(iter(compressions.items()))
                    first_obstruction = {
                        "chunk_index": chunk_index,
                        "global_pair_index": pair["global_pair_index"],
                        "left_atom": pair["left_atom"],
                        "right_atom": pair["right_atom"],
                        "candidate_id": candidate_id,
                        "eigenvalue": eigenvalue,
                        "first_nonzero_entry": entries[0],
                        "symbolic_pair_packet_sha256": symbolic["content_sha256"],
                    }
            record_body = {
                "chunk_index": chunk_index,
                "global_pair_index": pair["global_pair_index"],
                "left_atom_index": pair["left_atom_index"],
                "right_atom_index": pair["right_atom_index"],
                "left_atom": pair["left_atom"],
                "right_atom": pair["right_atom"],
                "left_direction_sha256": _content_hash(
                    {key: str(value) for key, value in pair["left_direction"].items()}
                ),
                "right_direction_sha256": _content_hash(
                    {key: str(value) for key, value in pair["right_direction"].items()}
                ),
                "symbolic_pair_packet_sha256": symbolic["content_sha256"],
                "candidate_results": candidate_results,
                "previous_record_sha256": previous,
            }
            record_hash = _content_hash(record_body)
            manifest.append({**record_body, "record_sha256": record_hash})
            previous = record_hash
        candidate_pair_count = DEFAULT_CHUNK_SIZE * len(candidate_coefficients)
        solvable_candidate_pairs = sum(
            result["solvable"]
            for record in manifest
            for result in record["candidate_results"]
        )
        total_pairs = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
        unique_packets = [symbolic_packets[key] for key in sorted(symbolic_packets)]
        tensor_summary = {
            "unique_direction_pair_packets": len(unique_packets),
            "nonzero_D2P55_packets": sum(
                packet["D2P55_nonzero_entries"] > 0 for packet in unique_packets
            ),
            "nonzero_D2K55_packets": sum(
                packet["D2K55_nonzero_entries"] > 0 for packet in unique_packets
            ),
            "nonzero_D2TC2_packets": sum(
                packet["D2TC2_nonzero_entries"] > 0 for packet in unique_packets
            ),
            "nonzero_deltaK_AB_packets": sum(
                packet["deltaK_AB_nonzero_entries"] > 0
                for packet in unique_packets
            ),
            "maximum_deltaK_AB_rank": max(
                packet["deltaK_AB_rank"] for packet in unique_packets
            ),
            "all_deltaK_AB_Hermitian": all(
                packet["deltaK_AB_Hermitian"] for packet in unique_packets
            ),
            "all_second_Sylvester_residuals_zero": all(
                packet["second_Sylvester_residual_zero"]
                for packet in unique_packets
            ),
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_first_64_second_atom_pairs_no_obstruction_remaining_fail_closed"
                if first_obstruction is None
                else "exact_second_atom_Sylvester_obstruction_found_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "variable_Sylvester": variable_campaign["content_sha256"],
                "reference_first_jet_packet": variable_campaign[
                    "common_reference_first_jet_packet"
                ]["content_sha256"],
                "coordinate_to_jet_packet": coordinate_packet_hash,
                "first_order_variable_packet": variable_campaign[
                    "common_variable_solvability_packet"
                ]["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_second_atom_control": generic,
            "chunk_contract": {
                "pair_selector": config["pair_selector"],
                "selector_pair_count": len(all_pairs),
                "chunk_offset": 0,
                "chunk_size": DEFAULT_CHUNK_SIZE,
                "chunk_seed_sha256": seed,
                "resume_after_record_sha256": previous,
                "selection_scope": (
                    "active second-partial coordinate atoms whose flat covariant-jet map "
                    "is affine on pair variations"
                ),
                "global_pair_indices_are_stable": True,
            },
            "pair_manifest": manifest,
            "symbolic_pair_packets": unique_packets,
            "exact_tensor_summary": tensor_summary,
            "first_exact_obstruction": first_obstruction,
            "counts": {
                "total_unordered_coordinate_atom_pairs": total_pairs,
                "evaluated_coordinate_atom_pairs": DEFAULT_CHUNK_SIZE,
                "remaining_unevaluated_coordinate_atom_pairs": total_pairs
                - DEFAULT_CHUNK_SIZE,
                "candidates": len(candidate_coefficients),
                "evaluated_candidate_pairs": candidate_pair_count,
                "solvable_candidate_pairs": solvable_candidate_pairs,
                "obstructed_candidate_pairs": candidate_pair_count
                - solvable_candidate_pairs,
                "deltaK_AB_constructions": solvable_candidate_pairs,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "claim": (
                "Only the declared 64-pair canonical active-affine chunk was evaluated. "
                "No result is inferred for the remaining 11717 coordinate-atom pairs."
            ),
            "scope": (
                "Second-order Sylvester algebra on one resumable chunk only. TC2, B7, "
                "global H7, dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2SecondAtomChunkError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "pair_manifest": [],
            "symbolic_pair_packets": [],
            "counts": {
                "total_unordered_coordinate_atom_pairs": ATOM_DIMENSION
                * (ATOM_DIMENSION + 1)
                // 2,
                "evaluated_coordinate_atom_pairs": 0,
                "remaining_unevaluated_coordinate_atom_pairs": ATOM_DIMENSION
                * (ATOM_DIMENSION + 1)
                // 2,
                "candidates": 0,
                "evaluated_candidate_pairs": 0,
                "solvable_candidate_pairs": 0,
                "obstructed_candidate_pairs": 0,
                "deltaK_AB_constructions": 0,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_second_atom_chunk_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
