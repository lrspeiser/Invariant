from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from itertools import combinations
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .horndeski_principal import _first_order_generalized_pencil
from .quartic_first_order_reduction_campaign import (
    _extract_spatial_blocks,
    _full_first_order_pencil,
    _symbol_data,
)
from .quartic_tc2_d4_curl_companion_range_campaign import _symmetric_basis
from .quartic_tc2_d4_minimal_tc2_escape_campaign import _correction_basis
from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _directional_derivative,
    _inverse_series,
    _matrix_payload,
    _projector_series,
    _reference_and_first_jet_packet,
    _series_product,
    _series_transpose,
)
from .quartic_tc2_fourth_jet_parallel_kernel import _combine_directions, _direction_key
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import STATE_DIMENSION

SCHEMA = "sigma-quartic-tc2-d4-parity-cubic-generic-direction-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-parity-cubic-generic-direction-config-1.0"
JET_ORDER = 4
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_DIRECTIONAL_EVALUATIONS = 15
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
EXPECTED_COMPANION_SHA256 = "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"


class QuarticTC2D4ParityCubicGenericDirectionError(ValueError):
    """Raised when the generic-direction cubic escape audit is invalid."""


def _zero_series(rows: int, columns: int) -> list[sp.Matrix]:
    return [sp.zeros(rows, columns) for _ in range(JET_ORDER + 1)]


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4ParityCubicGenericDirectionError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _frames() -> list[dict[str, Any]]:
    raw = [
        (
            "xy_3_4_5",
            sp.Matrix(
                [
                    [sp.Rational(3, 5), sp.Rational(-4, 5), 0],
                    [sp.Rational(4, 5), sp.Rational(3, 5), 0],
                    [0, 0, 1],
                ]
            ),
        ),
        (
            "xz_3_4_5",
            sp.Matrix(
                [
                    [sp.Rational(3, 5), 0, sp.Rational(-4, 5)],
                    [0, 1, 0],
                    [sp.Rational(4, 5), 0, sp.Rational(3, 5)],
                ]
            ),
        ),
        (
            "xyz_1_2_2",
            sp.Matrix(
                [
                    [sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)],
                    [sp.Rational(2, 3), sp.Rational(-2, 3), sp.Rational(1, 3)],
                    [sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(-2, 3)],
                ]
            ).T,
        ),
    ]
    result = []
    for name, rotation in raw:
        if rotation.T * rotation != sp.eye(3) or rotation.det() != 1:
            raise QuarticTC2D4ParityCubicGenericDirectionError(
                f"invalid rational orthonormal frame: {name}"
            )
        direction = tuple(rotation[:, 0])
        result.append({"name": name, "rotation": rotation, "direction": direction})
    return result


def _state_rotation(spatial_rotation: sp.Matrix) -> tuple[sp.Matrix, sp.Matrix]:
    spacetime = sp.diag(1, 1, 1, 1)
    spacetime[1:4, 1:4] = spatial_rotation
    basis = _symmetric_basis()
    field = sp.zeros(11)
    for source, source_basis in enumerate(basis):
        rotated = spacetime * source_basis * spacetime.T
        for target, target_basis in enumerate(basis):
            field[target, source] = sp.trace(target_basis.T * rotated)
    field[10, 10] = 1
    original = sp.zeros(STATE_DIMENSION)
    original[0:11, 0:11] = field
    original[11:22, 11:22] = field
    for target in range(3):
        for source in range(3):
            original[
                22 + 11 * target : 33 + 11 * target,
                22 + 11 * source : 33 + 11 * source,
            ] = spatial_rotation[target, source] * field
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    ordered = original.extract(ordering, ordering)
    if ordered.T * ordered != sp.eye(STATE_DIMENSION):
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "generic state representation is not orthogonal"
        )
    return ordered, field


def _generic_directional_packet(
    jet_direction: dict[str, sp.Expr], frame: Mapping[str, Any]
) -> dict[str, Any]:
    reference = _reference_and_first_jet_packet()
    data = _symbol_data()
    xi = data["xi_lower"]
    frequency = list(frame["direction"])
    state_rotation, field_rotation = _state_rotation(frame["rotation"])
    jets = reference["jets"]
    jet_symbols = {str(jet): jet for jet in jets}
    alpha, c20 = data["alpha"], data["c20"]
    substitutions: dict[sp.Symbol, sp.Expr] = {
        **{jet: 0 for jet in jets},
        data["m2"]: 1,
        xi[1]: frequency[0],
        xi[2]: frequency[1],
        xi[3]: frequency[2],
    }
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    coefficient_b = sum((frequency[index] * b_blocks[index] for index in range(3)), sp.zeros(11))
    coefficient_c_flux = [
        sum(
            (frequency[left] * c_blocks[left][right] for left in range(3)),
            sp.zeros(11),
        )
        for right in range(3)
    ]
    mass: list[sp.Matrix] = []
    evolution: list[sp.Matrix] = []
    for order in range(JET_ORDER + 1):
        if order == 0:
            extra = {alpha: 0, c20: 0}
            a_order = coefficient_a.subs({**substitutions, **extra})
            b_order = coefficient_b.subs({**substitutions, **extra})
            c_order = [matrix.subs({**substitutions, **extra}) for matrix in coefficient_c_flux]
        else:
            scale = sp.Rational(1, math.factorial(order))
            a_order = scale * _directional_derivative(
                coefficient_a, jet_direction, jet_symbols, substitutions, order
            )
            b_order = scale * _directional_derivative(
                coefficient_b, jet_direction, jet_symbols, substitutions, order
            )
            c_order = [
                scale
                * _directional_derivative(matrix, jet_direction, jet_symbols, substitutions, order)
                for matrix in coefficient_c_flux
            ]
        mass_order, evolution_order = _full_first_order_pencil(a_order, b_order, c_order, frequency)
        mass.append(mass_order)
        evolution.append(evolution_order)
    physical_original = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    physical_original[0] = mass[0].inv() * evolution[0]
    for order in range(1, JET_ORDER + 1):
        physical_original[order] = (
            mass[0].inv()
            * (
                evolution[order]
                - sum(
                    (
                        mass[index] * physical_original[order - index]
                        for index in range(1, order + 1)
                    ),
                    sp.zeros(STATE_DIMENSION),
                )
            )
        ).applyfunc(sp.factor)
    physical_coordinate = [
        matrix.extract(ordering, ordering).applyfunc(sp.factor) for matrix in physical_original
    ]
    physical = [
        (state_rotation.T * matrix * state_rotation).applyfunc(sp.factor)
        for matrix in physical_coordinate
    ]
    if physical[0] != reference["physical0"]:
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "generic direction failed isotropic reference rotation"
        )

    coupling = [matrix[33:55, 0:33] for matrix in physical]
    companion = [matrix[33:55, 33:55] for matrix in physical]
    nonzero_spectrum = (
        sp.Integer(1),
        sp.Integer(-1),
        sp.Rational(1, 2),
        sp.Rational(-1, 2),
        sp.Rational(1, 3),
        sp.Rational(-1, 3),
    )
    projectors = {
        eigenvalue: _projector_series(companion, eigenvalue, nonzero_spectrum)
        for eigenvalue in nonzero_spectrum
    }
    action = _first_order_generalized_pencil(data["action_symbol"], xi[0])
    action_a: list[sp.Matrix] = []
    action_b: list[sp.Matrix] = []
    for order in range(JET_ORDER + 1):
        if order == 0:
            extra = {alpha: 0, c20: 0}
            action_a.append(action["A"].subs({**substitutions, **extra}))
            action_b.append(action["B"].subs({**substitutions, **extra}))
        else:
            scale = sp.Rational(1, math.factorial(order))
            action_a.append(
                scale
                * _directional_derivative(
                    action["A"], jet_direction, jet_symbols, substitutions, order
                )
            )
            action_b.append(
                scale
                * _directional_derivative(
                    action["B"], jet_direction, jet_symbols, substitutions, order
                )
            )
    h_coordinate = [
        b.row_join(a).col_join(a.row_join(sp.zeros(11)))
        for a, b in zip(action_a, action_b, strict=True)
    ]
    action_rotation = sp.diag(field_rotation, field_rotation)
    h = [
        (action_rotation.T * matrix * action_rotation).applyfunc(sp.factor)
        for matrix in h_coordinate
    ]
    companion_energy = _zero_series(22, 22)
    identity22 = sp.eye(22)
    for eigenvalue, projector in projectors.items():
        metric = [
            h[order]
            if eigenvalue == 1
            else -h[order]
            if eigenvalue == -1
            else identity22
            if order == 0
            else sp.zeros(22)
            for order in range(JET_ORDER + 1)
        ]
        term = _series_product(_series_product(_series_transpose(projector), metric), projector)
        companion_energy = [
            (left + right).applyfunc(sp.factor)
            for left, right in zip(companion_energy, term, strict=True)
        ]
    inverse = _inverse_series(companion)
    cross = _series_product(_series_product(_series_transpose(coupling), companion_energy), inverse)
    energy = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    energy[0][0:33, 0:33] = sp.eye(33)
    for order in range(JET_ORDER + 1):
        energy[order][0:33, 33:55] = cross[order]
        energy[order][33:55, 0:33] = cross[order].T
        energy[order][33:55, 33:55] = companion_energy[order]
        energy[order] = energy[order].applyfunc(sp.factor)
    if energy[0] != reference["energy0"]:
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "generic direction failed energy reference rotation"
        )

    q = sp.zeros(11)
    q[0, 10], q[4, 10], q[10, 7], q[10, 9] = 2, -8, 2, 2
    embedded_q = sp.zeros(STATE_DIMENSION, 11)
    embedded_q[33:44, :] = q
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1
    block = [
        (alpha * matrix * embedded_q[:, 10] * high.T).applyfunc(sp.factor) for matrix in physical
    ]
    if block[0] != alpha * reference["block0"]:
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "generic registered TC2 block failed rotation"
        )
    skew = _series_product(energy, block)
    skew_transpose = _series_product(_series_transpose(block), energy)
    skew = [
        (left - right).applyfunc(sp.factor)
        for left, right in zip(skew, skew_transpose, strict=True)
    ]

    spectrum = tuple(reference["projectors"])
    delta = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    delta[0] = alpha * reference["delta0"]
    order_records = []
    for order in range(1, JET_ORDER + 1):
        rhs = (
            skew[order]
            + sum(
                (
                    delta[index] * physical[order - index]
                    - physical[order - index].T * delta[index]
                    for index in range(order)
                ),
                sp.zeros(STATE_DIMENSION),
            )
        ).applyfunc(sp.factor)
        compressions = {
            eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
            for eigenvalue, projector in reference["projectors"].items()
        }
        solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
        if solvable:
            for left in spectrum:
                for right in spectrum:
                    if left != right:
                        delta[order] += (
                            reference["projectors"][left].T
                            * rhs
                            * reference["projectors"][right]
                            / (left - right)
                        )
            delta[order] = delta[order].applyfunc(sp.factor)
        residual_order = (
            delta[order] * physical[0] - physical[0].T * delta[order] + rhs
        ).applyfunc(sp.factor)
        order_records.append(
            {
                "order": order,
                "rhs": rhs,
                "solvable": solvable,
                "residual_zero": residual_order.is_zero_matrix,
            }
        )
        if not solvable:
            break
    return {
        "physical": physical,
        "energy": energy,
        "block": block,
        "orders": order_records,
        "state_rotation": state_rotation,
        "alpha": alpha,
        "c20": c20,
    }


def _directional_fourth_payload(
    jet_direction: dict[str, sp.Expr], frame: Mapping[str, Any]
) -> dict[str, sp.Matrix]:
    packet = _generic_directional_packet(jet_direction, frame)
    if len(packet["orders"]) != JET_ORDER or not all(
        row["solvable"] and row["residual_zero"] for row in packet["orders"][:3]
    ):
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "mandatory lower recurrence failed at generic direction"
        )
    multiplier = math.factorial(JET_ORDER)
    return {
        "D4P55": (multiplier * packet["physical"][4]).applyfunc(sp.factor),
        "D4K55": (multiplier * packet["energy"][4]).applyfunc(sp.factor),
        "D4TC2": (multiplier * packet["block"][4]).applyfunc(sp.factor),
        "fourth_Sylvester_RHS": (multiplier * packet["orders"][3]["rhs"]).applyfunc(sp.factor),
    }


def _polarized_payload(
    frame: Mapping[str, Any], fourth_campaign: Mapping[str, Any]
) -> tuple[dict[str, sp.Matrix], int]:
    active = _active_directions()
    selector = fourth_campaign["selector"]
    basis_directions = [active[position] for position in selector["active_positions"]]
    directions = tuple(basis_directions[index]["direction"] for index in ACTIVE_INDICES)
    weights: dict[tuple[tuple[str, str], ...], int] = defaultdict(int)
    combined: dict[tuple[tuple[str, str], ...], dict[str, sp.Expr]] = {}
    for subset_size in range(1, JET_ORDER + 1):
        sign = -1 if (JET_ORDER - subset_size) % 2 else 1
        for subset in combinations(range(JET_ORDER), subset_size):
            direction = _combine_directions(directions, subset)
            key = _direction_key(direction)
            weights[key] += sign
            combined[key] = direction
    keys = sorted(key for key, weight in weights.items() if weight)
    payloads = [(weights[key], _directional_fourth_payload(combined[key], frame)) for key in keys]
    result = {
        name: (
            sum(
                (weight * payload[name] for weight, payload in payloads),
                sp.zeros(*payloads[0][1][name].shape),
            )
            / math.factorial(JET_ORDER)
        ).applyfunc(sp.factor)
        for name in payloads[0][1]
    }
    return result, len(payloads)


def _solve(rhs: sp.Matrix) -> tuple[bool, dict[str, Any]]:
    reference = _reference_and_first_jet_packet()
    compressions = {
        eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
    }
    solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
    nonzero = {
        str(eigenvalue): {
            "rank": matrix.rank(),
            "nonzero_entries": sum(value != 0 for value in matrix),
            "sha256": _content_hash(_matrix_payload(matrix)),
        }
        for eigenvalue, matrix in compressions.items()
        if not matrix.is_zero_matrix
    }
    return solvable, nonzero


def _audit_frame(
    frame: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    payload, directional_evaluations = _polarized_payload(frame, fourth_campaign)
    if directional_evaluations != EXPECTED_DIRECTIONAL_EVALUATIONS:
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "generic polarization evaluation count mismatch"
        )
    direction = list(frame["direction"])
    state_rotation, _ = _state_rotation(frame["rotation"])
    basis = _correction_basis()
    direction_1 = basis["block"]
    output = direction_1[:, 21]
    direction_2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    global_correction = (
        direction[0] ** 2 * (direction[0] * direction_1 + direction[1] * direction_2)
    ).applyfunc(sp.factor)
    correction = (state_rotation.T * global_correction * state_rotation).applyfunc(sp.factor)
    reference = _reference_and_first_jet_packet()
    correction_skew = (
        reference["energy0"] * correction - correction.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    rows = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                alpha: sp.sympify(candidate["a10"]),
                c20: sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        corrected = (
            candidate_rhs + sp.sympify(candidate["eta_unique_tuning"]) * correction_skew
        ).applyfunc(sp.factor)
        solvable, nonzero = _solve(corrected)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "a10": candidate["a10"],
                "c20": candidate["c20"],
                "eta": candidate["eta_unique_tuning"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
    compatible = sum(row["D4_Sylvester_solvable"] for row in rows)
    return {
        "frame_name": frame["name"],
        "direction": [str(value) for value in direction],
        "unit_norm": sum(value**2 for value in direction) == 1,
        "all_three_components_nonzero": all(value != 0 for value in direction),
        "frame_sha256": _content_hash(_matrix_payload(frame["rotation"])),
        "directional_evaluations": directional_evaluations,
        "all_seven_eigenspaces_checked_per_candidate": True,
        "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
        "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
        "cubic_correction_block_rank": correction.rank(),
        "cubic_correction_block_nonzero_entries": sum(value != 0 for value in correction),
        "cubic_correction_block_sha256": _content_hash(_matrix_payload(correction)),
        "cubic_correction_skew_rank": correction_skew.rank(),
        "candidate_compatibilities": compatible,
        "candidate_obstructions": EXPECTED_CANDIDATES - compatible,
        "candidate_records": rows,
    }


def _exact_campaign(
    minimal: Mapping[str, Any],
    cubic: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        cubic.get("claims", {}).get("all_12_axis2_D4_compatibilities_proved_for_cubic_symbol")
        is not True
        or cubic.get("claims", {}).get("generic_direction_D4_compatibility_proved") is not False
    ):
        raise QuarticTC2D4ParityCubicGenericDirectionError(
            "cubic predecessor generic-direction contract mismatch"
        )
    declared_frames = _frames()
    records = []
    stop_reason = "finite_basis_complete"
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        for frame in declared_frames:
            record = _audit_frame(frame, minimal, fourth_campaign)
            records.append(record)
            if record["candidate_obstructions"]:
                stop_reason = "first_exact_generic_direction_obstruction"
                break
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    evaluated_candidates = sum(len(record["candidate_records"]) for record in records)
    compatibilities = sum(record["candidate_compatibilities"] for record in records)
    obstructions = sum(record["candidate_obstructions"] for record in records)
    return {
        "selector": {
            "declared_rational_frame_names": [frame["name"] for frame in declared_frames],
            "declared_frame_count": len(declared_frames),
            "evaluation_order_deterministic": True,
            "stop_on_first_exact_obstruction": True,
            "frames_evaluated": len(records),
            "frames_unevaluated_after_stop": len(declared_frames) - len(records),
            "stop_reason": stop_reason,
            "not_an_interpolation_basis_for_the_full_sphere": True,
        },
        "direction_records": records,
        "result": {
            "candidate_direction_systems_evaluated": evaluated_candidates,
            "candidate_direction_compatibilities": compatibilities,
            "candidate_direction_obstructions": obstructions,
            "all_evaluated_generic_directions_compatible": obstructions == 0,
            "cubic_escape_all_direction_completion_rejected": obstructions > 0,
            "full_generic_direction_sphere_classified": False,
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2D4ParityCubicGenericDirectionError("generic direction config mismatch")
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    minimal = _load_bound(root, config["minimal_escape"])
    cubic = _load_bound(root, config["parity_cubic_escape"])
    fourth_campaign = _load_bound(root, config["fourth_campaign"])
    if (
        minimal.get("status") != "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only"
        or cubic.get("status")
        != "pass_exact_minimal_parity_preserving_cubic_angular_two_axis_escape"
        or fourth_campaign.get("status")
        != "pass_exact_fourth_jet_minimal_selector_manifest_no_evaluations_tube_fail_closed"
    ):
        raise QuarticTC2D4ParityCubicGenericDirectionError("predecessor status mismatch")
    exact = _exact_campaign(minimal, cubic, fourth_campaign)
    result = exact["result"]
    body = {
        "schema_version": SCHEMA,
        "status": (
            "pass_exact_generic_direction_obstruction_of_parity_cubic_escape"
            if result["candidate_direction_obstructions"]
            else "pass_exact_bounded_generic_direction_slice_no_obstruction"
        ),
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "minimal_escape",
                "parity_cubic_escape",
                "fourth_campaign",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "frequency_selector": "ordered_three_frame_rational_generic_direction_slice",
        },
        "exact_generic_direction_audit": exact,
        "counts": {
            "declared_rational_frames": exact["selector"]["declared_frame_count"],
            "frames_evaluated": exact["selector"]["frames_evaluated"],
            "frames_unevaluated_after_stop": exact["selector"]["frames_unevaluated_after_stop"],
            "directional_recurrence_evaluations": sum(
                record["directional_evaluations"] for record in exact["direction_records"]
            ),
            "candidate_direction_systems_evaluated": result[
                "candidate_direction_systems_evaluated"
            ],
            "candidate_direction_compatibilities": result["candidate_direction_compatibilities"],
            "candidate_direction_obstructions": result["candidate_direction_obstructions"],
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "infer_sphere_completion_from_finite_slice": {
                "full_generic_direction_sphere_classified": False,
                "rejected": True,
            },
            "continue_after_first_exact_obstruction": {
                "stop_on_first_exact_obstruction": True,
                "rejected": True,
            },
            "skip_lower_recurrence": {
                "orders_1_through_3_mandatory": True,
                "rejected": True,
            },
            "check_zero_speed_only": {
                "all_seven_eigenspaces_checked": True,
                "rejected": True,
            },
            "claim_local_covariant_origin": {
                "pseudodifferential_angular_predecessor": True,
                "rejected": True,
            },
            "promote_generic_obstruction_to_global_TC2": {
                "remaining_D4_selector_closed": False,
                "rejected": True,
            },
        },
        "claims": {
            "exact_rational_generic_direction_D4_recurrence_evaluated": True,
            "parity_cubic_all_direction_completion_rejected": result[
                "cubic_escape_all_direction_completion_rejected"
            ],
            "full_generic_direction_sphere_classified": False,
            "generic_direction_D4_compatibility_proved": False,
            "local_differential_operator_origin_proved": False,
            "covariant_action_origin_proved": False,
            "corrected_candidate_family_registered": False,
            "remaining_D4_selector_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "next_gate": (
            "Abandon the scalar-multiplied parity-cubic C12 escape after its first exact "
            "generic-direction obstruction. Seek a direction-dependent matrix completion with "
            "additional curl channels or a covariant local operator, then re-evaluate generic "
            "directions before any remaining-D4 or tube promotion."
        ),
        "scope": (
            "Exact full order-one-through-four D4 Sylvester recurrence on a deterministic "
            "rational generic-direction slice for the parity-cubic angular escape. All seven "
            "equal-eigenspace conditions are checked for every candidate and evaluation stops "
            "on the first exact obstruction. The finite slice is not an interpolation theorem "
            "for the direction sphere. Full D4, tube, CK1, CK3, TC2, B7, global-H7, and lifespan "
            "remain fail-closed."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4ParityCubicGenericDirectionError("content identity mismatch")
    exact = document.get("exact_generic_direction_audit", {})
    selector = exact.get("selector", {})
    result = exact.get("result", {})
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    if (
        document.get("status") != "pass_exact_generic_direction_obstruction_of_parity_cubic_escape"
        or selector.get("stop_reason") != "first_exact_generic_direction_obstruction"
        or selector.get("not_an_interpolation_basis_for_the_full_sphere") is not True
        or counts.get("declared_rational_frames") != 3
        or counts.get("frames_evaluated") != 1
        or counts.get("frames_unevaluated_after_stop") != 2
        or counts.get("directional_recurrence_evaluations") != 15
        or counts.get("candidate_direction_systems_evaluated") != 12
        or counts.get("candidate_direction_compatibilities") != 0
        or counts.get("candidate_direction_obstructions") != 12
        or counts.get("negative_controls") != 6
        or counts.get("inferred_global_passes") != 0
        or result.get("cubic_escape_all_direction_completion_rejected") is not True
        or result.get("full_generic_direction_sphere_classified") is not False
        or len(exact.get("direction_records", [])) != 1
        or exact.get("direction_records", [{}])[0].get("direction") != ["3/5", "4/5", "0"]
        or exact.get("direction_records", [{}])[0].get("candidate_obstructions") != 12
        or exact.get("direction_records", [{}])[0].get(
            "all_seven_eigenspaces_checked_per_candidate"
        )
        is not True
        or any(
            row.get("D4_Sylvester_solvable") is not False
            or not row.get("nonzero_equal_eigenspace_compressions")
            for row in exact.get("direction_records", [{}])[0].get("candidate_records", [])
        )
        or claims.get("exact_rational_generic_direction_D4_recurrence_evaluated") is not True
        or claims.get("parity_cubic_all_direction_completion_rejected") is not True
        or any(
            claims.get(key) is not False
            for key in (
                "full_generic_direction_sphere_classified",
                "generic_direction_D4_compatibility_proved",
                "local_differential_operator_origin_proved",
                "covariant_action_origin_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
        or len(document.get("negative_controls", {})) != 6
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4ParityCubicGenericDirectionError("exact/fail-closed mismatch")


def run_campaign(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the cubic escape at generic directions.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_campaign(args.project_root, args.config, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "frames_evaluated": artifact["counts"]["frames_evaluated"],
                "candidate_obstructions": artifact["counts"]["candidate_direction_obstructions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
