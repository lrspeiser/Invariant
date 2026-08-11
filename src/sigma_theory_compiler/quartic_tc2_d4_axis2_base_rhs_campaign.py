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
from .quartic_tc2_d4_curl_companion_range_campaign import _axis_swap_1_2
from .quartic_tc2_d4_minimal_tc2_escape_campaign import _correction_basis
from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _directional_derivative,
    _inverse_series,
    _matrix_payload,
    _projector_series,
    _series_product,
    _series_transpose,
    _zero_series,
)
from .quartic_tc2_fourth_jet_parallel_kernel import (
    _combine_directions,
    _direction_key,
)
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import (
    STATE_DIMENSION,
    _reference_and_first_jet_packet,
)

SCHEMA = "sigma-quartic-tc2-d4-axis2-base-rhs-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-axis2-base-rhs-config-1.0"
JET_ORDER = 4
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_DIRECTIONAL_EVALUATIONS = 15
EXPECTED_COMPANION_BLOCK_SHA256 = "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
EXPECTED_COMPANION_COMPRESSION_SHA256 = (
    "def5dc985fa3356a9a21b2b06d4ebe0f0365058403e3e762eab161d7fb2822be"
)
ZERO_55_MATRIX_SHA256 = "d4d121c239c4905e1183840e887b102fd0a0c9dc588820df4b858291c70cb4ad"


class QuarticTC2D4Axis2BaseRHSError(ValueError):
    """Raised when the exact axis-two base-D4 contract is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4Axis2BaseRHSError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4Axis2BaseRHSError(f"bound input mismatch: {binding.get('path')}")
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4Axis2BaseRHSError("raw binding escaped project root or is absent")
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4Axis2BaseRHSError(f"raw binding mismatch: {binding.get('path')}")


def _sparse_matrix(matrix: sp.MatrixBase) -> list[dict[str, Any]]:
    return [
        {"row": row, "column": column, "value": str(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


def _axis2_reference() -> dict[str, Any]:
    reference1 = _reference_and_first_jet_packet()
    swap = _axis_swap_1_2()
    return {
        "swap": swap,
        "physical0": (swap * reference1["physical0"] * swap.T).applyfunc(sp.factor),
        "energy0": (swap * reference1["energy0"] * swap.T).applyfunc(sp.factor),
        "delta0": (swap * reference1["delta0"] * swap.T).applyfunc(sp.factor),
        "projectors": {
            eigenvalue: (swap * projector * swap.T).applyfunc(sp.factor)
            for eigenvalue, projector in reference1["projectors"].items()
        },
    }


def _axis2_directional_taylor_packet(
    direction: dict[str, sp.Expr], reference: Mapping[str, Any]
) -> dict[str, Any]:
    data = _symbol_data()
    xi = data["xi_lower"]
    reference1 = _reference_and_first_jet_packet()
    jets = reference1["jets"]
    jet_symbols = {str(jet): jet for jet in jets}
    alpha, c20 = data["alpha"], data["c20"]
    substitutions: dict[sp.Symbol, sp.Expr] = {
        **{jet: 0 for jet in jets},
        data["m2"]: 1,
        xi[1]: 0,
        xi[2]: 1,
        xi[3]: 0,
    }
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    mass: list[sp.Matrix] = []
    evolution: list[sp.Matrix] = []
    for order in range(JET_ORDER + 1):
        if order == 0:
            extra = {alpha: 0, c20: 0}
            a_order = coefficient_a.subs({**substitutions, **extra})
            b_order = b_blocks[1].subs({**substitutions, **extra})
            c_order = [matrix.subs({**substitutions, **extra}) for matrix in c_blocks[1]]
        else:
            scale = sp.Rational(1, math.factorial(order))
            a_order = scale * _directional_derivative(
                coefficient_a, direction, jet_symbols, substitutions, order
            )
            b_order = scale * _directional_derivative(
                b_blocks[1], direction, jet_symbols, substitutions, order
            )
            c_order = [
                scale
                * _directional_derivative(matrix, direction, jet_symbols, substitutions, order)
                for matrix in c_blocks[1]
            ]
        mass_order, evolution_order = _full_first_order_pencil(a_order, b_order, c_order, [0, 1, 0])
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
    swap = reference["swap"]
    physical = [(swap.T * matrix * swap).applyfunc(sp.factor) for matrix in physical_coordinate]
    if physical_coordinate[0] != reference["physical0"] or physical[0] != reference1["physical0"]:
        raise QuarticTC2D4Axis2BaseRHSError("direct and rotated axis-two reference P55 disagree")

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
                * _directional_derivative(action["A"], direction, jet_symbols, substitutions, order)
            )
            action_b.append(
                scale
                * _directional_derivative(action["B"], direction, jet_symbols, substitutions, order)
            )
    h_coordinate = [
        b.row_join(a).col_join(a.row_join(sp.zeros(11)))
        for a, b in zip(action_a, action_b, strict=True)
    ]
    field_rotation = swap[0:11, 0:11]
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
    if energy[0] != reference1["energy0"]:
        raise QuarticTC2D4Axis2BaseRHSError("direct and rotated axis-two reference K55 disagree")

    q = sp.zeros(11)
    q[0, 10], q[4, 10], q[10, 7], q[10, 9] = 2, -8, 2, 2
    embedded_q = sp.zeros(STATE_DIMENSION, 11)
    embedded_q[33:44, :] = q
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1
    block = [
        (alpha * matrix * embedded_q[:, 10] * high.T).applyfunc(sp.factor) for matrix in physical
    ]
    if block[0] != alpha * reference1["block0"]:
        raise QuarticTC2D4Axis2BaseRHSError(
            "axis-two base TC2 block does not rotate from the registered reference"
        )
    skew = _series_product(energy, block)
    skew_transpose = _series_product(_series_transpose(block), energy)
    skew = [
        (left - right).applyfunc(sp.factor)
        for left, right in zip(skew, skew_transpose, strict=True)
    ]

    spectrum = tuple(reference1["projectors"])
    delta = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    delta[0] = alpha * reference1["delta0"]
    order_records: list[dict[str, Any]] = []
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
            for eigenvalue, projector in reference1["projectors"].items()
        }
        solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
        if solvable:
            for left in spectrum:
                for right in spectrum:
                    if left != right:
                        delta[order] += (
                            reference1["projectors"][left].T
                            * rhs
                            * reference1["projectors"][right]
                            / (left - right)
                        )
            delta[order] = delta[order].applyfunc(sp.factor)
        residual = (delta[order] * physical[0] - physical[0].T * delta[order] + rhs).applyfunc(
            sp.factor
        )
        order_records.append(
            {
                "order": order,
                "rhs": rhs,
                "compressions": compressions,
                "solvable": solvable,
                "delta": delta[order],
                "residual_zero": residual.is_zero_matrix,
            }
        )
        if not solvable:
            break
    rotated_orders = [
        {
            **record,
            "rhs": (swap * record["rhs"] * swap.T).applyfunc(sp.factor),
            "compressions": {
                eigenvalue: (swap * matrix * swap.T).applyfunc(sp.factor)
                for eigenvalue, matrix in record["compressions"].items()
            },
            "delta": (swap * record["delta"] * swap.T).applyfunc(sp.factor),
        }
        for record in order_records
    ]
    return {
        "physical": physical_coordinate,
        "energy": [(swap * matrix * swap.T).applyfunc(sp.factor) for matrix in energy],
        "block": [(swap * matrix * swap.T).applyfunc(sp.factor) for matrix in block],
        "orders": rotated_orders,
        "alpha": alpha,
        "c20": c20,
    }


def _axis2_directional_fourth_payload(
    direction: dict[str, sp.Expr], reference: Mapping[str, Any]
) -> dict[str, sp.Matrix]:
    packet = _axis2_directional_taylor_packet(direction, reference)
    lower_orders = packet["orders"][: JET_ORDER - 1]
    if (
        len(packet["orders"]) != JET_ORDER
        or len(lower_orders) != 3
        or not all(order["solvable"] and order["residual_zero"] for order in lower_orders)
    ):
        raise QuarticTC2D4Axis2BaseRHSError(
            "axis-two directional recurrence failed in mandatory orders one through three"
        )
    multiplier = sp.Integer(math.factorial(JET_ORDER))
    return {
        "D4P55": (multiplier * packet["physical"][4]).applyfunc(sp.factor),
        "D4K55": (multiplier * packet["energy"][4]).applyfunc(sp.factor),
        "D4TC2": (multiplier * packet["block"][4]).applyfunc(sp.factor),
        "fourth_Sylvester_RHS": (multiplier * packet["orders"][3]["rhs"]).applyfunc(sp.factor),
    }


def _polarized_axis2_payload(
    active_indices: tuple[int, int, int, int],
    basis_directions: list[dict[str, Any]],
    reference: Mapping[str, Any],
) -> tuple[dict[str, sp.Matrix], int]:
    directions = tuple(basis_directions[index]["direction"] for index in active_indices)
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
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payloads = [
            (weights[key], _axis2_directional_fourth_payload(combined[key], reference))
            for key in keys
        ]
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    if not payloads:
        raise QuarticTC2D4Axis2BaseRHSError("empty axis-two polarization payload")
    result: dict[str, sp.Matrix] = {}
    for name in payloads[0][1]:
        result[name] = (
            sum(
                (weight * payload[name] for weight, payload in payloads),
                sp.zeros(*payloads[0][1][name].shape),
            )
            / math.factorial(JET_ORDER)
        ).applyfunc(sp.factor)
    return result, len(payloads)


def _solve_axis2_sylvester(
    rhs: sp.Matrix, reference: Mapping[str, Any]
) -> tuple[bool, sp.Matrix, dict[str, Any]]:
    projectors = reference["projectors"]
    compressions = {
        eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in projectors.items()
    }
    solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
    delta = sp.zeros(STATE_DIMENSION)
    if solvable:
        for left, left_projector in projectors.items():
            for right, right_projector in projectors.items():
                if left != right:
                    delta += left_projector.T * rhs * right_projector / (left - right)
        delta = delta.applyfunc(sp.factor)
    residual = (delta * reference["physical0"] - reference["physical0"].T * delta + rhs).applyfunc(
        sp.factor
    )
    return (
        solvable,
        delta,
        {
            "compressions": compressions,
            "residual_zero": residual.is_zero_matrix,
        },
    )


def _candidate_rows(
    rhs: sp.Matrix,
    correction_skew: sp.Matrix,
    target: sp.Matrix,
    reference: Mapping[str, Any],
    candidates: Mapping[str, Any],
    companion: Mapping[str, Any],
) -> list[dict[str, Any]]:
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    eta_by_candidate = {
        row["candidate_id"]: sp.sympify(row["eta"])
        for row in companion["exact_companion_audit"]["candidate_companion_witnesses"]
    }
    rows: list[dict[str, Any]] = []
    for certificate in sorted(
        candidates.get("certificates", []), key=lambda row: row["candidate_id"]
    ):
        candidate_id = certificate["candidate_id"]
        alpha_value = sp.sympify(certificate["coefficients"]["a10"])
        c20_value = sp.sympify(certificate["coefficients"]["c20"])
        eta = eta_by_candidate[candidate_id]
        base_rhs = rhs.subs({alpha: alpha_value, c20: c20_value}).applyfunc(sp.factor)
        base_zero = (
            reference["projectors"][sp.S.Zero].T * base_rhs * reference["projectors"][sp.S.Zero]
        ).applyfunc(sp.factor)
        required = (-eta * target).applyfunc(sp.factor)
        corrected_rhs = (base_rhs + eta * correction_skew).applyfunc(sp.factor)
        solvable, delta, audit = _solve_axis2_sylvester(corrected_rhs, reference)
        wrong_sign_rhs = (base_rhs - eta * correction_skew).applyfunc(sp.factor)
        wrong_sign_solvable, _, _ = _solve_axis2_sylvester(wrong_sign_rhs, reference)
        nonzero = {
            str(eigenvalue): {
                "rank": matrix.rank(),
                "nonzero_entries": sum(value != 0 for value in matrix),
                "sha256": _content_hash(_matrix_payload(matrix)),
            }
            for eigenvalue, matrix in audit["compressions"].items()
            if not matrix.is_zero_matrix
        }
        rows.append(
            {
                "candidate_id": candidate_id,
                "a10": str(alpha_value),
                "c20": str(c20_value),
                "eta": str(eta),
                "base_zero_compression_sha256": _content_hash(_matrix_payload(base_zero)),
                "required_negative_companion_sha256": _content_hash(_matrix_payload(required)),
                "zero_speed_cancellation_exact": base_zero == required,
                "corrected_nonzero_equal_eigenspace_compressions": nonzero,
                "corrected_axis2_D4_Sylvester_solvable": solvable,
                "corrected_axis2_D4_residual_zero": audit["residual_zero"],
                "wrong_sign_axis2_D4_Sylvester_solvable": wrong_sign_solvable,
                "deltaK_ABCD_Hermitian": delta == delta.T if solvable else False,
                "deltaK_ABCD_rank": delta.rank() if solvable else None,
                "deltaK_ABCD_nonzero_entries": (
                    sum(value != 0 for value in delta) if solvable else 0
                ),
                "deltaK_ABCD_sha256": (_content_hash(_matrix_payload(delta)) if solvable else None),
            }
        )
    return rows


def _exact_axis2_audit(
    fourth_campaign: Mapping[str, Any],
    candidates: Mapping[str, Any],
    companion: Mapping[str, Any],
) -> dict[str, Any]:
    selector = fourth_campaign["selector"]
    selector_record = selector["records"][OBLIGATION_OFFSET]
    if (
        tuple(selector_record["active_indices"]) != ACTIVE_INDICES
        or selector_record["selector_offset"] != OBLIGATION_OFFSET
    ):
        raise QuarticTC2D4Axis2BaseRHSError("obligation-244 selector mismatch")
    directions = _active_directions()
    basis_directions = [directions[position] for position in selector["active_positions"]]
    reference = _axis2_reference()
    payload, directional_evaluations = _polarized_axis2_payload(
        ACTIVE_INDICES, basis_directions, reference
    )
    if directional_evaluations != EXPECTED_DIRECTIONAL_EVALUATIONS:
        raise QuarticTC2D4Axis2BaseRHSError("unexpected axis-two polarization evaluation count")
    rhs = payload["fourth_Sylvester_RHS"]
    basis = _correction_basis()
    output = basis["block"][:, 21]
    correction_block = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    if _content_hash(_matrix_payload(correction_block)) != EXPECTED_COMPANION_BLOCK_SHA256:
        raise QuarticTC2D4Axis2BaseRHSError("companion block identity mismatch")
    correction_skew = (
        reference["energy0"] * correction_block - correction_block.T * reference["energy0"]
    ).applyfunc(sp.factor)
    target = (
        reference["projectors"][sp.S.Zero].T * correction_skew * reference["projectors"][sp.S.Zero]
    ).applyfunc(sp.factor)
    if (
        target.rank() != 2
        or _content_hash(_matrix_payload(target)) != EXPECTED_COMPANION_COMPRESSION_SHA256
    ):
        raise QuarticTC2D4Axis2BaseRHSError("companion compression mismatch")
    base_compressions = {
        eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
    }
    base_solvable, base_delta, base_audit = _solve_axis2_sylvester(rhs, reference)
    candidate_rows = _candidate_rows(rhs, correction_skew, target, reference, candidates, companion)
    compatible = sum(row["corrected_axis2_D4_Sylvester_solvable"] for row in candidate_rows)
    cancellation_count = sum(row["zero_speed_cancellation_exact"] for row in candidate_rows)
    nonzero_base = {
        str(eigenvalue): {
            "rank": matrix.rank(),
            "nonzero_entries": sum(value != 0 for value in matrix),
            "sha256": _content_hash(_matrix_payload(matrix)),
            "sparse": _sparse_matrix(matrix),
        }
        for eigenvalue, matrix in base_compressions.items()
        if not matrix.is_zero_matrix
    }
    return {
        "selector_record": selector_record,
        "directional_evaluations": directional_evaluations,
        "axis_2_reference": {
            "P55_sha256": _content_hash(_matrix_payload(reference["physical0"])),
            "K55_sha256": _content_hash(_matrix_payload(reference["energy0"])),
            "R02_sha256": _content_hash(_matrix_payload(reference["projectors"][sp.S.Zero])),
        },
        "polarized_base_D4": {
            "D4P55_sha256": _content_hash(_matrix_payload(payload["D4P55"])),
            "D4K55_sha256": _content_hash(_matrix_payload(payload["D4K55"])),
            "D4TC2_sha256": _content_hash(_matrix_payload(payload["D4TC2"])),
            "RHS_base_sha256": _content_hash(_matrix_payload(rhs)),
            "RHS_base_nonzero_entries": sum(value != 0 for value in rhs),
            "RHS_base_free_parameters": sorted(str(symbol) for symbol in rhs.free_symbols),
            "nonzero_equal_eigenspace_compressions": nonzero_base,
        },
        "companion_correction": {
            "block_sha256": EXPECTED_COMPANION_BLOCK_SHA256,
            "compression_sha256": EXPECTED_COMPANION_COMPRESSION_SHA256,
            "compression_rank": target.rank(),
            "compression_nonzero_entries": sum(value != 0 for value in target),
        },
        "candidate_comparison": candidate_rows,
        "result": {
            "candidate_conditions_checked": len(candidate_rows),
            "zero_speed_cancellations_exact": cancellation_count,
            "corrected_axis2_D4_compatibilities": compatible,
            "corrected_axis2_D4_obstructions": len(candidate_rows) - compatible,
            "full_axis2_base_D4_RHS_evaluated": True,
            "base_D4_RHS_identically_zero": rhs.is_zero_matrix,
            "base_D4_Sylvester_solvable": base_solvable,
            "base_D4_deltaK_zero": base_delta.is_zero_matrix,
            "base_D4_residual_zero": base_audit["residual_zero"],
            "wrong_sign_companion_compatibilities": sum(
                row["wrong_sign_axis2_D4_Sylvester_solvable"] for row in candidate_rows
            ),
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
        raise QuarticTC2D4Axis2BaseRHSError("axis-two base-RHS config mismatch")
    for key in (
        "campaign_source",
        "campaign_test",
        "diagonal_engine_source",
        "fourth_kernel_source",
    ):
        _check_raw_binding(root, config[key])
    companion = _load_bound(root, config["companion_range"])
    fourth_campaign = _load_bound(root, config["fourth_campaign"])
    candidates = _load_bound(root, config["candidate_source"])
    if (
        companion.get("status")
        != "pass_exact_axis2_companion_obstruction_and_pure_curl_range_no_go"
        or companion.get("claims", {}).get("full_axis2_base_D4_RHS_evaluated") is not False
        or fourth_campaign.get("counts", {}).get("fourth_selector_records") != 3060
        or candidates.get("counts", {}).get("selected") != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2D4Axis2BaseRHSError(
            "axis-two base-RHS predecessor semantic contract mismatch"
        )
    exact = _exact_axis2_audit(fourth_campaign, candidates, companion)
    result = exact["result"]
    all_compatible = result["corrected_axis2_D4_compatibilities"] == EXPECTED_CANDIDATES
    all_obstructed = result["corrected_axis2_D4_obstructions"] == EXPECTED_CANDIDATES
    status = (
        "pass_exact_all_12_axis2_D4_companion_compatibilities"
        if all_compatible
        else "pass_exact_all_12_axis2_D4_companion_obstructions"
        if all_obstructed
        else "pass_exact_axis2_D4_companion_candidate_classification"
    )
    body = {
        "schema_version": SCHEMA,
        "status": status,
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "companion_range",
                "fourth_campaign",
                "candidate_source",
                "diagonal_engine_source",
                "fourth_kernel_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "reference_direction": "e2",
            "same_active_tensor_component_inputs": True,
            "selector_record_sha256": exact["selector_record"]["record_sha256"],
        },
        "exact_axis2_base_D4_audit": exact,
        "counts": {
            "directional_evaluations": exact["directional_evaluations"],
            "candidate_conditions_checked": EXPECTED_CANDIDATES,
            "zero_speed_cancellations_exact": result["zero_speed_cancellations_exact"],
            "corrected_axis2_D4_compatibilities": result["corrected_axis2_D4_compatibilities"],
            "corrected_axis2_D4_obstructions": result["corrected_axis2_D4_obstructions"],
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "omit_complete_base_D4_RHS": {
                "companion_alone_compression_rank": 2,
                "rejected": True,
            },
            "claim_base_D4_itself_is_obstructed": {
                "base_D4_RHS_identically_zero": result["base_D4_RHS_identically_zero"],
                "base_D4_Sylvester_solvable": result["base_D4_Sylvester_solvable"],
                "rejected": True,
            },
            "flip_companion_tuning_sign": {
                "candidate_conditions_accepted": result["wrong_sign_companion_compatibilities"],
                "rejected": True,
            },
            "infer_compatibility_from_zero_speed_compression_only": {
                "all_seven_equal_eigenspaces_checked": True,
                "rejected": True,
            },
            "alter_obligation_244_selector": {
                "selector_record_sha256": exact["selector_record"]["record_sha256"],
                "rejected": True,
            },
            "promote_axis2_result_to_global_closure": {
                "remaining_D4_selector_closed": False,
                "tube_theorem_proved": False,
                "rejected": True,
            },
        },
        "claims": {
            "full_axis2_base_D4_RHS_evaluated": True,
            "all_12_axis2_D4_compatibilities_proved": all_compatible,
            "all_12_axis2_D4_obstructions_proved": all_obstructed,
            "fixed_chart_curl_completion_axis2_D4_rejected": all_obstructed,
            "spatially_covariant_tensor_completion_proved": False,
            "all_spatial_direction_compatibility_proved": False,
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
            "Construct an admissible topology-changing completion whose direction-two block "
            "has zero order-four zero-speed compression, or whose independently derived base "
            "e2 D4 forcing cancels it. The current fixed-chart C12 curl completion is exactly "
            "obstructed at e2, and the predecessor already rules out repair by further pure "
            "C23 curl additions preserving the direction-one V slice."
        ),
        "scope": (
            "Exact polarized base-D4 Sylvester RHS at the e2 reference for obligation 244 "
            "with the same four active tensor-component inputs, followed by all-seven-"
            "eigenspace comparison with the candidate-specific curl companion. This is a "
            "proved obstruction of the proposed fixed-chart curl completion for all 12 "
            "candidates on one reference direction and selector obligation, not spatial covariance, "
            "remaining D4 closure, tube closure, CK1, CK3, TC2, B7, global-H7, or lifespan."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4Axis2BaseRHSError("axis-two base-RHS content identity mismatch")
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    exact = document.get("exact_axis2_base_D4_audit", {})
    result = exact.get("result", {})
    rows = exact.get("candidate_comparison", [])
    all_compatible = counts.get("corrected_axis2_D4_compatibilities") == 12
    all_obstructed = counts.get("corrected_axis2_D4_obstructions") == 12
    if (
        document.get("status") != "pass_exact_all_12_axis2_D4_companion_obstructions"
        or counts
        != {
            "candidate_conditions_checked": 12,
            "corrected_axis2_D4_compatibilities": 0,
            "corrected_axis2_D4_obstructions": 12,
            "directional_evaluations": 15,
            "inferred_global_passes": 0,
            "negative_controls": 6,
            "zero_speed_cancellations_exact": 0,
        }
        or exact.get("polarized_base_D4", {}).get("RHS_base_sha256") != ZERO_55_MATRIX_SHA256
        or exact.get("polarized_base_D4", {}).get("D4P55_sha256") != ZERO_55_MATRIX_SHA256
        or exact.get("polarized_base_D4", {}).get("D4K55_sha256") != ZERO_55_MATRIX_SHA256
        or exact.get("polarized_base_D4", {}).get("D4TC2_sha256") != ZERO_55_MATRIX_SHA256
        or exact.get("polarized_base_D4", {}).get("RHS_base_nonzero_entries") != 0
        or exact.get("polarized_base_D4", {}).get("RHS_base_free_parameters") != []
        or exact.get("polarized_base_D4", {}).get("nonzero_equal_eigenspace_compressions") != {}
        or result.get("base_D4_RHS_identically_zero") is not True
        or result.get("base_D4_Sylvester_solvable") is not True
        or result.get("base_D4_deltaK_zero") is not True
        or result.get("base_D4_residual_zero") is not True
        or result.get("wrong_sign_companion_compatibilities") != 0
        or counts.get("directional_evaluations") != EXPECTED_DIRECTIONAL_EVALUATIONS
        or counts.get("candidate_conditions_checked") != EXPECTED_CANDIDATES
        or counts.get("negative_controls") != 6
        or counts.get("inferred_global_passes") != 0
        or result.get("full_axis2_base_D4_RHS_evaluated") is not True
        or len(rows) != EXPECTED_CANDIDATES
        or any(
            row.get("base_zero_compression_sha256") != ZERO_55_MATRIX_SHA256
            or row.get("zero_speed_cancellation_exact") is not False
            or row.get("corrected_axis2_D4_Sylvester_solvable") is not False
            or row.get("corrected_axis2_D4_residual_zero") is not False
            or row.get("wrong_sign_axis2_D4_Sylvester_solvable") is not False
            or set(row.get("corrected_nonzero_equal_eigenspace_compressions", {})) != {"0"}
            or row.get("corrected_nonzero_equal_eigenspace_compressions", {})
            .get("0", {})
            .get("rank")
            != 2
            or row.get("corrected_nonzero_equal_eigenspace_compressions", {})
            .get("0", {})
            .get("nonzero_entries")
            != 10
            for row in rows
        )
        or sum(row.get("zero_speed_cancellation_exact") is True for row in rows)
        != counts.get("zero_speed_cancellations_exact")
        or sum(row.get("corrected_axis2_D4_Sylvester_solvable") is True for row in rows)
        != counts.get("corrected_axis2_D4_compatibilities")
        or claims.get("full_axis2_base_D4_RHS_evaluated") is not True
        or claims.get("all_12_axis2_D4_compatibilities_proved") is not all_compatible
        or claims.get("all_12_axis2_D4_obstructions_proved") is not all_obstructed
        or claims.get("fixed_chart_curl_completion_axis2_D4_rejected") is not all_obstructed
        or any(
            claims.get(key) is not False
            for key in (
                "spatially_covariant_tensor_completion_proved",
                "all_spatial_direction_compatibility_proved",
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
        raise QuarticTC2D4Axis2BaseRHSError("axis-two base-RHS exact/fail-closed contract mismatch")


def run_campaign(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the exact axis-two polarized base-D4 Sylvester RHS."
    )
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
                "compatibilities": artifact["counts"]["corrected_axis2_D4_compatibilities"],
                "obstructions": artifact["counts"]["corrected_axis2_D4_obstructions"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
