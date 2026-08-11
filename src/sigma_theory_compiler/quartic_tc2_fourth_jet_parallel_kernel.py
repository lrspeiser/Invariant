from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from multiprocessing import get_context
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _matrix_payload,
)
from .quartic_tc2_mixed_third_jet_chunk_campaign import _solve_sylvester

JET_ORDER = 4
EXPECTED_CANDIDATES = 12


class QuarticTC2FourthJetParallelKernelError(ValueError):
    """Raised when an exact fourth-jet worker violates its contract."""


def _combine_directions(
    directions: tuple[dict[str, sp.Expr], ...], indices: tuple[int, ...]
) -> dict[str, sp.Expr]:
    result: dict[str, sp.Expr] = {}
    for index in indices:
        for name, value in directions[index].items():
            result[name] = sp.factor(result.get(name, sp.S.Zero) + value)
    return {name: value for name, value in result.items() if value != 0}


def _direction_key(direction: dict[str, sp.Expr]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((name, str(value)) for name, value in direction.items()))


def _directional_fourth_payload(
    direction: dict[str, sp.Expr],
) -> dict[str, sp.Matrix]:
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        packet = directional_engine._directional_taylor_packet(
            {
                "atom_index": -1,
                "atom": "fourth_polarization_direction:"
                + _content_hash(_direction_key(direction))[:16],
                "direction": direction,
            }
        )
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    lower_orders = packet["orders"][: JET_ORDER - 1]
    if len(packet["orders"]) != JET_ORDER or len(lower_orders) != 3 or not all(
        order["solvable"] and order["residual_zero"] for order in lower_orders
    ):
        raise QuarticTC2FourthJetParallelKernelError(
            "directional recurrence failed in mandatory orders one through three"
        )
    multiplier = sp.Integer(24)
    return {
        "D4P55": (multiplier * packet["physical"][4]).applyfunc(sp.factor),
        "D4K55": (multiplier * packet["energy"][4]).applyfunc(sp.factor),
        "D4TC2": (multiplier * packet["block"][4]).applyfunc(sp.factor),
        "fourth_Sylvester_RHS": (
            multiplier * packet["orders"][3]["rhs"]
        ).applyfunc(sp.factor),
    }


def _polarized_fourth_payload(
    active_indices: tuple[int, int, int, int],
    basis_directions: list[dict[str, Any]],
) -> tuple[dict[str, sp.Matrix], int]:
    directions = tuple(
        basis_directions[index]["direction"] for index in active_indices
    )
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
    payloads = [(weights[key], _directional_fourth_payload(combined[key])) for key in keys]
    if not payloads:
        raise QuarticTC2FourthJetParallelKernelError("empty polarization payload")
    result: dict[str, sp.Matrix] = {}
    for name in payloads[0][1]:
        result[name] = (
            sum(
                (weight * payload[name] for weight, payload in payloads),
                sp.zeros(*payloads[0][1][name].shape),
            )
            / 24
        ).applyfunc(sp.factor)
    directional_engine._directional_taylor_packet_cached.cache_clear()
    return result, len(payloads)


def _candidate_result(
    rhs: sp.Matrix,
    alpha: sp.Symbol,
    c20: sp.Symbol,
    coefficients: dict[str, str],
) -> dict[str, Any]:
    candidate_rhs = rhs.subs(
        {
            alpha: sp.sympify(coefficients["a10"]),
            c20: sp.sympify(coefficients["c20"]),
        }
    ).applyfunc(sp.factor)
    solvable, delta, audit = _solve_sylvester(candidate_rhs)
    return {
        "solvable": solvable,
        "equal_eigenspace_compressions_zero": solvable,
        "nonzero_equal_eigenspace_compressions": audit[
            "nonzero_equal_eigenspace_compressions"
        ],
        "deltaK_ABCD_Hermitian": delta.equals(delta.T) if solvable else False,
        "deltaK_ABCD_nonzero_entries": sum(value != 0 for value in delta),
        "deltaK_ABCD_rank": delta.rank() if solvable else None,
        "deltaK_ABCD_sha256": (
            _content_hash(_matrix_payload(delta)) if solvable else None
        ),
        "fourth_Sylvester_residual_zero": audit["residual_zero"],
    }


def evaluate_fourth_obligation_kernel(
    request: tuple[
        tuple[int, int, int, int],
        list[int],
        dict[str, dict[str, str]],
    ],
) -> dict[str, Any]:
    active_indices, basis_positions, coefficients = request
    if (
        tuple(sorted(active_indices)) != active_indices
        or len(active_indices) != JET_ORDER
        or len(basis_positions) != 15
        or len(coefficients) != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2FourthJetParallelKernelError("invalid fourth obligation")
    directions = _active_directions()
    if max(basis_positions) >= len(directions) or max(active_indices) >= len(basis_positions):
        raise QuarticTC2FourthJetParallelKernelError("fourth obligation escaped basis")
    basis_directions = [directions[position] for position in basis_positions]
    payload, directional_evaluations = _polarized_fourth_payload(
        active_indices, basis_directions
    )
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    symbolic_solvable, symbolic_delta, symbolic_audit = _solve_sylvester(rhs)
    candidate_results: list[dict[str, Any]] = []
    obstructed_candidates: list[str] = []
    for candidate_id, candidate_coefficients in sorted(coefficients.items()):
        candidate = _candidate_result(rhs, alpha, c20, candidate_coefficients)
        if not candidate["solvable"]:
            obstructed_candidates.append(candidate_id)
        candidate_results.append({"candidate_id": candidate_id, **candidate})
    return {
        "active_indices": list(active_indices),
        "basis_positions": [basis_positions[index] for index in active_indices],
        "directional_evaluations": directional_evaluations,
        "D4P55_nonzero_entries": sum(value != 0 for value in payload["D4P55"]),
        "D4P55_sha256": _content_hash(_matrix_payload(payload["D4P55"])),
        "D4K55_nonzero_entries": sum(value != 0 for value in payload["D4K55"]),
        "D4K55_sha256": _content_hash(_matrix_payload(payload["D4K55"])),
        "D4TC2_nonzero_entries": sum(value != 0 for value in payload["D4TC2"]),
        "D4TC2_sha256": _content_hash(_matrix_payload(payload["D4TC2"])),
        "fourth_Sylvester_RHS_sha256": _content_hash(_matrix_payload(rhs)),
        "symbolic_parameter_compatible": symbolic_solvable,
        "symbolic_nonzero_equal_eigenspace_compressions": symbolic_audit[
            "nonzero_equal_eigenspace_compressions"
        ],
        "symbolic_deltaK_ABCD_Hermitian": (
            symbolic_delta.equals(symbolic_delta.T) if symbolic_solvable else False
        ),
        "symbolic_deltaK_ABCD_nonzero_entries": sum(value != 0 for value in symbolic_delta),
        "symbolic_deltaK_ABCD_rank": (
            symbolic_delta.rank() if symbolic_solvable else None
        ),
        "symbolic_deltaK_ABCD_sha256": (
            _content_hash(_matrix_payload(symbolic_delta)) if symbolic_solvable else None
        ),
        "candidate_results": candidate_results,
        "obstructed_candidate_ids": obstructed_candidates,
    }


def evaluate_fourth_obligations_process_pool(
    active_indices: list[tuple[int, int, int, int]],
    basis_positions: list[int],
    coefficients: dict[str, dict[str, str]],
    *,
    worker_count: int,
) -> list[dict[str, Any]]:
    if not active_indices or worker_count < 1 or worker_count > len(active_indices):
        raise QuarticTC2FourthJetParallelKernelError("invalid parallel batch")
    requests = [(indices, basis_positions, coefficients) for indices in active_indices]
    if worker_count == 1:
        results = [evaluate_fourth_obligation_kernel(request) for request in requests]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=get_context("spawn")
        ) as executor:
            results = list(
                executor.map(evaluate_fourth_obligation_kernel, requests, chunksize=1)
            )
    if [result["active_indices"] for result in results] != [
        list(indices) for indices in active_indices
    ]:
        raise QuarticTC2FourthJetParallelKernelError("parallel result order mismatch")
    return results
