from __future__ import annotations

import hashlib
import json
import math
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
from .quartic_tc2_quadratic_deltak_extension_campaign import (
    _collect_records,
)
from .quartic_tc2_second_atom_chunk_campaign import (
    _canonical_active_affine_pairs,
)
from .quartic_tc2_variable_sylvester_campaign import (
    STATE_DIMENSION,
    _reference_and_first_jet_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-diagonal-third-jet-campaign-1.0"
TAYLOR_ORDER = 3


class QuarticTC2DiagonalThirdJetError(ValueError):
    """Raised when a diagonal third-jet certificate is incomplete or overstated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(value: dict[str, Any]) -> bool:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return value.get("content_sha256") == _content_hash(body)


def _matrix_payload(matrix: sp.MatrixBase) -> list[list[str]]:
    return [
        [str(matrix[row, column]) for column in range(matrix.cols)] for row in range(matrix.rows)
    ]


def _zero_series(rows: int, columns: int) -> list[sp.Matrix]:
    return [sp.zeros(rows, columns) for _ in range(TAYLOR_ORDER + 1)]


def _series_product(left: list[sp.Matrix], right: list[sp.Matrix]) -> list[sp.Matrix]:
    result = _zero_series(left[0].rows, right[0].cols)
    for order in range(TAYLOR_ORDER + 1):
        result[order] = sum(
            (left[index] * right[order - index] for index in range(order + 1)),
            sp.zeros(left[0].rows, right[0].cols),
        ).applyfunc(sp.factor)
    return result


def _series_transpose(series: list[sp.Matrix]) -> list[sp.Matrix]:
    return [matrix.T for matrix in series]


def _directional_derivative(
    matrix: sp.Matrix,
    direction: dict[str, sp.Expr],
    jet_symbols: dict[str, sp.Symbol],
    substitutions: dict[sp.Symbol, sp.Expr],
    order: int,
) -> sp.Matrix:
    terms: list[tuple[sp.Expr, tuple[sp.Symbol, ...]]] = [(sp.S.One, ())]
    for _ in range(order):
        terms = [
            (coefficient * direction_coefficient, derivatives + (jet_symbols[name],))
            for coefficient, derivatives in terms
            for name, direction_coefficient in direction.items()
        ]
    return (
        sum(
            (coefficient * matrix.diff(*derivatives) for coefficient, derivatives in terms),
            sp.zeros(*matrix.shape),
        )
        .subs(substitutions)
        .applyfunc(sp.factor)
    )


def _projector_series(
    companion: list[sp.Matrix],
    eigenvalue: sp.Expr,
    spectrum: tuple[sp.Expr, ...],
) -> list[sp.Matrix]:
    identity = sp.eye(companion[0].rows)
    result = [identity, *[sp.zeros(*identity.shape) for _ in range(TAYLOR_ORDER)]]
    for other in spectrum:
        if other == eigenvalue:
            continue
        denominator = eigenvalue - other
        factor = [matrix / denominator for matrix in companion]
        factor[0] = (companion[0] - other * identity) / denominator
        result = _series_product(result, factor)
    return result


def _inverse_series(series: list[sp.Matrix]) -> list[sp.Matrix]:
    result = _zero_series(series[0].rows, series[0].cols)
    result[0] = series[0].inv()
    for order in range(1, TAYLOR_ORDER + 1):
        result[order] = (
            -result[0]
            * sum(
                (series[index] * result[order - index] for index in range(1, order + 1)),
                sp.zeros(*series[0].shape),
            )
        ).applyfunc(sp.factor)
    return result


def generic_diagonal_third_jet_control() -> tuple[bool, dict[str, Any]]:
    t = sp.Symbol("t")
    p = [
        sp.diag(1, 2),
        sp.Matrix([[0, 3], [5, 0]]),
        sp.Matrix([[7, 0], [0, 11]]),
        sp.Matrix([[0, 13], [17, 0]]),
    ]
    d = [
        sp.Matrix([[0, 19], [19, 0]]),
        sp.Matrix([[23, 0], [0, 29]]),
        sp.Matrix([[0, 31], [31, 0]]),
        sp.Matrix([[37, 0], [0, 41]]),
    ]
    s = [sp.zeros(2) for _ in range(4)]
    p_curve = sum((t**index * item for index, item in enumerate(p)), sp.zeros(2))
    d_curve = sum((t**index * item for index, item in enumerate(d)), sp.zeros(2))
    s_curve = sum((t**index * item for index, item in enumerate(s)), sp.zeros(2))
    exact = (d_curve * p_curve - p_curve.T * d_curve + s_curve).applyfunc(sp.expand)
    coefficient = exact.diff(t, 3).subs(t, 0) / math.factorial(3)
    recurrence = (
        d[3] * p[0]
        - p[0].T * d[3]
        + s[3]
        + sum(
            (d[index] * p[3 - index] - p[3 - index].T * d[index] for index in range(3)),
            sp.zeros(2),
        )
    ).applyfunc(sp.expand)
    corrupted = (recurrence - d[2] * p[1]).applyfunc(sp.expand)
    passed = bool(
        (coefficient - recurrence).is_zero_matrix and not (coefficient - corrupted).is_zero_matrix
    )
    return passed, {
        "control": "factorial-normalized diagonal cubic Sylvester recurrence",
        "Taylor_coefficient_residual_zero": (coefficient - recurrence).is_zero_matrix,
        "third_derivative_multiplier": math.factorial(3),
        "recurrence": (
            "R3=S3+deltaK0*P3-P3^dagger*deltaK0+"
            "deltaK1*P2-P2^dagger*deltaK1+deltaK2*P1-P1^dagger*deltaK2"
        ),
        "negative_controls": {
            "omit_deltaK2_P1": {
                "nonzero_entries": sum(value != 0 for value in coefficient - corrupted),
                "rejected": not (coefficient - corrupted).is_zero_matrix,
            },
            "infer_mixed_triples_from_diagonal_triples": {
                "missing": "polarized AAB, ABB, and ABC coefficients",
                "rejected": True,
            },
            "infer_full_tube_identity_from_third_jet": {
                "missing": "fourth-and-higher residual jets or a nonlinear range theorem",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _active_directions() -> list[dict[str, Any]]:
    directions: dict[int, dict[str, Any]] = {}
    for pair in _canonical_active_affine_pairs():
        for side in ("left", "right"):
            atom_index = int(pair[f"{side}_atom_index"])
            record = {
                "atom_index": atom_index,
                "atom": str(pair[f"{side}_atom"]),
                "direction": dict(pair[f"{side}_direction"]),
            }
            if atom_index in directions and directions[atom_index] != record:
                raise QuarticTC2DiagonalThirdJetError("active direction identity mismatch")
            directions[atom_index] = record
    return [directions[index] for index in sorted(directions)]


def _packet_matrix(
    entries: list[dict[str, Any]], symbols: dict[str, sp.Symbol] | None = None
) -> sp.Matrix:
    matrix = sp.zeros(STATE_DIMENSION)
    for entry in entries:
        matrix[int(entry["row"]), int(entry["column"])] = sp.sympify(
            entry["value"], locals=symbols or {}
        )
    return matrix


def _diagonal_second_packets(
    records: list[dict[str, Any]], packets: dict[str, dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    diagonal: dict[int, dict[str, Any]] = {}
    for record in records:
        left = int(record["left_atom_index"])
        right = int(record["right_atom_index"])
        if left != right:
            continue
        packet_hash = str(record["symbolic_pair_packet_sha256"])
        packet = packets.get(packet_hash)
        if packet is None or packet["content_sha256"] != packet_hash:
            raise QuarticTC2DiagonalThirdJetError("diagonal second packet is absent")
        diagonal[left] = packet
    return diagonal


@cache
def _directional_taylor_packet_cached(
    atom_index: int,
    atom: str,
    direction_key: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    direction = {name: sp.sympify(value) for name, value in direction_key}
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
    ordering = [*range(11), *range(33, 55), *range(11, 33)]
    coefficient_a = data["first_order"]["A"]
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], list(xi[1:])
    )
    mass: list[sp.Matrix] = []
    evolution: list[sp.Matrix] = []
    for order in range(TAYLOR_ORDER + 1):
        if order == 0:
            extra = {alpha: 0, c20: 0}
            a_order = coefficient_a.subs({**substitutions, **extra})
            b_order = b_blocks[0].subs({**substitutions, **extra})
            c_order = [matrix.subs({**substitutions, **extra}) for matrix in c_blocks[0]]
        else:
            scale = sp.Rational(1, math.factorial(order))
            a_order = scale * _directional_derivative(
                coefficient_a, direction, jet_symbols, substitutions, order
            )
            b_order = scale * _directional_derivative(
                b_blocks[0], direction, jet_symbols, substitutions, order
            )
            c_order = [
                scale
                * _directional_derivative(matrix, direction, jet_symbols, substitutions, order)
                for matrix in c_blocks[0]
            ]
        mass_order, evolution_order = _full_first_order_pencil(a_order, b_order, c_order, [1, 0, 0])
        mass.append(mass_order)
        evolution.append(evolution_order)
    physical_original = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    physical_original[0] = mass[0].inv() * evolution[0]
    for order in range(1, TAYLOR_ORDER + 1):
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
    physical = [
        matrix.extract(ordering, ordering).applyfunc(sp.factor) for matrix in physical_original
    ]
    if not physical[0].equals(reference["physical0"]):
        raise QuarticTC2DiagonalThirdJetError("P55 reference mismatch")

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
    for order in range(TAYLOR_ORDER + 1):
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
    h = [
        b.row_join(a).col_join(a.row_join(sp.zeros(11)))
        for a, b in zip(action_a, action_b, strict=True)
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
            for order in range(TAYLOR_ORDER + 1)
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
    for order in range(TAYLOR_ORDER + 1):
        energy[order][0:33, 33:55] = cross[order]
        energy[order][33:55, 0:33] = cross[order].T
        energy[order][33:55, 33:55] = companion_energy[order]
        energy[order] = energy[order].applyfunc(sp.factor)
    if not energy[0].equals(reference["energy0"]):
        raise QuarticTC2DiagonalThirdJetError("K55 reference mismatch")

    q = sp.zeros(11)
    q[0, 10], q[4, 10], q[10, 7], q[10, 9] = 2, -8, 2, 2
    embedded_q = sp.zeros(STATE_DIMENSION, 11)
    embedded_q[33:44, :] = q
    high = sp.zeros(STATE_DIMENSION, 1)
    high[54] = 1
    block = [
        (alpha * matrix * embedded_q[:, 10] * high.T).applyfunc(sp.factor) for matrix in physical
    ]
    skew = _series_product(energy, block)
    skew_transpose = _series_product(_series_transpose(block), energy)
    skew = [
        (left - right).applyfunc(sp.factor)
        for left, right in zip(skew, skew_transpose, strict=True)
    ]

    full_spectrum = (
        sp.Integer(0),
        sp.Integer(1),
        sp.Integer(-1),
        sp.Rational(1, 2),
        sp.Rational(-1, 2),
        sp.Rational(1, 3),
        sp.Rational(-1, 3),
    )
    reference_projectors = reference["projectors"]
    delta = _zero_series(STATE_DIMENSION, STATE_DIMENSION)
    delta[0] = alpha * reference["delta0"]
    order_records: list[dict[str, Any]] = []
    for order in range(1, TAYLOR_ORDER + 1):
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
            for eigenvalue, projector in reference_projectors.items()
        }
        solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
        if solvable:
            for left in full_spectrum:
                for right in full_spectrum:
                    if left != right:
                        delta[order] += (
                            reference_projectors[left].T
                            * rhs
                            * reference_projectors[right]
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
    return {
        "atom_index": atom_index,
        "atom": atom,
        "direction": direction,
        "physical": physical,
        "energy": energy,
        "block": block,
        "orders": order_records,
        "alpha": alpha,
        "c20": c20,
    }


def _directional_taylor_packet(direction_record: dict[str, Any]) -> dict[str, Any]:
    return _directional_taylor_packet_cached(
        int(direction_record["atom_index"]),
        str(direction_record["atom"]),
        tuple(sorted((name, str(value)) for name, value in direction_record["direction"].items())),
    )


def _candidate_third_record(packet: dict[str, Any], coefficients: dict[str, str]) -> dict[str, Any]:
    substitutions = {
        packet["alpha"]: sp.sympify(coefficients["a10"]),
        packet["c20"]: sp.sympify(coefficients["c20"]),
    }
    third = packet["orders"][2]
    rhs = third["rhs"].subs(substitutions).applyfunc(sp.factor)
    reference = _reference_and_first_jet_packet()
    compressions = {
        eigenvalue: (projector.T * rhs * projector).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
    }
    solvable = all(matrix.is_zero_matrix for matrix in compressions.values())
    delta = sp.zeros(STATE_DIMENSION)
    if solvable:
        for left, left_projector in reference["projectors"].items():
            for right, right_projector in reference["projectors"].items():
                if left != right:
                    delta += left_projector.T * rhs * right_projector / (left - right)
        delta = delta.applyfunc(sp.factor)
    residual = (delta * reference["physical0"] - reference["physical0"].T * delta + rhs).applyfunc(
        sp.factor
    )
    derivative = (math.factorial(3) * delta).applyfunc(sp.factor)
    nonzero_compressions = [
        {
            "eigenvalue": str(eigenvalue),
            "nonzero_entries": sum(value != 0 for value in matrix),
            "sha256": _content_hash(_matrix_payload(matrix)),
        }
        for eigenvalue, matrix in compressions.items()
        if not matrix.is_zero_matrix
    ]
    return {
        "solvable": solvable,
        "equal_eigenspace_compressions_zero": solvable,
        "nonzero_equal_eigenspace_compressions": nonzero_compressions,
        "deltaK_AAA_Hermitian": derivative.equals(derivative.T) if solvable else False,
        "deltaK_AAA_nonzero_entries": sum(value != 0 for value in derivative),
        "deltaK_AAA_rank": derivative.rank() if solvable else None,
        "deltaK_AAA_sha256": _content_hash(_matrix_payload(derivative)) if solvable else None,
        "third_Sylvester_residual_zero": residual.is_zero_matrix,
    }


def run_quartic_tc2_diagonal_third_jet_campaign(
    variable_campaign: dict[str, Any],
    quadratic_campaign: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2DiagonalThirdJetError("unsupported schema_version")
        if not _content_hash_matches(variable_campaign) or not _content_hash_matches(
            quadratic_campaign
        ):
            raise QuarticTC2DiagonalThirdJetError("upstream content hash mismatch")
        expected = config["expected_upstream_content_sha256"]
        actual = {
            "variable_sylvester": variable_campaign["content_sha256"],
            "quadratic_deltaK": quadratic_campaign["content_sha256"],
        }
        if actual != expected:
            raise QuarticTC2DiagonalThirdJetError("configured provenance mismatch")
        if quadratic_campaign.get("status") != (
            "pass_all_12_complete_reference_quadratic_deltaK_two_jets_full_identity_fail_closed"
        ):
            raise QuarticTC2DiagonalThirdJetError("quadratic campaign status mismatch")
        if any(
            config.get(key) != "fail_closed"
            for key in (
                "mixed_third_jet_policy",
                "full_tube_policy",
                "CK1_policy",
                "CK3_policy",
                "TC2_policy",
                "B7_policy",
                "global_H7_policy",
                "lifespan_policy",
            )
        ):
            raise QuarticTC2DiagonalThirdJetError("closure policy mismatch")
        if config.get("selector") != "all_41_canonical_active_affine_coordinate_diagonal_triples":
            raise QuarticTC2DiagonalThirdJetError("third-jet selector mismatch")
        passed, generic = generic_diagonal_third_jet_control()
        if not passed:
            raise QuarticTC2DiagonalThirdJetError("generic third-jet control failed")
        records, packets, artifact_hashes = _collect_records(
            canonical_artifacts,
            selector_key="selector_pair_index",
            expected_count=861,
        )
        if (
            artifact_hashes
            != quadratic_campaign["second_pair_artifact_content_sha256"]["canonical"]
        ):
            raise QuarticTC2DiagonalThirdJetError("canonical artifact sequence mismatch")
        diagonal_second = _diagonal_second_packets(records, packets)
        active = _active_directions()
        if len(active) != 41 or set(diagonal_second) != {
            int(item["atom_index"]) for item in active
        }:
            raise QuarticTC2DiagonalThirdJetError("41-direction diagonal coverage mismatch")
        coefficients = {
            str(item["candidate_id"]): item["coefficients"]
            for item in variable_campaign["certificates"]
        }
        if len(coefficients) != 12:
            raise QuarticTC2DiagonalThirdJetError("candidate count mismatch")
        direction_records: list[dict[str, Any]] = []
        candidate_passes = {candidate_id: 0 for candidate_id in coefficients}
        symbolic_third_passes = 0
        for direction_record in active:
            packet = _directional_taylor_packet(direction_record)
            if len(packet["orders"]) < 3 or not all(
                item["solvable"] and item["residual_zero"] for item in packet["orders"][:2]
            ):
                raise QuarticTC2DiagonalThirdJetError("lower-order recurrence mismatch")
            expected_second = _packet_matrix(
                diagonal_second[int(direction_record["atom_index"])]["deltaK_AB_entries"],
                {"alpha": packet["alpha"], "c20": packet["c20"]},
            )
            actual_second = (math.factorial(2) * packet["orders"][1]["delta"]).applyfunc(sp.factor)
            if not actual_second.equals(expected_second):
                raise QuarticTC2DiagonalThirdJetError("committed D2 deltaK mismatch")
            third_symbolic = packet["orders"][2]
            symbolic_third_passes += int(third_symbolic["solvable"])
            candidate_results: list[dict[str, Any]] = []
            for candidate_id, candidate_coefficients in sorted(coefficients.items()):
                candidate = _candidate_third_record(packet, candidate_coefficients)
                candidate_passes[candidate_id] += int(candidate["solvable"])
                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        **candidate,
                    }
                )
            d3_physical = (math.factorial(3) * packet["physical"][3]).applyfunc(sp.factor)
            d3_energy = (math.factorial(3) * packet["energy"][3]).applyfunc(sp.factor)
            d3_block = (math.factorial(3) * packet["block"][3]).applyfunc(sp.factor)
            d3_rhs = (math.factorial(3) * third_symbolic["rhs"]).applyfunc(sp.factor)
            d3_delta = (math.factorial(3) * third_symbolic["delta"]).applyfunc(sp.factor)
            direction_records.append(
                {
                    "atom_index": direction_record["atom_index"],
                    "atom": direction_record["atom"],
                    "direction": {
                        name: str(value) for name, value in direction_record["direction"].items()
                    },
                    "D3P55_nonzero_entries": sum(value != 0 for value in d3_physical),
                    "D3P55_sha256": _content_hash(_matrix_payload(d3_physical)),
                    "D3K55_nonzero_entries": sum(value != 0 for value in d3_energy),
                    "D3K55_sha256": _content_hash(_matrix_payload(d3_energy)),
                    "D3TC2_nonzero_entries": sum(value != 0 for value in d3_block),
                    "D3TC2_sha256": _content_hash(_matrix_payload(d3_block)),
                    "committed_D2_deltaK_reproduced": True,
                    "symbolic_equal_eigenspace_compressions_zero": third_symbolic["solvable"],
                    "third_Sylvester_RHS_sha256": _content_hash(_matrix_payload(d3_rhs)),
                    "symbolic_deltaK_AAA_Hermitian": d3_delta.equals(d3_delta.T),
                    "symbolic_deltaK_AAA_nonzero_entries": sum(value != 0 for value in d3_delta),
                    "symbolic_deltaK_AAA_rank": d3_delta.rank(),
                    "symbolic_deltaK_AAA_sha256": _content_hash(_matrix_payload(d3_delta)),
                    "candidate_results": candidate_results,
                }
            )
        candidate_certificates = [
            {
                "candidate_id": candidate_id,
                "coefficients": coefficients[candidate_id],
                "diagonal_active_coordinate_third_jets_tested": len(active),
                "diagonal_third_jets_solvable": candidate_passes[candidate_id],
                "diagonal_third_jets_obstructed": len(active) - candidate_passes[candidate_id],
                "all_41_diagonal_third_jets_closed": candidate_passes[candidate_id] == len(active),
                "mixed_third_jets_closed": False,
                "full_tube_Sylvester_identity": False,
                "CK1_closed": False,
                "CK3_closed": False,
                "TC2_closed": False,
                "B7_closed": False,
                "global_H7_closed": False,
                "lifespan_proved": False,
            }
            for candidate_id in sorted(coefficients)
        ]
        total_candidate_directions = len(active) * len(coefficients)
        total_candidate_passes = sum(candidate_passes.values())
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_"
                "mixed_triples_full_tube_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": actual,
            "canonical_second_pair_artifact_content_sha256": artifact_hashes,
            "canonical_second_pair_artifact_sequence_sha256": _content_hash(artifact_hashes),
            "config_sha256": _content_hash(config),
            "generic_diagonal_third_jet_control": generic,
            "slice_contract": {
                "coordinate_sector": "canonical active affine second-partial atoms at fixed q,p",
                "active_coordinate_directions": len(active),
                "tested_triples": "(A,A,A) for every one of the 41 active coordinate directions",
                "diagonal_triples": len(active),
                "mixed_AAB_ABB_ABC_triples": 0,
                "full_symmetric_triples_in_41_direction_sector": 41 * 42 * 43 // 6,
                "full_symmetric_triples_in_153_coordinate_basis": 153 * 154 * 155 // 6,
                "factorial_normalized_internal_recurrence": True,
                "reported_deltaK_AAA_is_third_derivative": True,
            },
            "counts": {
                "candidates": len(coefficients),
                "diagonal_direction_packets": len(active),
                "symbolic_parameter_diagonal_third_jet_passes": symbolic_third_passes,
                "candidate_direction_evaluations": total_candidate_directions,
                "candidate_direction_solvable": total_candidate_passes,
                "candidate_direction_obstructed": total_candidate_directions
                - total_candidate_passes,
                "candidates_all_41_diagonal_third_jets_closed": sum(
                    item["all_41_diagonal_third_jets_closed"] for item in candidate_certificates
                ),
                "mixed_third_jet_closures": 0,
                "full_tube_Sylvester_identities": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "direction_records": direction_records,
            "certificates": candidate_certificates,
            "first_remaining_blocker": {
                "gate": "polarized mixed third Sylvester jets",
                "required": (
                    "D3K55, D3P55, D3TC2 and deltaK_AAB/deltaK_ABB/deltaK_ABC "
                    "for the remaining 12,300 polarized mixed triples in the 12,341-triple "
                    "41-direction active sector, "
                    "or a rigorously uniform nonlinear Sylvester-range theorem"
                ),
                "then": "fourth-and-higher remainder or a tube-uniform nonlinear range proof",
                "closed": False,
            },
            "claim": (
                "The exact third Sylvester recurrence has been audited on every diagonal "
                "coordinate triple in the 41-direction canonical active affine sector, "
                "with D3P55, D3K55, D3TC2, equal-eigenspace compatibility, and candidate "
                "deltaK_AAA evaluated against the committed D2 construction."
            ),
            "scope": (
                "This bounded certificate covers only (A,A,A) triples. It does not infer "
                "polarized mixed triples or a full tube identity. CK1, CK3, TC2, B7, "
                "global H7, dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2DiagonalThirdJetError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "direction_records": [],
            "certificates": [],
            "counts": {
                "candidates": 0,
                "diagonal_direction_packets": 0,
                "candidate_direction_evaluations": 0,
                "candidate_direction_solvable": 0,
                "candidate_direction_obstructed": 0,
                "candidates_all_41_diagonal_third_jets_closed": 0,
                "mixed_third_jet_closures": 0,
                "full_tube_Sylvester_identities": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_diagonal_third_jet_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
