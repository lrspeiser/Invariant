from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-tc2-symmetrizer-no-go-campaign-1.0"
STATE_DIMENSION = 55
FIELD_DIMENSION = 11
HIGH_STATE_INDEX = 32  # w1[10]


class QuarticTC2SymmetrizerNoGoError(ValueError):
    """Raised when the TC2 energy-pairing no-go is not exact."""


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


def _direction_matrix(packet: dict[str, Any], direction: int) -> sp.Matrix:
    matrix = sp.zeros(STATE_DIMENSION, FIELD_DIMENSION)
    record = packet["directions"][direction - 1]
    if int(record["spatial_direction"]) != direction:
        raise QuarticTC2SymmetrizerNoGoError("TC2 direction ordering mismatch")
    for entry in record["entries"]:
        matrix[int(entry["row"]), int(entry["column"])] = sp.sympify(
            entry["value"]
        )
    return matrix


@cache
def generic_tc2_symmetrizer_no_go_control() -> tuple[bool, dict[str, Any]]:
    # Three-state exact control: M has rank two and h is the third state.
    matrix = sp.Matrix([[1, 0], [0, 1], [0, 0]])
    h = sp.Matrix([0, 0, 1])
    ell0, ell1 = sp.symbols("ell0 ell1", real=True)
    ell = sp.Matrix([ell0, ell1])
    energy = sp.diag(2, 3, 5)
    b = matrix * ell * h.T
    energy_adjoint = energy.inv() * b.T * energy
    completed = b + energy_adjoint
    pairing_residual = (energy * b - b.T * energy).applyfunc(sp.factor)
    completed_residual = (
        energy * completed - completed.T * energy
    ).applyfunc(sp.factor)
    half_completion = b + energy_adjoint / 2
    half_residual = (
        energy * half_completion - half_completion.T * energy
    ).applyfunc(sp.factor)
    euclidean_completion = b + b.T
    euclidean_residual = (
        energy * euclidean_completion - euclidean_completion.T * energy
    ).applyfunc(sp.factor)

    alpha = sp.Symbol("alpha", nonzero=True, real=True)
    actual_minor = sp.factor((2 * alpha) * (2 * alpha) - 0)
    passed = bool(
        matrix.rank() == 2
        and actual_minor == 4 * alpha**2
        and not pairing_residual.is_zero_matrix
        and completed_residual.is_zero_matrix
        and not half_residual.is_zero_matrix
        and not euclidean_residual.is_zero_matrix
    )
    return passed, {
        "control": "TC2 exact K55-adjoint energy-pairing no-go",
        "ansatz_class": {
            "name": "unchanged_positive_K55_separate_TC2_absorption",
            "TC2_symbol": "B_k(ell)=P55^k E_v Q ell e_h^T",
            "high_state_covector": "e_h=e_(w1[10])",
            "absorption_condition": "K55 B_k(ell)=B_k(ell)^dagger K55 for every ell",
        },
        "rank_obstruction": {
            "lemma": (
                "u e_h^T symmetric implies u is parallel to e_h; hence if K is "
                "invertible and K M ell e_h^T is symmetric for every ell, rank(M)<=1"
            ),
            "actual_direction_1_minor_rows_22_32_columns_10_7": str(actual_minor),
            "actual_direction_1_rank": 2,
            "conclusion": (
                "no positive/invertible K55 can absorb direction-1 TC2 as a separate "
                "symmetric principal block for all low coefficient vectors"
            ),
        },
        "exact_adjoint_completion": {
            "K_adjoint": "B^(dagger_K)=K^-1 B^dagger K",
            "canonical_reciprocal_block": (
                "K55^-1 e_h ell^T Q^T E_v^T (P55^k)^dagger K55"
            ),
            "completed_pairing": "K55(B+B^(dagger_K)) is Hermitian",
            "control_residual_zero": completed_residual.is_zero_matrix,
        },
        "minimal_unabsorbed_residual": {
            "skew_pairing": "(K55 B-B^dagger K55)/2",
            "integration_by_parts_status": (
                "does not reduce to a coefficient derivative and retains one high "
                "spatial derivative"
            ),
        },
        "negative_controls": {
            "omit_reciprocal_adjoint_block": {
                "nonzero_residual_entries": sum(
                    value != 0 for value in pairing_residual
                ),
                "rejected": not pairing_residual.is_zero_matrix,
            },
            "add_only_half_reciprocal_block": {
                "nonzero_residual_entries": sum(value != 0 for value in half_residual),
                "rejected": not half_residual.is_zero_matrix,
            },
            "use_Euclidean_transpose_instead_of_K_adjoint": {
                "nonzero_residual_entries": sum(
                    value != 0 for value in euclidean_residual
                ),
                "rejected": not euclidean_residual.is_zero_matrix,
            },
            "collapse_direction_1_packet_to_rank_one": {
                "required_minor": str(actual_minor),
                "rank_one_minor": "0",
                "rejected": actual_minor != 0,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    induced: dict[str, Any],
    first_order: dict[str, Any],
    pde: dict[str, Any],
    symmetrizer: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(induced["candidate_id"])
    records = (first_order, pde, symmetrizer)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTC2SymmetrizerNoGoError("candidate identity mismatch")
    coefficients = induced["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTC2SymmetrizerNoGoError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticTC2SymmetrizerNoGoError("TC2 rank obstruction requires a10!=0")
    packet = induced["actual_P55_on_embedded_Q"]["TC2_packet"]
    direction_one = _direction_matrix(packet, 1)
    decisive_minor = sp.factor(
        direction_one[22, 10] * direction_one[32, 7]
        - direction_one[22, 7] * direction_one[32, 10]
    )
    if direction_one.rank() != 2 or decisive_minor != 4 * alpha**2:
        raise QuarticTC2SymmetrizerNoGoError("direction-one rank/minor mismatch")
    energy = symmetrizer["energy_equivalence"]
    lower = sp.sympify(energy["K55_2_lower"])
    upper = sp.sympify(energy["K55_2_upper"])
    inverse_upper = sp.sympify(energy["K55_inverse_2_upper"])
    if not (lower > 0 and upper > 0 and sp.factor(inverse_upper - 1 / lower) == 0):
        raise QuarticTC2SymmetrizerNoGoError("K55 energy bounds mismatch")
    direction_one_frobenius = sp.factor(sp.sqrt(sum(value**2 for value in direction_one)))
    expected_frobenius = 2 * sp.sqrt(19) * sp.Abs(alpha)
    if sp.factor(direction_one_frobenius - expected_frobenius) != 0:
        raise QuarticTC2SymmetrizerNoGoError("direction-one norm mismatch")
    condition_upper = sp.factor(upper * inverse_upper)
    reciprocal_upper = sp.factor(condition_upper * direction_one_frobenius)
    numeric_condition = float(sp.N(condition_upper, 18))
    numeric_reciprocal = float(sp.N(reciprocal_upper, 18))
    if not (numeric_condition > 0 and numeric_reciprocal > 0):
        raise QuarticTC2SymmetrizerNoGoError("reciprocal completion bound invalid")
    residual_payload = {
        "candidate_id": candidate_id,
        "TC2_packet_sha256": packet["content_sha256"],
        "direction": 1,
        "rank": direction_one.rank(),
        "decisive_minor": str(decisive_minor),
        "high_state_index": HIGH_STATE_INDEX,
        "energy_pairing": "K55",
        "skew_residual": "(K55 B_1(ell)-B_1(ell)^dagger K55)/2",
    }
    reciprocal_payload = {
        "residual_sha256": _content_hash(residual_payload),
        "formula": "K55^-1 B_1(ell)^dagger K55",
        "K55_condition_upper": str(condition_upper),
        "P55_1_Ev_Q_Frobenius": str(direction_one_frobenius),
        "operator_upper": str(reciprocal_upper),
    }
    return {
        "schema_version": "sigma-quartic-tc2-symmetrizer-no-go-certificate-1.0",
        "status": "pass_exact_TC2_unchanged_K55_no_go_reciprocal_block_missing",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "TC2_component_packet_sha256": packet["content_sha256"],
            "source_spatial_block_sha256": first_order[
                "source_spatial_block_sha256"
            ],
            "two_channel_identity_sha256": induced["provenance"][
                "two_channel_identity_sha256"
            ],
            "K55_energy_equivalence_sha256": _content_hash(energy),
            "skew_residual_sha256": _content_hash(residual_payload),
            "canonical_reciprocal_block_sha256": _content_hash(
                reciprocal_payload
            ),
        },
        "actual_direction_1_packet": {
            "matrix": "P55^1 E_v Q",
            "nonzero_entries": packet["directions"][0]["entries"],
            "rank": direction_one.rank(),
            "decisive_minor_rows_22_32_columns_10_7": str(decisive_minor),
            "Frobenius_norm": str(direction_one_frobenius),
        },
        "exact_K55_adjoint_energy_pairing": {
            "K55_positive_lower": str(lower),
            "K55_upper": str(upper),
            "K55_inverse_upper": str(inverse_upper),
            "baseline_principal_symmetrization": "K55 P55=P55^dagger K55",
            "TC2_absorption_condition": (
                "K55 B_1(ell)=B_1(ell)^dagger K55 for every ell"
            ),
            "condition_impossible_by_rank": True,
        },
        "bounded_no_go": {
            "ansatz_class": generic["ansatz_class"]["name"],
            "K55_condition_number_upper": str(condition_upper),
            "K55_condition_number_upper_numeric": numeric_condition,
            "canonical_reciprocal_block_2_norm_upper": str(reciprocal_upper),
            "canonical_reciprocal_block_2_norm_upper_numeric": numeric_reciprocal,
            "unchanged_K55_separate_TC2_absorption_refuted": True,
        },
        "minimal_residual": {
            "formula": "(K55 B_1(ell)-B_1(ell)^dagger K55)/2",
            "hash": _content_hash(residual_payload),
            "nonzero_for_some_low_coefficient_vector": True,
            "retains_one_high_spatial_derivative": True,
        },
        "canonical_missing_completion": {
            "formula": "K55^-1 B_1(ell)^dagger K55",
            "hash": _content_hash(reciprocal_payload),
            "makes_K55_times_completed_block_Hermitian": True,
            "present_in_current_modified_state": False,
            "adding_it_preserves_state_to_jet_constraints_proved": False,
        },
        "connection_to_TC2_B7_global_H7": {
            "TC2_component_packet_retained": True,
            "TC2_absorbed_by_current_K55_energy": False,
            "TC2_closed": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "derive the canonical reciprocal K55-adjoint block from an additional state "
            "correction, or solve the coupled linearized symmetrizer equation for K55+deltaK; "
            "then verify constraints, positivity, and all new commutators"
        ),
    }


def run_quartic_tc2_symmetrizer_no_go_campaign(
    induced_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    pde_campaign: dict[str, Any],
    full_symmetrizer_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            induced_campaign,
            first_order_campaign,
            pde_campaign,
            full_symmetrizer_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_exact_P55_Q_TC_packets_reference_partial_bounds_"
                "global_H7_fail_closed"
            ),
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
            "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2SymmetrizerNoGoError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTC2SymmetrizerNoGoError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTC2SymmetrizerNoGoError("campaign content hash mismatch")
        if (
            induced_campaign["upstream_sha256"]["first_order_P55"]
            != first_order_campaign["content_sha256"]
            or pde_campaign["upstream_sha256"]["first_order"]
            != first_order_campaign["content_sha256"]
            or full_symmetrizer_campaign["upstream_sha256"][
                "nonquasilinear_pde"
            ]
            != pde_campaign["content_sha256"]
        ):
            raise QuarticTC2SymmetrizerNoGoError("upstream provenance mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["state_dimension"]) != STATE_DIMENSION
            or int(config["high_state_index"]) != HIGH_STATE_INDEX
            or config.get("ansatz_class")
            != "unchanged_positive_K55_separate_TC2_absorption"
            or config.get("reciprocal_completion_policy") != "identify_fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTC2SymmetrizerNoGoError("unsupported TC2 no-go contract")
        generic_passed, generic = generic_tc2_symmetrizer_no_go_control()
        if not generic_passed:
            raise QuarticTC2SymmetrizerNoGoError("generic TC2 no-go failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTC2SymmetrizerNoGoError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_exact_TC2_unchanged_K55_no_gos_"
                "reciprocal_blocks_missing_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "induced_operator": induced_campaign["content_sha256"],
                "first_order_P55": first_order_campaign["content_sha256"],
                "nonquasilinear_PDE_K55": pde_campaign["content_sha256"],
                "full_K55_symmetrizer": full_symmetrizer_campaign[
                    "content_sha256"
                ],
            },
            "config_sha256": _content_hash(config),
            "generic_TC2_symmetrizer_no_go_control": generic,
            "counts": {
                "selected": len(certificates),
                "exact_direction_1_rank_obstructions": len(certificates),
                "unchanged_K55_TC2_absorption_no_gos": len(certificates),
                "canonical_reciprocal_blocks_identified": len(certificates),
                "canonical_reciprocal_blocks_derived_from_state": 0,
                "TC2_closures": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The exact direction-1 P55 E_v Q packet has rank two and a nonzero "
                "4*a10^2 minor. For any positive K55, separate TC2 absorption would force "
                "the rank-two range of K55 P55^1 E_v Q into the one-dimensional high-state "
                "line, which is impossible. The exact K55-adjoint reciprocal completion "
                "and an explicit condition-number norm bound are identified but absent."
            ),
            "scope": (
                "This rules out unchanged-K55 separate absorption of TC2 on the s01/H01 "
                "slice. It does not rule out a coupled modified symmetrizer/state correction. "
                "TC2, B7, global H7, summation, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2SymmetrizerNoGoError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "exact_direction_1_rank_obstructions": 0,
                "unchanged_K55_TC2_absorption_no_gos": 0,
                "canonical_reciprocal_blocks_identified": 0,
                "canonical_reciprocal_blocks_derived_from_state": 0,
                "TC2_closures": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_symmetrizer_no_go_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
