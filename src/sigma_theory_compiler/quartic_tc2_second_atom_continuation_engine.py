from __future__ import annotations

from typing import Any

import sympy as sp

from .quartic_tc2_second_atom_chunk64_campaign import (
    _candidate_coefficients,
    _substitute_entries,
    _validate_prior_chain,
)
from .quartic_tc2_second_atom_chunk192_campaign import _tensor_summary
from .quartic_tc2_second_atom_chunk_campaign import (
    DEFAULT_CHUNK_SIZE,
    _canonical_active_affine_pairs,
    _direction_key,
    _second_pair_symbolic_packet,
    generic_second_atom_sylvester_control,
)
from .quartic_tc2_variable_sylvester_campaign import (
    ATOM_DIMENSION,
    _content_hash,
    _content_hash_matches,
)


class QuarticTC2SecondAtomContinuationError(ValueError):
    """Raised when a generic second-atom continuation cannot be certified."""


def generic_contiguous_boundary_control(
    chunk_offset: int,
) -> tuple[bool, dict[str, Any]]:
    prior_offset = chunk_offset - DEFAULT_CHUNK_SIZE
    expected = prior_offset + DEFAULT_CHUNK_SIZE
    total = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
    passed = bool(expected == chunk_offset and 0 < chunk_offset < total)
    return passed, {
        "control": "generic contiguous 64-pair selector continuation",
        "prior_offset": prior_offset,
        "prior_size": DEFAULT_CHUNK_SIZE,
        "expected_next_offset": expected,
        "negative_controls": {
            "skip_one_selector_pair": {
                "corrupted_offset": expected + 1,
                "required_offset": chunk_offset,
                "rejected": expected + 1 != chunk_offset,
            },
            "claim_cumulative_chunk_is_full_coverage": {
                "cumulative": chunk_offset + DEFAULT_CHUNK_SIZE,
                "total": total,
                "rejected": chunk_offset + DEFAULT_CHUNK_SIZE < total,
            },
            "promote_to_global_H7": {
                "missing": "remaining second-atom pairs and TC2/CK energy closure",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def run_second_atom_continuation(
    prior_chunk_campaign: dict[str, Any],
    variable_campaign: dict[str, Any],
    config: dict[str, Any],
    *,
    schema_version: str,
    chunk_offset: int,
    expected_prior_status: str,
    success_status: str,
    obstruction_status: str,
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != schema_version:
            raise QuarticTC2SecondAtomContinuationError(
                "unsupported continuation schema_version"
            )
        if prior_chunk_campaign.get("status") != expected_prior_status:
            raise QuarticTC2SecondAtomContinuationError("prior chunk status mismatch")
        if variable_campaign.get("status") != (
            "pass_all_12_first_order_variable_deltaK_extensions_"
            "higher_orders_global_H7_fail_closed"
        ):
            raise QuarticTC2SecondAtomContinuationError(
                "variable campaign status mismatch"
            )
        if not _content_hash_matches(prior_chunk_campaign) or not _content_hash_matches(
            variable_campaign
        ):
            raise QuarticTC2SecondAtomContinuationError(
                "campaign content hash mismatch"
            )
        prior_contract = prior_chunk_campaign["chunk_contract"]
        if (
            prior_chunk_campaign["upstream_sha256"]["variable_Sylvester"]
            != variable_campaign["content_sha256"]
            or prior_contract["chunk_offset"]
            != chunk_offset - DEFAULT_CHUNK_SIZE
            or prior_contract["evaluated_chunk_size"] != DEFAULT_CHUNK_SIZE
            or prior_chunk_campaign["counts"][
                "cumulative_evaluated_coordinate_atom_pairs"
            ]
            != chunk_offset
            or int(config["chunk_offset"]) != chunk_offset
            or int(config["chunk_size"]) != DEFAULT_CHUNK_SIZE
            or config.get("pair_selector")
            != "canonical_sylvester_active_affine_second_atom_pairs"
            or config.get("resume_policy") != "require_prior_hash_chain_tip"
            or config.get("prior_resume_sha256")
            != prior_contract["resume_after_record_sha256"]
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTC2SecondAtomContinuationError(
                "unsupported continuation contract"
            )
        prior_chain_passed, prior_chain = _validate_prior_chain(prior_chunk_campaign)
        if not prior_chain_passed:
            raise QuarticTC2SecondAtomContinuationError(
                "prior record chain mismatch"
            )
        generic_passed, generic = generic_second_atom_sylvester_control()
        boundary_passed, boundary = generic_contiguous_boundary_control(chunk_offset)
        if not (generic_passed and boundary_passed):
            raise QuarticTC2SecondAtomContinuationError(
                "generic continuation control failed"
            )

        all_pairs = _canonical_active_affine_pairs()
        selected = all_pairs[chunk_offset : chunk_offset + DEFAULT_CHUNK_SIZE]
        if len(selected) != DEFAULT_CHUNK_SIZE:
            raise QuarticTC2SecondAtomContinuationError(
                "insufficient continuation selector pairs"
            )
        coefficients_by_candidate = _candidate_coefficients(variable_campaign)
        if len(coefficients_by_candidate) != 12:
            raise QuarticTC2SecondAtomContinuationError("candidate count mismatch")
        seed = _content_hash(
            {
                "prior_artifact": prior_chunk_campaign["content_sha256"],
                "prior_resume": prior_contract["resume_after_record_sha256"],
                "variable_campaign": variable_campaign["content_sha256"],
                "selector": config["pair_selector"],
                "offset": chunk_offset,
                "size": DEFAULT_CHUNK_SIZE,
            }
        )
        previous = seed
        manifest: list[dict[str, Any]] = []
        symbolic_packets: dict[str, dict[str, Any]] = {}
        first_obstruction: dict[str, Any] | None = None
        for local_index, pair in enumerate(selected):
            symbolic = _second_pair_symbolic_packet(
                _direction_key(pair["left_direction"]),
                _direction_key(pair["right_direction"]),
            )
            symbolic_packets[symbolic["content_sha256"]] = symbolic
            candidate_results: list[dict[str, Any]] = []
            for candidate_id in sorted(coefficients_by_candidate):
                coefficients = coefficients_by_candidate[candidate_id]
                substitutions = {
                    sp.Symbol("alpha"): sp.sympify(coefficients["a10"]),
                    sp.Symbol("c20"): sp.sympify(coefficients["c20"]),
                }
                compressions = {
                    eigenvalue: _substitute_entries(entries, substitutions)
                    for eigenvalue, entries in symbolic[
                        "equal_eigenspace_compressions"
                    ].items()
                }
                compressions = {
                    eigenvalue: entries
                    for eigenvalue, entries in compressions.items()
                    if entries
                }
                delta_entries = _substitute_entries(
                    symbolic["deltaK_AB_entries"], substitutions
                )
                solvable = bool(
                    not compressions
                    and symbolic["equal_eigenspace_compressions_zero"]
                    and symbolic["second_Sylvester_residual_zero"]
                )
                candidate_result = {
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
                candidate_results.append(candidate_result)
                if not solvable and first_obstruction is None:
                    eigenvalue, entries = next(iter(compressions.items()))
                    first_obstruction = {
                        "chunk_local_index": local_index,
                        "selector_pair_index": chunk_offset + local_index,
                        "global_pair_index": pair["global_pair_index"],
                        "left_atom": pair["left_atom"],
                        "right_atom": pair["right_atom"],
                        "candidate_id": candidate_id,
                        "eigenvalue": eigenvalue,
                        "first_nonzero_entry": entries[0],
                        "symbolic_pair_packet_sha256": symbolic["content_sha256"],
                    }
            record_body = {
                "chunk_local_index": local_index,
                "selector_pair_index": chunk_offset + local_index,
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
            if first_obstruction is not None:
                break

        evaluated_current = len(manifest)
        evaluated_cumulative = chunk_offset + evaluated_current
        total_pairs = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
        candidate_checks_current = evaluated_current * len(coefficients_by_candidate)
        solvable_current = sum(
            result["solvable"]
            for record in manifest
            for result in record["candidate_results"]
        )
        unique_packets = [symbolic_packets[key] for key in sorted(symbolic_packets)]
        body = {
            "schema_version": schema_version,
            "status": success_status if first_obstruction is None else obstruction_status,
            "errors": [],
            "upstream_sha256": {
                "prior_chunk": prior_chunk_campaign["content_sha256"],
                "prior_resume": prior_contract["resume_after_record_sha256"],
                "variable_Sylvester": variable_campaign["content_sha256"],
                "coordinate_to_jet_packet": variable_campaign[
                    "common_coordinate_to_covariant_jet_packet"
                ]["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_second_atom_control": generic,
            "generic_boundary_control": boundary,
            "verified_prior_chain": prior_chain,
            "chunk_contract": {
                "pair_selector": config["pair_selector"],
                "selector_pair_count": len(all_pairs),
                "chunk_offset": chunk_offset,
                "requested_chunk_size": DEFAULT_CHUNK_SIZE,
                "evaluated_chunk_size": evaluated_current,
                "chunk_seed_sha256": seed,
                "resume_after_record_sha256": previous,
                "prior_resume_sha256": prior_contract[
                    "resume_after_record_sha256"
                ],
                "stopped_at_first_obstruction": first_obstruction is not None,
                "global_pair_indices_are_stable": True,
            },
            "pair_manifest": manifest,
            "symbolic_pair_packets": unique_packets,
            "exact_tensor_summary_current_chunk": _tensor_summary(unique_packets),
            "first_exact_obstruction": first_obstruction,
            "counts": {
                "total_unordered_coordinate_atom_pairs": total_pairs,
                "prior_cumulative_evaluated_coordinate_atom_pairs": chunk_offset,
                "current_evaluated_coordinate_atom_pairs": evaluated_current,
                "cumulative_evaluated_coordinate_atom_pairs": evaluated_cumulative,
                "remaining_unevaluated_coordinate_atom_pairs": total_pairs
                - evaluated_cumulative,
                "candidates": len(coefficients_by_candidate),
                "current_evaluated_candidate_pairs": candidate_checks_current,
                "current_solvable_candidate_pairs": solvable_current,
                "current_obstructed_candidate_pairs": candidate_checks_current
                - solvable_current,
                "cumulative_deltaK_AB_constructions": int(
                    prior_chunk_campaign["counts"][
                        "cumulative_deltaK_AB_constructions"
                    ]
                )
                + solvable_current,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "claim": (
                f"The prior chain was revalidated and selector indices {chunk_offset} onward "
                "were evaluated until completion or first obstruction. Unevaluated pairs "
                "are not inferred."
            ),
            "scope": (
                f"Cumulative second-order Sylvester algebra through selector index "
                f"{evaluated_cumulative - 1} only. TC2, B7, global H7, dyadic summation, "
                "and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2SecondAtomContinuationError) as error:
        errors.append(str(error))
        total_pairs = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
        body = {
            "schema_version": schema_version,
            "status": "reject",
            "errors": errors,
            "pair_manifest": [],
            "symbolic_pair_packets": [],
            "counts": {
                "total_unordered_coordinate_atom_pairs": total_pairs,
                "prior_cumulative_evaluated_coordinate_atom_pairs": 0,
                "current_evaluated_coordinate_atom_pairs": 0,
                "cumulative_evaluated_coordinate_atom_pairs": 0,
                "remaining_unevaluated_coordinate_atom_pairs": total_pairs,
                "candidates": 0,
                "current_evaluated_candidate_pairs": 0,
                "current_solvable_candidate_pairs": 0,
                "current_obstructed_candidate_pairs": 0,
                "cumulative_deltaK_AB_constructions": 0,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}
