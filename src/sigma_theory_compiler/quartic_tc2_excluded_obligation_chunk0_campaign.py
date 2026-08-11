from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_second_atom_chunk64_campaign import (
    _candidate_coefficients,
    _substitute_entries,
)
from .quartic_tc2_second_atom_chunk_campaign import (
    _direction_key,
    _second_pair_symbolic_packet,
)
from .quartic_tc2_variable_sylvester_campaign import (
    _content_hash,
    _content_hash_matches,
    _coordinate_atom_to_jet_packet,
)

SCHEMA_VERSION = "sigma-quartic-tc2-excluded-obligation-chunk0-1.0"
CHUNK_SIZE = 64


class QuarticTC2ExcludedObligationChunk0Error(ValueError):
    """Raised when excluded-obligation Sylvester work is not exact."""


def run_quartic_tc2_excluded_obligation_chunk0_campaign(
    variable: dict[str, Any],
    classification: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        manifest = classification["excluded_pair_manifest"]
        remaining = [item for item in manifest if not item["rigorously_discharged"]]
        selected = remaining[:CHUNK_SIZE]
        declared_selector = classification["next_exact_selector"]
        if (
            config.get("schema_version") != SCHEMA_VERSION
            or config.get("variable_campaign_sha256") != variable.get("content_sha256")
            or config.get("classification_content_sha256")
            != classification.get("content_sha256")
            or config.get("classification_manifest_sha256") != _content_hash(manifest)
            or config.get("selector_sha256") != _content_hash(selected)
            or int(config.get("chunk_offset", -1)) != 0
            or int(config.get("chunk_size", 0)) != CHUNK_SIZE
            or config.get("global_TC2_policy") != "fail_closed"
            or config.get("B7_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
            or not _content_hash_matches(variable)
            or not _content_hash_matches(classification)
            or classification.get("status")
            != (
                "pass_exact_excluded_pair_partition_with_zero_subfamily_"
                "remaining_obligations_fail_closed"
            )
            or len(remaining) != 2675
            or len(selected) != CHUNK_SIZE
            or declared_selector.get("selector_sha256") != _content_hash(selected)
            or declared_selector.get("pair_global_indices")
            != [item["global_pair_index"] for item in selected]
        ):
            raise QuarticTC2ExcludedObligationChunk0Error(
                "unsupported excluded-obligation chunk contract"
            )
        coordinate = _coordinate_atom_to_jet_packet()
        if (
            coordinate["packet"]["content_sha256"]
            != classification["upstream_sha256"]["coordinate_to_jet_packet"]
        ):
            raise QuarticTC2ExcludedObligationChunk0Error(
                "coordinate-to-jet provenance mismatch"
            )
        second_packets = {
            item["content_sha256"]: item["jet_entries"]
            for item in classification["second_coordinate_direction_packets"]
        }
        candidates = _candidate_coefficients(variable)
        if len(candidates) != 12:
            raise QuarticTC2ExcludedObligationChunk0Error(
                "candidate coefficient count mismatch"
            )
        seed = _content_hash(
            {
                "classification_content_sha256": classification["content_sha256"],
                "classification_manifest_sha256": _content_hash(manifest),
                "selector_sha256": _content_hash(selected),
                "variable_campaign_sha256": variable["content_sha256"],
                "offset": 0,
                "size": CHUNK_SIZE,
            }
        )
        previous = seed
        records: list[dict[str, Any]] = []
        symbolic_packets: dict[str, dict[str, Any]] = {}
        first_obstruction: dict[str, Any] | None = None
        omitted_pushforward_control: dict[str, Any] | None = None
        for local_index, obligation in enumerate(selected):
            left_direction = coordinate["maps"][obligation["left_atom_index"]]
            right_direction = coordinate["maps"][obligation["right_atom_index"]]
            second_serialized = second_packets.get(
                obligation["second_coordinate_direction_sha256"], {}
            )
            if bool(second_serialized) != obligation[
                "exact_second_coordinate_direction_nonzero"
            ]:
                raise QuarticTC2ExcludedObligationChunk0Error(
                    "second-coordinate packet/manifest mismatch"
                )
            second_direction = {
                name: sp.sympify(value)
                for name, value in second_serialized.items()
            }
            packet = _second_pair_symbolic_packet(
                _direction_key(left_direction),
                _direction_key(right_direction),
                _direction_key(second_direction),
            )
            symbolic_packets[packet["content_sha256"]] = packet
            if local_index == 0:
                omitted = _second_pair_symbolic_packet(
                    _direction_key(left_direction), _direction_key(right_direction)
                )
                omitted_pushforward_control = {
                    "pair_global_index": obligation["global_pair_index"],
                    "required_D2J_sha256": obligation[
                        "second_coordinate_direction_sha256"
                    ],
                    "with_pushforward_packet_sha256": packet["content_sha256"],
                    "omitted_pushforward_packet_sha256": omitted["content_sha256"],
                    "packet_changes_when_D2J_omitted": omitted["content_sha256"]
                    != packet["content_sha256"],
                    "rejected": omitted["content_sha256"]
                    != packet["content_sha256"],
                }
                if not omitted_pushforward_control["rejected"]:
                    raise QuarticTC2ExcludedObligationChunk0Error(
                        "omitted D2J negative control did not change exact packet"
                    )
            candidate_results: list[dict[str, Any]] = []
            for candidate_id in sorted(candidates):
                coefficients = candidates[candidate_id]
                substitutions = {
                    sp.Symbol("alpha"): sp.sympify(coefficients["a10"]),
                    sp.Symbol("c20"): sp.sympify(coefficients["c20"]),
                }
                compressions = {
                    eigenvalue: _substitute_entries(entries, substitutions)
                    for eigenvalue, entries in packet[
                        "equal_eigenspace_compressions"
                    ].items()
                }
                compressions = {
                    eigenvalue: entries
                    for eigenvalue, entries in compressions.items()
                    if entries
                }
                delta_entries = _substitute_entries(
                    packet["deltaK_AB_entries"], substitutions
                )
                solvable = bool(
                    not compressions
                    and packet["equal_eigenspace_compressions_zero"]
                    and packet["second_Sylvester_residual_zero"]
                )
                candidate = {
                    "candidate_id": candidate_id,
                    "solvable": solvable,
                    "compression_residual_sha256": _content_hash(compressions),
                    "deltaK_AB_nonzero_entries": len(delta_entries) if solvable else 0,
                    "deltaK_AB_sha256": (
                        _content_hash(delta_entries) if solvable else None
                    ),
                    "Hermitian": packet["deltaK_AB_Hermitian"] if solvable else False,
                    "second_Sylvester_residual_zero": (
                        packet["second_Sylvester_residual_zero"] if solvable else False
                    ),
                }
                candidate_results.append(candidate)
                if not solvable and first_obstruction is None:
                    eigenvalue, entries = next(iter(compressions.items()))
                    first_obstruction = {
                        "chunk_local_index": local_index,
                        "global_pair_index": obligation["global_pair_index"],
                        "left_atom": obligation["left_atom"],
                        "right_atom": obligation["right_atom"],
                        "requirement": obligation["requirement"],
                        "candidate_id": candidate_id,
                        "eigenvalue": eigenvalue,
                        "first_nonzero_entry": entries[0],
                        "symbolic_packet_sha256": packet["content_sha256"],
                        "second_coordinate_direction_sha256": obligation[
                            "second_coordinate_direction_sha256"
                        ],
                    }
            record_body = {
                "chunk_local_index": local_index,
                "obligation_selector_index": local_index,
                "global_pair_index": obligation["global_pair_index"],
                "left_atom_index": obligation["left_atom_index"],
                "right_atom_index": obligation["right_atom_index"],
                "left_atom": obligation["left_atom"],
                "right_atom": obligation["right_atom"],
                "requirement": obligation["requirement"],
                "left_direction_sha256": _content_hash(
                    {name: str(value) for name, value in left_direction.items()}
                ),
                "right_direction_sha256": _content_hash(
                    {name: str(value) for name, value in right_direction.items()}
                ),
                "second_coordinate_direction_sha256": obligation[
                    "second_coordinate_direction_sha256"
                ],
                "symbolic_packet_sha256": packet["content_sha256"],
                "candidate_results": candidate_results,
                "previous_record_sha256": previous,
            }
            record_hash = _content_hash(record_body)
            records.append({**record_body, "record_sha256": record_hash})
            previous = record_hash
            if first_obstruction is not None:
                break

        evaluated = len(records)
        candidate_checks = evaluated * len(candidates)
        solvable_checks = sum(
            candidate["solvable"]
            for record in records
            for candidate in record["candidate_results"]
        )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_first_64_excluded_obligations_no_obstruction_remaining_fail_closed"
                if first_obstruction is None
                else "exact_excluded_obligation_Sylvester_obstruction_found_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "variable_campaign": variable["content_sha256"],
                "classification": classification["content_sha256"],
                "classification_manifest": _content_hash(manifest),
                "coordinate_to_jet_packet": coordinate["packet"][
                    "content_sha256"
                ],
            },
            "config_sha256": _content_hash(config),
            "chunk_contract": {
                "pair_selector": "excluded_obligation_ascending_global_pair_index",
                "selector_pair_count": len(remaining),
                "chunk_offset": 0,
                "requested_chunk_size": CHUNK_SIZE,
                "evaluated_chunk_size": evaluated,
                "chunk_seed_sha256": seed,
                "resume_after_record_sha256": previous,
                "stopped_at_first_obstruction": first_obstruction is not None,
                "global_pair_indices_are_stable": True,
                "classification_selector_sha256": _content_hash(selected),
            },
            "omitted_D2J_negative_control": omitted_pushforward_control,
            "pair_manifest": records,
            "symbolic_pair_packets": [
                symbolic_packets[key] for key in sorted(symbolic_packets)
            ],
            "first_exact_obstruction": first_obstruction,
            "counts": {
                "full_unordered_coordinate_atom_pairs": 11781,
                "completed_canonical_active_pairs": 861,
                "classification_discharged_zero_pairs": 8245,
                "total_excluded_obligations": len(remaining),
                "current_evaluated_obligations": evaluated,
                "remaining_unevaluated_obligations": len(remaining) - evaluated,
                "candidates": len(candidates),
                "current_candidate_checks": candidate_checks,
                "current_solvable_candidate_checks": solvable_checks,
                "current_obstructed_candidate_checks": candidate_checks
                - solvable_checks,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "claim": (
                "Only the declared first excluded-obligation chunk was evaluated, with "
                "exact intrinsic jet Hessians and coordinate D2J pushforwards."
            ),
            "scope": (
                "Unevaluated excluded obligations are not inferred. TC2, B7, global H7, "
                "dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2ExcludedObligationChunk0Error) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "pair_manifest": [],
            "counts": {
                "full_unordered_coordinate_atom_pairs": 11781,
                "current_evaluated_obligations": 0,
                "remaining_unevaluated_obligations": 2675,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_excluded_obligation_chunk0_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
