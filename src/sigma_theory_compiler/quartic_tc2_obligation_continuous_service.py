from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_continuous_service import (
    _atomic_write,
    _checkpoint_hash_matches,
    _json_bytes,
    _with_checkpoint_hash,
)
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

SERVICE_SCHEMA = "sigma-quartic-tc2-obligation-continuous-service-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-obligation-continuous-checkpoint-1.0"
DEFAULT_CHUNK_SIZE = 64
TOTAL_OBLIGATIONS = 2675

Executor = Callable[
    [
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
    ],
    dict[str, Any],
]


class QuarticTC2ObligationContinuousServiceError(ValueError):
    """Raised when an obligation service checkpoint cannot advance exactly."""


def _status(cumulative: int) -> str:
    return (
        f"pass_cumulative_{cumulative}_excluded_obligations_"
        "no_obstruction_remaining_fail_closed"
    )


def _metadata(offset: int, size: int) -> dict[str, Any]:
    return {
        "schema_version": (
            f"sigma-quartic-tc2-obligation-continuous-chunk-{offset}-size-{size}-1.0"
        ),
        "success_status": _status(offset + size),
        "obstruction_status": (
            f"exact_excluded_obligation_Sylvester_obstruction_at_offset_{offset}_"
            "fail_closed"
        ),
        "offset": offset,
        "size": size,
    }


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remaining_manifest(classification: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in classification["excluded_pair_manifest"]
        if not item["rigorously_discharged"]
    ]


def _validate_record_chain(artifact: dict[str, Any]) -> bool:
    contract = artifact.get("chunk_contract", {})
    records = artifact.get("pair_manifest", [])
    previous = contract.get("chunk_seed_sha256")
    for record in records:
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != _content_hash(body)
        ):
            return False
        previous = record["record_sha256"]
    return bool(
        len(records) == contract.get("evaluated_chunk_size")
        and previous == contract.get("resume_after_record_sha256")
    )


def _chunk_config(
    prior: dict[str, Any],
    classification: dict[str, Any],
    offset: int,
    size: int,
) -> dict[str, Any]:
    selected = _remaining_manifest(classification)[offset : offset + size]
    return {
        "chunk_offset": offset,
        "chunk_size": size,
        "selector_sha256": _content_hash(selected),
        "prior_artifact_sha256": prior["content_sha256"],
        "prior_resume_sha256": prior["chunk_contract"][
            "resume_after_record_sha256"
        ],
        "classification_content_sha256": classification["content_sha256"],
        "classification_manifest_sha256": _content_hash(
            classification["excluded_pair_manifest"]
        ),
        "global_TC2_policy": "fail_closed",
        "B7_policy": "fail_closed",
        "global_H7_policy": "fail_closed",
        "lifespan_policy": "fail_closed",
    }


def exact_obligation_executor(
    prior: dict[str, Any],
    variable: dict[str, Any],
    classification: dict[str, Any],
    chunk_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        offset, size = metadata["offset"], metadata["size"]
        all_remaining = _remaining_manifest(classification)
        selected = all_remaining[offset : offset + size]
        if (
            len(all_remaining) != TOTAL_OBLIGATIONS
            or len(selected) != size
            or not _content_hash_matches(prior)
            or not _content_hash_matches(variable)
            or not _content_hash_matches(classification)
            or not _validate_record_chain(prior)
            or chunk_config != _chunk_config(prior, classification, offset, size)
            or prior["counts"].get("current_evaluated_obligations") is None
            or (
                prior["chunk_contract"]["chunk_offset"]
                + prior["chunk_contract"]["evaluated_chunk_size"]
                != offset
            )
            or chunk_config["global_TC2_policy"] != "fail_closed"
            or chunk_config["B7_policy"] != "fail_closed"
            or chunk_config["global_H7_policy"] != "fail_closed"
            or chunk_config["lifespan_policy"] != "fail_closed"
        ):
            raise QuarticTC2ObligationContinuousServiceError(
                "unsupported exact obligation continuation contract"
            )
        coordinate = _coordinate_atom_to_jet_packet()
        if (
            coordinate["packet"]["content_sha256"]
            != classification["upstream_sha256"]["coordinate_to_jet_packet"]
        ):
            raise QuarticTC2ObligationContinuousServiceError(
                "coordinate-to-jet provenance mismatch"
            )
        second_packets = {
            item["content_sha256"]: item["jet_entries"]
            for item in classification["second_coordinate_direction_packets"]
        }
        candidates = _candidate_coefficients(variable)
        if len(candidates) != 12:
            raise QuarticTC2ObligationContinuousServiceError(
                "candidate coefficient count mismatch"
            )
        seed = _content_hash(
            {
                "prior_artifact_sha256": prior["content_sha256"],
                "prior_resume_sha256": prior["chunk_contract"][
                    "resume_after_record_sha256"
                ],
                "classification_content_sha256": classification["content_sha256"],
                "classification_manifest_sha256": _content_hash(
                    classification["excluded_pair_manifest"]
                ),
                "selector_sha256": _content_hash(selected),
                "variable_campaign_sha256": variable["content_sha256"],
                "offset": offset,
                "size": size,
            }
        )
        previous = seed
        records: list[dict[str, Any]] = []
        symbolic_packets: dict[str, dict[str, Any]] = {}
        first_obstruction: dict[str, Any] | None = None
        for local_index, obligation in enumerate(selected):
            left_direction = coordinate["maps"][obligation["left_atom_index"]]
            right_direction = coordinate["maps"][obligation["right_atom_index"]]
            second_serialized = second_packets.get(
                obligation["second_coordinate_direction_sha256"], {}
            )
            if bool(second_serialized) != obligation[
                "exact_second_coordinate_direction_nonzero"
            ]:
                raise QuarticTC2ObligationContinuousServiceError(
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
                candidate_result = {
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
                candidate_results.append(candidate_result)
                if not solvable and first_obstruction is None:
                    eigenvalue, entries = next(iter(compressions.items()))
                    first_obstruction = {
                        "chunk_local_index": local_index,
                        "obligation_selector_index": offset + local_index,
                        "global_pair_index": obligation["global_pair_index"],
                        "left_atom": obligation["left_atom"],
                        "right_atom": obligation["right_atom"],
                        "requirement": obligation["requirement"],
                        "candidate_id": candidate_id,
                        "eigenvalue": eigenvalue,
                        "first_nonzero_entry": entries[0],
                        "symbolic_packet_sha256": packet["content_sha256"],
                    }
            record_body = {
                "chunk_local_index": local_index,
                "obligation_selector_index": offset + local_index,
                "global_pair_index": obligation["global_pair_index"],
                "left_atom_index": obligation["left_atom_index"],
                "right_atom_index": obligation["right_atom_index"],
                "left_atom": obligation["left_atom"],
                "right_atom": obligation["right_atom"],
                "requirement": obligation["requirement"],
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
        checks = evaluated * len(candidates)
        solved = sum(
            item["solvable"]
            for record in records
            for item in record["candidate_results"]
        )
        body = {
            "schema_version": metadata["schema_version"],
            "status": (
                metadata["success_status"]
                if first_obstruction is None
                else metadata["obstruction_status"]
            ),
            "errors": [],
            "upstream_sha256": {
                "prior_artifact": prior["content_sha256"],
                "prior_resume": prior["chunk_contract"][
                    "resume_after_record_sha256"
                ],
                "variable_campaign": variable["content_sha256"],
                "classification": classification["content_sha256"],
                "classification_manifest": _content_hash(
                    classification["excluded_pair_manifest"]
                ),
            },
            "chunk_config_sha256": _content_hash(chunk_config),
            "chunk_contract": {
                "pair_selector": "excluded_obligation_ascending_global_pair_index",
                "selector_pair_count": len(all_remaining),
                "chunk_offset": offset,
                "requested_chunk_size": size,
                "evaluated_chunk_size": evaluated,
                "chunk_seed_sha256": seed,
                "resume_after_record_sha256": previous,
                "prior_resume_sha256": prior["chunk_contract"][
                    "resume_after_record_sha256"
                ],
                "stopped_at_first_obstruction": first_obstruction is not None,
                "global_pair_indices_are_stable": True,
                "selector_sha256": _content_hash(selected),
                "final_partial_tail": size < DEFAULT_CHUNK_SIZE,
            },
            "pair_manifest": records,
            "symbolic_pair_packets": [
                symbolic_packets[key] for key in sorted(symbolic_packets)
            ],
            "first_exact_obstruction": first_obstruction,
            "counts": {
                "full_unordered_coordinate_atom_pairs": 11781,
                "completed_canonical_active_pairs": 861,
                "classification_discharged_zero_pairs": 8245,
                "total_excluded_obligations": len(all_remaining),
                "prior_cumulative_evaluated_obligations": offset,
                "current_evaluated_obligations": evaluated,
                "cumulative_evaluated_obligations": offset + evaluated,
                "remaining_unevaluated_obligations": len(all_remaining)
                - offset
                - evaluated,
                "candidates": len(candidates),
                "current_candidate_checks": checks,
                "current_solvable_candidate_checks": solved,
                "current_obstructed_candidate_checks": checks - solved,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "scope": (
                "Only this exact obligation chunk is evaluated. TC2, B7, global H7, "
                "dyadic summation, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2ObligationContinuousServiceError) as error:
        errors.append(str(error))
        body = {
            "schema_version": metadata["schema_version"],
            "status": "reject",
            "errors": errors,
            "pair_manifest": [],
            "counts": {
                "total_excluded_obligations": TOTAL_OBLIGATIONS,
                "current_evaluated_obligations": 0,
                "remaining_unevaluated_obligations": TOTAL_OBLIGATIONS,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def _validate_service_config(
    config: dict[str, Any],
    initial_prior: dict[str, Any],
    variable: dict[str, Any],
    classification: dict[str, Any],
) -> None:
    bounds = {
        "max_chunks_per_invocation": (1, 32),
        "max_wall_seconds": (1, 86400),
        "max_artifact_bytes": (1024, 64 * 1024 * 1024),
        "max_total_service_bytes": (2048, 1024 * 1024 * 1024),
        "max_history_records": (1, 128),
    }
    for key, (lower, upper) in bounds.items():
        if not lower <= int(config.get(key, 0)) <= upper:
            raise QuarticTC2ObligationContinuousServiceError(
                f"invalid obligation service bound: {key}"
            )
    if (
        config.get("schema_version") != SERVICE_SCHEMA
        or int(config.get("start_offset", -1)) != 64
        or config.get("initial_prior_sha256") != initial_prior.get("content_sha256")
        or config.get("initial_prior_resume_sha256")
        != initial_prior.get("chunk_contract", {}).get("resume_after_record_sha256")
        or config.get("variable_campaign_sha256") != variable.get("content_sha256")
        or config.get("classification_content_sha256")
        != classification.get("content_sha256")
        or config.get("classification_manifest_sha256")
        != _content_hash(classification.get("excluded_pair_manifest", []))
        or not _content_hash_matches(initial_prior)
        or not _validate_record_chain(initial_prior)
        or not _content_hash_matches(variable)
        or not _content_hash_matches(classification)
        or initial_prior.get("status")
        != "pass_first_64_excluded_obligations_no_obstruction_remaining_fail_closed"
        or any(
            config.get(policy) != "fail_closed"
            for policy in (
                "global_TC2_policy",
                "B7_policy",
                "global_H7_policy",
                "lifespan_policy",
            )
        )
    ):
        raise QuarticTC2ObligationContinuousServiceError(
            "unsupported initial obligation service contract"
        )


def _validate_result(
    result: dict[str, Any],
    prior: dict[str, Any],
    classification: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    offset, size = metadata["offset"], metadata["size"]
    counts, contract = result.get("counts", {}), result.get("chunk_contract", {})
    evaluated = counts.get("current_evaluated_obligations")
    success = result.get("status") == metadata["success_status"]
    expected_status = {metadata["success_status"], metadata["obstruction_status"]}
    exact_candidates = all(
        item.get("solvable")
        and item.get("Hermitian")
        and item.get("second_Sylvester_residual_zero")
        for record in result.get("pair_manifest", [])
        for item in record.get("candidate_results", [])
    )
    if (
        not _content_hash_matches(result)
        or result.get("status") not in expected_status
        or result.get("schema_version") != metadata["schema_version"]
        or result.get("upstream_sha256", {}).get("prior_artifact")
        != prior["content_sha256"]
        or result.get("upstream_sha256", {}).get("classification")
        != classification["content_sha256"]
        or contract.get("chunk_offset") != offset
        or contract.get("requested_chunk_size") != size
        or contract.get("evaluated_chunk_size") != evaluated
        or not isinstance(evaluated, int)
        or not 0 < evaluated <= size
        or len(result.get("pair_manifest", [])) != evaluated
        or not _validate_record_chain(result)
        or counts.get("cumulative_evaluated_obligations") != offset + evaluated
        or counts.get("remaining_unevaluated_obligations")
        != TOTAL_OBLIGATIONS - offset - evaluated
        or counts.get("TC2_closures") != 0
        or counts.get("global_H7_closures") != 0
        or counts.get("lifespans_proved") != 0
        or (success and (evaluated != size or not exact_candidates))
        or (result.get("first_exact_obstruction") is not None) == success
    ):
        raise QuarticTC2ObligationContinuousServiceError(
            "obligation executor result contract mismatch"
        )


def _disk_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_sha256": _content_hash(config),
        "initial_prior_sha256": config["initial_prior_sha256"],
        "variable_campaign_sha256": config["variable_campaign_sha256"],
        "classification_content_sha256": config["classification_content_sha256"],
        "classification_manifest_sha256": config[
            "classification_manifest_sha256"
        ],
        "next_offset": int(config["start_offset"]),
        "prior_resume_sha256": config["initial_prior_resume_sha256"],
        "current_artifact_path": None,
        "current_artifact_content_sha256": config["initial_prior_sha256"],
        "current_artifact_file_sha256": None,
        "completed_service_chunks": 0,
        "permanently_stopped": False,
        "stop_reason": None,
        "history": [],
        "claims": {
            "global_TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
    }
    return _with_checkpoint_hash(body)


def _load_state(checkpoint: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not checkpoint.exists():
        return _initial_state(config)
    state = _load(checkpoint)
    if (
        state.get("schema_version") != CHECKPOINT_SCHEMA
        or not _checkpoint_hash_matches(state)
        or state.get("config_sha256") != _content_hash(config)
        or len(state.get("history", [])) > int(config["max_history_records"])
        or any(state.get("claims", {}).values())
    ):
        raise QuarticTC2ObligationContinuousServiceError(
            "obligation checkpoint contract mismatch"
        )
    return state


def _prior_from_state(
    state: dict[str, Any], output: Path, initial_prior: dict[str, Any]
) -> dict[str, Any]:
    if not state["history"]:
        return initial_prior
    path = (output / state["current_artifact_path"]).resolve()
    if output.resolve() not in path.parents:
        raise QuarticTC2ObligationContinuousServiceError(
            "obligation checkpoint artifact escaped output root"
        )
    data = path.read_bytes()
    artifact = json.loads(data)
    if (
        _file_sha256(data) != state["current_artifact_file_sha256"]
        or artifact.get("content_sha256") != state["current_artifact_content_sha256"]
        or not _content_hash_matches(artifact)
        or not _validate_record_chain(artifact)
    ):
        raise QuarticTC2ObligationContinuousServiceError(
            "obligation checkpoint artifact hash mismatch"
        )
    return artifact


def run_obligation_continuous_service(
    initial_prior: dict[str, Any],
    variable: dict[str, Any],
    classification: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    *,
    executor: Executor = exact_obligation_executor,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    _validate_service_config(config, initial_prior, variable, classification)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    state = _load_state(checkpoint_path, config)
    prior = _prior_from_state(state, output, initial_prior)
    if state["permanently_stopped"]:
        return {
            "status": "already_stopped",
            "reason": state["stop_reason"],
            "chunks_advanced": 0,
            "checkpoint": state,
        }
    start, advanced, reason = monotonic(), 0, "chunk_limit"
    while advanced < int(config["max_chunks_per_invocation"]):
        if monotonic() - start >= float(config["max_wall_seconds"]):
            reason = "wall_time_limit"
            break
        offset = int(state["next_offset"])
        if offset >= TOTAL_OBLIGATIONS:
            reason = "obligation_selector_complete_global_scope_open"
            break
        if len(state["history"]) >= int(config["max_history_records"]):
            reason = "history_limit"
            break
        size = min(DEFAULT_CHUNK_SIZE, TOTAL_OBLIGATIONS - offset)
        metadata = _metadata(offset, size)
        relative = Path("chunks") / f"offset-{offset:06d}.json"
        artifact_path = output / relative
        if artifact_path.exists():
            result = _load(artifact_path)
            _validate_result(result, prior, classification, metadata)
        else:
            result = executor(
                prior,
                variable,
                classification,
                _chunk_config(prior, classification, offset, size),
                metadata,
            )
            _validate_result(result, prior, classification, metadata)
        artifact_data = _json_bytes(result)
        if len(artifact_data) > int(config["max_artifact_bytes"]):
            raise QuarticTC2ObligationContinuousServiceError(
                "obligation artifact byte limit exceeded"
            )
        existing = artifact_path.stat().st_size if artifact_path.exists() else 0
        if (
            _disk_bytes(output) - existing + len(artifact_data)
            > int(config["max_total_service_bytes"])
        ):
            raise QuarticTC2ObligationContinuousServiceError(
                "obligation service disk limit exceeded"
            )
        _atomic_write(artifact_path, artifact_data)
        obstruction = result["first_exact_obstruction"] is not None
        history = {
            "offset": offset,
            "size": size,
            "artifact_path": relative.as_posix(),
            "artifact_content_sha256": result["content_sha256"],
            "artifact_file_sha256": _file_sha256(artifact_data),
            "prior_artifact_sha256": prior["content_sha256"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "resume_after_record_sha256": result["chunk_contract"][
                "resume_after_record_sha256"
            ],
            "status": result["status"],
            "obstruction": obstruction,
        }
        state = _with_checkpoint_hash(
            {
                **{key: value for key, value in state.items() if key != "content_sha256"},
                "next_offset": result["counts"]["cumulative_evaluated_obligations"],
                "prior_resume_sha256": result["chunk_contract"][
                    "resume_after_record_sha256"
                ],
                "current_artifact_path": relative.as_posix(),
                "current_artifact_content_sha256": result["content_sha256"],
                "current_artifact_file_sha256": _file_sha256(artifact_data),
                "completed_service_chunks": state["completed_service_chunks"] + 1,
                "permanently_stopped": obstruction,
                "stop_reason": "exact_obstruction" if obstruction else None,
                "history": [*state["history"], history],
            }
        )
        checkpoint_data = _json_bytes(state)
        existing = checkpoint_path.stat().st_size if checkpoint_path.exists() else 0
        if (
            _disk_bytes(output) - existing + len(checkpoint_data)
            > int(config["max_total_service_bytes"])
        ):
            raise QuarticTC2ObligationContinuousServiceError(
                "obligation checkpoint disk limit exceeded"
            )
        _atomic_write(checkpoint_path, checkpoint_data)
        prior = result
        advanced += 1
        if obstruction:
            reason = "exact_obstruction"
            break
    return {
        "status": "stopped" if state["permanently_stopped"] else "checkpointed",
        "reason": reason,
        "chunks_advanced": advanced,
        "next_offset": state["next_offset"],
        "prior_resume_sha256": state["prior_resume_sha256"],
        "checkpoint": state,
    }
