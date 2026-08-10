from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
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
from .quartic_tc2_second_atom_continuation_engine import (
    run_second_atom_continuation,
)
from .quartic_tc2_variable_sylvester_campaign import (
    ATOM_DIMENSION,
    _content_hash,
    _content_hash_matches,
)

SERVICE_SCHEMA = "sigma-quartic-tc2-continuous-service-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-continuous-checkpoint-1.0"
PAIR_SELECTOR = "canonical_sylvester_active_affine_second_atom_pairs"

Executor = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    dict[str, Any],
]


class QuarticTC2ContinuousServiceError(ValueError):
    """Raised when a continuation checkpoint cannot be advanced exactly."""


def _json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _checkpoint_body(state: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in state.items() if key != "content_sha256"}


def _checkpoint_hash_matches(state: dict[str, Any]) -> bool:
    return state.get("content_sha256") == _content_hash(_checkpoint_body(state))


def _with_checkpoint_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _content_hash(body)}


def _status(offset: int) -> str:
    return f"pass_cumulative_{offset}_second_atom_pairs_no_obstruction_remaining_fail_closed"


def _chunk_metadata(
    offset: int, chunk_size: int = DEFAULT_CHUNK_SIZE
) -> dict[str, Any]:
    partial = chunk_size != DEFAULT_CHUNK_SIZE
    return {
        "schema_version": (
            f"sigma-quartic-tc2-continuous-partial-tail-{offset}-size-{chunk_size}-1.0"
            if partial
            else f"sigma-quartic-tc2-continuous-chunk-{offset}-1.0"
        ),
        "expected_prior_status": _status(offset),
        "success_status": _status(offset + chunk_size),
        "obstruction_status": (
            f"exact_second_atom_Sylvester_obstruction_found_in_continuous_chunk_"
            f"{offset}_global_H7_fail_closed"
        ),
        "chunk_size": chunk_size,
        "partial_tail": partial,
    }


def exact_continuation_executor(
    prior: dict[str, Any],
    variable: dict[str, Any],
    chunk_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    if metadata["partial_tail"]:
        return _run_exact_partial_tail(prior, variable, chunk_config, metadata)
    return run_second_atom_continuation(
        prior,
        variable,
        chunk_config,
        schema_version=metadata["schema_version"],
        chunk_offset=int(chunk_config["chunk_offset"]),
        expected_prior_status=metadata["expected_prior_status"],
        success_status=metadata["success_status"],
        obstruction_status=metadata["obstruction_status"],
    )


def _run_exact_partial_tail(
    prior: dict[str, Any],
    variable: dict[str, Any],
    chunk_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    offset = int(chunk_config["chunk_offset"])
    chunk_size = int(chunk_config["chunk_size"])
    all_pairs = _canonical_active_affine_pairs()
    errors: list[str] = []
    try:
        prior_contract = prior["chunk_contract"]
        if (
            chunk_config.get("schema_version") != metadata["schema_version"]
            or prior.get("status") != metadata["expected_prior_status"]
            or variable.get("status")
            != (
                "pass_all_12_first_order_variable_deltaK_extensions_"
                "higher_orders_global_H7_fail_closed"
            )
            or not _content_hash_matches(prior)
            or not _content_hash_matches(variable)
            or prior["upstream_sha256"]["variable_Sylvester"]
            != variable["content_sha256"]
            or prior_contract["chunk_offset"] != offset - DEFAULT_CHUNK_SIZE
            or prior_contract["evaluated_chunk_size"] != DEFAULT_CHUNK_SIZE
            or prior["counts"]["cumulative_evaluated_coordinate_atom_pairs"]
            != offset
            or chunk_config.get("pair_selector") != PAIR_SELECTOR
            or chunk_config.get("resume_policy") != "require_prior_hash_chain_tip"
            or chunk_config.get("prior_resume_sha256")
            != prior_contract["resume_after_record_sha256"]
            or chunk_config.get("global_H7_policy") != "fail_closed"
            or chunk_config.get("lifespan_policy") != "fail_closed"
            or chunk_size <= 0
            or chunk_size >= DEFAULT_CHUNK_SIZE
            or offset + chunk_size != len(all_pairs)
        ):
            raise QuarticTC2ContinuousServiceError(
                "unsupported exact partial-tail contract"
            )
        prior_chain_passed, prior_chain = _validate_prior_chain(prior)
        generic_passed, generic = generic_second_atom_sylvester_control()
        if not prior_chain_passed or not generic_passed:
            raise QuarticTC2ContinuousServiceError(
                "partial-tail prior/generic control failed"
            )

        selected = all_pairs[offset:]
        if len(selected) != chunk_size:
            raise QuarticTC2ContinuousServiceError(
                "partial-tail selector length mismatch"
            )
        coefficients_by_candidate = _candidate_coefficients(variable)
        if len(coefficients_by_candidate) != 12:
            raise QuarticTC2ContinuousServiceError("candidate count mismatch")
        seed = _content_hash(
            {
                "prior_artifact": prior["content_sha256"],
                "prior_resume": prior_contract["resume_after_record_sha256"],
                "variable_campaign": variable["content_sha256"],
                "selector": PAIR_SELECTOR,
                "offset": offset,
                "size": chunk_size,
                "partial_tail_exact": True,
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
                candidate_results.append(
                    {
                        "candidate_id": candidate_id,
                        "solvable": solvable,
                        "compression_residual_sha256": _content_hash(compressions),
                        "deltaK_AB_nonzero_entries": (
                            len(delta_entries) if solvable else 0
                        ),
                        "deltaK_AB_sha256": (
                            _content_hash(delta_entries) if solvable else None
                        ),
                        "Hermitian": (
                            symbolic["deltaK_AB_Hermitian"] if solvable else False
                        ),
                        "second_Sylvester_residual_zero": (
                            symbolic["second_Sylvester_residual_zero"]
                            if solvable
                            else False
                        ),
                    }
                )
                if not solvable and first_obstruction is None:
                    eigenvalue, entries = next(iter(compressions.items()))
                    first_obstruction = {
                        "chunk_local_index": local_index,
                        "selector_pair_index": offset + local_index,
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
                "selector_pair_index": offset + local_index,
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

        evaluated = len(manifest)
        cumulative = offset + evaluated
        total = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
        candidate_checks = evaluated * len(coefficients_by_candidate)
        solvable = sum(
            result["solvable"]
            for record in manifest
            for result in record["candidate_results"]
        )
        unique_packets = [symbolic_packets[key] for key in sorted(symbolic_packets)]
        body = {
            "schema_version": metadata["schema_version"],
            "status": (
                metadata["success_status"]
                if first_obstruction is None
                else metadata["obstruction_status"]
            ),
            "errors": [],
            "upstream_sha256": {
                "prior_chunk": prior["content_sha256"],
                "prior_resume": prior_contract["resume_after_record_sha256"],
                "variable_Sylvester": variable["content_sha256"],
                "coordinate_to_jet_packet": variable[
                    "common_coordinate_to_covariant_jet_packet"
                ]["content_sha256"],
            },
            "config_sha256": _content_hash(chunk_config),
            "generic_second_atom_control": generic,
            "partial_tail_control": {
                "canonical_active_selector_count": len(all_pairs),
                "tail_offset": offset,
                "tail_size": chunk_size,
                "tail_exhausts_selector_exactly": offset + chunk_size
                == len(all_pairs),
                "full_coordinate_pair_denominator": total,
                "unevaluated_coordinate_pairs_after_tail": total - len(all_pairs),
                "unevaluated_pairs_inferred": 0,
                "passed": True,
            },
            "verified_prior_chain": prior_chain,
            "chunk_contract": {
                "pair_selector": PAIR_SELECTOR,
                "selector_pair_count": len(all_pairs),
                "chunk_offset": offset,
                "requested_chunk_size": chunk_size,
                "evaluated_chunk_size": evaluated,
                "chunk_seed_sha256": seed,
                "resume_after_record_sha256": previous,
                "prior_resume_sha256": prior_contract[
                    "resume_after_record_sha256"
                ],
                "stopped_at_first_obstruction": first_obstruction is not None,
                "global_pair_indices_are_stable": True,
                "final_partial_canonical_active_tail": True,
            },
            "pair_manifest": manifest,
            "symbolic_pair_packets": unique_packets,
            "exact_tensor_summary_current_chunk": _tensor_summary(unique_packets),
            "first_exact_obstruction": first_obstruction,
            "counts": {
                "total_unordered_coordinate_atom_pairs": total,
                "canonical_active_selector_pairs": len(all_pairs),
                "prior_cumulative_evaluated_coordinate_atom_pairs": offset,
                "current_evaluated_coordinate_atom_pairs": evaluated,
                "cumulative_evaluated_coordinate_atom_pairs": cumulative,
                "remaining_unevaluated_coordinate_atom_pairs": total - cumulative,
                "candidates": len(coefficients_by_candidate),
                "current_evaluated_candidate_pairs": candidate_checks,
                "current_solvable_candidate_pairs": solvable,
                "current_obstructed_candidate_pairs": candidate_checks - solvable,
                "cumulative_deltaK_AB_constructions": int(
                    prior["counts"]["cumulative_deltaK_AB_constructions"]
                )
                + solvable,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "claim": (
                f"The exact final {chunk_size}-pair canonical-active selector tail was "
                "evaluated until completion or first obstruction. No other coordinate "
                "pairs are inferred."
            ),
            "scope": (
                f"Canonical-active second-order Sylvester algebra through selector index "
                f"{cumulative - 1} only; {total - cumulative} coordinate-atom pairs remain "
                "unevaluated. TC2, B7, global H7, dyadic summation, and lifespan remain "
                "fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTC2ContinuousServiceError) as error:
        errors.append(str(error))
        total = ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
        body = {
            "schema_version": metadata["schema_version"],
            "status": "reject",
            "errors": errors,
            "pair_manifest": [],
            "symbolic_pair_packets": [],
            "counts": {
                "total_unordered_coordinate_atom_pairs": total,
                "prior_cumulative_evaluated_coordinate_atom_pairs": 0,
                "current_evaluated_coordinate_atom_pairs": 0,
                "cumulative_evaluated_coordinate_atom_pairs": 0,
                "remaining_unevaluated_coordinate_atom_pairs": total,
                "TC2_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def _validate_config(
    config: dict[str, Any], initial_prior: dict[str, Any], variable: dict[str, Any]
) -> None:
    if config.get("schema_version") != SERVICE_SCHEMA:
        raise QuarticTC2ContinuousServiceError("unsupported service schema_version")
    numeric_bounds = {
        "max_chunks_per_invocation": (1, 32),
        "max_wall_seconds": (1, 86400),
        "max_artifact_bytes": (1024, 64 * 1024 * 1024),
        "max_total_service_bytes": (2048, 1024 * 1024 * 1024),
        "max_history_records": (1, 1024),
    }
    for key, (lower, upper) in numeric_bounds.items():
        value = int(config.get(key, 0))
        if not lower <= value <= upper:
            raise QuarticTC2ContinuousServiceError(f"invalid service bound: {key}")
    start_offset = int(config.get("start_offset", -1))
    contract = initial_prior.get("chunk_contract", {})
    if (
        int(config.get("chunk_size", 0)) != DEFAULT_CHUNK_SIZE
        or config.get("pair_selector") != PAIR_SELECTOR
        or config.get("global_TC2_policy") != "fail_closed"
        or config.get("B7_policy") != "fail_closed"
        or config.get("global_H7_policy") != "fail_closed"
        or config.get("lifespan_policy") != "fail_closed"
        or not _content_hash_matches(initial_prior)
        or not _content_hash_matches(variable)
        or initial_prior.get("status") != _status(start_offset)
        or initial_prior.get("content_sha256") != config.get("initial_prior_sha256")
        or variable.get("content_sha256") != config.get("variable_campaign_sha256")
        or contract.get("resume_after_record_sha256")
        != config.get("initial_prior_resume_sha256")
        or initial_prior.get("counts", {}).get(
            "cumulative_evaluated_coordinate_atom_pairs"
        )
        != start_offset
    ):
        raise QuarticTC2ContinuousServiceError("unsupported initial service contract")


def _validate_result(
    result: dict[str, Any],
    prior: dict[str, Any],
    offset: int,
    metadata: dict[str, Any],
) -> None:
    expected = {metadata["success_status"], metadata["obstruction_status"]}
    contract = result.get("chunk_contract", {})
    counts = result.get("counts", {})
    evaluated = counts.get("current_evaluated_coordinate_atom_pairs")
    success = result.get("status") == metadata["success_status"]
    manifest = result.get("pair_manifest", [])
    exact_candidate_pass = all(
        candidate.get("solvable")
        and candidate.get("Hermitian")
        and candidate.get("second_Sylvester_residual_zero")
        for record in manifest
        for candidate in record.get("candidate_results", [])
    )
    if (
        not _content_hash_matches(result)
        or result.get("schema_version") != metadata["schema_version"]
        or result.get("status") not in expected
        or result.get("upstream_sha256", {}).get("prior_chunk")
        != prior.get("content_sha256")
        or result.get("upstream_sha256", {}).get("prior_resume")
        != prior.get("chunk_contract", {}).get("resume_after_record_sha256")
        or contract.get("chunk_offset") != offset
        or contract.get("requested_chunk_size") != metadata["chunk_size"]
        or contract.get("evaluated_chunk_size") != evaluated
        or counts.get("prior_cumulative_evaluated_coordinate_atom_pairs") != offset
        or counts.get("total_unordered_coordinate_atom_pairs")
        != ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2
        or not isinstance(evaluated, int)
        or not 0 < evaluated <= metadata["chunk_size"]
        or counts.get("cumulative_evaluated_coordinate_atom_pairs")
        != offset + evaluated
        or counts.get("remaining_unevaluated_coordinate_atom_pairs")
        != ATOM_DIMENSION * (ATOM_DIMENSION + 1) // 2 - offset - evaluated
        or len(manifest) != evaluated
        or (success and evaluated != metadata["chunk_size"])
        or (success and not exact_candidate_pass)
        or counts.get("TC2_closures") != 0
        or counts.get("global_H7_closures") != 0
        or counts.get("lifespans_proved") != 0
    ):
        raise QuarticTC2ContinuousServiceError("executor result contract mismatch")
    obstructed = result.get("status") == metadata["obstruction_status"]
    if obstructed != bool(result.get("first_exact_obstruction")):
        raise QuarticTC2ContinuousServiceError("obstruction status/payload mismatch")


def _service_disk_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_from_checkpoint(
    state: dict[str, Any], output: Path, initial_prior: dict[str, Any]
) -> dict[str, Any]:
    if not state["history"]:
        return initial_prior
    relative = Path(state["current_artifact_path"])
    artifact_path = (output / relative).resolve()
    if output.resolve() not in artifact_path.parents:
        raise QuarticTC2ContinuousServiceError("checkpoint artifact escaped output root")
    data = artifact_path.read_bytes()
    artifact = json.loads(data)
    if (
        _file_sha256(data) != state["current_artifact_file_sha256"]
        or artifact.get("content_sha256") != state["current_artifact_content_sha256"]
        or not _content_hash_matches(artifact)
    ):
        raise QuarticTC2ContinuousServiceError("checkpoint artifact hash mismatch")
    return artifact


def _initial_state(config: dict[str, Any], initial_prior: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_sha256": _content_hash(config),
        "variable_campaign_sha256": config["variable_campaign_sha256"],
        "initial_prior_sha256": config["initial_prior_sha256"],
        "next_offset": int(config["start_offset"]),
        "prior_resume_sha256": config["initial_prior_resume_sha256"],
        "current_artifact_path": None,
        "current_artifact_content_sha256": initial_prior["content_sha256"],
        "current_artifact_file_sha256": None,
        "completed_chunks": 0,
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


def _load_state(
    checkpoint: Path, config: dict[str, Any], initial_prior: dict[str, Any]
) -> dict[str, Any]:
    if not checkpoint.exists():
        return _initial_state(config, initial_prior)
    state = _load_json(checkpoint)
    if (
        state.get("schema_version") != CHECKPOINT_SCHEMA
        or not _checkpoint_hash_matches(state)
        or state.get("config_sha256") != _content_hash(config)
        or state.get("variable_campaign_sha256")
        != config["variable_campaign_sha256"]
        or state.get("initial_prior_sha256") != config["initial_prior_sha256"]
        or len(state.get("history", [])) > int(config["max_history_records"])
        or any(state.get("claims", {}).values())
    ):
        raise QuarticTC2ContinuousServiceError("checkpoint contract mismatch")
    return state


def _chunk_config(
    config: dict[str, Any],
    prior: dict[str, Any],
    offset: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    metadata = _chunk_metadata(offset, chunk_size)
    return {
        "schema_version": metadata["schema_version"],
        "chunk_offset": offset,
        "chunk_size": chunk_size,
        "pair_selector": PAIR_SELECTOR,
        "resume_policy": "require_prior_hash_chain_tip",
        "prior_resume_sha256": prior["chunk_contract"][
            "resume_after_record_sha256"
        ],
        "global_H7_policy": config["global_H7_policy"],
        "lifespan_policy": config["lifespan_policy"],
    }


def _pending_artifact(
    path: Path,
    prior: dict[str, Any],
    offset: int,
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = _load_json(path)
    _validate_result(result, prior, offset, metadata)
    return result


def run_continuous_tc2_service(
    initial_prior: dict[str, Any],
    variable: dict[str, Any],
    config: dict[str, Any],
    output: Path,
    *,
    executor: Executor = exact_continuation_executor,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    _validate_config(config, initial_prior, variable)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "checkpoint.json"
    state = _load_state(checkpoint, config, initial_prior)
    prior = _artifact_from_checkpoint(state, output, initial_prior)
    if state["permanently_stopped"]:
        return {
            "status": "already_stopped",
            "reason": state["stop_reason"],
            "checkpoint": state,
            "chunks_advanced": 0,
        }

    start = monotonic()
    advanced = 0
    reason = "chunk_limit"
    while advanced < int(config["max_chunks_per_invocation"]):
        elapsed = monotonic() - start
        if elapsed >= float(config["max_wall_seconds"]):
            reason = "wall_time_limit"
            break
        offset = int(state["next_offset"])
        selector_remaining = len(_canonical_active_affine_pairs()) - offset
        if selector_remaining <= 0:
            reason = "canonical_active_selector_complete_global_scope_open"
            break
        if len(state["history"]) >= int(config["max_history_records"]):
            reason = "history_limit"
            break

        relative = Path("chunks") / f"offset-{offset:06d}.json"
        artifact_path = output / relative
        chunk_size = min(DEFAULT_CHUNK_SIZE, selector_remaining)
        metadata = _chunk_metadata(offset, chunk_size)
        result = _pending_artifact(artifact_path, prior, offset, metadata)
        if result is None:
            result = executor(
                prior,
                variable,
                _chunk_config(config, prior, offset, chunk_size),
                metadata,
            )
            _validate_result(result, prior, offset, metadata)
        artifact_data = _json_bytes(result)
        if len(artifact_data) > int(config["max_artifact_bytes"]):
            raise QuarticTC2ContinuousServiceError("artifact byte limit exceeded")
        existing_size = artifact_path.stat().st_size if artifact_path.exists() else 0
        projected = _service_disk_bytes(output) - existing_size + len(artifact_data)
        if projected > int(config["max_total_service_bytes"]):
            raise QuarticTC2ContinuousServiceError("service disk limit exceeded")
        _atomic_write(artifact_path, artifact_data)

        contract = result["chunk_contract"]
        obstruction = result["first_exact_obstruction"] is not None
        history_record = {
            "offset": offset,
            "artifact_path": relative.as_posix(),
            "artifact_content_sha256": result["content_sha256"],
            "artifact_file_sha256": _file_sha256(artifact_data),
            "prior_content_sha256": prior["content_sha256"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "resume_after_record_sha256": contract["resume_after_record_sha256"],
            "status": result["status"],
            "obstruction": obstruction,
        }
        state_body = {
            **_checkpoint_body(state),
            "next_offset": int(result["counts"]["cumulative_evaluated_coordinate_atom_pairs"]),
            "prior_resume_sha256": contract["resume_after_record_sha256"],
            "current_artifact_path": relative.as_posix(),
            "current_artifact_content_sha256": result["content_sha256"],
            "current_artifact_file_sha256": _file_sha256(artifact_data),
            "completed_chunks": int(state["completed_chunks"]) + 1,
            "permanently_stopped": obstruction,
            "stop_reason": "exact_obstruction" if obstruction else None,
            "history": [*state["history"], history_record],
        }
        state = _with_checkpoint_hash(state_body)
        checkpoint_data = _json_bytes(state)
        projected = _service_disk_bytes(output) + len(checkpoint_data)
        if checkpoint.exists():
            projected -= checkpoint.stat().st_size
        if projected > int(config["max_total_service_bytes"]):
            raise QuarticTC2ContinuousServiceError("checkpoint disk limit exceeded")
        _atomic_write(checkpoint, checkpoint_data)
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
