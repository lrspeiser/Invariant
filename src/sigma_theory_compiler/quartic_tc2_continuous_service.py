from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .quartic_tc2_second_atom_chunk_campaign import (
    DEFAULT_CHUNK_SIZE,
    _canonical_active_affine_pairs,
)
from .quartic_tc2_second_atom_continuation_engine import (
    run_second_atom_continuation,
)
from .quartic_tc2_variable_sylvester_campaign import (
    _content_hash,
    _content_hash_matches,
)

SERVICE_SCHEMA = "sigma-quartic-tc2-continuous-service-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-continuous-checkpoint-1.0"
PAIR_SELECTOR = "canonical_sylvester_active_affine_second_atom_pairs"

Executor = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]],
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


def _chunk_metadata(offset: int) -> dict[str, str]:
    return {
        "schema_version": f"sigma-quartic-tc2-continuous-chunk-{offset}-1.0",
        "expected_prior_status": _status(offset),
        "success_status": _status(offset + DEFAULT_CHUNK_SIZE),
        "obstruction_status": (
            f"exact_second_atom_Sylvester_obstruction_found_in_continuous_chunk_"
            f"{offset}_global_H7_fail_closed"
        ),
    }


def exact_continuation_executor(
    prior: dict[str, Any],
    variable: dict[str, Any],
    chunk_config: dict[str, Any],
    metadata: dict[str, str],
) -> dict[str, Any]:
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
    metadata: dict[str, str],
) -> None:
    expected = {metadata["success_status"], metadata["obstruction_status"]}
    contract = result.get("chunk_contract", {})
    counts = result.get("counts", {})
    if (
        not _content_hash_matches(result)
        or result.get("schema_version") != metadata["schema_version"]
        or result.get("status") not in expected
        or result.get("upstream_sha256", {}).get("prior_chunk")
        != prior.get("content_sha256")
        or result.get("upstream_sha256", {}).get("prior_resume")
        != prior.get("chunk_contract", {}).get("resume_after_record_sha256")
        or contract.get("chunk_offset") != offset
        or counts.get("prior_cumulative_evaluated_coordinate_atom_pairs") != offset
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


def _chunk_config(config: dict[str, Any], prior: dict[str, Any], offset: int) -> dict[str, Any]:
    metadata = _chunk_metadata(offset)
    return {
        "schema_version": metadata["schema_version"],
        "chunk_offset": offset,
        "chunk_size": DEFAULT_CHUNK_SIZE,
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
    metadata: dict[str, str],
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
        if offset + DEFAULT_CHUNK_SIZE > len(_canonical_active_affine_pairs()):
            reason = "selector_tail_requires_partial_chunk_engine"
            break
        if len(state["history"]) >= int(config["max_history_records"]):
            reason = "history_limit"
            break

        relative = Path("chunks") / f"offset-{offset:06d}.json"
        artifact_path = output / relative
        metadata = _chunk_metadata(offset)
        result = _pending_artifact(artifact_path, prior, offset, metadata)
        if result is None:
            result = executor(
                prior, variable, _chunk_config(config, prior, offset), metadata
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
