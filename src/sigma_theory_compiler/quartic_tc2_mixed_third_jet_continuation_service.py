from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from .quartic_tc2_mixed_third_jet_chunk_campaign import (
    DEFAULT_CHUNK_SIZE,
    TOTAL_MIXED_TRIPLES,
    _content_hash,
    _content_hash_matches,
    _mixed_selector,
    _record_hash_matches,
    _triple_kind,
    run_quartic_tc2_mixed_third_jet_chunk_campaign,
)
from .quartic_tc2_mixed_third_jet_chunk_campaign import (
    SCHEMA_VERSION as CHUNK_SCHEMA,
)

SERVICE_SCHEMA = "sigma-quartic-tc2-mixed-third-jet-continuation-service-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-mixed-third-jet-continuation-checkpoint-1.0"
STATUS_SCHEMA = "sigma-quartic-tc2-mixed-third-jet-continuation-status-1.0"
SELECTOR = "lexicographic_active_direction_multisets_excluding_AAA"

Executor = Callable[
    [dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]],
    dict[str, Any],
]


class QuarticTC2MixedThirdJetContinuationServiceError(ValueError):
    """Raised when the durable mixed-third-jet service cannot advance exactly."""


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


def _body(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "content_sha256"}


def _hash_matches(value: dict[str, Any]) -> bool:
    return value.get("content_sha256") == _content_hash(_body(value))


def _with_hash(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _content_hash(body)}


def _load_file(path: Path) -> tuple[dict[str, Any], bytes]:
    data = path.read_bytes()
    return json.loads(data), data


def _chunk_status(size: int) -> str:
    return (
        "pass_mixed_third_jet_chunk_64_global_closure_fail_closed"
        if size == DEFAULT_CHUNK_SIZE
        else f"pass_mixed_third_jet_exact_partial_tail_{size}_global_closure_fail_closed"
    )


def _chunk_config(
    config: dict[str, Any], prior_resume: str, offset: int, size: int
) -> dict[str, Any]:
    return {
        "schema_version": CHUNK_SCHEMA,
        "expected_upstream_content_sha256": {
            "diagonal_third_jet": config["diagonal_third_jet_content_sha256"],
            "quadratic_deltaK": config["quadratic_deltaK_content_sha256"],
        },
        "selector": SELECTOR,
        "chunk_offset": offset,
        "chunk_size": size,
        "expected_prior_resume_sha256": prior_resume,
        "resume_policy": "record_sha256_chain",
        "stop_on_first_obstruction": True,
        "unprocessed_mixed_third_jet_policy": "fail_closed",
        "full_tube_policy": "fail_closed",
        "CK1_policy": "fail_closed",
        "CK3_policy": "fail_closed",
        "TC2_policy": "fail_closed",
        "B7_policy": "fail_closed",
        "global_H7_policy": "fail_closed",
        "lifespan_policy": "fail_closed",
    }


def exact_mixed_third_jet_executor(
    diagonal: dict[str, Any],
    quadratic: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
    chunk_config: dict[str, Any],
) -> dict[str, Any]:
    return run_quartic_tc2_mixed_third_jet_chunk_campaign(
        diagonal, quadratic, canonical_artifacts, chunk_config
    )


def _validate_record_chain(
    artifact: dict[str, Any], *, expected_offset: int, expected_size: int
) -> None:
    contract = artifact.get("chunk_contract", {})
    manifest = artifact.get("triple_manifest", [])
    processed = contract.get("processed_count")
    if (
        not isinstance(processed, int)
        or contract.get("selector") != SELECTOR
        or contract.get("chunk_offset") != expected_offset
        or contract.get("requested_chunk_size") != expected_size
        or processed != len(manifest)
        or not 0 < processed <= expected_size
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError(
            "mixed chunk record-chain contract mismatch"
        )
    selector = _mixed_selector()
    expected_seed = _content_hash(
        {
            "upstream": artifact.get("upstream_sha256"),
            "canonical_D2_artifact_sequence_sha256": artifact.get(
                "canonical_D2_artifact_sequence_sha256"
            ),
            "selector": SELECTOR,
            "chunk_offset": expected_offset,
            "prior_resume_sha256": contract.get("prior_resume_sha256"),
        }
    )
    previous = contract.get("resume_seed_sha256")
    if previous != expected_seed:
        raise QuarticTC2MixedThirdJetContinuationServiceError("mixed chunk resume seed mismatch")
    for local_index, record in enumerate(manifest):
        selector_index = expected_offset + local_index
        if (
            not _record_hash_matches(record)
            or record.get("chunk_index") != local_index
            or record.get("selector_index") != selector_index
            or record.get("active_position_triple") != list(selector[selector_index])
            or record.get("triple_kind") != _triple_kind(selector[selector_index])
            or record.get("previous_record_sha256") != previous
        ):
            raise QuarticTC2MixedThirdJetContinuationServiceError(
                "mixed chunk record-chain integrity mismatch"
            )
        previous = record["record_sha256"]
    if previous != contract.get("resume_tip_sha256"):
        raise QuarticTC2MixedThirdJetContinuationServiceError("mixed chunk resume tip mismatch")


def _all_global_claims_false(ledger: dict[str, Any]) -> bool:
    return all(
        ledger.get(key) is False
        for key in (
            "all_12_300_mixed_third_jets_closed",
            "full_tube_Sylvester_identity",
            "CK1_closed",
            "CK3_closed",
            "TC2_closed",
            "B7_closed",
            "global_H7_closed",
            "lifespan_proved",
        )
    )


def _validate_initial_prior(
    artifact: dict[str, Any],
    artifact_data: bytes,
    initial_chunk_config: dict[str, Any],
    initial_config_data: bytes,
    diagonal: dict[str, Any],
    quadratic: dict[str, Any],
    config: dict[str, Any],
) -> None:
    contract = artifact.get("chunk_contract", {})
    counts = artifact.get("counts", {})
    if (
        _file_sha256(artifact_data) != config.get("initial_prior_file_sha256")
        or artifact.get("content_sha256") != config.get("initial_prior_content_sha256")
        or not _content_hash_matches(artifact)
        or _file_sha256(initial_config_data) != config.get("initial_chunk_config_file_sha256")
        or _content_hash(initial_chunk_config) != config.get("initial_chunk_config_content_sha256")
        or artifact.get("config_sha256") != _content_hash(initial_chunk_config)
        or artifact.get("status") != _chunk_status(DEFAULT_CHUNK_SIZE)
        or contract.get("chunk_offset") != 0
        or contract.get("processed_count") != DEFAULT_CHUNK_SIZE
        or contract.get("next_offset") != int(config["start_offset"])
        or contract.get("resume_tip_sha256") != config.get("initial_prior_resume_tip_sha256")
        or artifact.get("first_exact_obstruction") is not None
        or counts.get("selected") != DEFAULT_CHUNK_SIZE
        or counts.get("candidate_evaluations") != DEFAULT_CHUNK_SIZE * 12
        or counts.get("candidate_solvable") != DEFAULT_CHUNK_SIZE * 12
        or counts.get("candidate_obstructed") != 0
        or counts.get("mixed_triples_remaining") != TOTAL_MIXED_TRIPLES - DEFAULT_CHUNK_SIZE
        or artifact.get("upstream_sha256", {}).get("diagonal_third_jet")
        != diagonal.get("content_sha256")
        or artifact.get("upstream_sha256", {}).get("quadratic_deltaK")
        != quadratic.get("content_sha256")
        or artifact.get("canonical_D2_artifact_sequence_sha256")
        != config.get("canonical_D2_artifact_sequence_sha256")
        or artifact.get("closure_ledger", {}).get("processed_mixed_third_jets_closed")
        != DEFAULT_CHUNK_SIZE
        or not _all_global_claims_false(artifact.get("closure_ledger", {}))
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError(
            "initial prior artifact/config binding mismatch"
        )
    _validate_record_chain(artifact, expected_offset=0, expected_size=DEFAULT_CHUNK_SIZE)


def _validate_service_config(
    config: dict[str, Any], diagonal: dict[str, Any], quadratic: dict[str, Any]
) -> None:
    numeric_bounds = {
        "max_chunks_per_invocation": (1, 1),
        "max_wall_seconds": (1, 3600),
        "max_artifact_bytes": (1024, 64 * 1024 * 1024),
        "max_total_service_bytes": (4096, 1024 * 1024 * 1024),
        "max_history_records": (1, 512),
    }
    if config.get("schema_version") != SERVICE_SCHEMA or not _hash_matches(config):
        raise QuarticTC2MixedThirdJetContinuationServiceError("service config hash/schema mismatch")
    for key, (lower, upper) in numeric_bounds.items():
        value = int(config.get(key, 0))
        if not lower <= value <= upper:
            raise QuarticTC2MixedThirdJetContinuationServiceError(f"invalid service budget: {key}")
    if (
        int(config.get("chunk_size", 0)) != DEFAULT_CHUNK_SIZE
        or int(config.get("start_offset", -1)) != DEFAULT_CHUNK_SIZE
        or config.get("selector") != SELECTOR
        or config.get("resume_policy") != "record_sha256_chain"
        or config.get("orphan_recovery_policy") != "validate_and_adopt"
        or config.get("obstruction_policy") != "permanent_stop"
        or config.get("partial_tail_policy")
        != "evaluate_exact_remaining_selector_entries_without_padding_or_inference"
        or config.get("diagonal_third_jet_content_sha256") != diagonal.get("content_sha256")
        or config.get("quadratic_deltaK_content_sha256") != quadratic.get("content_sha256")
        or not _content_hash_matches(diagonal)
        or not _content_hash_matches(quadratic)
        or any(
            config.get(key) != "fail_closed"
            for key in (
                "CK1_policy",
                "CK3_policy",
                "TC2_policy",
                "B7_policy",
                "global_H7_policy",
                "lifespan_policy",
            )
        )
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError("unsupported service contract")


def _initial_state(config: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_content_sha256": config["content_sha256"],
        "initial_prior_file_sha256": config["initial_prior_file_sha256"],
        "initial_prior_content_sha256": config["initial_prior_content_sha256"],
        "initial_chunk_config_file_sha256": config["initial_chunk_config_file_sha256"],
        "initial_chunk_config_content_sha256": config["initial_chunk_config_content_sha256"],
        "next_offset": int(config["start_offset"]),
        "remaining_mixed_triples": TOTAL_MIXED_TRIPLES - int(config["start_offset"]),
        "prior_resume_sha256": config["initial_prior_resume_tip_sha256"],
        "current_artifact_path": None,
        "current_artifact_content_sha256": config["initial_prior_content_sha256"],
        "current_artifact_file_sha256": config["initial_prior_file_sha256"],
        "completed_chunks": 0,
        "permanently_stopped": False,
        "stop_reason": None,
        "history": [],
        "claims": {
            "full_mixed_sector_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
    }
    return _with_hash(body)


def _load_state(checkpoint_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not checkpoint_path.exists():
        return _initial_state(config)
    state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    history = state.get("history", [])
    if (
        state.get("schema_version") != CHECKPOINT_SCHEMA
        or not _hash_matches(state)
        or state.get("config_content_sha256") != config["content_sha256"]
        or state.get("initial_prior_file_sha256") != config["initial_prior_file_sha256"]
        or state.get("initial_prior_content_sha256") != config["initial_prior_content_sha256"]
        or len(history) != state.get("completed_chunks")
        or len(history) > int(config["max_history_records"])
        or any(state.get("claims", {}).values())
        or bool(state.get("permanently_stopped"))
        != (state.get("stop_reason") == "exact_obstruction")
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError("checkpoint contract mismatch")
    expected_offset = int(config["start_offset"])
    expected_remaining = TOTAL_MIXED_TRIPLES - expected_offset
    expected_prior_content = config["initial_prior_content_sha256"]
    expected_prior_file = config["initial_prior_file_sha256"]
    expected_prior_resume = config["initial_prior_resume_tip_sha256"]
    for record in history:
        processed = record.get("processed_count")
        closed = record.get("closed_count")
        if (
            not isinstance(processed, int)
            or not isinstance(closed, int)
            or not 0 < processed <= DEFAULT_CHUNK_SIZE
            or not 0 <= closed <= processed
            or record.get("offset") != expected_offset
            or record.get("next_offset") != expected_offset + processed
            or record.get("remaining_mixed_triples") != expected_remaining - closed
            or record.get("prior_content_sha256") != expected_prior_content
            or record.get("prior_resume_sha256") != expected_prior_resume
            or not isinstance(record.get("artifact_file_sha256"), str)
            or not isinstance(record.get("artifact_content_sha256"), str)
            or not isinstance(record.get("resume_tip_sha256"), str)
        ):
            raise QuarticTC2MixedThirdJetContinuationServiceError(
                "checkpoint history chain mismatch"
            )
        expected_offset = int(record["next_offset"])
        expected_remaining = int(record["remaining_mixed_triples"])
        expected_prior_content = str(record["artifact_content_sha256"])
        expected_prior_file = str(record["artifact_file_sha256"])
        expected_prior_resume = str(record["resume_tip_sha256"])
    if (
        state.get("next_offset") != expected_offset
        or state.get("remaining_mixed_triples") != expected_remaining
        or state.get("prior_resume_sha256") != expected_prior_resume
        or state.get("current_artifact_content_sha256") != expected_prior_content
        or state.get("current_artifact_file_sha256") != expected_prior_file
        or bool(history and history[-1].get("obstruction"))
        != bool(state.get("permanently_stopped"))
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError("checkpoint history tip mismatch")
    return state


def _artifact_from_state(
    state: dict[str, Any], output: Path, initial_prior: dict[str, Any]
) -> dict[str, Any]:
    if not state["history"]:
        return initial_prior
    relative = Path(str(state["current_artifact_path"]))
    artifact_path = (output / relative).resolve()
    if output.resolve() not in artifact_path.parents:
        raise QuarticTC2MixedThirdJetContinuationServiceError(
            "checkpoint artifact escaped service root"
        )
    artifact, data = _load_file(artifact_path)
    if (
        _file_sha256(data) != state["current_artifact_file_sha256"]
        or artifact.get("content_sha256") != state["current_artifact_content_sha256"]
        or not _content_hash_matches(artifact)
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError("checkpoint artifact hash mismatch")
    return artifact


def _validate_result(
    result: dict[str, Any],
    chunk_config: dict[str, Any],
    offset: int,
    requested_size: int,
    expected_canonical_sequence: str,
) -> None:
    contract = result.get("chunk_contract", {})
    counts = result.get("counts", {})
    manifest = result.get("triple_manifest", [])
    obstruction = result.get("first_exact_obstruction")
    obstructed = bool(obstruction)
    expected_status = (
        "stop_first_exact_mixed_third_jet_obstruction"
        if obstructed
        else _chunk_status(requested_size)
    )
    processed = counts.get("selected")
    closed = result.get("closure_ledger", {}).get("processed_mixed_third_jets_closed")
    if (
        not isinstance(processed, int)
        or not isinstance(closed, int)
        or not 0 < processed <= requested_size
        or not 0 <= closed <= processed
        or not _content_hash_matches(result)
        or result.get("status") != expected_status
        or result.get("config_sha256") != _content_hash(chunk_config)
        or result.get("upstream_sha256") != chunk_config["expected_upstream_content_sha256"]
        or result.get("canonical_D2_artifact_sequence_sha256") != expected_canonical_sequence
        or contract.get("prior_resume_sha256") != chunk_config["expected_prior_resume_sha256"]
        or contract.get("next_offset") != offset + processed
        or counts.get("candidate_evaluations") != processed * 12
        or counts.get("candidate_solvable") + counts.get("candidate_obstructed") != processed * 12
        or counts.get("mixed_triples_remaining") != TOTAL_MIXED_TRIPLES - offset - closed
        or not _all_global_claims_false(result.get("closure_ledger", {}))
        or any(
            counts.get(key) != 0
            for key in (
                "full_tube_Sylvester_identities",
                "TC2_closures",
                "B7_closures",
                "global_H7_closures",
                "lifespans_proved",
            )
        )
        or (not obstructed and processed != requested_size)
        or (not obstructed and counts.get("candidate_obstructed") != 0)
        or (not obstructed and closed != processed)
        or (obstructed and counts.get("candidate_obstructed", 0) <= 0)
        or (obstructed and closed >= processed)
        or len(manifest) != processed
    ):
        raise QuarticTC2MixedThirdJetContinuationServiceError("executor result contract mismatch")
    _validate_record_chain(result, expected_offset=offset, expected_size=requested_size)
    expected_kinds = Counter(
        _triple_kind(triple) for triple in _mixed_selector()[offset : offset + processed]
    )
    if counts.get("triple_kind_counts") != dict(sorted(expected_kinds.items())):
        raise QuarticTC2MixedThirdJetContinuationServiceError("executor selector-kind mismatch")
    partial = requested_size != DEFAULT_CHUNK_SIZE
    tail = result.get("partial_tail_control")
    if partial:
        if (
            offset + requested_size != TOTAL_MIXED_TRIPLES
            or contract.get("exact_final_partial_tail") is not True
            or tail
            != {
                "selector_total": TOTAL_MIXED_TRIPLES,
                "tail_offset": offset,
                "tail_size": requested_size,
                "tail_exhausts_selector_exactly": True,
                "padded_or_inferred_triples": 0,
                "passed": True,
            }
        ):
            raise QuarticTC2MixedThirdJetContinuationServiceError(
                "exact partial-tail contract mismatch"
            )
    elif tail is not None or contract.get("exact_final_partial_tail") is not None:
        raise QuarticTC2MixedThirdJetContinuationServiceError(
            "full chunk falsely marked as partial tail"
        )


def _service_disk_bytes(output: Path) -> int:
    return sum(path.stat().st_size for path in output.rglob("*") if path.is_file())


def _pending_artifact(
    path: Path,
    chunk_config: dict[str, Any],
    offset: int,
    size: int,
    expected_canonical_sequence: str,
) -> tuple[dict[str, Any], bytes] | None:
    if not path.exists():
        return None
    result, data = _load_file(path)
    _validate_result(result, chunk_config, offset, size, expected_canonical_sequence)
    return result, data


def _status_artifact(
    state: dict[str, Any], checkpoint_data: bytes, decision: str, reason: str
) -> dict[str, Any]:
    body = {
        "schema_version": STATUS_SCHEMA,
        "decision": decision,
        "reason": reason,
        "checkpoint_file_sha256": _file_sha256(checkpoint_data),
        "checkpoint_content_sha256": state["content_sha256"],
        "next_offset": state["next_offset"],
        "remaining_mixed_triples": state["remaining_mixed_triples"],
        "prior_resume_sha256": state["prior_resume_sha256"],
        "current_artifact_path": state["current_artifact_path"],
        "current_artifact_file_sha256": state["current_artifact_file_sha256"],
        "current_artifact_content_sha256": state["current_artifact_content_sha256"],
        "permanently_stopped": state["permanently_stopped"],
        "claims": state["claims"],
    }
    return _with_hash(body)


def run_mixed_third_jet_continuation_service(
    initial_prior_path: Path,
    initial_chunk_config_path: Path,
    diagonal: dict[str, Any],
    quadratic: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
    config: dict[str, Any],
    output: Path,
    *,
    executor: Executor = exact_mixed_third_jet_executor,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    _validate_service_config(config, diagonal, quadratic)
    initial_prior, initial_prior_data = _load_file(initial_prior_path)
    initial_chunk_config, initial_config_data = _load_file(initial_chunk_config_path)
    _validate_initial_prior(
        initial_prior,
        initial_prior_data,
        initial_chunk_config,
        initial_config_data,
        diagonal,
        quadratic,
        config,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    status_path = output / "service-status.json"
    state = _load_state(checkpoint_path, config)
    prior = _artifact_from_state(state, output, initial_prior)
    if state["permanently_stopped"]:
        return {
            "status": "already_stopped",
            "reason": state["stop_reason"],
            "chunks_advanced": 0,
            "checkpoint": state,
        }

    start = monotonic()
    advanced = 0
    reason = "chunk_limit"
    while advanced < int(config["max_chunks_per_invocation"]):
        if monotonic() - start >= float(config["max_wall_seconds"]):
            reason = "wall_time_limit"
            break
        offset = int(state["next_offset"])
        remaining = TOTAL_MIXED_TRIPLES - offset
        if remaining <= 0:
            reason = "mixed_selector_complete_full_tube_still_open"
            break
        if len(state["history"]) >= int(config["max_history_records"]):
            reason = "history_limit"
            break
        size = min(DEFAULT_CHUNK_SIZE, remaining)
        chunk_config = _chunk_config(config, state["prior_resume_sha256"], offset, size)
        relative = Path("chunks") / f"offset-{offset:06d}.json"
        artifact_path = output / relative
        pending = _pending_artifact(
            artifact_path,
            chunk_config,
            offset,
            size,
            config["canonical_D2_artifact_sequence_sha256"],
        )
        if pending is None:
            result = executor(diagonal, quadratic, canonical_artifacts, chunk_config)
            _validate_result(
                result,
                chunk_config,
                offset,
                size,
                config["canonical_D2_artifact_sequence_sha256"],
            )
            artifact_data = _json_bytes(result)
        else:
            result, artifact_data = pending
        if monotonic() - start > float(config["max_wall_seconds"]):
            raise QuarticTC2MixedThirdJetContinuationServiceError(
                "executor exceeded wall-time budget"
            )
        if len(artifact_data) > int(config["max_artifact_bytes"]):
            raise QuarticTC2MixedThirdJetContinuationServiceError("artifact byte budget exceeded")
        contract = result["chunk_contract"]
        obstruction = result["first_exact_obstruction"] is not None
        history_record = {
            "offset": offset,
            "next_offset": contract["next_offset"],
            "requested_chunk_size": size,
            "processed_count": contract["processed_count"],
            "closed_count": result["closure_ledger"]["processed_mixed_third_jets_closed"],
            "remaining_mixed_triples": result["counts"]["mixed_triples_remaining"],
            "artifact_path": relative.as_posix(),
            "artifact_file_sha256": _file_sha256(artifact_data),
            "artifact_content_sha256": result["content_sha256"],
            "chunk_config_content_sha256": _content_hash(chunk_config),
            "prior_content_sha256": prior["content_sha256"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "resume_tip_sha256": contract["resume_tip_sha256"],
            "status": result["status"],
            "obstruction": obstruction,
        }
        state = _with_hash(
            {
                **_body(state),
                "next_offset": contract["next_offset"],
                "remaining_mixed_triples": result["counts"]["mixed_triples_remaining"],
                "prior_resume_sha256": contract["resume_tip_sha256"],
                "current_artifact_path": relative.as_posix(),
                "current_artifact_file_sha256": _file_sha256(artifact_data),
                "current_artifact_content_sha256": result["content_sha256"],
                "completed_chunks": int(state["completed_chunks"]) + 1,
                "permanently_stopped": obstruction,
                "stop_reason": "exact_obstruction" if obstruction else None,
                "history": [*state["history"], history_record],
            }
        )
        checkpoint_data = _json_bytes(state)
        status = _status_artifact(
            state,
            checkpoint_data,
            "stopped" if obstruction else "checkpointed",
            "exact_obstruction" if obstruction else "chunk_limit",
        )
        status_data = _json_bytes(status)
        current_bytes = _service_disk_bytes(output)
        replacements = sum(
            path.stat().st_size
            for path in (artifact_path, checkpoint_path, status_path)
            if path.exists()
        )
        projected = (
            current_bytes
            - replacements
            + len(artifact_data)
            + len(checkpoint_data)
            + len(status_data)
        )
        if projected > int(config["max_total_service_bytes"]):
            raise QuarticTC2MixedThirdJetContinuationServiceError("service disk budget exceeded")
        _atomic_write(artifact_path, artifact_data)
        _atomic_write(checkpoint_path, checkpoint_data)
        _atomic_write(status_path, status_data)
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
