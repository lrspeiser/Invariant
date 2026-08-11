from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any

from .quartic_tc2_diagonal_third_jet_campaign import _content_hash
from .quartic_tc2_fourth_jet_parallel_kernel import (
    evaluate_fourth_obligations_process_pool,
)
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _body,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _service_disk_bytes,
    _with_hash,
)

SERVICE_SCHEMA = "sigma-quartic-tc2-fourth-jet-obligation-service-1.0"
CHUNK_SCHEMA = "sigma-quartic-tc2-fourth-jet-obligation-chunk-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-fourth-jet-obligation-checkpoint-1.0"
STATUS_SCHEMA = "sigma-quartic-tc2-fourth-jet-obligation-status-1.0"
TOTAL_OBLIGATIONS = 3060
DEFAULT_CHUNK_SIZE = 32
FINAL_TAIL_SIZE = 20
EXPECTED_CANDIDATES = 12


class QuarticTC2FourthJetObligationServiceError(ValueError):
    """Raised when the restart-safe fourth-jet service cannot advance exactly."""


Executor = Callable[
    [dict[str, Any], dict[str, Any], dict[str, Any]], dict[str, Any]
]


def _global_claims() -> dict[str, bool]:
    return {
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }


def _load_bound(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2FourthJetObligationServiceError("bound input escaped root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2FourthJetObligationServiceError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _validate_selector(campaign: dict[str, Any]) -> list[dict[str, Any]]:
    selector = campaign.get("selector", {})
    records = selector.get("records", [])
    prior = selector.get("seed_sha256")
    if (
        campaign.get("counts", {}).get("fourth_selector_records")
        != TOTAL_OBLIGATIONS
        or campaign.get("counts", {}).get("candidates") != EXPECTED_CANDIDATES
        or len(records) != TOTAL_OBLIGATIONS
        or len(selector.get("active_positions", [])) != 15
    ):
        raise QuarticTC2FourthJetObligationServiceError("selector count mismatch")
    for offset, record in enumerate(records):
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        if (
            record.get("selector_offset") != offset
            or record.get("prior_record_sha256") != prior
            or record.get("record_sha256") != _content_hash(body)
            or len(record.get("active_indices", [])) != 4
        ):
            raise QuarticTC2FourthJetObligationServiceError("selector chain mismatch")
        prior = record["record_sha256"]
    if prior != selector.get("tip_sha256"):
        raise QuarticTC2FourthJetObligationServiceError("selector tip mismatch")
    return records


def _validate_config(
    config: dict[str, Any],
    campaign: dict[str, Any],
    predecessor: dict[str, Any],
    candidates: dict[str, Any],
) -> None:
    if (
        config.get("schema_version") != SERVICE_SCHEMA
        or not _hash_matches(config)
        or config.get("selector")
        != "normalized_sym4_active_basis_lexicographic_record_chain"
        or config.get("chunk_size") != DEFAULT_CHUNK_SIZE
        or config.get("final_tail_size") != FINAL_TAIL_SIZE
        or config.get("parallel_worker_count") != 8
        or config.get("max_chunks_per_invocation") != 1
        or config.get("parallel_execution_policy")
        != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
        or config.get("stop_on_first_obstruction") is not True
        or config.get("orphan_policy")
        != "validate_and_adopt_exact_artifact_without_executor_reentry"
        or config.get("partial_tail_policy")
        != "evaluate_exact_remaining_selector_entries_without_padding_or_inference"
        or any(
            not isinstance(config.get(key), int) or int(config[key]) <= 0
            for key in (
                "max_wall_seconds",
                "max_artifact_bytes",
                "max_total_service_bytes",
                "max_history_records",
            )
        )
        or predecessor.get("claims", {}).get("full_mixed_sector_closed") is not True
        or predecessor.get("remaining_obligations") != 0
        or predecessor.get("next_obligation_offset") != 447
        or campaign.get("upstream_sha256", {}).get("completed_third_jet_checkpoint")
        != predecessor.get("content_sha256")
        or candidates.get("counts", {}).get("selected") != EXPECTED_CANDIDATES
        or len(candidates.get("certificates", [])) != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2FourthJetObligationServiceError("unsupported service contract")


def _initial_resume_sha256(
    campaign: dict[str, Any], predecessor: dict[str, Any]
) -> str:
    return _content_hash(
        {
            "fourth_campaign_content_sha256": campaign["content_sha256"],
            "selector_seed_sha256": campaign["selector"]["seed_sha256"],
            "selector_tip_sha256": campaign["selector"]["tip_sha256"],
            "third_jet_predecessor_content_sha256": predecessor["content_sha256"],
        }
    )


def _chunk_seed(
    campaign: dict[str, Any], offset: int, prior_resume_sha256: str
) -> str:
    return _content_hash(
        {
            "fourth_campaign_content_sha256": campaign["content_sha256"],
            "selector_tip_sha256": campaign["selector"]["tip_sha256"],
            "obligation_offset": offset,
            "prior_resume_sha256": prior_resume_sha256,
        }
    )


def _chunk_config(
    config: dict[str, Any],
    campaign: dict[str, Any],
    offset: int,
    size: int,
    prior_resume_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": CHUNK_SCHEMA,
        "selector": config["selector"],
        "fourth_campaign_content_sha256": campaign["content_sha256"],
        "selector_tip_sha256": campaign["selector"]["tip_sha256"],
        "obligation_offset": offset,
        "requested_size": size,
        "expected_prior_resume_sha256": prior_resume_sha256,
        "parallel_worker_count": config["parallel_worker_count"],
        "parallel_execution_policy": config["parallel_execution_policy"],
        "stop_on_first_obstruction": True,
        "resume_policy": "record_sha256_chain",
        "global_claim_policy": "fail_closed",
    }


def exact_fourth_jet_executor(
    campaign: dict[str, Any], candidates: dict[str, Any], chunk_config: dict[str, Any]
) -> dict[str, Any]:
    selector = campaign["selector"]["records"]
    offset = int(chunk_config["obligation_offset"])
    size = int(chunk_config["requested_size"])
    selected = selector[offset : offset + size]
    partial = size != DEFAULT_CHUNK_SIZE
    if (
        len(selected) != size
        or not 0 < size <= DEFAULT_CHUNK_SIZE
        or (partial and (size != FINAL_TAIL_SIZE or offset + size != TOTAL_OBLIGATIONS))
    ):
        raise QuarticTC2FourthJetObligationServiceError("chunk range mismatch")
    coefficient_map = {
        certificate["candidate_id"]: certificate["coefficients"]
        for certificate in candidates["certificates"]
    }
    indices = [tuple(record["active_indices"]) for record in selected]
    evaluated = evaluate_fourth_obligations_process_pool(
        indices,
        campaign["selector"]["active_positions"],
        coefficient_map,
        worker_count=int(chunk_config["parallel_worker_count"]),
    )
    if [tuple(result["active_indices"]) for result in evaluated] != indices:
        raise QuarticTC2FourthJetObligationServiceError("parallel result order mismatch")
    previous = _chunk_seed(
        campaign, offset, chunk_config["expected_prior_resume_sha256"]
    )
    manifest: list[dict[str, Any]] = []
    first_obstruction: dict[str, Any] | None = None
    for local_index, (selector_record, result) in enumerate(
        zip(selected, evaluated, strict=True)
    ):
        dynamic = dict(result)
        dynamic.pop("active_indices")
        obstructed = list(dynamic["obstructed_candidate_ids"])
        body = {
            "obligation_offset": offset + local_index,
            "chunk_index": local_index,
            "selector_record_sha256": selector_record["record_sha256"],
            "active_indices": selector_record["active_indices"],
            "active_positions": selector_record["active_positions"],
            "multiplicity_partition": selector_record["multiplicity_partition"],
            "fourth_campaign_content_sha256": campaign["content_sha256"],
            **dynamic,
            "previous_record_sha256": previous,
        }
        record = {**body, "record_sha256": _content_hash(body)}
        manifest.append(record)
        previous = record["record_sha256"]
        if obstructed:
            first_obstruction = {
                "obligation_offset": offset + local_index,
                "selector_record_sha256": selector_record["record_sha256"],
                "record_sha256": record["record_sha256"],
                "active_indices": selector_record["active_indices"],
                "active_positions": selector_record["active_positions"],
                "obstructed_candidate_ids": obstructed,
                "gate": "fourth-order equal-eigenspace Sylvester compatibility",
            }
            break
    processed = len(manifest)
    candidate_evaluations = processed * EXPECTED_CANDIDATES
    candidate_obstructions = sum(
        len(record["obstructed_candidate_ids"]) for record in manifest
    )
    passed = sum(not record["obstructed_candidate_ids"] for record in manifest)
    status = (
        "stop_first_exact_fourth_jet_obstruction"
        if first_obstruction
        else (
            f"pass_exact_fourth_jet_final_tail_{size}_tube_fail_closed"
            if partial
            else f"pass_exact_fourth_jet_chunk_{DEFAULT_CHUNK_SIZE}_tube_fail_closed"
        )
    )
    contract = {
        "selector": chunk_config["selector"],
        "global_obligation_count": TOTAL_OBLIGATIONS,
        "obligation_offset": offset,
        "requested_chunk_size": size,
        "processed_count": processed,
        "next_obligation_offset": offset + processed,
        "parallel_worker_count": 8,
        "parallel_execution_policy": chunk_config["parallel_execution_policy"],
        "bounded_speculative_evaluations_may_finish_after_first_obstruction": True,
        "records_after_first_obstruction_committed_or_inferred": 0,
        "stop_on_first_obstruction": True,
        "stopped_early": first_obstruction is not None,
        "resume_policy": "record_sha256_chain",
        "prior_resume_sha256": chunk_config["expected_prior_resume_sha256"],
        "resume_seed_sha256": _chunk_seed(
            campaign, offset, chunk_config["expected_prior_resume_sha256"]
        ),
        "resume_tip_sha256": previous,
        **({"exact_final_partial_tail": True} if partial else {}),
    }
    body = {
        "schema_version": CHUNK_SCHEMA,
        "status": status,
        "config_sha256": _content_hash(chunk_config),
        "upstream_sha256": {
            "fourth_campaign": campaign["content_sha256"],
            "candidate_source": candidates["content_sha256"],
        },
        "chunk_contract": contract,
        "counts": {
            "selected": processed,
            "partition_counts": dict(
                sorted(Counter(record["multiplicity_partition"] for record in manifest).items())
            ),
            "directional_evaluations": sum(
                record["directional_evaluations"] for record in manifest
            ),
            "symbolic_parameter_compatible": sum(
                bool(record["symbolic_parameter_compatible"]) for record in manifest
            ),
            "candidate_evaluations": candidate_evaluations,
            "candidate_solvable": candidate_evaluations - candidate_obstructions,
            "candidate_obstructed": candidate_obstructions,
            "fourth_obligations_remaining": TOTAL_OBLIGATIONS - offset - passed,
            "fourth_obligations_inferred_passed": 0,
        },
        "first_exact_obstruction": first_obstruction,
        "obligation_manifest": manifest,
        "closure_ledger": {
            "processed_fourth_obligations_closed": passed,
            "all_3060_fourth_obligations_closed": (
                first_obstruction is None and offset + passed == TOTAL_OBLIGATIONS
            ),
            **_global_claims(),
        },
        "scope": (
            "Only explicitly evaluated fourth-jet selector records are closed. Full tube, "
            "CK1, CK3, TC2, B7, global H7, and lifespan remain fail-closed."
        ),
        "errors": [],
    }
    return _with_hash(body)


def _validate_chunk_result(
    result: dict[str, Any],
    chunk_config: dict[str, Any],
    campaign: dict[str, Any],
    candidates: dict[str, Any],
) -> None:
    offset = int(chunk_config["obligation_offset"])
    size = int(chunk_config["requested_size"])
    selector = campaign["selector"]["records"]
    manifest = result.get("obligation_manifest", [])
    contract = result.get("chunk_contract", {})
    counts = result.get("counts", {})
    obstruction = result.get("first_exact_obstruction")
    processed = len(manifest)
    partial = size != DEFAULT_CHUNK_SIZE
    expected_status = (
        "stop_first_exact_fourth_jet_obstruction"
        if obstruction
        else (
            f"pass_exact_fourth_jet_final_tail_{size}_tube_fail_closed"
            if partial
            else f"pass_exact_fourth_jet_chunk_{DEFAULT_CHUNK_SIZE}_tube_fail_closed"
        )
    )
    if (
        not _hash_matches(result)
        or result.get("schema_version") != CHUNK_SCHEMA
        or result.get("status") != expected_status
        or result.get("config_sha256") != _content_hash(chunk_config)
        or result.get("upstream_sha256", {}).get("fourth_campaign")
        != campaign["content_sha256"]
        or result.get("upstream_sha256", {}).get("candidate_source")
        != candidates["content_sha256"]
        or not 0 < processed <= size
        or (not obstruction and processed != size)
        or contract.get("obligation_offset") != offset
        or contract.get("requested_chunk_size") != size
        or contract.get("processed_count") != processed
        or contract.get("next_obligation_offset") != offset + processed
        or contract.get("parallel_worker_count") != 8
        or contract.get("parallel_execution_policy")
        != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
        or contract.get("records_after_first_obstruction_committed_or_inferred") != 0
        or contract.get("stop_on_first_obstruction") is not True
        or contract.get("stopped_early") is not bool(obstruction)
        or counts.get("selected") != processed
        or counts.get("candidate_evaluations") != processed * EXPECTED_CANDIDATES
        or counts.get("candidate_solvable", 0) + counts.get("candidate_obstructed", 0)
        != processed * EXPECTED_CANDIDATES
        or counts.get("fourth_obligations_inferred_passed") != 0
        or result.get("closure_ledger", {}).get("processed_fourth_obligations_closed")
        != sum(not record.get("obstructed_candidate_ids") for record in manifest)
        or any(result.get("closure_ledger", {}).get(key) for key in _global_claims())
    ):
        raise QuarticTC2FourthJetObligationServiceError("chunk contract mismatch")
    previous = contract.get("resume_seed_sha256")
    first_obstructed_index: int | None = None
    for local_index, record in enumerate(manifest):
        selector_record = selector[offset + local_index]
        body = {key: value for key, value in record.items() if key != "record_sha256"}
        candidate_results = record.get("candidate_results", [])
        obstructed = sorted(
            item["candidate_id"] for item in candidate_results if not item.get("solvable")
        )
        if (
            record.get("obligation_offset") != offset + local_index
            or record.get("chunk_index") != local_index
            or record.get("selector_record_sha256")
            != selector_record["record_sha256"]
            or record.get("active_indices") != selector_record["active_indices"]
            or record.get("previous_record_sha256") != previous
            or record.get("record_sha256") != _content_hash(body)
            or len(candidate_results) != EXPECTED_CANDIDATES
            or sorted(record.get("obstructed_candidate_ids", [])) != obstructed
        ):
            raise QuarticTC2FourthJetObligationServiceError("result record-chain mismatch")
        if obstructed and first_obstructed_index is None:
            first_obstructed_index = local_index
        previous = record["record_sha256"]
    if previous != contract.get("resume_tip_sha256"):
        raise QuarticTC2FourthJetObligationServiceError("result resume tip mismatch")
    if (first_obstructed_index is None) is not (obstruction is None):
        raise QuarticTC2FourthJetObligationServiceError("obstruction ledger mismatch")
    if first_obstructed_index is not None and first_obstructed_index != processed - 1:
        raise QuarticTC2FourthJetObligationServiceError("post-obstruction record committed")
    if partial and (
        size != FINAL_TAIL_SIZE or offset + size != TOTAL_OBLIGATIONS
    ):
        raise QuarticTC2FourthJetObligationServiceError("partial tail mismatch")


def _initial_state(
    config: dict[str, Any], campaign: dict[str, Any], predecessor: dict[str, Any]
) -> dict[str, Any]:
    initial_resume = _initial_resume_sha256(campaign, predecessor)
    return _with_hash(
        {
            "schema_version": CHECKPOINT_SCHEMA,
            "config_content_sha256": config["content_sha256"],
            "fourth_campaign_content_sha256": campaign["content_sha256"],
            "fourth_campaign_file_sha256": config["fourth_campaign"]["file_sha256"],
            "third_jet_predecessor_content_sha256": predecessor["content_sha256"],
            "selector_seed_sha256": campaign["selector"]["seed_sha256"],
            "selector_tip_sha256": campaign["selector"]["tip_sha256"],
            "next_obligation_offset": 0,
            "remaining_obligations": TOTAL_OBLIGATIONS,
            "prior_resume_sha256": initial_resume,
            "current_artifact_path": config["fourth_campaign"]["path"],
            "current_artifact_file_sha256": config["fourth_campaign"]["file_sha256"],
            "current_artifact_content_sha256": campaign["content_sha256"],
            "completed_chunks": 0,
            "permanently_stopped": False,
            "stop_reason": None,
            "history": [],
            "claims": _global_claims(),
        }
    )


def _load_state(
    path: Path,
    output: Path,
    config: dict[str, Any],
    campaign: dict[str, Any],
    predecessor: dict[str, Any],
    candidates: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return _initial_state(config, campaign, predecessor)
    state, _ = _load_file(path)
    if (
        not _hash_matches(state)
        or state.get("schema_version") != CHECKPOINT_SCHEMA
        or state.get("config_content_sha256") != config["content_sha256"]
        or state.get("fourth_campaign_content_sha256") != campaign["content_sha256"]
        or state.get("selector_tip_sha256") != campaign["selector"]["tip_sha256"]
        or state.get("claims") != _global_claims()
        or state.get("completed_chunks") != len(state.get("history", []))
    ):
        raise QuarticTC2FourthJetObligationServiceError("checkpoint contract mismatch")
    expected_offset = 0
    expected_remaining = TOTAL_OBLIGATIONS
    expected_prior = _initial_resume_sha256(campaign, predecessor)
    for history in state["history"]:
        if (
            history.get("obligation_offset") != expected_offset
            or history.get("prior_resume_sha256") != expected_prior
        ):
            raise QuarticTC2FourthJetObligationServiceError("checkpoint history mismatch")
        artifact_path = (output / history["artifact_path"]).resolve()
        if output.resolve() not in artifact_path.parents:
            raise QuarticTC2FourthJetObligationServiceError("history artifact escaped root")
        artifact, data = _load_file(artifact_path)
        if (
            _file_sha256(data) != history["artifact_file_sha256"]
            or artifact.get("content_sha256") != history["artifact_content_sha256"]
        ):
            raise QuarticTC2FourthJetObligationServiceError("history artifact hash mismatch")
        dynamic = _chunk_config(
            config,
            campaign,
            int(history["obligation_offset"]),
            int(history["requested_chunk_size"]),
            str(history["prior_resume_sha256"]),
        )
        _validate_chunk_result(artifact, dynamic, campaign, candidates)
        expected_offset = int(history["next_obligation_offset"])
        expected_remaining = int(history["remaining_obligations"])
        expected_prior = str(history["resume_tip_sha256"])
    if (
        state.get("next_obligation_offset") != expected_offset
        or state.get("remaining_obligations") != expected_remaining
        or state.get("prior_resume_sha256") != expected_prior
        or bool(state["history"] and state["history"][-1]["obstruction"])
        != bool(state.get("permanently_stopped"))
    ):
        raise QuarticTC2FourthJetObligationServiceError("checkpoint tip mismatch")
    return state


def _status(
    state: dict[str, Any], checkpoint_data: bytes, decision: str, reason: str
) -> dict[str, Any]:
    return _with_hash(
        {
            "schema_version": STATUS_SCHEMA,
            "decision": decision,
            "reason": reason,
            "checkpoint_content_sha256": state["content_sha256"],
            "checkpoint_file_sha256": _file_sha256(checkpoint_data),
            "next_obligation_offset": state["next_obligation_offset"],
            "remaining_obligations": state["remaining_obligations"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "current_artifact_path": state["current_artifact_path"],
            "current_artifact_file_sha256": state["current_artifact_file_sha256"],
            "current_artifact_content_sha256": state[
                "current_artifact_content_sha256"
            ],
            "permanently_stopped": state["permanently_stopped"],
            "claims": _global_claims(),
        }
    )


def run_fourth_jet_obligation_service(
    project_root: Path,
    config_path: Path,
    output: Path,
    *,
    executor: Executor = exact_fourth_jet_executor,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    output = output.resolve()
    config, _ = _load_file(config_path.resolve())
    campaign = _load_bound(project_root, config["fourth_campaign"])
    predecessor = _load_bound(project_root, config["third_jet_predecessor"])
    candidates = _load_bound(project_root, config["candidate_source"])
    _validate_config(config, campaign, predecessor, candidates)
    _validate_selector(campaign)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    status_path = output / "service-status.json"
    state = _load_state(
        checkpoint_path,
        output,
        config,
        campaign,
        predecessor,
        candidates,
    )
    if state["permanently_stopped"]:
        return {"status": "already_stopped", "checkpoint": state, "chunks_advanced": 0}
    if state["remaining_obligations"] == 0:
        return {"status": "selector_complete", "checkpoint": state, "chunks_advanced": 0}
    start = monotonic()
    offset = int(state["next_obligation_offset"])
    remaining = TOTAL_OBLIGATIONS - offset
    size = min(DEFAULT_CHUNK_SIZE, remaining)
    if size != DEFAULT_CHUNK_SIZE and size != FINAL_TAIL_SIZE:
        raise QuarticTC2FourthJetObligationServiceError("unexpected final tail")
    chunk_config = _chunk_config(
        config, campaign, offset, size, state["prior_resume_sha256"]
    )
    relative = Path("chunks") / f"obligation-offset-{offset:06d}.json"
    artifact_path = output / relative
    orphan_adopted = False
    if artifact_path.exists():
        result, artifact_data = _load_file(artifact_path)
        _validate_chunk_result(result, chunk_config, campaign, candidates)
        orphan_adopted = True
    else:
        result = executor(campaign, candidates, chunk_config)
        _validate_chunk_result(result, chunk_config, campaign, candidates)
        artifact_data = _json_bytes(result)
    if monotonic() - start > int(config["max_wall_seconds"]):
        raise QuarticTC2FourthJetObligationServiceError("wall budget exceeded")
    if len(artifact_data) > int(config["max_artifact_bytes"]):
        raise QuarticTC2FourthJetObligationServiceError("artifact byte budget exceeded")
    contract = result["chunk_contract"]
    obstruction = result["first_exact_obstruction"] is not None
    history = {
        "obligation_offset": offset,
        "next_obligation_offset": contract["next_obligation_offset"],
        "requested_chunk_size": size,
        "processed_count": contract["processed_count"],
        "closed_count": result["closure_ledger"]["processed_fourth_obligations_closed"],
        "remaining_obligations": result["counts"]["fourth_obligations_remaining"],
        "artifact_path": relative.as_posix(),
        "artifact_file_sha256": _file_sha256(artifact_data),
        "artifact_content_sha256": result["content_sha256"],
        "chunk_config_content_sha256": _content_hash(chunk_config),
        "prior_resume_sha256": state["prior_resume_sha256"],
        "resume_tip_sha256": contract["resume_tip_sha256"],
        "status": result["status"],
        "obstruction": obstruction,
    }
    state = _with_hash(
        {
            **_body(state),
            "next_obligation_offset": contract["next_obligation_offset"],
            "remaining_obligations": result["counts"]["fourth_obligations_remaining"],
            "prior_resume_sha256": contract["resume_tip_sha256"],
            "current_artifact_path": relative.as_posix(),
            "current_artifact_file_sha256": _file_sha256(artifact_data),
            "current_artifact_content_sha256": result["content_sha256"],
            "completed_chunks": int(state["completed_chunks"]) + 1,
            "permanently_stopped": obstruction,
            "stop_reason": "exact_obstruction" if obstruction else None,
            "history": [*state["history"], history],
            "claims": _global_claims(),
        }
    )
    checkpoint_data = _json_bytes(state)
    status = _status(
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
        raise QuarticTC2FourthJetObligationServiceError("service disk budget exceeded")
    _atomic_write(artifact_path, artifact_data)
    _atomic_write(checkpoint_path, checkpoint_data)
    _atomic_write(status_path, status_data)
    return {
        "status": "stopped" if obstruction else "checkpointed",
        "reason": "exact_obstruction" if obstruction else "chunk_limit",
        "chunks_advanced": 1,
        "orphan_adopted": orphan_adopted,
        "next_obligation_offset": state["next_obligation_offset"],
        "remaining_obligations": state["remaining_obligations"],
        "prior_resume_sha256": state["prior_resume_sha256"],
        "checkpoint": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one restart-safe exact fourth-jet obligation chunk."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_fourth_jet_obligation_service(
        args.project_root, args.config, args.output
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
