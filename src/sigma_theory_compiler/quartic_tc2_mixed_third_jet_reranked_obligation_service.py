from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _content_hash_matches,
)
from .quartic_tc2_mixed_third_jet_basis_reduction_campaign import _load_bound_json
from .quartic_tc2_mixed_third_jet_chunk_campaign import (
    DEFAULT_CHUNK_SIZE,
    _pair_bindings,
    _record_hash_matches,
    _triple_kind,
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
from .quartic_tc2_mixed_third_jet_parallel_epoch import _canonical_second_atom_artifacts
from .quartic_tc2_mixed_third_jet_parallel_kernel import (
    evaluate_mixed_triples_process_pool,
)
from .quartic_tc2_quadratic_deltak_extension_campaign import _collect_records

SERVICE_SCHEMA = "sigma-quartic-tc2-reranked-obligation-service-1.0"
CHUNK_SCHEMA = "sigma-quartic-tc2-reranked-obligation-chunk-1.0"
CHECKPOINT_SCHEMA = "sigma-quartic-tc2-reranked-obligation-checkpoint-1.0"
STATUS_SCHEMA = "sigma-quartic-tc2-reranked-obligation-status-1.0"
TOTAL_OBLIGATIONS = 447
STABLE_PREFIX_COUNT = 1600

Executor = Callable[
    [
        dict[str, Any],
        dict[str, Any],
        list[dict[str, Any]],
        dict[str, Any],
        dict[str, Any],
    ],
    dict[str, Any],
]


class QuarticTC2RerankedObligationServiceError(ValueError):
    """Raised when the selective exact service cannot advance safely."""


def _claims(*, full_mixed_sector_closed: bool = False) -> dict[str, bool]:
    return {
        "full_mixed_sector_closed": full_mixed_sector_closed,
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }


def _initial_resume_sha256(reduction: dict[str, Any]) -> str:
    return _content_hash(
        {
            "stable_predecessor_resume_tip_sha256": reduction["stable_evidence"][
                "stable_resume_tip_sha256"
            ],
            "reranked_reduction_content_sha256": reduction["content_sha256"],
            "obligation_selector_seed_sha256": reduction["exact_reranking"][
                "obligation_seed_sha256"
            ],
            "obligation_selector_tip_sha256": reduction["exact_reranking"][
                "obligation_tip_sha256"
            ],
        }
    )


def _selector_records(reduction: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        reduction.get("status")
        != (
            "pass_exact_stopped_chain_1600_rerank_447_obligations_"
            "no_inferred_passes_global_closure_fail_closed"
        )
        or reduction.get("counts", {}).get("stable_mixed_triples_evaluated")
        != STABLE_PREFIX_COUNT
        or reduction.get("counts", {}).get("reranked_exact_obligations")
        != TOTAL_OBLIGATIONS
        or reduction.get("counts", {}).get("reranked_obligations_evaluated") != 0
        or reduction.get("counts", {}).get("remaining_active_triples_inferred_passed") != 0
        or reduction.get("exact_reranking", {}).get("completion_rank") != 680
        or reduction.get("exact_reranking", {}).get("drop_final_obligation_rank") != 679
        or reduction.get("exact_reranking", {}).get("reranked_obligation_count")
        != TOTAL_OBLIGATIONS
        or any(
            value
            for key, value in reduction.get("closure_ledger", {}).items()
            if key != "reranked_reduction_theorem_proved"
        )
    ):
        raise QuarticTC2RerankedObligationServiceError("reduction artifact contract mismatch")
    selector = reduction.get("reranked_obligation_selector", {}).get("obligations", [])
    if len(selector) != TOTAL_OBLIGATIONS:
        raise QuarticTC2RerankedObligationServiceError("reduction selector count mismatch")
    previous = reduction["exact_reranking"]["obligation_seed_sha256"]
    seen_global_indices: set[int] = set()
    for index, record in enumerate(selector):
        body = {key: value for key, value in record.items() if key != "obligation_sha256"}
        triple = tuple(int(value) for value in record.get("active_position_triple", []))
        global_index = record.get("global_selector_index")
        if (
            record.get("obligation_index") != index
            or len(triple) != 3
            or tuple(sorted(triple)) != triple
            or len(set(triple)) < 2
            or not isinstance(global_index, int)
            or global_index < STABLE_PREFIX_COUNT
            or global_index in seen_global_indices
            or record.get("triple_kind") != _triple_kind(triple)
            or record.get("previous_obligation_sha256") != previous
            or record.get("obligation_sha256") != _content_hash(body)
        ):
            raise QuarticTC2RerankedObligationServiceError("reduction selector chain mismatch")
        seen_global_indices.add(global_index)
        previous = record["obligation_sha256"]
    if previous != reduction["exact_reranking"]["obligation_tip_sha256"]:
        raise QuarticTC2RerankedObligationServiceError("reduction selector tip mismatch")
    return selector


def _validate_stopped_predecessor(
    checkpoint: dict[str, Any], supervisor: dict[str, Any], reduction: dict[str, Any]
) -> None:
    stable_tip = reduction["stable_evidence"]["stable_resume_tip_sha256"]
    if (
        checkpoint.get("completed_chunks") != 22
        or checkpoint.get("next_offset") != STABLE_PREFIX_COUNT
        or checkpoint.get("remaining_mixed_triples") != 10_700
        or checkpoint.get("prior_resume_sha256") != stable_tip
        or checkpoint.get("permanently_stopped") is not False
        or checkpoint.get("stop_reason") is not None
        or any(checkpoint.get("claims", {}).values())
        or supervisor.get("state") != "stopped"
        or supervisor.get("stop_reason") != "epoch_limit"
        or supervisor.get("pid") is not None
        or supervisor.get("next_offset") != STABLE_PREFIX_COUNT
        or supervisor.get("remaining_mixed_triples") != 10_700
        or supervisor.get("prior_resume_sha256") != stable_tip
        or supervisor.get("epoch_checkpoint_content_sha256") != checkpoint["content_sha256"]
        or any(supervisor.get("claims", {}).values())
    ):
        raise QuarticTC2RerankedObligationServiceError("stopped predecessor mismatch")


def _validate_config(
    config: dict[str, Any],
    diagonal: dict[str, Any],
    quadratic: dict[str, Any],
    reduction: dict[str, Any],
    checkpoint: dict[str, Any],
    supervisor: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if config.get("schema_version") != SERVICE_SCHEMA or not _hash_matches(config):
        raise QuarticTC2RerankedObligationServiceError("service config hash/schema mismatch")
    numeric = {
        "chunk_size": (1, DEFAULT_CHUNK_SIZE),
        "parallel_worker_count": (1, 16),
        "max_chunks_per_invocation": (1, 1),
        "max_wall_seconds": (1, 3600),
        "max_artifact_bytes": (1024, 64 * 1024 * 1024),
        "max_total_service_bytes": (4096, 1024 * 1024 * 1024),
        "max_history_records": (1, 32),
    }
    for key, (minimum, maximum) in numeric.items():
        if not minimum <= int(config.get(key, 0)) <= maximum:
            raise QuarticTC2RerankedObligationServiceError(f"invalid service budget: {key}")
    if (
        int(config["chunk_size"]) != DEFAULT_CHUNK_SIZE
        or int(config["parallel_worker_count"]) != 8
        or config.get("parallel_execution_policy")
        != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
        or config.get("selector")
        != "reranked_basis_coordinate_complement_of_stopped_1600_evidence_row_space"
        or config.get("resume_policy") != "record_sha256_chain"
        or config.get("orphan_recovery_policy") != "validate_and_adopt"
        or config.get("obstruction_policy") != "permanent_stop"
        or config.get("partial_tail_policy")
        != "evaluate_exact_remaining_obligation_entries_without_padding_or_inference"
        or config.get("stable_predecessor_prefix_count") != STABLE_PREFIX_COUNT
        or config.get("stable_predecessor_resume_tip_sha256")
        != reduction["stable_evidence"]["stable_resume_tip_sha256"]
        or config.get("obligation_selector_seed_sha256")
        != reduction["exact_reranking"]["obligation_seed_sha256"]
        or config.get("obligation_selector_tip_sha256")
        != reduction["exact_reranking"]["obligation_tip_sha256"]
        or config.get("total_obligations") != TOTAL_OBLIGATIONS
        or config.get("diagonal_third_jet_content_sha256") != diagonal.get("content_sha256")
        or config.get("quadratic_deltaK_content_sha256") != quadratic.get("content_sha256")
        or not _content_hash_matches(diagonal)
        or not _content_hash_matches(quadratic)
        or any(not _content_hash_matches(item) for item in canonical_artifacts)
        or _content_hash([item["content_sha256"] for item in canonical_artifacts])
        != config.get("canonical_D2_artifact_sequence_sha256")
        or any(
            config.get(key) != "fail_closed"
            for key in (
                "unprocessed_obligation_policy",
                "full_mixed_sector_policy",
                "full_tube_policy",
                "CK1_policy",
                "CK3_policy",
                "TC2_policy",
                "B7_policy",
                "global_H7_policy",
                "lifespan_policy",
            )
        )
    ):
        raise QuarticTC2RerankedObligationServiceError("unsupported service contract")
    _validate_stopped_predecessor(checkpoint, supervisor, reduction)
    return _selector_records(reduction)


def _provenance_packet(
    diagonal: dict[str, Any], canonical_artifacts: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    records, packets, artifact_hashes = _collect_records(
        canonical_artifacts, selector_key="selector_pair_index", expected_count=861
    )
    if _content_hash(artifact_hashes) != config["canonical_D2_artifact_sequence_sha256"]:
        raise QuarticTC2RerankedObligationServiceError("canonical D2 sequence mismatch")
    pair_packets: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        key = (int(record["left_atom_index"]), int(record["right_atom_index"]))
        packet = packets.get(str(record["symbolic_pair_packet_sha256"]))
        if packet is None:
            raise QuarticTC2RerankedObligationServiceError("D2 packet lookup failed")
        pair_packets[key] = packet
    directions = _active_directions()
    diagonal_records = {
        int(record["atom_index"]): record for record in diagonal["direction_records"]
    }
    coefficients = {
        str(item["candidate_id"]): item["coefficients"] for item in diagonal["certificates"]
    }
    if (
        len(pair_packets) != 861
        or len(directions) != 41
        or len(diagonal_records) != 41
        or len(coefficients) != 12
    ):
        raise QuarticTC2RerankedObligationServiceError("provenance coverage mismatch")
    return {
        "artifact_hashes": artifact_hashes,
        "pair_packets": pair_packets,
        "directions": directions,
        "diagonal_records": diagonal_records,
        "coefficients": coefficients,
    }


def _chunk_seed(
    reduction: dict[str, Any], config: dict[str, Any], offset: int, prior_resume: str
) -> str:
    return _content_hash(
        {
            "reranked_reduction_content_sha256": reduction["content_sha256"],
            "obligation_selector_tip_sha256": config["obligation_selector_tip_sha256"],
            "canonical_D2_artifact_sequence_sha256": config[
                "canonical_D2_artifact_sequence_sha256"
            ],
            "obligation_offset": offset,
            "prior_resume_sha256": prior_resume,
        }
    )


def exact_reranked_obligation_executor(
    diagonal: dict[str, Any],
    quadratic: dict[str, Any],
    canonical_artifacts: list[dict[str, Any]],
    reduction: dict[str, Any],
    chunk_config: dict[str, Any],
) -> dict[str, Any]:
    selector = _selector_records(reduction)
    offset = int(chunk_config["obligation_offset"])
    requested_size = int(chunk_config["requested_size"])
    selected = selector[offset : offset + requested_size]
    if (
        len(selected) != requested_size
        or requested_size < 1
        or requested_size > DEFAULT_CHUNK_SIZE
        or (
            requested_size != DEFAULT_CHUNK_SIZE
            and offset + requested_size != TOTAL_OBLIGATIONS
        )
    ):
        raise QuarticTC2RerankedObligationServiceError("selective chunk range mismatch")
    provenance = _provenance_packet(diagonal, canonical_artifacts, chunk_config)
    triples = [tuple(record["active_position_triple"]) for record in selected]
    evaluated = evaluate_mixed_triples_process_pool(
        triples,
        provenance["coefficients"],
        worker_count=int(chunk_config["parallel_worker_count"]),
    )
    if [tuple(item["active_position_triple"]) for item in evaluated] != triples:
        raise QuarticTC2RerankedObligationServiceError("parallel result order mismatch")
    directions = provenance["directions"]
    diagonal_records = provenance["diagonal_records"]
    pair_packets = provenance["pair_packets"]
    previous = _chunk_seed(
        reduction, chunk_config, offset, chunk_config["expected_prior_resume_sha256"]
    )
    manifest: list[dict[str, Any]] = []
    first_obstruction: dict[str, Any] | None = None
    for local_index, (selector_record, dynamic_result) in enumerate(
        zip(selected, evaluated, strict=True)
    ):
        triple = tuple(int(value) for value in selector_record["active_position_triple"])
        dynamic = dict(dynamic_result)
        dynamic.pop("active_position_triple")
        obstructed = list(dynamic["obstructed_candidate_ids"])
        unique_positions = sorted(set(triple))
        body = {
            "obligation_index": offset + local_index,
            "chunk_index": local_index,
            "global_selector_index": selector_record["global_selector_index"],
            "selector_obligation_sha256": selector_record["obligation_sha256"],
            "triple_kind": selector_record["triple_kind"],
            "active_position_triple": list(triple),
            "atom_index_triple": [int(directions[index]["atom_index"]) for index in triple],
            "atom_triple": [str(directions[index]["atom"]) for index in triple],
            "prior_bindings": {
                "D1_variable_sylvester_content_sha256": diagonal["upstream_sha256"][
                    "variable_sylvester"
                ],
                "D2_pair_packets": _pair_bindings(triple, directions, pair_packets),
                "diagonal_D3_direction_record_sha256": [
                    _content_hash(diagonal_records[int(directions[index]["atom_index"])])
                    for index in unique_positions
                ],
                "diagonal_third_jet_artifact_content_sha256": diagonal["content_sha256"],
                "reranked_reduction_content_sha256": reduction["content_sha256"],
            },
            **dynamic,
            "previous_record_sha256": previous,
        }
        record = {**body, "record_sha256": _content_hash(body)}
        manifest.append(record)
        previous = record["record_sha256"]
        if obstructed:
            first_obstruction = {
                "obligation_index": offset + local_index,
                "global_selector_index": selector_record["global_selector_index"],
                "record_sha256": record["record_sha256"],
                "active_position_triple": list(triple),
                "obstructed_candidate_ids": obstructed,
                "gate": "equal-eigenspace compatibility of reranked mixed third Sylvester RHS",
            }
            break
    processed = len(manifest)
    candidate_evaluations = sum(len(record["candidate_results"]) for record in manifest)
    candidate_obstructions = sum(
        len(record["obstructed_candidate_ids"]) for record in manifest
    )
    passed = sum(not record["obstructed_candidate_ids"] for record in manifest)
    selector_complete = (
        first_obstruction is None and offset + passed == TOTAL_OBLIGATIONS
    )
    partial_tail = requested_size != DEFAULT_CHUNK_SIZE
    status = (
        "stop_first_exact_reranked_obligation"
        if first_obstruction
        else (
            f"pass_reranked_obligation_exact_final_tail_{requested_size}_fail_closed"
            if partial_tail
            else "pass_reranked_obligation_chunk_64_fail_closed"
        )
    )
    contract = {
        "selector": chunk_config["selector"],
        "global_obligation_count": TOTAL_OBLIGATIONS,
        "obligation_offset": offset,
        "requested_chunk_size": requested_size,
        "processed_count": processed,
        "next_obligation_offset": offset + processed,
        "parallel_worker_count": int(chunk_config["parallel_worker_count"]),
        "parallel_execution_policy": chunk_config["parallel_execution_policy"],
        "bounded_speculative_evaluations_may_finish_after_first_obstruction": True,
        "records_after_first_obstruction_committed_or_inferred": 0,
        "stop_on_first_obstruction": True,
        "stopped_early": first_obstruction is not None,
        "resume_policy": "record_sha256_chain",
        "prior_resume_sha256": chunk_config["expected_prior_resume_sha256"],
        "resume_seed_sha256": _chunk_seed(
            reduction, chunk_config, offset, chunk_config["expected_prior_resume_sha256"]
        ),
        "resume_tip_sha256": previous,
        **({"exact_final_partial_tail": True} if partial_tail else {}),
    }
    body = {
        "schema_version": CHUNK_SCHEMA,
        "status": status,
        "errors": [],
        "config_sha256": _content_hash(chunk_config),
        "upstream_sha256": {
            "diagonal_third_jet": diagonal["content_sha256"],
            "quadratic_deltaK": quadratic["content_sha256"],
            "reranked_reduction": reduction["content_sha256"],
        },
        "canonical_D2_artifact_content_sha256": provenance["artifact_hashes"],
        "canonical_D2_artifact_sequence_sha256": _content_hash(
            provenance["artifact_hashes"]
        ),
        "chunk_contract": contract,
        **(
            {
                "partial_tail_control": {
                    "selector_total": TOTAL_OBLIGATIONS,
                    "tail_offset": offset,
                    "tail_size": requested_size,
                    "tail_exhausts_selector_exactly": offset + requested_size
                    == TOTAL_OBLIGATIONS,
                    "padded_or_inferred_obligations": 0,
                    "passed": True,
                }
            }
            if partial_tail
            else {}
        ),
        "counts": {
            "selected": processed,
            "triple_kind_counts": dict(
                sorted(Counter(record["triple_kind"] for record in manifest).items())
            ),
            "symbolic_parameter_compatible": sum(
                record["symbolic_parameter_compatible"] for record in manifest
            ),
            "candidate_evaluations": candidate_evaluations,
            "candidate_solvable": candidate_evaluations - candidate_obstructions,
            "candidate_obstructed": candidate_obstructions,
            "reranked_obligations_remaining": TOTAL_OBLIGATIONS - offset - passed,
            "stable_mixed_prefix_records": STABLE_PREFIX_COUNT,
            "remaining_active_triples_inferred_passed": 0,
            "TC2_closures": 0,
            "B7_closures": 0,
            "global_H7_closures": 0,
            "lifespans_proved": 0,
        },
        "first_exact_obstruction": first_obstruction,
        "obligation_manifest": manifest,
        "closure_ledger": {
            "processed_reranked_obligations_closed": passed,
            "all_447_reranked_obligations_closed": selector_complete,
            "all_12_300_mixed_third_jets_closed": selector_complete,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "scope": (
            "Only explicitly evaluated reranked obligations are closed. No remaining active "
            "triple is inferred passed; full mixed sector, full tube, CK1, CK3, TC2, B7, "
            "global H7, and lifespan remain fail-closed."
        ),
    }
    return {**body, "content_sha256": _content_hash(body)}


def _chunk_config(
    config: dict[str, Any], reduction: dict[str, Any], offset: int, size: int, prior: str
) -> dict[str, Any]:
    return {
        "schema_version": CHUNK_SCHEMA,
        "selector": config["selector"],
        "obligation_offset": offset,
        "requested_size": size,
        "expected_prior_resume_sha256": prior,
        "reranked_reduction_content_sha256": reduction["content_sha256"],
        "obligation_selector_tip_sha256": config["obligation_selector_tip_sha256"],
        "canonical_D2_artifact_sequence_sha256": config[
            "canonical_D2_artifact_sequence_sha256"
        ],
        "parallel_worker_count": config["parallel_worker_count"],
        "parallel_execution_policy": config["parallel_execution_policy"],
        "stop_on_first_obstruction": True,
        "resume_policy": "record_sha256_chain",
        "full_mixed_sector_policy": "fail_closed",
        "full_tube_policy": "fail_closed",
        "CK1_policy": "fail_closed",
        "CK3_policy": "fail_closed",
        "TC2_policy": "fail_closed",
        "B7_policy": "fail_closed",
        "global_H7_policy": "fail_closed",
        "lifespan_policy": "fail_closed",
    }


def _validate_chunk_result(
    result: dict[str, Any],
    chunk_config: dict[str, Any],
    reduction: dict[str, Any],
    service_config: dict[str, Any],
    offset: int,
    requested_size: int,
) -> None:
    selector = _selector_records(reduction)
    manifest = result.get("obligation_manifest", [])
    contract = result.get("chunk_contract", {})
    counts = result.get("counts", {})
    obstruction = result.get("first_exact_obstruction")
    processed = counts.get("selected")
    closed = result.get("closure_ledger", {}).get("processed_reranked_obligations_closed")
    partial = requested_size != DEFAULT_CHUNK_SIZE
    selector_complete = bool(
        obstruction is None
        and isinstance(closed, int)
        and offset + closed == TOTAL_OBLIGATIONS
    )
    expected_status = (
        "stop_first_exact_reranked_obligation"
        if obstruction
        else (
            f"pass_reranked_obligation_exact_final_tail_{requested_size}_fail_closed"
            if partial
            else "pass_reranked_obligation_chunk_64_fail_closed"
        )
    )
    if (
        not _content_hash_matches(result)
        or result.get("schema_version") != CHUNK_SCHEMA
        or result.get("status") != expected_status
        or result.get("config_sha256") != _content_hash(chunk_config)
        or result.get("upstream_sha256", {}).get("reranked_reduction")
        != reduction["content_sha256"]
        or result.get("upstream_sha256", {}).get("diagonal_third_jet")
        != service_config["diagonal_third_jet_content_sha256"]
        or result.get("upstream_sha256", {}).get("quadratic_deltaK")
        != service_config["quadratic_deltaK_content_sha256"]
        or result.get("canonical_D2_artifact_sequence_sha256")
        != chunk_config["canonical_D2_artifact_sequence_sha256"]
        or not isinstance(processed, int)
        or not isinstance(closed, int)
        or not 0 < processed <= requested_size
        or not 0 <= closed <= processed
        or len(manifest) != processed
        or contract.get("selector") != chunk_config["selector"]
        or contract.get("global_obligation_count") != TOTAL_OBLIGATIONS
        or contract.get("obligation_offset") != offset
        or contract.get("requested_chunk_size") != requested_size
        or contract.get("processed_count") != processed
        or contract.get("next_obligation_offset") != offset + processed
        or contract.get("prior_resume_sha256")
        != chunk_config["expected_prior_resume_sha256"]
        or contract.get("resume_seed_sha256")
        != _chunk_seed(
            reduction,
            chunk_config,
            offset,
            chunk_config["expected_prior_resume_sha256"],
        )
        or contract.get("parallel_worker_count") != 8
        or contract.get("parallel_execution_policy")
        != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
        or contract.get("bounded_speculative_evaluations_may_finish_after_first_obstruction")
        is not True
        or contract.get("records_after_first_obstruction_committed_or_inferred") != 0
        or contract.get("stop_on_first_obstruction") is not True
        or contract.get("stopped_early") is not bool(obstruction)
        or contract.get("resume_policy") != "record_sha256_chain"
        or counts.get("candidate_evaluations") != processed * 12
        or counts.get("candidate_solvable") + counts.get("candidate_obstructed")
        != processed * 12
        or counts.get("reranked_obligations_remaining") != TOTAL_OBLIGATIONS - offset - closed
        or counts.get("remaining_active_triples_inferred_passed") != 0
        or counts.get("symbolic_parameter_compatible")
        != sum(record.get("symbolic_parameter_compatible") is True for record in manifest)
        or (not obstruction and processed != requested_size)
        or (not obstruction and closed != processed)
        or (not obstruction and counts.get("candidate_obstructed") != 0)
        or (obstruction and counts.get("candidate_obstructed", 0) <= 0)
        or (obstruction and closed >= processed)
        or result.get("closure_ledger", {}).get("all_447_reranked_obligations_closed")
        is not selector_complete
        or result.get("closure_ledger", {}).get("all_12_300_mixed_third_jets_closed")
        is not selector_complete
        or any(
            result.get("closure_ledger", {}).get(key) is not False
            for key in (
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
    ):
        raise QuarticTC2RerankedObligationServiceError("executor result contract mismatch")
    previous = contract["resume_seed_sha256"]
    computed_candidate_evaluations = 0
    computed_candidate_obstructions = 0
    computed_closed = 0
    obstructed_record_indices: list[int] = []
    for local_index, record in enumerate(manifest):
        selector_record = selector[offset + local_index]
        triple = tuple(selector_record["active_position_triple"])
        candidates = record.get("candidate_results", [])
        computed_obstructed_ids = [
            candidate.get("candidate_id")
            for candidate in candidates
            if candidate.get("solvable") is not True
        ]
        if (
            not _record_hash_matches(record)
            or record.get("chunk_index") != local_index
            or record.get("obligation_index") != offset + local_index
            or record.get("global_selector_index") != selector_record["global_selector_index"]
            or record.get("selector_obligation_sha256")
            != selector_record["obligation_sha256"]
            or record.get("active_position_triple") != list(triple)
            or record.get("triple_kind") != _triple_kind(triple)
            or record.get("previous_record_sha256") != previous
            or len(candidates) != 12
            or len({candidate.get("candidate_id") for candidate in candidates}) != 12
            or record.get("obstructed_candidate_ids") != computed_obstructed_ids
            or any(
                candidate.get("third_Sylvester_residual_zero") is not True
                or (
                    candidate.get("solvable") is True
                    and (
                        candidate.get("equal_eigenspace_compressions_zero") is not True
                        or candidate.get("deltaK_ABC_Hermitian") is not True
                    )
                )
                for candidate in candidates
            )
        ):
            raise QuarticTC2RerankedObligationServiceError("result record-chain mismatch")
        previous = record["record_sha256"]
        computed_candidate_evaluations += len(candidates)
        computed_candidate_obstructions += len(computed_obstructed_ids)
        if computed_obstructed_ids:
            obstructed_record_indices.append(local_index)
        else:
            computed_closed += 1
    if previous != contract.get("resume_tip_sha256"):
        raise QuarticTC2RerankedObligationServiceError("result resume tip mismatch")
    if (
        computed_candidate_evaluations != counts.get("candidate_evaluations")
        or computed_candidate_obstructions != counts.get("candidate_obstructed")
        or computed_candidate_evaluations - computed_candidate_obstructions
        != counts.get("candidate_solvable")
        or computed_closed != closed
        or (
            obstruction is None
            and obstructed_record_indices
            or obstruction is not None
            and obstructed_record_indices != [processed - 1]
        )
    ):
        raise QuarticTC2RerankedObligationServiceError("result obstruction ledger mismatch")
    if obstruction is not None:
        last = manifest[-1]
        if obstruction != {
            "obligation_index": last["obligation_index"],
            "global_selector_index": last["global_selector_index"],
            "record_sha256": last["record_sha256"],
            "active_position_triple": last["active_position_triple"],
            "obstructed_candidate_ids": last["obstructed_candidate_ids"],
            "gate": "equal-eigenspace compatibility of reranked mixed third Sylvester RHS",
        }:
            raise QuarticTC2RerankedObligationServiceError(
                "result first obstruction mismatch"
            )
    expected_kinds = Counter(
        selector[offset + index]["triple_kind"] for index in range(processed)
    )
    if counts.get("triple_kind_counts") != dict(sorted(expected_kinds.items())):
        raise QuarticTC2RerankedObligationServiceError("result kind mismatch")
    tail = result.get("partial_tail_control")
    if partial:
        if (
            offset + requested_size != TOTAL_OBLIGATIONS
            or contract.get("exact_final_partial_tail") is not True
            or tail
            != {
                "selector_total": TOTAL_OBLIGATIONS,
                "tail_offset": offset,
                "tail_size": requested_size,
                "tail_exhausts_selector_exactly": True,
                "padded_or_inferred_obligations": 0,
                "passed": True,
            }
        ):
            raise QuarticTC2RerankedObligationServiceError("partial tail mismatch")
    elif tail is not None or contract.get("exact_final_partial_tail") is not None:
        raise QuarticTC2RerankedObligationServiceError("false partial tail marker")


def _initial_state(config: dict[str, Any], reduction: dict[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": CHECKPOINT_SCHEMA,
        "config_content_sha256": config["content_sha256"],
        "reranked_reduction_file_sha256": config["reranked_reduction"]["file_sha256"],
        "reranked_reduction_content_sha256": reduction["content_sha256"],
        "obligation_selector_seed_sha256": config["obligation_selector_seed_sha256"],
        "obligation_selector_tip_sha256": config["obligation_selector_tip_sha256"],
        "stable_predecessor_prefix_count": STABLE_PREFIX_COUNT,
        "stable_predecessor_resume_tip_sha256": config[
            "stable_predecessor_resume_tip_sha256"
        ],
        "next_obligation_offset": 0,
        "remaining_obligations": TOTAL_OBLIGATIONS,
        "prior_resume_sha256": _initial_resume_sha256(reduction),
        "current_artifact_path": None,
        "current_artifact_file_sha256": config["reranked_reduction"]["file_sha256"],
        "current_artifact_content_sha256": reduction["content_sha256"],
        "completed_chunks": 0,
        "permanently_stopped": False,
        "stop_reason": None,
        "history": [],
        "claims": _claims(),
    }
    return _with_hash(body)


def _load_state(path: Path, config: dict[str, Any], reduction: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return _initial_state(config, reduction)
    state = json.loads(path.read_text(encoding="utf-8"))
    history = state.get("history", [])
    if (
        state.get("schema_version") != CHECKPOINT_SCHEMA
        or not _hash_matches(state)
        or state.get("config_content_sha256") != config["content_sha256"]
        or state.get("reranked_reduction_content_sha256") != reduction["content_sha256"]
        or state.get("reranked_reduction_file_sha256")
        != config["reranked_reduction"]["file_sha256"]
        or state.get("obligation_selector_seed_sha256")
        != config["obligation_selector_seed_sha256"]
        or state.get("obligation_selector_tip_sha256")
        != config["obligation_selector_tip_sha256"]
        or state.get("stable_predecessor_prefix_count") != STABLE_PREFIX_COUNT
        or state.get("stable_predecessor_resume_tip_sha256")
        != config["stable_predecessor_resume_tip_sha256"]
        or len(history) != state.get("completed_chunks")
        or len(history) > int(config["max_history_records"])
        or state.get("claims")
        != _claims(full_mixed_sector_closed=state.get("remaining_obligations") == 0)
        or bool(state.get("permanently_stopped"))
        != (state.get("stop_reason") == "exact_obstruction")
    ):
        raise QuarticTC2RerankedObligationServiceError("checkpoint contract mismatch")
    expected_offset = 0
    expected_remaining = TOTAL_OBLIGATIONS
    expected_content = reduction["content_sha256"]
    expected_file = config["reranked_reduction"]["file_sha256"]
    expected_resume = _initial_resume_sha256(reduction)
    for record in history:
        processed = record.get("processed_count")
        closed = record.get("closed_count")
        if (
            not isinstance(processed, int)
            or not isinstance(closed, int)
            or not 0 < processed <= DEFAULT_CHUNK_SIZE
            or not 0 <= closed <= processed
            or record.get("obligation_offset") != expected_offset
            or record.get("next_obligation_offset") != expected_offset + processed
            or record.get("remaining_obligations") != expected_remaining - closed
            or record.get("prior_content_sha256") != expected_content
            or record.get("prior_resume_sha256") != expected_resume
            or not isinstance(record.get("artifact_content_sha256"), str)
            or not isinstance(record.get("artifact_file_sha256"), str)
            or not isinstance(record.get("resume_tip_sha256"), str)
            or record.get("requested_chunk_size")
            != min(DEFAULT_CHUNK_SIZE, TOTAL_OBLIGATIONS - expected_offset)
        ):
            raise QuarticTC2RerankedObligationServiceError("checkpoint history mismatch")
        expected_offset += processed
        expected_remaining -= closed
        expected_content = record["artifact_content_sha256"]
        expected_file = record["artifact_file_sha256"]
        expected_resume = record["resume_tip_sha256"]
    if (
        state.get("next_obligation_offset") != expected_offset
        or state.get("remaining_obligations") != expected_remaining
        or state.get("prior_resume_sha256") != expected_resume
        or state.get("current_artifact_content_sha256") != expected_content
        or state.get("current_artifact_file_sha256") != expected_file
        or bool(history and history[-1].get("obstruction"))
        != bool(state.get("permanently_stopped"))
    ):
        raise QuarticTC2RerankedObligationServiceError("checkpoint tip mismatch")
    return state


def _current_prior(
    state: dict[str, Any],
    output: Path,
    reduction: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if not state["history"]:
        return reduction
    path = (output / str(state["current_artifact_path"])).resolve()
    if output.resolve() not in path.parents:
        raise QuarticTC2RerankedObligationServiceError("checkpoint artifact escaped root")
    artifact, data = _load_file(path)
    if (
        _file_sha256(data) != state["current_artifact_file_sha256"]
        or artifact.get("content_sha256") != state["current_artifact_content_sha256"]
        or not _content_hash_matches(artifact)
    ):
        raise QuarticTC2RerankedObligationServiceError("checkpoint artifact hash mismatch")
    latest = state["history"][-1]
    offset = int(latest["obligation_offset"])
    size = int(latest["requested_chunk_size"])
    dynamic = _chunk_config(
        config,
        reduction,
        offset,
        size,
        str(latest["prior_resume_sha256"]),
    )
    _validate_chunk_result(artifact, dynamic, reduction, config, offset, size)
    return artifact


def _pending_artifact(
    path: Path,
    chunk_config: dict[str, Any],
    reduction: dict[str, Any],
    service_config: dict[str, Any],
    offset: int,
    size: int,
) -> tuple[dict[str, Any], bytes] | None:
    if not path.exists():
        return None
    artifact, data = _load_file(path)
    _validate_chunk_result(
        artifact, chunk_config, reduction, service_config, offset, size
    )
    return artifact, data


def _status(state: dict[str, Any], checkpoint_data: bytes, decision: str, reason: str) -> dict[str, Any]:
    return _with_hash(
        {
            "schema_version": STATUS_SCHEMA,
            "decision": decision,
            "reason": reason,
            "checkpoint_file_sha256": _file_sha256(checkpoint_data),
            "checkpoint_content_sha256": state["content_sha256"],
            "next_obligation_offset": state["next_obligation_offset"],
            "remaining_obligations": state["remaining_obligations"],
            "prior_resume_sha256": state["prior_resume_sha256"],
            "current_artifact_path": state["current_artifact_path"],
            "current_artifact_file_sha256": state["current_artifact_file_sha256"],
            "current_artifact_content_sha256": state["current_artifact_content_sha256"],
            "permanently_stopped": state["permanently_stopped"],
            "claims": state["claims"],
        }
    )


def run_reranked_obligation_service(
    project_root: Path,
    config_path: Path,
    output: Path,
    *,
    executor: Executor = exact_reranked_obligation_executor,
    monotonic: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    diagonal = _load_bound_json(project_root, config["diagonal_third_jet"])
    quadratic = _load_bound_json(project_root, config["quadratic_deltaK"])
    reduction = _load_bound_json(project_root, config["reranked_reduction"])
    predecessor = _load_bound_json(project_root, config["stopped_predecessor_checkpoint"])
    supervisor = _load_bound_json(project_root, config["stopped_supervisor_state"])
    canonical_artifacts = _canonical_second_atom_artifacts(project_root)
    _validate_config(
        config,
        diagonal,
        quadratic,
        reduction,
        predecessor,
        supervisor,
        canonical_artifacts,
    )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    status_path = output / "service-status.json"
    state = _load_state(checkpoint_path, config, reduction)
    prior = _current_prior(state, output, reduction, config)
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
        offset = int(state["next_obligation_offset"])
        remaining = TOTAL_OBLIGATIONS - offset
        if remaining <= 0:
            reason = "reranked_selector_complete_full_tube_still_open"
            break
        if len(state["history"]) >= int(config["max_history_records"]):
            reason = "history_limit"
            break
        size = min(DEFAULT_CHUNK_SIZE, remaining)
        chunk_config = _chunk_config(
            config, reduction, offset, size, state["prior_resume_sha256"]
        )
        relative = Path("chunks") / f"obligation-offset-{offset:06d}.json"
        artifact_path = output / relative
        pending = _pending_artifact(
            artifact_path, chunk_config, reduction, config, offset, size
        )
        if pending is None:
            result = executor(
                diagonal, quadratic, canonical_artifacts, reduction, chunk_config
            )
            _validate_chunk_result(result, chunk_config, reduction, config, offset, size)
            artifact_data = _json_bytes(result)
        else:
            result, artifact_data = pending
        if monotonic() - start > float(config["max_wall_seconds"]):
            raise QuarticTC2RerankedObligationServiceError("executor exceeded wall budget")
        if len(artifact_data) > int(config["max_artifact_bytes"]):
            raise QuarticTC2RerankedObligationServiceError("artifact byte budget exceeded")
        contract = result["chunk_contract"]
        obstruction = result["first_exact_obstruction"] is not None
        history_record = {
            "obligation_offset": offset,
            "next_obligation_offset": contract["next_obligation_offset"],
            "requested_chunk_size": size,
            "processed_count": contract["processed_count"],
            "closed_count": result["closure_ledger"][
                "processed_reranked_obligations_closed"
            ],
            "remaining_obligations": result["counts"]["reranked_obligations_remaining"],
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
                "next_obligation_offset": contract["next_obligation_offset"],
                "remaining_obligations": result["counts"]["reranked_obligations_remaining"],
                "prior_resume_sha256": contract["resume_tip_sha256"],
                "current_artifact_path": relative.as_posix(),
                "current_artifact_file_sha256": _file_sha256(artifact_data),
                "current_artifact_content_sha256": result["content_sha256"],
                "completed_chunks": int(state["completed_chunks"]) + 1,
                "permanently_stopped": obstruction,
                "stop_reason": "exact_obstruction" if obstruction else None,
                "history": [*state["history"], history_record],
                "claims": _claims(
                    full_mixed_sector_closed=(
                        not obstruction
                        and result["counts"]["reranked_obligations_remaining"] == 0
                    )
                ),
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
            raise QuarticTC2RerankedObligationServiceError("service disk budget exceeded")
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
        "next_obligation_offset": state["next_obligation_offset"],
        "remaining_obligations": state["remaining_obligations"],
        "prior_resume_sha256": state["prior_resume_sha256"],
        "checkpoint": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one exact reranked mixed-third-jet obligation chunk."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/backgrounds/quartic_tc2_mixed_third_jet_reranked_obligation_service.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/physics-language/quartic-tc2-mixed-third-jet-reranked-obligation-service"
        ),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output if args.output.is_absolute() else root / args.output
    print(
        json.dumps(
            run_reranked_obligation_service(root, config_path, output),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
