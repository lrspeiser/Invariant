from __future__ import annotations

import itertools
import json
from collections import Counter
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_diagonal_third_jet_campaign import _active_directions, _content_hash
from .quartic_tc2_mixed_third_jet_basis_reduction_campaign import (
    SYMMETRIC_CUBIC_DIMENSION,
    _all_global_claims_false,
    _load_bound_json,
)
from .quartic_tc2_mixed_third_jet_chunk_campaign import (
    ACTIVE_DIRECTION_COUNT,
    TOTAL_MIXED_TRIPLES,
    _mixed_selector,
    _record_hash_matches,
    _triple_kind,
)

SCHEMA_VERSION = "sigma-quartic-tc2-mixed-third-jet-reranked-reduction-campaign-1.0"
STABLE_PREFIX_COUNT = 1600
PRIOR_REDUCTION_PREFIX_COUNT = 576
PRIOR_REDUCTION_RANK = 120
PRIOR_REDUCTION_OBLIGATIONS = 560


class QuarticTC2MixedThirdJetRerankedReductionError(ValueError):
    """Raised when the stopped supervisor chain cannot support exact reranking."""


def _validate_diagonal(diagonal: dict[str, Any]) -> None:
    if (
        diagonal.get("status")
        != (
            "pass_bounded_all_41_diagonal_active_coordinate_third_jet_audit_"
            "mixed_triples_full_tube_global_H7_fail_closed"
        )
        or diagonal.get("counts", {}).get("diagonal_direction_packets") != 41
        or diagonal.get("counts", {}).get("symbolic_parameter_diagonal_third_jet_passes")
        != 41
        or diagonal.get("counts", {}).get("candidate_direction_evaluations") != 492
        or diagonal.get("counts", {}).get("candidate_direction_solvable") != 492
        or diagonal.get("counts", {}).get("candidate_direction_obstructed") != 0
        or len(diagonal.get("direction_records", [])) != ACTIVE_DIRECTION_COUNT
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError("diagonal evidence mismatch")
    if any(
        record.get("symbolic_equal_eigenspace_compressions_zero") is not True
        or len(record.get("candidate_results", [])) != 12
        or any(candidate.get("solvable") is not True for candidate in record["candidate_results"])
        for record in diagonal["direction_records"]
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError("diagonal record mismatch")


def _validate_chunk_chain(
    chunks: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[tuple[int, int, int]], str]:
    selector = _mixed_selector()
    expected_offset = 0
    prior_tip: str | None = None
    stable_triples: list[tuple[int, int, int]] = []
    for chunk in chunks:
        contract = chunk.get("chunk_contract", {})
        manifest = chunk.get("triple_manifest", [])
        counts = chunk.get("counts", {})
        parallel_expected = expected_offset >= 192
        if (
            int(contract.get("chunk_offset", -1)) != expected_offset
            or int(contract.get("requested_chunk_size", -1)) != 64
            or int(contract.get("processed_count", -1)) != 64
            or int(contract.get("next_offset", -1)) != expected_offset + 64
            or contract.get("prior_resume_sha256") != prior_tip
            or contract.get("stopped_early") is not False
            or len(manifest) != 64
            or chunk.get("first_exact_obstruction") is not None
            or counts.get("selected") != 64
            or counts.get("symbolic_parameter_compatible") != 64
            or counts.get("candidate_evaluations") != 768
            or counts.get("candidate_solvable") != 768
            or counts.get("candidate_obstructed") != 0
            or not _all_global_claims_false(chunk.get("closure_ledger", {}))
            or (
                parallel_expected
                and (
                    contract.get("parallel_worker_count") != 8
                    or contract.get("parallel_execution_policy")
                    != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
                    or contract.get("records_after_first_obstruction_committed_or_inferred") != 0
                )
            )
        ):
            raise QuarticTC2MixedThirdJetRerankedReductionError("stable chunk mismatch")
        previous = contract.get("resume_seed_sha256")
        for local_index, record in enumerate(manifest):
            selector_index = expected_offset + local_index
            triple = tuple(int(value) for value in record.get("active_position_triple", []))
            if (
                record.get("selector_index") != selector_index
                or triple != selector[selector_index]
                or record.get("previous_record_sha256") != previous
                or not _record_hash_matches(record)
                or record.get("symbolic_parameter_compatible") is not True
                or record.get("obstructed_candidate_ids") != []
                or len(record.get("candidate_results", [])) != 12
                or any(
                    candidate.get("solvable") is not True
                    for candidate in record["candidate_results"]
                )
            ):
                raise QuarticTC2MixedThirdJetRerankedReductionError(
                    "stable record-chain mismatch"
                )
            previous = record["record_sha256"]
            stable_triples.append(triple)
        if previous != contract.get("resume_tip_sha256"):
            raise QuarticTC2MixedThirdJetRerankedReductionError("stable resume tip mismatch")
        prior_tip = previous
        expected_offset += 64
    if (
        expected_offset != STABLE_PREFIX_COUNT
        or expected_offset != int(config["stable_predecessor_prefix_count"])
        or prior_tip != config["stable_predecessor_resume_tip_sha256"]
        or stable_triples != list(selector[:STABLE_PREFIX_COUNT])
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError("stable boundary mismatch")
    return stable_triples, str(prior_tip)


def _validate_stopped_boundary(
    boundary: dict[str, dict[str, Any]], chunks: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    checkpoint = boundary["checkpoint"]
    service_status = boundary["service_status"]
    supervisor_config = boundary["supervisor_config"]
    supervisor_state = boundary["supervisor_state"]
    supervisor_status = boundary["supervisor_status"]
    latest = chunks[-1]
    expected_offsets = list(range(192, STABLE_PREFIX_COUNT, 64))
    if (
        checkpoint.get("completed_chunks") != len(expected_offsets)
        or [record.get("offset") for record in checkpoint.get("history", [])]
        != expected_offsets
        or checkpoint.get("next_offset") != STABLE_PREFIX_COUNT
        or checkpoint.get("remaining_mixed_triples")
        != TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT
        or checkpoint.get("prior_resume_sha256")
        != config["stable_predecessor_resume_tip_sha256"]
        or checkpoint.get("current_artifact_content_sha256") != latest["content_sha256"]
        or checkpoint.get("permanently_stopped") is not False
        or checkpoint.get("stop_reason") is not None
        or any(checkpoint.get("claims", {}).values())
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError("checkpoint boundary mismatch")
    if (
        service_status.get("checkpoint_content_sha256") != checkpoint["content_sha256"]
        or service_status.get("next_offset") != STABLE_PREFIX_COUNT
        or service_status.get("remaining_mixed_triples")
        != TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT
        or service_status.get("prior_resume_sha256")
        != config["stable_predecessor_resume_tip_sha256"]
        or service_status.get("decision") != "checkpointed"
        or service_status.get("permanently_stopped") is not False
        or any(service_status.get("claims", {}).values())
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError("service status mismatch")
    if (
        supervisor_config.get("maximum_epochs_per_run") != 16
        or supervisor_config.get("parallel_worker_count") != 8
        or supervisor_config.get("parallel_execution_policy")
        != "ordered_spawn_pool_bounded_speculation_no_post_obstruction_commit"
        or supervisor_config.get("stop_on_first_obstruction") is not True
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError("supervisor config mismatch")
    for state, require_alive in ((supervisor_state, None), (supervisor_status, False)):
        if (
            state.get("state") != "stopped"
            or state.get("stop_reason") != "epoch_limit"
            or state.get("pid") is not None
            or state.get("epochs_completed") != 20
            or state.get("chunks_advanced") != 20
            or state.get("next_offset") != STABLE_PREFIX_COUNT
            or state.get("remaining_mixed_triples")
            != TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT
            or state.get("prior_resume_sha256")
            != config["stable_predecessor_resume_tip_sha256"]
            or any(state.get("claims", {}).values())
            or (require_alive is not None and state.get("alive") is not require_alive)
        ):
            raise QuarticTC2MixedThirdJetRerankedReductionError(
                "supervisor stopped-boundary mismatch"
            )
    if (
        supervisor_state.get("epoch_checkpoint_content_sha256")
        != checkpoint["content_sha256"]
    ):
        raise QuarticTC2MixedThirdJetRerankedReductionError(
            "supervisor/checkpoint binding mismatch"
        )


@cache
def _reranked_packet() -> dict[str, Any]:
    directions = _active_directions()
    coordinate_names = tuple(
        sorted({name for direction in directions for name in direction["direction"]})
    )
    direction_matrix = sp.Matrix(
        [
            [direction["direction"].get(name, sp.S.Zero) for direction in directions]
            for name in coordinate_names
        ]
    )
    _, pivot_positions = direction_matrix.rref()
    rank = len(pivot_positions)
    basis_matrix = direction_matrix[:, pivot_positions]
    _, independent_rows = basis_matrix.T.rref()
    coordinates = basis_matrix.extract(independent_rows, range(rank)).inv() * (
        direction_matrix.extract(independent_rows, range(ACTIVE_DIRECTION_COUNT))
    )
    if not (basis_matrix * coordinates - direction_matrix).is_zero_matrix:
        raise QuarticTC2MixedThirdJetRerankedReductionError("direction span residual")

    cubic_basis = tuple(itertools.combinations_with_replacement(range(rank), 3))
    cubic_index = {triple: index for index, triple in enumerate(cubic_basis)}
    if len(cubic_basis) != SYMMETRIC_CUBIC_DIMENSION:
        raise QuarticTC2MixedThirdJetRerankedReductionError("cubic dimension mismatch")

    def row(triple: tuple[int, int, int]) -> list[sp.Expr]:
        result = [sp.S.Zero] * len(cubic_basis)
        for left in range(rank):
            left_value = coordinates[left, triple[0]]
            if left_value == 0:
                continue
            for middle in range(rank):
                middle_value = coordinates[middle, triple[1]]
                if middle_value == 0:
                    continue
                for right in range(rank):
                    right_value = coordinates[right, triple[2]]
                    if right_value == 0:
                        continue
                    key = tuple(sorted((left, middle, right)))
                    index = cubic_index[key]
                    result[index] = sp.expand(
                        result[index] + left_value * middle_value * right_value
                    )
        return result

    diagonals = sp.Matrix([row((index, index, index)) for index in range(41)])
    selector = _mixed_selector()
    prior_prefix = sp.Matrix([row(triple) for triple in selector[:PRIOR_REDUCTION_PREFIX_COUNT]])
    stable_prefix = sp.Matrix([row(triple) for triple in selector[:STABLE_PREFIX_COUNT]])
    stable_evidence = diagonals.col_join(stable_prefix)
    _, pivot_coordinates = stable_evidence.rref()
    complement_coordinates = [
        index for index in range(len(cubic_basis)) if index not in pivot_coordinates
    ]
    active_obligations = [
        tuple(pivot_positions[index] for index in cubic_basis[column])
        for column in complement_coordinates
    ]
    completion_rows = sp.Matrix(
        [
            [sp.S.One if column == index else sp.S.Zero for column in range(len(cubic_basis))]
            for index in complement_coordinates
        ]
    )
    completion = stable_evidence.col_join(completion_rows)
    if completion.rank() != len(cubic_basis) or completion[:-1, :].rank() != len(cubic_basis) - 1:
        raise QuarticTC2MixedThirdJetRerankedReductionError("complement rank mismatch")

    selector_index = {triple: index for index, triple in enumerate(selector)}
    seed = _content_hash(
        {
            "basis_active_positions": list(pivot_positions),
            "stable_prefix_count": STABLE_PREFIX_COUNT,
            "stable_evidence_rank": len(pivot_coordinates),
        }
    )
    previous = seed
    obligations: list[dict[str, Any]] = []
    for obligation_index, triple in enumerate(active_obligations):
        body = {
            "obligation_index": obligation_index,
            "global_selector_index": selector_index[triple],
            "triple_kind": _triple_kind(triple),
            "active_position_triple": list(triple),
            "atom_index_triple": [int(directions[index]["atom_index"]) for index in triple],
            "atom_triple": [str(directions[index]["atom"]) for index in triple],
            "previous_obligation_sha256": previous,
        }
        record = {**body, "obligation_sha256": _content_hash(body)}
        obligations.append(record)
        previous = record["obligation_sha256"]

    return {
        "coordinate_names": list(coordinate_names),
        "active_direction_rank": rank,
        "basis_active_positions": list(pivot_positions),
        "symmetric_cubic_dimension": len(cubic_basis),
        "diagonal_evidence_rank": diagonals.rank(),
        "prior_576_prefix_rank": prior_prefix.rank(),
        "stable_1600_prefix_rank": stable_prefix.rank(),
        "stable_combined_evidence_rank": len(pivot_coordinates),
        "added_prefix_records": STABLE_PREFIX_COUNT - PRIOR_REDUCTION_PREFIX_COUNT,
        "rank_gain_over_prior_reduction": len(pivot_coordinates) - PRIOR_REDUCTION_RANK,
        "prior_reduced_obligation_count": PRIOR_REDUCTION_OBLIGATIONS,
        "reranked_obligation_count": len(obligations),
        "obligations_removed_by_added_evidence": PRIOR_REDUCTION_OBLIGATIONS - len(obligations),
        "reranked_obligation_kind_counts": dict(
            sorted(Counter(item["triple_kind"] for item in obligations).items())
        ),
        "first_selector_index": obligations[0]["global_selector_index"],
        "last_selector_index": obligations[-1]["global_selector_index"],
        "obligation_seed_sha256": seed,
        "obligation_tip_sha256": previous,
        "completion_rank": completion.rank(),
        "drop_final_obligation_rank": completion[:-1, :].rank(),
        "obligations": obligations,
    }


def run_quartic_tc2_mixed_third_jet_reranked_reduction_campaign(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2MixedThirdJetRerankedReductionError("unsupported schema_version")
        if any(
            config.get(key) != "fail_closed"
            for key in (
                "unevaluated_reduced_obligation_policy",
                "unprocessed_mixed_third_jet_policy",
                "full_tube_policy",
                "CK1_policy",
                "CK3_policy",
                "TC2_policy",
                "B7_policy",
                "global_H7_policy",
                "lifespan_policy",
            )
        ):
            raise QuarticTC2MixedThirdJetRerankedReductionError("closure policy mismatch")
        diagonal = _load_bound_json(root, config["diagonal_evidence"])
        chunks = [_load_bound_json(root, item) for item in config["stable_chunk_evidence"]]
        boundary = {
            item["label"]: _load_bound_json(root, item)
            for item in config["stopped_boundary_evidence"]
        }
        _validate_diagonal(diagonal)
        stable_triples, stable_tip = _validate_chunk_chain(chunks, config)
        _validate_stopped_boundary(boundary, chunks, config)
        packet = _reranked_packet()
        expected = config["expected_reranking"]
        actual = {
            key: packet[key]
            for key in (
                "active_direction_rank",
                "basis_active_positions",
                "symmetric_cubic_dimension",
                "stable_combined_evidence_rank",
                "rank_gain_over_prior_reduction",
                "reranked_obligation_count",
            )
        }
        if actual != expected:
            raise QuarticTC2MixedThirdJetRerankedReductionError("configured reranking mismatch")
        if stable_triples != list(_mixed_selector()[:STABLE_PREFIX_COUNT]) or any(
            item["global_selector_index"] < STABLE_PREFIX_COUNT
            for item in packet["obligations"]
        ):
            raise QuarticTC2MixedThirdJetRerankedReductionError("reranked selector overlap")
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_exact_stopped_chain_1600_rerank_447_obligations_"
                "no_inferred_passes_global_closure_fail_closed"
            ),
            "errors": [],
            "config_sha256": _content_hash(config),
            "stable_evidence": {
                "diagonal_content_sha256": diagonal["content_sha256"],
                "chunk_count": len(chunks),
                "chunk_content_sha256": [chunk["content_sha256"] for chunk in chunks],
                "mixed_prefix_records": len(stable_triples),
                "mixed_candidate_evaluations": len(stable_triples) * 12,
                "mixed_candidate_solvable": len(stable_triples) * 12,
                "mixed_candidate_obstructed": 0,
                "stable_resume_tip_sha256": stable_tip,
                "checkpoint_content_sha256": boundary["checkpoint"]["content_sha256"],
                "service_status_content_sha256": boundary["service_status"][
                    "content_sha256"
                ],
                "supervisor_state_content_sha256": boundary["supervisor_state"][
                    "content_sha256"
                ],
                "supervisor_status_content_sha256": boundary["supervisor_status"][
                    "content_sha256"
                ],
                "supervisor_stop_reason": "epoch_limit",
            },
            "exact_reranking": {
                key: value for key, value in packet.items() if key != "obligations"
            },
            "reranked_obligation_selector": {
                "selector": "basis_coordinate_complement_of_stopped_1600_evidence_row_space",
                "stable_remaining_mixed_triples": TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT,
                "exact_obligations": packet["reranked_obligation_count"],
                "candidate_evaluations_if_all_obligations_are_run": (
                    packet["reranked_obligation_count"] * 12
                ),
                "unevaluated_obligations_counted_as_passes": 0,
                "remaining_active_triples_counted_as_inferred_passes": 0,
                "obligations": packet["obligations"],
            },
            "theorem": {
                "statement": (
                    "The stopped exact 1600-record prefix and 41 diagonal evaluations span "
                    "233 functionals in the 680-dimensional symmetric cubic dual. The 447 "
                    "listed nonpivot basis-coordinate functionals are an exact minimal "
                    "complement. Only after those obligations are evaluated can linearity of "
                    "third jets, equal-eigenspace compression, candidate specialization, and "
                    "the Sylvester inverse close the full active-sector third jet."
                ),
                "exact_field": "Q(sqrt(2))",
                "completion_rank": packet["completion_rank"],
                "drop_final_obligation_rank": packet["drop_final_obligation_rank"],
                "minimal_complement": True,
            },
            "negative_controls": {
                "remove_one_reranked_obligation": {
                    "resulting_rank": packet["drop_final_obligation_rank"],
                    "full_rank": packet["symmetric_cubic_dimension"],
                    "rejected": True,
                },
                "count_unevaluated_obligations_as_passes": {
                    "inferred_pass_count": 0,
                    "rejected": True,
                },
                "promote_third_jet_to_full_tube": {
                    "missing": "fourth-and-higher residual jets or a nonlinear range theorem",
                    "rejected": True,
                },
            },
            "counts": {
                "stable_mixed_triples_evaluated": STABLE_PREFIX_COUNT,
                "stable_mixed_triples_remaining": TOTAL_MIXED_TRIPLES - STABLE_PREFIX_COUNT,
                "reranked_exact_obligations": packet["reranked_obligation_count"],
                "reranked_obligations_evaluated": 0,
                "reranked_obligations_passed": 0,
                "remaining_active_triples_inferred_passed": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
            "closure_ledger": {
                "reranked_reduction_theorem_proved": True,
                "all_447_reranked_obligations_closed": False,
                "all_12_300_mixed_third_jets_closed": False,
                "full_tube_Sylvester_identity": False,
                "CK1_closed": False,
                "CK3_closed": False,
                "TC2_closed": False,
                "B7_closed": False,
                "global_H7_closed": False,
                "lifespan_proved": False,
            },
            "claim": (
                "The stopped 1600-record chain raises exact evidence rank from 120 to 233 "
                "and reduces the residual exact selector from 560 to 447 obligations."
            ),
            "scope": (
                "This artifact evaluates none of the 447 residual obligations and infers no "
                "additional passes. Mixed closure, full tube, CK1, CK3, TC2, B7, global H7, "
                "and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "counts": {
                "stable_mixed_triples_evaluated": 0,
                "stable_mixed_triples_remaining": TOTAL_MIXED_TRIPLES,
                "reranked_exact_obligations": 0,
                "reranked_obligations_evaluated": 0,
                "reranked_obligations_passed": 0,
                "remaining_active_triples_inferred_passed": 0,
                "TC2_closures": 0,
                "B7_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_mixed_third_jet_reranked_reduction_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
