from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from typing import Any

import sympy as sp

from .quartic_tc2_diagonal_third_jet_campaign import (
    _active_directions,
    _content_hash,
    _directional_taylor_packet_cached,
    _matrix_payload,
)
from .quartic_tc2_mixed_third_jet_chunk_campaign import (
    _candidate_result,
    _polarized_third_payload,
    _solve_sylvester,
)


class QuarticTC2MixedThirdJetParallelKernelError(ValueError):
    """Raised when a parallel mixed-third-jet evaluation is not deterministic."""


def select_resource_aware_worker_count(
    *, max_workers: int, logical_processors: int, cpu_utilization_percent: float
) -> int:
    """Reserve already-busy logical processors and return a bounded worker count."""
    if max_workers < 1 or logical_processors < 1:
        raise QuarticTC2MixedThirdJetParallelKernelError("invalid worker capacity")
    if not 0.0 <= cpu_utilization_percent <= 100.0:
        raise QuarticTC2MixedThirdJetParallelKernelError("invalid CPU utilization")
    idle_processors = int(logical_processors * max(0.0, 100.0 - cpu_utilization_percent) / 100.0)
    return max(1, min(max_workers, idle_processors))


def evaluate_mixed_triple_kernel(
    request: tuple[tuple[int, int, int], dict[str, dict[str, str]]],
) -> dict[str, Any]:
    """Evaluate one exact triple without provenance or record-chain side effects."""
    triple, coefficients = request
    if tuple(sorted(triple)) != triple or len(triple) != 3 or len(set(triple)) < 2:
        raise QuarticTC2MixedThirdJetParallelKernelError("invalid mixed triple")
    if len(coefficients) != 12:
        raise QuarticTC2MixedThirdJetParallelKernelError("candidate coefficient count mismatch")
    directions = _active_directions()
    if max(triple) >= len(directions):
        raise QuarticTC2MixedThirdJetParallelKernelError("mixed triple escapes direction basis")
    payload = _polarized_third_payload(triple, directions)
    rhs = payload["third_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    symbolic_solvable, symbolic_delta, symbolic_audit = _solve_sylvester(rhs)
    candidate_results: list[dict[str, Any]] = []
    obstructed_candidates: list[str] = []
    for candidate_id, candidate_coefficients in sorted(coefficients.items()):
        candidate = _candidate_result(rhs, alpha, c20, candidate_coefficients)
        if not candidate["solvable"]:
            obstructed_candidates.append(candidate_id)
        candidate_results.append({"candidate_id": candidate_id, **candidate})
    result = {
        "active_position_triple": list(triple),
        "D3P55_nonzero_entries": sum(value != 0 for value in payload["D3P55"]),
        "D3P55_sha256": _content_hash(_matrix_payload(payload["D3P55"])),
        "D3K55_nonzero_entries": sum(value != 0 for value in payload["D3K55"]),
        "D3K55_sha256": _content_hash(_matrix_payload(payload["D3K55"])),
        "D3TC2_nonzero_entries": sum(value != 0 for value in payload["D3TC2"]),
        "D3TC2_sha256": _content_hash(_matrix_payload(payload["D3TC2"])),
        "third_Sylvester_RHS_sha256": _content_hash(_matrix_payload(rhs)),
        "symbolic_parameter_compatible": symbolic_solvable,
        "symbolic_nonzero_equal_eigenspace_compressions": symbolic_audit[
            "nonzero_equal_eigenspace_compressions"
        ],
        "symbolic_deltaK_ABC_Hermitian": (
            symbolic_delta.equals(symbolic_delta.T) if symbolic_solvable else False
        ),
        "symbolic_deltaK_ABC_nonzero_entries": sum(value != 0 for value in symbolic_delta),
        "symbolic_deltaK_ABC_rank": (symbolic_delta.rank() if symbolic_solvable else None),
        "symbolic_deltaK_ABC_sha256": (
            _content_hash(_matrix_payload(symbolic_delta)) if symbolic_solvable else None
        ),
        "candidate_results": candidate_results,
        "obstructed_candidate_ids": obstructed_candidates,
    }
    _directional_taylor_packet_cached.cache_clear()
    return result


def evaluate_mixed_triples_process_pool(
    triples: list[tuple[int, int, int]],
    coefficients: dict[str, dict[str, str]],
    *,
    worker_count: int,
) -> list[dict[str, Any]]:
    """Evaluate a bounded ordered batch using isolated spawn workers."""
    if not triples or worker_count < 1 or worker_count > len(triples):
        raise QuarticTC2MixedThirdJetParallelKernelError("invalid parallel batch")
    requests = [(triple, coefficients) for triple in triples]
    if worker_count == 1:
        results = [evaluate_mixed_triple_kernel(request) for request in requests]
    else:
        with ProcessPoolExecutor(
            max_workers=worker_count, mp_context=get_context("spawn")
        ) as executor:
            results = list(executor.map(evaluate_mixed_triple_kernel, requests, chunksize=1))
    if [result["active_position_triple"] for result in results] != [
        list(triple) for triple in triples
    ]:
        raise QuarticTC2MixedThirdJetParallelKernelError("parallel result order mismatch")
    return results
