from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_mixed_third_jet_parallel_kernel import (
    QuarticTC2MixedThirdJetParallelKernelError,
    evaluate_mixed_triples_process_pool,
    select_resource_aware_worker_count,
)

REPO = Path(__file__).resolve().parents[1]


def _json(path: str) -> dict:
    return json.loads((REPO / path).read_text(encoding="utf-8"))


def test_resource_aware_worker_count_reserves_busy_processors() -> None:
    assert (
        select_resource_aware_worker_count(
            max_workers=12, logical_processors=24, cpu_utilization_percent=16.0
        )
        == 12
    )
    assert (
        select_resource_aware_worker_count(
            max_workers=12, logical_processors=24, cpu_utilization_percent=67.0
        )
        == 7
    )
    assert (
        select_resource_aware_worker_count(
            max_workers=12, logical_processors=24, cpu_utilization_percent=100.0
        )
        == 1
    )
    with pytest.raises(QuarticTC2MixedThirdJetParallelKernelError):
        select_resource_aware_worker_count(
            max_workers=0, logical_processors=24, cpu_utilization_percent=10.0
        )
    with pytest.raises(QuarticTC2MixedThirdJetParallelKernelError):
        select_resource_aware_worker_count(
            max_workers=4, logical_processors=24, cpu_utilization_percent=101.0
        )


def test_two_process_exact_kernel_matches_committed_sequential_records() -> None:
    diagonal = _json("runs/physics-language/quartic-tc2-diagonal-third-jet-campaign/campaign.json")
    committed = _json(
        "runs/physics-language/quartic-tc2-mixed-third-jet-continuation-service/"
        "chunks/offset-000064.json"
    )
    coefficients = {
        str(item["candidate_id"]): item["coefficients"] for item in diagonal["certificates"]
    }
    records = committed["triple_manifest"][:2]
    triples = [tuple(record["active_position_triple"]) for record in records]
    results = evaluate_mixed_triples_process_pool(triples, coefficients, worker_count=2)
    dynamic_keys = {
        "active_position_triple",
        "D3P55_nonzero_entries",
        "D3P55_sha256",
        "D3K55_nonzero_entries",
        "D3K55_sha256",
        "D3TC2_nonzero_entries",
        "D3TC2_sha256",
        "third_Sylvester_RHS_sha256",
        "symbolic_parameter_compatible",
        "symbolic_nonzero_equal_eigenspace_compressions",
        "symbolic_deltaK_ABC_Hermitian",
        "symbolic_deltaK_ABC_nonzero_entries",
        "symbolic_deltaK_ABC_rank",
        "symbolic_deltaK_ABC_sha256",
        "candidate_results",
        "obstructed_candidate_ids",
    }
    assert [{key: record[key] for key in dynamic_keys} for record in records] == results
