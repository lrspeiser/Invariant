from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_parameter_cell_execution import (
    GrammarV3ParameterCellExecution,
    _attest_reviewed_campaign_once_per_process,
    build_portable_parameter_cell_status,
    iter_parameter_cell_range,
    load_bound_callback_descriptor,
)
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
EXECUTION_CONFIG = ROOT / "configs" / "grammar_v3_parameter_cell_execution.json"
COORDINATOR_CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
RESOURCE_PROFILE = ROOT / "configs" / "resource_profile_5090.json"
MANIFEST = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-parameter-cell-execution-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _coordinator_config(execution: dict) -> dict:
    config = _load(COORDINATOR_CONFIG)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": execution["budget"]["maximum_tasks"],
        "maximum_attempts": 3,
        "lease_seconds": 2,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        "maximum_tasks": execution["budget"]["maximum_tasks"],
        "maximum_wall_seconds": execution["budget"]["maximum_wall_seconds"],
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 2}
    return config


def _adapter(
    database: Path, *, reviewed: bool = True, execution: dict | None = None
) -> GrammarV3ParameterCellExecution:
    execution = execution or _load(EXECUTION_CONFIG)
    coordinator = PersistentParallelSearch(
        database, _coordinator_config(execution), _load(RESOURCE_PROFILE)
    )
    descriptor = (
        load_bound_callback_descriptor(execution, EXECUTION_CONFIG)
        if reviewed
        else None
    )
    return GrammarV3ParameterCellExecution(
        coordinator,
        execution,
        MANIFEST,
        callback_descriptor=descriptor,
    )


def _run(database: Path) -> dict:
    adapter = _adapter(database)
    admission = adapter.enqueue()
    assert admission == {
        "accepted": 6,
        "duplicate": 0,
        "backpressured": 0,
        "budget_rejected": 0,
        "requested": 6,
        "parameter_cell_manifest_root_sha256": adapter.parameter_cell_manifest_root_sha256,
        "checkpoint_sha256": admission["checkpoint_sha256"],
    }
    execution = adapter.run_bounded()
    assert execution["executed"] == 6
    return build_portable_parameter_cell_status(execution["status"], adapter.config)


def test_real_finite_range_attests_once_executes_and_replays(tmp_path: Path) -> None:
    _attest_reviewed_campaign_once_per_process.cache_clear()
    database = tmp_path / "cells.sqlite"
    adapter = _adapter(database)
    assert _attest_reviewed_campaign_once_per_process.cache_info().misses == 1
    cells = list(iter_parameter_cell_range(adapter.seed_adapter.manifest, adapter.config))
    assert len(cells) == 6
    assert len({cell["parameter_cell_id"] for cell in cells}) == 6
    assert adapter.enqueue()["accepted"] == 6
    report = adapter.run_bounded()
    assert report["executed"] == 6
    assert report["status"]["work_state_counts"] == {"succeeded": 6}
    assert report["status"]["decision_counts"] == {"blocked": 6}
    assert report["status"]["paid_llm_spend_usd"] == 0.0
    assert report["status"]["observational_data_opened"] is False
    assert report["status"]["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert report["status"]["database_bytes"] <= adapter.config["budget"][
        "maximum_database_bytes"
    ]
    assert _attest_reviewed_campaign_once_per_process.cache_info().misses == 1

    resumed = _adapter(database)
    assert resumed.enqueue()["duplicate"] == 6
    assert resumed.status()["work_records_root_sha256"] == report["status"][
        "work_records_root_sha256"
    ]
    assert _attest_reviewed_campaign_once_per_process.cache_info().misses == 1


def test_expired_lease_recovery_and_missing_callback_fail_closed(tmp_path: Path) -> None:
    database = tmp_path / "recovery.sqlite"
    adapter = _adapter(database)
    assert adapter.enqueue()["accepted"] == 6
    abandoned = adapter.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = _adapter(database)
    recovered = resumed.run_bounded()
    assert recovered["recovered"] == {"recovered": 1, "failed": 0}
    assert recovered["status"]["decision_counts"] == {"blocked": 6}
    assert all(record["attempt"] >= 1 for record in recovered["status"]["work_records"])

    missing = _adapter(tmp_path / "missing.sqlite", reviewed=False)
    assert missing.callback_attestation["state"] == (
        "reviewed_candidate_compiler_formal_callback_missing"
    )
    assert missing.enqueue()["accepted"] == 6
    blocked = missing.run_bounded()["status"]
    assert blocked["decision_counts"] == {"blocked": 6}
    with missing.coordinator.connect() as connection:
        results = [
            json.loads(row[0])
            for row in connection.execute("SELECT result_json FROM work ORDER BY ordinal")
        ]
    assert all(
        result["reviewed_result"]["blocker"]
        == "reviewed_candidate_compiler_formal_callback_missing"
        for result in results
    )


def test_range_disk_and_descriptor_tamper_are_fail_closed(tmp_path: Path) -> None:
    execution = _load(EXECUTION_CONFIG)
    execution["range"] = {"start": 0, "stop": 7}
    execution["budget"]["maximum_tasks"] = 7
    with pytest.raises(ValueError, match="invalid or unbounded|finite six-cell"):
        _adapter(tmp_path / "range.sqlite", execution=execution)

    execution = _load(EXECUTION_CONFIG)
    execution["budget"]["maximum_database_bytes"] = 4096
    disk_limited = _adapter(tmp_path / "disk.sqlite", execution=execution)
    with pytest.raises(RuntimeError, match="disk budget"):
        disk_limited.enqueue()

    execution = _load(EXECUTION_CONFIG)
    execution["reviewed_callback"]["descriptor_file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="descriptor hash"):
        load_bound_callback_descriptor(execution, EXECUTION_CONFIG)


def test_portable_parameter_cell_status_is_exact(tmp_path: Path) -> None:
    rebuilt = _run(tmp_path / "portable.sqlite")
    assert rebuilt == _load(PORTABLE)
