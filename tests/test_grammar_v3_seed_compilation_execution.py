from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_seed_compilation_execution_campaign import (
    run_grammar_v3_seed_compilation_execution,
)
from sigma_theory_compiler.grammar_v3_seed_execution import GrammarV3SeedExecution
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json"
DESCRIPTOR = ROOT / "configs" / "grammar_v3_seed_compilation_callback.json"
CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
PORTABLE = ROOT / "runs" / "engine" / "grammar-v3-seed-compilation-execution-status.json"
CAMPAIGN_UPSTREAM = (
    ROOT / "configs" / "covariant_grammar_v3_seed_compilation_campaign.json",
    ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json",
    ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json",
    ROOT / "runs" / "formal-controls-v1" / "formal-controls.json",
    ROOT / "configs" / "covariant_action_grammar.json",
    ROOT / "configs" / "covariant_field_contract.json",
)
MANIFEST_FILE_SHA = "8c9a2454a71b9a6deb32f12c875a4d8cad8a1da5d78d7ded09d1798641b7d290"
MANIFEST_CONTENT_SHA = "e28ad576a68648f11892a3ff1fff5b7e18057f1ba0b27c89d538499e46de171b"
DESCRIPTOR_FILE_SHA = "27bd97a37155cce3708ee114f6e22fadde08270d3ff93f8289130379458ce1ed"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config() -> dict:
    config = _load(CONFIG)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 6,
        "maximum_attempts": 3,
        "lease_seconds": 2,
        "checkpoint_every_completions": 1,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 6,
        "maximum_wall_seconds": 120,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 2}
    return config


def _descriptor() -> dict:
    descriptor = _load(DESCRIPTOR)
    descriptor["artifact_path"] = str((ROOT / descriptor["artifact_path"]).resolve())
    return descriptor


def _adapter(database: Path) -> GrammarV3SeedExecution:
    coordinator = PersistentParallelSearch(database, _config(), _load(PROFILE))
    return GrammarV3SeedExecution(
        coordinator,
        MANIFEST,
        expected_manifest_file_sha256=MANIFEST_FILE_SHA,
        expected_manifest_content_sha256=MANIFEST_CONTENT_SHA,
        callback_descriptor=_descriptor(),
    )


def _run(database: Path) -> dict:
    return run_grammar_v3_seed_compilation_execution(
        database,
        _config(),
        _load(PROFILE),
        manifest_path=MANIFEST,
        manifest_file_sha256=MANIFEST_FILE_SHA,
        manifest_content_sha256=MANIFEST_CONTENT_SHA,
        callback_descriptor_path=DESCRIPTOR,
        callback_descriptor_file_sha256=DESCRIPTOR_FILE_SHA,
    )


def test_real_campaign_callback_executes_six_blocked_seeds_and_replays(
    tmp_path: Path,
) -> None:
    database = tmp_path / "real-seeds.sqlite"
    upstream_before = {path: _file_sha(path) for path in CAMPAIGN_UPSTREAM}
    report = _run(database)
    assert {path: _file_sha(path) for path in CAMPAIGN_UPSTREAM} == upstream_before
    assert report["seed_count"] == 6
    assert report["decision_counts"] == {"blocked": 6}
    assert report["work_state_counts"] == {"succeeded": 6}
    assert report["checkpoint_sequence"] == 2
    assert report["observational_data_opened"] is False
    assert report["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert report["paid_llm_spend_usd"] == 0.0
    assert len(report["portable_result_registry_root_sha256"]) == 64
    assert len(report["work_records_root_sha256"]) == 64

    replay = _adapter(database)
    admission = replay.enqueue()
    assert admission["accepted"] == 0
    assert admission["duplicate"] == 6
    replay_status = replay.status()
    assert replay_status["decision_counts"] == {"blocked": 6}
    assert replay_status["work_records_root_sha256"] == report[
        "work_records_root_sha256"
    ]
    with replay.coordinator.connect() as connection:
        results = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT result_json FROM work ORDER BY ordinal"
            )
        ]
    assert all(
        item["reviewed_result"]["candidate_compilation"][
            "campaign_result_file_sha256"
        ]
        == "88c72002d12fd57a8ef79b166363319fe5dc372b52901c1e581bb382fa3e0c21"
        for item in results
    )
    assert all(item["reviewed_result"]["formal_result"]["decision"] == "blocked" for item in results)


def test_real_callback_recovers_expired_lease_and_descriptor_tamper_fails(
    tmp_path: Path,
) -> None:
    database = tmp_path / "recovery.sqlite"
    adapter = _adapter(database)
    assert adapter.enqueue()["accepted"] == 6
    abandoned = adapter.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = _adapter(database)
    assert resumed.recovered_on_start == {"recovered": 1, "failed": 0}
    completed = resumed.run_bounded(maximum_tasks=6)
    assert completed["status"]["work_state_counts"] == {"succeeded": 6}
    assert completed["status"]["decision_counts"] == {"blocked": 6}

    with pytest.raises(ValueError, match="descriptor file hash"):
        run_grammar_v3_seed_compilation_execution(
            tmp_path / "tampered.sqlite",
            _config(),
            _load(PROFILE),
            manifest_path=MANIFEST,
            manifest_file_sha256=MANIFEST_FILE_SHA,
            manifest_content_sha256=MANIFEST_CONTENT_SHA,
            callback_descriptor_path=DESCRIPTOR,
            callback_descriptor_file_sha256="0" * 64,
        )


def test_portable_status_is_exact_bounded_rebuild(tmp_path: Path) -> None:
    artifact = _load(PORTABLE)
    rebuilt = _run(tmp_path / "portable.sqlite")
    assert rebuilt == artifact
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert _file_sha(DESCRIPTOR) == DESCRIPTOR_FILE_SHA
