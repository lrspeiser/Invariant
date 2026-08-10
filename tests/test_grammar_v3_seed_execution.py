from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_seed_execution import (
    CALLBACK_RESULT_SCHEMA,
    CALLBACK_SCHEMA,
    GrammarV3SeedExecution,
)
from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-manifest.json"
CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"
MANIFEST_FILE_SHA = "8c9a2454a71b9a6deb32f12c875a4d8cad8a1da5d78d7ded09d1798641b7d290"
MANIFEST_CONTENT_SHA = "e28ad576a68648f11892a3ff1fff5b7e18057f1ba0b27c89d538499e46de171b"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def reviewed_seed_callback(seed: dict, context: dict) -> dict:
    return {
        "schema_version": CALLBACK_RESULT_SCHEMA,
        "decision": "blocked",
        "candidate_compilation": {
            "seed_id": seed["seed_id"],
            "operator_atoms": seed["operator_atoms"],
            "compilation_sha256": _sha(
                {
                    "seed_lineage_sha256": seed["seed_lineage_sha256"],
                    "input_lineage_sha256": context["input_lineage_sha256"],
                }
            ),
        },
        "formal_result": {
            "decision": "blocked",
            "blocker": "reviewed_fixture_formal_gate_not_implemented",
        },
        "blocker": "reviewed_fixture_formal_gate_not_implemented",
        "data_eligibility": dict(ELIGIBILITY),
    }


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
    return {
        "schema_version": CALLBACK_SCHEMA,
        "callback_id": "reviewed-seed-compiler-formal-fixture-v1",
        "callback": "test_grammar_v3_seed_execution:reviewed_seed_callback",
        "artifact_path": str(Path(__file__).resolve()),
        "artifact_sha256": _file_sha(Path(__file__).resolve()),
        "data_eligibility": dict(ELIGIBILITY),
    }


def _coordinator(path: Path) -> PersistentParallelSearch:
    return PersistentParallelSearch(path, _config(), _load(PROFILE))


def _adapter(
    coordinator: PersistentParallelSearch,
    descriptor: dict | None,
    manifest: Path = MANIFEST,
    *,
    file_sha: str = MANIFEST_FILE_SHA,
    content_sha: str = MANIFEST_CONTENT_SHA,
) -> GrammarV3SeedExecution:
    return GrammarV3SeedExecution(
        coordinator,
        manifest,
        expected_manifest_file_sha256=file_sha,
        expected_manifest_content_sha256=content_sha,
        callback_descriptor=descriptor,
    )


def test_six_seed_queue_recovers_lease_replays_and_calls_reviewed_interface(
    tmp_path: Path,
) -> None:
    database = tmp_path / "seed-queue.sqlite"
    first = _adapter(_coordinator(database), _descriptor())
    admitted = first.enqueue()
    assert admitted["accepted"] == admitted["requested"] == 6
    initial = first.status()
    work_ids = [record["work_id"] for record in initial["work_records"]]
    adapter_ids = [record["adapter_work_id"] for record in initial["work_records"]]
    assert len(set(work_ids)) == len(set(adapter_ids)) == 6

    abandoned = first.coordinator.claim("cpu", "crashed-worker", lease_seconds=-1)
    assert abandoned is not None
    resumed = _adapter(_coordinator(database), _descriptor())
    assert resumed.recovered_on_start == {"recovered": 1, "failed": 0}
    completed = resumed.run_bounded(maximum_tasks=6)
    status = completed["status"]
    assert status["work_state_counts"] == {"succeeded": 6}
    assert status["decision_counts"] == {"blocked": 6}
    assert status["recovered_leases"] == 1
    assert status["checkpoint_sequence"] >= 2
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["paid_llm_spend_usd"] == 0.0

    replay = _adapter(_coordinator(database), _descriptor())
    duplicated = replay.enqueue()
    assert duplicated["accepted"] == 0
    assert duplicated["duplicate"] == 6
    replay_status = replay.status()
    assert [record["work_id"] for record in replay_status["work_records"]] == work_ids
    assert replay_status["work_records_root_sha256"] == status[
        "work_records_root_sha256"
    ]
    with replay.coordinator.connect() as connection:
        connection.execute(
            "UPDATE work SET seed=seed+1 WHERE work_id=(SELECT work_id FROM work LIMIT 1)"
        )
    with pytest.raises(ValueError, match="coordinator identity"):
        replay.status()


def test_missing_reviewed_callback_blocks_seed_without_opening_data(tmp_path: Path) -> None:
    adapter = _adapter(_coordinator(tmp_path / "missing.sqlite"), None)
    assert adapter.enqueue()["accepted"] == 6
    run = adapter.run_bounded(maximum_tasks=1)
    assert run["status"]["decision_counts"] == {"blocked": 1}
    with adapter.coordinator.connect() as connection:
        result = json.loads(
            connection.execute(
                "SELECT result_json FROM work WHERE state='succeeded'"
            ).fetchone()[0]
        )
    assert result["reviewed_result"]["blocker"] == (
        "reviewed_candidate_compiler_formal_callback_missing"
    )
    assert result["paid_llm_spend_usd"] == 0.0
    with pytest.raises(ValueError, match="changed grammar-v3 seed adapter"):
        _adapter(_coordinator(tmp_path / "missing.sqlite"), _descriptor())


def test_manifest_seed_and_callback_tampering_fail_closed(tmp_path: Path) -> None:
    tampered = tmp_path / "manifest.json"
    value = _load(MANIFEST)
    value["scalable_generator_hook"]["concrete_seeds"][0]["parameters"] = {"c1": "9"}
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    value["content_sha256"] = _sha(body)
    tampered.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    with pytest.raises(ValueError, match="concrete seed lineage"):
        _adapter(
            _coordinator(tmp_path / "tampered.sqlite"),
            None,
            tampered,
            file_sha=_file_sha(tampered),
            content_sha=value["content_sha256"],
        )

    descriptor = _descriptor()
    descriptor["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="callback artifact hash"):
        _adapter(_coordinator(tmp_path / "callback-tampered.sqlite"), descriptor)

    with pytest.raises(ValueError, match="manifest file hash"):
        _adapter(
            _coordinator(tmp_path / "manifest-hash.sqlite"),
            None,
            MANIFEST,
            file_sha="0" * 64,
        )
