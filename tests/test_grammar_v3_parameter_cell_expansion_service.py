from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.grammar_v3_parameter_cell_expansion_service import (
    GrammarV3ParameterCellExpansionService,
    portable_status,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "grammar_v3_parameter_cell_expansion_service.json"
ARTIFACT = ROOT / "runs" / "engine" / "grammar-v3-parameter-cell-expansion-service-status.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _enabled() -> dict:
    config = _load(CONFIG)
    config["execution_enabled"] = True
    return config


def _run(directory: Path) -> tuple[GrammarV3ParameterCellExpansionService, dict]:
    service = GrammarV3ParameterCellExpansionService(directory, _enabled(), ROOT)
    admission = service.enqueue()
    assert admission["accepted"] == 3
    result = service.run_bounded()
    assert result["executed_chunks"] == 3
    return service, result["status"]


def test_checked_in_service_is_disabled_and_chunks_are_exact(tmp_path: Path) -> None:
    service = GrammarV3ParameterCellExpansionService(tmp_path / "disabled", _load(CONFIG), ROOT)
    assert service.status()["execution_enabled"] is False
    assert [chunk["range"] for chunk in service.chunks] == [
        {"start": 0, "stop": 2},
        {"start": 2, "stop": 4},
        {"start": 4, "stop": 6},
    ]
    identities = [
        cell["parameter_cell_id"]
        for chunk in service.chunks
        for cell in chunk["parameter_cells"]
    ]
    assert len(identities) == len(set(identities)) == 6
    with pytest.raises(RuntimeError, match="disabled by config"):
        service.enqueue()


def test_real_reviewed_queue_chunks_replay_and_seals(tmp_path: Path) -> None:
    service, status = _run(tmp_path / "real")
    assert status["work_state_counts"] == {"succeeded": 3}
    assert status["decision_counts"] == {"blocked": 6}
    assert status["paid_llm_spend_usd"] == 0.0
    assert status["observational_data_opened"] is False
    assert status["data_eligibility"] == {**ELIGIBILITY, "passed": True}
    assert status["disk_bytes"] <= service.config["budget"]["maximum_disk_bytes"]
    assert all(record["result_sha256"] for record in status["chunk_records"])

    resumed = GrammarV3ParameterCellExpansionService(tmp_path / "real", _enabled(), ROOT)
    assert resumed.enqueue()["duplicate"] == 3
    replay = resumed.run_bounded()
    assert replay["executed_chunks"] == 0
    assert replay["status"]["chunk_records_root_sha256"] == status[
        "chunk_records_root_sha256"
    ]
    assert portable_status(replay["status"]) == portable_status(status)


def test_outer_lease_recovery_and_missing_callback_block(tmp_path: Path) -> None:
    directory = tmp_path / "recover"
    service = GrammarV3ParameterCellExpansionService(directory, _enabled(), ROOT)
    assert service.enqueue()["accepted"] == 3
    abandoned = service.coordinator.claim("cpu", "crashed", lease_seconds=-1)
    assert abandoned is not None
    resumed = GrammarV3ParameterCellExpansionService(directory, _enabled(), ROOT)
    result = resumed.run_bounded()
    assert result["recovered"] == {"recovered": 1, "failed": 0}
    assert result["status"]["decision_counts"] == {"blocked": 6}
    assert result["status"]["chunk_records"][0]["attempt"] == 2

    missing = GrammarV3ParameterCellExpansionService(
        tmp_path / "missing", _enabled(), ROOT, register_reviewed_callback=False
    )
    assert missing.enqueue()["accepted"] == 3
    blocked = missing.run_bounded()["status"]
    assert blocked["decision_counts"] == {"blocked": 6}
    for database in (tmp_path / "missing" / "chunks").glob("*.sqlite"):
        assert database.is_file()


def test_config_binding_and_stored_payload_tamper_fail_closed(tmp_path: Path) -> None:
    config = _enabled()
    config["parameter_cell_execution_config"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file hash mismatch"):
        GrammarV3ParameterCellExpansionService(tmp_path / "binding", config, ROOT)

    directory = tmp_path / "tamper"
    service = GrammarV3ParameterCellExpansionService(directory, _enabled(), ROOT)
    service.enqueue()
    with service.coordinator.connect() as connection:
        row = connection.execute(
            "SELECT work_id,payload_json FROM work ORDER BY ordinal LIMIT 1"
        ).fetchone()
        payload = json.loads(row["payload_json"])
        payload["range"]["stop"] = 5
        connection.execute(
            "UPDATE work SET payload_json=? WHERE work_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), row["work_id"]),
        )
    with pytest.raises(ValueError, match="chunk was tampered"):
        service.status()


def test_portable_bounded_artifact_is_exact(tmp_path: Path) -> None:
    _, status = _run(tmp_path / "artifact")
    assert portable_status(status) == _load(ARTIFACT)
