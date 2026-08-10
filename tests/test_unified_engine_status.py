from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest

from sigma_theory_compiler.unified_engine_status import build_unified_snapshot, load_config

REPO = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [
    "runs/engine/rust-streaming-billion-status.json",
    "runs/engine/composite-promotion-overlay-production-status.json",
    "runs/engine/grammar-v3-parameter-cell-execution-status.json",
    "runs/engine/grammar-v3-evidence-pareto-report.json",
    "runs/engine/grammar-v3-followup-service-status.json",
    "runs/engine/grammar-v3-followup-queue-status.json",
    "configs/resource_profile_5090.json",
]
LABELS = [
    "billion_streaming",
    "promotion_overlay",
    "grammar_parameter_cells",
    "evidence_pareto",
    "followup_service",
    "followup_queue",
    "resource_profile",
]


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    specs = []
    for label, rel in zip(LABELS, SOURCE_PATHS, strict=True):
        source = REPO / rel
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        raw = target.read_bytes()
        value = json.loads(raw)
        claimed = value.get("content_sha256")
        if claimed is None:
            claimed = hashlib.sha256(_canonical(value)).hexdigest()
        specs.append({
            "label": label,
            "path": rel,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": claimed,
        })
    database = tmp_path / "runs/campaigns/watchdog.sqlite"
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE campaigns (
          campaign_id TEXT, state TEXT, deadline_utc TEXT, max_tasks INTEGER,
          tasks_started INTEGER, tasks_succeeded INTEGER, tasks_failed INTEGER,
          max_cycles INTEGER, cycles_completed INTEGER, stop_reason TEXT
        );
        CREATE TABLE tasks (task_type TEXT, status TEXT);
        CREATE TABLE candidates (status TEXT);
        CREATE TABLE evidence (outcome TEXT);
        CREATE TABLE llm_budgets (
          limit_microusd INTEGER, reserved_microusd INTEGER, spent_microusd INTEGER,
          max_calls INTEGER, calls_started INTEGER, calls_completed INTEGER
        );
        CREATE TABLE events (created_utc TEXT);
        INSERT INTO campaigns VALUES
          ('fixture','active','2026-08-21T00:00:00+00:00',100,4,1,0,8,1,NULL);
        INSERT INTO tasks VALUES ('covariant_lift','queued'),('llm_research','running'),
          ('candidate_dossier','deferred');
        INSERT INTO candidates VALUES ('active'),('rejected'),('deferred');
        INSERT INTO evidence VALUES ('pass'),('reject'),('unresolved');
        INSERT INTO llm_budgets VALUES (500000000,0,1250000,250,2,2);
        INSERT INTO events VALUES ('2026-08-10T20:00:00+00:00');
    """)
    connection.commit()
    connection.close()
    config = {
        "watchdog_database": "runs/campaigns/watchdog.sqlite",
        "watchdog_stale_after_seconds": 1800,
        "sources": specs,
    }
    return tmp_path, config, database


def test_read_only_snapshot_is_deterministic_and_does_not_mutate_database(tmp_path: Path) -> None:
    root, config, database = _fixture(tmp_path)
    before = hashlib.sha256(database.read_bytes()).hexdigest()
    sampled_at = datetime(2026, 8, 10, 20, 10, tzinfo=UTC)
    first = build_unified_snapshot(
        root,
        config,
        now_utc=sampled_at,
        physical_gpu={"availability": "available", "utilization_percent": 4.0},
    )
    second = build_unified_snapshot(
        root,
        config,
        now_utc=sampled_at,
        physical_gpu={"availability": "available", "utilization_percent": 99.0},
    )
    after = hashlib.sha256(database.read_bytes()).hexdigest()

    assert before == after
    assert first["core"] == second["core"]
    assert first["core_content_sha256"] == second["core_content_sha256"]
    assert first["volatile"] != second["volatile"]
    watchdog = first["core"]["campaign_watchdog"]
    assert watchdog["read_contract"] == "sqlite_uri_mode_ro_plus_query_only_transaction"
    assert watchdog["candidate_counts"] == {"active": 1, "deferred": 1, "rejected": 1}
    assert watchdog["evidence_outcome_counts"] == {"pass": 1, "reject": 1, "unresolved": 1}
    assert first["core"]["scheduler_lanes"]["llm_research"] == {
        "capacity": 4,
        "running": 1,
        "queued": 0,
        "scheduler_occupancy_fraction": 0.25,
    }
    assert first["volatile"]["campaign_watchdog_freshness"]["stale"] is False
    assert first["core"]["llm"]["spent_usd"] == 1.25
    assert "C:\\" not in json.dumps(first)


def test_stage_counts_and_missing_evaluator_blockers_are_not_collapsed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    result = build_unified_snapshot(
        root,
        config,
        now_utc=datetime(2026, 8, 10, 22, tzinfo=UTC),
        physical_gpu={"availability": "unavailable", "reason": "fixture"},
    )
    core = result["core"]
    assert core["billion_formula_streaming"]["sampled_static_stage"]["pass"] == 5855
    assert core["billion_formula_streaming"]["sampled_static_stage"]["normalized_outcomes"] == {
        "pass": 5855,
        "reject": None,
        "block": 0,
    }
    assert core["promotion_overlay"]["formal"] == {"pass": 0, "reject": 70, "block": 0}
    assert core["grammar_parameter_cells"]["scientific_decision_counts"] == {"blocked": 6}
    assert core["grammar_parameter_cells"]["normalized_scientific_outcomes"] == {
        "pass": 0,
        "reject": 0,
        "block": 6,
    }
    assert core["evidence_pareto"]["calibration_control_counts"] == {"pass": 13, "reject": 1}
    assert core["followup_service"]["followup_decision_counts"] == {"blocked": 10}
    assert core["followup_service"]["current_missing_evaluator_blockers"] == {
        "g3_global_lapse_dirac_contract": 1,
        "g3_uniform_interval_cell": 1,
        "g4_global_lapse_invertibility": 1,
        "g4_global_positive_energy": 1,
    }
    assert core["cross_pipeline_total"]["status"] == "not_computed"
    assert result["volatile"]["campaign_watchdog_freshness"]["stale"] is True
    assert result["volatile"]["campaign_watchdog_freshness"]["stale_source_reason"]


def test_hash_tamper_fails_closed(tmp_path: Path) -> None:
    root, config, _ = _fixture(tmp_path)
    target = root / SOURCE_PATHS[0]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="file hash mismatch"):
        build_unified_snapshot(root, config, physical_gpu={"availability": "unavailable"})


def test_portable_artifact_core_and_config_are_hash_bound() -> None:
    config = load_config(REPO / "configs/unified_engine_status.json")
    artifact = json.loads((REPO / "runs/engine/unified-engine-status.json").read_text())
    assert config["schema_version"] == "sigma-unified-engine-status-config-1.0"
    assert artifact["core"]["schema_version"] == "sigma-unified-engine-status-1.0"
    assert hashlib.sha256(_canonical(artifact["core"])).hexdigest() == artifact["core_content_sha256"]
    assert artifact["core"]["data_seals"] == {
        "dark_matter_or_halo_inputs": False,
        "observations_opened": False,
        "paid_llm_in_streaming_promotion_grammar": False,
        "redshift_distance_inputs": False,
    }
