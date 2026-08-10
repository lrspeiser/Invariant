import json
from pathlib import Path

import pytest

from sigma_theory_compiler.persistent_parallel_search import (
    PersistentParallelSearch,
    plan_parallel_capacity,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _small_config() -> dict:
    config = _load(CONFIG)
    config["queue"] = {**config["queue"], "maximum_pending_work": 2, "lease_seconds": 2}
    config["budget"] = {**config["budget"], "maximum_tasks": 3, "maximum_wall_seconds": 60}
    config["gpu"] = {**config["gpu"], "maximum_batch_candidates": 100}
    return config


def test_5090_capacity_and_batch_plan_are_resource_bound() -> None:
    plan = plan_parallel_capacity(_load(PROFILE), _load(CONFIG))
    assert plan["cpu"]["workers"] == 16
    assert plan["gpu"]["available"]
    assert plan["gpu"]["workers"] == 1
    assert plan["gpu"]["batch_candidates"] == 2_000_000
    assert plan["simultaneous_lane_owners"] == 17
    assert plan["paid_llm_workers"] == 0
    assert plan["gpu"]["measured_candidates_per_second"] > 7_500_000


def test_bounded_queue_deterministic_seed_checkpoint_and_resume(tmp_path: Path) -> None:
    database = tmp_path / "parallel.sqlite"
    config = _small_config()
    profile = _load(PROFILE)
    coordinator = PersistentParallelSearch(database, config, profile)
    items = [
        {"ordinal": 10, "formula": "R+a*X"},
        {"ordinal": 11, "formula": "R+b*X"},
        {"ordinal": 12, "formula": "R+c*X"},
    ]
    admitted = coordinator.enqueue(items, lane="cpu")
    assert admitted == {
        "accepted": 2,
        "duplicate": 0,
        "backpressured": 1,
        "budget_rejected": 0,
    }
    first = coordinator.claim("cpu", "worker-1")
    assert first and first.ordinal == 10
    seed = first.seed
    assert coordinator.finish(first, "worker-1", {"score": 1})
    second_admission = coordinator.enqueue([items[2]], lane="cpu")
    assert second_admission["accepted"] == 1
    checkpoint = coordinator.checkpoint()
    assert checkpoint["sequence"] == 1

    resumed = PersistentParallelSearch(database, config, profile)
    assert resumed.telemetry()["checkpoint_sequence"] == 1
    duplicate = resumed.enqueue([items[0]], lane="cpu")
    assert duplicate["duplicate"] == 1
    with resumed.connect() as connection:
        stored_seed = connection.execute(
            "SELECT seed FROM work WHERE ordinal=10 AND lane='cpu'"
        ).fetchone()[0]
    assert stored_seed == seed
    changed = json.loads(json.dumps(config))
    changed["determinism"]["master_seed"] += 1
    with pytest.raises(ValueError, match="different execution config"):
        PersistentParallelSearch(database, changed, profile)


def test_expired_lease_recovers_and_telemetry_exposes_underutilization(tmp_path: Path) -> None:
    coordinator = PersistentParallelSearch(
        tmp_path / "recovery.sqlite", _small_config(), _load(PROFILE)
    )
    assert coordinator.enqueue(
        [{"ordinal": 1, "candidate_count": 10}], lane="gpu", max_attempts=2
    )["accepted"] == 1
    lease = coordinator.claim("gpu", "dead-gpu", lease_seconds=-1)
    assert lease and lease.attempt == 1
    assert coordinator.recover_expired() == {"recovered": 1, "failed": 0}
    resumed = coordinator.claim("gpu", "replacement", lease_seconds=30)
    assert resumed and resumed.seed == lease.seed and resumed.attempt == 2
    assert coordinator.finish(resumed, "replacement", {"survivors": 2})
    telemetry = coordinator.telemetry()
    assert telemetry["recovered_leases"] == 1
    assert telemetry["lanes"]["cpu"]["utilization"] == 0
    assert "cpu:queue_starved" in telemetry["underutilization_warnings"]
    assert telemetry["paid_llm_calls_enabled"] is False


def test_task_budget_stops_admission_and_failed_work_is_terminal(tmp_path: Path) -> None:
    coordinator = PersistentParallelSearch(
        tmp_path / "budget.sqlite", _small_config(), _load(PROFILE)
    )
    assert coordinator.enqueue(
        [{"ordinal": 1}, {"ordinal": 2}], lane="cpu", max_attempts=1
    )["accepted"] == 2
    lease = coordinator.claim("cpu", "worker")
    assert lease
    assert coordinator.fail(lease, "worker", "synthetic crash") == "failed"
    assert coordinator.enqueue([{"ordinal": 3}], lane="cpu")["accepted"] == 1
    rejected = coordinator.enqueue([{"ordinal": 4}], lane="cpu")
    assert rejected["budget_rejected"] == 1
    telemetry = coordinator.telemetry()
    assert telemetry["budget"]["remaining_tasks"] == 0
    assert telemetry["counts"]["failed"] == 1

    timed = PersistentParallelSearch(
        tmp_path / "time-budget.sqlite", _small_config(), _load(PROFILE)
    )
    with timed.connect() as connection:
        connection.execute(
            "UPDATE execution SET deadline_utc='2000-01-01T00:00:00+00:00'"
        )
    assert timed.enqueue([{"ordinal": 99}], lane="cpu")["budget_rejected"] == 1

    paid = _small_config()
    paid["external_paid_llm_calls"] = True
    with pytest.raises(ValueError, match="paid LLM calls"):
        PersistentParallelSearch(tmp_path / "paid.sqlite", paid, _load(PROFILE))
