import json
from pathlib import Path

from sigma_theory_compiler.persistent_parallel_search import PersistentParallelSearch
from sigma_theory_compiler.persistent_parallel_supervisor import (
    PersistentParallelSupervisor,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "persistent_parallel_search_5090.json"
PROFILE = ROOT / "configs" / "resource_profile_5090.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _config(*, lease_seconds: int = 2) -> dict:
    config = _load(CONFIG)
    config["queue"] = {
        **config["queue"],
        "maximum_pending_work": 64,
        "maximum_attempts": 3,
        "lease_seconds": lease_seconds,
        "checkpoint_every_completions": 2,
    }
    config["budget"] = {
        **config["budget"],
        "maximum_tasks": 64,
        "maximum_wall_seconds": 60,
    }
    config["cpu"] = {**config["cpu"], "maximum_workers": 2}
    config["gpu"] = {**config["gpu"], "maximum_batch_candidates": 100}
    config["supervisor"] = {
        "cpu_workers": 2,
        "gpu_workers": 1,
        "cpu_evaluator": "synthetic_cpu",
        "gpu_evaluator": "synthetic_gpu_owner",
        "worker_poll_seconds": 0.01,
        "telemetry_interval_seconds": 0.02,
        "refill_interval_seconds": 0.01,
        "maximum_telemetry_bytes": 4 * 1024 * 1024,
        "maximum_wall_seconds_per_run": 10,
        "maximum_process_restarts": 4,
        "shutdown_grace_seconds": 1,
    }
    return config


def _results(coordinator: PersistentParallelSearch) -> list[dict]:
    with coordinator.connect() as connection:
        return [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT result_json FROM work WHERE state='succeeded' ORDER BY ordinal,lane"
            )
        ]


def test_spawned_cpu_and_single_gpu_owner_execute_queue_end_to_end(tmp_path: Path) -> None:
    config = _config()
    profile = _load(PROFILE)
    database = tmp_path / "parallel.sqlite"
    telemetry = tmp_path / "telemetry.jsonl"
    coordinator = PersistentParallelSearch(database, config, profile)
    assert coordinator.enqueue(
        [
            {"ordinal": index, "formula": f"R+{index}*X", "sleep_ms": 80}
            for index in range(4)
        ],
        lane="cpu",
    )["accepted"] == 4
    assert coordinator.enqueue(
        [
            {"ordinal": 100 + index, "candidate_count": 25, "sleep_ms": 80}
            for index in range(2)
        ],
        lane="gpu",
    )["accepted"] == 2
    report = PersistentParallelSupervisor(
        database, config, profile, telemetry
    ).run(maximum_wall_seconds=8)
    assert report["stop_reason"] == "queue_drained"
    assert report["final_telemetry"]["counts"]["succeeded"] == 6
    assert report["process_starts"] == 3
    assert report["process_crashes"] == 0
    assert report["paid_llm_calls_made"] == 0
    assert report["utilization"]["cpu"]["peak"] == 1.0
    assert report["utilization"]["gpu"]["peak"] == 1.0
    records = [json.loads(line) for line in telemetry.read_text().splitlines()]
    assert len(records) == report["telemetry_records_written"]
    assert all(record["execution"]["paid_llm_calls_enabled"] is False for record in records)
    results = _results(coordinator)
    assert len(results) == 6
    assert {item["evaluator"] for item in results} == {
        "synthetic_cpu",
        "synthetic_gpu_owner",
    }


def test_clean_restart_preserves_results_seeds_and_checkpoint_chain(tmp_path: Path) -> None:
    config = _config()
    profile = _load(PROFILE)
    database = tmp_path / "restart.sqlite"
    telemetry = tmp_path / "restart-telemetry.jsonl"
    first = PersistentParallelSearch(database, config, profile)
    first.enqueue([{"ordinal": 1, "formula": "R+X"}], lane="cpu")
    first_report = PersistentParallelSupervisor(
        database, config, profile, telemetry
    ).run(maximum_wall_seconds=5)
    first_result = _results(first)[0]
    first_sequence = first_report["checkpoint"]["sequence"]

    resumed = PersistentParallelSearch(database, config, profile)
    resumed.enqueue([{"ordinal": 2, "formula": "R-X"}], lane="cpu")
    second_report = PersistentParallelSupervisor(
        database, config, profile, telemetry
    ).run(maximum_wall_seconds=5)
    results = _results(resumed)
    assert results[0] == first_result
    assert len(results) == 2
    assert second_report["checkpoint"]["sequence"] > first_sequence
    lines = telemetry.read_text().splitlines()
    assert len(lines) == (
        first_report["telemetry_records_written"]
        + second_report["telemetry_records_written"]
    )


def test_hard_worker_crash_expires_lease_and_replacement_recovers(tmp_path: Path) -> None:
    config = _config(lease_seconds=1)
    config["supervisor"] = {**config["supervisor"], "cpu_workers": 1, "gpu_workers": 0}
    profile = _load(PROFILE)
    database = tmp_path / "crash.sqlite"
    telemetry = tmp_path / "crash-telemetry.jsonl"
    coordinator = PersistentParallelSearch(database, config, profile)
    coordinator.enqueue(
        [
            {
                "ordinal": 7,
                "formula": "R+crash*X",
                "hard_crash_on_attempt": 1,
            }
        ],
        lane="cpu",
        max_attempts=2,
    )
    report = PersistentParallelSupervisor(
        database, config, profile, telemetry
    ).run(maximum_wall_seconds=6)
    assert report["stop_reason"] == "queue_drained"
    assert report["process_crashes"] >= 1
    assert report["process_restarts"] >= 1
    assert report["final_telemetry"]["recovered_leases"] == 1
    assert report["final_telemetry"]["counts"]["succeeded"] == 1
    with coordinator.connect() as connection:
        row = connection.execute("SELECT attempt,state FROM work").fetchone()
    assert dict(row) == {"attempt": 2, "state": "succeeded"}


def test_telemetry_disk_cap_is_explicit_and_reported(tmp_path: Path) -> None:
    config = _config()
    config["supervisor"] = {
        **config["supervisor"],
        "cpu_workers": 0,
        "gpu_workers": 0,
        "maximum_telemetry_bytes": 1,
    }
    report = PersistentParallelSupervisor(
        tmp_path / "capped.sqlite",
        config,
        _load(PROFILE),
        tmp_path / "capped.jsonl",
    ).run(maximum_wall_seconds=1)
    assert report["telemetry_records_written"] == 0
    assert report["telemetry_records_dropped_by_disk_cap"] == 1


def test_external_stop_preserves_queued_work_for_clean_resume(tmp_path: Path) -> None:
    config = _config()
    profile = _load(PROFILE)
    database = tmp_path / "external-stop.sqlite"
    stop = tmp_path / "stop.request"
    coordinator = PersistentParallelSearch(database, config, profile)
    coordinator.enqueue([{"ordinal": 1, "formula": "R+X"}], lane="cpu")
    stop.write_text("stop\n", encoding="utf-8")
    stopped = PersistentParallelSupervisor(
        database, config, profile, tmp_path / "external-stop.jsonl"
    ).run(external_stop_path=stop)
    assert stopped["stop_reason"] == "external_stop_requested"
    assert coordinator.telemetry()["counts"]["queued"] == 1
    stop.unlink()
    resumed = PersistentParallelSupervisor(
        database, config, profile, tmp_path / "external-stop.jsonl"
    ).run(external_stop_path=stop)
    assert resumed["stop_reason"] == "queue_drained"
    assert coordinator.telemetry()["counts"]["succeeded"] == 1
