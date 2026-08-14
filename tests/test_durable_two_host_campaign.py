from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sigma_theory_compiler.durable_two_host_campaign import (
    DurableCampaignError,
    DurableTwoHostCampaign,
    DurationNotReachedError,
    StorageCeilingError,
    load_config,
    main,
    run_host_slice,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "durable_two_host_campaign.json"


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 13, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def _config() -> dict:
    return load_config(CONFIG)


def _fast_config() -> dict:
    config = _config()
    config["queue"]["lease_seconds"] = 1
    config["queue"]["dead_host_seconds"] = 2
    config["duration"]["heartbeat_interval_seconds"] = 0.05
    config["duration"]["poll_interval_seconds"] = 0.01
    return config


def test_two_independent_logical_hosts_claim_distinct_work_and_resume(tmp_path: Path) -> None:
    config = _fast_config()
    first = DurableTwoHostCampaign(tmp_path, config)
    second = DurableTwoHostCampaign(tmp_path, config)
    assert first.enqueue([{"ordinal": 1}, {"ordinal": 2}]) == {
        "accepted": 2,
        "duplicate": 0,
        "backpressured": 0,
    }
    first.register_host("host-a", "a-1")
    second.register_host("host-b", "b-1")
    lease_a = first.claim("host-a", "a-1")
    lease_b = second.claim("host-b", "b-1")
    assert lease_a is not None and lease_b is not None
    assert lease_a.work_id != lease_b.work_id
    assert first.heartbeat_lease(lease_a)
    assert second.heartbeat_lease(lease_b)
    assert first.finish(lease_a, {"decision": "control-a"})
    assert second.finish(lease_b, {"decision": "control-b"})
    first.close_host("host-a", "a-1", monotonic_elapsed_seconds=0, stop_reason="test")
    second.close_host("host-b", "b-1", monotonic_elapsed_seconds=0, stop_reason="test")
    assert first.status()["work_counts"] == {"succeeded": 2}

    resumed = DurableTwoHostCampaign(tmp_path, config)
    resumed.register_host("host-a", "a-2")
    resumed.close_host("host-a", "a-2", monotonic_elapsed_seconds=0, stop_reason="resume")
    assert resumed.status()["host_counts"] == {"stopped": 2}


def test_dead_host_recovery_fences_old_session_and_retries(tmp_path: Path) -> None:
    clock = MutableClock()
    campaign = DurableTwoHostCampaign(tmp_path, _fast_config(), clock=clock)
    campaign.enqueue([{"ordinal": 7, "case": "dead-host"}])
    campaign.register_host("host-a", "a-dead")
    old = campaign.claim("host-a", "a-dead")
    assert old is not None and old.attempt == 1
    clock.advance(3)
    recovery = campaign.recover_dead_hosts()
    assert recovery == {"dead_hosts": 1, "expired_leases": 0, "recovered": 1, "failed": 0}
    assert not campaign.heartbeat_host("host-a", "a-dead")
    assert not campaign.heartbeat_lease(old)
    assert not campaign.finish(old, {"decision": "must-not-land"})

    campaign.register_host("host-b", "b-recovery")
    retry = campaign.claim("host-b", "b-recovery")
    assert retry is not None and retry.attempt == 2 and retry.work_id == old.work_id
    assert campaign.finish(retry, {"decision": "recovered"})
    campaign.close_host(
        "host-b", "b-recovery", monotonic_elapsed_seconds=0, stop_reason="recovered"
    )
    status = campaign.status()
    assert status["work_counts"] == {"succeeded": 1}
    assert status["duration"]["dead_session_credit_policy"] == "zero_seconds"


def test_expired_lease_is_recovered_while_host_is_live(tmp_path: Path) -> None:
    clock = MutableClock()
    campaign = DurableTwoHostCampaign(tmp_path, _fast_config(), clock=clock)
    campaign.enqueue([{"ordinal": 11}])
    campaign.register_host("host-a", "a-live")
    old = campaign.claim("host-a", "a-live")
    assert old is not None
    clock.advance(1.25)
    assert campaign.heartbeat_host("host-a", "a-live")
    recovery = campaign.recover_dead_hosts()
    assert recovery == {"dead_hosts": 0, "expired_leases": 1, "recovered": 1, "failed": 0}
    assert not campaign.finish(old, {"decision": "stale"})
    retry = campaign.claim("host-a", "a-live")
    assert retry is not None and retry.attempt == 2
    assert campaign.finish(retry, {"decision": "fresh"})


def test_config_binding_and_event_chain_tamper_fail_closed(tmp_path: Path) -> None:
    config = _fast_config()
    campaign = DurableTwoHostCampaign(tmp_path, config)
    campaign.enqueue([{"ordinal": 13}])
    changed = _fast_config()
    changed["campaign_id"] = "different"
    with pytest.raises(DurableCampaignError, match="different config"):
        DurableTwoHostCampaign(tmp_path, changed)

    with sqlite3.connect(campaign.database) as connection:
        connection.execute("UPDATE events SET payload_json='{}' WHERE sequence=2")
    with pytest.raises(DurableCampaignError, match="event chain changed"):
        campaign.status()


def test_work_source_and_result_seals_reject_direct_database_tamper(tmp_path: Path) -> None:
    campaign = DurableTwoHostCampaign(tmp_path, _fast_config())
    campaign.enqueue([{"ordinal": 14, "source": "sealed"}])
    campaign.register_host("host-a", "a-seal")
    lease = campaign.claim("host-a", "a-seal")
    assert lease is not None
    assert campaign.finish(lease, {"decision": "sealed-result"})
    campaign.close_host("host-a", "a-seal", monotonic_elapsed_seconds=0, stop_reason="test")
    with sqlite3.connect(campaign.database) as connection:
        connection.execute(
            "UPDATE work SET result_json=? WHERE work_id=?",
            ('{"decision":"tampered-result"}', lease.work_id),
        )
    with pytest.raises(DurableCampaignError, match="result seal changed"):
        campaign.status()


def test_sqlite_wal_shm_ceiling_blocks_before_persistent_overrun(tmp_path: Path) -> None:
    config = _fast_config()
    config["storage"].update(
        maximum_sqlite_family_bytes=512 * 1024,
        maximum_transaction_reserve_bytes=96 * 1024,
        wal_autocheckpoint_pages=1,
    )
    config["queue"]["maximum_payload_bytes"] = 8192
    config["queue"]["maximum_result_bytes"] = 8192
    campaign = DurableTwoHostCampaign(tmp_path, config)
    blocked = False
    for ordinal in range(1000):
        try:
            campaign.enqueue([{"ordinal": ordinal, "padding": "x" * 7000}])
        except StorageCeilingError:
            blocked = True
            break
    assert blocked
    snapshot = campaign.storage_snapshot()
    assert snapshot["within_ceiling"]
    assert snapshot["total_sqlite_family_bytes"] <= snapshot["maximum_sqlite_family_bytes"]
    assert snapshot["paths_persisted"] is False


def test_real_short_slices_are_resumable_but_cannot_issue_six_hour_receipt(
    tmp_path: Path,
) -> None:
    config = _fast_config()
    campaign = DurableTwoHostCampaign(tmp_path, config)
    campaign.enqueue([{"ordinal": 21}, {"ordinal": 22}])
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(run_host_slice, campaign, host, maximum_slice_seconds=0.12)
            for host in ("host-a", "host-b")
        ]
        first, second = [future.result() for future in futures]
    resumed = run_host_slice(campaign, "host-a", maximum_slice_seconds=0.04)
    assert first["succeeded"] + second["succeeded"] == 2
    assert resumed["credited_seconds"] > 0
    status = campaign.status()
    assert 0 < status["duration"]["credited_wall_seconds"] < 21600
    assert status["duration"]["overlap_policy"] == "union_not_sum"
    assert status["duration"]["credited_wall_seconds"] < sum(
        status["duration"]["credited_seconds_by_host"].values()
    )
    assert not status["duration_receipt_eligible"]
    assert status["claims"]["six_hour_campaign_completed"] is False
    with pytest.raises(DurationNotReachedError, match="real >=6h"):
        campaign.build_duration_receipt()


def test_public_cli_blocks_premature_receipt_without_creating_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    state = tmp_path / "state"
    receipt = tmp_path / "six-hour-receipt.json"
    base = ["--config", str(CONFIG), "--state-directory", str(state)]
    assert main([*base, "init"]) == 0
    assert main([*base, "receipt", "--output", str(receipt)]) == 20
    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert output["decision"] == "BLOCK"
    assert not receipt.exists()


def test_production_config_has_no_fast_duration_escape_hatch() -> None:
    config = _config()
    assert config["duration"]["required_credited_seconds"] == 21600
    assert config["duration"]["maximum_run_slice_seconds"] >= 21600
    shortened = _config()
    shortened["duration"]["required_credited_seconds"] = 1
    with pytest.raises(DurableCampaignError, match="real-duration contract"):
        DurableTwoHostCampaign(Path.cwd() / ".must-not-create", shortened)
