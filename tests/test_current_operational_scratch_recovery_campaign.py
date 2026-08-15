from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import pytest

from sigma_theory_compiler.current_operational_scratch_recovery_campaign import (
    CONFIG_REL,
    RESULT_REL,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def receipt() -> dict[str, object]:
    value = json.loads((ROOT / RESULT_REL).read_text(encoding="utf-8"))
    validate_campaign(value, ROOT)
    return value


def test_current_resources_and_exact_operational_cursor_are_audited(receipt) -> None:
    resources = receipt["resource_admission"]
    assert len(resources["samples"]) == 3
    assert resources["all_samples_admitted"] is True
    assert float(resources["maximum_cpu_utilization_percent"]) < 92
    assert resources["minimum_available_ram_mib"] >= 32768
    batch = receipt["operational_audit"]["batch0003"]
    assert batch["executed_leaf_indices"] == [11, 12]
    assert batch["counts"]["cumulative_formally_checked_candidates"] == 226
    assert batch["counts"]["remaining_pending_formal_receipts"] == 11023
    assert batch["counts"]["candidate_promotions"] == 0
    assert batch["counts"]["rank_assignments"] == 0


def test_isolated_three_task_recovery_is_terminal_and_checkpointed(receipt) -> None:
    scratch = receipt["scratch_recovery"]
    assert scratch["tasks_admitted"] == {
        "accepted": 3,
        "backpressured": 0,
        "budget_rejected": 0,
        "duplicate": 0,
    }
    assert scratch["recovery"] == {"recovered": 1, "failed": 0}
    assert scratch["terminal_counts"] == {"succeeded": 3}
    assert [row["attempt"] for row in scratch["completed"]] == [2, 1, 1]
    assert scratch["checkpoint"]["sequence"] == 1
    assert scratch["database_inside_repository_runs"] is False


def test_production_live_state_is_metadata_only_and_claims_fail_closed(receipt) -> None:
    live = receipt["operational_audit"]["live_scheduler_metadata_only"]
    assert live["sqlite_opened"] is False
    assert live["wal_or_shm_opened"] is False
    assert live["lease_rows_read"] is False
    assert live["fresh_live_lease_claimed"] is False
    assert live["interpretation"] == "metadata_only_cannot_establish_current_lease_freshness"
    assert receipt["claims"] == {
        "external_process_signals": False,
        "gpu_or_cuda_access": False,
        "live_sqlite_opened": False,
        "network_access": False,
        "observations_opened": False,
        "production_namespace_written": False,
        "production_scheduler_freshness_established": False,
        "promotion_or_rank_write": False,
        "scientific_pass": False,
        "synthetic_control_only": True,
    }


def test_fresh_owned_scratch_replays_same_scheduler_control(receipt) -> None:
    samples = [
        {key: item for key, item in row.items() if key != "admitted"}
        for row in receipt["resource_admission"]["samples"]
    ]
    observed_at = receipt["operational_audit"]["observed_at"]
    with tempfile.TemporaryDirectory(prefix="invariant-current-recovery-") as directory:
        replay = build_campaign(ROOT, Path(directory) / "scratch", samples, observed_at=observed_at)
    assert replay == receipt


def test_nonempty_or_repository_runtime_scratch_is_rejected(tmp_path: Path, receipt) -> None:
    samples = [
        {key: item for key, item in row.items() if key != "admitted"}
        for row in receipt["resource_admission"]["samples"]
    ]
    observed_at = receipt["operational_audit"]["observed_at"]
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "foreign").write_text("foreign", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        build_campaign(ROOT, nonempty, samples, observed_at=observed_at)
    with pytest.raises(ValueError, match="may not use repository runtime"):
        build_campaign(
            ROOT, ROOT / "runs/forbidden-current-recovery", samples, observed_at=observed_at
        )


def test_config_and_resealed_claim_tamper_fail(receipt) -> None:
    config = json.loads((ROOT / CONFIG_REL).read_text(encoding="utf-8"))
    assert config["scratch_contract"]["tasks"] == 3
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["production_namespace_written"] = True
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ValueError, match="boundary changed"):
        validate_campaign(tampered, ROOT)
