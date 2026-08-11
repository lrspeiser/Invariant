from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sigma_theory_compiler.reviewed_local_formula_epoch import (
    ReviewedLocalEpochError,
    build_readiness_artifact,
    run_bounded_mock_epoch,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/reviewed_local_formula_epoch.json"
ARTIFACT = ROOT / "runs/engine/reviewed-local-formula-epoch-status.json"


def test_bounded_epoch_exact_end_to_end_status_and_capability_cleanup(tmp_path: Path) -> None:
    env_name = "SIGMA_LOCAL_MOCK_LLM_CAPABILITY"
    prior = os.environ.get(env_name)
    status = run_bounded_mock_epoch(ROOT, tmp_path / "epoch")
    assert status["decision_counts"] == {"block": 1, "dedup": 1, "pass": 1, "reject": 1}
    assert status["proposal_quarantine_count"] == 4
    assert status["compiler_receipt_pass_count"] == 2
    assert status["candidate_count"] == 1
    assert status["next_stage_enqueue_count"] == 1
    assert status["policy_pass_count"] == 1
    assert status["crash_recovered_admission_attempt"] == 2
    assert status["lineage_preserved"] is True
    assert status["network_calls"] == 0
    assert status["paid_spend_usd"] == "0.000000"
    assert status["formula_body_persistence"] is False
    assert status["secret_or_capability_persistence"] is False
    assert os.environ.get(env_name) == prior


def test_bounded_epoch_core_is_deterministic_across_isolated_replay(tmp_path: Path) -> None:
    first = run_bounded_mock_epoch(ROOT, tmp_path / "first")
    second = run_bounded_mock_epoch(ROOT, tmp_path / "second")
    assert first == second
    assert first["core_sha256"] == "f4d1db005a8a03491e6239a7aa486a07bd112b0c6a6cb9571f593ca38377cdf0"


def test_checked_in_epoch_is_disabled_hash_bound_and_matches_artifact() -> None:
    built = build_readiness_artifact(ROOT, CONFIG)
    assert built == build_readiness_artifact(ROOT, CONFIG)
    assert built["default_execution_enabled"] is False
    assert built["network_calls"] == 0
    assert built["paid_spend_usd"] == "0.000000"
    assert built["maximum_total_usd"] == "500.000000"
    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == built


def test_component_hash_tamper_fails_closed(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text(encoding="utf-8"))
    first = next(iter(raw["component_sha256"]))
    raw["component_sha256"][first] = "0" * 64
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReviewedLocalEpochError, match="component binding mismatch"):
        build_readiness_artifact(ROOT, path)
