from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from sigma_theory_compiler.alpha_operational_rehearsal import (
    _gpu_control,
    run_operational_rehearsal,
    validate_operational_receipt,
)

ROOT = Path(__file__).resolve().parents[1]


def test_owned_recovery_zero_spend_and_gpu_not_started(tmp_path: Path) -> None:
    receipt = run_operational_rehearsal(ROOT, tmp_path / "owned")
    validate_operational_receipt(receipt)
    scheduler = receipt["scheduler_control"]
    assert scheduler["recovery"] == {"recovered": 1, "failed": 0}
    assert scheduler["counts"] == {"succeeded": 1}
    assert scheduler["checkpoint"]["sequence"] == 1
    assert scheduler["worker_counts"] == {"cpu": 1, "gpu": 0}
    assert receipt["llm_control"]["budget"]["settled_usd"] == "0.000000"
    assert receipt["llm_control"]["provider_calls"] == 0
    assert receipt["gpu_control"]["execution_started"] is False
    assert receipt["gpu_control"]["gpu_reserved"] is False
    assert receipt["gpu_control"]["nvml_sampled"] is False
    assert receipt["claims"]["scientific_pass"] is False
    assert receipt["claims"]["promotion"] is False


def test_receipt_replay_is_stable_across_fresh_owned_scratch(tmp_path: Path) -> None:
    first = run_operational_rehearsal(ROOT, tmp_path / "one")
    second = run_operational_rehearsal(ROOT, tmp_path / "two")
    assert first == second


def test_owned_scratch_is_cleanup_safe_after_return() -> None:
    with tempfile.TemporaryDirectory(prefix="sigma-alpha-owned-") as directory:
        scratch = Path(directory) / "scratch"
        run_operational_rehearsal(ROOT, scratch)
        assert scratch.is_dir()


def test_no_secret_prompt_or_output_is_persisted(tmp_path: Path) -> None:
    scratch = tmp_path / "owned"
    run_operational_rehearsal(ROOT, scratch)
    persisted = b"".join(path.read_bytes() for path in scratch.rglob("*") if path.is_file())
    assert b"SIGMA_FORMULA_LLM_API_KEY" not in persisted
    assert b"Bounded synthetic formula proposal rehearsal" not in persisted
    assert b"formula-proposals" not in persisted


def test_runtime_and_nonempty_scratch_are_rejected(tmp_path: Path) -> None:
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "foreign.txt").write_text("foreign", encoding="utf-8")
    with pytest.raises(ValueError, match="must be empty"):
        run_operational_rehearsal(ROOT, nonempty)
    with pytest.raises(ValueError, match="may not use repository runtime"):
        run_operational_rehearsal(ROOT, ROOT / "runs/engine/forbidden-alpha-rehearsal")


def test_receipt_and_gpu_readiness_tamper_fail(tmp_path: Path) -> None:
    receipt = run_operational_rehearsal(ROOT, tmp_path / "owned")
    forged = json.loads(json.dumps(receipt))
    forged["claims"]["promotion"] = True
    with pytest.raises(ValueError, match="hash or schema"):
        validate_operational_receipt(forged)

    readiness = (
        ROOT / "runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-adapter-readiness.json"
    )
    copied = (
        tmp_path
        / "fake-repo/runs/engine/kastner-schlatter-set-indexed-gpu-scheduler-adapter-readiness.json"
    )
    copied.parent.mkdir(parents=True)
    copied.write_bytes(
        readiness.read_bytes().replace(b'"gpu_owner_count":1', b'"gpu_owner_count":2')
    )
    with pytest.raises(ValueError, match="content hash mismatch"):
        _gpu_control(tmp_path / "fake-repo")
