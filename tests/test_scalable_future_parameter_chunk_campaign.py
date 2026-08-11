from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.scalable_campaign_epoch_service import (
    reviewed_future_manifest_admission_adapter,
)
from sigma_theory_compiler.scalable_future_parameter_chunk_campaign import (
    ScalableFutureParameterCompilationService,
    build_future_parameter_manifest_chunk,
    compile_future_parameter_chunk,
    publish_future_parameter_manifest_chunk,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/scalable_future_parameter_chunk_campaign.json"


def _config(enabled: bool = False) -> dict:
    value = json.loads(CONFIG.read_text())
    value["execution_enabled"] = enabled
    return value


def test_exact_future_chunk_and_reviewed_compilation_counts() -> None:
    config = _config()
    chunk = build_future_parameter_manifest_chunk(config, ROOT)
    assert chunk["range"] == {"start": 256, "stop": 288}
    assert chunk["family_cell_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 16,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 4,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 4,
        "KESSENCE_G2_CONVEX": 8,
    }
    assert chunk["content_sha256"] == "c4addebc881d796a30e86aed126c42ecd0f27a9111cf36ab9f7bee1709435550"
    admission = reviewed_future_manifest_admission_adapter(chunk)
    assert admission["decision"] == "admit"
    assert admission["scientific_compilation_started"] is False

    result = compile_future_parameter_chunk(config, ROOT, chunk)
    assert result["disposition_counts"] == {
        "admitted_new_candidate": 19,
        "deduplicated_existing_candidate": 13,
    }
    assert result["content_sha256"] == "07bb2fa0756e4ee6943136ea29dc80e0b044ccfba28e3b8726e66a3cb6d571ab"
    assert result["receipt_registry_root_sha256"] == "3ec87607dd32a45ff711e5538306e01446b9768027d36270c26c6db64aefaede"
    assert all(receipt["data_eligibility"] == config["data_eligibility"] for receipt in result["receipts"])


def test_persistent_service_restart_replay_and_recovery(tmp_path: Path) -> None:
    config = _config(True)
    service = ScalableFutureParameterCompilationService(tmp_path / "service", config, ROOT)
    assert service.enqueue()["accepted"] == 4
    lease = service.coordinator.claim("cpu", "crashed", lease_seconds=1)
    assert lease is not None
    with service.coordinator.connect() as connection:
        connection.execute(
            "UPDATE work SET lease_expires_utc='2000-01-01T00:00:00+00:00' WHERE work_id=?",
            (lease.work_id,),
        )
    resumed = ScalableFutureParameterCompilationService(tmp_path / "service", config, ROOT)
    assert resumed.recovered_on_start == {"recovered": 1, "failed": 0}
    assert resumed.run_ready() == 4
    first = resumed.status()
    assert first["queue_counts"] == {"succeeded": 4}
    replay = ScalableFutureParameterCompilationService(tmp_path / "service", config, ROOT)
    assert replay.enqueue()["duplicate"] == 4
    assert replay.run_ready() == 0
    assert replay.status()["compilation_result_content_sha256"] == first["compilation_result_content_sha256"]


def test_disabled_tamper_and_forbidden_input_fail_closed(tmp_path: Path) -> None:
    config = _config()
    service = ScalableFutureParameterCompilationService(tmp_path / "disabled", config, ROOT)
    with pytest.raises(PermissionError, match="disabled"):
        service.enqueue()

    chunk = build_future_parameter_manifest_chunk(config, ROOT)
    chunk["parameter_cells"][0]["parameters"]["c1"] = "1/3"
    with pytest.raises(ValueError, match="validation failed"):
        compile_future_parameter_chunk(config, ROOT, chunk)

    forbidden = _config()
    forbidden["data_eligibility"]["redshift_distance_inputs"] = True
    with pytest.raises(ValueError, match="seals are open"):
        build_future_parameter_manifest_chunk(forbidden, ROOT)


def test_atomic_publication_is_idempotent_and_refuses_divergence(tmp_path: Path) -> None:
    target = tmp_path / "future.json"
    first = publish_future_parameter_manifest_chunk(_config(), ROOT, target)
    assert publish_future_parameter_manifest_chunk(_config(), ROOT, target) == first
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="divergent"):
        publish_future_parameter_manifest_chunk(_config(), ROOT, target)


def test_descriptor_is_exact_hash_bound_and_sealed() -> None:
    descriptor = json.loads(
        (ROOT / "configs/reviewed_future_parameter_compiler_adapter.json").read_text()
    )
    body = {key: value for key, value in descriptor.items() if key != "content_sha256"}
    from sigma_theory_compiler.scalable_future_parameter_chunk_campaign import _sha

    assert descriptor["content_sha256"] == _sha(body)
    assert descriptor["callback_source_file_sha256"] == hashlib.sha256(
        (ROOT / descriptor["callback_source_path"]).read_bytes()
    ).hexdigest()
    assert descriptor["data_eligibility"] == _config()["data_eligibility"]
    assert descriptor["external_paid_llm_calls"] is False
    artifact = json.loads(
        (ROOT / "runs/engine/scalable-future-parameter-chunk-001-status.json").read_text()
    )
    artifact_body = {
        key: value for key, value in artifact.items() if key != "content_sha256"
    }
    assert artifact["content_sha256"] == _sha(artifact_body)
