from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.continuous_scientific_pipeline_formal_receipt_batch_0003_blocked_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / readiness.CONFIG_REL
ARTIFACT = ROOT / readiness.READINESS_REL


def _checked() -> dict[str, object]:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def _raw_samples(value: dict[str, object]) -> list[dict[str, object]]:
    return [
        {key: item for key, item in row.items() if key != "sample_admitted"}
        for row in value["resource_samples"]
    ]


def test_checked_readiness_is_replayable_blocked_and_not_an_execution_result() -> None:
    value = _checked()
    readiness.validate_readiness(value, ROOT, CONFIG)
    assert readiness.replay_readiness(value, ROOT, CONFIG) == value
    assert value["decision"] == readiness.DECISION
    assert value["admitted"] is False
    assert value["execution_started"] is False
    assert value["result_artifact_created"] is False
    assert value["leaf_artifacts_created"] is False
    assert value["cursor_artifact_created"] is False
    assert value["complete_global_formal_receipts"] is False
    assert value["complete_comparable_evidence"] is False
    assert not any(value["promotion_contract"].values())
    assert not any(value["seals"].values())


def test_exact_resource_samples_reconcile_with_strict_thresholds() -> None:
    value = _checked()
    contract = value["resource_sampling_contract"]
    samples = value["resource_samples"]
    assert len(samples) == contract["sample_count"] == 3
    assert contract["cpu_utilization_strictly_below_percent"] == 92
    assert contract["minimum_available_ram_mib"] == 32_768
    assert all(sample["cpu_utilization_percent"] >= 92 for sample in samples)
    assert all(sample["available_ram_mib"] >= 32_768 for sample in samples)
    assert all(sample["sample_admitted"] is False for sample in samples)
    assert value["resource_summary"] == {
        "maximum_cpu_utilization_percent": max(
            sample["cpu_utilization_percent"] for sample in samples
        ),
        "minimum_available_ram_mib": min(sample["available_ram_mib"] for sample in samples),
        "admitted_sample_count": 0,
        "non_admitted_sample_count": 3,
        "all_resource_samples_admissible": False,
        "first_blocker": "cpu_utilization_not_strictly_below_92_percent",
    }


def test_readiness_binds_untouched_batch_implementation_and_predecessor() -> None:
    value = _checked()
    config = readiness.load_config(ROOT, CONFIG)
    assert value["target_batch_bindings"] == {
        "config": config["target_batch_config"],
        "source": config["target_batch_source"],
        "test": config["target_batch_test"],
    }
    assert value["predecessor_result_binding"] == config["predecessor_result"]
    assert value["target_batch_bindings"]["config"]["file_sha256"] == (
        "20563808fddda8e12cd67594fd8c784bcb5f564fa80a2230d96591029453623f"
    )
    assert value["target_batch_bindings"]["source"]["file_sha256"] == (
        "33c09ef3248d0acd1f74bb91a3d25f5f4c921ce90db179f3a4fed4a1ecf011e8"
    )
    assert value["target_batch_bindings"]["test"]["file_sha256"] == (
        "dabbfeb8dc81203f9b2a811e9687fdbd2b0cda2a12d93c7cffa589d27e9a460f"
    )


def test_execution_namespace_absence_is_historical_and_paths_are_exact() -> None:
    value = _checked()
    absence = value["execution_artifact_absence"]
    assert absence == {
        "artifact_directory": (
            "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003"
        ),
        "artifact_directory_existed": False,
        "result_path": (
            "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/"
            "result.json"
        ),
        "result_existed": False,
        "cursor_path": (
            "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/"
            "cumulative-cursor.json"
        ),
        "cursor_existed": False,
        "selected_leaf_paths": [
            (
                "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-"
                "0003/leaf-000012.json"
            ),
            (
                "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-"
                "0003/leaf-000013.json"
            ),
        ],
        "selected_leaf_artifacts_existed": False,
        "candidate_artifacts_path": (
            "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/"
            "candidate-artifacts"
        ),
        "candidate_artifacts_existed": False,
    }
    assert not (ROOT / absence["artifact_directory"]).exists()


def test_boundary_cpu_and_any_admitted_sample_cannot_form_blocked_readiness() -> None:
    checked = _checked()
    samples = _raw_samples(checked)
    samples[0]["cpu_utilization_percent"] = 92
    normalized = readiness._normalize_samples(samples, checked["resource_sampling_contract"])
    assert normalized[0]["sample_admitted"] is False
    samples[0]["cpu_utilization_percent"] = 91.999
    with pytest.raises(ValueError, match="every recorded sample"):
        readiness._normalize_samples(samples, checked["resource_sampling_contract"])


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("admitted", True),
        lambda value: value.__setitem__("execution_started", True),
        lambda value: value.__setitem__("result_artifact_created", True),
        lambda value: value["resource_sampling_contract"].__setitem__(
            "cpu_utilization_strictly_below_percent", 101
        ),
        lambda value: value["predecessor_result_binding"].__setitem__("content_sha256", "0" * 64),
        lambda value: value["promotion_contract"].__setitem__(
            "candidate_promotion_performed", True
        ),
    ],
)
def test_resealed_tamper_fails_closed(mutation: object) -> None:
    body = copy.deepcopy(_checked())
    body.pop("content_sha256")
    mutation(body)  # type: ignore[operator]
    with pytest.raises(ValueError):
        readiness.validate_readiness(readiness._sealed(body), ROOT, CONFIG)


def test_sample_shape_time_order_and_exact_types_fail_closed() -> None:
    checked = _checked()
    samples = _raw_samples(checked)
    samples[1]["sampled_at"] = samples[0]["sampled_at"]
    with pytest.raises(ValueError, match="strictly increasing"):
        readiness._normalize_samples(samples, checked["resource_sampling_contract"])
    samples = _raw_samples(checked)
    samples[0]["cpu_utilization_percent"] = True
    with pytest.raises(ValueError, match="value contract"):
        readiness._normalize_samples(samples, checked["resource_sampling_contract"])
    samples = _raw_samples(checked)[:-1]
    with pytest.raises(ValueError, match="count mismatch"):
        readiness._normalize_samples(samples, checked["resource_sampling_contract"])


def test_validation_never_opens_runtime_sqlite_gpu_or_supervisor_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _checked()
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def excluded(path: Path) -> bool:
        normalized = path.as_posix().lower()
        return any(
            token in normalized
            for token in (
                "campaign-v1-live.sqlite",
                "service-runtime",
                "gpu-scheduler-runtime",
                "parallel-supervisor",
                ".lease",
            )
        )

    def guarded_text(path: Path, *args: object, **kwargs: object) -> str:
        if excluded(path):
            raise AssertionError("readiness validator opened excluded mutable state")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if excluded(path):
            raise AssertionError("readiness validator opened excluded mutable state")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    readiness.validate_readiness(value, ROOT, CONFIG)


def test_config_and_artifact_paths_are_separate_from_execution_namespace() -> None:
    config = readiness.load_config(ROOT, CONFIG)
    assert config["readiness_artifact"] == readiness.READINESS_REL
    assert "blocked-readiness.json" in readiness.READINESS_REL
    assert readiness.READINESS_REL != (
        "runs/engine/continuous-scientific-pipeline-epoch-003-formal-receipt-batch-0003/result.json"
    )
    assert not any(config["seals"].values())
