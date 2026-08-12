from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.continuous_scientific_pipeline_epoch_result import (
    PREFLIGHT_REL,
    RESULT_REL,
    _sealed,
    build_epoch_result,
    validate_epoch_result,
)

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / "runs/engine/continuous-scientific-pipeline-service-runtime-v2/epoch-003"


def test_terminal_epoch_result_is_exact_complete_and_fail_closed() -> None:
    value = build_epoch_result(ROOT)
    validate_epoch_result(value, ROOT)
    assert value["decision"] == "bounded_epoch_complete_fail_closed_no_promotion"
    assert value["coverage"]["start_ordinal"] == 1_004_013_056
    assert value["coverage"]["stop_ordinal_exclusive"] == 1_007_945_216
    assert value["coverage"]["unique_formula_count"] == 3_932_160
    assert value["coverage"]["real_CPU_batches"] == 8
    assert len(value["coverage"]["intervals"]) == 8
    assert value["coverage"]["intervals"][0]["start_ordinal"] == 1_004_013_056
    assert value["coverage"]["intervals"][-1]["stop_ordinal_exclusive"] == 1_007_945_216
    assert value["runtime_binding"]["terminal_state"] == "bounded_complete"
    assert value["runtime_binding"]["cycles"] == 23
    assert value["outcomes"] == {
        "sampled_static_reject_batches": 2,
        "sampled_static_pass_batches": 6,
        "formal_receipts": 6,
        "formal_blocks": 6,
        "formal_passes": 0,
        "leaderboard_rebuild_requests": 0,
        "rank_assignments": 0,
    }
    assert not any(value["promotion_contract"].values())
    assert not any(value["seals"].values())


def test_checked_result_matches_runtime_construction() -> None:
    checked = json.loads((ROOT / RESULT_REL).read_text(encoding="utf-8"))
    assert checked == build_epoch_result(ROOT)
    validate_epoch_result(checked, ROOT)


def test_validator_never_reads_mutable_epoch_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    value = json.loads((ROOT / RESULT_REL).read_text(encoding="utf-8"))
    original_text = Path.read_text
    original_bytes = Path.read_bytes

    def is_epoch_runtime(path: Path) -> bool:
        return (
            "continuous-scientific-pipeline-service-runtime-v2" in path.parts
            and "epoch-003" in path.parts
        )

    def guarded_text(path: Path, *args: object, **kwargs: object) -> str:
        if is_epoch_runtime(path):
            raise AssertionError("validator opened mutable epoch runtime")
        return original_text(path, *args, **kwargs)

    def guarded_bytes(path: Path, *args: object, **kwargs: object) -> bytes:
        if is_epoch_runtime(path):
            raise AssertionError("validator opened mutable epoch runtime")
        return original_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_bytes)
    validate_epoch_result(value, ROOT)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["coverage"]["intervals"][2].__setitem__(
            "start_ordinal", value["coverage"]["intervals"][2]["start_ordinal"] + 1
        ),
        lambda value: value["completed_receipt_bindings"][0]["generated_receipt"].__setitem__(
            "candidate_root_sha256", "0" * 64
        ),
        lambda value: value["terminal_runtime_archive"]["queue"].__setitem__(
            "next_ordinal", value["terminal_runtime_archive"]["queue"]["next_ordinal"] - 1
        ),
        lambda value: value["replay_dependencies"].__setitem__(
            "replay_dependency_root_sha256", "0" * 64
        ),
        lambda value: value["preflight_resource_evidence"]["summary"].__setitem__(
            "maximum_cpu_percent", 92.0
        ),
    ],
)
def test_terminal_archive_tampering_fails_closed(mutation: object) -> None:
    value = build_epoch_result(ROOT)
    tampered = copy.deepcopy(value)
    tampered.pop("content_sha256")
    mutation(tampered)  # type: ignore[operator]
    with pytest.raises(ValueError):
        validate_epoch_result(_sealed(tampered), ROOT)


def test_preflight_is_historical_and_resource_admissible() -> None:
    value = json.loads((ROOT / PREFLIGHT_REL).read_text(encoding="utf-8"))
    assert value["sampling_contract"]["historical_measurement_not_current_runtime_claim"] is True
    assert value["summary"] == {
        "maximum_cpu_percent": 74.1,
        "minimum_available_ram_mib": 49_168,
        "all_resource_samples_admissible": True,
    }
    assert not any(value["historical_path_state"].values())
