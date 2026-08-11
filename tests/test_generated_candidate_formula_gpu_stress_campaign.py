from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.generated_candidate_formula_gpu_stress_campaign import (
    deterministic_integer_inputs,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/generated_candidate_formula_gpu_stress_campaign.json"
ARTIFACT = ROOT / "runs/engine/generated-candidate-formula-gpu-stress-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_deterministic_manifest_and_integrated_artifact_validate() -> None:
    config = _load(CONFIG)
    first = deterministic_integer_inputs(config)
    second = deterministic_integer_inputs(config)
    assert first.tobytes() == second.tobytes()
    artifact = _load(ARTIFACT)
    assert artifact["deterministic_manifest"]["input_integer_tensor_sha256"] == hashlib.sha256(
        first.tobytes()
    ).hexdigest()
    validate_campaign(artifact, CONFIG)


def test_counts_error_bounds_and_fail_closed_claims_are_explicit() -> None:
    artifact = _load(ARTIFACT)
    counts = artifact["counts"]
    assert counts["candidate_count"] == 163
    assert counts["gpu_measured_candidate_formula_evaluations"] == (
        163 * counts["synthetic_points_per_candidate"] * counts["gpu_measured_repetitions"]
    )
    assert artifact["exact_cpu_control"]["crosscheck_count"] == 163 * 32
    assert artifact["gpu_cpu_comparison"]["within_bounds"] is True
    assert artifact["gpu_cpu_comparison"]["violating_point_count"] == 0
    assert artifact["synthetic_only"] is True
    for key in (
        "observations_opened",
        "dark_matter_or_halo_inputs",
        "redshift_distance_inputs",
        "paid_llm_calls",
        "formal_pass_inferred",
        "candidate_backend_metric_variation_executed",
        "field_equations_proven",
        "candidate_rejection_authorized",
        "scientific_ranking_authorized",
    ):
        assert artifact[key] is False


def test_content_manifest_and_seal_tampering_fail_closed() -> None:
    artifact = _load(ARTIFACT)
    tampered = copy.deepcopy(artifact)
    tampered["deterministic_manifest"]["seed"] += 1
    with pytest.raises(ValueError, match="content hash"):
        validate_campaign(tampered, CONFIG)
    tampered = copy.deepcopy(artifact)
    tampered["formal_pass_inferred"] = True
    body = {key: value for key, value in tampered.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    tampered["content_sha256"] = hashlib.sha256(canonical.encode("ascii")).hexdigest()
    with pytest.raises(ValueError, match="claim seal"):
        validate_campaign(tampered, CONFIG)


def test_host_paths_and_secret_markers_are_absent() -> None:
    text = ARTIFACT.read_text(encoding="utf-8")
    lowered = text.lower()
    assert "c:" + "\\users\\" not in lowered
    assert "/" + "home/" not in lowered
    for marker in ("api" + "_key", "author" + "ization", "bear" + "er ", "s" + "k-"):
        assert marker not in lowered
