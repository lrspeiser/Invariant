from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_cuda_falsification_design import (
    deterministic_inputs,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_cuda_falsification_design.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-cuda-falsification-design.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(document: dict) -> None:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_artifact_validates_and_inputs_are_deterministic() -> None:
    config = _load(CONFIG)
    first = deterministic_inputs(config)
    second = deterministic_inputs(config)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].tobytes() == second[key].tobytes()
    validate_campaign(_load(ARTIFACT), CONFIG)


def test_power_controls_and_gpu_cpu_crosscheck() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["poisson_power_control"]["empirical_null_rejection_rate"] <= 0.01
    assert artifact["poisson_power_control"]["empirical_alternative_detection_rate"] >= 0.99
    assert artifact["btfr_power_control"]["empirical_null_rejection_rate"] <= 0.002
    assert artifact["btfr_power_control"]["empirical_alternative_detection_rate"] >= 0.99
    assert artifact["gpu_cpu_crosscheck"]["maximum_absolute_statistic_error"] <= 1e-12
    assert artifact["gpu_cpu_crosscheck"]["all_rejection_decisions_byte_equal"] is True


def test_evidence_classes_and_observational_seals() -> None:
    artifact = _load(ARTIFACT)
    assert artifact["evidence_classification"] == {
        "gpu_execution": "implementation_stress_test",
        "synthetic_power": "falsification_design_control",
        "scientific_test": "not_performed",
        "observational_test": "not_performed",
    }
    assert artifact["observational_bridge"]["registration_fields_advanced"] == 0
    assert artifact["observational_bridge"]["real_bundle_fields_filled"] == 0
    assert artifact["btfr_power_control"]["extended_galaxy_geometry_tested"] is False
    assert artifact["synthetic_only"] is True
    for key in (
        "observations_opened",
        "scientific_test_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        assert artifact[key] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.__setitem__("scientific_test_pass", True), "claim or data seal"),
        (
            lambda value: value["observational_bridge"].__setitem__(
                "registration_fields_advanced", 1
            ),
            "observational bridge overclaimed",
        ),
        (
            lambda value: value["deterministic_manifest"]["input_sha256"].__setitem__(
                "poisson_null", "0" * 64
            ),
            "deterministic input manifest",
        ),
    ],
)
def test_rehashed_claim_manifest_and_readiness_tampering_fails(mutation, message: str) -> None:
    artifact = copy.deepcopy(_load(ARTIFACT))
    mutation(artifact)
    _rehash(artifact)
    with pytest.raises(ValueError, match=message):
        validate_campaign(artifact, CONFIG)


def test_no_host_paths_or_secret_markers() -> None:
    text = ARTIFACT.read_text(encoding="utf-8").lower()
    assert "c:" + "\\users\\" not in text
    assert "/" + "home/" not in text
    for marker in ("api" + "_key", "bear" + "er ", "s" + "k-"):
        assert marker not in text
