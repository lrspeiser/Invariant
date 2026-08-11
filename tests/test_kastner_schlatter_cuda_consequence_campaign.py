from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_cuda_consequence_campaign import (
    deterministic_inputs,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_cuda_consequence_campaign.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-cuda-consequence-campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(document: dict) -> None:
    body = {key: value for key, value in document.items() if key != "content_sha256"}
    document["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    ).hexdigest()


def test_deterministic_manifest_and_artifact_validate() -> None:
    config = _load(CONFIG)
    first = deterministic_inputs(config)
    second = deterministic_inputs(config)
    assert first.keys() == second.keys()
    expected = {}
    for key in first:
        assert first[key].tobytes() == second[key].tobytes()
        expected[key] = hashlib.sha256(first[key].tobytes()).hexdigest()
    artifact = _load(ARTIFACT)
    assert artifact["deterministic_manifest"]["input_sha256"] == expected
    validate_campaign(artifact, CONFIG)


def test_cuda_controls_counts_and_claim_seals() -> None:
    artifact = _load(ARTIFACT)
    counts = artifact["counts"]
    assert counts["poisson_samples"] == 3 * 2 * 262144
    assert counts["sds_cases"] == 262144
    assert counts["mond_cases"] == 262144
    assert artifact["gpu_cpu_bindings"]["poisson_output_byte_equal"] is True
    assert artifact["poisson_four_volume_control"]["all_statistical_controls_closed"] is True
    assert artifact["sds_root_domain_control"]["all_static_patch_domains_valid"] is True
    assert artifact["mond_btfr_control"]["galaxy_geometry_or_data_tested"] is False
    assert artifact["synthetic_only"] is True
    for key in (
        "observations_opened",
        "ontology_pass",
        "theory_pass",
        "formal_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        assert artifact[key] is False


def test_equation_35_ambiguity_fails_closed() -> None:
    artifact = _load(ARTIFACT)
    gate = artifact["equation_35_normalization_gate"]
    assert gate["decision"] == "blocked"
    assert gate["middle_expression_coefficient_in_lP2_q_units"] == "8*pi^2"
    assert gate["printed_final_coefficient_in_lP2_q_units"] == "4*pi^2"
    assert gate["exact_ratio_middle_to_printed"] == "2"
    assert gate["lambda_values_emitted"] == 0
    assert gate["cosmology_locked_a0_values_emitted"] == 0


def test_tamper_and_host_secret_controls() -> None:
    artifact = _load(ARTIFACT)
    tampered = copy.deepcopy(artifact)
    tampered["theory_pass"] = True
    _rehash(tampered)
    with pytest.raises(ValueError, match="claim seal"):
        validate_campaign(tampered, CONFIG)
    text = ARTIFACT.read_text(encoding="utf-8").lower()
    assert "c:" + "\\users\\" not in text
    assert "/" + "home/" not in text
    for marker in ("api" + "_key", "author" + "ization", "bear" + "er ", "s" + "k-"):
        assert marker not in text


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value["counts"].__setitem__("poisson_samples", 1),
            "counters",
        ),
        (
            lambda value: value["gpu_cpu_bindings"].__setitem__(
                "sds_gpu_cpu_max_root_error", 1.0
            ),
            "Poisson CUDA output",
        ),
        (
            lambda value: value["poisson_four_volume_control"].__setitem__(
                "cdf_error_bound", 1.0
            ),
            "Poisson statistical",
        ),
        (
            lambda value: value["sds_root_domain_control"].__setitem__("case_count", 1),
            "SdS consequence",
        ),
        (
            lambda value: value["mond_btfr_control"].__setitem__(
                "galaxy_geometry_or_data_tested", True
            ),
            "MOND consequence",
        ),
        (
            lambda value: value["runtime_measurement"]["utilization"].__setitem__(
                "counter_scope", "lane-only"
            ),
            "runtime or device-wide",
        ),
        (
            lambda value: value["runtime_measurement"].__setitem__(
                "gpu_consequence_evaluations_per_second", 1.0
            ),
            "runtime or device-wide",
        ),
        (
            lambda value: value["source_bindings"]["source"].__setitem__(
                "path", "tests/test_kastner_schlatter_cuda_consequence_campaign.py"
            ),
            "path binding",
        ),
    ],
)
def test_rehashed_scientific_and_runtime_tampering_fails_closed(mutate, message: str) -> None:
    tampered = copy.deepcopy(_load(ARTIFACT))
    mutate(tampered)
    _rehash(tampered)
    with pytest.raises(ValueError, match=message):
        validate_campaign(tampered, CONFIG)
