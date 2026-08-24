from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import creative_modular_gpu_prefilter as C
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> dict:
    return C.validate_config(
        {
            "batch_size": 128,
            "campaign_id": "invariant.creativity.modular-prefilter.test-v1",
            "coefficient_max": 2,
            "coefficient_min": -2,
            "cpu_benchmark_candidates": 125,
            "feature_matrix": [[1, 1, 1], [1, 2, 4], [1, 3, 9], [1, 4, 16]],
            "maximum_recorded_survivors": 8,
            "primes": [101, 103],
            "required_device_name": "NVIDIA GeForce RTX 5090",
            "sample_candidates": 125,
            "schema_version": C.CONFIG_SCHEMA,
            "target_coefficients": [1, -1, 2],
            "targets": [2, 7, 16, 29],
        }
    )


def test_ordinal_encoding_is_bijective_on_small_lattice() -> None:
    config = _small_config()
    vectors = [C.decode_ordinal(ordinal, config) for ordinal in range(C.search_space_size(config))]
    assert len(vectors) == len(set(vectors)) == 125
    assert [C.encode_coefficients(vector, config) for vector in vectors] == list(range(125))


def test_independent_cpu_screen_has_one_exact_survivor() -> None:
    config = _small_config()
    survivors = [
        ordinal
        for ordinal in range(C.search_space_size(config))
        if C.cpu_modular_survives(ordinal, config)
    ]
    target = C.encode_coefficients(config["target_coefficients"], config)
    assert survivors == [target]
    assert C.cpu_exact_match(target, config)
    mutated = list(config["targets"])
    mutated[0] += 1
    assert not C.cpu_modular_survives(target, config, targets=mutated)
    assert not C.cpu_exact_match(target, config, targets=mutated)


def test_gpu_small_lattice_matches_independent_cpu() -> None:
    try:
        result = C.run_gpu_screen(_small_config())
    except RuntimeError as error:
        pytest.skip(str(error))
    config = _small_config()
    target = C.encode_coefficients(config["target_coefficients"], config)
    assert result["candidate_count"] == 125
    assert result["survivor_ordinals"] == [target]
    assert result["sample_crosscheck"]["statuses_agree"]
    assert result["cpu_benchmark"]["status_sha256"] == result["gpu_benchmark"]["status_sha256"]
    assert result["mutation_control"]["target_candidate_rejected"]


def test_stored_5090_receipt_validates_without_requiring_cuda() -> None:
    receipt = json.loads((ROOT / C.OUTPUT_PATH).read_text(encoding="utf-8"))
    C.validate_receipt(receipt, ROOT)
    assert receipt["summary"]["candidates_classified"] == 33**5
    assert receipt["summary"]["gpu_modular_survivors"] == 1
    assert receipt["summary"]["exact_survivors"] == 1
    assert receipt["execution_boundary"]["paid_llm_calls_made"] == 0


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_resealed_receipt_cannot_promote_gpu_survival_to_proof() -> None:
    receipt = json.loads((ROOT / C.OUTPUT_PATH).read_text(encoding="utf-8"))
    changed = copy.deepcopy(receipt)
    changed["claims"]["modular_screen_is_a_formal_proof"] = True
    _reseal(changed)
    with pytest.raises(C.CreativeModularGPUError, match="claim boundary"):
        C.validate_receipt(changed, ROOT)


def test_resealed_receipt_cannot_hide_a_survivor_or_forge_cpu_agreement() -> None:
    receipt = json.loads((ROOT / C.OUTPUT_PATH).read_text(encoding="utf-8"))
    changed = copy.deepcopy(receipt)
    changed["gpu_execution"]["survivor_ordinals"] = []
    changed["summary"]["gpu_modular_survivors"] = 0
    _reseal(changed)
    with pytest.raises(C.CreativeModularGPUError, match="exact survivor replay"):
        C.validate_receipt(changed, ROOT)

    changed = copy.deepcopy(receipt)
    changed["gpu_execution"]["sample_crosscheck"]["cpu_status_sha256"] = "0" * 64
    changed["gpu_execution"]["sample_crosscheck"]["gpu_status_sha256"] = "0" * 64
    _reseal(changed)
    with pytest.raises(C.CreativeModularGPUError, match="independent modular sample"):
        C.validate_receipt(changed, ROOT)


def test_bad_prime_and_inconsistent_control_fail_closed() -> None:
    config = _small_config()
    config["primes"] = [101, 105]
    with pytest.raises(C.CreativeModularGPUError, match="primes"):
        C.validate_config(config)
    config = _small_config()
    config["targets"][0] += 1
    with pytest.raises(C.CreativeModularGPUError, match="do not generate"):
        C.validate_config(config)
