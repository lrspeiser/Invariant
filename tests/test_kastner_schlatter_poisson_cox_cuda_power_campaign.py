from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_poisson_cox_cuda_power_campaign import (
    _content_sha,
    _load,
    analytic_moments,
    exact_witness_certificate,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_poisson_cox_cuda_power_campaign.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-poisson-cox-cuda-power-campaign.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_exact_registered_witness_and_general_moments() -> None:
    witness = exact_witness_certificate()
    assert witness["conditional_rate_support"] == ["1", "3"]
    assert witness["mean_count"] == "2"
    assert witness["count_variance"] == "3"
    assert witness["fano_factor"] == "3/2"
    assert witness["factorial_excess"] == "1"
    general = analytic_moments(Fraction(2), Fraction(1, 2))
    assert general["conditional_rates"] == ["1", "3"]
    assert general["fano_factor"] == "3/2"
    assert general["factorial_excess"] == "1"


def test_artifact_validates_and_covers_full_grid() -> None:
    result = _artifact()
    validate_campaign(result, CONFIG)
    assert result["counts"]["scenario_cells"] == 144
    assert result["counts"]["registered_witness_scenario_cells"] == 12
    assert result["counts"]["gpu_generated_count_values"] == 110100480
    assert result["counts"]["metric_replicate_values_cpu_gpu_checked"] == 1769472
    assert result["counts"]["null_calibration_replicates"] == 49152
    assert result["counts"]["finite_sample_evaluation_replicate_tests"] == 294912
    assert result["gpu_cpu_crosscheck"]["all_rejection_decisions_byte_equal"] is True


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("observations_opened", True),
        ("scientific_test_pass", True),
        ("paper_pass", True),
        ("dark_matter_or_halo_inputs", True),
        ("redshift_or_cosmology_inputs", True),
    ],
)
def test_claim_and_data_seal_tampering_fails(key: str, value: bool) -> None:
    tampered = copy.deepcopy(_artifact())
    tampered[key] = value
    tampered["content_sha256"] = _content_sha(tampered)
    with pytest.raises(ValueError, match="claim or data seal"):
        validate_campaign(tampered, CONFIG)


def test_content_and_predecessor_tampering_fail(tmp_path: Path) -> None:
    tampered = copy.deepcopy(_artifact())
    tampered["counts"]["scenario_cells"] = 1
    with pytest.raises(ValueError, match="content hash"):
        validate_campaign(tampered, CONFIG)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["predecessor"]["file_sha256"] = "0" * 64
    bad_config = tmp_path / "configs" / CONFIG.name
    bad_config.parent.mkdir()
    bad_config.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        _load(bad_config)


def test_no_observational_or_scientific_advance() -> None:
    result = _artifact()
    assert result["synthetic_only"] is True
    assert result["observations_opened"] is False
    assert result["readiness_advanced"] is False
    assert result["paper_pass"] is False
    assert result["qed_pass"] is False
    assert result["theory_pass"] is False
    assert result["ontology_pass"] is False
    assert result["counts"]["observational_records_accessed"] == 0
    assert result["counts"]["readiness_fields_advanced"] == 0
