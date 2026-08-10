import json
from pathlib import Path

from sigma_theory_compiler.quartic_positive_quantization_campaign import (
    generic_gaussian_anti_wick_control,
    run_quartic_positive_quantization_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
LOW_PATH = RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_positive_quantization_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-positive-quantization-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gaussian_window_and_resolution_identity_are_exact() -> None:
    passed, evidence = generic_gaussian_anti_wick_control()
    assert passed
    assert evidence["window_norm_squared"] == "1"
    assert evidence["resolution_of_identity_coefficient"] == "1"
    assert evidence["matrix_energy_control"]["lower_residual"] == (
        "b**2*(Lambda - lambda)"
    )
    assert evidence["matrix_energy_control"]["upper_residual"] == (
        "a**2*(Lambda - lambda)"
    )
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_positive_uniform_quantizations() -> None:
    result = run_quartic_positive_quantization_campaign(
        _load(LOW_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == "pass_all_12_uniform_positive_anti_wick_K55_operators"
    assert result["counts"] == {
        "selected": 12,
        "positive_quantizations_passed": 12,
        "rejected": 0,
    }
    assert all(
        item["operator_energy_equivalence"]["lower_numeric"] > 4e-26
        and item["operator_energy_equivalence"]["uniform_in_h"]
        and item["operator_energy_equivalence"][
            "exactly_matches_pointwise_symbol_bounds"
        ]
        and item["quantization"]["self_adjoint"]
        and item["quantization"]["positive"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_wrong_dimension_window_scale_and_provenance_reject() -> None:
    low = _load(LOW_PATH)
    config = _load(CONFIG_PATH)
    wrong_dimension = dict(config)
    wrong_dimension["state_dimension"] = 22
    result = run_quartic_positive_quantization_campaign(low, wrong_dimension)
    assert result["status"] == "reject"
    assert "55-state" in result["errors"][0]

    wrong_window = dict(config)
    wrong_window["coherent_window"] = "box"
    result = run_quartic_positive_quantization_campaign(low, wrong_window)
    assert result["status"] == "reject"
    assert "window" in result["errors"][0]

    wrong_scale = dict(config)
    wrong_scale["semiclassical_scale_domain"] = "h=1"
    result = run_quartic_positive_quantization_campaign(low, wrong_scale)
    assert result["status"] == "reject"
    assert "scale domain" in result["errors"][0]

    corrupt = json.loads(json.dumps(low))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_positive_quantization_campaign(corrupt, config)
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
