import json
from pathlib import Path

from sigma_theory_compiler.quartic_euler_remainder_majorant_campaign import (
    generic_euler_remainder_majorant_control,
    run_quartic_euler_remainder_majorant_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
TUBE_PATH = RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_euler_remainder_majorant_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-euler-remainder-majorant-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict]:
    return _load(PDE_PATH), _load(TUBE_PATH), _load(CONFIG_PATH)


def test_generic_term_inventory_and_auxiliary_metric_contractions() -> None:
    passed, evidence = generic_euler_remainder_majorant_control()
    assert passed
    assert evidence["term_counts"] == {
        "quartic_metric_lower": 8,
        "G2_metric_lower": 2,
        "scalar_euler": 3,
        "modified_harmonic_gauge": 4,
    }
    contractions = evidence["auxiliary_metric_contractions"]
    assert contractions["tilde_component_l1"] == 7
    assert contractions["hat_projector_symmetric_row_l1_max"] == "18"
    assert contractions["hat_projector_symmetric_row_l1"]["00"] == "18"


def test_all_candidates_receive_complete_remainder_majorants() -> None:
    result = run_quartic_euler_remainder_majorant_campaign(*_inputs())
    assert (
        result["status"]
        == "pass_all_12_complete_coordinate_tube_euler_remainder_majorants"
    )
    assert result["counts"]["Euler_remainder_majorants_passed"] == 12
    assert len(result["certificates"]) == 12
    assert all(
        item["status"] == "pass_complete_coordinate_tube_euler_remainder_majorant"
        and item["Euler_remainder_component_upper_numeric"] > 0
        and item["solved_acceleration_component_upper_numeric"] > 0
        and set(item["Euler_remainder_Frechet_derivative_uppers"])
        == {"0", "1", "2", "3", "4"}
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_remainder_derivatives_are_nonnegative_through_order_four() -> None:
    result = run_quartic_euler_remainder_majorant_campaign(*_inputs())
    assert all(
        derivative["numeric"] >= 0
        for item in result["certificates"]
        for derivative in item["Euler_remainder_Frechet_derivative_uppers"].values()
    )


def test_corrupt_term_count_and_provenance_reject() -> None:
    pde, tube, config = _inputs()
    corrupted_config = dict(config)
    corrupted_config["required_scalar_term_count"] = 2
    result = run_quartic_euler_remainder_majorant_campaign(
        pde, tube, corrupted_config
    )
    assert result["status"] == "reject"
    assert "term inventory" in result["errors"][0]

    corrupted_tube = dict(tube)
    corrupted_tube["nonquasilinear_pde_campaign_sha256"] = "corrupt"
    result = run_quartic_euler_remainder_majorant_campaign(
        pde, corrupted_tube, config
    )
    assert result["status"] == "reject"
    assert "provenance" in result["errors"][0]
