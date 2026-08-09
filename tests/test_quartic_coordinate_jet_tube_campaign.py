import json
from pathlib import Path

from sigma_theory_compiler.quartic_coordinate_jet_tube_campaign import (
    generic_coordinate_jet_majorant_control,
    run_quartic_coordinate_jet_tube_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_coordinate_jet_tube_campaign.json"
ARTIFACT_PATH = RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_coordinate_majorants_fit_strictly_inside_covariant_box() -> None:
    passed, evidence = generic_coordinate_jet_majorant_control()
    assert passed
    assert evidence["orthonormal_symmetric_metric_basis_residual"] == "0"
    assert evidence["bounded_coordinate_atoms"]["total"] == 153
    assert all(
        item["strict_margin_numeric"] > 0
        for item in evidence["covariant_hyperbolicity_components"].values()
    )
    assert (
        evidence["covariant_hyperbolicity_components"]["einstein_upper"][
            "upper_numeric"
        ]
        < 2e-10
    )
    assert evidence["negative_control"]["rejected"]


def test_coordinate_majorants_have_nonnegative_derivatives_through_order_four() -> None:
    passed, evidence = generic_coordinate_jet_majorant_control()
    assert passed
    derivative_data = evidence["Frechet_majorant_derivatives"]
    assert derivative_data["orders"] == [0, 1, 2, 3, 4]
    assert all(
        item[str(order)]["numeric"] >= 0
        for item in derivative_data["families"].values()
        for order in derivative_data["orders"]
    )


def test_all_candidates_receive_common_coordinate_jet_tube() -> None:
    result = run_quartic_coordinate_jet_tube_campaign(
        _load(PDE_PATH), _load(CONFIG_PATH)
    )
    assert (
        result["status"]
        == "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes"
    )
    assert result["counts"]["coordinate_jet_tubes_passed"] == 12
    assert len(result["certificates"]) == 12
    assert all(
        item["status"]
        == "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube"
        and item["bounded_coordinate_atom_count"] == 153
        and item["Frechet_majorant_order"] == 4
        and item["coordinate_component_radius"] == "1/10000000000000"
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_too_large_coordinate_radius_and_corrupt_prerequisite_reject() -> None:
    pde = _load(PDE_PATH)
    config = _load(CONFIG_PATH)
    too_large = dict(config)
    too_large["coordinate_component_radius"] = "1/1000000000000"
    result = run_quartic_coordinate_jet_tube_campaign(pde, too_large)
    assert result["status"] == "reject"
    assert "coordinate radius" in result["errors"][0]

    corrupted = dict(pde)
    corrupted["status"] = "reject"
    result = run_quartic_coordinate_jet_tube_campaign(corrupted, config)
    assert result["status"] == "reject"
    assert "prerequisite" in result["errors"][0]
