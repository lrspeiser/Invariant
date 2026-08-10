import json
from pathlib import Path

from sigma_theory_compiler.quartic_solved_source_moser_campaign import (
    generic_quadratic_composition_control,
    run_quartic_solved_source_moser_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
MOSER_PATH = RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
TUBE_PATH = RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"
EULER_PATH = RUNS / "quartic-euler-remainder-majorant-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_solved_source_moser_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict]:
    return (
        _load(MOSER_PATH),
        _load(PDE_PATH),
        _load(TUBE_PATH),
        _load(EULER_PATH),
        _load(CONFIG_PATH),
    )


def test_quadratic_composition_and_inverse_product_multiplicities_are_exact() -> None:
    passed, evidence = generic_quadratic_composition_control()
    assert passed
    assert set(evidence["quadratic_composition_residuals"].values()) == {"0"}
    assert set(evidence["inverse_product_residuals"].values()) == {"0"}
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_coordinate_atom_c4_solved_source_bounds() -> None:
    result = run_quartic_solved_source_moser_campaign(*_inputs())
    assert (
        result["status"]
        == "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes"
    )
    assert result["counts"] == {
        "selected": 12,
        "solved_source_moser_envelopes_passed": 12,
        "rejected": 0,
    }
    assert all(
        item["dominant_family"] == "einstein_upper_component"
        for item in result["coordinate_jet_Frechet_envelopes"]["envelopes"].values()
    )
    assert all(
        set(item["solved_source_Frechet_derivatives"]["2_norm_envelopes_numeric"])
        == {"0", "1", "2", "3", "4"}
        and all(
            value > 0
            for value in item["solved_source_Frechet_derivatives"][
                "2_norm_envelopes_numeric"
            ].values()
        )
        and item["order_zero_acceleration_crosscheck"]["relative_residual"] < 1e-12
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_fourth_order_bound_exposes_conservative_source_growth() -> None:
    result = run_quartic_solved_source_moser_campaign(*_inputs())
    fourth_order = [
        item["solved_source_Frechet_derivatives"]["2_norm_envelopes_numeric"]["4"]
        for item in result["certificates"]
    ]
    assert min(fourth_order) > 1e19
    assert max(fourth_order) < 2e20


def test_insufficient_order_and_corrupt_moser_provenance_reject() -> None:
    moser, pde, tube, euler, config = _inputs()
    insufficient = dict(config)
    insufficient["required_Frechet_majorant_order"] = 3
    result = run_quartic_solved_source_moser_campaign(
        moser, pde, tube, euler, insufficient
    )
    assert result["status"] == "reject"
    assert "order four" in result["errors"][0]

    corrupted_pde = json.loads(json.dumps(pde))
    corrupted_pde["upstream_sha256"]["moser"] = "corrupt"
    result = run_quartic_solved_source_moser_campaign(
        moser, corrupted_pde, tube, euler, config
    )
    assert result["status"] == "reject"
    assert "Moser provenance" in result["errors"][0]
