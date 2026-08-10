import json
from pathlib import Path

from sigma_theory_compiler.quartic_full_symmetrizer_moser_campaign import (
    generic_symmetrizer_derivative_control,
    run_quartic_full_symmetrizer_moser_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SYMMETRIZER_PATH = RUNS / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
MOSER_PATH = RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
TUBE_PATH = RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"
SOURCE_PATH = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_full_symmetrizer_moser_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    return (
        _load(SYMMETRIZER_PATH),
        _load(MOSER_PATH),
        _load(PDE_PATH),
        _load(TUBE_PATH),
        _load(SOURCE_PATH),
        _load(CONFIG_PATH),
    )


def test_resolvent_product_and_chain_rule_multiplicities_are_exact() -> None:
    passed, evidence = generic_symmetrizer_derivative_control()
    assert passed
    assert set(evidence["resolvent_residuals"].values()) == {"0"}
    assert set(evidence["triple_product_residuals"].values()) == {"0"}
    assert set(evidence["chain_rule_residuals"].values()) == {"0"}
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_full_k55_c4_derivative_envelopes() -> None:
    result = run_quartic_full_symmetrizer_moser_campaign(*_inputs())
    assert (
        result["status"]
        == "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes"
    )
    assert result["counts"] == {
        "selected": 12,
        "full_K55_C4_derivative_envelopes_passed": 12,
        "rejected": 0,
    }
    assert all(
        set(item["K55_coordinate_atom_Frechet_derivative_2_norm_envelopes_numeric"])
        == {"0", "1", "2", "3", "4"}
        and all(
            value > 0
            for value in item[
                "K55_coordinate_atom_Frechet_derivative_2_norm_envelopes_numeric"
            ].values()
        )
        and item["physical_H_star_Frechet_derivative_2_norm_envelopes_numeric"]["3"]
        == 0
        and item["physical_H_star_Frechet_derivative_2_norm_envelopes_numeric"]["4"]
        == 0
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_conservative_fourth_order_symmetrizer_growth_is_explicit() -> None:
    result = run_quartic_full_symmetrizer_moser_campaign(*_inputs())
    fourth_order = [
        item["K55_coordinate_atom_Frechet_derivative_2_norm_envelopes_numeric"]["4"]
        for item in result["certificates"]
    ]
    assert min(fourth_order) > 1e54
    assert max(fourth_order) < 5e55
    assert all(
        item["Leibniz_commutator_coefficient_multipliers_numeric"]["4"]
        >= item["K55_coordinate_atom_Frechet_derivative_2_norm_envelopes_numeric"]["4"]
        for item in result["certificates"]
    )


def test_insufficient_order_and_corrupt_pde_provenance_reject() -> None:
    symmetrizer, moser, pde, tube, source, config = _inputs()
    insufficient = dict(config)
    insufficient["required_Frechet_majorant_order"] = 3
    result = run_quartic_full_symmetrizer_moser_campaign(
        symmetrizer, moser, pde, tube, source, insufficient
    )
    assert result["status"] == "reject"
    assert "order four" in result["errors"][0]

    corrupted_pde = json.loads(json.dumps(pde))
    corrupted_pde["upstream_sha256"]["symmetrizer"] = "corrupt"
    result = run_quartic_full_symmetrizer_moser_campaign(
        symmetrizer, moser, corrupted_pde, tube, source, config
    )
    assert result["status"] == "reject"
    assert "PDE provenance" in result["errors"][0]
