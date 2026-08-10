import json
from pathlib import Path

from sigma_theory_compiler.quartic_r3_sobolev_calculus_campaign import (
    generic_r3_sobolev_chain_control,
    run_quartic_r3_sobolev_calculus_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
LOW_PATH = RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json"
EVOLUTION_PATH = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"
TUBE_PATH = RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"
SOURCE_PATH = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_r3_sobolev_calculus_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-r3-sobolev-calculus-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_r3_fourier_constants_and_chain_multiplicities_are_exact() -> None:
    passed, evidence = generic_r3_sobolev_chain_control()
    assert passed
    assert evidence["H6_embedding_constant_squares"] == {
        "0": "7/(1024*pi)",
        "1": "3/(1024*pi)",
        "2": "3/(1024*pi)",
        "3": "7/(1024*pi)",
        "4": "63/(1024*pi)",
    }
    assert set(evidence["embedding_residuals"].values()) == {"0"}
    assert set(evidence["spatial_chain_residuals"].values()) == {"0"}
    assert set(evidence["time_chain_residuals"].values()) == {"0"}
    assert evidence["weighted_convolution_binomial_sums"] == {
        "left": "32",
        "right": "32",
    }
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_r3_h6_spatialized_symbol_bounds() -> None:
    result = run_quartic_r3_sobolev_calculus_campaign(
        _load(LOW_PATH),
        _load(EVOLUTION_PATH),
        _load(TUBE_PATH),
        _load(SOURCE_PATH),
        _load(CONFIG_PATH),
    )
    assert result["status"] == "pass_all_12_R3_H6_spatialized_K55_P55_symbol_bounds"
    assert result["counts"] == {
        "selected": 12,
        "R3_symbol_spatializations_passed": 12,
        "rejected": 0,
    }
    assert all(
        len(item["spatialized_global_K55_bounds"]) == 15
        and len(item["spatialized_dyadic_P55_bounds"]) == 15
        and len(item["spatialized_time_K55_bounds"]) == 10
        and item["sufficient_H6_radius_for_state_and_spatial_jet_tube"][
            "H6_radius_numeric"
        ]
        > 0
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_wrong_sobolev_order_dimension_and_provenance_reject() -> None:
    low = _load(LOW_PATH)
    evolution = _load(EVOLUTION_PATH)
    tube = _load(TUBE_PATH)
    source = _load(SOURCE_PATH)
    config = _load(CONFIG_PATH)

    wrong_order = dict(config)
    wrong_order["sobolev_order"] = 5
    result = run_quartic_r3_sobolev_calculus_campaign(
        low, evolution, tube, source, wrong_order
    )
    assert result["status"] == "reject"
    assert "requires H6" in result["errors"][0]

    wrong_dimension = dict(config)
    wrong_dimension["spatial_dimension"] = 2
    result = run_quartic_r3_sobolev_calculus_campaign(
        low, evolution, tube, source, wrong_dimension
    )
    assert result["status"] == "reject"
    assert "R3, 55 evolution states" in result["errors"][0]

    corrupt = json.loads(json.dumps(evolution))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_r3_sobolev_calculus_campaign(
        low, corrupt, tube, source, config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
