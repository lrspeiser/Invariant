import json
from pathlib import Path

from sigma_theory_compiler.quartic_symmetrizer_symbol_moser_campaign import (
    generic_bivariate_symbol_derivative_control,
    run_quartic_symmetrizer_symbol_moser_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SYMMETRIZER_PATH = RUNS / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
MOSER_PATH = RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
TUBE_PATH = RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"
SOURCE_PATH = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
FULL_PATH = RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_symmetrizer_symbol_moser_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-symmetrizer-symbol-moser-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict, dict]:
    return (
        _load(SYMMETRIZER_PATH),
        _load(MOSER_PATH),
        _load(PDE_PATH),
        _load(TUBE_PATH),
        _load(SOURCE_PATH),
        _load(FULL_PATH),
        _load(CONFIG_PATH),
    )


def test_bivariate_inverse_and_product_multiplicities_are_exact() -> None:
    passed, evidence = generic_bivariate_symbol_derivative_control()
    assert passed
    assert len(evidence["multiindices"]) == 15
    assert set(evidence["inverse_residuals"].values()) == {"0"}
    assert set(evidence["triple_product_residuals"].values()) == {"0"}
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_mixed_state_direction_k55_bounds() -> None:
    result = run_quartic_symmetrizer_symbol_moser_campaign(*_inputs())
    assert (
        result["status"]
        == "pass_all_12_full_K55_mixed_state_direction_C4_symbol_envelopes"
    )
    assert result["counts"] == {
        "selected": 12,
        "mixed_symbol_envelopes_passed": 12,
        "rejected": 0,
    }
    expected = {
        f"{state},{total - state}"
        for total in range(5)
        for state in range(total + 1)
    }
    assert len(expected) == 15
    assert all(
        set(item["K55_mixed_Frechet_2_norm_envelopes_numeric"]) == expected
        and all(item["state_only_coverage_crosscheck"][str(order)]["covers"] for order in range(5))
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_raw_direction_degree_and_total_four_growth_are_explicit() -> None:
    result = run_quartic_symmetrizer_symbol_moser_campaign(*_inputs())
    raw = result["uniform_raw_mixed_derivative_envelopes"]
    assert raw["B"]["0,2"]["exact"] == "0"
    assert raw["H_star"]["0,2"]["exact"] == "0"
    assert raw["C"]["0,2"]["numeric"] > 0
    maxima = [
        max(item["K55_total_order_four_envelopes_numeric"].values())
        for item in result["certificates"]
    ]
    assert min(maxima) > 8e44
    assert max(maxima) < 9e44


def test_insufficient_total_order_and_corrupt_provenance_reject() -> None:
    symmetrizer, moser, pde, tube, source, full, config = _inputs()
    insufficient = dict(config)
    insufficient["maximum_total_derivative_order"] = 3
    result = run_quartic_symmetrizer_symbol_moser_campaign(
        symmetrizer, moser, pde, tube, source, full, insufficient
    )
    assert result["status"] == "reject"
    assert "total order four" in result["errors"][0]

    corrupted_full = json.loads(json.dumps(full))
    corrupted_full["upstream_sha256"]["moser"] = "corrupt"
    result = run_quartic_symmetrizer_symbol_moser_campaign(
        symmetrizer, moser, pde, tube, source, corrupted_full, config
    )
    assert result["status"] == "reject"
    assert "full-symmetrizer provenance" in result["errors"][0]
