import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_evolution_symbol_campaign import (
    generic_degree_one_evolution_symbol_control,
    run_quartic_evolution_symbol_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
FIRST_PATH = RUNS / "quartic-first-order-reduction-campaign" / "campaign.json"
PDE_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"
SYMBOL_PATH = RUNS / "quartic-symmetrizer-symbol-moser-campaign" / "campaign.json"
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_evolution_symbol_campaign.json"
ARTIFACT_PATH = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_block_lift_and_degree_one_product_recurrence_are_exact() -> None:
    passed, evidence = generic_degree_one_evolution_symbol_control()
    assert passed
    assert evidence["block_scalar_residual"] == "0"
    assert evidence["radius_map_Frechet_majorants"] == {
        "0": 1,
        "1": 1,
        "2": 2,
        "3": 6,
        "4": 36,
    }
    assert set(evidence["radial_Leibniz_residuals"].values()) == {"0"}
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_bind_the_exact_55_state_evolution_symbol() -> None:
    result = run_quartic_evolution_symbol_campaign(
        _load(FIRST_PATH), _load(PDE_PATH), _load(SYMBOL_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == (
        "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds"
    )
    assert result["counts"] == {
        "selected": 12,
        "evolution_symbol_bounds_passed": 12,
        "rejected": 0,
    }
    assert all(
        len(item["homogeneous_principal_P55_bounds"]) == 15
        and item["exact_reduction_provenance"]["state_dimension"] == 55
        and item["exact_reduction_provenance"][
            "nonzero_characteristic_lift_residual_zero"
        ]
        and all(
            entry["equal"] for entry in item["state_only_crosscheck"].values()
        )
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_wrong_order_dimension_and_corrupt_provenance_reject() -> None:
    first = _load(FIRST_PATH)
    pde = _load(PDE_PATH)
    symbol = _load(SYMBOL_PATH)
    config = _load(CONFIG_PATH)

    wrong_order = dict(config)
    wrong_order["maximum_total_derivative_order"] = 3
    result = run_quartic_evolution_symbol_campaign(first, pde, symbol, wrong_order)
    assert result["status"] == "reject"
    assert "total order four" in result["errors"][0]

    wrong_dimension = dict(config)
    wrong_dimension["state_dimension"] = 22
    result = run_quartic_evolution_symbol_campaign(
        first, pde, symbol, wrong_dimension
    )
    assert result["status"] == "reject"
    assert "55 states" in result["errors"][0]

    corrupt = json.loads(json.dumps(symbol))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_evolution_symbol_campaign(first, pde, corrupt, config)
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]

    wrong_chain = json.loads(json.dumps(pde))
    wrong_chain["upstream_sha256"]["first_order"] = "0" * 64
    body = {
        key: value for key, value in wrong_chain.items() if key != "content_sha256"
    }
    wrong_chain["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()
    result = run_quartic_evolution_symbol_campaign(
        first, wrong_chain, symbol, config
    )
    assert result["status"] == "reject"
    assert "provenance mismatch" in result["errors"][0]
