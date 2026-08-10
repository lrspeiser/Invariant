import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_homogeneous_frequency_symbol_campaign import (
    generic_homogeneous_frequency_chain_rule_control,
    run_quartic_homogeneous_frequency_symbol_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SYMBOL_PATH = RUNS / "quartic-symmetrizer-symbol-moser-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_homogeneous_frequency_symbol_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-homogeneous-frequency-symbol-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    campaign["content_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()


def test_normalization_and_bell_constants_are_exact() -> None:
    passed, evidence = generic_homogeneous_frequency_chain_rule_control()
    assert passed
    assert evidence["inverse_radius_Frechet_majorants"] == {
        "0": 1,
        "1": 1,
        "2": 4,
        "3": 24,
        "4": 204,
    }
    assert evidence["normalization_map_Frechet_majorants"] == {
        "0": 1,
        "1": 2,
        "2": 6,
        "3": 36,
        "4": 300,
    }
    assert set(evidence["bell_chain_rule_residuals"].values()) == {"0"}
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_homogeneous_frequency_bounds() -> None:
    result = run_quartic_homogeneous_frequency_symbol_campaign(
        _load(SYMBOL_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == "pass_all_12_full_K55_homogeneous_frequency_C4_bounds"
    assert result["counts"] == {
        "selected": 12,
        "homogeneous_frequency_bounds_passed": 12,
        "rejected": 0,
    }
    assert all(
        len(item["homogeneous_frequency_K55_bounds"]) == 15
        and all(entry["equal"] for entry in item["state_only_crosscheck"].values())
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_coordinate_multiindex_coverage_and_fourth_order_growth_are_explicit() -> None:
    result = run_quartic_homogeneous_frequency_symbol_campaign(
        _load(SYMBOL_PATH), _load(CONFIG_PATH)
    )
    first = result["certificates"][0]["homogeneous_frequency_K55_bounds"]
    assert first["0,4"]["coordinate_multiindices_covered"] == 15
    assert first["1,3"]["coordinate_multiindices_covered"] == 10
    assert first["2,2"]["coordinate_multiindices_covered"] == 6
    assert first["3,1"]["coordinate_multiindices_covered"] == 3
    assert first["4,0"]["coordinate_multiindices_covered"] == 1
    maxima = [
        max(
            float(entry["numeric_at_radius_lower"])
            for key, entry in item["homogeneous_frequency_K55_bounds"].items()
            if sum(int(value) for value in key.split(",")) == 4
        )
        for item in result["certificates"]
    ]
    assert min(maxima) > 8e44
    assert max(maxima) < 9e44


def test_insufficient_order_dimension_and_missing_ceiling_reject() -> None:
    symbol = _load(SYMBOL_PATH)
    config = _load(CONFIG_PATH)
    insufficient = dict(config)
    insufficient["maximum_total_derivative_order"] = 3
    result = run_quartic_homogeneous_frequency_symbol_campaign(symbol, insufficient)
    assert result["status"] == "reject"
    assert "total order four" in result["errors"][0]

    wrong_dimension = dict(config)
    wrong_dimension["spatial_dimension"] = 2
    result = run_quartic_homogeneous_frequency_symbol_campaign(symbol, wrong_dimension)
    assert result["status"] == "reject"
    assert "three spatial dimensions" in result["errors"][0]

    corrupt = json.loads(json.dumps(symbol))
    corrupt["certificates"][0].pop(
        "K55_mixed_Frechet_2_norm_envelope_integer_ceilings"
    )
    _rehash(corrupt)
    result = run_quartic_homogeneous_frequency_symbol_campaign(corrupt, config)
    assert result["status"] == "reject"
    assert "outward integer ceilings" in result["errors"][0]

    corrupt_hash = json.loads(json.dumps(symbol))
    corrupt_hash["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_homogeneous_frequency_symbol_campaign(corrupt_hash, config)
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
