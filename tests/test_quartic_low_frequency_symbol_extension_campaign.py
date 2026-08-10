import json
from pathlib import Path

from sigma_theory_compiler.quartic_low_frequency_symbol_extension_campaign import (
    generic_low_frequency_extension_control,
    run_quartic_low_frequency_symbol_extension_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
HOMOGENEOUS_PATH = RUNS / "quartic-homogeneous-frequency-symbol-campaign" / "campaign.json"
SYMBOL_PATH = RUNS / "quartic-symmetrizer-symbol-moser-campaign" / "campaign.json"
FULL_PATH = RUNS / "quartic-full-symmetrizer-moser-campaign" / "campaign.json"
CONFIG_PATH = (
    ROOT / "configs" / "backgrounds" / "quartic_low_frequency_symbol_extension_campaign.json"
)
ARTIFACT_PATH = RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict]:
    return _load(HOMOGENEOUS_PATH), _load(SYMBOL_PATH), _load(FULL_PATH), _load(CONFIG_PATH)


def test_cutoff_gluing_and_radial_constants_are_exact() -> None:
    passed, evidence = generic_low_frequency_extension_control()
    assert passed
    assert set(evidence["endpoint_C4_residuals"].values()) == {"0"}
    assert evidence["radius_map_Frechet_majorants"] == {
        "0": 1,
        "1": 1,
        "2": 2,
        "3": 6,
        "4": 36,
    }
    assert evidence["radial_cutoff_Frechet_majorants"] == {
        "0": 1,
        "1": 10080,
        "2": 80640,
        "3": 735840,
        "4": 7650720,
    }
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_positive_global_C4_extensions() -> None:
    result = run_quartic_low_frequency_symbol_extension_campaign(*_inputs())
    assert result["status"] == "pass_all_12_global_C4_positive_K55_symbol_extensions"
    assert result["counts"] == {
        "selected": 12,
        "global_C4_positive_symbol_extensions_passed": 12,
        "rejected": 0,
    }
    assert all(
        item["energy_equivalence"]["K55_2_lower_numeric"] > 0
        and len(item["global_C4_frequency_derivative_integer_ceilings"]) == 15
        and item["extension_definition"]["regularity"]
        == "C4 in xi and C4 in the certified state variables"
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_transition_bounds_cover_every_frequency_multiindex() -> None:
    result = run_quartic_low_frequency_symbol_extension_campaign(*_inputs())
    first = result["certificates"][0][
        "global_C4_frequency_derivative_integer_ceilings"
    ]
    assert first["0,4"]["coordinate_multiindices_covered"] == 15
    assert first["1,3"]["coordinate_multiindices_covered"] == 10
    assert first["2,2"]["coordinate_multiindices_covered"] == 6
    assert first["3,1"]["coordinate_multiindices_covered"] == 3
    assert first["4,0"]["coordinate_multiindices_covered"] == 1
    assert all(
        int(entry["global_ceiling"]) >= int(entry["high_frequency_ceiling"])
        and int(entry["global_ceiling"]) >= int(entry["transition_ceiling"])
        for entry in first.values()
    )


def test_wrong_order_cutoff_and_provenance_reject() -> None:
    homogeneous, symbol, full, config = _inputs()
    insufficient = dict(config)
    insufficient["maximum_total_derivative_order"] = 3
    result = run_quartic_low_frequency_symbol_extension_campaign(
        homogeneous, symbol, full, insufficient
    )
    assert result["status"] == "reject"
    assert "total order four" in result["errors"][0]

    wrong_cutoff = dict(config)
    wrong_cutoff["outer_radius"] = "3"
    result = run_quartic_low_frequency_symbol_extension_campaign(
        homogeneous, symbol, full, wrong_cutoff
    )
    assert result["status"] == "reject"
    assert "outer radius two" in result["errors"][0]

    corrupt = json.loads(json.dumps(homogeneous))
    corrupt["symbol_campaign_sha256"] = "corrupt"
    result = run_quartic_low_frequency_symbol_extension_campaign(
        corrupt, symbol, full, config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
