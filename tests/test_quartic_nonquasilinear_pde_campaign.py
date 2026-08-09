import json
from pathlib import Path

from sigma_theory_compiler.quartic_nonquasilinear_pde_campaign import (
    generic_full_symmetrizer_lift_control,
    generic_nonquasilinear_acceleration_control,
    run_quartic_nonquasilinear_pde_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_nonquasilinear_pde_campaign.json"
ARTIFACT_PATH = RUNS / "quartic-nonquasilinear-pde-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict, dict, dict, dict]:
    return (
        _load(RUNS / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"),
        _load(RUNS / "quartic-quasilinear-moser-campaign" / "campaign.json"),
        _load(RUNS / "quartic-first-order-reduction-campaign" / "campaign.json"),
        _load(RUNS / "quartic-geometric-jet-campaign" / "campaign.json"),
        _load(RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json"),
        _load(CONFIG_PATH),
    )


def test_generic_nonquasilinear_acceleration_identities_retain_second_spatial_jets() -> None:
    passed, evidence = generic_nonquasilinear_acceleration_control()
    assert passed
    assert evidence["mixed_identity_residual"] == "0"
    assert evidence["spatial_identity_residual"] == "0"
    assert "not lower order" in evidence["terminology"]
    assert evidence["negative_control"]["rejected"]


def test_generic_full_symmetrizer_lift_is_positive_and_exact() -> None:
    passed, evidence = generic_full_symmetrizer_lift_control()
    assert passed
    assert evidence["K55_M55_minus_M55_dagger_K55_zero"]
    assert evidence["all_LDL_pivots_positive"]
    assert evidence["negative_control"]["rejected"]


def test_all_candidates_receive_full_nonquasilinear_pde_certificate() -> None:
    result = run_quartic_nonquasilinear_pde_campaign(*_inputs())
    assert (
        result["status"]
        == "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts"
    )
    assert result["counts"]["full_55_state_symmetrizer_lifts_passed"] == 12
    assert len(result["certificates"]) == 12
    assert all(
        item["status"]
        == "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift"
        and item["full_first_order_state"]["total"] == 55
        and item["uniform_bounds"]["characteristic_absolute_lower"] == "1/4"
        and item["uniform_bounds"]["K55_2_lower_numeric"] > 0
        and item["conditional_local_wellposedness"]["status"]
        == "theorem_applies_to_compatible_vacuum_data_in_compact_box_interior"
        and "global or long-time preservation of the 2e-10 local-jet box"
        in item["conditional_local_wellposedness"]["not_certified"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT_PATH)


def test_characteristic_gap_and_provenance_corruptions_reject() -> None:
    symmetrizer, moser, first_order, geometric, nonlinear, config = _inputs()
    bad_gap = dict(config)
    bad_gap["characteristic_absolute_lower_bound"] = "0"
    result = run_quartic_nonquasilinear_pde_campaign(
        symmetrizer, moser, first_order, geometric, nonlinear, bad_gap
    )
    assert result["status"] == "reject"
    assert "characteristic lower bound" in result["errors"][0]

    corrupted = dict(nonlinear)
    corrupted["geometric_campaign_sha256"] = "corrupted"
    result = run_quartic_nonquasilinear_pde_campaign(
        symmetrizer, moser, first_order, geometric, corrupted, config
    )
    assert result["status"] == "reject"
    assert "provenance chain" in result["errors"][0]
