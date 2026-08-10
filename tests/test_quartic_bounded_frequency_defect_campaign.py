import json
from pathlib import Path

from sigma_theory_compiler.quartic_bounded_frequency_defect_campaign import (
    generic_compact_frequency_defect_control,
    run_quartic_bounded_frequency_defect_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
LOW = RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json"
EVOLUTION = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"
FIRST_ORDER = RUNS / "quartic-first-order-reduction-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_bounded_frequency_defect_campaign.json"
ARTIFACT = RUNS / "quartic-bounded-frequency-defect-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_compact_defect_schur_and_scale_controls_are_exact() -> None:
    passed, control = generic_compact_frequency_defect_control()
    assert passed
    assert control["compact_symbol_Schur_lemma"]["exact_coefficient"] == "4/3"
    assert control["defect_identity_residual_after_outer_symmetrization"] == "0"
    assert control["radial_majorant_provenance"]["upstream_control_passed"]
    assert control["defect_derivative_multipliers_0_through_4"] == [
        4,
        40322,
        362880,
        3427200,
        36489600,
    ]
    assert control["physical_scale_contract"]["high_shell_defect_zero"]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_receive_physical_low_frequency_operator_bounds() -> None:
    result = run_quartic_bounded_frequency_defect_campaign(
        _load(LOW), _load(EVOLUTION), _load(FIRST_ORDER), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_actual_P55_compact_frequency_defect_KN_L2_lemmas"
    )
    assert result["counts"]["compact_frequency_defect_lemmas_passed"] == 12
    assert all(
        item["physical_pencil_provenance"][
            "frequency_derivatives_order_2_and_higher_zero"
        ]
        and item["physical_scale_contract_passed"]
        and not item["full_energy_closed"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_unscaled_radius_and_corrupt_provenance_reject() -> None:
    low, evolution, first_order, config = map(
        _load, (LOW, EVOLUTION, FIRST_ORDER, CONFIG)
    )
    wrong_scale = dict(config)
    wrong_scale["physical_frequency_radii"] = [1, 3]
    result = run_quartic_bounded_frequency_defect_campaign(
        low, evolution, first_order, wrong_scale
    )
    assert result["status"] == "reject"

    unscaled = dict(config)
    unscaled["semiclassical_symbol"] = "K_ext(U,x,eta)"
    result = run_quartic_bounded_frequency_defect_campaign(
        low, evolution, first_order, unscaled
    )
    assert result["status"] == "reject"
    assert "compact-defect domain" in result["errors"][0]

    corrupt = json.loads(json.dumps(first_order))
    corrupt["certificates"][0]["source_spatial_block_sha256"] = "corrupt"
    result = run_quartic_bounded_frequency_defect_campaign(
        low, evolution, corrupt, config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
