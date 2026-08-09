from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.quartic_quasilinear_moser_campaign import (
    generic_inverse_product_derivative_control,
    run_quartic_quasilinear_moser_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SYMMETRIZER_PATH = RUNS / "quartic-symmetrizer-uniform-domain-campaign" / "campaign.json"
AUXILIARY_PATH = RUNS / "quartic-auxiliary-time-campaign" / "campaign.json"
CONFIG_PATH = ROOT / "configs" / "backgrounds" / "quartic_quasilinear_moser_campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_inverse_product_derivative_recurrence_is_exact() -> None:
    passed, evidence = generic_inverse_product_derivative_control()
    assert passed
    assert set(evidence["residuals"].values()) == {"0"}
    assert evidence["negative_control"]["rejected"]


def test_all_quartic_candidates_have_c4_companion_coefficient_envelopes() -> None:
    result = run_quartic_quasilinear_moser_campaign(
        _load(SYMMETRIZER_PATH), _load(AUXILIARY_PATH), _load(CONFIG_PATH)
    )
    assert result["status"] == "pass_all_12_quasilinear_coefficient_derivative_envelopes"
    assert result["counts"] == {
        "selected": 12,
        "quasilinear_coefficient_envelopes_passed": 12,
        "rejected": 0,
    }
    assert result["negative_controls"]["false_linear_raw_coefficient_declaration"]["rejected"]
    assert result["negative_controls"]["insufficient_sobolev_order"]["rejected"]
    for item in result["certificates"]:
        assert item["raw_coefficient_degree"] == {"A": 2, "B": 2, "C": 2}
        raw = item["raw_Frechet_derivative_2_norm_envelopes"]
        assert all(raw[name]["3"] == "0" and raw[name]["4"] == "0" for name in raw)
        companion = item["companion_Frechet_derivative_2_norm_envelopes_numeric"]
        assert all(companion[str(order)] > 0 for order in range(5))
        assert "does not reconstruct the nonlinear state-to-covariant-jet map" in item["scope"]


def test_quasilinear_campaign_rejects_prerequisite_and_candidate_mismatch() -> None:
    symmetrizers = _load(SYMMETRIZER_PATH)
    auxiliaries = _load(AUXILIARY_PATH)
    config = _load(CONFIG_PATH)
    corrupted = json.loads(json.dumps(auxiliaries))
    corrupted["certificates"][0]["candidate_id"] = "corrupted-candidate"
    result = run_quartic_quasilinear_moser_campaign(symmetrizers, corrupted, config)
    assert result["status"] == "reject"
    assert "campaign candidate sets do not match" in result["errors"]

    missing_prerequisite = json.loads(json.dumps(auxiliaries))
    missing_prerequisite["status"] = "reject"
    result = run_quartic_quasilinear_moser_campaign(
        symmetrizers, missing_prerequisite, config
    )
    assert result["status"] == "reject"
    assert "auxiliary-time campaign prerequisite failed" in result["errors"]
