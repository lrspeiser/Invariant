import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_anti_wick_composition_campaign import (
    generic_anti_wick_composition_audit,
    run_quartic_anti_wick_composition_campaign,
    validate_quartic_anti_wick_composition_artifact,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
LOW = RUNS / "quartic-low-frequency-symbol-extension-campaign" / "campaign.json"
EVOLUTION = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"
R3 = RUNS / "quartic-r3-sobolev-calculus-campaign" / "campaign.json"
TIME = RUNS / "quartic-time-atom-budget-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_anti_wick_composition_campaign.json"
ARTIFACT = RUNS / "quartic-anti-wick-composition-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_heat_symbol_annular_schur_and_smoothing_witness_are_exact() -> None:
    passed, control = generic_anti_wick_composition_audit()
    assert passed
    assert control["anti_wick_to_weyl"]["heat_time"] == "h/4"
    assert control["anti_wick_to_weyl"]["coherent_projector_midpoint_square_residual"] == "0"
    assert control["anti_wick_to_weyl"]["frequency_heat_transform_residual"] == "0"
    assert control["annular_positive_energy"]["positive"]
    assert control["amplitude_Schur_lemma"]["exact_coefficient"] == "1/(8*pi)"
    assert control["smoothing_defect_negative"]["omitting_smoothing_family_rejected"]
    assert control["smoothing_defect_negative"]["pointwise_KA_minus_ATK"] == (
        "Matrix([[0, 0], [0, 0]])"
    )
    assert control["exact_composition_amplitude"]["FTOC_polynomial_residual"] == "0"
    assert control["exact_composition_amplitude"]["phase_transfer_residual"] == "0"
    assert control["derivative_audit"]["required_maximum_mixed_total_order"] == 6


def test_all_candidates_are_audited_and_held_fail_closed_at_C4() -> None:
    result = run_quartic_anti_wick_composition_campaign(
        _load(LOW), _load(EVOLUTION), _load(R3), _load(TIME), _load(CONFIG)
    )
    assert result["status"] == ("pass_exact_anti_wick_composition_prerequisite_audit_C6_required")
    assert result["counts"] == {
        "selected": 12,
        "exact_composition_prerequisite_audits_passed": 12,
        "anti_wick_compositions_closed": 0,
        "C6_extensions_required": 12,
        "rejected": 0,
    }
    assert all(
        item["status"] == "fail_closed_requires_C6_spatial_frequency_symbol_bounds"
        and not item["anti_wick_composition_closed"]
        and item["time_K55_order_0_bound_available"]
        and item["P55_A0_A1_bounds_available"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_false_C4_closure_and_corrupt_provenance_reject() -> None:
    low, evolution, r3, time, config = map(_load, (LOW, EVOLUTION, R3, TIME, CONFIG))
    false_closure = dict(config)
    false_closure["required_mixed_total_order"] = 4
    result = run_quartic_anti_wick_composition_campaign(low, evolution, r3, time, false_closure)
    assert result["status"] == "reject"

    corrupt = json.loads(json.dumps(time))
    corrupt["upstream_sha256"]["low_frequency"] = "corrupt"
    corrupt_body = {key: value for key, value in corrupt.items() if key != "content_sha256"}
    corrupt["content_sha256"] = hashlib.sha256(
        json.dumps(corrupt_body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    result = run_quartic_anti_wick_composition_campaign(low, evolution, r3, corrupt, config)
    assert result["status"] == "reject"
    assert "not the registered artifact" in result["errors"][0]


def test_public_validator_rejects_resealed_promotion_and_bound_input_substitution() -> None:
    artifact, config = _load(ARTIFACT), _load(CONFIG)
    validate_quartic_anti_wick_composition_artifact(artifact, ROOT, config)
    promoted = json.loads(json.dumps(artifact))
    promoted["counts"]["anti_wick_compositions_closed"] = 12
    body = {key: value for key, value in promoted.items() if key != "content_sha256"}
    promoted["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        validate_quartic_anti_wick_composition_artifact(promoted, ROOT, config)

    low = _load(LOW)
    low["counts"]["selected"] = 0
    body = {key: value for key, value in low.items() if key != "content_sha256"}
    low["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    result = run_quartic_anti_wick_composition_campaign(
        low, _load(EVOLUTION), _load(R3), _load(TIME), config
    )
    assert result["status"] == "reject"
    assert "not the registered artifact" in result["errors"][0]
