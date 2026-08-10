import json
from pathlib import Path

from sigma_theory_compiler.quartic_paradifferential_good_unknown_campaign import (
    generic_paradifferential_good_unknown_control,
    run_quartic_paradifferential_good_unknown_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
DYADIC = RUNS / "quartic-dyadic-localization-campaign" / "campaign.json"
SOURCE = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
EVOLUTION = RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"
FIRST_ORDER = RUNS / "quartic-first-order-reduction-campaign" / "campaign.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_paradifferential_good_unknown_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-paradifferential-good-unknown-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_bony_partition_top_jet_and_naive_negative_are_exact() -> None:
    passed, control = generic_paradifferential_good_unknown_control()
    assert passed
    assert control["dyadic_interaction_partition"]["one_hot_residuals_all_zero"]
    assert set(control["finite_Fourier_Bony_identity"]["output_residuals"].values()) == {
        "0"
    }
    assert control["top_derivative_isolation"]["D_y7_remainder_residual"] == "0"
    assert control["naive_commutator_negative"]["exact_classification"] == (
        "high_low"
    )
    assert control["naive_commutator_negative"]["rejected"]


def test_all_candidates_get_framework_but_component_binding_stays_closed() -> None:
    result = run_quartic_paradifferential_good_unknown_campaign(
        _load(DYADIC),
        _load(SOURCE),
        _load(EVOLUTION),
        _load(FIRST_ORDER),
        _load(CONFIG),
    )
    assert result["status"] == (
        "pass_all_12_paradifferential_good_unknown_audits_"
        "component_binding_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "good_unknown_frameworks_passed": 12,
        "component_source_pencil_bindings_closed": 0,
        "global_H7_summations_applied": 0,
        "rejected": 0,
    }
    assert all(
        item["naive_H6_to_H7_commutator_rejected"]
        and not item["source_to_physical_pencil_component_binding"]["closed"]
        and not item["H7_derivative_loss_resolved"]
        and not item["global_dyadic_summation_applied"]
        for item in result["certificates"]
    )
    assert result == _load(ARTIFACT)


def test_false_global_closure_and_corrupt_provenance_reject() -> None:
    campaigns = tuple(map(_load, (DYADIC, SOURCE, EVOLUTION, FIRST_ORDER)))
    config = _load(CONFIG)
    false_closure = dict(config)
    false_closure["declare_global_H7_closed"] = True
    result = run_quartic_paradifferential_good_unknown_campaign(
        *campaigns, false_closure
    )
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[1]))
    corrupt["certificates"][0]["candidate_id"] = "corrupt"
    result = run_quartic_paradifferential_good_unknown_campaign(
        campaigns[0], corrupt, *campaigns[2:], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
