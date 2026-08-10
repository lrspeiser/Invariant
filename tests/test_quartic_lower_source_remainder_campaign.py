import json
from pathlib import Path

from sigma_theory_compiler.quartic_lower_source_remainder_campaign import (
    generic_component_tensor_remainder_control,
    run_quartic_lower_source_remainder_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
UNSPECIALIZED = RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json"
CONTRACT = RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json"
NONLINEAR = RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json"
SOLVED = RUNS / "quartic-solved-source-moser-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_lower_source_remainder_campaign.json"
ARTIFACT = RUNS / "quartic-lower-source-remainder-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_component_tensor_remainder_control_and_negatives() -> None:
    passed, control = generic_component_tensor_remainder_control()
    assert passed
    assert control["exact_tensor_entry_count"] == 56
    assert control["identity_residual_zero"]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_lower_columns_are_mapped_without_norm_inference() -> None:
    result = run_quartic_lower_source_remainder_campaign(
        _load(UNSPECIALIZED),
        _load(CONTRACT),
        _load(NONLINEAR),
        _load(SOLVED),
        _load(CONFIG),
    )
    assert result["status"] == (
        "audit_all_12_lower_source_maps_component_remainder_fail_closed"
    )
    assert result["counts"]["lower_entries_missing_per_candidate"] == 594
    assert result["counts"]["candidate_component_remainders_proved"] == 0
    for certificate in result["certificates"]:
        packet = certificate["lower_source_column_map"]
        assert packet["column_count"] == 54
        assert packet["missing_entry_count"] == 594
        assert [item["column"] for item in packet["columns"]] == list(range(54))
        assert len({item["atom"] for item in packet["columns"]}) == 54
        assert certificate["source_jacobian_completion"]["exact_entries_completed"] == 1089
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = tuple(map(_load, (UNSPECIALIZED, CONTRACT, NONLINEAR, SOLVED)))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_component_remainder_proved"] = True
    result = run_quartic_lower_source_remainder_campaign(*campaigns, false_claim)
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[2]))
    corrupt["certificates"][0]["evolution_formula_contract_sha256"] = "corrupt"
    result = run_quartic_lower_source_remainder_campaign(
        campaigns[0], campaigns[1], corrupt, campaigns[3], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
