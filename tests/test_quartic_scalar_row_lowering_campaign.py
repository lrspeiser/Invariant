import json
from pathlib import Path

from sigma_theory_compiler.quartic_scalar_row_lowering_campaign import (
    generic_scalar_row_affinity_control,
    run_quartic_scalar_row_lowering_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SEMANTIC = RUNS / "quartic-universal-source-dag-campaign" / "campaign.json"
NONLINEAR = RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_scalar_row_lowering_campaign.json"
ARTIFACT = RUNS / "quartic-scalar-row-lowering-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_scalar_row_universal_affinity_and_negatives() -> None:
    passed, control = generic_scalar_row_affinity_control()
    assert passed
    assert control["universal_inverse_metric_symbol_count"] == 10
    assert control["G_upper_00_acceleration_part"] == "0"
    assert control["acceleration_total_degree"] == 1
    assert control["affine_residual"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_lower_scalar_row_and_checkpoint_mixed_tensors() -> None:
    result = run_quartic_scalar_row_lowering_campaign(
        _load(SEMANTIC), _load(NONLINEAR), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_universal_scalar_row_affinity_partial_mixed_checkpoints"
    )
    assert result["counts"]["mixed_A_W_component_roots_per_candidate"] == 144
    assert result["counts"]["solved_source_component_derivatives"] == 0
    for certificate in result["certificates"]:
        assert certificate["lowered_row"] == 10
        assert certificate["acceleration_affine_residual_proved_zero"]
        assert certificate["arithmetic_dag"]["node_count"] > 100
        assert certificate["exact_mixed_component_roots"] == 144
        assert len(certificate["mixed_derivative_checkpoints"]) == 2
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = (_load(SEMANTIC), _load(NONLINEAR))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_solved_source_remainder_proved"] = True
    result = run_quartic_scalar_row_lowering_campaign(*campaigns, false_claim)
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[1]))
    corrupt["certificates"][0]["evolution_formula_contract_sha256"] = "corrupt"
    result = run_quartic_scalar_row_lowering_campaign(
        campaigns[0], corrupt, config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
