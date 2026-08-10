import json
from pathlib import Path

from sigma_theory_compiler.quartic_metric_rows_tensor_dag_campaign import (
    generic_metric_row_affinity_control,
    run_quartic_metric_rows_tensor_dag_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SEMANTIC = RUNS / "quartic-universal-source-dag-campaign" / "campaign.json"
SCALAR = RUNS / "quartic-scalar-row-lowering-campaign" / "campaign.json"
PRINCIPAL = RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json"
NONLINEAR = RUNS / "quartic-nonlinear-evolution-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_metric_rows_tensor_dag_campaign.json"
ARTIFACT = RUNS / "quartic-metric-rows-tensor-dag-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_all_ten_metric_row_affinity_cancellations_and_negative() -> None:
    passed, control = generic_metric_row_affinity_control()
    assert passed
    assert set(control["Hessian_quadratic_residuals"].values()) == {"0"}
    assert control["curvature_acceleration_degree"] == 1
    assert control["gauge_acceleration_degree"] == 1
    assert control["negative_control"]["rejected_rows"] == 10


def test_all_candidates_emit_complete_row_dag_and_partial_mixed_roots() -> None:
    result = run_quartic_metric_rows_tensor_dag_campaign(
        _load(SEMANTIC),
        _load(SCALAR),
        _load(PRINCIPAL),
        _load(NONLINEAR),
        _load(CONFIG),
    )
    assert result["status"] == (
        "pass_all_12_all_Euler_rows_tensor_lowered_mixed_incomplete_fail_closed"
    )
    assert result["counts"]["metric_rows_lowered_per_candidate"] == 10
    assert result["counts"]["Euler_rows_affine_per_candidate"] == 11
    assert result["counts"]["lower_Jacobian_roots_per_candidate"] == 594
    assert result["counts"]["A_W_mixed_roots_per_candidate"] == 1440
    assert result["counts"]["selected_mixed_F_roots_per_candidate"] == 132
    assert len(result["common_explicit_tensor_dag_packet"]["row_checkpoints"]) == 5
    for certificate in result["certificates"]:
        assert certificate["all_11_Euler_rows_acceleration_affine"]
        assert certificate["full_11x153_source_Jacobian_operational_roots_emitted"]
        assert not certificate["full_11x153_source_Jacobian_entrywise_materialized"]
        assert not certificate["full_component_Frechet_tensors_complete"]
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = tuple(map(_load, (SEMANTIC, SCALAR, PRINCIPAL, NONLINEAR)))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_full_component_remainder_proved"] = True
    result = run_quartic_metric_rows_tensor_dag_campaign(*campaigns, false_claim)
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[1]))
    corrupt["certificates"][0]["provenance"][
        "evolution_formula_contract_sha256"
    ] = "corrupt"
    result = run_quartic_metric_rows_tensor_dag_campaign(
        campaigns[0], corrupt, campaigns[2], campaigns[3], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
