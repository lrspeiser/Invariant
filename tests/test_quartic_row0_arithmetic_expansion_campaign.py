import json
from pathlib import Path

from sigma_theory_compiler.quartic_row0_arithmetic_expansion_campaign import (
    generic_arithmetic_materialization_control,
    run_quartic_row0_arithmetic_expansion_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
METRIC = RUNS / "quartic-metric-rows-tensor-dag-campaign" / "campaign.json"
PRINCIPAL = RUNS / "quartic-unspecialized-source-jacobian-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_row0_arithmetic_expansion_campaign.json"
ARTIFACT = RUNS / "quartic-row0-arithmetic-expansion-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_faddeev_leverrier_and_mixed_recurrence_control() -> None:
    passed, control = generic_arithmetic_materialization_control()
    assert passed
    assert control["Faddeev_Leverrier_inverse_residual"] == (
        "Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])"
    )
    assert set(control["mixed_equation_residuals"].values()) == {
        "Matrix([[0], [0], [0]])"
    }
    assert control["negative_control"]["rejected"]


def test_row0_all_lower_entries_are_arithmetic_only_and_reproducible() -> None:
    result = run_quartic_row0_arithmetic_expansion_campaign(
        _load(METRIC), _load(PRINCIPAL), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_row0_arithmetic_materialized_other_rows_fail_closed"
    )
    packet = result["common_row0_arithmetic_packet"]
    assert packet["counts"]["lower_entries_normalized"] == 54
    assert packet["counts"]["mixed_entries_normalized"] == 6
    assert packet["counts"]["semantic_or_tensor_operations_in_output_dag"] == 0
    assert all(
        item["normalized_residual"] == "0"
        for item in packet["lower_Jacobian_row0"]
    )
    assert all(
        item["normalized_coefficient_residual"] == "0"
        for item in packet["selected_mixed_F_row0"]
    )
    for certificate in result["certificates"]:
        assert certificate["lower_Jacobian_arithmetic_entries_normalized"] == 54
        assert certificate["row_coverage"]["0"]["complete_for_configured_slice"]
        assert not certificate["row_coverage"]["1"]["complete_for_configured_slice"]
        assert not certificate["full_11x153_source_Jacobian_entrywise_materialized"]
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = (_load(METRIC), _load(PRINCIPAL))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_component_remainder_proved"] = True
    result = run_quartic_row0_arithmetic_expansion_campaign(*campaigns, false_claim)
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[0]))
    corrupt["certificates"][0]["provenance"]["common_root_packet_sha256"] = "corrupt"
    result = run_quartic_row0_arithmetic_expansion_campaign(
        corrupt, campaigns[1], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
