import json
from pathlib import Path

from sigma_theory_compiler.quartic_row4_arithmetic_expansion_campaign import (
    run_quartic_row4_arithmetic_expansion_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ROW3 = RUNS / "quartic-row3-arithmetic-expansion-campaign" / "campaign.json"
METRIC = RUNS / "quartic-metric-rows-tensor-dag-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_row4_arithmetic_expansion_campaign.json"
ARTIFACT = RUNS / "quartic-row4-arithmetic-expansion-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_row4_exact_current_and_cumulative_coverage() -> None:
    result = run_quartic_row4_arithmetic_expansion_campaign(
        _load(ROW3), _load(METRIC), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_rows0_4_arithmetic_materialized_other_rows_fail_closed"
    )
    packet = result["common_row4_arithmetic_packet"]
    assert packet["counts"]["lower_entries_normalized"] == 54
    assert packet["counts"]["mixed_entries_normalized"] == 6
    assert packet["counts"]["semantic_or_tensor_operations_in_output_dag"] == 0
    assert all(item["output_row"] == 4 for item in packet["lower_Jacobian"])
    assert all(item["normalized_residual"] == "0" for item in packet["lower_Jacobian"])
    assert all(
        item["normalized_coefficient_residual"] == "0"
        for item in packet["selected_mixed_F"]
    )
    assert result["counts"]["cumulative_lower_entries_normalized_per_candidate"] == 270
    assert result["counts"]["cumulative_selected_mixed_entries_per_candidate"] == 30
    for certificate in result["certificates"]:
        assert certificate["row_coverage"]["4"][
            "lower_entries_arithmetic_normalized"
        ] == 54
        assert certificate["row_coverage"]["5"][
            "lower_entries_arithmetic_normalized"
        ] == 0
        assert not certificate["full_11x153_source_Jacobian_entrywise_materialized"]
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = (_load(ROW3), _load(METRIC))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_component_remainder_proved"] = True
    result = run_quartic_row4_arithmetic_expansion_campaign(*campaigns, false_claim)
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[0]))
    corrupt["certificates"][0]["provenance"]["row3_arithmetic_dag_sha256"] = "corrupt"
    result = run_quartic_row4_arithmetic_expansion_campaign(
        corrupt, campaigns[1], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]
