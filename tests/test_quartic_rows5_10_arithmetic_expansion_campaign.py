import json
from pathlib import Path

from sigma_theory_compiler.quartic_row0_arithmetic_expansion_campaign import (
    _content_hash,
)
from sigma_theory_compiler.quartic_rows5_10_arithmetic_expansion_campaign import (
    run_quartic_rows5_10_arithmetic_expansion_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ROW4 = RUNS / "quartic-row4-arithmetic-expansion-campaign" / "campaign.json"
METRIC = RUNS / "quartic-metric-rows-tensor-dag-campaign" / "campaign.json"
CONFIG = ROOT / "configs" / "backgrounds" / "quartic_rows5_10_arithmetic_expansion_campaign.json"
ARTIFACT = RUNS / "quartic-rows5-10-arithmetic-expansion-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_rows5_10_complete_lower_arithmetic_and_cumulative_coverage() -> None:
    result = run_quartic_rows5_10_arithmetic_expansion_campaign(
        _load(ROW4), _load(METRIC), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_all_lower_rows_arithmetic_mixed_tensor_fail_closed"
    )
    packet = result["common_rows5_10_arithmetic_packet"]
    assert packet["output_rows"] == [5, 6, 7, 8, 9, 10]
    assert packet["counts"]["lower_entries_normalized"] == 324
    assert packet["counts"]["mixed_entries_normalized"] == 36
    assert packet["counts"]["semantic_or_tensor_operations_in_output_dag"] == 0
    dag = packet["arithmetic_dag"]
    assert dag["node_count"] == 48129
    assert {node["op"] for node in dag["nodes"]} <= set(dag["allowed_operations"])
    assert {
        (item["output_row"], item["column"]) for item in packet["lower_Jacobian"]
    } == {(row, column) for row in range(5, 11) for column in range(54)}
    assert len(packet["lower_Jacobian"]) == 324
    assert all(item["normalized_residual"] == "0" for item in packet["lower_Jacobian"])
    assert len(packet["selected_mixed_F"]) == 36
    assert all(
        item["normalized_coefficient_residual"] == "0"
        for item in packet["selected_mixed_F"]
    )
    assert result["counts"]["cumulative_lower_entries_normalized_per_candidate"] == 594
    assert result["counts"]["cumulative_selected_mixed_entries_per_candidate"] == 66
    for certificate in result["certificates"]:
        assert certificate["full_lower_11x54_Jacobian_entrywise_materialized"]
        assert not certificate["full_11x153_source_Jacobian_entrywise_materialized"]
        assert not certificate["full_component_Frechet_tensors_complete"]
        assert not certificate["paralinearization_remainder_bound_proved"]
        assert not certificate["full_H7_commutator_closed"]
        assert all(
            certificate["row_coverage"][str(row)][
                "lower_entries_arithmetic_normalized"
            ]
            == 54
            for row in range(11)
        )
        assert {
            f"row{row}_arithmetic_dag_sha256" for row in range(5)
        } <= set(certificate["provenance"])
    assert result == _load(ARTIFACT)


def test_false_remainder_and_corrupt_provenance_reject() -> None:
    campaigns = (_load(ROW4), _load(METRIC))
    config = _load(CONFIG)
    false_claim = dict(config)
    false_claim["declare_component_remainder_proved"] = True
    result = run_quartic_rows5_10_arithmetic_expansion_campaign(
        *campaigns, false_claim
    )
    assert result["status"] == "reject"
    assert "cannot be declared" in result["errors"][0]

    corrupt = json.loads(json.dumps(campaigns[0]))
    corrupt["certificates"][0]["provenance"]["row4_arithmetic_dag_sha256"] = "corrupt"
    result = run_quartic_rows5_10_arithmetic_expansion_campaign(
        corrupt, campaigns[1], config
    )
    assert result["status"] == "reject"
    assert "content hash mismatch" in result["errors"][0]

    rebound = json.loads(json.dumps(campaigns[0]))
    rebound["certificates"][0]["provenance"].pop(
        "row3_arithmetic_dag_sha256"
    )
    rebound_body = {key: value for key, value in rebound.items() if key != "content_sha256"}
    rebound["content_sha256"] = _content_hash(rebound_body)
    result = run_quartic_rows5_10_arithmetic_expansion_campaign(
        rebound, campaigns[1], config
    )
    assert result["status"] == "reject"
    assert "rows0-4 arithmetic provenance is incomplete" in result["errors"][0]
