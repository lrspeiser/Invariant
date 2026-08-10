import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_row0_component_remainder_campaign import (
    generic_row0_component_remainder_control,
    run_quartic_row0_component_remainder_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
METRIC = RUNS / "quartic-metric-rows-tensor-dag-campaign" / "campaign.json"
ROW0 = RUNS / "quartic-row0-arithmetic-expansion-campaign" / "campaign.json"
GLOBAL = RUNS / "quartic-global-h7-energy-campaign" / "campaign.json"
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_row0_component_remainder_campaign.json"
)
ARTIFACT = RUNS / "quartic-row0-component-remainder-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple[dict, dict, dict]:
    return _load(METRIC), _load(ROW0), _load(GLOBAL)


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    ).hexdigest()


def test_constant_row_absorption_and_incomplete_mixed_negative_are_exact() -> None:
    passed, control = generic_row0_component_remainder_control()
    assert passed
    assert control["constant_matrix_Littlewood_Paley_commutator"] == (
        "Matrix([[0, 0, 0], [0, 0, 0]])"
    )
    assert control["energy_absorption"]["residual"] == "0"
    assert control["full_D2_to_D4_symmetric_component_total"] > 6
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_row0_reference_linear_CL_is_absorbed_but_CB_remains_open() -> None:
    result = run_quartic_row0_component_remainder_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_row0_reference_linear_slices_"
        "nonlinear_and_global_remainders_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "row0_reference_linear_C_L_contributions_certified": 12,
        "row0_nonlinear_C_B_contributions_certified": 0,
        "complete_row0_remainders_closed": 0,
        "full_11_row_remainders_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    lower = result["arithmetic_packet_audit"]["lower_Jacobian_roots"]
    mixed = result["arithmetic_packet_audit"]["selected_mixed_roots"]
    assert lower == {
        "reachable_nodes": 29963,
        "component_inputs": 7260,
        "quantitatively_bounded_component_inputs": 0,
        "division_nodes": 132,
        "all_component_inputs_quantitatively_bounded": False,
    }
    assert mixed["component_inputs"] == 1716
    assert mixed["quantitatively_bounded_component_inputs"] == 0
    for item in result["certificates"]:
        linear = item["row0_reference_linear_lower_slice"]
        assert linear["C_L_contribution"] == "41354"
        assert linear["C_B_contribution_for_this_linear_slice"] == "0"
        assert item["source_scale_recovery"]["finite_source_encoding_residual"] == "0"
        assert item["source_scale_recovery"]["source_ceiling_squared"] == (
            "1710153317"
        )
        assert item["row0_reference_linear_C_L_certified"]
        assert not item["row0_nonlinear_C_B_certified"]
        assert not item["full_11_row_remainder_closed"]
        assert not item["global_H7_differential_inequality_closed"]
        assert not item["nonlinear_lifespan_proved"]
    assert result == _load(ARTIFACT)


def test_omission_corruption_and_false_closure_policies_reject() -> None:
    metric, row0, global_h7 = _inputs()
    config = _load(CONFIG)

    result = run_quartic_row0_component_remainder_campaign(
        metric, {}, global_h7, config
    )
    assert result["status"] == "reject"
    assert "campaign prerequisite status mismatch" in result["errors"][0]

    corrupt = json.loads(json.dumps(row0))
    corrupt["upstream_sha256"]["metric_rows_tensor_dag"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_row0_component_remainder_campaign(
        metric, corrupt, global_h7, config
    )
    assert result["status"] == "reject"
    assert "row0-metric provenance mismatch" in result["errors"][0]

    for policy in (
        "row0_nonlinear_remainder_policy",
        "full_11_row_remainder_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_closure = dict(config)
        false_closure[policy] = "pass"
        result = run_quartic_row0_component_remainder_campaign(
            metric, row0, global_h7, false_closure
        )
        assert result["status"] == "reject"
