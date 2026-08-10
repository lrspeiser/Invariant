import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_two_channel_induced_operator_campaign import (
    generic_two_channel_induced_operator_control,
    run_quartic_two_channel_induced_operator_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-two-channel-good-unknown-slice-campaign" / "campaign.json",
    RUNS / "quartic-first-order-reduction-campaign" / "campaign.json",
    RUNS / "quartic-dyadic-localization-campaign" / "campaign.json",
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_two_channel_induced_operator_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-two-channel-induced-operator-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> list[dict]:
    return [_load(path) for path in PATHS]


def _rehash(campaign: dict) -> None:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    campaign["content_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def test_reference_constants_taylor_packet_and_negatives_are_exact() -> None:
    passed, control = generic_two_channel_induced_operator_control()
    assert passed
    assert control["H7_to_W2infinity_constant"] == "sqrt(5)/(64*sqrt(pi))"
    assert control["TC1_principal_shell_bound"]["reference_principal_part_closed"]
    assert control["TC3_spatial_product_rule_shell_bound"][
        "reference_principal_part_closed"
    ]
    assert control["TC5_exact_Taylor_packet"]["integral_weight"] == "1/2"
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_actual_packets_partial_bounds_and_fail_closed_scope() -> None:
    result = run_quartic_two_channel_induced_operator_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_exact_P55_Q_TC_packets_reference_partial_bounds_"
        "global_H7_fail_closed"
    )
    assert result["common_reference_P55_Q_packet"][
        "nonzero_counts_P55k_Ev_Q"
    ] == [4, 5, 5]
    assert result["common_reference_P55_Q_packet"][
        "nonzero_counts_Q_EvT_P55k"
    ] == [6, 9, 9]
    assert result["counts"] == {
        "selected": 12,
        "actual_P55_Q_component_packets": 12,
        "TC1_reference_principal_shell_bounds_closed": 12,
        "TC2_component_packets_materialized": 12,
        "TC2_full_bounds_closed": 0,
        "TC3_reference_shell_bounds_closed": 12,
        "TC5_pointwise_Taylor_bounds_closed": 12,
        "TC5_H7_bounds_closed": 0,
        "all_induced_terms_closed": 0,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["actual_P55_on_embedded_Q"]["total_nonzero_entries"] == 14
    assert first["TC1_low_factor_evolution"][
        "reference_principal_shell_bound_closed"
    ]
    assert not first["TC2_physical_operator_on_correction"]["full_TC2_closed"]
    assert first["TC3_operator_paraproduct_commutator"][
        "reference_shell_bound_closed"
    ]
    assert first["TC5_nonlinear_substitution_remainder"]["pointwise_bound_closed"]
    assert not first["closure_ledger"]["all_induced_terms_closed"]
    assert result == _load(ARTIFACT)


def test_hash_tamper_reference_policy_and_false_promotions_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["upstream_sha256"]["full_source_jacobian"] = "0" * 64
    _rehash(corrupt)
    result = run_quartic_two_channel_induced_operator_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "upstream provenance mismatch" in result["errors"][0]

    wrong_reference = dict(config)
    wrong_reference["reference_packet"] = "arbitrary_variable_jet"
    result = run_quartic_two_channel_induced_operator_campaign(
        *inputs, wrong_reference
    )
    assert result["status"] == "reject"
    assert "unsupported induced-operator contract" in result["errors"][0]

    for policy in (
        "variable_coefficient_extension_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_two_channel_induced_operator_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
