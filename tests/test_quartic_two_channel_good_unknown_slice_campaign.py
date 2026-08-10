import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_two_channel_good_unknown_slice_campaign import (
    generic_two_channel_good_unknown_slice_control,
    run_quartic_two_channel_good_unknown_slice_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-rank-one-good-unknown-no-go-campaign" / "campaign.json",
    RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json",
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json",
    RUNS / "quartic-h7-resonant-remedy-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_two_channel_good_unknown_slice_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-two-channel-good-unknown-slice-campaign" / "campaign.json"
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


def test_two_channel_time_identity_ledger_and_negatives_are_exact() -> None:
    passed, control = generic_two_channel_good_unknown_slice_control()
    assert passed
    identity = control["time_differentiated_identity"]
    assert identity["full_four_entry_residual_zero"]
    assert identity["fixed_Fourier_LP_time_commutator"] == (
        "[partial_t,Delta_j]=0 exactly"
    )
    terms = control["induced_top_order_ledger"]["terms"]
    assert [item["id"] for item in terms] == [
        "TC1_low_factor_evolution",
        "TC2_physical_operator_on_correction",
        "TC3_operator_paraproduct_commutator",
        "TC4_fixed_LP_time_commutator",
        "TC5_nonlinear_substitution_remainder",
    ]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_full_s01_slice_closes_but_induced_terms_and_global_h7_stay_open() -> None:
    result = run_quartic_two_channel_good_unknown_slice_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_exact_two_channel_s01_slice_identities_"
        "induced_commutators_global_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "two_channel_full_s01_slice_identities_proved": 12,
        "four_entry_cancellations_proved": 12,
        "fixed_LP_time_commutators_closed": 12,
        "remaining_induced_term_ledgers_materialized": 12,
        "all_induced_term_bounds_closed": 0,
        "full_high_atom_families_closed": 0,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    first = result["certificates"][0]
    assert first["principal_slice_identity"]["after_J_s01_residual_zero"]
    assert first["principal_slice_identity"]["all_four_nonzero_entries_cancelled"]
    assert first["modified_state"]["built_only_from_actual_state_variables"]
    assert not first["induced_term_closure"]["all_induced_terms_closed"]
    assert first["connection_to_B7_global_H7"][
        "representative_full_s01_H01_slice_removed_from_B7"
    ]
    assert not first["connection_to_B7_global_H7"]["B7_fully_replaced"]
    assert result == _load(ARTIFACT)


def test_kinematic_tamper_omitted_derivative_and_false_promotions_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[1]))
    corrupt["generic_component_jacobian_contract_control"][
        "kinematic_evolution_rows"
    ]["residuals"]["22,11"] = "1"
    _rehash(corrupt)
    relinked_no_go = json.loads(json.dumps(inputs[0]))
    relinked_no_go["upstream_sha256"]["component_J_contract"] = corrupt[
        "content_sha256"
    ]
    _rehash(relinked_no_go)
    result = run_quartic_two_channel_good_unknown_slice_campaign(
        relinked_no_go, corrupt, *inputs[2:], config
    )
    assert result["status"] == "reject"
    assert "kinematic state-to-jet identity mismatch" in result["errors"][0]

    omitted_derivative = dict(config)
    omitted_derivative["low_derivative"] = "v0"
    result = run_quartic_two_channel_good_unknown_slice_campaign(
        *inputs, omitted_derivative
    )
    assert result["status"] == "reject"
    assert "unsupported two-channel contract" in result["errors"][0]

    for policy in (
        "induced_commutator_policy",
        "global_H7_policy",
        "lifespan_policy",
    ):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_two_channel_good_unknown_slice_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
