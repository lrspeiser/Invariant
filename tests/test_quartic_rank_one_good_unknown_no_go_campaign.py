import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_rank_one_good_unknown_no_go_campaign import (
    generic_rank_one_good_unknown_no_go_control,
    run_quartic_rank_one_good_unknown_no_go_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
PATHS = (
    RUNS / "quartic-high-atom-d2-good-unknown-campaign" / "campaign.json",
    RUNS / "quartic-component-jacobian-contract-campaign" / "campaign.json",
    RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json",
    RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json",
    RUNS / "quartic-h7-resonant-remedy-campaign" / "campaign.json",
)
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_rank_one_good_unknown_no_go_campaign.json"
)
ARTIFACT = (
    RUNS / "quartic-rank-one-good-unknown-no-go-campaign" / "campaign.json"
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


def test_rank_one_minor_no_go_factorization_and_negatives_are_exact() -> None:
    passed, control = generic_rank_one_good_unknown_no_go_control()
    assert passed
    required = control["actual_required_correction"]
    assert required["rank_for_alpha_nonzero"] == 2
    assert required["decisive_minor_rows_0_10_columns_10_7"] == "4*alpha**2"
    assert required["generic_rank_one_same_minor"] == "0"
    factorization = control["minimal_algebraic_factorization"]
    assert factorization["exactly_cancels_source_slice"]
    assert factorization["exactly_cancels_after_J_s01"]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_all_candidates_have_rank_one_no_go_but_no_b7_promotion() -> None:
    result = run_quartic_rank_one_good_unknown_no_go_campaign(
        *_inputs(), _load(CONFIG)
    )
    assert result["status"] == (
        "pass_all_12_exact_single_channel_good_unknown_no_gos_"
        "rank_two_targets_identified_global_H7_fail_closed"
    )
    assert result["counts"] == {
        "selected": 12,
        "single_channel_rank_one_no_gos_proved": 12,
        "minimal_rank_two_algebraic_targets_identified": 12,
        "rank_two_modified_unknown_identities_proved": 0,
        "B7_branches_closed": 0,
        "global_H7_closures": 0,
        "lifespans_proved": 0,
        "rejected": 0,
    }
    assert result["s01_injection_slice"]["matrix_identity"] == "J_s01=i*xi1*I_11"
    first = result["certificates"][0]
    assert first["single_channel_no_go"]["proved"]
    assert first["minimal_algebraic_target"]["full_slice_residual_zero"]
    assert not first["minimal_algebraic_target"][
        "actual_modified_unknown_lift_proved"
    ]
    assert not first["connection_to_B7_global_H7"]["B7_fully_replaced"]
    assert result == _load(ARTIFACT)


def test_tampered_source_J_and_false_promotion_contracts_reject() -> None:
    inputs = _inputs()
    config = _load(CONFIG)

    corrupt = json.loads(json.dumps(inputs[0]))
    corrupt["actual_reference_audit"]["nonzero_D2_entries_in_s01_block"][0][
        "value"
    ] = "-alpha"
    _rehash(corrupt)
    result = run_quartic_rank_one_good_unknown_no_go_campaign(
        corrupt, *inputs[1:], config
    )
    assert result["status"] == "reject"
    assert "D2 source slice mismatch" in result["errors"][0]

    corrupt_j = json.loads(json.dumps(inputs[1]))
    corrupt_j["generic_component_jacobian_contract_control"][
        "principal_jet_injection"
    ]["entries"][0]["coefficient"] = "I*xi2"
    _rehash(corrupt_j)
    result = run_quartic_rank_one_good_unknown_no_go_campaign(
        inputs[0], corrupt_j, *inputs[2:], config
    )
    assert result["status"] == "reject"
    assert "s01 injection slice mismatch" in result["errors"][0]

    for policy in ("global_H7_policy", "lifespan_policy"):
        false_promotion = dict(config)
        false_promotion[policy] = "pass"
        result = run_quartic_rank_one_good_unknown_no_go_campaign(
            *inputs, false_promotion
        )
        assert result["status"] == "reject"
