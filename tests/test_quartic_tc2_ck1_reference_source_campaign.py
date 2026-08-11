import copy
import hashlib
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_ck1_reference_source_campaign import (
    _content_hash_matches,
    _reference_ck1_component_packet,
    generic_ck1_reference_source_control,
    run_quartic_tc2_ck1_reference_source_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
SERVICE = RUNS / "quartic-tc2-obligation-continuous-service"
ARTIFACT = RUNS / "quartic-tc2-ck1-reference-source-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple:
    checkpoint = _load(SERVICE / "checkpoint.json")
    final_path = SERVICE / checkpoint["current_artifact_path"]
    return (
        _load(RUNS / "quartic-tc2-full-sylvester-reference-campaign" / "campaign.json"),
        _load(RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"),
        _load(RUNS / "quartic-two-channel-induced-operator-campaign" / "campaign.json"),
        _load(RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json"),
        _load(RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json"),
        _load(RUNS / "quartic-h7-paracomposition-topology-campaign" / "campaign.json"),
        checkpoint,
        _load(final_path),
        hashlib.sha256(final_path.read_bytes()).hexdigest(),
        _load(
            ROOT
            / "configs"
            / "backgrounds"
            / "quartic_tc2_ck1_reference_source_campaign.json"
        ),
    )


def test_generic_ck1_reference_source_control_and_exact_packet() -> None:
    passed, control = generic_ck1_reference_source_control()
    assert passed
    assert all(item["rejected"] for item in control["negative_controls"].values())
    packet = _reference_ck1_component_packet()
    assert packet["row_norm_sum"] == 3
    assert packet["packet"]["deltaK_basis_Frobenius_square"] == "1253060/9"
    assert [
        item["e10_EvT_P55k_entries"] for item in packet["packet"]["directions"]
    ] == [
        [{"row": 0, "column": 54, "value": "1"}],
        [{"row": 0, "column": 21, "value": "1"}],
        [{"row": 0, "column": 32, "value": "1"}],
    ]
    assert [
        item["product_nonzero_entries"] for item in packet["packet"]["directions"]
    ] == [24, 24, 24]


def test_real_campaign_closes_only_bounded_reference_slice() -> None:
    result = run_quartic_tc2_ck1_reference_source_campaign(*_inputs())
    assert result["status"] == (
        "pass_all_12_CK1_reference_principal_and_source_topologies_"
        "TC2_global_fail_closed"
    )
    assert _content_hash_matches(result)
    assert result["counts"]["selected"] == 12
    assert result["counts"]["excluded_obligations_verified_complete"] == 2675
    assert result["counts"]["CK1_reference_principal_packets_closed"] == 12
    assert result["counts"]["CK1_reference_F10_source_topologies_closed"] == 12
    assert result["counts"]["TC1_Q_source_topologies_closed"] == 12
    assert result["counts"]["TC2_closures"] == 0
    for certificate in result["certificates"]:
        ledger = certificate["closure_ledger"]
        assert ledger["CK1_reference_principal_part_closed"]
        assert ledger["CK1_reference_F10_source_remainder_closed"]
        assert ledger["TC1_Q_contracted_reference_source_remainder_closed"]
        assert not ledger["variable_CK1_all_terms_closed"]
        assert not ledger["TC2_closed"]
        assert not ledger["B7_closed"]
        assert not ledger["global_H7_closed"]
        assert not ledger["lifespan_proved"]


def test_checked_artifact_is_exactly_reproducible() -> None:
    result = run_quartic_tc2_ck1_reference_source_campaign(*_inputs())
    artifact = _load(ARTIFACT)
    assert artifact == result
    assert _content_hash_matches(artifact)


def test_tampered_completion_tip_and_false_promotion_reject() -> None:
    inputs = list(_inputs())
    tampered_checkpoint = copy.deepcopy(inputs[6])
    tampered_checkpoint["prior_resume_sha256"] = "0" * 64
    inputs[6] = tampered_checkpoint
    rejected = run_quartic_tc2_ck1_reference_source_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["selected"] == 0

    inputs = list(_inputs())
    config = copy.deepcopy(inputs[-1])
    config["declare_full_CK1_closed"] = True
    inputs[-1] = config
    rejected = run_quartic_tc2_ck1_reference_source_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["TC2_closures"] == 0
