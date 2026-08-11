import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_ck1_p55_tube_envelope_campaign import (
    _content_hash_matches,
    generic_p55_coordinate_chain_rule_control,
    run_quartic_tc2_ck1_p55_tube_envelope_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ARTIFACT = RUNS / "quartic-tc2-ck1-p55-tube-envelope-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple:
    return (
        _load(RUNS / "quartic-tc2-ck1-variable-commutator-campaign" / "campaign.json"),
        _load(RUNS / "quartic-evolution-symbol-campaign" / "campaign.json"),
        _load(RUNS / "quartic-coordinate-jet-tube-campaign" / "campaign.json"),
        _load(RUNS / "quartic-h7-paracomposition-topology-campaign" / "campaign.json"),
        _load(ROOT / "configs" / "backgrounds" / "quartic_tc2_ck1_p55_tube_envelope_campaign.json"),
    )


def test_generic_chain_rule_requires_both_branches() -> None:
    passed, control = generic_p55_coordinate_chain_rule_control()
    assert passed
    assert all(item["rejected"] for item in control["negative_controls"].values())
    assert control["operator_bound"] == "C_coordinate_D2P=P2*J1^2+P1*J2"


def test_campaign_closes_exact_bounded_p55_gate_only() -> None:
    result = run_quartic_tc2_ck1_p55_tube_envelope_campaign(*_inputs())
    assert result["status"] == (
        "pass_all_12_affine_deltaK_tube_uniform_D2P55_envelopes_global_fail_closed"
    )
    assert _content_hash_matches(result)
    packet = result["common_P55_tube_packet"]
    assert packet["coordinate_map_envelopes"]["1"]["integer_ceiling"] == 481
    assert packet["coordinate_map_envelopes"]["2"]["integer_ceiling"] == 26860
    chain = packet["chain_rule"]
    assert chain["intrinsic_D2P55_DJ_DJ_contribution"] == "4842388274971"
    assert chain["coordinate_D2J_pushforward_contribution"] == "2478425920"
    assert chain["coordinate_DP55_integer_ceiling"] == "44382832"
    assert chain["coordinate_D2P55_integer_ceiling"] == "4844866700891"
    assert all(item["rejected"] for item in packet["exact_negative_controls"].values())
    assert result["counts"]["tube_uniform_D2P55_envelopes_closed"] == 12
    assert result["counts"]["affine_deltaK_tube_P55_slices_closed"] == 12
    assert result["counts"]["full_variable_CK1_closures"] == 0
    for certificate in result["certificates"]:
        assert len(certificate["provenance"]["H7_topology_certificate_sha256"]) == 64
        ledger = certificate["closure_ledger"]
        assert ledger["three_pencil_coordinate_D2P55_tube_envelope_closed"]
        assert ledger["Dr_k_difference_control_closed"]
        assert ledger["affine_deltaK_tube_P55_commutators_closed"]
        assert not ledger["non_affine_deltaK_extension_closed"]
        assert not ledger["variable_CK1_all_terms_closed"]
        assert not ledger["TC2_closed"]
        assert not ledger["B7_closed"]
        assert not ledger["global_H7_closed"]
        assert not ledger["lifespan_proved"]


def test_checked_artifact_is_exactly_reproducible() -> None:
    result = run_quartic_tc2_ck1_p55_tube_envelope_campaign(*_inputs())
    assert _load(ARTIFACT) == result


def test_tampering_and_false_promotion_reject() -> None:
    inputs = list(_inputs())
    corrupt = copy.deepcopy(inputs[1])
    corrupt["content_sha256"] = "0" * 64
    inputs[1] = corrupt
    rejected = run_quartic_tc2_ck1_p55_tube_envelope_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["selected"] == 0

    inputs = list(_inputs())
    config = copy.deepcopy(inputs[-1])
    config["declare_full_variable_CK1_closed"] = True
    inputs[-1] = config
    rejected = run_quartic_tc2_ck1_p55_tube_envelope_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["TC2_closures"] == 0
