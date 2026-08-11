import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_ck1_variable_commutator_campaign import (
    _content_hash_matches,
    _variable_ck1_commutator_packet,
    generic_ck1_variable_commutator_control,
    run_quartic_tc2_ck1_variable_commutator_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ARTIFACT = (
    RUNS / "quartic-tc2-ck1-variable-commutator-campaign" / "campaign.json"
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs() -> tuple:
    return (
        _load(RUNS / "quartic-tc2-ck1-reference-source-campaign" / "campaign.json"),
        _load(RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"),
        _load(RUNS / "quartic-full-source-jacobian-arithmetic-campaign" / "campaign.json"),
        _load(RUNS / "quartic-solved-source-c9-extension-campaign" / "campaign.json"),
        _load(RUNS / "quartic-h7-paracomposition-topology-campaign" / "campaign.json"),
        _load(RUNS / "quartic-reference-equilibrium-campaign" / "campaign.json"),
        _load(
            ROOT
            / "configs"
            / "backgrounds"
            / "quartic_tc2_ck1_variable_commutator_campaign.json"
        ),
    )


def test_generic_product_rule_and_exact_component_packet() -> None:
    passed, control = generic_ck1_variable_commutator_control()
    assert passed
    assert control["residual"] == "0"
    assert all(item["rejected"] for item in control["negative_controls"].values())
    packet = _variable_ck1_commutator_packet()
    assert _content_hash_matches(packet)
    assert packet["counts"]["coordinate_atoms"] == 153
    assert packet["counts"]["atom_direction_packets"] == 459
    assert packet["counts"]["nonzero_deltaK_A_atoms"] == 41
    assert packet["counts"]["nonzero_DP55_scalar_row_packets"] == 123
    assert packet["all_DP55_coordinate_derivatives_linear_in_a10_and_c20_absent"]
    assert packet["reference_P55_scalar_rows"] == [
        [{"row": 0, "column": 54, "value": "1"}],
        [{"row": 0, "column": 21, "value": "1"}],
        [{"row": 0, "column": 32, "value": "1"}],
    ]
    assert all(
        item["rejected"] for item in packet["exact_negative_controls"].values()
    )
    assert packet["exact_negative_controls"]["drop_spatial_direction_three"][
        "missing_reference_row"
    ] == [{"row": 0, "column": 32, "value": "1"}]


def test_campaign_closes_only_reference_and_source_slices() -> None:
    result = run_quartic_tc2_ck1_variable_commutator_campaign(*_inputs())
    assert result["status"] == (
        "pass_all_12_reference_variable_CK1_P55_source_commutators_"
        "tube_P55_fail_closed"
    )
    assert _content_hash_matches(result)
    assert result["counts"]["selected"] == 12
    assert result["counts"]["exact_atom_direction_packets"] == 459
    assert result["counts"]["reference_P55_commutator_slices_closed"] == 12
    assert result["counts"]["affine_deltaK_source_slices_closed"] == 12
    assert result["counts"]["tube_uniform_P55_commutator_closures"] == 0
    assert result["counts"]["variable_CK1_closures"] == 0
    for certificate in result["certificates"]:
        ledger = certificate["closure_ledger"]
        assert ledger["reference_DdeltaK_times_P55_closed"]
        assert ledger["reference_deltaK0_times_DP55_closed"]
        assert ledger["affine_deltaK_source_commutators_closed_on_tube"]
        assert not ledger["tube_uniform_variable_P55_commutators_closed"]
        assert not ledger["variable_CK1_all_terms_closed"]
        assert not ledger["TC2_closed"]
        assert not ledger["B7_closed"]
        assert not ledger["global_H7_closed"]
        assert not ledger["lifespan_proved"]
        assert certificate["first_remaining_blocker"]["tensor"] == (
            "D2P55(Y) for all three spatial pencils"
        )


def test_checked_artifact_is_exactly_reproducible() -> None:
    result = run_quartic_tc2_ck1_variable_commutator_campaign(*_inputs())
    assert _load(ARTIFACT) == result


def test_tampered_upstream_and_false_promotion_reject() -> None:
    inputs = list(_inputs())
    corrupt = copy.deepcopy(inputs[0])
    corrupt["content_sha256"] = "0" * 64
    inputs[0] = corrupt
    rejected = run_quartic_tc2_ck1_variable_commutator_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["selected"] == 0

    inputs = list(_inputs())
    config = copy.deepcopy(inputs[-1])
    config["declare_variable_CK1_closed"] = True
    inputs[-1] = config
    rejected = run_quartic_tc2_ck1_variable_commutator_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["TC2_closures"] == 0
