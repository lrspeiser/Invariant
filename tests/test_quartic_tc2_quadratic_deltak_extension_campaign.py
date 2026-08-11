import copy
import json
from pathlib import Path

from sigma_theory_compiler.quartic_tc2_quadratic_deltak_extension_campaign import (
    _content_hash_matches,
    generic_quadratic_sylvester_jet_control,
    run_quartic_tc2_quadratic_deltak_extension_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "physics-language"
ARTIFACT = RUNS / "quartic-tc2-quadratic-deltak-extension-campaign" / "campaign.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_artifacts() -> list[dict]:
    paths = [
        RUNS / "quartic-tc2-second-atom-chunk-campaign" / "campaign.json",
        *[
            RUNS / f"quartic-tc2-second-atom-chunk{offset}-campaign" / "campaign.json"
            for offset in range(64, 704, 64)
        ],
        *sorted((RUNS / "quartic-tc2-continuous-service" / "chunks").glob("offset-*.json")),
    ]
    return [_load(path) for path in paths]


def _obligation_artifacts() -> list[dict]:
    paths = [
        RUNS / "quartic-tc2-excluded-obligation-chunk0-campaign" / "campaign.json",
        *sorted(
            (RUNS / "quartic-tc2-obligation-continuous-service" / "chunks").glob(
                "offset-*.json"
            )
        ),
    ]
    return [_load(path) for path in paths]


def _inputs() -> tuple:
    return (
        _load(RUNS / "quartic-tc2-variable-sylvester-campaign" / "campaign.json"),
        _load(RUNS / "quartic-tc2-excluded-pair-classification-campaign" / "campaign.json"),
        _load(RUNS / "quartic-tc2-continuous-service" / "checkpoint.json"),
        _load(RUNS / "quartic-tc2-obligation-continuous-service" / "checkpoint.json"),
        _load(RUNS / "quartic-tc2-ck1-p55-tube-envelope-campaign" / "campaign.json"),
        _canonical_artifacts(),
        _obligation_artifacts(),
        _load(
            ROOT
            / "configs"
            / "backgrounds"
            / "quartic_tc2_quadratic_deltak_extension_campaign.json"
        ),
    )


def test_generic_control_rejects_two_jet_promotion() -> None:
    passed, control = generic_quadratic_sylvester_jet_control()
    assert passed
    assert control["reference_jet_orders_closed"] == [0, 1, 2]
    assert not control["full_tube_Sylvester_identity_closed"]
    assert all(item["rejected"] for item in control["negative_controls"].values())


def test_campaign_closes_complete_quadratic_two_jet_only() -> None:
    result = run_quartic_tc2_quadratic_deltak_extension_campaign(*_inputs())
    assert result["status"] == (
        "pass_all_12_complete_reference_quadratic_deltaK_two_jets_full_identity_fail_closed"
    )
    assert _content_hash_matches(result)
    assert result["pair_partition"] == {
        "total_unordered_coordinate_pairs": 11781,
        "canonical_active_exact_pairs": 861,
        "excluded_exact_obligations": 2675,
        "entrywise_zero_chain_rule_pairs": 8245,
        "coverage_complete": True,
        "global_pair_index_set_sha256": result["pair_partition"][
            "global_pair_index_set_sha256"
        ],
    }
    assert result["counts"]["reference_two_jets_closed"] == 12
    assert result["counts"]["full_tube_Sylvester_identities"] == 0
    assert all(
        item["D2_deltaK_coordinate_linf_to_Frobenius_integer_ceiling"] > 0
        for item in result["quadratic_D2_envelopes"]
    )
    for certificate in result["certificates"]:
        ledger = certificate["closure_ledger"]
        assert ledger["complete_reference_deltaK_two_jet"]
        assert ledger["quadratic_D1_D2_bounds"]
        assert ledger["quadratic_Hermiticity"]
        assert ledger["quadratic_positivity_under_explicit_ell10_condition"]
        assert not ledger["full_tube_Sylvester_identity"]
        assert not ledger["variable_CK1_all_terms_closed"]
        assert not ledger["CK3_closed"]
        assert not ledger["TC2_closed"]
        assert not ledger["B7_closed"]
        assert not ledger["global_H7_closed"]
        assert not ledger["lifespan_proved"]


def test_checked_artifact_is_reproducible() -> None:
    assert _load(ARTIFACT) == run_quartic_tc2_quadratic_deltak_extension_campaign(
        *_inputs()
    )


def test_hash_tamper_and_false_policy_reject() -> None:
    inputs = list(_inputs())
    corrupt = copy.deepcopy(inputs[2])
    corrupt["content_sha256"] = "0" * 64
    inputs[2] = corrupt
    rejected = run_quartic_tc2_quadratic_deltak_extension_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["reference_two_jets_closed"] == 0

    inputs = list(_inputs())
    config = copy.deepcopy(inputs[-1])
    config["TC2_policy"] = "closed"
    inputs[-1] = config
    rejected = run_quartic_tc2_quadratic_deltak_extension_campaign(*inputs)
    assert rejected["status"] == "reject"
    assert rejected["counts"]["TC2_closures"] == 0
