import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_fourth_jet_range_obligation_campaign import (
    QuarticTC2FourthJetRangeObligationCampaignError,
    _partition_name,
    build_fourth_jet_range_obligation_campaign,
)
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_continuation_service import (
    _body,
    _hash_matches,
    _with_hash,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_fourth_jet_range_obligation_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-fourth-jet-range-obligation-campaign"
    / "campaign.json"
)


def test_committed_fourth_selector_reexecutes_exactly() -> None:
    expected = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    actual = build_fourth_jet_range_obligation_campaign(ROOT, CONFIG)
    assert actual == expected
    assert _hash_matches(actual)
    assert actual["counts"] == {
        "active_directions": 15,
        "candidate_fourth_jet_obligations": 36_720,
        "candidates": 12,
        "fourth_jet_obligations_evaluated": 0,
        "fourth_jet_obligations_inferred_passed": 0,
        "fourth_jet_obligations_passed": 0,
        "fourth_selector_records": 3_060,
        "negative_controls": 3,
    }


def test_selector_is_complete_unique_and_chained() -> None:
    artifact = build_fourth_jet_range_obligation_campaign(ROOT, CONFIG)
    selector = artifact["selector"]
    records = selector["records"]
    assert len(records) == 3_060
    assert len({tuple(record["multi_index"]) for record in records}) == 3_060
    assert Counter(record["multiplicity_partition"] for record in records) == {
        "AAAA": 15,
        "AAAB": 210,
        "AABB": 105,
        "AABC": 1365,
        "ABCD": 1365,
    }
    prior = selector["seed_sha256"]
    for offset, record in enumerate(records):
        assert record["selector_offset"] == offset
        assert record["prior_record_sha256"] == prior
        prior = record["record_sha256"]
    assert prior == selector["tip_sha256"]
    assert _partition_name((0, 0, 1, 2)) == "AABC"


def test_negative_controls_block_finite_jet_and_subset_inference() -> None:
    artifact = build_fourth_jet_range_obligation_campaign(ROOT, CONFIG)
    controls = artifact["negative_controls"]
    assert all(control["rejected"] for control in controls.values())
    assert controls["hidden_fourth_order_cokernel_perturbation"][
        "derivatives_at_zero_orders_0_through_3"
    ] == ["0", "0", "0", "0"]
    assert controls["single_omitted_selector_entry"][
        "omitted_entry_witnesses"
    ] == 3_060
    assert artifact["exact_remainder_constants"][
        "coordinatewise_normalized_derivative_sum_factor"
    ] == "16875/8"
    claims = artifact["claims"]
    assert claims["reference_mixed_third_jet_closed"] is True
    assert claims["fourth_jet_minimal_selector_constructed"] is True
    assert sum(claims.values()) == 2


def test_bound_input_tamper_is_rejected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(config)
    tampered["completed_third_jet_checkpoint"]["file_sha256"] = "0" * 64
    tampered = _with_hash(_body(tampered))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        QuarticTC2FourthJetRangeObligationCampaignError,
        match="bound input mismatch",
    ):
        build_fourth_jet_range_obligation_campaign(ROOT, path)


def test_selector_count_policy_tamper_is_rejected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["expected_fourth_selector_count"] = 3_059
    tampered = _with_hash(_body(config))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(
        QuarticTC2FourthJetRangeObligationCampaignError,
        match="unsupported campaign contract",
    ):
        build_fourth_jet_range_obligation_campaign(ROOT, path)
