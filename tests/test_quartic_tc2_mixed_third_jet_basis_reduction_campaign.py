import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_mixed_third_jet_basis_reduction_campaign import (
    _content_hash_matches,
    run_quartic_tc2_mixed_third_jet_basis_reduction_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_mixed_third_jet_basis_reduction_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-mixed-third-jet-basis-reduction-campaign"
    / "campaign.json"
)


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return run_quartic_tc2_mixed_third_jet_basis_reduction_campaign(ROOT, config)


def test_exact_basis_reduction_is_complete_minimal_and_hash_bound(
    result: dict[str, object],
) -> None:
    assert _content_hash_matches(result)
    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == result
    assert result["status"] == (
        "pass_exact_15_direction_basis_reduction_560_obligations_"
        "no_inferred_passes_global_closure_fail_closed"
    )
    assert result["errors"] == []
    reduction = result["exact_active_direction_reduction"]
    assert reduction["ambient_coordinate_count"] == 16
    assert reduction["active_direction_count"] == 41
    assert reduction["active_direction_rank"] == 15
    assert reduction["basis_active_positions"] == [
        0,
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        13,
        15,
        21,
        27,
        32,
        35,
        40,
    ]
    assert reduction["direction_coordinate_support_counts"] == {"1": 38, "3": 3}
    assert reduction["symmetric_cubic_dimension"] == 680
    assert reduction["diagonal_evidence_functional_rank"] == 16
    assert reduction["stable_prefix_functional_rank"] == 105
    assert reduction["combined_evidence_functional_rank"] == 120
    assert reduction["reduced_obligation_count"] == 560
    assert reduction["reduced_obligation_kind_counts"] == {
        "AAB": 90,
        "ABB": 96,
        "ABC": 374,
    }
    assert reduction["reduced_obligation_first_selector_index"] == 650
    assert reduction["reduced_obligation_last_selector_index"] == 12_269
    assert reduction["completion_rank"] == 680
    assert reduction["drop_final_obligation_rank"] == 679


def test_reduced_selector_declares_no_inferred_passes_and_keeps_closure_false(
    result: dict[str, object],
) -> None:
    counts = result["counts"]
    assert counts["stable_mixed_triples_evaluated"] == 576
    assert counts["stable_mixed_triples_remaining"] == 11_724
    assert counts["reduced_exact_obligations"] == 560
    assert counts["reduced_obligations_evaluated"] == 0
    assert counts["reduced_obligations_passed"] == 0
    assert counts["remaining_active_triples_inferred_passed"] == 0
    selector = result["reduced_obligation_selector"]
    assert selector["candidate_evaluations_if_all_obligations_are_run"] == 6_720
    assert selector["unevaluated_obligations_counted_as_passes"] == 0
    assert selector["remaining_active_triples_counted_as_inferred_passes"] == 0
    obligations = selector["obligations"]
    assert len(obligations) == 560
    assert len({item["global_selector_index"] for item in obligations}) == 560
    assert min(item["global_selector_index"] for item in obligations) == 650
    previous = result["exact_active_direction_reduction"][
        "reduced_obligation_seed_sha256"
    ]
    for item in obligations:
        assert item["previous_obligation_sha256"] == previous
        previous = item["obligation_sha256"]
    assert previous == result["exact_active_direction_reduction"][
        "reduced_obligation_tip_sha256"
    ]
    ledger = result["closure_ledger"]
    assert ledger["basis_reduction_theorem_proved"] is True
    assert not any(value for key, value in ledger.items() if key != "basis_reduction_theorem_proved")


def test_bound_predecessor_tamper_is_rejected() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(config)
    tampered["stable_chunk_evidence"][-1]["file_sha256"] = "0" * 64
    result = run_quartic_tc2_mixed_third_jet_basis_reduction_campaign(ROOT, tampered)
    assert _content_hash_matches(result)
    assert result["status"] == "reject"
    assert result["errors"] == ["bound artifact file hash mismatch"]
    assert result["counts"]["reduced_exact_obligations"] == 0
