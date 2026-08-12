import copy
import json
from pathlib import Path

import pytest

import sigma_theory_compiler.quartic_tc2_mixed_third_jet_reranked_reduction_campaign as campaign_module
from sigma_theory_compiler.quartic_tc2_mixed_third_jet_basis_reduction_campaign import (
    _content_hash_matches,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_mixed_third_jet_reranked_reduction_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-mixed-third-jet-reranked-reduction-campaign"
    / "campaign.json"
)
SUPERVISOR_CONTENT_SHA256 = {
    "supervisor_state": "3b19200e7f2f7cf2f89f9f92bff4e44654f5895ec088072467da0f7867ae2ad1",
    "supervisor_status": "08482014b6012bb2a28654c517a92090c7f43b3f9b119a61ebee62d446bf26ea",
}


def _run_with_portable_stopped_boundary(config: dict[str, object]) -> dict[str, object]:
    """Exercise reranking without reading ignored supervisor runtime snapshots."""

    bindings = {item["label"]: item for item in config["stopped_boundary_evidence"]}
    claims = {
        "B7_closed": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "full_mixed_sector_closed": False,
        "full_tube_Sylvester_identity": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }
    common = {
        "state": "stopped",
        "stop_reason": "epoch_limit",
        "pid": None,
        "epochs_completed": 20,
        "chunks_advanced": 20,
        "next_offset": 1600,
        "remaining_mixed_triples": 10_700,
        "prior_resume_sha256": config["stable_predecessor_resume_tip_sha256"],
        "claims": claims,
    }
    controls = {
        "supervisor_state": {
            **common,
            "content_sha256": SUPERVISOR_CONTENT_SHA256["supervisor_state"],
            "epoch_checkpoint_content_sha256": bindings["checkpoint"][
                "content_sha256"
            ],
        },
        "supervisor_status": {
            **common,
            "alive": False,
            "content_sha256": SUPERVISOR_CONTENT_SHA256["supervisor_status"],
        },
    }
    original_loader = campaign_module._load_bound_json

    def portable_loader(root: Path, binding: dict[str, object]) -> dict[str, object]:
        for label, control in controls.items():
            if binding.get("path") == bindings[label]["path"]:
                if binding.get("content_sha256") != SUPERVISOR_CONTENT_SHA256[label]:
                    raise ValueError("bound artifact content hash mismatch")
                return copy.deepcopy(control)
        return original_loader(root, binding)

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(campaign_module, "_load_bound_json", portable_loader)
        return campaign_module.run_quartic_tc2_mixed_third_jet_reranked_reduction_campaign(
            ROOT, config
        )


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    return _run_with_portable_stopped_boundary(config)


def test_stopped_chain_reranking_is_exact_minimal_and_persisted(
    result: dict[str, object],
) -> None:
    assert _content_hash_matches(result)
    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == result
    assert result["status"] == (
        "pass_exact_stopped_chain_1600_rerank_447_obligations_"
        "no_inferred_passes_global_closure_fail_closed"
    )
    assert result["errors"] == []
    stable = result["stable_evidence"]
    assert stable["chunk_count"] == 25
    assert stable["mixed_prefix_records"] == 1600
    assert stable["mixed_candidate_evaluations"] == 19_200
    assert stable["mixed_candidate_solvable"] == 19_200
    assert stable["mixed_candidate_obstructed"] == 0
    assert stable["supervisor_stop_reason"] == "epoch_limit"
    reranking = result["exact_reranking"]
    assert reranking["active_direction_rank"] == 15
    assert reranking["symmetric_cubic_dimension"] == 680
    assert reranking["diagonal_evidence_rank"] == 16
    assert reranking["prior_576_prefix_rank"] == 105
    assert reranking["stable_1600_prefix_rank"] == 219
    assert reranking["stable_combined_evidence_rank"] == 233
    assert reranking["added_prefix_records"] == 1024
    assert reranking["rank_gain_over_prior_reduction"] == 113
    assert reranking["prior_reduced_obligation_count"] == 560
    assert reranking["reranked_obligation_count"] == 447
    assert reranking["obligations_removed_by_added_evidence"] == 113
    assert reranking["reranked_obligation_kind_counts"] == {
        "AAB": 77,
        "ABB": 81,
        "ABC": 289,
    }
    assert reranking["first_selector_index"] == 1634
    assert reranking["last_selector_index"] == 12_269
    assert reranking["completion_rank"] == 680
    assert reranking["drop_final_obligation_rank"] == 679


def test_reranked_selector_is_chained_and_claims_no_inferred_passes(
    result: dict[str, object],
) -> None:
    selector = result["reranked_obligation_selector"]
    obligations = selector["obligations"]
    assert selector["stable_remaining_mixed_triples"] == 10_700
    assert selector["exact_obligations"] == 447
    assert selector["candidate_evaluations_if_all_obligations_are_run"] == 5_364
    assert selector["unevaluated_obligations_counted_as_passes"] == 0
    assert selector["remaining_active_triples_counted_as_inferred_passes"] == 0
    assert len(obligations) == 447
    assert len({record["global_selector_index"] for record in obligations}) == 447
    assert min(record["global_selector_index"] for record in obligations) == 1634
    previous = result["exact_reranking"]["obligation_seed_sha256"]
    for record in obligations:
        assert record["previous_obligation_sha256"] == previous
        previous = record["obligation_sha256"]
    assert previous == result["exact_reranking"]["obligation_tip_sha256"]
    counts = result["counts"]
    assert counts["stable_mixed_triples_evaluated"] == 1600
    assert counts["stable_mixed_triples_remaining"] == 10_700
    assert counts["reranked_exact_obligations"] == 447
    assert counts["reranked_obligations_evaluated"] == 0
    assert counts["reranked_obligations_passed"] == 0
    assert counts["remaining_active_triples_inferred_passed"] == 0
    ledger = result["closure_ledger"]
    assert ledger["reranked_reduction_theorem_proved"] is True
    assert not any(
        value for key, value in ledger.items() if key != "reranked_reduction_theorem_proved"
    )


def test_stopped_boundary_hash_tamper_is_rejected() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(config)
    tampered["stopped_boundary_evidence"][-1]["content_sha256"] = "0" * 64
    result = _run_with_portable_stopped_boundary(tampered)
    assert _content_hash_matches(result)
    assert result["status"] == "reject"
    assert result["errors"] == ["bound artifact content hash mismatch"]
    assert result["counts"]["reranked_exact_obligations"] == 0
