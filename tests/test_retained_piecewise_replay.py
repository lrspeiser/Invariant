from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import retained_piecewise_replay as R
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_all_retained_live_piecewise_ideas_now_replay_exactly() -> None:
    receipt = R.build_receipt(ROOT)
    R.validate_receipt(receipt, ROOT)
    assert receipt["summary"] == {
        "admitted_by_current_executor": 9,
        "exact_primary_independent_agreements": 9,
        "extended_arithmetic_feature_counts": {
            "decimal_to_rational": 4,
            "exact_conditional": 1,
            "exact_floor_division": 1,
            "exact_modulo": 7,
            "exact_round_ties_to_even": 4,
        },
        "llm_self_assessed_origin_counts": {
            "cross_domain_synthesis": 1,
            "known_rewrite": 1,
            "uncertain": 7,
        },
        "resource_matched_controls": 9,
        "retained_piecewise_ideas": 9,
        "status": "PASS_RETAINED_PIECEWISE_REPLAY",
        "train_exact_holdout_failed": 1,
        "zero_holdout_loss_bounded_unknown": 0,
        "zero_holdout_loss_candidates": 1,
        "zero_train_loss_candidates": 2,
    }
    assert all(
        row["execution"]["primary_independent_exact_agreement"]
        and row["execution"]["resource_profile_exact_match"]
        for row in receipt["replays"]
    )


def test_replay_is_credential_free_and_makes_no_new_provider_calls() -> None:
    receipt = R.build_receipt(ROOT)
    encoded = json.dumps(receipt, sort_keys=True)
    assert receipt["chronology"] == {
        "new_provider_calls": 0,
        "replayed_after_live_generation": True,
        "source_outputs_are_credential_free": True,
        "uses_retained_sanitized_lineage_only": True,
    }
    assert "sk-ant-" not in encoded
    assert ".invariant.env" not in encoded


def test_stored_replay_receipt_reproduces_exactly() -> None:
    stored = json.loads((ROOT / R.OUTPUT_PATH).read_text(encoding="utf-8"))
    R.validate_receipt(stored, ROOT)
    assert stored == R.build_receipt(ROOT)


def test_resealed_replay_cannot_promote_execution_to_novelty() -> None:
    changed = copy.deepcopy(R.build_receipt(ROOT))
    changed["claim_boundary"]["llm_origin_assessment_establishes_novelty"] = True
    changed["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed.items() if key != "content_sha256"}
    )
    with pytest.raises(R.RetainedPiecewiseReplayError, match="policy"):
        R.validate_receipt(changed, ROOT)
