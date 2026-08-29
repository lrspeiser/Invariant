from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_prior_art_positioning as prior_art

ROOT = Path(__file__).resolve().parents[1]


def test_prior_art_search_positions_candidate_without_novelty_claim() -> None:
    receipt = prior_art.build_receipt(ROOT)
    assert receipt["decision"] == (
        "PRIOR_ART_POSITIONED_CLOSE_2026_NEIGHBOR_HISTORICAL_NOVELTY_UNRESOLVED"
    )
    assert set(receipt["completed_goal_evidence"]) == {"CP2.2", "CP2.3", "CP2.4"}
    assert set(receipt["blocked_goal_evidence"]) == {"CP2.5", "CP2.6"}
    assert receipt["claims"]["multi_database_search_complete_to_cutoff"] is True
    assert receipt["claims"]["exact_rewrite_identified"] is False
    assert receipt["claims"]["close_behavioral_equivalent_identified"] is True
    assert receipt["claims"]["corpus_absence_is_authoritative"] is False
    assert receipt["claims"]["historical_novelty_established"] is False


def test_penner_2026_is_retained_as_close_but_not_exact_neighbor() -> None:
    receipt = prior_art.build_receipt(ROOT)
    neighbor = receipt["closest_behavioral_neighbor"]
    assert neighbor["source_id"] == "PENNER_MODIFIED_GRAS_AQUAL_2026"
    assert neighbor["exact_rewrite"] is False
    assert "response slope" in neighbor["decisive_difference"]
    assert "two fixed log-radius averages" in neighbor["decisive_difference"]


def test_rewrite_family_combination_and_unresolved_labels_are_separate() -> None:
    config = prior_art.load_config(ROOT)
    labels = {row["classification"] for row in config["equation_comparison"]}
    assert "rewrite_of_known_transition_motif" in labels
    assert "new_combination_of_known_motifs" in labels
    assert "known_operator_family_structurally_unmatched_instantiation" in labels
    assert "structurally_unmatched_exact_combination_novelty_unresolved" in labels
    assert "known_method_not_candidate_novelty" in labels


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["candidate"].__setitem__("historical_novelty_claim", True),
            "novelty seal",
        ),
        (
            lambda value: value["database_search_audit"][0].__setitem__(
                "authoritative_absence_claim", True
            ),
            "overclaims",
        ),
        (
            lambda value: value["candidate_adjudication"].__setitem__(
                "historical_novelty_resolved", True
            ),
            "overclaims novelty",
        ),
        (
            lambda value: value["human_review"].__setitem__(
                "historical_novelty_sentence_authorized", True
            ),
            "human novelty-review",
        ),
    ],
)
def test_novelty_overclaims_fail_closed(mutation: object, match: str) -> None:
    config = copy.deepcopy(prior_art.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(prior_art.GravityClusterPriorArtError, match=match):
        prior_art.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / prior_art.OUTPUT_PATH).read_text(encoding="utf-8"))
    prior_art.validate_receipt(stored, ROOT)
    assert stored == prior_art.build_receipt(ROOT)
