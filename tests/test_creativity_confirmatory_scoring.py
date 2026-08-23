from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import creativity_ablation as ablation
from sigma_theory_compiler import creativity_confirmatory_generation as confirmatory
from sigma_theory_compiler import creativity_confirmatory_scoring as C
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _contains_key(value: object, key: str) -> bool:
    if isinstance(value, dict):
        return key in value or any(_contains_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, key) for item in value)
    return False


def _completed_draft(name: str, affiliation: str, *, score: int = 3) -> dict:
    draft = C.make_review_template(ROOT)
    draft["reviewer"] = {
        "affiliation": affiliation,
        "conflict_disclosure": "No conflict known.",
        "full_name": name,
        "reviewed_utc": "2026-08-23T12:00:00Z",
        "reviewer_is_generation_operator": False,
    }
    draft["attestations"] = {key: True for key in C._ATTESTATIONS}
    for row in draft["ratings"]:
        for axis in draft["rating_policy"]["axes"]:
            row[axis] = score
    return draft


def _idea(
    behavior: str,
    proof: str,
    *,
    origin: str = "uncertain",
    initial: str = "untested",
    reused: bool = False,
) -> dict:
    return {
        "behavior_sha256": behavior,
        "human_reviews": [
            {
                "reviewer_id": "reviewer.one",
                "coherence": 3,
                "nontriviality": 3,
                "followup_value": 3,
            },
            {
                "reviewer_id": "reviewer.two",
                "coherence": 4,
                "nontriviality": 4,
                "followup_value": 4,
            },
        ],
        "initial_check_status": initial,
        "later_used_as_parent": reused,
        "llm_origin_assessment": origin,
        "prior_art_classification": "review_pending",
        "proof_mechanism_sha256": proof,
        "representation": "recurrence",
        "source_domains": ["algebra", "combinatorics"],
    }


def _record(task: str, arm: str, ideas: list[dict]) -> dict:
    return {
        "arm": arm,
        "blinded_output_id": f"output.{task}.{arm}",
        "ideas": ideas,
        "resource_budget": {
            "calls": 2,
            "grammar_depth": 8,
            "tokens": 200_000,
            "verifier_invocations": 5,
            "wall_clock_milliseconds": 240_000,
        },
        "task_id": task,
        "tokens_used": 10_000,
        "typed_usable_ideas": len(ideas),
    }


def test_live_template_covers_every_branch_without_arm_or_target_mapping() -> None:
    config = C.load_config(ROOT)
    template = C.make_review_template(ROOT, config)
    packet = json.loads((ROOT / config["review_packet"]["path"]).read_text(encoding="utf-8"))
    expected = sum(len(output["branches"]) for output in packet["blinded_outputs"])
    assert expected == 353
    assert len(template["ratings"]) == expected
    assert len(
        {
            (row["blinded_output_id"], row["task_id"], row["branch_id"])
            for row in template["ratings"]
        }
    ) == expected
    assert not _contains_key(template, "arm")
    assert not _contains_key(template, "target")


def test_seal_requires_specific_human_complete_coverage_and_affirmative_attestations() -> None:
    draft = _completed_draft("Named Reviewer", "Independent Mathematics Lab")
    draft["reviewer"]["full_name"] = "anonymous"
    with pytest.raises(C.ConfirmatoryScoringError, match="specifically named"):
        C.seal_review(draft, ROOT)

    draft = _completed_draft("Ada Lovelace", "Independent Mathematics Lab")
    draft["ratings"].pop()
    with pytest.raises(C.ConfirmatoryScoringError, match="coverage"):
        C.seal_review(draft, ROOT)

    draft = _completed_draft("Ada Lovelace", "Independent Mathematics Lab")
    draft["attestations"]["withheld_targets_not_accessed"] = False
    with pytest.raises(C.ConfirmatoryScoringError, match="attestations"):
        C.seal_review(draft, ROOT)

    draft = _completed_draft("Ada Lovelace", "Independent Mathematics Lab")
    draft["reviewer"]["reviewer_is_generation_operator"] = True
    with pytest.raises(C.ConfirmatoryScoringError, match="generation operator"):
        C.seal_review(draft, ROOT)


def test_review_pair_requires_two_distinct_named_reviewers() -> None:
    left = C.seal_review(_completed_draft("Ada Lovelace", "Mathematics Lab A"), ROOT)
    right = C.seal_review(_completed_draft("Emmy Noether", "Mathematics Lab B"), ROOT)
    C.validate_review_pair([left, right], ROOT)
    with pytest.raises(C.ConfirmatoryScoringError, match="not distinct"):
        C.validate_review_pair([left, copy.deepcopy(left)], ROOT)


def test_invalid_reviews_fail_before_private_unblinding_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    invalid = C.make_review_template(ROOT)
    paths = []
    for index in range(2):
        path = tmp_path / f"invalid-{index}.json"
        path.write_text(json.dumps(invalid), encoding="utf-8")
        paths.append(path)

    def forbidden_private_read(_path: Path):
        raise AssertionError("private journal opened before review validation")

    monkeypatch.setattr(C.DurableAttemptJournal, "load", forbidden_private_read)
    with pytest.raises(C.ConfirmatoryScoringError, match="sealed review form keys"):
        C.score_from_review_paths(ROOT, paths)


def test_behavior_novelty_and_proof_mechanism_diversity_are_separate() -> None:
    policy = C.load_config(ROOT)["review_policy"]
    records = [
        _record(
            "task.01",
            "baseline",
            [
                _idea("behavior.same", "proof.route.one"),
                _idea("behavior.same", "proof.route.two", initial="blocked", reused=True),
            ],
        ),
        _record(
            "task.01",
            "full_creativity_first",
            [_idea("behavior.treatment", "proof.route.treatment")],
        ),
    ]
    metrics = C._secondary_metrics(records, policy)["baseline"]
    assert metrics["useful_distinct_behaviors"] == 1
    assert metrics["distinct_proof_mechanisms"] == 2
    assert metrics["proof_route_count_is_separate_from_behavior_count"] is True
    assert metrics["productive_failed_or_blocked_parent_reuse"] == 1


def test_preregistered_primary_rule_can_detect_a_large_matched_improvement() -> None:
    protocol = ablation.load_protocol(ROOT)
    records = []
    for index in range(24):
        task = f"task.{index:02d}"
        records.extend(
            [
                _record(task, "baseline", [_idea(f"baseline.{index}", f"proof.b.{index}")]),
                _record(
                    task,
                    "full_creativity_first",
                    [
                        _idea(f"treatment.{index}.a", f"proof.t.{index}.a"),
                        _idea(f"treatment.{index}.b", f"proof.t.{index}.b"),
                    ],
                ),
            ]
        )
    observations = {
        "schema_version": ablation.OBSERVATION_SCHEMA,
        "experiment_id": protocol["experiment_id"],
        "baseline_commit": protocol["baseline_commit"],
        "treatment_commit": "1" * 40,
        "records": records,
    }
    result = ablation.score_experiment(observations, protocol)
    assert result["verdict"] == "MORE_CREATIVE_ON_PREREGISTERED_BOUNDED_PROTOCOL"
    assert result["relative_improvement"] == "1/1"
    assert result["one_sided_sign_test_pvalue"] == "1/16777216"


def test_private_binding_rejects_public_journal_commitment_mismatch(tmp_path: Path) -> None:
    config = C.load_config(ROOT)
    review = json.loads((ROOT / config["review_packet"]["path"]).read_text(encoding="utf-8"))
    public = json.loads(
        (ROOT / config["generation_receipt"]["path"]).read_text(encoding="utf-8")
    )
    journal = C.DurableAttemptJournal.create(
        tmp_path / "attempts.jsonl",
        experiment_id=config["experiment_id"],
        source_bindings={
            key: value
            for key, value in public["source_bindings"].items()
            if key != "generation_packet_content_sha256"
        },
        unblinding_key=b"k" * 32,
    )
    mapping = []
    for task_id in sorted({output["task_id"] for output in review["blinded_outputs"]}):
        outputs = sorted(
            output["blinded_output_id"]
            for output in review["blinded_outputs"]
            if output["task_id"] == task_id
        )
        mapping.extend(
            [
                {"arm": "baseline", "blinded_output_id": outputs[0], "task_id": task_id},
                {
                    "arm": "full_creativity_first",
                    "blinded_output_id": outputs[1],
                    "task_id": task_id,
                },
            ]
        )
    coordinator = {
        "schema_version": confirmatory.COORDINATOR_SCHEMA,
        "experiment_id": config["experiment_id"],
        "public_receipt_content_sha256": public["content_sha256"],
        "review_packet_content_sha256": review["content_sha256"],
        "attempt_journal_content_sha256": journal.content_sha256,
        "unblinding_key_hex": journal.unblinding_key.hex(),
        "mapping": mapping,
        "arm_outcome_status_counts": {},
        "claims": {"safe_to_publish_before_review": False},
    }
    coordinator["content_sha256"] = canonical_sha256(coordinator)
    with pytest.raises(C.ConfirmatoryScoringError, match="public attempt journal commitment"):
        C._validate_private_bindings(ROOT, coordinator, journal, review, public, config)


def test_result_validation_keeps_single_rotation_and_novelty_claims_closed() -> None:
    config = C.load_config(ROOT)
    generation = confirmatory.load_config(ROOT)
    packet = json.loads((ROOT / config["review_packet"]["path"]).read_text(encoding="utf-8"))
    public = json.loads(
        (ROOT / config["generation_receipt"]["path"]).read_text(encoding="utf-8")
    )
    reviewer_rows = []
    for name, affiliation, seal in (
        ("Ada Lovelace", "Mathematics Lab A", "a" * 64),
        ("Emmy Noether", "Mathematics Lab B", "b" * 64),
    ):
        reviewer_rows.append(
            {
                "affiliation": affiliation,
                "content_sha256": seal,
                "full_name": name,
                "reviewer_id": "reviewer."
                + canonical_sha256(
                    {"affiliation": affiliation.casefold(), "full_name": name.casefold()}
                )[:24],
            }
        )
    result = {
        "schema_version": C.RESULT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "source_bindings": {
            "baseline_commit": generation["baseline_commit"],
            "decision_protocol": config["decision_protocol"],
            "generation_config": config["generation_config"],
            "generation_receipt_content_sha256": public["content_sha256"],
            "review_packet_content_sha256": packet["content_sha256"],
            "scorer": {
                "path": C.SCORER_PATH,
                "normalized_file_sha256": C.pilot._normalized_file_sha256(ROOT / C.SCORER_PATH),
            },
            "treatment_commit": generation["treatment_commit"],
        },
        "reviewers": reviewer_rows,
        "unblinding": {
            "attempt_journal_content_sha256": "c" * 64,
            "coordinator_content_sha256": "d" * 64,
            "mapping_opened_only_after_two_sealed_reviews_validated": True,
            "review_content_hashes": ["a" * 64, "b" * 64],
            "withheld_targets_opened": False,
        },
        "primary": {
            "metric": "blinded_useful_distinct_behavior_branches_per_10000_tokens",
            "paired_tasks": 24,
            "baseline_mean": "0/1",
            "treatment_mean": "0/1",
            "relative_improvement": "0/1",
            "one_sided_sign_test_pvalue": "1/1",
            "typed_usability_baseline_mean": "1/1",
            "typed_usability_treatment_mean": "1/1",
            "typed_usability_noninferior": True,
            "behavior_deduplicated": True,
        },
        "bounded_rotation_outcome": "NOT_ESTABLISHED_ON_ROTATION",
        "secondary": {
            arm: {"proof_route_count_is_separate_from_behavior_count": True}
            for arm in C._ARMS
        },
        "paired_task_deltas": [
            {
                "task_id": f"task.{index:02d}",
                "baseline": "0/1",
                "treatment": "0/1",
                "treatment_minus_baseline": "0/1",
            }
            for index in range(24)
        ],
        "reviewer_agreement": {
            "rated_branch_rows": 353,
            "exact_all_axis_agreements": 353,
            "rows_with_any_disagreement": 0,
            "axis_disagreement_counts": {
                "coherence": 0,
                "nontriviality": 0,
                "followup_value": 0,
            },
            "agreement_is_not_independence_proof": True,
        },
        "release_gate": {
            "component_knockouts_complete": False,
            "three_independent_level5_successes_complete": False,
            "repeated_rotating_external_benchmarks_complete": False,
            "claim_specific_prior_art_complete": False,
            "exact_cas_smt_interval_lean_claim_ladder_complete": False,
            "famous_problem_escalation_allowed": False,
            "system_wide_creativity_claim_allowed": False,
        },
        "claims": {
            "bounded_rotation_primary_rule_passed": False,
            "human_reviews_establish_literature_novelty": False,
            "internal_behavior_novelty_is_literature_novelty": False,
            "internal_proof_mechanism_novelty_is_literature_novelty": False,
            "reviewer_independence_externally_or_cryptographically_proven": False,
            "system_wide_more_creative_established": False,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    C.validate_result(result, ROOT)
    tampered = copy.deepcopy(result)
    tampered["claims"]["system_wide_more_creative_established"] = True
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(C.ConfirmatoryScoringError, match="claim boundary"):
        C.validate_result(tampered, ROOT)
