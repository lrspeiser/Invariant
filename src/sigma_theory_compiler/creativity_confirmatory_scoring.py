"""Seal blinded human reviews and score one bounded confirmatory creativity rotation.

Review forms contain no arm identities.  The scoring command validates and seals exactly two
complete, named, independent reviews before it reads the private attempt journal or coordinator.
The primary outcome deduplicates useful branches by behavior; proof-mechanism diversity remains a
separate secondary outcome and neither measure is represented as literature novelty.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import creativity_ablation as ablation
from . import creativity_confirmatory_generation as confirmatory
from . import creativity_confirmatory_recovery as recovery
from . import creativity_tournament_generation as pilot
from .durable_llm_attempt_journal import DurableAttemptJournal
from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/creativity_confirmatory_scoring.json"
SCORER_PATH = "src/sigma_theory_compiler/creativity_confirmatory_scoring.py"
CONFIG_SCHEMA = "invariant-creativity-confirmatory-scoring-config-1.0"
DRAFT_REVIEW_SCHEMA = "invariant-creativity-confirmatory-review-form-draft-1.0"
SEALED_REVIEW_SCHEMA = "invariant-creativity-confirmatory-review-form-1.0"
RESULT_SCHEMA = "invariant-creativity-confirmatory-scored-rotation-1.0"
_ARMS = ("baseline", "full_creativity_first")
_HEX = frozenset("0123456789abcdef")
_ATTESTATIONS = {
    "arm_identity_unknown_during_review",
    "completed_independently",
    "literature_novelty_not_scored",
    "other_reviewer_scores_unknown_during_review",
    "withheld_targets_not_accessed",
}
_CLAIMS = {
    "arm_identity_known_during_review": False,
    "external_independence_cryptographically_proven": False,
    "literature_novelty_scored": False,
    "other_reviewer_scores_known_during_review": False,
    "withheld_targets_accessed": False,
}
_REVIEW_MATERIAL_FIELDS = (
    "branch_kind",
    "expression",
    "falsifiers",
    "family",
    "generation_contract_status",
    "initial_check_status",
    "invariants",
    "known_analogues",
    "proof_plan",
    "rationale",
    "representation",
    "source_domains",
    "synthesis_note",
)


class ConfirmatoryScoringError(ValueError):
    """A review, unblinding binding, score, or bounded claim changed or is incomplete."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ConfirmatoryScoringError(f"{label} keys changed")


def _sha(value: Any, label: str, *, length: int = 64) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in _HEX for character in value)
    ):
        raise ConfirmatoryScoringError(f"{label} is not a lowercase digest")
    return value


def _bounded_text(
    value: Any, label: str, *, maximum_bytes: int = 2048, allow_empty: bool = False
) -> str:
    if not isinstance(value, str):
        raise ConfirmatoryScoringError(f"{label} is not text")
    normalized = value.strip()
    if not allow_empty and not normalized:
        raise ConfirmatoryScoringError(f"{label} is empty")
    if len(normalized.encode("utf-8")) > maximum_bytes:
        raise ConfirmatoryScoringError(f"{label} exceeds its byte budget")
    return normalized


def _recursive_key(value: Any, forbidden: str) -> bool:
    if isinstance(value, Mapping):
        return forbidden in value or any(_recursive_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_recursive_key(item, forbidden) for item in value)
    return False


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ConfirmatoryScoringError(f"{label} could not be read") from error
    if not isinstance(value, dict):
        raise ConfirmatoryScoringError(f"{label} is not an object")
    return value


def _under(root: Path, path: Path, label: str) -> Path:
    root = root.resolve()
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ConfirmatoryScoringError(f"{label} escapes the repository") from error
    return resolved


def _private_path(root: Path, value: Any, label: str) -> Path:
    path = _bounded_text(value, label, maximum_bytes=512)
    private_root = (root.resolve() / "work" / "creativity-confirmatory").resolve()
    resolved = _under(root, Path(path), label)
    try:
        resolved.relative_to(private_root)
    except ValueError as error:
        raise ConfirmatoryScoringError(f"{label} is outside the private experiment directory") from error
    return resolved


def _file_binding(root: Path, value: Mapping[str, Any], label: str) -> Path:
    _strict(value, {"path", "normalized_file_sha256"}, label)
    path = _under(root, Path(_bounded_text(value["path"], f"{label} path", maximum_bytes=512)), label)
    expected = _sha(value["normalized_file_sha256"], f"{label} normalized file hash")
    if pilot._normalized_file_sha256(path) != expected:
        raise ConfirmatoryScoringError(f"{label} normalized file binding changed")
    return path


def _artifact_binding(root: Path, value: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    _strict(value, {"content_sha256", "normalized_file_sha256", "path"}, label)
    path = _under(root, Path(_bounded_text(value["path"], f"{label} path", maximum_bytes=512)), label)
    if pilot._normalized_file_sha256(path) != _sha(
        value["normalized_file_sha256"], f"{label} normalized file hash"
    ):
        raise ConfirmatoryScoringError(f"{label} normalized file binding changed")
    artifact = _read_json(path, label)
    if artifact.get("content_sha256") != _sha(value["content_sha256"], f"{label} content hash"):
        raise ConfirmatoryScoringError(f"{label} content binding changed")
    return path, artifact


def load_config(root: Path) -> dict[str, Any]:
    """Load all public bindings.  This function intentionally reads no private artifacts."""

    root = root.resolve()
    value = _read_json(root / CONFIG_PATH, "confirmatory scoring config")
    _strict(
        value,
        {
            "claim_boundary",
            "decision_protocol",
            "experiment_id",
            "generation_config",
            "generation_receipt",
            "private_attempt_journal_path",
            "private_coordinator_path",
            "review_packet",
            "review_policy",
            "schema_version",
            "unblinding_policy",
        },
        "confirmatory scoring config",
    )
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["experiment_id"]
        != "creativity-first-vs-falsification-first-confirmatory-001"
    ):
        raise ConfirmatoryScoringError("confirmatory scoring config identity changed")
    _private_path(root, value["private_coordinator_path"], "private coordinator path")
    _private_path(root, value["private_attempt_journal_path"], "private journal path")
    _, review = _artifact_binding(root, value["review_packet"], "public review packet")
    _, public = _artifact_binding(root, value["generation_receipt"], "public generation receipt")
    generation_path = _file_binding(root, value["generation_config"], "generation config")
    protocol_path = _file_binding(root, value["decision_protocol"], "decision protocol")
    generation = confirmatory.load_config(root)
    protocol = ablation.load_protocol(root)
    if generation_path != (root / confirmatory.CONFIG_PATH).resolve():
        raise ConfirmatoryScoringError("generation config path changed")
    if protocol_path != (root / ablation.PROTOCOL_PATH).resolve():
        raise ConfirmatoryScoringError("decision protocol path changed")
    confirmatory.validate_public(review, public, root)
    recovery.validate_recovery_public(root, review, public)
    review_policy = value["review_policy"]
    _strict(
        review_policy,
        {
            "axes",
            "minimum_named_independent_reviewers",
            "rating_scale",
            "reviewer_name_placeholders",
            "useful_threshold_each_axis",
        },
        "confirmatory review policy",
    )
    if (
        review_policy["axes"] != generation["review"]["axes"]
        or review_policy["axes"] != protocol["human_review"]["axes"]
        or review_policy["rating_scale"] != generation["review"]["rating_scale"]
        or review_policy["rating_scale"] != protocol["human_review"]["rating_scale"]
        or review_policy["useful_threshold_each_axis"]
        != protocol["human_review"]["useful_threshold_each_axis"]
        or review_policy["minimum_named_independent_reviewers"] != 2
        or generation["review"]["minimum_named_reviewers"] != 2
        or protocol["human_review"]["minimum_independent_reviewers"] != 2
        or generation["baseline_commit"] != protocol["baseline_commit"]
    ):
        raise ConfirmatoryScoringError("confirmatory review or decision policy changed")
    if (
        review_policy["axes"] != ["coherence", "nontriviality", "followup_value"]
        or review_policy["rating_scale"] != [0, 1, 2, 3, 4]
        or not isinstance(review_policy["reviewer_name_placeholders"], list)
        or not review_policy["reviewer_name_placeholders"]
        or any(
            not isinstance(item, str) or not item.strip()
            for item in review_policy["reviewer_name_placeholders"]
        )
    ):
        raise ConfirmatoryScoringError("confirmatory review scale or placeholder policy changed")
    unblinding = value["unblinding_policy"]
    _strict(
        unblinding,
        {
            "exact_review_count",
            "review_content_sealed_before_mapping_read",
            "reviewers_must_be_distinct",
            "withheld_targets_remain_closed_during_review",
        },
        "unblinding policy",
    )
    if unblinding != {
        "exact_review_count": 2,
        "review_content_sealed_before_mapping_read": True,
        "reviewers_must_be_distinct": True,
        "withheld_targets_remain_closed_during_review": True,
    }:
        raise ConfirmatoryScoringError("unblinding policy weakened")
    _strict(
        value["claim_boundary"],
        {
            "bounded_single_rotation_can_establish_system_wide_superiority",
            "human_usefulness_review_establishes_literature_novelty",
            "internal_behavior_diversity_is_literature_novelty",
            "proof_mechanism_diversity_is_literature_novelty",
        },
        "confirmatory claim boundary",
    )
    if any(item is not False for item in value["claim_boundary"].values()):
        raise ConfirmatoryScoringError("confirmatory claim boundary changed")
    return value


def _review_packet(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    return _read_json(_under(root, Path(config["review_packet"]["path"]), "review packet"), "review packet")


def _rating_rows(review: Mapping[str, Any], axes: Sequence[str]) -> list[dict[str, Any]]:
    return [
        {
            "blinded_output_id": output["blinded_output_id"],
            "branch_id": branch["branch_id"],
            "task_id": output["task_id"],
            "review_material": {field: branch[field] for field in _REVIEW_MATERIAL_FIELDS},
            **{axis: None for axis in axes},
            "note": "",
        }
        for output in review["blinded_outputs"]
        for branch in output["branches"]
    ]


def make_review_template(root: Path, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = dict(config or load_config(root))
    review = _review_packet(root, config)
    axes = config["review_policy"]["axes"]
    form: dict[str, Any] = {
        "schema_version": DRAFT_REVIEW_SCHEMA,
        "experiment_id": config["experiment_id"],
        "review_packet_binding": dict(config["review_packet"]),
        "reviewer": {
            "affiliation": "",
            "conflict_disclosure": "",
            "full_name": "",
            "reviewed_utc": None,
            "reviewer_is_generation_operator": None,
        },
        "attestations": {key: None for key in sorted(_ATTESTATIONS)},
        "rating_policy": {
            "axes": list(axes),
            "rating_scale": list(config["review_policy"]["rating_scale"]),
            "useful_threshold_each_axis": config["review_policy"][
                "useful_threshold_each_axis"
            ],
        },
        "ratings": _rating_rows(review, axes),
        "claims": dict(_CLAIMS),
    }
    if _recursive_key(form, "arm"):
        raise ConfirmatoryScoringError("draft review form disclosed an arm identity")
    return form


def _expected_rating_keys(review: Mapping[str, Any]) -> set[tuple[str, str, str]]:
    return {
        (output["blinded_output_id"], output["task_id"], branch["branch_id"])
        for output in review["blinded_outputs"]
        for branch in output["branches"]
    }


def _validate_reviewer(reviewer: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    _strict(
        reviewer,
        {
            "affiliation",
            "conflict_disclosure",
            "full_name",
            "reviewed_utc",
            "reviewer_is_generation_operator",
        },
        "reviewer",
    )
    name = _bounded_text(reviewer["full_name"], "reviewer full name", maximum_bytes=160)
    normalized_words = [word for word in name.replace("-", " ").split() if word]
    placeholders = {item.casefold() for item in config["review_policy"]["reviewer_name_placeholders"]}
    if len(normalized_words) < 2 or name.casefold() in placeholders or any(
        word.casefold() in placeholders for word in normalized_words
    ):
        raise ConfirmatoryScoringError("reviewer must be a specifically named person")
    affiliation = _bounded_text(reviewer["affiliation"], "reviewer affiliation", maximum_bytes=256)
    _bounded_text(reviewer["conflict_disclosure"], "conflict disclosure", maximum_bytes=1024)
    if reviewer["reviewer_is_generation_operator"] is not False:
        raise ConfirmatoryScoringError("generation operator cannot serve as a blinded reviewer")
    reviewed_utc = _bounded_text(reviewer["reviewed_utc"], "review timestamp", maximum_bytes=64)
    try:
        parsed = datetime.fromisoformat(reviewed_utc)
    except ValueError as error:
        raise ConfirmatoryScoringError("review timestamp is invalid") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ConfirmatoryScoringError("review timestamp is not timezone-aware")
    return "reviewer." + canonical_sha256(
        {"affiliation": affiliation.casefold(), "full_name": name.casefold()}
    )[:24]


def _validate_ratings(
    ratings: Any, review: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    if not isinstance(ratings, list):
        raise ConfirmatoryScoringError("review ratings are not an array")
    axes = config["review_policy"]["axes"]
    expected_fields = {
        "blinded_output_id",
        "branch_id",
        "note",
        "review_material",
        "task_id",
        *axes,
    }
    observed: set[tuple[str, str, str]] = set()
    rating_scale = set(config["review_policy"]["rating_scale"])
    expected_material = {
        (output["blinded_output_id"], output["task_id"], branch["branch_id"]): {
            field: branch[field] for field in _REVIEW_MATERIAL_FIELDS
        }
        for output in review["blinded_outputs"]
        for branch in output["branches"]
    }
    for row in ratings:
        _strict(row, expected_fields, "review rating")
        key = (row["blinded_output_id"], row["task_id"], row["branch_id"])
        if key in observed:
            raise ConfirmatoryScoringError("duplicate branch rating")
        observed.add(key)
        if row["review_material"] != expected_material.get(key):
            raise ConfirmatoryScoringError("review material changed")
        for axis in axes:
            if isinstance(row[axis], bool) or not isinstance(row[axis], int) or row[axis] not in rating_scale:
                raise ConfirmatoryScoringError(f"{axis} rating is outside the registered scale")
        _bounded_text(row["note"], "review note", maximum_bytes=2048, allow_empty=True)
    if observed != _expected_rating_keys(review):
        raise ConfirmatoryScoringError("review rating coverage changed")


def validate_sealed_review(
    form: Mapping[str, Any],
    root: Path,
    config: Mapping[str, Any] | None = None,
    review: Mapping[str, Any] | None = None,
) -> None:
    root = root.resolve()
    config = dict(config or load_config(root))
    review = dict(review or _review_packet(root, config))
    _strict(
        form,
        {
            "attestations",
            "claims",
            "content_sha256",
            "experiment_id",
            "rating_policy",
            "ratings",
            "review_packet_binding",
            "reviewer",
            "reviewer_id",
            "schema_version",
        },
        "sealed review form",
    )
    body = {key: item for key, item in form.items() if key != "content_sha256"}
    if (
        form["schema_version"] != SEALED_REVIEW_SCHEMA
        or form["experiment_id"] != config["experiment_id"]
        or form["review_packet_binding"] != config["review_packet"]
        or form["content_sha256"] != canonical_sha256(body)
        or _recursive_key(form, "arm")
    ):
        raise ConfirmatoryScoringError("sealed review identity, binding, blinding, or seal changed")
    expected_reviewer_id = _validate_reviewer(form["reviewer"], config)
    if form["reviewer_id"] != expected_reviewer_id:
        raise ConfirmatoryScoringError("reviewer identity digest changed")
    _strict(form["attestations"], _ATTESTATIONS, "review attestations")
    if any(form["attestations"][key] is not True for key in _ATTESTATIONS):
        raise ConfirmatoryScoringError("all review attestations must be affirmative")
    if form["claims"] != _CLAIMS:
        raise ConfirmatoryScoringError("review claim boundary changed")
    expected_policy = {
        "axes": config["review_policy"]["axes"],
        "rating_scale": config["review_policy"]["rating_scale"],
        "useful_threshold_each_axis": config["review_policy"]["useful_threshold_each_axis"],
    }
    if form["rating_policy"] != expected_policy:
        raise ConfirmatoryScoringError("review rating policy changed")
    _validate_ratings(form["ratings"], review, config)


def seal_review(
    draft: Mapping[str, Any], root: Path, config: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    root = root.resolve()
    config = dict(config or load_config(root))
    review = _review_packet(root, config)
    _strict(
        draft,
        {
            "attestations",
            "claims",
            "experiment_id",
            "rating_policy",
            "ratings",
            "review_packet_binding",
            "reviewer",
            "schema_version",
        },
        "draft review form",
    )
    if draft["schema_version"] != DRAFT_REVIEW_SCHEMA or _recursive_key(draft, "arm"):
        raise ConfirmatoryScoringError("draft review identity or blinding changed")
    sealed = json.loads(json.dumps(draft))
    sealed["schema_version"] = SEALED_REVIEW_SCHEMA
    if sealed["reviewer"].get("reviewed_utc") is None:
        sealed["reviewer"]["reviewed_utc"] = datetime.now(UTC).isoformat().replace(
            "+00:00", "Z"
        )
    sealed["reviewer_id"] = _validate_reviewer(sealed["reviewer"], config)
    _validate_ratings(sealed["ratings"], review, config)
    if sealed["experiment_id"] != config["experiment_id"]:
        raise ConfirmatoryScoringError("draft review experiment changed")
    if sealed["review_packet_binding"] != config["review_packet"]:
        raise ConfirmatoryScoringError("draft review packet binding changed")
    if sealed["claims"] != _CLAIMS:
        raise ConfirmatoryScoringError("draft review claim boundary changed")
    _strict(sealed["attestations"], _ATTESTATIONS, "review attestations")
    if any(sealed["attestations"][key] is not True for key in _ATTESTATIONS):
        raise ConfirmatoryScoringError("all review attestations must be affirmative")
    expected_policy = {
        "axes": config["review_policy"]["axes"],
        "rating_scale": config["review_policy"]["rating_scale"],
        "useful_threshold_each_axis": config["review_policy"]["useful_threshold_each_axis"],
    }
    if sealed["rating_policy"] != expected_policy:
        raise ConfirmatoryScoringError("draft review rating policy changed")
    sealed["content_sha256"] = canonical_sha256(sealed)
    validate_sealed_review(sealed, root, config, review)
    return sealed


def validate_review_pair(
    forms: Sequence[Mapping[str, Any]],
    root: Path,
    config: Mapping[str, Any] | None = None,
    review: Mapping[str, Any] | None = None,
) -> None:
    root = root.resolve()
    config = dict(config or load_config(root))
    review = dict(review or _review_packet(root, config))
    if len(forms) != config["unblinding_policy"]["exact_review_count"]:
        raise ConfirmatoryScoringError("exactly two sealed review forms are required")
    for form in forms:
        validate_sealed_review(form, root, config, review)
    reviewer_ids = {form["reviewer_id"] for form in forms}
    reviewer_names = {form["reviewer"]["full_name"].strip().casefold() for form in forms}
    if len(reviewer_ids) != 2 or len(reviewer_names) != 2:
        raise ConfirmatoryScoringError("reviewers are not distinct")


def _validate_private_bindings(
    root: Path,
    coordinator: Mapping[str, Any],
    journal: DurableAttemptJournal,
    review: Mapping[str, Any],
    public: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str]:
    if public.get("attempt_accounting", {}).get("attempt_journal_content_sha256") != journal.content_sha256:
        raise ConfirmatoryScoringError("public attempt journal commitment changed")
    confirmatory.validate_coordinator(coordinator, review, public, journal)
    header = journal.header
    # The journal header intentionally excludes the generation packet content digest.
    expected = dict(public["source_bindings"])
    expected.pop("generation_packet_content_sha256", None)
    if header.get("experiment_id") != config["experiment_id"] or header.get(
        "source_bindings"
    ) != expected:
        raise ConfirmatoryScoringError("private journal source binding changed")
    mapping = coordinator.get("mapping")
    if not isinstance(mapping, list) or len(mapping) != len(review["blinded_outputs"]):
        raise ConfirmatoryScoringError("private mapping coverage changed")
    indexed_outputs = {
        (item["task_id"], item["blinded_output_id"]) for item in review["blinded_outputs"]
    }
    observed: set[tuple[str, str]] = set()
    arms_by_task: dict[str, set[str]] = defaultdict(set)
    key = journal.unblinding_key
    for item in mapping:
        _strict(item, {"arm", "blinded_output_id", "task_id"}, "private mapping row")
        if item["arm"] not in _ARMS:
            raise ConfirmatoryScoringError("private mapping arm changed")
        pair = (item["task_id"], item["blinded_output_id"])
        if pair in observed or pair not in indexed_outputs:
            raise ConfirmatoryScoringError("private mapping is duplicated or not in the review packet")
        observed.add(pair)
        arms_by_task[item["task_id"]].add(item["arm"])
        if pilot._blinded_id(key, item["task_id"], item["arm"]) != item["blinded_output_id"]:
            raise ConfirmatoryScoringError("private mapping HMAC changed")
    if observed != indexed_outputs or any(arms != set(_ARMS) for arms in arms_by_task.values()):
        raise ConfirmatoryScoringError("private paired mapping coverage changed")
    return {item["blinded_output_id"]: item["arm"] for item in mapping}


def _review_index(forms: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    result: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for form in forms:
        for row in form["ratings"]:
            result[(row["blinded_output_id"], row["branch_id"])].append(
                {
                    "reviewer_id": form["reviewer_id"],
                    "coherence": row["coherence"],
                    "followup_value": row["followup_value"],
                    "nontriviality": row["nontriviality"],
                }
            )
    return result


def _observations(
    review: Mapping[str, Any],
    forms: Sequence[Mapping[str, Any]],
    mapping: Mapping[str, str],
    generation: Mapping[str, Any],
    experiment_id: str,
) -> dict[str, Any]:
    reviews = _review_index(forms)
    records = []
    for output in review["blinded_outputs"]:
        ideas = []
        for branch in output["branches"]:
            if branch["branch_kind"] == "all_proposals_rejected_outcome":
                continue
            ideas.append(
                {
                    "behavior_sha256": branch["behavior_sha256"],
                    "human_reviews": reviews[(output["blinded_output_id"], branch["branch_id"])],
                    "initial_check_status": branch["initial_check_status"],
                    "later_used_as_parent": branch["later_used_as_parent"],
                    "llm_origin_assessment": branch["llm_origin_assessment"],
                    "prior_art_classification": "review_pending",
                    "proof_mechanism_sha256": branch["proof_mechanism_sha256"],
                    "representation": branch["representation"],
                    "source_domains": branch["source_domains"],
                }
            )
        records.append(
            {
                "arm": mapping[output["blinded_output_id"]],
                "blinded_output_id": output["blinded_output_id"],
                "ideas": ideas,
                "resource_budget": output["resource_budget"],
                "task_id": output["task_id"],
                "tokens_used": output["tokens_used"],
                "typed_usable_ideas": output["typed_usable_ideas"],
            }
        )
    return {
        "schema_version": ablation.OBSERVATION_SCHEMA,
        "experiment_id": experiment_id,
        "baseline_commit": generation["baseline_commit"],
        "treatment_commit": generation["treatment_commit"],
        "records": records,
    }


def _useful_branch(branch: Mapping[str, Any], policy: Mapping[str, Any]) -> bool:
    threshold = policy["useful_threshold_each_axis"]
    axes = policy["axes"]
    return all(all(review[axis] >= threshold for axis in axes) for review in branch["human_reviews"])


def _fraction(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _secondary_metrics(
    records: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]
) -> dict[str, Any]:
    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_arm[record["arm"]].append(record)
    result: dict[str, Any] = {}
    for arm in _ARMS:
        arm_records = by_arm[arm]
        tokens = sum(item["tokens_used"] for item in arm_records)
        useful = [
            idea
            for record in arm_records
            for idea in record["ideas"]
            if _useful_branch(idea, policy)
        ]
        behavior_hashes = {item["behavior_sha256"] for item in useful}
        proof_hashes = {item["proof_mechanism_sha256"] for item in useful}
        all_ideas = [idea for record in arm_records for idea in record["ideas"]]
        all_origin_counts = Counter(item["llm_origin_assessment"] for item in all_ideas)
        useful_origin_counts = Counter(item["llm_origin_assessment"] for item in useful)
        recovered = sum(
            idea["initial_check_status"] in {"blocked", "failed"}
            and idea["later_used_as_parent"] is True
            for record in arm_records
            for idea in record["ideas"]
        )
        result[arm] = {
            "tokens": tokens,
            "useful_reviewed_branch_rows": len(useful),
            "useful_distinct_behaviors": len(behavior_hashes),
            "useful_distinct_behaviors_per_10000_tokens": _fraction(
                Fraction(10_000 * len(behavior_hashes), tokens)
            ),
            "distinct_proof_mechanisms": len(proof_hashes),
            "distinct_proof_mechanisms_per_10000_tokens": _fraction(
                Fraction(10_000 * len(proof_hashes), tokens)
            ),
            "proof_route_count_is_separate_from_behavior_count": True,
            "cross_domain_useful_branch_rows": sum(
                len(set(item["source_domains"])) >= 2 for item in useful
            ),
            "cross_domain_useful_branch_rate": _fraction(
                Fraction(
                    sum(len(set(item["source_domains"])) >= 2 for item in useful),
                    max(1, len(useful)),
                )
            ),
            "representations": sorted({item["representation"] for item in useful}),
            "representation_coverage_count": len(
                {item["representation"] for item in useful}
            ),
            "productive_failed_or_blocked_parent_reuse": recovered,
            "productive_failed_or_blocked_parent_reuse_rate": _fraction(
                Fraction(recovered, max(1, len(all_ideas)))
            ),
            "all_llm_origin_label_counts_not_novelty_judgments": dict(
                sorted(all_origin_counts.items())
            ),
            "useful_llm_origin_label_counts_not_novelty_judgments": dict(
                sorted(useful_origin_counts.items())
            ),
        }
    return result


def _paired_deltas(
    records: Sequence[Mapping[str, Any]], protocol: Mapping[str, Any]
) -> list[dict[str, Any]]:
    indexed = {(item["task_id"], item["arm"]): item for item in records}
    rows = []
    for task_id in sorted({item["task_id"] for item in records}):
        baseline = ablation._arm_task_metrics(indexed[(task_id, "baseline")], protocol)
        treatment = ablation._arm_task_metrics(
            indexed[(task_id, "full_creativity_first")], protocol
        )
        delta = Fraction(treatment["useful_behavior_branches_per_10000_tokens"]) - Fraction(
            baseline["useful_behavior_branches_per_10000_tokens"]
        )
        rows.append(
            {
                "task_id": task_id,
                "baseline": baseline["useful_behavior_branches_per_10000_tokens"],
                "treatment": treatment["useful_behavior_branches_per_10000_tokens"],
                "treatment_minus_baseline": _fraction(delta),
            }
        )
    return rows


def _reviewer_agreement(forms: Sequence[Mapping[str, Any]], axes: Sequence[str]) -> dict[str, Any]:
    left = {(row["blinded_output_id"], row["branch_id"]): row for row in forms[0]["ratings"]}
    right = {(row["blinded_output_id"], row["branch_id"]): row for row in forms[1]["ratings"]}
    exact_rows = sum(all(left[key][axis] == right[key][axis] for axis in axes) for key in left)
    axis_disagreements = {
        axis: sum(left[key][axis] != right[key][axis] for key in left) for axis in axes
    }
    return {
        "rated_branch_rows": len(left),
        "exact_all_axis_agreements": exact_rows,
        "rows_with_any_disagreement": len(left) - exact_rows,
        "axis_disagreement_counts": axis_disagreements,
        "agreement_is_not_independence_proof": True,
    }


def score_rotation(
    root: Path,
    forms: Sequence[Mapping[str, Any]],
    coordinator: Mapping[str, Any],
    journal: DurableAttemptJournal,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score already validated reviews against explicitly supplied private artifacts."""

    root = root.resolve()
    config = dict(config or load_config(root))
    review = _review_packet(root, config)
    public = _read_json(
        _under(root, Path(config["generation_receipt"]["path"]), "generation receipt"),
        "generation receipt",
    )
    validate_review_pair(forms, root, config, review)
    mapping = _validate_private_bindings(root, coordinator, journal, review, public, config)
    generation = confirmatory.load_config(root)
    protocol = ablation.load_protocol(root)
    adapted_protocol = json.loads(json.dumps(protocol))
    adapted_protocol["experiment_id"] = config["experiment_id"]
    observations = _observations(review, forms, mapping, generation, config["experiment_id"])
    scored = ablation.score_experiment(observations, adapted_protocol)
    bounded_outcome = (
        "BOUNDED_ROTATION_PRIMARY_RULE_PASSED"
        if scored["verdict"] == "MORE_CREATIVE_ON_PREREGISTERED_BOUNDED_PROTOCOL"
        else "NOT_ESTABLISHED_ON_ROTATION"
    )
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "experiment_id": config["experiment_id"],
        "source_bindings": {
            "baseline_commit": generation["baseline_commit"],
            "decision_protocol": dict(config["decision_protocol"]),
            "generation_config": dict(config["generation_config"]),
            "generation_receipt_content_sha256": public["content_sha256"],
            "review_packet_content_sha256": review["content_sha256"],
            "scorer": {
                "path": SCORER_PATH,
                "normalized_file_sha256": pilot._normalized_file_sha256(root / SCORER_PATH),
            },
            "treatment_commit": generation["treatment_commit"],
        },
        "reviewers": [
            {
                "affiliation": form["reviewer"]["affiliation"],
                "content_sha256": form["content_sha256"],
                "full_name": form["reviewer"]["full_name"],
                "reviewer_id": form["reviewer_id"],
            }
            for form in sorted(forms, key=lambda item: item["reviewer_id"])
        ],
        "unblinding": {
            "attempt_journal_content_sha256": journal.content_sha256,
            "coordinator_content_sha256": coordinator["content_sha256"],
            "mapping_opened_only_after_two_sealed_reviews_validated": True,
            "review_content_hashes": sorted(form["content_sha256"] for form in forms),
            "withheld_targets_opened": False,
        },
        "primary": {
            "metric": scored["primary_metric"],
            "paired_tasks": scored["paired_tasks"],
            "baseline_mean": scored["baseline_mean"],
            "treatment_mean": scored["treatment_mean"],
            "relative_improvement": scored["relative_improvement"],
            "one_sided_sign_test_pvalue": scored["one_sided_sign_test_pvalue"],
            "typed_usability_baseline_mean": scored["typed_usability_baseline_mean"],
            "typed_usability_treatment_mean": scored["typed_usability_treatment_mean"],
            "typed_usability_noninferior": scored["typed_usability_noninferior"],
            "behavior_deduplicated": True,
        },
        "bounded_rotation_outcome": bounded_outcome,
        "secondary": _secondary_metrics(observations["records"], config["review_policy"]),
        "paired_task_deltas": _paired_deltas(observations["records"], adapted_protocol),
        "reviewer_agreement": _reviewer_agreement(forms, config["review_policy"]["axes"]),
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
            "bounded_rotation_primary_rule_passed": bounded_outcome
            == "BOUNDED_ROTATION_PRIMARY_RULE_PASSED",
            "human_reviews_establish_literature_novelty": False,
            "internal_behavior_novelty_is_literature_novelty": False,
            "internal_proof_mechanism_novelty_is_literature_novelty": False,
            "reviewer_independence_externally_or_cryptographically_proven": False,
            "system_wide_more_creative_established": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_result(body, root)
    return body


def score_from_review_paths(root: Path, review_paths: Sequence[Path]) -> dict[str, Any]:
    """Validate both public review forms before opening either configured private path."""

    root = root.resolve()
    config = load_config(root)
    review = _review_packet(root, config)
    forms = [_read_json(path.resolve(), "sealed review form") for path in review_paths]
    validate_review_pair(forms, root, config, review)
    # Do not move either read above validate_review_pair: this is the enforced unblinding boundary.
    journal = DurableAttemptJournal.load(
        _private_path(root, config["private_attempt_journal_path"], "private journal path")
    )
    coordinator = _read_json(
        _private_path(root, config["private_coordinator_path"], "private coordinator path"),
        "private coordinator",
    )
    return score_rotation(root, forms, coordinator, journal, config)


def validate_result(result: Mapping[str, Any], root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    generation = confirmatory.load_config(root)
    review = _review_packet(root, config)
    public = _read_json(
        _under(root, Path(config["generation_receipt"]["path"]), "generation receipt"),
        "generation receipt",
    )
    expected = {
        "bounded_rotation_outcome",
        "claims",
        "content_sha256",
        "experiment_id",
        "paired_task_deltas",
        "primary",
        "release_gate",
        "reviewer_agreement",
        "reviewers",
        "schema_version",
        "secondary",
        "source_bindings",
        "unblinding",
    }
    _strict(result, expected, "confirmatory scored result")
    body = {key: item for key, item in result.items() if key != "content_sha256"}
    if (
        result["schema_version"] != RESULT_SCHEMA
        or result["experiment_id"] != config["experiment_id"]
        or result["content_sha256"] != canonical_sha256(body)
        or result["bounded_rotation_outcome"]
        not in {"BOUNDED_ROTATION_PRIMARY_RULE_PASSED", "NOT_ESTABLISHED_ON_ROTATION"}
    ):
        raise ConfirmatoryScoringError("confirmatory scored result identity or seal changed")
    claims = result["claims"]
    _strict(
        claims,
        {
            "bounded_rotation_primary_rule_passed",
            "human_reviews_establish_literature_novelty",
            "internal_behavior_novelty_is_literature_novelty",
            "internal_proof_mechanism_novelty_is_literature_novelty",
            "reviewer_independence_externally_or_cryptographically_proven",
            "system_wide_more_creative_established",
        },
        "confirmatory result claims",
    )
    required_false = {
        "human_reviews_establish_literature_novelty",
        "internal_behavior_novelty_is_literature_novelty",
        "internal_proof_mechanism_novelty_is_literature_novelty",
        "reviewer_independence_externally_or_cryptographically_proven",
        "system_wide_more_creative_established",
    }
    if any(claims.get(key) is not False for key in required_false):
        raise ConfirmatoryScoringError("confirmatory scored claim boundary changed")
    if claims["bounded_rotation_primary_rule_passed"] is not (
        result["bounded_rotation_outcome"] == "BOUNDED_ROTATION_PRIMARY_RULE_PASSED"
    ):
        raise ConfirmatoryScoringError("bounded rotation outcome and claim disagree")
    _strict(
        result["release_gate"],
        {
            "claim_specific_prior_art_complete",
            "component_knockouts_complete",
            "exact_cas_smt_interval_lean_claim_ladder_complete",
            "famous_problem_escalation_allowed",
            "repeated_rotating_external_benchmarks_complete",
            "system_wide_creativity_claim_allowed",
            "three_independent_level5_successes_complete",
        },
        "confirmatory release gate",
    )
    if any(result["release_gate"].values()):
        raise ConfirmatoryScoringError("confirmatory release gate was opened by one rotation")
    _strict(
        result["source_bindings"],
        {
            "baseline_commit",
            "decision_protocol",
            "generation_config",
            "generation_receipt_content_sha256",
            "review_packet_content_sha256",
            "scorer",
            "treatment_commit",
        },
        "confirmatory result source bindings",
    )
    expected_bindings = {
        "baseline_commit": generation["baseline_commit"],
        "decision_protocol": dict(config["decision_protocol"]),
        "generation_config": dict(config["generation_config"]),
        "generation_receipt_content_sha256": public["content_sha256"],
        "review_packet_content_sha256": review["content_sha256"],
        "scorer": {
            "path": SCORER_PATH,
            "normalized_file_sha256": pilot._normalized_file_sha256(root / SCORER_PATH),
        },
        "treatment_commit": generation["treatment_commit"],
    }
    if result["source_bindings"] != expected_bindings:
        raise ConfirmatoryScoringError("confirmatory result source binding changed")
    primary_keys = {
        "baseline_mean",
        "behavior_deduplicated",
        "metric",
        "one_sided_sign_test_pvalue",
        "paired_tasks",
        "relative_improvement",
        "treatment_mean",
        "typed_usability_baseline_mean",
        "typed_usability_noninferior",
        "typed_usability_treatment_mean",
    }
    _strict(result["primary"], primary_keys, "confirmatory primary result")
    if (
        result["primary"]["paired_tasks"] != 24
        or result["primary"]["behavior_deduplicated"] is not True
        or result["primary"]["metric"]
        != "blinded_useful_distinct_behavior_branches_per_10000_tokens"
    ):
        raise ConfirmatoryScoringError("confirmatory primary metric changed")
    if set(result["secondary"]) != set(_ARMS) or any(
        item.get("proof_route_count_is_separate_from_behavior_count") is not True
        for item in result["secondary"].values()
    ):
        raise ConfirmatoryScoringError("confirmatory secondary metric separation changed")
    deltas = result["paired_task_deltas"]
    if (
        not isinstance(deltas, list)
        or len(deltas) != 24
        or len({item.get("task_id") for item in deltas}) != 24
    ):
        raise ConfirmatoryScoringError("confirmatory paired task deltas changed")
    reviewers = result["reviewers"]
    if not isinstance(reviewers, list) or len(reviewers) != 2:
        raise ConfirmatoryScoringError("confirmatory named reviewer count changed")
    reviewer_ids = set()
    reviewer_names = set()
    for reviewer in reviewers:
        _strict(
            reviewer,
            {"affiliation", "content_sha256", "full_name", "reviewer_id"},
            "confirmatory named reviewer",
        )
        name = _bounded_text(reviewer["full_name"], "reviewer full name", maximum_bytes=160)
        affiliation = _bounded_text(
            reviewer["affiliation"], "reviewer affiliation", maximum_bytes=256
        )
        expected_id = "reviewer." + canonical_sha256(
            {"affiliation": affiliation.casefold(), "full_name": name.casefold()}
        )[:24]
        if reviewer["reviewer_id"] != expected_id:
            raise ConfirmatoryScoringError("confirmatory reviewer identity changed")
        _sha(reviewer["content_sha256"], "sealed review hash")
        reviewer_ids.add(reviewer["reviewer_id"])
        reviewer_names.add(name.casefold())
    if len(reviewer_ids) != 2 or len(reviewer_names) != 2:
        raise ConfirmatoryScoringError("confirmatory result reviewers are not distinct")
    _strict(
        result["unblinding"],
        {
            "attempt_journal_content_sha256",
            "coordinator_content_sha256",
            "mapping_opened_only_after_two_sealed_reviews_validated",
            "review_content_hashes",
            "withheld_targets_opened",
        },
        "confirmatory unblinding receipt",
    )
    if (
        result["unblinding"]["mapping_opened_only_after_two_sealed_reviews_validated"]
        is not True
        or result["unblinding"]["withheld_targets_opened"] is not False
        or sorted(result["unblinding"]["review_content_hashes"])
        != sorted(reviewer["content_sha256"] for reviewer in reviewers)
    ):
        raise ConfirmatoryScoringError("confirmatory unblinding receipt changed")
    agreement = result["reviewer_agreement"]
    _strict(
        agreement,
        {
            "agreement_is_not_independence_proof",
            "axis_disagreement_counts",
            "exact_all_axis_agreements",
            "rated_branch_rows",
            "rows_with_any_disagreement",
        },
        "reviewer agreement",
    )
    if (
        agreement["rated_branch_rows"] != 353
        or agreement["agreement_is_not_independence_proof"] is not True
        or agreement["exact_all_axis_agreements"] + agreement["rows_with_any_disagreement"]
        != agreement["rated_branch_rows"]
    ):
        raise ConfirmatoryScoringError("reviewer agreement accounting changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template")
    template.add_argument("--root", type=Path, default=Path.cwd())
    template.add_argument("--output", type=Path, required=True)
    seal = subparsers.add_parser("seal-review")
    seal.add_argument("--root", type=Path, default=Path.cwd())
    seal.add_argument("--draft", type=Path, required=True)
    seal.add_argument("--output", type=Path, required=True)
    validate_review = subparsers.add_parser("validate-review")
    validate_review.add_argument("--root", type=Path, default=Path.cwd())
    validate_review.add_argument("--review-form", type=Path, required=True)
    score = subparsers.add_parser("score")
    score.add_argument("--root", type=Path, default=Path.cwd())
    score.add_argument("--review-form", type=Path, action="append", required=True)
    score.add_argument("--output", type=Path, required=True)
    validate_scored = subparsers.add_parser("validate-result")
    validate_scored.add_argument("--root", type=Path, default=Path.cwd())
    validate_scored.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "template":
        value = make_review_template(root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"ratings": len(value["ratings"]), "status": "DRAFT_CREATED"}))
    elif args.command == "seal-review":
        value = seal_review(_read_json(args.draft.resolve(), "draft review form"), root)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"reviewer_id": value["reviewer_id"], "status": "SEALED"}))
    elif args.command == "validate-review":
        value = _read_json(args.review_form.resolve(), "sealed review form")
        validate_sealed_review(value, root)
        print(json.dumps({"reviewer_id": value["reviewer_id"], "status": "VALID"}))
    elif args.command == "score":
        value = score_from_review_paths(root, args.review_form)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"outcome": value["bounded_rotation_outcome"], "status": "SCORED"}))
    else:
        value = _read_json(args.result.resolve(), "scored result")
        validate_result(value, root)
        print(json.dumps({"outcome": value["bounded_rotation_outcome"], "status": "VALID"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
