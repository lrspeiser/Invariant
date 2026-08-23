"""Materialize a blinded zero-retained-idea outcome without repeating generation.

The confirmatory generator can validly admit three proposals and then have its critic reject all
three.  The original public validator required at least one branch per blinded output, so such an
outcome stopped publication after all provider calls had completed.  This recovery is deliberately
post-generation and pre-review: it reconstructs the sealed calls, forbids transport, inserts a
single scored rejection placeholder while preserving ``typed_usable_ideas == 0``, and publishes a
source-bound deviation record.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import creativity_confirmatory_generation as confirmatory
from . import creativity_tournament_generation as pilot
from .durable_llm_attempt_journal import DurableAttemptJournal
from .sigma_core import canonical_sha256

RECOVERY_PATH = "src/sigma_theory_compiler/creativity_confirmatory_recovery.py"
DEVIATION_SCHEMA = "invariant-creativity-confirmatory-post-generation-deviation-1.0"


class ConfirmatoryRecoveryError(ValueError):
    """The recovery escaped its no-call, pre-review, or zero-idea boundary."""


def _reseal(value: dict[str, Any]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def _verify_seal(value: Mapping[str, Any], label: str) -> None:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise ConfirmatoryRecoveryError(f"{label} content seal changed before recovery")


def _rejection_placeholder(task_id: str, blinded_output_id: str) -> dict[str, Any]:
    branch: dict[str, Any] = {
        "behavior_sha256": canonical_sha256(
            {
                "blinded_output_id": blinded_output_id,
                "status": "all_schema_admitted_proposals_rejected",
                "task_id": task_id,
            }
        ),
        "branch_kind": "all_proposals_rejected_outcome",
        "expression": "No hypothesis survived the scheduled critic's rejection decisions.",
        "family": "scored_zero_retained_ideas",
        "falsifiers": ["a retained hypothesis branch exists for this blinded output"],
        "generation_contract_status": "pass",
        "initial_check_status": "failed",
        "invariants": [
            "provider_calls_not_repeated",
            "typed_usable_ideas_remains_zero",
            "blinded_arm_mapping_not_opened",
        ],
        "known_analogues": [],
        "later_used_as_parent": False,
        "llm_origin_assessment": "uncertain",
        "proof_mechanism": "none_all_proposals_rejected",
        "proof_mechanism_sha256": canonical_sha256("none_all_proposals_rejected"),
        "proof_plan": [
            "score this placeholder as the system's zero-retained-idea outcome",
            "do not infer or disclose the generating arm before review",
        ],
        "rationale": (
            "The schema-valid proposer and critic calls completed, but deterministic branch "
            "materialization retained no hypothesis. The zero outcome must remain reviewable."
        ),
        "representation": "other_typed_relation",
        "scheduled_outcomes": ["contract_pass", "contract_pass"],
        "source_domains": ["scheduled_critic_decision"],
        "synthesis_note": "Post-generation placeholder only; this is not a mathematical idea.",
    }
    branch["branch_id"] = "branch." + canonical_sha256(branch)[:24]
    return branch


def materialize_rejection_placeholders(
    root: Path,
    review: Mapping[str, Any],
    public: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    journal: DurableAttemptJournal,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Represent every empty blinded output without changing its zero-useful-idea count."""

    root = root.resolve()
    for value, label in (
        (review, "review packet"),
        (public, "public receipt"),
        (coordinator, "private coordinator"),
    ):
        _verify_seal(value, label)
    recovered_review = json.loads(json.dumps(review))
    recovered_public = json.loads(json.dumps(public))
    recovered_coordinator = json.loads(json.dumps(coordinator))
    empty_outputs = [item for item in recovered_review["blinded_outputs"] if not item["branches"]]
    if not empty_outputs:
        raise ConfirmatoryRecoveryError("recovery trigger absent: no blinded output is empty")
    for item in empty_outputs:
        if item.get("typed_usable_ideas") != 0:
            raise ConfirmatoryRecoveryError("empty output did not preserve zero typed usable ideas")
        item["branches"] = [
            _rejection_placeholder(item["task_id"], item["blinded_output_id"])
        ]
    _reseal(recovered_review)
    deviation: dict[str, Any] = {
        "schema_version": DEVIATION_SCHEMA,
        "trigger": "schema_valid_calls_produced_zero_materialized_branches",
        "application_scope": "all_empty_blinded_outputs_only",
        "empty_output_count": len(empty_outputs),
        "llm_calls_repeated": False,
        "provider_transport_permitted": False,
        "typed_usable_idea_counts_changed": False,
        "arm_mapping_opened": False,
        "review_scores_observed": False,
        "source_binding": {
            "path": RECOVERY_PATH,
            "sha256": pilot._normalized_file_sha256(root / RECOVERY_PATH),
        },
    }
    deviation["content_sha256"] = canonical_sha256(deviation)
    recovered_public["post_generation_deviation"] = deviation
    recovered_public["blinding"]["review_packet_content_sha256"] = recovered_review[
        "content_sha256"
    ]
    _reseal(recovered_public)
    recovered_coordinator["post_generation_deviation_content_sha256"] = deviation[
        "content_sha256"
    ]
    recovered_coordinator["review_packet_content_sha256"] = recovered_review[
        "content_sha256"
    ]
    recovered_coordinator["public_receipt_content_sha256"] = recovered_public[
        "content_sha256"
    ]
    _reseal(recovered_coordinator)
    confirmatory.validate_public(recovered_review, recovered_public, root)
    confirmatory.validate_coordinator(
        recovered_coordinator, recovered_review, recovered_public, journal
    )
    validate_recovery(
        root,
        recovered_review,
        recovered_public,
        recovered_coordinator,
        journal,
    )
    return recovered_review, recovered_public, recovered_coordinator


def validate_recovery(
    root: Path,
    review: Mapping[str, Any],
    public: Mapping[str, Any],
    coordinator: Mapping[str, Any],
    journal: DurableAttemptJournal,
) -> None:
    validate_recovery_public(root, review, public)
    deviation = public["post_generation_deviation"]
    if coordinator.get("post_generation_deviation_content_sha256") != deviation.get(
        "content_sha256"
    ):
        raise ConfirmatoryRecoveryError("private deviation binding changed")
    message_dispatches = sum(
        event["event_kind"] == "message_dispatch" for event in journal.events
    )
    outcomes = sum(
        event["event_kind"] == "scheduled_call_outcome" for event in journal.events
    )
    if (
        message_dispatches != 96
        or outcomes != 96
        or public.get("attempt_accounting", {}).get("provider_message_attempts") != 96
    ):
        raise ConfirmatoryRecoveryError("recovery changed sealed attempt accounting")


def validate_recovery_public(
    root: Path,
    review: Mapping[str, Any],
    public: Mapping[str, Any],
) -> None:
    """Validate the publishable recovery evidence without the private unblinding journal."""

    confirmatory.validate_public(review, public, root.resolve())
    deviation = public.get("post_generation_deviation", {})
    placeholders = [
        (output, branch)
        for output in review.get("blinded_outputs", [])
        for branch in output.get("branches", [])
        if branch.get("branch_kind") == "all_proposals_rejected_outcome"
    ]
    if (
        deviation.get("schema_version") != DEVIATION_SCHEMA
        or deviation.get("content_sha256")
        != canonical_sha256(
            {key: item for key, item in deviation.items() if key != "content_sha256"}
        )
        or deviation.get("empty_output_count") != len(placeholders)
        or deviation.get("llm_calls_repeated") is not False
        or deviation.get("provider_transport_permitted") is not False
        or deviation.get("typed_usable_idea_counts_changed") is not False
        or deviation.get("arm_mapping_opened") is not False
        or deviation.get("review_scores_observed") is not False
        or deviation.get("source_binding")
        != {
            "path": RECOVERY_PATH,
            "sha256": pilot._normalized_file_sha256(root.resolve() / RECOVERY_PATH),
        }
        or any(output.get("typed_usable_ideas") != 0 for output, _ in placeholders)
        or public.get("attempt_accounting", {}).get("provider_message_attempts") != 96
        or public.get("attempt_accounting", {}).get("scheduled_slots") != 96
    ):
        raise ConfirmatoryRecoveryError("post-generation deviation boundary changed")


def reconstruct_without_transport(
    root: Path,
    journal: DurableAttemptJournal,
    credential_file: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Rebuild the pre-publication objects from outcomes while making transport impossible."""

    original_public = confirmatory.validate_public
    original_coordinator = confirmatory.validate_coordinator

    def no_validation(*_args: Any, **_kwargs: Any) -> None:
        return None

    def deny_transport(*_args: Any, **_kwargs: Any) -> Any:
        raise ConfirmatoryRecoveryError("recovery attempted provider transport")

    confirmatory.validate_public = no_validation
    confirmatory.validate_coordinator = no_validation
    try:
        return confirmatory.run_generation(
            root,
            journal=journal,
            credential_file=credential_file,
            transport=deny_transport,
        )
    finally:
        confirmatory.validate_public = original_public
        confirmatory.validate_coordinator = original_coordinator


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--root", type=Path, default=Path.cwd())
    recover.add_argument("--credential-file", type=Path, required=True)
    recover.add_argument("--journal", type=Path, required=True)
    recover.add_argument("--review-output", type=Path, required=True)
    recover.add_argument("--receipt-output", type=Path, required=True)
    recover.add_argument("--coordinator-output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--review-packet", type=Path, required=True)
    validate.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "validate":
        review = json.loads(args.review_packet.read_text(encoding="utf-8"))
        public = json.loads(args.receipt.read_text(encoding="utf-8"))
        validate_recovery_public(root, review, public)
        print(
            json.dumps(
                {
                    "deviation_content_sha256": public["post_generation_deviation"][
                        "content_sha256"
                    ],
                    "empty_outputs_materialized": public["post_generation_deviation"][
                        "empty_output_count"
                    ],
                    "provider_message_attempts": public["attempt_accounting"][
                        "provider_message_attempts"
                    ],
                    "review_packet_content_sha256": review["content_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    journal_path = confirmatory._private_path(root, args.journal)
    coordinator_path = confirmatory._private_path(root, args.coordinator_output)
    journal = DurableAttemptJournal.load(journal_path)
    review, public, coordinator = reconstruct_without_transport(
        root, journal, args.credential_file.resolve()
    )
    review, public, coordinator = materialize_rejection_placeholders(
        root, review, public, coordinator, journal
    )
    for path, value in (
        (args.review_output, review),
        (args.receipt_output, public),
        (coordinator_path, coordinator),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "confirmatory_generation_eligible": public["release_gate"][
                    "confirmatory_generation_eligible"
                ],
                "deviation_content_sha256": public["post_generation_deviation"][
                    "content_sha256"
                ],
                "empty_outputs_materialized": public["post_generation_deviation"][
                    "empty_output_count"
                ],
                "provider_message_attempts": public["attempt_accounting"][
                    "provider_message_attempts"
                ],
                "review_packet_content_sha256": review["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
