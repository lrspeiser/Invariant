"""Replay retained live piecewise ideas through the current exact executor.

The source core receipt contains credential-free idea lineage captured during an authenticated
Claude campaign.  This module performs no provider calls: it recompiles every retained
``piecewise_relation`` against the current bounded DSL, runs the primary and independent exact
evaluators over the already sealed train/holdout rows, and constructs a resource-matched random
control.  Admission and fit remain execution evidence, never novelty or proof claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .claude_creativity_api import ClaudeCreativityError, ClaudeHypothesis
from .external_creativity_validation import (
    CAMPAIGN_CONFIG_PATH,
    INDEPENDENT_EVALUATOR_PATH,
    PUBLIC_CONFIG_PATH,
    _behavior,
    _candidate_resource_profile,
    _claude_candidate,
    _fraction_text,
    _loss,
    _proof_plan_search,
    independently_predict,
    load_public_benchmarks,
    predict,
    random_controls,
    unseal_targets,
)
from .external_creativity_validation import SOURCE_PATH as EXECUTOR_SOURCE_PATH
from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-retained-piecewise-replay-1.0"
SOURCE_PATH = "src/sigma_theory_compiler/retained_piecewise_replay.py"
SOURCE_RECEIPT_PATH = "runs/math/core-creative-discovery/live-runtime.json"
OUTPUT_PATH = "runs/math/retained-piecewise-replay/receipt.json"
TEST_PATH = "tests/test_retained_piecewise_replay.py"
CONTROL_SEED_NAMESPACE = "invariant.retained-piecewise-replay.2026-08-24"
REPLAY_BUDGET = {
    "maximum_evaluation_operations": 1_000_000,
    "maximum_grammar_depth": 64,
    "maximum_verifier_invocations": 5,
}


class RetainedPiecewiseReplayError(ValueError):
    """The credential-free retained-idea replay failed closed."""


def _normalized_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _rooted_path(root: Path, value: str | Path) -> tuple[Path, str]:
    candidate = Path(value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as error:
        raise RetainedPiecewiseReplayError("replay path escaped the repository root") from error
    return resolved, relative


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RetainedPiecewiseReplayError(f"{label} is not readable canonical JSON") from error
    if not isinstance(value, dict):
        raise RetainedPiecewiseReplayError(f"{label} root is not an object")
    return value


def _sealed_source_receipt(path: Path) -> dict[str, Any]:
    receipt = _load_json(path, "source core receipt")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        receipt.get("content_sha256") != canonical_sha256(body)
        or receipt.get("app_id") != "invariant.core-creative-discovery"
        or receipt.get("claude_runtime", {}).get("authenticated_messages_api_working") is not True
        or receipt.get("claude_runtime", {}).get("completed_calls", 0) < 8
        or receipt.get("claims", {}).get("credential_material_persisted") is not False
    ):
        raise RetainedPiecewiseReplayError("source core receipt is not reusable live evidence")
    return receipt


def _hypothesis(idea: Mapping[str, Any]) -> ClaudeHypothesis:
    fields = {
        key: idea[key]
        for key in (
            "expression",
            "falsifiers",
            "family",
            "hypothesis_id",
            "invariants",
            "known_analogues",
            "llm_origin_assessment",
            "proof_plan",
            "rationale",
            "representation",
            "source_idea_domains",
            "synthesis_note",
        )
    }
    try:
        return ClaudeHypothesis.from_mapping(fields)
    except (ClaudeCreativityError, KeyError, TypeError) as error:
        raise RetainedPiecewiseReplayError("retained hypothesis lineage is malformed") from error


def _predictions(values: Sequence[Any]) -> list[str | None]:
    return [None if value is None else _fraction_text(value) for value in values]


def _source_arithmetic_features(expression: str) -> list[str]:
    features = []
    for present, name in (
        ("%" in expression, "exact_modulo"),
        ("//" in expression, "exact_floor_division"),
        ("round(" in expression, "exact_round_ties_to_even"),
        (" if " in expression and " else " in expression, "exact_conditional"),
        (
            re.search(r"(?<![A-Za-z0-9_])[0-9]+\.[0-9]+", expression) is not None,
            "decimal_to_rational",
        ),
    ):
        if present:
            features.append(name)
    return features


def _replay_row(
    idea: Mapping[str, Any],
    benchmark: Any,
    target: Any,
) -> dict[str, Any]:
    hypothesis = _hypothesis(idea)
    candidate, admission = _claude_candidate(benchmark, hypothesis)
    row: dict[str, Any] = {
        "admission": admission,
        "benchmark_id": benchmark.benchmark_id,
        "hypothesis_id": hypothesis.hypothesis_id,
        "lineage_id": idea["lineage_id"],
        "llm_self_assessed_origin": hypothesis.llm_origin_assessment,
        "original_retention_status": idea["retention_status"],
        "source_blind_id": idea["benchmark_id"],
        "source_arithmetic_features": _source_arithmetic_features(hypothesis.expression),
        "source_expression_sha256": hashlib.sha256(hypothesis.expression.encode()).hexdigest(),
        "target_kind": target.target_kind,
    }
    if candidate is None:
        return row

    rows = (*benchmark.observations, *target.holdout_records)
    primary = predict(candidate, benchmark, rows)
    independent = independently_predict(candidate, benchmark, rows)
    train_count = len(benchmark.observations)
    seed = int(
        hashlib.sha256(
            f"{CONTROL_SEED_NAMESPACE}:{idea['lineage_id']}".encode()
        ).hexdigest()[:16],
        16,
    )
    control = random_controls(
        benchmark,
        {"retained_piecewise_replay": (candidate,)},
        seed,
    )["retained_piecewise_replay"][0]
    candidate_profile = _candidate_resource_profile(
        candidate, benchmark, target, REPLAY_BUDGET
    )
    control_profile = _candidate_resource_profile(control, benchmark, target, REPLAY_BUDGET)
    proof_search = _proof_plan_search(
        candidate,
        target,
        REPLAY_BUDGET["maximum_verifier_invocations"],
    )
    row["execution"] = {
        "behavior": _behavior(candidate, benchmark, rows),
        "candidate_id": candidate.candidate_id,
        "candidate_resource_profile": candidate_profile,
        "control_behavior": _behavior(control, benchmark, rows),
        "control_candidate_id": control.candidate_id,
        "control_resource_profile": control_profile,
        "holdout_loss": _fraction_text(_loss(primary[train_count:], target.holdout_records)),
        "independent_predictions": _predictions(independent),
        "normalized_expression": candidate.expression,
        "primary_independent_exact_agreement": primary == independent,
        "primary_predictions": _predictions(primary),
        "proof_plan_search": proof_search,
        "resource_profile_exact_match": candidate_profile == control_profile,
        "train_loss": _fraction_text(_loss(primary[:train_count], benchmark.observations)),
        "undefined_rows": sum(value is None for value in primary),
    }
    return row


def build_receipt(
    root: Path,
    source_receipt: str | Path = SOURCE_RECEIPT_PATH,
) -> dict[str, Any]:
    root = root.resolve()
    source_path, source_relative = _rooted_path(root, source_receipt)
    source = _sealed_source_receipt(source_path)
    public, benchmarks = load_public_benchmarks(root)
    targets = unseal_targets(root, public, benchmarks)
    benchmark_by_id = {
        identifier: item
        for item in benchmarks
        for identifier in (item.benchmark_id, item.blind_id)
    }
    target_by_id = {item.benchmark_id: item for item in targets}
    ideas = source.get("idea_lineage_archive", {}).get("ideas")
    if not isinstance(ideas, list):
        raise RetainedPiecewiseReplayError("source idea archive is missing")
    selected = [item for item in ideas if item.get("representation") == "piecewise_relation"]
    if not selected:
        raise RetainedPiecewiseReplayError("source receipt contains no retained piecewise ideas")
    replays = []
    for idea in selected:
        benchmark_id = idea.get("benchmark_id")
        benchmark = benchmark_by_id.get(benchmark_id)
        target = None if benchmark is None else target_by_id.get(benchmark.benchmark_id)
        if benchmark is None or target is None:
            raise RetainedPiecewiseReplayError("retained idea lost its sealed benchmark binding")
        replays.append(_replay_row(idea, benchmark, target))
    replays.sort(key=lambda item: item["lineage_id"])
    admitted = [item for item in replays if "execution" in item]
    origins: dict[str, int] = {}
    feature_counts: dict[str, int] = {}
    for item in replays:
        origin = item["llm_self_assessed_origin"]
        origins[origin] = origins.get(origin, 0) + 1
        for feature in item["source_arithmetic_features"]:
            feature_counts[feature] = feature_counts.get(feature, 0) + 1
    body = {
        "schema_version": SCHEMA_VERSION,
        "source_bindings": {
            "campaign_config": {
                "path": CAMPAIGN_CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / CAMPAIGN_CONFIG_PATH),
            },
            "executor_source": {
                "path": EXECUTOR_SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / EXECUTOR_SOURCE_PATH),
            },
            "independent_evaluator_source": {
                "path": INDEPENDENT_EVALUATOR_PATH,
                "sha256": _normalized_file_sha256(root / INDEPENDENT_EVALUATOR_PATH),
            },
            "public_benchmarks": {
                "path": PUBLIC_CONFIG_PATH,
                "sha256": _normalized_file_sha256(root / PUBLIC_CONFIG_PATH),
            },
            "replay_source": {
                "path": SOURCE_PATH,
                "sha256": _normalized_file_sha256(root / SOURCE_PATH),
            },
            "source_core_receipt": {
                "idea_lineage_archive_sha256": source["idea_lineage_archive"][
                    "content_sha256"
                ],
                "live_evidence_sha256": source["claude_runtime"]["evidence"][
                    "content_sha256"
                ],
                "path": source_relative,
            },
            "tests": {
                "path": TEST_PATH,
                "sha256": _normalized_file_sha256(root / TEST_PATH),
            },
        },
        "chronology": {
            "new_provider_calls": 0,
            "replayed_after_live_generation": True,
            "source_outputs_are_credential_free": True,
            "uses_retained_sanitized_lineage_only": True,
        },
        "replay_budget": dict(REPLAY_BUDGET),
        "replays": replays,
        "summary": {
            "admitted_by_current_executor": len(admitted),
            "exact_primary_independent_agreements": sum(
                item["execution"]["primary_independent_exact_agreement"] for item in admitted
            ),
            "extended_arithmetic_feature_counts": dict(sorted(feature_counts.items())),
            "llm_self_assessed_origin_counts": dict(sorted(origins.items())),
            "resource_matched_controls": sum(
                item["execution"]["resource_profile_exact_match"] for item in admitted
            ),
            "retained_piecewise_ideas": len(replays),
            "train_exact_holdout_failed": sum(
                item["execution"]["train_loss"] == "0"
                and item["execution"]["holdout_loss"] != "0"
                for item in admitted
            ),
            "zero_holdout_loss_bounded_unknown": sum(
                item["target_kind"] == "bounded_unknown"
                and item["execution"]["holdout_loss"] == "0"
                for item in admitted
            ),
            "zero_holdout_loss_candidates": sum(
                item["execution"]["holdout_loss"] == "0" for item in admitted
            ),
            "zero_train_loss_candidates": sum(
                item["execution"]["train_loss"] == "0" for item in admitted
            ),
            "status": "PASS_RETAINED_PIECEWISE_REPLAY",
        },
        "claim_boundary": {
            "executor_admission_establishes_correctness": False,
            "fit_establishes_general_formula": False,
            "llm_origin_assessment_establishes_novelty": False,
            "replay_establishes_proof": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    if receipt.get("schema_version") != SCHEMA_VERSION:
        raise RetainedPiecewiseReplayError("replay receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise RetainedPiecewiseReplayError("replay receipt content seal changed")
    chronology = receipt.get("chronology", {})
    claims = receipt.get("claim_boundary", {})
    summary = receipt.get("summary", {})
    if (
        chronology.get("new_provider_calls") != 0
        or chronology.get("uses_retained_sanitized_lineage_only") is not True
        or claims
        != {
            "executor_admission_establishes_correctness": False,
            "fit_establishes_general_formula": False,
            "llm_origin_assessment_establishes_novelty": False,
            "replay_establishes_proof": False,
        }
        or summary.get("status") != "PASS_RETAINED_PIECEWISE_REPLAY"
        or summary.get("retained_piecewise_ideas", 0) < 1
        or summary.get("admitted_by_current_executor")
        != summary.get("exact_primary_independent_agreements")
        or summary.get("admitted_by_current_executor")
        != summary.get("resource_matched_controls")
    ):
        raise RetainedPiecewiseReplayError("replay receipt policy or coverage changed")
    source = receipt.get("source_bindings", {}).get("source_core_receipt", {})
    if not isinstance(source, Mapping) or not isinstance(source.get("path"), str):
        raise RetainedPiecewiseReplayError("source core receipt binding is malformed")
    expected = build_receipt(root, source["path"])
    if dict(receipt) != expected:
        raise RetainedPiecewiseReplayError("replay receipt does not reproduce exactly")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="replay retained piecewise ideas")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--source", type=Path, default=Path(SOURCE_RECEIPT_PATH))
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate", help="validate a stored replay receipt")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    arguments = parser.parse_args(argv)
    root = arguments.root.resolve()
    if arguments.command == "run":
        receipt = build_receipt(root, arguments.source)
        output, _ = _rooted_path(root, arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt_path, _ = _rooted_path(root, arguments.receipt)
        receipt = _load_json(receipt_path, "stored replay receipt")
        validate_receipt(receipt, root)
    print(json.dumps(receipt["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())


__all__ = [
    "OUTPUT_PATH",
    "RetainedPiecewiseReplayError",
    "build_receipt",
    "main",
    "validate_receipt",
]
