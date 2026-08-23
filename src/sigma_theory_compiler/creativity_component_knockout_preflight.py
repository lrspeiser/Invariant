"""Seal the four preregistered creativity component-knockout schedules without paid calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/creativity_component_knockouts.json"
RUNNER_PATH = "src/sigma_theory_compiler/creativity_component_knockout_preflight.py"
OUTPUT_PATH = "runs/math/creativity-component-knockouts/preflight.json"
CONFIG_SCHEMA = "invariant-creativity-component-knockout-config-1.0"
RECEIPT_SCHEMA = "invariant-creativity-component-knockout-preflight-1.0"
SUITE_ID = "creativity-first-component-knockouts-001"
FEATURES = (
    "expanded_typed_grammar",
    "independent_proof_recombination",
    "origin_lineage_labels",
    "verifier_failure_non_pruning",
)
REFERENCE_ARM = "full_creativity_first"
KNOCKOUTS = {
    "minus_expanded_grammar": "expanded_typed_grammar",
    "minus_independent_proof_recombination": "independent_proof_recombination",
    "minus_lineage_labels": "origin_lineage_labels",
    "minus_non_pruning": "verifier_failure_non_pruning",
}
ROLES = ("proposer", "critic")
_HEX = frozenset("0123456789abcdef")


class ComponentKnockoutPreflightError(ValueError):
    """The knockout design, budget match, schedule, or source binding failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ComponentKnockoutPreflightError(f"{label} keys changed")


def _sha(value: Any, label: str, *, prefix: bool = False) -> str:
    if prefix:
        if not isinstance(value, str) or not value.startswith("sha256:"):
            raise ComponentKnockoutPreflightError(f"{label} is not a sha256-prefixed digest")
        value = value.removeprefix("sha256:")
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ComponentKnockoutPreflightError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ComponentKnockoutPreflightError(f"{label} escapes the repository") from error
    if not path.is_file():
        raise ComponentKnockoutPreflightError(f"{label} is missing")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ComponentKnockoutPreflightError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise ComponentKnockoutPreflightError(f"{label} is not a JSON object")
    return value


def _normalized_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _canonical_file_sha256(path: Path, label: str) -> str:
    return canonical_sha256(_read_json(path, label))


def _validate_binding(root: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    _strict(binding, {"canonical_sha256", "path"}, label)
    expected = _sha(binding["canonical_sha256"], f"{label} canonical hash")
    path = _under(root, str(binding["path"]), label)
    if _canonical_file_sha256(path, label) != expected:
        raise ComponentKnockoutPreflightError(f"{label} canonical binding changed")
    return _read_json(path, label)


def _validate_generation_packet(
    root: Path, binding: Mapping[str, Any]
) -> dict[str, Any]:
    _strict(binding, {"content_sha256", "path", "required_tasks"}, "generation packet binding")
    path = _under(root, str(binding["path"]), "generation packet")
    value = _read_json(path, "generation packet")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("schema_version") != "invariant-rotating-external-generation-packet-1.0"
        or value.get("content_sha256") != canonical_sha256(body)
        or value.get("content_sha256")
        != _sha(binding["content_sha256"], "generation packet content hash")
        or binding["required_tasks"] != 24
        or len(value.get("tasks", [])) != 24
    ):
        raise ComponentKnockoutPreflightError("generation packet seal or task coverage changed")
    serialized = json.dumps(value, sort_keys=True).lower()
    if any(term in serialized for term in ('"holdout"', '"source_uri"', '"source_id"')):
        raise ComponentKnockoutPreflightError("generation packet leaked sealed target material")
    task_ids = [task.get("task_id") for task in value["tasks"]]
    if any(not isinstance(item, str) or not item for item in task_ids) or len(set(task_ids)) != 24:
        raise ComponentKnockoutPreflightError("generation packet task identities changed")
    return value


def validate_config(value: Mapping[str, Any], root: Path) -> dict[str, Any]:
    root = root.resolve()
    _strict(
        value,
        {
            "arms",
            "attempt_policy",
            "confirmatory_binding",
            "execution_gate",
            "experiments",
            "feature_order",
            "generation_packet",
            "matched_resource_budget",
            "protocol_binding",
            "review_policy",
            "schema_version",
            "suite_id",
        },
        "component-knockout config",
    )
    if value["schema_version"] != CONFIG_SCHEMA or value["suite_id"] != SUITE_ID:
        raise ComponentKnockoutPreflightError("component-knockout config identity changed")
    if tuple(value["feature_order"]) != FEATURES:
        raise ComponentKnockoutPreflightError("component-knockout feature order changed")

    protocol = _validate_binding(root, value["protocol_binding"], "ablation protocol")
    confirmatory = _validate_binding(
        root, value["confirmatory_binding"], "confirmatory generation config"
    )
    generation = _validate_generation_packet(root, value["generation_packet"])
    expected_arms = {REFERENCE_ARM, *KNOCKOUTS}
    if set(protocol.get("arms", [])) != {"baseline", *expected_arms}:
        raise ComponentKnockoutPreflightError("ablation protocol arm registry changed")
    if set(value["arms"]) != expected_arms:
        raise ComponentKnockoutPreflightError("component-knockout arm coverage changed")

    full_flags = {feature: True for feature in FEATURES}
    for arm, arm_value in value["arms"].items():
        _strict(
            arm_value,
            {"enforcement", "feature_flags", "instruction_delta"},
            f"component-knockout arm {arm}",
        )
        flags = arm_value["feature_flags"]
        _strict(flags, set(FEATURES), f"component-knockout feature flags {arm}")
        if any(not isinstance(item, bool) for item in flags.values()):
            raise ComponentKnockoutPreflightError("component-knockout feature flag is not boolean")
        if not isinstance(arm_value["enforcement"], str) or not arm_value["enforcement"].strip():
            raise ComponentKnockoutPreflightError("component-knockout enforcement is empty")
        if (
            not isinstance(arm_value["instruction_delta"], str)
            or not arm_value["instruction_delta"].strip()
        ):
            raise ComponentKnockoutPreflightError("component-knockout instruction delta is empty")
        expected_flags = dict(full_flags)
        if arm in KNOCKOUTS:
            expected_flags[KNOCKOUTS[arm]] = False
        if flags != expected_flags:
            raise ComponentKnockoutPreflightError(
                f"component-knockout arm {arm} does not remove exactly its registered feature"
            )

    experiments = value["experiments"]
    if not isinstance(experiments, list) or len(experiments) != 4:
        raise ComponentKnockoutPreflightError("component-knockout experiment count changed")
    seen_arms: set[str] = set()
    seen_ids: set[str] = set()
    for experiment in experiments:
        _strict(
            experiment,
            {
                "arm_order_seed",
                "experiment_id",
                "knockout_arm",
                "reference_arm",
                "removed_feature",
            },
            "component-knockout experiment",
        )
        experiment_id = experiment["experiment_id"]
        arm = experiment["knockout_arm"]
        if (
            not isinstance(experiment_id, str)
            or not experiment_id.startswith("creativity-knockout-")
            or experiment_id in seen_ids
            or experiment["reference_arm"] != REFERENCE_ARM
            or arm not in KNOCKOUTS
            or arm in seen_arms
            or experiment["removed_feature"] != KNOCKOUTS[arm]
        ):
            raise ComponentKnockoutPreflightError("component-knockout experiment registry changed")
        _sha(experiment["arm_order_seed"], "component-knockout arm-order seed", prefix=True)
        seen_ids.add(experiment_id)
        seen_arms.add(arm)
    if seen_arms != set(KNOCKOUTS):
        raise ComponentKnockoutPreflightError("component-knockout experiment coverage changed")

    resource = value["matched_resource_budget"]
    if resource != confirmatory.get("matched_resource_budget"):
        raise ComponentKnockoutPreflightError("component-knockout resource budget is not matched")
    if value["attempt_policy"] != confirmatory.get("attempt_policy"):
        raise ComponentKnockoutPreflightError("component-knockout attempt policy diverged")
    claude = confirmatory.get("claude", {})
    if (
        resource.get("calls_per_task") != len(ROLES)
        or resource.get("tokens_per_arm") != claude.get("maximum_total_tokens_per_arm")
        or claude.get("maximum_scheduled_calls_per_arm") != 48
        or any(
            not isinstance(item, int) or isinstance(item, bool) or item < 1
            for item in resource.values()
        )
    ):
        raise ComponentKnockoutPreflightError("component-knockout resource ceilings changed")
    review = value["review_policy"]
    _strict(
        review,
        {
            "arm_identity_hidden_until_both_reviews_sealed",
            "axes",
            "minimum_named_reviewers",
            "rating_scale",
            "reviewers_must_be_distinct",
            "useful_threshold_each_axis",
        },
        "component-knockout review policy",
    )
    confirmatory_review = confirmatory.get("review", {})
    if (
        review["axes"] != confirmatory_review.get("axes")
        or review["minimum_named_reviewers"] != confirmatory_review.get("minimum_named_reviewers")
        or review["rating_scale"] != confirmatory_review.get("rating_scale")
        or review["useful_threshold_each_axis"]
        != confirmatory_review.get("useful_threshold_each_axis")
        or review["reviewers_must_be_distinct"] is not True
        or review["arm_identity_hidden_until_both_reviews_sealed"] is not True
    ):
        raise ComponentKnockoutPreflightError("component-knockout review policy diverged")
    if value["execution_gate"] != {
        "credential_access_during_preflight": False,
        "live_provider_calls_during_preflight": 0,
        "paid_execution_requires_explicit_separate_authorization": True,
    }:
        raise ComponentKnockoutPreflightError("component-knockout preflight execution gate weakened")
    return {"confirmatory": confirmatory, "generation": generation, "protocol": protocol}


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    root = root.resolve()
    value = _read_json(_under(root, CONFIG_PATH, "component-knockout config"), "component-knockout config")
    dependencies = validate_config(value, root)
    return value, dependencies


def _ordered_arms(experiment: Mapping[str, Any], task_id: str, role: str) -> list[str]:
    arms = [experiment["reference_arm"], experiment["knockout_arm"]]
    return sorted(
        arms,
        key=lambda arm: canonical_sha256(
            {
                "arm": arm,
                "experiment_id": experiment["experiment_id"],
                "role": role,
                "seed": experiment["arm_order_seed"],
                "task_id": task_id,
            }
        ),
    )


def _schedule(config: Mapping[str, Any], generation: Mapping[str, Any]) -> dict[str, Any]:
    resources = config["matched_resource_budget"]
    slots: list[dict[str, Any]] = []
    per_experiment = []
    ordinal = 0
    tasks = sorted(generation["tasks"], key=lambda task: task["task_id"])
    for experiment in config["experiments"]:
        experiment_slots = []
        for task in tasks:
            task_id = task["task_id"]
            for role in ROLES:
                for arm_position, arm in enumerate(_ordered_arms(experiment, task_id, role)):
                    body = {
                        "arm": arm,
                        "arm_position": arm_position,
                        "experiment_id": experiment["experiment_id"],
                        "feature_flags_sha256": canonical_sha256(
                            config["arms"][arm]["feature_flags"]
                        ),
                        "ordinal": ordinal,
                        "resource_budget_sha256": canonical_sha256(resources),
                        "role": role,
                        "task_id": task_id,
                    }
                    slot = {
                        **body,
                        "slot_id": "slot." + canonical_sha256(body)[:24],
                    }
                    slot["slot_sha256"] = canonical_sha256(slot)
                    slots.append(slot)
                    experiment_slots.append(slot)
                    ordinal += 1
        counts = Counter((slot["arm"], slot["role"]) for slot in experiment_slots)
        if set(counts.values()) != {24} or len(counts) != 4 or len(experiment_slots) != 96:
            raise ComponentKnockoutPreflightError("component-knockout experiment schedule is unbalanced")
        per_experiment.append(
            {
                "experiment_id": experiment["experiment_id"],
                "knockout_arm": experiment["knockout_arm"],
                "removed_feature": experiment["removed_feature"],
                "scheduled_slots": len(experiment_slots),
                "schedule_root_sha256": canonical_sha256(
                    [slot["slot_sha256"] for slot in experiment_slots]
                ),
            }
        )
    if len(slots) != 384 or len({slot["slot_id"] for slot in slots}) != 384:
        raise ComponentKnockoutPreflightError("component-knockout global schedule coverage changed")
    return {
        "experiments": per_experiment,
        "role_order": list(ROLES),
        "schedule_root_sha256": canonical_sha256([slot["slot_sha256"] for slot in slots]),
        "slot_sha256s": [slot["slot_sha256"] for slot in slots],
        "task_ids_sha256": canonical_sha256([task["task_id"] for task in tasks]),
        "total_scheduled_slots": len(slots),
    }


def _source_bindings(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = {
        "ablation_protocol": config["protocol_binding"]["path"],
        "component_knockout_config": CONFIG_PATH,
        "confirmatory_generation_config": config["confirmatory_binding"]["path"],
        "generation_packet": config["generation_packet"]["path"],
        "preflight_runner": RUNNER_PATH,
    }
    return {
        name: {"path": path, "sha256": _normalized_file_sha256(_under(root, path, name))}
        for name, path in sorted(paths.items())
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, dependencies = load_config(root)
    schedule = _schedule(config, dependencies["generation"])
    experiment_count = len(config["experiments"])
    arm_count = 2 * experiment_count
    maximum_tokens = arm_count * config["matched_resource_budget"]["tokens_per_arm"]
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "suite_id": SUITE_ID,
        "source_bindings": _source_bindings(root, config),
        "design": {
            "experiments": experiment_count,
            "feature_order": list(FEATURES),
            "one_feature_removed_per_experiment": True,
            "reference_arm": REFERENCE_ARM,
            "tasks_per_experiment": 24,
        },
        "resource_accounting": {
            "matched_resource_budget": config["matched_resource_budget"],
            "maximum_provider_calls_if_all_four_runs_are_authorized": schedule[
                "total_scheduled_slots"
            ],
            "maximum_total_tokens_if_all_four_runs_are_authorized": maximum_tokens,
            "preflight_credential_accesses": 0,
            "preflight_provider_calls": 0,
        },
        "schedule": schedule,
        "release_gate": {
            "component_knockout_live_runs_complete": False,
            "named_blinded_reviews_complete": False,
            "paid_execution_separately_authorized": False,
            "status": "PASS_PREFLIGHT_LIVE_EXECUTION_NOT_RUN",
        },
        "claims": {
            "component_knockouts_complete": False,
            "credential_accessed_during_preflight": False,
            "literature_novelty_established": False,
            "live_provider_calls_performed": False,
            "more_creative_established": False,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    validate_receipt(receipt, root, rebuild=False)
    return receipt


def validate_receipt(
    value: Mapping[str, Any], root: Path, *, rebuild: bool = True
) -> None:
    root = root.resolve()
    _strict(
        value,
        {
            "claims",
            "content_sha256",
            "design",
            "release_gate",
            "resource_accounting",
            "schedule",
            "schema_version",
            "source_bindings",
            "suite_id",
        },
        "component-knockout preflight receipt",
    )
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value["schema_version"] != RECEIPT_SCHEMA
        or value["suite_id"] != SUITE_ID
        or value["content_sha256"] != canonical_sha256(body)
    ):
        raise ComponentKnockoutPreflightError("component-knockout preflight seal changed")
    config, dependencies = load_config(root)
    expected_schedule = _schedule(config, dependencies["generation"])
    if value["schedule"] != expected_schedule:
        raise ComponentKnockoutPreflightError("component-knockout preflight schedule changed")
    if value["source_bindings"] != _source_bindings(root, config):
        raise ComponentKnockoutPreflightError("component-knockout preflight source binding changed")
    if value["release_gate"] != {
        "component_knockout_live_runs_complete": False,
        "named_blinded_reviews_complete": False,
        "paid_execution_separately_authorized": False,
        "status": "PASS_PREFLIGHT_LIVE_EXECUTION_NOT_RUN",
    } or value["claims"] != {
        "component_knockouts_complete": False,
        "credential_accessed_during_preflight": False,
        "literature_novelty_established": False,
        "live_provider_calls_performed": False,
        "more_creative_established": False,
    }:
        raise ComponentKnockoutPreflightError("component-knockout preflight claim boundary changed")
    if rebuild:
        expected = build_receipt(root)
        if value != expected:
            raise ComponentKnockoutPreflightError(
                "component-knockout preflight does not rebuild from current sources"
            )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", type=Path, default=Path.cwd())
    build.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "build":
        receipt = build_receipt(root)
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
        receipt = _read_json(receipt_path, "component-knockout preflight receipt")
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "experiments": receipt["design"]["experiments"],
                "preflight_provider_calls": receipt["resource_accounting"][
                    "preflight_provider_calls"
                ],
                "scheduled_slots": receipt["schedule"]["total_scheduled_slots"],
                "status": receipt["release_gate"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
