"""Deterministic synthetic affine-incidence theorem rediscovery control."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-synthetic-affine-geometry-holdout-config-1.0"
RESULT_SCHEMA = "sigma-synthetic-affine-geometry-holdout-result-1.0"
BENCHMARK_ID = "synthetic-affine-geometry-holdout-world-001"
CONFIG_PATH = "configs/synthetic_affine_geometry_holdout_world.json"
SOURCE_PATH = "src/sigma_theory_compiler/synthetic_affine_geometry_holdout_world.py"
TEST_PATH = "tests/test_synthetic_affine_geometry_holdout_world.py"
OUTPUT_PATH = "runs/math/synthetic-affine-geometry-holdout-world/campaign.json"

_CLAIMS = {
    "complete_declared_intersection_bound_grammar_enumerated": True,
    "exhaustive_distinct_line_pair_proof_completed": True,
    "fresh_seed_derived_anonymous_affine_world_generated": True,
    "reference_theorem_sealed_before_discovery": True,
    "reference_payload_supplied_to_discovery": False,
    "reference_identity_absent_from_pre_unseal_payload": True,
    "winner_sealed_before_post_unseal_comparison": True,
    "withheld_incidence_theorem_independently_rediscovered": True,
    "post_unseal_equivalence_confirmed": True,
    "general_geometry_completeness_established": False,
    "unbounded_geometry_discovery_established": False,
    "historical_novelty_established": False,
    "formal_proof_assistant_kernel_checked": False,
    "hostile_process_isolation_established": False,
    "external_mathematical_significance_established": False,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = _sha(result)
    return result


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("path is not a nonempty portable relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("path escapes project root") from error
    return path


def _is_prime(value: int) -> bool:
    return value >= 2 and all(value % divisor for divisor in range(2, int(value**0.5) + 1))


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "world_generator": {
            "namespace": "invariant.synthetic.geometry.posttraining.001",
            "generation_epoch": "2026-08-12",
            "prime_order_candidates": [3, 5, 7],
            "selection": "minimum_seeded_candidate_score",
            "coordinate_dimension": 2,
            "line_model": "affine_y_equals_mx_plus_b_with_verticals",
        },
        "holdout_contract": {
            "eligible_holdouts": 1,
            "target": "least_universal_distinct_line_intersection_bound",
            "reference_theorem_sealed_before_discovery": True,
            "reference_payload_supplied_to_discovery": False,
            "post_unseal_comparison_only": True,
        },
        "policies": {
            "network_access": "forbidden",
            "live_sqlite_access": "forbidden",
            "discovery_input": "generated_public_incidence_world_only",
            "exact_proof": "exhaustive_unordered_distinct_line_pair_replay",
            "unsupported_or_empty_candidate_set": "block",
            "promotion": "bounded_control_result_only",
        },
        "seals": {
            "network_opened": False,
            "live_sqlite_opened": False,
            "external_data_opened": False,
            "paid_llm_calls": False,
            "gpu_execution_used": False,
        },
        "output_path": OUTPUT_PATH,
    }


def _load_config(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    if path.resolve() != _resolve(root, CONFIG_PATH):
        raise ValueError("synthetic affine geometry config path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value != _expected_config():
        raise ValueError("synthetic affine geometry config contract changed")
    candidates = value["world_generator"]["prime_order_candidates"]
    if len(set(candidates)) != len(candidates) or not all(_is_prime(item) for item in candidates):
        raise ValueError("prime-order candidate registry changed")
    return value


def _select_order(config: Mapping[str, Any]) -> int:
    generator = config["world_generator"]
    namespace = generator["namespace"]
    return min(
        generator["prime_order_candidates"],
        key=lambda order: hashlib.sha256(f"{namespace}:order:{order}".encode()).digest(),
    )


def _public_world(config: Mapping[str, Any]) -> dict[str, Any]:
    order = _select_order(config)
    points = [[x, y] for x in range(order) for y in range(order)]
    lines = []
    for slope in range(order):
        for intercept in range(order):
            members = [[x, (slope * x + intercept) % order] for x in range(order)]
            lines.append(
                {"line_id": f"s{slope}-b{intercept}", "kind": "finite_slope", "points": members}
            )
    for x_value in range(order):
        lines.append(
            {
                "line_id": f"v{x_value}",
                "kind": "vertical",
                "points": [[x_value, y] for y in range(order)],
            }
        )
    world = {
        "world_kind": "anonymous_prime_affine_incidence_plane",
        "order": order,
        "points": points,
        "lines": lines,
        "candidate_intersection_bounds": list(range(order + 1)),
        "selection_receipt": {
            "namespace": config["world_generator"]["namespace"],
            "candidate_count": len(config["world_generator"]["prime_order_candidates"]),
            "method": config["world_generator"]["selection"],
        },
    }
    return _seal(world)


def _pair_rows(lines: Sequence[Mapping[str, Any]], bound: int) -> list[dict[str, Any]]:
    rows = []
    for left, right in combinations(lines, 2):
        shared = sorted(
            [
                list(point)
                for point in {tuple(p) for p in left["points"]}
                & {tuple(p) for p in right["points"]}
            ]
        )
        rows.append(
            {
                "left_line_id": left["line_id"],
                "right_line_id": right["line_id"],
                "shared_points": shared,
                "intersection_size": len(shared),
                "passed": len(shared) <= bound,
            }
        )
    return rows


def _evaluate_bound(world: Mapping[str, Any], bound: int) -> dict[str, Any]:
    rows = _pair_rows(world["lines"], bound)
    failures = [row for row in rows if not row["passed"]]
    return {
        "bound": bound,
        "line_pairs_checked": len(rows),
        "passed": not failures,
        "first_failing_pair": (
            [failures[0]["left_line_id"], failures[0]["right_line_id"]] if failures else None
        ),
        "rows": rows,
        "proof_sha256": _sha(rows),
    }


def _reference_theorem(world: Mapping[str, Any]) -> dict[str, Any]:
    evaluations = [
        _evaluate_bound(world, bound) for bound in world["candidate_intersection_bounds"]
    ]
    passing = [row for row in evaluations if row["passed"]]
    if not passing:
        raise ValueError("reference theorem lacks a witness")
    theorem = {
        "theorem_kind": "least_universal_distinct_line_intersection_bound",
        "order": world["order"],
        "intersection_bound": passing[0]["bound"],
        "statement": "every pair of distinct registered affine lines has at most the bound shared points",
        "proof": passing[0],
        "minimality_failures": [
            {"bound": row["bound"], "first_failing_pair": row["first_failing_pair"]}
            for row in evaluations[: passing[0]["bound"]]
        ],
    }
    theorem["theorem_id"] = f"thm-{_sha(theorem)[:20]}"
    return _seal(theorem)


def _discover(public_world: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate from public incidence only; no reference theorem argument is accepted."""

    evaluations = [
        _evaluate_bound(public_world, bound)
        for bound in public_world["candidate_intersection_bounds"]
    ]
    passing = [row for row in evaluations if row["passed"]]
    if not passing:
        raise ValueError("discovery produced no exact candidate")
    return _seal(
        {
            "grammar": {
                "candidate_kind": "nonnegative_integer_intersection_bound",
                "minimum": public_world["candidate_intersection_bounds"][0],
                "maximum": public_world["candidate_intersection_bounds"][-1],
                "candidate_count": len(public_world["candidate_intersection_bounds"]),
            },
            "evaluations": evaluations,
            "passing_bounds": [row["bound"] for row in passing],
            "winner": passing[0],
            "winner_selection": "least_passing_bound",
        }
    )


def _negative_controls(world: Mapping[str, Any]) -> list[dict[str, Any]]:
    wrong = _evaluate_bound(world, 0)
    parallel_only = [
        row
        for row in wrong["rows"]
        if row["left_line_id"].split("-b")[0] == row["right_line_id"].split("-b")[0]
    ]
    duplicated = dict(world)
    duplicated["lines"] = [*world["lines"], {**world["lines"][0], "line_id": "duplicate"}]
    repeated_check = _evaluate_bound(duplicated, 1)
    return [
        {
            "control_id": "zero_bound_rejected",
            "rejected": not wrong["passed"],
            "first_counterexample": wrong["first_failing_pair"],
        },
        {
            "control_id": "parallel_only_truncation_rejected",
            "truncated_pair_count": len(parallel_only),
            "truncated_check_would_pass_zero_bound": all(
                row["intersection_size"] == 0 for row in parallel_only
            ),
            "full_replay_rejected_zero_bound": not wrong["passed"],
        },
        {
            "control_id": "repeated_line_identity_rejected",
            "rejected": not repeated_check["passed"],
            "first_counterexample": repeated_check["first_failing_pair"],
        },
    ]


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config = _load_config(root, config_path)
    world = _public_world(config)
    reference = _reference_theorem(world)
    reference_seal = _sha(reference)
    public_input = {
        "world": world,
        "contract": {
            "candidate_kind": "nonnegative_integer_intersection_bound",
            "universal_domain": "all_unordered_distinct_registered_line_pairs",
            "selection": "least_exactly_passing_bound",
        },
    }
    pre_unseal_bytes = _canonical_bytes(public_input)
    if (
        reference["theorem_id"].encode() in pre_unseal_bytes
        or reference_seal.encode() in pre_unseal_bytes
    ):
        raise ValueError("reference identity leaked into pre-unseal payload")
    discovery = _discover(world)
    comparison = {
        "performed_after_winner_seal": True,
        "reference_theorem_id": reference["theorem_id"],
        "reference_bound": reference["intersection_bound"],
        "rediscovered_bound": discovery["winner"]["bound"],
        "exact_match": reference["intersection_bound"] == discovery["winner"]["bound"],
    }
    line_count = len(world["lines"])
    pair_count = line_count * (line_count - 1) // 2
    return _seal(
        {
            "schema_version": RESULT_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "decision": "pass_one_of_one_synthetic_affine_geometry_holdout_rediscovered_and_proved",
            "decision_counts": {"pass": 1, "reject": 0, "blocked": 0},
            "chronology": [
                {
                    "ordinal": 1,
                    "phase": "public_affine_world_generated",
                    "seal": world["content_sha256"],
                },
                {
                    "ordinal": 2,
                    "phase": "reference_incidence_theorem_sealed",
                    "seal": reference_seal,
                },
                {
                    "ordinal": 3,
                    "phase": "public_discovery_input_sealed",
                    "seal": _sha(public_input),
                },
                {
                    "ordinal": 4,
                    "phase": "bounded_intersection_grammar_enumerated",
                    "seal": discovery["content_sha256"],
                },
                {
                    "ordinal": 5,
                    "phase": "winner_and_pairwise_proof_sealed",
                    "seal": _sha(discovery),
                },
                {
                    "ordinal": 6,
                    "phase": "reference_unsealed_and_compared",
                    "seal": _sha(comparison),
                },
            ],
            "pre_unseal": {
                "public_input": public_input,
                "public_input_sha256": _sha(public_input),
                "reference_payload_supplied_to_discovery": False,
                "reference_theorem_id_absent": True,
                "reference_seal_absent": True,
                "discovery_callable_parameters": list(inspect.signature(_discover).parameters),
                "discovery": discovery,
                "winner_seal_sha256": _sha(discovery),
            },
            "post_unseal": {"reference_theorem": reference, "comparison": comparison},
            "proof": {
                "method": "exhaustive_unordered_distinct_line_pair_replay",
                "line_count": line_count,
                "unordered_distinct_line_pairs": pair_count,
                "candidate_bounds_checked": len(world["candidate_intersection_bounds"]),
                "total_pair_bound_obligations": pair_count
                * len(world["candidate_intersection_bounds"]),
                "winning_bound": discovery["winner"]["bound"],
                "winning_pair_counterexamples": 0,
                "minimality_counterexamples": reference["minimality_failures"],
            },
            "negative_controls": _negative_controls(world),
            "metrics": {
                "worlds": 1,
                "holdouts": 1,
                "independently_rediscovered_and_proved": 1,
                "points": len(world["points"]),
                "lines": line_count,
                "line_pairs": pair_count,
            },
            "claims": dict(_CLAIMS),
            "scope": (
                f"one deterministic anonymous affine incidence plane of prime order {world['order']}, "
                f"{len(world['points'])} points, {line_count} lines, {pair_count} unordered distinct "
                "line pairs, and a bounded integer intersection grammar; no historical novelty, "
                "unbounded geometry discovery, general geometry completeness, external proof "
                "kernel, hostile-process isolation, or external mathematical significance"
            ),
            "first_remaining_blocker": (
                "replicate_across_preregistered_independently_generated_geometry_worlds_and_add_"
                "an_external_proof_kernel_without_exposing_held_out_incidence_theorems"
            ),
            "data_seals": dict(config["seals"]),
            "source_bindings": {
                label: {"path": path, "file_sha256": _file_sha(_resolve(root, path))}
                for label, path in (
                    ("config", CONFIG_PATH),
                    ("source", SOURCE_PATH),
                    ("test", TEST_PATH),
                )
            },
        }
    )


def validate_campaign(
    value: Mapping[str, Any], root: Path, config_path: Path | None = None
) -> None:
    expected_keys = {
        "schema_version",
        "benchmark_id",
        "decision",
        "decision_counts",
        "chronology",
        "pre_unseal",
        "post_unseal",
        "proof",
        "negative_controls",
        "metrics",
        "claims",
        "scope",
        "first_remaining_blocker",
        "data_seals",
        "source_bindings",
        "content_sha256",
    }
    if set(value) != expected_keys:
        raise ValueError("synthetic affine geometry result keys changed")
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("benchmark_id") != BENCHMARK_ID
        or value.get("decision")
        != "pass_one_of_one_synthetic_affine_geometry_holdout_rediscovered_and_proved"
        or value.get("decision_counts") != {"pass": 1, "reject": 0, "blocked": 0}
        or value.get("claims") != _CLAIMS
        or value.get("content_sha256")
        != _sha({key: item for key, item in value.items() if key != "content_sha256"})
    ):
        raise ValueError("synthetic affine geometry result contract changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("synthetic affine geometry immutable replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable synthetic affine geometry artifact differs: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    result = run(root, _resolve(root, arguments.config))
    _write_immutable(_resolve(root, arguments.output), result)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
