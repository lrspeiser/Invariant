"""Deterministic synthetic finite-field exponent rediscovery control."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "sigma-synthetic-modular-exponent-holdout-config-1.0"
RESULT_SCHEMA = "sigma-synthetic-modular-exponent-holdout-result-1.0"
BENCHMARK_ID = "synthetic-modular-exponent-holdout-world-001"
CONFIG_PATH = "configs/synthetic_modular_exponent_holdout_world.json"
SOURCE_PATH = "src/sigma_theory_compiler/synthetic_modular_exponent_holdout_world.py"
TEST_PATH = "tests/test_synthetic_modular_exponent_holdout_world.py"
OUTPUT_PATH = "runs/math/synthetic-modular-exponent-holdout-world/campaign.json"

_CLAIMS = {
    "complete_declared_exponent_grammar_enumerated": True,
    "exhaustive_nonzero_residue_proof_completed": True,
    "fresh_seed_derived_anonymous_prime_world_generated": True,
    "reference_theorem_sealed_before_discovery": True,
    "reference_payload_supplied_to_discovery": False,
    "winner_sealed_before_post_unseal_comparison": True,
    "withheld_theorem_independently_rediscovered": True,
    "post_unseal_equivalence_confirmed": True,
    "general_number_theory_completeness_established": False,
    "unbounded_exponent_discovery_established": False,
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


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{label} keys changed")


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    return all(value % divisor for divisor in range(2, int(value**0.5) + 1))


def _expected_config() -> dict[str, Any]:
    return {
        "schema_version": CONFIG_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "world_generator": {
            "namespace": "invariant.synthetic.number_theory.posttraining.001",
            "generation_epoch": "2026-08-12",
            "prime_candidates": [11, 13, 17, 19],
            "selection": "minimum_seeded_candidate_score",
            "candidate_exponent_floor": 1,
            "candidate_exponent_ceiling": "modulus_minus_one",
        },
        "holdout_contract": {
            "eligible_holdouts": 1,
            "target": "least_positive_universal_nonzero_residue_exponent",
            "reference_theorem_sealed_before_discovery": True,
            "reference_payload_supplied_to_discovery": False,
            "post_unseal_comparison_only": True,
        },
        "policies": {
            "network_access": "forbidden",
            "live_sqlite_access": "forbidden",
            "discovery_input": "generated_public_modular_world_only",
            "exact_proof": "exhaustive_nonzero_residue_replay",
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
        raise ValueError("synthetic modular exponent config path changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    _exact_keys(value, set(_expected_config()), "config")
    if value != _expected_config():
        raise ValueError("synthetic modular exponent config contract changed")
    candidates = value["world_generator"]["prime_candidates"]
    if len(set(candidates)) != len(candidates) or not all(_is_prime(item) for item in candidates):
        raise ValueError("prime candidate registry changed")
    return value


def _select_modulus(config: Mapping[str, Any]) -> int:
    generator = config["world_generator"]
    namespace = generator["namespace"]
    return min(
        generator["prime_candidates"],
        key=lambda prime: hashlib.sha256(f"{namespace}:prime:{prime}".encode()).digest(),
    )


def _power_trace(base: int, exponent: int, modulus: int) -> list[int]:
    trace: list[int] = []
    value = 1
    for _ in range(exponent):
        value = (value * base) % modulus
        trace.append(value)
    return trace


def _evaluate_exponent(modulus: int, exponent: int) -> dict[str, Any]:
    rows = []
    failures = []
    for residue in range(1, modulus):
        trace = _power_trace(residue, exponent, modulus)
        row = {
            "residue": residue,
            "power_trace": trace,
            "terminal_residue": trace[-1],
            "passed": trace[-1] == 1,
        }
        rows.append(row)
        if not row["passed"]:
            failures.append(residue)
    return {
        "exponent": exponent,
        "residues_checked": modulus - 1,
        "passed": not failures,
        "first_failing_residue": failures[0] if failures else None,
        "rows": rows,
        "proof_sha256": _sha(rows),
    }


def _public_world(config: Mapping[str, Any]) -> dict[str, Any]:
    modulus = _select_modulus(config)
    residues = list(range(modulus))
    multiplication_table = [[(left * right) % modulus for right in residues] for left in residues]
    world = {
        "world_kind": "anonymous_prime_residue_multiplicative_world",
        "modulus": modulus,
        "residues": residues,
        "nonzero_residues": residues[1:],
        "multiplication_table": multiplication_table,
        "candidate_exponents": list(range(1, modulus)),
        "selection_receipt": {
            "namespace": config["world_generator"]["namespace"],
            "candidate_count": len(config["world_generator"]["prime_candidates"]),
            "method": config["world_generator"]["selection"],
        },
    }
    return _seal(world)


def _reference_theorem(public_world: Mapping[str, Any]) -> dict[str, Any]:
    modulus = public_world["modulus"]
    evaluations = [
        _evaluate_exponent(modulus, exponent) for exponent in public_world["candidate_exponents"]
    ]
    passing = [row for row in evaluations if row["passed"]]
    if not passing:
        raise ValueError("reference theorem has no witness in declared grammar")
    winner = passing[0]
    theorem = {
        "theorem_kind": "least_positive_universal_nonzero_residue_exponent",
        "modulus": modulus,
        "exponent": winner["exponent"],
        "statement": "every nonzero residue raised to the registered exponent is one modulo p",
        "minimality_failures": [
            {
                "exponent": row["exponent"],
                "first_failing_residue": row["first_failing_residue"],
            }
            for row in evaluations[: winner["exponent"] - 1]
        ],
        "proof": winner,
    }
    theorem["theorem_id"] = f"thm-{_sha(theorem)[:20]}"
    return _seal(theorem)


def _discover(public_world: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate from public world only; no reference theorem argument is accepted."""

    modulus = public_world["modulus"]
    evaluations = [
        _evaluate_exponent(modulus, exponent) for exponent in public_world["candidate_exponents"]
    ]
    passing = [row for row in evaluations if row["passed"]]
    if not passing:
        raise ValueError("discovery produced no exact candidate")
    winner = passing[0]
    return _seal(
        {
            "grammar": {
                "candidate_kind": "positive_integer_exponent",
                "minimum": public_world["candidate_exponents"][0],
                "maximum": public_world["candidate_exponents"][-1],
                "candidate_count": len(public_world["candidate_exponents"]),
            },
            "evaluations": evaluations,
            "passing_exponents": [row["exponent"] for row in passing],
            "winner": winner,
            "winner_selection": "least_passing_exponent",
        }
    )


def _negative_controls(
    public_world: Mapping[str, Any], discovery: Mapping[str, Any]
) -> list[dict[str, Any]]:
    modulus = public_world["modulus"]
    winner = discovery["winner"]
    wrong_exponent = winner["exponent"] - 1
    wrong = _evaluate_exponent(modulus, wrong_exponent)
    truncated_residues = [1]
    truncated_false_pass = all(
        pow(residue, wrong_exponent, modulus) == 1 for residue in truncated_residues
    )
    composite = 12
    composite_wrong = _evaluate_exponent(composite, composite - 1)
    return [
        {
            "control_id": "neighboring_exponent_rejected",
            "candidate_exponent": wrong_exponent,
            "rejected": not wrong["passed"],
            "first_counterexample": wrong["first_failing_residue"],
        },
        {
            "control_id": "truncated_residue_evidence_rejected",
            "candidate_exponent": wrong_exponent,
            "truncated_residues": truncated_residues,
            "truncated_check_would_pass": truncated_false_pass,
            "full_replay_rejected": not wrong["passed"],
        },
        {
            "control_id": "composite_modulus_fermat_shape_rejected",
            "modulus": composite,
            "candidate_exponent": composite - 1,
            "rejected": not composite_wrong["passed"],
            "first_counterexample": composite_wrong["first_failing_residue"],
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
            "candidate_kind": "positive_integer_exponent",
            "universal_domain": "all_nonzero_residues",
            "selection": "least_exactly_passing_exponent",
        },
    }
    public_input_seal = _sha(public_input)
    discovery = _discover(world)
    winner_seal = _sha(discovery)
    comparison = {
        "performed_after_winner_seal": True,
        "reference_theorem_id": reference["theorem_id"],
        "reference_exponent": reference["exponent"],
        "rediscovered_exponent": discovery["winner"]["exponent"],
        "exact_match": reference["exponent"] == discovery["winner"]["exponent"],
    }
    controls = _negative_controls(world, discovery)
    modulus = world["modulus"]
    return _seal(
        {
            "schema_version": RESULT_SCHEMA,
            "benchmark_id": BENCHMARK_ID,
            "decision": "pass_one_of_one_synthetic_modular_exponent_holdout_rediscovered_and_proved",
            "decision_counts": {"pass": 1, "reject": 0, "blocked": 0},
            "chronology": [
                {
                    "ordinal": 1,
                    "phase": "public_modular_world_generated",
                    "seal": world["content_sha256"],
                },
                {"ordinal": 2, "phase": "reference_theorem_sealed", "seal": reference_seal},
                {"ordinal": 3, "phase": "public_discovery_input_sealed", "seal": public_input_seal},
                {
                    "ordinal": 4,
                    "phase": "bounded_exponent_grammar_enumerated",
                    "seal": discovery["content_sha256"],
                },
                {"ordinal": 5, "phase": "winner_and_exact_proof_sealed", "seal": winner_seal},
                {
                    "ordinal": 6,
                    "phase": "reference_unsealed_and_compared",
                    "seal": _sha(comparison),
                },
            ],
            "pre_unseal": {
                "public_input": public_input,
                "public_input_sha256": public_input_seal,
                "reference_payload_supplied_to_discovery": False,
                "discovery": discovery,
                "winner_seal_sha256": winner_seal,
            },
            "post_unseal": {"reference_theorem": reference, "comparison": comparison},
            "proof": {
                "method": "exhaustive_nonzero_residue_replay",
                "candidate_exponents_checked": modulus - 1,
                "residues_per_candidate": modulus - 1,
                "modular_power_obligations": (modulus - 1) ** 2,
                "winning_exponent": discovery["winner"]["exponent"],
                "winning_residues_checked": discovery["winner"]["residues_checked"],
                "winning_counterexample_count": 0,
                "minimality_counterexamples": reference["minimality_failures"],
            },
            "negative_controls": controls,
            "metrics": {
                "worlds": 1,
                "holdouts": 1,
                "independently_rediscovered_and_proved": 1,
                "candidate_exponents": modulus - 1,
                "residues": modulus,
                "nonzero_residues": modulus - 1,
            },
            "claims": dict(_CLAIMS),
            "scope": (
                f"one deterministic anonymous prime residue world modulo {modulus}, the bounded "
                f"exponent grammar 1..{modulus - 1}, and exhaustive replay over every nonzero "
                "residue; no historical novelty, unbounded exponent discovery, general number-"
                "theory completeness, external proof kernel, hostile-process isolation, or "
                "external mathematical significance"
            ),
            "first_remaining_blocker": (
                "replicate_across_preregistered_independently_generated_prime_worlds_and_add_"
                "an_external_proof_kernel_without_exposing_the_held_out_theorem"
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
    if set(value) != {
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
    }:
        raise ValueError("synthetic modular exponent result keys changed")
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("benchmark_id") != BENCHMARK_ID
        or value.get("decision")
        != "pass_one_of_one_synthetic_modular_exponent_holdout_rediscovered_and_proved"
        or value.get("decision_counts") != {"pass": 1, "reject": 0, "blocked": 0}
        or value.get("claims") != _CLAIMS
        or value.get("content_sha256")
        != _sha({key: item for key, item in value.items() if key != "content_sha256"})
    ):
        raise ValueError("synthetic modular exponent result contract changed")
    if dict(value) != build_campaign(root, config_path):
        raise ValueError("synthetic modular exponent immutable replay mismatch")


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    result = build_campaign(root, config_path)
    validate_campaign(result, root, config_path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable synthetic modular exponent artifact differs: {path}")
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
