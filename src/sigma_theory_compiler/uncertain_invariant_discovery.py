"""Retain multiplicative invariant candidates under noise, missingness, and censoring.

The learner receives positive-rational ratio observations with exact, interval, missing, or
one-sided-censored semantics.  It enumerates a bounded primitive exponent grammar without seeing
sealed targets, rejects only candidates whose observable interval excludes one, and preserves the
entire training-compatible set beside an exact deployment filter.  Ambiguity is an output, not a
reason to manufacture a unique formula.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_PATH = "configs/uncertain_invariant_discovery.json"
TARGETS_PATH = "configs/uncertain_invariant_discovery_targets.json"
OUTPUT_PATH = "runs/math/uncertain-invariant-discovery/receipt.json"
SOURCE_PATH = "src/sigma_theory_compiler/uncertain_invariant_discovery.py"
TEST_PATH = "tests/test_uncertain_invariant_discovery.py"
CONFIG_SCHEMA = "invariant-uncertain-invariant-discovery-config-1.0"
TARGETS_SCHEMA = "invariant-uncertain-invariant-discovery-targets-1.0"
RECEIPT_SCHEMA = "invariant-uncertain-invariant-discovery-receipt-1.0"
CAMPAIGN_ID = "uncertain-invariant-controls-2026-08-23-001"
_EXPECTED_PROBLEMS = {
    "control.noisy-ratio-action": ("measurement_noise", "noisy_interval"),
    "control.missing-ratio-action": ("missing_data", "missingness"),
    "control.censored-ratio-action": ("censored_data", "one_sided_censoring"),
}
_EXPECTED_STATUSES = {
    "noisy_interval": "NOISY_RETAIN_INTERVAL_COMPATIBLE_SET",
    "missingness": "MISSINGNESS_RETAIN_PARTIALLY_OBSERVED_SET",
    "one_sided_censoring": "CENSORED_RETAIN_SET_VALUED_CANDIDATES",
}
_ORIGIN_LABELS = [
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
]


class UncertainInvariantError(ValueError):
    """An uncertain observation, candidate set, replay, seal, or claim boundary failed."""


def _strict(value: Any, expected: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise UncertainInvariantError(f"{label} keys changed")
    return value


def _bound_path(root: Path, relative: str | Path, label: str) -> tuple[str, Path]:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        normalized = path.relative_to(root).as_posix()
    except ValueError as error:
        raise UncertainInvariantError(f"{label} escaped the repository root") from error
    return normalized, path


def _normalized_sha256(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").encode()
    except UnicodeDecodeError:
        pass
    return hashlib.sha256(raw).hexdigest()


def _positive_fraction(value: Any, label: str) -> Fraction:
    if not isinstance(value, str):
        raise UncertainInvariantError(f"{label} is not a rational string")
    try:
        parsed = Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise UncertainInvariantError(f"{label} is not rational") from error
    if parsed <= 0:
        raise UncertainInvariantError(f"{label} must be positive")
    return parsed


def _integer_vector(value: Any, width: int, label: str) -> tuple[int, ...]:
    if (
        not isinstance(value, list)
        or len(value) != width
        or any(not isinstance(item, int) or isinstance(item, bool) for item in value)
    ):
        raise UncertainInvariantError(f"{label} is not an integer vector of length {width}")
    return tuple(value)


def _validate_observation(value: Any, label: str) -> str:
    if not isinstance(value, Mapping) or "kind" not in value:
        raise UncertainInvariantError(f"{label} observation is malformed")
    kind = value["kind"]
    if kind == "exact":
        _strict(value, {"kind", "value"}, label)
        _positive_fraction(value["value"], f"{label} value")
    elif kind == "interval":
        _strict(value, {"kind", "lower", "upper"}, label)
        lower = _positive_fraction(value["lower"], f"{label} lower")
        upper = _positive_fraction(value["upper"], f"{label} upper")
        if lower >= upper:
            raise UncertainInvariantError(f"{label} interval is empty")
    elif kind == "missing":
        _strict(value, {"kind"}, label)
    elif kind == "right_censored":
        _strict(value, {"kind", "lower"}, label)
        _positive_fraction(value["lower"], f"{label} lower")
    elif kind == "left_censored":
        _strict(value, {"kind", "upper"}, label)
        _positive_fraction(value["upper"], f"{label} upper")
    else:
        raise UncertainInvariantError(f"{label} observation kind changed")
    return str(kind)


def _validate_transformations(
    value: Any,
    variables: Sequence[str],
    minimum: int,
    label: str,
) -> set[str]:
    if not isinstance(value, list) or len(value) < minimum:
        raise UncertainInvariantError(f"{label} lacks transformations")
    identifiers: set[str] = set()
    kinds: set[str] = set()
    for transformation in value:
        _strict(transformation, {"observations", "transformation_id"}, label)
        identifier = transformation["transformation_id"]
        observations = transformation["observations"]
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in identifiers
            or not isinstance(observations, Mapping)
            or set(observations) != set(variables)
        ):
            raise UncertainInvariantError(f"{label} identity or observation coverage changed")
        observed_nontrivial = False
        for name in variables:
            observation = observations[name]
            kind = _validate_observation(observation, f"{identifier} {name}")
            kinds.add(kind)
            if kind != "missing" and not (
                kind == "exact" and _positive_fraction(observation["value"], name) == 1
            ):
                observed_nontrivial = True
        if not observed_nontrivial:
            raise UncertainInvariantError(f"{label} contains a trivial transformation")
        identifiers.add(identifier)
    return kinds


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(value, {"campaign_id", "policy", "problems", "schema_version"}, "uncertain config")
    if value["schema_version"] != CONFIG_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise UncertainInvariantError("uncertain config identity changed")
    policy = _strict(
        value["policy"],
        {
            "maximum_absolute_exponent",
            "maximum_candidates",
            "maximum_l1_norm",
            "minimum_deployment_transformations",
            "minimum_evaluable_training_transformations",
            "minimum_training_transformations",
        },
        "uncertain policy",
    )
    if policy != {
        "maximum_absolute_exponent": 2,
        "maximum_candidates": 16,
        "maximum_l1_norm": 5,
        "minimum_deployment_transformations": 2,
        "minimum_evaluable_training_transformations": 2,
        "minimum_training_transformations": 4,
    }:
        raise UncertainInvariantError("uncertain policy changed")
    problems = value["problems"]
    if not isinstance(problems, list) or len(problems) != len(_EXPECTED_PROBLEMS):
        raise UncertainInvariantError("uncertain problem coverage changed")
    observed: dict[str, tuple[str, str]] = {}
    for problem in problems:
        _strict(
            problem,
            {
                "deployment_transformations",
                "domain",
                "observation_mode",
                "problem_id",
                "training_transformations",
                "variables",
            },
            "uncertain problem",
        )
        problem_id = problem["problem_id"]
        identity = (problem["domain"], problem["observation_mode"])
        variables = problem["variables"]
        if (
            _EXPECTED_PROBLEMS.get(problem_id) != identity
            or problem_id in observed
            or not isinstance(variables, list)
            or len(variables) != 3
            or len(variables) != len(set(variables))
            or any(not isinstance(name, str) or not name for name in variables)
        ):
            raise UncertainInvariantError("uncertain problem identity changed")
        training_kinds = _validate_transformations(
            problem["training_transformations"],
            variables,
            policy["minimum_training_transformations"],
            f"{problem_id} training",
        )
        deployment_kinds = _validate_transformations(
            problem["deployment_transformations"],
            variables,
            policy["minimum_deployment_transformations"],
            f"{problem_id} deployment",
        )
        expected_training_kinds = {
            "noisy_interval": {"interval"},
            "missingness": {"exact", "missing"},
            "one_sided_censoring": {"exact", "left_censored", "right_censored"},
        }[problem["observation_mode"]]
        if training_kinds != expected_training_kinds or deployment_kinds != {"exact"}:
            raise UncertainInvariantError("uncertain observation-mode coverage changed")
        observed[problem_id] = identity
    if observed != _EXPECTED_PROBLEMS:
        raise UncertainInvariantError("uncertain problem set changed")
    return dict(value)


def validate_targets(value: Mapping[str, Any], problems: set[str]) -> dict[str, Any]:
    _strict(value, {"campaign_id", "controls", "schema_version"}, "uncertain targets")
    if value["schema_version"] != TARGETS_SCHEMA or value["campaign_id"] != CAMPAIGN_ID:
        raise UncertainInvariantError("uncertain target identity changed")
    controls = value["controls"]
    if not isinstance(controls, list) or len(controls) != len(problems):
        raise UncertainInvariantError("uncertain target coverage changed")
    observed: set[str] = set()
    for control in controls:
        _strict(
            control,
            {
                "expected_deployment_failed_candidates",
                "expected_deployment_survivor_set",
                "expected_status",
                "expected_training_candidate_set",
                "problem_id",
            },
            "uncertain target",
        )
        problem_id = control["problem_id"]
        training = control["expected_training_candidate_set"]
        survivors = control["expected_deployment_survivor_set"]
        if (
            problem_id not in problems
            or problem_id in observed
            or control["expected_status"] not in set(_EXPECTED_STATUSES.values())
            or not isinstance(training, list)
            or not training
            or not isinstance(survivors, list)
            or not survivors
            or not isinstance(control["expected_deployment_failed_candidates"], int)
        ):
            raise UncertainInvariantError("uncertain target shape changed")
        training_vectors = [_integer_vector(vector, 3, problem_id) for vector in training]
        survivor_vectors = [_integer_vector(vector, 3, problem_id) for vector in survivors]
        if (
            len(set(training_vectors)) != len(training_vectors)
            or len(set(survivor_vectors)) != len(survivor_vectors)
            or not set(survivor_vectors).issubset(training_vectors)
            or control["expected_deployment_failed_candidates"]
            != len(training_vectors) - len(survivor_vectors)
        ):
            raise UncertainInvariantError("uncertain target candidate sets changed")
        observed.add(problem_id)
    if observed != problems:
        raise UncertainInvariantError("uncertain target problem set changed")
    return dict(value)


def load_config(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    _, config_path = _bound_path(root, CONFIG_PATH, "uncertain config")
    config = validate_config(json.loads(config_path.read_text(encoding="utf-8")))
    _, targets_path = _bound_path(root, TARGETS_PATH, "uncertain targets")
    targets = validate_targets(
        json.loads(targets_path.read_text(encoding="utf-8")),
        {problem["problem_id"] for problem in config["problems"]},
    )
    return config, targets


def _canonical_vector(values: Sequence[int]) -> tuple[int, ...]:
    divisor = math.gcd(*(abs(item) for item in values if item))
    vector = tuple(item // divisor for item in values)
    first = next(item for item in vector if item)
    return tuple(-item for item in vector) if first < 0 else vector


def _observation_bounds(
    observation: Mapping[str, Any], exponent: int
) -> tuple[Fraction, Fraction | None, bool]:
    if exponent == 0:
        return Fraction(1), Fraction(1), False
    kind = observation["kind"]
    if kind == "missing":
        return Fraction(0), None, True
    if kind == "exact":
        lower = upper = _positive_fraction(observation["value"], "exact observation")
    elif kind == "interval":
        lower = _positive_fraction(observation["lower"], "interval lower")
        upper = _positive_fraction(observation["upper"], "interval upper")
    elif kind == "right_censored":
        lower = _positive_fraction(observation["lower"], "censored lower")
        upper = None
    else:
        lower = Fraction(0)
        upper = _positive_fraction(observation["upper"], "censored upper")
    power = abs(exponent)
    if exponent > 0:
        return lower**power, None if upper is None else upper**power, False
    inverted_lower = Fraction(0) if upper is None else Fraction(1, upper**power)
    inverted_upper = None if lower == 0 else Fraction(1, lower**power)
    return inverted_lower, inverted_upper, False


def _replay(
    transformation: Mapping[str, Any], variables: Sequence[str], vector: Sequence[int]
) -> dict[str, Any]:
    lower = Fraction(1)
    upper: Fraction | None = Fraction(1)
    for name, exponent in zip(variables, vector, strict=True):
        item_lower, item_upper, missing = _observation_bounds(
            transformation["observations"][name], exponent
        )
        if missing:
            return {
                "lower": None,
                "status": "UNRESOLVED_MISSING_ACTIVE_VARIABLE",
                "transformation_id": transformation["transformation_id"],
                "upper": None,
            }
        lower *= item_lower
        upper = None if upper is None or item_upper is None else upper * item_upper
    compatible = lower <= 1 and (upper is None or upper >= 1)
    return {
        "lower": str(lower),
        "status": "COMPATIBLE_CONTAINS_ONE" if compatible else "REJECT_INTERVAL_EXCLUDES_ONE",
        "transformation_id": transformation["transformation_id"],
        "upper": "infinity" if upper is None else str(upper),
    }


def _finite_corner_bounds(
    transformation: Mapping[str, Any], variables: Sequence[str], vector: Sequence[int]
) -> tuple[Fraction, Fraction] | None:
    endpoints = []
    for name, exponent in zip(variables, vector, strict=True):
        if exponent == 0:
            endpoints.append([Fraction(1)])
            continue
        observation = transformation["observations"][name]
        if observation["kind"] == "exact":
            values = [_positive_fraction(observation["value"], name)]
        elif observation["kind"] == "interval":
            values = [
                _positive_fraction(observation["lower"], name),
                _positive_fraction(observation["upper"], name),
            ]
        else:
            return None
        endpoints.append([value**exponent for value in values])
    products = [math.prod(corner, start=Fraction(1)) for corner in itertools.product(*endpoints)]
    return min(products), max(products)


def _expression(variables: Sequence[str], vector: Sequence[int]) -> str:
    numerator = []
    denominator = []
    for name, exponent in zip(variables, vector, strict=True):
        if not exponent:
            continue
        factor = name if abs(exponent) == 1 else f"{name}**{abs(exponent)}"
        (numerator if exponent > 0 else denominator).append(factor)
    numerator_text = "*".join(numerator) or "1"
    if not denominator:
        return numerator_text
    denominator_text = "*".join(denominator)
    if len(denominator) > 1:
        denominator_text = f"({denominator_text})"
    return f"{numerator_text}/{denominator_text}"


def _candidate_vectors(policy: Mapping[str, Any], width: int) -> list[tuple[int, ...]]:
    values = range(
        -policy["maximum_absolute_exponent"],
        policy["maximum_absolute_exponent"] + 1,
    )
    vectors = []
    for candidate in itertools.product(values, repeat=width):
        if (
            not any(candidate)
            or sum(abs(item) for item in candidate) > policy["maximum_l1_norm"]
            or candidate != _canonical_vector(candidate)
        ):
            continue
        vectors.append(candidate)
    return sorted(vectors, key=lambda item: (sum(abs(value) for value in item), item))


def learn_problem(problem: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Enumerate uncertainty-compatible candidates without accepting a sealed target."""

    variables = list(problem["variables"])
    retained = []
    finite_corner_checks = 0
    for vector in _candidate_vectors(policy, len(variables)):
        replays = [
            _replay(transformation, variables, vector)
            for transformation in problem["training_transformations"]
        ]
        if any(replay["status"] == "REJECT_INTERVAL_EXCLUDES_ONE" for replay in replays):
            continue
        evaluable = sum(
            replay["status"] != "UNRESOLVED_MISSING_ACTIVE_VARIABLE" for replay in replays
        )
        if evaluable < policy["minimum_evaluable_training_transformations"]:
            continue
        for transformation, replay in zip(
            problem["training_transformations"], replays, strict=True
        ):
            corner_bounds = _finite_corner_bounds(transformation, variables, vector)
            if corner_bounds is None:
                continue
            finite_corner_checks += 1
            if replay["lower"] != str(corner_bounds[0]) or replay["upper"] != str(
                corner_bounds[1]
            ):
                raise UncertainInvariantError("interval and finite-corner evaluators disagree")
        retained.append(
            {
                "evaluable_training_transformations": evaluable,
                "exponents": list(vector),
                "expression": _expression(variables, vector),
                "missing_training_transformations": len(replays) - evaluable,
                "training_replays": replays,
            }
        )
    if not retained or len(retained) > policy["maximum_candidates"]:
        raise UncertainInvariantError("uncertain candidate set is empty or exceeds its bound")

    deployment_records = []
    survivors = []
    for candidate in retained:
        vector = candidate["exponents"]
        replays = [
            _replay(transformation, variables, vector)
            for transformation in problem["deployment_transformations"]
        ]
        survives = all(replay["status"] == "COMPATIBLE_CONTAINS_ONE" for replay in replays)
        if survives:
            survivors.append(candidate)
        deployment_records.append(
            {
                "exponents": list(vector),
                "expression": candidate["expression"],
                "replays": replays,
                "survives_exact_deployment": survives,
            }
        )
    status = _EXPECTED_STATUSES[problem["observation_mode"]]
    return {
        "creative_brief": {
            "candidate_invariant_coordinates": [item["expression"] for item in retained],
            "constraint_statement": (
                "Retain every training-compatible coordinate. Keep exact-deployment failures as "
                "repair or recombination branches; noise intervals, missing values, and censoring "
                "create set-valued evidence rather than permission to force a winner."
            ),
            "deployment_surviving_coordinates": [item["expression"] for item in survivors],
            "identifiability_status": status,
            "llm_origin_assessment_labels": list(_ORIGIN_LABELS),
            "novelty_caution": (
                "Uncertainty compatibility is neither an empirical law nor evidence of literature "
                "novelty; all three problems are synthetic calibration controls."
            ),
            "observation_mode": problem["observation_mode"],
        },
        "deployment": {
            "failed_candidates": len(retained) - len(survivors),
            "records": deployment_records,
            "surviving_candidates": len(survivors),
            "transformations": len(problem["deployment_transformations"]),
        },
        "domain": problem["domain"],
        "independent_evaluators": {
            "agreement": True,
            "finite_bounded_corner_checks": finite_corner_checks,
            "interval_monotonicity_replays": sum(
                len(item["training_replays"]) for item in retained
            ),
        },
        "observation_mode": problem["observation_mode"],
        "problem_id": problem["problem_id"],
        "status": status,
        "target_access": {
            "learner_input_sha256": canonical_sha256(problem),
            "target_visible_to_learner": False,
        },
        "training_candidates": retained,
        "variables": variables,
    }


def _evaluate_target(result: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    training = [tuple(item["exponents"]) for item in result["training_candidates"]]
    survivors = [
        tuple(item["exponents"])
        for item in result["deployment"]["records"]
        if item["survives_exact_deployment"]
    ]
    expected_training = [
        _integer_vector(vector, 3, f"{result['problem_id']} training target")
        for vector in target["expected_training_candidate_set"]
    ]
    expected_survivors = [
        _integer_vector(vector, 3, f"{result['problem_id']} survivor target")
        for vector in target["expected_deployment_survivor_set"]
    ]
    if (
        result["status"] != target["expected_status"]
        or training != expected_training
        or survivors != expected_survivors
        or result["deployment"]["failed_candidates"]
        != target["expected_deployment_failed_candidates"]
    ):
        raise UncertainInvariantError(f"{result['problem_id']} sealed control failed")
    return {
        "deployment_filter_matched": True,
        "expected_status_matched": True,
        "target_visible_to_learner": False,
        "training_candidate_set_matched": True,
    }


def _source_binding(root: Path, relative: str) -> dict[str, str]:
    path_text, path = _bound_path(root, relative, "uncertain source binding")
    if not path.is_file():
        raise UncertainInvariantError(f"uncertain source binding is missing: {path_text}")
    return {"normalized_sha256": _normalized_sha256(path), "path": path_text}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config, targets = load_config(root)
    target_by_problem = {control["problem_id"]: control for control in targets["controls"]}
    results = []
    for problem in config["problems"]:
        result = learn_problem(problem, config["policy"])
        result["sealed_control_evaluation"] = _evaluate_target(
            result, target_by_problem[problem["problem_id"]]
        )
        results.append(result)
    body: dict[str, Any] = {
        "campaign_id": CAMPAIGN_ID,
        "claims": {
            "empirical_law_discovered": False,
            "literature_novelty_established": False,
            "target_used_for_learning": False,
            "uncertainty_resolved_by_assumption": False,
            "unique_formula_identified": False,
        },
        "release_gate": {
            "creative_context_ready": True,
            "serious_claim_released": False,
            "status": "PASS_UNCERTAIN_CONTROLS_NO_LAW_OR_NOVELTY_CLAIM",
        },
        "results": results,
        "schema_version": RECEIPT_SCHEMA,
        "source_bindings": {
            "config": _source_binding(root, CONFIG_PATH),
            "source": _source_binding(root, SOURCE_PATH),
            "targets": _source_binding(root, TARGETS_PATH),
            "test": _source_binding(root, TEST_PATH),
        },
        "summary": {
            "censored_controls": sum(
                result["observation_mode"] == "one_sided_censoring" for result in results
            ),
            "controls": len(results),
            "deployment_failed_candidates": sum(
                result["deployment"]["failed_candidates"] for result in results
            ),
            "deployment_surviving_candidates": sum(
                result["deployment"]["surviving_candidates"] for result in results
            ),
            "missingness_controls": sum(
                result["observation_mode"] == "missingness" for result in results
            ),
            "noisy_controls": sum(
                result["observation_mode"] == "noisy_interval" for result in results
            ),
            "status": "PASS_UNCERTAIN_INVARIANT_BRANCH_CONTROLS",
            "target_blind_controls": sum(
                result["target_access"]["target_visible_to_learner"] is False
                for result in results
            ),
            "training_candidates_retained": sum(
                len(result["training_candidates"]) for result in results
            ),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _strict(
        receipt,
        {
            "campaign_id",
            "claims",
            "content_sha256",
            "release_gate",
            "results",
            "schema_version",
            "source_bindings",
            "summary",
        },
        "uncertain receipt",
    )
    if receipt["schema_version"] != RECEIPT_SCHEMA or receipt["campaign_id"] != CAMPAIGN_ID:
        raise UncertainInvariantError("uncertain receipt identity changed")
    claims = _strict(
        receipt["claims"],
        {
            "empirical_law_discovered",
            "literature_novelty_established",
            "target_used_for_learning",
            "uncertainty_resolved_by_assumption",
            "unique_formula_identified",
        },
        "uncertain claims",
    )
    if any(claims.values()):
        raise UncertainInvariantError("uncertain claim boundary changed")
    expected = build_receipt(root)
    if receipt != expected:
        raise UncertainInvariantError("uncertain receipt no longer reproduces")
    return dict(receipt)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "validate"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--root", type=Path, default=Path.cwd())
        command_parser.add_argument(
            "--output" if command == "build" else "--receipt",
            type=Path,
            default=Path(OUTPUT_PATH),
        )
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "build":
        receipt = build_receipt(root)
        _, output = _bound_path(root, args.output, "uncertain receipt output")
        _write_json(output, receipt)
    else:
        _, receipt_path = _bound_path(root, args.receipt, "uncertain receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "controls": receipt["summary"]["controls"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
