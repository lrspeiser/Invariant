"""Prospective target-sealed independence test for native non-polynomial generators."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_SCHEMA = "invariant-native-nonpolynomial-independence-config-1.0"
TARGET_SCHEMA = "invariant-native-nonpolynomial-target-batch-1.0"
RESULT_SCHEMA = "invariant-native-nonpolynomial-independence-result-1.0"
CERTIFICATE_SCHEMA = "invariant-native-structured-identity-certificate-1.0"
CAMPAIGN_ID = "native-nonpolynomial-independence-tournament-001"
CONFIG_PATH = "configs/native_nonpolynomial_independence_tournament.json"
TARGET_PATH = "configs/native_nonpolynomial_independence_targets.json"
SOURCE_PATH = "src/sigma_theory_compiler/native_nonpolynomial_independence_tournament.py"
TEST_PATH = "tests/test_native_nonpolynomial_independence_tournament.py"
DOC_PATH = "docs/NATIVE_NONPOLYNOMIAL_INDEPENDENCE.md"
OUTPUT_PATH = "runs/math/native-nonpolynomial-independence/receipt.json"

FAMILY_DESCRIPTORS = {
    "reciprocal_elimination": {
        "algorithm": "three_point_fractional_linear_elimination",
        "family_id": "reciprocal_elimination",
        "supported_class": "rational_linear_fractional",
        "version": "1.0.0",
    },
    "difference_ratio": {
        "algorithm": "three_point_geometric_first_difference_ratio",
        "family_id": "difference_ratio",
        "supported_class": "shifted_exponential",
        "version": "1.0.0",
    },
    "bounded_structure_enumerator": {
        "algorithm": "seed_ordered_bounded_integer_structure_enumeration",
        "family_id": "bounded_structure_enumerator",
        "supported_class": "declared_benchmark_classes",
        "version": "1.0.0",
    },
}
FAMILIES = tuple(FAMILY_DESCRIPTORS)
WORLD_CLASSES = ("rational_linear_fractional", "shifted_exponential")
CLAIMS = {
    "atomic_target_unseal_batches": 1,
    "bayesian_generator_calls": 0,
    "candidate_generation_after_target_unseal": 0,
    "candidate_tuning_after_target_unseal": 0,
    "declared_benchmark_classes_leave_one_family_out_covered": True,
    "formula_discovery_job_delegations": 0,
    "generic_exact_linear_solver_calls": 0,
    "general_formula_discovery_established": False,
    "multiple_independent_native_families_exercised": True,
    "multiple_nonpolynomial_worlds_exercised": True,
    "novelty_established": False,
    "target_records_read_before_candidate_freeze": 0,
}


class NativeIndependenceError(ValueError):
    """Raised when a closed contract, seal, or exact replay changes."""


@dataclass(frozen=True)
class GeneratorOutcome:
    family: str
    status: str
    candidate: dict[str, Any] | None
    work_units: int
    blocker: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "blocker": self.blocker,
            "candidate": self.candidate,
            "family": self.family,
            "status": self.status,
            "work_units": self.work_units,
        }


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise NativeIndependenceError("path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise NativeIndependenceError("path escapes repository root") from error
    return path


def _file_sha256(path: Path) -> str:
    try:
        data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    except OSError as error:
        raise NativeIndependenceError("bound file unavailable") from error
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeIndependenceError("tournament JSON unavailable") from error
    if not isinstance(value, dict):
        raise NativeIndependenceError("tournament JSON must be an object")
    return value


def _fraction(value: Mapping[str, Any]) -> Fraction:
    if set(value) != {"p", "q"}:
        raise NativeIndependenceError("rational value schema changed")
    p, q = value["p"], value["q"]
    if (
        not isinstance(p, int)
        or isinstance(p, bool)
        or not isinstance(q, int)
        or isinstance(q, bool)
        or q <= 0
        or Fraction(p, q).numerator != p
        or Fraction(p, q).denominator != q
    ):
        raise NativeIndependenceError("rational value is not canonical")
    return Fraction(p, q)


def _fraction_data(value: Fraction | int) -> dict[str, int]:
    rational = Fraction(value)
    return {"p": rational.numerator, "q": rational.denominator}


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "campaign_id",
        "family_contracts",
        "output_path",
        "policies",
        "schema_version",
        "target_fixture",
        "worlds",
    }:
        raise NativeIndependenceError("config keys changed")
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
    ):
        raise NativeIndependenceError("config identity changed")
    if config.get("policies") != {
        "atomic_target_unseal_batches": 1,
        "candidate_generation_after_unseal": 0,
        "candidate_tuning_after_unseal": 0,
        "generic_exact_linear_solver_call_cap": 0,
        "target_reads_before_candidate_freeze": 0,
    }:
        raise NativeIndependenceError("prospective policy changed")
    target_fixture = config.get("target_fixture")
    if not isinstance(target_fixture, Mapping) or set(target_fixture) != {
        "content_sha256",
        "path",
    }:
        raise NativeIndependenceError("target fixture binding changed")
    if target_fixture.get("path") != TARGET_PATH:
        raise NativeIndependenceError("target fixture path changed")

    contracts = config.get("family_contracts")
    if not isinstance(contracts, list) or len(contracts) != len(FAMILIES):
        raise NativeIndependenceError("family contract inventory changed")
    if [row.get("family") for row in contracts if isinstance(row, Mapping)] != list(FAMILIES):
        raise NativeIndependenceError("family order changed")
    for row in contracts:
        if not isinstance(row, Mapping) or set(row) != {
            "code_identity_sha256",
            "family",
            "seed",
            "work_budget",
        }:
            raise NativeIndependenceError("family contract schema changed")
        family = row["family"]
        if row["code_identity_sha256"] != canonical_sha256(FAMILY_DESCRIPTORS[family]):
            raise NativeIndependenceError("family code identity changed")
        if (
            not isinstance(row["seed"], int)
            or isinstance(row["seed"], bool)
            or not isinstance(row["work_budget"], int)
            or isinstance(row["work_budget"], bool)
            or row["work_budget"] < 1
            or row["work_budget"] > 4096
        ):
            raise NativeIndependenceError("family seed or work budget invalid")

    worlds = config.get("worlds")
    if not isinstance(worlds, list) or len(worlds) != 2:
        raise NativeIndependenceError("world inventory changed")
    if [world.get("class") for world in worlds if isinstance(world, Mapping)] != list(
        WORLD_CLASSES
    ):
        raise NativeIndependenceError("benchmark class inventory changed")
    world_ids: set[str] = set()
    for world in worlds:
        if not isinstance(world, Mapping) or set(world) != {
            "class",
            "public_rows",
            "sealed_target_sha256",
            "variable",
            "world_id",
        }:
            raise NativeIndependenceError("public world schema changed")
        if world["world_id"] in world_ids or world["variable"] not in {"n", "x"}:
            raise NativeIndependenceError("public world identity changed")
        world_ids.add(world["world_id"])
        rows = world["public_rows"]
        if not isinstance(rows, list) or len(rows) < 4 or len(rows) > 16:
            raise NativeIndependenceError("public row budget changed")
        points: set[int] = set()
        for row in rows:
            if not isinstance(row, Mapping) or set(row) != {"point", "value"}:
                raise NativeIndependenceError("public row schema changed")
            point = row["point"]
            if not isinstance(point, int) or isinstance(point, bool) or point in points:
                raise NativeIndependenceError("public point changed")
            points.add(point)
            _fraction(row["value"])


def _rows(world: Mapping[str, Any]) -> list[tuple[int, Fraction]]:
    return [(row["point"], _fraction(row["value"])) for row in world["public_rows"]]


def _candidate(
    family: str,
    world: Mapping[str, Any],
    kind: str,
    parameters: Mapping[str, Fraction | int],
    *,
    seed: int,
    work_units: int,
) -> dict[str, Any]:
    body = {
        "construction_audit": {
            "bayesian_generator_calls": 0,
            "formula_discovery_job_delegations": 0,
            "generic_exact_linear_solver_calls": 0,
            "target_fields_read": [],
        },
        "family": family,
        "kind": kind,
        "parameters": {key: _fraction_data(value) for key, value in sorted(parameters.items())},
        "public_rows_sha256": canonical_sha256(world["public_rows"]),
        "seed": seed,
        "work_units": work_units,
        "world_id": world["world_id"],
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _constant_baseline(family: str, world: Mapping[str, Any], *, seed: int) -> GeneratorOutcome:
    first = _rows(world)[0][1]
    candidate = _candidate(
        family,
        world,
        "constant",
        {"value": first},
        seed=seed,
        work_units=1,
    )
    return GeneratorOutcome(family, "CANDIDATE", candidate, 1, None)


def _reciprocal_elimination(
    world: Mapping[str, Any], *, seed: int, budget: int
) -> GeneratorOutcome:
    family = "reciprocal_elimination"
    if budget < 8:
        return GeneratorOutcome(family, "BLOCK", None, budget, "work_budget_exhausted")
    if world["class"] != "rational_linear_fractional":
        return _constant_baseline(family, world, seed=seed)
    rows = _rows(world)
    if [point for point, _ in rows[:3]] != [0, 1, 2]:
        return GeneratorOutcome(family, "BLOCK", None, 1, "required_point_pattern_absent")
    y0, y1, y2 = (value for _, value in rows[:3])
    denominator = 2 * y1 - y0 - y2
    if denominator == 0:
        return GeneratorOutcome(family, "BLOCK", None, 4, "degenerate_elimination")
    c = 2 * (y2 - y1) / denominator
    b = y0 * c
    a = y1 + (y1 - y0) * c
    candidate = _candidate(
        family,
        world,
        "rational_linear_fractional",
        {"a": a, "b": b, "c": c},
        seed=seed,
        work_units=8,
    )
    if not _matches_public(candidate, world):
        return GeneratorOutcome(family, "BLOCK", None, 8, "public_validation_failed")
    return GeneratorOutcome(family, "CANDIDATE", candidate, 8, None)


def _difference_ratio(world: Mapping[str, Any], *, seed: int, budget: int) -> GeneratorOutcome:
    family = "difference_ratio"
    if budget < 7:
        return GeneratorOutcome(family, "BLOCK", None, budget, "work_budget_exhausted")
    if world["class"] != "shifted_exponential":
        return _constant_baseline(family, world, seed=seed)
    rows = _rows(world)
    if [point for point, _ in rows[:3]] != [0, 1, 2]:
        return GeneratorOutcome(family, "BLOCK", None, 1, "required_point_pattern_absent")
    y0, y1, y2 = (value for _, value in rows[:3])
    d0, d1 = y1 - y0, y2 - y1
    if d0 == 0 or d1 == d0:
        return GeneratorOutcome(family, "BLOCK", None, 3, "degenerate_difference_ratio")
    r = d1 / d0
    a = d0 / (r - 1)
    b = y0 - a
    candidate = _candidate(
        family,
        world,
        "shifted_exponential",
        {"a": a, "b": b, "r": r},
        seed=seed,
        work_units=7,
    )
    if not _matches_public(candidate, world):
        return GeneratorOutcome(family, "BLOCK", None, 7, "public_validation_failed")
    return GeneratorOutcome(family, "CANDIDATE", candidate, 7, None)


def _ordered_integers(low: int, high: int, seed: int) -> list[int]:
    values = list(range(low, high + 1))
    if seed % 2:
        values.reverse()
    return values


def _bounded_structure_enumerator(
    world: Mapping[str, Any], *, seed: int, budget: int
) -> GeneratorOutcome:
    family = "bounded_structure_enumerator"
    work = 0
    values = _ordered_integers(-8, 8, seed)
    if world["class"] == "rational_linear_fractional":
        for c in _ordered_integers(1, 6, seed):
            for a in values:
                for b in values:
                    work += 1
                    if work > budget:
                        return GeneratorOutcome(
                            family, "BLOCK", None, budget, "work_budget_exhausted"
                        )
                    candidate = _candidate(
                        family,
                        world,
                        "rational_linear_fractional",
                        {"a": a, "b": b, "c": c},
                        seed=seed,
                        work_units=work,
                    )
                    if _matches_public(candidate, world):
                        return GeneratorOutcome(family, "CANDIDATE", candidate, work, None)
    elif world["class"] == "shifted_exponential":
        for a in _ordered_integers(-5, 5, seed):
            for r in _ordered_integers(-4, 4, seed):
                for b in values:
                    work += 1
                    if work > budget:
                        return GeneratorOutcome(
                            family, "BLOCK", None, budget, "work_budget_exhausted"
                        )
                    candidate = _candidate(
                        family,
                        world,
                        "shifted_exponential",
                        {"a": a, "b": b, "r": r},
                        seed=seed,
                        work_units=work,
                    )
                    if _matches_public(candidate, world):
                        return GeneratorOutcome(family, "CANDIDATE", candidate, work, None)
    return GeneratorOutcome(family, "BLOCK", None, work, "no_candidate_within_bounds")


def _parameter(candidate: Mapping[str, Any], name: str) -> Fraction:
    parameters = candidate.get("parameters")
    if not isinstance(parameters, Mapping) or name not in parameters:
        raise NativeIndependenceError("candidate parameter missing")
    return _fraction(parameters[name])


def _evaluate_candidate(candidate: Mapping[str, Any], point: int) -> Fraction | None:
    kind = candidate.get("kind")
    if kind == "constant":
        return _parameter(candidate, "value")
    if kind == "rational_linear_fractional":
        a = _parameter(candidate, "a")
        b = _parameter(candidate, "b")
        c = _parameter(candidate, "c")
        denominator = Fraction(point) + c
        return None if denominator == 0 else (a * point + b) / denominator
    if kind == "shifted_exponential":
        if point < 0:
            return None
        a = _parameter(candidate, "a")
        b = _parameter(candidate, "b")
        r = _parameter(candidate, "r")
        return a * (r**point) + b
    raise NativeIndependenceError("candidate kind changed")


def _matches_public(candidate: Mapping[str, Any], world: Mapping[str, Any]) -> bool:
    return all(
        _evaluate_candidate(candidate, point) == expected for point, expected in _rows(world)
    )


def run_generator(world: Mapping[str, Any], contract: Mapping[str, Any]) -> GeneratorOutcome:
    """Run one native family under its caller-frozen seed and hard work cap."""

    family = contract["family"]
    seed = contract["seed"]
    budget = contract["work_budget"]
    if family == "reciprocal_elimination":
        return _reciprocal_elimination(world, seed=seed, budget=budget)
    if family == "difference_ratio":
        return _difference_ratio(world, seed=seed, budget=budget)
    if family == "bounded_structure_enumerator":
        return _bounded_structure_enumerator(world, seed=seed, budget=budget)
    raise NativeIndependenceError("unknown generator family")


def _prepare_phase_a(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    outcomes: dict[str, list[Any]] = {}
    for world in config["worlds"]:
        outcomes[world["world_id"]] = [
            run_generator(world, contract).to_dict() for contract in config["family_contracts"]
        ]
    body = {
        "bayesian_generator_calls": sum(
            result["candidate"]["construction_audit"]["bayesian_generator_calls"]
            for rows in outcomes.values()
            for result in rows
            if result["candidate"] is not None
        ),
        "candidate_generation_complete": True,
        "candidate_generation_events": len(config["worlds"]) * len(FAMILIES),
        "candidate_generation_after_target_unseal": 0,
        "family_code_identities": {
            family: canonical_sha256(descriptor)
            for family, descriptor in FAMILY_DESCRIPTORS.items()
        },
        "family_contracts_sha256": canonical_sha256(config["family_contracts"]),
        "frozen_generator_outcomes": outcomes,
        "generic_exact_linear_solver_calls": sum(
            result["candidate"]["construction_audit"]["generic_exact_linear_solver_calls"]
            for rows in outcomes.values()
            for result in rows
            if result["candidate"] is not None
        ),
        "phase": "candidate_set_frozen",
        "public_worlds_sha256": canonical_sha256(config["worlds"]),
        "target_records_read": 0,
    }
    return {**body, "content_sha256": canonical_sha256(body)}, outcomes


@contextmanager
def _deny_target_fixture_reads(root: Path):
    target = _resolve(root, TARGET_PATH)
    original = io.open
    audit = {
        "attempted_reads": 0,
        "denied_content_bytes_exposed": 0,
        "denied_reads": 0,
    }

    def guarded_open(file: Any, *args: Any, **kwargs: Any):
        try:
            resolved = Path(file).resolve()
        except TypeError:
            resolved = None
        if resolved == target:
            audit["attempted_reads"] += 1
            audit["denied_reads"] += 1
            raise PermissionError("target fixture is sealed during candidate generation")
        return original(file, *args, **kwargs)

    io.open = guarded_open
    try:
        yield audit
    finally:
        io.open = original


def _bind_phase_a_access_audit(
    phase_a: Mapping[str, Any], access_audit: Mapping[str, int]
) -> dict[str, Any]:
    body = {
        **{key: value for key, value in phase_a.items() if key != "content_sha256"},
        "target_access_enforcement": dict(access_audit),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def _unseal_targets(
    root: Path, config: Mapping[str, Any], phase_a: Mapping[str, Any]
) -> tuple[dict[str, dict[str, Any]], str]:
    if (
        phase_a.get("phase") != "candidate_set_frozen"
        or phase_a.get("candidate_generation_complete") is not True
        or phase_a.get("target_records_read") != 0
        or phase_a.get("content_sha256")
        != canonical_sha256(
            {key: value for key, value in phase_a.items() if key != "content_sha256"}
        )
    ):
        raise NativeIndependenceError("target unseal attempted before candidate freeze")
    target_path = _resolve(root, config["target_fixture"]["path"])
    try:
        raw = target_path.read_bytes()
        fixture = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeIndependenceError("target fixture unavailable") from error
    if (
        not isinstance(fixture, dict)
        or canonical_sha256(fixture) != config["target_fixture"]["content_sha256"]
    ):
        raise NativeIndependenceError("target fixture content changed")
    if set(fixture) != {"schema_version", "targets"} or fixture["schema_version"] != TARGET_SCHEMA:
        raise NativeIndependenceError("target fixture schema changed")
    targets = fixture["targets"]
    if not isinstance(targets, list) or len(targets) != len(config["worlds"]):
        raise NativeIndependenceError("atomic target batch incomplete")
    by_world: dict[str, dict[str, Any]] = {}
    configured = {world["world_id"]: world for world in config["worlds"]}
    for target in targets:
        if not isinstance(target, dict) or set(target) != {
            "class",
            "holdout_points",
            "parameters",
            "world_id",
        }:
            raise NativeIndependenceError("target record schema changed")
        world_id = target["world_id"]
        if world_id not in configured or world_id in by_world:
            raise NativeIndependenceError("target world identity changed")
        if target["class"] != configured[world_id]["class"]:
            raise NativeIndependenceError("target class changed")
        if canonical_sha256(target) != configured[world_id]["sealed_target_sha256"]:
            raise NativeIndependenceError("target commitment did not open")
        if not isinstance(target["holdout_points"], list) or not target["holdout_points"]:
            raise NativeIndependenceError("target holdout inventory changed")
        by_world[world_id] = target
    if set(by_world) != set(configured):
        raise NativeIndependenceError("atomic target world coverage changed")
    normalized = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return by_world, hashlib.sha256(normalized).hexdigest()


def _target_candidate(target: Mapping[str, Any]) -> dict[str, Any]:
    parameters = target["parameters"]
    if not isinstance(parameters, Mapping):
        raise NativeIndependenceError("target parameters changed")
    body = {
        "kind": target["class"],
        "parameters": dict(parameters),
    }
    return body


def _assess(candidate: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    target_candidate = _target_candidate(target)
    candidate_identity = {
        "kind": candidate["kind"],
        "parameters": candidate["parameters"],
    }
    if candidate_identity == target_candidate:
        residuals = {
            key: _fraction_data(
                _fraction(candidate["parameters"][key]) - _fraction(target["parameters"][key])
            )
            for key in sorted(target["parameters"])
        }
        body = {
            "candidate_content_sha256": candidate["content_sha256"],
            "decision": "proved_exact_structured_identity",
            "method": "same_normal_form_kind_and_exact_parameter_equality",
            "parameter_residuals": residuals,
            "schema_version": CERTIFICATE_SCHEMA,
            "target_sha256": canonical_sha256(target),
        }
        return {
            "counterexample": None,
            "proof_certificate": {**body, "content_sha256": canonical_sha256(body)},
            "reason_codes": [],
            "status": "PASS",
        }
    target_for_evaluation = {
        "kind": target["class"],
        "parameters": target["parameters"],
    }
    for point in target["holdout_points"]:
        if not isinstance(point, int) or isinstance(point, bool):
            raise NativeIndependenceError("holdout point changed")
        observed = _evaluate_candidate(candidate, point)
        expected = _evaluate_candidate(target_for_evaluation, point)
        if observed != expected:
            return {
                "counterexample": {
                    "candidate_value": None if observed is None else _fraction_data(observed),
                    "point": point,
                    "residual": (
                        None
                        if observed is None or expected is None
                        else _fraction_data(observed - expected)
                    ),
                    "target_value": None if expected is None else _fraction_data(expected),
                },
                "proof_certificate": None,
                "reason_codes": ["exact_hidden_holdout_counterexample"],
                "status": "REJECT",
            }
    raise NativeIndependenceError("nonidentical candidate lacked exact counterexample")


def _leave_one_family_out(
    world_results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for excluded in FAMILIES:
        class_coverage: dict[str, bool] = {}
        for world in world_results:
            class_coverage[world["class"]] = any(
                result["status"] == "PASS" and result["family"] != excluded
                for result in world["candidate_results"]
            )
        rows.append(
            {
                "all_declared_classes_covered": all(class_coverage.values()),
                "class_coverage": dict(sorted(class_coverage.items())),
                "excluded_family": excluded,
            }
        )
    return {
        "decision": (
            "PASS" if all(row["all_declared_classes_covered"] for row in rows) else "BLOCK"
        ),
        "rows": rows,
    }


def _bindings(root: Path, target_file_sha256: str) -> dict[str, dict[str, str]]:
    paths = {
        "config": CONFIG_PATH,
        "documentation": DOC_PATH,
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    bindings = {
        role: {"file_sha256": _file_sha256(_resolve(root, path)), "path": path}
        for role, path in sorted(paths.items())
    }
    bindings["target_fixture"] = {
        "file_sha256": target_file_sha256,
        "path": TARGET_PATH,
    }
    return bindings


def build_campaign(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    """Build the fixed campaign; the target fixture is read only after Phase A is sealed."""

    root = root.resolve()
    config = _load_json(config_path or _resolve(root, CONFIG_PATH))
    _validate_config(config)
    with _deny_target_fixture_reads(root) as access_audit:
        phase_a, outcomes = _prepare_phase_a(config)
        try:
            _resolve(root, TARGET_PATH).read_bytes()
        except PermissionError:
            pass
        else:
            raise NativeIndependenceError("pre-unseal target read was not denied")
    phase_a = _bind_phase_a_access_audit(phase_a, access_audit)
    if (
        phase_a["generic_exact_linear_solver_calls"] != 0
        or phase_a["bayesian_generator_calls"] != 0
    ):
        raise NativeIndependenceError("prohibited generator or solver call budget exceeded")
    targets, target_file_sha256 = _unseal_targets(root, config, phase_a)
    world_results: list[dict[str, Any]] = []
    for world in config["worlds"]:
        candidate_results: list[dict[str, Any]] = []
        for outcome in outcomes[world["world_id"]]:
            if outcome["candidate"] is None:
                candidate_results.append(
                    {
                        "blocker": outcome["blocker"],
                        "family": outcome["family"],
                        "status": "BLOCK",
                        "work_units": outcome["work_units"],
                    }
                )
            else:
                candidate_results.append(
                    {
                        "candidate": outcome["candidate"],
                        "family": outcome["family"],
                        "work_units": outcome["work_units"],
                        **_assess(outcome["candidate"], targets[world["world_id"]]),
                    }
                )
        counts = Counter(row["status"].lower() for row in candidate_results)
        world_results.append(
            {
                "candidate_results": candidate_results,
                "class": world["class"],
                "counts": {
                    "block": counts["block"],
                    "pass": counts["pass"],
                    "reject": counts["reject"],
                },
                "public_rows_sha256": canonical_sha256(world["public_rows"]),
                "sealed_target_sha256": world["sealed_target_sha256"],
                "unsealed_target": targets[world["world_id"]],
                "world_id": world["world_id"],
            }
        )
    ablation = _leave_one_family_out(world_results)
    pass_counts = [world["counts"]["pass"] for world in world_results]
    decision = (
        "PASS"
        if ablation["decision"] == "PASS" and all(count >= 2 for count in pass_counts)
        else "BLOCK"
    )
    totals = Counter(
        row["status"].lower() for world in world_results for row in world["candidate_results"]
    )
    body = {
        "campaign_id": CAMPAIGN_ID,
        "chronology": [
            {
                "event": "public_constraints_seeds_budgets_and_code_identities_frozen",
                "target_reads": 0,
            },
            {"event": "all_native_generator_outcomes_frozen", "target_reads": 0},
            {
                "candidate_set_sha256": phase_a["content_sha256"],
                "event": "phase_a_receipt_sealed",
                "target_reads": 0,
            },
            {"event": "atomic_two_record_target_unseal", "target_reads": 1},
            {"event": "exact_assessment_and_leave_one_family_out", "target_reads": 1},
        ],
        "claims": CLAIMS,
        "counts": {
            "bayesian_generator_calls": 0,
            "candidate_blocks": totals["block"],
            "candidate_passes": totals["pass"],
            "candidate_rejects": totals["reject"],
            "declared_benchmark_classes": len(WORLD_CLASSES),
            "exact_counterexamples": totals["reject"],
            "exact_identity_certificates": totals["pass"],
            "formula_discovery_job_delegations": 0,
            "generator_families": len(FAMILIES),
            "generic_exact_linear_solver_calls": 0,
            "post_unseal_generation_events": 0,
            "target_fixture_reads": 1,
            "target_fixture_reads_denied_before_unseal": 1,
            "worlds": len(world_results),
        },
        "decision": decision,
        "first_blocker": (
            None
            if decision == "PASS"
            else "leave_one_family_out_or_multiple_recovery_requirement_failed"
        ),
        "leave_one_family_out": ablation,
        "phase_a": phase_a,
        "schema_version": RESULT_SCHEMA,
        "scope": (
            "Two fixed synthetic non-polynomial structured worlds and three frozen native, "
            "non-Bayesian constructors. PASS establishes bounded prospective generator-family "
            "independence on these declared classes only; it does not establish general formula "
            "discovery, novelty, or scientific truth."
        ),
        "source_bindings": _bindings(root, target_file_sha256),
        "world_results": world_results,
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_campaign(
    value: Mapping[str, Any], *, root: Path, config_path: Path | None = None
) -> None:
    """Reject any tamper or environmental drift by exact deterministic replay."""

    if value.get("schema_version") != RESULT_SCHEMA:
        raise NativeIndependenceError("receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise NativeIndependenceError("receipt seal changed")
    if dict(value) != build_campaign(root, config_path):
        raise NativeIndependenceError("receipt exact replay changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    encoded = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != encoded:
            raise NativeIndependenceError("refusing to overwrite immutable receipt")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default=OUTPUT_PATH)
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output = _resolve(root, args.output)
    if args.validate_checked:
        validate_campaign(_load_json(output), root=root)
        return 0
    result = build_campaign(root)
    _write_immutable(output, result)
    validate_campaign(result, root=root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
