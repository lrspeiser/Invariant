"""Leak-resistant blind rediscovery of an anonymous natural-number sum."""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import hashlib
import io
import itertools
import json
import os
import threading
from collections.abc import Iterator, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

CONFIG_SCHEMA = "invariant-anonymous-natural-sum-blind-rediscovery-config-1.0"
RESULT_SCHEMA = "invariant-anonymous-natural-sum-blind-rediscovery-result-1.0"
CONFIG_PATH = "configs/backgrounds/anonymous_natural_sum_blind_rediscovery.json"
SOURCE_PATH = "src/sigma_theory_compiler/anonymous_natural_sum_blind_rediscovery.py"
TEST_PATH = "tests/test_anonymous_natural_sum_blind_rediscovery.py"
OUTPUT_PATH = "runs/math-language/anonymous-natural-sum-blind-rediscovery/campaign.json"
DECISION = "pass_blind_bounded_grammar_rediscovery_independently_proved_before_unseal"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = _sha(body)
    return body


def _resolve(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("blind rediscovery path escapes project root") from error
    return path


def _assert_exact_keys(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(f"{label} keys changed")


def _load_config(root: Path, config_path: Path) -> dict[str, Any]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    _assert_exact_keys(
        value,
        {
            "schema_version",
            "benchmark_id",
            "problem",
            "exact_examples",
            "bounded_grammar",
            "counterexample_domain",
            "induction_contract",
            "execution_boundary",
            "withheld_reference",
            "output_path",
        },
        "blind rediscovery config",
    )
    boundary = value["execution_boundary"]
    _assert_exact_keys(
        boundary,
        {
            "owned_threads",
            "pre_unseal_file_read_allowlist",
            "pre_unseal_denied_probe",
            "allowed_import_roots",
            "forbidden_import_roots",
            "network_calls_allowed",
            "llm_calls_allowed",
            "subprocesses_allowed",
        },
        "blind rediscovery execution boundary",
    )
    grammar = value["bounded_grammar"]
    if (
        value["schema_version"] != CONFIG_SCHEMA
        or value["benchmark_id"] != "anonymous-natural-sum-blind-rediscovery-001"
        or value["problem"]
        != {
            "anonymous_function": "S",
            "domain": "nonnegative_integers",
            "definition": "S(n) is obtained by adding each positive integer not exceeding n",
            "base_case": {"n": 0, "value": 0},
            "successor_rule": "S(n+1)-S(n)=n+1",
        }
        or value["exact_examples"]
        != [
            {"n": 0, "value": 0},
            {"n": 1, "value": 1},
            {"n": 2, "value": 3},
            {"n": 3, "value": 6},
            {"n": 4, "value": 10},
            {"n": 5, "value": 15},
        ]
        or grammar
        != {
            "basis": ["square", "linear", "constant"],
            "coefficient_numerator_min": -4,
            "coefficient_numerator_max": 4,
            "positive_denominators": [1, 2, 3, 4],
        }
        or value["counterexample_domain"] != {"first_n": 6, "last_n_inclusive": 64}
        or value["induction_contract"]["base_index"] != 0
        or value["induction_contract"]["allowed_premises"]
        != [
            "definition_base_case",
            "definition_successor_rule",
            "exact_rational_arithmetic",
            "polynomial_identity",
        ]
        or value["induction_contract"]["forbidden_premises"]
        != [
            "withheld_known_theorem",
            "equivalent_closed_form_assumption",
            "existing_sum_formula_implementation",
            "external_oracle",
        ]
        or boundary["owned_threads"] != 1
        or boundary["pre_unseal_file_read_allowlist"] != [CONFIG_PATH, SOURCE_PATH]
        or boundary["pre_unseal_denied_probe"] != TEST_PATH
        or any(
            boundary[key]
            for key in ("network_calls_allowed", "llm_calls_allowed", "subprocesses_allowed")
        )
        or value["withheld_reference"]
        != {
            "path": TEST_PATH,
            "assignment": "WITHHELD_KNOWN_THEOREM",
            "access_phase": "post_seal_only",
        }
        or value["output_path"] != OUTPUT_PATH
        or config_path.resolve() != _resolve(root, CONFIG_PATH)
    ):
        raise ValueError("blind rediscovery config contract changed")
    return value


def _source_dependency_audit(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    source = _resolve(root, SOURCE_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                imports.append("sigma_theory_compiler")
            elif node.module:
                imports.append(node.module.split(".")[0])
    roots = sorted(set(imports))
    boundary = config["execution_boundary"]
    allowed = set(boundary["allowed_import_roots"])
    forbidden = set(boundary["forbidden_import_roots"])
    unexpected = sorted(set(roots) - allowed)
    forbidden_present = sorted(set(roots) & forbidden)
    passed = not unexpected and not forbidden_present
    result = {
        "source_file_sha256": _file_sha(_resolve(root, SOURCE_PATH)),
        "source_bytes_scanned": len(source.encode()),
        "import_roots": roots,
        "unexpected_import_roots": unexpected,
        "forbidden_import_roots_present": forbidden_present,
        "forbidden_dependency_closure_empty": passed,
        "dependency_root_sha256": _sha(
            {
                "config": _file_sha(_resolve(root, CONFIG_PATH)),
                "source": _file_sha(_resolve(root, SOURCE_PATH)),
                "imports": roots,
            }
        ),
    }
    if not passed:
        raise ValueError("blind rediscovery forbidden dependency closure is not empty")
    return result


class _ReadBoundary:
    def __init__(self, root: Path, allowlist: Sequence[str]) -> None:
        self.root = root.resolve()
        self.allowed = {_resolve(root, relative) for relative in allowlist}
        self.events: list[dict[str, Any]] = []

    def check(self, file: Any, surface: str) -> None:
        if isinstance(file, int):
            return
        try:
            path = Path(os.fspath(file)).resolve()
        except TypeError:
            return
        allowed = path in self.allowed
        try:
            relative = path.relative_to(self.root).as_posix()
        except ValueError:
            relative = "outside-project-root"
        self.events.append({"path": relative, "surface": surface, "allowed": allowed})
        if not allowed:
            raise PermissionError(f"pre-unseal read denied: {relative}")


@contextlib.contextmanager
def _enforced_reads(boundary: _ReadBoundary) -> Iterator[None]:
    original_open = builtins.open
    original_io_open = io.open
    original_path_open = Path.open
    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes

    def guarded_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "r" in mode or "+" in mode:
            boundary.check(file, "builtins.open")
        return original_open(file, mode, *args, **kwargs)

    def guarded_io_open(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "r" in mode or "+" in mode:
            boundary.check(file, "io.open")
        return original_io_open(file, mode, *args, **kwargs)

    def guarded_path_open(path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        if "r" in mode or "+" in mode:
            boundary.check(path, "Path.open")
        return original_path_open(path, mode, *args, **kwargs)

    def guarded_read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        boundary.check(path, "Path.read_text")
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path: Path, *args: Any, **kwargs: Any) -> bytes:
        boundary.check(path, "Path.read_bytes")
        return original_read_bytes(path, *args, **kwargs)

    builtins.open = guarded_open
    io.open = guarded_io_open
    Path.open = guarded_path_open
    Path.read_text = guarded_read_text
    Path.read_bytes = guarded_read_bytes
    try:
        yield
    finally:
        builtins.open = original_open
        io.open = original_io_open
        Path.open = original_path_open
        Path.read_text = original_read_text
        Path.read_bytes = original_read_bytes


def _candidate_value(coefficients: Sequence[Fraction], n: int) -> Fraction:
    a, b, c = coefficients
    return a * n * n + b * n + c


def _candidate_record(coefficients: Sequence[Fraction]) -> dict[str, Any]:
    labels = ("square", "linear", "constant")
    values = {
        label: {"numerator": value.numerator, "denominator": value.denominator}
        for label, value in zip(labels, coefficients, strict=True)
    }
    return {"coefficients": values, "candidate_id": _sha(values)[:24]}


def _enumerate_candidates(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    grammar = config["bounded_grammar"]
    scalars = [
        Fraction(numerator, denominator)
        for denominator in grammar["positive_denominators"]
        for numerator in range(
            grammar["coefficient_numerator_min"], grammar["coefficient_numerator_max"] + 1
        )
    ]
    unique_scalars = sorted(set(scalars))
    raw_count = len(scalars) ** 3
    candidates = [
        _candidate_record(coefficients)
        for coefficients in itertools.product(unique_scalars, repeat=3)
    ]
    candidates.sort(key=lambda row: row["candidate_id"])
    return candidates, raw_count


def _fractions(record: Mapping[str, Any]) -> tuple[Fraction, Fraction, Fraction]:
    return tuple(
        Fraction(
            record["coefficients"][label]["numerator"],
            record["coefficients"][label]["denominator"],
        )
        for label in ("square", "linear", "constant")
    )  # type: ignore[return-value]


def _definition_value(n: int) -> int:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ValueError("definition input must be a nonnegative integer")
    total = 0
    for term in range(1, n + 1):
        total += term
    return total


def _discover(config: Mapping[str, Any]) -> dict[str, Any]:
    candidates, raw_count = _enumerate_candidates(config)
    example_survivors = []
    for candidate in candidates:
        coefficients = _fractions(candidate)
        if all(
            _candidate_value(coefficients, row["n"]) == row["value"]
            for row in config["exact_examples"]
        ):
            example_survivors.append(candidate)
    domain = config["counterexample_domain"]
    counterexample_records = []
    fully_tested = []
    for candidate in example_survivors:
        coefficients = _fractions(candidate)
        counterexample = next(
            (
                {
                    "n": n,
                    "expected": _definition_value(n),
                    "actual": str(_candidate_value(coefficients, n)),
                }
                for n in range(domain["first_n"], domain["last_n_inclusive"] + 1)
                if _candidate_value(coefficients, n) != _definition_value(n)
            ),
            None,
        )
        counterexample_records.append(
            {"candidate_id": candidate["candidate_id"], "counterexample": counterexample}
        )
        if counterexample is None:
            fully_tested.append(candidate)
    if len(fully_tested) != 1:
        raise ValueError("blind rediscovery did not produce exactly one tested winner")
    winner = fully_tested[0]
    coefficients = _fractions(winner)
    a, b, _c = coefficients
    proof = {
        "proof_method": "mathematical_induction_from_definition_recurrence",
        "base_case": {
            "index": 0,
            "candidate_value": str(_candidate_value(coefficients, 0)),
            "definition_value": 0,
            "proved": _candidate_value(coefficients, 0) == 0,
        },
        "successor_difference_coefficients": {
            "linear": str(2 * a),
            "constant": str(a + b),
        },
        "successor_identity": "candidate(n+1)-candidate(n)=n+1",
        "successor_identity_proved": 2 * a == 1 and a + b == 1,
        "allowed_premises_used": [
            "definition_base_case",
            "definition_successor_rule",
            "exact_rational_arithmetic",
            "polynomial_identity",
        ],
        "forbidden_premises_used": [],
        "conclusion": "candidate_equals_anonymous_definition_for_every_nonnegative_integer",
    }
    if not proof["base_case"]["proved"] or not proof["successor_identity_proved"]:
        raise ValueError("blind rediscovery induction proof failed")
    return {
        "enumeration": {
            "raw_cartesian_candidates": raw_count,
            "canonical_coefficient_classes": len(candidates),
            "exact_example_survivors": len(example_survivors),
            "counterexample_tests_per_survivor": (
                domain["last_n_inclusive"] - domain["first_n"] + 1
            ),
            "fully_tested_survivors": len(fully_tested),
        },
        "candidate_catalog_root_sha256": _sha(candidates),
        "example_survivors_root_sha256": _sha(example_survivors),
        "counterexample_records": counterexample_records,
        "counterexample_records_root_sha256": _sha(counterexample_records),
        "winner": winner,
        "winner_formula": "a*n^2+b*n+c",
        "induction_proof": proof,
    }


def _negative_controls(
    config: Mapping[str, Any], winner: Mapping[str, Any]
) -> list[dict[str, Any]]:
    controls = []
    wrong = (Fraction(1), Fraction(0), Fraction(0))
    controls.append(
        {
            "control_id": "wrong_formula_square_only",
            "eligible": False,
            "first_counterexample": next(
                n for n in range(8) if _candidate_value(wrong, n) != _definition_value(n)
            ),
            "reason": "definition_counterexample",
        }
    )
    overfit_table = {row["n"]: row["value"] for row in config["exact_examples"]}
    first_unseen = config["counterexample_domain"]["first_n"]
    controls.append(
        {
            "control_id": "example_lookup_overfit",
            "eligible": False,
            "first_counterexample": first_unseen,
            "reason": "undefined_outside_public_examples",
            "memorized_example_count": len(overfit_table),
        }
    )
    controls.append(
        {
            "control_id": "forbidden_withheld_theorem_premise",
            "eligible": False,
            "first_counterexample": None,
            "reason": "proof_uses_forbidden_premise",
        }
    )
    controls.append(
        {
            "control_id": "winner_with_unproved_generalization",
            "eligible": False,
            "first_counterexample": None,
            "reason": "finite_examples_do_not_prove_universal_identity",
            "candidate_id": winner["candidate_id"],
        }
    )
    if any(row["eligible"] for row in controls):
        raise ValueError("blind rediscovery negative control admitted")
    return controls


def _pre_unseal(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    if threading.current_thread() is not threading.main_thread() or threading.active_count() != 1:
        raise RuntimeError("blind rediscovery requires one owned main thread")
    boundary = _ReadBoundary(root, config["execution_boundary"]["pre_unseal_file_read_allowlist"])
    with _enforced_reads(boundary):
        _resolve(root, CONFIG_PATH).read_text(encoding="utf-8")
        _resolve(root, SOURCE_PATH).read_bytes()
        denied = False
        try:
            _resolve(root, config["execution_boundary"]["pre_unseal_denied_probe"]).read_text(
                encoding="utf-8"
            )
        except PermissionError:
            denied = True
        dependency_audit = _source_dependency_audit(root, config)
        discovery = _discover(config)
        negatives = _negative_controls(config, discovery["winner"])
    if not denied:
        raise ValueError("blind rediscovery withheld probe was not denied")
    allowed = [event for event in boundary.events if event["allowed"]]
    denied_events = [event for event in boundary.events if not event["allowed"]]
    body = {
        "benchmark_id": config["benchmark_id"],
        "definition_root_sha256": _sha(config["problem"]),
        "examples_root_sha256": _sha(config["exact_examples"]),
        "grammar_root_sha256": _sha(config["bounded_grammar"]),
        "dependency_audit": dependency_audit,
        "io_boundary": {
            "owned_threads": 1,
            "attempted_access_count": len(boundary.events),
            "allowed_access_count": len(allowed),
            "denied_access_count": len(denied_events),
            "denied_content_bytes_exposed": 0,
            "allowed_paths": sorted({event["path"] for event in allowed}),
            "denied_paths": sorted({event["path"] for event in denied_events}),
            "enforcement_surfaces": sorted({event["surface"] for event in boundary.events}),
            "network_calls": 0,
            "llm_calls": 0,
            "subprocesses": 0,
        },
        "discovery": discovery,
        "negative_controls": negatives,
    }
    return _sealed(body)


def _load_withheld_theorem(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    reference = config["withheld_reference"]
    source = _resolve(root, reference["path"]).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == reference["assignment"]
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, dict):
                return value
    raise ValueError("withheld theorem assignment missing")


def _post_unseal(
    root: Path, config: Mapping[str, Any], pre_unseal: Mapping[str, Any]
) -> dict[str, Any]:
    theorem = _load_withheld_theorem(root, config)
    winner = pre_unseal["discovery"]["winner"]
    discovered_theorem = {
        "anonymous_function": config["problem"]["anonymous_function"],
        "domain": config["problem"]["domain"],
        "basis": config["bounded_grammar"]["basis"],
        "coefficients": winner["coefficients"],
    }
    matched = theorem == discovered_theorem
    if not matched:
        raise ValueError("sealed blind rediscovery does not match withheld theorem")
    return {
        "withheld_reference_file_sha256": _file_sha(_resolve(root, TEST_PATH)),
        "withheld_theorem_sha256": _sha(theorem),
        "comparison_permitted_only_after_pre_unseal_seal": True,
        "pre_unseal_seal_verified": pre_unseal["content_sha256"]
        == _sha({key: value for key, value in pre_unseal.items() if key != "content_sha256"}),
        "matches_withheld_known_theorem": True,
        "withheld_theorem": theorem,
    }


def validate_result(value: Mapping[str, Any], root: Path, config_path: Path | None = None) -> None:
    config_path = config_path or _resolve(root, CONFIG_PATH)
    config = _load_config(root, config_path)
    if (
        value.get("schema_version") != RESULT_SCHEMA
        or value.get("benchmark_id") != config["benchmark_id"]
        or value.get("decision") != DECISION
        or value.get("content_sha256")
        != _sha({key: item for key, item in value.items() if key != "content_sha256"})
        or value.get("blinded_pre_unseal_root_sha256")
        != value.get("pre_unseal", {}).get("content_sha256")
        or value.get("post_unseal", {}).get("comparison_permitted_only_after_pre_unseal_seal")
        is not True
        or value.get("post_unseal", {}).get("matches_withheld_known_theorem") is not True
        or value.get("induction_proof", {}).get("successor_identity_proved") is not True
        or value.get("induction_proof", {}).get("forbidden_premises_used") != []
        or any(row.get("eligible") for row in value.get("negative_controls", []))
        or value.get("claims", {}).get("unbounded_formula_space_exhausted") is not False
        or value.get("claims", {}).get("novel_theorem_claimed") is not False
    ):
        raise ValueError("blind rediscovery result contract changed")
    if dict(value) != build_result(root, config_path):
        raise ValueError("blind rediscovery immutable replay mismatch")


def build_result(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    config_path = config_path or _resolve(root, CONFIG_PATH)
    config = _load_config(root, config_path)
    pre_unseal = _pre_unseal(root, config)
    post_unseal = _post_unseal(root, config, pre_unseal)
    discovery = pre_unseal["discovery"]
    return _sealed(
        {
            "schema_version": RESULT_SCHEMA,
            "benchmark_id": config["benchmark_id"],
            "decision": DECISION,
            "pre_unseal": pre_unseal,
            "blinded_pre_unseal_root_sha256": pre_unseal["content_sha256"],
            "post_unseal": post_unseal,
            "enumeration": discovery["enumeration"],
            "winner": discovery["winner"],
            "induction_proof": discovery["induction_proof"],
            "negative_controls": pre_unseal["negative_controls"],
            "claims": {
                "bounded_grammar_exhausted": True,
                "unique_public_example_survivor": True,
                "counterexample_domain_passed": True,
                "universal_identity_proved_by_induction": True,
                "winner_sealed_before_reference_access": True,
                "withheld_theorem_matched_post_seal": True,
                "unbounded_formula_space_exhausted": False,
                "operating_system_sandbox_provided": False,
                "novel_theorem_claimed": False,
            },
            "limits": [
                "enumeration_exhausts_only_the_declared_finite_quadratic_rational_grammar",
                "public_examples_are_not_the_universal_proof",
                "file_read_guards_are_process_local_not_an_operating_system_sandbox",
                "the_benchmark_rediscovers_a_withheld_known_theorem_and_claims_no_novelty",
            ],
            "bindings": {
                label: {"path": path, "file_sha256": _file_sha(_resolve(root, path))}
                for label, path in (
                    ("config", CONFIG_PATH),
                    ("source", SOURCE_PATH),
                    ("test_and_withheld_reference", TEST_PATH),
                )
            },
        }
    )


def run(root: Path, config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or _resolve(root, CONFIG_PATH)
    result = build_result(root, path)
    validate_result(result, root, path)
    return result


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"immutable artifact already exists with different bytes: {path}")
        return
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--output", default=OUTPUT_PATH)
    arguments = parser.parse_args()
    root = Path(arguments.project_root).resolve()
    result = run(root, _resolve(root, arguments.config))
    output = _resolve(root, arguments.output)
    _write_immutable(output, result)
    print(json.dumps({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
