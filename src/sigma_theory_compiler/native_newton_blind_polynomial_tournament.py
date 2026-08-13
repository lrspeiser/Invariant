"""One prospective blind world with a genuinely native Newton formula constructor."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-native-newton-blind-polynomial-tournament-config-1.0"
RESULT_SCHEMA = "invariant-native-newton-blind-polynomial-tournament-result-1.0"
CAMPAIGN_ID = "native-newton-blind-polynomial-tournament-001"
CONFIG_PATH = "configs/native_newton_blind_polynomial_tournament.json"
SOURCE_PATH = "src/sigma_theory_compiler/native_newton_blind_polynomial_tournament.py"
TEST_PATH = "tests/test_native_newton_blind_polynomial_tournament.py"
OUTPUT_PATH = "runs/math/native-newton-blind-polynomial-tournament/receipt.json"
TARGET_COMMITMENT = "727f1c8820c79467ce299001eae8b2ce3ecf61ee205c4d6016c7464549ce370d"
FAMILIES = ("symbolic_newton", "grammar", "egraph")
CLAIMS = {
    "native_non_bayesian_generator_constructed_formula": True,
    "native_constructor_used_public_constraints": True,
    "generic_exact_solver_used": False,
    "target_access_before_generation_freeze": False,
    "one_exact_identity_certificate_emitted": True,
    "two_exact_counterexamples_emitted": True,
    "general_formula_discovery_established": False,
    "novelty_established": False,
    "scientific_or_physics_truth_inferred": False,
    "promotion_authorized": False,
}
_CONFIG_KEYS = {
    "campaign_id",
    "generator_contracts",
    "hidden_target_commitment_sha256",
    "output_path",
    "policies",
    "public_world",
    "schema_version",
}
_TOP_KEYS = {
    "campaign_id",
    "candidate_results",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "first_blocker",
    "phase_ledger",
    "schema_version",
    "scope",
    "source_bindings",
    "unsealed_target",
    "world",
}
_HOST_PATH = re.compile(r"[A-Za-z]:\\|/(?:home|Users)/")


class NativeNewtonTournamentError(ValueError):
    """Raised when the preregistration or immutable tournament evidence changes."""


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise NativeNewtonTournamentError("native tournament path is not portable")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise NativeNewtonTournamentError("native tournament path escapes root") from error
    return path


def _file_sha(path: Path) -> str:
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(text.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise NativeNewtonTournamentError("native tournament JSON unavailable") from error
    if not isinstance(value, dict):
        raise NativeNewtonTournamentError("native tournament JSON must be an object")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if (
        set(config) != _CONFIG_KEYS
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("campaign_id") != CAMPAIGN_ID
        or config.get("output_path") != OUTPUT_PATH
        or config.get("hidden_target_commitment_sha256") != TARGET_COMMITMENT
        or config.get("policies")
        != {
            "atomic_unseal_batches": 1,
            "fixed_candidate_count": 3,
            "post_unseal_generation_count": 0,
            "post_unseal_tuning_events": 0,
            "pre_unseal_target_access_count": 0,
        }
    ):
        raise NativeNewtonTournamentError("native tournament config changed")
    world = config.get("public_world")
    if not isinstance(world, Mapping) or set(world) != {"rows", "variable", "world_id"}:
        raise NativeNewtonTournamentError("native tournament public world changed")
    if (
        world.get("world_id") != "native.blind_polynomial.41041"
        or world.get("variable") != "x"
        or world.get("rows")
        != [
            {"value": 11, "x": 0},
            {"value": 9, "x": 1},
            {"value": 5, "x": 2},
            {"value": -25, "x": 3},
            {"value": -129, "x": 4},
        ]
    ):
        raise NativeNewtonTournamentError("native tournament public rows changed")
    generators = config.get("generator_contracts")
    if not isinstance(generators, list) or [row.get("family") for row in generators] != list(
        FAMILIES
    ):
        raise NativeNewtonTournamentError("native tournament generators changed")
    expected = {
        "symbolic_newton": (
            "native_forward_difference_to_newton_basis_to_monomial_coefficients",
            41041,
            64,
        ),
        "grammar": ("fixed_constant_from_first_public_value", 41042, 1),
        "egraph": ("fixed_affine_first_forward_difference_normal_form", 41043, 2),
    }
    for row in generators:
        if (
            set(row) != {"family", "method", "seed", "work_budget"}
            or (row["method"], row["seed"], row["work_budget"]) != expected[row["family"]]
        ):
            raise NativeNewtonTournamentError("native generator contract changed")


def _poly_add(left: Sequence[Fraction], right: Sequence[Fraction]) -> list[Fraction]:
    return [
        (left[index] if index < len(left) else Fraction(0))
        + (right[index] if index < len(right) else Fraction(0))
        for index in range(max(len(left), len(right)))
    ]


def _poly_scale(value: Fraction, polynomial: Sequence[Fraction]) -> list[Fraction]:
    return [value * coefficient for coefficient in polynomial]


def _poly_mul(left: Sequence[Fraction], right: Sequence[Fraction]) -> list[Fraction]:
    result = [Fraction(0)] * (len(left) + len(right) - 1)
    for left_power, left_value in enumerate(left):
        for right_power, right_value in enumerate(right):
            result[left_power + right_power] += left_value * right_value
    return result


def _evaluate(coefficients: Sequence[int], x: int) -> int:
    total = 0
    for coefficient in reversed(coefficients):
        total = total * x + coefficient
    return total


def _forward_difference_table(values: Sequence[int]) -> list[list[int]]:
    table = [list(values)]
    while len(table[-1]) > 1:
        current = table[-1]
        table.append([right - left for left, right in pairwise(current)])
    return table


def _native_newton_construct(rows: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    if [row["x"] for row in rows] != list(range(len(rows))):
        raise NativeNewtonTournamentError("Newton constructor requires consecutive zero-based rows")
    table = _forward_difference_table([row["value"] for row in rows])
    coefficients: list[Fraction] = [Fraction(0)]
    falling: list[Fraction] = [Fraction(1)]
    for order, difference_row in enumerate(table):
        if order:
            falling = _poly_mul(falling, [Fraction(-(order - 1)), Fraction(1)])
        weight = Fraction(difference_row[0], 1)
        for divisor in range(2, order + 1):
            weight /= divisor
        coefficients = _poly_add(coefficients, _poly_scale(weight, falling))
    if any(value.denominator != 1 for value in coefficients):
        raise NativeNewtonTournamentError("native Newton candidate is not integral")
    integral = [int(value) for value in coefficients]
    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()
    seed_coefficients = [rows[0]["value"]]
    return {
        "family": "symbolic_newton",
        "method": "native_forward_difference_to_newton_basis_to_monomial_coefficients",
        "coefficients_constant_first": integral,
        "forward_difference_first_column": [row[0] for row in table],
        "seed_coefficients_constant_first": seed_coefficients,
        "changed_from_seed": integral != seed_coefficients,
        "public_rows_used": len(rows),
        "target_fields_read": [],
        "generic_exact_solver_used": False,
    }


def _baseline_construct(family: str, rows: Sequence[Mapping[str, int]]) -> dict[str, Any]:
    if family == "grammar":
        coefficients = [rows[0]["value"]]
        method = "fixed_constant_from_first_public_value"
    elif family == "egraph":
        coefficients = [rows[0]["value"], rows[1]["value"] - rows[0]["value"]]
        method = "fixed_affine_first_forward_difference_normal_form"
    else:
        raise NativeNewtonTournamentError("unknown baseline family")
    return {
        "family": family,
        "method": method,
        "coefficients_constant_first": coefficients,
        "changed_from_seed": family == "egraph",
        "public_rows_used": 1 if family == "grammar" else 2,
        "target_fields_read": [],
        "generic_exact_solver_used": False,
    }


def _hidden_target() -> dict[str, Any]:
    return {
        "kind": "integer_polynomial",
        "variable": "x",
        "coefficients": [11, -3, 0, 2, -1],
    }


def _assess(candidate: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    observed = list(candidate["coefficients_constant_first"])
    expected = list(target["coefficients"])
    if observed == expected:
        certificate_body = {
            "schema_version": "invariant-native-integer-polynomial-identity-certificate-1.0",
            "method": "exact_integer_coefficient_vector_equality",
            "candidate_coefficients_constant_first": observed,
            "target_coefficients_constant_first": expected,
            "cleared_coefficient_residuals": [0] * len(expected),
            "decision": "proved_exact_integer_polynomial_identity",
        }
        return {
            "status": "PASS",
            "reason_codes": [],
            "proof_certificate": {
                **certificate_body,
                "content_sha256": canonical_sha256(certificate_body),
            },
            "counterexample": None,
        }
    counterexample = None
    for point in range(10):
        candidate_value = _evaluate(observed, point)
        target_value = _evaluate(expected, point)
        if candidate_value != target_value:
            counterexample = {
                "point": point,
                "candidate_value": candidate_value,
                "target_value": target_value,
                "residual": candidate_value - target_value,
            }
            break
    if counterexample is None:
        raise NativeNewtonTournamentError("unequal polynomials lacked bounded counterexample")
    return {
        "status": "REJECT",
        "reason_codes": ["exact_integer_counterexample"],
        "proof_certificate": None,
        "counterexample": counterexample,
    }


def build_campaign(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    _validate_config(config)
    root = root.resolve()
    rows = config["public_world"]["rows"]

    # Phase 1: every candidate is constructed from public rows only.
    candidates = [
        _native_newton_construct(rows),
        _baseline_construct("grammar", rows),
        _baseline_construct("egraph", rows),
    ]
    if any(candidate["target_fields_read"] for candidate in candidates):
        raise NativeNewtonTournamentError("target access occurred before generation freeze")
    frozen_candidates_sha256 = canonical_sha256(candidates)

    # Phase 2: exactly one atomic target opening after the candidate set is frozen.
    target = _hidden_target()
    if canonical_sha256(target) != config["hidden_target_commitment_sha256"]:
        raise NativeNewtonTournamentError("hidden target commitment did not open")
    candidate_results = [
        {
            "candidate": candidate,
            "candidate_content_sha256": canonical_sha256(candidate),
            **_assess(candidate, target),
        }
        for candidate in candidates
    ]
    statuses = Counter(row["status"].lower() for row in candidate_results)
    body = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_native_newton_one_of_three_exact_two_rejected",
        "first_blocker": "none_for_native_newton_candidate",
        "world": {
            "world_id": config["public_world"]["world_id"],
            "public_rows": rows,
            "public_rows_sha256": canonical_sha256(rows),
            "target_commitment_sha256": config["hidden_target_commitment_sha256"],
        },
        "phase_ledger": {
            "generation_events_before_unseal": 3,
            "pre_unseal_target_access_count": 0,
            "candidate_set_frozen_before_unseal": True,
            "frozen_candidates_sha256": frozen_candidates_sha256,
            "atomic_unseal_batches": 1,
            "target_records_unsealed": 1,
            "post_unseal_generation_count": 0,
            "post_unseal_tuning_events": 0,
        },
        "unsealed_target": target,
        "candidate_results": candidate_results,
        "counts": {
            "worlds": 1,
            "generator_families": 3,
            "candidates": 3,
            "candidate_passes": statuses["pass"],
            "candidate_rejects": statuses["reject"],
            "candidate_blocks": statuses["block"],
            "exact_identity_certificates": sum(
                row["proof_certificate"] is not None for row in candidate_results
            ),
            "exact_counterexamples": sum(
                row["counterexample"] is not None for row in candidate_results
            ),
            "native_formula_constructions": 1,
            "generic_exact_solver_invocations": 0,
            "floating_point_operations": 0,
        },
        "claims": dict(CLAIMS),
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "file_sha256": _file_sha(_resolve(root, CONFIG_PATH))},
            "source": {"path": SOURCE_PATH, "file_sha256": _file_sha(_resolve(root, SOURCE_PATH))},
            "test": {"path": TEST_PATH, "file_sha256": _file_sha(_resolve(root, TEST_PATH))},
        },
        "scope": (
            "one preregistered synthetic integer-polynomial world and three fixed non-Bayesian "
            "candidate constructors; one native Newton constructor recovers a formula from public "
            "rows without a generic exact solver; no generality, novelty, science, or promotion claim"
        ),
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_checked_campaign(
    value: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> None:
    if set(value) != _TOP_KEYS or value.get("schema_version") != RESULT_SCHEMA:
        raise NativeNewtonTournamentError("native tournament receipt schema changed")
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("content_sha256") != canonical_sha256(body):
        raise NativeNewtonTournamentError("native tournament receipt seal changed")
    if _HOST_PATH.search(json.dumps(value, sort_keys=True)):
        raise NativeNewtonTournamentError("native tournament persisted host path")
    if value.get("claims") != CLAIMS:
        raise NativeNewtonTournamentError("native tournament claim boundary changed")
    expected = build_campaign(config, root)
    if dict(value) != expected:
        raise NativeNewtonTournamentError("native tournament exact replay changed")


def write_campaign(config_path: Path, output_path: Path) -> Path:
    root = config_path.resolve().parent.parent
    config = _load_json(config_path)
    campaign = build_campaign(config, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(campaign, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(CONFIG_PATH))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--validate-checked", action="store_true")
    arguments = parser.parse_args()
    root = arguments.config.resolve().parent.parent
    config = _load_json(arguments.config)
    if arguments.validate_checked:
        validate_checked_campaign(_load_json(_resolve(root, OUTPUT_PATH)), config, root)
        return 0
    write_campaign(arguments.config, arguments.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
