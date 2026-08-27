"""Replayable pseudorandom formula search on exact and real measured-data problems."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import anonymous_monomial_discovery as monomial
from . import archimedes_real_data_confirmation as archimedes
from . import nasa_exoplanet_task1 as nasa
from .pseudorandom_ordinal import (
    ALGORITHM,
    PseudorandomChunkSchedule,
    PseudorandomOrdinalPermutation,
    build_prefix_receipt,
)
from .sigma_core import canonical_sha256

SCHEMA = "invariant-pseudorandom-formula-benchmarks-1.0"
OUTPUT_PATH = "runs/math/pseudorandom-formula-benchmarks/receipt-v1.json"
NASA_CONFIG_PATH = "configs/nasa_exoplanet_task1.json"
NASA_TRAINING_PATH = "runs/math/nasa-exoplanet-task1/exploratory-training-v1.json"
NASA_SNAPSHOT_PATH = "runs/math/nasa-exoplanet-task1/nasa-ps-source-snapshot-v1.csv"
ARCHIMEDES_CONFIG_PATH = "configs/archimedes_real_data_confirmation.json"
IMPLEMENTATION_PATHS = (
    "src/sigma_theory_compiler/pseudorandom_ordinal.py",
    "src/sigma_theory_compiler/pseudorandom_formula_benchmarks.py",
    "src/sigma_theory_compiler/anonymous_monomial_discovery.py",
    "src/sigma_theory_compiler/archimedes_real_data_confirmation.py",
    "src/sigma_theory_compiler/nasa_exoplanet_task1.py",
)
TRILLION_GRAMMAR_SIZE = 2_127_732_389_840


class PseudorandomFormulaBenchmarkError(ValueError):
    """A benchmark source, result, or replay seal changed."""


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise PseudorandomFormulaBenchmarkError(f"JSON root is not an object: {path}")
    return value


def _normalized_file_sha256(path: Path) -> str:
    return hashlib.sha256(
        path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    ).hexdigest()


def _synthetic_coefficients(ordinal: int) -> tuple[int, int, int, int]:
    coefficients = []
    for _ in range(4):
        ordinal, digit = divmod(ordinal, 11)
        coefficients.append(digit - 5)
    return tuple(coefficients)  # type: ignore[return-value]


def _polynomial_expression(coefficients: Sequence[int]) -> str:
    terms = []
    for degree, coefficient in enumerate(coefficients):
        if not coefficient:
            continue
        basis = "1" if degree == 0 else ("x" if degree == 1 else f"x^{degree}")
        terms.append(f"{coefficient:+d}*{basis}")
    return " ".join(terms).lstrip("+") or "0"


def _synthetic_polynomial_problem() -> dict[str, Any]:
    seed = "Invariant pseudorandom formula benchmark / exact cubic / v1"
    size = 11**4
    permutation = PseudorandomOrdinalPermutation(size, seed)
    target = (5, -2, 3, 0)
    xs = tuple(range(-4, 5))
    ys = tuple(sum(target[degree] * x**degree for degree in range(4)) for x in xs)
    best_key: tuple[Any, ...] | None = None
    best: tuple[int, ...] | None = None
    target_visit_position = None
    first_exact_position = None
    incumbent_updates = 0
    total_cycle_steps = 0
    for position in range(size):
        ordinal, steps = permutation.at_with_cycle_steps(position)
        total_cycle_steps += steps
        coefficients = _synthetic_coefficients(ordinal)
        residuals = [
            abs(
                sum(coefficients[degree] * x**degree for degree in range(4)) - expected
            )
            for x, expected in zip(xs, ys, strict=True)
        ]
        key = (sum(residuals), max(residuals), sum(map(abs, coefficients)), coefficients)
        if best_key is None or key < best_key:
            best_key = key
            best = coefficients
            incumbent_updates += 1
        if coefficients == target:
            target_visit_position = position
        if first_exact_position is None and not any(residuals):
            first_exact_position = position
    if best != target or target_visit_position is None or first_exact_position is None:
        raise PseudorandomFormulaBenchmarkError("exact polynomial recovery changed")
    return {
        "problem_id": "synthetic.exact-polynomial-coefficients",
        "source_kind": "synthetic_exact_integer_control",
        "grammar": {
            "formula": "c0 + c1*x + c2*x^2 + c3*x^3",
            "coefficient_alphabet": list(range(-5, 6)),
            "candidate_count": size,
        },
        "permutation": permutation.descriptor(),
        "search": {
            "candidates_tested": size,
            "first_exact_position_zero_based": first_exact_position,
            "target_visit_position_zero_based": target_visit_position,
            "incumbent_updates": incumbent_updates,
            "cycle_walk_total_steps": total_cycle_steps,
        },
        "winner": {
            "coefficients": list(best),
            "expression": _polynomial_expression(best),
            "exact_on_all_rows": best_key[:2] == (0, 0),
        },
        "classification": "known_planted_rediscovery_not_novel",
        "decision": "PASS",
    }


def _archimedes_problem(root: Path) -> dict[str, Any]:
    seed = "Invariant pseudorandom formula benchmark / Archimedes real data / v1"
    config, _path = archimedes.load_config(root)
    columns, rows = archimedes._readings(config)
    candidates = archimedes._candidate_vectors(config["search_policy"], len(columns))
    permutation = PseudorandomOrdinalPermutation(len(candidates), seed)
    training_ids = set(config["search_policy"]["initial_training_object_ids"])
    holdout_ids = set(config["search_policy"]["initial_holdout_object_ids"])
    training = [
        row
        for source, row in zip(config["rows"], rows, strict=True)
        if source["object_id"] in training_ids
    ]
    holdout = [
        row
        for source, row in zip(config["rows"], rows, strict=True)
        if source["object_id"] in holdout_ids
    ]
    target = (1, 1, -1, -1)
    best_key: tuple[Any, ...] | None = None
    best: tuple[int, ...] | None = None
    target_visit_position = None
    incumbent_updates = 0
    total_cycle_steps = 0
    for position in range(len(candidates)):
        ordinal, steps = permutation.at_with_cycle_steps(position)
        total_cycle_steps += steps
        vector = candidates[ordinal]
        score = archimedes._score(training, vector, archimedes._fraction_residual)
        key = (
            Fraction(score["mean_absolute_residual_newton"]),
            Fraction(score["maximum_absolute_residual_newton"]),
            score["l1_norm"],
            score["coefficients"],
        )
        if best_key is None or key < best_key:
            best_key = key
            best = vector
            incumbent_updates += 1
        if vector == target:
            target_visit_position = position
    if best != target or target_visit_position is None:
        raise PseudorandomFormulaBenchmarkError("Archimedes recovery changed")
    training_score = archimedes._score(training, best, archimedes._fraction_residual)
    holdout_score = archimedes._score(holdout, best, archimedes._fraction_residual)
    return {
        "problem_id": "real-data.archimedes-force-balance",
        "source_kind": "real_measured_public_laboratory_readings",
        "source": {
            "publisher": config["source"]["publisher"],
            "config_sha256": canonical_sha256(config),
            "measurement_rows": len(rows),
        },
        "grammar": {
            "formula": "primitive homogeneous integer linear relation",
            "coefficient_range": [-2, 2],
            "maximum_l1_norm": 4,
            "candidate_count": len(candidates),
        },
        "permutation": permutation.descriptor(),
        "search": {
            "candidates_tested": len(candidates),
            "target_visit_position_zero_based": target_visit_position,
            "incumbent_updates": incumbent_updates,
            "cycle_walk_total_steps": total_cycle_steps,
        },
        "winner": {
            "coefficients": list(best),
            "expression": archimedes._expression(columns, best),
            "training_mean_absolute_residual_newton": training_score[
                "mean_absolute_residual_newton"
            ],
            "holdout_mean_absolute_residual_newton": holdout_score[
                "mean_absolute_residual_newton"
            ],
        },
        "limitations": [
            "Only four displayed measurements are available.",
            "The source supplies no complete experimental uncertainty budget.",
            "The previously exposed known concept is Archimedes' principle.",
        ],
        "classification": "known_real_data_rediscovery_not_novel",
        "decision": "PASS_LIMITED_REAL_DATA_REDISCOVERY",
    }


def _nasa_problem(root: Path) -> dict[str, Any]:
    seed = "Invariant pseudorandom formula benchmark / NASA anonymous columns / v1"
    config = nasa.load_config(root, NASA_CONFIG_PATH)
    snapshot_path = root / NASA_SNAPSHOT_PATH
    eligible, _exclusions = nasa.parse_snapshot(snapshot_path.read_bytes(), config)
    training, holdout, split = nasa.split_and_sanitize(eligible, config)
    committed_training = _read_json(root / NASA_TRAINING_PATH)
    if committed_training.get("rows") != training:
        raise PseudorandomFormulaBenchmarkError("NASA anonymous training rows changed")
    vectors = monomial.enumerate_exponents(3, 12)
    permutation = PseudorandomOrdinalPermutation(len(vectors), seed)
    parsed = monomial._parse_rows(training, 3)
    target = (2, -3, 1)
    best_key: tuple[Any, ...] | None = None
    best: Mapping[str, Any] | None = None
    target_visit_position = None
    incumbent_updates = 0
    total_cycle_steps = 0
    for position in range(len(vectors)):
        ordinal, steps = permutation.at_with_cycle_steps(position)
        total_cycle_steps += steps
        vector = vectors[ordinal]
        candidate = monomial._fit_parsed_candidate(parsed, vector)
        key = monomial._selection_key(candidate)
        if best_key is None or key < best_key:
            best_key = key
            best = candidate
            incumbent_updates += 1
        if vector == target:
            target_visit_position = position
    if best is None or tuple(best["exponents"]) != target or target_visit_position is None:
        raise PseudorandomFormulaBenchmarkError("NASA monomial recovery changed")
    holdout_score = monomial.score_frozen_candidate(holdout, best)
    return {
        "problem_id": "real-data.nasa-exoplanet-anonymous-monomial",
        "source_kind": "real_external_catalog_snapshot",
        "source": {
            "external_principal_id": config["source"]["external_principal_id"],
            "snapshot_sha256": _normalized_file_sha256(snapshot_path),
            "training_rows": len(training),
            "holdout_rows": len(holdout),
            "host_disjoint_split": split["host_intersection_count"] == 0,
        },
        "grammar": {
            "formula": "x0^a*x1^b*x2^c = constant",
            "primitive_integer_exponent_bound": 12,
            "candidate_count": len(vectors),
        },
        "permutation": permutation.descriptor(),
        "search": {
            "candidates_tested": len(vectors),
            "target_visit_position_zero_based": target_visit_position,
            "incumbent_updates": incumbent_updates,
            "cycle_walk_total_steps": total_cycle_steps,
        },
        "winner": {
            "exponents": list(best["exponents"]),
            "expression": best["expression"],
            "training_median_absolute_log_residual": best["fit_metrics"][
                "median_absolute_log_residual"
            ],
            "holdout_median_absolute_response_log_error": holdout_score[
                "median_absolute_response_log_error"
            ],
            "holdout_within_1sigma_fraction": holdout_score["within_1sigma_fraction"],
            "holdout_within_2sigma_fraction": holdout_score["within_2sigma_fraction"],
        },
        "limitations": [
            "Catalog parameters may be inferred or mutually dependent measurements.",
            "This target and dataset were previously exposed and are calibration material.",
            "Recovery does not independently confirm the physical law or establish novelty.",
        ],
        "classification": "known_catalog_relation_rediscovery_not_novel",
        "decision": "PASS_KNOWN_REAL_DATA_REDISCOVERY",
    }


def build_benchmark_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    ordinal_prefix = build_prefix_receipt(
        size=TRILLION_GRAMMAR_SIZE,
        seed="Invariant pseudorandom formula benchmark / 100 basis up to 7 terms / v1",
        sample_count=10_000,
    )
    chunk_schedule = PseudorandomChunkSchedule(
        TRILLION_GRAMMAR_SIZE,
        10_000_000,
        "Invariant pseudorandom formula benchmark / trillion GPU chunks / v1",
    )
    chunk_prefix = list(chunk_schedule.iter(stop_position=1000))
    scale_probe_body = {
        "schema_version": "invariant-pseudorandom-trillion-scale-probe-1.0",
        "ordinal_prefix_receipt": ordinal_prefix,
        "gpu_chunk_schedule": {
            "descriptor": chunk_schedule.descriptor(),
            "sampled_chunk_count": len(chunk_prefix),
            "first_chunks": chunk_prefix[:16],
            "sampled_chunks_sha256": canonical_sha256(chunk_prefix),
            "sampled_chunk_ids_unique": len({row["chunk_id"] for row in chunk_prefix})
            == len(chunk_prefix),
        },
        "claims": {
            "formula_candidates_evaluated": 0,
            "trillion_scale_iteration_completed": False,
        },
    }
    scale_probe = {**scale_probe_body, "content_sha256": canonical_sha256(scale_probe_body)}
    problems = [
        _synthetic_polynomial_problem(),
        _archimedes_problem(root),
        _nasa_problem(root),
    ]
    body = {
        "schema_version": SCHEMA,
        "generator_algorithm": ALGORITHM,
        "scale_probe": scale_probe,
        "problems": problems,
        "counts": {
            "problems": len(problems),
            "problems_passed": sum(str(row["decision"]).startswith("PASS") for row in problems),
            "formula_candidates_actually_tested": sum(
                row["search"]["candidates_tested"] for row in problems
            ),
            "trillion_scale_ordinals_sampled_not_formula_tested": ordinal_prefix["sample"][
                "count"
            ],
        },
        "source_bindings": {
            **{
                path: _normalized_file_sha256(root / path)
                for path in IMPLEMENTATION_PATHS
            },
            ARCHIMEDES_CONFIG_PATH: _normalized_file_sha256(root / ARCHIMEDES_CONFIG_PATH),
            NASA_CONFIG_PATH: _normalized_file_sha256(root / NASA_CONFIG_PATH),
            NASA_TRAINING_PATH: _normalized_file_sha256(root / NASA_TRAINING_PATH),
            NASA_SNAPSHOT_PATH: _normalized_file_sha256(root / NASA_SNAPSHOT_PATH),
        },
        "claims": {
            "collision_free_prefix_observed": True,
            "complete_coverage_is_by_permutation_construction": True,
            "trillion_formula_campaign_executed": False,
            "known_results_rediscovered": 3,
            "new_formula_discovered": False,
            "historical_novelty_established": False,
        },
    }
    return {**body, "content_sha256": canonical_sha256(body)}


def validate_benchmark_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    if receipt.get("schema_version") != SCHEMA:
        raise PseudorandomFormulaBenchmarkError("benchmark receipt schema changed")
    body = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if receipt.get("content_sha256") != canonical_sha256(body):
        raise PseudorandomFormulaBenchmarkError("benchmark receipt seal changed")
    if dict(receipt) != build_benchmark_receipt(root):
        raise PseudorandomFormulaBenchmarkError("benchmark receipt replay changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    started = time.perf_counter()
    receipt = build_benchmark_receipt(root)
    validate_benchmark_receipt(receipt, root)
    elapsed = time.perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "content_sha256": receipt["content_sha256"],
                "elapsed_seconds_build_plus_replay": format(elapsed, ".6f"),
                "formula_candidates_actually_tested": receipt["counts"][
                    "formula_candidates_actually_tested"
                ],
                "problems_passed": receipt["counts"]["problems_passed"],
                "trillion_formula_campaign_executed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
