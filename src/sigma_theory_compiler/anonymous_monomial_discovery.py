"""Generic bounded discovery of multiplicative relations in anonymous positive columns.

The module deliberately knows nothing about any scientific domain.  It receives rows of
positive ``x0 .. xN`` values, evaluates primitive integer exponent vectors, and freezes the
lowest-spread relation under one of three declared search orders.  Candidate generation never
reads a target formula or holdout rows.

The resulting object is a search receipt, not a law: a small log residual can result from
dependent columns, derived catalog values, selection effects, or an inadequate uncertainty
model.  Scientific interpretation belongs to an external evaluator.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Mapping, Sequence
from math import gcd
from typing import Any

from .sigma_core import canonical_sha256

SCHEMA_VERSION = "invariant-anonymous-monomial-search-1.0"
STRATEGIES = ("new_occam", "old_pairwise", "uniform_random")


class AnonymousMonomialError(ValueError):
    """Raised when a declared search cannot be replayed exactly."""


def _primitive(vector: Sequence[int]) -> bool:
    divisor = 0
    for value in vector:
        divisor = gcd(divisor, abs(value))
    return divisor == 1


def _canonical_orientation(vector: Sequence[int]) -> bool:
    return next((value for value in vector if value), 0) > 0


def enumerate_exponents(arity: int, bound: int) -> list[tuple[int, ...]]:
    """Return the complete primitive search space in deterministic Occam order.

    ``x0`` is the declared response and must participate.  At least one other column must
    participate.  Opposite vectors describe the same invariant, so only one orientation is
    retained.
    """

    if not 2 <= arity <= 8:
        raise AnonymousMonomialError("arity must be between 2 and 8")
    if not 1 <= bound <= 32:
        raise AnonymousMonomialError("exponent bound must be between 1 and 32")

    vectors: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...]) -> None:
        if len(prefix) == arity:
            if prefix[0] == 0 or sum(value != 0 for value in prefix) < 2:
                return
            if not _primitive(prefix) or not _canonical_orientation(prefix):
                return
            vectors.append(prefix)
            return
        for value in range(-bound, bound + 1):
            visit((*prefix, value))

    visit(())
    vectors.sort(
        key=lambda vector: (
            sum(abs(value) for value in vector),
            sum(value != 0 for value in vector),
            tuple(abs(value) for value in vector),
            vector,
        )
    )
    return vectors


def _parse_rows(rows: Sequence[Mapping[str, Any]], arity: int) -> list[dict[str, Any]]:
    if len(rows) < max(10, arity * 3):
        raise AnonymousMonomialError("too few rows for the declared search")
    parsed: list[dict[str, Any]] = []
    labels: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != {"label", "uncertainties", "values"}:
            raise AnonymousMonomialError(f"row {index} schema changed")
        label = row["label"]
        values = row["values"]
        uncertainties = row["uncertainties"]
        if not isinstance(label, str) or not label or label in labels:
            raise AnonymousMonomialError("row labels must be unique non-empty text")
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != arity
            or not isinstance(uncertainties, Sequence)
            or isinstance(uncertainties, (str, bytes))
            or len(uncertainties) != arity
        ):
            raise AnonymousMonomialError(f"row {label} arity changed")
        try:
            numeric_values = [float(value) for value in values]
            numeric_uncertainties = [float(value) for value in uncertainties]
        except (TypeError, ValueError) as error:
            raise AnonymousMonomialError(f"row {label} is not numeric") from error
        if any(not math.isfinite(value) or value <= 0 for value in numeric_values):
            raise AnonymousMonomialError(f"row {label} has a non-positive value")
        if any(not math.isfinite(value) or value < 0 for value in numeric_uncertainties):
            raise AnonymousMonomialError(f"row {label} has an invalid uncertainty")
        labels.add(label)
        parsed.append(
            {
                "label": label,
                "logs": [math.log(value) for value in numeric_values],
                "relative_uncertainties": [
                    uncertainty / value
                    for value, uncertainty in zip(numeric_values, numeric_uncertainties, strict=True)
                ],
            }
        )
    return parsed


def _nearest_rank(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise AnonymousMonomialError("cannot summarize an empty metric")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def _text(value: float) -> str:
    if not math.isfinite(value):
        raise AnonymousMonomialError("non-finite search metric")
    return format(value, ".17g")


def relation_expression(exponents: Sequence[int]) -> str:
    factors: list[str] = []
    for index, exponent in enumerate(exponents):
        if exponent == 0:
            continue
        name = f"x{index}"
        factors.append(name if exponent == 1 else f"{name}^{exponent}")
    return "*".join(factors) + " = constant"


def fit_candidate(
    rows: Sequence[Mapping[str, Any]], exponents: Sequence[int]
) -> dict[str, Any]:
    """Fit the invariant centre using training rows only."""

    vector = tuple(int(value) for value in exponents)
    parsed = _parse_rows(rows, len(vector))
    return _fit_parsed_candidate(parsed, vector)


def _fit_parsed_candidate(
    parsed: Sequence[Mapping[str, Any]], vector: tuple[int, ...]
) -> dict[str, Any]:
    if vector[0] == 0 or not _primitive(vector) or not _canonical_orientation(vector):
        raise AnonymousMonomialError("candidate is not a canonical response-bearing primitive")
    logs = [
        sum(exponent * value for exponent, value in zip(vector, row["logs"], strict=True))
        for row in parsed
    ]
    centre = statistics.median(logs)
    residuals = [abs(value - centre) for value in logs]
    return {
        "complexity_l1": sum(abs(value) for value in vector),
        "exponents": list(vector),
        "expression": relation_expression(vector),
        "fit_log_constant": _text(centre),
        "fit_metrics": {
            "maximum_absolute_log_residual": _text(max(residuals)),
            "median_absolute_log_residual": _text(statistics.median(residuals)),
            "p90_absolute_log_residual": _text(_nearest_rank(residuals, 0.9)),
            "rows": len(parsed),
        },
        "support_size": sum(value != 0 for value in vector),
    }


def score_frozen_candidate(
    rows: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Score a frozen training candidate without refitting its constant."""

    expected = {
        "complexity_l1",
        "exponents",
        "expression",
        "fit_log_constant",
        "fit_metrics",
        "support_size",
    }
    if set(candidate) != expected:
        raise AnonymousMonomialError("frozen candidate schema changed")
    vector = tuple(int(value) for value in candidate["exponents"])
    parsed = _parse_rows(rows, len(vector))
    centre = float(candidate["fit_log_constant"])
    residuals: list[float] = []
    standardized: list[float] = []
    response_residuals: list[float] = []
    for row in parsed:
        invariant = sum(
            exponent * value for exponent, value in zip(vector, row["logs"], strict=True)
        )
        residual = abs(invariant - centre)
        sigma = math.sqrt(
            sum(
                (exponent * uncertainty) ** 2
                for exponent, uncertainty in zip(
                    vector, row["relative_uncertainties"], strict=True
                )
            )
        )
        residuals.append(residual)
        standardized.append(residual / max(sigma, 1e-12))
        response_residuals.append(residual / abs(vector[0]))
    return {
        "maximum_absolute_log_residual": _text(max(residuals)),
        "median_absolute_log_residual": _text(statistics.median(residuals)),
        "median_absolute_response_log_error": _text(statistics.median(response_residuals)),
        "median_standardized_residual": _text(statistics.median(standardized)),
        "p90_absolute_log_residual": _text(_nearest_rank(residuals, 0.9)),
        "p90_standardized_residual": _text(_nearest_rank(standardized, 0.9)),
        "rows": len(parsed),
        "within_1sigma_fraction": _text(sum(value <= 1 for value in standardized) / len(parsed)),
        "within_2sigma_fraction": _text(sum(value <= 2 for value in standardized) / len(parsed)),
    }


def _selection_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = candidate["fit_metrics"]
    return (
        float(metrics["median_absolute_log_residual"]),
        float(metrics["p90_absolute_log_residual"]),
        candidate["complexity_l1"],
        candidate["support_size"],
        tuple(candidate["exponents"]),
    )


def discover(
    rows: Sequence[Mapping[str, Any]],
    *,
    arity: int,
    exponent_bound: int,
    candidate_budget: int,
    strategy: str,
    random_seed: int | None = None,
) -> dict[str, Any]:
    """Run one declared candidate order and freeze its best training relation."""

    if strategy not in STRATEGIES:
        raise AnonymousMonomialError("unknown search strategy")
    if not 1 <= candidate_budget <= 100_000:
        raise AnonymousMonomialError("candidate budget is outside the supported range")
    parsed = _parse_rows(rows, arity)
    full_pool = enumerate_exponents(arity, exponent_bound)
    if strategy == "old_pairwise":
        eligible = [vector for vector in full_pool if sum(value != 0 for value in vector) == 2]
        ordered = eligible
    elif strategy == "new_occam":
        eligible = full_pool
        ordered = eligible
    else:
        if not isinstance(random_seed, int):
            raise AnonymousMonomialError("uniform_random requires an integer seed")
        eligible = full_pool
        ordered = list(eligible)
        random.Random(random_seed).shuffle(ordered)
    if len(ordered) < candidate_budget:
        raise AnonymousMonomialError("candidate budget exceeds the eligible search space")
    evaluated_vectors = ordered[:candidate_budget]
    evaluated = [_fit_parsed_candidate(parsed, vector) for vector in evaluated_vectors]
    best = min(evaluated, key=_selection_key)
    ranked = sorted(evaluated, key=_selection_key)
    return {
        "best_candidate": best,
        "candidate_budget": candidate_budget,
        "evaluated_exponent_sha256": canonical_sha256([list(vector) for vector in evaluated_vectors]),
        "full_eligible_count": len(eligible),
        "full_pool_sha256": canonical_sha256([list(vector) for vector in full_pool]),
        "random_seed": random_seed,
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "top_candidates": ranked[:10],
    }


def validate_search(
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    exponent_bound: int,
) -> dict[str, Any]:
    """Replay a search receipt and reject any changed candidate, pool, or ordering."""

    expected = {
        "best_candidate",
        "candidate_budget",
        "evaluated_exponent_sha256",
        "full_eligible_count",
        "full_pool_sha256",
        "random_seed",
        "schema_version",
        "strategy",
        "top_candidates",
    }
    if set(receipt) != expected or receipt.get("schema_version") != SCHEMA_VERSION:
        raise AnonymousMonomialError("search receipt schema changed")
    arity = len(receipt["best_candidate"]["exponents"])
    replayed = discover(
        rows,
        arity=arity,
        exponent_bound=exponent_bound,
        candidate_budget=int(receipt["candidate_budget"]),
        strategy=str(receipt["strategy"]),
        random_seed=receipt["random_seed"],
    )
    if dict(receipt) != replayed:
        raise AnonymousMonomialError("search receipt does not replay")
    return dict(receipt)
