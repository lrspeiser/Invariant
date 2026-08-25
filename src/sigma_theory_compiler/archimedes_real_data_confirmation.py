"""Replay a target-blind Archimedes search against measured university lab data.

This check distinguishes three conclusions that are easy to conflate: selection of
the best bounded relation, approximate empirical support for a known physical law,
and proof or novelty. The first is reproducible here; the latter two are not claimed.
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

import sympy as sp

CONFIG_PATH = "configs/archimedes_real_data_confirmation.json"
RECEIPT_PATH = "runs/math/archimedes-real-data-confirmation/receipt.json"
CONFIG_SCHEMA = "invariant-archimedes-real-data-config-1.0"
RECEIPT_SCHEMA = "invariant-archimedes-real-data-receipt-1.0"
MODULE_PATH = "src/sigma_theory_compiler/archimedes_real_data_confirmation.py"


class RealDataConfirmationError(ValueError):
    """Raised when the frozen real-data confirmation fails closed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise RealDataConfirmationError(f"{label} fields changed")


def _fraction(value: Any, label: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise RealDataConfirmationError(f"{label} is not an exact rational")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as error:
        raise RealDataConfirmationError(f"{label} is not an exact rational") from error


def _fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(
        value,
        {
            "campaign_id",
            "columns",
            "display_resolution",
            "rows",
            "schema_version",
            "search_policy",
            "source",
            "units",
        },
        "real-data config",
    )
    if value["schema_version"] != CONFIG_SCHEMA:
        raise RealDataConfirmationError("real-data config schema changed")
    if value["campaign_id"] != "archimedes-real-data-confirmation-2026-08-25-001":
        raise RealDataConfirmationError("campaign identity changed")
    expected_columns = [
        "empty_container",
        "object_air",
        "object_submerged",
        "container_with_displaced_water",
    ]
    if value["columns"] != expected_columns or value["units"] != "newton":
        raise RealDataConfirmationError("measurement columns or units changed")
    resolution = _fraction(value["display_resolution"], "display resolution")
    if resolution != Fraction(1, 100):
        raise RealDataConfirmationError("display resolution changed")

    source = value["source"]
    _strict(
        source,
        {
            "assets",
            "landing_page_sha256",
            "landing_page_url",
            "provenance_statement",
            "publisher",
            "retrieved_date",
            "transcription_method",
        },
        "source",
    )
    if (
        source["publisher"] != "University of Tennessee, Knoxville Physics 221"
        or source["retrieved_date"] != "2026-08-25"
        or not str(source["landing_page_url"]).startswith("https://labs.phys.utk.edu/")
    ):
        raise RealDataConfirmationError("source identity changed")
    hashes = [source["landing_page_sha256"]]
    assets = source["assets"]
    if not isinstance(assets, list) or len(assets) != 12:
        raise RealDataConfirmationError("source asset manifest changed")
    asset_ids: set[str] = set()
    for asset in assets:
        _strict(asset, {"asset_id", "sha256", "url"}, "source asset")
        if asset["asset_id"] in asset_ids or not str(asset["url"]).startswith(
            "https://labs.phys.utk.edu/"
        ):
            raise RealDataConfirmationError("source asset identity changed")
        asset_ids.add(asset["asset_id"])
        hashes.append(asset["sha256"])
    if any(
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in hashes
    ):
        raise RealDataConfirmationError("source hash is malformed")

    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != 4:
        raise RealDataConfirmationError("exactly four measured objects are required")
    expected_row_fields = {"object_id", *expected_columns}
    object_ids = []
    for row in rows:
        _strict(row, expected_row_fields, "measurement row")
        object_ids.append(row["object_id"])
        for column in expected_columns:
            if _fraction(row[column], f"{row['object_id']}.{column}") <= 0:
                raise RealDataConfirmationError("force readings must be positive")
    if object_ids != ["object_1", "object_2", "object_3", "object_4"]:
        raise RealDataConfirmationError("measurement object identities changed")

    search = value["search_policy"]
    _strict(
        search,
        {
            "coefficient_maximum",
            "coefficient_minimum",
            "expected_relation_supplied_to_search",
            "initial_holdout_object_ids",
            "initial_training_object_ids",
            "maximum_l1_norm",
            "normalization",
            "ranking",
            "relation_family",
        },
        "search policy",
    )
    if search != {
        "relation_family": "primitive homogeneous integer linear relations",
        "coefficient_minimum": -2,
        "coefficient_maximum": 2,
        "maximum_l1_norm": 4,
        "normalization": "gcd_one_first_nonzero_positive",
        "ranking": (
            "mean_absolute_residual_then_max_absolute_residual_then_l1_then_lexicographic"
        ),
        "initial_training_object_ids": ["object_1", "object_2", "object_3"],
        "initial_holdout_object_ids": ["object_4"],
        "expected_relation_supplied_to_search": False,
    }:
        raise RealDataConfirmationError("frozen search policy changed")
    return dict(value)


def load_config(root: Path) -> tuple[dict[str, Any], Path]:
    path = (root / CONFIG_PATH).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise RealDataConfirmationError("real-data config is unavailable")
    return validate_config(json.loads(path.read_text(encoding="utf-8"))), path


def _candidate_vectors(search: Mapping[str, Any], width: int) -> list[tuple[int, ...]]:
    candidates = []
    coefficient_range = range(
        search["coefficient_minimum"], search["coefficient_maximum"] + 1
    )
    for vector in itertools.product(coefficient_range, repeat=width):
        if not any(vector) or sum(abs(value) for value in vector) > search["maximum_l1_norm"]:
            continue
        divisor = 0
        for value in vector:
            divisor = math.gcd(divisor, abs(value))
        if divisor != 1 or next(value for value in vector if value) < 0:
            continue
        candidates.append(vector)
    return candidates


def _fraction_residual(row: Sequence[Fraction], vector: Sequence[int]) -> Fraction:
    return sum(
        (Fraction(coefficient) * value for coefficient, value in zip(vector, row, strict=True)),
        start=Fraction(0),
    )


def _sympy_residual(row: Sequence[Fraction], vector: Sequence[int]) -> Fraction:
    result = sum(
        (
            sp.Integer(coefficient) * sp.Rational(value.numerator, value.denominator)
            for coefficient, value in zip(vector, row, strict=True)
        ),
        start=sp.Integer(0),
    )
    result = sp.cancel(result)
    return Fraction(int(sp.numer(result)), int(sp.denom(result)))


def _expression(columns: Sequence[str], vector: Sequence[int]) -> str:
    terms = []
    for column, coefficient in zip(columns, vector, strict=True):
        if not coefficient:
            continue
        magnitude = "" if abs(coefficient) == 1 else f"{abs(coefficient)}*"
        term = f"{magnitude}{column}"
        if not terms:
            terms.append(term if coefficient > 0 else f"-{term}")
        else:
            terms.append(f" {'+' if coefficient > 0 else '-'} {term}")
    return "".join(terms) + " = 0"


def _score(
    rows: Sequence[Sequence[Fraction]],
    vector: Sequence[int],
    evaluator: Any,
) -> dict[str, Any]:
    residuals = [evaluator(row, vector) for row in rows]
    absolute = [abs(value) for value in residuals]
    return {
        "coefficients": list(vector),
        "residuals_newton": [_fraction_text(value) for value in residuals],
        "mean_absolute_residual_newton": _fraction_text(sum(absolute) / len(absolute)),
        "maximum_absolute_residual_newton": _fraction_text(max(absolute)),
        "l1_norm": sum(abs(value) for value in vector),
    }


def _rank(
    rows: Sequence[Sequence[Fraction]],
    columns: Sequence[str],
    candidates: Sequence[Sequence[int]],
    evaluator: Any,
) -> list[dict[str, Any]]:
    scores = [_score(rows, vector, evaluator) for vector in candidates]
    scores.sort(
        key=lambda item: (
            Fraction(item["mean_absolute_residual_newton"]),
            Fraction(item["maximum_absolute_residual_newton"]),
            item["l1_norm"],
            item["coefficients"],
        )
    )
    for item in scores:
        item["expression"] = _expression(columns, item["coefficients"])
    return scores


def _discover(
    rows: Sequence[Sequence[Fraction]],
    columns: Sequence[str],
    candidates: Sequence[Sequence[int]],
) -> dict[str, Any]:
    primary = _rank(rows, columns, candidates, _fraction_residual)
    independent = _rank(rows, columns, candidates, _sympy_residual)
    if primary != independent:
        raise RealDataConfirmationError("Fraction and SymPy rankings disagree")
    return {
        "candidate_count": len(candidates),
        "winner": primary[0],
        "runner_up": primary[1],
        "top_five": primary[:5],
        "independent_exact_evaluators_agree": True,
    }


def _readings(config: Mapping[str, Any]) -> tuple[list[str], list[list[Fraction]]]:
    columns = config["columns"]
    rows = [[_fraction(row[column], column) for column in columns] for row in config["rows"]]
    return columns, rows


def _physical_comparison(config: Mapping[str, Any], rows: Sequence[Sequence[Fraction]]) -> list[dict]:
    comparisons = []
    for source_row, values in zip(config["rows"], rows, strict=True):
        empty, air, submerged, displaced = values
        buoyant = air - submerged
        displaced_water = displaced - empty
        absolute_difference = abs(buoyant - displaced_water)
        symmetric_percent = 200 * absolute_difference / (buoyant + displaced_water)
        comparisons.append(
            {
                "object_id": source_row["object_id"],
                "buoyant_force_newton": _fraction_text(buoyant),
                "displaced_water_weight_newton": _fraction_text(displaced_water),
                "absolute_difference_newton": _fraction_text(absolute_difference),
                "symmetric_percent_difference": _fraction_text(symmetric_percent),
            }
        )
    return comparisons


def _leave_one_out(
    config: Mapping[str, Any],
    rows: Sequence[Sequence[Fraction]],
    columns: Sequence[str],
    candidates: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    folds = []
    for holdout_index, source_row in enumerate(config["rows"]):
        training = [row for index, row in enumerate(rows) if index != holdout_index]
        discovery = _discover(training, columns, candidates)
        holdout = _score([rows[holdout_index]], discovery["winner"]["coefficients"], _fraction_residual)
        holdout["expression"] = _expression(columns, holdout["coefficients"])
        folds.append(
            {
                "holdout_object_id": source_row["object_id"],
                "training_winner": discovery["winner"],
                "training_runner_up": discovery["runner_up"],
                "holdout_score": holdout,
            }
        )
    return folds


def _pairing_specificity(rows: Sequence[Sequence[Fraction]], winner: Sequence[int]) -> dict:
    observed_mae = Fraction(
        _score(rows, winner, _fraction_residual)["mean_absolute_residual_newton"]
    )
    permuted_maes = []
    for permutation in itertools.permutations(range(len(rows))):
        permuted = [list(row) for row in rows]
        for row_index, displaced_index in enumerate(permutation):
            permuted[row_index][3] = rows[displaced_index][3]
        permuted_maes.append(
            Fraction(_score(permuted, winner, _fraction_residual)["mean_absolute_residual_newton"])
        )
    no_worse_count = sum(value <= observed_mae for value in permuted_maes)
    return {
        "permutation_count": len(permuted_maes),
        "unique_mae_count": len(set(permuted_maes)),
        "observed_mae_newton": _fraction_text(observed_mae),
        "best_permuted_mae_newton": _fraction_text(min(permuted_maes)),
        "permutations_no_worse_than_observed": no_worse_count,
        "status": "PAIRING_SPECIFICITY_UNIDENTIFIABLE",
        "reason": (
            "all 24 displaced-water permutations have the same aggregate absolute residual; "
            "the four low-variation rows cannot validate object-level pairing"
        ),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    config, config_path = load_config(root)
    columns, rows = _readings(config)
    candidates = _candidate_vectors(config["search_policy"], len(columns))
    if len(candidates) != 112:
        raise RealDataConfirmationError("bounded candidate count changed")

    training_ids = set(config["search_policy"]["initial_training_object_ids"])
    holdout_ids = set(config["search_policy"]["initial_holdout_object_ids"])
    training = [
        values
        for source_row, values in zip(config["rows"], rows, strict=True)
        if source_row["object_id"] in training_ids
    ]
    holdout = [
        values
        for source_row, values in zip(config["rows"], rows, strict=True)
        if source_row["object_id"] in holdout_ids
    ]
    discovery = _discover(training, columns, candidates)
    holdout_score = _score(
        holdout, discovery["winner"]["coefficients"], _fraction_residual
    )
    holdout_score["expression"] = _expression(columns, holdout_score["coefficients"])
    folds = _leave_one_out(config, rows, columns, candidates)
    expected_relation = [1, 1, -1, -1]
    all_folds_match = all(
        fold["training_winner"]["coefficients"] == expected_relation for fold in folds
    )

    half_resolution = _fraction(config["display_resolution"], "display resolution") / 2
    quantization_residual_bound = sum(abs(value) for value in expected_relation) * half_resolution
    physical = _physical_comparison(config, rows)
    rows_within_quantization = sum(
        Fraction(item["absolute_difference_newton"]) <= quantization_residual_bound
        for item in physical
    )
    pairing = _pairing_specificity(rows, expected_relation)
    selected_matches = discovery["winner"]["coefficients"] == expected_relation
    if not selected_matches or not all_folds_match:
        raise RealDataConfirmationError("Archimedes-equivalent relation was not stable")

    payload = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "config": {"path": CONFIG_PATH, "normalized_sha256": _file_sha(config_path)},
            "module": {"path": MODULE_PATH, "normalized_sha256": _file_sha(Path(__file__))},
            "external_source": config["source"],
            "measurement_content_sha256": _sha(config["rows"]),
        },
        "measurement_summary": {
            "row_count": len(rows),
            "units": config["units"],
            "display_resolution_newton": config["display_resolution"],
            "readings": config["rows"],
            "physical_comparisons": physical,
        },
        "initial_split": {
            "training_object_ids": sorted(training_ids),
            "holdout_object_ids": sorted(holdout_ids),
            "discovery": discovery,
            "holdout_score": holdout_score,
        },
        "leave_one_object_out": {
            "fold_count": len(folds),
            "same_archimedes_equivalent_winner_every_fold": all_folds_match,
            "folds": folds,
        },
        "measurement_resolution_control": {
            "per_reading_half_resolution_newton": _fraction_text(half_resolution),
            "winner_residual_bound_from_display_quantization_newton": _fraction_text(
                quantization_residual_bound
            ),
            "rows_compatible_with_zero_at_display_quantization_only": rows_within_quantization,
            "row_count": len(rows),
            "strict_equality_confirmed": rows_within_quantization == len(rows),
            "reason": (
                "the source provides displayed readings but no experimental uncertainty budget"
            ),
        },
        "pairing_specificity_control": pairing,
        "interpretation": {
            "selected_relation": discovery["winner"]["expression"],
            "equivalent_physical_statement": (
                "object_air - object_submerged = "
                "container_with_displaced_water - empty_container"
            ),
            "known_concept": "Archimedes' principle",
            "novel_theory": False,
            "independent_in_what_sense": (
                "the bounded search was not supplied the expected coefficient vector"
            ),
            "not_independent_in_what_sense": (
                "the source is an Archimedes lab and the post-search interpretation uses known physics"
            ),
            "claim_boundary": (
                "real measured-data rediscovery of the best simple relation; not a proof, causal "
                "identification, or historical novelty claim"
            ),
        },
        "result": {
            "candidate_relations_tested_per_search": len(candidates),
            "cross_validation_searches": len(folds),
            "selected_relation_matches_archimedes_equivalence": selected_matches,
            "same_winner_in_every_leave_one_out_fold": all_folds_match,
            "strict_equality_confirmed": False,
            "pairing_specificity_identified": False,
            "status": "PASS_REAL_MEASURED_BEST_RELATION_LIMITED_CONFIRMATION",
        },
    }
    return {**payload, "content_sha256": _sha(payload)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> dict[str, Any]:
    expected = build_receipt(root)
    if receipt != expected:
        raise RealDataConfirmationError("real-data receipt does not replay exactly")
    return {
        "content_sha256": receipt["content_sha256"],
        "selected_relation": receipt["interpretation"]["selected_relation"],
        "status": receipt["result"]["status"],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "validate"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--root", type=Path, default=Path("."))
        subparser.add_argument("--receipt", type=Path, default=Path(RECEIPT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
    if args.command == "run":
        receipt = build_receipt(root)
        _write_json(receipt_path, receipt)
        print(json.dumps(validate_receipt(receipt, root), sort_keys=True))
        return 0
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(json.dumps(validate_receipt(receipt, root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
