from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler import open_gravity_rg_sings_seven_holdout_velocity_score_v1 as score

CONFIG_PATH = Path("configs/open_gravity_rg_sings_seven_holdout_postscore_robustness_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_sings_seven_holdout_postscore_robustness_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_sings_seven_holdout_postscore_robustness_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-sings-seven-holdout-postscore-robustness-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-postscore-robustness-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-postscore-robustness-receipt-1.0"
_CONFIG_RAW_SHA256 = "f34469f5ba48e59c214f21b9e9989b34041fb29b200dc82450ec12b5ec182821"
_CONFIG_CONTENT_SHA256 = "88da88867dae385aa93a1dfbd24b072856be5ed4734b0f4b448347d0041ad069"
_MODULE_SEMANTIC_SHA256 = "cd2b0f6f8d2e3711b02a3df6fa11d8640f0fd3e546a02a49281ef88d5174d6de"
_TEST_RAW_SHA256 = "0b47be1cdda4285ac8f2925d8dfb90c8895befb2718b371ed8894159d8a81e4c"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')

NEWTON = "NEWTON_3D_DST"
RAR = "RAR_2016_ON_NEWTON_3D"
MOND = "MOND_STANDARD_MU_ON_NEWTON_3D"
RG = "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
STD = "mean_squared_standardized_residual"
FRAC = "mean_squared_fractional_residual"
MEDABS = "median_absolute_standardized_residual"


class PostscoreRobustnessError(RuntimeError):
    """Raised when the sealed post-score robustness analysis fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostscoreRobustnessError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path) -> str:
    raw = path.read_bytes()
    normalized, count = _MODULE_PIN_PATTERN.subn(rb"\g<1>" + b"0" * 64 + rb"\g<2>", raw)
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    candidate = (_ROOT / relative).resolve()
    _require(candidate == _ROOT or _ROOT in candidate.parents, "path escaped repository")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PostscoreRobustnessError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def _package_bindings() -> dict[str, str]:
    return {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_POST_RESPONSE_DESCRIPTIVE_ROBUSTNESS_NO_RETUNING",
        "status changed",
    )
    predecessor = config["predecessor_score"]
    _require(predecessor["objects"] == 7, "object count changed")
    _require(predecessor["scored_cells"] == 24, "cell count changed")
    _require(predecessor["candidate_ids"] == [NEWTON, RAR, MOND, RG], "candidates changed")
    analysis = config["analysis_contract"]
    for key in (
        "parameters_fitted",
        "formulas_changed",
        "row_gates_changed",
        "uncertainty_floors_added",
        "best_cell_selection_events",
        "candidate_pruning_events",
    ):
        _require(analysis[key] == 0, f"forbidden analysis mutation: {key}")
    _require(analysis["retain_every_failure_and_counterexample"] is True, "failures hidden")
    interpretation = config["interpretation_contract"]
    _require(all(value is True for value in interpretation.values()), "interpretation weakened")
    access = config["access_ceiling"]
    _require(access["sealed_score_receipts_opened"] == 1, "score receipt ceiling changed")
    for key, value in access.items():
        if key != "sealed_score_receipts_opened":
            _require(value == 0, f"forbidden access enabled: {key}")
    _require(all(value is False for value in config["claim_boundary"].values()), "claim promoted")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


def _validate_package() -> None:
    if _MODULE_SEMANTIC_SHA256 != "0" * 64:
        _require(
            module_semantic_sha256(_repo_path(MODULE_PATH)) == _MODULE_SEMANTIC_SHA256,
            "module changed",
        )
    if _TEST_RAW_SHA256 != "0" * 64:
        _require(file_sha256(_repo_path(TEST_PATH)) == _TEST_RAW_SHA256, "tests changed")


def load_config(*, verify_package: bool = True) -> dict[str, Any]:
    path = _repo_path(CONFIG_PATH)
    if _CONFIG_RAW_SHA256 != "0" * 64:
        _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package()
    return config


def _load_score_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["predecessor_score"]
    exact_paths = {
        "config_path": score.CONFIG_PATH,
        "module_path": score.MODULE_PATH,
        "test_path": score.TEST_PATH,
        "receipt_path": score.OUTPUT_PATH,
    }
    for key, expected in exact_paths.items():
        _require(binding[key] == expected.as_posix(), f"predecessor {key} changed")
    for key, hash_key in (
        ("config_path", "config_raw_sha256"),
        ("module_path", "module_raw_sha256"),
        ("test_path", "test_raw_sha256"),
        ("receipt_path", "receipt_raw_sha256"),
    ):
        _require(file_sha256(_repo_path(binding[key])) == binding[hash_key], f"{key} bytes changed")
    score_config = score.load_config()
    _require(
        content_sha256(score_config) == binding["config_content_sha256"], "score config changed"
    )
    _require(
        score.module_semantic_sha256(_repo_path(score.MODULE_PATH))
        == binding["module_semantic_sha256"],
        "score module semantics changed",
    )
    receipt = _read_json(_repo_path(binding["receipt_path"]), "score receipt")
    score.validate_receipt(score_config, receipt)
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "score content changed"
    )
    _require(receipt["status"] == binding["receipt_status"], "score status changed")
    _require(receipt["candidate_ids"] == binding["candidate_ids"], "score candidates changed")
    return receipt


def _order(values: Mapping[str, float], candidates: Sequence[str]) -> list[str]:
    return sorted(candidates, key=lambda candidate: (values[candidate], candidate))


def _mean(values: Sequence[float]) -> float:
    _require(len(values) > 0, "empty mean")
    return math.fsum(values) / len(values)


def _primary_cells(score_receipt: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    ids = score_receipt["primary_aggregate"]["primary_cell_ids"]
    by_id = {row["cell_run_id"]: row for row in score_receipt["cell_scores"]}
    _require(len(by_id) == 24, "cell IDs repeated")
    _require(len(ids) == 7 and len(set(ids)) == 7, "primary IDs changed")
    rows = [by_id[cell_id] for cell_id in ids]
    _require(len({row["object_id"] for row in rows}) == 7, "primary objects repeated")
    return rows


def _aggregate(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str], metric: str, *, median: bool
) -> dict[str, Any]:
    values = {
        candidate: [float(row["metrics"][candidate][metric]) for row in rows]
        for candidate in candidates
    }
    aggregate = {
        candidate: float(statistics.median(items) if median else _mean(items))
        for candidate, items in values.items()
    }
    order = _order(aggregate, candidates)
    return {"values": aggregate, "order": order, "winner": order[0]}


def _leave_one_out(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str], metric: str
) -> list[dict[str, Any]]:
    result = []
    for dropped in sorted(row["object_id"] for row in rows):
        kept = [row for row in rows if row["object_id"] != dropped]
        aggregate = _aggregate(kept, candidates, metric, median=False)
        result.append(
            {
                "dropped_object_id": dropped,
                "values": aggregate["values"],
                "order": aggregate["order"],
                "winner": aggregate["winner"],
                "rg_rank": aggregate["order"].index(RG) + 1,
            }
        )
    return result


def _winner_counts(
    rows: Sequence[Mapping[str, Any]], candidates: Sequence[str], metric: str
) -> dict[str, int]:
    counts = Counter(
        min(candidates, key=lambda candidate: (row["metrics"][candidate][metric], candidate))
        for row in rows
    )
    return {candidate: counts[candidate] for candidate in candidates}


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    _require(len(xs) == len(ys) and len(xs) > 0, "correlation inputs changed")
    mean_x = _mean(xs)
    mean_y = _mean(ys)
    sum_xy = math.fsum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
    sum_x2 = math.fsum((x - mean_x) ** 2 for x in xs)
    sum_y2 = math.fsum((y - mean_y) ** 2 for y in ys)
    if sum_x2 == 0.0 or sum_y2 == 0.0:
        return None
    return sum_xy / math.sqrt(sum_x2 * sum_y2)


def _rg_bias(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in sorted(rows, key=lambda item: item["object_id"]):
        accepted = row["accepted_rows"]
        residuals = [float(item["fractional_residual"][RG]) for item in accepted]
        radii = [float(item["radius_source_kpc"]) for item in accepted]
        result.append(
            {
                "object_id": row["object_id"],
                "accepted_rows": len(accepted),
                "mean_signed_fractional_residual": _mean(residuals),
                "median_signed_fractional_residual": float(statistics.median(residuals)),
                "minimum_signed_fractional_residual": min(residuals),
                "maximum_signed_fractional_residual": max(residuals),
                "radius_residual_pearson": _pearson(radii, residuals),
            }
        )
    return result


def _build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    score_receipt = _load_score_receipt(config)
    candidates = list(score_receipt["candidate_ids"])
    all_cells = score_receipt["cell_scores"]
    primary = _primary_cells(score_receipt)

    mean_standardized = _aggregate(primary, candidates, STD, median=False)
    mean_fractional = _aggregate(primary, candidates, FRAC, median=False)
    median_standardized = _aggregate(primary, candidates, STD, median=True)
    median_fractional = _aggregate(primary, candidates, FRAC, median=True)
    mean_median_absolute = _aggregate(primary, candidates, MEDABS, median=False)
    loo_standardized = _leave_one_out(primary, candidates, STD)
    loo_fractional = _leave_one_out(primary, candidates, FRAC)

    rg_all_three = [
        row["cell_run_id"] for row in all_cells if row["rg_beats_all_three_comparators"]
    ]
    rg_all_three_by_object = Counter(
        row["object_id"] for row in all_cells if row["rg_beats_all_three_comparators"]
    )
    holmberg = [row for row in all_cells if row["object_id"] == "UGC04305"]
    non_holmberg = [row for row in all_cells if row["object_id"] != "UGC04305"]
    _require(len(holmberg) == 9 and len(non_holmberg) == 15, "Holmberg partition changed")

    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PARTIAL_OBJECT_SPECIFIC_RG_SIGNAL_NOT_GENERAL_OR_PUBLICATION_READY",
        "package_bindings": _package_bindings(),
        "predecessor_score_receipt_sha256": score_receipt["content_sha256"],
        "candidate_ids": candidates,
        "object_count": 7,
        "cell_count": 24,
        "primary_aggregates": {
            "mean_object_standardized_mse": mean_standardized,
            "mean_object_fractional_mse": mean_fractional,
            "median_object_standardized_mse": median_standardized,
            "median_object_fractional_mse": median_fractional,
            "mean_object_median_absolute_standardized_residual": mean_median_absolute,
        },
        "leave_one_object_out": {
            "mean_object_standardized_mse": loo_standardized,
            "mean_object_fractional_mse": loo_fractional,
        },
        "all_cell_winner_counts": {
            "standardized_mse": _winner_counts(all_cells, candidates, STD),
            "fractional_mse": _winner_counts(all_cells, candidates, FRAC),
        },
        "rg_all_three_comparator_wins": {
            "cell_count": len(rg_all_three),
            "cell_ids": rg_all_three,
            "by_object": dict(sorted(rg_all_three_by_object.items())),
            "holmberg_ii_cells": sum(row["rg_beats_all_three_comparators"] for row in holmberg),
            "holmberg_ii_total_cells": len(holmberg),
            "non_holmberg_cells": sum(
                row["rg_beats_all_three_comparators"] for row in non_holmberg
            ),
            "non_holmberg_total_cells": len(non_holmberg),
        },
        "primary_rg_signed_bias": _rg_bias(primary),
        "robustness_findings": {
            "published_error_primary_winner": mean_standardized["winner"],
            "fractional_mean_winner": mean_fractional["winner"],
            "fractional_median_winner": median_fractional["winner"],
            "rg_fractional_leave_one_out_wins": sum(row["winner"] == RG for row in loo_fractional),
            "rg_standardized_leave_one_out_wins": sum(
                row["winner"] == RG for row in loo_standardized
            ),
            "rg_fractional_rank_without_holmberg_ii": next(
                row["rg_rank"] for row in loo_fractional if row["dropped_object_id"] == "UGC04305"
            ),
            "all_rg_all_three_wins_are_holmberg_ii": all(
                cell_id.startswith("UGC04305__") for cell_id in rg_all_three
            ),
            "signal_classification": "FOLLOW_UP_WORTHY_HOLMBERG_II_SPECIFIC_SIGNAL",
            "aggregate_classification": "MEAN_FRACTIONAL_ADVANTAGE_FAILS_MEDIAN_AND_HOLMBERG_REMOVAL",
        },
        "required_next_tests": [
            "response-source-matched Holmberg II rotation curve with the same frozen predictions",
            "preregistered larger low-mass dwarf cohort with fixed conversion and geometry cells",
            "independent full-3D source reconstruction rather than model-lifted vertical structure",
            "same fixed refracted-gravity law against MOND/RAR/Newton with declared systematic-error sensitivity",
        ],
        "access_accounting": dict(config["access_ceiling"]),
        "claim_boundary": {
            **config["claim_boundary"],
            "postscore_robustness_completed": True,
            "holmberg_ii_specific_signal_observed": True,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(receipt["schema"] == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["package_id"] == config["package_id"], "receipt package changed")
    _require(receipt["package_bindings"] == _package_bindings(), "receipt package seal changed")
    _require(
        receipt["status"] == "PARTIAL_OBJECT_SPECIFIC_RG_SIGNAL_NOT_GENERAL_OR_PUBLICATION_READY",
        "status changed",
    )
    _require(receipt["object_count"] == 7 and receipt["cell_count"] == 24, "coverage changed")
    _require(receipt["access_accounting"] == config["access_ceiling"], "access changed")
    claims = receipt["claim_boundary"]
    _require(claims["postscore_robustness_completed"] is True, "analysis completion lost")
    _require(claims["holmberg_ii_specific_signal_observed"] is True, "observed signal lost")
    for key in (
        "refracted_gravity_generalizes",
        "refracted_gravity_preferred",
        "holmberg_ii_inclination_resolved",
        "stellar_conversion_resolved",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overpromoted: {key}")
    copy = dict(receipt)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "receipt content hash changed")
    _require(dict(receipt) == _build_receipt(config), "receipt deterministic rebuild differs")


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _build_receipt(config)
    validate_receipt(config, receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent output differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(build_receipt(config)))


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    stored = _read_json(path, "receipt")
    validate_receipt(config, stored)
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        print(check_receipt())
    else:
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "primary_winner": receipt["robustness_findings"][
                        "published_error_primary_winner"
                    ],
                    "fractional_mean_winner": receipt["robustness_findings"][
                        "fractional_mean_winner"
                    ],
                    "fractional_median_winner": receipt["robustness_findings"][
                        "fractional_median_winner"
                    ],
                    "rg_all_three_wins": receipt["rg_all_three_comparator_wins"]["cell_count"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
