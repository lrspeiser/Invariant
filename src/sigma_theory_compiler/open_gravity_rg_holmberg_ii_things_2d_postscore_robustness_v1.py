"""Audit Holmberg II score robustness without reopening response arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_rg_holmberg_ii_things_2d_postscore_robustness_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_holmberg_ii_things_2d_postscore_robustness_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_holmberg_ii_things_2d_postscore_robustness_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-holmberg-ii-things-2d-postscore-robustness-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-postscore-robustness-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-holmberg-ii-things-2d-postscore-robustness-receipt-1.0"
_CANDIDATES = (
    "NEWTON_3D_DST",
    "RAR_2016_ON_NEWTON_3D",
    "MOND_STANDARD_MU_ON_NEWTON_3D",
    "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
)
_NEWTON = _CANDIDATES[0]
_RG = _CANDIDATES[3]
_METRICS = (
    "rmse_m_s",
    "mae_m_s",
    "median_absolute_residual_m_s",
    "mom2_scaled_mean_squared_residual",
    "model_specific_best_offset_rmse_m_s",
)
_CONFIG_RAW_SHA256 = "c7436ca96be1600a6803051e1d8c52740bf12c3205db5056a17c8ded58c84db4"
_CONFIG_CONTENT_SHA256 = "8aab355821f0b16fb36e51c012d632f7f98a9cfaba656bf30580b8e46d02f38b"
_MODULE_SEMANTIC_SHA256 = "9bbcc5bfc0a4a5af6d0b999396fbaa8abcd59e80ba44ad9422de98c10e5256b4"
_TEST_RAW_SHA256 = "dee14110483115ed2c75ff15e6e8c5847371d15eae0eb4c06743e475366c5cd8"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class HolmbergRobustnessError(RuntimeError):
    """Raised when a frozen score or post-score contract changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HolmbergRobustnessError(message)


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
    normalized, count = _MODULE_PIN_PATTERN.subn(
        rb"\g<1>" + b"0" * 64 + rb"\g<2>", path.read_bytes()
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    path = (_ROOT / relative).resolve()
    _require(path == _ROOT or _ROOT in path.parents, "path escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HolmbergRobustnessError(f"invalid {label}") from exc
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
        config["status"] == "POST_RESPONSE_DESCRIPTIVE_ROBUSTNESS_NOT_CONFIRMATION",
        "status changed",
    )
    binding = config["score_binding"]
    _require(
        binding["required_decision"]
        == "HOLMBERG_II_RG_SIGNAL_DOES_NOT_REPLICATE_ON_FIXED_PRIMARY_2D_CELL",
        "primary failure changed",
    )
    _require(binding["required_score_cells"] == 18, "score cell count changed")
    analysis = config["analysis_contract"]
    _require(analysis["candidate_ids"] == list(_CANDIDATES), "candidates changed")
    _require(analysis["winner_metrics"] == list(_METRICS), "metrics changed")
    _require(
        analysis["groupings"] == ["inclination_deg", "conversion_cell_id", "resolution"],
        "groups changed",
    )
    for key in (
        "paired_rg_minus_newton_rmse",
        "all_cells_retained",
        "no_response_reopening",
        "no_prediction_recomputation",
        "no_parameter_tuning",
        "no_p_values",
    ):
        _require(analysis[key] is True, f"analysis boundary changed: {key}")
    interpretation = config["interpretation_contract"]
    for key in (
        "primary_failure_must_remain_primary",
        "27_degree_cells_are_predeclared_published_sensitivity_cases",
        "27_degree_cells_are_not_independent_objects",
        "consistent_low_inclination_direction_may_motivate_follow_up",
        "inclination_conditional_preference_is_not_general_gravity_preference",
        "one_source_matched_object_cannot_establish_uniqueness",
    ):
        _require(interpretation[key] is True, f"interpretation changed: {key}")
    boundary = config["scientific_boundary"]
    _require(boundary["score_receipts_opened"] == 1, "receipt accounting changed")
    for key in (
        "response_files_opened",
        "response_pixels_decoded",
        "predictions_recomputed",
        "scores_recomputed",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(boundary[key] == 0, f"forbidden post-score access: {key}")
    claims = config["claim_boundary"]
    _require(claims["primary_rg_replication_passed"] is False, "primary failure erased")
    _require(
        claims["descriptive_inclination_pattern_may_be_reported"] is True, "pattern suppressed"
    )
    for key in (
        "inclination_measured_by_this_analysis",
        "independent_confirmation",
        "refracted_gravity_confirmed",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(claims[key] is False, f"claim overpromoted: {key}")
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


def _load_score(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["score_binding"]
    for artifact in binding["artifacts"]:
        path = _repo_path(artifact["path"])
        _require(path.is_file(), "score artifact missing")
        _require(file_sha256(path) == artifact["sha256"], "score artifact changed")
    receipt = _read_json(_repo_path(binding["artifacts"][-1]["path"]), "score receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "score content changed"
    )
    _require(receipt["decision"] == binding["required_decision"], "score decision changed")
    _require(len(receipt["scores"]) == binding["required_score_cells"], "score rows changed")
    return receipt


def _metric_value(row: Mapping[str, Any], candidate: str, metric: str) -> float:
    if metric == "model_specific_best_offset_rmse_m_s":
        return float(row["models"][candidate]["model_specific_best_offset_diagnostic"]["rmse_m_s"])
    return float(row["models"][candidate][metric])


def _winner(row: Mapping[str, Any], metric: str) -> str:
    return min(
        _CANDIDATES, key=lambda candidate: (_metric_value(row, candidate, metric), candidate)
    )


def _winner_counts(rows: Sequence[Mapping[str, Any]], metric: str) -> dict[str, int]:
    counts = Counter(_winner(row, metric) for row in rows)
    return {candidate: int(counts[candidate]) for candidate in _CANDIDATES}


def _group_rows(
    rows: Sequence[Mapping[str, Any]], key: str
) -> list[tuple[Any, list[Mapping[str, Any]]]]:
    values = sorted({row[key] for row in rows}, key=lambda value: (str(type(value)), value))
    return [(value, [row for row in rows if row[key] == value]) for value in values]


def _inclination_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for inclination, selected in _group_rows(rows, "inclination_deg"):
        differences = [
            _metric_value(row, _RG, "rmse_m_s") - _metric_value(row, _NEWTON, "rmse_m_s")
            for row in selected
        ]
        output.append(
            {
                "inclination_deg": inclination,
                "cell_count": len(selected),
                "rg_rmse_wins": sum(value < 0.0 for value in differences),
                "newton_rmse_wins": sum(value > 0.0 for value in differences),
                "mean_rg_minus_newton_rmse_m_s": sum(differences) / len(differences),
                "minimum_rg_minus_newton_rmse_m_s": min(differences),
                "maximum_rg_minus_newton_rmse_m_s": max(differences),
                "winner_counts_by_metric": {
                    metric: _winner_counts(selected, metric) for metric in _METRICS
                },
            }
        )
    return output


def _crossings(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    conversions = sorted({row["conversion_cell_id"] for row in rows})
    resolutions = sorted({row["resolution"] for row in rows})
    for conversion in conversions:
        for resolution in resolutions:
            selected = sorted(
                [
                    row
                    for row in rows
                    if row["conversion_cell_id"] == conversion and row["resolution"] == resolution
                ],
                key=lambda row: row["inclination_deg"],
            )
            _require(
                [row["inclination_deg"] for row in selected] == [27.0, 38.0, 49.0],
                "inclination cells changed",
            )
            first, second = selected[:2]
            y1 = _metric_value(first, _RG, "rmse_m_s") - _metric_value(first, _NEWTON, "rmse_m_s")
            y2 = _metric_value(second, _RG, "rmse_m_s") - _metric_value(second, _NEWTON, "rmse_m_s")
            _require(y1 * y2 < 0.0, "27-38 crossing absent")
            crossing = 27.0 + (-y1) * 11.0 / (y2 - y1)
            output.append(
                {
                    "conversion_cell_id": conversion,
                    "resolution": resolution,
                    "rg_minus_newton_rmse_at_27_deg_m_s": y1,
                    "rg_minus_newton_rmse_at_38_deg_m_s": y2,
                    "linear_descriptive_crossing_deg": crossing,
                    "is_fitted_inclination": False,
                    "is_confidence_interval": False,
                }
            )
    return output


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    score = _load_score(config)
    rows = score["scores"]
    winner_counts = {metric: _winner_counts(rows, metric) for metric in _METRICS}
    inclination = _inclination_summary(rows)
    crossings = _crossings(rows)
    low = next(row for row in inclination if row["inclination_deg"] == 27.0)
    middle = next(row for row in inclination if row["inclination_deg"] == 38.0)
    high = next(row for row in inclination if row["inclination_deg"] == 49.0)
    all_metric_low_rg = all(counts[_RG] == 6 for counts in low["winner_counts_by_metric"].values())
    all_metric_higher_newton = all(
        counts[_NEWTON] == 6
        for group in (middle, high)
        for counts in group["winner_counts_by_metric"].values()
    )
    rg_beats_mond_rar = {
        candidate: sum(
            _metric_value(row, _RG, "rmse_m_s") < _metric_value(row, candidate, "rmse_m_s")
            for row in rows
        )
        for candidate in _CANDIDATES[1:3]
    }
    pattern = all_metric_low_rg and all_metric_higher_newton
    decision = (
        "FOLLOW_UP_WORTHY_INCLINATION_CONDITIONAL_RG_PATTERN_NOT_GENERAL_PREFERENCE"
        if pattern
        else "NO_STABLE_INCLINATION_CONDITIONAL_RG_PATTERN"
    )
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_POSTSCORE_DESCRIPTIVE_ROBUSTNESS",
        "decision": decision,
        "package_bindings": _package_bindings(),
        "score_receipt_content_sha256": score["content_sha256"],
        "primary_result_retained": {
            "cell_score_id": score["primary_cell"]["cell_score_id"],
            "winner": score["primary_cell"]["winner"],
            "rg_replication_passed": score["primary_cell"]["rg_beats_all_three_comparators"],
            "decision": score["decision"],
        },
        "winner_counts_by_metric_all_18_cells": winner_counts,
        "inclination_groups": inclination,
        "conversion_groups": [
            {
                "conversion_cell_id": value,
                "winner_counts_by_metric": {
                    metric: _winner_counts(selected, metric) for metric in _METRICS
                },
            }
            for value, selected in _group_rows(rows, "conversion_cell_id")
        ],
        "resolution_groups": [
            {
                "resolution": value,
                "winner_counts_by_metric": {
                    metric: _winner_counts(selected, metric) for metric in _METRICS
                },
            }
            for value, selected in _group_rows(rows, "resolution")
        ],
        "rg_rmse_beats_mond_and_rar_counts": rg_beats_mond_rar,
        "linear_descriptive_crossings": crossings,
        "crossing_range_deg": {
            "minimum": min(row["linear_descriptive_crossing_deg"] for row in crossings),
            "maximum": max(row["linear_descriptive_crossing_deg"] for row in crossings),
            "not_an_inclination_fit": True,
        },
        "pattern_checks": {
            "rg_wins_all_six_27_degree_cells_on_all_five_metrics": all_metric_low_rg,
            "newton_wins_all_twelve_38_or_49_degree_cells_on_all_five_metrics": all_metric_higher_newton,
            "same_direction_at_natural_and_robust_resolution": all(
                group["winner_counts_by_metric"]["rmse_m_s"][_RG] == 3
                for group in [
                    {"winner_counts_by_metric": {"rmse_m_s": _winner_counts(selected, "rmse_m_s")}}
                    for _value, selected in _group_rows(
                        [row for row in rows if row["inclination_deg"] == 27.0], "resolution"
                    )
                ]
            ),
            "pattern_is_inclination_conditional_not_general_preference": pattern,
        },
        "next_action": "PREDECLARE_EXTERNAL_INCLINATION_CONSTRAINT_AND_MULTI_GALAXY_2D_REPLICATION",
        "scientific_boundary": config["scientific_boundary"],
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(dict(receipt) == build_receipt(config), "receipt differs from deterministic rebuild")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing nonidentical overwrite")
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
            _require(path.read_bytes() == payload, "concurrent nonidentical output")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    receipt = build_receipt(config)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt) + b"\n")


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    validate_receipt(config, _read_json(path, "receipt"))
    return "VALID"


def status() -> dict[str, Any]:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    if not path.is_file():
        return {"package_id": config["package_id"], "status": "FROZEN_UNRUN"}
    receipt = _read_json(path, "receipt")
    return {
        "package_id": config["package_id"],
        "status": receipt["status"],
        "decision": receipt["decision"],
        "crossing_range_deg": receipt["crossing_range_deg"],
        "pattern_checks": receipt["pattern_checks"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sub.add_parser("check")
    sub.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "build":
        print(write_receipt())
    elif arguments.command == "check":
        print(check_receipt())
    else:
        print(json.dumps(status(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
