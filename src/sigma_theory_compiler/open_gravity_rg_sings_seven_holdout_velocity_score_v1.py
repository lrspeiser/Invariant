from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import (
    open_gravity_rg_sings_seven_holdout_response_blind_predictions_v1 as predictions,
)

CONFIG_PATH = Path("configs/open_gravity_rg_sings_seven_holdout_velocity_score_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_sings_seven_holdout_velocity_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_sings_seven_holdout_velocity_score_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-rg-sings-seven-holdout-velocity-score-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-velocity-score-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-sings-seven-holdout-velocity-score-receipt-1.0"
_CONFIG_RAW_SHA256 = "59587397536d6207aed0db7ae10330a826ef59c3c3efb273a96de5e9f26be96e"
_CONFIG_CONTENT_SHA256 = "97e831de09e4aa4fa76ec4bc444e3663b080fdff8040575357f24f301e696224"
_MODULE_SEMANTIC_SHA256 = "02755f9094ffa8b65b4fd50c40b73a0f7275beb9372cb27886042038285222ba"
_TEST_RAW_SHA256 = "840b5eccc1806ff061fb4231915eaa91c1d1be5dd3c2c4fa336437dd5b9a91e6"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')
_KPC_M = 3.085677581491367e19


class VelocityScoreError(RuntimeError):
    """Raised when the fixed one-pass velocity score fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VelocityScoreError(message)


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
        raise VelocityScoreError(f"invalid {label}") from exc
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
        config["status"] == "FROZEN_ONE_PASS_SEVEN_HOLDOUT_ALL_CELL_VELOCITY_SCORE",
        "status changed",
    )
    prediction = config["prediction_binding"]
    _require(prediction["completed_prediction_cells"] == 24, "prediction count changed")
    _require(prediction["response_values_opened_when_predictions_sealed"] == 0, "blind seal lost")
    _require(
        prediction["candidate_ids"]
        == [
            "NEWTON_3D_DST",
            "RAR_2016_ON_NEWTON_3D",
            "MOND_STANDARD_MU_ON_NEWTON_3D",
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
        ],
        "candidate inventory changed",
    )
    responses = config["response_files"]
    _require(len(responses) == 7, "response inventory changed")
    _require(sum(row["rows"] for row in responses) == 303, "response row count changed")
    _require(len({row["object_id"] for row in responses}) == 7, "response object repeated")
    geometry = config["geometry_transform"]
    _require(geometry["response_based_geometry_selection"] is False, "geometry selection enabled")
    row_gate = config["row_gate"]
    _require(row_gate["uncertainty_floor_km_s"] is None, "uncertainty floor introduced")
    _require(row_gate["same_rows_for_all_four_candidates"] is True, "row parity lost")
    _require(row_gate["retain_every_excluded_row_with_reason"] is True, "exclusions hidden")
    score = config["score_contract"]
    _require(len(score["primary_cells"]) == 7, "primary cells changed")
    _require(score["all_24_cells_scored"] is True, "cell coverage reduced")
    _require(score["best_cell_selection_allowed"] is False, "best-cell selection enabled")
    _require(score["parameters_fitted"] == 0, "parameter fitting enabled")
    _require(score["thresholds_tuned"] == 0, "threshold tuning enabled")
    _require(score["retain_every_failure_and_counterexample"] is True, "failures hidden")
    access = config["access_ceiling"]
    _require(access["response_files_opened_by_execution"] == 7, "file ceiling changed")
    _require(access["response_rows_parsed_by_execution"] == 303, "row ceiling changed")
    for key in (
        "network_calls_by_execution",
        "model_calls",
        "paid_calls",
        "tuning_calls",
        "selection_events",
    ):
        _require(access[key] == 0, f"forbidden access enabled: {key}")
    _require(
        all(value is False for value in config["claim_boundary"].values()),
        "claim promoted before score",
    )
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


def _load_prediction_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["prediction_binding"]
    receipt_path: Path | None = None
    for artifact in binding["artifacts"]:
        path = _repo_path(artifact["path"])
        _require(path.is_file(), "prediction artifact missing")
        _require(file_sha256(path) == artifact["sha256"], "prediction artifact changed")
        if artifact["path"].endswith("receipt.json"):
            receipt_path = path
    _require(receipt_path is not None, "prediction receipt missing")
    receipt = _read_json(receipt_path, "prediction receipt")
    _require(
        receipt["content_sha256"] == binding["receipt_content_sha256"], "prediction receipt changed"
    )
    _require(
        receipt["status"] == "PASS_RESPONSE_BLIND_ALL_CELL_PREDICTIONS_BUILT",
        "predictions incomplete",
    )
    _require(receipt["completed_prediction_cells"] == 24, "prediction cells changed")
    _require(all(value == 0 for value in receipt["response_boundary"].values()), "prediction leak")
    return receipt


def _response_binding(config: Mapping[str, Any], object_id: str) -> Mapping[str, Any]:
    row = next((row for row in config["response_files"] if row["object_id"] == object_id), None)
    _require(row is not None, "response binding missing")
    return row


def _verified_response_text(binding: Mapping[str, Any]) -> str:
    path = _repo_path(binding["path"])
    _require(path.is_file(), "response file missing")
    _require(path.stat().st_size == binding["bytes"], "response bytes changed")
    _require(file_sha256(path) == binding["sha256"], "response hash changed")
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise VelocityScoreError("invalid response file") from exc


def _parse_sparc(binding: Mapping[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in _verified_response_text(binding).splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        tokens = line.split()
        _require(len(tokens) == 8, "SPARC response schema changed")
        try:
            radius, velocity, uncertainty = map(float, tokens[:3])
        except ValueError as exc:
            raise VelocityScoreError("invalid SPARC response value") from exc
        rows.append(
            {
                "radius_response_kpc": radius,
                "velocity_response_km_s": velocity,
                "uncertainty_response_km_s": uncertainty,
            }
        )
    _require(len(rows) == binding["rows"], "SPARC response rows changed")
    return rows


def _parse_little_things(binding: Mapping[str, Any]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in _verified_response_text(binding).splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tokens = line.split()
        if len(tokens) < 8 or tokens[1:3] != ["DDO_50", "Data"]:
            continue
        try:
            r03, v03, radius_scaled, velocity_scaled, uncertainty_scaled = map(float, tokens[3:8])
        except ValueError as exc:
            raise VelocityScoreError("invalid LITTLE THINGS response value") from exc
        rows.append(
            {
                "radius_response_kpc": r03 * radius_scaled,
                "velocity_response_km_s": v03 * velocity_scaled,
                "uncertainty_response_km_s": v03 * uncertainty_scaled,
            }
        )
    _require(len(rows) == binding["rows"], "LITTLE THINGS response rows changed")
    return rows


def load_response_rows(
    config: Mapping[str, Any], object_id: str, source_inclination_deg: float
) -> list[dict[str, float]]:
    binding = _response_binding(config, object_id)
    raw = (
        _parse_sparc(binding)
        if binding["format"] == "SPARC_ROTMOD"
        else _parse_little_things(binding)
    )
    distance_scale = float(binding["source_distance_mpc"]) / float(binding["response_distance_mpc"])
    inclination_scale = math.sin(
        math.radians(float(binding["response_inclination_deg"]))
    ) / math.sin(math.radians(source_inclination_deg))
    _require(math.isfinite(distance_scale) and distance_scale > 0.0, "invalid distance scale")
    _require(
        math.isfinite(inclination_scale) and inclination_scale > 0.0, "invalid inclination scale"
    )
    rows: list[dict[str, float]] = []
    for index, row in enumerate(raw):
        rows.append(
            {
                "response_row_index": index,
                "radius_source_kpc": row["radius_response_kpc"] * distance_scale,
                "velocity_source_km_s": row["velocity_response_km_s"] * inclination_scale,
                "uncertainty_source_km_s": row["uncertainty_response_km_s"] * inclination_scale,
            }
        )
    return rows


def _prediction_cell_index(
    config: Mapping[str, Any], prediction_receipt: Mapping[str, Any]
) -> dict[str, tuple[Mapping[str, Any], dict[str, Any]]]:
    prediction_config = predictions.load_config()
    source_receipt = predictions._load_predecessors(prediction_config)[
        "SEVEN_HOLDOUT_SOURCE_BUILDER"
    ]
    source_by_id = {
        predictions.cell_run_id(row): row for row in predictions._built_source_cells(source_receipt)
    }
    artifacts = {row["cell_run_id"]: row for row in prediction_receipt["cell_artifacts"]}
    _require(set(artifacts) == set(source_by_id), "prediction cell inventory changed")
    result: dict[str, tuple[Mapping[str, Any], dict[str, Any]]] = {}
    for cell_id in sorted(artifacts):
        artifact = artifacts[cell_id]
        path = _repo_path(artifact["relative_path"])
        _require(path.is_file(), "prediction cell missing")
        _require(file_sha256(path) == artifact["file_sha256"], "prediction cell changed")
        payload = _read_json(path, "prediction cell")
        predictions.validate_cell(prediction_config, source_by_id[cell_id], payload)
        _require(
            payload["content_sha256"] == artifact["content_sha256"], "prediction content changed"
        )
        result[cell_id] = (source_by_id[cell_id], payload)
    _require(len(result) == 24, "prediction cell count changed")
    return result


def _interpolated_prediction(
    cell: Mapping[str, Any], radius_kpc: float, candidate_id: str
) -> tuple[float | None, str | None]:
    rows = cell["fine"]["profiles"][candidate_id]
    radii = np.asarray([float(row["radius_kpc"]) for row in rows], dtype=np.float64)
    if not math.isfinite(radius_kpc) or radius_kpc <= 0.0:
        return None, "RADIUS_NOT_POSITIVE_FINITE"
    if radius_kpc < radii[0] or radius_kpc > radii[-1]:
        return None, "RADIUS_OUTSIDE_PREDICTION_GRID"
    right = int(np.searchsorted(radii, radius_kpc, side="left"))
    if right == 0:
        left = right = 0
    elif right == len(radii):
        left = right = len(radii) - 1
    elif radii[right] == radius_kpc:
        left = right
    else:
        left = right - 1
    mask_rows = cell["numerical_mask"]["rows"]
    if not bool(mask_rows[left]["eligible"]) or not bool(mask_rows[right]["eligible"]):
        return None, "NUMERICAL_MASK_FAILED"
    accelerations = np.asarray(
        [float(row["radial_acceleration_m_s2"]) for row in rows], dtype=np.float64
    )
    acceleration = float(np.interp(radius_kpc, radii, accelerations))
    if not math.isfinite(acceleration) or acceleration < 0.0:
        return None, "PREDICTION_INVALID"
    velocity = math.sqrt(acceleration * radius_kpc * _KPC_M) / 1000.0
    return velocity, None


def _score_cell(
    config: Mapping[str, Any], source_cell: Mapping[str, Any], prediction_cell: Mapping[str, Any]
) -> dict[str, Any]:
    cell_id = predictions.cell_run_id(source_cell)
    object_id = str(source_cell["object_id"])
    inclination = float(source_cell["geometry"]["inclination_deg"])
    responses = load_response_rows(config, object_id, inclination)
    candidate_ids = config["prediction_binding"]["candidate_ids"]
    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for response in responses:
        radius = float(response["radius_source_kpc"])
        observed = float(response["velocity_source_km_s"])
        uncertainty = float(response["uncertainty_source_km_s"])
        if not math.isfinite(observed) or observed < 0.0:
            excluded.append({**response, "reason": "VELOCITY_NOT_NONNEGATIVE_FINITE"})
            continue
        if not math.isfinite(uncertainty) or uncertainty <= 0.0:
            excluded.append({**response, "reason": "UNCERTAINTY_NOT_POSITIVE_FINITE"})
            continue
        values: dict[str, float] = {}
        reason: str | None = None
        for candidate_id in candidate_ids:
            value, candidate_reason = _interpolated_prediction(
                prediction_cell, radius, candidate_id
            )
            if candidate_reason is not None:
                reason = candidate_reason
                break
            _require(value is not None, "prediction reason/value mismatch")
            values[candidate_id] = value
        if reason is not None:
            excluded.append({**response, "reason": reason})
            continue
        accepted.append(
            {
                **response,
                "predicted_velocity_km_s": values,
                "standardized_residual": {
                    key: (value - observed) / uncertainty for key, value in values.items()
                },
                "fractional_residual": {
                    key: (value - observed) / max(abs(observed), 1.0)
                    for key, value in values.items()
                },
            }
        )
    metrics: dict[str, dict[str, float]] = {}
    for candidate_id in candidate_ids:
        standardized = [float(row["standardized_residual"][candidate_id]) for row in accepted]
        fractional = [float(row["fractional_residual"][candidate_id]) for row in accepted]
        _require(standardized, "cell has no eligible response rows")
        metrics[candidate_id] = {
            "mean_squared_standardized_residual": float(np.mean(np.square(standardized))),
            "mean_squared_fractional_residual": float(np.mean(np.square(fractional))),
            "median_absolute_standardized_residual": float(np.median(np.abs(standardized))),
        }
    rg_id = "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
    rg_loss = metrics[rg_id]["mean_squared_standardized_residual"]
    comparator_losses = {
        key: value["mean_squared_standardized_residual"]
        for key, value in metrics.items()
        if key != rg_id
    }
    return {
        "cell_run_id": cell_id,
        "object_id": object_id,
        "conversion_cell_id": source_cell["conversion_cell_id"],
        "geometry_variant_id": source_cell["geometry"]["geometry_variant_id"],
        "source_inclination_deg": inclination,
        "response_rows_parsed": len(responses),
        "accepted_row_count": len(accepted),
        "excluded_row_count": len(excluded),
        "accepted_rows": accepted,
        "excluded_rows": excluded,
        "metrics": metrics,
        "rg_beats_newton": rg_loss < comparator_losses["NEWTON_3D_DST"],
        "rg_beats_rar": rg_loss < comparator_losses["RAR_2016_ON_NEWTON_3D"],
        "rg_beats_standard_mond": rg_loss < comparator_losses["MOND_STANDARD_MU_ON_NEWTON_3D"],
        "rg_beats_all_three_comparators": rg_loss < min(comparator_losses.values()),
    }


def _primary_aggregate(
    config: Mapping[str, Any], cells: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    primary_ids = config["score_contract"]["primary_cells"]
    by_id = {row["cell_run_id"]: row for row in cells}
    _require(set(primary_ids).issubset(by_id), "primary cell missing")
    primary = [by_id[cell_id] for cell_id in primary_ids]
    candidate_ids = config["prediction_binding"]["candidate_ids"]
    losses = {
        candidate_id: float(
            np.mean(
                [
                    row["metrics"][candidate_id]["mean_squared_standardized_residual"]
                    for row in primary
                ]
            )
        )
        for candidate_id in candidate_ids
    }
    fractional = {
        candidate_id: float(
            np.mean(
                [
                    row["metrics"][candidate_id]["mean_squared_fractional_residual"]
                    for row in primary
                ]
            )
        )
        for candidate_id in candidate_ids
    }
    ordering = sorted(candidate_ids, key=lambda key: (losses[key], key))
    return {
        "primary_cell_ids": primary_ids,
        "equal_object_mean_squared_standardized_residual": losses,
        "equal_object_mean_squared_fractional_residual": fractional,
        "candidate_order_best_to_worst": ordering,
        "best_candidate_id": ordering[0],
        "rg_object_wins_vs_newton": sum(row["rg_beats_newton"] for row in primary),
        "rg_object_wins_vs_rar": sum(row["rg_beats_rar"] for row in primary),
        "rg_object_wins_vs_standard_mond": sum(row["rg_beats_standard_mond"] for row in primary),
        "rg_object_wins_vs_all_three": sum(
            row["rg_beats_all_three_comparators"] for row in primary
        ),
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    prediction_receipt = _load_prediction_receipt(config)
    cell_index = _prediction_cell_index(config, prediction_receipt)
    scored = [
        _score_cell(config, source_cell, prediction_cell)
        for source_cell, prediction_cell in cell_index.values()
    ]
    _require(len(scored) == 24, "scored cell count changed")
    parsed_by_object: dict[str, int] = {}
    for row in scored:
        parsed_by_object.setdefault(row["object_id"], row["response_rows_parsed"])
        _require(
            parsed_by_object[row["object_id"]] == row["response_rows_parsed"],
            "object response rows changed across cells",
        )
    _require(sum(parsed_by_object.values()) == 303, "response accounting changed")
    aggregate = _primary_aggregate(config, scored)
    rg_id = "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"
    claim = {
        **config["claim_boundary"],
        "seven_holdout_velocity_score_completed": True,
        "refracted_gravity_preferred": aggregate["best_candidate_id"] == rg_id,
    }
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_ONE_PASS_ALL_CELL_SCORE_RETAINING_EVERY_COUNTEREXAMPLE",
        "package_bindings": _package_bindings(),
        "prediction_receipt_sha256": config["prediction_binding"]["receipt_content_sha256"],
        "candidate_ids": config["prediction_binding"]["candidate_ids"],
        "response_files_opened": 7,
        "response_rows_parsed": 303,
        "prediction_cells_opened": 24,
        "scored_cell_count": len(scored),
        "cell_scores": scored,
        "primary_aggregate": aggregate,
        "counterexample_cells": [
            row["cell_run_id"] for row in scored if not row["rg_beats_all_three_comparators"]
        ],
        "access_accounting": {
            "network_calls_by_execution": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
            "selection_events": 0,
        },
        "claim_boundary": claim,
    }
    receipt["content_sha256"] = content_sha256(receipt)
    validate_receipt(config, receipt)
    return receipt


def validate_receipt(config: Mapping[str, Any], receipt: Mapping[str, Any]) -> None:
    _require(receipt["schema"] == _RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["package_id"] == config["package_id"], "receipt package changed")
    _require(receipt["package_bindings"] == _package_bindings(), "receipt package seal changed")
    _require(receipt["response_files_opened"] == 7, "receipt files changed")
    _require(receipt["response_rows_parsed"] == 303, "receipt rows changed")
    _require(receipt["prediction_cells_opened"] == 24, "receipt prediction count changed")
    _require(receipt["scored_cell_count"] == 24, "receipt score count changed")
    _require(len(receipt["cell_scores"]) == 24, "receipt cell ledger changed")
    _require(
        receipt["access_accounting"]
        == {
            "network_calls_by_execution": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
            "selection_events": 0,
        },
        "receipt access changed",
    )
    _require(
        receipt["claim_boundary"]["seven_holdout_velocity_score_completed"] is True,
        "score claim lost",
    )
    for key in (
        "refracted_gravity_generalizes",
        "stellar_conversion_resolved",
        "holmberg_ii_inclination_resolved",
        "general_3d_gravity_validated",
        "unique_theory_established",
        "publication_ready",
    ):
        _require(receipt["claim_boundary"][key] is False, f"claim overpromoted: {key}")
    copy = dict(receipt)
    observed = copy.pop("content_sha256")
    _require(observed == content_sha256(copy), "receipt content hash changed")


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
    receipt = build_receipt(config)
    return _atomic_no_clobber(_repo_path(OUTPUT_PATH), canonical_bytes(receipt))


def check_receipt() -> str:
    config = load_config()
    path = _repo_path(OUTPUT_PATH)
    _require(path.is_file(), "receipt missing")
    stored = _read_json(path, "receipt")
    validate_receipt(config, stored)
    _require(stored == build_receipt(config), "receipt rebuild differs")
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
                    "best_candidate_id": receipt["primary_aggregate"]["best_candidate_id"],
                    "scored_cell_count": receipt["scored_cell_count"],
                    "response_rows_parsed": receipt["response_rows_parsed"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
