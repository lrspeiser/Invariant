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
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as controls,
)
from sigma_theory_compiler import sparc_full_sample

CONFIG_PATH = Path("configs/open_gravity_rg_five_galaxy_fixed_response_score_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_rg_five_galaxy_fixed_response_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_rg_five_galaxy_fixed_response_score_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-rg-five-galaxy-fixed-response-score-v1/receipt.json")

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-rg-five-galaxy-fixed-response-score-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-rg-five-galaxy-fixed-response-score-receipt-1.0"
_CONFIG_RAW_SHA256 = "9ad25148022b702e660e7633dd58725a6e0626a7513d4919f47fb9a9e6e5600d"
_CONFIG_CONTENT_SHA256 = "1cbedb33560e6d32b99b7e412a7828c3d0cc0b3c7a502bc98a3d1097ebc42e5e"
_MODULE_SEMANTIC_SHA256 = "41a368755579a28fcb0d15407b09999f6875efc19c82a2cb2a8ac8569deac760"
_TEST_RAW_SHA256 = "d4ce5ab6afd300fd6e01542b344e93a29827d4ab2a3c47460ea836b53fc07744"
_MODULE_PIN_PATTERN = re.compile(rb"(?m)^_MODULE_SEMANTIC_SHA256 = .+$")
_KPC_M = 3.085677581491367e19


class FixedResponseScoreError(RuntimeError):
    """Raised when the fixed five-galaxy score fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FixedResponseScoreError(message)


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
    normalized, count = _MODULE_PIN_PATTERN.subn(
        b'_MODULE_SEMANTIC_SHA256 = "' + b"0" * 64 + b'"', raw
    )
    _require(count == 1, "module semantic pin pattern changed")
    return hashlib.sha256(normalized).hexdigest()


def _repo_path(relative: Path | str) -> Path:
    candidate = (_ROOT / relative).resolve()
    _require(candidate == _ROOT or _ROOT in candidate.parents, "path escaped repository")
    return candidate


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixedResponseScoreError(f"invalid {label}") from exc
    _require(type(payload) is dict, f"{label} must be an object")
    return payload


def validate_config(config: Mapping[str, Any]) -> None:
    if _CONFIG_CONTENT_SHA256 != "0" * 64:
        _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_DEVELOPMENT_ONLY_FIXED_RESPONSE_SCORE",
        "status changed",
    )
    admission = config["builder_admission_rule"]
    for key in (
        "real_public_source_data_required",
        "primary_measurement_or_data_release_paper_required",
        "independent_analytic_manufactured_or_reference_benchmark_required",
        "all_three_requirements_satisfied_by_bound_predecessor",
    ):
        _require(admission[key] is True, f"builder admission weakened: {key}")
    _require(admission["spherical_or_1d_data_validate_general_3d"] is False, "3D overclaim")
    response = config["response_binding"]
    _require(
        response["selected_objects"] == ["NGC2903", "NGC2976", "NGC3198", "NGC3521", "NGC4214"],
        "selected objects changed",
    )
    _require(response["selection_frozen_before_response_access"] is True, "selection leak")
    _require(response["container_galaxies_opened"] == 175, "SPARC object count changed")
    _require(response["container_response_rows_opened"] == 3391, "SPARC row count changed")
    _require(response["selected_rows_available"] == 159, "selected row count changed")
    candidate = config["candidate_contract"]
    _require(
        candidate["candidate_ids"]
        == [
            "NEWTON_3D_DST",
            "RAR_2016_ON_NEWTON_3D",
            "MOND_STANDARD_MU_ON_NEWTON_3D",
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
        ],
        "candidate inventory changed",
    )
    _require(candidate["per_object_fitted_parameters"] == 0, "object fitting enabled")
    _require(candidate["global_fitted_parameters"] == 0, "global fitting enabled")
    _require(candidate["retuning_calls"] == 0, "retuning enabled")
    radius = config["radius_gate"]
    _require(radius["velocity_values_used_by_gate"] is False, "response-selected radii")
    _require(radius["fine_vs_convergence_maximum_relative_difference"] == 0.05, "gate changed")
    _require(radius["minimum_fine_cells"] == 3.0, "cell gate changed")
    scoring = config["scoring_contract"]
    _require(scoring["candidate_row_parity_required"] is True, "row parity removed")
    _require(scoring["family_elimination_from_this_run"] is False, "family elimination enabled")
    _require(scoring["retain_every_failure_and_counterexample"] is True, "failures discarded")
    access = config["access_scope"]
    _require(access["development_only"] is True, "development boundary removed")
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        _require(access[key] == 0, f"forbidden access enabled: {key}")
    _require(config["claim_boundary"]["family_eliminated"] is False, "family overclaim")
    _require(config["claim_boundary"]["publication_ready"] is False, "publication overclaim")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _validate_package_files() -> None:
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
        _validate_package_files()
    return config


def _verify_bound_file(binding: Mapping[str, Any], role: str) -> Path:
    path = _repo_path(binding[f"{role}_path"])
    _require(path.is_file(), f"bound {role} missing")
    _require(file_sha256(path) == binding[f"{role}_raw_sha256"], f"bound {role} changed")
    return path


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, Any]:
    predecessor = config["predecessor_binding"]
    for role in ("config", "module", "test", "receipt"):
        _verify_bound_file(predecessor, role)
    field_receipt = _read_json(_repo_path(predecessor["receipt_path"]), "field receipt")
    _require(
        field_receipt["content_sha256"] == predecessor["receipt_content_sha256"],
        "field receipt content changed",
    )
    _require(field_receipt["all_object_gates_pass"] is True, "field gate failed")
    _require(
        field_receipt["decision"] == "READY_FOR_FIXED_HELD_SPARC_RESPONSE_SCORE",
        "field decision changed",
    )
    _require(
        field_receipt["scientific_boundary"]["response_files_opened"] == 0, "field response leak"
    )
    _require(field_receipt["claim_boundary"]["observational_fit_tested"] is False, "response leak")
    _require(
        field_receipt["claim_boundary"]["paper_and_real_source_anchored"] is True,
        "source-paper admission lost",
    )
    response = config["response_binding"]
    for role in ("dataset", "module", "test"):
        _verify_bound_file(response, role)
    control = config["control_binding"]
    for role in ("config", "module", "test"):
        _verify_bound_file(control, role)
    return field_receipt


def _load_selected_sparc(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = config["response_binding"]
    dataset = _verify_bound_file(binding, "dataset")
    _verify_bound_file(binding, "module")
    _verify_bound_file(binding, "test")
    galaxies, provenance = sparc_full_sample.load_full_sample(dataset)
    _require(len(galaxies) == binding["container_galaxies_opened"], "SPARC galaxies changed")
    row_count = sum(row.count for row in galaxies)
    _require(row_count == binding["container_response_rows_opened"], "SPARC rows changed")
    wanted = set(binding["selected_objects"])
    selected = {row.name: row for row in galaxies if row.name in wanted}
    _require(set(selected) == wanted, "selected SPARC objects changed")
    selected_rows = sum(row.count for row in selected.values())
    _require(selected_rows == binding["selected_rows_available"], "selected rows changed")
    return selected, {
        "container_galaxies_opened": len(galaxies),
        "container_response_rows_opened": row_count,
        "selected_objects": sorted(selected),
        "selected_rows_available": selected_rows,
        "dataset_sha256": provenance["dataset_sha256"],
    }


def _candidate_acceleration(
    candidate_id: str, newton: np.ndarray, refracted: np.ndarray, a0: float
) -> np.ndarray:
    _require(bool(np.all(np.isfinite(newton) & (newton >= 0.0))), "invalid Newton field")
    _require(bool(np.all(np.isfinite(refracted) & (refracted >= 0.0))), "invalid RG field")
    if candidate_id == "NEWTON_3D_DST":
        return newton.copy()
    if candidate_id == "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG":
        return refracted.copy()
    if candidate_id == "RAR_2016_ON_NEWTON_3D":
        return np.asarray([controls.rar_2016(float(value), a0) for value in newton])
    if candidate_id == "MOND_STANDARD_MU_ON_NEWTON_3D":
        return np.asarray([controls.mond_standard(float(value), a0) for value in newton])
    raise FixedResponseScoreError("unknown candidate")


def _object_field(field_receipt: Mapping[str, Any], object_id: str) -> Mapping[str, Any]:
    row = next((row for row in field_receipt["objects"] if row["object_id"] == object_id), None)
    _require(row is not None, "field object missing")
    _require(row["all_object_gates_pass"] is True, "object field gate failed")
    return row


def _profiles(
    object_row: Mapping[str, Any], grid_key: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = object_row[grid_key]
    newton_rows = grid["profiles"]["NEWTON_3D_DST"]
    rg_rows = grid["profiles"]["REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG"]
    radius = np.asarray([float(row["radius_kpc"]) for row in newton_rows])
    _require(
        radius.tolist() == [float(row["radius_kpc"]) for row in rg_rows],
        "profile radii changed",
    )
    newton = np.asarray([float(row["radial_acceleration_m_s2"]) for row in newton_rows])
    refracted = np.asarray([float(row["radial_acceleration_m_s2"]) for row in rg_rows])
    return radius, newton, refracted


def _radius_prediction(
    config: Mapping[str, Any], object_row: Mapping[str, Any], radius_kpc: float
) -> tuple[bool, dict[str, float], dict[str, Any]]:
    fine_radius, fine_newton, fine_rg = _profiles(object_row, "fine")
    coarse_radius, coarse_newton, coarse_rg = _profiles(object_row, "convergence")
    _require(np.array_equal(fine_radius, coarse_radius), "grid radii differ")
    candidate_ids = config["candidate_contract"]["candidate_ids"]
    a0 = float(config["candidate_contract"]["a0_m_s2"])
    fine = {key: _candidate_acceleration(key, fine_newton, fine_rg, a0) for key in candidate_ids}
    coarse = {
        key: _candidate_acceleration(key, coarse_newton, coarse_rg, a0) for key in candidate_ids
    }
    gate = config["radius_gate"]
    in_range = float(gate["minimum_kpc"]) <= radius_kpc <= float(gate["maximum_kpc"])
    enough_cells = radius_kpc / float(gate["fine_spacing_kpc"]) >= float(gate["minimum_fine_cells"])
    values: dict[str, float] = {}
    relative: dict[str, float] = {}
    if in_range:
        for key in candidate_ids:
            fine_value = float(np.interp(radius_kpc, fine_radius, fine[key]))
            coarse_value = float(np.interp(radius_kpc, coarse_radius, coarse[key]))
            values[key] = fine_value
            relative[key] = abs(fine_value - coarse_value) / max(
                abs(fine_value), abs(coarse_value), 1.0e-30
            )
    positive_finite = in_range and all(
        math.isfinite(value) and value > 0.0 for value in values.values()
    )
    converged = in_range and all(
        value <= float(gate["fine_vs_convergence_maximum_relative_difference"])
        for value in relative.values()
    )
    eligible = in_range and enough_cells and positive_finite and converged
    return (
        eligible,
        values,
        {
            "radius_kpc": radius_kpc,
            "in_range": in_range,
            "enough_fine_cells": enough_cells,
            "positive_finite": positive_finite,
            "all_candidates_converged": converged,
            "maximum_relative_difference": max(relative.values(), default=None),
        },
    )


def _response_rows(galaxy: Any) -> list[dict[str, float]]:
    return [
        {
            "radius_kpc": float(radius),
            "velocity_km_s": float(velocity),
            "error_km_s": float(error),
        }
        for radius, velocity, error in zip(galaxy.radius, galaxy.v_obs, galaxy.e_v_obs, strict=True)
    ]


def _score_object(
    config: Mapping[str, Any],
    field_receipt: Mapping[str, Any],
    object_id: str,
    response_rows: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    object_row = _object_field(field_receipt, object_id)
    candidate_ids = config["candidate_contract"]["candidate_ids"]
    used: list[Mapping[str, float]] = []
    predictions: dict[str, list[float]] = {key: [] for key in candidate_ids}
    exclusions: list[dict[str, Any]] = []
    for row in response_rows:
        radius = float(row["radius_kpc"])
        eligible, values, evidence = _radius_prediction(config, object_row, radius)
        if eligible:
            used.append(row)
            for key in candidate_ids:
                predictions[key].append(values[key])
        else:
            exclusions.append(evidence)
    _require(
        len(used) >= int(config["scoring_contract"]["minimum_rows_per_object"]),
        "insufficient eligible rows",
    )
    radius = np.asarray([float(row["radius_kpc"]) for row in used])
    observed = np.asarray([float(row["velocity_km_s"]) for row in used])
    sigma = np.asarray([float(row["error_km_s"]) for row in used])
    _require(bool(np.all(np.isfinite(sigma) & (sigma > 0.0))), "invalid uncertainties")
    scores: dict[str, Any] = {}
    for key in candidate_ids:
        acceleration = np.asarray(predictions[key])
        predicted = np.sqrt(acceleration * radius * _KPC_M) / 1000.0
        residual = (predicted - observed) / sigma
        squared = residual * residual
        worst = int(np.argmax(squared))
        scores[key] = {
            "loss": float(np.mean(squared)),
            "rows_scored": int(squared.size),
            "worst_radius_kpc": float(radius[worst]),
            "worst_standardized_residual": float(residual[worst]),
            "worst_standardized_square": float(squared[worst]),
        }
    return {
        "object_id": object_id,
        "eligibility": {
            "rows_available": len(response_rows),
            "rows_scored_common": len(used),
            "rows_excluded": len(exclusions),
            "excluded_rows": exclusions,
            "eligibility_used_velocity_values": False,
        },
        "candidates": scores,
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    per_object = [
        {"object_id": row["object_id"], **row["candidates"][candidate_id]} for row in rows
    ]
    worst = max(per_object, key=lambda row: (float(row["loss"]), str(row["object_id"])))
    return {
        "loss": float(np.mean([float(row["loss"]) for row in per_object])),
        "object_count": len(per_object),
        "rows_scored": sum(int(row["rows_scored"]) for row in per_object),
        "worst_object": worst["object_id"],
        "worst_object_loss": worst["loss"],
        "objects": per_object,
    }


def _fractional_improvement(candidate: float, control: float) -> float:
    _require(math.isfinite(candidate) and candidate >= 0.0, "invalid candidate loss")
    _require(math.isfinite(control) and control > 0.0, "invalid control loss")
    return (control - candidate) / control


def _adjudicate(config: Mapping[str, Any], scores: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    candidate_ids = config["candidate_contract"]["candidate_ids"]
    primary = config["candidate_contract"]["primary_candidate_id"]
    comparators = [key for key in candidate_ids if key != primary]
    aggregates = {key: _aggregate(scores, key) for key in candidate_ids}
    best = min(comparators, key=lambda key: aggregates[key]["loss"])
    improvement = _fractional_improvement(
        float(aggregates[primary]["loss"]), float(aggregates[best]["loss"])
    )
    object_rows = []
    for row in scores:
        best_object = min(comparators, key=lambda key: row["candidates"][key]["loss"])
        primary_loss = float(row["candidates"][primary]["loss"])
        best_loss = float(row["candidates"][best_object]["loss"])
        object_rows.append(
            {
                "object_id": row["object_id"],
                "best_comparator_id": best_object,
                "best_comparator_loss": best_loss,
                "refracted_gravity_loss": primary_loss,
                "fractional_improvement": _fractional_improvement(primary_loss, best_loss),
                "supports_refracted_gravity": primary_loss < best_loss,
            }
        )
    support = sum(bool(row["supports_refracted_gravity"]) for row in object_rows)
    material = improvement > float(
        config["scoring_contract"]["minimum_material_fractional_improvement"]
    )
    if material:
        classification = "MATERIAL_AGGREGATE_IMPROVEMENT"
    elif improvement > 0.0:
        classification = "DIRECTIONAL_AGGREGATE_IMPROVEMENT"
    elif support > 0:
        classification = "MIXED_LOCAL_SUPPORT_WITH_AGGREGATE_COUNTEREVIDENCE"
    else:
        classification = "NO_IMPROVEMENT_IN_THIS_FIXED_CELL"
    return {
        "primary_candidate_id": primary,
        "candidate_aggregates": aggregates,
        "best_comparator_id": best,
        "fractional_improvement_over_best_comparator": improvement,
        "object_support_count": support,
        "object_counterexample_count": len(object_rows) - support,
        "object_comparisons": object_rows,
        "classification": classification,
        "material_improvement": material,
        "family_eliminated": False,
        "reason_family_not_eliminated": "This is one fixed published RG parameter cell on five response-blind source-selected development galaxies; all failures remain evidence for source-systematic and architecture follow-up.",
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    field_receipt = validate_predecessors(config)
    selected, access = _load_selected_sparc(config)
    scores = [
        _score_object(
            config,
            field_receipt,
            object_id,
            _response_rows(selected[object_id]),
        )
        for object_id in config["response_binding"]["selected_objects"]
    ]
    adjudication = _adjudicate(config, scores)
    status_by_class = {
        "MATERIAL_AGGREGATE_IMPROVEMENT": "DEVELOPMENT_FIXED_RG_MATERIAL_SIGNAL",
        "DIRECTIONAL_AGGREGATE_IMPROVEMENT": "DEVELOPMENT_FIXED_RG_DIRECTIONAL_SIGNAL",
        "MIXED_LOCAL_SUPPORT_WITH_AGGREGATE_COUNTEREVIDENCE": "DEVELOPMENT_FIXED_RG_MIXED_RESULT_RETAINED",
        "NO_IMPROVEMENT_IN_THIS_FIXED_CELL": "DEVELOPMENT_FIXED_RG_NO_IMPROVEMENT_RETAINED",
    }
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": status_by_class[adjudication["classification"]],
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "field_receipt_content_sha256": field_receipt["content_sha256"],
        "builder_admission_rule": config["builder_admission_rule"],
        "primary_data_and_method_anchors": config["primary_data_and_method_anchors"],
        "candidate_contract": config["candidate_contract"],
        "radius_gate": config["radius_gate"],
        "object_scores": scores,
        "adjudication": adjudication,
        "access_accounting": {
            "sparc": access,
            "selected_rows_scored_common": sum(
                row["eligibility"]["rows_scored_common"] for row in scores
            ),
            "selected_rows_excluded": sum(row["eligibility"]["rows_excluded"] for row in scores),
            "object_candidate_scores_computed": len(scores)
            * len(config["candidate_contract"]["candidate_ids"]),
            "response_container_opens": 1,
            "confirmation_rows_opened": 0,
            "independent_rows_opened": 0,
            "group_rows_opened": 0,
            "lensing_rows_opened": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(config), "receipt differs from deterministic rebuild")


def _output_path() -> Path:
    path = _repo_path(OUTPUT_PATH)
    _require(path == (_ROOT / OUTPUT_PATH).resolve(), "output path changed")
    return path


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "refusing to overwrite nonidentical receipt")
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
            _require(path.read_bytes() == payload, "concurrent nonidentical receipt")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_receipt() -> str:
    config = load_config()
    return _atomic_no_clobber(_output_path(), canonical_bytes(build_receipt(config)) + b"\n")


def validate_receipt() -> None:
    config = load_config()
    path = _output_path()
    _require(path.is_file(), "receipt missing")
    validate_receipt_payload(config, _read_json(path, "receipt"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"), nargs="?", default="check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt(load_config())
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "classification": receipt["adjudication"]["classification"],
                    "fractional_improvement": receipt["adjudication"][
                        "fractional_improvement_over_best_comparator"
                    ],
                    "object_support_count": receipt["adjudication"]["object_support_count"],
                    "rows_scored": receipt["access_accounting"]["selected_rows_scored_common"],
                    "family_eliminated": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
