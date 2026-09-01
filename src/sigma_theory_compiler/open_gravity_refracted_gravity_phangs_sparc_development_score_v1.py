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
    open_gravity_matched_acceleration_cross_scale_development_score_v1 as responses,
)
from sigma_theory_compiler import (
    open_gravity_matched_acceleration_cross_scale_predictions_v1 as controls,
)

CONFIG_PATH = Path("configs/open_gravity_refracted_gravity_phangs_sparc_development_score_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_refracted_gravity_phangs_sparc_development_score_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_refracted_gravity_phangs_sparc_development_score_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-refracted-gravity-phangs-sparc-development-score-v1/receipt.json"
)

_ROOT = Path(__file__).resolve().parents[2]
_SCHEMA = "invariant-open-gravity-refracted-gravity-phangs-sparc-development-score-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-refracted-gravity-phangs-sparc-development-score-receipt-1.0"
)
_CONFIG_RAW_SHA256 = "87e8ec42380e66302c045330fd6cfe401e01e81cb4ff3ca7169d5dbe03647ceb"
_CONFIG_CONTENT_SHA256 = "40988871305ea0d9c269f10bc264a1a1b968b7d868665565895081f8d8a3d84d"
_MODULE_SEMANTIC_SHA256 = "e9d5ba52812ece318b7b3e2c0dbf71ddbaa3b9e05263f7f4834b86ed6671e348"
_TEST_RAW_SHA256 = "32c7de2e2247bc032b4abc82841aa7abea2d2ce826b49a06f5f786eff570fab8"
_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256 = ")[0-9a-f]{64}("\r?\n)')


class DevelopmentScoreError(RuntimeError):
    """Raised when the fixed development score contract fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentScoreError(message)


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
        raise DevelopmentScoreError(f"invalid {label}") from exc
    _require(type(value) is dict, f"{label} must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_DEVELOPMENT_ONLY_RESPONSE_SCORE_NO_TUNING",
        "status changed",
    )
    method = config["method_rule"]
    _require(method["real_source_required"] is True, "real-source gate removed")
    _require(
        method["primary_paper_or_analytic_benchmark_required"] is True,
        "benchmark gate removed",
    )
    _require(method["response_parameter_source_radius_or_threshold_tuning"] is False, "tuning")
    _require(method["published_prior_corners_may_compete"] is False, "prior fishing enabled")
    candidates = config["candidate_contract"]
    _require(
        candidates["candidate_ids"]
        == [
            "NEWTON_3D_DST",
            "RAR_2016_ON_NEWTON_3D",
            "MOND_STANDARD_MU_ON_NEWTON_3D",
            "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG",
        ],
        "candidate inventory changed",
    )
    _require(candidates["per_object_fitted_parameters"] == 0, "per-object fitting enabled")
    _require(candidates["global_fitted_parameters"] == 0, "global fitting enabled")
    radius = config["radius_gate"]
    _require(radius["velocity_values_used_by_gate"] is False, "response radius selection enabled")
    _require(radius["fine_vs_convergence_maximum_relative_difference"] == 0.05, "gate changed")
    _require(radius["require_all_four_candidate_predictions_converged"] is True, "row parity lost")
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
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output changed")


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
    _require(file_sha256(path) == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "config")
    validate_config(config)
    if verify_package:
        _validate_package_files()
    return config


def validate_predecessors(config: Mapping[str, Any]) -> dict[str, Any]:
    expected_roles = [
        "SCORING_RESOLUTION_FIELDS",
        "EXISTING_DEVELOPMENT_RESPONSE_CONTRACT",
        "PUBLISHED_CONTROL_FORMULAS",
        "SOURCE_SYSTEMATIC_ENVELOPE",
    ]
    _require(
        [row["role"] for row in config["predecessor_bindings"]] == expected_roles,
        "predecessor roles changed",
    )
    receipts_by_role: dict[str, dict[str, Any]] = {}
    for binding in config["predecessor_bindings"]:
        _require(binding["commit"] is None, "unverified commit introduced")
        _require(binding["promotion_authority"] is False, "uncommitted authority overclaimed")
        for artifact in binding["artifacts"]:
            path = _repo_path(artifact["path"])
            _require(path.is_file(), "predecessor artifact missing")
            _require(file_sha256(path) == artifact["sha256"], "predecessor artifact changed")
        if "receipt_content_sha256" in binding:
            artifact = next(
                row for row in binding["artifacts"] if row["path"].endswith("receipt.json")
            )
            receipt = _read_json(_repo_path(artifact["path"]), "predecessor receipt")
            _require(
                receipt["content_sha256"] == binding["receipt_content_sha256"],
                "predecessor receipt content changed",
            )
            receipts_by_role[binding["role"]] = receipt
    field = receipts_by_role["SCORING_RESOLUTION_FIELDS"]
    _require(field["all_object_gates_pass"] is True, "scoring-resolution field gate failed")
    _require(field["scientific_boundary"]["response_files_opened"] == 0, "field response leak")
    envelope = receipts_by_role["SOURCE_SYSTEMATIC_ENVELOPE"]
    _require(envelope["registered_source_parameter_pairs"] == 2025, "source envelope changed")
    response_config = responses.load_config()
    control_config = controls.load_config()
    return {
        "field_receipt": field,
        "source_envelope_receipt": envelope,
        "response_config": response_config,
        "control_config": control_config,
    }


def _candidate_acceleration(
    candidate_id: str,
    newton: np.ndarray,
    refracted: np.ndarray,
    *,
    a0_m_s2: float,
) -> np.ndarray:
    _require(np.all(np.isfinite(newton) & (newton >= 0.0)), "invalid Newton prediction")
    _require(np.all(np.isfinite(refracted) & (refracted >= 0.0)), "invalid RG prediction")
    if candidate_id == "NEWTON_3D_DST":
        return newton.copy()
    if candidate_id == "REFRACTED_GRAVITY_DISKMASS_MEDIAN_3D_PCG":
        return refracted.copy()
    if candidate_id == "RAR_2016_ON_NEWTON_3D":
        return np.asarray([controls.rar_2016(float(value), a0_m_s2) for value in newton])
    if candidate_id == "MOND_STANDARD_MU_ON_NEWTON_3D":
        return np.asarray([controls.mond_standard(float(value), a0_m_s2) for value in newton])
    raise DevelopmentScoreError("unknown candidate")


def _object_field_row(field_receipt: Mapping[str, Any], object_id: str) -> Mapping[str, Any]:
    row = next((row for row in field_receipt["objects"] if row["object_id"] == object_id), None)
    _require(row is not None, "field object missing")
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
        "profile radii differ",
    )
    newton = np.asarray([float(row["radial_acceleration_m_s2"]) for row in newton_rows])
    refracted = np.asarray([float(row["radial_acceleration_m_s2"]) for row in rg_rows])
    return radius, newton, refracted


def _common_eligible_rows(
    config: Mapping[str, Any],
    object_row: Mapping[str, Any],
    response_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    fine_radius, fine_newton, fine_rg = _profiles(object_row, "fine")
    coarse_radius, coarse_newton, coarse_rg = _profiles(object_row, "convergence")
    _require(np.array_equal(fine_radius, coarse_radius), "fine and convergence radii differ")
    candidate_ids = config["candidate_contract"]["candidate_ids"]
    a0 = float(config["candidate_contract"]["a0_m_s2"])
    fine_candidates = {
        candidate_id: _candidate_acceleration(candidate_id, fine_newton, fine_rg, a0_m_s2=a0)
        for candidate_id in candidate_ids
    }
    coarse_candidates = {
        candidate_id: _candidate_acceleration(candidate_id, coarse_newton, coarse_rg, a0_m_s2=a0)
        for candidate_id in candidate_ids
    }
    gate = config["radius_gate"]
    threshold = float(gate["fine_vs_convergence_maximum_relative_difference"])
    eligible_rows: list[Mapping[str, Any]] = []
    predictions: dict[str, list[float]] = {candidate_id: [] for candidate_id in candidate_ids}
    exclusions: list[dict[str, Any]] = []
    for row in response_rows:
        radius = float(row["radius_kpc"])
        in_range = float(gate["minimum_kpc"]) <= radius <= float(gate["maximum_kpc"])
        enough_cells = radius / float(gate["fine_spacing_kpc"]) >= float(gate["minimum_fine_cells"])
        relative: dict[str, float] = {}
        values: dict[str, float] = {}
        if in_range:
            for candidate_id in candidate_ids:
                fine_value = float(np.interp(radius, fine_radius, fine_candidates[candidate_id]))
                coarse_value = float(
                    np.interp(radius, coarse_radius, coarse_candidates[candidate_id])
                )
                values[candidate_id] = fine_value
                relative[candidate_id] = abs(fine_value - coarse_value) / max(
                    abs(fine_value), abs(coarse_value), 1.0e-30
                )
        all_converged = in_range and all(value <= threshold for value in relative.values())
        finite_positive = in_range and all(
            math.isfinite(value) and value > 0.0 for value in values.values()
        )
        eligible = in_range and enough_cells and all_converged and finite_positive
        if eligible:
            eligible_rows.append(row)
            for candidate_id in candidate_ids:
                predictions[candidate_id].append(values[candidate_id])
        else:
            exclusions.append(
                {
                    "radius_kpc": radius,
                    "in_range": in_range,
                    "enough_fine_cells": enough_cells,
                    "all_candidates_converged": all_converged,
                    "finite_positive": finite_positive,
                    "maximum_relative_difference": max(relative.values(), default=None),
                }
            )
    _require(
        len(eligible_rows) >= int(config["scoring_contract"]["minimum_rows_per_object"]),
        "insufficient common eligible rows",
    )
    return (
        eligible_rows,
        {key: np.asarray(value, dtype=np.float64) for key, value in predictions.items()},
        {
            "rows_available": len(response_rows),
            "rows_scored_common": len(eligible_rows),
            "rows_excluded": len(exclusions),
            "excluded_rows": exclusions,
            "eligibility_used_velocity_values": False,
        },
    )


def _score_object(
    config: Mapping[str, Any],
    field_receipt: Mapping[str, Any],
    object_id: str,
    response_rows: Sequence[Mapping[str, Any]],
    *,
    asymmetric: bool,
) -> dict[str, Any]:
    object_row = _object_field_row(field_receipt, object_id)
    used, gravity, eligibility = _common_eligible_rows(config, object_row, response_rows)
    radius = np.asarray([float(row["radius_kpc"]) for row in used])
    observed = np.asarray([float(row["velocity_km_s"]) for row in used])
    results: dict[str, Any] = {}
    for candidate_id, acceleration in gravity.items():
        predicted = np.sqrt(acceleration * radius * 3.085677581491367e19) / 1000.0
        if asymmetric:
            upper = np.asarray([float(row["upper_error_km_s"]) for row in used])
            lower = np.asarray([float(row["lower_error_km_s"]) for row in used])
            sigma = np.where(predicted >= observed, upper, lower)
        else:
            sigma = np.asarray([float(row["error_km_s"]) for row in used])
        _require(bool(np.all(np.isfinite(sigma) & (sigma > 0.0))), "invalid uncertainty")
        residual = (predicted - observed) / sigma
        squared = residual * residual
        worst = int(np.argmax(squared))
        results[candidate_id] = {
            "loss": float(np.mean(squared)),
            "rows_scored": int(squared.size),
            "worst_radius_kpc": float(radius[worst]),
            "worst_standardized_residual": float(residual[worst]),
            "worst_standardized_square": float(squared[worst]),
        }
    return {"object_id": object_id, "eligibility": eligibility, "candidates": results}


def _aggregate(object_rows: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    rows = [
        {"object_id": row["object_id"], **row["candidates"][candidate_id]} for row in object_rows
    ]
    worst = max(rows, key=lambda row: (float(row["loss"]), str(row["object_id"])))
    return {
        "loss": float(np.mean([float(row["loss"]) for row in rows])),
        "object_count": len(rows),
        "rows_scored": sum(int(row["rows_scored"]) for row in rows),
        "worst_object": worst["object_id"],
        "worst_object_loss": worst["loss"],
        "objects": rows,
    }


def _fractional_improvement(candidate: float, control: float) -> float:
    _require(math.isfinite(candidate) and candidate >= 0.0, "invalid candidate loss")
    _require(math.isfinite(control) and control > 0.0, "invalid control loss")
    return (control - candidate) / control


def _adjudicate(
    config: Mapping[str, Any],
    phangs: Sequence[Mapping[str, Any]],
    sparc: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    candidate_id = config["candidate_contract"]["primary_test_candidate"]
    comparator_ids = [
        row for row in config["candidate_contract"]["candidate_ids"] if row != candidate_id
    ]
    phangs_aggregate = {
        row: _aggregate(phangs, row) for row in config["candidate_contract"]["candidate_ids"]
    }
    sparc_aggregate = {
        row: _aggregate(sparc, row) for row in config["candidate_contract"]["candidate_ids"]
    }
    best_phangs = min(comparator_ids, key=lambda row: phangs_aggregate[row]["loss"])
    best_sparc = min(comparator_ids, key=lambda row: sparc_aggregate[row]["loss"])
    phangs_improvement = _fractional_improvement(
        float(phangs_aggregate[candidate_id]["loss"]),
        float(phangs_aggregate[best_phangs]["loss"]),
    )
    sparc_improvement = _fractional_improvement(
        float(sparc_aggregate[candidate_id]["loss"]),
        float(sparc_aggregate[best_sparc]["loss"]),
    )
    support_rows = []
    for object_row in phangs:
        rg_loss = float(object_row["candidates"][candidate_id]["loss"])
        best_id = min(comparator_ids, key=lambda row: object_row["candidates"][row]["loss"])
        best_loss = float(object_row["candidates"][best_id]["loss"])
        support_rows.append(
            {
                "object_id": object_row["object_id"],
                "best_comparator_id": best_id,
                "best_comparator_loss": best_loss,
                "refracted_gravity_loss": rg_loss,
                "fractional_improvement": _fractional_improvement(rg_loss, best_loss),
                "supports_refracted_gravity": rg_loss < best_loss,
            }
        )
    threshold = float(config["adjudication"]["minimum_meaningful_fractional_improvement"])
    support = sum(row["supports_refracted_gravity"] for row in support_rows)
    checks = {
        "phangs_improvement_above_threshold": phangs_improvement > threshold,
        "phangs_object_support": support
        >= int(config["adjudication"]["phangs_minimum_object_support_against_best_comparator"]),
        "sparc_same_improvement_direction": sparc_improvement > 0.0,
    }
    signal = all(checks.values())
    return {
        "candidate_id": candidate_id,
        "phangs": {
            "candidate_aggregates": phangs_aggregate,
            "best_comparator_id": best_phangs,
            "fractional_improvement_over_best_comparator": phangs_improvement,
            "object_support_count": support,
            "object_comparisons": support_rows,
        },
        "sparc": {
            "candidate_aggregates": sparc_aggregate,
            "best_comparator_id": best_sparc,
            "fractional_improvement_over_best_comparator": sparc_improvement,
        },
        "checks": checks,
        "development_signal": signal,
        "source_systematic_robustness_established": False,
        "global_discovery_p_value_claimed": False,
        "maximum_claim": config["adjudication"]["maximum_claim"],
    }


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    evidence = validate_predecessors(config)
    field = evidence["field_receipt"]
    response_config = evidence["response_config"]
    phangs_responses, phangs_access = responses._load_phangs_responses(response_config)
    sparc_responses, sparc_access = responses._load_sparc_responses(response_config)
    phangs_scores = [
        _score_object(config, field, object_id, phangs_responses[object_id], asymmetric=True)
        for object_id in config["access_scope"]["phangs_selected_objects"]
    ]
    sparc_scores = [
        _score_object(
            config,
            field,
            object_id,
            responses._sparc_rows(sparc_responses[object_id]),
            asymmetric=False,
        )
        for object_id in config["access_scope"]["sparc_selected_objects"]
    ]
    adjudication = _adjudicate(config, phangs_scores, sparc_scores)
    envelope = evidence["source_envelope_receipt"]
    source_summary = {
        "fixed_median_source_envelopes": envelope["fixed_median_source_envelopes"],
        "retained_counterexamples": envelope["retained_counterexamples"],
        "maximum_source_acceleration_ratio_by_object": {
            object_id: max(
                float(row["maximum_to_minimum_ratio"])
                for row in envelope["fixed_median_source_envelopes"]
                if row["object_id"] == object_id
            )
            for object_id in config["access_scope"]["phangs_selected_objects"]
        },
        "loss_robustness_recomputed": False,
        "reason": "The 225-cell source screen supplies fixed-radius acceleration envelopes, not high-resolution response losses; it is retained as a required uncertainty warning rather than converted into a false score uncertainty.",
    }
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": (
            "DEVELOPMENT_ONLY_FIXED_RG_SIGNAL_RETAINED_FOR_SOURCE_PROPAGATION"
            if adjudication["development_signal"]
            else "NO_DEVELOPMENT_SIGNAL_FOR_FIXED_PUBLISHED_RG_CONTROL"
        ),
        "config_raw_sha256": file_sha256(_repo_path(CONFIG_PATH)),
        "config_content_sha256": content_sha256(config),
        "module_semantic_sha256": module_semantic_sha256(_repo_path(MODULE_PATH)),
        "test_raw_sha256": file_sha256(_repo_path(TEST_PATH)),
        "field_receipt_content_sha256": field["content_sha256"],
        "source_envelope_receipt_content_sha256": envelope["content_sha256"],
        "primary_data_and_method_anchors": config["primary_data_and_method_anchors"],
        "candidate_contract": config["candidate_contract"],
        "radius_gate": config["radius_gate"],
        "phangs_object_scores": phangs_scores,
        "sparc_object_scores": sparc_scores,
        "adjudication": adjudication,
        "source_systematic_context": source_summary,
        "access_accounting": {
            "phangs": phangs_access,
            "sparc": sparc_access,
            "phangs_selected_rows_scored": sum(
                row["eligibility"]["rows_scored_common"] for row in phangs_scores
            ),
            "sparc_selected_rows_scored": sum(
                row["eligibility"]["rows_scored_common"] for row in sparc_scores
            ),
            "object_candidate_scores_computed": (len(phangs_scores) + len(sparc_scores))
            * len(config["candidate_contract"]["candidate_ids"]),
            "confirmation_rows_opened": 0,
            "independent_rows_opened": 0,
            "group_rows_opened": 0,
            "lensing_rows_opened": 0,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "tuning_calls": 0,
        },
        "claim_boundary": {
            "development_fit_tested": True,
            "fixed_published_rg_control_tested": True,
            "source_systematic_score_robustness_established": False,
            "independent_confirmation": False,
            "cluster_fit_tested": False,
            "lensing_closure_established": False,
            "relativistic_completion_established": False,
            "novelty_established": False,
            "publication_ready": False,
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = content_sha256({**receipt, "content_sha256": ""})
    return receipt


def validate_receipt_payload(config: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
    expected = build_receipt(config)
    _require(dict(payload) == expected, "receipt does not match deterministic rebuild")


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
    receipt = build_receipt(config)
    return _atomic_no_clobber(_output_path(), canonical_bytes(receipt) + b"\n")


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
                    "development_signal": receipt["adjudication"]["development_signal"],
                    "phangs_fractional_improvement": receipt["adjudication"]["phangs"][
                        "fractional_improvement_over_best_comparator"
                    ],
                    "sparc_fractional_improvement": receipt["adjudication"]["sparc"][
                        "fractional_improvement_over_best_comparator"
                    ],
                    "tuning_calls": receipt["access_accounting"]["tuning_calls"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
