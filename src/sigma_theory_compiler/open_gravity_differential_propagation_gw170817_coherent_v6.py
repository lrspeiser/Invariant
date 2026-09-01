"""Strict pre-response optimizer and power repair for coherent GW170817.

V6 reads no strain samples.  It preserves v4/v5, revalidates all frozen source
and runtime bytes, rebuilds target-free injections with symmetric full-common
fits, and separates optimizer recovery from registered statistical power.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import math
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from scipy.optimize import minimize

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v4 as v4,
)
from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v5 as v5,
)

CONFIG_PATH = Path("configs/open_gravity_differential_propagation_gw170817_coherent_v6.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_differential_propagation_gw170817_coherent_v6.py"
)
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_gw170817_coherent_v6.py")
RUN_DIR = Path("runs/gravity/open-gravity-differential-propagation-gw170817-coherent-v6")
PREDICTION_PATH = RUN_DIR / "prediction-receipt.json"
ARTIFACT_DIR = RUN_DIR / "artifacts"

CONFIG_SCHEMA = "invariant-open-gravity-differential-propagation-gw170817-coherent-config-6.0"
PREDICTION_SCHEMA = (
    "invariant-open-gravity-differential-propagation-gw170817-coherent-prediction-receipt-6.0"
)


class CoherentV6Error(RuntimeError):
    """Fail-closed v6 error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoherentV6Error(message)


def _base(root: Path | None = None) -> Path:
    return (root or Path.cwd()).resolve()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(v4._canonical(value) + b"\n")


def _path_values(value: Any, key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            yield from _path_values(child_value, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _path_values(child, key)
    elif isinstance(value, str) and (key == "path" or key.endswith("_path")):
        yield key, value


def _validate_paths(config: Mapping[str, Any]) -> None:
    for key, value in _path_values(config):
        normalized = value.replace("\\", "/")
        parts = PurePosixPath(normalized).parts
        _require(".." not in parts, f"parent alias rejected in {key}: {value}")
        _require(not normalized.startswith("/"), f"absolute path rejected in {key}")
        _require(not re.match(r"^[A-Za-z]:/", normalized), f"drive path rejected in {key}")


def load_config(root: Path | None = None) -> dict[str, Any]:
    config = _read_json(_base(root) / CONFIG_PATH)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "wrong schema")
    _require(
        config.get("analysis_id") == "open-gravity-differential-propagation-gw170817-coherent-v6",
        "wrong analysis id",
    )
    _require(
        config.get("status") == "FROZEN_APPEND_ONLY_PRE_RESPONSE_OPTIMIZER_AND_POWER_REPAIR",
        "v6 not frozen",
    )
    _require(
        config["strict_audit_of_v5"]["label"] == "FAIL_OPTIMIZER_AND_REGISTERED_POWER_GATE",
        "v5 failure mislabeled",
    )
    _require(
        not config["science_inheritance"]["post_freeze_scientific_retuning_allowed"],
        "retuning allowed",
    )
    _validate_paths(config)
    sample = config["sample_count_hardening"]
    _require(
        sample["analysis_sample_count"]
        == sample["analysis_duration_seconds"] * sample["sample_rate_hz"]
        == 1048576,
        "analysis sample count mutation",
    )
    optimizer = config["symmetric_optimizer"]
    _require(
        optimizer["objectives"]
        == [
            "unpenalized maximum log likelihood",
            "maximum log posterior with frozen distance-volume and calibration-normal priors",
        ],
        "ML/MAP objectives changed",
    )
    _require(optimizer["separate_optimization"], "ML and MAP not separate")
    _require(len(optimizer["seeds"]) == 3, "three seeds required")
    _require(optimizer["finite_required"], "finite gate disabled")
    _require(optimizer["scipy_success_required"], "success gate disabled")
    _require(
        config["reservoir_power_calibration"]["optimal_network_snr_levels"] == [32.0, 48.0, 64.0],
        "reservoir power levels changed",
    )
    freeze = config["freeze_boundary"]
    _require(freeze["v6_strain_values_read_before_and_during_freeze"] == 0, "strain leak")
    _require(freeze["gw190425_status"] == "SEALED_NOT_ACQUIRED_NOT_OPENED", "holdout opened")
    _require(not freeze["real_response_authorized"], "response authorization forbidden")


def _validate_predecessor_hashes(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    groups = {
        "v4": config["blocked_v4_hashes"],
        "v5": config["strict_audit_of_v5"],
    }
    observed: dict[str, Any] = {}
    for group_name, group in groups.items():
        rows: dict[str, str] = {}
        for label in ["config", "module", "test", "prediction"]:
            path = base / group[f"{label}_path"]
            digest = v4._sha256_file(path)
            _require(digest == group[f"{label}_raw_sha256"], f"{group_name} {label} drift")
            rows[label] = digest
        observed[group_name] = rows
    v5_receipt = _read_json(base / config["strict_audit_of_v5"]["prediction_path"])
    _require(
        v5_receipt["content_sha256"] == config["strict_audit_of_v5"]["prediction_content_sha256"],
        "v5 content hash drift",
    )
    _require(v4.check(base).startswith("BLOCKED_PRE_RESPONSE"), "v4 replay failed")
    _require(
        v5.check(base).startswith("FROZEN_PRE_RESPONSE_IDENTIFIABILITY_FAIL"), "v5 replay failed"
    )
    return {"status": "PASS_V4_V5_EXACT_AND_REPLAYED", "hashes": observed}


def _science(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    v5_config = _read_json(base / config["strict_audit_of_v5"]["config_path"])
    science = v5.compose_science_config(v5_config, base)
    prep = science["preprocessing"]
    sample = config["sample_count_hardening"]
    _require(
        prep["analysis_duration_seconds"] == sample["analysis_duration_seconds"], "duration drift"
    )
    _require(prep["sample_rate_hz"] == sample["sample_rate_hz"], "sample rate drift")
    _require(prep["analysis_sample_count"] == sample["analysis_sample_count"], "sample count drift")
    return science


def _additional_source_audit(config: Mapping[str, Any], base: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for source in config["additional_source_and_runtime_bytes"]:
        path = base / source["path"]
        _require(path.is_file(), f"missing additional source {source['path']}")
        size = path.stat().st_size
        digest = v4._sha256_file(path)
        _require(size == source["bytes"], f"additional source byte drift {source['role']}")
        _require(digest == source["sha256"], f"additional source hash drift {source['role']}")
        rows.append(
            {"role": source["role"], "path": source["path"], "bytes": size, "sha256": digest}
        )
    return {"status": "PASS_ADDITIONAL_SOURCE_BYTES", "files": rows}


def _decode_record_digest(value: str) -> bytes:
    algorithm, encoded = value.split("=", 1)
    _require(algorithm == "sha256", f"unsupported RECORD hash {algorithm}")
    return base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))


def _runtime_byte_audit(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    runtime = config["runtime_integrity"]
    wheel_spec = next(
        row
        for row in config["additional_source_and_runtime_bytes"]
        if row["role"] == "exact_lalsuite_runtime_wheel"
    )
    wheel_path = base / wheel_spec["path"]
    wheel_verified = 0
    with zipfile.ZipFile(wheel_path) as archive:
        record_bytes = archive.read(runtime["wheel_record_member"])
        _require(len(record_bytes) == runtime["wheel_record_bytes"], "wheel RECORD bytes drift")
        _require(
            v4._sha256_bytes(record_bytes) == runtime["wheel_record_sha256"], "wheel RECORD drift"
        )
        for filename, digest_text, size_text in csv.reader(
            io.StringIO(record_bytes.decode("utf-8"))
        ):
            if not digest_text:
                continue
            payload = archive.read(filename)
            _require(len(payload) == int(size_text), f"wheel member size drift {filename}")
            _require(
                v4._sha256_bytes(payload).encode()
                == _decode_record_digest(digest_text).hex().encode(),
                f"wheel member hash drift {filename}",
            )
            wheel_verified += 1

    installed_record_path = base / runtime["installed_record_path"]
    installed_record_bytes = installed_record_path.read_bytes()
    _require(
        len(installed_record_bytes) == runtime["installed_record_bytes"],
        "installed RECORD bytes drift",
    )
    _require(
        v4._sha256_bytes(installed_record_bytes) == runtime["installed_record_sha256"],
        "installed RECORD hash drift",
    )
    site_packages = installed_record_path.parent.parent
    runtime_root = (base / science["package"]["runtime_path"]).parent.parent.resolve()
    installed_verified = 0
    installed_rows = 0
    for filename, digest_text, size_text in csv.reader(
        io.StringIO(installed_record_bytes.decode("utf-8"))
    ):
        installed_rows += 1
        if not digest_text:
            continue
        path = (site_packages / filename).resolve()
        try:
            path.relative_to(runtime_root)
        except ValueError as error:
            raise CoherentV6Error(f"installed RECORD escapes runtime: {filename}") from error
        _require(path.is_file(), f"installed runtime file missing {filename}")
        payload = path.read_bytes()
        _require(len(payload) == int(size_text), f"installed file size drift {filename}")
        _require(
            v4._sha256_bytes(payload).encode() == _decode_record_digest(digest_text).hex().encode(),
            f"installed file hash drift {filename}",
        )
        installed_verified += 1
    _require(
        installed_rows == runtime["installed_distribution_files"],
        "installed distribution file count drift",
    )
    version_audit = v4._runtime_audit(science)
    return {
        "status": "PASS_WHEEL_AND_INSTALLED_RUNTIME_BYTES",
        "wheel_hashed_members_verified": wheel_verified,
        "installed_hashed_files_verified": installed_verified,
        "installed_record_rows": installed_rows,
        "versions": version_audit,
    }


def _projected_snr_from_components(
    detector_dh: Mapping[str, float], detector_hh: Mapping[str, float]
) -> dict[str, Any]:
    _require(detector_dh.keys() == detector_hh.keys(), "SNR detector mismatch")
    per_detector = {
        detector: float(detector_dh[detector] / math.sqrt(detector_hh[detector]))
        for detector in detector_dh
    }
    coherent = float(sum(detector_dh.values()) / math.sqrt(sum(detector_hh.values())))
    return {"per_detector": per_detector, "coherent_network": coherent}


def _fit_projection(
    context: v4.SyntheticContext,
    data: Mapping[str, np.ndarray],
    waveform: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    detector_dh = {
        detector: v4._inner(context, detector, data[detector], waveform[detector])
        for detector in context.config["event"]["detectors"]
    }
    detector_hh = {
        detector: v4._inner(context, detector, waveform[detector], waveform[detector])
        for detector in context.config["event"]["detectors"]
    }
    result = _projected_snr_from_components(detector_dh, detector_hh)
    result["detector_dh"] = detector_dh
    result["detector_hh"] = detector_hh
    return result


def _physical_to_normalized(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return (values - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])


def _normalized_to_physical(values: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    return bounds[:, 0] + values * (bounds[:, 1] - bounds[:, 0])


def _branch_parameter_names(science: Mapping[str, Any], branch_id: str) -> list[str]:
    return [row["name"] for row in v4._branch_spec(science, branch_id)["parameters"]]


def _fit_one_objective(
    context: v4.SyntheticContext,
    data: Mapping[str, np.ndarray],
    common_anchor: Mapping[str, float],
    branch_id: str,
    objective_id: str,
    seed: int,
    seed_index: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    science = context.config
    common_bounds = np.asarray(v4._common_bounds(science), dtype=float)
    branch_bounds = np.asarray(v4._branch_bounds(science, branch_id), dtype=float)
    if branch_bounds.size == 0:
        branch_bounds = branch_bounds.reshape((0, 2))
    bounds = np.vstack([common_bounds, branch_bounds])
    common_values = np.asarray([common_anchor[name] for name in v4.COMMON_PARAMETER_NAMES])
    common_start = _physical_to_normalized(common_values, common_bounds)
    rng = np.random.default_rng(seed)
    common_start = np.clip(
        common_start
        + rng.normal(
            0.0,
            float(
                config["symmetric_optimizer"]["common_start_rule"].split("sigma ")[1].split(" ")[0]
            ),
            len(common_start),
        ),
        0.0,
        1.0,
    )
    branch_starts = config["symmetric_optimizer"]["branch_start_rule"][branch_id]
    branch_physical = np.asarray(branch_starts[seed_index % len(branch_starts)], dtype=float)
    branch_start = (
        _physical_to_normalized(branch_physical, branch_bounds)
        if len(branch_physical)
        else np.empty(0, dtype=float)
    )
    initial = np.concatenate([common_start, branch_start])

    def evaluate(normalized: np.ndarray) -> tuple[float, float, dict[str, float], dict[str, float]]:
        physical = _normalized_to_physical(normalized, bounds)
        common = v4._vector_to_common(physical[: len(v4.COMMON_PARAMETER_NAMES)])
        branch = v4._vector_to_branch(
            science, branch_id, physical[len(v4.COMMON_PARAMETER_NAMES) :]
        )
        waveform = v4._network_waveform(context, common, branch_id, branch)
        log_likelihood, _ = v4._log_likelihood_ratio(context, data, waveform)
        log_posterior = log_likelihood + v4._log_prior(common)
        return log_likelihood, log_posterior, common, branch

    def objective(normalized: np.ndarray) -> float:
        try:
            log_likelihood, log_posterior, _, _ = evaluate(normalized)
            value = log_likelihood if objective_id == "ML" else log_posterior
            return -value if math.isfinite(value) else 1.0e100
        except (RuntimeError, ValueError, FloatingPointError):
            return 1.0e100

    optimizer = config["symmetric_optimizer"]
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * len(initial),
        options={
            "maxiter": int(optimizer["max_iterations"]),
            "ftol": float(optimizer["ftol"]),
            "gtol": float(optimizer["gtol"]),
            "maxls": int(optimizer["max_line_search_steps"]),
        },
    )
    log_likelihood, log_posterior, common, branch = evaluate(fitted.x)
    waveform = v4._network_waveform(context, common, branch_id, branch)
    projection = _fit_projection(context, data, waveform)
    finite = bool(
        np.all(np.isfinite(fitted.x))
        and math.isfinite(log_likelihood)
        and math.isfinite(log_posterior)
        and math.isfinite(projection["coherent_network"])
    )
    return {
        "objective": objective_id,
        "seed": seed,
        "success": bool(fitted.success),
        "finite": finite,
        "message": str(fitted.message),
        "evaluations": int(fitted.nfev),
        "iterations": int(fitted.nit),
        "log_likelihood": log_likelihood,
        "log_posterior_without_shared_constants": log_posterior,
        "common_parameters": common,
        "branch_parameters": branch,
        "projected_matched_snr": projection,
    }


def _fit_model(
    context: v4.SyntheticContext,
    data: Mapping[str, np.ndarray],
    common_anchor: Mapping[str, float],
    branch_id: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = config["symmetric_optimizer"]["seeds"]
    by_objective: dict[str, Any] = {}
    for objective_id in ["ML", "MAP"]:
        results = [
            _fit_one_objective(
                context,
                data,
                common_anchor,
                branch_id,
                objective_id,
                int(seed),
                seed_index,
                config,
            )
            for seed_index, seed in enumerate(seeds)
        ]
        metric = (
            "log_likelihood" if objective_id == "ML" else "log_posterior_without_shared_constants"
        )
        best = max(results, key=lambda row: row[metric])
        spread = 2.0 * (max(row[metric] for row in results) - min(row[metric] for row in results))
        threshold = float(
            config["symmetric_optimizer"][
                "three_seed_delta_2_log_likelihood_max"
                if objective_id == "ML"
                else "three_seed_delta_2_log_posterior_max"
            ]
        )
        valid = bool(
            all(row["success"] and row["finite"] for row in results)
            and math.isfinite(spread)
            and spread <= threshold
        )
        by_objective[objective_id] = {
            "seeds": results,
            "best": best,
            "delta_2_objective_spread": spread,
            "convergence_threshold": threshold,
            "valid": valid,
        }
    return {
        "branch": branch_id,
        "ML": by_objective["ML"],
        "MAP": by_objective["MAP"],
        "optimizer_valid": by_objective["ML"]["valid"] and by_objective["MAP"]["valid"],
    }


def _injection_common(science: Mapping[str, Any]) -> dict[str, float]:
    injection = science["target_free_gates"]["zero_noise_gr_control"]
    common = dict(injection["parameters"])
    common["luminosity_distance_mpc"] = 40.0
    common.pop("calibration_coefficients", None)
    for detector in science["event"]["detectors"]:
        common[f"calibration_amplitude_{detector}"] = 0.0
        common[f"calibration_phase_{detector}"] = 0.0
    return common


def _fit_all_models(
    context: v4.SyntheticContext,
    data: Mapping[str, np.ndarray],
    common_anchor: Mapping[str, float],
    config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    models = [row["id"] for row in context.config["transfer_branches"]]
    fits = [_fit_model(context, data, common_anchor, branch_id, config) for branch_id in models]
    n = 2 * len(context.frequency) * len(context.config["event"]["detectors"])
    for fit in fits:
        extras = int(v4._branch_spec(context.config, fit["branch"])["extra_parameter_count"])
        fit["parameter_count"] = len(v4.COMMON_PARAMETER_NAMES) + extras
        fit["bic"] = -2.0 * fit["ML"]["best"]["log_likelihood"] + fit["parameter_count"] * math.log(
            n
        )
    return fits


def _unique_bic_winner(fits: Sequence[Mapping[str, Any]]) -> str | None:
    ordered = sorted(fits, key=lambda row: row["bic"])
    return str(ordered[0]["branch"]) if ordered[1]["bic"] - ordered[0]["bic"] > 1.0e-9 else None


def _amplitude_check(
    science: Mapping[str, Any],
    fits: Sequence[Mapping[str, Any]],
    injected_branch: str,
    injected_parameters: Mapping[str, float],
) -> dict[str, Any]:
    amplitude_name = _branch_parameter_names(science, injected_branch)[0]
    injected = float(injected_parameters[amplitude_name])
    fit = next(row for row in fits if row["branch"] == injected_branch)
    recovered = float(fit["ML"]["best"]["branch_parameters"][amplitude_name])
    tolerance = max(0.25 * abs(injected), 0.05)
    return {
        "name": amplitude_name,
        "injected": injected,
        "recovered": recovered,
        "absolute_error": abs(recovered - injected),
        "tolerance": tolerance,
        "passed": abs(recovered - injected) <= tolerance,
    }


def _optimizer_recovery(
    context: v4.SyntheticContext,
    common_template: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    science = context.config
    target_snr = float(science["target_free_gates"]["zero_noise_gr_control"]["network_snr"])
    data, common = v4._normalized_synthetic_data(context, common_template, target_snr)
    fits = _fit_all_models(context, data, common, config)
    gr = next(row for row in fits if row["branch"] == "GR")
    gr_best = gr["ML"]["best"]
    winner = _unique_bic_winner(fits)
    checks = {
        "all_model_optimizers_valid": all(row["optimizer_valid"] for row in fits),
        "gr_projected_snr_fraction": gr_best["projected_matched_snr"]["coherent_network"]
        / target_snr
        >= 0.99,
        "gr_chirp_mass": abs(
            gr_best["common_parameters"]["chirp_mass_detector_solar"]
            - common["chirp_mass_detector_solar"]
        )
        <= 0.0015,
        "gr_geocentric_time": abs(
            gr_best["common_parameters"]["geocentric_time_offset_seconds"]
            - common["geocentric_time_offset_seconds"]
        )
        <= 0.002,
        "gr_unique_minimum_bic": winner == "GR",
    }
    return {
        "status": "PASS_OPTIMIZER_RECOVERY_GATE"
        if all(checks.values())
        else "FAIL_OPTIMIZER_RECOVERY_GATE",
        "target_snr": target_snr,
        "injected_common": common,
        "fits": fits,
        "bic_winner": winner,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _registered_power(
    context: v4.SyntheticContext,
    common_template: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    science = context.config
    target_snr = float(science["target_free_gates"]["zero_noise_gr_control"]["network_snr"])
    rows: list[dict[str, Any]] = []
    for injection in science["target_free_gates"]["branch_injections"]:
        data, common = v4._normalized_synthetic_data(
            context,
            common_template,
            target_snr,
            injection["branch"],
            injection["parameters"],
        )
        fits = _fit_all_models(context, data, common, config)
        all_valid = all(row["optimizer_valid"] for row in fits)
        winner = _unique_bic_winner(fits)
        amplitude = _amplitude_check(science, fits, injection["branch"], injection["parameters"])
        identifiable = all_valid and winner == injection["branch"] and amplitude["passed"]
        classification = (
            "INVALID_BRANCH_OPTIMIZER"
            if not all_valid
            else (
                "IDENTIFIABLE_AT_REGISTERED_SNR32"
                if identifiable
                else "POWERLESS_AT_REGISTERED_SNR32"
            )
        )
        rows.append(
            {
                "injection_id": injection["id"],
                "injected_branch": injection["branch"],
                "injected_parameters": injection["parameters"],
                "target_snr": target_snr,
                "fits": fits,
                "bic_winner": winner,
                "amplitude_recovery": amplitude,
                "all_model_optimizers_valid": all_valid,
                "classification": classification,
            }
        )
    optimizer_gate_passed = all(row["all_model_optimizers_valid"] for row in rows)
    return {
        "status": (
            "PASS_BRANCH_OPTIMIZERS_POWER_CLASSIFIED"
            if optimizer_gate_passed
            else "FAIL_BRANCH_OPTIMIZER_GATE"
        ),
        "optimizer_gate_passed": optimizer_gate_passed,
        "injections": rows,
        "identifiable_count": sum(
            row["classification"] == "IDENTIFIABLE_AT_REGISTERED_SNR32" for row in rows
        ),
        "powerless_count": sum(
            row["classification"] == "POWERLESS_AT_REGISTERED_SNR32" for row in rows
        ),
    }


def _reservoir_power_calibration(
    context: v4.SyntheticContext,
    common_template: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    calibration = config["reservoir_power_calibration"]
    science = context.config
    rows: list[dict[str, Any]] = []
    for target_snr in calibration["optimal_network_snr_levels"]:
        data, common = v4._normalized_synthetic_data(
            context,
            common_template,
            float(target_snr),
            "RESERVOIR",
            calibration["injected_parameters"],
        )
        fits = _fit_all_models(context, data, common, config)
        all_valid = all(row["optimizer_valid"] for row in fits)
        winner = _unique_bic_winner(fits)
        amplitude = _amplitude_check(science, fits, "RESERVOIR", calibration["injected_parameters"])
        identifiable = all_valid and winner == "RESERVOIR" and amplitude["passed"]
        rows.append(
            {
                "target_snr": float(target_snr),
                "fits": fits,
                "bic_winner": winner,
                "amplitude_recovery": amplitude,
                "all_model_optimizers_valid": all_valid,
                "identifiable": identifiable,
            }
        )
    valid = all(row["all_model_optimizers_valid"] for row in rows)
    identifiable_levels = [row["target_snr"] for row in rows if row["identifiable"]]
    classification = (
        "INVALID_RESERVOIR_POWER_OPTIMIZER"
        if not valid
        else (
            f"IDENTIFIABLE_FROM_SNR{min(identifiable_levels):g}"
            if identifiable_levels
            else "POWERLESS_THROUGH_SNR64"
        )
    )
    return {
        "status": (
            "PASS_RESERVOIR_POWER_OPTIMIZERS_CLASSIFIED"
            if valid
            else "FAIL_RESERVOIR_POWER_OPTIMIZER_GATE"
        ),
        "injected_parameters": calibration["injected_parameters"],
        "levels": rows,
        "classification": classification,
        "optimizer_gate_passed": valid,
    }


def _target_free_rebuild(
    config: Mapping[str, Any], science: Mapping[str, Any], base: Path
) -> dict[str, Any]:
    context = v4._synthetic_context(science, base)
    common = _injection_common(science)
    optimizer = _optimizer_recovery(context, common, config)
    registered = _registered_power(context, common, config)
    reservoir = _reservoir_power_calibration(context, common, config)
    method_passed = bool(
        optimizer["passed"]
        and registered["optimizer_gate_passed"]
        and reservoir["optimizer_gate_passed"]
    )
    return {
        "status": "PASS_METHOD_GATES" if method_passed else "FAIL_METHOD_GATES",
        "optimizer_recovery": optimizer,
        "registered_branch_power": registered,
        "reservoir_power_calibration": reservoir,
        "method_passed": method_passed,
        "strain_values_read": 0,
        "real_likelihood_values_computed": 0,
        "gw190425_opened": 0,
    }


def _static_rebuild(config: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    predecessors = _validate_predecessor_hashes(config, base)
    science = _science(config, base)
    source = v4._source_audit(science, base)
    additional = _additional_source_audit(config, base)
    v5_config = _read_json(base / config["strict_audit_of_v5"]["config_path"])
    schemas = v5._source_schema_audit(v5_config, science, base)
    hdf = v5._hdf_and_dq_audit(v5_config, science, base)
    support = v5._support_and_nyquist_audit(science)
    runtime = _runtime_byte_audit(config, science, base)
    return science, {
        "predecessors": predecessors,
        "source": source,
        "additional_sources": additional,
        "schemas": schemas,
        "hdf_and_dq": hdf,
        "support_and_nyquist": support,
        "runtime": runtime,
    }


def _package_hashes(config: Mapping[str, Any], base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": v4._sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": v4._sha256_file(base / MODULE_PATH),
        "test_raw_sha256": v4._sha256_file(base / TEST_PATH),
        "v4_config_raw_sha256": config["blocked_v4_hashes"]["config_raw_sha256"],
        "v4_prediction_raw_sha256": config["blocked_v4_hashes"]["prediction_raw_sha256"],
        "v5_config_raw_sha256": config["strict_audit_of_v5"]["config_raw_sha256"],
        "v5_prediction_raw_sha256": config["strict_audit_of_v5"]["prediction_raw_sha256"],
    }


def freeze(root: Path | None = None) -> str:
    base = _base(root)
    config = load_config(base)
    science, static = _static_rebuild(config, base)
    controls = _target_free_rebuild(config, science, base)
    artifacts = {
        "strict-static-audit.json": static,
        "target-free-optimizer-and-power-gates.json": controls,
    }
    for name, value in artifacts.items():
        _write_json(base / ARTIFACT_DIR / name, value)
    decision = (
        "FROZEN_PRE_RESPONSE_METHOD_GATES_PASS_POWER_CLASSIFIED_PENDING_INDEPENDENT_AUDIT"
        if controls["method_passed"]
        else "FROZEN_PRE_RESPONSE_METHOD_GATES_FAIL_NO_RESPONSE"
    )
    receipt: dict[str, Any] = {
        "schema_version": PREDICTION_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": decision,
        "v5_audit_label": config["strict_audit_of_v5"]["label"],
        "package_hashes": _package_hashes(config, base),
        "artifact_sha256": {
            name: v4._sha256_file(base / ARTIFACT_DIR / name) for name in artifacts
        },
        "artifact_content_sha256": {
            name: v4._sha256_bytes(v4._canonical(value)) for name, value in artifacts.items()
        },
        "optimizer_recovery_status": controls["optimizer_recovery"]["status"],
        "registered_branch_power_status": controls["registered_branch_power"]["status"],
        "reservoir_power_status": controls["reservoir_power_calibration"]["status"],
        "reservoir_power_classification": controls["reservoir_power_calibration"]["classification"],
        "method_passed": controls["method_passed"],
        "real_response_authorized": False,
        "independent_audit_required": True,
        "access_ledger": {
            "source_files_hashed": len(static["source"]["files"])
            + len(static["additional_sources"]["files"]),
            "wheel_hashed_members_verified": static["runtime"]["wheel_hashed_members_verified"],
            "installed_runtime_files_verified": static["runtime"][
                "installed_hashed_files_verified"
            ],
            "hdf5_headers_opened": static["hdf_and_dq"]["hdf5_files_opened"],
            "dq_values_read": static["hdf_and_dq"]["dq_values_read"],
            "strain_values_read": 0,
            "real_likelihood_values_computed": 0,
            "gw190425_opened": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
    }
    receipt["content_sha256"] = v4._self_hash(receipt)
    _write_json(base / PREDICTION_PATH, receipt)
    return decision


def check(root: Path | None = None) -> str:
    base = _base(root)
    config = load_config(base)
    receipt = _read_json(base / PREDICTION_PATH)
    _require(receipt["schema_version"] == PREDICTION_SCHEMA, "receipt schema drift")
    _require(receipt["content_sha256"] == v4._self_hash(receipt), "receipt content drift")
    _require(receipt["package_hashes"] == _package_hashes(config, base), "package drift")
    science, static = _static_rebuild(config, base)
    controls = _target_free_rebuild(config, science, base)
    rebuilt = {
        "strict-static-audit.json": static,
        "target-free-optimizer-and-power-gates.json": controls,
    }
    for name, value in rebuilt.items():
        _require(
            v4._sha256_bytes(v4._canonical(value)) == receipt["artifact_content_sha256"][name],
            f"rebuilt content drift {name}",
        )
        _require(
            v4._sha256_file(base / ARTIFACT_DIR / name) == receipt["artifact_sha256"][name],
            f"artifact byte drift {name}",
        )
    _require(
        controls["optimizer_recovery"]["status"] == receipt["optimizer_recovery_status"],
        "optimizer recovery decision drift",
    )
    _require(
        controls["registered_branch_power"]["status"] == receipt["registered_branch_power_status"],
        "registered power decision drift",
    )
    _require(
        controls["reservoir_power_calibration"]["status"] == receipt["reservoir_power_status"],
        "reservoir power decision drift",
    )
    _require(receipt["access_ledger"]["strain_values_read"] == 0, "strain access")
    _require(receipt["access_ledger"]["gw190425_opened"] == 0, "holdout access")
    _require(not receipt["real_response_authorized"], "response audit bypass")
    return receipt["decision"]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("freeze")
    subparsers.add_parser("check")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    print(freeze() if arguments.command == "freeze" else check())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
