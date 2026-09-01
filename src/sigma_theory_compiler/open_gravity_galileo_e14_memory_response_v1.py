"""Frozen E14-only real-response test of the persistent causal-memory quadrature."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_galileo_e14_memory_response_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_galileo_e14_memory_response_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_galileo_e14_memory_response_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-galileo-e14-memory-response-v1/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "41749b6a941db912abca8ce48b480cbcd3d14312b4420faf958bbdc3c43613f1"
_CONFIG_CONTENT_SHA256 = "a42c6d7defae93bf2c2f27add22581559ba3c7c1e9518af8216494148d7523d6"
_MODULE_SEMANTIC_SHA256 = "c24d26a1b4433dfb644ee34381e0434d1bd5879a3431268e725b6a58170a6dc3"
_TEST_RAW_SHA256 = "5435fcfcace2f4b9b23e205a3291cc74e2396db0bfef7aca0998783a3d829369"
_SCHEMA = "invariant-open-gravity-galileo-e14-memory-response-1.0"


class GalileoE14ResponseError(RuntimeError):
    """Raised when a frozen E14 response invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GalileoE14ResponseError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    marker = '_MODULE_SEMANTIC_SHA256 = "'
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    normalized = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(normalized.encode()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GalileoE14ResponseError(f"invalid JSON: {path}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["status"] == "FROZEN_E14_DEVELOPMENT_ONLY_BEFORE_RESPONSE_DECODE",
        "status changed",
    )
    auth = config["authorization"]
    _require(auth["development_response_prns"] == ["E14"], "response authorization widened")
    _require(auth["forbidden_response_prns"] == ["E18", "E25", "E31"], "forbidden set changed")
    _require(auth["other_response_prns"] == "FORBIDDEN", "other PRNs authorized")
    _require(auth["formula_retuning_events"] == 0, "retuning authorized")
    source = config["source"]
    _require(source["source_only_period_samples"] == 155, "source period changed")
    _require(source["source_only_circular_controls"] == ["E25", "E31"], "controls changed")
    _require(
        source["phase_scramble_shifts_samples"] == [-78, -58, -39, -19, 19, 39, 58, 78],
        "phase shifts changed",
    )
    _require(
        config["physical_template"]["tau_seconds"]
        == [300.0, 1800.0, 7200.0, 21600.0, 43200.0],
        "tau grid changed",
    )
    _require("Always BLOCKED" in config["development_decision"]["E18_promotion_gate"], "E18 gate widened")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "output path changed")


def load_config() -> dict[str, Any]:
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw hash changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw hash changed")
    config = _read_json(CONFIG_PATH)
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _epoch(parts: Sequence[str]) -> datetime:
    second = float(parts[5])
    whole = math.floor(second)
    microsecond = round((second - whole) * 1_000_000)
    return datetime(
        int(parts[0]),
        int(parts[1]),
        int(parts[2]),
        int(parts[3]),
        int(parts[4]),
        whole,
        microsecond,
        tzinfo=UTC,
    )


def read_e14_clk_values(path: Path) -> tuple[dict[datetime, float], dict[str, int]]:
    """Decode numeric payloads only for exact AS E14 records after the header."""
    values: dict[datetime, float] = {}
    after_header = False
    header_lines = 0
    skipped_payload_rows = 0
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if not after_header:
                header_lines += 1
                if line[60:80].strip() == "END OF HEADER":
                    after_header = True
                continue
            if line[:2] != "AS":
                continue
            # Only these first six characters are inspected for non-E14 payload rows.
            if line[3:6] != "E14":
                skipped_payload_rows += 1
                continue
            fields = line.split()
            _require(fields[:2] == ["AS", "E14"], f"malformed E14 CLK record: {path.name}")
            _require(len(fields) >= 10, f"short E14 CLK record: {path.name}")
            epoch = _epoch(fields[2:8])
            _require(epoch not in values, f"duplicate E14 CLK epoch: {path.name}")
            _require(int(fields[8]) >= 1, f"E14 CLK value count absent: {path.name}")
            values[epoch] = float(fields[9])
    _require(after_header, f"CLK header boundary absent: {path.name}")
    _require(values, f"no E14 CLK rows: {path.name}")
    return values, {
        "header_lines": header_lines,
        "e14_numeric_rows_decoded": len(values),
        "non_e14_numeric_payload_rows_decoded": 0,
        "non_e14_payload_rows_prefix_skipped": skipped_payload_rows,
    }


def read_e14_sp3(path: Path) -> dict[datetime, np.ndarray]:
    positions: dict[datetime, np.ndarray] = {}
    current: datetime | None = None
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.startswith("*"):
                current = _epoch(line[1:].split())
            elif line.startswith("PE14") and current is not None:
                fields = line[4:].split()
                _require(len(fields) >= 3, f"short E14 SP3 record: {path.name}")
                positions[current] = np.asarray([float(value) * 1000.0 for value in fields[:3]])
    _require(positions, f"no E14 SP3 positions: {path.name}")
    return positions


def merge_source_positions(daily: Sequence[Mapping[datetime, np.ndarray]]) -> dict[datetime, np.ndarray]:
    """Retain the earlier daily solution at duplicated midnight epochs."""
    merged: dict[datetime, np.ndarray] = {}
    for rows in daily:
        for epoch, xyz in rows.items():
            if epoch not in merged:
                merged[epoch] = xyz
    return merged


def instantaneous_source(
    positions: Mapping[datetime, np.ndarray], constants: Mapping[str, float]
) -> dict[datetime, float]:
    epochs = sorted(positions)
    xyz = np.asarray([positions[epoch] for epoch in epochs])
    seconds = np.asarray([(epoch - epochs[0]).total_seconds() for epoch in epochs])
    velocity_ecef = np.empty_like(xyz)
    velocity_ecef[0] = (xyz[1] - xyz[0]) / (seconds[1] - seconds[0])
    velocity_ecef[-1] = (xyz[-1] - xyz[-2]) / (seconds[-1] - seconds[-2])
    delta = (seconds[2:] - seconds[:-2])[:, None]
    velocity_ecef[1:-1] = (xyz[2:] - xyz[:-2]) / delta
    omega = float(constants["earth_rotation_rad_s"])
    omega_cross_r = np.column_stack((-omega * xyz[:, 1], omega * xyz[:, 0], np.zeros(len(xyz))))
    velocity = velocity_ecef + omega_cross_r
    radius = np.linalg.norm(xyz, axis=1)
    speed_squared = np.sum(velocity * velocity, axis=1)
    gm = float(constants["earth_GM_m3_s2"])
    c = float(constants["speed_of_light_m_s"])
    q = -(gm / radius + speed_squared / 2.0) / c**2
    return dict(zip(epochs, q, strict=True))


def periodic_memory(source: Sequence[float], tau_seconds: float, period: int) -> np.ndarray:
    q = np.asarray(source, dtype=float)
    _require(len(q) > period >= 1, "invalid source-only period")
    a = math.exp(-300.0 / tau_seconds)
    weights = np.asarray([a ** (period - 1 - index) for index in range(period)])
    psi0 = (1.0 - a) * float(np.dot(weights, q[:period])) / (1.0 - a**period)
    psi = np.empty_like(q)
    psi[0] = psi0
    for index in range(len(q) - 1):
        psi[index + 1] = a * psi[index] + (1.0 - a) * q[index]
    return psi


def project_day(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    _require(array.ndim in (1, 2), "day projection requires vector or matrix")
    rows = len(array)
    time = np.linspace(-1.0, 1.0, rows)
    nuisance = np.column_stack((np.ones(rows), time))
    coefficients = np.linalg.lstsq(nuisance, array, rcond=None)[0]
    return array - nuisance @ coefficients


def ar_transform(values: np.ndarray, rho: float) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    transformed = np.empty_like(array)
    transformed[0] = math.sqrt(1.0 - rho * rho) * array[0]
    transformed[1:] = array[1:] - rho * array[:-1]
    return transformed


def fit_ar1(
    y_days: Sequence[np.ndarray], x_days: Sequence[np.ndarray]
) -> dict[str, Any]:
    column_count = 0 if not x_days else x_days[0].shape[1]
    rho = 0.0
    beta = np.zeros(column_count)
    for _ in range(20):
        y_t = np.concatenate([ar_transform(day, rho) for day in y_days])
        if column_count:
            x_t = np.vstack([ar_transform(day, rho) for day in x_days])
            beta = np.linalg.lstsq(x_t, y_t, rcond=None)[0]
            residual_days = [y - x @ beta for y, x in zip(y_days, x_days, strict=True)]
        else:
            residual_days = list(y_days)
        numerator = sum(float(np.dot(day[1:], day[:-1])) for day in residual_days)
        denominator = sum(float(np.dot(day[:-1], day[:-1])) for day in residual_days)
        new_rho = float(np.clip(numerator / denominator, -0.995, 0.995))
        if abs(new_rho - rho) < 1.0e-10:
            rho = new_rho
            break
        rho = new_rho
    transformed_residual = np.concatenate([ar_transform(day, rho) for day in residual_days])
    variance = max(float(np.dot(transformed_residual, transformed_residual) / len(transformed_residual)), 1e-40)
    return {"beta": beta, "rho": rho, "innovation_variance": variance}


def ar1_log_likelihood(residual: np.ndarray, rho: float, variance: float) -> float:
    transformed = ar_transform(residual, rho)
    rows = len(residual)
    return float(
        -0.5 * rows * math.log(2.0 * math.pi * variance)
        + 0.5 * math.log(1.0 - rho * rho)
        - 0.5 * float(np.dot(transformed, transformed)) / variance
    )


def cross_validate(
    y_days: Sequence[np.ndarray], x_days: Sequence[np.ndarray]
) -> dict[str, Any]:
    _require(len(y_days) == len(x_days) == 7, "exactly seven days required")
    folds: list[dict[str, Any]] = []
    for heldout in range(7):
        train_y = [day for index, day in enumerate(y_days) if index != heldout]
        train_x = [day for index, day in enumerate(x_days) if index != heldout]
        null_fit = fit_ar1(train_y, [])
        alt_fit = fit_ar1(train_y, train_x)
        y_test = y_days[heldout]
        x_test = x_days[heldout]
        null_ll = ar1_log_likelihood(
            y_test, null_fit["rho"], null_fit["innovation_variance"]
        )
        alt_residual = y_test - x_test @ alt_fit["beta"]
        alt_ll = ar1_log_likelihood(
            alt_residual, alt_fit["rho"], alt_fit["innovation_variance"]
        )
        folds.append(
            {
                "heldout_day": heldout,
                "null_log_likelihood": null_ll,
                "alternative_log_likelihood": alt_ll,
                "delta_log_likelihood": alt_ll - null_ll,
                "beta": [float(value) for value in alt_fit["beta"]],
                "null_rho": null_fit["rho"],
                "alternative_rho": alt_fit["rho"],
                "null_innovation_sigma": math.sqrt(null_fit["innovation_variance"]),
                "alternative_innovation_sigma": math.sqrt(alt_fit["innovation_variance"]),
            }
        )
    return {
        "total_delta_log_likelihood": sum(row["delta_log_likelihood"] for row in folds),
        "folds": folds,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"no CSV rows for {path.name}")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    _atomic_write(path, stream.getvalue().encode())


def _load_development(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    source_dir = Path(config["source"]["directory"])
    daily_sp3: list[dict[datetime, np.ndarray]] = []
    daily_clk: list[dict[datetime, float]] = []
    access: list[dict[str, int]] = []
    for day in config["source"]["days"]:
        sp3 = read_e14_sp3(source_dir / f"esoc2044{day}.sp3")
        clk, counts = read_e14_clk_values(source_dir / f"esoc2044{day}.clk")
        daily_sp3.append(sp3)
        daily_clk.append(clk)
        access.append(counts)
    merged_positions = merge_source_positions(daily_sp3)
    source_by_epoch = instantaneous_source(
        merged_positions, config["physical_template"]["constants"]
    )
    source_epochs = sorted(source_by_epoch)
    q = np.asarray([source_by_epoch[epoch] for epoch in source_epochs])
    source_index = {epoch: index for index, epoch in enumerate(source_epochs)}
    x_by_tau_epoch: dict[float, np.ndarray] = {}
    for tau in config["physical_template"]["tau_seconds"]:
        psi = periodic_memory(q, tau, config["source"]["source_only_period_samples"])
        x_by_tau_epoch[tau] = psi - q

    y_days: list[np.ndarray] = []
    q_days: list[np.ndarray] = []
    dq_days: list[np.ndarray] = []
    raw_x_days: dict[float, list[np.ndarray]] = {
        tau: [] for tau in config["physical_template"]["tau_seconds"]
    }
    daily_summary: list[dict[str, Any]] = []
    for day, (sp3, clk) in enumerate(zip(daily_sp3, daily_clk, strict=True)):
        common = sorted(set(sp3) & set(clk))[1:-1]
        _require(len(common) >= 3, f"too few common epochs day {day}")
        pairs = list(pairwise(common))
        durations = np.asarray(
            [(right - left).total_seconds() for left, right in pairs]
        )
        _require(np.all(durations == 300.0), f"non-300 s retained interval day {day}")
        y = np.asarray(
            [(clk[right] - clk[left]) / 300.0 for left, right in pairs]
        )
        q_mid = np.asarray(
            [
                0.5 * (source_by_epoch[left] + source_by_epoch[right])
                for left, right in pairs
            ]
        )
        dq = np.asarray(
            [
                (source_by_epoch[right] - source_by_epoch[left]) / 300.0
                for left, right in pairs
            ]
        )
        y_days.append(y)
        q_days.append(q_mid)
        dq_days.append(dq)
        for tau, x_epochs in x_by_tau_epoch.items():
            raw_x_days[tau].append(
                np.asarray(
                    [
                        0.5 * (x_epochs[source_index[left]] + x_epochs[source_index[right]])
                        for left, right in pairs
                    ]
                )
            )
        daily_summary.append(
            {
                "day": day,
                "clk_e14_rows_decoded": len(clk),
                "sp3_e14_epochs": len(sp3),
                "common_epochs_before_edge_exclusion": len(set(sp3) & set(clk)),
                "retained_common_epochs": len(common),
                "response_intervals": len(y),
                "response_mean": float(np.mean(y)),
                "response_std": float(np.std(y)),
            }
        )
    data = {
        "y_raw": y_days,
        "q_raw": q_days,
        "dq_raw": dq_days,
        "x_raw": raw_x_days,
        "daily_summary": daily_summary,
    }
    accounting = {
        "clk_header_lines_opened": sum(row["header_lines"] for row in access),
        "E14_clock_value_rows_decoded": sum(row["e14_numeric_rows_decoded"] for row in access),
        "E18_clock_value_rows_decoded": 0,
        "circular_control_clock_value_rows_decoded": 0,
        "other_prn_numeric_payload_rows_decoded": 0,
        "other_prn_payload_rows_prefix_skipped": sum(
            row["non_e14_payload_rows_prefix_skipped"] for row in access
        ),
        "scientific_response_intervals_scored": sum(len(day) for day in y_days),
    }
    return data, accounting


def _one_column(days: Sequence[np.ndarray], scale: float) -> list[np.ndarray]:
    return [project_day(day / scale)[:, None] for day in days]


def _multi_column(columns: Sequence[Sequence[np.ndarray]], scales: Sequence[float]) -> list[np.ndarray]:
    return [
        project_day(np.column_stack([column[day] / scale for column, scale in zip(columns, scales, strict=True)]))
        for day in range(7)
    ]


def build_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    input_sha256: dict[str, str] = {}
    for binding in config["bindings"]:
        path = Path(binding["path"])
        _require(path.is_file(), f"missing binding: {binding['role']}")
        digest = file_sha256(path)
        _require(digest == binding["sha256"], f"binding changed: {binding['role']}")
        if "content_sha256" in binding:
            _require(_read_json(path)["content_sha256"] == binding["content_sha256"], "source content changed")
        input_sha256[binding["role"]] = digest
    data, access = _load_development(config)
    y_projected = [project_day(day) for day in data["y_raw"]]

    tau_results: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    scales: dict[float, float] = {}
    projected_x: dict[float, list[np.ndarray]] = {}
    for tau in config["physical_template"]["tau_seconds"]:
        scale = float(np.std(np.concatenate(data["x_raw"][tau])))
        _require(scale > 0.0, f"zero predictor scale at tau {tau}")
        scales[tau] = scale
        x_days = _one_column(data["x_raw"][tau], scale)
        projected_x[tau] = x_days
        cv = cross_validate(y_projected, x_days)
        signs = [int(np.sign(row["beta"][0] / scale)) for row in cv["folds"]]
        same_sign = max(signs.count(-1), signs.count(1))
        tau_result = {
            "tau_seconds": tau,
            "predictor_source_std": scale,
            "total_delta_log_likelihood": cv["total_delta_log_likelihood"],
            "same_nonzero_beta_sign_folds": same_sign,
            "fold_physical_betas": [row["beta"][0] / scale for row in cv["folds"]],
        }
        tau_results.append(tau_result)
        for row in cv["folds"]:
            fold_rows.append(
                {
                    "model": "memory",
                    "tau_seconds": tau,
                    "heldout_day": row["heldout_day"],
                    "delta_log_likelihood": row["delta_log_likelihood"],
                    "standardized_beta": row["beta"][0],
                    "physical_beta": row["beta"][0] / scale,
                    "null_rho": row["null_rho"],
                    "alternative_rho": row["alternative_rho"],
                    "null_innovation_sigma": row["null_innovation_sigma"],
                    "alternative_innovation_sigma": row["alternative_innovation_sigma"],
                }
            )
    selected = max(tau_results, key=lambda row: (row["total_delta_log_likelihood"], -row["tau_seconds"]))
    selected_tau = float(selected["tau_seconds"])

    phase_rows: list[dict[str, Any]] = []
    for shift in config["source"]["phase_scramble_shifts_samples"]:
        shifted_raw = [np.roll(day, shift) for day in data["x_raw"][selected_tau]]
        phase_cv = cross_validate(y_projected, _one_column(shifted_raw, scales[selected_tau]))
        phase_rows.append(
            {"shift_samples": shift, "total_delta_log_likelihood": phase_cv["total_delta_log_likelihood"]}
        )

    q_scale = float(np.std(np.concatenate(data["q_raw"])))
    dq_scale = float(np.std(np.concatenate(data["dq_raw"])))
    instantaneous_cv = cross_validate(y_projected, _one_column(data["q_raw"], q_scale))
    free_phase_days = _multi_column((data["q_raw"], data["dq_raw"]), (q_scale, dq_scale))
    free_phase_cv = cross_validate(y_projected, free_phase_days)
    combined_days = _multi_column(
        (data["q_raw"], data["dq_raw"], data["x_raw"][selected_tau]),
        (q_scale, dq_scale, scales[selected_tau]),
    )
    combined_cv = cross_validate(y_projected, combined_days)
    incremental_free_phase = (
        combined_cv["total_delta_log_likelihood"] - free_phase_cv["total_delta_log_likelihood"]
    )

    gates = {
        "positive_primary_score": selected["total_delta_log_likelihood"] > 0.0,
        "beta_sign_at_least_six_of_seven": selected["same_nonzero_beta_sign_folds"] >= 6,
        "beats_every_phase_scramble": selected["total_delta_log_likelihood"]
        > max(row["total_delta_log_likelihood"] for row in phase_rows),
        "positive_increment_beyond_free_orbital_phase": incremental_free_phase > 0.0,
    }
    interesting = all(gates.values())
    summary = {
        "daily": data["daily_summary"],
        "tau_scan": tau_results,
        "selected_tau_seconds": selected_tau,
        "selected_total_delta_log_likelihood": selected["total_delta_log_likelihood"],
        "selected_same_nonzero_beta_sign_folds": selected["same_nonzero_beta_sign_folds"],
        "maximum_phase_scramble_delta_log_likelihood": max(
            row["total_delta_log_likelihood"] for row in phase_rows
        ),
        "instantaneous_total_delta_log_likelihood": instantaneous_cv["total_delta_log_likelihood"],
        "free_orbital_phase_total_delta_log_likelihood": free_phase_cv["total_delta_log_likelihood"],
        "combined_memory_and_free_phase_total_delta_log_likelihood": combined_cv[
            "total_delta_log_likelihood"
        ],
        "memory_increment_beyond_free_orbital_phase": incremental_free_phase,
        "interesting_observation_gates": gates,
        "interesting_development_pattern": interesting,
        "E18_promotion": "BLOCKED_UNOPENED_CIRCULAR_ORBIT_ENVIRONMENT_CONTROLS",
        "unavailable_controls": [
            "E25_AND_E31_CLOCK_RESPONSES_NOT_AUTHORIZED",
            "E18_HOLDOUT_NOT_AUTHORIZED",
            "NO_SLR_OR_EPOCH_ORBIT_COVARIANCE",
            "NO_TEMPERATURE_MAGNETIC_ATTITUDE_CLOCK_IDENTITY_OR_SRP_SERIES",
            "NO_SEPARATE_FREQUENCY_PLASMA_RESPONSE",
        ],
    }
    diagnostics_path = ARTIFACT_DIR / "development-diagnostics.json"
    folds_path = ARTIFACT_DIR / "leave-one-day-out-folds.csv"
    phase_path = ARTIFACT_DIR / "phase-scramble-controls.csv"
    report_path = ARTIFACT_DIR / "development-report.md"
    _write_json(diagnostics_path, summary)
    _write_csv(folds_path, fold_rows)
    _write_csv(phase_path, phase_rows)
    report = (
        "# Frozen Galileo E14 causal-memory development result\n\n"
        f"Selected tau: {selected_tau:.0f} s. Conditional seven-fold heldout delta log likelihood: "
        f"{selected['total_delta_log_likelihood']:.9g}. Same-sign beta folds: "
        f"{selected['same_nonzero_beta_sign_folds']}/7.\n\n"
        f"Best phase-scramble score: {summary['maximum_phase_scramble_delta_log_likelihood']:.9g}. "
        f"Increment beyond the free orbital-phase countermodel: {incremental_free_phase:.9g}.\n\n"
        f"Development gate: {'PASS_PATTERN_ONLY' if interesting else 'FAIL'}. E18 remains unopened and blocked. "
        "This product cannot distinguish gravity from orbit, environment, plasma, or clock systematics because "
        "the required independent series were absent or deliberately unopened.\n"
    )
    _atomic_write(report_path, report.encode())
    artifacts = []
    for path in (diagnostics_path, folds_path, phase_path, report_path):
        artifacts.append({"path": path.as_posix(), "bytes": path.stat().st_size, "sha256": file_sha256(path)})
    receipt = {
        "schema": "invariant-open-gravity-galileo-e14-memory-response-receipt-1.0",
        "package_id": config["package_id"],
        "status": (
            "INTERESTING_E14_DEVELOPMENT_PATTERN_ONLY_E18_BLOCKED"
            if interesting
            else "E14_DEVELOPMENT_FAILED_FROZEN_INTERESTING_GATE_E18_BLOCKED"
        ),
        "decision": "DO_NOT_OPEN_E18_NO_RETUNING_RETAIN_ALL_COUNTEREXAMPLES",
        "input_sha256": input_sha256,
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "summary": summary,
        "artifacts": artifacts,
        "access_accounting": {
            **access,
            "network_calls": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "formula_retuning_events": 0,
        },
        "claim_boundary": {
            "real_E14_response_scored": True,
            "E18_response_opened": False,
            "circular_control_response_opened": False,
            "fully_predictive_day_ahead_likelihood": False,
            "gravity_or_time_well_established": False,
            "novelty_established": False,
            "publication_ready": False,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def build(output: Path = OUTPUT_PATH) -> dict[str, Any]:
    receipt = build_receipt(load_config())
    _write_json(output, receipt)
    return receipt


def check(output: Path = OUTPUT_PATH) -> dict[str, Any]:
    _require(output.is_file(), "receipt missing")
    expected = build_receipt(load_config())
    observed = _read_json(output)
    _require(observed == expected, "receipt differs from deterministic rebuild")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    receipt = build(args.output) if args.command == "build" else check(args.output)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
