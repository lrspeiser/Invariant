"""Executable dimensioned source drivers feeding frozen dynamic-memory kernels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_dynamic_source_memory_kernels_v1 as legacy

CONFIG_PATH = Path("configs/open_gravity_dynamic_source_memory_kernels_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_dynamic_source_memory_kernels_v2.py")
TEST_PATH = Path("tests/test_open_gravity_dynamic_source_memory_kernels_v2.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-dynamic-source-memory-kernels-v2/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
CONFIG_SCHEMA = "invariant-open-gravity-dynamic-source-memory-kernels-config-2.0"
RECEIPT_SCHEMA = "invariant-open-gravity-dynamic-source-memory-kernels-receipt-2.0"
DECISION = (
    "PASS_EXECUTABLE_DIMENSIONED_DRIVER_PIPELINES_STRUCTURAL_TRIAGE_"
    "SOURCE_BLOCKED_NO_RESPONSE_ACCESS"
)

DRIVER_IDS = (
    "D01_ACC",
    "D02_POT",
    "D03_RAD",
    "D04_RHO",
    "D05_SIG",
    "D06_SLOPE",
    "D07_TIDE",
    "D08_BAL",
    "D09_FLAT",
    "D10_MULT",
    "D11_EDGE",
    "D12_ENV",
    "D13_GASF",
    "D14_ION",
    "D15_COOL",
    "D16_NTH",
    "D17_AGE",
    "D18_RELAX",
    "D19_COH",
    "D20_EPOCH",
)
KERNEL_IDS = (
    "K01_RETARDED",
    "K02_EXPONENTIAL",
    "K03_BIEXPONENTIAL",
    "K04_DAMPED_RESONANCE",
    "K05_HYSTERETIC",
    "K06_STOCHASTIC_OU",
)
DIMENSIONLESS = (0, 0, 0)


class DynamicMemoryV2Error(RuntimeError):
    """Raised when a driver, pipeline, source gate, or seal fails."""


@dataclass(frozen=True)
class TypedValue:
    value: float | np.ndarray
    type: str
    unit: tuple[int, int, int]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DynamicMemoryV2Error(message)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _array_sha256(value: np.ndarray) -> str:
    return _sha256_bytes(np.asarray(value, dtype="<f8").tobytes())


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical(body))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicMemoryV2Error(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _unit(raw: Sequence[int]) -> tuple[int, int, int]:
    _require(len(raw) == 3 and all(isinstance(item, int) for item in raw), "invalid L,M,T unit")
    return (raw[0], raw[1], raw[2])


def _unit_add(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(a + b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _unit_sub(left: tuple[int, int, int], right: tuple[int, int, int]) -> tuple[int, int, int]:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _fixture_times(config: Mapping[str, Any]) -> np.ndarray:
    fixture = config["fixture"]
    samples = int(fixture["samples"])
    dt = float(fixture["dt"])
    _require(samples >= 5 and dt > 0.0, "invalid driver fixture grid")
    return np.arange(samples, dtype=float) * dt


def _fixture_value(declaration: Mapping[str, Any], times: np.ndarray) -> float | np.ndarray:
    fixture = declaration["fixture"]
    kind = fixture["kind"]
    if kind == "constant":
        return float(fixture["value"])
    frequency = float(fixture["frequency"])
    phase = float(fixture["phase"])
    amplitude = float(fixture["amplitude"])
    harmonic = float(fixture.get("harmonic", 0.0))
    if kind == "positive_log_wave":
        return float(fixture["scale"]) * np.exp(
            float(fixture["bias"])
            + amplitude * np.sin(frequency * times + phase)
            + harmonic * np.cos(2.0 * frequency * times - phase)
        )
    if kind == "positive_exp_ramp":
        return float(fixture["scale"]) * np.exp(
            float(fixture["bias"])
            + float(fixture["slope"]) * times
            + amplitude * np.sin(frequency * times + phase)
        )
    if kind == "signed_wave":
        return float(fixture["scale"]) * (
            float(fixture["bias"])
            + amplitude * np.sin(frequency * times + phase)
            + harmonic * np.cos(2.0 * frequency * times - phase)
        )
    if kind in {"bounded_fraction_wave", "bounded_nonnegative_wave"}:
        return (
            float(fixture["bias"])
            + amplitude * np.sin(frequency * times + phase)
            + harmonic * np.cos(2.0 * frequency * times - phase)
        )
    raise DynamicMemoryV2Error(f"unknown fixture generator: {kind}")


def _domain_ok(value: float | np.ndarray, domain: str) -> bool:
    values = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(values)):
        return False
    if domain == "real":
        return True
    if domain == "positive":
        return bool(np.all(values > 0.0))
    if domain == "nonnegative":
        return bool(np.all(values >= 0.0))
    if domain == "unit_interval":
        return bool(np.all((values >= 0.0) & (values <= 1.0)))
    raise DynamicMemoryV2Error(f"unknown domain: {domain}")


def _collect_vars(node: Any) -> set[str]:
    if isinstance(node, list):
        return set().union(*(_collect_vars(item) for item in node)) if node else set()
    if not isinstance(node, dict):
        return set()
    found = {str(node["name"])} if node.get("op") == "var" else set()
    for value in node.values():
        found.update(_collect_vars(value))
    return found


def evaluate_driver_ast(
    node: Mapping[str, Any], environment: Mapping[str, TypedValue]
) -> TypedValue:
    op = node.get("op")
    if op == "var":
        name = str(node["name"])
        _require(name in environment, f"undeclared driver variable: {name}")
        return environment[name]
    if op in {"abs", "neg"}:
        argument = evaluate_driver_ast(node["arg"], environment)
        function = np.abs if op == "abs" else np.negative
        result = function(argument.value)
        return TypedValue(result, argument.type, argument.unit)
    if op == "div":
        numerator = evaluate_driver_ast(node["num"], environment)
        denominator = evaluate_driver_ast(node["den"], environment)
        _require(not np.any(np.asarray(denominator.value) == 0.0), "driver division by zero")
        result = np.asarray(numerator.value) / np.asarray(denominator.value)
        value: float | np.ndarray = float(result) if result.ndim == 0 else result
        result_type = "vector" if isinstance(value, np.ndarray) else "scalar"
        return TypedValue(value, result_type, _unit_sub(numerator.unit, denominator.unit))
    if op in {"tanh", "log"}:
        argument = evaluate_driver_ast(node["arg"], environment)
        _require(argument.unit == DIMENSIONLESS, f"{op} requires dimensionless input")
        if op == "log":
            _require(np.all(np.asarray(argument.value) > 0.0), "log requires positive input")
            result = np.log(argument.value)
        else:
            result = np.tanh(argument.value)
        return TypedValue(result, argument.type, DIMENSIONLESS)
    if op == "gradient":
        y_value = evaluate_driver_ast(node["y"], environment)
        x_value = evaluate_driver_ast(node["x"], environment)
        _require(
            y_value.type == x_value.type == "vector"
            and y_value.unit == x_value.unit == DIMENSIONLESS,
            "gradient requires dimensionless vector coordinates",
        )
        y_array = np.asarray(y_value.value, dtype=float)
        x_array = np.asarray(x_value.value, dtype=float)
        _require(y_array.shape == x_array.shape and y_array.size >= 3, "gradient shape mismatch")
        _require(np.all(np.diff(x_array) != 0.0), "gradient coordinate repeats")
        return TypedValue(np.gradient(y_array, x_array, edge_order=2), "vector", DIMENSIONLESS)
    raise DynamicMemoryV2Error(f"unknown driver AST operation: {op}")


def execute_driver(config: Mapping[str, Any], driver: Mapping[str, Any]) -> dict[str, Any]:
    times = _fixture_times(config)
    declarations = driver.get("variables")
    _require(
        isinstance(declarations, list) and declarations, f"missing variables: {driver.get('id')}"
    )
    names = [str(row["name"]) for row in declarations]
    _require(len(names) == len(set(names)), f"duplicate variables: {driver.get('id')}")
    environment: dict[str, TypedValue] = {}
    input_rows = []
    for declaration in declarations:
        _require(
            set(declaration) == {"name", "role", "type", "unit", "domain", "fixture"},
            f"driver declaration changed: {driver.get('id')}",
        )
        role = str(declaration["role"])
        _require(
            "response" not in role.lower(), f"response-derived driver role: {driver.get('id')}"
        )
        value = _fixture_value(declaration, times)
        expected_type = str(declaration["type"])
        observed_type = "vector" if isinstance(value, np.ndarray) else "scalar"
        _require(observed_type == expected_type, f"driver type mismatch: {declaration['name']}")
        if observed_type == "vector":
            _require(value.shape == times.shape, f"driver shape mismatch: {declaration['name']}")
        _require(
            _domain_ok(value, str(declaration["domain"])),
            f"driver domain violation: {declaration['name']}",
        )
        typed = TypedValue(value, expected_type, _unit(declaration["unit"]))
        environment[str(declaration["name"])] = typed
        input_rows.append(
            {
                "name": declaration["name"],
                "role": role,
                "type": expected_type,
                "unit": declaration["unit"],
                "value_sha256": _array_sha256(np.atleast_1d(value)),
            }
        )
    used = _collect_vars(driver["program"])
    _require(used == set(names), f"driver variables not exactly used: {driver.get('id')}")
    result = evaluate_driver_ast(driver["program"], environment)
    _require(
        result.type == "vector" and result.unit == DIMENSIONLESS,
        "driver output not dimensionless vector",
    )
    output = np.asarray(result.value, dtype=float)
    _require(output.shape == times.shape and np.all(np.isfinite(output)), "invalid driver output")
    return {
        "driver_id": driver["id"],
        "description": driver["description"],
        "program_sha256": _sha256_bytes(_canonical(driver["program"])),
        "declared_variable_count": len(names),
        "declared_variables_exactly_used": True,
        "output_unit": [0, 0, 0],
        "output_sha256": _array_sha256(output),
        "output_min": float(np.min(output)),
        "output_max": float(np.max(output)),
        "output_rms": float(np.sqrt(np.mean(output**2))),
        "inputs": input_rows,
        "times": times,
        "output": output,
    }


def _legacy_config(base: Path) -> dict[str, Any]:
    try:
        return legacy.load_config(base)
    except legacy.DynamicMemoryError as exc:
        raise DynamicMemoryV2Error(f"v1 predecessor validation failed: {exc}") from exc


def execute_pipelines(
    config: Mapping[str, Any], legacy_config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kernel_map = legacy._kernel_map(legacy_config)
    driver_executions = [execute_driver(config, driver) for driver in config["drivers"]]
    _require(
        len({row["output_sha256"] for row in driver_executions}) == len(DRIVER_IDS),
        "driver fixtures do not produce unique executed inputs",
    )
    rows: list[dict[str, Any]] = []
    for driver in driver_executions:
        source = driver["output"]
        times = driver["times"]
        for kernel_id in KERNEL_IDS:
            response = legacy.simulate_kernel(
                kernel_id, times, source, kernel_map[kernel_id]["parameters"]
            )
            input_sha = _array_sha256(source)
            _require(input_sha == driver["output_sha256"], "kernel input bypassed driver output")
            program_identity = {
                "driver_program_sha256": driver["program_sha256"],
                "kernel_id": kernel_id,
                "kernel_parameters": kernel_map[kernel_id]["parameters"],
            }
            rows.append(
                {
                    "concept_id": f"{driver['driver_id']}::{kernel_id}",
                    "driver_id": driver["driver_id"],
                    "kernel_id": kernel_id,
                    "driver_program_sha256": driver["program_sha256"],
                    "driver_output_sha256": driver["output_sha256"],
                    "kernel_input_sha256": input_sha,
                    "pipeline_program_sha256": _sha256_bytes(_canonical(program_identity)),
                    "response_sha256": _array_sha256(response),
                    "response_rms": float(np.sqrt(np.mean(response**2))),
                    "status": "EXECUTED_DRIVER_OUTPUT_AS_KERNEL_INPUT",
                }
            )
    _require(len(rows) == 120, "driver-kernel execution count changed")
    _require(
        len({row["pipeline_program_sha256"] for row in rows}) == 120,
        "pipeline program identities collide",
    )
    return driver_executions, rows


def _structural_triage(
    config: Mapping[str, Any], legacy_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    tolerances = config["structural_triage"]["feature_tolerances"]
    kernel_map = legacy._kernel_map(legacy_config)
    step = legacy._step_signatures(legacy_config)
    _, rate = legacy._rate_sweep(legacy_config)
    energy = legacy._energy_ledger(legacy_config, rate)
    ou = legacy._ou_signature(legacy_config)
    omega = np.asarray([0.25, 0.5, 1.0, 2.0, 4.0, 7.0], dtype=float)
    rows: list[dict[str, Any]] = []
    mean_fingerprints: dict[str, str] = {}
    frequency_features: dict[str, dict[str, bool]] = {}
    for kernel_id in KERNEL_IDS:
        if kernel_id == "K05_HYSTERETIC":
            times, source = legacy._step_fixture(legacy_config)
            response = legacy.simulate_kernel(
                kernel_id, times, source, kernel_map[kernel_id]["parameters"]
            )
            mean_fingerprints[kernel_id] = _array_sha256(response)
            frequency_features[kernel_id] = {
                "attenuation": False,
                "phase_curvature": False,
                "gain_above_unity": False,
            }
            continue
        response = legacy.transfer_response(kernel_id, omega, kernel_map[kernel_id]["parameters"])
        packed = np.column_stack((response.real, response.imag)).astype("<f8")
        mean_fingerprints[kernel_id] = _sha256_bytes(packed.tobytes())
        magnitude = np.abs(response)
        phase = np.unwrap(np.angle(response))
        line = np.polyfit(omega, phase, 1)
        phase_residual = float(np.max(np.abs(phase - np.polyval(line, omega))))
        frequency_features[kernel_id] = {
            "attenuation": bool(np.any(magnitude < 1.0 - tolerances["magnitude"])),
            "phase_curvature": phase_residual > tolerances["phase"],
            "gain_above_unity": bool(np.any(magnitude > 1.0 + tolerances["magnitude"])),
        }
    class_sizes = {
        fingerprint: sum(value == fingerprint for value in mean_fingerprints.values())
        for fingerprint in set(mean_fingerprints.values())
    }
    for kernel_id in KERNEL_IDS:
        features = dict(frequency_features[kernel_id])
        features.update(
            {
                "step_overshoot": step[kernel_id]["overshoot_above_unit"] > tolerances["overshoot"],
                "post_source_tail": step[kernel_id]["post_off_absolute_area_over_source_area"]
                > tolerances["tail"],
                "quasistatic_loop_persistence": abs(rate[kernel_id]["linear_zero_rate_intercept"])
                > tolerances["quasistatic_intercept"],
                "conditional_variance": kernel_id == "K06_STOCHASTIC_OU"
                and ou["analytic_stationary_variance"] > tolerances["variance"],
            }
        )
        channel_count = sum(bool(value) for value in features.values())
        single_event_identifiable = kernel_id != "K01_RETARDED"
        completion = str(energy[kernel_id]["completion_status"]).startswith("NORMALIZED_")
        fingerprint = mean_fingerprints[kernel_id]
        sort_key = [
            -int(single_event_identifiable),
            -channel_count,
            class_sizes[fingerprint],
            -int(completion),
            kernel_id,
        ]
        rows.append(
            {
                "kernel_id": kernel_id,
                "single_event_identifiable": single_event_identifiable,
                "computed_discriminator_channels": features,
                "computed_discriminator_channel_count": channel_count,
                "conditional_mean_fingerprint_sha256": fingerprint,
                "conditional_mean_equivalence_class_size": class_sizes[fingerprint],
                "normalized_energy_ledger_closed": completion,
                "computed_sort_key": sort_key,
                "empirical_rank": None,
            }
        )
    ordered = sorted(rows, key=lambda row: tuple(row["computed_sort_key"]))
    for index, row in enumerate(ordered, start=1):
        row["structural_order"] = index
    return ordered


def _countermodel_executions(
    config: Mapping[str, Any], legacy_config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    times, source = legacy._step_fixture(legacy_config)
    kernel_map = legacy._kernel_map(legacy_config)
    rows = []
    for countermodel in config["countermodels"]:
        if countermodel["kind"] == "kernel":
            kernel_id = countermodel["kernel_id"]
            response = legacy.simulate_kernel(
                kernel_id, times, source, kernel_map[kernel_id]["parameters"]
            )
            input_sha = _array_sha256(source)
            program_sha = _sha256_bytes(
                _canonical(
                    {
                        "kernel_id": kernel_id,
                        "parameters": kernel_map[kernel_id]["parameters"],
                    }
                )
            )
        else:
            parameters = countermodel["parameters"]
            onset = float(parameters["onset"])
            mask = times >= onset
            ringdown = np.zeros_like(source)
            elapsed = times[mask] - onset
            ringdown[mask] = (
                float(parameters["amplitude"])
                * np.exp(-elapsed / float(parameters["decay_time"]))
                * np.sin(float(parameters["omega"]) * elapsed)
            )
            response = source + ringdown
            input_sha = _array_sha256(source)
            program_sha = _sha256_bytes(_canonical(countermodel["parameters"]))
        rows.append(
            {
                "id": countermodel["id"],
                "kind": countermodel["kind"],
                "kernel_id": countermodel["kernel_id"],
                "role": countermodel["role"],
                "input_sha256": input_sha,
                "program_sha256": program_sha,
                "response_sha256": _array_sha256(response),
                "post_source_rms": float(np.sqrt(np.mean(response[times >= 6.0] ** 2))),
                "status": "EXECUTED_TARGET_FREE_COUNTERMODEL",
            }
        )
    return rows


def validate_config(config: Mapping[str, Any], base: Path | None = None) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-dynamic-source-memory-kernels-v2",
        "analysis ID changed",
    )
    _require(
        config.get("status")
        == "FROZEN_EXECUTABLE_DIMENSIONED_DRIVERS_STRUCTURAL_TRIAGE_SOURCE_BLOCKED",
        "status changed",
    )
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
            "artifact_directory": ARTIFACT_DIR.as_posix(),
        },
        "package paths changed",
    )
    predecessor = config.get("predecessor")
    _require(isinstance(predecessor, dict), "predecessor missing")
    if base is not None:
        for path_key, hash_key in (
            ("receipt_path", "receipt_raw_sha256"),
            ("config_path", "config_raw_sha256"),
            ("module_path", "module_raw_sha256"),
            ("test_path", "test_raw_sha256"),
            ("source_path", "source_sha256"),
        ):
            path = base / predecessor[path_key]
            _require(path.is_file(), f"predecessor file missing: {path}")
            _require(_sha256_file(path) == predecessor[hash_key], f"predecessor changed: {path}")
        receipt = _read_json(base / predecessor["receipt_path"])
        _require(
            receipt.get("content_sha256") == predecessor["receipt_content_sha256"],
            "predecessor content hash changed",
        )
    unit_contract = config.get("unit_contract")
    _require(
        isinstance(unit_contract, dict)
        and unit_contract.get("basis") == ["L", "M", "T"]
        and unit_contract.get("all_declared_inputs_must_be_used") is True
        and unit_contract.get("response_derived_inputs_forbidden") is True
        and unit_contract.get("driver_output_must_equal_kernel_input_bytes") is True,
        "unit/driver contract changed",
    )
    _require(tuple(config.get("kernel_ids", [])) == KERNEL_IDS, "kernel inventory changed")
    drivers = config.get("drivers")
    _require(
        isinstance(drivers, list) and tuple(row.get("id") for row in drivers) == DRIVER_IDS,
        "driver inventory changed",
    )
    executions = [execute_driver(config, driver) for driver in drivers]
    _require(len({row["output_sha256"] for row in executions}) == 20, "driver outputs collide")
    _require(
        [row.get("id") for row in config.get("countermodels", [])]
        == [
            "C01_FREE_DELAY",
            "C02_SINGLE_LTI",
            "C03_DOUBLE_LTI",
            "C04_SOURCE_RINGDOWN",
            "C05_OU_NOISE",
        ],
        "countermodel inventory changed",
    )
    triage = config.get("structural_triage")
    _require(
        isinstance(triage, dict)
        and triage.get("kind") == "COMPUTED_LEXICOGRAPHIC_NOT_EMPIRICAL_RANKING"
        and triage.get("future_metrics_must_not_affect_current_order") is True,
        "structural triage changed",
    )
    heldout = config.get("future_heldout_metrics")
    _require(
        isinstance(heldout, list)
        and len(heldout) == 6
        and all(row.get("status") == "NOT_EVALUATED_RESPONSE_UNOPENED" for row in heldout),
        "future metric status changed",
    )
    preflight = config.get("observational_preflight")
    _require(
        isinstance(preflight, dict)
        and preflight.get("status") == "SOURCE_BLOCKED_MISSING_PAYLOAD_HASHES_AND_SCHEMA_RECEIPTS",
        "response source gate changed",
    )
    products = preflight.get("products")
    _require(
        isinstance(products, list)
        and [row.get("detector") for row in products] == ["H1", "L1"]
        and all(row.get("resolved_url") == row.get("requested_url") for row in products)
        and all(row.get("http_status") == 200 for row in products)
        and all(row.get("payload_sha256") is None for row in products),
        "GWOSC product gate changed",
    )
    _require(
        preflight.get("metadata_receipt")
        == {
            "method": "HEAD_ONLY",
            "requests": 2,
            "response_body_bytes": 0,
            "observed_http_date": "Mon, 31 Aug 2026 20:08:33 GMT",
        },
        "metadata-only receipt changed",
    )
    _require(
        set(preflight.get("preprocessing", {}))
        == {
            "analysis_interval_gps",
            "psd_intervals_gps",
            "dq_rule",
            "detrend",
            "analysis_window",
            "psd",
            "band_hz",
            "fft",
            "resampling",
        }
        and set(preflight.get("likelihood", {}))
        == {
            "kind",
            "formula",
            "detectors",
            "kernel_scale",
            "template",
            "nuisance_grid",
            "optimizer",
            "validation_suite",
            "decision_tolerance",
        }
        and preflight.get("missing"),
        "preprocessing/likelihood freeze incomplete",
    )
    access = config.get("access_contract")
    _require(
        access
        == {
            "metadata_head_requests": 2,
            "response_body_bytes": 0,
            "observational_response_files_opened": 0,
            "observational_response_rows_read": 0,
            "real_response_scores": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "access contract changed",
    )
    boundary = config.get("claim_boundary")
    _require(
        isinstance(boundary, dict)
        and boundary.get("executed_driver_kernel_pipelines") == 120
        and boundary.get("unique_empirical_theories_claimed") is False
        and boundary.get("structural_triage_only") is True
        and boundary.get("observational_source_ready") is False
        and boundary.get("real_response_fit") is False
        and boundary.get("publication_ready") is False
        and boundary.get("strict_reaudit_ready") is True,
        "claim boundary widened",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = _read_json(base / CONFIG_PATH)
    validate_config(config, base)
    return config


def _csv_bytes(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _report(
    drivers: Sequence[Mapping[str, Any]],
    pipelines: Sequence[Mapping[str, Any]],
    triage: Sequence[Mapping[str, Any]],
) -> str:
    structural_order = ", ".join(str(row["kernel_id"]) for row in triage)
    return f"""# Dynamic source-memory kernels v2 strict repair

## Repaired result

All {len(drivers)} source drivers are executable typed programs.  Every driver declares source variables and L/M/T dimensions, produces a dimensionless time series, and has that exact byte sequence passed into each of six frozen predecessor kernels.  The {len(pipelines)} rows are executed pipelines, not 120 claims of empirically distinct theories.  The v1 label-times-kernel count is retained as the motivating counterexample and is not accepted as evidence.

## Structural triage, not hand ranking

The target-free structural order is `{structural_order}`.  It is recomputed from frozen transfer, step, tail, quasistatic-loop, variance, equivalence-class, and normalized-energy features using the declared lexicographic rule.  `empirical_rank` is null for every row.  Six held-out metrics are frozen but remain `NOT_EVALUATED_RESPONSE_UNOPENED`.

## Countermodels retained

Pure delay, one-pole LTI, two-pole LTI, an explicit source-ringdown program, and OU noise are all executed and retained.  A K04 tail is therefore not labeled gravity memory merely because it rings, and a K06 mean is not separated from K02 without its variance channel.

## GW150914 source gate

Two metadata-only HEAD requests resolved the exact H1/L1 URLs and confirmed byte counts, ETags, and last-modified headers while reading zero response-body bytes.  Payload SHA-256 values and verified HDF5/calibration/waveform receipts cannot be obtained without payload access, so the source is `SOURCE_BLOCKED`.  The event interval, PSD construction, band, FFT, relative Whittle likelihood, nuisance boundary, and no-per-event kernel scale are frozen for a future authorized acquisition.

## Claim boundary

This is a response-blind executable methods repair.  It reports no observational fit, empirical rank, historical novelty, gravity effect, or publication-ready physics result.
"""


def artifact_payloads(
    config: Mapping[str, Any], base: Path
) -> tuple[dict[str, bytes], dict[str, Any]]:
    legacy_config = _legacy_config(base)
    driver_executions, pipelines = execute_pipelines(config, legacy_config)
    triage = _structural_triage(config, legacy_config)
    countermodels = _countermodel_executions(config, legacy_config)
    clean_drivers = [
        {key: value for key, value in row.items() if key not in {"times", "output"}}
        for row in driver_executions
    ]
    driver_programs = {
        "schema_version": "invariant-dimensioned-source-driver-programs-2.0",
        "unit_contract": config["unit_contract"],
        "drivers": config["drivers"],
        "executions": clean_drivers,
    }
    triage_object = {
        "schema_version": "invariant-dynamic-memory-structural-triage-2.0",
        "algorithm": config["structural_triage"],
        "rows": triage,
        "empirical_ranks_computed": 0,
    }
    countermodel_object = {
        "schema_version": "invariant-dynamic-memory-countermodels-2.0",
        "definitions": config["countermodels"],
        "executions": countermodels,
    }
    future_object = {
        "schema_version": "invariant-dynamic-memory-future-heldout-metrics-2.0",
        "metrics": config["future_heldout_metrics"],
        "evaluated_metrics": 0,
    }
    source_object = {
        "schema_version": "invariant-dynamic-memory-response-source-gate-2.0",
        "preflight": config["observational_preflight"],
        "access_contract": config["access_contract"],
    }
    pipeline_rows = [
        [
            row["concept_id"],
            row["driver_id"],
            row["kernel_id"],
            row["driver_program_sha256"],
            row["driver_output_sha256"],
            row["kernel_input_sha256"],
            row["pipeline_program_sha256"],
            row["response_sha256"],
            row["response_rms"],
            row["status"],
        ]
        for row in pipelines
    ]
    triage_rows = [
        [
            row["structural_order"],
            row["kernel_id"],
            row["single_event_identifiable"],
            row["computed_discriminator_channel_count"],
            row["conditional_mean_equivalence_class_size"],
            row["normalized_energy_ledger_closed"],
            json.dumps(row["computed_sort_key"], separators=(",", ":")),
            "",
        ]
        for row in triage
    ]
    payloads = {
        "driver-programs-and-executions.json": _canonical(driver_programs) + b"\n",
        "executed-driver-kernel-pipelines.csv": _csv_bytes(
            [
                "concept_id",
                "driver_id",
                "kernel_id",
                "driver_program_sha256",
                "driver_output_sha256",
                "kernel_input_sha256",
                "pipeline_program_sha256",
                "response_sha256",
                "response_rms",
                "status",
            ],
            pipeline_rows,
        ),
        "computed-structural-triage.json": _canonical(triage_object) + b"\n",
        "computed-structural-triage.csv": _csv_bytes(
            [
                "structural_order",
                "kernel_id",
                "single_event_identifiable",
                "discriminator_channel_count",
                "mean_equivalence_class_size",
                "normalized_energy_closed",
                "computed_sort_key",
                "empirical_rank",
            ],
            triage_rows,
        ),
        "executed-countermodels.json": _canonical(countermodel_object) + b"\n",
        "future-heldout-metrics.json": _canonical(future_object) + b"\n",
        "gw150914-response-source-gate.json": _canonical(source_object) + b"\n",
        "report.md": _report(clean_drivers, pipelines, triage).encode("utf-8"),
    }
    summary = {
        "driver_executions": clean_drivers,
        "pipelines": pipelines,
        "triage": triage,
        "countermodels": countermodels,
    }
    return payloads, summary


def _package_hashes(base: Path) -> dict[str, str]:
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(base / MODULE_PATH),
        "test_raw_sha256": _sha256_file(base / TEST_PATH),
    }


def build_receipt(config: Mapping[str, Any], base: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    payloads, summary = artifact_payloads(config, base)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "decision": DECISION,
        "content_sha256": "",
        "predecessor": config["predecessor"],
        "package_hashes": _package_hashes(base),
        "config_content_sha256": _sha256_bytes(_canonical(config)),
        "artifact_sha256": {
            name: _sha256_bytes(payload) for name, payload in sorted(payloads.items())
        },
        "counts": {
            "executable_dimensioned_drivers": len(summary["driver_executions"]),
            "executed_driver_kernel_pipelines": len(summary["pipelines"]),
            "unique_empirical_theories_claimed": 0,
            "computed_structural_triage_rows": len(summary["triage"]),
            "empirical_ranks": 0,
            "future_heldout_metrics_evaluated": 0,
            "executed_countermodels": len(summary["countermodels"]),
            "metadata_head_requests": 2,
            "response_body_bytes": 0,
            "observational_response_rows": 0,
        },
        "internal_checks": {
            "all_driver_variables_typed_and_dimensioned": True,
            "all_declared_driver_variables_exactly_used": True,
            "driver_output_equals_kernel_input_hash": True,
            "all_120_pipelines_executed": True,
            "pipeline_program_identities_unique": True,
            "v1_label_cross_product_not_accepted_as_execution": True,
            "structural_triage_computed_not_hand_ranked": True,
            "delay_LTI_source_ringdown_OU_countermodels_retained": True,
            "GW150914_source_blocked_without_payload_hashes": True,
        },
        "structural_order": [row["kernel_id"] for row in summary["triage"]],
        "observational_preflight_status": config["observational_preflight"]["status"],
        "access_ledger": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, payloads


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def build(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    receipt, payloads = build_receipt(config, base)
    targets = {base / ARTIFACT_DIR / name: payload for name, payload in payloads.items()}
    targets[base / OUTPUT_PATH] = _canonical(receipt) + b"\n"
    existing = [path for path in targets if path.exists()]
    if existing:
        _require(len(existing) == len(targets), "partial v2 output package exists")
        for path, payload in targets.items():
            _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    for path, payload in targets.items():
        _atomic_write(path, payload)
    return "CREATED"


def check(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    expected, payloads = build_receipt(config, base)
    observed = _read_json(base / OUTPUT_PATH)
    _require(observed.get("content_sha256") == _self_hash(observed), "receipt self-hash invalid")
    _require(observed == expected, "receipt differs from deterministic rebuild")
    for name, payload in payloads.items():
        path = base / ARTIFACT_DIR / name
        _require(path.is_file(), f"missing artifact: {name}")
        _require(path.read_bytes() == payload, f"artifact differs: {name}")
    return "VALID"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "build":
        print(build())
        return 0
    if args.command == "check":
        print(check())
        return 0
    config = load_config()
    print(
        json.dumps(
            {
                "analysis_id": config["analysis_id"],
                "status": config["status"],
                "drivers": len(config["drivers"]),
                "pipelines": len(config["drivers"]) * len(config["kernel_ids"]),
                "observational_rows": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
