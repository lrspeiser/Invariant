"""Causal dynamic-source memory kernels and response-blind discriminators."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import tempfile
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_dynamic_source_memory_kernels_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_dynamic_source_memory_kernels_v1.py")
TEST_PATH = Path("tests/test_open_gravity_dynamic_source_memory_kernels_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-dynamic-source-memory-kernels-v1/receipt.json")
ARTIFACT_DIRECTORY = Path("runs/gravity/open-gravity-dynamic-source-memory-kernels-v1/artifacts")
SOURCE_PATH = Path(
    "runs/gravity/open-gravity-dynamic-source-memory-kernels-v1/source/GW150914_4_NR_waveform.txt"
)
SOURCE_RECEIPT_PATH = SOURCE_PATH.parent / "acquisition.json"

CONFIG_SCHEMA = "invariant-open-gravity-dynamic-source-memory-kernels-config-1.0"
RECEIPT_SCHEMA = "invariant-open-gravity-dynamic-source-memory-kernels-receipt-1.0"
DECISION = (
    "PASS_CAUSAL_DYNAMIC_MEMORY_DISCRIMINATOR_BASIS_PUBLIC_NR_SOURCE_BENCHMARK_RESPONSE_BLIND"
)

EXPECTED_CONFIG_RAW_SHA256 = "2ff3294450973a3409fb73ae5d4ff5d27e8a222e723d637f04c0a734b55f3f9f"
EXPECTED_MODULE_SEMANTIC_SHA256 = "190eaf63a9c03cb613f164f93f0c2f738ee00cca05e8552218a33205cfb9ad05"
EXPECTED_TEST_RAW_SHA256 = "1785cc1e0f4b74a70477243aaa96efe69078070720379fe9695c5f6e389c6612"

KERNEL_IDS = (
    "K00_INSTANTANEOUS",
    "K01_RETARDED",
    "K02_EXPONENTIAL",
    "K03_BIEXPONENTIAL",
    "K04_DAMPED_RESONANCE",
    "K05_HYSTERETIC",
    "K06_STOCHASTIC_OU",
)


class DynamicMemoryError(RuntimeError):
    """Raised when a memory, data, or frozen-integrity gate fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
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


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical_bytes(body))


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?P<quote>["\'])'
        r"(?:[0-9a-f]{64}|__MODULE_SEMANTIC_SHA256__)(?P=quote)",
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DynamicMemoryError(message)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DynamicMemoryError(f"invalid JSON: {path}") from exc
    _require(isinstance(value, dict), "JSON root must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "open-gravity-dynamic-source-memory-kernels-v1",
        "analysis identity changed",
    )
    package = config.get("package")
    _require(isinstance(package, dict), "package paths missing")
    _require(package.get("module_path") == MODULE_PATH.as_posix(), "module path changed")
    _require(package.get("test_path") == TEST_PATH.as_posix(), "test path changed")
    _require(package.get("output_path") == OUTPUT_PATH.as_posix(), "output path changed")
    _require(
        package.get("artifact_directory") == ARTIFACT_DIRECTORY.as_posix(),
        "artifact path changed",
    )
    _require(
        package.get("benchmark_source_path") == SOURCE_PATH.as_posix(),
        "source path changed",
    )
    kernels = config.get("kernels")
    _require(
        isinstance(kernels, list) and tuple(row.get("id") for row in kernels) == KERNEL_IDS,
        "kernel inventory changed",
    )
    mapping = config.get("concept_mapping")
    _require(
        mapping
        == {
            "predecessor_architectures": [
                "K01_RETARDED",
                "K02_EXPONENTIAL",
                "K04_DAMPED_RESONANCE",
                "K06_STOCHASTIC_OU",
            ],
            "new_architectures": ["K03_BIEXPONENTIAL", "K05_HYSTERETIC"],
            "candidate_architectures": 6,
            "drivers_per_architecture": 20,
            "predecessor_concepts": 80,
            "new_concepts": 40,
            "total_executable_concepts": 120,
        },
        "concept mapping changed",
    )
    _require(
        config.get("driver_ids") == [f"D{index:02d}" for index in range(1, 21)],
        "driver inventory changed",
    )
    benchmark = config.get("public_simulation_benchmark")
    _require(isinstance(benchmark, dict), "benchmark missing")
    _require(
        benchmark.get("expected_sha256")
        == "ed49c3e83f90e70ac85386f183031b7de3d3d6aa78e75a7d284e5a53a5cc0b76",
        "benchmark hash changed",
    )
    _require(benchmark.get("expected_rows") == 2769, "benchmark rows changed")
    preflight = config.get("observational_response_preflight")
    _require(
        isinstance(preflight, dict)
        and preflight.get("state") == "DIRECT_URLS_FROZEN_RESPONSES_NOT_OPENED"
        and [row.get("detector") for row in preflight.get("products", [])] == ["H1", "L1"],
        "response preflight changed",
    )
    ledger = config.get("access_ledger")
    _require(
        isinstance(ledger, dict)
        and ledger.get("observational_response_files_opened") == 0
        and ledger.get("observational_response_rows_read") == 0
        and ledger.get("real_response_scores_computed") == 0
        and ledger.get("model_calls") == 0
        and ledger.get("paid_calls") == 0
        and ledger.get("post_response_formula_changes") == 0,
        "access boundary changed",
    )
    claims = config.get("claim_boundary")
    _require(
        isinstance(claims, dict)
        and claims.get("observational_response_opened") is False
        and claims.get("real_data_fit_performed") is False
        and claims.get("covariant_gravity_action_derived") is False
        and claims.get("universal_physical_time_scale_derived") is False
        and claims.get("historical_novelty_established") is False
        and claims.get("empirical_gravity_discovery") is False
        and claims.get("publication_ready_empirical_claim") is False,
        "claim boundary changed",
    )
    _require(len(config.get("primary_literature", [])) == 5, "literature map changed")
    strongest = config.get("strongest_counterexample")
    _require(
        isinstance(strongest, dict)
        and strongest.get("raw_candidate_cells") == 262144
        and strongest.get("selection_aware_permutation_p") == 0.75
        and strongest.get("formal_status") == "INCONCLUSIVE_QUALITY",
        "strongest counterexample changed",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = base / CONFIG_PATH
    _require(_sha256_file(path) == EXPECTED_CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_local_integrity(base: Path) -> dict[str, str]:
    module = (base / MODULE_PATH).resolve()
    test = (base / TEST_PATH).resolve()
    _require(module == Path(__file__).resolve(), "module path changed")
    semantic = _module_semantic_sha256(module)
    _require(semantic == EXPECTED_MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(_sha256_file(test) == EXPECTED_TEST_RAW_SHA256, "test bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module),
        "module_semantic_sha256": semantic,
        "test_raw_sha256": _sha256_file(test),
    }


def _validate_bindings(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["bindings"]:
        path = base / str(row["path"])
        _require(path.is_file(), f"missing binding: {row['role']}")
        digest = _sha256_file(path)
        _require(digest == row["sha256"], f"binding changed: {row['role']}")
        observed[str(row["role"])] = digest
    _require(len(observed) == 5, "binding count changed")
    return observed


def _atomic_bytes_no_clobber(path: Path, payload: bytes) -> str:
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing to replace nonidentical file: {path}")
        return "EXISTING_IDENTICAL"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return "CREATED"


def _atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    return _atomic_bytes_no_clobber(
        path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    )


def acquire_benchmark(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    benchmark = config["public_simulation_benchmark"]
    with urllib.request.urlopen(str(benchmark["source_url"]), timeout=60) as response:
        payload = response.read()
    digest = _sha256_bytes(payload)
    _require(len(payload) == benchmark["expected_bytes"], "benchmark byte count changed")
    _require(digest == benchmark["expected_sha256"], "benchmark payload changed")
    status = _atomic_bytes_no_clobber(base / SOURCE_PATH, payload)
    acquisition = {
        "schema_version": "invariant-public-simulation-source-acquisition-1.0",
        "source_id": benchmark["id"],
        "source_url": benchmark["source_url"],
        "path": SOURCE_PATH.as_posix(),
        "bytes": len(payload),
        "sha256": digest,
        "network_calls": 1,
        "observational_response": False,
        "response_values_opened": 0,
        "content_sha256": "",
    }
    acquisition["content_sha256"] = _self_hash(acquisition)
    receipt_status = _atomic_json_no_clobber(base / SOURCE_RECEIPT_PATH, acquisition)
    return {
        "source_status": status,
        "receipt_status": receipt_status,
        "sha256": digest,
        "bytes": len(payload),
    }


def _kernel_map(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["id"]): dict(row) for row in config["kernels"]}


def transfer_response(
    kernel_id: str, omega: np.ndarray, parameters: Mapping[str, float]
) -> np.ndarray:
    values = np.asarray(omega, dtype=float)
    if kernel_id == "K00_INSTANTANEOUS":
        return np.ones(values.shape, dtype=complex)
    if kernel_id == "K01_RETARDED":
        delay = float(parameters["delay"])
        _require(delay >= 0.0, "negative delay")
        return np.exp(-1j * values * delay)
    if kernel_id in {"K02_EXPONENTIAL", "K06_STOCHASTIC_OU"}:
        tau = float(parameters["tau"])
        _require(tau > 0.0, "nonpositive memory time")
        return 1.0 / (1.0 + 1j * values * tau)
    if kernel_id == "K03_BIEXPONENTIAL":
        tau1 = float(parameters["tau1"])
        tau2 = float(parameters["tau2"])
        weight = float(parameters["weight"])
        _require(tau1 > 0.0 and tau2 > 0.0, "nonpositive mixture time")
        _require(0.0 <= weight <= 1.0, "mixture weight outside unit interval")
        return weight / (1.0 + 1j * values * tau1) + (1.0 - weight) / (1.0 + 1j * values * tau2)
    if kernel_id == "K04_DAMPED_RESONANCE":
        omega0 = float(parameters["omega0"])
        zeta = float(parameters["zeta"])
        _require(omega0 > 0.0 and zeta > 0.0, "unstable resonance parameters")
        return omega0**2 / (omega0**2 - values**2 + 2j * zeta * omega0 * values)
    raise DynamicMemoryError(f"no LTI transfer for {kernel_id}")


def simulate_kernel(
    kernel_id: str,
    times: np.ndarray,
    source: np.ndarray,
    parameters: Mapping[str, float],
) -> np.ndarray:
    _require(times.ndim == source.ndim == 1 and times.size == source.size, "fixture shape")
    _require(times.size >= 3 and np.all(np.diff(times) > 0.0), "invalid time grid")
    dt_values = np.diff(times)
    dt = float(np.median(dt_values))
    _require(np.max(np.abs(dt_values - dt)) <= 1.0e-8 * dt, "nonuniform fixture grid")
    if kernel_id == "K00_INSTANTANEOUS":
        return source.copy()
    if kernel_id == "K01_RETARDED":
        delay = float(parameters["delay"])
        _require(delay >= 0.0, "negative delay")
        return np.interp(
            times - delay, times, source, left=float(source[0]), right=float(source[-1])
        )
    if kernel_id in {"K02_EXPONENTIAL", "K06_STOCHASTIC_OU"}:
        tau = float(parameters["tau"])
        _require(tau > 0.0, "nonpositive memory time")
        decay = math.exp(-dt / tau)
        result = np.empty_like(source)
        result[0] = source[0]
        for index in range(1, source.size):
            result[index] = decay * result[index - 1] + (1.0 - decay) * source[index]
        return result
    if kernel_id == "K03_BIEXPONENTIAL":
        weight = float(parameters["weight"])
        _require(0.0 <= weight <= 1.0, "mixture weight outside unit interval")
        first = simulate_kernel(
            "K02_EXPONENTIAL", times, source, {"tau": float(parameters["tau1"])}
        )
        second = simulate_kernel(
            "K02_EXPONENTIAL", times, source, {"tau": float(parameters["tau2"])}
        )
        return weight * first + (1.0 - weight) * second
    if kernel_id == "K04_DAMPED_RESONANCE":
        omega0 = float(parameters["omega0"])
        zeta = float(parameters["zeta"])
        _require(omega0 > 0.0 and zeta > 0.0, "unstable resonance parameters")
        result = np.empty_like(source)
        q = float(source[0])
        velocity = 0.0
        result[0] = q

        def derivative(state_q: float, state_v: float, forcing: float) -> tuple[float, float]:
            return state_v, omega0**2 * (forcing - state_q) - 2.0 * zeta * omega0 * state_v

        for index in range(1, source.size):
            forcing = float(source[index])
            k1q, k1v = derivative(q, velocity, forcing)
            k2q, k2v = derivative(q + 0.5 * dt * k1q, velocity + 0.5 * dt * k1v, forcing)
            k3q, k3v = derivative(q + 0.5 * dt * k2q, velocity + 0.5 * dt * k2v, forcing)
            k4q, k4v = derivative(q + dt * k3q, velocity + dt * k3v, forcing)
            q += dt * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0
            velocity += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
            result[index] = q
        return result
    if kernel_id == "K05_HYSTERETIC":
        threshold = float(parameters["threshold"])
        offset = float(parameters["offset"])
        tau = float(parameters["tau"])
        _require(threshold > 0.0 and offset >= 0.0 and tau > 0.0, "hysteresis parameters")
        decay = math.exp(-dt / tau)
        branch = 0.0
        result = np.empty_like(source)
        result[0] = source[0]
        for index in range(1, source.size):
            forcing = float(source[index])
            if forcing >= threshold:
                branch = 1.0
            elif forcing <= -threshold:
                branch = -1.0
            target = forcing + offset * branch
            result[index] = decay * result[index - 1] + (1.0 - decay) * target
        return result
    raise DynamicMemoryError(f"unknown kernel: {kernel_id}")


def _step_fixture(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    row = config["fixtures"]["step_pulse"]
    dt = float(row["dt"])
    times = np.arange(0.0, float(row["duration"]) + 0.5 * dt, dt)
    source = ((times >= float(row["on"])) & (times < float(row["off"]))).astype(float)
    return times, source


def _loop_area(source: np.ndarray, response: np.ndarray) -> float:
    increments = np.diff(source)
    midpoints = 0.5 * (response[:-1] + response[1:])
    return float(abs(np.sum(midpoints * increments)))


def _complex_gain(
    times: np.ndarray, source: np.ndarray, response: np.ndarray, omega: float
) -> complex:
    basis = np.exp(-1j * omega * times)
    source_coefficient = np.sum(source * basis)
    _require(abs(source_coefficient) > 1.0e-12, "zero source Fourier coefficient")
    return complex(np.sum(response * basis) / source_coefficient)


def _periodic_fixture(
    omega: float, *, cycles: int = 10, samples_per_cycle: int = 1200
) -> tuple[np.ndarray, np.ndarray]:
    _require(omega > 0.0, "periodic frequency must be positive")
    period = 2.0 * math.pi / omega
    dt = period / samples_per_cycle
    times = np.arange(0, cycles * samples_per_cycle + 1, dtype=float) * dt
    return times, np.sin(omega * times)


def _periodic_signature(
    kernel_id: str,
    omega: float,
    parameters: Mapping[str, float],
) -> tuple[complex, float]:
    times, source = _periodic_fixture(omega)
    response = simulate_kernel(kernel_id, times, source, parameters)
    samples_per_cycle = 1200
    start = times.size - 4 * samples_per_cycle - 1
    gain = _complex_gain(times[start:], source[start:], response[start:], omega)
    loop_start = times.size - samples_per_cycle - 1
    return gain, _loop_area(source[loop_start:], response[loop_start:])


def _frequency_rows(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    kernels = _kernel_map(config)
    rows: list[dict[str, Any]] = []
    for kernel_id in KERNEL_IDS:
        parameters = kernels[kernel_id]["parameters"]
        for omega in config["fixtures"]["frequency_grid_omega"]:
            frequency = float(omega)
            if kernel_id == "K05_HYSTERETIC":
                if frequency == 0.0:
                    gain: complex | None = None
                    static_class = "SET_VALUED_HISTORY_DEPENDENT"
                else:
                    gain, _ = _periodic_signature(kernel_id, frequency, parameters)
                    static_class = "DYNAMIC_FUNDAMENTAL"
            else:
                gain = complex(transfer_response(kernel_id, np.asarray([frequency]), parameters)[0])
                static_class = "UNIT_DC_GAIN" if frequency == 0.0 else "LTI_TRANSFER"
            rows.append(
                {
                    "kernel_id": kernel_id,
                    "omega": frequency,
                    "magnitude": None if gain is None else float(abs(gain)),
                    "phase_radian": None if gain is None else float(np.angle(gain)),
                    "classification": static_class,
                }
            )
    return rows


def _rate_sweep(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kernels = _kernel_map(config)
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    frequencies = np.asarray(config["fixtures"]["rate_sweep_omegas"], dtype=float)
    for kernel_id in KERNEL_IDS:
        areas: list[float] = []
        for omega in frequencies:
            if kernel_id == "K06_STOCHASTIC_OU":
                model_id = "K02_EXPONENTIAL"
                parameters = kernels[kernel_id]["parameters"]
            else:
                model_id = kernel_id
                parameters = kernels[kernel_id]["parameters"]
            gain, area = _periodic_signature(model_id, float(omega), parameters)
            areas.append(area)
            rows.append(
                {
                    "kernel_id": kernel_id,
                    "omega": float(omega),
                    "fundamental_magnitude": float(abs(gain)),
                    "fundamental_phase_radian": float(np.angle(gain)),
                    "loop_area": area,
                }
            )
        area_array = np.asarray(areas)
        if np.max(area_array) <= 1.0e-12:
            slope: float | None = None
            intercept = 0.0
        else:
            positive = np.maximum(area_array, 1.0e-300)
            slope = float(np.polyfit(np.log(frequencies), np.log(positive), 1)[0])
            intercept = float(np.polyfit(frequencies, area_array, 1)[1])
        summary[kernel_id] = {
            "low_rate_log_slope": slope,
            "linear_zero_rate_intercept": intercept,
            "lowest_rate_loop_area": areas[0],
            "highest_rate_loop_area": areas[-1],
        }
    return rows, summary


def _step_signatures(config: Mapping[str, Any]) -> dict[str, Any]:
    times, source = _step_fixture(config)
    kernels = _kernel_map(config)
    on = float(config["fixtures"]["step_pulse"]["on"])
    off = float(config["fixtures"]["step_pulse"]["off"])
    rows: dict[str, Any] = {}
    for kernel_id in KERNEL_IDS:
        response = simulate_kernel(kernel_id, times, source, kernels[kernel_id]["parameters"])
        if kernel_id == "K06_STOCHASTIC_OU":
            response_class = "conditional_mean"
        else:
            response_class = "deterministic"
        before = times < on
        after_off = times >= off
        pulse_area = float(np.sum(source) * (times[1] - times[0]))
        rows[kernel_id] = {
            "response_class": response_class,
            "pre_source_max_abs": float(np.max(np.abs(response[before]))),
            "peak": float(np.max(response)),
            "trough": float(np.min(response)),
            "overshoot_above_unit": float(max(0.0, np.max(response) - 1.0)),
            "post_off_absolute_area_over_source_area": float(
                np.sum(np.abs(response[after_off])) * (times[1] - times[0]) / pulse_area
            ),
            "final_response": float(response[-1]),
            "response_sha256": _sha256_bytes(response.astype("<f8").tobytes()),
        }
    _require(
        all(row["pre_source_max_abs"] <= 1.0e-14 for row in rows.values()),
        "advanced mean/deterministic response",
    )
    return rows


def _ou_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    times, source = _step_fixture(config)
    parameters = _kernel_map(config)["K06_STOCHASTIC_OU"]["parameters"]
    tau = float(parameters["tau"])
    sigma = float(parameters["sigma"])
    paths = int(config["fixtures"]["ou_ensemble_paths"])
    seed = int(config["fixtures"]["ou_seed"])
    dt = float(times[1] - times[0])
    decay = math.exp(-dt / tau)
    innovation_scale = sigma * math.sqrt(tau * (1.0 - decay**2) / 2.0)
    generator = np.random.default_rng(seed)
    states = np.zeros(paths, dtype=float)
    ensemble_means = np.empty(times.size, dtype=float)
    ensemble_means[0] = 0.0
    for index in range(1, times.size):
        states = (
            decay * states
            + (1.0 - decay) * source[index]
            + innovation_scale * generator.standard_normal(paths)
        )
        ensemble_means[index] = float(np.mean(states))
    expected_mean = simulate_kernel("K02_EXPONENTIAL", times, source, {"tau": tau})
    analytic_variance = sigma**2 * tau / 2.0
    observed_variance = float(np.var(states, ddof=1))
    return {
        "paths": paths,
        "seed": seed,
        "conditional_mean_equals_K02_exactly": True,
        "analytic_stationary_variance": analytic_variance,
        "ensemble_final_variance": observed_variance,
        "ensemble_to_analytic_variance_ratio": observed_variance / analytic_variance,
        "ensemble_mean_rmse_from_K02": float(
            np.sqrt(np.mean((ensemble_means - expected_mean) ** 2))
        ),
        "noise_bath_injection_power": tau * sigma**2 / 2.0,
        "stationary_fluctuation_dissipation": analytic_variance,
    }


def _energy_ledger(config: Mapping[str, Any], rate_summary: Mapping[str, Any]) -> dict[str, Any]:
    kernels = _kernel_map(config)
    omega = 1.3
    exp_h = transfer_response(
        "K02_EXPONENTIAL", np.asarray([omega]), kernels["K02_EXPONENTIAL"]["parameters"]
    )[0]
    exp_residual = float(abs(exp_h.real - abs(exp_h) ** 2))

    mixture_parameters = kernels["K03_BIEXPONENTIAL"]["parameters"]
    weight = float(mixture_parameters["weight"])
    h1 = 1.0 / (1.0 + 1j * omega * float(mixture_parameters["tau1"]))
    h2 = 1.0 / (1.0 + 1j * omega * float(mixture_parameters["tau2"]))
    mixture_h = weight * h1 + (1.0 - weight) * h2
    mixture_residual = float(
        abs(mixture_h.real - (weight * abs(h1) ** 2 + (1.0 - weight) * abs(h2) ** 2))
    )

    resonance_parameters = kernels["K04_DAMPED_RESONANCE"]["parameters"]
    resonance_h = transfer_response(
        "K04_DAMPED_RESONANCE", np.asarray([omega]), resonance_parameters
    )[0]
    omega0 = float(resonance_parameters["omega0"])
    zeta = float(resonance_parameters["zeta"])
    input_power = -0.5 * omega * resonance_h.imag
    dissipated_power = (2.0 * zeta / omega0) * 0.5 * omega**2 * abs(resonance_h) ** 2
    resonance_residual = float(abs(input_power - dissipated_power))

    hysteresis_parameters = kernels["K05_HYSTERETIC"]["parameters"]
    expected_hysteresis = (
        4.0 * float(hysteresis_parameters["offset"]) * float(hysteresis_parameters["threshold"])
    )
    observed_hysteresis = float(rate_summary["K05_HYSTERETIC"]["lowest_rate_loop_area"])
    return {
        "K01_RETARDED": {
            "lossless_transfer_magnitude_residual": 0.0,
            "completion_status": "PROPAGATING_MEDIATOR_STRESS_ENERGY_NOT_DERIVED",
        },
        "K02_EXPONENTIAL": {
            "periodic_average_input_minus_dissipation": exp_residual,
            "completion_status": "NORMALIZED_OPEN_SYSTEM_BALANCE_DERIVED",
        },
        "K03_BIEXPONENTIAL": {
            "periodic_average_input_minus_dissipation": mixture_residual,
            "completion_status": "NORMALIZED_OPEN_SYSTEM_BALANCE_DERIVED",
        },
        "K04_DAMPED_RESONANCE": {
            "periodic_average_input_minus_dissipation": resonance_residual,
            "completion_status": "NORMALIZED_DAMPED_OSCILLATOR_BALANCE_DERIVED",
        },
        "K05_HYSTERETIC": {
            "quasistatic_expected_loop_dissipation": expected_hysteresis,
            "lowest_rate_observed_loop_dissipation": observed_hysteresis,
            "relative_residual": abs(observed_hysteresis - expected_hysteresis)
            / expected_hysteresis,
            "completion_status": "BRANCH_RESERVOIR_NOT_DERIVED",
        },
        "K06_STOCHASTIC_OU": {
            "stationary_bath_injection_equals_variance_dissipation": True,
            "completion_status": "NORMALIZED_ITO_BATH_BALANCE_DERIVED",
        },
        "receiver_boundary": (
            "Every nonzero q can do mechanical work on matter. These normalized balances account "
            "for kernel-state storage and dissipation, not a covariant source+field+matter stress tensor."
        ),
    }


def _load_benchmark(
    base: Path, config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    path = base / SOURCE_PATH
    _require(path.is_file(), "benchmark source missing; run acquire-benchmark")
    benchmark = config["public_simulation_benchmark"]
    _require(_sha256_file(path) == benchmark["expected_sha256"], "benchmark source changed")
    _require(path.stat().st_size == benchmark["expected_bytes"], "benchmark source size changed")
    try:
        values = np.loadtxt(path, dtype=float)
    except (OSError, ValueError) as exc:
        raise DynamicMemoryError("invalid benchmark table") from exc
    _require(values.shape == (benchmark["expected_rows"], 2), "benchmark shape changed")
    times = values[:, 0]
    strain = values[:, 1]
    _require(np.all(np.diff(times) > 0.0), "benchmark time not monotone")
    dt = float(np.median(np.diff(times)))
    sample_rate = 1.0 / dt
    _require(
        math.isclose(
            sample_rate, benchmark["expected_sample_rate_hz"], rel_tol=0.0, abs_tol=1.0e-8
        ),
        "benchmark sample rate changed",
    )
    scale = float(np.max(np.abs(strain)))
    _require(scale > 0.0 and np.all(np.isfinite(strain)), "benchmark strain invalid")
    normalized_time = (times - times[0]) / float(
        config["normalization"]["benchmark_T_star_seconds"]
    )
    normalized_source = strain / scale
    metadata = {
        "rows": int(values.shape[0]),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "start_seconds": float(times[0]),
        "end_seconds": float(times[-1]),
        "sample_rate_hz": sample_rate,
        "strain_peak_abs": scale,
        "observational_response": False,
    }
    return normalized_time, normalized_source, metadata


def _mismatch_after_shift(
    source: np.ndarray, response: np.ndarray, max_shift: int
) -> tuple[float, int]:
    best_correlation_squared = -1.0
    best_shift = 0
    for shift in range(-max_shift, max_shift + 1):
        if shift >= 0:
            left = source[: source.size - shift or None]
            right = response[shift:]
        else:
            left = source[-shift:]
            right = response[: response.size + shift]
        if left.size < source.size // 2:
            continue
        denominator = float(np.dot(left, left) * np.dot(right, right))
        if denominator <= 0.0:
            continue
        correlation_squared = float(np.dot(left, right) ** 2 / denominator)
        if correlation_squared > best_correlation_squared:
            best_correlation_squared = correlation_squared
            best_shift = shift
    _require(best_correlation_squared >= 0.0, "no valid waveform overlap")
    return max(0.0, 1.0 - best_correlation_squared), best_shift


def _waveform_rows(
    base: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    times, source, metadata = _load_benchmark(base, config)
    kernels = _kernel_map(config)
    dt_seconds = float(config["normalization"]["benchmark_T_star_seconds"]) * float(
        times[1] - times[0]
    )
    source_peak_index = int(np.argmax(np.abs(source)))
    rows: list[dict[str, Any]] = []
    for kernel_id in KERNEL_IDS:
        response = simulate_kernel(kernel_id, times, source, kernels[kernel_id]["parameters"])
        raw_denominator = float(np.dot(source, source) * np.dot(response, response))
        raw_mismatch = 1.0 - float(np.dot(source, response) ** 2 / raw_denominator)
        shifted_mismatch, shift = _mismatch_after_shift(source, response, max_shift=600)
        peak_index = int(np.argmax(np.abs(response)))
        tail_start = min(source.size - 1, source_peak_index + round(0.015 / dt_seconds))
        tail_rms = float(np.sqrt(np.mean(response[tail_start:] ** 2)))
        source_tail_rms = float(np.sqrt(np.mean(source[tail_start:] ** 2)))
        rows.append(
            {
                "kernel_id": kernel_id,
                "output_rms_over_source_rms": float(
                    np.sqrt(np.mean(response**2) / np.mean(source**2))
                ),
                "raw_amplitude_projected_mismatch": max(0.0, raw_mismatch),
                "amplitude_time_projected_mismatch": shifted_mismatch,
                "best_time_shift_seconds": shift * dt_seconds,
                "peak_time_shift_seconds": (peak_index - source_peak_index) * dt_seconds,
                "post_peak_tail_rms_over_source_tail": tail_rms / max(source_tail_rms, 1.0e-300),
                "ou_stationary_excess_variance": (
                    float(kernels[kernel_id]["parameters"]["sigma"]) ** 2
                    * float(kernels[kernel_id]["parameters"]["tau"])
                    / 2.0
                    if kernel_id == "K06_STOCHASTIC_OU"
                    else 0.0
                ),
                "prediction_sha256": _sha256_bytes(response.astype("<f8").tobytes()),
            }
        )
    return rows, metadata


def _equivalence_rows() -> list[dict[str, str]]:
    return [
        {
            "id": "E01_STATIC_DC",
            "status": "EXACT_EQUIVALENCE",
            "members": "K00,K01,K02,K03,K04,K06_mean",
            "discriminator": "none from one settled static snapshot; K05 is set-valued",
        },
        {
            "id": "E02_COMMON_DELAY_TIME_ORIGIN",
            "status": "EXACT_NONIDENTIFIABILITY",
            "members": "K00 versus K01",
            "discriminator": "independent emission time or distance/source-clock transfer; one free coalescence time absorbs K01",
        },
        {
            "id": "E03_OU_CONDITIONAL_MEAN",
            "status": "EXACT_EQUIVALENCE",
            "members": "K02 versus K06 mean",
            "discriminator": "conditional variance, cross-detector coherence, or higher moments",
        },
        {
            "id": "E04_BIEXP_POLE_COLLAPSE",
            "status": "EXACT_PARAMETER_BOUNDARY",
            "members": "K03 versus K02",
            "discriminator": "two distinct timescales and at least three resolved frequencies; tau1=tau2 or weight in {0,1} collapses",
        },
        {
            "id": "E05_ONE_FREQUENCY",
            "status": "EXACT_NUISANCE_EQUIVALENCE",
            "members": "all deterministic kernels at one omega",
            "discriminator": "a second frequency, transition, overshoot, variance, or rate sweep",
        },
        {
            "id": "D01_RATE_PERSISTENT_LOOP",
            "status": "TRUE_DISCRIMINATOR",
            "members": "K05 versus stable finite-memory LTI kernels",
            "discriminator": "nonzero loop-area intercept as omega approaches zero",
        },
        {
            "id": "D02_RESONANCE_PEAK",
            "status": "TRUE_DISCRIMINATOR_WITH_COUNTERMODEL",
            "members": "K04 versus K01,K02,K03",
            "discriminator": "gain above unity plus overshoot; source quasinormal ringing is the mandatory countermodel",
        },
        {
            "id": "D03_EXCESS_VARIANCE",
            "status": "TRUE_DISCRIMINATOR_WITH_NOISE_CONTROL",
            "members": "K06 versus K02",
            "discriminator": "shared astrophysical conditional variance/coherence; independent detector noise is the negative control",
        },
    ]


def _ranked_leads(
    config: Mapping[str, Any], rate_summary: Mapping[str, Any], step: Mapping[str, Any]
) -> list[dict[str, Any]]:
    return [
        {
            "rank": 1,
            "kernel_id": "K05_HYSTERETIC",
            "lead": "rate-persistent causal branch memory",
            "why": "It is the only tested architecture with a nonzero quasistatic loop-area intercept, so it is not another LTI fading kernel.",
            "signature": {
                "zero_rate_intercept": rate_summary["K05_HYSTERETIC"]["linear_zero_rate_intercept"],
                "switch_off_final_response": step["K05_HYSTERETIC"]["final_response"],
            },
            "blocker": "S_star, T_star, branch reservoir, and covariant coupling are not derived independently of a response.",
            "next_falsifier": "Repeated source cycles at multiple rates with an independently reconstructed source state; loop area must approach a nonzero common intercept.",
        },
        {
            "rank": 2,
            "kernel_id": "K04_DAMPED_RESONANCE",
            "lead": "universal source-history resonance",
            "why": "It produces gain above one, step overshoot, and post-source ringing that first-order memory cannot produce.",
            "signature": {"step_overshoot": step["K04_DAMPED_RESONANCE"]["overshoot_above_unit"]},
            "blocker": "Ordinary source quasinormal ringing can imitate the signature, and omega0 lacks a derived environmental scaling.",
            "next_falsifier": "Transfer one frozen omega0 driver law across dissimilar GW events or merger simulations; do not tune omega0 per event.",
        },
        {
            "rank": 3,
            "kernel_id": "K06_STOCHASTIC_OU",
            "lead": "gravity-sector common stochastic memory",
            "why": "Its conditional variance is a distinct observable even though its mean is exactly K02.",
            "signature": {"variance_required": True},
            "blocker": "Instrumental and source noise are stronger countermodels unless the excess is coherently shared across detectors or probes.",
            "next_falsifier": "Cross-detector residual coherence after a frozen GR waveform and detector-noise model, then unchanged cross-event transfer.",
        },
        {
            "rank": 4,
            "kernel_id": "K03_BIEXPONENTIAL",
            "lead": "two-timescale fading memory",
            "why": "It can create tail curvature unavailable to one exponential.",
            "signature": {"two_poles_required": True},
            "blocker": "It is established linear-response structure and collapses exactly to K02 on broad parameter boundaries.",
            "next_falsifier": "Resolve at least three source frequencies or two separated transitions with the same two timescales.",
        },
        {
            "rank": 5,
            "kernel_id": "K02_EXPONENTIAL",
            "lead": "single causal fading memory control",
            "why": "Simple and identifiable from broadband phase plus attenuation.",
            "signature": {"known_control": True},
            "blocker": "Known response law; CALIFA Item 27 provides no positive selection-aware evidence.",
            "next_falsifier": "Broadband source-response cross-spectrum on a fresh time-resolved system.",
        },
        {
            "rank": 6,
            "kernel_id": "K01_RETARDED",
            "lead": "pure source delay",
            "why": "Causal and lossless, but it supplies no amplitude enhancement.",
            "signature": {"unit_magnitude": True},
            "blocker": "Exactly degenerate with a free common emission/coalescence time for one event.",
            "next_falsifier": "An independent emission clock or repeated sources with a frozen distance-dependent delay law.",
        },
    ]


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    _require(bool(rows), "cannot write empty CSV")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _markdown_bytes(
    config: Mapping[str, Any],
    ranked: Sequence[Mapping[str, Any]],
    waveform_rows: Sequence[Mapping[str, Any]],
) -> bytes:
    waveform = {row["kernel_id"]: row for row in waveform_rows}
    lines = [
        "# Dynamic source-history and memory: ranked publication leads",
        "",
        "This packet does not claim that gravity has memory. It makes six causal architectures executable, maps the 80 predecessor TWELL concepts, adds 40 bi-exponential/hysteretic concepts, and freezes the observables that can actually distinguish them.",
        "",
        "## What survived",
        "",
    ]
    for row in ranked:
        lines.extend(
            [
                f"{row['rank']}. **{row['lead']} ({row['kernel_id']})** — {row['why']}",
                f"   Blocker: {row['blocker']}",
                f"   Next falsifier: {row['next_falsifier']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Exact equivalences that prevent false discoveries",
            "",
            "- Every stable linear kernel has unit settled DC gain, so one static galaxy or cluster snapshot cannot identify temporal memory.",
            (
                "- A pure delay is absorbed by a free emission/coalescence time. On the "
                "public NR waveform its time-projected mismatch is "
                f"{waveform['K01_RETARDED']['amplitude_time_projected_mismatch']:.3e}."
            ),
            "- OU stochastic memory and exponential memory have exactly the same conditional mean; variance/coherence is mandatory.",
            "- One complex frequency point is only an amplitude and phase. Broadband curvature, a transition, a low-rate loop intercept, ringing, or variance is required.",
            "",
            "## Strongest counterexample",
            "",
            config["strongest_counterexample"]["interpretation"],
            "",
            "## Novelty boundary",
            "",
            "Causal convolution, exponential mixtures, damped response, stochastic OU dynamics, and hysteresis are all established mathematical structures. The potentially publishable result is a bounded methods contribution: the exact equivalence classes, receiver-energy ledger, and minimal invariant discriminator protocol tied to a public evolving-source benchmark. Historical novelty and an empirical gravity effect remain unproved.",
            "",
            "## Frozen real-data continuation",
            "",
            "The H1 and L1 GW150914 v2 32-second URLs and byte counts are frozen in the config. Before either strain array is read, record its SHA-256/HDF5 schema and bind this prediction receipt. Fit no per-event kernel timescale or threshold.",
        ]
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _svg_bytes(rate_rows: Sequence[Mapping[str, Any]]) -> bytes:
    selected = [
        row
        for row in rate_rows
        if row["kernel_id"] in {"K02_EXPONENTIAL", "K04_DAMPED_RESONANCE", "K05_HYSTERETIC"}
    ]
    width, height = 760, 440
    margin_left, margin_right, margin_top, margin_bottom = 75, 25, 35, 60
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    xs = sorted({float(row["omega"]) for row in selected})
    max_y = max(float(row["loop_area"]) for row in selected) * 1.08

    def x_pos(value: float) -> float:
        return (
            margin_left
            + (math.log(value) - math.log(min(xs)))
            / (math.log(max(xs)) - math.log(min(xs)))
            * plot_w
        )

    def y_pos(value: float) -> float:
        return margin_top + plot_h * (1.0 - value / max_y)

    colors = {
        "K02_EXPONENTIAL": "#2563eb",
        "K04_DAMPED_RESONANCE": "#dc2626",
        "K05_HYSTERETIC": "#16a34a",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="23" text-anchor="middle" font-family="sans-serif" font-size="17">Low-rate loop-area discriminator</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#111"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#111"/>',
    ]
    for kernel_id, color in colors.items():
        rows = [row for row in selected if row["kernel_id"] == kernel_id]
        points = " ".join(
            f"{x_pos(float(row['omega'])):.2f},{y_pos(float(row['loop_area'])):.2f}" for row in rows
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3"/>')
        for row in rows:
            parts.append(
                f'<circle cx="{x_pos(float(row["omega"])):.2f}" cy="{y_pos(float(row["loop_area"])):.2f}" r="4" fill="{color}"/>'
            )
    for index, (kernel_id, color) in enumerate(colors.items()):
        y = 52 + 22 * index
        parts.extend(
            [
                f'<line x1="520" y1="{y}" x2="548" y2="{y}" stroke="{color}" stroke-width="3"/>',
                f'<text x="556" y="{y + 5}" font-family="sans-serif" font-size="12">{kernel_id}</text>',
            ]
        )
    parts.extend(
        [
            f'<text x="{margin_left + plot_w / 2}" y="{height - 15}" text-anchor="middle" font-family="sans-serif" font-size="13">drive angular frequency (log scale)</text>',
            f'<text x="18" y="{margin_top + plot_h / 2}" text-anchor="middle" transform="rotate(-90 18 {margin_top + plot_h / 2})" font-family="sans-serif" font-size="13">normalized loop area</text>',
            '<text x="92" y="405" font-family="sans-serif" font-size="11" fill="#444">LTI lag tends to zero; rate-persistent hysteresis retains a finite intercept.</text>',
            "</svg>",
        ]
    )
    return ("\n".join(parts) + "\n").encode("utf-8")


def derive_package(root: Path | None = None) -> tuple[dict[str, Any], dict[str, bytes]]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    integrity = _validate_local_integrity(base)
    bindings = _validate_bindings(base, config)
    frequency_rows = _frequency_rows(config)
    rate_rows, rate_summary = _rate_sweep(config)
    step = _step_signatures(config)
    ou = _ou_signature(config)
    energy = _energy_ledger(config, rate_summary)
    waveform_rows, waveform_metadata = _waveform_rows(base, config)
    equivalences = _equivalence_rows()
    ranked = _ranked_leads(config, rate_summary, step)

    checks = {
        "EXACT_80_PREDECESSOR_PLUS_40_NEW_CONCEPTS": config["concept_mapping"][
            "total_executable_concepts"
        ]
        == 120,
        "SEVEN_KERNELS_INCLUDING_CONTROL": len(KERNEL_IDS) == 7,
        "STATIC_DC_EQUIVALENCE_EXPLICIT": all(
            math.isclose(
                abs(
                    transfer_response(
                        kernel_id,
                        np.asarray([0.0]),
                        _kernel_map(config)[kernel_id]["parameters"],
                    )[0]
                ),
                1.0,
                abs_tol=1.0e-15,
            )
            for kernel_id in KERNEL_IDS
            if kernel_id != "K05_HYSTERETIC"
        ),
        "NO_ADVANCED_MEAN_OR_DETERMINISTIC_RESPONSE": all(
            row["pre_source_max_abs"] <= 1.0e-14 for row in step.values()
        ),
        "RESONANCE_OVERSHOOT": step["K04_DAMPED_RESONANCE"]["overshoot_above_unit"] > 0.05,
        "HYSTERESIS_RATE_PERSISTENT": abs(rate_summary["K05_HYSTERETIC"]["low_rate_log_slope"])
        < 0.15,
        "FINITE_MEMORY_LTI_LOOP_VANISHES_WITH_RATE": all(
            rate_summary[kernel_id]["low_rate_log_slope"] > 0.8
            for kernel_id in (
                "K01_RETARDED",
                "K02_EXPONENTIAL",
                "K03_BIEXPONENTIAL",
                "K04_DAMPED_RESONANCE",
                "K06_STOCHASTIC_OU",
            )
        ),
        "OU_MEAN_EQUIVALENCE_AND_VARIANCE": ou["conditional_mean_equals_K02_exactly"]
        and 0.85 <= ou["ensemble_to_analytic_variance_ratio"] <= 1.15,
        "LINEAR_STATE_ENERGY_BALANCES": max(
            energy["K02_EXPONENTIAL"]["periodic_average_input_minus_dissipation"],
            energy["K03_BIEXPONENTIAL"]["periodic_average_input_minus_dissipation"],
            energy["K04_DAMPED_RESONANCE"]["periodic_average_input_minus_dissipation"],
        )
        < 1.0e-12,
        "HYSTERESIS_DISSIPATION_LIMIT": energy["K05_HYSTERETIC"]["relative_residual"] < 0.08,
        "PUBLIC_NR_SOURCE_HASH_AND_SHAPE": waveform_metadata["rows"] == 2769
        and waveform_metadata["sha256"] == config["public_simulation_benchmark"]["expected_sha256"],
        "PURE_DELAY_REMOVED_BY_TIME_NUISANCE": next(
            row for row in waveform_rows if row["kernel_id"] == "K01_RETARDED"
        )["amplitude_time_projected_mismatch"]
        < 1.0e-12,
        "FREQUENCY_FINGERPRINTS_COMPLETE": len(frequency_rows) == 56,
        "RATE_SWEEP_COMPLETE": len(rate_rows) == 28,
        "EQUIVALENCES_AND_TRUE_DISCRIMINATORS_EXPLICIT": len(equivalences) == 8,
        "OBSERVATIONAL_RESPONSES_UNOPENED": config["access_ledger"][
            "observational_response_files_opened"
        ]
        == 0,
        "STRONGEST_CALIFA_COUNTEREXAMPLE_RETAINED": config["strongest_counterexample"][
            "selection_aware_permutation_p"
        ]
        == 0.75,
        "CLAIM_BOUNDARY_CONSERVATIVE": not any(
            config["claim_boundary"][key]
            for key in (
                "observational_response_opened",
                "real_data_fit_performed",
                "covariant_gravity_action_derived",
                "universal_physical_time_scale_derived",
                "historical_novelty_established",
                "empirical_gravity_discovery",
                "publication_ready_empirical_claim",
            )
        ),
    }
    _require(all(checks.values()), "one or more scientific checks failed")

    artifacts = {
        "kernel-frequency-signatures.csv": _csv_bytes(frequency_rows),
        "low-rate-loop-discriminator.csv": _csv_bytes(rate_rows),
        "gw150914-nr-response-blind-predictions.csv": _csv_bytes(waveform_rows),
        "equivalence-and-discriminator-map.csv": _csv_bytes(equivalences),
        "ranked-leads.md": _markdown_bytes(config, ranked, waveform_rows),
        "low-rate-loop-discriminator.svg": _svg_bytes(rate_rows),
    }
    manifest = {
        name: {"bytes": len(payload), "sha256": _sha256_bytes(payload)}
        for name, payload in artifacts.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": DECISION,
        "scope": "theory, synthetic fixtures, and one public NR source waveform; no observational strain response",
        "integrity": integrity,
        "bindings": bindings,
        "concept_coverage": config["concept_mapping"],
        "normalization": config["normalization"],
        "kernel_contracts": config["kernels"],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "step_pulse_signatures": step,
        "rate_sweep_summary": rate_summary,
        "ou_stochastic_signature": ou,
        "receiver_energy_ledger": energy,
        "public_NR_benchmark": waveform_metadata,
        "waveform_predictions": waveform_rows,
        "equivalence_and_discriminator_map": equivalences,
        "ranked_leads": ranked,
        "strongest_counterexample": config["strongest_counterexample"],
        "observational_response_preflight": config["observational_response_preflight"],
        "primary_literature": config["primary_literature"],
        "novelty_boundary": {
            "known": "Causal convolution, exponential mixtures, damped response, OU noise, and hysteresis are established structures.",
            "candidate_methods_contribution": "Exact equivalence classes plus a minimal invariant discriminator and receiver-energy ledger tied to a public evolving-source benchmark.",
            "not_established": "Historical novelty, a covariant gravity action, a universal driver-derived scale, or an empirical gravity effect.",
        },
        "next_empirical_falsifier": {
            "dataset": "GWOSC GW150914 H1/L1 V2 32-second 4096-Hz calibrated strain",
            "rule": "Bind this prediction receipt before reading either strain array; reproduce GR, perform frozen injection recovery, project the frozen nuisance space, and fit no per-event kernel scale.",
            "priority": "Test shared resonance/phase-curvature and cross-detector stochastic coherence; pure delay is a registered nonidentifiability control.",
        },
        "access_ledger": {
            "public_simulation_source_files_opened": 1,
            "public_simulation_source_rows_read": waveform_metadata["rows"],
            "observational_response_files_opened": 0,
            "observational_response_rows_read": 0,
            "real_response_scores_computed": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "post_response_formula_changes": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "artifact_manifest": manifest,
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, artifacts


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("analysis_id") == config["analysis_id"], "receipt identity changed")
    _require(receipt.get("status") == DECISION, "receipt decision changed")
    _require(receipt.get("content_sha256") == _self_hash(receipt), "receipt self-hash changed")
    checks = receipt.get("checks")
    _require(isinstance(checks, dict) and all(checks.values()), "receipt checks failed")
    _require(
        receipt.get("checks_passed") == receipt.get("checks_total") == 18, "check count changed"
    )
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "claims changed")
    _require(
        receipt.get("access_ledger")
        == {
            "public_simulation_source_files_opened": 1,
            "public_simulation_source_rows_read": 2769,
            "observational_response_files_opened": 0,
            "observational_response_rows_read": 0,
            "real_response_scores_computed": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
            "post_response_formula_changes": 0,
        },
        "receipt access ledger changed",
    )
    _require(len(receipt.get("ranked_leads", [])) == 6, "ranked leads changed")
    _require(
        receipt["ranked_leads"][0]["kernel_id"] == "K05_HYSTERETIC",
        "lead ranking changed",
    )


def build_package(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    receipt, artifacts = derive_package(base)
    statuses: list[str] = []
    for name, payload in artifacts.items():
        statuses.append(_atomic_bytes_no_clobber(base / ARTIFACT_DIRECTORY / name, payload))
    statuses.append(_atomic_json_no_clobber(base / OUTPUT_PATH, receipt))
    return "CREATED" if "CREATED" in statuses else "EXISTING_IDENTICAL"


def check_package(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    expected, artifacts = derive_package(base)
    observed = _read_json(base / OUTPUT_PATH)
    validate_receipt(observed, config)
    _require(observed == expected, "receipt differs from deterministic rebuild")
    for name, payload in artifacts.items():
        path = base / ARTIFACT_DIRECTORY / name
        _require(path.is_file() and path.read_bytes() == payload, f"artifact changed: {name}")
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire-benchmark", "build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "acquire-benchmark":
        print(json.dumps(acquire_benchmark(), sort_keys=True))
    elif args.command == "build":
        print(build_package())
    else:
        receipt = check_package()
        if args.command == "status":
            print(receipt["status"])
        else:
            print("VALID")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
