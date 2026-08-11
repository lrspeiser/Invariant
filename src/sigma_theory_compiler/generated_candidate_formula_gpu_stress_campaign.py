from __future__ import annotations

import argparse
import hashlib
import json
import threading
import time
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "sigma-generated-candidate-formula-gpu-stress-1.0"
CONFIG_SCHEMA = "sigma-generated-candidate-formula-gpu-stress-config-1.0"

FAMILIES = {
    "AETHER_K1234_PARAMETER_CELL",
    "KESSENCE_G2_CONVEX",
    "CUBIC_HORNDESKI_G3_WEAK_CELL",
    "CONFORMAL_G4_PHI_SCALAR_TENSOR",
}

FEATURES = (
    "aether_E_EH",
    "aether_E_K1",
    "aether_E_K2",
    "aether_E_K3",
    "aether_E_K4",
    "aether_E_lambda",
    "X_phi",
    "phi",
    "theta",
    "p_mu_p_nu",
    "g_mu_nu",
    "q_mu_p_nu",
    "g_mu_nu_q_rho_p_rho",
    "G_mu_nu",
    "H_mu_nu",
)

PROJECTION_BASIS = (
    "aether_E_EH",
    "aether_E_K1",
    "aether_E_K2",
    "aether_E_K3",
    "aether_E_K4",
    "aether_E_lambda",
    "p_mu_p_nu",
    "X_phi_times_g_mu_nu",
    "X_phi_times_p_mu_p_nu",
    "X_phi_squared_times_g_mu_nu",
    "theta_times_p_mu_p_nu",
    "q_mu_p_nu",
    "g_mu_nu_q_rho_p_rho",
    "phi_theta_times_g_mu_nu",
    "G_mu_nu",
    "phi_squared_times_G_mu_nu",
    "phi_times_H_mu_nu",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(document: Mapping[str, Any]) -> str:
    return _sha({key: value for key, value in document.items() if key != "content_sha256"})


def _root(config_path: Path) -> Path:
    return config_path.resolve().parents[1]


def _load_inputs(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported GPU stress config schema")
    required_false = (
        "observations_opened",
        "dark_matter_or_halo_inputs",
        "redshift_distance_inputs",
        "paid_llm_calls",
        "formal_pass_inference_allowed",
    )
    if config.get("synthetic_only") is not True or any(config.get(key) is not False for key in required_false):
        raise ValueError("GPU stress data and claim seals must remain closed")
    root = _root(config_path)
    binding = config["metric_variation_artifact"]
    source_path = root / binding["path"]
    if _file_sha(source_path) != binding["file_sha256"]:
        raise ValueError("metric-variation artifact file hash mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if source.get("content_sha256") != binding["content_sha256"] or _content_sha(source) != binding["content_sha256"]:
        raise ValueError("metric-variation artifact content hash mismatch")
    counts = source.get("metric_variation_execution_counts", {})
    if counts.get("candidate_action_hashes_specialized") != 163 or len(source.get("candidate_records", [])) != 163:
        raise ValueError("GPU stress source is not the reviewed 163-action registry")
    return config, source, root


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return values ^ (values >> np.uint64(31))


def deterministic_integer_inputs(config: Mapping[str, Any]) -> np.ndarray:
    """Return backend-independent signed dyadic numerators in feature-major order."""
    points = int(config["synthetic_points_per_candidate"])
    bits = int(config["synthetic_integer_bits"])
    if not 1 <= bits <= 30 or points < int(config["exact_cpu_points_per_candidate"]):
        raise ValueError("invalid deterministic synthetic point contract")
    total = len(FEATURES) * points
    indexes = np.arange(total, dtype=np.uint64)
    values = _splitmix64(indexes + np.uint64(int(config["deterministic_seed"])))
    modulus = np.uint64(1 << (bits + 1))
    signed = (values % modulus).astype(np.int64) - (1 << bits)
    return np.ascontiguousarray(signed.reshape(len(FEATURES), points), dtype="<i8")


def _float_inputs(integers: np.ndarray, bits: int) -> dict[str, np.ndarray]:
    scale = float(1 << bits)
    arrays = {name: integers[index].astype(np.float64) / scale for index, name in enumerate(FEATURES)}
    arrays["X_phi"] = np.abs(integers[FEATURES.index("X_phi")]).astype(np.float64) % (1 << (bits - 5))
    arrays["X_phi"] /= scale
    arrays["phi"] = integers[FEATURES.index("phi")].astype(np.float64) / float(1 << (bits + 5))
    return arrays


def _fraction_inputs(integers: np.ndarray, bits: int, point: int) -> dict[str, Fraction]:
    values = {name: Fraction(int(integers[index, point]), 1 << bits) for index, name in enumerate(FEATURES)}
    values["X_phi"] = Fraction(abs(int(integers[FEATURES.index("X_phi"), point])) % (1 << (bits - 5)), 1 << bits)
    values["phi"] = Fraction(int(integers[FEATURES.index("phi"), point]), 1 << (bits + 5))
    return values


def _records(source: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = sorted(source["candidate_records"], key=lambda row: row["candidate_id"])
    ids = [row["candidate_id"] for row in records]
    actions = [row["action_sha256"] for row in records]
    eulers = [
        row["metric_variation_execution"]["specialization"]["candidate_metric_euler_sha256"]
        for row in records
    ]
    if len(set(ids)) != 163 or len(set(actions)) != 163 or len(set(eulers)) != 163:
        raise ValueError("candidate/formula/action uniqueness changed")
    if {row["family_id"] for row in records} != FAMILIES:
        raise ValueError("candidate family registry changed")
    return records


def _rational_parameters(record: Mapping[str, Any]) -> dict[str, Fraction]:
    specialization = record["metric_variation_execution"]["specialization"]
    if record["family_id"] == "AETHER_K1234_PARAMETER_CELL":
        return {key: Fraction(value) for key, value in specialization["candidate_substitution"].items()}
    return {
        key: Fraction(value)
        for key, value in specialization.get("exact_rational_parameter_substitutions", {}).items()
    }


def _project_one(record: Mapping[str, Any], values: Mapping[str, Any]) -> Any:
    family = record["family_id"]
    parameters = _rational_parameters(record)
    if family == "AETHER_K1234_PARAMETER_CELL":
        return (
            values["aether_E_EH"]
            + parameters["c1"] * values["aether_E_K1"]
            + parameters["c2"] * values["aether_E_K2"]
            + parameters["c3"] * values["aether_E_K3"]
            + parameters["c4"] * values["aether_E_K4"]
            + values["aether_E_lambda"]
        )
    x_phi = values["X_phi"]
    pmpn = values["p_mu_p_nu"]
    metric = values["g_mu_nu"]
    if family == "KESSENCE_G2_CONVEX":
        q = parameters["q"]
        return (1 + 2 * q * x_phi) * pmpn + (x_phi + q * x_phi**2) * metric
    if family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        beta = parameters["beta"]
        return (
            (1 - beta * values["theta"]) * pmpn
            + x_phi * metric
            - 2 * beta * values["q_mu_p_nu"]
            + beta * values["g_mu_nu_q_rho_p_rho"]
        )
    if family == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        phi = values["phi"]
        return (
            Fraction(49, 50) * pmpn
            + (Fraction(24, 25) * x_phi + phi * values["theta"] / 50) * metric
            + (Fraction(1, 2) + phi**2 / 100) * values["G_mu_nu"]
            - phi * values["H_mu_nu"] / 50
        )
    raise ValueError("unsupported candidate family")


def _coefficient_matrix(records: list[dict[str, Any]]) -> np.ndarray:
    coefficients = np.zeros((len(records), len(PROJECTION_BASIS)), dtype=np.float64)
    column = {name: index for index, name in enumerate(PROJECTION_BASIS)}
    for row, record in enumerate(records):
        parameters = {key: float(value) for key, value in _rational_parameters(record).items()}
        family = record["family_id"]
        if family == "AETHER_K1234_PARAMETER_CELL":
            coefficients[row, column["aether_E_EH"]] = 1.0
            coefficients[row, column["aether_E_lambda"]] = 1.0
            for index in range(1, 5):
                coefficients[row, column[f"aether_E_K{index}"]] = parameters[f"c{index}"]
        elif family == "KESSENCE_G2_CONVEX":
            q = parameters["q"]
            coefficients[row, column["p_mu_p_nu"]] = 1.0
            coefficients[row, column["X_phi_times_g_mu_nu"]] = 1.0
            coefficients[row, column["X_phi_times_p_mu_p_nu"]] = 2.0 * q
            coefficients[row, column["X_phi_squared_times_g_mu_nu"]] = q
        elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
            beta = parameters["beta"]
            coefficients[row, column["p_mu_p_nu"]] = 1.0
            coefficients[row, column["X_phi_times_g_mu_nu"]] = 1.0
            coefficients[row, column["theta_times_p_mu_p_nu"]] = -beta
            coefficients[row, column["q_mu_p_nu"]] = -2.0 * beta
            coefficients[row, column["g_mu_nu_q_rho_p_rho"]] = beta
        else:
            coefficients[row, column["p_mu_p_nu"]] = 49.0 / 50.0
            coefficients[row, column["X_phi_times_g_mu_nu"]] = 24.0 / 25.0
            coefficients[row, column["phi_theta_times_g_mu_nu"]] = 1.0 / 50.0
            coefficients[row, column["G_mu_nu"]] = 0.5
            coefficients[row, column["phi_squared_times_G_mu_nu"]] = 1.0 / 100.0
            coefficients[row, column["phi_times_H_mu_nu"]] = -1.0 / 50.0
    return np.ascontiguousarray(coefficients)


def _basis_matrix(arrays: Mapping[str, Any], xp: Any) -> Any:
    x_phi = arrays["X_phi"]
    phi = arrays["phi"]
    metric = arrays["g_mu_nu"]
    pmpn = arrays["p_mu_p_nu"]
    rows = [
        arrays["aether_E_EH"],
        arrays["aether_E_K1"],
        arrays["aether_E_K2"],
        arrays["aether_E_K3"],
        arrays["aether_E_K4"],
        arrays["aether_E_lambda"],
        pmpn,
        x_phi * metric,
        x_phi * pmpn,
        x_phi * x_phi * metric,
        arrays["theta"] * pmpn,
        arrays["q_mu_p_nu"],
        arrays["g_mu_nu_q_rho_p_rho"],
        phi * arrays["theta"] * metric,
        arrays["G_mu_nu"],
        phi * phi * arrays["G_mu_nu"],
        phi * arrays["H_mu_nu"],
    ]
    return xp.stack(rows, axis=0)


def _project(coefficients: Any, basis: Any, xp: Any) -> Any:
    return xp.matmul(coefficients, basis)


class _NvmlSampler:
    def __init__(self, device_index: int, interval: float) -> None:
        self.device_index = device_index
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def sample() -> None:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
                while not self._stop.is_set():
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    self.samples.append(
                        {
                            "gpu_percent": int(utilization.gpu),
                            "memory_percent": int(utilization.memory),
                            "memory_used_mib": int(memory.used // (1024 * 1024)),
                            "power_milliwatts": int(pynvml.nvmlDeviceGetPowerUsage(handle)),
                        }
                    )
                    self._stop.wait(self.interval)
                pynvml.nvmlShutdown()
            except Exception as error:  # noqa: BLE001 - optional hardware telemetry
                self.error = f"{type(error).__name__}: {error}"

        self._thread = threading.Thread(target=sample, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if not self.samples:
            return {"available": False, "sample_count": 0, "unavailable_reason": self.error or "no samples"}
        count = len(self.samples)
        return {
            "available": True,
            "sample_count": count,
            "sample_interval_seconds": self.interval,
            "gpu_percent_mean": sum(row["gpu_percent"] for row in self.samples) / count,
            "gpu_percent_max": max(row["gpu_percent"] for row in self.samples),
            "memory_percent_mean": sum(row["memory_percent"] for row in self.samples) / count,
            "memory_percent_max": max(row["memory_percent"] for row in self.samples),
            "memory_used_mib_max": max(row["memory_used_mib"] for row in self.samples),
            "power_watts_max": max(row["power_milliwatts"] for row in self.samples) / 1000.0,
            "counter_scope": (
                "device-wide NVML samples during measured synchronized GPU repetitions; counters "
                "can include concurrent processes and are not a continuous or lane-only utilization claim"
            ),
        }


def _exact_cpu_control(
    records: list[dict[str, Any]], integers: np.ndarray, cpu: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    points = int(config["exact_cpu_points_per_candidate"])
    bits = int(config["synthetic_integer_bits"])
    entries = []
    maximum = 0.0
    equality_count = 0
    for row, record in enumerate(records):
        for point in range(points):
            exact = _project_one(record, _fraction_inputs(integers, bits, point))
            reference = float(exact)
            error = abs(float(cpu[row, point]) - reference)
            maximum = max(maximum, error)
            equality_count += int(error == 0.0)
            entries.append(
                {
                    "candidate_id": record["candidate_id"],
                    "point": point,
                    "numerator": exact.numerator,
                    "denominator": exact.denominator,
                }
            )
    count = len(entries)
    bound = float(config["exact_cpu_float_error_bound"])
    return {
        "method": "Python Fraction evaluation of every candidate on the declared dyadic sentinel points",
        "crosscheck_count": count,
        "candidate_count": len(records),
        "points_per_candidate": points,
        "exact_result_registry_root_sha256": _sha(entries),
        "float64_max_absolute_error_after_single_reference_conversion": maximum,
        "bit_equal_to_converted_exact_count": equality_count,
        "error_bound": bound,
        "within_bound": maximum <= bound,
    }


def _gpu_device(cp: Any) -> dict[str, Any]:
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    return {
        "backend": "cupy_cuda",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_index": int(device.id),
        "device_name": name,
        "compute_capability": f"{int(props['major'])}.{int(props['minor'])}",
        "total_global_memory_mib": int(props["totalGlobalMem"] // (1024 * 1024)),
    }


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config, source, root = _load_inputs(config_path)
    records = _records(source)
    integers = deterministic_integer_inputs(config)
    host_arrays = _float_inputs(integers, int(config["synthetic_integer_bits"]))
    host_coefficients = _coefficient_matrix(records)
    host_basis = _basis_matrix(host_arrays, np)

    cpu_started = time.perf_counter()
    cpu = _project(host_coefficients, host_basis, np)
    cpu_elapsed = time.perf_counter() - cpu_started
    exact = _exact_cpu_control(records, integers, cpu, config)
    if not exact["within_bound"]:
        raise ValueError("float64 CPU projection exceeded exact-rational control bound")

    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CuPy CUDA backend unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CuPy reports no CUDA device")
    device = _gpu_device(cp)
    gpu_coefficients = cp.asarray(host_coefficients)
    gpu_basis = cp.asarray(host_basis)
    warmups = int(config["gpu_warmup_repetitions"])
    repetitions = int(config["gpu_measured_repetitions"])
    for _ in range(warmups):
        gpu = _project(gpu_coefficients, gpu_basis, cp)
    cp.cuda.Device().synchronize()
    sampler = _NvmlSampler(device["device_index"], float(config["utilization_sample_interval_seconds"]))
    sampler.start()
    gpu_started = time.perf_counter()
    for _ in range(repetitions):
        gpu = _project(gpu_coefficients, gpu_basis, cp)
    cp.cuda.Device().synchronize()
    gpu_elapsed = time.perf_counter() - gpu_started
    utilization = sampler.stop()
    gpu_host = cp.asnumpy(gpu)

    absolute = np.abs(gpu_host - cpu)
    floor = float(config["gpu_cpu_error_bounds"]["relative_floor"])
    relative = absolute / np.maximum(np.abs(cpu), floor)
    absolute_bound = float(config["gpu_cpu_error_bounds"]["absolute"])
    relative_bound = float(config["gpu_cpu_error_bounds"]["relative"])
    violating = (absolute > absolute_bound) & (relative > relative_bound)
    numeric = {
        "comparison_count": int(cpu.size),
        "max_absolute_error": float(np.max(absolute)),
        "max_relative_error": float(np.max(relative)),
        "absolute_error_bound": absolute_bound,
        "relative_error_bound": relative_bound,
        "relative_floor": floor,
        "bound_semantics": "a point violates only when both absolute and relative bounds are exceeded",
        "violating_point_count": int(np.count_nonzero(violating)),
        "within_bounds": not bool(np.any(violating)),
        "cpu_output_sha256": hashlib.sha256(np.ascontiguousarray(cpu, dtype="<f8").tobytes()).hexdigest(),
        "gpu_output_sha256": hashlib.sha256(np.ascontiguousarray(gpu_host, dtype="<f8").tobytes()).hexdigest(),
    }
    if not numeric["within_bounds"]:
        raise ValueError("GPU output exceeded explicit CPU comparison bounds")

    points = int(config["synthetic_points_per_candidate"])
    family_counts = dict(sorted(Counter(row["family_id"] for row in records).items()))
    registry = [
        {
            "candidate_id": row["candidate_id"],
            "family_id": row["family_id"],
            "action_sha256": row["action_sha256"],
            "candidate_metric_euler_sha256": row["metric_variation_execution"]["specialization"]["candidate_metric_euler_sha256"],
        }
        for row in records
    ]
    source_path = root / config["metric_variation_artifact"]["path"]
    profile_path = root / config["resource_profile"]["path"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": (
            "candidate-bound numerical stress of the 163 materialized metric-Euler formula "
            "projections on independent synthetic dyadic operator coordinates"
        ),
        "interpretation": (
            "This is a numerical backend/throughput control. Synthetic operator coordinates need "
            "not be realizable field jets, and agreement cannot establish a field equation, formal "
            "pass, phenomenological fitness, or observational support."
        ),
        "source_bindings": {
            "metric_variation_artifact": {
                "path": config["metric_variation_artifact"]["path"],
                "file_sha256": _file_sha(source_path),
                "content_sha256": source["content_sha256"],
            },
            "resource_profile": {
                "path": config["resource_profile"]["path"],
                "file_sha256": _file_sha(profile_path),
            },
            "campaign_config": {
                "path": config_path.resolve().relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
        },
        "deterministic_manifest": {
            "seed": int(config["deterministic_seed"]),
            "generator": "vectorized_splitmix64_v1",
            "synthetic_integer_bits": int(config["synthetic_integer_bits"]),
            "feature_order": list(FEATURES),
            "feature_count": len(FEATURES),
            "synthetic_points_per_candidate": points,
            "input_integer_tensor_sha256": hashlib.sha256(integers.tobytes()).hexdigest(),
            "candidate_formula_registry_root_sha256": _sha(registry),
        },
        "counts": {
            "candidate_count": len(records),
            "family_count": len(family_counts),
            "synthetic_points_per_candidate": points,
            "unique_candidate_point_pairs": len(records) * points,
            "cpu_full_projection_evaluations": len(records) * points,
            "cpu_exact_rational_crosschecks": exact["crosscheck_count"],
            "gpu_warmup_repetitions": warmups,
            "gpu_measured_repetitions": repetitions,
            "gpu_measured_candidate_formula_evaluations": len(records) * points * repetitions,
            "gpu_projection_dispatches": warmups + repetitions,
            "formal_passes_inferred": 0,
            "observational_records_accessed": 0,
            "paid_llm_calls": 0,
        },
        "family_counts": family_counts,
        "exact_cpu_control": exact,
        "gpu_cpu_comparison": numeric,
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "cpu_full_projection_wall_seconds": cpu_elapsed,
            "gpu_measured_wall_seconds": gpu_elapsed,
            "gpu_candidate_formula_evaluations_per_second": len(records) * points * repetitions / gpu_elapsed,
            "gpu_allocated_input_bytes": int(gpu_coefficients.nbytes + gpu_basis.nbytes),
            "gpu_allocated_output_bytes": int(gpu.nbytes),
            "utilization": utilization,
            "timing_scope": "single measured local run; not deterministic and not a sustained-capacity guarantee",
        },
        "campaign_decision": "completed_numerical_stress_control_only",
        "candidate_backend_metric_variation_executed": False,
        "field_equations_proven": False,
        "formal_pass_inferred": False,
        "candidate_rejection_authorized": False,
        "scientific_ranking_authorized": False,
        "synthetic_only": True,
        "observations_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
    }
    source_file = root / "src/sigma_theory_compiler/generated_candidate_formula_gpu_stress_campaign.py"
    test_file = root / "tests/test_generated_candidate_formula_gpu_stress_campaign.py"
    result["source_bindings"]["campaign_source"] = {
        "path": source_file.relative_to(root).as_posix(),
        "file_sha256": _file_sha(source_file),
    }
    if test_file.is_file():
        result["source_bindings"]["campaign_test"] = {
            "path": test_file.relative_to(root).as_posix(),
            "file_sha256": _file_sha(test_file),
        }
    result["content_sha256"] = _content_sha(result)
    validate_campaign(result, config_path)
    return result


def validate_campaign(document: Mapping[str, Any], config_path: str | Path) -> None:
    config_path = Path(config_path)
    config, source, root = _load_inputs(config_path)
    if document.get("schema_version") != SCHEMA or document.get("content_sha256") != _content_sha(document):
        raise ValueError("GPU stress artifact identity/content hash mismatch")
    seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_calls": False,
        "formal_pass_inferred": False,
        "candidate_backend_metric_variation_executed": False,
        "field_equations_proven": False,
        "candidate_rejection_authorized": False,
        "scientific_ranking_authorized": False,
    }
    if any(document.get(key) != value for key, value in seals.items()):
        raise ValueError("GPU stress artifact violates a data or claim seal")
    counts = document.get("counts", {})
    if (
        counts.get("candidate_count") != 163
        or counts.get("formal_passes_inferred") != 0
        or counts.get("observational_records_accessed") != 0
        or counts.get("paid_llm_calls") != 0
        or sum(document.get("family_counts", {}).values()) != 163
    ):
        raise ValueError("GPU stress counter accounting mismatch")
    if document.get("exact_cpu_control", {}).get("within_bound") is not True:
        raise ValueError("exact CPU control did not close")
    comparison = document.get("gpu_cpu_comparison", {})
    if comparison.get("within_bounds") is not True or comparison.get("violating_point_count") != 0:
        raise ValueError("GPU/CPU comparison did not close")
    bindings = document.get("source_bindings", {})
    expected = {
        "metric_variation_artifact": config["metric_variation_artifact"]["path"],
        "resource_profile": config["resource_profile"]["path"],
        "campaign_config": config_path.resolve().relative_to(root).as_posix(),
    }
    for key, relative in expected.items():
        binding = bindings.get(key, {})
        if binding.get("path") != relative or binding.get("file_sha256") != _file_sha(root / relative):
            raise ValueError(f"GPU stress source binding mismatch: {key}")
    metric = bindings["metric_variation_artifact"]
    if metric.get("content_sha256") != source["content_sha256"]:
        raise ValueError("GPU stress predecessor content binding mismatch")
    for key in ("campaign_source", "campaign_test"):
        if key in bindings:
            binding = bindings[key]
            if binding.get("file_sha256") != _file_sha(root / binding["path"]):
                raise ValueError(f"GPU stress local binding mismatch: {key}")
    rebuilt = deterministic_integer_inputs(config)
    manifest = document.get("deterministic_manifest", {})
    if manifest.get("input_integer_tensor_sha256") != hashlib.sha256(rebuilt.tobytes()).hexdigest():
        raise ValueError("GPU stress deterministic input manifest mismatch")
    registry = [
        {
            "candidate_id": row["candidate_id"],
            "family_id": row["family_id"],
            "action_sha256": row["action_sha256"],
            "candidate_metric_euler_sha256": row["metric_variation_execution"]["specialization"]["candidate_metric_euler_sha256"],
        }
        for row in _records(source)
    ]
    if manifest.get("candidate_formula_registry_root_sha256") != _sha(registry):
        raise ValueError("GPU stress candidate/formula registry root mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    document = build_campaign(args.config)
    Path(args.output).write_text(_canonical(document) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
