from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import mpmath
import numpy as np

SCHEMA = "sigma-kastner-schlatter-cuda-consequence-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-cuda-consequence-config-1.0"

POISSON_KERNEL = r"""
extern "C" __global__
void poisson_inverse(
    const unsigned long long* randoms,
    const double* cdf,
    const int row_count,
    const int sample_count,
    const int maximum_count,
    unsigned char* output) {
  long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  long long total = (long long)row_count * sample_count;
  if (index >= total) return;
  int row = (int)(index / sample_count);
  double u = ((double)(randoms[index] >> 11) + 0.5) * 1.1102230246251565e-16;
  const double* thresholds = cdf + (long long)row * (maximum_count + 1);
  int count = 0;
  while (count < maximum_count && u > thresholds[count]) ++count;
  output[index] = (unsigned char)count;
}
"""

SDS_KERNEL = r"""
extern "C" __global__
void sds_roots(
    const double* epsilon,
    const int count,
    const int iterations,
    double* black_root,
    double* cosmological_root,
    double* maximum_residual,
    unsigned char* domain) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= count) return;
  double e = epsilon[i];
  double lo = e;
  double hi = 0.5;
  for (int step = 0; step < iterations; ++step) {
    double mid = 0.5 * (lo + hi);
    double f = 1.0 - mid * mid - e / mid;
    if (f > 0.0) hi = mid; else lo = mid;
  }
  double rb = 0.5 * (lo + hi);
  lo = 0.5;
  hi = 1.0;
  for (int step = 0; step < iterations; ++step) {
    double mid = 0.5 * (lo + hi);
    double f = 1.0 - mid * mid - e / mid;
    if (f > 0.0) lo = mid; else hi = mid;
  }
  double rc = 0.5 * (lo + hi);
  double fb = fabs(1.0 - rb * rb - e / rb);
  double fc = fabs(1.0 - rc * rc - e / rc);
  double rm = sqrt(rb * rc);
  double fm = 1.0 - rm * rm - e / rm;
  black_root[i] = rb;
  cosmological_root[i] = rc;
  maximum_residual[i] = fb > fc ? fb : fc;
  domain[i] = (rb > 0.0 && rb < rc && rc < 1.0 && fm > 0.0) ? 1 : 0;
}
"""

MOND_KERNEL = r"""
extern "C" __global__
void mond_consequences(
    const double* mass,
    const double* radius,
    const int count,
    double* exact_acceleration,
    double* asymptotic_acceleration,
    double* btfr_v4,
    unsigned char* domain) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= count) return;
  double m = mass[i];
  double r = radius[i];
  double r0 = sqrt(m);
  double phi = m / r;
  double exact = phi * (1.0 + sqrt(1.0 + 1.0 / m));
  double asymptotic = sqrt(m) / r;
  exact_acceleration[i] = exact;
  asymptotic_acceleration[i] = asymptotic;
  btfr_v4[i] = (asymptotic * r) * (asymptotic * r);
  domain[i] = (m > 0.0 && r > r0 && phi < 1.0) ? 1 : 0;
}
"""


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


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported consequence campaign config")
    expected = {
        "synthetic_only": True,
        "observations_opened": False,
        "ontology_pass_allowed": False,
        "theory_pass_allowed": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("synthetic consequence campaign seals are not fail-closed")
    root = _root(config_path)
    predecessor_path = root / config["predecessor"]["path"]
    if _file_sha(predecessor_path) != config["predecessor"]["file_sha256"]:
        raise ValueError("transactional-gravity intake file hash mismatch")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("content_sha256") != config["predecessor"]["content_sha256"]
        or _content_sha(predecessor) != config["predecessor"]["content_sha256"]
    ):
        raise ValueError("transactional-gravity intake content hash mismatch")
    return config, predecessor, root


def _splitmix64(values: np.ndarray) -> np.ndarray:
    mask = np.uint64(0xFFFFFFFFFFFFFFFF)
    values = (values + np.uint64(0x9E3779B97F4A7C15)) & mask
    values = ((values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & mask
    values = ((values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & mask
    return values ^ (values >> np.uint64(31))


def deterministic_uint64(seed: int, count: int, stream: int) -> np.ndarray:
    offset = np.uint64((stream * 0xD1342543DE82EF95) & ((1 << 64) - 1))
    indexes = np.arange(count, dtype=np.uint64)
    return np.ascontiguousarray(_splitmix64(indexes + np.uint64(seed) + offset), dtype="<u8")


def deterministic_inputs(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rows = len(config["poisson"]["means"]) * len(config["poisson"]["equal_volume_frame_labels"])
    poisson_count = rows * int(config["poisson"]["samples_per_mean_per_frame"])
    seed = int(config["deterministic_seed"])
    poisson = deterministic_uint64(seed, poisson_count, 1).reshape(rows, -1)

    sds_count = int(config["sds"]["case_count"])
    sds_raw = deterministic_uint64(seed, sds_count, 2)
    sds_u = ((sds_raw >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    e_min = float(config["sds"]["epsilon_min"])
    e_max = float(config["sds"]["epsilon_max"])
    epsilon = np.exp(np.log(e_min) + sds_u * (np.log(e_max) - np.log(e_min)))

    mond_count = int(config["mond"]["case_count"])
    mass_raw = deterministic_uint64(seed, mond_count, 3)
    radius_raw = deterministic_uint64(seed, mond_count, 4)
    mass_u = ((mass_raw >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    radius_u = ((radius_raw >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    m_min = float(config["mond"]["mass_min"])
    m_max = float(config["mond"]["mass_max"])
    mass = np.exp(np.log(m_min) + mass_u * (np.log(m_max) - np.log(m_min)))
    k_min = float(config["mond"]["radius_over_r0_min"])
    k_max = float(config["mond"]["radius_over_r0_max"])
    multiplier = np.exp(np.log(k_min) + radius_u * (np.log(k_max) - np.log(k_min)))
    radius = multiplier * np.sqrt(mass)
    return {
        "poisson_random": np.ascontiguousarray(poisson),
        "sds_epsilon": np.ascontiguousarray(epsilon, dtype="<f8"),
        "mond_mass": np.ascontiguousarray(mass, dtype="<f8"),
        "mond_radius": np.ascontiguousarray(radius, dtype="<f8"),
    }


def _poisson_cdf(config: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, Any]]]:
    means = [Fraction(value) for value in config["poisson"]["means"]]
    frames = config["poisson"]["equal_volume_frame_labels"]
    maximum = int(config["poisson"]["maximum_count"])
    rows = []
    high_precision = []
    mpmath.mp.dps = 80
    for mean in means:
        mu = float(mean)
        probabilities = [math.exp(-mu)]
        for count in range(1, maximum + 1):
            probabilities.append(probabilities[-1] * mu / count)
        cdf = np.cumsum(probabilities, dtype=np.float64)
        exact_cdf = []
        term = mpmath.exp(-mpmath.mpf(mean.numerator) / mean.denominator)
        total = term
        exact_cdf.append(total)
        for count in range(1, maximum + 1):
            term *= mpmath.mpf(mean.numerator) / (mean.denominator * count)
            total += term
            exact_cdf.append(total)
        error = max(abs(float(exact_cdf[index]) - cdf[index]) for index in range(maximum + 1))
        high_precision.append(
            {
                "mean": str(mean),
                "maximum_cdf_float64_error": error,
                "tail_probability_high_precision": str(1 - exact_cdf[-1]),
            }
        )
        for _ in frames:
            rows.append(cdf)
    return np.ascontiguousarray(rows, dtype="<f8"), high_precision


def _cpu_poisson(randoms: np.ndarray, cdf: np.ndarray) -> np.ndarray:
    uniforms = ((randoms >> np.uint64(11)).astype(np.float64) + 0.5) * 2.0**-53
    output = np.empty(randoms.shape, dtype=np.uint8)
    for row in range(len(randoms)):
        output[row] = np.searchsorted(cdf[row], uniforms[row], side="left").astype(np.uint8)
    return output


def _poisson_statistics(
    output: np.ndarray, config: Mapping[str, Any], high_precision: list[dict[str, Any]]
) -> dict[str, Any]:
    means = [float(Fraction(value)) for value in config["poisson"]["means"]]
    frames = list(config["poisson"]["equal_volume_frame_labels"])
    samples = output.shape[1]
    rows = []
    all_closed = True
    for mean_index, mu in enumerate(means):
        pair = []
        for frame_index, frame in enumerate(frames):
            row_index = mean_index * len(frames) + frame_index
            values = output[row_index].astype(np.float64)
            sample_mean = float(np.mean(values))
            variance = float(np.var(values, ddof=1))
            mean_z = abs(sample_mean - mu) / math.sqrt(mu / samples)
            fano = variance / sample_mean
            closed = (
                mean_z <= float(config["poisson"]["mean_z_limit"])
                and abs(fano - 1.0) <= float(config["poisson"]["fano_absolute_tolerance"])
            )
            all_closed &= closed
            record = {
                "mean": mu,
                "frame": frame,
                "sample_mean": sample_mean,
                "sample_variance": variance,
                "mean_z": mean_z,
                "fano_factor": fano,
                "statistical_control_closed": closed,
            }
            rows.append(record)
            pair.append(record)
        equal_z = abs(pair[0]["sample_mean"] - pair[1]["sample_mean"]) / math.sqrt(2 * mu / samples)
        pair[0]["equal_four_volume_pair_mean_z"] = equal_z
        pair[1]["equal_four_volume_pair_mean_z"] = equal_z
        equal_closed = equal_z <= float(config["poisson"]["equal_volume_z_limit"])
        pair[0]["equal_four_volume_control_closed"] = equal_closed
        pair[1]["equal_four_volume_control_closed"] = equal_closed
        all_closed &= equal_closed
    precision_bound = float(config["poisson"]["cdf_cpu_high_precision_error_bound"])
    precision_closed = all(row["maximum_cdf_float64_error"] <= precision_bound for row in high_precision)
    return {
        "decision": "synthetic_statistical_control_closed" if all_closed and precision_closed else "reject",
        "rows": rows,
        "cpu_high_precision_cdf_controls": high_precision,
        "cdf_error_bound": precision_bound,
        "all_statistical_controls_closed": all_closed,
        "all_cpu_high_precision_controls_closed": precision_closed,
        "interpretation": (
            "tests deterministic sampling consequences of an assumed homogeneous Poisson rate; "
            "does not test the paper's QED, actualization, stationarity, or ontology premise"
        ),
    }


def _cpu_sds(epsilon: np.ndarray, iterations: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    black_lo = epsilon.copy()
    black_hi = np.full_like(epsilon, 0.5)
    for _ in range(iterations):
        mid = 0.5 * (black_lo + black_hi)
        f_mid = 1.0 - mid * mid - epsilon / mid
        positive = f_mid > 0.0
        black_hi = np.where(positive, mid, black_hi)
        black_lo = np.where(positive, black_lo, mid)
    black = 0.5 * (black_lo + black_hi)
    cosm_lo = np.full_like(epsilon, 0.5)
    cosm_hi = np.ones_like(epsilon)
    for _ in range(iterations):
        mid = 0.5 * (cosm_lo + cosm_hi)
        f_mid = 1.0 - mid * mid - epsilon / mid
        positive = f_mid > 0.0
        cosm_lo = np.where(positive, mid, cosm_lo)
        cosm_hi = np.where(positive, cosm_hi, mid)
    cosm = 0.5 * (cosm_lo + cosm_hi)
    residual = np.maximum(
        np.abs(1.0 - black * black - epsilon / black),
        np.abs(1.0 - cosm * cosm - epsilon / cosm),
    )
    return black, cosm, residual


def _cpu_mond(mass: np.ndarray, radius: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    exact = mass / radius * (1.0 + np.sqrt(1.0 + 1.0 / mass))
    asymptotic = np.sqrt(mass) / radius
    v4 = (asymptotic * radius) ** 2
    return exact, asymptotic, v4


class _NvmlSampler:
    def __init__(self, device: int, interval: float) -> None:
        self.device = device
        self.interval = interval
        self.samples: list[dict[str, int]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            try:
                import pynvml

                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(self.device)
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
            except Exception as error:  # noqa: BLE001 - optional device telemetry
                self.error = f"{type(error).__name__}: {error}"

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if not self.samples:
            return {"available": False, "sample_count": 0, "reason": self.error or "no samples"}
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
            "power_watts_max": max(row["power_milliwatts"] for row in self.samples) / 1000,
            "counter_scope": (
                "device-wide NVML samples during synchronized kernels; may include concurrent "
                "processes and are not a lane-only or sustained-capacity claim"
            ),
        }


def _normalization_gate(predecessor: Mapping[str, Any]) -> dict[str, Any]:
    contract = next(
        row for row in predecessor["formula_contracts"] if row["contract_id"] == "transaction_cosmological_term"
    )
    check = next(
        row for row in predecessor["synthetic_checks"] if row["check_id"] == "transaction_pressure_lambda_identity"
    )
    if check.get("status") != "block" or "needs clarification" not in contract.get("normalization_note", ""):
        raise ValueError("predecessor no longer carries the equation-35 normalization blocker")
    return {
        "decision": "blocked",
        "paper_equation": 35,
        "source_normalization_note": contract["normalization_note"],
        "substitution": "h=2*pi*hbar and l_P^2=G*hbar/c^3",
        "middle_expression_coefficient_in_lP2_q_units": "8*pi^2",
        "printed_final_coefficient_in_lP2_q_units": "4*pi^2",
        "exact_ratio_middle_to_printed": "2",
        "lambda_values_emitted": 0,
        "lambda_gpu_evaluations": 0,
        "cosmology_locked_a0_values_emitted": 0,
        "first_missing_premise": "authoritative_equation_35_h_versus_hbar_normalization_clarification",
        "interpretation": "the campaign records both source-bound branches and selects neither",
    }


def _device(cp: Any) -> dict[str, Any]:
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"])
    return {
        "backend": "cupy_raw_cuda",
        "cupy_version": cp.__version__,
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_index": int(device.id),
        "device_name": name,
        "compute_capability": f"{props['major']}.{props['minor']}",
        "total_global_memory_mib": int(props["totalGlobalMem"] // (1024 * 1024)),
    }


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    config, predecessor, root = _load(config_path)
    inputs = deterministic_inputs(config)
    cdf, high_precision = _poisson_cdf(config)
    cpu_poisson = _cpu_poisson(inputs["poisson_random"], cdf)
    poisson = _poisson_statistics(cpu_poisson, config, high_precision)
    if poisson["decision"] == "reject":
        raise ValueError("deterministic Poisson statistical controls rejected")

    sds_iterations = int(config["sds"]["bisection_iterations"])
    cpu_black, cpu_cosm, cpu_sds_residual = _cpu_sds(inputs["sds_epsilon"], sds_iterations)
    cpu_mond_exact, cpu_mond_asymptotic, cpu_btfr = _cpu_mond(
        inputs["mond_mass"], inputs["mond_radius"]
    )
    normalization = _normalization_gate(predecessor)

    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CUDA consequence campaign unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("CUDA consequence campaign found no CUDA device")
    device = _device(cp)
    poisson_kernel = cp.RawKernel(POISSON_KERNEL, "poisson_inverse")
    sds_kernel = cp.RawKernel(SDS_KERNEL, "sds_roots")
    mond_kernel = cp.RawKernel(MOND_KERNEL, "mond_consequences")
    threads = 256

    d_random = cp.asarray(inputs["poisson_random"].reshape(-1))
    d_cdf = cp.asarray(cdf.reshape(-1))
    d_poisson = cp.empty(d_random.size, dtype=cp.uint8)
    d_epsilon = cp.asarray(inputs["sds_epsilon"])
    d_black = cp.empty_like(d_epsilon)
    d_cosm = cp.empty_like(d_epsilon)
    d_sds_residual = cp.empty_like(d_epsilon)
    d_sds_domain = cp.empty(d_epsilon.size, dtype=cp.uint8)
    d_mass = cp.asarray(inputs["mond_mass"])
    d_radius = cp.asarray(inputs["mond_radius"])
    d_mond_exact = cp.empty_like(d_mass)
    d_mond_asymptotic = cp.empty_like(d_mass)
    d_btfr = cp.empty_like(d_mass)
    d_mond_domain = cp.empty(d_mass.size, dtype=cp.uint8)

    def dispatch() -> None:
        poisson_kernel(
            ((d_random.size + threads - 1) // threads,),
            (threads,),
            (
                d_random,
                d_cdf,
                np.int32(cdf.shape[0]),
                np.int32(cdf.shape[1] and inputs["poisson_random"].shape[1]),
                np.int32(int(config["poisson"]["maximum_count"])),
                d_poisson,
            ),
        )
        sds_kernel(
            ((d_epsilon.size + threads - 1) // threads,),
            (threads,),
            (
                d_epsilon,
                np.int32(d_epsilon.size),
                np.int32(sds_iterations),
                d_black,
                d_cosm,
                d_sds_residual,
                d_sds_domain,
            ),
        )
        mond_kernel(
            ((d_mass.size + threads - 1) // threads,),
            (threads,),
            (
                d_mass,
                d_radius,
                np.int32(d_mass.size),
                d_mond_exact,
                d_mond_asymptotic,
                d_btfr,
                d_mond_domain,
            ),
        )

    warmups = int(config["gpu_warmup_repetitions"])
    repetitions = int(config["gpu_measured_repetitions"])
    for _ in range(warmups):
        dispatch()
    cp.cuda.Device().synchronize()
    sampler = _NvmlSampler(device["device_index"], float(config["utilization_sample_interval_seconds"]))
    sampler.start()
    started = time.perf_counter()
    for _ in range(repetitions):
        dispatch()
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    utilization = sampler.stop()

    gpu_poisson = cp.asnumpy(d_poisson).reshape(inputs["poisson_random"].shape)
    gpu_black = cp.asnumpy(d_black)
    gpu_cosm = cp.asnumpy(d_cosm)
    gpu_sds_residual = cp.asnumpy(d_sds_residual)
    gpu_sds_domain = cp.asnumpy(d_sds_domain)
    gpu_mond_exact = cp.asnumpy(d_mond_exact)
    gpu_mond_asymptotic = cp.asnumpy(d_mond_asymptotic)
    gpu_btfr = cp.asnumpy(d_btfr)
    gpu_mond_domain = cp.asnumpy(d_mond_domain)

    poisson_equal = bool(np.array_equal(gpu_poisson, cpu_poisson))
    sds_error = max(
        float(np.max(np.abs(gpu_black - cpu_black))),
        float(np.max(np.abs(gpu_cosm - cpu_cosm))),
    )
    sds_max_residual = float(np.max(gpu_sds_residual))
    nariai = 2.0 / (3.0 * math.sqrt(3.0))
    negative_epsilon = nariai * 1.001
    negative_stationary_x = (negative_epsilon / 2.0) ** (1.0 / 3.0)
    negative_maximum_f = 1.0 - negative_stationary_x**2 - negative_epsilon / negative_stationary_x
    sds = {
        "decision": "synthetic_domain_control_closed",
        "normalized_metric": "f(x)=1-x^2-epsilon/x with x=r/R0 and epsilon=Rs/R0",
        "case_count": len(gpu_black),
        "gpu_cpu_max_root_error": sds_error,
        "gpu_max_root_residual": sds_max_residual,
        "all_static_patch_domains_valid": bool(np.all(gpu_sds_domain == 1)),
        "all_cosmological_horizons_contracted": bool(np.all(gpu_cosm < 1.0)),
        "nariai_epsilon": nariai,
        "above_nariai_negative_control_epsilon": negative_epsilon,
        "above_nariai_stationary_maximum_f": negative_maximum_f,
        "above_nariai_two_horizon_domain_rejected": negative_maximum_f < 0.0,
        "scope": "dimensionless SdS root and static-domain consequences only",
    }
    if (
        sds_error > float(config["sds"]["gpu_cpu_error_bound"])
        or sds_max_residual > float(config["sds"]["root_residual_bound"])
        or not sds["all_static_patch_domains_valid"]
        or not sds["above_nariai_two_horizon_domain_rejected"]
    ):
        raise ValueError("SdS CUDA consequence controls rejected")

    mond_error = max(
        float(np.max(np.abs(gpu_mond_exact - cpu_mond_exact))),
        float(np.max(np.abs(gpu_mond_asymptotic - cpu_mond_asymptotic))),
        float(np.max(np.abs(gpu_btfr - cpu_btfr))),
    )
    asymptotic_relative = np.abs(gpu_mond_exact - gpu_mond_asymptotic) / gpu_mond_asymptotic
    btfr_relative = np.abs(gpu_btfr - inputs["mond_mass"]) / inputs["mond_mass"]
    mond = {
        "decision": "conditional_synthetic_asymptote_control_closed",
        "normalization": "G=c=a0=1 with a0 independent; no equation-35 or cosmology propagation",
        "case_count": len(gpu_mond_exact),
        "gpu_cpu_max_absolute_error": mond_error,
        "maximum_equation62_to_deep_mond_relative_difference": float(np.max(asymptotic_relative)),
        "maximum_btfr_v4_equals_mass_relative_residual": float(np.max(btfr_relative)),
        "all_radii_above_r0_and_weak_potential": bool(np.all(gpu_mond_domain == 1)),
        "btfr_relation": "v^4=G*M*a0 conditional on independently supplied positive a0",
        "galaxy_geometry_or_data_tested": False,
        "interpretation": "tests equations 62, 65, 68 and 69 only; not a galaxy, MOND, or theory pass",
    }
    if (
        mond_error > float(config["mond"]["gpu_cpu_error_bound"])
        or mond["maximum_equation62_to_deep_mond_relative_difference"]
        > float(config["mond"]["asymptotic_relative_error_bound"])
        or not mond["all_radii_above_r0_and_weak_potential"]
    ):
        raise ValueError("MOND CUDA consequence controls rejected")

    poisson_values = int(gpu_poisson.size)
    sds_values = int(gpu_black.size)
    mond_values = int(gpu_mond_exact.size)
    measured_evaluations = repetitions * (poisson_values + sds_values + mond_values)
    input_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in inputs.items()
    }
    predecessor_path = root / config["predecessor"]["path"]
    profile_path = root / config["resource_profile"]
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "scope": "source-bound synthetic numerical consequences of the Kastner-Schlatter intake",
        "predecessor_binding": {
            "path": config["predecessor"]["path"],
            "file_sha256": _file_sha(predecessor_path),
            "content_sha256": predecessor["content_sha256"],
            "primary_pdf_sha256": predecessor["source_binding"]["official_pdf_sha256"],
        },
        "deterministic_manifest": {
            "seed": int(config["deterministic_seed"]),
            "generator": "vectorized_splitmix64_v1",
            "input_sha256": input_hashes,
            "manifest_root_sha256": _sha(input_hashes),
        },
        "poisson_four_volume_control": poisson,
        "equation_35_normalization_gate": normalization,
        "sds_root_domain_control": sds,
        "mond_btfr_control": mond,
        "gpu_cpu_bindings": {
            "poisson_output_byte_equal": poisson_equal,
            "poisson_cpu_sha256": hashlib.sha256(cpu_poisson.tobytes()).hexdigest(),
            "poisson_gpu_sha256": hashlib.sha256(gpu_poisson.tobytes()).hexdigest(),
            "sds_cpu_max_root_residual": float(np.max(cpu_sds_residual)),
            "sds_gpu_cpu_max_root_error": sds_error,
            "mond_gpu_cpu_max_absolute_error": mond_error,
        },
        "counts": {
            "poisson_means": len(config["poisson"]["means"]),
            "lorentz_equal_volume_frames": len(config["poisson"]["equal_volume_frame_labels"]),
            "poisson_samples": poisson_values,
            "sds_cases": sds_values,
            "mond_cases": mond_values,
            "gpu_warmup_repetitions": warmups,
            "gpu_measured_repetitions": repetitions,
            "gpu_kernel_dispatches": 3 * (warmups + repetitions),
            "gpu_measured_consequence_evaluations": measured_evaluations,
            "lambda_values_emitted": 0,
            "cosmology_locked_a0_values_emitted": 0,
            "observational_records_accessed": 0,
            "formal_or_theory_passes_inferred": 0,
            "ontology_passes_inferred": 0,
            "paid_llm_calls": 0,
        },
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "gpu_measured_wall_seconds": elapsed,
            "gpu_consequence_evaluations_per_second": measured_evaluations / elapsed,
            "gpu_array_bytes": sum(
                int(array.nbytes)
                for array in (
                    d_random,
                    d_cdf,
                    d_poisson,
                    d_epsilon,
                    d_black,
                    d_cosm,
                    d_sds_residual,
                    d_sds_domain,
                    d_mass,
                    d_radius,
                    d_mond_exact,
                    d_mond_asymptotic,
                    d_btfr,
                    d_mond_domain,
                )
            ),
            "utilization": utilization,
            "timing_scope": "single local run; nondeterministic and not a sustained throughput claim",
        },
        "decision": "synthetic_consequences_executed_equation_35_and_physics_claims_blocked",
        "first_blocker": normalization["first_missing_premise"],
        "limitations": [
            "Poisson sampling assumes rather than derives a homogeneous transaction rate",
            "Lorentz equal-volume checks are distributional implementation controls, not a covariance proof",
            "equation 35 is not propagated because the source-bound normalization differs by factor two",
            "SdS tests cover normalized static weak-domain roots, not spacetime emergence or global geometry",
            "MOND and BTFR tests use an independent normalized a0 and no galaxy geometry or data",
            "no fundamental action, variational equations, ontology, theory validity, or observation is tested",
        ],
        "synthetic_only": True,
        "observations_opened": False,
        "ontology_pass": False,
        "theory_pass": False,
        "formal_pass": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    source_file = root / "src/sigma_theory_compiler/kastner_schlatter_cuda_consequence_campaign.py"
    test_file = root / "tests/test_kastner_schlatter_cuda_consequence_campaign.py"
    result["source_bindings"] = {
        "config": {
            "path": config_path.resolve().relative_to(root).as_posix(),
            "file_sha256": _file_sha(config_path),
        },
        "source": {
            "path": source_file.relative_to(root).as_posix(),
            "file_sha256": _file_sha(source_file),
        },
        "resource_profile": {
            "path": config["resource_profile"],
            "file_sha256": _file_sha(profile_path),
        },
    }
    if test_file.is_file():
        result["source_bindings"]["test"] = {
            "path": test_file.relative_to(root).as_posix(),
            "file_sha256": _file_sha(test_file),
        }
    result["content_sha256"] = _content_sha(result)
    validate_campaign(result, config_path)
    return result


def validate_campaign(document: Mapping[str, Any], config_path: str | Path) -> None:
    config_path = Path(config_path)
    config, predecessor, root = _load(config_path)
    if document.get("schema_version") != SCHEMA or document.get("content_sha256") != _content_sha(document):
        raise ValueError("consequence campaign content hash mismatch")
    seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "ontology_pass": False,
        "theory_pass": False,
        "formal_pass": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(document.get(key) != value for key, value in seals.items()):
        raise ValueError("consequence campaign data or claim seal changed")
    gate = document.get("equation_35_normalization_gate", {})
    if (
        gate.get("decision") != "blocked"
        or gate.get("exact_ratio_middle_to_printed") != "2"
        or gate.get("lambda_values_emitted") != 0
        or gate.get("cosmology_locked_a0_values_emitted") != 0
    ):
        raise ValueError("equation-35 normalization did not fail closed")
    counts = document.get("counts", {})
    if (
        counts.get("lambda_values_emitted") != 0
        or counts.get("cosmology_locked_a0_values_emitted") != 0
        or counts.get("observational_records_accessed") != 0
        or counts.get("formal_or_theory_passes_inferred") != 0
        or counts.get("ontology_passes_inferred") != 0
        or counts.get("paid_llm_calls") != 0
    ):
        raise ValueError("consequence campaign counters violate fail-closed scope")
    if document.get("gpu_cpu_bindings", {}).get("poisson_output_byte_equal") is not True:
        raise ValueError("Poisson CUDA output did not match CPU")
    if document.get("poisson_four_volume_control", {}).get("all_statistical_controls_closed") is not True:
        raise ValueError("Poisson statistical controls did not close")
    if document.get("sds_root_domain_control", {}).get("decision") != "synthetic_domain_control_closed":
        raise ValueError("SdS consequence control did not close")
    if document.get("mond_btfr_control", {}).get("decision") != "conditional_synthetic_asymptote_control_closed":
        raise ValueError("MOND consequence control did not close")
    predecessor_binding = document.get("predecessor_binding", {})
    if (
        predecessor_binding.get("file_sha256") != config["predecessor"]["file_sha256"]
        or predecessor_binding.get("content_sha256") != predecessor["content_sha256"]
        or predecessor_binding.get("primary_pdf_sha256")
        != predecessor["source_binding"]["official_pdf_sha256"]
    ):
        raise ValueError("consequence predecessor binding mismatch")
    rebuilt = deterministic_inputs(config)
    hashes = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in rebuilt.items()
    }
    if document.get("deterministic_manifest", {}).get("input_sha256") != hashes:
        raise ValueError("consequence deterministic input manifest mismatch")
    for binding in document.get("source_bindings", {}).values():
        path = root / binding["path"]
        if binding.get("file_sha256") != _file_sha(path):
            raise ValueError("consequence local source binding mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    document = build_campaign(args.config)
    Path(args.output).write_text(_canonical(document) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
