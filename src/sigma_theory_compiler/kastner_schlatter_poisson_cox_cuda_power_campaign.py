"""Synthetic CUDA power study for the registered Poisson-versus-Cox witness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .kastner_schlatter_cuda_consequence_campaign import _device, _NvmlSampler

SCHEMA = "sigma-kastner-schlatter-poisson-cox-cuda-power-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-poisson-cox-cuda-power-config-1.0"
METRICS = ("log_likelihood_ratio", "dispersion", "void_excess", "factorial_excess")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _fraction(text: str) -> Fraction:
    return Fraction(text)


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    root = config_path.resolve().parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported Poisson-Cox CUDA power config")
    seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "scientific_test_pass_allowed": False,
        "paper_or_qed_inference_allowed": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(config.get(key) != value for key, value in seals.items()):
        raise ValueError("scientific or data seal changed")
    binding = config["predecessor"]
    predecessor_path = (root / binding["path"]).resolve()
    try:
        predecessor_path.relative_to(root)
    except ValueError as error:
        raise ValueError("predecessor path escapes repository") from error
    if _file_sha(predecessor_path) != binding["file_sha256"]:
        raise ValueError("predecessor file hash mismatch")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if (
        predecessor.get("content_sha256") != binding["content_sha256"]
        or _content_sha(predecessor) != binding["content_sha256"]
    ):
        raise ValueError("predecessor content hash mismatch")
    witness = predecessor.get("exact_mixed_poisson_control", {})
    if (
        predecessor.get("decision")
        != "stationary_conditional_poisson_interface_closed_dynamic_derivation_blocked"
        or witness.get("intensity_support") != ["1", "3"]
        or witness.get("probabilities") != ["1/2", "1/2"]
        or witness.get("Fano_factor") != "3/2"
        or predecessor.get("counts", {}).get("observational_or_theory_passes") != 0
        or predecessor.get("data_seals", {}).get("observations_opened") is not False
    ):
        raise ValueError("registered Poisson-Cox witness boundary changed")
    if int(config["replicates"]) < 32 or not 0 < float(config["empirical_size_alpha"]) < 0.5:
        raise ValueError("invalid finite-sample design")
    return config, predecessor, root


def exact_witness_certificate() -> dict[str, Any]:
    support = (Fraction(1), Fraction(3))
    mean = sum(support, Fraction(0)) / 2
    second = sum((value * value for value in support), Fraction(0)) / 2
    latent_variance = second - mean * mean
    count_variance = mean + latent_variance
    factorial = second
    return {
        "conditional_rate_support": [str(value) for value in support],
        "weights": ["1/2", "1/2"],
        "mean_count": str(mean),
        "latent_rate_variance": str(latent_variance),
        "count_variance": str(count_variance),
        "fano_factor": str(count_variance / mean),
        "factorial_second_moment": str(factorial),
        "poisson_null_factorial_second_moment": str(mean * mean),
        "factorial_excess": str(factorial - mean * mean),
        "void_probability_symbolic": "(exp(-1)+exp(-3))/2",
    }


def analytic_moments(mu: Fraction, delta: Fraction) -> dict[str, Any]:
    latent_variance = mu * mu * delta * delta
    return {
        "mean": str(mu),
        "conditional_rates": [str(mu * (1 - delta)), str(mu * (1 + delta))],
        "latent_rate_variance": str(latent_variance),
        "count_variance": str(mu + latent_variance),
        "fano_factor": str(1 + mu * delta * delta),
        "factorial_second_moment": str(mu * mu * (1 + delta * delta)),
        "factorial_excess": str(latent_variance),
        "void_probability_formula": "exp(-mu)*cosh(mu*delta)",
    }


def _scenario_seed(master_seed: int, *parts: object) -> int:
    token = ":".join((str(master_seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "little")


def _gpu_statistics(cp: Any, counts: Any, mu: float, delta: float) -> dict[str, Any]:
    data = counts.astype(cp.float64, copy=False)
    sample_mean = cp.mean(data, axis=1)
    dispersion = cp.var(data, axis=1, ddof=1) / cp.maximum(sample_mean, 1e-300)
    void_excess = cp.mean(data == 0.0, axis=1) - math.exp(-mu)
    factorial_excess = cp.mean(data * (data - 1.0), axis=1) - mu * mu
    low = mu * (1.0 - delta)
    high = mu * (1.0 + delta)
    log_mix = cp.logaddexp(
        data * math.log(low) - low,
        data * math.log(high) - high,
    ) - math.log(2.0)
    log_null = data * math.log(mu) - mu
    llr = cp.sum(log_mix - log_null, axis=1)
    return {
        "log_likelihood_ratio": llr,
        "dispersion": dispersion,
        "void_excess": void_excess,
        "factorial_excess": factorial_excess,
    }


def _cpu_statistics(counts: np.ndarray, mu: float, delta: float) -> dict[str, np.ndarray]:
    data = counts.astype(np.float64, copy=False)
    sample_mean = np.mean(data, axis=1)
    dispersion = np.var(data, axis=1, ddof=1) / np.maximum(sample_mean, 1e-300)
    void_excess = np.mean(data == 0.0, axis=1) - math.exp(-mu)
    factorial_excess = np.mean(data * (data - 1.0), axis=1) - mu * mu
    low = mu * (1.0 - delta)
    high = mu * (1.0 + delta)
    log_mix = np.logaddexp(
        data * math.log(low) - low,
        data * math.log(high) - high,
    ) - math.log(2.0)
    log_null = data * math.log(mu) - mu
    return {
        "log_likelihood_ratio": np.sum(log_mix - log_null, axis=1),
        "dispersion": dispersion,
        "void_excess": void_excess,
        "factorial_excess": factorial_excess,
    }


def _hash_update(digest: Any, label: str, values: np.ndarray) -> None:
    digest.update(label.encode())
    digest.update(np.ascontiguousarray(values).tobytes())


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, _predecessor, root = _load(config_path)
    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CUDA unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("no CUDA device")
    device = _device(cp)
    if config["required_device_name_fragment"] not in device["device_name"]:
        raise RuntimeError("required RTX 5090 device not present")

    exposures = [_fraction(value) for value in config["exposure_four_volumes"]]
    intensities = [_fraction(value) for value in config["intensities"]]
    deltas = [_fraction(value) for value in config["symmetric_mixing_deltas"]]
    sample_sizes = [int(value) for value in config["sample_sizes"]]
    replicates = int(config["replicates"])
    alpha = float(config["empirical_size_alpha"])
    master_seed = int(config["deterministic_seed"])
    abs_bound = float(config["gpu_cpu_absolute_error_bound"])
    rel_bound = float(config["gpu_cpu_relative_error_bound"])
    sampler = _NvmlSampler(
        device["device_index"], float(config["utilization_sample_interval_seconds"])
    )
    sample_digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    max_abs_error = 0.0
    max_rel_error = 0.0
    decisions_equal = True
    statistic_vectors_checked = 0
    replicate_metric_values_checked = 0
    count_values = 0
    scenario_count = 0

    sampler.start()
    started = time.perf_counter()
    for exposure in exposures:
        for intensity in intensities:
            mu_fraction = exposure * intensity
            mu = float(mu_fraction)
            for sample_size in sample_sizes:
                calibration_seed = _scenario_seed(
                    master_seed, exposure, intensity, sample_size, "null_calibration"
                )
                null_seed = _scenario_seed(
                    master_seed, exposure, intensity, sample_size, "null_evaluation"
                )
                calibration_rng = cp.random.RandomState(calibration_seed)
                null_rng = cp.random.RandomState(null_seed)
                calibration_counts_gpu = calibration_rng.poisson(
                    mu, size=(replicates, sample_size), dtype=cp.int32
                )
                null_counts_gpu = null_rng.poisson(
                    mu, size=(replicates, sample_size), dtype=cp.int32
                )
                count_values += 2 * replicates * sample_size
                calibration_counts = cp.asnumpy(calibration_counts_gpu)
                null_counts = cp.asnumpy(null_counts_gpu)
                _hash_update(
                    sample_digest,
                    f"{exposure}:{intensity}:{sample_size}:null_calibration",
                    calibration_counts,
                )
                _hash_update(
                    sample_digest,
                    f"{exposure}:{intensity}:{sample_size}:null",
                    null_counts,
                )
                for delta_fraction in deltas:
                    delta = float(delta_fraction)
                    alt_seed = _scenario_seed(
                        master_seed, exposure, intensity, sample_size, delta_fraction, "cox"
                    )
                    alt_rng = cp.random.RandomState(alt_seed)
                    selectors = alt_rng.randint(
                        0, 2, size=(replicates, sample_size), dtype=cp.int32
                    )
                    alt_rates = mu * (1.0 + delta * (2.0 * selectors - 1.0))
                    alt_counts_gpu = alt_rng.poisson(alt_rates, dtype=cp.int32)
                    calibration_gpu = _gpu_statistics(cp, calibration_counts_gpu, mu, delta)
                    null_gpu = _gpu_statistics(cp, null_counts_gpu, mu, delta)
                    alt_gpu = _gpu_statistics(cp, alt_counts_gpu, mu, delta)
                    cp.cuda.Device().synchronize()
                    alt_counts = cp.asnumpy(alt_counts_gpu)
                    calibration_gpu_host = {
                        key: cp.asnumpy(value) for key, value in calibration_gpu.items()
                    }
                    null_gpu_host = {key: cp.asnumpy(value) for key, value in null_gpu.items()}
                    alt_gpu_host = {key: cp.asnumpy(value) for key, value in alt_gpu.items()}
                    calibration_cpu = _cpu_statistics(calibration_counts, mu, delta)
                    null_cpu = _cpu_statistics(null_counts, mu, delta)
                    alt_cpu = _cpu_statistics(alt_counts, mu, delta)
                    _hash_update(
                        sample_digest,
                        f"{exposure}:{intensity}:{sample_size}:{delta_fraction}:alt",
                        alt_counts,
                    )
                    thresholds: dict[str, float] = {}
                    empirical_size: dict[str, float] = {}
                    empirical_power: dict[str, float] = {}
                    empirical_separation: dict[str, dict[str, float]] = {}
                    for metric in METRICS:
                        for gpu_values, cpu_values in (
                            (calibration_gpu_host[metric], calibration_cpu[metric]),
                            (null_gpu_host[metric], null_cpu[metric]),
                            (alt_gpu_host[metric], alt_cpu[metric]),
                        ):
                            difference = np.abs(gpu_values - cpu_values)
                            scale = np.maximum(np.abs(cpu_values), 1.0)
                            max_abs_error = max(max_abs_error, float(np.max(difference)))
                            max_rel_error = max(max_rel_error, float(np.max(difference / scale)))
                            statistic_vectors_checked += 1
                            replicate_metric_values_checked += int(cpu_values.size)
                        threshold = float(
                            np.quantile(calibration_cpu[metric], 1.0 - alpha, method="higher")
                        )
                        decision_guard = float(config["decision_numerical_guard_multiplier"]) * max(
                            abs_bound, rel_bound * max(abs(threshold), 1.0)
                        )
                        guarded_threshold = threshold + decision_guard
                        cpu_null_decision = null_cpu[metric] > guarded_threshold
                        cpu_alt_decision = alt_cpu[metric] > guarded_threshold
                        gpu_null_decision = null_gpu_host[metric] > guarded_threshold
                        gpu_alt_decision = alt_gpu_host[metric] > guarded_threshold
                        decisions_equal = (
                            decisions_equal
                            and np.array_equal(cpu_null_decision, gpu_null_decision)
                            and np.array_equal(cpu_alt_decision, gpu_alt_decision)
                        )
                        thresholds[metric] = guarded_threshold
                        empirical_size[metric] = float(np.mean(cpu_null_decision))
                        empirical_power[metric] = float(np.mean(cpu_alt_decision))
                        null_metric_mean = float(np.mean(null_cpu[metric]))
                        alternative_metric_mean = float(np.mean(alt_cpu[metric]))
                        empirical_separation[metric] = {
                            "null_mean": null_metric_mean,
                            "alternative_mean": alternative_metric_mean,
                            "alternative_minus_null": alternative_metric_mean - null_metric_mean,
                        }
                    rows.append(
                        {
                            "exposure_four_volume": str(exposure),
                            "intensity": str(intensity),
                            "null_mean": str(mu_fraction),
                            "mixing_delta": str(delta_fraction),
                            "sample_size": sample_size,
                            "null_calibration_seed": calibration_seed,
                            "null_evaluation_seed": null_seed,
                            "alternative_seed": alt_seed,
                            "analytic": analytic_moments(mu_fraction, delta_fraction),
                            "upper_tail_thresholds": thresholds,
                            "empirical_null_rejection_rates": empirical_size,
                            "empirical_alternative_detection_rates": empirical_power,
                            "empirical_metric_separation": empirical_separation,
                        }
                    )
                    count_values += replicates * sample_size
                    scenario_count += 1
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    utilization = sampler.stop()
    if max_abs_error > abs_bound and max_rel_error > rel_bound:
        raise ValueError("GPU/CPU statistic error bounds exceeded")
    if not decisions_equal:
        raise ValueError("GPU/CPU rejection decisions differ")

    witness_rows = [row for row in rows if row["null_mean"] == "2" and row["mixing_delta"] == "1/2"]
    if len(witness_rows) != 12:
        raise ValueError("registered 1/3-rate witness grid coverage changed")
    exact = exact_witness_certificate()
    if exact["fano_factor"] != "3/2" or exact["factorial_excess"] != "1":
        raise ValueError("exact witness sentinel failed")

    source_path = (
        root / "src/sigma_theory_compiler/kastner_schlatter_poisson_cox_cuda_power_campaign.py"
    )
    test_path = root / "tests/test_kastner_schlatter_poisson_cox_cuda_power_campaign.py"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "registered_poisson_cox_witness": config["predecessor"],
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": source_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(source_path),
            },
            "test": {
                "path": test_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(test_path),
            },
        },
        "registered_witness_exact_sentinel": exact,
        "covariant_count_surrogate_contract": {
            "mean_definition": "mu=q0*Vol_g(B)",
            "exposure_interpretation": "declared invariant four-volume surrogate only",
            "null": "N conditional on fixed mu is Poisson(mu)",
            "alternative": "equal-weight Cox mixture with conditional rates mu*(1-delta) and mu*(1+delta)",
            "registered_witness_grid_point": "mu=2, delta=1/2 gives conditional rates 1 and 3",
            "action_derived": False,
            "operational_transaction_event_defined": False,
        },
        "design": {
            "exposure_four_volumes": config["exposure_four_volumes"],
            "intensities": config["intensities"],
            "symmetric_mixing_deltas": config["symmetric_mixing_deltas"],
            "sample_sizes": sample_sizes,
            "replicates_per_calibration_evaluation_and_alternative": replicates,
            "empirical_size_alpha": alpha,
            "threshold_rule": "strict upper tail beyond an independent null-calibration higher-quantile plus a declared four-times GPU/CPU numerical guard",
            "metrics": list(METRICS),
        },
        "scenario_results": rows,
        "gpu_cpu_crosscheck": {
            "scope": "every statistic for every null and alternative replicate recomputed on CPU from identical GPU-generated counts",
            "statistic_vectors_checked": statistic_vectors_checked,
            "replicate_metric_values_checked": replicate_metric_values_checked,
            "maximum_absolute_error": max_abs_error,
            "maximum_relative_error_scaled_at_one": max_rel_error,
            "absolute_error_bound": abs_bound,
            "relative_error_bound": rel_bound,
            "all_rejection_decisions_byte_equal": decisions_equal,
            "exact_registered_witness_sentinel_passed": True,
        },
        "deterministic_manifest": {
            "master_seed": master_seed,
            "gpu_generator": "cupy_randomstate_xorshift128",
            "scenario_seed_derivation": "little-endian uint32 from first four SHA-256 bytes of colon-delimited design coordinates",
            "realized_count_stream_sha256": sample_digest.hexdigest(),
            "grid_manifest_sha256": _sha(
                {
                    "exposure": config["exposure_four_volumes"],
                    "intensity": config["intensities"],
                    "delta": config["symmetric_mixing_deltas"],
                    "sample_sizes": sample_sizes,
                    "replicates": replicates,
                }
            ),
        },
        "counts": {
            "scenario_cells": scenario_count,
            "registered_witness_scenario_cells": len(witness_rows),
            "gpu_generated_count_values": count_values,
            "null_calibration_replicates": len(exposures)
            * len(intensities)
            * len(sample_sizes)
            * replicates,
            "finite_sample_evaluation_replicate_tests": 2 * scenario_count * replicates,
            "metric_replicate_values_cpu_gpu_checked": replicate_metric_values_checked,
            "observational_records_accessed": 0,
            "readiness_fields_advanced": 0,
            "scientific_tests_passed": 0,
            "paper_or_qed_inferences": 0,
        },
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "wall_seconds": elapsed,
            "gpu_generated_count_values_per_second": count_values / elapsed,
            "utilization": utilization,
            "scope": "device-wide counters during one local run; concurrent GPU work may contribute and lane exclusivity is not claimed",
        },
        "interpretation": {
            "implementation_stress_test": "CUDA generation, reductions, deterministic manifest, and CPU parity",
            "synthetic_falsification_power": "finite-sample separability of a declared Poisson null from the registered symmetric Cox witness family",
            "scientific_or_observational_test": "not performed",
            "non_test": "does not test transaction ontology, action derivation, paper validity, QED validity, or real-event correspondence",
        },
        "decision": "synthetic_poisson_cox_falsification_power_measured_scientific_and_observational_claims_blocked",
        "synthetic_only": True,
        "observations_opened": False,
        "scientific_test_pass": False,
        "observational_test_pass": False,
        "readiness_advanced": False,
        "paper_pass": False,
        "qed_pass": False,
        "theory_pass": False,
        "ontology_pass": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    result["content_sha256"] = _content_sha(result)
    if len((_canonical(result) + "\n").encode()) > int(config["maximum_output_bytes"]):
        raise ValueError("artifact exceeds configured size ceiling")
    return result


def validate_campaign(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, _, root = _load(Path(config_path))
    if result.get("schema_version") != SCHEMA or result.get("content_sha256") != _content_sha(
        result
    ):
        raise ValueError("artifact schema or content hash mismatch")
    if (
        result.get("decision")
        != "synthetic_poisson_cox_falsification_power_measured_scientific_and_observational_claims_blocked"
    ):
        raise ValueError("decision changed")
    if result.get("synthetic_only") is not True:
        raise ValueError("synthetic seal changed")
    for key in (
        "observations_opened",
        "scientific_test_pass",
        "observational_test_pass",
        "readiness_advanced",
        "paper_pass",
        "qed_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        if result.get(key) is not False:
            raise ValueError(f"claim or data seal changed: {key}")
    counts = result.get("counts", {})
    expected_scenarios = (
        len(config["exposure_four_volumes"])
        * len(config["intensities"])
        * len(config["symmetric_mixing_deltas"])
        * len(config["sample_sizes"])
    )
    if counts.get("scenario_cells") != expected_scenarios:
        raise ValueError("scenario count mismatch")
    if any(
        counts.get(key) != 0
        for key in (
            "observational_records_accessed",
            "readiness_fields_advanced",
            "scientific_tests_passed",
            "paper_or_qed_inferences",
        )
    ):
        raise ValueError("forbidden result count advanced")
    if result.get("registered_witness_exact_sentinel") != exact_witness_certificate():
        raise ValueError("exact witness sentinel mismatch")
    crosscheck = result.get("gpu_cpu_crosscheck", {})
    if (
        crosscheck.get("maximum_absolute_error", math.inf) > config["gpu_cpu_absolute_error_bound"]
        and crosscheck.get("maximum_relative_error_scaled_at_one", math.inf)
        > config["gpu_cpu_relative_error_bound"]
    ) or crosscheck.get("all_rejection_decisions_byte_equal") is not True:
        raise ValueError("GPU/CPU crosscheck failed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"{name} binding mismatch")
    predecessor_binding = result["source_bindings"]["registered_poisson_cox_witness"]
    if predecessor_binding != config["predecessor"]:
        raise ValueError("predecessor binding mismatch")
    rows = result.get("scenario_results", [])
    if (
        len(rows) != expected_scenarios
        or sum(row["null_mean"] == "2" and row["mixing_delta"] == "1/2" for row in rows) != 12
    ):
        raise ValueError("scenario grid or witness coverage mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate")
    args = parser.parse_args()
    config_path = Path(args.config)
    if args.validate:
        validate_campaign(json.loads(Path(args.validate).read_text(encoding="utf-8")), config_path)
        return 0
    result = build_campaign(config_path)
    root = config_path.resolve().parents[1]
    output = root / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
