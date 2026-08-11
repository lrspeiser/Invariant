"""GPU-heavy set-indexed point-process falsification controls."""

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

SCHEMA = "sigma-kastner-schlatter-set-indexed-cuda-falsification-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-set-indexed-cuda-falsification-config-1.0"
METRICS = (
    "joint_pgf_laplace_residual",
    "cross_factorial_excess",
    "union_dispersion",
    "marginal_dispersion_deviation",
)


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _fraction(value: str | int) -> Fraction:
    return Fraction(value)


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    root = config_path.resolve().parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported set-indexed CUDA config")
    seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "scientific_test_pass_allowed": False,
        "readiness_advance_allowed": False,
        "paper_or_qed_inference_allowed": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(config.get(key) != expected for key, expected in seals.items()):
        raise ValueError("scientific or data seal changed")
    loaded: dict[str, dict[str, Any]] = {}
    for name, binding in config["predecessors"].items():
        path = (root / binding["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} path escapes repository") from error
        if _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
        value = json.loads(path.read_text(encoding="utf-8"))
        if (
            value.get("content_sha256") != binding["content_sha256"]
            or _content_sha(value) != binding["content_sha256"]
        ):
            raise ValueError(f"{name} content hash mismatch")
        loaded[name] = value
    selector = loaded["selector_contract"]
    prior = loaded["poisson_cox_power"]
    if (
        selector.get("decision")
        != "Poisson_selector_contract_registered_no_registered_derivation_path"
        or selector.get("data_seals", {}).get("observations_opened") is not False
        or prior.get("decision")
        != "synthetic_poisson_cox_falsification_power_measured_scientific_and_observational_claims_blocked"
        or prior.get("counts", {}).get("scenario_cells") != 144
        or prior.get("counts", {}).get("scientific_tests_passed") != 0
    ):
        raise ValueError("predecessor scientific boundary changed")
    if int(config["replicates"]) < 64 or int(config["pgf_probe_count"]) < 16:
        raise ValueError("set-indexed workload too small")
    return config, loaded, root


def exact_common_shock_sentinel() -> dict[str, Any]:
    mu = Fraction(2)
    shock_fraction = Fraction(1, 4)
    shock_rate = mu * shock_fraction
    group_size = 2
    z = Fraction(1, 2)
    null_exponent = group_size * mu * (z - 1)
    alternative_exponent = (
        group_size * (mu - shock_rate) * (z - 1)
        + shock_rate * (z**group_size - 1)
    )
    return {
        "marginal_mean": str(mu),
        "marginal_variance": str(mu),
        "marginal_fano": "1",
        "shared_shock_rate": str(shock_rate),
        "within_group_cross_covariance": str(shock_rate),
        "within_group_cross_factorial_excess": str(shock_rate),
        "two_cell_union_mean": str(group_size * mu),
        "two_cell_union_variance": str(group_size * mu + 2 * shock_rate),
        "two_cell_union_fano": str(1 + shock_rate / mu),
        "pgf_probe_z": str(z),
        "null_joint_pgf": f"exp({null_exponent})",
        "alternative_joint_pgf": f"exp({alternative_exponent})",
        "alternative_to_null_pgf_ratio": f"exp({alternative_exponent - null_exponent})",
    }


def _scenario_seed(master: int, *parts: object) -> int:
    payload = ":".join((str(master), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def _probe_panel(
    master: int, cell_count: int, group_size: int, probe_count: int, active: list[Fraction]
) -> tuple[np.ndarray, list[list[str]], str]:
    rng = np.random.Generator(
        np.random.PCG64(_scenario_seed(master, cell_count, group_size, "pgf_probes"))
    )
    probes = np.ones((probe_count, cell_count), dtype=np.float64)
    exact: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    densities = (0.125, 0.25, 0.5, 0.75)
    for probe_index in range(probe_count):
        value = active[probe_index % len(active)]
        for _attempt in range(10_000):
            mask = rng.random(cell_count) < densities[probe_index % len(densities)]
            group_index = probe_index % (cell_count // group_size)
            start = group_index * group_size
            mask[start : start + min(group_size, 2)] = True
            row = [Fraction(1) for _ in range(cell_count)]
            for cell_index in np.flatnonzero(mask):
                row[int(cell_index)] = value
            signature = tuple(str(item) for item in row)
            if signature not in seen:
                seen.add(signature)
                break
        else:
            raise ValueError("unable to construct unique PGF probe panel")
        probes[probe_index] = np.asarray([float(item) for item in row])
        exact.append([str(item) for item in row])
    if len({tuple(row) for row in exact}) != probe_count:
        raise ValueError("PGF probe panel is not unique")
    return probes, exact, hashlib.sha256(probes.astype("<f8").tobytes()).hexdigest()


def _pgf_expectations(
    exact_probes: list[list[str]], mu: Fraction, group_size: int, shock_rate: Fraction
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    null_values: list[float] = []
    alternative_values: list[float] = []
    null_exponents: list[str] = []
    alternative_exponents: list[str] = []
    for row_text in exact_probes:
        row = [Fraction(item) for item in row_text]
        null_exponent = sum((mu * (z - 1) for z in row), Fraction(0))
        alternative_exponent = Fraction(0)
        for start in range(0, len(row), group_size):
            group = row[start : start + group_size]
            alternative_exponent += sum(
                ((mu - shock_rate) * (z - 1) for z in group), Fraction(0)
            ) + shock_rate * (math.prod(group) - 1)
        null_exponents.append(str(null_exponent))
        alternative_exponents.append(str(alternative_exponent))
        null_values.append(math.exp(float(null_exponent)))
        alternative_values.append(math.exp(float(alternative_exponent)))
    return (
        np.asarray(null_values, dtype=np.float64),
        np.asarray(alternative_values, dtype=np.float64),
        null_exponents,
        alternative_exponents,
    )


def _gpu_metrics(
    cp: Any,
    counts: Any,
    mu: float,
    group_size: int,
    probes: Any,
    null_pgf: Any,
) -> tuple[dict[str, Any], Any]:
    data = counts.astype(cp.float64, copy=False)
    rows = data.reshape((-1, data.shape[2]))
    projected = rows @ cp.log(probes).T
    pgf = cp.exp(projected).reshape((data.shape[0], data.shape[1], probes.shape[0]))
    empirical_pgf = cp.mean(pgf, axis=1)
    pgf_score = cp.mean(empirical_pgf - null_pgf[None, :], axis=1)
    centered = data - mu
    grouped = centered.reshape(
        (data.shape[0], data.shape[1], data.shape[2] // group_size, group_size)
    )
    group_sum = cp.sum(grouped, axis=3)
    ordered_cross = group_sum * group_sum - cp.sum(grouped * grouped, axis=3)
    cross = cp.mean(ordered_cross / (group_size * (group_size - 1)), axis=(1, 2))
    union = cp.sum(data, axis=2)
    union_dispersion = cp.var(union, axis=1, ddof=1) / cp.maximum(
        cp.mean(union, axis=1), 1e-300
    )
    first_cell = data[:, :, 0]
    marginal_dispersion = cp.var(first_cell, axis=1, ddof=1) / cp.maximum(
        cp.mean(first_cell, axis=1), 1e-300
    )
    return {
        "joint_pgf_laplace_residual": pgf_score,
        "cross_factorial_excess": cross,
        "union_dispersion": union_dispersion,
        "marginal_dispersion_deviation": cp.abs(marginal_dispersion - 1.0),
    }, empirical_pgf


def _cpu_metrics(
    counts: np.ndarray,
    mu: float,
    group_size: int,
    probes: np.ndarray,
    null_pgf: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    data = counts.astype(np.float64, copy=False)
    rows = data.reshape((-1, data.shape[2]))
    pgf = np.exp(rows @ np.log(probes).T).reshape(
        (data.shape[0], data.shape[1], probes.shape[0])
    )
    empirical_pgf = np.mean(pgf, axis=1)
    centered = data - mu
    grouped = centered.reshape(
        (data.shape[0], data.shape[1], data.shape[2] // group_size, group_size)
    )
    group_sum = np.sum(grouped, axis=3)
    ordered_cross = group_sum * group_sum - np.sum(grouped * grouped, axis=3)
    union = np.sum(data, axis=2)
    first_cell = data[:, :, 0]
    marginal = np.var(first_cell, axis=1, ddof=1) / np.maximum(
        np.mean(first_cell, axis=1), 1e-300
    )
    return {
        "joint_pgf_laplace_residual": np.mean(empirical_pgf - null_pgf[None, :], axis=1),
        "cross_factorial_excess": np.mean(
            ordered_cross / (group_size * (group_size - 1)), axis=(1, 2)
        ),
        "union_dispersion": np.var(union, axis=1, ddof=1)
        / np.maximum(np.mean(union, axis=1), 1e-300),
        "marginal_dispersion_deviation": np.abs(marginal - 1.0),
    }, empirical_pgf


def execute_gpu_workload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Execute bounded pure compute and return results without writing any artifact."""
    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CUDA unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("no CUDA device")
    device = _device(cp)
    if config["required_device_name_fragment"] not in device["device_name"]:
        raise RuntimeError("required RTX 5090 device not present")
    means = [_fraction(item) for item in config["marginal_poisson_means"]]
    cell_counts = [int(item) for item in config["disjoint_cell_counts"]]
    group_sizes = [int(item) for item in config["common_shock_group_sizes"]]
    shock_fractions = [_fraction(item) for item in config["common_shock_fractions"]]
    samples = [int(item) for item in config["repeated_set_samples"]]
    active = [_fraction(item) for item in config["pgf_active_values"]]
    replicates = int(config["replicates"])
    probe_count = int(config["pgf_probe_count"])
    sentinel_replicates = int(config["cpu_sentinel_replicates"])
    sentinel_probes = int(config["cpu_sentinel_probes"])
    alpha = float(config["empirical_size_alpha"])
    abs_bound = float(config["gpu_cpu_absolute_error_bound"])
    rel_bound = float(config["gpu_cpu_relative_error_bound"])
    master = int(config["deterministic_seed"])
    sampler = _NvmlSampler(device["device_index"], float(config["utilization_sample_interval_seconds"]))
    rows: list[dict[str, Any]] = []
    stream_digest = hashlib.sha256()
    probe_receipts: dict[str, str] = {}
    count_values = 0
    pgf_terms = 0
    projection_multiply_adds = 0
    max_abs_error = 0.0
    max_rel_error = 0.0
    sentinel_metric_values = 0
    sentinel_pgf_values = 0
    decisions_equal = True
    sampler.start()
    started = time.perf_counter()
    for mu_fraction in means:
        mu = float(mu_fraction)
        for cell_count in cell_counts:
            for sample_count in samples:
                calibration_seed = _scenario_seed(master, mu_fraction, cell_count, sample_count, "cal")
                evaluation_seed = _scenario_seed(master, mu_fraction, cell_count, sample_count, "eval")
                calibration_gpu = cp.random.RandomState(calibration_seed).poisson(
                    mu, size=(replicates, sample_count, cell_count), dtype=cp.int32
                )
                evaluation_gpu = cp.random.RandomState(evaluation_seed).poisson(
                    mu, size=(replicates, sample_count, cell_count), dtype=cp.int32
                )
                count_values += 2 * replicates * sample_count * cell_count
                calibration_slice = cp.asnumpy(calibration_gpu[:sentinel_replicates])
                evaluation_slice = cp.asnumpy(evaluation_gpu[:sentinel_replicates])
                stream_digest.update(calibration_slice.tobytes())
                stream_digest.update(evaluation_slice.tobytes())
                for group_size in group_sizes:
                    if cell_count % group_size:
                        raise ValueError("group size does not divide disjoint cell count")
                    probes, exact_probes, probe_sha = _probe_panel(
                        master, cell_count, group_size, probe_count, active
                    )
                    probe_receipts[f"cells={cell_count}:group={group_size}"] = probe_sha
                    probes_gpu = cp.asarray(probes)
                    null_exact, _, null_exponents, _ = _pgf_expectations(
                        exact_probes, mu_fraction, group_size, Fraction(0)
                    )
                    null_gpu = cp.asarray(null_exact)
                    calibration_metrics_gpu, calibration_pgf_gpu = _gpu_metrics(
                        cp, calibration_gpu, mu, group_size, probes_gpu, null_gpu
                    )
                    evaluation_metrics_gpu, evaluation_pgf_gpu = _gpu_metrics(
                        cp, evaluation_gpu, mu, group_size, probes_gpu, null_gpu
                    )
                    pgf_terms += 2 * replicates * sample_count * probe_count
                    projection_multiply_adds += (
                        2 * replicates * sample_count * probe_count * cell_count
                    )
                    calibration_metrics = {
                        key: cp.asnumpy(value) for key, value in calibration_metrics_gpu.items()
                    }
                    evaluation_metrics = {
                        key: cp.asnumpy(value) for key, value in evaluation_metrics_gpu.items()
                    }
                    calibration_cpu, calibration_pgf_cpu = _cpu_metrics(
                        calibration_slice,
                        mu,
                        group_size,
                        probes[:sentinel_probes],
                        null_exact[:sentinel_probes],
                    )
                    evaluation_cpu, evaluation_pgf_cpu = _cpu_metrics(
                        evaluation_slice,
                        mu,
                        group_size,
                        probes[:sentinel_probes],
                        null_exact[:sentinel_probes],
                    )
                    for gpu_pgf, cpu_pgf in (
                        (cp.asnumpy(calibration_pgf_gpu[:sentinel_replicates, :sentinel_probes]), calibration_pgf_cpu),
                        (cp.asnumpy(evaluation_pgf_gpu[:sentinel_replicates, :sentinel_probes]), evaluation_pgf_cpu),
                    ):
                        difference = np.abs(gpu_pgf - cpu_pgf)
                        max_abs_error = max(max_abs_error, float(np.max(difference)))
                        max_rel_error = max(
                            max_rel_error,
                            float(np.max(difference / np.maximum(np.abs(cpu_pgf), 1.0))),
                        )
                        sentinel_pgf_values += int(cpu_pgf.size)
                    for shock_fraction in shock_fractions:
                        shock_rate_fraction = mu_fraction * shock_fraction
                        shock_rate = float(shock_rate_fraction)
                        alt_seed = _scenario_seed(
                            master,
                            mu_fraction,
                            cell_count,
                            sample_count,
                            group_size,
                            shock_fraction,
                            "alternative",
                        )
                        rng = cp.random.RandomState(alt_seed)
                        base = rng.poisson(
                            mu - shock_rate,
                            size=(replicates, sample_count, cell_count),
                            dtype=cp.int32,
                        )
                        shocks = rng.poisson(
                            shock_rate,
                            size=(replicates, sample_count, cell_count // group_size),
                            dtype=cp.int32,
                        )
                        alternative_gpu = base + cp.repeat(shocks, group_size, axis=2)
                        count_values += replicates * sample_count * cell_count
                        alternative_slice = cp.asnumpy(alternative_gpu[:sentinel_replicates])
                        stream_digest.update(alternative_slice.tobytes())
                        _, alternative_exact, _, alternative_exponents = _pgf_expectations(
                            exact_probes, mu_fraction, group_size, shock_rate_fraction
                        )
                        alternative_metrics_gpu, alternative_pgf_gpu = _gpu_metrics(
                            cp, alternative_gpu, mu, group_size, probes_gpu, null_gpu
                        )
                        pgf_terms += replicates * sample_count * probe_count
                        projection_multiply_adds += (
                            replicates * sample_count * probe_count * cell_count
                        )
                        alternative_metrics = {
                            key: cp.asnumpy(value) for key, value in alternative_metrics_gpu.items()
                        }
                        alternative_cpu, alternative_pgf_cpu = _cpu_metrics(
                            alternative_slice,
                            mu,
                            group_size,
                            probes[:sentinel_probes],
                            null_exact[:sentinel_probes],
                        )
                        gpu_alt_pgf = cp.asnumpy(
                            alternative_pgf_gpu[:sentinel_replicates, :sentinel_probes]
                        )
                        difference = np.abs(gpu_alt_pgf - alternative_pgf_cpu)
                        max_abs_error = max(max_abs_error, float(np.max(difference)))
                        max_rel_error = max(
                            max_rel_error,
                            float(
                                np.max(
                                    difference
                                    / np.maximum(np.abs(alternative_pgf_cpu), 1.0)
                                )
                            ),
                        )
                        sentinel_pgf_values += int(alternative_pgf_cpu.size)
                        thresholds: dict[str, float] = {}
                        null_rates: dict[str, float] = {}
                        powers: dict[str, float] = {}
                        separations: dict[str, float] = {}
                        for metric in METRICS:
                            if metric == "joint_pgf_laplace_residual":
                                sentinel_pairs = (
                                    (
                                        np.mean(
                                            cp.asnumpy(
                                                calibration_pgf_gpu[
                                                    :sentinel_replicates, :sentinel_probes
                                                ]
                                            )
                                            - null_exact[None, :sentinel_probes],
                                            axis=1,
                                        ),
                                        calibration_cpu[metric],
                                    ),
                                    (
                                        np.mean(
                                            cp.asnumpy(
                                                evaluation_pgf_gpu[
                                                    :sentinel_replicates, :sentinel_probes
                                                ]
                                            )
                                            - null_exact[None, :sentinel_probes],
                                            axis=1,
                                        ),
                                        evaluation_cpu[metric],
                                    ),
                                    (
                                        np.mean(
                                            gpu_alt_pgf - null_exact[None, :sentinel_probes], axis=1
                                        ),
                                        alternative_cpu[metric],
                                    ),
                                )
                            else:
                                sentinel_pairs = (
                                    (
                                        calibration_metrics[metric][:sentinel_replicates],
                                        calibration_cpu[metric],
                                    ),
                                    (
                                        evaluation_metrics[metric][:sentinel_replicates],
                                        evaluation_cpu[metric],
                                    ),
                                    (
                                        alternative_metrics[metric][:sentinel_replicates],
                                        alternative_cpu[metric],
                                    ),
                                )
                            for gpu_values, cpu_values in sentinel_pairs:
                                diff = np.abs(gpu_values - cpu_values)
                                max_abs_error = max(max_abs_error, float(np.max(diff)))
                                max_rel_error = max(
                                    max_rel_error,
                                    float(np.max(diff / np.maximum(np.abs(cpu_values), 1.0))),
                                )
                                sentinel_metric_values += int(cpu_values.size)
                            threshold = float(
                                np.quantile(calibration_metrics[metric], 1 - alpha, method="higher")
                            )
                            guard = float(config["decision_numerical_guard_multiplier"]) * max(
                                abs_bound, rel_bound * max(abs(threshold), 1.0)
                            )
                            threshold += guard
                            cpu_null = evaluation_metrics[metric] > threshold
                            cpu_alt = alternative_metrics[metric] > threshold
                            gpu_null = cp.asnumpy(evaluation_metrics_gpu[metric] > threshold)
                            gpu_alt = cp.asnumpy(alternative_metrics_gpu[metric] > threshold)
                            decisions_equal = decisions_equal and np.array_equal(
                                cpu_null, gpu_null
                            ) and np.array_equal(cpu_alt, gpu_alt)
                            thresholds[metric] = threshold
                            null_rates[metric] = float(np.mean(cpu_null))
                            powers[metric] = float(np.mean(cpu_alt))
                            separations[metric] = float(
                                np.mean(alternative_metrics[metric])
                                - np.mean(evaluation_metrics[metric])
                            )
                        rows.append(
                            {
                                "marginal_poisson_mean": str(mu_fraction),
                                "disjoint_cell_count": cell_count,
                                "common_shock_group_size": group_size,
                                "common_shock_fraction": str(shock_fraction),
                                "common_shock_rate": str(shock_rate_fraction),
                                "repeated_set_samples": sample_count,
                                "calibration_seed": calibration_seed,
                                "evaluation_seed": evaluation_seed,
                                "alternative_seed": alt_seed,
                                "analytic": {
                                    "each_marginal_distribution": f"Poisson({mu_fraction}) under null and alternative",
                                    "marginal_fano_null_and_alternative": "1",
                                    "within_group_cross_covariance": str(shock_rate_fraction),
                                    "union_fano_null": "1",
                                    "union_fano_alternative": str(
                                        1 + (group_size - 1) * shock_fraction
                                    ),
                                    "first_null_pgf_exponent": null_exponents[0],
                                    "first_alternative_pgf_exponent": alternative_exponents[0],
                                    "first_alternative_pgf": float(alternative_exact[0]),
                                },
                                "upper_tail_thresholds": thresholds,
                                "heldout_null_rejection_rates": null_rates,
                                "heldout_alternative_detection_rates": powers,
                                "empirical_alternative_minus_null_separation": separations,
                            }
                        )
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    utilization = sampler.stop()
    if max_abs_error > abs_bound and max_rel_error > rel_bound:
        raise ValueError("GPU/CPU sentinel bounds exceeded")
    if not decisions_equal:
        raise ValueError("GPU/CPU held-out decisions differ")
    return {
        "scenario_results": rows,
        "device": device,
        "elapsed_seconds": elapsed,
        "utilization": utilization,
        "realized_sentinel_stream_sha256": stream_digest.hexdigest(),
        "probe_panel_sha256": probe_receipts,
        "counts": {
            "scenario_cells": len(rows),
            "gpu_generated_unique_count_values": count_values,
            "joint_pgf_terms_evaluated": pgf_terms,
            "projection_multiply_adds": projection_multiply_adds,
            "cpu_gpu_sentinel_metric_values": sentinel_metric_values,
            "cpu_gpu_sentinel_pgf_values": sentinel_pgf_values,
        },
        "crosscheck": {
            "maximum_absolute_error": max_abs_error,
            "maximum_relative_error_scaled_at_one": max_rel_error,
            "all_heldout_decisions_byte_equal": decisions_equal,
        },
    }


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, _loaded, root = _load(config_path)
    workload = execute_gpu_workload(config)
    replicates = int(config["replicates"])
    hoeffding = math.sqrt(math.log(40.0) / (2 * replicates))
    source_path = root / "src/sigma_theory_compiler/kastner_schlatter_set_indexed_cuda_falsification_campaign.py"
    test_path = root / "tests/test_kastner_schlatter_set_indexed_cuda_falsification_campaign.py"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            **config["predecessors"],
            "config": {"path": config_path.relative_to(root).as_posix(), "file_sha256": _file_sha(config_path)},
            "source": {"path": source_path.relative_to(root).as_posix(), "file_sha256": _file_sha(source_path)},
            "test": {"path": test_path.relative_to(root).as_posix(), "file_sha256": _file_sha(test_path)},
        },
        "set_indexed_contract": {
            "null": "independent Poisson counts on every declared disjoint cell",
            "alternative": "block-Poisson common shocks preserve every one-cell Poisson marginal but violate disjoint-cell independence",
            "joint_test_objects": ["finite-dimensional PGF", "Laplace functional at z=exp(-f)", "cross-factorial moment", "union dispersion"],
            "action_derived": False,
            "operational_transaction_events_defined": False,
        },
        "exact_common_shock_sentinel": exact_common_shock_sentinel(),
        "design": {
            "marginal_poisson_means": config["marginal_poisson_means"],
            "disjoint_cell_counts": config["disjoint_cell_counts"],
            "common_shock_group_sizes": config["common_shock_group_sizes"],
            "common_shock_fractions": config["common_shock_fractions"],
            "repeated_set_samples": config["repeated_set_samples"],
            "replicates_per_calibration_evaluation_and_alternative": replicates,
            "unique_pgf_laplace_probes_per_panel": config["pgf_probe_count"],
            "metrics": list(METRICS),
            "heldout_calibration": "null calibration, null evaluation, and alternative streams use distinct SHA-derived seeds",
            "marginal_dispersion_role": "single predeclared first-cell negative control because both hypotheses have exactly the same one-cell Poisson law",
            "power_monte_carlo_95pct_hoeffding_halfwidth": hoeffding,
        },
        "scenario_results": workload["scenario_results"],
        "gpu_cpu_crosscheck": {
            **workload["crosscheck"],
            "absolute_error_bound": config["gpu_cpu_absolute_error_bound"],
            "relative_error_bound": config["gpu_cpu_relative_error_bound"],
            "scope": "exact-formula sentinel plus CPU recomputation of bounded slices from every GPU dataset and PGF panel",
        },
        "deterministic_manifest": {
            "master_seed": config["deterministic_seed"],
            "gpu_generator_api": "cupy.random.RandomState",
            "scenario_seed_derivation": "little-endian uint32 from SHA-256 design coordinate prefix",
            "realized_sentinel_stream_sha256": workload["realized_sentinel_stream_sha256"],
            "probe_panel_sha256": workload["probe_panel_sha256"],
            "design_root_sha256": _sha({key: config[key] for key in ("marginal_poisson_means", "disjoint_cell_counts", "common_shock_group_sizes", "common_shock_fractions", "repeated_set_samples", "replicates", "pgf_probe_count")}),
        },
        "counts": {
            **workload["counts"],
            "prior_poisson_cox_scenario_cells": 144,
            "observational_records_accessed": 0,
            "readiness_fields_advanced": 0,
            "scientific_tests_passed": 0,
            "paper_or_qed_inferences": 0,
        },
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": workload["device"],
            "wall_seconds": workload["elapsed_seconds"],
            "unique_count_values_per_second": workload["counts"]["gpu_generated_unique_count_values"] / workload["elapsed_seconds"],
            "effective_joint_pgf_terms_per_second": workload["counts"]["joint_pgf_terms_evaluated"] / workload["elapsed_seconds"],
            "utilization": workload["utilization"],
            "scope": "device-wide nonexclusive counters; unique count generation and effective non-padding PGF/projection evaluations are reported separately",
        },
        "evaluator_entry_point": {
            "callable": "sigma_theory_compiler.kastner_schlatter_set_indexed_cuda_falsification_campaign.execute_gpu_workload",
            "writes_artifact": False,
            "bounded_by_config": True,
            "scheduler_or_persistent_service_used_for_this_run": False,
        },
        "interpretation": {
            "implementation_stress": "fused batched CUDA PGF/Laplace projections and joint-moment reductions",
            "synthetic_falsification_power": "ability to distinguish independent increments from a declared marginal-preserving correlated alternative",
            "scientific_or_observational_test": "not performed",
            "non_test": "does not validate transactions, ontology, the paper, QED, an action-derived point process, or real-event correspondence",
        },
        "decision": "synthetic_set_indexed_independence_falsification_power_measured_all_scientific_claims_blocked",
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
        raise ValueError("artifact exceeds configured output bound")
    return result


def validate_campaign(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, _, root = _load(Path(config_path))
    if result.get("schema_version") != SCHEMA or result.get("content_sha256") != _content_sha(result):
        raise ValueError("artifact schema or content hash mismatch")
    if result.get("synthetic_only") is not True:
        raise ValueError("synthetic seal changed")
    for key in (
        "observations_opened", "scientific_test_pass", "observational_test_pass", "readiness_advanced",
        "paper_pass", "qed_pass", "theory_pass", "ontology_pass", "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs", "paid_llm_calls",
    ):
        if result.get(key) is not False:
            raise ValueError(f"claim or data seal changed: {key}")
    expected_scenarios = (
        len(config["marginal_poisson_means"])
        * len(config["disjoint_cell_counts"])
        * len(config["common_shock_group_sizes"])
        * len(config["common_shock_fractions"])
        * len(config["repeated_set_samples"])
    )
    counts = result.get("counts", {})
    if counts.get("scenario_cells") != expected_scenarios or any(
        counts.get(key) != 0 for key in (
            "observational_records_accessed", "readiness_fields_advanced", "scientific_tests_passed", "paper_or_qed_inferences"
        )
    ):
        raise ValueError("scenario or forbidden count mismatch")
    if result.get("exact_common_shock_sentinel") != exact_common_shock_sentinel():
        raise ValueError("exact common-shock sentinel mismatch")
    cross = result.get("gpu_cpu_crosscheck", {})
    if (
        cross.get("maximum_absolute_error", math.inf) > config["gpu_cpu_absolute_error_bound"]
        and cross.get("maximum_relative_error_scaled_at_one", math.inf) > config["gpu_cpu_relative_error_bound"]
    ) or cross.get("all_heldout_decisions_byte_equal") is not True:
        raise ValueError("GPU/CPU crosscheck failed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"{name} binding mismatch")
    if result["source_bindings"]["selector_contract"] != config["predecessors"]["selector_contract"]:
        raise ValueError("selector predecessor binding mismatch")


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
