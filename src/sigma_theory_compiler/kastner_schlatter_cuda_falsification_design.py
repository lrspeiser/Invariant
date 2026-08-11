"""Synthetic CUDA power controls for Kastner--Schlatter falsification design."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from .kastner_schlatter_cuda_consequence_campaign import _device, _NvmlSampler

SCHEMA = "sigma-kastner-schlatter-cuda-falsification-design-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-cuda-falsification-design-config-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    root = config_path.resolve().parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported CUDA falsification-design config")
    seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "scientific_test_pass_allowed": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(config.get(key) != value for key, value in seals.items()):
        raise ValueError("falsification-design seals changed")
    loaded: dict[str, dict[str, Any]] = {}
    for name, binding in config["predecessors"].items():
        path = (root / binding["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"{name} path escapes repository") from error
        if _file_sha(path) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
        document = json.loads(path.read_text(encoding="utf-8"))
        if (
            document.get("content_sha256") != binding["content_sha256"]
            or _content_sha(document) != binding["content_sha256"]
        ):
            raise ValueError(f"{name} content hash mismatch")
        loaded[name] = document
    readiness = loaded["observational_readiness"]
    consequence = loaded["cuda_consequence"]
    if (
        readiness.get("decision") != "blocked_registration_incomplete_observations_sealed"
        or readiness.get("observational_access_count") != 0
        or readiness.get("real_data_pass_count") != 0
        or consequence.get("synthetic_only") is not True
        or consequence.get("observations_opened") is not False
        or consequence.get("equation_35_normalization_gate", {}).get("decision") != "blocked"
    ):
        raise ValueError("predecessor scientific boundary changed")
    return config, loaded, root


def deterministic_inputs(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(int(config["deterministic_seed"])))
    p = config["poisson_power"]
    replicates = int(p["replicates"])
    exposure_blocks = int(p["exposure_blocks_per_replicate"])
    mean = float(p["null_mean"])
    null = rng.poisson(mean, size=(replicates, exposure_blocks)).astype("<f8")
    parity = np.indices((replicates, exposure_blocks)).sum(axis=0) & 1
    low = mean * float(p["alternative_low_multiplier"])
    high = mean * float(p["alternative_high_multiplier"])
    alternative_means = np.where(parity == 0, low, high)
    alternative = rng.poisson(alternative_means).astype("<f8")

    b = config["btfr_power"]
    groups = int(b["galaxy_groups"])
    records = int(b["records_per_group"])
    sigma = float(b["log_residual_sigma"])
    null_residual = rng.normal(0.0, sigma, size=(groups, records)).astype("<f8")
    alternative_residual = rng.normal(
        float(b["injected_log_offset"]), sigma, size=(groups, records)
    ).astype("<f8")
    return {
        "poisson_null": np.ascontiguousarray(null),
        "poisson_overdispersed_alternative": np.ascontiguousarray(alternative),
        "btfr_null_log_residual": np.ascontiguousarray(null_residual),
        "btfr_offset_alternative_log_residual": np.ascontiguousarray(alternative_residual),
    }


def _cpu_statistics(inputs: Mapping[str, np.ndarray], config: Mapping[str, Any]) -> dict[str, Any]:
    p_null = inputs["poisson_null"]
    p_alt = inputs["poisson_overdispersed_alternative"]
    null_fano = np.var(p_null, axis=1, ddof=1) / np.mean(p_null, axis=1)
    alt_fano = np.var(p_alt, axis=1, ddof=1) / np.mean(p_alt, axis=1)
    threshold = float(config["poisson_power"]["absolute_fano_deviation_threshold"])
    p_null_reject = np.abs(null_fano - 1.0) > threshold
    p_alt_reject = np.abs(alt_fano - 1.0) > threshold

    sigma = float(config["btfr_power"]["log_residual_sigma"])
    records = int(config["btfr_power"]["records_per_group"])
    null_z = np.mean(inputs["btfr_null_log_residual"], axis=1) * math.sqrt(records) / sigma
    alt_z = (
        np.mean(inputs["btfr_offset_alternative_log_residual"], axis=1)
        * math.sqrt(records)
        / sigma
    )
    z_threshold = float(config["btfr_power"]["absolute_group_mean_z_threshold"])
    return {
        "poisson_null_fano": null_fano,
        "poisson_alt_fano": alt_fano,
        "poisson_null_reject": p_null_reject,
        "poisson_alt_reject": p_alt_reject,
        "btfr_null_z": null_z,
        "btfr_alt_z": alt_z,
        "btfr_null_reject": np.abs(null_z) > z_threshold,
        "btfr_alt_reject": np.abs(alt_z) > z_threshold,
    }


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, loaded, root = _load(config_path)
    inputs = deterministic_inputs(config)
    cpu = _cpu_statistics(inputs, config)
    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CUDA unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("no CUDA device")
    device = _device(cp)
    arrays = {key: cp.asarray(value) for key, value in inputs.items()}

    def dispatch() -> dict[str, Any]:
        pn = arrays["poisson_null"]
        pa = arrays["poisson_overdispersed_alternative"]
        bn = arrays["btfr_null_log_residual"]
        ba = arrays["btfr_offset_alternative_log_residual"]
        p_threshold = float(config["poisson_power"]["absolute_fano_deviation_threshold"])
        z_threshold = float(config["btfr_power"]["absolute_group_mean_z_threshold"])
        sigma = float(config["btfr_power"]["log_residual_sigma"])
        scale = math.sqrt(int(config["btfr_power"]["records_per_group"])) / sigma
        null_fano = cp.var(pn, axis=1, ddof=1) / cp.mean(pn, axis=1)
        alt_fano = cp.var(pa, axis=1, ddof=1) / cp.mean(pa, axis=1)
        null_z = cp.mean(bn, axis=1) * scale
        alt_z = cp.mean(ba, axis=1) * scale
        return {
            "poisson_null_fano": null_fano,
            "poisson_alt_fano": alt_fano,
            "poisson_null_reject": cp.abs(null_fano - 1.0) > p_threshold,
            "poisson_alt_reject": cp.abs(alt_fano - 1.0) > p_threshold,
            "btfr_null_z": null_z,
            "btfr_alt_z": alt_z,
            "btfr_null_reject": cp.abs(null_z) > z_threshold,
            "btfr_alt_reject": cp.abs(alt_z) > z_threshold,
        }

    for _ in range(int(config["gpu_warmup_repetitions"])):
        gpu = dispatch()
    cp.cuda.Device().synchronize()
    sampler = _NvmlSampler(device["device_index"], float(config["utilization_sample_interval_seconds"]))
    sampler.start()
    started = time.perf_counter()
    for _ in range(int(config["gpu_measured_repetitions"])):
        gpu = dispatch()
    cp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    utilization = sampler.stop()
    gpu_host = {key: cp.asnumpy(value) for key, value in gpu.items()}
    max_error = max(
        float(np.max(np.abs(gpu_host[key] - cpu[key])))
        for key in ("poisson_null_fano", "poisson_alt_fano", "btfr_null_z", "btfr_alt_z")
    )
    decisions_equal = all(
        np.array_equal(gpu_host[key], cpu[key])
        for key in (
            "poisson_null_reject",
            "poisson_alt_reject",
            "btfr_null_reject",
            "btfr_alt_reject",
        )
    )
    if max_error > float(config["gpu_cpu_error_bound"]) or not decisions_equal:
        raise ValueError("GPU/CPU falsification statistic crosscheck failed")

    poisson_replicates = int(config["poisson_power"]["replicates"])
    btfr_groups = int(config["btfr_power"]["galaxy_groups"])
    poisson_null_rate = float(np.mean(cpu["poisson_null_reject"]))
    poisson_power = float(np.mean(cpu["poisson_alt_reject"]))
    btfr_null_rate = float(np.mean(cpu["btfr_null_reject"]))
    btfr_power = float(np.mean(cpu["btfr_alt_reject"]))
    if (
        poisson_null_rate > float(config["poisson_power"]["maximum_null_rejection_rate"])
        or poisson_power < float(config["poisson_power"]["minimum_alternative_detection_rate"])
        or btfr_null_rate > float(config["btfr_power"]["maximum_null_rejection_rate"])
        or btfr_power < float(config["btfr_power"]["minimum_alternative_detection_rate"])
    ):
        raise ValueError("synthetic falsification power control failed")

    input_hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()}
    repetitions = int(config["gpu_measured_repetitions"])
    values_per_dispatch = sum(value.size for value in inputs.values())
    source_path = root / "src/sigma_theory_compiler/kastner_schlatter_cuda_falsification_design.py"
    test_path = root / "tests/test_kastner_schlatter_cuda_falsification_design.py"
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "observational_readiness": config["predecessors"]["observational_readiness"],
            "cuda_consequence": config["predecessors"]["cuda_consequence"],
            "config": {"path": config_path.relative_to(root).as_posix(), "file_sha256": _file_sha(config_path)},
            "source": {"path": source_path.relative_to(root).as_posix(), "file_sha256": _file_sha(source_path)},
            "test": {"path": test_path.relative_to(root).as_posix(), "file_sha256": _file_sha(test_path)},
            "primary_pdf_sha256": loaded["cuda_consequence"]["predecessor_binding"]["primary_pdf_sha256"],
        },
        "workload_priority": [
            {"rank": 1, "workload": "Poisson overdispersion power", "executed": True, "evidence_class": "falsification_design_control"},
            {"rank": 2, "workload": "BTFR group-residual offset power", "executed": True, "evidence_class": "falsification_design_control"},
            {"rank": 3, "workload": "operational transaction detector response", "executed": False, "blocker": "no source-supported transaction-event detector equivalence"},
            {"rank": 4, "workload": "extended-galaxy geometry likelihood", "executed": False, "blocker": "no source-supported disk or enclosed-mass operator"},
            {"rank": 5, "workload": "Lambda observational propagation", "executed": False, "blocker": "equation-35 normalization and cosmology authorization"},
        ],
        "poisson_power_control": {
            "replicates": poisson_replicates,
            "exposure_blocks_per_replicate": int(config["poisson_power"]["exposure_blocks_per_replicate"]),
            "null_mean": float(config["poisson_power"]["null_mean"]),
            "alternative": "equal-weight Poisson-rate mixture at 0.5*mu and 1.5*mu",
            "analytic_null_fano": 1.0,
            "analytic_alternative_fano": 2.0,
            "empirical_null_rejection_rate": poisson_null_rate,
            "empirical_alternative_detection_rate": poisson_power,
            "scope": "tests a declared count-statistic design if operational events and exposure later exist",
        },
        "btfr_power_control": {
            "galaxy_groups": btfr_groups,
            "records_per_group": int(config["btfr_power"]["records_per_group"]),
            "relation": "log(v^4/(G*M*a0))=0 for the point-mass asymptotic source equation",
            "injected_log_offset": float(config["btfr_power"]["injected_log_offset"]),
            "empirical_null_rejection_rate": btfr_null_rate,
            "empirical_alternative_detection_rate": btfr_power,
            "group_split_unit": "synthetic galaxy identifier; radius-row splitting forbidden",
            "extended_galaxy_geometry_tested": False,
        },
        "observational_bridge": {
            "synthetically_exercised_missing_fields": [
                "transaction.independent_increments_diagnostic",
                "transaction.stationarity_diagnostic",
                "transaction.likelihood_threshold",
                "mond.residual_statistic_threshold",
                "mond.covariance",
                "mond.galaxy_group_heldout_ids",
            ],
            "registration_fields_advanced": 0,
            "real_bundle_fields_filled": 0,
            "interpretation": "schema and power rehearsal only; synthetic values do not fill registration fields",
        },
        "gpu_cpu_crosscheck": {
            "maximum_absolute_statistic_error": max_error,
            "error_bound": float(config["gpu_cpu_error_bound"]),
            "all_rejection_decisions_byte_equal": decisions_equal,
        },
        "deterministic_manifest": {
            "seed": int(config["deterministic_seed"]),
            "generator": "numpy_pcg64",
            "input_sha256": input_hashes,
            "manifest_root_sha256": _sha(input_hashes),
        },
        "counts": {
            "poisson_synthetic_count_values": int(inputs["poisson_null"].size + inputs["poisson_overdispersed_alternative"].size),
            "btfr_synthetic_residual_values": int(inputs["btfr_null_log_residual"].size + inputs["btfr_offset_alternative_log_residual"].size),
            "gpu_measured_repetitions": repetitions,
            "gpu_measured_value_evaluations": repetitions * values_per_dispatch,
            "observational_records_accessed": 0,
            "scientific_tests_passed": 0,
            "readiness_fields_advanced": 0,
        },
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "gpu_measured_wall_seconds": elapsed,
            "gpu_value_evaluations_per_second": repetitions * values_per_dispatch / elapsed,
            "gpu_array_bytes": sum(int(value.nbytes) for value in arrays.values()),
            "utilization": utilization,
            "scope": "single device-wide local run; not a sustained or lane-exclusive throughput claim",
        },
        "evidence_classification": {
            "gpu_execution": "implementation_stress_test",
            "synthetic_power": "falsification_design_control",
            "scientific_test": "not_performed",
            "observational_test": "not_performed",
        },
        "decision": "synthetic_falsification_design_controls_closed_observational_lanes_still_blocked",
        "synthetic_only": True,
        "observations_opened": False,
        "scientific_test_pass": False,
        "theory_pass": False,
        "ontology_pass": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_campaign(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, _, root = _load(Path(config_path))
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("content hash mismatch")
    for key in ("synthetic_only",):
        if result.get(key) is not True:
            raise ValueError("synthetic seal changed")
    for key in (
        "observations_opened",
        "scientific_test_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        if result.get(key) is not False:
            raise ValueError("claim or data seal changed")
    if result.get("counts", {}).get("readiness_fields_advanced") != 0:
        raise ValueError("synthetic controls advanced readiness")
    if result.get("observational_bridge", {}).get("registration_fields_advanced") != 0:
        raise ValueError("observational bridge overclaimed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"{name} source binding mismatch")
    inputs = deterministic_inputs(config)
    expected = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()}
    if result.get("deterministic_manifest", {}).get("input_sha256") != expected:
        raise ValueError("deterministic input manifest mismatch")
    if result["gpu_cpu_crosscheck"]["maximum_absolute_statistic_error"] > config["gpu_cpu_error_bound"]:
        raise ValueError("GPU/CPU error bound exceeded")
    if result["gpu_cpu_crosscheck"]["all_rejection_decisions_byte_equal"] is not True:
        raise ValueError("GPU/CPU rejection decisions differ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config)
    result = build_campaign(config_path)
    root = config_path.resolve().parents[1]
    output = root / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
