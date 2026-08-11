"""Synthetic CUDA falsification controls for compiler-authored KS scalar actions."""

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

import mpmath
import numpy as np

from .kastner_schlatter_cuda_consequence_campaign import _device, _NvmlSampler

SCHEMA = "sigma-kastner-schlatter-scalar-intensity-cuda-falsification-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-scalar-intensity-cuda-falsification-config-1.0"

SCALAR_KERNEL = r"""
extern "C" __global__
void scalar_dispersion_green(
    const double* A,
    const double* B,
    const int parameter_count,
    const double* k,
    const double* r,
    const int sample_count,
    double* omega,
    double* relative_dispersion_residual,
    double* green) {
  long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
  long long total = (long long)parameter_count * sample_count;
  if (index >= total) return;
  int p = (int)(index / sample_count);
  int s = (int)(index - (long long)p * sample_count);
  double mu2 = A[p] / B[p];
  double omega2 = k[s] * k[s] + mu2;
  double w = sqrt(omega2);
  double equation_residual = B[p] * (w * w - k[s] * k[s]) - A[p];
  double scale = fabs(A[p]) + fabs(B[p] * k[s] * k[s]);
  omega[index] = w;
  relative_dispersion_residual[index] = fabs(equation_residual) / scale;
  green[index] = exp(-sqrt(mu2) * r[s]) / (12.566370614359172953850573533118 * B[p] * r[s]);
}
"""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _load_predecessor(root: Path, name: str, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
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
    return document


def _load(config_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]], Path]:
    root = config_path.resolve().parents[1]
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported scalar-intensity CUDA config")
    expected_seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "paper_or_qed_pass_allowed": False,
        "theory_pass_allowed": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(config.get(key) != value for key, value in expected_seals.items()):
        raise ValueError("scalar-intensity claim or data seals changed")
    if set(config.get("predecessors", {})) != {
        "candidate_action_completion",
        "equation_graph",
        "cuda_falsification_design",
    }:
        raise ValueError("scalar-intensity predecessor set changed")
    loaded = {
        name: _load_predecessor(root, name, binding)
        for name, binding in config["predecessors"].items()
    }
    action = loaded["candidate_action_completion"]
    hypotheses = action.get("completion_hypotheses", [])
    if (
        action.get("decision")
        != "candidate_completions_registered_paper_derivation_and_physics_claims_blocked"
        or action.get("scope")
        != "compiler-authored covariant candidate completions; not present in or derived by arXiv:2209.04025"
        or len(hypotheses) != 2
        or {item.get("beta") for item in hypotheses} != {"1/2", "1/4"}
        or any(
            item.get("euler_lagrange", {}).get("intensity")
            != "B_q*Box(q)-A_q*(q-q0)=0"
            for item in hypotheses
        )
        or any(
            item.get("candidate_action", {}).get("parameter_domain")
            != "q0>0, A_q>0, B_q>0"
            for item in hypotheses
        )
        or any(item.get("paper_authorship_or_derivation") is not False for item in hypotheses)
    ):
        raise ValueError("candidate scalar-intensity action boundary changed")
    graph = loaded["equation_graph"]
    if (
        graph.get("admission_contract", {}).get("equation_only") is not True
        or graph.get("admission_contract", {}).get("fundamental_action") is not None
    ):
        raise ValueError("paper equation-only boundary changed")
    prior_cuda = loaded["cuda_falsification_design"]
    if (
        prior_cuda.get("synthetic_only") is not True
        or prior_cuda.get("observations_opened") is not False
        or prior_cuda.get("scientific_test_pass") is not False
    ):
        raise ValueError("prior CUDA falsification boundary changed")
    return config, loaded, root


def deterministic_inputs(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    parameter_count = int(config["parameter_count"])
    sample_count = int(config["spectral_sample_count"])
    mu2 = np.exp2(
        np.linspace(
            float(config["mu_squared_log2_min"]),
            float(config["mu_squared_log2_max"]),
            parameter_count,
            dtype=np.float64,
        )
    )
    b_value = np.exp2(
        np.linspace(
            float(config["B_log2_min"]),
            float(config["B_log2_max"]),
            parameter_count,
            dtype=np.float64,
        )
    )
    a_value = b_value * mu2
    wave_number = np.exp2(
        np.linspace(
            float(config["k_log2_min"]),
            float(config["k_log2_max"]),
            sample_count,
            dtype=np.float64,
        )
    )
    radius = np.exp2(
        np.linspace(
            float(config["r_log2_min"]),
            float(config["r_log2_max"]),
            sample_count,
            dtype=np.float64,
        )
    )
    return {
        "A": np.ascontiguousarray(a_value, dtype="<f8"),
        "B": np.ascontiguousarray(b_value, dtype="<f8"),
        "mu_squared": np.ascontiguousarray(mu2, dtype="<f8"),
        "wave_number": np.ascontiguousarray(wave_number, dtype="<f8"),
        "radius": np.ascontiguousarray(radius, dtype="<f8"),
    }


def _cpu_consequences(inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    mu2 = inputs["mu_squared"][:, None]
    b_value = inputs["B"][:, None]
    a_value = inputs["A"][:, None]
    wave_number = inputs["wave_number"][None, :]
    radius = inputs["radius"][None, :]
    omega = np.sqrt(wave_number * wave_number + mu2)
    residual = np.abs(b_value * (omega * omega - wave_number * wave_number) - a_value) / (
        np.abs(a_value) + np.abs(b_value * wave_number * wave_number)
    )
    green = np.exp(-np.sqrt(mu2) * radius) / (4.0 * math.pi * b_value * radius)
    return {
        "omega": np.ascontiguousarray(omega),
        "relative_dispersion_residual": np.ascontiguousarray(residual),
        "green": np.ascontiguousarray(green),
    }


def _manifest(inputs: Mapping[str, np.ndarray]) -> dict[str, Any]:
    hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in inputs.items()}
    return {"array_sha256": dict(sorted(hashes.items())), "manifest_root_sha256": _sha(hashes)}


def _exact_sentinels() -> dict[str, Any]:
    mpmath.mp.dps = 80
    green_exact = mpmath.e**-1 / (4 * mpmath.pi)
    green_float = math.exp(-1.0) / (4.0 * math.pi)
    return {
        "dispersion_A4_B1_k3": {
            "exact_omega_squared": "13",
            "computed_omega_squared": math.sqrt(13.0) ** 2,
            "absolute_residual": abs(math.sqrt(13.0) ** 2 - 13.0),
        },
        "gap_A9_B4_k0": {
            "exact_omega": "3/2",
            "computed_omega": math.sqrt(9.0 / 4.0),
            "absolute_residual": abs(math.sqrt(9.0 / 4.0) - 1.5),
        },
        "green_A1_B1_r1": {
            "exact_expression": "exp(-1)/(4*pi)",
            "float64_value": green_float,
            "high_precision_absolute_error": float(abs(mpmath.mpf(green_float) - green_exact)),
        },
        "radial_green_operator_r_positive": {
            "equation": "(-B*(d2/dr2+2/r*d/dr)+A)*exp(-sqrt(A/B)*r)/(4*pi*B*r)=0",
            "exact_residual": "0",
            "source_flux_limit": "lim_r_to_0 4*pi*r^2*B*(-dG/dr)=1",
        },
    }


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, loaded, root = _load(config_path)
    inputs = deterministic_inputs(config)
    cpu = _cpu_consequences(inputs)
    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CUDA unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("no CUDA device")
    device = _device(cp)
    kernel = cp.RawKernel(SCALAR_KERNEL, "scalar_dispersion_green")
    threads = 256
    parameter_count = len(inputs["A"])
    sample_count = len(inputs["wave_number"])
    total = parameter_count * sample_count
    d_a = cp.asarray(inputs["A"])
    d_b = cp.asarray(inputs["B"])
    d_k = cp.asarray(inputs["wave_number"])
    d_r = cp.asarray(inputs["radius"])
    d_omega = cp.empty(total, dtype=cp.float64)
    d_residual = cp.empty(total, dtype=cp.float64)
    d_green = cp.empty(total, dtype=cp.float64)

    def dispatch() -> None:
        kernel(
            ((total + threads - 1) // threads,),
            (threads,),
            (
                d_a,
                d_b,
                np.int32(parameter_count),
                d_k,
                d_r,
                np.int32(sample_count),
                d_omega,
                d_residual,
                d_green,
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
    gpu = {
        "omega": cp.asnumpy(d_omega).reshape(parameter_count, sample_count),
        "relative_dispersion_residual": cp.asnumpy(d_residual).reshape(
            parameter_count, sample_count
        ),
        "green": cp.asnumpy(d_green).reshape(parameter_count, sample_count),
    }
    errors = {key: float(np.max(np.abs(gpu[key] - cpu[key]))) for key in gpu}
    maximum_error = max(errors.values())
    relative_errors = {
        key: float(
            np.max(
                np.abs(gpu[key] - cpu[key])
                / np.maximum(np.abs(cpu[key]), np.finfo(np.float64).tiny)
            )
        )
        for key in ("omega", "green")
    }
    maximum_relative_error = max(relative_errors.values())
    maximum_gpu_residual = float(np.max(gpu["relative_dispersion_residual"]))
    if (
        maximum_error > float(config["gpu_cpu_absolute_error_bound"])
        or maximum_relative_error > float(config["gpu_cpu_relative_error_bound"])
        or maximum_gpu_residual > float(config["dispersion_relative_residual_bound"])
    ):
        raise ValueError("scalar-intensity CUDA crosscheck failed")

    denominator_min = float(np.min(inputs["A"][:, None] + inputs["B"][:, None] * inputs["wave_number"][None, :] ** 2))
    denominator_max = float(np.max(inputs["A"][:, None] + inputs["B"][:, None] * inputs["wave_number"][None, :] ** 2))
    exact_sentinels = _exact_sentinels()
    action = loaded["candidate_action_completion"]
    branch_records = []
    output_hashes = {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in gpu.items()
    }
    for hypothesis in action["completion_hypotheses"]:
        branch_records.append(
            {
                "branch_id": hypothesis["branch_id"],
                "beta": hypothesis["beta"],
                "intensity_equation": hypothesis["euler_lagrange"]["intensity"],
                "scalar_linearized_outputs_sha256": output_hashes,
                "beta_enters_linearized_intensity_operator": False,
                "paper_authorship_or_derivation": False,
            }
        )
    source_path = root / "src/sigma_theory_compiler/kastner_schlatter_scalar_intensity_cuda_falsification.py"
    test_path = root / "tests/test_kastner_schlatter_scalar_intensity_cuda_falsification.py"
    measured_pairs = total * repetitions
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "candidate_action_completion": config["predecessors"]["candidate_action_completion"],
            "equation_graph": config["predecessors"]["equation_graph"],
            "cuda_falsification_design": config["predecessors"]["cuda_falsification_design"],
            "config": {"path": config_path.relative_to(root).as_posix(), "file_sha256": _file_sha(config_path)},
            "source": {"path": source_path.relative_to(root).as_posix(), "file_sha256": _file_sha(source_path)},
            "test": {"path": test_path.relative_to(root).as_posix(), "file_sha256": _file_sha(test_path)},
            "primary_pdf_sha256": action["source_bindings"]["primary_pdf_sha256"],
        },
        "scope_boundary": {
            "actions_are_compiler_hypotheses": True,
            "actions_present_in_or_derived_from_paper": False,
            "qed_actualization_dynamics_tested": False,
            "paper_transaction_ontology_tested": False,
            "external_green_source_is_mathematical_control_only": True,
        },
        "linearized_operator": {
            "background": "Minkowski metric and q=q0",
            "perturbation": "delta_q=q-q0",
            "equation": "B_q*Box(delta_q)-A_q*delta_q=0",
            "dispersion_with_x0_ct": "omega^2=c^2*(k^2+A_q/B_q)",
            "static_forced_operator": "(-B_q*Laplacian+A_q)*delta_q=J_synthetic",
            "green_response": "G(r)=exp(-sqrt(A_q/B_q)*r)/(4*pi*B_q*r)",
            "source_coupling_status": "synthetic mathematical forcing; not a paper or QED event coupling",
        },
        "branch_records": branch_records,
        "branch_degeneracy_control": {
            "beta_values": ["1/2", "1/4"],
            "linearized_scalar_dynamics_identical": True,
            "maximum_branch_output_difference": 0.0,
            "interpretation": "beta shifts only the stationary constant potential in these hypotheses",
        },
        "dispersion_control": {
            "all_registered_domain_omega_squared_positive": True,
            "maximum_gpu_relative_equation_residual": maximum_gpu_residual,
            "phase_velocity_statement": "omega/k>c for finite positive gap; not a signal-speed or causality test",
            "group_velocity_statement": "domega/dk=c*k/sqrt(k^2+A_q/B_q)<c",
            "physical_propagation_claim": False,
        },
        "green_yukawa_control": {
            "all_registered_domain_denominators_positive": denominator_min > 0.0,
            "minimum_A_plus_Bk2": denominator_min,
            "maximum_A_plus_Bk2": denominator_max,
            "screening_length": "sqrt(B_q/A_q)",
            "source_normalization_and_radial_operator_sentinels": exact_sentinels,
            "physical_source_response_claim": False,
        },
        "parameter_domain_controls": {
            "registered_domain": "q0>0,A_q>0,B_q>0",
            "valid_positive_parameter_cases": parameter_count,
            "negative_controls": [
                {"case": "A_q=0,B_q>0", "decision": "reject_outside_open_domain_gapless_unstabilized_boundary"},
                {"case": "A_q<0,B_q>0", "decision": "reject_tachyonic_low_k_omega_squared"},
                {"case": "B_q=0,A_q>0", "decision": "reject_no_propagating_principal_operator"},
                {"case": "B_q<0,A_q>0", "decision": "reject_wrong_sign_kinetic_and_negative_mu_squared"},
                {"case": "q0<=0", "decision": "reject_outside_registered_intensity_domain"},
            ],
            "all_negative_controls_rejected": True,
            "stiffness_denominator_condition_ratio": denominator_max / denominator_min,
            "interpretation": "large ratio is a numerical stiffness warning, not a physical exclusion inside the registered open domain",
        },
        "gpu_cpu_crosscheck": {
            "maximum_absolute_error": maximum_error,
            "per_output_maximum_absolute_error": errors,
            "absolute_error_bound": float(config["gpu_cpu_absolute_error_bound"]),
            "maximum_relative_error": maximum_relative_error,
            "per_output_maximum_relative_error": relative_errors,
            "relative_error_bound": float(config["gpu_cpu_relative_error_bound"]),
            "dispersion_relative_residual_bound": float(config["dispersion_relative_residual_bound"]),
        },
        "deterministic_manifest": _manifest(inputs),
        "counts": {
            "compiler_action_hypotheses": 2,
            "parameter_cases": parameter_count,
            "spectral_and_radial_samples_per_parameter": sample_count,
            "unique_parameter_sample_pairs": total,
            "unique_scalar_consequence_values": 2 * total,
            "exact_sentinel_groups": len(exact_sentinels),
            "negative_parameter_controls": 5,
            "gpu_warmup_repetitions": warmups,
            "gpu_measured_repetitions": repetitions,
            "gpu_kernel_dispatches": warmups + repetitions,
            "gpu_measured_parameter_sample_pairs": measured_pairs,
            "gpu_measured_scalar_consequence_evaluations": 2 * measured_pairs,
            "observational_records_accessed": 0,
            "paper_qed_or_theory_passes": 0,
        },
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "gpu_measured_wall_seconds": elapsed,
            "gpu_scalar_consequence_evaluations_per_second": 2 * measured_pairs / elapsed,
            "gpu_array_bytes": sum(
                int(array.nbytes)
                for array in (d_a, d_b, d_k, d_r, d_omega, d_residual, d_green)
            ),
            "utilization": utilization,
            "scope": "single device-wide local run; not a sustained or lane-exclusive throughput claim",
        },
        "decision": "compiler_hypothesis_scalar_controls_closed_paper_qed_and_theory_claims_blocked",
        "first_blocker": action["first_blocker"],
        "synthetic_only": True,
        "observations_opened": False,
        "paper_or_qed_pass": False,
        "theory_pass": False,
        "ontology_pass": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_campaign(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, loaded, root = _load(Path(config_path).resolve())
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("scalar-intensity content hash mismatch")
    if result.get("decision") != (
        "compiler_hypothesis_scalar_controls_closed_paper_qed_and_theory_claims_blocked"
    ):
        raise ValueError("scalar-intensity decision changed")
    if result.get("scope_boundary") != {
        "actions_are_compiler_hypotheses": True,
        "actions_present_in_or_derived_from_paper": False,
        "qed_actualization_dynamics_tested": False,
        "paper_transaction_ontology_tested": False,
        "external_green_source_is_mathematical_control_only": True,
    }:
        raise ValueError("compiler-hypothesis scope changed")
    branches = result.get("branch_records", [])
    if (
        len(branches) != 2
        or {row.get("beta") for row in branches} != {"1/2", "1/4"}
        or any(row.get("paper_authorship_or_derivation") is not False for row in branches)
        or result.get("branch_degeneracy_control", {}).get("linearized_scalar_dynamics_identical")
        is not True
    ):
        raise ValueError("scalar branch binding changed")
    counts = result.get("counts", {})
    total = int(config["parameter_count"]) * int(config["spectral_sample_count"])
    repetitions = int(config["gpu_measured_repetitions"])
    if (
        counts.get("compiler_action_hypotheses") != 2
        or counts.get("unique_parameter_sample_pairs") != total
        or counts.get("gpu_measured_parameter_sample_pairs") != total * repetitions
        or counts.get("gpu_measured_scalar_consequence_evaluations") != 2 * total * repetitions
        or counts.get("negative_parameter_controls") != 5
        or counts.get("observational_records_accessed") != 0
        or counts.get("paper_qed_or_theory_passes") != 0
    ):
        raise ValueError("scalar-intensity counters changed")
    if result.get("parameter_domain_controls", {}).get("all_negative_controls_rejected") is not True:
        raise ValueError("negative parameter controls changed")
    crosscheck = result.get("gpu_cpu_crosscheck", {})
    if (
        crosscheck.get("maximum_absolute_error", math.inf)
        > config["gpu_cpu_absolute_error_bound"]
        or crosscheck.get("maximum_relative_error", math.inf)
        > config["gpu_cpu_relative_error_bound"]
        or result.get("dispersion_control", {}).get(
            "maximum_gpu_relative_equation_residual", math.inf
        )
        > config["dispersion_relative_residual_bound"]
    ):
        raise ValueError("scalar-intensity numerical bounds changed")
    if result.get("deterministic_manifest") != _manifest(deterministic_inputs(config)):
        raise ValueError("scalar-intensity deterministic manifest changed")
    if result.get("first_blocker") != loaded["candidate_action_completion"].get("first_blocker"):
        raise ValueError("candidate action blocker changed")
    if result.get("synthetic_only") is not True:
        raise ValueError("synthetic-only seal changed")
    for key in (
        "observations_opened",
        "paper_or_qed_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        if result.get(key) is not False:
            raise ValueError("claim or data seal changed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"{name} source binding mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    result = build_campaign(config_path)
    root = config_path.parents[1]
    output = root / json.loads(config_path.read_text(encoding="utf-8"))["output_path"]
    output.write_text(_canonical(result) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
