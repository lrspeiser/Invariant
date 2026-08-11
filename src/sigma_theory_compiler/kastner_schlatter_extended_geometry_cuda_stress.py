"""CUDA adversarial controls for hypothetical extended-source KS completions."""

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

SCHEMA = "sigma-kastner-schlatter-extended-geometry-cuda-stress-1.0"
CONFIG_SCHEMA = "sigma-kastner-schlatter-extended-geometry-cuda-stress-config-1.0"

LOCAL_SUPERPOSITION_KERNEL = r"""
extern "C" __global__
void local_point_superposition(
    const double* sx,
    const double* sy,
    const double* sz,
    const double* mass,
    const int source_count,
    const double* evaluation_radius,
    const int evaluation_count,
    double* ax,
    double* ay,
    double* az) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i >= evaluation_count) return;
  double x = evaluation_radius[i];
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  for (int j = 0; j < source_count; ++j) {
    double dx = sx[j] - x;
    double dy = sy[j];
    double dz = sz[j];
    double distance_squared = dx * dx + dy * dy + dz * dz;
    double coefficient = sqrt(mass[j]) / distance_squared;
    sum_x += coefficient * dx;
    sum_y += coefficient * dy;
    sum_z += coefficient * dz;
  }
  ax[i] = sum_x;
  ay[i] = sum_y;
  az[i] = sum_z;
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
        raise ValueError("unsupported extended-geometry CUDA config")
    expected_seals = {
        "synthetic_only": True,
        "observations_opened": False,
        "physical_pass_allowed": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    if any(config.get(key) != value for key, value in expected_seals.items()):
        raise ValueError("extended-geometry data or claim seals changed")
    if config.get("normalization") != "G*a0=1,total_baryonic_mass=1,source_radius<=1":
        raise ValueError("extended-geometry normalization changed")
    loaded: dict[str, dict[str, Any]] = {}
    if set(config.get("predecessors", {})) != {
        "equation_graph",
        "observational_readiness",
        "cuda_falsification_design",
    }:
        raise ValueError("extended-geometry predecessor set changed")
    for name, binding in config["predecessors"].items():
        loaded[name] = _load_predecessor(root, name, binding)
    graph = loaded["equation_graph"]
    readiness = loaded["observational_readiness"]
    design = loaded["cuda_falsification_design"]
    graph_text = _canonical(graph["knowledge_graph"])
    if (
        graph.get("admission_contract", {}).get("equation_only") is not True
        or graph.get("admission_contract", {}).get("fundamental_action") is not None
        or "EQ-KS-69-VELOCITY" not in graph_text
        or readiness.get("lane_decisions", {}).get("mond_btfr")
        != "blocked_no_extended_baryonic_geometry_operator_or_observational_bundle"
        or design.get("btfr_power_control", {}).get("extended_galaxy_geometry_tested") is not False
        or design.get("observations_opened") is not False
    ):
        raise ValueError("extended-source predecessor boundary changed")
    return config, loaded, root


def _antipodal(points: np.ndarray) -> np.ndarray:
    return np.concatenate((points, -points), axis=0)


def deterministic_sources(kind: str, count: int) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0 or count % 2:
        raise ValueError("source count must be positive and even")
    half = count // 2
    index = np.arange(half, dtype=np.float64)
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    angle = golden_angle * index
    if kind == "thin_shell":
        z = 1.0 - 2.0 * (index + 0.5) / half
        radial = np.sqrt(1.0 - z * z)
        base = np.column_stack((radial * np.cos(angle), radial * np.sin(angle), z))
    elif kind == "thin_ring":
        base = np.column_stack((np.cos(angle), np.sin(angle), np.zeros(half)))
    elif kind == "disk_like":
        radial = np.sqrt((index + 0.5) / half)
        base = np.column_stack((radial * np.cos(angle), radial * np.sin(angle), np.zeros(half)))
    elif kind == "spherical_volume":
        radial_3d = ((index + 0.5) / half) ** (1.0 / 3.0)
        z_unit = 1.0 - 2.0 * ((index * 0.6180339887498949) % 1.0)
        planar = np.sqrt(1.0 - z_unit * z_unit)
        base = np.column_stack(
            (
                radial_3d * planar * np.cos(angle),
                radial_3d * planar * np.sin(angle),
                radial_3d * z_unit,
            )
        )
    else:
        raise ValueError(f"unknown synthetic geometry: {kind}")
    positions = np.ascontiguousarray(_antipodal(base), dtype="<f8")
    masses = np.full(count, 1.0 / count, dtype="<f8")
    return positions, masses


def deterministic_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    radii = np.geomspace(
        float(config["far_field_radius_min"]),
        float(config["far_field_radius_max"]),
        int(config["evaluation_radius_count"]),
        dtype=np.float64,
    )
    cases = []
    for geometry in config["geometries"]:
        for count in config["source_counts"]:
            positions, masses = deterministic_sources(geometry, int(count))
            cases.append(
                {
                    "geometry": geometry,
                    "source_count": int(count),
                    "positions": positions,
                    "masses": masses,
                }
            )
    return {"evaluation_radii": np.ascontiguousarray(radii, dtype="<f8"), "cases": cases}


def _cpu_local_superposition(
    positions: np.ndarray, masses: np.ndarray, radii: np.ndarray
) -> np.ndarray:
    result = np.empty((len(radii), 3), dtype=np.float64)
    chunk_size = 128
    root_mass = np.sqrt(masses)[:, None]
    for start in range(0, len(radii), chunk_size):
        stop = min(start + chunk_size, len(radii))
        evaluation = np.zeros((stop - start, 3), dtype=np.float64)
        evaluation[:, 0] = radii[start:stop]
        delta = positions[:, None, :] - evaluation[None, :, :]
        distance_squared = np.sum(delta * delta, axis=2)
        result[start:stop] = np.sum(root_mass[:, :, None] * delta / distance_squared[:, :, None], axis=0)
    return result


def _manifest(inputs: Mapping[str, Any]) -> dict[str, Any]:
    hashes: dict[str, str] = {
        "evaluation_radii": hashlib.sha256(inputs["evaluation_radii"].tobytes()).hexdigest()
    }
    for case in inputs["cases"]:
        label = f"{case['geometry']}_{case['source_count']}"
        hashes[f"{label}_positions"] = hashlib.sha256(case["positions"].tobytes()).hexdigest()
        hashes[f"{label}_masses"] = hashlib.sha256(case["masses"].tobytes()).hexdigest()
    return {"array_sha256": dict(sorted(hashes.items())), "manifest_root_sha256": _sha(hashes)}


def build_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, loaded, root = _load(config_path)
    inputs = deterministic_inputs(config)
    manifest = _manifest(inputs)
    try:
        import cupy as cp
    except Exception as error:
        raise RuntimeError(f"CUDA unavailable: {type(error).__name__}: {error}") from error
    if int(cp.cuda.runtime.getDeviceCount()) < 1:
        raise RuntimeError("no CUDA device")
    device = _device(cp)
    kernel = cp.RawKernel(LOCAL_SUPERPOSITION_KERNEL, "local_point_superposition")
    threads = 256
    d_radii = cp.asarray(inputs["evaluation_radii"])
    jobs: list[dict[str, Any]] = []
    for case in inputs["cases"]:
        positions = case["positions"]
        masses = case["masses"]
        jobs.append(
            {
                "case": case,
                "sx": cp.asarray(positions[:, 0]),
                "sy": cp.asarray(positions[:, 1]),
                "sz": cp.asarray(positions[:, 2]),
                "mass": cp.asarray(masses),
                "ax": cp.empty_like(d_radii),
                "ay": cp.empty_like(d_radii),
                "az": cp.empty_like(d_radii),
            }
        )

    def dispatch() -> None:
        blocks = ((d_radii.size + threads - 1) // threads,)
        for job in jobs:
            kernel(
                blocks,
                (threads,),
                (
                    job["sx"],
                    job["sy"],
                    job["sz"],
                    job["mass"],
                    np.int32(job["case"]["source_count"]),
                    d_radii,
                    np.int32(d_radii.size),
                    job["ax"],
                    job["ay"],
                    job["az"],
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

    case_records = []
    max_cpu_error = 0.0
    max_far_ratio_error = 0.0
    for job in jobs:
        case = job["case"]
        gpu = np.column_stack(
            (cp.asnumpy(job["ax"]), cp.asnumpy(job["ay"]), cp.asnumpy(job["az"]))
        )
        cpu = _cpu_local_superposition(case["positions"], case["masses"], inputs["evaluation_radii"])
        error = float(np.max(np.abs(gpu - cpu)))
        max_cpu_error = max(max_cpu_error, error)
        far_acceleration = float(np.linalg.norm(gpu[-1]))
        point_mass_acceleration = 1.0 / float(inputs["evaluation_radii"][-1])
        measured_ratio = far_acceleration / point_mass_acceleration
        predicted_ratio = math.sqrt(case["source_count"])
        relative_error = abs(measured_ratio - predicted_ratio) / predicted_ratio
        max_far_ratio_error = max(max_far_ratio_error, relative_error)
        case_records.append(
            {
                "geometry": case["geometry"],
                "source_count": case["source_count"],
                "total_mass": float(np.sum(case["masses"])),
                "gpu_cpu_max_absolute_component_error": error,
                "far_field_local_superposition_to_point_mass_ratio": measured_ratio,
                "coincident_split_prediction_sqrt_N": predicted_ratio,
                "relative_error_to_sqrt_N_far_coefficient": relative_error,
                "point_mass_asymptote_recovered": relative_error
                <= float(config["point_mass_recovery_relative_bound"])
                and case["source_count"] == 1,
            }
        )
    if max_cpu_error > float(config["gpu_cpu_absolute_error_bound"]):
        raise ValueError("extended-geometry GPU/CPU crosscheck failed")
    if max_far_ratio_error > float(config["far_field_sqrt_n_relative_error_bound"]):
        raise ValueError("far-field discretization sensitivity control failed")

    source_counts = [int(value) for value in config["source_counts"]]
    split_controls = [
        {
            "source_count": count,
            "exact_local_superposition_coincident_split_ratio": math.sqrt(count),
            "point_mass_recovery": count == 1,
        }
        for count in source_counts
    ]
    mass_one = 0.25
    mass_two = 0.75
    force_on_one = mass_one * math.sqrt(mass_two)
    force_on_two = mass_two * math.sqrt(mass_one)
    pair_net_force = force_on_two - force_on_one
    interactions_per_dispatch = int(
        len(inputs["evaluation_radii"]) * sum(case["source_count"] for case in inputs["cases"])
    )
    source_path = root / "src/sigma_theory_compiler/kastner_schlatter_extended_geometry_cuda_stress.py"
    test_path = root / "tests/test_kastner_schlatter_extended_geometry_cuda_stress.py"
    graph_nodes = loaded["equation_graph"]["knowledge_graph"]["nodes"]
    graph_node_ids = {node["node_id"] for node in graph_nodes}
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "equation_graph": config["predecessors"]["equation_graph"],
            "observational_readiness": config["predecessors"]["observational_readiness"],
            "cuda_falsification_design": config["predecessors"]["cuda_falsification_design"],
            "config": {"path": config_path.relative_to(root).as_posix(), "file_sha256": _file_sha(config_path)},
            "source": {"path": source_path.relative_to(root).as_posix(), "file_sha256": _file_sha(source_path)},
            "test": {"path": test_path.relative_to(root).as_posix(), "file_sha256": _file_sha(test_path)},
        },
        "paper_boundary": {
            "registered_point_mass_nodes": sorted(
                graph_node_ids & {"EQ-KS-62-SDS-MOND", "EQ-KS-68-DEEP-ACCELERATION", "EQ-KS-69-VELOCITY"}
            ),
            "point_mass_relation": "abar=sqrt(G*M*a0)/r and v^4=G*M*a0 in the declared approximation domain",
            "extended_source_operator_registered": False,
            "covariant_extended_metric_registered": False,
            "lensing_deflection_operator_registered": False,
            "source_not_claimed": ["galaxy-data likelihood pass", "lensing or cluster agreement", "dark-matter elimination"],
        },
        "completion_hypotheses": {
            "H_enclosed_mass": {
                "rule": "replace M by baryonic M(<r) in the point-mass acceleration",
                "source_status": "unproved_hypothesis_not_in_paper_equation_graph",
                "far_point_mass_asymptote": "preserved exactly for r beyond compact support",
                "total_mass_conservation_control": "pass_for_synthetic_equal_mass_quadrature",
                "geometry_blind_shell_ring_control": "fails_to_distinguish_thin_shell_from_thin_ring_with_same_radial_mass_support",
                "covariant_or_lensing_completion": "absent",
                "decision": "blocked_not_a_registered_extended_source_law",
            },
            "H_local_superposition": {
                "rule": "sum vector accelerations sqrt(G*m_j*a0)/distance_j from point elements",
                "source_status": "unproved_hypothesis_not_in_paper_equation_graph",
                "point_mass_baseline": {
                    "source_count": 1,
                    "far_field_ratio": 1,
                    "status": "source_formula_control_not_an_extended_completion",
                },
                "coincident_split_controls": split_controls,
                "unequal_pair_matter_force_control": {
                    "m1": mass_one,
                    "m2": mass_two,
                    "normalized_force_on_m1": force_on_one,
                    "normalized_force_on_m2": force_on_two,
                    "normalized_net_matter_force": pair_net_force,
                    "exact_force_on_m1": "sqrt(3)/8",
                    "exact_force_on_m2": "3/8",
                    "exact_net_matter_force": "(3-sqrt(3))/8",
                    "action_reaction_balance": False,
                    "scope": "matter-only naive pair rule; no claim about field momentum without a covariant completion",
                },
                "far_field_case_records": case_records,
                "point_mass_aggregation_invariant": False,
                "continuum_discretization_invariant": False,
                "decision": "hypothesis_rejected_by_exact_splitting_and_pair_balance_controls",
            },
        },
        "lensing_rotation_consistency_gate": {
            "executed": False,
            "decision": "blocked",
            "first_missing_field": "source_supported_covariant_extended_metric_and_null_geodesic_deflection_operator",
            "rotation_curve_values_promoted": 0,
            "lensing_values_emitted": 0,
        },
        "gpu_cpu_crosscheck": {
            "maximum_absolute_component_error": max_cpu_error,
            "absolute_error_bound": float(config["gpu_cpu_absolute_error_bound"]),
            "maximum_far_coefficient_relative_error_to_sqrt_N": max_far_ratio_error,
            "far_coefficient_relative_error_bound": float(config["far_field_sqrt_n_relative_error_bound"]),
        },
        "deterministic_manifest": manifest,
        "counts": {
            "synthetic_geometry_classes": len(config["geometries"]),
            "source_resolutions": len(source_counts),
            "geometry_resolution_cases": len(inputs["cases"]),
            "evaluation_radii_per_case": len(inputs["evaluation_radii"]),
            "unique_source_evaluation_interactions": interactions_per_dispatch,
            "gpu_warmup_repetitions": warmups,
            "gpu_measured_repetitions": repetitions,
            "gpu_kernel_dispatches": len(jobs) * (warmups + repetitions),
            "gpu_measured_source_evaluation_interactions": interactions_per_dispatch * repetitions,
            "extended_source_laws_registered": 0,
            "lensing_cases_executed": 0,
            "observational_records_accessed": 0,
            "physical_or_theory_passes": 0,
        },
        "runtime_measurement": {
            "measured_utc": datetime.now(UTC).isoformat(),
            "device": device,
            "gpu_measured_wall_seconds": elapsed,
            "gpu_source_evaluation_interactions_per_second": interactions_per_dispatch * repetitions / elapsed,
            "gpu_array_bytes": int(d_radii.nbytes)
            + sum(
                int(job[key].nbytes)
                for job in jobs
                for key in ("sx", "sy", "sz", "mass", "ax", "ay", "az")
            ),
            "utilization": utilization,
            "scope": "single device-wide local run; not a sustained or lane-exclusive throughput claim",
        },
        "decision": "no_source_supported_extended_completion_local_superposition_rejected_enclosed_mass_blocked",
        "interpretation": "CUDA establishes implementation sensitivity of declared hypotheses only; it does not test the paper, a galaxy, lensing, or gravity",
        "synthetic_only": True,
        "observations_opened": False,
        "physical_pass": False,
        "theory_pass": False,
        "ontology_pass": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_or_cosmology_inputs": False,
        "paid_llm_calls": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_campaign(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, _, root = _load(Path(config_path).resolve())
    if result.get("content_sha256") != _content_sha(result):
        raise ValueError("extended-geometry content hash mismatch")
    if result.get("decision") != (
        "no_source_supported_extended_completion_local_superposition_rejected_enclosed_mass_blocked"
    ):
        raise ValueError("extended-geometry decision changed")
    if result.get("paper_boundary", {}).get("extended_source_operator_registered") is not False:
        raise ValueError("extended source operator was invented")
    if result.get("lensing_rotation_consistency_gate", {}).get("executed") is not False:
        raise ValueError("unsupported lensing gate was executed")
    local = result.get("completion_hypotheses", {}).get("H_local_superposition", {})
    if (
        local.get("point_mass_aggregation_invariant") is not False
        or local.get("continuum_discretization_invariant") is not False
        or local.get("unequal_pair_matter_force_control", {}).get("action_reaction_balance")
        is not False
    ):
        raise ValueError("local-superposition obstruction changed")
    counts = result.get("counts", {})
    expected_interactions = (
        int(config["evaluation_radius_count"])
        * len(config["geometries"])
        * sum(int(value) for value in config["source_counts"])
    )
    if (
        counts.get("unique_source_evaluation_interactions") != expected_interactions
        or counts.get("gpu_measured_source_evaluation_interactions")
        != expected_interactions * int(config["gpu_measured_repetitions"])
        or counts.get("extended_source_laws_registered") != 0
        or counts.get("lensing_cases_executed") != 0
        or counts.get("observational_records_accessed") != 0
        or counts.get("physical_or_theory_passes") != 0
    ):
        raise ValueError("extended-geometry counters changed")
    for key in ("synthetic_only",):
        if result.get(key) is not True:
            raise ValueError("synthetic-only seal changed")
    for key in (
        "observations_opened",
        "physical_pass",
        "theory_pass",
        "ontology_pass",
        "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
        "paid_llm_calls",
    ):
        if result.get(key) is not False:
            raise ValueError("claim or data seal changed")
    crosscheck = result.get("gpu_cpu_crosscheck", {})
    if (
        crosscheck.get("maximum_absolute_component_error", math.inf)
        > config["gpu_cpu_absolute_error_bound"]
        or crosscheck.get("maximum_far_coefficient_relative_error_to_sqrt_N", math.inf)
        > config["far_field_sqrt_n_relative_error_bound"]
    ):
        raise ValueError("GPU or far-field bound changed")
    if result.get("deterministic_manifest") != _manifest(deterministic_inputs(config)):
        raise ValueError("deterministic geometry manifest changed")
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
