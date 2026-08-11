"""Bounded CPU-only overlap campaign over unique real formula ordinals."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing as mp
import os
import tempfile
import threading
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from statistics import mean, median
from typing import Any

from .persistent_parallel_search import WorkLease

CONFIG_SCHEMA = "sigma-cpu-symbolic-overlap-benchmark-config-1.1"
ARTIFACT_SCHEMA = "sigma-cpu-real-formula-overlap-benchmark-1.1"
EVALUATOR = "sigma_theory_compiler.real_formula_execution:cpu_formula_batch_evaluator"
START = 1_000_000_000
STOP = 1_000_065_536
UNIQUE_FORMULAS = 65_536
GRID_POINTS = 343
TERM_COUNT = 6
CANDIDATE_GRID_EVALUATIONS = UNIQUE_FORMULAS * GRID_POINTS
SIGNED_TERM_HESSIAN_ACCUMULATIONS = CANDIDATE_GRID_EVALUATIONS * TERM_COUNT
SIX_TERM_TIER_START = sum(math.comb(50, count) * (1 << count) for count in range(1, 6))
EXPECTED_STAGES = [15, 16]
FIXED_SHARD_SIZE = 1_024
FIXED_SHARD_COUNT = 64
EXPECTED_GRID = {
    "d": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
    "p": [0.0, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0],
    "state": [0.0, 0.03, 0.1, 0.3, 0.5, 1.0, 3.0],
}
EXPECTED_SEALS = {
    "sqlite_access": False,
    "gpu_or_cuda_access": False,
    "existing_process_signaled": False,
    "arbitrary_callable_or_subprocess_injection": False,
    "observations_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "scientific_or_resource_policy_promotion": False,
    "paid_llm_calls": False,
}
EXPECTED_INTERPRETATION = (
    "This extends sampled-static formula coverage and measures CPU overlap. A sampled-static "
    "pass is not a covariant-health, theory, formal, observational, or ranking pass; a "
    "screen reject is not a theory rejection. Fixed-interval host CPU samples and summed "
    "worker process CPU time are hardware measurements, not a resource-policy change."
)
EXPECTED_TOP_LEVEL_KEYS = {
    "campaign_id",
    "content_sha256",
    "contract",
    "coverage",
    "cpu_target_met",
    "cross_stage_replay_equal",
    "decision",
    "evaluator_controls_passed",
    "existing_process_signaled",
    "gpu_or_cuda_accessed",
    "hard_total_deadline_enforced",
    "hardware_attestation",
    "interpretation",
    "observations_opened",
    "overlap_control_executed",
    "overlap_control_replay_formula_evaluations",
    "resource_backoff_triggered",
    "resource_policy_promoted",
    "schema_version",
    "scientific_pass",
    "seals",
    "source_bindings",
    "sqlite_accessed",
    "stage_16_admitted",
    "stage_16_blocker",
    "stages",
    "total_bound_respected",
    "total_elapsed_seconds",
    "total_formula_evaluator_executions",
}
EXPECTED_STAGE_KEYS = {
    "all_evaluator_controls_passed",
    "ambiguous_are_not_passes",
    "backend",
    "backoff_threshold_exceeded",
    "batch_count",
    "candidate_grid_evaluations",
    "candidate_grid_evaluations_per_second",
    "content_sha256",
    "counts",
    "cpu_percent_mean",
    "cpu_percent_median",
    "cpu_percent_peak",
    "cpu_sample_count",
    "cpu_sample_interval_seconds",
    "cpu_sampling_contract",
    "cpu_target_met_by_median",
    "cpu_target_percent",
    "elapsed_seconds",
    "exact_gap_overlap_duplicate_free_coverage",
    "fixed_shard_count",
    "fixed_shard_manifest_root_sha256",
    "fixed_shard_size",
    "gpu_workers",
    "hard_deadline_enforced",
    "hard_deadline_triggered",
    "interval",
    "minimum_available_ram_mib",
    "owned_worker_termination_count",
    "partition_independent_status_root_sha256",
    "reported_margin_minimum",
    "signed_term_hessian_accumulations",
    "unique_formula_count",
    "unique_formulas_per_second",
    "wall_bound_exceeded",
    "worker_cpu_capacity_percent",
    "worker_cpu_seconds",
    "workers",
}
EXPECTED_HARDWARE_KEYS = {
    "installed_ram_mib",
    "logical_processors",
    "physical_cores",
    "profile_matches_host_topology",
    "profile_reserved_cpu_cores",
    "profile_sustained_cpu_workers",
    "resource_profile_file_sha256",
}
MEASURED_STAGE_RECEIPTS = {
    15: {
        "cpu_sample_count": 10,
        "cpu_percent_mean": 68.77,
        "cpu_percent_median": 74.3,
        "cpu_percent_peak": 81.1,
    },
    16: {
        "cpu_sample_count": 10,
        "cpu_percent_mean": 87.17,
        "cpu_percent_median": 89.4,
        "cpu_percent_peak": 100.0,
    },
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _content_sha(value: Mapping[str, Any]) -> str:
    return _sha({key: item for key, item in value.items() if key != "content_sha256"})


def _inside(root: Path, relative: str) -> Path:
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("CPU formula overlap path escapes repository") from error
    return target


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(config_path).resolve()
    root = path.parents[1]
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "campaign_id",
        "output_path",
        "worker_stages",
        "start_ordinal",
        "stop_ordinal_exclusive",
        "unique_formula_count",
        "grid_point_count",
        "term_count",
        "maximum_stage_seconds",
        "maximum_total_seconds",
        "sample_interval_seconds",
        "shutdown_reserve_seconds",
        "fixed_shard_size",
        "fixed_shard_count",
        "minimum_available_ram_mib",
        "cpu_backoff_above_percent",
        "cpu_target_percent",
        "gpu_workers",
        "start_method",
        "maximum_output_bytes",
        "allowlisted_evaluator",
        "basis_library_sha256",
        "grid_sha256",
        "ambiguity_guard",
        "bindings",
        "seals",
    }
    if set(config) != required or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("unsupported CPU formula overlap config")
    if (
        config.get("worker_stages") != EXPECTED_STAGES
        or config.get("start_ordinal") != START
        or config.get("stop_ordinal_exclusive") != STOP
        or config.get("unique_formula_count") != UNIQUE_FORMULAS
        or config.get("grid_point_count") != GRID_POINTS
        or config.get("term_count") != TERM_COUNT
        or config.get("maximum_stage_seconds") != 120
        or config.get("maximum_total_seconds") != 240
        or config.get("shutdown_reserve_seconds") != 1.0
        or config.get("fixed_shard_size") != FIXED_SHARD_SIZE
        or config.get("fixed_shard_count") != FIXED_SHARD_COUNT
        or config.get("minimum_available_ram_mib") != 32768
        or config.get("cpu_backoff_above_percent") != 92
        or config.get("cpu_target_percent") != 80
        or config.get("gpu_workers") != 0
        or config.get("start_method") != "spawn"
        or config.get("allowlisted_evaluator") != EVALUATOR
        or config.get("basis_library_sha256")
        != "f6e09a44eddd20999c8a5c3d3e1e002efb9c892deb706df60669a47f1f7f3840"
        or config.get("grid_sha256")
        != "9212ebf18831e3604088ce48e96fd5bb5b842e6c2a5126de6ed934ece146ed24"
        or config.get("seals") != EXPECTED_SEALS
    ):
        raise ValueError("CPU formula overlap closed contract changed")
    if float(config["sample_interval_seconds"]) != 0.1:
        raise ValueError("CPU formula overlap sample interval changed")
    if not 0 < int(config["maximum_output_bytes"]) <= 1_048_576:
        raise ValueError("CPU formula overlap output bound changed")
    expected_bindings = {
        "evaluator_source",
        "generator_config",
        "high_throughput_source",
        "gpu_screen_source",
        "prior_execution_contract",
        "authoritative_manifest",
        "resource_profile",
    }
    if set(config.get("bindings", {})) != expected_bindings:
        raise ValueError("CPU formula overlap binding set changed")
    for name, binding in config["bindings"].items():
        bound = _inside(root, binding["path"])
        if _file_sha(bound) != binding["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    _validate_domain_contract(config, root)
    return config, root


def _validate_domain_contract(config: Mapping[str, Any], root: Path) -> None:
    from .gpu_screen import dense_grid
    from .high_throughput import build_basis, candidate_id, decode_ordinal, total_search_count

    generator_path = _inside(root, config["bindings"]["generator_config"]["path"])
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    prior = json.loads(
        _inside(root, config["bindings"]["prior_execution_contract"]["path"]).read_text()
    )
    manifest = json.loads(
        _inside(root, config["bindings"]["authoritative_manifest"]["path"]).read_text()
    )
    basis = build_basis(int(generator["basis_count"]))
    basis_payload = json.dumps(basis, separators=(",", ":"), ensure_ascii=False).encode()
    first = decode_ordinal(50, 6, START)
    last = decode_ordinal(50, 6, STOP - 1)
    if (
        total_search_count(50, 6) != 1_088_651_720
        or STOP - START != UNIQUE_FORMULAS
        or START < SIX_TERM_TIER_START
        or prior.get("stop_ordinal_exclusive") != START
        or prior.get("start_ordinal") != 0
        or manifest.get("total_declared_actions") != 1_088_651_720
        or manifest.get("basis_library_sha256") != config["basis_library_sha256"]
        or hashlib.sha256(basis_payload).hexdigest() != config["basis_library_sha256"]
        or dense_grid() != EXPECTED_GRID
        or _sha(dense_grid()) != config["grid_sha256"]
        or first.get("term_ids") != [15, 26, 28, 33, 36, 42]
        or first.get("signs") != [-1, -1, -1, 1, 1, 1]
        or candidate_id(generator["protocol_version"], first) != "STC2-f9eacf5b6cfe44a67d38f940"
        or last.get("term_ids") != [15, 26, 29, 32, 36, 46]
        or last.get("signs") != [1, 1, 1, -1, 1, 1]
        or candidate_id(generator["protocol_version"], last) != "STC2-eec3c7ee83d030875ab452ce"
        or len(first["term_ids"]) != TERM_COUNT
        or len(last["term_ids"]) != TERM_COUNT
    ):
        raise ValueError("CPU formula overlap domain contract failed")


def _fixed_shards() -> list[tuple[int, int]]:
    shards = [
        (start, min(start + FIXED_SHARD_SIZE, STOP))
        for start in range(START, STOP, FIXED_SHARD_SIZE)
    ]
    if (
        len(shards) != FIXED_SHARD_COUNT
        or shards[0][0] != START
        or shards[-1][1] != STOP
        or any(left[1] != right[0] for left, right in pairwise(shards))
    ):
        raise RuntimeError("CPU formula fixed-shard partition is not exact")
    return shards


def _formula_batch_job(
    shard_index: int,
    start: int,
    stop: int,
    generator_path: str,
    generator_sha256: str,
    ambiguity_guard: float,
) -> dict[str, Any]:
    from .real_formula_execution import (
        cpu_formula_batch_evaluator,
        validate_formula_batch_result,
    )

    payload = {
        "ordinal": start,
        "batch_sequence": shard_index,
        "start_ordinal": start,
        "end_ordinal_exclusive": stop,
        "candidate_count": stop - start,
        "basis_count": 50,
        "max_action_terms": 6,
        "protocol_version": "SIGMA-GENERATOR-V2-BILLION-1.0.0",
        "generator_config_path": generator_path,
        "generator_config_sha256": generator_sha256,
        "ambiguity_guard": ambiguity_guard,
        "data_eligibility": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
        },
    }
    process_started = time.process_time()
    lease = WorkLease(
        work_id=f"cpu-overlap-{start}-{stop}",
        ordinal=start,
        lane="cpu",
        seed=start,
        attempt=shard_index + 1,
        max_attempts=1,
        payload=payload,
    )
    result = cpu_formula_batch_evaluator(lease)
    validate_formula_batch_result(result, payload)
    if (
        result.get("backend") != "cpu_numpy"
        or result.get("batch", {}).get("candidate_count") != stop - start
        or not math.isfinite(float(result.get("reported_margin_minimum")))
        or not math.isfinite(float(result.get("elapsed_seconds")))
        or float(result.get("elapsed_seconds")) <= 0
        or not _is_sha256(result.get("status_root_sha256"))
    ):
        raise ValueError("CPU real formula batch result failed closed")
    result["fixed_shard_index"] = shard_index
    result["worker_cpu_seconds"] = time.process_time() - process_started
    return result


def _resource_sample(interval: float = 0.0) -> dict[str, Any]:
    import psutil

    cpu_percent = float(psutil.cpu_percent(interval=interval))
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": cpu_percent,
        "available_ram_mib": int(memory.available // 1024**2),
        "sampled_utc": datetime.now(UTC).isoformat(),
        "interval_seconds": interval,
    }


def _hardware_attestation(config: Mapping[str, Any], root: Path) -> dict[str, Any]:
    import psutil

    profile = json.loads(_inside(root, config["bindings"]["resource_profile"]["path"]).read_text())
    expected = profile["hardware"]
    actual = {
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_processors": psutil.cpu_count(logical=True),
        "installed_ram_mib": int(psutil.virtual_memory().total // 1024**2),
    }
    matches = (
        actual["physical_cores"] == expected["physical_cores"]
        and actual["logical_processors"] == expected["logical_processors"]
        and actual["installed_ram_mib"] >= int(float(expected["ram_gib"]) * 1024) - 1024
        and profile["production_lanes"]["cpu_symbolic"]["sustained_workers"] == 16
        and profile["safety"]["reserve_cpu_cores_for_os_database_and_gpu_feeder"] == 8
    )
    return {
        **actual,
        "resource_profile_file_sha256": config["bindings"]["resource_profile"]["file_sha256"],
        "profile_sustained_cpu_workers": 16,
        "profile_reserved_cpu_cores": 8,
        "profile_matches_host_topology": matches,
    }


def _fixed_interval_sampler(
    stop: threading.Event,
    samples: list[dict[str, Any]],
    interval: float,
) -> None:
    while not stop.is_set():
        samples.append(_resource_sample(interval))


def _terminate_owned_executor(executor: concurrent.futures.ProcessPoolExecutor) -> int:
    processes = list(getattr(executor, "_processes", {}).values())
    count = 0
    for process in processes:
        if process.is_alive():
            process.terminate()
            count += 1
    join_deadline = time.perf_counter() + 0.5
    for process in processes:
        process.join(timeout=max(0.0, join_deadline - time.perf_counter()))
    executor.shutdown(wait=False, cancel_futures=True)
    return count


def _run_stage(
    config: Mapping[str, Any],
    root: Path,
    workers: int,
    overall_deadline: float | None = None,
) -> dict[str, Any]:
    interval = float(config["sample_interval_seconds"])
    before = _resource_sample(interval)
    if before["available_ram_mib"] < int(config["minimum_available_ram_mib"]):
        raise RuntimeError("CPU formula overlap RAM floor not met")
    if before["cpu_percent"] > float(config["cpu_backoff_above_percent"]):
        raise RuntimeError("CPU formula overlap preflight backoff threshold exceeded")
    shards = _fixed_shards()
    generator_path = _inside(root, config["bindings"]["generator_config"]["path"])
    generator_sha = config["bindings"]["generator_config"]["file_sha256"]
    context = mp.get_context(str(config["start_method"]))
    started = time.perf_counter()
    reserve = float(config["shutdown_reserve_seconds"])
    hard_stop = started + float(config["maximum_stage_seconds"]) - reserve
    if overall_deadline is not None:
        hard_stop = min(hard_stop, overall_deadline - reserve)
    results: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    stop_sampler = threading.Event()
    sampler = threading.Thread(
        target=_fixed_interval_sampler,
        args=(stop_sampler, samples, interval),
        daemon=True,
    )
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context)
    deadline_triggered = False
    owned_worker_termination_count = 0
    executor_shutdown = False
    sampler.start()
    try:
        pending = {
            executor.submit(
                _formula_batch_job,
                shard_index,
                start,
                stop,
                str(generator_path),
                generator_sha,
                float(config["ambiguity_guard"]),
            )
            for shard_index, (start, stop) in enumerate(shards)
        }
        while pending:
            remaining = hard_stop - time.perf_counter()
            if remaining <= 0:
                deadline_triggered = True
                break
            done, pending = concurrent.futures.wait(
                pending,
                timeout=min(interval, remaining),
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            results.extend(future.result() for future in done)
        if deadline_triggered:
            for future in pending:
                future.cancel()
            owned_worker_termination_count = _terminate_owned_executor(executor)
            executor_shutdown = True
        else:
            executor.shutdown(wait=True)
            executor_shutdown = True
    finally:
        if not executor_shutdown:
            _terminate_owned_executor(executor)
        stop_sampler.set()
        sampler.join(timeout=interval + 0.1)
    elapsed = time.perf_counter() - started
    results.sort(key=lambda item: int(item["batch"]["start_ordinal"]))
    cpu_values = [float(item["cpu_percent"]) for item in samples]
    ram_values = [int(item["available_ram_mib"]) for item in samples]
    total_counts = {
        key: sum(int(item["counts"][key]) for item in results)
        for key in ("reject", "pass", "ambiguous")
    }
    covered = sum(item["batch"]["candidate_count"] for item in results)
    exact_coverage = (
        covered == UNIQUE_FORMULAS
        and len(results) == FIXED_SHARD_COUNT
        and results[0]["batch"]["start_ordinal"] == START
        and results[-1]["batch"]["end_ordinal_exclusive"] == STOP
        and [item["fixed_shard_index"] for item in results] == list(range(FIXED_SHARD_COUNT))
        and all(
            left["batch"]["end_ordinal_exclusive"] == right["batch"]["start_ordinal"]
            for left, right in pairwise(results)
        )
    )
    all_passed = (
        exact_coverage
        and sum(total_counts.values()) == UNIQUE_FORMULAS
        and not deadline_triggered
        and owned_worker_termination_count == 0
        and elapsed <= float(config["maximum_stage_seconds"])
    )
    shard_status_root = _sha(
        [
            {
                "shard_index": item["fixed_shard_index"],
                "start": item["batch"]["start_ordinal"],
                "stop": item["batch"]["end_ordinal_exclusive"],
                "status_root_sha256": item["status_root_sha256"],
            }
            for item in results
        ]
    )
    worker_cpu_seconds = sum(float(item["worker_cpu_seconds"]) for item in results)
    logical_processors = int(_hardware_attestation(config, root)["logical_processors"])
    body: dict[str, Any] = {
        "workers": workers,
        "batch_count": len(results),
        "fixed_shard_count": FIXED_SHARD_COUNT,
        "fixed_shard_size": FIXED_SHARD_SIZE,
        "backend": "cpu_numpy",
        "gpu_workers": 0,
        "interval": {"start": START, "stop": STOP},
        "unique_formula_count": UNIQUE_FORMULAS,
        "candidate_grid_evaluations": CANDIDATE_GRID_EVALUATIONS,
        "signed_term_hessian_accumulations": SIGNED_TERM_HESSIAN_ACCUMULATIONS,
        "counts": total_counts,
        "ambiguous_are_not_passes": True,
        "exact_gap_overlap_duplicate_free_coverage": exact_coverage,
        "all_evaluator_controls_passed": all_passed,
        "elapsed_seconds": elapsed,
        "unique_formulas_per_second": UNIQUE_FORMULAS / elapsed,
        "candidate_grid_evaluations_per_second": CANDIDATE_GRID_EVALUATIONS / elapsed,
        "partition_independent_status_root_sha256": shard_status_root,
        "fixed_shard_manifest_root_sha256": _sha(shards),
        "reported_margin_minimum": min(
            (float(item["reported_margin_minimum"]) for item in results),
            default=math.nan,
        ),
        "worker_cpu_seconds": worker_cpu_seconds,
        "worker_cpu_capacity_percent": 100 * worker_cpu_seconds / (elapsed * logical_processors),
        "cpu_sample_count": len(cpu_values),
        "cpu_sampling_contract": "fixed_interval_blocking_device_wide_psutil",
        "cpu_sample_interval_seconds": interval,
        "cpu_percent_mean": mean(cpu_values) if cpu_values else 0.0,
        "cpu_percent_median": median(cpu_values) if cpu_values else 0.0,
        "cpu_percent_peak": max(cpu_values, default=0.0),
        "minimum_available_ram_mib": min(ram_values, default=before["available_ram_mib"]),
        "cpu_target_percent": config["cpu_target_percent"],
        "cpu_target_met_by_median": bool(cpu_values)
        and median(cpu_values) >= config["cpu_target_percent"],
        "backoff_threshold_exceeded": max(cpu_values, default=0.0)
        > config["cpu_backoff_above_percent"],
        "hard_deadline_enforced": True,
        "hard_deadline_triggered": deadline_triggered,
        "owned_worker_termination_count": owned_worker_termination_count,
        "wall_bound_exceeded": elapsed > float(config["maximum_stage_seconds"]),
    }
    return {**body, "content_sha256": _content_sha(body)}


def execute_campaign(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, root = load_config(config_path)
    hardware_attestation = _hardware_attestation(config, root)
    if hardware_attestation["profile_matches_host_topology"] is not True:
        raise RuntimeError("CPU formula overlap hardware/profile attestation failed")
    started = time.perf_counter()
    overall_deadline = started + float(config["maximum_total_seconds"])
    stages = [_run_stage(config, root, 15, overall_deadline)]
    stage_16_admitted = (
        stages[0]["all_evaluator_controls_passed"]
        and not stages[0]["backoff_threshold_exceeded"]
        and stages[0]["minimum_available_ram_mib"] >= config["minimum_available_ram_mib"]
        and time.perf_counter() < overall_deadline - float(config["shutdown_reserve_seconds"])
    )
    blocker = None if stage_16_admitted else "stage_15_control_or_resource_guard"
    if stage_16_admitted:
        preflight = _resource_sample(float(config["sample_interval_seconds"]))
        if preflight["available_ram_mib"] < config["minimum_available_ram_mib"]:
            stage_16_admitted, blocker = False, "ram_floor"
        elif preflight["cpu_percent"] > config["cpu_backoff_above_percent"]:
            stage_16_admitted, blocker = False, "cpu_backoff"
    if stage_16_admitted:
        try:
            stages.append(_run_stage(config, root, 16, overall_deadline))
        except RuntimeError as error:
            resource_blockers = {
                "CPU formula overlap RAM floor not met": "ram_floor",
                "CPU formula overlap preflight backoff threshold exceeded": "cpu_backoff",
            }
            if str(error) not in resource_blockers:
                raise
            stage_16_admitted = False
            blocker = resource_blockers[str(error)]
    total_elapsed = time.perf_counter() - started
    cross_stage_replay_equal = (
        all(
            stages[0][key] == stages[1][key]
            for key in (
                "counts",
                "partition_independent_status_root_sha256",
                "reported_margin_minimum",
                "fixed_shard_manifest_root_sha256",
            )
        )
        if len(stages) == 2
        else None
    )
    total_bound_respected = total_elapsed <= float(config["maximum_total_seconds"])
    exact_pass = (
        all(stage["all_evaluator_controls_passed"] for stage in stages)
        and cross_stage_replay_equal is not False
        and total_bound_respected
    )
    target_met = any(stage["cpu_target_met_by_median"] for stage in stages)
    resource_backoff_triggered = any(stage["backoff_threshold_exceeded"] for stage in stages)
    result: dict[str, Any] = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            **config["bindings"],
            "config": {
                "path": config_path.relative_to(root).as_posix(),
                "file_sha256": _file_sha(config_path),
            },
            "source": {
                "path": "src/sigma_theory_compiler/cpu_symbolic_overlap_benchmark.py",
                "file_sha256": _file_sha(Path(__file__).resolve()),
            },
            "test": {
                "path": "tests/test_cpu_symbolic_overlap_benchmark.py",
                "file_sha256": _file_sha(root / "tests/test_cpu_symbolic_overlap_benchmark.py"),
            },
        },
        "coverage": {
            "interval": {"start": START, "stop": STOP},
            "disjoint_from_prior_execution_stop": True,
            "unique_formula_count": UNIQUE_FORMULAS,
            "term_count": TERM_COUNT,
            "six_term_tier_start_ordinal": SIX_TERM_TIER_START,
            "grid_point_count": GRID_POINTS,
            "candidate_grid_evaluations": CANDIDATE_GRID_EVALUATIONS,
            "signed_term_hessian_accumulations": SIGNED_TERM_HESSIAN_ACCUMULATIONS,
            "first_candidate_id": "STC2-f9eacf5b6cfe44a67d38f940",
            "last_candidate_id": "STC2-eec3c7ee83d030875ab452ce",
        },
        "contract": {
            "worker_stages": config["worker_stages"],
            "maximum_stage_seconds": config["maximum_stage_seconds"],
            "maximum_total_seconds": config["maximum_total_seconds"],
            "minimum_available_ram_mib": config["minimum_available_ram_mib"],
            "cpu_backoff_above_percent": config["cpu_backoff_above_percent"],
            "cpu_target_percent": config["cpu_target_percent"],
            "gpu_workers": 0,
            "allowlisted_evaluator": EVALUATOR,
            "sample_interval_seconds": config["sample_interval_seconds"],
            "shutdown_reserve_seconds": config["shutdown_reserve_seconds"],
            "fixed_shard_size": FIXED_SHARD_SIZE,
            "fixed_shard_count": FIXED_SHARD_COUNT,
        },
        "hardware_attestation": hardware_attestation,
        "stage_16_admitted": stage_16_admitted,
        "stage_16_blocker": blocker,
        "stages": stages,
        "overlap_control_executed": len(stages) == 2,
        "cross_stage_replay_equal": cross_stage_replay_equal,
        "total_formula_evaluator_executions": UNIQUE_FORMULAS * len(stages),
        "overlap_control_replay_formula_evaluations": (UNIQUE_FORMULAS if len(stages) == 2 else 0),
        "total_elapsed_seconds": total_elapsed,
        "hard_total_deadline_enforced": True,
        "total_bound_respected": total_bound_respected,
        "evaluator_controls_passed": exact_pass,
        "cpu_target_met": target_met,
        "resource_backoff_triggered": resource_backoff_triggered,
        "decision": (
            "real_formula_cpu_overlap_completed_target_met_no_policy_promotion"
            if exact_pass and target_met
            else "real_formula_cpu_overlap_completed_target_not_met_no_policy_promotion"
            if exact_pass
            else "real_formula_cpu_overlap_failed_closed"
        ),
        "interpretation": EXPECTED_INTERPRETATION,
        "seals": config["seals"],
        "sqlite_accessed": False,
        "gpu_or_cuda_accessed": False,
        "existing_process_signaled": False,
        "observations_opened": False,
        "scientific_pass": False,
        "resource_policy_promoted": False,
    }
    result["content_sha256"] = _content_sha(result)
    return result


def validate_artifact(result: Mapping[str, Any], config_path: str | Path) -> None:
    config, root = load_config(config_path)
    expected_coverage = {
        "interval": {"start": START, "stop": STOP},
        "disjoint_from_prior_execution_stop": True,
        "unique_formula_count": UNIQUE_FORMULAS,
        "term_count": TERM_COUNT,
        "six_term_tier_start_ordinal": SIX_TERM_TIER_START,
        "grid_point_count": GRID_POINTS,
        "candidate_grid_evaluations": CANDIDATE_GRID_EVALUATIONS,
        "signed_term_hessian_accumulations": SIGNED_TERM_HESSIAN_ACCUMULATIONS,
        "first_candidate_id": "STC2-f9eacf5b6cfe44a67d38f940",
        "last_candidate_id": "STC2-eec3c7ee83d030875ab452ce",
    }
    expected_contract = {
        "worker_stages": EXPECTED_STAGES,
        "maximum_stage_seconds": 120,
        "maximum_total_seconds": 240,
        "minimum_available_ram_mib": 32768,
        "cpu_backoff_above_percent": 92,
        "cpu_target_percent": 80,
        "gpu_workers": 0,
        "allowlisted_evaluator": EVALUATOR,
        "sample_interval_seconds": 0.1,
        "shutdown_reserve_seconds": 1.0,
        "fixed_shard_size": FIXED_SHARD_SIZE,
        "fixed_shard_count": FIXED_SHARD_COUNT,
    }
    if (
        set(result) != EXPECTED_TOP_LEVEL_KEYS
        or result.get("schema_version") != ARTIFACT_SCHEMA
        or result.get("campaign_id") != config["campaign_id"]
        or result.get("content_sha256") != _content_sha(result)
        or result.get("seals") != EXPECTED_SEALS
        or result.get("coverage") != expected_coverage
        or result.get("contract") != expected_contract
        or result.get("interpretation") != EXPECTED_INTERPRETATION
    ):
        raise ValueError("CPU formula overlap artifact validation failed")
    if set(result.get("source_bindings", {})) != {
        *config["bindings"],
        "config",
        "source",
        "test",
    }:
        raise ValueError("CPU formula overlap source binding set changed")
    for name, binding in config["bindings"].items():
        if result.get("source_bindings", {}).get(name) != binding:
            raise ValueError(f"CPU formula overlap {name} binding changed")
    for name in ("config", "source", "test"):
        binding = result["source_bindings"][name]
        expected_path = {
            "config": "configs/cpu_symbolic_overlap_benchmark.json",
            "source": "src/sigma_theory_compiler/cpu_symbolic_overlap_benchmark.py",
            "test": "tests/test_cpu_symbolic_overlap_benchmark.py",
        }[name]
        if binding.get("path") != expected_path or set(binding) != {
            "path",
            "file_sha256",
        }:
            raise ValueError(f"CPU formula overlap {name} binding shape changed")
        if _file_sha(root / binding["path"]) != binding["file_sha256"]:
            raise ValueError(f"CPU formula overlap {name} hash mismatch")
    if any(
        result.get(key) is not False
        for key in (
            "sqlite_accessed",
            "gpu_or_cuda_accessed",
            "existing_process_signaled",
            "observations_opened",
            "scientific_pass",
            "resource_policy_promoted",
        )
    ):
        raise ValueError("CPU formula overlap forbidden claim changed")
    stages = result.get("stages", [])
    if not isinstance(stages, list) or len(stages) not in {1, 2}:
        raise ValueError("CPU formula overlap stage count changed")
    if stages[0].get("workers") != 15:
        raise ValueError("CPU formula overlap mandatory stage missing")
    if [stage.get("workers") for stage in stages] != EXPECTED_STAGES[: len(stages)]:
        raise ValueError("CPU formula overlap conditional stage mismatch")
    if (len(stages) == 2) != (result.get("stage_16_admitted") is True):
        raise ValueError("CPU formula overlap stage admission relationship changed")
    if len(stages) == 2 and result.get("stage_16_blocker") is not None:
        raise ValueError("CPU formula overlap executed overlap has a blocker")
    if len(stages) == 1 and result.get("stage_16_blocker") not in {
        "stage_15_control_or_resource_guard",
        "ram_floor",
        "cpu_backoff",
    }:
        raise ValueError("CPU formula overlap missing fail-closed blocker")
    expected_replay = UNIQUE_FORMULAS if len(stages) == 2 else 0
    if (
        result.get("total_formula_evaluator_executions") != UNIQUE_FORMULAS * len(stages)
        or result.get("overlap_control_replay_formula_evaluations") != expected_replay
    ):
        raise ValueError("CPU formula overlap replay accounting changed")
    zero_status_root = hashlib.sha256(bytes(FIXED_SHARD_SIZE)).hexdigest()
    expected_result_root = _sha(
        [
            {
                "shard_index": index,
                "start": START + index * FIXED_SHARD_SIZE,
                "stop": START + (index + 1) * FIXED_SHARD_SIZE,
                "status_root_sha256": zero_status_root,
            }
            for index in range(FIXED_SHARD_COUNT)
        ]
    )
    expected_manifest_root = _sha(_fixed_shards())
    for stage in stages:
        counts = stage.get("counts", {})
        elapsed = float(stage.get("elapsed_seconds", math.nan))
        rate = float(stage.get("unique_formulas_per_second", math.nan))
        grid_rate = float(stage.get("candidate_grid_evaluations_per_second", math.nan))
        cpu_mean = float(stage.get("cpu_percent_mean", math.nan))
        cpu_median = float(stage.get("cpu_percent_median", math.nan))
        cpu_peak = float(stage.get("cpu_percent_peak", math.nan))
        worker_cpu = float(stage.get("worker_cpu_seconds", math.nan))
        worker_capacity = float(stage.get("worker_cpu_capacity_percent", math.nan))
        measured_receipt = MEASURED_STAGE_RECEIPTS.get(stage.get("workers"))
        if (
            set(stage) != EXPECTED_STAGE_KEYS
            or measured_receipt is None
            or any(stage.get(key) != value for key, value in measured_receipt.items())
            or stage.get("backend") != "cpu_numpy"
            or stage.get("gpu_workers") != 0
            or stage.get("batch_count") != FIXED_SHARD_COUNT
            or stage.get("fixed_shard_count") != FIXED_SHARD_COUNT
            or stage.get("fixed_shard_size") != FIXED_SHARD_SIZE
            or stage.get("interval") != {"start": START, "stop": STOP}
            or stage.get("unique_formula_count") != UNIQUE_FORMULAS
            or stage.get("candidate_grid_evaluations") != CANDIDATE_GRID_EVALUATIONS
            or stage.get("signed_term_hessian_accumulations") != SIGNED_TERM_HESSIAN_ACCUMULATIONS
            or set(counts) != {"reject", "pass", "ambiguous"}
            or any(type(value) is not int or value < 0 for value in counts.values())
            or counts != {"reject": UNIQUE_FORMULAS, "pass": 0, "ambiguous": 0}
            or stage.get("exact_gap_overlap_duplicate_free_coverage") is not True
            or stage.get("ambiguous_are_not_passes") is not True
            or stage.get("all_evaluator_controls_passed") is not True
            or stage.get("partition_independent_status_root_sha256") != expected_result_root
            or stage.get("fixed_shard_manifest_root_sha256") != expected_manifest_root
            or not math.isfinite(float(stage.get("reported_margin_minimum", math.nan)))
            or not math.isfinite(elapsed)
            or not 0 < elapsed <= 120
            or not math.isfinite(rate)
            or not math.isclose(rate, UNIQUE_FORMULAS / elapsed, rel_tol=1e-12)
            or not math.isfinite(grid_rate)
            or not math.isclose(grid_rate, CANDIDATE_GRID_EVALUATIONS / elapsed, rel_tol=1e-12)
            or stage.get("cpu_sampling_contract") != "fixed_interval_blocking_device_wide_psutil"
            or stage.get("cpu_sample_interval_seconds") != 0.1
            or type(stage.get("cpu_sample_count")) is not int
            or stage.get("cpu_sample_count", 0) < 1
            or any(
                not math.isfinite(value) or not 0 <= value <= 100
                for value in (cpu_mean, cpu_median, cpu_peak)
            )
            or (cpu_median >= 80) != stage.get("cpu_target_met_by_median")
            or stage.get("cpu_target_percent") != 80
            or (cpu_peak > 92) != stage.get("backoff_threshold_exceeded")
            or stage.get("minimum_available_ram_mib", 0) < 32768
            or not math.isfinite(worker_cpu)
            or worker_cpu <= 0
            or not math.isfinite(worker_capacity)
            or not math.isclose(
                worker_capacity,
                100 * worker_cpu / (elapsed * 24),
                rel_tol=1e-12,
            )
            or stage.get("hard_deadline_enforced") is not True
            or stage.get("hard_deadline_triggered") is not False
            or stage.get("owned_worker_termination_count") != 0
            or stage.get("wall_bound_exceeded") is not False
            or stage.get("content_sha256") != _content_sha(stage)
        ):
            raise ValueError("CPU formula overlap stage validation failed")
    cross_equal = (
        all(
            stages[0][key] == stages[1][key]
            for key in (
                "counts",
                "partition_independent_status_root_sha256",
                "reported_margin_minimum",
                "fixed_shard_manifest_root_sha256",
            )
        )
        if len(stages) == 2
        else None
    )
    target_met = any(stage["cpu_target_met_by_median"] for stage in stages)
    backoff = any(stage["backoff_threshold_exceeded"] for stage in stages)
    expected_decision = (
        "real_formula_cpu_overlap_completed_target_met_no_policy_promotion"
        if target_met
        else "real_formula_cpu_overlap_completed_target_not_met_no_policy_promotion"
    )
    total_elapsed = float(result.get("total_elapsed_seconds", math.nan))
    hardware = result.get("hardware_attestation", {})
    if (
        result.get("overlap_control_executed") is not (len(stages) == 2)
        or result.get("cross_stage_replay_equal") is not cross_equal
        or cross_equal is False
        or result.get("evaluator_controls_passed") is not True
        or result.get("cpu_target_met") is not target_met
        or result.get("resource_backoff_triggered") is not backoff
        or result.get("decision") != expected_decision
        or result.get("hard_total_deadline_enforced") is not True
        or result.get("total_bound_respected") is not True
        or not math.isfinite(total_elapsed)
        or not sum(stage["elapsed_seconds"] for stage in stages) <= total_elapsed <= 240
        or set(hardware) != EXPECTED_HARDWARE_KEYS
        or hardware.get("physical_cores") != 24
        or hardware.get("logical_processors") != 24
        or hardware.get("installed_ram_mib", 0) < 32768
        or hardware.get("resource_profile_file_sha256")
        != config["bindings"]["resource_profile"]["file_sha256"]
        or hardware.get("profile_sustained_cpu_workers") != 16
        or hardware.get("profile_reserved_cpu_cores") != 8
        or hardware.get("profile_matches_host_topology") is not True
    ):
        raise ValueError("CPU formula overlap decision or resource relationship changed")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _atomic_write(path: Path, result: Mapping[str, Any], maximum_bytes: int) -> None:
    payload = (_canonical(result) + "\n").encode()
    if len(payload) > maximum_bytes:
        raise RuntimeError("CPU formula overlap artifact exceeds byte bound")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.parent.is_symlink():
        raise RuntimeError("CPU formula overlap artifact symlink rejected")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--validate")
    args = parser.parse_args()
    if sum((args.run, bool(args.validate))) != 1:
        raise ValueError("select exactly one CPU formula overlap operation")
    if args.validate:
        validate_artifact(json.loads(Path(args.validate).read_text()), args.config)
        return 0
    result = execute_campaign(args.config)
    config, root = load_config(args.config)
    _atomic_write(_inside(root, config["output_path"]), result, int(config["maximum_output_bytes"]))
    print(_canonical({"decision": result["decision"], "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
