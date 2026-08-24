"""Exhaustive GPU modular prefilter for compiled creative candidate lattices.

The GPU is only a rejection accelerator.  Every surviving coefficient vector is replayed with
ordinary Python integers, and a modular survivor is never promoted as a proof or formula result.
The input is representation-neutral: a typed compiler supplies integer feature rows and targets,
while this module exhaustively enumerates the declared bounded coefficient lattice.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from .sigma_core import canonical_sha256

CONFIG_SCHEMA = "invariant-creative-modular-gpu-config-1.0"
RESULT_SCHEMA = "invariant-creative-modular-gpu-receipt-1.0"
CONFIG_PATH = "configs/creative_modular_gpu_prefilter.json"
SOURCE_PATH = "src/sigma_theory_compiler/creative_modular_gpu_prefilter.py"
OUTPUT_PATH = "runs/math/creative-modular-gpu-prefilter/receipt.json"
BACKEND = "cupy-raw-kernel-modular-prefilter"
STATUS = "PASS_GPU_MODULAR_PREFILTER_CONTROL"


class CreativeModularGPUError(ValueError):
    """The modular prefilter configuration or receipt failed closed."""


KERNEL = r"""
extern "C" __device__ unsigned long long signed_mod(
    const long long value,
    const unsigned int prime) {
  long long residue = value % (long long)prime;
  return (unsigned long long)(residue < 0 ? residue + (long long)prime : residue);
}

extern "C" __global__ void modular_prefilter(
    const unsigned long long start_ordinal,
    const unsigned long long candidate_count,
    const long long coefficient_minimum,
    const unsigned int radix,
    const int feature_count,
    const int observation_count,
    const int prime_count,
    const long long* feature_matrix,
    const long long* targets,
    const unsigned int* primes,
    unsigned char* statuses) {
  const unsigned long long local =
      (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
  if (local >= candidate_count) return;

  unsigned long long cursor = start_ordinal + local;
  long long coefficients[16];
  for (int feature = 0; feature < feature_count; ++feature) {
    coefficients[feature] = coefficient_minimum + (long long)(cursor % radix);
    cursor /= radix;
  }

  for (int prime_index = 0; prime_index < prime_count; ++prime_index) {
    const unsigned int prime = primes[prime_index];
    for (int observation = 0; observation < observation_count; ++observation) {
      unsigned long long accumulator = 0;
      for (int feature = 0; feature < feature_count; ++feature) {
        const long long value =
            feature_matrix[observation * feature_count + feature];
        const unsigned long long coefficient_residue =
            signed_mod(coefficients[feature], prime);
        const unsigned long long value_residue = signed_mod(value, prime);
        const unsigned long long term =
            (coefficient_residue * value_residue) % prime;
        accumulator = (accumulator + term) % prime;
      }
      if (accumulator != signed_mod(targets[observation], prime)) {
        statuses[local] = 0;
        return;
      }
    }
  }
  statuses[local] = 1;
}
"""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CreativeModularGPUError(f"{label} keys changed")


def _normalized_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _is_prime(value: int) -> bool:
    if value < 2 or value % 2 == 0:
        return value == 2
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    _strict(
        value,
        {
            "batch_size",
            "campaign_id",
            "coefficient_max",
            "coefficient_min",
            "cpu_benchmark_candidates",
            "feature_matrix",
            "maximum_recorded_survivors",
            "primes",
            "required_device_name",
            "sample_candidates",
            "schema_version",
            "target_coefficients",
            "targets",
        },
        "modular GPU config",
    )
    if value.get("schema_version") != CONFIG_SCHEMA:
        raise CreativeModularGPUError("modular GPU config schema changed")
    integer_fields = (
        "batch_size",
        "coefficient_max",
        "coefficient_min",
        "cpu_benchmark_candidates",
        "maximum_recorded_survivors",
        "sample_candidates",
    )
    if any(type(value.get(key)) is not int for key in integer_fields):
        raise CreativeModularGPUError("modular GPU integer policy changed")
    minimum = value["coefficient_min"]
    maximum = value["coefficient_max"]
    if minimum >= maximum or minimum < -32768 or maximum > 32767:
        raise CreativeModularGPUError("coefficient lattice is invalid")
    if not 1 <= value["batch_size"] <= 2**24:
        raise CreativeModularGPUError("GPU batch size is invalid")
    if not 1 <= value["sample_candidates"] <= 16384:
        raise CreativeModularGPUError("sample candidate count is invalid")
    if not 1 <= value["cpu_benchmark_candidates"] <= 2**20:
        raise CreativeModularGPUError("CPU benchmark count is invalid")
    if not 1 <= value["maximum_recorded_survivors"] <= 4096:
        raise CreativeModularGPUError("survivor recording bound is invalid")

    matrix = value.get("feature_matrix")
    targets = value.get("targets")
    coefficients = value.get("target_coefficients")
    primes = value.get("primes")
    if (
        not isinstance(matrix, list)
        or len(matrix) < 2
        or not isinstance(coefficients, list)
        or not 1 <= len(coefficients) <= 16
        or not isinstance(targets, list)
        or len(targets) != len(matrix)
        or not isinstance(primes, list)
        or not 2 <= len(primes) <= 8
    ):
        raise CreativeModularGPUError("modular GPU matrix shape changed")
    feature_count = len(coefficients)
    if any(
        not isinstance(row, list)
        or len(row) != feature_count
        or any(type(item) is not int or abs(item) > 2**31 for item in row)
        for row in matrix
    ):
        raise CreativeModularGPUError("feature matrix is not bounded integer data")
    if any(type(item) is not int or abs(item) > 2**47 for item in targets):
        raise CreativeModularGPUError("targets are not bounded integer data")
    if any(
        type(item) is not int or not minimum <= item <= maximum for item in coefficients
    ):
        raise CreativeModularGPUError("target coefficients leave the search lattice")
    if any(
        sum(coefficient * feature for coefficient, feature in zip(coefficients, row))
        != target
        for row, target in zip(matrix, targets)
    ):
        raise CreativeModularGPUError("declared target coefficients do not generate targets")
    if (
        any(type(prime) is not int or prime >= 2**31 or not _is_prime(prime) for prime in primes)
        or len(set(primes)) != len(primes)
    ):
        raise CreativeModularGPUError("modular screen primes are invalid")
    if (
        not isinstance(value.get("campaign_id"), str)
        or not value["campaign_id"].startswith("invariant.creativity.")
        or not isinstance(value.get("required_device_name"), str)
        or not value["required_device_name"]
    ):
        raise CreativeModularGPUError("campaign or device identity is invalid")
    return json.loads(json.dumps(value))


def load_config(root: Path) -> dict[str, Any]:
    return validate_config(json.loads((root.resolve() / CONFIG_PATH).read_text(encoding="utf-8")))


def search_space_size(config: Mapping[str, Any]) -> int:
    return (config["coefficient_max"] - config["coefficient_min"] + 1) ** len(
        config["target_coefficients"]
    )


def decode_ordinal(ordinal: int, config: Mapping[str, Any]) -> tuple[int, ...]:
    total = search_space_size(config)
    if type(ordinal) is not int or not 0 <= ordinal < total:
        raise CreativeModularGPUError("candidate ordinal leaves the declared lattice")
    radix = config["coefficient_max"] - config["coefficient_min"] + 1
    cursor = ordinal
    result = []
    for _ in config["target_coefficients"]:
        result.append(config["coefficient_min"] + cursor % radix)
        cursor //= radix
    return tuple(result)


def encode_coefficients(coefficients: Sequence[int], config: Mapping[str, Any]) -> int:
    if len(coefficients) != len(config["target_coefficients"]):
        raise CreativeModularGPUError("coefficient vector width changed")
    radix = config["coefficient_max"] - config["coefficient_min"] + 1
    ordinal = 0
    scale = 1
    for coefficient in coefficients:
        if type(coefficient) is not int or not config["coefficient_min"] <= coefficient <= config[
            "coefficient_max"
        ]:
            raise CreativeModularGPUError("coefficient leaves the declared lattice")
        ordinal += (coefficient - config["coefficient_min"]) * scale
        scale *= radix
    return ordinal


def cpu_modular_survives(
    ordinal: int, config: Mapping[str, Any], *, targets: Sequence[int] | None = None
) -> bool:
    coefficients = decode_ordinal(ordinal, config)
    expected = config["targets"] if targets is None else targets
    for prime in config["primes"]:
        for row, target in zip(config["feature_matrix"], expected):
            if sum(coefficient * value for coefficient, value in zip(coefficients, row)) % prime != target % prime:
                return False
    return True


def cpu_exact_match(
    ordinal: int, config: Mapping[str, Any], *, targets: Sequence[int] | None = None
) -> bool:
    coefficients = decode_ordinal(ordinal, config)
    expected = config["targets"] if targets is None else targets
    return all(
        sum(coefficient * value for coefficient, value in zip(coefficients, row)) == target
        for row, target in zip(config["feature_matrix"], expected)
    )


def _sample_ordinals(total: int, target: int, requested: int) -> list[int]:
    wanted = min(total, requested)
    positions = {0, total - 1, target}
    cursor = hashlib.sha256(f"{total}:{target}:{requested}".encode()).digest()
    sequence = 0
    while len(positions) < wanted:
        cursor = hashlib.sha256(cursor + sequence.to_bytes(8, "little")).digest()
        positions.add(int.from_bytes(cursor[:8], "little") % total)
        sequence += 1
    return sorted(positions)[:wanted]


def _status_root(records: Sequence[tuple[int, int]]) -> str:
    return canonical_sha256([[ordinal, status] for ordinal, status in records])


def _decimal(value: float) -> str:
    if not math.isfinite(value) or value < 0:
        raise CreativeModularGPUError("non-finite benchmark measurement")
    return format(Decimal(str(value)).quantize(Decimal("0.000001")), "f")


def _cupy_assets(config: Mapping[str, Any]):
    try:
        import cupy as cp
    except ImportError as error:
        raise RuntimeError("CuPy is required for the creative modular GPU prefilter") from error
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device is available for the creative modular GPU prefilter")
    device = cp.cuda.Device()
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode()
    if name != config["required_device_name"]:
        raise RuntimeError(
            f"required CUDA device {config['required_device_name']!r} is unavailable; found {name!r}"
        )
    kernel = cp.RawKernel(KERNEL, "modular_prefilter")
    matrix = cp.asarray(config["feature_matrix"], dtype=cp.int64)
    targets = cp.asarray(config["targets"], dtype=cp.int64)
    primes = cp.asarray(config["primes"], dtype=cp.uint32)
    return cp, device, name, kernel, matrix, targets, primes


def _gpu_batch(
    cp,
    kernel,
    matrix,
    targets,
    primes,
    config: Mapping[str, Any],
    start: int,
    count: int,
):
    statuses = cp.empty(count, dtype=cp.uint8)
    threads = 256
    kernel(
        ((count + threads - 1) // threads,),
        (threads,),
        (
            cp.uint64(start),
            cp.uint64(count),
            cp.int64(config["coefficient_min"]),
            cp.uint32(config["coefficient_max"] - config["coefficient_min"] + 1),
            cp.int32(len(config["target_coefficients"])),
            cp.int32(len(config["feature_matrix"])),
            cp.int32(len(config["primes"])),
            matrix,
            targets,
            primes,
            statuses,
        ),
    )
    return statuses


def run_gpu_screen(config: Mapping[str, Any]) -> dict[str, Any]:
    config = validate_config(config)
    cp, device, device_name, kernel, matrix, targets, primes = _cupy_assets(config)
    total = search_space_size(config)
    target_ordinal = encode_coefficients(config["target_coefficients"], config)
    samples = _sample_ordinals(total, target_ordinal, config["sample_candidates"])
    sample_set = set(samples)
    sampled_gpu: dict[int, int] = {}
    survivors: list[int] = []
    status_digest = hashlib.sha256()

    _gpu_batch(cp, kernel, matrix, targets, primes, config, target_ordinal, 1)
    device.synchronize()
    started = time.perf_counter()
    batch_count = 0
    for start in range(0, total, config["batch_size"]):
        count = min(config["batch_size"], total - start)
        host = cp.asnumpy(
            _gpu_batch(cp, kernel, matrix, targets, primes, config, start, count)
        )
        batch_count += 1
        status_digest.update(host.tobytes())
        for local in host.nonzero()[0].tolist():
            if len(survivors) >= config["maximum_recorded_survivors"]:
                raise CreativeModularGPUError("modular survivor recording bound exceeded")
            survivors.append(start + int(local))
        for ordinal in sample_set:
            if start <= ordinal < start + count:
                sampled_gpu[ordinal] = int(host[ordinal - start])
    device.synchronize()
    elapsed = time.perf_counter() - started

    cpu_sample = [(ordinal, int(cpu_modular_survives(ordinal, config))) for ordinal in samples]
    gpu_sample = [(ordinal, sampled_gpu[ordinal]) for ordinal in samples]
    prefix = min(config["cpu_benchmark_candidates"], total)
    cpu_started = time.perf_counter()
    cpu_prefix = [(ordinal, int(cpu_modular_survives(ordinal, config))) for ordinal in range(prefix)]
    cpu_elapsed = time.perf_counter() - cpu_started
    gpu_started = time.perf_counter()
    gpu_prefix_host = cp.asnumpy(_gpu_batch(cp, kernel, matrix, targets, primes, config, 0, prefix))
    device.synchronize()
    gpu_prefix_elapsed = time.perf_counter() - gpu_started
    gpu_prefix = [(ordinal, int(gpu_prefix_host[ordinal])) for ordinal in range(prefix)]

    mutated_targets = list(config["targets"])
    mutated_targets[0] += 1
    device_mutation = cp.asarray(mutated_targets, dtype=cp.int64)
    mutation_status = int(
        cp.asnumpy(
            _gpu_batch(
                cp,
                kernel,
                matrix,
                device_mutation,
                primes,
                config,
                target_ordinal,
                1,
            )
        )[0]
    )
    properties = cp.cuda.runtime.getDeviceProperties(device.id)
    return {
        "backend": BACKEND,
        "batch_count": batch_count,
        "batch_size": config["batch_size"],
        "candidate_count": total,
        "candidate_status_sha256": status_digest.hexdigest(),
        "candidates_per_second": _decimal(total / elapsed),
        "cpu_benchmark": {
            "candidate_count": prefix,
            "elapsed_seconds": _decimal(cpu_elapsed),
            "status_sha256": _status_root(cpu_prefix),
        },
        "elapsed_seconds": _decimal(elapsed),
        "gpu": {
            "compute_capability": str(device.compute_capability),
            "cupy_version": cp.__version__,
            "device_id": device.id,
            "device_name": device_name,
            "driver_version": cp.cuda.runtime.driverGetVersion(),
            "runtime_version": cp.cuda.runtime.runtimeGetVersion(),
            "total_global_memory_bytes": int(properties["totalGlobalMem"]),
        },
        "gpu_benchmark": {
            "candidate_count": prefix,
            "elapsed_seconds": _decimal(gpu_prefix_elapsed),
            "speedup_over_cpu": _decimal(cpu_elapsed / gpu_prefix_elapsed),
            "status_sha256": _status_root(gpu_prefix),
        },
        "mutation_control": {
            "mutation": "targets[0] += 1",
            "target_candidate_gpu_status": mutation_status,
            "target_candidate_rejected": mutation_status == 0,
        },
        "sample_crosscheck": {
            "cpu_status_sha256": _status_root(cpu_sample),
            "gpu_status_sha256": _status_root(gpu_sample),
            "sample_count": len(samples),
            "statuses_agree": cpu_sample == gpu_sample,
        },
        "survivor_ordinals": survivors,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    result = run_gpu_screen(config)
    exact_survivors = [
        ordinal for ordinal in result["survivor_ordinals"] if cpu_exact_match(ordinal, config)
    ]
    target_ordinal = encode_coefficients(config["target_coefficients"], config)
    source_bindings = {
        "config": {
            "normalized_file_sha256": _normalized_file_sha256(root / CONFIG_PATH),
            "path": CONFIG_PATH,
        },
        "runner": {
            "normalized_file_sha256": _normalized_file_sha256(root / SOURCE_PATH),
            "path": SOURCE_PATH,
        },
    }
    body: dict[str, Any] = {
        "campaign_id": config["campaign_id"],
        "claims": {
            "gpu_survival_establishes_formula_correctness": False,
            "gpu_survival_establishes_literature_novelty": False,
            "modular_screen_is_a_formal_proof": False,
            "open_problem_solved": False,
        },
        "compiled_lattice": {
            "coefficient_max": config["coefficient_max"],
            "coefficient_min": config["coefficient_min"],
            "feature_count": len(config["target_coefficients"]),
            "maximum_congruence_checks": search_space_size(config)
            * len(config["feature_matrix"])
            * len(config["primes"]),
            "observation_count": len(config["feature_matrix"]),
            "ordinal_order": "least-significant feature coefficient first",
            "prime_count": len(config["primes"]),
            "search_space_candidates": search_space_size(config),
        },
        "execution_boundary": {
            "credential_accessed": False,
            "exact_cpu_replay_required": True,
            "paid_llm_calls_made": 0,
            "provider_transport_accessed": False,
        },
        "generated_utc": datetime.now(UTC).isoformat(),
        "gpu_execution": result,
        "independent_exact_replay": {
            "all_modular_survivors_replayed": len(exact_survivors)
            == len(result["survivor_ordinals"]),
            "exact_survivor_coefficients": [
                list(decode_ordinal(ordinal, config)) for ordinal in exact_survivors
            ],
            "exact_survivor_ordinals": exact_survivors,
            "target_candidate_survived": target_ordinal in exact_survivors,
            "target_ordinal": target_ordinal,
        },
        "schema_version": RESULT_SCHEMA,
        "source_bindings": source_bindings,
        "summary": {
            "candidates_classified": result["candidate_count"],
            "exact_survivors": len(exact_survivors),
            "gpu_modular_survivors": len(result["survivor_ordinals"]),
            "rejected_by_gpu": result["candidate_count"] - len(result["survivor_ordinals"]),
            "status": STATUS,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_receipt(body, root)
    return body


def _positive_decimal(value: Any, label: str) -> None:
    try:
        parsed = Decimal(value)
    except Exception as error:
        raise CreativeModularGPUError(f"{label} is not a decimal measurement") from error
    if not parsed.is_finite() or parsed <= 0:
        raise CreativeModularGPUError(f"{label} is not positive")


def validate_receipt(value: Mapping[str, Any], root: Path) -> None:
    root = root.resolve()
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != RESULT_SCHEMA or value.get("content_sha256") != canonical_sha256(body):
        raise CreativeModularGPUError("modular GPU receipt identity or seal changed")
    config = load_config(root)
    total = search_space_size(config)
    target_ordinal = encode_coefficients(config["target_coefficients"], config)
    expected_bindings = {"config": CONFIG_PATH, "runner": SOURCE_PATH}
    bindings = value.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_bindings):
        raise CreativeModularGPUError("modular GPU source bindings changed")
    for key, path in expected_bindings.items():
        binding = bindings[key]
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"normalized_file_sha256", "path"}
            or binding.get("path") != path
            or binding.get("normalized_file_sha256")
            != _normalized_file_sha256(root / path)
        ):
            raise CreativeModularGPUError("modular GPU source binding changed")
    claims = value.get("claims")
    if (
        not isinstance(claims, Mapping)
        or set(claims)
        != {
            "gpu_survival_establishes_formula_correctness",
            "gpu_survival_establishes_literature_novelty",
            "modular_screen_is_a_formal_proof",
            "open_problem_solved",
        }
        or any(item is not False for item in claims.values())
    ):
        raise CreativeModularGPUError("modular GPU claim boundary changed")
    lattice = value.get("compiled_lattice", {})
    expected_lattice = {
        "coefficient_max": config["coefficient_max"],
        "coefficient_min": config["coefficient_min"],
        "feature_count": len(config["target_coefficients"]),
        "maximum_congruence_checks": total
        * len(config["feature_matrix"])
        * len(config["primes"]),
        "observation_count": len(config["feature_matrix"]),
        "ordinal_order": "least-significant feature coefficient first",
        "prime_count": len(config["primes"]),
        "search_space_candidates": total,
    }
    if lattice != expected_lattice:
        raise CreativeModularGPUError("compiled modular lattice changed")
    execution = value.get("execution_boundary")
    if execution != {
        "credential_accessed": False,
        "exact_cpu_replay_required": True,
        "paid_llm_calls_made": 0,
        "provider_transport_accessed": False,
    }:
        raise CreativeModularGPUError("modular GPU execution boundary changed")
    gpu = value.get("gpu_execution", {})
    if (
        gpu.get("backend") != BACKEND
        or gpu.get("candidate_count") != total
        or gpu.get("batch_size") != config["batch_size"]
        or gpu.get("batch_count") != math.ceil(total / config["batch_size"])
        or not isinstance(gpu.get("candidate_status_sha256"), str)
        or len(gpu["candidate_status_sha256"]) != 64
        or gpu.get("gpu", {}).get("device_name") != config["required_device_name"]
        or gpu.get("sample_crosscheck", {}).get("sample_count")
        != min(total, config["sample_candidates"])
        or gpu.get("sample_crosscheck", {}).get("statuses_agree") is not True
        or gpu.get("sample_crosscheck", {}).get("cpu_status_sha256")
        != gpu.get("sample_crosscheck", {}).get("gpu_status_sha256")
        or gpu.get("cpu_benchmark", {}).get("status_sha256")
        != gpu.get("gpu_benchmark", {}).get("status_sha256")
        or gpu.get("mutation_control", {}).get("target_candidate_rejected") is not True
        or gpu.get("mutation_control", {}).get("target_candidate_gpu_status") != 0
    ):
        raise CreativeModularGPUError("modular GPU execution evidence changed")
    for container, key in (
        (gpu, "elapsed_seconds"),
        (gpu, "candidates_per_second"),
        (gpu.get("cpu_benchmark", {}), "elapsed_seconds"),
        (gpu.get("gpu_benchmark", {}), "elapsed_seconds"),
        (gpu.get("gpu_benchmark", {}), "speedup_over_cpu"),
    ):
        _positive_decimal(container.get(key), key)
    samples = _sample_ordinals(total, target_ordinal, config["sample_candidates"])
    expected_sample = [(ordinal, int(cpu_modular_survives(ordinal, config))) for ordinal in samples]
    if gpu["sample_crosscheck"]["cpu_status_sha256"] != _status_root(expected_sample):
        raise CreativeModularGPUError("independent modular sample replay changed")
    prefix = min(config["cpu_benchmark_candidates"], total)
    expected_prefix = [(ordinal, int(cpu_modular_survives(ordinal, config))) for ordinal in range(prefix)]
    if (
        gpu["cpu_benchmark"].get("candidate_count") != prefix
        or gpu["gpu_benchmark"].get("candidate_count") != prefix
        or gpu["cpu_benchmark"]["status_sha256"] != _status_root(expected_prefix)
    ):
        raise CreativeModularGPUError("CPU benchmark replay changed")
    survivors = gpu.get("survivor_ordinals")
    replay = value.get("independent_exact_replay", {})
    if (
        survivors != [target_ordinal]
        or replay
        != {
            "all_modular_survivors_replayed": True,
            "exact_survivor_coefficients": [config["target_coefficients"]],
            "exact_survivor_ordinals": [target_ordinal],
            "target_candidate_survived": True,
            "target_ordinal": target_ordinal,
        }
        or not cpu_exact_match(target_ordinal, config)
    ):
        raise CreativeModularGPUError("exact survivor replay changed")
    summary = value.get("summary")
    if summary != {
        "candidates_classified": total,
        "exact_survivors": 1,
        "gpu_modular_survivors": 1,
        "rejected_by_gpu": total - 1,
        "status": STATUS,
    }:
        raise CreativeModularGPUError("modular GPU summary changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--root", type=Path, default=Path.cwd())
    run.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, default=Path.cwd())
    validate.add_argument("--receipt", type=Path, default=Path(OUTPUT_PATH))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "run":
        receipt = build_receipt(root)
        output = args.output if args.output.is_absolute() else root / args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        receipt_path = args.receipt if args.receipt.is_absolute() else root / args.receipt
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
    print(
        json.dumps(
            {
                "candidates_classified": receipt["summary"]["candidates_classified"],
                "exact_survivors": receipt["summary"]["exact_survivors"],
                "status": receipt["summary"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
