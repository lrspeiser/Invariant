"""Target-free identifiability checks for all registered temporal architectures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/open_gravity_temporal_transfer_identifiability_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_temporal_transfer_identifiability_v1.py")
TEST_PATH = Path("tests/test_open_gravity_temporal_transfer_identifiability_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-temporal-transfer-identifiability-v1/receipt.json")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-temporal-transfer-identifiability-v1/receipt.json"
)
_SCHEMA = "invariant-open-gravity-temporal-transfer-identifiability-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-temporal-transfer-identifiability-receipt-1.0"
_CONFIG_CONTENT_SHA256 = "3dc8edd24c52bf049b9788b453922cdfdae51fb0314023ea3023f232903ffa47"
_ARCHITECTURES = ("A15_RETARDED", "A16_MEMORY", "A17_RESONANCE", "A18_STOCHASTIC")


class TemporalIdentifiabilityError(RuntimeError):
    """Raised when a temporal signature or frozen contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TemporalIdentifiabilityError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TemporalIdentifiabilityError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    required = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "architectures",
        "driver_ids",
        "numerical_contract",
        "history_fixtures",
        "required_checks",
        "decision_rules",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(set(config) == required, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-temporal-transfer-identifiability-v1",
        "ID changed",
    )
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(
        tuple(row["id"] for row in config["architectures"]) == _ARCHITECTURES,
        "architectures changed",
    )
    _require(config["driver_ids"] == [f"D{i:02d}" for i in range(1, 21)], "drivers changed")
    _require(len(config["required_checks"]) == 10, "checks changed")
    _require(set(config["access_contract"].values()) == {0}, "access changed")
    frequencies = config["numerical_contract"]["frequency_grid"]
    _require(frequencies == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0], "frequencies changed")


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "temporal config")
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _validate_bindings(config: Mapping[str, Any]) -> dict[str, str]:
    observed: dict[str, str] = {}
    for row in config["bindings"]:
        path = Path(row["path"])
        _require(path.is_file(), f"missing binding: {row['role']}")
        digest = file_sha256(path)
        _require(digest == row["sha256"], f"binding changed: {row['role']}")
        observed[row["role"]] = digest
    _require(len(observed) == 5, "binding count changed")
    return observed


def transfer_retarded(omega: np.ndarray, *, delay: float) -> np.ndarray:
    _require(delay > 0.0, "delay must be positive")
    return np.exp(-1j * omega * delay)


def transfer_memory(omega: np.ndarray, *, tau: float) -> np.ndarray:
    _require(tau > 0.0, "memory time must be positive")
    return 1.0 / (1.0 + 1j * omega * tau)


def transfer_resonance(omega: np.ndarray, *, omega0: float, zeta: float) -> np.ndarray:
    _require(omega0 > 0.0 and 0.0 < zeta < 1.0, "resonance parameters invalid")
    return omega0**2 / (omega0**2 - omega**2 + 2j * zeta * omega0 * omega)


def transfer_telegraph(
    omega: np.ndarray, wave_number: np.ndarray, *, length: float, tau: float
) -> np.ndarray:
    _require(length > 0.0 and tau > 0.0, "telegraph scales invalid")
    return 1.0 / (1.0 + (length * wave_number) ** 2 - (tau * omega) ** 2 - 1j * tau * omega)


def _step_time(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    numerical = config["numerical_contract"]
    dt = float(numerical["step_dt"])
    duration = float(numerical["step_duration"])
    transition = float(numerical["step_time"])
    times = np.arange(0.0, duration + 0.5 * dt, dt)
    source = (times >= transition).astype(float)
    return times, source


def step_retarded(times: np.ndarray, source: np.ndarray, *, delay: float) -> np.ndarray:
    dt = float(times[1] - times[0])
    shift = round(delay / dt)
    _require(abs(shift * dt - delay) <= 1.0e-12, "delay not aligned to grid")
    result = np.zeros_like(source)
    result[shift:] = source[:-shift]
    return result


def step_memory(times: np.ndarray, source: np.ndarray, *, tau: float) -> np.ndarray:
    dt = float(times[1] - times[0])
    decay = math.exp(-dt / tau)
    result = np.zeros_like(source)
    for index in range(1, result.size):
        result[index] = source[index] + (result[index - 1] - source[index]) * decay
    return result


def step_resonance(
    times: np.ndarray, source: np.ndarray, *, omega0: float, zeta: float
) -> np.ndarray:
    dt = float(times[1] - times[0])
    q = 0.0
    velocity = 0.0
    result = np.zeros_like(source)

    def derivative(state_q: float, state_v: float, forcing: float) -> tuple[float, float]:
        return state_v, omega0**2 * (forcing - state_q) - 2.0 * zeta * omega0 * state_v

    for index in range(1, result.size):
        forcing = float(source[index])
        k1q, k1v = derivative(q, velocity, forcing)
        k2q, k2v = derivative(q + 0.5 * dt * k1q, velocity + 0.5 * dt * k1v, forcing)
        k3q, k3v = derivative(q + 0.5 * dt * k2q, velocity + 0.5 * dt * k2v, forcing)
        k4q, k4v = derivative(q + dt * k3q, velocity + dt * k3v, forcing)
        q += dt * (k1q + 2.0 * k2q + 2.0 * k3q + k4q) / 6.0
        velocity += dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        result[index] = q
    return result


def _complex_rows(values: np.ndarray, frequencies: np.ndarray) -> list[dict[str, float]]:
    return [
        {
            "omega": float(omega),
            "magnitude": float(abs(value)),
            "phase": float(np.angle(value)),
        }
        for omega, value in zip(frequencies, values, strict=True)
    ]


def _dynamic_fingerprints(config: Mapping[str, Any]) -> dict[str, Any]:
    frequencies = np.asarray(config["numerical_contract"]["frequency_grid"], dtype=float)
    retarded = transfer_retarded(frequencies, delay=0.5)
    memory = transfer_memory(frequencies, tau=0.5)
    resonance = transfer_resonance(frequencies, omega0=2.0, zeta=0.25)
    stochastic_mean = transfer_memory(frequencies, tau=0.5)
    stationary_variance = 0.05**2 * 0.5 / 2.0
    rows = {
        "A15_RETARDED": _complex_rows(retarded, frequencies),
        "A16_MEMORY": _complex_rows(memory, frequencies),
        "A17_RESONANCE": _complex_rows(resonance, frequencies),
        "A18_STOCHASTIC_MEAN": _complex_rows(stochastic_mean, frequencies),
    }
    vectors = {
        "A15_RETARDED": np.concatenate((np.abs(retarded), np.angle(retarded), [0.0])),
        "A16_MEMORY": np.concatenate((np.abs(memory), np.angle(memory), [0.0])),
        "A17_RESONANCE": np.concatenate((np.abs(resonance), np.angle(resonance), [0.0])),
        "A18_STOCHASTIC": np.concatenate(
            (np.abs(stochastic_mean), np.angle(stochastic_mean), [stationary_variance])
        ),
    }
    distances: dict[str, float] = {}
    names = tuple(vectors)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            distances[f"{left}__{right}"] = float(np.linalg.norm(vectors[left] - vectors[right]))
    _require(all(value > 0.0 for value in distances.values()), "dynamic fingerprints collide")
    _require(np.array_equal(memory, stochastic_mean), "stochastic mean must equal memory")
    return {
        "transfer_rows": rows,
        "pairwise_fingerprint_distances": distances,
        "stochastic_stationary_variance": stationary_variance,
        "memory_and_stochastic_mean_exactly_equal": True,
    }


def _step_signatures(config: Mapping[str, Any]) -> dict[str, Any]:
    times, source = _step_time(config)
    transition = float(config["numerical_contract"]["step_time"])
    retarded = step_retarded(times, source, delay=0.5)
    memory = step_memory(times, source, tau=0.5)
    resonance = step_resonance(times, source, omega0=2.0, zeta=0.25)
    before_source = times < transition
    before_retarded_arrival = times < transition + 0.5
    _require(np.max(np.abs(retarded[before_retarded_arrival])) == 0.0, "retarded output advanced")
    _require(np.max(np.abs(memory[before_source])) == 0.0, "memory output advanced")
    _require(np.max(np.abs(resonance[before_source])) == 0.0, "resonance output advanced")
    post = memory[times >= transition]
    _require(np.all(np.diff(post) >= -1.0e-14), "memory step is not monotone")
    _require(float(np.max(resonance)) > 1.1, "resonance has no overshoot")
    return {
        "retarded_first_positive_time": float(times[np.flatnonzero(retarded > 0.0)[0]]),
        "memory_final": float(memory[-1]),
        "memory_monotone": True,
        "resonance_peak": float(np.max(resonance)),
        "resonance_final": float(resonance[-1]),
        "no_advanced_response": True,
        "step_signature_root_sha256": content_sha256(
            {
                "retarded": retarded.tolist(),
                "memory": memory.tolist(),
                "resonance": resonance.tolist(),
            }
        ),
    }


def _telegraph_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    numerical = config["numerical_contract"]
    frequencies = np.asarray(numerical["frequency_grid"], dtype=float)
    wave_numbers = np.asarray(numerical["telegraph_k_grid"], dtype=float)
    length = float(numerical["telegraph_L"])
    tau = float(numerical["telegraph_tau"])
    mesh_omega, mesh_k = np.meshgrid(frequencies, wave_numbers, indexing="ij")
    response = transfer_telegraph(mesh_omega, mesh_k, length=length, tau=tau)
    speed = length / tau
    _require(speed <= 1.0, "telegraph speed exceeds normalized c")
    static = transfer_telegraph(np.zeros_like(wave_numbers), wave_numbers, length=length, tau=tau)
    expected_static = 1.0 / (1.0 + (length * wave_numbers) ** 2)
    _require(
        np.allclose(static.real, expected_static, rtol=0.0, atol=1.0e-15), "static kernel changed"
    )
    return {
        "necessary_speed_over_c": speed,
        "static_spatial_attenuation": expected_static.tolist(),
        "space_time_transfer_sha256": hashlib.sha256(response.tobytes()).hexdigest(),
        "instantaneous_baryonic_source_still_unresolved": True,
    }


def _concept_coverage(config: Mapping[str, Any]) -> dict[str, Any]:
    concepts = [
        {
            "concept_id": f"TW2-{architecture.split('_', 1)[0]}-{driver}",
            "architecture": architecture,
            "driver": driver,
            "theory_status": "PASS_TARGET_FREE_TEMPORAL_SIGNATURE",
            "empirical_status": "UNTESTED_SOURCE_HISTORY_REQUIRED",
        }
        for architecture in _ARCHITECTURES
        for driver in config["driver_ids"]
    ]
    _require(
        len(concepts) == 80 and len({row["concept_id"] for row in concepts}) == 80,
        "coverage changed",
    )
    return {
        "concepts": 80,
        "by_architecture": {architecture: 20 for architecture in _ARCHITECTURES},
        "concept_root_sha256": content_sha256(concepts),
        "rows": concepts,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    fingerprints = _dynamic_fingerprints(config)
    steps = _step_signatures(config)
    telegraph = _telegraph_signature(config)
    coverage = _concept_coverage(config)
    checks = {
        "EXACT_4_BY_20_CONCEPT_COVERAGE": coverage["concepts"] == 80,
        "STATIC_ZERO_FREQUENCY_DEGENERACY": all(
            row[0]["magnitude"] == 1.0 and row[0]["phase"] == 0.0
            for row in fingerprints["transfer_rows"].values()
        ),
        "RETARDED_UNIT_MAGNITUDE_AND_DEAD_TIME": all(
            abs(row["magnitude"] - 1.0) <= 1.0e-15
            for row in fingerprints["transfer_rows"]["A15_RETARDED"]
        )
        and steps["retarded_first_positive_time"] >= 1.5,
        "MEMORY_ATTENUATION_PHASE_AND_MONOTONE_STEP": steps["memory_monotone"]
        and fingerprints["transfer_rows"]["A16_MEMORY"][-1]["magnitude"] < 1.0,
        "RESONANCE_PEAK_AND_OVERSHOOT": steps["resonance_peak"] > 1.1,
        "STOCHASTIC_MEAN_DEGENERACY_AND_POSITIVE_VARIANCE": fingerprints[
            "memory_and_stochastic_mean_exactly_equal"
        ]
        and fingerprints["stochastic_stationary_variance"] > 0.0,
        "PAIRWISE_DYNAMIC_FINGERPRINTS_DISTINCT_WITH_VARIANCE_CHANNEL": all(
            value > 0.0 for value in fingerprints["pairwise_fingerprint_distances"].values()
        ),
        "TELEGRAPH_SPACE_TIME_TRANSFER_AND_SPEED_BOUND": telegraph["necessary_speed_over_c"] <= 1.0,
        "NO_ADVANCED_RESPONSE": steps["no_advanced_response"],
        "ZERO_SCIENTIFIC_RESPONSE_ACCESS": set(config["access_contract"].values()) == {0},
    }
    _require(all(checks.values()), "required temporal check failed")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_TARGET_FREE_TEMPORAL_IDENTIFIABILITY_SOURCE_HISTORIES_STILL_REQUIRED",
        "bindings": bindings,
        "coverage": coverage,
        "fingerprints": fingerprints,
        "step_signatures": steps,
        "telegraph": telegraph,
        "decision_rules": config["decision_rules"],
        "checks": checks,
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
        "artifact_bindings": {
            "config_sha256": file_sha256(CONFIG_PATH),
            "module_sha256": file_sha256(MODULE_PATH),
            "test_sha256": file_sha256(TEST_PATH),
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(dict(payload) == build_receipt(), "temporal receipt differs from rebuild")


def write_receipt() -> str:
    payload = build_receipt()
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        _require(OUTPUT_PATH.read_bytes() == encoded, "existing temporal receipt differs")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, OUTPUT_PATH)
    except FileExistsError:
        _require(OUTPUT_PATH.read_bytes() == encoded, "concurrent temporal receipt differs")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def validate_receipt() -> None:
    _require(OUTPUT_PATH.is_file(), "temporal receipt absent")
    payload = _read_json(OUTPUT_PATH, "temporal receipt")
    _require(type(payload) is dict, "receipt is not an object")
    validate_receipt_payload(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        validate_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(
            json.dumps(
                {
                    "status": receipt["status"],
                    "concepts": receipt["coverage"]["concepts"],
                    "checks_passed": sum(receipt["checks"].values()),
                    "raw_scientific_rows": receipt["access_accounting"]["raw_scientific_rows"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
