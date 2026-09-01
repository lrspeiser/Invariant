"""Lane-9 v3 repairs and identifiable correlation-only law."""

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
from numpy.typing import NDArray

CONFIG_PATH = Path("configs/open_gravity_void_gravitational_load_v3.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_gravitational_load_v3.py")
TEST_PATH = Path("tests/test_open_gravity_void_gravitational_load_v3.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-gravitational-load-v3/receipt.json")
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
_CONFIG_RAW_SHA256 = "d228ae2012f3fcc75ced2fa78db3de60fd6dd70d8f2abdb01b25d97c082a2588"
_CONFIG_CONTENT_SHA256 = "6539e69870546b3fd5451796ea9fb88ef2ff8aec33684c57c0ffdaa251b0e35d"
_MODULE_SEMANTIC_SHA256 = "e1a88a0e24462f61e2864a59940c4a2d799e637469c2aee6178e3e26168344e3"
_TEST_RAW_SHA256 = "1a34a542633562b4acffab7b2b865154e2b0ab355ae054e11aa033807a9a8274"

Array = NDArray[np.float64]


class VoidLoadV3Error(RuntimeError):
    """Raised when a v3 repair invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidLoadV3Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in ("_CONFIG_RAW_SHA256", "_CONFIG_CONTENT_SHA256", "_MODULE_SEMANTIC_SHA256", "_TEST_RAW_SHA256"):
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        text = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config semantic drift")
    _require(value["status"] == "FROZEN_REPAIRED_CORRELATION_LAWS_ROWS_UNOPENED", "status drift")
    _require(value["access_accounting"]["scientific_rows_decoded"] == 0, "rows opened")
    _require(value["response_contract"]["only_fitted_law_parameter"] == "delta_H", "law widened")
    return value


def _bind_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(row["path"])
    _require(path.is_file() and file_sha256(path) == row["raw_sha256"], "bound receipt raw drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value["content_sha256"] == row["content_sha256"], "bound receipt content drift")
    _require(_self_hash(value) == row["content_sha256"], "bound receipt self-hash invalid")
    return value


def periodic_laplacian(field: Array, dx: float) -> Array:
    _require(field.ndim == 3 and dx > 0.0, "invalid 3D grid")
    result = np.zeros_like(field)
    for axis in range(3):
        result += np.roll(field, 1, axis=axis) + np.roll(field, -1, axis=axis) - 2.0 * field
    return result / dx**2


def local_equilibrium(rho: Array, source: Array, gamma0: float, gamma_b: float) -> Array:
    gamma = gamma0 + gamma_b * rho
    _require(bool(np.all(gamma > 0.0)), "nonpositive relaxation")
    return source / gamma


def solve_parabolic_reservoir(
    rho: Array,
    source: Array,
    *,
    diffusivity: float,
    gamma0: float,
    gamma_b: float,
    dx: float,
    dt: float,
    steps: int,
    initial: Array | None = None,
) -> Array:
    _require(rho.ndim == 3 and source.shape == rho.shape, "reservoir shape mismatch")
    _require(diffusivity >= 0.0 and dt > 0.0 and steps >= 0, "invalid IVP grid")
    gamma = gamma0 + gamma_b * rho
    _require(bool(np.all(gamma > 0.0)), "nonpositive relaxation")
    q0 = np.zeros_like(rho) if initial is None else np.asarray(initial, dtype=float).copy()
    _require(q0.shape == rho.shape, "initial shape mismatch")
    duration = dt * steps
    if diffusivity == 0.0:
        equilibrium = source / gamma
        return equilibrium + (q0 - equilibrium) * np.exp(-gamma * duration)
    q = q0
    def rhs(state: Array) -> Array:
        return diffusivity * periodic_laplacian(state, dx) + source - gamma * state
    for _ in range(steps):
        k1 = rhs(q)
        k2 = rhs(q + 0.5 * dt * k1)
        k3 = rhs(q + 0.5 * dt * k2)
        k4 = rhs(q + dt * k3)
        q += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return q


def _trilinear(field: Array, points: Array) -> Array:
    upper = np.asarray(field.shape, dtype=float) - 1.0
    clipped = np.clip(points, 0.0, upper)
    lo = np.floor(clipped).astype(int)
    hi = np.minimum(lo + 1, np.asarray(field.shape) - 1)
    frac = clipped - lo
    values = np.zeros(points.shape[0], dtype=float)
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                indices = tuple((hi[:, axis] if bit else lo[:, axis]) for axis, bit in enumerate((bx, by, bz)))
                weight = np.prod(np.stack([(frac[:, axis] if bit else 1.0 - frac[:, axis]) for axis, bit in enumerate((bx, by, bz))]), axis=0)
                values += weight * field[indices]
    return values


def self_cell_yukawa_average(length: float, dx: float, order: int) -> float:
    _require(length > 0.0 and dx > 0.0 and order >= 2 and order % 2 == 0, "invalid self quadrature")
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, y, z = np.meshgrid(0.5 * dx * nodes, 0.5 * dx * nodes, 0.5 * dx * nodes, indexing="ij")
    wx, wy, wz = np.meshgrid(weights, weights, weights, indexing="ij")
    radius = np.sqrt(x**2 + y**2 + z**2)
    kernel = np.exp(-radius / length) / (4.0 * math.pi * length**2 * radius)
    return float(np.sum(wx * wy * wz * kernel) / 8.0)


def column_attenuated_feed(
    rho: Array,
    *,
    amplitude: float,
    length: float,
    sigma: float,
    dx: float,
    midpoint_samples: int,
    self_order: int,
) -> Array:
    _require(rho.ndim == 3 and sigma >= 0.0 and midpoint_samples >= 1, "invalid column solve")
    coordinates = np.stack(np.indices(rho.shape), axis=-1).reshape(-1, 3).astype(float)
    masses = rho.reshape(-1) * dx**3
    output = np.zeros(coordinates.shape[0], dtype=float)
    self_kernel = self_cell_yukawa_average(length, dx, self_order)
    fractions = (np.arange(midpoint_samples, dtype=float) + 0.5) / midpoint_samples
    for target_index, target in enumerate(coordinates):
        delta = target[None, :] - coordinates
        distance = np.linalg.norm(delta, axis=1) * dx
        kernel = np.empty_like(distance)
        nonself = distance > 0.0
        kernel[nonself] = np.exp(-distance[nonself] / length) / (4.0 * math.pi * length**2 * distance[nonself])
        kernel[~nonself] = self_kernel
        columns = np.zeros_like(distance)
        if np.any(nonself):
            points = coordinates[nonself, None, :] + fractions[None, :, None] * (target[None, None, :] - coordinates[nonself, None, :])
            sampled = _trilinear(rho, points.reshape(-1, 3)).reshape(-1, midpoint_samples)
            columns[nonself] = np.mean(sampled, axis=1) * distance[nonself]
        output[target_index] = amplitude * float(np.sum(masses * kernel * np.exp(-sigma * columns)))
    return output.reshape(rho.shape)


def union_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged: list[list[float]] = []
    for start, stop in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, stop])
        else:
            merged[-1][1] = max(merged[-1][1], stop)
    return [(row[0], row[1]) for row in merged]


def ray_sphere_intervals(direction: Array, distance: float, spheres: Sequence[tuple[Array, float]]) -> list[tuple[float, float]]:
    unit = np.asarray(direction, dtype=float)
    _require(unit.shape == (3,) and distance >= 0.0, "invalid ray")
    norm = float(np.linalg.norm(unit))
    _require(norm > 0.0, "zero direction")
    unit /= norm
    intervals: list[tuple[float, float]] = []
    for center, radius in spheres:
        center_array = np.asarray(center, dtype=float)
        _require(center_array.shape == (3,) and radius > 0.0, "invalid sphere")
        projection = float(np.dot(unit, center_array))
        transverse2 = float(np.dot(center_array, center_array) - projection**2)
        if transverse2 <= radius**2:
            half = math.sqrt(max(radius**2 - transverse2, 0.0))
            start, stop = max(0.0, projection - half), min(distance, projection + half)
            if stop > start:
                intervals.append((start, stop))
    return union_intervals(intervals)


def _intersection_length(left: Sequence[tuple[float, float]], right: Sequence[tuple[float, float]]) -> float:
    return sum(max(0.0, min(b, d) - max(a, c)) for a, b in union_intervals(left) for c, d in union_intervals(right))


def path_partition(
    direction: Array,
    distance: float,
    spheres: Sequence[tuple[Array, float]],
    observed_intervals: Sequence[tuple[float, float]],
) -> dict[str, float]:
    observed = union_intervals([(max(0.0, a), min(distance, b)) for a, b in observed_intervals])
    void = ray_sphere_intervals(direction, distance, spheres)
    observed_length = sum(b - a for a, b in observed)
    void_observed = _intersection_length(void, observed)
    result = {
        "L_void": void_observed,
        "L_observed_matter": observed_length - void_observed,
        "L_unobserved": distance - observed_length,
        "D": distance,
    }
    _require(all(value >= -1e-12 for value in result.values()), "invalid path partition")
    _require(math.isclose(result["L_void"] + result["L_observed_matter"] + result["L_unobserved"], distance, abs_tol=1e-12), "partition does not close")
    return result


def identifiable_void_prediction(delta_h: float, l_void: float, c: float) -> tuple[float, float]:
    _require(c > 0.0 and l_void >= 0.0, "invalid observable")
    residual_log_redshift = delta_h * l_void / c
    return residual_log_redshift, c * residual_log_redshift


def observed_log_redshift(v3k: float, c: float) -> float:
    _require(c > 0.0 and 1.0 + v3k / c > 0.0, "invalid catalog cz")
    return math.log1p(v3k / c)


def canonical_1pgc_bytes(value: int | str) -> bytes:
    text = str(value)
    _require(text.isascii() and text.isdigit() and int(text) > 0 and str(int(text)) == text, "noncanonical 1PGC")
    return text.encode("ascii")


def split_bucket(value: int | str) -> tuple[int, str]:
    digest = hashlib.sha256(canonical_1pgc_bytes(value)).digest()
    bucket = int.from_bytes(digest[:8], "big", signed=False) % 10
    role = "development" if bucket <= 5 else "validation" if bucket <= 7 else "sealed_confirmation"
    return bucket, role


def synthetic_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    p = config["parameters"]
    grid = np.indices((5, 5, 5), dtype=float)
    rho = 0.1 + np.exp(-((grid[0] - 1.0) ** 2 + (grid[1] - 3.0) ** 2 + (grid[2] - 2.0) ** 2))
    source = p["A_feed"] * rho
    d0 = solve_parabolic_reservoir(rho, source, diffusivity=0.0, gamma0=p["Gamma_0"], gamma_b=p["Gamma_b"], dx=1.0, dt=0.01, steps=100)
    gamma = p["Gamma_0"] + p["Gamma_b"] * rho
    expected_d0 = source / gamma * (1.0 - np.exp(-gamma))
    column = column_attenuated_feed(rho, amplitude=p["A_load"], length=p["L_g"], sigma=p["sigma_g"], dx=1.0, midpoint_samples=p["column_midpoint_samples"], self_order=p["self_cell_gauss_order"])
    spheres = [(np.array([3.0, 0.0, 0.0]), 2.0), (np.array([5.0, 0.0, 0.0]), 2.0), (np.array([9.0, 0.6, 0.0]), 1.0)]
    partition = path_partition(np.array([1.0, 0.0, 0.0]), 12.0, spheres, [(0.5, 10.0)])
    logz, velocity = identifiable_void_prediction(0.02, partition["L_void"], 1.0)
    checks = [
        ("VQ05_D0_FINITE_IVP", float(np.max(np.abs(d0 - expected_d0))) < 1e-14, float(np.max(np.abs(d0 - expected_d0)))),
        ("VQ05_NOT_INSTANT_EQUILIBRIUM", float(np.max(np.abs(d0 - source / gamma))) > 1e-3, float(np.max(np.abs(d0 - source / gamma)))),
        ("VQ06_MIDPOINT_SELF_FINITE", bool(np.all(np.isfinite(column))) and float(np.max(column)) > 0.0, float(np.max(column))),
        ("VQ08_ANALYTIC_UNION", math.isclose(partition["L_void"], 7.6, abs_tol=1e-12), abs(partition["L_void"] - 7.6)),
        ("VQ08_THREE_WAY_PARTITION", math.isclose(sum(partition[key] for key in ("L_void", "L_observed_matter", "L_unobserved")), 12.0, abs_tol=1e-12), abs(sum(partition[key] for key in ("L_void", "L_observed_matter", "L_unobserved")) - 12.0)),
        ("DELTA_H_SIGN_AND_UNITS", logz > 0.0 and math.isclose(velocity, 0.02 * partition["L_void"]), abs(velocity - 0.02 * partition["L_void"])),
        ("CANONICAL_SPLIT_STABLE", split_bucket("12345") == split_bucket(12345), float(split_bucket(12345)[0])),
        ("RESPONSES_UNOPENED", config["access_accounting"]["scientific_rows_decoded"] == 0, 0.0),
    ]
    return [{"check_id": name, "passed": bool(passed), "diagnostic": float(diagnostic)} for name, passed, diagnostic in checks]


def build_receipt() -> tuple[dict[str, Any], dict[Path, bytes]]:
    config = load_config()
    blocked = _bind_receipt(config["blocked_predecessor"])
    geometry = _bind_receipt(config["geometry_source"])
    _require(geometry["status"] == config["geometry_source"]["status"], "geometry status drift")
    gates = synthetic_gates(config)
    _require(all(row["passed"] for row in gates), "repair gate failed")
    payloads = {
        ARTIFACT_DIR / "repair-gates.json": _pretty(gates),
        ARTIFACT_DIR / "report.md": b"# Lane-9 law v3\n\nAll repaired synthetic gates pass. Only delta_H is identifiable and frozen for a future correlation-only executor. No CF4 or VAST scientific row was decoded.\n",
    }
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-gravitational-load-receipt-3.0",
        "package_id": config["package_id"],
        "status": "PASS_REPAIRED_CORRELATION_LAWS_ROWS_UNOPENED",
        "decision": "CORRELATION_ONLY_EXECUTOR_MAY_NOW_BE_FROZEN_BEFORE_ROW_DECODE",
        "counterevidence_receipt_content_sha256": blocked["content_sha256"],
        "geometry_source_receipt_content_sha256": geometry["content_sha256"],
        "dimensions_LMT": config["dimensions_LMT"],
        "repairs": config["repairs"],
        "response_contract": config["response_contract"],
        "split_contract": config["split_contract"],
        "gates": gates,
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
        "bindings": {
            "config_raw_sha256": file_sha256(CONFIG_PATH), "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH), "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
            "blocked_predecessor_raw_sha256": config["blocked_predecessor"]["raw_sha256"],
            "geometry_source_raw_sha256": config["geometry_source"]["raw_sha256"],
        },
        "artifact_index": [{"path": path.as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()} for path, payload in sorted(payloads.items(), key=lambda row: row[0].as_posix())],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt, payloads


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"existing output differs: {path}")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_package() -> str:
    receipt, payloads = build_receipt()
    for path, payload in payloads.items():
        _atomic_no_clobber(path, payload)
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(receipt))


def check_package() -> dict[str, Any]:
    _require(OUTPUT_PATH.is_file(), "receipt missing")
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected, payloads = build_receipt()
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "receipt drift")
    for path, payload in payloads.items():
        _require(path.is_file() and path.read_bytes() == payload, f"artifact drift: {path}")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_package())
    else:
        receipt = check_package()
        print("VALID" if args.command == "check" else receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
