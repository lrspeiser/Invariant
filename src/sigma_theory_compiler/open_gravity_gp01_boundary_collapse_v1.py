"""Exact GP01 zero-length boundary-collapse theorem and probes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_gp01_boundary_collapse_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_gp01_boundary_collapse_v1.py")
TEST_PATH = Path("tests/test_open_gravity_gp01_boundary_collapse_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-gp01-boundary-collapse-v1/receipt.json")
_CANONICAL_OUTPUT_PATH = Path("runs/gravity/open-gravity-gp01-boundary-collapse-v1/receipt.json")
_SCHEMA = "invariant-open-gravity-gp01-boundary-collapse-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-gp01-boundary-collapse-receipt-1.0"
_CONFIG_CONTENT_SHA256 = "574eb2689b890e19f73289eff6719ffd509cd7f712c8b26ad7273c89fcf32639"


class BoundaryCollapseError(RuntimeError):
    """Raised when the boundary theorem or its frozen evidence fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BoundaryCollapseError(message)


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
        raise BoundaryCollapseError(f"invalid {label}") from error


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    required = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "definitions",
        "theorem",
        "probe_contract",
        "empirical_boundary_evidence",
        "decision",
        "required_checks",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(set(config) == required, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == "open-gravity-gp01-boundary-collapse-v1", "ID changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(len(config["required_checks"]) == 9, "check inventory changed")
    _require(set(config["access_contract"].values()) == {0}, "access changed")
    evidence = config["empirical_boundary_evidence"]
    _require(evidence["objects"] == 8, "object count changed")
    _require(evidence["elliptic_beats_equilibrium_on_objects"] == 0, "counterexample erased")
    _require(config["decision"]["broader_gp01_history_family"] == "ACTIVE", "family eliminated")


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH, "boundary config")
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
    _require(len(observed) == 3, "binding count changed")
    return observed


def equilibrium_log_gain(y: float, n: int) -> float:
    _require(math.isfinite(y) and y > 0.0, "y must be positive")
    _require(n in (1, 2, 4), "n changed")
    return math.log1p(y ** (-n)) / (2.0 * n)


def environment_window(
    density: float,
    tide: float,
    *,
    density_threshold: float,
    tide_threshold: float,
    q: int,
    p: int,
) -> float:
    _require(density >= 0.0 and tide >= 0.0, "source variables negative")
    _require(density_threshold > 0.0 and tide_threshold > 0.0, "threshold invalid")
    _require(q in (1, 2) and p in (1, 2), "exponent changed")
    return 1.0 / (1.0 + (density / density_threshold) ** q + (tide / tide_threshold) ** p)


def bounded_target(gamma_l: float, *, window: float, gamma_max: float) -> float:
    _require(gamma_l >= 0.0 and gamma_max > 0.0, "gain parameters invalid")
    _require(0.0 <= window <= 1.0, "window invalid")
    return window * gamma_max * math.tanh(gamma_l / gamma_max)


def analytic_error_bound(gamma_l: float, *, window: float, gamma_max: float) -> float:
    return (1.0 - window) * gamma_l + window * gamma_l**3 / (3.0 * gamma_max**2)


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    probe = config["probe_contract"]
    cases: list[dict[str, float | int]] = []
    maximum_identity_error = 0.0
    maximum_bound_excess = 0.0
    minimum_ordering_margin = math.inf
    for n, y, gamma_max, density_ratio, tide_ratio, q, p in itertools.product(
        probe["n_grid"],
        probe["y_grid"],
        probe["gamma_max_grid"],
        probe["threshold_ratio_grid"],
        probe["threshold_ratio_grid"],
        probe["exponent_grid"],
        probe["exponent_grid"],
    ):
        gamma_l = equilibrium_log_gain(float(y), int(n))
        window = environment_window(
            1.0,
            1.0,
            density_threshold=float(density_ratio),
            tide_threshold=float(tide_ratio),
            q=int(q),
            p=int(p),
        )
        target = bounded_target(gamma_l, window=window, gamma_max=float(gamma_max))
        error = gamma_l - target
        decomposition = (1.0 - window) * gamma_l + window * (
            gamma_l - float(gamma_max) * math.tanh(gamma_l / float(gamma_max))
        )
        bound = analytic_error_bound(gamma_l, window=window, gamma_max=float(gamma_max))
        maximum_identity_error = max(maximum_identity_error, abs(error - decomposition))
        maximum_bound_excess = max(maximum_bound_excess, error - bound)
        minimum_ordering_margin = min(minimum_ordering_margin, target, error)
        cases.append(
            {
                "n": int(n),
                "y": float(y),
                "gamma_max": float(gamma_max),
                "density_ratio": float(density_ratio),
                "tide_ratio": float(tide_ratio),
                "q": int(q),
                "p": int(p),
                "gamma_l": gamma_l,
                "window": window,
                "target": target,
                "error": error,
                "bound": bound,
            }
        )
    _require(minimum_ordering_margin >= -1.0e-15, "ordering failed")
    _require(maximum_identity_error <= 2.0e-15, "decomposition failed")
    _require(maximum_bound_excess <= 2.0e-15, "upper bound failed")

    monotone_gamma = True
    monotone_threshold = True
    for n, y in itertools.product(probe["n_grid"], probe["y_grid"]):
        gamma_l = equilibrium_log_gain(float(y), int(n))
        gamma_values = [
            bounded_target(gamma_l, window=1.0, gamma_max=float(value))
            for value in probe["gamma_max_grid"]
        ]
        monotone_gamma &= all(right >= left for left, right in itertools.pairwise(gamma_values))
        threshold_values = [
            bounded_target(
                gamma_l,
                window=environment_window(
                    1.0,
                    1.0,
                    density_threshold=float(value),
                    tide_threshold=float(value),
                    q=2,
                    p=2,
                ),
                gamma_max=1000.0,
            )
            for value in probe["threshold_ratio_grid"]
        ]
        monotone_threshold &= all(
            right >= left for left, right in itertools.pairwise(threshold_values)
        )
    _require(monotone_gamma, "gamma_max is not monotone")
    _require(monotone_threshold, "threshold is not monotone")

    limit_relative_errors = []
    for n, y in itertools.product(probe["n_grid"], probe["y_grid"]):
        gamma_l = equilibrium_log_gain(float(y), int(n))
        window = environment_window(
            1.0,
            1.0,
            density_threshold=1.0e12,
            tide_threshold=1.0e12,
            q=2,
            p=2,
        )
        target = bounded_target(gamma_l, window=window, gamma_max=1000.0)
        limit_relative_errors.append(abs(target - gamma_l) / max(gamma_l, 1.0e-300))
    maximum_limit_relative_error = max(limit_relative_errors)
    _require(maximum_limit_relative_error < 1.0e-5, "numeric collapse limit too weak")
    return {
        "probe_cases": len(cases),
        "case_root_sha256": content_sha256(cases),
        "minimum_ordering_margin": minimum_ordering_margin,
        "maximum_decomposition_error": maximum_identity_error,
        "maximum_bound_excess": maximum_bound_excess,
        "monotone_gamma_max": monotone_gamma,
        "monotone_environment_thresholds": monotone_threshold,
        "maximum_limit_relative_error": maximum_limit_relative_error,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    bindings = _validate_bindings(config)
    suite = run_suite(config)
    evidence = config["empirical_boundary_evidence"]
    checks = {
        "NONNEGATIVE_ORDERING": suite["minimum_ordering_margin"] >= -1.0e-15,
        "EXACT_ERROR_DECOMPOSITION": suite["maximum_decomposition_error"] <= 2.0e-15,
        "GLOBAL_ERROR_UPPER_BOUND": suite["maximum_bound_excess"] <= 2.0e-15,
        "MONOTONE_GAMMA_MAX": suite["monotone_gamma_max"],
        "MONOTONE_ENVIRONMENT_THRESHOLDS": suite["monotone_environment_thresholds"],
        "NUMERIC_COLLAPSE_LIMIT": suite["maximum_limit_relative_error"] < 1.0e-5,
        "EMPIRICAL_COUNTEREXAMPLE_RETAINED": evidence["elliptic_beats_equilibrium_on_objects"] == 0
        and evidence["elliptic_to_equilibrium_robust_loss_ratio"] > 1.0,
        "BROADER_DYNAMIC_FAMILY_RETAINED": config["decision"]["broader_gp01_history_family"]
        == "ACTIVE",
        "ZERO_RESPONSE_ACCESS": set(config["access_contract"].values()) == {0},
    }
    _require(all(checks.values()), "boundary-collapse check failed")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": config["status"],
        "bindings": bindings,
        "theorem": config["theorem"],
        "suite": suite,
        "empirical_boundary_evidence": evidence,
        "decision": config["decision"],
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
    _require(dict(payload) == build_receipt(), "boundary receipt differs from rebuild")


def write_receipt() -> str:
    payload = build_receipt()
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        _require(OUTPUT_PATH.read_bytes() == encoded, "existing boundary receipt differs")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=OUTPUT_PATH.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, OUTPUT_PATH)
    except FileExistsError:
        _require(OUTPUT_PATH.read_bytes() == encoded, "concurrent boundary receipt differs")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def validate_receipt() -> None:
    _require(OUTPUT_PATH.is_file(), "boundary receipt absent")
    payload = _read_json(OUTPUT_PATH, "boundary receipt")
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
                    "probe_cases": receipt["suite"]["probe_cases"],
                    "checks_passed": sum(receipt["checks"].values()),
                    "broader_history_family": receipt["decision"]["broader_gp01_history_family"],
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
