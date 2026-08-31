"""Target-free recovery, null, and falsification gates for executable 3-D fields."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler import open_gravity_3d_halo_modified_gravity_comparators_v1 as comp
from sigma_theory_compiler import open_gravity_3d_newton_aqual_qumond_baselines_v1 as base
from sigma_theory_compiler import open_gravity_common_3d_synthetic_universe_v1 as universe
from sigma_theory_compiler import open_gravity_gp01_full3d_dynamics_v1 as gp3d

CONFIG_PATH = Path("configs/open_gravity_3d_synthetic_recovery_falsification_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_synthetic_recovery_falsification_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_3d_synthetic_recovery_falsification_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-3d-synthetic-recovery-falsification-v1/receipt.json")
_CANONICAL_CONFIG_PATH = Path("configs/open_gravity_3d_synthetic_recovery_falsification_v1.json")
_CANONICAL_MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_3d_synthetic_recovery_falsification_v1.py"
)
_CANONICAL_TEST_PATH = Path("tests/test_open_gravity_3d_synthetic_recovery_falsification_v1.py")
_CANONICAL_OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-3d-synthetic-recovery-falsification-v1/receipt.json"
)
_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_RAW_SHA256 = "f62f4a3d3b55daacf3e78a76504ad109569445bb29b657ebb8d03d19a1e6638b"
_CONFIG_CONTENT_SHA256 = "261567c375eb389a1177129b77b653babb25204785ed14072e1ba51a81bbb64d"
_SCHEMA = "invariant-open-gravity-3d-synthetic-recovery-falsification-1.0"
_RECEIPT_SCHEMA = "invariant-open-gravity-3d-synthetic-recovery-falsification-receipt-1.0"
_MECHANISMS = (
    "NEWTON",
    "AQUAL",
    "QUMOND",
    "MOG_WEAK_FIELD",
    "REFRACTED_GRAVITY",
    "GP01_ELLIPTIC",
)


class RecoveryError(RuntimeError):
    """Raised whenever the recovery packet fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(current: Path, expected: Path, label: str) -> Path:
    _require(current == expected, f"canonical {label} path changed")
    path = (_ROOT / expected).resolve()
    _require(path.is_relative_to(_ROOT), f"{label} escaped repository")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read {label}") from exc
    _require(type(value) is dict, f"{label} is not an object")
    return value


def _git_show(commit: str, path: str) -> bytes:
    try:
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=_ROOT, check=True, capture_output=True
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError("committed binding unavailable") from exc


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema",
        "package_id",
        "status",
        "purpose",
        "bindings",
        "mechanisms",
        "test_contract",
        "required_gates",
        "blocked_or_nonindependent",
        "access_contract",
        "claim_boundary",
        "output_path",
    }
    _require(type(config) is dict and set(config) == expected, "config keys changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(
        config["package_id"] == "open-gravity-3d-synthetic-recovery-falsification-v1",
        "ID changed",
    )
    _require(config["status"] == "FROZEN_TARGET_FREE_RECOVERY_NULL_FALSIFICATION", "status changed")
    _require(config["output_path"] == _CANONICAL_OUTPUT_PATH.as_posix(), "output changed")
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(tuple(row["id"] for row in config["mechanisms"]) == _MECHANISMS, "mechanisms changed")
    _require(len(config["required_gates"]) == 11, "gates changed")
    _require(all(value == 0 for value in config["access_contract"].values()), "access changed")


def load_config() -> dict[str, Any]:
    path = _path(CONFIG_PATH, _CANONICAL_CONFIG_PATH, "config")
    raw = path.read_bytes()
    _require(hashlib.sha256(raw).hexdigest() == _CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path, "recovery config")
    validate_config(config)
    for binding in config["bindings"]:
        for artifact in binding["artifacts"]:
            expected = artifact["sha256"]
            _require(
                hashlib.sha256(_git_show(binding["commit"], artifact["path"])).hexdigest()
                == expected,
                f"committed {binding['role']} changed",
            )
            _require(
                file_sha256(_ROOT / artifact["path"]) == expected,
                f"working {binding['role']} changed",
            )
    return config


def _relative(first: np.ndarray, second: np.ndarray) -> float:
    scale = max(float(np.linalg.norm(second.ravel())), 1.0e-30)
    return float(np.linalg.norm((first - second).ravel()) / scale)


def solve_mechanism(
    mechanism: str,
    density: np.ndarray,
    fixtures: universe.FixtureSet,
    parameters: Mapping[str, Any],
) -> tuple[np.ndarray, float, bool]:
    _require(mechanism in _MECHANISMS, "unknown mechanism")
    grid = fixtures.grid
    zero = np.zeros(grid.shape, dtype=np.float64)
    rhs = 4.0 * math.pi * np.asarray(density, dtype=np.float64)
    if mechanism == "NEWTON":
        result = base.solve_poisson(rhs, zero, grid.spacing)
        return result.potential, result.relative_residual, result.converged
    if mechanism == "AQUAL":
        result = base.solve_aqual(
            rhs,
            zero,
            grid.spacing,
            a0=float(parameters["a0"]),
            mu_floor=float(parameters["mu_floor"]),
        )
        return result.potential, result.relative_residual, result.converged
    if mechanism == "QUMOND":
        _newton, result, _effective = base.solve_qumond(
            rhs,
            zero,
            zero,
            grid.spacing,
            a0=float(parameters["a0"]),
            nu_floor=float(parameters["nu_floor"]),
        )
        return result.potential, result.relative_residual, result.converged
    if mechanism == "MOG_WEAK_FIELD":
        result, _newton, _yukawa = comp.solve_mog_weak_field(
            density,
            zero,
            grid.spacing,
            alpha=float(parameters["alpha"]),
            mu=float(parameters["mu"]),
        )
        return result.potential, result.relative_residual, True
    if mechanism == "REFRACTED_GRAVITY":
        result, _coefficient = comp.solve_refracted(
            density,
            zero,
            grid.spacing,
            epsilon_0=float(parameters["epsilon_0"]),
            rho_c=float(parameters["rho_c"]),
            q_slope=float(parameters["q_slope"]),
        )
        return result.potential, result.relative_residual, True
    result = gp3d.solve_gp01_elliptic(
        density,
        grid,
        n=int(parameters["n"]),
        gamma_max=float(parameters["Gamma_max"]),
        length=float(parameters["L_g"]),
    )
    return result.potential, max(result.gain_residual, result.potential_residual), True


def _gate(passed: bool, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {"passed": bool(passed), "metrics": dict(metrics)}


def run_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    fixtures = universe.build_fixtures(universe.load_config())
    density = fixtures.sources[config["test_contract"]["primary_fixture"]]
    parameter_map = {row["id"]: row["parameters"] for row in config["mechanisms"]}
    outputs: dict[str, np.ndarray] = {}
    residuals: dict[str, float] = {}
    convergence: dict[str, bool] = {}
    for mechanism in _MECHANISMS:
        field, residual, converged = solve_mechanism(
            mechanism, density, fixtures, parameter_map[mechanism]
        )
        outputs[mechanism] = field
        residuals[mechanism] = residual
        convergence[mechanism] = converged and bool(np.all(np.isfinite(field)))

    gates: dict[str, dict[str, Any]] = {}
    gates["ALL_EXECUTABLE_MECHANISMS_RETURN_FINITE_FIELDS"] = _gate(
        all(convergence.values())
        and residuals["AQUAL"] < 2.0e-7
        and max(value for key, value in residuals.items() if key != "AQUAL") < 1.0e-10,
        {"mechanisms": len(outputs), "residuals": residuals, "converged": convergence},
    )

    pattern = (
        np.sin(math.pi * fixtures.grid.x)
        * np.cos(math.pi * fixtures.grid.y)
        * np.sin(math.pi * fixtures.grid.z)
    )
    noise_level = float(config["test_contract"]["injection_relative_noise"])
    recovered: dict[str, str] = {}
    margins: dict[str, float] = {}
    for injected, reference in outputs.items():
        scale = max(float(np.max(np.abs(reference))), 1.0)
        probe = reference + noise_level * scale * pattern
        distances = {candidate: _relative(probe, field) for candidate, field in outputs.items()}
        ordered = sorted(distances, key=distances.get)
        recovered[injected] = ordered[0]
        margins[injected] = distances[ordered[1]] - distances[ordered[0]]
    gates["SELF_INJECTION_NEAREST_SIGNATURE_RECOVERY"] = _gate(
        all(injected == selected for injected, selected in recovered.items())
        and min(margins.values()) > 0.0,
        {"recovered": recovered, "minimum_nearest_margin": min(margins.values())},
    )

    zero_density = np.zeros_like(density)
    null_maxima: dict[str, float] = {}
    for mechanism in _MECHANISMS:
        field, _residual, converged = solve_mechanism(
            mechanism, zero_density, fixtures, parameter_map[mechanism]
        )
        _require(converged, f"{mechanism} null did not converge")
        null_maxima[mechanism] = float(np.max(np.abs(field)))
    gates["ZERO_SOURCE_NULL_RETAINED"] = _gate(
        max(null_maxima.values()) <= float(config["test_contract"]["null_tolerance"]),
        {"maximum_by_mechanism": null_maxima},
    )

    pairwise: dict[str, float] = {}
    for index, first in enumerate(_MECHANISMS):
        for second in _MECHANISMS[index + 1 :]:
            pairwise[f"{first}__{second}"] = min(
                _relative(outputs[first], outputs[second]),
                _relative(outputs[second], outputs[first]),
            )
    gates["NONSYMMETRIC_CLOSEST_COMPARATORS_DISTINGUISHED"] = _gate(
        min(pairwise.values())
        > float(config["test_contract"]["minimum_distinct_signature_distance"]),
        {"minimum_pairwise_distance": min(pairwise.values()), "pairwise_distances": pairwise},
    )

    parent_baseline = base.run_suite(base.load_config())
    spherical = parent_baseline["metrics"]["spherical_aqual_qumond_relative_difference"]
    disk = parent_baseline["metrics"]["nonspherical_disk_aqual_qumond_relative_difference"]
    gates["SPHERICAL_DEGENERACY_REPORTED_NOT_PROMOTED"] = _gate(
        spherical < disk,
        {
            "spherical_aqual_qumond_difference": spherical,
            "disk_aqual_qumond_difference": disk,
            "independent_discovery_chance": False,
        },
    )

    shuffled_density = np.roll(density, 2, axis=0)
    shuffled, _residual, _converged = solve_mechanism(
        "NEWTON", shuffled_density, fixtures, parameter_map["NEWTON"]
    )
    shuffle_difference = _relative(shuffled, outputs["NEWTON"])
    gates["SOURCE_SHUFFLE_BREAKS_SIGNATURE"] = _gate(
        shuffle_difference > 1.0e-2,
        {"newton_signature_relative_change": shuffle_difference},
    )

    parent_comparators = comp.run_suite(comp.load_config())
    parent_gp01 = gp3d.run_suite(gp3d.load_config())
    rotation_metrics = {
        "baseline_bar": parent_baseline["metrics"]["bar_rotation_relative_error"],
        "mog": parent_comparators["gates"]["MOG_ROTATION_COVARIANCE"]["metrics"][
            "relative_rotation_error"
        ],
        "refracted": parent_comparators["gates"]["REFRACTED_ROTATION_COVARIANCE"]["metrics"][
            "relative_rotation_error"
        ],
        "gp01": max(
            parent_gp01["gates"]["ROTATION_COVARIANCE"]["metrics"]["gamma_relative_error"],
            parent_gp01["gates"]["ROTATION_COVARIANCE"]["metrics"]["potential_relative_error"],
        ),
    }
    gates["ROTATION_COVARIANCE_REPLAY"] = _gate(
        max(rotation_metrics.values()) < float(config["test_contract"]["rotation_tolerance"]),
        rotation_metrics,
    )

    high_aqual = parent_baseline["metrics"]["high_acceleration_aqual_newton_relative_difference"]
    high_qumond = parent_baseline["metrics"]["high_acceleration_qumond_newton_relative_difference"]
    gates["HIGH_ACCELERATION_NEWTON_LIMIT_REPLAY"] = _gate(
        max(high_aqual, high_qumond) < 1.0e-3,
        {"aqual_relative_difference": high_aqual, "qumond_relative_difference": high_qumond},
    )

    external_change = parent_baseline["metrics"]["external_field_internal_relative_change"]
    saddle_field = parent_gp01["gates"]["SADDLE_EXACT_NULL_TARGET"]["metrics"][
        "central_field_magnitude"
    ]
    gates["EXTERNAL_FIELD_AND_SADDLE_REPLAY"] = _gate(
        external_change > 0.1 and saddle_field < 1.0e-12,
        {"aqual_external_field_change": external_change, "gp01_saddle_field": saddle_field},
    )

    gp_dispositions = {
        row["id"]: row["status"] for row in gp3d.load_config()["branch_dispositions"]
    }
    blocked = set(gp_dispositions) >= {"GP01-T1", "GP01-T2", "GP01-ACTION-PLACEHOLDER"}
    gates["PARENT_COUNTEREXAMPLES_AND_BLOCKS_RETAINED"] = _gate(
        blocked
        and gp_dispositions["GP01-T1"].startswith("BLOCKED_")
        and gp_dispositions["GP01-T2"].startswith("BLOCKED_")
        and gp_dispositions["GP01-ACTION-PLACEHOLDER"] == "INCOMPLETE_QUARANTINE",
        {
            "gp01_local_generic_3d_control_only": True,
            "transport_branches_blocked": 2,
            "action_quarantined": True,
            "unimplemented_comparators": 4,
        },
    )
    gates["ZERO_RESPONSE_ACCESS"] = _gate(
        all(value == 0 for value in config["access_contract"].values()), config["access_contract"]
    )

    _require(list(gates) == config["required_gates"], "gate order changed")
    _require(all(row["passed"] is True for row in gates.values()), "recovery gate failed")
    return {
        "mechanisms": len(_MECHANISMS),
        "gates": gates,
        "passed": len(gates),
        "failed": 0,
        "real_response_scoring_eligible": False,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    module_path = _path(MODULE_PATH, _CANONICAL_MODULE_PATH, "module")
    test_path = _path(TEST_PATH, _CANONICAL_TEST_PATH, "test")
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": config["package_id"],
        "status": "PASS_TARGET_FREE_RECOVERY_NULL_FALSIFICATION_SIX_MECHANISMS",
        "bindings": {
            "config": {
                "path": _CANONICAL_CONFIG_PATH.as_posix(),
                "sha256": file_sha256(_ROOT / _CANONICAL_CONFIG_PATH),
                "content_sha256": content_sha256(config),
            },
            "module": {
                "path": _CANONICAL_MODULE_PATH.as_posix(),
                "sha256": file_sha256(module_path),
            },
            "test": {"path": _CANONICAL_TEST_PATH.as_posix(), "sha256": file_sha256(test_path)},
            "predecessors": config["bindings"],
        },
        "suite": run_suite(config),
        "blocked_or_nonindependent": config["blocked_or_nonindependent"],
        "access_accounting": config["access_contract"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt_payload(payload: Mapping[str, Any]) -> None:
    _require(type(payload) is dict, "receipt is not an object")
    _require(payload == build_receipt(), "receipt is not reproducible")
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    _require(payload["content_sha256"] == content_sha256(body), "receipt self-hash changed")


def _output_path() -> Path:
    return _path(OUTPUT_PATH, _CANONICAL_OUTPUT_PATH, "output")


def write_receipt() -> str:
    path = _output_path()
    payload = json.dumps(build_receipt(), sort_keys=True, indent=2).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return "CREATED"


def validate_receipt() -> None:
    validate_receipt_payload(_read_json(_output_path(), "recovery receipt"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
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
                    "mechanisms": receipt["suite"]["mechanisms"],
                    "gates_passed": receipt["suite"]["passed"],
                    "observational_authority": False,
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
