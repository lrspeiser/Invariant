"""Restricted universal-conformal action for the quadrature shared formula."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_quadrature_covariant_action_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_shared_quadrature_covariant_action.py")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_covariant_action.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-covariant-action-v1.json")

EXPECTED_CONFIG_FILE_SHA256 = "dca011dce2390ea37a853a44b501d24865185c83a7f7ecf6ab3e0eeaf69fdc64"
EXPECTED_CONFIG_CONTENT_SHA256 = "9467ccf6c7bd2d5512f537eaa3f323a46b861fa10d8ad3b0b1133d8962ea240b"
DECISION = (
    "RESTRICTED_QUADRATURE_UNIVERSAL_CONFORMAL_ACTION_DERIVED_EXACT_MOTION_LAW_"
    "AND_STRESS_ESTABLISHED_DIRECT_LENSING_CANCELS_CAUSAL_ENDPOINT_GLOBAL_AND_"
    "QUANTITATIVE_LENSING_GATES_FAIL"
)
SYMBOLIC_CHECK_IDS = (
    "S01_KINETIC_DENSITY_DERIVATIVE",
    "S02_BRANCH_COORDINATE_DERIVATIVE",
    "S03_KINETIC_COEFFICIENT",
    "S04_LONGITUDINAL_COEFFICIENT",
    "S05_QUADRATURE_INVERSE_RELATION",
    "S06_SPHERICAL_FLUX_NORMALIZATION",
    "S07_QUADRATURE_MOTION_LAW",
    "S08_SCALAR_CONE_RATIO",
    "S09_STATIC_ENERGY_DENSITY",
    "S10_STATIC_RADIAL_PRESSURE",
    "S11_STATIC_TANGENTIAL_PRESSURE",
    "S12_STATIC_ANISOTROPY",
    "S13_RADIAL_NEC",
    "S14_TANGENTIAL_NEC",
    "S15_LOW_GRADIENT_DENSITY_ASYMPTOTE",
    "S16_LOW_GRADIENT_C_ZERO",
    "S17_LOW_GRADIENT_K_ZERO",
    "S18_LOW_GRADIENT_CONE_LIMIT",
    "S19_FINITE_GRADIENT_ENDPOINT",
    "S20_HIGH_SOURCE_C_DIVERGENCE",
    "S21_HIGH_SOURCE_K_DIVERGENCE",
    "S22_HIGH_SOURCE_DENSITY_DIVERGENCE",
    "S23_DIRECT_CONFORMAL_LENSING_CANCELLATION",
    "S24_CONFORMAL_NULL_CONE_IDENTITY",
)


class QuadratureActionError(RuntimeError):
    """Raised when the frozen action package changes or a derivation fails."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QuadratureActionError(message)


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    _require(path.is_file(), f"missing config: {path}")
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "action config file hash changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QuadratureActionError(f"cannot load action config: {path}") from exc
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "action config content changed")
    _require(
        config["schema_version"]
        == "invariant-gravity-shared-quadrature-covariant-action-config-1.0",
        "action config schema changed",
    )
    _require(config["adjudication"]["overall_decision"] == DECISION, "decision changed")
    _require(
        tuple(config["machine_check_contract"]["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic check inventory changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    try:
        kind = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.strip()
        _require(kind == b"commit", "predecessor binding is not a commit")
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise QuadratureActionError("cannot validate predecessor git binding") from exc


def validate_predecessors(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for binding in config["predecessor_bindings"]:
        commit = str(binding["git_commit"])
        artifacts = list(binding["artifacts"])
        _require(artifacts, "predecessor artifact inventory is empty")
        for artifact in artifacts:
            path = Path(str(artifact["path"]))
            expected = str(artifact["file_sha256"])
            _require((root / path).is_file(), f"missing predecessor: {path}")
            _require(_file_sha(root / path) == expected, f"predecessor hash changed: {path}")
            _require(
                hashlib.sha256(_git_bytes(root, commit, path.as_posix())).hexdigest() == expected,
                f"predecessor commit bytes changed: {path}",
            )
        receipt_path = Path(str(binding["receipt_path"]))
        try:
            receipt = json.loads((root / receipt_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QuadratureActionError(f"cannot load predecessor receipt: {receipt_path}") from exc
        _require(
            receipt["content_sha256"] == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        _require(
            receipt["schema_version"] == binding["receipt_schema_version"],
            "predecessor receipt schema changed",
        )
        _require(receipt["decision"] == binding["receipt_decision"], "predecessor decision changed")
        results.append(
            {
                "binding_id": binding["binding_id"],
                "git_commit": commit,
                "artifact_count": len(artifacts),
                "all_current_and_commit_hashes_match": True,
            }
        )
    return results


def _check(check_id: str, expression: sp.Expr, statement: str) -> dict[str, Any]:
    residual = sp.simplify(expression)
    _require(residual == 0, f"symbolic check failed: {check_id}: {residual}")
    return {"check_id": check_id, "passed": True, "statement": statement}


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    s, alpha = sp.symbols("s alpha", positive=True)
    p = s**2 / 4 + s / 4 + sp.log(1 - 2 * s) / 8
    p_s = sp.diff(p, s)
    xbar = -(s**2) / (2 * alpha**2)
    xbar_s = sp.diff(xbar, s)
    p_x = sp.simplify(p_s / xbar_s)
    p_xx = sp.diff(p_x, s) / xbar_s
    longitudinal = sp.simplify(p_x + 2 * xbar * p_xx)
    x = s**2 / (1 - 2 * s)
    y = x + s
    cone_ratio = sp.simplify(longitudinal / p_x)

    rho = -p
    radial_pressure = p + p_x * s**2 / alpha**2
    tangential_pressure = p

    phi_e, psi_e, conformal_shift = sp.symbols("Phi_E Psi_E a", real=True)
    phi_tilde = phi_e + conformal_shift
    psi_tilde = psi_e - conformal_shift
    conformal_factor, gkk = sp.symbols("A2 gkk", positive=True)

    checks = [
        _check(
            "S01_KINETIC_DENSITY_DERIVATIVE",
            p_s + s**2 / (1 - 2 * s),
            "The branch kinetic density differentiates to -s^2/(1-2s).",
        ),
        _check(
            "S02_BRANCH_COORDINATE_DERIVATIVE",
            1 / xbar_s + alpha**2 / s,
            "The branch coordinate obeys ds/dXbar=-alpha^2/s.",
        ),
        _check(
            "S03_KINETIC_COEFFICIENT",
            p_x - alpha**2 * s / (1 - 2 * s),
            "The action gives the frozen minimal kinetic coefficient.",
        ),
        _check(
            "S04_LONGITUDINAL_COEFFICIENT",
            longitudinal - 2 * alpha**2 * s * (1 - s) / (1 - 2 * s) ** 2,
            "The longitudinal coefficient is positive on the open branch.",
        ),
        _check(
            "S05_QUADRATURE_INVERSE_RELATION",
            x * (1 - 2 * s) - s**2,
            "The branch inverse relation is x=s^2/(1-2s).",
        ),
        _check(
            "S06_SPHERICAL_FLUX_NORMALIZATION",
            p_x * s / alpha - alpha * x,
            "The integrated scalar flux exactly matches alpha times g_N.",
        ),
        _check(
            "S07_QUADRATURE_MOTION_LAW",
            y**2 - x**2 - x,
            "The positive total acceleration branch is y=sqrt(x^2+x).",
        ),
        _check(
            "S08_SCALAR_CONE_RATIO",
            cone_ratio - 2 * (1 - s) / (1 - 2 * s),
            "The scalar radial cone ratio exceeds two on 0<s<1/2.",
        ),
        _check(
            "S09_STATIC_ENERGY_DENSITY",
            rho + p,
            "The static scalar energy density is -p in frozen units.",
        ),
        _check(
            "S10_STATIC_RADIAL_PRESSURE",
            radial_pressure - (p + s**3 / (1 - 2 * s)),
            "The radial pressure includes the scalar-gradient anisotropy.",
        ),
        _check(
            "S11_STATIC_TANGENTIAL_PRESSURE",
            tangential_pressure - p,
            "The two tangential pressures equal p.",
        ),
        _check(
            "S12_STATIC_ANISOTROPY",
            radial_pressure - tangential_pressure - s**3 / (1 - 2 * s),
            "The static pressure anisotropy is positive on the open branch.",
        ),
        _check(
            "S13_RADIAL_NEC",
            rho + radial_pressure - s**3 / (1 - 2 * s),
            "The radial null-energy combination is positive.",
        ),
        _check(
            "S14_TANGENTIAL_NEC",
            rho + tangential_pressure,
            "The tangential null-energy combination is saturated.",
        ),
        _check(
            "S15_LOW_GRADIENT_DENSITY_ASYMPTOTE",
            sp.limit(rho / s**3, s, 0, dir="+") - sp.Rational(1, 3),
            "The low-gradient energy density scales as s^3/3.",
        ),
        _check(
            "S16_LOW_GRADIENT_C_ZERO",
            sp.limit(p_x, s, 0, dir="+"),
            "The kinetic coefficient vanishes at the low-gradient boundary.",
        ),
        _check(
            "S17_LOW_GRADIENT_K_ZERO",
            sp.limit(longitudinal, s, 0, dir="+"),
            "The longitudinal coefficient vanishes at the low-gradient boundary.",
        ),
        _check(
            "S18_LOW_GRADIENT_CONE_LIMIT",
            sp.limit(cone_ratio, s, 0, dir="+") - 2,
            "The scalar radial cone ratio tends to two, not one.",
        ),
        _check(
            "S19_FINITE_GRADIENT_ENDPOINT",
            sp.limit(xbar, s, sp.Rational(1, 2), dir="-") + 1 / (8 * alpha**2),
            "The high-source limit reaches a finite scalar-gradient endpoint.",
        ),
        _check(
            "S20_HIGH_SOURCE_C_DIVERGENCE",
            sp.limit(1 / p_x, s, sp.Rational(1, 2), dir="-"),
            "The inverse kinetic coefficient vanishes at the endpoint.",
        ),
        _check(
            "S21_HIGH_SOURCE_K_DIVERGENCE",
            sp.limit(1 / longitudinal, s, sp.Rational(1, 2), dir="-"),
            "The inverse longitudinal coefficient vanishes at the endpoint.",
        ),
        _check(
            "S22_HIGH_SOURCE_DENSITY_DIVERGENCE",
            sp.limit(1 / rho, s, sp.Rational(1, 2), dir="-"),
            "The inverse energy density vanishes at the endpoint.",
        ),
        _check(
            "S23_DIRECT_CONFORMAL_LENSING_CANCELLATION",
            phi_tilde + psi_tilde - phi_e - psi_e,
            "The direct conformal scalar shifts cancel from Phi+Psi.",
        ),
        _check(
            "S24_CONFORMAL_NULL_CONE_IDENTITY",
            (conformal_factor * gkk).subs(gkk, 0),
            "A positive conformal factor preserves every Einstein-frame null direction.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic check order changed",
    )
    formulas = {
        "dimensionless_kinetic_density": "s^2/4+s/4+ln(1-2*s)/8",
        "Xbar": "-s^2/(2*alpha^2)",
        "p_Xbar": "alpha^2*s/(1-2*s)",
        "K": "2*alpha^2*s*(1-s)/(1-2*s)^2",
        "motion_law": "y=sqrt(x^2+x)",
        "normalized_energy_density": "-p",
        "normalized_radial_pressure": "p+s^3/(1-2*s)",
        "normalized_tangential_pressure": "p",
    }
    return checks, formulas


def numeric_checks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    results: list[dict[str, Any]] = []
    for s in map(float, config["machine_check_contract"]["branch_probes"]):
        _require(0.0 < s < 0.5, "numeric branch probe left the declared domain")
        p = s * s / 4.0 + s / 4.0 + math.log1p(-2.0 * s) / 8.0
        c = s / (1.0 - 2.0 * s)
        k = 2.0 * s * (1.0 - s) / (1.0 - 2.0 * s) ** 2
        rho = -p
        radial_pressure = p + s**3 / (1.0 - 2.0 * s)
        tangential_pressure = p
        radial_nec = rho + radial_pressure
        tangential_nec = rho + tangential_pressure
        passed = (
            p < 0.0
            and c > 0.0
            and k > 0.0
            and k / c > 2.0
            and rho > 0.0
            and radial_nec > 0.0
            and abs(tangential_nec) <= tolerance
        )
        _require(passed, "numeric restricted-action branch probe failed")
        results.append(
            {
                "s": s,
                "p": p,
                "C_over_alpha_squared": c,
                "K_over_alpha_squared": k,
                "K_over_C": k / c,
                "normalized_energy_density": rho,
                "normalized_radial_pressure": radial_pressure,
                "normalized_tangential_pressure": tangential_pressure,
                "normalized_radial_NEC": radial_nec,
                "normalized_tangential_NEC": tangential_nec,
                "passed": True,
            }
        )
    return results


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    predecessors = validate_predecessors(root, config)
    symbolic, formulas = symbolic_checks()
    numeric = numeric_checks(config)
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-shared-quadrature-covariant-action-receipt-1.0",
        "analysis_id": config["analysis_id"],
        "status": config["status"],
        "decision": DECISION,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(root / CONFIG_PATH),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "source_path": SOURCE_PATH.as_posix(),
            "source_file_sha256": _file_sha(root / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(root / TEST_PATH),
        },
        "predecessor_validation": predecessors,
        "restricted_action_contract": config["restricted_action_contract"],
        "spherical_motion_contract": config["spherical_motion_contract"],
        "stress_and_lensing_contract": config["stress_and_lensing_contract"],
        "obstruction_contract": config["obstruction_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "symbolic_formulas": formulas,
            "numeric_branch_probes": numeric,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic),
            "numeric_branch_probes": len(numeric),
            "numeric_branch_probes_passed": sum(item["passed"] for item in numeric),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "remaining_obligations": config["remaining_obligations"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The action is defined only on an open static spacelike branch and is not a global cosmological completion.",
            "The exact spherical motion derivation assumes weak-field isolated dust and does not solve an extended nonspherical system.",
            "Universal conformal coupling supplies no independent photon adjustment; quantitative lensing still requires the unsolved Einstein-scalar boundary-value problem.",
            "Positive local energy and NEC combinations do not repair the scalar cone, transition degeneracy, or finite-gradient endpoint.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _require(dict(receipt) == build_receipt(root), "stored quadrature action receipt changed")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return "EXISTING_IDENTICAL" if path.read_bytes() == payload else "EXISTING_DIFFERENT"
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            return "EXISTING_IDENTICAL" if path.read_bytes() == payload else "EXISTING_DIFFERENT"
        return "CREATED"
    finally:
        temp.unlink(missing_ok=True)


def write_receipt(root: Path) -> Path:
    path = root / OUTPUT_PATH
    payload = json.dumps(build_receipt(root), indent=2, sort_keys=True).encode() + b"\n"
    outcome = _atomic_no_clobber(path, payload)
    _require(outcome != "EXISTING_DIFFERENT", f"refusing to overwrite existing receipt: {path}")
    return path


def check_receipt(root: Path) -> dict[str, Any]:
    path = root / OUTPUT_PATH
    _require(path.is_file(), f"missing quadrature action receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root)
    return {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "numeric_branch_probes_passed": receipt["counts"]["numeric_branch_probes_passed"],
        "quadrature_motion_law_recovered_exactly": receipt["adjudication"][
            "quadrature_motion_law_recovered_exactly"
        ],
        "same_action_quantitative_lensing_solution_derived": receipt["adjudication"][
            "same_action_quantitative_lensing_solution_derived"
        ],
        "content_sha256": receipt["content_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "write":
        print(json.dumps({"path": str(write_receipt(root))}, sort_keys=True))
        return 0
    print(json.dumps(check_receipt(root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
