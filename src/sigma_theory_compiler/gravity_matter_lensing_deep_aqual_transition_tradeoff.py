"""Conditional deep-AQUAL transition obstruction and regulated escape tradeoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_deep_aqual_transition_tradeoff_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_deep_aqual_transition_tradeoff.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_deep_aqual_transition_tradeoff.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-deep-aqual-transition-tradeoff-v1.json")

EXPECTED_CONFIG_FILE_SHA256 = "2d5a67b6231c5fafabffdef5b369e9a42c73e179fd943c2c2925373a0c270277"
EXPECTED_CONFIG_CONTENT_SHA256 = "8d613b3c1d641fa221fc878918b5e989e421e0679264bc89ac6f567a9cde2aa0"
DECISION = (
    "CONDITIONAL_EXACT_DEEP_AQUAL_TRANSITION_NO_GO_DERIVED_REGULATED_ESCAPE_HAS_"
    "ACCURACY_CONE_AND_UNIFORMITY_COSTS_CP11_4_BLOCKED"
)
SYMBOLIC_CHECK_IDS = (
    "S01_EXACT_BRANCH_P_XX",
    "S02_EXACT_BRANCH_K",
    "S03_EXACT_BRANCH_EFFECTIVE_DETERMINANT",
    "S04_EXACT_BRANCH_C_ZERO_LIMIT",
    "S05_EXACT_BRANCH_K_ZERO_LIMIT",
    "S06_EXACT_BRANCH_DETERMINANT_ZERO_LIMIT",
    "S07_DEEP_AQUAL_P_XX_DIVERGENCE",
    "S08_DEEP_AQUAL_K_RATIO_TWO",
    "S09_DEEP_AQUAL_DETERMINANT_POWER",
    "S10_REGULATED_SPACELIKE_P_X",
    "S11_REGULATED_SPACELIKE_P_XX",
    "S12_REGULATED_SPACELIKE_K",
    "S13_REGULATED_TIMELIKE_P_XX",
    "S14_REGULATED_TIMELIKE_K",
    "S15_TRANSITION_C_MATCH",
    "S16_TRANSITION_P_XX_MATCH",
    "S17_TRANSITION_K_MATCH",
    "S18_REGULATED_SPACELIKE_SPEED",
    "S19_REGULATED_TIMELIKE_SPEED",
    "S20_REGULATED_AQUAL_RELATIVE_ERROR",
    "S21_REGULATED_LARGE_T_AQUAL_LIMIT",
    "S22_TIMELIKE_C_POSITIVE_FINITE_DOMAIN",
    "S23_TIMELIKE_K_POSITIVE_FINITE_DOMAIN",
    "S24_TIMELIKE_K_ZERO_ASYMPTOTE",
)


class DeepAqualTransitionError(RuntimeError):
    """Raised when the frozen transition package changes or a check fails."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DeepAqualTransitionError(message)


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    _require(path.is_file(), f"missing config: {path}")
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "transition config file hash changed")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DeepAqualTransitionError(f"cannot load transition config: {path}") from exc
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "transition config content changed")
    _require(
        config["schema_version"]
        == "invariant-gravity-matter-lensing-deep-aqual-transition-tradeoff-config-1.0",
        "transition config schema changed",
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
        raise DeepAqualTransitionError("cannot validate predecessor git binding") from exc


def validate_predecessors(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for binding in config["predecessor_bindings"]:
        commit = str(binding["git_commit"])
        for path_key, hash_key in (
            ("config_path", "config_file_sha256"),
            ("module_path", "module_file_sha256"),
            ("test_path", "test_file_sha256"),
            ("receipt_path", "receipt_file_sha256"),
        ):
            path = Path(str(binding[path_key]))
            expected = str(binding[hash_key])
            _require((root / path).is_file(), f"missing predecessor: {path}")
            _require(_file_sha(root / path) == expected, f"predecessor hash changed: {path}")
            _require(
                hashlib.sha256(_git_bytes(root, commit, path.as_posix())).hexdigest() == expected,
                f"predecessor commit bytes changed: {path}",
            )
        receipt = json.loads((root / str(binding["receipt_path"])).read_text(encoding="utf-8"))
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
                "all_current_and_commit_hashes_match": True,
            }
        )
    return results


def _check(check_id: str, expression: sp.Expr, statement: str) -> dict[str, Any]:
    residual = sp.simplify(expression)
    _require(residual == 0, f"symbolic check failed: {check_id}: {residual}")
    return {"check_id": check_id, "passed": True, "statement": statement}


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    r, amplitude, exponent = sp.symbols("r A p", positive=True)
    x = sp.symbols("X", negative=True)
    exact_c = amplitude * r**exponent
    exact_p_xx = -amplitude * exponent * r ** (exponent - 1)
    exact_k = (1 + 2 * exponent) * exact_c
    exact_det = exact_c**3 * exact_k
    deep_c = exact_c.subs(exponent, sp.Rational(1, 2))
    deep_k = exact_k.subs(exponent, sp.Rational(1, 2))
    deep_p_xx = exact_p_xx.subs(exponent, sp.Rational(1, 2))

    mu, slope = sp.symbols("mu A", positive=True)
    x_real = sp.symbols("X", real=True)
    p_space = 2 * (mu**3 - (mu**2 - slope**2 * x_real) ** sp.Rational(3, 2)) / (3 * slope**2)
    c_space = sp.sqrt(mu**2 - slope**2 * x_real)
    c_space_x = sp.diff(c_space, x_real)
    k_space = sp.simplify(c_space + 2 * x_real * c_space_x)
    t_time = slope**2 * x_real / mu**2
    c_time = mu / sp.sqrt(1 + t_time)
    c_time_x = sp.diff(c_time, x_real)
    k_time = sp.simplify(c_time + 2 * x_real * c_time_x)
    t = sp.symbols("t", positive=True)
    relative_error = sp.sqrt(1 + 1 / t) - 1

    checks = [
        _check(
            "S01_EXACT_BRANCH_P_XX",
            sp.diff(amplitude * (-x) ** exponent, x)
            - (-amplitude * exponent * (-x) ** (exponent - 1)),
            "The X derivative of the exact spacelike branch has the frozen sign and power.",
        ),
        _check(
            "S02_EXACT_BRANCH_K",
            exact_c + 2 * (-r) * exact_p_xx - exact_k,
            "K=(1+2p)A(-X)^p on the exact branch.",
        ),
        _check(
            "S03_EXACT_BRANCH_EFFECTIVE_DETERMINANT",
            exact_c**3 * exact_k - (1 + 2 * exponent) * amplitude**4 * r ** (4 * exponent),
            "The effective mixed determinant has the exact fourth-power scaling.",
        ),
        _check(
            "S04_EXACT_BRANCH_C_ZERO_LIMIT",
            sp.limit(exact_c, r, 0, dir="+") - 0,
            "C tends to zero at the exact transition for p>0.",
        ),
        _check(
            "S05_EXACT_BRANCH_K_ZERO_LIMIT",
            sp.limit(exact_k, r, 0, dir="+") - 0,
            "K tends to zero at the exact transition for p>0.",
        ),
        _check(
            "S06_EXACT_BRANCH_DETERMINANT_ZERO_LIMIT",
            sp.limit(exact_det, r, 0, dir="+") - 0,
            "The effective determinant tends to zero at the exact transition.",
        ),
        _check(
            "S07_DEEP_AQUAL_P_XX_DIVERGENCE",
            sp.limit(1 / deep_p_xx, r, 0, dir="+") - 0,
            "The inverse of P_XX tends to zero, recording the p=1/2 divergence.",
        ),
        _check(
            "S08_DEEP_AQUAL_K_RATIO_TWO",
            deep_k / deep_c - 2,
            "The deep-AQUAL longitudinal coefficient ratio is two.",
        ),
        _check(
            "S09_DEEP_AQUAL_DETERMINANT_POWER",
            exact_det.subs(exponent, sp.Rational(1, 2)) - 2 * amplitude**4 * r**2,
            "The deep-AQUAL determinant vanishes quadratically in |X|.",
        ),
        _check(
            "S10_REGULATED_SPACELIKE_P_X",
            sp.diff(p_space, x_real) - c_space,
            "The regulated spacelike P differentiates to the frozen positive C.",
        ),
        _check(
            "S11_REGULATED_SPACELIKE_P_XX",
            c_space_x + slope**2 / (2 * c_space),
            "The regulated spacelike P_XX is finite and negative at the transition.",
        ),
        _check(
            "S12_REGULATED_SPACELIKE_K",
            k_space - (mu**2 - 2 * slope**2 * x_real) / c_space,
            "The regulated spacelike K is positive for X<=0.",
        ),
        _check(
            "S13_REGULATED_TIMELIKE_P_XX",
            c_time_x + slope**2 / (2 * mu) * (1 + t_time) ** (-sp.Rational(3, 2)),
            "The regulated timelike derivative has the transition-matching value.",
        ),
        _check(
            "S14_REGULATED_TIMELIKE_K",
            k_time - mu * (1 + t_time) ** (-sp.Rational(3, 2)),
            "The regulated timelike K is positive for finite X.",
        ),
        _check(
            "S15_TRANSITION_C_MATCH",
            sp.limit(c_space, x_real, 0, dir="-") - sp.limit(c_time, x_real, 0, dir="+"),
            "The two regulated C branches meet at mu.",
        ),
        _check(
            "S16_TRANSITION_P_XX_MATCH",
            sp.limit(c_space_x, x_real, 0, dir="-") - sp.limit(c_time_x, x_real, 0, dir="+"),
            "The regulated P_XX branches match at X=0.",
        ),
        _check(
            "S17_TRANSITION_K_MATCH",
            sp.limit(k_space, x_real, 0, dir="-") - sp.limit(k_time, x_real, 0, dir="+"),
            "The regulated K branches meet at mu.",
        ),
        _check(
            "S18_REGULATED_SPACELIKE_SPEED",
            ((mu**2 - 2 * slope**2 * x_real) / (mu**2 - slope**2 * x_real)).subs(
                x_real, -t * mu**2 / slope**2
            )
            - (1 + 2 * t) / (1 + t),
            "The spacelike longitudinal speed interpolates from one to two.",
        ),
        _check(
            "S19_REGULATED_TIMELIKE_SPEED",
            c_time / k_time - (1 + t_time),
            "The timelike sound speed squared is 1+A^2 X/mu^2.",
        ),
        _check(
            "S20_REGULATED_AQUAL_RELATIVE_ERROR",
            (c_space / (slope * sp.sqrt(-x_real)) - 1).subs(x_real, -t * mu**2 / slope**2)
            - relative_error,
            "The finite-floor AQUAL relative error is exact.",
        ),
        _check(
            "S21_REGULATED_LARGE_T_AQUAL_LIMIT",
            sp.limit(relative_error, t, sp.oo),
            "The regulated spacelike branch approaches deep-AQUAL at large t.",
        ),
        _check(
            "S22_TIMELIKE_C_POSITIVE_FINITE_DOMAIN",
            c_time * sp.sqrt(1 + t_time) - mu,
            "The timelike C is positive at every finite X.",
        ),
        _check(
            "S23_TIMELIKE_K_POSITIVE_FINITE_DOMAIN",
            k_time * (1 + t_time) ** sp.Rational(3, 2) - mu,
            "The timelike K is positive at every finite X.",
        ),
        _check(
            "S24_TIMELIKE_K_ZERO_ASYMPTOTE",
            sp.limit(k_time, x_real, sp.oo),
            "The timelike K lacks a positive lower bound on an unbounded domain.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS, "check order changed"
    )
    formulas = {
        "exact_C": "A*(-X)^p",
        "exact_K": "(1+2*p)*A*(-X)^p",
        "exact_effective_determinant": "(1+2*p)*A^4*(-X)^(4*p)",
        "regulated_spacelike_C": "sqrt(mu^2-A^2*X)",
        "regulated_spacelike_K": "(mu^2-2*A^2*X)/sqrt(mu^2-A^2*X)",
        "regulated_timelike_C": "mu/sqrt(1+A^2*X/mu^2)",
        "regulated_timelike_K": "mu/(1+A^2*X/mu^2)^(3/2)",
    }
    return checks, formulas


def numeric_checks(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    results: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["numeric_cases"]:
        x = float(case["X"])
        mu = float(case["mu"])
        slope = float(case["A"])
        if x <= 0:
            c_value = (mu * mu - slope * slope * x) ** 0.5
            k_value = (mu * mu - 2 * slope * slope * x) / c_value
            speed_squared = k_value / c_value
            relative_error = None
            if x < 0:
                relative_error = c_value / (slope * (-x) ** 0.5) - 1.0
        else:
            scaled = slope * slope * x / (mu * mu)
            c_value = mu / (1.0 + scaled) ** 0.5
            k_value = mu / (1.0 + scaled) ** 1.5
            speed_squared = c_value / k_value
            relative_error = None
        expected_error = case["expected_aqual_relative_error"]
        error_matches = expected_error is None and relative_error is None
        if expected_error is not None and relative_error is not None:
            error_matches = abs(relative_error - float(expected_error)) <= tolerance
        superluminal = speed_squared > 1.0 + tolerance
        passed = (
            (c_value > 0) is case["expected_C_positive"]
            and (k_value > 0) is case["expected_K_positive"]
            and superluminal is case["expected_superluminal_relative_conformal_cone"]
            and error_matches
        )
        _require(passed, f"numeric transition case failed: {case['case_id']}")
        results.append(
            {
                "case_id": case["case_id"],
                "C": c_value,
                "K": k_value,
                "aqual_relative_error": relative_error,
                "passed": True,
                "speed_squared": speed_squared,
                "superluminal_relative_conformal_cone": superluminal,
            }
        )
    return results


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    predecessors = validate_predecessors(root, config)
    symbolic, formulas = symbolic_checks()
    numeric = numeric_checks(config)
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-matter-lensing-deep-aqual-transition-tradeoff-receipt-1.0",
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
        "branch_and_scope_contract": config["branch_and_scope_contract"],
        "exact_transition_theorem": config["exact_transition_theorem"],
        "regulated_example_contract": config["regulated_example_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "numeric_cases": numeric,
            "formulas": formulas,
        },
        "counts": {
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic),
            "numeric_cases": len(numeric),
            "numeric_cases_passed": sum(item["passed"] for item in numeric),
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
            "The no-go is conditional on exact deep-AQUAL scaling continuing arbitrarily close to an X=0 crossing.",
            "The positive-floor example is chosen to expose the tradeoff; it is not derived from the shared empirical formula or promoted as the physical kinetic function.",
            "Positive finite-domain scalar coefficients do not establish the full metric-matter Hamiltonian, cutoff, causal compatibility, or an on-shell solution.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    expected = build_receipt(root)
    _require(dict(receipt) == expected, "stored transition receipt does not rebuild exactly")


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
    _require(path.is_file(), f"missing transition receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root)
    return {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "numeric_cases_passed": receipt["counts"]["numeric_cases_passed"],
        "CP11_4_complete": receipt["adjudication"]["CP11_4_complete"],
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
    result = check_receipt(root)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
