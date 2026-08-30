"""Map frozen shared formulas into a restricted minimal scalar kinetic theory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_formula_scalar_kinetic_reconstruction_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_shared_formula_scalar_kinetic_reconstruction.py"
)
TEST_PATH = Path("tests/test_gravity_shared_formula_scalar_kinetic_reconstruction.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-formula-scalar-kinetic-reconstruction-v1.json")

EXPECTED_CONFIG_FILE_SHA256 = "df1e6236612720f8a0a31b4780fa9beff2d1fa3fc00bc32bf0f969a022d4eb41"
EXPECTED_CONFIG_CONTENT_SHA256 = "c53030121ca93220a915241ba34335167a6765afcf6a191e85d0fa92e0263618"
DECISION = (
    "MINIMAL_FORMULA_TO_KINETIC_RECONSTRUCTION_DERIVED_ONLY_QUADRATURE_SOURCE_"
    "ONLY_CLASS_IS_SINGLE_VALUED_POSITIVE_BUT_CAUSAL_AND_ENDPOINT_GATES_FAIL_"
    "FULL_THEORY_BLOCKED"
)
SYMBOLIC_CHECK_IDS = (
    "S01_GENERAL_EXCESS",
    "S02_GENERAL_SPACELIKE_INVARIANT",
    "S03_GENERAL_KINETIC_COEFFICIENT",
    "S04_GENERAL_LONGITUDINAL_COEFFICIENT",
    "S05_QUADRATURE_EXCESS_POSITIVE_IDENTITY",
    "S06_QUADRATURE_EXCESS_DERIVATIVE_POSITIVE_IDENTITY",
    "S07_QUADRATURE_PARAMETRIC_INVERSE",
    "S08_QUADRATURE_C_MAP",
    "S09_QUADRATURE_K_MAP",
    "S10_QUADRATURE_SPEED_RATIO",
    "S11_QUADRATURE_FINITE_EXCESS_ENDPOINT",
    "S12_RAR_EXCESS",
    "S13_RAR_EXCESS_DERIVATIVE",
    "S14_RAR_K_MAP",
    "S15_RAR_TURNOVER_EQUATION",
    "S16_NEWTONIAN_ZERO_EXCESS",
)


class FormulaKineticReconstructionError(RuntimeError):
    """Raised when the frozen reconstruction package changes or a check fails."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FormulaKineticReconstructionError(message)


def load_config(root: Path) -> dict[str, Any]:
    path = root / CONFIG_PATH
    _require(path.is_file(), f"missing config: {path}")
    _require(
        _file_sha(path) == EXPECTED_CONFIG_FILE_SHA256,
        "formula kinetic reconstruction config file hash changed",
    )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormulaKineticReconstructionError(f"cannot load config: {path}") from exc
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        _sha(config) == EXPECTED_CONFIG_CONTENT_SHA256,
        "formula kinetic reconstruction config content changed",
    )
    _require(
        config["schema_version"]
        == "invariant-gravity-shared-formula-scalar-kinetic-reconstruction-config-1.0",
        "config schema changed",
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
        raise FormulaKineticReconstructionError("cannot validate predecessor git binding") from exc


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
            raise FormulaKineticReconstructionError(
                f"cannot load predecessor receipt: {receipt_path}"
            ) from exc
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


def _variables(node: Any) -> set[str]:
    variables: set[str] = set()
    if isinstance(node, Mapping):
        if "var" in node:
            variables.add(str(node["var"]))
        for value in node.values():
            variables.update(_variables(value))
    elif isinstance(node, list):
        for value in node:
            variables.update(_variables(value))
    return variables


def classify_registry(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    binding = next(
        item
        for item in config["predecessor_bindings"]
        if item["binding_id"] == "ben_synthetic_registry"
    )
    receipt_path = root / str(binding["receipt_path"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    registry = receipt["candidate_registry"]
    content = {key: value for key, value in registry.items() if key != "content_sha256"}
    expected_registry_sha = config["registry_structural_contract"]["registry_content_sha256"]
    _require(registry["content_sha256"] == expected_registry_sha, "registry seal changed")
    _require(_sha(content) == expected_registry_sha, "registry content does not reconstruct")
    classes = list(registry["equivalence_classes"])
    _require(len(classes) == 60, "canonical class count changed")

    signature_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for item in classes:
        variables = sorted(_variables(item["canonical_ast"]))
        signature = "+".join(variables)
        signature_counts[signature] += 1
        source_only = variables == ["x_source"]
        records.append(
            {
                "class_id": item["class_id"],
                "canonical_expression": item["canonical_expression"],
                "variables": variables,
                "minimal_single_scalar_reconstruction_eligible": source_only,
            }
        )
    expected_counts = config["registry_structural_contract"]["variable_signature_counts"]
    _require(dict(signature_counts) == expected_counts, "registry variable signatures changed")
    source_only_records = [
        item for item in records if item["minimal_single_scalar_reconstruction_eligible"]
    ]
    source_only_ids = [item["class_id"] for item in source_only_records]
    _require(
        source_only_ids == config["registry_structural_contract"]["source_only_class_ids"],
        "source-only class inventory changed",
    )
    expression_by_id = {
        item["class_id"]: item["canonical_expression"] for item in source_only_records
    }
    _require(
        expression_by_id["ben.a3da343620d23c63b16a"]
        == "sqrt_positive(((x_source*x_source)+x_source))",
        "quadrature class expression changed",
    )
    _require(
        expression_by_id["ben.cfe53a02a87ccf24af9c"]
        == "divide_safe(x_source,(1-exp_negative(sqrt_positive(x_source))))",
        "RAR-like class expression changed",
    )
    _require(
        expression_by_id["ben.f9a69717841da3b4e1cc"] == "x_source",
        "Newtonian class expression changed",
    )
    return {
        "registry_content_sha256": expected_registry_sha,
        "canonical_classes": len(records),
        "source_only_classes": len(source_only_records),
        "auxiliary_dependent_classes": len(records) - len(source_only_records),
        "variable_signature_counts": dict(signature_counts),
        "source_only_classes_detail": source_only_records,
        "all_classes_structurally_classified": True,
    }


def _check(check_id: str, expression: sp.Expr, statement: str) -> dict[str, Any]:
    residual = sp.simplify(expression)
    _require(residual == 0, f"symbolic check failed: {check_id}: {residual}")
    return {"check_id": check_id, "passed": True, "statement": statement}


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    a0, alpha, g_n, g_phi = sp.symbols("a0 alpha g_N g_phi", positive=True)
    x = sp.symbols("x", positive=True)
    y = (g_n + alpha * g_phi) / a0
    x_source = g_n / a0
    excess = alpha * g_phi / a0
    xbar = -(excess**2) / (2 * alpha**2)

    s_function = sp.Function("s")
    sx = s_function(x)
    general_xbar = -(sx**2) / (2 * alpha**2)
    general_c = alpha**2 * x / sx
    general_c_x = sp.diff(general_c, x) / sp.diff(general_xbar, x)
    general_k = sp.simplify(general_c + 2 * general_xbar * general_c_x)

    q = sp.sqrt(x**2 + x)
    quadrature_s = q - x
    quadrature_s_derivative = sp.diff(quadrature_s, x)
    t = sp.symbols("s", positive=True)
    x_of_t = t**2 / (1 - 2 * t)
    quadrature_c = alpha**2 * t / (1 - 2 * t)
    quadrature_k = 2 * alpha**2 * t * (1 - t) / (1 - 2 * t) ** 2

    r = sp.symbols("r", positive=True)
    rar_s = r**2 / (sp.exp(r) - 1)
    rar_y = r**2 / (1 - sp.exp(-r))
    rar_ds_dx = sp.diff(rar_s, r) / (2 * r)
    rar_ds_expected = (2 * (sp.exp(r) - 1) - r * sp.exp(r)) / (2 * (sp.exp(r) - 1) ** 2)
    rar_k = 2 * alpha**2 * (sp.exp(r) - 1) ** 2 / (2 * (sp.exp(r) - 1) - r * sp.exp(r))

    checks = [
        _check(
            "S01_GENERAL_EXCESS",
            y - x_source - excess,
            "The dimensionless excess equals alpha times the scalar force in a0 units.",
        ),
        _check(
            "S02_GENERAL_SPACELIKE_INVARIANT",
            -(g_phi**2) / (2 * a0**2) - xbar,
            "The scalar-gradient invariant is -s^2/(2 alpha^2).",
        ),
        _check(
            "S03_GENERAL_KINETIC_COEFFICIENT",
            alpha * g_n / g_phi - alpha**2 * x_source / excess,
            "The integrated spherical equation gives C=alpha^2 x/s.",
        ),
        _check(
            "S04_GENERAL_LONGITUDINAL_COEFFICIENT",
            general_k - alpha**2 / sp.diff(sx, x),
            "The longitudinal coefficient is alpha^2/(ds/dx).",
        ),
        _check(
            "S05_QUADRATURE_EXCESS_POSITIVE_IDENTITY",
            quadrature_s * (q + x) - x,
            "Quadrature excess equals x/(sqrt(x^2+x)+x), which is positive.",
        ),
        _check(
            "S06_QUADRATURE_EXCESS_DERIVATIVE_POSITIVE_IDENTITY",
            quadrature_s_derivative - 1 / (2 * q * (2 * x + 1 + 2 * q)),
            "Quadrature excess has a strictly positive derivative for x>0.",
        ),
        _check(
            "S07_QUADRATURE_PARAMETRIC_INVERSE",
            (x_of_t + t) ** 2 - x_of_t**2 - x_of_t,
            "The quadrature branch inverts to x=s^2/(1-2s).",
        ),
        _check(
            "S08_QUADRATURE_C_MAP",
            alpha**2 * x_of_t / t - quadrature_c,
            "The quadrature kinetic map is C=alpha^2 s/(1-2s).",
        ),
        _check(
            "S09_QUADRATURE_K_MAP",
            alpha**2 * sp.diff(x_of_t, t) - quadrature_k,
            "The quadrature longitudinal map is the inverse excess derivative.",
        ),
        _check(
            "S10_QUADRATURE_SPEED_RATIO",
            quadrature_k / quadrature_c - 2 * (1 - t) / (1 - 2 * t),
            "The quadrature scalar cone ratio exceeds two and diverges at s=1/2.",
        ),
        _check(
            "S11_QUADRATURE_FINITE_EXCESS_ENDPOINT",
            sp.limit(-(t**2) / (2 * alpha**2), t, sp.Rational(1, 2), dir="-") + 1 / (8 * alpha**2),
            "The x-to-infinity limit occurs at finite Xbar=-1/(8 alpha^2).",
        ),
        _check(
            "S12_RAR_EXCESS",
            rar_y - r**2 - rar_s,
            "The RAR-like excess is r^2/(exp(r)-1).",
        ),
        _check(
            "S13_RAR_EXCESS_DERIVATIVE",
            rar_ds_dx - rar_ds_expected,
            "The RAR-like excess derivative has the frozen turnover numerator.",
        ),
        _check(
            "S14_RAR_K_MAP",
            alpha**2 / rar_ds_dx - rar_k,
            "The RAR-like K changes sign with the excess derivative.",
        ),
        _check(
            "S15_RAR_TURNOVER_EQUATION",
            (2 * (sp.exp(r) - 1) - r * sp.exp(r)) / sp.exp(r) - (2 * (1 - sp.exp(-r)) - r),
            "The positive turnover solves 2(1-exp(-r))=r.",
        ),
        _check(
            "S16_NEWTONIAN_ZERO_EXCESS",
            x - x,
            "The Newtonian control has zero scalar excess.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic check order changed",
    )
    formulas = {
        "general_Xbar": "-s^2/(2*alpha^2)",
        "general_C": "alpha^2*x/s",
        "general_K_source_only": "alpha^2/(ds/dx)",
        "quadrature_x_of_s": "s^2/(1-2*s)",
        "quadrature_C": "alpha^2*s/(1-2*s)",
        "quadrature_K": "2*alpha^2*s*(1-s)/(1-2*s)^2",
        "rar_excess": "r^2/(exp(r)-1)",
    }
    return checks, formulas


def _bisect(function: Any, low: float, high: float) -> float:
    f_low = float(function(low))
    f_high = float(function(high))
    _require(math.isfinite(f_low) and math.isfinite(f_high), "nonfinite root bracket")
    _require(f_low * f_high < 0.0, "root bracket does not change sign")
    for _ in range(200):
        middle = (low + high) / 2.0
        f_middle = float(function(middle))
        if f_middle == 0.0:
            return middle
        if f_low * f_middle < 0.0:
            high = middle
        else:
            low = middle
            f_low = f_middle
    return (low + high) / 2.0


def numeric_checks(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["machine_check_contract"]
    tolerance = float(contract["numeric_tolerance"])
    quadrature: list[dict[str, Any]] = []
    for x in map(float, contract["quadrature_source_probes"]):
        q = math.sqrt(x * x + x)
        s = x / (q + x)
        derivative = 1.0 / (2.0 * q * (2.0 * x + 1.0 + 2.0 * q))
        c = x / s
        k = 1.0 / derivative
        speed_ratio = k / c
        inverse_x = s * s / (1.0 - 2.0 * s)
        relative_error = abs(inverse_x - x) / max(x, 1.0)
        defining_relation_residual = abs((x + s) ** 2 - x * x - x) / max(x * x + x, 1.0)
        _require(s > 0.0 and s < 0.5, "quadrature excess left its frozen interval")
        _require(derivative > 0.0 and c > 0.0 and k > 0.0, "quadrature health failed")
        _require(speed_ratio > 2.0, "quadrature cone obstruction disappeared")
        _require(defining_relation_residual <= tolerance, "quadrature relation replay failed")
        quadrature.append(
            {
                "x": x,
                "s": s,
                "C_over_alpha_squared": c,
                "K_over_alpha_squared": k,
                "K_over_C": speed_ratio,
                "inverse_relative_error": relative_error,
                "defining_relation_scaled_residual": defining_relation_residual,
                "passed": True,
            }
        )

    turnover_bracket = list(map(float, contract["rar_turnover_bracket"]))
    turnover_r = _bisect(
        lambda value: 2.0 * (1.0 - math.exp(-value)) - value,
        turnover_bracket[0],
        turnover_bracket[1],
    )
    turnover_x = turnover_r * turnover_r
    turnover_s = turnover_x / math.expm1(turnover_r)
    expected_turnover = contract["rar_turnover_expected"]
    for actual, key in (
        (turnover_r, "r"),
        (turnover_x, "x"),
        (turnover_s, "s"),
    ):
        _require(abs(actual - float(expected_turnover[key])) <= tolerance, "RAR turnover changed")

    target_s = float(contract["rar_same_excess"])
    witness: list[dict[str, Any]] = []
    for bracket, expected in zip(
        contract["rar_same_excess_r_brackets"],
        contract["rar_same_excess_expected"],
        strict=True,
    ):
        low, high = map(float, bracket)
        r_value = _bisect(
            lambda value: value * value / math.expm1(value) - target_s,
            low,
            high,
        )
        x_value = r_value * r_value
        c_value = x_value / target_s
        _require(abs(x_value - float(expected["x"])) <= tolerance, "RAR witness x changed")
        _require(
            abs(c_value - float(expected["C_over_alpha_squared"])) <= tolerance,
            "RAR witness kinetic coefficient changed",
        )
        witness.append(
            {
                "r": r_value,
                "x": x_value,
                "s": target_s,
                "C_over_alpha_squared": c_value,
                "passed": True,
            }
        )
    _require(
        abs(witness[0]["C_over_alpha_squared"] - witness[1]["C_over_alpha_squared"]) > tolerance,
        "RAR multivalued witness collapsed",
    )
    derivative_below = (
        2.0 * math.expm1(turnover_r / 2.0) - (turnover_r / 2.0) * math.exp(turnover_r / 2.0)
    ) / (2.0 * math.expm1(turnover_r / 2.0) ** 2)
    derivative_above = (
        2.0 * math.expm1(2.0 * turnover_r) - (2.0 * turnover_r) * math.exp(2.0 * turnover_r)
    ) / (2.0 * math.expm1(2.0 * turnover_r) ** 2)
    _require(derivative_below > 0.0 and derivative_above < 0.0, "RAR K sign change failed")
    return {
        "quadrature_probes": quadrature,
        "rar_turnover": {
            "r": turnover_r,
            "x": turnover_x,
            "s": turnover_s,
            "passed": True,
        },
        "rar_same_excess_witness": witness,
        "rar_excess_derivative_below_turnover": derivative_below,
        "rar_excess_derivative_above_turnover": derivative_above,
        "rar_multivalued_and_K_sign_change_confirmed": True,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root)
    predecessors = validate_predecessors(root, config)
    registry = classify_registry(root, config)
    symbolic, formulas = symbolic_checks()
    numeric = numeric_checks(config)
    receipt: dict[str, Any] = {
        "schema_version": (
            "invariant-gravity-shared-formula-scalar-kinetic-reconstruction-receipt-1.0"
        ),
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
        "minimal_spherical_mapping_contract": config["minimal_spherical_mapping_contract"],
        "registry_structural_result": registry,
        "source_only_adjudication_contract": config["source_only_adjudication_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "symbolic_formulas": formulas,
            "numeric_checks": numeric,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "canonical_formula_classes": registry["canonical_classes"],
            "source_only_formula_classes": registry["source_only_classes"],
            "auxiliary_dependent_formula_classes": registry["auxiliary_dependent_classes"],
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic),
            "quadrature_numeric_probes": len(numeric["quadrature_probes"]),
            "rar_same_excess_witness_points": len(numeric["rar_same_excess_witness"]),
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
            "The map is derived only in a static spherical conformal-force branch with a regular integrated source equation.",
            "The registry classification is structural and target-independent; it neither scores nor prunes any empirical formula class.",
            "Auxiliary-dependent formulas require covariant dynamics for every auxiliary predictor before they can define a field theory.",
            "Local positive scalar coefficients do not establish a causal globally regular action, metric constraints, lensing, or observational support.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    expected = build_receipt(root)
    _require(dict(receipt) == expected, "stored formula kinetic receipt does not rebuild exactly")


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
    _require(path.is_file(), f"missing formula kinetic receipt: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root)
    return {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "canonical_formula_classes": receipt["counts"]["canonical_formula_classes"],
        "source_only_formula_classes": receipt["counts"]["source_only_formula_classes"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "full_covariant_formula_bridge_derived": receipt["adjudication"][
            "full_covariant_formula_bridge_derived"
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
