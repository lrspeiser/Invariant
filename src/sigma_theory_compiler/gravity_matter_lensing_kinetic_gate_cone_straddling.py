"""Exact metric-cone straddling theorem for an active two-field kinetic gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_cone_straddling_v1.json")
MODULE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_kinetic_gate_cone_straddling.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_kinetic_gate_cone_straddling.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-kinetic-gate-cone-straddling-v1.json")
POLICY_PATH = Path("docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-cone-straddling-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-cone-straddling-receipt-1.0"
DECISION = "EXACT_ACTIVE_TIMELIKE_METRIC_CONE_STRADDLING_ESTABLISHED_CAUSALITY_AND_NOVELTY_OPEN"

EXPECTED_CONFIG_RAW_SHA256 = "8cc0dff83390723cbd8f9a15cff44ba5df0d0e74642e6643e5cd1eadeac03d18"
EXPECTED_MODULE_SEMANTIC_SHA256 = "dfb31af290d936e469ddef197af10745171185276ed22529eb2e0883a5605ce4"
EXPECTED_TEST_RAW_SHA256 = "fdbbdbf28437cdb1c888c70b45c0588b2b630524b8bb6d7dc3f1a23bee5594fc"


class KineticGateConeStraddlingError(RuntimeError):
    """Raised when a theorem, source, binding, or publication gate fails."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return _sha256_bytes(_canonical_bytes(body))


def _module_semantic_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'EXPECTED_MODULE_SEMANTIC_SHA256 = (?:"[0-9a-f]{64}"|"__MODULE_SEMANTIC_SHA256__")',
        'EXPECTED_MODULE_SEMANTIC_SHA256 = "<SELF>"',
        text,
        count=1,
    )
    return _sha256_bytes(normalized.encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KineticGateConeStraddlingError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise KineticGateConeStraddlingError("JSON artifact must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KineticGateConeStraddlingError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise KineticGateConeStraddlingError("bound Git artifact unavailable")
    return completed.stdout


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "package",
            "predecessor",
            "source_or_paper_gate",
            "paper_anchors",
            "theorem",
            "numeric_cases",
            "required_checks",
            "claim_boundary",
            "access_ledger",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-kinetic-gate-cone-straddling-v1",
        "analysis identity changed",
    )
    _require(
        config["status"] == "FROZEN_THEORY_ONLY_PRIMARY_PAPER_ANCHORED_CONE_STRADDLING_ANALYSIS",
        "config status changed",
    )
    _require(
        config["package"]
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
        },
        "package paths changed",
    )
    gate = config["source_or_paper_gate"]
    _require(
        gate
        == {
            "project_admission_policy_path": POLICY_PATH.as_posix(),
            "project_admission_policy_sha256": (
                "b3518291131b0ece9f05c966e55b40c40e549d5c27e87a2a128b70ad76a864fe"
            ),
            "artifact_type": "THEORY_ONLY",
            "primary_paper_and_analytic_benchmark_required": True,
            "real_observational_source_required_before_observational_scoring": True,
            "observational_scoring_authorized": False,
            "missing_paper_action": "SOURCE_BLOCKED",
            "failed_analytic_benchmark_action": "BENCHMARK_FAILED",
            "rule": (
                "A theorem-only result may advance on primary papers plus independent exact analytic "
                "and numerical checks. Any later observational builder must separately bind a suitable "
                "real public source and response dataset before scoring."
            ),
        },
        "source-or-paper gate changed",
    )
    papers = config["paper_anchors"]
    _require(
        len(papers) == 4
        and {item["arxiv"] for item in papers}
        == {"0806.0336", "0708.0561", "gr-qc/0607055", "2603.13986v2"}
        and not any(item["exact_straddling_theorem_found"] for item in papers),
        "primary-paper inventory changed",
    )
    theorem = config["theorem"]
    _require(theorem["exact_identity"] == "det(K-G)=-4 X Y Z_X^2<0", "identity changed")
    _require(
        theorem["conclusion"]
        == "Every healthy active background in this architecture has c_-^2<1<c_+^2 relative to the g-null cone.",
        "theorem conclusion changed",
    )
    _require(
        theorem["escape_cases"]
        == [
            "X=0",
            "Y=0",
            "Z_X=0",
            "a degenerate or non-positive K or G outside the theorem hypotheses",
            "a different action architecture with additional derivative operators that change both K and G",
        ],
        "escape cases changed",
    )
    _require(len(config["numeric_cases"]) == 3, "numeric case inventory changed")
    _require(
        config["required_checks"]
        == [
            "C01_CONFIG_AND_PACKAGE_SEALS",
            "C02_PREDECESSOR_COMMIT_AND_BYTES",
            "C03_GENERAL_MULTIFIELD_MATRIX_MAP",
            "C04_EXACT_K_MINUS_G_DETERMINANT",
            "C05_CHARACTERISTIC_POLYNOMIAL_SIGNS",
            "C06_POSITIVE_GENERALIZED_EIGENVALUES",
            "C07_STRICT_CONE_STRADDLING",
            "C08_P0_INDEPENDENCE",
            "C09_LINEAR_FIELD_REDEFINITION_INVARIANCE",
            "C10_ESCAPE_CASES",
            "C11_PRIMARY_PAPER_SCOPE",
            "C12_CLAIM_CEILING",
            "C13_ZERO_OBSERVATIONAL_ACCESS",
        ],
        "required checks changed",
    )
    _require(
        config["claim_boundary"]
        == {
            "exact_metric_cone_straddling_theorem_established": True,
            "unconditional_causality_violation_established": False,
            "global_strong_hyperbolicity_established": False,
            "full_action_no_go_established": False,
            "historical_novelty_established": False,
            "observational_support": False,
            "healthy_modified_gravity_model_established": False,
            "publication_ready": False,
        },
        "claim boundary changed",
    )
    _require(
        config["access_ledger"]
        == {
            "observational_files_opened": 0,
            "observational_rows_read": 0,
            "scores_computed": 0,
            "network_calls_by_builder": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "access ledger changed",
    )


def load_config(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / CONFIG_PATH).resolve()
    _require(path == (base / CONFIG_PATH).resolve(), "config path changed")
    _require(_sha256_file(path) == EXPECTED_CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_local_integrity(base: Path) -> dict[str, str]:
    module = (base / MODULE_PATH).resolve()
    test = (base / TEST_PATH).resolve()
    policy = (base / POLICY_PATH).resolve()
    _require(module == Path(__file__).resolve(), "module path changed")
    module_semantic = _module_semantic_sha256(module)
    _require(module_semantic == EXPECTED_MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(_sha256_file(test) == EXPECTED_TEST_RAW_SHA256, "test bytes changed")
    _require(
        _sha256_file(policy) == "b3518291131b0ece9f05c966e55b40c40e549d5c27e87a2a128b70ad76a864fe",
        "builder/solver admission policy changed",
    )
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module),
        "module_semantic_sha256": module_semantic,
        "test_raw_sha256": _sha256_file(test),
        "admission_policy_raw_sha256": _sha256_file(policy),
    }


def _validate_predecessor(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    predecessor = config["predecessor"]
    commit = predecessor["commit"]
    output = {"commit": commit}
    for role in ("config", "module", "test", "receipt"):
        relative = predecessor[f"{role}_path"]
        expected = predecessor[f"{role}_sha256"]
        _require(_sha256_file(base / relative) == expected, "predecessor worktree bytes changed")
        _require(
            _sha256_bytes(_git_show(base, commit, relative)) == expected,
            "predecessor commit bytes changed",
        )
        output[f"{role}_sha256"] = expected
    receipt = _read_json(base / predecessor["receipt_path"])
    _require(
        receipt.get("content_sha256") == predecessor["receipt_content_sha256"],
        "predecessor receipt content changed",
    )
    output["receipt_content_sha256"] = predecessor["receipt_content_sha256"]
    return output


def symbolic_checks() -> dict[str, bool]:
    x, y = sp.symbols("X Y", positive=True)
    c, z, zx, zxx, p0x, p0xx, speed = sp.symbols("C Z Z_X Z_XX P0_X P0_XX c2", real=True)
    p0 = sp.Function("P0")(x)
    z_function = sp.Function("Z")(x)
    lagrangian = p0 + z_function * y
    p_x = sp.diff(lagrangian, x)
    p_y = sp.diff(lagrangian, y)
    hessian_map = sp.Matrix(
        [
            [p_x + 2 * x * sp.diff(p_x, x), 2 * sp.sqrt(x * y) * sp.diff(lagrangian, x, y)],
            [2 * sp.sqrt(x * y) * sp.diff(lagrangian, x, y), p_y + 2 * y * sp.diff(p_y, y)],
        ]
    )
    expected_map = sp.Matrix(
        [
            [
                sp.diff(p0, x)
                + y * sp.diff(z_function, x)
                + 2 * x * (sp.diff(p0, x, 2) + y * sp.diff(z_function, x, 2)),
                2 * sp.sqrt(x * y) * sp.diff(z_function, x),
            ],
            [2 * sp.sqrt(x * y) * sp.diff(z_function, x), z_function],
        ]
    )
    k = sp.Matrix(
        [[c + 2 * x * (p0xx + y * zxx), 2 * zx * sp.sqrt(x * y)], [2 * zx * sp.sqrt(x * y), z]]
    )
    g = sp.diag(c, z)
    det_difference = sp.factor((k - g).det())
    polynomial = sp.expand((g - speed * k).det())
    checks = {
        "matrix_map": all(sp.simplify(value) == 0 for value in (hessian_map - expected_map)),
        "det_k_minus_g": sp.simplify(det_difference + 4 * x * y * zx**2) == 0,
        "characteristic_at_zero": sp.simplify(polynomial.subs(speed, 0) - c * z) == 0,
        "characteristic_at_one": sp.simplify(polynomial.subs(speed, 1) + 4 * x * y * zx**2) == 0,
        "positive_leading_coefficient": sp.simplify(sp.Poly(polynomial, speed).LC() - k.det()) == 0,
        "p0_independent_straddling_identity": not det_difference.has(p0x, p0xx),
        "escape_x_zero": sp.simplify(det_difference.subs(x, 0)) == 0,
        "escape_y_zero": sp.simplify(det_difference.subs(y, 0)) == 0,
        "escape_zx_zero": sp.simplify(det_difference.subs(zx, 0)) == 0,
    }
    _require(all(checks.values()), "symbolic theorem benchmark failed")
    return checks


def _gate_derivatives(case: Mapping[str, Any]) -> tuple[float, float, float]:
    x = float(case["X"])
    beta = float(case["beta"])
    power = float(case["p"])
    u = beta * x * x
    if case["family"] == "SHIFTED_POWER":
        z = (1.0 + u) ** power
        zx = 2.0 * beta * x * power * (1.0 + u) ** (power - 1.0)
        zxx = 2.0 * beta * power * (1.0 + u) ** (
            power - 1.0
        ) + 4.0 * beta * beta * x * x * power * (power - 1.0) * (1.0 + u) ** (power - 2.0)
    elif case["family"] == "EXPONENTIAL":
        alpha = float(case["alpha"])
        z = math.exp(alpha * u**power)
        ux = 2.0 * beta * x
        uxx = 2.0 * beta
        wx = alpha * power * u ** (power - 1.0) * ux
        wxx = (
            alpha
            * power
            * ((power - 1.0) * u ** (power - 2.0) * ux * ux + u ** (power - 1.0) * uxx)
        )
        zx = z * wx
        zxx = z * (wxx + wx * wx)
    else:
        raise KineticGateConeStraddlingError("unknown gate family")
    return z, zx, zxx


def _generalized_speeds(k: np.ndarray, g: np.ndarray) -> np.ndarray:
    values = np.linalg.eigvals(np.linalg.solve(k, g))
    _require(np.max(np.abs(values.imag)) < 1.0e-12, "complex generalized speed")
    return np.sort(values.real)


def numeric_suite(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    transform = np.array([[1.0, 0.3], [-0.2, 1.1]], dtype=float)
    for case in config["numeric_cases"]:
        x = float(case["X"])
        y = float(case["Y"])
        lam = float(case["lambda"])
        z, zx, zxx = _gate_derivatives(case)
        p0x = 1.0 + 2.0 * lam * x
        p0xx = 2.0 * lam
        c = p0x + y * zx
        k = np.array(
            [
                [c + 2.0 * x * (p0xx + y * zxx), 2.0 * zx * math.sqrt(x * y)],
                [2.0 * zx * math.sqrt(x * y), z],
            ],
            dtype=float,
        )
        g = np.diag([c, z])
        speeds = _generalized_speeds(k, g)
        transformed = _generalized_speeds(transform.T @ k @ transform, transform.T @ g @ transform)
        identity = -4.0 * x * y * zx * zx
        measured = float(np.linalg.det(k - g))
        tolerance = 2.0e-12 * max(1.0, abs(identity))
        passed = (
            np.min(np.linalg.eigvalsh(k)) > 0.0
            and np.min(np.linalg.eigvalsh(g)) > 0.0
            and 0.0 < speeds[0] < 1.0 < speeds[1]
            and abs(measured - identity) <= tolerance
            and np.allclose(speeds, transformed, rtol=2.0e-12, atol=2.0e-12)
        )
        _require(bool(passed), f"numeric theorem benchmark failed: {case['case_id']}")
        records.append(
            {
                "case_id": case["case_id"],
                "family": case["family"],
                "K_eigenvalues": [format(float(value), ".17g") for value in np.linalg.eigvalsh(k)],
                "G_eigenvalues": [format(float(value), ".17g") for value in np.linalg.eigvalsh(g)],
                "generalized_speed_squared": [format(float(value), ".17g") for value in speeds],
                "det_K_minus_G": format(measured, ".17g"),
                "exact_identity_value": format(identity, ".17g"),
                "field_redefinition_speed_squared": [
                    format(float(value), ".17g") for value in transformed
                ],
                "passed": True,
            }
        )
    return records


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _validate_local_integrity(base)
    predecessor = _validate_predecessor(base, config)
    symbolic = symbolic_checks()
    numeric = numeric_suite(config)
    checks = {
        "C01_CONFIG_AND_PACKAGE_SEALS": True,
        "C02_PREDECESSOR_COMMIT_AND_BYTES": True,
        "C03_GENERAL_MULTIFIELD_MATRIX_MAP": symbolic["matrix_map"],
        "C04_EXACT_K_MINUS_G_DETERMINANT": symbolic["det_k_minus_g"],
        "C05_CHARACTERISTIC_POLYNOMIAL_SIGNS": all(
            symbolic[key]
            for key in (
                "characteristic_at_zero",
                "characteristic_at_one",
                "positive_leading_coefficient",
            )
        ),
        "C06_POSITIVE_GENERALIZED_EIGENVALUES": all(
            min(float(value) for value in item["generalized_speed_squared"]) > 0.0
            for item in numeric
        ),
        "C07_STRICT_CONE_STRADDLING": all(
            float(item["generalized_speed_squared"][0])
            < 1.0
            < float(item["generalized_speed_squared"][1])
            for item in numeric
        ),
        "C08_P0_INDEPENDENCE": symbolic["p0_independent_straddling_identity"],
        "C09_LINEAR_FIELD_REDEFINITION_INVARIANCE": all(item["passed"] for item in numeric),
        "C10_ESCAPE_CASES": all(
            symbolic[key] for key in ("escape_x_zero", "escape_y_zero", "escape_zx_zero")
        ),
        "C11_PRIMARY_PAPER_SCOPE": len(config["paper_anchors"]) == 4
        and not any(item["exact_straddling_theorem_found"] for item in config["paper_anchors"]),
        "C12_CLAIM_CEILING": config["claim_boundary"][
            "exact_metric_cone_straddling_theorem_established"
        ]
        and not any(
            value
            for key, value in config["claim_boundary"].items()
            if key != "exact_metric_cone_straddling_theorem_established"
        ),
        "C13_ZERO_OBSERVATIONAL_ACCESS": all(
            value == 0 for value in config["access_ledger"].values()
        ),
    }
    _require(list(checks) == config["required_checks"], "check inventory changed")
    _require(all(checks.values()), "required theorem check failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "PASS_EXACT_THEORY_ONLY_PRIMARY_PAPER_ANCHORED_METRIC_CONE_STRADDLING",
        "decision": DECISION,
        "package_bindings": local,
        "predecessor_binding": predecessor,
        "source_or_paper_gate": config["source_or_paper_gate"],
        "paper_anchors": config["paper_anchors"],
        "theorem": config["theorem"],
        "symbolic_checks": symbolic,
        "numeric_benchmarks": numeric,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
        "limitations": [
            "Metric-cone superluminality is not by itself an unconditional causal paradox.",
            "The result is local, constant-background, scalar-block, and architecture-specific.",
            "Metric constraints, global hyperbolicity, cutoff, on-shell phenomenology, and observations remain open.",
            "Historical novelty requires independent expert literature review beyond this bounded primary-paper set.",
            "Any future 3D or observational implementation must separately bind a public source dataset, its primary paper, and an independent solver benchmark before scoring.",
        ],
    }
    receipt["content_sha256"] = _self_hash(receipt)
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(
        receipt,
        {
            "schema_version",
            "analysis_id",
            "status",
            "decision",
            "package_bindings",
            "predecessor_binding",
            "source_or_paper_gate",
            "paper_anchors",
            "theorem",
            "symbolic_checks",
            "numeric_benchmarks",
            "checks",
            "checks_passed",
            "checks_total",
            "claim_boundary",
            "access_ledger",
            "limitations",
            "content_sha256",
        },
        "receipt",
    )
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["analysis_id"] == config["analysis_id"], "receipt identity changed")
    _require(receipt["decision"] == DECISION, "decision changed")
    _require(
        receipt["source_or_paper_gate"] == config["source_or_paper_gate"], "source gate changed"
    )
    _require(receipt["paper_anchors"] == config["paper_anchors"], "paper anchors changed")
    _require(receipt["theorem"] == config["theorem"], "theorem changed")
    _require(receipt["claim_boundary"] == config["claim_boundary"], "claims changed")
    _require(receipt["access_ledger"] == config["access_ledger"], "access ledger changed")
    _require(receipt["checks_total"] == 13 and receipt["checks_passed"] == 13, "checks incomplete")
    _require(receipt["content_sha256"] == _self_hash(receipt), "receipt self-hash invalid")


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return "EXISTING_IDENTICAL"
        raise KineticGateConeStraddlingError("refusing to replace nonidentical receipt")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_path, path)
        except FileExistsError as exc:
            raise KineticGateConeStraddlingError("receipt publication race") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    return _atomic_no_clobber(base / OUTPUT_PATH, build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    path = (base / OUTPUT_PATH).resolve()
    _require(path == (base / OUTPUT_PATH).resolve(), "receipt path changed")
    stored = _read_json(path)
    expected = build_receipt(base)
    _require(stored == expected, "stored receipt does not match deterministic rebuild")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("write", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "write":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID")
    else:
        print(check_receipt()["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
