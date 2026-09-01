"""Local common-time and frozen-coefficient hyperbolicity for kinetic-gate cones."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_kinetic_gate_common_time_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_kinetic_gate_common_time.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_kinetic_gate_common_time.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-kinetic-gate-common-time-v1.json")
POLICY_PATH = Path("docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-common-time-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-kinetic-gate-common-time-receipt-1.0"
DECISION = (
    "PASS_LOCAL_COMMON_TIME_AND_CONSTANT_COEFFICIENT_SCALAR_STRONG_HYPERBOLICITY_"
    "CONE_STRADDLING_RETAINED_GLOBAL_CAUSALITY_OPEN"
)

EXPECTED_CONFIG_RAW_SHA256 = "63d44d59bbbe4a1baf6341721eff4381340c636e139ec666e623546e4a5630f5"
EXPECTED_MODULE_SEMANTIC_SHA256 = "dc326078710ffa88535c37f37623b062b4a647edb6df366f8655241d5dde8bad"
EXPECTED_TEST_RAW_SHA256 = "832fc086bc3e8d0186ba4ca17ecf05bd22d8278fac53d220e3a44b891baa3eb8"


class KineticGateCommonTimeError(RuntimeError):
    """Raised when a binding or causal-interpretation gate fails."""


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
        raise KineticGateCommonTimeError("invalid JSON artifact") from exc
    if not isinstance(value, dict):
        raise KineticGateCommonTimeError("JSON artifact must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise KineticGateCommonTimeError(message)


def _git_show(root: Path, commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise KineticGateCommonTimeError("committed predecessor unavailable")
    return completed.stdout


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config.get("schema_version") == CONFIG_SCHEMA, "config schema changed")
    _require(
        config.get("analysis_id") == "gravity-matter-lensing-kinetic-gate-common-time-v1",
        "analysis identity changed",
    )
    _require(
        config.get("package")
        == {
            "module_path": MODULE_PATH.as_posix(),
            "test_path": TEST_PATH.as_posix(),
            "output_path": OUTPUT_PATH.as_posix(),
        },
        "package paths changed",
    )
    policy = config.get("admission_policy")
    _require(
        policy
        == {
            "path": POLICY_PATH.as_posix(),
            "sha256": "b3518291131b0ece9f05c966e55b40c40e549d5c27e87a2a128b70ad76a864fe",
            "artifact_type": "THEORY_ONLY",
            "paper_and_analytic_benchmark_required": True,
            "observational_scoring_authorized": False,
        },
        "admission policy changed",
    )
    bindings = config.get("bindings")
    _require(
        isinstance(bindings, list)
        and [item["id"] for item in bindings]
        == ["EXTERNAL_METRIC_PRINCIPAL_SYMBOL", "CONE_STRADDLING_THEOREM"],
        "binding inventory changed",
    )
    _require(
        bindings[0]["state"] == "COMMITTED"
        and bindings[0]["commit"] == "d1e2491baa9c4ecb408fd28424787efce66e5b9b"
        and bindings[1]["state"] == "MUTATION_FROZEN_UNCOMMITTED"
        and bindings[1]["commit"] is None,
        "binding states changed",
    )
    _require(len(config.get("primary_paper_anchors", [])) == 3, "paper inventory changed")
    theorem = config.get("theorem")
    _require(
        theorem["common_time"].startswith("The covector dt evaluates to -K")
        and theorem["constant_coefficient_result"].endswith(
            "hence it is strongly hyperbolic at frozen constant coefficients."
        )
        and theorem["strict_limit"].startswith("This does not establish a global time function"),
        "theorem scope changed",
    )
    _require(len(config.get("numeric_cases", [])) == 2, "numeric inventory changed")
    _require(
        config.get("claim_boundary")
        == {
            "constant_coefficient_scalar_block_strongly_hyperbolic": True,
            "local_common_time_covector_with_metric": True,
            "cone_straddling_retained": True,
            "local_causal_paradox_established": False,
            "global_time_function_established": False,
            "variable_coefficient_strong_hyperbolicity": False,
            "full_metric_scalar_matter_system_healthy": False,
            "ultraviolet_completion": False,
            "observational_support": False,
            "publication_ready": False,
        },
        "claim boundary changed",
    )
    _require(
        config.get("access_ledger")
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
    _require(_sha256_file(path) == EXPECTED_CONFIG_RAW_SHA256, "config bytes changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_local_integrity(base: Path, config: Mapping[str, Any]) -> dict[str, str]:
    module = (base / MODULE_PATH).resolve()
    test = (base / TEST_PATH).resolve()
    policy = (base / POLICY_PATH).resolve()
    _require(module == Path(__file__).resolve(), "module path changed")
    semantic = _module_semantic_sha256(module)
    _require(semantic == EXPECTED_MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(_sha256_file(test) == EXPECTED_TEST_RAW_SHA256, "test bytes changed")
    _require(_sha256_file(policy) == config["admission_policy"]["sha256"], "policy bytes changed")
    return {
        "config_raw_sha256": _sha256_file(base / CONFIG_PATH),
        "module_raw_sha256": _sha256_file(module),
        "module_semantic_sha256": semantic,
        "test_raw_sha256": _sha256_file(test),
        "policy_raw_sha256": _sha256_file(policy),
    }


def _validate_bindings(base: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for binding in config["bindings"]:
        for role in ("config", "module", "test", "receipt"):
            relative = binding[f"{role}_path"]
            expected = binding[f"{role}_sha256"]
            _require(
                _sha256_file(base / relative) == expected, "predecessor worktree bytes changed"
            )
            if binding["state"] == "COMMITTED":
                _require(
                    _sha256_bytes(_git_show(base, binding["commit"], relative)) == expected,
                    "predecessor commit bytes changed",
                )
        receipt = _read_json(base / binding["receipt_path"])
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            "predecessor receipt content changed",
        )
        receipts[binding["id"]] = receipt
    return receipts


def symbolic_checks() -> dict[str, bool]:
    c1, c2, omega, k = sp.symbols("c1 c2 omega k", positive=True)
    diagonal_k = sp.eye(2)
    diagonal_g = sp.diag(c1**2, c2**2)
    principal = -(omega**2) * diagonal_k + k**2 * diagonal_g
    energy_matrix = sp.diag(1, 1, c1**2, c2**2)
    first_order_symbol = sp.diag(
        sp.Matrix([[0, c1], [c1, 0]]),
        sp.Matrix([[0, c2], [c2, 0]]),
    )
    checks = {
        "decoupled_characteristic_polynomial": sp.simplify(
            principal.det() - (omega**2 - c1**2 * k**2) * (omega**2 - c2**2 * k**2)
        )
        == 0,
        "positive_energy_matrix": energy_matrix.is_positive_definite is True,
        "symmetric_first_order_symbol": first_order_symbol == first_order_symbol.T,
        "complete_real_first_order_spectrum": first_order_symbol.eigenvals()
        == {-c1: 1, c1: 1, -c2: 1, c2: 1},
        "dt_scalar_form_negative": (-diagonal_k).is_negative_definite is True,
    }
    _require(all(checks.values()), "symbolic common-time benchmark failed")
    return checks


def _inverse_square_root(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh(matrix)
    _require(np.min(values) > 0.0, "matrix not positive definite")
    return vectors @ np.diag(values**-0.5) @ vectors.T


def _generalized_speeds(k_matrix: np.ndarray, g_matrix: np.ndarray) -> np.ndarray:
    inverse_root = _inverse_square_root(k_matrix)
    h_matrix = inverse_root @ g_matrix @ inverse_root
    _require(np.allclose(h_matrix, h_matrix.T, atol=1.0e-13), "H is not symmetric")
    values = np.linalg.eigvalsh(h_matrix)
    _require(np.min(values) > 0.0, "nonpositive generalized speed")
    return values


def numeric_suite(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    transform = np.array([[1.0, 0.35], [-0.2, 1.1]], dtype=float)
    records: list[dict[str, Any]] = []
    for case in config["numeric_cases"]:
        k_matrix = np.array(case["K"], dtype=float)
        g_matrix = np.array(case["G"], dtype=float)
        inverse_root = _inverse_square_root(k_matrix)
        h_matrix = inverse_root @ g_matrix @ inverse_root
        speeds, rotation = np.linalg.eigh(h_matrix)
        transformed_speeds = _generalized_speeds(
            transform.T @ k_matrix @ transform,
            transform.T @ g_matrix @ transform,
        )
        canonical_k = rotation.T @ inverse_root @ k_matrix @ inverse_root @ rotation
        canonical_g = rotation.T @ h_matrix @ rotation
        time_vector = np.array([0.7, -0.4])
        time_energy = float(time_vector @ k_matrix @ time_vector)
        spatial_vector = np.array([-0.2, 0.9])
        spatial_energy = float(spatial_vector @ g_matrix @ spatial_vector)
        passed = (
            np.min(np.linalg.eigvalsh(k_matrix)) > 0.0
            and np.min(np.linalg.eigvalsh(g_matrix)) > 0.0
            and np.min(speeds) > 0.0
            and np.allclose(canonical_k, np.eye(2), rtol=2.0e-12, atol=2.0e-12)
            and np.allclose(canonical_g, np.diag(speeds), rtol=2.0e-12, atol=2.0e-12)
            and np.allclose(speeds, transformed_speeds, rtol=2.0e-12, atol=2.0e-12)
            and time_energy > 0.0
            and spatial_energy > 0.0
        )
        _require(bool(passed), f"numeric common-time benchmark failed: {case['case_id']}")
        records.append(
            {
                "case_id": case["case_id"],
                "K_eigenvalues": [
                    format(float(value), ".17g") for value in np.linalg.eigvalsh(k_matrix)
                ],
                "G_eigenvalues": [
                    format(float(value), ".17g") for value in np.linalg.eigvalsh(g_matrix)
                ],
                "generalized_speed_squared": [format(float(value), ".17g") for value in speeds],
                "canonical_K_max_error": format(
                    float(np.max(np.abs(canonical_k - np.eye(2)))), ".17g"
                ),
                "canonical_G_max_error": format(
                    float(np.max(np.abs(canonical_g - np.diag(speeds)))), ".17g"
                ),
                "field_redefinition_invariant": bool(
                    np.allclose(speeds, transformed_speeds, rtol=2.0e-12, atol=2.0e-12)
                ),
                "positive_principal_energy_probe": time_energy > 0.0 and spatial_energy > 0.0,
                "passed": True,
            }
        )
    return records


def build_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    config = load_config(base)
    local = _validate_local_integrity(base, config)
    bound = _validate_bindings(base, config)
    symbolic = symbolic_checks()
    numeric = numeric_suite(config)
    cone = bound["CONE_STRADDLING_THEOREM"]
    cone_speeds = [
        [float(value) for value in row["generalized_speed_squared"]]
        for row in cone["numeric_benchmarks"]
    ]
    checks = {
        "T01_PACKAGE_POLICY_AND_BINDINGS": True,
        "T02_K_G_POSITIVE_DEFINITE": all(
            min(float(value) for value in row["K_eigenvalues"]) > 0.0
            and min(float(value) for value in row["G_eigenvalues"]) > 0.0
            for row in numeric
        ),
        "T03_SYMMETRIC_POSITIVE_H": all(
            min(float(value) for value in row["generalized_speed_squared"]) > 0.0 for row in numeric
        ),
        "T04_ORTHOGONAL_DIAGONALIZATION": all(
            float(row["canonical_K_max_error"]) < 2.0e-12
            and float(row["canonical_G_max_error"]) < 2.0e-12
            for row in numeric
        ),
        "T05_DECOUPLED_WAVE_FORM": symbolic["decoupled_characteristic_polynomial"],
        "T06_POSITIVE_REAL_SPEEDS": all(
            min(float(value) for value in row["generalized_speed_squared"]) > 0.0 for row in numeric
        ),
        "T07_POSITIVE_PRINCIPAL_ENERGY": symbolic["positive_energy_matrix"]
        and all(row["positive_principal_energy_probe"] for row in numeric),
        "T08_COMMON_DT_COVECTOR": symbolic["dt_scalar_form_negative"],
        "T09_CONSTANT_COEFFICIENT_STRONG_HYPERBOLICITY": symbolic["symmetric_first_order_symbol"]
        and symbolic["complete_real_first_order_spectrum"],
        "T10_STRADDLING_RETAINED": all(0.0 < values[0] < 1.0 < values[1] for values in cone_speeds),
        "T11_LINEAR_FIELD_REDEFINITION_INVARIANCE": all(
            row["field_redefinition_invariant"] for row in numeric
        ),
        "T12_PAPER_SCOPE_AND_CAUSAL_CAVEAT": len(config["primary_paper_anchors"]) == 3
        and config["claim_boundary"]["local_causal_paradox_established"] is False,
        "T13_CLAIM_CEILING": config["claim_boundary"][
            "constant_coefficient_scalar_block_strongly_hyperbolic"
        ]
        and config["claim_boundary"]["local_common_time_covector_with_metric"]
        and all(
            config["claim_boundary"][key] is False
            for key in (
                "local_causal_paradox_established",
                "global_time_function_established",
                "variable_coefficient_strong_hyperbolicity",
                "full_metric_scalar_matter_system_healthy",
                "ultraviolet_completion",
                "observational_support",
                "publication_ready",
            )
        ),
        "T14_ZERO_OBSERVATIONAL_ACCESS": all(
            value == 0 for value in config["access_ledger"].values()
        ),
    }
    _require(list(checks) == config["required_checks"], "check inventory changed")
    _require(all(checks.values()), "common-time adjudication failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "PASS_LOCAL_COMMON_TIME_CONSTANT_COEFFICIENT_SCALAR_STRONG_HYPERBOLICITY",
        "decision": DECISION,
        "package_bindings": local,
        "predecessor_receipt_content_sha256": {
            item["id"]: item["receipt_content_sha256"] for item in config["bindings"]
        },
        "primary_paper_anchors": config["primary_paper_anchors"],
        "theorem": config["theorem"],
        "symbolic_checks": symbolic,
        "numeric_benchmarks": numeric,
        "retained_cone_speed_squared": [
            row["generalized_speed_squared"] for row in cone["numeric_benchmarks"]
        ],
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "claim_boundary": config["claim_boundary"],
        "access_ledger": config["access_ledger"],
        "interpretation": (
            "The faster scalar is retained. Positive K and G make the frozen scalar block strongly "
            "hyperbolic with a common local time covector, so metric-cone straddling is not itself a "
            "local PDE failure or causal paradox. Global and matter-cone questions remain open."
        ),
    }
    receipt["content_sha256"] = _self_hash(receipt)
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("analysis_id") == config["analysis_id"], "receipt identity changed")
    _require(receipt.get("decision") == DECISION, "receipt decision changed")
    _require(receipt.get("checks_passed") == receipt.get("checks_total") == 14, "checks incomplete")
    _require(receipt.get("theorem") == config["theorem"], "theorem changed")
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "claims changed")
    _require(receipt.get("access_ledger") == config["access_ledger"], "access ledger changed")
    _require(receipt.get("content_sha256") == _self_hash(receipt), "receipt self-hash invalid")


def _atomic_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == payload:
            return "EXISTING_IDENTICAL"
        raise KineticGateCommonTimeError("refusing to replace nonidentical receipt")
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
            raise KineticGateCommonTimeError("receipt publication race") from exc
    finally:
        temporary_path.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path | None = None) -> str:
    base = (root or _repo_root()).resolve()
    return _atomic_no_clobber(base / OUTPUT_PATH, build_receipt(base))


def check_receipt(root: Path | None = None) -> dict[str, Any]:
    base = (root or _repo_root()).resolve()
    stored = _read_json((base / OUTPUT_PATH).resolve())
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
