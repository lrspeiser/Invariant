"""Restricted universal vector-metric completion of the quadrature force law."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_shared_quadrature_universal_vector_metric_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_universal_vector_metric.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-universal-vector-metric-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-universal-vector-metric-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-universal-vector-metric-receipt-1.0"
STATUS = (
    "restricted_universal_vector_metric_quadrature_action_derived_no_data_global_health_blocked"
)
DECISION = (
    "RESTRICTED_QUADRATURE_UNIVERSAL_VECTOR_METRIC_ACTION_DERIVED_LEADING_MOTION_LENSING_"
    "RELATION_AND_FIXED_AETHER_SCALAR_CAUSALITY_PASS_GLOBAL_VECTOR_ENDPOINT_SOLAR_GW_"
    "COSMOLOGY_GATES_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "424cbece2339c9ec0dc83c8f8dd7679822dfcba95e4a60cb4e1938587ac631af"
EXPECTED_TEST_FILE_SHA256 = "7b22dc7fb157ce59552c68c760b735424257a4fb4abc57d5b4dbe304761abedf"
EXPECTED_SECTION_HASHES = {
    "predecessor_bindings": "ce7a4ad093fb41de32ee31cfbacc86142f0c9bc935bf708b28613b7b5a8d26fc",
    "restricted_action_contract": "cf05065ad4ad0ff45f424d57a267d4b1a90940006ba9f2a269a408530a7747f2",
    "universal_metric_contract": "de584bcb85eec4c01f32550cfef0489b9903cb8682623c05c6a2938fe179b4ff",
    "fixed_aether_scalar_principal_contract": "7e7b9fdee90dded02502beb03065f005825c029efa8848fde28821f84543c03d",
    "gw_solar_cosmology_contract": "299853b4c1fa43b008586ed8cb8778c968813f036e595484db9ae0549ef36343",
    "obstruction_contract": "7aa6b2b898e762fa726b015c8cc8d217d9bf3d7437f44fb92aea746e5c28edf9",
    "machine_check_contract": "66e7e880d36593285eebc2cc51c9bd10af227490f3639cf72091e456358e44b3",
    "adjudication": "9f2192a19976e2a71580654cd06fddbb7a7306332f4a4ce5607f1764d2a8bf4c",
    "claim_boundary": "11c8590056b6a67f4b0e3e6eb0adc7b3a89652610d9d3e0dc3b4492dc46addf6",
    "zero_access_and_compute": "e78c1585d70382245459c43ceeacbd416dcc9aede3ed6dcca8c0e9add7cae190",
}


class QuadratureUniversalVectorMetricError(RuntimeError):
    """Raised when a frozen vector-metric contract or derivation changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QuadratureUniversalVectorMetricError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureUniversalVectorMetricError(f"JSON root is not an object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise QuadratureUniversalVectorMetricError(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "restricted_action_contract",
            "universal_metric_contract",
            "fixed_aether_scalar_principal_contract",
            "gw_solar_cosmology_contract",
            "obstruction_contract",
            "machine_check_contract",
            "adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["analysis_id"] != "gravity-shared-quadrature-universal-vector-metric-v1"
        or config["status"] != "frozen_no_data_restricted_universal_vector_metric_action"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureUniversalVectorMetricError("config identity changed")
    for key, expected in EXPECTED_SECTION_HASHES.items():
        if _sha(config[key]) != expected:
            raise QuadratureUniversalVectorMetricError(f"config {key} changed")
    machine = config["machine_check_contract"]
    if len(machine["required_symbolic_checks"]) != 21 or len(machine["numeric_cases"]) != 4:
        raise QuadratureUniversalVectorMetricError("machine-check inventory changed")
    if config["adjudication"]["overall_decision"] != DECISION:
        raise QuadratureUniversalVectorMetricError("adjudication changed")
    if set(config["zero_access_and_compute"].values()) != {0}:
        raise QuadratureUniversalVectorMetricError("zero-access contract changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureUniversalVectorMetricError("config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise QuadratureUniversalVectorMetricError("predecessor Git binding failed") from error


def validate_predecessors(
    root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for binding in bindings:
        commit = str(binding["git_commit"])
        if _git("cat-file", "-t", commit, root=root).strip() != b"commit":
            raise QuadratureUniversalVectorMetricError("predecessor commit changed")
        for artifact in binding["artifacts"]:
            relative = str(artifact["path"])
            current = root / relative
            if not current.is_file() or _file_sha(current) != artifact["file_sha256"]:
                raise QuadratureUniversalVectorMetricError("predecessor artifact changed")
            if _git("show", f"{commit}:{relative}", root=root) != current.read_bytes():
                raise QuadratureUniversalVectorMetricError("predecessor commit bytes changed")
        receipt = _read_json(root / str(binding["receipt_path"]))
        body = {key: value for key, value in receipt.items() if key != "content_sha256"}
        if (
            receipt.get("content_sha256") != binding["receipt_content_sha256"]
            or _sha(body) != binding["receipt_content_sha256"]
            or receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
        ):
            raise QuadratureUniversalVectorMetricError("predecessor receipt changed")
        validated.append(
            {
                "binding_id": binding["binding_id"],
                "git_commit": commit,
                "artifact_count": len(binding["artifacts"]),
                "receipt_content_sha256": receipt["content_sha256"],
                "valid": True,
            }
        )
    return validated


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    if isinstance(simplified, sp.MatrixBase):
        passed = all(sp.simplify(entry) == 0 for entry in simplified)
    else:
        passed = simplified == 0
    return {
        "check_id": check_id,
        "statement": statement,
        "residual": str(simplified),
        "passed": passed,
    }


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    s, alpha = sp.symbols("s alpha", positive=True)
    weak_varphi = sp.symbols("weak_varphi", real=True)
    epsilon = sp.symbols("epsilon", real=True)
    phi_e, psi_e = sp.symbols("Phi_E Psi_E", real=True)
    delta = sp.symbols("delta", positive=True)
    c13 = sp.symbols("c13", real=True)
    rho = sp.symbols("rho", positive=True)

    eta = sp.diag(-1, 1, 1, 1)
    u_up = sp.Matrix([1, 0, 0, 0])
    u_down = eta * u_up
    h_up = eta + u_up * u_up.T
    h_down = eta + u_down * u_down.T
    physical_a = sp.exp(-2 * weak_varphi) * h_down - sp.exp(2 * weak_varphi) * (u_down * u_down.T)
    physical_b = sp.exp(-2 * weak_varphi) * eta - 2 * sp.sinh(2 * weak_varphi) * (u_down * u_down.T)
    physical_inverse = sp.exp(2 * weak_varphi) * h_up - sp.exp(-2 * weak_varphi) * (u_up * u_up.T)

    gtt = -sp.exp(2 * epsilon * weak_varphi) * (1 + 2 * epsilon * phi_e)
    gxx = sp.exp(-2 * epsilon * weak_varphi) * (1 - 2 * epsilon * psi_e)
    phi_tilde = -sp.diff(gtt, epsilon).subs(epsilon, 0) / 2
    psi_tilde = -sp.diff(gxx, epsilon).subs(epsilon, 0) / 2
    motion_shift = sp.simplify(phi_tilde - phi_e)
    lensing_shift = sp.simplify(phi_tilde + psi_tilde - phi_e - psi_e)

    p = s**2 / 4 + s / 4 + sp.log(1 - 2 * s) / 8
    xbar = -(s**2) / (2 * alpha**2)
    c_coeff = sp.simplify(sp.diff(p, s) / sp.diff(xbar, s))
    k_coeff = sp.simplify(c_coeff + 2 * xbar * (sp.diff(c_coeff, s) / sp.diff(xbar, s)))
    a_coeff = sp.exp(-4 * weak_varphi) * k_coeff
    c_photon_sq = sp.exp(4 * weak_varphi)
    c_parallel_sq = sp.simplify(k_coeff / a_coeff)
    c_transverse_sq = sp.simplify(c_coeff / a_coeff)
    transverse_ratio = sp.simplify(c_transverse_sq / c_photon_sq)

    lower = -sp.log(1 + delta) / 2
    upper = -sp.log(1 - delta) / 2
    dust_source = sp.simplify(
        -sp.Rational(1, 2)
        * rho
        * sp.diff(-sp.exp(2 * weak_varphi), weak_varphi).subs(weak_varphi, 0)
    )
    checks = [
        _check(
            "S01_PHYSICAL_METRIC_EQUIVALENT_FORMS",
            physical_a - physical_b,
            "The projector and sinh forms of the physical metric coincide.",
        ),
        _check(
            "S02_PHYSICAL_METRIC_INVERSE",
            physical_a * physical_inverse - sp.eye(4),
            "The frozen physical inverse is exact in the U-rest frame.",
        ),
        _check(
            "S03_PHYSICAL_VOLUME_FACTOR",
            sp.det(physical_a) + sp.exp(-4 * weak_varphi),
            "sqrt(-gtilde)=exp(-2varphi)sqrt(-g).",
        ),
        _check(
            "S04_WEAK_TIME_POTENTIAL",
            phi_tilde - (phi_e + weak_varphi),
            "Phi_tilde=Phi_E+varphi.",
        ),
        _check(
            "S05_WEAK_SPACE_POTENTIAL",
            psi_tilde - (psi_e + weak_varphi),
            "Psi_tilde=Psi_E+varphi.",
        ),
        _check(
            "S06_SCALAR_MOTION_SHIFT",
            motion_shift - weak_varphi,
            "The scalar motion shift is varphi.",
        ),
        _check(
            "S07_SCALAR_LENSING_SHIFT",
            lensing_shift - 2 * weak_varphi,
            "The scalar lensing-potential shift is twice varphi.",
        ),
        _check(
            "S08_GR_EQUIVALENT_MOTION_LENSING_RELATION",
            lensing_shift - 2 * motion_shift,
            "The leading scalar lensing gradient is twice its motion gradient.",
        ),
        _check(
            "S09_KINETIC_DENSITY_DERIVATIVE",
            sp.diff(p, s) + s**2 / (1 - 2 * s),
            "The quadrature kinetic density derivative is unchanged.",
        ),
        _check(
            "S10_SPATIAL_TRANSVERSE_COEFFICIENT",
            c_coeff - alpha**2 * s / (1 - 2 * s),
            "C=alpha^2 s/(1-2s).",
        ),
        _check(
            "S11_SPATIAL_LONGITUDINAL_COEFFICIENT",
            k_coeff - 2 * alpha**2 * s * (1 - s) / (1 - 2 * s) ** 2,
            "K=2alpha^2s(1-s)/(1-2s)^2.",
        ),
        _check(
            "S12_TIME_COEFFICIENT",
            a_coeff - sp.exp(-4 * weak_varphi) * k_coeff,
            "A=exp(-4varphi)K.",
        ),
        _check(
            "S13_LONGITUDINAL_PHYSICAL_CONE_MATCH",
            c_parallel_sq - c_photon_sq,
            "The fixed-aether longitudinal scalar and physical photon cones coincide.",
        ),
        _check(
            "S14_TRANSVERSE_PHYSICAL_CONE_RATIO",
            transverse_ratio - (1 - 2 * s) / (2 * (1 - s)),
            "The transverse scalar cone lies inside the physical cone.",
        ),
        _check(
            "S15_TRANSVERSE_CONE_STRICT_INTERIOR",
            1 - transverse_ratio - 1 / (2 * (1 - s)),
            "The positive cone gap is 1/[2(1-s)].",
        ),
        _check(
            "S16_LOW_GRADIENT_C_ZERO",
            sp.limit(c_coeff, s, 0, dir="+"),
            "C vanishes at the deep-gradient boundary.",
        ),
        _check(
            "S17_LOW_GRADIENT_K_ZERO",
            sp.limit(k_coeff, s, 0, dir="+"),
            "K vanishes at the deep-gradient boundary.",
        ),
        _check(
            "S18_FINITE_ENDPOINT_K_DIVERGENCE",
            sp.limit(1 / k_coeff, s, sp.Rational(1, 2), dir="-"),
            "K diverges at the finite-gradient endpoint.",
        ),
        _check(
            "S19_TENSOR_C13_G_NULL_CONDITION",
            (1 / (1 - c13)).subs(c13, 0) - 1,
            "c13=0 makes the adopted Einstein-aether tensor cone g-null.",
        ),
        _check(
            "S20_GW_VARPHI_INTERVAL_ENDPOINTS",
            (sp.exp(-2 * lower) - (1 + delta)) ** 2 + (sp.exp(-2 * upper) - (1 - delta)) ** 2,
            "The frozen varphi interval saturates the two GW cone bounds.",
        ),
        _check(
            "S21_WEAK_DUST_SOURCE_NORMALIZATION",
            dust_source - rho,
            "The universal metric gives the leading nonrelativistic dust source with unit varphi coefficient.",
        ),
    ]
    expressions = {
        "physical_metric_rest_frame": str(physical_a),
        "physical_inverse_rest_frame": str(physical_inverse),
        "C": str(c_coeff),
        "K": str(k_coeff),
        "A": str(a_coeff),
        "transverse_physical_cone_ratio": str(transverse_ratio),
        "gw_varphi_lower": str(lower),
        "gw_varphi_upper": str(upper),
    }
    return checks, expressions


def numeric_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    rows: list[dict[str, Any]] = []
    alpha = 1.0
    for case in config["machine_check_contract"]["numeric_cases"]:
        s = float(case["s"])
        varphi = float(case["varphi"])
        c_coeff = alpha**2 * s / (1 - 2 * s)
        k_coeff = 2 * alpha**2 * s * (1 - s) / (1 - 2 * s) ** 2
        a_coeff = math.exp(-4 * varphi) * k_coeff
        photon_sq = math.exp(4 * varphi)
        parallel_sq = k_coeff / a_coeff
        transverse_sq = c_coeff / a_coeff
        passed = (
            c_coeff > 0
            and k_coeff > 0
            and a_coeff > 0
            and abs(parallel_sq - photon_sq) <= tolerance
            and 0 < transverse_sq < photon_sq
        )
        rows.append(
            {
                "s": s,
                "varphi": varphi,
                "C": c_coeff,
                "K": k_coeff,
                "A": a_coeff,
                "physical_photon_speed_squared": photon_sq,
                "scalar_parallel_speed_squared": parallel_sq,
                "scalar_transverse_speed_squared": transverse_sq,
                "passed": passed,
            }
        )
    return rows


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    predecessors = validate_predecessors(root, config["predecessor_bindings"])
    test_path = root / TEST_PATH
    if _file_sha(test_path) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureUniversalVectorMetricError("test file hash changed")
    symbolic, expressions = symbolic_checks()
    numeric = numeric_cases(config)
    if tuple(row["check_id"] for row in symbolic) != tuple(
        config["machine_check_contract"]["required_symbolic_checks"]
    ):
        raise QuadratureUniversalVectorMetricError("symbolic inventory changed")
    if not all(row["passed"] for row in symbolic):
        raise QuadratureUniversalVectorMetricError("symbolic derivation failed")
    if not all(row["passed"] for row in numeric):
        raise QuadratureUniversalVectorMetricError("numeric case failed")
    source_path = Path(__file__).resolve()
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": STATUS,
        "decision": DECISION,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(root / CONFIG_PATH),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "source_path": source_path.relative_to(root).as_posix(),
            "source_file_sha256": _file_sha(source_path),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(test_path),
        },
        "predecessor_validation": predecessors,
        "restricted_action_contract": config["restricted_action_contract"],
        "universal_metric_contract": config["universal_metric_contract"],
        "fixed_aether_scalar_principal_contract": config["fixed_aether_scalar_principal_contract"],
        "gw_solar_cosmology_contract": config["gw_solar_cosmology_contract"],
        "obstruction_contract": config["obstruction_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "derived_expressions": expressions,
            "numeric_cases": numeric,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "predecessor_artifacts": sum(row["artifact_count"] for row in predecessors),
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(bool(row["passed"]) for row in symbolic),
            "numeric_cases": len(numeric),
            "numeric_cases_passed": sum(bool(row["passed"]) for row in numeric),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The leading weak-field potential relation is exact within the frozen local expansion, but no nonlinear metric-vector-scalar boundary-value problem is solved.",
            "The scalar cone result freezes vector and metric perturbations; full coupled characteristics and constraint propagation remain unproved.",
            "The known TeVeS-type physical metric removes the conformal lensing cancellation without a photon-only factor, but introduces vector health, PPN, GW-boundary, and cosmological obligations.",
            "The exact quadrature low-gradient degeneracy and finite-gradient endpoint singularity remain unchanged.",
            "No observational, lensing, confirmation, or independent row was accessed and no novelty or viability claim is allowed.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body):
        raise QuadratureUniversalVectorMetricError("receipt hash changed")
    if dict(receipt) != build_receipt(root):
        raise QuadratureUniversalVectorMetricError("receipt evidence changed")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise QuadratureUniversalVectorMetricError("refusing to overwrite receipt")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if path.read_bytes() == payload:
                return "EXISTING_IDENTICAL"
            raise QuadratureUniversalVectorMetricError("receipt publication race") from error
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path) -> tuple[Path, str]:
    path = root.resolve() / OUTPUT_PATH
    return path, _atomic_no_clobber(path, _canonical_bytes(build_receipt(root)))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        path, publication = write_receipt(root)
        output: Any = {"path": str(path), "publication": publication}
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {
            "valid": True,
            "decision": receipt["decision"],
            "content_sha256": receipt["content_sha256"],
        }
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "leading_motion_lensing_relation_matched": receipt["adjudication"][
                "leading_scalar_motion_and_lensing_relation_matched"
            ],
            "full_covariant_health_established": receipt["claim_boundary"][
                "full_covariant_health_established"
            ],
            "content_sha256": receipt["content_sha256"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
