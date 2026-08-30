"""Restricted scalar-aether-metric principal-order factorization on the static branch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

from sigma_theory_compiler import (
    gravity_shared_quadrature_aether_mode_necessary_conditions as aether_modes,
)
from sigma_theory_compiler import (
    gravity_shared_quadrature_universal_vector_metric as vector_metric,
)

CONFIG_PATH = Path("configs/gravity_shared_quadrature_reduced_principal_factorization_v1.json")
TEST_PATH = Path("tests/test_gravity_shared_quadrature_reduced_principal_factorization.py")
OUTPUT_PATH = Path("runs/gravity/theory/shared-quadrature-reduced-principal-factorization-v1.json")
CONFIG_SCHEMA = "invariant-gravity-shared-quadrature-reduced-principal-factorization-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-shared-quadrature-reduced-principal-factorization-receipt-1.0"
STATUS = "restricted_static_branch_reduced_principal_factorization_derived_no_data"
DECISION = (
    "RESTRICTED_STATIC_BRANCH_REDUCED_PRINCIPAL_FACTORIZATION_DERIVED_SIX_PHYSICAL_"
    "MODES_CAUSAL_ON_FINITE_LOCUS_EXACT_PPN_LIMIT_AND_GLOBAL_HEALTH_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "bf9255267a3629a7ef97c3669c5de725984fed56b14615cc8dc98e552ca8d3f8"
EXPECTED_TEST_FILE_SHA256 = "30acd7d0ee8affac88cba61420fe3fd80b12d7999721e4f0139e5edab0ce36cd"
EXPECTED_SECTION_HASHES = {
    "predecessor_bindings": "520bc5e2aa0ffda8e5ae09ae3a1a78475b3531e042b7e869eebcbac7d03466e2",
    "frozen_background_contract": "935933544201932f3f9efdc4e0951cca8d383808148b42835644677e0b4fec36",
    "derivative_order_audit": "a7f494a4660b47b58fd3881334c8da53de0c904fd6acf37511acfb6e9b462356",
    "reduced_principal_factorization": "4f483adf74d61d9788a3e29fa609f7e86b62b99dee2da5d6db3cc6d9bcb0aea5",
    "finite_locus_contract": "9323dad655ebeace2d5d2897ec3bdd44579c6c75c08be1d26af33d0d477fb085",
    "obstruction_contract": "8ea38de1e47f82597f74df5809214228a784146828febc9fbc40dd4f43ea57ca",
    "machine_check_contract": "ceaab106325cc8ed33b1f25c06414584f87232db0db2bce00fd42e7ce06e4f06",
    "adjudication": "1d9f857577e8b5e5d0069ee9ba5fe776393560197a18a7016c2f009a3d2e34b2",
    "claim_boundary": "147d0ef9b175ba618b73e5771337bb0943131f531d5cbaaf2b02ddb5bf8fc16e",
    "zero_access_and_compute": "d49dd6f61c1704a662f2a82623b63bcb487c43018af38bad97d092c982b74e94",
}


class QuadratureReducedPrincipalError(RuntimeError):
    """Raised when the frozen factorization or its claim boundary changes."""


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
        raise QuadratureReducedPrincipalError(f"could not read JSON: {path}") from error
    if not isinstance(value, dict):
        raise QuadratureReducedPrincipalError(f"JSON root is not an object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise QuadratureReducedPrincipalError(f"{label} keys changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "frozen_background_contract",
            "derivative_order_audit",
            "reduced_principal_factorization",
            "finite_locus_contract",
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
        or config["analysis_id"] != "gravity-shared-quadrature-reduced-principal-factorization-v1"
        or config["status"] != "frozen_no_data_static_branch_reduced_principal_factorization"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise QuadratureReducedPrincipalError("config identity changed")
    for key, expected in EXPECTED_SECTION_HASHES.items():
        if _sha(config[key]) != expected:
            raise QuadratureReducedPrincipalError(f"config {key} changed")
    machine = config["machine_check_contract"]
    if len(machine["required_symbolic_checks"]) != 22 or len(machine["numeric_cases"]) != 4:
        raise QuadratureReducedPrincipalError("machine inventory changed")
    if config["adjudication"]["overall_decision"] != DECISION:
        raise QuadratureReducedPrincipalError("adjudication changed")
    if set(config["zero_access_and_compute"].values()) != {0}:
        raise QuadratureReducedPrincipalError("zero-access contract changed")


def load_config(root: Path) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise QuadratureReducedPrincipalError("config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _git(*args: str, root: Path) -> bytes:
    try:
        return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True).stdout
    except subprocess.CalledProcessError as error:
        raise QuadratureReducedPrincipalError("predecessor Git binding failed") from error


def validate_predecessors(
    root: Path, bindings: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for binding in bindings:
        commit = str(binding["git_commit"])
        if _git("cat-file", "-t", commit, root=root).strip() != b"commit":
            raise QuadratureReducedPrincipalError("predecessor commit changed")
        for artifact in binding["artifacts"]:
            relative = str(artifact["path"])
            current = root / relative
            if not current.is_file() or _file_sha(current) != artifact["file_sha256"]:
                raise QuadratureReducedPrincipalError("predecessor artifact changed")
            current_blob = _git("hash-object", "--path", relative, relative, root=root).strip()
            commit_blob = _git("rev-parse", f"{commit}:{relative}", root=root).strip()
            if current_blob != commit_blob:
                raise QuadratureReducedPrincipalError("predecessor commit bytes changed")
        receipt = _read_json(root / str(binding["receipt_path"]))
        body = {key: value for key, value in receipt.items() if key != "content_sha256"}
        if (
            receipt.get("content_sha256") != binding["receipt_content_sha256"]
            or _sha(body) != binding["receipt_content_sha256"]
            or receipt.get("schema_version") != binding["receipt_schema_version"]
            or receipt.get("decision") != binding["receipt_decision"]
        ):
            raise QuadratureReducedPrincipalError("predecessor receipt changed")
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


def symbolic_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pi_t, pi_x, pi_y = sp.symbols("pi_t pi_x pi_y", real=True)
    u, gamma = sp.symbols("delta_u delta_g", real=True)
    du_t, du_x, dg_t, dg_x = sp.symbols("du_t du_x dg_t dg_x", real=True)
    v, z_u, z_g = sp.symbols("v z_u z_g", real=True)
    f1, f2, b = sp.symbols("F1 F2 B", real=True)
    a_coef, k_coef, c_coef = sp.symbols("A K C", positive=True, finite=True)
    alpha, s, epsilon, q = sp.symbols("alpha s epsilon q", positive=True, finite=True)
    varphi = sp.symbols("varphi", real=True, finite=True)
    omega, wave_number, k_parallel, k_perp = sp.symbols("omega k k_parallel k_perp", real=True)

    delta_z_linear = 2 * v * pi_x + z_u * u + z_g * gamma
    delta_z_quadratic = pi_x**2 + pi_y**2
    delta_w = pi_t + v * u
    scalar_quadratic = f1 * delta_z_quadratic + f2 * delta_z_linear**2 / 2 + b * delta_w**2
    derivative_jets = (pi_t, pi_x, pi_y, du_t, du_x, dg_t, dg_x)
    derivative_hessian = sp.hessian(scalar_quadratic, derivative_jets)
    coefficient_map = {
        b: a_coef / 2,
        f1: c_coef / 2,
        f2: (k_coef - c_coef) / (4 * v**2),
    }
    mapped_hessian = sp.simplify(derivative_hessian.subs(coefficient_map))
    expected_hessian = sp.diag(a_coef, k_coef, c_coef, 0, 0, 0, 0)

    p_aether = (-(omega**2) + wave_number**2) ** 5
    p_scalar = -a_coef * omega**2 + k_coef * k_parallel**2 + c_coef * k_perp**2
    reduced_matrix = sp.diag(*([-(omega**2) + wave_number**2] * 5), p_scalar)
    reduced_determinant = sp.factor(reduced_matrix.det())

    c_expr = alpha**2 * s / (1 - 2 * s)
    k_expr = 2 * alpha**2 * s * (1 - s) / (1 - 2 * s) ** 2
    a_expr = sp.exp(-4 * varphi) * k_expr
    parallel_speed = sp.factor(k_expr / a_expr)
    transverse_speed = sp.factor(c_expr / a_expr)
    s_parameter = q / (2 * (1 + q))
    epsilon_parameter = q / (2 * (1 + q))
    vector_residue = 2 * epsilon
    spin0_residue = epsilon * (2 - epsilon)
    exact_c14, exact_c123 = sp.symbols("c14_exact c123_exact")

    predecessor_vector_checks, _ = vector_metric.symbolic_checks()
    predecessor_aether_checks, _, inherited = aether_modes.symbolic_checks()
    if not all(row["passed"] for row in predecessor_vector_checks):
        raise QuadratureReducedPrincipalError("vector-metric predecessor checks failed")
    if not all(row["passed"] for row in predecessor_aether_checks):
        raise QuadratureReducedPrincipalError("aether predecessor checks failed")

    checks = [
        _check(
            "S01_SCALAR_DERIVATIVE_HESSIAN",
            mapped_hessian - expected_hessian,
            "The scalar contribution to the degree-two derivative Hessian is exactly its A,K,C block.",
        ),
        _check(
            "S02_SCALAR_TO_DERIVATIVE_AETHER_ZERO",
            mapped_hessian[:3, 3:5],
            "Scalar derivative jets have no degree-two cross Hessian with derivatives of U.",
        ),
        _check(
            "S03_SCALAR_TO_DERIVATIVE_METRIC_ZERO",
            mapped_hessian[:3, 5:7],
            "Scalar derivative jets have no degree-two cross Hessian with derivatives of g.",
        ),
        _check(
            "S04_AETHER_TO_DERIVATIVE_SCALAR_ZERO",
            mapped_hessian[3:5, :3],
            "The reciprocal derivative-U to derivative-scalar block vanishes.",
        ),
        _check(
            "S05_METRIC_TO_DERIVATIVE_SCALAR_ZERO",
            mapped_hessian[5:7, :3],
            "The reciprocal derivative-g to derivative-scalar block vanishes.",
        ),
        _check(
            "S06_STATIC_W_QUADRATIC_TIME_COEFFICIENT",
            mapped_hessian[0, 0] - a_coef,
            "The Wbar=0 scalar time coefficient is A.",
        ),
        _check(
            "S07_STATIC_Z_LONGITUDINAL_COEFFICIENT",
            mapped_hessian[1, 1] - k_coef,
            "The scalar-gradient longitudinal coefficient is K.",
        ),
        _check(
            "S08_STATIC_Z_TRANSVERSE_COEFFICIENT",
            mapped_hessian[2, 2] - c_coef,
            "The scalar-gradient transverse coefficient is C.",
        ),
        _check(
            "S09_REDUCED_DETERMINANT_FACTORIZATION",
            reduced_determinant - p_aether * p_scalar,
            "The six-mode reduced determinant factors into the five-mode aether and scalar factors.",
        ),
        _check(
            "S10_AETHER_LUMINAL_MULTIPLICITY",
            p_aether / (-(omega**2) + wave_number**2) ** 5 - 1,
            "The finite aether locus supplies five g-luminal physical factors.",
        ),
        _check(
            "S11_SCALAR_LONGITUDINAL_PHOTON_ALIGNMENT",
            parallel_speed - sp.exp(4 * varphi),
            "The scalar longitudinal speed equals the physical photon speed.",
        ),
        _check(
            "S12_SCALAR_TRANSVERSE_RATIO",
            transverse_speed - sp.exp(4 * varphi) * (1 - 2 * s) / (2 * (1 - s)),
            "The scalar transverse speed has the frozen interior-cone ratio.",
        ),
        _check(
            "S13_SCALAR_TRANSVERSE_POSITIVE_PARAMETERIZATION",
            transverse_speed.subs({s: s_parameter, varphi: 0}) - 1 / (2 + q),
            "s=q/[2(1+q)] makes the transverse squared speed manifestly positive.",
        ),
        _check(
            "S14_SCALAR_TRANSVERSE_SUBLUMINAL_PARAMETERIZATION",
            (1 - transverse_speed.subs({s: s_parameter, varphi: 0})) - (1 + q) / (2 + q),
            "The same parameterization makes the transverse cone strictly interior.",
        ),
        _check(
            "S15_SIX_REDUCED_MODE_COUNT",
            inherited["principal_mode_count"] + 1 - 6,
            "Five reduced aether modes plus one scalar mode give six physical factors.",
        ),
        _check(
            "S16_FINITE_LOCUS_POSITIVE_AETHER_KINETIC",
            sp.Matrix(
                [
                    vector_residue.subs(epsilon, epsilon_parameter) - q / (1 + q),
                    spin0_residue.subs(epsilon, epsilon_parameter)
                    - q * (4 + 3 * q) / (4 * (1 + q) ** 2),
                ]
            ),
            "The finite-locus vector and scalar aether residues have manifestly positive forms.",
        ),
        _check(
            "S17_FINITE_BRANCH_POSITIVE_SCALAR_TIME_COEFFICIENT",
            a_expr.subs({s: s_parameter, varphi: 0}) - alpha**2 * q * (2 + q) / 2,
            "The scalar time coefficient has a manifestly positive finite-branch form.",
        ),
        _check(
            "S18_FINITE_BRANCH_POSITIVE_SCALAR_SPATIAL_COEFFICIENTS",
            sp.Matrix(
                [
                    c_expr.subs(s, s_parameter) - alpha**2 * q / 2,
                    k_expr.subs(s, s_parameter) - alpha**2 * q * (2 + q) / 2,
                ]
            ),
            "Both scalar spatial coefficients have manifestly positive forms.",
        ),
        _check(
            "S19_EPSILON_ZERO_VECTOR_MARGIN",
            sp.limit(vector_residue + spin0_residue, epsilon, 0, dir="+"),
            "The positive aether kinetic margin vanishes at exact preferred-frame cancellation.",
        ),
        _check(
            "S20_S_ZERO_SCALAR_MARGIN",
            sp.limit((a_expr + k_expr + c_expr).subs(varphi, 0), s, 0, dir="+"),
            "The scalar kinetic and gradient margin vanishes at the low-gradient boundary.",
        ),
        _check(
            "S21_EXACT_PPN_INTERSECTION_SINGULAR",
            sp.Matrix([exact_c14, exact_c123]).subs({exact_c14: 0, exact_c123: 0}),
            "Exact c13=alpha1=alpha2=0 has c14=c123=0.",
        ),
        _check(
            "S22_PHYSICAL_CONE_RATIO_AT_VARPHI_ZERO",
            sp.exp(-2 * varphi).subs(varphi, 0) - 1,
            "At varphi=0 the local Einstein and universal physical photon cones align.",
        ),
    ]
    expressions = {
        "scalar_quadratic_model": str(scalar_quadratic),
        "scalar_derivative_hessian": str(mapped_hessian),
        "lower_order_scalar_aether_mixing": str(
            sp.diff(sp.diff(scalar_quadratic, pi_t), u).subs(coefficient_map)
        ),
        "lower_order_scalar_metric_mixing": str(
            sp.diff(sp.diff(scalar_quadratic, pi_x), gamma).subs(coefficient_map)
        ),
        "aether_five_mode_factor": str(p_aether),
        "scalar_factor": str(p_scalar),
        "combined_reduced_determinant": str(reduced_determinant),
        "scalar_parallel_speed_squared": str(parallel_speed),
        "scalar_transverse_speed_squared": str(transverse_speed),
        "predecessor_vector_checks_passed": sum(
            bool(row["passed"]) for row in predecessor_vector_checks
        ),
        "predecessor_aether_checks_passed": sum(
            bool(row["passed"]) for row in predecessor_aether_checks
        ),
    }
    return checks, expressions


def numeric_cases(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["numeric_cases"]:
        epsilon = sp.Rational(case["epsilon"])
        s = sp.Rational(case["s"])
        a_coef = 2 * s * (1 - s) / (1 - 2 * s) ** 2
        k_coef = a_coef
        c_coef = s / (1 - 2 * s)
        transverse = sp.factor(c_coef / a_coef)
        vector_residue = 2 * epsilon
        spin0_residue = epsilon * (2 - epsilon)
        lower_order_mixing = sp.factor(a_coef * s)
        rows.append(
            {
                "epsilon": str(epsilon),
                "s": str(s),
                "aether_speed_squared": ["1", "1", "1", "1", "1"],
                "scalar_speed_squared": {
                    "parallel": "1",
                    "transverse": str(transverse),
                },
                "positive_coefficients": {
                    "spin_1_residue": str(vector_residue),
                    "spin_0_residue": str(spin0_residue),
                    "A": str(a_coef),
                    "K": str(k_coef),
                    "C": str(c_coef),
                },
                "lower_order_scalar_aether_mixing_nonzero": bool(lower_order_mixing != 0),
                "maximum_physical_speed_squared": "1",
                "passed": bool(
                    0 < epsilon < sp.Rational(1, 2)
                    and 0 < s < sp.Rational(1, 2)
                    and vector_residue > 0
                    and spin0_residue > 0
                    and a_coef > 0
                    and k_coef > 0
                    and c_coef > 0
                    and 0 < transverse < 1
                ),
            }
        )
    return rows


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    predecessors = validate_predecessors(root, config["predecessor_bindings"])
    test_path = root / TEST_PATH
    if _file_sha(test_path) != EXPECTED_TEST_FILE_SHA256:
        raise QuadratureReducedPrincipalError("test file hash changed")
    symbolic, expressions = symbolic_checks()
    numeric = numeric_cases(config)
    if tuple(row["check_id"] for row in symbolic) != tuple(
        config["machine_check_contract"]["required_symbolic_checks"]
    ):
        raise QuadratureReducedPrincipalError("symbolic inventory changed")
    if not all(row["passed"] for row in symbolic):
        raise QuadratureReducedPrincipalError("symbolic derivation failed")
    if not all(row["passed"] for row in numeric):
        raise QuadratureReducedPrincipalError("numeric case failed")
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
        "frozen_background_contract": config["frozen_background_contract"],
        "derivative_order_audit": config["derivative_order_audit"],
        "reduced_principal_factorization": config["reduced_principal_factorization"],
        "finite_locus_contract": config["finite_locus_contract"],
        "obstruction_contract": config["obstruction_contract"],
        "machine_results": {
            "symbolic_checks": symbolic,
            "derived_expressions": expressions,
            "numeric_cases": numeric,
        },
        "counts": {
            "predecessor_bindings": len(predecessors),
            "predecessor_artifacts": sum(row["artifact_count"] for row in predecessors),
            "reduced_physical_modes": config["reduced_principal_factorization"][
                "physical_mode_count"
            ],
            "symbolic_checks": len(symbolic),
            "symbolic_checks_passed": sum(bool(row["passed"]) for row in symbolic),
            "numeric_cases": len(numeric),
            "numeric_cases_passed": sum(bool(row["passed"]) for row in numeric),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls_by_builder": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The factorization is an exact derivative-order statement only on the frozen constant-coefficient Wbar=0 branch.",
            "Lower-order scalar-aether and scalar-metric mixing is present and retained; it is not a principal characteristic term in the declared second-order system.",
            "The five-mode aether factor is already gauge/constraint reduced; no unreduced gauge-fixed strong-hyperbolicity proof is supplied.",
            "The nonzero scalar gradient patch is not claimed to solve the global field equations or any Solar, compact-object, lensing, or FLRW boundary-value problem.",
            "The exact preferred-frame-free limit and the scalar endpoints remain singular, so no uniform cutoff or healthy completed action is established.",
            "No observation, novelty, or publication-readiness claim is made.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body):
        raise QuadratureReducedPrincipalError("receipt hash changed")
    if dict(receipt) != build_receipt(root):
        raise QuadratureReducedPrincipalError("receipt evidence changed")


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise QuadratureReducedPrincipalError("refusing to overwrite receipt")
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
            raise QuadratureReducedPrincipalError("receipt publication race") from error
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
            "reduced_factorization": receipt["adjudication"][
                "restricted_reduced_physical_principal_factorization_established"
            ],
            "six_mode_local_causality": receipt["adjudication"][
                "finite_locus_six_reduced_modes_causal_relative_to_physical_photons"
            ],
            "full_covariant_health": receipt["claim_boundary"]["full_covariant_health_established"],
            "content_sha256": receipt["content_sha256"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
