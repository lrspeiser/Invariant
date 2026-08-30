"""No-data static source-bound analysis for the split-gate scalar action."""

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

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_split_gate_source_bound_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_split_gate_source_bound.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_split_gate_source_bound.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-split-gate-source-bound-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-split-gate-source-bound-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-split-gate-source-bound-receipt-1.0"
DECISION = (
    "PARTIAL_STATIC_SPLIT_GATE_SOURCE_CEILING_DERIVED_PHYSICAL_SOURCE_AND_"
    "ON_SHELL_BACKGROUND_NOT_ESTABLISHED"
)
EXPECTED_CONFIG_FILE_SHA256 = "a06b6155d46d81bf0d54075ff2a57c7aecfb4b6050267faa27c0b7c1c14c66cd"
EXPECTED_CONFIG_CONTENT_SHA256 = "685b01f26e2da761bf734619e25b54b9622d0e65ccf2b081112d59f4b5652ae3"

SYMBOLIC_CHECK_IDS = (
    "S01_FOURIER_RESPONSE",
    "S02_YUKAWA_RADIAL_GREEN_AWAY_FROM_SOURCE",
    "S03_YUKAWA_KERNEL_L1_NORM",
    "S04_LONG_WAVELENGTH_RESPONSE",
    "S05_FINITE_K_SUPPRESSION_FACTOR",
    "S06_FINITE_K_RELAXED_SOURCE_CEILING",
    "S07_C_BRANCH_SOURCE_SATURATION",
    "S08_K_BRANCH_SOURCE_SATURATION",
    "S09_HIGH_U_C_SOURCE_SCALING",
    "S10_HIGH_U_K_SOURCE_SCALING",
    "S11_R_TWO_RATIO_LIMIT",
    "S12_R_FIVE_HALVES_RATIO_LIMIT",
    "S13_R_THREE_RATIO_DIVERGENCE",
    "S14_CHI_CEILING_HIGH_U_REGRESSION",
    "S15_SOURCE_CEILING_HIGH_U_REGRESSION",
    "S16_FULL_COUPLED_DETERMINANT_CAVEAT",
    "S17_PREDECESSOR_SOURCE_SIGN_REGRESSION",
)


class SplitGateSourceBoundError(RuntimeError):
    """Raised when a binding, derivation, or publication guard fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SplitGateSourceBoundError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitGateSourceBoundError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _zero(value: Any) -> bool:
    if isinstance(value, sp.MatrixBase):
        return all(sp.simplify(item) == 0 for item in value)
    if isinstance(value, (tuple, list)):
        return all(_zero(item) for item in value)
    return sp.simplify(value) == 0


def _render(value: Any) -> Any:
    if _zero(value):
        return "ZERO_MATRIX" if isinstance(value, sp.MatrixBase) else "0"
    if isinstance(value, (tuple, list)):
        return [_render(item) for item in value]
    return sp.sstr(
        value.applyfunc(sp.simplify) if isinstance(value, sp.MatrixBase) else sp.simplify(value)
    )


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": _zero(residual),
        "residual": _render(residual),
        "statement": statement,
    }


def run_symbolic_suite() -> dict[str, Any]:
    y0, mass, z, k, source, chi_max = sp.symbols("Y0 m_chi Z k Q_chi chi_max", positive=True)
    m_eff_squared = mass**2 * z / y0
    fourier_response = -source / (y0 * (k**2 + m_eff_squared))
    fourier_equation = -y0 * (k**2 + m_eff_squared) * fourier_response - source

    radius, m_eff = sp.symbols("r m_eff", positive=True)
    green = sp.exp(-m_eff * radius) / (4 * sp.pi * y0 * radius)
    radial_operator = (
        sp.diff(green, radius, 2) + 2 * sp.diff(green, radius) / radius - m_eff**2 * green
    )
    kernel_l1 = sp.integrate(4 * sp.pi * radius**2 * green, (radius, 0, sp.oo))

    long_response = -source / (mass**2 * z)
    suppression = sp.simplify(fourier_response / long_response)
    finite_source_ceiling = (mass**2 * z + y0 * k**2) * chi_max

    x, beta, a0, b0 = sp.symbols("X beta A0 B0", positive=True)
    u = beta * x**2
    gate = (1 + u) ** 2
    gate_x = 4 * beta * x * (1 + u)
    h = 4 * beta * x * (3 + 7 * u)
    chi_c_max = sp.sqrt(2 * a0 / (mass**2 * gate_x))
    chi_k_max = sp.sqrt(2 * b0 / (mass**2 * h))
    source_c_max = sp.simplify(mass**2 * gate * chi_c_max)
    source_k_max = sp.simplify(mass**2 * gate * chi_k_max)
    source_c_limit = mass * beta * sp.sqrt(a0 / 2)
    source_k_limit = mass * beta * sp.sqrt(b0 / 14)

    q0 = sp.symbols("q0", positive=True)
    ratio_r2 = q0 * x**2 / source_k_max
    ratio_r_five_halves = q0 * x ** sp.Rational(5, 2) / source_k_max
    reciprocal_ratio_r3 = source_k_max / (q0 * x**3)

    d_phi, d_chi, d, vk = sp.symbols("D_phi D_chi d vk", real=True)
    coupled = sp.Matrix([[d_phi, -sp.I * d * vk], [sp.I * d * vk, d_chi]])
    coupled_expected = d_phi * d_chi - d**2 * vk**2
    chi_symbol = sp.symbols("chi_symbol", real=True)
    scalar_static_euler = -(y0 * k**2 + mass**2 * z) * chi_symbol
    matter_euler = -source
    total_euler = scalar_static_euler + matter_euler
    predecessor_sign_solution = sp.solve(total_euler, chi_symbol)[0]

    checks = [
        _check(
            "S01_FOURIER_RESPONSE",
            fourier_equation,
            "The frozen Fourier response solves Y0*(-k^2-m_eff^2)*chi_k=Q_k.",
        ),
        _check(
            "S02_YUKAWA_RADIAL_GREEN_AWAY_FROM_SOURCE",
            radial_operator,
            "The Yukawa kernel solves the homogeneous radial equation for r>0.",
        ),
        _check(
            "S03_YUKAWA_KERNEL_L1_NORM",
            kernel_l1 - 1 / (y0 * m_eff**2),
            "The positive R3 kernel has L1 norm 1/(Y0*m_eff^2).",
        ),
        _check(
            "S04_LONG_WAVELENGTH_RESPONSE",
            sp.limit(fourier_response, k, 0) - long_response,
            "The frozen-phi k->0 response is -Q_chi/(m_chi^2*Z).",
        ),
        _check(
            "S05_FINITE_K_SUPPRESSION_FACTOR",
            suppression - 1 / (1 + k**2 / m_eff_squared),
            "Finite k suppresses the frozen-phi response by 1/(1+k^2/m_eff^2).",
        ),
        _check(
            "S06_FINITE_K_RELAXED_SOURCE_CEILING",
            fourier_response.subs(source, finite_source_ceiling) + chi_max,
            "A positive source at the single-mode magnitude ceiling gives -chi_max; the absolute ceiling relaxes by 1+k^2/m_eff^2.",
        ),
        _check(
            "S07_C_BRANCH_SOURCE_SATURATION",
            source_c_max / (mass**2 * gate) - chi_c_max,
            "The exact C-branch source ceiling saturates the predecessor chi bound.",
        ),
        _check(
            "S08_K_BRANCH_SOURCE_SATURATION",
            source_k_max / (mass**2 * gate) - chi_k_max,
            "The exact K-branch source ceiling saturates the predecessor chi bound.",
        ),
        _check(
            "S09_HIGH_U_C_SOURCE_SCALING",
            sp.limit(source_c_max / x ** sp.Rational(5, 2), x, sp.oo) - source_c_limit,
            "The C-branch source ceiling scales as X^(5/2).",
        ),
        _check(
            "S10_HIGH_U_K_SOURCE_SCALING",
            sp.limit(source_k_max / x ** sp.Rational(5, 2), x, sp.oo) - source_k_limit,
            "The K-branch source ceiling scales as X^(5/2).",
        ),
        _check(
            "S11_R_TWO_RATIO_LIMIT",
            sp.limit(ratio_r2, x, sp.oo),
            "An r=2 source becomes asymptotically subcritical.",
        ),
        _check(
            "S12_R_FIVE_HALVES_RATIO_LIMIT",
            sp.limit(ratio_r_five_halves, x, sp.oo) - q0 / source_k_limit,
            "An r=5/2 source approaches a coefficient-dependent ratio.",
        ),
        _check(
            "S13_R_THREE_RATIO_DIVERGENCE",
            sp.limit(reciprocal_ratio_r3, x, sp.oo),
            "The reciprocal r=3 ratio vanishes, so the source/ceiling ratio diverges.",
        ),
        _check(
            "S14_CHI_CEILING_HIGH_U_REGRESSION",
            sp.limit(x**3 * chi_k_max**2, x, sp.oo) - b0 / (14 * mass**2 * beta**2),
            "The predecessor K-branch chi-squared ceiling retains X^-3 scaling.",
        ),
        _check(
            "S15_SOURCE_CEILING_HIGH_U_REGRESSION",
            sp.limit(source_k_max / x ** sp.Rational(5, 2), x, sp.oo) - source_k_limit,
            "Multiplication by m_chi^2*Z converts X^-3/2 amplitude scaling to X^(5/2) source scaling.",
        ),
        _check(
            "S16_FULL_COUPLED_DETERMINANT_CAVEAT",
            sp.det(coupled) - coupled_expected,
            "The general mixed local poles remain roots of D_phi*D_chi-d^2*(v.k)^2.",
        ),
        _check(
            "S17_PREDECESSOR_SOURCE_SIGN_REGRESSION",
            predecessor_sign_solution + source / (y0 * k**2 + mass**2 * z),
            "With predecessor E_chi and matter Euler density -Q_chi, total variation E_chi-Q_chi=0 fixes the negative Fourier response.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic derivation failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "checks": checks,
        "all_passed": True,
        "derived_expressions": {
            "fourier_response": "-Q_chi,k/(Y0*k^2+m_chi^2*Z)",
            "green_response": "chi=-G_Y*Q_chi with positive G_Y=exp(-m_eff*r)/(4*pi*Y0*r)",
            "sup_norm_bound": "norm(chi)_infinity<=norm(Q_chi)_infinity/(m_chi^2*Z)",
            "source_ceiling": "m_chi*Z*min[sqrt(2*A0/Z_X),sqrt(2*B0/H)]",
            "source_ceiling_high_u": "m_chi*beta*min[sqrt(A0/2),sqrt(B0/14)]*X^(5/2)",
            "general_coupled_determinant": "D_phi*D_chi-d^2*(v.k)^2",
        },
    }


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    probes = [float(value) for value in config["machine_check_contract"]["X_probes"]]
    records: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["source_scaling_cases"]:
        statuses: list[str] = []
        points: list[dict[str, Any]] = []
        r = float(case["r"])
        q0 = float(case["q0"])
        for x in probes:
            u = x**2
            gate = (1.0 + u) ** 2
            gate_x = 4.0 * x * (1.0 + u)
            h = 4.0 * x * (3.0 + 7.0 * u)
            chi_max = min(math.sqrt(2.0 / gate_x), math.sqrt(2.0 / h))
            source_max = gate * chi_max
            source_value = q0 * x**r
            ratio = source_value / source_max
            status = "pass" if ratio < 1.0 else "fail"
            statuses.append(status)
            points.append(
                {
                    "X": format(x, ".17g"),
                    "abs_Q_chi": format(source_value, ".17g"),
                    "abs_Q_chi_max": format(source_max, ".17g"),
                    "absolute_ratio": format(ratio, ".17g"),
                    "status": status,
                }
            )
        passed = statuses == case["expected"]
        records.append(
            {
                "case_id": case["case_id"],
                "r": format(r, ".17g"),
                "q0": format(q0, ".17g"),
                "points": points,
                "statuses": statuses,
                "designed_failure": case["designed_failure"],
                "passed": passed,
            }
        )
    _require(all(item["passed"] for item in records), "source-scaling numeric suite failed")
    _require(
        sum(item["designed_failure"] for item in records) == 2, "designed failure inventory changed"
    )

    x_reference = 1.0
    z_reference = (1.0 + x_reference**2) ** 2
    m_eff_squared = z_reference
    finite_k_records: list[dict[str, Any]] = []
    previous_suppression = math.inf
    for k_value in config["machine_check_contract"]["finite_k_probes"]:
        k = float(k_value)
        suppression = 1.0 / (1.0 + k**2 / m_eff_squared)
        relaxation = 1.0 + k**2 / m_eff_squared
        response_for_positive_unit_source = -1.0 / (k**2 + m_eff_squared)
        passed = (
            suppression <= previous_suppression
            and response_for_positive_unit_source < 0.0
            and abs(suppression * relaxation - 1.0) <= 1e-12
        )
        finite_k_records.append(
            {
                "k": format(k, ".17g"),
                "suppression": format(suppression, ".17g"),
                "source_ceiling_relaxation": format(relaxation, ".17g"),
                "chi_response_for_positive_unit_Q": format(
                    response_for_positive_unit_source, ".17g"
                ),
                "passed": passed,
            }
        )
        previous_suppression = suppression
    _require(all(item["passed"] for item in finite_k_records), "finite-k regression failed")
    return {
        "parameters": config["machine_check_contract"]["numeric_parameters"],
        "source_scaling_cases": records,
        "finite_k_regression": finite_k_records,
        "all_passed": True,
        "designed_failure_cases_preserved": 2,
        "failed_source_points_preserved": sum(
            point["status"] == "fail" for record in records for point in record["points"]
        ),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_binding",
            "frozen_source_problem",
            "finite_wavelength_contract",
            "amplitude_and_source_ceiling",
            "source_scaling_contract",
            "source_and_lensing_obligations",
            "machine_check_contract",
            "adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-split-gate-source-bound-v1",
        "analysis identity changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    binding = config["predecessor_binding"]
    _strict(
        binding,
        {
            "binding_id",
            "git_commit",
            "config_path",
            "config_file_sha256",
            "module_path",
            "module_file_sha256",
            "test_path",
            "test_file_sha256",
            "receipt_path",
            "receipt_file_sha256",
            "receipt_content_sha256",
            "receipt_schema_version",
            "receipt_decision",
        },
        "predecessor",
    )
    _require(
        binding["binding_id"] == "split_gate_action"
        and binding["git_commit"] == "03a652acaded1be4cca9af48782b8d54138e54c3",
        "predecessor identity changed",
    )
    source = config["frozen_source_problem"]
    _require(
        "total variation gives E_chi=Q_chi" in source["sign_convention"]
        and "Y0*(nabla^2-m_eff^2)*chi=Q_chi" in source["sign_convention"],
        "source sign convention changed",
    )
    _require(
        source["fourier_response"].startswith("chi_k=-Q_chi,k/")
        and "chi(x)=-integral" in source["green_response"]
        and "approximately -Q_chi,k" in source["long_wavelength_response"],
        "response sign changed",
    )
    _require(
        "delta_phi" in source["bare_response_scope"]
        and "D_phi*D_chi-d^2*(v.k)^2" in source["bare_response_scope"],
        "coupled-response caveat changed",
    )
    finite = config["finite_wavelength_contract"]
    _require(
        "conservative mode by mode" in finite["conservative_statement"]
        and "exact sufficient L-infinity" in finite["sup_norm_statement"],
        "finite-k conservatism changed",
    )
    ceiling = config["amplitude_and_source_ceiling"]
    _require(
        "Q_chi,max(X)=m_chi^2*Z(X)*chi_max(X)" in ceiling["exact_sufficient_source_ceiling"],
        "source ceiling changed",
    )
    _require(
        "X^(5/2)" in ceiling["high_u_overall_scaling"]
        and "X^-3/2" in ceiling["high_u_overall_scaling"],
        "high-u scaling changed",
    )
    scaling = config["source_scaling_contract"]
    _require(
        "X^(r-5/2)" in scaling["asymptotic_ratio"]
        and "No relation Q_chi(rho_baryon,X)" in scaling["forbidden_inference"],
        "source-scaling boundary changed",
    )
    obligations = config["source_and_lensing_obligations"]
    _require(
        "same universal matter action" in obligations["source_definition"]
        and "no independent photon multiplier" in obligations["lensing_boundary"],
        "source/lensing obligation changed",
    )
    machine = config["machine_check_contract"]
    _require(
        tuple(machine["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic contract changed",
    )
    _require(
        tuple(machine["X_probes"]) == (10.0, 20.0, 40.0)
        and len(machine["source_scaling_cases"]) == 4,
        "source probes changed",
    )
    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["sufficient_source_ceiling_derived"] is True
        and adjudication["physical_Q_chi_derived"] is False
        and adjudication["physical_on_shell_background"] is False,
        "partial adjudication changed",
    )
    claims = config["claim_boundary"]
    _require(
        claims["restricted_static_source_bound_established"] is True, "restricted claim disabled"
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key != "restricted_static_source_bound_established"
        ),
        "claim boundary overstated",
    )
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "access state changed",
    )


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    path = root.resolve() / CONFIG_PATH
    _require(path.is_file(), "config missing")
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "config file hash changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _validate_predecessor(root: Path, config: Mapping[str, Any]) -> None:
    binding = config["predecessor_binding"]
    for path_key, hash_key in (
        ("config_path", "config_file_sha256"),
        ("module_path", "module_file_sha256"),
        ("test_path", "test_file_sha256"),
        ("receipt_path", "receipt_file_sha256"),
    ):
        path = root / binding[path_key]
        _require(path.is_file(), f"predecessor missing: {path_key}")
        _require(_file_sha(path) == binding[hash_key], f"predecessor changed: {path_key}")
    receipt = _read_json(root / binding["receipt_path"])
    _require(
        receipt.get("schema_version") == binding["receipt_schema_version"],
        "predecessor schema changed",
    )
    _require(
        receipt.get("content_sha256") == binding["receipt_content_sha256"],
        "predecessor content changed",
    )
    _require(receipt.get("decision") == binding["receipt_decision"], "predecessor decision changed")


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    _validate_predecessor(root, config)
    _require((root / SOURCE_PATH).is_file(), "implementation missing")
    _require((root / TEST_PATH).is_file(), "test missing")
    symbolic = run_symbolic_suite()
    numeric = run_numeric_suite(config)
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "restricted_static_source_ceiling_machine_derived_not_physical_on_shell",
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
        "predecessor_binding": config["predecessor_binding"],
        "frozen_source_problem": config["frozen_source_problem"],
        "finite_wavelength_contract": config["finite_wavelength_contract"],
        "amplitude_and_source_ceiling": config["amplitude_and_source_ceiling"],
        "source_scaling_contract": config["source_scaling_contract"],
        "source_and_lensing_obligations": config["source_and_lensing_obligations"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "source_scaling_cases": len(numeric["source_scaling_cases"]),
            "source_scaling_cases_passed": sum(
                item["passed"] for item in numeric["source_scaling_cases"]
            ),
            "finite_k_probes": len(numeric["finite_k_regression"]),
            "finite_k_probes_passed": sum(
                item["passed"] for item in numeric["finite_k_regression"]
            ),
            "designed_failure_cases_preserved": numeric["designed_failure_cases_preserved"],
            "failed_source_points_preserved": numeric["failed_source_points_preserved"],
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The Green and source-ceiling results freeze X, Z, Y0, the external metric, and delta_phi on R3; they are not a physical source-supported solution.",
            "The exact long-wavelength L-infinity bound uses the positive decaying R3 Yukawa kernel; different boundaries or varying coefficients require a new estimate.",
            "Finite k suppresses each frozen-phi Fourier response, but active first-derivative phi-chi mixing restores the predecessor coupled determinant.",
            "The X^(5/2) ceiling is a mathematical allowance, not a prediction that a baryonic or material source scales with X.",
            "Q_chi has not been derived from a universal matter metric, equation of state, conserved source, or solved metric background.",
            "No physical on-shell, health, Solar, galaxy, cluster, lensing, cosmology, observational, novelty, or publication claim is established.",
        ],
    }
    receipt = {**body, "content_sha256": _sha(body)}
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
            "config_binding",
            "implementation_binding",
            "predecessor_binding",
            "frozen_source_problem",
            "finite_wavelength_contract",
            "amplitude_and_source_ceiling",
            "source_scaling_contract",
            "source_and_lensing_obligations",
            "symbolic_suite",
            "numeric_suite",
            "adjudication",
            "counts",
            "claim_boundary",
            "zero_access_and_compute",
            "limitations",
            "content_sha256",
        },
        "receipt",
    )
    body = dict(receipt)
    content_sha = body.pop("content_sha256")
    _require(content_sha == _sha(body), "receipt content hash changed")
    _require(
        receipt["schema_version"] == RECEIPT_SCHEMA and receipt["decision"] == DECISION,
        "receipt identity changed",
    )
    _require(
        receipt["config_binding"]["content_sha256"] == _sha(config),
        "receipt config binding changed",
    )
    _require(
        receipt["predecessor_binding"] == config["predecessor_binding"],
        "receipt predecessor changed",
    )
    _require(
        receipt["amplitude_and_source_ceiling"] == config["amplitude_and_source_ceiling"]
        and receipt["source_scaling_contract"] == config["source_scaling_contract"],
        "source contract changed",
    )
    _require(
        receipt["adjudication"] == config["adjudication"]
        and receipt["claim_boundary"] == config["claim_boundary"],
        "claims changed",
    )
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 17,
        "symbolic count changed",
    )
    _require(
        counts["source_scaling_cases"] == counts["source_scaling_cases_passed"] == 4,
        "source-case count changed",
    )
    _require(
        counts["finite_k_probes"] == counts["finite_k_probes_passed"] == 4, "finite-k count changed"
    )
    _require(
        counts["designed_failure_cases_preserved"] == 2
        and counts["failed_source_points_preserved"] == 4,
        "failure inventory changed",
    )
    _require(
        all(
            counts[key] == 0
            for key in (
                "observational_files_opened",
                "network_calls",
                "model_or_paid_calls",
                "gpu_calls",
            )
        ),
        "receipt access changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise SplitGateSourceBoundError(f"refusing to overwrite different receipt: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise SplitGateSourceBoundError(
                f"concurrent creator won; output preserved: {path}"
            ) from exc
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> tuple[dict[str, Any], str]:
    root = root.resolve()
    receipt = build_receipt(root)
    return receipt, _atomic_no_replace(root / OUTPUT_PATH, _canonical_bytes(receipt))


def check_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    expected = build_receipt(root)
    stored = _read_json(root / OUTPUT_PATH)
    validate_receipt(stored, config)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    return stored


def _summary(receipt: Mapping[str, Any], publication: str | None = None) -> dict[str, Any]:
    result = {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "content_sha256": receipt["content_sha256"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "source_cases_passed": receipt["counts"]["source_scaling_cases_passed"],
        "designed_failures_preserved": receipt["counts"]["designed_failure_cases_preserved"],
        "physical_source_derived": receipt["adjudication"]["physical_Q_chi_derived"],
        "physical_on_shell_background": receipt["adjudication"]["physical_on_shell_background"],
    }
    if publication is not None:
        result["publication"] = publication
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        if args.action == "write":
            receipt, publication = write_receipt(args.root)
            result = _summary(receipt, publication)
        else:
            result = _summary(check_receipt(args.root))
        print(json.dumps(result, sort_keys=True))
        return 0
    except SplitGateSourceBoundError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
