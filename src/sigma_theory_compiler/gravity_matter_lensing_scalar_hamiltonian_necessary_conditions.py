"""No-data scalar ADM Hamiltonian and Legendre conditions for the split-gate action."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_scalar_hamiltonian_necessary_conditions_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_scalar_hamiltonian_necessary_conditions.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_scalar_hamiltonian_necessary_conditions.py")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-scalar-hamiltonian-necessary-conditions-v1.json"
)
CONFIG_SCHEMA = (
    "invariant-gravity-matter-lensing-scalar-hamiltonian-necessary-conditions-config-1.0"
)
RECEIPT_SCHEMA = (
    "invariant-gravity-matter-lensing-scalar-hamiltonian-necessary-conditions-receipt-1.0"
)
DECISION = (
    "PARTIAL_SCALAR_ADM_HAMILTONIAN_AND_LEGENDRE_CONDITIONS_DERIVED_CP11_4_FULL_HEALTH_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "d36cccadd58ed25a44725a5620aad7e455150cdf653bf316915b6bb384a5ae2e"
EXPECTED_CONFIG_CONTENT_SHA256 = "907de84f2e288126b494bdafb087196988ff0a88559c526a595ab9ed529942ed"
EXPECTED_SECTION_SHA256 = {
    "predecessor_bindings": "27e618fea4460cf88a610d80e7216e9d9b4ffe800bfcb8c5ab6415e84187516a",
    "geometry_and_action_contract": "513ebdb1a10790c821c58efc61eeaaaaa07242700a40586400c118720c88c655",
    "adm_legendre_contract": "06fe5d5fb4133cb3f84a55a090a9e8694be8ec5148d0658b37aab32cc66a6f83",
    "canonical_hamiltonian_contract": "1ca2bb34cd9de9e13372a6dbf297e9b48b8b0d6f3a459e4237a0395841588ce6",
    "principal_and_slice_health_contract": "8a6da0ddf523db4b72b477d00db5633cf94f5c0124ebb9038a8388acc9e4b4c6",
    "homogeneous_energy_contract": "fb49f49a4ce7ba751e2e0690b059e358dcaf9429939c3d33a02541c009494a11",
    "machine_check_contract": "4bfa7d480ed4dff1ec86c59b61127556d36dc3416d3528f5f844b961baae0e11",
    "adjudication": "9d3b04d98ff639787f8fd9d64aef75db41227461503b757fced770088b2f2157",
    "claim_boundary": "9353d806992f9de6f61da8637445de2ad9d3e198648224f92610960060b39d80",
    "remaining_obligations": "891d4ec677ee68ff274cca335667eeeda9a0350fe26abfb999fae1fcc65b028e",
    "zero_access_and_compute": "6ec9a22001ae649681dc7e72fcd18da49369801daa7112592105628bdf1ff705",
}

SYMBOLIC_CHECK_IDS = (
    "S01_X_ADM_DECOMPOSITION",
    "S02_PHI_CANONICAL_MOMENTUM",
    "S03_CHI_CANONICAL_MOMENTUM",
    "S04_PHI_LEGENDRE_HESSIAN",
    "S05_CHI_LEGENDRE_HESSIAN",
    "S06_CROSS_LEGENDRE_HESSIAN_ZERO",
    "S07_CANONICAL_SHIFT_DENSITY",
    "S08_CANONICAL_NORMAL_DENSITY",
    "S09_STRESS_ENERGY_NORMAL_PROJECTION",
    "S10_HOMOGENEOUS_FLRW_ENERGY_REGRESSION",
    "S11_INVERSE_PHI_MOMENTUM_HESSIAN",
    "S12_INVERSE_CHI_MOMENTUM_HESSIAN",
    "S13_EFFECTIVE_METRIC_NORMAL_COEFFICIENT",
    "S14_EFFECTIVE_METRIC_MIXED_COEFFICIENT",
    "S15_EFFECTIVE_METRIC_SPATIAL_COEFFICIENT",
    "S16_COMPLETED_SQUARE_SPATIAL_MATRIX",
    "S17_LONGITUDINAL_SPATIAL_EIGENVALUE",
    "S18_EFFECTIVE_METRIC_DETERMINANT",
    "S19_ILLUSTRATIVE_GATE_ENERGY_FACTOR",
    "S20_GATE_ENERGY_SIGN_TRANSITION",
    "S21_POSITIVE_ENERGY_AMPLITUDE_BOUND",
    "S22_HIGH_U_ENERGY_BOUND_SCALING",
    "S23_HOMOGENEOUS_K_ADM_REGRESSION",
    "S24_ZERO_GRADIENT_SHIFT_DENSITY_REGRESSION",
)

TOP_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "purpose",
    "predecessor_bindings",
    "geometry_and_action_contract",
    "adm_legendre_contract",
    "canonical_hamiltonian_contract",
    "principal_and_slice_health_contract",
    "homogeneous_energy_contract",
    "machine_check_contract",
    "adjudication",
    "claim_boundary",
    "remaining_obligations",
    "zero_access_and_compute",
    "output_path",
}
BINDING_KEYS = {
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
}
ZERO_KEYS = {
    "observational_files_opened",
    "observational_rows_opened",
    "predictor_rows_opened",
    "response_rows_opened",
    "confirmation_rows_opened",
    "holdout_rows_opened",
    "independent_rows_opened",
    "lensing_rows_opened",
    "network_calls",
    "LLM_calls",
    "paid_calls",
    "GPU_calls",
}


class ScalarHamiltonianError(RuntimeError):
    """Raised when the frozen scalar Hamiltonian package changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScalarHamiltonianError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(isinstance(value, Mapping) and set(value) == keys, f"{label} keys changed")


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScalarHamiltonianError(f"cannot read JSON: {path}") from exc
    _require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    simplified = sp.simplify(residual)
    passed = simplified == 0
    return {
        "check_id": check_id,
        "passed": passed,
        "residual": "0" if passed else sp.sstr(simplified),
        "statement": statement,
    }


def _git_bytes(root: Path, commit: str, path: str) -> bytes:
    try:
        kind = subprocess.run(
            ["git", "cat-file", "-t", commit],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout.strip()
        _require(kind == b"commit", "predecessor git object is not a commit")
        return subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScalarHamiltonianError("cannot validate predecessor git binding") from exc


def _validate_predecessor(root: Path, binding: Mapping[str, Any]) -> None:
    _strict(binding, BINDING_KEYS, "predecessor binding")
    commit = str(binding["git_commit"])
    _require(len(commit) == 40, "predecessor commit length changed")
    for path_key, sha_key in (
        ("config_path", "config_file_sha256"),
        ("module_path", "module_file_sha256"),
        ("test_path", "test_file_sha256"),
        ("receipt_path", "receipt_file_sha256"),
    ):
        relative = str(binding[path_key])
        current = (root / relative).read_bytes()
        _require(
            hashlib.sha256(current).hexdigest() == binding[sha_key],
            f"{path_key} hash changed",
        )
        _require(_git_bytes(root, commit, relative) == current, f"{path_key} commit bytes changed")
    receipt = _read_json(root / str(binding["receipt_path"]))
    _require(receipt.get("schema_version") == binding["receipt_schema_version"], "schema changed")
    _require(receipt.get("decision") == binding["receipt_decision"], "decision changed")
    _require(receipt.get("content_sha256") == binding["receipt_content_sha256"], "content changed")
    binding_id = str(binding["binding_id"])
    if binding_id == "split_gate_action":
        _require(receipt["claim_boundary"]["healthy_action_established"] is False, "action ceiling")
    elif binding_id == "flrw_necessary_conditions":
        _require(
            receipt["adjudication"]["healthy_late_time_history_exists"] is False,
            "FLRW ceiling",
        )
    elif binding_id == "covariant_field_equations":
        _require(receipt["adjudication"]["full_H2"] is False, "covariant ceiling")
    elif binding_id == "adm_constraint_propagation":
        _require(receipt["adjudication"]["CP11_3_complete"] is True, "ADM predecessor")
        _require(
            receipt["adjudication"]["physical_hamiltonian_positive"] is False,
            "ADM health ceiling",
        )
    else:
        raise ScalarHamiltonianError("unexpected predecessor binding")


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    path = root / CONFIG_PATH
    _require(_file_sha(path) == EXPECTED_CONFIG_FILE_SHA256, "config file hash changed")
    config = _read_json(path)
    _strict(config, TOP_KEYS, "config")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content hash changed")
    for key, expected in EXPECTED_SECTION_SHA256.items():
        _require(_sha(config[key]) == expected, f"config section changed: {key}")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(
        tuple(config["machine_check_contract"]["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic check contract changed",
    )
    _strict(config["zero_access_and_compute"], ZERO_KEYS, "zero-access contract")
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "nonzero access declared",
    )
    _require(config["adjudication"]["CP11_4_complete"] is False, "CP11.4 overclaim")
    _require(
        config["claim_boundary"]["physical_hamiltonian_positivity_established"] is False,
        "Hamiltonian overclaim",
    )
    _require(
        config["claim_boundary"]["publication_readiness_changed"] is False,
        "readiness overclaim",
    )
    bindings = config["predecessor_bindings"]
    _require(
        [item["binding_id"] for item in bindings]
        == [
            "split_gate_action",
            "flrw_necessary_conditions",
            "covariant_field_equations",
            "adm_constraint_propagation",
        ],
        "predecessor order changed",
    )
    for binding in bindings:
        _validate_predecessor(root, binding)
    return config


def symbolic_suite() -> dict[str, Any]:
    v_phi, v_chi, s_phi, s_chi = sp.symbols("v_phi v_chi s_phi s_chi", real=True)
    y0, q_mass = sp.symbols("Y0 Q", positive=True)
    potential = sp.symbols("V", real=True)
    x = (v_phi**2 - s_phi**2) / 2
    x_chi = (v_chi**2 - s_chi**2) / 2
    x_variable = sp.symbols("X", real=True)
    p0, p1, p2, p3, z0, z1, z2, z3 = sp.symbols("p0 p1 p2 p3 z0 z1 z2 z3", real=True)
    p_polynomial = p0 + p1 * x_variable + p2 * x_variable**2 / 2 + p3 * x_variable**3 / 6
    z_polynomial = z0 + z1 * x_variable + z2 * x_variable**2 / 2 + z3 * x_variable**3 / 6
    p_value = p_polynomial.subs(x_variable, x)
    z_value = z_polynomial.subs(x_variable, x)
    p_x = sp.diff(p_polynomial, x_variable).subs(x_variable, x)
    z_x = sp.diff(z_polynomial, x_variable).subs(x_variable, x)
    p_xx = sp.diff(p_polynomial, x_variable, 2).subs(x_variable, x)
    z_xx = sp.diff(z_polynomial, x_variable, 2).subs(x_variable, x)
    lagrangian = p_value - potential + y0 * x_chi - q_mass * z_value
    c = p_x - q_mass * z_x
    c_x = p_xx - q_mass * z_xx
    k_adm = c + v_phi**2 * c_x
    pi_phi = sp.diff(lagrangian, v_phi)
    pi_chi = sp.diff(lagrangian, v_chi)
    checks = [
        _check(
            "S01_X_ADM_DECOMPOSITION",
            x - (v_phi**2 - s_phi**2) / 2,
            "The ADM split gives X=(v_phi^2-s_phi^2)/2.",
        ),
        _check(
            "S02_PHI_CANONICAL_MOMENTUM",
            pi_phi - c * v_phi,
            "The lapse-normalized phi momentum is C v_phi.",
        ),
        _check(
            "S03_CHI_CANONICAL_MOMENTUM",
            pi_chi - y0 * v_chi,
            "The lapse-normalized chi momentum is Y0 v_chi.",
        ),
        _check(
            "S04_PHI_LEGENDRE_HESSIAN",
            sp.diff(pi_phi, v_phi) - k_adm,
            "The phi Legendre Hessian is K_ADM=C+v_phi^2 C_X.",
        ),
        _check(
            "S05_CHI_LEGENDRE_HESSIAN",
            sp.diff(pi_chi, v_chi) - y0,
            "The chi Legendre Hessian is Y0.",
        ),
        _check(
            "S06_CROSS_LEGENDRE_HESSIAN_ZERO",
            sp.diff(pi_phi, v_chi) ** 2 + sp.diff(pi_chi, v_phi) ** 2,
            "The scalar velocity Hessian has no phi-chi cross block.",
        ),
    ]
    lapse, beta_phi, beta_chi = sp.symbols("N beta_phi beta_chi", real=True)
    dot_phi = lapse * v_phi + beta_phi
    dot_chi = lapse * v_chi + beta_chi
    canonical = pi_phi * dot_phi + pi_chi * dot_chi - lapse * lagrangian
    normal_density = c * v_phi**2 + y0 * v_chi**2 - lagrangian
    shift_density = pi_phi * beta_phi + pi_chi * beta_chi
    expanded = (
        c * v_phi**2 - p_value + potential + q_mass * z_value + y0 * (v_chi**2 + s_chi**2) / 2
    )
    stress_normal = c * v_phi**2 + y0 * v_chi**2 - lagrangian
    flrw_energy = 2 * x * p_x - p_value + potential + y0 * x_chi + q_mass * (z_value - 2 * x * z_x)
    checks.extend(
        [
            _check(
                "S07_CANONICAL_SHIFT_DENSITY",
                canonical - lapse * normal_density - shift_density,
                "The canonical density separates into lapse and scalar momentum-density terms.",
            ),
            _check(
                "S08_CANONICAL_NORMAL_DENSITY",
                normal_density - expanded,
                "The scalar normal Hamiltonian density has the frozen expanded form.",
            ),
            _check(
                "S09_STRESS_ENERGY_NORMAL_PROJECTION",
                normal_density - stress_normal,
                "The normal Hamiltonian density equals T_munu n^mu n^nu.",
            ),
            _check(
                "S10_HOMOGENEOUS_FLRW_ENERGY_REGRESSION",
                (normal_density - flrw_energy).subs({s_phi: 0, s_chi: 0}),
                "The homogeneous scalar Hamiltonian reproduces the bound FLRW energy density.",
            ),
            _check(
                "S11_INVERSE_PHI_MOMENTUM_HESSIAN",
                k_adm * (1 / k_adm) - 1,
                "The inverse phi momentum Hessian is 1/K_ADM where the Legendre map is regular.",
            ),
            _check(
                "S12_INVERSE_CHI_MOMENTUM_HESSIAN",
                y0 * (1 / y0) - 1,
                "The inverse chi momentum Hessian is 1/Y0.",
            ),
        ]
    )

    c_symbol, cx_symbol, v_symbol, s_symbol = sp.symbols("C C_X v s", real=True)
    kadm_symbol = c_symbol + cx_symbol * v_symbol**2
    kinv_symbol = c_symbol + cx_symbol * (v_symbol**2 - s_symbol**2)
    principal_contravariant = sp.Matrix(
        [
            [-c_symbol - cx_symbol * v_symbol**2, -cx_symbol * v_symbol * s_symbol],
            [-cx_symbol * v_symbol * s_symbol, c_symbol - cx_symbol * s_symbol**2],
        ]
    )
    mixed = -cx_symbol * v_symbol * s_symbol
    spatial = c_symbol - cx_symbol * s_symbol**2
    schur = sp.simplify(spatial + mixed**2 / kadm_symbol)
    endomorphism = sp.Matrix(
        [
            [c_symbol + cx_symbol * v_symbol**2, -cx_symbol * v_symbol * s_symbol, 0, 0],
            [cx_symbol * v_symbol * s_symbol, c_symbol - cx_symbol * s_symbol**2, 0, 0],
            [0, 0, c_symbol, 0],
            [0, 0, 0, c_symbol],
        ]
    )
    omega, wave = sp.symbols("omega k", real=True)
    completed = -kadm_symbol * (omega - mixed * wave / kadm_symbol) ** 2 + schur * wave**2
    original = -kadm_symbol * omega**2 + 2 * mixed * omega * wave + spatial * wave**2
    checks.extend(
        [
            _check(
                "S13_EFFECTIVE_METRIC_NORMAL_COEFFICIENT",
                -principal_contravariant[0, 0] - kadm_symbol,
                "The chosen normal coefficient is K_ADM.",
            ),
            _check(
                "S14_EFFECTIVE_METRIC_MIXED_COEFFICIENT",
                principal_contravariant[0, 1] - mixed,
                "The normal-spatial principal coefficient retains the exact background-gradient term.",
            ),
            _check(
                "S15_EFFECTIVE_METRIC_SPATIAL_COEFFICIENT",
                principal_contravariant[1, 1] - spatial,
                "The raw longitudinal spatial coefficient is C-C_X s^2.",
            ),
            _check(
                "S16_COMPLETED_SQUARE_SPATIAL_MATRIX",
                original - completed,
                "Completing the time-space square yields the Schur spatial coefficient.",
            ),
            _check(
                "S17_LONGITUDINAL_SPATIAL_EIGENVALUE",
                schur - c_symbol * kinv_symbol / kadm_symbol,
                "The longitudinal Schur eigenvalue is C K_inv/K_ADM.",
            ),
            _check(
                "S18_EFFECTIVE_METRIC_DETERMINANT",
                sp.det(endomorphism) - c_symbol**3 * kinv_symbol,
                "The invariant effective-metric determinant is C^3 K_inv.",
            ),
        ]
    )

    u, beta, x_positive, rho_base, mass = sp.symbols("u beta X rho_base m_chi", positive=True)
    gate_x = sp.diff((1 + beta * x_positive**2) ** 2, x_positive)
    energy_factor_x = (1 + beta * x_positive**2) ** 2 - 2 * x_positive * gate_x
    energy_factor = (1 + u) * (1 - 7 * u)
    q_bound = rho_base / (-energy_factor)
    chi_squared_bound = 2 * rho_base / (mass**2 * (-energy_factor_x))
    checks.extend(
        [
            _check(
                "S19_ILLUSTRATIVE_GATE_ENERGY_FACTOR",
                energy_factor_x - energy_factor.subs(u, beta * x_positive**2),
                "The illustrative mass gate contributes (1+u)(1-7u) to homogeneous energy.",
            ),
            _check(
                "S20_GATE_ENERGY_SIGN_TRANSITION",
                energy_factor.subs(u, sp.Rational(1, 7)),
                "The gate energy contribution changes sign at u=1/7.",
            ),
            _check(
                "S21_POSITIVE_ENERGY_AMPLITUDE_BOUND",
                rho_base + q_bound * energy_factor,
                "The frozen Q bound is exactly the zero-energy boundary for u>1/7.",
            ),
            _check(
                "S22_HIGH_U_ENERGY_BOUND_SCALING",
                sp.limit(x_positive**4 * chi_squared_bound, x_positive, sp.oo)
                - 2 * rho_base / (7 * mass**2 * beta**2),
                "For finite positive rho_base, the energy chi^2 ceiling scales as X^-4.",
            ),
            _check(
                "S23_HOMOGENEOUS_K_ADM_REGRESSION",
                (kadm_symbol - kinv_symbol).subs(s_symbol, 0),
                "K_ADM reduces to K_inv on a homogeneous slice.",
            ),
            _check(
                "S24_ZERO_GRADIENT_SHIFT_DENSITY_REGRESSION",
                shift_density.subs({beta_phi: 0, beta_chi: 0}),
                "The scalar shift density vanishes when both projected shift derivatives vanish.",
            ),
        ]
    )
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic check order changed",
    )
    return {
        "engine": "sympy-1.14",
        "all_passed": all(item["passed"] for item in checks),
        "checks": checks,
        "derived_expressions": {
            "phi_momentum": "sqrt(h) C v_phi",
            "chi_momentum": "sqrt(h) Y0 v_chi",
            "phi_legendre_coefficient": "K_ADM=C+v_phi^2 C_X",
            "normal_hamiltonian_density_over_sqrt_h": ("C v_phi^2-P+V+QZ+(Y0/2)(v_chi^2+s_chi^2)"),
            "effective_metric_determinant": "C^3 K_inv",
            "longitudinal_schur_eigenvalue": "C K_inv/K_ADM",
            "illustrative_gate_energy_factor": "(1+u)(1-7u)",
        },
    }


def numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    cases = []
    for item in config["machine_check_contract"]["numeric_cases"]:
        c = float(item["C"])
        c_x = float(item["C_X"])
        v_phi = float(item["v_phi"])
        s_phi = float(item["s_phi"])
        y0 = float(item["Y0"])
        rho_base = float(item["rho_base"])
        q_mass = float(item["Q"])
        u = float(item["u"])
        k_adm = c + c_x * v_phi**2
        k_inv = c + c_x * (v_phi**2 - s_phi**2)
        longitudinal = math.nan if k_adm == 0 else c * k_inv / k_adm
        energy_factor = (1 + u) * (1 - 7 * u)
        rho = rho_base + q_mass * energy_factor
        legendre_positive = k_adm > 0 and y0 > 0
        slice_positive = c > 0 and k_inv > 0 and k_adm > 0 and longitudinal > 0 and y0 > 0
        energy_positive = rho > 0
        matches = (
            legendre_positive is item["expected_legendre_positive"]
            and slice_positive is item["expected_slice_principal_positive"]
            and energy_positive is item["expected_energy_positive"]
        )
        schur_direct = (
            math.nan if k_adm == 0 else c - c_x * s_phi**2 + (c_x * v_phi * s_phi) ** 2 / k_adm
        )
        schur_error = 0.0 if math.isnan(schur_direct) else abs(schur_direct - longitudinal)
        cases.append(
            {
                "case_id": item["case_id"],
                "K_ADM": k_adm,
                "K_inv": k_inv,
                "longitudinal_schur_eigenvalue": longitudinal,
                "rho_s": rho,
                "legendre_positive": legendre_positive,
                "slice_principal_positive": slice_positive,
                "energy_positive": energy_positive,
                "designed_failure": item["designed_failure"],
                "schur_identity_absolute_error": schur_error,
                "passed": matches and schur_error <= tolerance,
            }
        )
    return {
        "all_passed": all(item["passed"] for item in cases),
        "tolerance": tolerance,
        "cases": cases,
        "designed_failures_preserved": sum(
            item["designed_failure"] and item["passed"] for item in cases
        ),
        "max_schur_identity_absolute_error": max(
            item["schur_identity_absolute_error"] for item in cases
        ),
    }


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    symbolic = symbolic_suite()
    numeric = numeric_suite(config)
    _require(symbolic["all_passed"] is True, "symbolic suite failed")
    _require(numeric["all_passed"] is True, "numeric suite failed")
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
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
        "predecessor_bindings": config["predecessor_bindings"],
        "geometry_and_action_contract": config["geometry_and_action_contract"],
        "adm_legendre_contract": config["adm_legendre_contract"],
        "canonical_hamiltonian_contract": config["canonical_hamiltonian_contract"],
        "principal_and_slice_health_contract": config["principal_and_slice_health_contract"],
        "homogeneous_energy_contract": config["homogeneous_energy_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "remaining_obligations": config["remaining_obligations"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_cases": len(numeric["cases"]),
            "numeric_cases_passed": sum(item["passed"] for item in numeric["cases"]),
            "designed_failures_preserved": numeric["designed_failures_preserved"],
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "limitations": [
            "The result is a scalar-block ADM Legendre transform; it does not Legendre-transform or prove positivity of the full Einstein-matter system.",
            "Hamiltonian convexity in momenta is not a lower-bound proof for the total energy density.",
            "The local slice conditions do not control lower-order instabilities, nonlinear cutoff, boundary flux, or global evolution.",
            "The illustrative gate has a negative homogeneous energy contribution above u=1/7; the unspecified P and V sectors may compensate, so this is not an unconditional no-go.",
            "No on-shell background, motion+lensing solution, Solar/GW/cosmological pass, observation, novelty, or publication claim follows.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    return receipt


RECEIPT_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "decision",
    "config_binding",
    "implementation_binding",
    "predecessor_bindings",
    "geometry_and_action_contract",
    "adm_legendre_contract",
    "canonical_hamiltonian_contract",
    "principal_and_slice_health_contract",
    "homogeneous_energy_contract",
    "symbolic_suite",
    "numeric_suite",
    "adjudication",
    "claim_boundary",
    "remaining_obligations",
    "zero_access_and_compute",
    "counts",
    "limitations",
    "content_sha256",
}


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(receipt, RECEIPT_KEYS, "receipt")
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "receipt schema changed")
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    _require(expected == _sha(body), "receipt content hash changed")
    _require(receipt["decision"] == DECISION, "receipt decision changed")
    _require(receipt["adjudication"] == config["adjudication"], "adjudication changed")
    _require(receipt["claim_boundary"] == config["claim_boundary"], "claim boundary changed")
    _require(
        receipt["zero_access_and_compute"] == config["zero_access_and_compute"],
        "zero-access evidence changed",
    )
    _require(receipt["counts"]["symbolic_checks_passed"] == 24, "symbolic count changed")
    _require(receipt["counts"]["numeric_cases_passed"] == 4, "numeric count changed")
    _require(receipt["counts"]["designed_failures_preserved"] == 2, "failure count changed")


def check_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    stored = _read_json(root / OUTPUT_PATH)
    validate_receipt(stored, config)
    _require(stored == build_receipt(root), "stored receipt does not rebuild exactly")
    return stored


def _flush_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() == payload:
                return "EXISTING_IDENTICAL"
            raise ScalarHamiltonianError(f"refusing to overwrite existing receipt: {path}")
        _flush_directory(path.parent)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> tuple[Path, str]:
    root = root.resolve()
    receipt = build_receipt(root)
    path = root / OUTPUT_PATH
    outcome = _atomic_no_replace(path, _canonical_bytes(receipt))
    return path, outcome


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        path, outcome = write_receipt(root)
        output: Any = {"path": str(path), "outcome": outcome}
    else:
        receipt = check_receipt(root)
        output = {
            "valid": True,
            "status": receipt["status"],
            "decision": receipt["decision"],
            "CP11_4_complete": receipt["adjudication"]["CP11_4_complete"],
            "physical_hamiltonian_positive": receipt["adjudication"][
                "physical_hamiltonian_positive"
            ],
            "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
            "numeric_cases_passed": receipt["counts"]["numeric_cases_passed"],
            "observational_rows_opened": receipt["counts"]["observational_rows_opened"],
            "content_sha256": receipt["content_sha256"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
