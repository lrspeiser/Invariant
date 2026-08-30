"""Restricted no-data derivation of the constant-kinetic split-gate scalar action."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_split_gate_action_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_split_gate_action.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_split_gate_action.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-split-gate-action-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-split-gate-action-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-split-gate-action-receipt-1.0"
DECISION = (
    "PARTIAL_SPLIT_GATE_SCALAR_ACTION_DERIVED_KINETIC_MIXING_AVOIDED_"
    "TIMELIKE_CHI_AMPLITUDE_RISK_PRESERVED"
)
EXPECTED_CONFIG_FILE_SHA256 = "f16f1d711762fea34fb7c53de1fa559c69423be6b6bdc938063e313e30a74035"
EXPECTED_CONFIG_CONTENT_SHA256 = "cb6d58146f3bda3409cf4207ee3dac9e25387d7da28f4a04a815f50ecdc06d15"

SYMBOLIC_CHECK_IDS = (
    "S01_HESSIAN_PHI_BLOCK",
    "S02_HESSIAN_CHI_BLOCK",
    "S03_HESSIAN_CROSS_BLOCK_ZERO",
    "S04_FLUX_PHI_BLOCK",
    "S05_FLUX_CHI_BLOCK",
    "S06_FLUX_CROSS_BLOCK_ZERO",
    "S07_C_EXACT",
    "S08_C_X_EXACT",
    "S09_CHI_EOM_MASS_TERM",
    "S10_Z_X_EXACT",
    "S11_Z_XX_EXACT",
    "S12_H_EXACT",
    "S13_BARE_FROZEN_PHI_DISPERSION",
    "S14_TIMELIKE_C_BOUNDARY",
    "S15_TIMELIKE_K_BOUNDARY",
    "S16_X_ZERO_GATE_LIMIT",
    "S17_X_ZERO_PRINCIPAL_CORRECTION_LIMIT",
    "S18_BARE_FROZEN_PHI_HIGH_U_RANGE_SCALING",
    "S19_CONSTANT_Y_NO_GO_COMBINATION_ZERO",
    "S20_STRESS_COEFFICIENT_FROM_METRIC_RESPONSE",
    "S21_FIRST_DERIVATIVE_MIXING_COEFFICIENT",
    "S22_RECIPROCAL_FIRST_DERIVATIVE_MIXING",
    "S23_COUPLED_DISPERSION_DETERMINANT",
    "S24_MIXING_VANISHING_LIMIT",
    "S25_HIGH_U_CHI_SQUARED_SCALING",
    "S26_HIGH_U_CHI_AMPLITUDE_SCALING",
)


class SplitGateActionError(RuntimeError):
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
        raise SplitGateActionError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SplitGateActionError(message)


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
    metric = sp.diag(-1, 1)
    p = sp.Matrix(sp.symbols("p0:2", real=True))
    q = sp.Matrix(sp.symbols("q0:2", real=True))
    p_up = metric * p
    q_up = metric * q
    x = -sp.Rational(1, 2) * (p.T * metric * p)[0]
    x_chi = -sp.Rational(1, 2) * (q.T * metric * q)[0]
    beta, y0, mass = sp.symbols("beta Y0 m_chi", positive=True)
    chi = sp.symbols("chi", real=True)
    p0_l, p1, p2 = sp.symbols("P0 P1 P2", real=True)
    u = beta * x**2
    gate = (1 + u) ** 2
    p_lagrangian = p0_l + p1 * x + p2 * x**2 / 2
    q_mass = mass**2 * chi**2 / 2
    lagrangian = p_lagrangian + y0 * x_chi - q_mass * gate
    gate_x = 4 * beta * x * (1 + u)
    gate_xx = 4 * beta * (1 + 3 * u)
    c = p1 + p2 * x - q_mass * gate_x
    c_x = p2 - q_mass * gate_xx

    h_pp = sp.hessian(lagrangian, tuple(p))
    h_qq = sp.hessian(lagrangian, tuple(q))
    h_pq = sp.Matrix(2, 2, lambda i, j: sp.diff(lagrangian, p[i], q[j]))
    expected_pp = sp.Matrix(2, 2, lambda i, j: c * metric[i, j] - c_x * p_up[i] * p_up[j])
    expected_qq = y0 * metric

    phi_flux = c * p_up
    chi_flux = y0 * q_up
    flux_pp = sp.Matrix(2, 2, lambda i, j: sp.diff(phi_flux[i], p[j]))
    flux_pq = sp.Matrix(2, 2, lambda i, j: sp.diff(phi_flux[i], q[j]))
    flux_qp = sp.Matrix(2, 2, lambda i, j: sp.diff(chi_flux[i], p[j]))
    flux_qq = sp.Matrix(2, 2, lambda i, j: sp.diff(chi_flux[i], q[j]))

    x_symbol, beta_symbol = sp.symbols("X beta_symbol", real=True)
    u_symbol = beta_symbol * x_symbol**2
    z_symbol = (1 + u_symbol) ** 2
    zx_symbol = sp.diff(z_symbol, x_symbol)
    zxx_symbol = sp.diff(zx_symbol, x_symbol)
    h_symbol = sp.simplify(zx_symbol + 2 * x_symbol * zxx_symbol)
    expected_zx = 4 * beta_symbol * x_symbol * (1 + u_symbol)
    expected_zxx = 4 * beta_symbol * (1 + 3 * u_symbol)
    expected_h = 4 * beta_symbol * x_symbol * (3 + 7 * u_symbol)

    omega, wave, z_local = sp.symbols("omega k Z", positive=True)
    dispersion = y0 * (-(omega**2) + wave**2) + mass**2 * z_local
    omega_squared = wave**2 + mass**2 * z_local / y0

    a0, b0, amplitude = sp.symbols("A0 B0 chi2", positive=True)
    c_abstract = a0 - mass**2 * amplitude * zx_symbol / 2
    k_abstract = b0 - mass**2 * amplitude * h_symbol / 2
    c_boundary = 2 * a0 / (mass**2 * zx_symbol)
    k_boundary = 2 * b0 / (mass**2 * h_symbol)

    u_free = sp.symbols("u_free", positive=True)
    bare_range_expression = sp.sqrt(y0) / (mass * (1 + u_free))
    constant_gate = sp.symbols("Y_const", positive=True)
    yx = sp.diff(constant_gate, x_symbol)
    yxx = sp.diff(constant_gate, x_symbol, 2)
    no_go_mixing = constant_gate * (yx + 2 * x_symbol * yxx) - 4 * x_symbol * yx**2

    x_free = sp.symbols("X_free", real=True)
    p_function = sp.Function("P")
    z_function = sp.Function("Z")
    abstract_lagrangian = p_function(x_free) - mass**2 * chi**2 * z_function(x_free) / 2
    abstract_c = (
        sp.diff(p_function(x_free), x_free)
        - mass**2 * chi**2 * sp.diff(z_function(x_free), x_free) / 2
    )

    mixing_d = mass**2 * chi * gate_x
    chi_eom_lower = -(mass**2) * gate * chi
    reciprocal_mixing = sp.Matrix([sp.diff(chi_eom_lower, p[index]) for index in range(2)])
    expected_reciprocal = mixing_d * p_up
    d_symbol, d_phi, d_chi, vk_symbol = sp.symbols("d D_phi D_chi vk_symbol", real=True)
    coupled_matrix = sp.Matrix(
        [
            [d_phi, -sp.I * d_symbol * vk_symbol],
            [sp.I * d_symbol * vk_symbol, d_chi],
        ]
    )
    expected_coupled_determinant = d_phi * d_chi - d_symbol**2 * vk_symbol**2

    x_positive, a_positive, b_positive = sp.symbols("X_pos A_pos B_pos", positive=True)
    zx_positive = 4 * beta * x_positive * (1 + beta * x_positive**2)
    h_positive = 4 * beta * x_positive * (3 + 7 * beta * x_positive**2)
    bound_c_positive = 2 * a_positive / (mass**2 * zx_positive)
    bound_k_positive = 2 * b_positive / (mass**2 * h_positive)
    c_squared_limit = a_positive / (2 * mass**2 * beta**2)
    k_squared_limit = b_positive / (14 * mass**2 * beta**2)

    checks = [
        _check(
            "S01_HESSIAN_PHI_BLOCK",
            -h_pp - expected_pp,
            "The exact negative gradient Hessian gives C*g-C_X*v*v.",
        ),
        _check(
            "S02_HESSIAN_CHI_BLOCK",
            -h_qq - expected_qq,
            "Constant Y0 gives the canonical scaled chi principal tensor.",
        ),
        _check(
            "S03_HESSIAN_CROSS_BLOCK_ZERO", h_pq, "The gradient Hessian has no phi-chi cross block."
        ),
        _check(
            "S04_FLUX_PHI_BLOCK",
            flux_pp - expected_pp,
            "Independent phi-flux linearization agrees with the Hessian route.",
        ),
        _check(
            "S05_FLUX_CHI_BLOCK",
            flux_qq - expected_qq,
            "Independent chi-flux linearization agrees with the Hessian route.",
        ),
        _check(
            "S06_FLUX_CROSS_BLOCK_ZERO",
            (flux_pq, flux_qp),
            "Both flux cross-principal blocks vanish.",
        ),
        _check("S07_C_EXACT", c - (p1 + p2 * x - q_mass * gate_x), "C=P_X-Q*Z_X exactly."),
        _check("S08_C_X_EXACT", c_x - (p2 - q_mass * gate_xx), "C_X=P_XX-Q*Z_XX exactly."),
        _check(
            "S09_CHI_EOM_MASS_TERM",
            sp.diff(lagrangian, chi) + mass**2 * gate * chi,
            "The chi Euler derivative supplies -m_chi^2*Z*chi.",
        ),
        _check("S10_Z_X_EXACT", zx_symbol - expected_zx, "The illustrative gate has exact Z_X."),
        _check(
            "S11_Z_XX_EXACT", zxx_symbol - expected_zxx, "The illustrative gate has exact Z_XX."
        ),
        _check(
            "S12_H_EXACT",
            h_symbol - expected_h,
            "The second principal combination is H=4*beta*X*(3+7u).",
        ),
        _check(
            "S13_BARE_FROZEN_PHI_DISPERSION",
            dispersion.subs(omega**2, omega_squared),
            "For frozen delta_phi or d*(v.k)=0, the bare mediator mass parameter is m_chi^2*Z/Y0; this is not the general coupled dispersion.",
        ),
        _check(
            "S14_TIMELIKE_C_BOUNDARY",
            c_abstract.subs(amplitude, c_boundary),
            "The first timelike amplitude equality is the C=0 boundary.",
        ),
        _check(
            "S15_TIMELIKE_K_BOUNDARY",
            k_abstract.subs(amplitude, k_boundary),
            "The second timelike amplitude equality is the K=0 boundary.",
        ),
        _check(
            "S16_X_ZERO_GATE_LIMIT",
            z_symbol.subs(x_symbol, 0) - 1,
            "The gate returns to unity at X=0.",
        ),
        _check(
            "S17_X_ZERO_PRINCIPAL_CORRECTION_LIMIT",
            (zx_symbol.subs(x_symbol, 0), h_symbol.subs(x_symbol, 0)),
            "Both gate corrections vanish at X=0.",
        ),
        _check(
            "S18_BARE_FROZEN_PHI_HIGH_U_RANGE_SCALING",
            sp.limit(u_free * bare_range_expression, u_free, sp.oo) - sp.sqrt(y0) / mass,
            "Only the bare/frozen-delta-phi high-u range scales as 1/u; the general coupled poles use the full determinant.",
        ),
        _check(
            "S19_CONSTANT_Y_NO_GO_COMBINATION_ZERO",
            no_go_mixing,
            "Constant Y makes the prior kinetic mixing combination identically zero.",
        ),
        _check(
            "S20_STRESS_COEFFICIENT_FROM_METRIC_RESPONSE",
            sp.diff(abstract_lagrangian, x_free) - abstract_c,
            "The coefficient multiplying d_phi*d_phi in the metric response is C.",
        ),
        _check(
            "S21_FIRST_DERIVATIVE_MIXING_COEFFICIENT",
            sp.diff(c, chi) + mixing_d,
            "The phi flux retains C_chi=-d with d=m_chi^2*chi*Z_X.",
        ),
        _check(
            "S22_RECIPROCAL_FIRST_DERIVATIVE_MIXING",
            reciprocal_mixing - expected_reciprocal,
            "The chi equation carries the reciprocal +d*v dot derivative coupling.",
        ),
        _check(
            "S23_COUPLED_DISPERSION_DETERMINANT",
            sp.det(coupled_matrix) - expected_coupled_determinant,
            "The frozen Fourier matrix has determinant D_phi*D_chi-d^2*(v.k)^2.",
        ),
        _check(
            "S24_MIXING_VANISHING_LIMIT",
            coupled_matrix.subs(d_symbol, 0) - sp.diag(d_phi, d_chi),
            "The bare mediator pole is recovered only when the first-derivative mixing vanishes.",
        ),
        _check(
            "S25_HIGH_U_CHI_SQUARED_SCALING",
            (
                sp.limit(x_positive**3 * bound_c_positive, x_positive, sp.oo) - c_squared_limit,
                sp.limit(x_positive**3 * bound_k_positive, x_positive, sp.oo) - k_squared_limit,
            ),
            "Both timelike chi-squared ceilings scale as X^-3 with exact branch constants.",
        ),
        _check(
            "S26_HIGH_U_CHI_AMPLITUDE_SCALING",
            (
                sp.limit(
                    x_positive ** sp.Rational(3, 2) * sp.sqrt(bound_c_positive),
                    x_positive,
                    sp.oo,
                )
                - sp.sqrt(c_squared_limit),
                sp.limit(
                    x_positive ** sp.Rational(3, 2) * sp.sqrt(bound_k_positive),
                    x_positive,
                    sp.oo,
                )
                - sp.sqrt(k_squared_limit),
            ),
            "Both absolute chi ceilings scale as X^-3/2 with exact branch constants.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic derivation failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "derivation_routes": [
            "negative exact quadratic gradient Hessian",
            "independent Euler-Lagrange flux linearization",
        ],
        "checks": checks,
        "all_passed": True,
        "derived_expressions": {
            "C": "P_X-m_chi^2*chi^2*Z_X/2",
            "C_X": "P_XX-m_chi^2*chi^2*Z_XX/2",
            "cross_principal": "0",
            "Z_X": "4*beta*X*(1+u)",
            "Z_XX": "4*beta*(1+3*u)",
            "H": "4*beta*X*(3+7*u)",
            "bare_frozen_phi_range": "sqrt(Y0)/(m_chi*(1+u)) when d*(v.k)=0",
            "first_derivative_mixing": "d=m_chi^2*chi*Z_X",
            "coupled_determinant": "D_phi*D_chi-d^2*(v.k)^2",
            "high_u_chi_squared": "chi_max^2 proportional to X^-3",
            "high_u_chi_amplitude": "abs(chi)_max proportional to X^-3/2",
        },
    }


def _sign(value: float, tolerance: float) -> str:
    if abs(value) <= tolerance:
        return "zero"
    return "positive" if value > 0 else "negative"


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    records: list[dict[str, Any]] = []
    for case in config["machine_check_contract"]["frozen_numeric_cases"]:
        x = float(case["X"])
        chi = float(case["chi"])
        u = x**2
        z = (1.0 + u) ** 2
        zx = 4.0 * x * (1.0 + u)
        h = 4.0 * x * (3.0 + 7.0 * u)
        q_mass = chi**2 / 2.0
        c = 1.0 - q_mass * zx
        k = 1.0 - q_mass * h
        zxx = 4.0 * (1.0 + 3.0 * u)
        c_x = -q_mass * zxx
        mixing_d = chi * zx
        vk_squared = abs(x) / 2.0
        d_phi = c - c_x * vk_squared
        d_chi = 1.0 + z
        bare_product = d_phi * d_chi
        coupled_determinant = bare_product - mixing_d**2 * vk_squared
        bare_range = 1.0 / math.sqrt(z)
        c_sign = _sign(c, tolerance)
        k_sign = _sign(k, tolerance)
        mixing_class = "zero" if abs(mixing_d) <= tolerance else "nonzero"
        coupled_sign = _sign(coupled_determinant, tolerance)
        passed = (
            c_sign == case["expected_C_sign"]
            and k_sign == case["expected_K_sign"]
            and mixing_class == case["expected_first_derivative_mixing"]
            and coupled_sign == case["expected_coupled_determinant_sign"]
        )
        records.append(
            {
                "case_id": case["case_id"],
                "X": format(x, ".17g"),
                "u": format(u, ".17g"),
                "chi": format(chi, ".17g"),
                "C": format(c, ".17g"),
                "K": format(k, ".17g"),
                "C_sign": c_sign,
                "K_sign": k_sign,
                "first_derivative_mixing_d": format(mixing_d, ".17g"),
                "first_derivative_mixing_class": mixing_class,
                "probe_vk_squared": format(vk_squared, ".17g"),
                "bare_diagonal_product": format(bare_product, ".17g"),
                "coupled_determinant": format(coupled_determinant, ".17g"),
                "coupled_determinant_sign": coupled_sign,
                "mixing_determinant_shift": format(-(mixing_d**2) * vk_squared, ".17g"),
                "bare_frozen_phi_range": format(bare_range, ".17g"),
                "designed_failure": case["designed_failure"],
                "passed": passed,
            }
        )
    _require(all(item["passed"] for item in records), "numeric suite failed")
    _require(
        sum(item["designed_failure"] for item in records) == 3, "designed failure inventory changed"
    )
    _require(
        all(item["K_sign"] == "negative" for item in records if item["designed_failure"]),
        "designed failure lost",
    )
    high_u_records: list[dict[str, Any]] = []
    previous_c_error = math.inf
    previous_k_error = math.inf
    for x_probe in config["machine_check_contract"]["high_u_scaling_probes"]:
        x_value = float(x_probe)
        zx_value = 4.0 * x_value * (1.0 + x_value**2)
        h_value = 4.0 * x_value * (3.0 + 7.0 * x_value**2)
        chi_c_squared = 2.0 / zx_value
        chi_k_squared = 2.0 / h_value
        scaled_c_squared = x_value**3 * chi_c_squared
        scaled_k_squared = x_value**3 * chi_k_squared
        scaled_c_amplitude = x_value**1.5 * math.sqrt(chi_c_squared)
        scaled_k_amplitude = x_value**1.5 * math.sqrt(chi_k_squared)
        c_error = abs(scaled_c_squared - 0.5)
        k_error = abs(scaled_k_squared - 1.0 / 14.0)
        passed = (
            c_error < previous_c_error
            and k_error < previous_k_error
            and abs(scaled_c_amplitude**2 - scaled_c_squared) <= tolerance
            and abs(scaled_k_amplitude**2 - scaled_k_squared) <= tolerance
        )
        high_u_records.append(
            {
                "X": format(x_value, ".17g"),
                "X_cubed_chi_C_max_squared": format(scaled_c_squared, ".17g"),
                "X_cubed_chi_K_max_squared": format(scaled_k_squared, ".17g"),
                "X_three_halves_abs_chi_C_max": format(scaled_c_amplitude, ".17g"),
                "X_three_halves_abs_chi_K_max": format(scaled_k_amplitude, ".17g"),
                "approaches_frozen_limits": passed,
            }
        )
        previous_c_error = c_error
        previous_k_error = k_error
    _require(
        all(item["approaches_frozen_limits"] for item in high_u_records),
        "high-u numeric scaling regression failed",
    )
    return {
        "parameters": config["machine_check_contract"]["numeric_parameters"],
        "tolerance": format(tolerance, ".17g"),
        "cases": records,
        "high_u_scaling_regression": high_u_records,
        "all_passed": True,
        "designed_failures_preserved": 3,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "analysis_id",
            "status",
            "purpose",
            "predecessor_receipt_bindings",
            "action_contract",
            "eom_and_principal_contract",
            "range_and_limits",
            "local_health_contract",
            "no_go_adjudication",
            "stress_conservation_and_lensing",
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
        config["analysis_id"] == "gravity-matter-lensing-split-gate-action-v1",
        "analysis identity changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    bindings = config["predecessor_receipt_bindings"]
    _require(
        tuple(item["binding_id"] for item in bindings)
        == ("theory_preflight", "conditional_kinetic_gate_no_go", "remedy_comparison"),
        "predecessor inventory changed",
    )
    binding_keys = {
        "binding_id",
        "git_commit",
        "receipt_path",
        "receipt_file_sha256",
        "receipt_content_sha256",
        "receipt_schema_version",
        "receipt_decision",
    }
    for binding in bindings:
        _strict(binding, binding_keys, f"predecessor {binding.get('binding_id')}")
    action = config["action_contract"]
    _require(
        action["scalar_action"]
        == "S_scalar=integral sqrt(-g)*[P(X)-V(phi)+Y0*X_chi-m_chi^2*Z(u)*chi^2/2]",
        "action changed",
    )
    _require(action["illustrative_gate"] == "Z(u)=(1+u)^2", "gate changed")
    _require(
        action["novelty_label"]
        == "KNOWN_FORM_REUSE_AND_STRUCTURAL_RECOMBINATION_NOT_A_NOVELTY_CLAIM",
        "novelty label changed",
    )
    _require(
        "one non-derivative" in action["universal_matter_action"]
        and "No independent photon multiplier" in action["forbidden"],
        "universal metric boundary changed",
    )
    eom = config["eom_and_principal_contract"]
    _require(
        eom["cross_principal"].startswith("The second-order blocks P_phichi=P_chiphi=0 exactly"),
        "cross-principal claim changed",
    )
    _require(
        "d=m_chi^2*chi*Z_X" in eom["first_derivative_mixing"]
        and "D_phi*D_chi-d^2*(v.k)^2" in eom["coupled_local_dispersion"],
        "first-derivative mixing contract changed",
    )
    _require("-Q*Z_X" in eom["C"] and "-Q*Z_XX" in eom["C_X"], "self-principal correction changed")
    health = config["local_health_contract"]
    _require(
        tuple(health["necessary_external_scalar_conditions"])
        == (
            "Y0>0",
            "Z>0 and finite bare/frozen-delta-phi mass parameter m_chi^2*Z/Y0",
            "C>0",
            "K>0",
        ),
        "health conditions changed",
    )
    _require("chi^2<min" in health["timelike_amplitude_bounds"], "timelike bound changed")
    _require(
        "chi_max^2 proportional to X^-3" in health["high_u_amplitude_scaling"]
        and "|chi|_max proportional to X^-3/2" in health["high_u_amplitude_scaling"],
        "high-u amplitude scaling changed",
    )
    _require(
        "imposes no upper chi-amplitude bound" in health["quasistatic_spacelike_bounds"],
        "spacelike bound changed",
    )
    no_go = config["no_go_adjudication"]
    _require("M_Y=0 identically" in no_go["constant_Y_result"], "no-go result changed")
    _require(
        "does not establish a healthy action" in no_go["replacement_risk"],
        "replacement risk removed",
    )
    _require(
        "first-derivative mode mixing" in no_go["replacement_risk"],
        "first-derivative replacement risk removed",
    )
    range_contract = config["range_and_limits"]
    _require(
        "only for frozen delta_phi" in range_contract["bare_range"]
        and "only for frozen delta_phi" in range_contract["bare_dispersion"]
        and "not the full coupled range" in range_contract["high_u_range"],
        "bare-range caveat changed",
    )
    stress = config["stress_conservation_and_lensing"]
    _require(
        "same tilde_g" in stress["same_action_lensing_boundary"]
        and "no independent lensing" in stress["same_action_lensing_boundary"],
        "lensing boundary changed",
    )
    machine = config["machine_check_contract"]
    _require(
        tuple(machine["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic contract changed",
    )
    _require(
        len(machine["frozen_numeric_cases"]) == 6
        and sum(item["designed_failure"] for item in machine["frozen_numeric_cases"]) == 3,
        "numeric cases changed",
    )
    _require(
        tuple(machine["high_u_scaling_probes"]) == (10.0, 20.0, 40.0),
        "high-u numeric probes changed",
    )
    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["specific_kinetic_gate_no_go_avoided"] is True
        and adjudication["timelike_chi_amplitude_risk_found"] is True,
        "bounded adjudication changed",
    )
    _require(
        adjudication["first_derivative_mode_mixing_retained"] is True
        and adjudication["bare_range_is_full_coupled_range"] is False,
        "mixing/range adjudication changed",
    )
    _require(
        all(
            adjudication[key] is False
            for key in (
                "full_H3",
                "full_H4",
                "metric_constraints",
                "on_shell_backgrounds",
                "Solar_screening",
                "lensing_prediction",
                "cosmology",
                "global_strong_hyperbolicity",
                "EFT_cutoff_established",
                "full_health_established",
                "novelty_established",
            )
        ),
        "full gate unlocked",
    )
    claims = config["claim_boundary"]
    _require(
        claims["restricted_external_metric_scalar_derivation_completed"] is True
        and claims["specific_kinetic_mixing_no_go_avoided"] is True,
        "restricted result disabled",
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key
            not in {
                "restricted_external_metric_scalar_derivation_completed",
                "specific_kinetic_mixing_no_go_avoided",
            }
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


def _validate_predecessors(root: Path, config: Mapping[str, Any]) -> None:
    for binding in config["predecessor_receipt_bindings"]:
        path = root / binding["receipt_path"]
        _require(path.is_file(), f"predecessor missing: {binding['binding_id']}")
        _require(
            _file_sha(path) == binding["receipt_file_sha256"],
            f"predecessor changed: {binding['binding_id']}",
        )
        receipt = _read_json(path)
        _require(
            receipt.get("schema_version") == binding["receipt_schema_version"],
            f"predecessor schema changed: {binding['binding_id']}",
        )
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            f"predecessor content changed: {binding['binding_id']}",
        )
        _require(
            receipt.get("decision") == binding["receipt_decision"],
            f"predecessor decision changed: {binding['binding_id']}",
        )


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    _validate_predecessors(root, config)
    _require((root / SOURCE_PATH).is_file(), "implementation missing")
    _require((root / TEST_PATH).is_file(), "test missing")
    symbolic = run_symbolic_suite()
    numeric = run_numeric_suite(config)
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "analysis_id": config["analysis_id"],
        "status": "restricted_split_gate_action_machine_derived_designed_failures_preserved",
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
        "predecessor_receipt_bindings": config["predecessor_receipt_bindings"],
        "action_contract": config["action_contract"],
        "eom_and_principal_contract": config["eom_and_principal_contract"],
        "range_and_limits": config["range_and_limits"],
        "local_health_contract": config["local_health_contract"],
        "no_go_adjudication": config["no_go_adjudication"],
        "stress_conservation_and_lensing": config["stress_conservation_and_lensing"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_cases": len(numeric["cases"]),
            "numeric_cases_passed": sum(item["passed"] for item in numeric["cases"]),
            "high_u_scaling_probes": len(numeric["high_u_scaling_regression"]),
            "high_u_scaling_probes_passed": sum(
                item["approaches_frozen_limits"] for item in numeric["high_u_scaling_regression"]
            ),
            "designed_failures_preserved": numeric["designed_failures_preserved"],
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The derivation is the two-scalar block on a fixed external metric and local constant jets; the metric constraints and matter characteristics are absent.",
            "Constant Y0 exactly removes the prior second-order derivative cross-mixing combination, but the mass gate leaves chi-amplitude corrections in the phi self-principal tensor and first-derivative mode mixing.",
            "The quoted sqrt(Y0/Z)/m_chi range is only the bare frozen-phi or mixing-vanishing range; coupled poles obey D_phi*D_chi-d^2*(v.k)^2=0.",
            "The illustrative Z=(1+u)^2 is a known-form reuse and structural recombination, not a novelty claim or a fitted law.",
            "The timelike C=0 or K=0 boundaries warn of principal degeneration and possible strong coupling; no nonlinear cutoff has been computed.",
            "The favorable local spacelike signs do not establish causality, subluminality, positive energy, source solutions, or full hyperbolicity.",
            "No GR, Solar, galaxy, cluster, cosmology, lensing, observational, or publication claim is established.",
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
            "predecessor_receipt_bindings",
            "action_contract",
            "eom_and_principal_contract",
            "range_and_limits",
            "local_health_contract",
            "no_go_adjudication",
            "stress_conservation_and_lensing",
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
        receipt["predecessor_receipt_bindings"] == config["predecessor_receipt_bindings"],
        "receipt predecessors changed",
    )
    _require(
        receipt["action_contract"] == config["action_contract"]
        and receipt["local_health_contract"] == config["local_health_contract"],
        "action or health contract changed",
    )
    _require(
        receipt["adjudication"] == config["adjudication"]
        and receipt["claim_boundary"] == config["claim_boundary"],
        "claims changed",
    )
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 26,
        "symbolic count changed",
    )
    _require(
        counts["numeric_cases"] == counts["numeric_cases_passed"] == 6, "numeric count changed"
    )
    _require(counts["designed_failures_preserved"] == 3, "designed failures changed")
    _require(
        counts["high_u_scaling_probes"] == counts["high_u_scaling_probes_passed"] == 3,
        "high-u scaling count changed",
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
        raise SplitGateActionError(f"refusing to overwrite different receipt: {path}")
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
            raise SplitGateActionError(f"concurrent creator won; output preserved: {path}") from exc
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
        "numeric_cases_passed": receipt["counts"]["numeric_cases_passed"],
        "designed_failures_preserved": receipt["counts"]["designed_failures_preserved"],
        "specific_kinetic_no_go_avoided": receipt["adjudication"][
            "specific_kinetic_gate_no_go_avoided"
        ],
        "full_health_established": receipt["adjudication"]["full_health_established"],
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
    except SplitGateActionError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
