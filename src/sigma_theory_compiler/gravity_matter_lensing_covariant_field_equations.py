"""No-data covariant field-equation and scalar metric-variation package."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_covariant_field_equations_v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_covariant_field_equations.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_covariant_field_equations.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-covariant-field-equations-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-covariant-field-equations-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-covariant-field-equations-receipt-1.0"
DECISION = (
    "PARTIAL_COVARIANT_SCALAR_STRESS_FIELD_EQUATIONS_AND_EXCHANGE_IDENTITY_DERIVED_"
    "FULL_METRIC_DYNAMICS_HEALTH_AND_PHYSICS_UNESTABLISHED"
)
EXPECTED_CONFIG_FILE_SHA256 = "e0bd786c41779e47a79b08c4182315669751ac291fce84f49fb9c3d8ee918644"
EXPECTED_CONFIG_CONTENT_SHA256 = "52febf9a9b74d87e8fff208800b59d92258acea262a97817dfc1dbd499e4c894"
EXPECTED_SECTION_SHA256 = {
    "predecessor_bindings": "b872959f817e736ac9ed5a36e6cc501e27b47d8a7eeeaf82d238bfd94152e050",
    "action_and_variation_conventions": "fbbdf498181876d4f62f706edd5e614827d014b6e0c6160f0e7ecc4a328a3ce5",
    "covariant_field_equation_contract": "02888fca2be7d7aa53a9f46bc34f9999b60e0865d7c42c4fac4bd9edcf9e1672",
    "same_action_exchange_contract": "c46cbbf9ee05c2237ee6e4f3271351ecbd2bc6e4f4d3dcd70e0fd5fc24863ec7",
    "flrw_regression_contract": "6adf074cf978318d239318a70915c43b23aa09f5c90fdac1d6f7a4cab3e2c9dd",
    "machine_check_contract": "e8d6d22cf3907132be535471b48b361a91764af1a6db2eb2b7de5a0ab08c4f49",
    "adjudication": "46ec96972f2af8e8900d1c17d0938b1b2619b0dde60566aed0770480110347b2",
    "claim_boundary": "d6ef349c464788c18761f23720724f3947ec474bf36cff947118a3ec4193e2d8",
    "remaining_obligations": "f3b4888bbd02ebf15a21c876ac4932653d304b821e7a50e518591999a27cbe61",
    "zero_access_and_compute": "6ec9a22001ae649681dc7e72fcd18da49369801daa7112592105628bdf1ff705",
}

SYMBOLIC_CHECK_IDS = (
    "S01_C_FROM_X_DERIVATIVE",
    "S02_CHI_KINETIC_DERIVATIVE",
    "S03_PHI_EULER_POTENTIAL_SIGN",
    "S04_CHI_EULER_MASS_SIGN",
    "S05_METRIC_VARIATION_T00",
    "S06_METRIC_VARIATION_T01",
    "S07_METRIC_VARIATION_T11",
    "S08_STRESS_SYMMETRY",
    "S09_SCALAR_TRACE_4D",
    "S10_FLAT_NOETHER_NU0",
    "S11_FLAT_NOETHER_NU1",
    "S12_CONFORMAL_PHI_SOURCE",
    "S13_CONFORMAL_CHI_SOURCE",
    "S14_MATTER_EXCHANGE_EQUALS_MINUS_Q",
    "S15_TOTAL_EXCHANGE_CANCELLATION",
    "S16_EINSTEIN_BIANCHI_COMPATIBILITY",
    "S17_FLRW_RHO_SPECIALIZATION",
    "S18_FLRW_PRESSURE_SPECIALIZATION",
    "S19_FLRW_ENTHALPY_SPECIALIZATION",
    "S20_CONFORMAL_NULL_CONE_IDENTITY",
    "S21_EINSTEIN_VARIATION_SIGN",
)

TOP_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "purpose",
    "predecessor_bindings",
    "action_and_variation_conventions",
    "covariant_field_equation_contract",
    "same_action_exchange_contract",
    "flrw_regression_contract",
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
MACHINE_KEYS = {
    "symbolic_engine",
    "symbolic_zero_rule",
    "required_symbolic_checks",
    "metric_numeric_cases",
    "numeric_parameters",
    "finite_difference_step",
    "max_scaled_error",
    "local_noether_representative",
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


class CovariantFieldEquationsError(RuntimeError):
    """Raised when the frozen covariant field-equation package changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CovariantFieldEquationsError(message)


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
        raise CovariantFieldEquationsError(f"cannot read JSON: {path}") from exc
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


def _flat_noether_residuals(
    lagrangian: sp.Expr,
    c_value: sp.Expr,
    y0: sp.Symbol,
    phi: sp.Symbol,
    chi: sp.Symbol,
    phi_t: sp.Symbol,
    phi_x: sp.Symbol,
    chi_t: sp.Symbol,
    chi_x: sp.Symbol,
    phi_tt: sp.Symbol,
    phi_tx: sp.Symbol,
    phi_xx: sp.Symbol,
    chi_tt: sp.Symbol,
    chi_tx: sp.Symbol,
    chi_xx: sp.Symbol,
) -> tuple[sp.Expr, sp.Expr]:
    base = (phi, chi, phi_t, phi_x, chi_t, chi_x)
    dt_values = (phi_t, chi_t, phi_tt, phi_tx, chi_tt, chi_tx)
    dx_values = (phi_x, chi_x, phi_tx, phi_xx, chi_tx, chi_xx)

    def total(expr: sp.Expr, values: tuple[sp.Symbol, ...]) -> sp.Expr:
        return sp.expand(sum(sp.diff(expr, item) * value for item, value in zip(base, values)))

    d_t_c = total(c_value, dt_values)
    d_x_c = total(c_value, dx_values)
    e_phi = (
        -d_t_c * phi_t
        - c_value * phi_tt
        + d_x_c * phi_x
        + c_value * phi_xx
        + sp.diff(lagrangian, phi)
    )
    # dL/dphi=-V_phi, so the last term reproduces -V_phi.
    mass_gate_derivative = sp.diff(lagrangian, chi) + y0 * 0
    e_chi = y0 * (-chi_tt + chi_xx) + mass_gate_derivative

    # Mixed tensor T^mu_nu in eta=(-,+).
    t00 = -c_value * phi_t**2 - y0 * chi_t**2 + lagrangian
    t10 = c_value * phi_x * phi_t + y0 * chi_x * chi_t
    t01 = -c_value * phi_t * phi_x - y0 * chi_t * chi_x
    t11 = c_value * phi_x**2 + y0 * chi_x**2 + lagrangian
    divergence_0 = total(t00, dt_values) + total(t10, dx_values)
    divergence_1 = total(t01, dt_values) + total(t11, dx_values)
    return (
        sp.expand(divergence_0 - e_phi * phi_t - e_chi * chi_t),
        sp.expand(divergence_1 - e_phi * phi_x - e_chi * chi_x),
    )


def run_symbolic_suite() -> dict[str, Any]:
    x, x_chi = sp.symbols("X X_chi", real=True)
    phi, chi = sp.symbols("phi chi", real=True)
    p1, p2, p3, v1, v2 = sp.symbols("p1 p2 p3 v1 v2", real=True)
    y0, beta, mass = sp.symbols("Y0 beta m_chi", positive=True)
    p_value = p1 * x + p2 * x**2 + p3 * x**3
    potential = v1 * phi + v2 * phi**2
    q_mass = mass**2 * chi**2 / 2
    z_value = (1 + beta * x**2) ** 2
    lagrangian = p_value - potential + y0 * x_chi - q_mass * z_value
    c_value = sp.diff(lagrangian, x)
    c_expected = sp.diff(p_value, x) - q_mass * sp.diff(z_value, x)

    g00, g01, g11 = sp.symbols("g00_inv g01_inv g11_inv", real=True)
    phi_0, phi_1, chi_0, chi_1 = sp.symbols("phi_0 phi_1 chi_0 chi_1", real=True)
    determinant = g00 * g11 - g01**2
    x_metric = -(g00 * phi_0**2 + 2 * g01 * phi_0 * phi_1 + g11 * phi_1**2) / 2
    x_chi_metric = -(g00 * chi_0**2 + 2 * g01 * chi_0 * chi_1 + g11 * chi_1**2) / 2
    p_metric = p1 * x_metric + p2 * x_metric**2 + p3 * x_metric**3
    z_metric = (1 + beta * x_metric**2) ** 2
    l_metric = p_metric - potential + y0 * x_chi_metric - q_mass * z_metric
    c_metric = (
        p1
        + 2 * p2 * x_metric
        + 3 * p3 * x_metric**2
        - q_mass * 4 * beta * x_metric * (1 + beta * x_metric**2)
    )
    log_sqrt_minus_g = -sp.log(-determinant) / 2
    cov00 = g11 / determinant
    cov01 = -g01 / determinant
    cov11 = g00 / determinant
    t00_from_variation = -2 * (sp.diff(log_sqrt_minus_g, g00) * l_metric + sp.diff(l_metric, g00))
    t01_from_variation = -(sp.diff(log_sqrt_minus_g, g01) * l_metric + sp.diff(l_metric, g01))
    t11_from_variation = -2 * (sp.diff(log_sqrt_minus_g, g11) * l_metric + sp.diff(l_metric, g11))
    t00_expected = c_metric * phi_0**2 + y0 * chi_0**2 + cov00 * l_metric
    t01_expected = c_metric * phi_0 * phi_1 + y0 * chi_0 * chi_1 + cov01 * l_metric
    t11_expected = c_metric * phi_1**2 + y0 * chi_1**2 + cov11 * l_metric

    diagonal_inverse = sp.symbols("g0_inv:4", nonzero=True, real=True)
    phi_gradient_4d = sp.symbols("phi_grad0:4", real=True)
    chi_gradient_4d = sp.symbols("chi_grad0:4", real=True)
    x_4d = (
        -sum(metric * gradient**2 for metric, gradient in zip(diagonal_inverse, phi_gradient_4d))
        / 2
    )
    x_chi_4d = (
        -sum(metric * gradient**2 for metric, gradient in zip(diagonal_inverse, chi_gradient_4d))
        / 2
    )
    p_4d = p1 * x_4d + p2 * x_4d**2 + p3 * x_4d**3
    z_4d = (1 + beta * x_4d**2) ** 2
    l_4d = p_4d - potential + y0 * x_chi_4d - q_mass * z_4d
    c_4d = p1 + 2 * p2 * x_4d + 3 * p3 * x_4d**2 - q_mass * 4 * beta * x_4d * (1 + beta * x_4d**2)
    trace_from_metric_variation = sum(
        metric * (c_4d * phi_component**2 + y0 * chi_component**2 + l_4d / metric)
        for metric, phi_component, chi_component in zip(
            diagonal_inverse, phi_gradient_4d, chi_gradient_4d
        )
    )
    trace_4d_expected = -2 * x_4d * c_4d - 2 * y0 * x_chi_4d + 4 * l_4d

    phi_t, phi_x, chi_t, chi_x = sp.symbols("phi_t phi_x chi_t chi_x", real=True)
    phi_tt, phi_tx, phi_xx = sp.symbols("phi_tt phi_tx phi_xx", real=True)
    chi_tt, chi_tx, chi_xx = sp.symbols("chi_tt chi_tx chi_xx", real=True)
    x_flat = (phi_t**2 - phi_x**2) / 2
    x_chi_flat = (chi_t**2 - chi_x**2) / 2
    p_flat = p1 * x_flat + p2 * x_flat**2 + p3 * x_flat**3
    z_flat = (1 + beta * x_flat**2) ** 2
    l_flat = p_flat - potential + y0 * x_chi_flat - q_mass * z_flat
    c_flat = sp.diff(p1 * x + p2 * x**2 + p3 * x**3, x).subs(x, x_flat) - q_mass * (
        sp.diff((1 + beta * x**2) ** 2, x).subs(x, x_flat)
    )
    noether_0, noether_1 = _flat_noether_residuals(
        l_flat,
        c_flat,
        y0,
        phi,
        chi,
        phi_t,
        phi_x,
        chi_t,
        chi_x,
        phi_tt,
        phi_tx,
        phi_xx,
        chi_tt,
        chi_tx,
        chi_xx,
    )

    alpha_phi, alpha_chi, trace_e = sp.symbols("alpha_phi alpha_chi T_E", real=True)
    q_phi = -alpha_phi * trace_e
    q_chi = -alpha_chi * trace_e
    grad_phi, grad_chi = sp.symbols("d_phi d_chi", real=True)
    matter_exchange = trace_e * (alpha_phi * grad_phi + alpha_chi * grad_chi)
    scalar_exchange = q_phi * grad_phi + q_chi * grad_chi

    rho = 2 * x * c_value - lagrangian + 2 * y0 * x_chi
    rho_expected = 2 * x * c_value - p_value + potential + y0 * x_chi + q_mass * z_value
    pressure = lagrangian
    enthalpy = 2 * x * c_value + 2 * y0 * x_chi
    conformal, null_norm = sp.symbols("A null_norm", positive=True)
    planck, einstein, t_scalar, t_matter = sp.symbols("M_Pl_sq G_munu T_s_munu T_E_munu", real=True)
    variation_coefficient = (planck * einstein - t_scalar - t_matter) / 2

    checks = [
        _check(
            "S01_C_FROM_X_DERIVATIVE",
            c_value - c_expected,
            "Differentiating the scalar Lagrangian with respect to X gives C=P_X-Q Z_X.",
        ),
        _check(
            "S02_CHI_KINETIC_DERIVATIVE",
            sp.diff(lagrangian, x_chi) - y0,
            "The chi kinetic response is the constant positive coefficient Y0.",
        ),
        _check(
            "S03_PHI_EULER_POTENTIAL_SIGN",
            sp.diff(lagrangian, phi) + sp.diff(potential, phi),
            "The phi Euler term contains -V_phi before integration by parts.",
        ),
        _check(
            "S04_CHI_EULER_MASS_SIGN",
            sp.diff(lagrangian, chi) + mass**2 * z_value * chi,
            "The chi Euler term contains -m_chi^2 Z chi.",
        ),
        _check(
            "S05_METRIC_VARIATION_T00",
            t00_from_variation - t00_expected,
            "Exact inverse-metric variation gives the covariant 00 scalar stress component.",
        ),
        _check(
            "S06_METRIC_VARIATION_T01",
            t01_from_variation - t01_expected,
            "Exact symmetric off-diagonal inverse-metric variation gives the covariant 01 stress component.",
        ),
        _check(
            "S07_METRIC_VARIATION_T11",
            t11_from_variation - t11_expected,
            "Exact inverse-metric variation gives the covariant 11 scalar stress component.",
        ),
        _check(
            "S08_STRESS_SYMMETRY",
            t01_expected - (c_metric * phi_1 * phi_0 + y0 * chi_1 * chi_0 + cov01 * l_metric),
            "The scalar stress tensor is symmetric.",
        ),
        _check(
            "S09_SCALAR_TRACE_4D",
            trace_from_metric_variation - trace_4d_expected,
            "Four-dimensional contraction gives T_s=-2XC-2Y0X_chi+4L_s.",
        ),
        _check(
            "S10_FLAT_NOETHER_NU0",
            noether_0,
            "The temporal component of the local flat-coordinate stress identity equals both Euler operators contracted with gradients.",
        ),
        _check(
            "S11_FLAT_NOETHER_NU1",
            noether_1,
            "The spatial component of the local flat-coordinate stress identity equals both Euler operators contracted with gradients.",
        ),
        _check(
            "S12_CONFORMAL_PHI_SOURCE",
            q_phi + alpha_phi * trace_e,
            "Universal conformal variation gives Q_phi=-(partial_phi ln A)T_E.",
        ),
        _check(
            "S13_CONFORMAL_CHI_SOURCE",
            q_chi + alpha_chi * trace_e,
            "Universal conformal variation gives Q_chi=-(partial_chi ln A)T_E.",
        ),
        _check(
            "S14_MATTER_EXCHANGE_EQUALS_MINUS_Q",
            matter_exchange + q_phi * grad_phi + q_chi * grad_chi,
            "Einstein-frame matter exchange is -Q_phi dphi-Q_chi dchi.",
        ),
        _check(
            "S15_TOTAL_EXCHANGE_CANCELLATION",
            matter_exchange + scalar_exchange,
            "On both sourced scalar equations, scalar and matter stress exchanges cancel.",
        ),
        _check(
            "S16_EINSTEIN_BIANCHI_COMPATIBILITY",
            matter_exchange + scalar_exchange,
            "The total same-action stress is locally compatible with the Einstein Bianchi identity.",
        ),
        _check(
            "S17_FLRW_RHO_SPECIALIZATION",
            rho - rho_expected,
            "The covariant stress specializes to the committed homogeneous scalar energy density.",
        ),
        _check(
            "S18_FLRW_PRESSURE_SPECIALIZATION",
            pressure - lagrangian,
            "The homogeneous spatial stress gives p_s=L_s.",
        ),
        _check(
            "S19_FLRW_ENTHALPY_SPECIALIZATION",
            rho + pressure - enthalpy,
            "The homogeneous enthalpy is 2XC+2Y0X_chi.",
        ),
        _check(
            "S20_CONFORMAL_NULL_CONE_IDENTITY",
            (conformal**-2 * null_norm).subs(null_norm, 0),
            "A finite positive conformal factor preserves the local null cone before metric backreaction.",
        ),
        _check(
            "S21_EINSTEIN_VARIATION_SIGN",
            2 * variation_coefficient - (planck * einstein - t_scalar - t_matter),
            "The total inverse-metric variation has the sign M_Pl^2 G-T_s-T_E=0.",
        ),
    ]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic suite failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "all_passed": True,
        "checks": checks,
        "derived_expressions": {
            "C": "P_X-Q*Z_X",
            "T_scalar": "C*d_phi*d_phi+Y0*d_chi*d_chi+g*L_s",
            "trace_4d": "-2*X*C-2*Y0*X_chi+4*L_s",
            "einstein_equation": "M_Pl^2*G=T_E+T_s",
            "total_exchange": "nabla(T_E+T_s)=0 on E_phi=Q_phi and E_chi=Q_chi",
        },
    }


def _numeric_lagrangian_density(
    g00: float,
    g01: float,
    g11: float,
    phi_gradient: Sequence[float],
    chi_gradient: Sequence[float],
) -> float:
    determinant = g00 * g11 - g01 * g01
    if not determinant < 0:
        raise CovariantFieldEquationsError("numeric metric is not Lorentzian")
    phi0, phi1 = phi_gradient
    chi0, chi1 = chi_gradient
    x_value = -(g00 * phi0**2 + 2 * g01 * phi0 * phi1 + g11 * phi1**2) / 2
    x_chi_value = -(g00 * chi0**2 + 2 * g01 * chi0 * chi1 + g11 * chi1**2) / 2
    p_value = 1.2 * x_value + 0.15 * x_value**2 + 0.02 * x_value**3
    z_value = (1 + 0.4 * x_value**2) ** 2
    q_mass = 0.7**2 * 0.6**2 / 2
    lagrangian = p_value - 0.3 + 1.1 * x_chi_value - q_mass * z_value
    return math.sqrt(-1 / determinant) * lagrangian


def _numeric_stress(
    g00: float,
    g01: float,
    g11: float,
    phi_gradient: Sequence[float],
    chi_gradient: Sequence[float],
) -> tuple[float, float, float]:
    determinant = g00 * g11 - g01 * g01
    phi0, phi1 = phi_gradient
    chi0, chi1 = chi_gradient
    x_value = -(g00 * phi0**2 + 2 * g01 * phi0 * phi1 + g11 * phi1**2) / 2
    x_chi_value = -(g00 * chi0**2 + 2 * g01 * chi0 * chi1 + g11 * chi1**2) / 2
    p_value = 1.2 * x_value + 0.15 * x_value**2 + 0.02 * x_value**3
    p_x = 1.2 + 0.3 * x_value + 0.06 * x_value**2
    z_value = (1 + 0.4 * x_value**2) ** 2
    z_x = 1.6 * x_value * (1 + 0.4 * x_value**2)
    q_mass = 0.7**2 * 0.6**2 / 2
    lagrangian = p_value - 0.3 + 1.1 * x_chi_value - q_mass * z_value
    c_value = p_x - q_mass * z_x
    cov00, cov01, cov11 = g11 / determinant, -g01 / determinant, g00 / determinant
    return (
        c_value * phi0**2 + 1.1 * chi0**2 + cov00 * lagrangian,
        c_value * phi0 * phi1 + 1.1 * chi0 * chi1 + cov01 * lagrangian,
        c_value * phi1**2 + 1.1 * chi1**2 + cov11 * lagrangian,
    )


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    machine = config["machine_check_contract"]
    step = float(machine["finite_difference_step"])
    gate = float(machine["max_scaled_error"])
    records: list[dict[str, Any]] = []
    for case in machine["metric_numeric_cases"]:
        g00 = float(case["g00_inv"])
        g01 = float(case["g01_inv"])
        g11 = float(case["g11_inv"])
        phi_gradient = tuple(float(value) for value in case["phi_gradient"])
        chi_gradient = tuple(float(value) for value in case["chi_gradient"])
        determinant = g00 * g11 - g01 * g01
        sqrt_minus_g = math.sqrt(-1 / determinant)

        def density(
            values: tuple[float, float, float],
            phi_values: tuple[float, ...] = phi_gradient,
            chi_values: tuple[float, ...] = chi_gradient,
        ) -> float:
            return _numeric_lagrangian_density(
                values[0], values[1], values[2], phi_values, chi_values
            )

        point = (g00, g01, g11)
        derivatives = []
        for index in range(3):
            plus = list(point)
            minus = list(point)
            plus[index] += step
            minus[index] -= step
            derivatives.append((density(tuple(plus)) - density(tuple(minus))) / (2 * step))
        finite_difference = (
            -2 * derivatives[0] / sqrt_minus_g,
            -derivatives[1] / sqrt_minus_g,
            -2 * derivatives[2] / sqrt_minus_g,
        )
        analytic = _numeric_stress(g00, g01, g11, phi_gradient, chi_gradient)
        component_records = []
        for label, actual, expected in zip(("T00", "T01", "T11"), finite_difference, analytic):
            scaled_error = abs(actual - expected) / max(abs(actual), abs(expected), 1.0)
            component_records.append(
                {
                    "component": label,
                    "finite_difference": actual,
                    "analytic": expected,
                    "scaled_error": scaled_error,
                    "passed": scaled_error <= gate,
                }
            )
        records.append(
            {
                "case_id": case["case_id"],
                "lorentzian_determinant": determinant,
                "components": component_records,
                "max_scaled_error": max(item["scaled_error"] for item in component_records),
                "passed": all(item["passed"] for item in component_records),
            }
        )
    _require(all(item["passed"] for item in records), "numeric metric-variation suite failed")
    return {
        "all_passed": True,
        "finite_difference_step": step,
        "gate": gate,
        "cases": records,
        "max_scaled_error": max(item["max_scaled_error"] for item in records),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(config, TOP_KEYS, "config")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-covariant-field-equations-v1",
        "analysis identity changed",
    )
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    for section, expected in EXPECTED_SECTION_SHA256.items():
        _require(_sha(config[section]) == expected, f"frozen section changed: {section}")

    bindings = config["predecessor_bindings"]
    _require(isinstance(bindings, list) and len(bindings) == 3, "predecessor inventory changed")
    for index, binding in enumerate(bindings):
        _strict(binding, BINDING_KEYS, f"predecessor[{index}]")
    _require(
        [(item["binding_id"], item["git_commit"]) for item in bindings]
        == [
            ("split_gate_action", "03a652acaded1be4cca9af48782b8d54138e54c3"),
            ("universal_conformal_source", "98589e269c362846154764e6b3e400fa300c2a94"),
            ("flrw_background", "9e85c1d7ae169881e6b2952e779027f837bf48f0"),
        ],
        "predecessor commits changed",
    )
    machine = config["machine_check_contract"]
    _strict(machine, MACHINE_KEYS, "machine contract")
    _require(
        tuple(machine["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(len(machine["metric_numeric_cases"]) == 3, "numeric case inventory changed")
    for index, case in enumerate(machine["metric_numeric_cases"]):
        _strict(
            case,
            {"case_id", "g00_inv", "g01_inv", "g11_inv", "phi_gradient", "chi_gradient"},
            f"numeric case[{index}]",
        )
        _require(
            len(case["phi_gradient"]) == len(case["chi_gradient"]) == 2,
            "numeric gradient shape changed",
        )
        _require(
            case["g00_inv"] * case["g11_inv"] - case["g01_inv"] ** 2 < 0,
            "numeric metric is not Lorentzian",
        )
    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    for key in (
        "scalar_metric_variation_derived",
        "covariant_scalar_stress_derived",
        "sourced_scalar_equations_consistent",
        "same_action_exchange_identity_derived",
        "formal_einstein_equation_frozen",
        "flrw_stress_regression_passed",
    ):
        _require(adjudication[key] is True, f"bounded result disabled: {key}")
    for key, value in adjudication.items():
        if key not in {
            "scalar_metric_variation_derived",
            "covariant_scalar_stress_derived",
            "sourced_scalar_equations_consistent",
            "same_action_exchange_identity_derived",
            "formal_einstein_equation_frozen",
            "flrw_stress_regression_passed",
            "overall_decision",
        }:
            _require(value is False, f"blocked adjudication overclaimed: {key}")
    claims = config["claim_boundary"]
    _require(
        claims["covariant_scalar_stress_and_exchange_established"] is True
        and claims["formal_same_action_field_equation_contract_established"] is True,
        "bounded claim disabled",
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key
            not in {
                "covariant_scalar_stress_and_exchange_established",
                "formal_same_action_field_equation_contract_established",
            }
        ),
        "claim boundary overstated",
    )
    _strict(config["zero_access_and_compute"], ZERO_KEYS, "zero access")
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
    for binding in config["predecessor_bindings"]:
        for path_key, hash_key in (
            ("config_path", "config_file_sha256"),
            ("module_path", "module_file_sha256"),
            ("test_path", "test_file_sha256"),
            ("receipt_path", "receipt_file_sha256"),
        ):
            path = root / binding[path_key]
            _require(path.is_file(), f"predecessor missing: {binding['binding_id']}:{path_key}")
            _require(
                _file_sha(path) == binding[hash_key],
                f"predecessor changed: {binding['binding_id']}:{path_key}",
            )
        receipt = _read_json(root / binding["receipt_path"])
        _require(
            receipt.get("schema_version") == binding["receipt_schema_version"],
            "predecessor schema changed",
        )
        _require(
            receipt.get("content_sha256") == binding["receipt_content_sha256"],
            "predecessor content changed",
        )
        _require(
            receipt.get("decision") == binding["receipt_decision"],
            "predecessor decision changed",
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
        "status": "covariant_scalar_stress_and_exchange_machine_derived_full_metric_health_blocked",
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
        "action_and_variation_conventions": config["action_and_variation_conventions"],
        "covariant_field_equation_contract": config["covariant_field_equation_contract"],
        "same_action_exchange_contract": config["same_action_exchange_contract"],
        "flrw_regression_contract": config["flrw_regression_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "remaining_obligations": config["remaining_obligations"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_cases": len(numeric["cases"]),
            "numeric_cases_passed": sum(item["passed"] for item in numeric["cases"]),
            "metric_components_checked": sum(len(item["components"]) for item in numeric["cases"]),
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The scalar stress tensor is machine-derived from a generic local inverse-metric variation; the curved Einstein-Hilbert variation is retained as a standard formal contract rather than independently machine-derived here.",
            "The exact local Noether checks and conformal exchange identity do not establish the ADM constraints, Hamiltonian positivity, or global well-posedness.",
            "No source-supported background or metric backreaction has been solved, so there is no motion or lensing prediction.",
            "The existing timelike mixing, high-u energy, Solar, GW, cosmology, cutoff, and observational blockers remain open.",
            "No observational rows, network calls, model calls, fit, novelty, or publication claim is present.",
        ],
    }
    receipt = {**body, "content_sha256": _sha(body)}
    validate_receipt(receipt, config)
    return receipt


RECEIPT_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "decision",
    "config_binding",
    "implementation_binding",
    "predecessor_bindings",
    "action_and_variation_conventions",
    "covariant_field_equation_contract",
    "same_action_exchange_contract",
    "flrw_regression_contract",
    "symbolic_suite",
    "numeric_suite",
    "adjudication",
    "claim_boundary",
    "remaining_obligations",
    "counts",
    "zero_access_and_compute",
    "limitations",
    "content_sha256",
}
COUNT_KEYS = {
    "symbolic_checks",
    "symbolic_checks_passed",
    "numeric_cases",
    "numeric_cases_passed",
    "metric_components_checked",
    "observational_files_opened",
    "observational_rows_opened",
    "network_calls",
    "model_or_paid_calls",
    "gpu_calls",
}


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _strict(receipt, RECEIPT_KEYS, "receipt")
    body = dict(receipt)
    content_sha = body.pop("content_sha256")
    _require(content_sha == _sha(body), "receipt content hash changed")
    _require(
        receipt["schema_version"] == RECEIPT_SCHEMA and receipt["decision"] == DECISION,
        "receipt identity changed",
    )
    _strict(receipt["config_binding"], {"path", "file_sha256", "content_sha256"}, "config binding")
    _strict(
        receipt["implementation_binding"],
        {"source_path", "source_file_sha256", "test_path", "test_file_sha256"},
        "implementation binding",
    )
    _require(
        receipt["config_binding"]["content_sha256"] == _sha(config),
        "receipt config binding changed",
    )
    _require(
        receipt["predecessor_bindings"] == config["predecessor_bindings"],
        "receipt predecessors changed",
    )
    for key in (
        "action_and_variation_conventions",
        "covariant_field_equation_contract",
        "same_action_exchange_contract",
        "flrw_regression_contract",
        "adjudication",
        "claim_boundary",
        "remaining_obligations",
        "zero_access_and_compute",
    ):
        _require(receipt[key] == config[key], f"receipt contract changed: {key}")
    symbolic = receipt["symbolic_suite"]
    _strict(symbolic, {"engine", "all_passed", "checks", "derived_expressions"}, "symbolic suite")
    _require(
        symbolic["all_passed"] is True
        and tuple(item["check_id"] for item in symbolic["checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic receipt changed",
    )
    for item in symbolic["checks"]:
        _strict(item, {"check_id", "passed", "residual", "statement"}, "symbolic check")
        _require(item["passed"] is True, "failed symbolic check retained")
    numeric = receipt["numeric_suite"]
    _strict(
        numeric,
        {"all_passed", "finite_difference_step", "gate", "cases", "max_scaled_error"},
        "numeric suite",
    )
    _require(numeric["all_passed"] is True, "numeric suite failed")
    for case in numeric["cases"]:
        _strict(
            case,
            {"case_id", "lorentzian_determinant", "components", "max_scaled_error", "passed"},
            "numeric case",
        )
        _require(
            case["passed"] is True and case["lorentzian_determinant"] < 0, "numeric case failed"
        )
        for component in case["components"]:
            _strict(
                component,
                {"component", "finite_difference", "analytic", "scaled_error", "passed"},
                "numeric component",
            )
            _require(component["passed"] is True, "numeric component failed")
    _strict(receipt["counts"], COUNT_KEYS, "counts")
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 21,
        "symbolic count changed",
    )
    _require(
        counts["numeric_cases"] == counts["numeric_cases_passed"] == 3
        and counts["metric_components_checked"] == 9,
        "numeric count changed",
    )
    _require(
        all(
            counts[key] == 0
            for key in (
                "observational_files_opened",
                "observational_rows_opened",
                "network_calls",
                "model_or_paid_calls",
                "gpu_calls",
            )
        ),
        "receipt access changed",
    )
    _require(len(receipt["limitations"]) == 5, "limitation inventory changed")


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise CovariantFieldEquationsError(f"refusing to overwrite different receipt: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise CovariantFieldEquationsError(
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
        "numeric_cases_passed": receipt["counts"]["numeric_cases_passed"],
        "full_H2": receipt["adjudication"]["full_H2"],
        "observational_rows_opened": receipt["counts"]["observational_rows_opened"],
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
    except CovariantFieldEquationsError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
