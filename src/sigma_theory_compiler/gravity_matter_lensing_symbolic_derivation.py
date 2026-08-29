"""Machine-check a bounded subset of the corrected two-scalar theory preflight."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_symbolic_derivation_v1.json")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-symbolic-derivation-v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_matter_lensing_symbolic_derivation.py")
TEST_PATH = Path("tests/test_gravity_matter_lensing_symbolic_derivation.py")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-symbolic-derivation-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-symbolic-derivation-receipt-1.0"
EXPECTED_CONFIG_FILE_SHA256 = "20444df546a1010a90cf0086341eb0da892a330a15d2b7961d418b4b7fd045c6"
EXPECTED_CONFIG_CONTENT_SHA256 = "8b9fb6ed434c4bed727f54de36bd52cffa7c9697b1bdfcb9848dae6bfb14a9ab"
DECISION = (
    "PARTIAL_H2_BOUNDED_SCALAR_DERIVATION_PASSED_"
    "FULL_COVARIANT_HEALTH_AND_LENSING_GATES_REMAIN_BLOCKED"
)
SYMBOLIC_CHECK_IDS = (
    "S1_GENERIC_DERIVATIVE_XPHI",
    "S2_GENERIC_DERIVATIVE_XCHI",
    "S3_GENERIC_DERIVATIVE_CHI",
    "S4_TIME_KINETIC_PHI",
    "S5_TIME_KINETIC_CHI",
    "S6_TIME_KINETIC_CROSS_ZERO",
    "S7_GRADIENT_PHI",
    "S8_GRADIENT_CHI",
    "S9_GRADIENT_CROSS_ZERO",
    "S10_MASS_PHI",
    "S11_MASS_CHI",
    "S12_MASS_CROSS_ZERO",
    "S13_REDUCED_NOETHER_IDENTITY",
    "S14_LINEAR_CONFORMAL_CANCELLATION",
    "S15_DISFORMAL_LINEAR_TERM_ZERO",
    "S16_GREEN_HELMHOLTZ_R_GT_ZERO",
    "S17_GREEN_FIXED_RANGE",
    "S18_GATE_AMPLITUDE_IDENTITY",
    "S19_BOUNDED_0P1_EULER_PHI",
    "S20_BOUNDED_0P1_EULER_CHI",
)
NUMERIC_CHECK_IDS = (
    "N1_DERIVATIVE_XPHI_FINITE_DIFFERENCE",
    "N2_DERIVATIVE_XCHI_FINITE_DIFFERENCE",
    "N3_DERIVATIVE_CHI_FINITE_DIFFERENCE",
    "N4_GATE_AMPLITUDE_B_SQUARED",
    "N5_GREEN_FIXED_RANGE",
    "N6_CONFORMAL_LENSING_SUM",
)


class GravityMatterLensingSymbolicDerivationError(RuntimeError):
    """Raised when a frozen derivation, predecessor, or receipt changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GravityMatterLensingSymbolicDerivationError(
            f"cannot read JSON {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise GravityMatterLensingSymbolicDerivationError(f"JSON object required: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GravityMatterLensingSymbolicDerivationError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _zero(value: sp.Expr) -> bool:
    return sp.simplify(sp.factor(value)) == 0


def _check(check_id: str, residual: sp.Expr, detail: str) -> dict[str, Any]:
    simplified = sp.simplify(sp.factor(residual))
    return {
        "check_id": check_id,
        "passed": simplified == 0,
        "residual": sp.sstr(simplified),
        "detail": detail,
    }


def _generic_scalar_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    x_phi, x_chi, chi, m_chi, m_phi, m_pl, a0 = sp.symbols(
        "X_phi X_chi chi m_chi M_phi M_Pl a0", nonzero=True
    )
    y = x_phi / m_phi**4
    u = 4 * x_phi**2 / (m_pl**4 * a0**4)
    z = (1 + u) ** 2
    l_chi = x_chi - m_chi**2 * chi**2 / 2
    p = sp.Function("P")
    lagrangian = m_phi**4 * p(y) + z * l_chi
    eta = sp.Symbol("eta")
    p_y = sp.Subs(sp.diff(p(eta), eta), eta, y)
    z_u = 2 * (1 + u)
    u_x = 8 * x_phi / (m_pl**4 * a0**4)
    expected_x_phi = p_y + l_chi * z_u * u_x
    expected_x_chi = z
    expected_chi = -z * m_chi**2 * chi
    checks = [
        _check(
            "S1_GENERIC_DERIVATIVE_XPHI",
            sp.diff(lagrangian, x_phi) - expected_x_phi,
            "Direct symbolic differentiation retains the full chi kinetic-minus-mass bracket.",
        ),
        _check(
            "S2_GENERIC_DERIVATIVE_XCHI",
            sp.diff(lagrangian, x_chi) - expected_x_chi,
            "The chi kinetic derivative is Z.",
        ),
        _check(
            "S3_GENERIC_DERIVATIVE_CHI",
            sp.diff(lagrangian, chi) - expected_chi,
            "The chi mass derivative carries the same Z as the kinetic term.",
        ),
    ]
    expressions = {
        "dL_dX_phi": sp.sstr(sp.simplify(expected_x_phi)),
        "dL_dX_chi": sp.sstr(sp.simplify(expected_x_chi)),
        "dL_dchi": sp.sstr(sp.simplify(expected_chi)),
    }
    return checks, expressions


def _quadratic_block_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    v_phi, v_chi, s_phi, s_chi, chi = sp.symbols("v_phi v_chi s_phi s_chi chi", real=True)
    m_phi, m_pl, a0, m_chi = sp.symbols("M_phi M_Pl a0 m_chi", positive=True)
    y0, vpp, delta_phi = sp.symbols("y0 V_phiphi delta_phi", real=True)
    p0, p1, p2, p3 = sp.symbols("p0 p1 p2 p3", real=True)
    x_phi = (v_phi**2 - s_phi**2) / 2
    x_chi = (v_chi**2 - s_chi**2) / 2
    y = x_phi / m_phi**4
    p = p0 + p1 * y + p2 * y**2 + p3 * y**3
    p_y = sp.diff(p0 + p1 * y0 + p2 * y0**2 + p3 * y0**3, y0)
    p_yy = sp.diff(p_y, y0)
    u = 4 * x_phi**2 / (m_pl**4 * a0**4)
    z = (1 + u) ** 2
    lagrangian = m_phi**4 * p + z * (x_chi - m_chi**2 * chi**2 / 2) - vpp * delta_phi**2 / 2
    base = {v_chi: 0, s_phi: 0, s_chi: 0, chi: 0, v_phi**2: 2 * m_phi**4 * y0}

    def at_base(expression: sp.Expr) -> sp.Expr:
        return sp.simplify(expression.subs(base).subs(v_phi**2, 2 * m_phi**4 * y0))

    z_bar = sp.simplify(z.subs({s_phi: 0, v_phi**2: 2 * m_phi**4 * y0}))
    h_time_phi = at_base(sp.diff(lagrangian, v_phi, v_phi))
    h_time_chi = at_base(sp.diff(lagrangian, v_chi, v_chi))
    h_time_cross = at_base(sp.diff(lagrangian, v_phi, v_chi))
    h_grad_phi = at_base(-sp.diff(lagrangian, s_phi, s_phi))
    h_grad_chi = at_base(-sp.diff(lagrangian, s_chi, s_chi))
    h_grad_cross = at_base(-sp.diff(lagrangian, s_phi, s_chi))
    h_mass_phi = at_base(-sp.diff(lagrangian, delta_phi, delta_phi))
    h_mass_chi = at_base(-sp.diff(lagrangian, chi, chi))
    h_mass_cross = at_base(-sp.diff(lagrangian, delta_phi, chi))
    expected_time_phi = p_y + 2 * y0 * p_yy
    checks = [
        _check(
            "S4_TIME_KINETIC_PHI",
            h_time_phi - expected_time_phi,
            "Homogeneous phi time coefficient is P_y+2*y*P_yy.",
        ),
        _check(
            "S5_TIME_KINETIC_CHI",
            h_time_chi - z_bar,
            "Chi time coefficient is Z on chi_bar=0.",
        ),
        _check(
            "S6_TIME_KINETIC_CROSS_ZERO",
            h_time_cross,
            "The scalar time-kinetic cross block vanishes only on the frozen chi_bar=0 background.",
        ),
        _check(
            "S7_GRADIENT_PHI",
            h_grad_phi - p_y,
            "Homogeneous phi spatial-gradient coefficient is P_y.",
        ),
        _check(
            "S8_GRADIENT_CHI",
            h_grad_chi - z_bar,
            "Chi spatial-gradient coefficient is Z on the frozen background.",
        ),
        _check(
            "S9_GRADIENT_CROSS_ZERO",
            h_grad_cross,
            "The scalar spatial cross block vanishes only on the frozen background.",
        ),
        _check(
            "S10_MASS_PHI",
            h_mass_phi - vpp,
            "The raw quadratic phi mass entry is V_phi,phiphi on the declared background.",
        ),
        _check(
            "S11_MASS_CHI",
            h_mass_chi - z_bar * m_chi**2,
            "The raw quadratic chi mass entry is Z_bar*m_chi^2, paired with kinetic Z_bar.",
        ),
        _check(
            "S12_MASS_CROSS_ZERO",
            h_mass_cross,
            "The raw scalar mass cross entry vanishes on chi_bar=0 for the frozen potential template.",
        ),
    ]
    expressions = {
        "time_kinetic_matrix": "diag(P_y+2*y*P_yy,Z_bar)",
        "gradient_matrix": "diag(P_y,Z_bar)",
        "mass_matrix": "diag(V_phi,phiphi,Z_bar*m_chi^2)",
        "sound_speeds": "c_phi^2=P_y/(P_y+2*y*P_yy); c_chi^2=1",
        "Z_bar": sp.sstr(z_bar),
        "general_velocity_cross_derivative": sp.sstr(sp.factor(sp.diff(lagrangian, v_phi, v_chi))),
    }
    return checks, expressions


def _reduced_noether_check() -> dict[str, Any]:
    phi, chi, v_phi, v_chi, a_phi, a_chi = sp.symbols("phi chi v_phi v_chi a_phi a_chi", real=True)
    m_chi, m_pl, a0, lam = sp.symbols("m_chi M_Pl a0 lambda", positive=True)
    p0, p1, p2 = sp.symbols("p0 p1 p2", real=True)
    x_phi = v_phi**2 / 2
    x_chi = v_chi**2 / 2
    u = 4 * x_phi**2 / (m_pl**4 * a0**4)
    z = (1 + u) ** 2
    p = p0 + p1 * x_phi + p2 * x_phi**2
    lagrangian = p - lam * phi**2 / 2 + z * (x_chi - m_chi**2 * chi**2 / 2)
    coordinates = (phi, chi)
    velocities = (v_phi, v_chi)
    accelerations = (a_phi, a_chi)

    def total_time_derivative(expression: sp.Expr) -> sp.Expr:
        return sum(
            sp.diff(expression, q) * v + sp.diff(expression, v) * a
            for q, v, a in zip(coordinates, velocities, accelerations, strict=True)
        )

    energy = sum(v * sp.diff(lagrangian, v) for v in velocities) - lagrangian
    euler = [
        total_time_derivative(sp.diff(lagrangian, v)) - sp.diff(lagrangian, q)
        for q, v in zip(coordinates, velocities, strict=True)
    ]
    residual = total_time_derivative(energy) - sum(
        v * equation for v, equation in zip(velocities, euler, strict=True)
    )
    return _check(
        "S13_REDUCED_NOETHER_IDENTITY",
        residual,
        "The autonomous 0+1 reduction obeys the exact off-shell energy/Noether identity, including Z(X_phi); this is not the four-dimensional diffeomorphism identity.",
    )


def _weak_field_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    eps, phi_e, psi_e, conformal, d_phi, k2, m_phi = sp.symbols(
        "epsilon Phi_E Psi_E a d_phi k2 M_phi", real=True
    )
    a2 = 1 + 2 * eps * conformal
    g00_tilde = a2 * (-1 - 2 * eps * phi_e)
    gii_tilde = a2 * (1 - 2 * eps * psi_e)
    phi = -sp.diff(g00_tilde, eps).subs(eps, 0) / 2
    psi = -sp.diff(gii_tilde, eps).subs(eps, 0) / 2
    disformal = d_phi * eps**2 * k2 / m_phi**4
    checks = [
        _check(
            "S14_LINEAR_CONFORMAL_CANCELLATION",
            sp.simplify(phi + psi - phi_e - psi_e),
            "The universal conformal perturbation cancels from Phi+Psi at linear order.",
        ),
        _check(
            "S15_DISFORMAL_LINEAR_TERM_ZERO",
            sp.diff(disformal, eps).subs(eps, 0),
            "The derivative-disformal term has no linear contribution only about a constant scalar background.",
        ),
    ]
    return checks, {
        "Phi": sp.sstr(sp.simplify(phi)),
        "Psi": sp.sstr(sp.simplify(psi)),
        "lensing_sum": sp.sstr(sp.simplify(phi + psi)),
    }


def _green_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    r, m_chi, u = sp.symbols("r m_chi u", positive=True)
    radial = sp.exp(-m_chi * r) / r
    helmholtz = sp.diff(r**2 * sp.diff(radial, r), r) / r**2 - m_chi**2 * radial
    inverse_range = -sp.diff(sp.log(r * radial), r)
    b = 1 / (1 + u)
    z = (1 + u) ** 2
    checks = [
        _check(
            "S16_GREEN_HELMHOLTZ_R_GT_ZERO",
            helmholtz,
            "The Yukawa kernel solves the homogeneous radial Helmholtz equation for r>0.",
        ),
        _check(
            "S17_GREEN_FIXED_RANGE",
            inverse_range - m_chi,
            "The exponential inverse range is m_chi and is independent of Z when Z is locally constant.",
        ),
        _check(
            "S18_GATE_AMPLITUDE_IDENTITY",
            1 / z - b**2,
            "Dividing the constant-Z field equation gates the source amplitude by 1/Z=B^2.",
        ),
    ]
    return checks, {
        "radial_kernel_outside_source": sp.sstr(radial),
        "inverse_range": sp.sstr(sp.simplify(inverse_range)),
        "source_amplitude": sp.sstr(sp.simplify(1 / z)),
    }


def _bounded_euler_checks() -> tuple[list[dict[str, Any]], dict[str, str]]:
    t = sp.Symbol("t", real=True)
    phi = sp.Function("phi")(t)
    chi = sp.Function("chi")(t)
    phi_dot = sp.diff(phi, t)
    chi_dot = sp.diff(chi, t)
    m_chi, m_pl, a0 = sp.symbols("m_chi M_Pl a0", positive=True)
    p0, p1, p2, v2, v3, q_phi, q_chi = sp.symbols("p0 p1 p2 V2 V3 Q_phi Q_chi", real=True)
    x_phi = phi_dot**2 / 2
    x_chi = chi_dot**2 / 2
    u = 4 * x_phi**2 / (m_pl**4 * a0**4)
    z = (1 + u) ** 2
    l_chi = x_chi - m_chi**2 * chi**2 / 2
    potential = v2 * phi**2 / 2 + v3 * phi**3 / 3
    scalar_lagrangian = p0 + p1 * x_phi + p2 * x_phi**2 - potential + z * l_chi
    reduced_matter_lagrangian = -q_phi * phi - q_chi * chi
    total_lagrangian = scalar_lagrangian + reduced_matter_lagrangian
    c_phi = p1 + 2 * p2 * x_phi + l_chi * 2 * (1 + u) * (8 * x_phi / (m_pl**4 * a0**4))
    canonical_phi_scalar = sp.diff(c_phi * phi_dot, t) + sp.diff(potential, phi)
    canonical_chi_scalar = sp.diff(z * chi_dot, t) + z * m_chi**2 * chi
    actual_phi_total = sp.diff(sp.diff(total_lagrangian, phi_dot), t) - sp.diff(
        total_lagrangian, phi
    )
    actual_chi_total = sp.diff(sp.diff(total_lagrangian, chi_dot), t) - sp.diff(
        total_lagrangian, chi
    )
    checks = [
        _check(
            "S19_BOUNDED_0P1_EULER_PHI",
            actual_phi_total - canonical_phi_scalar - q_phi,
            "With L_m,0+1=-Q_phi*phi, the canonical equation is C_phi+Q_phi=0, so the covariant-oriented operator -C_phi equals +Q_phi.",
        ),
        _check(
            "S20_BOUNDED_0P1_EULER_CHI",
            actual_chi_total - canonical_chi_scalar - q_chi,
            "With L_m,0+1=-Q_chi*chi, the canonical equation is C_chi+Q_chi=0, so the covariant-oriented operator -C_chi equals +Q_chi.",
        ),
    ]
    return checks, {
        "source_definition": "Q_i=-delta(L_m)/delta(field_i)",
        "reduced_matter_lagrangian": "L_m,0+1=-Q_phi*phi-Q_chi*chi",
        "phi_covariant_oriented_equation": "-[d/dt(C_phi*dot(phi))+V_phi,phi]=Q_phi",
        "chi_covariant_oriented_equation": "-[d/dt(Z*dot(chi))+Z*m_chi^2*chi]=Q_chi",
        "general_covariant_equations_machine_verified": "false",
    }


def run_symbolic_suite() -> dict[str, Any]:
    generic, generic_expressions = _generic_scalar_checks()
    quadratic, quadratic_expressions = _quadratic_block_checks()
    weak, weak_expressions = _weak_field_checks()
    green, green_expressions = _green_checks()
    bounded_euler, bounded_euler_expressions = _bounded_euler_checks()
    checks = generic + quadratic + [_reduced_noether_check()] + weak + green + bounded_euler
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS, "symbolic order changed"
    )
    return {
        "engine": f"sympy-{sp.__version__}",
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks),
        "expressions": {
            "generic_scalar": generic_expressions,
            "constant_background": quadratic_expressions,
            "weak_field": weak_expressions,
            "green_function": green_expressions,
            "bounded_0p1_euler": bounded_euler_expressions,
        },
    }


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(1.0, abs(expected))


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    contract = config["machine_check_contract"]
    probe = contract["frozen_probe"]
    xp = float(probe["X_phi"])
    xc = float(probe["X_chi"])
    chi = float(probe["chi"])
    m = float(probe["m_chi"])
    m_phi = float(probe["M_phi"])
    m_pl = float(probe["M_Pl"])
    a0 = float(probe["a0"])
    p0, p1, p2, p3 = (float(item) for item in probe["P_coefficients"])
    h = float(probe["finite_difference_step"])

    def p_function(y: float) -> float:
        return p0 + p1 * y + p2 * y**2 + p3 * y**3

    def lagrangian(x_phi: float, x_chi: float, chi_value: float) -> float:
        y = x_phi / m_phi**4
        u = 4 * x_phi**2 / (m_pl**4 * a0**4)
        z = (1 + u) ** 2
        return m_phi**4 * p_function(y) + z * (x_chi - m**2 * chi_value**2 / 2)

    def central(argument: int) -> float:
        values = [xp, xc, chi]
        plus = values.copy()
        minus = values.copy()
        plus[argument] += h
        minus[argument] -= h
        return (lagrangian(*plus) - lagrangian(*minus)) / (2 * h)

    y = xp / m_phi**4
    p_y = p1 + 2 * p2 * y + 3 * p3 * y**2
    u = 4 * xp**2 / (m_pl**4 * a0**4)
    z = (1 + u) ** 2
    z_u = 2 * (1 + u)
    u_x = 8 * xp / (m_pl**4 * a0**4)
    l_chi = xc - m**2 * chi**2 / 2
    analytic = (p_y + l_chi * z_u * u_x, z, -z * m**2 * chi)
    finite = tuple(central(index) for index in range(3))
    errors = tuple(_relative_error(a, e) for a, e in zip(finite, analytic, strict=True))
    tolerance = max(
        float(contract["numeric_absolute_tolerance"]),
        float(contract["numeric_relative_tolerance"]),
    )
    r1, r2 = (float(item) for item in probe["green_radii"])
    kernel1 = math.exp(-m * r1) / r1
    kernel2 = math.exp(-m * r2) / r2
    inferred_m = -math.log((r2 * kernel2) / (r1 * kernel1)) / (r2 - r1)
    gate_actual = 1 / z
    gate_expected = (1 / (1 + u)) ** 2
    weak_probe = probe["weak_field_probe"]
    phi_e = float(weak_probe["Phi_E"])
    psi_e = float(weak_probe["Psi_E"])
    conformal = float(weak_probe["a"])
    _require(conformal != 0.0, "weak-field numeric conformal probe must be nonzero")
    phi_physical = phi_e + conformal
    psi_physical = psi_e - conformal
    lens_actual = phi_physical + psi_physical
    lens_expected = phi_e + psi_e
    numeric_records = [
        ("N1_DERIVATIVE_XPHI_FINITE_DIFFERENCE", finite[0], analytic[0], errors[0]),
        ("N2_DERIVATIVE_XCHI_FINITE_DIFFERENCE", finite[1], analytic[1], errors[1]),
        ("N3_DERIVATIVE_CHI_FINITE_DIFFERENCE", finite[2], analytic[2], errors[2]),
        (
            "N4_GATE_AMPLITUDE_B_SQUARED",
            gate_actual,
            gate_expected,
            _relative_error(gate_actual, gate_expected),
        ),
        ("N5_GREEN_FIXED_RANGE", inferred_m, m, _relative_error(inferred_m, m)),
        (
            "N6_CONFORMAL_LENSING_SUM",
            lens_actual,
            lens_expected,
            _relative_error(lens_actual, lens_expected),
        ),
    ]
    checks = [
        {
            "check_id": check_id,
            "actual": format(actual, ".17g"),
            "expected": format(expected, ".17g"),
            "scaled_error": format(error, ".17g"),
            "passed": error <= tolerance,
        }
        for check_id, actual, expected, error in numeric_records
    ]
    lens_check = next(item for item in checks if item["check_id"] == "N6_CONFORMAL_LENSING_SUM")
    lens_check["inputs"] = {
        "Phi_E": format(phi_e, ".17g"),
        "Psi_E": format(psi_e, ".17g"),
        "a": format(conformal, ".17g"),
    }
    lens_check["derived"] = {
        "Phi": format(phi_physical, ".17g"),
        "Psi": format(psi_physical, ".17g"),
    }
    _require(
        tuple(item["check_id"] for item in checks) == NUMERIC_CHECK_IDS, "numeric order changed"
    )
    return {
        "method": contract["independent_numeric_method"],
        "checks": checks,
        "all_passed": all(item["passed"] for item in checks),
        "maximum_scaled_error": format(max(float(item["scaled_error"]) for item in checks), ".17g"),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "derivation_id",
            "status",
            "purpose",
            "predecessor_binding",
            "action_and_invariants",
            "frozen_assumptions",
            "expected_equations",
            "machine_check_contract",
            "adjudication_contract",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    predecessor = config["predecessor_binding"]
    _require(predecessor["git_commit"] == "27d8cae5", "predecessor commit changed")
    _require(
        predecessor["receipt_decision"].startswith("BLOCKED_"),
        "predecessor claim ceiling changed",
    )
    action = config["action_and_invariants"]
    _require("Z(u)*(X_chi-m_chi^2*chi^2/2)" in action["scalar_lagrangian"], "action changed")
    checks = config["machine_check_contract"]
    _require(
        tuple(checks["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS, "symbolic checks changed"
    )
    _require(
        tuple(checks["required_numeric_checks"]) == NUMERIC_CHECK_IDS, "numeric checks changed"
    )
    adjudication = config["adjudication_contract"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["H2_general_covariant_scalar_equations"] == "UNVERIFIED_STORED_CONTRACT_ONLY",
        "covariant scalar-equation claim changed",
    )
    _require(
        adjudication["H2_full_metric_variation"].startswith("BLOCKED_"), "metric gate unlocked"
    )
    _require(
        adjudication["H2_hamiltonian_constraints"].startswith("BLOCKED_"),
        "constraint gate unlocked",
    )
    _require(adjudication["H3_no_ghost_health"].startswith("BLOCKED_"), "ghost gate unlocked")
    _require(adjudication["H4_global_hyperbolicity"].startswith("BLOCKED_"), "H4 unlocked")
    _require(
        adjudication["H9_joint_matter_lensing_completion"].startswith("BLOCKED_"), "H9 unlocked"
    )
    _require(
        all(value is False for value in config["claim_boundary"].values()), "claim overstatement"
    )
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()), "access changed"
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
        receipt.get("schema_version") == binding["receipt_schema_version"], "receipt schema changed"
    )
    _require(
        receipt.get("content_sha256") == binding["receipt_content_sha256"],
        "receipt content changed",
    )
    _require(receipt.get("decision") == binding["receipt_decision"], "receipt decision changed")


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    _validate_predecessor(root, config)
    _require((root / SOURCE_PATH).is_file(), "source missing")
    _require((root / TEST_PATH).is_file(), "test missing")
    symbolic = run_symbolic_suite()
    numeric = run_numeric_suite(config)
    _require(symbolic["all_passed"] is True, "symbolic suite failed")
    _require(numeric["all_passed"] is True, "numeric suite failed")
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "derivation_id": config["derivation_id"],
        "status": "partial_bounded_symbolic_derivation_passed_full_covariant_gates_blocked",
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
        "action_and_invariants": config["action_and_invariants"],
        "frozen_assumptions": config["frozen_assumptions"],
        "expected_equations": config["expected_equations"],
        "symbolic_suite": symbolic,
        "independent_numeric_suite": numeric,
        "adjudication": config["adjudication_contract"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "independent_numeric_checks": len(numeric["checks"]),
            "independent_numeric_checks_passed": sum(item["passed"] for item in numeric["checks"]),
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "Generic scalar Lagrangian coefficient identities are machine-checked, but the stored general covariant field-equation strings are unverified.",
            "The machine-derived scalar Euler operators are limited to a homogeneous 0+1 flat external metric; Q_i=-delta(L_m)/delta(field_i) is implemented by L_m,0+1=-Q_i*field_i so the covariant-oriented operator equals +Q_i.",
            "The kinetic, gradient, mass, and sound-speed blocks apply only to a locally constant homogeneous timelike phi background with chi_bar=0 and no metric-scalar principal mixing.",
            "The chi_bar=0 matrices describe an on-shell perturbation background only when Q_chi=0 and the remaining background equations hold; otherwise they are off-shell Hessians.",
            "The machine-checked Noether identity is the exact autonomous 0+1 reduction, not the four-dimensional diffeomorphism identity.",
            "The conformal lensing cancellation is linear about constant scalar backgrounds; it does not derive a viable disformal lensing law.",
            "The Yukawa range and B^2 amplitude are local constant-Z results outside the point source; gradients of Z, curved geometry, boundaries, and source contact terms remain blocked.",
            "No result establishes a healthy action, full H2, global H3/H4, joint H9, observational support, or publication readiness.",
        ],
    }
    receipt = {**body, "content_sha256": _sha(body)}
    validate_receipt(receipt, config)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    body = dict(receipt)
    content = body.pop("content_sha256", None)
    _require(content == _sha(body), "receipt content hash changed")
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt.get("decision") == DECISION, "receipt decision changed")
    _require(
        receipt.get("config_binding", {}).get("content_sha256") == _sha(config), "config unbound"
    )
    _require(
        receipt.get("action_and_invariants") == config["action_and_invariants"], "action changed"
    )
    _require(receipt.get("adjudication") == config["adjudication_contract"], "adjudication changed")
    _require(receipt.get("claim_boundary") == config["claim_boundary"], "claims changed")
    counts = receipt.get("counts")
    _require(isinstance(counts, dict), "counts missing")
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 20,
        "symbolic count changed",
    )
    _require(
        counts["independent_numeric_checks"] == counts["independent_numeric_checks_passed"] == 6,
        "numeric count changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise GravityMatterLensingSymbolicDerivationError(
            f"refusing to overwrite different receipt: {path}"
        )
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
            raise GravityMatterLensingSymbolicDerivationError(
                f"concurrent creator won; output preserved: {path}"
            ) from exc
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> tuple[dict[str, Any], str]:
    receipt = build_receipt(root)
    status = _atomic_no_replace(root.resolve() / OUTPUT_PATH, _canonical_bytes(receipt))
    return receipt, status


def check_receipt(root: Path = Path(".")) -> dict[str, Any]:
    config = load_config(root)
    expected = build_receipt(root)
    stored = _read_json(root.resolve() / OUTPUT_PATH)
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
        "numeric_checks_passed": receipt["counts"]["independent_numeric_checks_passed"],
        "full_H2_passed": receipt["claim_boundary"]["full_H2_passed"],
        "healthy_action_established": receipt["claim_boundary"]["healthy_action_established"],
        "observational_files_opened": receipt["counts"]["observational_files_opened"],
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
    except GravityMatterLensingSymbolicDerivationError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
