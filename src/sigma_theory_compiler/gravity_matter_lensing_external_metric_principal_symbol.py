"""Restricted external-metric principal symbol for the two-scalar gravity template."""

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

CONFIG_PATH = Path("configs/gravity_matter_lensing_external_metric_principal_symbol_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_external_metric_principal_symbol.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_external_metric_principal_symbol.py")
OUTPUT_PATH = Path("runs/gravity/theory/matter-lensing-external-metric-principal-symbol-v1.json")
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-external-metric-principal-symbol-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-external-metric-principal-symbol-receipt-1.0"
DECISION = (
    "PARTIAL_H3_SCALAR_EXTERNAL_METRIC_AND_H4_CONSTANT_COEFFICIENT_SYMBOL_DERIVED_"
    "U_ONE_THIRD_OBSTRUCTION_PRESERVED"
)
EXPECTED_CONFIG_FILE_SHA256 = "c0c937c1e67df4ab5caa55c1ef20cf16a84f92205a4a085e56457e8009c74903"
EXPECTED_CONFIG_CONTENT_SHA256 = "5a526c4333ebf666fefca3ca4df5a98e05fa852ffec71792ebc41b5c99193440"

SYMBOLIC_CHECK_IDS = (
    "S01_QUADRATIC_HESSIAN_PHI_PHI",
    "S02_QUADRATIC_HESSIAN_CHI_CHI",
    "S03_QUADRATIC_HESSIAN_CROSS",
    "S04_LINEARIZED_EOM_PHI_PHI",
    "S05_LINEARIZED_EOM_CHI_CHI",
    "S06_LINEARIZED_EOM_CROSS",
    "S07_INDEPENDENT_SYMBOL_MATCH",
    "S08_GENERAL_DETERMINANT",
    "S09_TIMELIKE_KINETIC_MATRIX",
    "S10_TIMELIKE_GRADIENT_MATRIX",
    "S11_TIMELIKE_DETERMINANT",
    "S12_TIMELIKE_SOUND_POLYNOMIAL",
    "S13_SPACELIKE_TIME_BLOCK",
    "S14_SPACELIKE_TRANSVERSE_BLOCK",
    "S15_SPACELIKE_LONGITUDINAL_BLOCK",
    "S16_SPACELIKE_LONGITUDINAL_DETERMINANT",
    "S17_Z_X_EXACT",
    "S18_Z_XX_EXACT",
    "S19_U_ONE_THIRD_FACTORIZATION",
    "S20_OBSTRUCTION_DETERMINANT_SLICE",
    "S21_XPHI_ZERO_REGULAR_FORM",
    "S22_CONSTANT_Z_YUKAWA_SYMBOL",
    "S23_CONSTANT_Z_YUKAWA_RANGE",
    "S24_CHI_GRADIENT_ZERO_REGRESSION",
    "S25_DEEP_AQUAL_LONGITUDINAL_RATIO",
    "S26_DEEP_AQUAL_TRANSITION_DEGENERACY",
    "S27_TIMELIKE_COMMON_COVECTOR_PRECHECK",
    "S28_SPACELIKE_COMMON_COVECTOR_PRECHECK",
)


class GravityMatterLensingPrincipalSymbolError(RuntimeError):
    """Raised when a binding, derivation, or publication gate fails."""


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
        raise GravityMatterLensingPrincipalSymbolError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GravityMatterLensingPrincipalSymbolError(message)


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    _require(set(value) == keys, f"{label} keys changed")


def _zero(value: Any) -> bool:
    if isinstance(value, sp.MatrixBase):
        return all(sp.simplify(item) == 0 for item in value)
    if isinstance(value, (tuple, list)):
        return all(_zero(item) for item in value)
    return sp.simplify(value) == 0


def _render_residual(value: Any) -> Any:
    if isinstance(value, sp.MatrixBase):
        simplified = value.applyfunc(sp.simplify)
        if _zero(simplified):
            return "ZERO_MATRIX"
        return [[sp.sstr(item) for item in simplified.row(row)] for row in range(simplified.rows)]
    if isinstance(value, (tuple, list)):
        return [_render_residual(item) for item in value]
    simplified = sp.simplify(value)
    return "0" if simplified == 0 else sp.sstr(simplified)


def _check(check_id: str, residual: Any, statement: str) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "passed": _zero(residual),
        "residual": _render_residual(residual),
        "statement": statement,
    }


def _local_jet_symbol_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    g = sp.diag(-1, 1, 1, 1)
    p = sp.Matrix(sp.symbols("p0:4", real=True))
    q = sp.Matrix(sp.symbols("q0:4", real=True))
    k = sp.Matrix(sp.symbols("k0:4", real=True))
    p_up = g * p
    q_up = g * q
    x_phi = -sp.Rational(1, 2) * (p.T * g * p)[0]
    x_chi = -sp.Rational(1, 2) * (q.T * g * q)[0]
    lower_chi, f0, f1, f2, z0, z1, z2 = sp.symbols("U_chi F0 F1 F2 Z0 Z1 Z2", real=True)
    f = f0 + f1 * x_phi + f2 * x_phi**2 / 2
    z = z0 + z1 * x_phi + z2 * x_phi**2 / 2
    chi_bracket = x_chi - lower_chi
    lagrangian = f + z * chi_bracket
    c = f1 + f2 * x_phi + (z1 + z2 * x_phi) * chi_bracket
    c_x = f2 + z2 * chi_bracket
    z_x = z1 + z2 * x_phi

    h_pp = sp.hessian(lagrangian, tuple(p))
    h_qq = sp.hessian(lagrangian, tuple(q))
    h_pq = sp.Matrix(4, 4, lambda i, j: sp.diff(lagrangian, p[i], q[j]))
    expected_pp = sp.Matrix(4, 4, lambda i, j: c * g[i, j] - c_x * p_up[i] * p_up[j])
    expected_qq = z * g
    expected_pq = sp.Matrix(4, 4, lambda i, j: -z_x * p_up[i] * q_up[j])
    expected_qp = expected_pq.T

    # Independent route: linearize the Euler-Lagrange fluxes C*p^mu and Z*q^mu.
    phi_flux = c * p_up
    chi_flux = z * q_up
    eom_pp = sp.Matrix(4, 4, lambda i, j: sp.diff(phi_flux[i], p[j]))
    eom_pq = sp.Matrix(4, 4, lambda i, j: sp.diff(phi_flux[i], q[j]))
    eom_qp = sp.Matrix(4, 4, lambda i, j: sp.diff(chi_flux[i], p[j]))
    eom_qq = sp.Matrix(4, 4, lambda i, j: sp.diff(chi_flux[i], q[j]))

    k2 = (k.T * g * k)[0]
    vk = (p_up.T * k)[0]
    wk = (q_up.T * k)[0]
    symbol_from_hessian = sp.Matrix(
        [
            [(-h_pp * k).dot(k), (-h_pq * k).dot(k)],
            [(-h_pq.T * k).dot(k), (-h_qq * k).dot(k)],
        ]
    )
    symbol_from_eom = sp.Matrix(
        [
            [(eom_pp * k).dot(k), (eom_pq * k).dot(k)],
            [(eom_qp * k).dot(k), (eom_qq * k).dot(k)],
        ]
    )
    expected_symbol = sp.Matrix(
        [
            [c * k2 - c_x * vk**2, -z_x * vk * wk],
            [-z_x * vk * wk, z * k2],
        ]
    )
    expected_determinant = c * z * k2**2 - c_x * z * k2 * vk**2 - z_x**2 * vk**2 * wk**2
    checks = [
        _check(
            "S01_QUADRATIC_HESSIAN_PHI_PHI",
            -h_pp - expected_pp,
            "The negative quadratic gradient Hessian gives C*g^{mu nu}-C_X*v^mu*v^nu.",
        ),
        _check(
            "S02_QUADRATIC_HESSIAN_CHI_CHI",
            -h_qq - expected_qq,
            "The chi diagonal principal tensor is Z*g^{mu nu}.",
        ),
        _check(
            "S03_QUADRATIC_HESSIAN_CROSS",
            (-h_pq - expected_pq, -h_pq.T - expected_qp),
            "The two cross Hessian blocks are -Z_X*v^mu*w^nu and its transpose.",
        ),
        _check(
            "S04_LINEARIZED_EOM_PHI_PHI",
            eom_pp - expected_pp,
            "Independent linearization of the phi flux recovers the phi principal tensor.",
        ),
        _check(
            "S05_LINEARIZED_EOM_CHI_CHI",
            eom_qq - expected_qq,
            "Independent linearization of the chi flux recovers the chi principal tensor.",
        ),
        _check(
            "S06_LINEARIZED_EOM_CROSS",
            (eom_pq - expected_pq, eom_qp - expected_qp),
            "Independent flux linearization recovers both cross-principal blocks.",
        ),
        _check(
            "S07_INDEPENDENT_SYMBOL_MATCH",
            symbol_from_hessian - symbol_from_eom,
            "The Hessian and independently linearized Euler-Lagrange symbols agree.",
        ),
        _check(
            "S08_GENERAL_DETERMINANT",
            sp.det(expected_symbol) - expected_determinant,
            "The exact two-by-two determinant retains both background-gradient contractions.",
        ),
    ]
    expressions = {
        "local_Taylor_jet_basis": {
            "F": "F0+F1*X_phi+F2*X_phi^2/2",
            "Z": "Z0+Z1*X_phi+Z2*X_phi^2/2",
            "interpretation": "F1,F2,Z1,Z2 are arbitrary first and second derivative values at the local jet, so the principal result is general for twice-differentiable F and Z.",
        },
        "matrix": [
            ["C*k2-C_X*vk^2", "-Z_X*vk*wk"],
            ["-Z_X*vk*wk", "Z*k2"],
        ],
        "determinant": "C*Z*k2^2-C_X*Z*k2*vk^2-Z_X^2*vk^2*wk^2",
    }
    return checks, expressions


def _aligned_block_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    c, c_x, z, z_x = sp.symbols("C C_X Z Z_X", real=True)
    conformal, m_phi_fourth = sp.symbols("A_conformal M_phi4", positive=True)
    disformal = sp.symbols("d_phi", real=True)
    x_phi, x_chi, omega, k_long, k_trans, speed2 = sp.symbols(
        "X_phi X_chi omega k_L k_T c_s2", real=True
    )
    a, b = sp.symbols("a b", positive=True)

    k2_t = -(omega**2) + k_long**2
    vk_t = a * omega
    wk_t = b * omega
    symbol_t = sp.Matrix(
        [
            [c * k2_t - c_x * vk_t**2, -z_x * vk_t * wk_t],
            [-z_x * vk_t * wk_t, z * k2_t],
        ]
    )
    kinetic = sp.Matrix(
        [
            [c + 2 * x_phi * c_x, 2 * z_x * sp.sqrt(x_phi) * sp.sqrt(x_chi)],
            [2 * z_x * sp.sqrt(x_phi) * sp.sqrt(x_chi), z],
        ]
    )
    gradient = sp.diag(c, z)
    timelike_subs = {a**2: 2 * x_phi, b**2: 2 * x_chi, a * b: sp.sqrt(4 * x_phi * x_chi)}
    symbol_t = symbol_t.subs(timelike_subs)
    determinant = z * (c + 2 * x_phi * c_x) - 4 * x_phi * x_chi * z_x**2
    sound_polynomial = determinant * speed2**2 - z * (2 * c + 2 * x_phi * c_x) * speed2 + c * z

    k2_s = -(omega**2) + k_long**2 + k_trans**2
    vk_s = a * k_long
    wk_s = b * k_long
    symbol_s = sp.Matrix(
        [
            [c * k2_s - c_x * vk_s**2, -z_x * vk_s * wk_s],
            [-z_x * vk_s * wk_s, z * k2_s],
        ]
    )
    spacelike_subs = {
        a**2: -2 * x_phi,
        b**2: -2 * x_chi,
        a * b: sp.sqrt(4 * x_phi * x_chi),
    }
    symbol_s = symbol_s.subs(spacelike_subs)
    time_s = sp.diag(c, z)
    transverse_s = sp.diag(c, z)
    longitudinal_s = sp.Matrix(
        [
            [
                c + 2 * x_phi * c_x,
                -2 * z_x * sp.sqrt(-x_phi) * sp.sqrt(-x_chi),
            ],
            [-2 * z_x * sp.sqrt(-x_phi) * sp.sqrt(-x_chi), z],
        ]
    )
    physical_delta = 1 - 2 * disformal * x_phi / m_phi_fourth
    physical_timelike = conformal**2 * sp.diag(-physical_delta, 1, 1, 1)
    physical_spacelike = conformal**2 * sp.diag(-1, physical_delta, 1, 1)

    checks = [
        _check(
            "S09_TIMELIKE_KINETIC_MATRIX",
            -symbol_t.diff(omega, 2) / 2 - kinetic,
            "Aligned timelike jets give the exact mixed time-kinetic matrix.",
        ),
        _check(
            "S10_TIMELIKE_GRADIENT_MATRIX",
            symbol_t.diff(k_long, 2) / 2 - gradient,
            "Aligned timelike jets have diagonal spatial-gradient matrix diag(C,Z).",
        ),
        _check(
            "S11_TIMELIKE_DETERMINANT",
            sp.det(kinetic) - determinant,
            "The timelike Sylvester determinant retains the Z_X mixing penalty.",
        ),
        _check(
            "S12_TIMELIKE_SOUND_POLYNOMIAL",
            sp.det(speed2 * kinetic - gradient) - sound_polynomial,
            "The two squared sound speeds are roots of the frozen generalized-eigenvalue polynomial.",
        ),
        _check(
            "S13_SPACELIKE_TIME_BLOCK",
            -symbol_s.diff(omega, 2) / 2 - time_s,
            "Aligned spacelike jets retain diagonal time matrix diag(C,Z).",
        ),
        _check(
            "S14_SPACELIKE_TRANSVERSE_BLOCK",
            symbol_s.diff(k_trans, 2) / 2 - transverse_s,
            "The transverse spatial block equals diag(C,Z).",
        ),
        _check(
            "S15_SPACELIKE_LONGITUDINAL_BLOCK",
            symbol_s.diff(k_long, 2) / 2 - longitudinal_s,
            "The longitudinal spatial block contains the exact gradient mixing.",
        ),
        _check(
            "S16_SPACELIKE_LONGITUDINAL_DETERMINANT",
            sp.det(longitudinal_s) - determinant,
            "The aligned spacelike longitudinal determinant has the same algebraic form.",
        ),
        _check(
            "S27_TIMELIKE_COMMON_COVECTOR_PRECHECK",
            (
                symbol_t.subs({omega: 1, k_long: 0}) + kinetic,
                physical_timelike.inv()[0, 0] + 1 / (conformal**2 * physical_delta),
            ),
            "dt evaluates to minus the positive scalar kinetic form and has physical inverse-metric norm -1/(A^2*Delta) when Delta=1-2*d_phi*X_phi/M_phi^4>0.",
        ),
        _check(
            "S28_SPACELIKE_COMMON_COVECTOR_PRECHECK",
            (
                symbol_s.subs({omega: 1, k_long: 0, k_trans: 0}) + time_s,
                physical_spacelike.inv()[0, 0] + 1 / conformal**2,
            ),
            "dt evaluates to minus diag(C,Z) and has physical inverse-metric norm -1/A^2 on aligned spacelike jets when the physical spatial signature condition Delta>0 holds.",
        ),
    ]
    expressions = {
        "timelike": {
            "K": [["C+2*X_phi*C_X", "Z_X*sqrt(4*X_phi*X_chi)"], ["Z_X*sqrt(4*X_phi*X_chi)", "Z"]],
            "G": [["C", "0"], ["0", "Z"]],
            "det_K": "Z*(C+2*X_phi*C_X)-4*X_phi*X_chi*Z_X^2",
            "conditions": ["C>0", "Z>0", "C+2*X_phi*C_X>0", "det_K>0"],
        },
        "spacelike": {
            "K": [["C", "0"], ["0", "Z"]],
            "G_transverse": [["C", "0"], ["0", "Z"]],
            "G_longitudinal": [
                ["C+2*X_phi*C_X", "-Z_X*sqrt(4*X_phi*X_chi)"],
                ["-Z_X*sqrt(4*X_phi*X_chi)", "Z"],
            ],
            "det_G_longitudinal": "Z*(C+2*X_phi*C_X)-4*X_phi*X_chi*Z_X^2",
            "conditions": ["C>0", "Z>0", "C+2*X_phi*C_X>0", "det_G_longitudinal>0"],
        },
        "physical_metric_common_cone_precheck": {
            "Delta": "1-2*d_phi*X_phi/M_phi^4",
            "timelike_aligned_covariant_metric": "A^2*diag(-Delta,1,1,1)",
            "timelike_dt_inverse_norm": "-1/(A^2*Delta)",
            "spacelike_aligned_covariant_metric": "A^2*diag(-1,Delta,1,1)",
            "spacelike_dt_inverse_norm": "-1/A^2",
            "condition": "A>0 and Delta>0",
            "scope": "algebraic local-cone precheck only; metric-scalar and matter principal systems are absent",
        },
    }
    return checks, expressions


def _gate_transition_regression_checks() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    x, beta, x_chi, p_y, p_xx = sp.symbols("X_phi beta X_chi P_y P_XX", real=True)
    u = beta * x**2
    z = (1 + u) ** 2
    z_x = sp.diff(z, x)
    z_xx = sp.diff(z_x, x)
    factor = z * (z_x + 2 * x * z_xx) - 4 * x * z_x**2
    factor_expected = 12 * beta * x * (1 + u) ** 2 * (1 - 3 * u)
    determinant_slice = z * (p_y + 2 * x * p_xx) + x_chi * factor
    determinant_from_c = (
        z * (p_y + z_x * x_chi + 2 * x * (p_xx + z_xx * x_chi)) - 4 * x * x_chi * z_x**2
    )

    k2, vk, wk, c, c_x, z_symbol, z_x_symbol = sp.symbols("k2 vk wk C C_X Z Z_X", real=True)
    general_symbol = sp.Matrix(
        [[c * k2 - c_x * vk**2, -z_x_symbol * vk * wk], [-z_x_symbol * vk * wk, z_symbol * k2]]
    )
    constant_z_symbol = general_symbol.subs({z_x_symbol: 0, c: p_y, c_x: p_xx})
    expected_constant_z = sp.diag(p_y * k2 - p_xx * vk**2, z_symbol * k2)

    radius, mass = sp.symbols("r m", positive=True)
    yukawa = sp.exp(-mass * radius) / radius
    radial_operator = sp.diff(yukawa, radius, 2) + 2 * sp.diff(yukawa, radius) / radius
    aqual_amplitude = sp.symbols("A", positive=True)
    p_y_aqual = aqual_amplitude * sp.sqrt(-x)
    p_long_aqual = sp.simplify(p_y_aqual + 2 * x * sp.diff(p_y_aqual, x))

    checks = [
        _check(
            "S17_Z_X_EXACT",
            z_x - 4 * beta * x * (1 + u),
            "Z_X is differentiated exactly from Z=(1+beta*X_phi^2)^2.",
        ),
        _check(
            "S18_Z_XX_EXACT",
            z_xx - 4 * beta * (1 + 3 * u),
            "Z_XX is differentiated exactly from the same frozen gate.",
        ),
        _check(
            "S19_U_ONE_THIRD_FACTORIZATION",
            factor - factor_expected,
            "The chi-gradient contribution factors exactly through (1-3*u).",
        ),
        _check(
            "S20_OBSTRUCTION_DETERMINANT_SLICE",
            determinant_from_c - determinant_slice,
            "At chi_bar=0 with nonzero timelike X_chi, the determinant separates into the P sector plus the sign-changing gate contribution.",
        ),
        _check(
            "S21_XPHI_ZERO_REGULAR_FORM",
            sp.limit(factor_expected, x, 0),
            "The apparent factorized 1/X_phi expression has a regular zero limit.",
        ),
        _check(
            "S22_CONSTANT_Z_YUKAWA_SYMBOL",
            constant_z_symbol - expected_constant_z,
            "Z_X=Z_XX=0 removes principal mixing and leaves the constant-Z chi wave symbol.",
        ),
        _check(
            "S23_CONSTANT_Z_YUKAWA_RANGE",
            radial_operator - mass**2 * yukawa,
            "For r>0, constant Z leaves the Yukawa inverse range equal to m_chi.",
        ),
        _check(
            "S24_CHI_GRADIENT_ZERO_REGRESSION",
            general_symbol.subs(wk, 0) - sp.diag(c * k2 - c_x * vk**2, z_symbol * k2),
            "w_mu=0 removes cross-principal mixing and recovers the predecessor diagonal block.",
        ),
        _check(
            "S25_DEEP_AQUAL_LONGITUDINAL_RATIO",
            p_long_aqual / p_y_aqual - 2,
            "For P_y proportional to sqrt(-X_phi), the spacelike longitudinal speed ratio is exactly two.",
        ),
        _check(
            "S26_DEEP_AQUAL_TRANSITION_DEGENERACY",
            (sp.limit(p_y_aqual, x, 0, dir="-"), sp.limit(p_long_aqual, x, 0, dir="-")),
            "Both deep-AQUAL principal coefficients vanish at X_phi->0- despite their finite ratio.",
        ),
    ]
    expressions = {
        "Z": "(1+u)^2",
        "Z_X": "4*beta*X_phi*(1+u)",
        "Z_XX": "4*beta*(1+3*u)",
        "obstruction_factor": "12*beta*X_phi*(1+u)^2*(1-3*u)",
        "equivalent_nonzero_Xphi_form": "12*u*(1+u)^2*(1-3*u)/X_phi",
        "determinant_slice": "Z*(P_y+2*y*P_yy)+X_chi*12*u*(1+u)^2*(1-3*u)/X_phi",
        "deep_AQUAL": {
            "P_y": "A*sqrt(-X_phi)",
            "P_y_plus_2X_P_XX": "2*A*sqrt(-X_phi)",
            "longitudinal_speed_squared": "2",
            "transition": "both coefficients vanish as X_phi->0-",
        },
    }
    return checks, expressions


def run_symbolic_suite() -> dict[str, Any]:
    local, local_expressions = _local_jet_symbol_checks()
    aligned, aligned_expressions = _aligned_block_checks()
    gate, gate_expressions = _gate_transition_regression_checks()
    checks = local + aligned[:8] + gate + aligned[8:]
    _require(
        tuple(item["check_id"] for item in checks) == SYMBOLIC_CHECK_IDS,
        "symbolic check order changed",
    )
    _require(all(item["passed"] for item in checks), "symbolic derivation failed")
    return {
        "engine": f"sympy-{sp.__version__}",
        "derivation_routes": [
            "negative exact quadratic gradient Hessian",
            "independent linearization of Euler-Lagrange fluxes",
        ],
        "checks": checks,
        "all_passed": True,
        "expressions": {
            "general_local_jet": local_expressions,
            "aligned_blocks": aligned_expressions,
            "gate_transition_and_regressions": gate_expressions,
        },
    }


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    tolerance = float(config["machine_check_contract"]["numeric_tolerance"])
    records: list[dict[str, Any]] = []
    for probe in config["machine_check_contract"]["frozen_numeric_probes"]:
        u = float(probe["u"])
        beta = float(probe["beta"])
        x_phi = math.sqrt(u / beta)
        x_chi = float(probe["X_chi"])
        p_y = float(probe["P_y"])
        p_xx = float(probe["P_XX"])
        z = (1.0 + u) ** 2
        z_x = 4.0 * beta * x_phi * (1.0 + u)
        z_xx = 4.0 * beta * (1.0 + 3.0 * u)
        c = p_y + z_x * x_chi
        c_x = p_xx + z_xx * x_chi
        k_phi = c + 2.0 * x_phi * c_x
        determinant_direct = z * k_phi - 4.0 * x_phi * x_chi * z_x**2
        factor_direct = z * (z_x + 2.0 * x_phi * z_xx) - 4.0 * x_phi * z_x**2
        factorized = 12.0 * u * (1.0 + u) ** 2 * (1.0 - 3.0 * u) / x_phi
        factor_error = abs(factor_direct - factorized) / max(1.0, abs(factorized))
        gate_contribution = x_chi * factor_direct
        gate_sign = (
            "positive" if gate_contribution > 0 else "negative" if gate_contribution < 0 else "zero"
        )
        determinant_sign = (
            "positive"
            if determinant_direct > 0
            else "negative"
            if determinant_direct < 0
            else "zero"
        )
        passed = (
            factor_error <= tolerance
            and gate_sign == probe["expected_gate_contribution_sign"]
            and determinant_sign == probe["expected_determinant_sign"]
            and c > 0
            and z > 0
            and k_phi > 0
        )
        records.append(
            {
                "probe_id": probe["probe_id"],
                "u": format(u, ".17g"),
                "X_phi": format(x_phi, ".17g"),
                "X_chi": format(x_chi, ".17g"),
                "C": format(c, ".17g"),
                "Z": format(z, ".17g"),
                "K_phi_phi": format(k_phi, ".17g"),
                "gate_contribution": format(gate_contribution, ".17g"),
                "gate_contribution_sign": gate_sign,
                "kinetic_determinant": format(determinant_direct, ".17g"),
                "kinetic_determinant_sign": determinant_sign,
                "factorization_scaled_error": format(factor_error, ".17g"),
                "passed": passed,
            }
        )
    _require(all(item["passed"] for item in records), "numeric probe suite failed")
    _require(
        records[-1]["probe_id"] == "above_u_one_third_designed_failure"
        and records[-1]["kinetic_determinant_sign"] == "negative",
        "designed failure was not preserved",
    )
    return {
        "method": config["machine_check_contract"]["numeric_method"],
        "tolerance": format(tolerance, ".17g"),
        "probes": records,
        "all_passed": True,
        "designed_failure_preserved": True,
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "derivation_id",
            "status",
            "purpose",
            "predecessor_bindings",
            "scope_and_conventions",
            "principal_symbol_contract",
            "aligned_timelike_contract",
            "aligned_spacelike_contract",
            "designed_obstruction",
            "transition_and_regressions",
            "machine_check_contract",
            "adjudication",
            "claim_boundary",
            "zero_access_and_compute",
            "output_path",
        },
        "principal-symbol config",
    )
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["derivation_id"] == "gravity-matter-lensing-external-metric-principal-symbol-v1",
        "derivation identity changed",
    )
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    bindings = config["predecessor_bindings"]
    _require(
        tuple(item["binding_id"] for item in bindings)
        == ("theory_preflight", "bounded_symbolic_derivation"),
        "predecessor inventory changed",
    )
    for item in bindings:
        _strict(
            item,
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
            f"predecessor binding {item.get('binding_id')}",
        )
    _require(
        tuple(config["machine_check_contract"]["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic contract changed",
    )
    obstruction = config["designed_obstruction"]
    _require(obstruction["no_go_claim"] is False, "obstruction overstated")
    _require("1-3*u" in obstruction["exact_identity"], "u=1/3 identity changed")
    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    _require(
        adjudication["H3_scalar_external_metric"].startswith("PARTIAL_"),
        "H3 scope changed",
    )
    _require(
        adjudication["H4_constant_coefficient"].startswith("PARTIAL_"),
        "H4 scope changed",
    )
    _require(
        all(
            value is False
            for key, value in adjudication.items()
            if key
            not in {"H3_scalar_external_metric", "H4_constant_coefficient", "overall_decision"}
        ),
        "full adjudication gate unlocked",
    )
    _require(
        all(value is False for value in config["claim_boundary"].values()),
        "claim boundary overstated",
    )
    _require(
        all(value == 0 for value in config["zero_access_and_compute"].values()),
        "access or compute state changed",
    )


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    path = root / CONFIG_PATH
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
            _require(path.is_file(), f"predecessor missing: {binding['binding_id']} {path_key}")
            _require(
                _file_sha(path) == binding[hash_key],
                f"predecessor changed: {binding['binding_id']} {path_key}",
            )
        receipt = _read_json(root / binding["receipt_path"])
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
    _require((root / SOURCE_PATH).is_file(), "implementation source missing")
    _require((root / TEST_PATH).is_file(), "implementation test missing")
    symbolic = run_symbolic_suite()
    numeric = run_numeric_suite(config)
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "derivation_id": config["derivation_id"],
        "status": "partial_external_metric_scalar_symbol_derived_designed_obstruction_preserved",
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
        "scope_and_conventions": config["scope_and_conventions"],
        "principal_symbol_contract": config["principal_symbol_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "designed_obstruction": config["designed_obstruction"],
        "transition_and_regressions": config["transition_and_regressions"],
        "adjudication": config["adjudication"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "numeric_probes": len(numeric["probes"]),
            "numeric_probes_passed": sum(item["passed"] for item in numeric["probes"]),
            "designed_failures_preserved": int(numeric["designed_failure_preserved"]),
            "observational_files_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "claim_boundary": config["claim_boundary"],
        "zero_access_and_compute": config["zero_access_and_compute"],
        "limitations": [
            "The derived system is the two-scalar principal block on a fixed external metric and constant local jets; metric perturbations, lapse/shift constraints, and disformal matter characteristics are absent.",
            "The local jets are off shell unless separate background equations and sources are solved; no Solar, galaxy, group, cluster, or FLRW background has been established.",
            "The u=1/3 sign flip is a serious determinant obstruction on the declared timelike slice, not an unconditional no-go theorem and not a healthy-domain proof.",
            "The algebraic common-covector checks on aligned constant jets do not establish variable-coefficient or global strong hyperbolicity.",
            "The deep-AQUAL X_phi->0- limit has finite longitudinal speed ratio two but vanishing principal coefficients, so strict hyperbolicity and EFT control at the transition remain unresolved.",
            "No cutoff, gravitational constraints, universal physical-metric characteristic system, lensing law, observational support, or publication claim is established.",
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
            "derivation_id",
            "status",
            "decision",
            "config_binding",
            "implementation_binding",
            "predecessor_bindings",
            "scope_and_conventions",
            "principal_symbol_contract",
            "symbolic_suite",
            "numeric_suite",
            "designed_obstruction",
            "transition_and_regressions",
            "adjudication",
            "counts",
            "claim_boundary",
            "zero_access_and_compute",
            "limitations",
            "content_sha256",
        },
        "principal-symbol receipt",
    )
    body = dict(receipt)
    content_sha = body.pop("content_sha256")
    _require(content_sha == _sha(body), "receipt content hash changed")
    _require(receipt["schema_version"] == RECEIPT_SCHEMA, "receipt schema changed")
    _require(receipt["decision"] == DECISION, "receipt decision changed")
    _require(
        receipt["config_binding"]["content_sha256"] == _sha(config),
        "receipt config binding changed",
    )
    _require(
        receipt["predecessor_bindings"] == config["predecessor_bindings"],
        "receipt predecessors changed",
    )
    _require(
        receipt["designed_obstruction"] == config["designed_obstruction"],
        "receipt obstruction changed",
    )
    _require(receipt["adjudication"] == config["adjudication"], "adjudication changed")
    _require(receipt["claim_boundary"] == config["claim_boundary"], "claims changed")
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 28,
        "symbolic count changed",
    )
    _require(
        counts["numeric_probes"] == counts["numeric_probes_passed"] == 2,
        "numeric probe count changed",
    )
    _require(counts["designed_failures_preserved"] == 1, "designed failure lost")
    _require(
        all(
            value == 0
            for key, value in counts.items()
            if key
            in {"observational_files_opened", "network_calls", "model_or_paid_calls", "gpu_calls"}
        ),
        "receipt access changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise GravityMatterLensingPrincipalSymbolError(
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
            raise GravityMatterLensingPrincipalSymbolError(
                f"concurrent creator won; output preserved: {path}"
            ) from exc
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt(root: Path = Path(".")) -> tuple[dict[str, Any], str]:
    root = root.resolve()
    receipt = build_receipt(root)
    publication = _atomic_no_replace(root / OUTPUT_PATH, _canonical_bytes(receipt))
    return receipt, publication


def check_receipt(root: Path = Path(".")) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    expected = build_receipt(root)
    stored = _read_json(root / OUTPUT_PATH)
    validate_receipt(stored, config)
    _require(stored == expected, "stored receipt differs from deterministic rebuild")
    return stored


def _summary(receipt: Mapping[str, Any], publication: str | None = None) -> dict[str, Any]:
    summary = {
        "valid": True,
        "status": receipt["status"],
        "decision": receipt["decision"],
        "content_sha256": receipt["content_sha256"],
        "symbolic_checks_passed": receipt["counts"]["symbolic_checks_passed"],
        "numeric_probes_passed": receipt["counts"]["numeric_probes_passed"],
        "designed_failures_preserved": receipt["counts"]["designed_failures_preserved"],
        "full_H3_passed": receipt["claim_boundary"]["full_H3_passed"],
        "full_H4_passed": receipt["claim_boundary"]["full_H4_passed"],
    }
    if publication is not None:
        summary["publication"] = publication
    return summary


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
    except GravityMatterLensingPrincipalSymbolError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
