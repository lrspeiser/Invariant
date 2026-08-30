"""No-data exact flat-FLRW equations and necessary cosmology conditions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import sympy as sp

CONFIG_PATH = Path("configs/gravity_matter_lensing_flrw_background_necessary_conditions_v1.json")
SOURCE_PATH = Path(
    "src/sigma_theory_compiler/gravity_matter_lensing_flrw_background_necessary_conditions.py"
)
TEST_PATH = Path("tests/test_gravity_matter_lensing_flrw_background_necessary_conditions.py")
OUTPUT_PATH = Path(
    "runs/gravity/theory/matter-lensing-flrw-background-necessary-conditions-v1.json"
)
CONFIG_SCHEMA = "invariant-gravity-matter-lensing-flrw-background-necessary-conditions-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-matter-lensing-flrw-background-necessary-conditions-receipt-1.0"
DECISION = (
    "PARTIAL_EXACT_FLRW_BACKGROUND_EQUATIONS_AND_NECESSARY_CONDITIONS_DERIVED_"
    "COSMOLOGICAL_HISTORY_AND_PHYSICAL_GATES_BLOCKED"
)
EXPECTED_CONFIG_FILE_SHA256 = "061b8ac6655cdb734f04f161daac369b78a0bf10157ff4957b1f3918c303a0fe"
EXPECTED_CONFIG_CONTENT_SHA256 = "25163367ef538618e90566e78d4cdd09f4e39684b9190b901d94ae380a1e09cb"

SYMBOLIC_CHECK_IDS = (
    "S01_LAPSE_ENERGY_ROUTE",
    "S02_COVARIANT_ENERGY_ROUTE",
    "S03_PRESSURE_SCALE_ROUTE",
    "S04_ENTHALPY_IDENTITY",
    "S05_PHI_EULER_COVARIANT_MATCH",
    "S06_CHI_EULER_COVARIANT_MATCH",
    "S07_PHI_GATE_TIME_DERIVATIVE",
    "S08_SCALAR_CONTINUITY_IDENTITY",
    "S09_PHYSICAL_TO_EINSTEIN_CONTINUITY",
    "S10_TOTAL_CONTINUITY_CANCELLATION",
    "S11_FRIEDMANN_TO_RAYCHAUDHURI",
    "S12_SPATIAL_EINSTEIN_RAYCHAUDHURI",
    "S13_CONSTANT_W_MATTER_SOLUTION",
    "S14_RADIATION_SCALING",
    "S15_DUST_SCALING",
    "S16_VACUUM_SCALING",
    "S17_X_ZERO_GATE_LIMIT",
    "S18_GATE_ENERGY_FACTORIZATION",
    "S19_GATE_ENTHALPY_FACTORIZATION",
    "S20_HIGH_U_GATE_ENERGY_SCALING",
    "S21_HIGH_U_GATE_PRESSURE_SCALING",
    "S22_HIGH_U_GATE_ENTHALPY_SCALING",
    "S23_TIMELIKE_K_COEFFICIENT",
    "S24_CONFORMAL_TENSOR_PHOTON_CONE",
    "S25_DISFORMAL_Q_ENDPOINTS",
)

TOP_KEYS = {
    "schema_version",
    "analysis_id",
    "status",
    "purpose",
    "predecessor_bindings",
    "conventions",
    "stress_energy_contract",
    "background_equations",
    "matter_exchange_contract",
    "gate_limits_and_necessary_conditions",
    "tensor_photon_and_disformal_contract",
    "machine_check_contract",
    "adjudication",
    "claim_boundary",
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
CONVENTION_KEYS = {
    "signature",
    "einstein_frame_metric",
    "cosmic_time_gauge",
    "kinetic_invariants",
    "scalar_lagrangian",
    "gate_derivatives",
    "universal_metric",
    "matter_frames",
    "source_sign",
}
STRESS_KEYS = {
    "C_definition",
    "covariant_stress",
    "scalar_energy_density",
    "scalar_pressure",
    "enthalpy",
    "lapse_route",
    "scale_route",
}
BACKGROUND_KEYS = {
    "friedmann",
    "raychaudhuri",
    "acceleration_identity",
    "phi_eom",
    "chi_eom",
    "phi_gate_content",
    "covariant_specialization",
    "constraint_propagation",
}
EXCHANGE_KEYS = {
    "physical_frame",
    "einstein_frame",
    "scalar_continuity",
    "total_continuity",
    "constant_w_solution",
    "radiation_probe",
    "dust_probe",
    "vacuum_probe",
}
LIMIT_KEYS = {
    "x_zero",
    "exact_gate_energy_factor",
    "exact_gate_enthalpy_factor",
    "high_u",
    "timelike_local_health",
    "friedmann_reality",
    "regularity",
    "history_consistency",
    "cutoff_and_perturbations",
}
CONE_KEYS = {
    "conformal_branch",
    "disformal_scope",
    "homogeneous_disformal_variable",
    "inherited_speed_threshold",
    "nonnegative_q_ceiling",
    "status",
}
MACHINE_KEYS = {
    "symbolic_engine",
    "symbolic_zero_rule",
    "required_symbolic_checks",
    "gate_u_probes",
    "disformal_q_probes",
}
ADJUDICATION_KEYS = {
    "exact_homogeneous_stress_energy_derived",
    "friedmann_raychaudhuri_derived",
    "scalar_eom_with_gate_and_sources_derived",
    "physical_einstein_exchange_derived",
    "continuity_identity_verified",
    "matter_limit_probes_derived",
    "gate_limit_obstruction_derived",
    "conformal_cone_alignment_derived",
    "disformal_necessary_interval_inherited_and_checked",
    "healthy_late_time_history_exists",
    "accelerating_solution_exists",
    "BBN_passed",
    "CMB_passed",
    "growth_passed",
    "cutoff_established",
    "perturbation_stability_established",
    "full_constrained_characteristics_established",
    "parameter_values_frozen",
    "initial_conditions_frozen",
    "observational_fit_performed",
    "overall_decision",
}
CLAIM_KEYS = {
    "restricted_flat_flrw_equations_established",
    "cosmological_solution_established",
    "late_time_acceleration_established",
    "healthy_history_established",
    "BBN_viability_established",
    "CMB_viability_established",
    "growth_viability_established",
    "cutoff_established",
    "perturbation_stability_established",
    "full_covariant_health_established",
    "GW_observational_pass_established",
    "lensing_success_established",
    "observational_support",
    "novelty_established",
    "publication_readiness_changed",
    "scientific_observational_claim_allowed",
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


class FlrwNecessaryConditionsError(RuntimeError):
    """Raised when the frozen FLRW necessary-condition package changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FlrwNecessaryConditionsError(message)


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
        raise FlrwNecessaryConditionsError(f"cannot read JSON: {path}") from exc
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


def run_symbolic_suite() -> dict[str, Any]:
    lapse, scale = sp.symbols("N a", positive=True)
    phi_dot, chi_dot = sp.symbols("phi_dot chi_dot", real=True)
    phi_ddot, chi_ddot = sp.symbols("phi_ddot chi_ddot", real=True)
    chi, mass = sp.symbols("chi m_chi", real=True)
    y0, potential, potential_phi = sp.symbols("Y0 V V_phi", real=True)
    hubble, hubble_dot, planck = sp.symbols("H H_dot M_Pl", real=True)
    alpha_phi, alpha_chi, trace_e = sp.symbols("alpha_phi alpha_chi T_E", real=True)
    x_symbol, w_symbol = sp.symbols("X X_chi", nonnegative=True)
    p_function = sp.Function("P")
    z_function = sp.Function("Z")
    x_lapse = phi_dot**2 / (2 * lapse**2)
    w_lapse = chi_dot**2 / (2 * lapse**2)
    q_mass = mass**2 * chi**2 / 2
    lagrangian_lapse = p_function(x_lapse) - potential + y0 * w_lapse - q_mass * z_function(x_lapse)
    mini_scalar = lapse * scale**3 * lagrangian_lapse
    rho_lapse = sp.simplify(-sp.diff(mini_scalar, lapse).subs(lapse, 1) / scale**3)
    p_scale = sp.simplify(sp.diff(mini_scalar, scale).subs(lapse, 1) / (3 * scale**2))

    p_value, p_x, p_xx, z_value, z_x, z_xx = sp.symbols("P P_X P_XX Z Z_X Z_XX", real=True)
    c_value = p_x - q_mass * z_x
    lagrangian = p_value - potential + y0 * w_symbol - q_mass * z_value
    rho_expected = 2 * x_symbol * c_value - lagrangian + 2 * y0 * w_symbol
    pressure_expected = lagrangian
    enthalpy = 2 * x_symbol * c_value + 2 * y0 * w_symbol

    x_cosmic = phi_dot**2 / 2
    w_cosmic = chi_dot**2 / 2
    x_argument = sp.symbols("x_argument", real=True)
    p_prime_cosmic = sp.diff(p_function(x_argument), x_argument).subs(x_argument, x_cosmic)
    z_prime_cosmic = sp.diff(z_function(x_argument), x_argument).subs(x_argument, x_cosmic)
    rho_lapse_expected = (
        phi_dot**2 * (p_prime_cosmic - q_mass * z_prime_cosmic)
        - p_function(x_cosmic)
        + potential
        + y0 * w_cosmic
        + q_mass * z_function(x_cosmic)
    )
    pressure_lapse_expected = (
        p_function(x_cosmic) - potential + y0 * w_cosmic - q_mass * z_function(x_cosmic)
    )

    rho_covariant = c_value * (2 * x_symbol) + 2 * y0 * w_symbol - pressure_expected

    time = sp.symbols("t", real=True)
    scale_t = sp.Function("a")(time)
    phi_t = sp.Function("phi")(time)
    chi_t = sp.Function("chi")(time)
    p_time = sp.Function("P_time")
    z_time = sp.Function("Z_time")
    v_time = sp.Function("V_time")
    x_time = sp.diff(phi_t, time) ** 2 / 2
    q_time = mass**2 * chi_t**2 / 2
    l_time = (
        p_time(x_time)
        - v_time(phi_t)
        + y0 * sp.diff(chi_t, time) ** 2 / 2
        - q_time * z_time(x_time)
    )
    mini_time = scale_t**3 * l_time
    phi_euler_minispace = sp.simplify(
        (sp.diff(sp.diff(mini_time, sp.diff(phi_t, time)), time) - sp.diff(mini_time, phi_t))
        / scale_t**3
        - alpha_phi * trace_e
    )
    p_time_x = sp.diff(p_time(x_symbol), x_symbol).subs(x_symbol, x_time)
    z_time_x = sp.diff(z_time(x_symbol), x_symbol).subs(x_symbol, x_time)
    c_time = p_time_x - q_time * z_time_x
    phi_euler_covariant = sp.simplify(
        sp.diff(c_time * sp.diff(phi_t, time), time)
        + 3 * sp.diff(scale_t, time) * c_time * sp.diff(phi_t, time) / scale_t
        + sp.diff(v_time(phi_t), phi_t)
        - alpha_phi * trace_e
    )
    chi_euler_minispace = sp.simplify(
        (sp.diff(sp.diff(mini_time, sp.diff(chi_t, time)), time) - sp.diff(mini_time, chi_t))
        / scale_t**3
        - alpha_chi * trace_e
    )
    chi_euler_covariant = sp.simplify(
        y0 * (sp.diff(chi_t, time, 2) + 3 * sp.diff(scale_t, time) * sp.diff(chi_t, time) / scale_t)
        + mass**2 * z_time(x_time) * chi_t
        - alpha_chi * trace_e
    )

    x_dot = phi_dot * phi_ddot
    w_dot = chi_dot * chi_ddot
    q_dot = mass**2 * chi * chi_dot
    c_dot = p_xx * x_dot - q_dot * z_x - q_mass * z_xx * x_dot
    rho_dot = (
        2 * x_dot * c_value
        + 2 * x_symbol * c_dot
        - p_x * x_dot
        + potential_phi * phi_dot
        + y0 * w_dot
        + q_dot * z_value
        + q_mass * z_x * x_dot
    )
    scalar_continuity = sp.expand(rho_dot + 3 * hubble * enthalpy)
    scalar_euler_power = sp.expand(
        phi_dot
        * (c_value * phi_ddot + c_dot * phi_dot + 3 * hubble * c_value * phi_dot + potential_phi)
        + chi_dot * (y0 * chi_ddot + 3 * hubble * y0 * chi_dot + mass**2 * z_value * chi)
    )

    rho_e, p_e, rho_e_dot, ln_a_dot = sp.symbols("rho_E p_E rho_E_dot lambda_dot", real=True)
    physical_conservation_in_e_variables = (
        rho_e_dot - 4 * ln_a_dot * rho_e + 3 * (hubble + ln_a_dot) * (rho_e + p_e)
    )
    einstein_continuity = rho_e_dot + 3 * hubble * (rho_e + p_e) + (-rho_e + 3 * p_e) * ln_a_dot
    scalar_exchange = trace_e * ln_a_dot
    matter_exchange = -trace_e * ln_a_dot

    total_enthalpy = sp.symbols("rho_plus_p", real=True)
    friedmann_derivative_after_continuity = (
        6 * planck**2 * hubble * hubble_dot + 3 * hubble * total_enthalpy
    )
    spatial_einstein = planck**2 * (2 * hubble_dot + 3 * hubble**2) + sp.symbols(
        "p_total", real=True
    )
    rho_total, p_total = sp.symbols("rho_total p_total", real=True)

    eos = sp.symbols("w", real=True)
    general_log_derivative = (1 - 3 * eos) * ln_a_dot - 3 * (1 + eos) * hubble
    required_log_derivative = -3 * hubble * (1 + eos) + (1 - 3 * eos) * ln_a_dot

    beta, u = sp.symbols("beta u", positive=True)
    x_positive = sp.symbols("X_pos", positive=True)
    z_gate = (1 + beta * x_positive**2) ** 2
    zx_gate = sp.diff(z_gate, x_positive)
    zxx_gate = sp.diff(zx_gate, x_positive)
    gate_energy = sp.factor(z_gate - 2 * x_positive * zx_gate)
    gate_enthalpy = sp.factor(-2 * x_positive * zx_gate)
    gate_energy_u = (1 + u) * (1 - 7 * u)
    gate_enthalpy_u = -8 * u * (1 + u)
    gate_pressure_u = -((1 + u) ** 2)
    k_value = p_x + 2 * x_positive * p_xx - q_mass * (zx_gate + 2 * x_positive * zxx_gate)
    k_expected = (
        p_x
        + 2 * x_positive * p_xx
        - q_mass * 4 * beta * x_positive * (3 + 7 * beta * x_positive**2)
    )

    conformal, null_norm = sp.symbols("A null_norm", positive=True)
    delta_g = sp.Rational(1, 1_000_000_000_000_000)
    q_lower = 1 - 1 / (1 - delta_g) ** 2
    q_upper = 1 - 1 / (1 + delta_g) ** 2

    checks = [
        _check(
            "S01_LAPSE_ENERGY_ROUTE",
            rho_lapse - rho_lapse_expected,
            "Lapse variation of the homogeneous scalar action gives the frozen energy density.",
        ),
        _check(
            "S02_COVARIANT_ENERGY_ROUTE",
            rho_covariant - rho_expected,
            "The covariant stress tensor independently gives the same scalar energy density.",
        ),
        _check(
            "S03_PRESSURE_SCALE_ROUTE",
            p_scale - pressure_lapse_expected,
            "Scale-factor variation gives p_s=L_s for homogeneous scalars.",
        ),
        _check(
            "S04_ENTHALPY_IDENTITY",
            rho_expected + pressure_expected - enthalpy,
            "The exact scalar enthalpy is 2XC+2Y0X_chi.",
        ),
        _check(
            "S05_PHI_EULER_COVARIANT_MATCH",
            phi_euler_minispace - phi_euler_covariant,
            "Direct minisuperspace Euler variation matches the multiplied homogeneous covariant phi equation including the conformal source.",
        ),
        _check(
            "S06_CHI_EULER_COVARIANT_MATCH",
            chi_euler_minispace - chi_euler_covariant,
            "Direct minisuperspace Euler variation matches the multiplied homogeneous covariant chi equation with the predecessor sign.",
        ),
        _check(
            "S07_PHI_GATE_TIME_DERIVATIVE",
            c_dot - (p_xx * x_dot - mass**2 * chi * chi_dot * z_x - q_mass * z_xx * x_dot),
            "dot(C) retains both chi-amplitude and X-dependent mass-gate terms.",
        ),
        _check(
            "S08_SCALAR_CONTINUITY_IDENTITY",
            (scalar_continuity - scalar_euler_power).subs(
                {x_symbol: phi_dot**2 / 2, w_symbol: chi_dot**2 / 2}
            ),
            "The scalar continuity expression equals the two Euler equations contracted with field velocities.",
        ),
        _check(
            "S09_PHYSICAL_TO_EINSTEIN_CONTINUITY",
            physical_conservation_in_e_variables - einstein_continuity,
            "Physical-frame conservation converts exactly to Einstein-frame conformal exchange.",
        ),
        _check(
            "S10_TOTAL_CONTINUITY_CANCELLATION",
            scalar_exchange + matter_exchange,
            "Scalar and Einstein-frame matter exchanges cancel in total continuity.",
        ),
        _check(
            "S11_FRIEDMANN_TO_RAYCHAUDHURI",
            friedmann_derivative_after_continuity
            - 3 * hubble * (2 * planck**2 * hubble_dot + total_enthalpy),
            "Differentiated Friedmann plus total continuity factors through Raychaudhuri.",
        ),
        _check(
            "S12_SPATIAL_EINSTEIN_RAYCHAUDHURI",
            spatial_einstein.subs(sp.symbols("p_total", real=True), p_total).subs(
                hubble**2, rho_total / (3 * planck**2)
            )
            - (2 * planck**2 * hubble_dot + rho_total + p_total),
            "The independent spatial Einstein equation plus Friedmann gives Raychaudhuri without dividing by H.",
        ),
        _check(
            "S13_CONSTANT_W_MATTER_SOLUTION",
            general_log_derivative - required_log_derivative,
            "The constant-w Einstein density scales as A^(1-3w)*a^[-3(1+w)].",
        ),
        _check(
            "S14_RADIATION_SCALING",
            general_log_derivative.subs(eos, sp.Rational(1, 3)) + 4 * hubble,
            "Radiation scales as a^-4 with zero classical conformal exchange.",
        ),
        _check(
            "S15_DUST_SCALING",
            general_log_derivative.subs(eos, 0) - (ln_a_dot - 3 * hubble),
            "Dust scales as A*a^-3.",
        ),
        _check(
            "S16_VACUUM_SCALING",
            general_log_derivative.subs(eos, -1) - 4 * ln_a_dot,
            "A physical constant vacuum density maps to rho_E proportional to A^4.",
        ),
        _check(
            "S17_X_ZERO_GATE_LIMIT",
            gate_energy.subs(x_positive, 0) - 1,
            "At X=0 the mass gate contributes +Q to energy and -Q to pressure.",
        ),
        _check(
            "S18_GATE_ENERGY_FACTORIZATION",
            gate_energy.subs(beta * x_positive**2, u) - gate_energy_u,
            "The exact mass-gate energy factor is (1+u)(1-7u).",
        ),
        _check(
            "S19_GATE_ENTHALPY_FACTORIZATION",
            gate_enthalpy.subs(beta * x_positive**2, u) - gate_enthalpy_u,
            "The exact mass-gate enthalpy factor is -8u(1+u).",
        ),
        _check(
            "S20_HIGH_U_GATE_ENERGY_SCALING",
            sp.limit(gate_energy_u / u**2, u, sp.oo) + 7,
            "The fixed-chi gate energy approaches -7Q*u^2.",
        ),
        _check(
            "S21_HIGH_U_GATE_PRESSURE_SCALING",
            sp.limit(gate_pressure_u / u**2, u, sp.oo) + 1,
            "The fixed-chi gate pressure approaches -Q*u^2.",
        ),
        _check(
            "S22_HIGH_U_GATE_ENTHALPY_SCALING",
            sp.limit(gate_enthalpy_u / u**2, u, sp.oo) + 8,
            "The fixed-chi gate enthalpy approaches -8Q*u^2.",
        ),
        _check(
            "S23_TIMELIKE_K_COEFFICIENT",
            k_value - k_expected,
            "The homogeneous timelike kinetic coefficient retains Z_X+2XZ_XX exactly.",
        ),
        _check(
            "S24_CONFORMAL_TENSOR_PHOTON_CONE",
            (conformal**-2 * null_norm).subs(null_norm, 0),
            "A positive conformal factor leaves every g-null covector photon-null.",
        ),
        _check(
            "S25_DISFORMAL_Q_ENDPOINTS",
            1 / (1 - q_lower) + 1 / (1 - q_upper) - 2 * (1 + delta_g**2),
            "The inherited disformal interval endpoints map to squared speed ratios (1-delta_G)^2 and (1+delta_G)^2.",
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
            "rho_s": "2*X*P_X-P+V+Y0*X_chi+Q*(Z-2*X*Z_X)",
            "p_s": "P-V+Y0*X_chi-Q*Z",
            "phi_eom": "a^-3*d(a^3*C*phi_dot)/dt+V_phi=alpha_phi*T_E",
            "chi_eom": "Y0*(chi_ddot+3*H*chi_dot)+m_chi^2*Z*chi=alpha_chi*T_E",
            "gate_energy_factor": "(1+u)*(1-7*u)",
            "gate_high_u": "rho_gate/Q/u^2->-7; p_gate/Q/u^2->-1; enthalpy_gate/Q/u^2->-8",
        },
    }


def _sign_rational(value: sp.Rational) -> str:
    if value == 0:
        return "zero"
    return "positive" if value > 0 else "negative"


def run_numeric_suite(config: Mapping[str, Any]) -> dict[str, Any]:
    gate_records = []
    for probe in config["machine_check_contract"]["gate_u_probes"]:
        u = sp.Rational(probe["u"])
        factor = sp.factor((1 + u) * (1 - 7 * u))
        sign = _sign_rational(factor)
        gate_records.append(
            {
                "u": probe["u"],
                "rho_gate_factor": str(factor),
                "sign": sign,
                "passed": sign == probe["expected_rho_gate_factor_sign"],
            }
        )

    delta = sp.Rational(1, 1_000_000_000_000_000)
    q_lower = 1 - 1 / (1 - delta) ** 2
    q_upper = 1 - 1 / (1 + delta) ** 2
    q_records = []
    for probe in config["machine_check_contract"]["disformal_q_probes"]:
        q = sp.Rational(probe["q"])
        within = q_lower <= q <= q_upper
        q_records.append(
            {
                "q": probe["q"],
                "within_frozen_interval": bool(within),
                "expected_within_frozen_interval": probe["expected_within_frozen_interval"],
                "passed": bool(within) is probe["expected_within_frozen_interval"],
            }
        )
    _require(all(item["passed"] for item in gate_records), "gate probe failed")
    _require(all(item["passed"] for item in q_records), "disformal probe failed")
    return {
        "all_passed": True,
        "gate_u_probes": gate_records,
        "disformal_q_interval": {
            "lower": str(q_lower),
            "upper": str(q_upper),
        },
        "disformal_q_probes": q_records,
        "designed_failures_preserved": sum(
            not item["within_frozen_interval"] for item in q_records
        ),
    }


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(config, TOP_KEYS, "config")
    _require(config["schema_version"] == CONFIG_SCHEMA, "config schema changed")
    _require(
        config["analysis_id"] == "gravity-matter-lensing-flrw-background-necessary-conditions-v1",
        "analysis identity changed",
    )
    _require(_sha(config) == EXPECTED_CONFIG_CONTENT_SHA256, "config content changed")
    _require(config["output_path"] == OUTPUT_PATH.as_posix(), "output path changed")

    bindings = config["predecessor_bindings"]
    _require(isinstance(bindings, list) and len(bindings) == 4, "predecessor inventory changed")
    for index, binding in enumerate(bindings):
        _strict(binding, BINDING_KEYS, f"predecessor[{index}]")
    _require(
        [(item["binding_id"], item["git_commit"]) for item in bindings]
        == [
            ("split_gate_action", "03a652acaded1be4cca9af48782b8d54138e54c3"),
            ("split_gate_source_bound", "7216ff7319e0f38c7639926b31e7f4e881f9a64a"),
            ("universal_conformal_source", "98589e269c362846154764e6b3e400fa300c2a94"),
            ("solar_gw_necessary_conditions", "ad11eba2ebfc7c12f107cfbea8b969dfd05de101"),
        ],
        "predecessor commits changed",
    )

    for key, keys, label in (
        ("conventions", CONVENTION_KEYS, "conventions"),
        ("stress_energy_contract", STRESS_KEYS, "stress contract"),
        ("background_equations", BACKGROUND_KEYS, "background equations"),
        ("matter_exchange_contract", EXCHANGE_KEYS, "matter exchange"),
        ("gate_limits_and_necessary_conditions", LIMIT_KEYS, "necessary conditions"),
        ("tensor_photon_and_disformal_contract", CONE_KEYS, "cone contract"),
        ("machine_check_contract", MACHINE_KEYS, "machine contract"),
        ("adjudication", ADJUDICATION_KEYS, "adjudication"),
        ("claim_boundary", CLAIM_KEYS, "claim boundary"),
        ("zero_access_and_compute", ZERO_KEYS, "zero access"),
    ):
        _strict(config[key], keys, label)

    conventions = config["conventions"]
    _require(conventions["signature"] == "(-,+,+,+)", "signature changed")
    _require(
        "N=1 only after lapse variation" in conventions["cosmic_time_gauge"],
        "lapse convention changed",
    )
    _require("Q_i=-(partial_i ln A)*T_E" in conventions["source_sign"], "source sign changed")
    stress = config["stress_energy_contract"]
    _require("Q*(Z-2*X*Z_X)" in stress["scalar_energy_density"], "energy density changed")
    _require(stress["scalar_pressure"] == "p_s=L_s=P-V+Y0*X_chi-Q*Z", "pressure changed")
    background = config["background_equations"]
    _require("3*M_Pl^2*H^2=rho_E+rho_s" == background["friedmann"], "Friedmann changed")
    _require(
        "alpha_phi*T_E" in background["phi_eom"] and "alpha_chi*T_E" in background["chi_eom"],
        "scalar source changed",
    )
    _require("m_chi^2*chi*dot(chi)*Z_X" in background["phi_gate_content"], "gate mixing omitted")
    exchange = config["matter_exchange_contract"]
    _require("=-T_E*dot(ln A)" in exchange["einstein_frame"], "matter exchange changed")
    _require("=T_E*dot(ln A)" in exchange["scalar_continuity"], "scalar exchange changed")
    limits = config["gate_limits_and_necessary_conditions"]
    _require("(1+u)*(1-7*u)" in limits["exact_gate_energy_factor"], "gate energy factor changed")
    _require("rho_gate~-7*Q*u^2" in limits["high_u"], "high-u obstruction changed")
    cones = config["tensor_photon_and_disformal_contract"]
    _require("c_GW/c_gamma=1" in cones["conformal_branch"], "conformal cone changed")
    _require(cones["status"].startswith("BLOCKED_"), "disformal gate unblocked")

    machine = config["machine_check_contract"]
    _require(
        tuple(machine["required_symbolic_checks"]) == SYMBOLIC_CHECK_IDS,
        "symbolic inventory changed",
    )
    _require(
        len(machine["gate_u_probes"]) == 4 and len(machine["disformal_q_probes"]) == 4,
        "probe inventory changed",
    )
    for probe in machine["gate_u_probes"]:
        _strict(probe, {"u", "expected_rho_gate_factor_sign"}, "gate probe")
    for probe in machine["disformal_q_probes"]:
        _strict(probe, {"q", "expected_within_frozen_interval"}, "disformal probe")

    adjudication = config["adjudication"]
    _require(adjudication["overall_decision"] == DECISION, "decision changed")
    for key in (
        "exact_homogeneous_stress_energy_derived",
        "friedmann_raychaudhuri_derived",
        "scalar_eom_with_gate_and_sources_derived",
        "physical_einstein_exchange_derived",
        "continuity_identity_verified",
        "matter_limit_probes_derived",
        "gate_limit_obstruction_derived",
        "conformal_cone_alignment_derived",
        "disformal_necessary_interval_inherited_and_checked",
    ):
        _require(adjudication[key] is True, f"derived result disabled: {key}")
    for key in (
        "healthy_late_time_history_exists",
        "accelerating_solution_exists",
        "BBN_passed",
        "CMB_passed",
        "growth_passed",
        "cutoff_established",
        "perturbation_stability_established",
        "full_constrained_characteristics_established",
        "parameter_values_frozen",
        "initial_conditions_frozen",
        "observational_fit_performed",
    ):
        _require(adjudication[key] is False, f"blocked result overclaimed: {key}")
    claims = config["claim_boundary"]
    _require(
        claims["restricted_flat_flrw_equations_established"] is True, "restricted claim disabled"
    )
    _require(
        all(
            value is False
            for key, value in claims.items()
            if key != "restricted_flat_flrw_equations_established"
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
            receipt.get("decision") == binding["receipt_decision"], "predecessor decision changed"
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
        "status": "exact_flat_flrw_equations_machine_derived_cosmological_history_blocked",
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
        "conventions": config["conventions"],
        "stress_energy_contract": config["stress_energy_contract"],
        "background_equations": config["background_equations"],
        "matter_exchange_contract": config["matter_exchange_contract"],
        "gate_limits_and_necessary_conditions": config["gate_limits_and_necessary_conditions"],
        "tensor_photon_and_disformal_contract": config["tensor_photon_and_disformal_contract"],
        "symbolic_suite": symbolic,
        "numeric_suite": numeric,
        "adjudication": config["adjudication"],
        "claim_boundary": config["claim_boundary"],
        "counts": {
            "symbolic_checks": len(symbolic["checks"]),
            "symbolic_checks_passed": sum(item["passed"] for item in symbolic["checks"]),
            "gate_u_probes": len(numeric["gate_u_probes"]),
            "gate_u_probes_passed": sum(item["passed"] for item in numeric["gate_u_probes"]),
            "disformal_q_probes": len(numeric["disformal_q_probes"]),
            "disformal_q_probes_passed": sum(
                item["passed"] for item in numeric["disformal_q_probes"]
            ),
            "designed_failures_preserved": numeric["designed_failures_preserved"],
            "observational_files_opened": 0,
            "observational_rows_opened": 0,
            "network_calls": 0,
            "model_or_paid_calls": 0,
            "gpu_calls": 0,
        },
        "zero_access_and_compute": config["zero_access_and_compute"],
        "remaining_blockers": [
            "Freeze P(X), V(phi), A(phi,chi), all parameters, the EFT cutoff, and initial conditions without post-result choices.",
            "Prove a regular solution exists over a declared interval and preserves Friedmann plus the independent spatial constraint.",
            "Show whether the negative high-u gate-energy contribution is dynamically avoided or consistently compensated without violating C>0 and K>0.",
            "Derive the complete scalar-metric-matter quadratic action and establish ghost, gradient, hyperbolicity, and strong-coupling control.",
            "Demonstrate acceleration rather than merely satisfying its necessary rho+3p inequality.",
            "Preregister and pass BBN, CMB, expansion-history, growth, Solar, GW, and lensing tests using one parameter set.",
            "Any disformal extension requires a new action-level derivation plus the inherited q interval; it is not validated here.",
        ],
        "limitations": [
            "These are exact homogeneous flat-FLRW equations for the frozen action, not proof that a physical cosmological history exists.",
            "The high-u negative gate-energy term is an obstruction requiring compensation, not an unconditional no-go theorem.",
            "Background C>0 and K>0 are necessary local scalar conditions, not the full constrained perturbation analysis.",
            "Conformal cone alignment is structural; no observational GW or photon propagation pass is claimed.",
            "No observational rows, network, model calls, likelihood, fit, novelty, or publication claim is present.",
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
    "conventions",
    "stress_energy_contract",
    "background_equations",
    "matter_exchange_contract",
    "gate_limits_and_necessary_conditions",
    "tensor_photon_and_disformal_contract",
    "symbolic_suite",
    "numeric_suite",
    "adjudication",
    "claim_boundary",
    "counts",
    "zero_access_and_compute",
    "remaining_blockers",
    "limitations",
    "content_sha256",
}
COUNT_KEYS = {
    "symbolic_checks",
    "symbolic_checks_passed",
    "gate_u_probes",
    "gate_u_probes_passed",
    "disformal_q_probes",
    "disformal_q_probes_passed",
    "designed_failures_preserved",
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
        "conventions",
        "stress_energy_contract",
        "background_equations",
        "matter_exchange_contract",
        "gate_limits_and_necessary_conditions",
        "tensor_photon_and_disformal_contract",
    ):
        _require(receipt[key] == config[key], f"receipt contract changed: {key}")
    _require(
        receipt["adjudication"] == config["adjudication"]
        and receipt["claim_boundary"] == config["claim_boundary"],
        "claims changed",
    )
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
        {
            "all_passed",
            "gate_u_probes",
            "disformal_q_interval",
            "disformal_q_probes",
            "designed_failures_preserved",
        },
        "numeric suite",
    )
    _require(numeric["all_passed"] is True, "numeric suite failed")
    _strict(numeric["disformal_q_interval"], {"lower", "upper"}, "disformal interval")
    for item in numeric["gate_u_probes"]:
        _strict(item, {"u", "rho_gate_factor", "sign", "passed"}, "gate result")
        _require(item["passed"] is True, "failed gate result retained")
    for item in numeric["disformal_q_probes"]:
        _strict(
            item,
            {
                "q",
                "within_frozen_interval",
                "expected_within_frozen_interval",
                "passed",
            },
            "disformal result",
        )
        _require(item["passed"] is True, "failed disformal result retained")
    _strict(receipt["counts"], COUNT_KEYS, "counts")
    counts = receipt["counts"]
    _require(
        counts["symbolic_checks"] == counts["symbolic_checks_passed"] == 25,
        "symbolic count changed",
    )
    _require(
        counts["gate_u_probes"] == counts["gate_u_probes_passed"] == 4, "gate probe count changed"
    )
    _require(
        counts["disformal_q_probes"] == counts["disformal_q_probes_passed"] == 4,
        "disformal probe count changed",
    )
    _require(counts["designed_failures_preserved"] == 2, "designed failure count changed")
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
    _require(
        receipt["zero_access_and_compute"] == config["zero_access_and_compute"],
        "zero-access ledger changed",
    )
    _require(
        len(receipt["remaining_blockers"]) == 7 and len(receipt["limitations"]) == 5,
        "blocker inventory changed",
    )


def _atomic_no_replace(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return "EXISTING_IDENTICAL"
        raise FlrwNecessaryConditionsError(f"refusing to overwrite different receipt: {path}")
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
            raise FlrwNecessaryConditionsError(
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
        "healthy_history_exists": receipt["adjudication"]["healthy_late_time_history_exists"],
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
    except FlrwNecessaryConditionsError as exc:
        print(json.dumps({"valid": False, "error": str(exc)}, sort_keys=True))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
