from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .legendre_ir import build_local_kinetic_model
from .q_tilt import projected_aether_q_constant_tilt_root_audit
from .stability_ir import _declared_domain, _prove_relation

SCHEMA_VERSION = "sigma-q-operator-ir-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compile_q_operator_ir(action_ir: dict[str, Any]) -> dict[str, Any]:
    """Audit the aligned quadratic and homogeneous Legendre rank of Q_a_u actions.

    This is a necessary negative/regularity gate, not a generic-tilt health proof.  On an aligned
    Minkowski background, a_i is first order, Q_a_u is second order in perturbation amplitude,
    and Q_a_u^n for n>1 cannot repair the quadratic kinetic symbol.
    """

    if not action_ir.get("valid"):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "promotion_allowed": False,
            "input_action_sha256": action_ir.get("content_sha256"),
            "errors": ["covariant action IR is invalid"],
        }
    model = build_local_kinetic_model(action_ir)
    coefficients = model["coefficients"]
    term_ids = set(coefficients)
    q_powers = {
        1: "AETHER_Q1",
        2: "AETHER_Q2",
        3: "AETHER_Q3",
    }
    present = {power: term_id for power, term_id in q_powers.items() if term_id in term_ids}
    if not present:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "promotion_allowed": False,
            "input_action_sha256": action_ir["content_sha256"],
            "applicable": False,
            "reason": "action contains no Q_a_u term",
            "observational_data_opened": False,
        }
        content = _canonical_json(body)
        return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}

    eta, wave_number, velocity_square = sp.symbols(
        "eta k vdot_squared", nonnegative=True
    )
    coefficient_symbols = {
        str(symbol): symbol
        for expression in coefficients.values()
        for symbol in expression.free_symbols
    }
    length_scale = coefficient_symbols.get(
        "L_sigma", sp.Symbol("L_sigma", positive=True)
    )
    acceleration_scale = coefficient_symbols.get(
        "a_sigma", sp.Symbol("a_sigma", positive=True)
    )
    q_quadratic = eta**2 * length_scale**2 * wave_number**2 * velocity_square / acceleration_scale**2
    perturbative_orders = {
        term_id: 2 * power for power, term_id in sorted(present.items())
    }
    quadratic_coefficients = {
        term_id: str(
            sp.factor(
                sp.diff(coefficients[term_id] * q_quadratic**power, eta, 2).subs(
                    eta, 0
                )
                / 2
            )
        )
        for power, term_id in sorted(present.items())
    }
    linear_q_coefficient = coefficients.get("AETHER_Q1", sp.Integer(0))
    local_acceleration_coefficient = sp.factor(
        -coefficients.get("AETHER_K1", 0) + coefficients.get("AETHER_K4", 0)
        + coefficients.get("AETHER_X_SQRT1P", 0) / (2 * acceleration_scale**2)
    )
    per_component_hessian = sp.factor(
        2
        * (
            local_acceleration_coefficient
            + linear_q_coefficient
            * length_scale**2
            * wave_number**2
            / acceleration_scale**2
        )
    )
    homogeneous_hessian = sp.factor(per_component_hessian.subs(wave_number, 0))
    inhomogeneous_q_hessian = sp.factor(
        per_component_hessian.subs(local_acceleration_coefficient, 0)
    )
    has_local_regularizer = local_acceleration_coefficient != 0
    has_linear_q = linear_q_coefficient != 0
    local_sign_certificate = (
        _prove_relation(
            sp.Gt(local_acceleration_coefficient, 0), _declared_domain(action_ir)
        )
        if has_local_regularizer
        else {
            "status": "reject",
            "condition": "0 > 0",
            "proof": "no local acceleration kinetic coefficient",
        }
    )
    transverse_gradient_energy = sp.factor(-coefficients.get("AETHER_K1", 0))
    longitudinal_gradient_energy = sp.factor(
        -coefficients.get("AETHER_K1", 0)
        - coefficients.get("AETHER_K2", 0)
        - coefficients.get("AETHER_K3", 0)
    )
    domain = _declared_domain(action_ir)
    gradient_sign_certificates = {
        "transverse": _prove_relation(sp.Gt(transverse_gradient_energy, 0), domain),
        "longitudinal": _prove_relation(sp.Gt(longitudinal_gradient_energy, 0), domain),
    }
    q_kinetic_operator_coefficient = sp.factor(
        2 * linear_q_coefficient * length_scale**2 / acceleration_scale**2
    )
    q_kinetic_sign_certificate = _prove_relation(
        sp.Gt(q_kinetic_operator_coefficient, 0), domain
    )
    low_frequency_speed_squared = sp.factor(
        2 * transverse_gradient_energy / homogeneous_hessian
    ) if homogeneous_hessian != 0 else sp.nan
    epsilon_symbol = coefficient_symbols.get("epsilon")
    gamma_symbol = coefficient_symbols.get("gamma")
    cone_relation = None
    if homogeneous_hessian != 0:
        cone_relation = (
            sp.Le(gamma_symbol, epsilon_symbol)
            if epsilon_symbol is not None
            and gamma_symbol is not None
            and sp.factor(low_frequency_speed_squared - gamma_symbol / epsilon_symbol) == 0
            else sp.Le(low_frequency_speed_squared, 1)
        )
    matter_cone_certificate = (
        _prove_relation(cone_relation, domain)
        if homogeneous_hessian != 0
        else {
            "status": "reject",
            "condition": "undefined <= 1",
            "proof": "homogeneous kinetic Hessian vanishes",
        }
    )
    tilt_audit_passed, tilt_root_audit = projected_aether_q_constant_tilt_root_audit()
    if not has_linear_q:
        status = "reject"
        conclusion = (
            "Q_a_u powers n>1 begin at perturbative order four or higher and provide no "
            "linearized Aether equation on the zero-Q background"
        )
    elif not has_local_regularizer:
        status = "reject"
        conclusion = (
            "the velocity Hessian has rank three for k!=0 but rank zero for k=0; a spatially "
            "homogeneous u_i(t) has arbitrary time dependence, so the Cauchy evolution is "
            "underdetermined on the aligned zero-Q background"
        )
    elif local_sign_certificate["status"] == "reject":
        status = "reject"
        conclusion = (
            "the declared parameter domain makes the homogeneous local Aether acceleration "
            "kinetic coefficient nonpositive, producing a kinetic ghost or zero-rank mode"
        )
    elif q_kinetic_sign_certificate["status"] == "reject":
        status = "reject"
        conclusion = (
            "the declared parameter domain gives the spatially dispersive Q_a_u kinetic "
            "operator a nonpositive coefficient, producing a high-wave-number kinetic ghost"
        )
    elif any(
        item["status"] == "reject" for item in gradient_sign_certificates.values()
    ):
        status = "reject"
        conclusion = (
            "the aligned quadratic Aether sector lacks strictly positive transverse and "
            "longitudinal spatial-gradient energy; arbitrary static spatial vector patterns "
            "therefore have zero or negative quadratic cost"
        )
    elif tilt_audit_passed and tilt_root_audit["generic_tilt_hyperbolicity_status"] == "reject":
        status = "reject"
        conclusion = (
            "the complete constant-background lab-frequency polynomial is quartic for every "
            "nonzero tilt but has exactly two real roots; its remaining conjugate pair is "
            "nonreal, so the Q operator has no open hyperbolicity cone around the aligned time "
            "covector in the present generic unit-vector theory"
        )
    else:
        status = "unresolved"
        conclusion = (
            "a local acceleration term removes this specific k=0 rank loss, but generic-tilt "
            "ADM/Dirac closure and the full higher-jet principal symbol remain unproved"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "applicable": True,
        "background": "Minkowski, u^mu=(1,0,0,0), zero-Q, Fourier spatial wave number k",
        "q_term_ids": [present[power] for power in sorted(present)],
        "perturbative_order_in_vector_amplitude": perturbative_orders,
        "quadratic_lagrangian_coefficients": quadratic_coefficients,
        "linear_q_coefficient": str(linear_q_coefficient),
        "local_aether_acceleration_coefficient": str(local_acceleration_coefficient),
        "local_acceleration_sign_certificate": local_sign_certificate,
        "gradient_energy_coefficients": {
            "transverse": str(transverse_gradient_energy),
            "longitudinal": str(longitudinal_gradient_energy),
        },
        "gradient_sign_certificates": gradient_sign_certificates,
        "q_kinetic_operator_coefficient": str(q_kinetic_operator_coefficient),
        "q_kinetic_sign_certificate": q_kinetic_sign_certificate,
        "nonlinear_x_quadratic_identity": (
            "d^2/deta^2 [sqrt(1+eta^2 vdot^2/a_sigma^2)-1] at eta=0 "
            "equals vdot^2/a_sigma^2"
        ),
        "nonlinear_x_convexity": {
            "status": (
                "pass"
                if "AETHER_X_SQRT1P" in term_ids
                and local_sign_certificate["status"] == "pass"
                else "not_applicable_or_unresolved"
            ),
            "velocity_hessian_transverse_eigenvalue": (
                "C_X/(a_sigma^2*sqrt(1+v^2/a_sigma^2))"
            ),
            "velocity_hessian_longitudinal_eigenvalue": (
                "C_X/(a_sigma^2*(1+v^2/a_sigma^2)^(3/2))"
            ),
            "domain": "v^2>=0, a_sigma>0, C_X>0",
        },
        "constant_background_covariant_principal": {
            "status": (
                "reject"
                if tilt_root_audit["generic_tilt_hyperbolicity_status"] == "reject"
                else "unresolved"
            ),
            "rest_frame_polynomial": (
                "Omega^2*(K0+K2*kappa^2)-G*kappa^2=0, "
                "Omega=-u^mu xi_mu, kappa^2=P^mu_nu xi_mu xi_nu"
            ),
            "kinetic_symbol": str(per_component_hessian),
            "low_frequency_speed_squared": str(low_frequency_speed_squared),
            "dispersion_relation": (
                "Omega^2=(G*kappa^2)/(K0+K2*kappa^2)"
            ),
            "group_speed_bound": (
                "max_kappa |dOmega/dkappa|=sqrt(low_frequency_speed_squared)"
            ),
            "matter_cone_certificate": matter_cone_certificate,
            "constant_tilt_argument": (
                "real rest-frame branches map monotonically, but that counts only two roots "
                "of a quartic lab-frequency polynomial and misses its nonreal conjugate pair"
            ),
            "complete_tilt_root_audit": tilt_root_audit,
            "scope": (
                "constant unit-Aether background and frozen metric; background derivatives, "
                "metric-vector constraint mixing, and nonlinear generic-tilt closure excluded"
            ),
        },
        "per_component_velocity_hessian": str(per_component_hessian),
        "homogeneous_velocity_hessian_k0": str(homogeneous_hessian),
        "pure_q_inhomogeneous_velocity_hessian": str(inhomogeneous_q_hessian),
        "rank_certificate": {
            "spatial_vector_components": 3,
            "rank_at_k_nonzero_without_local_regularizer": 3 if has_linear_q else 0,
            "rank_at_k_zero_without_local_regularizer": 0,
            "constant_rank": has_local_regularizer,
        },
        "euler_lagrange_principal_factor_without_regularizer": (
            "2*C_Q1*(L_sigma^2/a_sigma^2)*(-Delta)*partial_t^2 u_i"
        ),
        "conclusion": conclusion,
        "observational_data_opened": False,
        "proof_scope": (
            "exact perturbative-order and Fourier Legendre-rank certificate on the aligned "
            "zero-Q background; rejection is decisive for an action with no local Aether "
            "acceleration kinetic term, while a regularized action still needs the full backend"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def write_q_operator_ir(payload: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
