from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .legendre_ir import build_local_kinetic_model

SCHEMA_VERSION = "sigma-q-variation-ir-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _algebraic_first_variation_controls() -> dict[str, Any]:
    eta = sp.Symbol("eta", real=True)
    metric = sp.diag(-1, 1, 1)
    u = sp.Matrix([sp.Rational(5, 4), sp.Rational(3, 4), 0])
    delta_u = sp.Matrix([sp.Rational(2, 7), sp.Rational(-1, 3), sp.Rational(4, 5)])
    derivative_u = sp.Matrix(
        [
            [sp.Rational(1, 2), -1, sp.Rational(2, 3)],
            [sp.Rational(3, 5), sp.Rational(1, 7), -sp.Rational(2, 9)],
            [sp.Rational(4, 11), -sp.Rational(5, 13), sp.Rational(3, 8)],
        ]
    )
    delta_derivative_u = sp.Matrix(
        [
            [sp.Rational(2, 5), sp.Rational(1, 3), -sp.Rational(1, 4)],
            [-sp.Rational(3, 7), sp.Rational(2, 9), sp.Rational(5, 6)],
            [sp.Rational(1, 8), -sp.Rational(4, 9), sp.Rational(2, 7)],
        ]
    )
    b = sp.Matrix(
        [
            [sp.Rational(2, 3), -sp.Rational(1, 5), sp.Rational(4, 7)],
            [sp.Rational(3, 8), sp.Rational(5, 11), -sp.Rational(2, 9)],
            [-sp.Rational(1, 6), sp.Rational(7, 10), sp.Rational(3, 13)],
        ]
    )
    delta_b = sp.Matrix(
        [
            [sp.Rational(1, 4), sp.Rational(2, 7), -sp.Rational(3, 5)],
            [-sp.Rational(2, 11), sp.Rational(4, 9), sp.Rational(1, 3)],
            [sp.Rational(5, 12), -sp.Rational(1, 8), sp.Rational(2, 5)],
        ]
    )

    def projector(vector: sp.Matrix) -> sp.Matrix:
        return metric + vector * vector.T

    def q_contraction(vector: sp.Matrix, tensor: sp.Matrix) -> sp.Expr:
        p = projector(vector)
        return sp.expand(
            sum(
                p[mu, rho] * p[nu, sigma] * tensor[mu, nu] * tensor[rho, sigma]
                for mu in range(3)
                for nu in range(3)
                for rho in range(3)
                for sigma in range(3)
            )
        )

    direct_q = sp.diff(q_contraction(u + eta * delta_u, b + eta * delta_b), eta).subs(
        eta, 0
    )
    p = projector(u)
    c = p * b * p
    b_variation = 2 * sum(c[mu, nu] * delta_b[mu, nu] for mu in range(3) for nu in range(3))
    projector_variation = 2 * sum(
        delta_u[lam]
        * (
            sum(
                u[rho] * p[nu, sigma] * b[lam, nu] * b[rho, sigma]
                for rho in range(3)
                for nu in range(3)
                for sigma in range(3)
            )
            + sum(
                u[sigma] * p[mu, rho] * b[mu, lam] * b[rho, sigma]
                for sigma in range(3)
                for mu in range(3)
                for rho in range(3)
            )
        )
        for lam in range(3)
    )
    q_residual = sp.factor(direct_q - b_variation - projector_variation)

    acceleration = derivative_u.T * u
    delta_acceleration = derivative_u.T * delta_u + delta_derivative_u.T * u
    acceleration_eta = (derivative_u + eta * delta_derivative_u).T * (
        u + eta * delta_u
    )
    direct_x = sp.diff((acceleration_eta.T * metric * acceleration_eta)[0], eta).subs(
        eta, 0
    )
    x_residual = sp.factor(
        direct_x - 2 * (acceleration.T * metric * delta_acceleration)[0]
    )
    return {
        "projector_and_B_first_variation_residual": str(q_residual),
        "acceleration_norm_first_variation_residual": str(x_residual),
        "passed": q_residual == 0 and x_residual == 0,
        "arithmetic": "exact rational tensor contractions in signature (-,+,+)",
    }


def projected_aether_q_first_variation_control() -> tuple[bool, dict[str, Any]]:
    evidence = _algebraic_first_variation_controls()
    evidence["claim_scope"] = (
        "raw fixed-metric first variation of the projected acceleration-gradient contraction "
        "and acceleration norm before covariant integration by parts"
    )
    return bool(evidence["passed"]), evidence


def compile_q_variation_ir(action_ir: dict[str, Any]) -> dict[str, Any]:
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
    q_terms = sorted(term_id for term_id in coefficients if term_id.startswith("AETHER_Q"))
    has_x = "AETHER_X_SQRT1P" in coefficients
    controls = _algebraic_first_variation_controls()
    if not q_terms and not has_x:
        status = "pass"
        applicable = False
    else:
        status = "unresolved"
        applicable = True
    q_derivative = " + ".join(
        f"{power}*C_Q{power}*Q_a_u^{power - 1}"
        for power in (1, 2, 3)
        if f"AETHER_Q{power}" in q_terms
    ) or "0"
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "applicable": applicable,
        "q_term_ids": q_terms,
        "nonlinear_x_present": has_x,
        "definitions": {
            "P_mu_nu": "g_mu_nu+u_mu u_nu",
            "a_mu": "u^alpha nabla_alpha u_mu",
            "B_mu_nu": "nabla_mu a_nu",
            "Q_a_u": (
                "(L_sigma^2/a_sigma^2) P^{mu rho}P^{nu sigma}B_mu_nu B_rho_sigma"
            ),
            "X_a_u": "a_mu a^mu/a_sigma^2",
        },
        "fixed_metric_vector_variation": {
            "status": "pass" if controls["passed"] else "reject",
            "q_action_derivative": q_derivative,
            "q_auxiliary": (
                "C^{mu nu}=F_Q*(L_sigma^2/a_sigma^2) "
                "P^{mu rho}P^{nu sigma}B_rho_sigma; Y^nu=nabla_mu C^{mu nu}"
            ),
            "q_projector_contribution_E_lambda": (
                "2*F_Q*(L_sigma^2/a_sigma^2)*[u^rho P^{nu sigma} "
                "B_lambda_nu B_rho_sigma + u^sigma P^{mu rho} "
                "B_mu_lambda B_rho_sigma]"
            ),
            "q_B_contribution_E_lambda": (
                "-2 Y^nu nabla_lambda u_nu + "
                "2 g_nu_lambda nabla_alpha(Y^nu u^alpha)"
            ),
            "x_auxiliary": "Z^nu=2*C_X*F_X*a^nu/a_sigma^2",
            "x_contribution_E_lambda": (
                "Z^nu nabla_lambda u_nu - "
                "g_nu_lambda nabla_alpha(Z^nu u^alpha)"
            ),
            "unit_constraint_contribution": "2 lambda_u u_lambda",
            "boundary_contract": "compact-support variations or fixed field/normal data",
            "algebraic_controls": controls,
        },
        "metric_variation": {
            "status": "unresolved" if applicable else "not_applicable",
            "missing_terms": [
                "variation of both projectors and raised indices",
                "variation of the Levi-Civita connection inside a_mu and B_mu_nu",
                "boundary completion for second derivatives",
            ] if applicable else [],
        },
        "diffeomorphism_noether_identity": {
            "status": "unresolved" if applicable else "not_applicable",
            "required_identity": (
                "2 nabla_mu E_g^{mu}{}_nu - E_u^mu nabla_nu u_mu "
                "- nabla_mu(E_u^mu u_nu) + E_lambda nabla_nu lambda_u = 0"
            ) if applicable else None,
        },
        "observational_data_opened": False,
        "proof_scope": (
            "exact fixed-metric vector first variation and rational tensor negative residuals; "
            "the connection-dependent metric variation, boundary completion, and full Noether "
            "identity remain mandatory before this action can pass covariant variation"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def write_q_variation_ir(payload: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
