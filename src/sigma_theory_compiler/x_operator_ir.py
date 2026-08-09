from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .legendre_ir import build_local_kinetic_model
from .stability_ir import _declared_domain, _prove_relation

SCHEMA_VERSION = "sigma-x-operator-ir-1.0"
_EXPONENTS = {
    "AETHER_X_SQRT1P": sp.Rational(1, 2),
    "AETHER_X_P2_3": sp.Rational(2, 3),
    "AETHER_X_P3_4": sp.Rational(3, 4),
}
_MATCHED_COMPLETIONS = {
    "AETHER_MATCHED_K14_P1_2": sp.Rational(1, 2),
    "AETHER_MATCHED_K14_P2_3": sp.Rational(2, 3),
    "AETHER_MATCHED_K14_P3_4": sp.Rational(3, 4),
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compile_x_operator_ir(action_ir: dict[str, Any]) -> dict[str, Any]:
    """Audit finite-background convexity, energy, and cone limits of nonlinear X_a terms."""

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
    present = sorted(set(coefficients) & set(_EXPONENTS))
    if not present:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "promotion_allowed": False,
            "input_action_sha256": action_ir["content_sha256"],
            "applicable": False,
            "reason": "action contains no registered nonlinear X_a term",
            "observational_data_opened": False,
        }
        content = _canonical_json(body)
        return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}
    if len(present) != 1:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "unresolved",
            "promotion_allowed": False,
            "input_action_sha256": action_ir["content_sha256"],
            "applicable": True,
            "term_ids": present,
            "reason": "mixed nonlinear X_a atoms require a combined Hessian adapter",
            "observational_data_opened": False,
        }
        content = _canonical_json(body)
        return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}

    term_id = present[0]
    exponent = _EXPONENTS[term_id]
    matched_terms = sorted(set(coefficients) & set(_MATCHED_COMPLETIONS))
    x = sp.Symbol("X", nonnegative=True, real=True)
    coefficient = coefficients[term_id]
    coefficient_symbols = {
        str(symbol): symbol
        for expression in coefficients.values()
        for symbol in expression.free_symbols
    }
    gamma_symbol = coefficient_symbols.get("gamma")
    epsilon_symbol = coefficient_symbols.get("epsilon")
    acceleration_scale = coefficient_symbols.get(
        "a_sigma", sp.Symbol("a_sigma", positive=True, real=True)
    )
    nonlinear_scale = sp.factor(coefficient / acceleration_scale**2)
    local_linear_hessian = sp.factor(
        2 * (-coefficients.get("AETHER_K1", 0) + coefficients.get("AETHER_K4", 0))
    )
    transverse_kinetic = sp.factor(
        local_linear_hessian
        + 2 * exponent * nonlinear_scale * (1 + x) ** (exponent - 1)
    )
    longitudinal_kinetic = sp.factor(
        local_linear_hessian
        + 2
        * exponent
        * nonlinear_scale
        * (1 + x) ** (exponent - 2)
        * (1 + (2 * exponent - 1) * x)
    )
    matched_term = matched_terms[0] if len(matched_terms) == 1 else None
    matched_exponent = _MATCHED_COMPLETIONS.get(matched_term) if matched_term else None
    matched_consistent = matched_exponent == exponent
    matched_weight = sp.factor(
        (1 + x) ** (exponent - 2) * (1 + (2 * exponent - 1) * x)
    )
    matched_weight_derivative = sp.factor(sp.diff(matched_weight, x))
    if matched_term and matched_consistent:
        matched_gradient_base = sp.factor(-2 * coefficients[matched_term])
        transverse_gradient = sp.factor(matched_gradient_base * matched_weight)
        longitudinal_gradient = transverse_gradient
        completion_kind = "derivative_matched_static_null_K1_plus_K4"
    else:
        matched_gradient_base = None
        transverse_gradient = sp.factor(-2 * coefficients.get("AETHER_K1", 0))
        longitudinal_gradient = sp.factor(
            -2
            * (
                coefficients.get("AETHER_K1", 0)
                + coefficients.get("AETHER_K2", 0)
                + coefficients.get("AETHER_K3", 0)
            )
        )
        completion_kind = "constant_or_unmatched_gradient"
    transverse_speed = sp.factor(transverse_gradient / transverse_kinetic)
    longitudinal_speed = sp.factor(longitudinal_gradient / longitudinal_kinetic)
    transverse_limit = sp.limit(transverse_speed, x, sp.oo)
    longitudinal_limit = sp.limit(longitudinal_speed, x, sp.oo)

    field = (1 + x) ** exponent - 1
    radial_hamiltonian = sp.factor(
        coefficient * (2 * exponent * x * (1 + x) ** (exponent - 1) - field)
    )
    radial_hamiltonian_derivative = sp.simplify(sp.diff(radial_hamiltonian, x))
    expected_derivative = sp.factor(
        coefficient
        * exponent
        * (1 + x) ** (exponent - 2)
        * (1 + (2 * exponent - 1) * x)
    )
    hamiltonian_derivative_residual = sp.simplify(
        radial_hamiltonian_derivative - expected_derivative
    )
    domain = _declared_domain(action_ir)
    gradient_sign_target = (
        matched_gradient_base if matched_gradient_base is not None else transverse_gradient
    )
    longitudinal_sign_target = (
        matched_gradient_base if matched_gradient_base is not None else longitudinal_gradient
    )
    sign_certificates = {
        "kinetic_at_origin": _prove_relation(sp.Gt(transverse_kinetic.subs(x, 0), 0), domain),
        "transverse_gradient": _prove_relation(sp.Gt(gradient_sign_target, 0), domain),
        "longitudinal_gradient": _prove_relation(sp.Gt(longitudinal_sign_target, 0), domain),
    }
    coupled_legendre: dict[str, Any] | None = None
    generic_shear_witness: dict[str, Any] | None = None
    if matched_term and matched_consistent:
        eh_scale = sp.factor(2 * coefficients.get("EH_R", 0))
        normalized_gamma = sp.factor(-2 * coefficients[matched_term] / eh_scale)
        metric_traceless = sp.factor(eh_scale * (1 - normalized_gamma * matched_weight))
        metric_trace = sp.factor(-eh_scale * (2 + normalized_gamma * matched_weight))
        metric_determinant = sp.factor(
            -8
            * eh_scale**6
            * (1 - normalized_gamma * matched_weight) ** 5
            * (2 + normalized_gamma * matched_weight)
        )
        vector_determinant = sp.factor(transverse_kinetic**2 * longitudinal_kinetic)
        coupled_determinant = sp.factor(metric_determinant * vector_determinant)
        sign_certificates["coupled_metric_traceless_at_worst_case_X0"] = _prove_relation(
            sp.Gt(1 - normalized_gamma, 0), domain
        )
        coupled_legendre = {
            "background": "aligned Aether, frozen local orthonormal frame, K_ij=0, X>=0",
            "normalized_completion_strength": str(normalized_gamma),
            "matched_weight": str(matched_weight),
            "matched_weight_derivative": str(matched_weight_derivative),
            "matched_weight_range": "0 < W_p(X) <= 1 for X>=0 and 1/2<=p<1",
            "metric_hessian_eigenvalues": {
                "traceless_diagonal_multiplicity_2": str(metric_traceless),
                "off_diagonal_multiplicity_3": str(2 * metric_traceless),
                "conformal_trace_multiplicity_1": str(metric_trace),
            },
            "metric_hessian_determinant": str(metric_determinant),
            "vector_hessian_determinant": str(vector_determinant),
            "coupled_hessian_determinant": str(coupled_determinant),
            "cross_block": "zero at K_ij=0; nonzero generic-K mixing remains outside this certificate",
        }
        normalized_epsilon = sp.factor(nonlinear_scale / eh_scale)
        shear_invariant = sp.Symbol("R_K", nonnegative=True, real=True)
        metric_shear_factor = sp.factor(1 - normalized_gamma * matched_weight)
        longitudinal_shear_coefficient = sp.factor(
            -normalized_gamma
            * (matched_weight_derivative + 2 * x * sp.diff(matched_weight_derivative, x))
            - 4
            * normalized_gamma**2
            * x
            * matched_weight_derivative**2
            / metric_shear_factor
        )
        schur_longitudinal = sp.factor(
            2 * exponent * normalized_epsilon * matched_weight
            + shear_invariant * longitudinal_shear_coefficient
        )
        witness_substitutions = {
            gamma_symbol: sp.Rational(1, 2),
            epsilon_symbol: sp.Integer(1),
            x: sp.Integer(1),
            shear_invariant: sp.Integer(8),
        }
        witness_value = sp.simplify(schur_longitudinal.subs(witness_substitutions))
        canonical_domain = domain["canonical"]
        parameter_substitutions = {
            symbol: witness_substitutions.get(symbol, sp.Integer(1))
            for symbol in domain["symbols"].values()
        }
        domain_relations_hold = all(
            bool(relation.subs(parameter_substitutions)) for relation in domain["relations"]
        )
        positive_hold = all(
            parameter_substitutions[domain["symbols"][name]] > 0
            for name in canonical_domain.get("positive", [])
        )
        witness_is_negative = witness_value.is_negative is True
        generic_shear_witness = {
            "status": "reject" if domain_relations_hold and positive_hold and witness_is_negative else "unresolved",
            "background": (
                "aligned Aether with X=1 and pure traceless K_ij shear; "
                "R_K=K_ij K^ij/a_sigma^2=8"
            ),
            "parameter_point": {
                "gamma": "1/2",
                "epsilon": "1",
                "all_other_positive_constants": "1",
            },
            "parameter_point_satisfies_declared_domain": domain_relations_hold and positive_hold,
            "metric_shear_factor": str(metric_shear_factor),
            "longitudinal_shear_coefficient": str(longitudinal_shear_coefficient),
            "schur_reduced_longitudinal_hessian": str(schur_longitudinal),
            "exact_witness_value": str(witness_value),
            "numeric_witness_value": str(sp.N(witness_value, 16)),
            "negative_exactly": witness_is_negative,
            "consequence": (
                "the longitudinal Schur complement is positive at R_K=0 and negative at R_K=8, "
                "so continuity forces a finite Legendre-rank-changing surface"
            ),
        }
    unbounded_speed = any(
        limit.has(sp.oo, -sp.oo, sp.zoo)
        for limit in (transverse_limit, longitudinal_limit)
    )
    maximum_speed = (
        sp.factor(longitudinal_speed)
        if matched_term and matched_consistent and x not in longitudinal_speed.free_symbols
        else None
    )
    cone_certificate: dict[str, Any]
    if maximum_speed is not None and gamma_symbol is not None and epsilon_symbol is not None:
        prefactor = sp.factor(maximum_speed / (gamma_symbol / epsilon_symbol))
        base_cone = _prove_relation(sp.Le(gamma_symbol, epsilon_symbol), domain)
        if prefactor.is_Rational and 0 < prefactor <= 1 and base_cone["status"] == "pass":
            cone_certificate = {
                "status": "pass",
                "condition": f"{maximum_speed} <= 1",
                "normalized_property": f"gamma/epsilon <= 1 and prefactor={prefactor} <= 1",
                "proof": "declared gamma<=epsilon plus exact rational speed prefactor",
            }
        else:
            cone_certificate = _prove_relation(sp.Le(maximum_speed, 1), domain)
    else:
        cone_certificate = {
            "status": "unresolved",
            "condition": "global speed maximum not derived",
        }
    if matched_terms and not matched_consistent:
        status = "reject"
        conclusion = "the nonlinear X exponent and derivative-matched completion exponent disagree"
    elif any(item["status"] == "reject" for item in sign_certificates.values()):
        status = "reject"
        conclusion = "the declared parameter domain fails a necessary kinetic or gradient sign"
    elif any(item["status"] != "pass" for item in sign_certificates.values()):
        status = "unresolved"
        conclusion = "the declared parameter domain does not prove every necessary kinetic sign"
    elif generic_shear_witness and generic_shear_witness["status"] == "reject":
        status = "reject"
        conclusion = (
            "a finite declared-domain traceless-curvature witness makes the Schur-reduced "
            "longitudinal kinetic eigenvalue negative, forcing a Legendre-rank-changing surface"
        )
    elif unbounded_speed:
        status = "reject"
        conclusion = (
            "the nonlinear kinetic Hessian decays at high X while the static-null K1 gradient "
            "coefficient stays finite, so at least one characteristic speed is unbounded"
        )
    elif cone_certificate["status"] == "reject":
        status = "reject"
        conclusion = "the declared parameter domain does not keep the matched cone inside the matter cone"
    else:
        status = "pass"
        conclusion = (
            "the necessary finite-background vector-sector checks pass, but the full "
            "metric-vector Dirac and principal system remains required"
        )
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "applicable": True,
        "term_id": term_id,
        "completion_term_id": matched_term,
        "completion_kind": completion_kind,
        "completion_exponent_consistent": matched_consistent if matched_term else None,
        "exponent": str(exponent),
        "background_variable": "X=v_i v^i/a_sigma^2>=0 in the aligned frozen local frame",
        "local_linear_acceleration_hessian": str(local_linear_hessian),
        "kinetic_hessian_eigenvalues": {
            "transverse": str(transverse_kinetic),
            "longitudinal": str(longitudinal_kinetic),
        },
        "gradient_hessian_eigenvalues": {
            "transverse": str(transverse_gradient),
            "longitudinal": str(longitudinal_gradient),
        },
        "characteristic_speed_squared": {
            "transverse": str(transverse_speed),
            "longitudinal": str(longitudinal_speed),
        },
        "high_X_speed_squared_limits": {
            "transverse": str(transverse_limit),
            "longitudinal": str(longitudinal_limit),
        },
        "unbounded_characteristic_speed": unbounded_speed,
        "global_maximum_speed_squared": str(maximum_speed) if maximum_speed is not None else None,
        "matter_cone_certificate": cone_certificate,
        "radial_legendre_hamiltonian": str(radial_hamiltonian),
        "radial_hamiltonian_derivative": str(radial_hamiltonian_derivative),
        "hamiltonian_derivative_residual": str(hamiltonian_derivative_residual),
        "hamiltonian_positivity_argument": (
            "H(0)=0 and dH/dX>0 for coefficient>0, X>=0, and p>=1/2"
        ),
        "sign_certificates": sign_certificates,
        "coupled_zero_extrinsic_curvature_legendre": coupled_legendre,
        "generic_traceless_curvature_legendre_witness": generic_shear_witness,
        "conclusion": conclusion,
        "observational_data_opened": False,
        "proof_scope": (
            "exact finite-background aligned vector-sector Legendre Hessian, radial energy, "
            "characteristic-speed limits, and for matched terms the coupled metric-vector "
            "Hessian at K_ij=0, plus an exact generic-shear rejection witness when applicable; "
            "distributed closure remains separate"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}


def write_x_operator_ir(payload: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
