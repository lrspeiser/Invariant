from __future__ import annotations

import ast
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .legendre_ir import build_local_kinetic_model

SCHEMA_VERSION = "sigma-static-dictionary-ir-1.0"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _parse_generator_expression(value: str) -> sp.Expr:
    tree = ast.parse(value.replace("^", "**"), mode="eval")
    symbols = {name: sp.Symbol(name, real=True) for name in ("x", "q", "z")}

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return convert(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return sp.sympify(node.value)
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = convert(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "sqrt"
            and len(node.args) == 1
            and not node.keywords
        ):
            return sp.sqrt(convert(node.args[0]))
        raise TypeError(f"unsupported generator expression syntax: {type(node).__name__}")

    return sp.factor(convert(tree))


def classify_generator_expression(
    expression: str, *, aether_x_available: bool, exact_q_action_match: bool = False
) -> dict[str, Any]:
    parsed = _parse_generator_expression(expression)
    x, q, z = sp.symbols("x q z", real=True)
    used = sorted(str(symbol) for symbol in parsed.free_symbols)
    if z in parsed.free_symbols:
        decision = "reject_forbidden_baryonic_action_atom"
        reason = (
            "z_b is a diagnostic of a selected matter current and cannot enter S_grav under "
            "universal minimal matter coupling"
        )
    elif q in parsed.free_symbols and exact_q_action_match:
        decision = "supported_exact_projected_aether_q_lift"
        reason = (
            "the action's normalized Q_a_u term polynomial exactly equals this legacy "
            "generator expression on the derived static ansatz"
        )
    elif q in parsed.free_symbols:
        decision = "unresolved_missing_covariant_q_atom"
        reason = (
            "q requires a projected spatial derivative of Aether acceleration (or another "
            "explicit covariant completion) absent from the bounded action grammar"
        )
    elif x in parsed.free_symbols:
        affine = sp.diff(parsed, x, 2) == 0
        if affine and aether_x_available:
            decision = "supported_linear_aether_x_lift"
            reason = "the affine x correction maps to the derived unit-Aether acceleration sector"
        elif not aether_x_available:
            decision = "unresolved_no_unit_aether_x_dictionary"
            reason = "this action has no unit-timelike Aether acceleration congruence"
        else:
            decision = "unresolved_missing_nonlinear_aether_acceleration_adapter"
            reason = "nonlinear F(x) is outside the current linear K1..K4 action grammar"
    else:
        decision = "unresolved_constant_term_not_in_grammar"
        reason = "a standalone constant density/cosmological term is not in the bounded grammar"
    return {
        "expression": expression,
        "canonical_expression": str(parsed),
        "used_legacy_variables": used,
        "decision": decision,
        "reason": reason,
    }


def _static_tensor_reductions() -> dict[str, Any]:
    metric = sp.diag(-1, 1, 1, 1)
    unit_up = sp.Matrix([1, 0, 0, 0])
    unit_down = metric * unit_up
    a1, a2, a3 = sp.symbols("a1 a2 a3", real=True)
    acceleration_up = sp.Matrix([0, a1, a2, a3])
    acceleration_down = metric * acceleration_up
    derivative = -unit_down * acceleration_down.T
    derivative_up = metric * derivative * metric
    acceleration_squared = sp.factor((acceleration_down.T * acceleration_up)[0])
    k1 = sp.factor(sum(derivative[i, j] * derivative_up[i, j] for i in range(4) for j in range(4)))
    divergence = sp.factor(sum(metric[i, j] * derivative[i, j] for i in range(4) for j in range(4)))
    k2 = sp.factor(divergence**2)
    k3 = sp.factor(sum(derivative[i, j] * derivative_up[j, i] for i in range(4) for j in range(4)))
    aether_acceleration_down = sp.simplify(unit_up.T * derivative).T
    k4 = sp.factor((aether_acceleration_down.T * metric * aether_acceleration_down)[0])

    projector_up = metric + unit_up * unit_up.T
    b_components = sp.symbols("b11 b12 b13 b21 b22 b23 b31 b32 b33", real=True)
    projected_acceleration_gradient = sp.zeros(4)
    for row in range(3):
        for column in range(3):
            projected_acceleration_gradient[row + 1, column + 1] = b_components[
                3 * row + column
            ]
    q_contraction = sp.factor(
        sum(
            projector_up[mu, rho]
            * projector_up[nu, sigma]
            * projected_acceleration_gradient[mu, nu]
            * projected_acceleration_gradient[rho, sigma]
            for mu in range(4)
            for nu in range(4)
            for rho in range(4)
            for sigma in range(4)
        )
    )
    q_expected = sp.factor(sum(component**2 for component in b_components))

    s1, s2, s3, lambda_phi = sp.symbols("s1 s2 s3 Lambda_phi", real=True)
    scalar_gradient = sp.Matrix([0, s1, s2, s3])
    scalar_spatial_square = sp.factor(sum(item**2 for item in (s1, s2, s3)))
    x_phi = sp.factor(
        -(scalar_gradient.T * metric * scalar_gradient)[0] / (2 * lambda_phi**4)
    )

    electric = sp.symbols("E1:4", real=True)
    magnetic = sp.symbols("B1:4", real=True)
    field_strength = sp.zeros(4)
    for index in range(3):
        field_strength[0, index + 1] = electric[index]
        field_strength[index + 1, 0] = -electric[index]
    field_strength[1, 2], field_strength[2, 1] = magnetic[2], -magnetic[2]
    field_strength[2, 3], field_strength[3, 2] = magnetic[0], -magnetic[0]
    field_strength[3, 1], field_strength[1, 3] = magnetic[1], -magnetic[1]
    field_strength_up = metric * field_strength * metric
    f2 = sp.factor(
        sum(
            field_strength[i, j] * field_strength_up[i, j]
            for i in range(4)
            for j in range(4)
        )
    )

    number_density, reference_density = sp.symbols("n_b n_0", positive=True)
    baryon_current_up = number_density * unit_up
    baryon_current_down = metric * baryon_current_up
    z_b = sp.factor(
        -(baryon_current_down.T * baryon_current_up)[0] / reference_density**2
    )

    lapse, sqrt_h, curvature3, laplacian_lapse = sp.symbols(
        "N sqrt_h R3 D2N", real=True
    )
    curvature4 = curvature3 - 2 * laplacian_lapse / lapse
    eh_density_residual = sp.factor(
        lapse * sqrt_h * curvature4
        - (lapse * sqrt_h * curvature3 - 2 * sqrt_h * laplacian_lapse)
    )
    return {
        "static_ansatz": {
            "line_element": "ds^2=-N(x)^2 dt^2+h_ij(x) dx^i dx^j",
            "shift": "N^i=0",
            "time_independence": "partial_t N=partial_t h_ij=partial_t fields=0",
            "extrinsic_curvature": "K_ij=0",
            "unit_aether": "u^a=n^a=N^-1(partial_t)^a",
            "aether_acceleration": "a_i=D_i ln N",
        },
        "aether": {
            "derivative_tensor": str(derivative),
            "identity": "nabla_mu u_nu=-u_mu a_nu when K_mu_nu=0",
            "orthogonality_residual": str((unit_down.T * acceleration_up)[0]),
            "acceleration_squared": str(acceleration_squared),
            "K1_over_Lu2": str(k1),
            "K2_over_Lu2": str(k2),
            "K3_over_Lu2": str(k3),
            "K4_over_Lu2": str(k4),
            "expected": {
                "K1_over_Lu2": str(-acceleration_squared),
                "K2_over_Lu2": "0",
                "K3_over_Lu2": "0",
                "K4_over_Lu2": str(acceleration_squared),
            },
        },
        "aether_acceleration_gradient": {
            "projector": "P_mu_nu=g_mu_nu+u_mu u_nu",
            "acceleration": "a_mu=u^alpha nabla_alpha u_mu",
            "covariant_invariant": (
                "Q_a_u=(L_sigma^2/a_sigma^2) P^{mu rho} P^{nu sigma} "
                "nabla_mu(a_nu) nabla_rho(a_sigma)"
            ),
            "static_projected_gradient": str(projected_acceleration_gradient),
            "projected_contraction": str(q_contraction),
            "expected_spatial_contraction": str(q_expected),
            "static_Q_a_u": f"L_sigma^2*({q_contraction})/a_sigma^2",
            "legacy_q_identification": (
                "D_i=c^2 a_i gives q=L_sigma^2 c^4 "
                "(D_i a_j)(D^i a^j)/a_sigma^2; c=1 in the covariant contract"
            ),
            "generic_tilt_warning": (
                "P is orthogonal to u, not necessarily to the ADM normal n; for tilted u, "
                "the projected derivative can contain ADM-normal derivatives of a_mu"
            ),
        },
        "scalar": {
            "static_gradient_squared": str(scalar_spatial_square),
            "X_phi": str(x_phi),
            "expected": str(sp.factor(-scalar_spatial_square / (2 * lambda_phi**4))),
        },
        "proca": {
            "F_mu_nu_F_mu_nu": str(f2),
            "expected": str(
                sp.factor(
                    2
                    * (
                        sum(item**2 for item in magnetic)
                        - sum(item**2 for item in electric)
                    )
                )
            ),
        },
        "einstein_hilbert": {
            "R4_static": str(curvature4),
            "density_split": "N sqrt(h) R4=N sqrt(h) R3-2 sqrt(h) D^2 N",
            "density_split_residual": str(eh_density_residual),
            "boundary_term": "sqrt(h) D^2 N=partial_i[sqrt(h) D^i N]",
        },
        "baryonic_diagnostic": {
            "current": "J_b^mu=n_b v_b^mu, v_b^mu v^b_mu=-1",
            "z_b": str(z_b),
            "expected": "n_b**2/n_0**2",
            "locally_measurable": (
                "n_b=-v^b_mu J_b^mu is the local baryon-number density in the matter rest frame"
            ),
            "generator_status": "forbidden action atom; diagnostic only",
        },
    }


def compile_static_dictionary_ir(
    action_ir: dict[str, Any], generator_expression: str | None = None
) -> dict[str, Any]:
    if not action_ir.get("valid"):
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "promotion_allowed": False,
            "input_action_sha256": action_ir.get("content_sha256"),
            "errors": ["covariant action IR is invalid"],
        }
    reductions = _static_tensor_reductions()
    aether = reductions["aether"]
    aether_q = reductions["aether_acceleration_gradient"]
    scalar = reductions["scalar"]
    proca = reductions["proca"]
    eh = reductions["einstein_hilbert"]
    exact_reductions = (
        aether["orthogonality_residual"] == "0"
        and aether["K1_over_Lu2"] == aether["expected"]["K1_over_Lu2"]
        and aether["K2_over_Lu2"] == "0"
        and aether["K3_over_Lu2"] == "0"
        and aether["K4_over_Lu2"] == aether["expected"]["K4_over_Lu2"]
        and aether_q["projected_contraction"]
        == aether_q["expected_spatial_contraction"]
        and scalar["X_phi"] == scalar["expected"]
        and proca["F_mu_nu_F_mu_nu"] == proca["expected"]
        and eh["density_split_residual"] == "0"
        and reductions["baryonic_diagnostic"]["z_b"]
        == reductions["baryonic_diagnostic"]["expected"]
    )
    fields = set(action_ir["canonical"]["fields"])
    model = build_local_kinetic_model(action_ir)
    coefficients = model["coefficients"]
    aether_static_coefficient = sp.factor(
        -coefficients.get("AETHER_K1", 0) + coefficients.get("AETHER_K4", 0)
    )
    term_ids = {item["id"] for item in action_ir["canonical"]["terms"]}
    supported_terms = {
        "EH_R",
        "SCALAR_X",
        "SCALAR_MASS",
        "PROCA_F2",
        "PROCA_MASS",
        "AETHER_K1",
        "AETHER_K2",
        "AETHER_K3",
        "AETHER_K4",
        "AETHER_X_SQRT1P",
        "AETHER_X_P2_3",
        "AETHER_X_P3_4",
        "AETHER_MATCHED_K14_P1_2",
        "AETHER_MATCHED_K14_P2_3",
        "AETHER_MATCHED_K14_P3_4",
        "AETHER_Q1",
        "AETHER_Q2",
        "AETHER_Q3",
        "UNIT_VECTOR_CONSTRAINT",
    }
    unsupported_terms = sorted(term_ids - supported_terms)
    q_term_powers = {"AETHER_Q1": 1, "AETHER_Q2": 2, "AETHER_Q3": 3}
    q_terms = sorted(term_ids & set(q_term_powers))
    x_terms = sorted(
        item["id"]
        for item in action_ir["canonical"]["terms"]
        if item.get("invariant") == "X_a_u" and item.get("static_legacy_atom") is not None
    )
    legacy_shape: sp.Expr | None = None
    q_match: bool | None = None
    target_expression = generator_expression
    if target_expression is None:
        target_expression = action_ir["canonical"].get("generator_origin", {}).get(
            "correction_expression"
        )
    legacy_terms = sorted(
        item["id"]
        for item in action_ir["canonical"]["terms"]
        if item.get("static_legacy_atom") is not None
    )
    if legacy_terms:
        legacy_atoms = {
            item["id"]: _parse_generator_expression(item["static_legacy_atom"])
            for item in action_ir["canonical"]["terms"]
            if item["id"] in legacy_terms
        }
        reference_coefficient = coefficients[legacy_terms[0]]
        legacy_shape = sp.factor(
            sum(
                sp.cancel(coefficients[term_id] / reference_coefficient)
                * legacy_atoms[term_id]
                for term_id in legacy_terms
            )
        )
        if target_expression is not None:
            q_match = sp.factor(
                _parse_generator_expression(target_expression) - legacy_shape
            ) == 0
    expression_classification = (
        classify_generator_expression(
            target_expression,
            aether_x_available="u_mu" in fields,
            exact_q_action_match=bool(q_terms) and q_match is True,
        )
        if target_expression is not None
        else None
    )
    generator_match_ok = q_match is not False
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "pass"
            if exact_reductions and not unsupported_terms and generator_match_ok
            else "unresolved"
        ),
        "promotion_allowed": False,
        "input_action_sha256": action_ir["content_sha256"],
        "source_role": action_ir["canonical"]["source_role"],
        "matter_metric": action_ir["canonical"]["matter_metric"],
        "universal_matter_coupling_preserved": (
            action_ir["canonical"]["matter_metric"] == "g_mu_nu"
            and "z_b" not in {item.get("invariant") for item in action_ir["canonical"]["terms"]}
        ),
        "static_reductions": reductions,
        "term_adapter_coverage": {
            "term_ids": sorted(term_ids),
            "unsupported_term_ids": unsupported_terms,
            "all_current_terms_supported": not unsupported_terms,
        },
        "aether_static_acceleration_sector": {
            "present": "u_mu" in fields,
            "action_density_coefficient_of_a_i_a^i": str(aether_static_coefficient),
            "derivation": "-C_K1+C_K4 because K1/Lu^2=-a^2 and K4/Lu^2=+a^2",
        },
        "static_null_matched_completions": {
            "term_ids": sorted(
                term_ids
                & {
                    "AETHER_MATCHED_K14_P1_2",
                    "AETHER_MATCHED_K14_P2_3",
                    "AETHER_MATCHED_K14_P3_4",
                }
            ),
            "exact_static_density": "W_p(x)*(K1_u+K4_u)/L_u^2=0",
            "derivation": "K1_u/L_u^2=-a_i a^i and K4_u/L_u^2=+a_i a^i",
        },
        "legacy_generator_dictionary": {
            "x": {
                "status": (
                    "derived_and_generator_matched"
                    if q_match is True and x_terms
                    else ("derived" if "u_mu" in fields else "unavailable_for_this_action")
                ),
                "definition": "x=c^4 a_i a^i/a_sigma^2, a_i=D_i ln N",
                "measurement": "a_i is the proper acceleration measured by a static-observer accelerometer",
                "nonlinear_action_term_present": bool(x_terms),
                "action_term_ids": x_terms,
            },
            "q": {
                "status": (
                    "derived_and_generator_matched"
                    if q_terms and q_match is True
                    else (
                        "derived_covariant_atom"
                        if q_terms and q_match is None
                        else (
                            "generator_expression_mismatch"
                            if q_match is False
                            else "missing_covariant_action_atom"
                        )
                    )
                ),
                "covariant_definition": aether_q["covariant_invariant"],
                "static_definition": (
                    "q=L_sigma^2 c^4 (D_i a_j)(D^i a^j)/a_sigma^2"
                ),
                "action_term_ids": q_terms,
                "normalized_action_shape": (
                    str(legacy_shape) if legacy_shape is not None else None
                ),
                "target_generator_expression": target_expression,
                "exact_shape_match": q_match,
                "warning": (
                    "the static contraction is derived, but generic-tilt variation, ADM/Dirac, "
                    "Hamiltonian, and hyperbolicity adapters are still required"
                ),
            },
            "z": reductions["baryonic_diagnostic"],
        },
        "generator_expression_classification": expression_classification,
        "proof_scope": (
            "exact local tensor contractions on the static zero-shift, K_ij=0 ansatz; the "
            "dictionary derives invariant reductions but does not assert that an unsupported "
            "legacy q or nonlinear F(x) already has a healthy covariant completion"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def audit_priority_static_lift(
    priority_report: dict[str, Any], dictionary_ir: dict[str, Any]
) -> dict[str, Any]:
    if dictionary_ir.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported static dictionary IR")
    queue = priority_report.get("work_queue", [])
    records: list[dict[str, Any]] = []
    decisions: Counter[str] = Counter()
    for candidate in queue:
        classification = classify_generator_expression(
            candidate["correction_expression"],
            aether_x_available=dictionary_ir["legacy_generator_dictionary"]["x"]["status"]
            == "derived",
        )
        decisions[classification["decision"]] += 1
        records.append(
            {
                "family_id": candidate["family_id"],
                "ordinal": candidate["ordinal"],
                "pareto_front": candidate.get("pareto_front"),
                **classification,
            }
        )
    liftable = [
        item for item in records if item["decision"] == "supported_linear_aether_x_lift"
    ]
    q_only = [
        item
        for item in records
        if item["decision"] == "unresolved_missing_covariant_q_atom"
    ]
    body = {
        "schema_version": "sigma-static-lift-audit-1.0",
        "status": "pass",
        "input_dictionary_sha256": dictionary_ir.get("content_sha256"),
        "priority_schema_version": priority_report.get("schema_version"),
        "queue_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "currently_liftable_count": len(liftable),
        "currently_liftable": liftable,
        "q_backend_queue_count": len(q_only),
        "highest_priority_q_backend_candidates": q_only[:20],
        "records": records,
        "next_backend_target": (
            "derive and formally analyze the projected Aether acceleration-gradient q invariant"
            if q_only
            else "no missing q candidates in this queue"
        ),
        "observational_data_opened": False,
        "interpretation": (
            "static survival is not covariant liftability; z-containing formulas reject, q "
            "formulas wait for a new action/ADM/Dirac adapter, and nonlinear x formulas wait for "
            "a nonlinear Aether acceleration adapter"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}


def write_static_artifact(payload: dict[str, Any], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
