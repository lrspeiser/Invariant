from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    _first_order_generalized_pencil,
    build_quartic_horndeski_x2_kessence_modified_harmonic_symbol,
    quartic_horndeski_baseline_riesz_symmetrizer_control,
)

SCHEMA_VERSION = "sigma-quartic-symmetrizer-domain-campaign-1.0"


class QuarticSymmetrizerDomainError(ValueError):
    """Raised when a quartic candidate cannot be bound to the proven domain."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _polynomial_absolute_bound(
    expression: sp.Expr,
    variables: list[sp.Symbol],
    bounds: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), *variables)
    result = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        term = abs(coefficient)
        for variable, power in zip(variables, powers, strict=True):
            term *= bounds[variable] ** power
        result += term
    return sp.factor(result)


def _matrix_frobenius_bound(
    matrix: sp.Matrix,
    variables: list[sp.Symbol],
    bounds: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    entry_bounds = [
        _polynomial_absolute_bound(entry, variables, bounds) for entry in matrix
    ]
    return sp.sqrt(sp.factor(sum(value**2 for value in entry_bounds)))


def _positive(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 80) > 0)


def _pure_gauge_kernel_is_exact(data: dict[str, Any]) -> bool:
    xi_lower = data["xi_lower"]
    action_symbol = data["action_symbol"]
    basis = data["basis"]
    for gauge_index in range(4):
        gauge_covector = sp.zeros(4, 1)
        gauge_covector[gauge_index] = 1
        tensor = xi_lower * gauge_covector.T + gauge_covector * xi_lower.T
        vector = sp.Matrix(
            [
                sum(
                    item[row, column] * tensor[row, column]
                    for row in range(4)
                    for column in range(4)
                )
                for item in basis
            ]
            + [0]
        )
        if not (action_symbol * vector).applyfunc(sp.factor).is_zero_matrix:
            return False
    return True


def certify_quartic_symmetrizer_domain(
    coefficients: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    """Certify one fixed-coefficient quartic candidate on a local-jet box."""

    data = build_quartic_horndeski_x2_kessence_modified_harmonic_symbol()
    alpha = data["alpha"]
    c20 = data["c20"]
    m2 = data["m2"]
    xi_lower = data["xi_lower"]
    alpha_value = sp.sympify(coefficients["a10"])
    c20_value = sp.sympify(coefficients["c20"])
    m2_value = sp.sympify(coefficients["m2"])
    if m2_value != 1:
        raise QuarticSymmetrizerDomainError("current exact baseline requires m2=1")
    if sp.sympify(coefficients["a20"]) != 0 or sp.sympify(
        coefficients["d10"]
    ) != 0:
        raise QuarticSymmetrizerDomainError(
            "candidate is not G3-free with G4 linear in X"
        )
    for name in ("a01", "c02", "c11", "d01"):
        if sp.sympify(coefficients[name]) != 0:
            raise QuarticSymmetrizerDomainError(
                f"coefficient {name} is outside the extracted shift-symmetric action"
            )

    radius = sp.sympify(config["normalized_local_jet_component_abs"])
    if radius.is_positive is not True:
        raise QuarticSymmetrizerDomainError("local-jet radius must be positive")
    spatial_covector_bound = sp.sympify(
        config.get("spatial_covector_component_abs", "1")
    )
    if spatial_covector_bound < 1:
        raise QuarticSymmetrizerDomainError(
            "direction component bound must contain the Euclidean unit sphere"
        )

    gradient_symbols = list(data["gradient_lower"])
    hessian_symbols = sorted(data["hessian_lower"].free_symbols, key=str)
    einstein_symbols = sorted(data["einstein_upper"].free_symbols, key=str)
    jet_symbols = gradient_symbols + hessian_symbols + einstein_symbols
    direction_symbols = list(xi_lower[1:])
    variables = jet_symbols + direction_symbols
    bounds = {
        **{symbol: radius for symbol in jet_symbols},
        **{symbol: spatial_covector_bound for symbol in direction_symbols},
    }
    candidate_substitutions = {alpha: alpha_value, c20: c20_value, m2: 1}
    baseline_substitutions = {alpha: 0, c20: 0, m2: 1}

    first_order = data["first_order"]
    candidate_a = first_order["A"].subs(candidate_substitutions)
    candidate_b = first_order["B"].subs(candidate_substitutions)
    candidate_c = first_order["C"].subs(candidate_substitutions)
    baseline_a = first_order["A"].subs(baseline_substitutions)
    baseline_b = first_order["B"].subs(baseline_substitutions)
    baseline_c = first_order["C"].subs(baseline_substitutions)
    delta_a = (candidate_a - baseline_a).applyfunc(sp.expand)
    delta_b = (candidate_b - baseline_b).applyfunc(sp.expand)
    delta_c = (candidate_c - baseline_c).applyfunc(sp.expand)
    delta_a_bound = _matrix_frobenius_bound(delta_a, variables, bounds)
    delta_b_bound = _matrix_frobenius_bound(delta_b, variables, bounds)
    delta_c_bound = _matrix_frobenius_bound(delta_c, variables, bounds)
    baseline_b_bound = _matrix_frobenius_bound(
        baseline_b, direction_symbols, bounds
    )
    baseline_c_bound = _matrix_frobenius_bound(
        baseline_c, direction_symbols, bounds
    )

    baseline_a_inverse_bound = sp.Integer(4)
    neumann_denominator = sp.factor(
        1 - baseline_a_inverse_bound * delta_a_bound
    )
    if not _positive(neumann_denominator):
        raise QuarticSymmetrizerDomainError(
            "time block is outside the baseline Neumann-invertibility radius"
        )
    candidate_a_inverse_bound = sp.factor(
        baseline_a_inverse_bound / neumann_denominator
    )
    inverse_difference_bound = sp.factor(
        candidate_a_inverse_bound
        * delta_a_bound
        * baseline_a_inverse_bound
    )
    companion_b_block_bound = sp.factor(
        candidate_a_inverse_bound * delta_b_bound
        + inverse_difference_bound * baseline_b_bound
    )
    companion_c_block_bound = sp.factor(
        candidate_a_inverse_bound * delta_c_bound
        + inverse_difference_bound * baseline_c_bound
    )
    companion_perturbation_bound = sp.sqrt(
        sp.factor(companion_b_block_bound**2 + companion_c_block_bound**2)
    )

    action_first_order = _first_order_generalized_pencil(
        data["action_symbol"], xi_lower[0]
    )
    canonical_action_first_order = _first_order_generalized_pencil(
        data["canonical_quartic_data"]["action_symbol"], xi_lower[0]
    )
    candidate_action_a = action_first_order["A"].subs(candidate_substitutions)
    candidate_action_b = action_first_order["B"].subs(candidate_substitutions)
    candidate_action_c = action_first_order["C"].subs(candidate_substitutions)
    baseline_action_a = canonical_action_first_order["A"].subs(
        {alpha: 0, m2: 1}
    )
    baseline_action_b = canonical_action_first_order["B"].subs(
        {alpha: 0, m2: 1}
    )
    baseline_action_c = canonical_action_first_order["C"].subs(
        {alpha: 0, m2: 1}
    )
    delta_action_a_bound = _matrix_frobenius_bound(
        (candidate_action_a - baseline_action_a).applyfunc(sp.expand),
        variables,
        bounds,
    )
    delta_action_b_bound = _matrix_frobenius_bound(
        (candidate_action_b - baseline_action_b).applyfunc(sp.expand),
        variables,
        bounds,
    )
    delta_action_c_bound = _matrix_frobenius_bound(
        (candidate_action_c - baseline_action_c).applyfunc(sp.expand),
        variables,
        bounds,
    )
    h_star_perturbation_bound = sp.sqrt(
        sp.factor(delta_action_b_bound**2 + 2 * delta_action_a_bound**2)
    )
    hat_action_symbol_perturbation_bound = sp.factor(
        delta_action_a_bound / 9
        + delta_action_b_bound / 3
        + delta_action_c_bound
    )

    baseline_passed, baseline_evidence = (
        quartic_horndeski_baseline_riesz_symmetrizer_control()
    )
    if not baseline_passed:
        raise QuarticSymmetrizerDomainError("baseline symmetrizer control failed")
    contract = baseline_evidence[
        "quantitative_physical_group_perturbation_contract"
    ]
    companion_budget = sp.sympify(
        contract["required_companion_2_norm_perturbation_upper"]
    )
    h_star_budget = sp.sympify(
        contract["required_H_star_2_norm_perturbation_upper"]
    )
    hat_rank_budget = sp.sympify(
        baseline_evidence["hat_group_baseline_restricted_rank"][
            "smallest_nonzero_singular_value"
        ]
    )
    companion_margin = sp.factor(companion_budget - companion_perturbation_bound)
    h_star_margin = sp.factor(h_star_budget - h_star_perturbation_bound)
    hat_rank_margin = sp.factor(
        hat_rank_budget - hat_action_symbol_perturbation_bound
    )
    pure_gauge_exact = _pure_gauge_kernel_is_exact(data)
    passed = bool(
        pure_gauge_exact
        and _positive(companion_margin)
        and _positive(h_star_margin)
        and _positive(hat_rank_margin)
    )
    return {
        "schema_version": "sigma-quartic-symmetrizer-domain-certificate-1.0",
        "status": "pass_uniform_local_jet_strong_hyperbolicity"
        if passed
        else "reject",
        "coefficients": {name: str(value) for name, value in sorted(coefficients.items())},
        "domain": {
            "frame": "local physical orthonormal frame",
            "normalized_local_jet_component_abs": str(radius),
            "bounded_components": {
                "nabla_mu_phi": 4,
                "nabla_mu_nabla_nu_phi_symmetric": 10,
                "Einstein_tensor_symmetric": 10,
            },
            "spatial_direction": (
                "all Euclidean-unit covectors; bounded through the containing component cube "
                f"|n_i|<={spatial_covector_bound}"
            ),
            "contains_exact_solution": (
                "Minkowski metric with constant phi, for which every bounded local jet is zero"
            ),
            "on_shell_invariance_status": "unresolved",
        },
        "uniform_matrix_bounds": {
            "Delta_A_F": str(delta_a_bound),
            "Delta_B_F": str(delta_b_bound),
            "Delta_C_F": str(delta_c_bound),
            "baseline_A_inverse_2": str(baseline_a_inverse_bound),
            "neumann_denominator_lower": str(neumann_denominator),
            "candidate_A_inverse_2_upper": str(candidate_a_inverse_bound),
            "baseline_B_F_cube_upper": str(baseline_b_bound),
            "baseline_C_F_cube_upper": str(baseline_c_bound),
            "companion_2_norm_perturbation_upper": str(
                companion_perturbation_bound
            ),
            "companion_2_norm_perturbation_upper_numeric": float(
                sp.N(companion_perturbation_bound, 17)
            ),
            "companion_budget": str(companion_budget),
            "companion_margin": str(companion_margin),
            "companion_margin_numeric": float(sp.N(companion_margin, 17)),
            "H_star_2_norm_perturbation_upper": str(h_star_perturbation_bound),
            "H_star_2_norm_perturbation_upper_numeric": float(
                sp.N(h_star_perturbation_bound, 17)
            ),
            "H_star_budget": str(h_star_budget),
            "H_star_margin": str(h_star_margin),
            "hat_action_symbol_perturbation_upper": str(
                hat_action_symbol_perturbation_bound
            ),
            "hat_baseline_smallest_nonzero_singular_value": str(hat_rank_budget),
            "hat_rank_margin": str(hat_rank_margin),
        },
        "theorem_binding": {
            "time_block_noncharacteristic": _positive(neumann_denominator),
            "six_Riesz_groups_remain_disjoint": _positive(companion_margin),
            "physical_H_star_positive": _positive(h_star_margin),
            "four_action_pure_gauge_kernel_vectors_exact": pure_gauge_exact,
            "hat_restricted_action_rank_exactly_seven": _positive(hat_rank_margin)
            and pure_gauge_exact,
            "auxiliary_cones": "fixed nested speeds 1/2 and 1/3",
            "source": "Kovacs--Reall arXiv:2003.08398, equations Mdef, Pidef, defHstar, Hsym2",
        },
        "claim": (
            "The complete 22-by-22 modified-harmonic companion matrix is strongly hyperbolic "
            "throughout the declared arbitrary-local-jet box for every spatial direction."
        ),
        "scope": (
            "This is a sufficient pointwise principal-symbol theorem around the exact "
            "Einstein-scalar/Minkowski baseline. It does not prove nonlinear evolution preserves "
            "the box, global Hamiltonian positivity, or observational viability."
        ),
    }


def run_quartic_symmetrizer_domain_campaign(
    ir: dict[str, Any], binding_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticSymmetrizerDomainError("unsupported campaign schema_version")
        if binding_campaign.get("status") != (
            "pass_exact_symbol_binding_uniform_symmetrizer_unresolved"
        ):
            raise QuarticSymmetrizerDomainError(
                "input binding campaign is not an exact 12-candidate binding"
            )
        if binding_campaign.get("source_ir_sha256") != ir.get("content_sha256"):
            raise QuarticSymmetrizerDomainError("binding campaign source IR hash mismatch")
        candidates = binding_campaign.get("candidates", [])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidates) != expected:
            raise QuarticSymmetrizerDomainError(
                f"expected {expected} candidates, found {len(candidates)}"
            )
        certificates = [
            {
                "candidate_id": record["candidate_id"],
                **certify_quartic_symmetrizer_domain(record["coefficients"], config),
            }
            for record in candidates
        ]
        passed_count = sum(
            record["status"] == "pass_uniform_local_jet_strong_hyperbolicity"
            for record in certificates
        )

        bad_config = json.loads(json.dumps(config))
        bad_config["normalized_local_jet_component_abs"] = "1/1000000"
        bad_radius = certify_quartic_symmetrizer_domain(
            candidates[0]["coefficients"], bad_config
        )
        corrupted_coefficients = dict(candidates[0]["coefficients"])
        corrupted_coefficients["a01"] = "1"
        coefficient_rejected = False
        coefficient_error = ""
        try:
            certify_quartic_symmetrizer_domain(corrupted_coefficients, config)
        except QuarticSymmetrizerDomainError as error:
            coefficient_rejected = True
            coefficient_error = str(error)
        if bad_radius.get("status") != "reject" or not coefficient_rejected:
            raise QuarticSymmetrizerDomainError("a campaign negative control did not reject")

        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
            if passed_count == expected
            else "reject",
            "errors": [],
            "source_ir_sha256": ir.get("content_sha256"),
            "binding_campaign_sha256": binding_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(
                _canonical_json(config).encode()
            ).hexdigest(),
            "counts": {
                "selected": len(candidates),
                "uniform_local_jet_strong_hyperbolicity_passed": passed_count,
                "rejected": len(candidates) - passed_count,
            },
            "certificates": certificates,
            "negative_controls": {
                "large_jet_box": {
                    "radius": bad_config["normalized_local_jet_component_abs"],
                    "status": bad_radius.get("status"),
                    "companion_margin_numeric": bad_radius.get(
                        "uniform_matrix_bounds", {}
                    ).get("companion_margin_numeric"),
                    "rejected": bad_radius.get("status") == "reject",
                },
                "phi_dependent_G4": {
                    "mutation": {"a01": "1"},
                    "rejected": coefficient_rejected,
                    "error": coefficient_error,
                },
            },
            "claim": (
                "Every one of the 12 fixed-coefficient G3-free, linear-X quartic candidates "
                "has a nonzero arbitrary-local-jet box satisfying the complete modified-harmonic "
                "strong-hyperbolicity construction for every spatial direction."
            ),
            "scope": (
                "The common box is deliberately conservative and centered on an exact Minkowski/"
                "constant-scalar solution. Nonlinear evolution-invariance, larger connected "
                "domains, global energy, and observation gates remain separate."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticSymmetrizerDomainError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "counts": {
                "selected": 0,
                "uniform_local_jet_strong_hyperbolicity_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_symmetrizer_domain_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
