from __future__ import annotations

import hashlib
import json
from functools import cache
from itertools import product
from math import comb
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    build_quartic_horndeski_x2_kessence_modified_harmonic_symbol,
)

SCHEMA_VERSION = "sigma-quartic-quasilinear-moser-campaign-1.0"


class QuarticQuasilinearMoserError(ValueError):
    """Raised when quasilinear coefficient regularity cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


@cache
def _symbol_data() -> dict[str, Any]:
    return build_quartic_horndeski_x2_kessence_modified_harmonic_symbol()


@cache
def generic_inverse_product_derivative_control() -> tuple[bool, dict[str, Any]]:
    """Check the differentiated identity A F=X and its binomial factors through order four."""

    u = sp.Symbol("u", real=True, finite=True)
    a0, a1, a2, x0, x1, x2 = sp.symbols(
        "a0 a1 a2 x0 x1 x2", real=True, finite=True
    )
    a = a0 + a1 * u + a2 * u**2 / 2
    x = x0 + x1 * u + x2 * u**2 / 2
    f = sp.cancel(x / a)
    residuals: dict[str, str] = {}
    exact_residuals: list[sp.Expr] = []
    for order in range(1, 5):
        rhs = sp.diff(x, u, order) - sum(
            sp.binomial(order, derivative_order)
            * sp.diff(a, u, derivative_order)
            * sp.diff(f, u, order - derivative_order)
            for derivative_order in range(1, order + 1)
        )
        residual = sp.factor(sp.diff(f, u, order) - rhs / a)
        exact_residuals.append(residual)
        residuals[str(order)] = str(residual)

    corrupted_second = sp.factor(
        sp.diff(f, u, 2)
        - (
            sp.diff(x, u, 2)
            - sp.diff(a, u) * sp.diff(f, u)
            - sp.diff(a, u, 2) * f
        )
        / a
    )
    witness = {
        u: 2,
        a0: 3,
        a1: 5,
        a2: 7,
        x0: 11,
        x1: 13,
        x2: 17,
    }
    corrupted_witness = sp.factor(corrupted_second.subs(witness))
    passed = all(value == 0 for value in exact_residuals) and corrupted_witness != 0
    return bool(passed), {
        "control": "inverse-product Frechet derivative recurrence through order four",
        "identity": (
            "D^n(A F)=sum_{k=0}^n binomial(n,k) D^k A D^(n-k)F=D^n X, "
            "with F=A^{-1}X"
        ),
        "orders": [1, 2, 3, 4],
        "residuals": residuals,
        "negative_control": {
            "corruption": "replace the n=2 coefficient 2 on DA*DF by 1",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": corrupted_witness != 0,
        },
        "passed": bool(passed),
        "scope": (
            "Exact scalar control of the matrix-norm recurrence. Candidate matrix derivative "
            "envelopes and inverse-time-block bounds are supplied separately."
        ),
    }


def _jet_and_direction_symbols(data: dict[str, Any]) -> tuple[list[sp.Symbol], list[sp.Symbol]]:
    jets = (
        list(data["gradient_lower"])
        + sorted(data["hessian_lower"].free_symbols, key=str)
        + sorted(data["einstein_upper"].free_symbols, key=str)
    )
    directions = list(data["xi_lower"][1:])
    return jets, directions


def _maximum_jet_degree(matrix: sp.Matrix, jets: list[sp.Symbol]) -> int:
    degrees = [
        sp.Poly(sp.expand(entry), *jets).total_degree()
        for entry in matrix
        if entry != 0
    ]
    return max(degrees, default=0)


def _polynomial_absolute_bound(
    expression: sp.Expr,
    variables: list[sp.Symbol],
    bounds: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), *variables)
    total = sp.Integer(0)
    for powers, coefficient in polynomial.terms():
        term = abs(coefficient)
        for variable, power in zip(variables, powers, strict=True):
            term *= bounds[variable] ** power
        total += term
    return sp.factor(total)


def _matrix_entrywise_l1_bound(
    matrix: sp.Matrix,
    variables: list[sp.Symbol],
    bounds: dict[sp.Symbol, sp.Expr],
) -> sp.Expr:
    """An exact entrywise L1 envelope, hence an upper bound on Frobenius and 2-norm."""

    return sp.factor(
        sum(_polynomial_absolute_bound(entry, variables, bounds) for entry in matrix)
    )


def _matrix_derivative_tensor_bound(
    matrix: sp.Matrix,
    order: int,
    jets: list[sp.Symbol],
    all_variables: list[sp.Symbol],
    bounds: dict[sp.Symbol, sp.Expr],
) -> tuple[sp.Expr, int]:
    """Return an exact L-infinity-to-matrix-2-norm Frechet envelope."""

    if order == 0:
        return _matrix_entrywise_l1_bound(matrix, all_variables, bounds), 1
    total = sp.Integer(0)
    nonzero = 0
    for indices in product(range(len(jets)), repeat=order):
        derivative = matrix.diff(*(jets[index] for index in indices))
        if derivative.is_zero_matrix:
            continue
        envelope = _matrix_entrywise_l1_bound(derivative, all_variables, bounds)
        total += envelope
        nonzero += 1
    return sp.factor(total), nonzero


def _inverse_product_bounds(
    inverse_a_upper: sp.Expr,
    a_derivatives: list[sp.Expr],
    x_derivatives: list[sp.Expr],
    order: int,
) -> list[sp.Expr]:
    """Propagate uniform Frechet bounds for F=A^{-1}X."""

    result = [sp.factor(inverse_a_upper * x_derivatives[0])]
    for derivative_order in range(1, order + 1):
        rhs = x_derivatives[derivative_order]
        for a_order in range(1, min(derivative_order, len(a_derivatives) - 1) + 1):
            rhs += (
                comb(derivative_order, a_order)
                * a_derivatives[a_order]
                * result[derivative_order - a_order]
            )
        result.append(sp.factor(inverse_a_upper * rhs))
    return result


def certify_quartic_quasilinear_moser_candidate(
    symmetrizer_candidate: dict[str, Any],
    auxiliary_time_candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = symmetrizer_candidate.get("candidate_id")
    if candidate_id != auxiliary_time_candidate.get("candidate_id"):
        raise QuarticQuasilinearMoserError("candidate ID mismatch")
    if symmetrizer_candidate.get("status") != "pass_uniform_local_jet_strong_hyperbolicity":
        raise QuarticQuasilinearMoserError("candidate lacks the symmetrizer-domain prerequisite")
    if auxiliary_time_candidate.get("status") != "pass_linear_auxiliary_time_reconstruction":
        raise QuarticQuasilinearMoserError("candidate lacks the auxiliary-time prerequisite")
    if symmetrizer_candidate.get("coefficients") != auxiliary_time_candidate.get("coefficients"):
        raise QuarticQuasilinearMoserError("candidate coefficient mismatch")
    generic_passed, _ = generic_inverse_product_derivative_control()
    if not generic_passed:
        raise QuarticQuasilinearMoserError("inverse-product derivative control failed")

    required_order = int(config["required_sobolev_order"])
    declared_degree = int(config["declared_raw_coefficient_jet_degree"])
    if required_order < 4:
        raise QuarticQuasilinearMoserError(
            "Sobolev order must be at least four for the declared three-dimensional C2 tube"
        )
    if declared_degree != 2:
        raise QuarticQuasilinearMoserError(
            "raw A/B/C coefficient degree must be declared as the extracted quadratic degree two"
        )

    data = _symbol_data()
    coefficients = symmetrizer_candidate["coefficients"]
    substitutions = {
        data["alpha"]: sp.sympify(coefficients["a10"]),
        data["c20"]: sp.sympify(coefficients["c20"]),
        data["m2"]: sp.sympify(coefficients["m2"]),
    }
    jets, directions = _jet_and_direction_symbols(data)
    radius = sp.sympify(
        symmetrizer_candidate["domain"]["normalized_local_jet_component_abs"]
    )
    direction_bound = sp.sympify(
        config.get("spatial_covector_component_abs", "1")
    )
    if not (_positive(radius) and direction_bound >= 1):
        raise QuarticQuasilinearMoserError("invalid jet or spatial-direction domain")
    variables = jets + directions
    bounds = {
        **{symbol: radius for symbol in jets},
        **{symbol: direction_bound for symbol in directions},
    }

    raw_matrices = {
        name: data["first_order"][name].subs(substitutions).applyfunc(sp.expand)
        for name in ("A", "B", "C")
    }
    degrees = {
        name: _maximum_jet_degree(matrix, jets)
        for name, matrix in raw_matrices.items()
    }
    if any(degree > declared_degree for degree in degrees.values()):
        raise QuarticQuasilinearMoserError(
            "an extracted raw coefficient exceeds the declared quadratic jet degree"
        )

    raw_bounds: dict[str, list[sp.Expr]] = {}
    nonzero_derivative_components: dict[str, dict[str, int]] = {}
    for name, matrix in raw_matrices.items():
        hierarchy: list[sp.Expr] = []
        counts: dict[str, int] = {}
        for order in range(required_order + 1):
            if order > declared_degree:
                bound, count = sp.Integer(0), 0
            else:
                bound, count = _matrix_derivative_tensor_bound(
                    matrix, order, jets, variables, bounds
                )
            hierarchy.append(sp.factor(bound))
            counts[str(order)] = count
        raw_bounds[name] = hierarchy
        nonzero_derivative_components[name] = counts
    if any(raw_bounds[name][order] != 0 for name in raw_bounds for order in range(3, required_order + 1)):
        raise QuarticQuasilinearMoserError("raw quadratic coefficient has a nonzero third derivative")

    certified_inverse_a_upper = sp.sympify(
        symmetrizer_candidate["uniform_matrix_bounds"]["candidate_A_inverse_2_upper"]
    )
    inverse_a_upper = sp.sympify(config["inverse_time_block_2_norm_ceiling"])
    if not (
        _positive(certified_inverse_a_upper)
        and _positive(inverse_a_upper - certified_inverse_a_upper)
    ):
        raise QuarticQuasilinearMoserError("inverse time-block bound is not positive")
    b_product = _inverse_product_bounds(
        inverse_a_upper, raw_bounds["A"], raw_bounds["B"], required_order
    )
    c_product = _inverse_product_bounds(
        inverse_a_upper, raw_bounds["A"], raw_bounds["C"], required_order
    )
    # ||[P Q; I 0]||_2 <= ||P||_2+||Q||_2+||I||_2. The identity block has
    # norm one; four is retained as an explicit conservative rational ceiling at order zero.
    companion_bounds = [
        sp.factor(
            b_product[order]
            + c_product[order]
            + (sp.Integer(4) if order == 0 else 0)
        )
        for order in range(required_order + 1)
    ]
    if not all(_positive(value) for value in companion_bounds):
        raise QuarticQuasilinearMoserError("companion derivative hierarchy is not positive")

    final_energy_radius = auxiliary_time_candidate["chained_energy_tube"][
        "final_initial_E_s_strict_upper"
    ]
    final_energy_radius_numeric = float(
        auxiliary_time_candidate["chained_energy_tube"][
            "final_initial_E_s_strict_upper_numeric"
        ]
    )
    if final_energy_radius_numeric <= 0:
        raise QuarticQuasilinearMoserError("upstream chained energy radius is not positive")

    return {
        "schema_version": "sigma-quartic-quasilinear-moser-certificate-1.0",
        "status": "pass_quasilinear_coefficient_derivative_envelopes",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "domain": {
            "normalized_local_jet_component_abs": str(radius),
            "covariant_jet_components": len(jets),
            "jet_partition": {
                "nabla_phi": 4,
                "symmetric_nabla_nabla_phi": 10,
                "symmetric_Einstein_tensor": 10,
            },
            "spatial_covector_component_abs": str(direction_bound),
            "required_sobolev_order": required_order,
        },
        "raw_coefficient_degree": degrees,
        "raw_Frechet_derivative_2_norm_envelopes": {
            name: {str(order): str(value) for order, value in enumerate(values)}
            for name, values in raw_bounds.items()
        },
        "raw_Frechet_derivative_2_norm_envelopes_numeric": {
            name: {
                str(order): float(sp.N(value, 18))
                for order, value in enumerate(values)
            }
            for name, values in raw_bounds.items()
        },
        "nonzero_ordered_derivative_tensor_components": nonzero_derivative_components,
        "inverse_time_block_2_norm_certified_upper": str(certified_inverse_a_upper),
        "inverse_time_block_2_norm_rational_ceiling": str(inverse_a_upper),
        "companion_Frechet_derivative_2_norm_envelopes": {
            str(order): str(value) for order, value in enumerate(companion_bounds)
        },
        "companion_Frechet_derivative_2_norm_envelopes_numeric": {
            str(order): float(sp.N(value, 18))
            for order, value in enumerate(companion_bounds)
        },
        "upstream_linear_tube": {
            "final_initial_E_s_strict_upper": final_energy_radius,
            "final_initial_E_s_strict_upper_numeric": final_energy_radius_numeric,
            "binding": (
                "The coefficient envelopes hold throughout the symmetrizer jet box; the "
                "upstream number is recorded but is not promoted to a nonlinear invariant radius."
            ),
        },
        "generic_inverse_product_identity_passed": generic_passed,
        "claim": (
            "The action-derived 22-variable companion coefficient is uniformly C4 in all 24 "
            "declared covariant background-jet components on the certified hyperbolicity box, "
            "with explicit candidate-specific derivative envelopes."
        ),
        "scope": (
            "This supplies the coefficient-composition half of a Sobolev/Moser estimate. It does "
            "not reconstruct the nonlinear state-to-covariant-jet map, bound nonlinear source "
            "terms or symmetrizer derivatives, control modified-harmonic gauge variables, close "
            "the commuted energy inequality, or prove bootstrap/PDE evolution invariance."
        ),
    }


def run_quartic_quasilinear_moser_campaign(
    symmetrizer_campaign: dict[str, Any],
    auxiliary_time_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticQuasilinearMoserError("unsupported campaign schema_version")
        if symmetrizer_campaign.get("status") != (
            "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes"
        ):
            raise QuarticQuasilinearMoserError("symmetrizer campaign prerequisite failed")
        if auxiliary_time_campaign.get("status") != (
            "pass_all_12_linear_auxiliary_time_reconstructions"
        ):
            raise QuarticQuasilinearMoserError("auxiliary-time campaign prerequisite failed")
        expected = int(config.get("expected_candidate_count", 12))
        symmetrizers = {
            item["candidate_id"]: item
            for item in symmetrizer_campaign.get("certificates", [])
        }
        auxiliaries = {
            item["candidate_id"]: item
            for item in auxiliary_time_campaign.get("certificates", [])
        }
        if len(symmetrizers) != expected or set(symmetrizers) != set(auxiliaries):
            raise QuarticQuasilinearMoserError("campaign candidate sets do not match")
        certificates = [
            certify_quartic_quasilinear_moser_candidate(
                symmetrizers[candidate_id], auxiliaries[candidate_id], config
            )
            for candidate_id in sorted(symmetrizers)
        ]
        first_candidate_id = min(symmetrizers)
        passed_count = sum(
            item["status"] == "pass_quasilinear_coefficient_derivative_envelopes"
            for item in certificates
        )

        bad_degree_config = dict(config)
        bad_degree_config["declared_raw_coefficient_jet_degree"] = 1
        degree_rejected = False
        degree_error = ""
        try:
            certify_quartic_quasilinear_moser_candidate(
                symmetrizers[first_candidate_id],
                auxiliaries[first_candidate_id],
                bad_degree_config,
            )
        except QuarticQuasilinearMoserError as error:
            degree_rejected = True
            degree_error = str(error)
        bad_order_config = dict(config)
        bad_order_config["required_sobolev_order"] = 3
        order_rejected = False
        order_error = ""
        try:
            certify_quartic_quasilinear_moser_candidate(
                symmetrizers[first_candidate_id],
                auxiliaries[first_candidate_id],
                bad_order_config,
            )
        except QuarticQuasilinearMoserError as error:
            order_rejected = True
            order_error = str(error)
        if not (degree_rejected and order_rejected):
            raise QuarticQuasilinearMoserError("a campaign negative control did not reject")

        identity_passed, identity = generic_inverse_product_derivative_control()
        if not identity_passed:
            raise QuarticQuasilinearMoserError("generic inverse-product control failed")
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_quasilinear_coefficient_derivative_envelopes"
            if passed_count == expected
            else "reject",
            "errors": [],
            "symmetrizer_campaign_sha256": symmetrizer_campaign.get("content_sha256"),
            "auxiliary_time_campaign_sha256": auxiliary_time_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "counts": {
                "selected": len(certificates),
                "quasilinear_coefficient_envelopes_passed": passed_count,
                "rejected": len(certificates) - passed_count,
            },
            "generic_inverse_product_control": identity,
            "certificates": certificates,
            "negative_controls": {
                "false_linear_raw_coefficient_declaration": {
                    "declared_degree": 1,
                    "rejected": degree_rejected,
                    "error": degree_error,
                },
                "insufficient_sobolev_order": {
                    "declared_order": 3,
                    "rejected": order_rejected,
                    "error": order_error,
                },
            },
            "claim": (
                "All 12 fixed-coefficient linear-X quartic candidates have explicit uniform C4 "
                "derivative envelopes for their action-derived 22-variable quasilinear companion "
                "coefficients on the certified strong-hyperbolicity boxes."
            ),
            "scope": (
                "This closes raw coefficient differentiation and inverse-time-block composition, "
                "not the nonlinear state-to-jet/source Moser estimate or PDE bootstrap. Gauge "
                "reconstruction, commuted symmetrizer energy, boundary energy, and evolution "
                "invariance remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticQuasilinearMoserError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "symmetrizer_campaign_sha256": symmetrizer_campaign.get("content_sha256"),
            "auxiliary_time_campaign_sha256": auxiliary_time_campaign.get("content_sha256"),
            "counts": {
                "selected": 0,
                "quasilinear_coefficient_envelopes_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_quasilinear_moser_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
