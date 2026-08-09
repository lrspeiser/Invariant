from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-coordinate-jet-tube-campaign-1.0"


class QuarticCoordinateJetTubeError(ValueError):
    """Raised when a coordinate-state tube cannot be placed inside the covariant box."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


def _nonnegative(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_nonnegative
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) >= 0)


def _geometry_majorants(radius: sp.Symbol) -> dict[str, sp.Expr]:
    """Return componentwise absolute majorants for the acceleration-free coordinate 2-jet."""

    root_ten = sp.sqrt(10)
    inverse = 1 / (1 - root_ten * radius)
    inverse_deviation = root_ten * radius * inverse
    # Each row of a four-dimensional matrix has l1 norm at most twice its 2-norm.
    connection = 3 * inverse * radius
    inverse_first_row_l1 = 2 * root_ten * inverse**2 * radius
    connection_first = sp.factor(
        (3 * radius * inverse_first_row_l1 + 6 * inverse * radius) / 2
    )
    scalar_hessian = sp.factor(radius + 4 * connection * radius)
    riemann_up = sp.factor(2 * connection_first + 8 * connection**2)
    ricci_lower = sp.factor(4 * riemann_up)
    scalar_curvature = sp.factor(8 * inverse * ricci_lower)
    einstein_lower = sp.factor(
        ricci_lower + (1 + radius) * scalar_curvature / 2
    )
    einstein_upper = sp.factor(4 * inverse**2 * einstein_lower)
    return {
        "metric_deviation_F": root_ten * radius,
        "inverse_metric_2": inverse,
        "inverse_metric_deviation_2": inverse_deviation,
        "inverse_metric_first_row_l1": inverse_first_row_l1,
        "connection_component": connection,
        "connection_first_component": connection_first,
        "scalar_gradient_component": radius,
        "scalar_hessian_component": scalar_hessian,
        "riemann_up_component": riemann_up,
        "ricci_lower_component": ricci_lower,
        "scalar_curvature_abs": scalar_curvature,
        "einstein_lower_component": einstein_lower,
        "einstein_upper_component": einstein_upper,
    }


def _derivative_hierarchy(
    expression: sp.Expr, radius: sp.Symbol, value: sp.Expr, order: int
) -> dict[str, dict[str, float | str]]:
    hierarchy: dict[str, dict[str, float | str]] = {}
    for derivative_order in range(order + 1):
        derivative = sp.factor(sp.diff(expression, radius, derivative_order).subs(radius, value))
        if not _nonnegative(derivative):
            raise QuarticCoordinateJetTubeError("a geometry majorant derivative is negative")
        hierarchy[str(derivative_order)] = {
            "exact": str(derivative),
            "numeric": float(sp.N(derivative, 18)),
        }
    return hierarchy


@cache
def generic_coordinate_jet_majorant_control() -> tuple[bool, dict[str, Any]]:
    """Prove the orthonormal metric basis and a nonzero coordinate-to-covariant tube."""

    components = sp.symbols("h_0:10", real=True, finite=True)
    pairs = tuple((left, right) for left in range(4) for right in range(left, 4))
    perturbation = sp.zeros(4)
    for component, (left, right) in zip(components, pairs, strict=True):
        value = component if left == right else component / sp.sqrt(2)
        perturbation[left, right] = value
        perturbation[right, left] = value
    basis_residual = sp.factor(
        sum(value**2 for value in perturbation) - sum(value**2 for value in components)
    )

    radius = sp.Symbol("rho", positive=True, finite=True)
    majorants = _geometry_majorants(radius)
    coordinate_radius = sp.Rational(1, 10**13)
    target_radius = sp.Rational(1, 5_000_000_000)
    substituted = {
        name: sp.factor(expression.subs(radius, coordinate_radius))
        for name, expression in majorants.items()
    }
    covariant_outputs = {
        "nabla_phi": substituted["scalar_gradient_component"],
        "nabla_nabla_phi": substituted["scalar_hessian_component"],
        "einstein_upper": substituted["einstein_upper_component"],
    }
    margins = {
        name: sp.factor(target_radius - value)
        for name, value in covariant_outputs.items()
    }
    derivative_order = 4
    derivatives = {
        name: _derivative_hierarchy(expression, radius, coordinate_radius, derivative_order)
        for name, expression in majorants.items()
    }
    rejected_radius = sp.Rational(1, 10**12)
    rejected_einstein = sp.factor(
        majorants["einstein_upper_component"].subs(radius, rejected_radius)
    )
    rejected_margin = sp.factor(target_radius - rejected_einstein)
    passed = bool(
        basis_residual == 0
        and _positive(1 - substituted["metric_deviation_F"])
        and all(_positive(margin) for margin in margins.values())
        and not _positive(rejected_margin)
    )
    return passed, {
        "control": "uniform coordinate-state 2-jet to covariant-jet majorant theorem",
        "coordinate_atom_norm": (
            "maximum absolute normalized q_A metric deviation, first partial, or "
            "acceleration-free symmetric second partial"
        ),
        "bounded_coordinate_atoms": {
            "metric_deviation_components": 10,
            "field_first_partial_components": 44,
            "acceleration_free_symmetric_second_partial_components": 99,
            "total": 153,
            "scalar_field_value": "unrestricted because this action is shift symmetric",
        },
        "orthonormal_symmetric_metric_basis_residual": str(basis_residual),
        "coordinate_component_radius": str(coordinate_radius),
        "target_covariant_component_radius": str(target_radius),
        "majorants_at_coordinate_radius": {
            name: {
                "exact": str(value),
                "numeric": float(sp.N(value, 18)),
            }
            for name, value in substituted.items()
        },
        "covariant_hyperbolicity_components": {
            name: {
                "upper": str(value),
                "upper_numeric": float(sp.N(value, 18)),
                "strict_margin": str(margins[name]),
                "strict_margin_numeric": float(sp.N(margins[name], 18)),
            }
            for name, value in covariant_outputs.items()
        },
        "Frechet_majorant_derivatives": {
            "input_norm": "component l_infinity",
            "output_norm": "component l_infinity",
            "orders": list(range(derivative_order + 1)),
            "interpretation": (
                "The positive-coefficient radial majorants bound the ordered Frechet "
                "derivative tensors when every one of the 153 coordinate atoms is bounded "
                "by the common radius."
            ),
            "families": derivatives,
        },
        "majorant_derivation": {
            "inverse_metric": "Neumann series with ||eta h||_2<=sqrt(10) rho",
            "connection": "row-l1 inverse bound times the three-term Christoffel bracket",
            "connection_first": "inverse-derivative product plus coordinate second partials",
            "curvature": "two connection derivatives plus eight connection-square products",
            "einstein_upper": "four Ricci contractions, scalar trace, then two inverse metrics",
        },
        "negative_control": {
            "coordinate_component_radius": str(rejected_radius),
            "einstein_upper_majorant": str(rejected_einstein),
            "target_margin": str(rejected_margin),
            "rejected": not _positive(rejected_margin),
        },
        "passed": passed,
        "scope": (
            "This bounds the acceleration-free coordinate 2-jet map uniformly. It does not "
            "yet bound the Euler remainder, commuted gauge/constraint source, a lifespan, "
            "or preservation of the coordinate tube under evolution."
        ),
    }


def _certify_candidate(
    prerequisite: dict[str, Any], config: dict[str, Any], control: dict[str, Any]
) -> dict[str, Any]:
    if prerequisite.get("status") != (
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift"
    ):
        raise QuarticCoordinateJetTubeError("candidate lacks the full-state PDE prerequisite")
    coordinate_radius = sp.sympify(config["coordinate_component_radius"])
    target_radius = sp.sympify(
        prerequisite["domain"]["normalized_local_jet_component_abs"]
    )
    if coordinate_radius != sp.sympify(control["coordinate_component_radius"]):
        raise QuarticCoordinateJetTubeError("coordinate radius differs from the proved majorant")
    if target_radius != sp.sympify(control["target_covariant_component_radius"]):
        raise QuarticCoordinateJetTubeError("candidate covariant radius differs from the target")
    margins = control["covariant_hyperbolicity_components"]
    if not all(_positive(sp.sympify(item["strict_margin"])) for item in margins.values()):
        raise QuarticCoordinateJetTubeError("coordinate tube does not fit the covariant box")
    return {
        "schema_version": "sigma-quartic-coordinate-jet-tube-certificate-1.0",
        "status": "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "candidate_id": prerequisite["candidate_id"],
        "coefficients": prerequisite["coefficients"],
        "coordinate_component_radius": str(coordinate_radius),
        "covariant_component_radius": str(target_radius),
        "bounded_coordinate_atom_count": control["bounded_coordinate_atoms"]["total"],
        "covariant_component_bounds": margins,
        "Frechet_majorant_order": max(
            control["Frechet_majorant_derivatives"]["orders"]
        ),
        "full_state_symmetrizer_K55_lower": prerequisite["uniform_bounds"][
            "K55_2_lower"
        ],
        "claim": (
            "Every compatible acceleration-free coordinate 2-jet in the declared component "
            "cube maps strictly inside this candidate's covariant hyperbolicity box."
        ),
        "remaining_gate": "Euler_remainder_and_commuted_gauge_source_derivative_envelopes",
        "scope": control["scope"],
    }


def run_quartic_coordinate_jet_tube_campaign(
    nonquasilinear_pde_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticCoordinateJetTubeError("unsupported campaign schema_version")
        if nonquasilinear_pde_campaign.get("status") != (
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts"
        ):
            raise QuarticCoordinateJetTubeError("full-state PDE campaign prerequisite failed")
        required_order = int(config["required_Frechet_majorant_order"])
        if required_order != 4:
            raise QuarticCoordinateJetTubeError("the coordinate tube requires order four")
        control_passed, control = generic_coordinate_jet_majorant_control()
        if not control_passed:
            raise QuarticCoordinateJetTubeError("generic coordinate majorant control failed")
        expected = int(config.get("expected_candidate_count", 12))
        prerequisites = nonquasilinear_pde_campaign.get("certificates", [])
        if len(prerequisites) != expected:
            raise QuarticCoordinateJetTubeError("unexpected candidate count")
        certificates = [
            _certify_candidate(item, config, control)
            for item in sorted(prerequisites, key=lambda value: value["candidate_id"])
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            "errors": [],
            "nonquasilinear_pde_campaign_sha256": nonquasilinear_pde_campaign.get(
                "content_sha256"
            ),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_coordinate_jet_majorant_control": control,
            "counts": {
                "selected": len(certificates),
                "coordinate_jet_tubes_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 quartic candidates have a common nonzero coordinate-state 2-jet "
                "cube that maps strictly inside their certified covariant hyperbolicity box, "
                "with derivative majorants through order four."
            ),
            "scope": control["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticCoordinateJetTubeError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "coordinate_jet_tubes_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_coordinate_jet_tube_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
