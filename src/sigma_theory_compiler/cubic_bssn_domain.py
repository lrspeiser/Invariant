from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import mpmath as mp
import sympy as sp

SCHEMA_VERSION = "sigma-cubic-bssn-domain-certificate-1.0"
_ETA = (-1, 1, 1, 1)


class CubicDomainCertificationError(ValueError):
    """Raised when a declared cubic-Horndeski local-jet box cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _interval(lower: Any, upper: Any | None = None) -> Any:
    return mp.iv.mpf([lower, lower if upper is None else upper])


def _bounds(value: Any) -> tuple[float, float]:
    if hasattr(value, "a") and hasattr(value, "b"):
        return float(value.a), float(value.b)
    numeric = float(value)
    return numeric, numeric


def _maximum_absolute(value: Any) -> float:
    lower, upper = _bounds(value)
    return max(abs(lower), abs(upper))


def _minimum_absolute(value: Any) -> float:
    lower, upper = _bounds(value)
    if lower <= 0 <= upper:
        return 0.0
    return min(abs(lower), abs(upper))


def _record(value: Any) -> dict[str, float]:
    lower, upper = _bounds(value)
    return {"lower": lower, "upper": upper}


def cubic_scalar_effective_metric(
    *,
    x: Any,
    gradient_covariant: list[Any],
    hessian_covariant: list[list[Any]],
    g2_x: Any,
    g2_xx: Any,
    g3_phi: Any,
    g3_x_phi: Any,
    g3_x: Any,
    g3_xx: Any,
) -> list[list[Any]]:
    """Return P'^mu-nu for the source-normalized cubic-Horndeski scalar equation."""

    gradient_contravariant = [
        _ETA[index] * gradient_covariant[index] for index in range(4)
    ]
    gradient_squared = sum(
        _ETA[index] * gradient_covariant[index] ** 2 for index in range(4)
    )
    box_phi = sum(
        _ETA[index] * hessian_covariant[index][index] for index in range(4)
    )
    hessian_contravariant = [
        [
            _ETA[row] * _ETA[column] * hessian_covariant[row][column]
            for column in range(4)
        ]
        for row in range(4)
    ]
    gradient_hessian_gradient = sum(
        gradient_contravariant[row]
        * gradient_contravariant[column]
        * hessian_covariant[row][column]
        for row in range(4)
        for column in range(4)
    )
    hessian_gradient = [
        _ETA[row]
        * sum(
            gradient_contravariant[column] * hessian_covariant[column][row]
            for column in range(4)
        )
        for row in range(4)
    ]
    isotropic = 1 + g2_x + 2 * x * g2_xx + 2 * g3_phi + 2 * x * g3_x_phi
    disformal = g2_xx + 2 * g3_x_phi
    matrix: list[list[Any]] = []
    for row in range(4):
        entries: list[Any] = []
        for column in range(4):
            eta = _ETA[row] if row == column else 0
            gradient_product = (
                gradient_contravariant[row] * gradient_contravariant[column]
            )
            value = isotropic * eta
            value += disformal * (gradient_squared * eta - gradient_product)
            value += 2 * g3_x * (
                box_phi * eta - hessian_contravariant[row][column]
            )
            value -= g3_xx * (
                box_phi * gradient_product
                + gradient_hessian_gradient * eta
                - gradient_contravariant[row] * hessian_gradient[column]
                - hessian_gradient[row] * gradient_contravariant[column]
            )
            # Trace-reversed metric-equation substitution in P'_phi-phi.
            value -= (g3_x**2) * (gradient_squared**2) * eta / 4
            value += (g3_x**2) * gradient_squared * gradient_product
            entries.append(value)
        matrix.append(entries)
    return matrix


def generic_cubic_scalar_effective_metric_control() -> tuple[bool, dict[str, Any]]:
    """Verify the trace-reversed cubic scalar principal metric exactly."""

    x = sp.Symbol("X", real=True)
    g3_x = sp.Symbol("G3_X", real=True)
    p_cov = list(sp.symbols("p0:4", real=True))
    xi_cov = list(sp.symbols("xi0:4", real=True))
    zero_hessian = [[sp.Integer(0) for _ in range(4)] for _ in range(4)]
    effective = cubic_scalar_effective_metric(
        x=x,
        gradient_covariant=p_cov,
        hessian_covariant=zero_hessian,
        g2_x=0,
        g2_xx=0,
        g3_phi=0,
        g3_x_phi=0,
        g3_x=g3_x,
        g3_xx=0,
    )
    p_up = [_ETA[index] * p_cov[index] for index in range(4)]
    xi_up = [_ETA[index] * xi_cov[index] for index in range(4)]
    p_squared = sum(_ETA[index] * p_cov[index] ** 2 for index in range(4))
    xi_squared = sum(_ETA[index] * xi_cov[index] ** 2 for index in range(4))
    xi_dot_p = sum(xi_up[index] * p_cov[index] for index in range(4))
    p4 = p_squared**2
    pg = [
        [
            sp.Rational(1, 2) * g3_x * p_up[row] * p_up[column] * xi_squared
            - g3_x
            * xi_dot_p
            * (
                sp.Rational(1, 2)
                * (xi_up[row] * p_up[column] + xi_up[column] * p_up[row])
                - sp.Rational(1, 2)
                * (_ETA[row] if row == column else 0)
                * xi_dot_p
            )
            for column in range(4)
        ]
        for row in range(4)
    ]
    pg_trace = sp.expand(
        sum(_ETA[index] * pg[index][index] for index in range(4))
    )
    pg_p_p = sp.expand(
        sum(
            p_cov[row] * p_cov[column] * pg[row][column]
            for row in range(4)
            for column in range(4)
        )
    )
    source_trace_reversed_correction = sp.expand(
        -g3_x * (pg_p_p - sp.Rational(1, 2) * p_squared * pg_trace)
    )
    implemented_correction = sp.expand(
        sum(
            xi_cov[row]
            * (
                effective[row][column]
                - (_ETA[row] if row == column else 0)
            )
            * xi_cov[column]
            for row in range(4)
            for column in range(4)
        )
    )
    correction_residual = sp.factor(
        implemented_correction - source_trace_reversed_correction
    )
    closed_correction = sp.expand(
        -(g3_x**2) * p4 * xi_squared / 4
        + (g3_x**2) * p_squared * xi_dot_p**2
    )
    closed_residual = sp.factor(implemented_correction - closed_correction)
    symmetry_residuals = [
        sp.factor(effective[row][column] - effective[column][row])
        for row in range(4)
        for column in range(row)
    ]
    omitted_trace_reversal = sp.factor(
        implemented_correction - sp.expand(-g3_x * pg_p_p)
    )
    witness = {
        p_cov[0]: 2,
        p_cov[1]: 1,
        p_cov[2]: 0,
        p_cov[3]: 0,
        xi_cov[0]: 1,
        xi_cov[1]: 2,
        xi_cov[2]: 0,
        xi_cov[3]: 0,
        g3_x: 1,
    }
    corrupted_witness = sp.factor(omitted_trace_reversal.subs(witness))
    passed = (
        correction_residual == 0
        and closed_residual == 0
        and all(residual == 0 for residual in symmetry_residuals)
        and corrupted_witness != 0
    )
    return passed, {
        "control": "generic cubic-Horndeski trace-reversed scalar effective metric",
        "source": {
            "title": "Well-posedness of cubic Horndeski theories",
            "url": "https://arxiv.org/abs/1904.00963",
            "equation": "P'_phi_phi=P_phi_phi-G3_X trace_reverse(P_gphi) p p",
        },
        "implemented_trace_reversed_correction": str(implemented_correction),
        "closed_correction": str(closed_correction),
        "source_contraction_residual": str(correction_residual),
        "closed_form_residual": str(closed_residual),
        "symmetry_residuals": [str(residual) for residual in symmetry_residuals],
        "negative_control": {
            "corruption": "omit trace reversal in metric-equation substitution",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": bool(corrupted_witness != 0),
        },
        "scope": (
            "Exact algebraic validation of the complete G3_X-squared trace-reversed correction. "
            "The domain certifier adds all G2/G3 derivative and Hessian terms from the source."
        ),
    }


def _compile_candidate_functions(
    ir: dict[str, Any], coefficients: dict[str, Any]
) -> dict[str, Any]:
    u, x = sp.symbols("u x", real=True)
    coefficient_symbols = {
        name: sp.Symbol(name, real=True) for name in coefficients
    }
    locals_map = {"u": u, "x": x, **coefficient_symbols}
    formulation = ir.get("formulation_classification")
    if not isinstance(formulation, dict):
        raise CubicDomainCertificationError("IR has no formulation classification")
    canonical_g2 = sp.sympify(str(formulation.get("canonical_G2")), locals=locals_map)
    canonical_g3 = sp.sympify(str(formulation.get("canonical_G3")), locals=locals_map)
    g4_x = sp.sympify(str(formulation.get("G4_X")), locals=locals_map)
    substitutions = {
        coefficient_symbols[name]: sp.Rational(str(value))
        for name, value in coefficients.items()
    }
    canonical_g2 = sp.factor(canonical_g2.subs(substitutions))
    canonical_g3 = sp.factor(canonical_g3.subs(substitutions))
    g4_x = sp.factor(g4_x.subs(substitutions))
    unresolved = sorted(
        {
            str(symbol)
            for expression in (canonical_g2, canonical_g3, g4_x)
            for symbol in expression.free_symbols
            if symbol not in {u, x}
        }
    )
    if unresolved:
        raise CubicDomainCertificationError(
            "unresolved coefficients: " + ", ".join(unresolved)
        )
    if canonical_g3 == 0 or g4_x != 0:
        raise CubicDomainCertificationError(
            "candidate is not in the canonical-G3-only cubic subclass"
        )

    # Kovacs arXiv:1904.00963 uses R+X+G2+G3*box(phi).  The compiler IR
    # stores the full G2 and -G3*box(phi).
    source_g2 = sp.factor(canonical_g2 - x)
    source_g3 = sp.factor(-canonical_g3)
    expressions = {
        "G2": source_g2,
        "G2_X": sp.diff(source_g2, x),
        "G2_XX": sp.diff(source_g2, x, 2),
        "G3": source_g3,
        "G3_phi": sp.diff(source_g3, u),
        "G3_X_phi": sp.diff(source_g3, x, u),
        "G3_X": sp.diff(source_g3, x),
        "G3_XX": sp.diff(source_g3, x, 2),
    }
    derivative_ledger: dict[str, tuple[sp.Expr, int]] = {}
    for family, parent, phi_limit in (
        ("G2", source_g2, 1),
        ("G3", source_g3, 2),
    ):
        for x_order in range(3):
            for phi_order in range(phi_limit + 1):
                name = f"{family}_X{x_order}_phi{phi_order}"
                exponent = 2 * x_order + 2 if family == "G2" else 2 * x_order
                derivative_ledger[name] = (
                    sp.factor(sp.diff(parent, x, x_order, u, phi_order)),
                    exponent,
                )
    arguments = (u, x)
    return {
        "source_G2": str(source_g2),
        "source_G3": str(source_g3),
        "functions": {
            name: sp.lambdify(arguments, expression, modules=mp.iv)
            for name, expression in expressions.items()
        },
        "function_expressions": {
            name: str(expression) for name, expression in expressions.items()
        },
        "weak_field": {
            name: {
                "function": sp.lambdify(arguments, expression, modules=mp.iv),
                "expression": str(expression),
                "E_exponent": exponent,
            }
            for name, (expression, exponent) in derivative_ledger.items()
        },
    }


def _matrix_bounds(matrix: list[list[Any]]) -> dict[str, Any]:
    spatial_lower = math.inf
    spatial_upper = -math.inf
    for row in range(1, 4):
        diagonal_lower, diagonal_upper = _bounds(matrix[row][row])
        off_diagonal_radius = sum(
            _maximum_absolute(matrix[row][column])
            for column in range(1, 4)
            if column != row
        )
        spatial_lower = min(spatial_lower, diagonal_lower - off_diagonal_radius)
        spatial_upper = max(spatial_upper, diagonal_upper + off_diagonal_radius)
    time_space_norm_upper = math.sqrt(
        sum(_maximum_absolute(matrix[0][column]) ** 2 for column in range(1, 4))
    )
    return {
        "P00": _record(matrix[0][0]),
        "spatial_eigenvalue_lower_Gershgorin": spatial_lower,
        "spatial_eigenvalue_upper_Gershgorin": spatial_upper,
        "time_space_vector_norm_upper": time_space_norm_upper,
        "components": [
            [_record(matrix[row][column]) for column in range(4)] for row in range(4)
        ],
    }


def certify_cubic_bssn_domain(
    ir: dict[str, Any],
    trajectory_certificate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Certify one arbitrary-local-jet box for a cubic-Horndeski candidate."""

    errors: list[str] = []
    try:
        if config.get("schema_version") != "sigma-cubic-bssn-domain-run-1.0":
            raise CubicDomainCertificationError("unsupported cubic BSSN domain schema")
        if trajectory_certificate.get("status") != "pass_interval_certified":
            raise CubicDomainCertificationError("trajectory certificate is not a pass")
        precision = int(config.get("precision_digits", 60))
        if precision < 30:
            raise CubicDomainCertificationError("precision_digits must be at least 30")
        mp.iv.dps = precision
        coefficients = trajectory_certificate.get("coefficients")
        if not isinstance(coefficients, dict):
            raise CubicDomainCertificationError("trajectory has no coefficient binding")
        compiled = _compile_candidate_functions(ir, coefficients)
        hull = trajectory_certificate.get("trajectory_hull")
        if not isinstance(hull, dict):
            raise CubicDomainCertificationError("trajectory has no interval hull")
        extension = config.get("domain_extension")
        if not isinstance(extension, dict):
            raise CubicDomainCertificationError("domain_extension must be an object")
        phi_padding = float(extension.get("phi_padding", 0.0))
        p0_padding = float(extension.get("normal_gradient_padding", 0.0))
        spatial_gradient_abs = float(extension["spatial_gradient_abs"])
        hessian_abs = float(extension["hessian_component_abs"])
        curvature_abs = float(extension["riemann_component_abs"])
        if min(spatial_gradient_abs, hessian_abs, curvature_abs) <= 0:
            raise CubicDomainCertificationError("domain component bounds must be positive")
        u_lower = float(hull["u"]["lower"]) - phi_padding
        u_upper = float(hull["u"]["upper"]) + phi_padding
        x_lower = float(hull["x"]["lower"])
        x_upper = float(hull["x"]["upper"])
        p0_lower = max(0.0, math.sqrt(2 * x_lower) - p0_padding)
        p0_upper = math.sqrt(2 * x_upper) + p0_padding
        gradient_covariant = [
            _interval(p0_lower, p0_upper),
            *[_interval(-spatial_gradient_abs, spatial_gradient_abs) for _ in range(3)],
        ]
        hessian_covariant = [
            [_interval(-hessian_abs, hessian_abs) for _ in range(4)]
            for _ in range(4)
        ]
        for row in range(4):
            for column in range(row):
                hessian_covariant[row][column] = hessian_covariant[column][row]
        x = (
            gradient_covariant[0] ** 2
            - sum(component**2 for component in gradient_covariant[1:])
        ) / 2
        derived_x_lower, _ = _bounds(x)
        if derived_x_lower <= 0:
            raise CubicDomainCertificationError(
                "declared gradient box crosses the non-timelike X<=0 surface"
            )
        phi = _interval(u_lower, u_upper)
        functions = {
            name: function(phi, x)
            for name, function in compiled["functions"].items()
        }
        matrix = cubic_scalar_effective_metric(
            x=x,
            gradient_covariant=gradient_covariant,
            hessian_covariant=hessian_covariant,
            g2_x=functions["G2_X"],
            g2_xx=functions["G2_XX"],
            g3_phi=functions["G3_phi"],
            g3_x_phi=functions["G3_X_phi"],
            g3_x=functions["G3_X"],
            g3_xx=functions["G3_XX"],
        )
        matrix_bounds = _matrix_bounds(matrix)
        time_upper = matrix_bounds["P00"]["upper"]
        spatial_lower = matrix_bounds["spatial_eigenvalue_lower_Gershgorin"]
        time_margin = float(config.get("time_covector_margin", 1e-6))
        spatial_margin = float(config.get("spatial_block_margin", 1e-6))
        cone_margin = float(config.get("cone_separation_margin", 1e-6))
        sigma = float(config.get("slicing_parameter_sigma", 1.0))
        momentum_parameter = float(config.get("momentum_parameter_m", 1.0))
        if sigma <= 0.5 or momentum_parameter <= 0.25:
            raise CubicDomainCertificationError(
                "cubic BSSN requires sigma>1/2 and m>1/4"
            )
        if time_upper >= -time_margin:
            raise CubicDomainCertificationError(
                "effective scalar time coefficient does not stay negative"
            )
        if spatial_lower <= spatial_margin:
            raise CubicDomainCertificationError(
                "effective scalar spatial block is not uniformly positive"
            )
        gauge_speed = math.sqrt(2 * sigma)
        cone_polynomial_upper = (
            2 * sigma * time_upper
            + matrix_bounds["spatial_eigenvalue_upper_Gershgorin"]
            + 2
            * gauge_speed
            * matrix_bounds["time_space_vector_norm_upper"]
        )
        if cone_polynomial_upper >= -cone_margin:
            raise CubicDomainCertificationError(
                "BSSN slicing cone is not uniformly separated from the scalar cone"
            )
        discriminant_lower = -time_upper * spatial_lower
        if discriminant_lower <= 0:
            raise CubicDomainCertificationError(
                "scalar characteristic discriminant lower bound is non-positive"
            )
        weak_field_scale = max(
            math.sqrt(curvature_abs),
            max(_maximum_absolute(component) for component in gradient_covariant),
            math.sqrt(hessian_abs),
        )
        weak_field_ratios = {
            name: _maximum_absolute(item["function"](phi, x))
            * weak_field_scale ** item["E_exponent"]
            for name, item in compiled["weak_field"].items()
        }
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_uniform_local_jet_box",
            "errors": [],
            "source": {
                "title": "Well-posedness of cubic Horndeski theories",
                "url": "https://arxiv.org/abs/1904.00963",
                "equations": "source-normalized P'_phi_phi and BSSN conditions",
            },
            "source_ir_sha256": ir.get("content_sha256"),
            "trajectory_certificate_sha256": trajectory_certificate.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "coefficients": coefficients,
            "normalization_binding": {
                "source_G2": compiled["source_G2"],
                "source_G3": compiled["source_G3"],
                "explanation": "source action R+X+G2+G3 box(phi); compiler stores full G2 and -G3 box(phi)",
            },
            "domain": {
                "phi": {"lower": u_lower, "upper": u_upper},
                "gradient_covariant": [_record(value) for value in gradient_covariant],
                "derived_X": _record(x),
                "symmetric_hessian_component_abs": hessian_abs,
                "riemann_component_abs": curvature_abs,
                "frame": (
                    "local orthonormal tetrad with timelike leg equal to the declared BSSN "
                    "foliation normal; every symmetric Hessian component varies independently"
                ),
                "rotation_invariant_subdomains_contained": {
                    "spatial_gradient_norm": f"<= {spatial_gradient_abs}",
                    "tetrad_hessian_Frobenius_norm": f"<= {hessian_abs}",
                    "tetrad_Riemann_Frobenius_norm": f"<= {curvature_abs}",
                    "explanation": (
                        "Each norm ball is a subset of the independently bounded component cube, "
                        "so the interval proof covers it for every spatial tetrad rotation."
                    ),
                },
            },
            "BSSN_parameters": {"m": momentum_parameter, "sigma": sigma},
            "effective_metric": matrix_bounds,
            "uniform_proof": {
                "common_time_covector_upper_P00": time_upper,
                "spatial_block_eigenvalue_lower": spatial_lower,
                "characteristic_discriminant_lower": discriminant_lower,
                "slicing_cone_polynomial_upper": cone_polynomial_upper,
                "time_margin": time_margin,
                "spatial_margin": spatial_margin,
                "cone_margin": cone_margin,
                "direction_sphere_method": (
                    "Gershgorin bounds for every unit spatial covector plus the Euclidean "
                    "time-space block norm; no direction sampling"
                ),
            },
            "weak_field_diagnostic": {
                "E_upper_bound": weak_field_scale,
                "derivative_ratio_upper_bounds": weak_field_ratios,
                "maximum_derivative_ratio": max(weak_field_ratios.values(), default=0.0),
                "interpretation": (
                    "The source's qualitative much-less-than-one condition has no universal "
                    "numeric cutoff. The direct effective-metric and cone proof above supplies "
                    "the executable domain verdict; these ratios remain diagnostics."
                ),
            },
            "scope": (
                "Uniform pointwise principal-symbol certificate over the declared local-jet box "
                "and every spatial covector direction. It is not a PDE evolution invariant proving "
                "that arbitrary initial data remain inside the box, nor a global energy theorem."
            ),
        }
    except (CubicDomainCertificationError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "trajectory_certificate_sha256": trajectory_certificate.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def run_cubic_bssn_domain_campaign(
    ir: dict[str, Any],
    flrw_campaign: dict[str, Any],
    trajectory_certificates: dict[str, dict[str, Any]],
    campaign_config: dict[str, Any],
) -> dict[str, Any]:
    """Find nested sufficient local-jet boxes for all screened cubic candidates."""

    errors: list[str] = []
    output_certificates: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    try:
        if (
            campaign_config.get("schema_version")
            != "sigma-cubic-bssn-domain-campaign-1.0"
        ):
            raise CubicDomainCertificationError(
                "unsupported cubic BSSN domain campaign schema"
            )
        template = campaign_config.get("domain_template")
        if not isinstance(template, dict):
            raise CubicDomainCertificationError("domain_template must be an object")
        search = campaign_config.get("hessian_radius_search")
        if not isinstance(search, dict):
            raise CubicDomainCertificationError(
                "hessian_radius_search must be an object"
            )
        minimum = float(search["minimum"])
        maximum = float(search["maximum"])
        iterations = int(search.get("bisection_iterations", 12))
        if not 0 < minimum < maximum or iterations < 1:
            raise CubicDomainCertificationError(
                "require 0<minimum<maximum and positive bisection iterations"
            )
        candidates = [
            record
            for record in flrw_campaign.get("candidates", [])
            if record.get("status")
            == "pass_flrw_interval_cubic_weak_field_bounds_unresolved"
        ]
        if not candidates:
            raise CubicDomainCertificationError(
                "FLRW campaign has no screened cubic candidates"
            )
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            trajectory = trajectory_certificates.get(candidate_id)
            record: dict[str, Any] = {
                "candidate_id": candidate_id,
                "mutation_assignment": candidate["mutation_assignment"],
            }
            if not isinstance(trajectory, dict):
                record.update(
                    {
                        "status": "reject",
                        "errors": ["missing trajectory certificate"],
                    }
                )
                records.append(record)
                continue

            def evaluate(
                radius: float, trajectory: dict[str, Any] = trajectory
            ) -> dict[str, Any]:
                config = copy.deepcopy(template)
                config["domain_extension"]["hessian_component_abs"] = radius
                return certify_cubic_bssn_domain(ir, trajectory, config)

            lower_certificate = evaluate(minimum)
            if lower_certificate["status"] != "pass_uniform_local_jet_box":
                record.update(
                    {
                        "status": "reject_no_certified_minimum_box",
                        "errors": lower_certificate["errors"],
                    }
                )
                records.append(record)
                continue
            upper_certificate = evaluate(maximum)
            if upper_certificate["status"] == "pass_uniform_local_jet_box":
                certified_lower = maximum
                failing_upper: float | None = None
                final_certificate = upper_certificate
            else:
                certified_lower = minimum
                failing_upper = maximum
                final_certificate = lower_certificate
                for _ in range(iterations):
                    midpoint = (certified_lower + failing_upper) / 2
                    midpoint_certificate = evaluate(midpoint)
                    if midpoint_certificate["status"] == "pass_uniform_local_jet_box":
                        certified_lower = midpoint
                        final_certificate = midpoint_certificate
                    else:
                        failing_upper = midpoint
            output_certificates[candidate_id] = final_certificate
            record.update(
                {
                    "status": "pass_uniform_local_jet_box",
                    "certified_hessian_component_radius_lower": certified_lower,
                    "first_failing_hessian_component_radius_upper": failing_upper,
                    "bisection_iterations": iterations if failing_upper is not None else 0,
                    "certificate": f"certificates/{candidate_id}.json",
                    "certificate_sha256": final_certificate["content_sha256"],
                    "uniform_proof": final_certificate["uniform_proof"],
                    "weak_field_diagnostic": final_certificate[
                        "weak_field_diagnostic"
                    ],
                }
            )
            records.append(record)
        ranking = sorted(
            [record for record in records if record["status"] == "pass_uniform_local_jet_box"],
            key=lambda record: (
                -record["certified_hessian_component_radius_lower"],
                record["weak_field_diagnostic"]["maximum_derivative_ratio"],
            ),
        )
        counts = {
            "screened_cubic_candidates": len(candidates),
            "uniform_domain_certified": len(ranking),
            "rejected": sum(
                not record["status"].startswith("pass") for record in records
            ),
        }
        body = {
            "schema_version": "sigma-cubic-bssn-domain-campaign-1.0",
            "status": (
                "pass_all_screened_cubic_candidates_have_uniform_local_jet_boxes"
                if counts["uniform_domain_certified"] == counts["screened_cubic_candidates"]
                else "incomplete_or_reject"
            ),
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "flrw_campaign_sha256": flrw_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(
                _canonical_json(campaign_config).encode()
            ).hexdigest(),
            "counts": counts,
            "ranking": [
                {
                    "candidate_id": record["candidate_id"],
                    "mutation_assignment": record["mutation_assignment"],
                    "certified_hessian_component_radius_lower": record[
                        "certified_hessian_component_radius_lower"
                    ],
                    "first_failing_hessian_component_radius_upper": record[
                        "first_failing_hessian_component_radius_upper"
                    ],
                    "slicing_cone_polynomial_upper": record["uniform_proof"][
                        "slicing_cone_polynomial_upper"
                    ],
                    "spatial_block_eigenvalue_lower": record["uniform_proof"][
                        "spatial_block_eigenvalue_lower"
                    ],
                    "maximum_derivative_ratio": record["weak_field_diagnostic"][
                        "maximum_derivative_ratio"
                    ],
                }
                for record in ranking
            ],
            "candidates": records,
            "claim": (
                "Every screened cubic candidate has a sufficient pointwise strong-hyperbolicity "
                "certificate over a declared arbitrary local-jet box and every spatial direction."
            ),
            "scope": (
                "The boxes are nested sufficient principal-symbol domains anchored to each FLRW "
                "trajectory hull. This does not prove nonlinear PDE evolution keeps data inside "
                "the box or establish global Hamiltonian energy positivity."
            ),
        }
    except (CubicDomainCertificationError, KeyError, TypeError, ValueError) as error:
        errors.append(str(error))
        body = {
            "schema_version": "sigma-cubic-bssn-domain-campaign-1.0",
            "status": "reject",
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "counts": {
                "screened_cubic_candidates": 0,
                "uniform_domain_certified": 0,
                "rejected": 0,
            },
            "ranking": [],
            "candidates": [],
        }
    manifest = {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }
    return {"manifest": manifest, "certificates": output_certificates}


def write_cubic_bssn_domain_campaign(
    result: dict[str, Any], output: Path
) -> tuple[Path, list[Path]]:
    output.mkdir(parents=True, exist_ok=True)
    certificate_directory = output / "certificates"
    certificate_directory.mkdir(parents=True, exist_ok=True)
    certificate_paths: list[Path] = []
    for candidate_id, certificate in sorted(result["certificates"].items()):
        path = certificate_directory / f"{candidate_id}.json"
        path.write_text(
            json.dumps(certificate, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        certificate_paths.append(path)
    manifest_path = output / "campaign.json"
    manifest_path.write_text(
        json.dumps(result["manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, certificate_paths
