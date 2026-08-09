from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import mpmath as mp
import sympy as sp

SCHEMA_VERSION = "sigma-flrw-background-certificate-1.0"
_STATE_NAMES = ("u", "x", "h")
_DERIVATIVE_NAMES = ("h_tau", "x_tau")


class BackgroundCertificationError(ValueError):
    """Raised when a background cannot be certified on the declared patch."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _bounds(value: Any) -> tuple[float, float]:
    if hasattr(value, "a") and hasattr(value, "b"):
        lower = float(value.a)
        upper = float(value.b)
    else:
        lower = float(value)
        upper = lower
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise BackgroundCertificationError("non-finite interval encountered")
    return lower, upper


def _interval(lower: Any, upper: Any | None = None) -> Any:
    high = lower if upper is None else upper
    if lower > high:
        lower, high = high, lower
    return mp.iv.mpf([lower, high])


def _interval_record(value: Any) -> dict[str, float]:
    lower, upper = _bounds(value)
    return {"lower": lower, "upper": upper}


def _contains_zero(value: Any) -> bool:
    lower, upper = _bounds(value)
    return lower <= 0 <= upper


def _minimum_absolute(value: Any) -> float:
    lower, upper = _bounds(value)
    if lower <= 0 <= upper:
        return 0.0
    return min(abs(lower), abs(upper))


def _maximum_absolute(value: Any) -> float:
    lower, upper = _bounds(value)
    return max(abs(lower), abs(upper))


def _hull(left: Any, right: Any) -> Any:
    left_lower, left_upper = _bounds(left)
    right_lower, right_upper = _bounds(right)
    return _interval(min(left_lower, right_lower), max(left_upper, right_upper))


def _inflate(value: Any, absolute: float, relative: float) -> Any:
    lower, upper = _bounds(value)
    width = upper - lower
    padding = absolute + relative * max(width, absolute)
    return _interval(lower - padding, upper + padding)


def _subset(inner: Any, outer: Any) -> bool:
    inner_lower, inner_upper = _bounds(inner)
    outer_lower, outer_upper = _bounds(outer)
    return inner_lower >= outer_lower and inner_upper <= outer_upper


def _parse_expression(raw: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    locals_map: dict[str, Any] = {**symbols, "sqrt": sp.sqrt, "Matrix": sp.Matrix}
    expression = sp.sympify(raw, locals=locals_map)
    if not isinstance(expression, sp.Expr):
        raise BackgroundCertificationError("expected a scalar symbolic expression")
    return expression


def _parse_matrix(raw: str, symbols: dict[str, sp.Symbol]) -> sp.Matrix:
    locals_map: dict[str, Any] = {**symbols, "sqrt": sp.sqrt, "Matrix": sp.Matrix}
    expression = sp.sympify(raw, locals=locals_map)
    if not isinstance(expression, sp.MatrixBase):
        raise BackgroundCertificationError("expected a symbolic matrix expression")
    return sp.Matrix(expression)


def _compile_bundle(ir: dict[str, Any], coefficients: dict[str, Any]) -> dict[str, Any]:
    symbols = {
        name: sp.Symbol(name, real=True)
        for name in (*_STATE_NAMES, *_DERIVATIVE_NAMES)
    }
    background = ir.get("compiled_flrw_background_system")
    if not isinstance(background, dict):
        raise BackgroundCertificationError("IR has no compiled FLRW background system")

    matrix = _parse_matrix(str(background.get("evolution_matrix")), symbols)
    source = _parse_matrix(str(background.get("evolution_source")), symbols)
    if matrix.shape != (2, 2) or source.shape != (2, 1):
        raise BackgroundCertificationError("FLRW evolution system must be 2x2 with a 2x1 source")

    expressions = {
        "constraint": _parse_expression(str(background.get("energy_constraint_E")), symbols),
        "G_T": _parse_expression(str(ir.get("compiled_tensor_G_T")), symbols),
        "F_T": _parse_expression(str(ir.get("compiled_tensor_F_T")), symbols),
        "Theta": _parse_expression(str(ir.get("compiled_scalar_Theta")), symbols),
        "G_S": _parse_expression(str(ir.get("compiled_scalar_G_S")), symbols),
        "F_S": _parse_expression(str(ir.get("compiled_scalar_F_S")), symbols),
    }
    formulation = ir.get("formulation_classification")
    if not isinstance(formulation, dict):
        raise BackgroundCertificationError("IR has no formulation classification")
    formulation_expressions = {
        "canonical_G2": _parse_expression(str(formulation.get("canonical_G2")), symbols),
        "canonical_G3": _parse_expression(str(formulation.get("canonical_G3")), symbols),
        "G4_X": _parse_expression(str(formulation.get("G4_X")), symbols),
        "kessence_gradient": _parse_expression(
            str(ir.get("compiled_kessence_gradient")), symbols
        ),
        "kessence_legendre_jacobian": _parse_expression(
            str(ir.get("compiled_kessence_homogeneous_legendre_jacobian")), symbols
        ),
        "kessence_energy_density": _parse_expression(
            str(ir.get("compiled_kessence_homogeneous_energy_density")), symbols
        ),
    }
    expressions.update(formulation_expressions)
    all_expressions = [*matrix, *source, *expressions.values()]
    dynamic_names = set(symbols)
    coefficient_symbols = {
        str(symbol): symbol
        for expression in all_expressions
        for symbol in expression.free_symbols
        if str(symbol) not in dynamic_names
    }
    required_coefficients = sorted(coefficient_symbols)
    supplied = set(coefficients)
    missing = sorted(set(required_coefficients) - supplied)
    extra = sorted(supplied - set(required_coefficients))
    if missing:
        raise BackgroundCertificationError(
            "missing coefficient assignments: " + ", ".join(missing)
        )
    if extra:
        raise BackgroundCertificationError(
            "unknown coefficient assignments: " + ", ".join(extra)
        )
    substitutions = {
        coefficient_symbols[name]: sp.Rational(str(coefficients[name]))
        for name in required_coefficients
    }
    matrix = matrix.subs(substitutions).applyfunc(sp.factor)
    source = source.subs(substitutions).applyfunc(sp.factor)
    expressions = {
        name: sp.factor(expression.subs(substitutions))
        for name, expression in expressions.items()
    }
    generalized_harmonic_eligible = (
        expressions["canonical_G3"] == 0 and expressions["G4_X"] == 0
    )
    cubic_g3_only = (
        expressions["canonical_G3"] != 0 and expressions["G4_X"] == 0
    )
    cubic_weak_field_expressions: dict[str, tuple[sp.Expr, int]] = {}
    if cubic_g3_only:
        for family, phi_order_limit in (("G2", 1), ("G3", 2)):
            parent = expressions[f"canonical_{family}"]
            for x_order in range(3):
                for phi_order in range(phi_order_limit + 1):
                    name = f"{family}_X{x_order}_phi{phi_order}"
                    exponent = 2 * x_order + 2 if family == "G2" else 2 * x_order
                    cubic_weak_field_expressions[name] = (
                        sp.factor(
                            sp.diff(
                                parent,
                                symbols["x"],
                                x_order,
                                symbols["u"],
                                phi_order,
                            )
                        ),
                        exponent,
                    )
    unresolved = sorted(
        {
            str(symbol)
            for expression in [*matrix, *source, *expressions.values()]
            for symbol in expression.free_symbols
            if str(symbol) not in dynamic_names
        }
    )
    if unresolved:
        raise BackgroundCertificationError(
            "unresolved symbols after coefficient binding: " + ", ".join(unresolved)
        )

    state_symbols = tuple(symbols[name] for name in _STATE_NAMES)
    health_symbols = (*state_symbols, *(symbols[name] for name in _DERIVATIVE_NAMES))
    return {
        "required_coefficients": required_coefficients,
        "matrix_expressions": [[str(matrix[row, col]) for col in range(2)] for row in range(2)],
        "source_expressions": [str(source[row, 0]) for row in range(2)],
        "matrix": [
            [sp.lambdify(state_symbols, matrix[row, col], modules=mp.iv) for col in range(2)]
            for row in range(2)
        ],
        "source": [
            sp.lambdify(state_symbols, source[row, 0], modules=mp.iv) for row in range(2)
        ],
        "constraint": sp.lambdify(
            state_symbols, expressions["constraint"], modules=mp.iv
        ),
        "constraint_expression": expressions["constraint"],
        "state_symbols": state_symbols,
        "health": {
            name: sp.lambdify(health_symbols, expressions[name], modules=mp.iv)
            for name in ("G_T", "F_T", "Theta", "G_S", "F_S")
        },
        "health_expressions": {
            name: str(expressions[name])
            for name in ("G_T", "F_T", "Theta", "G_S", "F_S")
        },
        "formulation": {
            "route": (
                "generalized_harmonic_kessence"
                if generalized_harmonic_eligible
                else "modified_harmonic_uniform_bound_required"
            ),
            "generalized_harmonic_eligible": generalized_harmonic_eligible,
            "cubic_g3_only": cubic_g3_only,
            "proof_route": (
                "generalized_harmonic_kessence"
                if generalized_harmonic_eligible
                else "cubic_horndeski_bssn_weak_field"
                if cubic_g3_only
                else "general_horndeski_modified_harmonic_weak_coupling"
            ),
            "canonical_G3": str(expressions["canonical_G3"]),
            "G4_X": str(expressions["G4_X"]),
            "kessence_health": (
                {
                    name: sp.lambdify(
                        state_symbols,
                        expressions[name],
                        modules=mp.iv,
                    )
                    for name in (
                        "kessence_gradient",
                        "kessence_legendre_jacobian",
                        "kessence_energy_density",
                    )
                }
                if generalized_harmonic_eligible
                else {}
            ),
            "kessence_health_expressions": {
                name: str(expressions[name])
                for name in (
                    "kessence_gradient",
                    "kessence_legendre_jacobian",
                    "kessence_energy_density",
                )
            },
            "cubic_weak_field": {
                name: {
                    "function": sp.lambdify(
                        state_symbols, expression, modules=mp.iv
                    ),
                    "expression": str(expression),
                    "E_exponent": exponent,
                }
                for name, (expression, exponent) in cubic_weak_field_expressions.items()
            },
        },
    }


def _flow(
    state: tuple[Any, Any, Any], bundle: dict[str, Any], determinant_floor: float
) -> tuple[tuple[Any, Any, Any], Any]:
    matrix = [
        [bundle["matrix"][row][col](*state) for col in range(2)]
        for row in range(2)
    ]
    source = [bundle["source"][row](*state) for row in range(2)]
    determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    if _minimum_absolute(determinant) <= determinant_floor:
        raise BackgroundCertificationError(
            "FLRW evolution determinant interval reaches its singularity floor"
        )
    h_tau = (source[0] * matrix[1][1] - matrix[0][1] * source[1]) / determinant
    x_tau = (matrix[0][0] * source[1] - source[0] * matrix[1][0]) / determinant
    x_lower, _ = _bounds(state[1])
    if x_lower <= 0:
        raise BackgroundCertificationError("timelike scalar branch requires x>0")
    u_tau = mp.iv.sqrt(2 * state[1])
    for value in (u_tau, x_tau, h_tau, determinant):
        _bounds(value)
    return (u_tau, x_tau, h_tau), determinant


def _health(
    state: tuple[Any, Any, Any],
    flow: tuple[Any, Any, Any],
    bundle: dict[str, Any],
    margin: float,
) -> dict[str, Any]:
    arguments = (*state, flow[2], flow[1])
    values = {name: function(*arguments) for name, function in bundle["health"].items()}
    for name in ("G_T", "F_T", "G_S", "F_S"):
        lower, _ = _bounds(values[name])
        if lower <= margin:
            raise BackgroundCertificationError(
                f"{name} interval reaches the non-positive health margin"
            )
    if _minimum_absolute(values["Theta"]) <= margin:
        raise BackgroundCertificationError("Theta interval reaches its singularity margin")
    for name, function in bundle["formulation"]["kessence_health"].items():
        values[name] = function(*state)
        lower, _ = _bounds(values[name])
        if lower <= margin:
            raise BackgroundCertificationError(
                f"{name} interval reaches the non-positive k-essence health margin"
            )
    return values


def _cubic_weak_field_diagnostic(
    state: tuple[Any, Any, Any],
    flow: tuple[Any, Any, Any],
    health: dict[str, Any],
    bundle: dict[str, Any],
    slicing_parameter: float,
    cone_margin: float,
) -> dict[str, Any]:
    curvature_component_max = max(
        _maximum_absolute(state[2] ** 2),
        _maximum_absolute(flow[2] + state[2] ** 2),
    )
    scalar_gradient_max = _maximum_absolute(flow[0])
    scalar_hessian_component_max = max(
        _maximum_absolute(flow[1] / flow[0]),
        _maximum_absolute(state[2] * flow[0]),
    )
    weak_field_scale = max(
        math.sqrt(curvature_component_max),
        scalar_gradient_max,
        math.sqrt(scalar_hessian_component_max),
    )
    ratios: dict[str, float] = {}
    for name, item in bundle["formulation"]["cubic_weak_field"].items():
        derivative_bound = _maximum_absolute(item["function"](*state))
        ratios[name] = derivative_bound * weak_field_scale ** item["E_exponent"]
    scalar_speed_squared = health["F_S"] / health["G_S"]
    slicing_cone_gap = 2 * slicing_parameter - scalar_speed_squared
    gap_minimum_absolute = _minimum_absolute(slicing_cone_gap)
    if gap_minimum_absolute <= cone_margin:
        raise BackgroundCertificationError(
            "cubic BSSN scalar/slicing cone gap reaches its declared margin"
        )
    return {
        "E_upper_bound": weak_field_scale,
        "derivative_ratios_upper_bounds": ratios,
        "scalar_slicing_cone_gap_min_abs": gap_minimum_absolute,
    }


def _analytic_reference(
    reference: dict[str, Any], tau_end: float
) -> dict[str, float] | None:
    if reference.get("type") != "canonical_scalar_stiff":
        return None
    h0 = float(reference["h0"])
    u0 = float(reference.get("u0", 0.0))
    denominator = 1 + 3 * h0 * tau_end
    h = h0 / denominator
    return {
        "u": u0 + math.sqrt(6) * math.log(denominator) / 3,
        "x": 3 * h**2,
        "h": h,
    }


def certify_flrw_background(ir: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Interval-certify one on-shell FLRW trajectory and its perturbative health patch."""

    errors: list[str] = []
    partial: dict[str, Any] = {}
    try:
        if config.get("schema_version") != "sigma-flrw-background-run-1.0":
            raise BackgroundCertificationError("unsupported FLRW run schema")
        if ir.get("schema_version") != "sigma-scalar-tensor-pack-ir-1.0":
            raise BackgroundCertificationError("unsupported scalar-tensor IR schema")
        precision = int(config.get("precision_digits", 40))
        if precision < 20:
            raise BackgroundCertificationError("precision_digits must be at least 20")
        mp.iv.dps = precision

        coefficients = config.get("coefficients")
        if not isinstance(coefficients, dict):
            raise BackgroundCertificationError("coefficients must be an object")
        bundle = _compile_bundle(ir, coefficients)
        partial["required_coefficients"] = bundle["required_coefficients"]

        initial = config.get("initial_state")
        if not isinstance(initial, dict) or set(initial) != set(_STATE_NAMES):
            raise BackgroundCertificationError("initial_state must contain exactly u, x, and h")
        initial_radius = float(config.get("initial_radius", 0.0))
        if initial_radius < 0:
            raise BackgroundCertificationError("initial_radius must be non-negative")
        state = tuple(
            _interval(str(initial[name]))
            if initial_radius == 0
            else _interval(
                float(initial[name]) - initial_radius,
                float(initial[name]) + initial_radius,
            )
            for name in _STATE_NAMES
        )
        tau_start = float(config.get("tau_start", 0.0))
        tau_end = float(config["tau_end"])
        step = float(config["step"])
        if not tau_end > tau_start or not step > 0:
            raise BackgroundCertificationError("require tau_end>tau_start and step>0")
        step_count_float = (tau_end - tau_start) / step
        step_count = round(step_count_float)
        if step_count <= 0 or abs(step_count * step - (tau_end - tau_start)) > 1e-12:
            raise BackgroundCertificationError("the time span must be an integer multiple of step")

        determinant_floor = float(config.get("determinant_floor", 1e-12))
        health_margin = float(config.get("health_margin", 1e-12))
        cubic_bssn_slicing_parameter = float(
            config.get("cubic_bssn_slicing_parameter", 1.0)
        )
        cubic_bssn_cone_margin = float(
            config.get("cubic_bssn_cone_margin", health_margin)
        )
        constraint_tolerance = float(config.get("constraint_tolerance", 1e-7))
        picard_iterations = int(config.get("picard_iterations", 16))
        inflation_absolute = float(config.get("inflation_absolute", 1e-15))
        inflation_relative = float(config.get("inflation_relative", 0.1))
        if min(determinant_floor, health_margin, constraint_tolerance) <= 0:
            raise BackgroundCertificationError("all certification tolerances must be positive")
        if bundle["formulation"]["cubic_g3_only"] and (
            cubic_bssn_slicing_parameter <= 0.5 or cubic_bssn_cone_margin <= 0
        ):
            raise BackgroundCertificationError(
                "cubic BSSN requires slicing parameter >1/2 and positive cone margin"
            )
        if picard_iterations < 2:
            raise BackgroundCertificationError("picard_iterations must be at least two")

        initial_constraint = bundle["constraint"](*state)
        if not _contains_zero(initial_constraint):
            raise BackgroundCertificationError("initial energy-constraint interval excludes zero")
        if _maximum_absolute(initial_constraint) > constraint_tolerance:
            raise BackgroundCertificationError(
                "initial energy-constraint enclosure exceeds the drift tolerance"
            )

        global_hull = state
        health_minima = {name: math.inf for name in ("G_T", "F_T", "G_S", "F_S")}
        kessence_health_minima = {
            name: math.inf for name in bundle["formulation"]["kessence_health"]
        }
        cubic_weak_field_ratio_maxima = {
            name: 0.0 for name in bundle["formulation"]["cubic_weak_field"]
        }
        cubic_weak_field_scale_maximum = 0.0
        cubic_scalar_slicing_cone_gap_minimum = math.inf
        theta_minimum_absolute = math.inf
        determinant_minimum_absolute = math.inf
        maximum_constraint_enclosure = _maximum_absolute(initial_constraint)
        steps: list[dict[str, Any]] = []
        tau = tau_start
        time_interval = _interval(0.0, step)

        for index in range(step_count):
            initial_flow, initial_determinant = _flow(state, bundle, determinant_floor)
            endpoint_euler = tuple(
                state[position] + step * initial_flow[position] for position in range(3)
            )
            enclosure = tuple(
                _inflate(
                    _hull(state[position], endpoint_euler[position]),
                    inflation_absolute,
                    inflation_relative,
                )
                for position in range(3)
            )
            included = False
            enclosure_flow = initial_flow
            enclosure_determinant = initial_determinant
            iterations_used = 0
            for iteration in range(1, picard_iterations + 1):
                iterations_used = iteration
                enclosure_flow, enclosure_determinant = _flow(
                    enclosure, bundle, determinant_floor
                )
                reach = tuple(
                    state[position] + time_interval * enclosure_flow[position]
                    for position in range(3)
                )
                if all(_subset(reach[position], enclosure[position]) for position in range(3)):
                    included = True
                    break
                enclosure = tuple(
                    _inflate(
                        _hull(enclosure[position], reach[position]),
                        inflation_absolute,
                        inflation_relative,
                    )
                    for position in range(3)
                )
            if not included:
                raise BackgroundCertificationError(
                    f"Picard enclosure failed at step {index}; reduce the step size"
                )

            health = _health(enclosure, enclosure_flow, bundle, health_margin)
            cubic_diagnostic: dict[str, Any] | None = None
            if bundle["formulation"]["cubic_g3_only"]:
                cubic_diagnostic = _cubic_weak_field_diagnostic(
                    enclosure,
                    enclosure_flow,
                    health,
                    bundle,
                    cubic_bssn_slicing_parameter,
                    cubic_bssn_cone_margin,
                )
            endpoint = tuple(
                state[position] + step * enclosure_flow[position] for position in range(3)
            )
            endpoint_constraint = bundle["constraint"](*endpoint)
            constraint_bound = _maximum_absolute(endpoint_constraint)
            if not _contains_zero(endpoint_constraint):
                raise BackgroundCertificationError(
                    f"constraint enclosure excludes zero at step {index}"
                )
            if constraint_bound > constraint_tolerance:
                raise BackgroundCertificationError(
                    f"constraint-drift enclosure exceeds tolerance at step {index}"
                )

            for name, previous_minimum in health_minima.items():
                lower, _ = _bounds(health[name])
                health_minima[name] = min(previous_minimum, lower)
            for name, previous_minimum in kessence_health_minima.items():
                lower, _ = _bounds(health[name])
                kessence_health_minima[name] = min(previous_minimum, lower)
            if cubic_diagnostic is not None:
                cubic_weak_field_scale_maximum = max(
                    cubic_weak_field_scale_maximum,
                    cubic_diagnostic["E_upper_bound"],
                )
                for name, value in cubic_diagnostic[
                    "derivative_ratios_upper_bounds"
                ].items():
                    cubic_weak_field_ratio_maxima[name] = max(
                        cubic_weak_field_ratio_maxima[name], value
                    )
                cubic_scalar_slicing_cone_gap_minimum = min(
                    cubic_scalar_slicing_cone_gap_minimum,
                    cubic_diagnostic["scalar_slicing_cone_gap_min_abs"],
                )
            theta_minimum_absolute = min(
                theta_minimum_absolute, _minimum_absolute(health["Theta"])
            )
            determinant_minimum_absolute = min(
                determinant_minimum_absolute,
                _minimum_absolute(enclosure_determinant),
            )
            maximum_constraint_enclosure = max(
                maximum_constraint_enclosure, constraint_bound
            )
            global_hull = tuple(
                _hull(global_hull[position], enclosure[position]) for position in range(3)
            )
            tau = tau_start + (index + 1) * step
            steps.append(
                {
                    "index": index,
                    "tau_end": tau,
                    "picard_iterations": iterations_used,
                    "state_endpoint": {
                        name: _interval_record(endpoint[position])
                        for position, name in enumerate(_STATE_NAMES)
                    },
                    "constraint_max_abs": constraint_bound,
                    "health_lower_bounds": {
                        name: _bounds(health[name])[0] for name in health_minima
                    },
                    "kessence_health_lower_bounds": {
                        name: _bounds(health[name])[0]
                        for name in kessence_health_minima
                    },
                    "cubic_weak_field_diagnostic": cubic_diagnostic,
                    "Theta_min_abs": _minimum_absolute(health["Theta"]),
                    "determinant_min_abs": _minimum_absolute(enclosure_determinant),
                }
            )
            state = endpoint

        reference = _analytic_reference(config.get("analytic_reference", {}), tau_end)
        reference_check: dict[str, Any] | None = None
        if reference is not None:
            contained = {}
            for position, name in enumerate(_STATE_NAMES):
                lower, upper = _bounds(state[position])
                contained[name] = lower <= reference[name] <= upper
            if not all(contained.values()):
                raise BackgroundCertificationError(
                    "interval endpoint does not contain the declared analytic reference"
                )
            reference_check = {"values": reference, "contained": contained, "passed": True}

        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_interval_certified",
            "errors": [],
            "certification": {
                "method": "outward-rounded interval Picard enclosure",
                "precision_digits": precision,
                "claim": (
                    "Every accepted step encloses the continuous FLRW solution and uniformly "
                    "excludes the declared determinant and perturbative-health singular surfaces; "
                    "generalized-harmonic k-essence routes also prove positive homogeneous scalar "
                    "Legendre and energy margins along the trajectory."
                ),
                "limitations": (
                    "This is an FLRW trajectory certificate, not an arbitrary-inhomogeneous "
                    "strong-hyperbolicity or nonlinear global-energy theorem."
                ),
            },
            "source_ir_sha256": ir.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "coefficients": coefficients,
            "required_coefficients": bundle["required_coefficients"],
            "time": {
                "tau_start": tau_start,
                "tau_end": tau_end,
                "step": step,
                "steps": step_count,
            },
            "initial_state": initial,
            "endpoint_enclosure": {
                name: _interval_record(state[position])
                for position, name in enumerate(_STATE_NAMES)
            },
            "trajectory_hull": {
                name: _interval_record(global_hull[position])
                for position, name in enumerate(_STATE_NAMES)
            },
            "uniform_certificate": {
                "health_lower_bounds": health_minima,
                "kessence_health_lower_bounds": kessence_health_minima,
                "Theta_min_abs": theta_minimum_absolute,
                "evolution_determinant_min_abs": determinant_minimum_absolute,
                "constraint_max_abs_enclosure": maximum_constraint_enclosure,
                "constraint_tolerance": constraint_tolerance,
            },
            "analytic_reference": reference_check,
            "formulation_certificate": {
                "route": bundle["formulation"]["route"],
                "status": (
                    "pass_generalized_harmonic_kessence_on_certified_trajectory"
                    if bundle["formulation"]["generalized_harmonic_eligible"]
                    else "unresolved_cubic_bssn_uniform_inhomogeneous_weak_field_bounds_required"
                    if bundle["formulation"]["cubic_g3_only"]
                    else "unresolved_modified_harmonic_uniform_bound_required"
                ),
                "proof_route": bundle["formulation"]["proof_route"],
                "canonical_G3": bundle["formulation"]["canonical_G3"],
                "G4_X": bundle["formulation"]["G4_X"],
                "kessence_health_expressions": bundle["formulation"][
                    "kessence_health_expressions"
                ],
                "uniform_kessence_health_lower_bounds": kessence_health_minima,
                "nonlinear_scalar_energy_certificate": (
                    "pass_positive_homogeneous_energy_and_legendre_margin"
                    if bundle["formulation"]["generalized_harmonic_eligible"]
                    else "not_applicable_modified_harmonic_route"
                ),
                "cubic_bssn_homogeneous_diagnostic": (
                    {
                        "status": (
                            "pass_homogeneous_derivative_ratios_and_scalar_slicing_cone_gap_reported"
                        ),
                        "source": "https://arxiv.org/abs/1904.00963",
                        "slicing_parameter_sigma": cubic_bssn_slicing_parameter,
                        "weak_field_scale_E_upper_bound": (
                            cubic_weak_field_scale_maximum
                        ),
                        "derivative_ratio_upper_bounds": (
                            cubic_weak_field_ratio_maxima
                        ),
                        "maximum_derivative_ratio": max(
                            cubic_weak_field_ratio_maxima.values(), default=0.0
                        ),
                        "scalar_slicing_cone_gap_min_abs": (
                            cubic_scalar_slicing_cone_gap_minimum
                        ),
                        "cone_margin": cubic_bssn_cone_margin,
                        "interpretation": (
                            "Quantitative homogeneous-frame diagnostic only. The source requires "
                            "these ratios to be much less than one but supplies no universal "
                            "numeric threshold; an inhomogeneous-domain bound is still required."
                        ),
                        "expressions": {
                            name: {
                                "expression": item["expression"],
                                "E_exponent": item["E_exponent"],
                            }
                            for name, item in bundle["formulation"][
                                "cubic_weak_field"
                            ].items()
                        },
                    }
                    if bundle["formulation"]["cubic_g3_only"]
                    else None
                ),
                "scope": (
                    "The pass applies to the enclosed homogeneous trajectory. An arbitrary-"
                    "inhomogeneous domain still requires common-time and uniform background "
                    "bounds; cubic G3-only candidates require uniform BSSN weak-field/cone bounds, "
                    "while G4_X modified-harmonic candidates require a positive symmetrizer and "
                    "correction/cone-separation certificate. The local scalar "
                    "energy margin is not a global gravitational positive-energy theorem."
                ),
            },
            "expressions": {
                "evolution_matrix": bundle["matrix_expressions"],
                "evolution_source": bundle["source_expressions"],
                "health": bundle["health_expressions"],
            },
            "step_certificates": steps,
        }
        return {
            **body,
            "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
        }
    except (BackgroundCertificationError, KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "source_ir_sha256": ir.get("content_sha256"),
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            **partial,
        }
        return {
            **body,
            "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
        }
