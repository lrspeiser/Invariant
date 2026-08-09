from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import sympy as sp

from .grammar import Q, X, Z


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    reason: str
    evidence: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_finite_number(value: sp.Expr) -> bool:
    return bool(value.is_number and value.is_finite)


def algebraic_gates(expression: sp.Expr, constants_count: int, maximum_constants: int) -> list[GateResult]:
    origin = sp.simplify(expression.subs({X: 0, Q: 0, Z: 0}))
    finite_origin = _is_finite_number(origin)
    vacuum_zero = finite_origin and origin == 0

    try:
        high_field = sp.simplify(sp.limit(expression.subs({Q: 1, Z: 1}) / X, X, sp.oo))
        screened = high_field == 0
    except (ValueError, TypeError, NotImplementedError):
        high_field = sp.nan
        screened = False

    has_spatial_state = expression.has(Q) or expression.has(Z)
    return [
        GateResult(
            "finite_origin",
            "pass" if finite_origin else "reject",
            "Correction is finite at vanishing flux and state."
            if finite_origin
            else "Correction is singular or undefined at the vacuum origin.",
            {"value": str(origin)},
        ),
        GateResult(
            "vacuum_zero",
            "pass" if vacuum_zero else "reject",
            "No source-independent vacuum offset is introduced."
            if vacuum_zero
            else "The candidate changes the vacuum energy in this static normalization.",
            {"value": str(origin)},
        ),
        GateResult(
            "high_field_newtonian_limit",
            "pass" if screened else "reject",
            "The correction divided by D^2 vanishes as x approaches infinity at fixed q,z."
            if screened
            else "The correction does not decouple relative to D^2 in the declared high-field ray.",
            {"limit_correction_over_x": str(high_field), "fixed_q": 1, "fixed_z": 1},
        ),
        GateResult(
            "new_spatial_state_information",
            "pass" if has_spatial_state else "reject",
            "Candidate depends on the v18 spatial-state sector q or z."
            if has_spatial_state
            else "Flux-only candidates collapse back toward the already-spent AQUAL class.",
            {"depends_on_q": expression.has(Q), "depends_on_z": expression.has(Z)},
        ),
        GateResult(
            "universal_constant_cap",
            "pass" if constants_count <= maximum_constants else "reject",
            f"Uses {constants_count} universal constants against a cap of {maximum_constants}.",
            {"count": constants_count, "maximum": maximum_constants},
        ),
        GateResult(
            "derivative_order",
            "pass",
            "Grammar contains D and first spatial derivatives only; Euler-Lagrange equations are at most second order in this static surrogate.",
            {"maximum_field_derivatives_in_H": 1, "maximum_in_static_equation": 2},
        ),
        GateResult(
            "one_metric_no_private_lensing_law",
            "pass",
            "The grammar changes one constitutive H and contains no object switch or lensing-only parameter.",
            {"object_specific_parameters": 0, "lensing_only_parameters": 0},
        ),
    ]


def sampled_static_convexity(
    expression: sp.Expr, coupling: float, samples: dict[str, list[float]], tolerance: float
) -> GateResult:
    d, p, state = sp.symbols("d p state", real=True)
    # Units a_sigma=L_sigma=Z_0=1. q=p^2 and z=state^2 retain the radial
    # flux/state sector while making the Hessian test explicit and reproducible.
    hamiltonian = sp.Rational(1, 2) * d**2 + sp.Float(coupling) * expression.subs(
        {X: d**2, Q: p**2, Z: state**2}
    )
    hessian = sp.hessian(hamiltonian, (d, p))
    minimum = math.inf
    worst: dict[str, object] = {}
    evaluated = 0
    try:
        # Lambdify scalar entries instead of the Matrix object. The latter emits
        # an ImmutableDenseMatrix constructor that the stdlib math backend does
        # not define.
        evaluators = [
            sp.lambdify((d, p, state), hessian[row, column], modules="math")
            for row in range(2)
            for column in range(2)
        ]
        for d_value in samples["d"]:
            for p_value in samples["p"]:
                for state_value in samples["state"]:
                    a, b, c, e = [
                        float(evaluator(d_value, p_value, state_value))
                        for evaluator in evaluators
                    ]
                    trace = a + e
                    discriminant = max(0.0, (a - e) ** 2 + 4.0 * b * c)
                    eigenvalues = ((trace - math.sqrt(discriminant)) / 2, (trace + math.sqrt(discriminant)) / 2)
                    local_min = min(eigenvalues)
                    evaluated += 1
                    if not math.isfinite(local_min):
                        raise ValueError("non-finite Hessian eigenvalue")
                    if local_min < minimum:
                        minimum = local_min
                        worst = {
                            "d": d_value,
                            "p": p_value,
                            "state": state_value,
                            "eigenvalues": [float(value) for value in eigenvalues],
                        }
    except (ArithmeticError, TypeError, ValueError, ZeroDivisionError) as error:
        return GateResult(
            "sampled_static_convexity",
            "reject",
            "The constitutive Hessian could not be evaluated as a finite real matrix.",
            {"error": str(error), "hessian": str(hessian), "evaluated_points": evaluated},
        )

    passed = minimum > tolerance
    return GateResult(
        "sampled_static_convexity",
        "pass" if passed else "reject",
        "Every sampled radial constitutive Hessian eigenvalue is strictly positive."
        if passed
        else "At least one sampled radial constitutive Hessian is non-positive.",
        {
            "minimum_eigenvalue": minimum,
            "strict_tolerance": tolerance,
            "worst_point": worst,
            "evaluated_points": evaluated,
            "hessian": str(hessian),
            "warning": "A sampled radial test is a kill gate, not a proof of global tensor convexity.",
        },
    )


def deferred_gates() -> list[GateResult]:
    reasons = {
        "covariant_action_completion": "The MVP compiles the frozen static flux sector; no covariant lift has yet been generated.",
        "hamiltonian_and_degree_count": "Requires an unreduced covariant action, constraint analysis, and physical Hamiltonian.",
        "characteristic_cones": "Requires principal symbols on Minkowski, FLRW, and declared tilted backgrounds.",
        "one_metric_matter_and_lensing": "Must be derived from the covariant action rather than imposed on the static surrogate.",
        "solar_system": "Requires a complete screened solution and explicit Cassini and Mercury calculations.",
        "observational_transfer": "SPARC, cluster, raw-lensing, and holdout data remain sealed until all pre-data gates pass.",
    }
    return [GateResult(name, "deferred", reason, {}) for name, reason in reasons.items()]
