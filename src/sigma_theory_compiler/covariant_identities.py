from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import sympy as sp


def _christoffel_symbols(
    metric: sp.Matrix, inverse_metric: sp.Matrix, coordinates: Sequence[sp.Symbol]
) -> list[list[list[sp.Expr]]]:
    dimension = len(coordinates)
    return [
        [
            [
                sp.simplify(
                    sum(
                        inverse_metric[upper, delta]
                        * (
                            sp.diff(metric[delta, beta], coordinates[alpha])
                            + sp.diff(metric[delta, alpha], coordinates[beta])
                            - sp.diff(metric[alpha, beta], coordinates[delta])
                        )
                        for delta in range(dimension)
                    )
                    / 2
                )
                for beta in range(dimension)
            ]
            for alpha in range(dimension)
        ]
        for upper in range(dimension)
    ]


def proca_stress_noether_residuals(
    metric: sp.Matrix,
    coordinates: Sequence[sp.Symbol],
    vector_down: Sequence[sp.Expr],
    *,
    mass: sp.Expr,
) -> list[sp.Expr]:
    """Return ∇^μT_{μν} - F_{νρ}E^ρ + A_ν∇_ρE^ρ for minimally coupled Proca."""

    dimension = len(coordinates)
    if metric.shape != (dimension, dimension) or len(vector_down) != dimension:
        raise ValueError("metric, coordinates, and vector must have the same dimension")
    inverse_metric = sp.simplify(metric.inv())
    connection = _christoffel_symbols(metric, inverse_metric, coordinates)
    vector = sp.Matrix(vector_down)
    vector_up = sp.simplify(inverse_metric * vector)
    field_down = sp.MutableDenseMatrix(
        dimension,
        dimension,
        lambda mu, nu: sp.diff(vector[nu], coordinates[mu])
        - sp.diff(vector[mu], coordinates[nu]),
    )
    field_up = sp.simplify(inverse_metric * field_down * inverse_metric)
    field_squared = sp.factor(
        sum(
            field_down[mu, nu] * field_up[mu, nu]
            for mu in range(dimension)
            for nu in range(dimension)
        )
    )
    vector_squared = sp.factor((vector.T * inverse_metric * vector)[0])
    stress_down = sp.MutableDenseMatrix(
        dimension,
        dimension,
        lambda mu, nu: sp.factor(
            sum(
                field_down[mu, rho]
                * sum(
                    inverse_metric[rho, sigma] * field_down[nu, sigma]
                    for sigma in range(dimension)
                )
                for rho in range(dimension)
            )
            - sp.Rational(1, 4) * metric[mu, nu] * field_squared
            + mass**2
            * (
                vector[mu] * vector[nu]
                - sp.Rational(1, 2) * metric[mu, nu] * vector_squared
            )
        ),
    )
    volume_density = sp.sqrt(-sp.factor(metric.det()))
    euler_up = [
        sp.factor(
            sum(
                sp.diff(volume_density * field_up[mu, rho], coordinates[mu])
                for mu in range(dimension)
            )
            / volume_density
            - mass**2 * vector_up[rho]
        )
        for rho in range(dimension)
    ]
    euler_divergence = sp.factor(
        sum(
            sp.diff(volume_density * euler_up[rho], coordinates[rho])
            for rho in range(dimension)
        )
        / volume_density
    )
    residuals: list[sp.Expr] = []
    for nu in range(dimension):
        stress_divergence = 0
        for mu in range(dimension):
            for alpha in range(dimension):
                covariant_derivative = sp.diff(stress_down[mu, nu], coordinates[alpha])
                covariant_derivative -= sum(
                    connection[lam][alpha][mu] * stress_down[lam, nu]
                    + connection[lam][alpha][nu] * stress_down[mu, lam]
                    for lam in range(dimension)
                )
                stress_divergence += inverse_metric[mu, alpha] * covariant_derivative
        euler_force = sum(
            field_down[nu, rho] * euler_up[rho] for rho in range(dimension)
        )
        residuals.append(
            sp.simplify(
                sp.trigsimp(stress_divergence - euler_force + vector[nu] * euler_divergence)
            )
        )
    return residuals


def proca_curved_background_noether_controls() -> dict[str, Any]:
    mass = sp.Symbol("m", real=True)

    t, x, y, z = sp.symbols("t x y z", real=True)
    scale = sp.Function("a")(t)
    flrw_vector = [sp.Function(f"A{mu}")(t) for mu in range(4)]
    flrw_residuals = proca_stress_noether_residuals(
        sp.diag(-1, scale**2, scale**2, scale**2),
        (t, x, y, z),
        flrw_vector,
        mass=mass,
    )

    time, radius, theta, phi = sp.symbols("t r theta phi", real=True)
    lapse = sp.Function("f")(radius)
    static_vector = [
        sp.Function("A_t")(radius),
        sp.Function("A_r")(radius),
        sp.Integer(0),
        sp.Integer(0),
    ]
    static_residuals = proca_stress_noether_residuals(
        sp.diag(
            -lapse,
            1 / lapse,
            radius**2,
            radius**2 * sp.sin(theta) ** 2,
        ),
        (time, radius, theta, phi),
        static_vector,
        mass=mass,
    )
    return {
        "identity": "nabla^mu T_mu_nu - F_nu_rho E^rho + A_nu nabla_rho E^rho = 0",
        "flrw": {
            "metric": "diag(-1,a(t)^2,a(t)^2,a(t)^2)",
            "profile": "arbitrary homogeneous A_mu(t)",
            "residuals": [str(item) for item in flrw_residuals],
        },
        "static_spherical": {
            "metric": "diag(-f(r),1/f(r),r^2,r^2 sin(theta)^2)",
            "profile": "arbitrary static radial A_t(r), A_r(r)",
            "residuals": [str(item) for item in static_residuals],
        },
        "passed": all(item == 0 for item in [*flrw_residuals, *static_residuals]),
        "scope": "exact nonflat connection-dependent controls; not a proof for every metric/vector profile",
    }


def einstein_aether_flrw_variation_control() -> dict[str, Any]:
    """Reduce K1..K4 on lapse-FLRW and verify its time-diffeomorphism Noether identity."""

    time, x, y, z = sp.symbols("t x y z", real=True)
    coordinates = (time, x, y, z)
    lapse = sp.Function("N")(time)
    scale = sp.Function("a")(time)
    aether_time = sp.Function("U")(time)
    multiplier = sp.Function("lambda")(time)
    c1, c2, c3, c4 = sp.symbols("c1 c2 c3 c4", real=True)
    metric = sp.diag(-lapse**2, scale**2, scale**2, scale**2)
    inverse_metric = sp.simplify(metric.inv())
    connection = _christoffel_symbols(metric, inverse_metric, coordinates)
    vector = sp.Matrix([aether_time, 0, 0, 0])
    vector_up = sp.simplify(inverse_metric * vector)
    dimension = 4
    derivative = sp.MutableDenseMatrix(
        dimension,
        dimension,
        lambda mu, nu: sp.diff(vector[nu], coordinates[mu])
        - sum(
            connection[rho][mu][nu] * vector[rho] for rho in range(dimension)
        ),
    )
    k1 = sp.factor(
        sum(
            inverse_metric[mu, rho]
            * inverse_metric[nu, sigma]
            * derivative[mu, nu]
            * derivative[rho, sigma]
            for mu in range(dimension)
            for nu in range(dimension)
            for rho in range(dimension)
            for sigma in range(dimension)
        )
    )
    expansion = sp.factor(
        sum(
            inverse_metric[mu, nu] * derivative[mu, nu]
            for mu in range(dimension)
            for nu in range(dimension)
        )
    )
    k2 = sp.factor(expansion**2)
    k3 = sp.factor(
        sum(
            inverse_metric[nu, rho]
            * inverse_metric[mu, sigma]
            * derivative[mu, nu]
            * derivative[rho, sigma]
            for mu in range(dimension)
            for nu in range(dimension)
            for rho in range(dimension)
            for sigma in range(dimension)
        )
    )
    acceleration_down = [
        sp.factor(sum(vector_up[mu] * derivative[mu, nu] for mu in range(dimension)))
        for nu in range(dimension)
    ]
    k4 = sp.factor(
        sum(
            inverse_metric[mu, nu] * acceleration_down[mu] * acceleration_down[nu]
            for mu in range(dimension)
            for nu in range(dimension)
        )
    )
    norm = sp.factor((vector.T * inverse_metric * vector)[0])
    reduced_lagrangian = sp.factor(
        lapse
        * scale**3
        * (
            -sp.Rational(1, 2) * (c1 * k1 + c2 * k2 + c3 * k3 - c4 * k4)
            + multiplier * (norm + 1)
        )
    )

    def euler_lagrange(field: sp.Expr) -> sp.Expr:
        return sp.factor(
            sp.diff(reduced_lagrangian, field)
            - sp.diff(sp.diff(reduced_lagrangian, sp.diff(field, time)), time)
        )

    lapse_euler, scale_euler, vector_euler, multiplier_euler = [
        euler_lagrange(field) for field in (lapse, scale, aether_time, multiplier)
    ]
    noether_residual = sp.factor(
        lapse_euler * sp.diff(lapse, time)
        + scale_euler * sp.diff(scale, time)
        + vector_euler * sp.diff(aether_time, time)
        + multiplier_euler * sp.diff(multiplier, time)
        - sp.diff(lapse * lapse_euler + aether_time * vector_euler, time)
    )
    velocities = (
        sp.diff(lapse, time),
        sp.diff(scale, time),
        sp.diff(aether_time, time),
    )
    hessian = sp.simplify(sp.hessian(reduced_lagrangian, velocities))
    gauge_null = sp.Matrix([lapse, 0, aether_time])
    null_residual = [sp.factor(item) for item in hessian * gauge_null]
    rank_control = hessian.subs(
        {
            lapse: 2,
            scale: 3,
            aether_time: 2,
            c1: sp.Rational(1, 10),
            c2: sp.Rational(1, 20),
            c3: sp.Rational(1, 30),
            c4: sp.Rational(1, 40),
        }
    ).rank()
    passed = noether_residual == 0 and all(item == 0 for item in null_residual) and rank_control == 2
    return {
        "action": "sqrt(-g)[-(c1 K1+c2 K2+c3 K3-c4 K4)/2 + lambda(u^2+1)]",
        "background": "lapse-FLRW diag(-N(t)^2,a(t)^2,a(t)^2,a(t)^2)",
        "aether_profile": "independent covector u_mu=(U(t),0,0,0), constraint not substituted",
        "invariants": {"K1": str(k1), "K2": str(k2), "K3": str(k3), "K4": str(k4)},
        "unit_norm": str(norm),
        "noether_identity": "E_N Ndot + E_a adot + E_U Udot + E_lambda lambdadot - d_t(N E_N + U E_U) = 0",
        "noether_residual": str(noether_residual),
        "velocity_hessian": str(hessian),
        "hessian_determinant": str(sp.factor(hessian.det())),
        "gauge_null_vector": ["N", "0", "U"],
        "gauge_null_residual": [str(item) for item in null_residual],
        "declared_point_rank": int(rank_control),
        "passed": passed,
        "scope": "exact nonlinear homogeneous reduction; not the full inhomogeneous Dirac algebra",
    }
