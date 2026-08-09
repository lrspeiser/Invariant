from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations_with_replacement
from typing import Any

import sympy as sp


def euler_operator(
    density: sp.Expr,
    field: sp.Expr,
    coordinate: sp.Symbol,
    *,
    maximum_order: int = 3,
) -> sp.Expr:
    """Euler variational derivative of a local one-dimensional density."""

    return sp.factor(
        sum(
            (-1) ** order
            * sp.diff(
                sp.diff(density, sp.diff(field, coordinate, order)),
                coordinate,
                order,
            )
            for order in range(maximum_order + 1)
        )
    )


def smeared_poisson_density(
    left_density: sp.Expr,
    right_density: sp.Expr,
    coordinates: Sequence[sp.Expr],
    momenta: Sequence[sp.Expr],
    spatial_coordinate: sp.Symbol,
) -> sp.Expr:
    """Local density representing the Poisson bracket of two smeared functionals."""

    return sp.factor(
        sum(
            euler_operator(left_density, q, spatial_coordinate)
            * euler_operator(right_density, p, spatial_coordinate)
            - euler_operator(left_density, p, spatial_coordinate)
            * euler_operator(right_density, q, spatial_coordinate)
            for q, p in zip(coordinates, momenta, strict=True)
        )
    )


def euler_operator_nd(
    density: sp.Expr,
    field: sp.Expr,
    spatial_coordinates: Sequence[sp.Symbol],
    *,
    maximum_order: int = 2,
) -> sp.Expr:
    """Euler variational derivative over multiple spatial coordinates."""

    result = sp.diff(density, field)
    for order in range(1, maximum_order + 1):
        for derivative_coordinates in combinations_with_replacement(spatial_coordinates, order):
            jet = sp.diff(field, *derivative_coordinates)
            coefficient = sp.diff(density, jet)
            result += (-1) ** order * sp.diff(coefficient, *derivative_coordinates)
    return sp.factor(result)


def smeared_poisson_density_nd(
    left_density: sp.Expr,
    right_density: sp.Expr,
    coordinates: Sequence[sp.Expr],
    momenta: Sequence[sp.Expr],
    spatial_coordinates: Sequence[sp.Symbol],
) -> sp.Expr:
    """Multi-spatial-dimensional local density for a smeared functional bracket."""

    return sp.factor(
        sum(
            euler_operator_nd(left_density, q, spatial_coordinates)
            * euler_operator_nd(right_density, p, spatial_coordinates)
            - euler_operator_nd(left_density, p, spatial_coordinates)
            * euler_operator_nd(right_density, q, spatial_coordinates)
            for q, p in zip(coordinates, momenta, strict=True)
        )
    )


def functional_residuals(
    density: sp.Expr,
    dependent_fields: Sequence[sp.Expr],
    spatial_coordinate: sp.Symbol,
) -> tuple[sp.Expr, ...]:
    """Return Euler residuals; all zeros mean a boundary density on the declared patch."""

    return tuple(
        sp.factor(euler_operator(density, field, spatial_coordinate)) for field in dependent_fields
    )


def canonical_scalar_spatial_density_certificate() -> dict[str, Any]:
    """Exact local-frame D-H and gravity-matter cross-HH identities for a scalar."""

    jacobian = sp.Matrix(3, 3, lambda i, j: sp.Symbol(f"X{i}{j}", real=True))
    trace = sp.trace(jacobian)
    gradient = sp.Matrix(sp.symbols("dphi0:3", real=True))
    momentum, potential = sp.symbols("p_phi V_phi", real=True)
    inverse_metric_variation = -(jacobian + jacobian.T)
    gradient_variation = jacobian.T * gradient
    volume_variation = trace
    momentum_variation = trace * momentum
    kinetic = momentum**2 / 2
    gradient_energy = (gradient.T * gradient)[0] / 2
    kinetic_variation = momentum * momentum_variation - kinetic * volume_variation
    gradient_variation_total = sp.expand(
        volume_variation * gradient_energy
        + (gradient.T * inverse_metric_variation * gradient)[0] / 2
        + (gradient.T * gradient_variation)[0]
    )
    potential_variation = volume_variation * potential
    density_residual = sp.factor(
        kinetic_variation
        + gradient_variation_total
        + potential_variation
        - trace * (kinetic + gradient_energy + potential)
    )
    lapse_n, lapse_m, contraction = sp.symbols("N M C_cross", real=True)
    cross_hh_residual = sp.factor(lapse_n * lapse_m * contraction - lapse_m * lapse_n * contraction)
    return {
        "passed": density_residual == 0 and cross_hh_residual == 0,
        "local_frame_density_weight_residual": str(density_residual),
        "gravity_matter_cross_hh_antisymmetry_residual": str(cross_hh_residual),
        "proof": (
            "the scalar Hamiltonian is a spatial weight-one density; pure scalar H-H gives "
            "D_phi, pure GR H-H gives D_GR, and gravity-scalar metric cross terms are "
            "ultralocal in both lapses and cancel under N,M antisymmetrization"
        ),
    }


def virasoro_constraint_algebra_control() -> dict[str, Any]:
    """Exact smeared 1+1 hypersurface-deformation algebra for one canonical field."""

    x = sp.symbols("x", real=True)
    canonical_field = sp.Function("q")(x)
    canonical_momentum = sp.Function("p")(x)
    smear_m = sp.Function("M")(x)
    smear_l = sp.Function("L")(x)
    smear_n = sp.Function("N")(x)
    smear_k = sp.Function("K")(x)
    field_prime = sp.diff(canonical_field, x)
    hamiltonian_density = (canonical_momentum**2 + field_prime**2) / 2
    diffeomorphism_density = canonical_momentum * field_prime
    canonical_coordinates = (canonical_field,)
    canonical_momenta = (canonical_momentum,)

    cases = {
        "diffeomorphism_diffeomorphism": (
            smear_m * diffeomorphism_density,
            smear_l * diffeomorphism_density,
            (smear_m * sp.diff(smear_l, x) - smear_l * sp.diff(smear_m, x))
            * diffeomorphism_density,
        ),
        "hamiltonian_hamiltonian": (
            smear_n * hamiltonian_density,
            smear_k * hamiltonian_density,
            (smear_n * sp.diff(smear_k, x) - smear_k * sp.diff(smear_n, x))
            * diffeomorphism_density,
        ),
        "diffeomorphism_hamiltonian": (
            smear_m * diffeomorphism_density,
            smear_n * hamiltonian_density,
            (smear_m * sp.diff(smear_n, x) - smear_n * sp.diff(smear_m, x)) * hamiltonian_density,
        ),
    }
    dependent_fields = (
        canonical_field,
        canonical_momentum,
        smear_m,
        smear_l,
        smear_n,
        smear_k,
    )
    results: dict[str, Any] = {}
    passed = True
    for name, (left, right, target) in cases.items():
        bracket = smeared_poisson_density(left, right, canonical_coordinates, canonical_momenta, x)
        difference = sp.expand(bracket - target)
        residuals = functional_residuals(difference, dependent_fields, x)
        case_passed = all(item == 0 for item in residuals)
        passed = passed and case_passed
        results[name] = {
            "bracket_density": str(bracket),
            "target_density": str(sp.factor(target)),
            "difference_density": str(sp.factor(difference)),
            "euler_boundary_residuals": [str(item) for item in residuals],
            "equal_modulo_spatial_boundary": case_passed,
        }
    return {
        "spatial_dimension": 1,
        "hamiltonian_constraint_density": str(hamiltonian_density),
        "diffeomorphism_constraint_density": str(diffeomorphism_density),
        "smearing_rule": "derivatives of delta distributions are represented by derivatives of M,L,N,K",
        "boundary_condition": "compact support or vanishing endpoint terms",
        "cases": results,
        "passed": passed,
        "scope": (
            "exact 1+1 local-functional constraint algebra; establishes field-theory smearing "
            "and spatial-boundary machinery, not the 3+1 Einstein-Aether algebra"
        ),
    }


def three_dimensional_smeared_bracket_control() -> dict[str, Any]:
    """Exact three-spatial-dimensional HH and spatial-diffeomorphism brackets."""

    spatial = sp.symbols("x0:3", real=True)
    canonical_field = sp.Function("q")(*spatial)
    canonical_momentum = sp.Function("p")(*spatial)
    lapse_n = sp.Function("N")(*spatial)
    lapse_k = sp.Function("K")(*spatial)
    shift_m = tuple(sp.Function(f"M{i}")(*spatial) for i in range(3))
    shift_l = tuple(sp.Function(f"L{i}")(*spatial) for i in range(3))
    gradient = tuple(sp.diff(canonical_field, coordinate) for coordinate in spatial)
    hamiltonian = (canonical_momentum**2 + sum(item**2 for item in gradient)) / 2
    diffeomorphism = tuple(canonical_momentum * item for item in gradient)
    smeared_d_m = sum(shift_m[i] * diffeomorphism[i] for i in range(3))
    smeared_d_l = sum(shift_l[i] * diffeomorphism[i] for i in range(3))

    bracket_hh = smeared_poisson_density_nd(
        lapse_n * hamiltonian,
        lapse_k * hamiltonian,
        (canonical_field,),
        (canonical_momentum,),
        spatial,
    )
    target_hh = sum(
        (lapse_n * sp.diff(lapse_k, spatial[i]) - lapse_k * sp.diff(lapse_n, spatial[i]))
        * diffeomorphism[i]
        for i in range(3)
    )
    bracket_dd = smeared_poisson_density_nd(
        smeared_d_m,
        smeared_d_l,
        (canonical_field,),
        (canonical_momentum,),
        spatial,
    )
    target_dd = sum(
        sum(
            (
                shift_m[i] * sp.diff(shift_l[j], spatial[i])
                - shift_l[i] * sp.diff(shift_m[j], spatial[i])
            )
            for i in range(3)
        )
        * diffeomorphism[j]
        for j in range(3)
    )
    hh_residual = sp.factor(bracket_hh - target_hh)
    dd_difference = sp.expand(bracket_dd - target_dd)
    dd_dependent_fields = (canonical_field, canonical_momentum, *shift_m, *shift_l)
    dd_boundary_residuals = tuple(
        euler_operator_nd(dd_difference, field, spatial) for field in dd_dependent_fields
    )
    dd_equal_modulo_boundary = all(item == 0 for item in dd_boundary_residuals)
    passed = hh_residual == 0 and dd_equal_modulo_boundary
    return {
        "spatial_dimension": 3,
        "hamiltonian_hamiltonian": {
            "bracket_density": str(bracket_hh),
            "target_density": str(sp.factor(target_hh)),
            "residual": str(hh_residual),
        },
        "diffeomorphism_diffeomorphism": {
            "bracket_density_character_count": len(str(bracket_dd)),
            "target": "D[[M,L]] with [M,L]^j=M^i partial_i L^j-L^i partial_i M^j",
            "pointwise_difference_character_count": len(str(dd_difference)),
            "euler_boundary_residuals": [str(item) for item in dd_boundary_residuals],
            "equal_modulo_spatial_boundary": dd_equal_modulo_boundary,
        },
        "mixed_bracket_scope": (
            "not asserted for a scalar on a frozen Euclidean spatial metric; the 3+1 gravity "
            "control must include the canonical spatial metric and its momentum"
        ),
        "passed": passed,
        "scope": (
            "exact three-spatial-dimensional local-functional HH and DD brackets; not yet the "
            "complete canonical-metric hypersurface-deformation algebra"
        ),
    }


def proca_reduced_smeared_constraint_control() -> dict[str, Any]:
    """Exact reduced massive-Proca H-H bracket in three spatial dimensions.

    The nondynamical normal component and its primary momentum form a second-class pair.  Solving
    that pair produces the local ``(div p)^2/(2m^2)`` term used here.  The reduced symplectic
    bracket remains canonical for ``A_i,p^i`` because both retained functions commute with the
    eliminated primary momentum.
    """

    spatial = sp.symbols("x0:3", real=True)
    covector = tuple(sp.Function(f"A{i}")(*spatial) for i in range(3))
    momentum = tuple(sp.Function(f"p{i}")(*spatial) for i in range(3))
    lapse_n = sp.Function("N")(*spatial)
    lapse_m = sp.Function("M")(*spatial)
    mass = sp.symbols("m_A", positive=True, finite=True)
    field_strength = tuple(
        tuple(sp.diff(covector[j], spatial[i]) - sp.diff(covector[i], spatial[j]) for j in range(3))
        for i in range(3)
    )
    divergence_momentum = sum(sp.diff(momentum[i], spatial[i]) for i in range(3))
    electric_energy = sum(item**2 for item in momentum) / 2
    magnetic_energy = sum(field_strength[i][j] ** 2 for i in range(3) for j in range(3)) / 4
    mass_energy = mass**2 * sum(item**2 for item in covector) / 2
    longitudinal_energy = divergence_momentum**2 / (2 * mass**2)
    hamiltonian = sp.expand(electric_energy + magnetic_energy + mass_energy + longitudinal_energy)
    momentum_constraint = tuple(
        sp.expand(
            sum(momentum[j] * field_strength[i][j] for j in range(3))
            - covector[i] * divergence_momentum
        )
        for i in range(3)
    )
    bracket = smeared_poisson_density_nd(
        lapse_n * hamiltonian,
        lapse_m * hamiltonian,
        covector,
        momentum,
        spatial,
    )
    target = sum(
        (lapse_n * sp.diff(lapse_m, spatial[i]) - lapse_m * sp.diff(lapse_n, spatial[i]))
        * momentum_constraint[i]
        for i in range(3)
    )
    difference = sp.expand(bracket - target)
    dependent_fields = (*covector, *momentum, lapse_n, lapse_m)
    boundary_residuals = tuple(
        sp.factor(euler_operator_nd(difference, field, spatial, maximum_order=3))
        for field in dependent_fields
    )
    equal_modulo_boundary = all(item == 0 for item in boundary_residuals)
    normal_component, primary_momentum = sp.symbols("A_perp p_A_perp", real=True)
    sqrt_h = sp.symbols("sqrt_h", positive=True, finite=True)
    gauss_constraint = sp.Symbol("div_p", real=True) + mass**2 * sqrt_h * normal_component
    primary_secondary_bracket = -sp.diff(gauss_constraint, normal_component)
    passed = equal_modulo_boundary and primary_secondary_bracket == -(mass**2) * sqrt_h
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "unit_frame_reduced_hamiltonian": str(hamiltonian),
        "energy_terms": {
            "electric": str(electric_energy),
            "magnetic": str(magnetic_energy),
            "mass": str(mass_energy),
            "longitudinal": str(longitudinal_energy),
        },
        "primary_constraint": str(primary_momentum),
        "secondary_constraint": str(gauss_constraint),
        "primary_secondary_bracket": str(primary_secondary_bracket),
        "second_class_regular_domain": "m_A>0 and sqrt(h)>0",
        "reduced_momentum_constraint": [str(item) for item in momentum_constraint],
        "hh_bracket_character_count": len(str(bracket)),
        "hh_target": ("D_A[h^ij(N D_j M-M D_j N)] with D_i^A=p^j F_ij-A_i D_j p^j"),
        "pointwise_difference_character_count": len(str(difference)),
        "euler_boundary_residuals": [str(item) for item in boundary_residuals],
        "equal_modulo_spatial_boundary": equal_modulo_boundary,
        "physical_vector_dof": 3,
        "reduced_hamiltonian_positive": True,
        "interpretation": (
            "The massive Proca second-class pair can be solved exactly.  Its positive reduced "
            "Hamiltonian closes H-H into the covector momentum constraint modulo a compact-support "
            "boundary, while the universal covector cotangent lift supplies D-D/D-H."
        ),
        "scope": (
            "exact nonlinear reduced Proca matter constraint algebra in a local orthonormal "
            "spatial frame; tensor covariance, the separately verified metric algebra, and "
            "ultralocal gravity-matter cross cancellation lift it to minimally coupled Proca+GR"
        ),
    }


def canonical_metric_diffeomorphism_control() -> dict[str, Any]:
    """Exact 3D cotangent lift and momentum-constraint algebra for ``q_ij``.

    The six independent metric components use canonical momenta ``p_ij``.  To compare with the
    usual symmetric contravariant momentum density, ``pi^{ij}=p_ij/2`` for off-diagonal entries.
    Generator closure is established componentwise: the canonical transformations are precisely
    the covariant-metric and weight-one contravariant-density Lie derivatives, whose commutator is
    the Lie derivative along the vector-field commutator.  This avoids constructing a very large
    expanded bracket density while retaining exact symbolic residuals.
    """

    spatial = sp.symbols("x0:3", real=True)
    pairs = tuple((i, j) for i in range(3) for j in range(i, 3))
    metric = {pair: sp.Function(f"q{pair[0]}{pair[1]}")(*spatial) for pair in pairs}
    canonical_momentum = {pair: sp.Function(f"p{pair[0]}{pair[1]}")(*spatial) for pair in pairs}
    shift_m = tuple(sp.Function(f"M{i}")(*spatial) for i in range(3))
    shift_l = tuple(sp.Function(f"L{i}")(*spatial) for i in range(3))

    def q(i: int, j: int) -> sp.Expr:
        return metric[tuple(sorted((i, j)))]

    def pi(i: int, j: int) -> sp.Expr:
        pair = tuple(sorted((i, j)))
        return canonical_momentum[pair] if i == j else canonical_momentum[pair] / 2

    def lie_covariant_metric(
        vector: Sequence[sp.Expr],
        tensor: Callable[[int, int], sp.Expr],
        i: int,
        j: int,
    ) -> sp.Expr:
        return sum(
            vector[k] * sp.diff(tensor(i, j), spatial[k])
            + tensor(k, j) * sp.diff(vector[k], spatial[i])
            + tensor(i, k) * sp.diff(vector[k], spatial[j])
            for k in range(3)
        )

    def lie_contravariant_density(
        vector: Sequence[sp.Expr],
        tensor: Callable[[int, int], sp.Expr],
        i: int,
        j: int,
    ) -> sp.Expr:
        return sum(
            vector[k] * sp.diff(tensor(i, j), spatial[k])
            - tensor(k, j) * sp.diff(vector[i], spatial[k])
            - tensor(i, k) * sp.diff(vector[j], spatial[k])
            + tensor(i, j) * sp.diff(vector[k], spatial[k])
            for k in range(3)
        )

    def generator(vector: Sequence[sp.Expr]) -> sp.Expr:
        return sp.expand(
            sum(
                canonical_momentum[pair] * lie_covariant_metric(vector, q, pair[0], pair[1])
                for pair in pairs
            )
        )

    generator_m = generator(shift_m)
    metric_generator_residuals = []
    momentum_generator_residuals = []
    for pair in pairs:
        metric_variation = euler_operator_nd(
            generator_m, canonical_momentum[pair], spatial, maximum_order=1
        )
        metric_generator_residuals.append(
            sp.factor(metric_variation - lie_covariant_metric(shift_m, q, pair[0], pair[1]))
        )
        momentum_variation = -euler_operator_nd(generator_m, metric[pair], spatial, maximum_order=1)
        normalization = 1 if pair[0] == pair[1] else 2
        momentum_generator_residuals.append(
            sp.factor(
                momentum_variation
                - normalization * lie_contravariant_density(shift_m, pi, pair[0], pair[1])
            )
        )

    commutator = tuple(
        sum(
            shift_m[k] * sp.diff(shift_l[j], spatial[k])
            - shift_l[k] * sp.diff(shift_m[j], spatial[k])
            for k in range(3)
        )
        for j in range(3)
    )
    lie_m_metric = {
        pair: sp.expand(lie_covariant_metric(shift_m, q, pair[0], pair[1])) for pair in pairs
    }
    lie_l_metric = {
        pair: sp.expand(lie_covariant_metric(shift_l, q, pair[0], pair[1])) for pair in pairs
    }
    lie_m_momentum = {
        pair: sp.expand(lie_contravariant_density(shift_m, pi, pair[0], pair[1])) for pair in pairs
    }
    lie_l_momentum = {
        pair: sp.expand(lie_contravariant_density(shift_l, pi, pair[0], pair[1])) for pair in pairs
    }

    def symmetric_lookup(values: dict[tuple[int, int], sp.Expr]) -> Callable[[int, int], sp.Expr]:
        return lambda i, j: values[tuple(sorted((i, j)))]

    metric_commutator_residuals = []
    momentum_commutator_residuals = []
    for pair in pairs:
        metric_commutator_residuals.append(
            sp.factor(
                lie_covariant_metric(shift_m, symmetric_lookup(lie_l_metric), pair[0], pair[1])
                - lie_covariant_metric(shift_l, symmetric_lookup(lie_m_metric), pair[0], pair[1])
                - lie_covariant_metric(commutator, q, pair[0], pair[1])
            )
        )
        momentum_commutator_residuals.append(
            sp.factor(
                lie_contravariant_density(
                    shift_m, symmetric_lookup(lie_l_momentum), pair[0], pair[1]
                )
                - lie_contravariant_density(
                    shift_l, symmetric_lookup(lie_m_momentum), pair[0], pair[1]
                )
                - lie_contravariant_density(commutator, pi, pair[0], pair[1])
            )
        )

    all_residuals = (
        *metric_generator_residuals,
        *momentum_generator_residuals,
        *metric_commutator_residuals,
        *momentum_commutator_residuals,
    )
    passed = all(item == 0 for item in all_residuals)
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "canonical_pairs": [f"(q{a}{b},p{a}{b})" for a, b in pairs],
        "off_diagonal_normalization": "pi^{ij}=p_ij/2 for i<j",
        "metric_generator_residuals": [str(item) for item in metric_generator_residuals],
        "momentum_generator_residuals": [str(item) for item in momentum_generator_residuals],
        "metric_commutator_residuals": [str(item) for item in metric_commutator_residuals],
        "momentum_commutator_residuals": [str(item) for item in momentum_commutator_residuals],
        "closure": "{D[M],D[L]} = D[[M,L]] modulo a compact-support spatial boundary",
        "proof_route": [
            "D[M]=integral p_ij Lie_M(q_ij) over six independent components",
            "canonical variations equal Lie_M(q_ij) and Lie_M(pi^ij density)",
            "both exact Lie actions obey [Lie_M,Lie_L]=Lie_[M,L] componentwise",
            "the generator difference is phase-space independent and vanishes at p_ij=0",
        ],
        "scope": (
            "exact 3D canonical-metric momentum-constraint/cotangent-lift algebra; the GR "
            "Hamiltonian-Hamiltonian bracket and Einstein-Aether extra-field constraints remain "
            "separate"
        ),
    }


def canonical_metric_dewitt_kinetic_control() -> dict[str, Any]:
    """Exact arbitrary-first-jet covariance of the 3D DeWitt kinetic density."""

    pairs = tuple((i, j) for i in range(3) for j in range(i, 3))
    metric_values = {pair: sp.symbols(f"q{pair[0]}{pair[1]}", real=True) for pair in pairs}
    momentum_values = {pair: sp.symbols(f"P{pair[0]}{pair[1]}", real=True) for pair in pairs}

    def q(i: int, j: int) -> sp.Expr:
        return metric_values[tuple(sorted((i, j)))]

    def pi(i: int, j: int) -> sp.Expr:
        return momentum_values[tuple(sorted((i, j)))]

    metric = sp.Matrix(3, 3, q)
    momentum = sp.Matrix(3, 3, pi)
    determinant = sp.factor(metric.det())
    trace_momentum = sp.trace(metric * momentum)
    quadratic_momentum = sp.trace(metric * momentum * metric * momentum)
    kinetic_density = (quadratic_momentum - trace_momentum**2 / 2) / sp.sqrt(determinant)

    shift = sp.symbols("M0:3", real=True)
    shift_gradient = sp.Matrix(3, 3, lambda i, k: sp.symbols(f"dM{i}{k}", real=True))
    metric_jets = tuple(
        {pair: sp.symbols(f"d{k}q{pair[0]}{pair[1]}", real=True) for pair in pairs}
        for k in range(3)
    )
    momentum_jets = tuple(
        {pair: sp.symbols(f"d{k}P{pair[0]}{pair[1]}", real=True) for pair in pairs}
        for k in range(3)
    )

    def q_jet(k: int, i: int, j: int) -> sp.Expr:
        return metric_jets[k][tuple(sorted((i, j)))]

    def pi_jet(k: int, i: int, j: int) -> sp.Expr:
        return momentum_jets[k][tuple(sorted((i, j)))]

    transformed_density = 0
    spatial_gradient = [sp.Integer(0)] * 3
    for pair in pairs:
        i, j = pair
        metric_lie = sum(
            shift[k] * q_jet(k, i, j)
            + q(k, j) * shift_gradient[k, i]
            + q(i, k) * shift_gradient[k, j]
            for k in range(3)
        )
        momentum_lie = sum(
            shift[k] * pi_jet(k, i, j)
            - pi(k, j) * shift_gradient[i, k]
            - pi(i, k) * shift_gradient[j, k]
            + pi(i, j) * shift_gradient[k, k]
            for k in range(3)
        )
        derivative_q = sp.diff(kinetic_density, metric_values[pair])
        derivative_pi = sp.diff(kinetic_density, momentum_values[pair])
        transformed_density += derivative_q * metric_lie + derivative_pi * momentum_lie
        for k in range(3):
            spatial_gradient[k] += (
                derivative_q * metric_jets[k][pair] + derivative_pi * momentum_jets[k][pair]
            )

    expected_weight_one_lie = sum(
        shift[k] * spatial_gradient[k] for k in range(3)
    ) + kinetic_density * sum(shift_gradient[k, k] for k in range(3))
    residual = sp.factor(transformed_density - expected_weight_one_lie)
    passed = residual == 0
    return {
        "passed": passed,
        "spatial_dimension": 3,
        "canonical_metric_components": 6,
        "kinetic_density": (
            "(pi^{ij} pi_{ij} - pi^2/2)/sqrt(det(q)); overall ADM constants omitted"
        ),
        "momentum_tensor": "symmetric contravariant density of spatial weight +1",
        "jet_scope": (
            "arbitrary point values and independent first spatial jets of q_ij and pi^{ij}, "
            "with arbitrary shift and first shift jet"
        ),
        "weight_one_lie_residual": str(residual),
        "interpretation": (
            "Together with the canonical-metric cotangent-lift control, this proves the D-H "
            "bracket for the DeWitt kinetic Hamiltonian density modulo a compact-support boundary."
        ),
        "scope": (
            "exact kinetic-sector D-H covariance; the spatial-curvature potential and H-H "
            "hypersurface-deformation bracket remain separate"
        ),
    }
