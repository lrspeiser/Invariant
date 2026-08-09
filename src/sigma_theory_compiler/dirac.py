from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp


def poisson_bracket(
    left: sp.Expr,
    right: sp.Expr,
    coordinates: Sequence[sp.Symbol],
    momenta: Sequence[sp.Symbol],
) -> sp.Expr:
    return sp.simplify(
        sum(
            sp.diff(left, q) * sp.diff(right, p)
            - sp.diff(left, p) * sp.diff(right, q)
            for q, p in zip(coordinates, momenta, strict=True)
        )
    )


@dataclass(frozen=True)
class DiracResult:
    coordinates: tuple[sp.Symbol, ...]
    velocities: tuple[sp.Symbol, ...]
    momenta: tuple[sp.Symbol, ...]
    velocity_hessian: sp.Matrix
    canonical_momenta: tuple[sp.Expr, ...]
    primary_constraints: tuple[sp.Expr, ...]
    canonical_hamiltonian: sp.Expr
    secondary_constraints: tuple[sp.Expr, ...]
    higher_generation_constraints: tuple[sp.Expr, ...]
    constraint_generations: tuple[tuple[sp.Expr, ...], ...]
    constraint_matrix_off_surface: sp.Matrix
    constraint_matrix_off_surface_rank: int
    constraint_matrix: sp.Matrix
    constraint_matrix_rank: int
    independent_constraints: int
    first_class_constraints: int
    second_class_constraints: int
    physical_dof: int
    closure: bool
    multiplier_solution: dict[sp.Symbol, sp.Expr]
    unresolved_consistency_conditions: tuple[sp.Expr, ...]

    def as_dict(self) -> dict[str, Any]:
        constraints = (
            self.primary_constraints
            + self.secondary_constraints
            + self.higher_generation_constraints
        )
        return {
            "coordinates": [str(item) for item in self.coordinates],
            "velocities": [str(item) for item in self.velocities],
            "momenta": [str(item) for item in self.momenta],
            "velocity_hessian": str(self.velocity_hessian),
            "hessian_rank": int(self.velocity_hessian.rank()),
            "hessian_nullity": len(self.velocity_hessian.nullspace()),
            "canonical_momenta": [str(item) for item in self.canonical_momenta],
            "primary_constraints": [str(item) for item in self.primary_constraints],
            "canonical_hamiltonian": str(self.canonical_hamiltonian),
            "secondary_constraints": [str(item) for item in self.secondary_constraints],
            "higher_generation_constraints": [
                str(item) for item in self.higher_generation_constraints
            ],
            "constraint_generations": [
                [str(item) for item in generation]
                for generation in self.constraint_generations
            ],
            "all_constraints": [str(item) for item in constraints],
            "constraint_poisson_matrix_off_surface": str(
                self.constraint_matrix_off_surface
            ),
            "constraint_matrix_off_surface_rank": self.constraint_matrix_off_surface_rank,
            "constraint_poisson_matrix": str(self.constraint_matrix),
            "constraint_matrix_rank": self.constraint_matrix_rank,
            "constraint_rank_scope": "generic rank after quotient-ring reduction on the full constraint surface",
            "independent_constraints": self.independent_constraints,
            "first_class_constraints": self.first_class_constraints,
            "second_class_constraints": self.second_class_constraints,
            "physical_dof": self.physical_dof,
            "closure": self.closure,
            "multiplier_solution": {
                str(key): str(value) for key, value in self.multiplier_solution.items()
            },
            "unresolved_consistency_conditions": [
                str(item) for item in self.unresolved_consistency_conditions
            ],
        }


def reduce_on_constraint_surface(
    expression: sp.Expr,
    constraints: Sequence[sp.Expr],
    phase_variables: Sequence[sp.Symbol],
) -> sp.Expr:
    """Reduce a rational expression modulo the polynomial constraint ideal.

    The result is valid on the regular patch where the expression's denominator is nonzero.  A
    non-polynomial constraint is rejected instead of silently falling back to off-surface rank.
    """

    if not constraints:
        return sp.factor(expression)
    variables = tuple(phase_variables)
    constraint_numerators = [sp.together(item).as_numer_denom()[0] for item in constraints]
    try:
        basis = sp.groebner(
            constraint_numerators, *variables, order="grevlex", domain="EX"
        )
        numerator, denominator = sp.together(expression).as_numer_denom()
        numerator_remainder = basis.reduce(numerator)[1]
        denominator_remainder = basis.reduce(denominator)[1]
    except sp.PolynomialError as error:
        raise ValueError(
            "constraint-surface reduction requires polynomial constraints and rational brackets"
        ) from error
    if denominator_remainder == 0:
        raise ValueError("a Poisson-bracket denominator vanishes on the constraint surface")
    return sp.factor(numerator_remainder / denominator)


def reduce_poisson_matrix_on_constraint_surface(
    constraints: Sequence[sp.Expr],
    coordinates: Sequence[sp.Symbol],
    momenta: Sequence[sp.Symbol],
) -> tuple[sp.Matrix, sp.Matrix, int]:
    """Return off-surface and quotient-reduced Poisson matrices plus constraint rank."""

    if not constraints:
        empty = sp.zeros(0, 0)
        return empty, empty, 0
    phase_variables = tuple(coordinates) + tuple(momenta)
    off_surface = sp.Matrix(
        [
            [poisson_bracket(left, right, coordinates, momenta) for right in constraints]
            for left in constraints
        ]
    )
    on_surface = off_surface.applyfunc(
        lambda item: reduce_on_constraint_surface(item, constraints, phase_variables)
    )
    jacobian = sp.Matrix(constraints).jacobian(phase_variables)
    surface_jacobian = jacobian.applyfunc(
        lambda item: reduce_on_constraint_surface(item, constraints, phase_variables)
    )
    return off_surface, on_surface, int(surface_jacobian.rank())


def partial_velocity_solution(
    hessian: sp.Matrix,
    affine_momenta: sp.Matrix,
    velocities: Sequence[sp.Symbol],
    momenta: Sequence[sp.Symbol],
) -> tuple[dict[sp.Symbol, sp.Expr], tuple[sp.Symbol, ...]]:
    """Invert a maximal nonsingular Hessian minor without imposing primary constraints."""

    rank = int(hessian.rank())
    if rank == 0:
        return {}, tuple(velocities)
    _, pivot_columns = hessian.rref()
    dynamic_columns = tuple(pivot_columns[:rank])
    column_basis = hessian[:, dynamic_columns]
    _, pivot_rows = column_basis.T.rref()
    dynamic_rows = tuple(pivot_rows[:rank])
    minor = hessian.extract(dynamic_rows, dynamic_columns)
    if sp.simplify(minor.det()) == 0:
        raise ValueError("failed to find a nonsingular Hessian rank-profile minor")
    null_columns = tuple(index for index in range(hessian.cols) if index not in dynamic_columns)
    right_hand_side = sp.Matrix(
        [momenta[index] - affine_momenta[index] for index in dynamic_rows]
    )
    if null_columns:
        right_hand_side -= hessian.extract(dynamic_rows, null_columns) * sp.Matrix(
            [velocities[index] for index in null_columns]
        )
    dynamic_solution = minor.inv() * right_hand_side
    solution = {
        velocities[index]: sp.factor(dynamic_solution[position])
        for position, index in enumerate(dynamic_columns)
    }
    return solution, tuple(velocities[index] for index in null_columns)


def analyze_quadratic_lagrangian(
    lagrangian: sp.Expr,
    coordinates: Sequence[sp.Symbol],
    velocities: Sequence[sp.Symbol],
    *,
    momentum_prefix: str = "p",
    max_constraint_generations: int = 6,
) -> DiracResult:
    """Run the finite-mode Dirac algorithm for a Lagrangian at most quadratic in velocities.

    Spatial derivatives may appear as external symbols. The calculation is exact. Constraint
    independence is inferred from the generic symbolic rank, so callers must separately declare
    exceptional parameter surfaces.
    """

    q = tuple(coordinates)
    v = tuple(velocities)
    if len(q) != len(v):
        raise ValueError("coordinates and velocities must have equal length")
    if any(sp.diff(lagrangian, va, vb, vc) != 0 for va in v for vb in v for vc in v):
        raise ValueError("Dirac analyzer currently accepts only velocity-quadratic Lagrangians")
    p = tuple(sp.symbols(f"{momentum_prefix}0:{len(q)}"))
    canonical_momenta = tuple(sp.diff(lagrangian, item) for item in v)
    hessian = sp.hessian(lagrangian, v)
    affine = sp.Matrix(canonical_momenta) - hessian * sp.Matrix(v)
    primary = tuple(
        sp.factor((null.T * (sp.Matrix(p) - affine))[0]) for null in hessian.nullspace()
    )

    velocity_solution, unresolved = partial_velocity_solution(hessian, affine, v, p)
    canonical_hamiltonian = sp.expand(sum(p_i * v_i for p_i, v_i in zip(p, v, strict=True)) - lagrangian)
    canonical_hamiltonian = sp.simplify(canonical_hamiltonian.subs(velocity_solution))
    # Remaining velocities multiply primary constraints. Setting them to zero selects H_c from H_T.
    unresolved_velocities = {item: 0 for item in unresolved}
    canonical_hamiltonian = sp.factor(canonical_hamiltonian.subs(unresolved_velocities))

    phase_variables = q + p
    # Dummy symbols cannot collide with physical coordinates such as the conventional Aether
    # components u0,u1,... . A name collision here can incorrectly turn a genuine higher
    # constraint into a multiplier equation and produce a false closure result.
    multipliers = tuple(sp.Dummy(f"dirac_multiplier_{index}") for index in range(len(primary)))
    total_hamiltonian = canonical_hamiltonian + sum(
        multiplier * constraint for multiplier, constraint in zip(multipliers, primary, strict=True)
    )

    # Dirac-Bergmann consistency starts with d_a+C_ab u^b=0.  Only the projections of d_a
    # along the left null space of the primary Poisson matrix are genuine secondary
    # constraints.  The complementary equations determine primary multipliers.  Treating every
    # nonzero {phi_a,H_c} as secondary gives false constraints whenever primaries are already
    # second class (for example the auxiliary first-order form of a nondegenerate
    # higher-derivative oscillator).
    secondary: list[sp.Expr] = []
    primary_multiplier_conditions: list[sp.Expr] = []
    if primary:
        _, primary_poisson_matrix, _ = reduce_poisson_matrix_on_constraint_surface(
            primary, q, p
        )
        primary_drift = sp.Matrix(
            [
                reduce_on_constraint_surface(
                    poisson_bracket(constraint, canonical_hamiltonian, q, p),
                    primary,
                    phase_variables,
                )
                for constraint in primary
            ]
        )
        primary_multiplier_conditions = [
            sp.factor(item)
            for item in (
                primary_drift + primary_poisson_matrix * sp.Matrix(multipliers)
            )
            if item != 0
        ]
        for null in primary_poisson_matrix.T.nullspace():
            consistency = sp.factor((null.T * primary_drift)[0])
            consistency = reduce_on_constraint_surface(
                consistency, primary, phase_variables
            )
            if consistency != 0 and not any(
                sp.simplify(consistency - old) == 0 for old in secondary
            ):
                secondary.append(consistency)

    generations: list[tuple[sp.Expr, ...]] = [primary]
    if secondary:
        generations.append(tuple(secondary))
    constraints_list = [*primary, *secondary]
    frontier = list(secondary)
    multiplier_conditions: list[sp.Expr] = list(primary_multiplier_conditions)
    generation_limit_reached = False
    for _generation in range(max_constraint_generations):
        if not frontier:
            break
        new_constraints: list[sp.Expr] = []
        for constraint in frontier:
            consistency = sp.factor(poisson_bracket(constraint, total_hamiltonian, q, p))
            consistency = reduce_on_constraint_surface(
                consistency, constraints_list, phase_variables
            )
            if consistency == 0:
                continue
            if any(consistency.has(multiplier) for multiplier in multipliers):
                multiplier_conditions.append(consistency)
            elif not any(
                sp.simplify(consistency - old) == 0
                for old in [*constraints_list, *new_constraints]
            ):
                new_constraints.append(consistency)
        if not new_constraints:
            frontier = []
            break
        generation_tuple = tuple(new_constraints)
        generations.append(generation_tuple)
        constraints_list.extend(new_constraints)
        frontier = new_constraints
    else:
        generation_limit_reached = bool(frontier)

    constraints = tuple(constraints_list)
    reduced_multiplier_conditions = [
        reduce_on_constraint_surface(item, constraints, phase_variables)
        for item in multiplier_conditions
    ]
    reduced_multiplier_conditions = [
        item for item in reduced_multiplier_conditions if item != 0
    ]
    closure_equations = [sp.Eq(item, 0) for item in reduced_multiplier_conditions]
    multiplier_solutions = sp.solve(
        closure_equations, multipliers, dict=True, simplify=True
    )
    multiplier_solution = multiplier_solutions[0] if multiplier_solutions else {}
    closure = not generation_limit_reached and (
        not reduced_multiplier_conditions or bool(multiplier_solutions)
    )
    matrix_off_surface, matrix, independent_constraints = (
        reduce_poisson_matrix_on_constraint_surface(constraints, q, p)
    )
    off_surface_rank = int(matrix_off_surface.rank()) if constraints else 0
    rank = int(matrix.rank()) if constraints else 0
    second_class = rank
    first_class = independent_constraints - rank
    phase_dimension = 2 * len(q)
    dof_numerator = phase_dimension - 2 * first_class - second_class
    if dof_numerator < 0 or dof_numerator % 2:
        raise ValueError("inconsistent constraint count from constraint-surface rank")
    higher = tuple(item for generation in generations[2:] for item in generation)
    return DiracResult(
        coordinates=q,
        velocities=v,
        momenta=p,
        velocity_hessian=hessian,
        canonical_momenta=canonical_momenta,
        primary_constraints=primary,
        canonical_hamiltonian=canonical_hamiltonian,
        secondary_constraints=tuple(secondary),
        higher_generation_constraints=higher,
        constraint_generations=tuple(generations),
        constraint_matrix_off_surface=matrix_off_surface,
        constraint_matrix_off_surface_rank=off_surface_rank,
        constraint_matrix=matrix,
        constraint_matrix_rank=rank,
        independent_constraints=independent_constraints,
        first_class_constraints=first_class,
        second_class_constraints=second_class,
        physical_dof=dof_numerator // 2,
        closure=closure,
        multiplier_solution=multiplier_solution,
        unresolved_consistency_conditions=(
            tuple(reduced_multiplier_conditions) if not closure else ()
        ),
    )


def regular_holonomic_multiplier_dirac_control() -> dict[str, Any]:
    """Exact Dirac theorem for a holonomic constraint with arbitrary kinetic mixing.

    Two unconstrained coordinates are sufficient to prove the dimension-independent local
    structure: the only decisive scalar is the constraint normality
    ``N=C_,A G^AB C_,B``.  The same Pfaffian argument applies for any number of regular
    coordinates because the multiplier chain itself always contains four constraints.
    """

    q0, q1, multiplier = sp.symbols("q0 q1 lambda_C", real=True)
    p0, p1, p_multiplier = sp.symbols("p0 p1 p_lambda_C", real=True)
    coordinates = (q0, q1, multiplier)
    momenta = (p0, p1, p_multiplier)
    g00 = sp.Function("G00")(q0, q1)
    g01 = sp.Function("G01")(q0, q1)
    g11 = sp.Function("G11")(q0, q1)
    inverse_kinetic = sp.Matrix([[g00, g01], [g01, g11]])
    constraint = sp.Function("C")(q0, q1)
    potential = sp.Function("V")(q0, q1)
    physical_momenta = sp.Matrix([p0, p1])
    gradient = sp.Matrix([sp.diff(constraint, q0), sp.diff(constraint, q1)])
    hamiltonian_zero = sp.expand(
        (physical_momenta.T * inverse_kinetic * physical_momenta)[0] / 2
        + potential
    )
    hamiltonian = hamiltonian_zero - multiplier * constraint
    normality = sp.expand((gradient.T * inverse_kinetic * gradient)[0])
    tangency = sp.factor(
        poisson_bracket(constraint, hamiltonian_zero, coordinates, momenta)
    )
    force = sp.factor(
        poisson_bracket(tangency, hamiltonian_zero, coordinates, momenta)
    )
    multiplier_fixing = sp.expand(force + multiplier * normality)
    constraints = (p_multiplier, constraint, tangency, multiplier_fixing)
    poisson_matrix = sp.Matrix(
        [
            [poisson_bracket(left, right, coordinates, momenta) for right in constraints]
            for left in constraints
        ]
    )
    determinant = sp.factor(poisson_matrix.det())
    expected_tangency = sp.expand((gradient.T * inverse_kinetic * physical_momenta)[0])
    primary_consistency = sp.factor(
        poisson_bracket(p_multiplier, hamiltonian, coordinates, momenta)
    )
    secondary_consistency = sp.factor(
        poisson_bracket(constraint, hamiltonian, coordinates, momenta)
    )
    tertiary_consistency = sp.factor(
        poisson_bracket(tangency, hamiltonian, coordinates, momenta)
    )
    final_multiplier_coefficient = sp.factor(
        poisson_bracket(multiplier_fixing, p_multiplier, coordinates, momenta)
    )
    passed = (
        sp.factor(primary_consistency - constraint) == 0
        and sp.factor(secondary_consistency - tangency) == 0
        and sp.factor(tertiary_consistency - multiplier_fixing) == 0
        and sp.factor(tangency - expected_tangency) == 0
        and sp.factor(determinant - normality**4) == 0
        and sp.factor(final_multiplier_coefficient - normality) == 0
    )
    return {
        "passed": passed,
        "representative_regular_coordinates": 2,
        "dimension_independence": (
            "the four-constraint multiplier-chain Pfaffian uses only N=C_,A G^AB C_,B"
        ),
        "inverse_kinetic_metric": str(inverse_kinetic),
        "constraint": str(constraint),
        "potential": str(potential),
        "normality": str(normality),
        "normality_domain": "N != 0",
        "constraint_generations": [
            [str(p_multiplier)],
            [str(constraint)],
            [str(tangency)],
            [str(multiplier_fixing)],
        ],
        "generation_roles": [
            "multiplier momentum primary",
            "holonomic constraint secondary",
            "kinetic-normal tangency tertiary",
            "canonical multiplier-fixing quaternary",
        ],
        "consistency_residuals": [
            str(sp.factor(primary_consistency - constraint)),
            str(sp.factor(secondary_consistency - tangency)),
            str(sp.factor(tertiary_consistency - multiplier_fixing)),
        ],
        "poisson_matrix": str(poisson_matrix),
        "poisson_determinant": str(determinant),
        "expected_poisson_determinant": "N^4",
        "constraint_matrix_rank_on_regular_patch": 4,
        "first_class_constraints": 0,
        "second_class_constraints": 4,
        "final_total_hamiltonian_multiplier_coefficient": str(
            final_multiplier_coefficient
        ),
        "physical_dof_for_n_regular_coordinates": "n-1",
        "interpretation": (
            "arbitrary coordinate dependence, off-diagonal kinetic mixing, and an arbitrary "
            "potential cannot change the four-second-class unit chain while N is nonzero"
        ),
        "scope": (
            "exact local regular-holonomic Dirac theorem; it does not prove any model's lapse/shift "
            "Hamiltonian constraints, hypersurface-deformation algebra, or reduced boundedness"
        ),
    }


def proca_fourier_dirac_control() -> dict[str, Any]:
    """Exact Proca Dirac analysis for a Fourier mode with k aligned to the first axis."""

    a0, a1, a2, a3 = sp.symbols("A0:4", real=True)
    v0, v1, v2, v3 = sp.symbols("v0:4", real=True)
    mass, wave_number = sp.symbols("m k", positive=True, finite=True)
    electric = ((v1 - wave_number * a0) ** 2 + v2**2 + v3**2) / 2
    magnetic = wave_number**2 * (a2**2 + a3**2) / 2
    mass_term = mass**2 * (a0**2 - a1**2 - a2**2 - a3**2) / 2
    lagrangian = sp.expand(electric - magnetic + mass_term)
    result = analyze_quadratic_lagrangian(
        lagrangian,
        (a0, a1, a2, a3),
        (v0, v1, v2, v3),
    )
    secondary = result.secondary_constraints[0]
    a0_solution = sp.solve(sp.Eq(secondary, 0), a0, dict=True)[0]
    reduced_hamiltonian = sp.factor(result.canonical_hamiltonian.subs(a0_solution))
    physical_phase_variables = (a1, a2, a3, result.momenta[1], result.momenta[2], result.momenta[3])
    reduced_hessian = sp.hessian(reduced_hamiltonian, physical_phase_variables)
    positive_diagonal = all(
        reduced_hessian[index, index].is_positive for index in range(reduced_hessian.rows)
    )
    return {
        **result.as_dict(),
        "lagrangian": str(lagrangian),
        "parameter_domain": ["m > 0", "k >= 0", "spatial wave vector aligned by rotational symmetry"],
        "reduced_hamiltonian": str(reduced_hamiltonian),
        "reduced_hamiltonian_hessian": str(reduced_hessian),
        "hamiltonian_positive_definite": bool(
            reduced_hessian.is_diagonal() and positive_diagonal
        ),
    }
