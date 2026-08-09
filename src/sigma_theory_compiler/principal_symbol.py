from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp


@dataclass(frozen=True)
class PrincipalSymbolResult:
    kinetic_matrix: sp.Matrix
    gradient_matrix: sp.Matrix
    principal_polynomial: sp.Expr
    speed_squared: tuple[sp.Expr, ...]
    ghost_free: bool
    gradient_stable: bool
    real_characteristics: bool
    strongly_hyperbolic: bool
    cone_policy_pass: bool
    maximum_speed_squared: sp.Expr | None

    @property
    def passed(self) -> bool:
        return (
            self.ghost_free
            and self.gradient_stable
            and self.real_characteristics
            and self.strongly_hyperbolic
            and self.cone_policy_pass
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "kinetic_matrix": str(self.kinetic_matrix),
            "gradient_matrix": str(self.gradient_matrix),
            "principal_polynomial": str(self.principal_polynomial),
            "speed_squared": [str(item) for item in self.speed_squared],
            "ghost_free": self.ghost_free,
            "gradient_stable": self.gradient_stable,
            "real_characteristics": self.real_characteristics,
            "strongly_hyperbolic": self.strongly_hyperbolic,
            "cone_policy_pass": self.cone_policy_pass,
            "maximum_speed_squared": (
                None if self.maximum_speed_squared is None else str(self.maximum_speed_squared)
            ),
            "passed": self.passed,
        }


def _is_real_nonnegative(value: sp.Expr) -> bool:
    simplified = sp.simplify(value)
    return simplified.is_real is True and simplified.is_nonnegative is True


def analyze_isotropic_second_order_symbol(
    kinetic_matrix: sp.Matrix,
    gradient_matrix: sp.Matrix,
    *,
    maximum_speed_squared: sp.Expr | float = 1,
) -> PrincipalSymbolResult:
    """Analyze L2 = 1/2 dot(u)^T K dot(u) - 1/2 grad(u)^T G grad(u).

    The caller must first remove gauge and constrained variables. This function deliberately rejects
    a singular K rather than pretending the unreduced determinant is a physical characteristic
    polynomial.
    """

    kinetic = sp.Matrix(kinetic_matrix)
    gradient = sp.Matrix(gradient_matrix)
    if kinetic.rows != kinetic.cols or gradient.shape != kinetic.shape:
        raise ValueError("kinetic and gradient matrices must be square with equal shape")
    if kinetic != kinetic.T or gradient != gradient.T:
        raise ValueError("quadratic kinetic and gradient matrices must be symmetric")
    omega, wave_number = sp.symbols("omega k", real=True)
    principal = sp.factor((-omega**2 * kinetic + wave_number**2 * gradient).det())
    ghost_free = kinetic.is_positive_definite is True
    if kinetic.det() == 0:
        return PrincipalSymbolResult(
            kinetic,
            gradient,
            principal,
            (),
            False,
            False,
            False,
            False,
            False,
            None,
        )
    propagation = sp.simplify(kinetic.inv() * gradient)
    eigenvalues = propagation.eigenvals()
    speeds: list[sp.Expr] = []
    for value, multiplicity in eigenvalues.items():
        speeds.extend([sp.simplify(value)] * int(multiplicity))
    speeds.sort(key=sp.default_sort_key)
    real_characteristics = all(value.is_real is True for value in speeds)
    gradient_stable = all(_is_real_nonnegative(value) for value in speeds)
    eigenvector_count = sum(len(vectors) for _, _, vectors in propagation.eigenvects())
    diagonalizable = eigenvector_count == kinetic.rows
    # A symmetric hyperbolic second-order system follows when K is positive definite, G is
    # symmetric, and the reduced propagation matrix has a complete real eigenbasis.
    strongly_hyperbolic = ghost_free and real_characteristics and diagonalizable
    speed_cap = sp.sympify(maximum_speed_squared)
    cone_pass = all(sp.simplify(speed_cap - value).is_nonnegative is True for value in speeds)
    max_speed = max(speeds, key=lambda item: float(sp.N(item))) if speeds else None
    return PrincipalSymbolResult(
        kinetic,
        gradient,
        principal,
        tuple(speeds),
        ghost_free,
        gradient_stable,
        real_characteristics,
        strongly_hyperbolic,
        cone_pass,
        max_speed,
    )


def analyze_anisotropic_second_order_symbol(
    kinetic_matrix: sp.Matrix,
    gradient_blocks: Sequence[Sequence[sp.Matrix]],
    directions: Sequence[Sequence[sp.Expr | float]],
    *,
    maximum_speed_squared: sp.Expr | float = 1,
) -> dict[str, Any]:
    """Analyze exact directional symbols G(n)=G^{ij} n_i n_j / |n|^2.

    A finite direction set is a falsification control, not a proof over the full unit sphere.
    """

    kinetic = sp.Matrix(kinetic_matrix)
    spatial_dimension = len(gradient_blocks)
    if spatial_dimension == 0 or any(
        len(row) != spatial_dimension for row in gradient_blocks
    ):
        raise ValueError("gradient_blocks must be a nonempty square spatial block matrix")
    blocks = [
        [sp.Matrix(gradient_blocks[i][j]) for j in range(spatial_dimension)]
        for i in range(spatial_dimension)
    ]
    for i in range(spatial_dimension):
        for j in range(spatial_dimension):
            if blocks[i][j].shape != kinetic.shape:
                raise ValueError("every anisotropic gradient block must match the kinetic matrix")
            if blocks[i][j] != blocks[j][i].T:
                raise ValueError("anisotropic blocks must obey G^{ij}=(G^{ji})^T")
    directional_results: list[dict[str, Any]] = []
    for raw_direction in directions:
        if len(raw_direction) != spatial_dimension:
            raise ValueError("each direction must match the number of spatial dimensions")
        direction = tuple(sp.sympify(item) for item in raw_direction)
        norm_squared = sp.factor(sum(item**2 for item in direction))
        if norm_squared == 0:
            raise ValueError("the zero spatial direction has no principal cone")
        directional_gradient = sp.zeros(*kinetic.shape)
        for i in range(spatial_dimension):
            for j in range(spatial_dimension):
                directional_gradient += direction[i] * direction[j] * blocks[i][j]
        directional_gradient = sp.simplify(directional_gradient / norm_squared)
        result = analyze_isotropic_second_order_symbol(
            kinetic,
            directional_gradient,
            maximum_speed_squared=maximum_speed_squared,
        )
        directional_results.append(
            {
                "direction": [str(item) for item in direction],
                "norm_squared": str(norm_squared),
                "directional_gradient_matrix": str(directional_gradient),
                **result.as_dict(),
            }
        )
    return {
        "spatial_dimension": spatial_dimension,
        "directions_tested": len(directional_results),
        "directional_results": directional_results,
        "passed": bool(directional_results)
        and all(item["passed"] for item in directional_results),
        "scope": (
            "exact declared-direction anisotropic falsification; finite directions do not prove "
            "strong hyperbolicity uniformly over the sphere"
        ),
    }


def run_anisotropic_principal_symbol_controls() -> dict[str, Any]:
    directions = (
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, -1, 0),
        (1, 1, 1),
    )
    zero = sp.zeros(1)
    isotropic_blocks = [
        [sp.eye(1) if i == j else zero for j in range(3)] for i in range(3)
    ]
    isotropic = analyze_anisotropic_second_order_symbol(
        sp.eye(1), isotropic_blocks, directions
    )
    off_axis_unstable_blocks = [
        [sp.eye(1) if i == j else sp.zeros(1) for j in range(3)] for i in range(3)
    ]
    off_axis_unstable_blocks[0][1] = sp.Matrix([[2]])
    off_axis_unstable_blocks[1][0] = sp.Matrix([[2]])
    off_axis_unstable = analyze_anisotropic_second_order_symbol(
        sp.eye(1), off_axis_unstable_blocks, directions
    )
    axes_pass = all(
        item["passed"]
        for item in off_axis_unstable["directional_results"]
        if item["direction"] in (["1", "0", "0"], ["0", "1", "0"], ["0", "0", "1"])
    )
    oblique_failure = next(
        item
        for item in off_axis_unstable["directional_results"]
        if item["direction"] == ["1", "-1", "0"]
    )
    passed = (
        isotropic["passed"]
        and axes_pass
        and not off_axis_unstable["passed"]
        and not oblique_failure["gradient_stable"]
    )
    return {
        "passed": passed,
        "directions": [[str(item) for item in direction] for direction in directions],
        "isotropic_metric_cone": isotropic,
        "off_axis_gradient_negative_control": off_axis_unstable,
        "negative_control_axes_pass": axes_pass,
        "negative_control_oblique_failure": oblique_failure,
    }


def extract_reduced_quadratic_principal_blocks(
    lagrangian: sp.Expr,
    velocities: Sequence[sp.Symbol],
    spatial_gradients: Sequence[Sequence[sp.Symbol]],
) -> dict[str, Any]:
    """Extract K, G^{ij}, and unsupported time-space mixed principal blocks."""

    velocity_tuple = tuple(velocities)
    gradients = tuple(tuple(row) for row in spatial_gradients)
    if len(gradients) != len(velocity_tuple) or not gradients:
        raise ValueError("one spatial-gradient row is required per reduced field")
    spatial_dimension = len(gradients[0])
    if spatial_dimension == 0 or any(len(row) != spatial_dimension for row in gradients):
        raise ValueError("spatial-gradient rows must have one common nonzero dimension")
    if any(
        sp.diff(lagrangian, first, second, third) != 0
        for first in velocity_tuple
        for second in velocity_tuple
        for third in velocity_tuple
    ):
        raise ValueError("principal extraction requires a velocity-quadratic Lagrangian")
    kinetic = sp.hessian(lagrangian, velocity_tuple)
    blocks: list[list[sp.Matrix]] = []
    for i in range(spatial_dimension):
        row: list[sp.Matrix] = []
        for j in range(spatial_dimension):
            row.append(
                sp.Matrix(
                    len(velocity_tuple),
                    len(velocity_tuple),
                    lambda a, b, i=i, j=j: -sp.diff(
                        lagrangian, gradients[a][i], gradients[b][j]
                    ),
                )
            )
        blocks.append(row)
    mixed_blocks = [
        sp.Matrix(
            len(velocity_tuple),
            len(velocity_tuple),
            lambda a, b, i=i: sp.diff(
                lagrangian, velocity_tuple[a], gradients[b][i]
            ),
        )
        for i in range(spatial_dimension)
    ]
    mixed_present = any(block != sp.zeros(*block.shape) for block in mixed_blocks)
    return {
        "kinetic_matrix": kinetic,
        "gradient_blocks": blocks,
        "time_space_mixed_blocks": mixed_blocks,
        "time_space_mixed_present": mixed_present,
        "field_count": len(velocity_tuple),
        "spatial_dimension": spatial_dimension,
        "isotropic_reduction_supported": not mixed_present,
        "general_matrix_polynomial_supported": True,
        "supported": True,
    }


def analyze_general_directional_matrix_polynomial(
    kinetic_matrix: sp.Matrix,
    gradient_blocks: Sequence[Sequence[sp.Matrix]],
    mixed_blocks: Sequence[sp.Matrix],
    directions: Sequence[Sequence[sp.Expr | float]],
    *,
    maximum_speed_squared: sp.Expr | float = 1,
) -> dict[str, Any]:
    """Analyze -c^2 K + c(B(n)+B(n)^T) + G(n) by companion linearization."""

    kinetic = sp.Matrix(kinetic_matrix)
    spatial_dimension = len(gradient_blocks)
    if len(mixed_blocks) != spatial_dimension:
        raise ValueError("one time-space mixed block is required per spatial dimension")
    if kinetic.det() == 0:
        return {
            "passed": False,
            "status": "unresolved",
            "reason": "singular kinetic matrix requires constraint/gauge reduction",
            "directional_results": [],
        }
    ghost_free = kinetic.is_positive_definite is True
    cap_squared = sp.sympify(maximum_speed_squared)
    field_count = kinetic.rows
    identity = sp.eye(field_count)
    zero = sp.zeros(field_count)
    results: list[dict[str, Any]] = []
    for raw_direction in directions:
        if len(raw_direction) != spatial_dimension:
            raise ValueError("each direction must match the number of spatial dimensions")
        direction = tuple(sp.sympify(item) for item in raw_direction)
        norm_squared = sp.factor(sum(item**2 for item in direction))
        if norm_squared == 0:
            raise ValueError("the zero spatial direction has no principal cone")
        norm = sp.sqrt(norm_squared)
        directional_gradient = sp.zeros(field_count)
        directional_mixed = sp.zeros(field_count)
        for i in range(spatial_dimension):
            directional_mixed += direction[i] * sp.Matrix(mixed_blocks[i]) / norm
            for j in range(spatial_dimension):
                directional_gradient += (
                    direction[i]
                    * direction[j]
                    * sp.Matrix(gradient_blocks[i][j])
                    / norm_squared
                )
        symmetric_mixed = sp.simplify(directional_mixed + directional_mixed.T)
        directional_gradient = sp.simplify(directional_gradient)
        inverse_kinetic = kinetic.inv()
        companion = sp.Matrix.vstack(
            sp.Matrix.hstack(zero, identity),
            sp.Matrix.hstack(
                inverse_kinetic * directional_gradient,
                inverse_kinetic * symmetric_mixed,
            ),
        )
        eigenvalues = companion.eigenvals()
        speeds: list[sp.Expr] = []
        for value, multiplicity in eigenvalues.items():
            speeds.extend([sp.simplify(value)] * int(multiplicity))
        speeds.sort(key=sp.default_sort_key)
        real = all(value.is_real is True for value in speeds)
        eigenvector_count = sum(len(vectors) for _, _, vectors in companion.eigenvects())
        complete = eigenvector_count == 2 * field_count
        cone_pass = real and all(
            sp.simplify(cap_squared - value**2).is_nonnegative is True for value in speeds
        )
        speed_squares = [sp.simplify(value**2) for value in speeds]
        results.append(
            {
                "direction": [str(item) for item in direction],
                "norm_squared": str(norm_squared),
                "directional_gradient_matrix": str(directional_gradient),
                "symmetric_mixed_matrix": str(symmetric_mixed),
                "companion_matrix": str(companion),
                "characteristic_speeds": [str(item) for item in speeds],
                "characteristic_speed_squared": [str(item) for item in speed_squares],
                "real_characteristics": real,
                "complete_eigenbasis": complete,
                "strongly_hyperbolic": ghost_free and real and complete,
                "cone_policy_pass": cone_pass,
                "passed": ghost_free and real and complete and cone_pass,
            }
        )
    return {
        "passed": bool(results) and all(item["passed"] for item in results),
        "status": "pass" if results and all(item["passed"] for item in results) else "reject",
        "ghost_free": ghost_free,
        "directions_tested": len(results),
        "directional_results": results,
        "scope": (
            "exact finite-direction quadratic matrix-polynomial characteristics; uniform sphere "
            "bounds and constrained/gauge reduction remain separate"
        ),
    }


def analyze_reduced_quadratic_lagrangian_symbol(
    lagrangian: sp.Expr,
    velocities: Sequence[sp.Symbol],
    spatial_gradients: Sequence[Sequence[sp.Symbol]],
    directions: Sequence[Sequence[sp.Expr | float]],
    *,
    maximum_speed_squared: sp.Expr | float = 1,
) -> dict[str, Any]:
    extraction = extract_reduced_quadratic_principal_blocks(
        lagrangian, velocities, spatial_gradients
    )
    serializable_extraction = {
        "kinetic_matrix": str(extraction["kinetic_matrix"]),
        "gradient_blocks": [
            [str(block) for block in row] for row in extraction["gradient_blocks"]
        ],
        "time_space_mixed_blocks": [
            str(block) for block in extraction["time_space_mixed_blocks"]
        ],
        "time_space_mixed_present": extraction["time_space_mixed_present"],
        "field_count": extraction["field_count"],
        "spatial_dimension": extraction["spatial_dimension"],
        "isotropic_reduction_supported": extraction["isotropic_reduction_supported"],
        "general_matrix_polynomial_supported": extraction[
            "general_matrix_polynomial_supported"
        ],
        "supported": extraction["supported"],
    }
    if extraction["time_space_mixed_present"]:
        analysis = analyze_general_directional_matrix_polynomial(
            extraction["kinetic_matrix"],
            extraction["gradient_blocks"],
            extraction["time_space_mixed_blocks"],
            directions,
            maximum_speed_squared=maximum_speed_squared,
        )
    else:
        analysis = analyze_anisotropic_second_order_symbol(
            extraction["kinetic_matrix"],
            extraction["gradient_blocks"],
            directions,
            maximum_speed_squared=maximum_speed_squared,
        )
    return {
        "passed": analysis["passed"],
        "status": "pass" if analysis["passed"] else "reject",
        "extraction": serializable_extraction,
        "analysis": analysis,
    }


def run_extracted_principal_symbol_controls() -> dict[str, Any]:
    velocity = sp.symbols("v")
    gradients = sp.symbols("gx gy gz")
    directions = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, -1, 0), (1, 1, 1))
    canonical_lagrangian = (
        velocity**2 - sum(gradient**2 for gradient in gradients)
    ) / 2
    canonical = analyze_reduced_quadratic_lagrangian_symbol(
        canonical_lagrangian, (velocity,), (gradients,), directions
    )
    mixed_lagrangian = canonical_lagrangian + velocity * gradients[0]
    mixed = analyze_reduced_quadratic_lagrangian_symbol(
        mixed_lagrangian,
        (velocity,),
        (gradients,),
        directions,
        maximum_speed_squared=6,
    )
    defective_mixed_lagrangian = (
        velocity**2
        + 2 * velocity * gradients[0]
        + gradients[0] ** 2
        - gradients[1] ** 2
        - gradients[2] ** 2
    ) / 2
    defective_mixed = analyze_reduced_quadratic_lagrangian_symbol(
        defective_mixed_lagrangian,
        (velocity,),
        (gradients,),
        directions,
        maximum_speed_squared=6,
    )
    defective_axis = defective_mixed["analysis"]["directional_results"][0]
    passed = (
        canonical["passed"]
        and canonical["extraction"]["kinetic_matrix"] == "Matrix([[1]])"
        and mixed["passed"]
        and mixed["status"] == "pass"
        and mixed["extraction"]["time_space_mixed_present"]
        and not defective_mixed["passed"]
        and not defective_axis["complete_eigenbasis"]
    )
    return {
        "passed": passed,
        "canonical_scalar": canonical,
        "time_space_mixed_characteristics": mixed,
        "defective_mixed_negative_control": defective_mixed,
        "scope": (
            "automatic reduced quadratic K/G^{ij} extraction; gauge reduction, nonlinear "
            "background substitution, and general omega-k mixed symbols remain separate"
        ),
    }


def analyze_uniform_scalar_anisotropy(
    kinetic_coefficient: sp.Expr | float,
    spatial_anisotropy_matrix: sp.Matrix,
    *,
    maximum_speed_squared: sp.Expr | float = 1,
) -> dict[str, Any]:
    """Prove scalar-sector speed bounds uniformly over the Euclidean direction sphere."""

    kinetic = sp.sympify(kinetic_coefficient)
    spatial = sp.Matrix(spatial_anisotropy_matrix)
    if spatial.rows != spatial.cols or spatial != spatial.T:
        raise ValueError("the scalar spatial anisotropy matrix must be symmetric and square")
    if kinetic.is_positive is not True:
        return {
            "passed": False,
            "ghost_free": False,
            "reason": "kinetic coefficient is not provably positive",
        }
    eigenvalues: list[sp.Expr] = []
    for value, multiplicity in spatial.eigenvals().items():
        eigenvalues.extend([sp.simplify(value / kinetic)] * int(multiplicity))
    eigenvalues.sort(key=sp.default_sort_key)
    real = all(item.is_real is True for item in eigenvalues)
    gradient_stable = real and all(item.is_nonnegative is True for item in eigenvalues)
    speed_cap = sp.sympify(maximum_speed_squared)
    cone_pass = real and all(
        sp.simplify(speed_cap - item).is_nonnegative is True for item in eigenvalues
    )
    minimum = min(eigenvalues, key=lambda item: float(sp.N(item))) if eigenvalues else None
    maximum = max(eigenvalues, key=lambda item: float(sp.N(item))) if eigenvalues else None
    return {
        "passed": gradient_stable and cone_pass,
        "ghost_free": True,
        "spatial_anisotropy_matrix": str(spatial),
        "principal_speed_squared_eigenvalues": [str(item) for item in eigenvalues],
        "minimum_speed_squared": None if minimum is None else str(minimum),
        "maximum_speed_squared": None if maximum is None else str(maximum),
        "real_characteristics": real,
        "gradient_stable": gradient_stable,
        "cone_policy_pass": cone_pass,
        "scope": (
            "exact uniform scalar-sector result over every nonzero spatial direction, using the "
            "Rayleigh quotient extrema of the symmetric anisotropy matrix"
        ),
    }


def run_uniform_scalar_anisotropy_controls() -> dict[str, Any]:
    metric_cone = analyze_uniform_scalar_anisotropy(1, sp.eye(3))
    stable_anisotropic = analyze_uniform_scalar_anisotropy(
        1, sp.diag(sp.Rational(1, 4), sp.Rational(1, 2), 1)
    )
    off_axis_unstable = analyze_uniform_scalar_anisotropy(
        1, sp.Matrix([[1, 2, 0], [2, 1, 0], [0, 0, 1]])
    )
    ghost = analyze_uniform_scalar_anisotropy(-1, sp.eye(3))
    passed = (
        metric_cone["passed"]
        and stable_anisotropic["passed"]
        and stable_anisotropic["minimum_speed_squared"] == "1/4"
        and stable_anisotropic["maximum_speed_squared"] == "1"
        and not off_axis_unstable["passed"]
        and not off_axis_unstable["gradient_stable"]
        and not ghost["passed"]
        and not ghost["ghost_free"]
    )
    return {
        "passed": passed,
        "metric_cone": metric_cone,
        "stable_anisotropic": stable_anisotropic,
        "off_axis_unstable": off_axis_unstable,
        "ghost": ghost,
    }


def analyze_uniform_multifield_block_certificate(
    kinetic_matrix: sp.Matrix,
    gradient_blocks: Sequence[Sequence[sp.Matrix]],
    *,
    maximum_speed_squared: sp.Expr | float = 1,
) -> dict[str, Any]:
    """Certify all-direction multi-field stability using spatial-field block PSD bounds.

    Positive semidefiniteness of the full block matrices is sufficient, but not necessary, for the
    associated biquadratic forms on rank-one direction-field products. An inconclusive certificate
    therefore remains unresolved rather than becoming a rejection of the underlying theory.
    """

    kinetic = sp.Matrix(kinetic_matrix)
    spatial_dimension = len(gradient_blocks)
    if kinetic.rows != kinetic.cols:
        raise ValueError("kinetic matrix must be square")
    if spatial_dimension == 0 or any(
        len(row) != spatial_dimension for row in gradient_blocks
    ):
        raise ValueError("gradient blocks must form a nonempty square spatial matrix")
    blocks = [
        [sp.Matrix(gradient_blocks[i][j]) for j in range(spatial_dimension)]
        for i in range(spatial_dimension)
    ]
    for i in range(spatial_dimension):
        for j in range(spatial_dimension):
            if blocks[i][j].shape != kinetic.shape:
                raise ValueError("every gradient block must match the kinetic matrix")
            if blocks[i][j] != blocks[j][i].T:
                raise ValueError("gradient blocks must obey G^{ij}=(G^{ji})^T")
    full_gradient = sp.Matrix.vstack(
        *[sp.Matrix.hstack(*blocks[i]) for i in range(spatial_dimension)]
    )
    cap = sp.sympify(maximum_speed_squared)
    full_kinetic = sp.kronecker_product(sp.eye(spatial_dimension), kinetic)
    cone_remainder = sp.simplify(cap * full_kinetic - full_gradient)
    ghost_free = kinetic.is_positive_definite is True
    gradient_certificate = full_gradient.is_positive_semidefinite is True
    cone_certificate = cone_remainder.is_positive_semidefinite is True
    certificate_conclusive = ghost_free and gradient_certificate and cone_certificate
    return {
        "passed": certificate_conclusive,
        "status": "pass" if certificate_conclusive else "unresolved",
        "field_count": kinetic.rows,
        "spatial_dimension": spatial_dimension,
        "kinetic_matrix": str(kinetic),
        "full_spatial_field_gradient_matrix": str(full_gradient),
        "cone_remainder_matrix": str(cone_remainder),
        "ghost_free": ghost_free,
        "gradient_block_positive_semidefinite": gradient_certificate,
        "cone_block_positive_semidefinite": cone_certificate,
        "uniform_strong_hyperbolicity": certificate_conclusive,
        "certificate_kind": "sufficient spatial-field block PSD certificate",
        "scope": (
            "exact all-direction multi-field certificate for symmetric no-mixed-term systems; "
            "failure is inconclusive because block PSD is stronger than rank-one biquadratic PSD"
        ),
    }


def run_uniform_multifield_block_controls() -> dict[str, Any]:
    kinetic = sp.eye(2)
    zero = sp.zeros(2)
    stable_field_gradient = sp.diag(1, sp.Rational(1, 2))
    stable_blocks = [
        [stable_field_gradient if i == j else zero for j in range(3)]
        for i in range(3)
    ]
    stable = analyze_uniform_multifield_block_certificate(kinetic, stable_blocks)

    off_axis_blocks = [
        [sp.eye(2) if i == j else sp.zeros(2) for j in range(3)] for i in range(3)
    ]
    off_axis_blocks[0][1] = sp.diag(2, 0)
    off_axis_blocks[1][0] = sp.diag(2, 0)
    off_axis = analyze_uniform_multifield_block_certificate(kinetic, off_axis_blocks)

    superluminal_blocks = [
        [2 * sp.eye(2) if i == j else sp.zeros(2) for j in range(3)]
        for i in range(3)
    ]
    superluminal = analyze_uniform_multifield_block_certificate(
        kinetic, superluminal_blocks
    )
    passed = (
        stable["passed"]
        and stable["uniform_strong_hyperbolicity"]
        and not off_axis["passed"]
        and not off_axis["gradient_block_positive_semidefinite"]
        and not superluminal["passed"]
        and not superluminal["cone_block_positive_semidefinite"]
    )
    return {
        "passed": passed,
        "stable_two_field": stable,
        "off_axis_inconclusive_negative_control": off_axis,
        "superluminal_inconclusive_negative_control": superluminal,
    }


def run_principal_symbol_controls() -> dict[str, Any]:
    scalar = analyze_isotropic_second_order_symbol(sp.eye(1), sp.eye(1))
    proca_reduced = analyze_isotropic_second_order_symbol(sp.eye(3), sp.eye(3))
    ghost = analyze_isotropic_second_order_symbol(sp.diag(1, -1), sp.eye(2))
    gradient = analyze_isotropic_second_order_symbol(sp.eye(2), sp.diag(1, -1))
    superluminal = analyze_isotropic_second_order_symbol(sp.eye(1), sp.Matrix([[4]]))
    passed = (
        scalar.passed
        and proca_reduced.passed
        and not ghost.passed
        and not ghost.ghost_free
        and not gradient.passed
        and not gradient.gradient_stable
        and not superluminal.passed
        and not superluminal.cone_policy_pass
    )
    return {
        "passed": passed,
        "convention": "L2 = dot(u)^T K dot(u)/2 - partial_i(u)^T G partial_i(u)/2",
        "cone_policy": "0 <= c_mode^2 <= 1 relative to the physical metric cone",
        "scope": "gauge/constraint-reduced isotropic second-order systems on a frozen local background",
        "controls": {
            "canonical_scalar": scalar.as_dict(),
            "reduced_proca": proca_reduced.as_dict(),
            "negative_kinetic_ghost": ghost.as_dict(),
            "negative_gradient": gradient.as_dict(),
            "superluminal_cone": superluminal.as_dict(),
        },
    }
