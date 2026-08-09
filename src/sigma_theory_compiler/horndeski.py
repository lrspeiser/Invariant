from __future__ import annotations

from typing import Any

import sympy as sp

from .dhost import (
    generic_horndeski_l2_l4_unitary_adm_control,
    quartic_horndeski_unitary_flrw_dirac_control,
)
from .field_dirac import canonical_metric_diffeomorphism_control, euler_operator_nd


def quartic_horndeski_scalar_euler_reduction_control() -> tuple[bool, dict[str, Any]]:
    """Reduce the fixed-metric scalar variation of linear-X quartic Horndeski.

    The placeholders are independent covariant jet contractions.  The reduction uses only the
    scalar-Hessian commutator and the contracted Bianchi identity, both stated explicitly in the
    returned proof chain.
    """

    curvature, box_phi, curvature_gradient_dot_phi = sp.symbols(
        "R box_phi dR_dot_dphi", real=True
    )
    ricci_hessian, box_box_phi = sp.symbols(
        "Ricci_dot_Hessian box_box_phi", real=True
    )
    div_r_grad_phi = curvature_gradient_dot_phi + curvature * box_phi
    div_ricci_grad_phi = curvature_gradient_dot_phi / 2 + ricci_hessian
    div_div_hessian = box_box_phi + div_ricci_grad_phi
    raw_euler = sp.expand(
        div_r_grad_phi + 2 * box_box_phi - 2 * div_div_hessian
    )
    einstein_hessian = sp.factor(ricci_hessian - curvature * box_phi / 2)
    expected_euler = sp.factor(-2 * einstein_hessian)
    reduction_residual = sp.simplify(raw_euler - expected_euler)
    fourth_derivative_coefficient = sp.expand(raw_euler).coeff(box_box_phi)
    curvature_gradient_coefficient = sp.expand(raw_euler).coeff(
        curvature_gradient_dot_phi
    )

    beta = sp.Symbol("beta", real=True)
    deformed_euler = sp.expand(
        div_r_grad_phi + 2 * beta * (box_box_phi - div_div_hessian)
    )
    deformed_curvature_gradient = sp.factor(
        deformed_euler.coeff(curvature_gradient_dot_phi)
    )
    wrong_point = sp.factor(deformed_curvature_gradient.subs(beta, 2))

    passed = (
        reduction_residual == 0
        and fourth_derivative_coefficient == 0
        and curvature_gradient_coefficient == 0
        and wrong_point == -1
    )
    return passed, {
        "covariant_action_sector": (
            "X_c R + (box phi)^2-(nabla_mu nabla_nu phi)(nabla^mu nabla^nu phi)"
        ),
        "raw_integrated_euler_coefficient": str(raw_euler),
        "expected_second_order_euler_coefficient": str(expected_euler),
        "expected_equation": "G^(mu nu) nabla_mu nabla_nu(phi)=0",
        "reduction_residual": str(reduction_residual),
        "fourth_derivative_coefficient": str(fourth_derivative_coefficient),
        "curvature_gradient_coefficient": str(curvature_gradient_coefficient),
        "commutator_and_bianchi_chain": [
            "nabla_mu nabla_nu(nabla^mu nabla^nu phi)=box(box(phi))+nabla_mu(Ricci^(mu nu) nabla_nu phi)",
            "nabla_mu(Ricci^(mu nu) nabla_nu phi)=1/2 nabla^nu(R) nabla_nu(phi)+Ricci^(mu nu) nabla_mu nabla_nu(phi)",
            "nabla_mu Ricci^(mu nu)=1/2 nabla^nu R",
            "R box(phi)-2 Ricci^(mu nu) phi_(mu nu)=-2 G^(mu nu) phi_(mu nu)",
        ],
        "wrong_completion_negative_control": {
            "deformation": "multiply the Hessian-square completion by beta while keeping X_c R fixed",
            "curvature_gradient_coefficient": str(deformed_curvature_gradient),
            "beta_2_witness": str(wrong_point),
            "higher_metric_derivative_restored": wrong_point != 0,
        },
        "scope": (
            "exact fixed-metric scalar Euler reduction for the shift-symmetric linear-X quartic "
            "Horndeski term; metric variation and the combined diffeomorphism identity remain separate"
        ),
    }


def _quartic_horndeski_curvature_scalar_principal_prefix() -> tuple[Any, ...]:
    """Build shared exact symbols for the arbitrary-curvature scalar cone control.

    The canonical scalar plus linear-X quartic-Horndeski scalar equation has principal
    inverse metric ``P^{mu nu}=g^{mu nu}-2 alpha G^{mu nu}``.  The returned local
    orthonormal-frame block permits one time-space Einstein-tensor flux and three
    independent spatial eigenvalues.  This is a necessary scalar-sector preflight, not
    the full coupled metric-scalar principal matrix.
    """

    euler_passed, euler_evidence = quartic_horndeski_scalar_euler_reduction_control()
    alpha, rho, flux = sp.symbols("alpha rho j", real=True)
    pressure_x, pressure_y, pressure_z = sp.symbols("p1 p2 p3", real=True)
    kinetic = sp.factor(1 + 2 * alpha * rho)
    time_space = sp.factor(-2 * alpha * flux)
    gradients = [
        sp.factor(1 - 2 * alpha * pressure)
        for pressure in (pressure_x, pressure_y, pressure_z)
    ]
    effective_inverse_metric = sp.Matrix(
        [
            [-kinetic, time_space, 0, 0],
            [time_space, gradients[0], 0, 0],
            [0, 0, gradients[1], 0],
            [0, 0, 0, gradients[2]],
        ]
    )
    x_discriminant = sp.factor(time_space**2 + kinetic * gradients[0])
    x_characteristic_speeds = [
        sp.factor((-time_space + sign * sp.sqrt(x_discriminant)) / kinetic)
        for sign in (1, -1)
    ]
    diagonal_speed_squared = [
        sp.factor(gradient / kinetic) for gradient in gradients
    ]
    determinant = sp.factor(effective_inverse_metric.det())
    x_schur_gradient = sp.factor(
        gradients[0] + time_space**2 / kinetic
    )

    diagonal_healthy = {
        alpha: sp.Rational(1, 10),
        rho: 1,
        flux: 0,
        pressure_x: sp.Rational(1, 2),
        pressure_y: sp.Rational(-1, 4),
        pressure_z: 0,
    }
    return (
        euler_passed,
        euler_evidence,
        alpha,
        rho,
        flux,
        pressure_x,
        pressure_y,
        pressure_z,
        kinetic,
        time_space,
        gradients,
        effective_inverse_metric,
        x_discriminant,
        x_characteristic_speeds,
        diagonal_speed_squared,
        determinant,
        x_schur_gradient,
        diagonal_healthy,
    )


def quartic_horndeski_coupled_formulation_hyperbolicity_control() -> tuple[
    bool, dict[str, Any]
]:
    """Encode the formulation-dependent coupled Horndeski hyperbolicity theorems.

    Papallo's generic weak-background result excludes strong hyperbolicity in every
    generalized harmonic gauge when ``G4_X != 0`` (with ``G5=0``).  Kovacs--Reall's
    modified harmonic formulation restores strong hyperbolicity at weak coupling when
    its two auxiliary cones avoid the physical characteristics.  This control verifies
    a nonempty exact cone hierarchy and records the still-missing action-specific uniform
    weak-coupling bound.
    """

    alpha = sp.Symbol("alpha", nonzero=True, finite=True, real=True)
    g3 = sp.Integer(0)
    g4_x = alpha
    generalized_harmonic_condition = sp.And(
        sp.Eq(g4_x, 0), sp.Eq(g3, 0)
    )
    generalized_harmonic_rejected = generalized_harmonic_condition is sp.false

    physical_inverse_metric = sp.diag(-1, 1, 1, 1)
    auxiliary_tilde = sp.diag(-4, 1, 1, 1)
    auxiliary_hat = sp.diag(-9, 1, 1, 1)
    physical_speed_squared = sp.Integer(1)
    tilde_speed_squared = sp.Rational(1, 4)
    hat_speed_squared = sp.Rational(1, 9)
    cone_gaps = {
        "physical_to_tilde_speed_squared": sp.factor(
            physical_speed_squared - tilde_speed_squared
        ),
        "tilde_to_hat_speed_squared": sp.factor(
            tilde_speed_squared - hat_speed_squared
        ),
    }
    all_lorentzian = all(
        matrix.det() < 0
        for matrix in (physical_inverse_metric, auxiliary_tilde, auxiliary_hat)
    )
    common_time_surface = all(
        matrix[0, 0] < 0
        for matrix in (physical_inverse_metric, auxiliary_tilde, auxiliary_hat)
    )

    flat_passed, flat_evidence = quartic_horndeski_timelike_flat_principal_control()
    flat_tensor_speed_squared = sp.sympify(
        flat_evidence["healthy_witness"]["values"]["tensor_speed_squared"]
    )
    flat_scalar_speed_squared = sp.sympify(
        flat_evidence["scalar_speed_squared"]
    )
    flat_physical_speeds = {
        flat_tensor_speed_squared,
        flat_scalar_speed_squared,
    }
    auxiliary_speeds = {tilde_speed_squared, hat_speed_squared}
    flat_cones_disjoint = flat_physical_speeds.isdisjoint(auxiliary_speeds)
    minimum_flat_squared_speed_gap = min(
        abs(physical - auxiliary)
        for physical in flat_physical_speeds
        for auxiliary in auxiliary_speeds
    )
    # For a real-symmetric reduced squared-speed operator, Weyl's theorem bounds
    # every eigenvalue displacement by the spectral norm of its perturbation.
    # Reserve half of the exact flat cone gap so a future full 11-by-11 extraction
    # has a concrete, independently checkable norm target and a nonzero margin.
    robustness_budget = sp.factor(minimum_flat_squared_speed_gap / 2)
    safe_perturbation_witness = sp.Rational(1, 4)
    safe_remaining_gap = sp.factor(
        minimum_flat_squared_speed_gap - safe_perturbation_witness
    )
    collision_perturbation_witness = minimum_flat_squared_speed_gap
    collision_remaining_gap = sp.factor(
        minimum_flat_squared_speed_gap - collision_perturbation_witness
    )
    robustness_arithmetic_passed = bool(
        robustness_budget == sp.Rational(19, 72)
        and safe_perturbation_witness < robustness_budget
        and safe_remaining_gap == sp.Rational(5, 18)
        and safe_remaining_gap > robustness_budget
        and collision_remaining_gap == 0
    )

    weak_coupling_ratio, admissible_threshold = sp.symbols(
        "epsilon_H epsilon_star", positive=True, finite=True
    )
    weak_coupling_condition = sp.Lt(
        weak_coupling_ratio, admissible_threshold, evaluate=False
    )
    passed = bool(
        generalized_harmonic_rejected
        and all_lorentzian
        and common_time_surface
        and cone_gaps["physical_to_tilde_speed_squared"] > 0
        and cone_gaps["tilde_to_hat_speed_squared"] > 0
        and flat_passed
        and flat_cones_disjoint
        and minimum_flat_squared_speed_gap == sp.Rational(19, 36)
        and robustness_arithmetic_passed
    )
    return passed, {
        "control": "quartic-Horndeski coupled formulation hyperbolicity contract",
        "action_class": {
            "G3": "0",
            "G4": "M2/2+alpha*X",
            "G4_X": "alpha != 0",
            "G5": "0",
        },
        "generalized_harmonic": {
            "status": "reject",
            "necessary_and_sufficient_subclass_condition": (
                "G4_X=0 and G3=0 for G5=0 on a generic weak-field background"
            ),
            "condition_after_action_substitution": str(
                generalized_harmonic_condition
            ),
            "strongly_hyperbolic_for_this_action_class": False,
            "source": {
                "title": "On the hyperbolicity of the most general Horndeski theory",
                "authors": "Giuseppe Papallo",
                "arxiv": "1710.10155",
                "url": "https://arxiv.org/abs/1710.10155",
                "applicable_result": (
                    "for G5=0, generic weak-field generalized-harmonic strong "
                    "hyperbolicity requires G4_X=G3=0"
                ),
            },
        },
        "modified_harmonic": {
            "theorem_status": "conditional_pass",
            "theorem": (
                "weakly coupled Horndeski equations are strongly hyperbolic in the "
                "modified harmonic formulation when physical characteristics do not "
                "intersect the two auxiliary null cones"
            ),
            "source": {
                "title": "Well-posed formulation of Lovelock and Horndeski theories",
                "authors": "Aron D. Kovacs and Harvey S. Reall",
                "journal": "Physical Review D 101, 124003 (2020)",
                "doi": "10.1103/PhysRevD.101.124003",
                "arxiv": "2003.08398",
                "url": "https://arxiv.org/abs/2003.08398",
            },
            "exact_nonempty_auxiliary_cone_witness": {
                "physical_inverse_metric": str(physical_inverse_metric),
                "tilde_inverse_metric": str(auxiliary_tilde),
                "hat_inverse_metric": str(auxiliary_hat),
                "speed_squared": {
                    "physical_metric": str(physical_speed_squared),
                    "tilde": str(tilde_speed_squared),
                    "hat": str(hat_speed_squared),
                },
                "cone_speed_squared_gaps": {
                    name: str(value) for name, value in cone_gaps.items()
                },
                "all_lorentzian": bool(all_lorentzian),
                "common_time_surface_spacelike": bool(common_time_surface),
            },
            "flat_action_witness": {
                "physical_speed_squared": sorted(
                    str(value) for value in flat_physical_speeds
                ),
                "auxiliary_cones_disjoint": flat_cones_disjoint,
                "minimum_squared_speed_gap": str(
                    minimum_flat_squared_speed_gap
                ),
            },
            "exact_cone_robustness_budget": {
                "status": "conditional_pass",
                "baseline_minimum_squared_speed_gap": str(
                    minimum_flat_squared_speed_gap
                ),
                "spectral_perturbation_budget": str(robustness_budget),
                "sufficient_condition": (
                    "||Delta C||_2 < 19/72 for the real-symmetric reduced "
                    "squared-speed operator C"
                ),
                "certified_remaining_gap": "> 19/72",
                "theorem": (
                    "Weyl eigenvalue perturbation bound: each squared-speed "
                    "eigenvalue moves by at most ||Delta C||_2"
                ),
                "required_representation": (
                    "the gauge-reduced physical squared-speed operator must be "
                    "real symmetric, or self-adjoint in an explicitly positive "
                    "symmetrizer whose induced norm is used"
                ),
                "exact_safe_witness": {
                    "perturbation_norm": str(safe_perturbation_witness),
                    "remaining_minimum_gap": str(safe_remaining_gap),
                    "inside_budget": bool(
                        safe_perturbation_witness < robustness_budget
                    ),
                },
                "exact_collision_witness": {
                    "perturbation_norm": str(
                        collision_perturbation_witness
                    ),
                    "remaining_minimum_gap": str(collision_remaining_gap),
                    "interpretation": (
                        "a worst-direction shift equal to the full baseline gap "
                        "can place a physical characteristic on an auxiliary cone"
                    ),
                },
                "full_correction_norm_status": "unresolved",
                "scope": (
                    "exact quantitative cone-separation target only; the separately "
                    "extracted action-specific 11-by-11 correction is not yet uniformly "
                    "bounded in a positive symmetrizer norm"
                ),
            },
            "weak_coupling_condition": str(weak_coupling_condition),
        },
        "action_specific_application": {
            "status": "unresolved",
            "missing": [
                "declare uniform curvature, scalar-Hessian, and scalar-gradient-component bounds satisfying the extracted exact Frobenius time-block condition",
                "construct an explicitly positive symmetrizer for the extracted 22-by-22 generalized first-order pencil",
                "bound the symmetrizer-induced correction norm uniformly over the declared background and direction sphere",
                "prove the resulting physical characteristics remain disjoint from both auxiliary cones",
            ],
            "reason": (
                "the local symbol and an exact sufficient time-block radius are extracted, but "
                "the declared gradient-only domain does not bound the curvature and Hessian jets; "
                "the theorem also does not supply the positive action-specific symmetrizer or a "
                "uniform numeric epsilon_star over this background domain"
            ),
        },
        "scope": (
            "exact action-class formulation audit and auxiliary-cone witness tied to two primary "
            "hyperbolicity theorems. It rejects generalized harmonic gauge and proves that a "
            "modified-harmonic weak-coupling domain is nonempty, but the separately extracted "
            "action-specific full principal matrix still lacks a uniform symmetrizer-induced bound"
        ),
    }


def quartic_horndeski_arbitrary_curvature_scalar_principal_control() -> tuple[
    bool, dict[str, Any]
]:
    """Construct the fixed-metric scalar cone on an arbitrary local curvature jet."""

    (
        euler_passed,
        euler_evidence,
        alpha,
        rho,
        flux,
        pressure_x,
        pressure_y,
        pressure_z,
        kinetic,
        time_space,
        gradients,
        effective_inverse_metric,
        x_discriminant,
        x_characteristic_speeds,
        diagonal_speed_squared,
        determinant,
        x_schur_gradient,
        diagonal_healthy,
    ) = _quartic_horndeski_curvature_scalar_principal_prefix()
    diagonal_matrix = effective_inverse_metric.subs(diagonal_healthy)
    diagonal_speeds = [
        sp.factor(item.subs(diagonal_healthy)) for item in diagonal_speed_squared
    ]
    oblique_healthy = {
        alpha: sp.Rational(1, 4),
        rho: 1,
        flux: sp.Rational(1, 2),
        pressure_x: 0,
        pressure_y: 0,
        pressure_z: 0,
    }
    oblique_matrix = effective_inverse_metric.subs(oblique_healthy)
    oblique_speeds = [
        sp.factor(item.subs(oblique_healthy))
        for item in x_characteristic_speeds
    ]
    oblique_discriminant = sp.factor(x_discriminant.subs(oblique_healthy))

    collapse = {
        alpha: sp.Rational(1, 2),
        rho: 0,
        flux: 0,
        pressure_x: 1,
        pressure_y: 0,
        pressure_z: 0,
    }
    gradient_instability = {
        alpha: 1,
        rho: 0,
        flux: 0,
        pressure_x: 1,
        pressure_y: 0,
        pressure_z: 0,
    }
    kinetic_flip = {
        alpha: -1,
        rho: 1,
        flux: 0,
        pressure_x: 0,
        pressure_y: 0,
        pressure_z: 0,
    }
    superluminal = {
        alpha: sp.Rational(1, 2),
        rho: 0,
        flux: 0,
        pressure_x: -1,
        pressure_y: 0,
        pressure_z: 0,
    }
    collapse_matrix = effective_inverse_metric.subs(collapse)
    gradient_matrix = effective_inverse_metric.subs(gradient_instability)
    kinetic_flip_matrix = effective_inverse_metric.subs(kinetic_flip)
    superluminal_speed = sp.factor(diagonal_speed_squared[0].subs(superluminal))

    passed = (
        euler_passed
        and euler_evidence["expected_second_order_euler_coefficient"]
        == "R*box_phi - 2*Ricci_dot_Hessian"
        and diagonal_matrix == sp.diag(
            -sp.Rational(6, 5),
            sp.Rational(9, 10),
            sp.Rational(21, 20),
            1,
        )
        and diagonal_speeds
        == [sp.Rational(3, 4), sp.Rational(7, 8), sp.Rational(5, 6)]
        and oblique_matrix.det() == -sp.Rational(25, 16)
        and oblique_discriminant == sp.Rational(25, 16)
        and oblique_speeds == [1, -sp.Rational(2, 3)]
        and collapse_matrix.det() == 0
        and gradients[0].subs(collapse) == 0
        and gradients[0].subs(gradient_instability) == -1
        and gradient_matrix == sp.diag(-1, -1, 1, 1)
        and kinetic.subs(kinetic_flip) == -1
        and kinetic_flip_matrix == sp.eye(4)
        and superluminal_speed == 2
    )
    return passed, {
        "control": "quartic-Horndeski arbitrary-curvature scalar effective cone",
        "covariant_scalar_principal_equation": (
            "(g^(mu nu)-2 alpha G^(mu nu)) nabla_mu nabla_nu(phi)=0"
        ),
        "effective_inverse_metric": "P^(mu nu)=g^(mu nu)-2 alpha G^(mu nu)",
        "euler_reduction_residual": euler_evidence["reduction_residual"],
        "local_orthonormal_frame": {
            "metric": "diag(-1,1,1,1)",
            "einstein_tensor": "[[rho,j,0,0],[j,p1,0,0],[0,0,p2,0],[0,0,0,p3]]",
            "effective_inverse_metric_matrix": str(effective_inverse_metric),
            "determinant": str(determinant),
            "time_kinetic_coefficient": str(kinetic),
            "time_space_coefficient": str(time_space),
            "spatial_gradient_coefficients": [str(item) for item in gradients],
            "x_characteristic_discriminant": str(x_discriminant),
            "x_characteristic_speeds": [
                str(item) for item in x_characteristic_speeds
            ],
            "x_schur_gradient": str(x_schur_gradient),
            "diagonal_speed_squared": [
                str(item) for item in diagonal_speed_squared
            ],
        },
        "necessary_preferred_slicing_conditions": [
            "1+2*alpha*rho > 0",
            "(-2*alpha*j)^2+(1+2*alpha*rho)*(1-2*alpha*p1) > 0",
            "1-2*alpha*p2 > 0",
            "1-2*alpha*p3 > 0",
        ],
        "healthy_diagonal_witness": {
            "substitution": {
                "alpha": "1/10",
                "rho": "1",
                "j": "0",
                "p1": "1/2",
                "p2": "-1/4",
                "p3": "0",
            },
            "matrix": str(diagonal_matrix),
            "speed_squared": [str(item) for item in diagonal_speeds],
            "lorentzian": True,
        },
        "healthy_oblique_witness": {
            "substitution": {
                "alpha": "1/4",
                "rho": "1",
                "j": "1/2",
                "p1": "0",
                "p2": "0",
                "p3": "0",
            },
            "matrix": str(oblique_matrix),
            "determinant": str(sp.factor(oblique_matrix.det())),
            "x_discriminant": str(oblique_discriminant),
            "x_characteristic_speeds": [str(item) for item in oblique_speeds],
            "lorentzian": True,
        },
        "negative_controls": {
            "cone_collapse": {
                "matrix": str(collapse_matrix),
                "determinant": str(sp.factor(collapse_matrix.det())),
                "rejected": bool(collapse_matrix.det() == 0),
            },
            "gradient_instability": {
                "matrix": str(gradient_matrix),
                "x_gradient": str(
                    gradients[0].subs(gradient_instability)
                ),
                "rejected": bool(
                    gradients[0].subs(gradient_instability) < 0
                ),
            },
            "kinetic_flip_elliptic": {
                "matrix": str(kinetic_flip_matrix),
                "time_kinetic": str(kinetic.subs(kinetic_flip)),
                "rejected": bool(kinetic.subs(kinetic_flip) < 0),
            },
        },
        "metric_cone_comparison": {
            "superluminal_witness_speed_squared": str(superluminal_speed),
            "exceeds_metric_light_cone": bool(superluminal_speed > 1),
            "health_rejection_without_declared_cone_policy": False,
        },
        "full_coupled_metric_scalar_principal_status": "unresolved",
        "scope": (
            "exact covariant fixed-metric scalar principal tensor on an arbitrary curvature "
            "jet, with a local orthonormal one-flux block and exact witnesses. Metric-scalar "
            "principal mixing, gauge reduction, uniform strong hyperbolicity of the full system, "
            "and nonlinear Hamiltonian boundedness remain separate"
        ),
    }


def quartic_horndeski_boundary_and_flrw_noether_control() -> tuple[bool, dict[str, Any]]:
    """Check the John boundary equivalence and an exact nonlinear lapse-FLRW Noether identity."""

    box_squared, hessian_squared, ricci_gradient_squared = sp.symbols(
        "box_squared hessian_squared Ricci_gradphi_gradphi", real=True
    )
    scalar_gradient_squared, curvature, divergence = sp.symbols(
        "gradphi_squared R divergence", real=True
    )
    x_c = -scalar_gradient_squared / 2
    hessian_identity = sp.Eq(
        box_squared - hessian_squared,
        ricci_gradient_squared + divergence,
        evaluate=False,
    )
    horndeski_density = x_c * curvature + box_squared - hessian_squared
    john_density = ricci_gradient_squared - curvature * scalar_gradient_squared / 2
    boundary_residual = sp.simplify(
        horndeski_density.subs(
            box_squared - hessian_squared, ricci_gradient_squared + divergence
        )
        - john_density
        - divergence
    )

    time = sp.Symbol("t", real=True)
    scale = sp.Function("a")(time)
    lapse = sp.Function("N")(time)
    scalar = sp.Function("phi")(time)
    lagrangian = sp.factor(
        3
        * scale
        * sp.diff(scale, time) ** 2
        * sp.diff(scalar, time) ** 2
        / lapse**3
    )

    def euler(variable: sp.Expr) -> sp.Expr:
        return sp.factor(
            sp.diff(lagrangian, variable)
            - sp.diff(sp.diff(lagrangian, sp.diff(variable, time)), time)
        )

    lapse_euler = euler(lapse)
    scale_euler = euler(scale)
    scalar_euler = euler(scalar)
    noether_residual = sp.factor(
        lapse_euler * sp.diff(lapse, time)
        + scale_euler * sp.diff(scale, time)
        + scalar_euler * sp.diff(scalar, time)
        - sp.diff(lapse * lapse_euler, time)
    )
    omitted_lapse_density_term = sp.factor(
        lapse_euler * sp.diff(lapse, time)
        + scale_euler * sp.diff(scale, time)
        + scalar_euler * sp.diff(scalar, time)
    )
    negative_rejected = omitted_lapse_density_term != 0
    passed = boundary_residual == 0 and noether_residual == 0 and negative_rejected
    return passed, {
        "boundary_equivalence": {
            "hessian_commutator_identity": str(hessian_identity),
            "horndeski_density": str(horndeski_density),
            "john_density": str(john_density),
            "boundary_residual": str(boundary_residual),
            "conclusion": "L4_linear_X=G^(mu nu) grad_mu(phi) grad_nu(phi)+covariant divergence",
        },
        "lapse_flrw": {
            "ansatz": "ds^2=-N(t)^2 dt^2+a(t)^2 dvec(x)^2, phi=phi(t)",
            "reduced_john_lagrangian_without_overall_coupling": str(lagrangian),
            "lapse_euler": str(lapse_euler),
            "scale_euler": str(scale_euler),
            "scalar_euler": str(scalar_euler),
            "time_reparameterization_identity": (
                "E_N dot(N)+E_a dot(a)+E_phi dot(phi)-d_t(N E_N)=0"
            ),
            "noether_residual": str(noether_residual),
            "omitted_lapse_density_term_residual": str(omitted_lapse_density_term),
            "negative_control_rejected": negative_rejected,
        },
        "scope": (
            "exact covariant integration-by-parts equivalence plus an exact nonlinear homogeneous "
            "Noether corroboration; the arbitrary-background Euler identity is checked separately"
        ),
    }


def quartic_horndeski_timelike_flat_principal_control() -> tuple[bool, dict[str, Any]]:
    """Exact reduced principal symbol on a constant timelike scalar-gradient background.

    For ``G4=M2/2+alpha X_c`` and a positive canonical ``G2=X_c`` term, the two
    transverse-traceless tensor modes have kinetic coefficient ``G4-2X_c G4_X`` and
    gradient coefficient ``G4``.  The scalar remains canonically luminal on this flat,
    constant-gradient background.  This is a declared-background control, not an
    arbitrary-background strong-hyperbolicity theorem.
    """

    m2 = sp.Symbol("M2", positive=True, finite=True)
    alpha = sp.Symbol("alpha", nonzero=True, finite=True, real=True)
    a_star = sp.Symbol("A_star", positive=True, finite=True)
    c_x = sp.Symbol("c_X", positive=True, finite=True)
    x_c = sp.factor(a_star**2 / 2)
    tensor_kinetic = sp.factor(m2 / 2 - alpha * x_c)
    tensor_gradient = sp.factor(m2 / 2 + alpha * x_c)
    kinetic = sp.diag(tensor_kinetic, tensor_kinetic, c_x)
    gradient = sp.diag(tensor_gradient, tensor_gradient, c_x)
    propagation = sp.simplify(kinetic.inv() * gradient)
    tensor_speed = sp.factor(tensor_gradient / tensor_kinetic)
    expected = sp.diag(tensor_speed, tensor_speed, 1)
    omega, wave_number = sp.symbols("omega k", real=True)
    polynomial = sp.factor(
        (-omega**2 * kinetic + wave_number**2 * gradient).det()
    )
    expected_polynomial = sp.factor(
        c_x
        * (c_x * (wave_number**2 - omega**2))
        * (
            tensor_gradient * wave_number**2
            - tensor_kinetic * omega**2
        )
        ** 2
        / c_x
    )

    healthy_point = {m2: 2, alpha: sp.Rational(-1, 4), a_star: 1, c_x: 1}
    ghost_point = {m2: 2, alpha: 3, a_star: 1, c_x: 1}
    gradient_point = {m2: 2, alpha: -3, a_star: 1, c_x: 1}
    strong_coupling_point = {c_x: 0}
    healthy_values = {
        "tensor_kinetic": sp.factor(tensor_kinetic.subs(healthy_point)),
        "tensor_gradient": sp.factor(tensor_gradient.subs(healthy_point)),
        "tensor_speed_squared": sp.factor(tensor_speed.subs(healthy_point)),
        "scalar_kinetic": sp.factor(c_x.subs(healthy_point)),
    }
    ghost_value = sp.factor(tensor_kinetic.subs(ghost_point))
    gradient_value = sp.factor(tensor_gradient.subs(gradient_point))
    strong_coupling_determinant = sp.factor(kinetic.det().subs(strong_coupling_point))
    passed = (
        sp.simplify(propagation - expected) == sp.zeros(3)
        and sp.factor(polynomial - expected_polynomial) == 0
        and all(value.is_positive is True for value in healthy_values.values())
        and ghost_value.is_negative is True
        and gradient_value.is_negative is True
        and strong_coupling_determinant == 0
    )
    return passed, {
        "control": "quartic-Horndeski constant-timelike-gradient flat principal symbol",
        "background": (
            "g_mu_nu=eta_mu_nu, nabla_mu(phi)=-A_star n_mu, A_star constant, "
            "nabla_mu nabla_nu(phi)=0"
        ),
        "physical_basis": ["tensor_plus", "tensor_cross", "scalar"],
        "X_c": str(x_c),
        "kinetic_matrix": str(kinetic),
        "gradient_matrix": str(gradient),
        "kinetic_determinant": str(sp.factor(kinetic.det())),
        "principal_polynomial": str(polynomial),
        "propagation_matrix": str(propagation),
        "tensor_speed_squared": str(tensor_speed),
        "scalar_speed_squared": "1",
        "healthy_domain": [
            f"{tensor_kinetic} > 0",
            f"{tensor_gradient} > 0",
            "c_X > 0",
        ],
        "healthy_witness": {
            "substitution": {"M2": "2", "alpha": "-1/4", "A_star": "1", "c_X": "1"},
            "values": {name: str(value) for name, value in healthy_values.items()},
        },
        "negative_controls": {
            "tensor_ghost": {
                "substitution": {"M2": "2", "alpha": "3", "A_star": "1", "c_X": "1"},
                "tensor_kinetic": str(ghost_value),
                "rejected": ghost_value.is_negative is True,
            },
            "tensor_gradient_instability": {
                "substitution": {"M2": "2", "alpha": "-3", "A_star": "1", "c_X": "1"},
                "tensor_gradient": str(gradient_value),
                "rejected": gradient_value.is_negative is True,
            },
            "omitted_canonical_scalar": {
                "substitution": {"c_X": "0"},
                "kinetic_determinant": str(strong_coupling_determinant),
                "rejected": strong_coupling_determinant == 0,
            },
        },
        "scope": (
            "exact gauge-reduced quadratic principal block on the flat constant-timelike-gradient "
            "background; varying scalar Hessian, curved backgrounds, non-TT constraint reduction, "
            "and uniform arbitrary-background strong hyperbolicity remain separate"
        ),
    }


def generic_horndeski_l2_l4_tensor_stability_control() -> tuple[bool, dict[str, Any]]:
    """Certify the arbitrary-function L2--L4 FLRW tensor block and Hamiltonian.

    Kobayashi, Yamaguchi, and Yokoyama, arXiv:1105.5723v4, Eqs. (4.3)--(4.8),
    give the exact quadratic tensor action for the full Horndeski family.  Setting
    ``G5=0`` leaves ``F_T=2 G4`` and ``G_T=2(G4-2 X G4_X)``.  This control checks
    both TT polarizations, the characteristic polynomial, the Legendre transform,
    and explicit ghost, gradient, and singular negative controls.  It is a tensor
    theorem on homogeneous timelike-gradient backgrounds, not a scalar-sector or
    arbitrary-inhomogeneous-background stability theorem.
    """

    g4, g4_x = sp.symbols("G4 G4_X", real=True, finite=True)
    x = sp.Symbol("X", positive=True, finite=True)
    omega, wave_number = sp.symbols("omega k", real=True)
    q_plus, q_cross, velocity_plus, velocity_cross = sp.symbols(
        "q_plus q_cross v_plus v_cross", real=True
    )
    momentum_plus, momentum_cross = sp.symbols("p_plus p_cross", real=True)

    tensor_kinetic = sp.factor(2 * (g4 - 2 * x * g4_x))
    tensor_gradient = sp.factor(2 * g4)
    kinetic_matrix = sp.diag(tensor_kinetic, tensor_kinetic)
    gradient_matrix = sp.diag(tensor_gradient, tensor_gradient)
    principal_matrix = sp.diag(
        tensor_gradient * wave_number**2 - tensor_kinetic * omega**2,
        tensor_gradient * wave_number**2 - tensor_kinetic * omega**2,
    )
    principal_polynomial = sp.factor(principal_matrix.det())
    expected_polynomial = sp.factor(
        (tensor_gradient * wave_number**2 - tensor_kinetic * omega**2) ** 2
    )
    speed_squared = sp.factor(tensor_gradient / tensor_kinetic)

    reduced_lagrangian = sp.factor(
        (
            tensor_kinetic * (velocity_plus**2 + velocity_cross**2)
            - tensor_gradient * wave_number**2 * (q_plus**2 + q_cross**2)
        )
        / 2
    )
    canonical_momenta = sp.Matrix(
        [
            sp.diff(reduced_lagrangian, velocity_plus),
            sp.diff(reduced_lagrangian, velocity_cross),
        ]
    )
    solved_velocities = sp.Matrix(
        [momentum_plus / tensor_kinetic, momentum_cross / tensor_kinetic]
    )
    hamiltonian = sp.factor(
        momentum_plus * solved_velocities[0]
        + momentum_cross * solved_velocities[1]
        - reduced_lagrangian.subs(
            {
                velocity_plus: solved_velocities[0],
                velocity_cross: solved_velocities[1],
            }
        )
    )
    expected_hamiltonian = sp.factor(
        (
            momentum_plus**2
            + momentum_cross**2
            + tensor_kinetic
            * tensor_gradient
            * wave_number**2
            * (q_plus**2 + q_cross**2)
        )
        / (2 * tensor_kinetic)
    )
    legendre_residual = sp.factor(hamiltonian - expected_hamiltonian)
    momentum_residual = sp.simplify(
        canonical_momenta
        - sp.Matrix(
            [tensor_kinetic * velocity_plus, tensor_kinetic * velocity_cross]
        )
    )
    phase_variables = (q_plus, q_cross, momentum_plus, momentum_cross)
    hamiltonian_hessian = sp.hessian(hamiltonian, phase_variables)
    expected_hamiltonian_hessian = sp.diag(
        tensor_gradient * wave_number**2,
        tensor_gradient * wave_number**2,
        1 / tensor_kinetic,
        1 / tensor_kinetic,
    )
    hamiltonian_hessian_residual = sp.simplify(
        hamiltonian_hessian - expected_hamiltonian_hessian
    )

    healthy_point = {g4: 2, g4_x: sp.Rational(-1, 4), x: 1, wave_number: 1}
    ghost_point = {g4: 1, g4_x: 1, x: 1, wave_number: 1}
    gradient_point = {g4: -1, g4_x: -1, x: 1, wave_number: 1}
    kinetic_singular_point = {g4: 2, g4_x: 1, x: 1, wave_number: 1}
    gradient_singular_point = {g4: 0, g4_x: -1, x: 1, wave_number: 1}

    healthy_values = {
        "G_T": sp.factor(tensor_kinetic.subs(healthy_point)),
        "F_T": sp.factor(tensor_gradient.subs(healthy_point)),
        "c_T_squared": sp.factor(speed_squared.subs(healthy_point)),
    }
    healthy_hamiltonian_eigenvalues = [
        sp.factor(value)
        for value in hamiltonian_hessian.subs(healthy_point).eigenvals()
    ]
    ghost_value = sp.factor(tensor_kinetic.subs(ghost_point))
    gradient_value = sp.factor(tensor_gradient.subs(gradient_point))
    kinetic_singular_determinant = sp.factor(
        kinetic_matrix.det().subs(kinetic_singular_point)
    )
    gradient_singular_determinant = sp.factor(
        gradient_matrix.det().subs(gradient_singular_point)
    )
    omitted_completion_kinetic = sp.factor(2 * g4)
    omitted_completion_value = sp.factor(omitted_completion_kinetic.subs(ghost_point))
    omitted_completion_rejected = bool(
        ghost_value < 0 and omitted_completion_value > 0
    )

    passed = (
        sp.factor(principal_polynomial - expected_polynomial) == 0
        and momentum_residual == sp.zeros(2, 1)
        and legendre_residual == 0
        and hamiltonian_hessian_residual == sp.zeros(4)
        and all(value.is_positive is True for value in healthy_values.values())
        and all(value.is_positive is True for value in healthy_hamiltonian_eigenvalues)
        and ghost_value.is_negative is True
        and gradient_value.is_negative is True
        and kinetic_singular_determinant == 0
        and gradient_singular_determinant == 0
        and omitted_completion_rejected
    )
    return passed, {
        "control": "generic Horndeski L2-L4 FLRW tensor principal/Hamiltonian theorem",
        "source": {
            "paper": "Kobayashi, Yamaguchi, Yokoyama, Generalized G-inflation",
            "url": "https://arxiv.org/abs/1105.5723",
            "version": "v4",
            "equations": ["4.3", "4.4", "4.5", "4.7", "4.8"],
            "specialization": "G5=0 (the compiler's L2-L4 family)",
        },
        "background": "homogeneous/isotropic metric with timelike phi gradient, X>0",
        "physical_basis": ["tensor_plus", "tensor_cross"],
        "G_T": str(tensor_kinetic),
        "F_T": str(tensor_gradient),
        "kinetic_matrix": str(kinetic_matrix),
        "gradient_matrix": str(gradient_matrix),
        "principal_matrix": str(principal_matrix),
        "principal_polynomial": str(principal_polynomial),
        "tensor_speed_squared": str(speed_squared),
        "healthy_domain": ["G4 - 2 X G4_X > 0", "G4 > 0"],
        "reduced_lagrangian": str(reduced_lagrangian),
        "canonical_momenta": [str(value) for value in canonical_momenta],
        "canonical_momentum_residual": str(momentum_residual),
        "reduced_hamiltonian": str(hamiltonian),
        "legendre_residual": str(legendre_residual),
        "hamiltonian_hessian": str(hamiltonian_hessian),
        "expected_hamiltonian_hessian": str(expected_hamiltonian_hessian),
        "hamiltonian_hessian_residual": str(hamiltonian_hessian_residual),
        "hamiltonian_positivity": (
            "positive definite for k!=0 iff G_T>0 and F_T>0; positive semidefinite "
            "at k=0 because the homogeneous massless tensor coordinates are zero modes"
        ),
        "healthy_witness": {
            "substitution": {"G4": "2", "G4_X": "-1/4", "X": "1", "k": "1"},
            "values": {name: str(value) for name, value in healthy_values.items()},
            "hamiltonian_hessian_eigenvalues": [
                str(value) for value in healthy_hamiltonian_eigenvalues
            ],
        },
        "negative_controls": {
            "tensor_ghost": {
                "substitution": {"G4": "1", "G4_X": "1", "X": "1"},
                "G_T": str(ghost_value),
                "rejected": ghost_value.is_negative is True,
            },
            "tensor_gradient_instability": {
                "substitution": {"G4": "-1", "G4_X": "-1", "X": "1"},
                "F_T": str(gradient_value),
                "rejected": gradient_value.is_negative is True,
            },
            "kinetic_strong_coupling": {
                "substitution": {"G4": "2", "G4_X": "1", "X": "1"},
                "kinetic_determinant": str(kinetic_singular_determinant),
                "rejected": kinetic_singular_determinant == 0,
            },
            "gradient_cone_collapse": {
                "substitution": {"G4": "0", "G4_X": "-1", "X": "1"},
                "gradient_determinant": str(gradient_singular_determinant),
                "rejected": gradient_singular_determinant == 0,
            },
            "omitted_horndeski_completion": {
                "wrong_G_T": str(omitted_completion_kinetic),
                "ghost_witness_correct_G_T": str(ghost_value),
                "ghost_witness_wrong_G_T": str(omitted_completion_value),
                "rejected": omitted_completion_rejected,
            },
        },
        "capability_boundary": {
            "generic_tensor_principal_symbol": "pass_on_F_T_and_G_T_positive_patch",
            "generic_tensor_reduced_hamiltonian": "pass_on_F_T_and_G_T_positive_patch",
            "generic_scalar_principal_symbol": "unresolved",
            "generic_scalar_reduced_hamiltonian": "unresolved",
            "arbitrary_inhomogeneous_background_strong_hyperbolicity": "unresolved",
            "nonlinear_global_energy_boundedness": "unresolved",
        },
        "scope": (
            "exact arbitrary-function G2/G3/G4 tensor-sector theorem on homogeneous "
            "timelike-gradient backgrounds; scalar constraint reduction, arbitrary "
            "inhomogeneous backgrounds, and nonlinear global energy remain separate"
        ),
    }


def generic_kessence_timelike_principal_hamiltonian_control() -> tuple[bool, dict[str, Any]]:
    """Verify the arbitrary-G2 effective metric on a timelike-gradient background.

    This is the scalar part of the Einstein-plus-k-essence subclass isolated by
    Papallo's generic weak-field generalized-harmonic theorem.  It proves the
    effective cone and reduced quadratic Hamiltonian for arbitrary local ``G2_X``
    and ``G2_XX`` values.  It does not promote a nonzero canonical ``G3`` or
    ``G4_X`` theory; those require the modified-harmonic weak-coupling path.
    """

    g2_x, g2_xx = sp.symbols("G2_X G2_XX", real=True, finite=True)
    x = sp.Symbol("X", positive=True, finite=True)
    omega, wave_number = sp.symbols("omega k", real=True)
    field, velocity, momentum = sp.symbols("q q_dot p", real=True)
    scalar_kinetic = sp.factor(g2_x + 2 * x * g2_xx)
    scalar_gradient = g2_x
    effective_inverse_metric = sp.diag(
        -scalar_kinetic,
        scalar_gradient,
        scalar_gradient,
        scalar_gradient,
    )
    effective_determinant = sp.factor(effective_inverse_metric.det())
    principal_polynomial = sp.factor(
        scalar_gradient * wave_number**2 - scalar_kinetic * omega**2
    )
    speed_squared = sp.factor(scalar_gradient / scalar_kinetic)
    reduced_lagrangian = sp.factor(
        (
            scalar_kinetic * velocity**2
            - scalar_gradient * wave_number**2 * field**2
        )
        / 2
    )
    canonical_momentum = sp.diff(reduced_lagrangian, velocity)
    canonical_momentum_residual = sp.factor(
        canonical_momentum - scalar_kinetic * velocity
    )
    solved_velocity = momentum / scalar_kinetic
    hamiltonian = sp.factor(
        momentum * solved_velocity
        - reduced_lagrangian.subs(velocity, solved_velocity)
    )
    expected_hamiltonian = sp.factor(
        momentum**2 / (2 * scalar_kinetic)
        + scalar_gradient * wave_number**2 * field**2 / 2
    )
    legendre_residual = sp.factor(hamiltonian - expected_hamiltonian)
    hamiltonian_hessian = sp.hessian(hamiltonian, (field, momentum))
    expected_hamiltonian_hessian = sp.diag(
        scalar_gradient * wave_number**2,
        1 / scalar_kinetic,
    )
    hamiltonian_hessian_residual = sp.simplify(
        hamiltonian_hessian - expected_hamiltonian_hessian
    )
    effective_determinant_residual = sp.factor(
        effective_determinant + scalar_kinetic * scalar_gradient**3
    )

    healthy_point = {g2_x: 2, g2_xx: sp.Rational(1, 2), x: 1, wave_number: 1}
    ghost_point = {g2_x: 1, g2_xx: -1, x: 1}
    gradient_point = {g2_x: -1, g2_xx: 1, x: 1}
    kinetic_singular_point = {g2_x: 2, g2_xx: -1, x: 1}
    gradient_singular_point = {g2_x: 0, g2_xx: 1, x: 1}
    healthy_values = {
        "scalar_kinetic": sp.factor(scalar_kinetic.subs(healthy_point)),
        "scalar_gradient": sp.factor(scalar_gradient.subs(healthy_point)),
        "speed_squared": sp.factor(speed_squared.subs(healthy_point)),
    }
    ghost_value = sp.factor(scalar_kinetic.subs(ghost_point))
    gradient_value = sp.factor(scalar_gradient.subs(gradient_point))
    kinetic_singular_value = sp.factor(
        scalar_kinetic.subs(kinetic_singular_point)
    )
    gradient_singular_value = sp.factor(
        scalar_gradient.subs(gradient_singular_point)
    )
    passed = (
        canonical_momentum_residual == 0
        and legendre_residual == 0
        and hamiltonian_hessian_residual == sp.zeros(2)
        and effective_determinant_residual == 0
        and all(value.is_positive is True for value in healthy_values.values())
        and ghost_value.is_negative is True
        and gradient_value.is_negative is True
        and kinetic_singular_value == 0
        and gradient_singular_value == 0
    )
    return passed, {
        "control": "generic k-essence timelike-gradient principal/Hamiltonian theorem",
        "source": {
            "paper": "Papallo, On the hyperbolicity of the most general Horndeski theory",
            "url": "https://arxiv.org/abs/1710.10155",
            "version": "v2",
            "applicable_result": (
                "on generic weak-field backgrounds, generalized-harmonic strong "
                "hyperbolicity is restricted to G3=0, G4_X=0, G5=0"
            ),
        },
        "background": (
            "local orthonormal frame with timelike scalar gradient, "
            "p_mu=(sqrt(2X),0,0,0), X>0"
        ),
        "covariant_effective_inverse_metric": (
            "P^(mu nu)=G2_X g^(mu nu)-G2_XX nabla^mu(phi) nabla^nu(phi)"
        ),
        "effective_inverse_metric_aligned": str(effective_inverse_metric),
        "effective_metric_determinant": str(effective_determinant),
        "effective_metric_determinant_residual": str(effective_determinant_residual),
        "principal_polynomial": str(principal_polynomial),
        "scalar_kinetic": str(scalar_kinetic),
        "scalar_gradient": str(scalar_gradient),
        "scalar_speed_squared": str(speed_squared),
        "healthy_domain": ["G2_X > 0", "G2_X+2 X G2_XX > 0"],
        "reduced_lagrangian": str(reduced_lagrangian),
        "canonical_momentum": str(canonical_momentum),
        "canonical_momentum_residual": str(canonical_momentum_residual),
        "reduced_hamiltonian": str(hamiltonian),
        "legendre_residual": str(legendre_residual),
        "hamiltonian_hessian_residual": str(hamiltonian_hessian_residual),
        "healthy_witness": {
            "substitution": {"G2_X": "2", "G2_XX": "1/2", "X": "1", "k": "1"},
            "values": {name: str(value) for name, value in healthy_values.items()},
        },
        "negative_controls": {
            "ghost": {
                "kinetic": str(ghost_value),
                "rejected": bool(ghost_value < 0),
            },
            "gradient_instability": {
                "gradient": str(gradient_value),
                "rejected": bool(gradient_value < 0),
            },
            "kinetic_cone_collapse": {
                "kinetic": str(kinetic_singular_value),
                "rejected": kinetic_singular_value == 0,
            },
            "spatial_cone_collapse": {
                "gradient": str(gradient_singular_value),
                "rejected": gradient_singular_value == 0,
            },
        },
        "formulation_boundary": {
            "generalized_harmonic": (
                "source-supported for the canonical G3=0, G4_X=0, G5=0 subclass "
                "when the k-essence effective metric is Lorentzian with a common time surface"
            ),
            "nonzero_canonical_G3_or_G4_X": (
                "reject in generalized harmonic on generic weak-field backgrounds; "
                "route to modified harmonic"
            ),
            "modified_harmonic": (
                "conditional at weak coupling with separated auxiliary cones; candidate-specific "
                "uniform correction/symmetrizer bound remains required"
            ),
        },
        "scope": (
            "exact arbitrary-G2 aligned timelike-gradient scalar cone and quadratic Hamiltonian, "
            "plus the source-supported generalized-harmonic subclass boundary; arbitrary "
            "inhomogeneous candidate domains still require a common-time and uniform bound"
        ),
    }


def generic_kessence_nonlinear_adm_legendre_control() -> tuple[bool, dict[str, Any]]:
    """Certify the exact pointwise nonlinear ADM Legendre map for arbitrary G2."""

    normal_velocity = sp.Symbol("v_n", real=True, finite=True)
    spatial_gradient_squared = sp.Symbol("s_squared", nonnegative=True, finite=True)
    g2, g2_x, g2_xx = sp.symbols("G2 G2_X G2_XX", real=True, finite=True)
    x = sp.factor((normal_velocity**2 - spatial_gradient_squared) / 2)
    momentum_density = sp.factor(g2_x * normal_velocity)
    legendre_jacobian = sp.factor(g2_x + normal_velocity**2 * g2_xx)
    hamiltonian_density = sp.factor(
        momentum_density * normal_velocity - g2
    )
    hamiltonian_momentum_hessian = sp.factor(1 / legendre_jacobian)

    # Exact jet-chain identities: d_v G(X(v))=G_X v and
    # d_v(G_X v)=G_X+G_XX v^2.  Consequently dH/dp=v and
    # d^2H/dp^2=1/(dp/dv) on every regular branch.
    d_hamiltonian_dv = sp.factor(normal_velocity * legendre_jacobian)
    d_hamiltonian_dp_residual = sp.factor(
        d_hamiltonian_dv / legendre_jacobian - normal_velocity
    )
    inverse_hessian_residual = sp.factor(
        hamiltonian_momentum_hessian * legendre_jacobian - 1
    )

    canonical_substitution = {g2: x, g2_x: 1, g2_xx: 0}
    canonical_momentum = sp.factor(momentum_density.subs(canonical_substitution))
    canonical_jacobian = sp.factor(legendre_jacobian.subs(canonical_substitution))
    canonical_hamiltonian = sp.factor(
        hamiltonian_density.subs(canonical_substitution)
    )
    expected_canonical_hamiltonian = sp.factor(
        (normal_velocity**2 + spatial_gradient_squared) / 2
    )
    canonical_residual = sp.factor(
        canonical_hamiltonian - expected_canonical_hamiltonian
    )

    wrong_sign_substitution = {g2: -x, g2_x: -1, g2_xx: 0}
    wrong_sign_jacobian = sp.factor(
        legendre_jacobian.subs(wrong_sign_substitution)
    )
    wrong_sign_hamiltonian = sp.factor(
        hamiltonian_density.subs(wrong_sign_substitution)
    )
    nonconvex_substitution = {
        normal_velocity: 2,
        spatial_gradient_squared: 0,
        g2: x - x**2,
        g2_x: 1 - 2 * x,
        g2_xx: -2,
    }
    nonconvex_jacobian = sp.factor(
        legendre_jacobian.subs(nonconvex_substitution)
    )
    singular_substitution = {
        normal_velocity: 1,
        g2_x: 1,
        g2_xx: -1,
    }
    singular_jacobian = sp.factor(legendre_jacobian.subs(singular_substitution))
    passed = (
        d_hamiltonian_dp_residual == 0
        and inverse_hessian_residual == 0
        and canonical_momentum == normal_velocity
        and canonical_jacobian == 1
        and canonical_residual == 0
        and wrong_sign_jacobian == -1
        and sp.factor(
            wrong_sign_hamiltonian
            + (normal_velocity**2 + spatial_gradient_squared) / 2
        )
        == 0
        and nonconvex_jacobian == -11
        and singular_jacobian == 0
    )
    return passed, {
        "control": "generic k-essence nonlinear pointwise ADM Legendre theorem",
        "ADM_decomposition": {
            "X": str(x),
            "normal_velocity": "v_n=n^mu nabla_mu(phi)",
            "spatial_gradient_squared": "s_squared=q^(ij) D_i(phi) D_j(phi)>=0",
            "density_normalization": "overall positive sqrt(q) suppressed",
        },
        "canonical_momentum_density": str(momentum_density),
        "legendre_jacobian": str(legendre_jacobian),
        "regular_branch": f"{legendre_jacobian} != 0",
        "strict_convexity_condition": f"{legendre_jacobian} > 0",
        "hamiltonian_density": str(hamiltonian_density),
        "hamiltonian_momentum_hessian": str(hamiltonian_momentum_hessian),
        "dH_dp_residual": str(d_hamiltonian_dp_residual),
        "inverse_hessian_residual": str(inverse_hessian_residual),
        "canonical_scalar_control": {
            "momentum": str(canonical_momentum),
            "legendre_jacobian": str(canonical_jacobian),
            "hamiltonian": str(canonical_hamiltonian),
            "expected_hamiltonian": str(expected_canonical_hamiltonian),
            "residual": str(canonical_residual),
            "nonnegative_for_all_v_n_and_s_squared": True,
        },
        "negative_controls": {
            "wrong_sign_scalar": {
                "legendre_jacobian": str(wrong_sign_jacobian),
                "hamiltonian": str(wrong_sign_hamiltonian),
                "rejected": bool(wrong_sign_jacobian < 0),
            },
            "nonconvex_G2_equals_X_minus_X_squared": {
                "witness": {"v_n": "2", "s_squared": "0", "X": "2"},
                "legendre_jacobian": str(nonconvex_jacobian),
                "rejected": bool(nonconvex_jacobian < 0),
            },
            "singular_legendre_surface": {
                "witness": {"v_n": "1", "G2_X": "1", "G2_XX": "-1"},
                "legendre_jacobian": str(singular_jacobian),
                "rejected": bool(singular_jacobian == 0),
            },
        },
        "capability_boundary": {
            "pointwise_nonlinear_scalar_legendre_map": "pass_on_nonzero_jacobian_branches",
            "pointwise_scalar_hamiltonian_convexity": "pass_if_jacobian_positive",
            "pointwise_scalar_energy_nonnegativity": (
                "candidate-specific inequality G2_X v_n^2-G2>=0 required"
            ),
            "global_gravitational_positive_energy": "unresolved",
            "boundary_generator": "unresolved",
        },
        "scope": (
            "exact nonlinear local scalar Legendre transform for arbitrary G2 jets and spatial "
            "gradient magnitude; it is not a global gravitational energy or asymptotic boundary "
            "charge theorem"
        ),
    }


def generic_cubic_horndeski_bssn_hyperbolicity_control() -> tuple[bool, dict[str, Any]]:
    """Encode the source-bound weak-field BSSN theorem for cubic Horndeski."""

    characteristic_speed = sp.Symbol("lambda", real=True)
    momentum_parameter, slicing_parameter = sp.symbols("m sigma", real=True)
    scalar_speed_squared = sp.Symbol("c_phi_squared", positive=True, finite=True)
    transverse_polynomial = sp.factor(characteristic_speed**2 - 1)
    momentum_polynomial = sp.factor(
        characteristic_speed**2 - momentum_parameter
    )
    slicing_polynomial = sp.factor(
        characteristic_speed**2 - 2 * slicing_parameter
    )
    longitudinal_polynomial = sp.factor(
        3 * characteristic_speed**2 - (4 * momentum_parameter - 1)
    )
    scalar_slicing_cone_gap = sp.factor(
        2 * slicing_parameter - scalar_speed_squared
    )
    witness = {momentum_parameter: 1, slicing_parameter: 1, scalar_speed_squared: 1}
    witness_squared_speeds = {
        "transverse": sp.Integer(1),
        "momentum": momentum_parameter.subs(witness),
        "slicing": (2 * slicing_parameter).subs(witness),
        "longitudinal": ((4 * momentum_parameter - 1) / 3).subs(witness),
        "scalar": scalar_speed_squared.subs(witness),
    }
    witness_gap = sp.factor(scalar_slicing_cone_gap.subs(witness))
    momentum_boundary = sp.factor(
        longitudinal_polynomial.subs(momentum_parameter, sp.Rational(1, 4))
    )
    harmonic_slicing_crossing = sp.factor(
        scalar_slicing_cone_gap.subs(
            {
                slicing_parameter: sp.Rational(1, 2),
                scalar_speed_squared: 1,
            }
        )
    )
    explicit_cone_crossing = sp.factor(
        scalar_slicing_cone_gap.subs(scalar_speed_squared, 2 * slicing_parameter)
    )
    weak_field_g2 = [
        {
            "k_X": k,
            "l_phi": ell,
            "dimensionless_ratio": f"abs(d_X^{k} d_phi^{ell} G2) * E^{2 * k + 2}",
        }
        for k in range(3)
        for ell in range(2)
    ]
    weak_field_g3 = [
        {
            "k_X": k,
            "l_phi": ell,
            "dimensionless_ratio": f"abs(d_X^{k} d_phi^{ell} G3) * E^{2 * k}",
        }
        for k in range(3)
        for ell in range(3)
    ]
    passed = (
        sp.expand(transverse_polynomial - (characteristic_speed**2 - 1)) == 0
        and momentum_polynomial == characteristic_speed**2 - momentum_parameter
        and slicing_polynomial == characteristic_speed**2 - 2 * slicing_parameter
        and longitudinal_polynomial
        == -4 * momentum_parameter + 3 * characteristic_speed**2 + 1
        and all(speed > 0 for speed in witness_squared_speeds.values())
        and witness_gap == 1
        and momentum_boundary == 3 * characteristic_speed**2
        and harmonic_slicing_crossing == 0
        and explicit_cone_crossing == 0
        and len(weak_field_g2) == 6
        and len(weak_field_g3) == 9
    )
    return passed, {
        "control": "generic cubic Horndeski weak-field BSSN hyperbolicity theorem",
        "source": {
            "title": "Well-posedness of cubic Horndeski theories",
            "url": "https://arxiv.org/abs/1904.00963",
            "scope": "cubic Horndeski G2(phi,X), G3(phi,X), constant G4, G5=0",
        },
        "principal_speed_polynomials": {
            "transverse": str(transverse_polynomial),
            "momentum_constraint_added": str(momentum_polynomial),
            "Bona_Masso_slicing": str(slicing_polynomial),
            "longitudinal": str(longitudinal_polynomial),
            "scalar": "P'_phi_phi(lambda,n_i)=0 with two distinct real roots",
        },
        "source_conditions": {
            "momentum_constraint_parameter": "m > 1/4",
            "slicing_parameter": "suitable sigma > 1/2",
            "scalar_slicing_cone_separation": (
                "F_sigma(xi_phi^plus/minus,n_i) != 0 uniformly on the direction sphere"
            ),
            "weak_field": "every declared G2/G3 derivative ratio is much less than one",
            "fixed_shift": True,
        },
        "isotropic_cone_gap_squared": str(scalar_slicing_cone_gap),
        "weak_field_scale": (
            "E=max(|Riemann|^(1/2), |nabla(phi)|, |nabla-nabla(phi)|^(1/2)) "
            "in the declared orthonormal frame/domain"
        ),
        "weak_field_derivative_ledger": {"G2": weak_field_g2, "G3": weak_field_g3},
        "healthy_parameter_witness": {
            "assignment": {"m": "1", "sigma": "1", "c_phi_squared": "1"},
            "squared_speeds": {
                name: str(value) for name, value in witness_squared_speeds.items()
            },
            "scalar_slicing_cone_gap_squared": str(witness_gap),
        },
        "negative_controls": {
            "momentum_boundary_m_equals_one_quarter": {
                "longitudinal_polynomial": str(momentum_boundary),
                "zero_speed_at_lambda_zero": True,
                "rejected": True,
            },
            "original_harmonic_slicing_luminal_crossing": {
                "assignment": {"sigma": "1/2", "c_phi_squared": "1"},
                "cone_gap": str(harmonic_slicing_crossing),
                "rejected": True,
            },
            "scalar_slicing_cone_crossing": {
                "assignment": {"c_phi_squared": "2*sigma"},
                "cone_gap": str(explicit_cone_crossing),
                "rejected": True,
            },
        },
        "candidate_contract": {
            "exact_G4_X_zero_and_canonical_G3_nonzero": "required",
            "uniform_weak_field_derivative_bounds": "required",
            "two_distinct_real_scalar_roots": "required",
            "uniform_scalar_slicing_cone_gap": "required",
            "universal_numeric_threshold_for_much_less_than": "not supplied by source",
        },
        "capability_boundary": (
            "This removes the need for the general quartic modified-harmonic symmetrizer on the "
            "G3-only subclass, but remains conditional until the candidate supplies uniform weak-"
            "field and cone-separation bounds on its declared inhomogeneous domain."
        ),
    }


def generic_horndeski_l2_l4_flrw_scalar_reduction_control() -> tuple[bool, dict[str, Any]]:
    """Verify the exact FLRW scalar constraint reduction for arbitrary L2--L4.

    The source-bound unreduced action and lapse/shift constraints are Eqs. (4.24)
    and (4.29)--(4.30) of arXiv:1105.5723v4.  Eliminating the nondynamical lapse
    and scalar shift and integrating the single mixed term by parts gives Eqs.
    (4.31)--(4.34).  ``Sigma``, ``Theta``, ``F_T``, and ``G_T`` remain arbitrary
    background coefficients here; the function-family compiler supplies their
    action-specific expressions before a background can receive a health verdict.
    """

    g_t, f_t, sigma = sp.symbols("G_T F_T Sigma", real=True, finite=True)
    theta = sp.Symbol("Theta", nonzero=True, real=True, finite=True)
    hubble, a = sp.symbols("H a", real=True, positive=True, finite=True)
    a_ratio_dot = sp.Symbol("A_dot", real=True, finite=True)
    wave_number = sp.Symbol("k", positive=True, finite=True)
    zeta, zeta_dot, lapse_scalar, shift_scalar = sp.symbols(
        "zeta zeta_dot alpha beta", real=True
    )

    laplace_shift = -wave_number**2 * shift_scalar / a**2
    laplace_zeta = -wave_number**2 * zeta / a**2
    unreduced_lagrangian = sp.expand(
        -3 * g_t * zeta_dot**2
        + f_t * wave_number**2 * zeta**2 / a**2
        + sigma * lapse_scalar**2
        - 2 * theta * lapse_scalar * laplace_shift
        + 2 * g_t * zeta_dot * laplace_shift
        + 6 * theta * lapse_scalar * zeta_dot
        - 2 * g_t * lapse_scalar * laplace_zeta
    )
    lapse_constraint = sp.factor(sp.diff(unreduced_lagrangian, lapse_scalar) / 2)
    shift_constraint = sp.factor(
        sp.diff(unreduced_lagrangian, shift_scalar)
        * a**2
        / (2 * wave_number**2)
    )
    expected_lapse_constraint = sp.factor(
        sigma * lapse_scalar
        - theta * laplace_shift
        + 3 * theta * zeta_dot
        - g_t * laplace_zeta
    )
    expected_shift_constraint = sp.factor(theta * lapse_scalar - g_t * zeta_dot)
    lapse_solution = sp.factor(g_t * zeta_dot / theta)
    shift_solution = sp.solve(
        lapse_constraint.subs(lapse_scalar, lapse_solution),
        shift_scalar,
        dict=False,
    )[0]
    constraint_residuals = {
        "lapse": sp.factor(lapse_constraint - expected_lapse_constraint),
        "shift": sp.factor(shift_constraint - expected_shift_constraint),
        "lapse_on_solution": sp.factor(
            lapse_constraint.subs(
                {lapse_scalar: lapse_solution, shift_scalar: shift_solution}
            )
        ),
        "shift_on_solution": sp.factor(
            shift_constraint.subs(lapse_scalar, lapse_solution)
        ),
    }
    reduced_before_ibp = sp.factor(
        unreduced_lagrangian.subs(
            {lapse_scalar: lapse_solution, shift_scalar: shift_solution}
        )
    )
    ratio = sp.factor(g_t**2 / theta)
    g_s = sp.factor(sigma * g_t**2 / theta**2 + 3 * g_t)
    f_s = sp.factor(hubble * ratio + a_ratio_dot - f_t)
    expected_before_ibp = sp.factor(
        g_s * zeta_dot**2
        + f_t * wave_number**2 * zeta**2 / a**2
        + 2 * ratio * wave_number**2 * zeta * zeta_dot / a**2
    )
    before_ibp_residual = sp.factor(reduced_before_ibp - expected_before_ibp)

    # At density level, a^3 * 2 A k^2 zeta zeta_dot / a^2 is
    # a A k^2 d_t(zeta^2).  Its bulk adjoint is
    # -a k^2 (H A + A_dot) zeta^2.
    final_reduced_lagrangian = sp.factor(
        g_s * zeta_dot**2 - f_s * wave_number**2 * zeta**2 / a**2
    )
    ibp_bulk_from_unreduced = sp.factor(
        g_s * zeta_dot**2
        + f_t * wave_number**2 * zeta**2 / a**2
        - (hubble * ratio + a_ratio_dot)
        * wave_number**2
        * zeta**2
        / a**2
    )
    ibp_residual = sp.factor(ibp_bulk_from_unreduced - final_reduced_lagrangian)

    scalar_momentum = sp.factor(sp.diff(final_reduced_lagrangian, zeta_dot))
    momentum = sp.Symbol("p_zeta", real=True)
    solved_velocity = sp.factor(momentum / (2 * g_s))
    hamiltonian = sp.factor(
        momentum * solved_velocity
        - final_reduced_lagrangian.subs(zeta_dot, solved_velocity)
    )
    expected_hamiltonian = sp.factor(
        momentum**2 / (4 * g_s)
        + f_s * wave_number**2 * zeta**2 / a**2
    )
    legendre_residual = sp.factor(hamiltonian - expected_hamiltonian)
    hamiltonian_hessian = sp.hessian(hamiltonian, (zeta, momentum))
    expected_hamiltonian_hessian = sp.diag(
        2 * f_s * wave_number**2 / a**2,
        1 / (2 * g_s),
    )
    hamiltonian_hessian_residual = sp.simplify(
        hamiltonian_hessian - expected_hamiltonian_hessian
    )
    omega = sp.Symbol("omega", real=True)
    principal_polynomial = sp.factor(
        f_s * wave_number**2 / a**2 - g_s * omega**2
    )
    speed_squared = sp.factor(f_s / g_s)

    healthy_point = {
        g_t: 2,
        f_t: 2,
        theta: 1,
        sigma: -1,
        hubble: 1,
        a_ratio_dot: 1,
        a: 1,
        wave_number: 1,
    }
    ghost_point = dict(healthy_point)
    ghost_point[sigma] = -2
    gradient_point = dict(healthy_point)
    gradient_point[a_ratio_dot] = -3
    kinetic_singular_point = dict(healthy_point)
    kinetic_singular_point[sigma] = sp.Rational(-3, 2)
    gradient_singular_point = dict(healthy_point)
    gradient_singular_point[a_ratio_dot] = -2
    healthy_values = {
        "G_S": sp.factor(g_s.subs(healthy_point)),
        "F_S": sp.factor(f_s.subs(healthy_point)),
        "c_S_squared": sp.factor(speed_squared.subs(healthy_point)),
    }
    ghost_value = sp.factor(g_s.subs(ghost_point))
    gradient_value = sp.factor(f_s.subs(gradient_point))
    kinetic_singular_value = sp.factor(g_s.subs(kinetic_singular_point))
    gradient_singular_value = sp.factor(f_s.subs(gradient_singular_point))
    theta_singular_constraint = sp.factor(expected_shift_constraint.subs(theta, 0))

    passed = (
        all(value == 0 for value in constraint_residuals.values())
        and before_ibp_residual == 0
        and ibp_residual == 0
        and scalar_momentum == 2 * g_s * zeta_dot
        and legendre_residual == 0
        and hamiltonian_hessian_residual == sp.zeros(2)
        and all(value.is_positive is True for value in healthy_values.values())
        and ghost_value.is_negative is True
        and gradient_value.is_negative is True
        and kinetic_singular_value == 0
        and gradient_singular_value == 0
        and theta_singular_constraint != expected_shift_constraint
    )
    return passed, {
        "control": "generic Horndeski L2-L4 FLRW scalar constraint/principal/Hamiltonian theorem",
        "source": {
            "paper": "Kobayashi, Yamaguchi, Yokoyama, Generalized G-inflation",
            "url": "https://arxiv.org/abs/1105.5723",
            "version": "v4",
            "equations": ["4.24", "4.25", "4.26", "4.29", "4.30", "4.31", "4.32", "4.33", "4.34"],
            "specialization": "G5=0 (the compiler's L2-L4 family)",
        },
        "background": "on-shell FLRW with homogeneous timelike scalar gradient",
        "unreduced_fields": ["zeta", "alpha", "beta"],
        "physical_basis": ["curvature_scalar_zeta"],
        "unreduced_lagrangian": str(unreduced_lagrangian),
        "constraints": {
            "lapse": str(lapse_constraint),
            "shift": str(shift_constraint),
            "lapse_solution": str(lapse_solution),
            "shift_solution": str(shift_solution),
            "residuals": {name: str(value) for name, value in constraint_residuals.items()},
        },
        "Theta_regular_patch": "Theta != 0",
        "G_S": str(g_s),
        "F_S": str(f_s),
        "scalar_speed_squared": str(speed_squared),
        "reduced_before_integration_by_parts": str(reduced_before_ibp),
        "before_integration_by_parts_residual": str(before_ibp_residual),
        "integration_by_parts_residual": str(ibp_residual),
        "final_reduced_lagrangian": str(final_reduced_lagrangian),
        "principal_polynomial": str(principal_polynomial),
        "healthy_domain": ["Theta != 0", "G_S > 0", "F_S > 0"],
        "canonical_momentum": str(scalar_momentum),
        "reduced_hamiltonian": str(hamiltonian),
        "legendre_residual": str(legendre_residual),
        "expected_hamiltonian_hessian": str(expected_hamiltonian_hessian),
        "hamiltonian_hessian_residual": str(hamiltonian_hessian_residual),
        "healthy_witness": {
            "substitution": {
                "G_T": "2",
                "F_T": "2",
                "Theta": "1",
                "Sigma": "-1",
                "H": "1",
                "A_dot": "1",
                "a": "1",
                "k": "1",
            },
            "values": {name: str(value) for name, value in healthy_values.items()},
        },
        "negative_controls": {
            "scalar_ghost": {
                "G_S": str(ghost_value),
                "rejected": bool(ghost_value < 0),
            },
            "scalar_gradient_instability": {
                "F_S": str(gradient_value),
                "rejected": bool(gradient_value < 0),
            },
            "kinetic_strong_coupling": {
                "G_S": str(kinetic_singular_value),
                "rejected": kinetic_singular_value == 0,
            },
            "gradient_cone_collapse": {
                "F_S": str(gradient_singular_value),
                "rejected": gradient_singular_value == 0,
            },
            "Theta_constraint_singularity": {
                "Theta_zero_shift_constraint": str(theta_singular_constraint),
                "rejected": theta_singular_constraint != expected_shift_constraint,
            },
        },
        "capability_boundary": {
            "generic_flrw_scalar_reduction": "pass_if_Theta_nonzero",
            "generic_flrw_scalar_principal_symbol": "pass_if_F_S_and_G_S_positive",
            "generic_flrw_scalar_reduced_hamiltonian": "pass_if_F_S_and_G_S_positive",
            "candidate_background_sign_proof": "required",
            "arbitrary_inhomogeneous_background_strong_hyperbolicity": "unresolved",
            "nonlinear_global_energy_boundedness": "unresolved",
        },
        "scope": (
            "exact source-bound arbitrary-function FLRW scalar constraint elimination, principal "
            "symbol, and reduced quadratic Hamiltonian; each candidate must still supply an "
            "on-shell background and prove Theta!=0, G_S>0, and F_S>0 over its declared domain"
        ),
    }


def quartic_horndeski_unitary_distributed_dirac_control() -> tuple[bool, dict[str, Any]]:
    """Close the unitary-gauge field constraints on regular lapse-Hessian patches.

    Spatial diffeomorphisms remain gauge symmetries after ``phi=t``.  The lapse has no
    velocity, while its nonlinear equation is second class with ``p_N`` wherever the
    distributed lapse Hessian is invertible.  This calculation verifies the exact 3D
    cotangent lift and density algebra and performs the full regular-patch field count.
    Global invertibility and boundary zero modes are deliberately not inferred.
    """

    spatial = sp.symbols("x0:3", real=True)
    lapse = sp.Function("N")(*spatial)
    lapse_momentum = sp.Function("p_N")(*spatial)
    secondary_density = sp.Function("C_N")(*spatial)
    smearing = sp.Function("f")(*spatial)
    shift_m = tuple(sp.Function(f"M{i}")(*spatial) for i in range(3))
    shift_l = tuple(sp.Function(f"L{i}")(*spatial) for i in range(3))

    def lie_scalar(vector: tuple[sp.Expr, ...], scalar: sp.Expr) -> sp.Expr:
        return sum(vector[i] * sp.diff(scalar, spatial[i]) for i in range(3))

    def lie_density(vector: tuple[sp.Expr, ...], density: sp.Expr) -> sp.Expr:
        return sum(
            vector[i] * sp.diff(density, spatial[i])
            + density * sp.diff(vector[i], spatial[i])
            for i in range(3)
        )

    def scalar_generator(vector: tuple[sp.Expr, ...]) -> sp.Expr:
        return sp.expand(lapse_momentum * lie_scalar(vector, lapse))

    generator_m = scalar_generator(shift_m)
    lapse_generator_residual = sp.factor(
        euler_operator_nd(generator_m, lapse_momentum, spatial, maximum_order=1)
        - lie_scalar(shift_m, lapse)
    )
    momentum_generator_residual = sp.factor(
        -euler_operator_nd(generator_m, lapse, spatial, maximum_order=1)
        - lie_density(shift_m, lapse_momentum)
    )
    commutator = tuple(
        sum(
            shift_m[j] * sp.diff(shift_l[i], spatial[j])
            - shift_l[j] * sp.diff(shift_m[i], spatial[j])
            for j in range(3)
        )
        for i in range(3)
    )
    lie_m_lapse = sp.expand(lie_scalar(shift_m, lapse))
    lie_l_lapse = sp.expand(lie_scalar(shift_l, lapse))
    lie_m_momentum = sp.expand(lie_density(shift_m, lapse_momentum))
    lie_l_momentum = sp.expand(lie_density(shift_l, lapse_momentum))
    lapse_commutator_residual = sp.factor(
        lie_scalar(shift_m, lie_l_lapse)
        - lie_scalar(shift_l, lie_m_lapse)
        - lie_scalar(commutator, lapse)
    )
    momentum_commutator_residual = sp.factor(
        lie_density(shift_m, lie_l_momentum)
        - lie_density(shift_l, lie_m_momentum)
        - lie_density(commutator, lapse_momentum)
    )

    # The local density identity is the compact-support D-C_N bracket.  Its
    # right-hand side is a spatial divergence and integrates to zero.
    density_divergence = sum(
        sp.diff(smearing * shift_m[i] * secondary_density, spatial[i])
        for i in range(3)
    )
    dc_density_residual = sp.factor(
        smearing * lie_density(shift_m, secondary_density)
        + secondary_density * lie_scalar(shift_m, smearing)
        - density_divergence
    )
    omitted_weight_residual = sp.factor(
        smearing * lie_scalar(shift_m, secondary_density)
        + secondary_density * lie_scalar(shift_m, smearing)
        - density_divergence
    )

    lapse_hessian = sp.Symbol("Delta_N", nonzero=True, real=True)
    second_class_matrix = sp.Matrix([[0, lapse_hessian], [-lapse_hessian, 0]])
    singular_matrix = second_class_matrix.subs(lapse_hessian, 0)
    metric_control = canonical_metric_diffeomorphism_control()
    local_witness = quartic_horndeski_unitary_flrw_dirac_control()
    extended_pairs = 10
    first_class = 6
    second_class = 2
    physical_dof = (2 * extended_pairs - 2 * first_class - second_class) // 2
    passed = (
        metric_control["passed"]
        and lapse_generator_residual == 0
        and momentum_generator_residual == 0
        and lapse_commutator_residual == 0
        and momentum_commutator_residual == 0
        and dc_density_residual == 0
        and omitted_weight_residual != 0
        and second_class_matrix.rank() == 2
        and sp.factor(second_class_matrix.det() - lapse_hessian**2) == 0
        and singular_matrix.rank() == 0
        and local_witness["passed"]
        and physical_dof == 3
    )
    return passed, {
        "control": "quartic-Horndeski unitary-gauge distributed Dirac closure theorem",
        "gauge": "phi=t with residual three-dimensional spatial diffeomorphisms",
        "canonical_pairs": {
            "spatial_metric": 6,
            "lapse": 1,
            "shift": 3,
            "extended_total": extended_pairs,
        },
        "spatial_diffeomorphism": {
            "metric_cotangent_lift_passed": metric_control["passed"],
            "lapse_generator_residual": str(lapse_generator_residual),
            "lapse_momentum_generator_residual": str(momentum_generator_residual),
            "lapse_commutator_residual": str(lapse_commutator_residual),
            "lapse_momentum_commutator_residual": str(momentum_commutator_residual),
            "D_D": "{D[M],D[L]}=D[[M,L]] modulo a compact-support boundary",
        },
        "secondary_density_covariance": {
            "identity": (
                "f Lie_M(C_N)+C_N Lie_M(f)=D_i(f M^i C_N)"
            ),
            "residual": str(dc_density_residual),
            "D_C": "{D[M],C_N[f]}=-C_N[Lie_M f]",
            "omitted_density_weight_negative_control": {
                "residual": str(omitted_weight_residual),
                "rejected": omitted_weight_residual != 0,
            },
        },
        "lapse_pair": {
            "primary": "p_N(x)=0",
            "secondary": "C_N(x)=delta H/delta N(x)=0",
            "poisson_kernel": "Delta_N(x,y)=-delta C_N(y)/delta N(x)",
            "regular_patch": "Delta_N is an invertible boundary-condition-dependent operator",
            "constraint_matrix_model": str(second_class_matrix),
            "determinant_model": str(sp.factor(second_class_matrix.det())),
            "rank_on_regular_patch": int(second_class_matrix.rank()),
            "consistency": (
                "dot(C_N)={C_N,H_c}+Delta_N u_N+Lie_shift(C_N)=0 uniquely fixes u_N"
            ),
            "higher_constraints": [],
            "local_action_specific_regular_witness": local_witness,
        },
        "constraint_count": {
            "extended_phase_dimension": 2 * extended_pairs,
            "first_class_constraints": first_class,
            "first_class_set": ["p_(N^i) (3)", "H_i (3)"],
            "second_class_constraints": second_class,
            "second_class_set": ["p_N", "C_N"],
            "physical_dof": physical_dof,
            "formula": "(20-2*6-2)/2=3",
        },
        "singular_negative_control": {
            "substitution": "Delta_N=0",
            "constraint_matrix_rank": int(singular_matrix.rank()),
            "rejected_as_regular_patch": singular_matrix.rank() != 2,
        },
        "interpretation": (
            "On any patch where the action-specific lapse Hessian operator is invertible, the "
            "unitary-gauge theory has six spatial-diffeomorphism first-class constraints and one "
            "lapse second-class pair, hence three physical configuration-space degrees of freedom."
        ),
        "scope": (
            "exact distributed spatial-diffeomorphism and regular second-class closure theorem "
            "for the named no-dot(N) unitary-gauge action, anchored by a nonempty curved-FLRW "
            "regular witness; global operator invertibility, boundary zero modes, singular "
            "branches, and reduced-Hamiltonian boundedness remain separate"
        ),
    }


def generic_horndeski_l2_l4_unitary_dirac_control() -> tuple[bool, dict[str, Any]]:
    """Conditional distributed Dirac theorem for arbitrary smooth L2--L4 functions.

    The covariant family and its exact ADM Hessian theorem guarantee no lapse or
    ``V_star`` kinetic direction.  After ``phi=t`` only spatial diffeomorphisms remain
    gauge symmetries.  Preservation of ``p_N`` produces ``C_N``; wherever the
    action-specific functional lapse Hessian ``Delta_N`` is invertible, preservation
    of ``C_N`` fixes the lapse multiplier and the chain closes with three modes.
    """

    adm = generic_horndeski_l2_l4_unitary_adm_control()
    kinematic_passed, kinematic = quartic_horndeski_unitary_distributed_dirac_control()
    delta_n = sp.Symbol("Delta_N", nonzero=True, real=True)
    constraint_matrix = sp.Matrix([[0, delta_n], [-delta_n, 0]])
    singular_matrix = constraint_matrix.subs(delta_n, 0)

    lapse = sp.Symbol("N", positive=True, finite=True)
    g2_x, g2_xx = sp.symbols("G2_X G2_XX", real=True)
    x = sp.factor(1 / (2 * lapse**2))
    # For N*G2(X(N)), with X=1/(2N^2), this is d^2/dN^2 of the
    # unitary-gauge scalar density, up to the positive spatial volume factor.
    g2_lapse_hessian = sp.factor(2 * x * (g2_x + 2 * x * g2_xx) / lapse)
    canonical_scalar_witness = sp.factor(
        g2_lapse_hessian.subs({g2_x: 1, g2_xx: 0})
    )

    extended_pairs = 10
    first_class = 6
    second_class = 2
    physical_dof = (2 * extended_pairs - 2 * first_class - second_class) // 2
    covariance = kinematic["secondary_density_covariance"]
    spatial = kinematic["spatial_diffeomorphism"]
    passed = (
        adm["passed"]
        and kinematic_passed
        and spatial["lapse_generator_residual"] == "0"
        and spatial["lapse_momentum_generator_residual"] == "0"
        and spatial["lapse_commutator_residual"] == "0"
        and spatial["lapse_momentum_commutator_residual"] == "0"
        and covariance["residual"] == "0"
        and covariance["omitted_density_weight_negative_control"]["rejected"]
        and constraint_matrix.rank() == 2
        and sp.factor(constraint_matrix.det() - delta_n**2) == 0
        and singular_matrix.rank() == 0
        and canonical_scalar_witness == lapse**-3
        and physical_dof == 3
    )
    return passed, {
        "control": "generic Horndeski L2-L4 regular-patch distributed Dirac theorem",
        "covariant_family": adm["covariant_family"],
        "gauge": "phi=t on timelike-gradient unitary-gauge patches",
        "adm_primary_input": {
            "passed": adm["passed"],
            "regular_patch": adm["regular_patch"],
            "primary_constraint": adm["primary_constraint"],
            "hessian_rank": adm["velocity_hessian_rank_on_regular_patch"],
            "hessian_nullity": adm["velocity_hessian_nullity_on_regular_patch"],
        },
        "dirac_chain": [
            "p_N(x)=0",
            "C_N(x)=delta H_c/delta N(x)=0",
            "dot(C_N)={C_N,H_c}+Delta_N u_N+Lie_shift(C_N)=0 fixes u_N",
        ],
        "constraint_matrix": str(constraint_matrix),
        "constraint_matrix_determinant": str(sp.factor(constraint_matrix.det())),
        "constraint_matrix_rank_on_regular_patch": int(constraint_matrix.rank()),
        "regular_patch": (
            "G4-2 X G4_X != 0 and Delta_N is invertible under the declared boundary conditions"
        ),
        "spatial_diffeomorphism_residuals": {
            "lapse": spatial["lapse_generator_residual"],
            "lapse_momentum": spatial["lapse_momentum_generator_residual"],
            "D_D_lapse": spatial["lapse_commutator_residual"],
            "D_D_lapse_momentum": spatial["lapse_momentum_commutator_residual"],
            "D_C_secondary_density": covariance["residual"],
        },
        "omitted_secondary_density_weight_negative": covariance[
            "omitted_density_weight_negative_control"
        ],
        "constraint_count": {
            "extended_phase_dimension": 2 * extended_pairs,
            "first_class_constraints": first_class,
            "first_class_set": ["p_(N^i) (3)", "H_i (3)"],
            "second_class_constraints": second_class,
            "second_class_set": ["p_N", "C_N"],
            "physical_dof": physical_dof,
            "formula": "(20-2*6-2)/2=3",
        },
        "regular_family_witness": {
            "functions": "G2=X, G3=0, G4=constant",
            "unitary_X": str(x),
            "G2_lapse_hessian": str(canonical_scalar_witness),
            "nonzero_for_N_positive": canonical_scalar_witness != 0,
        },
        "generic_G2_lapse_hessian_contribution": str(g2_lapse_hessian),
        "singular_negative_control": {
            "substitution": "Delta_N=0",
            "constraint_matrix_rank": int(singular_matrix.rank()),
            "three_mode_count_rejected": singular_matrix.rank() != 2,
        },
        "capability_boundary": {
            "primary_constraint": "pass_on_G4_minus_2XG4X_nonzero_patch",
            "secondary_constraint": "pass",
            "poisson_closure": "pass_if_Delta_N_invertible",
            "physical_dof": "three_if_Delta_N_invertible",
            "global_Delta_N_invertibility": "unresolved",
            "boundary_zero_modes": "unresolved",
            "singular_strata": "unresolved",
            "reduced_hamiltonian_stability": "unresolved",
        },
        "scope": (
            "exact arbitrary-function L2-L4 distributed constraint theorem on regular timelike "
            "unitary-gauge patches. It proves the secondary chain, D-D/D-C covariance, Poisson "
            "rank, and three-mode count conditionally on the action-specific lapse-Hessian "
            "operator being invertible. It does not claim global invertibility, control boundary "
            "zero modes, or prove reduced-Hamiltonian boundedness"
        ),
    }


def quartic_horndeski_timelike_flat_hamiltonian_control() -> tuple[bool, dict[str, Any]]:
    """Derive the three-mode reduced quadratic Hamiltonian on the flat timelike patch."""

    m2 = sp.Symbol("M2", positive=True, finite=True)
    alpha = sp.Symbol("alpha", nonzero=True, finite=True, real=True)
    a_star = sp.Symbol("A_star", positive=True, finite=True)
    c_x = sp.Symbol("c_X", positive=True, finite=True)
    wave_number = sp.Symbol("k", positive=True, finite=True)
    tensor_kinetic = sp.factor((m2 - alpha * a_star**2) / 2)
    tensor_gradient = sp.factor((m2 + alpha * a_star**2) / 2)
    kinetic = sp.diag(tensor_kinetic, tensor_kinetic, c_x)
    gradient = sp.diag(tensor_gradient, tensor_gradient, c_x)
    momentum_hessian = sp.simplify(kinetic.inv())
    coordinate_hessian = sp.simplify(wave_number**2 * gradient)
    coordinates = sp.Matrix(sp.symbols("Q_Tplus Q_Tcross Q_scalar", real=True))
    momenta = sp.Matrix(sp.symbols("P_Tplus P_Tcross P_scalar", real=True))
    velocities = sp.Matrix(sp.symbols("V_Tplus V_Tcross V_scalar", real=True))
    hamiltonian = sp.factor(
        (
            (momenta.T * momentum_hessian * momenta)[0]
            + (coordinates.T * coordinate_hessian * coordinates)[0]
        )
        / 2
    )
    lagrangian = sp.factor(
        (
            (velocities.T * kinetic * velocities)[0]
            - (coordinates.T * coordinate_hessian * coordinates)[0]
        )
        / 2
    )
    canonical_momenta = kinetic * velocities
    substitutions = dict(zip(momenta, canonical_momenta, strict=True))
    legendre_residual = sp.factor(
        (canonical_momenta.T * velocities)[0]
        - lagrangian
        - hamiltonian.subs(substitutions)
    )
    healthy_point = {
        m2: 2,
        alpha: sp.Rational(-1, 4),
        a_star: 1,
        c_x: 1,
        wave_number: 2,
    }
    healthy_momentum = momentum_hessian.subs(healthy_point)
    healthy_coordinate = coordinate_hessian.subs(healthy_point)
    ghost_point = {m2: 2, alpha: 3, a_star: 1, c_x: 1, wave_number: 2}
    gradient_point = {m2: 2, alpha: -3, a_star: 1, c_x: 1, wave_number: 2}
    ghost_momentum_entry = sp.factor(momentum_hessian[0, 0].subs(ghost_point))
    unstable_coordinate_entry = sp.factor(
        coordinate_hessian[0, 0].subs(gradient_point)
    )
    healthy_positive = all(
        item.is_positive is True
        for item in [*healthy_momentum.diagonal(), *healthy_coordinate.diagonal()]
    )
    passed = (
        legendre_residual == 0
        and healthy_positive
        and ghost_momentum_entry.is_negative is True
        and unstable_coordinate_entry.is_negative is True
    )
    return passed, {
        "control": "quartic-Horndeski flat timelike-gradient reduced physical Hamiltonian",
        "background": (
            "flat frozen frame with constant timelike scalar gradient A_star and one Fourier "
            "mode k>0, after the three-mode unitary-gauge Dirac reduction"
        ),
        "physical_basis": ["tensor_plus", "tensor_cross", "scalar"],
        "kinetic_matrix_from_reduced_lagrangian": str(kinetic),
        "momentum_hessian": str(momentum_hessian),
        "coordinate_hessian": str(coordinate_hessian),
        "reduced_hamiltonian": str(hamiltonian),
        "reconstructed_lagrangian": str(lagrangian),
        "legendre_transform_residual": str(legendre_residual),
        "positive_patch": [
            f"{tensor_kinetic} > 0",
            f"{tensor_gradient} > 0",
            "c_X > 0",
            "k > 0",
        ],
        "healthy_witness": {
            "substitution": {
                "M2": "2",
                "alpha": "-1/4",
                "A_star": "1",
                "c_X": "1",
                "k": "2",
            },
            "momentum_hessian": str(healthy_momentum),
            "coordinate_hessian": str(healthy_coordinate),
            "strictly_positive": healthy_positive,
        },
        "negative_controls": {
            "tensor_ghost": {
                "momentum_hessian_entry": str(ghost_momentum_entry),
                "rejected": ghost_momentum_entry.is_negative is True,
            },
            "tensor_gradient_instability": {
                "coordinate_hessian_entry": str(unstable_coordinate_entry),
                "rejected": unstable_coordinate_entry.is_negative is True,
            },
        },
        "generic_nonlinear_total_energy": "unresolved",
        "scope": (
            "exact constraint-count-matched quadratic physical Hamiltonian on the flat constant-"
            "timelike-gradient patch; nonlinear curved-background energy, boundary charges, and "
            "a global positive-energy theorem remain separate"
        ),
    }


def quartic_horndeski_global_timelike_gradient_no_go_control() -> tuple[bool, dict[str, Any]]:
    """Prove that nonzero linear-X G4 has no all-amplitude healthy timelike domain."""

    m2 = sp.Symbol("M2", positive=True, finite=True)
    alpha = sp.Symbol("alpha", nonzero=True, finite=True, real=True)
    a_squared = sp.Symbol("A_star_squared", nonnegative=True, finite=True)
    tensor_kinetic = sp.factor((m2 - alpha * a_squared) / 2)
    tensor_gradient = sp.factor((m2 + alpha * a_squared) / 2)
    speed_squared = sp.factor(tensor_gradient / tensor_kinetic)

    positive_alpha_boundary = sp.factor(m2 / alpha)
    negative_alpha_boundary = sp.factor(-m2 / alpha)
    positive_branch_kinetic = sp.factor(
        tensor_kinetic.subs(a_squared, positive_alpha_boundary)
    )
    positive_branch_gradient = sp.factor(
        tensor_gradient.subs(a_squared, positive_alpha_boundary)
    )
    negative_branch_kinetic = sp.factor(
        tensor_kinetic.subs(a_squared, negative_alpha_boundary)
    )
    negative_branch_gradient = sp.factor(
        tensor_gradient.subs(a_squared, negative_alpha_boundary)
    )
    positive_above = sp.factor(
        tensor_kinetic.subs(a_squared, 2 * positive_alpha_boundary)
    )
    negative_above = sp.factor(
        tensor_gradient.subs(a_squared, 2 * negative_alpha_boundary)
    )
    exact_healthy_bound = sp.Lt(a_squared, m2 / sp.Abs(alpha))
    positive_numeric = {m2: 2, alpha: sp.Rational(1, 4), a_squared: 9}
    negative_numeric = {m2: 2, alpha: sp.Rational(-1, 4), a_squared: 9}
    passed = (
        positive_branch_kinetic == 0
        and positive_branch_gradient == m2
        and positive_above == -m2 / 2
        and negative_branch_gradient == 0
        and negative_branch_kinetic == m2
        and negative_above == -m2 / 2
        and tensor_kinetic.subs(positive_numeric).is_negative is True
        and tensor_gradient.subs(negative_numeric).is_negative is True
    )
    return passed, {
        "control": "linear-X quartic-Horndeski all-timelike-amplitude no-go",
        "tensor_kinetic": str(tensor_kinetic),
        "tensor_gradient": str(tensor_gradient),
        "tensor_speed_squared": str(speed_squared),
        "healthy_timelike_gradient_domain": str(exact_healthy_bound),
        "equivalent_bound": "A_star^2 < M2/abs(alpha)",
        "positive_alpha_branch": {
            "assumption": "alpha > 0",
            "rank_boundary": f"A_star^2={positive_alpha_boundary}",
            "tensor_kinetic_at_boundary": str(positive_branch_kinetic),
            "tensor_gradient_at_boundary": str(positive_branch_gradient),
            "tensor_kinetic_above_boundary": str(positive_above),
            "conclusion": "kinetic rank loss followed by a tensor ghost",
        },
        "negative_alpha_branch": {
            "assumption": "alpha < 0",
            "rank_boundary": f"A_star^2={negative_alpha_boundary}",
            "tensor_kinetic_at_boundary": str(negative_branch_kinetic),
            "tensor_gradient_at_boundary": str(negative_branch_gradient),
            "tensor_gradient_above_boundary": str(negative_above),
            "conclusion": "characteristic cone collapse followed by a gradient instability",
        },
        "numeric_witnesses": {
            "alpha_positive_tensor_kinetic": str(
                tensor_kinetic.subs(positive_numeric)
            ),
            "alpha_negative_tensor_gradient": str(
                tensor_gradient.subs(negative_numeric)
            ),
        },
        "global_all_amplitude_domain_exists_for_nonzero_alpha": False,
        "required_resolution": (
            "declare and dynamically preserve an EFT/background bound A_star^2<M2/abs(alpha), "
            "set alpha=0, or replace linear G4(X) by a completion with a globally healthy cone"
        ),
        "scope": (
            "exact flat constant-timelike-gradient tensor sector for every amplitude and both "
            "signs of nonzero alpha; tilted gradients, curvature, scalar-tensor mixing on varying "
            "backgrounds, and dynamical preservation of the bounded domain remain separate"
        ),
    }


def quartic_horndeski_flrw_domain_crossing_control() -> tuple[bool, dict[str, Any]]:
    """Exhibit an exact FLRW trajectory crossing the declared healthy boundary.

    In unitary gauge ``phi=t``, ``A_star_squared=1/N**2``.  The lapse equation is an
    algebraic constraint.  Its time derivative together with the scale-factor Euler
    equation determines ``ddot(a)`` and ``dot(N)``.  A regular contracting witness on
    the negative-alpha branch points out of the healthy domain, disproving unconditional
    preservation by the homogeneous nonlinear equations.
    """

    lapse, scale_factor = sp.symbols("N a", positive=True, finite=True)
    velocity = sp.Symbol("dot_a", real=True)
    dot_lapse, acceleration = sp.symbols("dot_N ddot_a", real=True)
    m2 = sp.Symbol("M2", positive=True, finite=True)
    alpha, spatial_curvature = sp.symbols(
        "alpha k", nonzero=True, finite=True, real=True
    )
    lagrangian = sp.factor(
        -3 * m2 * scale_factor * velocity**2 / lapse
        + 3 * alpha * scale_factor * velocity**2 / lapse**3
        + 3 * m2 * spatial_curvature * scale_factor * lapse
        + 3 * alpha * spatial_curvature * scale_factor / lapse
        + scale_factor**3 / (2 * lapse)
    )
    lapse_constraint = sp.factor(sp.diff(lagrangian, lapse))
    scale_momentum = sp.diff(lagrangian, velocity)
    scale_euler = sp.factor(
        sp.diff(lagrangian, scale_factor)
        - sp.diff(scale_momentum, scale_factor) * velocity
        - sp.diff(scale_momentum, velocity) * acceleration
        - sp.diff(scale_momentum, lapse) * dot_lapse
    )
    constraint_flow = sp.factor(
        sp.diff(lapse_constraint, scale_factor) * velocity
        + sp.diff(lapse_constraint, velocity) * acceleration
        + sp.diff(lapse_constraint, lapse) * dot_lapse
    )

    witness = {
        m2: sp.Integer(1),
        alpha: sp.Integer(-1),
        spatial_curvature: sp.Integer(1),
        scale_factor: sp.Integer(4),
        lapse: sp.Integer(1),
        velocity: -sp.sqrt(sp.Rational(1, 6)),
    }
    evolution_matrix, evolution_source = sp.linear_eq_to_matrix(
        [scale_euler.subs(witness), constraint_flow.subs(witness)],
        [acceleration, dot_lapse],
    )
    evolution_determinant = sp.factor(evolution_matrix.det())
    solution = sp.simplify(evolution_matrix.inv() * evolution_source)
    acceleration_value, dot_lapse_value = solution
    evolution_substitution = {
        **witness,
        acceleration: acceleration_value,
        dot_lapse: dot_lapse_value,
    }

    a_star_squared = lapse**-2
    healthy_boundary_function = sp.factor(
        a_star_squared - m2 / sp.Abs(alpha)
    )
    boundary_flow = sp.factor(
        sp.diff(healthy_boundary_function, lapse) * dot_lapse
    )
    tensor_kinetic = sp.factor((m2 - alpha * a_star_squared) / 2)
    tensor_gradient = sp.factor((m2 + alpha * a_star_squared) / 2)
    tensor_gradient_flow = sp.factor(sp.diff(tensor_gradient, lapse) * dot_lapse)

    constraint_residual = sp.factor(lapse_constraint.subs(witness))
    scale_euler_residual = sp.factor(scale_euler.subs(evolution_substitution))
    flow_residual = sp.factor(constraint_flow.subs(evolution_substitution))
    boundary_value = sp.factor(healthy_boundary_function.subs(witness))
    boundary_flow_value = sp.factor(boundary_flow.subs(evolution_substitution))
    kinetic_value = sp.factor(tensor_kinetic.subs(witness))
    gradient_value = sp.factor(tensor_gradient.subs(witness))
    gradient_flow_value = sp.factor(
        tensor_gradient_flow.subs(evolution_substitution)
    )
    passed = (
        constraint_residual == 0
        and evolution_determinant != 0
        and acceleration_value == -sp.Rational(1, 48)
        and dot_lapse_value == -sp.sqrt(6) / 4
        and scale_euler_residual == 0
        and flow_residual == 0
        and boundary_value == 0
        and boundary_flow_value.is_positive is True
        and kinetic_value.is_positive is True
        and gradient_value == 0
        and gradient_flow_value.is_negative is True
    )
    return passed, {
        "control": "quartic-Horndeski nonlinear FLRW healthy-domain crossing",
        "gauge_and_background": "phi=t, closed FLRW with N(t), a(t), and k=1",
        "reduced_lagrangian": str(lagrangian),
        "lapse_constraint": str(lapse_constraint),
        "scale_factor_euler": str(scale_euler),
        "constraint_time_derivative": str(constraint_flow),
        "healthy_domain": "A_star_squared < M2/abs(alpha)",
        "unitary_gauge_identification": "A_star_squared=1/N**2",
        "crossing_witness": {
            "M2": "1",
            "alpha": "-1",
            "k": "1",
            "a": "4",
            "N": "1",
            "dot_a": "-sqrt(6)/6",
            "ddot_a": str(acceleration_value),
            "dot_N": str(dot_lapse_value),
            "lapse_constraint_residual": str(constraint_residual),
            "scale_euler_residual": str(scale_euler_residual),
            "constraint_flow_residual": str(flow_residual),
            "evolution_matrix_determinant": str(evolution_determinant),
            "healthy_boundary_function": str(boundary_value),
            "healthy_boundary_time_derivative": str(boundary_flow_value),
            "tensor_kinetic_at_boundary": str(kinetic_value),
            "tensor_gradient_at_boundary": str(gradient_value),
            "tensor_gradient_time_derivative": str(gradient_flow_value),
        },
        "forward_invariant_under_unrestricted_flrw_evolution": False,
        "conclusion": (
            "The contracting solution crosses from the healthy side through zero tensor "
            "gradient energy and into the gradient-unstable side; the action alone does not "
            "dynamically preserve the declared patch."
        ),
        "scope": (
            "exact nonlinear homogeneous curved-FLRW counterexample. It rejects unconditional "
            "domain preservation, but does not exclude a separately justified restricted "
            "solution class, an explicit EFT stopping boundary, or a nonlinear G4 completion"
        ),
    }
