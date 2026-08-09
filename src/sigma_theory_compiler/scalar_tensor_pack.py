from __future__ import annotations

import ast
import hashlib
import json
import math
from functools import lru_cache
from itertools import product
from typing import Any

import sympy as sp

from .dhost import generic_horndeski_l2_l4_unitary_adm_control
from .g4_curved_witness import (
    generic_g4_curved_rnc_witness_control,
    generic_g4_curved_symbolic_rnc_control,
)
from .horndeski import (
    generic_cubic_horndeski_bssn_hyperbolicity_control,
    generic_horndeski_l2_l4_flrw_scalar_reduction_control,
    generic_horndeski_l2_l4_tensor_stability_control,
    generic_horndeski_l2_l4_unitary_dirac_control,
    generic_kessence_nonlinear_adm_legendre_control,
    generic_kessence_timelike_principal_hamiltonian_control,
)

SCHEMA_VERSION = "sigma-scalar-tensor-pack-1.0"
_FUNCTION_NAMES = ("g2", "g3", "g4")
_SAFE_FUNCTIONS: dict[str, Any] = {
    "sqrt": sp.sqrt,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
}


@lru_cache(maxsize=1)
def generic_g2_variation_noether_control() -> tuple[bool, dict[str, Any]]:
    """Verify the arbitrary-G2 metric/field Noether identity on an arbitrary local jet."""

    signature = (-1, 1, 1, 1)
    gradient = sp.symbols("p_0:4", real=True)
    hessian_symbols = sp.symbols(
        "H_00 H_01 H_02 H_03 H_11 H_12 H_13 H_22 H_23 H_33", real=True
    )
    hessian = sp.zeros(4, 4)
    cursor = 0
    for row in range(4):
        for column in range(row, 4):
            hessian[row, column] = hessian_symbols[cursor]
            hessian[column, row] = hessian_symbols[cursor]
            cursor += 1
    g2_u, g2_x, g2_xu, g2_xx = sp.symbols(
        "g2_u g2_x g2_xu g2_xx", real=True
    )
    x_gradient = [
        -sum(
            signature[index] * gradient[index] * hessian[index, covector]
            for index in range(4)
        )
        for covector in range(4)
    ]
    g2_x_gradient = [
        g2_xu * gradient[index] + g2_xx * x_gradient[index] for index in range(4)
    ]
    box_phi = sum(signature[index] * hessian[index, index] for index in range(4))
    derivative_dot_gradient = sum(
        signature[index] * g2_x_gradient[index] * gradient[index] for index in range(4)
    )
    scalar_euler = sp.factor(g2_u + derivative_dot_gradient + g2_x * box_phi)
    residuals: list[sp.Expr] = []
    corrupted_residuals: list[sp.Expr] = []
    for covector in range(4):
        raised_hessian_gradient = sum(
            signature[index] * gradient[index] * hessian[index, covector]
            for index in range(4)
        )
        pressure_gradient = g2_u * gradient[covector] + g2_x * x_gradient[covector]
        divergence = sp.factor(
            derivative_dot_gradient * gradient[covector]
            + g2_x * box_phi * gradient[covector]
            + g2_x * raised_hessian_gradient
            + pressure_gradient
        )
        residuals.append(sp.factor(divergence - scalar_euler * gradient[covector]))
        corrupted_divergence = sp.factor(
            derivative_dot_gradient * gradient[covector]
            + g2_x * box_phi * gradient[covector]
            + g2_x * raised_hessian_gradient
            - pressure_gradient
        )
        corrupted_residuals.append(
            sp.factor(corrupted_divergence - scalar_euler * gradient[covector])
        )
    passed = all(item == 0 for item in residuals) and any(
        item != 0 for item in corrupted_residuals
    )
    return passed, {
        "status": "pass" if passed else "fail",
        "x_definition": "x=-nabla_mu(phi)nabla^mu(phi)/2 in Lambda_phi=1 units",
        "field_euler_coefficient": (
            "E_phi=g2_u+nabla_mu(g2_x)nabla^mu(phi)+g2_x box(phi)"
        ),
        "metric_stress_tensor": (
            "T_mu_nu=g2_x nabla_mu(phi)nabla_nu(phi)+g2 g_mu_nu"
        ),
        "identity": "nabla^mu T_mu_nu=E_phi nabla_nu(phi)",
        "local_jet_residuals": [str(item) for item in residuals],
        "corrupted_metric_pressure_sign_residuals": [
            str(item) for item in corrupted_residuals
        ],
        "corrupted_sign_rejected": any(item != 0 for item in corrupted_residuals),
        "scope": (
            "exact arbitrary local scalar-gradient/Hessian jet and arbitrary g2 derivative "
            "coefficients; connection vanishes at the chosen Riemann-normal-coordinate point, "
            "but curvature is unrestricted and does not enter this first-derivative matter sector"
        ),
    }


@lru_cache(maxsize=1)
def generic_g3_variation_noether_control() -> tuple[bool, dict[str, Any]]:
    """Certify the covariant variation and Noether identity of -G3(phi,X) box(phi).

    The calculation is kept in a basis of independent covariant contractions.  In
    particular, it uses the scalar Hessian commutator rather than setting curvature
    or third derivatives to zero.  Symbols named ``p_dot_*`` are contractions with
    p_mu=nabla_mu(phi), q_mu=nabla_mu(X), and theta=box(phi).
    """

    x = sp.Symbol("x", real=True)
    theta = sp.Symbol("theta", real=True)
    hessian_squared = sp.Symbol("hessian_squared", real=True)
    ricci_pp = sp.Symbol("ricci_pp", real=True)
    p_dot_q = sp.Symbol("p_dot_q", real=True)
    q_squared = sp.Symbol("q_squared", real=True)
    p_dot_grad_theta = sp.Symbol("p_dot_grad_theta", real=True)
    g_phi, g_x = sp.symbols("g_phi g_x", real=True)
    g_phiphi, g_phix, g_xx = sp.symbols(
        "g_phiphi g_phix g_xx", real=True
    )

    grad_gx_dot_p = -2 * x * g_phix + g_xx * p_dot_q
    grad_gphi_dot_p = -2 * x * g_phiphi + g_phix * p_dot_q
    grad_gx_dot_q = g_phix * p_dot_q + g_xx * q_squared
    # q_mu=nabla_mu X=-p^nu phi_(mu nu).  The Ricci term is essential on a
    # curved background and fixes the commutator convention used by the control.
    divergence_q = -hessian_squared - p_dot_grad_theta - ricci_pp

    compact_euler = sp.factor(
        -g_phi * theta
        - (grad_gx_dot_p * theta + g_x * p_dot_grad_theta + g_x * theta**2)
        - (
            grad_gphi_dot_p
            + g_phi * theta
            + grad_gx_dot_q
            + g_x * divergence_q
        )
    )
    second_order_euler = sp.factor(
        -2 * g_phi * theta
        + (2 * x * g_phix - g_xx * p_dot_q) * theta
        - g_x * theta**2
        + 2 * x * g_phiphi
        - 2 * g_phix * p_dot_q
        - g_xx * q_squared
        + g_x * (hessian_squared + ricci_pp)
    )
    third_derivative_cancellation = sp.factor(compact_euler - second_order_euler)

    # Independently differentiate every component of the Hilbert tensor on an
    # arbitrary flat third jet.  This does not use the compact divergence
    # reduction below, so it catches product-rule, coefficient, and index-sign
    # errors in either the stress tensor or scalar Euler coefficient.
    signature = (-1, 1, 1, 1)
    flat_p = sp.symbols("flat_p_0:4", real=True)
    flat_h_symbols = sp.symbols(
        "flat_H_00 flat_H_01 flat_H_02 flat_H_03 flat_H_11 flat_H_12 "
        "flat_H_13 flat_H_22 flat_H_23 flat_H_33",
        real=True,
    )
    flat_h = sp.zeros(4, 4)
    cursor = 0
    for row in range(4):
        for column in range(row, 4):
            flat_h[row, column] = flat_h_symbols[cursor]
            flat_h[column, row] = flat_h_symbols[cursor]
            cursor += 1
    flat_j: dict[tuple[int, int, int], sp.Symbol] = {}

    def third_jet(first: int, second: int, third: int) -> sp.Symbol:
        key = tuple(sorted((first, second, third)))
        if key not in flat_j:
            flat_j[key] = sp.Symbol("flat_J_" + "".join(map(str, key)), real=True)
        return flat_j[key]

    flat_theta = sum(signature[index] * flat_h[index, index] for index in range(4))
    flat_dtheta = [
        sum(
            signature[index] * third_jet(mu, index, index)
            for index in range(4)
        )
        for mu in range(4)
    ]
    flat_q = [
        -sum(
            signature[index] * flat_p[index] * flat_h[mu, index]
            for index in range(4)
        )
        for mu in range(4)
    ]
    flat_dq = [
        [
            -sum(
                signature[index]
                * (
                    flat_h[mu, index] * flat_h[nu, index]
                    + flat_p[index] * third_jet(mu, nu, index)
                )
                for index in range(4)
            )
            for nu in range(4)
        ]
        for mu in range(4)
    ]
    flat_u = [g_phi * flat_p[index] + g_x * flat_q[index] for index in range(4)]
    flat_da = [
        g_phiphi * flat_p[index] + g_phix * flat_q[index]
        for index in range(4)
    ]
    flat_db = [
        g_phix * flat_p[index] + g_xx * flat_q[index] for index in range(4)
    ]
    flat_du = [
        [
            flat_da[mu] * flat_p[nu]
            + g_phi * flat_h[mu, nu]
            + flat_db[mu] * flat_q[nu]
            + g_x * flat_dq[mu][nu]
            for nu in range(4)
        ]
        for mu in range(4)
    ]
    flat_du_dot_p = [
        sum(
            signature[index]
            * (flat_du[mu][index] * flat_p[index] + flat_u[index] * flat_h[mu, index])
            for index in range(4)
        )
        for mu in range(4)
    ]

    def flat_d_stress(derivative: int, first: int, second: int) -> sp.Expr:
        metric_entry = signature[first] if first == second else 0
        return sp.expand(
            -(flat_db[derivative] * flat_theta + g_x * flat_dtheta[derivative])
            * flat_p[first]
            * flat_p[second]
            - g_x
            * flat_theta
            * (
                flat_h[derivative, first] * flat_p[second]
                + flat_p[first] * flat_h[derivative, second]
            )
            - flat_du[derivative][first] * flat_p[second]
            - flat_u[first] * flat_h[derivative, second]
            - flat_du[derivative][second] * flat_p[first]
            - flat_u[second] * flat_h[derivative, first]
            + metric_entry * flat_du_dot_p[derivative]
        )

    flat_divergence = [
        sp.factor(
            sum(
                signature[index] * flat_d_stress(index, index, nu)
                for index in range(4)
            )
        )
        for nu in range(4)
    ]
    flat_div_b_theta_p = sum(
        signature[index]
        * (
            (flat_db[index] * flat_theta + g_x * flat_dtheta[index])
            * flat_p[index]
            + g_x * flat_theta * flat_h[index, index]
        )
        for index in range(4)
    )
    flat_box_g = sum(
        signature[index] * flat_du[index][index] for index in range(4)
    )
    flat_euler = sp.expand(-g_phi * flat_theta - flat_div_b_theta_p - flat_box_g)
    flat_noether_residuals = [
        sp.factor(flat_divergence[index] - flat_euler * flat_p[index])
        for index in range(4)
    ]

    # A direct divergence of
    # T_mn=-G_X theta p_m p_n-2 nabla_(m G p_n)+g_mn nabla_r G p^r
    # reduces by Hessian symmetry and d(dG)=0 to E_phi p_n.  Retain p_n and
    # q_n as independent component symbols so an omitted braiding term cannot
    # disappear accidentally.
    p_components = sp.symbols("p_0:4", real=True)
    q_components = sp.symbols("q_0:4", real=True)
    p_dot_grad_gx_theta = sp.Symbol("p_dot_grad_gx_theta", real=True)
    box_g = sp.Symbol("box_g", real=True)
    divergence_core = p_dot_grad_gx_theta + g_x * theta**2
    compact_euler_for_identity = sp.factor(
        -g_phi * theta - divergence_core - box_g
    )
    divergence_components = [
        sp.factor(-(divergence_core + box_g + g_phi * theta) * component)
        for component in p_components
    ]
    noether_residuals = [
        sp.factor(divergence - compact_euler_for_identity * component)
        for divergence, component in zip(divergence_components, p_components, strict=True)
    ]

    # Corrupt the Hilbert tensor by deleting -G_X box(phi) p_m p_n.  Its
    # divergence then leaves a nonzero p_n/q_n vector on an arbitrary jet.
    corrupted_residuals = [
        sp.factor(
            divergence_core * p_component
            - g_x * theta * q_component
        )
        for p_component, q_component in zip(
            p_components, q_components, strict=True
        )
    ]
    # A second negative control removes the Ricci commutator from div(q).  It
    # must fail whenever both G_X and Ricci(p,p) are nonzero.
    omitted_ricci_residual = sp.factor(g_x * ricci_pp)

    passed = (
        third_derivative_cancellation == 0
        and all(item == 0 for item in flat_noether_residuals)
        and all(item == 0 for item in noether_residuals)
        and all(item != 0 for item in corrupted_residuals)
        and omitted_ricci_residual != 0
    )
    return passed, {
        "status": "pass" if passed else "fail",
        "lagrangian": "L3=-G3(phi,X) box(phi)",
        "x_definition": "X=-nabla_mu(phi)nabla^mu(phi)/2",
        "boundary_equivalent_density": (
            "nabla_mu(G3)nabla^mu(phi)=-2 X G3_phi+G3_X "
            "nabla_mu(X)nabla^mu(phi)"
        ),
        "field_euler_compact": (
            "E_phi=-G3_phi theta-nabla_mu(G3_X theta p^mu)-box(G3)"
        ),
        "field_euler_second_order": str(second_order_euler),
        "metric_stress_tensor": (
            "T_mu_nu=-G3_X theta p_mu p_nu-2 nabla_(mu(G3) p_nu)"
            "+g_mu_nu nabla_rho(G3)p^rho"
        ),
        "identity": "nabla^mu T_mu_nu=E_phi p_nu",
        "source": "Kobayashi-Yamaguchi-Yokoyama 2011, equations B.3, B.7, and B.11",
        "source_url": "https://arxiv.org/abs/1105.5723",
        "hessian_commutator": (
            "nabla_mu H^(mu)_nu=nabla_nu theta+R_nu_rho p^rho"
        ),
        "third_derivative_cancellation_residual": str(
            third_derivative_cancellation
        ),
        "flat_arbitrary_third_jet_noether_residuals": [
            str(item) for item in flat_noether_residuals
        ],
        "noether_residuals": [str(item) for item in noether_residuals],
        "omitted_braiding_stress_residuals": [
            str(item) for item in corrupted_residuals
        ],
        "omitted_braiding_stress_rejected": all(
            item != 0 for item in corrupted_residuals
        ),
        "omitted_ricci_commutator_residual": str(omitted_ricci_residual),
        "omitted_ricci_commutator_rejected": omitted_ricci_residual != 0,
        "scope": (
            "exact four-dimensional covariant contraction algebra for arbitrary G3(phi,X), "
            "arbitrary scalar gradient/Hessian/third jet, and arbitrary Ricci contraction; "
            "the compact Euler coefficient, explicitly second-order reduction, Hilbert stress "
            "tensor, and off-shell Noether identity are checked with two independent negative "
            "controls"
        ),
    }


@lru_cache(maxsize=1)
def generic_g4_phi_variation_noether_control() -> tuple[bool, dict[str, Any]]:
    """Prove variation and Noether closure for the arbitrary F(phi) R subfamily."""

    f_phi, curvature = sp.symbols("F_phi R", real=True)
    p_components = sp.symbols("p_0:4", real=True)
    ricci_p_components = sp.symbols("Ricci_p_0:4", real=True)

    # For H_mn=(1/sqrt(-g)) delta S/delta g^(mn), contracted Bianchi and
    # [box,nabla_nu]F=R_nu^rho nabla_rho F give
    # nabla^mu H_mu_nu=-F_phi R p_nu/2.  The scalar Euler coefficient is
    # E_phi=F_phi R, so diffeomorphism invariance requires 2 div(H)+E p=0.
    metric_divergence = [
        sp.factor(-f_phi * curvature * component / 2)
        for component in p_components
    ]
    scalar_euler = f_phi * curvature
    noether_residuals = [
        sp.factor(2 * divergence + scalar_euler * component)
        for divergence, component in zip(
            metric_divergence, p_components, strict=True
        )
    ]

    # Omitting (g_mn box-nabla_mn)F leaves div(F G_mn).  After combining
    # with the scalar Euler term, the residual is 2 F_phi R_nu_rho p^rho.
    omitted_completion_residuals = [
        sp.factor(2 * f_phi * component) for component in ricci_p_components
    ]
    wrong_scalar_sign_residuals = [
        sp.factor(-2 * f_phi * curvature * component)
        for component in p_components
    ]

    x, theta = sp.symbols("X theta", real=True)
    f_phiphi = sp.Symbol("F_phiphi", real=True)
    box_f = sp.factor(f_phi * theta - 2 * x * f_phiphi)
    constant_limit = {
        f_phi: 0,
        f_phiphi: 0,
    }
    constant_box_residual = sp.factor(box_f.subs(constant_limit))
    passed = (
        all(item == 0 for item in noether_residuals)
        and all(item != 0 for item in omitted_completion_residuals)
        and all(item != 0 for item in wrong_scalar_sign_residuals)
        and constant_box_residual == 0
    )
    return passed, {
        "status": "pass" if passed else "fail",
        "lagrangian": "L4_phi=F(phi) R",
        "metric_euler_tensor": (
            "H_mu_nu=F G_mu_nu+(g_mu_nu box-nabla_mu nabla_nu)F"
        ),
        "scalar_euler_coefficient": "E_phi=F_phi R",
        "function_chain_rules": {
            "nabla_mu_F": "F_phi p_mu",
            "nabla_mu_nabla_nu_F": (
                "F_phi H_mu_nu+F_phiphi p_mu p_nu"
            ),
            "box_F": str(box_f),
        },
        "identity": "2 nabla^mu H_mu_nu+E_phi p_nu=0",
        "source": "Kobayashi-Yamaguchi-Yokoyama 2011, G4=F(phi) reduction of B.4/B.8/B.12",
        "source_url": "https://arxiv.org/abs/1105.5723",
        "identity_proof_chain": [
            "nabla^mu(F G_mu_nu)=F_phi p^mu G_mu_nu",
            "nabla_nu box(F)-box nabla_nu(F)=-R_nu_rho nabla^rho(F)",
            "G_mu_nu p^mu-R_nu_rho p^rho=-R p_nu/2",
        ],
        "noether_residuals": [str(item) for item in noether_residuals],
        "omitted_metric_completion_residuals": [
            str(item) for item in omitted_completion_residuals
        ],
        "omitted_metric_completion_rejected": all(
            item != 0 for item in omitted_completion_residuals
        ),
        "wrong_scalar_sign_residuals": [
            str(item) for item in wrong_scalar_sign_residuals
        ],
        "wrong_scalar_sign_rejected": all(
            item != 0 for item in wrong_scalar_sign_residuals
        ),
        "constant_F_limit": {
            "metric_euler_tensor": "F G_mu_nu",
            "scalar_euler_coefficient": "0",
            "box_F_residual": str(constant_box_residual),
            "reduces_to_einstein_hilbert": True,
        },
        "scope": (
            "exact arbitrary-background covariant metric and scalar variation for every smooth "
            "X-independent G4=F(phi), with contracted-Bianchi and scalar-commutator Noether "
            "closure plus two independent negative controls; G4_X dependence is excluded"
        ),
    }


@lru_cache(maxsize=1)
def generic_g4_scalar_variation_control() -> tuple[bool, dict[str, Any]]:
    """Verify the arbitrary-G4 fixed-metric scalar current and its flat-jet order."""

    from .horndeski import (
        quartic_horndeski_boundary_and_flrw_noether_control,
        quartic_horndeski_scalar_euler_reduction_control,
    )

    signature = (-1, 1, 1, 1)
    p = sp.symbols("g4_p_0:4", real=True)
    h_symbols = sp.symbols(
        "g4_H_00 g4_H_01 g4_H_02 g4_H_03 g4_H_11 g4_H_12 "
        "g4_H_13 g4_H_22 g4_H_23 g4_H_33",
        real=True,
    )
    hessian = sp.zeros(4, 4)
    cursor = 0
    unique_hessian: list[tuple[sp.Symbol, tuple[int, int]]] = []
    for row in range(4):
        for column in range(row, 4):
            symbol = h_symbols[cursor]
            hessian[row, column] = symbol
            hessian[column, row] = symbol
            unique_hessian.append((symbol, (row, column)))
            cursor += 1
    third_symbols: dict[tuple[int, int, int], sp.Symbol] = {}

    def third_jet(first: int, second: int, third: int) -> sp.Symbol:
        key = tuple(sorted((first, second, third)))
        if key not in third_symbols:
            third_symbols[key] = sp.Symbol(
                "g4_J_" + "".join(map(str, key)), real=True
            )
        return third_symbols[key]

    theta = sum(signature[index] * hessian[index, index] for index in range(4))
    hessian_squared = sum(
        signature[first]
        * signature[second]
        * hessian[first, second] ** 2
        for first in range(4)
        for second in range(4)
    )
    q = [
        -sum(
            signature[index] * p[index] * hessian[mu, index]
            for index in range(4)
        )
        for mu in range(4)
    ]
    q_upper = [signature[index] * q[index] for index in range(4)]
    p_upper = [signature[index] * p[index] for index in range(4)]
    hessian_upper = [
        [
            signature[first] * signature[second] * hessian[first, second]
            for second in range(4)
        ]
        for first in range(4)
    ]
    hessian_difference = sp.expand(theta**2 - hessian_squared)

    g4, g4_phi, g4_x = sp.symbols("g4 g4_phi g4_x", real=True)
    g4_phiphi, g4_xx, g4_phix = sp.symbols(
        "g4_phiphi g4_xx g4_phix", real=True
    )
    g4_xxx, g4_phixx, g4_phiphix = sp.symbols(
        "g4_xxx g4_phixx g4_phiphix", real=True
    )
    g4_phiphiphi = sp.Symbol("g4_phiphiphi", real=True)
    current_upper = [
        sp.expand(
            -g4_xx * hessian_difference * p_upper[mu]
            - 2
            * g4_xx
            * (
                theta * q_upper[mu]
                - sum(q[nu] * hessian_upper[mu][nu] for nu in range(4))
            )
            - 2 * g4_phix * (theta * p_upper[mu] + q_upper[mu])
        )
        for mu in range(4)
    ]

    base_derivatives: list[tuple[sp.Symbol, list[sp.Expr]]] = []
    for index, symbol in enumerate(p):
        base_derivatives.append(
            (symbol, [hessian[mu, index] for mu in range(4)])
        )
    for symbol, (first, second) in unique_hessian:
        base_derivatives.append(
            (symbol, [third_jet(mu, first, second) for mu in range(4)])
        )
    base_derivatives.extend(
        [
            (
                g4,
                [g4_phi * p[mu] + g4_x * q[mu] for mu in range(4)],
            ),
            (
                g4_phi,
                [
                    g4_phiphi * p[mu] + g4_phix * q[mu]
                    for mu in range(4)
                ],
            ),
            (
                g4_x,
                [g4_phix * p[mu] + g4_xx * q[mu] for mu in range(4)],
            ),
            (
                g4_phiphi,
                [
                    g4_phiphiphi * p[mu] + g4_phiphix * q[mu]
                    for mu in range(4)
                ],
            ),
            (
                g4_xx,
                [
                    g4_phixx * p[mu] + g4_xxx * q[mu]
                    for mu in range(4)
                ],
            ),
            (
                g4_phix,
                [
                    g4_phiphix * p[mu] + g4_phixx * q[mu]
                    for mu in range(4)
                ],
            ),
        ]
    )

    def flat_derivative(mu: int, expression: sp.Expr) -> sp.Expr:
        return sp.expand(
            sum(
                sp.diff(expression, symbol) * derivatives[mu]
                for symbol, derivatives in base_derivatives
            )
        )

    current_divergence = sp.expand(
        sum(flat_derivative(mu, current_upper[mu]) for mu in range(4))
    )
    explicit_phi_derivative = g4_phix * hessian_difference
    flat_euler = sp.expand(explicit_phi_derivative - current_divergence)
    third_derivative_coefficients = {
        symbol.name: str(sp.factor(sp.diff(flat_euler, symbol)))
        for symbol in sorted(third_symbols.values(), key=lambda item: item.name)
    }
    surviving_third_derivatives = {
        name: coefficient
        for name, coefficient in third_derivative_coefficients.items()
        if coefficient != "0"
    }

    # Evaluate the full source-form metric Euler tensor on the same arbitrary
    # flat jet.  Although the background curvature vanishes at the point, the
    # tensor retains every nonlinear G4_X, G4_XX, and G4_phiX Hessian term.
    x_flat = sp.factor(
        -sum(signature[index] * p[index] ** 2 for index in range(4)) / 2
    )
    grad_g4_x = [g4_phix * p[index] + g4_xx * q[index] for index in range(4)]
    grad_g4_x_dot_p = sum(
        signature[index] * grad_g4_x[index] * p[index]
        for index in range(4)
    )
    hessian_pp = sum(
        signature[first]
        * signature[second]
        * hessian[first, second]
        * p[first]
        * p[second]
        for first in range(4)
        for second in range(4)
    )
    hessian_square_pp = sum(
        signature[first]
        * signature[second]
        * signature[contracted]
        * p[first]
        * p[second]
        * hessian[first, contracted]
        * hessian[second, contracted]
        for first in range(4)
        for second in range(4)
        for contracted in range(4)
    )

    def flat_metric_euler(first: int, second: int) -> sp.Expr:
        metric_entry = signature[first] if first == second else 0
        hessian_product = sum(
            signature[index]
            * hessian[index, first]
            * hessian[index, second]
            for index in range(4)
        )
        grad_braiding = sum(
            signature[index]
            * grad_g4_x[index]
            * (
                hessian[index, first] * p[second]
                + hessian[index, second] * p[first]
            )
            for index in range(4)
        )
        phi_braiding = sum(
            signature[index]
            * p[index]
            * (
                hessian[index, first] * p[second]
                + hessian[index, second] * p[first]
            )
            for index in range(4)
        )
        metric_scalar = (
            g4_phi * theta
            - 2 * x_flat * g4_phiphi
            - 2 * g4_phix * hessian_pp
            + g4_xx * hessian_square_pp
            + g4_x * hessian_difference / 2
            + grad_g4_x_dot_p * theta
        )
        return sp.expand(
            -g4_xx * hessian_difference * p[first] * p[second] / 2
            - g4_x * theta * hessian[first, second]
            + g4_x * hessian_product
            + grad_braiding
            - grad_g4_x_dot_p * hessian[first, second]
            + metric_entry * metric_scalar
            - theta
            * (
                grad_g4_x[first] * p[second]
                + grad_g4_x[second] * p[first]
            )
            - g4_phi * hessian[first, second]
            - g4_phiphi * p[first] * p[second]
            + g4_phix * phi_braiding
            - g4_xx * q[first] * q[second]
        )

    flat_metric = [
        [flat_metric_euler(first, second) for second in range(4)]
        for first in range(4)
    ]
    flat_metric_symmetry_residuals = [
        sp.factor(flat_metric[first][second] - flat_metric[second][first])
        for first in range(4)
        for second in range(first + 1, 4)
    ]
    flat_metric_divergence = [
        sp.expand(
            sum(
                signature[first]
                * flat_derivative(first, flat_metric[first][second])
                for first in range(4)
            )
        )
        for second in range(4)
    ]
    flat_combined_noether_residuals = [
        sp.factor(
            2 * flat_metric_divergence[index] + flat_euler * p[index]
        )
        for index in range(4)
    ]
    omitted_q_outer_product_residuals = [
        sp.factor(
            2
            * sum(
                signature[first]
                * flat_derivative(
                    first, g4_xx * q[first] * q[second]
                )
                for first in range(4)
            )
        )
        for second in range(4)
    ]

    linear_x_current_curved = "J4^mu=2 G^(mu nu) p_nu"
    linear_x_euler_curved = "E4=-2 G^(mu nu) H_mu_nu"
    linear_x_control_passed, linear_x_control = (
        quartic_horndeski_scalar_euler_reduction_control()
    )
    boundary_passed, boundary_control = (
        quartic_horndeski_boundary_and_flrw_noether_control()
    )
    phi_only_passed, phi_only_control = generic_g4_phi_variation_noether_control()
    passed = (
        not surviving_third_derivatives
        and all(item == 0 for item in flat_metric_symmetry_residuals)
        and all(item == 0 for item in flat_combined_noether_residuals)
        and all(item != 0 for item in omitted_q_outer_product_residuals)
        and linear_x_control_passed
        and linear_x_control["reduction_residual"] == "0"
        and boundary_passed
        and boundary_control["boundary_equivalence"]["boundary_residual"] == "0"
        and phi_only_passed
    )
    return passed, {
        "status": "pass" if passed else "fail",
        "lagrangian": (
            "L4=G4(phi,X) R+G4_X[(box(phi))^2-H_mu_nu H^(mu nu)]"
        ),
        "fixed_metric_scalar_variation": {
            "equation": "E4=P4_phi-nabla_mu(J4^mu)",
            "P4_phi": "G4_phi R+G4_phix[(box(phi))^2-H_mu_nu H^(mu nu)]",
            "J4_mu": (
                "-L4_X p_mu+2 G4_X R_mu_nu p^nu"
                "-2 G4_XX[box(phi) q_mu-q_nu H_mu^nu]"
                "-2 G4_phix[box(phi) p_mu+q_mu]"
            ),
            "source": "Kobayashi-Yamaguchi-Yokoyama 2011, equations B.8 and B.12",
            "source_url": "https://arxiv.org/abs/1105.5723",
        },
        "flat_arbitrary_third_jet": {
            "dimension": 4,
            "independent_gradient_components": 4,
            "independent_hessian_components": 10,
            "independent_symmetric_third_derivatives": len(third_symbols),
            "third_derivative_coefficients": third_derivative_coefficients,
            "surviving_third_derivatives": surviving_third_derivatives,
            "second_order": not surviving_third_derivatives,
            "second_order_expression_operation_count": int(sp.count_ops(flat_euler)),
        },
        "flat_arbitrary_metric_scalar_noether": {
            "metric_euler_source": (
                "Kobayashi-Yamaguchi-Yokoyama 2011 equation B.4 evaluated at zero "
                "background curvature with arbitrary scalar jet"
            ),
            "metric_symmetry_residuals": [
                str(item) for item in flat_metric_symmetry_residuals
            ],
            "combined_identity": "2 partial^mu H_mu_nu+E4 p_nu=0",
            "combined_noether_residuals": [
                str(item) for item in flat_combined_noether_residuals
            ],
            "omitted_G4_XX_q_mu_q_nu_residuals": [
                str(item) for item in omitted_q_outer_product_residuals
            ],
            "omitted_G4_XX_q_mu_q_nu_rejected": all(
                item != 0 for item in omitted_q_outer_product_residuals
            ),
            "verified_function_derivatives": [
                "G4_phi",
                "G4_X",
                "G4_phiphi",
                "G4_phiX",
                "G4_XX",
                "G4_phiphiphi",
                "G4_phiphiX",
                "G4_phiXX",
                "G4_XXX",
            ],
        },
        "curved_linear_x_reduction": {
            "current": linear_x_current_curved,
            "euler": linear_x_euler_curved,
            "reduction_residual": linear_x_control["reduction_residual"],
            "fourth_derivative_coefficient": linear_x_control[
                "fourth_derivative_coefficient"
            ],
            "curvature_gradient_coefficient": linear_x_control[
                "curvature_gradient_coefficient"
            ],
            "wrong_completion_rejected": linear_x_control[
                "wrong_completion_negative_control"
            ]["higher_metric_derivative_restored"],
            "boundary_equivalence_residual": boundary_control[
                "boundary_equivalence"
            ]["boundary_residual"],
        },
        "phi_only_reduction": {
            "status": phi_only_control["status"],
            "noether_residuals": phi_only_control["noether_residuals"],
        },
        "generic_x_dependent_metric_variation": "pass_on_arbitrary_flat_local_jet",
        "generic_x_dependent_combined_noether_identity": (
            "pass_on_arbitrary_flat_local_jet_curved_completion_unresolved"
        ),
        "scope": (
            "exact source-form scalar current for arbitrary smooth G4(phi,X), automatic "
            "third-derivative cancellation on an arbitrary four-dimensional flat third jet, "
            "exact flat nonlinear-X metric/scalar Noether closure, and exact curved linear-X and "
            "phi-only reductions; the arbitrary curved nonlinear-X metric Euler tensor and "
            "combined Noether identity remain fail-closed"
        ),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def parse_dimensionless_expression(text: str, coefficient_names: set[str]) -> sp.Expr:
    normalized = text.strip().replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"invalid function expression: {error.msg}") from error
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Constant,
        ast.Call,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Load,
    )
    allowed_names = {"u", "x", *_SAFE_FUNCTIONS, *coefficient_names}
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise TypeError(f"unsupported function syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"undeclared scalar-tensor symbol: {node.id}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCTIONS:
                raise ValueError("only allowlisted dimensionless functions may be called")
            if node.keywords or len(node.args) != 1:
                raise ValueError("allowlisted functions require exactly one positional argument")
    symbols = {
        name: sp.Symbol(name, real=True) for name in sorted(coefficient_names | {"u", "x"})
    }
    return sp.sympify(normalized, locals={**symbols, **_SAFE_FUNCTIONS}, evaluate=True)


def _axis_cardinality(
    axes: Any, coefficient_names: set[str], errors: list[str]
) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(axes, list):
        errors.append("mutation_axes must be a list")
        return 0, []
    cardinality = 1
    compiled: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(axes):
        if not isinstance(raw, dict):
            errors.append(f"mutation_axes[{index}] must be an object")
            continue
        coefficient = str(raw.get("coefficient", ""))
        if coefficient not in coefficient_names:
            errors.append(f"mutation axis uses undeclared coefficient {coefficient!r}")
        if coefficient in seen:
            errors.append(f"duplicate mutation axis for {coefficient}")
        seen.add(coefficient)
        values = raw.get("values", [])
        if not isinstance(values, list) or not values:
            errors.append(f"mutation axis {coefficient} requires one or more values")
            continue
        parsed_values: list[str] = []
        for value in values:
            try:
                parsed = sp.sympify(str(value), evaluate=True)
            except (sp.SympifyError, TypeError, ValueError) as error:
                errors.append(f"invalid value for {coefficient}: {error}")
                continue
            if parsed.free_symbols or not parsed.is_real:
                errors.append(f"mutation value for {coefficient} must be an exact real constant")
                continue
            parsed_values.append(str(parsed))
        cardinality *= len(parsed_values)
        compiled.append({"coefficient": coefficient, "values": parsed_values})
    return cardinality, compiled


def compile_scalar_tensor_pack(spec: dict[str, Any]) -> dict[str, Any]:
    """Compile a normalized Horndeski L2-L4 function family without enumerating coefficients."""

    errors: list[str] = []
    if spec.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    normalization = spec.get("normalization", {})
    expected_normalization = {
        "u": "phi/Lambda_phi",
        "x": "-nabla_phi_squared/(2*Lambda_phi**4)",
        "Lambda_phi_positive": True,
    }
    if normalization != expected_normalization:
        errors.append("normalization must use the canonical dimensionless u and x definitions")

    raw_coefficients = spec.get("coefficients", [])
    if not isinstance(raw_coefficients, list) or any(
        not isinstance(item, str) or not item.isidentifier() for item in raw_coefficients
    ):
        errors.append("coefficients must be a list of valid identifiers")
        raw_coefficients = []
    coefficient_names = set(raw_coefficients)
    if len(coefficient_names) != len(raw_coefficients):
        errors.append("coefficients must be unique")
    if coefficient_names & {"u", "x", *_SAFE_FUNCTIONS}:
        errors.append("coefficient names collide with reserved language symbols")

    raw_functions = spec.get("functions", {})
    if not isinstance(raw_functions, dict):
        errors.append("functions must be an object")
        raw_functions = {}
    unknown_functions = sorted(set(raw_functions) - set(_FUNCTION_NAMES))
    if unknown_functions:
        errors.append("unknown scalar-tensor functions: " + ", ".join(unknown_functions))

    expressions: dict[str, sp.Expr] = {}
    for name in _FUNCTION_NAMES:
        raw = raw_functions.get(name)
        if not isinstance(raw, str):
            errors.append(f"functions.{name} must be a string")
            continue
        try:
            expressions[name] = sp.factor(
                parse_dimensionless_expression(raw, coefficient_names)
            )
        except (TypeError, ValueError, sp.SympifyError) as error:
            errors.append(f"functions.{name}: {error}")

    u = sp.Symbol("u", real=True)
    x = sp.Symbol("x", real=True)
    derivatives: dict[str, sp.Expr] = {}
    for name, expression in expressions.items():
        derivatives[f"{name}_u"] = sp.factor(sp.diff(expression, u))
        derivatives[f"{name}_x"] = sp.factor(sp.diff(expression, x))
        derivatives[f"{name}_xx"] = sp.factor(sp.diff(expression, x, 2))

    overrides = spec.get("derivative_overrides", {})
    if not isinstance(overrides, dict):
        errors.append("derivative_overrides must be an object")
        overrides = {}
    override_residuals: dict[str, str] = {}
    for name, raw in sorted(overrides.items()):
        if name not in derivatives:
            errors.append(f"unknown derivative override: {name}")
            continue
        if not isinstance(raw, str):
            errors.append(f"derivative override {name} must be a string")
            continue
        try:
            proposed = parse_dimensionless_expression(raw, coefficient_names)
        except (TypeError, ValueError, sp.SympifyError) as error:
            errors.append(f"derivative override {name}: {error}")
            continue
        residual = sp.factor(proposed - derivatives[name])
        override_residuals[name] = str(residual)
        if residual != 0:
            errors.append(f"derivative override {name} is inconsistent with its parent function")

    cardinality, axes = _axis_cardinality(
        spec.get("mutation_axes", []), coefficient_names, errors
    )
    expression_strings = {name: str(value) for name, value in sorted(expressions.items())}
    derivative_strings = {name: str(value) for name, value in sorted(derivatives.items())}
    g4_completion = {
        "curvature_coefficient": expression_strings.get("g4"),
        "hessian_difference_coefficient": derivative_strings.get("g4_x"),
        "independent_choice_forbidden": True,
        "normalized_density": (
            "g4(u,x) R/Lambda_phi^2 + d(g4)/dx "
            "[(box(phi))^2-phi_(mu nu)phi^(mu nu)]/Lambda_phi^6"
        ),
    }
    g2_control_passed, g2_control = generic_g2_variation_noether_control()
    g3_control_passed, g3_control = generic_g3_variation_noether_control()
    g4_phi_control_passed, g4_phi_control = generic_g4_phi_variation_noether_control()
    g4_scalar_control_passed, g4_scalar_control = generic_g4_scalar_variation_control()
    g4_curved_witness_passed, g4_curved_witness = (
        generic_g4_curved_rnc_witness_control()
    )
    g4_curved_symbolic_passed, g4_curved_symbolic = (
        generic_g4_curved_symbolic_rnc_control()
    )
    generic_adm_passed = generic_horndeski_l2_l4_unitary_adm_control()
    generic_dirac_passed, generic_dirac_control = (
        generic_horndeski_l2_l4_unitary_dirac_control()
    )
    generic_tensor_passed, generic_tensor_control = (
        generic_horndeski_l2_l4_tensor_stability_control()
    )
    generic_scalar_passed, generic_scalar_control = (
        generic_horndeski_l2_l4_flrw_scalar_reduction_control()
    )
    generic_kessence_passed, generic_kessence_control = (
        generic_kessence_timelike_principal_hamiltonian_control()
    )
    generic_kessence_nonlinear_passed, generic_kessence_nonlinear_control = (
        generic_kessence_nonlinear_adm_legendre_control()
    )
    generic_cubic_bssn_passed, generic_cubic_bssn_control = (
        generic_cubic_horndeski_bssn_hyperbolicity_control()
    )
    g4_is_phi_only = derivatives.get("g4_x") == 0
    compiled_adm_regularity_factor = sp.factor(
        expressions.get("g4", sp.Integer(0))
        - 2 * x * derivatives.get("g4_x", sp.Integer(0))
    )
    compiled_g2_lapse_hessian_factor = sp.factor(
        2
        * x
        * (
            derivatives.get("g2_x", sp.Integer(0))
            + 2 * x * derivatives.get("g2_xx", sp.Integer(0))
        )
    )
    compiled_tensor_g_t = sp.factor(2 * compiled_adm_regularity_factor)
    compiled_tensor_f_t = sp.factor(2 * expressions.get("g4", sp.Integer(0)))
    compiled_tensor_speed_squared = sp.factor(
        compiled_tensor_f_t / compiled_tensor_g_t
    )
    h, h_tau, x_tau = sp.symbols("h h_tau x_tau", real=True)
    scalar_velocity = sp.sqrt(2 * x)
    g2 = expressions.get("g2", sp.Integer(0))
    g3 = expressions.get("g3", sp.Integer(0))
    g4 = expressions.get("g4", sp.Integer(0))
    g2_x = sp.diff(g2, x)
    g2_xx = sp.diff(g2, x, 2)
    g3_x = sp.diff(g3, x)
    g3_xx = sp.diff(g3, x, 2)
    g3_u = sp.diff(g3, u)
    g3_ux = sp.diff(g3, u, x)
    g4_x = sp.diff(g4, x)
    g4_xx = sp.diff(g4, x, 2)
    g4_xxx = sp.diff(g4, x, 3)
    g4_u = sp.diff(g4, u)
    g4_ux = sp.diff(g4, u, x)
    g4_uxx = sp.diff(g4, u, x, 2)
    g4_uu = sp.diff(g4, u, 2)
    g3_phi_only = sp.factor(g3.subs(x, 0))
    canonical_g2 = sp.factor(g2 - 2 * x * sp.diff(g3_phi_only, u))
    canonical_g3 = sp.factor(g3 - g3_phi_only)
    generalized_harmonic_obstructions = {
        "canonical_G3": canonical_g3,
        "G4_X": g4_x,
    }
    generalized_harmonic_symbolically_eligible = all(
        residual == 0 for residual in generalized_harmonic_obstructions.values()
    )
    canonical_g2_x = sp.factor(sp.diff(canonical_g2, x))
    canonical_g2_xx = sp.factor(sp.diff(canonical_g2, x, 2))
    compiled_kessence_kinetic = sp.factor(
        canonical_g2_x + 2 * x * canonical_g2_xx
    )
    compiled_kessence_gradient = canonical_g2_x
    compiled_kessence_homogeneous_energy_density = sp.factor(
        2 * x * canonical_g2_x - canonical_g2
    )
    compiled_kessence_homogeneous_legendre_jacobian = compiled_kessence_kinetic
    compiled_kessence_speed_squared = sp.factor(
        compiled_kessence_gradient / compiled_kessence_kinetic
    )
    formulation_partition: dict[str, Any] = {
        "status": "not_counted_cardinality_above_limit",
        "counting_limit": 10000,
        "generalized_harmonic_eligible": None,
        "modified_harmonic_required": None,
    }
    if cardinality <= formulation_partition["counting_limit"]:
        eligible_count = 0
        modified_count = 0
        eligible_examples: list[dict[str, str]] = []
        obstruction_class_counts = {
            "generalized_harmonic_kessence": 0,
            "modified_harmonic_G3_only": 0,
            "modified_harmonic_G4_X_only": 0,
            "modified_harmonic_G3_and_G4_X": 0,
        }
        proof_subclass_counts = {
            "generalized_harmonic_kessence": 0,
            "cubic_G3_only": 0,
            "quartic_linear_X_G4_only_G2_linear_X": 0,
            "quartic_linear_X_G4_only_G2_nonlinear_X": 0,
            "quartic_nonlinear_X_G4_only": 0,
            "mixed_G3_linear_X_G4": 0,
            "mixed_G3_nonlinear_X_G4": 0,
        }
        assignment_classifications: list[dict[str, Any]] = []
        value_lists = [axis["values"] for axis in axes]
        for values in product(*value_lists):
            assignment = {
                axis["coefficient"]: value
                for axis, value in zip(axes, values, strict=True)
            }
            substitutions = {
                sp.Symbol(name, real=True): sp.sympify(value)
                for name, value in assignment.items()
            }
            residuals = {
                name: sp.factor(residual.subs(substitutions))
                for name, residual in generalized_harmonic_obstructions.items()
            }
            active_obstructions = [
                name for name, residual in residuals.items() if residual != 0
            ]
            g4_xx_residual = sp.factor(g4_xx.subs(substitutions))
            g2_xx_residual = sp.factor(canonical_g2_xx.subs(substitutions))
            if not active_obstructions:
                obstruction_class = "generalized_harmonic_kessence"
                eligible_count += 1
                if len(eligible_examples) < 5:
                    eligible_examples.append(assignment)
            else:
                modified_count += 1
                obstruction_class = (
                    "modified_harmonic_G3_and_G4_X"
                    if len(active_obstructions) == 2
                    else "modified_harmonic_G3_only"
                    if active_obstructions[0] == "canonical_G3"
                    else "modified_harmonic_G4_X_only"
                )
            obstruction_class_counts[obstruction_class] += 1
            proof_subclass = (
                "generalized_harmonic_kessence"
                if obstruction_class == "generalized_harmonic_kessence"
                else "cubic_G3_only"
                if obstruction_class == "modified_harmonic_G3_only"
                else "quartic_linear_X_G4_only_G2_linear_X"
                if obstruction_class == "modified_harmonic_G4_X_only"
                and g4_xx_residual == 0
                and g2_xx_residual == 0
                else "quartic_linear_X_G4_only_G2_nonlinear_X"
                if obstruction_class == "modified_harmonic_G4_X_only"
                and g4_xx_residual == 0
                else "quartic_nonlinear_X_G4_only"
                if obstruction_class == "modified_harmonic_G4_X_only"
                else "mixed_G3_linear_X_G4"
                if g4_xx_residual == 0
                else "mixed_G3_nonlinear_X_G4"
            )
            proof_subclass_counts[proof_subclass] += 1
            proof_route = {
                "generalized_harmonic_kessence": "generalized_harmonic_kessence",
                "cubic_G3_only": "cubic_horndeski_bssn_or_ccz4_weak_field",
                "quartic_linear_X_G4_only_G2_linear_X": (
                    "linear_X_quartic_full_symbol_requires_phi_coefficients_fixed_zero"
                ),
                "quartic_linear_X_G4_only_G2_nonlinear_X": (
                    "linear_X_quartic_plus_kessence_full_symbol_requires_phi_coefficients_fixed_zero"
                ),
                "quartic_nonlinear_X_G4_only": (
                    "general_horndeski_modified_harmonic_weak_coupling"
                ),
                "mixed_G3_linear_X_G4": (
                    "mixed_cubic_quartic_modified_harmonic_adapter_required"
                ),
                "mixed_G3_nonlinear_X_G4": (
                    "general_horndeski_modified_harmonic_weak_coupling"
                ),
            }[proof_subclass]
            assignment_classifications.append(
                {
                    "assignment": assignment,
                    "obstruction_class": obstruction_class,
                    "active_obstructions": active_obstructions,
                    "G2_XX_residual": str(g2_xx_residual),
                    "G4_XX_residual": str(g4_xx_residual),
                    "proof_subclass": proof_subclass,
                    "proof_route": proof_route,
                    "proof_route_fixed_coefficient_requirements": (
                        {"c11": "0", "c02": "0", "d01": "0", "a01": "0"}
                        if proof_subclass.startswith("quartic_linear_X_G4_only")
                        else {}
                    ),
                    "identity_residuals": {
                        name: str(residual) for name, residual in residuals.items()
                    },
                }
            )
        formulation_partition = {
            "status": "exact_axis_partition",
            "counting_limit": 10000,
            "generalized_harmonic_eligible": eligible_count,
            "modified_harmonic_required": modified_count,
            "eligible_examples": eligible_examples,
            "obstruction_class_counts": obstruction_class_counts,
            "proof_subclass_counts": proof_subclass_counts,
            "assignment_classifications": assignment_classifications,
            "count_residual": cardinality - eligible_count - modified_count,
        }
    scalar_acceleration = x_tau / scalar_velocity
    compiled_flrw_energy_constraint = sp.factor(
        2 * x * g2_x
        - g2
        + 6 * x * scalar_velocity * h * g3_x
        - 2 * x * g3_u
        - 6 * h**2 * g4
        + 24 * h**2 * x * (g4_x + x * g4_xx)
        - 12 * h * x * scalar_velocity * g4_ux
        - 6 * h * scalar_velocity * g4_u
    )
    compiled_flrw_pressure_equation = sp.factor(
        g2
        - 2 * x * (g3_u + scalar_acceleration * g3_x)
        + 2 * (3 * h**2 + 2 * h_tau) * g4
        - 12 * h**2 * x * g4_x
        - 4 * h * x_tau * g4_x
        - 8 * h * x * x_tau * g4_xx
        + 2 * (scalar_acceleration + 2 * h * scalar_velocity) * g4_u
        + 4 * x * g4_uu
        + 4
        * x
        * (scalar_acceleration - 2 * h * scalar_velocity)
        * g4_ux
    )
    compiled_flrw_constraint_flow = sp.factor(
        sp.diff(compiled_flrw_energy_constraint, u) * scalar_velocity
        + sp.diff(compiled_flrw_energy_constraint, x) * x_tau
        + sp.diff(compiled_flrw_energy_constraint, h) * h_tau
    )
    compiled_flrw_evolution_matrix = sp.Matrix(
        [
            [
                sp.diff(compiled_flrw_constraint_flow, h_tau),
                sp.diff(compiled_flrw_constraint_flow, x_tau),
            ],
            [
                sp.diff(compiled_flrw_pressure_equation, h_tau),
                sp.diff(compiled_flrw_pressure_equation, x_tau),
            ],
        ]
    ).applyfunc(sp.factor)
    compiled_flrw_evolution_source = sp.Matrix(
        [
            -compiled_flrw_constraint_flow.subs({h_tau: 0, x_tau: 0}),
            -compiled_flrw_pressure_equation.subs({h_tau: 0, x_tau: 0}),
        ]
    ).applyfunc(sp.factor)
    compiled_flrw_evolution_residual = sp.simplify(
        compiled_flrw_evolution_matrix * sp.Matrix([h_tau, x_tau])
        - compiled_flrw_evolution_source
        - sp.Matrix(
            [compiled_flrw_constraint_flow, compiled_flrw_pressure_equation]
        )
    )
    compiled_flrw_evolution_determinant = sp.factor(
        compiled_flrw_evolution_matrix.det()
    )
    compiled_scalar_sigma = sp.factor(
        x * g2_x
        + 2 * x**2 * g2_xx
        + 12 * h * scalar_velocity * x * g3_x
        + 6 * h * scalar_velocity * x**2 * g3_xx
        - 2 * x * g3_u
        - 2 * x**2 * g3_ux
        - 6 * h**2 * g4
        + 6
        * (
            h**2
            * (7 * x * g4_x + 16 * x**2 * g4_xx + 4 * x**3 * g4_xxx)
            - h
            * scalar_velocity
            * (g4_u + 5 * x * g4_ux + 2 * x**2 * g4_uxx)
        )
    )
    compiled_scalar_theta = sp.factor(
        -scalar_velocity * x * g3_x
        + 2 * h * g4
        - 8 * h * x * g4_x
        - 8 * h * x**2 * g4_xx
        + scalar_velocity * g4_u
        + 2 * x * scalar_velocity * g4_ux
    )
    compiled_tensor_ratio = sp.factor(
        compiled_tensor_g_t**2 / compiled_scalar_theta
    )
    compiled_tensor_ratio_tau_derivative = sp.factor(
        sp.diff(compiled_tensor_ratio, u) * scalar_velocity
        + sp.diff(compiled_tensor_ratio, x) * x_tau
        + sp.diff(compiled_tensor_ratio, h) * h_tau
    )
    compiled_scalar_f_s = sp.factor(
        h * compiled_tensor_ratio
        + compiled_tensor_ratio_tau_derivative
        - compiled_tensor_f_t
    )
    compiled_scalar_g_s = sp.factor(
        compiled_scalar_sigma
        * compiled_tensor_g_t**2
        / compiled_scalar_theta**2
        + 3 * compiled_tensor_g_t
    )
    compiled_scalar_speed_squared = sp.factor(
        compiled_scalar_f_s / compiled_scalar_g_s
    )
    body = {
        "schema_version": "sigma-scalar-tensor-pack-ir-1.0",
        "source_schema_version": SCHEMA_VERSION,
        "status": "reject" if errors else "compiled_formal_adapters_unresolved",
        "errors": errors,
        "normalization": expected_normalization,
        "dimension_contract": {
            "G2": "Lambda_phi^4*g2(u,x)",
            "G3": "Lambda_phi*g3(u,x)",
            "G4": "Lambda_phi^2*g4(u,x)",
            "all_g_functions_dimensionless": True,
        },
        "functions": expression_strings,
        "derived_function_derivatives": derivative_strings,
        "derivative_override_residuals": override_residuals,
        "l4_differential_completion": g4_completion,
        "normalized_action": (
            "sqrt(-g) Lambda_phi^4 {g2 - g3 box(phi)/Lambda_phi^3 + "
            "g4 R/Lambda_phi^2 + g4_x[(box(phi))^2-phi_(mu nu)phi^(mu nu)]/"
            "Lambda_phi^6}"
        ),
        "mutation_space": {
            "axes": axes,
            "declared_cardinality": cardinality,
            "log10_cardinality": (
                None if cardinality <= 0 else 0.0 if cardinality == 1 else math.log10(cardinality)
            ),
            "enumerated": False,
        },
        "capability_status": {
            "typed_normalized_covariant_family": "pass" if not errors else "reject",
            "l4_parent_derivative_binding": "pass" if not errors else "reject",
            "generic_g2_variation_and_noether": (
                "pass" if not errors and g2_control_passed else "reject"
            ),
            "generic_g3_variation_and_noether": (
                "pass" if not errors and g3_control_passed else "reject"
            ),
            "generic_g4_phi_only_adapter": (
                "available" if g4_phi_control_passed else "reject"
            ),
            "generic_g4_fixed_metric_scalar_variation": (
                "pass" if not errors and g4_scalar_control_passed else "reject"
            ),
            "generic_g4_flat_metric_noether": (
                "pass" if not errors and g4_scalar_control_passed else "reject"
            ),
            "generic_g4_curved_exact_witnesses": (
                "pass" if not errors and g4_curved_witness_passed else "reject"
            ),
            "generic_g4_curved_all_jet_theorem": (
                "pass" if not errors and g4_curved_symbolic_passed else "reject"
            ),
            "generic_g4_independent_backend_metric_variation": "unresolved",
            "compiled_g4_phi_only_variation_and_noether": (
                "pass"
                if not errors and g4_is_phi_only and g4_phi_control_passed
                else "not_applicable_x_dependent"
                if not errors and not g4_is_phi_only
                else "reject"
            ),
            "generic_covariant_variation": (
                "pass"
                if not errors
                and g2_control_passed
                and g3_control_passed
                and g4_is_phi_only
                and g4_phi_control_passed
                else "unresolved"
                if not errors
                else "reject"
            ),
            "generic_noether_identity": (
                "pass"
                if not errors
                and g2_control_passed
                and g3_control_passed
                and g4_curved_symbolic_passed
                else "unresolved"
                if not errors
                else "reject"
            ),
            "generic_adm_kinetic_primary_constraint": (
                "pass" if not errors and generic_adm_passed["passed"] else "reject"
            ),
            "generic_adm_dirac": (
                "pass_on_regular_lapse_hessian_patches"
                if not errors and generic_adm_passed["passed"] and generic_dirac_passed
                else "reject"
            ),
            "generic_tensor_hamiltonian": (
                "pass_on_F_T_and_G_T_positive_patches"
                if not errors and generic_tensor_passed
                else "reject"
            ),
            "generic_tensor_principal_symbol": (
                "pass_on_F_T_and_G_T_positive_patches"
                if not errors and generic_tensor_passed
                else "reject"
            ),
            "generic_flrw_scalar_reduction": (
                "pass_with_background_sign_proof_required"
                if not errors and generic_scalar_passed
                else "reject"
            ),
            "generic_flrw_scalar_hamiltonian": (
                "pass_on_Theta_nonzero_F_S_and_G_S_positive_patches"
                if not errors and generic_scalar_passed
                else "reject"
            ),
            "generic_flrw_scalar_principal_symbol": (
                "pass_on_Theta_nonzero_F_S_and_G_S_positive_patches"
                if not errors and generic_scalar_passed
                else "reject"
            ),
            "generic_kessence_effective_metric_and_hamiltonian": (
                "pass" if not errors and generic_kessence_passed else "reject"
            ),
            "generic_kessence_nonlinear_adm_legendre": (
                "pass_with_candidate_convexity_and_energy_inequalities_required"
                if not errors and generic_kessence_nonlinear_passed
                else "reject"
            ),
            "generic_cubic_horndeski_bssn_hyperbolicity": (
                "conditional_requires_candidate_uniform_weak_field_and_cone_bounds"
                if not errors and generic_cubic_bssn_passed
                else "reject"
            ),
            "generic_weak_field_generalized_harmonic": (
                "structurally_eligible_kessence_subclass"
                if not errors and generalized_harmonic_symbolically_eligible
                else "partition_by_canonical_G3_and_G4_X_zero"
                if not errors
                else "reject"
            ),
            "generic_weak_field_modified_harmonic": (
                "conditional_requires_candidate_weak_coupling_cone_and_symmetrizer_bounds"
                if not errors
                else "reject"
            ),
            "generic_hamiltonian": "pass_on_flrw_healthy_patches_global_unresolved",
            "generic_principal_symbol": (
                "pass_on_flrw_healthy_patches_inhomogeneous_unresolved"
            ),
            "flrw_interval_background_certificate": "available_requires_run_config",
            "observations": "sealed",
        },
        "generic_g2_variation_noether_control": g2_control,
        "generic_g3_variation_noether_control": g3_control,
        "generic_g4_phi_variation_noether_control": g4_phi_control,
        "generic_g4_scalar_variation_control": g4_scalar_control,
        "generic_g4_curved_rnc_witness_control": g4_curved_witness,
        "generic_g4_curved_symbolic_rnc_control": g4_curved_symbolic,
        "generic_horndeski_l2_l4_unitary_adm_control": generic_adm_passed,
        "generic_horndeski_l2_l4_unitary_dirac_control": generic_dirac_control,
        "generic_horndeski_l2_l4_tensor_stability_control": generic_tensor_control,
        "generic_horndeski_l2_l4_flrw_scalar_reduction_control": (
            generic_scalar_control
        ),
        "generic_kessence_timelike_principal_hamiltonian_control": (
            generic_kessence_control
        ),
        "generic_kessence_nonlinear_adm_legendre_control": (
            generic_kessence_nonlinear_control
        ),
        "generic_cubic_horndeski_bssn_hyperbolicity_control": (
            generic_cubic_bssn_control
        ),
        "formulation_classification": {
            "source": {
                "title": "On the hyperbolicity of the most general Horndeski theory",
                "url": "https://arxiv.org/abs/1710.10155",
                "result": "generic weak-field generalized-harmonic eligibility requires canonical G3=0, G4_X=0, G5=0",
            },
            "G3_phi_only_absorbed_into_G2": str(g3_phi_only),
            "canonical_G2": str(canonical_g2),
            "canonical_G3": str(canonical_g3),
            "G4_X": str(g4_x),
            "generalized_harmonic_identity_conditions": [
                f"{canonical_g3} == 0",
                f"{g4_x} == 0",
            ],
            "whole_symbolic_family_eligible": generalized_harmonic_symbolically_eligible,
            "mutation_axis_partition": formulation_partition,
            "noneligible_route": (
                "canonical-G3-only candidates use the dedicated cubic-Horndeski BSSN/CCZ4 "
                "weak-field theorem; G4_X candidates use general modified harmonic. No route "
                "passes until its candidate-specific uniform weak-field/cone conditions hold"
            ),
        },
        "compiled_kessence_G2": str(canonical_g2),
        "compiled_kessence_kinetic": str(compiled_kessence_kinetic),
        "compiled_kessence_gradient": str(compiled_kessence_gradient),
        "compiled_kessence_homogeneous_energy_density": str(
            compiled_kessence_homogeneous_energy_density
        ),
        "compiled_kessence_homogeneous_legendre_jacobian": str(
            compiled_kessence_homogeneous_legendre_jacobian
        ),
        "compiled_kessence_speed_squared": str(compiled_kessence_speed_squared),
        "compiled_kessence_healthy_patch": (
            f"{compiled_kessence_gradient} > 0 and "
            f"{compiled_kessence_homogeneous_legendre_jacobian} > 0 and "
            f"{compiled_kessence_homogeneous_energy_density} >= 0"
        ),
        "compiled_adm_regularity_factor": str(compiled_adm_regularity_factor),
        "compiled_adm_regular_patch": (
            f"{compiled_adm_regularity_factor} != 0"
        ),
        "compiled_g2_unitary_lapse_hessian_factor": str(
            compiled_g2_lapse_hessian_factor
        ),
        "compiled_tensor_G_T": str(compiled_tensor_g_t),
        "compiled_tensor_F_T": str(compiled_tensor_f_t),
        "compiled_tensor_speed_squared": str(compiled_tensor_speed_squared),
        "compiled_tensor_healthy_patch": (
            f"{compiled_tensor_g_t} > 0 and {compiled_tensor_f_t} > 0"
        ),
        "compiled_flrw_background_variables": {
            "h": "H/Lambda_phi",
            "h_tau": "d(H/Lambda_phi)/d(Lambda_phi*t)",
            "x_tau": "dx/d(Lambda_phi*t)",
            "scalar_branch": "du/d(Lambda_phi*t)=sqrt(2*x), x>0",
            "on_shell_required": True,
        },
        "compiled_flrw_background_system": {
            "source_equations": ["arXiv:1105.5723v4 Eq. 3.1", "Eq. 3.6"],
            "energy_constraint_E": str(compiled_flrw_energy_constraint),
            "pressure_equation_P": str(compiled_flrw_pressure_equation),
            "constraint_flow_dE_d_tau": str(compiled_flrw_constraint_flow),
            "evolution_unknowns": ["h_tau", "x_tau"],
            "evolution_matrix": str(compiled_flrw_evolution_matrix),
            "evolution_source": str(compiled_flrw_evolution_source),
            "evolution_reconstruction_residual": str(
                compiled_flrw_evolution_residual
            ),
            "evolution_determinant": str(compiled_flrw_evolution_determinant),
            "regular_patch": f"{compiled_flrw_evolution_determinant} != 0 and x>0",
            "integration_contract": (
                "initialize on E=0, solve M*[h_tau,x_tau]^T=source, and reject any "
                "step leaving E=0 tolerance or crossing det(M)=0, x=0, Theta=0, "
                "G_T=0, F_T=0, G_S=0, or F_S=0"
            ),
        },
        "compiled_scalar_Sigma": str(compiled_scalar_sigma),
        "compiled_scalar_Theta": str(compiled_scalar_theta),
        "compiled_scalar_F_S": str(compiled_scalar_f_s),
        "compiled_scalar_G_S": str(compiled_scalar_g_s),
        "compiled_scalar_speed_squared": str(compiled_scalar_speed_squared),
        "compiled_flrw_healthy_patch": (
            f"{compiled_tensor_g_t} > 0 and {compiled_tensor_f_t} > 0 and "
            f"{compiled_scalar_theta} != 0 and {compiled_scalar_g_s} > 0 and "
            f"{compiled_scalar_f_s} > 0 on an on-shell declared FLRW background"
        ),
        "compiled_dirac_regular_patch": (
            f"{compiled_adm_regularity_factor} != 0 and the complete action-specific "
            "Delta_N operator is invertible under the declared boundary conditions"
        ),
        "scope": (
            "normalized Horndeski L2-L4 function-family front end; this compiler proves syntax, "
            "dimensions, the G4/G4_X differential completion, generic G2/G3 covariant "
            "variation/Noether identities, and the generic unitary-gauge ADM primary "
            "degeneracy, conditional regular-patch distributed Dirac closure, and the exact "
            "arbitrary-function FLRW tensor and constraint-reduced scalar principal/Hamiltonian "
            "blocks. Each candidate still requires an on-shell background sign proof for Theta, "
            "G_S, and F_S. Global lapse-operator invertibility, arbitrary inhomogeneous "
            "backgrounds, singular branches, and nonlinear global energy remain separate"
        ),
    }
    canonical = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(canonical.encode()).hexdigest()}
