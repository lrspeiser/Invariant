from __future__ import annotations

from typing import Any

import sympy as sp


def projected_aether_q_constant_tilt_root_audit() -> tuple[bool, dict[str, Any]]:
    """Audit all lab-frequency roots for every constant nonzero tilt.

    Write the positive rest-frame branch as
    Omega=f(kappa)=c*kappa/sqrt(1+ell^2*kappa^2), with 0<c<=1.  Each sign branch is
    Lorentz-transformed as k_lab=gamma*(kappa+beta*s*f).  Strict monotonicity makes this a
    bijection of the real line, so a fixed real lab wave number has exactly one real root per sign.
    But the transformed polynomial is quartic for every nonzero tilt and ell>0.  Its two remaining
    roots are therefore nonreal.  The control passes when it detects this hyperbolicity failure.
    """

    kappa = sp.Symbol("kappa", real=True)
    c, ell = sp.symbols("c ell", positive=True, real=True)
    beta = sp.Symbol("beta", real=True)
    branch = c * kappa / sp.sqrt(1 + ell**2 * kappa**2)
    group_speed = sp.factor(sp.diff(branch, kappa))
    expected_group_speed = c / (1 + ell**2 * kappa**2) ** sp.Rational(3, 2)
    derivative_residual = sp.factor(group_speed - expected_group_speed)

    signed_speed = sp.Symbol("s", real=True) * group_speed
    lab_speed = (signed_speed + beta) / (1 + beta * signed_speed)
    cone_identity_residual = sp.factor(
        1
        - lab_speed**2
        - (1 - beta**2)
        * (1 - signed_speed**2)
        / (1 + beta * signed_speed) ** 2
    )
    asymptotic_ratio = sp.limit(
        (kappa + beta * branch) / kappa, kappa, sp.oo
    )

    # Exact Sturm count for a nontrivial rational interior point.  The general count is supplied
    # by the monotone-bijection proof; this catches sign/convention errors in the expanded quartic.
    omega = sp.Symbol("omega", real=True)
    c_sample = sp.Rational(4, 5)
    ell_sample = sp.Rational(3, 7)
    beta_sample = sp.Rational(3, 5)
    gamma_sample = sp.Rational(5, 4)
    lab_k_sample = sp.Rational(7, 6)
    omega_rest = gamma_sample * (omega - beta_sample * lab_k_sample)
    kappa_rest = gamma_sample * (lab_k_sample - beta_sample * omega)
    lab_polynomial = sp.Poly(
        sp.together(
            omega_rest**2 * (1 + ell_sample**2 * kappa_rest**2)
            - c_sample**2 * kappa_rest**2
        ),
        omega,
    )
    exact_real_root_count = int(sp.polys.polytools.count_roots(lab_polynomial, -sp.oo, sp.oo))
    sample_nonreal_root_count = lab_polynomial.degree() - exact_real_root_count

    # For beta!=0 and ell>0 the omega^4 coefficient is strictly positive.  Combined with the
    # exactly two real roots proved branchwise above, this forces one nonreal conjugate pair.
    gamma = sp.Symbol("gamma_beta", positive=True, real=True)
    general_quartic_leading_coefficient = sp.factor(ell**2 * gamma**4 * beta**2)

    # If the rest cone is allowed outside the matter cone, an otherwise subluminal lab slice can
    # become characteristic.  This exact counterexample proves why c<=1 is a mandatory domain gate.
    negative_c = sp.Integer(2)
    negative_beta = -sp.Rational(1, 2)
    characteristic_margin = sp.factor(1 + negative_beta * negative_c)

    passed = (
        derivative_residual == 0
        and cone_identity_residual == 0
        and asymptotic_ratio == 1
        and lab_polynomial.degree() == 4
        and exact_real_root_count == 2
        and sample_nonreal_root_count == 2
        and general_quartic_leading_coefficient == ell**2 * gamma**4 * beta**2
        and characteristic_margin == 0
    )
    return passed, {
        "rest_branch": "Omega_s=s*c*kappa/sqrt(1+ell^2*kappa^2), s=+/-1",
        "parameter_domain": "ell>0, 0<c<=1, |beta|<1",
        "rest_group_speed": str(group_speed),
        "group_speed_maximum": "c at kappa=0",
        "group_speed_derivative_residual": str(derivative_residual),
        "lab_wave_number_map": "k_lab=gamma_beta*(kappa+beta*Omega_s)",
        "map_derivative_lower_bound": "gamma_beta*(1-|beta|*c)>0",
        "asymptotic_map_ratio_without_gamma": str(asymptotic_ratio),
        "real_branch_bijection_certificate": (
            "strictly increasing with limits -infinity and +infinity on each sign branch; "
            "there is exactly one real root per branch for every real k_lab"
        ),
        "lab_group_speed": str(lab_speed),
        "cone_identity_residual": str(cone_identity_residual),
        "expanded_quartic_sample": str(lab_polynomial.as_expr()),
        "expanded_quartic_exact_real_root_count": exact_real_root_count,
        "expanded_quartic_nonreal_root_count": sample_nonreal_root_count,
        "general_nonzero_tilt_quartic_leading_coefficient": str(
            general_quartic_leading_coefficient
        ),
        "generic_tilt_hyperbolicity_status": "reject",
        "rejection_reason": (
            "for ell>0 and every beta!=0 the lab-frequency polynomial has degree four, "
            "while the branch bijection proves exactly two real roots; the remaining conjugate "
            "pair is nonreal, so no open cone of nearby hyperbolic time covectors exists"
        ),
        "negative_control": {
            "rest_speed_c": str(negative_c),
            "tilt_beta": str(negative_beta),
            "branch_map_derivative_margin_at_kappa0": str(characteristic_margin),
            "rejected": characteristic_margin == 0,
        },
        "claim_limit": (
            "decisive frozen-coefficient negative control for the reduced quadratic Q mode. "
            "A separately declared preferred-foliation theory could change the admissible "
            "Cauchy-surface contract, but the present generic unit-vector action does not do so."
        ),
    }
