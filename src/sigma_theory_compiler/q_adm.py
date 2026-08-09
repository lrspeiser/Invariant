from __future__ import annotations

from typing import Any

import sympy as sp


def projected_aether_q_3plus1_control() -> tuple[bool, dict[str, Any]]:
    """Exact rational control of the generic-tilt projector block used by ADM IR.

    The control is deliberately kinematic.  It proves the 3+1 tensor contraction and exposes
    its normal-normal blocks; it does not prove higher-derivative constraint degeneracy.
    """

    spatial_a = sp.Matrix([sp.Rational(3, 4), 0, 0])
    chi = sp.Rational(5, 4)
    u = sp.Matrix([chi, *spatial_a])
    metric = sp.diag(-1, 1, 1, 1)
    projector = metric + u * u.T
    expected = sp.zeros(4)
    expected[0, 0] = (spatial_a.T * spatial_a)[0]
    for index in range(3):
        expected[0, index + 1] = chi * spatial_a[index]
        expected[index + 1, 0] = chi * spatial_a[index]
    expected[1:, 1:] = sp.eye(3) + spatial_a * spatial_a.T

    b_tensor = sp.Matrix(
        [
            [sp.Rational(1, 2), -1, sp.Rational(2, 3), sp.Rational(3, 7)],
            [sp.Rational(4, 5), sp.Rational(1, 7), -sp.Rational(2, 9), 1],
            [-sp.Rational(3, 8), sp.Rational(5, 6), sp.Rational(2, 11), -sp.Rational(1, 4)],
            [sp.Rational(7, 13), -sp.Rational(2, 5), sp.Rational(3, 10), sp.Rational(4, 9)],
        ]
    )
    direct = sp.expand(
        sum(
            projector[alpha, gamma]
            * projector[beta, delta]
            * b_tensor[alpha, beta]
            * b_tensor[gamma, delta]
            for alpha in range(4)
            for beta in range(4)
            for gamma in range(4)
            for delta in range(4)
        )
    )
    block = sp.expand(
        sum(
            expected[alpha, gamma]
            * expected[beta, delta]
            * b_tensor[alpha, beta]
            * b_tensor[gamma, delta]
            for alpha in range(4)
            for beta in range(4)
            for gamma in range(4)
            for delta in range(4)
        )
    )
    unit_residual = sp.factor((u.T * metric * u)[0] + 1)
    annihilation = [sp.factor(value) for value in projector * metric * u]
    contraction_residual = sp.factor(direct - block)
    passed = (
        projector == expected
        and unit_residual == 0
        and all(value == 0 for value in annihilation)
        and contraction_residual == 0
        and projector.rank() == 3
        and projector[0, 0] != 0
    )
    return passed, {
        "unit_branch": "chi^2=1+A_i A^i, chi>0",
        "projector_blocks": {
            "P_nn": "A_i A^i",
            "P_ni": "chi A^i",
            "P_ij": "h^ij+A^i A^j",
        },
        "unit_norm_residual": str(unit_residual),
        "projector_annihilation_residuals": [str(value) for value in annihilation],
        "projector_rank": int(projector.rank()),
        "generic_tilt_normal_normal_entry": str(projector[0, 0]),
        "q_contraction_residual": str(contraction_residual),
        "arithmetic": "exact rational Lorentzian 3+1 block contraction",
        "claim_limit": (
            "The nonzero P_nn block at generic tilt exposes L_n(a) higher-time-derivative "
            "channels; no lapse/shift primary constraint or Ostrogradsky degeneracy is inferred."
        ),
    }
