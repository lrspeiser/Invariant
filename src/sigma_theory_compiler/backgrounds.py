from __future__ import annotations

from typing import Any

import sympy as sp


def curved_background_principal_controls() -> dict[str, Any]:
    """Exact metric-cone controls for canonical second-order modes on two curved backgrounds."""

    omega, radial_covector, transverse_covector = sp.symbols(
        "omega k_r k_perp", real=True
    )
    scale_factor = sp.Symbol("a", positive=True, finite=True)
    flrw_polynomial = sp.factor(
        -omega**2 + (radial_covector**2 + transverse_covector**2) / scale_factor**2
    )
    flrw_coordinate_speed_squared = sp.simplify(
        sp.solve(
            sp.Eq(flrw_polynomial.subs(transverse_covector, 0), 0),
            omega**2,
        )[0]
        / radial_covector**2
    )
    flrw_physical_speed_squared = sp.simplify(
        scale_factor**2 * flrw_coordinate_speed_squared
    )

    lapse_function, radial_function, radius = sp.symbols(
        "A B r", positive=True, finite=True
    )
    spherical_radial_polynomial = sp.factor(
        -omega**2 / lapse_function + radial_covector**2 / radial_function
    )
    spherical_coordinate_speed_squared = sp.simplify(
        sp.solve(sp.Eq(spherical_radial_polynomial, 0), omega**2)[0]
        / radial_covector**2
    )
    # Local proper time is sqrt(A) dt and proper radial distance is sqrt(B) dr.
    spherical_physical_speed_squared = sp.simplify(
        radial_function * spherical_coordinate_speed_squared / lapse_function
    )

    mass = sp.Symbol("M", positive=True, finite=True)
    schwarzschild_a = 1 - 2 * mass / radius
    schwarzschild_coordinate_speed_squared = sp.factor(
        spherical_coordinate_speed_squared.subs(
            {lapse_function: schwarzschild_a, radial_function: 1 / schwarzschild_a}
        )
    )
    schwarzschild_physical_speed_squared = sp.factor(
        spherical_physical_speed_squared.subs(
            {lapse_function: schwarzschild_a, radial_function: 1 / schwarzschild_a}
        )
    )

    passed = (
        flrw_physical_speed_squared == 1
        and spherical_physical_speed_squared == 1
        and schwarzschild_physical_speed_squared == 1
        and flrw_coordinate_speed_squared.is_positive is True
        and spherical_coordinate_speed_squared.is_positive is True
    )
    shared_modes = {
        "canonical_scalar": 1,
        "reduced_proca": 3,
        "einstein_hilbert_tt": 2,
    }
    return {
        "passed": passed,
        "scope": "principal, two-derivative metric-coupled sectors after gauge/constraint reduction; lower-order curvature terms do not enter",
        "mode_multiplicities": shared_modes,
        "flrw": {
            "metric": "ds^2 = -dt^2 + a(t)^2 dvec(x)^2",
            "domain": "a(t) > 0",
            "principal_polynomial": str(flrw_polynomial),
            "radial_coordinate_speed_squared": str(flrw_coordinate_speed_squared),
            "orthonormal_physical_speed_squared": str(flrw_physical_speed_squared),
            "strong_hyperbolicity": True,
        },
        "static_spherical": {
            "metric": "ds^2 = -A(r)dt^2 + B(r)dr^2 + r^2 dOmega^2",
            "domain": "A(r) > 0 and B(r) > 0",
            "radial_principal_polynomial": str(spherical_radial_polynomial),
            "radial_coordinate_speed_squared": str(spherical_coordinate_speed_squared),
            "orthonormal_physical_speed_squared": str(spherical_physical_speed_squared),
            "strong_hyperbolicity": True,
        },
        "schwarzschild_exterior": {
            "domain": "r > 2M",
            "radial_coordinate_speed_squared": str(schwarzschild_coordinate_speed_squared),
            "orthonormal_physical_speed_squared": str(schwarzschild_physical_speed_squared),
            "interpretation": "coordinate light speed changes, local physical characteristic speed remains one",
        },
        "warning": "This transports already-reduced canonical principal sectors to curved metrics; it does not extract a principal matrix from an arbitrary generated nonminimal action.",
    }
