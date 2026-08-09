from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_dirac_hamiltonian_campaign import _symbolic_flrw_control

SCHEMA_VERSION = "sigma-quartic-linearized-energy-campaign-1.0"


class QuarticLinearizedEnergyError(ValueError):
    """Raised when the reduced inhomogeneous energy theorem cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


def _polynomial_abs_upper(
    expression: sp.Expr, variable: sp.Symbol, upper: sp.Expr
) -> sp.Expr:
    polynomial = sp.Poly(sp.expand(expression), variable)
    return sp.factor(
        sum(
            abs(coefficient) * upper ** powers[0]
            for powers, coefficient in polynomial.terms()
        )
    )


def _signed_polynomial_lower(
    expression: sp.Expr, variable: sp.Symbol, upper: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
    polynomial = sp.Poly(sp.expand(expression), variable)
    constant = polynomial.coeff_monomial(1)
    sign = sp.sign(constant)
    if sign not in (-1, 1):
        raise QuarticLinearizedEnergyError(
            "a rational energy coefficient has zero constant term"
        )
    tail = sum(
        abs(coefficient) * upper ** powers[0]
        for powers, coefficient in polynomial.terms()
        if powers[0] > 0
    )
    lower = sp.factor(abs(constant) - tail)
    if not _positive(lower):
        raise QuarticLinearizedEnergyError(
            "interval polynomial sign cannot be certified"
        )
    return sign, lower


def _positive_rational_bounds(
    expression: sp.Expr, variable: sp.Symbol, upper: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
    numerator, denominator = sp.fraction(sp.cancel(expression))
    numerator_sign, numerator_lower = _signed_polynomial_lower(
        numerator, variable, upper
    )
    denominator_sign, denominator_lower = _signed_polynomial_lower(
        denominator, variable, upper
    )
    if numerator_sign != denominator_sign:
        raise QuarticLinearizedEnergyError(
            "energy coefficient is negative at the exact baseline"
        )
    numerator_upper = _polynomial_abs_upper(numerator, variable, upper)
    denominator_upper = _polynomial_abs_upper(denominator, variable, upper)
    lower = sp.factor(numerator_lower / denominator_upper)
    upper_bound = sp.factor(numerator_upper / denominator_lower)
    if not _positive(lower):
        raise QuarticLinearizedEnergyError(
            "energy coefficient lacks a strictly positive interval lower bound"
        )
    return lower, upper_bound


def _rational_abs_upper(
    expression: sp.Expr, variable: sp.Symbol, upper: sp.Expr
) -> sp.Expr:
    numerator, denominator = sp.fraction(sp.cancel(expression))
    _, denominator_lower = _signed_polynomial_lower(denominator, variable, upper)
    return sp.factor(
        _polynomial_abs_upper(numerator, variable, upper) / denominator_lower
    )


def _coefficient_energy_bounds(
    expression: sp.Expr,
    u: sp.Expr,
    variable: sp.Symbol,
    upper: sp.Expr,
) -> dict[str, sp.Expr]:
    lower, upper_bound = _positive_rational_bounds(expression, variable, upper)
    logarithmic_drift = sp.factor(
        2 * u * variable * sp.diff(expression, variable) / expression
    )
    return {
        "lower": lower,
        "upper": upper_bound,
        "abs_d_log_coefficient_d_log_a_upper": _rational_abs_upper(
            logarithmic_drift, variable, upper
        ),
    }


def _sobolev_c1_embedding_constant_upper(order: int) -> sp.Expr:
    # On the 2*pi-periodic three-torus, Cauchy--Schwarz gives the exact Fourier
    # lattice sum.  The product majorant below replaces it by three convergent
    # one-dimensional sums, each bounded by its monotone integral.
    exponent = sp.Rational(order - 1, 3)
    if exponent <= sp.Rational(1, 2):
        raise QuarticLinearizedEnergyError(
            "Sobolev order must exceed 5/2 to control first spatial derivatives"
        )
    one_dimensional_sum_upper = sp.factor(
        1
        + sp.sqrt(sp.pi)
        * sp.gamma(exponent - sp.Rational(1, 2))
        / sp.gamma(exponent)
    )
    return sp.factor(one_dimensional_sum_upper ** sp.Rational(3, 2))


def certify_quartic_linearized_energy_candidate(
    candidate: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if candidate.get("status") != (
        "pass_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
    ):
        raise QuarticLinearizedEnergyError(
            "candidate lacks the prerequisite ADM/Dirac/Hamiltonian certificate"
        )
    if not candidate.get("forward_homogeneous_invariant_domain", {}).get("passed"):
        raise QuarticLinearizedEnergyError(
            "candidate lacks a forward homogeneous invariant domain"
        )
    coefficients = candidate.get("coefficients")
    if not isinstance(coefficients, dict):
        raise QuarticLinearizedEnergyError("candidate coefficients are missing")
    alpha_value = sp.sympify(coefficients["a10"])
    c20_value = sp.sympify(coefficients["c20"])
    amplitude = sp.sympify(
        candidate["on_shell_local_flrw_witness"]["A_star"]
    )
    y_max = sp.factor(amplitude**2)
    if not _positive(y_max):
        raise QuarticLinearizedEnergyError("initial clock amplitude must be positive")

    terminal_fraction = sp.sympify(config["terminal_amplitude_squared_fraction"])
    if not (_positive(terminal_fraction) and _positive(1 - terminal_fraction)):
        raise QuarticLinearizedEnergyError(
            "terminal amplitude-squared fraction must lie strictly between zero and one"
        )
    sobolev_order = int(config["sobolev_order"])
    physical_budget = sp.sympify(config["physical_mode_derivative_budget"])
    if not _positive(physical_budget):
        raise QuarticLinearizedEnergyError(
            "physical-mode derivative budget must be positive"
        )

    symbolic = _symbolic_flrw_control()
    symbols = symbolic["symbols"]
    y = sp.Symbol("y", positive=True, finite=True)
    substitution = {
        symbols["alpha"]: alpha_value,
        symbols["c20"]: c20_value,
        symbols["A_star"]: sp.sqrt(y),
    }
    functions = {
        name: sp.factor(symbolic[name].subs(substitution))
        for name in ("G_T", "F_T", "G_S", "F_S")
    }
    u = sp.factor(symbolic["u"].subs(substitution))
    bounds = {
        name: _coefficient_energy_bounds(expression, u, y, y_max)
        for name, expression in functions.items()
    }

    h_squared_upper = sp.sympify(
        candidate["forward_homogeneous_invariant_domain"]["uniform_absolute_bounds"][
            "H_squared"
        ]
    )
    u_abs_upper = sp.sympify(
        candidate["forward_homogeneous_invariant_domain"]["uniform_absolute_bounds"][
            "abs_u"
        ]
    )
    if not (_positive(h_squared_upper) and _positive(u_abs_upper)):
        raise QuarticLinearizedEnergyError(
            "background evolution bounds are not strictly positive"
        )
    h_upper = sp.sqrt(h_squared_upper)
    logarithmic_interval = sp.log(1 / terminal_fraction)
    proper_time_horizon = sp.factor(
        logarithmic_interval / (2 * u_abs_upper * h_upper)
    )

    gamma_by_sector = {
        "tensor_momentum": sp.factor(
            3 + bounds["G_T"]["abs_d_log_coefficient_d_log_a_upper"]
        ),
        "tensor_gradient": sp.factor(
            1 + bounds["F_T"]["abs_d_log_coefficient_d_log_a_upper"]
        ),
        "scalar_momentum": sp.factor(
            3 + bounds["G_S"]["abs_d_log_coefficient_d_log_a_upper"]
        ),
        "scalar_gradient": sp.factor(
            1 + bounds["F_S"]["abs_d_log_coefficient_d_log_a_upper"]
        ),
    }
    gamma_upper = max(gamma_by_sector.values(), key=lambda value: float(sp.N(value)))
    amplification = sp.factor(
        terminal_fraction ** (-gamma_upper / (2 * u_abs_upper))
    )
    coercivity_lower = min(
        (bounds[name]["lower"] for name in bounds),
        key=lambda value: float(sp.N(value)),
    )
    sobolev_constant = _sobolev_c1_embedding_constant_upper(sobolev_order)
    initial_energy_upper = sp.factor(
        coercivity_lower
        * physical_budget**2
        / (2 * sobolev_constant**2 * amplification)
    )
    if not _positive(initial_energy_upper):
        raise QuarticLinearizedEnergyError(
            "certified initial physical energy radius is not positive"
        )

    return {
        "schema_version": "sigma-quartic-linearized-energy-certificate-1.0",
        "status": "pass_finite_horizon_all_wavenumber_linearized_physical_energy",
        "candidate_id": candidate["candidate_id"],
        "coefficients": coefficients,
        "background_compact_subdomain": {
            "initial_A_star_squared": str(y_max),
            "terminal_A_star_squared": str(sp.factor(terminal_fraction * y_max)),
            "terminal_fraction": str(terminal_fraction),
            "proper_time_horizon_lower": str(proper_time_horizon),
            "proper_time_horizon_lower_numeric": float(sp.N(proper_time_horizon, 18)),
            "proof": (
                "|d log(A_star^2)/dt|<=2*u_abs_upper*H_upper and A_star^2 "
                "decreases, so it cannot fall below the declared terminal fraction before "
                "the certified proper-time horizon"
            ),
        },
        "quadratic_energy": {
            "physical_modes": ["tensor_plus", "tensor_cross", "scalar_zeta"],
            "spatial_domain": "three-torus with coordinate period 2*pi",
            "sobolev_order": sobolev_order,
            "definition": (
                "E_s=sum_modes 1/2[a^3 G_mode ||dot Q||_Hs^2 + "
                "a F_mode ||grad Q||_Hs^2]"
            ),
            "all_spatial_wavenumbers": True,
            "zero_mode_scope": (
                "constant coordinate modes carry no gradient energy; their time derivatives "
                "are controlled and constant shifts do not enter the certified derivative ledger"
            ),
            "coefficient_functions": {
                name: str(expression) for name, expression in functions.items()
            },
            "coefficient_interval_bounds": {
                name: {key: str(value) for key, value in record.items()}
                for name, record in bounds.items()
            },
            "logarithmic_rate_multipliers": {
                name: str(value) for name, value in gamma_by_sector.items()
            },
            "gamma_upper": str(gamma_upper),
            "energy_amplification_upper": str(amplification),
            "energy_amplification_upper_numeric": float(sp.N(amplification, 18)),
            "gronwall_identity": (
                "For each canonical Fourier mode H_k=p^2/(2a^3G)+aF k^2q^2/2; "
                "Hamilton's equations cancel the pq terms and explicit time dependence gives "
                "dE_s/dt<=gamma_upper*H*E_s"
            ),
        },
        "physical_derivative_tube": {
            "sobolev_C1_embedding_constant_upper": str(sobolev_constant),
            "coercivity_lower": str(coercivity_lower),
            "declared_pointwise_budget": str(physical_budget),
            "initial_E_s_strict_upper": str(initial_energy_upper),
            "initial_E_s_strict_upper_numeric": float(sp.N(initial_energy_upper, 18)),
            "controlled_quantities": [
                "dot Q and its first spatial derivatives",
                "grad Q and its first spatial derivatives",
            ],
            "proof": (
                "E_s(t)<=amplification*E_s(0), coercivity controls the H^s norms, "
                "and the explicit Fourier-lattice Sobolev majorant controls C^1 point values"
            ),
        },
        "primary_sources": [
            {
                "title": "Generalized G-inflation",
                "url": "https://arxiv.org/abs/1105.5723",
                "equations": "4.3-4.8 and 4.24-4.34",
            },
            {
                "title": "Well-posed formulation of Lovelock and Horndeski theories",
                "url": "https://arxiv.org/abs/2003.08398",
                "result": "modified-harmonic strong hyperbolicity in weak coupling",
            },
        ],
        "claim": (
            "Every physical Fourier mode, at every spatial wavenumber, has a coercive "
            "time-dependent quadratic energy with an explicit finite-horizon amplification "
            "bound on the declared compact segment of the expanding FLRW branch."
        ),
        "scope": (
            "This is a linearized inhomogeneous physical-mode theorem, not a nonlinear PDE "
            "trapping theorem. It does not reconstruct lapse, shift, gauge, constraint, or all "
            "spacetime-jet components from the reduced variables; it therefore does not claim "
            "that the full 22-variable nonlinear solution remains in the symmetrizer box."
        ),
    }


def run_quartic_linearized_energy_campaign(
    dirac_campaign: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticLinearizedEnergyError("unsupported campaign schema_version")
        if dirac_campaign.get("status") != (
            "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        ):
            raise QuarticLinearizedEnergyError(
                "input ADM/Dirac/Hamiltonian campaign has not passed"
            )
        expected = int(config.get("expected_candidate_count", 12))
        candidates = dirac_campaign.get("certificates", [])
        if len(candidates) != expected:
            raise QuarticLinearizedEnergyError(
                f"expected {expected} prerequisite candidates, found {len(candidates)}"
            )
        certificates = [
            certify_quartic_linearized_energy_candidate(candidate, config)
            for candidate in candidates
        ]
        passed_count = sum(
            certificate["status"]
            == "pass_finite_horizon_all_wavenumber_linearized_physical_energy"
            for certificate in certificates
        )

        bad_fraction = json.loads(json.dumps(config))
        bad_fraction["terminal_amplitude_squared_fraction"] = "1"
        fraction_rejected = False
        fraction_error = ""
        try:
            certify_quartic_linearized_energy_candidate(candidates[0], bad_fraction)
        except QuarticLinearizedEnergyError as error:
            fraction_rejected = True
            fraction_error = str(error)
        ghost_rejected = False
        ghost_error = ""
        try:
            y = sp.Symbol("y", positive=True)
            _positive_rational_bounds(-sp.Integer(1), y, sp.Rational(1, 10))
        except QuarticLinearizedEnergyError as error:
            ghost_rejected = True
            ghost_error = str(error)
        if not (fraction_rejected and ghost_rejected):
            raise QuarticLinearizedEnergyError(
                "a linearized-energy negative control did not reject"
            )

        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_finite_horizon_linearized_inhomogeneous_energies"
            if passed_count == expected
            else "reject",
            "errors": [],
            "dirac_campaign_sha256": dirac_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(
                _canonical_json(config).encode()
            ).hexdigest(),
            "counts": {
                "selected": len(candidates),
                "finite_horizon_linearized_energy_passed": passed_count,
                "rejected": len(candidates) - passed_count,
            },
            "certificates": sorted(
                certificates, key=lambda item: item["candidate_id"]
            ),
            "negative_controls": {
                "zero_length_compact_background_segment": {
                    "terminal_fraction": "1",
                    "rejected": fraction_rejected,
                    "error": fraction_error,
                },
                "scalar_gradient_ghost": {
                    "mutated_F_S": "-1",
                    "rejected": ghost_rejected,
                    "error": ghost_error,
                },
            },
            "claim": (
                "All 12 quartic candidates have an explicit all-wavenumber finite-horizon "
                "linearized inhomogeneous physical-energy estimate on a compact portion of "
                "their exact expanding FLRW branches."
            ),
            "scope": (
                "This closes the reduced linear physical-mode energy step only. Full nonlinear "
                "PDE trapping, constraint/gauge reconstruction, nonlinear boundary energy, and "
                "observational promotion remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticLinearizedEnergyError) as error:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": [str(error)],
            "dirac_campaign_sha256": dirac_campaign.get("content_sha256"),
            "counts": {
                "selected": 0,
                "finite_horizon_linearized_energy_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_linearized_energy_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
