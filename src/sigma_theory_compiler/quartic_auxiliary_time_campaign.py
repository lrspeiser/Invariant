from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_constraint_reconstruction_campaign import (
    generic_scalar_constraint_reconstruction_control,
)
from .quartic_dirac_hamiltonian_campaign import _symbolic_flrw_control
from .quartic_linearized_energy_campaign import _rational_abs_upper

SCHEMA_VERSION = "sigma-quartic-auxiliary-time-campaign-1.0"


class QuarticAuxiliaryTimeError(ValueError):
    """Raised when time-differentiated auxiliary reconstruction cannot be bounded."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = expression.is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


@cache
def generic_auxiliary_time_reconstruction_control() -> tuple[bool, dict[str, Any]]:
    """Differentiate the KYY auxiliaries and eliminate ddot(zeta) exactly."""

    r, s, r_dot, s_dot = sp.symbols("R S R_dot S_dot", real=True, finite=True)
    scale, hubble, speed_squared = sp.symbols(
        "a H c_s_squared", positive=True, finite=True
    )
    damping = sp.Symbol("D", real=True, finite=True)
    wave_number = sp.Symbol("k", positive=True, finite=True)
    zeta, zeta_dot, zeta_ddot = sp.symbols(
        "zeta zeta_dot zeta_ddot", real=True, finite=True
    )
    scalar_equation = sp.factor(
        zeta_ddot
        + damping * zeta_dot
        + speed_squared * wave_number**2 * zeta / scale**2
    )
    acceleration_solution = sp.factor(
        -damping * zeta_dot
        - speed_squared * wave_number**2 * zeta / scale**2
    )

    lapse = sp.factor(r * zeta_dot)
    lapse_time_derivative = sp.factor(r_dot * zeta_dot + r * zeta_ddot)
    expected_lapse_time = sp.factor(
        (r_dot - r * damping) * zeta_dot
        - r * speed_squared * wave_number**2 * zeta / scale**2
    )
    lapse_residual = sp.factor(
        lapse_time_derivative.subs(zeta_ddot, acceleration_solution)
        - expected_lapse_time
    )

    # B is the signed scalar amplitude of the physical longitudinal shift
    # vector. Direction cosines have norm one and do not change these bounds.
    shift_vector = sp.factor(
        -r * wave_number * zeta - scale**2 * s * zeta_dot / wave_number
    )
    shift_time_derivative = sp.factor(
        -r_dot * wave_number * zeta
        - r * wave_number * zeta_dot
        - scale**2 * (2 * hubble * s + s_dot) * zeta_dot / wave_number
        - scale**2 * s * zeta_ddot / wave_number
    )
    expected_shift_time = sp.factor(
        -r_dot * wave_number * zeta
        - r * wave_number * zeta_dot
        - scale**2
        * (2 * hubble * s + s_dot - s * damping)
        * zeta_dot
        / wave_number
        + s * speed_squared * wave_number * zeta
    )
    shift_residual = sp.factor(
        shift_time_derivative.subs(zeta_ddot, acceleration_solution)
        - expected_shift_time
    )
    omitted_acceleration_substitution = sp.factor(
        shift_time_derivative - expected_shift_time
    )
    witness = {
        r: 2,
        s: 3,
        r_dot: 5,
        s_dot: 7,
        scale: 1,
        hubble: 11,
        speed_squared: 13,
        damping: 17,
        wave_number: 1,
        zeta: 19,
        zeta_dot: 23,
        zeta_ddot: 29,
    }
    corrupted_witness = sp.factor(omitted_acceleration_substitution.subs(witness))
    prerequisite_passed, _ = generic_scalar_constraint_reconstruction_control()
    passed = bool(
        prerequisite_passed
        and scalar_equation.subs(zeta_ddot, acceleration_solution) == 0
        and lapse_residual == 0
        and shift_residual == 0
        and corrupted_witness != 0
    )
    return passed, {
        "control": "time-differentiated KYY lapse/longitudinal-shift reconstruction",
        "source": {
            "title": "Generalized G-inflation",
            "url": "https://arxiv.org/abs/1105.5723",
            "equations": "4.29-4.34 plus the Euler equation from 4.31",
        },
        "definitions": {
            "R": "G_T/Theta",
            "S": "G_S/G_T",
            "D": "H*(3+d log(G_S)/d log(a))",
            "c_s_squared": "F_S/G_S",
        },
        "scalar_equation": str(scalar_equation),
        "lapse": str(lapse),
        "shift_vector": str(shift_vector),
        "lapse_time_derivative_after_eom": str(expected_lapse_time),
        "shift_time_derivative_after_eom": str(expected_shift_time),
        "identity_residuals": {
            "scalar_equation_on_acceleration_solution": "0",
            "lapse_time": str(lapse_residual),
            "shift_time": str(shift_residual),
        },
        "negative_control": {
            "corruption": "differentiate auxiliaries but omit the scalar acceleration equation",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": corrupted_witness != 0,
        },
        "passed": passed,
        "scope": (
            "Exact linear FLRW auxiliary-time identities. Candidate interval bounds and the "
            "higher Sobolev estimate are supplied by the campaign."
        ),
    }


def _sobolev_c2_embedding_constant_upper(order: int) -> sp.Expr:
    exponent = sp.Rational(order - 2, 3)
    if exponent <= sp.Rational(1, 2):
        raise QuarticAuxiliaryTimeError(
            "Sobolev order must exceed 7/2 to control second spatial derivatives"
        )
    one_dimensional_sum_upper = (
        1
        + sp.sqrt(sp.pi)
        * sp.gamma(exponent - sp.Rational(1, 2))
        / sp.gamma(exponent)
    )
    return one_dimensional_sum_upper ** sp.Rational(3, 2)


def certify_quartic_auxiliary_time_candidate(
    dirac_candidate: dict[str, Any],
    energy_candidate: dict[str, Any],
    reconstruction_candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = dirac_candidate.get("candidate_id")
    if not (
        candidate_id == energy_candidate.get("candidate_id")
        == reconstruction_candidate.get("candidate_id")
    ):
        raise QuarticAuxiliaryTimeError("candidate ID mismatch")
    if reconstruction_candidate.get("status") != (
        "pass_linear_lapse_shift_constraint_reconstruction"
    ):
        raise QuarticAuxiliaryTimeError(
            "candidate lacks the spatial auxiliary reconstruction prerequisite"
        )
    generic_passed, _ = generic_auxiliary_time_reconstruction_control()
    if not generic_passed:
        raise QuarticAuxiliaryTimeError(
            "generic auxiliary-time identity control failed"
        )

    sobolev_order = int(energy_candidate["quadratic_energy"]["sobolev_order"])
    sobolev_c2 = _sobolev_c2_embedding_constant_upper(sobolev_order)
    time_budget = sp.sympify(config["auxiliary_time_C1_budget"])
    minimum_wave_number = sp.sympify(
        reconstruction_candidate["fourier_reconstruction"][
            "minimum_nonzero_wave_number"
        ]
    )
    if not (_positive(time_budget) and _positive(minimum_wave_number)):
        raise QuarticAuxiliaryTimeError(
            "time budget and minimum nonzero wave number must be positive"
        )

    coefficients = dirac_candidate["coefficients"]
    alpha_value = sp.sympify(coefficients["a10"])
    c20_value = sp.sympify(coefficients["c20"])
    amplitude = sp.sympify(
        dirac_candidate["on_shell_local_flrw_witness"]["A_star"]
    )
    y_max = sp.factor(amplitude**2)
    symbolic = _symbolic_flrw_control()
    symbols = symbolic["symbols"]
    y = sp.Symbol("y", positive=True, finite=True)
    substitution = {
        symbols["alpha"]: alpha_value,
        symbols["c20"]: c20_value,
        symbols["A_star"]: sp.sqrt(y),
    }
    g_t = sp.factor(symbolic["G_T"].subs(substitution))
    g_s = sp.factor(symbolic["G_S"].subs(substitution))
    theta = sp.factor(symbolic["Theta"].subs(substitution))
    u = sp.factor(symbolic["u"].subs(substitution))
    r_function = sp.factor(g_t / theta)
    s_function = sp.factor(g_s / g_t)
    drift_r = sp.factor(2 * u * y * sp.diff(r_function, y) / r_function)
    drift_s = sp.factor(2 * u * y * sp.diff(s_function, y) / s_function)
    drift_g_s = sp.factor(2 * u * y * sp.diff(g_s, y) / g_s)
    drift_bounds = {
        "abs_d_log_R_d_log_a_upper": _rational_abs_upper(
            drift_r, y, y_max
        ),
        "abs_d_log_S_d_log_a_upper": _rational_abs_upper(
            drift_s, y, y_max
        ),
        "abs_d_log_G_S_d_log_a_upper": _rational_abs_upper(
            drift_g_s, y, y_max
        ),
    }

    h_upper = sp.sqrt(
        sp.sympify(
            dirac_candidate["forward_homogeneous_invariant_domain"][
                "uniform_absolute_bounds"
            ]["H_squared"]
        )
    )
    operator_bounds = reconstruction_candidate["operator_norm_bounds"]
    r_upper = sp.sympify(operator_bounds["G_T_over_abs_Theta_upper"])
    s_upper = sp.sympify(operator_bounds["G_S_over_G_T_upper"])
    scale_upper = sp.sympify(
        reconstruction_candidate["compact_background_bounds"][
            "scale_factor_upper"
        ]
    )
    energy_bounds = energy_candidate["quadratic_energy"][
        "coefficient_interval_bounds"
    ]
    sound_speed_squared_upper = sp.factor(
        sp.sympify(energy_bounds["F_S"]["upper"])
        / sp.sympify(energy_bounds["G_S"]["lower"])
    )
    damping_upper = h_upper * (
        3 + drift_bounds["abs_d_log_G_S_d_log_a_upper"]
    )
    r_dot_upper = r_upper * h_upper * drift_bounds[
        "abs_d_log_R_d_log_a_upper"
    ]
    s_dot_upper = s_upper * h_upper * drift_bounds[
        "abs_d_log_S_d_log_a_upper"
    ]
    inverse_wave_number = 1 / minimum_wave_number
    lapse_time_operator = (
        r_dot_upper
        + r_upper * damping_upper
        + r_upper * sound_speed_squared_upper
    )
    shift_time_operator = (
        r_upper
        + r_dot_upper
        + scale_upper**2
        * (2 * h_upper * s_upper + s_dot_upper)
        * inverse_wave_number
        + scale_upper**2 * s_upper * damping_upper * inverse_wave_number
        + s_upper * sound_speed_squared_upper
    )
    combined_time_operator = max(
        (lapse_time_operator, shift_time_operator),
        key=lambda value: float(sp.N(value, 30)),
    )

    physical_tube = energy_candidate["physical_derivative_tube"]
    coercivity_lower = sp.sympify(physical_tube["coercivity_lower"])
    amplification = sp.sympify(
        energy_candidate["quadratic_energy"]["energy_amplification_upper"]
    )
    spatial_chain_upper = sp.sympify(
        reconstruction_candidate["chained_energy_tube"][
            "final_initial_E_s_strict_upper"
        ]
    )
    time_reconstruction_upper = (
        coercivity_lower
        * (time_budget / combined_time_operator) ** 2
        / (2 * sobolev_c2**2 * amplification)
    )
    final_initial_energy_upper = min(
        (spatial_chain_upper, time_reconstruction_upper),
        key=lambda value: float(sp.N(value, 30)),
    )
    if not _positive(final_initial_energy_upper):
        raise QuarticAuxiliaryTimeError(
            "auxiliary-time chained energy radius is not positive"
        )

    return {
        "schema_version": "sigma-quartic-auxiliary-time-certificate-1.0",
        "status": "pass_linear_auxiliary_time_reconstruction",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "coefficient_drift_bounds": {
            name: str(value) for name, value in drift_bounds.items()
        },
        "evolution_bounds": {
            "H_upper": str(h_upper),
            "R_G_T_over_Theta_upper": str(r_upper),
            "S_G_S_over_G_T_upper": str(s_upper),
            "R_dot_upper": str(r_dot_upper),
            "S_dot_upper": str(s_dot_upper),
            "scalar_damping_D_upper": str(damping_upper),
            "sound_speed_squared_upper": str(sound_speed_squared_upper),
            "scale_factor_upper": str(scale_upper),
        },
        "time_reconstruction_operator": {
            "lapse_time_C1_upper": str(lapse_time_operator),
            "shift_time_C1_upper": str(shift_time_operator),
            "combined_upper": str(combined_time_operator),
            "combined_upper_numeric": float(sp.N(combined_time_operator, 18)),
            "minimum_nonzero_wave_number": str(minimum_wave_number),
        },
        "chained_energy_tube": {
            "sobolev_order": sobolev_order,
            "sobolev_C2_embedding_constant_upper": str(sobolev_c2),
            "auxiliary_time_C1_budget": str(time_budget),
            "spatial_reconstruction_initial_E_s_upper": str(spatial_chain_upper),
            "time_reconstruction_initial_E_s_upper": str(
                time_reconstruction_upper
            ),
            "final_initial_E_s_strict_upper": str(final_initial_energy_upper),
            "final_initial_E_s_strict_upper_numeric": float(
                sp.N(final_initial_energy_upper, 18)
            ),
            "controlled": [
                "time derivative of lapse and its first spatial derivatives",
                "time derivative of physical longitudinal shift and its first spatial derivatives",
            ],
        },
        "generic_time_identity_passed": generic_passed,
        "claim": (
            "The scalar evolution equation converts the time derivatives of the exactly "
            "reconstructed linear lapse and longitudinal shift into bounded operators on the "
            "same physical Sobolev solution, with an explicit positive chained energy radius."
        ),
        "scope": (
            "This completes linear scalar auxiliary reconstruction through one time derivative. "
            "It does not bound nonlinear constraint products, modified-harmonic gauge-sector "
            "variables, quasilinear commutators, or every full-system spacetime jet; nonlinear "
            "PDE trapping and boundary energy remain unresolved."
        ),
    }


def run_quartic_auxiliary_time_campaign(
    dirac_campaign: dict[str, Any],
    energy_campaign: dict[str, Any],
    reconstruction_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticAuxiliaryTimeError("unsupported campaign schema_version")
        if reconstruction_campaign.get("status") != (
            "pass_all_12_linear_constraint_reconstructions"
        ):
            raise QuarticAuxiliaryTimeError(
                "input spatial reconstruction campaign has not passed"
            )
        if reconstruction_campaign.get("dirac_campaign_sha256") != (
            dirac_campaign.get("content_sha256")
        ) or reconstruction_campaign.get("energy_campaign_sha256") != (
            energy_campaign.get("content_sha256")
        ):
            raise QuarticAuxiliaryTimeError(
                "reconstruction prerequisite hash mismatch"
            )
        expected = int(config.get("expected_candidate_count", 12))
        dirac = {item["candidate_id"]: item for item in dirac_campaign["certificates"]}
        energy = {item["candidate_id"]: item for item in energy_campaign["certificates"]}
        reconstruction = {
            item["candidate_id"]: item
            for item in reconstruction_campaign["certificates"]
        }
        if not (
            len(dirac) == expected
            and set(dirac) == set(energy) == set(reconstruction)
        ):
            raise QuarticAuxiliaryTimeError("candidate sets do not match")
        certificates = [
            certify_quartic_auxiliary_time_candidate(
                dirac[candidate_id],
                energy[candidate_id],
                reconstruction[candidate_id],
                config,
            )
            for candidate_id in sorted(dirac)
        ]
        passed_count = sum(
            item["status"] == "pass_linear_auxiliary_time_reconstruction"
            for item in certificates
        )
        bad_energy = json.loads(json.dumps(energy_campaign))
        for item in bad_energy["certificates"]:
            item["quadratic_energy"]["sobolev_order"] = 3
        sobolev_rejected = False
        sobolev_error = ""
        try:
            certify_quartic_auxiliary_time_candidate(
                next(iter(dirac.values())),
                bad_energy["certificates"][0],
                next(iter(reconstruction.values())),
                config,
            )
        except QuarticAuxiliaryTimeError as error:
            sobolev_rejected = True
            sobolev_error = str(error)
        identity_passed, identity = generic_auxiliary_time_reconstruction_control()
        if not (sobolev_rejected and identity_passed):
            raise QuarticAuxiliaryTimeError(
                "an auxiliary-time negative control failed"
            )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_linear_auxiliary_time_reconstructions"
            if passed_count == expected
            else "reject",
            "errors": [],
            "dirac_campaign_sha256": dirac_campaign.get("content_sha256"),
            "energy_campaign_sha256": energy_campaign.get("content_sha256"),
            "reconstruction_campaign_sha256": reconstruction_campaign.get(
                "content_sha256"
            ),
            "config_sha256": hashlib.sha256(
                _canonical_json(config).encode()
            ).hexdigest(),
            "counts": {
                "selected": len(certificates),
                "linear_auxiliary_time_reconstruction_passed": passed_count,
                "rejected": len(certificates) - passed_count,
            },
            "generic_identity_control": identity,
            "certificates": certificates,
            "negative_controls": {
                "omitted_scalar_acceleration_equation": identity[
                    "negative_control"
                ],
                "insufficient_sobolev_order": {
                    "mutated_order": 3,
                    "rejected": sobolev_rejected,
                    "error": sobolev_error,
                },
            },
            "claim": (
                "All 12 candidates have bounded linear lapse/shift reconstruction through one "
                "time derivative, chained to explicit positive physical-energy radii."
            ),
            "scope": (
                "The linear scalar reconstruction chain is complete through spatial C1 and one "
                "time derivative. Nonlinear products, full gauge-sector variables, quasilinear "
                "commutators, PDE trapping, and boundary energy remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticAuxiliaryTimeError) as error:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": [str(error)],
            "dirac_campaign_sha256": dirac_campaign.get("content_sha256"),
            "energy_campaign_sha256": energy_campaign.get("content_sha256"),
            "reconstruction_campaign_sha256": reconstruction_campaign.get(
                "content_sha256"
            ),
            "counts": {
                "selected": 0,
                "linear_auxiliary_time_reconstruction_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_auxiliary_time_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
