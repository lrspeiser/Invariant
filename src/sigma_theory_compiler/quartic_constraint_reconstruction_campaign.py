from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski import generic_horndeski_l2_l4_flrw_scalar_reduction_control

SCHEMA_VERSION = "sigma-quartic-constraint-reconstruction-campaign-1.0"


class QuarticConstraintReconstructionError(ValueError):
    """Raised when auxiliary FLRW perturbations cannot be bounded."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = expression.is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


@cache
def _source_constraint_control() -> tuple[bool, dict[str, Any]]:
    return generic_horndeski_l2_l4_flrw_scalar_reduction_control()


@cache
def generic_scalar_constraint_reconstruction_control() -> tuple[bool, dict[str, Any]]:
    """Derive the exact KYY lapse/shift reconstruction and its IR-safe form."""

    g_t, g_s, sigma = sp.symbols("G_T G_S Sigma", real=True, finite=True)
    theta = sp.Symbol("Theta", nonzero=True, real=True, finite=True)
    scale, wave_number = sp.symbols("a k", positive=True, finite=True)
    zeta, zeta_dot = sp.symbols("zeta zeta_dot", real=True, finite=True)
    lapse, shift = sp.symbols("alpha beta", real=True, finite=True)
    laplace_shift = -wave_number**2 * shift / scale**2
    laplace_zeta = -wave_number**2 * zeta / scale**2
    lapse_constraint = sp.factor(
        sigma * lapse
        - theta * laplace_shift
        + 3 * theta * zeta_dot
        - g_t * laplace_zeta
    )
    shift_constraint = sp.factor(theta * lapse - g_t * zeta_dot)
    lapse_solution = sp.factor(g_t * zeta_dot / theta)
    raw_shift_solution = sp.solve(
        lapse_constraint.subs(lapse, lapse_solution), shift, dict=False
    )[0]
    sigma_from_g_s = sp.factor(theta**2 * (g_s - 3 * g_t) / g_t**2)
    shift_solution = sp.factor(raw_shift_solution.subs(sigma, sigma_from_g_s))
    expected_shift = sp.factor(
        -g_t * zeta / theta
        - scale**2 * g_s * zeta_dot / (g_t * wave_number**2)
    )
    residuals = {
        "lapse": sp.factor(
            lapse_constraint.subs(
                {
                    lapse: lapse_solution,
                    shift: expected_shift,
                    sigma: sigma_from_g_s,
                }
            )
        ),
        "shift": sp.factor(shift_constraint.subs(lapse, lapse_solution)),
        "closed_shift": sp.factor(shift_solution - expected_shift),
    }
    longitudinal_shift_vector = sp.factor(wave_number * expected_shift)
    differentiated_shift_vector = sp.factor(wave_number**2 * expected_shift)
    theta_zero_residual = sp.factor(shift_constraint.subs(theta, 0))
    passed = bool(
        all(value == 0 for value in residuals.values())
        and theta_zero_residual != shift_constraint
        and longitudinal_shift_vector.has(wave_number**-1)
        and not differentiated_shift_vector.has(wave_number**-1)
    )
    return passed, {
        "control": "generic KYY scalar lapse/shift Fourier reconstruction",
        "source": {
            "title": "Generalized G-inflation",
            "url": "https://arxiv.org/abs/1105.5723",
            "equations": "4.29-4.34",
        },
        "constraints": {
            "lapse": str(lapse_constraint),
            "shift": str(shift_constraint),
        },
        "solutions": {
            "alpha_k": str(lapse_solution),
            "beta_k": str(expected_shift),
            "longitudinal_shift_vector_k_beta": str(longitudinal_shift_vector),
            "spatial_derivative_of_shift_vector_k2_beta": str(
                differentiated_shift_vector
            ),
        },
        "identity_residuals": {name: str(value) for name, value in residuals.items()},
        "kernel_contract": (
            "The k=0 scalar shift potential is an undetermined Laplacian kernel, but its "
            "physical longitudinal shift vector i*k_i*beta vanishes. On the 2*pi torus, "
            "every nonzero mode has |k|>=1, so the single inverse derivative in k*beta is bounded."
        ),
        "negative_control": {
            "Theta_zero_shift_constraint": str(theta_zero_residual),
            "rejected": theta_zero_residual != shift_constraint,
        },
        "passed": passed,
        "scope": (
            "Exact linear scalar constraint reconstruction on homogeneous FLRW. Time derivatives "
            "of the reconstructed auxiliaries and nonlinear constraint products are separate."
        ),
    }


def certify_quartic_constraint_reconstruction_candidate(
    dirac_candidate: dict[str, Any],
    energy_candidate: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if dirac_candidate.get("candidate_id") != energy_candidate.get("candidate_id"):
        raise QuarticConstraintReconstructionError("candidate ID mismatch")
    if dirac_candidate.get("status") != (
        "pass_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
    ):
        raise QuarticConstraintReconstructionError(
            "candidate lacks the prerequisite Dirac certificate"
        )
    if energy_candidate.get("status") != (
        "pass_finite_horizon_all_wavenumber_linearized_physical_energy"
    ):
        raise QuarticConstraintReconstructionError(
            "candidate lacks the prerequisite physical energy certificate"
        )
    source_passed, _ = _source_constraint_control()
    generic_passed, _ = generic_scalar_constraint_reconstruction_control()
    if not (source_passed and generic_passed):
        raise QuarticConstraintReconstructionError(
            "generic source-bound scalar constraint control failed"
        )

    minimum_wave_number = sp.sympify(config["minimum_nonzero_wave_number"])
    auxiliary_budget = sp.sympify(config["auxiliary_C1_budget"])
    if not _positive(minimum_wave_number):
        raise QuarticConstraintReconstructionError(
            "minimum nonzero wave number must be strictly positive"
        )
    if not _positive(auxiliary_budget):
        raise QuarticConstraintReconstructionError(
            "auxiliary C1 budget must be strictly positive"
        )

    coefficients = dirac_candidate["coefficients"]
    alpha_coupling = sp.sympify(coefficients["a10"])
    c20 = sp.sympify(coefficients["c20"])
    amplitude = sp.sympify(
        dirac_candidate["on_shell_local_flrw_witness"]["A_star"]
    )
    y_max = sp.factor(amplitude**2)
    terminal_fraction = sp.sympify(
        energy_candidate["background_compact_subdomain"]["terminal_fraction"]
    )
    y_min = sp.factor(terminal_fraction * y_max)
    alpha_abs = abs(alpha_coupling)
    c20_abs = abs(c20)
    one_minus_3alpha = sp.factor(1 - 3 * alpha_abs * y_max)
    one_minus_alpha = sp.factor(1 - alpha_abs * y_max)
    two_minus_3c = sp.factor(2 - 3 * c20_abs * y_max)
    if not all(
        _positive(value)
        for value in (one_minus_3alpha, one_minus_alpha, two_minus_3c)
    ):
        raise QuarticConstraintReconstructionError(
            "compact background segment crosses a reconstruction singular surface"
        )

    h_squared_lower = sp.factor(
        y_min
        * two_minus_3c
        / (12 * (1 + 3 * alpha_abs * y_max))
    )
    theta_abs_lower = sp.factor(sp.sqrt(h_squared_lower) * one_minus_3alpha)
    g_t_upper = sp.factor(1 + alpha_abs * y_max)
    lapse_ratio_upper = sp.factor(g_t_upper / theta_abs_lower)
    energy_bounds = energy_candidate["quadratic_energy"][
        "coefficient_interval_bounds"
    ]
    g_s_upper = sp.sympify(energy_bounds["G_S"]["upper"])
    scalar_shift_ratio_upper = sp.factor(g_s_upper / one_minus_alpha)
    u_abs_upper = sp.sympify(
        dirac_candidate["forward_homogeneous_invariant_domain"][
            "uniform_absolute_bounds"
        ]["abs_u"]
    )
    scale_factor_upper = terminal_fraction ** (-1 / (2 * u_abs_upper))
    inverse_wave_number_upper = sp.factor(1 / minimum_wave_number)
    shift_vector_ratio_upper = (
        lapse_ratio_upper
        + scale_factor_upper**2
        * scalar_shift_ratio_upper
        * inverse_wave_number_upper
    )
    reconstruction_operator_upper = max(
        (lapse_ratio_upper, shift_vector_ratio_upper),
        key=lambda value: float(sp.N(value, 30)),
    )

    physical_tube = energy_candidate["physical_derivative_tube"]
    sobolev_constant = sp.sympify(
        physical_tube["sobolev_C1_embedding_constant_upper"]
    )
    coercivity_lower = sp.sympify(physical_tube["coercivity_lower"])
    amplification = sp.sympify(
        energy_candidate["quadratic_energy"]["energy_amplification_upper"]
    )
    source_initial_energy_upper = sp.sympify(
        physical_tube["initial_E_s_strict_upper"]
    )
    reconstruction_initial_energy_upper = (
        coercivity_lower
        * (auxiliary_budget / reconstruction_operator_upper) ** 2
        / (2 * sobolev_constant**2 * amplification)
    )
    chained_initial_energy_upper = min(
        (source_initial_energy_upper, reconstruction_initial_energy_upper),
        key=lambda value: float(sp.N(value, 30)),
    )
    if not _positive(chained_initial_energy_upper):
        raise QuarticConstraintReconstructionError(
            "chained initial-energy radius is not positive"
        )

    clock_gradient_perturbation_upper = sp.factor(
        amplitude * auxiliary_budget
    )
    return {
        "schema_version": "sigma-quartic-constraint-reconstruction-certificate-1.0",
        "status": "pass_linear_lapse_shift_constraint_reconstruction",
        "candidate_id": dirac_candidate["candidate_id"],
        "coefficients": coefficients,
        "compact_background_bounds": {
            "A_star_squared": {"lower": str(y_min), "upper": str(y_max)},
            "H_squared_lower": str(h_squared_lower),
            "Theta_abs_lower": str(theta_abs_lower),
            "G_T_upper": str(g_t_upper),
            "G_T_lower": str(one_minus_alpha),
            "G_S_upper": str(g_s_upper),
            "scale_factor_upper": str(scale_factor_upper),
        },
        "fourier_reconstruction": {
            "lapse_alpha_k": "(G_T/Theta)*dot(zeta_k)",
            "shift_beta_k_nonzero": (
                "-(G_T/Theta)*zeta_k-a^2*(G_S/G_T)*dot(zeta_k)/|k|^2"
            ),
            "physical_shift_vector": "B_i(k)=i*k_i*beta_k",
            "zero_mode": (
                "beta_0 is a potential kernel and B_i(0)=0; alpha_0 remains fixed by "
                "the momentum constraint"
            ),
            "minimum_nonzero_wave_number": str(minimum_wave_number),
            "inverse_wave_number_upper": str(inverse_wave_number_upper),
        },
        "operator_norm_bounds": {
            "G_T_over_abs_Theta_upper": str(lapse_ratio_upper),
            "G_S_over_G_T_upper": str(scalar_shift_ratio_upper),
            "lapse_C1_from_physical_tube_upper": str(lapse_ratio_upper),
            "longitudinal_shift_C1_from_physical_tube_upper": str(
                shift_vector_ratio_upper
            ),
            "combined_reconstruction_upper": str(reconstruction_operator_upper),
            "combined_reconstruction_upper_numeric": float(
                sp.N(reconstruction_operator_upper, 18)
            ),
        },
        "chained_energy_tube": {
            "auxiliary_C1_budget": str(auxiliary_budget),
            "source_physical_initial_E_s_upper": str(source_initial_energy_upper),
            "reconstruction_initial_E_s_upper": str(
                reconstruction_initial_energy_upper
            ),
            "final_initial_E_s_strict_upper": str(chained_initial_energy_upper),
            "final_initial_E_s_strict_upper_numeric": float(
                sp.N(chained_initial_energy_upper, 18)
            ),
            "controlled": [
                "lapse alpha and its first spatial derivatives",
                "physical longitudinal shift B_i and its first spatial derivatives",
                "unitary-clock normal-gradient perturbation delta A_star=-A_star*alpha",
            ],
            "clock_gradient_perturbation_upper": str(
                clock_gradient_perturbation_upper
            ),
        },
        "source_constraint_control_passed": source_passed,
        "generic_reconstruction_identity_passed": generic_passed,
        "claim": (
            "On the declared compact FLRW segment and periodic spatial domain, the exact linear "
            "scalar constraints reconstruct lapse and the physical longitudinal shift as bounded "
            "operators on the previously certified physical Sobolev energy tube."
        ),
        "scope": (
            "This closes spatial C1 reconstruction of the linear auxiliary variables. It does "
            "not control their time derivatives, nonlinear constraint products, vector gauge "
            "sectors, or every Einstein/scalar-Hessian jet in the full physical-space first-order system; a "
            "nonlinear PDE trapping theorem and global boundary energy remain unresolved."
        ),
    }


def run_quartic_constraint_reconstruction_campaign(
    dirac_campaign: dict[str, Any],
    energy_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticConstraintReconstructionError(
                "unsupported campaign schema_version"
            )
        if dirac_campaign.get("status") != (
            "pass_all_12_local_on_shell_adm_dirac_and_quadratic_hamiltonian"
        ):
            raise QuarticConstraintReconstructionError(
                "input Dirac campaign has not passed"
            )
        if energy_campaign.get("status") != (
            "pass_all_12_finite_horizon_linearized_inhomogeneous_energies"
        ):
            raise QuarticConstraintReconstructionError(
                "input physical-energy campaign has not passed"
            )
        if energy_campaign.get("dirac_campaign_sha256") != dirac_campaign.get(
            "content_sha256"
        ):
            raise QuarticConstraintReconstructionError(
                "physical-energy to Dirac campaign hash mismatch"
            )
        expected = int(config.get("expected_candidate_count", 12))
        dirac_candidates = {
            item["candidate_id"]: item for item in dirac_campaign.get("certificates", [])
        }
        energy_candidates = {
            item["candidate_id"]: item for item in energy_campaign.get("certificates", [])
        }
        if len(dirac_candidates) != expected or set(dirac_candidates) != set(
            energy_candidates
        ):
            raise QuarticConstraintReconstructionError(
                "candidate sets do not match the declared count"
            )
        certificates = [
            certify_quartic_constraint_reconstruction_candidate(
                dirac_candidates[candidate_id], energy_candidates[candidate_id], config
            )
            for candidate_id in sorted(dirac_candidates)
        ]
        passed_count = sum(
            item["status"] == "pass_linear_lapse_shift_constraint_reconstruction"
            for item in certificates
        )

        infrared_config = json.loads(json.dumps(config))
        infrared_config["minimum_nonzero_wave_number"] = "0"
        infrared_rejected = False
        infrared_error = ""
        try:
            certify_quartic_constraint_reconstruction_candidate(
                next(iter(dirac_candidates.values())),
                next(iter(energy_candidates.values())),
                infrared_config,
            )
        except QuarticConstraintReconstructionError as error:
            infrared_rejected = True
            infrared_error = str(error)
        generic_passed, generic_evidence = generic_scalar_constraint_reconstruction_control()
        if not (infrared_rejected and generic_passed):
            raise QuarticConstraintReconstructionError(
                "a reconstruction negative control failed"
            )

        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_linear_constraint_reconstructions"
            if passed_count == expected
            else "reject",
            "errors": [],
            "dirac_campaign_sha256": dirac_campaign.get("content_sha256"),
            "energy_campaign_sha256": energy_campaign.get("content_sha256"),
            "config_sha256": hashlib.sha256(
                _canonical_json(config).encode()
            ).hexdigest(),
            "counts": {
                "selected": len(certificates),
                "linear_constraint_reconstruction_passed": passed_count,
                "rejected": len(certificates) - passed_count,
            },
            "generic_identity_control": generic_evidence,
            "certificates": certificates,
            "negative_controls": {
                "Theta_constraint_singularity": generic_evidence[
                    "negative_control"
                ],
                "unbounded_inverse_laplacian_infrared": {
                    "minimum_nonzero_wave_number": "0",
                    "rejected": infrared_rejected,
                    "error": infrared_error,
                },
            },
            "claim": (
                "All 12 candidates have exact bounded linear lapse and longitudinal-shift "
                "reconstruction operators chained to positive physical Sobolev energy radii."
            ),
            "scope": (
                "This is the linear spatial auxiliary reconstruction bridge, not the final "
                "nonlinear physical-space trapping or boundary-energy theorem."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticConstraintReconstructionError) as error:
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": [str(error)],
            "dirac_campaign_sha256": dirac_campaign.get("content_sha256"),
            "energy_campaign_sha256": energy_campaign.get("content_sha256"),
            "counts": {
                "selected": 0,
                "linear_constraint_reconstruction_passed": 0,
                "rejected": 0,
            },
            "certificates": [],
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_constraint_reconstruction_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
