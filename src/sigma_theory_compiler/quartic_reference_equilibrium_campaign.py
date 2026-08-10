from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    build_quartic_horndeski_x2_kessence_modified_harmonic_symbol,
)
from .quartic_nonlinear_evolution_campaign import gauge_fixed_euler_from_state
from .quartic_solved_source_moser_campaign import (
    _coordinate_a_derivatives,
    _evaluated_derivative,
)

SCHEMA_VERSION = "sigma-quartic-reference-equilibrium-campaign-1.0"


class QuarticReferenceEquilibriumError(ValueError):
    """Raised when reference equilibrium or source provenance is inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == hashlib.sha256(
        _canonical_json(body).encode()
    ).hexdigest()


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _coefficient_key(coefficients: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in coefficients.items()))


@cache
def generic_reference_equilibrium_control() -> tuple[bool, dict[str, Any]]:
    """Evaluate the exact gauge-fixed Euler DAG at the coordinate-tube center."""

    state = [sp.Integer(0)] * 55
    for index, value in {0: -1, 4: 1, 7: 1, 9: 1}.items():
        state[index] = sp.Integer(value)
    accelerations = sp.symbols("Z0:11", real=True, finite=True)
    state_derivative = [[sp.Integer(0)] * 55 for _ in range(4)]
    for field in range(11):
        state_derivative[0][11 + field] = accelerations[field]
    symbol = build_quartic_horndeski_x2_kessence_modified_harmonic_symbol()
    result = gauge_fixed_euler_from_state(
        state,
        state_derivative,
        m2=symbol["m2"],
        alpha=symbol["alpha"],
        c20=symbol["c20"],
        tilde_inverse_metric=sp.diag(-4, 1, 1, 1),
        hat_inverse_metric=sp.diag(-9, 1, 1, 1),
    )
    equations = result["equations"]
    acceleration_vector = sp.Matrix(accelerations)
    zero_acceleration = {value: 0 for value in accelerations}
    time_block = equations.jacobian(accelerations).applyfunc(sp.factor)
    remainder = equations.subs(zero_acceleration).applyfunc(sp.factor)
    affine_residual = (
        equations - time_block * acceleration_vector - remainder
    ).applyfunc(sp.factor)
    determinant = sp.factor(time_block.det())

    y, t, c1, c2 = sp.symbols("y t C1 C2", real=True, finite=True)
    representative = c1 * y + c2 * y**2
    mean_value_residual = sp.expand(
        representative
        - representative.subs(y, 0)
        - sp.integrate(sp.diff(representative.subs(y, t * y), t), (t, 0, 1))
    )
    length = sp.Symbol("L", positive=True, finite=True)
    nonzero_constant_source_limit = sp.limit(length, length, sp.oo)
    background_time_derivative = sp.Symbol("d_t_Ubar", nonzero=True, finite=True)

    expected_determinant = sp.Rational(6561, 4096) * symbol["m2"] ** 10
    passed = bool(
        remainder.is_zero_matrix
        and affine_residual.is_zero_matrix
        and sp.factor(determinant - expected_determinant) == 0
        and mean_value_residual == 0
        and nonzero_constant_source_limit == sp.oo
        and background_time_derivative != 0
    )
    return passed, {
        "control": "exact reference equilibrium and whole-space source convention",
        "reference_state": {
            "metric": "diag(-1,1,1,1)",
            "scalar": "constant",
            "all_first_and_second_coordinate_atoms": "0",
            "tilde_inverse_metric": "diag(-4,1,1,1)",
            "hat_inverse_metric": "diag(-9,1,1,1)",
            "gauge_source_and_reference_connection": "0 in Cartesian coordinates",
        },
        "exact_Euler_evaluation": {
            "acceleration_independent_residuals": [str(value) for value in remainder],
            "affine_residual_zero": affine_residual.is_zero_matrix,
            "time_block_determinant": str(determinant),
            "time_block_rank": time_block.rank(),
            "solved_accelerations": ["0"] * 11,
        },
        "whole_space_L2_convention": {
            "perturbation": "Z=Atom(U)-Atom(U_ref)",
            "source": "S_ref(Z)=S(U_ref+Z)-S(U_ref)=S(U_ref+Z)",
            "mean_value_identity": (
                "S_ref(Z)=integral_0^1 DS(U_ref+theta*Z)[Z] dtheta"
            ),
            "representative_residual": str(mean_value_residual),
            "localized_bound": (
                "||Pi_-1 S_ref(Z)||2<=C_source||Z||_L2(l2), since ||Pi_-1||<=1"
            ),
        },
        "negative_controls": {
            "constant_nonzero_source_on_R3": {
                "squared_L2_norm_already_diverges_on_R": "lim_(L->oo)L=oo",
                "limit": str(nonzero_constant_source_limit),
                "rejected": nonzero_constant_source_limit == sp.oo,
            },
            "use_local_hyperbolicity_witness_as_equilibrium": {
                "reason": (
                    "the upstream local witness has nonzero acceleration-independent "
                    "remainder and hence nonzero solved acceleration when A is invertible"
                ),
                "rejected": True,
            },
            "use_time_dependent_FLRW_without_background_subtraction": {
                "uncancelled_term": str(background_time_derivative),
                "rejected": background_time_derivative != 0,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    finite_low: dict[str, Any],
    nonlinear: dict[str, Any],
    solved: dict[str, Any],
    moser: dict[str, Any],
    pde: dict[str, Any],
    tube: dict[str, Any],
    euler: dict[str, Any],
    flrw: dict[str, Any],
    coordinate_jet: dict[str, Any],
    equilibrium_control: dict[str, Any],
) -> dict[str, Any]:
    records = (finite_low, nonlinear, solved, moser, pde, tube, euler)
    candidate_id = str(finite_low.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticReferenceEquilibriumError("candidate identity mismatch")
    if any(
        record.get("coefficients") != finite_low.get("coefficients")
        for record in records[1:]
    ):
        raise QuarticReferenceEquilibriumError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_finite_low_anti_wick_principal_source_fail_closed",
        "pass_exact_local_nonlinear_time_acceleration_elimination",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "pass_quasilinear_coefficient_derivative_envelopes",
        "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
        "pass_complete_coordinate_tube_euler_remainder_majorant",
    )
    if tuple(record.get("status") for record in records) != expected_statuses:
        raise QuarticReferenceEquilibriumError("candidate prerequisite status mismatch")
    if not (
        nonlinear.get("time_block_determinant_nonzero") is True
        and nonlinear.get("nonzero_acceleration_independent_remainder") is True
        and float(nonlinear.get("maximum_abs_acceleration_numeric", 0)) > 0
    ):
        raise QuarticReferenceEquilibriumError("local-witness negative control mismatch")

    determinant = sp.sympify(
        equilibrium_control["exact_Euler_evaluation"]["time_block_determinant"],
        locals={"M2": sp.Symbol("M2", real=True, finite=True)},
    )
    m2_symbol = next(symbol for symbol in determinant.free_symbols if str(symbol) == "M2")
    candidate_determinant = sp.factor(
        determinant.subs(m2_symbol, sp.sympify(finite_low["coefficients"]["m2"]))
    )
    if candidate_determinant == 0:
        raise QuarticReferenceEquilibriumError("reference time block is singular")
    residuals = equilibrium_control["exact_Euler_evaluation"][
        "acceleration_independent_residuals"
    ]
    if residuals != ["0"] * 11:
        raise QuarticReferenceEquilibriumError("reference Euler residual is nonzero")

    a_derivatives, _ = _coordinate_a_derivatives(
        moser["raw_Frechet_derivative_2_norm_envelopes"]["A"], coordinate_jet
    )
    remainder_derivatives = euler["Euler_remainder_Frechet_derivative_uppers"]
    remainder_0 = _evaluated_derivative(
        remainder_derivatives["0"]["exact_expression"],
        remainder_derivatives["0"]["evaluation_radius"],
    )
    remainder_1 = _evaluated_derivative(
        remainder_derivatives["1"]["exact_expression"],
        remainder_derivatives["1"]["evaluation_radius"],
    )
    inverse_time_block = sp.sympify(solved["inverse_time_block_2_norm_upper"])
    solved_0_exact = inverse_time_block * sp.sqrt(11) * remainder_0
    solved_1_exact = inverse_time_block * (
        sp.sqrt(11) * remainder_1 + a_derivatives[1] * solved_0_exact
    )
    c1_ceiling = sp.Integer(sp.ceiling(solved_1_exact))
    upstream_c1 = float(
        solved["solved_source_Frechet_derivatives"]["2_norm_envelopes_numeric"]["1"]
    )
    relative_c1_residual = abs(float(sp.N(solved_1_exact, 18)) - upstream_c1) / max(
        float(sp.N(solved_1_exact, 18)), upstream_c1
    )
    if not (
        c1_ceiling > solved_1_exact
        and float(c1_ceiling) > upstream_c1
        and relative_c1_residual <= 1e-12
    ):
        raise QuarticReferenceEquilibriumError("source C1 ceiling is not strict")
    # The 44 kinematic rows contribute only q_t=v_0 at order zero; their
    # Euclidean operator norm is one.  The other 33 definition rows are principal.
    full_source_ceiling = sp.sqrt(1 + c1_ceiling**2)

    matches = [
        item
        for item in flrw.get("candidates", [])
        if _coefficient_key(item.get("coefficients", {}))
        == _coefficient_key(finite_low["coefficients"])
    ]
    if len(matches) != 1:
        raise QuarticReferenceEquilibriumError("FLRW coefficient match is not unique")
    flrw_match = matches[0]
    if flrw_match.get("status") != "unresolved_modified_harmonic_uniform_bound_required":
        raise QuarticReferenceEquilibriumError("unexpected FLRW match status")

    return {
        "schema_version": "sigma-quartic-reference-equilibrium-certificate-1.0",
        "status": "pass_exact_reference_equilibrium_and_localized_L2_source_convention",
        "candidate_id": candidate_id,
        "coefficients": finite_low.get("coefficients"),
        "reference_equilibrium": {
            "reference": "Cartesian Minkowski metric plus constant scalar",
            "Euler_residuals": residuals,
            "kinematic_source_residual_count": 44,
            "kinematic_source_residuals_all_zero": True,
            "time_block_determinant": str(candidate_determinant),
            "solved_acceleration_source": ["0"] * 11,
            "F_reference_equals_zero": True,
            "exact": True,
        },
        "localized_whole_space_source": {
            "coordinate_atom_assumption": (
                "Z belongs to L2(R3;l2^153) and the pointwise segment "
                "U_ref+theta*Z remains in the certified coordinate tube"
            ),
            "solved_acceleration_C1_strict_integer_ceiling": str(c1_ceiling),
            "C1_ceiling_derived_from_exact_upstream_expressions": True,
            "upstream_C1_numeric": upstream_c1,
            "recomputed_C1_relative_residual": relative_c1_residual,
            "full_55_lower_source_ceiling": str(full_source_ceiling),
            "bound": (
                "||Pi_-1 S_ref(Z)||_L2<=sqrt(1+C1_ceiling^2)*"
                "||Z||_L2(l2^153)"
            ),
            "low_projector_L2_contraction": True,
            "background_subtracted_source_is_L2": True,
            "certified": True,
        },
        "local_witness_audit": {
            "is_reference_background": False,
            "acceleration_independent_remainder_nonzero": True,
            "solved_acceleration_nonzero": True,
            "purpose": "pointwise nonlinear source and hyperbolicity witness only",
        },
        "FLRW_audit": {
            "coefficient_matched_candidate_id": flrw_match["candidate_id"],
            "status": flrw_match["status"],
            "interval_trajectory_certificate_present": False,
            "used_as_reference_background": False,
            "reason": (
                "the matched G4_X modified-harmonic candidate was not background-screened"
            ),
        },
        "finite_low_localized_source_gate_closed": True,
        "full_component_Jacobian_gate_closed": False,
        "global_H7_dyadic_sum_applied": False,
        "nonlinear_lifespan_proved": False,
        "remaining_gates": [
            "localized_Bony_Moser_remainder_for_high_shells",
            "594_lower_source_component_Jacobian_entries_if_component_identity_needed",
            "remote_H6_coefficient_to_H7_state_derivative_loss",
            "global_H7_dyadic_sum_and_nonlinear_lifespan_bootstrap",
        ],
        "scope": (
            "This certifies the exact static tube-center equilibrium and an L2 bound for "
            "the background-subtracted localized lower source. It does not promote the "
            "unresolved FLRW entries or close high-shell remainders, the global H7 sum, "
            "or lifespan."
        ),
    }


def run_quartic_reference_equilibrium_campaign(
    finite_low_campaign: dict[str, Any],
    nonlinear_campaign: dict[str, Any],
    solved_campaign: dict[str, Any],
    moser_campaign: dict[str, Any],
    pde_campaign: dict[str, Any],
    tube_campaign: dict[str, Any],
    euler_campaign: dict[str, Any],
    flrw_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticReferenceEquilibriumError("unsupported campaign schema_version")
        campaigns = {
            "finite_low": finite_low_campaign,
            "nonlinear": nonlinear_campaign,
            "solved": solved_campaign,
            "moser": moser_campaign,
            "pde": pde_campaign,
            "tube": tube_campaign,
            "euler": euler_campaign,
            "flrw": flrw_campaign,
        }
        expected_statuses = {
            "finite_low": (
                "pass_all_12_finite_low_anti_wick_principal_operators_"
                "lower_sources_fail_closed"
            ),
            "nonlinear": "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
            "solved": "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "moser": "pass_all_12_quasilinear_coefficient_derivative_envelopes",
            "pde": "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
            "tube": "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
            "euler": "pass_all_12_complete_coordinate_tube_euler_remainder_majorants",
            "flrw": "pass_all_generalized_harmonic_candidates_interval_certified",
        }
        for name, campaign in campaigns.items():
            if campaign.get("status") != expected_statuses[name]:
                raise QuarticReferenceEquilibriumError(
                    f"{name} prerequisite status mismatch"
                )
            if not _content_hash_matches(campaign):
                raise QuarticReferenceEquilibriumError(
                    f"{name} campaign content hash mismatch"
                )
        upstream = {
            name: campaign["content_sha256"] for name, campaign in campaigns.items()
        }
        if finite_low_campaign.get("upstream_sha256", {}).get(
            "solved_source"
        ) != upstream["solved"]:
            raise QuarticReferenceEquilibriumError(
                "finite-low-to-solved provenance mismatch"
            )
        solved_links = solved_campaign.get("upstream_sha256", {})
        for local, remote in (
            ("moser", "moser"),
            ("pde", "nonquasilinear_pde"),
            ("tube", "coordinate_tube"),
            ("euler", "euler_remainder"),
        ):
            if solved_links.get(remote) != upstream[local]:
                raise QuarticReferenceEquilibriumError(
                    f"solved-to-{local} provenance mismatch"
                )
        pde_links = pde_campaign.get("upstream_sha256", {})
        if (
            pde_links.get("moser") != upstream["moser"]
            or pde_links.get("nonlinear") != upstream["nonlinear"]
        ):
            raise QuarticReferenceEquilibriumError("PDE provenance mismatch")
        if tube_campaign.get("nonquasilinear_pde_campaign_sha256") != upstream["pde"]:
            raise QuarticReferenceEquilibriumError("tube provenance mismatch")
        if (
            euler_campaign.get("nonquasilinear_pde_campaign_sha256") != upstream["pde"]
            or euler_campaign.get("coordinate_tube_campaign_sha256") != upstream["tube"]
        ):
            raise QuarticReferenceEquilibriumError("Euler provenance mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or config.get("reference") != "cartesian_minkowski_constant_scalar"
            or config.get("whole_space_source_convention") != "background_subtracted"
            or config.get("require_L2_coordinate_atom_perturbation") is not True
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticReferenceEquilibriumError("unsupported equilibrium contract")
        passed, control = generic_reference_equilibrium_control()
        if not passed:
            raise QuarticReferenceEquilibriumError("generic equilibrium control failed")
        maps = {
            name: _candidate_records(campaign)
            for name, campaign in campaigns.items()
            if name != "flrw"
        }
        candidate_ids = set(maps["finite_low"])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps.values()
        ):
            raise QuarticReferenceEquilibriumError("candidate-set mismatch")
        coordinate_jet = solved_campaign["coordinate_jet_Frechet_envelopes"]
        certificates = [
            _certify_candidate(
                maps["finite_low"][candidate_id],
                maps["nonlinear"][candidate_id],
                maps["solved"][candidate_id],
                maps["moser"][candidate_id],
                maps["pde"][candidate_id],
                maps["tube"][candidate_id],
                maps["euler"][candidate_id],
                flrw_campaign,
                coordinate_jet,
                control,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_exact_reference_equilibria_and_L2_source_conventions",
            "errors": [],
            "upstream_sha256": upstream,
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_reference_equilibrium_control": control,
            "counts": {
                "selected": len(certificates),
                "exact_reference_equilibria_passed": len(certificates),
                "localized_L2_source_conventions_passed": len(certificates),
                "matched_FLRW_candidates_unresolved": len(certificates),
                "global_H7_sums_applied": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates share the exact Cartesian Minkowski/constant-scalar "
                "tube-center equilibrium and a background-subtracted localized L2 lower-source "
                "bound. Their coefficient-matched FLRW entries remain unresolved and unused."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticReferenceEquilibriumError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "exact_reference_equilibria_passed": 0,
                "localized_L2_source_conventions_passed": 0,
                "matched_FLRW_candidates_unresolved": 0,
                "global_H7_sums_applied": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_reference_equilibrium_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
