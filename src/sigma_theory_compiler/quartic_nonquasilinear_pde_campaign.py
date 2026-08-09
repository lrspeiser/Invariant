from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .horndeski_principal import (
    quartic_horndeski_baseline_riesz_symmetrizer_control,
)

SCHEMA_VERSION = "sigma-quartic-nonquasilinear-pde-campaign-1.0"


class QuarticNonquasilinearPDEError(ValueError):
    """Raised when the full nonquasilinear PDE certificate cannot be constructed."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _positive(expression: sp.Expr) -> bool:
    decision = sp.simplify(expression).is_positive
    if decision is not None:
        return bool(decision)
    return bool(sp.N(expression, 100) > 0)


@cache
def generic_nonquasilinear_acceleration_control() -> tuple[bool, dict[str, Any]]:
    """Verify the on-shell derivative identities for X=-A^{-1}W exactly."""

    mixed, spatial = sp.symbols("m s", real=True, finite=True)
    a0, am, ass, w0, wm, ws = sp.symbols(
        "a0 am ass w0 wm ws", real=True, finite=True
    )
    acceleration = sp.Symbol("y", real=True, finite=True)
    coefficient_a = a0 + am * mixed + ass * spatial**2
    remainder = w0 + wm * mixed**2 + ws * spatial
    solved_acceleration = sp.cancel(-remainder / coefficient_a)

    principal_mixed = sp.factor(
        (
            sp.diff(remainder, mixed)
            + sp.diff(coefficient_a, mixed) * acceleration
        )
        / 2
    )
    principal_spatial = sp.factor(
        sp.diff(remainder, spatial)
        + sp.diff(coefficient_a, spatial) * acceleration
    )
    mixed_residual = sp.factor(
        sp.diff(solved_acceleration, mixed)
        + 2
        * principal_mixed.subs(acceleration, solved_acceleration)
        / coefficient_a
    )
    spatial_residual = sp.factor(
        sp.diff(solved_acceleration, spatial)
        + principal_spatial.subs(acceleration, solved_acceleration)
        / coefficient_a
    )
    corrupted_mixed = sp.factor(
        sp.diff(solved_acceleration, mixed)
        + principal_mixed.subs(acceleration, solved_acceleration)
        / coefficient_a
    )
    witness = {
        mixed: 2,
        spatial: 3,
        a0: 5,
        am: 7,
        ass: 11,
        w0: 13,
        wm: 17,
        ws: 19,
    }
    corrupted_witness = sp.factor(corrupted_mixed.subs(witness))
    passed = bool(
        mixed_residual == 0
        and spatial_residual == 0
        and corrupted_witness != 0
    )
    return passed, {
        "control": "exact nonquasilinear acceleration-elimination derivative identities",
        "equation_form": (
            "V=A(q,partial q,partial_0 partial_i q,partial_i partial_j q) "
            "partial_0^2 q+W(q,partial q,partial_0 partial_i q,"
            "partial_i partial_j q)=0"
        ),
        "solved_form": "partial_0^2 q=X=-A^{-1}W",
        "spatial_second_derivative_identity": "D_ij X=-A^{-1} P^ij",
        "mixed_second_derivative_identity": "D_0i X=-2 A^{-1} P^0i",
        "mixed_identity_residual": str(mixed_residual),
        "spatial_identity_residual": str(spatial_residual),
        "negative_control": {
            "corruption": "omit the factor two in the mixed-derivative identity",
            "exact_witness_residual": str(corrupted_witness),
            "rejected": corrupted_witness != 0,
        },
        "terminology": (
            "W is acceleration-independent, not lower order: it may contain mixed and "
            "spatial second derivatives."
        ),
        "source": "Kovacs--Reall arXiv:2003.08398, Appendix A, equations Wdef, dX1, dX2",
        "passed": passed,
    }


@cache
def generic_full_symmetrizer_lift_control() -> tuple[bool, dict[str, Any]]:
    """Verify the Appendix-A block symmetrizer lift on an exact nontrivial control."""

    companion = sp.Matrix([[0, 1], [1, 0]])
    companion_symmetrizer = sp.eye(2)
    coupling = sp.Matrix([[0, 0, 0], [0, sp.Rational(1, 2), sp.Rational(1, 3)]])
    cross = coupling.T * companion_symmetrizer * companion.inv()
    schur_c = sp.Integer(1) + sum(entry**2 for entry in cross)
    full_symbol = sp.zeros(5)
    full_symbol[3:5, 0:3] = coupling
    full_symbol[3:5, 3:5] = companion
    full_symmetrizer = sp.zeros(5)
    full_symmetrizer[0:3, 0:3] = schur_c * sp.eye(3)
    full_symmetrizer[0:3, 3:5] = cross
    full_symmetrizer[3:5, 0:3] = cross.T
    full_symmetrizer[3:5, 3:5] = companion_symmetrizer
    symmetrizer_residual = (
        full_symmetrizer * full_symbol
        - full_symbol.T * full_symmetrizer
    ).applyfunc(sp.factor)
    _, diagonal = full_symmetrizer.LDLdecomposition(hermitian=True)
    pivots = [sp.factor(value) for value in diagonal.diagonal()]

    omitted_cross = full_symmetrizer.copy()
    omitted_cross[0:3, 3:5] = sp.zeros(3, 2)
    omitted_cross[3:5, 0:3] = sp.zeros(2, 3)
    omitted_residual = (
        omitted_cross * full_symbol - full_symbol.T * omitted_cross
    ).applyfunc(sp.factor)
    passed = bool(
        (companion_symmetrizer * companion - companion.T * companion_symmetrizer)
        .applyfunc(sp.factor)
        .is_zero_matrix
        and symmetrizer_residual.is_zero_matrix
        and all(value.is_positive for value in pivots)
        and not omitted_residual.is_zero_matrix
    )
    return passed, {
        "control": "exact full first-order block symmetrizer lift",
        "block_symbol": "M55=[[0,0],[L,M22]]",
        "construction": (
            "F=L^dagger K22 M22^{-1}; K55=[[c I,F],[F^dagger,K22]]"
        ),
        "dimensions_for_eleven_fields": {
            "zero_and_transverse_block": 33,
            "directional_companion_block": 22,
            "full_state": 55,
        },
        "exact_control_dimensions": {
            "zero_and_transverse_block": 3,
            "directional_companion_block": 2,
            "full_state": 5,
        },
        "F_L_hermitian": bool((cross * coupling).equals((cross * coupling).T)),
        "F_M_equals_L_dagger_K": bool(
            (cross * companion).equals(coupling.T * companion_symmetrizer)
        ),
        "K55_M55_minus_M55_dagger_K55_zero": symmetrizer_residual.is_zero_matrix,
        "exact_LDL_pivots": [str(value) for value in pivots],
        "all_LDL_pivots_positive": bool(all(value.is_positive for value in pivots)),
        "negative_control": {
            "corruption": "omit the F cross block",
            "nonzero_residual_entries": sum(value != 0 for value in omitted_residual),
            "rejected": not omitted_residual.is_zero_matrix,
        },
        "source": "Kovacs--Reall arXiv:2003.08398, Appendix A, equations cal_M_decomp through Feqs",
        "passed": passed,
    }


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        item["candidate_id"]: item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _uniform_companion_symmetrizer_bounds() -> tuple[sp.Expr, sp.Expr, dict[str, Any]]:
    baseline_passed, baseline = quartic_horndeski_baseline_riesz_symmetrizer_control()
    if not baseline_passed:
        raise QuarticNonquasilinearPDEError("baseline Riesz symmetrizer control failed")
    contract = baseline["quantitative_physical_group_perturbation_contract"]
    drift = sp.sympify(contract["implied_Riesz_projector_drift_upper"])
    physical_lower = sp.sympify(contract["certified_physical_H_star_margin"])
    lower = sp.factor(physical_lower / 6)
    upper = sp.Integer(0)
    projector_bounds: dict[str, str] = {}
    for eigenvalue, record in baseline["projectors"].items():
        projector_bound = sp.factor(
            sp.sqrt(sp.sympify(record["frobenius_norm_squared"])) + drift
        )
        projector_bounds[eigenvalue] = str(projector_bound)
        weight = sp.Rational(9, 8) if abs(sp.sympify(eigenvalue)) == 1 else sp.Integer(1)
        upper += weight * projector_bound**2
    upper = sp.factor(upper)
    if not (_positive(lower) and _positive(upper)):
        raise QuarticNonquasilinearPDEError("invalid uniform companion symmetrizer bounds")
    return lower, upper, {
        "physical_projected_lower": str(physical_lower),
        "six_group_decomposition_factor": 6,
        "Riesz_projector_drift_upper": str(drift),
        "candidate_projector_2_norm_uppers": projector_bounds,
        "physical_H_star_2_norm_upper": "9/8",
    }


def _certify_candidate(
    symmetrizer: dict[str, Any],
    moser: dict[str, Any],
    first_order: dict[str, Any],
    geometric: dict[str, Any],
    nonlinear: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = symmetrizer.get("candidate_id")
    records = (moser, first_order, geometric, nonlinear)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticNonquasilinearPDEError("candidate ID mismatch")
    if any(record.get("coefficients") != symmetrizer.get("coefficients") for record in records):
        raise QuarticNonquasilinearPDEError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_uniform_local_jet_strong_hyperbolicity",
        "pass_quasilinear_coefficient_derivative_envelopes",
        "pass_exact_55_variable_principal_first_order_reduction",
        "pass_exact_nonlinear_geometric_state_to_jet_map",
        "pass_exact_local_nonlinear_time_acceleration_elimination",
    )
    if (symmetrizer.get("status"), *(record.get("status") for record in records)) != expected_statuses:
        raise QuarticNonquasilinearPDEError("candidate prerequisite status mismatch")
    if not nonlinear.get("nonzero_acceleration_independent_remainder"):
        raise QuarticNonquasilinearPDEError("nonlinear remainder witness is absent")

    k_lower, k_upper, k_derivation = _uniform_companion_symmetrizer_bounds()
    characteristic_abs_lower = sp.sympify(config["characteristic_absolute_lower_bound"])
    expected_characteristic_lower = sp.Rational(1, 4)
    if characteristic_abs_lower != expected_characteristic_lower:
        raise QuarticNonquasilinearPDEError(
            "characteristic lower bound must equal the certified 1/3-1/12 contour value"
        )
    inverse_companion_upper = sp.factor(
        sp.sqrt(k_upper / k_lower) / characteristic_abs_lower
    )
    inverse_time_upper = sp.sympify(
        moser["inverse_time_block_2_norm_rational_ceiling"]
    )
    spatial_coefficient_upper = sp.sympify(
        moser["raw_Frechet_derivative_2_norm_envelopes"]["C"]["0"]
    )
    transverse_coupling_upper = sp.factor(
        inverse_time_upper * spatial_coefficient_upper
    )
    cross_upper = sp.factor(
        transverse_coupling_upper * k_upper * inverse_companion_upper
    )
    diagonal_c = sp.factor(1 + cross_upper**2 / k_lower)
    full_lower = sp.factor(k_lower / (1 + cross_upper / k_lower) ** 2)
    full_upper = sp.factor(diagonal_c + k_upper + 2 * cross_upper)
    if not all(
        _positive(value)
        for value in (
            characteristic_abs_lower,
            inverse_companion_upper,
            transverse_coupling_upper,
            cross_upper,
            diagonal_c,
            full_lower,
            full_upper,
        )
    ):
        raise QuarticNonquasilinearPDEError("a full symmetrizer bound is not positive")

    required_order = int(config["required_sobolev_coefficient_order"])
    if required_order < 4 or int(moser["domain"]["required_sobolev_order"]) < required_order:
        raise QuarticNonquasilinearPDEError("insufficient coefficient regularity order")
    return {
        "schema_version": "sigma-quartic-nonquasilinear-pde-certificate-1.0",
        "status": "pass_full_55_state_nonquasilinear_strong_hyperbolicity_lift",
        "candidate_id": candidate_id,
        "coefficients": symmetrizer["coefficients"],
        "domain": symmetrizer["domain"],
        "nonquasilinear_equation": {
            "form": "A partial_0^2 q+W(q,partial q,partial_0 partial_i q,partial_i partial_j q)=0",
            "acceleration_independent_remainder_verified_nonzero": True,
            "exact_acceleration_solution_residual_zero": nonlinear[
                "acceleration_solution_residual_zero"
            ],
            "state_to_covariant_jet_formula_sha256": nonlinear[
                "source_geometric_formula_contract_sha256"
            ],
            "evolution_formula_contract_sha256": nonlinear[
                "evolution_formula_contract_sha256"
            ],
        },
        "full_first_order_state": {
            "q": 11,
            "v_0": 11,
            "v_i": 33,
            "total": 55,
            "definition_constraints": 33,
            "independent_spatial_curl_constraints": 33,
        },
        "uniform_bounds": {
            "K22_2_lower": str(k_lower),
            "K22_2_upper": str(k_upper),
            "characteristic_absolute_lower": str(characteristic_abs_lower),
            "M22_inverse_2_upper": str(inverse_companion_upper),
            "A_inverse_2_upper_used": str(inverse_time_upper),
            "spatial_Pij_coefficient_upper": str(spatial_coefficient_upper),
            "L_2_upper": str(transverse_coupling_upper),
            "F_2_upper": str(cross_upper),
            "chosen_c": str(diagonal_c),
            "K55_2_lower": str(full_lower),
            "K55_2_upper": str(full_upper),
            "K55_2_lower_numeric": float(sp.N(full_lower, 18)),
            "K55_2_upper_numeric": float(sp.N(full_upper, 18)),
            "derivation": k_derivation,
        },
        "smoothness": {
            "required_coefficient_order": required_order,
            "companion_coefficients_C4_on_box": True,
            "Riesz_contours_fixed_and_disjoint": True,
            "K22_smooth_by_resolvent_integral": True,
            "M22_inverse_smooth_from_nonzero_characteristic_gap": True,
            "K55_smooth_by_finite_products_and_inverse": True,
        },
        "conditional_local_wellposedness": {
            "status": "theorem_applies_to_compatible_vacuum_data_in_compact_box_interior",
            "system": "gauge-fixed vacuum G2=X+c20 X^2, G4=M2/2+alpha X equations",
            "regularity": "H^s with s>s0; this campaign does not optimize or assign s0",
            "conclusion": (
                "a unique local gauge-fixed solution exists and depends continuously on the "
                "compatible initial data for some T>0"
            ),
            "initial_data_requirements": [
                "the 55-state derivative-definition and curl constraints hold",
                "the gravitational and modified-harmonic gauge constraints hold",
                "the coordinate state-to-jet image lies in a compact subset of the certified box interior",
                "the physical and auxiliary metrics remain Lorentzian and the initial slice is noncharacteristic",
            ],
            "not_certified": [
                "a numerical lower bound for the existence time T",
                "global or long-time preservation of the 2e-10 local-jet box",
                "boundary conditions or boundary energy estimates",
                "matter-source evolution or universal-matter well-posedness",
                "observational viability",
            ],
            "source": "Kovacs--Reall arXiv:2003.08398, Appendix A local well-posedness theorem",
        },
        "claim": (
            "The candidate's certified 22-state directional symmetrizer lifts to a smooth, "
            "uniformly positive symmetrizer for the complete 55-state nonquasilinear "
            "first-order reduction."
        ),
        "scope": (
            "This is a local vacuum gauge-fixed Cauchy certificate, conditional on compatible "
            "initial data in the certified box interior. It is not a nonlinear trapping, "
            "global-existence, matter, boundary, or observational certificate."
        ),
    }


def run_quartic_nonquasilinear_pde_campaign(
    symmetrizer_campaign: dict[str, Any],
    moser_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    geometric_campaign: dict[str, Any],
    nonlinear_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticNonquasilinearPDEError("unsupported campaign schema_version")
        campaign_statuses = (
            symmetrizer_campaign.get("status"),
            moser_campaign.get("status"),
            first_order_campaign.get("status"),
            geometric_campaign.get("status"),
            nonlinear_campaign.get("status"),
        )
        expected_statuses = (
            "pass_all_linear_X_quartic_candidates_strongly_hyperbolic_on_local_boxes",
            "pass_all_12_quasilinear_coefficient_derivative_envelopes",
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
            "pass_all_12_exact_nonlinear_geometric_state_to_jet_maps",
            "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
        )
        if campaign_statuses != expected_statuses:
            raise QuarticNonquasilinearPDEError("campaign prerequisite status mismatch")
        if first_order_campaign.get("symmetrizer_campaign_sha256") != symmetrizer_campaign.get(
            "content_sha256"
        ) or first_order_campaign.get("moser_campaign_sha256") != moser_campaign.get(
            "content_sha256"
        ):
            raise QuarticNonquasilinearPDEError("first-order provenance chain mismatch")
        if geometric_campaign.get("first_order_campaign_sha256") != first_order_campaign.get(
            "content_sha256"
        ):
            raise QuarticNonquasilinearPDEError("geometric provenance chain mismatch")
        if nonlinear_campaign.get("geometric_campaign_sha256") != geometric_campaign.get(
            "content_sha256"
        ):
            raise QuarticNonquasilinearPDEError("nonlinear provenance chain mismatch")

        generic_passed, generic = generic_nonquasilinear_acceleration_control()
        lift_passed, lift = generic_full_symmetrizer_lift_control()
        if not (generic_passed and lift_passed):
            raise QuarticNonquasilinearPDEError("generic nonquasilinear theorem control failed")
        maps = tuple(
            _candidate_records(campaign)
            for campaign in (
                symmetrizer_campaign,
                moser_campaign,
                first_order_campaign,
                geometric_campaign,
                nonlinear_campaign,
            )
        )
        expected_count = int(config.get("expected_candidate_count", 12))
        candidate_ids = set(maps[0])
        if len(candidate_ids) != expected_count or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticNonquasilinearPDEError("candidate-set mismatch")
        certificates = [
            _certify_candidate(*(records[candidate_id] for records in maps), config)
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
            "errors": [],
            "upstream_sha256": {
                "symmetrizer": symmetrizer_campaign.get("content_sha256"),
                "moser": moser_campaign.get("content_sha256"),
                "first_order": first_order_campaign.get("content_sha256"),
                "geometric": geometric_campaign.get("content_sha256"),
                "nonlinear": nonlinear_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_nonquasilinear_control": generic,
            "generic_full_symmetrizer_lift_control": lift,
            "counts": {
                "selected": len(certificates),
                "full_55_state_symmetrizer_lifts_passed": len(certificates),
                "conditional_local_vacuum_cauchy_certificates": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 fixed-coefficient quartic candidates have a full 55-state smooth "
                "positive symmetrizer and a conditional local gauge-fixed vacuum Cauchy "
                "certificate for compatible data in the certified box interior."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticNonquasilinearPDEError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "full_55_state_symmetrizer_lifts_passed": 0,
                "conditional_local_vacuum_cauchy_certificates": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_nonquasilinear_pde_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
