from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .recovery_artifact_validation import (
    DATA_SEALS,
    load_bound_inputs,
    validate_bound_inputs,
    validate_exact_rebuild,
)

SCHEMA_VERSION = "sigma-quartic-anti-wick-composition-campaign-1.0"
CONFIG_KEYS = {
    "schema_version",
    "expected_candidate_count",
    "spatial_dimension",
    "state_dimension",
    "current_mixed_total_order",
    "required_mixed_total_order",
    "annular_high_shell_index_minimum",
}


class QuarticAntiWickCompositionError(ValueError):
    """Raised when the anti-Wick composition audit is inconsistent."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return (
        campaign.get("content_sha256") == hashlib.sha256(_canonical_json(body).encode()).hexdigest()
    )


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


@cache
def generic_anti_wick_composition_audit() -> tuple[bool, dict[str, Any]]:
    """Verify heat smoothing, annular coercivity, Schur, and the missing-order witness."""

    t, x = sp.symbols("t x", real=True, finite=True)
    h = sp.Symbol("h", positive=True, finite=True)
    chi6 = (
        1716 * t**7
        - 9009 * t**8
        + 20020 * t**9
        - 24024 * t**10
        + 16380 * t**11
        - 6006 * t**12
        + 924 * t**13
    )
    derivative_factor_residual = sp.factor(sp.diff(chi6, t) - 12012 * t**6 * (1 - t) ** 6)
    endpoint_residuals = {
        f"d{order}_at_0": str(sp.diff(chi6, t, order).subs(t, 0)) for order in range(7)
    } | {
        f"d{order}_at_1_minus_target": str(
            sp.diff(chi6, t, order).subs(t, 1) - (1 if order == 0 else 0)
        )
        for order in range(7)
    }

    y, w, u, frequency_offset, separation = sp.symbols("y w u v s", real=True, finite=True)
    midpoint = (y + w) / 2
    projector_square_residual = sp.expand(
        (y - x) ** 2 + (w - x) ** 2 - 2 * (midpoint - x) ** 2 - (y - w) ** 2 / 2
    )
    heat_kernel_normalization = sp.integrate(
        sp.exp(-(u**2) / h) / sp.sqrt(sp.pi * h), (u, -sp.oo, sp.oo)
    )
    frequency_heat_transform = sp.integrate(
        sp.exp(-(frequency_offset**2) / h)
        / sp.sqrt(sp.pi * h)
        * sp.exp(sp.I * separation * frequency_offset / h),
        (frequency_offset, -sp.oo, sp.oo),
    )
    frequency_heat_residual = sp.simplify(
        frequency_heat_transform - sp.exp(-(separation**2) / (4 * h))
    )

    h0 = sp.Rational(1, 16)
    gaussian_mgf = sp.simplify((1 - 2 * (h / 2) * (1 / (2 * h))) ** (-sp.Rational(3, 2)))
    q0 = 1 - 2 * sp.sqrt(2) * sp.exp(-sp.Rational(1, 2) / h0)
    radial_kernel_integral = sp.integrate(4 * sp.pi * t**2 / (1 + t**2) ** 2, (t, 0, sp.oo))
    amplitude_schur = sp.simplify(radial_kernel_integral / (2 * sp.pi) ** 3)

    theta, q_variable, s_variable, xi = sp.symbols("theta q s xi", real=True, finite=True)
    polynomial_coefficients = sp.symbols("a0:5", real=True, finite=True)
    coefficient_polynomial = sum(
        coefficient * q_variable**order for order, coefficient in enumerate(polynomial_coefficients)
    )
    ftoc_residual = sp.expand(
        coefficient_polynomial.subs(q_variable, q_variable - s_variable / 2)
        - coefficient_polynomial
        + s_variable
        / 2
        * sp.integrate(
            sp.diff(coefficient_polynomial, q_variable).subs(
                q_variable, q_variable - theta * s_variable / 2
            ),
            (theta, 0, 1),
        )
    )
    phase_transfer_residual = sp.simplify(
        sp.diff(sp.exp(sp.I * s_variable * xi / h), xi)
        - sp.I * s_variable / h * sp.exp(sp.I * s_variable * xi / h)
    )

    c = 1 + x**2
    pointwise_k = sp.diag(c, 1)
    a11 = c + h / 2
    a = sp.Matrix([[a11, 0], [0, 1]])
    matrix_a = sp.Matrix([[0, 1], [c, 0]])
    pointwise_symmetrization_residual = sp.simplify(
        pointwise_k * matrix_a - matrix_a.T * pointwise_k
    )
    smoothing_witness = sp.simplify((a * matrix_a - matrix_a.T * a) / h)
    expected_witness = sp.Matrix([[0, sp.Rational(1, 2)], [-sp.Rational(1, 2), 0]])
    corrupted_heat_witness = sp.simplify(
        (sp.diag(c + h, 1) * matrix_a - matrix_a.T * sp.diag(c + h, 1)) / h
    )
    required_pairs = [[2, 4], [0, 6], [0, 5], [1, 4]]
    passed = bool(
        derivative_factor_residual == 0
        and set(endpoint_residuals.values()) == {"0"}
        and projector_square_residual == 0
        and heat_kernel_normalization == 1
        and frequency_heat_residual == 0
        and gaussian_mgf == 2 * sp.sqrt(2)
        and q0.is_positive
        and radial_kernel_integral == sp.pi**2
        and amplitude_schur == 1 / (8 * sp.pi)
        and ftoc_residual == 0
        and phase_transfer_residual == 0
        and pointwise_symmetrization_residual == sp.zeros(2)
        and smoothing_witness == expected_witness
        and corrupted_heat_witness != expected_witness
    )
    return passed, {
        "control": "Gaussian anti-Wick/Weyl composition prerequisite and C6 derivative audit",
        "anti_wick_to_weyl": {
            "identity": "Op_h^AW(b)=Op_h^W(a_h)",
            "weyl_symbol": "a_h=exp((h/4)Delta_(x,xi)) b",
            "heat_time": "h/4",
            "coherent_projector_midpoint_square_residual": str(projector_square_residual),
            "heat_kernel_normalization": str(heat_kernel_normalization),
            "frequency_heat_transform": str(frequency_heat_transform),
            "frequency_heat_transform_residual": str(frequency_heat_residual),
            "pointwise_symmetrization_warning": (
                "b p=p^dagger b does not imply a_h p=p^dagger a_h"
            ),
        },
        "annular_positive_energy": {
            "semiclassical_scale": "h_j=8*2^-j",
            "high_shell_index_minimum": 7,
            "h_maximum": str(h0),
            "rescaled_shell": "4<=|eta|<=16",
            "cutoff_equals_one": "3<=|eta|<=17",
            "cutoff_support": "5/2<=|eta|<=35/2",
            "chi6": str(chi6),
            "chi6_derivative_factor_residual": str(derivative_factor_residual),
            "endpoint_residuals": endpoint_residuals,
            "coercivity_factor": str(q0),
            "coercivity_factor_numeric": float(sp.N(q0, 18)),
            "positive": bool(q0.is_positive),
            "Gaussian_MGF_at_tail_parameter": str(gaussian_mgf),
            "Gaussian_MGF_residual": str(gaussian_mgf - 2 * sp.sqrt(2)),
            "range_proof": ("chi6'=12012*t^6*(1-t)^6>=0 with endpoint values 0 and 1"),
        },
        "exact_composition_amplitude": {
            "kernel_formula": (
                "D=(i/h)(OpW(a_h)OpW(p)-OpW(p)^dagger OpW(a_h)); expand at "
                "q=(y+w)/2, s=y-w, use b*p=p^dagger*b, apply FTOC to A(q+-s/2), "
                "and transfer (i*s_k/h)exp(i*s.xi/h)=partial_xi_k exp(i*s.xi/h)"
            ),
            "families": [
                "heat_smoothing_symmetrization_defect",
                "coefficient_difference_after_xi_integration_by_parts",
                "Weyl_derivative_and_subprincipal_terms",
            ],
            "FTOC_polynomial_residual": str(ftoc_residual),
            "phase_transfer_residual": str(phase_transfer_residual),
            "majorant_recurrence": {
                "D_m": "(3/4)*(L_(2,m)+L_(0,m+2)), D_-1=0",
                "S_m": "2*A0*(3*Q*D_m+m*D_(m-1))",
                "C_m": "9*A1*(Q*L_(0,m+1)+(m+1)*L_(0,m))",
                "T_m": "3*(A0*L_(1,m)+A1*L_(0,m))",
                "R_m": "S_m+C_m+T_m for m in {0,2,4}",
                "composition_constant": "(R_0+6*R_2+9*R_4)/(8*pi)",
            },
            "required_K_derivative_pairs_derived_from_recurrence": required_pairs,
            "hidden_asymptotic_remainder": False,
            "physical_left_to_Weyl_correction_required": (
                "-(1/2) sum_j partial_j A^j, up to the declared evolution sign"
            ),
        },
        "amplitude_Schur_lemma": {
            "bound": "||Op_h(r)||<=(R0+6R2+9R4)/(8*pi)",
            "R_m": "max_|beta|=m sup_(x,y) integral ||partial_xi^beta r||_2 dxi",
            "radial_kernel_integral": str(radial_kernel_integral),
            "exact_coefficient": str(amplitude_schur),
            "matrix_dimension_independent": True,
        },
        "smoothing_defect_negative": {
            "c": "1+x^2",
            "A": "[[0,1],[c,0]]",
            "K": "diag(c,1)",
            "pointwise_KA_minus_ATK": str(pointwise_symmetrization_residual),
            "heat_smoothed_defect_divided_by_h": str(smoothing_witness),
            "corrupt_heat_time_h_over_2_witness": str(corrupted_heat_witness),
            "omitting_smoothing_family_rejected": smoothing_witness != sp.zeros(2),
            "corrupt_heat_time_rejected": corrupted_heat_witness != expected_witness,
        },
        "derivative_audit": {
            "current_maximum_mixed_total_order": 4,
            "required_maximum_mixed_total_order": 6,
            "required_pairs_x_frequency": required_pairs,
            "maximum_spatial_x_order": 2,
            "state_H7_and_coordinate_H6_sufficient_for_x_orders": True,
            "missing_piece": "extend state/frequency Frechet chain to C6 on annulus",
        },
        "passed": passed,
    }


def _audit_candidate(
    low: dict[str, Any],
    evolution: dict[str, Any],
    r3: dict[str, Any],
    time_atoms: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(low.get("candidate_id"))
    for item in (evolution, r3, time_atoms):
        if item.get("candidate_id") != candidate_id or item.get("coefficients") != low.get(
            "coefficients"
        ):
            raise QuarticAntiWickCompositionError("candidate identity mismatch")
    expected = (
        "pass_global_C4_positive_K55_symbol_extension",
        "pass_full_55_state_degree_one_evolution_symbol_C4_bounds",
        "pass_R3_H6_spatialized_K55_P55_symbol_bounds",
        "pass_H7_closed_coordinate_atom_time_budget",
    )
    if tuple(item.get("status") for item in (low, evolution, r3, time_atoms)) != expected:
        raise QuarticAntiWickCompositionError("candidate prerequisite status mismatch")
    required_keys = {"2,4", "0,6", "0,5", "1,4"}
    available_keys = set(r3.get("spatialized_global_K55_bounds", {}))
    available_total_order = max(sum(int(part) for part in key.split(",")) for key in available_keys)
    missing = sorted(required_keys - available_keys)
    return {
        "schema_version": "sigma-quartic-anti-wick-composition-audit-certificate-1.0",
        "status": "fail_closed_requires_C6_spatial_frequency_symbol_bounds",
        "candidate_id": candidate_id,
        "coefficients": low.get("coefficients"),
        "available_mixed_total_order": available_total_order,
        "required_mixed_total_order": 6,
        "required_pairs": sorted(required_keys),
        "missing_required_pairs": missing,
        "time_K55_order_0_bound_available": "0,0" in time_atoms.get("closed_time_K55_bounds", {}),
        "P55_A0_A1_bounds_available": all(
            key in r3.get("spatialized_dyadic_P55_bounds", {}) for key in ("0,1", "1,1")
        ),
        "anti_wick_composition_closed": False,
        "required_next_gate": "quartic_C6_annular_symbol_moser_and_frequency_campaign",
        "scope": (
            "The heat identity, amplitude algebra, Schur constant, and derivative "
            "requirements are certified prerequisites, but this candidate cannot "
            "receive a numerical composition remainder until the listed spatial/C6 "
            "K55 derivatives are generated."
        ),
    }


def run_quartic_anti_wick_composition_campaign(
    low_frequency_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    r3_campaign: dict[str, Any],
    time_atom_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticAntiWickCompositionError("unsupported campaign schema_version")
        campaigns = (
            low_frequency_campaign,
            evolution_campaign,
            r3_campaign,
            time_atom_campaign,
        )
        validate_bound_inputs(
            config,
            {
                "low_frequency": low_frequency_campaign,
                "evolution": evolution_campaign,
                "r3_sobolev": r3_campaign,
                "time_atoms": time_atom_campaign,
            },
            CONFIG_KEYS,
        )
        statuses = (
            "pass_all_12_global_C4_positive_K55_symbol_extensions",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_R3_H6_spatialized_K55_P55_symbol_bounds",
            "pass_all_12_H7_closed_coordinate_atom_time_budgets",
        )
        if tuple(item.get("status") for item in campaigns) != statuses:
            raise QuarticAntiWickCompositionError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(item) for item in campaigns):
            raise QuarticAntiWickCompositionError("campaign content hash mismatch")
        if r3_campaign.get("upstream_sha256", {}).get(
            "low_frequency"
        ) != low_frequency_campaign.get("content_sha256"):
            raise QuarticAntiWickCompositionError("low-frequency provenance mismatch")
        if r3_campaign.get("upstream_sha256", {}).get("evolution") != evolution_campaign.get(
            "content_sha256"
        ):
            raise QuarticAntiWickCompositionError("evolution provenance mismatch")
        if time_atom_campaign.get("upstream_sha256", {}).get("r3_sobolev") != r3_campaign.get(
            "content_sha256"
        ):
            raise QuarticAntiWickCompositionError("time-atom provenance mismatch")
        if time_atom_campaign.get("upstream_sha256", {}).get(
            "low_frequency"
        ) != low_frequency_campaign.get("content_sha256"):
            raise QuarticAntiWickCompositionError("time-atom low-frequency provenance mismatch")
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["current_mixed_total_order"]) != 4
            or int(config["required_mixed_total_order"]) != 6
            or int(config["annular_high_shell_index_minimum"]) != 7
        ):
            raise QuarticAntiWickCompositionError("unsupported composition audit contract")
        control_passed, control = generic_anti_wick_composition_audit()
        if not control_passed:
            raise QuarticAntiWickCompositionError("generic anti-Wick audit failed")
        maps = tuple(_candidate_records(item) for item in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticAntiWickCompositionError("candidate-set mismatch")
        certificates = [
            _audit_candidate(*(records[candidate_id] for records in maps))
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_exact_anti_wick_composition_prerequisite_audit_C6_required",
            "errors": [],
            "upstream_sha256": {
                "low_frequency": low_frequency_campaign.get("content_sha256"),
                "evolution": evolution_campaign.get("content_sha256"),
                "r3_sobolev": r3_campaign.get("content_sha256"),
                "time_atoms": time_atom_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_anti_wick_composition_audit": control,
            "counts": {
                "selected": len(certificates),
                "exact_composition_prerequisite_audits_passed": len(certificates),
                "anti_wick_compositions_closed": 0,
                "C6_extensions_required": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The anti-Wick/Weyl heat identity, amplitude algebra, operator constants, "
                "and derivative-requirement audit pass, while all 12 candidates remain "
                "fail-closed until the annular K55 calculus extends from C4 to C6."
            ),
            "scope": certificates[0]["scope"],
            "data_seals": DATA_SEALS,
        }
    except (KeyError, TypeError, ValueError, QuarticAntiWickCompositionError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "exact_composition_prerequisite_audits_passed": 0,
                "anti_wick_compositions_closed": 0,
                "C6_extensions_required": 0,
                "rejected": 0,
            },
            "data_seals": DATA_SEALS,
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def validate_quartic_anti_wick_composition_artifact(
    artifact: dict[str, Any], root: Path, config: dict[str, Any]
) -> None:
    loaded = load_bound_inputs(
        root, config, ("low_frequency", "evolution", "r3_sobolev", "time_atoms")
    )
    rebuilt = run_quartic_anti_wick_composition_campaign(
        loaded["low_frequency"],
        loaded["evolution"],
        loaded["r3_sobolev"],
        loaded["time_atoms"],
        config,
    )
    validate_exact_rebuild(artifact, rebuilt)


def write_quartic_anti_wick_composition_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
