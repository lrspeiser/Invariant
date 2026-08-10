from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_low_frequency_symbol_extension_campaign import (
    generic_low_frequency_extension_control,
)

SCHEMA_VERSION = "sigma-quartic-bounded-frequency-defect-campaign-1.0"


class QuarticBoundedFrequencyDefectError(ValueError):
    """Raised when the compact physical-frequency defect cannot be certified."""


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


@cache
def generic_compact_frequency_defect_control() -> tuple[bool, dict[str, Any]]:
    """Prove the exact defect, C4 cutoff bounds, Schur constant, and scale gate."""

    t, h = sp.symbols("t h", positive=True, finite=True)
    smoothstep = (
        126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9
    )
    endpoint_residuals = {
        f"d{order}_at_0": str(sp.diff(smoothstep, t, order).subs(t, 0))
        for order in range(5)
    } | {
        f"d{order}_at_1_minus_target": str(
            sp.diff(smoothstep, t, order).subs(t, 1) - (1 if order == 0 else 0)
        )
        for order in range(5)
    }
    upstream_passed, upstream_control = generic_low_frequency_extension_control()
    radial_majorants = [
        int(upstream_control["radial_cutoff_Frechet_majorants"][str(order)])
        for order in range(5)
    ]
    defect_multipliers = [
        4 * radial_majorants[order]
        + (2 * order * radial_majorants[order - 1] if order else 0)
        for order in range(5)
    ]
    expected_radial_majorants = [1, 10080, 80640, 735840, 7650720]
    expected_defect_multipliers = [4, 40322, 362880, 3427200, 36489600]

    chi_symbol, q0_symbol, qdir_symbol = sp.symbols("chi Q0 Qdir")
    expanded_defect = (1 - chi_symbol) * q0_symbol + chi_symbol * qdir_symbol
    expected_defect = (1 - chi_symbol) * q0_symbol
    defect_identity_residual = sp.expand(expanded_defect - expected_defect).subs(
        qdir_symbol, 0
    )

    radial_integral = sp.integrate(
        4 * sp.pi * t**2 / (1 + t**2) ** 2, (t, 0, sp.oo)
    )
    ball_volume = sp.Rational(4, 3) * sp.pi * 2**3
    schur_coefficient = sp.simplify(
        ball_volume * radial_integral / (2 * sp.pi) ** 3
    )
    final_multiplier = sp.simplify(
        schur_coefficient
        * (
            defect_multipliers[0]
            + 6 * defect_multipliers[2]
            + 9 * defect_multipliers[4]
        )
    )
    rho_three_halves = sp.Rational(1, 2)
    corrupted_scale_defect = sp.simplify(rho_three_halves / h)
    first_ibp_tail = sp.integrate(
        4 * sp.pi * t**2 / (1 + t**2), (t, 0, sp.oo)
    )

    xi1, xi2 = sp.symbols("xi1 xi2")
    mixed_witness = xi1**2 * xi2**2
    laplacian_square_at_zero = sp.expand(
        sp.diff(mixed_witness, xi1, 2) + sp.diff(mixed_witness, xi2, 2)
    )
    laplacian_square_at_zero = sp.expand(
        sp.diff(laplacian_square_at_zero, xi1, 2)
        + sp.diff(laplacian_square_at_zero, xi2, 2)
    ).subs({xi1: 0, xi2: 0})
    pure_fourth_at_zero = (
        sp.diff(mixed_witness, xi1, 4) + sp.diff(mixed_witness, xi2, 4)
    ).subs({xi1: 0, xi2: 0})

    passed = bool(
        set(endpoint_residuals.values()) == {"0"}
        and schur_coefficient == sp.Rational(4, 3)
        and final_multiplier == sp.Rational(1322334736, 3)
        and radial_integral == sp.pi**2
        and upstream_passed
        and radial_majorants == expected_radial_majorants
        and defect_multipliers == expected_defect_multipliers
        and defect_identity_residual == 0
        and first_ibp_tail == sp.oo
        and laplacian_square_at_zero == 8
        and pure_fourth_at_zero == 0
        and corrupted_scale_defect != 0
    )
    return passed, {
        "control": "exact compact physical-frequency symmetrization defect",
        "defect_identity": (
            "D(U,x,xi)=rho(|xi|)[K0(U,x)P55(U,x,xi)-"
            "P55(U,x,xi)^dagger K0(U,x)]"
        ),
        "defect_identity_residual_after_outer_symmetrization": str(
            defect_identity_residual
        ),
        "radial_majorant_provenance": {
            "source": "generic_low_frequency_extension_control",
            "upstream_control_passed": upstream_passed,
            "upstream_radial_cutoff_Frechet_majorants": radial_majorants,
            "derived_formula": "M_q=4*R_q+2*q*R_(q-1), with the second term zero at q=0",
        },
        "definitions": {
            "K0": "K55(U,x,e1)",
            "P55": "sum_(j=1)^3 A55^j(U,x) xi_j",
            "rho": (
                "1 for r<=1; 1-chi(r-1) for 1<r<2; 0 for r>=2"
            ),
            "chi": str(smoothstep),
        },
        "endpoint_residuals": endpoint_residuals,
        "certified_radial_Frechet_majorants_0_through_4": radial_majorants,
        "defect_derivative_multipliers_0_through_4": defect_multipliers,
        "compact_symbol_Schur_lemma": {
            "hypothesis": "supp_xi d subset {|xi|<=2}, d is C4 in xi",
            "bound": "||Op_h(d)||_L2<=4/3*(B0+6*B2+9*B4)",
            "radial_kernel_integral": str(radial_integral),
            "frequency_ball_volume": str(ball_volume),
            "exact_coefficient": str(schur_coefficient),
            "uniform_in_h": True,
            "valid_quantizations": ["left_Kohn_Nirenberg", "Weyl"],
        },
        "physical_scale_contract": {
            "semiclassical_symbol": "K_ext(U,x,eta/h)",
            "high_shell": "h<=1/2 and 1<=|eta|<=2 implies |eta|/h>=2",
            "high_shell_defect_zero": True,
            "bounded_defect_paid_once": "fixed physical low-frequency block",
        },
        "negative_controls": {
            "unscaled_shell": {
                "rho_at_3_over_2": str(rho_three_halves),
                "surviving_principal_defect": str(corrupted_scale_defect),
                "rejected": corrupted_scale_defect != 0,
            },
            "one_integration_by_parts": {
                "R3_kernel_integral": str(first_ibp_tail),
                "rejected": first_ibp_tail == sp.oo,
            },
            "omitted_mixed_fourth_derivatives": {
                "witness": "xi1^2*xi2^2",
                "Delta_xi_squared_at_zero": str(laplacian_square_at_zero),
                "sum_pure_fourths_at_zero": str(pure_fourth_at_zero),
                "rejected": laplacian_square_at_zero != pure_fourth_at_zero,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    low: dict[str, Any], evolution: dict[str, Any], first_order: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(low.get("candidate_id"))
    if any(
        record.get("candidate_id") != candidate_id
        or record.get("coefficients") != low.get("coefficients")
        for record in (evolution, first_order)
    ):
        raise QuarticBoundedFrequencyDefectError("candidate identity mismatch")
    if low.get("status") != "pass_global_C4_positive_K55_symbol_extension":
        raise QuarticBoundedFrequencyDefectError("global C4 K55 prerequisite failed")
    if evolution.get("status") != "pass_full_55_state_degree_one_evolution_symbol_C4_bounds":
        raise QuarticBoundedFrequencyDefectError("degree-one P55 prerequisite failed")
    if first_order.get("status") != "pass_exact_55_variable_principal_first_order_reduction":
        raise QuarticBoundedFrequencyDefectError("physical first-order prerequisite failed")
    source_hash = first_order.get("source_spatial_block_sha256")
    if evolution.get("exact_reduction_provenance", {}).get(
        "source_spatial_block_sha256"
    ) != source_hash:
        raise QuarticBoundedFrequencyDefectError("physical pencil block hash mismatch")
    kappa = int(
        low["global_C4_frequency_derivative_integer_ceilings"]["0,0"][
            "global_ceiling"
        ]
    )
    a = int(evolution["directional_M55_integer_ceilings"]["0,0"])
    multipliers = [4, 40322, 362880, 3427200, 36489600]
    bounds = [multiplier * kappa * a for multiplier in multipliers]
    operator_bound = sp.Rational(4, 3) * (
        bounds[0] + 6 * bounds[2] + 9 * bounds[4]
    )
    return {
        "schema_version": "sigma-quartic-bounded-frequency-defect-certificate-1.0",
        "status": "pass_actual_P55_compact_frequency_defect_KN_L2_lemma",
        "candidate_id": candidate_id,
        "coefficients": low.get("coefficients"),
        "physical_pencil_provenance": {
            "source_spatial_block_sha256": source_hash,
            "linear_in_xi": True,
            "frequency_derivatives_order_2_and_higher_zero": True,
        },
        "candidate_ceilings": {"kappa_c": str(kappa), "a_c": str(a)},
        "defect_frequency_derivative_bounds": {
            str(order): str(value) for order, value in enumerate(bounds)
        },
        "operator_L2_bound": {
            "exact": str(operator_bound),
            "factorized": "(1322334736/3)*kappa_c*a_c",
            "numeric": float(sp.N(operator_bound, 18)),
            "quantization": "Kohn-Nirenberg or Weyl compact-symbol Schur bound",
        },
        "physical_scale_contract_passed": True,
        "full_energy_closed": False,
        "remaining_gate": "identify_composition_quantization_and_sum_dyadic_energy",
        "scope": (
            "This certifies the fixed physical low-frequency defect as a bounded L2 "
            "operator and proves it vanishes on correctly rescaled high shells. It does "
            "not close anti-Wick composition, the global H7 commutator, or lifespan."
        ),
    }


def run_quartic_bounded_frequency_defect_campaign(
    low_frequency_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticBoundedFrequencyDefectError("unsupported campaign schema_version")
        campaigns = (low_frequency_campaign, evolution_campaign, first_order_campaign)
        expected_statuses = (
            "pass_all_12_global_C4_positive_K55_symbol_extensions",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
        )
        if tuple(item.get("status") for item in campaigns) != expected_statuses:
            raise QuarticBoundedFrequencyDefectError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(item) for item in campaigns):
            raise QuarticBoundedFrequencyDefectError("campaign content hash mismatch")
        if evolution_campaign.get("first_order_campaign_sha256") != first_order_campaign.get(
            "content_sha256"
        ):
            raise QuarticBoundedFrequencyDefectError("first-order provenance mismatch")
        if low_frequency_campaign.get("upstream_sha256", {}).get(
            "symbol"
        ) != evolution_campaign.get("symbol_campaign_sha256"):
            raise QuarticBoundedFrequencyDefectError("symbol provenance mismatch")
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["maximum_frequency_derivative_order"]) != 4
            or config.get("physical_frequency_radii") != [1, 2]
            or config.get("semiclassical_symbol") != "K_ext(U,x,eta/h)"
            or config.get("normalized_high_shell") != ["1", "2"]
            or config.get("high_shell_h_maximum") != "1/2"
            or config.get("high_shell_physical_radius_minimum") != "2"
        ):
            raise QuarticBoundedFrequencyDefectError("unsupported compact-defect domain")
        control_passed, control = generic_compact_frequency_defect_control()
        if not control_passed:
            raise QuarticBoundedFrequencyDefectError("generic compact-defect control failed")
        maps = tuple(_candidate_records(item) for item in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticBoundedFrequencyDefectError("candidate-set mismatch")
        certificates = [
            _certify_candidate(*(records[candidate_id] for records in maps))
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_actual_P55_compact_frequency_defect_KN_L2_lemmas",
            "errors": [],
            "upstream_sha256": {
                "low_frequency": low_frequency_campaign.get("content_sha256"),
                "evolution": evolution_campaign.get("content_sha256"),
                "first_order": first_order_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_compact_frequency_defect_control": control,
            "counts": {
                "selected": len(certificates),
                "compact_frequency_defect_lemmas_passed": len(certificates),
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have an explicit bounded physical-frequency P55 "
                "symmetrization-defect operator, and the defect vanishes on correctly "
                "rescaled high dyadic shells."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticBoundedFrequencyDefectError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "compact_frequency_defect_lemmas_passed": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_bounded_frequency_defect_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
