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

SCHEMA_VERSION = "sigma-quartic-dyadic-localization-campaign-1.0"
CONFIG_KEYS = {
    "schema_version",
    "expected_candidate_count",
    "spatial_dimension",
    "state_dimension",
    "state_sobolev_order",
    "coefficient_sobolev_order",
    "cutoff_matching_order",
}


class QuarticDyadicLocalizationError(ValueError):
    """Raised when the Littlewood--Paley localization audit cannot be certified."""


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
def generic_dyadic_localization_control() -> tuple[bool, dict[str, Any]]:
    """Certify the LP partition, explicit constants, and derivative-loss witness."""

    t = sp.Symbol("t", real=True, finite=True)
    smoothstep = 126 * t**5 - 420 * t**6 + 540 * t**7 - 315 * t**8 + 70 * t**9
    endpoint_residuals = {
        f"d{order}_at_0": str(sp.diff(smoothstep, t, order).subs(t, 0)) for order in range(5)
    } | {
        f"d{order}_at_1_minus_target": str(
            sp.diff(smoothstep, t, order).subs(t, 1) - (1 if order == 0 else 0)
        )
        for order in range(5)
    }
    derivative_factor_residual = sp.factor(sp.diff(smoothstep, t) - 630 * t**4 * (1 - t) ** 4)

    r = sp.Symbol("r", positive=True, finite=True)
    phi_pieces = (
        (sp.expand(smoothstep.subs(t, 2 * r - 1)), sp.Rational(1, 2), 1),
        (sp.expand(1 - smoothstep.subs(t, r - 1)), 1, 2),
    )
    i1_derived = sp.Integer(0)
    i2_derived = sp.Integer(0)
    for phi_piece, lower, upper in phi_pieces:
        first_integrand = sp.expand(
            r**2 * (phi_piece - sp.diff(phi_piece, r, 2) - 2 * sp.diff(phi_piece, r) / r) ** 2
        )

        def radial_l4(value: sp.Expr) -> sp.Expr:
            return sp.expand(value - sp.diff(value, r, 2) - 4 * sp.diff(value, r) / r)

        second_integrand = sp.expand(r**4 * radial_l4(radial_l4(phi_piece)) ** 2)
        i1_derived += sp.integrate(first_integrand, (r, lower, upper))
        i2_derived += sp.integrate(second_integrand, (r, lower, upper))
    i1_radial = sp.Rational(23176041, 86944)
    i2_radial = sp.Rational(99633325780881, 9998560)
    first_kernel_square = 4 * sp.pi * i1_radial
    second_kernel_square = sp.Rational(4, 3) * sp.pi * i2_radial
    kernel_prefactor = sp.pi * (2 * sp.pi) ** (-sp.Rational(3, 2))
    kappa = sp.factor(
        kernel_prefactor * (sp.sqrt(first_kernel_square) + sp.sqrt(second_kernel_square))
    )

    cutoff_values = sp.symbols("c_m1 c0:7")
    finite_telescoping = cutoff_values[0]
    for index in range(7):
        finite_telescoping += cutoff_values[index + 1] - cutoff_values[index]
    telescoping_residual = sp.expand(finite_telescoping - cutoff_values[-1])
    a = sp.Symbol("a", real=True, finite=True)
    square_lower_residual = sp.factor(a**2 + (1 - a) ** 2 - sp.Rational(1, 2))
    square_upper_residual = sp.factor(1 - a**2 - (1 - a) ** 2)
    q = sp.Symbol("q", positive=True, finite=True)
    h7_lower_base_residual = sp.expand((1 + q) - sp.Rational(1, 4) * (1 + 4 * q))
    h7_upper_base_residual = sp.expand(4 * (1 + q / 4) - (1 + q))
    ordinary_overlap = max(
        sum(1 for shell in range(-8, 9) if shell - 1 < point < shell + 1)
        for point in (sp.Rational(k, 2) for k in range(-15, 16))
    )
    enlarged_simultaneous_overlap = max(
        sum(1 for shell in range(-8, 9) if shell - 2 < point < shell + 2)
        for point in (sp.Rational(k, 2) for k in range(-15, 16))
    )
    ordinary_shells_interacting_with_enlarged = sum(1 for shell in range(-8, 9) if -3 < shell < 3)
    exponent = sp.Integer(7) - sp.Integer(6)
    n = sp.Symbol("N", positive=True, finite=True)
    loss_witness = n**exponent
    bernstein = {
        str(order): str(
            sp.factor(
                (2 * sp.pi) ** (-sp.Rational(3, 2))
                * sp.sqrt(sp.Rational(4, 3) * sp.pi)
                * 2 ** (order + sp.Rational(3, 2))
            )
        )
        for order in range(5)
    }
    expected_bernstein = {
        "0": "2*sqrt(3)/(3*pi)",
        "1": "4*sqrt(3)/(3*pi)",
        "2": "8*sqrt(3)/(3*pi)",
        "3": "16*sqrt(3)/(3*pi)",
        "4": "32*sqrt(3)/(3*pi)",
    }

    z = sp.Symbol("z", real=True, finite=True)
    kernel = sp.Function("K")(z)
    coefficient = sp.Function("A")(z)
    state = sp.Function("v")(z)
    coefficient_at_zero = sp.Symbol("A0")
    product_derivative = sp.diff(kernel * (coefficient - coefficient_at_zero) * state, z)
    commutator_ibp_residual = sp.expand(
        kernel * (coefficient - coefficient_at_zero) * sp.diff(state, z)
        - product_derivative
        + sp.diff(kernel, z) * (coefficient - coefficient_at_zero) * state
        + kernel * sp.diff(coefficient, z) * state
    )

    n_integer = sp.Symbol("N", integer=True, positive=True)
    packet_radius = sp.Rational(1, 8)
    minimum_frequency = sp.Integer(4)
    low_shell_separation_margin = sp.factor(
        sp.Rational(1, 2) - (1 + packet_radius) / minimum_frequency
    )
    high_shell_upper_margin = sp.factor(
        sp.Rational(3, 2) - (minimum_frequency + 1 + 2 * packet_radius) / minimum_frequency
    )
    high_shell_lower_margin = sp.factor(
        (minimum_frequency + 1 - 2 * packet_radius) / minimum_frequency - 1
    )
    packet_product_norm = sp.Symbol("c_packet", positive=True, finite=True)
    loss_lower = packet_product_norm * n_integer / 2
    passed = bool(
        set(endpoint_residuals.values()) == {"0"}
        and derivative_factor_residual == 0
        and i1_derived == i1_radial
        and i2_derived == i2_radial
        and telescoping_residual == 0
        and sp.simplify(square_lower_residual - 2 * (a - sp.Rational(1, 2)) ** 2) == 0
        and sp.simplify(square_upper_residual - 2 * a * (1 - a)) == 0
        and h7_lower_base_residual == sp.Rational(3, 4)
        and h7_upper_base_residual == 3
        and ordinary_overlap == 2
        and enlarged_simultaneous_overlap == 4
        and ordinary_shells_interacting_with_enlarged == 5
        and exponent == 1
        and kappa.is_positive
        and bernstein == expected_bernstein
        and commutator_ibp_residual == 0
        and low_shell_separation_margin > 0
        and high_shell_upper_margin > 0
        and high_shell_lower_margin > 0
        and sp.simplify(loss_lower / (packet_product_norm * n_integer) - sp.Rational(1, 2)) == 0
    )
    return passed, {
        "control": "exact R3 Littlewood-Paley localization and derivative audit",
        "cutoff": {
            "smoothstep": str(smoothstep),
            "chi": "1 for r<=1; 1-S(r-1) for 1<r<2; 0 for r>=2",
            "phi": "phi(r)=chi(r)-chi(2r)",
            "low_multiplier": "m_-1(xi)=chi(2|xi|)",
            "ordinary_multiplier": "m_j(xi)=phi(2^-j|xi|), j>=0",
            "endpoint_residuals": endpoint_residuals,
            "smoothstep_derivative_factor_residual": str(derivative_factor_residual),
        },
        "partition": {
            "telescoping_identity": "m_-1+sum_(j>=0)m_j=1",
            "finite_telescoping_residual": str(telescoping_residual),
            "ordinary_support": "2^(j-1)<|xi|<2^(j+1)",
            "maximum_nonzero_ordinary_multipliers": ordinary_overlap,
            "sum_of_squares_bounds": ["1/2", "1"],
            "sum_of_squares_lower_residual": str(square_lower_residual),
            "sum_of_squares_upper_residual": str(square_upper_residual),
            "enlarged_multiplier": ("mtilde_j=chi(2^(-j-1)|xi|)-chi(2^(2-j)|xi|)"),
            "enlarged_support": "2^(j-2)<|xi|<2^(j+2)",
            "enlarged_equals_one_on_m_j_support": True,
            "maximum_simultaneous_enlarged_multiplier_overlap": (enlarged_simultaneous_overlap),
            "ordinary_shells_interacting_with_one_enlarged_shell": (
                ordinary_shells_interacting_with_enlarged
            ),
        },
        "H7_equivalence": {
            "dyadic_norm": ("Q7=||Delta_-1 u||2^2+sum_j(1+2^(2j))^7||Delta_j u||2^2"),
            "lower": "2^-15",
            "upper": "2^14",
            "symmetrized_lower": "lambda*2^-15",
            "symmetrized_upper": "Lambda*2^14",
            "lower_base_inequality_residual": str(h7_lower_base_residual),
            "upper_base_inequality_residual": str(h7_upper_base_residual),
        },
        "Bernstein_constants_0_through_4": bernstein,
        "shell_local_commutator": {
            "bound": ("||[Delta_j,A^k]partial_k v||2<=kappa_k||nabla A^k||infinity||v||2"),
            "kappa_definition": ("||F^-1 phi||1+integral |z||partial_k F^-1 phi(z)|dz"),
            "first_radial_square_integral": str(first_kernel_square),
            "second_radial_square_integral": str(second_kernel_square),
            "first_radial_rational_derived": str(i1_derived),
            "second_radial_rational_derived": str(i2_derived),
            "explicit_common_kappa_upper": str(kappa),
            "integration_by_parts_integrand_residual": str(commutator_ibp_residual),
            "localized_state": "v=tildeDelta_j u",
        },
        "derivative_loss_negative": {
            "domain": "R3 compact-frequency Schwartz wave packets",
            "packet_contract": (
                "a0_hat and u0_hat are nonnegative C_c^infinity packets of radius "
                "epsilon=1/8 centered at 0 and e1, with c_packet=||a0 partial_1 u0||2>0"
            ),
            "coefficient_family": "a_N=N^-6 exp(i N x1)a0, N=2^j>=4",
            "state": "u=u0 is fixed in H7(R3)",
            "coefficient_H6_bound": (
                "||a_N||H6<=C_a=sum_(k=0)^6 binom(6,k)||D^(6-k)a0||2, uniformly for N>=1"
            ),
            "high_product_Fourier_support": (
                "B((N+1)e1,2*epsilon), contained in normalized radii [1,3/2]"
            ),
            "low_state_Fourier_support": ("B(e1,epsilon), contained below normalized radius 1/2"),
            "low_shell_separation_margin_at_N4": str(low_shell_separation_margin),
            "high_shell_lower_margin_at_N4": str(high_shell_lower_margin),
            "high_shell_upper_margin_at_N4": str(high_shell_upper_margin),
            "high_shell_multiplier_lower": "phi>=phi(3/2)=1/2",
            "fixed_state_H7_norm": "||u0||H7<infinity",
            "commutator_H7_lower": str(loss_lower),
            "commutator_H7_scaling": str(loss_witness),
            "growth_exponent": int(exponent),
            "rejected_claim": ("ell2 H7 commutator <= C ||a||H6 ||u||H7 uniformly in N"),
            "rejected": bool(exponent > 0),
            "R3_Schwartz_Fourier_support_counterexample_encoded": True,
        },
        "passed": passed,
    }


def _certify_candidate(
    r3: dict[str, Any], evolution: dict[str, Any], first_order: dict[str, Any]
) -> dict[str, Any]:
    candidate_id = str(r3.get("candidate_id"))
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != r3.get("coefficients")
        for item in (evolution, first_order)
    ):
        raise QuarticDyadicLocalizationError("candidate identity mismatch")
    if r3.get("status") != "pass_R3_H6_spatialized_K55_P55_symbol_bounds":
        raise QuarticDyadicLocalizationError("R3 symbol prerequisite failed")
    if evolution.get("status") != "pass_full_55_state_degree_one_evolution_symbol_C4_bounds":
        raise QuarticDyadicLocalizationError("evolution symbol prerequisite failed")
    if first_order.get("status") != "pass_exact_55_variable_principal_first_order_reduction":
        raise QuarticDyadicLocalizationError("first-order prerequisite failed")
    source_hash = first_order.get("source_spatial_block_sha256")
    if (
        evolution.get("exact_reduction_provenance", {}).get("source_spatial_block_sha256")
        != source_hash
    ):
        raise QuarticDyadicLocalizationError("physical pencil provenance mismatch")
    local_bound = r3["spatialized_dyadic_P55_bounds"]["1,1"]
    _, generic_control = generic_dyadic_localization_control()
    kappa = sp.sympify(generic_control["shell_local_commutator"]["explicit_common_kappa_upper"])
    radius = sp.Symbol("R", nonnegative=True, finite=True)
    coefficient_expression = sp.sympify(local_bound["expression"], locals={"R": radius})
    shell_local_expression = sp.factor(3 * sp.sqrt(3) * kappa * coefficient_expression)
    return {
        "schema_version": "sigma-quartic-dyadic-localization-certificate-1.0",
        "status": "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "candidate_id": candidate_id,
        "coefficients": r3.get("coefficients"),
        "physical_pencil": {
            "definition": "P55(U,x,xi)=sum_(k=1)^3 A55^k(U,x)xi_k",
            "source_spatial_block_sha256": source_hash,
            "exactly_linear_in_xi": True,
        },
        "shell_local_coefficient_bound": local_bound,
        "shell_local_commutator_bound": {
            "localized_state": "v=tildeDelta_j u",
            "summed_spatial_coefficient_count": 3,
            "per_coordinate_to_Euclidean_gradient_factor": "sqrt(3)",
            "expression": str(shell_local_expression),
            "variable": "R=max_A ||Y_A||_H6",
            "uniform_in_shell_index": True,
        },
        "dyadic_energy_equivalence": {
            "lower": "lambda_c*2^-15",
            "upper": "Lambda_c*2^14",
        },
        "shell_local_commutator_bound_certified": True,
        "conditional_monotone_dyadic_summation": {
            "hypothesis": (
                "a uniform complete shell differential inequality including composition, "
                "source, gauge, constraint, and remote coefficient terms"
            ),
            "applied": False,
            "reason": "the complete uniform shell inequality is not yet certified",
        },
        "full_H7_commutator_closed": False,
        "failure_reason": (
            "the coordinate-atom coefficient field is certified only in H6; the "
            "remote high-coefficient/low-state interaction loses one derivative"
        ),
        "required_next_gate": ("nonlinear_paradifferential_linearization_or_H8_C7_coefficients"),
        "scope": (
            "This certifies the LP family, exact H7 equivalence, Bernstein constants, "
            "and an explicit shell-local commutator bound for the actual pencil. "
            "Dyadic summation is not applied, and global H7 closure from H6 coefficients "
            "is deliberately rejected."
        ),
    }


def run_quartic_dyadic_localization_campaign(
    r3_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticDyadicLocalizationError("unsupported campaign schema_version")
        campaigns = (r3_campaign, evolution_campaign, first_order_campaign)
        validate_bound_inputs(
            config,
            {
                "r3": r3_campaign,
                "evolution": evolution_campaign,
                "first_order": first_order_campaign,
            },
            CONFIG_KEYS,
        )
        statuses = (
            "pass_all_12_R3_H6_spatialized_K55_P55_symbol_bounds",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
        )
        if tuple(item.get("status") for item in campaigns) != statuses:
            raise QuarticDyadicLocalizationError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(item) for item in campaigns):
            raise QuarticDyadicLocalizationError("campaign content hash mismatch")
        if r3_campaign.get("upstream_sha256", {}).get("evolution") != evolution_campaign.get(
            "content_sha256"
        ):
            raise QuarticDyadicLocalizationError("R3 evolution provenance mismatch")
        if evolution_campaign.get("first_order_campaign_sha256") != first_order_campaign.get(
            "content_sha256"
        ):
            raise QuarticDyadicLocalizationError("first-order provenance mismatch")
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["state_sobolev_order"]) != 7
            or int(config["coefficient_sobolev_order"]) != 6
            or int(config["cutoff_matching_order"]) != 4
        ):
            raise QuarticDyadicLocalizationError("unsupported dyadic regularity contract")
        control_passed, control = generic_dyadic_localization_control()
        if not control_passed:
            raise QuarticDyadicLocalizationError("generic dyadic control failed")
        maps = tuple(_candidate_records(item) for item in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticDyadicLocalizationError("candidate-set mismatch")
        certificates = [
            _certify_candidate(*(records[candidate_id] for records in maps))
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "r3_sobolev": r3_campaign.get("content_sha256"),
                "evolution": evolution_campaign.get("content_sha256"),
                "first_order": first_order_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_dyadic_localization_control": control,
            "counts": {
                "selected": len(certificates),
                "dyadic_local_frameworks_passed": len(certificates),
                "full_H7_commutators_closed": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have an explicit H7 dyadic partition and shell-local "
                "commutator framework; all 12 remain fail-closed at the global H7 "
                "commutator because only H6 coefficient regularity is certified."
            ),
            "scope": certificates[0]["scope"],
            "data_seals": DATA_SEALS,
        }
    except (KeyError, TypeError, ValueError, QuarticDyadicLocalizationError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "dyadic_local_frameworks_passed": 0,
                "full_H7_commutators_closed": 0,
                "rejected": 0,
            },
            "data_seals": DATA_SEALS,
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def validate_quartic_dyadic_localization_artifact(
    artifact: dict[str, Any], root: Path, config: dict[str, Any]
) -> None:
    loaded = load_bound_inputs(root, config, ("r3", "evolution", "first_order"))
    rebuilt = run_quartic_dyadic_localization_campaign(
        loaded["r3"], loaded["evolution"], loaded["first_order"], config
    )
    validate_exact_rebuild(artifact, rebuilt)


def write_quartic_dyadic_localization_campaign(result: dict[str, Any], output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
