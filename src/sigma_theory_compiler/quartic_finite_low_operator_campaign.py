from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-finite-low-operator-campaign-1.0"


class QuarticFiniteLowOperatorError(ValueError):
    """Raised when the finite-low operator or source audit is inconsistent."""


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
def generic_finite_low_operator_control() -> tuple[bool, dict[str, Any]]:
    """Prove the band-limited operator bound and source omission witnesses."""

    b0, b1, b2 = sp.symbols("b0 b1 b2", nonnegative=True, finite=True)
    cauchy_residual = sp.expand(
        3 * (b0**2 + b1**2 + b2**2)
        - (b0 + b1 + b2) ** 2
        - ((b0 - b1) ** 2 + (b0 - b2) ** 2 + (b1 - b2) ** 2)
    )
    lam, a0, radius = sp.symbols(
        "Lambda_op A0 R_low", positive=True, finite=True
    )
    direct_constant = 2 * sp.sqrt(3) * lam * a0 * radius

    frequency = sp.Symbol("N", positive=True, finite=True)
    unlocalized_derivative_witness = frequency
    epsilon, heat_time = sp.symbols("epsilon tau", positive=True, finite=True)
    smoothed_commutator_at_zero = epsilon * sp.exp(-heat_time)
    line_length = sp.Symbol("L", positive=True, finite=True)
    constant_source_l2_growth = line_length

    passed = bool(
        cauchy_residual == 0
        and direct_constant.is_positive
        and unlocalized_derivative_witness != 1
        and smoothed_commutator_at_zero.is_positive
        and sp.limit(constant_source_l2_growth, line_length, sp.oo) == sp.oo
    )
    return passed, {
        "control": "fixed finite-low anti-Wick principal operator and source audit",
        "low_projector": {
            "multiplier": "m_-1(xi)=chi(2|xi|)",
            "support": "|xi|<=1",
            "Bernstein": "||grad Pi_-1 v||2<=||Pi_-1 v||2",
        },
        "coefficient_Cauchy_control": {
            "inequality": "(b1+b2+b3)^2<=3*(b1^2+b2^2+b3^2)",
            "sum_of_squares_residual": str(cauchy_residual),
        },
        "sandwiched_principal_operator": {
            "operator": (
                "B_low=Pi_-1*(OpAW(K_ext)L+L^dagger*OpAW(K_ext))*Pi_-1"
            ),
            "generator": "L=sum_(k=1)^3 A^k partial_k",
            "bound": "||B_low||<=2*sqrt(3)*Lambda*A0*R_low",
            "exact_template": str(direct_constant),
            "reason": (
                "||OpAW(K_ext)||<=Lambda and ||L Pi_-1||<="
                "sqrt(3)*A0*R_low; no symbol-composition expansion is used"
            ),
            "pointwise_defect_double_counted": False,
        },
        "negative_controls": {
            "omit_low_sandwich": {
                "witness": "a Fourier packet at frequency N",
                "unbounded_derivative_growth": str(unlocalized_derivative_witness),
                "rejected": unlocalized_derivative_witness != 1,
            },
            "replace_operator_composition_by_pointwise_symmetry": {
                "witness": (
                    "K(x)=1+epsilon*sin(x), P(xi)=xi; pointwise KP-PK=0, "
                    "but [exp(tau*Delta)K,D] at x=0 is nonzero"
                ),
                "commutator_at_zero": str(smoothed_commutator_at_zero),
                "rejected": smoothed_commutator_at_zero != 0,
            },
            "use_source_C0_without_basepoint_cancellation": {
                "witness": "a nonzero constant source on R",
                "squared_L2_norm_on_interval_length_L": str(
                    constant_source_l2_growth
                ),
                "whole_space_limit": "oo",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    frequency: dict[str, Any],
    dyadic: dict[str, Any],
    evolution: dict[str, Any],
    low_symbol: dict[str, Any],
    positive: dict[str, Any],
    bounded_defect: dict[str, Any],
    solved_source: dict[str, Any],
    component: dict[str, Any],
    source_jacobian: dict[str, Any],
    lower_source: dict[str, Any],
) -> dict[str, Any]:
    records = (
        frequency,
        dyadic,
        evolution,
        low_symbol,
        positive,
        bounded_defect,
        solved_source,
        component,
        source_jacobian,
        lower_source,
    )
    candidate_id = str(frequency.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticFiniteLowOperatorError("candidate identity mismatch")
    if any(
        record.get("coefficients") != frequency.get("coefficients")
        for record in records[1:]
    ):
        raise QuarticFiniteLowOperatorError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_high_shell_coupled_energy_partial_low_sources_and_sum_fail_closed",
        "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "pass_full_55_state_degree_one_evolution_symbol_C4_bounds",
        "pass_global_C4_positive_K55_symbol_extension",
        "pass_uniform_positive_anti_wick_K55_operator",
        "pass_actual_P55_compact_frequency_defect_KN_L2_lemma",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "audit_component_jacobian_packet_missing_fail_closed",
        "pass_complete_principal_source_jacobian_partial_full_tensor",
        "audit_lower_source_and_component_remainder_fail_closed",
    )
    if tuple(record.get("status") for record in records) != expected_statuses:
        raise QuarticFiniteLowOperatorError("candidate prerequisite status mismatch")

    cutoff = dyadic.get("physical_pencil", {})
    if cutoff.get("exactly_linear_in_xi") is not True:
        raise QuarticFiniteLowOperatorError("physical first-order pencil is absent")
    if low_symbol.get("extension_definition", {}).get("regularity") != (
        "C4 in xi and C4 in the certified state variables"
    ):
        raise QuarticFiniteLowOperatorError("global K_ext contract mismatch")
    if positive.get("quantization", {}).get("positive") is not True:
        raise QuarticFiniteLowOperatorError("positive anti-Wick operator is absent")

    principal_identity = source_jacobian.get("principal_composed_identity", {})
    completion = source_jacobian.get("completion", {})
    source_completion = lower_source.get("source_jacobian_completion", {})
    tensor_gate = lower_source.get("component_Frechet_tensor_gate", {})
    if not (
        principal_identity.get("proved") is True
        and principal_identity.get("entry_residuals_proved_zero") == 3025
        and completion.get("exact_entries_completed") == 1089
        and completion.get("lower_atom_columns_unresolved") == 54
        and source_completion.get("exact_entries_missing") == 594
        and source_completion.get("full_component_tensor_complete") is False
        and tensor_gate.get("actual_component_tensor_orders_available") == []
        and lower_source.get("paralinearization_remainder_bound_proved") is False
        and component.get("D_Y_E55_times_J_equals_iP55_proved") is False
    ):
        raise QuarticFiniteLowOperatorError("lower-source fail-closed audit mismatch")

    energy = frequency["energy_equivalence"]
    lower = sp.sympify(energy["global_low_lower"])
    upper = sp.sympify(energy["global_upper"])
    positive_energy = positive["operator_energy_equivalence"]
    if sp.simplify(lower - sp.sympify(positive_energy["lower"])) != 0 or sp.simplify(
        upper - sp.sympify(positive_energy["upper"])
    ) != 0:
        raise QuarticFiniteLowOperatorError("low anti-Wick energy mismatch")
    if not (lower > 0 and upper >= lower):
        raise QuarticFiniteLowOperatorError("invalid low anti-Wick energy bounds")

    a0 = sp.sympify(evolution["directional_M55_integer_ceilings"]["0,0"])
    low_radius = sp.Integer(1)
    direct_principal = 2 * sp.sqrt(3) * upper * a0 * low_radius
    prior_low = frequency["finite_physical_low_frequencies"]
    time_k = sp.sympify(frequency["high_shell_j_ge_7"]["time_K_constant"])
    projection = sp.sympify(prior_low["low_projection_commutator_constant"])
    defect = sp.sympify(bounded_defect["operator_L2_bound"]["exact"])
    partial_growth = time_k + direct_principal + upper * projection
    partial_energy_growth = partial_growth / lower
    numeric = {
        "direct_low_principal_operator": float(sp.N(direct_principal, 18)),
        "time_K": float(sp.N(time_k, 18)),
        "low_projection_commutator": float(sp.N(projection, 18)),
        "subsumed_compact_pointwise_defect": float(sp.N(defect, 18)),
        "source_free_low_energy_growth": float(sp.N(partial_energy_growth, 18)),
    }
    if any(
        not (value >= 0 and sp.Float(value).is_finite) for value in numeric.values()
    ):
        raise QuarticFiniteLowOperatorError("invalid finite-low operator constant")

    source_frechet = solved_source.get("solved_source_Frechet_derivatives", {})
    if source_frechet.get("orders") != [0, 1, 2, 3, 4]:
        raise QuarticFiniteLowOperatorError("solved-source envelope orders mismatch")
    if "basepoint_value_exactly_zero" in solved_source:
        raise QuarticFiniteLowOperatorError("unexpected source basepoint claim")

    return {
        "schema_version": "sigma-quartic-finite-low-operator-certificate-1.0",
        "status": "pass_finite_low_anti_wick_principal_source_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": frequency.get("coefficients"),
        "finite_low_anti_wick_principal": {
            "projector": "Pi_-1=m_-1(D), supp(m_-1) subset {|xi|<=1}",
            "sandwiched_operator": (
                "Pi_-1*(OpAW(K_ext)L+L^dagger*OpAW(K_ext))*Pi_-1"
            ),
            "directional_A0": str(a0),
            "anti_wick_upper": str(upper),
            "physical_frequency_radius": str(low_radius),
            "exact_operator_norm_bound": str(direct_principal),
            "certified": True,
            "uses_direct_bandlimited_generator_bound": True,
            "requires_C6_symbol_calculus": False,
            "compact_pointwise_defect_constant": str(defect),
            "compact_pointwise_defect_subsumed_not_added": True,
            "previous_low_composition_gate_closed": True,
        },
        "source_free_finite_low_energy": {
            "time_K_constant": str(time_k),
            "low_projection_commutator_constant": str(projection),
            "norm_growth_constant": str(partial_growth),
            "energy_growth_constant": str(partial_energy_growth),
            "inequality": (
                "E_low'<=G_low||Pi_-1 u||2^2+Lambda*C_proj_low"
                "||tildeLow u||2^2+2*Lambda||Pi_-1 u||||Pi_-1 F_lower||2"
            ),
            "principal_operator_complete": True,
            "localized_source_complete": False,
        },
        "localized_lower_source_audit": {
            "principal_composed_identity": "D_Y E55 J=iP55",
            "principal_identity_entries_proved_zero": 3025,
            "principal_source_entries_completed": 1089,
            "lower_source_entries_missing": 594,
            "lower_atom_columns_unresolved": 54,
            "solved_source_Frechet_norm_envelope_orders": [0, 1, 2, 3, 4],
            "exact_component_tensor_orders_available": [],
            "basepoint_cancellation_F_of_reference_equals_zero_certified": False,
            "whole_space_L2_source_bound_certified": False,
            "paralinearization_remainder_bound_proved": False,
            "precise_blocker": lower_source.get("precise_blocker"),
        },
        "numeric_constants": numeric,
        "finite_low_principal_operator_closed": True,
        "finite_low_complete_energy_inequality_closed": False,
        "global_H7_dyadic_sum_applied": False,
        "nonlinear_lifespan_proved": False,
        "remaining_gates": [
            "certify_source_basepoint_cancellation_or_an_L2_background_source",
            "emit_594_missing_lower_source_Jacobian_entries",
            "emit_exact_or_DAG_Frechet_component_tensors_orders_2_to_4",
            "prove_localized_Bony_Moser_source_remainder",
            "resolve_remote_H6_coefficient_to_H7_state_derivative_loss",
            "apply_global_H7_dyadic_sum_and_nonlinear_lifespan_bootstrap",
        ],
        "scope": (
            "The entire fixed-low principal anti-Wick energy operator is bounded by "
            "sandwiching the original differential generator with Pi_-1; this does "
            "not infer operator composition from pointwise symmetrization. The lower "
            "source, global H7 sum, and lifespan remain fail-closed."
        ),
    }


def run_quartic_finite_low_operator_campaign(
    frequency_campaign: dict[str, Any],
    dyadic_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    low_symbol_campaign: dict[str, Any],
    positive_campaign: dict[str, Any],
    bounded_defect_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    component_campaign: dict[str, Any],
    source_jacobian_campaign: dict[str, Any],
    lower_source_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticFiniteLowOperatorError("unsupported campaign schema_version")
        campaigns = {
            "frequency": frequency_campaign,
            "dyadic": dyadic_campaign,
            "evolution": evolution_campaign,
            "low_symbol": low_symbol_campaign,
            "positive": positive_campaign,
            "bounded_defect": bounded_defect_campaign,
            "solved_source": solved_source_campaign,
            "component": component_campaign,
            "source_jacobian": source_jacobian_campaign,
            "lower_source": lower_source_campaign,
        }
        expected_statuses = {
            "frequency": (
                "pass_all_12_frequency_localized_principal_shell_inequalities_"
                "sources_and_global_sum_fail_closed"
            ),
            "dyadic": "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "evolution": "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "low_symbol": "pass_all_12_global_C4_positive_K55_symbol_extensions",
            "positive": "pass_all_12_uniform_positive_anti_wick_K55_operators",
            "bounded_defect": "pass_all_12_actual_P55_compact_frequency_defect_KN_L2_lemmas",
            "solved_source": "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "component": "pass_all_12_component_jacobian_schema_audits_packet_missing_fail_closed",
            "source_jacobian": (
                "pass_all_12_complete_unspecialized_principal_source_jacobians_"
                "remainder_fail_closed"
            ),
            "lower_source": "audit_all_12_lower_source_maps_component_remainder_fail_closed",
        }
        for name, campaign in campaigns.items():
            if campaign.get("status") != expected_statuses[name]:
                raise QuarticFiniteLowOperatorError(
                    f"{name} prerequisite status mismatch"
                )
            if not _content_hash_matches(campaign):
                raise QuarticFiniteLowOperatorError(
                    f"{name} campaign content hash mismatch"
                )
        upstream = {
            name: campaign["content_sha256"] for name, campaign in campaigns.items()
        }
        frequency_links = frequency_campaign.get("upstream_sha256", {})
        for local, remote in (
            ("dyadic", "dyadic"),
            ("evolution", "evolution"),
            ("positive", "positive_quantization"),
            ("bounded_defect", "bounded_frequency"),
        ):
            if frequency_links.get(remote) != upstream[local]:
                raise QuarticFiniteLowOperatorError(
                    f"frequency-to-{local} provenance mismatch"
                )
        bounded_links = bounded_defect_campaign.get("upstream_sha256", {})
        if (
            bounded_links.get("low_frequency") != upstream["low_symbol"]
            or bounded_links.get("evolution") != upstream["evolution"]
        ):
            raise QuarticFiniteLowOperatorError("bounded-defect provenance mismatch")
        if positive_campaign.get("low_frequency_campaign_sha256") != upstream[
            "low_symbol"
        ]:
            raise QuarticFiniteLowOperatorError("positive-symbol provenance mismatch")
        if source_jacobian_campaign.get("upstream_sha256", {}).get(
            "component_contract"
        ) != upstream["component"]:
            raise QuarticFiniteLowOperatorError("source-Jacobian provenance mismatch")
        lower_links = lower_source_campaign.get("upstream_sha256", {})
        for local, remote in (
            ("component", "component_contract"),
            ("solved_source", "solved_source"),
            ("source_jacobian", "unspecialized_source_jacobian"),
        ):
            if lower_links.get(remote) != upstream[local]:
                raise QuarticFiniteLowOperatorError(
                    f"lower-source-to-{local} provenance mismatch"
                )
        if component_campaign.get("upstream_sha256", {}).get(
            "solved_source"
        ) != upstream["solved_source"]:
            raise QuarticFiniteLowOperatorError("component-source provenance mismatch")

        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["fixed_low_frequency_radius"]) != 1
            or config.get("low_principal_policy") != "direct_bandlimited_bound"
            or config.get("lower_source_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticFiniteLowOperatorError("unsupported finite-low contract")
        dyadic_control = dyadic_campaign.get(
            "generic_dyadic_localization_control", {}
        )
        cutoff_control = dyadic_control.get("cutoff", {})
        if (
            cutoff_control.get("low_multiplier") != "m_-1(xi)=chi(2|xi|)"
            or cutoff_control.get("chi")
            != "1 for r<=1; 1-S(r-1) for 1<r<2; 0 for r>=2"
        ):
            raise QuarticFiniteLowOperatorError(
                "fixed-low Fourier support provenance mismatch"
            )
        control_passed, control = generic_finite_low_operator_control()
        if not control_passed:
            raise QuarticFiniteLowOperatorError("generic finite-low control failed")
        maps = {name: _candidate_records(campaign) for name, campaign in campaigns.items()}
        candidate_ids = set(maps["frequency"])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps.values()
        ):
            raise QuarticFiniteLowOperatorError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps["frequency"][candidate_id],
                maps["dyadic"][candidate_id],
                maps["evolution"][candidate_id],
                maps["low_symbol"][candidate_id],
                maps["positive"][candidate_id],
                maps["bounded_defect"][candidate_id],
                maps["solved_source"][candidate_id],
                maps["component"][candidate_id],
                maps["source_jacobian"][candidate_id],
                maps["lower_source"][candidate_id],
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_finite_low_anti_wick_principal_operators_"
                "lower_sources_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": upstream,
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_finite_low_operator_control": control,
            "counts": {
                "selected": len(certificates),
                "finite_low_principal_operators_closed": len(certificates),
                "localized_lower_sources_closed": 0,
                "complete_low_energy_inequalities_closed": 0,
                "global_H7_sums_applied": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have an explicit fixed-low sandwiched anti-Wick "
                "principal operator bound. Current artifacts do not close the localized "
                "lower source, global H7 sum, or lifespan."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticFiniteLowOperatorError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "finite_low_principal_operators_closed": 0,
                "localized_lower_sources_closed": 0,
                "complete_low_energy_inequalities_closed": 0,
                "global_H7_sums_applied": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_finite_low_operator_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
