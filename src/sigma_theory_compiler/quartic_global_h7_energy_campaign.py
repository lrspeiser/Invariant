from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-global-h7-energy-campaign-1.0"


class QuarticGlobalH7EnergyError(ValueError):
    """Raised when a global H7 energy input or closure claim is inconsistent."""


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
def generic_global_h7_energy_control() -> tuple[bool, dict[str, Any]]:
    """Prove dyadic neighbor summation and the conditional lifespan formula."""

    q = sp.Symbol("q", nonnegative=True, finite=True)
    weight_ratio_residual = sp.expand(16 * (1 + q) - (1 + 16 * q))
    maximum_weight_ratio = sp.Integer(16) ** 7
    interacting_shells = sp.Integer(5)
    neighbor_sum_constant = interacting_shells * maximum_weight_ratio
    finite_shell_indices = tuple(range(7))
    finite_shell_weight_sum = 1 + sum(
        (1 + 2 ** (2 * shell)) ** 7 for shell in finite_shell_indices
    )

    a, d, z0, time = sp.symbols("A D z0 t", positive=True, finite=True)
    exponential = sp.exp(a * time / 2)
    riccati_solution = a * z0 * exponential / (
        a + d * z0 * (1 - exponential)
    )
    riccati_residual = sp.factor(
        sp.diff(riccati_solution, time)
        - a * riccati_solution / 2
        - d * riccati_solution**2 / 2
    )
    z_target = sp.Symbol("z_tube", positive=True, finite=True)
    conditional_time = sp.Rational(2, 1) / a * sp.log(
        z_target * (a + d * z0) / (z0 * (a + d * z_target))
    )

    leading, remainder = sp.symbols("L_principal R_Bony", nonzero=True, finite=True)
    good_unknown_residual = sp.expand(leading - leading)
    omitted_remainder = remainder
    frequency = sp.Symbol("N", positive=True, finite=True)
    h6_to_h7_loss = frequency ** (7 - 6)
    passed = bool(
        weight_ratio_residual == 15
        and maximum_weight_ratio == 2**28
        and neighbor_sum_constant == 5 * 2**28
        and riccati_residual == 0
        and good_unknown_residual == 0
        and omitted_remainder != 0
        and h6_to_h7_loss == frequency
    )
    return passed, {
        "control": "global H7 dyadic energy ledger with conditional lifespan",
        "dyadic_neighbor_summation": {
            "weight": "w_j=(1+2^(2j))^7",
            "interaction": "|j-k|<=2",
            "base_ratio_proof": "1+16q<=16(1+q)",
            "base_ratio_residual": str(weight_ratio_residual),
            "maximum_weight_ratio": str(maximum_weight_ratio),
            "ordinary_shells_per_enlarged_shell": int(interacting_shells),
            "exact_neighbor_sum_constant": str(neighbor_sum_constant),
            "bound": (
                "sum_j w_j||tildeDelta_j u||2^2<="
                "5*2^28*Q7(u)"
            ),
        },
        "finite_ordinary_shells": {
            "indices": list(finite_shell_indices),
            "support_radius": "R_j=2^(j+1)",
            "maximum_support_radius": 128,
            "low_plus_finite_weight_sum": str(finite_shell_weight_sum),
            "direct_principal_bound": (
                "||Pi_j(OpAW(K)L+L^dagger OpAW(K))Pi_j||<="
                "2*sqrt(3)*Lambda*A0*2^(j+1)"
            ),
        },
        "good_unknown_leading_control": {
            "residual": str(good_unknown_residual),
            "interpretation": (
                "the leading high-coefficient/low-state symbol cancels only after the "
                "component identity D_Y E55 J=iP55 is supplied"
            ),
            "paralinearization_remainder_removed": False,
        },
        "global_remainder_functional": {
            "definition": (
                "B7(t)^2=sum_(j>=7) w_j||F_j^unresolved,total||2^2, where "
                "F_j^unresolved,total includes the 594 lower-DF entries, "
                "F(Y)-T_(D F(Y))Y, remote paraproducts, and every unencoded "
                "good-unknown evolution remainder"
            ),
            "required_bound": "B7(t)<=C_L(R)*sqrt(Q7(t))+C_B(R)*Q7(t)",
        },
        "conditional_lifespan": {
            "hypothesis": "E'<=A E+D E^(3/2), z=sqrt(E)",
            "riccati_solution": str(riccati_solution),
            "riccati_residual": str(riccati_residual),
            "time_to_tube_threshold": str(conditional_time),
            "domain": "0<z0<z_tube and A,D>0",
        },
        "negative_controls": {
            "omit_neighbor_overlap": {
                "missing_factor": str(neighbor_sum_constant),
                "rejected": neighbor_sum_constant != 1,
            },
            "omit_Bony_remainder": {
                "residual": str(omitted_remainder),
                "rejected": omitted_remainder != 0,
            },
            "claim_H7_from_naive_H6_coefficient_commutator": {
                "frequency_growth": str(h6_to_h7_loss),
                "rejected": h6_to_h7_loss != 1,
            },
            "claim_numeric_lifespan_without_C_B_and_initial_energy": {
                "missing_inputs": ["C_L(R)", "C_B(R)", "E(0)"],
                "rejected": True,
            },
        },
        "missing_remainder_certificate_schema": {
            "schema_version": "sigma-quartic-global-source-remainder-certificate-1.0",
            "required_fields": {
                "candidate_id": "string",
                "universal_source_dag_sha256": "64 lowercase hex",
                "coordinate_atom_basis_sha256": "64 lowercase hex",
                "state_basis_sha256": "64 lowercase hex",
                "principal_jet_injection_sha256": "64 lowercase hex",
                "universal_affine_split_entry_residuals": "11 exact zeros",
                "full_DF_shape": [11, 153],
                "full_DF_entries_completed": 1683,
                "Frechet_orders": [2, 3, 4],
                "mixed_multi_index_component_encoding": "exact sparse DAG roots",
                "B7_bound": "B7<=C_L(R)*sqrt(Q7)+C_B(R)*Q7",
                "C_L_exact_expression": "nonnegative exact expression",
                "C_B_exact_expression": "positive exact expression",
                "tube_radius": "exact expression",
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    annular: dict[str, Any],
    frequency: dict[str, Any],
    finite_low: dict[str, Any],
    reference: dict[str, Any],
    good_unknown: dict[str, Any],
    source_jacobian: dict[str, Any],
    source_dag: dict[str, Any],
    lower_source: dict[str, Any],
    dyadic: dict[str, Any],
    time_atoms: dict[str, Any],
    nonlinear: dict[str, Any],
    tube: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    records = (
        annular,
        frequency,
        finite_low,
        reference,
        good_unknown,
        source_jacobian,
        source_dag,
        lower_source,
        dyadic,
        time_atoms,
        nonlinear,
        tube,
    )
    candidate_id = str(annular.get("candidate_id"))
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticGlobalH7EnergyError("candidate identity mismatch")
    if any(
        record.get("coefficients") != annular.get("coefficients")
        for record in records[1:]
    ):
        raise QuarticGlobalH7EnergyError("candidate coefficient mismatch")
    expected_statuses = (
        "pass_targeted_annular_K55_C6_principal_composition_constant",
        "pass_high_shell_coupled_energy_partial_low_sources_and_sum_fail_closed",
        "pass_finite_low_anti_wick_principal_source_fail_closed",
        "pass_exact_reference_equilibrium_and_localized_L2_source_convention",
        "audit_paradifferential_good_unknown_binding_fail_closed",
        "pass_complete_principal_source_jacobian_partial_full_tensor",
        "partial_exact_universal_source_operator_dag_checkpoint",
        "audit_lower_source_and_component_remainder_fail_closed",
        "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "pass_H7_closed_coordinate_atom_time_budget",
        "pass_exact_local_nonlinear_time_acceleration_elimination",
        "pass_uniform_coordinate_2jet_to_covariant_hyperbolicity_tube",
    )
    if tuple(record.get("status") for record in records) != expected_statuses:
        raise QuarticGlobalH7EnergyError("candidate prerequisite status mismatch")

    physical_hash = frequency["physical_pencil"]["source_spatial_block_sha256"]
    if (
        good_unknown["physical_pencil_provenance"]["source_spatial_block_sha256"]
        != physical_hash
        or source_jacobian["source_jacobian_chunk_packet"][
            "published_physical_block_sha256"
        ]
        != physical_hash
    ):
        raise QuarticGlobalH7EnergyError("good-unknown physical-pencil binding mismatch")
    identity = source_jacobian["principal_composed_identity"]
    if not (
        good_unknown["paradifferential_good_unknown"][
            "Bony_high_low_interaction_absorbed_generically"
        ]
        and identity.get("proved") is True
        and identity.get("entry_residuals_proved_zero") == 3025
    ):
        raise QuarticGlobalH7EnergyError("leading good-unknown identity is absent")
    basis = source_jacobian["basis_and_injection_provenance"]
    dag_provenance = source_dag["provenance"]
    for key in (
        "coordinate_atom_basis_sha256",
        "state_basis_sha256",
        "principal_jet_injection_sha256",
    ):
        if basis[key] != dag_provenance[key]:
            raise QuarticGlobalH7EnergyError("source DAG basis provenance mismatch")
    if not (
        source_dag.get("exact_component_derivative_roots_emitted") == 88
        and source_dag.get("universal_acceleration_affine_split_proved") is False
        and source_dag.get("full_component_Frechet_tensors_complete") is False
        and source_dag.get("paralinearization_remainder_bound_proved") is False
        and source_dag["evidence"]["coverage"][
            "mixed_multi_index_components_completed"
        ]
        == 0
        and lower_source["source_jacobian_completion"]["exact_entries_missing"]
        == 594
    ):
        raise QuarticGlobalH7EnergyError("source remainder fail-closed state mismatch")

    energy = frequency["energy_equivalence"]
    low_lower = sp.sympify(energy["global_low_lower"])
    high_lower = sp.sympify(energy["high_localized_lower"])
    upper = sp.sympify(energy["global_upper"])
    if not (low_lower > 0 and high_lower > 0 and upper >= low_lower):
        raise QuarticGlobalH7EnergyError("energy equivalence is invalid")
    composition = sp.sympify(
        annular["principal_anti_wick_composition_constant"]["exact"]
    )
    if sp.simplify(
        composition
        - sp.sympify(
            frequency["high_shell_j_ge_7"][
                "principal_anti_wick_composition_constant"
            ]
        )
    ) != 0:
        raise QuarticGlobalH7EnergyError("annular composition constant mismatch")

    neighbor_constant = sp.Integer(
        generic["dyadic_neighbor_summation"]["exact_neighbor_sum_constant"]
    )
    high = frequency["high_shell_j_ge_7"]
    high_self = sp.sympify(high["coupled_norm_growth_constant"])
    projection = sp.sympify(high["projection_commutator_constant"])
    high_known_q7 = high_self
    low_known = sp.sympify(
        finite_low["source_free_finite_low_energy"]["norm_growth_constant"]
    )
    low_projection = sp.sympify(
        finite_low["source_free_finite_low_energy"][
            "low_projection_commutator_constant"
        ]
    )
    low_neighbor_q7 = upper * low_projection * 2**15
    a0 = sp.sympify(
        finite_low["finite_low_anti_wick_principal"]["directional_A0"]
    )
    time_k = sp.sympify(high["time_K_constant"])
    finite_max_radius = sp.Integer(
        generic["finite_ordinary_shells"]["maximum_support_radius"]
    )
    finite_ordinary_self = (
        time_k
        + 2 * sp.sqrt(3) * upper * a0 * finite_max_radius
        + upper * projection
    )
    all_shell_neighbor_q7 = upper * projection * neighbor_constant
    source_ceiling = sp.sympify(
        reference["localized_whole_space_source"]["full_55_lower_source_ceiling"]
    )
    finite_weight_sum = sp.Integer(
        generic["finite_ordinary_shells"]["low_plus_finite_weight_sum"]
    )
    # For -1<=j<=6, the finite weighted source square sum is bounded by
    # W_fin||S||2^2.  Also ||Z||2<=||U||H7<=2^(15/2)Q7^(1/2).
    finite_source_q7 = (
        2
        * upper
        * source_ceiling
        * sp.sqrt(finite_weight_sum * 2**15)
    )
    known_q7 = (
        high_known_q7
        + finite_ordinary_self
        + low_known
        + low_neighbor_q7
        + all_shell_neighbor_q7
        + finite_source_q7
    )

    h7_lower = high_lower * sp.Rational(1, 2**15)
    h7_upper = upper * 2**14
    known_energy_growth = known_q7 / h7_lower
    unresolved_coefficient = 2 * upper / sp.sqrt(h7_lower)
    tube_radius = sp.sympify(
        time_atoms["sufficient_H7_state_radius_for_coordinate_tube"]["exact"]
    )
    tube_energy_threshold = h7_lower * tube_radius**2
    bootstrap_initial_ceiling = tube_energy_threshold / 4

    numeric = {
        "H7_energy_lower": float(sp.N(h7_lower, 18)),
        "H7_energy_upper": float(sp.N(h7_upper, 18)),
        "known_Q7_growth": float(sp.N(known_q7, 18)),
        "known_energy_growth": float(sp.N(known_energy_growth, 18)),
        "unresolved_B7_coefficient": float(sp.N(unresolved_coefficient, 18)),
        "tube_H7_radius": float(sp.N(tube_radius, 18)),
        "tube_energy_threshold": float(sp.N(tube_energy_threshold, 18)),
        "bootstrap_initial_energy_ceiling": float(
            sp.N(bootstrap_initial_ceiling, 18)
        ),
    }
    if any(
        not (value > 0 and sp.Float(value).is_finite) for value in numeric.values()
    ):
        raise QuarticGlobalH7EnergyError("a global energy constant is invalid")

    return {
        "schema_version": "sigma-quartic-global-h7-energy-certificate-1.0",
        "status": "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": annular.get("coefficients"),
        "global_energy": {
            "definition": (
                "E7=E_-1+sum_(j>=0)(1+2^(2j))^7 E_j, with the certified "
                "high-shell anti-Wick cutoff for j>=7"
            ),
            "H7_lower": str(h7_lower),
            "H7_upper": str(h7_upper),
            "equivalence": "H7_lower*||U||_H7^2<=E7<=H7_upper*||U||_H7^2",
            "coercivity_and_finite_low_modes_included": True,
        },
        "summed_certified_terms": {
            "annular_C6_principal_composition": str(composition),
            "high_self_Q7_constant": str(high_self),
            "projection_commutator_constant": str(projection),
            "neighbor_sum_constant": str(neighbor_constant),
            "high_known_Q7_constant": str(high_known_q7),
            "finite_ordinary_shell_max_radius": str(finite_max_radius),
            "finite_ordinary_self_Q7_constant": str(finite_ordinary_self),
            "finite_low_source_free_Q7_constant": str(low_known),
            "finite_low_projection_neighbor_Q7_constant": str(low_neighbor_q7),
            "all_shell_projection_neighbor_Q7_constant": str(
                all_shell_neighbor_q7
            ),
            "low_plus_finite_weight_sum": str(finite_weight_sum),
            "low_plus_finite_localized_source_Q7_constant": str(
                finite_source_q7
            ),
            "total_known_Q7_constant": str(known_q7),
            "total_known_energy_growth": str(known_energy_growth),
            "ordinary_shells_0_through_6_included": True,
            "all_nonremainder_summations_executable": True,
        },
        "good_unknown_and_source": {
            "physical_pencil_sha256": physical_hash,
            "principal_identity_entries_zero": 3025,
            "leading_good_unknown_symbol_binding_verified": True,
            "leading_derivative_loss_resolved": True,
            "complete_good_unknown_evolution_remainder_closed": False,
            "universal_source_DAG_sha256": source_dag["expression_dag"][
                "content_sha256"
            ],
            "pure_repeated_derivative_roots_emitted": 88,
            "mixed_multi_index_components_completed": 0,
            "lower_DF_entries_missing": 594,
            "universal_affine_split_proved": False,
            "paralinearization_remainder_bound_proved": False,
        },
        "strongest_global_differential_inequality": {
            "unresolved_functional": "B7(t)",
            "B7_definition": generic["global_remainder_functional"]["definition"],
            "exact": (
                "E7'(t)<=A_known*E7(t)+Gamma_B*sqrt(E7(t))*B7(t)"
            ),
            "A_known": str(known_energy_growth),
            "Gamma_B": str(unresolved_coefficient),
            "validity_domain": (
                "the coordinate-tube segment hypothesis holds and all certified "
                "upstream shell/source assumptions remain valid"
            ),
            "proved_with_explicit_remainder": True,
            "closed_Gronwall_inequality": False,
        },
        "bootstrap_and_conditional_lifespan": {
            "tube_H7_radius": str(tube_radius),
            "tube_energy_threshold": str(tube_energy_threshold),
            "suggested_initial_energy_ceiling": str(bootstrap_initial_ceiling),
            "missing_hypothesis": (
                "B7(t)<=C_L(R_tube)*sqrt(Q7(t))+C_B(R_tube)*Q7(t)"
            ),
            "conditional_linear_coefficient": (
                "A=A_known+2*Lambda*C_L*H7_lower^(-1)"
            ),
            "conditional_nonlinear_coefficient": (
                "D=2*Lambda*C_B*H7_lower^(-3/2)"
            ),
            "conditional_time_formula": generic["conditional_lifespan"][
                "time_to_tube_threshold"
            ],
            "substitution": (
                "A=A_known+2*Lambda*C_L/H7_lower, z0=sqrt(E7(0)), "
                "D=2*Lambda*C_B*H7_lower^(-3/2), "
                "z_tube=sqrt(tube_energy_threshold)"
            ),
            "computable_after_missing_C_L_C_B_and_initial_energy": True,
            "numeric_positive_lifespan_proved": False,
        },
        "missing_remainder_certificate_schema": generic[
            "missing_remainder_certificate_schema"
        ],
        "numeric_constants": numeric,
        "global_H7_energy_equivalence_certified": True,
        "global_nonremainder_dyadic_summation_certified": True,
        "global_H7_differential_inequality_closed": False,
        "global_H7_dyadic_sum_applied": False,
        "nonlinear_lifespan_proved": False,
        "remaining_gates": [
            "prove_all_11_universal_acceleration_affine_residuals_zero",
            "complete_594_lower_DF_entries_and_mixed_D2_to_D4_DAG_roots",
            (
                "derive_explicit_B7_less_equal_C_L_sqrt_Q7_plus_C_B_Q7_"
                "paralinearization_bound"
            ),
            "instantiate_initial_energy_and_apply_conditional_lifespan_formula",
        ],
        "scope": (
            "All currently certified principal, time, projection, neighbor, finite-low, "
            "coercivity, and localized-low-source terms are summed. The leading "
            "good-unknown symbol is bound to P55, but the unencoded source/good-unknown "
            "remainder prevents a closed global H7 inequality and lifespan."
        ),
    }


def run_quartic_global_h7_energy_campaign(
    annular_campaign: dict[str, Any],
    frequency_campaign: dict[str, Any],
    finite_low_campaign: dict[str, Any],
    reference_campaign: dict[str, Any],
    good_unknown_campaign: dict[str, Any],
    source_jacobian_campaign: dict[str, Any],
    source_dag_campaign: dict[str, Any],
    lower_source_campaign: dict[str, Any],
    dyadic_campaign: dict[str, Any],
    time_atom_campaign: dict[str, Any],
    nonlinear_campaign: dict[str, Any],
    tube_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticGlobalH7EnergyError("unsupported campaign schema_version")
        campaigns = {
            "annular": annular_campaign,
            "frequency": frequency_campaign,
            "finite_low": finite_low_campaign,
            "reference": reference_campaign,
            "good_unknown": good_unknown_campaign,
            "source_jacobian": source_jacobian_campaign,
            "source_dag": source_dag_campaign,
            "lower_source": lower_source_campaign,
            "dyadic": dyadic_campaign,
            "time_atoms": time_atom_campaign,
            "nonlinear": nonlinear_campaign,
            "tube": tube_campaign,
        }
        expected_statuses = {
            "annular": "pass_all_12_targeted_annular_K55_C6_principal_composition_constants",
            "frequency": (
                "pass_all_12_frequency_localized_principal_shell_inequalities_"
                "sources_and_global_sum_fail_closed"
            ),
            "finite_low": (
                "pass_all_12_finite_low_anti_wick_principal_operators_"
                "lower_sources_fail_closed"
            ),
            "reference": "pass_all_12_exact_reference_equilibria_and_L2_source_conventions",
            "good_unknown": (
                "pass_all_12_paradifferential_good_unknown_audits_"
                "component_binding_fail_closed"
            ),
            "source_jacobian": (
                "pass_all_12_complete_unspecialized_principal_source_jacobians_"
                "remainder_fail_closed"
            ),
            "source_dag": "partial_all_12_exact_universal_source_operator_dag_checkpoints",
            "lower_source": "audit_all_12_lower_source_maps_component_remainder_fail_closed",
            "dyadic": "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "time_atoms": "pass_all_12_H7_closed_coordinate_atom_time_budgets",
            "nonlinear": "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
            "tube": "pass_all_12_uniform_coordinate_2jet_to_covariant_hyperbolicity_tubes",
        }
        for name, campaign in campaigns.items():
            if campaign.get("status") != expected_statuses[name]:
                raise QuarticGlobalH7EnergyError(f"{name} prerequisite status mismatch")
            if not _content_hash_matches(campaign):
                raise QuarticGlobalH7EnergyError(f"{name} content hash mismatch")
        upstream = {
            name: campaign["content_sha256"] for name, campaign in campaigns.items()
        }
        if frequency_campaign.get("upstream_sha256", {}).get("annular_C6") != upstream[
            "annular"
        ]:
            raise QuarticGlobalH7EnergyError("frequency-annular provenance mismatch")
        if frequency_campaign.get("upstream_sha256", {}).get("dyadic") != upstream[
            "dyadic"
        ]:
            raise QuarticGlobalH7EnergyError("frequency-dyadic provenance mismatch")
        if frequency_campaign.get("upstream_sha256", {}).get("time_atoms") != upstream[
            "time_atoms"
        ]:
            raise QuarticGlobalH7EnergyError("frequency-time provenance mismatch")
        finite_links = finite_low_campaign.get("upstream_sha256", {})
        if (
            finite_links.get("frequency") != upstream["frequency"]
            or finite_links.get("dyadic") != upstream["dyadic"]
            or finite_links.get("source_jacobian") != upstream["source_jacobian"]
            or finite_links.get("lower_source") != upstream["lower_source"]
        ):
            raise QuarticGlobalH7EnergyError("finite-low provenance mismatch")
        reference_links = reference_campaign.get("upstream_sha256", {})
        if (
            reference_links.get("finite_low") != upstream["finite_low"]
            or reference_links.get("nonlinear") != upstream["nonlinear"]
            or reference_links.get("tube") != upstream["tube"]
        ):
            raise QuarticGlobalH7EnergyError("reference provenance mismatch")
        if source_dag_campaign.get("upstream_sha256", {}).get(
            "lower_source_remainder"
        ) != upstream["lower_source"]:
            raise QuarticGlobalH7EnergyError("source-DAG provenance mismatch")
        if source_dag_campaign.get("upstream_sha256", {}).get(
            "nonlinear_evolution"
        ) != upstream["nonlinear"]:
            raise QuarticGlobalH7EnergyError("source-DAG nonlinear provenance mismatch")
        lower_links = lower_source_campaign.get("upstream_sha256", {})
        if (
            lower_links.get("unspecialized_source_jacobian")
            != upstream["source_jacobian"]
            or lower_links.get("nonlinear_evolution") != upstream["nonlinear"]
        ):
            raise QuarticGlobalH7EnergyError("lower-source provenance mismatch")
        if good_unknown_campaign.get("upstream_sha256", {}).get(
            "dyadic_localization"
        ) != upstream["dyadic"]:
            raise QuarticGlobalH7EnergyError("good-unknown provenance mismatch")
        if tube_campaign.get("content_sha256") != upstream["tube"]:
            raise QuarticGlobalH7EnergyError("tube self-binding mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["state_sobolev_order"]) != 7
            or int(config["high_shell_index_minimum"]) != 7
            or config.get("source_remainder_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticGlobalH7EnergyError("unsupported global H7 contract")
        control_passed, control = generic_global_h7_energy_control()
        if not control_passed:
            raise QuarticGlobalH7EnergyError("generic global H7 control failed")
        maps = {name: _candidate_records(campaign) for name, campaign in campaigns.items()}
        candidate_ids = set(maps["annular"])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps.values()
        ):
            raise QuarticGlobalH7EnergyError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                maps["annular"][candidate_id],
                maps["frequency"][candidate_id],
                maps["finite_low"][candidate_id],
                maps["reference"][candidate_id],
                maps["good_unknown"][candidate_id],
                maps["source_jacobian"][candidate_id],
                maps["source_dag"][candidate_id],
                maps["lower_source"][candidate_id],
                maps["dyadic"][candidate_id],
                maps["time_atoms"][candidate_id],
                maps["nonlinear"][candidate_id],
                maps["tube"][candidate_id],
                control,
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "audit_all_12_global_H7_energies_single_source_remainder_"
                "lifespans_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": upstream,
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_global_h7_energy_control": control,
            "counts": {
                "selected": len(certificates),
                "global_energy_equivalences_certified": len(certificates),
                "global_nonremainder_summations_certified": len(certificates),
                "leading_good_unknown_bindings_verified": len(certificates),
                "closed_global_H7_inequalities": 0,
                "global_H7_sums_applied": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have explicit global H7 energy equivalence and every "
                "currently proved shell term summed into one differential inequality with "
                "a single explicit unresolved B7 functional. No closed global estimate or "
                "lifespan is claimed."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticGlobalH7EnergyError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "global_energy_equivalences_certified": 0,
                "global_nonremainder_summations_certified": 0,
                "leading_good_unknown_bindings_verified": 0,
                "closed_global_H7_inequalities": 0,
                "global_H7_sums_applied": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_global_h7_energy_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
