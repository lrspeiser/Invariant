from __future__ import annotations

import hashlib
import json
from functools import cache
from math import isfinite
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_h7_paracomposition_topology_campaign import (
    _faa_coefficient,
    _partitions,
    generic_h7_paracomposition_topology_control,
)
from .quartic_r3_sobolev_calculus_campaign import (
    r3_sobolev_embedding_constant,
)

SCHEMA_VERSION = "sigma-quartic-h7-resonant-remedy-campaign-1.0"


class QuarticH7ResonantRemedyError(ValueError):
    """Raised when a resonant bound or derivative-loss remedy is overstated."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


@cache
def generic_h7_resonant_remedy_control() -> tuple[bool, dict[str, Any]]:
    _, topology = generic_h7_paracomposition_topology_control()
    remote = sp.sympify(
        topology["remote_resonant_shell_summation"]["remote_weight_sum_upper"]
    )
    balanced_partners = sp.Integer(7)
    bernstein_zero = 2 * sp.sqrt(3) / (3 * sp.pi)
    support_shift_weight = sp.Integer(2) ** 42
    h6_bernstein_ratio = sp.Integer(2) ** 36
    resonant_square = sp.factor(
        balanced_partners
        * remote
        * bernstein_zero**2
        * support_shift_weight
        * h6_bernstein_ratio
    )
    resonant_norm = sp.sqrt(resonant_square)

    partner_cauchy_residual = sp.Integer(7) - sp.Integer(1)
    frequency = sp.Symbol("N", integer=True, positive=True)
    omitted_bernstein_growth = frequency ** sp.Rational(3, 2)
    remote_output_residual = sp.expand(frequency + (-frequency))

    h8_embeddings = {
        order: r3_sobolev_embedding_constant(8, order) for order in range(7)
    }
    h8_coordinate_linf = {
        order: sp.factor(
            sp.sqrt(h8_embeddings[order] ** 2 + 3 * h8_embeddings[order + 1] ** 2)
        )
        for order in range(4)
    }
    h8_partitions = []
    for derivative_order in range(1, 8):
        for parts in _partitions(derivative_order):
            largest = max(parts)
            others = list(parts)
            others.remove(largest)
            h8_partitions.append(
                {
                    "spatial_order": derivative_order,
                    "partition": list(parts),
                    "outer_DF_composition_order": len(parts) + 1,
                    "largest_Y_derivative": largest,
                    "largest_underlying_U_derivative": largest + 1,
                    "largest_other_Y_derivative": max(others, default=0),
                    "compatible": (
                        largest + 1 <= 8 and max(others, default=0) <= 3
                    ),
                }
            )
    h8_packet = frequency**-7
    h7_packet_norm = sp.simplify(frequency**7 * h8_packet)
    h8_packet_norm = sp.simplify(frequency**8 * h8_packet)
    passed = bool(
        remote > 0
        and balanced_partners == 7
        and resonant_square > 0
        and sp.simplify(resonant_norm**2 - resonant_square) == 0
        and partner_cauchy_residual == 6
        and omitted_bernstein_growth != 1
        and remote_output_residual == 0
        and len(h8_partitions) == sum(sp.partition(order) for order in range(1, 8))
        and all(item["compatible"] for item in h8_partitions)
        and max(item["outer_DF_composition_order"] for item in h8_partitions) == 8
        and h7_packet_norm == 1
        and h8_packet_norm == frequency
    )
    return passed, {
        "control": "exact resonant Fourier bound and honest one-derivative remedy audit",
        "resonant_Poincare_Plancherel_Young_bound": {
            "operator": (
                "R(a,u)=sum_(|k-l|<=3) Delta_j(Delta_k a Delta_l u)"
            ),
            "cutoff_multiplier_L2_upper": "1",
            "balanced_partners": str(balanced_partners),
            "Bernstein_L2_to_Linfinity_constant": str(bernstein_zero),
            "Bernstein_source": "exact ball support |xi|<=2^(l+1)",
            "output_support": "j<=max(k,l)+3, with remote low outputs included",
            "remote_weight_sum": str(remote),
            "maximum_three_shell_H7_weight_shift_square": str(
                support_shift_weight
            ),
            "H6_to_Bernstein_denominator_ratio_square": str(
                h6_bernstein_ratio
            ),
            "operator_norm_square_upper": str(resonant_square),
            "operator_norm_upper": str(resonant_norm),
            "bound": "||R(a,u)||H7<=C_res||a||H6||u||H7",
            "instantiated": True,
        },
        "conditional_H8_to_H7_topology": {
            "state": "U in H8(R3;l2^55)",
            "coordinate_atoms": {
                "54_low_atoms": "H8",
                "99_second_atoms": "H7",
                "one_derivative_of_second_atoms": "H6",
            },
            "H8_embedding_constants_orders_0_to_6": {
                str(order): str(value) for order, value in h8_embeddings.items()
            },
            "coordinate_Linfinity_constants_orders_0_to_3": {
                str(order): str(value) for order, value in h8_coordinate_linf.items()
            },
            "DF_composition_partitions_orders_1_to_7": h8_partitions,
            "highest_required_solved_source_Frechet_order": 8,
            "all_partition_topologies_compatible": all(
                item["compatible"] for item in h8_partitions
            ),
            "interpretation": (
                "an a priori H8 state supplies H7 regularity of the 99 second atoms and "
                "therefore a standard H7 high-low product bound"
            ),
        },
        "H8_not_controlled_by_H7": {
            "packet": "U_N=N^-7 exp(iNx_1)u_0",
            "H7_scaling": str(h7_packet_norm),
            "H8_scaling": str(h8_packet_norm),
            "conclusion": (
                "the conditional H8 remedy does not close an autonomous E7 estimate"
            ),
        },
        "negative_controls": {
            "omit_seven_partner_Cauchy_factor": {
                "exact_missing_residual": str(partner_cauchy_residual),
                "rejected": partner_cauchy_residual != 0,
            },
            "omit_Bernstein_frequency_factor": {
                "growth": str(omitted_bernstein_growth),
                "rejected": omitted_bernstein_growth != 1,
            },
            "omit_remote_resonant_outputs": {
                "opposite_input_frequency_output": str(remote_output_residual),
                "rejected": remote_output_residual == 0,
            },
            "claim_E8_control_from_E7": {
                "packet_H8_growth": str(h8_packet_norm),
                "rejected": h8_packet_norm != 1,
            },
            "claim_modified_good_unknown_without_D2_identity": {
                "required_evidence": (
                    "componentwise high-atom D2F contraction/cancellation hash"
                ),
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _conditional_h8_coefficient_bound(
    c9: dict[str, Any], generic: dict[str, Any]
) -> dict[str, Any]:
    m = {
        order: sp.Integer(
            c9["solved_source_Frechet_operator_integer_uppers"][str(order)]
        )
        for order in range(10)
    }
    linf = {
        order: sp.sympify(
            generic["conditional_H8_to_H7_topology"][
                "coordinate_Linfinity_constants_orders_0_to_3"
            ][str(order)]
        )
        for order in range(4)
    }
    radius = sp.Symbol("R8", nonnegative=True, finite=True)
    derivatives: dict[str, Any] = {
        "0": {
            "expression": str(sp.factor(2 * m[2] * radius)),
            "object": "DF(Y)-DF(0)",
        }
    }
    for spatial_order in range(1, 8):
        terms: dict[int, sp.Expr] = {}
        for parts in _partitions(spatial_order):
            largest = max(parts)
            others = list(parts)
            others.remove(largest)
            coefficient = (
                sp.Integer(_faa_coefficient(parts))
                * 2
                * m[len(parts) + 1]
            )
            for derivative in others:
                coefficient *= linf[derivative]
            power = len(parts)
            terms[power] = sp.factor(terms.get(power, 0) + coefficient)
        expression = sp.factor(
            sum(value * radius**power for power, value in terms.items())
        )
        derivatives[str(spatial_order)] = {
            "expression": str(expression),
            "coefficients_by_R8_power": {
                str(power): str(value) for power, value in sorted(terms.items())
            },
        }
    full = sp.sqrt(
        sum(
            sp.binomial(7, order)
            * 3**order
            * sp.sympify(derivatives[str(order)]["expression"]) ** 2
            for order in range(8)
        )
    )
    h7_algebra = 2 * sp.sqrt(21) / sp.sqrt(sp.pi)
    return {
        "radius": "R8=||U||H8(l2^55)",
        "coefficient": "DF(Y)-DF(0)",
        "ordered_derivative_bounds": derivatives,
        "H7_coefficient_upper": str(full),
        "conditional_high_low_product_upper": str(sp.factor(h7_algebra * full)),
        "product_bound": (
            "||T_v(DF(Y)-DF(0))||H7<="
            "C_alg*C_DF(R8)*||v||H7"
        ),
        "C9_orders_used": list(range(2, 9)),
        "conditional_on_finite_R8": True,
        "autonomous_in_E7": False,
    }


def _certify_candidate(
    topology: dict[str, Any],
    c9: dict[str, Any],
    jacobian: dict[str, Any],
    dyadic: dict[str, Any],
    global_energy: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(topology.get("candidate_id"))
    records = (c9, jacobian, dyadic, global_energy)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticH7ResonantRemedyError("candidate identity mismatch")
    if any(record.get("coefficients") != topology.get("coefficients") for record in records):
        raise QuarticH7ResonantRemedyError("candidate coefficient mismatch")
    expected = (
        "pass_H7_atom_topology_and_recombined_tame_ledger_high_low_paraproduct_fail_closed",
        "pass_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
        "pass_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
        "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "audit_global_H7_energy_single_source_remainder_lifespan_fail_closed",
    )
    if tuple(record.get("status") for record in (topology, *records)) != expected:
        raise QuarticH7ResonantRemedyError("candidate prerequisite status mismatch")
    if not (
        topology["Bony_branches"]["coefficient_high_state_low"]["status"]
        == "fail_closed"
        and topology["Bony_branches"]["balanced_resonant"]["status"]
        == "shell_index_sum_only_operator_constant_fail_closed"
        and jacobian["full_component_Frechet_tensors_orders_2_to_4_complete"]
        is False
    ):
        raise QuarticH7ResonantRemedyError("prior branch audit mismatch")
    conditional = _conditional_h8_coefficient_bound(c9, generic)
    radius_symbol = next(
        symbol
        for symbol in sp.sympify(conditional["H7_coefficient_upper"]).free_symbols
        if str(symbol) == "R8"
    )
    test_value = sp.sympify(conditional["H7_coefficient_upper"]).subs(
        radius_symbol, sp.Rational(1, 10**13)
    )
    numeric = float(sp.N(test_value, 18))
    if not (isfinite(numeric) and numeric > 0):
        raise QuarticH7ResonantRemedyError("conditional H8 constant is invalid")
    return {
        "schema_version": "sigma-quartic-h7-resonant-remedy-certificate-1.0",
        "status": (
            "pass_resonant_H6xH7_operator_and_conditional_H8_remedy_"
            "actual_high_low_cancellation_fail_closed"
        ),
        "candidate_id": candidate_id,
        "coefficients": topology["coefficients"],
        "provenance": {
            "physical_pencil_sha256": dyadic["physical_pencil"][
                "source_spatial_block_sha256"
            ],
            "full_entry_manifest_sha256": jacobian["provenance"][
                "full_entry_manifest_sha256"
            ],
            "coordinate_atom_basis_sha256": jacobian["provenance"][
                "coordinate_atom_basis_sha256"
            ],
            "C9_orders": c9["orders_cumulatively_closed"],
        },
        "resonant_branch": {
            "status": "closed_H6_coefficient_times_H7_state_to_H7",
            **generic["resonant_Poincare_Plancherel_Young_bound"],
        },
        "actual_high_low_cancellation_audit": {
            "status": "unproved_fail_closed",
            "full_DF_entries_available": 1683,
            "component_D2_D4_tensors_available": False,
            "C9_information_type": "operator norms only",
            "required_identity": (
                "the coefficient-high projection of D2F(Y)[deltaY_low,Y_high] "
                "cancels with an explicitly named modified-good-unknown term"
            ),
            "required_identity_hash_available": False,
            "Schwartz_counterexample_still_applicable_without_identity": True,
        },
        "minimal_honest_remedy": {
            "choice": "conditional_H8_state_bound_for_the_H7_energy",
            "why_minimal": (
                "one extra state derivative raises the 99 second atoms from H6 to H7"
            ),
            "quantitative_coefficient_bound": conditional,
            "numeric_H7_coefficient_upper_at_R8_1e_minus_13": numeric,
            "proved_conditionally": True,
            "autonomous_H7_closure": False,
            "autonomous_H8_closure": False,
            "reason_not_autonomous": generic["H8_not_controlled_by_H7"]["conclusion"],
            "modified_good_unknown_status": (
                "potential same-order remedy but requires the missing component D2 identity"
            ),
            "different_energy_status": (
                "an E8 energy alone repeats the one-derivative issue unless new structure "
                "or a derivative-loss/Nash-Moser theorem is supplied"
            ),
        },
        "connection_to_B7_global_H7": {
            "resonant_branch_removed_from_B7": True,
            "coefficient_high_state_low_branch_removed_from_B7": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "materialize and prove the system-specific high-atom D2F modified-good-unknown "
            "cancellation, or build a derivative-loss local theory with independent H8 control"
        ),
    }


def run_quartic_h7_resonant_remedy_campaign(
    topology_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    dyadic_campaign: dict[str, Any],
    global_h7_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            topology_campaign,
            c9_campaign,
            full_jacobian_campaign,
            dyadic_campaign,
            global_h7_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_H7_atom_topologies_and_recombined_tame_ledgers_"
                "high_low_paraproduct_fail_closed"
            ),
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "audit_all_12_global_H7_energies_single_source_remainder_lifespans_fail_closed",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticH7ResonantRemedyError("unsupported campaign schema_version")
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticH7ResonantRemedyError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticH7ResonantRemedyError("campaign content hash mismatch")
        links = topology_campaign["upstream_sha256"]
        if (
            links["solved_source_C9"] != c9_campaign["content_sha256"]
            or links["full_source_jacobian"] != full_jacobian_campaign["content_sha256"]
            or links["dyadic_localization"] != dyadic_campaign["content_sha256"]
            or links["global_H7"] != global_h7_campaign["content_sha256"]
        ):
            raise QuarticH7ResonantRemedyError("topology provenance mismatch")
        dyadic_control = dyadic_campaign["generic_dyadic_localization_control"]
        if not (
            dyadic_control["Bernstein_constants_0_through_4"]["0"]
            == "2*sqrt(3)/(3*pi)"
            and dyadic_control["partition"][
                "ordinary_shells_interacting_with_one_enlarged_shell"
            ]
            == 5
            and dyadic_control["derivative_loss_negative"]["growth_exponent"] == 1
        ):
            raise QuarticH7ResonantRemedyError("exact LP prerequisite mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or int(config["spatial_dimension"]) != 3
            or int(config["target_energy_order"]) != 7
            or int(config["conditional_state_order"]) != 8
            or int(config["balanced_gap"]) != 4
            or config.get("actual_high_low_cancellation_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticH7ResonantRemedyError("unsupported resonant/remedy contract")
        generic_passed, generic = generic_h7_resonant_remedy_control()
        if not generic_passed:
            raise QuarticH7ResonantRemedyError("generic resonant/remedy control failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticH7ResonantRemedyError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_resonant_H6xH7_operators_and_conditional_H8_"
                "remedies_actual_high_low_cancellation_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "H7_topology": topology_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "dyadic_localization": dyadic_campaign["content_sha256"],
                "global_H7": global_h7_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_H7_resonant_remedy_control": generic,
            "counts": {
                "selected": len(certificates),
                "resonant_Fourier_operator_constants_instantiated": len(certificates),
                "resonant_branches_closed": len(certificates),
                "actual_high_low_cancellations_proved": 0,
                "conditional_H8_to_H7_remedies_proved": len(certificates),
                "autonomous_H7_closures": 0,
                "autonomous_H8_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The exact declared LP cutoffs now have a finite explicit resonant "
                "H6-by-H7-to-H7 operator constant for all 12 candidates. The actual "
                "coefficient-high/state-low cancellation cannot be inferred from DF roots "
                "and C9 norms because its component D2 identity is absent. One a priori "
                "H8 derivative conditionally controls the H7 branch but does not close "
                "autonomous E7, E8, B7, or lifespan."
            ),
            "scope": (
                "Resonance is closed; the source-specific high-low cancellation remains "
                "the single H7 paracomposition branch gate."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticH7ResonantRemedyError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "resonant_Fourier_operator_constants_instantiated": 0,
                "resonant_branches_closed": 0,
                "actual_high_low_cancellations_proved": 0,
                "conditional_H8_to_H7_remedies_proved": 0,
                "autonomous_H7_closures": 0,
                "autonomous_H8_closures": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_h7_resonant_remedy_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
