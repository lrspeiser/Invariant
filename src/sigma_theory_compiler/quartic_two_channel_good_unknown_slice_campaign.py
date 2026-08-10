from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_rank_one_good_unknown_no_go_campaign import (
    _minimal_rank_two_target,
    _source_obstruction,
)

SCHEMA_VERSION = "sigma-quartic-two-channel-good-unknown-slice-campaign-1.0"
DIMENSION = 11


class QuarticTwoChannelGoodUnknownSliceError(ValueError):
    """Raised when the two-channel slice identity is overstated."""


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


def _matrix_entries(matrix: sp.MatrixBase) -> list[dict[str, Any]]:
    return [
        {"row": row, "column": column, "value": str(matrix[row, column])}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


@cache
def generic_two_channel_good_unknown_slice_control() -> tuple[bool, dict[str, Any]]:
    alpha = sp.Symbol("alpha", nonzero=True)
    eta1 = sp.Symbol("eta1", real=True)
    obstruction = _source_obstruction(alpha)
    correction, channels = _minimal_rank_two_target(alpha)

    # eta is the low input frequency and kappa is the high w1[10] frequency.
    # The kinematic identity d_t w1[10]=d_1 v0[10] supplies H_01 on the high slot.
    source_symbol = sp.I * eta1 * obstruction
    differentiated_correction_symbol = sp.I * eta1 * correction
    cancellation = (source_symbol + differentiated_correction_symbol).applyfunc(
        sp.factor
    )
    zero_order_residual = (source_symbol + correction).applyfunc(sp.factor)
    wrong_sign_residual = (source_symbol - differentiated_correction_symbol).applyfunc(
        sp.factor
    )

    low_time_atoms = sp.Matrix(sp.symbols("dL0:11"))
    high_w = sp.Symbol("w_high")
    induced_low_evolution = correction * low_time_atoms * high_w
    omitted_product_rule_defect = induced_low_evolution.applyfunc(sp.factor)

    first_only = (obstruction + channels[0]).applyfunc(sp.factor)
    second_only = (obstruction + channels[1]).applyfunc(sp.factor)
    passed = bool(
        cancellation.is_zero_matrix
        and not zero_order_residual.is_zero_matrix
        and not wrong_sign_residual.is_zero_matrix
        and not omitted_product_rule_defect.is_zero_matrix
        and not first_only.is_zero_matrix
        and not second_only.is_zero_matrix
        and correction.rank() == 2
    )
    return passed, {
        "control": "two-channel differentiated-paraproduct identity on s01/H01",
        "actual_state_to_jet_derivation": {
            "high_state": "w1[10]=partial_1 phi",
            "kinematic_evolution": "partial_t w1[10]=s01[10]=partial_1 v0[10]",
            "low_dynamic_vector": "v=(v0[0],...,v0[10])",
            "principal_injection": "J_s01=i*eta1*I_11 on the low v slot",
        },
        "modified_state_ansatz": {
            "definition": (
                "v_sharp=v+Q T_(partial_1 v_low) w1[10]_high, with the "
                "paraproduct applied componentwise in the low v slot"
            ),
            "Q": "2*alpha*((e0-4*e4)e10^T+e10(e7+e9)^T)",
            "channel_1": "2*alpha*(e0-4*e4) T_(partial_1 v0[10]_low) w1[10]_high",
            "channel_2": (
                "2*alpha*e10 T_(partial_1(v0[7]+v0[9])_low) w1[10]_high"
            ),
            "paradifferential_order": (
                "one derivative on the spectrally low factor and zero derivatives on "
                "the high state factor"
            ),
        },
        "time_differentiated_identity": {
            "product_rule": (
                "partial_t C_Q=Q T_(partial_1 partial_t v_low) w_high + "
                "Q T_(partial_1 v_low) partial_t w_high"
            ),
            "desired_principal_piece": (
                "Q T_(partial_1 v_low) H01_high"
            ),
            "source_symbol": "i*eta1*R",
            "correction_symbol": "i*eta1*Q",
            "full_four_entry_residual_zero": cancellation.is_zero_matrix,
            "fixed_Fourier_LP_time_commutator": "[partial_t,Delta_j]=0 exactly",
        },
        "induced_top_order_ledger": {
            "complete_identity": (
                "(partial_t-T_P)C_Q = Q T_(partial_1 partial_t v_low)w + "
                "Q T_(partial_1 v_low)partial_t w - T_P(Q T_(partial_1 v_low)w)"
            ),
            "terms": [
                {
                    "id": "TC1_low_factor_evolution",
                    "term": "Q T_(partial_1 partial_t v_low) w_high",
                    "origin": "time product rule",
                    "exactly_present": True,
                    "topology": "coefficient_low/state_high",
                    "quantitative_bound_instantiated": False,
                },
                {
                    "id": "TC2_physical_operator_on_correction",
                    "term": "-T_P(Q T_(partial_1 v_low) w_high)",
                    "origin": "substitution v=v_sharp-C_Q into the 55-state operator",
                    "exactly_present": True,
                    "topology": "first-order physical operator composed with paraproduct",
                    "quantitative_bound_instantiated": False,
                },
                {
                    "id": "TC3_operator_paraproduct_commutator",
                    "term": (
                        "-[T_P,Q T_(partial_1 v_low)]w_high after adding and "
                        "subtracting Q T_(partial_1 v_low)T_P w_high"
                    ),
                    "origin": "exact decomposition of TC2",
                    "exactly_present": True,
                    "topology": "variable-coefficient Coifman-Meyer commutator",
                    "quantitative_bound_instantiated": False,
                },
                {
                    "id": "TC4_fixed_LP_time_commutator",
                    "term": "[partial_t,T_(partial_1 v_low)]w_high",
                    "origin": "time-dependent product differentiation",
                    "exactly_zero_for_declared_fixed_Fourier_cutoffs": True,
                    "quantitative_bound_instantiated": True,
                },
                {
                    "id": "TC5_nonlinear_substitution_remainder",
                    "term": "F(Y(v_sharp-C_Q))-F(Y(v_sharp))+DF(Y)J(C_Q)",
                    "origin": "nonlinear change of state beyond the quadratic slice",
                    "exactly_present": True,
                    "topology": "at least quadratic in C_Q / cubic in state",
                    "quantitative_bound_instantiated": False,
                },
            ],
            "exhaustiveness_scope": (
                "all terms generated algebraically by applying partial_t-T_P and the "
                "nonlinear source substitution to this correction; boundary, gauge, and "
                "other high-atom families are outside this slice"
            ),
        },
        "negative_controls": {
            "omit_low_spatial_derivative": {
                "residual_nonzero_entries": sum(
                    value != 0 for value in zero_order_residual
                ),
                "frequency_obstruction": "i*eta1*R+Q is not the zero polynomial",
                "rejected": not zero_order_residual.is_zero_matrix,
            },
            "reverse_modified_state_sign": {
                "residual_nonzero_entries": sum(
                    value != 0 for value in wrong_sign_residual
                ),
                "rejected": not wrong_sign_residual.is_zero_matrix,
            },
            "omit_low_factor_time_derivative": {
                "formal_defect_nonzero_entries": sum(
                    value != 0 for value in omitted_product_rule_defect
                ),
                "rejected": not omitted_product_rule_defect.is_zero_matrix,
            },
            "retain_only_channel_1": {
                "residual_nonzero_entries": sum(value != 0 for value in first_only),
                "rejected": not first_only.is_zero_matrix,
            },
            "retain_only_channel_2": {
                "residual_nonzero_entries": sum(value != 0 for value in second_only),
                "rejected": not second_only.is_zero_matrix,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    no_go: dict[str, Any],
    component: dict[str, Any],
    full: dict[str, Any],
    c9: dict[str, Any],
    remedy: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(no_go["candidate_id"])
    records = (component, full, c9, remedy)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTwoChannelGoodUnknownSliceError("candidate identity mismatch")
    coefficients = no_go["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTwoChannelGoodUnknownSliceError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticTwoChannelGoodUnknownSliceError("slice requires a10!=0")
    obstruction = _source_obstruction(alpha)
    correction, channels = _minimal_rank_two_target(alpha)
    eta1 = sp.Symbol("eta1", real=True)
    residual = (sp.I * eta1 * (obstruction + correction)).applyfunc(sp.factor)
    if not residual.is_zero_matrix:
        raise QuarticTwoChannelGoodUnknownSliceError("full-slice residual is nonzero")
    prior_target = no_go["minimal_algebraic_target"]
    if not (
        prior_target["rank"] == 2
        and prior_target["full_slice_residual_zero"]
        and component["principal_jet_injection"]["content_sha256"]
        == full["provenance"]["principal_jet_injection_sha256"]
    ):
        raise QuarticTwoChannelGoodUnknownSliceError("prior target/J mismatch")
    identity_payload = {
        "candidate_id": candidate_id,
        "source_D2_packet_sha256": no_go["provenance"][
            "D2_arithmetic_packet_sha256"
        ],
        "J_s01_slice_sha256": no_go["provenance"][
            "s01_injection_slice_sha256"
        ],
        "source_R": _matrix_entries(obstruction),
        "correction_Q": _matrix_entries(correction),
        "channel_1": _matrix_entries(channels[0]),
        "channel_2": _matrix_entries(channels[1]),
        "composed_residual": _matrix_entries(residual),
        "kinematic_identity": "partial_t w1[10]=s01[10]=partial_1 v0[10]",
    }
    ledger_payload = {
        "identity_sha256": _content_hash(identity_payload),
        "ledger": generic["induced_top_order_ledger"],
        "C9_orders": c9["orders_cumulatively_closed"],
    }
    return {
        "schema_version": "sigma-quartic-two-channel-good-unknown-slice-certificate-1.0",
        "status": "pass_exact_two_channel_s01_slice_induced_commutators_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "D2_arithmetic_packet_sha256": no_go["provenance"][
                "D2_arithmetic_packet_sha256"
            ],
            "full_entry_manifest_sha256": full["provenance"][
                "full_entry_manifest_sha256"
            ],
            "principal_jet_injection_sha256": component[
                "principal_jet_injection"
            ]["content_sha256"],
            "s01_injection_slice_sha256": no_go["provenance"][
                "s01_injection_slice_sha256"
            ],
            "C9_orders": c9["orders_cumulatively_closed"],
            "two_channel_identity_sha256": _content_hash(identity_payload),
            "induced_top_order_ledger_sha256": _content_hash(ledger_payload),
        },
        "modified_state": {
            "dynamic_block": "v0[0]..v0[10]",
            "high_state": "w1[10]",
            "definition": generic["modified_state_ansatz"]["definition"],
            "channel_1": generic["modified_state_ansatz"]["channel_1"],
            "channel_2": generic["modified_state_ansatz"]["channel_2"],
            "built_only_from_actual_state_variables": True,
            "uses_certified_kinematic_state_to_jet_map": True,
        },
        "principal_slice_identity": {
            "source_obstruction_entries": _matrix_entries(obstruction),
            "correction_entries": _matrix_entries(correction),
            "after_J_s01_residual_entries": _matrix_entries(residual),
            "after_J_s01_residual_zero": residual.is_zero_matrix,
            "all_four_nonzero_entries_cancelled": True,
            "time_differentiated_principal_contribution": (
                "Q T_(partial_1 v_low) H01_high"
            ),
            "proved": True,
        },
        "induced_top_order_commutators": generic["induced_top_order_ledger"],
        "induced_term_closure": {
            "fixed_LP_time_commutator_closed": True,
            "low_factor_evolution_bound_closed": False,
            "physical_operator_on_correction_bound_closed": False,
            "operator_paraproduct_commutator_bound_closed": False,
            "nonlinear_substitution_remainder_bound_closed": False,
            "all_induced_terms_closed": False,
        },
        "connection_to_B7_global_H7": {
            "representative_full_s01_H01_slice_removed_from_B7": True,
            "other_high_atom_slices_removed_from_B7": False,
            "induced_two_channel_terms_removed_from_B7": False,
            "coefficient_high_state_low_branch_removed_from_B7": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "materialize P55 acting on the embedded two-channel correction and prove "
            "bounds for TC1, TC2, TC3, and TC5 before extending to other high atoms"
        ),
    }


def run_quartic_two_channel_good_unknown_slice_campaign(
    no_go_campaign: dict[str, Any],
    component_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    resonant_remedy_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            no_go_campaign,
            component_campaign,
            full_jacobian_campaign,
            c9_campaign,
            resonant_remedy_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_exact_single_channel_good_unknown_no_gos_"
                "rank_two_targets_identified_global_H7_fail_closed"
            ),
            "pass_all_12_component_jacobian_schema_audits_packet_missing_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            (
                "pass_all_12_resonant_H6xH7_operators_and_conditional_H8_"
                "remedies_actual_high_low_cancellation_fail_closed"
            ),
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTwoChannelGoodUnknownSliceError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTwoChannelGoodUnknownSliceError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTwoChannelGoodUnknownSliceError("campaign content hash mismatch")
        if (
            no_go_campaign["upstream_sha256"]["component_J_contract"]
            != component_campaign["content_sha256"]
            or no_go_campaign["upstream_sha256"]["full_source_jacobian"]
            != full_jacobian_campaign["content_sha256"]
            or no_go_campaign["upstream_sha256"]["solved_source_C9"]
            != c9_campaign["content_sha256"]
            or no_go_campaign["upstream_sha256"]["resonant_remedy"]
            != resonant_remedy_campaign["content_sha256"]
        ):
            raise QuarticTwoChannelGoodUnknownSliceError("upstream provenance mismatch")
        kinematic = component_campaign[
            "generic_component_jacobian_contract_control"
        ]["kinematic_evolution_rows"]
        if not (
            kinematic["identity"] == "D_Y(dot w_i=s_0i) J=i xi_i delta v0"
            and all(value == "0" for value in kinematic["residuals"].values())
        ):
            raise QuarticTwoChannelGoodUnknownSliceError(
                "kinematic state-to-jet identity mismatch"
            )
        if (
            int(config["expected_candidate_count"]) != 12
            or config.get("high_state") != "w1[10]"
            or config.get("low_derivative") != "partial_1_v0"
            or int(config["channel_count"]) != 2
            or config.get("induced_commutator_policy") != "fail_closed"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTwoChannelGoodUnknownSliceError(
                "unsupported two-channel contract"
            )
        generic_passed, generic = generic_two_channel_good_unknown_slice_control()
        if not generic_passed:
            raise QuarticTwoChannelGoodUnknownSliceError(
                "generic two-channel control failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTwoChannelGoodUnknownSliceError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_exact_two_channel_s01_slice_identities_"
                "induced_commutators_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "rank_one_no_go": no_go_campaign["content_sha256"],
                "component_J_contract": component_campaign["content_sha256"],
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
                "resonant_remedy": resonant_remedy_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_two_channel_good_unknown_slice_control": generic,
            "counts": {
                "selected": len(certificates),
                "two_channel_full_s01_slice_identities_proved": len(certificates),
                "four_entry_cancellations_proved": len(certificates),
                "fixed_LP_time_commutators_closed": len(certificates),
                "remaining_induced_term_ledgers_materialized": len(certificates),
                "all_induced_term_bounds_closed": 0,
                "full_high_atom_families_closed": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The two required field-space channels lift to the actual state correction "
                "Q T_(partial_1 v_low)w1[10]_high. The certified kinematic identity makes "
                "its differentiated principal term cancel all four s01/H01 obstruction "
                "entries exactly for all candidates. Four nonzero induced operator/source "
                "classes remain unbounded, so this is a slice identity rather than H7 closure."
            ),
            "scope": (
                "Only the complete four-entry s01/H01 reference slice is removed. Other "
                "high atoms, the induced TC1/TC2/TC3/TC5 terms, B7, global summation, and "
                "lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticTwoChannelGoodUnknownSliceError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "two_channel_full_s01_slice_identities_proved": 0,
                "four_entry_cancellations_proved": 0,
                "fixed_LP_time_commutators_closed": 0,
                "remaining_induced_term_ledgers_materialized": 0,
                "all_induced_term_bounds_closed": 0,
                "full_high_atom_families_closed": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_two_channel_good_unknown_slice_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
