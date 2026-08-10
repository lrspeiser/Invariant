from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-rank-one-good-unknown-no-go-campaign-1.0"
DIMENSION = 11


class QuarticRankOneGoodUnknownNoGoError(ValueError):
    """Raised when the rank-one no-go is not exactly supported."""


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


def _source_obstruction(alpha: sp.Expr) -> sp.Matrix:
    matrix = sp.zeros(DIMENSION)
    matrix[0, 10] = -2 * alpha
    matrix[4, 10] = 8 * alpha
    matrix[10, 7] = -2 * alpha
    matrix[10, 9] = -2 * alpha
    return matrix


def _minimal_rank_two_target(alpha: sp.Expr) -> tuple[sp.Matrix, list[sp.Matrix]]:
    e = [sp.eye(DIMENSION)[:, index] for index in range(DIMENSION)]
    scalar_to_metric = 2 * alpha * (e[0] - 4 * e[4]) * e[10].T
    metric_to_scalar = 2 * alpha * e[10] * (e[7] + e[9]).T
    return scalar_to_metric + metric_to_scalar, [scalar_to_metric, metric_to_scalar]


def _s01_injection_slice(component_campaign: dict[str, Any]) -> dict[str, Any]:
    generic = component_campaign["generic_component_jacobian_contract_control"]
    injection = generic["principal_jet_injection"]
    entries = [
        item
        for item in injection["entries"]
        if 54 <= int(item["row"]) < 65
    ]
    expected = [
        {
            "row": 54 + field,
            "column": 11 + field,
            "coefficient": "I*xi1",
        }
        for field in range(DIMENSION)
    ]
    if entries != expected:
        raise QuarticRankOneGoodUnknownNoGoError("s01 injection slice mismatch")
    body = {
        "parent_injection_sha256": injection["content_sha256"],
        "atom_rows": "s01[0]..s01[10] = coordinate rows 54..64",
        "state_columns": "v0[0]..v0[10] = state columns 11..21",
        "entries": entries,
        "matrix_identity": "J_s01=i*xi1*I_11",
    }
    return {**body, "content_sha256": _content_hash(body)}


@cache
def generic_rank_one_good_unknown_no_go_control() -> tuple[bool, dict[str, Any]]:
    alpha, xi1 = sp.symbols("alpha xi1", nonzero=True)
    obstruction = _source_obstruction(alpha)
    required, channels = _minimal_rank_two_target(alpha)
    cancellation = (obstruction + required).applyfunc(sp.factor)
    composed = sp.I * xi1 * obstruction
    required_composed = sp.I * xi1 * required

    u0, u10, ell7, ell10 = sp.symbols("u0 u10 ell7 ell10")
    generic_rank_one_minor = sp.expand(
        (u0 * ell10) * (u10 * ell7) - (u0 * ell7) * (u10 * ell10)
    )
    decisive_minor = sp.factor(
        required[0, 10] * required[10, 7]
        - required[0, 7] * required[10, 10]
    )
    first_only_residual = (obstruction + channels[0]).applyfunc(sp.factor)
    second_only_residual = (obstruction + channels[1]).applyfunc(sp.factor)
    corrupted, _ = _minimal_rank_two_target(alpha)
    corrupted[4, 10] = -4 * alpha
    corrupted_residual = (obstruction + corrupted).applyfunc(sp.factor)
    wrong_injection_residual = sp.I * (sp.Symbol("xi2") - xi1) * obstruction
    passed = bool(
        cancellation.is_zero_matrix
        and (composed + required_composed).is_zero_matrix
        and generic_rank_one_minor == 0
        and decisive_minor == 4 * alpha**2
        and required.subs(alpha, 1).rank() == 2
        and not first_only_residual.is_zero_matrix
        and not second_only_residual.is_zero_matrix
        and not corrupted_residual.is_zero_matrix
        and not wrong_injection_residual.is_zero_matrix
    )
    return passed, {
        "control": "exact full-s01-slice one-channel modified-unknown no-go",
        "ansatz_class": {
            "name": "single_scalar_single_output_Alinhac_channel",
            "definition": (
                "one coefficient-high scalar amplitude theta_j, one low-state "
                "functional ell^T V_low, and one dynamic output direction u; its "
                "11x11 correction symbol is i*xi1*u*ell^T"
            ),
            "rank_upper": 1,
            "why_structure_derived": (
                "a single scalar paraproduct channel factors through C and therefore "
                "has outer-product field-space symbol"
            ),
        },
        "actual_required_correction": {
            "source_matrix_nonzero_entries": {
                "0,10": "2*alpha",
                "4,10": "-8*alpha",
                "10,7": "2*alpha",
                "10,9": "2*alpha",
            },
            "after_J_s01": "i*xi1*Q_required",
            "rank_for_alpha_nonzero": 2,
            "decisive_minor_rows_0_10_columns_10_7": str(decisive_minor),
            "generic_rank_one_same_minor": str(generic_rank_one_minor),
        },
        "minimal_algebraic_factorization": {
            "channel_1": "2*alpha*(e0-4*e4)*e10^T",
            "channel_2": "2*alpha*e10*(e7+e9)^T",
            "exactly_cancels_source_slice": cancellation.is_zero_matrix,
            "exactly_cancels_after_J_s01": (
                composed + required_composed
            ).is_zero_matrix,
            "interpretation": (
                "two algebraic channels are necessary and sufficient on this reference "
                "slice, but their occurrence as time derivatives of a modified unknown "
                "is a separate unproved identity"
            ),
        },
        "negative_controls": {
            "retain_only_scalar_to_metric_channel": {
                "nonzero_residual_entries": sum(
                    value != 0 for value in first_only_residual
                ),
                "rejected": not first_only_residual.is_zero_matrix,
            },
            "retain_only_metric_to_scalar_channel": {
                "nonzero_residual_entries": sum(
                    value != 0 for value in second_only_residual
                ),
                "rejected": not second_only_residual.is_zero_matrix,
            },
            "corrupt_metric_row_ratio_minus4_to_minus2": {
                "row4_column10_residual": str(corrupted_residual[4, 10]),
                "rejected": not corrupted_residual.is_zero_matrix,
            },
            "replace_J_s01_xi1_by_xi2": {
                "nonzero_residual_entries": sum(
                    value != 0 for value in wrong_injection_residual
                ),
                "rejected": not wrong_injection_residual.is_zero_matrix,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    d2: dict[str, Any],
    component: dict[str, Any],
    full: dict[str, Any],
    c9: dict[str, Any],
    remedy: dict[str, Any],
    injection_slice: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(d2["candidate_id"])
    records = (component, full, c9, remedy)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticRankOneGoodUnknownNoGoError("candidate identity mismatch")
    coefficients = d2["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticRankOneGoodUnknownNoGoError("candidate coefficient mismatch")
    alpha = sp.sympify(coefficients["a10"])
    if alpha == 0:
        raise QuarticRankOneGoodUnknownNoGoError("rank-two no-go requires alpha!=0")
    obstruction = _source_obstruction(alpha)
    target, channels = _minimal_rank_two_target(alpha)
    required_minor = sp.factor(
        target[0, 10] * target[10, 7] - target[0, 7] * target[10, 10]
    )
    if not (
        (obstruction + target).is_zero_matrix
        and obstruction.rank() == 2
        and target.rank() == 2
        and required_minor == 4 * alpha**2
        and d2["representative_slice"]["component_D2_value"]
        == str(obstruction[0, 10])
    ):
        raise QuarticRankOneGoodUnknownNoGoError("candidate rank-two identity mismatch")
    obstruction_payload = {
        "candidate_id": candidate_id,
        "source_D2_packet_sha256": d2["provenance"][
            "D2_arithmetic_packet_sha256"
        ],
        "source_slice": [
            {"row": row, "column": column, "value": str(obstruction[row, column])}
            for row in range(DIMENSION)
            for column in range(DIMENSION)
            if obstruction[row, column] != 0
        ],
        "J_s01_slice_sha256": injection_slice["content_sha256"],
        "composed_symbol": "i*xi1*source_slice",
    }
    target_payload = {
        "obstruction_sha256": _content_hash(obstruction_payload),
        "target": [
            {"row": row, "column": column, "value": str(target[row, column])}
            for row in range(DIMENSION)
            for column in range(DIMENSION)
            if target[row, column] != 0
        ],
        "channel_1": [str(value) for value in channels[0]],
        "channel_2": [str(value) for value in channels[1]],
        "decisive_minor": str(required_minor),
    }
    return {
        "schema_version": "sigma-quartic-rank-one-good-unknown-no-go-certificate-1.0",
        "status": "pass_exact_single_channel_no_go_rank_two_target_not_lifted",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "D2_arithmetic_packet_sha256": d2["provenance"][
                "D2_arithmetic_packet_sha256"
            ],
            "full_entry_manifest_sha256": full["provenance"][
                "full_entry_manifest_sha256"
            ],
            "principal_jet_injection_sha256": component[
                "principal_jet_injection"
            ]["content_sha256"],
            "s01_injection_slice_sha256": injection_slice["content_sha256"],
            "C9_orders": c9["orders_cumulatively_closed"],
            "composed_obstruction_sha256": _content_hash(obstruction_payload),
            "minimal_rank_two_target_sha256": _content_hash(target_payload),
        },
        "actual_A_W_J_source_identity": {
            "source_D2_nonzero_entries": obstruction_payload["source_slice"],
            "J_s01": "i*xi1*I_11 from v0[field] to s01[field]",
            "composed_obstruction": "i*xi1*D_H01(D_s01 F)",
            "source_rank": obstruction.rank(),
            "composed_rank_for_xi1_nonzero": obstruction.rank(),
        },
        "single_channel_no_go": {
            "ansatz_symbol": "i*xi1*u*ell^T",
            "ansatz_rank_upper": 1,
            "required_rank": target.rank(),
            "decisive_required_minor": str(required_minor),
            "same_minor_for_every_rank_one_outer_product": "0",
            "proved": True,
        },
        "minimal_algebraic_target": {
            "rank": target.rank(),
            "channel_count": 2,
            "channel_1": "2*a10*(e0-4*e4)*e10^T",
            "channel_2": "2*a10*e10*(e7+e9)^T",
            "full_slice_residual_zero": (obstruction + target).is_zero_matrix,
            "actual_modified_unknown_lift_proved": False,
            "missing_identity": (
                "exhibit a two-channel state correction whose differentiated evolution "
                "produces i*xi1*Q_required without new top-order remainders"
            ),
        },
        "connection_to_B7_global_H7": {
            "rank_one_ansatz_eliminated": True,
            "rank_two_algebraic_target_identified": True,
            "rank_two_modified_unknown_identity_proved": False,
            "coefficient_high_state_low_branch_removed_from_B7": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "derive the exact two-channel rank-two correction from the time-differentiated "
            "state variables and audit every new top-order commutator"
        ),
    }


def run_quartic_rank_one_good_unknown_no_go_campaign(
    d2_campaign: dict[str, Any],
    component_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    resonant_remedy_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            d2_campaign,
            component_campaign,
            full_jacobian_campaign,
            c9_campaign,
            resonant_remedy_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_exact_representative_D2_obstructions_"
                "named_good_unknown_cancellation_refuted_global_H7_fail_closed"
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
            raise QuarticRankOneGoodUnknownNoGoError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticRankOneGoodUnknownNoGoError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticRankOneGoodUnknownNoGoError("campaign content hash mismatch")
        if (
            d2_campaign["upstream_sha256"]["full_source_jacobian"]
            != full_jacobian_campaign["content_sha256"]
            or d2_campaign["upstream_sha256"]["solved_source_C9"]
            != c9_campaign["content_sha256"]
            or d2_campaign["upstream_sha256"]["resonant_remedy"]
            != resonant_remedy_campaign["content_sha256"]
        ):
            raise QuarticRankOneGoodUnknownNoGoError("upstream provenance mismatch")
        injection_slice = _s01_injection_slice(component_campaign)
        full_injection_hash = full_jacobian_campaign["certificates"][0][
            "provenance"
        ]["principal_jet_injection_sha256"]
        if injection_slice["parent_injection_sha256"] != full_injection_hash:
            raise QuarticRankOneGoodUnknownNoGoError("J provenance mismatch")
        expected_source = {
            (0, 10): "-2*alpha",
            (4, 10): "8*alpha",
            (10, 7): "-2*alpha",
            (10, 9): "-2*alpha",
        }
        actual_source = {
            (int(item["source_row"]), int(item["principal_field"])): item["value"]
            for item in d2_campaign["actual_reference_audit"][
                "nonzero_D2_entries_in_s01_block"
            ]
        }
        if actual_source != expected_source:
            raise QuarticRankOneGoodUnknownNoGoError("D2 source slice mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or config.get("ansatz_class")
            != "single_scalar_single_output_Alinhac_channel"
            or int(config["maximum_ansatz_rank"]) != 1
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticRankOneGoodUnknownNoGoError("unsupported no-go contract")
        generic_passed, generic = generic_rank_one_good_unknown_no_go_control()
        if not generic_passed:
            raise QuarticRankOneGoodUnknownNoGoError("generic rank-one no-go failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticRankOneGoodUnknownNoGoError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), injection_slice
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_exact_single_channel_good_unknown_no_gos_"
                "rank_two_targets_identified_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "high_atom_D2": d2_campaign["content_sha256"],
                "component_J_contract": component_campaign["content_sha256"],
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
                "resonant_remedy": resonant_remedy_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_rank_one_good_unknown_no_go_control": generic,
            "s01_injection_slice": injection_slice,
            "counts": {
                "selected": len(certificates),
                "single_channel_rank_one_no_gos_proved": len(certificates),
                "minimal_rank_two_algebraic_targets_identified": len(certificates),
                "rank_two_modified_unknown_identities_proved": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "The exact A/W source slice composed with the certified J_s01=i*xi1 I "
                "has rank two for every candidate. Every one-scalar/one-output Alinhac "
                "channel has rank at most one, and a required 2x2 minor equals 4*a10^2, "
                "so no member of that ansatz class can cancel the four-entry obstruction."
            ),
            "scope": (
                "The minimal algebraic two-channel target is exact on this slice. It is "
                "not yet derived as the time derivative of a modified state, so B7, global "
                "H7, summation, and lifespan remain fail-closed."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticRankOneGoodUnknownNoGoError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "single_channel_rank_one_no_gos_proved": 0,
                "minimal_rank_two_algebraic_targets_identified": 0,
                "rank_two_modified_unknown_identities_proved": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_rank_one_good_unknown_no_go_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
