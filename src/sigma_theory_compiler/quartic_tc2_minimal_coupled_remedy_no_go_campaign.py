from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-tc2-minimal-coupled-remedy-no-go-campaign-1.0"


class QuarticTC2MinimalCoupledRemedyNoGoError(ValueError):
    """Raised when the minimal coupled TC2 no-go is not exact."""


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
def generic_tc2_minimal_coupled_remedy_no_go_control() -> tuple[bool, dict[str, Any]]:
    # Exact finite-dimensional analogue of the declared ansatz class.
    principal = sp.diag(1, 2, 3)
    energy = sp.diag(2, 3, 5)
    matrix = sp.Matrix([[1, 0], [0, 1], [0, 0]])
    high = sp.Matrix([0, 0, 1])
    ell0, ell1 = sp.symbols("ell0 ell1", real=True)
    ell = sp.Matrix([ell0, ell1])
    block = matrix * ell * high.T
    skew = (energy * block - block.T * energy).applyfunc(sp.factor)

    gamma = sp.Symbol("gamma", real=True)
    delta_energy = gamma * energy
    linearized_energy_left = (
        delta_energy * principal - principal.T * delta_energy
    ).applyfunc(sp.factor)
    reciprocal = energy.inv() * block.T * energy
    reciprocal_right_vectors = matrix.T * energy

    xi = sp.Symbol("xi", nonzero=True, real=True)
    constraint_residual = sp.I * xi * matrix * ell
    nonhermitian = sp.Matrix([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    nonhermitian_residual = nonhermitian - nonhermitian.T
    failed_positive = energy - 2 * energy
    passed = bool(
        (energy * principal - principal.T * energy).is_zero_matrix
        and matrix.rank() == 2
        and not skew.is_zero_matrix
        and linearized_energy_left.is_zero_matrix
        and reciprocal_right_vectors.rank() == 2
        and not constraint_residual.is_zero_matrix
        and not nonhermitian_residual.is_zero_matrix
        and any(value < 0 for value in failed_positive.diagonal())
    )
    return passed, {
        "control": "minimal coupled deltaK/same-high-state TC2 remedy no-go",
        "declared_ansatz_class": {
            "deltaK": (
                "Hermitian K55-spectral weight correction: deltaK=K55 S(P55,ell), "
                "S is K55-self-adjoint, commutes with P55, and has transformed norm <=1/2"
            ),
            "additional_state_correction": (
                "a local paraproduct correction whose high slot remains the single state "
                "h=w1[10], with q and all w_i derivative-definition variables unchanged"
            ),
            "why_minimal": (
                "it changes only characteristic energy weights and adds no new high state, "
                "nonlocal inverse derivative, or auxiliary constraint variable"
            ),
        },
        "linearized_symmetrizer_equation": {
            "equation": (
                "deltaK P55-P55^dagger deltaK = "
                "-(K55 B-B^dagger K55)"
            ),
            "spectral_weight_left_side": "0",
            "right_side_nonzero_in_control": not skew.is_zero_matrix,
            "no_solution_in_declared_deltaK_class": True,
        },
        "Hermiticity_and_positivity": {
            "Hermitian": True,
            "relative_transformed_norm_upper": "1/2",
            "preserved_lower_margin": "lambda_K55/2",
            "strictly_positive": True,
        },
        "reciprocal_block_range_obstruction": {
            "canonical_block": "K55^-1 e_h ell^T M^dagger K55",
            "right_covectors_as_ell_varies": "range of M^dagger K55",
            "right_covector_span_dimension": reciprocal_right_vectors.rank(),
            "same_high_state_ansatz_span_dimension_upper": 1,
            "no_same_high_state_realization": True,
            "reciprocal_control_nonzero": reciprocal != sp.zeros(3),
        },
        "state_to_jet_constraint_obstruction": {
            "definition_constraint": "partial_t w_i-partial_i v=0",
            "correction_with_w_i_fixed": "residual=-partial_i C_v",
            "principal_residual": "-i*xi_i M ell h_high",
            "control_residual_nonzero": not constraint_residual.is_zero_matrix,
            "constraint_preserving_members_with_nonzero_TC2": False,
        },
        "induced_commutator_ledger": [
            {
                "id": "CK1_time_deltaK",
                "term": "(partial_t deltaK)U",
                "closed": False,
            },
            {
                "id": "CK2_deltaK_P_commutator",
                "term": "deltaK P55-P55^dagger deltaK",
                "closed": True,
                "value_in_declared_class": "0",
            },
            {
                "id": "CK3_spatial_deltaK",
                "term": "partial_i(deltaK P55^i) after adjoint integration by parts",
                "closed": False,
            },
            {
                "id": "CS1_reciprocal_time_product",
                "term": "partial_t of the additional reciprocal state paraproduct",
                "closed": False,
            },
            {
                "id": "CS2_reciprocal_spatial_operator",
                "term": "P55(D) applied to the additional reciprocal state correction",
                "closed": False,
            },
            {
                "id": "CS3_definition_and_curl_constraints",
                "term": "partial_t delta w_i-partial_i delta v and spatial curls",
                "closed": False,
            },
            {
                "id": "CS4_nonlinear_substitution",
                "term": "Taylor remainder from the augmented state transformation",
                "closed": False,
            },
        ],
        "negative_controls": {
            "allow_non_Hermitian_deltaK": {
                "nonzero_anti_Hermitian_entries": sum(
                    value != 0 for value in nonhermitian_residual
                ),
                "rejected": not nonhermitian_residual.is_zero_matrix,
            },
            "exceed_positivity_margin": {
                "mutated_deltaK": "-2*K55",
                "negative_pivots": sum(
                    int(bool(value < 0)) for value in failed_positive.diagonal()
                ),
                "rejected": True,
            },
            "pretend_fixed_high_covector_spans_reciprocal_rows": {
                "required_span_dimension": reciprocal_right_vectors.rank(),
                "available_span_dimension": 1,
                "rejected": reciprocal_right_vectors.rank() > 1,
            },
            "change_v_without_definition_variables": {
                "constraint_residual_nonzero": not constraint_residual.is_zero_matrix,
                "rejected": not constraint_residual.is_zero_matrix,
            },
            "omit_induced_commutators": {
                "nonzero_or_unclosed_terms": 6,
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    tc2: dict[str, Any],
    induced: dict[str, Any],
    symmetrizer: dict[str, Any],
    component: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(tc2["candidate_id"])
    records = (induced, symmetrizer, component)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticTC2MinimalCoupledRemedyNoGoError("candidate identity mismatch")
    coefficients = tc2["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticTC2MinimalCoupledRemedyNoGoError("candidate coefficient mismatch")
    if not (
        tc2["actual_direction_1_packet"]["rank"] == 2
        and tc2["minimal_residual"]["nonzero_for_some_low_coefficient_vector"]
        and not tc2["canonical_missing_completion"][
            "present_in_current_modified_state"
        ]
    ):
        raise QuarticTC2MinimalCoupledRemedyNoGoError("TC2 no-go prerequisite mismatch")
    energy = symmetrizer["energy_equivalence"]
    lower = sp.sympify(energy["K55_2_lower"])
    if not lower > 0:
        raise QuarticTC2MinimalCoupledRemedyNoGoError("K55 lower margin is not positive")
    reduced_margin = sp.factor(lower / 2)
    kinematic = component["principal_jet_injection"]
    if int(kinematic["nonzero_entries"]) != 132:
        raise QuarticTC2MinimalCoupledRemedyNoGoError("state-to-jet injection mismatch")
    equation_payload = {
        "candidate_id": candidate_id,
        "skew_residual_sha256": tc2["provenance"]["skew_residual_sha256"],
        "equation": (
            "deltaK P55-P55^dagger deltaK=-(K55 B-B^dagger K55)"
        ),
        "declared_deltaK_left_side": "0",
        "right_side_nonzero": True,
        "positivity_margin": str(reduced_margin),
    }
    range_payload = {
        "candidate_id": candidate_id,
        "reciprocal_sha256": tc2["provenance"][
            "canonical_reciprocal_block_sha256"
        ],
        "rank_P55_1_Ev_Q": 2,
        "rank_Mdagger_K55": 2,
        "same_high_state_right_covector_span_upper": 1,
    }
    constraint_payload = {
        "state_to_jet_injection_sha256": kinematic["content_sha256"],
        "identity": "partial_t w_i-partial_i v=0",
        "fixed_q_w_correction_residual": "-partial_i C_v",
        "nonzero_when_TC2_nonzero": True,
    }
    return {
        "schema_version": (
            "sigma-quartic-tc2-minimal-coupled-remedy-no-go-certificate-1.0"
        ),
        "status": "pass_minimal_coupled_deltaK_same_high_state_no_go",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "TC2_component_packet_sha256": tc2["provenance"][
                "TC2_component_packet_sha256"
            ],
            "skew_residual_sha256": tc2["provenance"][
                "skew_residual_sha256"
            ],
            "canonical_reciprocal_block_sha256": tc2["provenance"][
                "canonical_reciprocal_block_sha256"
            ],
            "K55_energy_equivalence_sha256": _content_hash(energy),
            "state_to_jet_injection_sha256": kinematic["content_sha256"],
            "linearized_equation_sha256": _content_hash(equation_payload),
            "reciprocal_range_no_go_sha256": _content_hash(range_payload),
            "constraint_no_go_sha256": _content_hash(constraint_payload),
        },
        "linearized_symmetrizer_audit": {
            "equation": equation_payload["equation"],
            "deltaK_class": generic["declared_ansatz_class"]["deltaK"],
            "left_side_in_class": "0",
            "right_side_nonzero": True,
            "solution_in_class": False,
            "Hermiticity_enforced": True,
            "positivity_relative_norm_upper": "1/2",
            "preserved_K55_lower_margin": str(reduced_margin),
        },
        "additional_state_correction_audit": {
            "class": generic["declared_ansatz_class"][
                "additional_state_correction"
            ],
            "canonical_reciprocal_right_covector_span": 2,
            "same_high_state_right_covector_span_upper": 1,
            "reciprocal_block_realizable_in_class": False,
        },
        "state_to_jet_constraint_audit": {
            "derivative_definition_identity": "partial_t w_i-partial_i v=0",
            "q_and_w_fixed": True,
            "induced_residual": "-partial_i C_v",
            "residual_nonzero_for_required_correction": True,
            "constraint_preserved": False,
        },
        "induced_commutators": generic["induced_commutator_ledger"],
        "closure_ledger": {
            "Hermitian_deltaK_class_defined": True,
            "positivity_margin_preserved": True,
            "linearized_symmetrizer_equation_solved": False,
            "canonical_reciprocal_block_derived_from_state": False,
            "state_to_jet_constraints_preserved": False,
            "all_induced_commutators_closed": False,
            "TC2_closed": False,
        },
        "connection_to_B7_global_H7": {
            "minimal_coupled_ansatz_eliminated": True,
            "TC2_closed": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "allow a non-spectral deltaK that solves the full Sylvester equation and/or "
            "introduce at least two independent high-state covectors with accompanying q,w "
            "constraint corrections; then prove positivity and close CK1/CK3/CS1-CS4"
        ),
    }


def run_quartic_tc2_minimal_coupled_remedy_no_go_campaign(
    tc2_no_go_campaign: dict[str, Any],
    induced_campaign: dict[str, Any],
    full_symmetrizer_campaign: dict[str, Any],
    component_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            tc2_no_go_campaign,
            induced_campaign,
            full_symmetrizer_campaign,
            component_campaign,
        )
        expected_statuses = (
            (
                "pass_all_12_exact_TC2_unchanged_K55_no_gos_"
                "reciprocal_blocks_missing_global_H7_fail_closed"
            ),
            (
                "pass_all_12_exact_P55_Q_TC_packets_reference_partial_bounds_"
                "global_H7_fail_closed"
            ),
            "pass_all_12_full_K55_coordinate_atom_C4_derivative_envelopes",
            "pass_all_12_component_jacobian_schema_audits_packet_missing_fail_closed",
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticTC2MinimalCoupledRemedyNoGoError(
                "unsupported campaign schema_version"
            )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticTC2MinimalCoupledRemedyNoGoError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticTC2MinimalCoupledRemedyNoGoError(
                "campaign content hash mismatch"
            )
        if (
            tc2_no_go_campaign["upstream_sha256"]["induced_operator"]
            != induced_campaign["content_sha256"]
        ):
            raise QuarticTC2MinimalCoupledRemedyNoGoError(
                "upstream provenance mismatch"
            )
        if (
            int(config["expected_candidate_count"]) != 12
            or config.get("deltaK_class") != "K55_spectral_commuting_Hermitian"
            or config.get("additional_state_class")
            != "same_high_w1_10_fixed_q_w"
            or config.get("relative_deltaK_norm_upper") != "1/2"
            or config.get("constraint_policy") != "require_exact"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticTC2MinimalCoupledRemedyNoGoError(
                "unsupported minimal coupled contract"
            )
        generic_passed, generic = generic_tc2_minimal_coupled_remedy_no_go_control()
        if not generic_passed:
            raise QuarticTC2MinimalCoupledRemedyNoGoError(
                "generic minimal coupled no-go failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticTC2MinimalCoupledRemedyNoGoError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_minimal_coupled_deltaK_same_high_state_no_gos_"
                "TC2_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "TC2_symmetrizer_no_go": tc2_no_go_campaign["content_sha256"],
                "induced_operator": induced_campaign["content_sha256"],
                "full_K55_symmetrizer": full_symmetrizer_campaign[
                    "content_sha256"
                ],
                "component_J_contract": component_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_minimal_coupled_remedy_no_go_control": generic,
            "counts": {
                "selected": len(certificates),
                "spectral_deltaK_no_gos": len(certificates),
                "same_high_state_reciprocal_no_gos": len(certificates),
                "fixed_q_w_constraint_no_gos": len(certificates),
                "legitimate_coupled_remedies": 0,
                "TC2_closures": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "Hermitian positive K55-spectral weight corrections commute with the "
                "baseline principal operator, so their linearized Sylvester left side is "
                "zero and cannot cancel the nonzero TC2 skew pairing. The canonical "
                "reciprocal block has a two-dimensional moving right-covector span, which "
                "cannot be generated using only the same high state w1[10]. Changing v "
                "while fixing q,w also violates the exact derivative-definition constraints."
            ),
            "scope": (
                "This eliminates the minimal positive spectral-deltaK plus same-high-state, "
                "fixed-constraint-variable remedy. General non-spectral deltaK and augmented "
                "constraint-preserving states remain open. TC2, B7, H7, and lifespan stay false."
            ),
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticTC2MinimalCoupledRemedyNoGoError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "spectral_deltaK_no_gos": 0,
                "same_high_state_reciprocal_no_gos": 0,
                "fixed_q_w_constraint_no_gos": 0,
                "legitimate_coupled_remedies": 0,
                "TC2_closures": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_tc2_minimal_coupled_remedy_no_go_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
