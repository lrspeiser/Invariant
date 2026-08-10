from __future__ import annotations

import hashlib
import json
from functools import cache
from math import factorial
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-paradifferential-good-unknown-campaign-1.0"


class QuarticParadifferentialGoodUnknownError(ValueError):
    """Raised when the good-unknown audit cannot be certified."""


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


def _interaction(left: int, right: int, gap: int) -> str:
    if left <= right - gap:
        return "low_high"
    if right <= left - gap:
        return "high_low"
    return "balanced"


@cache
def generic_paradifferential_good_unknown_control() -> tuple[bool, dict[str, Any]]:
    """Prove finite Bony identities and exact top-derivative isolation."""

    gap = 4
    levels = tuple(range(-6, 7))
    classification_counts = {"low_high": 0, "high_low": 0, "balanced": 0}
    one_hot_residuals: dict[str, int] = {}
    for left in levels:
        for right in levels:
            label = _interaction(left, right, gap)
            classification_counts[label] += 1
            indicators = {
                name: int(label == name)
                for name in ("low_high", "high_low", "balanced")
            }
            one_hot_residuals[f"{left},{right}"] = sum(indicators.values()) - 1

    modes = tuple(range(-4, 5))
    left_symbols = {mode: sp.Symbol(f"a_{mode + 4}") for mode in modes}
    right_symbols = {mode: sp.Symbol(f"u_{mode + 4}") for mode in modes}
    bony_residuals: dict[str, str] = {}
    for output in range(-8, 9):
        product = sp.Integer(0)
        low_high = sp.Integer(0)
        high_low = sp.Integer(0)
        balanced = sp.Integer(0)
        for left in modes:
            right = output - left
            if right not in right_symbols:
                continue
            term = left_symbols[left] * right_symbols[right]
            product += term
            label = _interaction(left, right, gap)
            if label == "low_high":
                low_high += term
            elif label == "high_low":
                high_low += term
            else:
                balanced += term
        bony_residuals[str(output)] = str(
            sp.expand(product - low_high - high_low - balanced)
        )

    y_symbols = sp.symbols("y0:8", real=True, finite=True)
    epsilon = sp.Symbol("epsilon", real=True, finite=True)
    coefficients = sp.symbols("f0:5", real=True, finite=True)
    jet = sum(
        y_symbols[order] * epsilon**order / factorial(order)
        for order in range(8)
    )
    argument = sp.Symbol("argument")
    nonlinear_map = sum(
        coefficients[order] * argument**order for order in range(5)
    )
    differentiated = sp.expand(
        sp.diff(nonlinear_map.subs(argument, jet), epsilon, 7).subs(epsilon, 0)
    )
    jacobian = sp.diff(nonlinear_map, argument).subs(argument, y_symbols[0])
    principal_top_jet = sp.expand(jacobian * y_symbols[7])
    tame_remainder = sp.expand(differentiated - principal_top_jet)
    top_jet_remainder_residual = sp.expand(sp.diff(tame_remainder, y_symbols[7]))
    omitted_principal_residual = sp.expand(sp.diff(differentiated, y_symbols[7]))

    naive_high_level = 6
    naive_low_level = 0
    naive_class = _interaction(naive_high_level, naive_low_level, gap)
    corrupted_class = "low_high"
    corrupted_classification_rejected = naive_class != corrupted_class

    passed = bool(
        set(one_hot_residuals.values()) == {0}
        and sum(classification_counts.values()) == len(levels) ** 2
        and set(bony_residuals.values()) == {"0"}
        and top_jet_remainder_residual == 0
        and omitted_principal_residual == jacobian
        and omitted_principal_residual != 0
        and naive_class == "high_low"
        and corrupted_classification_rejected
    )
    return passed, {
        "control": "exact finite Bony decomposition and top-jet good unknown",
        "dyadic_interaction_partition": {
            "gap": gap,
            "low_high": "ell_left<=ell_right-4",
            "high_low": "ell_right<=ell_left-4",
            "balanced": "|ell_left-ell_right|<=3",
            "level_range": [levels[0], levels[-1]],
            "classification_counts": classification_counts,
            "one_hot_residuals_all_zero": set(one_hot_residuals.values()) == {0},
            "one_hot_residual_count": len(one_hot_residuals),
        },
        "finite_Fourier_Bony_identity": {
            "identity": "a*u=T_a u+T_u a+R(a,u)",
            "mode_range": [modes[0], modes[-1]],
            "output_residuals": bony_residuals,
        },
        "top_derivative_isolation": {
            "nonlinear_test_map": str(nonlinear_map),
            "derivative_order": 7,
            "identity": "D^7 F(y)=DF(y)D^7 y+R_7(y,...,D^6 y)",
            "principal_top_jet": str(principal_top_jet),
            "remainder": str(tame_remainder),
            "D_y7_remainder_residual": str(top_jet_remainder_residual),
            "good_unknown": (
                "V_j=Delta_j U, deltaY_j=J(D)V_j; the principal high-frequency "
                "term is T_(D_Y E(Y)) deltaY_j"
            ),
        },
        "naive_commutator_negative": {
            "interaction": [naive_high_level, naive_low_level],
            "exact_classification": naive_class,
            "corruption": "classify the high-coefficient/low-state pair as T_a partial u",
            "corrupted_classification": corrupted_class,
            "omitted_principal_residual": str(omitted_principal_residual),
            "rejected": corrupted_classification_rejected,
        },
        "actual_system_binding_contract": {
            "coordinate_atom_variation": "deltaY=J_153x55(D) deltaU",
            "solved_evolution_linearization": (
                "D_Y E_55(Y) J_153x55(xi)"
            ),
            "required_identity": (
                "D_Y E_55(Y) J_153x55(xi)=i P55(Y,xi), componentwise on the tube"
            ),
            "required_evidence": (
                "an exact component tensor or content hash shared by the solved-source "
                "linearization and the physical 55-state pencil"
            ),
        },
        "passed": passed,
        "scope": (
            "The generic paraproduct and top-derivative construction is exact. Candidate "
            "H7 closure additionally requires the actual solved-source Jacobian to be "
            "identified componentwise with P55."
        ),
    }


def _certify_candidate(
    dyadic: dict[str, Any],
    solved_source: dict[str, Any],
    evolution: dict[str, Any],
    first_order: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(dyadic.get("candidate_id"))
    others = (solved_source, evolution, first_order)
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != dyadic.get("coefficients")
        for item in others
    ):
        raise QuarticParadifferentialGoodUnknownError("candidate identity mismatch")
    expected_statuses = (
        "pass_H7_dyadic_partition_and_shell_local_commutator_framework",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
        "pass_full_55_state_degree_one_evolution_symbol_C4_bounds",
        "pass_exact_55_variable_principal_first_order_reduction",
    )
    if tuple(
        item.get("status")
        for item in (dyadic, solved_source, evolution, first_order)
    ) != expected_statuses:
        raise QuarticParadifferentialGoodUnknownError(
            "candidate prerequisite status mismatch"
        )
    source_hash = first_order.get("source_spatial_block_sha256")
    if evolution.get("exact_reduction_provenance", {}).get(
        "source_spatial_block_sha256"
    ) != source_hash:
        raise QuarticParadifferentialGoodUnknownError(
            "physical-pencil provenance mismatch"
        )
    if dyadic.get("physical_pencil", {}).get(
        "source_spatial_block_sha256"
    ) != source_hash:
        raise QuarticParadifferentialGoodUnknownError(
            "dyadic-pencil provenance mismatch"
        )
    frechet = solved_source.get("solved_source_Frechet_derivatives", {})
    if frechet.get("orders") != [0, 1, 2, 3, 4]:
        raise QuarticParadifferentialGoodUnknownError(
            "solved-source Frechet hierarchy mismatch"
        )
    source_linearization_hash = solved_source.get(
        "source_to_P55_component_linearization_sha256"
    )
    pencil_linearization_hash = evolution.get(
        "source_to_P55_component_linearization_sha256"
    )
    component_binding_closed = bool(
        source_linearization_hash
        and pencil_linearization_hash
        and source_linearization_hash == pencil_linearization_hash
    )
    return {
        "schema_version": (
            "sigma-quartic-paradifferential-good-unknown-certificate-1.0"
        ),
        "status": "audit_paradifferential_good_unknown_binding_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": dyadic.get("coefficients"),
        "physical_pencil_provenance": {
            "source_spatial_block_sha256": source_hash,
            "state_dimension": evolution["exact_reduction_provenance"][
                "state_dimension"
            ],
            "exactly_linear_in_xi": dyadic["physical_pencil"][
                "exactly_linear_in_xi"
            ],
        },
        "paradifferential_good_unknown": {
            "state": "V_j=Delta_j U in C55",
            "coordinate_atom_variation": "deltaY_j=J_153x55(D)V_j",
            "candidate_principal": "T_(D_Y E_55(Y)) deltaY_j",
            "target_physical_operator": "i T_(P55(Y,xi)) V_j",
            "Bony_high_low_interaction_absorbed_generically": True,
        },
        "solved_source_Frechet_orders_available": frechet["orders"],
        "source_to_physical_pencil_component_binding": {
            "source_hash": source_linearization_hash,
            "pencil_hash": pencil_linearization_hash,
            "closed": component_binding_closed,
            "missing_evidence": (
                None
                if component_binding_closed
                else (
                    "the source artifact supplies norm envelopes but no componentwise "
                    "55x153 Jacobian tensor/hash equated to P55"
                )
            ),
        },
        "naive_H6_to_H7_commutator_rejected": dyadic[
            "full_H7_commutator_closed"
        ]
        is False,
        "H7_derivative_loss_resolved": component_binding_closed,
        "global_dyadic_summation_applied": False,
        "required_next_gate": (
            "emit_and_verify_componentwise_DY_E55_times_J_equals_iP55_then_bound_"
            "the_paralinearization_remainder"
        ),
        "scope": (
            "The actual 55-state and solved-source provenance are joined, and the generic "
            "good-unknown construction is executable. The decisive componentwise source-"
            "Jacobian-to-pencil identity is absent, so H7 closure and summation remain "
            "fail-closed."
        ),
    }


def run_quartic_paradifferential_good_unknown_campaign(
    dyadic_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticParadifferentialGoodUnknownError(
                "unsupported campaign schema_version"
            )
        campaigns = (
            dyadic_campaign,
            solved_source_campaign,
            evolution_campaign,
            first_order_campaign,
        )
        expected_statuses = (
            "pass_all_12_H7_dyadic_local_frameworks_global_commutator_fail_closed",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticParadifferentialGoodUnknownError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticParadifferentialGoodUnknownError(
                "campaign content hash mismatch"
            )
        if dyadic_campaign.get("upstream_sha256", {}).get(
            "evolution"
        ) != evolution_campaign.get("content_sha256"):
            raise QuarticParadifferentialGoodUnknownError(
                "dyadic-evolution provenance mismatch"
            )
        if dyadic_campaign.get("upstream_sha256", {}).get(
            "first_order"
        ) != first_order_campaign.get("content_sha256"):
            raise QuarticParadifferentialGoodUnknownError(
                "dyadic-first-order provenance mismatch"
            )
        if (
            int(config["spatial_dimension"]) != 3
            or int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["state_sobolev_order"]) != 7
            or int(config["coefficient_sobolev_order"]) != 6
            or int(config["paraproduct_gap"]) != 4
        ):
            raise QuarticParadifferentialGoodUnknownError(
                "unsupported good-unknown regularity contract"
            )
        if bool(config.get("declare_global_H7_closed", False)):
            raise QuarticParadifferentialGoodUnknownError(
                "global H7 closure cannot be declared before the component binding"
            )
        generic_passed, generic = generic_paradifferential_good_unknown_control()
        if not generic_passed:
            raise QuarticParadifferentialGoodUnknownError(
                "generic good-unknown control failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticParadifferentialGoodUnknownError("candidate-set mismatch")
        certificates = [
            _certify_candidate(*(records[candidate_id] for records in maps))
            for candidate_id in sorted(candidate_ids)
        ]
        closed = sum(
            int(item["H7_derivative_loss_resolved"]) for item in certificates
        )
        if closed and closed != len(certificates):
            raise QuarticParadifferentialGoodUnknownError(
                "partial candidate binding is not a universal campaign closure"
            )
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_paradifferential_good_unknown_audits_"
                "component_binding_fail_closed"
                if closed == 0
                else "pass_all_12_H7_paradifferential_good_unknown_closures"
            ),
            "errors": [],
            "upstream_sha256": {
                "dyadic_localization": dyadic_campaign.get("content_sha256"),
                "solved_source": solved_source_campaign.get("content_sha256"),
                "evolution": evolution_campaign.get("content_sha256"),
                "first_order": first_order_campaign.get("content_sha256"),
            },
            "config_sha256": hashlib.sha256(_canonical_json(config).encode()).hexdigest(),
            "generic_paradifferential_good_unknown_control": generic,
            "counts": {
                "selected": len(certificates),
                "good_unknown_frameworks_passed": len(certificates),
                "component_source_pencil_bindings_closed": closed,
                "global_H7_summations_applied": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates are attached to an executable Bony/good-unknown "
                "framework. None yet has the componentwise solved-source Jacobian identity "
                "needed to replace the rejected naive H6-to-H7 commutator."
            ),
            "scope": certificates[0]["scope"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticParadifferentialGoodUnknownError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "good_unknown_frameworks_passed": 0,
                "component_source_pencil_bindings_closed": 0,
                "global_H7_summations_applied": 0,
                "rejected": 0,
            },
        }
    return {
        **body,
        "content_sha256": hashlib.sha256(_canonical_json(body).encode()).hexdigest(),
    }


def write_quartic_paradifferential_good_unknown_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
