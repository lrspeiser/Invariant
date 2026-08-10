from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .composite_covariant_lift_campaign import compile_composite_aether_action
from .composite_q_degenerate_formal_campaign import (
    _parse_typed_expression,
    _validate_lift,
    evaluate_zero_local_acceleration_family,
)
from .formal_backend import load_field_contract
from .production_covariant_provenance import (
    map_candidate_to_covariant_action,
    production_blocked_candidates,
)
from .promotion_orchestrator import ELIGIBILITY

SCHEMA_VERSION = "sigma-composite-negative-local-kinetic-campaign-1.0"
ADAPTER_SCHEMA = "sigma-composite-negative-local-kinetic-adapter-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": _sha(body)}


def evaluate_negative_local_kinetic_family(lift: dict[str, Any]) -> dict[str, Any]:
    """Reject exact F(X,Q) actions with a negative aligned local kinetic coefficient."""

    action, term = _validate_lift(lift)
    expression = _parse_typed_expression(str(term["typed_expression_srepr"]))
    symbols = {str(symbol): symbol for symbol in expression.free_symbols}
    x = symbols.get("X_a_u", sp.Symbol("X_a_u", real=True))
    q = symbols.get("Q_a_u", sp.Symbol("Q_a_u", real=True))
    origin = {x: 0, q: 0}
    f_x = sp.simplify(sp.diff(expression, x).subs(origin))
    f_q = sp.simplify(sp.diff(expression, q).subs(origin))
    identity = {
        "schema_version": ADAPTER_SCHEMA,
        "candidate_id": lift["candidate_id"],
        "ordinal": lift["ordinal"],
        "input_action_sha256": action["content_sha256"],
        "provenance_binding_sha256": lift["covariant_action_provenance"][
            "provenance_binding_sha256"
        ],
        "static_shape_certificate_sha256": lift["static_shape_certificate"][
            "content_sha256"
        ],
        "linearized_coefficients": {
            "F_X_at_origin": str(f_x),
            "F_Q_at_origin": str(f_q),
        },
    }
    if f_x.is_negative is not True:
        return {
            **identity,
            "decision": "blocked",
            "blocker": "nonnegative_local_acceleration_coefficient_outside_bounded_family",
            "data_eligibility": dict(ELIGIBILITY),
        }
    if f_q.is_nonnegative is not True:
        return {
            **identity,
            "decision": "blocked",
            "blocker": "q_kinetic_coefficient_sign_not_certified_nonnegative",
            "data_eligibility": dict(ELIGIBILITY),
        }

    eta, x2, q2 = sp.symbols("eta X2 Q2", real=True)
    quadratic = sp.simplify(
        sp.diff(expression.subs({x: eta**2 * x2, q: eta**2 * q2}), eta, 2).subs(
            eta, 0
        )
        / 2
    )
    residual = sp.simplify(quadratic - f_x * x2 - f_q * q2)
    if residual != 0:
        raise ValueError("exact composite weak-field chain-rule certificate failed")

    wave_number, length_scale = sp.symbols("k L_sigma", nonnegative=True)
    normalized_hessian = sp.factor(2 * (f_x + f_q * length_scale**2 * wave_number**2))
    finite_rank_loss = f_q.is_positive is True
    if finite_rank_loss:
        critical_k_squared = sp.factor(-f_x / (f_q * length_scale**2))
        reason = "negative_low_frequency_kinetic_energy_and_finite_k_rank_loss"
    else:
        critical_k_squared = None
        reason = "negative_local_aether_kinetic_energy_without_constraint_removal"
    body = {
        **identity,
        "decision": "reject",
        "reason": reason,
        "covariant_variation_preflight": {
            "status": "bounded_chain_rule_derived",
            "fixed_metric_first_variation": "delta F=F_X delta X_a_u+F_Q delta Q_a_u",
            "metric_variation_status": "unresolved_not_needed_for_decisive_negative_gate",
            "claim_scope": "exact chain rule for the hash-bound typed covariant scalar",
        },
        "weak_field_certificate": {
            "background": "Minkowski, u^mu=(1,0,0,0), X_a_u=Q_a_u=0",
            "quadratic_expression": str(quadratic),
            "chain_rule_residual": str(residual),
            "positive_overall_prefactor": "epsilon*M_Pl^2 with epsilon>0 and M_Pl>0",
        },
        "adm_kinetic_preflight": {
            "status": "reject",
            "normalized_spatial_vector_velocity_hessian": str(normalized_hessian),
            "homogeneous_k_zero_eigenvalue": str(2 * f_x),
            "negative_directions_at_k_zero": 3,
            "critical_k_squared": (
                str(critical_k_squared) if critical_k_squared is not None else None
            ),
            "rank_at_generic_k": 3,
            "rank_at_critical_k": 0 if finite_rank_loss else None,
        },
        "dirac_preflight": {
            "status": "reject",
            "homogeneous_kinetic_primary_constraint_count": 0,
            "negative_spatial_vector_directions_not_projected_out": 3,
            "constraint_rank_uniform_in_k": not finite_rank_loss,
            "critical_stratum_primary_constraint_seed_count": (
                3 if finite_rank_loss else 0
            ),
            "claim_scope": (
                "aligned spatial Aether tangent directions satisfying the linearized unit "
                "constraint; full nonlinear constraint algebra not claimed"
            ),
        },
        "principal_preflight": {
            "status": "reject" if finite_rank_loss else "unresolved",
            "normalized_time_block": str(f_x + f_q * length_scale**2 * wave_number**2),
            "failure": (
                "time block vanishes at a finite real spatial wave number"
                if finite_rank_loss
                else "energy rejection is decisive; full coupled characteristic roots not derived"
            ),
        },
        "formal_pass": False,
        "solar_bundle_outcome": {
            "decision": "blocked",
            "blocker": "upstream_formal_rejected",
            "bundle_generated": False,
        },
        "observational_data_opened": False,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The exact covariant action has a negative kinetic quadratic form on allowed "
            "unit-Aether tangent perturbations. Static shape is not used as formal evidence."
        ),
    }
    return _content(body)


def run_composite_negative_local_kinetic_campaign(
    config: dict[str, Any],
    database_path: str | Path,
    generator_path: str | Path,
    grammar_path: str | Path,
    field_contract_path: str | Path,
    static_dictionary_path: str | Path,
    prior_campaign_path: str | Path,
) -> dict[str, Any]:
    paths = {
        "database": Path(database_path).resolve(),
        "generator": Path(generator_path).resolve(),
        "grammar": Path(grammar_path).resolve(),
        "field_contract": Path(field_contract_path).resolve(),
        "static_dictionary": Path(static_dictionary_path).resolve(),
        "prior_campaign": Path(prior_campaign_path).resolve(),
    }
    for name, path in paths.items():
        if _file_sha(path) != config[f"{name}_file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    prior = json.loads(paths["prior_campaign"].read_text(encoding="utf-8"))
    prior_body = {key: value for key, value in prior.items() if key != "content_sha256"}
    if prior.get("content_sha256") != _sha(prior_body) or prior.get(
        "content_sha256"
    ) != config["prior_campaign_content_sha256"]:
        raise ValueError("prior formal campaign content binding mismatch")
    if prior.get("remaining_formal_blocked_count") != int(
        config["expected_input_candidate_count"]
    ):
        raise ValueError("prior formal campaign blocked count mismatch")

    generator = json.loads(paths["generator"].read_text(encoding="utf-8"))
    grammar = json.loads(paths["grammar"].read_text(encoding="utf-8"))
    field_contract = load_field_contract(paths["field_contract"])
    static_dictionary = json.loads(paths["static_dictionary"].read_text(encoding="utf-8"))
    if static_dictionary.get("content_sha256") != config[
        "static_dictionary_content_sha256"
    ]:
        raise ValueError("static dictionary content hash mismatch")

    outcomes: list[dict[str, Any]] = []
    candidates = production_blocked_candidates(paths["database"])
    for candidate in candidates:
        existing = map_candidate_to_covariant_action(
            candidate,
            generator,
            grammar,
            field_contract,
            source_sha256=config["source_summary_file_sha256"],
        )
        if existing["decision"] == "mapped":
            continue
        lift = compile_composite_aether_action(
            candidate,
            field_contract,
            field_contract_file_sha256=config["field_contract_file_sha256"],
            static_dictionary_file_sha256=config["static_dictionary_file_sha256"],
            static_dictionary_content_sha256=config[
                "static_dictionary_content_sha256"
            ],
            source_sha256=config["source_summary_file_sha256"],
        )
        prior_outcome = evaluate_zero_local_acceleration_family(lift)
        if prior_outcome["decision"] != "blocked":
            continue
        outcomes.append(evaluate_negative_local_kinetic_family(lift))
    if len(outcomes) != int(config["expected_input_candidate_count"]):
        raise ValueError("remaining composite candidate count mismatch")

    decisions = Counter(outcome["decision"] for outcome in outcomes)
    reasons = Counter(
        outcome.get("reason", outcome.get("blocker", "unknown")) for outcome in outcomes
    )
    rejected = [outcome for outcome in outcomes if outcome["decision"] == "reject"]
    sampled = [
        rejected[0],
        next(
            item
            for item in rejected
            if item["linearized_coefficients"]["F_Q_at_origin"] == "0"
        ),
        rejected[-1],
    ]
    identities = [
        {
            "candidate_id": outcome["candidate_id"],
            "ordinal": outcome["ordinal"],
            "decision": outcome["decision"],
            "reason_or_blocker": outcome.get("reason", outcome.get("blocker")),
            "input_action_sha256": outcome["input_action_sha256"],
            "provenance_binding_sha256": outcome["provenance_binding_sha256"],
            "formal_evidence_sha256": outcome.get("content_sha256"),
        }
        for outcome in outcomes
    ]
    body = {
        "schema_version": SCHEMA_VERSION,
        "input_candidate_count": len(outcomes),
        "decision_counts": dict(sorted(decisions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "bounded_family_reject_count": len(rejected),
        "remaining_formal_blocked_count": decisions.get("blocked", 0),
        "formal_pass_count": decisions.get("pass", 0),
        "solar_bundle_generated_count": 0,
        "candidate_evidence_root_sha256": _sha(identities),
        "prior_campaign_content_sha256": prior["content_sha256"],
        "sample_rejections": [
            {
                "candidate_id": item["candidate_id"],
                "ordinal": item["ordinal"],
                "reason": item["reason"],
                "input_action_sha256": item["input_action_sha256"],
                "provenance_binding_sha256": item["provenance_binding_sha256"],
                "static_shape_certificate_sha256": item[
                    "static_shape_certificate_sha256"
                ],
                "linearized_coefficients": item["linearized_coefficients"],
                "adm_kinetic_preflight": item["adm_kinetic_preflight"],
                "dirac_preflight": item["dirac_preflight"],
                "principal_preflight": item["principal_preflight"],
                "formal_evidence_sha256": item["content_sha256"],
            }
            for item in sampled
        ],
        "source_database_file_sha256": _file_sha(paths["database"]),
        "source_summary_file_sha256": config["source_summary_file_sha256"],
        "policy": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_calls": False,
            "static_shape_treated_as_formal_proof": False,
        },
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "This bounded campaign rejects only exact hash-bound F(X,Q) actions with "
            "F_X(0,0)<0 and a certified nonnegative F_Q(0,0). Other actions remain blocked."
        ),
    }
    return _content(body)
