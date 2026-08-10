from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .composite_covariant_lift_campaign import compile_composite_aether_action
from .composite_negative_local_kinetic_campaign import (
    evaluate_negative_local_kinetic_family,
)
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

SCHEMA_VERSION = "sigma-composite-positive-qx-tilt-campaign-1.0"
ADAPTER_SCHEMA = "sigma-composite-positive-qx-tilt-adapter-1.0"


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


def evaluate_positive_qx_tilt_family(lift: dict[str, Any]) -> dict[str, Any]:
    """Reject positive X+Q quadratic actions by their exact generic-tilt roots."""

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
    if f_x.is_positive is not True or f_q.is_positive is not True:
        return {
            **identity,
            "decision": "blocked",
            "blocker": "strictly_positive_qx_coefficients_not_certified",
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

    omega, wave_number = sp.symbols("omega k", real=True)
    beta = sp.Symbol("beta", real=True, nonzero=True)
    gamma = sp.Symbol("gamma_beta", positive=True, real=True)
    length_scale = sp.Symbol("L_sigma", positive=True, real=True)
    k0 = f_x
    k2 = f_q * length_scale**2
    omega_rest = gamma * (omega - beta * wave_number)
    kappa_rest = gamma * (wave_number - beta * omega)
    lab_polynomial = sp.factor(omega_rest**2 * (k0 + k2 * kappa_rest**2))
    root_real = beta * wave_number
    imaginary_offset = sp.sqrt(k0 / k2) / (gamma * beta)
    root_plus = wave_number / beta + sp.I * imaginary_offset
    root_minus = wave_number / beta - sp.I * imaginary_offset
    root_residuals = {
        "real_double_root": str(sp.simplify(lab_polynomial.subs(omega, root_real))),
        "complex_plus": str(sp.simplify(lab_polynomial.subs(omega, root_plus))),
        "complex_minus": str(sp.simplify(lab_polynomial.subs(omega, root_minus))),
    }
    if set(root_residuals.values()) != {"0"}:
        raise ValueError("generic-tilt lab-frequency root certificate failed")

    sample_polynomial = sp.Poly(
        lab_polynomial.subs(
            {
                beta: sp.Rational(3, 5),
                gamma: sp.Rational(5, 4),
                wave_number: sp.Rational(7, 6),
                length_scale: 1,
            }
        ),
        omega,
    )
    sample_distinct_real_roots = int(
        sp.polys.polytools.count_roots(sample_polynomial, -sp.oo, sp.oo)
    )
    sample_real_root_multiplicity = 2
    sample_nonreal_roots = sample_polynomial.degree() - sample_real_root_multiplicity
    if (
        sample_polynomial.degree() != 4
        or sample_distinct_real_roots != 1
        or sample_nonreal_roots != 2
    ):
        raise ValueError("exact rational generic-tilt Sturm control failed")

    body = {
        **identity,
        "decision": "reject",
        "reason": "generic_tilt_lab_frequency_polynomial_has_nonreal_conjugate_pair",
        "covariant_variation_preflight": {
            "status": "bounded_chain_rule_derived",
            "fixed_metric_first_variation": "delta F=F_X delta X_a_u+F_Q delta Q_a_u",
            "metric_variation_status": "unresolved_not_needed_for_decisive_negative_gate",
            "claim_scope": "exact chain rule for the hash-bound typed covariant scalar",
        },
        "weak_field_certificate": {
            "background": "constant unit-Aether Minkowski background",
            "quadratic_expression": str(quadratic),
            "chain_rule_residual": str(residual),
            "rest_frame_operator": "Omega^2*(K0+K2*kappa^2)",
            "K0": str(k0),
            "K2": str(k2),
        },
        "adm_kinetic_preflight": {
            "status": "pass_necessary_condition_only",
            "normalized_rest_velocity_hessian": str(
                sp.factor(2 * (k0 + k2 * wave_number**2))
            ),
            "rank_for_all_real_k": 3,
            "positive_for_all_real_k": True,
        },
        "dirac_preflight": {
            "status": "unresolved_after_uniform_tangent_hessian",
            "spatial_vector_tangent_primary_constraint_count": 0,
            "full_metric_vector_constraint_closure_claimed": False,
        },
        "principal_preflight": {
            "status": "reject",
            "lab_polynomial": str(lab_polynomial),
            "parameter_domain": "K0>0, K2>0, gamma_beta>0, beta!=0",
            "roots": {
                "real_double": str(root_real),
                "complex_pair": (
                    "k/beta +/- I*sqrt(K0/K2)/(gamma_beta*beta)"
                ),
            },
            "root_substitution_residuals": root_residuals,
            "nonreal_root_count": 2,
            "exact_rational_sturm_control": {
                "beta": "3/5",
                "gamma_beta": "5/4",
                "k": "7/6",
                "degree": sample_polynomial.degree(),
                "distinct_real_root_count": sample_distinct_real_roots,
                "real_root_count_with_multiplicity": sample_real_root_multiplicity,
                "nonreal_root_count": sample_nonreal_roots,
            },
            "claim_scope": (
                "frozen-coefficient reduced quadratic Aether mode for generic tilted time "
                "covectors; a separately declared preferred-foliation Cauchy contract is absent"
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
            "Positive rest-frame kinetic rank is only a necessary condition. The exact tilted "
            "lab-frequency polynomial has a nonreal conjugate pair, so promotion is rejected."
        ),
    }
    return _content(body)


def run_composite_positive_qx_tilt_campaign(
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
    for candidate in production_blocked_candidates(paths["database"]):
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
        if evaluate_zero_local_acceleration_family(lift)["decision"] != "blocked":
            continue
        if evaluate_negative_local_kinetic_family(lift)["decision"] != "blocked":
            continue
        outcomes.append(evaluate_positive_qx_tilt_family(lift))
    if len(outcomes) != int(config["expected_input_candidate_count"]):
        raise ValueError("remaining composite candidate count mismatch")

    decisions = Counter(outcome["decision"] for outcome in outcomes)
    reasons = Counter(
        outcome.get("reason", outcome.get("blocker", "unknown")) for outcome in outcomes
    )
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
    samples = [outcomes[0], outcomes[len(outcomes) // 2], outcomes[-1]]
    body = {
        "schema_version": SCHEMA_VERSION,
        "input_candidate_count": len(outcomes),
        "decision_counts": dict(sorted(decisions.items())),
        "reason_counts": dict(sorted(reasons.items())),
        "formal_pass_count": decisions.get("pass", 0),
        "remaining_formal_blocked_count": decisions.get("blocked", 0),
        "solar_bundle_generated_count": 0,
        "candidate_evidence_root_sha256": _sha(identities),
        "prior_campaign_content_sha256": prior["content_sha256"],
        "sample_rejections": [
            {
                "candidate_id": item["candidate_id"],
                "ordinal": item["ordinal"],
                "input_action_sha256": item["input_action_sha256"],
                "provenance_binding_sha256": item["provenance_binding_sha256"],
                "linearized_coefficients": item["linearized_coefficients"],
                "adm_kinetic_preflight": item["adm_kinetic_preflight"],
                "dirac_preflight": item["dirac_preflight"],
                "principal_preflight": item["principal_preflight"],
                "formal_evidence_sha256": item["content_sha256"],
            }
            for item in samples
        ],
        "source_database_file_sha256": _file_sha(paths["database"]),
        "source_summary_file_sha256": config["source_summary_file_sha256"],
        "policy": {
            "observational_data_opened": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "paid_llm_calls": False,
            "partial_evidence_promoted": False,
            "static_shape_treated_as_formal_proof": False,
        },
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "The remaining positive-X, positive-Q actions pass only the aligned kinetic-rank "
            "preflight and are rejected by exact generic-tilt nonreal principal roots."
        ),
    }
    return _content(body)
