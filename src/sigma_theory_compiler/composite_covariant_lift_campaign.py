from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sympy as sp

from .formal_backend import load_field_contract
from .production_covariant_provenance import (
    map_candidate_to_covariant_action,
    production_blocked_candidates,
)
from .promotion_orchestrator import ELIGIBILITY
from .static_dictionary import _parse_generator_expression

CAMPAIGN_SCHEMA = "sigma-composite-covariant-lift-campaign-1.0"
ACTION_SCHEMA = "sigma-composite-aether-action-ir-1.0"
CERTIFICATE_SCHEMA = "sigma-universal-static-shape-certificate-1.0"
PROVENANCE_SCHEMA = "sigma-composite-covariant-action-provenance-1.0"


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


def compile_composite_aether_action(
    candidate: dict[str, Any],
    field_contract: dict[str, Any],
    *,
    field_contract_file_sha256: str,
    static_dictionary_file_sha256: str,
    static_dictionary_content_sha256: str,
    source_sha256: str,
) -> dict[str, Any]:
    """Lift F(x,q) to F(X_a_u,Q_a_u) with an exact universal static certificate."""

    if candidate.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("candidate eligibility is not fail-closed")
    expression = str(candidate["correction_expression"])
    static_expression = _parse_generator_expression(expression)
    used = {str(symbol) for symbol in static_expression.free_symbols}
    if "z" in used:
        return {
            "decision": "reject",
            "candidate_id": str(candidate["candidate_id"]),
            "reason": "forbidden_baryonic_action_atom",
            "data_eligibility": dict(ELIGIBILITY),
        }
    if not used or not used <= {"x", "q"} or "q" not in used:
        return {
            "decision": "blocked",
            "candidate_id": str(candidate["candidate_id"]),
            "blocker": "expression_is_not_a_q_aether_composite",
            "data_eligibility": dict(ELIGIBILITY),
        }
    invariants = {
        item["id"]: item
        for item in field_contract["generator_invariants"]
        if item["id"] in {"X_a_u", "Q_a_u"}
    }
    if set(invariants) != {"X_a_u", "Q_a_u"}:
        raise ValueError("field contract lacks typed Aether X/Q invariants")
    x, q = sp.symbols("x q", real=True)
    covariant_x, covariant_q = sp.symbols("X_a_u Q_a_u", real=True)
    covariant_expression = sp.factor(
        static_expression.xreplace({x: covariant_x, q: covariant_q})
    )
    recovered = sp.factor(
        covariant_expression.xreplace({covariant_x: x, covariant_q: q})
    )
    residual = sp.factor(recovered - static_expression)
    if residual != 0:
        raise ValueError("composite covariant substitution has nonzero static residual")
    certificate_body = {
        "schema_version": CERTIFICATE_SCHEMA,
        "generator_expression": expression,
        "canonical_static_expression": str(static_expression),
        "canonical_covariant_expression": str(covariant_expression),
        "static_substitution": {"X_a_u": "x", "Q_a_u": "q"},
        "universal_shape_residual": str(residual),
        "identity_scope": "all x,q where the expression is defined",
        "forbidden_symbols_absent": sorted(used - {"x", "q"}) == [],
        "field_contract_file_sha256": field_contract_file_sha256,
        "static_dictionary_file_sha256": static_dictionary_file_sha256,
        "static_dictionary_content_sha256": static_dictionary_content_sha256,
    }
    certificate = _content(certificate_body)
    used_covariant_invariants = [
        invariant_id
        for invariant_id in ("Q_a_u", "X_a_u")
        if invariant_id in str(covariant_expression)
    ]
    base_invariants = [
        {
            "id": invariant_id,
            "definition": invariants[invariant_id]["definition"],
            "maximum_derivatives_per_field": invariants[invariant_id][
                "maximum_derivatives_per_field"
            ],
            "formal_status": invariants[invariant_id]["formal_status"],
        }
        for invariant_id in used_covariant_invariants
    ]
    action_body = {
        "schema_version": ACTION_SCHEMA,
        "candidate_id": str(candidate["candidate_id"]),
        "ordinal": int(candidate["ordinal"]),
        "source_sha256": source_sha256,
        "source_candidate_payload_sha256": _sha(candidate),
        "source_role": "candidate",
        "fields": ["g_mu_nu", "u_mu", "lambda_u"],
        "matter_metric": "g_mu_nu",
        "universal_matter_coupling_preserved": True,
        "terms": [
            {
                "id": "EH_R",
                "role": "einstein_hilbert",
            },
            {
                "id": f"COMPOSITE_QX_{hashlib.sha256(expression.encode()).hexdigest()[:16]}",
                "role": "derived_dimensionless_aether_scalar",
                "density": (
                    "sqrt(-g)*epsilon*M_Pl^2*a_sigma^2*("
                    f"{covariant_expression})"
                ),
                "typed_expression_srepr": sp.srepr(covariant_expression),
                "base_invariants": base_invariants,
            },
            {
                "id": "UNIT_VECTOR_CONSTRAINT",
                "role": "unit_timelike_constraint",
            },
        ],
        "coefficient_domain": {
            "positive": ["M_Pl", "L_sigma", "a_sigma", "epsilon"]
        },
        "static_shape_certificate_content_sha256": certificate["content_sha256"],
        "observational_data_opened": False,
        "formal_scope": (
            "exact typed covariant scalar and static shape only; arbitrary composite "
            "variation, ADM/Dirac closure, principal symbol, and energy are unresolved"
        ),
    }
    action_ir = _content(action_body)
    provenance_body = {
        "schema_version": PROVENANCE_SCHEMA,
        "candidate_id": str(candidate["candidate_id"]),
        "ordinal": int(candidate["ordinal"]),
        "candidate_payload_sha256": _sha(candidate),
        "source_sha256": source_sha256,
        "input_action_sha256": action_ir["content_sha256"],
        "static_shape_certificate_sha256": certificate["content_sha256"],
        "field_contract_file_sha256": field_contract_file_sha256,
        "static_dictionary_file_sha256": static_dictionary_file_sha256,
        "static_dictionary_content_sha256": static_dictionary_content_sha256,
        "universal_shape_residual": "0",
        "universal_matter_coupling_preserved": True,
        "data_eligibility": dict(ELIGIBILITY),
    }
    provenance = {
        **provenance_body,
        "provenance_binding_sha256": _sha(provenance_body),
    }
    return {
        "decision": "mapped",
        "candidate_id": str(candidate["candidate_id"]),
        "ordinal": int(candidate["ordinal"]),
        "correction_expression": expression,
        "action_ir": action_ir,
        "static_shape_certificate": certificate,
        "covariant_action_provenance": provenance,
        "formal_outcome": {
            "decision": "blocked",
            "blocker": "missing_candidate_specific_adm_dirac_principal_adapter_for_composite_qx_action",
            "q_base_invariant_formal_status": invariants["Q_a_u"]["formal_status"],
        },
        "solar_bundle_outcome": {
            "decision": "blocked",
            "blocker": "awaiting_candidate_specific_formal_health_pass",
            "bundle_generated": False,
        },
        "galaxy_prediction_outcome": {
            "decision": "blocked",
            "blocker": "missing_candidate_direct_observable_prediction_bundle",
            "prediction_bundle_generated": False,
        },
        "data_eligibility": dict(ELIGIBILITY),
    }


def run_composite_covariant_lift_campaign(
    config: dict[str, Any],
    database_path: str | Path,
    generator_path: str | Path,
    grammar_path: str | Path,
    field_contract_path: str | Path,
    static_dictionary_path: str | Path,
) -> dict[str, Any]:
    database_path = Path(database_path).resolve()
    generator_path = Path(generator_path).resolve()
    grammar_path = Path(grammar_path).resolve()
    field_contract_path = Path(field_contract_path).resolve()
    static_dictionary_path = Path(static_dictionary_path).resolve()
    expected_hashes = {
        "database": database_path,
        "generator": generator_path,
        "grammar": grammar_path,
        "field_contract": field_contract_path,
        "static_dictionary": static_dictionary_path,
    }
    for key, path in expected_hashes.items():
        if _file_sha(path) != config[f"{key}_file_sha256"]:
            raise ValueError(f"{key} file hash mismatch")
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    grammar = json.loads(grammar_path.read_text(encoding="utf-8"))
    field_contract = load_field_contract(field_contract_path)
    static_dictionary = json.loads(static_dictionary_path.read_text(encoding="utf-8"))
    if static_dictionary.get("content_sha256") != config[
        "static_dictionary_content_sha256"
    ]:
        raise ValueError("static dictionary content hash mismatch")
    candidates = production_blocked_candidates(database_path)
    if len(candidates) != int(config["expected_candidate_count"]):
        raise ValueError("production blocked candidate count mismatch")
    records: list[dict[str, Any]] = []
    newly_mapped = 0
    for candidate in candidates:
        bounded = map_candidate_to_covariant_action(
            candidate,
            generator,
            grammar,
            field_contract,
            source_sha256=config["source_summary_file_sha256"],
        )
        if bounded["decision"] == "mapped":
            record = {
                "decision": "mapped_existing_typed_action",
                "candidate_id": bounded["candidate_id"],
                "ordinal": bounded["ordinal"],
                "correction_expression": bounded["correction_expression"],
                "covariant_action_provenance": bounded[
                    "covariant_action_provenance"
                ],
                "formal_outcome": {
                    "decision": "reject",
                    "reason": bounded["formal_preflight"]["q_operator_conclusion"],
                },
                "solar_bundle_outcome": {
                    "decision": "blocked",
                    "blocker": "upstream_formal_rejected",
                    "bundle_generated": False,
                },
                "galaxy_prediction_outcome": {
                    "decision": "blocked",
                    "blocker": "upstream_formal_rejected",
                    "prediction_bundle_generated": False,
                },
                "data_eligibility": dict(ELIGIBILITY),
            }
        else:
            record = compile_composite_aether_action(
                candidate,
                field_contract,
                field_contract_file_sha256=config["field_contract_file_sha256"],
                static_dictionary_file_sha256=config[
                    "static_dictionary_file_sha256"
                ],
                static_dictionary_content_sha256=config[
                    "static_dictionary_content_sha256"
                ],
                source_sha256=config["source_summary_file_sha256"],
            )
            if record["decision"] == "mapped":
                newly_mapped += 1
        records.append(record)
    decision_counts = Counter(record["decision"] for record in records)
    formal_counts = Counter(record["formal_outcome"]["decision"] for record in records)
    solar_counts = Counter(
        record["solar_bundle_outcome"]["decision"] for record in records
    )
    galaxy_counts = Counter(
        record["galaxy_prediction_outcome"]["decision"] for record in records
    )
    identities = [
        {
            "candidate_id": record["candidate_id"],
            "ordinal": record["ordinal"],
            "decision": record["decision"],
            "action_sha256": record["covariant_action_provenance"][
                "input_action_sha256"
            ],
            "provenance_binding_sha256": record["covariant_action_provenance"][
                "provenance_binding_sha256"
            ],
            "formal_outcome": record["formal_outcome"],
        }
        for record in records
    ]
    sample_records = []
    for record in (records[0], records[1], records[-1]):
        sample = {
            "candidate_id": record["candidate_id"],
            "ordinal": record["ordinal"],
            "correction_expression": record["correction_expression"],
            "decision": record["decision"],
            "covariant_action_provenance": record[
                "covariant_action_provenance"
            ],
            "formal_outcome": record["formal_outcome"],
            "solar_bundle_outcome": record["solar_bundle_outcome"],
            "galaxy_prediction_outcome": record["galaxy_prediction_outcome"],
        }
        if "static_shape_certificate" in record:
            sample["static_shape_certificate"] = record[
                "static_shape_certificate"
            ]
        sample_records.append(sample)
    body = {
        "schema_version": CAMPAIGN_SCHEMA,
        "input_candidate_count": len(records),
        "decision_counts": dict(sorted(decision_counts.items())),
        "newly_exact_composite_lift_count": newly_mapped,
        "total_exact_lift_count": sum(
            count for decision, count in decision_counts.items() if decision.startswith("mapped")
        ),
        "formal_outcome_counts": dict(sorted(formal_counts.items())),
        "solar_bundle_outcome_counts": dict(sorted(solar_counts.items())),
        "galaxy_prediction_outcome_counts": dict(sorted(galaxy_counts.items())),
        "candidate_provenance_root_sha256": _sha(identities),
        "sample_records": sample_records,
        "source_database_file_sha256": _file_sha(database_path),
        "source_summary_file_sha256": config["source_summary_file_sha256"],
        "policy": {
            "z_forbidden": True,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "observations_opened": False,
            "paid_llm_calls": False,
        },
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "Exact covariant shape is not formal health. No candidate reached Solar or galaxy "
            "evaluation because none passed candidate-specific formal health."
        ),
    }
    return _content(body)
