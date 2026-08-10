from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any

from .action_ir import compile_action_spec
from .formal_backend import load_field_contract
from .high_throughput import (
    build_basis,
    candidate_id,
    correction_expression,
    decode_ordinal,
)
from .promotion_orchestrator import ELIGIBILITY
from .q_operator_ir import compile_q_operator_ir
from .static_dictionary import _parse_generator_expression, compile_static_dictionary_ir

CAMPAIGN_SCHEMA = "sigma-production-covariant-provenance-campaign-1.0"
PROVENANCE_SCHEMA = "sigma-covariant-action-provenance-1.0"
SUPPORTED_LINEAR_TERM_IDS = {
    3: "AETHER_Q1",
    7: "AETHER_X_SQRT1P",
}
NONLINEAR_Q_TERM_IDS = {15, 36}


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


def _validate_candidate(candidate: dict[str, Any], generator: dict[str, Any]) -> dict[str, Any]:
    if candidate.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("candidate eligibility is not fail-closed")
    decoded = decode_ordinal(
        int(generator["basis_count"]),
        int(generator["max_action_terms"]),
        int(candidate["ordinal"]),
    )
    basis = build_basis(int(generator["basis_count"]))
    expected_id = candidate_id(str(generator["protocol_version"]), decoded)
    expected_expression = correction_expression(decoded, basis)
    if (
        candidate.get("candidate_id") != expected_id
        or candidate.get("term_ids") != list(decoded["term_ids"])
        or candidate.get("signs") != list(decoded["signs"])
        or candidate.get("correction_expression") != expected_expression
    ):
        raise ValueError("candidate differs from its generator ordinal identity")
    return decoded


def _action_spec(
    candidate: dict[str, Any], decomposition: dict[str, int], source_sha256: str
) -> dict[str, Any]:
    coefficient = "epsilon*M_Pl^2*a_sigma^2"
    return {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu", "u_mu", "lambda_u"],
        "matter_metric": "g_mu_nu",
        "terms": ["EH_R", *sorted(decomposition), "UNIT_VECTOR_CONSTRAINT"],
        "coefficients": {
            term_id: coefficient if sign > 0 else f"-{coefficient}"
            for term_id, sign in sorted(decomposition.items())
        },
        "universal_constants": ["M_Pl", "L_sigma", "a_sigma", "epsilon"],
        "parameter_domain": {
            "positive": ["M_Pl", "L_sigma", "a_sigma", "epsilon"]
        },
        "static_dictionary_status": "derived",
        "generator_origin": {
            "family_id": str(candidate["candidate_id"]),
            "ordinal": int(candidate["ordinal"]),
            "correction_expression": str(candidate["correction_expression"]),
            "pareto_front": -1,
            "source_priority_sha256": source_sha256,
        },
    }


def map_candidate_to_covariant_action(
    candidate: dict[str, Any],
    generator: dict[str, Any],
    grammar: dict[str, Any],
    field_contract: dict[str, Any],
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Build an exact typed action map or give a fail-closed reason why none is known."""

    decoded = _validate_candidate(candidate, generator)
    expression = str(candidate["correction_expression"])
    parsed = _parse_generator_expression(expression)
    used_symbols = {str(symbol) for symbol in parsed.free_symbols}
    identity = {
        "candidate_id": str(candidate["candidate_id"]),
        "ordinal": int(candidate["ordinal"]),
        "correction_expression": expression,
        "candidate_payload_sha256": _sha(candidate),
        "term_ids": list(decoded["term_ids"]),
        "signs": list(decoded["signs"]),
    }
    if "z" in used_symbols:
        return {
            **identity,
            "decision": "reject",
            "reason": "forbidden_baryonic_action_atom",
            "data_eligibility": dict(ELIGIBILITY),
        }
    term_ids = set(decoded["term_ids"])
    blockers: list[str] = []
    nonlinear = sorted(term_ids & NONLINEAR_Q_TERM_IDS)
    if nonlinear:
        blockers.append("nonlinear_q_power_requires_separate_formal_derivation")
    unsupported = sorted(term_ids - set(SUPPORTED_LINEAR_TERM_IDS))
    if unsupported:
        blockers.append("unsupported_generator_atom_in_covariant_action_dsl")
    if 3 not in term_ids:
        blockers.append("missing_linear_projected_aether_q_anchor")
    if blockers:
        return {
            **identity,
            "decision": "blocked",
            "blockers": sorted(set(blockers)),
            "unsupported_term_ids": unsupported,
            "nonlinear_q_term_ids": nonlinear,
            "data_eligibility": dict(ELIGIBILITY),
        }
    decomposition = {
        SUPPORTED_LINEAR_TERM_IDS[term_id]: sign
        for term_id, sign in zip(decoded["term_ids"], decoded["signs"], strict=True)
    }
    spec = _action_spec(candidate, decomposition, source_sha256)
    action_ir = compile_action_spec(spec, grammar, field_contract)
    static_ir = compile_static_dictionary_ir(action_ir, expression)
    q_ir = compile_q_operator_ir(action_ir)
    q_dictionary = static_ir.get("legacy_generator_dictionary", {}).get("q", {})
    if (
        not action_ir.get("valid")
        or static_ir.get("status") != "pass"
        or q_dictionary.get("status") != "derived_and_generator_matched"
        or q_dictionary.get("exact_shape_match") is not True
        or static_ir.get("universal_matter_coupling_preserved") is not True
    ):
        return {
            **identity,
            "decision": "blocked",
            "blockers": ["exact_static_covariant_equivalence_not_certified"],
            "action_errors": list(action_ir.get("errors", [])),
            "data_eligibility": dict(ELIGIBILITY),
        }
    provenance_body = {
        "schema_version": PROVENANCE_SCHEMA,
        **identity,
        "source_sha256": source_sha256,
        "basis_decomposition": dict(sorted(decomposition.items())),
        "action_spec_sha256": _sha(spec),
        "input_action_sha256": action_ir["content_sha256"],
        "static_dictionary_content_sha256": static_ir["content_sha256"],
        "q_operator_content_sha256": q_ir["content_sha256"],
        "exact_static_shape_match": True,
        "universal_matter_coupling_preserved": True,
        "data_eligibility": dict(ELIGIBILITY),
    }
    provenance = {
        **provenance_body,
        "provenance_binding_sha256": _sha(provenance_body),
    }
    return {
        **identity,
        "decision": "mapped",
        "covariant_action_provenance": provenance,
        "action_spec": spec,
        "formal_preflight": {
            "decision": (
                "reject_higher_jet_regularity"
                if q_ir.get("status") == "reject"
                else "formal_backend_queue"
            ),
            "q_operator_status": q_ir.get("status"),
            "q_operator_conclusion": q_ir.get("conclusion"),
        },
        "scope": (
            "exact static generator-to-typed-covariant-action equivalence only; mapping is "
            "not ADM/Dirac health, observational support, novelty, or theory promotion"
        ),
        "data_eligibility": dict(ELIGIBILITY),
    }


def production_blocked_candidates(database: str | Path) -> list[dict[str, Any]]:
    path = Path(database).resolve()
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = connection.execute(
            "SELECT c.payload_json FROM candidates c JOIN candidate_stages s "
            "USING(candidate_id) WHERE s.stage_name='covariant_symbolic_health' "
            "AND s.state='blocked' ORDER BY c.ordinal"
        ).fetchall()
    finally:
        connection.close()
    return [json.loads(row[0]) for row in rows]


def run_production_covariant_provenance_campaign(
    campaign_config: dict[str, Any],
    summary_path: str | Path,
    database_path: str | Path,
    generator_path: str | Path,
    grammar_path: str | Path,
    field_contract_path: str | Path,
) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    database_path = Path(database_path).resolve()
    generator_path = Path(generator_path).resolve()
    if _file_sha(summary_path) != campaign_config["source_summary_file_sha256"]:
        raise ValueError("production summary file hash mismatch")
    if _file_sha(database_path) != campaign_config["source_database_file_sha256"]:
        raise ValueError("production promotion database hash mismatch")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("production summary eligibility is not fail-closed")
    expected = int(campaign_config["expected_covariant_lift_blocked"])
    if (
        int(summary.get("promotion", {}).get("blocked", -1)) != expected
        or int(summary.get("candidate_count", -1))
        != int(campaign_config["expected_candidate_count"])
    ):
        raise ValueError("production summary counts differ from campaign contract")
    candidates = production_blocked_candidates(database_path)
    if len(candidates) != expected:
        raise ValueError("production database blocked count differs from summary")
    generator = json.loads(generator_path.read_text(encoding="utf-8"))
    grammar = json.loads(Path(grammar_path).read_text(encoding="utf-8"))
    contract = load_field_contract(field_contract_path)
    source_sha = campaign_config["source_summary_file_sha256"]
    records = [
        map_candidate_to_covariant_action(
            candidate,
            generator,
            grammar,
            contract,
            source_sha256=source_sha,
        )
        for candidate in candidates
    ]
    decisions = Counter(record["decision"] for record in records)
    preflight = Counter(
        record["formal_preflight"]["decision"]
        for record in records
        if record["decision"] == "mapped"
    )
    blocker_counts: Counter[str] = Counter()
    for record in records:
        for blocker in record.get("blockers", []):
            blocker_counts[blocker] += 1
        if record["decision"] == "reject":
            blocker_counts[str(record["reason"])] += 1
    input_identities = [
        {
            "candidate_id": record["candidate_id"],
            "ordinal": record["ordinal"],
            "candidate_payload_sha256": record["candidate_payload_sha256"],
        }
        for record in records
    ]
    body = {
        "schema_version": CAMPAIGN_SCHEMA,
        "source_summary_file_sha256": _file_sha(summary_path),
        "source_database_file_sha256": _file_sha(database_path),
        "generator_config_file_sha256": _file_sha(generator_path),
        "input_candidate_count": len(records),
        "decision_counts": dict(sorted(decisions.items())),
        "unblocked_covariant_lift_count": decisions["mapped"],
        "formal_preflight_counts": dict(sorted(preflight.items())),
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "input_candidate_root_sha256": _sha(input_identities),
        "decision_root_sha256": _sha(records),
        "mapped_records": [
            record for record in records if record["decision"] == "mapped"
        ],
        "rejected_records": [
            record for record in records if record["decision"] == "reject"
        ],
        "policy": {
            "supported_linear_term_ids": {
                str(term_id): action_term
                for term_id, action_term in sorted(SUPPORTED_LINEAR_TERM_IDS.items())
            },
            "nonlinear_q_term_ids_fail_closed": sorted(NONLINEAR_Q_TERM_IDS),
            "z_forbidden": True,
            "observations_opened": False,
        },
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "An unblocked covariant lift proves only exact static equivalence to a typed action. "
            "Formal preflight rejection remains a rejection, and no observation was opened."
        ),
    }
    return {**body, "content_sha256": _sha(body)}
