from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import sympy as sp

from .action_ir import compile_action_spec
from .q_operator_ir import compile_q_operator_ir
from .static_dictionary import _parse_generator_expression, compile_static_dictionary_ir

SCHEMA_VERSION = "sigma-covariant-export-1.0"

_BASIS = {
    "AETHER_Q1": "q",
    "AETHER_Q2": "q^2",
    "AETHER_Q3": "q^3",
    "AETHER_X_SQRT1P": "sqrt(1+x)-1",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _decompose(expression: str) -> dict[str, int] | None:
    target = _parse_generator_expression(expression)
    basis = {term_id: _parse_generator_expression(atom) for term_id, atom in _BASIS.items()}
    term_ids = sorted(basis)
    for signs in product((-1, 0, 1), repeat=len(term_ids)):
        if all(sign == 0 for sign in signs):
            continue
        represented = sum(
            sign * basis[term_id]
            for term_id, sign in zip(term_ids, signs, strict=True)
        )
        if sp.factor(target - represented) == 0:
            return {
                term_id: sign
                for term_id, sign in zip(term_ids, signs, strict=True)
                if sign
            }
    return None


def export_representable_covariant_candidates(
    priority_report: dict[str, Any],
    grammar: dict[str, Any],
    field_contract: dict[str, Any],
    *,
    source_priority_sha256: str,
) -> tuple[dict[str, Any], list[tuple[str, dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    specs: list[tuple[str, dict[str, Any]]] = []
    for candidate in priority_report.get("work_queue", []):
        decomposition = _decompose(candidate["correction_expression"])
        if decomposition is None or "AETHER_Q1" not in decomposition:
            continue
        coefficients = {
            term_id: (
                "epsilon*M_Pl^2*a_sigma^2"
                if sign > 0
                else "-epsilon*M_Pl^2*a_sigma^2"
            )
            for term_id, sign in decomposition.items()
        }
        spec = {
            "schema_version": "sigma-action-spec-1.0",
            "role": "candidate",
            "fields": ["g_mu_nu", "u_mu", "lambda_u"],
            "matter_metric": "g_mu_nu",
            "terms": ["EH_R", *sorted(decomposition), "UNIT_VECTOR_CONSTRAINT"],
            "coefficients": coefficients,
            "universal_constants": ["M_Pl", "L_sigma", "a_sigma", "epsilon"],
            "parameter_domain": {
                "positive": ["M_Pl", "L_sigma", "a_sigma", "epsilon"]
            },
            "static_dictionary_status": "derived",
            "generator_origin": {
                "family_id": candidate["family_id"],
                "ordinal": candidate["ordinal"],
                "correction_expression": candidate["correction_expression"],
                "pareto_front": candidate["pareto_front"],
                "source_priority_sha256": source_priority_sha256,
            },
        }
        action_ir = compile_action_spec(spec, grammar, field_contract)
        static_ir = compile_static_dictionary_ir(action_ir)
        q_ir = compile_q_operator_ir(action_ir)
        if not action_ir["valid"] or static_ir.get("status") != "pass":
            preflight = "reject_export_contract"
        elif q_ir["status"] == "reject":
            preflight = "reject_higher_jet_regularity"
        else:
            preflight = "formal_backend_queue"
        filename = f"{candidate['family_id']}.json"
        specs.append((filename, spec))
        records.append(
            {
                "family_id": candidate["family_id"],
                "ordinal": candidate["ordinal"],
                "pareto_front": candidate["pareto_front"],
                "correction_expression": candidate["correction_expression"],
                "basis_decomposition": decomposition,
                "action_valid": action_ir["valid"],
                "action_sha256": action_ir.get("content_sha256"),
                "static_dictionary_status": static_ir.get("status"),
                "exact_static_shape_match": static_ir.get("legacy_generator_dictionary", {})
                .get("q", {})
                .get("exact_shape_match"),
                "higher_jet_regularity_status": q_ir.get("status"),
                "higher_jet_conclusion": q_ir.get("conclusion"),
                "preflight_decision": preflight,
                "spec_file": filename,
            }
        )
    counts: dict[str, int] = {}
    for record in records:
        decision = record["preflight_decision"]
        counts[decision] = counts.get(decision, 0) + 1
    queued = [record for record in records if record["preflight_decision"] == "formal_backend_queue"]
    queued.sort(key=lambda item: (item["pareto_front"], item["ordinal"]))
    body = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "input_priority_schema_version": priority_report.get("schema_version"),
        "input_priority_sha256": source_priority_sha256,
        "representable_count": len(records),
        "decision_counts": dict(sorted(counts.items())),
        "formal_backend_queue": queued,
        "records": records,
        "basis": _BASIS,
        "observational_data_opened": False,
        "interpretation": (
            "representation and necessary formal preflight only; formal_backend_queue is not "
            "promotion, empirical support, or a probability of truth"
        ),
    }
    content = _canonical_json(body)
    return {**body, "content_sha256": hashlib.sha256(content.encode()).hexdigest()}, specs


def export_covariant_candidate_files(
    priority_path: str | Path,
    grammar: dict[str, Any],
    field_contract: dict[str, Any],
    output_directory: str | Path,
) -> Path:
    priority_path = Path(priority_path)
    raw = priority_path.read_bytes()
    priority = json.loads(raw.decode("utf-8"))
    source_hash = hashlib.sha256(raw).hexdigest()
    report, specs = export_representable_covariant_candidates(
        priority,
        grammar,
        field_contract,
        source_priority_sha256=source_hash,
    )
    output = Path(output_directory)
    specs_directory = output / "specs"
    specs_directory.mkdir(parents=True, exist_ok=True)
    for filename, spec in specs:
        (specs_directory / filename).write_text(
            json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    report_path = output / "covariant-export-report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report_path
