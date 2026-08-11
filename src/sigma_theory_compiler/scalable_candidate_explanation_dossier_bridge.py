from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY
from .scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

CONFIG_SCHEMA = "sigma-scalable-candidate-explanation-dossier-bridge-config-1.0"
ARTIFACT_SCHEMA = "sigma-scalable-candidate-explanation-dossier-bridge-1.0"
DOSSIER_SCHEMA = "sigma-scalable-candidate-explanation-dossier-1.0"
NODE_STATUSES = {"proven", "rejected", "blocked", "calibration_only"}
DECISION_NODE_STATUS = {"pass": "proven", "reject": "rejected", "blocked": "blocked"}
SOURCE_FAMILIES = {
    "aether_parameter_cell_formal_gate": (
        "AETHER_K1234_PARAMETER_CELL",
        "aether_status",
    ),
    "grammar_v3_g2_candidate_formal": ("KESSENCE_G2_CONVEX", "g2_status"),
    "grammar_v3_g3_candidate_formal": (
        "CUBIC_HORNDESKI_G3_WEAK_CELL",
        "g3_status",
    ),
    "g4_scalable_action_formal_followup": (
        "CONFORMAL_G4_PHI_SCALAR_TENSOR",
        "g4_followup",
    ),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_bound(root: Path, descriptor: dict[str, Any], label: str) -> dict[str, Any]:
    if set(descriptor) - {"path", "file_sha256", "content_sha256"} or not {
        "path",
        "file_sha256",
    }.issubset(descriptor):
        raise ValueError(f"invalid source binding: {label}")
    path = (root / descriptor["path"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"source path escapes repository: {label}") from error
    if not path.is_file() or _file_sha(path) != descriptor["file_sha256"]:
        raise ValueError(f"bound file hash mismatch: {label}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"bound source must contain an object: {label}")
    expected = descriptor.get("content_sha256")
    if expected is not None:
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != expected or _sha(body) != expected:
            raise ValueError(f"bound content hash mismatch: {label}")
    return value


def _validate_config(config: dict[str, Any]) -> None:
    if (
        set(config)
        != {
            "schema_version",
            "campaign_id",
            "source_bindings",
            "budget",
            "data_eligibility",
            "observational_authorization",
            "external_paid_llm_calls",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
    ):
        raise ValueError("explanation dossier bridge config is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("explanation dossier bridge eligibility is not fail-closed")
    if config.get("observational_authorization") is not False:
        raise ValueError("observational authorization must remain false")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("paid LLM calls must remain false")
    if set(config.get("source_bindings", {})) != {
        "scalable_export",
        "aether_status",
        "g2_status",
        "g3_status",
        "g4_followup",
    }:
        raise ValueError("explanation dossier source binding set changed")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_candidates",
        "maximum_output_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget["maximum_candidates"]) != 163
        or not 1024 * 1024 <= int(budget["maximum_output_bytes"]) <= 32 * 1024 * 1024
        or float(budget["maximum_paid_llm_spend_usd"]) != 0.0
    ):
        raise ValueError("explanation dossier bridge budget is invalid")


def _evidence(
    bindings: dict[str, Any], key: str, locator: str, evidence_sha256: str
) -> dict[str, Any]:
    descriptor = bindings[key]
    return {
        "artifact_key": key,
        "artifact_path": descriptor["path"],
        "artifact_file_sha256": descriptor["file_sha256"],
        "artifact_content_sha256": descriptor.get("content_sha256"),
        "json_locator": locator,
        "evidence_sha256": evidence_sha256,
    }


def _node(
    node_id: str,
    status: str,
    scope: str,
    evidence: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    if status not in NODE_STATUSES or not evidence:
        raise ValueError(f"invalid explanation hierarchy node: {node_id}")
    body = {
        "node_id": node_id,
        "status": status,
        "scope": scope,
        "evidence": evidence,
        **extra,
    }
    return {**body, "content_sha256": _sha(body)}


def _human_readable_action(formula: dict[str, Any]) -> dict[str, Any]:
    """Format only exact exported densities; family labels never enter this function."""
    densities = [item["density"] for item in formula["ordered_operator_densities"]]
    return {
        "display_kind": "verbatim_ordered_covariant_density_concatenation",
        "display_text": "S = integral d^4x [" + " + ".join(f"({term})" for term in densities) + "]",
        "scope": (
            "Display-only concatenation of the exact exported operator densities. No field "
            "equation, missing matter term, or family-template term is inferred."
        ),
    }


def _unanimous_gate_summary(status: dict[str, Any], expected_count: int) -> dict[str, str]:
    output: dict[str, str] = {}
    for gate, counts in sorted(status.get("gate_counts", {}).items()):
        if (
            not isinstance(counts, dict)
            or len(counts) != 1
            or sum(int(value) for value in counts.values()) != expected_count
        ):
            raise ValueError(f"family gate is not unanimous: {gate}")
        output[gate] = next(iter(counts))
    if not output:
        raise ValueError("family gate summary is empty")
    return output


def _family_formal_context(
    record: dict[str, Any], sources: dict[str, dict[str, Any]]
) -> tuple[str, dict[str, Any]]:
    evidence_source = record["evidence_source"]
    if evidence_source not in SOURCE_FAMILIES:
        raise ValueError("unreviewed candidate evidence source")
    expected_family, source_key = SOURCE_FAMILIES[evidence_source]
    if record["family_id"] != expected_family:
        raise ValueError("family label and exact evidence source disagree")
    status = sources[source_key]
    candidate_id = record["candidate_id"]
    if source_key == "aether_status":
        bindings = {item["candidate_id"]: item for item in status.get("candidate_bindings", [])}
        bound = bindings.get(candidate_id)
        if (
            status.get("candidate_count") != 128
            or status.get("decision_counts") != {"blocked": 126, "reject": 2}
            or bound is None
            or bound.get("action_sha256") != record["action_sha256"]
            or bound.get("decision") != record["final_decision"]
            or bound.get("blocker") != record["blocker"]
            or bound.get("candidate_gate_record_sha256") != record["result_sha256"]
        ):
            raise ValueError("Aether candidate formal binding changed")
        context = {
            "gate_summary_scope": "candidate-specific final necessary-condition record",
            "candidate_gate_record_sha256": bound["candidate_gate_record_sha256"],
            "specialization_sha256": bound["specialization_sha256"],
            "reviewed_adapter_evidence": status["reviewed_formal_adapter_evidence"],
        }
    elif source_key == "g2_status":
        if (
            status.get("candidate_count") != 2
            or status.get("decision_counts") != {"blocked": 2}
            or record["final_decision"] != "blocked"
            or record["blocker"] != "hash_bound_general_nonmaximal_positive_mass_theorem"
        ):
            raise ValueError("G2 candidate formal binding changed")
        context = {
            "gate_summary_scope": "unanimous reviewed result across both exact G2 actions",
            "unanimous_gate_outcomes": _unanimous_gate_summary(status, 2),
            "candidate_result_sha256": record["result_sha256"],
            "reviewed_adapter_registry_root_sha256": status[
                "reviewed_adapter_registry_root_sha256"
            ],
        }
    elif source_key == "g3_status":
        if (
            status.get("candidate_count") != 32
            or status.get("decision_counts") != {"blocked": 32}
            or record["final_decision"] != "blocked"
            or record["blocker"] != "uniformly_invertible_Delta_N_on_AF_decaying_gradient_domain"
        ):
            raise ValueError("G3 candidate formal binding changed")
        context = {
            "gate_summary_scope": "unanimous reviewed result across all 32 exact G3 actions",
            "unanimous_gate_outcomes": _unanimous_gate_summary(status, 32),
            "candidate_result_sha256": record["result_sha256"],
            "reviewed_adapter_registry_root_sha256": status[
                "reviewed_adapter_registry_root_sha256"
            ],
        }
    else:
        certificate = status.get("equivalence_certificate", {})
        if (
            status.get("candidate_id") != candidate_id
            or status.get("formal_followup_decision") != "pass"
            or status.get("decision_counts") != {"pass": 1}
            or record["final_decision"] != "pass"
            or record["blocker"] is not None
            or record["result_sha256"] != status.get("content_sha256")
            or certificate.get("action_density_projection_equal") is not True
            or certificate.get("family_label_used_as_equivalence_evidence") is not False
            or certificate.get("scalable_action_sha256") != record["action_sha256"]
        ):
            raise ValueError("G4 candidate formal follow-up binding changed")
        context = {
            "gate_summary_scope": "candidate-specific exact action-equivalence formal follow-up",
            "candidate_gate_outcomes": {
                gate: value["status"] for gate, value in sorted(status["gate_ledger"].items())
            },
            "equivalence_certificate_sha256": certificate["content_sha256"],
            "family_label_used_as_equivalence_evidence": False,
        }
    return source_key, context


def _dossier(
    record: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    bindings: dict[str, Any],
) -> dict[str, Any]:
    formula = record["theory_formula_inputs"]
    source_key, formal_context = _family_formal_context(record, sources)
    export_evidence = _evidence(
        bindings,
        "scalable_export",
        f"candidate_records[candidate_id={record['candidate_id']}]",
        record["action_sha256"],
    )
    formal_evidence = _evidence(
        bindings,
        source_key,
        f"candidate_formal_evidence[candidate_id={record['candidate_id']}]",
        record["result_sha256"],
    )
    decision = record["final_decision"]
    if decision == "pass":
        formal_scope = (
            "Exact action-level formal evidence is complete for the registered domain. This is "
            "not Solar, galaxy, or total observational validation."
        )
    elif decision == "reject":
        formal_scope = (
            "An exact necessary-condition violation rejects this candidate only inside its "
            "declared comparison class."
        )
    else:
        formal_scope = (
            "The first missing formal premise remains unresolved. Blocked means missing proof, "
            "not a measured failure or rejection."
        )
    nodes = [
        _node(
            "exact_covariant_action_inputs",
            "proven",
            "Exact fields, parameters, and ordered operator densities copied from the immutable per-candidate export.",
            [export_evidence],
            action_sha256=record["action_sha256"],
            formula_inputs_sha256=formula["formula_inputs_sha256"],
        ),
        _node(
            "reviewed_formal_evidence",
            DECISION_NODE_STATUS[decision],
            formal_scope,
            [export_evidence, formal_evidence],
            decision=decision,
            blocker=record["blocker"],
            direct_metrics=record["direct_metrics"],
            metric_source_sha256=record["metric_source_sha256"],
            **formal_context,
        ),
        _node(
            "family_label_and_control_boundary",
            "calibration_only",
            (
                "Family identifiers and reviewed control material route evidence only. They do "
                "not supply action terms, prove equivalence, or change the candidate decision."
            ),
            [export_evidence, formal_evidence],
            family_label_used_as_action_source=False,
            family_label_used_as_equivalence_proof=False,
        ),
        _node(
            "downstream_observational_evidence",
            "blocked",
            (
                "No observation is opened by this explanatory bridge. Missing downstream evidence "
                "is untested, not poor performance."
            ),
            [export_evidence],
            observational_authorization=False,
            observational_data_opened=False,
        ),
    ]
    body = {
        "schema_version": DOSSIER_SCHEMA,
        "candidate_id": record["candidate_id"],
        "role": "generated_candidate",
        "family_id": record["family_id"],
        "action": {
            "action_sha256": record["action_sha256"],
            "formula_inputs_sha256": formula["formula_inputs_sha256"],
            "fields": formula["fields"],
            "parameters": formula["parameters"],
            "ordered_operator_densities": formula["ordered_operator_densities"],
            "human_readable_action": _human_readable_action(formula),
        },
        "alias_lineage": {
            "alias_count": record["alias_count"],
            "alias_lineage_root_sha256": record["alias_lineage_root_sha256"],
        },
        "preflight": {
            "decision": record["preflight_decision"],
            "result_sha256": record["preflight_result_sha256"],
            "scope": "historical prerequisite result; the final reviewed evidence below is authoritative",
        },
        "formal_decision": decision,
        "first_blocker": record["blocker"],
        "comparison_contract": {
            "comparison_data_class": record["comparison_data_class"],
            "rank": None,
            "rank_eligible_within_declared_class": decision in {"pass", "reject"},
            "cross_class_ranking_allowed": False,
            "promotion_eligible": False,
        },
        "hierarchy_nodes": nodes,
    }
    return {**body, "content_sha256": _sha(body)}


def _validate_output(artifact: dict[str, Any], maximum_output_bytes: int) -> None:
    dossiers = artifact["dossiers"]
    if (
        len(dossiers) != 163
        or len({item["candidate_id"] for item in dossiers}) != 163
        or artifact["candidate_count"] != 163
        or artifact["alias_count"] != 93
        or artifact["formal_decision_counts"] != {"blocked": 160, "pass": 1, "reject": 2}
    ):
        raise ValueError("explanation dossier candidate accounting changed")
    for dossier in dossiers:
        if dossier["comparison_contract"]["rank"] is not None:
            raise ValueError("explanation dossier assigned a rank")
        for node in dossier["hierarchy_nodes"]:
            if node["status"] not in NODE_STATUSES or not node["evidence"]:
                raise ValueError("explanation dossier hierarchy is unbound")
            body = {key: value for key, value in node.items() if key != "content_sha256"}
            if node["content_sha256"] != _sha(body):
                raise ValueError("explanation dossier hierarchy hash changed")
    serialized = _canonical(artifact)
    lowered = serialized.lower()
    if (
        len(serialized.encode()) > maximum_output_bytes
        or "c:\\users\\" in lowered
        or "c:/users/" in lowered
        or any(token in lowered for token in ('"truth_score"', '"overall_score"', '"probability"'))
    ):
        raise ValueError("explanation dossier output violates portability or scoring policy")


def build_scalable_candidate_explanation_dossier_bridge(
    config: dict[str, Any], root: str | Path
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    bindings = config["source_bindings"]
    sources = {key: _load_bound(root, descriptor, key) for key, descriptor in bindings.items()}
    export = sources["scalable_export"]
    validate_scalable_formal_candidate_evidence_export(export)
    if export.get("data_eligibility") != {**ELIGIBILITY, "passed": True}:
        raise ValueError("scalable export data seal changed")
    for key, value in sources.items():
        if value.get("observational_data_opened") not in {None, False}:
            raise ValueError(f"source opened observations: {key}")
        eligibility = value.get("data_eligibility")
        if eligibility is not None and eligibility not in (
            ELIGIBILITY,
            {**ELIGIBILITY, "passed": True},
        ):
            raise ValueError(f"source eligibility changed: {key}")
    records = iter_scalable_formal_candidate_evidence_records(export)
    dossiers = [_dossier(record, sources, bindings) for record in records]
    node_counts = Counter(
        node["status"] for dossier in dossiers for node in dossier["hierarchy_nodes"]
    )
    family_counts = Counter(item["family_id"] for item in dossiers)
    decision_counts = Counter(item["formal_decision"] for item in dossiers)
    comparison_counts = Counter(
        item["comparison_contract"]["comparison_data_class"] or "unranked_incomplete"
        for item in dossiers
    )
    negative_controls = {
        "action_display_uses_exact_exported_densities_only": "pass",
        "family_label_is_not_action_or_equivalence_evidence": "pass",
        "blocked_is_not_rewritten_as_rejected": "pass",
        "comparison_classes_are_not_merged": "pass",
        "observations_and_paid_llm_remain_sealed": "pass",
    }
    provenance_body = {
        "source_file_sha256": {
            key: descriptor["file_sha256"] for key, descriptor in sorted(bindings.items())
        },
        "source_content_sha256": {
            key: descriptor["content_sha256"]
            for key, descriptor in sorted(bindings.items())
            if descriptor.get("content_sha256") is not None
        },
        "dossier_registry_root_sha256": _sha(
            [[item["candidate_id"], item["content_sha256"]] for item in dossiers]
        ),
        "data_eligibility": ELIGIBILITY,
    }
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": bindings,
        "candidate_count": len(dossiers),
        "alias_count": sum(item["alias_lineage"]["alias_count"] for item in dossiers),
        "family_counts": dict(sorted(family_counts.items())),
        "formal_decision_counts": dict(sorted(decision_counts.items())),
        "hierarchy_node_status_counts": dict(sorted(node_counts.items())),
        "comparison_data_class_counts": dict(sorted(comparison_counts.items())),
        "dossiers": dossiers,
        "negative_control_results": negative_controls,
        "provenance": {**provenance_body, "binding_sha256": _sha(provenance_body)},
        "observational_authorization": False,
        "observational_data_opened": False,
        "dark_matter_or_halo_inputs": False,
        "redshift_distance_inputs": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": dict(ELIGIBILITY),
        "interpretation": (
            "An immutable explanation bridge over 163 exact candidate action inputs and their "
            "reviewed formal evidence. Proven, rejected, blocked, and calibration-only nodes retain "
            "their source scope. Blockers are missing premises rather than failures; no family "
            "template supplies equations; comparison classes remain separate; no observation is opened."
        ),
    }
    artifact = {**body, "content_sha256": _sha(body)}
    _validate_output(artifact, int(config["budget"]["maximum_output_bytes"]))
    return artifact
