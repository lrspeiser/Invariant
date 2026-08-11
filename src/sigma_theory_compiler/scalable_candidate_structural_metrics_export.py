"""Immutable structural metrics for the 163 scalable grammar-v3 candidates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from .promotion_orchestrator import ELIGIBILITY
from .scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

CONFIG_SCHEMA = "sigma-scalable-candidate-structural-metrics-config-1.0"
EXPORT_SCHEMA = "sigma-scalable-candidate-structural-metrics-export-1.0"
RECORD_SCHEMA = "sigma-scalable-candidate-structural-metrics-record-1.0"
STRUCTURAL_CLASS = "typed_action_formula_structure_v1"
EQUIVALENCE_CLASS = "exact_action_hash_and_parameter_cell_aliases_v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path.name} must contain an object")
    return value


def _bound_path(root: Path, binding: Mapping[str, Any], label: str) -> Path:
    if set(binding) - {"path", "file_sha256", "content_sha256"} or not {
        "path",
        "file_sha256",
    }.issubset(binding):
        raise ValueError(f"{label} binding fields are invalid")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} path escapes repository") from error
    if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
        raise ValueError(f"{label} file hash mismatch")
    return path


def _bound_json(
    root: Path, binding: Mapping[str, Any], label: str
) -> dict[str, Any]:
    value = _load(_bound_path(root, binding, label))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if (
        value.get("content_sha256") != binding.get("content_sha256")
        or _sha(body) != binding.get("content_sha256")
    ):
        raise ValueError(f"{label} content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != {
        "schema_version",
        "campaign_id",
        "source_export",
        "campaign_source",
        "budget",
        "data_eligibility",
        "external_paid_llm_calls",
    } or config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError("structural metrics config is invalid")
    budget = config.get("budget", {})
    if set(budget) != {
        "maximum_candidates",
        "maximum_aliases",
        "maximum_output_bytes",
        "maximum_paid_llm_spend_usd",
    } or (
        int(budget.get("maximum_candidates", 0)) != 163
        or int(budget.get("maximum_aliases", 0)) != 93
        or not 262_144 <= int(budget.get("maximum_output_bytes", 0)) <= 8_388_608
        or float(budget.get("maximum_paid_llm_spend_usd", -1)) != 0.0
    ):
        raise ValueError("structural metrics budget is invalid")
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("structural metrics eligibility is not fail-closed")
    if config.get("external_paid_llm_calls") is not False:
        raise ValueError("structural metrics enabled paid LLM calls")


def _formula_structure(formula: Mapping[str, Any]) -> dict[str, Any]:
    operators = formula["ordered_operator_densities"]
    body = {
        "fields": list(formula["fields"]),
        "parameters": dict(formula["parameters"]),
        "ordered_operator_densities": [dict(operator) for operator in operators],
        "action_content_sha256": formula["action_content_sha256"],
        "formula_inputs_sha256": formula["formula_inputs_sha256"],
    }
    return body


def _metrics(formula: Mapping[str, Any]) -> dict[str, int]:
    fields = formula["fields"]
    parameters = formula["parameters"]
    operators = formula["ordered_operator_densities"]
    return {
        "operator_count": len(operators),
        "distinct_operator_atom_count": len({item["atom"] for item in operators}),
        "field_count": len(fields),
        "parameter_count": len(parameters),
        "operator_density_character_count": sum(len(item["density"]) for item in operators),
        "parameter_payload_character_count": len(_canonical(parameters)),
        "formula_payload_character_count": len(
            _canonical(
                {
                    "fields": fields,
                    "parameters": parameters,
                    "ordered_operator_densities": operators,
                }
            )
        ),
    }


def _simplicity_key(record: Mapping[str, Any]) -> tuple[int, ...]:
    metrics = record["structural_metrics"]
    return (
        metrics["operator_count"],
        metrics["field_count"],
        metrics["parameter_count"],
        metrics["formula_payload_character_count"],
    )


def _alias_key(record: Mapping[str, Any]) -> tuple[int]:
    return (-int(record["equivalence_evidence"]["parameter_cell_class_size"]),)


def _assign_tied_ranks(
    records: list[dict[str, Any]],
    key: Callable[[Mapping[str, Any]], tuple[Any, ...]],
    field: str,
) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda item: (*key(item), item["candidate_id"]))
    previous: tuple[Any, ...] | None = None
    rank = 0
    for index, record in enumerate(ordered, 1):
        current = key(record)
        if current != previous:
            rank = index
            previous = current
        record[field] = rank
    return ordered


def _pareto_front(records: list[dict[str, Any]]) -> list[str]:
    vectors = {record["candidate_id"]: _simplicity_key(record) for record in records}
    front = []
    for candidate_id, vector in vectors.items():
        dominated = any(
            other_id != candidate_id
            and all(left <= right for left, right in zip(other, vector, strict=True))
            and any(left < right for left, right in zip(other, vector, strict=True))
            for other_id, other in vectors.items()
        )
        if not dominated:
            front.append(candidate_id)
    return sorted(front)


def _top10_summary(
    ordered: list[dict[str, Any]], rank_field: str
) -> dict[str, Any]:
    selected = ordered[:10]
    cutoff = int(selected[-1][rank_field])
    return {
        "candidate_ids": [record["candidate_id"] for record in selected],
        "cutoff_tied_rank": cutoff,
        "boundary_tie_total_count": sum(
            int(record[rank_field]) == cutoff for record in ordered
        ),
        "deterministic_tie_break": "candidate_id_ascending",
    }


def validate_scalable_candidate_structural_metrics_export(
    export: Mapping[str, Any],
) -> None:
    body = {key: value for key, value in export.items() if key != "content_sha256"}
    if export.get("schema_version") != EXPORT_SCHEMA or export.get("content_sha256") != _sha(
        body
    ):
        raise ValueError("structural metrics export content hash mismatch")
    records = export.get("candidate_records")
    if not isinstance(records, list) or len(records) != 163:
        raise ValueError("structural metrics candidate count changed")
    if len({record.get("candidate_id") for record in records}) != 163:
        raise ValueError("structural metrics candidate identity collision")
    if len({record.get("action_sha256") for record in records}) != 163:
        raise ValueError("exact action equivalence dedup changed")
    aliases = 0
    formal = Counter()
    for record in records:
        record_body = {key: value for key, value in record.items() if key != "content_sha256"}
        if record.get("schema_version") != RECORD_SCHEMA or record.get("content_sha256") != _sha(
            record_body
        ):
            raise ValueError("structural metrics record hash mismatch")
        formula = record["formula_structure"]
        formula_body = {
            "fields": formula["fields"],
            "parameters": formula["parameters"],
            "ordered_operator_densities": formula["ordered_operator_densities"],
            "action_content_sha256": formula["action_content_sha256"],
        }
        if (
            formula["formula_inputs_sha256"] != _sha(formula_body)
            or formula["action_content_sha256"] != record["action_sha256"]
            or record["structural_metrics"] != _metrics(formula)
        ):
            raise ValueError("structural metrics do not rederive from exact formula inputs")
        evidence = record["equivalence_evidence"]
        if (
            evidence["exact_action_equivalence_class_sha256"] != record["action_sha256"]
            or evidence["representative_action_class_size"] != 1
            or evidence["parameter_cell_class_size"] != evidence["alias_count"] + 1
            or evidence["alias_lineage_root_sha256"] != record["alias_lineage_root_sha256"]
        ):
            raise ValueError("structural exact-equivalence evidence changed")
        if (
            record["structural_evidence_status"] != "measured"
            or record["structural_comparison_class"] != STRUCTURAL_CLASS
            or record["equivalence_comparison_class"] != EQUIVALENCE_CLASS
            or record["scientific_validity_inference"] is not False
            or not isinstance(record["simplicity_tied_rank"], int)
            or not isinstance(record["alias_multiplicity_tied_rank"], int)
        ):
            raise ValueError("structural measurement leaked into scientific validity")
        forbidden_text = _canonical(record).lower()
        if any(
            token in forbidden_text
            for token in ("truth_score", "overall_score", "global_score", "dark_matter", "halo_target", "redshift_distance")
        ):
            raise ValueError("forbidden score or inferred target entered structural metrics")
        aliases += int(evidence["alias_count"])
        formal[record["formal_context"]["decision"]] += 1
    if aliases != 93 or dict(sorted(formal.items())) != {
        "blocked": 158,
        "pass": 3,
        "reject": 2,
    }:
        raise ValueError("structural metrics source accounting changed")
    if export.get("alias_count") != 93 or export.get("formal_decision_counts") != {
        "blocked": 158,
        "pass": 3,
        "reject": 2,
    }:
        raise ValueError("structural metrics aggregate accounting changed")
    family_counts = dict(sorted(Counter(record["family_id"] for record in records).items()))
    if export.get("family_counts") != family_counts or family_counts != {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 2,
    }:
        raise ValueError("structural metrics family accounting changed")
    alias_class_sizes = dict(
        sorted(
            Counter(
                str(record["equivalence_evidence"]["parameter_cell_class_size"])
                for record in records
            ).items()
        )
    )
    if export.get("parameter_cell_class_size_counts") != alias_class_sizes or alias_class_sizes != {
        "1": 160,
        "32": 3,
    }:
        raise ValueError("structural alias-class accounting changed")
    simplicity = sorted(records, key=lambda item: (*_simplicity_key(item), item["candidate_id"]))
    alias_multiplicity = sorted(records, key=lambda item: (*_alias_key(item), item["candidate_id"]))
    for ordered, key, field in (
        (simplicity, _simplicity_key, "simplicity_tied_rank"),
        (alias_multiplicity, _alias_key, "alias_multiplicity_tied_rank"),
    ):
        previous = None
        expected_rank = 0
        for index, record in enumerate(ordered, 1):
            current = key(record)
            if current != previous:
                expected_rank = index
                previous = current
            if record[field] != expected_rank:
                raise ValueError("structural tied rank changed")
    if (
        export.get("simplicity_top10")
        != _top10_summary(simplicity, "simplicity_tied_rank")
        or export.get("alias_multiplicity_top10")
        != _top10_summary(alias_multiplicity, "alias_multiplicity_tied_rank")
    ):
        raise ValueError("structural deterministic top-10 changed")
    front = _pareto_front(records)
    if export.get("simplicity_pareto_front") != {
        "candidate_count": len(front),
        "candidate_ids": front,
        "registry_root_sha256": _sha(front),
    }:
        raise ValueError("structural Pareto front changed")
    if export.get("comparison_classes") != {
        "simplicity_complexity": STRUCTURAL_CLASS,
        "internal_exact_equivalence": EQUIVALENCE_CLASS,
    } or export.get("simplicity_metric_order") != [
        "operator_count",
        "field_count",
        "parameter_count",
        "formula_payload_character_count",
    ]:
        raise ValueError("structural comparison contract changed")
    if export.get("candidate_record_registry_root_sha256") != _sha(
        [record["content_sha256"] for record in records]
    ):
        raise ValueError("structural metrics candidate registry root changed")
    if export.get("data_eligibility") != {**ELIGIBILITY, "passed": True}:
        raise ValueError("structural metrics opened forbidden data")


def build_scalable_candidate_structural_metrics_export(
    config: Mapping[str, Any], root: str | Path
) -> dict[str, Any]:
    _validate_config(config)
    root = Path(root).resolve()
    _bound_path(root, config["campaign_source"], "campaign_source")
    source = _bound_json(root, config["source_export"], "source_export")
    validate_scalable_formal_candidate_evidence_export(source)
    source_records = iter_scalable_formal_candidate_evidence_records(source)
    if len(source_records) != 163 or sum(item["alias_count"] for item in source_records) != 93:
        raise ValueError("structural metrics source population changed")
    records = []
    for source_record in source_records:
        formula = _formula_structure(source_record["theory_formula_inputs"])
        body = {
            "schema_version": RECORD_SCHEMA,
            "candidate_id": source_record["candidate_id"],
            "family_id": source_record["family_id"],
            "action_sha256": source_record["action_sha256"],
            "formula_structure": formula,
            "structural_metrics": _metrics(formula),
            "structural_evidence_status": "measured",
            "structural_comparison_class": STRUCTURAL_CLASS,
            "equivalence_comparison_class": EQUIVALENCE_CLASS,
            "alias_lineage_root_sha256": source_record["alias_lineage_root_sha256"],
            "equivalence_evidence": {
                "exact_action_equivalence_class_sha256": source_record["action_sha256"],
                "representative_action_class_size": 1,
                "parameter_cell_class_size": source_record["alias_count"] + 1,
                "alias_count": source_record["alias_count"],
                "alias_lineage_root_sha256": source_record["alias_lineage_root_sha256"],
                "literature_novelty_claimed": False,
            },
            "formal_context": {
                "decision": source_record["final_decision"],
                "comparison_data_class": source_record["comparison_data_class"],
                "blocker": source_record["blocker"],
                "result_sha256": source_record["result_sha256"],
                "used_for_structural_rank": False,
            },
            "simplicity_tied_rank": None,
            "alias_multiplicity_tied_rank": None,
            "scientific_validity_inference": False,
        }
        records.append(body)
    simplicity = _assign_tied_ranks(records, _simplicity_key, "simplicity_tied_rank")
    alias_multiplicity = _assign_tied_ranks(
        records, _alias_key, "alias_multiplicity_tied_rank"
    )
    records = [
        {**record, "content_sha256": _sha(record)}
        for record in sorted(records, key=lambda item: item["candidate_id"])
    ]
    front = _pareto_front(records)
    decision_counts = dict(
        sorted(Counter(record["formal_context"]["decision"] for record in records).items())
    )
    body = {
        "schema_version": EXPORT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "source_bindings": {
            "source_export": dict(config["source_export"]),
            "campaign_source": dict(config["campaign_source"]),
        },
        "candidate_count": len(records),
        "alias_count": sum(
            record["equivalence_evidence"]["alias_count"] for record in records
        ),
        "parameter_cell_count": 256,
        "representative_exact_action_class_count": 163,
        "representative_exact_action_duplicate_count": 0,
        "formal_decision_counts": decision_counts,
        "family_counts": dict(
            sorted(Counter(record["family_id"] for record in records).items())
        ),
        "parameter_cell_class_size_counts": dict(
            sorted(
                Counter(
                    str(record["equivalence_evidence"]["parameter_cell_class_size"])
                    for record in records
                ).items()
            )
        ),
        "structural_measurement_counts": {"measured": len(records)},
        "comparison_classes": {
            "simplicity_complexity": STRUCTURAL_CLASS,
            "internal_exact_equivalence": EQUIVALENCE_CLASS,
        },
        "simplicity_metric_order": [
            "operator_count",
            "field_count",
            "parameter_count",
            "formula_payload_character_count",
        ],
        "simplicity_top10": _top10_summary(simplicity, "simplicity_tied_rank"),
        "alias_multiplicity_top10": _top10_summary(
            alias_multiplicity, "alias_multiplicity_tied_rank"
        ),
        "simplicity_pareto_front": {
            "candidate_count": len(front),
            "candidate_ids": front,
            "registry_root_sha256": _sha(front),
        },
        "candidate_record_registry_root_sha256": _sha(
            [record["content_sha256"] for record in records]
        ),
        "candidate_records": records,
        "ranking_contract": (
            "category-local exact structural tuples only; tied ranks share identical metrics; "
            "candidate_id is only the deterministic display tie-break; no scalar or validity score"
        ),
        "observational_data_opened": False,
        "paid_llm_spend_usd": 0.0,
        "data_eligibility": {**ELIGIBILITY, "passed": True},
    }
    export = {**body, "content_sha256": _sha(body)}
    if len(_canonical(export).encode()) > int(config["budget"]["maximum_output_bytes"]):
        raise ValueError("structural metrics export exceeds bounded output budget")
    validate_scalable_candidate_structural_metrics_export(export)
    return export
