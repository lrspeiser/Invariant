from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from .knowledge import pareto_fronts
from .promotion_orchestrator import ELIGIBILITY, PIPELINE_SCHEMA, STATUS_SCHEMA

DOSSIER_SCHEMA = "sigma-promotion-dossier-1.0"


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _scalar_reasons(output: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for key in (
        "blocker",
        "reason",
        "failure",
        "check",
        "status",
        "decision",
    ):
        value = output.get(key)
        if isinstance(value, (str, int, float, bool)) and str(value):
            reasons.append(f"{key}={value}")
    return reasons


def _load_ledger(database: Path) -> tuple[dict[str, Any], list[sqlite3.Row]]:
    if not database.is_file():
        raise FileNotFoundError(database)
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        pipeline_row = connection.execute(
            "SELECT config_json,config_sha256 FROM pipeline WHERE singleton=1"
        ).fetchone()
        if pipeline_row is None:
            raise ValueError("promotion ledger has no pipeline")
        pipeline = json.loads(pipeline_row["config_json"])
        if pipeline.get("schema_version") != PIPELINE_SCHEMA or _sha(pipeline) != pipeline_row[
            "config_sha256"
        ]:
            raise ValueError("promotion pipeline hash or schema is invalid")
        rows = connection.execute(
            "SELECT c.candidate_id,c.ordinal,c.source_sha256,c.payload_json,"
            "c.initial_lineage_sha256,s.stage_index,s.stage_name,s.category,s.state,s.blocker,"
            "s.attempt,s.evaluator_binding_sha256,s.input_lineage_sha256,s.result_json,"
            "s.result_sha256,s.output_lineage_sha256 FROM candidates c JOIN candidate_stages s "
            "USING(candidate_id) ORDER BY c.candidate_id,s.stage_index"
        ).fetchall()
    finally:
        connection.close()
    return pipeline, rows


def build_promotion_dossiers(database: str | Path) -> dict[str, Any]:
    database_path = Path(database).resolve()
    pipeline, rows = _load_ledger(database_path)
    stages = list(pipeline["stages"])
    grouped: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)

    dossiers: list[dict[str, Any]] = []
    ledger_records: list[dict[str, Any]] = []
    eligible_for_queue: list[dict[str, Any]] = []
    for candidate_id, candidate_rows in grouped.items():
        if len(candidate_rows) != len(stages):
            raise ValueError(f"candidate {candidate_id} does not have every configured stage")
        payload = json.loads(candidate_rows[0]["payload_json"])
        if payload.get("candidate_id") != candidate_id or payload.get("data_eligibility") != ELIGIBILITY:
            raise ValueError(f"candidate {candidate_id} payload is not eligible or identity-bound")

        stage_reports: list[dict[str, Any]] = []
        passed_depth = 0
        exact_result_count = 0
        rejected = False
        first_nonpass: dict[str, Any] | None = None
        prior_output: str | None = None
        for index, row in enumerate(candidate_rows):
            stage = stages[index]
            if (
                int(row["stage_index"]) != index
                or row["stage_name"] != stage["name"]
                or row["category"] != stage["category"]
            ):
                raise ValueError(f"candidate {candidate_id} stage order differs from pipeline")
            result = json.loads(row["result_json"]) if row["result_json"] else None
            evidence_sha: str | None = None
            reasons: list[str] = []
            if index == 0:
                if result is None or row["state"] != "passed":
                    raise ValueError("sampled-static stage must contain passing evidence")
                initial = _sha({"candidate": payload, "sampled_static_evidence": result})
                if (
                    initial != row["initial_lineage_sha256"]
                    or initial != row["output_lineage_sha256"]
                    or _sha(result) != row["result_sha256"]
                    or result.get("source_result_sha256") != row["source_sha256"]
                    or result.get("data_eligibility") != ELIGIBILITY
                ):
                    raise ValueError(f"candidate {candidate_id} sampled-static lineage is invalid")
                evidence_sha = str(row["result_sha256"])
                prior_output = initial
                exact_result_count += 1
            elif result is not None:
                binding = row["evaluator_binding_sha256"]
                if not _is_sha256(binding) or prior_output is None:
                    raise ValueError(f"candidate {candidate_id} evaluated stage lacks provenance")
                expected_input = _sha(
                    {
                        "candidate_id": candidate_id,
                        "stage": stage,
                        "prior_lineage_sha256": prior_output,
                        "evaluator_binding_sha256": binding,
                    }
                )
                output = result.get("output")
                expected_result = _sha(result)
                expected_output = _sha(
                    {
                        "input_lineage_sha256": expected_input,
                        "result_sha256": expected_result,
                    }
                )
                if (
                    result.get("schema_version") != "sigma-promotion-gate-result-1.0"
                    or result.get("candidate_id") != candidate_id
                    or result.get("stage_name") != stage["name"]
                    or result.get("category") != stage["category"]
                    or result.get("evaluator_binding_sha256") != binding
                    or result.get("input_lineage_sha256") != expected_input
                    or result.get("data_eligibility") != ELIGIBILITY
                    or not isinstance(output, dict)
                    or output.get("data_eligibility") != ELIGIBILITY
                    or row["input_lineage_sha256"] != expected_input
                    or row["result_sha256"] != expected_result
                    or row["output_lineage_sha256"] != expected_output
                ):
                    raise ValueError(f"candidate {candidate_id} gate-result lineage is invalid")
                evidence_sha = expected_result
                prior_output = expected_output
                exact_result_count += 1
                reasons = _scalar_reasons(output)
            elif row["result_sha256"] is not None or row["output_lineage_sha256"] is not None:
                raise ValueError(f"candidate {candidate_id} has orphaned result hashes")

            state = str(row["state"])
            if state == "passed" and first_nonpass is None:
                passed_depth += 1
            elif first_nonpass is None:
                first_nonpass = {
                    "stage_name": stage["name"],
                    "state": state,
                    "blocker": row["blocker"],
                    "reasons": reasons,
                }
            rejected = rejected or state == "rejected"
            stage_report = {
                "index": index,
                "stage_name": stage["name"],
                "category": stage["category"],
                "state": state,
                "blocker": row["blocker"],
                "attempt": int(row["attempt"]),
                "evidence_sha256": evidence_sha,
                "output_lineage_sha256": row["output_lineage_sha256"],
                "reasons": reasons,
            }
            stage_reports.append(stage_report)
            ledger_records.append({"candidate_id": candidate_id, **stage_report})

        term_ids = payload.get("term_ids")
        term_count = len(term_ids) if isinstance(term_ids, list) and term_ids else 1
        if rejected:
            disposition = "terminally_rejected"
            explanation = "A hard gate rejected this exact candidate; later evidence cannot rescue it."
        elif first_nonpass is None:
            disposition = "passed_all_configured_gates"
            explanation = "The candidate passed every configured gate; this is not proof of nature."
        else:
            disposition = "blocked_missing_or_unresolved_evidence"
            explanation = (
                f"Work stops at {first_nonpass['stage_name']}: "
                f"{first_nonpass['blocker'] or first_nonpass['state']}."
            )
        dossier = {
            "candidate_id": candidate_id,
            "ordinal": int(candidate_rows[0]["ordinal"]),
            "correction_expression": payload.get("correction_expression", payload.get("formula")),
            "term_count": term_count,
            "passed_gate_count": passed_depth,
            "exact_result_count": exact_result_count,
            "disposition": disposition,
            "first_nonpass": first_nonpass,
            "explanation": explanation,
            "initial_lineage_sha256": candidate_rows[0]["initial_lineage_sha256"],
            "stages": stage_reports,
        }
        dossiers.append(dossier)
        if not rejected:
            eligible_for_queue.append(
                {
                    "formula_id": candidate_id,
                    "candidate_id": candidate_id,
                    "passed_gate_count": passed_depth,
                    "exact_result_count": exact_result_count,
                    "parsimony": 1.0 / term_count,
                    "disposition": disposition,
                    "next_gate": first_nonpass["stage_name"] if first_nonpass else None,
                    "blocker": first_nonpass["blocker"] if first_nonpass else None,
                }
            )

    axes = ["passed_gate_count", "exact_result_count", "parsimony"]
    fronts = pareto_fronts(eligible_for_queue, axes)
    queue: list[dict[str, Any]] = []
    for front_index, front in enumerate(fronts, start=1):
        for item in front:
            queue.append({**item, "pareto_front": front_index})
    report = {
        "schema_version": DOSSIER_SCHEMA,
        "source_status_schema": STATUS_SCHEMA,
        "pipeline_sha256": _sha(pipeline),
        "candidate_count": len(dossiers),
        "terminal_rejection_count": sum(
            item["disposition"] == "terminally_rejected" for item in dossiers
        ),
        "work_queue_count": len(queue),
        "pareto_axes": axes,
        "pareto_work_queue": queue,
        "candidate_dossiers": dossiers,
        "ledger_root_sha256": _sha(ledger_records),
        "data_eligibility": {**ELIGIBILITY, "passed": True},
        "interpretation": (
            "The queue allocates follow-up work among non-rejected candidates. Pareto position is "
            "not a truth probability, cannot override a hard rejection, and opens no observations."
        ),
    }
    report["content_sha256"] = _sha(report)
    return report


def write_promotion_dossiers(database: str | Path, output: str | Path) -> Path:
    report = build_promotion_dossiers(database)
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
