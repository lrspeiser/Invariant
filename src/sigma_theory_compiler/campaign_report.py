from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .campaign import CampaignStore, sha256_file


def build_campaign_report(
    store: CampaignStore,
    campaign_id: str,
    output_directory: str | Path,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    status = store.status(campaign_id)
    with store.connect() as connection:
        candidate_rows = connection.execute(
            "SELECT c.*, "
            "SUM(CASE WHEN e.is_hard=1 AND e.outcome='pass' THEN 1 ELSE 0 END) AS hard_passes, "
            "SUM(CASE WHEN e.is_hard=1 AND e.outcome='reject' THEN 1 ELSE 0 END) AS hard_rejections "
            "FROM candidates c LEFT JOIN evidence e ON e.candidate_id=c.candidate_id "
            "WHERE c.campaign_id=? GROUP BY c.candidate_id ORDER BY "
            "CASE WHEN c.pareto_front IS NULL THEN 999999 ELSE c.pareto_front END, "
            "hard_rejections,hard_passes DESC,c.generation,c.candidate_id",
            (campaign_id,),
        ).fetchall()
        artifacts = [
            dict(row)
            for row in connection.execute(
                "SELECT artifact_id,candidate_id,task_id,kind,path,sha256,size_bytes,created_utc "
                "FROM artifacts WHERE campaign_id=? ORDER BY created_utc",
                (campaign_id,),
            ).fetchall()
        ]
        clusters = [
            dict(row)
            for row in connection.execute(
                "SELECT gate_id,mechanism_tag,rejection_count,summary FROM failure_clusters "
                "WHERE campaign_id=? ORDER BY rejection_count DESC,gate_id,mechanism_tag",
                (campaign_id,),
            ).fetchall()
        ]
        proposals = [
            dict(row)
            for row in connection.execute(
                "SELECT proposal_id,parent_candidate_id,status,validation_json,created_utc FROM proposals "
                "WHERE campaign_id=? ORDER BY created_utc",
                (campaign_id,),
            ).fetchall()
        ]

    candidates = []
    for row in candidate_rows:
        item = dict(row)
        item["mechanism_tags"] = json.loads(item.pop("mechanism_tags_json"))
        item["canonical"] = json.loads(item.pop("canonical_json"))
        item["remaining_claims"] = store.unresolved_claims(item["candidate_id"])
        item["why_prioritized"] = {
            "inherited_pareto_front": item["pareto_front"],
            "completed_hard_gate_passes": item["hard_passes"],
            "hard_gate_rejections": item["hard_rejections"],
            "status": item["status"],
            "semantics": "work ordering within declared grammar; not probability of truth",
        }
        candidates.append(item)

    discovery = [
        item for item in candidates if item["kind"] != "gr_control" and item["status"] != "rejected"
    ]
    report = {
        "schema_version": "sigma-campaign-report-1.0",
        "created_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "scientific_claim": (
            "This report identifies work-priority candidates only within the declared grammar and "
            "completed gates. It does not claim a true or uniquely best gravity theory."
        ),
        "leading_work_candidates": discovery[:25],
        "all_candidates": candidates,
        "failure_clusters": clusters,
        "proposals": proposals,
        "artifacts": artifacts,
        "forbidden_rescues": [
            "dark or invisible halo targets",
            "redshift-derived distances",
            "supernova distance moduli",
            "derived GR/NFW lensing targets",
            "observational fit quality replacing action health",
        ],
    }
    json_path = output_directory / "campaign-report.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path = output_directory / "campaign-report.md"
    lines = [
        "# Sigma Campaign Engine report",
        "",
        f"Campaign: `{campaign_id}` — state **{status['state']}**.",
        "",
        report["scientific_claim"],
        "",
        "## Durable accounting",
        "",
        f"- Tasks: `{status['task_counts']}`",
        f"- Candidates: `{status['candidate_counts']}`",
        f"- Hard-gate evidence: `{status['hard_gate_evidence']}`",
        f"- Database integrity: `{store.integrity_check()}`",
        "",
        "## Leading work candidates",
        "",
    ]
    for item in discovery[:25]:
        lines.append(
            f"- `{item['candidate_id']}` — `{item['expression']}`; front "
            f"`{item['pareto_front']}`, hard passes `{item['hard_passes']}`, "
            f"remaining claims `{len(item['remaining_claims'])}`."
        )
    lines.extend(["", "## Failure clusters", ""])
    if clusters:
        lines.extend(
            f"- `{item['gate_id']}` / `{item['mechanism_tag']}`: {item['rejection_count']}"
            for item in clusters
        )
    else:
        lines.append("- No terminal hard-gate failure clusters recorded yet.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "A candidate is promoted only by completing hard gates. Historical results and "
                "LLM proposals can schedule work but cannot rescue a rejection."
            ),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    report["report_files"] = {
        "json": str(json_path),
        "json_sha256": sha256_file(json_path),
        "markdown": str(markdown_path),
        "markdown_sha256": sha256_file(markdown_path),
    }
    return report
