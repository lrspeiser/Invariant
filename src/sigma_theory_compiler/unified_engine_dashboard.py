"""Small dependency-free HTML view for a unified engine status snapshot."""

from __future__ import annotations

import html
import json
from collections.abc import Mapping
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _metric(label: str, value: Any, css_class: str = "") -> str:
    return (
        f'<div class="metric {css_class}"><span>{_escape(label)}</span>'
        f"<strong>{_escape(value)}</strong></div>"
    )


def _outcomes(values: Mapping[str, Any]) -> str:
    ordered = [key for key in ("pass", "reject", "block") if key in values]
    ordered.extend(sorted(set(values) - set(ordered)))
    return "".join(
        _metric(name, "not reported" if values[name] is None else values[name], name)
        for name in ordered
    )


def _formula_html(
    row: Mapping[str, Any], dossiers: Mapping[str, Mapping[str, Any]]
) -> str:
    formula = row["theory_formula"]
    parameters = formula["parameters"]
    parameter_text = ", ".join(
        f"{key}={value}" for key, value in sorted(parameters.items())
    ) or "none"
    field_text = ", ".join(formula["fields"]) or "see bound artifact"
    operator_terms = "".join(
        f"<li><code>{_escape(term)}</code></li>"
        for term in formula["operator_terms"]
    ) or "<li>No exact operator expansion is attached to this row.</li>"
    action_hash = formula["action_content_sha256"]
    action_hash_html = (
        f"<br><small>Action SHA: <code>{_escape(action_hash)}</code></small>"
        if action_hash
        else ""
    )
    dossier_id = (
        "GR-EINSTEIN-HILBERT"
        if row["candidate_id"] == "KNOWN-ANSWER-EINSTEIN-HILBERT"
        else row["candidate_id"]
    )
    dossier = dossiers.get(dossier_id)
    dossier_html = ""
    if dossier is not None:
        counts = dossier["hierarchy_status_counts"]
        nodes = "".join(
            '<li class="proof-node">'
            f'<span class="proof-status proof-{_escape(node["status"])}">'
            f'{_escape(node["status"])}</span><div><strong>{_escape(node["node_id"])}</strong>'
            f'<br><small>{_escape(node["scope"])}</small></div></li>'
            for node in dossier["hierarchy_nodes"]
        )
        dossier_html = (
            '<details class="formula-proof"><summary>Proof and test hierarchy '
            f'({_escape(counts.get("proven", 0))} proven, '
            f'{_escape(counts.get("blocked", 0))} blocked, '
            f'{_escape(counts.get("calibration_only", 0))} calibration-only)</summary>'
            f'<ul>{nodes}</ul><small>Overall: {_escape(dossier["overall_status"])}<br>'
            f'Dossier: <code>{_escape(dossier["artifact_link"])}</code><br>'
            f'Dossier SHA: <code>{_escape(dossier["content_sha256"])}</code></small></details>'
        )
    return (
        '<details class="formula"><summary>'
        f"{_escape(formula['title'])}</summary>"
        f'<div class="formula-action"><code>{_escape(formula["defining_action"])}</code></div>'
        f"<p>{_escape(formula['plain_language'])}</p>"
        f"<small>Fields: {_escape(field_text)}<br>Parameters: {_escape(parameter_text)}</small>"
        "<details class=\"formula-terms\"><summary>Derived operator terms / evidence scope</summary>"
        f"<ul>{operator_terms}</ul><p>{_escape(formula['scope_note'])}</p></details>"
        f"{dossier_html}{action_hash_html}</details>"
    )


def render_dashboard(snapshot: Mapping[str, Any]) -> str:
    """Render only the redacted snapshot; this function never opens source files."""
    core = snapshot["core"]
    volatile = snapshot["volatile"]
    campaign = core["campaign_watchdog"]
    gpu = volatile["physical_gpu"]
    lanes = core["scheduler_lanes"]
    lane_rows = "".join(
        "<tr>"
        f"<td>{_escape(name)}</td><td>{_escape(lane['running'])}</td>"
        f"<td>{_escape(lane['queued'])}</td><td>{_escape(lane['capacity'])}</td>"
        f"<td>{_escape(lane['scheduler_occupancy_fraction'])}</td>"
        "</tr>"
        for name, lane in sorted(lanes.items())
    )
    blockers = core["followup_service"]["current_missing_evaluator_blockers"]
    blocker_rows = "".join(
        f"<li><code>{_escape(name)}</code><strong>{_escape(count)}</strong></li>"
        for name, count in sorted(blockers.items())
    ) or "<li>None</li>"
    freshness = volatile["campaign_watchdog_freshness"]
    freshness_text = (
        freshness["stale_source_reason"] or "fresh under configured threshold"
    )
    gpu_metrics = (
        _metric("Physical GPU utilization", f"{gpu.get('utilization_percent')}%")
        + _metric(
            "VRAM",
            f"{gpu.get('memory_used_mib')} / {gpu.get('memory_total_mib')} MiB",
        )
        if gpu.get("availability") == "available"
        else _metric("Physical GPU telemetry", gpu.get("reason", "unavailable"))
    )
    source_revision_count = len(core["source_revisions"])
    leaderboard_html = _leaderboards_html(core)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sigma Gravity Engine Status</title>
<style>
:root{{--bg:#0b1020;--panel:#151c31;--line:#2b3657;--text:#e8edf8;--muted:#9eabc7;--green:#4ade80;--red:#fb7185;--amber:#fbbf24;--blue:#60a5fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}} main{{max-width:1600px;margin:auto;padding:28px}} h1{{margin:0;font-size:28px}} h2{{font-size:17px;margin:0 0 14px}} p{{color:var(--muted)}} .head{{display:flex;justify-content:space-between;gap:20px;align-items:start;margin-bottom:22px}} .badge{{padding:7px 11px;border:1px solid var(--line);border-radius:999px;color:var(--green)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}} section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:17px;margin-bottom:14px}} .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px}} .metric{{background:#0d1428;border-radius:8px;padding:10px}} .metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{font-size:18px}} .pass strong{{color:var(--green)}} .reject strong{{color:var(--red)}} .block strong{{color:var(--amber)}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:8px}} th{{color:var(--muted)}} ul{{list-style:none;padding:0;margin:0}} li{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--line)}} code{{color:var(--blue)}} .formula{{min-width:310px;max-width:540px;white-space:normal}} .formula>summary{{cursor:pointer;color:var(--green);font-weight:650}} .formula-action{{margin:8px 0;padding:9px;background:#0d1428;border-radius:7px;overflow-wrap:anywhere}} .formula p{{margin:7px 0}} .formula small{{color:var(--muted);overflow-wrap:anywhere}} .formula-terms,.formula-proof{{margin-top:8px}} .formula-terms>summary,.formula-proof>summary{{cursor:pointer;color:var(--blue)}} .formula-terms li{{display:block;overflow-wrap:anywhere}} .proof-node{{justify-content:start;gap:8px;align-items:start}} .proof-status{{min-width:92px;padding:2px 5px;border-radius:5px;text-align:center;font-size:11px}} .proof-proven{{color:var(--green);border:1px solid var(--green)}} .proof-blocked{{color:var(--amber);border:1px solid var(--amber)}} .proof-calibration_only{{color:var(--blue);border:1px solid var(--blue)}} footer{{color:var(--muted);font-size:12px;margin-top:18px;overflow-wrap:anywhere}}
</style>
</head>
<body><main>
<div class="head"><div><h1>Sigma Gravity Engine</h1><p>Read-only evidence and execution status</p></div><div class="badge">{_escape(campaign['state'])}</div></div>
<div class="grid">
<section><h2>Live campaign evidence</h2><div class="metrics">{_outcomes(campaign['normalized_evidence_outcomes'])}</div><p>Deadline: {_escape(campaign['deadline_utc'])} · {_escape(volatile['deadline_state'])}</p><p>Freshness: {_escape(freshness_text)}</p></section>
<section><h2>Formal promotion overlay</h2><div class="metrics">{_outcomes(core['promotion_overlay']['formal'])}</div><p>Observational gates opened: {_escape(core['promotion_overlay']['observational_opened'])}</p></section>
<section><h2>Grammar-v3 candidates</h2><div class="metrics">{_outcomes(core['grammar_parameter_cells']['normalized_scientific_outcomes'])}</div><p>Follow-up packets processed: {_escape(core['followup_service']['processed'])}; deferred: {_escape(core['followup_service']['deferred'])}</p></section>
<section><h2>Physical hardware sample</h2><div class="metrics">{gpu_metrics}</div><p>Source: {_escape(gpu.get('source', 'not sampled'))}. This is separate from scheduler occupancy.</p></section>
</div>
<section><h2>Scheduler lanes</h2><table><thead><tr><th>Lane</th><th>Running</th><th>Queued</th><th>Capacity</th><th>Scheduler occupancy</th></tr></thead><tbody>{lane_rows}</tbody></table></section>
<div class="grid">
<section><h2>Current missing-evaluator blockers</h2><ul>{blocker_rows}</ul></section>
<section><h2>LLM budget and proposal quarantine</h2><div class="metrics">{_metric('Campaign budget', '$' + str(core['llm']['configured_budget_usd']))}{_metric('Campaign spent', '$' + str(core['llm']['spent_usd']))}{_metric('Adapter cap', '$' + core['llm']['proposal_adapter']['maximum_total_usd'])}{_metric('Adapter calls', core['llm']['proposal_adapter']['network_calls_made'])}</div><p>Adapter: {_escape(core['llm']['proposal_adapter']['status'])}. Paid calls are disabled by default; any future output remains {_escape(core['llm']['proposal_adapter']['output_status'])}.</p></section>
<section><h2>Billion-formula screen</h2><div class="metrics">{_metric('Source formulas', core['billion_formula_streaming']['source_formula_count'])}{_metric('Static survivors', core['billion_formula_streaming']['sampled_static_stage']['pass'])}{_metric('Lift rejected', core['billion_formula_streaming']['promotion_stage']['lift_reject'])}{_metric('Lift blocked', core['billion_formula_streaming']['promotion_stage']['lift_block'])}</div></section>
</div>
{leaderboard_html}
<footer>Core SHA-256: {_escape(snapshot['core_content_sha256'])} · sampled {_escape(volatile['sampled_at_utc'])} · {source_revision_count} immutable source revisions. Cross-pipeline totals are intentionally not summed because candidate sets and gate semantics overlap.</footer>
</main></body></html>"""


def validate_dashboard_input(snapshot: Mapping[str, Any], expected_core_sha: str) -> None:
    if snapshot.get("core_content_sha256") != expected_core_sha:
        raise ValueError("dashboard snapshot core hash mismatch")
    encoded = json.dumps(snapshot, sort_keys=True)
    if "file://" in encoded.lower():
        raise ValueError("dashboard snapshot contains a local file URI")


def _leaderboards_html(core: Mapping[str, Any]) -> str:
    leaderboards = core.get("scientific_leaderboards")
    if not leaderboards:
        return ""
    sections = []
    dossiers = leaderboards["theory_dossiers"]
    for category, board in sorted(leaderboards["categories"].items()):
        display_rows = board["top10"] + board["unranked_blocked_or_untested"]
        rows = "".join(
            "<tr>"
            f"<td>{_escape(row['rank'] if row['rank'] is not None else 'unranked')}</td><td><code>{_escape(row['candidate_id'])}</code></td>"
            f"<td>{_formula_html(row, dossiers)}</td><td>{_escape(row['role'])}</td><td>{_escape(json.dumps(row['metrics'], sort_keys=True, separators=(',', ':')))}</td>"
            f"<td>{_escape(row['evidence_status'])}</td><td>{_escape(row['data_class'])}</td>"
            f"<td>{_escape(row['gate_completeness'])}</td><td>{_escape(row['blocker'])}</td>"
            f"<td><code>{_escape(row['lineage']['artifact_link'])}</code><br><code>{_escape(row['lineage']['artifact_content_sha256'])}</code></td>"
            f"<td>{_escape(row['uncertainty'])}</td>"
            "</tr>"
            for row in display_rows
        )
        if not rows:
            rows = '<tr><td colspan="11">No completed or blocked candidate evidence is available.</td></tr>'
        sections.append(
            f'<section><h2>Category leaderboard: {_escape(category)}</h2>'
            f"<p>{_escape(board['ranking_scope'])}. Ranked: {_escape(board['ranked_count'])}; "
            f"blocked/untested: {_escape(board['unranked_count'])}. Availability: "
            f"{_escape(board['availability'])}. {_escape(board['absence_reason'] or '')}</p>"
            "<div style=\"overflow:auto\"><table><thead><tr><th>Rank</th><th>Candidate</th>"
            "<th>Theory formula</th><th>Role</th><th>Exact metrics</th><th>Status</th><th>Data class</th>"
            "<th>Gate completeness</th><th>Blocker</th><th>Artifact content SHA</th>"
            f"<th>Uncertainty</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
        )
    return "".join(sections)
