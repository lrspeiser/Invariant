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
            f'{_escape(counts.get("rejected", 0))} rejected, '
            f'{_escape(counts.get("blocked", 0))} blocked, '
            f'{_escape(counts.get("calibration_only", 0))} calibration-only)</summary>'
            f'<ul>{nodes}</ul><small>{_escape(dossier.get("status_label", "Overall"))}: '
            f'{_escape(dossier["overall_status"])}<br>'
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
    readiness = volatile["scheduler_readiness"]
    lane_rows = "".join(
        "<tr>"
        f"<td>{_escape(name)}</td><td>{_escape(lane['running'])}</td>"
        f"<td>{_escape(readiness[name]['runnable_now'])}</td>"
        f"<td>{_escape(readiness[name]['delayed_until_not_before'])}</td>"
        f"<td>{_escape(readiness[name]['earliest_future_not_before_utc'])}</td>"
        f"<td>{_escape(lane['capacity'])}</td>"
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
    scalable_outcomes = core["grammar_parameter_cells"][
        "scalable_unique_action_formal_outcomes"
    ]
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sigma Gravity Engine Status</title>
<style>
:root{{--bg:#0b1020;--panel:#151c31;--line:#2b3657;--text:#e8edf8;--muted:#9eabc7;--green:#4ade80;--red:#fb7185;--amber:#fbbf24;--blue:#60a5fa}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px/1.45 system-ui,sans-serif}} main{{max-width:1600px;margin:auto;padding:28px}} h1{{margin:0;font-size:28px}} h2{{font-size:17px;margin:0 0 14px}} p{{color:var(--muted)}} .head{{display:flex;justify-content:space-between;gap:20px;align-items:start;margin-bottom:22px}} .badge{{padding:7px 11px;border:1px solid var(--line);border-radius:999px;color:var(--green)}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px}} section{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:17px;margin-bottom:14px}} .metrics{{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px}} .metric{{background:#0d1428;border-radius:8px;padding:10px}} .metric span{{display:block;color:var(--muted);font-size:12px}} .metric strong{{font-size:18px}} .pass strong{{color:var(--green)}} .reject strong{{color:var(--red)}} .block strong{{color:var(--amber)}} table{{width:100%;border-collapse:collapse}} th,td{{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:8px}} th{{color:var(--muted)}} ul{{list-style:none;padding:0;margin:0}} li{{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--line)}} code{{color:var(--blue)}} .formula{{min-width:310px;max-width:540px;white-space:normal}} .formula>summary{{cursor:pointer;color:var(--green);font-weight:650}} .formula-action{{margin:8px 0;padding:9px;background:#0d1428;border-radius:7px;overflow-wrap:anywhere}} .formula p{{margin:7px 0}} .formula small{{color:var(--muted);overflow-wrap:anywhere}} .formula-terms,.formula-proof{{margin-top:8px}} .formula-terms>summary,.formula-proof>summary{{cursor:pointer;color:var(--blue)}} .formula-terms li{{display:block;overflow-wrap:anywhere}} .proof-node{{justify-content:start;gap:8px;align-items:start}} .proof-status{{min-width:92px;padding:2px 5px;border-radius:5px;text-align:center;font-size:11px}} .proof-proven{{color:var(--green);border:1px solid var(--green)}} .proof-rejected{{color:var(--red);border:1px solid var(--red)}} .proof-blocked{{color:var(--amber);border:1px solid var(--amber)}} .proof-calibration_only{{color:var(--blue);border:1px solid var(--blue)}} footer{{color:var(--muted);font-size:12px;margin-top:18px;overflow-wrap:anywhere}}
</style>
</head>
<body><main>
<div class="head"><div><h1>Sigma Gravity Engine</h1><p>Read-only evidence and execution status</p></div><div class="badge">{_escape(campaign['state'])}</div></div>
<div class="grid">
<section><h2>Live campaign evidence</h2><div class="metrics">{_outcomes(campaign['normalized_evidence_outcomes'])}</div><p>Deadline: {_escape(campaign['deadline_utc'])} · {_escape(volatile['deadline_state'])}</p><p>Freshness: {_escape(freshness_text)}</p></section>
<section><h2>Formal promotion overlay</h2><div class="metrics">{_outcomes(core['promotion_overlay']['formal'])}</div><p>Observational gates opened: {_escape(core['promotion_overlay']['observational_opened'])}</p></section>
<section><h2>Conformal G4 Solar evaluator</h2><div class="metrics">{_metric('Decision', core['g4_solar_evaluator']['decision'])}{_metric('Registered', core['g4_solar_evaluator']['filled_registration_hash_count'])}{_metric('Still missing', core['g4_solar_evaluator']['missing_registration_hash_count'])}{_metric('Leased executions', core['g4_solar_evaluator']['durable_execution']['task_count'])}{_metric('GR controls', core['g4_solar_evaluator']['synthetic_GR_golden_pass_count'])}</div><p>Reviewed descriptor ready: {_escape(core['g4_solar_evaluator']['descriptor_implementation_ready'])}. Observations opened: {_escape(core['g4_solar_evaluator']['observational_data_opened'])}. First missing premise: {_escape(core['g4_solar_evaluator']['first_missing_premise'])}.</p></section>
<section><h2>Conformal G4 galaxy evaluator</h2><div class="metrics">{_metric('Decision', core['g4_galaxy_evaluator']['decision'])}{_metric('Verified registrations', core['g4_galaxy_evaluator']['registration']['filled_registration_hash_count'])}{_metric('Still missing', core['g4_galaxy_evaluator']['registration']['missing_registration_hash_count'])}{_metric('Analytic controls', str(core['g4_galaxy_evaluator']['forward_model']['analytic_known_answer_pass_count']) + '/3')}{_metric('Object-specific gravity parameters', core['g4_galaxy_evaluator']['object_specific_gravity_parameter_count'])}</div><p>Rotation/lensing implementations, the conditional branch/domain and non-redshift geometry contracts, plus shared-calibration, covariance, likelihood, and stopping policies are hash-bound. The manifest auditor, prediction-bundle builder, and source-registry admission callback are implemented but disabled; synthetic fixtures are not registration evidence. Source records admitted: {_escape(core['g4_galaxy_evaluator']['registration']['source_registry_admission']['source_records_admitted'])}. Real split commitment registered: {_escape(core['g4_galaxy_evaluator']['registration']['real_split_commitment_registered'])}. Prediction bundle registered: {_escape(core['g4_galaxy_evaluator']['registration']['prediction_bundle_registered'])}. Observations opened: {_escape(core['g4_galaxy_evaluator']['registration']['observational_data_opened'])}. First missing premise: {_escape(core['g4_galaxy_evaluator']['registration']['source_registry_admission']['first_missing_premise'])}.</p></section>
<section><h2>Grammar-v3 scalable candidates</h2><div class="metrics">{_outcomes(scalable_outcomes)}{_metric('Reviewed parameter cells', core['grammar_parameter_cells']['reviewed_manifest']['parameter_cell_count'])}{_metric('Unique typed actions', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['unique_candidate_count'])}{_metric('Equivalent aliases', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['equivalent_duplicate_count'])}{_metric('Explained candidates', core['grammar_parameter_cells']['explanation_dossiers']['candidate_count'])}{_metric('Structural rows', core['grammar_parameter_cells']['structural_metrics']['candidate_count'])}{_metric('Simplicity Pareto front', core['grammar_parameter_cells']['structural_metrics']['simplicity_pareto_front']['candidate_count'])}{_metric('Formal queues admitted', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['eligible_candidate_count'])}{_metric('Future reviewed cells', core['grammar_parameter_cells']['staged_epoch']['reviewed_future_chunk']['input_cell_count'])}{_metric('Future new candidates', core['grammar_parameter_cells']['staged_epoch']['reviewed_future_chunk']['disposition_counts']['admitted_new_candidate'])}{_metric('Future preflight passes', core['grammar_parameter_cells']['staged_epoch']['reviewed_future_chunk']['preflight']['decision_counts']['pass'])}{_metric('Future preflight rejects', core['grammar_parameter_cells']['staged_epoch']['reviewed_future_chunk']['preflight']['decision_counts']['reject'])}{_metric('Future preflight blocks', core['grammar_parameter_cells']['staged_epoch']['reviewed_future_chunk']['preflight']['decision_counts']['blocked'])}{_metric('G4 preflight blocked', core['grammar_parameter_cells']['scalable_preflight_blocked_excluded_count'])}{_metric('Aether rejected', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['family_formal_execution']['aether']['decision_counts']['reject'])}{_metric('Aether blocked', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['family_formal_execution']['aether']['decision_counts']['blocked'])}{_metric('G2 formal passes', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['family_formal_execution']['g2']['decision_counts']['pass'])}{_metric('G2 Solar analytic branches', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['family_formal_execution']['g2']['solar_readiness']['analytic_prediction_pass_count'])}{_metric('G2 Solar fields remaining', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['family_formal_execution']['g2']['solar_readiness']['registration_advance']['after_missing_field_count'])}{_metric('G3 blocked', core['grammar_parameter_cells']['reviewed_manifest']['compilation']['formal_preflight']['promotion_admission']['family_formal_execution']['g3']['decision_counts']['blocked'])}</div><p>Across the 163 unique typed actions, the current exact formal tally is {_escape(scalable_outcomes['pass'])} pass, {_escape(scalable_outcomes['reject'])} reject, and {_escape(scalable_outcomes['block'])} blocked; the older six-seed execution is tracked separately. All 163 now have a hash-bound defining action, exact structural measurements, and a separate candidate-specific proof hierarchy. Structural and alias rankings never imply scientific validity or literature novelty. Durable preflight admitted 162 into family queues and initially kept one conformal-G4 action blocked. A candidate-specific exact covariant-density and domain-inclusion follow-up now transfers the reviewed action-level formal pass to that G4 action; Solar and observational validation remain separate and blocked. Exact Aether necessary gates reject two spin-0-degenerate actions and leave 126 blocked on twist/coercivity or generic nonlinear-energy premises. Both k-essence actions now also pass the reviewed nonmaximal positive-mass theorem on the explicitly registered complete boundaryless asymptotically-Euclidean constraint-data domain. Each has an exact constant-scalar GR-like Solar prediction branch with G_cav/G*=1 and PPN gamma=beta=1. Six action-bound protocol registrations are now sealed, leaving four external fields: real source/domain instantiation, an actual held-out split commitment, selected primary roots, and separate observation authorization. Both candidates remain unranked, with zero primary or held-out accesses and zero real-data passes. A new reviewed 32-cell epoch chunk compiled to 19 new action classes and 13 exact deduplications. Reviewed preflight now gives 14 Aether prerequisite passes, two exact Aether principal-mode rejects, and three cubic-G3 blocks because their qualitative jet domains do not instantiate the componentwise normalized common-cone box. These are prerequisite outcomes, not full-formal passes; no automatic downstream enqueue occurred. All 32 earlier cubic-G3 actions pass local principal, common-cone and periodic Dirac gates. Their first exact blocker is lack of a uniformly invertible lapse operator on the registered asymptotically-flat decaying-gradient domain; an AF Einstein-constraint solution and global energy are separately unproved.</p></section>
<section><h2>Quartic nonlinear closure</h2><div class="metrics">{_metric('Candidates', core['quartic_nonlinear_closure']['candidate_count'])}{_metric('Coordinate pairs', core['quartic_nonlinear_closure']['coordinate_pair_partition']['total_unordered_coordinate_pairs'])}{_metric('Two-jets closed', core['quartic_nonlinear_closure']['quadratic_deltaK_two_jet']['closed_candidate_count'])}{_metric('D2 deltaK ceiling', core['quartic_nonlinear_closure']['quadratic_deltaK_two_jet']['D2_coordinate_linf_to_Frobenius_ceiling'])}{_metric('Diagonal third jets', core['quartic_nonlinear_closure']['diagonal_third_jet']['diagonal_triples_closed'])}{_metric('Third-jet evaluations', core['quartic_nonlinear_closure']['diagonal_third_jet']['candidate_direction_evaluations'])}{_metric('Mixed triples remaining', core['quartic_nonlinear_closure']['diagonal_third_jet']['remaining_mixed_triples'])}{_metric('Full tube identities', core['quartic_nonlinear_closure']['closure_counts']['full_tube_Sylvester_identities'])}{_metric('Global H7 closures', core['quartic_nonlinear_closure']['closure_counts']['global_H7_closures'])}</div><p>All coordinate pairs are classified and all 12 reference quadratic deltaK two-jets satisfy the Sylvester identity through derivative orders 0, 1, and 2. The exact third-order recurrence now also passes all 41 diagonal active-coordinate triples and all 492 candidate-direction evaluations. This is not a full tube solution: 12,300 polarized mixed triples remain before fourth-and-higher remainder control or a nonlinear range theorem. First missing premise: {_escape(core['quartic_nonlinear_closure']['first_missing_premise'])}. CK1, CK3, TC2, B7, global H7, and lifespan remain fail-closed.</p></section>
<section><h2>Physical hardware sample</h2><div class="metrics">{gpu_metrics}</div><p>Source: {_escape(gpu.get('source', 'not sampled'))}. This is separate from scheduler occupancy.</p></section>
</div>
<section><h2>Scheduler lanes</h2><table><thead><tr><th>Lane</th><th>Running</th><th>Runnable now</th><th>Scheduled later</th><th>Earliest ready</th><th>Capacity</th><th>Scheduler occupancy</th></tr></thead><tbody>{lane_rows}</tbody></table><p>Runnable and scheduled counts honor each task's durable <code>not_before_utc</code>; scheduler occupancy remains distinct from CPU/GPU hardware utilization.</p></section>
<div class="grid">
<section><h2>Current missing-evaluator blockers</h2><ul>{blocker_rows}</ul></section>
<section><h2>LLM budget and proposal quarantine</h2><div class="metrics">{_metric('Campaign budget', '$' + str(core['llm']['configured_budget_usd']))}{_metric('Campaign spent', '$' + str(core['llm']['spent_usd']))}{_metric('Adapter cap', '$' + core['llm']['proposal_adapter']['maximum_total_usd'])}{_metric('Adapter calls', core['llm']['proposal_adapter']['network_calls_made'])}{_metric('Local epoch candidates', core['llm']['reviewed_local_epoch']['expected_bounded_status']['candidate_count'])}{_metric('Local policy passes', core['llm']['reviewed_local_epoch']['expected_bounded_status']['policy_pass_count'])}</div><p>Adapter: {_escape(core['llm']['proposal_adapter']['status'])}. Campaign bridge: {_escape(core['llm']['campaign_bridge']['status'])}. Typed-DSL admission: {_escape(core['llm']['typed_dsl_admission']['status'])}. Candidate-registry bridge: {_escape(core['llm']['compiler_registry_bridge']['status'])}. Local composed epoch: {_escape(core['llm']['reviewed_local_epoch']['status'])}. Restart-safe service: {_escape(core['llm']['reviewed_local_service']['status'])}. Paid calls and durable formula-body persistence are disabled by default; future outputs remain {_escape(core['llm']['proposal_adapter']['output_status'])}, and only exact hash-bound proposals can enter the separate reviewed compiler queue.</p></section>
<section><h2>Billion-formula screen</h2><div class="metrics">{_metric('Source formulas', core['billion_formula_streaming']['source_formula_count'])}{_metric('Static survivors', core['billion_formula_streaming']['sampled_static_stage']['pass'])}{_metric('Lift rejected', core['billion_formula_streaming']['promotion_stage']['lift_reject'])}{_metric('Lift blocked', core['billion_formula_streaming']['promotion_stage']['lift_block'])}</div></section>
</div>
<section><h2>How to read a candidate theory</h2><p>Open a candidate name to see its defining action: this is the compact master formula. The field equations, constraints, observable predictions, and pass/block certificates are derived from that action and appear in the separate proof and test hierarchy. They are supporting equations, not extra fitted pieces silently added to the theory.</p></section>
{leaderboard_html}
<footer>Core SHA-256: {_escape(snapshot['core_content_sha256'])} · sampled {_escape(volatile['sampled_at_utc'])} · {source_revision_count} immutable source revisions. Cross-pipeline totals are intentionally not summed because candidate sets and gate semantics overlap.</footer>
</main></body></html>"""


def validate_dashboard_input(snapshot: Mapping[str, Any], expected_core_sha: str) -> None:
    if snapshot.get("core_content_sha256") != expected_core_sha:
        raise ValueError("dashboard snapshot core hash mismatch")
    encoded = json.dumps(snapshot, sort_keys=True)
    if "file://" in encoded.lower():
        raise ValueError("dashboard snapshot contains a local file URI")


def _rank_label(row: Mapping[str, Any]) -> str | int:
    if row["rank"] is not None:
        return row["rank"]
    if row.get("comparison_group_rank") is not None:
        return f"class #{row['comparison_group_rank']}"
    return "unranked"


def _leaderboards_html(core: Mapping[str, Any]) -> str:
    leaderboards = core.get("scientific_leaderboards")
    if not leaderboards:
        return ""
    sections = []
    dossiers = leaderboards["theory_dossiers"]
    for category, board in sorted(leaderboards["categories"].items()):
        unranked_rows = board["unranked_blocked_or_untested"]
        displayed_unranked = unranked_rows[:25]
        display_rows = (
            board["top10"]
            + board.get("completed_incomparable_evidence", [])
            + displayed_unranked
        )
        rows = "".join(
            "<tr>"
            f"<td>{_escape(_rank_label(row))}</td><td><code>{_escape(row['candidate_id'])}</code></td>"
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
            f"completed in separate evidence classes: {_escape(board.get('completed_separate_class_count', 0))}; "
            f"blocked/untested: {_escape(board['unranked_count'])} "
            f"(showing {_escape(len(displayed_unranked))}; full rows remain in the JSON snapshot). Availability: "
            f"{_escape(board['availability'])}. {_escape(board['absence_reason'] or '')}</p>"
            "<div style=\"overflow:auto\"><table><thead><tr><th>Rank</th><th>Candidate</th>"
            "<th>Theory formula</th><th>Role</th><th>Exact metrics</th><th>Status</th><th>Data class</th>"
            "<th>Gate completeness</th><th>Blocker</th><th>Artifact content SHA</th>"
            f"<th>Uncertainty</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
        )
    return "".join(sections)
