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


def _formula_html(row: Mapping[str, Any], dossiers: Mapping[str, Mapping[str, Any]]) -> str:
    formula = row["theory_formula"]
    parameters = formula["parameters"]
    parameter_text = (
        ", ".join(f"{key}={value}" for key, value in sorted(parameters.items())) or "none"
    )
    field_text = ", ".join(formula["fields"]) or "see bound artifact"
    operator_terms = (
        "".join(f"<li><code>{_escape(term)}</code></li>" for term in formula["operator_terms"])
        or "<li>No exact operator expansion is attached to this row.</li>"
    )
    action_hash = formula["action_content_sha256"]
    action_hash_html = (
        f"<br><small>Action SHA: <code>{_escape(action_hash)}</code></small>" if action_hash else ""
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
            f"{_escape(node['status'])}</span><div><strong>{_escape(node['node_id'])}</strong>"
            f"<br><small>{_escape(node['scope'])}</small></div></li>"
            for node in dossier["hierarchy_nodes"]
        )
        dossier_html = (
            '<details class="formula-proof"><summary>Proof and test hierarchy '
            f"({_escape(counts.get('proven', 0))} proven, "
            f"{_escape(counts.get('rejected', 0))} rejected, "
            f"{_escape(counts.get('blocked', 0))} blocked, "
            f"{_escape(counts.get('calibration_only', 0))} calibration-only)</summary>"
            f"<ul>{nodes}</ul><small>{_escape(dossier.get('status_label', 'Overall'))}: "
            f"{_escape(dossier['overall_status'])}<br>"
            f"Dossier: <code>{_escape(dossier['artifact_link'])}</code><br>"
            f"Dossier SHA: <code>{_escape(dossier['content_sha256'])}</code></small></details>"
        )
    return (
        '<details class="formula"><summary>'
        f"{_escape(formula['title'])}</summary>"
        f'<div class="formula-action"><code>{_escape(formula["defining_action"])}</code></div>'
        f"<p>{_escape(formula['plain_language'])}</p>"
        f"<small>Fields: {_escape(field_text)}<br>Parameters: {_escape(parameter_text)}</small>"
        '<details class="formula-terms"><summary>Derived operator terms / evidence scope</summary>'
        f"<ul>{operator_terms}</ul><p>{_escape(formula['scope_note'])}</p></details>"
        f"{dossier_html}{action_hash_html}</details>"
    )


def _future_dossiers_html(core: Mapping[str, Any]) -> str:
    staged = core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]
    dossier_set = staged["action_dossiers"]
    cards = []
    for record in dossier_set["records"]:
        action = record["action"]
        parameters = (
            ", ".join(f"{key}={value}" for key, value in sorted(action["parameters"].items()))
            or "none"
        )
        fields = ", ".join(action["fields"])
        terms = "".join(
            f"<li><code>{_escape(operator['density'])}</code></li>"
            for operator in action["ordered_operator_densities"]
        )
        counts = {
            status: sum(node["status"] == status for node in record["hierarchy_nodes"])
            for status in ("proven", "rejected", "blocked")
        }
        nodes = "".join(
            '<li class="proof-node">'
            f'<span class="proof-status proof-{_escape(node["status"])}">'
            f"{_escape(node['status'])}</span><div><strong>{_escape(node['node_id'])}</strong>"
            f"<br><small>{_escape(node['scope'])}</small></div></li>"
            for node in record["hierarchy_nodes"]
        )
        cards.append(
            '<details class="formula"><summary>'
            f"{_escape(record['candidate_id'])} · {_escape(record['family_id'])} · "
            f"{_escape(record['formal_decision'])}</summary>"
            f'<div class="formula-action"><code>{_escape(action["human_readable_action"]["display_text"])}</code></div>'
            f"<small>Fields: {_escape(fields)}<br>Parameters: {_escape(parameters)}</small>"
            '<details class="formula-terms"><summary>Exact ordered covariant densities</summary>'
            f"<ul>{terms}</ul><p>{_escape(action['human_readable_action']['scope'])}</p></details>"
            '<details class="formula-proof"><summary>Proof and test hierarchy '
            f"({_escape(counts['proven'])} proven, {_escape(counts['rejected'])} rejected, "
            f"{_escape(counts['blocked'])} blocked)</summary><ul>{nodes}</ul>"
            f"<small>Formal decision: {_escape(record['formal_decision'])}<br>"
            f"First blocker: {_escape(record['first_blocker'])}<br>"
            f"Action SHA: <code>{_escape(action['action_sha256'])}</code></small></details></details>"
        )
    return (
        '<section><h2>Staged future candidate formulas (unranked)</h2><div class="metrics">'
        f"{_metric('Candidates', dossier_set['candidate_count'])}"
        f"{_metric('Blocked', dossier_set['decision_counts']['blocked'])}"
        f"{_metric('Rejected', dossier_set['decision_counts']['reject'])}"
        f"{_metric('Ranked', dossier_set['ranked_candidate_count'])}</div>"
        "<p>These master actions are recompiled from the exact typed cells. Their proof nodes are "
        "shown separately; blocked and rejected staged actions never enter a scientific ranking.</p>"
        + "".join(cards)
        + "</section>"
    )


def render_dashboard(snapshot: Mapping[str, Any]) -> str:
    """Render only the redacted snapshot; this function never opens source files."""
    core = snapshot["core"]
    volatile = snapshot["volatile"]
    campaign = core["campaign_watchdog"]
    cpu = volatile["physical_cpu"]
    gpu = volatile["physical_gpu"]
    lanes = core["scheduler_lanes"]
    readiness = volatile["scheduler_readiness"]
    live_dashboard = volatile.get(
        "unified_live_dashboard_service",
        {"availability": "not_configured", "alive": False, "refresh_count": 0},
    )
    continuous_dashboard = core["continuous_dashboard"]
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
    blocker_rows = (
        "".join(
            f"<li><code>{_escape(name)}</code><strong>{_escape(count)}</strong></li>"
            for name, count in sorted(blockers.items())
        )
        or "<li>None</li>"
    )
    freshness = volatile["campaign_watchdog_freshness"]
    freshness_text = freshness["stale_source_reason"] or "fresh under configured threshold"
    cpu_metrics = (
        _metric("Physical CPU utilization", f"{cpu.get('utilization_percent')}%")
        + _metric(
            "CPU topology",
            f"{cpu.get('physical_cores')} cores / {cpu.get('logical_processors')} logical",
        )
        + _metric(
            "Host RAM",
            f"{round(cpu.get('memory_used_bytes', 0) / 2**30, 2)} / "
            f"{round(cpu.get('memory_total_bytes', 0) / 2**30, 2)} GiB",
        )
        if cpu.get("availability") == "available"
        else _metric("Physical CPU telemetry", cpu.get("reason", "unavailable"))
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
    future_dossier_html = _future_dossiers_html(core)
    scalable_outcomes = core["grammar_parameter_cells"]["scalable_unique_action_formal_outcomes"]
    g4_b4 = core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"][
        "generated_action_export"
    ]["generic_g4_B4_termwise_normalization"]
    gpu_formula = core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"][
        "generated_action_export"
    ]["gpu_synthetic_formula_stress"]
    transactional = core["transactional_gravity_proposal"]
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
<div class="head"><div><h1>Sigma Gravity Engine</h1><p>Read-only evidence and execution status</p></div><div class="badge">{_escape(campaign["state"])}</div></div>
<div class="grid">
<section><h2>Live campaign evidence</h2><div class="metrics">{_outcomes(campaign["normalized_evidence_outcomes"])}</div><p>Deadline: {_escape(campaign["deadline_utc"])} · {_escape(volatile["deadline_state"])}</p><p>Freshness: {_escape(freshness_text)}</p></section>
<section><h2>Live dashboard refresh service</h2><div class="metrics">{_metric("Readiness", continuous_dashboard["decision"])}{_metric("Availability", live_dashboard.get("availability"))}{_metric("State", live_dashboard.get("state"))}{_metric("Alive", live_dashboard.get("alive"))}{_metric("Refreshes", live_dashboard.get("refresh_count", 0))}{_metric("Refresh interval", str(continuous_dashboard["refresh_interval_seconds"]) + "s")}{_metric("Maximum refreshes", continuous_dashboard["maximum_refreshes"])}{_metric("Consecutive failures", live_dashboard.get("consecutive_failures", 0))}</div><p>This volatile panel is excluded from the deterministic core. The bounded service opens the campaign database with the read-only/query-only contract, publishes atomically into an ignored runtime directory, and never overwrites the immutable checked snapshot.</p></section>
<section><h2>Formal promotion overlay</h2><div class="metrics">{_outcomes(core["promotion_overlay"]["formal"])}</div><p>Observational gates opened: {_escape(core["promotion_overlay"]["observational_opened"])}</p></section>
<section><h2>Conformal G4 Solar evaluator</h2><div class="metrics">{_metric("Decision", core["g4_solar_evaluator"]["decision"])}{_metric("Registered", core["g4_solar_evaluator"]["filled_registration_hash_count"])}{_metric("Still missing", core["g4_solar_evaluator"]["missing_registration_hash_count"])}{_metric("Leased executions", core["g4_solar_evaluator"]["durable_execution"]["task_count"])}{_metric("GR controls", core["g4_solar_evaluator"]["synthetic_GR_golden_pass_count"])}</div><p>Reviewed descriptor ready: {_escape(core["g4_solar_evaluator"]["descriptor_implementation_ready"])}. Observations opened: {_escape(core["g4_solar_evaluator"]["observational_data_opened"])}. First missing premise: {_escape(core["g4_solar_evaluator"]["first_missing_premise"])}.</p></section>
<section><h2>Conformal G4 galaxy evaluator</h2><div class="metrics">{_metric("Decision", core["g4_galaxy_evaluator"]["decision"])}{_metric("Verified registrations", core["g4_galaxy_evaluator"]["registration"]["filled_registration_hash_count"])}{_metric("Still missing", core["g4_galaxy_evaluator"]["registration"]["missing_registration_hash_count"])}{_metric("Analytic controls", str(core["g4_galaxy_evaluator"]["forward_model"]["analytic_known_answer_pass_count"]) + "/3")}{_metric("Object-specific gravity parameters", core["g4_galaxy_evaluator"]["object_specific_gravity_parameter_count"])}</div><p>Rotation/lensing implementations, the conditional branch/domain and non-redshift geometry contracts, plus shared-calibration, covariance, likelihood, and stopping policies are hash-bound. The manifest auditor, prediction-bundle builder, and source-registry admission callback are implemented but disabled; synthetic fixtures are not registration evidence. Source records admitted: {_escape(core["g4_galaxy_evaluator"]["registration"]["source_registry_admission"]["source_records_admitted"])}. Real split commitment registered: {_escape(core["g4_galaxy_evaluator"]["registration"]["real_split_commitment_registered"])}. Prediction bundle registered: {_escape(core["g4_galaxy_evaluator"]["registration"]["prediction_bundle_registered"])}. Observations opened: {_escape(core["g4_galaxy_evaluator"]["registration"]["observational_data_opened"])}. First missing premise: {_escape(core["g4_galaxy_evaluator"]["registration"]["source_registry_admission"]["first_missing_premise"])}.</p></section>
<section><h2>Generic G4 equation B.4 normalization</h2><div class="metrics">{_metric("Canonical contractions", g4_b4["canonical_term_count"])}{_metric("Exact matches", str(g4_b4["matched_term_count"]) + "/24")}{_metric("Nonzero residuals", g4_b4["nonzero_residual_count"])}{_metric("Candidate formal passes inferred", g4_b4["full_candidate_formal_pass_inferred"])}</div><p>The independently executed Cadabra metric Euler coefficient matches all 24 canonical contractions in Kobayashi–Yamaguchi–Yokoyama equation B.4. The checked local coefficient transcription is hash-bound. This closes tensor spelling and coefficient normalization only; it does not establish the scalar equation, Noether identity, global energy, nonlinear stability, observations, or any candidate-level formal pass.</p></section>
<section><h2>RTX 5090 synthetic formula stress</h2><div class="metrics">{_metric("Candidate formulas", gpu_formula["counts"]["candidate_count"])}{_metric("Unique synthetic pairs", gpu_formula["counts"]["unique_candidate_point_pairs"])}{_metric("Exact CPU checks", gpu_formula["counts"]["cpu_exact_rational_crosschecks"])}{_metric("GPU/CPU violations", gpu_formula["gpu_cpu_comparison"]["violating_point_count"])}{_metric("Measured evaluations", gpu_formula["counts"]["gpu_measured_candidate_formula_evaluations"])}{_metric("Measured throughput", f'{gpu_formula["runtime_measurement"]["gpu_candidate_formula_evaluations_per_second"] / 1e9:.2f}B/s')}{_metric("Device-wide GPU mean", f'{gpu_formula["runtime_measurement"]["utilization"]["gpu_percent_mean"]:.2f}%')}</div><p>All 163 materialized formula projections were evaluated on deterministic synthetic dyadic operator coordinates. The full 5,341,184-value GPU output agrees with CPU evaluation within the registered error bounds, and 5,216 sentinel evaluations were independently checked with exact rational arithmetic. The 87.5-billion-evaluation timing loop is a single local throughput stress run. NVML counters are device-wide and can include concurrent processes; they are neither lane-only nor a sustained-capacity guarantee. Synthetic coordinates need not be realizable field jets, so this produces no field-equation proof, formal pass, ranking, rejection, or observational support.</p></section>
<section><h2>Kastner–Schlatter transactional-gravity proposal</h2><div class="metrics">{_metric("Decision", transactional["decision"])}{_metric("Equation checks", transactional["equation_preflight_counts"]["pass"])}{_metric("Graph nodes", transactional["equation_graph"]["counts"]["nodes"])}{_metric("Compiler action hypotheses", transactional["candidate_action_completion"]["counts"]["complete_local_deterministic_action_hypotheses"])}{_metric("Paper-derived actions", transactional["candidate_action_completion"]["counts"]["paper_derived_actions"])}{_metric("Registered fields", transactional["observational_readiness"]["registration_counts"]["by_status"]["source_registered"])}{_metric("Missing fields", transactional["observational_readiness"]["registration_counts"]["by_status"]["missing_required"])}{_metric("Observation access", transactional["observational_readiness"]["observational_access_count"])}{_metric("Synthetic power evals", transactional["cuda_falsification_design"]["counts"]["gpu_measured_value_evaluations"])}{_metric("Poisson alt detection", f'{100 * transactional["cuda_falsification_design"]["poisson_power_control"]["empirical_alternative_detection_rate"]:.2f}%')}{_metric("BTFR alt detection", f'{100 * transactional["cuda_falsification_design"]["btfr_power_control"]["empirical_alternative_detection_rate"]:.2f}%')}{_metric("Extended-source laws", transactional["extended_geometry_cuda_stress"]["counts"]["extended_source_laws_registered"])}{_metric("Geometry cases", transactional["extended_geometry_cuda_stress"]["counts"]["geometry_resolution_cases"])}{_metric("Source interactions", transactional["extended_geometry_cuda_stress"]["counts"]["gpu_measured_source_evaluation_interactions"])}{_metric("Lensing cases", transactional["extended_geometry_cuda_stress"]["counts"]["lensing_cases_executed"])}{_metric("Readiness advanced", transactional["cuda_falsification_design"]["counts"]["readiness_fields_advanced"])}{_metric("Scientific tests", transactional["cuda_falsification_design"]["counts"]["scientific_tests_passed"])}</div><p>The source-bound intake and typed graph represent the proposal as a dependency chain of 25 formulas, not one standalone proof. Two complete local EH-plus-scalar-intensity actions are compiler hypotheses: beta=1/2 matches Equation 35's first/middle normalization and beta=1/4 matches its printed final Planck-unit normalization. Neither is derived by the paper or by QED, and neither selects the normalization as fact. The observational contract registers 19 of 88 required fields, leaves 58 missing and four source-blocked, and opens zero observations. RTX 5090 campaigns rehearse Poisson-overdispersion, point-mass BTFR, and extended-source stress controls. The naive local-superposition completion is rejected as a hypothesis because coincident subdivision changes the far coefficient by sqrt(N) and an unequal pair violates matter-only action-reaction balance; the enclosed-mass completion stays blocked and geometry-blind. No source-supported covariant extended metric or lensing operator exists, so lensing and rotation tests remain unexecuted. NVML figures are device-wide and not lane-exclusive. First broader blocker: {_escape(transactional["first_blocker"])}. Synthetic power and algebraic agreement do not establish the transactional ontology, a paper-derived action, GR equivalence, dark-sector elimination, or observational validity.</p></section>
<section><h2>Grammar-v3 scalable candidates</h2><div class="metrics">{_outcomes(scalable_outcomes)}{_metric("Reviewed parameter cells", core["grammar_parameter_cells"]["reviewed_manifest"]["parameter_cell_count"])}{_metric("Unique typed actions", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["unique_candidate_count"])}{_metric("Equivalent aliases", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["equivalent_duplicate_count"])}{_metric("Sandboxed actions", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["generated_action_export"]["action_export_counts"]["sandbox_parsed_and_canonicalised"])}{_metric("Independent backend variations", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["generated_action_export"]["metric_variation_counts"]["executed_by_this_campaign"])}{_metric("Euler specializations", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["generated_action_export"]["candidate_metric_specialization"]["counts"]["candidate_euler_expressions_materialized"])}{_metric("Explained candidates", core["grammar_parameter_cells"]["explanation_dossiers"]["candidate_count"])}{_metric("Structural rows", core["grammar_parameter_cells"]["structural_metrics"]["candidate_count"])}{_metric("Simplicity Pareto front", core["grammar_parameter_cells"]["structural_metrics"]["simplicity_pareto_front"]["candidate_count"])}{_metric("Formal queues admitted", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["formal_preflight"]["promotion_admission"]["eligible_candidate_count"])}{_metric("Future reviewed cells", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["input_cell_count"])}{_metric("Future new candidates", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["disposition_counts"]["admitted_new_candidate"])}{_metric("Future preflight passes", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["preflight"]["decision_counts"]["pass"])}{_metric("Future preflight rejects", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["preflight"]["decision_counts"]["reject"])}{_metric("Future preflight blocks", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["preflight"]["decision_counts"]["blocked"])}{_metric("Future Aether blocked", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["decision_counts"]["blocked"])}{_metric("Negative finite seeds", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["exact_negative_static_source_monopole_count"])}{_metric("Forced characteristic crossings", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["negative_source_family_forced_characteristic_crossing_count"])}{_metric("Regular-ADM prerequisites", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["regular_ADM_implicit_lift_prerequisite_pass_count"])}{_metric("Legendre inverse margins", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["uniform_Aether_Legendre_block_inverse_pass_count"])}{_metric("Negative source margins", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["strict_negative_source_margin_pass_count"])}{_metric("Weighted contracts complete", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["typed_weighted_operator_contract_complete_count"])}{_metric("Future G3 uniform boxes", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["nonzero_componentwise_box_pass_count"])}{_metric("Future G3 AF profiles", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["AF_decaying_gradient_profile_pass_count"])}{_metric("Radial BVP no-go", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["positive_global_radial_Lichnerowicz_solution_nonexistence_count"])}{_metric("Nonradial York no-go", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["conformally_flat_bounded_mean_curvature_York_class_reject_count"])}{_metric("York cap extensions", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["strict_extension_beyond_kappa_6_over_5_pass_count"])}{_metric("Next caps inconclusive", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["next_grid_cap_inconclusive_count"])}{_metric("Nontrivial AF solutions", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["AF_Einstein_constraint_solution_pass_count"])}{_metric("G4 preflight blocked", core["grammar_parameter_cells"]["scalable_preflight_blocked_excluded_count"])}{_metric("Aether rejected", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["formal_preflight"]["promotion_admission"]["family_formal_execution"]["aether"]["decision_counts"]["reject"])}{_metric("Aether blocked", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["formal_preflight"]["promotion_admission"]["family_formal_execution"]["aether"]["decision_counts"]["blocked"])}{_metric("G2 formal passes", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["formal_preflight"]["promotion_admission"]["family_formal_execution"]["g2"]["decision_counts"]["pass"])}{_metric("G2 Solar fields remaining", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["formal_preflight"]["promotion_admission"]["family_formal_execution"]["g2"]["solar_readiness"]["registration_advance"]["after_missing_field_count"])}{_metric("G3 blocked", core["grammar_parameter_cells"]["reviewed_manifest"]["compilation"]["formal_preflight"]["promotion_admission"]["family_formal_execution"]["g3"]["decision_counts"]["blocked"])}</div><p>Across the 163 unique typed actions, the current exact formal tally is {_escape(scalable_outcomes["pass"])} pass, {_escape(scalable_outcomes["reject"])} reject, and {_escape(scalable_outcomes["block"])} blocked; the older six-seed execution is tracked separately. Every action has an exact human-readable master formula and a separate proof hierarchy. All 163 formulas parse and canonicalise in one isolated Cadabra batch, and all 163 now have exact candidate Euler expressions materialized by substitution into independently executed reviewed generic metric-variation theorems. This is not 163 independent backend variations and does not infer 163 formal passes. The 14 future Aether survivors remain blocked, not rejected: all have explicit finite-amplitude negative source seeds, but 11 are forced across an ADM characteristic shell before the proved negativity threshold. The other three have exact uniform Legendre-sector inverse bounds and strict negative source-energy margins; the weighted-IFT contract gate records eleven absent candidate-bound norm, gauge, inverse, remainder, and boundary fields for each, so no full inverse or sign theorem is inferred. For the three future G3 actions, exact Hamiltonian reduction and a nonradial Green comparison now extend the rejected conformally-flat York class to candidate caps 1.211, 1.211, and 1.210. The immediately next 0.001 grid point is inconclusive; that rejects only the registered ansatz class, not the actions. Non-conformally-flat geometry, larger mean curvature, different scalar data, global energy, and full-formal passes remain open. Observational data, halo/DM targets, and redshift-distance inputs remain sealed.</p></section>
<section><h2>Latest future formal boundaries</h2><div class="metrics">{_metric("Aether metric weighted contracts", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["declared_metric_weighted_contract_count"])}{_metric("Aether reference ellipticity", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["metric_reference_principal_ellipticity_pass_count"])}{_metric("Aether candidate blocks derived", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["candidate_Aether_constraint_principal_block_pass_count"])}{_metric("G3 analytic York thresholds", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["exact_algebraic_threshold_pass_count"])}{_metric("G3 closed endpoints excluded", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["closed_threshold_endpoint_reject_count"])}{_metric("Above-threshold controls inconclusive", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["above_threshold_control_inconclusive_count"])}</div><p>This section supersedes the earlier bounded-step summaries. For each of the three regular Aether candidates, concrete H2_-1/2 to L2_-5/2 metric spaces, conformal/York gauge, reference spectrum (2, 2, 8/3, 4), ellipticity margin 2, and a trivial decaying reference kernel are exact. The candidate Aether-variable block and metric-Aether off-diagonal principal symbol remain missing, so the full coupled Fredholm inverse, nonlinear remainder, and boundary-sign theorem remain blocked. For each future G3 candidate, the grid cap is replaced by an exact positive algebraic threshold kappa_star. The closed class |K| &lt;= kappa_star*v is excluded including its endpoint; rational controls above the threshold are inconclusive. These are ansatz-class results, not action rejection or AF-solution evidence.</p></section>
<section><h2>Current exact Aether and G3 boundary</h2><div class="metrics">{_metric("Aether canonical backgrounds", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["full_canonical_background_point_registered_count"])}{_metric("Local regular-stratum H-core", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["regular_stratum_flat_chart_H_core_contract_registered_count"])}{_metric("Global declared-profile H-core", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["declared_profile_global_flat_chart_H_core_registered_count"])}{_metric("Characteristic shell", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["characteristic_shell_condition"])}{_metric("Shell rank", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["characteristic_shell_rank"])}{_metric("Shell nullity", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["characteristic_shell_nullity"])}{_metric("Aether covariant H/D DAGs", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["aether"]["metric_covariantized_H_D_Frechet_DAG_registered_count"])}{_metric("G3 radial momentum audits", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["radial_momentum_leading_order_pass_count"])}{_metric("Real joint coefficients", core["grammar_parameter_cells"]["staged_epoch"]["reviewed_future_chunk"]["family_followup"]["g3"]["joint_real_asymptotic_coefficient_solution_count"])}</div><p>The sole uniformly elliptic Aether candidate has a candidate-bound flat-chart canonical seed and an exact local <code>H_core</code> on the regular stratum <code>F^2!=31</code>. Its declared amplitude-10 compact profile crosses <code>F^2=31</code>, where the Legendre Hessian has rank seven and nullity two. The seed momentum lies in the singular image, but velocities are nonunique, so no single smooth global flat-chart H-core exists on that profile. The off-flat covariant H/D Frechet DAG, weighted Fredholm inverse, nonlinear remainder, completed-boundary sign, and candidate rejection remain blocked. For the three future G3 actions, real flat spherical AF York data with standard r^-2 falloff imply <code>1+2k^2=0</code>; this excludes that matched data class, not the actions or theories.</p></section>
  {future_dossier_html}
<section><h2>Quartic nonlinear closure</h2><div class="metrics">{_metric("Candidates", core["quartic_nonlinear_closure"]["candidate_count"])}{_metric("Coordinate pairs", core["quartic_nonlinear_closure"]["coordinate_pair_partition"]["total_unordered_coordinate_pairs"])}{_metric("Two-jets closed", core["quartic_nonlinear_closure"]["quadratic_deltaK_two_jet"]["closed_candidate_count"])}{_metric("Diagonal third jets", core["quartic_nonlinear_closure"]["diagonal_third_jet"]["diagonal_triples_closed"])}{_metric("Reference mixed sector", core["quartic_nonlinear_closure"]["mixed_third_jet_chunk"]["full_mixed_sector_closed"])}{_metric("Evidence rank", str(core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["completion_rank"]) + "/" + str(core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["symmetric_cubic_dimension"]))}{_metric("Reduced obligations", core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["reranked_exact_obligations"])}{_metric("Reduced obligations evaluated", core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["obligations_evaluated"])}{_metric("Reduced obligations remaining", core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["obligations_remaining"])}{_metric("Reduced candidate checks", core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["candidate_evaluations"])}{_metric("Reduced candidate budget", core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["candidate_evaluation_budget"])}{_metric("Brute-force avoided", core["quartic_nonlinear_closure"]["mixed_third_jet_reduction"]["brute_force_unevaluated_triples"])}{_metric("Full tube identities", core["quartic_nonlinear_closure"]["closure_counts"]["full_tube_Sylvester_identities"])}{_metric("Global H7 closures", core["quartic_nonlinear_closure"]["closure_counts"]["global_H7_closures"])}</div><p>All coordinate pairs are classified and all 12 reference quadratic deltaK two-jets satisfy the Sylvester identity through derivative orders 0, 1, and 2. The exact third-order recurrence passes all 41 diagonal active-coordinate triples. Twenty-five exact mixed chunks supply 1,600 stable records. The rank-15 active-direction reduction proves a minimal 447-obligation complement; the restart-safe selective service has now closed all 447/447 obligations and all 5,364/5,364 candidate systems with zero obstruction. The resulting rank is 680/680, so the full 12,300-entry reference mixed third-jet sector is closed without evaluating the other 10,700 lexicographic triples. First missing premise: {_escape(core["quartic_nonlinear_closure"]["first_missing_premise"])}. This is a reference Taylor-sector theorem, not a tube-uniform nonlinear solution: CK1, CK3, TC2, B7, global H7, and lifespan remain fail-closed.</p></section>
<section><h2>Fourth-order range closure</h2><div class="metrics">{_metric("Exact selector", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["selector_obligations"])}{_metric("Obligations closed", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["obligations_closed"])}{_metric("Unevaluated after stop", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["obligations_remaining"])}{_metric("Canonical obstructions", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["candidate_obstructed"])}{_metric("Polarization directions", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["canonical_obstruction_certificate"]["homogeneous_freedom_reduction"]["polarization_directions_checked"])}{_metric("Lower-jet slots covered", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["canonical_obstruction_certificate"]["homogeneous_freedom_reduction"]["lower_jet_reference_kernel_slots_covered_by_identity"])}{_metric("Algebraic correction basis", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["canonical_obstruction_certificate"]["minimal_algebraic_TC2_escape"]["correction_basis_dimension"])}{_metric("Correction-map rank", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["canonical_obstruction_certificate"]["minimal_algebraic_TC2_escape"]["induced_cokernel_map_rank"])}{_metric("Tuned D4 solutions", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["canonical_obstruction_certificate"]["minimal_algebraic_TC2_escape"]["candidate_D4_solutions_after_tuning"])}{_metric("Covariant origin", core["quartic_nonlinear_closure"]["fourth_jet_range_obligations"]["canonical_obstruction_certificate"]["minimal_algebraic_TC2_escape"]["correction_ansatz"]["covariant_or_action_derived"])}</div><p>The service closed 244 obligations and found an exact rank-two obstruction at selector offset 244 for all 12 candidates. The homogeneous-freedom theorem uses <code>R0^T(HP-P^T H)R0=0</code> to prove that no homogeneous lower-jet completion can cancel it. A new one-dimensional rank-one state-space correction does span the obstruction line: the unique candidate-specific tuning <code>eta=-(34816/15) alpha^5</code> gives exact D4 solutions for all 12 algebraic specializations. This is an algebraic escape, not a physical correction: it has no covariant-action origin, no gauge/constraint compatibility proof, no registered corrected candidates, and no single universal eta. The remaining 2,816 obligations are unevaluated, and full D4, tube, CK1, CK3, TC2, B7, global H7, and lifespan remain false.</p></section>
<section><h2>Physical hardware sample</h2><div class="metrics">{cpu_metrics}{gpu_metrics}</div><p>CPU source: {_escape(cpu.get("source", "not sampled"))}. GPU source: {_escape(gpu.get("source", "not sampled"))}. These instantaneous host/device sensors are separate from durable scheduler occupancy.</p></section>
</div>
<section><h2>Scheduler lanes</h2><table><thead><tr><th>Lane</th><th>Running</th><th>Runnable now</th><th>Scheduled later</th><th>Earliest ready</th><th>Capacity</th><th>Scheduler occupancy</th></tr></thead><tbody>{lane_rows}</tbody></table><p>Runnable and scheduled counts honor each task's durable <code>not_before_utc</code>; scheduler occupancy remains distinct from CPU/GPU hardware utilization.</p></section>
<div class="grid">
<section><h2>Current missing-evaluator blockers</h2><ul>{blocker_rows}</ul></section>
<section><h2>LLM budget and proposal quarantine</h2><div class="metrics">{_metric("Campaign budget", "$" + str(core["llm"]["configured_budget_usd"]))}{_metric("Campaign spent", "$" + str(core["llm"]["spent_usd"]))}{_metric("Adapter cap", "$" + core["llm"]["proposal_adapter"]["maximum_total_usd"])}{_metric("Adapter calls", core["llm"]["proposal_adapter"]["network_calls_made"])}{_metric("Local epoch candidates", core["llm"]["reviewed_local_epoch"]["expected_bounded_status"]["candidate_count"])}{_metric("Local policy passes", core["llm"]["reviewed_local_epoch"]["expected_bounded_status"]["policy_pass_count"])}</div><p>Adapter: {_escape(core["llm"]["proposal_adapter"]["status"])}. Campaign bridge: {_escape(core["llm"]["campaign_bridge"]["status"])}. Typed-DSL admission: {_escape(core["llm"]["typed_dsl_admission"]["status"])}. Candidate-registry bridge: {_escape(core["llm"]["compiler_registry_bridge"]["status"])}. Local composed epoch: {_escape(core["llm"]["reviewed_local_epoch"]["status"])}. Restart-safe service: {_escape(core["llm"]["reviewed_local_service"]["status"])}. Paid calls and durable formula-body persistence are disabled by default; future outputs remain {_escape(core["llm"]["proposal_adapter"]["output_status"])}, and only exact hash-bound proposals can enter the separate reviewed compiler queue.</p></section>
<section><h2>Billion-formula screen</h2><div class="metrics">{_metric("Source formulas", core["billion_formula_streaming"]["source_formula_count"])}{_metric("Static survivors", core["billion_formula_streaming"]["sampled_static_stage"]["pass"])}{_metric("Lift rejected", core["billion_formula_streaming"]["promotion_stage"]["lift_reject"])}{_metric("Lift blocked", core["billion_formula_streaming"]["promotion_stage"]["lift_block"])}</div></section>
</div>
<section><h2>How to read a candidate theory</h2><p>Open a candidate name to see its defining action: this is the compact master formula. The field equations, constraints, observable predictions, and pass/block certificates are derived from that action and appear in the separate proof and test hierarchy. They are supporting equations, not extra fitted pieces silently added to the theory.</p><details class="formula-terms"><summary>Notation guide for the displayed actions</summary><p><code>S</code> is the action; <code>g_mu_nu</code> is the spacetime metric; <code>sqrt(-g) d^4x</code> is the invariant spacetime volume; <code>R</code> is the Ricci scalar; <code>phi</code> is a scalar field; <code>X_phi</code> is its kinetic scalar; and <code>Lambda_phi</code> is its normalization scale. The exact ordered covariant densities remain available under each formula, so this display glossary changes notation only and never adds a theory term.</p></details></section>
{leaderboard_html}
<footer>Core SHA-256: {_escape(snapshot["core_content_sha256"])} · sampled {_escape(volatile["sampled_at_utc"])} · {source_revision_count} immutable source revisions. Cross-pipeline totals are intentionally not summed because candidate sets and gate semantics overlap.</footer>
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
            board["top10"] + board.get("completed_incomparable_evidence", []) + displayed_unranked
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
            f"<section><h2>Category leaderboard: {_escape(category)}</h2>"
            f"<p>{_escape(board['ranking_scope'])}. Ranked: {_escape(board['ranked_count'])}; "
            f"completed in separate evidence classes: {_escape(board.get('completed_separate_class_count', 0))}; "
            f"blocked/untested: {_escape(board['unranked_count'])} "
            f"(showing {_escape(len(displayed_unranked))}; full rows remain in the JSON snapshot). Availability: "
            f"{_escape(board['availability'])}. {_escape(board['absence_reason'] or '')}</p>"
            '<div style="overflow:auto"><table><thead><tr><th>Rank</th><th>Candidate</th>'
            "<th>Theory formula</th><th>Role</th><th>Exact metrics</th><th>Status</th><th>Data class</th>"
            "<th>Gate completeness</th><th>Blocker</th><th>Artifact content SHA</th>"
            f"<th>Uncertainty</th></tr></thead><tbody>{rows}</tbody></table></div></section>"
        )
    return "".join(sections)
