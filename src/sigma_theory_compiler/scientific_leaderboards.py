"""Category-specific, hash-bound scientific leaderboards without a truth score."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CATEGORIES = (
    "formal_adm_dirac",
    "hyperbolicity_common_cone",
    "nonlinear_energy",
    "solar_known_answer",
    "galaxy_direct_observable",
    "lensing_cluster",
    "simplicity_complexity",
    "novelty_non_equivalence",
    "computational_robustness",
)
ELIGIBILITY = {
    "observational_data_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_distance_inputs": False,
    "paid_llm_calls": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def load_leaderboard_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    claimed = value.pop("content_sha256", None)
    if claimed != _sha(value):
        raise ValueError("scientific leaderboard config hash mismatch")
    value["content_sha256"] = claimed
    return value


def _sources(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    loaded = {}
    for label, binding in config["sources"].items():
        path = (root / binding["path"]).resolve()
        if root.resolve() not in path.parents:
            raise ValueError("leaderboard source escapes project root")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != binding["file_sha256"]:
            raise ValueError(f"leaderboard source file hash mismatch: {label}")
        value = json.loads(raw)
        body = {key: item for key, item in value.items() if key != "content_sha256"}
        if value.get("content_sha256") != binding["content_sha256"] or _sha(body) != binding[
            "content_sha256"
        ]:
            raise ValueError(f"leaderboard source content hash mismatch: {label}")
        loaded[label] = value
    return loaded


def _entry(
    candidate_id: str,
    role: str,
    metrics: dict[str, Any],
    evidence_status: str,
    data_class: str,
    gate_completeness: str,
    blocker: str | None,
    source_label: str,
    source_binding: Mapping[str, Any],
    lineage_sha256: str | None,
    uncertainty: str,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "role": role,
        "rank": None,
        "metrics": metrics,
        "evidence_status": evidence_status,
        "data_class": data_class,
        "gate_completeness": gate_completeness,
        "blocker": blocker,
        "lineage": {
            "source_label": source_label,
            "artifact_link": source_binding["path"],
            "artifact_file_sha256": source_binding["file_sha256"],
            "artifact_content_sha256": source_binding["content_sha256"],
            "candidate_lineage_sha256": lineage_sha256,
        },
        "uncertainty": uncertainty,
        "promotion_eligible": False,
    }


def _sort_key(category: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    if category == "formal_adm_dirac":
        return (0 if row["evidence_status"] == "pass" else 1, -metrics["passed_gate_count"], row["candidate_id"])
    if category == "hyperbolicity_common_cone":
        return (-metrics["characteristic_discriminant_lower"], row["candidate_id"])
    if category == "solar_known_answer":
        return (-metrics["passed_control_count"], row["candidate_id"])
    if category == "simplicity_complexity":
        return (metrics["operator_count"], metrics["field_count"], metrics["parameter_count"], row["candidate_id"])
    if category == "novelty_non_equivalence":
        return (-int(metrics["unique_within_manifest"]), row["candidate_id"])
    if category == "computational_robustness":
        return (metrics["attempt"], row["candidate_id"])
    return (row["candidate_id"],)


def _admit_and_rank(category: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    ranked = [
        row for row in rows
        if row["evidence_status"] in {"pass", "reject", "measured"}
        and row["gate_completeness"] == "complete_for_category"
    ]
    unranked = [row for row in rows if row not in ranked]
    ranked.sort(key=lambda row: _sort_key(category, row))
    for rank, row in enumerate(ranked, 1):
        row["rank"] = rank
    availability = (
        "completed_comparable_evidence"
        if ranked
        else "blocked_or_untested_only"
        if rows
        else "no_hash_bound_evidence_packets"
    )
    return {
        "category": category,
        "ranking_scope": "completed comparable evidence within this category only",
        "availability": availability,
        "absence_reason": (
            "no hash-bound lensing or cluster evaluator artifact is registered"
            if category == "lensing_cluster" and not rows
            else None
        ),
        "ranked_count": len(ranked),
        "unranked_count": len(unranked),
        "top10": ranked[:10],
        "full_ranked": ranked,
        "unranked_blocked_or_untested": sorted(unranked, key=lambda row: row["candidate_id"]),
    }


def _build_rows(sources: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {category: [] for category in CATEGORIES}
    formal = sources["formal_adm_dirac"]
    for control, role in ((formal["known_answer_control"], "known_answer_control"), (formal["generated_candidate_negative_control"], "generated_candidate")):
        gates = control["gate_statuses"]
        rows["formal_adm_dirac"].append(_entry(
            control["candidate_id"], role,
            {"passed_gate_count": sum(value == "pass" for value in gates.values()), "rejected_gate_count": sum(value == "reject" for value in gates.values()), "unresolved_gate_count": sum(value == "unresolved" for value in gates.values())},
            control["decision"], "exact_formal_artifact", "complete_for_category", None if control["decision"] == "pass" else ",".join(control.get("rejected_gates", [])),
            "formal_adm_dirac", bindings["formal_adm_dirac"], control.get("bundle_binding_sha256"),
            "Known-answer controls calibrate the evaluator only; formal health is not total-energy or observational evidence.",
        ))
    for candidate in formal["real_survivor_decisions"]:
        rows["formal_adm_dirac"].append(_entry(candidate["candidate_id"], "generated_candidate", {}, "blocked", "exact_formal_artifact", "incomplete", candidate["blocker"], "formal_adm_dirac", bindings["formal_adm_dirac"], candidate.get("input_lineage_sha256"), "No exact candidate-to-action map; this is missing evidence, not measured poor performance."))

    followup = sources["grammar_v3_formal_followup"]
    g4_records = [
        record
        for record in followup["work_records"]
        if record["task_type"]
        in {"g4_global_lapse_invertibility", "g4_global_positive_energy"}
    ]
    g4_ids = {record["candidate_id"] for record in g4_records}
    if (
        len(g4_records) != 2
        or len(g4_ids) != 1
        or followup["followup_decision_counts"] != {"blocked": 8, "pass": 2}
        or followup["candidate_scientific_decisions_changed"] != 1
        or any(record["state"] != "succeeded" for record in g4_records)
    ):
        raise ValueError("G4 formal follow-up evidence is incomplete or inconsistent")
    g4_candidate_id = next(iter(g4_ids))
    rows["formal_adm_dirac"].append(_entry(
        g4_candidate_id,
        "generated_candidate",
        {
            "passed_gate_count": min(
                record["pareto_axis_values"]["formal_pass_count"]
                for record in g4_records
            ),
            "rejected_gate_count": 0,
            "unresolved_gate_count": 0,
            "reviewed_followup_packet_count": len(g4_records),
        },
        "pass",
        "exact_formal_artifact",
        "complete_for_category",
        None,
        "grammar_v3_formal_followup",
        bindings["grammar_v3_formal_followup"],
        _sha([record["followup_lineage_sha256"] for record in g4_records]),
        "Full formal pass through the reviewed Einstein-frame generalized-harmonic audit; this is not Solar or observational evidence.",
    ))

    cone_record = sources["g3_common_cone"]["candidate_records"][0]
    proof = cone_record["principal_common_cone_certificate"]["uniform_proof"]
    rows["hyperbolicity_common_cone"].append(_entry(
        cone_record["seed_id"], "generated_candidate",
        {"characteristic_discriminant_lower": proof["characteristic_discriminant_lower"], "spatial_eigenvalue_lower": proof["spatial_block_eigenvalue_lower"], "common_time_P00_upper": proof["common_time_covector_upper_P00"]},
        "pass", "uniform_local_jet_box_common_cone", "complete_for_category", None,
        "g3_common_cone", bindings["g3_common_cone"], cone_record["provenance"]["binding_sha256"],
        "Uniform on the declared local box and every spatial direction; not an evolution-invariant or global-energy theorem.",
    ))

    for label in ("aether_energy", "g2_energy"):
        for record in sources[label]["candidate_records"]:
            rows["nonlinear_energy"].append(_entry(
                record["seed_id"], "generated_candidate",
                {"negative_energy_counterexample_found": record.get("negative_energy_mode_found", record.get("negative_total_energy_counterexample_found", False))},
                "blocked", "global_nonlinear_energy", "incomplete", record["first_missing_premise"], label, bindings[label], record["provenance"]["binding_sha256"],
                "Restricted/local positive results cannot rank as a completed global nonlinear-energy result.",
            ))

    solar = sources["solar_known_answer"]
    gr = solar["known_answer_control"]
    rows["solar_known_answer"].append(_entry(
        gr["candidate_id"], "known_answer_control", {"passed_control_count": sum(value == "pass" for value in gr["golden_statuses"].values()), "total_control_count": len(gr["golden_statuses"])},
        "pass", "sealed_solar_known_answer", "complete_for_category", None, "solar_known_answer", bindings["solar_known_answer"], gr["bundle_binding_sha256"],
        "GR is a solver calibration control, not a generated-candidate promotion entry or observational discovery result.",
    ))
    blocked = solar["unmapped_candidate_control"]
    rows["solar_known_answer"].append(_entry(blocked["candidate_id"], "known_answer_control", {}, "blocked", "sealed_solar_known_answer", "incomplete", blocked["blocker"], "solar_known_answer", bindings["solar_known_answer"], blocked["input_lineage_sha256"], "Missing action-bound Solar bundle; untested rather than poor."))

    g4_solar = sources["g4_solar_promotion"]
    if (
        g4_solar["formal_pass_verified"] is not True
        or g4_solar["prediction_bundle_descriptor_registered"] is not False
        or g4_solar["reviewed_solar_evaluator_invoked"] is not False
        or g4_solar["solar_evaluator_opened"] is not False
        or g4_solar["work_state_counts"]
        != {"deferred_missing_prediction_bundle_descriptor": 1}
    ):
        raise ValueError("G4 Solar boundary status is inconsistent")
    g4_solar_row = g4_solar["category_leaderboard"]["blocked_or_untested"][0]
    rows["solar_known_answer"].append(_entry(
        g4_solar_row["candidate_id"],
        "generated_candidate",
        {
            "analytic_prediction_bundle_count": g4_solar[
                "reviewed_prediction_audit_binding"
            ]["analytic_bundle_count"],
            "real_solar_bundle_count": g4_solar[
                "reviewed_prediction_audit_binding"
            ]["real_bundle_count"],
            "analytic_newtonian_ppn_status": "pass_on_declared_scalar_free_background",
        },
        "blocked",
        "sealed_candidate_specific_solar_prediction",
        "incomplete",
        g4_solar_row["blocker"],
        "g4_solar_promotion",
        bindings["g4_solar_promotion"],
        g4_solar_row["lineage_sha256"],
        "Analytic GR-like Newtonian/PPN predictions exist, but no real-source branch-uniqueness contract or registered candidate-use Solar bundle exists; untested, not poor measured performance.",
    ))

    galaxy = sources["galaxy_direct_observable"]
    rows["galaxy_direct_observable"].append(_entry(
        "PRODUCTION-CANDIDATE-SET-70", "generated_candidate_set", {"candidate_count": galaxy["production_candidate_count"], "registered_prediction_bundle_count": galaxy["registered_prediction_bundle_count"]},
        "blocked", "sealed_direct_observable_scaffold", "incomplete", next(iter(galaxy["blocker_counts"])), "galaxy_direct_observable", bindings["galaxy_direct_observable"], galaxy["evaluator_binding_sha256"], "No observational source or candidate prediction bundle was opened; no galaxy performance was measured."))

    compilation = sources["grammar_compilation"]
    action_hashes = [record["typed_action_ir"]["content_sha256"] for record in compilation["candidate_records"]]
    for record in compilation["candidate_records"]:
        action = record["typed_action_ir"]
        lineage = record["provenance"]["binding_sha256"]
        rows["simplicity_complexity"].append(_entry(
            record["seed_id"], "generated_candidate", {"operator_count": len(action["operators"]), "field_count": len(action["fields"]), "parameter_count": len(action.get("parameters", {})), "maximum_derivative_order": max(item["maximum_field_derivative_order"] for item in action["operators"])},
            "measured", "typed_action_structural_complexity", "complete_for_category", None, "grammar_compilation", bindings["grammar_compilation"], lineage, "Structural counts only; lower complexity does not imply greater physical truth."))
        rows["novelty_non_equivalence"].append(_entry(
            record["seed_id"], "generated_candidate", {"unique_within_manifest": action_hashes.count(action["content_sha256"]) == 1, "action_content_sha256": action["content_sha256"]},
            "measured", "internal_exact_action_non_equivalence", "complete_for_category", None, "grammar_compilation", bindings["grammar_compilation"], lineage, "Only exact non-equivalence within this six-seed manifest is measured; literature novelty remains untested."))

    execution = sources["computational_execution"]
    for record in execution["work_records"]:
        rows["computational_robustness"].append(_entry(
            record["seed_id"], "generated_candidate", {"attempt": record["attempt"], "state": record["state"], "recovered_lease": False},
            "measured" if record["state"] == "succeeded" else "blocked", "deterministic_bounded_execution", "complete_for_category" if record["state"] == "succeeded" else "incomplete", None if record["state"] == "succeeded" else "execution_incomplete", "computational_execution", bindings["computational_execution"], record["output_lineage_sha256"], "A successful deterministic execution is software robustness evidence, not scientific validity."))
    return rows


def _validate_no_collapse(board: Mapping[str, Any]) -> None:
    text = json.dumps(board, sort_keys=True).lower()
    for forbidden in ("truth_score", "overall_score", "composite_score", "probability_of_truth"):
        if forbidden in text:
            raise ValueError("scalar truth-score collapse is forbidden")
    if board["data_eligibility"] != ELIGIBILITY:
        raise ValueError("leaderboards opened a forbidden data class")
    for value in board["categories"].values():
        for row in value["full_ranked"] + value["unranked_blocked_or_untested"]:
            if row["role"] == "known_answer_control" and row["promotion_eligible"] is not False:
                raise ValueError("known-answer control leaked into candidate promotion")
            if row["rank"] is not None and row["gate_completeness"] != "complete_for_category":
                raise ValueError("missing evidence entered a ranked leaderboard")
            if row["rank"] is not None and row["data_class"] != value["comparison_data_class"]:
                raise ValueError("cross-category metric or data-class mixing detected")
            scientific_target = json.dumps(
                {
                    "metrics": row["metrics"],
                    "data_class": row["data_class"],
                    "blocker": row["blocker"],
                },
                sort_keys=True,
            ).lower()
            if any(
                forbidden in scientific_target
                for forbidden in ("dark_matter", "halo_target", "redshift_distance")
            ):
                raise ValueError("forbidden inferred target entered a leaderboard")


def validate_scientific_leaderboards(board: Mapping[str, Any]) -> None:
    """Public fail-closed validation used by import/export negative controls."""
    _validate_no_collapse(board)


def build_scientific_leaderboards(
    project_root: Path,
    config: Mapping[str, Any],
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if config.get("data_eligibility") != ELIGIBILITY:
        raise ValueError("leaderboard config eligibility is not fail-closed")
    sources = _sources(project_root.resolve(), config)
    rows = _build_rows(sources, config["sources"])
    categories = {}
    for category in CATEGORIES:
        category_board = _admit_and_rank(category, rows[category])
        ranked = category_board["full_ranked"]
        category_board["comparison_data_class"] = ranked[0]["data_class"] if ranked else None
        if ranked and any(row["data_class"] != category_board["comparison_data_class"] for row in ranked):
            raise ValueError("category mixes incomparable data classes")
        category_board["category_root_sha256"] = _sha(category_board)
        categories[category] = category_board
    current_root = _sha(categories)
    previous_ranks = {}
    history = []
    if previous:
        history = list(previous.get("history", []))
        for category, board in previous.get("categories", {}).items():
            previous_ranks[category] = {row["candidate_id"]: row["rank"] for row in board["full_ranked"]}
    if not history or history[-1]["leaderboard_root_sha256"] != current_root:
        history.append({"leaderboard_root_sha256": current_root, "category_roots": {key: value["category_root_sha256"] for key, value in categories.items()}})
    history = history[-int(config["maximum_history_snapshots"]):]
    deltas = {
        category: [
            {"candidate_id": row["candidate_id"], "previous_rank": previous_ranks.get(category, {}).get(row["candidate_id"]), "current_rank": row["rank"], "rank_delta": None if previous_ranks.get(category, {}).get(row["candidate_id"]) is None else previous_ranks[category][row["candidate_id"]] - row["rank"]}
            for row in board["full_ranked"]
        ]
        for category, board in categories.items()
    }
    body = {
        "schema_version": "sigma-scientific-leaderboards-1.0",
        "ranking_contract": "no global score; category-local completed comparable evidence only; candidate_id is the final deterministic tie-break",
        "categories": categories,
        "leaderboard_root_sha256": current_root,
        "history": history,
        "deltas_from_previous": deltas,
        "data_eligibility": dict(ELIGIBILITY),
    }
    _validate_no_collapse(body)
    return {**body, "content_sha256": _sha(body)}
