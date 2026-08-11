"""Category-specific, hash-bound scientific leaderboards without a truth score."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .scalable_candidate_structural_metrics_export import (
    validate_scalable_candidate_structural_metrics_export,
)
from .scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
    validate_scalable_formal_candidate_evidence_export,
)

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


def _generated_action_formula(action: Mapping[str, Any]) -> dict[str, Any]:
    family = action["family_id"]
    parameters = dict(action.get("parameters", {}))
    if family == "AETHER_K1234_PARAMETER_CELL":
        title = "Einstein–Aether gravity"
        defining_action = (
            "S = ∫ d⁴x √(-g) [(M_Pl²/2)R − (M_Pl²/2)"
            "(c₁K₁+c₂K₂+c₃K₃−c₄K₄) + λᵤ(u·u+1)] + S_m[g,ψ]"
        )
        explanation = (
            "Einstein gravity plus a unit timelike vector field with four derivative "
            "couplings; matter couples universally to the metric."
        )
    elif family == "KESSENCE_G2_CONVEX":
        title = "Einstein gravity + convex k-essence"
        defining_action = (
            "S = ∫ d⁴x √(-g) [(M_Pl²/2)R + Λφ⁴ G₂(Xφ)] + S_m[g,ψ], "
            f"G₂ = {parameters['G2']}"
        )
        explanation = (
            "Einstein gravity plus a scalar whose kinetic energy contains a positive "
            "quadratic correction."
        )
    elif family == "CUBIC_HORNDESKI_G3_WEAK_CELL":
        title = "Einstein gravity + cubic Horndeski scalar"
        defining_action = (
            "S = ∫ d⁴x √(-g) [(M_Pl²/2)R + Λφ⁴Xφ − "
            "(Λφ/100)Xφ □φ] + S_m[g,ψ]"
        )
        explanation = (
            "Einstein gravity plus a scalar kinetic term and a weak cubic derivative "
            "interaction."
        )
    elif family == "CONFORMAL_G4_PHI_SCALAR_TENSOR":
        title = "Conformal scalar–tensor gravity"
        defining_action = (
            "S = ∫ d⁴x √(-g) [Λφ⁴Xφ + Λφ²(1/2 + φ²/100)R] "
            "+ S_m[g,ψ]"
        )
        explanation = (
            "A scalar field changes the effective curvature coupling; at φ=0 the "
            "theory has an exact GR branch."
        )
    else:
        title = family
        defining_action = "S = ∫ d⁴x Σ(operator densities)"
        explanation = "Typed covariant action; expand the bound operator terms for details."
    return {
        "formula_type": "defining_action",
        "title": title,
        "defining_action": defining_action,
        "plain_language": explanation,
        "fields": list(action["fields"]),
        "parameters": parameters,
        "operator_terms": [operator["density"] for operator in action["operators"]],
        "action_content_sha256": action["content_sha256"],
        "scope_note": (
            "This compact action defines the candidate. Field equations and proof/test "
            "certificates are derived evidence, not extra fitted formula terms."
        ),
    }


def _theory_formula(
    candidate_id: str, action_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    action = action_by_id.get(candidate_id)
    if action is not None:
        return _generated_action_formula(action)
    if candidate_id == "KNOWN-ANSWER-EINSTEIN-HILBERT":
        return {
            "formula_type": "defining_action",
            "title": "General relativity (Einstein–Hilbert)",
            "defining_action": "S = ∫ d⁴x √(-g) [(M_Pl²/2)R + L_m]",
            "plain_language": "Spacetime curvature is sourced by universally coupled matter.",
            "fields": ["g_mu_nu", "matter"],
            "parameters": {},
            "operator_terms": ["sqrt(-g)*(M_Pl^2/2)*R", "sqrt(-g)*L_m"],
            "action_content_sha256": None,
            "scope_note": "Calibration control; not a generated discovery candidate.",
        }
    if candidate_id == "KNOWN-ANSWER-EINSTEIN-AETHER":
        return {
            "formula_type": "theory_family_action",
            "title": "Einstein–Aether known-answer control",
            "defining_action": (
                "S = ∫ d⁴x √(-g) [(M_Pl²/2)(R − c₁K₁ − c₂K₂ − c₃K₃ "
                "+ c₄K₄) + λᵤ(u·u+1)] + S_m"
            ),
            "plain_language": "Einstein gravity plus a constrained unit timelike vector.",
            "fields": ["g_mu_nu", "u_mu", "lambda_u", "matter"],
            "parameters": {"c1..c4": "known-answer bundle values"},
            "operator_terms": ["R", "K1", "K2", "K3", "K4", "u_mu*u^mu+1"],
            "action_content_sha256": None,
            "scope_note": "Calibration control; not a generated discovery candidate.",
        }
    return {
        "formula_type": "aggregate_or_unexpanded_artifact",
        "title": "No single display formula",
        "defining_action": "See the hash-bound candidate or candidate-set artifact.",
        "plain_language": (
            "This row represents an aggregate, an unmapped candidate, or a negative "
            "control rather than one typed action."
        ),
        "fields": [],
        "parameters": {},
        "operator_terms": [],
        "action_content_sha256": None,
        "scope_note": "No formula is inferred when exact typed-action evidence is absent.",
    }


def _theory_dossier_registry(
    artifact: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        artifact.get("dossier_count") != 7
        or artifact.get("known_answer_control_count") != 1
        or artifact.get("generated_candidate_count") != 6
        or artifact.get("hierarchy_node_status_counts")
        != {"blocked": 17, "calibration_only": 2, "proven": 34}
        or artifact.get("observational_authorization") is not False
        or artifact.get("observational_data_opened") is not False
    ):
        raise ValueError("candidate theory dossier campaign is inconsistent")
    registry: dict[str, dict[str, Any]] = {}
    for dossier in artifact["dossiers"]:
        dossier_id = dossier["dossier_id"]
        nodes = [
            {
                "content_sha256": node["content_sha256"],
                "evidence_count": len(node["evidence"]),
                "node_id": node["node_id"],
                "scope": node["scope"],
                "status": node["status"],
            }
            for node in dossier["hierarchy_nodes"]
        ]
        counts = dict(sorted(Counter(node["status"] for node in nodes).items()))
        if dossier_id in registry or _sha(
            {key: value for key, value in dossier.items() if key != "content_sha256"}
        ) != dossier["content_sha256"]:
            raise ValueError("candidate theory dossier identity or content hash mismatch")
        registry[dossier_id] = {
            "artifact_content_sha256": binding["content_sha256"],
            "artifact_file_sha256": binding["file_sha256"],
            "artifact_link": binding["path"],
            "content_sha256": dossier["content_sha256"],
            "dossier_id": dossier_id,
            "hierarchy_nodes": nodes,
            "hierarchy_status_counts": counts,
            "overall_status": dossier["overall_status"],
            "status_label": "Overall",
            "role": dossier["role"],
        }
    return registry


def _scalable_theory_dossier_registry(
    artifact: Mapping[str, Any], binding: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    if (
        artifact.get("candidate_count") != 163
        or artifact.get("alias_count") != 93
        or artifact.get("formal_decision_counts")
            != {"blocked": 158, "pass": 3, "reject": 2}
        or artifact.get("hierarchy_node_status_counts")
        != {
                "blocked": 321,
                "calibration_only": 163,
                "proven": 166,
            "rejected": 2,
        }
        or artifact.get("observational_authorization") is not False
        or artifact.get("observational_data_opened") is not False
    ):
        raise ValueError("scalable candidate explanation bridge is inconsistent")
    registry: dict[str, dict[str, Any]] = {}
    for dossier in artifact.get("dossiers", []):
        dossier_id = dossier["candidate_id"]
        dossier_body = {
            key: value for key, value in dossier.items() if key != "content_sha256"
        }
        if dossier_id in registry or _sha(dossier_body) != dossier["content_sha256"]:
            raise ValueError("scalable candidate dossier identity or content hash mismatch")
        nodes = []
        for node in dossier["hierarchy_nodes"]:
            node_body = {
                key: value for key, value in node.items() if key != "content_sha256"
            }
            if _sha(node_body) != node["content_sha256"]:
                raise ValueError("scalable candidate dossier node hash mismatch")
            nodes.append(
                {
                    "content_sha256": node["content_sha256"],
                    "evidence_count": len(node["evidence"]),
                    "node_id": node["node_id"],
                    "scope": node["scope"],
                    "status": node["status"],
                }
            )
        counts = dict(sorted(Counter(node["status"] for node in nodes).items()))
        registry[dossier_id] = {
            "artifact_content_sha256": binding["content_sha256"],
            "artifact_file_sha256": binding["file_sha256"],
            "artifact_link": binding["path"],
            "content_sha256": dossier["content_sha256"],
            "dossier_id": dossier_id,
            "hierarchy_nodes": nodes,
            "hierarchy_status_counts": counts,
            "overall_status": dossier["formal_decision"],
            "status_label": "Formal decision",
            "role": dossier["role"],
        }
    expected_root = _sha(
        [[item["candidate_id"], item["content_sha256"]] for item in artifact["dossiers"]]
    )
    if (
        len(registry) != 163
        or artifact.get("provenance", {}).get("dossier_registry_root_sha256")
        != expected_root
    ):
        raise ValueError("scalable candidate dossier registry is incomplete")
    return registry


def _comparison_key(category: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = row["metrics"]
    if category == "formal_adm_dirac":
        # A gate count is meaningful only inside one evaluator schema.  The
        # formal-action comparison class intentionally combines an original
        # reviewed G4 action and an exactly density-equivalent restricted-domain
        # action, whose artifacts enumerate different ledgers.  Their completed
        # formal decisions are therefore tied instead of ordered by raw gate
        # count.
        return (0 if row["evidence_status"] == "pass" else 1,)
    if category == "hyperbolicity_common_cone":
        return (-metrics["characteristic_discriminant_lower"],)
    if category == "solar_known_answer":
        return (-metrics["passed_control_count"],)
    if category == "simplicity_complexity":
        return (
            metrics["operator_count"],
            metrics["field_count"],
            metrics["parameter_count"],
            metrics.get("formula_payload_character_count", 0),
        )
    if category == "novelty_non_equivalence":
        return (
            -int(
                metrics.get(
                    "parameter_cell_class_size",
                    int(metrics.get("unique_within_manifest", False)),
                )
            ),
        )
    if category == "computational_robustness":
        return (metrics["attempt"],)
    return ()


def _sort_key(category: str, row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (*_comparison_key(category, row), row["candidate_id"])


def _admit_and_rank(category: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        row for row in rows
        if row["evidence_status"] in {"pass", "reject", "measured"}
        and row["gate_completeness"] == "complete_for_category"
    ]
    unranked = [row for row in rows if row not in completed]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in completed:
        grouped.setdefault(row["data_class"], []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda row: _sort_key(category, row))
        previous_key: tuple[Any, ...] | None = None
        comparison_rank = 0
        for ordinal, row in enumerate(group_rows, 1):
            current_key = _comparison_key(category, row)
            if current_key != previous_key:
                comparison_rank = ordinal
                previous_key = current_key
            row["comparison_group_rank"] = comparison_rank
    preferred_primary_class = {
        "formal_adm_dirac": "full_formal_action_evidence",
        "simplicity_complexity": "typed_action_formula_structure_v1",
        "novelty_non_equivalence": "exact_action_hash_and_parameter_cell_aliases_v1",
    }.get(category)
    if not grouped:
        primary_class = None
    elif preferred_primary_class in grouped:
        primary_class = preferred_primary_class
    elif len(grouped) == 1:
        primary_class = next(iter(grouped))
    else:
        raise ValueError(f"{category} contains multiple incomparable completed evidence classes")
    ranked = grouped.get(primary_class, []) if primary_class is not None else []
    for row in ranked:
        row["rank"] = row["comparison_group_rank"]
    completed_separate = [
        row
        for data_class, group_rows in sorted(grouped.items())
        if data_class != primary_class
        for row in group_rows
    ]
    comparison_groups = []
    for data_class, group_rows in sorted(grouped.items()):
        comparison_groups.append(
            {
                "data_class": data_class,
                "primary": data_class == primary_class,
                "completed_count": len(group_rows),
                "decision_counts": dict(
                    sorted(Counter(row["evidence_status"] for row in group_rows).items())
                ),
                "top10_candidate_ids": [row["candidate_id"] for row in group_rows[:10]],
                "group_root_sha256": _sha(group_rows),
            }
        )
    availability = (
        "completed_comparable_evidence"
        if completed
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
        "completed_separate_class_count": len(completed_separate),
        "unranked_count": len(unranked),
        "comparison_groups": comparison_groups,
        "top10": ranked[:10],
        "full_ranked": ranked,
        "completed_incomparable_evidence": completed_separate,
        "unranked_blocked_or_untested": sorted(unranked, key=lambda row: row["candidate_id"]),
    }


def _build_rows(sources: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows = {category: [] for category in CATEGORIES}
    compilation = sources["grammar_compilation"]
    action_by_id = {
        record["seed_id"]: record["typed_action_ir"]
        for record in compilation["candidate_records"]
    }
    scalable_export = sources["scalable_formal_candidates"]
    validate_scalable_formal_candidate_evidence_export(scalable_export)
    scalable_records = iter_scalable_formal_candidate_evidence_records(
        scalable_export
    )
    for record in scalable_records:
        formula = record["theory_formula_inputs"]
        action_by_id[record["candidate_id"]] = {
            "family_id": record["family_id"],
            "fields": formula["fields"],
            "parameters": formula["parameters"],
            "operators": formula["ordered_operator_densities"],
            "content_sha256": formula["action_content_sha256"],
        }
    structural_export = sources["scalable_structural_metrics"]
    validate_scalable_candidate_structural_metrics_export(structural_export)
    structural_records = structural_export["candidate_records"]
    if {record["candidate_id"] for record in structural_records} != {
        record["candidate_id"] for record in scalable_records
    }:
        raise ValueError("scalable structural metrics candidate registry changed")
    scalable_dossier_by_id = {
        dossier["candidate_id"]: dossier
        for dossier in sources["scalable_explanation_dossiers"]["dossiers"]
    }
    if set(scalable_dossier_by_id) != {
        record["candidate_id"] for record in scalable_records
    }:
        raise ValueError("scalable explanation dossier candidate registry changed")
    scalable_formula_by_id: dict[str, dict[str, Any]] = {}
    for candidate_id, dossier in scalable_dossier_by_id.items():
        action = action_by_id[candidate_id]
        exact = dossier["action"]
        densities = [
            operator["density"] for operator in exact["ordered_operator_densities"]
        ]
        if (
            exact["action_sha256"] != action["content_sha256"]
            or exact["fields"] != action["fields"]
            or exact["parameters"] != action["parameters"]
            or densities != [operator["density"] for operator in action["operators"]]
        ):
            raise ValueError("scalable exact action display disagrees with typed action")
        formula = _generated_action_formula(action)
        formula.update(
            {
                "defining_action": exact["human_readable_action"]["display_text"],
                "fields": list(exact["fields"]),
                "parameters": dict(exact["parameters"]),
                "operator_terms": densities,
                "action_content_sha256": exact["action_sha256"],
                "scope_note": exact["human_readable_action"]["scope"],
            }
        )
        scalable_formula_by_id[candidate_id] = formula
    formal = sources["formal_adm_dirac"]
    for control, role in ((formal["known_answer_control"], "known_answer_control"), (formal["generated_candidate_negative_control"], "generated_candidate")):
        gates = control["gate_statuses"]
        comparison_class = (
            "known_answer_formal_calibration"
            if role == "known_answer_control"
            else "generated_formal_negative_control"
        )
        rows["formal_adm_dirac"].append(_entry(
            control["candidate_id"], role,
            {"passed_gate_count": sum(value == "pass" for value in gates.values()), "rejected_gate_count": sum(value == "reject" for value in gates.values()), "unresolved_gate_count": sum(value == "unresolved" for value in gates.values())},
            control["decision"], comparison_class, "complete_for_category", None if control["decision"] == "pass" else ",".join(control.get("rejected_gates", [])),
            "formal_adm_dirac", bindings["formal_adm_dirac"], control.get("bundle_binding_sha256"),
            "Known-answer controls calibrate the evaluator only; formal health is not total-energy or observational evidence.",
        ))
    for candidate in formal["real_survivor_decisions"]:
        rows["formal_adm_dirac"].append(_entry(candidate["candidate_id"], "generated_candidate", {}, "blocked", "exact_formal_artifact", "incomplete", candidate["blocker"], "formal_adm_dirac", bindings["formal_adm_dirac"], candidate.get("input_lineage_sha256"), "No exact candidate-to-action map; this is missing evidence, not measured poor performance."))

    for record in scalable_records:
        decision = record["final_decision"]
        contract = record["leaderboard_contract"]
        metrics = {
            "alias_count": record["alias_count"],
            "preflight_decision": record["preflight_decision"],
            "passed_gate_count": 1 if decision == "pass" else 0,
            "rejected_gate_count": 1 if decision == "reject" else 0,
            "unresolved_gate_count": 1 if decision == "blocked" else 0,
        }
        if record["direct_metrics"]:
            metrics["decisive_metrics"] = record["direct_metrics"]
            metrics["metric_source_sha256"] = record["metric_source_sha256"]
        data_class = (
            record["comparison_data_class"]
            if decision in {"pass", "reject"}
            else "scalable_formal_candidate_evidence"
        )
        pass_scope = (
            "Conditional positive ADM four-momentum is proven on the registered complete boundaryless asymptotically-Euclidean nonmaximal Einstein-G2 constraint domain; existence, evolution, stability, Solar, and observational evidence remain separate."
            if record["family_id"] == "KESSENCE_G2_CONVEX"
            else "Exact action-level formal pass transferred by covariant-density equivalence and domain inclusion; Solar and observational evidence remain separate."
        )
        uncertainty = (
            pass_scope
            if decision == "pass"
            else "Exact decisive Aether principal necessary-condition rejection; ranked only inside its separate comparison class."
            if decision == "reject"
            else "The first exact blocker is retained with rank=None; unproved downstream gates are missing evidence rather than measured poor performance."
        )
        rows["formal_adm_dirac"].append(
            _entry(
                record["candidate_id"],
                "generated_candidate",
                metrics,
                contract["evidence_status"],
                data_class,
                contract["gate_completeness"],
                record["blocker"],
                "scalable_formal_candidates",
                bindings["scalable_formal_candidates"],
                record["result_sha256"],
                uncertainty,
            )
        )

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
        "full_formal_action_evidence",
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
    g4_protocol = sources["g4_solar_protocol"]
    if (
        g4_protocol["candidate"]["candidate_id"] != g4_solar_row["candidate_id"]
        or g4_protocol["candidate_use_authorized"] is not False
        or g4_protocol["observational_authorization"] is not False
        or g4_protocol["descriptor_registration_status"]
        != "blocked_required_values_unset"
        or g4_protocol["remaining_registration_field_count"] != 9
        or g4_protocol["frozen_contracts"]["source_physics"][
            "source_class_theorem"
        ]["status"]
        != "pass"
    ):
        raise ValueError("G4 candidate-use Solar protocol is inconsistent")
    g4_real_sun = sources["g4_real_sun_source"]
    g4_tail = sources["g4_trace_tail"]
    g4_tail_record = g4_tail["candidate_records"][0]
    solar_parser = sources["solar_parser_readiness"]
    solar_calibration = sources["solar_calibration_readiness"]
    if (
        g4_real_sun["decision"] != "blocked"
        or g4_real_sun["real_source_interval_certificate_admissible"] is not False
        or g4_real_sun["theorem_requirement_counts"] != {"blocked": 6, "pass": 0}
        or g4_tail["theorem_pass_count"] != 1
        or g4_tail["real_source_instantiation_pass_count"] != 0
        or g4_tail_record["overall_decision"] != "blocked"
        or g4_tail_record["theorem_decision"] != "pass"
        or g4_tail_record["real_Sun_instantiation_decision"] != "blocked"
        or g4_tail_record["real_solar_bundle_admissible"] is not False
        or solar_calibration["filled_registration_field_count"] != 3
        or solar_calibration["remaining_registration_field_count"] != 6
        or solar_calibration["primary_record_access_count"] != 0
        or solar_parser["metadata_selection"]["primary_record_access_count"] != 0
        or solar_calibration["observational_authorization"] is not False
        or solar_parser["observational_authorization"] is not False
    ):
        raise ValueError("G4 real-Sun, tail, or calibration evidence is inconsistent")
    g4_entry = _entry(
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
            "source_class_uniqueness_theorem": "pass",
            "noncompact_trace_tail_theorem": "pass_conditionally",
            "real_sun_theorem_requirement_pass_count": 0,
            "real_sun_theorem_requirement_blocked_count": 6,
            "verified_registration_field_count": solar_calibration[
                "filled_registration_field_count"
            ],
            "remaining_registration_field_count": solar_calibration[
                "remaining_registration_field_count"
            ],
            "selected_detached_label_count": solar_parser["metadata_selection"][
                "selected_identity_count"
            ],
            "primary_record_access_count": solar_parser["metadata_selection"][
                "primary_record_access_count"
            ],
        },
        "blocked",
        "sealed_candidate_specific_solar_prediction",
        "incomplete",
        g4_tail_record["first_missing_premise"],
        "g4_trace_tail",
        bindings["g4_trace_tail"],
        g4_tail_record["provenance"]["binding_sha256"],
        "Analytic GR-like predictions, compact and noncompact source-class theorems, two verified parsers, and a covariance-aware raw-signal transform exist. Real-Sun tail facts and six registration fields remain blocked; no primary record was opened.",
    )
    g4_entry["lineage"]["supporting_artifacts"] = [
        bindings[label]
        for label in (
            "g4_solar_promotion",
            "g4_solar_protocol",
            "g4_real_sun_source",
            "solar_parser_readiness",
            "solar_calibration_readiness",
        )
    ]
    rows["solar_known_answer"].append(g4_entry)

    galaxy = sources["galaxy_direct_observable"]
    rows["galaxy_direct_observable"].append(_entry(
        "PRODUCTION-CANDIDATE-SET-70", "generated_candidate_set", {"candidate_count": galaxy["production_candidate_count"], "registered_prediction_bundle_count": galaxy["registered_prediction_bundle_count"]},
        "blocked", "sealed_direct_observable_scaffold", "incomplete", next(iter(galaxy["blocker_counts"])), "galaxy_direct_observable", bindings["galaxy_direct_observable"], galaxy["evaluator_binding_sha256"], "No observational source or candidate prediction bundle was opened; no galaxy performance was measured."))
    g4_galaxy = sources["g4_galaxy_readiness"]
    g4_galaxy_decision = g4_galaxy["current_evaluator_decision"]
    g4_forward = sources["g4_galaxy_forward_model"]
    g4_forward_decision = g4_forward["current_evaluator_decision"]
    g4_registration = sources["g4_galaxy_branch_distance"]
    g4_registration_decision = g4_registration["current_evaluator_decision"]
    g4_calibration = sources["g4_galaxy_calibration_evaluation"]
    g4_calibration_decision = g4_calibration["current_evaluator_decision"]
    g4_transform = sources["g4_galaxy_prediction_contract_transform"]
    g4_transform_decision = g4_transform["current_evaluator_decision"]
    g4_tooling = sources["g4_galaxy_manifest_bundle_tooling"]
    g4_tooling_decision = g4_tooling["unchanged_evaluator_decision"]
    g4_source_registry = sources["g4_galaxy_source_registry_admission"]
    g4_source_registry_decision = g4_source_registry["unchanged_evaluator_decision"]
    if (
        g4_galaxy["decision"] != "blocked"
        or g4_galaxy["descriptor_implementation_ready"] is not True
        or g4_galaxy["prediction_bundle_registered"] is not False
        or g4_galaxy["observational_data_opened"] is not False
        or g4_galaxy["primary_record_access_count"] != 0
        or g4_galaxy_decision["filled_registration_hash_count"] != 1
        or len(g4_galaxy_decision["missing_registration_hashes"]) != 17
        or g4_galaxy["synthetic_controls"]["shape"][
            "object_specific_gravity_parameter_count"
        ]
        != 0
    ):
        raise ValueError("reviewed G4 galaxy evaluator readiness is inconsistent")
    if (
        g4_forward["decision"] != "blocked"
        or g4_forward["prediction_bundle_registered"] is not False
        or g4_forward["observational_data_opened"] is not False
        or g4_forward["primary_record_access_count"] != 0
        or g4_forward["object_specific_gravity_parameter_count"] != 0
        or g4_forward["dark_matter_or_halo_inputs"] is not False
        or g4_forward["redshift_distance_inputs"] is not False
        or g4_forward_decision["filled_registration_hash_count"] != 3
        or len(g4_forward_decision["missing_registration_hashes"]) != 15
        or set(g4_forward["synthetic_controls"]["analytic_known_answers"].values())
        != {"pass"}
        or g4_forward["synthetic_controls"]["covariance"]["decision"] != "pass"
    ):
        raise ValueError("G4 galaxy forward-model readiness is inconsistent")
    if (
        g4_registration["decision"] != "blocked"
        or g4_registration["prediction_bundle_registered"] is not False
        or g4_registration["candidate_use_authorized"] is not False
        or g4_registration["observational_data_opened"] is not False
        or g4_registration["primary_record_access_count"] != 0
        or g4_registration["real_source_geometry_registered"] is not False
        or g4_registration["source_specific_branch_selection_proven"] is not False
        or g4_registration["object_specific_gravity_parameter_count"] != 0
        or g4_registration["dark_matter_or_halo_inputs"] is not False
        or g4_registration["redshift_distance_inputs"] is not False
        or g4_registration_decision["filled_registration_hash_count"] != 5
        or len(g4_registration_decision["missing_registration_hashes"]) != 13
        or set(g4_registration["newly_filled_registration_fields"])
        != {
            "branch_and_domain_contract_sha256",
            "distance_mode_contract_sha256",
        }
        or g4_registration["provenance"]["forward_model_predecessor_sha256"]
        != g4_forward["content_sha256"]
    ):
        raise ValueError("G4 galaxy branch/distance registration is inconsistent")
    if (
        g4_calibration["decision"] != "blocked"
        or g4_calibration["prediction_bundle_registered"] is not False
        or g4_calibration["candidate_use_authorized"] is not False
        or g4_calibration["observational_data_opened"] is not False
        or g4_calibration["primary_record_access_count"] != 0
        or g4_calibration["object_specific_gravity_parameter_count"] != 0
        or g4_calibration["dark_matter_or_halo_inputs"] is not False
        or g4_calibration["redshift_distance_inputs"] is not False
        or g4_calibration["paid_llm_spend_usd"] != 0.0
        or g4_calibration_decision["filled_registration_hash_count"] != 9
        or len(g4_calibration_decision["missing_registration_hashes"]) != 9
        or set(g4_calibration["newly_filled_registration_fields"])
        != {
            "baryonic_calibration_hierarchy_sha256",
            "joint_covariance_contract_sha256",
            "likelihood_contract_sha256",
            "stopping_rule_sha256",
        }
        or g4_calibration["provenance"]["predecessor_content_sha256"]
        != g4_registration["content_sha256"]
    ):
        raise ValueError("G4 galaxy calibration/evaluation registration is inconsistent")
    if (
        g4_transform["decision"] != "blocked"
        or g4_transform["prediction_bundle_registered"] is not False
        or g4_transform["candidate_use_authorized"] is not False
        or g4_transform["observational_data_opened"] is not False
        or g4_transform["primary_record_access_count"] != 0
        or g4_transform["object_specific_gravity_parameter_count"] != 0
        or g4_transform["dark_matter_or_halo_inputs"] is not False
        or g4_transform["redshift_distance_inputs"] is not False
        or g4_transform["paid_llm_spend_usd"] != 0.0
        or g4_transform["real_transform_inputs_registered"] is not False
        or g4_transform_decision["filled_registration_hash_count"] != 11
        or len(g4_transform_decision["missing_registration_hashes"]) != 7
        or set(g4_transform["newly_filled_registration_fields"])
        != {
            "prediction_bundle_contract_sha256",
            "raw_to_calibrated_transform_sha256",
        }
        or g4_transform["provenance"]["predecessor_content_sha256"]
        != g4_calibration["content_sha256"]
    ):
        raise ValueError("G4 galaxy prediction/transform registration is inconsistent")
    if (
        g4_tooling["decision"] != "blocked"
        or g4_tooling["candidate_use_authorized"] is not False
        or g4_tooling["observational_data_opened"] is not False
        or g4_tooling["primary_record_access_count"] != 0
        or g4_tooling["prediction_bundle_registered"] is not False
        or g4_tooling["dataset_manifest_registered"] is not False
        or g4_tooling["independent_registry_receipt_registered"] is not False
        or g4_tooling["dark_matter_or_halo_inputs"] is not False
        or g4_tooling["redshift_distance_inputs"] is not False
        or g4_tooling["paid_llm_spend_usd"] != 0.0
        or g4_tooling["filled_registration_hash_count"] != 11
        or g4_tooling["missing_registration_hash_count"] != 7
        or g4_tooling["newly_filled_registration_fields"] != {}
        or g4_tooling_decision["filled_registration_hash_count"] != 11
        or len(g4_tooling_decision["missing_registration_hashes"]) != 7
        or g4_tooling["synthetic_controls"][
            "manifest_audit_registration_admissible"
        ]
        is not False
        or g4_tooling["synthetic_controls"][
            "bundle_draft_registration_admissible"
        ]
        is not False
        or g4_tooling["synthetic_controls"]["synthetic_values_promoted"] is not False
        or g4_tooling["tooling_readiness"]["enabled"] is not False
        or g4_tooling["provenance"]["predecessor_content_sha256"]
        != g4_transform["content_sha256"]
    ):
        raise ValueError("G4 galaxy manifest/bundle tooling is inconsistent")
    if (
        g4_source_registry["decision"] != "blocked"
        or g4_source_registry["service_enabled"] is not False
        or g4_source_registry["start_requested"] is not False
        or g4_source_registry["source_records_admitted"] != 0
        or g4_source_registry["target_records_opened"] != 0
        or g4_source_registry["primary_record_access_count"] != 0
        or g4_source_registry["observation_opening_authorization_registered"]
        is not False
        or g4_source_registry["prediction_bundle_registered"] is not False
        or g4_source_registry["observational_data_opened"] is not False
        or g4_source_registry["dark_matter_or_halo_inputs"] is not False
        or g4_source_registry["redshift_distance_inputs"] is not False
        or g4_source_registry["object_specific_gravity_parameter_count"] != 0
        or g4_source_registry["paid_llm_spend_usd"] != 0.0
        or g4_source_registry["filled_registration_hash_count"] != 11
        or g4_source_registry["missing_registration_hash_count"] != 7
        or g4_source_registry["newly_filled_registration_fields"] != {}
        or g4_source_registry_decision["filled_registration_hash_count"] != 11
        or len(g4_source_registry_decision["missing_registration_hashes"]) != 7
        or g4_source_registry["provenance"]["manifest_bundle_tooling_sha256"]
        != g4_tooling["content_sha256"]
        or g4_source_registry["provenance"]["ledger_predecessor_sha256"]
        != g4_transform["content_sha256"]
    ):
        raise ValueError("G4 galaxy source-registry admission is inconsistent")
    g4_galaxy_entry = _entry(
        g4_galaxy["candidate"]["candidate_id"],
        "generated_candidate",
        {
            "filled_registration_hash_count": g4_transform_decision[
                "filled_registration_hash_count"
            ],
            "missing_registration_hash_count": len(
                g4_transform_decision["missing_registration_hashes"]
            ),
            "analytic_rotation_lensing_control_pass_count": sum(
                status == "pass"
                for status in g4_forward["synthetic_controls"][
                    "analytic_known_answers"
                ].values()
            ),
            "object_specific_gravity_parameter_count": 0,
            "prediction_bundle_registered": False,
            "primary_record_access_count": 0,
            "synthetic_covariance_control": g4_galaxy["synthetic_controls"][
                "covariance"
            ]["decision"],
            "synthetic_shape_control": g4_galaxy["synthetic_controls"]["shape"][
                "decision"
            ],
        },
        "blocked",
        "sealed_candidate_specific_galaxy_prediction",
        "incomplete",
        g4_transform["first_missing_premise"],
        "g4_galaxy_prediction_contract_transform",
        bindings["g4_galaxy_prediction_contract_transform"],
        g4_transform["provenance"]["binding_sha256"],
        "The exact scalar-free branch now also has a hash-bound prediction-bundle contract and a candidate-bound raw-to-calibrated covariance transform. Seven real-source, split, checkpoint, bundle-content/file, and primary-root hashes remain missing; no observation or halo/redshift-derived target was opened.",
    )
    g4_galaxy_entry["lineage"]["supporting_artifacts"] = [
        bindings["galaxy_direct_observable"],
        bindings["g4_galaxy_readiness"],
        bindings["g4_galaxy_forward_model"],
        bindings["g4_galaxy_branch_distance"],
        bindings["g4_galaxy_calibration_evaluation"],
        bindings["g4_galaxy_manifest_bundle_tooling"],
        bindings["g4_galaxy_source_registry_admission"],
    ]
    rows["galaxy_direct_observable"].append(g4_galaxy_entry)

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

    for record in structural_records:
        metrics = dict(record["structural_metrics"])
        rows["simplicity_complexity"].append(
            _entry(
                record["candidate_id"],
                "generated_candidate",
                metrics,
                "measured",
                record["structural_comparison_class"],
                "complete_for_category",
                None,
                "scalable_structural_metrics",
                bindings["scalable_structural_metrics"],
                record["content_sha256"],
                "Exact formula-structure measurements only; structural simplicity is not physical truth or viability.",
            )
        )
        equivalence = record["equivalence_evidence"]
        rows["novelty_non_equivalence"].append(
            _entry(
                record["candidate_id"],
                "generated_candidate",
                {
                    "action_content_sha256": record["action_sha256"],
                    "alias_count": equivalence["alias_count"],
                    "parameter_cell_class_size": equivalence[
                        "parameter_cell_class_size"
                    ],
                    "representative_action_class_size": equivalence[
                        "representative_action_class_size"
                    ],
                    "literature_novelty_claimed": False,
                },
                "measured",
                record["equivalence_comparison_class"],
                "complete_for_category",
                None,
                "scalable_structural_metrics",
                bindings["scalable_structural_metrics"],
                record["content_sha256"],
                "Exact equality and alias multiplicity inside this 256-cell manifest only; literature novelty is not claimed.",
            )
        )

    execution = sources["computational_execution"]
    for record in execution["work_records"]:
        rows["computational_robustness"].append(_entry(
            record["seed_id"], "generated_candidate", {"attempt": record["attempt"], "state": record["state"], "recovered_lease": False},
            "measured" if record["state"] == "succeeded" else "blocked", "deterministic_bounded_execution", "complete_for_category" if record["state"] == "succeeded" else "incomplete", None if record["state"] == "succeeded" else "execution_incomplete", "computational_execution", bindings["computational_execution"], record["output_lineage_sha256"], "A successful deterministic execution is software robustness evidence, not scientific validity."))
    for category_rows in rows.values():
        for row in category_rows:
            candidate_id = row["candidate_id"]
            row["theory_formula"] = (
                scalable_formula_by_id[candidate_id]
                if candidate_id in scalable_formula_by_id
                else _theory_formula(candidate_id, action_by_id)
            )
    return rows


def _validate_no_collapse(board: Mapping[str, Any]) -> None:
    text = json.dumps(board, sort_keys=True).lower()
    for forbidden in ("truth_score", "overall_score", "composite_score", "probability_of_truth"):
        if forbidden in text:
            raise ValueError("scalar truth-score collapse is forbidden")
    if board["data_eligibility"] != ELIGIBILITY:
        raise ValueError("leaderboards opened a forbidden data class")
    dossiers = board.get("theory_dossiers")
    if not isinstance(dossiers, Mapping) or len(dossiers) != 170:
        raise ValueError("theory dossier registry is incomplete")
    for dossier_id, dossier in dossiers.items():
        if dossier_id != dossier["dossier_id"]:
            raise ValueError("theory dossier registry key mismatch")
        node_counts = dict(
            sorted(Counter(node["status"] for node in dossier["hierarchy_nodes"]).items())
        )
        if node_counts != dossier["hierarchy_status_counts"]:
            raise ValueError("theory dossier hierarchy count mismatch")
        if any(
            node["status"]
            not in {"proven", "rejected", "blocked", "calibration_only"}
            or not node["scope"]
            or node["evidence_count"] < 1
            for node in dossier["hierarchy_nodes"]
        ):
            raise ValueError("theory dossier contains an invalid hierarchy node")
    for value in board["categories"].values():
        separate = value.get("completed_incomparable_evidence", [])
        if len(separate) != value.get("completed_separate_class_count", 0):
            raise ValueError("completed separate evidence count mismatch")
        if any(
            row["rank"] is not None
            or row["gate_completeness"] != "complete_for_category"
            or row["evidence_status"] not in {"pass", "reject", "measured"}
            or not isinstance(row.get("comparison_group_rank"), int)
            for row in separate
        ):
            raise ValueError("incomparable completed evidence entered the primary ranking")
        for row in (
            value["full_ranked"]
            + separate
            + value["unranked_blocked_or_untested"]
        ):
            formula = row.get("theory_formula")
            required_formula_keys = {
                "formula_type",
                "title",
                "defining_action",
                "plain_language",
                "fields",
                "parameters",
                "operator_terms",
                "action_content_sha256",
                "scope_note",
            }
            if not isinstance(formula, Mapping) or set(formula) != required_formula_keys:
                raise ValueError("leaderboard row lacks a complete theory formula")
            if not formula["title"] or not formula["defining_action"]:
                raise ValueError("leaderboard theory formula is empty")
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
    theory_dossiers = _theory_dossier_registry(
        sources["theory_dossiers"], config["sources"]["theory_dossiers"]
    )
    scalable_dossiers = _scalable_theory_dossier_registry(
        sources["scalable_explanation_dossiers"],
        config["sources"]["scalable_explanation_dossiers"],
    )
    if set(theory_dossiers) & set(scalable_dossiers):
        raise ValueError("theory dossier registries overlap candidate identities")
    theory_dossiers.update(scalable_dossiers)
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
        "schema_version": "sigma-scientific-leaderboards-1.1",
        "ranking_contract": "no global score; category-local completed comparable evidence only; equal comparison metrics share a rank and candidate_id only stabilizes display order",
        "categories": categories,
        "leaderboard_root_sha256": current_root,
        "history": history,
        "deltas_from_previous": deltas,
        "theory_dossiers": theory_dossiers,
        "data_eligibility": dict(ELIGIBILITY),
    }
    _validate_no_collapse(body)
    return {**body, "content_sha256": _sha(body)}
