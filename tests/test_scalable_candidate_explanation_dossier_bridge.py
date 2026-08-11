from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY
from sigma_theory_compiler.scalable_candidate_explanation_dossier_bridge import (
    _family_formal_context,
    _human_readable_action,
    _sha,
    build_scalable_candidate_explanation_dossier_bridge,
)
from sigma_theory_compiler.scalable_formal_candidate_evidence_export import (
    iter_scalable_formal_candidate_evidence_records,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "scalable_candidate_explanation_dossier_bridge.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "scalable-candidate-explanation-dossier-bridge.json"


@pytest.fixture(scope="module")
def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt(config: dict) -> dict:
    return build_scalable_candidate_explanation_dossier_bridge(config, ROOT)


@pytest.fixture(scope="module")
def export_records() -> dict[str, dict]:
    export = json.loads(
        (ROOT / "runs" / "engine" / "scalable-formal-candidate-evidence-export-v1.1.json").read_text(
            encoding="utf-8"
        )
    )
    return {
        item["candidate_id"]: item
        for item in iter_scalable_formal_candidate_evidence_records(export)
    }


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "2b1552f679f35e6fe499a3650b7f4f27e7d5b3ec5ba181b63d8fbfd3a64d3033"
    )


def test_complete_candidate_and_hierarchy_accounting(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 163
    assert rebuilt["alias_count"] == 93
    assert rebuilt["family_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 128,
        "CONFORMAL_G4_PHI_SCALAR_TENSOR": 1,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 32,
        "KESSENCE_G2_CONVEX": 2,
    }
    assert rebuilt["formal_decision_counts"] == {"blocked": 158, "pass": 3, "reject": 2}
    assert rebuilt["hierarchy_node_status_counts"] == {
        "blocked": 321,
        "calibration_only": 163,
        "proven": 166,
        "rejected": 2,
    }
    assert rebuilt["comparison_data_class_counts"] == {
        "aether_aligned_minkowski_principal_necessary_condition": 2,
        "full_formal_action_evidence": 3,
        "unranked_incomplete": 158,
    }


def test_actions_are_exact_export_copies_not_family_templates(
    rebuilt: dict, export_records: dict[str, dict]
) -> None:
    for dossier in rebuilt["dossiers"]:
        source = export_records[dossier["candidate_id"]]
        formula = source["theory_formula_inputs"]
        action = dossier["action"]
        assert action["action_sha256"] == source["action_sha256"]
        assert action["formula_inputs_sha256"] == formula["formula_inputs_sha256"]
        assert action["fields"] == formula["fields"]
        assert action["parameters"] == formula["parameters"]
        assert action["ordered_operator_densities"] == formula["ordered_operator_densities"]
        assert action["human_readable_action"] == _human_readable_action(formula)
        for term in formula["ordered_operator_densities"]:
            assert f"({term['density']})" in action["human_readable_action"]["display_text"]


def test_blockers_rejections_and_formal_pass_keep_distinct_semantics(rebuilt: dict) -> None:
    dossiers = rebuilt["dossiers"]
    formal_nodes = {
        item["candidate_id"]: next(
            node
            for node in item["hierarchy_nodes"]
            if node["node_id"] == "reviewed_formal_evidence"
        )
        for item in dossiers
    }
    decisions = Counter(item["formal_decision"] for item in dossiers)
    statuses = Counter(node["status"] for node in formal_nodes.values())
    assert decisions == Counter({"blocked": 158, "pass": 3, "reject": 2})
    assert statuses == Counter({"blocked": 158, "proven": 3, "rejected": 2})
    for dossier in dossiers:
        node = formal_nodes[dossier["candidate_id"]]
        if dossier["formal_decision"] == "blocked":
            assert "not a measured failure or rejection" in node["scope"]
        observation = next(
            item
            for item in dossier["hierarchy_nodes"]
            if item["node_id"] == "downstream_observational_evidence"
        )
        assert observation["status"] == "blocked"

    g4 = next(
        item
        for item in dossiers
        if item["family_id"] == "CONFORMAL_G4_PHI_SCALAR_TENSOR"
    )
    assert g4["candidate_id"] == "G3A-e0eff4150989e3522dc6ba03"
    assert g4["preflight"]["decision"] == "blocked"
    assert formal_nodes[g4["candidate_id"]]["family_label_used_as_equivalence_evidence"] is False
    g2 = [
        item
        for item in dossiers
        if item["family_id"] == "KESSENCE_G2_CONVEX"
    ]
    assert len(g2) == 2
    assert all(item["formal_decision"] == "pass" for item in g2)
    assert all(
        formal_nodes[item["candidate_id"]]["actual_initial_data_set_instantiated"]
        is False
        for item in g2
    )


def test_comparison_classes_and_calibration_boundaries_are_preserved(rebuilt: dict) -> None:
    for dossier in rebuilt["dossiers"]:
        comparison = dossier["comparison_contract"]
        assert comparison["rank"] is None
        assert comparison["cross_class_ranking_allowed"] is False
        assert comparison["promotion_eligible"] is False
        boundary = next(
            item
            for item in dossier["hierarchy_nodes"]
            if item["node_id"] == "family_label_and_control_boundary"
        )
        assert boundary["status"] == "calibration_only"
        assert boundary["family_label_used_as_action_source"] is False
        assert boundary["family_label_used_as_equivalence_proof"] is False
        if dossier["formal_decision"] == "blocked":
            assert comparison["comparison_data_class"] is None
            assert comparison["rank_eligible_within_declared_class"] is False


def test_family_label_tamper_cannot_route_different_formal_evidence(
    config: dict, export_records: dict[str, dict]
) -> None:
    sources = {
        key: json.loads((ROOT / value["path"]).read_text(encoding="utf-8"))
        for key, value in config["source_bindings"].items()
    }
    record = copy.deepcopy(next(iter(export_records.values())))
    original_display = _human_readable_action(record["theory_formula_inputs"])
    record["family_id"] = "KESSENCE_G2_CONVEX"
    assert _human_readable_action(record["theory_formula_inputs"]) == original_display
    with pytest.raises(ValueError, match="family label and exact evidence source disagree"):
        _family_formal_context(record, sources)


def test_binding_authorization_and_path_tamper_fail_closed(config: dict) -> None:
    bad_hash = copy.deepcopy(config)
    bad_hash["source_bindings"]["scalable_export"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound file hash mismatch"):
        build_scalable_candidate_explanation_dossier_bridge(bad_hash, ROOT)

    escaped = copy.deepcopy(config)
    escaped["source_bindings"]["scalable_export"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="source path escapes repository"):
        build_scalable_candidate_explanation_dossier_bridge(escaped, ROOT)

    authorized = copy.deepcopy(config)
    authorized["observational_authorization"] = True
    with pytest.raises(ValueError, match="authorization must remain false"):
        build_scalable_candidate_explanation_dossier_bridge(authorized, ROOT)


def test_seals_portability_and_no_scalar_ranking(rebuilt: dict) -> None:
    assert rebuilt["observational_authorization"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY
    assert rebuilt["negative_control_results"] == {
        "action_display_uses_exact_exported_densities_only": "pass",
        "blocked_is_not_rewritten_as_rejected": "pass",
        "comparison_classes_are_not_merged": "pass",
        "family_label_is_not_action_or_equivalence_evidence": "pass",
        "observations_and_paid_llm_remain_sealed": "pass",
    }
    serialized = json.dumps(rebuilt, sort_keys=True).lower()
    assert "c:\\users\\" not in serialized
    assert "c:/users/" not in serialized
    assert '"truth_score"' not in serialized
    assert '"overall_score"' not in serialized
    assert '"probability"' not in serialized
