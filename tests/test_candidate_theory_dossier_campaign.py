from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.candidate_theory_dossier_campaign import (
    SEED_IDS,
    _sha,
    build_candidate_theory_dossier_campaign,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "candidate_theory_dossier_campaign.json"
ARTIFACT_PATH = ROOT / "runs" / "engine" / "candidate-theory-dossier-campaign.json"


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return build_candidate_theory_dossier_campaign(config, ROOT)


def test_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert artifact == rebuilt
    body = {key: value for key, value in artifact.items() if key != "content_sha256"}
    assert artifact["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT_PATH.read_bytes()).hexdigest() == (
        "359d435a1dfd2903df4d64aa0e61659758e32517c2de80d4f41345202c461d0d"
    )


def test_exact_control_and_six_candidate_dossier_set(rebuilt: dict) -> None:
    assert rebuilt["dossier_count"] == 7
    assert rebuilt["known_answer_control_count"] == 1
    assert rebuilt["generated_candidate_count"] == 6
    assert rebuilt["hierarchy_node_status_counts"] == {
        "blocked": 17,
        "calibration_only": 2,
        "proven": 34,
    }
    ids = {item["dossier_id"] for item in rebuilt["dossiers"]}
    assert ids == {"GR-EINSTEIN-HILBERT", *SEED_IDS}


def test_typed_operator_terms_are_exact_compilation_copies(rebuilt: dict) -> None:
    compilation = json.loads(
        (ROOT / "runs" / "engine" / "covariant-grammar-v3-seed-compilation-campaign.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        item["seed_id"]: item["typed_action_ir"] for item in compilation["candidate_records"]
    }
    for dossier in rebuilt["dossiers"]:
        if dossier["role"] != "generated_candidate":
            continue
        nodes = {item["node_id"]: item for item in dossier["hierarchy_nodes"]}
        action = expected[dossier["dossier_id"]]
        assert nodes["defining_covariant_action"]["action_sha256"] == action[
            "content_sha256"
        ]
        assert nodes["exact_typed_operator_terms"]["operators"] == action["operators"]
        assert nodes["exact_typed_operator_terms"]["parameters"] == action[
            "parameters"
        ]


def test_every_hierarchy_node_is_hash_bound_and_host_path_free(rebuilt: dict) -> None:
    for dossier in rebuilt["dossiers"]:
        for node in dossier["hierarchy_nodes"]:
            assert node["status"] in {"proven", "blocked", "calibration_only"}
            assert len(node["content_sha256"]) == 64
            assert node["evidence"]
            for evidence in node["evidence"]:
                assert len(evidence["artifact_file_sha256"]) == 64
                assert not Path(evidence["artifact_path"]).is_absolute()
    serialized = json.dumps(rebuilt, sort_keys=True)
    assert "C:\\Users\\" not in serialized
    assert "C:/Users/" not in serialized


def test_gr_is_calibration_control_not_generated_candidate(rebuilt: dict) -> None:
    gr = next(item for item in rebuilt["dossiers"] if item["dossier_id"] == "GR-EINSTEIN-HILBERT")
    nodes = {item["node_id"]: item for item in gr["hierarchy_nodes"]}
    assert gr["role"] == "known_answer_calibration_control"
    assert gr["eligible_as_generated_candidate"] is False
    assert nodes["exact_typed_operator_terms"]["operator_terms"] == ["EH_R"]
    assert nodes["solar_known_answer_predictions"]["status"] == "calibration_only"
    assert nodes["solar_known_answer_predictions"]["pass_count"] == 5
    assert nodes["generic_nonlinear_total_energy_obligation"]["status"] == "blocked"
    assert nodes["generic_nonlinear_total_energy_obligation"]["upstream_status"] == (
        "not_claimed"
    )


def test_g4_formal_hierarchy_is_proven_but_real_solar_node_is_blocked(
    rebuilt: dict,
) -> None:
    dossier = next(
        item
        for item in rebuilt["dossiers"]
        if item["dossier_id"] == "G3-f9c598b70a77ea54009d8f18"
    )
    nodes = {item["node_id"]: item for item in dossier["hierarchy_nodes"]}
    assert dossier["overall_status"] == "blocked_after_formal_pass"
    for node_id in [
        "adm_dirac_obligation",
        "principal_symbol_obligation",
        "global_energy_obligation",
        "solar_analytic_prediction_on_scalar_free_branch",
        "solar_source_branch_theorem",
    ]:
        assert nodes[node_id]["status"] == "proven"
    assert nodes["solar_synthetic_GR_known_answer"]["status"] == "calibration_only"
    assert nodes["solar_prediction_obligation"]["status"] == "blocked"
    assert nodes["solar_prediction_obligation"]["first_missing_premise"] == (
        "registered_trace_tail_amplitude_decay_and_outer_transition"
    )


def test_other_generated_candidates_remain_fail_closed(rebuilt: dict) -> None:
    for dossier in rebuilt["dossiers"]:
        if dossier["role"] != "generated_candidate" or dossier["dossier_id"] == (
            "G3-f9c598b70a77ea54009d8f18"
        ):
            continue
        nodes = {item["node_id"]: item for item in dossier["hierarchy_nodes"]}
        assert dossier["overall_status"] == "blocked"
        assert nodes["global_energy_obligation"]["status"] == "blocked"
        assert nodes["solar_prediction_obligation"]["status"] == "blocked"
        assert nodes["euler_lagrange_and_noether"]["status"] == "proven"


def test_seals_and_no_observational_inference(rebuilt: dict) -> None:
    assert rebuilt["observational_authorization"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["tracking_target_values_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_tampered_binding_and_authorization_are_rejected() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    tampered_hash = copy.deepcopy(config)
    tampered_hash["source_bindings"]["compilation"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bound file hash mismatch"):
        build_candidate_theory_dossier_campaign(tampered_hash, ROOT)

    opened = copy.deepcopy(config)
    opened["observational_authorization"] = True
    with pytest.raises(ValueError, match="authorization must remain false"):
        build_candidate_theory_dossier_campaign(opened, ROOT)
