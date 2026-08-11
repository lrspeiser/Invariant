from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.g2_scalable_solar_prediction_readiness import (
    _sha,
    _solar_readiness,
    build_g2_scalable_solar_prediction_readiness,
)
from sigma_theory_compiler.promotion_orchestrator import ELIGIBILITY

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "g2_scalable_solar_prediction_readiness.json"
ARTIFACT = ROOT / "runs" / "engine" / "g2-scalable-solar-prediction-readiness.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_g2_scalable_solar_prediction_readiness(_load(CONFIG), ROOT)


def test_committed_artifact_is_exact_rebuild(rebuilt: dict) -> None:
    committed = _load(ARTIFACT)
    assert committed == rebuilt
    body = {key: value for key, value in committed.items() if key != "content_sha256"}
    assert committed["content_sha256"] == _sha(body)
    assert hashlib.sha256(ARTIFACT.read_bytes()).hexdigest() == (
        "6a8f971161d5535399a518954b5900f535e5c809c8fef0c019804bc4f8bbdb0a"
    )


def test_exact_action_replay_binds_universal_minimal_matter_coupling(rebuilt: dict) -> None:
    expected_actions = {
        "G3A-2f8983c88f504150381064f2": (
            "19f36a7c814ca11ace6de1270802a542872c35c27c7e64542eea672e16cbae88"
        ),
        "G3A-58e59412e5fe77cd54caf863": (
            "9457ba1ff99ecfdabc08200dda3ff15b8656b025d106fe2c2cd4abd77a01c3b5"
        ),
    }
    for record in rebuilt["candidate_records"]:
        replay = record["exact_action_replay"]
        assert replay["action_sha256"] == expected_actions[record["candidate_id"]]
        assert replay["fields"] == ["g_mu_nu", "phi"]
        assert [item["atom"] for item in replay["operators"]] == ["EH_R", "G2_PHI_X"]
        assert replay["matter_coupling"] == {"metric": "g_mu_nu", "universal": True}
        assert replay["data_eligibility"] == ELIGIBILITY


def test_candidate_specific_constant_scalar_branch_and_solar_formulas(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        prediction = record["scalar_free_prediction_certificate"]
        assert set(prediction["exact_residuals"].values()) == {"0"}
        assert prediction["linear_scalar_equation"] == ("box(delta_phi)=0_with_no_matter_source")
        assert prediction["Newtonian_prediction"] == {
            "G_cav_over_G_star": "1",
            "Poisson_equation": "laplacian(U)=4*pi*G_star*rho",
            "exterior_potential": "U=G_star*M/r",
            "status": "pass_on_exact_constant_phi_branch",
        }
        assert prediction["PPN_prediction"]["gamma"] == "1"
        assert prediction["PPN_prediction"]["beta"] == "1"
        assert prediction["vacuum_exterior"] == "Schwarzschild_is_exact_on_this_branch"
        assert prediction["known_answer_formulas"]["light_deflection"] == ("4*G_N*M/(b*c^2)")


def test_static_source_class_result_is_conditional_not_real_sun_evidence(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        source = record["static_source_class_certificate"]
        assert source["status"] == "pass_as_conditional_source_class_theorem"
        assert source["conditional_conclusion"] == "D_i(phi)=0_and_phi=phi_infinity"
        assert source["real_sun_instantiated"] is False
        assert source["required_domain"]["inner_boundary"] == "none"
        assert source["required_domain"]["ellipticity_sign"].endswith(">0_everywhere")
        assert len(source["missing_real_source_facts"]) == 4


def test_real_solar_readiness_has_exact_source_bundle_and_authorization_blockers(
    rebuilt: dict,
) -> None:
    expected_missing = [
        "candidate_specific_real_source_contract_sha256",
        "source_branch_domain_instantiation_sha256",
        "candidate_specific_evaluator_descriptor_sha256",
        "training_only_initial_state_sha256",
        "frozen_nuisance_likelihood_stopping_rule_sha256",
        "held_out_split_commitment_sha256",
        "action_bound_prediction_bundle_descriptor_sha256",
        "action_bound_prediction_bundle_file_sha256",
        "selected_primary_record_roots_sha256",
        "observation_opening_authorization_sha256",
    ]
    for record in rebuilt["candidate_records"]:
        readiness = record["real_solar_readiness"]
        assert readiness["decision"] == "blocked"
        assert readiness["first_missing_premise"] == (
            "registered_candidate_specific_real_source_and_action_bound_Solar_prediction_bundle"
        )
        assert readiness["missing_registration_fields"] == expected_missing
        assert readiness["candidate_use_authorized"] is False
        assert readiness["observation_opening_authorization_present"] is False
        assert readiness["observational_inputs_opened_by_this_audit"] is False
        assert record["solar_bundle"] == {
            "analytic_prediction_certificate_generated": True,
            "candidate_specific_evaluator_bundle_generated": False,
            "real_observational_bundle_generated": False,
            "real_observational_bundle_admissible": False,
            "status": "blocked_before_data_opening",
        }


def test_gr_controls_are_calibration_only_and_cannot_be_reused(rebuilt: dict) -> None:
    for record in rebuilt["candidate_records"]:
        control = record["GR_calibration_control"]
        assert control["role"] == "solver_calibration_control_not_candidate_evidence"
        assert len(control["statuses"]) == 5
        assert set(control["statuses"].values()) == {"pass"}
        assert record["real_solar_readiness"]["GR_control_bundle_reuse_negative"] == {
            "rejected": True,
            "reason": "Solar known-answer bundle cannot be attached to a discovery candidate",
        }


def test_counts_and_fail_closed_seals_are_exact(rebuilt: dict) -> None:
    assert rebuilt["candidate_count"] == 2
    assert rebuilt["decision_counts"] == {"blocked": 2}
    assert rebuilt["candidate_analytic_prediction_pass_count"] == 2
    assert rebuilt["conditional_static_source_class_pass_count"] == 2
    assert rebuilt["real_source_registration_pass_count"] == 0
    assert rebuilt["real_solar_bundle_count"] == 0
    assert rebuilt["real_solar_bundle_admissible_count"] == 0
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == ELIGIBILITY


def test_action_formal_and_source_authorization_tamper_fail_closed() -> None:
    config = _load(CONFIG)
    action = copy.deepcopy(config)
    action["targets"][0]["action_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="formal-pass target binding changed"):
        build_g2_scalable_solar_prediction_readiness(action, ROOT)

    formal = copy.deepcopy(config)
    formal["targets"][1]["formal_record_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="formal-pass target binding changed"):
        build_g2_scalable_solar_prediction_readiness(formal, ROOT)

    protocol = _load(ROOT / config["solar_contract"]["protocol"]["path"])
    audit = _load(ROOT / config["solar_contract"]["protocol_audit"]["path"])
    source = _load(ROOT / config["solar_contract"]["source_registration"]["path"])
    source["candidate_use_authorized"] = True
    target = {
        **config["targets"][0],
        "protocol_file_sha256": config["solar_contract"]["protocol"]["file_sha256"],
        "analytic_prediction_certificate_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="observation seal changed"):
        _solar_readiness(protocol, audit, source, target)


def test_adapter_hash_and_path_escape_fail_closed(tmp_path: Path) -> None:
    config = _load(CONFIG)
    source_hash = copy.deepcopy(config)
    source_hash["adapter_source"]["file_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="adapter source file hash mismatch"):
        build_g2_scalable_solar_prediction_readiness(source_hash, ROOT)

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    escaped = copy.deepcopy(config)
    escaped["formal_pass"] = {
        "path": str(outside),
        "file_sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "content_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="path escapes repository"):
        build_g2_scalable_solar_prediction_readiness(escaped, ROOT)
