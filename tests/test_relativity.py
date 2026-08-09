import json
from pathlib import Path

from sigma_theory_compiler.observation_eligibility import (
    audit_theory_observation_eligibility,
)
from sigma_theory_compiler.relativity import (
    galaxy_exterior_control,
    ppn_reference_check,
    run_relativity_reference_suite,
    schwarzschild_vacuum_check,
    solar_system_numeric_checks,
)

ROOT = Path(__file__).resolve().parents[1]


def test_schwarzschild_metric_is_vacuum() -> None:
    result = schwarzschild_vacuum_check()
    assert result["status"] == "pass"
    assert result["evidence"]["nonzero_components"] == {}


def test_gr_ppn_and_solar_golden_controls_pass() -> None:
    assert ppn_reference_check()["status"] == "pass"
    assert all(check["status"] == "pass" for check in solar_system_numeric_checks())


def test_baryons_only_exterior_has_keplerian_not_flat_scaling() -> None:
    result = galaxy_exterior_control()
    assert result["status"] == "expected_mismatch"
    assert result["evidence"]["predicted_log_slope"] == -0.5
    assert 0.49 < result["evidence"]["v_at_4theta_over_v_at_theta"] < 0.51
    assert "redshift-derived distance" in result["evidence"]["excluded_rescues"]


def test_reference_suite_blocks_without_formal_action_health() -> None:
    report = run_relativity_reference_suite()
    assert report["counts"] == {
        "golden_total": 5,
        "passed": 0,
        "failed": 0,
        "blocked": 5,
    }
    assert report["reference_action"]["action_variation_engine_status"] == "blocked"
    assert report["reference_action"]["constraint_and_degree_count_status"] == "blocked"


def test_reference_suite_is_bound_to_passing_einstein_hilbert_health() -> None:
    eligibility = audit_theory_observation_eligibility(
        ROOT
        / "runs"
        / "formal-controls-v1"
        / "action-health"
        / "einstein_hilbert_control"
        / "action-health.json",
        ROOT / "configs" / "observational_evidence_policy.json",
        mode="known_answer_reference",
    )
    assert eligibility["status"] == "eligible", eligibility["errors"]
    assert eligibility["reference_controls_allowed"]
    assert not eligibility["observational_dataset_opened"]
    report = run_relativity_reference_suite(eligibility)
    assert report["counts"] == {
        "golden_total": 5,
        "passed": 5,
        "failed": 0,
        "blocked": 0,
    }
    assert report["reference_action"]["action_variation_engine_status"] == "pass"
    assert report["reference_action"]["constraint_and_degree_count_status"] == "pass"


def test_control_health_does_not_authorize_candidate_dataset_opening() -> None:
    eligibility = audit_theory_observation_eligibility(
        ROOT
        / "runs"
        / "formal-controls-v1"
        / "action-health"
        / "einstein_hilbert_control"
        / "action-health.json",
        ROOT / "configs" / "observational_evidence_policy.json",
        mode="candidate_data",
    )
    assert eligibility["status"] == "ineligible"
    assert not eligibility["candidate_dataset_manifest_may_be_audited"]
    assert not eligibility["observational_dataset_opened"]
    assert eligibility["supernova_default_status"] == "excluded"
    assert not eligibility["redshift_distance_allowed_by_default"]


def test_observation_eligibility_rejects_stale_hamiltonian_hash(tmp_path) -> None:
    source = (
        ROOT
        / "runs"
        / "formal-controls-v1"
        / "action-health"
        / "einstein_hilbert_control"
        / "action-health.json"
    )
    health = json.loads(source.read_text(encoding="utf-8"))
    health["generated_hamiltonian_ir"]["content_sha256"] = "0" * 64
    stale = tmp_path / "stale-action-health.json"
    stale.write_text(json.dumps(health), encoding="utf-8")
    eligibility = audit_theory_observation_eligibility(
        stale,
        ROOT / "configs" / "observational_evidence_policy.json",
        mode="known_answer_reference",
    )
    assert eligibility["status"] == "ineligible"
    assert any(
        "generated_hamiltonian_ir content hash differs" in error
        for error in eligibility["errors"]
    )


def test_observation_eligibility_rejects_stale_q_operator_hash(tmp_path) -> None:
    source = (
        ROOT
        / "runs"
        / "formal-controls-v1"
        / "action-health"
        / "einstein_hilbert_control"
        / "action-health.json"
    )
    health = json.loads(source.read_text(encoding="utf-8"))
    health["generated_q_operator_ir"]["content_sha256"] = "0" * 64
    stale = tmp_path / "stale-q-action-health.json"
    stale.write_text(json.dumps(health), encoding="utf-8")
    eligibility = audit_theory_observation_eligibility(
        stale,
        ROOT / "configs" / "observational_evidence_policy.json",
        mode="known_answer_reference",
    )
    assert eligibility["status"] == "ineligible"
    assert any(
        "generated_q_operator_ir content hash differs" in error
        for error in eligibility["errors"]
    )


def test_observation_eligibility_rejects_stale_x_operator_hash(tmp_path) -> None:
    source = (
        ROOT
        / "runs"
        / "formal-controls-v1"
        / "action-health"
        / "einstein_hilbert_control"
        / "action-health.json"
    )
    health = json.loads(source.read_text(encoding="utf-8"))
    health["generated_x_operator_ir"]["content_sha256"] = "0" * 64
    stale = tmp_path / "stale-x-action-health.json"
    stale.write_text(json.dumps(health), encoding="utf-8")
    eligibility = audit_theory_observation_eligibility(
        stale,
        ROOT / "configs" / "observational_evidence_policy.json",
        mode="known_answer_reference",
    )
    assert eligibility["status"] == "ineligible"
    assert any(
        "generated_x_operator_ir content hash differs" in error
        for error in eligibility["errors"]
    )
