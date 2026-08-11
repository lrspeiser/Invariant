from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.reviewed_future_parameter_formal_preflight_campaign import (
    build_reviewed_future_parameter_formal_preflight,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/reviewed_future_parameter_formal_preflight_campaign.json"
ARTIFACT_PATH = ROOT / "runs/engine/reviewed-future-parameter-formal-preflight-001.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_reviewed_future_parameter_formal_preflight(_config(), ROOT)


def test_exact_19_candidate_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["source_input_cell_count"] == 32
    assert rebuilt["source_new_candidate_count"] == rebuilt["candidate_count"] == 19
    assert rebuilt["source_deduplicated_candidate_count"] == 13
    assert rebuilt["family_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": 16,
        "CUBIC_HORNDESKI_G3_WEAK_CELL": 3,
    }
    assert rebuilt["decision_counts"] == {"blocked": 3, "pass": 14, "reject": 2}
    assert rebuilt["family_decision_counts"] == {
        "AETHER_K1234_PARAMETER_CELL": {"pass": 14, "reject": 2},
        "CUBIC_HORNDESKI_G3_WEAK_CELL": {"blocked": 3},
    }


def test_aether_decisions_are_exact_candidate_specializations(rebuilt: dict) -> None:
    records = [
        item
        for item in rebuilt["candidate_records"]
        if item["family_id"] == "AETHER_K1234_PARAMETER_CELL"
    ]
    assert len(records) == 16
    rejected = [item for item in records if item["decision"] == "reject"]
    assert {(item["parameters"]["c2"], item["parameters"]["c3"]) for item in rejected} == {
        ("0", "-1/16"),
        ("1/32", "-1/16"),
    }
    for item in records:
        parameters = item["parameters"]
        c123 = sum(Fraction(parameters[name]) for name in ("c1", "c2", "c3"))
        specialization = item["exact_specialization"]
        assert specialization["combinations"]["c123"] == str(c123)
        assert specialization["parameters"] == parameters
        assert specialization["adm_aligned_regular"] is True
        expected = "pass" if c123 > 0 else "reject"
        assert item["decision"] == expected
        assert item["gate_ledger"]["principal_and_linear_mode_necessary_condition"] == expected
        if expected == "pass":
            assert item["preflight_pass_scope"].endswith("no global energy or theory pass")
            assert item["next_required_formal_stage"].startswith("candidate_specific_Aether")


def test_g3_family_label_and_smaller_beta_do_not_create_a_pass(rebuilt: dict) -> None:
    records = [
        item
        for item in rebuilt["candidate_records"]
        if item["family_id"] == "CUBIC_HORNDESKI_G3_WEAK_CELL"
    ]
    assert {item["parameters"]["G3"] for item in records} == {
        "(33/4000)*X_phi",
        "(17/2000)*X_phi",
        "(9/1000)*X_phi",
    }
    blocker = "componentwise_normalized_local_jet_box_and_uniform_cone_certificate_missing"
    assert all(item["decision"] == "blocked" for item in records)
    assert all(item["first_blocker"] == blocker for item in records)
    for item in records:
        center = item["exact_specialization"]["center_principal_calibration"]
        weak = item["exact_specialization"]["declared_weak_cell_audit"]
        assert center["status"] == "pass_at_center_only"
        assert weak["status"] == "blocked"
        assert weak["componentwise_gradient_bounds"] is None
        assert weak["componentwise_hessian_bounds"] is None
        assert weak["frame_and_normalization_binding"] is None
        assert weak["uniform_effective_metric_interval"] is None
        assert weak["uniform_direction_sphere_cone_gap"] is None


def test_hash_lineage_formal_scope_and_data_seals(rebuilt: dict) -> None:
    assert rebuilt["reviewed_adapter_invocation_count"] == 7
    assert rebuilt["formal_preflight_completed"] is True
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["promotion"] == {
        "eligible_for_candidate_specific_formal_queue": 14,
        "rejected_before_candidate_specific_formal_queue": 2,
        "blocked_pending_exact_domain_registration": 3,
        "automatic_downstream_enqueue_performed": False,
    }
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == _config()["data_eligibility"]
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    assert all(
        item["expensive_candidate_specific_formal_run"] is False
        and item["observational_data_opened"] is False
        and item["data_eligibility"] == rebuilt["data_eligibility"]
        for item in rebuilt["candidate_records"]
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={**config["data_eligibility"], "observational_data_opened": True}
            ),
            "eligibility is open",
        ),
        (
            lambda config: config["reviewed_adapters"].pop(),
            "adapter registry is incomplete",
        ),
        (
            lambda config: config["source_status"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "implementation file hash mismatch",
        ),
        (
            lambda config: config["reviewed_adapters"][0].update(source_file_sha256="0" * 64),
            "source file hash mismatch",
        ),
    ],
)
def test_open_seals_missing_adapters_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_reviewed_future_parameter_formal_preflight(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_status"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_reviewed_future_parameter_formal_preflight(config, ROOT)
