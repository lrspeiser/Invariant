import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.continuous_scientific_pipeline_admission import (
    admit_cycle,
    build_continuous_scientific_pipeline_readiness,
    validate_continuous_scientific_pipeline_readiness,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "continuous_scientific_pipeline_admission.json"
ARTIFACT = ROOT / "runs" / "engine" / "continuous-scientific-pipeline-admission-readiness.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _state() -> dict:
    return {
        "cpu_utilization_percent": 40,
        "available_ram_mib": 65536,
        "cpu_generation_owner_active": False,
        "gpu_handoff_state": "waiting",
        "generated_receipt": None,
        "formal_receipt": None,
        "last_ranked_candidate_root": None,
        "dashboard_service_healthy": True,
        "dashboard_core_parity": True,
    }


def _generated() -> dict:
    body = {
        "candidate_root_sha256": "a" * 64,
        "screen_decision": "pass",
        "unique_formula_count": 1024,
        "theory_pass_claimed": False,
        "observations_opened": False,
        "rank_eligible": False,
    }
    return {
        **body,
        "content_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _formal() -> dict:
    body = {
        "candidate_root_sha256": "a" * 64,
        "generated_receipt_sha256": _generated()["content_sha256"],
        "decision": "pass",
        "complete_comparable_evidence": True,
        "observations_opened": False,
        "forbidden_target_inputs_opened": False,
    }
    return {
        **body,
        "content_sha256": hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def test_readiness_is_exact_and_persisted() -> None:
    result = build_continuous_scientific_pipeline_readiness(ROOT, CONFIG)
    assert result == json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_continuous_scientific_pipeline_readiness(result, ROOT, CONFIG)
    assert result["decision"] == (
        "admission_state_machine_ready_continuous_service_loop_not_implemented_not_started"
    )
    assert result["counts"] == {
        "scenario_controls": 7,
        "generation_actions": 1,
        "formal_validation_actions": 1,
        "ranking_rebuild_actions": 1,
        "fail_closed_waits": 4,
        "services_started": 0,
        "databases_created_or_opened": 0,
        "GPU_owners_acquired": 0,
        "scientific_or_ranking_passes_promoted": 0,
    }
    assert not any(result["seals"].values())


def test_stage_barriers_and_resources_are_fail_closed() -> None:
    config = _config()
    generated = _generated()
    formal = _formal()
    assert admit_cycle(_state(), config)["action"] == "generate_and_screen"
    assert admit_cycle({**_state(), "generated_receipt": generated}, config)["action"] == (
        "formal_validate"
    )
    ranking = admit_cycle(
        {**_state(), "generated_receipt": generated, "formal_receipt": formal}, config
    )
    assert ranking["action"] == "rank_project"
    assert ranking["leaderboard_rebuild_admitted"]
    assert ranking["dashboard_publication_admitted"]
    assert not ranking["direct_rank_assignment"]
    assert not ranking["gpu_lane_control_attempted"]

    assert admit_cycle({**_state(), "cpu_utilization_percent": 92}, config)["action"] == ("wait")
    assert admit_cycle({**_state(), "available_ram_mib": 32767}, config)["action"] == ("wait")
    assert (
        admit_cycle({**_state(), "cpu_generation_owner_active": True}, config)["action"] == "wait"
    )


def test_screen_or_mismatched_formal_receipt_never_enters_ranking() -> None:
    config = _config()
    rejected_body = {**_generated(), "screen_decision": "reject"}
    rejected_body.pop("content_sha256")
    rejected = {
        **rejected_body,
        "content_sha256": hashlib.sha256(
            json.dumps(rejected_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    result = admit_cycle({**_state(), "generated_receipt": rejected}, config)
    assert result["action"] == "wait"
    assert result["blocker"] == "sampled_static_screen_not_a_formal_or_ranking_pass"

    mismatch_body = {**_formal(), "candidate_root_sha256": "b" * 64}
    mismatch_body.pop("content_sha256")
    mismatch = {
        **mismatch_body,
        "content_sha256": hashlib.sha256(
            json.dumps(mismatch_body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    result = admit_cycle(
        {**_state(), "generated_receipt": _generated(), "formal_receipt": mismatch}, config
    )
    assert result["action"] == "wait"
    assert result["blocker"] == "formal_receipt_candidate_root_mismatch"
    assert not result["leaderboard_rebuild_admitted"]

    tampered = {**_generated(), "unique_formula_count": 2048}
    with pytest.raises(ValueError, match="generated receipt contract"):
        admit_cycle({**_state(), "generated_receipt": tampered}, config)


def test_unknown_state_config_and_artifact_tamper_reject() -> None:
    config = _config()
    state = {**_state(), "invented_field": False}
    with pytest.raises(ValueError, match="cycle state contract"):
        admit_cycle(state, config)

    corrupt_config = copy.deepcopy(config)
    corrupt_config["stage_actions"]["formal_validate"] = "arbitrary.callable"
    with pytest.raises(ValueError, match="allowlist"):
        admit_cycle(_state(), corrupt_config)

    artifact = build_continuous_scientific_pipeline_readiness(ROOT, CONFIG)
    artifact["counts"]["ranking_rebuild_actions"] = 7
    with pytest.raises(ValueError, match="exact reconstruction"):
        validate_continuous_scientific_pipeline_readiness(artifact, ROOT, CONFIG)
