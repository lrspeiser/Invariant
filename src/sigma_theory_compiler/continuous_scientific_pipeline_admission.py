from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "sigma-continuous-scientific-pipeline-admission-config-1.0"
ARTIFACT_SCHEMA = "sigma-continuous-scientific-pipeline-admission-readiness-1.0"
CONFIG_RELATIVE_PATH = "configs/continuous_scientific_pipeline_admission.json"
SOURCE_RELATIVE_PATH = "src/sigma_theory_compiler/continuous_scientific_pipeline_admission.py"
TEST_RELATIVE_PATH = "tests/test_continuous_scientific_pipeline_admission.py"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

CONFIG_KEYS = {
    "schema_version",
    "contract_id",
    "stage_order",
    "stage_actions",
    "cpu_resource_gate",
    "gpu_lane_contract",
    "ranking_gate",
    "maximum_state_bytes",
    "bindings",
    "seals",
}
STATE_KEYS = {
    "cpu_utilization_percent",
    "available_ram_mib",
    "cpu_generation_owner_active",
    "gpu_handoff_state",
    "generated_receipt",
    "formal_receipt",
    "last_ranked_candidate_root",
    "dashboard_service_healthy",
    "dashboard_core_parity",
}
SEALS = {
    "service_started": False,
    "database_created_or_opened": False,
    "existing_process_signaled": False,
    "gpu_owner_acquired": False,
    "observations_opened": False,
    "dark_matter_or_halo_inputs": False,
    "redshift_or_cosmology_inputs": False,
    "paid_llm_calls": False,
    "scientific_or_ranking_pass_promoted": False,
}


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash_matches(value: Mapping[str, Any]) -> bool:
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    return value.get("content_sha256") == hashlib.sha256(_canonical(body)).hexdigest()


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any] | None:
    if set(binding) not in ({"path", "file_sha256"}, {"path", "file_sha256", "content_sha256"}):
        raise ValueError("binding key set mismatch")
    path = (root / str(binding["path"])).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("binding escapes project root") from error
    if not path.is_file() or _file_sha256(path) != binding["file_sha256"]:
        raise ValueError("binding file hash mismatch")
    if "content_sha256" not in binding:
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not _content_hash_matches(value) or value["content_sha256"] != binding["content_sha256"]:
        raise ValueError("binding content hash mismatch")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != CONFIG_KEYS or config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("config contract mismatch")
    if config.get("contract_id") != "continuous-scientific-generation-validation-ranking-001":
        raise ValueError("contract id mismatch")
    if config.get("stage_order") != ["generate_and_screen", "formal_validate", "rank_project"]:
        raise ValueError("stage order mismatch")
    if config.get("stage_actions") != {
        "generate_and_screen": "real_formula_cpu_batch_evaluator",
        "formal_validate": "candidate_specific_formal_receipt_gate",
        "rank_project": "scientific_leaderboard_rebuild_only",
    }:
        raise ValueError("stage action allowlist mismatch")
    if config.get("cpu_resource_gate") != {
        "continuous_worker_count": 15,
        "gpu_workers": 0,
        "minimum_available_ram_mib": 32768,
        "backoff_at_or_above_cpu_percent": 92,
        "maximum_actions_per_cycle": 1,
    }:
        raise ValueError("CPU resource contract mismatch")
    if config.get("gpu_lane_contract") != {
        "independent_deferred_owner": True,
        "gpu_owner_count": 1,
        "cpu_workers": 0,
        "coordinator_may_start_or_stop_gpu_lane": False,
    }:
        raise ValueError("GPU ownership contract mismatch")
    if config.get("ranking_gate") != {
        "sampled_static_screen_rank_eligible": False,
        "formal_pass_required": True,
        "complete_comparable_evidence_required": True,
        "candidate_root_must_match": True,
        "unchanged_root_rebuild_allowed": False,
        "direct_rank_assignment_allowed": False,
    }:
        raise ValueError("ranking contract mismatch")
    if config.get("maximum_state_bytes") != 65536 or config.get("seals") != SEALS:
        raise ValueError("state bound or seal mismatch")


def _validate_generated_receipt(receipt: Mapping[str, Any]) -> None:
    expected = {
        "content_sha256",
        "candidate_root_sha256",
        "screen_decision",
        "unique_formula_count",
        "theory_pass_claimed",
        "observations_opened",
        "rank_eligible",
    }
    if (
        set(receipt) != expected
        or not _content_hash_matches(receipt)
        or not _SHA256.fullmatch(str(receipt.get("candidate_root_sha256", "")))
        or receipt.get("screen_decision") not in {"pass", "reject", "ambiguous"}
        or not isinstance(receipt.get("unique_formula_count"), int)
        or isinstance(receipt.get("unique_formula_count"), bool)
        or receipt["unique_formula_count"] <= 0
        or receipt.get("theory_pass_claimed") is not False
        or receipt.get("observations_opened") is not False
        or receipt.get("rank_eligible") is not False
    ):
        raise ValueError("generated receipt contract mismatch")


def _validate_formal_receipt(receipt: Mapping[str, Any]) -> None:
    expected = {
        "content_sha256",
        "candidate_root_sha256",
        "generated_receipt_sha256",
        "decision",
        "complete_comparable_evidence",
        "observations_opened",
        "forbidden_target_inputs_opened",
    }
    if (
        set(receipt) != expected
        or not _content_hash_matches(receipt)
        or not _SHA256.fullmatch(str(receipt.get("candidate_root_sha256", "")))
        or not _SHA256.fullmatch(str(receipt.get("generated_receipt_sha256", "")))
        or receipt.get("decision") not in {"pass", "block", "reject"}
        or not isinstance(receipt.get("complete_comparable_evidence"), bool)
        or receipt.get("observations_opened") is not False
        or receipt.get("forbidden_target_inputs_opened") is not False
    ):
        raise ValueError("formal receipt contract mismatch")


def admit_cycle(state: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    """Select at most one reviewed pipeline action without executing it."""

    _validate_config(config)
    if set(state) != STATE_KEYS or len(_canonical(state)) > config["maximum_state_bytes"]:
        raise ValueError("cycle state contract mismatch")
    cpu = state["cpu_utilization_percent"]
    ram = state["available_ram_mib"]
    if not isinstance(cpu, (int, float)) or isinstance(cpu, bool) or not 0 <= cpu <= 100:
        raise ValueError("CPU utilization is invalid")
    if not isinstance(ram, int) or isinstance(ram, bool) or ram < 0:
        raise ValueError("available RAM is invalid")
    if state["gpu_handoff_state"] not in {"waiting", "running", "stopped"}:
        raise ValueError("GPU handoff state is invalid")
    for key in (
        "cpu_generation_owner_active",
        "dashboard_service_healthy",
        "dashboard_core_parity",
    ):
        if not isinstance(state[key], bool):
            raise TypeError(f"{key} is invalid")

    generated = state["generated_receipt"]
    formal = state["formal_receipt"]
    if generated is not None:
        if not isinstance(generated, Mapping):
            raise ValueError("generated receipt is invalid")
        _validate_generated_receipt(generated)
    if formal is not None:
        if not isinstance(formal, Mapping):
            raise ValueError("formal receipt is invalid")
        _validate_formal_receipt(formal)
    last_root = state["last_ranked_candidate_root"]
    if last_root is not None and not _SHA256.fullmatch(str(last_root)):
        raise ValueError("last ranked root is invalid")

    action = "wait"
    blocker = "no_admissible_scientific_action"
    if formal is not None:
        if (
            generated is None
            or formal["candidate_root_sha256"] != generated["candidate_root_sha256"]
            or formal["generated_receipt_sha256"] != generated["content_sha256"]
        ):
            blocker = "formal_receipt_candidate_root_mismatch"
        elif formal["decision"] != "pass" or not formal["complete_comparable_evidence"]:
            blocker = "formal_evidence_not_complete_and_comparable"
        elif formal["candidate_root_sha256"] == last_root:
            blocker = "candidate_root_already_rank_projected"
        else:
            action = "rank_project"
            blocker = None
    elif generated is not None:
        if generated["screen_decision"] == "pass":
            action = "formal_validate"
            blocker = None
        else:
            blocker = "sampled_static_screen_not_a_formal_or_ranking_pass"
    elif state["cpu_generation_owner_active"]:
        blocker = "CPU_generation_owner_already_active"
    elif cpu >= config["cpu_resource_gate"]["backoff_at_or_above_cpu_percent"]:
        blocker = "CPU_backoff_threshold_reached"
    elif ram < config["cpu_resource_gate"]["minimum_available_ram_mib"]:
        blocker = "available_RAM_below_floor"
    else:
        action = "generate_and_screen"
        blocker = None

    return {
        "action": action,
        "blocker": blocker,
        "action_count": 0 if action == "wait" else 1,
        "cpu_workers_if_generation": (
            config["cpu_resource_gate"]["continuous_worker_count"]
            if action == "generate_and_screen"
            else 0
        ),
        "gpu_workers_requested": 0,
        "gpu_lane_observed_state": state["gpu_handoff_state"],
        "gpu_lane_control_attempted": False,
        "direct_rank_assignment": False,
        "leaderboard_rebuild_admitted": action == "rank_project",
        "dashboard_publication_admitted": bool(
            action == "rank_project"
            and state["dashboard_service_healthy"]
            and state["dashboard_core_parity"]
        ),
    }


def build_continuous_scientific_pipeline_readiness(root: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    loaded = {label: _load_bound(root, binding) for label, binding in config["bindings"].items()}
    cpu_profile = loaded["cpu_profile"]
    gpu_scheduler = loaded["gpu_scheduler_readiness"]
    gpu_handoff = loaded["gpu_handoff_readiness"]
    leaderboard = loaded["leaderboard_config"]
    if (
        cpu_profile is None
        or cpu_profile.get("contract", {}).get("allowlisted_evaluator")
        != "sigma_theory_compiler.real_formula_execution:cpu_formula_batch_evaluator"
        or cpu_profile.get("contract", {}).get("gpu_workers") != 0
        or [stage.get("workers") for stage in cpu_profile.get("stages", [])] != [15, 16]
        or cpu_profile.get("stages", [None, {}])[1].get("backoff_threshold_exceeded") is not True
    ):
        raise ValueError("CPU scientific profile mismatch")
    if (
        gpu_scheduler is None
        or gpu_scheduler.get("scheduler_contract", {}).get("gpu_owner_count") != 1
        or gpu_scheduler.get("scheduler_contract", {}).get("cpu_worker_count") != 0
        or gpu_handoff is None
        or gpu_handoff.get("handoff_contract", {}).get("gpu_workers") != 1
        or gpu_handoff.get("handoff_contract", {}).get("cpu_workers") != 0
    ):
        raise ValueError("deferred GPU contract mismatch")
    if leaderboard is None or any(leaderboard.get("data_eligibility", {}).values()):
        raise ValueError("leaderboard data eligibility mismatch")

    root_a = "a" * 64
    root_b = "b" * 64
    generated_body = {
        "candidate_root_sha256": root_a,
        "screen_decision": "pass",
        "unique_formula_count": 1024,
        "theory_pass_claimed": False,
        "observations_opened": False,
        "rank_eligible": False,
    }
    generated = {
        **generated_body,
        "content_sha256": hashlib.sha256(_canonical(generated_body)).hexdigest(),
    }
    formal_body = {
        "candidate_root_sha256": root_a,
        "generated_receipt_sha256": generated["content_sha256"],
        "decision": "pass",
        "complete_comparable_evidence": True,
        "observations_opened": False,
        "forbidden_target_inputs_opened": False,
    }
    formal = {
        **formal_body,
        "content_sha256": hashlib.sha256(_canonical(formal_body)).hexdigest(),
    }
    rejected_body = {**generated_body, "screen_decision": "reject"}
    rejected = {
        **rejected_body,
        "content_sha256": hashlib.sha256(_canonical(rejected_body)).hexdigest(),
    }
    base = {
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
    scenarios = {
        "safe_idle_generation": admit_cycle(base, config),
        "generated_requires_formal": admit_cycle({**base, "generated_receipt": generated}, config),
        "formal_complete_rebuild": admit_cycle(
            {**base, "generated_receipt": generated, "formal_receipt": formal}, config
        ),
        "sampled_static_reject_not_ranked": admit_cycle(
            {**base, "generated_receipt": rejected}, config
        ),
        "cpu_overload_backoff": admit_cycle({**base, "cpu_utilization_percent": 92}, config),
        "unchanged_root_no_rebuild": admit_cycle(
            {
                **base,
                "generated_receipt": generated,
                "formal_receipt": formal,
                "last_ranked_candidate_root": root_a,
            },
            config,
        ),
        "root_mismatch_rejected": admit_cycle(
            {
                **base,
                "generated_receipt": generated,
                "formal_receipt": {
                    **formal,
                    "candidate_root_sha256": root_b,
                    "content_sha256": hashlib.sha256(
                        _canonical({**formal_body, "candidate_root_sha256": root_b})
                    ).hexdigest(),
                },
            },
            config,
        ),
    }
    expected = {
        "safe_idle_generation": "generate_and_screen",
        "generated_requires_formal": "formal_validate",
        "formal_complete_rebuild": "rank_project",
        "sampled_static_reject_not_ranked": "wait",
        "cpu_overload_backoff": "wait",
        "unchanged_root_no_rebuild": "wait",
        "root_mismatch_rejected": "wait",
    }
    if {key: value["action"] for key, value in scenarios.items()} != expected:
        raise ValueError("admission scenario control failed")

    implementation = {}
    for label, relative in (
        ("config", CONFIG_RELATIVE_PATH),
        ("source", SOURCE_RELATIVE_PATH),
        ("test", TEST_RELATIVE_PATH),
    ):
        path = root / relative
        implementation[label] = {"path": relative, "file_sha256": _file_sha256(path)}
    body = {
        "schema_version": ARTIFACT_SCHEMA,
        "contract_id": config["contract_id"],
        "decision": "admission_state_machine_ready_continuous_service_loop_not_implemented_not_started",
        "stage_order": config["stage_order"],
        "resource_contract": {
            "CPU_generation_workers": 15,
            "CPU_backoff_at_or_above_percent": 92,
            "minimum_available_ram_mib": 32768,
            "GPU_lane_independent_deferred_single_owner": True,
            "maximum_actions_per_cycle": 1,
        },
        "scientific_contract": {
            "sampled_static_screen_is_not_formal_or_ranking_pass": True,
            "formal_candidate_root_must_match_generation_root": True,
            "complete_comparable_evidence_required_for_ranking": True,
            "ranking_action_is_leaderboard_rebuild_not_direct_rank_assignment": True,
            "dashboard_publication_requires_healthy_core_parity": True,
        },
        "scenario_controls": scenarios,
        "counts": {
            "scenario_controls": len(scenarios),
            "generation_actions": 1,
            "formal_validation_actions": 1,
            "ranking_rebuild_actions": 1,
            "fail_closed_waits": 4,
            "services_started": 0,
            "databases_created_or_opened": 0,
            "GPU_owners_acquired": 0,
            "scientific_or_ranking_passes_promoted": 0,
        },
        "implementation_bindings": implementation,
        "predecessor_bindings": config["bindings"],
        "seals": SEALS,
        "first_remaining_blocker": (
            "implement_bounded_single_owner_service_loop_and_isolated_durable_queue_without_live_SQLite"
        ),
        "runtime_observations_in_immutable_artifact": False,
    }
    return {**body, "content_sha256": hashlib.sha256(_canonical(body)).hexdigest()}


def validate_continuous_scientific_pipeline_readiness(
    artifact: Mapping[str, Any], root: Path, config_path: Path
) -> None:
    rebuilt = build_continuous_scientific_pipeline_readiness(root, config_path)
    if not _content_hash_matches(artifact) or artifact != rebuilt:
        raise ValueError("continuous pipeline readiness differs from exact reconstruction")


def write_continuous_scientific_pipeline_readiness(
    artifact: Mapping[str, Any], output: Path
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output
