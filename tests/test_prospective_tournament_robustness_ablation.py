from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.prospective_tournament_robustness_ablation import (
    CLAIMS,
    CONFIG_PATH,
    FAMILIES,
    OUTPUT_PATH,
    SEEDS,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_campaign(ROOT)


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    )


def test_exactly_eight_seeds_and_bounded_backend_policy_are_preregistered() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert tuple(config["seeds"]) == SEEDS
    assert len(set(config["seeds"])) == 8
    assert config["budgets"] == {
        "cpu_exact_replays": 8,
        "leave_one_family_out_ablations": 7,
        "maximum_candidates_per_replay": 21,
        "maximum_evaluation_replays": 168,
        "maximum_pareto_recomputations": 32,
    }
    assert config["policies"] == {
        "gpu_runtime_access": "forbidden_historical_receipt_only",
        "live_sqlite_access": "forbidden",
        "network_access": "forbidden",
        "operational_rehearsal_execution": "forbidden_interface_binding_only",
        "process_control": "forbidden",
        "stability_establishes_truth": False,
    }


def test_eight_cpu_replays_have_exact_candidate_gate_and_front_stability(
    result: dict[str, object],
) -> None:
    assert [row["seed"] for row in result["cpu_exact_replays"]] == list(SEEDS)
    assert (
        len(
            {
                row["candidate_order"][0]["candidate"]["artifact_id"]
                for row in result["cpu_exact_replays"]
            }
        )
        > 1
    )
    for replay in result["cpu_exact_replays"]:
        assert len(replay["candidate_order"]) == 21
        assert replay["candidate_overlap"] == {
            "intersection": 21,
            "union": 21,
            "jaccard": {"numerator": 1, "denominator": 1},
        }
        assert replay["evaluation_replays"] == 21
        assert replay["gate_outcomes_compared"] == 42
        assert replay["gate_outcomes_stable"] is True
        assert replay["pareto_recomputations"] == 2
        assert replay["fronts_stable"] is True
    assert len({row["gate_status_sha256"] for row in result["cpu_exact_replays"]}) == 1
    assert len({row["front_assignment_sha256"] for row in result["cpu_exact_replays"]}) == 1


def test_leave_one_generator_out_ablation_reports_exact_effects(
    result: dict[str, object],
) -> None:
    assert [row["removed_family"] for row in result["ablations"]] == list(FAMILIES)
    for row in result["ablations"]:
        assert row["remaining_candidate_count"] == 18
        assert row["candidate_overlap"] == {
            "intersection": 18,
            "union": 21,
            "jaccard": {"numerator": 6, "denominator": 7},
        }
        if row["removed_family"] == "bayesian":
            assert row["remaining_pareto_eligible_candidates"] == 0
            assert row["world_pass_to_reject_count"] == 2
            assert row["front_assignment_change_count"] == 2
            assert row["pareto_recomputations"] == 0
        else:
            assert row["remaining_pareto_eligible_candidates"] == 2
            assert row["world_pass_to_reject_count"] == 0
            assert row["front_assignment_change_count"] == 0
            assert row["pareto_recomputations"] == 2


def test_gpu_and_operational_bindings_preserve_no_runtime_boundary(
    result: dict[str, object],
) -> None:
    gpu = result["gpu_backend_binding"]
    assert gpu["summary"] == {
        "candidate_count": 163,
        "comparison_count": 5_341_184,
        "exact_rational_checks": 5_216,
        "violations": 0,
        "within_bounds": True,
        "role": "historical_synthetic_backend_control_not_tournament_gpu_replay",
    }
    assert gpu["tournament_gpu_replay_performed"] is False
    operational = result["operational_interface_binding"]
    assert operational["executed"] is False
    assert operational["run_callable"].endswith(":run_operational_rehearsal")
    assert operational["validator_callable"].endswith(":validate_operational_receipt")
    assert result["counts"] == {
        "registered_seeds": 8,
        "cpu_replay_passes": 8,
        "candidate_evaluation_replays": 168,
        "gate_outcome_comparisons": 336,
        "pareto_recomputations": 28,
        "leave_one_family_out_ablations": 7,
        "world_pass_to_reject_ablation_changes": 2,
        "tournament_gpu_replays": 0,
        "operational_rehearsal_executions": 0,
    }
    assert result["claims"] == CLAIMS


def test_result_is_deterministic_validated_and_committed(result: dict[str, object]) -> None:
    assert build_campaign(ROOT) == result
    validate_campaign(result, ROOT)
    assert json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8")) == result
    assert result["claims"]["stability_establishes_truth"] is False
    assert result["claims"]["stability_establishes_novelty"] is False
    assert result["claims"]["ablation_authorizes_promotion"] is False


def test_resealed_seed_stability_gpu_and_claim_tampers_fail_closed(
    result: dict[str, object],
) -> None:
    unstable = copy.deepcopy(result)
    unstable["cpu_exact_replays"][0]["gate_outcomes_stable"] = False
    _reseal(unstable)
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_campaign(unstable, ROOT)

    gpu_forgery = copy.deepcopy(result)
    gpu_forgery["gpu_backend_binding"]["summary"]["violations"] = 1
    _reseal(gpu_forgery)
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_campaign(gpu_forgery, ROOT)

    promoted = copy.deepcopy(result)
    promoted["claims"]["ablation_authorizes_promotion"] = True
    _reseal(promoted)
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_campaign(promoted, ROOT)
