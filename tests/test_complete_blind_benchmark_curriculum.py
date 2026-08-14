from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import pytest

from sigma_theory_compiler.complete_blind_benchmark_curriculum import (
    CONFIG_PATH,
    OLD_OUTPUT_PATH,
    OUTPUT_PATH,
    TARGET_PATH,
    CompleteCurriculumError,
    _unseal_target_rules,
    build_curriculum,
    expand_registry,
    validate_curriculum,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def checked() -> dict:
    value = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_curriculum(value, root=ROOT)
    assert value == build_curriculum(ROOT)
    return value


def _reseal(value: dict) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: item for key, item in value.items() if key != "content_sha256"}
    )


def test_registry_is_exact_complete_cartesian_product() -> None:
    slots = expand_registry(CONFIG)
    assert len(slots) == len({slot["slot_id"] for slot in slots}) == 200
    assert Counter(slot["cohort"] for slot in slots) == {"historical": 100, "synthetic": 100}
    assert Counter(slot["domain"] for slot in slots) == {
        "algebra": 40,
        "arithmetic": 40,
        "combinatorics": 40,
        "geometry": 40,
        "number_theory": 40,
    }
    assert Counter(slot["artifact_type"] for slot in slots) == {
        "counterexample_construction": 100,
        "theorem_rediscovery": 100,
    }
    assert Counter(slot["level"] for slot in slots) == {1: 40, 2: 40, 3: 40, 4: 40, 5: 40}
    assert Counter(slot["ordinal"] for slot in slots) == {1: 100, 2: 100}


def test_current_authority_is_200_ready_zero_missing_or_invalid(checked: dict) -> None:
    assert checked["decision"] == "PASS"
    assert checked["counts"] == {
        "counterexample_artifacts": 100,
        "exact_proof_artifacts": 100,
        "historical_ready": 100,
        "invalid": 0,
        "missing": 0,
        "outcome_pass": 100,
        "outcome_reject": 100,
        "ready": 200,
        "registered": 200,
        "synthetic_ready": 100,
    }
    assert len(checked["registered_slots"]) == 200
    assert all(slot["ready"] for slot in checked["registered_slots"])
    assert checked["claims"]["curriculum_complete"] is True


def test_every_candidate_and_target_is_independently_bound_before_unseal(checked: dict) -> None:
    phase = checked["phase_a"]
    assert phase["candidate_count"] == 200
    assert len(phase["slot_candidates"]) == 200
    assert phase["target_reads"] == 0
    assert phase["candidate_generation_after_target_unseal"] == 0
    by_slot = {row["slot_id"]: row for row in phase["slot_candidates"]}
    for slot in checked["registered_slots"]:
        frozen = by_slot[slot["slot_id"]]
        assert frozen["candidate_sealed_before_target_unseal"] is True
        assert slot["candidate_sealed_before_target_unseal"] is True
        assert slot["candidate"] == frozen["candidate"]
        assert slot["candidate"]["target_fields_read"] == []
        assert slot["target_commitment_validated"] is True
        assert (
            slot["candidate"]["target_commitment_sha256"]
            == slot["target"]["target_commitment_sha256"]
        )
        candidate_body = {
            key: item for key, item in slot["candidate"].items() if key != "content_sha256"
        }
        target_body = {key: item for key, item in slot["target"].items() if key != "content_sha256"}
        slot_body = {key: item for key, item in slot.items() if key != "content_sha256"}
        assert slot["candidate"]["content_sha256"] == canonical_sha256(candidate_body)
        assert slot["target"]["content_sha256"] == canonical_sha256(target_body)
        assert slot["content_sha256"] == canonical_sha256(slot_body)


def test_every_slot_has_exact_proof_or_nonzero_exact_counterexample(checked: dict) -> None:
    for slot in checked["registered_slots"]:
        evaluation = slot["evaluation"]
        if slot["artifact_type"] == "theorem_rediscovery":
            assert evaluation["outcome"] == "PASS"
            assert evaluation["counterexample"] is None
            assert evaluation["exact_proof"]["coefficient_residual"] == 0
            assert evaluation["exact_proof"]["decision"] == (
                "proved_exact_template_identity_for_all_integer_inputs"
            )
        else:
            assert evaluation["outcome"] == "REJECT"
            assert evaluation["exact_proof"] is None
            assert evaluation["counterexample"]["residual"] != 0
            assert (
                evaluation["counterexample"]["candidate_value"]
                != evaluation["counterexample"]["target_value"]
            )


def test_historical_lineage_is_explicit_and_false_neighbors_are_honest(checked: dict) -> None:
    historical = [slot for slot in checked["registered_slots"] if slot["cohort"] == "historical"]
    assert len(historical) == 100
    assert all(slot["target"]["lineage"]["classical_source"] for slot in historical)
    assert all(slot["target"]["lineage"]["source_locator"] for slot in historical)
    false_neighbors = [
        slot for slot in historical if slot["artifact_type"] == "counterexample_construction"
    ]
    assert len(false_neighbors) == 50
    assert all(
        slot["candidate"]["candidate_origin"]
        == "generated_false_neighbor_control_not_historical_conjecture"
        for slot in false_neighbors
    )
    assert checked["claims"]["counterexample_controls_claimed_as_historical_conjectures"] is False


def test_thresholds_were_frozen_and_all_pass_honestly(checked: dict) -> None:
    assert checked["phase_a"]["thresholds"] == CONFIG["thresholds"]
    assert checked["phase_a"]["thresholds_sha256"] == canonical_sha256(CONFIG["thresholds"])
    evaluation = checked["threshold_evaluation"]
    assert evaluation["all_passed"] is True
    assert len(evaluation["rows"]) == 8
    assert all(row["passed"] for row in evaluation["rows"])


def test_partition_metrics_are_complete(checked: dict) -> None:
    partitions = checked["partitions"]
    assert {key: len(rows) for key, rows in partitions.items()} == {
        "artifact_type": 2,
        "cohort": 2,
        "domain": 5,
        "level": 5,
    }
    for rows in partitions.values():
        for row in rows:
            assert row["registered"] == row["ready"]
            assert row["missing"] == row["invalid"] == 0


def test_old_five_of_two_hundred_receipt_is_explicitly_superseded(checked: dict) -> None:
    authority = checked["authority"]
    assert authority["current_authority"] is True
    assert authority["current_authority_path"] == OUTPUT_PATH
    assert authority["contradictory_current_authorities"] == 0
    predecessor = authority["superseded_predecessor"]
    assert predecessor["path"] == OLD_OUTPUT_PATH
    assert predecessor["old_counts"] == {
        "invalid_slots": 0,
        "missing_slots": 195,
        "ready_slots": 5,
        "registered_slots": 200,
    }
    assert predecessor["supersession_status"] == (
        "superseded_historical_evidence_not_current_authority"
    )


def test_target_access_is_denied_before_one_atomic_unseal(
    checked: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = (ROOT / TARGET_PATH).resolve()
    original = Path.read_bytes
    calls = 0

    def audited(path: Path) -> bytes:
        nonlocal calls
        if path.resolve() == target:
            calls += 1
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", audited)
    assert build_curriculum(ROOT) == checked
    assert calls == 2
    assert checked["phase_a"]["target_access_enforcement"] == {
        "attempted": 1,
        "denied": 1,
        "exposed_bytes": 0,
    }
    assert checked["chronology"][3] == {
        "event": "one_atomic_target_rule_batch_unsealed",
        "target_reads": 1,
    }


def test_unseal_rejects_incomplete_or_resealed_phase_a(checked: dict) -> None:
    phase = copy.deepcopy(checked["phase_a"])
    phase["candidate_count"] = 199
    _reseal(phase)
    with pytest.raises(CompleteCurriculumError, match="before complete Phase A"):
        _unseal_target_rules(ROOT, CONFIG, phase)


def test_exact_replay_is_deterministic(checked: dict) -> None:
    assert build_curriculum(ROOT) == checked
    assert build_curriculum(ROOT) == build_curriculum(ROOT)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: value["counts"].__setitem__("ready", 199),
        lambda value: value["authority"].__setitem__("current_authority", False),
        lambda value: value["phase_a"].__setitem__("target_reads", 1),
        lambda value: value["registered_slots"][0]["candidate"].__setitem__(
            "candidate_coefficient", -1
        ),
        lambda value: value["registered_slots"][0]["evaluation"]["counterexample"].__setitem__(
            "residual", 0
        ),
        lambda value: value["registered_slots"][-1]["evaluation"]["exact_proof"].__setitem__(
            "coefficient_residual", 1
        ),
        lambda value: value["threshold_evaluation"].__setitem__("all_passed", False),
        lambda value: value.__setitem__("unknown", True),
    ],
)
def test_resealed_tamper_fails_exact_replay(checked: dict, mutator) -> None:
    tampered = copy.deepcopy(checked)
    mutator(tampered)
    _reseal(tampered)
    with pytest.raises(CompleteCurriculumError):
        validate_curriculum(tampered, root=ROOT)


def test_target_commitment_and_threshold_config_tamper_fail_closed(tmp_path: Path) -> None:
    target_tamper = copy.deepcopy(CONFIG)
    target_tamper["target_fixture"]["content_sha256"] = "0" * 64
    target_path = tmp_path / "target-tamper.json"
    target_path.write_text(json.dumps(target_tamper), encoding="utf-8")
    with pytest.raises(CompleteCurriculumError, match="commitment did not open"):
        build_curriculum(ROOT, target_path)

    threshold_tamper = copy.deepcopy(CONFIG)
    threshold_tamper["thresholds"]["min_ready"] = 199
    threshold_path = tmp_path / "threshold-tamper.json"
    threshold_path.write_text(json.dumps(threshold_tamper), encoding="utf-8")
    with pytest.raises(CompleteCurriculumError, match="thresholds changed"):
        build_curriculum(ROOT, threshold_path)
