from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_nonlinear_lift_characteristic_gate import (
    CHARACTERISTIC_BLOCKER,
    IFT_BLOCKER,
    build_future_aether_nonlinear_lift_characteristic_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_nonlinear_lift_characteristic_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-nonlinear-lift-characteristic-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-finite-amplitude-negative-seed-gate.json"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_nonlinear_lift_characteristic_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {
        CHARACTERISTIC_BLOCKER: 11,
        IFT_BLOCKER: 3,
    }
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0


def test_source_candidate_action_and_record_bindings_are_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_negative_seed_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_registered_seed_and_negative_family_characteristic_partition(rebuilt: dict) -> None:
    assert rebuilt["registered_seed_characteristic_crossing_count"] == 13
    assert rebuilt["negative_source_family_forced_characteristic_crossing_count"] == 11
    assert rebuilt["certified_negative_characteristic_free_amplitude_window_count"] == 2
    assert rebuilt["globally_noncharacteristic_candidate_count"] == 1
    assert rebuilt["regular_ADM_implicit_lift_prerequisite_pass_count"] == 3
    forced = 0
    windows = 0
    global_count = 0
    for item in rebuilt["candidate_records"]:
        certificate = item["nonlinear_lift_characteristic_certificate"]
        threshold = Fraction(certificate["negative_source_amplitude_threshold_squared"])
        assert threshold < 100
        if certificate["negative_source_family_forces_characteristic_crossing"]:
            forced += 1
            assert item["first_blocker"] == CHARACTERISTIC_BLOCKER
        elif certificate["certified_negative_characteristic_free_amplitude_window_exists"]:
            windows += 1
            amplitude = Fraction(certificate["adjusted_characteristic_free_amplitude_squared"])
            first = min(
                Fraction(value)
                for value in certificate["finite_characteristic_tilt_squared"].values()
            )
            assert threshold < amplitude < first
            assert item["first_blocker"] == IFT_BLOCKER
        else:
            global_count += 1
            assert certificate["globally_noncharacteristic_for_finite_unit_tilt"] is True
            assert item["first_blocker"] == IFT_BLOCKER
    assert (forced, windows, global_count) == (11, 2, 1)


def test_characteristic_shells_are_slicing_obstructions_not_theory_rejections(
    rebuilt: dict,
) -> None:
    for item in rebuilt["candidate_records"]:
        certificate = item["nonlinear_lift_characteristic_certificate"]
        for crossing in certificate["registered_seed_characteristic_crossings"].values():
            assert Fraction(crossing["tilt_squared"]) < 100
            assert crossing["exact_radius_squared"].startswith("1-(")
        assert certificate["full_nonlinear_constraint_solution_proven"] is False
        assert certificate["completed_boundary_sign_persistence_proven"] is False
        assert certificate["candidate_rejection_authorized"] is False
        assert item["decision"] == "blocked"
        assert item["formal_pass"] is False
        assert item["candidate_rejection_authorized"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["full_nonlinear_constraint_completion_count"] == 0
    assert rebuilt["completed_boundary_sign_persistence_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0
    assert rebuilt["bounded_nonlinear_lift_characteristic_gate_completed"] is True
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda config: config.update(
                data_eligibility={
                    **config["data_eligibility"],
                    "observational_data_opened": True,
                }
            ),
            "eligibility is open",
        ),
        (lambda config: config.update(observational_authorization=True), "opened observations"),
        (lambda config: config.update(external_paid_llm_calls=True), "enabled paid LLM calls"),
        (
            lambda config: config["budget"].update(registered_seed_amplitude_squared=99),
            "budget is not exact",
        ),
        (
            lambda config: config["source_negative_seed_artifact"].update(content_sha256="0" * 64),
            "content hash mismatch",
        ),
        (
            lambda config: config["campaign_implementation"].update(file_sha256="0" * 64),
            "file hash mismatch",
        ),
    ],
)
def test_open_seals_budget_and_hash_tampering_fail_closed(mutation, message: str) -> None:
    config = copy.deepcopy(_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        build_future_aether_nonlinear_lift_characteristic_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_negative_seed_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_nonlinear_lift_characteristic_gate(config, ROOT)
