from __future__ import annotations

import copy
import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.future_aether_finite_amplitude_negative_seed_gate import (
    build_future_aether_finite_amplitude_negative_seed_gate,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/future_aether_finite_amplitude_negative_seed_gate.json"
ARTIFACT_PATH = ROOT / "runs/engine/future-aether-finite-amplitude-negative-seed-gate.json"
SOURCE_PATH = ROOT / "runs/engine/future-aether-weak-field-ae-constraint-gate.json"
BLOCKER = (
    "nonlinear_Einstein_Aether_constraint_lift_of_explicit_compact_negative_source_seed_"
    "with_sign_preserving_completed_boundary_energy"
)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rebuilt() -> dict:
    return build_future_aether_finite_amplitude_negative_seed_gate(_config(), ROOT)


def test_exact_partition_and_portable_artifact(rebuilt: dict) -> None:
    assert rebuilt == json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    body = {key: value for key, value in rebuilt.items() if key != "content_sha256"}
    assert rebuilt["content_sha256"] == _sha(body)
    assert rebuilt["candidate_count"] == 14
    assert rebuilt["decision_counts"] == {"blocked": 14}
    assert rebuilt["first_blocker_counts"] == {BLOCKER: 14}
    assert rebuilt["formal_pass_count"] == 0
    assert rebuilt["candidate_rejection_authorized_count"] == 0
    assert rebuilt["constraint_satisfying_negative_total_energy_datum_count"] == 0


def test_source_candidate_action_and_record_bindings_are_preserved(rebuilt: dict) -> None:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    expected = {item["candidate_id"]: item for item in source["candidate_records"]}
    assert len(expected) == 14
    for item in rebuilt["candidate_records"]:
        predecessor = expected[item["candidate_id"]]
        assert item["typed_action_ir_sha256"] == predecessor["typed_action_ir_sha256"]
        assert (
            item["action_density_equivalence_sha256"]
            == predecessor["action_density_equivalence_sha256"]
        )
        assert item["compilation_receipt_sha256"] == predecessor["compilation_receipt_sha256"]
        assert item["source_weak_field_record_sha256"] == predecessor["content_sha256"]
        assert item["exact_specialization"] == predecessor["exact_specialization"]


def test_exact_compact_seed_integrals_and_worst_candidate_bound(rebuilt: dict) -> None:
    control = rebuilt["symbolic_finite_amplitude_control"]
    assert control["compact_seed"] == {
        "inside_unit_ball": "A_i=10*(1-r^2)^4*delta_i1",
        "outside_unit_ball": "A_i=0",
        "regularity": "C^3_compact_support",
        "maximum_tilt_squared": "100",
        "asymptotic_Aether": "unit_normal_outside_compact_support",
    }
    energy = control["exact_static_source_monopole"]
    assert energy["I_grad"] == "262144*pi/255255"
    assert energy["I_axis"] == "262144*pi/765765"
    assert energy["I_acceleration"] == "8589934592*pi/148767396525"
    assert Fraction(energy["worst_negativity_threshold_t_squared"]) < 100
    assert energy["worst_energy_upper_bound"] == "-1699374940160*pi/541513323351"
    assert energy["strictly_negative_for_all_candidate_cells"] is True


def test_all_candidates_have_negative_source_and_frozen_constraint_completion(
    rebuilt: dict,
) -> None:
    assert rebuilt["compact_finite_amplitude_Aether_seed_count"] == 14
    assert rebuilt["exact_negative_static_source_monopole_count"] == 14
    assert rebuilt["frozen_source_linearized_constraint_completion_count"] == 14
    assert rebuilt["negative_linearized_completed_boundary_energy_coefficient_count"] == 14
    assert rebuilt["full_nonlinear_constraint_completion_count"] == 0
    assert rebuilt["sign_preserving_nonlinear_boundary_completion_count"] == 0
    assert sum(rebuilt["static_source_energy_upper_bound_over_pi_counts"].values()) == 14
    for item in rebuilt["candidate_records"]:
        certificate = item["finite_amplitude_negative_seed_certificate"]
        assert Fraction(certificate["static_source_energy_upper_bound_over_pi"]) < 0
        assert certificate["exact_static_source_monopole_negative"] is True
        assert certificate["compact_asymptotically_Euclidean_Aether_seed"] is True
        assert certificate["frozen_source_linearized_Hamiltonian_constraint_completed"] is True
        assert certificate["frozen_source_linearized_momentum_constraint_completed"] is True
        assert certificate["negative_linearized_completed_boundary_energy_coefficient"] is True


def test_nonlinear_overclaim_and_candidate_rejection_remain_fail_closed(rebuilt: dict) -> None:
    assert "not the full nonlinear" in rebuilt["symbolic_finite_amplitude_control"]["scope"]
    for item in rebuilt["candidate_records"]:
        certificate = item["finite_amplitude_negative_seed_certificate"]
        assert certificate["full_nonlinear_Einstein_Aether_constraint_solution_proven"] is False
        assert certificate["sign_preserving_nonlinear_boundary_completion_proven"] is False
        assert certificate["constraint_satisfying_negative_total_energy_datum_proven"] is False
        assert item["decision"] == "blocked"
        assert item["first_blocker"] == BLOCKER
        assert item["formal_pass"] is False
        assert item["candidate_rejection_authorized"] is False


def test_hash_provenance_and_all_seals(rebuilt: dict) -> None:
    assert rebuilt["bounded_finite_amplitude_negative_seed_gate_completed"] is True
    assert rebuilt["full_candidate_specific_formal_completion_claimed"] is False
    assert rebuilt["automatic_downstream_enqueue_performed"] is False
    assert rebuilt["observational_data_opened"] is False
    assert rebuilt["dark_matter_or_halo_inputs"] is False
    assert rebuilt["redshift_distance_inputs"] is False
    assert rebuilt["paid_llm_spend_usd"] == 0.0
    assert rebuilt["data_eligibility"] == _config()["data_eligibility"]
    provenance = rebuilt["provenance"]
    provenance_body = {key: value for key, value in provenance.items() if key != "binding_sha256"}
    assert provenance["binding_sha256"] == _sha(provenance_body)
    for item in rebuilt["candidate_records"]:
        body = {key: value for key, value in item.items() if key != "content_sha256"}
        assert item["content_sha256"] == _sha(body)
        record_provenance = item["provenance"]
        record_body = {
            key: value for key, value in record_provenance.items() if key != "binding_sha256"
        }
        assert record_provenance["binding_sha256"] == _sha(record_body)
        assert item["observational_data_opened"] is False
        assert item["solar_bundle_generated"] is False


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
        (
            lambda config: config.update(observational_authorization=True),
            "opened observations",
        ),
        (
            lambda config: config.update(external_paid_llm_calls=True),
            "enabled paid LLM calls",
        ),
        (
            lambda config: config["budget"].update(seed_amplitude_squared=99),
            "budget is not exact",
        ),
        (
            lambda config: config["source_weak_field_artifact"].update(content_sha256="0" * 64),
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
        build_future_aether_finite_amplitude_negative_seed_gate(config, ROOT)


def test_bound_paths_cannot_escape_repository() -> None:
    config = copy.deepcopy(_config())
    config["source_weak_field_artifact"]["path"] = "../outside.json"
    with pytest.raises(ValueError, match="path escapes repository"):
        build_future_aether_finite_amplitude_negative_seed_gate(config, ROOT)
