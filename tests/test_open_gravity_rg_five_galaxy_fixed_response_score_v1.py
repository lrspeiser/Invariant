from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_rg_five_galaxy_fixed_response_score_v1 as score


@pytest.fixture(scope="module")
def config() -> dict:
    return score.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return score.build_receipt(config)


def test_builder_admission_rule_is_hard(config: dict) -> None:
    rule = config["builder_admission_rule"]
    assert rule["real_public_source_data_required"] is True
    assert rule["primary_measurement_or_data_release_paper_required"] is True
    assert rule["independent_analytic_manufactured_or_reference_benchmark_required"] is True
    assert rule["missing_source_disposition"] == "SOURCE_BLOCKED"
    assert rule["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY"
    assert rule["spherical_or_1d_data_validate_general_3d"] is False


def test_exact_five_objects_and_rows_are_frozen(config: dict) -> None:
    binding = config["response_binding"]
    assert binding["selected_objects"] == [
        "NGC2903",
        "NGC2976",
        "NGC3198",
        "NGC3521",
        "NGC4214",
    ]
    assert binding["container_galaxies_opened"] == 175
    assert binding["container_response_rows_opened"] == 3391
    assert binding["selected_rows_available"] == 159


def test_predecessor_fields_and_real_sources_pass(config: dict) -> None:
    predecessor = score.validate_predecessors(config)
    assert predecessor["all_object_gates_pass"] is True
    assert predecessor["decision"] == "READY_FOR_FIXED_HELD_SPARC_RESPONSE_SCORE"
    assert predecessor["scientific_boundary"]["response_files_opened"] == 0
    assert predecessor["claim_boundary"]["observational_fit_tested"] is False
    assert predecessor["claim_boundary"]["paper_and_real_source_anchored"] is True


def test_fixed_candidate_programs_are_finite(config: dict) -> None:
    newton = np.asarray([1.0e-13, 1.0e-11, 1.0e-9])
    refracted = np.asarray([2.0e-13, 2.0e-11, 2.0e-9])
    for candidate_id in config["candidate_contract"]["candidate_ids"]:
        values = score._candidate_acceleration(candidate_id, newton, refracted, 1.2e-10)
        assert np.all(np.isfinite(values) & (values > 0.0))


def test_radius_gate_cannot_use_velocity(config: dict) -> None:
    predecessor = score.validate_predecessors(config)
    object_row = score._object_field(predecessor, "NGC2903")
    before = score._radius_prediction(config, object_row, 7.5)
    after = score._radius_prediction(config, object_row, 7.5)
    assert before == after
    assert config["radius_gate"]["velocity_values_used_by_gate"] is False


def test_receipt_scores_all_objects_and_candidates(receipt: dict) -> None:
    assert [row["object_id"] for row in receipt["object_scores"]] == [
        "NGC2903",
        "NGC2976",
        "NGC3198",
        "NGC3521",
        "NGC4214",
    ]
    assert receipt["access_accounting"]["object_candidate_scores_computed"] == 20
    for row in receipt["object_scores"]:
        assert len(row["candidates"]) == 4
        assert row["eligibility"]["rows_scored_common"] >= 3
        assert row["eligibility"]["eligibility_used_velocity_values"] is False
        assert {value["rows_scored"] for value in row["candidates"].values()} == {
            row["eligibility"]["rows_scored_common"]
        }


def test_access_accounting_stays_development_only(receipt: dict) -> None:
    access = receipt["access_accounting"]
    assert access["sparc"]["container_galaxies_opened"] == 175
    assert access["sparc"]["container_response_rows_opened"] == 3391
    assert access["sparc"]["selected_rows_available"] == 159
    for key in (
        "confirmation_rows_opened",
        "independent_rows_opened",
        "group_rows_opened",
        "lensing_rows_opened",
        "network_calls",
        "model_calls",
        "paid_calls",
        "tuning_calls",
    ):
        assert access[key] == 0


def test_adjudication_retains_all_counterexamples(receipt: dict) -> None:
    adjudication = receipt["adjudication"]
    assert len(adjudication["object_comparisons"]) == 5
    assert adjudication["object_support_count"] + adjudication["object_counterexample_count"] == 5
    assert adjudication["family_eliminated"] is False
    assert receipt["claim_boundary"]["family_eliminated"] is False


def test_receipt_is_deterministic(config: dict, receipt: dict) -> None:
    assert score.build_receipt(config) == receipt
    expected = score.content_sha256({**receipt, "content_sha256": ""})
    assert receipt["content_sha256"] == expected


def test_config_mutations_fail_closed(config: dict) -> None:
    mutations = []
    for path, value in (
        (("status",), "CONFIRMED"),
        (("builder_admission_rule", "real_public_source_data_required"), False),
        (("response_binding", "selected_objects"), ["NGC2903"]),
        (("candidate_contract", "retuning_calls"), 1),
        (("radius_gate", "velocity_values_used_by_gate"), True),
        (("scoring_contract", "family_elimination_from_this_run"), True),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        mutations.append(mutated)
    for mutated in mutations:
        with pytest.raises(score.FixedResponseScoreError):
            score.validate_config(mutated)


def test_receipt_mutation_is_rejected(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["adjudication"]["family_eliminated"] = True
    mutated["content_sha256"] = score.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(score.FixedResponseScoreError):
        score.validate_receipt_payload(config, mutated)


def test_atomic_writer_is_no_clobber(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    assert score._atomic_no_clobber(path, b"one\n") == "CREATED"
    assert score._atomic_no_clobber(path, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(score.FixedResponseScoreError):
        score._atomic_no_clobber(path, b"two\n")
    assert path.read_bytes() == b"one\n"


def test_canonical_encoding_round_trips(receipt: dict) -> None:
    assert json.loads(score.canonical_bytes(receipt)) == receipt
