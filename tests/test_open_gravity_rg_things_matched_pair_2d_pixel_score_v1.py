from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import open_gravity_rg_things_matched_pair_2d_pixel_score_v1 as score


@pytest.fixture(scope="module")
def config() -> dict:
    return score.load_config(verify_package=False)


@pytest.fixture(scope="module")
def receipt(config: dict) -> dict:
    return score.build_receipt(config)


def test_admission_rule_requires_real_data_papers_benchmarks_and_controls(config: dict) -> None:
    admission = config["admission_rule"]
    assert admission["real_public_response_data_required"] is True
    assert admission["primary_measurement_and_method_papers_required"] is True
    assert admission["target_free_operator_and_projection_benchmarks_required"] is True
    assert admission["newtonian_and_systemic_only_controls_required"] is True
    assert admission["missing_data_disposition"] == "SOURCE_BLOCKED"
    assert admission["paper_only_disposition"] == "THEORY_BENCHMARK_ONLY"
    assert admission["model_lifted_vertical_structure_disposition"] == "MODEL_LIFTED_2P5D"
    assert admission["one_failure_never_prunes_family"] is True


def test_prediction_packet_and_private_arrays_are_exactly_bound(config: dict) -> None:
    prediction_receipt, manifest = score._load_prediction_evidence(config)
    assert prediction_receipt["status"] == (
        "PASS_FIXED_RESPONSE_BLIND_MODEL_LIFTED_2P5D_PREDICTIONS"
    )
    assert prediction_receipt["scientific_boundary"]["velocity_pixel_values_decoded"] == 0
    assert manifest["array_count"] == 18
    assert manifest["array_root_sha256"] == config["prediction_binding"]["array_root_sha256"]


def test_rotation_sign_is_model_independent_known_answer() -> None:
    major = np.asarray([-2.0, -1.0, 1.0, 2.0])
    observed = np.asarray([30.0, 20.0, 0.0, -10.0])
    sign, covariance = score._rotation_sign(major, observed, np.ones(4, dtype=bool))
    assert sign == -1.0
    assert covariance < 0.0


def test_model_metric_recovers_offset_and_prefers_correct_sign() -> None:
    predicted = np.asarray([-20.0, -10.0, 10.0, 20.0])
    observed = predicted + 125.0
    metrics = score._model_metrics(
        observed,
        np.full(4, 8000.0),
        predicted,
        np.ones(4, dtype=bool),
        1.0,
        3000.0,
    )
    assert metrics["systemic_offset_m_s"] == pytest.approx(125.0)
    assert metrics["rmse_m_s"] == pytest.approx(0.0, abs=1.0e-13)
    assert metrics["opposite_sign_control"]["rmse_m_s"] > 0.0


def test_real_things_responses_are_scored_once_under_fixed_contract(receipt: dict) -> None:
    assert receipt["status"] == "PASS_FIXED_MATCHED_PAIR_REAL_THINGS_PIXEL_SCORE"
    assert len(receipt["objects"]) == 2
    for row in receipt["objects"]:
        assert row["response"]["common_pixel_count"] > 0
        assert row["response"]["beam_equivalent_count"] > 0.0
        assert row["rotation_sign"] in (-1.0, 1.0)
        assert set(row["models"]) == set(score._MODELS)
        for metrics in row["models"].values():
            assert metrics["rmse_m_s"] > 0.0
            assert abs(metrics["residual_mean_m_s"]) < 1.0e-8
            assert metrics["opposite_sign_control"]["rmse_m_s"] > 0.0
    assert receipt["scientific_boundary"]["response_array_slots_decoded"] == 4_194_304
    assert receipt["scientific_boundary"]["tuning_calls"] == 0


def test_ngc4214_caveat_and_failure_retention_are_preserved(receipt: dict) -> None:
    row = next(item for item in receipt["objects"] if item["object_id"] == "NGC4214")
    assert row["inference_role"] == "LOW_INCLINATION_STRESS_TEST_RETAINED"
    assert row["standard_rotation_curve_claim_allowed"] is False
    assert receipt["claim_boundary"]["ngc4214_standard_rotation_curve_valid"] is False
    assert receipt["claim_boundary"]["refracted_gravity_confirmed"] is False
    assert receipt["claim_boundary"]["publication_ready"] is False


def test_decision_is_exactly_recomputed_from_preregistered_gate(
    config: dict, receipt: dict
) -> None:
    primary = next(item for item in receipt["objects"] if item["object_id"] == "NGC2976")
    stress = next(item for item in receipt["objects"] if item["object_id"] == "NGC4214")
    threshold = config["interesting_signal_rule"][
        "minimum_primary_rmse_fractional_improvement_over_newton"
    ]
    assert receipt["interesting_signal"]["primary_gate_pass"] == (
        primary["rg_fractional_rmse_improvement_over_newton"] >= threshold
    )
    assert receipt["interesting_signal"]["stress_test_rg_beats_newton"] == stress["rg_beats_newton"]
    assert receipt["interesting_signal"]["theory_confirmation"] is False


def test_receipt_is_deterministically_self_hashed(receipt: dict) -> None:
    assert receipt["content_sha256"] == score.content_sha256({**receipt, "content_sha256": ""})


def test_config_mutations_fail_closed(config: dict) -> None:
    for path, value in (
        (("status",), "PUBLICATION_READY"),
        (("admission_rule", "real_public_response_data_required"), False),
        (("response_contract", "minimum_dispersion_scale_m_s"), 5000.0),
        (("score_contract", "per_model_sign_selection"), True),
        (("score_contract", "response_tuning_calls"), 1),
        (
            ("interesting_signal_rule", "minimum_primary_rmse_fractional_improvement_over_newton"),
            0.0,
        ),
        (("scientific_boundary", "general_3d_validated"), True),
        (("claim_boundary", "publication_ready"), True),
    ):
        mutated = copy.deepcopy(config)
        target = mutated
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(score.MatchedPairPixelScoreError):
            score.validate_config(mutated)


def test_receipt_mutation_fails(config: dict, receipt: dict) -> None:
    mutated = copy.deepcopy(receipt)
    mutated["claim_boundary"]["refracted_gravity_confirmed"] = True
    mutated["content_sha256"] = score.content_sha256({**mutated, "content_sha256": ""})
    with pytest.raises(score.MatchedPairPixelScoreError):
        score.validate_receipt_payload(config, mutated, receipt)


def test_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    assert score._atomic_no_clobber(output, b"one\n") == "CREATED"
    assert score._atomic_no_clobber(output, b"one\n") == "EXISTING_IDENTICAL"
    with pytest.raises(score.MatchedPairPixelScoreError):
        score._atomic_no_clobber(output, b"two\n")
