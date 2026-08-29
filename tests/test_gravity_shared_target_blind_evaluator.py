from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_shared_target_blind_evaluator import (
    CONTROL_IDS,
    DOMAINS,
    SharedTargetBlindEvaluatorError,
    _read_json,
    adapt_galaxy_rotation_rows,
    adapt_lensing_rows,
    build_candidate_registry,
    build_generation_packet,
    build_receipt,
    build_synthetic_controls,
    config_contract_sha256,
    evaluate_control_scalar,
    evaluate_control_vector,
    load_config,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gravity_shared_target_blind_evaluator_v1.json"


def test_generation_packet_is_predictor_only_and_bound_before_adapters() -> None:
    config = load_config(ROOT)
    packet = build_generation_packet(config)
    assert set(packet) == set(config["generation_contract"]["allowed_top_level_fields"])
    encoded = str(packet).lower()
    for token in config["generation_contract"]["forbidden_tokens"]:
        assert token.lower() not in encoded
    assert all(row["dimension"] == "1" for row in packet["typed_dimensionless_variables"])
    registry = build_candidate_registry(config, packet["content_sha256"])
    assert registry["generation_packet_sha256"] == packet["content_sha256"]
    assert registry["ben_structural_candidate"]["executed"] is False
    assert registry["ben_structural_candidate"]["outcome_scores_computed"] == 0


def test_top_level_config_seal_and_expression_registry_fail_closed() -> None:
    config = _read_json(CONFIG)
    changed = copy.deepcopy(config)
    changed["seed"] += 1
    with pytest.raises(SharedTargetBlindEvaluatorError, match="top-level config"):
        validate_config(ROOT, changed)

    changed = copy.deepcopy(config)
    changed["control_formulas"]["galaxy_rotation"]["known_law"] = "x_source"
    changed["contract_sha256"] = config_contract_sha256(changed)
    with pytest.raises(SharedTargetBlindEvaluatorError, match="control expression"):
        validate_config(ROOT, changed)

    changed = copy.deepcopy(config)
    changed["adapters"]["group_bridge"]["raw_predictor_fields"].append("leak")
    changed["contract_sha256"] = config_contract_sha256(changed)
    with pytest.raises(SharedTargetBlindEvaluatorError, match="adapter contract"):
        validate_config(ROOT, changed)


def test_lensing_and_galaxy_typed_slots_are_semantically_aligned() -> None:
    lensing = adapt_lensing_rows(
        {
            "potential_ratio_phi": [1.0],
            "impact_parameter_ratio": [2.0],
            "potential_ratio_psi": [3.0],
            "geometry_ratio": [4.0],
        }
    )
    assert lensing.tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert evaluate_control_scalar("lensing_metric", "known_law", lensing[0]) == 4.0
    assert evaluate_control_scalar("lensing_metric", "wrong_law", lensing[0]) == 2.0

    galaxy = adapt_galaxy_rotation_rows(
        {
            "rad_kpc": [1.0, 2.0],
            "vgas_km_s": [10.0, 10.0],
            "vdisk_km_s": [20.0, 0.0],
            "vbul_km_s": [0.0, 20.0],
        }
    )
    assert galaxy[0, 3] == pytest.approx(0.0)
    assert galaxy[1, 3] == pytest.approx(1.0)


def test_injected_controls_recover_and_scalar_vector_parity() -> None:
    config = load_config(ROOT)
    result = build_synthetic_controls(config)
    assert result["all_parity_pass"] is True
    assert result["all_recovery_pass"] is True
    assert result["all_wrong_law_controls_pass"] is True
    for domain in DOMAINS:
        x = np.asarray([[0.4, 0.7, 0.2, 0.5], [1.1, 1.4, 0.6, 0.3]], dtype=np.float64)
        for control_id in CONTROL_IDS:
            vector = evaluate_control_vector(domain, control_id, x)
            scalar = np.asarray([evaluate_control_scalar(domain, control_id, row) for row in x])
            assert np.allclose(vector, scalar, rtol=0.0, atol=1.0e-12)


def test_receipt_is_honest_about_real_science_and_data_seals() -> None:
    receipt = build_receipt(ROOT)
    assert receipt["decision"] == "PASS_BOUNDED_SCAFFOLD_CONTROLS_ONLY_CHILD_NOT_EXECUTED"
    assert receipt["execution_scope"]["four_scale_infrastructure_exercised"] is True
    assert receipt["execution_scope"]["real_scientific_cross_scale_evaluation_completed"] is False
    assert receipt["candidate_accounting"]["candidate_registry_raw_entries"] == 13
    assert receipt["candidate_accounting"]["scientific_formula_candidates_evaluated"] == 0
    assert receipt["architecture_binding"]["structural_child_executed"] is False
    assert receipt["two_stage_separation"]["response_fields_read_before_generation"] == []
    for binding in receipt["two_stage_separation"]["adapter_execution_bindings"].values():
        assert (
            binding["generation_packet_sha256"]
            == receipt["two_stage_separation"]["generation_packet_sha256"]
        )
    assert receipt["retrospective_exposed_adapter_smokes"]["sealed_rows_opened"] == 0
    assert (
        receipt["retrospective_exposed_adapter_smokes"]["group_and_lensing_empirical_scores_absent"]
        is True
    )
    domains = receipt["retrospective_exposed_adapter_smokes"]["domains"]
    assert domains["galaxy_rotation"]["raw_data_rows_read"] == 0
    assert domains["xcop_thermodynamic"]["objects_read"] == 8
    assert domains["xcop_thermodynamic"]["new_formula_scores_computed"] == 0
    assert domains["xcop_thermodynamic"]["predictor_rows_consumed_by_adapter"] > 0
    for name in ("group_bridge", "lensing_metric"):
        assert domains[name]["empirical_rows_read"] == 0
        assert domains[name]["empirical_score_key_present"] is False
        assert "score" not in domains[name]
        assert "scores" not in domains[name]
    assert receipt["counterexample_policy"]["single_counterexample_is_universal_veto"] is False
    assert receipt["compute_accounting"]["gpu_calls"] == 0
    assert receipt["compute_accounting"]["model_calls"] == 0
    assert receipt["compute_accounting"]["paid_calls"] == 0
