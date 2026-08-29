from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import (
    gravity_cluster_nuisance_quotient_newtonian_control as control,
)
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler

CONFIG_SHA256 = "e8ac73bf8ee6f31b2fea7c00e5ff63554361088648d1ff11b85b0259d6468b8b"
SBC_SHA256 = "afb4aa64850fc6449272d8ac2174c7c6c7dfa20c319259c48c5cecbd11dcaa11"
ARTIFACT_DIR = control.ROOT / control.ARTIFACT_DIR
CONTROLS = ARTIFACT_DIR / "controls-sealed.json"
SMOKE = ARTIFACT_DIR / "bounded-smoke-sealed.npz"
UNAUTHORIZED = ARTIFACT_DIR / "authorization-current-unauthorized-sealed.json"
SBC_CONTROLS = ARTIFACT_DIR / "sbc-gate-controls-sealed.json"
IMPLEMENTATION = (
    control.ROOT / "runs/gravity/publication-readiness/"
    "nuisance-quotient-newtonian-control-implementation-v1-final.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_exactly_matches_candidate_mechanics_and_priors() -> None:
    contract = control.load_contract(control.ROOT / control.CONFIG_PATH, CONFIG_SHA256)
    assert contract["exact_primitive_priors"] == sampler.PRIMITIVE_PRIORS
    assert contract["production_settings"] == sampler.PRODUCTION_SETTINGS
    assert contract["smoke_settings"] == sampler.SMOKE_SETTINGS
    assert contract["diagnostic_validation"] == sampler.DIAGNOSTIC_VALIDATION
    assert contract["completion_thresholds"] == sampler.COMPLETION_THRESHOLDS
    assert contract["call_accounting"] == sampler.expected_call_accounting()
    assert contract["production_result_path"] == control.PRODUCTION_RESULT_PATH.as_posix()


def test_predictor_packet_is_target_blind_and_rank_ten() -> None:
    contract = control.load_contract(control.ROOT / control.CONFIG_PATH, CONFIG_SHA256)
    packet = control.load_predictor_packet(
        control.ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    assert packet["row_count"] == 80
    assert packet["predictor_semantics"]["target_fields_present"] is False
    assert packet["predictor_semantics"]["real_object_or_survey_labels_present"] is False
    assert not control._contains_forbidden_key(packet["rows"])
    basis = np.asarray([row["dimensionless_nuisance_basis"] for row in packet["rows"]])
    assert basis.shape == (80, 10)
    assert np.linalg.matrix_rank(basis) == 10


def test_identifiability_is_quotient_only_not_seventeen_primitives() -> None:
    contract = control.load_contract(control.ROOT / control.CONFIG_PATH, CONFIG_SHA256)
    packet = control.load_predictor_packet(
        control.ROOT / contract["predictor_packet"]["path"],
        contract["predictor_packet"]["file_sha256"],
    )
    result = control.identifiability_audit(packet, control.uncertainty.load_config(control.ROOT))
    assert result["passed"] is True
    assert result["primitive_to_observable_rank"] == 10
    assert result["primitive_null_dimensions"] == 7
    assert result["primitive_labels_separately_identified"] is False


def test_bounded_smoke_is_accounted_but_not_scientific_adjudication() -> None:
    controls, smoke = control.validate_bound_controls_and_smoke(CONFIG_SHA256, CONTROLS, SMOKE)
    assert controls["passed"] is True
    assert controls["real_target_rows_read"] == 0
    assert smoke["mode"] == "bounded_injected_newtonian_smoke"
    assert smoke["forward_call_accounting"]["total_forward_evaluations"] == 948
    assert smoke["forward_call_accounting"]["frozen_maximum_forward_evaluations"] == 2136
    assert smoke["all_mechanics_gates_passed"] is True
    assert smoke["all_coordinate_gates_passed"] is False
    assert smoke["production_passed"] is False
    assert smoke["authorization_consumed"] is None
    assert smoke["sbc_receipt_consumed"] is None
    assert smoke["claim_boundary"]["publication_readiness_changed"] is False
    assert (
        smoke["synthetic_truth_recovery"]["single_dataset_false_selection_rate_measured"] is False
    )
    assert all(
        smoke["data_boundary"][name] == 0
        for name in (
            "real_development_rows",
            "real_holdout_rows",
            "real_confirmation_rows",
            "real_independent_rows",
        )
    )


def test_failed_frozen_sbc_is_rejected_before_control_load(monkeypatch: pytest.MonkeyPatch) -> None:
    touched = {"contract": False, "predictors": False}

    def fail_contract(*_args: object, **_kwargs: object) -> dict[str, object]:
        touched["contract"] = True
        raise AssertionError("control contract must not load before SBC passes")

    def fail_predictors(*_args: object, **_kwargs: object) -> dict[str, object]:
        touched["predictors"] = True
        raise AssertionError("predictors must not load before SBC passes")

    monkeypatch.setattr(control, "load_contract", fail_contract)
    monkeypatch.setattr(control, "load_predictor_packet", fail_predictors)
    with pytest.raises(RuntimeError, match="did not pass frozen gates"):
        control.execute_production(
            control.ROOT / control.SBC_RECEIPT_PATH,
            SBC_SHA256,
            UNAUTHORIZED,
            _sha256(UNAUTHORIZED),
            control.ROOT / control.CONFIG_PATH,
            CONFIG_SHA256,
            control.ROOT / control.PRODUCTION_RESULT_PATH,
        )
    assert touched == {"contract": False, "predictors": False}
    assert not (control.ROOT / control.PRODUCTION_RESULT_PATH).exists()


def test_sbc_gate_control_records_retained_failure_without_unlocking() -> None:
    observed = json.loads(SBC_CONTROLS.read_text(encoding="utf-8"))
    assert observed == control.sbc_gate_controls()
    assert observed["passed"] is True
    assert observed["observed_sbc_receipt_sha256"] == SBC_SHA256
    assert observed["observed_sbc_status"] == "bounded_synthetic_sbc_failed_result_retained"
    assert observed["nonpassing_or_missing_sbc_receipt_rejected"] is True
    assert observed["newtonian_control_unlocked"] is False
    assert observed["predictor_rows_read_during_rejection"] == 0


def test_external_authorization_refusal_precedes_control_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = {"contract": False}

    monkeypatch.setattr(control, "validate_sbc_receipt", lambda *_args: {})

    def fail_contract(*_args: object, **_kwargs: object) -> dict[str, object]:
        touched["contract"] = True
        raise AssertionError("control contract must not load before authorization")

    monkeypatch.setattr(control, "load_contract", fail_contract)
    with pytest.raises(RuntimeError, match="externally unauthorized"):
        control.execute_production(
            control.ROOT / control.SBC_RECEIPT_PATH,
            SBC_SHA256,
            UNAUTHORIZED,
            _sha256(UNAUTHORIZED),
            control.ROOT / control.CONFIG_PATH,
            CONFIG_SHA256,
            control.ROOT / control.PRODUCTION_RESULT_PATH,
        )
    assert touched["contract"] is False
    assert not (control.ROOT / control.PRODUCTION_RESULT_PATH).exists()


def test_forged_passing_receipt_cannot_hide_failed_bounded_result() -> None:
    original = json.loads((control.ROOT / control.SBC_RECEIPT_PATH).read_text(encoding="utf-8"))
    forged = dict(original)
    forged["status"] = "bounded_synthetic_sbc_passed_not_candidate_production"
    forged["decision"] = "BOUNDED_SYNTHETIC_QUOTIENT_SBC_PASSED"
    forged["scenario_results"] = [{**row, "passed": True} for row in original["scenario_results"]]
    forged.pop("content_sha256", None)
    forged["content_sha256"] = control.sbc.content_sha256(forged)
    with tempfile.TemporaryDirectory(prefix="forged-sbc-", dir=ARTIFACT_DIR) as directory:
        path = Path(directory) / "forged.json"
        path.write_text(json.dumps(forged), encoding="utf-8")
        with pytest.raises(RuntimeError, match="does not bind a passing bounded synthetic result"):
            control.validate_sbc_receipt(path, _sha256(path))


def test_policy_mutation_is_fail_closed() -> None:
    body = json.loads((control.ROOT / control.CONFIG_PATH).read_text(encoding="utf-8"))
    body["recovery_thresholds"]["minimum_coordinates_with_truth_in_marginal_95_interval"] = 7
    with tempfile.TemporaryDirectory(prefix="newtonian-config-", dir=ARTIFACT_DIR) as directory:
        path = Path(directory) / "mutated.json"
        path.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(RuntimeError, match="matched frozen object changed"):
            control.load_contract(path, _sha256(path))


def test_implementation_receipt_is_locked_and_zero_real_data() -> None:
    checked = control.check_implementation(
        control.ROOT / control.CONFIG_PATH, CONFIG_SHA256, IMPLEMENTATION
    )
    assert checked["valid"] is True
    assert checked["sbc_passed"] is False
    assert checked["production_authorized"] is False
    assert checked["production_runs"] == 0
    assert checked["newtonian_control_unlocked"] is False
