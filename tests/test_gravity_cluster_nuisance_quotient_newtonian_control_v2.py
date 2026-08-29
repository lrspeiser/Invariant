from __future__ import annotations

import json

import numpy as np
import pytest

from sigma_theory_compiler import (
    gravity_cluster_nuisance_quotient_newtonian_control_v2 as control,
)
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v3 as sbc_v3
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

CONFIG_SHA256 = "7ae3d8a0d2e67b3883da6c216183f90a352c0d6a48a3321529df1f732ba2fde4"
ROOT = control.ROOT
CONFIG = ROOT / control.CONFIG_PATH
PREDICTORS = ROOT / control.PREDICTOR_PATH
CONTROLS = ROOT / control.CONTROLS_PATH
SMOKE = ROOT / control.SMOKE_PATH
UNAUTHORIZED = ROOT / control.UNAUTHORIZED_PATH
AUTHORIZATION_CONTROLS = ROOT / control.AUTHORIZATION_CONTROLS_PATH
IMPLEMENTATION = ROOT / control.IMPLEMENTATION_RECEIPT_PATH


def load_contract() -> dict[str, object]:
    return control.load_contract(CONFIG, CONFIG_SHA256)


def test_contract_matches_exact_v3_kernel_prior_diagnostics_and_calls() -> None:
    contract = load_contract()
    assert contract["exact_primitive_priors"] == sampler.PRIMITIVE_PRIORS
    assert contract["structural_kernel"] == sbc_v3.CANDIDATE_INFERENCE
    assert contract["diagnostic_protocol"] == {
        "implementation": "canonical_rank_normalized_split_rhat_bulk_tail_ess",
        "rank_protocol": sbc_v3.RANK_PROTOCOL,
        "scientific_gates": sbc_v3.GATES,
    }
    assert contract["call_accounting"] == control.per_fit_call_accounting()
    assert control.per_fit_call_accounting()[
        "maximum_newtonian_control_likelihood_evaluations"
    ] == 233_504
    assert control.per_fit_call_accounting()[
        "maximum_paired_likelihood_evaluations"
    ] == 467_008


def test_strict_v3_adjudicator_not_raw_v3_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raw_check_forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("raw V3 check must not authorize Newtonian control")

    monkeypatch.setattr(sbc_v3, "check", raw_check_forbidden)
    checked = control.validate_strict_v3_adjudicator()
    assert checked["valid"] is True
    assert checked["v3_synthetic_sbc_passed"] is True
    assert checked["newtonian_control_unlock"] is True
    assert checked["candidate_production_unlock"] is False
    assert checked["scientific_claim_allowed"] is False


def test_predictors_are_synthetic_target_blind_rank_ten() -> None:
    contract = load_contract()
    packet = control.load_predictors(
        PREDICTORS, contract["predictor_packet"]["file_sha256"]
    )
    assert packet["row_count"] == 80
    assert packet["predictor_semantics"]["synthetic_only"] is True
    assert packet["predictor_semantics"]["target_fields_present"] is False
    assert packet["predictor_semantics"][
        "real_object_or_survey_labels_present"
    ] is False
    assert not control._contains_forbidden_key(packet["rows"])
    basis = np.asarray(
        [row["dimensionless_nuisance_basis"] for row in packet["rows"]]
    )
    assert basis.shape == (80, 10)
    assert np.linalg.matrix_rank(basis) == 10


def test_sufficient_statistic_is_deterministic_positive_definite() -> None:
    contract = load_contract()
    packet = control.load_predictors(
        PREDICTORS, contract["predictor_packet"]["file_sha256"]
    )
    prior = uncertainty.load_config(ROOT)
    left = control.sufficient_observation(packet, prior)
    right = control.sufficient_observation(packet, prior)
    for one, two in zip(left, right, strict=True):
        assert np.array_equal(one, two)
        assert np.all(np.isfinite(one))
    assert left[0].shape == (10,)
    assert left[1].shape == (10, 10)
    assert np.all(np.linalg.eigvalsh(left[1]) > 0.0)
    assert left[2].shape == (80,)


def test_strict_adjudicator_failure_precedes_authorization_and_predictors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = {"authorization": False, "contract": False, "predictors": False}

    def reject_adjudicator() -> dict[str, object]:
        raise RuntimeError("injected strict adjudicator rejection")

    def touch_authorization(*_args: object, **_kwargs: object) -> None:
        touched["authorization"] = True

    def touch_contract(*_args: object, **_kwargs: object) -> None:
        touched["contract"] = True

    def touch_predictors(*_args: object, **_kwargs: object) -> None:
        touched["predictors"] = True

    monkeypatch.setattr(control, "validate_strict_v3_adjudicator", reject_adjudicator)
    monkeypatch.setattr(control, "validate_authorization", touch_authorization)
    monkeypatch.setattr(control, "load_contract", touch_contract)
    monkeypatch.setattr(control, "load_predictors", touch_predictors)
    with pytest.raises(RuntimeError, match="strict adjudicator rejection"):
        control.execute_production(
            UNAUTHORIZED,
            control.file_sha256(UNAUTHORIZED),
            CONFIG,
            CONFIG_SHA256,
            ROOT / control.PRODUCTION_RESULT_PATH,
        )
    assert touched == {"authorization": False, "contract": False, "predictors": False}


def test_external_authorization_failure_precedes_contract_and_predictors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = {"contract": False, "predictors": False}
    monkeypatch.setattr(
        control, "validate_strict_v3_adjudicator", lambda: {"passed": True}
    )

    def touch_contract(*_args: object, **_kwargs: object) -> None:
        touched["contract"] = True

    def touch_predictors(*_args: object, **_kwargs: object) -> None:
        touched["predictors"] = True

    monkeypatch.setattr(control, "load_contract", touch_contract)
    monkeypatch.setattr(control, "load_predictors", touch_predictors)
    with pytest.raises(RuntimeError, match="externally unauthorized"):
        control.execute_production(
            UNAUTHORIZED,
            control.file_sha256(UNAUTHORIZED),
            CONFIG,
            CONFIG_SHA256,
            ROOT / control.PRODUCTION_RESULT_PATH,
        )
    assert touched == {"contract": False, "predictors": False}


def test_current_manifest_is_distinct_and_unauthorized() -> None:
    body = control.validate_unauthorized(
        UNAUTHORIZED, control.file_sha256(UNAUTHORIZED), CONFIG_SHA256
    )
    assert body["schema_version"] == control.UNAUTHORIZED_SCHEMA
    assert body["status"] == "external_approval_required"
    assert body["production_authorized"] is False
    assert body["external_approval"] is None
    assert body["run_request"] == control.run_request()
    assert body["run_request"]["matched_newtonian_control_runs"] == 1
    assert body["run_request"]["maximum_paid_external_cost_usd"] == 0.0


def test_bounded_controls_and_smoke_are_not_production() -> None:
    controls = json.loads(CONTROLS.read_text(encoding="utf-8"))
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    authorization = json.loads(AUTHORIZATION_CONTROLS.read_text(encoding="utf-8"))
    assert controls["passed"] is True
    assert controls["exact_v3_structural_kernel_controls"]["passed"] is True
    assert controls["full_matched_newtonian_run_launched"] is False
    assert controls["production_likelihood_evaluations"] == 0
    assert smoke["passed"] is True
    assert smoke["transport_blocks"] == [
        row["block_id"] for row in sbc_v3.TRANSPORT_BLOCKS
    ]
    assert smoke["likelihood_evaluations"] == 224
    assert smoke["full_matched_newtonian_run_launched"] is False
    assert smoke["production_likelihood_evaluations"] == 0
    assert authorization["passed"] is True
    assert authorization["current_production_authorized"] is False
    assert authorization["authorized_manifest_persisted"] is False
    assert authorization["production_runs"] == 0


def test_no_authorized_manifest_or_production_result_exists() -> None:
    assert not (ROOT / control.PRODUCTION_RESULT_PATH).exists()
    assert not (
        ROOT
        / control.ARTIFACT_DIR
        / "authorization-current-authorized-v2.json"
    ).exists()


def test_implementation_receipt_is_valid_and_stays_locked() -> None:
    checked = control.check(CONFIG, CONFIG_SHA256, IMPLEMENTATION)
    assert checked["valid"] is True
    assert checked["strict_v3_adjudicator_passed"] is True
    assert checked["external_approval_present"] is False
    assert checked["production_authorized"] is False
    assert checked["full_matched_newtonian_run_completed"] is False
    assert checked["maximum_requested_likelihood_evaluations"] == 233_504
    assert checked["maximum_paid_external_cost_usd"] == 0.0
