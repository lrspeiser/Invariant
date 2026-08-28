"""Controls for the G4 effective auxiliary-action derivation."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g4_auxiliary_action_derivation import (
    OUTPUT_PATH,
    GravityG4AuxiliaryActionError,
    action_prediction2,
    load_config,
    symbolic_derivation,
    validate_receipt,
)
from sigma_theory_compiler.gravity_g4_nonlocal_profile_law_construction import (
    prepare_nonlocal_packets,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _packet() -> Any:
    if "packet" not in _CACHE:
        _CACHE["packet"] = prepare_nonlocal_packets(ROOT)[0]
    return _CACHE["packet"]


def test_action_contract_predicts_parent_constants_but_keeps_claims_bounded() -> None:
    config = _config()
    closure = config["dimensional_closure_hypothesis"]
    assert closure["effective_baryonic_support_dimension"] == 2
    assert float(closure["predicted_beta"]) == 0.5
    assert float(closure["predicted_log_radius_scale"]) == 0.25
    assert closure["post_v5_hypothesis"] is True
    assert config["inherited_unresolved_quantity"]["may_count_as_first_principles"] is False
    assert config["claim_boundaries"]["effective_radial_action_is_covariant_theory"] is False
    assert config["population"]["confirmation_evaluator_accesses_allowed"] == 0


def test_symbolic_action_variations_and_flux_identity_close_exactly() -> None:
    derivation = symbolic_derivation()
    assert derivation["all_exact_residuals_zero"] is True
    assert derivation["screened_euler_residual"] == "0"
    assert derivation["directed_constraint_residual"] == "0"
    assert derivation["flux_identity_residual"] == "0"
    assert derivation["dimension_closure"] == {"D": 2, "beta": "1/2", "ell": "1/4"}


def test_action_prediction_is_target_blind_and_dimension_derived() -> None:
    packet = _packet()
    expected = action_prediction2(packet, support_dimension=2)
    poisoned = copy.deepcopy(packet)
    poisoned["arrays"]["vobs"] = np.full_like(packet["arrays"]["vobs"], 1e99)
    poisoned["arrays"]["sigma"] = np.full_like(packet["arrays"]["sigma"], 1e-99)
    actual = action_prediction2(poisoned, support_dimension=2)
    assert actual["beta"] == 0.5
    assert actual["ell"] == 0.25
    for key in ("q", "chi", "psi", "prediction2"):
        np.testing.assert_array_equal(actual[key], expected[key])


def test_checked_action_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full auxiliary-action derivation has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_action_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full auxiliary-action derivation has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["covariant_action_derived"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG4AuxiliaryActionError, match="overstates covariance"):
        validate_receipt(tampered, root=ROOT)
