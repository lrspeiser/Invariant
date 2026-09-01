from __future__ import annotations

from pathlib import Path

import numpy as np

from sigma_theory_compiler import (
    open_gravity_differential_propagation_gw170817_coherent_v4 as module,
)

ROOT = Path(__file__).resolve().parents[1]


def test_blocked_config_and_holdout_boundary() -> None:
    config = module.load_config(ROOT)
    assert config["status"] == "BLOCKED_PRE_RESPONSE_END_TAPER_AUDIT"
    assert config["pre_response_blocker"]["coalescence_inside_end_taper"] is True
    assert config["freeze_boundary"]["gw190425_status"] == "SEALED_NOT_ACQUIRED_NOT_OPENED"


def test_transfer_gr_limits_and_reference_normalization() -> None:
    frequency = np.array([30.0, 100.0, 300.0])
    np.testing.assert_array_equal(
        module.transfer_function("GR", frequency, {}), np.ones(3, dtype=complex)
    )
    limits = {
        "DYNAMIC_PHASE": {"beta": 0.0},
        "ATTENUATION": {"alpha": 0.0},
        "RESERVOIR": {"r": 0.0, "log10_f_res_hz": 2.0},
        "NONLINEAR_PHASE": {"gamma": 0.0},
        "SCREENED_PHASE": {"beta_s": 0.0, "log10_f_screen_hz": 2.0},
    }
    for branch, parameters in limits.items():
        np.testing.assert_allclose(
            module.transfer_function(branch, frequency, parameters),
            np.ones(3, dtype=complex),
            atol=1.0e-14,
        )
        at_reference = module.transfer_function(
            branch,
            np.array([100.0]),
            {**parameters, **({"alpha": 0.2} if branch == "ATTENUATION" else {})},
        )
        np.testing.assert_allclose(at_reference, np.ones(1, dtype=complex), atol=1.0e-14)


def test_blocked_prediction_receipt_replays() -> None:
    decision = module.freeze(ROOT)
    assert decision == "BLOCKED_PRE_RESPONSE_END_TAPER_AUDIT_NO_INJECTIONS_NO_STRAIN"
    assert module.check(ROOT) == decision
