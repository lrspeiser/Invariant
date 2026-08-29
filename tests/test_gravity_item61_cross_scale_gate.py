from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item61_cross_scale_gate import (
    CONFIG_PATH,
    GravityItem61Error,
    _candidate_velocity,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_unbound_contract_has_one_unchanged_parameter_set() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    assert config["candidate"]["parameters"] == {"beta": 1.5}
    assert config["candidate"]["formula_refit_allowed"] is False
    assert config["candidate"]["scale_specific_parameter_allowed"] is False


def test_scale_specific_fit_and_single_counterexample_veto_are_rejected() -> None:
    retuned = deepcopy(_config())
    retuned["candidate"]["scale_specific_parameter_allowed"] = True
    with pytest.raises(GravityItem61Error, match="scale-specific"):
        validate_config(ROOT, retuned, require_bound=False)
    singleton = deepcopy(_config())
    singleton["counterexample_policy"]["single_counterexample_terminal"] = True
    with pytest.raises(GravityItem61Error, match="over-pruning"):
        validate_config(ROOT, singleton, require_bound=False)


def test_sealed_sparc_access_is_rejected() -> None:
    config = deepcopy(_config())
    config["populations"]["sealed_sparc_confirmation_rows_allowed"] = 1
    with pytest.raises(GravityItem61Error, match="sealed"):
        validate_config(ROOT, config, require_bound=False)


def test_candidate_velocity_is_positive_and_ordered() -> None:
    config59 = json.loads(
        (ROOT / "configs/gravity_item59_xcop_forward_observable_gate_v1.json").read_text(
            encoding="utf-8"
        )
    )
    radius = np.geomspace(0.5, 20.0, 20)
    vbar2 = np.linspace(100.0, 2500.0, len(radius))
    prediction = _candidate_velocity(radius, vbar2, config59, 1.5)
    assert prediction.shape == radius.shape
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > np.sqrt(vbar2))
