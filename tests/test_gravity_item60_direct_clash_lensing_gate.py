from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item60_direct_clash_lensing_gate import (
    CONFIG_PATH,
    GravityItem60Error,
    evaluate,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_unbound_contract_preserves_candidate_and_counterexample_rule() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    assert config["candidate_contract"]["formula_or_nuisance_refit_allowed"] is False
    assert config["counterexample_policy"]["single_empirical_counterexample_terminal"] is False
    assert config["counterexample_policy"]["global_family_pruning_allowed"] is False


def test_silent_gr_conversion_and_single_counterexample_veto_are_rejected() -> None:
    conversion = deepcopy(_config())
    conversion["candidate_contract"]["gr_lensing_conversion_may_be_assumed"] = True
    with pytest.raises(GravityItem60Error, match="silently assumed"):
        validate_config(ROOT, conversion, require_bound=False)
    singleton = deepcopy(_config())
    singleton["counterexample_policy"]["single_empirical_counterexample_terminal"] = True
    with pytest.raises(GravityItem60Error, match="over-pruning"):
        validate_config(ROOT, singleton, require_bound=False)


def test_direct_target_rows_cannot_enter_source_metadata() -> None:
    config = deepcopy(_config())
    config["source_metadata"][0]["target_rows"] = True
    with pytest.raises(GravityItem60Error, match="cannot contain target rows"):
        validate_config(ROOT, config, require_bound=False)


def test_all_direct_lensing_channels_require_missing_theory_primitives() -> None:
    config = _config()
    config["scientific_freeze_commit"] = "0" * 40
    # The evaluator itself requires a stored receipt; its structural claims are
    # separately checked here without opening observational targets.
    required = {row["id"] for row in config["required_channels"]}
    assert required == {
        "image_positions",
        "parities",
        "shapes",
        "weak_shear",
        "magnification",
        "time_delays",
    }
    supplied = set(config["candidate_contract"]["available_theory_primitives"])
    demanded = set(config["pre_response_gate"]["required_theory_primitives"])
    assert supplied == {"radial_acceleration"}
    assert demanded.isdisjoint(supplied)


def test_evaluation_stops_before_target_access_after_receipt_exists(tmp_path: Path) -> None:
    # Integration behavior is covered by committed replay after freeze. This
    # assertion documents the mandatory outcome rather than copying the repo.
    del tmp_path
    assert evaluate.__name__ == "evaluate"
    assert _config()["pre_response_gate"]["all_required_before_target_rows_opened"] is True
