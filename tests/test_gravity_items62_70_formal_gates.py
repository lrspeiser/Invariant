from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_items62_70_formal_gates import (
    CONFIG_PATH,
    GravityItems62To70Error,
    _strong_field_gate,
    load_config,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_unbound_contract_covers_every_ordered_gate() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    assert config["items"] == list(range(62, 71))
    assert config["target"]["post_item59_refit_allowed"] is False


def test_single_empirical_veto_and_family_scope_overreach_are_rejected() -> None:
    singleton = deepcopy(_config())
    singleton["counterexample_policy"]["single_empirical_counterexample_terminal"] = True
    with pytest.raises(GravityItems62To70Error, match="counterexample policy"):
        validate_config(ROOT, singleton, require_bound=False)
    family = deepcopy(_config())
    family["counterexample_policy"][
        "hard_witness_may_prune_broader_family_without_family_scope_proof"
    ] = True
    with pytest.raises(GravityItems62To70Error, match="counterexample policy"):
        validate_config(ROOT, family, require_bound=False)


def test_strong_field_check_is_a_large_repeated_not_empirical_singleton() -> None:
    config = _config()
    config["scientific_freeze_commit"] = "0" * 40
    result = _strong_field_gate(config)
    audit = result["audit"]
    assert audit["all_checks_fail"] is True
    assert audit["high_acceleration_asymptotic_ratio"] == pytest.approx(2.5)
    assert len(audit["solar_radius_checks"]) == 4
    assert result["claims"]["boundary_or_nonlocal_family_pruned"] is False
    assert result["claims"]["single_empirical_counterexample_used_as_veto"] is False


def test_bound_config_loads_after_freeze_binding() -> None:
    config = _config()
    if config["scientific_freeze_commit"] == "PENDING_FREEZE_COMMIT":
        pytest.skip("freeze binding is intentionally pending")
    assert load_config(ROOT)["items"] == list(range(62, 71))
