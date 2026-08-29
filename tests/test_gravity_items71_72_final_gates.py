from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_items71_72_final_gates import (
    CONFIG_PATH,
    GravityItems71To72Error,
    _behavioral_witness,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_unbound_contract_requires_external_independence_and_scoped_novelty() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    assert config["item71"]["pass_requires_external_data_or_independent_implementation"]
    assert config["item72"]["absence_from_scoped_search_proves_global_novelty"] is False


def test_internal_replay_cannot_be_relabeled_external_confirmation() -> None:
    config = deepcopy(_config())
    config["item71"]["pass_requires_external_data_or_independent_implementation"] = False
    with pytest.raises(GravityItems71To72Error, match="confirmation boundary"):
        validate_config(ROOT, config, require_bound=False)


def test_single_counterexample_veto_and_novelty_override_are_rejected() -> None:
    singleton = deepcopy(_config())
    singleton["counterexample_policy"]["single_empirical_counterexample_terminal"] = True
    with pytest.raises(GravityItems71To72Error, match="counterexample"):
        validate_config(ROOT, singleton, require_bound=False)
    override = deepcopy(_config())
    override["counterexample_policy"]["novelty_label_may_override_failed_physics_gate"] = True
    with pytest.raises(GravityItems71To72Error, match="counterexample"):
        validate_config(ROOT, override, require_bound=False)


def test_behavioral_witness_excludes_only_purely_local_rewrites() -> None:
    config = _config()
    config["scientific_freeze_commit"] = "0" * 40
    witness = _behavioral_witness(ROOT, config)
    assert witness["witness_passed"] is True
    assert witness["candidate_fractional_difference"] > 0.001
    assert witness["local_formula_fractional_difference"] == 0.0
    assert "does_not_establish" in witness
