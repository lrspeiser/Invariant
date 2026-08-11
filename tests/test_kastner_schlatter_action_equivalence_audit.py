from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_action_equivalence_audit import (
    _content_sha,
    _validate_result,
    build_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_action_equivalence_audit.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-action-equivalence-audit.json"


def test_exact_rebuild_matches_checked_artifact() -> None:
    built = build_audit(CONFIG)
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert built == checked
    assert _content_sha(checked) == checked["content_sha256"]


def test_two_branches_map_to_one_canonical_dynamic_class() -> None:
    result = build_audit(CONFIG)
    assert result["counts"]["canonical_dynamic_class_matches"] == 2
    assert result["counts"]["new_propagating_gravity_operator_classes"] == 0
    assert result["branch_comparison"]["same_propagating_operator_class"] is True
    assert result["branch_comparison"]["same_constant_vacuum_energy"] is False
    assert result["branch_comparison"]["vacuum_energy_ratio_beta_half_over_beta_quarter"] == "2"
    assert all(
        item["field_redefinition"] == "varphi=sqrt(B_q)*(q-q0)/Lambda_phi**2"
        for item in result["equivalence_certificates"]
    )


def test_constant_term_and_scientific_claims_remain_fail_closed() -> None:
    result = build_audit(CONFIG)
    assert result["counts"]["full_action_equalities_to_constant_free_control"] == 0
    assert result["counts"]["literature_novelty_claims"] == 0
    assert result["counts"]["observational_or_theory_passes"] == 0
    assert not any(result["claim_seals"].values())
    assert not any(result["data_seals"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("equivalence_certificates", 0, "full_action_equal_to_constant_free_control"), True),
        (("counts", "new_propagating_gravity_operator_classes"), 1),
        (("claim_seals", "literature_novelty_claimed"), True),
    ],
)
def test_rehashed_claim_tampering_rejected(path: tuple[object, ...], value: object) -> None:
    result = build_audit(CONFIG)
    mutated = copy.deepcopy(result)
    target = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    mutated["content_sha256"] = _content_sha(mutated)
    with pytest.raises(ValueError):
        _validate_result(mutated)
