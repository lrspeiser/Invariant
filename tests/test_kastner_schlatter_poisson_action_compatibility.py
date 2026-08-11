from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_poisson_action_compatibility import (
    _content_sha,
    _validate_result,
    build_audit,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_poisson_action_compatibility.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-poisson-action-compatibility.json"


def test_exact_rebuild_matches_checked_artifact() -> None:
    built = build_audit(CONFIG)
    checked = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert built == checked
    assert _content_sha(checked) == checked["content_sha256"]


def test_stationary_interface_closes_but_action_derivation_does_not() -> None:
    result = build_audit(CONFIG)
    assert result["counts"]["stationary_homogeneous_poisson_matches"] == 2
    assert result["counts"]["conditional_covariant_point_process_interfaces"] == 2
    assert result["counts"]["action_derived_point_process_measures"] == 0
    assert result["counts"]["positive_intensity_preservation_theorems"] == 0
    assert result["counts"]["qed_actualization_derivations"] == 0
    assert all(not item["beta_enters_count_law"] for item in result["branch_certificates"])


def test_exact_mixed_poisson_control_is_overdispersed() -> None:
    result = build_audit(CONFIG)
    control = result["exact_mixed_poisson_control"]
    assert control["E_N"] == "2"
    assert control["Var_N"] == "3"
    assert control["Fano_factor"] == "3/2"
    assert control["homogeneous_poisson_rejected_for_fluctuating_intensity"] is True
    assert result["counts"]["fluctuating_intensity_homogeneous_poisson_closures"] == 0


def test_all_scientific_and_data_claims_remain_sealed() -> None:
    result = build_audit(CONFIG)
    assert result["counts"]["observational_or_theory_passes"] == 0
    assert not any(result["claim_seals"].values())
    assert not any(result["data_seals"].values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("branch_certificates", 0, "probability_measure_derived_from_action"), True),
        (("branch_certificates", 1, "positive_intensity_preserved_by_scalar_dynamics"), True),
        (("branch_certificates", 0, "marginal_variance"), "Var(N(B))=E[mu_B]"),
        (("counts", "qed_actualization_derivations"), 1),
        (("claim_seals", "paper_transaction_ontology_validated"), True),
    ],
)
def test_rehashed_overclaim_tampering_rejected(path: tuple[object, ...], value: object) -> None:
    result = build_audit(CONFIG)
    mutated = copy.deepcopy(result)
    target = mutated
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    mutated["content_sha256"] = _content_sha(mutated)
    with pytest.raises(ValueError):
        _validate_result(mutated)
