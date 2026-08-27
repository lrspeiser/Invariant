"""Controls for the target-blind G1 cross-feature interaction repair."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g0_experiment import load_config as load_g0_config
from sigma_theory_compiler.gravity_g1_atlas_repair_v3 import (
    COMPONENT_COUNT,
    OUTPUT_PATH,
    PAIR_COUNT,
    GravityG1AtlasInteractionRepairError,
    build_receipt,
    interaction_components,
    load_config,
    normalized_baryonic_features,
    pair_batches,
    replay_candidate,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_full_sample import assemble

ROOT = Path(__file__).resolve().parents[1]
KNOWN_EXPLORATION_PASS_ORDINAL = 2_893 * COMPONENT_COUNT + 3_217
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _galaxy() -> Any:
    if "galaxy" not in _CACHE:
        _CACHE["galaxy"] = next(
            galaxy for galaxy in assemble(ROOT).exploration if galaxy.name == "NGC2955"
        )
    return _CACHE["galaxy"]


def test_interaction_contract_discloses_exploration_driven_design() -> None:
    config = _config()
    assert config["repair_galaxies"] == ["NGC2955"]
    assert config["component_grammar"]["component_count"] == COMPONENT_COUNT
    assert config["component_grammar"]["candidate_count"] == PAIR_COUNT == 6_081_328
    assert config["diagnostic_disclosure"] == {
        "same_exploration_counterexample_used_to_design_grammar": True,
        "a_member_of_this_family_was_observed_to_pass_before_sealing": True,
        "sealed_run_is": "an exhaustive reproduction and accounting test, not independent confirmation",
        "confirmation_partition_remains_unopened": True,
    }
    assert config["candidate_shell"]["historical_novelty_established"] is False
    assert config["candidate_shell"]["proposal_reads_vobs"] is False
    assert config["candidate_shell"]["maximum_local_constants"] == 2


def test_pair_schedule_is_unique_and_canonical() -> None:
    ordinals = np.concatenate(list(pair_batches(10_000, 257)))
    first = ordinals // COMPONENT_COUNT
    second = ordinals % COMPONENT_COUNT
    assert len(ordinals) == 10_000
    assert len(np.unique(ordinals)) == 10_000
    assert np.all(first < second)


def test_component_grammar_has_exact_declared_count_and_known_interactions() -> None:
    g0 = load_g0_config(ROOT)
    a0 = float(
        next(row for row in g0["baselines"] if row["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    normalized = normalized_baryonic_features(np, _galaxy(), a0, np.float64)
    components, metadata = interaction_components(np, normalized, np.float64)
    assert components.shape == (COMPONENT_COUNT, _galaxy().count)
    assert np.all(np.isfinite(components))
    assert metadata[2_893] == {
        "family": "chebyshev_feature_product",
        "first_degree": 3,
        "first_feature": "log_r_over_disk_peak",
        "second_degree": 6,
        "second_feature": "mass_proxy_fraction",
    }
    assert metadata[3_217] == {
        "family": "chebyshev_feature_product",
        "first_degree": 3,
        "first_feature": "disk_fraction",
        "second_degree": 6,
        "second_feature": "mass_proxy_fraction",
    }


def test_disclosed_exploration_formula_replays_in_cpu_fp64() -> None:
    candidate = replay_candidate(
        _galaxy(), KNOWN_EXPLORATION_PASS_ORDINAL, load_g0_config(ROOT)
    )
    assert candidate["admitted"] is True
    assert candidate["origin_assessment"] == "new_combination_of_known_ideas"
    assert candidate["historical_novelty_established"] is False
    assert all(all(fold["checks"].values()) for fold in candidate["folds"])
    assert all(candidate["aggregate_checks"].values())


def test_small_cpu_run_cannot_issue_the_full_pass() -> None:
    receipt = build_receipt(ROOT, candidate_count_override=4, use_gpu=False)
    assert receipt["decision"] == "BLOCK_G1_INTERACTION_REPAIR"
    assert receipt["counts"]["new_interaction_candidate_galaxy_trials"] == 4
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["claims"]["historical_novelty_established"] is False
    assert receipt["claims"]["independent_confirmation_completed"] is False


def test_checked_interaction_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 interaction repair has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_checked_interaction_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 interaction repair has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1AtlasInteractionRepairError, match="confirmation access"):
        validate_receipt(tampered, root=ROOT)
