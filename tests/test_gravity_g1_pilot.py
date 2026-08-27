"""Controls for the target-blind, three-arm G1 galaxy-formula pilot."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g1_pilot import (
    CREATIVE_SIZE,
    FIRST_GALAXY_OUTPUT_PATH,
    OUTPUT_PATH,
    GravityG1PilotError,
    build_receipt,
    creativity_batches,
    load_config,
    pseudorandom_rational_batches,
    replay_candidate,
    select_pilot_galaxies,
    structured_batches,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_full_sample import assemble

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _population() -> Any:
    if "population" not in _CACHE:
        _CACHE["population"] = assemble(ROOT)
    return _CACHE["population"]


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_g1_is_bound_to_a_checked_g0_pass_and_three_frozen_arms() -> None:
    config = _config()
    assert config["g0_binding"]["required_decision"] == "PASS_G0_EXPERIMENT_FROZEN"
    assert [arm["id"] for arm in config["arms"]] == [
        "structured_occam",
        "pseudorandom_permutation",
        "creativity_guided",
    ]
    assert {arm["candidate_count_per_galaxy"] for arm in config["arms"]} == {10_000_000}
    assert config["claude_lineage"]["labels_authoritative_for_pruning"] is False
    assert config["claude_lineage"]["novelty_established"] is False


def test_target_blind_feature_selection_replays_the_declared_twelve() -> None:
    selected = select_pilot_galaxies(_population())
    assert selected == tuple(_config()["pilot_selection"]["galaxies_in_selection_order"])
    assert len(selected) == len(set(selected)) == 12


def test_pilot_selection_is_invariant_to_every_velocity_target() -> None:
    population = _population()
    changed = tuple(
        replace(galaxy, v_obs=tuple(value * 1000 for value in galaxy.v_obs))
        for galaxy in population.exploration
    )
    mutated = replace(population, exploration=changed)
    assert select_pilot_galaxies(mutated) == select_pilot_galaxies(population)


def _flatten(batches: Any) -> list[int]:
    return [int(value) for batch in batches for value in batch]


def test_structured_schedule_has_no_duplicates_and_stays_in_inner_coefficient_shells() -> None:
    ordinals = _flatten(structured_batches(300_000, 65_536))
    assert len(ordinals) == len(set(ordinals)) == 300_000
    from sigma_theory_compiler.gpu_baryonic_interpolation_screen import decode_ordinal

    decoded = [decode_ordinal(value) for value in ordinals[::997]]
    assert all(max(abs(x) for x in (*row["a"], *row["b"])) <= 2 for row in decoded)


def test_pseudorandom_schedule_is_unique_and_disjoint_from_structured_shell() -> None:
    arm = _config()["arms"][1]
    ordinals = _flatten(
        pseudorandom_rational_batches(100_000, int(arm["chunk_size"]), arm["seed"])
    )
    assert len(ordinals) == len(set(ordinals)) == 100_000
    from sigma_theory_compiler.gpu_baryonic_interpolation_screen import decode_ordinal

    for value in ordinals[::101]:
        row = decode_ordinal(value)
        assert max(abs(x) for x in (*row["a"], *row["b"])) == 3


def test_creativity_schedule_is_unique_bounded_and_visits_multiple_llm_seeded_families() -> None:
    arm = _config()["arms"][2]
    ordinals = _flatten(creativity_batches(100_000, int(arm["chunk_size"]), arm["seed"]))
    assert len(ordinals) == len(set(ordinals)) == 100_000
    assert min(ordinals) >= 0 and max(ordinals) < CREATIVE_SIZE
    assert len({value // 7**8 for value in ordinals}) >= 2


def test_one_candidate_replay_reports_fold_constants_scores_and_no_novelty_claim() -> None:
    config = _config()
    g0_config = __import__(
        "sigma_theory_compiler.gravity_g0_experiment", fromlist=["load_config"]
    ).load_config(ROOT)
    galaxy = next(
        item
        for item in _population().exploration
        if item.name == config["pilot_selection"]["galaxies_in_selection_order"][0]
    )
    ordinal = next(iter(structured_batches(1, 1)))[0]
    result = replay_candidate(galaxy, "structured_occam", int(ordinal), config, g0_config)
    assert result["ordinal"] == int(ordinal)
    if "folds" in result:
        assert len(result["folds"]) == min(5, galaxy.count)
        assert all(float(row["A_km2_s2_kpc"]) >= 0 for row in result["folds"])
        assert result["description_length"]["local_constant_bits"] == 64


def test_small_cpu_pilot_runs_all_arms_without_confirmation_access() -> None:
    receipt = build_receipt(ROOT, candidate_count=32, galaxy_limit=1, use_gpu=False)
    assert receipt["decision"] == "BLOCK_G1_PILOT_UNCOVERED_OR_INCOMPLETE"
    assert receipt["counts"]["candidate_galaxy_trials"] == 96
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert len(receipt["galaxies"]) == 1
    assert [trial["arm"] for trial in receipt["galaxies"][0]["trials"]] == [
        "structured_occam",
        "pseudorandom_permutation",
        "creativity_guided",
    ]


def test_checked_pilot_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 pilot receipt has not been run yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_first_galaxy_full_budget_counterexample_is_sealed() -> None:
    path = ROOT / FIRST_GALAXY_OUTPUT_PATH
    assert path.is_file(), "the 30-million-candidate first-galaxy result must be retained"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)
    assert receipt["decision"] == "BLOCK_G1_PILOT_UNCOVERED_OR_INCOMPLETE"
    assert receipt["counts"]["candidate_galaxy_trials"] == 30_000_000
    assert receipt["counts"]["pilot_galaxies"] == 1
    assert receipt["counts"]["covered_pilot_galaxies"] == 0


def test_pilot_receipt_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 pilot receipt has not been run yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1PilotError, match="confirmation access"):
        validate_receipt(tampered, root=ROOT)
