"""Controls for the counterexample-guided two-kernel G1 pilot repair."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g0_experiment import load_config as load_g0_config
from sigma_theory_compiler.gravity_g1_pilot_v2 import (
    KERNEL_COMPONENTS,
    OUTPUT_PATH,
    STRUCTURED_COMPONENTS,
    GravityG1PilotV2Error,
    build_receipt,
    load_config,
    random_pair_batches,
    replay_candidate,
    structured_pair_batches,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256
from sigma_theory_compiler.sparc_full_sample import assemble

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _population() -> Any:
    if "population" not in _CACHE:
        _CACHE["population"] = assemble(ROOT)
    return _CACHE["population"]


def _flatten(batches: Any) -> list[int]:
    return [int(value) for batch in batches for value in batch]


def test_v2_is_bound_to_the_sealed_v1_counterexample() -> None:
    config = _config()
    assert config["predecessor_binding"]["required_decision"] == (
        "BLOCK_G1_PILOT_UNCOVERED_OR_INCOMPLETE"
    )
    assert "30 million" in config["predecessor_binding"]["excluded_cell"]
    assert [item["arm"] for item in config["component_grammars"]] == [
        "structured_occam",
        "pseudorandom_permutation",
        "creativity_guided",
    ]
    assert config["candidate_shell"]["local_constants"][0]["dimension"] == (
        "km^2 s^-2 kpc^-1"
    )
    assert len(config["candidate_shell"]["local_constants"]) == 2


def test_structured_pairs_are_canonical_unique_and_complete_for_a_prefix() -> None:
    ordinals = _flatten(structured_pair_batches(100_000, 65_536))
    assert len(ordinals) == len(set(ordinals)) == 100_000
    assert all(value // STRUCTURED_COMPONENTS < value % STRUCTURED_COMPONENTS for value in ordinals)


@pytest.mark.parametrize(
    ("seed", "component_count"),
    [
        ("invariant-gravity-g1-pilot-v2-skew-kernel-pairs", KERNEL_COMPONENTS),
        ("invariant-gravity-g1-pilot-v2-claude-kernel-pairs", KERNEL_COMPONENTS),
    ],
)
def test_random_pair_prefix_is_collision_free_and_canonical(
    seed: str, component_count: int
) -> None:
    ordinals = _flatten(
        random_pair_batches(
            100_000, 65_536, component_count=component_count, seed=seed
        )
    )
    assert len(ordinals) == len(set(ordinals)) == 100_000
    assert all(value // component_count < value % component_count for value in ordinals)


def test_counterexample_guided_known_answer_clears_every_ugc06787_fold() -> None:
    galaxy = next(item for item in _population().exploration if item.name == "UGC06787")
    # Nearest frozen-grid form to the diagnostic pair found before v2 was declared:
    # broad log-RBF (mu~1.6,width=4,q=2) plus narrow (mu~2.3,width=1,q=2).
    ordinal = 944 * STRUCTURED_COMPONENTS + 1368
    result = replay_candidate(
        galaxy, "structured_occam", ordinal, _config(), load_g0_config(ROOT)
    )
    assert result["admitted"] is True
    assert result["failure_obligations"] == []
    assert len(result["folds"]) == 5
    assert all(all(row["checks"].values()) for row in result["folds"])
    assert result["description_length"]["local_constant_bits"] == 128


def test_small_cpu_v2_run_covers_no_confirmation_galaxy() -> None:
    receipt = build_receipt(ROOT, candidate_count=32, galaxy_limit=1, use_gpu=False)
    assert receipt["decision"] == "BLOCK_G1_PILOT_V2_UNCOVERED_OR_INCOMPLETE"
    assert receipt["counts"]["candidate_galaxy_trials"] == 96
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert len(receipt["galaxies"]) == 1


def test_checked_v2_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 v2 pilot has not run yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_v2_receipt_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G1 v2 pilot has not run yet")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1PilotV2Error, match="confirmation access"):
        validate_receipt(tampered, root=ROOT)
