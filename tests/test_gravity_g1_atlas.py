"""Controls for the 139-galaxy, 100-million-candidate-per-galaxy G1 atlas."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g1_atlas import (
    OUTPUT_PATH,
    GravityG1AtlasError,
    build_atlas,
    load_config,
    run_galaxy,
    validate_atlas,
    validate_checkpoint,
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


def test_atlas_is_pilot_bound_and_budget_is_exactly_100_million_per_galaxy() -> None:
    config = _config()
    assert config["pilot_binding"]["required_decision"] == "PASS_G1_PILOT_UNION_12_OF_12"
    assert config["candidate_budget_per_galaxy"] == 100_000_000
    assert sum(row["candidate_count"] for row in config["segments"]) == 100_000_000
    assert config["population"] == {
        "galaxies": 139,
        "points": 2720,
        "confirmation_evaluator_accesses_allowed": 0,
        "order": "canonical admitted exploration order from sparc_full_sample.assemble",
    }


def test_small_cpu_checkpoint_covers_all_segments_and_no_confirmation() -> None:
    galaxy = _population().exploration[0]
    checkpoint = run_galaxy(
        ROOT, galaxy, _config(), candidate_count_override=4, use_gpu=False
    )
    validate_checkpoint(
        checkpoint, root=ROOT, config=_config(), expected_galaxy=galaxy.name
    )
    assert checkpoint["candidate_count"] == 12
    assert checkpoint["confirmation_evaluator_access_count"] == 0
    assert [row["segment_id"] for row in checkpoint["trials"]] == [
        "feature_rbf_all_pairs",
        "feature_skew_rbf_all_pairs",
        "creative_feature_pair_prefix",
    ]


def test_checkpoint_tamper_fails_closed_even_after_resealing() -> None:
    galaxy = _population().exploration[0]
    checkpoint = run_galaxy(
        ROOT, galaxy, _config(), candidate_count_override=4, use_gpu=False
    )
    tampered = copy.deepcopy(checkpoint)
    tampered["confirmation_evaluator_access_count"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1AtlasError, match="confirmation access"):
        validate_checkpoint(
            tampered, root=ROOT, config=_config(), expected_galaxy=galaxy.name
        )


def test_small_cpu_atlas_cannot_pass_the_full_run_gate() -> None:
    atlas = build_atlas(
        ROOT,
        galaxy_limit=1,
        candidate_count_override=4,
        use_gpu=False,
        persist_checkpoints=False,
    )
    assert atlas["decision"] == "BLOCK_G1_ATLAS_INCOMPLETE"
    assert atlas["counts"]["candidate_galaxy_trials"] == 12
    assert atlas["counts"]["confirmation_evaluator_accesses"] == 0


def test_checked_atlas_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full 139-galaxy atlas has not completed")
    atlas = json.loads(path.read_text(encoding="utf-8"))
    validate_atlas(atlas, root=ROOT)


def test_atlas_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full 139-galaxy atlas has not completed")
    atlas = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(atlas)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG1AtlasError, match="confirmation access"):
        validate_atlas(tampered, root=ROOT)
