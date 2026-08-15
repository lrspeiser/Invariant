from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_h_star_order_one_checkpointable_materializer import (
    C0_SCHEMA,
    HStarOrderOneMaterializerError,
    _atomic_write_immutable,
    _content_hash,
    _load_config,
    build_c0,
    build_final_result,
    build_plan,
    next_units,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/quartic_tc2_d4_h_star_order_one_checkpointable_materializer.json"
)
PLAN = ROOT / (
    "runs/physics-language/quartic-tc2-d4-h-star-order-one-checkpointable-materializer/plan.json"
)


def test_sealed_plan_is_path_free_and_exact(tmp_path: Path) -> None:
    plan = build_plan(ROOT, CONFIG, tmp_path / "caller-scratch")
    assert plan == json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["content_sha256"] == _content_hash(plan)
    assert plan["checkpoint_directory"] == "caller_owned_scratch"
    assert str(tmp_path) not in json.dumps(plan)
    assert [phase["units"] for phase in plan["phases"]] == [1, 4, 15, 1]
    assert plan["next_units"] == ["C0"]


def test_c0_then_resume_frontier_is_C1_without_cold_build(tmp_path: Path) -> None:
    config = _load_config(ROOT, CONFIG)
    scratch = tmp_path / "scratch"
    c0 = build_c0(ROOT, config)
    assert c0["schema_version"] == C0_SCHEMA
    _atomic_write_immutable(scratch / "c0-seals.json", c0, 1_000_000)
    units = next_units(ROOT, config, scratch)
    assert units == [f"C1_basis_{atom}" for atom in config["basis_jet_directions"]]


def test_immutable_checkpoint_conflict_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    first = {"schema_version": "control", "value": 1}
    second = {"schema_version": "control", "value": 2}
    _atomic_write_immutable(path, first, 10_000)
    _atomic_write_immutable(path, first, 10_000)
    with pytest.raises(HStarOrderOneMaterializerError, match="immutable checkpoint conflict"):
        _atomic_write_immutable(path, second, 10_000)


def test_incomplete_C1_C2_cannot_finalize(tmp_path: Path) -> None:
    with pytest.raises(HStarOrderOneMaterializerError, match="incomplete checkpoints"):
        build_final_result(ROOT, CONFIG, tmp_path / "empty")


def test_config_and_checkpoint_tamper_fail_closed(tmp_path: Path) -> None:
    document = json.loads(CONFIG.read_text(encoding="utf-8"))
    document["target"]["polarization_evaluations"] = 14
    document["content_sha256"] = _content_hash(document)
    tampered = tmp_path / "config.json"
    tampered.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HStarOrderOneMaterializerError, match="invalid H-star"):
        _load_config(ROOT, tampered)
    config = _load_config(ROOT, CONFIG)
    scratch = tmp_path / "scratch"
    c0 = build_c0(ROOT, config)
    _atomic_write_immutable(scratch / "c0-seals.json", c0, 1_000_000)
    broken = copy.deepcopy(c0)
    broken["basis_jet_directions"] = ["G_12"]
    (scratch / "c0-seals.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(HStarOrderOneMaterializerError, match="C0 checkpoint seal mismatch"):
        next_units(ROOT, config, scratch)
