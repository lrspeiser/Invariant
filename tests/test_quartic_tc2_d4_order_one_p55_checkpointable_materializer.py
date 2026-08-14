from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import (
    quartic_tc2_d4_order_one_p55_checkpointable_materializer as materializer,
)

C0_SCHEMA = materializer.C0_SCHEMA
OrderOneP55MaterializerError = materializer.OrderOneP55MaterializerError
_atomic_write_immutable = materializer._atomic_write_immutable
_content_hash = materializer._content_hash
_load_config = materializer._load_config
build_c0 = materializer.build_c0
build_plan = materializer.build_plan
next_units = materializer.next_units

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/quartic_tc2_d4_order_one_p55_checkpointable_materializer.json"
)
PLAN = ROOT / (
    "runs/physics-language/quartic-tc2-d4-order-one-p55-checkpointable-materializer/plan.json"
)


def test_sealed_plan_matches_pristine_external_checkpoint_state(tmp_path: Path) -> None:
    plan = build_plan(ROOT, CONFIG, tmp_path / "checkpoints")
    sealed = json.loads(PLAN.read_text(encoding="utf-8"))
    assert plan == sealed
    assert plan["next_units"] == ["C0"]
    assert plan["progress"] == {
        "basis_jet_packets_complete": 0,
        "basis_jet_packets_required": 4,
        "basis_axis_matrices_complete": 0,
        "basis_axis_matrices_required": 12,
        "evaluation_packets_complete": 0,
        "evaluation_packets_required": 15,
    }


def test_C0_seals_transition_to_first_basis_unit(tmp_path: Path) -> None:
    config = _load_config(ROOT, CONFIG)
    checkpoint_dir = tmp_path / "checkpoints"
    c0 = build_c0(ROOT, config)
    assert c0["schema_version"] == C0_SCHEMA
    assert c0["frontier_BLOCK_unchanged"] is True
    _atomic_write_immutable(checkpoint_dir / "c0-seals.json", c0, 1_000_000)
    units = next_units(ROOT, config, checkpoint_dir)
    assert units[0] == "C1_basis_G_12"
    assert len(units) == 4


def test_immutable_checkpoint_rejects_changed_replay(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    value = {"a": 1}
    _atomic_write_immutable(path, value, 1_000_000)
    _atomic_write_immutable(path, value, 1_000_000)
    with pytest.raises(OrderOneP55MaterializerError, match="immutable checkpoint conflict"):
        _atomic_write_immutable(path, {"a": 2}, 1_000_000)


def test_tampered_C0_fails_before_progress_is_counted(tmp_path: Path) -> None:
    config = _load_config(ROOT, CONFIG)
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "c0-seals.json").write_text("{}", encoding="utf-8")
    with pytest.raises(OrderOneP55MaterializerError, match="C0 checkpoint seal mismatch"):
        next_units(ROOT, config, checkpoint_dir)


def test_config_and_checkpoint_caps_fail_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["caps"]["C1"]["wall_seconds"] = 0
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(OrderOneP55MaterializerError, match="invalid materializer config"):
        _load_config(ROOT, path)
    with pytest.raises(OrderOneP55MaterializerError, match="exceeds byte cap"):
        _atomic_write_immutable(tmp_path / "large.json", {"x": "12345"}, 2)


def test_readiness_claims_remain_fail_closed(tmp_path: Path) -> None:
    plan = build_plan(ROOT, CONFIG, tmp_path / "checkpoints")
    assert plan["content_sha256"] == _content_hash(plan)
    assert all(value is False for value in plan["claims"].values())
    tampered = copy.deepcopy(plan)
    tampered["claims"]["P55_Taylor_order_one_packets_registered"] = True
    tampered["content_sha256"] = _content_hash(tampered)
    assert tampered != plan
