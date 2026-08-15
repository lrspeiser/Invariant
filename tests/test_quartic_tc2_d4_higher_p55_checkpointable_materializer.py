from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_higher_p55_checkpointable_materializer import (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    OUTPUT_PATH,
    HigherP55MaterializerError,
    _content_hash,
    build_result,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]


def test_exact_45_higher_p55_packets_and_atomic_manifest() -> None:
    result = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_result(result, ROOT)
    assert result["status"] == "pass_exact_45_higher_P55_packets_registered"
    assert result["counts"]["P55_higher_packets_registered"] == 45
    assert result["counts"]["exact_recurrence_nonzero_remainders"] == 0
    assert len(result["registered_P55_Taylor_orders_two_through_four_packets"]) == 45
    assert sorted(
        {packet["Taylor_order"] for packet in result["registered_P55_Taylor_orders_two_through_four_packets"]}
    ) == [2, 3, 4]
    family = next(
        row
        for row in result["required_symbolic_input_manifest"]
        if row["input_id"] == "polarized_P55_Taylor_packets"
    )
    assert family["registered_packets"] == 75
    assert family["registered_Taylor_orders"] == [0, 1, 2, 3, 4]
    assert family["missing_Taylor_orders"] == []
    assert result["counts"]["manifest_registered_after"] == 154


def test_result_replay_is_byte_deterministic() -> None:
    first = build_result(ROOT, ROOT / CONFIG_PATH, ROOT / CHECKPOINT_PATH)
    second = build_result(ROOT, ROOT / CONFIG_PATH, ROOT / CHECKPOINT_PATH)
    assert first == second
    assert first["content_sha256"] == _content_hash(first)


def test_missing_checkpoint_fails_before_partial_manifest_advance(tmp_path: Path) -> None:
    copied = tmp_path / "checkpoints"
    shutil.copytree(ROOT / CHECKPOINT_PATH, copied)
    (copied / "subset_0.json").unlink()
    with pytest.raises(
        HigherP55MaterializerError,
        match="first missing primitive: subset_0 P55 Taylor order 2",
    ):
        build_result(ROOT, ROOT / CONFIG_PATH, copied)


def test_tampered_checkpoint_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "checkpoints"
    shutil.copytree(ROOT / CHECKPOINT_PATH, copied)
    path = copied / "subset_0.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["exact_recurrence_nonzero_remainders"] = 1
    path.write_text(json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(HigherP55MaterializerError, match="checkpoint tamper"):
        build_result(ROOT, ROOT / CONFIG_PATH, copied)
