from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_higher_h_star_checkpointable_materializer import (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    OUTPUT_PATH,
    HigherHStarMaterializerError,
    build_result,
    validate_result,
)

ROOT = Path(__file__).resolve().parents[1]


def test_all_45_higher_physical_h_star_packets_are_source_derived() -> None:
    result = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_result(result, ROOT)
    assert result["counts"]["H_star_plus_higher_packets"] == 45
    assert result["counts"]["zero_packets_exactly_derived"] == 45
    assert result["counts"]["authoritative_action_derivative_nonzero_entries"] == 0
    assert result["counts"]["symmetry_remainder_entries"] == 0
    assert {packet["Taylor_order"] for packet in result["packets"]} == {2, 3, 4}
    assert all(packet["factorial_normalization"] == f"1/{packet['Taylor_order']}!" for packet in result["packets"])


def test_higher_h_star_result_is_deterministic() -> None:
    assert build_result(ROOT, ROOT / CONFIG_PATH, ROOT / CHECKPOINT_PATH) == build_result(
        ROOT, ROOT / CONFIG_PATH, ROOT / CHECKPOINT_PATH
    )


def test_missing_checkpoint_names_exact_first_primitive(tmp_path: Path) -> None:
    copied = tmp_path / "checkpoints"
    shutil.copytree(ROOT / CHECKPOINT_PATH, copied)
    (copied / "subset_0.json").unlink()
    with pytest.raises(
        HigherHStarMaterializerError,
        match="first missing primitive: subset_0 H_star_plus Taylor order 2",
    ):
        build_result(ROOT, ROOT / CONFIG_PATH, copied)


def test_tampered_checkpoint_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "checkpoints"
    shutil.copytree(ROOT / CHECKPOINT_PATH, copied)
    path = copied / "subset_0.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["authoritative_action_derivative_nonzero_entries"] = 1
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(HigherHStarMaterializerError, match="checkpoint tamper"):
        build_result(ROOT, ROOT / CONFIG_PATH, copied)
