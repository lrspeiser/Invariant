from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_higher_k55_checkpointable_registration import (
    CHECKPOINT_PATH,
    CONFIG_PATH,
    OUTPUT_PATH,
    HigherK55RegistrationError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_higher_k55_family_blocks_atomically_at_exact_order_three_residual() -> None:
    campaign = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(campaign, ROOT)
    assert campaign["counts"]["complete_evaluation_checkpoints"] == 2
    assert campaign["counts"]["higher_K55_packets_computed_but_unregistered"] == 6
    assert campaign["counts"]["higher_K55_packets_registered"] == 0
    assert campaign["counts"]["Riesz_recurrence_nonzero_remainders_before_failure"] == 0
    assert campaign["counts"]["companion_inverse_recurrence_nonzero_remainders_before_failure"] == 0
    assert campaign["counts"]["failure_sphere_symmetry_remainder_entries"] == 0
    assert campaign["counts"]["failure_sphere_symmetrizer_remainder_entries"] == 120
    assert campaign["counts"]["manifest_registered_after"] == 154
    family = next(row for row in campaign["required_symbolic_input_manifest"] if row["input_id"] == "polarized_K55_Taylor_packets")
    assert family["registered_packets"] == 30
    assert family["registered_Taylor_orders"] == [0, 1]
    assert family["missing_Taylor_orders"] == [2, 3, 4]
    failure = campaign["failure_checkpoint"]
    assert failure["evaluation_id"] == "subset_2"
    assert failure["Taylor_order"] == 3
    assert failure["sphere_symmetrizer_residual"]["nonzero_polynomial_entries"] == 120


def test_higher_k55_replay_is_deterministic() -> None:
    assert build_campaign(ROOT, ROOT / CONFIG_PATH, ROOT / CHECKPOINT_PATH) == build_campaign(ROOT, ROOT / CONFIG_PATH, ROOT / CHECKPOINT_PATH)


def test_missing_checkpoint_fails_before_manifest_advance(tmp_path: Path) -> None:
    copied = tmp_path / "checkpoints"
    shutil.copytree(ROOT / CHECKPOINT_PATH, copied)
    (copied / "failures" / "subset_2.json").unlink()
    (copied / "subset_0.json").unlink()
    with pytest.raises(HigherK55RegistrationError, match="first missing primitive: subset_0 K55 Taylor order 2"):
        build_campaign(ROOT, ROOT / CONFIG_PATH, copied)


def test_tampered_checkpoint_is_rejected(tmp_path: Path) -> None:
    copied = tmp_path / "checkpoints"
    shutil.copytree(ROOT / CHECKPOINT_PATH, copied)
    path = copied / "failures" / "subset_2.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["sphere_symmetrizer_remainder_entries"] = 119
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(HigherK55RegistrationError, match="failure checkpoint tamper"):
        build_campaign(ROOT, ROOT / CONFIG_PATH, copied)
