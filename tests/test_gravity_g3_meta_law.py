"""Controls for G3 whole-galaxy target-blind meta-law evaluation."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g0_experiment import load_config as load_g0_config
from sigma_theory_compiler.gravity_g1_pilot import _load_json
from sigma_theory_compiler.gravity_g3_meta_law import (
    OUTPUT_PATH,
    GravityG3MetaLawError,
    build_receipt,
    formula_basis,
    load_config,
    prepare_packets,
    project_formula,
    target_blind_matrix,
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


def _a0() -> float:
    config = load_g0_config(ROOT)
    return float(
        next(row for row in config["baselines"] if row["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )


def test_g3_is_whole_galaxy_target_blind_and_diagnostic_disclosed() -> None:
    config = _config()
    assert config["g2_binding"]["required_decision"] == "PASS_G2_EQUIVALENCE_COLLAPSE"
    assert config["whole_galaxy_cross_validation"]["unit"] == "whole galaxy"
    assert config["whole_galaxy_cross_validation"]["galaxy_id_available_to_model"] is False
    assert (
        config["whole_galaxy_cross_validation"][
            "heldout_vobs_available_to_model_or_formula_projection"
        ]
        is False
    )
    assert config["target_blind_inputs"]["total_model_features"] == 52
    assert config["diagnostic_disclosure"]["same_exploration_folds_inspected_during_model_development"] is True
    assert config["diagnostic_disclosure"]["result_is_independent_confirmation"] is False
    assert config["admission"]["confirmation_evaluator_accesses_allowed"] == 0


def test_model_features_do_not_change_when_vobs_and_error_are_poisoned() -> None:
    galaxy = assemble(ROOT).exploration[0]
    matrix, summary, _features = target_blind_matrix(galaxy, _a0())
    poisoned = replace(
        galaxy,
        v_obs=tuple(value * 1000 for value in galaxy.v_obs),
        e_v_obs=tuple(value * 1000 for value in galaxy.e_v_obs),
    )
    poisoned_matrix, poisoned_summary, _poisoned_features = target_blind_matrix(
        poisoned, _a0()
    )
    np.testing.assert_array_equal(matrix, poisoned_matrix)
    np.testing.assert_array_equal(summary, poisoned_summary)


def test_prepare_packets_has_139_galaxies_and_2720_points() -> None:
    packets = prepare_packets(ROOT, _config())
    assert len(packets) == 139
    assert sum(row["galaxy"].count for row in packets) == 2720
    assert all(row["model_matrix"].shape[1] == 52 for row in packets)


def test_projection_recovers_a_training_class_from_its_generated_target() -> None:
    packet = prepare_packets(ROOT, _config())[0]
    g2 = _load_json(ROOT / _config()["g2_binding"]["path"])
    for row in g2["structural_classes"]:
        base, columns = formula_basis(row["canonical_ir"], packet)
        gram = columns.T @ columns
        if float(np.linalg.det(gram)) > 1e-12 * float(gram[0, 0] * gram[1, 1]):
            break
    target = base + columns @ np.asarray([1.0, 1.0])
    assert np.all(target > 0)
    projected = project_formula(packet, target, [row])
    assert projected["class_id"] == row["class_id"]
    assert float(projected["normalized_projection_error"]) < 1e-20


def test_one_outer_fold_cannot_issue_full_g3_pass() -> None:
    receipt = build_receipt(ROOT, outer_fold_limit=1)
    assert receipt["decision"] == "BLOCK_G3_META_LAW"
    assert 0 < receipt["counts"]["predicted_galaxies"] < 139
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["claims"]["independent_confirmation_completed"] is False


def test_checked_g3_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G3 meta-law has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_checked_g3_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G3 meta-law has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["historical_novelty_established"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG3MetaLawError, match="overstates novelty"):
        validate_receipt(tampered, root=ROOT)
