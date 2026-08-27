"""Controls for the disclosed fixed-shrinkage G3 repair."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g3_meta_law_v2 import (
    OUTPUT_PATH,
    GravityG3FixedShrinkageError,
    build_receipt,
    load_config,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_g3_v2_is_fixed_post_diagnostic_and_confirmation_sealed() -> None:
    config = _config()
    assert config["predecessor_binding"]["required_decision"] == "BLOCK_G3_META_LAW"
    assert config["learned_residual"]["fixed_shrinkage"] == "0.3"
    assert config["learned_residual"]["n_estimators"] == 512
    assert (
        config["diagnostic_disclosure"][
            "fixed_shrinkage_selected_after_inspecting_g3_v1_outer_fold_results"
        ]
        is True
    )
    assert config["diagnostic_disclosure"]["result_is_independent_confirmation"] is False
    assert config["admission"]["confirmation_evaluator_accesses_allowed"] == 0


def test_one_outer_fold_cannot_issue_full_g3_v2_pass() -> None:
    receipt = build_receipt(ROOT, outer_fold_limit=1)
    assert receipt["decision"] == "BLOCK_G3_FIXED_SHRINKAGE"
    assert 0 < receipt["counts"]["predicted_galaxies"] < 139
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0
    assert receipt["claims"]["independent_confirmation_completed"] is False
    assert receipt["claims"]["g4_universal_law_authorized"] is False


def test_checked_g3_v2_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G3 fixed-shrinkage run has not completed")
    validate_receipt(json.loads(path.read_text(encoding="utf-8")), root=ROOT)


def test_checked_g3_v2_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G3 fixed-shrinkage run has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["independent_confirmation_completed"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG3FixedShrinkageError, match="overstates confirmation"):
        validate_receipt(tampered, root=ROOT)
