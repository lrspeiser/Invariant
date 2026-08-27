"""Controls for G2 structural and behavioral equivalence collapse."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.gravity_g1_pilot import _load_json
from sigma_theory_compiler.gravity_g2_equivalence import (
    OUTPUT_PATH,
    GravityG2EquivalenceError,
    adversarial_design,
    behavior_record,
    build_receipt,
    canonical_formula_ir,
    collect_survivors,
    load_config,
    structural_signature,
    validate_receipt,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def test_g2_is_bound_to_the_complete_g1_union_and_zero_confirmation() -> None:
    config = _config()
    assert config["g1_binding"]["required_decision"] == "PASS_G1_ATLAS_UNION_139_OF_139"
    assert config["g1_binding"]["required_survivor_count"] == 8_615
    assert config["g1_binding"]["required_galaxy_count"] == 139
    assert config["confirmation_evaluator_accesses_allowed"] == 0


def test_all_g1_survivors_are_collected_once() -> None:
    survivors = collect_survivors(ROOT, _config())
    assert len(survivors) == 8_615
    assert len({row["candidate_id"] for row in survivors}) == 8_615
    assert len({row["galaxy"] for row in survivors}) == 139


def test_structural_signature_ignores_component_order_and_llm_label() -> None:
    checkpoint = _load_json(ROOT / "runs/gravity/g1-atlas/checkpoints-v1/CamB.json")
    candidate = checkpoint["retained_pareto"][0]
    rewritten = copy.deepcopy(candidate)
    rewritten["components"] = list(reversed(rewritten["components"]))
    rewritten["components"][0]["llm_origin_assessment"] = "uncertain"
    rewritten["ordinal"] = -1
    rewritten["prediction_sha256"] = "different-fit-evidence"
    assert structural_signature(candidate) == structural_signature(rewritten)


def test_behavioral_signature_merges_a_known_orientation_redundancy() -> None:
    component = {
        "exponent": "2.000000000000e+00",
        "family": "localized_exponential",
        "feature": "log_y",
        "orientation": -1,
        "scale": "1.000000000000e+00",
    }
    second = {
        "center": "0.000000000000e+00",
        "family": "generalized_feature_rbf",
        "feature": "gas_fraction",
        "q": "2.000000000000e+00",
        "width": "1.000000000000e+00",
    }
    first_ir = {
        "base": "newtonian_baryons",
        "coefficient_model": "unordered_two_column_linear_span",
        "components": [component, second],
        "feature_normalization": "native_v3_feature",
        "output": "circular_velocity_squared",
    }
    changed = copy.deepcopy(first_ir)
    changed["components"][0]["orientation"] = 1
    design, probes = adversarial_design(257, 8)
    first_behavior = behavior_record(first_ir, design, probes, _config())
    second_behavior = behavior_record(changed, design, probes, _config())
    assert canonical_sha256(first_ir) != canonical_sha256(changed)
    assert first_behavior["signature"] == second_behavior["signature"]


def test_behavioral_signature_separates_changed_feature_identity() -> None:
    component = {
        "center": "0.000000000000e+00",
        "family": "generalized_feature_rbf",
        "feature": "log_y",
        "q": "2.000000000000e+00",
        "width": "1.000000000000e+00",
    }
    second = {
        "center": "0.000000000000e+00",
        "family": "generalized_feature_rbf",
        "feature": "gas_fraction",
        "q": "1.000000000000e+00",
        "width": "2.000000000000e+00",
    }
    first_ir = {
        "base": "newtonian_baryons",
        "coefficient_model": "unordered_two_column_linear_span",
        "components": [component, second],
        "feature_normalization": "native_v3_feature",
        "output": "circular_velocity_squared",
    }
    changed = copy.deepcopy(first_ir)
    changed["components"][0]["feature"] = "log_r_over_disk_peak"
    design, probes = adversarial_design(257, 8)
    assert behavior_record(first_ir, design, probes, _config())["signature"] != behavior_record(
        changed, design, probes, _config()
    )["signature"]


def test_limited_receipt_cannot_pass_but_mutation_controls_do() -> None:
    receipt = build_receipt(ROOT, survivor_limit=64)
    assert receipt["decision"] == "BLOCK_G2_EQUIVALENCE"
    controls = receipt["equivalence_validation"]["mutation_controls"]
    assert controls["positive_controls_pass"] is True
    assert controls["negative_controls_pass"] is True
    assert receipt["counts"]["confirmation_evaluator_accesses"] == 0


def test_checked_g2_receipt_is_sealed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G2 equivalence collapse has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)


def test_checked_g2_tamper_fails_closed_if_present() -> None:
    path = ROOT / OUTPUT_PATH
    if not path.is_file():
        pytest.skip("full G2 equivalence collapse has not completed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["claims"]["historical_novelty_established"] = True
    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG2EquivalenceError, match="overstates novelty"):
        validate_receipt(tampered, root=ROOT)


def test_canonical_formula_ir_marks_rar_normalization_separately() -> None:
    g1 = _load_json(ROOT / "runs/gravity/g1-atlas/repair-v3.json")
    ir = canonical_formula_ir(g1["repair"]["retained_pareto"][0])
    assert ir["base"] == "empirical_RAR"
    assert ir["feature_normalization"] == "within_galaxy_baryonic_minmax_to_minus1_plus1"
