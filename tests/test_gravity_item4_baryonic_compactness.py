from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item4_baryonic_compactness as item4
from sigma_theory_compiler import gravity_item4_baryonic_compactness_experiment as experiment
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict) -> dict:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_config_binds_item3_and_keeps_confirmation_closed() -> None:
    config = item4.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 4
    assert config["predecessor"]["required_decision"].endswith("ADVANCE_ITEM4")
    assert config["authorization"]["reserved_confirmation_member_rows_allowed"] is False


def test_compactness_builder_is_target_blind_and_scale_invariant_in_structure() -> None:
    config = item4.load_config(ROOT)
    ra = np.asarray([10.00, 10.01, 10.02, 10.03, 9.99, 9.98, 10.015, 9.985, 10.025, 9.975])
    dec = np.asarray([5.00, 5.01, 4.99, 5.02, 4.98, 5.015, 5.025, 4.975, 5.005, 4.995])
    light = np.arange(1.0, 11.0)
    first = item4.measure_compactness_only(ra, dec, light, 0.03, config)
    second = item4.measure_compactness_only(ra, dec, light * 3.0, 0.03, config)
    assert set(first) == set(second)
    assert np.isclose(first["log10_q_pair"], second["log10_q_pair"])
    assert np.isclose(first["log10_q_center"], second["log10_q_center"])
    assert "sigma" not in first
    assert "member_redshift" not in first


def test_mass_size_controls_are_declared_rewrites() -> None:
    config = item4.load_config(ROOT)
    assert config["scientific_contract"]["nonqualifying_rewrites"]
    models = {row["id"]: row for row in config["model_families"]}
    assert models["mass_size_rewrite"]["qualifying"] is False
    assert models["nuisance_plus_mass_size_rewrites"]["qualifying"] is False
    assert models["nuisance_plus_all_potential_structure"]["qualifying"] is True


def test_sample_manifest_is_target_blind_and_excludes_all_prior_groups() -> None:
    manifest = item4.build_sample_manifest(ROOT)
    assert manifest["counts"] == {
        "eligible_before_prior_exclusion": 523,
        "prior_group_ids_excluded": 450,
        "remaining": 73,
        "exploration": 52,
        "reserved_confirmation": 21,
    }
    assert all(value == 0 for value in manifest["selection_boundary"].values())
    assert not any(manifest["claims"].values())
    path = ROOT / item4.load_config(ROOT)["sample_manifest_output"]
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == manifest


def test_acquisition_is_exactly_the_exploration_partition() -> None:
    config = item4.load_config(ROOT)
    sample = _load(ROOT / config["sample_manifest_output"])
    source = _load(ROOT / config["source_manifest_output"])
    item4.validate_source_manifest(source, sample=sample)
    assert source["counts"] == {"bytes": 201028, "groups": 52, "member_rows": 912}
    assert source["preregistration"]["git_commit"] == item4.FREEZE_COMMIT
    assert source["boundary"] == {
        "exploration_groups_acquired": 52,
        "exploration_target_accesses": 52,
        "prior_group_target_reuse": 0,
        "published_group_velocity_columns_read": 0,
        "reserved_confirmation_groups_acquired": 0,
        "reserved_confirmation_target_accesses": 0,
    }
    selected = {int(row["group"]) for row in sample["objects"] if row["role"] == "exploration"}
    assert {int(row["group"]) for row in source["records"]} == selected


def test_extraction_retains_every_group_and_keeps_response_separate() -> None:
    config = item4.load_config(ROOT)
    summary = _load(ROOT / item4.EXTRACTION_SUMMARY_PATH)
    assert summary["decision"] == "PASS_ITEM4_EXPLORATION_REPRESENTATION_QUALITY"
    assert summary["counts"] == {
        "quality_failures": 0,
        "quality_passing": 52,
        "reserved_confirmation_target_accesses": 0,
        "selected_exploration": 52,
    }
    assert summary["leakage_boundary"] == {
        "compactness_finalized_before_response_function": True,
        "compactness_function_accepts_member_redshift": False,
        "published_group_velocity_columns_read": 0,
        "reserved_confirmation_target_accesses": 0,
    }
    rows = experiment._load_rows(ROOT, config)
    assert Counter(row["richness_stratum"] for row in rows) == {
        "10_14": 40,
        "15_plus": 12,
    }
    assert max(float(row["leave_one_out_gapper_fractional_range"]) for row in rows) <= 1


def test_outer_folds_hold_out_whole_groups_in_both_richness_strata() -> None:
    config = item4.load_config(ROOT)
    rows = experiment._load_rows(ROOT, config)
    assignments = experiment.fold_assignments(
        rows,
        salt=config["cross_validation"]["fold_salt"],
        folds=config["cross_validation"]["outer_folds"],
    )
    assert set(assignments) == {int(row["group"]) for row in rows}
    for fold in range(5):
        heldout = [row for row in rows if assignments[int(row["group"])] == fold]
        assert len(heldout) in {10, 11}
        assert {row["richness_stratum"] for row in heldout} == {"10_14", "15_plus"}


def test_receipt_replays_and_records_scoped_rejection() -> None:
    config = item4.load_config(ROOT)
    stored = _load(ROOT / config["output"])
    assert experiment.build_receipt(ROOT) == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "REJECT_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION"
    assert stored["counts"]["exploration_quality_passing"] == 52
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0
    assert float(stored["response"]["permutation_test"]["p_value"]) == 0.348
    assert sum(stored["gate_checks"].values()) == 4
    assert stored["gate_checks"]["reserved_confirmation_untouched"] is True
    assert stored["gate_checks"]["selected_model_qualifying_in_every_outer_fold"] is False


def test_resealed_false_pass_is_rejected() -> None:
    config = item4.load_config(ROOT)
    stored = copy.deepcopy(_load(ROOT / config["output"]))
    stored["decision"] = "PASS_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION_REQUIRES_AUTHORIZATION"
    with pytest.raises(experiment.GravityItem4CompactnessExperimentError):
        experiment.validate_receipt(_reseal(stored), root=ROOT)
