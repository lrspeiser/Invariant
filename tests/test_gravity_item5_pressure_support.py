from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item5_pressure_support as item5
from sigma_theory_compiler import gravity_item5_pressure_support_experiment as experiment
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict) -> dict:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_config_binds_item4_and_keeps_confirmation_closed() -> None:
    config = item5.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 5
    assert config["predecessor"]["required_decision"].endswith("ADVANCE_ITEM5")
    assert config["authorization"]["reserved_confirmation_archive_members_allowed"] is False
    assert config["authorization"]["reserved_confirmation_target_queries_allowed"] is False


def test_sample_is_target_blind_and_excludes_prior_ddo154() -> None:
    manifest = item5.build_sample_manifest(ROOT)
    assert manifest["counts"] == {
        "archive_galaxies_excluding_alternative": 17,
        "fresh_candidates": 16,
        "exploration": 11,
        "reserved_confirmation": 5,
    }
    assert manifest["selection_boundary"]["archive_member_contents_read"] == 0
    assert manifest["selection_boundary"]["independent_pipeline_target_rows_read"] == 0
    assert not any(manifest["claims"].values())
    assert "ddo154" not in {row["galaxy"] for row in manifest["objects"]}
    assert "ddo216b" not in {row["galaxy"] for row in manifest["objects"]}
    stored_path = ROOT / item5.load_config(ROOT)["sample_manifest_output"]
    if stored_path.exists():
        assert json.loads(stored_path.read_text(encoding="utf-8")) == manifest


def test_creativity_boundary_distinguishes_classical_from_nonlocal() -> None:
    config = item5.load_config(ROOT)
    models = {row["id"]: row for row in config["model_families"]}
    assert models["classical_local_pressure"]["qualifying"] is False
    assert models["nonlocal_pressure_coherence"]["qualifying"] is True
    assert models["interior_pressure_memory"]["qualifying"] is True
    assert config["derivation"]["feature_builder_accepts_target"] is False


def test_support_builder_cannot_accept_either_target_curve() -> None:
    radius = np.geomspace(0.2, 2.0, 8)
    vrot = np.linspace(10.0, 25.0, 8)
    sigma = np.linspace(9.0, 7.0, 8)
    density = np.geomspace(15.0, 1.0, 8)
    features = item5.measure_support_only(radius, vrot, sigma, density)
    assert set(features) == {
        "log10_radius",
        "log10_vrot",
        "log10_v_classical",
        "local_pressure_fraction",
        "pressure_curvature",
        "nonlocal_pressure_fraction",
        "local_nonlocal_slope_difference",
        "memory_pressure_fraction",
        "local_memory_slope_difference",
    }
    assert all(np.all(np.isfinite(value)) for value in features.values())


def test_source_opens_only_exploration_members_and_targets() -> None:
    config = item5.load_config(ROOT)
    sample = _load(ROOT / config["sample_manifest_output"])
    source = _load(ROOT / config["source_manifest_output"])
    item5.validate_source_manifest(source, sample=sample)
    assert source["preregistration"]["git_commit"] == item5.FREEZE_COMMIT
    assert source["boundary"] == {
        "exploration_galaxies_acquired": 11,
        "exploration_predictor_member_accesses": 12,
        "exploration_target_query_accesses": 12,
        "reserved_confirmation_predictor_member_accesses": 0,
        "reserved_confirmation_target_accesses": 0,
        "published_Iorio_Vc_used_as_predictor": False,
    }
    confirmation_members = {
        row["archive_member"] for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    assert not confirmation_members.intersection(
        {row["predictor"]["archive_member"] for row in source["records"]}
    )


def test_extraction_retains_six_representation_failures_without_replacement() -> None:
    config = item5.load_config(ROOT)
    summary = _load(ROOT / config["extraction_summary_output"])
    assert summary["decision"] == "FAIL_ITEM5_EXPLORATION_REPRESENTATION_QUALITY"
    assert summary["counts"] == {
        "exploration_galaxies": 11,
        "quality_passing_galaxies": 5,
        "quality_failures": 6,
        "radial_rows": 66,
        "reserved_confirmation_predictor_member_accesses": 0,
        "reserved_confirmation_target_accesses": 0,
    }
    assert {row["reason"] for row in summary["failures"]} == {"nonpositive classical support speed"}
    assert len({row["galaxy"] for row in summary["failures"]}) == 6
    rows = experiment._load_rows(ROOT, config)
    assert len(rows) == 66
    assert len({row["galaxy"] for row in rows}) == 5


def test_outer_folds_hold_out_one_whole_galaxy_each() -> None:
    config = item5.load_config(ROOT)
    rows = experiment._load_rows(ROOT, config)
    assignments = experiment.fold_assignments(
        rows,
        salt=config["cross_validation"]["fold_salt"],
        folds=config["cross_validation"]["outer_folds"],
    )
    assert set(assignments) == {row["galaxy"] for row in rows}
    assert sorted(assignments.values()) == [0, 1, 2, 3, 4]


def test_receipt_replays_and_cannot_overrule_quality_gate() -> None:
    config = item5.load_config(ROOT)
    stored = _load(ROOT / config["output"])
    assert experiment.build_receipt(ROOT) == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM5_PRESSURE_SUPPORT_QUALITY_GATE"
    assert stored["counts"]["exploration_quality_passing"] == 5
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0
    assert stored["gate_checks"]["all_11_exploration_galaxies_pass_frozen_quality"] is False


def test_resealed_false_pass_is_rejected() -> None:
    config = item5.load_config(ROOT)
    stored = copy.deepcopy(_load(ROOT / config["output"]))
    stored["decision"] = "PASS_ITEM5_PRESSURE_SUPPORT_EXPLORATION_REQUIRES_AUTHORIZATION"
    with pytest.raises(experiment.GravityItem5PressureSupportExperimentError):
        experiment.validate_receipt(_reseal(stored), root=ROOT)
