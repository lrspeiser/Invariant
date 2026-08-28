from __future__ import annotations

import copy
import inspect
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_item2_axes_group_geometry as groups
import sigma_theory_compiler.gravity_item2_axes_group_geometry_experiment as experiment
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / groups.SAMPLE_MANIFEST_PATH


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _reseal(value: dict[str, object]) -> dict[str, object]:
    value.pop("content_sha256", None)
    value["content_sha256"] = canonical_sha256(value)
    return value


def test_attempt_five_is_frozen_before_member_redshifts() -> None:
    config = groups.load_config(ROOT)
    assert config["status"] == "frozen_before_selected_member_redshift_access"
    assert config["authorization"]["selected_exploration_member_rows_allowed"] is True
    assert config["authorization"]["reserved_confirmation_member_rows_allowed"] is False
    assert config["authorization"]["published_group_velocity_columns_allowed"] is False
    assert config["target_blind_sample"]["reserved_confirmation_target_accesses_allowed"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False


def test_metadata_query_contains_no_published_dynamics_or_xray_target() -> None:
    config = groups.load_config(ROOT)
    source = config["catalog_sources"]
    assert source["metadata_allowed_columns"] == ["Group", "Nmemb", "zsp", "LR195", "D10"]
    assert set(source["metadata_forbidden_columns"]) == {
        "sigmaGAP",
        "e_sigmaGAP",
        "sigmaMAD",
        "R200c",
    }
    assert "tablec2" not in source["metadata_query_url"]
    assert "tablec3" not in source["metadata_query_url"]
    text = (ROOT / source["metadata_path"]).read_text(encoding="utf-8")
    for forbidden in source["metadata_forbidden_columns"]:
        assert f"#Column\t{forbidden}\t" not in text


def test_sample_is_deterministic_balanced_and_target_blind() -> None:
    config = groups.load_config(ROOT)
    manifest = _load(SAMPLE)
    groups.validate_sample_manifest(manifest, config=config)
    assert groups.build_sample_manifest(ROOT) == manifest
    assert manifest["selection_boundary"] == {
        "metadata_endpoint_queries": 1,
        "published_group_velocity_columns_read": 0,
        "selected_member_rows_opened": 0,
        "selected_member_redshifts_read": 0,
        "reserved_confirmation_target_accesses": 0,
        "xray_target_columns_read": 0,
    }
    objects = manifest["objects"]
    assert len(objects) == 270
    assert len({row["group"] for row in objects}) == 270
    assert Counter(row["role"] for row in objects) == {
        "exploration": 180,
        "reserved_confirmation": 90,
    }
    assert Counter((row["richness_bin"], row["role"]) for row in objects) == {
        (richness_bin, role): count
        for richness_bin in range(3)
        for role, count in (("exploration", 60), ("reserved_confirmation", 30))
    }


def test_selection_code_has_no_member_or_velocity_input() -> None:
    signature = inspect.signature(groups.eligible_metadata_rows)
    assert tuple(signature.parameters) == ("rows", "config")
    source = inspect.getsource(groups.build_sample_manifest)
    assert "member_query" not in source
    assert "sigmaGAP" not in source
    assert "sigmaMAD" not in source
    assert "R200c" not in source


@pytest.mark.parametrize(
    "claim",
    [
        "alternative_to_gr_established",
        "confirmation_opened",
        "group_finder_independence_established",
        "member_response_seen_during_selection",
        "roadmap_item_2_complete",
    ],
)
def test_resealed_sample_overclaim_is_rejected(claim: str) -> None:
    config = groups.load_config(ROOT)
    manifest = copy.deepcopy(_load(SAMPLE))
    manifest["claims"][claim] = True
    with pytest.raises(groups.GravityItem2AxesGroupError):
        groups.validate_sample_manifest(_reseal(manifest), config=config)


def test_config_admits_membership_provenance_limitation() -> None:
    config = groups.load_config(ROOT)
    limitations = " ".join(config["provenance_limitations"])
    assert "FoF" in limitations
    assert "Clean algorithm" in limitations
    assert "redshifts and mass-model assumptions" in limitations
    assert config["claim_boundaries"]["group_finder_independence_established"] is False


def test_member_identity_and_coordinate_schema_corrections_are_explicit() -> None:
    config = groups.load_config(ROOT)
    source = config["catalog_sources"]
    correction = source["member_coordinate_schema_correction"]
    assert correction["authorized_exploration_group"] == 1549
    assert correction["audit_queries"] == 2
    assert correction["scientific_contract_changed"] is False
    assert correction["sample_or_role_changed"] is False
    assert source["member_identity_schema"] == {
        "primary_member_key": "GalID",
        "SpecObjID_zero_is_missing_not_a_member_key": True,
        "sample_or_scientific_contract_changed": False,
    }


def test_acquisition_contains_only_frozen_exploration_groups() -> None:
    config = groups.load_config(ROOT)
    sample = _load(SAMPLE)
    manifest = _load(ROOT / groups.SOURCE_MANIFEST_PATH)
    groups.validate_source_manifest(manifest, config=config, sample=sample)
    assert manifest["decision"] == "PASS_EXPLORATION_MEMBER_SOURCE_ACQUISITION"
    assert manifest["boundary"] == {
        "exploration_groups_acquired": 180,
        "exploration_member_query_accesses": 180,
        "published_group_velocity_columns_read": 0,
        "reserved_confirmation_groups_acquired": 0,
        "reserved_confirmation_target_accesses": 0,
        "schema_audit_target_accesses": 2,
        "total_exploration_target_accesses": 182,
        "xray_target_columns_read": 0,
    }
    assert manifest["counts"]["member_rows"] == 4744
    assert {row["group"] for row in manifest["records"]} == {
        row["group"] for row in sample["objects"] if row["role"] == "exploration"
    }


def test_geometry_extractor_cannot_accept_member_redshifts() -> None:
    signature = inspect.signature(groups.measure_geometry_only)
    assert tuple(signature.parameters) == (
        "ra_deg",
        "dec_deg",
        "luminosity",
        "metadata_redshift",
        "config",
    )
    source = inspect.getsource(groups.measure_geometry_only)
    assert "member_redshift" not in source
    assert "velocity" not in source
    assert "sigma" not in source


def test_projected_geometry_is_rotation_and_reflection_invariant() -> None:
    x = np.asarray([-3.0, -2.0, -0.5, 1.0, 2.0, 4.0])
    y = np.asarray([0.1, 1.2, -0.7, 0.2, -1.0, 0.6])
    weights = np.asarray([1.0, 2.0, 1.5, 0.8, 2.2, 1.1])

    def invariant_values(x_value: np.ndarray, y_value: np.ndarray) -> tuple[float, ...]:
        measured = groups._weighted_geometry(x_value, y_value, weights)
        graph = groups._graph_features(
            x_value - measured["center_x"],
            y_value - measured["center_y"],
            measured["rms_radius"],
        )
        return (
            measured["axis_ratio"],
            measured["m2"],
            measured["m3"],
            measured["m4"],
            graph["mst_length_per_rms_radius"],
            graph["mst_diameter_efficiency"],
            graph["angular_gap_entropy"],
        )

    primary = invariant_values(x, y)
    angle = 0.73
    rotated = invariant_values(
        x * np.cos(angle) - y * np.sin(angle),
        x * np.sin(angle) + y * np.cos(angle),
    )
    reflected = invariant_values(-x, y)
    assert rotated == pytest.approx(primary, rel=0, abs=1.0e-12)
    assert reflected == pytest.approx(primary, rel=0, abs=1.0e-12)


def test_extraction_retains_all_frozen_quality_failures_without_replacement() -> None:
    summary = _load(ROOT / groups.EXTRACTION_SUMMARY_PATH)
    assert summary["decision"] == "FAIL_EXPLORATION_REPRESENTATION_QUALITY"
    assert summary["counts"] == {
        "quality_failures": 17,
        "quality_passing": 163,
        "reserved_confirmation_target_accesses": 0,
        "selected_exploration": 180,
    }
    assert len({row["group"] for row in summary["failures"]}) == 17
    assert {row["reason"] for row in summary["failures"]} == {
        "insufficient members in radial geometry split"
    }
    rows = experiment._load_feature_rows(ROOT, groups.load_config(ROOT))
    assert Counter(row["richness_bin"] for row in rows) == {0: 47, 1: 56, 2: 60}


def test_outer_folds_hold_out_whole_groups_and_retain_every_richness_bin() -> None:
    config = groups.load_config(ROOT)
    rows = experiment._load_feature_rows(ROOT, config)
    assignments = experiment.fold_assignments(
        rows,
        salt=config["cross_validation"]["fold_salt"],
        folds=config["cross_validation"]["outer_folds"],
    )
    assert set(assignments) == {row["group"] for row in rows}
    for fold in range(5):
        heldout = [row for row in rows if assignments[row["group"]] == fold]
        assert len(heldout) >= 32
        assert {row["richness_bin"] for row in heldout} == {0, 1, 2}


def test_receipt_replays_and_limits_the_positive_prediction_result() -> None:
    stored = _load(ROOT / experiment.OUTPUT_PATH)
    assert experiment.build_receipt(ROOT) == stored
    experiment.validate_receipt(stored, root=ROOT)
    assert stored["decision"] == "INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY_QUALITY_GATE"
    assert stored["counts"]["exploration_quality_passing"] == 163
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0
    assert float(stored["response"]["primary"]["selected_metrics"]["overall"]["r2"]) > 0.6
    assert (
        float(stored["response"]["permutation_test"]["p_value"])
        > float(groups.load_config(ROOT)["exploration_admission"]["stratified_permutation_p_must_be_at_most"])
    )
    assert stored["gate_checks"]["selected_model_qualifying_in_every_outer_fold"] is False
    assert stored["gate_checks"]["reserved_confirmation_untouched"] is True
    assert sum(stored["gate_checks"].values()) == 4


def test_resealed_false_pass_is_rejected() -> None:
    stored = copy.deepcopy(_load(ROOT / experiment.OUTPUT_PATH))
    stored["decision"] = "PASS_ITEM2_AXES_GROUP_GEOMETRY_EXPLORATION_REQUIRES_AUTHORIZATION"
    with pytest.raises(experiment.GravityItem2AxesGroupExperimentError):
        experiment.validate_receipt(_reseal(stored), root=ROOT)
