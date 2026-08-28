from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item5_pressure_cross_support as item5v2

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_attempt1_and_forbids_circular_mass_inputs() -> None:
    config = item5v2.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 5
    assert config["attempt_1"]["required_decision"].startswith("INCONCLUSIVE_")
    assert config["authorization"]["inferred_spt_mass_allowed_as_predictor"] is False
    assert config["authorization"]["sz_equivalent_velocity_dispersion_allowed"] is False
    thermal = config["sources"]["thermal_predictor"]
    assert not set(thermal["allowed_columns"]).intersection(thermal["forbidden_columns"])
    ruel = config["sources"]["robustness_responses"][1]
    assert "sigmaSPT" not in ruel["allowed_columns"]


def test_sample_has_44_exploration_and_18_response_blinded_confirmation() -> None:
    config = item5v2.load_config(ROOT)
    manifest = item5v2.build_sample_manifest(ROOT)
    item5v2.validate_sample_manifest(manifest, config)
    assert manifest["counts"] == {
        "metadata_overlap_candidates": 80,
        "quality_passing_candidates": 62,
        "exploration": 44,
        "reserved_confirmation": 18,
    }
    assert manifest["prefreeze_boundary"]["primary_velocity_response_values_read"] == 0
    assert manifest["prefreeze_boundary"]["robustness_velocity_response_values_read"] == 0
    assert manifest["prefreeze_boundary"]["reserved_confirmation_predictors_blinded"] is False
    assert manifest["prefreeze_boundary"]["reserved_confirmation_velocity_responses_blinded"]
    folds = [row["outer_fold"] for row in manifest["objects"] if row["role"] == "exploration"]
    assert sorted(folds.count(index) for index in range(5)) == [8, 9, 9, 9, 9]
    assert all(
        row["outer_fold"] is None for row in manifest["objects"] if row["role"] != "exploration"
    )


def test_feature_builder_is_target_blind_and_finite() -> None:
    signature = inspect.signature(item5v2.measure_pressure_features)
    assert "sigma" not in " ".join(signature.parameters)
    features = item5v2.measure_pressure_features(
        xi=8.49,
        theta_arcmin=0.5,
        ysz_1e6_arcmin2=82.0,
        e_ysz_1e6_arcmin2=12.0,
        redshift=0.7004,
    )
    assert set(features) == set(item5v2.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    assert features["coherence_squared"] >= 0


def test_creativity_boundary_makes_main_effects_nonqualifying() -> None:
    config = item5v2.load_config(ROOT)
    models = {row["id"]: row for row in config["model_families"]}
    assert models["self_similar_sz"]["qualifying"] is False
    assert models["raw_observable_nuisance"]["qualifying"] is False
    assert models["known_pressure_compactness"]["qualifying"] is False
    assert models["pressure_coherence_bridge"]["qualifying"] is True
    assert models["pressure_phase_balance"]["qualifying"] is True
    assert models["all_pressure_coherence"]["qualifying"] is True


def test_exact_source_query_cannot_silently_add_columns() -> None:
    url = item5v2._query_url(
        "J/ApJS/216/27/table4",
        columns=["SPT-CL", "xi", "theta", "YSZ", "e_YSZ", "z", "n_z"],
        constraint_name="SPT-CL",
        constraint_value="J0000-5748",
    )
    assert "M500c" not in url
    assert "sigma" not in url
    assert url.count("-out=") == 7


def test_synthetic_vizier_payload_parsers_use_only_frozen_fields() -> None:
    thermal = item5v2.parse_thermal_payload(
        b"# metadata\nJ0000-5748\t8.49\t0.50\t82\t12\t0.702\t+\n",
        cluster="0000-5748",
    )
    metadata = item5v2.parse_metadata_payload(
        b"# metadata\nSPT-CLJ0000-5748\t97\t56\t48\t8\t0.7004\t0.0011\t2,4\n",
        cluster="0000-5748",
    )
    response = item5v2.parse_primary_response_payload(
        b"# response\nSPT-CLJ0000-5748\t900\t100\n", cluster="0000-5748"
    )
    robust = item5v2.parse_robustness_payload(
        b"# response\nJ0000-5748\t910\t105\t895\t110\n",
        cluster="0000-5748",
        source_id="J/ApJS/227/3/table2",
    )
    assert thermal["ysz_1e6_arcmin2"] == 82
    assert metadata["n_members"] == 56
    assert response["sigma_km_s"] == 900
    assert [row["estimator"] for row in robust] == ["biweight", "gapper"]


def test_catalog_native_act_prefix_repair_does_not_change_sample_or_gates() -> None:
    config = item5v2.load_config(ROOT)
    audit = config["postfreeze_acquisition_audit"]
    assert audit["failure_cluster"] == "0232-5257"
    assert audit["primary_response_queries_issued"] == 8
    assert audit["reserved_confirmation_response_queries_issued"] == 0
    assert audit["formula_model_gate_or_sample_change"] is False
    assert item5v2._bayliss_identifier("0232-5257") == "ACT-CLJ0232-5257"
    assert item5v2._bayliss_identifier("0000-5748") == "SPT-CLJ0000-5748"
    parsed = item5v2.parse_metadata_payload(
        b"ACT-CLJ0232-5257\t40\t30\t25\t5\t0.55\t0.001\t1\n",
        cluster="0232-5257",
    )
    assert parsed["n_members"] == 30


def test_response_acquisition_is_impossible_before_freeze_binding() -> None:
    if item5v2.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(item5v2.GravityItem5PressureCrossSupportError, match="not bound"):
            item5v2.acquire_exploration(ROOT)


def test_stored_sample_matches_builder_after_it_is_written() -> None:
    config = item5v2.load_config(ROOT)
    path = ROOT / config["sample_manifest_output"]
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == item5v2.build_sample_manifest(ROOT)
