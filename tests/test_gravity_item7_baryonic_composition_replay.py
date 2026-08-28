from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_item7_baryonic_composition_replay as replay

ROOT = Path(__file__).resolve().parents[1]


def test_config_binds_attempt1_and_forbids_response_shortcuts() -> None:
    config = replay.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 7
    assert config["attempt_1"]["required_decision"].startswith("INCONCLUSIVE_ITEM7_")
    assert config["prefreeze_audit"]["hi_width_values_read"] == 0
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["authorization"]["reserved_confirmation_hi_width_responses_allowed"] is False
    assert config["authorization"]["dark_matter_or_dynamical_mass_allowed_as_predictor"] is False
    assert config["authorization"]["hi_or_co_line_width_allowed_as_predictor"] is False
    assert config["authorization"]["lensing_mass_allowed_as_predictor"] is False


def test_sample_is_response_blind_balanced_and_sealed() -> None:
    config = replay.load_config(ROOT)
    manifest = replay.build_sample_manifest(ROOT)
    replay.validate_sample_manifest(manifest, config)
    assert manifest["counts"] == {
        "xcold_rows": 532,
        "good_hi_detection_metadata_rows": 499,
        "quality_overlap": 129,
        "exploration": 96,
        "reserved_confirmation": 33,
    }
    folds = [row["outer_fold"] for row in manifest["objects"] if row["role"] == "exploration"]
    assert sorted(folds.count(index) for index in range(5)) == [19, 19, 19, 19, 20]
    assert manifest["selection"]["selection_used_hi_width_response"] is False
    assert manifest["prefreeze_boundary"]["hi_width_values_read"] == 0
    assert manifest["prefreeze_boundary"]["reserved_confirmation_hi_widths_blinded"]


def test_feature_builder_is_target_blind_and_finite() -> None:
    signature = inspect.signature(replay.measure_replay_features)
    parameters = " ".join(signature.parameters).lower()
    assert "width" not in parameters
    assert "velocity" not in parameters
    features = replay.measure_replay_features(
        log_mstar=10.4,
        log_mhi=9.7,
        log_mmol=9.1,
        half_light_radius_kpc=3.9,
        concentration_index=2.8,
        inclination_deg=42.0,
        hi_signal_to_noise=8.3,
    )
    assert set(features) == set(replay.FEATURE_NAMES)
    assert all(np.isfinite(value) for value in features.values())
    assert sum(
        features[name]
        for name in ("stellar_fraction", "atomic_fraction", "molecular_fraction")
    ) == pytest.approx(1.0)
    assert 0.0 <= features["phase_entropy"] <= 1.0


def test_creativity_boundary_separates_known_scalings_from_replayed_interactions() -> None:
    models = {row["id"]: row for row in replay.load_config(ROOT)["model_families"]}
    assert models["btfr_fixed"]["qualifying"] is False
    assert models["newtonian_mass_size_fixed"]["qualifying"] is False
    assert models["phase_main_effects"]["qualifying"] is False
    assert models["replayed_phase_balance"]["qualifying"] is True
    assert models["replayed_phase_structure"]["qualifying"] is True


def test_predictor_queries_exclude_hi_and_co_widths() -> None:
    config = replay.load_config(ROOT)
    composition = config["sources"]["composition"]
    composition_url = replay._query_url(
        composition["catalog_id"], columns=composition["allowed_columns"]
    )
    hi_id_url = replay._query_url(
        config["sources"]["hi_releases"][0]["catalog_id"], columns=["GASS"]
    )
    for url in (composition_url, hi_id_url):
        assert "W50" not in url
        assert "WCO" not in url
        assert "velocity" not in url.lower()


def test_synthetic_catalog_parsers() -> None:
    composition = replay.parse_composition_catalog(
        b"3189\t123766\t42.0\t3.9\t10.4\t1\t9.1\t2.8\n"
    )
    assert composition[3189] == {
        "gass_id": 3189,
        "sdss_id": "123766",
        "inclination_deg": 42.0,
        "r50_kpc": 3.9,
        "log_mstar": 10.4,
        "co_detection_flag": 1,
        "log_mmol": 9.1,
        "concentration_index": 2.8,
    }
    assert replay.parse_id_catalog(b"3189\n3465\n") == {3189, 3465}
    response = replay.parse_hi_response(
        b"3189\t310.0\t8.0\t9.4\t9.7\t1\n", gass_id=3189, source_id="dr1"
    )
    assert response["width_corrected_km_s"] == 310.0
    assert response["quality"] == 1


def test_source_validator_rejects_broad_response_query() -> None:
    config = replay.load_config(ROOT)
    source = config["sources"]["hi_releases"][0]
    identifier = config["sample"]["exploration"][0]
    url = replay._query_url(
        source["catalog_id"],
        columns=["GASS", "W50c", "e_W50", "S/N", "logMHI", source["quality_column"]],
        constraint_name="GASS",
        constraint_value=str(identifier),
        max_rows=10,
    )
    manifest = replay._seal(
        {
            "preregistration": {"scientific_freeze_commit": replay.SCIENTIFIC_FREEZE_COMMIT},
            "records": [
                {
                    "gass_id": value,
                    "hi": {"source_id": source["id"]},
                    "response_retrieval": {"url": url.replace(f"GASS={identifier}", "GASS=%3E0")},
                }
                for value in config["sample"]["exploration"]
            ],
            "boundary": {
                "exploration_primary_response_rows": 96,
                "reserved_confirmation_primary_response_queries": 0,
                "phangs_confirmation_response_queries": 0,
                "dynamical_or_dark_mass_values_acquired": 0,
                "lensing_mass_values_acquired": 0,
                "paid_model_calls": 0,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )
    with pytest.raises(replay.GravityItem7CompositionReplayError, match="one-galaxy"):
        replay.validate_source_manifest(manifest, config)


def test_response_acquisition_is_impossible_before_freeze_binding() -> None:
    if replay.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(replay.GravityItem7CompositionReplayError, match="not bound"):
            replay.acquire_exploration(ROOT)


def test_stored_sample_matches_builder_after_it_is_written() -> None:
    config = replay.load_config(ROOT)
    path = ROOT / config["sample_manifest_output"]
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == replay.build_sample_manifest(ROOT)


def test_source_manifest_keeps_confirmations_and_shortcuts_closed() -> None:
    config = replay.load_config(ROOT)
    path = ROOT / config["source_manifest_output"]
    if not path.exists():
        pytest.skip("exploration source not acquired")
    source = json.loads(path.read_text(encoding="utf-8"))
    replay.validate_source_manifest(source, config)
    assert source["boundary"]["reserved_confirmation_primary_response_queries"] == 0
    assert source["boundary"]["phangs_confirmation_response_queries"] == 0
    assert source["boundary"]["dynamical_or_dark_mass_values_acquired"] == 0
    assert source["boundary"]["lensing_mass_values_acquired"] == 0


def test_receipt_replays_if_experiment_has_run() -> None:
    config = replay.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("experiment not run")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert replay.build_receipt(ROOT) == stored
    replay.validate_receipt(stored, root=ROOT)
    assert stored["counts"]["reserved_confirmation_target_accesses"] == 0
    assert stored["counts"]["phangs_confirmation_target_accesses"] == 0


def test_resealed_false_pass_is_rejected_if_experiment_has_run() -> None:
    config = replay.load_config(ROOT)
    path = ROOT / config["output"]
    if not path.exists():
        pytest.skip("experiment not run")
    stored = copy.deepcopy(json.loads(path.read_text(encoding="utf-8")))
    stored["decision"] = "PASS_ITEM7_COMPOSITION_REPLAY_REQUIRES_CONFIRMATION_AUTHORIZATION"
    with pytest.raises(replay.GravityItem7CompositionReplayError):
        replay.validate_receipt(replay._seal(stored), root=ROOT)
