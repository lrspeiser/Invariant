from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler import gravity_item4_baryonic_compactness as item4

ROOT = Path(__file__).resolve().parents[1]


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
