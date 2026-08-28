from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item5_pressure_support as item5

ROOT = Path(__file__).resolve().parents[1]


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
