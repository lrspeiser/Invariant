from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Self

import pytest

from sigma_theory_compiler import gravity_item9_probes2_replay as replay

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_zero_tuning_cells_and_response_boundary() -> None:
    config = replay.load_config(ROOT)
    assert config["roadmap_binding"]["item_number"] == 9
    assert config["predecessor"]["required_decision"] == "REJECT_ITEM9_INTERIOR_EXTERIOR_EXPLORATION"
    assert len(config["frozen_cells"]) == 6
    assert len({row["candidate_id"] for row in config["frozen_cells"]}) == 6
    assert config["operator_grammar"]["candidate_selection_calls"] == 0
    assert config["operator_grammar"]["post_response_cells"] == 0
    assert config["authorization"]["rotation_curve_entry_payload_allowed_before_sample_freeze"] is False
    assert config["authorization"]["derived_mass_columns_allowed"] is False
    assert config["authorization"]["paid_model_calls_allowed"] is False
    assert config["prefreeze_audit"]["rotation_curve_rows_read"] == 0


def test_candidate_manifest_is_exact_attempt1_replay_without_selection() -> None:
    manifest = replay.build_candidate_manifest(ROOT)
    replay._validate_content_hash(manifest, "test candidate manifest")
    assert manifest["counts"] == {
        "atomic_formula_cells": 6,
        "ensemble_formula_cells": 2,
        "total_evaluated_formulas": 8,
        "candidate_selection_calls": 0,
        "post_response_formula_cells": 0,
        "response_rows_read": 0,
        "paid_model_calls": 0,
    }
    assert [row["role"] for row in manifest["cells"][:5]] == [
        f"attempt1_fold_{index}" for index in range(5)
    ]
    assert manifest["cells"][5]["exact_prior_focusing_cell"] is True
    assert manifest["ensembles"][0]["rule"] == "pointwise median log10 predicted speed"
    assert not any(row["historical_novelty_claimed"] for row in manifest["cells"])


def test_identity_normalization_is_conservative() -> None:
    assert replay.normalize_identity("UGC 1-234") == "UGC1234"
    assert replay.normalize_identity("ugc_1234") == "UGC1234"
    assert replay.normalize_identity("NGC\u00a0123") == "NGC123"
    assignments = replay._entry_assignments(
        ["RotationCurves/RC_UGC123_Source.csv"], ["UGC1", "UGC123"], "RC"
    )
    assert assignments["UGC1"] == []
    assert assignments["UGC123"] == ["RotationCurves/RC_UGC123_Source.csv"]
    assert replay._source_family(
        "probes2_files/RotationCurves/RC_UGC123_2015AJ....149..180O.csv", "UGC123"
    ) == "2015AJ....149..180O"


def test_metadata_parser_retains_only_allowlisted_columns(tmp_path: Path) -> None:
    config = replay.load_config(ROOT)
    header = config["metadata_allowlist"] + config["metadata_forbidden_columns"]
    values = ["UGC1", "10", "1", "0.5", "60"] + ["999"] * len(
        config["metadata_forbidden_columns"]
    )
    path = tmp_path / "metadata.csv"
    path.write_text(
        ",".join(header)
        + "\n"
        + ",".join(values)
        + "\n"
        + ",".join(values[:-1])
        + "\n",
        encoding="utf-8",
    )
    parsed_header, records = replay._metadata_records(path, config)
    assert parsed_header == header
    assert records == [dict(zip(config["metadata_allowlist"], values[:5]))] * 2
    assert not any(key in records[0] for key in config["metadata_forbidden_columns"])


def test_archive_and_response_access_are_impossible_before_bindings() -> None:
    assert "testzip" not in inspect.getsource(replay._zip_inventory)
    if replay.SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(replay.GravityItem9Probes2ReplayError, match="not bound"):
            replay.write_source_manifest(ROOT)
    if replay.SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        with pytest.raises(replay.GravityItem9Probes2ReplayError, match="not bound"):
            replay.extract_profiles(ROOT)


def test_source_etag_guard_treats_only_weak_prefix_as_transport_equivalent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Response:
        def __init__(self) -> None:
            self.headers = {"ETag": '"same"', "Last-Modified": "fixed"}

        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"data"

    monkeypatch.setattr(replay.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())
    receipt = replay._download_exact(
        "https://example.invalid/fixed", tmp_path / "x", 4, 'W/"same"'
    )
    assert receipt["etag"] == '"same"'
    with pytest.raises(replay.GravityItem9Probes2ReplayError, match="ETag changed"):
        replay._download_exact(
            "https://example.invalid/fixed", tmp_path / "y", 4, '"different"'
        )


def test_profile_parsers_follow_frozen_header_rules() -> None:
    config = replay.load_config(ROOT)
    light = (
        b"R,SB_r,SB_r_err\n"
        + b"".join(f"{index},{20 + index / 10},0.1\n".encode() for index in range(1, 31))
    )
    rows, light_receipt = replay._parse_light_profile(light, 10.0, 0.5, config)
    assert len(rows) == 30
    assert light_receipt["cumulative_light_fallback_used"] is True
    assert all(row["totmag"] is not None for row in rows)

    rotation = b"R_kpc,Vrot,Vrot_err\n1,40,3\n2,50,4\n"
    rc, rc_receipt = replay._parse_rotation_curve(rotation, 10.0, config)
    assert len(rc) == 2
    assert rc_receipt["radius_converted_from_kpc"] is True
    assert rc_receipt["velocity_rule"] == "published_deprojected_rotation_speed"


def test_formula_generator_has_no_response_parameter() -> None:
    signature = " ".join(inspect.signature(replay.build_candidate_manifest).parameters).lower()
    assert "response" not in signature
    assert "velocity" not in signature


def test_stored_prefreeze_artifacts_replay_if_present() -> None:
    config = replay.load_config(ROOT)
    candidate_path = ROOT / config["outputs"]["candidate_manifest"]
    if candidate_path.exists():
        candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
        replay.validate_candidate_manifest(candidates, ROOT)
    source_path = ROOT / config["outputs"]["source_manifest"]
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        replay.validate_source_manifest(source, ROOT)
    sample_path = ROOT / config["outputs"]["sample_manifest"]
    if sample_path.exists():
        sample = json.loads(sample_path.read_text(encoding="utf-8"))
        replay.validate_sample_manifest(sample, ROOT)
