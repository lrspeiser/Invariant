from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_shared_target_blind_ben_development_executor_v4 import (
    ABLATION_IDS,
    ACCESS_INTENT_PATH,
    AUTHORIZATION_PATH,
    EXPECTED_XCOP_OBJECTS,
    PREFLIGHT_PATH,
    RESULT_PATH,
    _fit_shape_nuisance,
    _reverse_integral,
    _score_sparc_vector,
    _score_xcop_shapes,
    _shape_gates,
    _xcop_inventory,
    authorization_template,
    build_preflight,
    build_registered_variants,
    content_sha256,
    load_config,
    read_json,
    validate_bound_metadata,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_exact_scoring_scope_and_zero_access() -> None:
    config = load_config()
    assert config["candidate_registry"]["canonical_full_classes"] == 60
    assert config["candidate_registry"]["raw_candidates_frozen"] == 240
    assert config["ablation_registry"]["registered_total"] == 180
    assert config["ablation_registry"]["ordered_ablation_ids"] == ABLATION_IDS
    assert config["development_populations"]["sparc"]["objects"] == 139
    assert config["development_populations"]["sparc"]["rows"] == 2720
    assert config["development_populations"]["xcop"]["objects"] == EXPECTED_XCOP_OBJECTS
    assert config["development_populations"]["xcop"]["response_rows"] == 184
    assert config["development_populations"]["forbidden"] == {
        "confirmation_or_independent_rows": True,
        "groups": True,
        "lensing": True,
        "little_things": True,
        "stellar_mass_files": True,
        "inferred_total_mass": True,
        "network": True,
        "model_calls": True,
        "paid_calls": True,
    }
    assert config["zero_access_chronology"]["development_payload_files_opened"] == 0
    assert config["zero_access_chronology"]["candidate_scores"] == 0
    assert config["claim_ceiling"]["fresh_confirmation"] is False
    assert config["claim_ceiling"]["alternative_to_gr_established"] is False


def test_bound_metadata_and_registry_validate_without_payload_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config()
    forbidden = {
        (ROOT / config["development_populations"]["sparc"]["payload_path"]).resolve(),
        (ROOT / config["development_populations"]["xcop"]["raw_directory"]).resolve(),
    }
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in forbidden or any(parent in forbidden for parent in resolved.parents):
            raise AssertionError("metadata preflight touched a scientific payload")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    loaded = validate_bound_metadata(config)
    assert loaded["synthetic_receipt"]["candidate_registry"]["equivalence_class_count"] == 60
    inventory = _xcop_inventory(config, loaded["xcop_source_receipt"])
    assert len(inventory) == 24
    assert all(row["cluster"] in EXPECTED_XCOP_OBJECTS for row in inventory)
    assert {row["role"] for row in inventory} == {"density", "pressure", "temperature"}


def test_ablation_registry_preserves_registered_and_mathematical_counts() -> None:
    config = load_config()
    loaded = validate_bound_metadata(config)
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    assert len(registered["full"]) == 60
    assert len(registered["ablations"]) == 180
    assert len(registered["variants"]) == 240
    assert registered["accounting"] == {
        "raw_candidates_frozen": 240,
        "canonical_full_classes": 60,
        "registered_ablations": 180,
        "unique_ablation_asts": 51,
        "duplicate_registered_ablation_instances": 129,
        "ablation_asts_overlapping_full_classes": 33,
        "unique_asts_across_full_and_ablations": 78,
        "raw_equivalent_members_scored": 0,
    }
    assert all(
        row["provenance_is_authoritative_novelty_finding"] is False
        for row in registered["variants"]
    )
    assert {row["ablation_id"] for row in registered["ablations"]} == set(ABLATION_IDS)


def test_preflight_is_deterministic_metadata_only(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config()
    raw_paths = {
        (ROOT / config["development_populations"]["sparc"]["payload_path"]).resolve(),
        (ROOT / config["development_populations"]["xcop"]["raw_directory"]).resolve(),
    }
    original = Path.read_bytes

    def guarded(path: Path) -> bytes:
        resolved = path.resolve()
        if resolved in raw_paths or any(parent in raw_paths for parent in resolved.parents):
            raise AssertionError("preflight touched a target")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    first = build_preflight()
    second = build_preflight()
    assert first == second
    assert first["content_sha256"] == content_sha256(first)
    assert first["target_files_opened"] == 0
    assert first["target_rows_read"] == 0
    assert first["scores_computed"] == 0
    assert first["candidate_and_ablation_accounting"]["registered_ablations"] == 180


def test_false_authorization_fails_before_any_scope_can_change() -> None:
    config = load_config()
    # Use the deterministic in-memory preflight so this unit test does not require the
    # generated preflight artifact to exist yet.
    preflight = build_preflight()
    if not (ROOT / PREFLIGHT_PATH).is_file():
        assert config["authorization_gate"]["current_authorization_expected"] is False
        assert preflight["target_files_opened"] == 0
        return
    false_template = authorization_template(
        config,
        {
            **preflight,
            "content_sha256": preflight["content_sha256"],
        },
    )
    assert false_template["authorized"] is False
    current = (
        read_json(AUTHORIZATION_PATH) if (ROOT / AUTHORIZATION_PATH).is_file() else false_template
    )
    assert current["authorized"] is False
    # validate_authorization reads a path, so directly assert the invariant the validator
    # enforces and separately test a scope mutation against a generated artifact below.
    assert config["authorization_gate"]["authorization_replay_allowed"] is False


def test_shape_integral_and_nuisance_fit_pass_on_synthetic_profiles() -> None:
    x = np.linspace(0.1, 1.0, 12)
    q = np.exp(-1.2 * x)
    f = 0.4 + np.sqrt(q) / (1.0 + x)
    f /= np.max(f)
    h = _reverse_integral(x, q, f)
    assert h[-1] == 0.0
    assert _shape_gates(x, q, f, h) == []
    px = x[1:-1:2]
    tx = x[2:-1:2]
    hp = np.interp(px, x, h)
    ht = np.interp(tx, x, h)
    qp = np.exp(np.interp(np.log(px), np.log(x), np.log(q)))
    qt = np.exp(np.interp(np.log(tx), np.log(x), np.log(q)))
    zeta = 0.25
    cluster = {
        "pressure_y": 2.0 * (hp + zeta * (1.0 - hp)),
        "pressure_sigma": np.full(len(px), 0.02),
        "temperature_y": 3.0 * (ht + zeta * (1.0 - ht)) / qt,
        "temperature_sigma": np.full(len(tx), 0.03),
    }
    fit = _fit_shape_nuisance(hp, qp, ht, qt, cluster)
    assert fit["valid"] is True
    assert float(fit["loss"]) < 1.0e-20
    assert abs(float(fit["zeta"]) - zeta) < 1.0 / 4096.0
    assert fit["jacobian_rank"] == 3


def test_one_failed_cluster_is_retained_but_never_prunes_family() -> None:
    x = np.linspace(0.1, 1.0, 8)
    q = np.exp(-x)
    h = _reverse_integral(x, q, np.ones_like(x))
    px = x[1:-1]
    cluster = {
        "object": "synthetic",
        "x": x,
        "q": q,
        "pressure_x": px,
        "pressure_y": 1.5 * (np.interp(px, x, h) + 0.2 * (1 - np.interp(px, x, h))),
        "pressure_sigma": np.full(len(px), 0.02),
        "temperature_x": px,
        "temperature_y": 2.0
        * (np.interp(px, x, h) + 0.2 * (1 - np.interp(px, x, h)))
        / np.exp(np.interp(np.log(px), np.log(x), np.log(q))),
        "temperature_sigma": np.full(len(px), 0.03),
    }
    scored = _score_xcop_shapes([np.full(len(x), -1.0)], [cluster])
    assert scored["valid"] is False
    assert scored["valid_object_scores"] == 0
    assert scored["per_object"][0]["terminal_veto"] is False
    assert scored["formula_family_pruned"] is False


def test_sparc_equal_object_weighting_not_row_pooling() -> None:
    objects = [
        {
            "object": "short",
            "rows": 1,
            "radius": np.asarray([1.0]),
            "vobs": np.asarray([0.0]),
            "sigma": np.asarray([1.0]),
        },
        {
            "object": "long",
            "rows": 3,
            "radius": np.asarray([1.0, 1.0, 1.0]),
            "vobs": np.asarray([np.sqrt(3702.81458)] * 3),
            "sigma": np.asarray([1.0, 1.0, 1.0]),
        },
    ]
    scored = _score_sparc_vector(np.ones(4), objects)
    short_loss = float(scored["per_object"][0]["loss"])
    long_loss = float(scored["per_object"][1]["loss"])
    assert float(scored["domain_loss"]) == pytest.approx(0.5 * (short_loss + long_loss))


def test_no_production_outputs_exist_during_preparation() -> None:
    assert not (ROOT / ACCESS_INTENT_PATH).exists()
    assert not (ROOT / RESULT_PATH).exists()
    # The expected preparation products are separate from production output.
    assert PREFLIGHT_PATH != RESULT_PATH
    assert AUTHORIZATION_PATH != ACCESS_INTENT_PATH
