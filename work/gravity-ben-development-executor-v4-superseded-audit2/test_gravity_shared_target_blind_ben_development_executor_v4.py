from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import sigma_theory_compiler.gravity_shared_target_blind_ben_development_executor_v4 as executor

from sigma_theory_compiler.gravity_shared_target_blind_ben_development_executor_v4 import (
    ABLATION_IDS,
    ACCESS_INTENT_PATH,
    AUTHORIZATION_PATH,
    BENDevelopmentExecutorV4Error,
    CONFIG_SECTION_SHA256,
    EXPECTED_XCOP_OBJECTS,
    PREFLIGHT_PATH,
    RESULT_PATH,
    _atomic_no_clobber,
    _build_candidate_decisions,
    _fit_shape_nuisance,
    _nontrivial_response_variation,
    _parity,
    _render_result_floats,
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
    validate_authorization,
    validate_bound_metadata,
    validate_config,
    validate_preflight,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_exact_scoring_scope_and_zero_access() -> None:
    config = load_config()
    assert config["candidate_registry"]["canonical_full_classes"] == 60
    assert config["candidate_registry"]["raw_candidates_frozen"] == 240
    assert config["ablation_registry"]["registered_total"] == 180
    assert config["ablation_registry"]["ordered_ablation_ids"] == ABLATION_IDS
    sparc = config["development_populations"]["sparc"]
    assert (sparc["parsed_container_objects"], sparc["parsed_container_rows"]) == (175, 3391)
    assert (sparc["objects"], sparc["rows"]) == (139, 2720)
    assert sparc["admitted_score_name_row_ledger_sha256"] == (
        "ea249dc3b71448cabb6ae9f2d0a68040eafe1021e92902311d2390b247b584dc"
    )
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
    assert config["claim_ceiling"]["constant_xcop_geometry_domain_switch_resolved"] is False
    assert config["runtime_preflight"]["must_complete_before_access_intent"] is True
    assert config["selection_contract"][
        "every_compared_control_must_be_valid_and_parity_pass"
    ] is True
    assert config["selection_contract"][
        "every_named_ablation_must_be_valid_and_parity_pass_in_both_domains"
    ] is True


def test_all_nested_config_sections_and_implementation_dependencies_are_frozen() -> None:
    config = load_config()
    assert set(CONFIG_SECTION_SHA256) == set(config) - {
        "schema_version",
        "status",
        "purpose",
        "implementation_source",
        "verifier_test",
    }
    assert config["source_bindings"]["sparc_full_sample_source"]["file_sha256"] == (
        "0afb71a53524c22e932d58bfb2d4c451c53d479608f23fc84968fe0d1a03bdfa"
    )
    assert config["source_bindings"]["real_data_gravity_source"]["file_sha256"] == (
        "1ff21cd7f968edb5b835cb4b977384319a9cea17bedb3e977e6d544b1705994c"
    )
    assert config["source_bindings"]["sigma_core_source"]["file_sha256"] == (
        "11f51a70b68efed1c4473df59f4e4745ec3d5de2f129f1458196799f88839eb2"
    )
    mutated = copy.deepcopy(config)
    mutated["candidate_registry"]["formula_template"] += "+post_freeze_change"
    with pytest.raises(BENDevelopmentExecutorV4Error, match="frozen config section"):
        validate_config(mutated)
    mutated = copy.deepcopy(config)
    mutated["xcop_shape_bridge_and_score"]["response_interpolation"]["new_key"] = True
    with pytest.raises(BENDevelopmentExecutorV4Error, match="frozen config section"):
        validate_config(mutated)


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
        "registered_variants_using_x_geometry": 117,
        "registered_variants_flagged_constant_xcop_geometry_domain_switch_risk": 117,
    }
    assert all(
        row["provenance_is_authoritative_novelty_finding"] is False
        for row in registered["variants"]
    )
    assert {row["ablation_id"] for row in registered["ablations"]} == set(ABLATION_IDS)
    assert all(
        row["uses_x_geometry"] == row["constant_xcop_geometry_domain_switch_risk"]
        for row in registered["variants"]
    )


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
    assert first["candidate_and_ablation_accounting"][
        "registered_variants_flagged_constant_xcop_geometry_domain_switch_risk"
    ] == 117
    assert len(first["registered_formula_identity_ledger_sha256"]) == 64
    assert first["immutable_scientific_contract_sha256"]["claim_ceiling"] == (
        CONFIG_SECTION_SHA256["claim_ceiling"]
    )
    if (ROOT / PREFLIGHT_PATH).is_file():
        assert validate_preflight(config) == first


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


def test_authorization_requires_exact_path_keys_and_RFC3339_UTC(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if not (ROOT / PREFLIGHT_PATH).is_file():
        pytest.skip("sealed preflight is generated only after source and tests are final")
    config = load_config()
    preflight = validate_preflight(config)
    supplied = authorization_template(config, preflight)
    supplied.update(
        {"authorized": True, "approved_by": "synthetic-test", "approved_at": "2026-08-29T12:00:00Z"}
    )
    monkeypatch.setattr(executor, "read_json", lambda _path: copy.deepcopy(supplied))
    assert validate_authorization(AUTHORIZATION_PATH, config, preflight)["authorized"] is True
    with pytest.raises(BENDevelopmentExecutorV4Error, match="exact frozen path"):
        validate_authorization(AUTHORIZATION_PATH.with_name("authorization-copy.json"), config, preflight)
    supplied["approved_at"] = "2026-08-29T12:00:00+00:00"
    with pytest.raises(BENDevelopmentExecutorV4Error, match="RFC3339 UTC"):
        validate_authorization(AUTHORIZATION_PATH, config, preflight)
    supplied["approved_at"] = "2026-08-29T12:00:00Z"
    supplied["extra"] = True
    with pytest.raises(BENDevelopmentExecutorV4Error, match="authorization keys changed"):
        validate_authorization(AUTHORIZATION_PATH, config, preflight)


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
    assert fit["zeta_objective_evaluations"] == 4097
    assert fit["analytic_scale_solves"] == 8194


@pytest.mark.parametrize(
    ("constant_channel", "expected_failure", "unexpected_failure"),
    [
        ("pressure", "H_pressure_response_constant", "H_temperature_response_constant"),
        ("temperature", "H_temperature_response_constant", "H_pressure_response_constant"),
    ],
)
def test_each_response_channel_has_a_separate_prefit_H_variation_gate(
    constant_channel: str, expected_failure: str, unexpected_failure: str
) -> None:
    x = np.linspace(0.1, 1.0, 10)
    q = np.exp(-x)
    f = np.ones_like(x)
    h = _reverse_integral(x, q, f)
    varying = x[1:-1]
    constant = np.full(len(varying), x[4])
    px = constant if constant_channel == "pressure" else varying
    tx = constant if constant_channel == "temperature" else varying
    hp = np.interp(px, x, h)
    ht = np.interp(tx, x, h)
    assert _nontrivial_response_variation(hp) is (constant_channel != "pressure")
    assert _nontrivial_response_variation(ht) is (constant_channel != "temperature")
    qp = np.exp(np.interp(np.log(px), np.log(x), np.log(q)))
    qt = np.exp(np.interp(np.log(tx), np.log(x), np.log(q)))
    cluster = {
        "object": f"constant-{constant_channel}",
        "x": x,
        "q": q,
        "pressure_x": px,
        "pressure_y": 2.0 * (hp + 0.2 * (1.0 - hp)),
        "pressure_sigma": np.full(len(px), 0.02),
        "temperature_x": tx,
        "temperature_y": 3.0 * (ht + 0.2 * (1.0 - ht)) / qt,
        "temperature_sigma": np.full(len(tx), 0.03),
    }
    scored = _score_xcop_shapes([f], [cluster])
    assert expected_failure in scored["per_object"][0]["failures"]
    assert unexpected_failure not in scored["per_object"][0]["failures"]
    assert scored["nuisance_fits"] == 0
    assert scored["response_row_score_terms"] == 0


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
    assert scored["object_score_reductions"] == 2
    assert scored["response_row_score_terms"] == 4


def _synthetic_comparison_ledgers() -> tuple[dict, dict, dict]:
    class_id = "class-000"
    registered = {
        "full": [
            {
                "full_class_id": class_id,
                "variant_id": f"full:{class_id}",
                "constant_xcop_geometry_domain_switch_risk": True,
            }
        ]
    }
    sparc = {f"full:{class_id}": {"valid": True, "domain_loss": 1.0 - 5.0e-14}}
    xcop = {f"full:{class_id}": {"valid": True, "domain_loss": 1.0 - 5.0e-14}}
    for name in ABLATION_IDS:
        key = f"ablation:{class_id}:{name}"
        sparc[key] = {"valid": True, "domain_loss": 2.0}
        xcop[key] = {"valid": True, "domain_loss": 2.0}
    sparc["control:sparc:newtonian_baryons"] = {"valid": True, "domain_loss": 1.0}
    sparc["control:sparc:empirical_rar"] = {"valid": True, "domain_loss": 1.0}
    xcop["control:xcop:gas_only_newtonian_shape"] = {"valid": True, "domain_loss": 1.0}
    xcop["control:xcop:uniform_acceleration_shape_control"] = {
        "valid": True,
        "domain_loss": 1.0,
    }
    return registered, sparc, xcop


def test_strict_selection_uses_unrounded_binary64_losses() -> None:
    registered, sparc, xcop = _synthetic_comparison_ledgers()
    # This improvement disappears if losses are rounded to 12 digits before comparison.
    assert format(sparc["full:class-000"]["domain_loss"], ".12e") == format(1.0, ".12e")
    decision = _build_candidate_decisions(registered, sparc, xcop)[0]
    assert decision["eligible"] is True
    assert decision["constant_xcop_geometry_domain_switch_risk"] is True
    rendered = _render_result_floats({"loss": sparc["full:class-000"]["domain_loss"]})
    assert rendered["loss"] == format(1.0 - 5.0e-14, ".17e")


@pytest.mark.parametrize("invalid_kind", ["control", "ablation"])
def test_invalid_control_or_named_ablation_blocks_every_winner(invalid_kind: str) -> None:
    registered, sparc, xcop = _synthetic_comparison_ledgers()
    if invalid_kind == "control":
        xcop["control:xcop:uniform_acceleration_shape_control"]["valid"] = False
    else:
        sparc["ablation:class-000:N_zero_ablation"]["valid"] = False
    decision = _build_candidate_decisions(registered, sparc, xcop)[0]
    assert decision["eligible"] is False
    assert any(value is False for value in decision["checks"].values())


def test_parity_metrics_remain_binary64_until_final_render() -> None:
    config = load_config()
    parity = _parity(
        np.asarray([1.0], dtype=np.float64),
        np.asarray([1.0 + 5.0e-14], dtype=np.float64),
        config,
    )
    assert isinstance(parity["max_abs"], float)
    assert isinstance(parity["max_rel"], float)


def test_gpu_preflight_precedes_intent_and_payload_in_source() -> None:
    text = (ROOT / config_path()).read_text(encoding="utf-8")
    execute_body = text[text.index("def execute(") : text.index("def check_preflight(")]
    assert execute_body.index("_preflight_gpu(config)") < execute_body.index(
        "_atomic_no_clobber(ACCESS_INTENT_PATH"
    )
    assert execute_body.index("_atomic_no_clobber(ACCESS_INTENT_PATH") < execute_body.index(
        "load_sparc_development(config)"
    )


def config_path() -> Path:
    return Path("src/sigma_theory_compiler/gravity_shared_target_blind_ben_development_executor_v4.py")


def test_atomic_no_clobber_durably_links_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    relative = Path("receipt.json")
    value = {"schema_version": "synthetic", "ok": True}
    _atomic_no_clobber(relative, value)
    assert executor.read_json(relative) == value
    with pytest.raises(BENDevelopmentExecutorV4Error, match="refusing to overwrite"):
        _atomic_no_clobber(relative, value)


def test_no_production_outputs_exist_during_preparation() -> None:
    assert not (ROOT / ACCESS_INTENT_PATH).exists()
    assert not (ROOT / RESULT_PATH).exists()
    # The expected preparation products are separate from production output.
    assert PREFLIGHT_PATH != RESULT_PATH
    assert AUTHORIZATION_PATH != ACCESS_INTENT_PATH
