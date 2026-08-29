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
    CONFIG_SECTION_SHA256,
    EXPECTED_XCOP_OBJECTS,
    PREFLIGHT_PATH,
    RESULT_PATH,
    BENDevelopmentExecutorV4Error,
    _activate_reference_svd_runtime,
    _atomic_no_clobber,
    _build_candidate_decisions,
    _failure_classification,
    _fit_shape_nuisance,
    _nontrivial_response_variation,
    _parity,
    _preflight_gpu,
    _read_payload_bytes,
    _render_result_floats,
    _reverse_integral,
    _score_sparc_vector,
    _score_xcop_shapes,
    _shape_gates,
    _static_runtime_receipt,
    _validate_rendered_parity,
    _write_access_failure_receipt,
    _write_phase_receipt,
    _xcop_inventory,
    authorization_template,
    build_preflight,
    build_registered_variants,
    check_result,
    content_sha256,
    load_config,
    read_json,
    validate_authorization,
    validate_bound_metadata,
    validate_config,
    validate_preflight,
    validate_result_document,
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
    assert config["zero_access_chronology"]["development_payload_file_open_attempts"] == 0
    assert config["zero_access_chronology"]["development_payload_file_reads_completed"] == 0
    assert config["zero_access_chronology"]["candidate_scores"] == 0
    assert config["claim_ceiling"]["fresh_confirmation"] is False
    assert config["claim_ceiling"]["alternative_to_gr_established"] is False
    assert config["claim_ceiling"]["constant_xcop_geometry_domain_switch_resolved"] is False
    assert config["runtime_preflight"]["must_complete_before_access_intent"] is True
    runtime = config["runtime_environment_contract"]
    assert runtime["comparison_operator"] == "binary64_numerical_indifference_band"
    assert runtime["reference_environment_validation_required_before_access_intent"] is True
    assert runtime["host"]["python_version"] == "3.13.5"
    assert runtime["packages"]["numpy"]["version"] == "2.2.6"
    assert runtime["packages"]["astropy"]["version"] == "7.1.1"
    assert runtime["packages"]["cupy-cuda12x"]["version"] == "13.5.1"
    assert runtime["cuda"] == {
        "runtime_version_integer": 12090,
        "driver_version_integer": 13000,
        "device_name": "NVIDIA GeForce RTX 5090",
        "compute_capability_major": 12,
        "compute_capability_minor": 0,
        "device_count": 1,
    }
    assert config["result_validation_contract"]["exact_result_schema_required"] is True
    assert (
        config["interrupted_run_contract"][
            "write_atomic_no_clobber_failure_receipt_on_any_post_intent_or_runtime_restoration_exception"
        ]
        is True
    )
    assert (
        config["selection_contract"]["every_compared_control_must_be_valid_and_parity_pass"] is True
    )
    assert (
        config["selection_contract"][
            "every_named_ablation_must_be_valid_and_parity_pass_in_both_domains"
        ]
        is True
    )


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
    assert (
        first["candidate_and_ablation_accounting"][
            "registered_variants_flagged_constant_xcop_geometry_domain_switch_risk"
        ]
        == 117
    )
    assert len(first["registered_formula_identity_ledger_sha256"]) == 64
    assert (
        first["immutable_scientific_contract_sha256"]["claim_ceiling"]
        == (CONFIG_SECTION_SHA256["claim_ceiling"])
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
        validate_authorization(
            AUTHORIZATION_PATH.with_name("authorization-copy.json"), config, preflight
        )
    supplied["approved_at"] = "2026-08-29T12:00:00+00:00"
    with pytest.raises(BENDevelopmentExecutorV4Error, match="RFC3339 UTC"):
        validate_authorization(AUTHORIZATION_PATH, config, preflight)
    supplied["approved_at"] = "2026-08-29T12:00:00Z"
    supplied["extra"] = True
    with pytest.raises(BENDevelopmentExecutorV4Error, match="authorization keys changed"):
        validate_authorization(AUTHORIZATION_PATH, config, preflight)


def _patch_minimal_check_result_preamble(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    config: dict = {}
    preflight = {"content_sha256": "b" * 64}
    template = {
        "schema_version": executor.AUTHORIZATION_SCHEMA,
        "authorization_id": "synthetic-authorization",
        "authorization_path": AUTHORIZATION_PATH.as_posix(),
        "authorized": False,
        "approved_by": None,
        "approved_at": None,
        "approval_text": "synthetic exact approval",
    }
    monkeypatch.setattr(executor, "load_config", lambda: config)
    monkeypatch.setattr(
        executor,
        "validate_bound_metadata",
        lambda _config: {"synthetic_receipt": {"candidate_registry": {}}},
    )
    monkeypatch.setattr(executor, "validate_preflight", lambda _config: preflight)
    monkeypatch.setattr(
        executor,
        "authorization_template",
        lambda _config, _preflight: copy.deepcopy(template),
    )
    lock_target = tmp_path / executor.TERMINAL_STATE_LOCK_PATH
    lock_target.parent.mkdir(parents=True, exist_ok=True)
    lock_target.write_bytes(b"0")
    terminal = {
        "schema_version": executor.TERMINAL_SUCCESS_SCHEMA,
        "status": "TERMINAL_SUCCESS_AFTER_RUNTIME_RESTORATION",
        "authorization_id": template["authorization_id"],
        "authorization_file_sha256": "a" * 64,
        "access_intent_content_sha256": "b" * 64,
        "result_content_sha256": "c" * 64,
        "adjudication_content_sha256": "d" * 64,
        "runtime_restoration_completed": True,
        "access_failure_absent_before_publication": True,
        "terminal_state_lock_held_during_publication": True,
        "authorization_replay_allowed": False,
    }
    terminal["content_sha256"] = content_sha256(terminal)
    terminal_target = tmp_path / executor.TERMINAL_SUCCESS_PATH
    terminal_target.write_text(executor.canonical_json(terminal) + "\n", encoding="utf-8")
    return template


@pytest.mark.parametrize("mode", ["missing", "swapped"])
def test_check_result_rejects_missing_or_swapped_exact_authorization(
    mode: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    template = _patch_minimal_check_result_preamble(tmp_path, monkeypatch)
    if mode == "swapped":
        supplied = {
            **template,
            "authorization_id": "different-authorization",
            "authorized": True,
            "approved_by": "synthetic-test",
            "approved_at": "2026-08-29T12:00:00Z",
        }
        target = tmp_path / AUTHORIZATION_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(executor.canonical_json(supplied) + "\n", encoding="utf-8")
    expected = "required artifact absent" if mode == "missing" else "authorization scope changed"
    with pytest.raises(BENDevelopmentExecutorV4Error, match=expected):
        check_result()


@pytest.mark.parametrize("mismatch_document", ["access_intent", "result", "adjudication"])
def test_check_result_matches_approved_authorization_bytes_across_every_output(
    mismatch_document: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    template = _patch_minimal_check_result_preamble(tmp_path, monkeypatch)
    supplied = {
        **template,
        "authorized": True,
        "approved_by": "synthetic-test",
        "approved_at": "2026-08-29T12:00:00Z",
    }
    authorization_target = tmp_path / AUTHORIZATION_PATH
    authorization_target.parent.mkdir(parents=True, exist_ok=True)
    authorization_target.write_text(executor.canonical_json(supplied) + "\n", encoding="utf-8")
    authorization_sha256 = executor.file_sha256(authorization_target)
    paths = {
        "access_intent": executor.ACCESS_INTENT_PATH,
        "result": executor.RESULT_PATH,
        "adjudication": executor.ADJUDICATION_PATH,
    }
    for label, relative in paths.items():
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        value = {
            "authorization_id": supplied["authorization_id"],
            "authorization_file_sha256": (
                "c" * 64 if label == mismatch_document else authorization_sha256
            ),
        }
        target.write_text(executor.canonical_json(value) + "\n", encoding="utf-8")
    monkeypatch.setattr(executor, "build_registered_variants", lambda _registry: {})
    monkeypatch.setattr(
        executor,
        "build_result_adjudication",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("authorization mismatch must precede adjudication")
        ),
    )
    with pytest.raises(BENDevelopmentExecutorV4Error, match="authorization file identity"):
        check_result()


def test_check_result_cannot_observe_terminal_state_while_executor_lock_is_held(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    monkeypatch.setattr(executor, "load_config", dict)
    active_lock = executor._acquire_terminal_state_lock(create=True)
    try:
        with pytest.raises(BENDevelopmentExecutorV4Error, match="terminal state is still active"):
            check_result()
    finally:
        executor._release_terminal_state_lock(active_lock)


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
    qt = np.exp(np.interp(np.log(tx), np.log(x), np.log(q)))
    zeta = 0.25
    cluster = {
        "pressure_y": 2.0 * (hp + zeta * (1.0 - hp)),
        "pressure_sigma": np.full(len(px), 0.02),
        "temperature_y": 3.0 * (ht + zeta * (1.0 - ht)) / qt,
        "temperature_sigma": np.full(len(tx), 0.03),
    }
    fit = _fit_shape_nuisance(hp, ht, qt, cluster)
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


def test_selection_uses_preregistered_numerical_indifference_band() -> None:
    registered, sparc, xcop = _synthetic_comparison_ledgers()
    config = load_config()
    # A microscopic binary64 difference is retained but cannot become a scientific win.
    assert format(sparc["full:class-000"]["domain_loss"], ".12e") == format(1.0, ".12e")
    decision = _build_candidate_decisions(registered, sparc, xcop, config)[0]
    assert decision["eligible"] is False
    assert decision["constant_xcop_geometry_domain_switch_risk"] is True
    rendered = _render_result_floats({"loss": sparc["full:class-000"]["domain_loss"]})
    assert rendered["loss"] == format(1.0 - 5.0e-14, ".17e")
    for domain in (sparc, xcop):
        domain["full:class-000"]["domain_loss"] = 1.0 - 1.0e-9
    assert _build_candidate_decisions(registered, sparc, xcop, config)[0]["eligible"] is True


@pytest.mark.parametrize("invalid_kind", ["control", "ablation"])
def test_invalid_control_or_named_ablation_blocks_every_winner(invalid_kind: str) -> None:
    registered, sparc, xcop = _synthetic_comparison_ledgers()
    config = load_config()
    sparc["full:class-000"]["domain_loss"] = 1.0 - 1.0e-9
    xcop["full:class-000"]["domain_loss"] = 1.0 - 1.0e-9
    if invalid_kind == "control":
        xcop["control:xcop:uniform_acceleration_shape_control"]["valid"] = False
    else:
        sparc["ablation:class-000:N_zero_ablation"]["valid"] = False
    decision = _build_candidate_decisions(registered, sparc, xcop, config)[0]
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
    rendered = _render_result_floats(parity)
    assert _validate_rendered_parity(rendered, config, "synthetic") is True


def test_reference_runtime_and_single_thread_svd_backend_are_enforceable() -> None:
    config = load_config()
    receipt = _static_runtime_receipt(config)
    assert receipt["host"] == config["runtime_environment_contract"]["host"]
    assert receipt["packages"] == config["runtime_environment_contract"]["packages"]
    assert (
        receipt["numpy_simd_dispatch"]
        == config["runtime_environment_contract"]["numpy_simd_dispatch"]
    )
    limiter, active = _activate_reference_svd_runtime(config)
    try:
        assert active["num_threads"] == 1
        assert active["version"] == "0.3.29"
        assert active["internal_api"] == "openblas"
    finally:
        limiter.restore_original_limits()


def test_reference_cuda_driver_and_rtx_5090_fp64_probe_pass_without_target_access() -> None:
    config = load_config()
    cp, receipt = _preflight_gpu(config)
    assert receipt["pass"] is True
    assert receipt["scientific_payload_access"] is False
    assert receipt["device_name"] == "NVIDIA GeForce RTX 5090"
    assert receipt["cuda_runtime_version_integer"] == 12090
    assert receipt["cuda_driver_version_integer"] == 13000
    assert (receipt["compute_capability_major"], receipt["compute_capability_minor"]) == (12, 0)
    cp.get_default_memory_pool().free_all_blocks()


def test_gpu_preflight_precedes_intent_and_payload_in_source() -> None:
    text = (ROOT / config_path()).read_text(encoding="utf-8")
    execute_body = text[text.index("def execute(") : text.index("def check_preflight(")]
    assert execute_body.index("_static_runtime_receipt(config)") < execute_body.index(
        "_preflight_gpu(config)"
    )
    assert execute_body.index("_activate_reference_svd_runtime(config)") < execute_body.index(
        "_preflight_gpu(config)"
    )
    assert execute_body.index("_preflight_gpu(config)") < execute_body.index(
        "_atomic_no_clobber(ACCESS_INTENT_PATH"
    )
    assert execute_body.index("_atomic_no_clobber(ACCESS_INTENT_PATH") < execute_body.index(
        "load_sparc_development(config, access_accounting)"
    )


def test_terminal_success_is_structurally_after_restoration_and_guarded_by_lock() -> None:
    text = (ROOT / config_path()).read_text(encoding="utf-8")
    execute_body = text[text.index("def execute(") : text.index("def check_preflight(")]
    assert execute_body.index("limiter.restore_original_limits()") < execute_body.index(
        "_build_terminal_success_marker("
    )
    assert execute_body.index("_build_terminal_success_marker(") < execute_body.index(
        "_atomic_no_clobber(TERMINAL_SUCCESS_PATH"
    )
    check_body = text[text.index("def check_result(") : text.index("def main(")]
    assert check_body.index("_acquire_terminal_state_lock(create=False)") < check_body.index(
        "_check_result_under_terminal_lock(config)"
    )


def config_path() -> Path:
    return Path(
        "src/sigma_theory_compiler/gravity_shared_target_blind_ben_development_executor_v4.py"
    )


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


def test_post_intent_failure_receipt_is_exact_durable_and_nonreplayable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    monkeypatch.setattr(executor, "file_sha256", lambda _path: "a" * 64)
    authorization = {"authorization_id": "synthetic-authorization"}
    preflight = {"content_sha256": "b" * 64}
    access_intent = {"content_sha256": "c" * 64}
    accounting = executor._new_access_accounting()
    accounting["payload_file_open_attempts"] = 1
    _write_access_failure_receipt(
        authorization,
        Path("authorization.json"),
        preflight,
        access_intent,
        "access_intent_committed",
        accounting,
        "sparc_payload_read",
        OSError(r"C:\private\payload.fits ANTHROPIC_API_KEY=sk-secret-value"),
    )
    receipt = executor.read_json(executor.ACCESS_FAILURE_PATH)
    assert receipt["schema_version"] == executor.ACCESS_FAILURE_SCHEMA
    assert receipt["status"] == "TERMINAL_STATE_FAILURE_SUCCESSOR_REQUIRED"
    assert receipt["last_completed_phase"] == "access_intent_committed"
    assert receipt["actual_access_accounting"] == accounting
    assert receipt["failure_operation"] == "sparc_payload_read"
    assert receipt["error_code"] == "PAYLOAD_READ_FAILURE"
    assert receipt["error_class"] == "payload_read"
    assert receipt["authorization_replay_allowed"] is False
    assert receipt["successor_contract_required"] is True
    assert receipt["content_sha256"] == content_sha256(receipt)
    serialized = executor.canonical_json(receipt)
    assert "private" not in serialized
    assert "ANTHROPIC" not in serialized
    assert "sk-secret-value" not in serialized


@pytest.mark.parametrize(
    ("error", "operation", "expected"),
    [
        (
            OSError("secret payload path"),
            "sparc_payload_read",
            ("PAYLOAD_READ_FAILURE", "payload_read"),
        ),
        (
            OSError("secret phase path"),
            "phase_receipt_publication",
            ("PHASE_RECEIPT_PUBLICATION_FAILURE", "phase_receipt_publication"),
        ),
        (
            OSError("secret result path"),
            "result_publication",
            ("RESULT_PUBLICATION_FAILURE", "result_publication"),
        ),
        (
            OSError("secret adjudication path"),
            "adjudication_publication",
            ("ADJUDICATION_PUBLICATION_FAILURE", "adjudication_publication"),
        ),
        (
            OSError("backend failure"),
            "formula_scoring",
            ("DETERMINISTIC_EXECUTION_FAILURE", "execution"),
        ),
        (
            OSError("validation input failure"),
            "result_validation",
            ("CONTRACT_OR_SCORE_VALIDATION_FAILURE", "validation"),
        ),
        (
            BENDevelopmentExecutorV4Error("payload seal changed"),
            "sparc_payload_read",
            ("CONTRACT_OR_SCORE_VALIDATION_FAILURE", "validation"),
        ),
        (
            RuntimeError("restore failed"),
            "runtime_restoration",
            ("RUNTIME_RESTORATION_FAILURE", "runtime_restoration"),
        ),
    ],
)
def test_failure_classification_is_phase_aware_and_value_free(
    error: BaseException, operation: str, expected: tuple[str, str]
) -> None:
    assert _failure_classification(error, operation) == expected


def test_payload_read_failure_counts_attempt_before_open_and_not_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"synthetic, non-scientific unit-test bytes")

    def fail_after_open(path: Path) -> bytes:
        with path.open("rb") as handle:
            assert handle.read(1) == b"s"
        raise OSError(r"C:\private\payload.bin API_KEY=sk-secret-value")

    monkeypatch.setattr(Path, "read_bytes", fail_after_open)
    accounting = executor._new_access_accounting()
    with pytest.raises(OSError):
        _read_payload_bytes(
            Path("payload.bin"),
            accounting,
            "sparc_file_open_attempts",
            "sparc_file_reads_completed",
        )
    assert accounting["payload_file_open_attempts"] == 1
    assert accounting["sparc_file_open_attempts"] == 1
    assert accounting["payload_file_reads_completed"] == 0
    assert accounting["sparc_file_reads_completed"] == 0


def test_completed_phase_receipt_is_append_only_and_records_actual_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    monkeypatch.setattr(executor, "file_sha256", lambda _path: "a" * 64)
    authorization = {"authorization_id": "synthetic-authorization"}
    preflight = {"content_sha256": "b" * 64}
    access_intent = {"content_sha256": "c" * 64}
    accounting = executor._new_access_accounting()
    accounting.update(
        {
            "payload_file_open_attempts": 1,
            "payload_file_reads_completed": 1,
            "sparc_file_open_attempts": 1,
            "sparc_file_reads_completed": 1,
            "sparc_container_objects_parsed": 175,
            "sparc_container_rows_parsed": 3391,
            "sparc_score_objects_loaded": 139,
            "sparc_score_rows_loaded": 2720,
        }
    )
    _write_phase_receipt(
        authorization,
        Path("authorization.json"),
        preflight,
        access_intent,
        "sparc_development_loaded",
        accounting,
    )
    path = executor.PHASE_RECEIPT_DIRECTORY / "phase-02-sparc_development_loaded.json"
    receipt = executor.read_json(path)
    assert receipt["schema_version"] == executor.PHASE_RECEIPT_SCHEMA
    assert receipt["phase_ordinal"] == 2
    assert receipt["actual_access_accounting"] == accounting
    assert receipt["content_sha256"] == content_sha256(receipt)
    with pytest.raises(BENDevelopmentExecutorV4Error, match="refusing to overwrite"):
        _write_phase_receipt(
            authorization,
            Path("authorization.json"),
            preflight,
            access_intent,
            "sparc_development_loaded",
            accounting,
        )


def test_final_phase_link_then_fsync_failure_forces_failure_receipt_and_blocks_check_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    monkeypatch.setattr(executor, "file_sha256", lambda _path: "a" * 64)
    authorization = {"authorization_id": "synthetic-authorization"}
    preflight = {"content_sha256": "b" * 64}
    access_intent = {"content_sha256": "c" * 64}
    accounting = executor._new_access_accounting()
    accounting["adjudication_receipts_committed"] = 1
    for relative in (executor.RESULT_PATH, executor.ADJUDICATION_PATH):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}\n", encoding="utf-8")
    phase_directory = tmp_path / executor.PHASE_RECEIPT_DIRECTORY
    phase_directory.mkdir(parents=True, exist_ok=True)
    for ordinal, phase in enumerate(executor.PHASES[:-1], start=1):
        (phase_directory / f"phase-{ordinal:02d}-{phase}.json").write_text("{}\n", encoding="utf-8")

    original_fsync_directory = executor._fsync_directory
    injected = False

    def fail_final_phase_directory_once(directory: Path) -> None:
        nonlocal injected
        if not injected and directory.resolve() == phase_directory.resolve():
            injected = True
            raise OSError(r"C:\private\phase-directory secret-token")
        original_fsync_directory(directory)

    monkeypatch.setattr(executor, "_fsync_directory", fail_final_phase_directory_once)
    with pytest.raises(OSError, match="secret-token") as caught:
        _write_phase_receipt(
            authorization,
            Path("authorization.json"),
            preflight,
            access_intent,
            "adjudication_committed",
            accounting,
        )
    final_phase = phase_directory / "phase-07-adjudication_committed.json"
    assert final_phase.exists()  # hard link exists even though directory fsync failed
    _write_access_failure_receipt(
        authorization,
        Path("authorization.json"),
        preflight,
        access_intent,
        "adjudication_committed",
        accounting,
        "phase_receipt_publication",
        caught.value,
    )
    failure = executor.read_json(executor.ACCESS_FAILURE_PATH)
    assert failure["error_code"] == "PHASE_RECEIPT_PUBLICATION_FAILURE"
    assert failure["failure_operation"] == "phase_receipt_publication"
    assert failure["result_exists"] is True
    assert failure["adjudication_exists"] is True
    assert len(list(phase_directory.glob("phase-*.json"))) == 7

    monkeypatch.setattr(executor, "load_config", dict)
    monkeypatch.setattr(
        executor,
        "validate_bound_metadata",
        lambda _config: (_ for _ in ()).throw(AssertionError("must fail before adjudication")),
    )
    with pytest.raises(BENDevelopmentExecutorV4Error, match="access-failure receipt exists"):
        check_result()


def _patch_synthetic_execute_scaffolding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limiter: object
) -> dict:
    config = load_config()
    monkeypatch.setattr(executor, "ROOT", tmp_path)
    authorization_target = tmp_path / AUTHORIZATION_PATH
    authorization_target.parent.mkdir(parents=True, exist_ok=True)
    authorization_target.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(executor, "file_sha256", lambda _path: "a" * 64)
    monkeypatch.setattr(executor, "load_config", lambda: config)
    monkeypatch.setattr(
        executor,
        "validate_bound_metadata",
        lambda _config: {
            "synthetic_receipt": {"candidate_registry": {}},
            "xcop_source_receipt": {},
        },
    )
    monkeypatch.setattr(
        executor, "validate_preflight", lambda _config: {"content_sha256": "b" * 64}
    )
    monkeypatch.setattr(
        executor,
        "validate_authorization",
        lambda _path, _config, _preflight: {"authorization_id": "synthetic-authorization"},
    )
    monkeypatch.setattr(
        executor,
        "build_registered_variants",
        lambda _registry: {"variants": [], "accounting": {}},
    )
    monkeypatch.setattr(executor, "_xcop_inventory", lambda _config, _receipt: [])
    monkeypatch.setattr(executor, "_static_runtime_receipt", lambda _config: {})
    monkeypatch.setattr(
        executor,
        "_activate_reference_svd_runtime",
        lambda _config: (limiter, {"synthetic": True}),
    )
    monkeypatch.setattr(
        executor,
        "_preflight_gpu",
        lambda _config: (
            object(),
            {"pass": True, "scientific_payload_access": False},
        ),
    )
    return config


def test_intent_link_then_directory_fsync_failure_is_retained_and_never_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Limiter:
        restored = False

        def restore_original_limits(self) -> None:
            self.restored = True

    limiter = Limiter()
    _patch_synthetic_execute_scaffolding(tmp_path, monkeypatch, limiter)
    monkeypatch.setattr(
        executor,
        "load_sparc_development",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("intent publication failure must precede target access")
        ),
    )
    original_fsync_directory = executor._fsync_directory
    injected = False

    def fail_first_directory_fsync(directory: Path) -> None:
        nonlocal injected
        if not injected:
            injected = True
            raise OSError(r"C:\private\intent-directory secret-token")
        original_fsync_directory(directory)

    monkeypatch.setattr(executor, "_fsync_directory", fail_first_directory_fsync)
    with pytest.raises(OSError, match="secret-token"):
        executor.execute(AUTHORIZATION_PATH)
    assert limiter.restored is True
    assert (tmp_path / executor.ACCESS_INTENT_PATH).exists()
    assert not (tmp_path / executor.TERMINAL_SUCCESS_PATH).exists()
    failure = executor.read_json(executor.ACCESS_FAILURE_PATH)
    assert failure["access_intent_exists"] is True
    assert failure["last_completed_phase"] is None
    assert failure["failure_operation"] == "access_intent_publication"
    assert failure["error_code"] == "ACCESS_INTENT_PUBLICATION_FAILURE"
    assert "secret-token" not in executor.canonical_json(failure)
    with pytest.raises(BENDevelopmentExecutorV4Error, match="access-failure receipt exists"):
        check_result()


def test_post_phase_seven_runtime_restoration_failure_blocks_terminal_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Limiter:
        calls = 0

        def restore_original_limits(self) -> None:
            self.calls += 1
            raise OSError(r"C:\private\blas-runtime secret-token")

    limiter = Limiter()
    _patch_synthetic_execute_scaffolding(tmp_path, monkeypatch, limiter)
    monkeypatch.setattr(
        executor,
        "load_sparc_development",
        lambda _config, _accounting: {"objects": [], "synthetic_population": True},
    )
    monkeypatch.setattr(
        executor,
        "load_xcop_development",
        lambda _config, _inventory, _accounting: {
            "objects": [],
            "synthetic_population": True,
        },
    )
    monkeypatch.setattr(
        executor,
        "_execute_scoring",
        lambda *_args, **_kwargs: {"actual_compute_accounting": {}},
    )
    monkeypatch.setattr(
        executor,
        "validate_result_document",
        lambda *_args, **_kwargs: {"validation_pass": True},
    )
    monkeypatch.setattr(
        executor,
        "build_result_adjudication",
        lambda *_args, **_kwargs: {"content_sha256": "d" * 64},
    )
    with pytest.raises(OSError, match="secret-token"):
        executor.execute(AUTHORIZATION_PATH)
    assert limiter.calls == 1
    assert (tmp_path / executor.RESULT_PATH).exists()
    assert (tmp_path / executor.ADJUDICATION_PATH).exists()
    assert len(list((tmp_path / executor.PHASE_RECEIPT_DIRECTORY).glob("phase-*.json"))) == 7
    assert not (tmp_path / executor.TERMINAL_SUCCESS_PATH).exists()
    failure = executor.read_json(executor.ACCESS_FAILURE_PATH)
    assert failure["last_completed_phase"] == "adjudication_committed"
    assert failure["failure_operation"] == "runtime_restoration"
    assert failure["error_code"] == "RUNTIME_RESTORATION_FAILURE"
    assert "secret-token" not in executor.canonical_json(failure)
    with pytest.raises(BENDevelopmentExecutorV4Error, match="access-failure receipt exists"):
        check_result()


def test_post_run_adjudicator_recomputes_all_60_decisions_from_retained_evidence() -> None:
    config = load_config()
    loaded = validate_bound_metadata(config)
    registered = build_registered_variants(loaded["synthetic_receipt"]["candidate_registry"])
    item61 = loaded["item61_sparc_evaluation"]
    sparc_ledger = [
        (row["galaxy"], row["rows"])
        for row in sorted(item61["sparc"]["per_object"], key=lambda row: row["galaxy"])
    ]

    def parity() -> dict:
        return {
            "pass": True,
            "max_abs": format(0.0, ".17e"),
            "max_rel": format(0.0, ".17e"),
            "reason": None,
        }

    def sparc_score(loss: float) -> dict:
        return {
            "valid": True,
            "domain_loss": format(loss, ".17e"),
            "per_object": [
                {
                    "object": name,
                    "rows": rows,
                    "loss": format(loss, ".17e"),
                    "terminal_veto": False,
                }
                for name, rows in sparc_ledger
            ],
            "failures": [],
            "object_score_reductions": 139,
            "response_row_score_terms": 2720,
            "cpu_gpu_parity": parity(),
        }

    def xcop_score(loss: float) -> dict:
        return {
            "valid": True,
            "domain_loss": format(loss, ".17e"),
            "valid_object_scores": 8,
            "per_object": [
                {
                    "object": name,
                    "response_rows": 23,
                    "valid": True,
                    "loss": format(loss, ".17e"),
                    "nuisance": {
                        "zeta": format(0.25, ".17e"),
                        "b_P": format(1.0, ".17e"),
                        "b_T": format(1.0, ".17e"),
                    },
                    "H_pressure_response_nontrivial": True,
                    "H_temperature_response_nontrivial": True,
                    "jacobian_rank": 3,
                    "jacobian_condition": format(1.0, ".17e"),
                    "failures": [],
                    "terminal_veto": False,
                    "nuisance_fit_executed": True,
                    "zeta_objective_evaluations": 4097,
                    "analytic_scale_solves": 8194,
                    "response_row_score_terms": 23,
                    "object_score_reductions": 1,
                }
                for name in EXPECTED_XCOP_OBJECTS
            ],
            "failures": [],
            "formula_family_pruned": False,
            "nuisance_fits": 8,
            "zeta_objective_evaluations": 8 * 4097,
            "analytic_scale_solves": 8 * 8194,
            "response_row_score_terms": 184,
            "object_score_reductions": 8,
            "cpu_gpu_parity": parity(),
        }

    sparc = {}
    xcop = {}
    for variant in registered["variants"]:
        loss = 1.0 if variant["kind"] == "full" else 2.0
        sparc[variant["variant_id"]] = sparc_score(loss)
        xcop[variant["variant_id"]] = xcop_score(loss)
    sparc["control:sparc:newtonian_baryons"] = sparc_score(3.0)
    sparc["control:sparc:empirical_rar"] = sparc_score(3.0)
    xcop["control:xcop:gas_only_newtonian_shape"] = xcop_score(3.0)
    xcop["control:xcop:uniform_acceleration_shape_control"] = xcop_score(3.0)
    summaries_s = {
        key: {"valid": value["valid"], "domain_loss": float(value["domain_loss"])}
        for key, value in sparc.items()
    }
    summaries_x = {
        key: {"valid": value["valid"], "domain_loss": float(value["domain_loss"])}
        for key, value in xcop.items()
    }
    decisions = _build_candidate_decisions(registered, summaries_s, summaries_x, config)
    eligible = [row["class_id"] for row in decisions if row["eligible"]]
    static = _static_runtime_receipt(config)
    svd = config["runtime_environment_contract"]["svd_threadpool"]
    cuda = config["runtime_environment_contract"]["cuda"]
    gpu = {
        "pass": True,
        "scientific_payload_access": False,
        "device_count": 1,
        "device_id": 0,
        "device_name": cuda["device_name"],
        "cuda_runtime_version_integer": cuda["runtime_version_integer"],
        "cuda_driver_version_integer": cuda["driver_version_integer"],
        "compute_capability_major": cuda["compute_capability_major"],
        "compute_capability_minor": cuda["compute_capability_minor"],
        "probe_dtype": "float64",
        "probe_values": 16,
        "gpu_runtime_preflight_calls": 1,
    }
    runtime = {
        **static,
        "active_svd_threadpool": {
            "user_api": svd["user_api"],
            "internal_api": svd["internal_api"],
            "prefix": svd["prefix"],
            "version": svd["version"],
            "threading_layer": svd["threading_layer"],
            "architecture": svd["architecture"],
            "library_filename": svd["library_filename"],
            "library_sha256": svd["library_sha256"],
            "num_threads": 1,
        },
        "gpu": gpu,
        "comparison_operator": "binary64_numerical_indifference_band",
        "numerical_indifference_band": config["selection_contract"]["numerical_indifference_band"],
        "reference_environment_validation_pass": True,
    }
    preflight = {"content_sha256": "b" * 64}
    access_intent = {
        "schema_version": executor.ACCESS_SCHEMA,
        "authorization_id": config["authorization_gate"]["authorization_id"],
        "authorization_file_sha256": "a" * 64,
        "config_file_sha256": executor.file_sha256(ROOT / executor.CONFIG_PATH),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "runtime_environment_receipt": runtime,
        "gpu_runtime_preflight": gpu,
        "payload_scope": {
            "sparc_files": 1,
            "sparc_container_parsed_objects": 175,
            "sparc_container_parsed_rows": 3391,
            "sparc_score_objects": 139,
            "sparc_score_rows": 2720,
            "xcop_files": 24,
            "xcop_objects": EXPECTED_XCOP_OBJECTS,
            "xcop_predictor_rows": 521,
            "xcop_response_rows": 184,
        },
        "forbidden_scope": config["development_populations"]["forbidden"],
        "payload_file_open_attempts_before_this_record": 0,
        "payload_file_reads_completed_before_this_record": 0,
        "scores_computed_before_this_record": 0,
        "authorization_replay_allowed": False,
    }
    access_intent["content_sha256"] = content_sha256(access_intent)
    actual = {
        "cpu_formula_domain_batches": 484,
        "gpu_formula_domain_batches": 484,
        "cpu_gpu_parity_comparisons": 484,
        "sparc_formula_row_cells_per_backend": 658240,
        "xcop_formula_row_cells_per_backend": 126082,
        "total_formula_row_cells_per_backend": 784322,
        "total_formula_row_cells_both_backends": 1568644,
        "xcop_coupled_three_parameter_nuisance_fits": 1936,
        "xcop_zeta_objective_evaluations": 7931792,
        "xcop_analytic_scale_solves": 15863584,
        "object_score_reductions": 35574,
        "response_row_score_terms": 702768,
        "candidate_selection_events": 1,
        "gpu_runtime_preflight_calls": 1,
        "gpu_runtime_preflight_probe_values": 16,
        "threshold_tuning_calls": 0,
        "formula_generation_calls": 0,
        "formula_repair_calls": 0,
        "network_calls": 0,
        "model_calls": 0,
        "paid_calls": 0,
        "api_spend_usd": 0.0,
        "payload_file_open_attempts": 25,
        "payload_file_reads_completed": 25,
        "result_validation_events": 1,
    }
    scores = {
        "sparc": sparc,
        "xcop": xcop,
        "candidate_decisions": decisions,
        "eligible_class_ids": eligible,
        "selected_class_id": eligible[0],
        "selection_events": 1,
        "counterexample_policy": {
            "single_counterexample_terminal": False,
            "counterexample_count_alone_terminal": False,
            "formula_families_pruned": 0,
            "all_scoped_failures_retained": True,
        },
        "actual_compute_accounting": _render_result_floats(actual),
    }
    result = {
        "schema_version": executor.RESULT_SCHEMA,
        "status": "development_only_scoring_complete",
        "authorization_id": config["authorization_gate"]["authorization_id"],
        "authorization_file_sha256": "a" * 64,
        "access_intent_content_sha256": access_intent["content_sha256"],
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "candidate_registry_content_sha256": config["candidate_registry"][
            "registry_content_sha256"
        ],
        "registered_formula_ledger": [
            {key: value for key, value in row.items() if key != "canonical_ast"}
            for row in registered["variants"]
        ],
        "candidate_and_ablation_accounting": registered["accounting"],
        "development_populations": {
            "sparc": {
                "payload_file_open_attempts": 1,
                "payload_file_reads_completed": 1,
                "parsed_container_objects": 175,
                "parsed_container_rows": 3391,
                "scored_objects": 139,
                "scored_rows": 2720,
                "split_confirmation_objects": 35,
                "split_exploration_before_admission_objects": 140,
                "admitted_score_name_root_sha256": config["development_populations"]["sparc"][
                    "admitted_score_name_root_sha256"
                ],
                "admitted_score_name_row_ledger_sha256": config["development_populations"]["sparc"][
                    "admitted_score_name_row_ledger_sha256"
                ],
            },
            "xcop": {
                "payload_file_open_attempts": 24,
                "payload_file_reads_completed": 24,
                "objects_parsed": 8,
                "predictor_rows_parsed_and_scored": 521,
                "response_rows_parsed_and_scored": 184,
            },
        },
        "runtime_environment_receipt": runtime,
        "gpu_runtime_preflight": gpu,
        "scores": scores,
        "claim_ceiling": config["claim_ceiling"],
        "claims": {
            "development_only_score": True,
            "fresh_confirmation": False,
            "absolute_cluster_prediction": False,
            "full_covariance": False,
            "historical_novelty_established": False,
            "dark_matter_eliminated": False,
            "alternative_to_gr_established": False,
            "publication_ready": False,
            "formula_family_pruned": False,
            "raw_per_object_losses_independently_recomputed": False,
        },
    }
    result["content_sha256"] = content_sha256(result)
    evidence = validate_result_document(result, config, registered, preflight, access_intent)
    assert evidence["candidate_decisions_recomputed"] == 60
    assert evidence["selected_class_id"] == eligible[0]
    mutated = copy.deepcopy(result)
    mutated["scores"]["selected_class_id"] = eligible[-1]
    mutated["content_sha256"] = content_sha256(mutated)
    with pytest.raises(BENDevelopmentExecutorV4Error, match="selection adjudication"):
        validate_result_document(mutated, config, registered, preflight, access_intent)


def test_no_production_outputs_exist_during_preparation() -> None:
    assert not (ROOT / ACCESS_INTENT_PATH).exists()
    assert not (ROOT / RESULT_PATH).exists()
    assert not (ROOT / executor.ADJUDICATION_PATH).exists()
    assert not (ROOT / executor.ACCESS_FAILURE_PATH).exists()
    assert not (ROOT / executor.PHASE_RECEIPT_DIRECTORY).exists()
    assert not (ROOT / executor.TERMINAL_SUCCESS_PATH).exists()
    assert not (ROOT / executor.TERMINAL_STATE_LOCK_PATH).exists()
    # The expected preparation products are separate from production output.
    assert PREFLIGHT_PATH != RESULT_PATH
    assert AUTHORIZATION_PATH != ACCESS_INTENT_PATH
