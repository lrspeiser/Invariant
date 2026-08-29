import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    GravityItem39Error,
    _contract_digest,
    admissible_candidates,
    boundary_coordinates,
    build_exposure_manifest,
    build_sample_manifest,
    decode_candidate,
    fixed_control_multiplier,
    generate_raw_candidates,
    load_config,
    metric_observables,
    predict_multiplier,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item39_config_binds_counterexample_policy_and_zero_response_access() -> None:
    config = load_config(ROOT)
    assert config["item"] == 39
    assert config["scope"]["exploration_response_opened"] is False
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["scope"]["paid_api_calls_authorized"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["discovery_policy"]["counterexample_count_alone_is_never_decisive"]
    assert not config["discovery_policy"]["finite_empirical_sample_may_prune_family"]


def test_item39_contract_digest_ignores_only_future_bindings() -> None:
    config = load_config(ROOT)
    changed = deepcopy(config)
    changed["scientific_freeze_commit"] = "a" * 40
    changed["predictor_freeze_commit"] = "b" * 40
    changed["sample_freeze_commit"] = "c" * 40
    assert _contract_digest(changed) == _contract_digest(config)
    changed["candidate_generator"]["raw_candidate_cells"] += 1
    assert _contract_digest(changed) != _contract_digest(config)


def test_item39_raw_grammar_has_exact_equal_capacity() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262144
    assert np.array_equal(raw["candidate_id"], np.arange(262144))
    assert {int(lane): int(np.sum(raw["lane"] == lane)) for lane in range(4)} == {
        0: 65536,
        1: 65536,
        2: 65536,
        3: 65536,
    }


def test_item39_boundary_coordinates_are_bounded_and_distinct() -> None:
    fraction = np.asarray([0.1, 0.5, 0.9])
    radius = np.asarray([0.2, 0.6, 1.0])
    slope = np.asarray([0.5, 2.0, 3.0])
    values = boundary_coordinates(fraction, radius, slope)
    assert values.shape == (4, 3)
    assert np.all(np.isfinite(values))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert len({tuple(np.round(row, 8)) for row in values}) == 4


def test_item39_candidate_equations_use_boundary_data_and_are_target_blind() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    ids = np.asarray([0, 65536, 131072, 196608])
    rows = {key: value[ids] for key, value in raw.items()}
    u = np.asarray([1e-4, 1e-2, 1.0])
    a = predict_multiplier(
        rows,
        u,
        np.asarray([0.1, 0.5, 0.9]),
        np.asarray([0.2, 0.6, 1.0]),
        np.asarray([0.5, 2.0, 3.0]),
        config,
    )
    b = predict_multiplier(
        rows,
        u,
        np.asarray([0.8, 0.2, 0.6]),
        np.asarray([0.9, 0.3, 0.7]),
        np.asarray([2.5, 0.4, 1.1]),
        config,
    )
    assert np.all(np.isfinite(a)) and np.all(a >= 1.0)
    assert np.all(np.any(np.abs(a - b) > 1e-12, axis=1))
    forbidden = json.dumps(config["predictor_derivation"]["forbidden"]).casefold()
    assert "vrot" in forbidden
    assert config["candidate_generator"]["post_response_cells"] == 0


def test_item39_admission_is_reproducible_and_keeps_every_niche() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config, batch_size=16384)
    assert 0 < audit["admitted_candidates"] <= 262144
    assert sum(audit["admitted_by_lane"].values()) == audit["admitted_candidates"]
    assert all(audit["admitted_by_lane"][str(lane)] > 0 for lane in range(4))
    replay, replay_audit = admissible_candidates(config, batch_size=32768)
    assert np.array_equal(admitted["candidate_id"], replay["candidate_id"])
    assert audit["behavioral_equivalence_classes"] == replay_audit["behavioral_equivalence_classes"]


def test_item39_metric_contract_uses_same_field_for_motion_and_light() -> None:
    gbar = np.asarray([1.0, 2.0, 3.0])
    multiplier = np.asarray([1.1, 2.0, 4.0])
    observables = metric_observables(gbar, multiplier)
    assert np.array_equal(observables["g_dynamics"], observables["grad_phi"])
    assert np.array_equal(observables["grad_phi"], observables["grad_psi"])
    assert np.array_equal(
        observables["lensing_integrand_grad_phi_plus_psi"],
        2.0 * observables["g_dynamics"],
    )


def test_item39_controls_reproduce_frozen_item38_formula() -> None:
    u = np.asarray([1e-6, 0.01, 1.0, 1e8])
    baryonic = fixed_control_multiplier("baryonic_newton", u)
    mond = fixed_control_multiplier("mond_RAR", u)
    item38 = fixed_control_multiplier("item38_selected", u)
    expected = 1.0 + 2.75 * u**-0.45 * (1.0 + (u / 0.01) ** 0.5) ** -2.0
    assert np.array_equal(baryonic, np.ones_like(u))
    assert np.all(mond >= 1.0)
    assert np.allclose(item38, expected)


def test_item39_decoder_discloses_known_and_synthesis_labels() -> None:
    config = load_config(ROOT)
    known = decode_candidate(0, config)
    synthesis = decode_candidate(2 * 65536, config)
    assert known["lane"] == "surface_bulk_equipartition_mismatch"
    assert known["creativity_label"].startswith("known_")
    assert synthesis["creativity_label"].startswith("potentially_new_observational_synthesis")


def test_item39_exposure_manifest_excludes_exploration_and_confirmation() -> None:
    config = load_config(ROOT)
    manifest = build_exposure_manifest(ROOT, config)
    assert manifest["role_counts"]["exploration"] > 0
    assert manifest["role_counts"]["reserved_confirmation"] > 0
    assert len(manifest["excluded_names"]) > 0
    assert manifest["response_values_read_while_building"] == 0


def test_item39_target_blind_sample_is_balanced_and_keeps_confirmation_sealed() -> None:
    sample = build_sample_manifest(ROOT)
    assert sample["counts"]["exploration"] == 60
    assert sample["counts"]["reserved_confirmation"] == 15
    assert sample["counts"]["response_rows_read"] == 0
    assert sample["counts"]["confirmation_rows_read"] == 0
    assert len(sample["cells"]) == 8
    assert all(
        counts["exploration"] > 0 and counts["reserved_confirmation"] > 0
        for counts in sample["cells"].values()
    )
    assert all(row["response_read"] is False for row in sample["objects"])


def test_item39_rejects_tampered_counterexample_metric_and_confirmation_boundaries() -> None:
    path = ROOT / "configs" / "gravity_item39_holographic_boundary_v1.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["discovery_policy"]["finite_empirical_sample_may_prune_family"] = True
    config["weak_field_metric_contract"]["gravitational_slip"] = "free"
    config["scope"]["confirmation_opening_authorized"] = True
    with pytest.raises(GravityItem39Error):
        validate_config(ROOT, config)
